# SPDX-License-Identifier: Apache-2.0
"""Hy4-Flash's decode attention half of a sliding layer in one NKI kernel: from the residual streams to this rank's
partial of the attention output, the iHC post weights and the rows' MLA rows (attention.Attention.decode, sliding).

Per layer and rank, for T <= 128 decode rows and the rank's ``h`` heads (2 at TP32):
  iHC mix (common.hc_mix), input_layernorm
  -> q_a, q_a_layernorm, q_b; kv_a, kv_a_layernorm on the latent; interleaved RoPE of the query's and the key's last
     64 channels (x cos + (x @ Pm) sin, Pm the pair swap with the sign: no strided access)
  -> the absorbed query (q_nope @ w_uk) and the sigmoid output gate
  -> per row: its window's keys (gathered by the graph a page at a time; the kernel's own per-partition gathers were
     software-DGE DMAs, a notification each, and overflowed the runtime's notification queue at 64 rows,
     NRT_EXEC_SW_NQ_OVERFLOW), the scores of each head on
     the vector engine (keys on the partitions, the query broadcast to them), the window mask (positions after the
     window's start and before the row's own), softmax with the head's sink and the row's own key (its MLA row from
     SBUF: the pool's copy is written after the kernel), the attended latent by a matmul of the probabilities
  -> w_uv, the gate, o_proj: this rank's partial [T, H] for the all-reduce
Outputs: the partial [T, H] bf16, post [T, hc] fp32, the rows' MLA rows [T, 576] bf16 (the caller writes them to
the pool). The two programs of a logical core split the rows of the attention and the columns of o_proj; both run the
projections before (common.hc_mix and the q / kv path).

Inputs (this rank's shapes at TP32: H 3072, hc 4, q_lora 1024, h 2, nope 192, rope 64, v 256, latent 512):
  streams [T, hc, H] bf16; fn_hi / fn_lo [hc H, 2 hc] bf16; hc_base [1, 2 hc]; hc_scale [1, 2] fp32; ln_w [1, H]
  q_a [H, q_lora]; qa_ln [1, q_lora]; q_b [q_lora, h x (nope + rope)]; kv_a [H, latent + rope]; kv_ln [1, latent]
  w_uk [h, nope, latent]; w_uvT [h, latent, v]; gate_w [H, h x v]; o_w [h x v, H] (bf16); sinks [1, h] fp32
  cos / sin [T, rope] fp32, each frequency on both channels of its pair; pm [rope, rope] bf16
  keys [T, blocks x 128, latent + rope] (bf16 or FP8): each row's window pages of the pool gathered (the K half then
  the V half of each slot's MLA row), the graph's page gather; first [T, 1] fp32, the position of each one's first
  slot;
  pos [T, 1] fp32, each row's position"""

import nki
import nki.isa as nisa
import nki.language as nl

from sglang_neuron.models.hy_v4_flash.kernels import common

P = 128
NEG = -1.0e30


def _rope(x, T: int, n: int, cos, sin, pmT):
    """[128, n] fp32 of the interleaved RoPE of ``x`` [128, n] bf16 (n = 64 a head, several heads side by side):
    x cos + (x @ Pm) sin."""
    xT = common.transpose(x, T, n, nl.bfloat16)
    sw = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
    R = 64
    for c in range(n // R):
        ps = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.psum)
        t, off = (c * R) // P, (c * R) % P
        nisa.nc_matmul(dst=ps[0:T, 0:R], stationary=xT[off:off + R, t, 0:T], moving=pmT[0:R, 0:R])
        nisa.tensor_copy(dst=sw[0:T, c * R:(c + 1) * R], src=ps[0:T, 0:R])
    out = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
    for c in range(n // R):
        a = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(a[0:T, 0:R], x[0:T, c * R:(c + 1) * R], cos[0:T, 0:R], nl.multiply)
        b = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(b[0:T, 0:R], sw[0:T, c * R:(c + 1) * R], sin[0:T, 0:R], nl.multiply)
        nisa.tensor_tensor(out[0:T, c * R:(c + 1) * R], a[0:T, 0:R], b[0:T, 0:R], nl.add)
    return out


def _heads(xT, qrT, q_b, kv_a, kv_ln, w_uk, gate_w, cos, sin, pm, T: int, eps: float, scale: float, name: str):
    """(the rows' queries [128, h x 576] bf16 ([latent | rope] a head), their MLA rows [128, 576] bf16, their own
    scores [128, h] fp32, the sigmoid gate [128, h x v] fp32) of the normed input ``xT`` and q_resid ``qrT``
    (transposed); the MLA rows stored to ``name`` [T, 576] in shared HBM."""
    h, NOPE, LAT = w_uk.shape
    QD = q_b.shape[1] // h
    ROPE = QD - NOPE
    KD = LAT + ROPE
    V = gate_w.shape[1] // h
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    q = common.project(qrT, T, q_b, nl.bfloat16)  # [T, h x QD]
    cs = nl.ndarray((P, ROPE), dtype=nl.float32, buffer=nl.sbuf)
    sn = nl.ndarray((P, ROPE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=cs[0:T, 0:ROPE], src=cos)
    nisa.dma_copy(dst=sn[0:T, 0:ROPE], src=sin)
    pmT = nl.ndarray((P, ROPE), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=pmT[0:ROPE, 0:ROPE], src=pm)
    # the row's queries, a head after the other: [latent | rope] in bf16, as the graph's einsums take them
    qcat = nl.ndarray((P, h * KD), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i in range(h):
        qrope_in = nl.ndarray((P, ROPE), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=qrope_in[0:T, 0:ROPE], src=q[0:T, i * QD + NOPE:(i + 1) * QD])
        qrope = _rope(qrope_in, T, ROPE, cs, sn, pmT)
        nisa.tensor_copy(dst=qcat[0:T, i * KD + LAT:(i + 1) * KD], src=qrope[0:T, 0:ROPE])
        qn = nl.ndarray((P, NOPE), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=qn[0:T, 0:NOPE], src=q[0:T, i * QD:i * QD + NOPE])
        qnT = common.transpose(qn, T, NOPE, nl.bfloat16)
        ql = common.project(qnT, T, w_uk.reshape((h * NOPE, LAT)), nl.bfloat16, row0=i * NOPE, rows=NOPE)
        nisa.tensor_copy(dst=qcat[0:T, i * KD:i * KD + LAT], src=ql[0:T, 0:LAT])

    # ---- the rows' MLA rows: kv_a, the latent's norm, the key's RoPE ----
    ckv = common.project(xT, T, kv_a, nl.bfloat16)  # [T, KD]
    lat_in = nl.ndarray((P, LAT), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=lat_in[0:T, 0:LAT], src=ckv[0:T, 0:LAT])
    kl = common.rms_norm(lat_in, T, LAT, kv_ln, eps)
    kr_in = nl.ndarray((P, ROPE), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=kr_in[0:T, 0:ROPE], src=ckv[0:T, LAT:KD])
    kr = _rope(kr_in, T, ROPE, cs, sn, pmT)
    own = nl.ndarray((P, KD), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=own[0:T, 0:LAT], src=kl[0:T, 0:LAT])
    nisa.tensor_copy(dst=own[0:T, LAT:KD], src=kr[0:T, 0:ROPE])
    # both programs hold the same rows: each stores its share of the columns
    rows_out = nl.ndarray((T, KD), dtype=nl.bfloat16, buffer=nl.shared_hbm, name=name)
    KH = KD // n_prg
    nisa.dma_copy(dst=rows_out.ap(pattern=[[KD, T], [1, KH]], offset=pid * KH),
                  src=own[0:T, pid * KH:(pid + 1) * KH])
    # each row's own score of each head: q . its own MLA row (the einsums' two parts each rounded to bf16)
    s_own = nl.ndarray((P, h), dtype=nl.float32, buffer=nl.sbuf)
    for i in range(h):
        prod = nl.ndarray((P, KD), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(prod[0:T, 0:KD], qcat[0:T, i * KD:(i + 1) * KD], own[0:T, 0:KD], nl.multiply)
        nisa.tensor_reduce(s_own[0:T, i:i + 1], op=nl.add, data=prod[0:T, 0:KD], axis=1, keepdims=True)
    nisa.tensor_scalar(dst=s_own[0:T, 0:h], data=s_own[0:T, 0:h], op0=nl.multiply, operand0=scale)

    # ---- the output gate ----
    gl = common.project(xT, T, gate_w, nl.bfloat16)
    gs = nl.ndarray((P, h * V), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=gs[0:T, 0:h * V], op=nl.sigmoid, data=gl[0:T, 0:h * V])

    return qcat, own, s_own, gs, rows_out


def _attend(qcat, own, s_own, sinks, keys_hbm, bias_all, T: int, NB: int, scale: float, name: str, rope: int = 64):
    """[T, h x latent] bf16 in shared HBM (``name``) of each row's attention over its keys ``keys_hbm`` [T, NB x 128,
    576] (gathered before the kernel, loaded by a plain DMA a row), tile j of row r with the additive mask
    ``bias_all`` (0 or NEG) column r x NB + j; the row's own key from ``own`` and the head's sink in the softmax. The
    two programs split the rows."""
    h = sinks.shape[1]
    KD = own.shape[1]
    LAT = KD - rope
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    sk = nl.ndarray((P, h), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sk[0:1, 0:h], src=sinks)
    row_ids = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(row_ids[0:P, 0:P], pattern=[[0, P]], offset=0, channel_multiplier=1)
    zeros = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(zeros[0:P, 0:P], 0.0)
    ones = nl.ndarray((P, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(ones[0:P, 0:1], 1.0)
    att = nl.ndarray((T, h * LAT), dtype=nl.bfloat16, buffer=nl.shared_hbm, name=name)
    for r in range(pid, T, n_prg):
        cols = [r * NB + j for j in range(NB)]
        # one-hot of row r over the T rows, its columns all alike: a matmul against it copies row r anywhere
        sel32 = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=sel32[0:T, 0:P], data=row_ids[0:T, 0:P], op0=nl.equal, operand0=float(r))
        sel = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=sel[0:T, 0:P], src=sel32[0:T, 0:P])
        kraw = nl.ndarray((P, NB, KD), dtype=keys_hbm.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=kraw[0:P, 0:NB, 0:KD],
                      src=keys_hbm.ap(pattern=[[KD, P], [P * KD, NB], [1, KD]], offset=r * NB * P * KD),
                      dge_mode=nisa.dge_mode.hwdge)
        keys = nl.ndarray((P, NB, KD), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=keys[0:P, 0:NB, 0:KD], src=kraw[0:P, 0:NB, 0:KD])
        # row r's own scores and MLA row onto partition 0
        own_ps = nl.ndarray((P, LAT), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=own_ps[0:1, 0:LAT], stationary=sel[0:T, 0:1], moving=own[0:T, 0:LAT])
        own_lat = nl.ndarray((P, LAT), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=own_lat[0:1, 0:LAT], src=own_ps[0:1, 0:LAT])
        # the own scores in fp32 (a bf16 move would round the logits)
        so_ps = nl.ndarray((P, h), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=so_ps[0:1, 0:h], stationary=sel32[0:T, 0:1], moving=s_own[0:T, 0:h])
        for i in range(h):
            # the head's query on every partition
            qb = nl.ndarray((P, KD), dtype=nl.bfloat16, buffer=nl.sbuf)
            for c0 in range(0, KD, 512):
                cw = min(512, KD - c0)
                qps = nl.ndarray((P, 512), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=qps[0:P, 0:cw], stationary=sel[0:T, 0:P],
                               moving=qcat[0:T, i * KD + c0:i * KD + c0 + cw])
                nisa.tensor_copy(dst=qb[0:P, c0:c0 + cw], src=qps[0:P, 0:cw])
            sc = nl.ndarray((P, NB), dtype=nl.float32, buffer=nl.sbuf)
            for j in range(NB):
                prod = nl.ndarray((P, KD), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(prod[0:P, 0:KD], keys[0:P, j, 0:KD], qb[0:P, 0:KD], nl.multiply)
                nisa.tensor_reduce(sc[0:P, j:j + 1], op=nl.add, data=prod[0:P, 0:KD], axis=1, keepdims=True)
            for j in range(NB):
                nisa.scalar_tensor_tensor(dst=sc[0:P, j:j + 1], data=sc[0:P, j:j + 1], op0=nl.multiply,
                                          operand0=scale, op1=nl.add, operand1=bias_all[0:P, cols[j]:cols[j] + 1])
            # the max over the window, the sink and the own score, broadcast back to the partitions
            cmax = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(cmax[0:P, 0:1], op=nl.maximum, data=sc[0:P, 0:NB], axis=1, keepdims=True)
            cmT = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=cmT[0:1, 0:P], data=cmax[0:P, 0:1])
            m0 = nl.ndarray((P, 4), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(m0[0:1, 0:1], op=nl.maximum, data=cmT[0:1, 0:P], axis=1, keepdims=True)
            nisa.tensor_tensor(m0[0:1, 0:1], m0[0:1, 0:1], sk[0:1, i:i + 1], nl.maximum)
            nisa.tensor_copy(dst=m0[0:1, 1:2], src=so_ps[0:1, i:i + 1])
            nisa.tensor_tensor(m0[0:1, 0:1], m0[0:1, 0:1], m0[0:1, 1:2], nl.maximum)
            mb_ps = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.psum)
            mrow = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=mrow[0:1, 0:P], data=zeros[0:1, 0:P], op0=nl.add, operand0=m0[0:1, 0:1])
            nisa.nc_transpose(dst=mb_ps[0:P, 0:1], data=mrow[0:1, 0:P])
            negm = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=negm[0:P, 0:1], data=mb_ps[0:P, 0:1], op0=nl.multiply, operand0=-1.0)
            pr = nl.ndarray((P, NB), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=pr[0:P, 0:NB], op=nl.exp, data=sc[0:P, 0:NB], bias=negm[0:P, 0:1])
            prb = nl.ndarray((P, NB), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=prb[0:P, 0:NB], src=pr[0:P, 0:NB])
            # the attended latent and the denominator over the window
            av = nl.ndarray((P, LAT), dtype=nl.float32, buffer=nl.psum)
            dn = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.psum)
            for j in range(NB):
                nisa.nc_matmul(dst=av[0:1, 0:LAT], stationary=prb[0:P, j:j + 1], moving=keys[0:P, j, 0:LAT])
                nisa.nc_matmul(dst=dn[0:1, 0:1], stationary=prb[0:P, j:j + 1], moving=ones[0:P, 0:1])
            # the own key and the sink: exp(s - m) each, on partition 0
            ex = nl.ndarray((P, 2), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=ex[0:1, 0:1], src=m0[0:1, 1:2])
            nisa.tensor_copy(dst=ex[0:1, 1:2], src=sk[0:1, i:i + 1])
            nisa.activation(dst=ex[0:1, 0:2], op=nl.exp, data=ex[0:1, 0:2], bias=negm[0:1, 0:1])
            den = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(den[0:1, 0:1], dn[0:1, 0:1], ex[0:1, 0:1], nl.add)
            nisa.tensor_tensor(den[0:1, 0:1], den[0:1, 0:1], ex[0:1, 1:2], nl.add)
            rden = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(dst=rden[0:1, 0:1], data=den[0:1, 0:1])
            # own_bf: the probability of the own key rounded to bf16 as the window's
            exb = nl.ndarray((P, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=exb[0:1, 0:1], src=ex[0:1, 0:1])
            o_row = nl.ndarray((P, LAT), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=o_row[0:1, 0:LAT], data=own_lat[0:1, 0:LAT], op0=nl.multiply,
                                      operand0=exb[0:1, 0:1], op1=nl.add, operand1=av[0:1, 0:LAT])
            o_b = nl.ndarray((P, LAT), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=o_b[0:1, 0:LAT], data=o_row[0:1, 0:LAT], op0=nl.multiply, operand0=rden[0:1, 0:1])
            nisa.dma_copy(dst=att.ap(pattern=[[h * LAT, 1], [1, LAT]], offset=r * h * LAT + i * LAT),
                          src=o_b[0:1, 0:LAT])
    if n_prg > 1:
        nisa.core_barrier(data=att, cores=(0, 1))

    return att


def _output(att, gs, w_uvT, o_w, T: int):
    """[128, H / programs] bf16 (rows 0 .. T - 1) of this program's columns of o_proj(w_uv(att) x gate)."""
    h, LAT, V = w_uvT.shape
    H = o_w.shape[1]
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    HH = H // n_prg
    h0 = pid * HH
    at = nl.ndarray((P, h * LAT), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=at[0:T, 0:h * LAT], src=att)
    vals = nl.ndarray((P, h * V), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i in range(h):
        ai = nl.ndarray((P, LAT), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ai[0:T, 0:LAT], src=at[0:T, i * LAT:(i + 1) * LAT])
        aT = common.transpose(ai, T, LAT, nl.bfloat16)
        vi = common.project(aT, T, w_uvT.reshape((h * LAT, V)), nl.bfloat16, row0=i * LAT, rows=LAT)
        vg = nl.ndarray((P, V), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(vg[0:T, 0:V], vi[0:T, 0:V], gs[0:T, i * V:(i + 1) * V], nl.multiply)
        nisa.tensor_copy(dst=vals[0:T, i * V:(i + 1) * V], src=vg[0:T, 0:V])
    vT = common.transpose(vals, T, h * V, nl.bfloat16)
    part = common.project(vT, T, o_w, nl.bfloat16, col0=h0, cols=HH)
    return part


@nki.jit
def attn_swa(streams, fn_hi, fn_lo, hc_base, hc_scale, ln_w, q_a, qa_ln, q_b, kv_a, kv_ln, w_uk, w_uvT, gate_w, o_w,
             sinks, cos, sin, pm, keys, first, pos, eps=1e-5, hc_eps=1e-6, hc_mag=2.0, scale=1.0, window=512):
    T, HC, H = streams.shape
    QL = q_a.shape[1]
    NB = keys.shape[1] // P
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    HH = H // n_prg
    h0 = pid * HH

    # ---- iHC mix, input_layernorm, q_a and its norm ----
    _, post, xb = common.hc_mix(streams, fn_hi, fn_lo, hc_base, hc_scale, T, eps, hc_eps, hc_mag)
    xn = common.rms_norm(xb, T, H, ln_w, eps)
    xT = common.transpose(xn, T, H, nl.bfloat16)
    qa = common.project(xT, T, q_a, nl.bfloat16)
    qr = common.rms_norm(qa, T, QL, qa_ln, eps)
    qrT = common.transpose(qr, T, QL, nl.bfloat16)
    qcat, own, s_own, gs, rows_out = _heads(xT, qrT, q_b, kv_a, kv_ln, w_uk, gate_w, cos, sin, pm, T, eps, scale,
                                            "attn_swa_rows")

    # ---- per row: the window's attention ----
    kp0 = nl.ndarray((P, NB), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(kp0[0:P, 0:NB], pattern=[[P, NB]], offset=0, channel_multiplier=1)
    # every row's window bias, before the loop: [128, T, NB]
    fa = nl.ndarray((P, T), dtype=nl.float32, buffer=nl.sbuf)
    pa = nl.ndarray((P, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=fa[0:P, 0:T], src=first.reshape((1, T)).ap(pattern=[[0, P], [1, T]], offset=0))
    nisa.dma_copy(dst=pa[0:P, 0:T], src=pos.reshape((1, T)).ap(pattern=[[0, P], [1, T]], offset=0))
    bias_all = nl.ndarray((P, T * NB), dtype=nl.float32, buffer=nl.sbuf)
    for r in range(T):
        kp = nl.ndarray((P, NB), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=kp[0:P, 0:NB], data=kp0[0:P, 0:NB], op0=nl.add, operand0=fa[0:P, r:r + 1])
        lo = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=lo[0:P, 0:1], data=pa[0:P, r:r + 1], op0=nl.subtract, operand0=float(window))
        m1 = nl.ndarray((P, NB), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=m1[0:P, 0:NB], data=kp[0:P, 0:NB], op0=nl.less, operand0=pa[0:P, r:r + 1])
        m2 = nl.ndarray((P, NB), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=m2[0:P, 0:NB], data=kp[0:P, 0:NB], op0=nl.greater, operand0=lo[0:P, 0:1])
        nisa.tensor_tensor(bias_all[0:P, r * NB:(r + 1) * NB], m1[0:P, 0:NB], m2[0:P, 0:NB], nl.multiply)
    # allowed 1 -> 0, masked 0 -> NEG
    nisa.tensor_scalar(dst=bias_all[0:P, 0:T * NB], data=bias_all[0:P, 0:T * NB], op0=nl.subtract, operand0=1.0,
                       op1=nl.multiply, operand1=-NEG)
    att = _attend(qcat, own, s_own, sinks, keys, bias_all, T, NB, scale, "attn_swa_attended")
    part = _output(att, gs, w_uvT, o_w, T)
    out = nl.ndarray((T, H), dtype=streams.dtype, buffer=nl.shared_hbm, name="attn_swa_out")
    nisa.dma_copy(dst=out.ap(pattern=[[H, T], [1, HH]], offset=h0), src=part[0:T, 0:HH])
    post_out = common.store_post(post, T, HC, "attn_swa_post")
    return out, post_out, rows_out

