"""Qwen3.5-397B PREFILL stack for one rank of a dedicated prefill node (NKI, LNC=1).

The three hardware-validated prefill mixers (``gdn_prefill_tkg``, ``attn_prefill_tkg``, the MoE pieces of
``moe_prefill``) composed into the real layer pattern - every 4th layer gated attention, the rest Gated DeltaNet,
MoE in every layer - with the collectives that move tokens between their layouts, and the P -> D hand-off in
decode's layouts.

Token layout (one NEFF for all ranks; ``tp`` = GDN TP size = the PROMPT GROUP size):

* The node is split into ``world / tp`` prompt groups of ``tp`` consecutive ranks. A group owns ``bpg`` prompts of
  ``L`` tokens; each prompt is cut into ``n_chunks`` token chunks of ``Lc = L / n_chunks`` tokens (micro-batches).
* Residual stream: token-sharded inside the group. Rank ``g`` (position in its group) holds ``T_pc = Lc / tp``
  rows of every chunk: ``x_shard[(b * n_chunks + c) * T_pc + j] = prompt b, token c * Lc + g * T_pc + j``.
* Per layer and micro-batch m = (b, c):
      xn_m  = RMSNorm_zc(x_m) * (1 + w_in)                              local, T_pc rows (norm BEFORE the gather)
      xg_m  = AllGather[group](xn_m)                                     [Lc, H], sequence order
      GDN:    out = gdn_prefill_tkg(xg_m, init = chunk c-1's states)     this rank's head shard, [Lc, H] partial
              r_m = x_m + ReduceScatter[group](out)
      attn:   out = attn_prefill_tkg(xg_m, start = c * Lc)               attention TP ``atp`` subgroups of the
              r_m = x_m + ReduceScatter[atp](out[slice])                 group, each REDUNDANTLY over the chunk;
                                                                         each subgroup reduces only the rows its
                                                                         members hold (runtime ``attn_sub``)
      MoE:    x_m = r_m + MoE(r_m)                                       EP replicas of ``ep`` consecutive ranks:
                                                                         route -> AllGather[EP] -> shared expert ->
                                                                         per-expert lists + experts ->
                                                                         ReduceScatter[EP] (moe_prefill pieces)
  Program order per layer (software pipeline over micro-batches): every micro-batch's norm + AllGather first, then
  the mixers (each ReduceScatter overlaps the next micro-batch's mixer), then the MoE fronts, then the MoE backs.

Outputs (the P -> D hand-off):
  out        [bpg * n_chunks * T_pc, H]  final residual rows (input layout); h_last [bpg, H] = this rank's last row
             of each prompt: on the group's LAST rank this is the prompt's last hidden state (first decode token)
  conv_all   [n_gdn * bpg, Cc, 3] bf16    GDN layer gi, prompt b at row gi * bpg + b: decode's conv_state rows
  rec_all    [n_gdn * bpg, 128, vh, 128]  fp32, decode's token-major rec_state rows (gdn_decode_tkg rec_row0)
  k_cache_t  [kvl * bpg, 256, S] bf16     attention layer j, prompt b in slot (j % kvl) * bpg + b: decode's
  v_cache    [kvl * bpg, S, 256] bf16     slot-contiguous caches, positions 0 .. L-1 written
The GDN / attention shards are decode's (same head -> rank map at the same TP degree), so a decode rank reads these
rows as its own state / cache slots; positions >= L of the caches are not written.

Timing scaffold (like ``qwen35_hybrid_stack``): layers of a kind share their weight tensors, but every layer issues
all of its own DMAs and collectives (the nested kernels load their weights per call).
"""

import os

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from nkilib.core.utils.kernel_assert import kernel_assert
from .attn_decode import attn_decode_tkg
from .attn_prefill import attn_prefill_tkg
from .gdn_decode import gdn_decode_tkg
from .gdn_prefill import gdn_prefill_tkg
from .moe_front import rmsnorm_zc_tok
from .moe_prefill import NMETA, _finish, _zero_hbm, moe_prefill_routed, prefill_route, prefill_shared

P = 128
HEAD = 128  # GDN head dim
D_ATT = 256  # attention head dim
KWIN = 4
_KMAJOR = os.environ.get("QWEN35_GDN_KMAJOR", "0") == "1"


# ----------------------------------------------------------------------------- small helpers


def _copy_rows(dst, d0, src, s0, n, W, dtype=nl.bfloat16):
    """dst[d0 : d0 + n] = src[s0 : s0 + n] (HBM, [rows, W]) through SBUF, 128-row tiles."""
    for r0 in range(0, n, P):
        m = min(P, n - r0)
        t = nl.ndarray((P, W), dtype=dtype, buffer=nl.sbuf)[:m]
        nisa.dma_copy(dst=t, src=src[s0 + r0 : s0 + r0 + m])
        nisa.dma_copy(dst=dst[d0 + r0 : d0 + r0 + m], src=t)


def _copy_slice_rt(dst, src, R, H, sub_sb):
    """dst [R, H] = rows sub * R .. sub * R + R - 1 of src [rows, H], sub a runtime SBUF [1, 1] int32."""
    nblk = src.shape[0] // R
    sv = src.reshape((nblk, R * H))
    for r0 in range(0, R, P):
        m = min(P, R - r0)
        t = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)[:m]
        nisa.dma_copy(dst=t, src=sv.ap(pattern=[[H, m], [1, H]], offset=r0 * H, scalar_offset=sub_sb, indirect_dim=0))
        nisa.dma_copy(dst=dst[r0 : r0 + m], src=t)


def _norm_rows(x, ln_w, n, H, xn):
    """xn [n, H] bf16 = RMSNorm_zc(x) * (1 + ln_w) (decode's rounding points), 128-row blocks."""
    w_row = nl.ndarray((P, H), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=w_row, src=ln_w.ap(pattern=[[0, P], [1, H]], offset=0))
    nisa.tensor_scalar(dst=w_row, data=w_row, op0=nl.add, operand0=1.0)
    for r0 in range(0, n, P):
        m = min(P, n - r0)
        xs = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)[:m]
        nisa.dma_copy(dst=xs, src=x[r0 : r0 + m])
        o = rmsnorm_zc_tok(xs, w_row[:m], m, H, 1e-6, nl.bfloat16)
        nisa.dma_copy(dst=xn[r0 : r0 + m], src=o)


def _add_rows(dst, a, b, n, H):
    """dst = bf16(a + b) for [n, H] HBM rows (decode's residual add)."""
    for r0 in range(0, n, P):
        m = min(P, n - r0)
        ta = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)[:m]
        nisa.dma_copy(dst=ta, src=a[r0 : r0 + m])
        tb = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)[:m]
        nisa.dma_copy(dst=tb, src=b[r0 : r0 + m])
        s = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)[:m]
        nisa.tensor_tensor(dst=s, data1=ta, data2=tb, op=nl.add)
        nisa.dma_copy(dst=dst[r0 : r0 + m], src=s)


def _const_i32(val, name):
    """HBM [1, 1] int32 holding a trace-time constant (runtime slot index of the attention kernel)."""
    t = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(t, pattern=[[0, 1]], offset=val, channel_multiplier=0)
    h = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.private_hbm, name=name)
    nisa.dma_copy(dst=h, src=t)
    return h


def _layer_kinds(num_layers):
    """[(is_attention, index within its kind)] for the Qwen3.5 pattern (layer i % 4 == 3 is attention)."""
    kinds, ng, na = [], 0, 0
    for i in range(num_layers):
        if i % 4 == 3:
            kinds.append((True, na))
            na += 1
        else:
            kinds.append((False, ng))
            ng += 1
    return kinds, ng, na


# ----------------------------------------------------------------------------- scratch (reused by every layer)


def _scratch(m, T_pc, Lc, H, tp, atp, ep, E_L):
    """Per micro-batch HBM buffers, allocated once and reused by every layer (an 8K prompt's gathered rows are
    64 MB, so per-layer buffers would not fit 60 layers): residual x, resid1 r, normed xn, gathered xg, mixer
    ReduceScatter destinations, MoE dispatch / combine buffers."""
    hbm = nl.private_hbm
    bf = nl.bfloat16
    t = f"_m{m}"
    T = ep * T_pc
    R = atp * T_pc
    s = {
        "x": nl.ndarray((T_pc, H), dtype=bf, buffer=hbm, name="pfs_x" + t),
        "r": nl.ndarray((T_pc, H), dtype=bf, buffer=hbm, name="pfs_r" + t),
        "xn": nl.ndarray((T_pc, H), dtype=bf, buffer=hbm, name="pfs_xn" + t),
        "xs": nl.ndarray((T_pc, H), dtype=bf, buffer=hbm, name="pfs_mxs" + t),
        "ms": nl.ndarray((T_pc, NMETA), dtype=nl.float32, buffer=hbm, name="pfs_mms" + t),
        "so": nl.ndarray((T_pc, H), dtype=bf, buffer=hbm, name="pfs_mso" + t),
        "L2d": nl.ndarray((E_L * (T + P), 1), dtype=nl.float32, buffer=hbm, name="pfs_lists" + t),
        "e_full": nl.ndarray((T + P, H), dtype=bf, buffer=hbm, name="pfs_efull" + t),
    }
    if tp > 1:
        s["xg"] = nl.ndarray((Lc, H), dtype=bf, buffer=hbm, name="pfs_xg" + t)
        # GDN partials come back in shared HBM (nested kernel output): the ReduceScatter reads them directly
        s["rs_g"] = nl.ndarray((T_pc, H), dtype=bf, buffer=nl.shared_hbm, name="pfs_rsg" + t)
    if tp > atp:
        s["rs_src"] = nl.ndarray((R, H), dtype=bf, buffer=hbm, name="pfs_rssrc" + t)
        if atp > 1:
            s["rs_a"] = nl.ndarray((T_pc, H), dtype=bf, buffer=hbm, name="pfs_rsa" + t)
    if ep > 1:
        s["xgm"] = nl.ndarray((T, H), dtype=bf, buffer=hbm, name="pfs_mxg" + t)
        s["mgm"] = nl.ndarray((T, NMETA), dtype=nl.float32, buffer=hbm, name="pfs_mmg" + t)
        s["e_dst"] = nl.ndarray((T_pc, H), dtype=bf, buffer=hbm, name="pfs_medst" + t)
    return s


# ----------------------------------------------------------------------------- mixers


def _gather(sc, G):
    """Normed rows of this micro-batch gathered over the prompt group -> [Lc, H] in sequence order."""
    if G["tp"] == 1:
        return sc["xn"]
    ncc.all_gather(dsts=[sc["xg"]], srcs=[sc["xn"]], replica_group=G["gdn_group"], collective_dim=0)
    return sc["xg"]


def _gdn_mixer(sc, xg, W, G, init, state_row=0, valid_len=None):
    """GDN prefill over the gathered chunk (initial states = the previous chunk's, or rows ``state_row`` of the stack's
    state tensors for the first chunk of a prefill from state), ReduceScatter, residual."""
    conv0, rec0 = init
    out, nc, nr = gdn_prefill_tkg(
        xg, W["w_proj_p"], W["conv_w"], W["a_log"], W["dt_bias"], W["norm_w"], W["w_out"], conv0, rec0, G["Lc"],
        valid_len=valid_len, state_row=state_row,
    )
    if G["tp"] > 1:
        ncc.reduce_scatter(dsts=[sc["rs_g"]], srcs=[out], op=nl.add, replica_group=G["gdn_group"], collective_dim=0)
        rs = sc["rs_g"]
    else:
        rs = out
    _add_rows(sc["r"], sc["x"], rs, G["T_pc"], G["H"])
    return nc, nr


def _attn_mixer(sc, xg, W, G, kc, vc, slot, cos_c, sin_c, start, sub_sb, paged=None, k_tm=False):
    """Gated attention prefill of the chunk (cache rows start .. start + Lc - 1 of the slot, or with ``paged`` =
    (write_pages, prior_pages, prior_len) the chunk's pages after a runtime prior), then the attention subgroup reduces
    only the rows its members hold."""
    if paged is not None:
        wp, pp, plen = paged
        out, _, _ = attn_prefill_tkg(
            xg, W["w_qkv"], W["q_norm_w"], W["k_norm_w"], W["w_gate_p"], cos_c, sin_c, kc, vc, None, W["w_o"], 0,
            write_pages=wp, prior_pages=pp, prior_len=plen, k_tm=k_tm,
        )
    else:
        out, _, _ = attn_prefill_tkg(
            xg, W["w_qkv"], W["q_norm_w"], W["k_norm_w"], W["w_gate_p"], cos_c, sin_c, kc, vc, slot, W["w_o"], start,
            k_tm=k_tm,
        )
    tp, atp = G["tp"], G["atp"]
    T_pc, H = G["T_pc"], G["H"]
    if tp == atp:  # the subgroup is the whole group: reduce the whole chunk
        if tp > 1:
            ncc.reduce_scatter(dsts=[sc["rs_g"]], srcs=[out], op=nl.add, replica_group=G["atp_group"], collective_dim=0)
            rs = sc["rs_g"]
        else:
            rs = out
    else:
        _copy_slice_rt(sc["rs_src"], out, atp * T_pc, H, sub_sb)
        if atp > 1:
            ncc.reduce_scatter(
                dsts=[sc["rs_a"]], srcs=[sc["rs_src"]], op=nl.add, replica_group=G["atp_group"], collective_dim=0
            )
            rs = sc["rs_a"]
        else:
            rs = sc["rs_src"]
    _add_rows(sc["r"], sc["x"], rs, T_pc, H)


# ----------------------------------------------------------------------------- MoE (moe_prefill_ep_body split in two)


def _moe_front(sc, W, G):
    """Zero the routed buffers, route the home rows, issue the EP AllGathers, shared expert under them."""
    T_pc, ep = G["T_pc"], G["ep"]
    E_L = W["gate_up_w"].shape[0]
    T = ep * T_pc
    _zero_hbm(sc["L2d"].reshape((P, E_L * (T + P) // P)), nl.float32)
    _zero_hbm(sc["e_full"], nl.bfloat16)
    prefill_route(sc["r"], W["post_ln_w"], W["w_router_p"], sc["xs"], sc["ms"])
    if ep > 1:
        ncc.all_gather(dsts=[sc["mgm"]], srcs=[sc["ms"]], replica_group=G["ep_group"], collective_dim=0)
        ncc.all_gather(dsts=[sc["xgm"]], srcs=[sc["xs"]], replica_group=G["ep_group"], collective_dim=0)
    prefill_shared(sc["xs"], W["w_shared_p"], W["w_sd"], sc["so"])


def _moe_back(sc, W, G):
    """Per-expert lists + experts on the gathered rows, ReduceScatter[EP], x = r + bf16(e + shared)."""
    T_pc, ep = G["T_pc"], G["ep"]
    T = ep * T_pc
    xg = sc["xgm"] if ep > 1 else sc["xs"]
    mg = sc["mgm"] if ep > 1 else sc["ms"]
    e_full = moe_prefill_routed(
        xg, mg, W["expert_ids"], W["gate_up_w"], W["gate_up_scale"], W["down_w"], W["down_scale"],
        (sc["L2d"], sc["e_full"]),
    )
    if ep > 1:
        ncc.reduce_scatter(dsts=[sc["e_dst"]], srcs=[e_full[0:T]], op=nl.add, replica_group=G["ep_group"],
                           collective_dim=0)
        e = sc["e_dst"]
    else:
        e = e_full
    _finish(sc["x"], sc["r"], e, sc["so"], T_pc, G["H"])


# ----------------------------------------------------------------------------- layer + stack bodies


def prefill_layer_body(scs, mbs, is_attn, li, W, G, st):
    """One prefill layer over all micro-batches (plain traced code, inlined once per layer).

    scs: per micro-batch scratch; mbs: [(prompt b, chunk c)]; li: index of the layer within its kind;
    st: stack state (state / cache outputs, slots, per-chunk RoPE tables, attention subgroup). Prefill from state
    (patch 0003): with ``st["from_state"]`` a prompt's first chunk starts from its rows of ``conv_all`` / ``rec_all``
    (read before the last chunk writes them); ``st["paged"]`` = {attention layer: (write_pages, prior_pages,
    prior_len)} runs the attention on page pools from a runtime prior; ``st["valid"]`` ([1, 1] int32, one chunk) marks
    the rows past it padding."""
    xgs = []
    for m in range(len(mbs)):  # every micro-batch's normed rows gathered up front
        _norm_rows(scs[m]["x"], W["in_ln_w"], G["T_pc"], G["H"], scs[m]["xn"])
        xgs.append(_gather(scs[m], G))
    init = {}
    for m, (b, c) in enumerate(mbs):
        if is_attn:
            kvl = st["kvl"]
            paged = st["paged"][li] if st.get("paged") else None
            slot = None if paged is not None else st["slots"][(li % kvl) * G["bpg"] + b]
            _attn_mixer(
                scs[m], xgs[m], W, G, st["kc"], st["vc"], slot, st["cos"][c], st["sin"][c], c * G["Lc"], st["sub"],
                paged=paged, k_tm=st.get("k_tm", False),
            )
        else:
            row = li * G["bpg"] + b
            first = (st["conv_all"], st["rec_all"]) if st.get("from_state") else (None, None)
            nc, nr = _gdn_mixer(scs[m], xgs[m], W, G, init.get(b, first), state_row=0 if b in init else row,
                                valid_len=st.get("valid"))
            init[b] = (nc, nr)
            if c == G["nch"] - 1:  # the prompt's final states -> decode's state rows of this layer
                row = li * G["bpg"] + b
                nisa.dma_copy(dst=st["conv_all"][row : row + 1], src=nc)
                nisa.dma_copy(dst=st["rec_all"][row : row + 1], src=nr)
    for m in range(len(mbs)):
        _moe_front(scs[m], W, G)
    for m in range(len(mbs)):
        _moe_back(scs[m], W, G)


def _stack_body(
    x_shard, in_ln_w, w_proj_p, conv_w, a_log, dt_bias, norm_w, w_out, w_qkv, q_norm_w, k_norm_w, w_gate_p, cos, sin,
    w_o, attn_sub, post_ln_w, w_router_p, w_shared_p, w_sd, expert_ids, gate_up_w, gate_up_scale, down_w, down_scale,
    gdn_group, atp_group, ep_group, tp_size, atp_size, ep_size, seq_len, num_layers, n_chunks, kv_len, kv_layers,
):
    T_rows, H = x_shard.shape
    L = seq_len
    nch = n_chunks
    Lc = L // nch
    T_pc = Lc // tp_size
    bpg = T_rows // (nch * T_pc)
    vh = a_log.shape[0]
    Vd = vh * HEAD
    NT = w_proj_p.shape[2]
    Kd = (NT - 2 * vh - 2 * Vd) // 2
    Cc = 2 * Kd + Vd
    S = kv_len or L
    E_L = gate_up_w.shape[0]
    kinds, n_gdn, n_attn = _layer_kinds(num_layers)
    kvl = kv_layers or max(1, n_attn)
    kernel_assert(not _KMAJOR, "the prefill stack hands off token-major GDN states (QWEN35_GDN_KMAJOR=0)")
    kernel_assert(L % nch == 0 and Lc % tp_size == 0, f"chunk {Lc} must split over the {tp_size}-rank group")
    kernel_assert(T_rows == bpg * nch * T_pc and bpg >= 1, f"x_shard rows {T_rows} must be bpg * n_chunks * {T_pc}")
    kernel_assert(tp_size % atp_size == 0, "attention subgroups must tile the prompt group")
    kernel_assert((ep_size * T_pc) % P == 0, f"EP rows {ep_size * T_pc} per micro-batch must be a multiple of 128")
    kernel_assert(n_gdn >= 1 and S % P == 0 and S >= L, f"kv_len {S} must be >= L and a multiple of 128")
    kernel_assert(cos.shape[0] == L, "cos / sin must be [L, 32] (positions 0 .. L-1)")

    G = dict(tp=tp_size, atp=atp_size, ep=ep_size, gdn_group=gdn_group, atp_group=atp_group, ep_group=ep_group,
             Lc=Lc, T_pc=T_pc, H=H, bpg=bpg, nch=nch)
    W = dict(in_ln_w=in_ln_w, w_proj_p=w_proj_p, conv_w=conv_w, a_log=a_log, dt_bias=dt_bias, norm_w=norm_w,
             w_out=w_out, w_qkv=w_qkv, q_norm_w=q_norm_w, k_norm_w=k_norm_w, w_gate_p=w_gate_p, w_o=w_o,
             post_ln_w=post_ln_w, w_router_p=w_router_p, w_shared_p=w_shared_p, w_sd=w_sd, expert_ids=expert_ids,
             gate_up_w=gate_up_w, gate_up_scale=gate_up_scale, down_w=down_w, down_scale=down_scale)

    out = nl.ndarray((T_rows, H), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    h_last = nl.ndarray((bpg, H), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    conv_all = nl.ndarray((n_gdn * bpg, Cc, KWIN - 1), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    rec_all = nl.ndarray((n_gdn * bpg, HEAD, vh, HEAD), dtype=nl.float32, buffer=nl.shared_hbm)
    kc = nl.ndarray((kvl * bpg, D_ATT, S), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    vc = nl.ndarray((kvl * bpg, S, D_ATT), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    st = {"conv_all": conv_all, "rec_all": rec_all, "kc": kc, "vc": vc, "kvl": kvl}
    st["slots"] = [_const_i32(s, f"pfs_slot{s}") for s in range(kvl * bpg)] if n_attn else []
    if nch == 1:
        st["cos"], st["sin"] = [cos], [sin]
    else:  # per-chunk RoPE tables (positions c * Lc ..), copied once
        st["cos"], st["sin"] = [], []
        for c in range(nch):
            for key, src in (("cos", cos), ("sin", sin)):
                t = nl.ndarray((Lc, src.shape[1]), dtype=nl.float32, buffer=nl.private_hbm, name=f"pfs_{key}{c}")
                _copy_rows(t, 0, src, c * Lc, Lc, src.shape[1], nl.float32)
                st[key].append(t)
    sub_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sub_sb, src=attn_sub)
    st["sub"] = sub_sb

    mbs = [(b, c) for b in range(bpg) for c in range(nch)]
    scs = [_scratch(m, T_pc, Lc, H, tp_size, atp_size, ep_size, E_L) for m in range(len(mbs))]
    for m in range(len(mbs)):
        _copy_rows(scs[m]["x"], 0, x_shard, m * T_pc, T_pc, H)

    for i in range(num_layers):
        is_attn, li = kinds[i]
        prefill_layer_body(scs, mbs, is_attn, li, W, G, st)

    for m in range(len(mbs)):
        _copy_rows(out, m * T_pc, scs[m]["x"], 0, T_pc, H)
    for b in range(bpg):
        _copy_rows(h_last, b, scs[b * nch + nch - 1]["x"], T_pc - 1, 1, H)
    return out, h_last, conv_all, rec_all, kc, vc


@nki.jit
def qwen35_prefill_stack(
    x_shard,  # [bpg * n_chunks * T_pc, H] bf16 residual rows (layout: module docstring)
    in_ln_w,  # [H] fp32 input_layernorm (zero-centred)
    w_proj_p, conv_w, a_log, dt_bias, norm_w, w_out,  # GDN TP shard (gdn_prefill_tkg / gdn_decode_tkg layouts)
    w_qkv, q_norm_w, k_norm_w, w_gate_p,  # attention TP shard (attn_prefill_tkg layouts)
    cos, sin,  # [L, 32] fp32 RoPE tables of positions 0 .. L-1
    w_o,  # [nq * 256, H] bf16
    attn_sub,  # [1, 1] int32 this rank's attention subgroup inside its prompt group (group position // atp)
    post_ln_w,  # [H] fp32 post_attention_layernorm
    w_router_p, w_shared_p, w_sd,  # MoE front (moe_prefill layouts; the whole shared expert)
    expert_ids,  # [1, E_L] fp32 global ids of the local experts
    gate_up_w, gate_up_scale, down_w, down_scale,  # local MXFP8 experts
    gdn_group: ReplicaGroup = None,  # prompt groups: rank lists of tp_size consecutive ranks
    atp_group: ReplicaGroup = None,  # attention TP subgroups: atp_size consecutive ranks
    ep_group: ReplicaGroup = None,  # MoE EP replicas: ep_size consecutive ranks
    tp_size: int = 1,
    atp_size: int = 1,
    ep_size: int = 1,
    seq_len: int = 0,  # L tokens per prompt
    num_layers: int = 4,
    n_chunks: int = 1,  # token-chunk micro-batches per prompt
    kv_len: int = 0,  # cache length S (0: L)
    kv_layers: int = 0,  # distinct cache layers (0: one per attention layer; 1: shared, timing only)
):
    return _stack_body(
        x_shard, in_ln_w, w_proj_p, conv_w, a_log, dt_bias, norm_w, w_out, w_qkv, q_norm_w, k_norm_w, w_gate_p, cos,
        sin, w_o, attn_sub, post_ln_w, w_router_p, w_shared_p, w_sd, expert_ids, gate_up_w, gate_up_scale, down_w,
        down_scale, gdn_group, atp_group, ep_group, tp_size, atp_size, ep_size, seq_len, num_layers, n_chunks, kv_len,
        kv_layers,
    )


@nki.jit
def qwen35_prefill_handoff(
    x_shard, in_ln_w, w_proj_p, conv_w, a_log, dt_bias, norm_w, w_out, w_qkv, q_norm_w, k_norm_w, w_gate_p, cos, sin,
    w_o, attn_sub, post_ln_w, w_router_p, w_shared_p, w_sd, expert_ids, gate_up_w, gate_up_scale, down_w, down_scale,
    x_dec,  # [n_dec * bpg, H] bf16 normed decode-step inputs (step-major), fed to every mixer layer
    cos_dec, sin_dec,  # [n_dec * bpg, 32] fp32 RoPE of positions L + s
    pos_dec,  # [n_dec * bpg, 1] int32 = L + s
    gdn_group: ReplicaGroup = None,
    atp_group: ReplicaGroup = None,
    ep_group: ReplicaGroup = None,
    tp_size: int = 1,
    atp_size: int = 1,
    ep_size: int = 1,
    seq_len: int = 0,
    num_layers: int = 4,
    n_chunks: int = 1,
    kv_len: int = 0,
    n_dec: int = 2,
):
    """Prefill stack, then the DECODE kernels continue from its hand-off (test of the P -> D contract): for every
    GDN layer ``gdn_decode_tkg`` runs n_dec steps from that layer's conv_all / rec_all rows, and ``attn_decode_tkg``
    runs n_dec steps at positions L, L+1, .. on the cache the prefill wrote (attention layer 0; needs one cache
    layer). Returns the stack outputs plus the decode partials dec_g [n_gdn * n_dec * bpg, H] (rows (gi, s, b)) and
    dec_a [n_dec * bpg, H]."""
    out, h_last, conv_all, rec_all, kc, vc = _stack_body(
        x_shard, in_ln_w, w_proj_p, conv_w, a_log, dt_bias, norm_w, w_out, w_qkv, q_norm_w, k_norm_w, w_gate_p, cos,
        sin, w_o, attn_sub, post_ln_w, w_router_p, w_shared_p, w_sd, expert_ids, gate_up_w, gate_up_scale, down_w,
        down_scale, gdn_group, atp_group, ep_group, tp_size, atp_size, ep_size, seq_len, num_layers, n_chunks, kv_len,
        0,
    )
    H = x_shard.shape[1]
    bpg = h_last.shape[0]
    _, n_gdn, n_attn = _layer_kinds(num_layers)
    kernel_assert(n_attn == 1, "the hand-off check decodes on one attention layer's cache (num_layers 4..7)")
    steps = []
    for s in range(n_dec):  # per-step inputs as whole tensors (the decode kernels address their inputs from 0)
        xs = nl.ndarray((bpg, H), dtype=nl.bfloat16, buffer=nl.private_hbm, name=f"pfh_x{s}")
        _copy_rows(xs, 0, x_dec, s * bpg, bpg, H)
        cs = nl.ndarray((bpg, cos_dec.shape[1]), dtype=nl.float32, buffer=nl.private_hbm, name=f"pfh_c{s}")
        _copy_rows(cs, 0, cos_dec, s * bpg, bpg, cos_dec.shape[1], nl.float32)
        sn = nl.ndarray((bpg, sin_dec.shape[1]), dtype=nl.float32, buffer=nl.private_hbm, name=f"pfh_s{s}")
        _copy_rows(sn, 0, sin_dec, s * bpg, bpg, sin_dec.shape[1], nl.float32)
        ps = nl.ndarray((bpg, 1), dtype=nl.int32, buffer=nl.private_hbm, name=f"pfh_p{s}")
        _copy_rows(ps, 0, pos_dec, s * bpg, bpg, 1, nl.int32)
        steps.append((xs, cs, sn, ps))

    dec_g = nl.ndarray((n_gdn * n_dec * bpg, H), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for gi in range(n_gdn):
        conv = conv_all[gi * bpg : (gi + 1) * bpg]
        rec = rec_all
        r0 = gi * bpg
        for s in range(n_dec):
            o, conv, rec = gdn_decode_tkg(steps[s][0], w_proj_p, conv_w, conv, a_log, dt_bias, norm_w, rec, w_out, r0)
            r0 = 0  # later steps read the fresh [bpg, ...] state tensors
            _copy_rows(dec_g, (gi * n_dec + s) * bpg, o, 0, bpg, H)
    # decode's cache buffers hold finite data beyond the history (attn_decode_tkg reads all S positions under a
    # -inf bias); the stack writes only positions 0 .. L-1, so zero the tail as decode's buffer would be
    S = kc.shape[2]
    L = seq_len
    if S > L:
        z = nl.ndarray((P, S - L), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(z, 0.0)
        for sl in range(kc.shape[0]):
            for j in range(D_ATT // P):
                nisa.dma_copy(dst=kc.ap(pattern=[[S, P], [1, S - L]], offset=(sl * D_ATT + j * P) * S + L), src=z)
        zv = nl.ndarray((P, D_ATT), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(zv, 0.0)
        for sl in range(vc.shape[0]):
            for r0 in range(L, S, P):
                m = min(P, S - r0)
                nisa.dma_copy(dst=vc.ap(pattern=[[D_ATT, m], [1, D_ATT]], offset=(sl * S + r0) * D_ATT), src=zv[:m])
    dec_a = nl.ndarray((n_dec * bpg, H), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for s in range(n_dec):
        xs, cs, sn, ps = steps[s]
        o, kc, vc = attn_decode_tkg(xs, w_qkv, q_norm_w, k_norm_w, w_gate_p, cs, sn, kc, vc, ps, w_o)
        _copy_rows(dec_a, s * bpg, o, 0, bpg, H)
    return out, h_last, conv_all, rec_all, kc, vc, dec_g, dec_a
