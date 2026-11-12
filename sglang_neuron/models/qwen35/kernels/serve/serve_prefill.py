# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-397B serving prefill: one segment of prefill layers with per-layer weights, writing the attention KV straight
into the decode's slot pool.

Adapted from an earlier PD serving copy of this module (``../PROVENANCE.md`` names the source)
(``qwen35_serve_prefill``), which composes the giga kernels' prefill stack body (prefill_stack.py
``prefill_layer_body``) over ``num_layers`` layers (a multiple of 4: 3 GDN + 1 attention) with stacked per-kind weights
and per-layer views; the host runs one compiled segment once per segment of the model, chaining the residual (output
``out`` of segment k is input ``x_shard`` of segment k + 1). Changes:

* the attention caches are the decode's whole layer-major slot pools (``k_cache_t [L_a * Spa, 256, S]``,
  ``v_cache [L_a * Spa, S, 256]``) and ``kv_slot [n_attn * bpg, 1]`` int32 gives each (segment attention layer,
  prompt)'s row in them (row ``a * Spa + page``), so the attention prefill writes the prompt's K / V in place at
  positions 0 .. L-1 of its slot (the original passed a hand-off cache of the segment's rows with constant slots);
* the GDN final states still go to the segment's hand-off rows ``conv_state [ng * bpg, Cc, 3]``,
  ``rec_state [ng * bpg, 128, vh, 128]`` (the host scatters them into the state pool's slots);
* imports name the in-tree prefill stack.

Prompts must be exactly ``seq_len`` tokens: the GDN states handed off are the states after token L-1.

Chunked prefill from a prefix (the radix cache's hits, the chunk grid's continuations; ``paged=True``, one prompt per
group, one micro-batch):

* the GDN hand-off rows are also the initial state: the host gathers them from the request's slot (a restored
  checkpoint, the previous chunk's end, or zeros for a fresh slot) and the first chunk starts from them;
* the attention caches are page pools (``k_cache_t [L_a * NPAGE, 256, PG]``, ``v_cache [L_a * NPAGE, PG, 256]``):
  ``write_pages [na, L / PG]`` int32 gives the pool row (``a * NPAGE + page``) of each PG rows of the chunk,
  ``prior_pages [na, MPP]`` (MPP = 0: no prior, a fresh prompt) the rows of the request's pages before it, padded with
  the layer's page 0, and ``prior_len [1, 1]`` the tokens before the chunk; ``cos`` / ``sin`` are of positions
  prior_len + t;
* ``valid_len [1, 1]`` int32: rows at or past it are padding (a prompt's last chunk in its bucket); the GDN states
  skip them.
"""

import nki
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from nkilib.core.utils.kernel_assert import kernel_assert
from ..giga.prefill_stack import (
    _KMAJOR,
    D_ATT,
    HEAD,
    KWIN,
    _copy_rows,
    _layer_kinds,
    _scratch,
    prefill_layer_body,
)

P = 128


def qwen35_serve_prefill(
    x_shard,  # [bpg * n_chunks * T_pc, H] bf16 residual rows (prefill_stack layout)
    in_ln_g, w_proj_p, conv_w, a_log, dt_bias, norm_w, w_out,  # GDN TP shard, stacked [ng, ...]
    in_ln_a, w_qkv, q_norm_w, k_norm_w, w_gate_p, w_o,  # attention TP shard, stacked [na, ...]
    cos, sin,  # [L, 32] fp32 RoPE of positions 0 .. L-1
    conv_state, rec_state,  # the segment's GDN hand-off rows (module docstring), written
    k_cache_t, v_cache,  # the decode's whole slot KV pools, written in place at the rows kv_slot names
    kv_slot,  # [na * bpg, 1] int32 pool row of each (attention layer of the segment, prompt)
    attn_sub,  # [1, 1] int32 attention subgroup inside the prompt group
    post_ln_w, w_router_p, w_shared_p, w_sd,  # MoE front, stacked [num_layers, ...]
    expert_ids,  # [1, E_L] fp32 global ids of the local experts
    gate_up_w, gate_up_scale, down_w, down_scale,  # local MXFP8 experts, stacked [num_layers, E_L, ...]
    gdn_group: ReplicaGroup = None,
    atp_group: ReplicaGroup = None,
    ep_group: ReplicaGroup = None,
    tp_size: int = 1,
    atp_size: int = 1,
    ep_size: int = 1,
    seq_len: int = 0,
    num_layers: int = 4,
    n_chunks: int = 1,
    write_pages=None,  # [na, L / PG] int32 (paged)
    prior_pages=None,  # [na, MPP] int32 (paged, MPP > 0)
    prior_len=None,  # [1, 1] int32 (paged, MPP > 0)
    valid_len=None,  # [1, 1] int32 (paged)
    paged: bool = False,
    k_tm: bool = False,  # the pool's K pages are token-major [rows, S, 256] (SGLANG_NEURON_QWEN35_KV_TM)
):
    T_rows, H = x_shard.shape
    L = seq_len
    nch = n_chunks
    Lc = L // nch
    T_pc = Lc // tp_size
    bpg = T_rows // (nch * T_pc)
    vh = a_log.shape[1]
    Vd = vh * HEAD
    NT = w_proj_p.shape[3]
    Kd = (NT - 2 * vh - 2 * Vd) // 2
    Cc = 2 * Kd + Vd
    S = v_cache.shape[1]  # the K stack's layout (k_tm) is the pool's; V is always [rows, S, 256]
    E_L = gate_up_w.shape[1]
    kinds, n_gdn, n_attn = _layer_kinds(num_layers)
    kernel_assert(num_layers % 4 == 0, "segments hold whole 3 GDN + 1 attention groups")
    kernel_assert(in_ln_g.shape[0] == n_gdn and in_ln_a.shape[0] == n_attn, "stacked weights per kind")
    kernel_assert(not _KMAJOR, "the prefill stack hands off token-major GDN states (QWEN35_GDN_KMAJOR=0)")
    kernel_assert(L % nch == 0 and Lc % tp_size == 0, f"chunk {Lc} must split over the {tp_size}-rank group")
    kernel_assert(T_rows == bpg * nch * T_pc and bpg >= 1, f"x_shard rows {T_rows} must be bpg * n_chunks * {T_pc}")
    kernel_assert(tp_size % atp_size == 0, "attention subgroups must tile the prompt group")
    kernel_assert((ep_size * T_pc) % P == 0, f"EP rows {ep_size * T_pc} per micro-batch must be a multiple of 128")
    kernel_assert(S % P == 0, f"the KV slot / page {S} must be a multiple of 128")
    kernel_assert(cos.shape[0] == L, "cos / sin must be [L, 32] (positions 0 .. L-1)")
    kernel_assert(tuple(conv_state.shape) == (n_gdn * bpg, Cc, KWIN - 1), "conv_state: the segment's GDN rows")
    kernel_assert(tuple(rec_state.shape) == (n_gdn * bpg, HEAD, vh, HEAD), "rec_state: the segment's GDN rows")
    if paged:
        kernel_assert(nch == 1 and bpg == 1, "a paged prefill is one prompt's chunk in one micro-batch")
        kernel_assert(tuple(write_pages.shape) == (n_attn, L // S), "write_pages: [na, L / PG]")
        kernel_assert(L % S == 0, f"the chunk {L} must be whole {S}-token pages")
    else:
        kernel_assert(S >= L, f"the KV slot {S} must be >= L")
    kernel_assert(tuple(k_cache_t.shape[1:]) == ((S, D_ATT) if k_tm else (D_ATT, S)),
                  "k_cache_t: the slot pool [rows, 256, S] (token-major [rows, S, 256] when k_tm)")
    kernel_assert(tuple(v_cache.shape[1:]) == (S, D_ATT), "v_cache: the slot pool [rows, S, 256]")
    kernel_assert(tuple(kv_slot.shape) == (n_attn * bpg, 1), "kv_slot: one pool row per (attention layer, prompt)")

    G = dict(tp=tp_size, atp=atp_size, ep=ep_size, gdn_group=gdn_group, atp_group=atp_group, ep_group=ep_group,
             Lc=Lc, T_pc=T_pc, H=H, bpg=bpg, nch=nch)

    def weights(i):
        is_attn, li = kinds[i]
        gi = 0 if is_attn else li  # the other kind's entries are unused by this layer
        ai = li if is_attn else 0
        return dict(
            in_ln_w=in_ln_a[ai] if is_attn else in_ln_g[gi],
            w_proj_p=w_proj_p[gi], conv_w=conv_w[gi], a_log=a_log[gi], dt_bias=dt_bias[gi], norm_w=norm_w[gi],
            w_out=w_out[gi], w_qkv=w_qkv[ai], q_norm_w=q_norm_w[ai], k_norm_w=k_norm_w[ai], w_gate_p=w_gate_p[ai],
            w_o=w_o[ai], post_ln_w=post_ln_w[i], w_router_p=w_router_p[i], w_shared_p=w_shared_p[i], w_sd=w_sd[i],
            expert_ids=expert_ids, gate_up_w=gate_up_w[i], gate_up_scale=gate_up_scale[i], down_w=down_w[i],
            down_scale=down_scale[i],
        )

    out = nl.ndarray((T_rows, H), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    h_last = nl.ndarray((bpg, H), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    st = {"conv_all": conv_state, "rec_all": rec_state, "kc": k_cache_t, "vc": v_cache, "kvl": n_attn,
          "k_tm": k_tm}
    # each (attention layer, prompt)'s pool row: a [1, 1] int32 tile of its own, as the stack's constant slots
    slots = []
    for s in range(n_attn * bpg):
        t = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.private_hbm, name=f"svp_slot{s}")
        nisa.dma_copy(dst=t, src=kv_slot[s : s + 1])
        slots.append(t)
    st["slots"] = slots
    if paged:
        # each attention layer's page rows as tensors of their own, as the slots above
        st["from_state"] = True
        mpp = 0 if prior_pages is None else prior_pages.shape[1]
        plen = None
        if mpp:
            plen = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.private_hbm, name="svp_plen")
            nisa.dma_copy(dst=plen, src=prior_len)
        vlen = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.private_hbm, name="svp_vlen")
        nisa.dma_copy(dst=vlen, src=valid_len)
        st["valid"] = vlen
        st["paged"] = {}
        for a in range(n_attn):
            wp = nl.ndarray((1, L // S), dtype=nl.int32, buffer=nl.private_hbm, name=f"svp_wp{a}")
            nisa.dma_copy(dst=wp, src=write_pages[a : a + 1])
            pp = None
            if mpp:
                pp = nl.ndarray((1, mpp), dtype=nl.int32, buffer=nl.private_hbm, name=f"svp_pp{a}")
                nisa.dma_copy(dst=pp, src=prior_pages[a : a + 1])
            st["paged"][a] = (wp, pp, plen)
    if nch == 1:
        st["cos"], st["sin"] = [cos], [sin]
    else:
        st["cos"], st["sin"] = [], []
        for c in range(nch):
            for key, src in (("cos", cos), ("sin", sin)):
                t = nl.ndarray((Lc, src.shape[1]), dtype=nl.float32, buffer=nl.private_hbm, name=f"svp_{key}{c}")
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
        prefill_layer_body(scs, mbs, is_attn, li, weights(i), G, st)
    for m in range(len(mbs)):
        _copy_rows(out, m * T_pc, scs[m]["x"], 0, T_pc, H)
    for b in range(bpg):
        _copy_rows(h_last, b, scs[b * nch + nch - 1]["x"], T_pc - 1, 1, H)
    return out, h_last, conv_state, rec_state, k_cache_t, v_cache


qwen35_serve_prefill_jit = nki.jit(qwen35_serve_prefill)
