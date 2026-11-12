"""Speculative-verify variants of the Qwen3.5 decode stack: k + 1 tokens per sequence (NKI, LNC=1).

Same parallel layout and token bookkeeping as ``layer_decode`` (token-SP residual, mixer TP
groups, EP over all ranks); every rank now holds ``S = k + 1`` consecutive token rows per
sequence it owns, so after any AllGather the rows are sequence-major (``b*S + s``), which is
the order ``gdn_verify_tkg`` / ``attn_verify_tkg`` expect.

  GDN layers       -> gdn_verify_tkg   (S sequential steps, per-step state snapshots, slot-indexed start)
  attention layers -> attn_verify_tkg  (S causal queries per sequence, S KV writes at pos + s)
  MoE              -> decode's fused MoE block; beyond 128 EP-gathered rows (C * S > 128) it runs
                      the router / shared expert per 128-row block and loads each local expert's
                      weights once for all blocks (as C=256 decode). QWEN35_VERIFY_MOE=compact
                      compacts each expert's routed rows instead (verify_moe); =chunk re-streams
                      the experts per 128-row chunk (upper bound).
"""

import os

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from .attn_verify import attn_verify_tkg
from .gdn_verify import gdn_verify_tkg
from .layer_decode import _add_rows, _gathered_norm, _moe_half, _scatter_residual
from .moe_experts_mx import _PREFETCH, experts_body
from .moe_front import moe_front_body, rmsnorm_zc_tok
from .verify_moe import moe_rows_compact_body

P = 128
# MoE beyond 128 EP-gathered rows: "blocked" (decode's fused block: router / shared per 128-row block,
# every expert applied densely to each block with its weights loaded once - as C=256 decode), "compact"
# (routed rows per expert compacted) or "chunk" (dense 128-row chunks, each re-streams the experts:
# upper bound). Compaction cap per expert.
_VERIFY_MOE = os.environ.get("QWEN35_VERIFY_MOE", "blocked")
_VERIFY_CAP = int(os.environ.get("QWEN35_VERIFY_CAP", "96"))


def _gathered_norm_rows(x_shard, in_ln_w, group, size, tag):
    """``_gathered_norm`` for any gathered row count (input RMSNorm in 128-row chunks)."""
    T_shard, H = x_shard.shape
    T_tp = T_shard * size
    if T_tp <= P:
        return _gathered_norm(x_shard, in_ln_w, group, size, tag)
    dt = x_shard.dtype
    hbm = nl.private_hbm
    a_src = nl.ndarray((T_shard, H), dtype=dt, buffer=hbm, name="a_src" + tag)
    a_dst = nl.ndarray((T_tp, H), dtype=dt, buffer=hbm, name="a_dst" + tag)
    nisa.dma_copy(dst=a_src, src=x_shard)
    ncc.all_gather(dsts=[a_dst], srcs=[a_src], replica_group=group, collective_dim=0)
    w_row = nl.ndarray((P, H), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=w_row, src=in_ln_w.ap(pattern=[[0, P], [1, H]], offset=0))
    nisa.tensor_scalar(dst=w_row, data=w_row, op0=nl.add, operand0=1.0)
    xn_hbm = nl.ndarray((T_tp, H), dtype=nl.bfloat16, buffer=hbm, name="xn" + tag)
    for r0 in range(0, T_tp, P):
        n = min(P, T_tp - r0)
        x_sb = nl.ndarray((P, H), dtype=dt, buffer=nl.sbuf)[:n]
        nisa.dma_copy(dst=x_sb, src=a_dst[r0 : r0 + n])
        xn = rmsnorm_zc_tok(x_sb, w_row[:n], n, H, 1e-6, nl.bfloat16)
        nisa.dma_copy(dst=xn_hbm[r0 : r0 + n], src=xn)
    return xn_hbm


def _gdn_verify_mixer(
    x_shard, in_ln_w, w_proj_p, conv_w, conv_in, a_log, dt_bias, norm_w, rec_in, src_slot, w_out, tp_group,
    tp_size, tag,
):
    """GDN over the TP group for S tokens per sequence; returns (resid1, conv_out, rec_out)."""
    xn_hbm = _gathered_norm_rows(x_shard, in_ln_w, tp_group, tp_size, tag)
    mix_part, conv_out, rec_out = gdn_verify_tkg(
        xn_hbm, w_proj_p, conv_w, conv_in, a_log, dt_bias, norm_w, rec_in, src_slot, w_out
    )
    return _scatter_residual(x_shard, mix_part, tp_group, tp_size, tag), conv_out, rec_out


def _attn_verify_mixer(
    x_shard, in_ln_w, w_qkv, q_norm_w, k_norm_w, w_gate_p, cos, sin, k_cache_t, v_cache, pos, w_o, atp_group,
    atp_size, tag,
):
    xn_hbm = _gathered_norm_rows(x_shard, in_ln_w, atp_group, atp_size, tag)
    mix_part, k_cache_t, v_cache = attn_verify_tkg(
        xn_hbm, w_qkv, q_norm_w, k_norm_w, w_gate_p, cos, sin, k_cache_t, v_cache, pos, w_o
    )
    return _scatter_residual(x_shard, mix_part, atp_group, atp_size, tag)


def _moe_half_rows(
    resid1, post_ln_w, w_router_p, sel, w_shared_p, w_sd, gate_up_w, gate_up_scale, down_w, down_scale,
    ep_group, ep_size, tag, out_buffer,
):
    """``_moe_half`` for any EP-gathered token count (128-row chunks beyond 128)."""
    T_shard, H = resid1.shape
    T_ep = T_shard * ep_size
    if T_ep <= P or _VERIFY_MOE == "blocked":
        return _moe_half(
            resid1, post_ln_w, w_router_p, sel, w_shared_p, w_sd, gate_up_w, gate_up_scale, down_w, down_scale,
            ep_group, ep_size, tag, out_buffer,
        )
    dt = resid1.dtype
    hbm = nl.private_hbm
    m_src = nl.ndarray((T_shard, H), dtype=dt, buffer=hbm, name="m_src" + tag)
    m_dst = nl.ndarray((T_ep, H), dtype=dt, buffer=hbm, name="m_dst" + tag)
    nisa.dma_copy(dst=m_src, src=resid1)
    ncc.all_gather(dsts=[m_dst], srcs=[m_src], replica_group=ep_group, collective_dim=0)
    e_src = nl.ndarray((T_ep, H), dtype=dt, buffer=hbm, name="e_src" + tag)
    if _VERIFY_MOE == "compact":
        moe_rows_compact_body(
            m_dst, post_ln_w, w_router_p, sel, w_shared_p, w_sd, gate_up_w, gate_up_scale, down_w, down_scale, e_src,
            _VERIFY_CAP,
        )
    for r0 in range(0, T_ep if _VERIFY_MOE != "compact" else 0, P):
        r1 = min(T_ep, r0 + P)
        xn_sb, aff_sb, so_sb = moe_front_body(m_dst[r0:r1], post_ln_w, w_router_p, sel, w_shared_p, w_sd)
        acc = experts_body(xn_sb, aff_sb, gate_up_w, gate_up_scale, down_w, down_scale, acc_init=so_sb)
        e_sb = nl.ndarray((P, H), dtype=dt, buffer=nl.sbuf)[: r1 - r0]
        nisa.tensor_copy(dst=e_sb, src=acc)
        nisa.dma_copy(dst=e_src[r0:r1], src=e_sb)
    e_dst = nl.ndarray((T_shard, H), dtype=dt, buffer=hbm, name="e_dst" + tag)
    ncc.reduce_scatter(dsts=[e_dst], srcs=[e_src], op=nl.add, replica_group=ep_group, collective_dim=0)
    out = nl.ndarray((T_shard, H), dtype=dt, buffer=out_buffer, name="out" + tag)
    _add_rows(out, resid1, e_dst, T_shard, H)
    return out


@nki.jit
def qwen35_gdn_moe_verify_layer(
    x_shard,  # [S * seqs_per_rank, H] bf16
    in_ln_w, w_proj_p, conv_w, conv_in, a_log, dt_bias, norm_w, rec_in, src_slot, w_out,
    post_ln_w, w_router_p, sel, w_shared_p, w_sd, gate_up_w, gate_up_scale, down_w, down_scale,
    tp_group: ReplicaGroup,
    ep_group: ReplicaGroup,
    tp_size: int,
    ep_size: int,
):
    """One GDN + MoE layer over S tokens per sequence (correctness target for the verify path)."""
    _PREFETCH.clear()
    resid1, conv_out, rec_out = _gdn_verify_mixer(
        x_shard, in_ln_w, w_proj_p, conv_w, conv_in, a_log, dt_bias, norm_w, rec_in, src_slot, w_out, tp_group,
        tp_size, "_L0",
    )
    out = _moe_half_rows(
        resid1, post_ln_w, w_router_p, sel, w_shared_p, w_sd, gate_up_w, gate_up_scale, down_w, down_scale,
        ep_group, ep_size, "_L0", nl.shared_hbm,
    )
    return out, conv_out, rec_out


@nki.jit
def qwen35_hybrid_verify_stack(
    x_shard,  # [S * seqs_per_rank, H]
    in_ln_w, w_proj_p, conv_w, conv_in, a_log, dt_bias, norm_w, rec_in, src_slot, w_out,  # GDN shard
    w_qkv, q_norm_w, k_norm_w, w_gate_p, cos, sin, k_cache_t, v_cache, pos, w_o,  # attention shard
    post_ln_w, w_router_p, sel, w_shared_p, w_sd, gate_up_w, gate_up_scale, down_w, down_scale,
    tp_group: ReplicaGroup,
    atp_group: ReplicaGroup,
    ep_group: ReplicaGroup,
    tp_size: int,
    atp_size: int,
    ep_size: int,
    num_layers: int,
):
    """Verify pass of the Qwen3.5 layer pattern in one NEFF (timing scaffold like qwen35_hybrid_stack:
    layers of a kind share weights / state buffers / the KV cache, but each issues all its DMAs)."""
    _PREFETCH.clear()
    x = x_shard
    for i in range(num_layers):
        tag = f"_L{i}"
        if i % 4 == 3:
            resid1 = _attn_verify_mixer(
                x, in_ln_w, w_qkv, q_norm_w, k_norm_w, w_gate_p, cos, sin, k_cache_t, v_cache, pos, w_o, atp_group,
                atp_size, tag,
            )
        else:
            resid1, _, _ = _gdn_verify_mixer(
                x, in_ln_w, w_proj_p, conv_w, conv_in, a_log, dt_bias, norm_w, rec_in, src_slot, w_out, tp_group,
                tp_size, tag,
            )
        x = _moe_half_rows(
            resid1, post_ln_w, w_router_p, sel, w_shared_p, w_sd, gate_up_w, gate_up_scale, down_w, down_scale,
            ep_group, ep_size, tag, nl.shared_hbm if i == num_layers - 1 else nl.private_hbm,
        )
    return x
