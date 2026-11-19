# SPDX-License-Identifier: Apache-2.0
"""One MiMo-V2 MoE decoder layer's decode in one NKI kernel, its two all-reduces inside: no XLA op between one layer's
input and the next's.

    attn = attn_half(x)              this rank's partial of the attention half (attn_half.py), the K/V written
    attn = all_reduce(attn)          over the attention TP ranks, inside the kernel
    h    = x + attn
    moe  = moe_half(h)               this rank's partial of the MoE half (moe_half.py)
    moe  = all_reduce(moe)
    out  = h + moe

The MoE layer follows nkilib's dense decode megakernel's SBUF path (``nkilib.experimental.transformer.transformer_tkg``,
``sbuf_residual_and_cc``): the residual stream stays in SBUF from the layer's load to its store, each half hands its
partial over in SBUF (this program's half of H), the all-reduce runs SBUF to SBUF and the halves cross between the two
programs by sendrecv (_sb_all_reduce). The dense layer still takes its HBM path: each partial in shared HBM, a core
barrier so both programs' halves have landed, the all-reduce into a shared HBM destination, the residual through
SBUF. Every shared HBM tensor of the kernel
has a name of its own: two unnamed ones made by the same helper landed on one buffer, the layer's output on its
attention residual (tools/mimo_v2/check_layer.py), as nkilib's allocator requires a unique name for each.

``mesh`` is every attention-TP group's ranks as a tuple of tuples (a ReplicaGroup cannot cross into the trace), as the
plugin's DCP decode passes its groups (ops/attention/attention_decode.py)."""

import nki
import nki.collectives as nccl
import nki.isa as nisa
import nki.language as nl
from nkilib.core.mlp.mlp import mlp
from nkilib.core.utils.common_types import ActFnType, NormType

from sglang_neuron.models.mimo_v2.kernels.attn_half import attn_half
from sglang_neuron.models.mimo_v2.kernels.moe_half import moe_half

P = 128


def _all_reduce(part, rg, n_prg: int, name: str):
    """The sum of ``part`` over the group. The send buffer is fenced after both programs filled it, and the destination
    before it is read: unfenced, in-kernel sums came back wrong and differed from run to run on trn2 (trnrt's kernels,
    docs/MEGAKERNELS.md, "Collectives inside a kernel")."""
    if n_prg > 1:
        nisa.core_barrier(data=part, cores=(0, 1))
    reduced = nl.ndarray(part.shape, dtype=part.dtype, buffer=nl.shared_hbm, name=name)
    nccl.all_reduce(dsts=[reduced], srcs=[part], op=nl.add, replica_group=rg)
    if n_prg > 1:
        nisa.core_barrier(data=reduced, cores=(0, 1))
    return reduced


def _sb_all_reduce(part, R: int, H: int, rg, pid, n_prg: int, pipe: int):
    """The sum over the group of the rows ``0 .. R - 1`` of each program's half of H (``part`` [128, H / programs] in
    SBUF), gathered whole into each program: [128, H] in SBUF, rows ``0 .. R - 1``. The all-reduce runs SBUF to SBUF,
    each program over its own half, and the halves cross between the two programs by sendrecv (nkilib transformer_tkg's
    ``sbuf_residual_and_cc``): no HBM partial and no core barrier (tools/mimo_v2/bench_cc.py: 36 against 45 us for
    [32, 4096] over 64 ranks, the graph's all-reduce 37.5)."""
    HH = H // n_prg
    src = nl.ndarray((R, HH), dtype=part.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=src, src=part[0:R, 0:HH])
    red = nl.ndarray((R, HH), dtype=part.dtype, buffer=nl.sbuf)
    nccl.all_reduce(dsts=[red], srcs=[src], op=nl.add, replica_group=rg)
    full = nl.ndarray((P, H), dtype=part.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=full[0:R, pid * HH:(pid + 1) * HH], src=red)
    if n_prg > 1:
        nisa.sendrecv(src=red, dst=full[0:R, (1 - pid) * HH:(2 - pid) * HH], send_to_rank=1 - pid,
                      recv_from_rank=1 - pid, pipe_id=pipe)
    return full


def _residual(x2d, add2d, B: int, H: int, name: str):
    """x + add, [B, H] in shared HBM."""
    out = nl.ndarray((B, H), dtype=x2d.dtype, buffer=nl.shared_hbm, name=name)
    a = nl.ndarray((B, H), dtype=x2d.dtype, buffer=nl.sbuf)
    b = nl.ndarray((B, H), dtype=x2d.dtype, buffer=nl.sbuf)
    s = nl.ndarray((B, H), dtype=x2d.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=a, src=x2d)
    nisa.dma_copy(dst=b, src=add2d)
    nisa.tensor_tensor(dst=s, data1=a, data2=b, op=nl.add)
    nisa.dma_copy(dst=out, src=s)
    return out


@nki.jit
def moe_layer(X, attn_gamma, W_qkv, cos, sin, active_blocks_table, K_cache, V_cache, attention_mask, sink, slots,
              W_o, moe_gamma, router_wT, expert_bias, ebase, gate_up_w, down_w, gate_up_scale, down_scale, mesh,
              k_scale=None, v_scale=None, eps=1e-6, softmax_scale=1.0, v_head_dim=128, route_scale=1.0, top_k=8,
              fp8_act=True):
    """X [B, S, H] bf16, the layer's input (S tokens a row, a NEURON_MTP span or 1); the attention half's inputs as
    attn_half's (``k_scale`` / ``v_scale`` over an FP8 cache), the MoE half's as moe_half's. Returns the layer's
    output [B x S, H] bf16. The caches are written in place and not returned: returned beside the output, the
    caller's graph got another tensor than the output in the output's place (measured by tools/mimo_v2/check_layer.py),
    while the in-place write lands without the return."""
    B, S, H = X.shape
    R = B * S
    n_prg = nl.num_programs(axes=0)
    cells = []
    for cell in mesh:  # NKI's tracer takes no list comprehension
        cells.append(list(cell))
    rg = nccl.ReplicaGroup(cells)

    pid = nl.program_id(axis=0)
    HH = H // n_prg
    # the layer's rows in SBUF, the residual stream from here to the store
    x_sb = nl.ndarray((P, H), dtype=X.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_sb[0:R, 0:H], src=X.reshape((R, H)).ap(pattern=[[H, R], [1, H]], offset=0))
    attn, _, _ = attn_half(X, attn_gamma, W_qkv, cos, sin, active_blocks_table, K_cache, V_cache,
                           attention_mask, sink, slots, W_o, k_scale, v_scale, rmsnorm_X_eps=eps,
                           softmax_scale=softmax_scale, v_head_dim=v_head_dim, out_in_sbuf=True)
    attn = _sb_all_reduce(attn, R, H, rg, pid, n_prg, 3)
    h = nl.ndarray((P, H), dtype=X.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=h[0:R, 0:H], data1=x_sb[0:R, 0:H], data2=attn[0:R, 0:H], op=nl.add)
    moe = moe_half(X.reshape((R, H)), moe_gamma, router_wT, expert_bias, ebase, gate_up_w, down_w, gate_up_scale,
                   down_scale, eps=eps, route_scale=route_scale, top_k=top_k, hidden_sb=h, out_in_sbuf=True,
                   fp8_act=fp8_act)
    moe = _sb_all_reduce(moe, R, H, rg, pid, n_prg, 3)
    o_sb = nl.ndarray((P, H), dtype=X.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=o_sb[0:R, 0:H], data1=h[0:R, 0:H], data2=moe[0:R, 0:H], op=nl.add)
    # each program stores its own half of H
    out = nl.ndarray((R, H), dtype=X.dtype, buffer=nl.shared_hbm, name="layer_output")
    nisa.dma_copy(dst=out.ap(pattern=[[H, R], [1, HH]], offset=pid * HH), src=o_sb[0:R, pid * HH:(pid + 1) * HH])
    return out


@nki.jit
def dense_layer(X, attn_gamma, W_qkv, cos, sin, active_blocks_table, K_cache, V_cache, attention_mask, sink, slots,
                W_o, mlp_gamma, W_gate, W_up, W_down, mesh, k_scale=None, v_scale=None, eps=1e-6, softmax_scale=1.0,
                v_head_dim=128):
    """A dense layer's decode (layer 0): moe_layer with nkilib's mlp (RMSNorm, SiLU-gated gate / up, down) in place of
    the MoE half; W_gate / W_up [H, I] and W_down [I, H] this rank's shard of the intermediate dimension. Returns the
    layer's output [B x S, H] bf16; the caches are written in place."""
    B, S, H = X.shape
    R = B * S
    n_prg = nl.num_programs(axes=0)
    cells = []
    for cell in mesh:
        cells.append(list(cell))
    rg = nccl.ReplicaGroup(cells)

    attn, _, _ = attn_half(X, attn_gamma, W_qkv, cos, sin, active_blocks_table, K_cache, V_cache, attention_mask,
                           sink, slots, W_o, k_scale, v_scale, rmsnorm_X_eps=eps, softmax_scale=softmax_scale,
                           v_head_dim=v_head_dim)
    attn = _all_reduce(attn, rg, n_prg, "dense_attn_reduced")
    h = _residual(X.reshape((R, H)), attn, R, H, "dense_attn_residual")
    outputs = mlp(hidden_tensor=h.reshape((R, 1, H)), gate_proj_weights_tensor=W_gate, up_proj_weights_tensor=W_up,
                  down_proj_weights_tensor=W_down, normalization_weights_tensor=mlp_gamma,
                  normalization_type=NormType.RMS_NORM, activation_fn=ActFnType.SiLU, eps=eps,
                  use_tkg_down_proj_column_tiling=False)
    part = outputs[0].reshape((R, H))
    dense = _all_reduce(part, rg, n_prg, "dense_mlp_reduced")
    return _residual(h, dense, R, H, "dense_layer_output")
