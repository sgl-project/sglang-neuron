# SPDX-License-Identifier: Apache-2.0
"""One MiMo-V2 MoE decoder layer's decode in one NKI kernel, its two all-reduces inside: no XLA op between one layer's
input and the next's.

    attn = attn_half(x)              this rank's partial of the attention half (attn_half.py), the K/V written
    attn = all_reduce(attn)          over the attention TP ranks, inside the kernel
    h    = x + attn
    moe  = moe_half(h)               this rank's partial of the MoE half (moe_half.py)
    moe  = all_reduce(moe)
    out  = h + moe

The structure is nkilib's dense decode megakernel's (``nkilib.experimental.transformer.transformer_tkg``, its HBM
path): each half's partial in shared HBM, a core barrier so both programs' halves have landed, the all-reduce issued by
both programs into a shared HBM destination, and the residual add through SBUF. Every shared HBM tensor of the kernel
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
              eps=1e-6, softmax_scale=1.0, v_head_dim=128, route_scale=1.0, top_k=8):
    """X [B, 1, H] bf16, the layer's input; the attention half's inputs as attn_half's, the MoE half's as moe_half's.
    Returns the layer's output [B, H] bf16. The caches are written in place and not returned: returned beside the
    output, the caller's graph got another tensor than the output in the output's place (measured by
    tools/mimo_v2/check_layer.py), while the in-place write lands without the return."""
    B, _, H = X.shape
    n_prg = nl.num_programs(axes=0)
    cells = []
    for cell in mesh:  # NKI's tracer takes no list comprehension
        cells.append(list(cell))
    rg = nccl.ReplicaGroup(cells)

    attn, _, _ = attn_half(X, attn_gamma, W_qkv, cos, sin, active_blocks_table, K_cache, V_cache,
                                       attention_mask, sink, slots, W_o, rmsnorm_X_eps=eps,
                                       softmax_scale=softmax_scale, v_head_dim=v_head_dim)
    attn = _all_reduce(attn, rg, n_prg, "attn_reduced")
    # both programs write the same residual and each reads its own write, so it needs no fence
    # (tools/mimo_v2/check_layer.py checks the result without one)
    h = _residual(X.reshape((B, H)), attn, B, H, "attn_residual")
    moe = moe_half(h, moe_gamma, router_wT, expert_bias, ebase, gate_up_w, down_w, gate_up_scale, down_scale,
                   eps=eps, route_scale=route_scale, top_k=top_k)
    moe = _all_reduce(moe, rg, n_prg, "moe_reduced")
    out = _residual(h, moe, B, H, "layer_output")
    return out


@nki.jit
def dense_layer(X, attn_gamma, W_qkv, cos, sin, active_blocks_table, K_cache, V_cache, attention_mask, sink, slots,
                W_o, mlp_gamma, W_gate, W_up, W_down, mesh, eps=1e-6, softmax_scale=1.0, v_head_dim=128):
    """A dense layer's decode (layer 0): moe_layer with nkilib's mlp (RMSNorm, SiLU-gated gate / up, down) in place of
    the MoE half; W_gate / W_up [H, I] and W_down [I, H] this rank's shard of the intermediate dimension. Returns the
    layer's output [B, H] bf16; the caches are written in place."""
    B, _, H = X.shape
    n_prg = nl.num_programs(axes=0)
    cells = []
    for cell in mesh:
        cells.append(list(cell))
    rg = nccl.ReplicaGroup(cells)

    attn, _, _ = attn_half(X, attn_gamma, W_qkv, cos, sin, active_blocks_table, K_cache, V_cache, attention_mask,
                           sink, slots, W_o, rmsnorm_X_eps=eps, softmax_scale=softmax_scale, v_head_dim=v_head_dim)
    attn = _all_reduce(attn, rg, n_prg, "dense_attn_reduced")
    h = _residual(X.reshape((B, H)), attn, B, H, "dense_attn_residual")
    outputs = mlp(hidden_tensor=h.reshape((B, 1, H)), gate_proj_weights_tensor=W_gate, up_proj_weights_tensor=W_up,
                  down_proj_weights_tensor=W_down, normalization_weights_tensor=mlp_gamma,
                  normalization_type=NormType.RMS_NORM, activation_fn=ActFnType.SiLU, eps=eps,
                  use_tkg_down_proj_column_tiling=False)
    part = outputs[0].reshape((B, H))
    dense = _all_reduce(part, rg, n_prg, "dense_mlp_reduced")
    return _residual(h, dense, B, H, "dense_layer_output")
