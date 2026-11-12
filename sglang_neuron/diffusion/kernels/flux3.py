# SPDX-License-Identifier: Apache-2.0
"""NKI kernels of the FLUX.3 block: nkilib's qkv_cte (QK RMSNorm
before RoPE, rotate-half RoPE), attention_cte (d-major output) and output_projection_cte from the container's nkilib,
and the FLUX kernels' AdaLN and const-max ring attention (``kernels/flux``); device only."""

from __future__ import annotations

import nki
import nki.language as nl
import torch
import torch.nn.functional as F
from nkilib.core.attention.attention_cte import attention_cte
from nkilib.core.output_projection.output_projection_cte import output_projection_cte
from nkilib.core.qkv.qkv_cte import qkv_cte
from nkilib.core.utils.common_types import NormType, QKNormConfig, QuantizationType

from sglang_neuron.backend.stack import wrap_nki
from sglang_neuron.diffusion.kernels.flux.adaln_quant import AdaLNQuantKernelArgs, adaln_quant_kernel
from sglang_neuron.diffusion.kernels.flux.ring_attention_const_max_fwd import ring_attention_const_max_fwd
from sglang_neuron.nki import logical_cores


@nki.jit
def _qkv(hidden, weight, cos, sin, q_gamma, k_gamma, d_head, heads, eps):
    qk_norm = QKNormConfig(q_norm=NormType.RMS_NORM, k_norm=NormType.RMS_NORM, eps=eps, q_gamma_norm_weights=q_gamma,
                           k_gamma_norm_weights=k_gamma)
    return qkv_cte(input=hidden, fused_qkv_weights=weight, fused_rope=True, cos_cache=cos, sin_cache=sin,
                   d_head=d_head, num_q_heads=heads, num_kv_heads=heads, qk_norm_pre_rope=qk_norm)


@nki.jit
def _attention(q, k, v):
    return attention_cte(q=q, k=k, v=v, scale=1.0, causal_mask=False, tp_q=True, tp_k=True, tp_out=True,
                         cache_softmax=False, softmax_dtype=nl.float32, mm_out_dtype=nl.float32)


@nki.jit
def _adaln(hidden, mul, add, residual, gate, eps):
    return adaln_quant_kernel(hidden, AdaLNQuantKernelArgs(norm_type=NormType.LAYER_NORM, eps=eps), mul=mul, add=add,
                              residual=residual, gate=gate)


def adaln_modulate(hidden: torch.Tensor, mul: torch.Tensor, add: torch.Tensor, *, eps: float,
                   residual: torch.Tensor | None = None, gate: torch.Tensor | None = None):
    """``LayerNorm(x) * mul + add`` of ``x = hidden``, or of ``x = residual + gate * hidden`` (returned too) given a
    residual, as one kernel launch per batch row; fewer token rows than the launch's programs cannot be split and run in
    torch (fp32)."""
    if hidden.shape[1] < logical_cores():
        x = hidden.float() if gate is None else hidden.float() * gate.float()
        summed = None if residual is None else (x + residual.float()).to(residual.dtype)
        x = hidden.float() if residual is None else summed.float()
        out = (F.layer_norm(x, (x.shape[-1],), eps=eps) * mul.float() + add.float()).to(hidden.dtype)
        return out if residual is None else (summed, out)
    launch = wrap_nki(_adaln)[logical_cores()]
    outs = [launch(hidden[b].contiguous(), mul[b].to(hidden.dtype).contiguous(), add[b].to(hidden.dtype).contiguous(),
                   None if residual is None else residual[b].contiguous(),
                   None if gate is None else gate[b].contiguous(), eps)
            for b in range(hidden.shape[0])]
    if residual is None:
        return torch.stack(outs)
    return torch.stack([r for r, _ in outs]), torch.stack([o for _, o in outs])


def qkv_proj(hidden: torch.Tensor, weight: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
             q_gamma: torch.Tensor, k_gamma: torch.Tensor, *, eps: float) -> torch.Tensor:
    """``hidden @ weight`` with QK RMSNorm and rotate-half RoPE on each head, in nkilib's qkv_cte; ``weight`` is at
    most 4096 columns wide (the kernel's limit)."""
    d_head = q_gamma.shape[-1]
    return wrap_nki(_qkv)[logical_cores()](hidden, weight, cos, sin, q_gamma, k_gamma, d_head,
                                           weight.shape[1] // (3 * d_head), eps)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    """Non-causal ``softmax(scale * q @ k^T) @ v`` in nkilib's attention_cte, d-major ``[B, N, d, S]`` out; it takes
    ``[B * N, S, d]`` operands, q pre-scaled. The keys may be longer than the queries."""
    b, n, s, d = q.shape
    q = (q.reshape(b * n, s, d) * scale).contiguous()
    k, v = k.reshape(b * n, -1, d).contiguous(), v.reshape(b * n, -1, d).contiguous()
    return wrap_nki(_attention)[logical_cores()](q, k, v).reshape(b, n, d, s)


def o_proj(attention: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``attention @ weight`` in nkilib's output_projection_cte (d-major input), validated for up to 17 heads."""
    return wrap_nki(output_projection_cte)[logical_cores()](attention=attention, weight=weight, bias=None,
                                                            quantization_type=QuantizationType.NONE,
                                                            input_scales=None, weight_scales=None)


def ring_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, kv_len: int,
                   replica_groups: tuple) -> torch.Tensor:
    """The FLUX kernels' const-max ring attention over token-major ``[B, S, N, d]`` local q, k, v of each
    sequence-parallel rank: K and V rotate over the ring while the local blocks run, and the keys past ``kv_len``,
    which the caller zeroed, drop out of every row exactly. Returns d-major ``[B, N, d, S]``."""
    workers = len(replica_groups[0])
    out = wrap_nki(ring_attention_const_max_fwd)[logical_cores()](
        q=q.contiguous(), k=k.contiguous(), v=v.contiguous(), replica_groups=replica_groups, num_workers=workers,
        softmax_scale=scale, training=False, num_pad_keys=q.shape[1] * workers - kv_len)
    return out.transpose(-2, -1)
