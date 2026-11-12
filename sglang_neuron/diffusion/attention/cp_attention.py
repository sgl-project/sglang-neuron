# SPDX-License-Identifier: Apache-2.0
"""NKI flash attention, the context-parallel self-attention fallback and the o-projection."""

import nki
import nki.language as nl
import torch
from nkilib.core.attention.attention_cte import attention_cte
from nkilib.core.output_projection.output_projection_cte import output_projection_cte
from nkilib.core.utils.common_types import QuantizationType

from sglang_neuron.diffusion.kernels.kernel_choice import device_reason, record_kernel_choice
from sglang_neuron.diffusion.neuron_api import wrap_nki_kernel
from sglang_neuron.diffusion.vn.utils.neuron_utils import can_run_kernel


@nki.jit
def _attention_kernel(q, k, v):
    """Non-causal flash attention with a **d-major** output (``tp_out=True``).

    Takes ``[BN, S, D]`` q/k/v and returns ``[BN, D, S_q]`` rather than
    ``[BN, S_q, D]``, because every consumer of this attention is an o-projection
    kernel that wants ``[B, N, D, S]`` — so emitting d-major here removes an HBM
    transpose of the whole attention output at each call site.

    ``tp_out`` only swaps which MM2 operand is stationary (both orientations need
    the same internal P-transpose), so the added in-kernel cost is just a 128xD
    ``nc_transpose`` of the softmax reciprocal per Q group.
    """
    return attention_cte(
        q=q,
        k=k,
        v=v,
        scale=1.0,
        causal_mask=False,
        tp_q=True,
        tp_k=True,
        tp_out=True,
        cache_softmax=False,
        softmax_dtype=nl.float32,
        mm_out_dtype=nl.float32,
    )


# Wrapped once, at import, as a module-level launcher. A wrap_nki call inside compiled code
# makes Dynamo guard on the @nki.jit object's __wrapped__; on a warm start from a Dynamo
# precompile package that attribute comes back as _Missing and the guard fails. Launches at the
# configured LNC, 1 or 2.
_attention_launch = wrap_nki_kernel(_attention_kernel, supported_lnc=(1, 2))


MAX_O_PROJ_HEADS = 17


def _can_use_o_proj_kernel(
    active: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> bool:
    if not can_run_kernel(active):
        return record_kernel_choice("o_proj", False, device_reason(active))

    if active.dim() != 4 or weight.dim() != 2:
        return record_kernel_choice("o_proj", False, "unexpected operand rank")

    B, N, D, S = active.shape
    ND, H = weight.shape

    if bias.dim() != 2 or bias.shape != (1, H):
        return record_kernel_choice("o_proj", False, "unexpected bias shape")

    eligible = (
        N * D == ND
        and N <= MAX_O_PROJ_HEADS
        and D <= 128
        and H <= 16384 + 4321
        and B * S <= 128 * 1024
        and H % 2 == 0
    )
    if not eligible:
        return record_kernel_choice("o_proj", False, "shape outside the kernel limits")
    return record_kernel_choice("o_proj", True, "eligible")


@nki.jit
def _o_proj_kernel(
    active,
    weight,
    bias,
):
    """Output projection using the NKI-Lib CTE kernel.

    Input:
        active: [B, N, D, S]
        weight: [N*D, H]
        bias:   [1, H]

    Output:
        [B, S, H]
    """
    return output_projection_cte(
        active,
        weight,
        bias,
        QuantizationType.NONE,
        None,
        None,
    )


# Launches at the configured LNC, 1 or 2.
_o_proj_launch = wrap_nki_kernel(_o_proj_kernel, supported_lnc=(1, 2))


def o_proj(
    active: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    if _can_use_o_proj_kernel(active, weight, bias):
        return _o_proj_launch(active, weight, bias)

    B, N, D, S = active.shape
    x = active.reshape(B, N * D, S).transpose(1, 2)
    return torch.matmul(x, weight) + bias


# attention_cte kernel constraints. The kernel folds the leading [B, N] dims into a single
# batch, so the effective batch it sees is B * N.
_ATTN_MAX_BS = 512


_ATTN_MAX_SEQLEN = 131072


_ATTN_MAX_HEAD_DIM = 128


def _can_use_attention_kernel(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> bool:
    """Whether the NKI ``attention_cte`` kernel can run for these inputs.

    Same contract as :func:`_can_use_o_proj_kernel`: returns False (so the
    caller takes the torch fallback) when NKI kernels can't run on this device
    (CPU tensors, native fake-tensor tracing, or kernels disabled — via
    ``can_run_kernel``) or when the reshaped ``[B*N, S, D]`` operands exceed the
    kernel's batch / sequence / head-dim tiling limits.
    """
    if not can_run_kernel(value):
        return record_kernel_choice("attention", False, device_reason(value))

    # query/key/value are 4D [B, N, S, D]; the kernel folds B*N into its batch.
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        return record_kernel_choice("attention", False, "q/k/v are not 4D [B, N, S, D]")

    B, N, S_q, D = query.shape
    S_k = key.shape[2]

    if B * N > _ATTN_MAX_BS:
        return record_kernel_choice("attention", False, "B*N exceeds the kernel batch limit")
    if S_q > _ATTN_MAX_SEQLEN or S_k > _ATTN_MAX_SEQLEN:
        return record_kernel_choice("attention", False, "sequence exceeds the kernel limit")
    if D > _ATTN_MAX_HEAD_DIM:
        return record_kernel_choice("attention", False, "head dim exceeds the kernel limit")

    return record_kernel_choice("attention", True, "eligible")


def _torch_attention(query, key, value, scale):
    """PyTorch fallback for :func:`nki_attention` (non-causal softmax attention).

    Matches the ``attention_cte`` math used by :func:`_attention_kernel`
    (``causal_mask=False``): ``softmax(scale * Q @ K^T) @ V`` with Q/K/V in the
    ``[B, N, S, D]`` layout. Softmax runs in float32 for parity with the kernel's
    ``softmax_dtype=nl.float32`` before casting back to the input dtype.
    """
    scores = torch.matmul(query.float() * scale, key.float().transpose(-2, -1))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, value.float()).to(query.dtype)


def nki_attention(query, key, value, scale):
    """Non-causal flash attention over ``[B, N, S, D]`` q/k/v, with a torch fallback.

    Returns **d-major** ``[B, N, D, S_q]`` — the layout the o-projection kernels
    consume — straight from the kernel (``tp_out=True``, see
    :func:`_attention_kernel`) rather than transposing the output afterwards.
    """
    if not _can_use_attention_kernel(query, key, value):
        return _torch_attention(query, key, value, scale).transpose(2, 3).contiguous()

    B, N, S_q, D = query.shape
    S_k = key.shape[2]

    # Wrapper configuration:
    # tp_q=True  -> [BN, S_q, D]
    # tp_k=True  -> [BN, S_k, D]
    # tp_out=True -> [BN, D, S_q]
    q_3d = (query.reshape(B * N, S_q, D) * scale).contiguous()
    k_3d = key.reshape(B * N, S_k, D).contiguous()
    v_3d = value.reshape(B * N, S_k, D).contiguous()

    return _attention_launch(q_3d, k_3d, v_3d).reshape(B, N, D, S_q)


def cp_self_attention(query, key, value, scale, cp_size, cp_group):
    """Context-parallel self-attention by all-gathering K/V, then local flash attention.

    Takes already-projected, RoPE'd, head-split q/k/v **token-major** in
    ``[B, local_S, N, D]`` and returns the attention output **d-major** as
    ``[B, N, D, local_S]`` — the layout the o-projection kernels consume. The FLUX blocks call
    it where their own CP kernels cannot run (CP size 1, CPU, fake-tensor tracing, NKI kernels
    disabled).
    """
    query = query.transpose(1, 2)  # [B, local_S, N, D] -> [B, N, local_S, D]
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    if cp_size > 1:
        # All-gather K/V across the CP group to the full sequence
        # ([B, N, local_S, D] -> [B, N, S, D]), then local flash attention.
        key = cp_group.all_gather(key.contiguous(), dim=2)
        value = cp_group.all_gather(value.contiguous(), dim=2)
    return nki_attention(query, key, value, scale)
