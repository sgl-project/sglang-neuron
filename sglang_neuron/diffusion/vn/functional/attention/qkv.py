# SPDX-License-Identifier: Apache-2.0
from typing import Optional
from torch import Tensor
import torch

import nki

from nkilib.core.qkv.qkv import qkv as nki_qkv_kernel
from nkilib.core.utils.common_types import (
    DtypeMode,
    NormType,
    QKNormConfig,
    QKVOutputLayout,
    QKVWeightLayout,
    QuantizationType,
)
from nkilib.core.utils.allocator import SbufManager

from sglang_neuron.diffusion.kernels.kernel_choice import device_reason, record_kernel_choice
from sglang_neuron.diffusion.neuron_api import wrap_nki_kernel
from sglang_neuron.diffusion.vn.utils.neuron_utils import can_run_kernel


def _maybe_build_sbm(
    sbm_lower_bound: Optional[int],
    sbm_upper_bound: Optional[int],
    sbm_use_auto_alloc: bool,
    sbm_default_stack_alloc: bool,
) -> Optional[SbufManager]:
    """Reconstruct SbufManager only when bounds are explicitly provided."""

    if sbm_lower_bound is None and sbm_upper_bound is None:
        return None

    return SbufManager(
        sb_lower_bound=sbm_lower_bound,
        sb_upper_bound=sbm_upper_bound,
        use_auto_alloc=sbm_use_auto_alloc,
        default_stack_alloc=sbm_default_stack_alloc,
    )


@nki.jit
def _torch_compatible_qkv_kernel(
    hidden: Tensor,
    qkv_weights: Tensor,
    norm_weights: Optional[Tensor] = None,
    mlp_prev: Optional[Tensor] = None,
    attn_prev: Optional[Tensor] = None,
    d_head: Optional[int] = None,
    output_layout: QKVOutputLayout = QKVOutputLayout.BSD,
    eps: float = 1e-6,
    norm_type: NormType = NormType.NO_NORM,
    use_dma_transpose: bool = True,
    store_output_in_sbuf: bool = False,
    bias: Optional[Tensor] = None,
    norm_b: Optional[Tensor] = None,
    hidden_actual: Optional[int] = None,
    cos_cache: Optional[Tensor] = None,
    sin_cache: Optional[Tensor] = None,
    num_q_heads: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
    quantization_type: QuantizationType = QuantizationType.NONE,
    qkv_w_scale: Optional[Tensor] = None,
    qkv_in_scale: Optional[Tensor] = None,
    weight_layout: QKVWeightLayout = QKVWeightLayout.CONTIGUOUS,
    fused_residual_add: Optional[bool] = False,
    fused_rope: Optional[bool] = False,
    qk_norm_pre_rope_q_norm: Optional[NormType] = None,
    qk_norm_pre_rope_k_norm: Optional[NormType] = None,
    qk_norm_pre_rope_eps: float = 1e-6,
    qk_norm_pre_rope_gamma_fused_in_rope_caches: bool = False,
    qk_norm_pre_rope_q_gamma: Optional[Tensor] = None,
    qk_norm_pre_rope_k_gamma: Optional[Tensor] = None,
    qk_norm_pre_rope_q_beta: Optional[Tensor] = None,
    qk_norm_pre_rope_k_beta: Optional[Tensor] = None,
    qk_norm_post_rope_q_norm: Optional[NormType] = None,
    qk_norm_post_rope_k_norm: Optional[NormType] = None,
    qk_norm_post_rope_eps: float = 1e-6,
    qk_norm_post_rope_gamma_fused_in_rope_caches: bool = False,
    qk_norm_post_rope_q_gamma: Optional[Tensor] = None,
    qk_norm_post_rope_k_gamma: Optional[Tensor] = None,
    qk_norm_post_rope_q_beta: Optional[Tensor] = None,
    qk_norm_post_rope_k_beta: Optional[Tensor] = None,
    sbm_lower_bound: int = None,
    sbm_upper_bound: int = None,
    sbm_use_auto_alloc: bool = False,
    sbm_default_stack_alloc: bool = True,
    use_auto_allocation: bool = False,
):
    sbm = _maybe_build_sbm(
        sbm_lower_bound=sbm_lower_bound,
        sbm_upper_bound=sbm_upper_bound,
        sbm_use_auto_alloc=sbm_use_auto_alloc,
        sbm_default_stack_alloc=sbm_default_stack_alloc,
    )

    has_pre_rope = (
        qk_norm_pre_rope_q_norm is not None or qk_norm_pre_rope_k_norm is not None
    )
    has_post_rope = (
        qk_norm_post_rope_q_norm is not None or qk_norm_post_rope_k_norm is not None
    )

    assert not has_pre_rope or (num_q_heads is not None and num_kv_heads is not None), (
        "num_q_heads and num_kv_heads are required when pre-rope qk_norm is enabled"
    )
    assert not has_post_rope or (
        num_q_heads is not None and num_kv_heads is not None
    ), "num_q_heads and num_kv_heads are required when post-rope qk_norm is enabled"

    qk_norm_pre_rope_config = (
        QKNormConfig(
            q_norm=qk_norm_pre_rope_q_norm,
            k_norm=qk_norm_pre_rope_k_norm,
            eps=qk_norm_pre_rope_eps,
            gamma_fused_in_rope_caches=qk_norm_pre_rope_gamma_fused_in_rope_caches,
        )
        if has_pre_rope
        else None
    )

    qk_norm_post_rope_config = (
        QKNormConfig(
            q_norm=qk_norm_post_rope_q_norm,
            k_norm=qk_norm_post_rope_k_norm,
            eps=qk_norm_post_rope_eps,
            gamma_fused_in_rope_caches=qk_norm_post_rope_gamma_fused_in_rope_caches,
        )
        if has_post_rope
        else None
    )

    return nki_qkv_kernel(
        # Core inputs
        input=hidden,
        fused_qkv_weights=qkv_weights,
        output_layout=output_layout,
        # Bias
        bias=bias,
        # Quantization
        quantization_type=quantization_type,
        qkv_w_scale=qkv_w_scale,
        qkv_in_scale=qkv_in_scale,
        weight_layout=weight_layout,
        # Fused Residual Add
        fused_residual_add=fused_residual_add,
        mlp_prev=mlp_prev,
        attention_prev=attn_prev,
        # Fused Norm
        fused_norm_type=norm_type,
        gamma_norm_weights=norm_weights,
        layer_norm_bias=norm_b,
        norm_eps=eps,
        hidden_actual=hidden_actual,
        # Fused RoPE
        fused_rope=fused_rope,
        cos_cache=cos_cache,
        sin_cache=sin_cache,
        d_head=d_head,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        # No in-kernel KV cache write. The cache arguments stay explicit, so the call does not
        # depend on the installed nkilib's defaults.
        k_cache=None,
        v_cache=None,
        k_scale=None,
        v_scale=None,
        fp8_max=None,
        fp8_min=None,
        kv_dtype=None,
        use_block_kv=False,
        transpose_k_cache=False,
        fp8_packed=False,
        block_size=None,
        slot_mapping=None,
        # Output storage
        store_output_in_sbuf=store_output_in_sbuf,
        # QK-Norm: passed directly as pre/post rope
        qk_norm_pre_rope=qk_norm_pre_rope_config,
        qk_norm_pre_rope_q_gamma=qk_norm_pre_rope_q_gamma,
        qk_norm_pre_rope_k_gamma=qk_norm_pre_rope_k_gamma,
        qk_norm_pre_rope_q_beta=qk_norm_pre_rope_q_beta,
        qk_norm_pre_rope_k_beta=qk_norm_pre_rope_k_beta,
        qk_norm_post_rope=qk_norm_post_rope_config,
        qk_norm_post_rope_q_gamma=qk_norm_post_rope_q_gamma,
        qk_norm_post_rope_k_gamma=qk_norm_post_rope_k_gamma,
        qk_norm_post_rope_q_beta=qk_norm_post_rope_q_beta,
        qk_norm_post_rope_k_beta=qk_norm_post_rope_k_beta,
        # SBUF management
        sbm=sbm,
        use_auto_allocation=use_auto_allocation,
        # DMA
        load_input_with_DMA_transpose=use_dma_transpose,
        # FP8 dtype: AUTO resolves to OCP (e4m3fn) on TRN3, NON_OCP on TRN2
        dtype_mode=DtypeMode.AUTO,
    )


def _apply_qk_norm(
    output: Tensor,
    fused_qkv_dim: int,
    d_head: Optional[int],
    num_q_heads: int,
    num_kv_heads: int,
    qk_norm_q_norm: Optional[NormType],
    qk_norm_k_norm: Optional[NormType],
    qk_norm_eps: float,
    qk_norm_gamma_fused_in_rope_caches: bool,
    qk_norm_q_gamma: Optional[Tensor],
    qk_norm_k_gamma: Optional[Tensor],
    qk_norm_q_beta: Optional[Tensor],
    qk_norm_k_beta: Optional[Tensor],
) -> Tensor:
    """Apply per-head normalization to Q and K heads in the fused QKV output."""
    assert not qk_norm_gamma_fused_in_rope_caches, (
        "gamma_fused_in_rope_caches is not supported in the PyTorch reference for now"
    )
    assert qk_norm_q_norm == NormType.RMS_NORM, (
        f"Only RMS_NORM is supported for QK-norm for now, got q_norm={qk_norm_q_norm}"
    )
    assert qk_norm_k_norm == NormType.RMS_NORM, (
        f"Only RMS_NORM is supported for QK-norm for now, got k_norm={qk_norm_k_norm}"
    )

    qk_d_head = (
        d_head
        if d_head is not None
        else fused_qkv_dim // (num_q_heads + 2 * num_kv_heads)
    )
    num_norm_heads = num_q_heads + num_kv_heads
    for head_idx in range(num_norm_heads):
        start = head_idx * qk_d_head
        end = start + qk_d_head
        head = output[:, :, start:end].float()
        variance = head.pow(2).mean(-1, keepdim=True)
        head = head * torch.rsqrt(variance + qk_norm_eps)
        gamma = qk_norm_q_gamma if head_idx < num_q_heads else qk_norm_k_gamma
        if gamma is not None:
            head = head * gamma
        beta = qk_norm_q_beta if head_idx < num_q_heads else qk_norm_k_beta
        if beta is not None:
            head = head + beta
        output[:, :, start:end] = head.to(output.dtype)
    return output


def _apply_qkv_output_layout(
    x: Tensor,
    d_head: Optional[int],
    output_layout: QKVOutputLayout,
) -> Tensor:
    """Permute a ``[B, S, num_heads * d_head]`` tensor into ``output_layout``.

    ``num_heads`` is ``x.shape[-1] // d_head``.

    Args:
        x: A projection output, shape ``[B, S, num_heads * d_head]``.
        d_head: The head dimension. Each layout except ``BSD`` needs it.
        output_layout: The requested layout.

    Returns:
        ``x`` when ``output_layout`` is ``BSD``. A permuted view for each other
        layout.

    Examples:
        >>> import torch
        >>> from nkilib.core.utils.common_types import QKVOutputLayout
        >>> q = torch.zeros(1, 8, 256)  # 4 heads x d_head 64
        >>> _apply_qkv_output_layout(q, 64, QKVOutputLayout.BNSd).shape
        torch.Size([1, 4, 8, 64])
    """
    if output_layout == QKVOutputLayout.BSD:
        return x

    if d_head is None:
        raise ValueError(f"d_head must be specified for {output_layout} output layout")

    B, S, D = x.shape
    x = x.reshape(B, S, D // d_head, d_head)

    if output_layout == QKVOutputLayout.BNSd:
        return x.permute(0, 2, 1, 3)
    if output_layout == QKVOutputLayout.NBSd:
        return x.permute(2, 0, 1, 3)
    if output_layout == QKVOutputLayout.NBdS:
        return x.permute(2, 0, 3, 1)
    raise ValueError(f"Unsupported output layout: {output_layout}")


def _torch_qkv_impl(
    hidden: Tensor,
    qkv_weights: Tensor,
    norm_weights: Optional[Tensor] = None,
    mlp_prev: Optional[Tensor] = None,
    attn_prev: Optional[Tensor] = None,
    d_head: Optional[int] = None,
    output_layout: QKVOutputLayout = QKVOutputLayout.BSD,
    eps: float = 1e-6,
    norm_type: NormType = NormType.NO_NORM,
    bias: Optional[Tensor] = None,
    norm_b: Optional[Tensor] = None,
    hidden_actual: Optional[int] = None,
    cos_cache: Optional[Tensor] = None,
    sin_cache: Optional[Tensor] = None,
    num_q_heads: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
    qk_norm_pre_rope_q_norm: Optional[NormType] = None,
    qk_norm_pre_rope_k_norm: Optional[NormType] = None,
    qk_norm_pre_rope_eps: float = 1e-6,
    qk_norm_pre_rope_gamma_fused_in_rope_caches: bool = False,
    qk_norm_pre_rope_q_gamma: Optional[Tensor] = None,
    qk_norm_pre_rope_k_gamma: Optional[Tensor] = None,
    qk_norm_pre_rope_q_beta: Optional[Tensor] = None,
    qk_norm_pre_rope_k_beta: Optional[Tensor] = None,
    qk_norm_post_rope_q_norm: Optional[NormType] = None,
    qk_norm_post_rope_k_norm: Optional[NormType] = None,
    qk_norm_post_rope_eps: float = 1e-6,
    qk_norm_post_rope_gamma_fused_in_rope_caches: bool = False,
    qk_norm_post_rope_q_gamma: Optional[Tensor] = None,
    qk_norm_post_rope_k_gamma: Optional[Tensor] = None,
    qk_norm_post_rope_q_beta: Optional[Tensor] = None,
    qk_norm_post_rope_k_beta: Optional[Tensor] = None,
) -> Tensor:
    """
    PyTorch reference implementation of QKV projection.
    """
    has_pre_rope = (
        qk_norm_pre_rope_q_norm is not None or qk_norm_pre_rope_k_norm is not None
    )
    has_post_rope = (
        qk_norm_post_rope_q_norm is not None or qk_norm_post_rope_k_norm is not None
    )

    if has_pre_rope or has_post_rope:
        assert num_q_heads is not None and num_kv_heads is not None, (
            "num_q_heads and num_kv_heads must be specified when qk_norm is enabled"
        )

    B, S, H = hidden.shape
    fused_qkv_dim = qkv_weights.shape[1]

    # Step 1: Fused add
    if mlp_prev is not None and attn_prev is not None:
        hidden = hidden + mlp_prev + attn_prev

    # Step 2: Apply normalization
    if hidden_actual is None:
        hidden_actual = H

    if norm_type == NormType.RMS_NORM:
        variance = hidden.pow(2).sum(-1, keepdim=True) / hidden_actual
        hidden = hidden * torch.rsqrt(variance + eps)
        if norm_weights is not None:
            hidden = hidden * norm_weights

    elif norm_type == NormType.LAYER_NORM:
        mean = hidden.mean(-1, keepdim=True)
        variance = hidden.var(-1, keepdim=True, unbiased=False)
        hidden = (hidden - mean) * torch.rsqrt(variance + eps)
        if norm_weights is not None:
            hidden = hidden * norm_weights
        if norm_b is not None:
            hidden = hidden + norm_b

    elif norm_type == NormType.RMS_NORM_SKIP_GAMMA:
        variance = hidden.pow(2).sum(-1, keepdim=True) / hidden_actual
        hidden = hidden * torch.rsqrt(variance + eps)

    # Step 3: Matmul
    output = hidden @ qkv_weights

    # Step 4: Add bias
    if bias is not None:
        output = output + bias

    # Step 4.5: Apply per-head QK-norm (pre-RoPE)
    if has_pre_rope:
        output = _apply_qk_norm(
            output,
            fused_qkv_dim,
            d_head,
            num_q_heads,
            num_kv_heads,
            qk_norm_pre_rope_q_norm,
            qk_norm_pre_rope_k_norm,
            qk_norm_pre_rope_eps,
            qk_norm_pre_rope_gamma_fused_in_rope_caches,
            qk_norm_pre_rope_q_gamma,
            qk_norm_pre_rope_k_gamma,
            qk_norm_pre_rope_q_beta,
            qk_norm_pre_rope_k_beta,
        )

    # Step 5: Apply RoPE if cos/sin caches provided
    fused_rope = (
        cos_cache is not None
        and sin_cache is not None
        and num_q_heads is not None
        and num_kv_heads is not None
    )

    if fused_rope:
        rope_d_head = (
            d_head
            if d_head is not None
            else fused_qkv_dim // (num_q_heads + 2 * num_kv_heads)
        )
        num_rope_heads = num_q_heads + num_kv_heads

        for head_idx in range(num_rope_heads):
            start_idx = head_idx * rope_d_head
            end_idx = start_idx + rope_d_head
            half_d = rope_d_head // 2

            head_output = output[:, :, start_idx:end_idx]
            x1 = head_output[:, :, :half_d]
            x2 = head_output[:, :, half_d:]

            cos = cos_cache
            sin = sin_cache[:, :, :half_d]

            rotated_x1 = x1 * cos[:, :, :half_d] - x2 * sin
            rotated_x2 = x2 * cos[:, :, half_d:] + x1 * sin

            output[:, :, start_idx : start_idx + half_d] = rotated_x1
            output[:, :, start_idx + half_d : end_idx] = rotated_x2

    # Step 5.5: Apply per-head QK-norm (post-RoPE)
    if has_post_rope:
        output = _apply_qk_norm(
            output,
            fused_qkv_dim,
            d_head,
            num_q_heads,
            num_kv_heads,
            qk_norm_post_rope_q_norm,
            qk_norm_post_rope_k_norm,
            qk_norm_post_rope_eps,
            qk_norm_post_rope_gamma_fused_in_rope_caches,
            qk_norm_post_rope_q_gamma,
            qk_norm_post_rope_k_gamma,
            qk_norm_post_rope_q_beta,
            qk_norm_post_rope_k_beta,
        )

    # Step 6: Handle output layout. Here the head count spans Q, K and V.
    return _apply_qkv_output_layout(output, d_head, output_layout)


def _qkv_kernel_refusal(
    hidden: Tensor,
    qkv_weights: Tensor,
    mlp_prev: Optional[Tensor],
    attn_prev: Optional[Tensor],
    d_head: Optional[int],
    output_layout: QKVOutputLayout,
    norm_type: NormType,
    norm_b: Optional[Tensor],
    quantization_type: QuantizationType = QuantizationType.NONE,
    qk_norm_pre_rope_q_beta: Optional[Tensor] = None,
    qk_norm_pre_rope_k_beta: Optional[Tensor] = None,
    qk_norm_post_rope_q_beta: Optional[Tensor] = None,
    qk_norm_post_rope_k_beta: Optional[Tensor] = None,
    fused_rope: bool = False,
) -> str | None:
    """Why the NKI QKV kernel cannot take these arguments, or None when it can.

    The reason is what ``qkv_proj`` reports for its torch fallback (``record_kernel_choice``).
    """
    if not can_run_kernel(hidden):
        return device_reason(hidden)

    # The fused RoPE path needs a d_head that divides the 512-element PSUM bank.
    # qkv_cte evicts PSUM per head and computes
    # ``NUM_HEADS_PER_PSUM_BANK = 512 // d_head``.
    # The division truncates, so a d_head that does not divide 512 leaves a remainder,
    # and each head that crosses a bank boundary reads the wrong PSUM offset and
    # produces incorrect output.
    if fused_rope and (d_head is None or 512 % d_head != 0):
        return "fused RoPE needs a d_head that divides 512"

    # nkilib nki_qkv_kernel does not support qk_norm beta weights yet
    if (
        qk_norm_pre_rope_q_beta is not None
        or qk_norm_pre_rope_k_beta is not None
        or qk_norm_post_rope_q_beta is not None
        or qk_norm_post_rope_k_beta is not None
    ):
        return "qk_norm beta weights are not supported"

    _, _, H = hidden.shape
    fused_qkv_dim = qkv_weights.shape[1]

    if H > 24576:
        return "hidden size above 24576"

    if H % 128 != 0 and quantization_type not in (
        QuantizationType.MX,
        QuantizationType.ROW_MX,
    ):
        return "hidden size not a multiple of 128 (without MX quantization)"

    if fused_qkv_dim > 4096:
        return "fused QKV width above 4096"

    # Residual add: both must be provided or neither
    if (mlp_prev is None) != (attn_prev is None):
        return "mlp_prev and attn_prev must be given together"

    # norm_b only supported for LAYER_NORM
    if norm_b is not None and norm_type != NormType.LAYER_NORM:
        return "norm_b needs LAYER_NORM"

    # NBdS layout is not supported in the kernel
    if output_layout == QKVOutputLayout.NBdS:
        return "output layout NBdS is not supported"

    # For NBSd layout, d_head must be 128
    if output_layout == QKVOutputLayout.NBSd:
        if d_head is None or d_head != 128:
            return "output layout NBSd needs d_head 128"

    return None


def _can_use_qkv_kernel(*args, **kwargs) -> bool:
    """Whether the NKI QKV kernel can take these arguments (:func:`_qkv_kernel_refusal`)."""
    return _qkv_kernel_refusal(*args, **kwargs) is None


def qkv_proj(
    hidden: Tensor,
    qkv_weights: Tensor,
    norm_weights: Optional[Tensor] = None,
    mlp_prev: Optional[Tensor] = None,
    attn_prev: Optional[Tensor] = None,
    d_head: Optional[int] = None,
    output_layout: QKVOutputLayout = QKVOutputLayout.BSD,
    eps: float = 1e-6,
    norm_type: NormType = NormType.NO_NORM,
    use_dma_transpose: bool = True,
    store_output_in_sbuf: bool = False,
    bias: Optional[Tensor] = None,
    norm_b: Optional[Tensor] = None,
    hidden_actual: Optional[int] = None,
    cos_cache: Optional[Tensor] = None,
    sin_cache: Optional[Tensor] = None,
    num_q_heads: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
    quantization_type: QuantizationType = QuantizationType.NONE,
    qkv_w_scale: Optional[Tensor] = None,
    qkv_in_scale: Optional[Tensor] = None,
    weight_layout: QKVWeightLayout = QKVWeightLayout.CONTIGUOUS,
    sbm_lower_bound: int = None,
    sbm_upper_bound: int = None,
    sbm_use_auto_alloc: bool = False,
    sbm_default_stack_alloc: bool = True,
    use_auto_allocation: bool = False,
    qk_norm_pre_rope_q_norm: Optional[NormType] = None,
    qk_norm_pre_rope_k_norm: Optional[NormType] = None,
    qk_norm_pre_rope_eps: float = 1e-6,
    qk_norm_pre_rope_gamma_fused_in_rope_caches: bool = False,
    qk_norm_pre_rope_q_gamma: Optional[Tensor] = None,
    qk_norm_pre_rope_k_gamma: Optional[Tensor] = None,
    qk_norm_pre_rope_q_beta: Optional[Tensor] = None,
    qk_norm_pre_rope_k_beta: Optional[Tensor] = None,
    qk_norm_post_rope_q_norm: Optional[NormType] = None,
    qk_norm_post_rope_k_norm: Optional[NormType] = None,
    qk_norm_post_rope_eps: float = 1e-6,
    qk_norm_post_rope_gamma_fused_in_rope_caches: bool = False,
    qk_norm_post_rope_q_gamma: Optional[Tensor] = None,
    qk_norm_post_rope_k_gamma: Optional[Tensor] = None,
    qk_norm_post_rope_q_beta: Optional[Tensor] = None,
    qk_norm_post_rope_k_beta: Optional[Tensor] = None,
) -> Tensor:
    """
    QKV Projection API that automatically selects between NKI kernel and PyTorch fallback.

    This function checks kernel constraints and dispatches to:
    - nki_qkv_kernel: When all constraints are satisfied and running on Neuron
    - PyTorch implementation: When constraints are violated or not on Neuron

    This kernel implements QKV projection with optional normalization fusion used in
    transformer models: output = norm(hidden + mlp_prev + attn_prev) @ qkv_weights + bias

    Dimensions:
        B: Batch size
        S: Sequence length
        H: Hidden dimension
        fused_qkv_dim: (num_q_heads + 2 * num_kv_heads) * d_head

    Args:
        hidden: Input tensor with shape [B, S, H]
        qkv_weights: Weight tensor with shape [H, fused_qkv_dim]
        norm_weights: Optional normalization weights (gamma) with shape [1, H]
        mlp_prev: Optional previous MLP output for residual with shape [B, S, H]
        attn_prev: Optional previous attention output for residual with shape [B, S, H]
        d_head: Head dimension. Required for each output layout except BSD, and
            for RoPE
        output_layout: Output tensor layout:
            - BSD: [B, S, fused_qkv_dim]
            - NBSd: [num_heads, B, S, d_head]
            - BNSd: [B, num_heads, S, d_head]
            - NBdS: [num_heads, B, d_head, S]
        eps: Epsilon for numerical stability in normalization. Default: 1e-6
        norm_type: Normalization type:
            - NO_NORM: No normalization
            - RMS_NORM: Root Mean Square normalization
            - LAYER_NORM: Layer normalization
            - RMS_NORM_SKIP_GAMMA: RMSNorm with gamma pre-multiplied in weights
        use_dma_transpose: Use DMA transpose for weight loading. Default: True
        store_output_in_sbuf: Store output in SBUF instead of HBM. Default: False
        bias: Optional bias tensor with shape [1, fused_qkv_dim]
        norm_b: Optional LayerNorm bias with shape [1, H]
        hidden_actual: Actual hidden dimension for padded tensors
        cos_cache: Cosine cache for RoPE with shape [B, S, d_head]
        sin_cache: Sine cache for RoPE with shape [B, S, d_head]
        num_q_heads: Number of query heads for RoPE
        num_kv_heads: Number of key/value heads for RoPE
        quantization_type: Type of quantization (NONE, ROW, STATIC). Default: NONE
        qkv_w_scale: QKV weight scale tensor for quantization
        qkv_in_scale: QKV input scale tensor for static quantization
        sbm: Optional SBUF manager for memory allocation control
        use_auto_allocation: Whether to use automatic SBUF allocation. Default: False
        qk_norm_pre_rope_q_norm: Normalization type for Q heads before RoPE
        qk_norm_pre_rope_k_norm: Normalization type for K heads before RoPE
        qk_norm_pre_rope_eps: Epsilon for pre-RoPE QK-norm. Default: 1e-6
        qk_norm_pre_rope_gamma_fused_in_rope_caches: Whether gamma is pre-multiplied
            into cos/sin caches for pre-RoPE norm. Default: False
        qk_norm_pre_rope_q_gamma: Per-head gamma for Q heads applied before RoPE [1, d_head]
        qk_norm_pre_rope_k_gamma: Per-head gamma for K heads applied before RoPE [1, d_head]
        qk_norm_pre_rope_q_beta: Per-head beta for Q heads applied before RoPE [1, d_head]
        qk_norm_pre_rope_k_beta: Per-head beta for K heads applied before RoPE [1, d_head]
        qk_norm_post_rope_q_norm: Normalization type for Q heads after RoPE
        qk_norm_post_rope_k_norm: Normalization type for K heads after RoPE
        qk_norm_post_rope_eps: Epsilon for post-RoPE QK-norm. Default: 1e-6
        qk_norm_post_rope_gamma_fused_in_rope_caches: Whether gamma is pre-multiplied
            into cos/sin caches for post-RoPE norm. Default: False
        qk_norm_post_rope_q_gamma: Per-head gamma for Q heads applied after RoPE [1, d_head]
        qk_norm_post_rope_k_gamma: Per-head gamma for K heads applied after RoPE [1, d_head]
        qk_norm_post_rope_q_beta: Per-head beta for Q heads applied after RoPE [1, d_head]
        qk_norm_post_rope_k_beta: Per-head beta for K heads applied after RoPE [1, d_head]

    Returns:
        Output tensor with shape determined by output_layout.

    Kernel Constraints (falls back to PyTorch if violated):
        - hidden must be 3D tensor with shape [B, S, H]
        - qkv_weights must be 2D tensor with shape [H, fused_qkv_dim]
        - Hidden dimension H must match weight's first dimension
        - Hidden dimension H must be divisible by 128
        - Hidden dimension H must be <= 24576
        - fused_qkv_dim must be <= 4096
        - mlp_prev and attn_prev must both be provided or neither
        - norm_b is only supported for LAYER_NORM
        - For NBSd layout, d_head must be 128
        - NBdS layout is not kernel-supported (falls back to PyTorch)

    Notes:
        - The kernel automatically selects between decode (TKG) and prefill (CTE)
          implementations based on sequence length threshold
        - RoPE fusion requires cos_cache, sin_cache, num_q_heads, and num_kv_heads
    """

    has_pre_rope = (
        qk_norm_pre_rope_q_norm is not None or qk_norm_pre_rope_k_norm is not None
    )
    has_post_rope = (
        qk_norm_post_rope_q_norm is not None or qk_norm_post_rope_k_norm is not None
    )

    if has_pre_rope or has_post_rope:
        assert num_q_heads is not None and num_kv_heads is not None, (
            "num_q_heads and num_kv_heads must be specified when qk_norm is enabled"
        )

    # Check if kernel can be used, and report the decision (SGLANG_NEURON_REQUIRE_NKI_KERNELS)
    refusal = _qkv_kernel_refusal(
        hidden=hidden,
        qkv_weights=qkv_weights,
        mlp_prev=mlp_prev,
        attn_prev=attn_prev,
        d_head=d_head,
        output_layout=output_layout,
        norm_type=norm_type,
        norm_b=norm_b,
        quantization_type=quantization_type,
        qk_norm_pre_rope_q_beta=qk_norm_pre_rope_q_beta,
        qk_norm_pre_rope_k_beta=qk_norm_pre_rope_k_beta,
        qk_norm_post_rope_q_beta=qk_norm_post_rope_q_beta,
        qk_norm_post_rope_k_beta=qk_norm_post_rope_k_beta,
        fused_rope=cos_cache is not None or sin_cache is not None,
    )
    can_use_kernel = record_kernel_choice("qkv", refusal is None, refusal or "eligible")

    if can_use_kernel:
        # Compute derived boolean flags
        fused_residual_add = mlp_prev is not None and attn_prev is not None
        fused_rope = (
            cos_cache is not None
            and sin_cache is not None
            and num_q_heads is not None
            and num_kv_heads is not None
        )

        # QK-norm and RoPE require d_head; infer from weight shape if not provided
        if d_head is None and num_q_heads is not None and num_kv_heads is not None:
            fused_qkv_dim = qkv_weights.shape[1]
            d_head = fused_qkv_dim // (num_q_heads + 2 * num_kv_heads)

        # Wrapped at call time (importing the runtime at module import would start it before the
        # worker claims its NeuronCore), at the configured LNC, which must be 2.
        launch = wrap_nki_kernel(_torch_compatible_qkv_kernel, supported_lnc=(2,))

        return launch(
            hidden=hidden,
            qkv_weights=qkv_weights,
            norm_weights=norm_weights,
            mlp_prev=mlp_prev,
            attn_prev=attn_prev,
            d_head=d_head,
            output_layout=output_layout,
            eps=eps,
            norm_type=norm_type,
            use_dma_transpose=use_dma_transpose,
            store_output_in_sbuf=store_output_in_sbuf,
            bias=bias,
            norm_b=norm_b,
            hidden_actual=hidden_actual,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            quantization_type=quantization_type,
            qkv_w_scale=qkv_w_scale,
            qkv_in_scale=qkv_in_scale,
            weight_layout=weight_layout,
            fused_residual_add=fused_residual_add,
            fused_rope=fused_rope,
            qk_norm_pre_rope_q_norm=qk_norm_pre_rope_q_norm,
            qk_norm_pre_rope_k_norm=qk_norm_pre_rope_k_norm,
            qk_norm_pre_rope_eps=qk_norm_pre_rope_eps,
            qk_norm_pre_rope_gamma_fused_in_rope_caches=qk_norm_pre_rope_gamma_fused_in_rope_caches,
            qk_norm_pre_rope_q_gamma=qk_norm_pre_rope_q_gamma,
            qk_norm_pre_rope_k_gamma=qk_norm_pre_rope_k_gamma,
            qk_norm_pre_rope_q_beta=qk_norm_pre_rope_q_beta,
            qk_norm_pre_rope_k_beta=qk_norm_pre_rope_k_beta,
            qk_norm_post_rope_q_norm=qk_norm_post_rope_q_norm,
            qk_norm_post_rope_k_norm=qk_norm_post_rope_k_norm,
            qk_norm_post_rope_eps=qk_norm_post_rope_eps,
            qk_norm_post_rope_gamma_fused_in_rope_caches=qk_norm_post_rope_gamma_fused_in_rope_caches,
            qk_norm_post_rope_q_gamma=qk_norm_post_rope_q_gamma,
            qk_norm_post_rope_k_gamma=qk_norm_post_rope_k_gamma,
            qk_norm_post_rope_q_beta=qk_norm_post_rope_q_beta,
            qk_norm_post_rope_k_beta=qk_norm_post_rope_k_beta,
            sbm_lower_bound=sbm_lower_bound,
            sbm_upper_bound=sbm_upper_bound,
            sbm_use_auto_alloc=sbm_use_auto_alloc,
            sbm_default_stack_alloc=sbm_default_stack_alloc,
            use_auto_allocation=use_auto_allocation,
        )
    else:
        if quantization_type != QuantizationType.NONE:
            raise NotImplementedError(
                "QKV Projection torch fallback currently does not support quantization"
            )

        return _torch_qkv_impl(
            hidden=hidden,
            qkv_weights=qkv_weights,
            norm_weights=norm_weights,
            mlp_prev=mlp_prev,
            attn_prev=attn_prev,
            d_head=d_head,
            output_layout=output_layout,
            eps=eps,
            norm_type=norm_type,
            bias=bias,
            norm_b=norm_b,
            hidden_actual=hidden_actual,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            qk_norm_pre_rope_q_norm=qk_norm_pre_rope_q_norm,
            qk_norm_pre_rope_k_norm=qk_norm_pre_rope_k_norm,
            qk_norm_pre_rope_eps=qk_norm_pre_rope_eps,
            qk_norm_pre_rope_gamma_fused_in_rope_caches=qk_norm_pre_rope_gamma_fused_in_rope_caches,
            qk_norm_pre_rope_q_gamma=qk_norm_pre_rope_q_gamma,
            qk_norm_pre_rope_k_gamma=qk_norm_pre_rope_k_gamma,
            qk_norm_pre_rope_q_beta=qk_norm_pre_rope_q_beta,
            qk_norm_pre_rope_k_beta=qk_norm_pre_rope_k_beta,
            qk_norm_post_rope_q_norm=qk_norm_post_rope_q_norm,
            qk_norm_post_rope_k_norm=qk_norm_post_rope_k_norm,
            qk_norm_post_rope_eps=qk_norm_post_rope_eps,
            qk_norm_post_rope_gamma_fused_in_rope_caches=qk_norm_post_rope_gamma_fused_in_rope_caches,
            qk_norm_post_rope_q_gamma=qk_norm_post_rope_q_gamma,
            qk_norm_post_rope_k_gamma=qk_norm_post_rope_k_gamma,
            qk_norm_post_rope_q_beta=qk_norm_post_rope_q_beta,
            qk_norm_post_rope_k_beta=qk_norm_post_rope_k_beta,
        )
