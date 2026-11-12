# SPDX-License-Identifier: Apache-2.0
from typing import Optional
from torch import Tensor

from nkilib.core.output_projection.output_projection_cte import (
    output_projection_cte as output_projection,
)
from nkilib.core.utils.common_types import (
    DtypeMode,
    OProjAttentionLayout,
    QuantizationType,
)

from sglang_neuron.diffusion.neuron_api import nki_wrap_function
from sglang_neuron.diffusion.vn.utils.hardware_config import get_logical_nc_config
from sglang_neuron.diffusion.vn.utils.neuron_utils import can_run_kernel


def wrap_nki(kernel):
    """The runtime's ``wrap_nki``, resolved at call time: importing a runtime at module import
    would start it before the worker claims its NeuronCore."""
    return nki_wrap_function()(kernel)


P_MAX = 128
MAX_VALIDATED_H_SIZE = 16384 + 4321
MAX_VALIDATED_B_TIMES_S_SIZE = 128 * 1024
MAX_VALIDATED_N_SIZE = 17


def _can_use_kernel(
    active: Tensor,
    weight: Tensor,
    quantization_type: QuantizationType = QuantizationType.NONE,
    attention_layout: Optional[OProjAttentionLayout] = None,
) -> bool:
    """Whether the output_projection_cte kernel supports these shapes on this device."""
    lnc_size = 2

    if not can_run_kernel(active):
        return False

    if active.dim() != 4:
        return False

    B, N, D, S = _unpack_attention_shape(
        active,
        quantization_type,
        attention_layout,
    )
    ND_weight, H = weight.shape

    if quantization_type == QuantizationType.MX:
        # MX packs 4 weights per uint32: [ND // 4, H].
        if N * D != ND_weight * 4:
            return False
    else:
        if N * D != ND_weight:
            return False

    if H > MAX_VALIDATED_H_SIZE:
        return False

    if B * S > MAX_VALIDATED_B_TIMES_S_SIZE:
        return False

    if N > MAX_VALIDATED_N_SIZE:
        return False

    if D > P_MAX:
        return False

    if H % lnc_size != 0:
        return False

    return True


def _unpack_attention_shape(
    active: Tensor,
    quantization_type: QuantizationType,
    attention_layout: Optional[OProjAttentionLayout],
) -> tuple[int, int, int, int]:
    """``(B, N, D, S)`` of a 4D ``active``; a None layout is the quantization type's."""
    if attention_layout is None:
        attention_layout = (
            OProjAttentionLayout.BSNd
            if quantization_type == QuantizationType.ROW
            else OProjAttentionLayout.BNdS
        )

    if attention_layout == OProjAttentionLayout.BSNd:
        B, S, N, D = active.shape
    elif attention_layout == OProjAttentionLayout.BNSd:
        B, N, S, D = active.shape
    elif attention_layout == OProjAttentionLayout.BNdS:
        B, N, D, S = active.shape
    else:
        raise ValueError(
            f"Attention layout {attention_layout} is not supported. "
            "Use BNdS, BSNd, or BNSd."
        )

    return B, N, D, S


def _torch_o_proj_impl(
    active: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    quantization_type: QuantizationType = QuantizationType.NONE,
    input_scales: Optional[Tensor] = None,
    weight_scales: Optional[Tensor] = None,
    attention_layout: Optional[OProjAttentionLayout] = None,
) -> Tensor:
    """Torch reference of :func:`o_proj`, unquantized only."""
    assert quantization_type == QuantizationType.NONE, (
        f"Quantization is not supported in the PyTorch fallback path. "
        f"Got quantization_type={quantization_type}"
    )
    assert input_scales is None, (
        "input_scales must be None for the PyTorch fallback path — "
        "quantization is not supported."
    )
    assert weight_scales is None, (
        "weight_scales must be None for the PyTorch fallback path — "
        "quantization is not supported."
    )

    if active.dim() == 3:
        # [B, S, N*D] @ [N*D, H] -> [B, S, H]
        output = active @ weight
    elif active.dim() == 4:
        B, N, D, S = _unpack_attention_shape(
            active,
            quantization_type,
            attention_layout,
        )
        if attention_layout in (None, OProjAttentionLayout.BNdS):
            x = active.permute(0, 3, 1, 2).reshape(B, S, N * D)
        elif attention_layout == OProjAttentionLayout.BSNd:
            x = active.reshape(B, S, N * D)
        elif attention_layout == OProjAttentionLayout.BNSd:
            x = active.permute(0, 2, 1, 3).reshape(B, S, N * D)
        else:
            raise ValueError(f"Attention layout {attention_layout} is not supported.")
        output = x @ weight
    else:
        raise ValueError(f"Expected 3D or 4D input, got {active.dim()}D")

    if bias is not None:
        output = output + bias

    return output


def o_proj(
    active: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    quantization_type: QuantizationType = QuantizationType.NONE,
    input_scales: Optional[Tensor] = None,
    weight_scales: Optional[Tensor] = None,
    attention_layout: Optional[OProjAttentionLayout] = None,
) -> Tensor:
    """``active @ weight + bias`` on the nkilib output_projection_cte kernel, else torch.

    Args:
        active: ``[B, N, D, S]`` (BNdS, the unquantized default), ``[B, S, N, D]`` (BSNd),
            ``[B, N, S, D]`` (BNSd), or ``[B, S, N*D]``.
        weight: ``[N*D, H]``; quantized types take the kernel's packed layouts. The kernel
            shards ``H`` over the LNC cores.
        bias: ``[1, H]``.
        quantization_type: Kernel path only; the torch fallback asserts NONE.
        input_scales: fp32 ``(128, 1)`` per-tensor scale for a quantized type.
        weight_scales: fp32 ``(128, 1)`` per-tensor dequant scale for a quantized type.
        attention_layout: Layout of a 4D ``active``. None is the quantization type's.

    Returns:
        ``[B, S, H]``.
    """
    # The kernel takes 4D: view [B, S, N*D] as BNdS with any D <= 128 that divides N*D,
    # since the projection contracts N and D together.
    if active.dim() == 3:
        B, S_len, ND = active.shape
        D = min(128, ND)
        while ND % D != 0 and D > 1:
            D -= 1
        N = ND // D
        active = active.transpose(1, 2).reshape(B, N, D, S_len)
        attention_layout = OProjAttentionLayout.BNdS

    if _can_use_kernel(active, weight, quantization_type, attention_layout):
        wrapped_output_projection = wrap_nki(output_projection)

        return wrapped_output_projection[get_logical_nc_config()](
            attention=active,
            weight=weight,
            bias=bias,
            quantization_type=quantization_type,
            input_scales=input_scales,
            weight_scales=weight_scales,
            output_dtype=None,
            dtype_mode=DtypeMode.NON_OCP,
            compact_weight_scales=False,
            attention_layout=attention_layout,
        )

    return _torch_o_proj_impl(
        active,
        weight,
        bias=bias,
        quantization_type=quantization_type,
        input_scales=input_scales,
        weight_scales=weight_scales,
        attention_layout=attention_layout,
    )
