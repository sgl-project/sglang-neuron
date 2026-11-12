# SPDX-License-Identifier: Apache-2.0
"""Fused residual add + RMSNorm for prefill on the nkilib kernel, with a torch fallback.

The kernel adds the residual in the DMA engines as it loads and also returns the pre-norm
sum, which the caller reuses as the next residual instead of adding again.
"""

from typing import Optional, Tuple

import torch

from nkilib.experimental.subkernels.rmsnorm_residual_prefill import (
    rmsnorm_residual_prefill as _rmsnorm_residual_prefill_kernel,
)

from sglang_neuron.diffusion.neuron_api import nki_wrap_function
from sglang_neuron.diffusion.vn.utils.hardware_config import get_logical_nc_config
from sglang_neuron.diffusion.vn.utils.neuron_utils import can_run_kernel


def wrap_nki(kernel):
    """The runtime's ``wrap_nki``, resolved at call time: importing a runtime at module import
    would start it before the worker claims its NeuronCore."""
    return nki_wrap_function()(kernel)


def _can_use_kernel(hidden_states: torch.Tensor) -> bool:
    """Whether the kernel can run: it tiles H and shards T itself, so only the device and a
    ``[T, H]`` or ``[B, S, H]`` input gate it."""
    if not can_run_kernel(hidden_states):
        return False

    return hidden_states.dim() in (2, 3)


def rmsnorm_residual_prefill(
    hidden_states: torch.Tensor,
    gamma: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    unpadded_hidden_size: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """``(rmsnorm(hidden_states + residual), hidden_states + residual)``.

    Args:
        hidden_states: ``[T, H]`` or ``[B, S, H]``.
        gamma: ``[H]`` or ``[1, H]``, in the same H order as ``hidden_states``.
        residual: Shaped like ``hidden_states``. None is a plain RMSNorm, and the second
            output is None.
        unpadded_hidden_size: True width of a padded H, the mean's divisor. Defaults to H.
    """
    # The kernel asserts gamma is [1, H]. A norm module holds it as [H].
    if gamma.ndim == 1:
        gamma = gamma.unsqueeze(0)

    # Check the kernel's contract on both paths, so the CPU fallback fails like the device.
    hidden_size = hidden_states.shape[-1]
    if residual is not None:
        assert residual.shape == hidden_states.shape, (
            f"residual {tuple(residual.shape)} must match "
            f"hidden {tuple(hidden_states.shape)}"
        )
    assert tuple(gamma.shape) == (1, hidden_size), (
        f"gamma {tuple(gamma.shape)} must be [{hidden_size}] or [1, {hidden_size}]"
    )
    assert unpadded_hidden_size is None or 0 < unpadded_hidden_size <= hidden_size, (
        f"unpadded_hidden_size {unpadded_hidden_size} must be in (0, {hidden_size}]"
    )

    if not _can_use_kernel(hidden_states):
        return _torch_impl(hidden_states, gamma, residual, eps, unpadded_hidden_size)

    # [lnc] is the launch grid, which must match the compiled LNC:
    # NEURON_ENFORCE_NKI_LNC_CONSISTENCY rejects a mismatch.
    outs = wrap_nki(_rmsnorm_residual_prefill_kernel)[get_logical_nc_config()](
        hidden=hidden_states,
        gamma=gamma,
        residual=residual,
        eps=eps,
        hidden_actual=unpadded_hidden_size,
    )

    if residual is not None:
        out, output_residual = outs
        return out.view_as(hidden_states), output_residual.view_as(hidden_states)
    out = outs[0] if isinstance(outs, (list, tuple)) else outs
    return out.view_as(hidden_states), None


def _torch_impl(
    hidden_states: torch.Tensor,
    gamma: torch.Tensor,
    residual: Optional[torch.Tensor],
    eps: float,
    unpadded_hidden_size: Optional[int],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Torch reference of the kernel: add, then RMSNorm over the full padded width.

    As in the kernel, the variance sums every column and divides by the unpadded width. A
    zero pad adds nothing wherever it sits in H, so the sum is not narrowed to a leading slice.
    """
    x = hidden_states if residual is None else hidden_states + residual
    pre_norm_sum = None if residual is None else x

    n = unpadded_hidden_size or x.shape[-1]
    x_fp32 = x.to(torch.float32)
    variance = x_fp32.pow(2).mean(-1, keepdim=True) * (x.shape[-1] / n)
    out = x_fp32 * torch.rsqrt(variance + eps) * gamma.to(torch.float32)

    return out.to(hidden_states.dtype), pre_norm_sum
