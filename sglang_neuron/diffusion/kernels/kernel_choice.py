# SPDX-License-Identifier: Apache-2.0
"""Log each NKI-kernel-versus-torch-fallback decision, and optionally forbid the fallback.

These kernel gates return through :func:`record_kernel_choice`:

- Shared DiT kernels: adaln, attention, o_proj, qkv.
- Wan2.2 DiT: ring_attention (cp_attention), wan_bf16_qkv, wan_mlp (feed_forward).
- FLUX DiT: flux_gather_attention, flux_gather_kv_attention, flux_ring_attention.
- LingBot-World DiT: lingbot_cache_attention and fa2_paged_cache_attention (paged_attention),
  lingbot_residual_norm.
- MiniMax-H3 DiT: minimax_h3_ring_attention.
- FLUX and Wan VAE: flux_group_norm (FLUX), vae_attention (Wan), vae_conv2d and vae_conv3d (Wan;
  static gates, checked when the filters are packed).

Dynamo folds the call to a constant (``assume_constant_result``), so it runs at trace time, off
the graph. This also works under ``fullgraph=True``, where a bare ``logger.info`` is a hard
failure.

Each new decision gives one greppable line::

    [nki_kernel] attention path=torch reason=can_run_kernel=False (device=cpu)

Set ``SGLANG_NEURON_REQUIRE_NKI_KERNELS=1`` to change a fallback in these gates into
:class:`NkiKernelRequiredError`. A test sets it to prove that the model runs these NKI kernels and
does not silently take a torch fallback. A fallback that the gate marks ``by_design`` is logged,
but it does not raise.
"""

from __future__ import annotations

import logging

import torch

from sglang_neuron.diffusion import envs

logger = logging.getLogger(__name__)

# The decisions already logged, so that each decision is logged only once.
_LOGGED: set[tuple[str, bool, str]] = set()


class NkiKernelRequiredError(RuntimeError):
    """A kernel took its torch fallback while ``SGLANG_NEURON_REQUIRE_NKI_KERNELS`` is set."""


@torch._dynamo.assume_constant_result
def record_kernel_choice(name: str, use_kernel: bool, reason: str, by_design: bool = False) -> bool:
    """Log the decision of one kernel gate and return ``use_kernel``.

    Pass Python bools and a constant str. A symbolic size or bool is not a constant, and it
    fails the trace.

    Args:
        name: Stable kernel name, for example ``"flux_ring_attention"``.
        use_kernel: True if the gate selects the NKI kernel, False for the torch fallback.
        reason: Why the gate decided this. For a fallback, give the failed condition.
        by_design: True if the torch path is the measured choice for this shape, for example a
            VAE conv with too few channels. The fallback is logged, but it never raises.

    Raises:
        NkiKernelRequiredError: ``use_kernel`` and ``by_design`` are False, and
            ``SGLANG_NEURON_REQUIRE_NKI_KERNELS`` is set.
    """
    decision = (name, use_kernel, reason)
    if decision not in _LOGGED:
        _LOGGED.add(decision)
        logger.info(
            "[nki_kernel] %s path=%s reason=%s%s",
            name,
            "nki" if use_kernel else "torch",
            reason,
            " (by design)" if by_design else "",
        )
    if not use_kernel and not by_design and envs.SGLANG_NEURON_REQUIRE_NKI_KERNELS:
        raise NkiKernelRequiredError(
            f"[nki_kernel] {name} took the torch fallback: {reason}. "
            "SGLANG_NEURON_REQUIRE_NKI_KERNELS is set."
        )
    return use_kernel


def device_reason(device: torch.Tensor | str) -> str:
    """Fallback reason for a tensor or device string that ``can_run_kernel`` refused.

    Checks the same conditions as ``vn/utils/neuron_utils.can_run_kernel``, in the same order:
    NKI kernels disabled, CPU mode, then the CPU device. Each condition is a trace-time constant.
    """
    from sglang_neuron.backend import envs as backend_envs

    if envs.SGLANG_NEURON_DISABLE_NKI_KERNELS:
        return "can_run_kernel=False (SGLANG_NEURON_DISABLE_NKI_KERNELS is set)"
    if backend_envs.SGLANG_NEURON_CPU_MODE:
        return "can_run_kernel=False (SGLANG_NEURON_CPU_MODE is set)"
    device_type = device.device.type if isinstance(device, torch.Tensor) else device
    return f"can_run_kernel=False (device={device_type})"
