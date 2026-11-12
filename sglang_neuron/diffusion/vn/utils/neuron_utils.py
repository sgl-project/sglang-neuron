# SPDX-License-Identifier: Apache-2.0
"""The NKI kernel gate ``can_run_kernel``."""

import torch

from sglang_neuron.diffusion import envs
from sglang_neuron.utils import neuron_utils


def can_run_kernel(device: torch.Tensor | str = "") -> bool:
    """Whether NKI kernels can run on the given device: the plugin's shared check
    (``sglang_neuron.utils.neuron_utils.can_run_kernel``: not in CPU mode, not on the host), and
    the diffusion kill switch, ``SGLANG_NEURON_DISABLE_NKI_KERNELS``, off."""
    if envs.SGLANG_NEURON_DISABLE_NKI_KERNELS:
        return False
    return neuron_utils.can_run_kernel(device)
