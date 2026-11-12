# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from sglang_neuron.backend import envs


def can_run_kernel(device: torch.Tensor | str = "") -> bool:
    """Whether NKI kernels can run on the given device: never in CPU mode (no Neuron device; the ops take their torch
    paths), else on any device but the host."""
    if envs.SGLANG_NEURON_CPU_MODE:
        return False
    device_str = str(device.device) if isinstance(device, torch.Tensor) else device
    return device_str != "cpu"
