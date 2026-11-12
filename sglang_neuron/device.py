# SPDX-License-Identifier: Apache-2.0
"""The runtime's device: SGLang's batch tensors, samplers and caches live on the host, and only the plugin's
runtime touches the rank's NeuronCore (runtime_init.neuron_device). Process groups are gloo; the device collectives
are lowered inside the compiled graphs."""

from __future__ import annotations

import json
import subprocess
import sys
from functools import lru_cache

from sglang.srt.platforms.device_mixin import DeviceMixin, PlatformEnum


@lru_cache(maxsize=1)
def neuron_ls() -> list:
    """The driver's device list (neuron-ls), read without starting the Neuron runtime."""
    return json.loads(subprocess.run(["neuron-ls", "--json-output"], check=True, capture_output=True,
                                     text=True).stdout)


class NeuronDeviceMixin(DeviceMixin):
    _enum = PlatformEnum.OOT
    device_name = "neuron"
    device_type = "cpu"

    def get_device_total_memory(self, device_id: int = 0) -> int:
        """One logical core's HBM bank. neuron-ls reports the device's memory and its logical core count; the
        physical cores of a logical core share one bank, and an allocation past it fails."""
        devices = neuron_ls()
        d = devices[int(device_id) % len(devices)]
        return int(d["memory_size"]) // max(1, int(d["nc_count"]))

    def get_current_memory_usage(self, device=None) -> float:
        """Bytes this process holds on its NeuronCore, after the caching allocator releases its free segments; 0 before
        the runtime starts. SGLang sizes the KV pool from this reading, so a query that fails refuses the launch."""
        from sglang_neuron.backend import device_io

        tnx = sys.modules.get("torch_neuronx")
        if tnx is not None and tnx.is_neuron_runtime_initialized():
            tnx.empty_cache()
        try:
            used = device_io.vnc_bytes_used()
        except RuntimeError as e:
            raise RuntimeError(f"the device memory in use cannot be read ({e}), so the KV pool cannot be sized") from e
        return float(used or 0)

    def is_pin_memory_available(self, device=None) -> bool:
        return False

    def get_device(self, device_id=0) -> str:
        return "cpu"

    def set_device(self, device) -> None:
        return None

    def get_device_name(self, device_id: int = 0) -> str:
        return neuron_ls()[0].get("instance_type", "neuron")

    def get_torch_distributed_backend_str(self) -> str:
        return "gloo"

    def synchronize(self) -> None:
        from sglang_neuron.backend import envs
        from sglang_neuron.model_executor import runtime_init

        if runtime_init._DONE and not envs.SGLANG_NEURON_CPU_MODE:
            from sglang_neuron.model_executor.async_execution import synchronize_pending

            synchronize_pending()

    def empty_cache(self) -> None:
        return None

    def get_torch_profiler_activity_str(self) -> str:
        # /start_profile asks for it and the base raises; device timelines come from the Neuron runtime's profiler
        return "CPU"

    def get_torch_profiler_activity(self):
        import torch

        return torch.profiler.ProfilerActivity.CPU
