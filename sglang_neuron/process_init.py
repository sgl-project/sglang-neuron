"""Per-process setup for a Neuron launch, run before SGLang builds its process groups."""

from __future__ import annotations

import functools
import logging
import os

logger = logging.getLogger(__name__)

_STARTED = False


def init_stack() -> None:
    """Set up this process for the runtime, once, before SGLang builds its process groups. This runs torch-neuronx's
    check and sets up the executor binding cache (backend.init), installs the shared gloo transport device, and makes
    the host SGLang's device module.

    Process groups keep the names torch gives them. torch names a group by its place in the creation order, which
    every rank follows, and builds the group's rendezvous under that name, so all members of a group hold the same
    name; torch-neuronx's collective legalization replaces the name of a traced collective's group with the group's
    registered partition before the graph is lowered, so the name does not reach the compiled program."""
    global _STARTED
    if _STARTED:
        return
    import torch

    from sglang_neuron.backend import envs as backend_envs
    from sglang_neuron.backend import stack, torch_pins
    from sglang_neuron.backend.init import init_backend
    from sglang_neuron.distributed import gloo_device

    init_backend()
    if os.environ.get("SGLANG_NEURON_DEBUG_MESH") == "1":
        for name in stack.mesh_logger_names():
            logging.getLogger(name).setLevel(logging.DEBUG)

    # SGLang's device is the host, so the no-argument torch.get_device_module() must be torch.cpu, not torch-neuronx's
    torch_pins.require("torch.get_device_module", torch.get_device_module, __name__)
    original_get_device_module = torch.get_device_module

    @functools.wraps(original_get_device_module)
    def get_device_module(device=None):
        return torch.cpu if device is None else original_get_device_module(device)

    # dynamo keys a special handler on torch.get_device_module.__wrapped__
    get_device_module.__wrapped__ = getattr(original_get_device_module, "__wrapped__", original_get_device_module)
    torch.get_device_module = get_device_module

    if backend_envs.SGLANG_NEURON_CPU_MODE:
        # torch's c10d helpers ask for the current accelerator; a CPU-mode process loads no torch-neuronx to answer
        torch._C._get_accelerator = lambda *a, **k: torch.device("cpu")  # type: ignore[attr-defined]

    gloo_device.install()
    _STARTED = True
