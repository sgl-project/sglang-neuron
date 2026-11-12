# SPDX-License-Identifier: Apache-2.0
"""Initialize torch-neuronx (or, in CPU mode, a stand-in neuron device) and the PyTorch device for SGLang."""

from __future__ import annotations

import logging
import os
import sys
import warnings

logger = logging.getLogger(__name__)

_initialized = False


def _init_cpu_mode_device() -> None:
    """CPU-only process: a minimal privateuse1 "neuron" device module, so dynamo's SymbolicStreamState does not fall
    through to CUDA. No Neuron runtime is loaded."""
    import types

    import torch

    try:
        torch.utils.rename_privateuse1_backend("neuron")
    except RuntimeError as e:
        # the stack already claimed PrivateUse1 as "neuron" at import
        if "already" not in str(e):
            raise
    neuron_module = types.ModuleType("torch.neuron")
    neuron_module.is_available = lambda: False
    neuron_module.current_device = lambda: 0
    neuron_module.current_stream = lambda device=None: type(
        "_S", (), {"device": torch.device("cpu")}
    )()
    neuron_module.set_stream = lambda stream: None
    sys.modules["torch.neuron"] = neuron_module
    torch.neuron = neuron_module

    # one generation registers every method at once, Tensor.is_neuron first: a process that already has it (an earlier
    # init of the stand-in) has them all, and torch refuses a second registration of any
    if not hasattr(torch.Tensor, "is_neuron"):
        torch.utils.generate_methods_for_privateuse1_backend(for_tensor=True, for_module=True, for_storage=True)


def _init_torchneuronx_backend() -> None:
    """Set up torch-neuronx for serving, before its runtime starts.

    In order: refuse a compiler environment from the launch and keep NKI's trace cache off (CPU mode too); reserve the
    compiler variables the compile scopes replace; turn off the vocab-sharding pass; check that torch_neuronx is the
    native backend with its "neuron" dynamo backend registered; then refuse a compile cache built from other NKI
    kernel sources or another kernel library (backend.compile_cache, which stamps an empty one) and install the
    executor binding cache and the submit's upload wait, all before the first compile. torch-neuronx registers its
    device, dynamo backend, torch.neuron and replica-mesh registry itself when torch imports it. A CPU-mode process
    only installs the stand-in device."""
    from sglang_neuron.backend import envs, stack

    stack.refuse_launch_compiler_env()
    stack.disable_nki_trace_cache()
    if envs.SGLANG_NEURON_CPU_MODE:
        # CPU tests run without the Neuron runtime (TORCH_DEVICE_BACKEND_AUTOLOAD=0 keeps torch from loading it)
        _init_cpu_mode_device()
        return

    # before the runtime starts: from here on the compile scopes only replace these values (reserve_compiler_env)
    stack.reserve_compiler_env()
    stack.disable_vocab_sharding_pass()
    import torch
    import torch_neuronx

    if not hasattr(torch_neuronx, "nki_hop"):
        raise RuntimeError(f"torch_neuronx at {torch_neuronx.__file__} is not torch-neuronx's native backend (no "
                           "torch_neuronx.nki_hop)")
    import torch._dynamo

    if "neuron" not in torch._dynamo.list_backends(exclude_tags=()):
        raise RuntimeError("torch-neuronx did not register its 'neuron' dynamo backend")
    from sglang_neuron.backend import compile_cache, device_io, executor_cache

    compile_cache.check_compile_caches()
    executor_cache.install()
    # every graph submit waits for its thread's uploads, whichever call site launches the graph
    device_io.install()


def init_backend() -> None:
    """Initialize the Neuron dynamo backend. Idempotent."""
    global _initialized
    if _initialized:
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Suppress PyTorch kernel override warnings
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=".*Overriding a previously registered kernel.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=".*other operators may also be overridden.*",
    )

    _init_torchneuronx_backend()

    # the port-hold patch (idempotent)
    from sglang_neuron.backend.port_hold import apply_port_hold_patch

    apply_port_hold_patch()

    _initialized = True
