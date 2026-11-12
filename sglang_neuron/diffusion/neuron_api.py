# SPDX-License-Identifier: Apache-2.0
"""The torch-neuronx APIs the Neuron diffusion platform uses.

Some of them are private (``torch_neuronx._C._get_memory_stats``, the mesh registry's
``_MESH_REGISTRY``) or live in internal packages (``neuron_dynamo_backend.backend``). Keep every
such import in this module, so the rest of the plugin does not depend on torch-neuronx's internal
package layout. Importing this module does not import torch_neuronx: each function imports it when
it is called, so the SGLang launcher can import the plugin without loading the Neuron runtime.
"""

import os
from importlib import import_module

from sglang_neuron.utils.hardware_config import get_logical_nc_config

_TORCH_NEURONX_PACKAGE_MODULE = "torch_neuronx"
_TORCH_NEURONX_MESH_REGISTRY_MODULE = f"{_TORCH_NEURONX_PACKAGE_MODULE}.distributed.mesh_registry"


def keep_runtime_out_of_host_processes() -> None:
    """Keep torch_neuronx out of the launcher and of worker start-up.

    Sets the variable as a default; the platform calls it before the workers spawn, so they inherit
    it. The launcher itself imports torch before any plugin code runs, so it must start with it in
    its environment (docs/model-recipes/flux1-dev.md, The runtime path): ``import torch`` imports
    torch_neuronx on its own (torch-neuronx registers a ``torch.backends`` autoload entry point;
    ``TORCH_DEVICE_BACKEND_AUTOLOAD=0`` turns autoload off). Once torch_neuronx is imported, the
    Neuron device is torch's current accelerator, and SGLang Diffusion queries its properties at
    import (Triton autotune configs in ``sglang/kernels/ops/diffusion/norm/norm_triton.py``), which
    starts the runtime on every visible NeuronCore. A launcher that does that holds all the cores.
    """
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")


def runtime_started(torch_neuronx) -> bool:
    """Whether torch-neuronx's runtime has started in this process.

    torch-neuronx 2.14.3 has two signals and sets them on different paths: the accelerator
    query that SGLang Diffusion runs at import sets ``_initialized``; a first device tensor
    starts the runtime with ``is_neuron_runtime_initialized()`` true and ``_initialized`` still
    false. Either one counts.
    """
    if getattr(torch_neuronx, "_initialized", False):
        return True
    started = getattr(torch_neuronx, "is_neuron_runtime_initialized", None)
    return bool(started()) if callable(started) else False


def nki_wrap_function():
    """The LLM path's ``backend.stack.wrap_nki`` (torch-neuronx's ``wrap_nki`` over an
    ``@nki.jit`` kernel): it returns an NKI HOP caller indexed by LNC. Resolved at call time:
    importing torch_neuronx at module import would start the runtime before the worker claims
    its NeuronCore."""
    from sglang_neuron.backend import stack

    return stack.wrap_nki


def coerce_lnc(lnc: int, *, supported_lnc: tuple[int, ...]) -> int:
    """Validate the configured LNC against the caller's supported launch degrees."""
    if lnc not in supported_lnc:
        raise ValueError(
            f"NEURON_LOGICAL_NC_CONFIG={lnc!r} is not supported; valid: {supported_lnc}."
        )
    return lnc


def wrap_nki_kernel(kernel, *, supported_lnc: tuple[int, ...]):
    """Return a traceable NKI HOP with the configured, caller-supported LNC.

    ``NEURON_LOGICAL_NC_CONFIG`` (:func:`get_logical_nc_config`: 1 or 2, unset is 2)
    selects the launch degree to match the compiler and runtime. A degree the caller
    does not support raises rather than selecting another degree.
    """
    lnc = coerce_lnc(get_logical_nc_config(), supported_lnc=supported_lnc)
    return nki_wrap_function()(kernel)[lnc]


def wrap_nki_kernel_if_supported(kernel, *, supported_lnc: tuple[int, ...]):
    """:func:`wrap_nki_kernel` at a supported configured LNC, else ``None`` (the kernel's gate then
    takes its torch path): an import-time launcher of an LNC 2 kernel in a module an LNC 1 model
    also imports."""
    if get_logical_nc_config() not in supported_lnc:
        return None
    return wrap_nki_kernel(kernel, supported_lnc=supported_lnc)


def initialize() -> None:
    """Import torch_neuronx, which registers the ``neuron`` device type and ``torch.neuron``."""
    import_module(_TORCH_NEURONX_PACKAGE_MODULE)


# torch-neuronx's ShardIndexInjection FX pass (neuron_dynamo_backend/fx/passes/
# vocab_sharding_spmd.py) looks for the per-rank vocabulary bounds of a vocab-sharded embedding,
# adds a ``vocab_shard_index`` graph input and rewrites the bounds in terms of it, so that every
# TP rank compiles one graph. This variable set to "1" turns the pass off. The plugin's only
# vocab-sharded module, the T5 encoder's VocabDimShardedEmbedding2D, already takes its TP rank as
# a graph input (NeuronT5Encoder.forward), so its graph is the same on every rank without the
# pass; the FLUX runs on torch-neuronx were validated with the pass off.
# The pass also makes each image slower to generate when it is on, because the injected index is
# a per-call device input that is copied between graphs.
# The plugin's LLM path keeps the pass off for the same reason, through the same call.
#
# Turned off at import rather than in initialize(): it must land before the first compile, and
# every process that compiles imports this module first (the compile backend does), the
# launcher too (the platform). A launch environment that sets another value is refused.
def _disable_vocab_sharding_pass() -> None:
    from sglang_neuron.backend import stack

    stack.disable_vocab_sharding_pass()


_disable_vocab_sharding_pass()


def device_count() -> int:
    """Return the visible NeuronCore count (imports torch_neuronx)."""
    import torch

    import_module(_TORCH_NEURONX_PACKAGE_MODULE)
    return torch.neuron.device_count()


def register_process_group_replica_groups(
    group_name: str,
    groups: list[list[int]],
) -> None:
    """Register replica groups in torch-neuronx's (private) mesh registry.

    torch-neuronx's collective legalization reads ``_MESH_REGISTRY[group_name]`` to turn a
    traced collective on one rank's group into the full SPMD replica groups. torch-neuronx fills
    the registry for ``DeviceMesh`` groups and (2.14.3) for ``dist.new_group`` groups too,
    with each group's ranks sorted ascending. The plugin writes its partitions in their declared
    order into the private dict, replacing that entry: SGLang's ``rank_in_group`` follows the
    declared order, and the trn2 CP groups are not ascending (see
    ``parallel_state.register_replica_groups``).
    """
    try:
        mesh_registry = import_module(_TORCH_NEURONX_MESH_REGISTRY_MODULE)
        mesh_registry._MESH_REGISTRY[group_name] = groups
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Incompatible torch-neuronx installation: expected "
            f"{_TORCH_NEURONX_MESH_REGISTRY_MODULE}._MESH_REGISTRY"
        ) from exc


def memory_stats_function():
    """torch-neuronx's private ``_C._get_memory_stats(device_index)``, or ``None`` if missing."""
    from torch_neuronx import _C

    return getattr(_C, "_get_memory_stats", None)


def neuron_dynamo_backend():
    """torch-neuronx's ``neuron`` torch.compile backend function."""
    from torch_neuronx.neuron_dynamo_backend.backend import neuron_backend

    return neuron_backend
