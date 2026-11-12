# SPDX-License-Identifier: Apache-2.0
"""Per-rank Neuron runtime start-up for SGLang Diffusion workers on torch-neuronx.

SGLang Diffusion runs one worker process per NeuronCore. Each worker, in this order:

1. ``configure_worker(local_rank)`` (from the platform's ``get_device``, before
   anything touches the device): one host thread per worker, this rank's core in
   ``NEURON_RT_VISIBLE_CORES``, then the torch_neuronx import and the ``torch.neuron``
   fill-ins SGLang's worker needs.
2. SGLang initializes ``torch.distributed`` with ``cpu:gloo,neuron:neuron``. torch-neuronx's
   ``neuron`` process group sets up the device runtime: rank and world size, the CCOM root
   address, ``nrt_init`` and a barrier.
3. ``set_worker_device()``: makes the worker's device (``neuron:<local_rank>``) current.
4. ``initialize_model_groups(...)``: the TP / CP / CFG groups the models use.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

_configured_local_rank: int | None = None
_rank_tensors: list[torch.Tensor] = []


def neuron_device() -> torch.device:
    """This worker's device. ``neuron`` is a torch device type only once torch_neuronx is imported.

    torch-neuronx numbers devices by local rank once its ``neuron`` process group is up (as
    CUDA does): the only valid index in a worker is then its local rank.
    """
    return torch.device("neuron", _configured_local_rank or 0)


def local_model_path(model_path: str) -> str:
    """``model_path`` itself when it is a directory, else its Hub snapshot's local directory."""
    if os.path.isdir(model_path):
        return model_path
    from huggingface_hub import snapshot_download

    return snapshot_download(model_path)


def freeze_gc_heap() -> None:
    """Move the loaded model's objects out of the collector's reach, so a request never pays
    for a full collection over hundreds of thousands of objects."""
    import gc

    gc.collect()
    gc.freeze()


def worker_core(local_rank: int) -> int:
    """The NeuronCore this worker owns: the ``local_rank``-th of the server's cores.

    The server's cores are ``cpu_placement.server_cores()``, as the LLM path reads them:
    ``NEURON_VISIBLE_DEVICES``, else the launcher's ``NEURON_RT_VISIBLE_CORES``. With
    neither set, the worker takes core ``local_rank``.
    """
    from sglang_neuron.cpu_placement import server_cores

    cores = server_cores()
    if cores is None:
        return local_rank
    if local_rank >= len(cores):
        spec = os.environ.get("NEURON_VISIBLE_DEVICES") or os.environ.get("NEURON_RT_VISIBLE_CORES")
        raise RuntimeError(f"local rank {local_rank} has no core in {spec!r}")
    return cores[local_rank]


def limit_worker_threads(threads: int = 1) -> None:
    """``threads`` torch and OpenMP threads: one worker per core must not oversubscribe the host.

    ``configure_worker`` starts every worker at 1; a pipeline that needs another count sets it at load.
    """
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    torch.set_num_threads(threads)


def install_device_module_shims() -> None:
    """Route the no-argument ``torch.get_device_module()`` to ``torch.neuron``.

    torch-neuronx registers the PrivateUse1 hooks when it is imported, but in a worker
    ``torch.get_device_module()`` with no argument still answers ``torch.cpu``: SGLang resolves
    the current accelerator at import, before ``configure_worker`` imports torch_neuronx. SGLang's
    worker then calls ``torch.cpu`` (``perf_logger.capture_memory_snapshot`` fails on
    ``torch.cpu.memory_allocated``), so the override answers ``torch.neuron`` instead.
    """
    # Call after the torch_neuronx import, which registers torch.neuron.
    module = torch.get_device_module("neuron")
    original = torch.get_device_module
    if getattr(original, "_sglang_neuron_shim", False):
        return

    def get_device_module(device=None):
        if device is None:
            return module
        return original(device)

    # As functools.wraps would set it: code that unwraps the override reaches torch's function.
    get_device_module.__wrapped__ = getattr(original, "__wrapped__", original)
    get_device_module.__doc__ = original.__doc__
    get_device_module._sglang_neuron_shim = True
    torch.get_device_module = get_device_module


def use_cheap_memory_counters() -> list[str]:
    """Answer torch.neuron's four memory counters from torch-neuronx's allocator counters.

    SGLang reads ``memory_allocated``, ``memory_reserved`` and their ``max_`` forms twice per
    request on every rank (``perf_logger.capture_memory_snapshot``, before and after the
    forward). In torch-neuronx 2.13.3 each of them calls ``memory_stats()``, which also walks
    the whole allocator snapshot in Python for the per-pool split
    (``memory._compute_per_pool_current``) and queries the runtime's usage breakdown; the
    four values use neither. The private ``_C._get_memory_stats`` gives the same ``*.all.*``
    values without the walk, so the plugin reads it. Keeps torch-neuronx's functions if that
    private call is missing.
    """
    import torch_neuronx

    from sglang_neuron.diffusion import neuron_api

    stats_fn = neuron_api.memory_stats_function()
    if stats_fn is None:
        logger.warning("torch_neuronx._C._get_memory_stats is missing; memory counters unchanged")
        return []
    module = torch.get_device_module("neuron")

    def index(device=None) -> int:
        return device_index(device, torch.neuron.current_device)

    def counter(category: str, metric: str):
        def read(device=None) -> int:
            return int(stats_fn(index(device)).get(category, {}).get(metric, 0))

        read.__name__ = f"{category}_{metric}"
        return read

    replaced = {
        "memory_allocated": counter("allocated_bytes", "current"),
        "max_memory_allocated": counter("allocated_bytes", "peak"),
        "memory_reserved": counter("reserved_bytes", "current"),
        "max_memory_reserved": counter("reserved_bytes", "peak"),
    }
    for name, fn in replaced.items():
        setattr(module, name, fn)
        setattr(torch_neuronx, name, fn)
    return sorted(replaced)


def device_index(device, current) -> int:
    """The device index that torch's memory counters take: ``device`` is None (``current()``,
    the current device), an index, a ``torch.device`` or a string such as ``"neuron:0"``."""
    if isinstance(device, str):
        device = torch.device(device)
    idx = getattr(device, "index", device)
    return current() if idx is None else int(idx)


def configure_worker(local_rank: int) -> torch.device:
    """Claim this worker's core and import torch_neuronx; idempotent per process."""
    global _configured_local_rank
    if _configured_local_rank is not None:
        if _configured_local_rank != local_rank:
            raise RuntimeError(
                f"worker already configured for local rank {_configured_local_rank}, "
                f"not {local_rank}"
            )
        return neuron_device()
    core = worker_core(local_rank)
    limit_worker_threads()
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(core)
    # Imported for its side effects too: the ShardIndexInjection opt-out
    # (TORCH_NEURONX_VOCAB_SHARDING_SPMD_DISABLE) must be set before the first compile.
    from sglang_neuron.diffusion import neuron_api

    # torch-neuronx runs a graph that fails on the device on CPU instead (and only
    # logs it); fail instead, so a broken graph cannot pass as a slow run.
    os.environ.setdefault("TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION", "1")
    # A NEFF's cache key covers neither the NKI kernel sources nor the kernel library, so a
    # cache built from other kernels would serve them: refuse it before the first compile, as
    # the LLM backend does (backend/compile_cache; stamps an empty cache).
    from sglang_neuron.backend import compile_cache

    compile_cache.check_compile_caches()
    # The runtime starts lazily, at the first device use or when the ``neuron``
    # process group is created (which must come first in a distributed run).
    neuron_api.initialize()
    install_device_module_shims()
    use_cheap_memory_counters()
    _configured_local_rank = local_rank
    logger.info("Neuron diffusion worker: local rank %d -> NeuronCore %d", local_rank, core)
    return neuron_device()


def set_worker_device() -> torch.device:
    """Make this worker's device current; call after ``torch.distributed`` is up.

    torch-neuronx's ``neuron`` process group sets up the runtime when the world group is
    created (root address from the store, ``nrt_init`` and a barrier on it) and numbers this
    process's device by local rank. Making that device current places tensors created on
    plain ``"neuron"`` on it.
    """
    device = neuron_device()
    torch.neuron.set_device(device)
    return device


def initialize_model_groups(
    *, tensor_parallel_size: int, context_parallel_size: int, cfg_parallel_size: int = 1
) -> None:
    """Bind the models to SGLang's groups once per process (after SGLang's distributed init)."""
    import torch.distributed as dist

    from sglang_neuron.diffusion.distributed import groups
    from sglang_neuron.diffusion.distributed.parallel_state import initialize_parallel_groups

    if groups.model_parallel_is_initialized():
        return
    _rank_tensors.extend(
        initialize_parallel_groups(
            neuron_device(),
            world_size=dist.get_world_size(),
            tensor_parallel_size=tensor_parallel_size,
            context_parallel_size=context_parallel_size,
            cfg_parallel_size=cfg_parallel_size,
        )
    )
