# SPDX-License-Identifier: Apache-2.0
"""SGLang Diffusion platform for AWS Neuron on torch-neuronx.

Registered through the ``sglang.multimodal_gen.platforms`` entry point
(``sglang_neuron.diffusion:activate``).
SGLang Diffusion runs one worker per NeuronCore; the Neuron pipelines it serves
(see ``sglang_neuron.diffusion.plugin``) bring their own compiled modules, process
groups and kernels, so this platform's job is the worker lifecycle around them:

- ``get_device`` claims the rank's core and imports torch_neuronx
  (``runtime.configure_worker``), before SGLang initializes ``torch.distributed``.
- ``apply_server_args_defaults`` turns off CPU offload, breakable CUDA graphs, FSDP
  and SGLang's torch.compile: the Neuron pipeline compiles its own graphs.
- Memory queries report the per-core HBM of the target; the launcher calls them
  before any worker exists, so they must not open the device or import torch_neuronx.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import torch
from sglang.multimodal_gen.runtime.platforms.interface import (
    AttentionBackendEnum,
    Platform,
    PlatformEnum,
)

logger = logging.getLogger(__name__)

# Per-core HBM in GiB at NEURON_LOGICAL_NC_CONFIG=2 (one logical core = one NC pair), by
# target family (``normalize_target_family``).
_HBM_GIB = {"trn2": 24, "trn3": 36}


# The EC2 instance type, for example "trn2.48xlarge".
_DMI_PRODUCT_NAME = "/sys/devices/virtual/dmi/id/product_name"


def _instance_target() -> str | None:
    """The target family of the instance type (``trn2.48xlarge`` -> ``trn2``), or ``None``."""
    from sglang_neuron.backend.stack import normalize_target_family

    try:
        with open(_DMI_PRODUCT_NAME) as f:
            return normalize_target_family(f.read().strip())
    except (OSError, RuntimeError):  # no DMI file, or not a Neuron instance type
        return None


def _platform_target() -> str:
    """The target family, without importing torch_neuronx in a process that has not.

    The launcher calls the memory queries after ``apply_server_args_defaults``
    (``ServerArgs._adjust_parameters``), before any worker owns a core; importing
    torch_neuronx there would make the Neuron device torch's accelerator in the launcher
    (``neuron_api.keep_runtime_out_of_host_processes``). So torch-neuronx is asked only once it
    is imported (a configured worker) or when ``NEURON_PLATFORM_TARGET_OVERRIDE`` names the
    target; otherwise the target comes from the instance type, else ``trn2``.
    """
    import os
    import sys

    from sglang_neuron.backend.stack import get_platform_target, normalize_target_family

    if "torch_neuronx" in sys.modules or os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE"):
        try:
            return normalize_target_family(get_platform_target())
        except Exception:  # an unknown target, or no runtime to read it from
            pass
    return _instance_target() or "trn2"


def _worker_reserved_bytes(device_id: int | None = None) -> int | None:
    """Bytes torch-neuronx's allocator has reserved, in a configured worker whose runtime is up.

    ``None`` elsewhere: in the launcher (torch_neuronx is not imported) and in a worker before
    torch-neuronx's ``neuron`` process group starts the runtime, which must come first. The
    counter is the cheap one ``runtime.use_cheap_memory_counters`` installs (no allocator walk).
    """
    import sys

    from sglang_neuron.diffusion import runtime

    torch_neuronx = sys.modules.get("torch_neuronx")
    if runtime._configured_local_rank is None or torch_neuronx is None:
        return None
    from sglang_neuron.diffusion import neuron_api

    if not neuron_api.runtime_started(torch_neuronx):
        return None
    return int(torch.neuron.memory_reserved(device_id))


def _apply_env_profiles(model_path: str | None) -> None:
    """Apply the environment profiles (compiler and runtime settings) before launch.

    ``PLATFORM_DEFAULTS`` always, then the profile of the checkpoint's ``model_index.json``
    class, if it has one; a Wan pipeline class gets ``env_profiles.WAN22`` only for a Wan2.2 A14B
    checkpoint (the only Wan checkpoints the Neuron Wan pipelines run; they refuse any other at
    construction, so it gets none). Every entry is a default: an exported value wins, except for
    the compiler variables, which an exported value may not change
    (``env_profiles.refuse_launch_compiler_env``), and a profile's ``assign`` entries: ``WAN22``
    forces its two CC DMA packet-size variables and ``LINGBOT_WORLD`` its LNC.

    ``model_index.json`` is read as SGLang's pipeline registry reads it; for a Hub ID only that
    file is fetched, or its cached copy read.
    """
    from sglang_neuron.diffusion import env_profiles
    from sglang_neuron.diffusion.model_index import read_model_index
    from sglang_neuron.diffusion.models.minimax_h3 import MINIMAX_H3_CLASS_NAME
    from sglang_neuron.diffusion.models.wan2_2.checkpoint import (
        WAN22_A14B_CLASS_NAMES,
        is_wan22_a14b,
    )

    class_name = read_model_index(model_path).get("_class_name")
    profile = {
        "FluxPipeline": env_profiles.FLUX1_DEV,
        MINIMAX_H3_CLASS_NAME: env_profiles.MINIMAX_H3,
        "LingBotWorldCausalDMDPipeline": env_profiles.LINGBOT_WORLD,
    }.get(class_name)
    if class_name in WAN22_A14B_CLASS_NAMES and is_wan22_a14b(model_path, class_name):
        profile = env_profiles.WAN22
    env_profiles.refuse_launch_compiler_env(profile)
    env_profiles.apply(env_profiles.PLATFORM_DEFAULTS)
    if profile is not None:
        env_profiles.apply(profile)
        if profile is env_profiles.WAN22:
            logger.info("Wan2.2 environment: %s", env_profiles.effective(profile))
        elif profile is env_profiles.LINGBOT_WORLD:
            logger.info("LingBot-World environment: %s", env_profiles.effective(profile))


def _limit_launcher_threads(tp_size: int) -> None:
    """Cap the launcher's CPU thread pool at ``cpus // tp`` threads (``thread_limits``).

    Without it the launcher's torch pool has one thread per CPU, and its
    OpenMP threads busy-wait after each output conversion, on the CPUs the workers use
    to submit the next request. The environment is read when torch loads, so the
    launcher (which has already imported torch) also sets the pool size directly;
    workers set their own pool to one thread at start-up.
    """
    import multiprocessing

    from sglang_neuron.diffusion import env_profiles

    env_profiles.apply(env_profiles.thread_limits(tp_size))
    if multiprocessing.current_process().name == "MainProcess":
        import os

        torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))


def _keep_runtime_out_of_torch_import() -> None:
    """Keep torch_neuronx out of the workers until ``configure_worker``.

    See ``neuron_api.keep_runtime_out_of_host_processes``. In a worker, an early
    torch_neuronx import also breaks SGLang's own imports
    (``torch.get_device_module().Event | None`` in its disaggregation transport manager:
    torch-neuronx's ``Event`` is a function). The launcher must have the same environment
    from its start; if it has already started the runtime, it holds every core and the
    workers cannot start, so fail here with the fix.
    """
    import sys

    from sglang_neuron.diffusion import neuron_api

    neuron_api.keep_runtime_out_of_host_processes()
    torch_neuronx = sys.modules.get("torch_neuronx")
    if torch_neuronx is not None and neuron_api.runtime_started(torch_neuronx):
        raise RuntimeError(
            "The SGLang launcher started the torch-neuronx runtime, so it holds every visible "
            "NeuronCore. Launch with TORCH_DEVICE_BACKEND_AUTOLOAD=0 in the environment."
        )
    if torch_neuronx is not None:
        logger.warning(
            "torch_neuronx is imported in the SGLang launcher; launch with "
            "TORCH_DEVICE_BACKEND_AUTOLOAD=0"
        )


# Options the Neuron pipelines do not support: they keep every component on the NeuronCores and
# compile their own graphs. ``apply_server_args_defaults`` turns them off.
_UNSUPPORTED_SERVER_ARGS = (
    "dit_cpu_offload",
    "dit_layerwise_offload",
    "text_encoder_cpu_offload",
    "image_encoder_cpu_offload",
    "vae_cpu_offload",
    "use_fsdp_inference",
    "enable_torch_compile",
    "enable_breakable_cuda_graph",
)
# SGLang's per-component placement options (component residency, CPU and layerwise offload of
# named components). SGLang resolves them after this hook, from their own values, so an explicit
# value is refused; unset, they leave every component resident. Not overwritten: their types
# are not boolean.
_UNSUPPORTED_PLACEMENT_ARGS = (
    "component_residency",
    "cpu_offload_components",
    "layerwise_offload_components",
)


def _refuse_unserved_server_args(server_args) -> None:
    """Refuse a launch that turns on an option the Neuron pipelines do not support, naming it.

    SGLang records which options the user set (``ServerArgs.is_arg_explicitly_set``, filled
    before this hook runs). An unsupported option set to a true value is refused, as the
    plugin's LLM platform refuses unserved launch arguments; SGLang's own defaults are turned
    off without a message.
    """
    is_set = getattr(server_args, "is_arg_explicitly_set", None)
    if not callable(is_set):
        return
    asked = [
        "--" + name.replace("_", "-")
        for name in (*_UNSUPPORTED_SERVER_ARGS, *_UNSUPPORTED_PLACEMENT_ARGS)
        if getattr(server_args, name, None) and is_set(name)
    ]
    if asked:
        raise ValueError(
            f"sglang-neuron's diffusion platform does not serve {', '.join(asked)}; "
            "launch without it"
        )


def _refuse_disaggregation(server_args) -> None:
    """Refuse a disaggregated or pool launch (``--disagg-role`` other than ``monolithic``).

    A role instance's local ranks start at its first GPU id (``launch_disagg_role`` passes
    ``base_gpu_id + rank``, the pool launch the role's GPU list), while torch-neuronx numbers a
    worker's device among the group's workers on the host, from 0. ``runtime.neuron_device``
    would then name a device the worker does not have. The hook runs before SGLang turns a
    ``disagg_role`` string into its enum, so both forms are read.
    """
    role = getattr(server_args, "disagg_role", None)
    role = str(getattr(role, "value", role) or "monolithic").lower()
    if role != "monolithic" or getattr(server_args, "disagg_mode", False):
        raise ValueError(
            f"sglang-neuron's diffusion platform does not serve --disagg-role {role} "
            "(disaggregated or pool serving); launch without it"
        )


class NeuronDiffusionPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name: str = "neuron"
    device_type: str = "neuron"
    dispatch_key: str = "PrivateUse1"
    # SGLang host helpers only; model forwards compile through the Neuron pipeline.
    simple_compile_backend: str = "eager"

    _local_rank: int | None = None

    def init_backend(self) -> None:
        # Nothing to do before the worker knows its rank: torch_neuronx is imported in
        # get_device (after SGLang's worker modules import, so their import-time
        # torch.get_device_module() calls still see the CPU accelerator).
        return None

    def apply_server_args_defaults(self, server_args) -> None:
        # Runs in the launcher before workers spawn, so the workers inherit the environment.
        _keep_runtime_out_of_torch_import()
        _refuse_disaggregation(server_args)
        _refuse_unserved_server_args(server_args)
        _apply_env_profiles(getattr(server_args, "model_path", None))
        _limit_launcher_threads(int(getattr(server_args, "tp_size", None) or 1))
        for name in _UNSUPPORTED_SERVER_ARGS:
            if hasattr(server_args, name):
                setattr(server_args, name, False)

    def get_compile_backend(self, mode: str | None = None) -> str:
        return "eager"

    def get_dispatch_key_name(self) -> str:
        return "native"

    # ---- device -------------------------------------------------------------------
    def get_device(self, local_rank: int) -> torch.device:
        """The first per-rank call in a worker: claim the rank's core and import torch_neuronx.

        In a worker process this sets ``NEURON_RT_VISIBLE_CORES`` to the rank's core and
        imports torch_neuronx (``runtime.configure_worker``); the device is
        ``neuron:<local_rank>``. Elsewhere (the launcher) it only names the ``neuron``
        device type, without importing torch_neuronx or opening a core.
        """
        import multiprocessing

        type(self)._local_rank = local_rank
        if multiprocessing.current_process().name == "MainProcess":
            # Name the device type only: importing torch_neuronx here would make the
            # Neuron device torch's accelerator in the launcher (see
            # neuron_api.keep_runtime_out_of_host_processes). torch documents
            # rename_privateuse1_backend as once per process; torch's private getter of the
            # current name guards a second call.
            if torch._C._get_privateuse1_backend_name() != "neuron":
                torch.utils.rename_privateuse1_backend("neuron")
            return torch.device("neuron", 0)
        from sglang_neuron.diffusion.runtime import configure_worker

        return configure_worker(local_rank)

    @classmethod
    def get_local_torch_device(cls) -> torch.device:
        from sglang_neuron.diffusion.runtime import neuron_device

        return neuron_device()

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        # The device is claimed in get_device and made current once the process group is up
        # (runtime.set_worker_device).
        return None

    @classmethod
    def synchronize(cls) -> None:
        torch.neuron.synchronize()

    @classmethod
    def empty_cache(cls) -> None:
        return None

    def tensor_on_device(self, t: torch.Tensor) -> bool:
        return t.device.type == "neuron"

    @classmethod
    def inference_mode(cls):
        # Always inference mode: apply_server_args_defaults turns off FSDP and offload.
        return torch.inference_mode(mode=True)

    @classmethod
    def is_amp_supported(cls) -> bool:
        return False

    @classmethod
    def is_float64_supported(cls) -> bool:
        return False

    @classmethod
    def enable_dit_layerwise_offload_by_default(cls) -> bool:
        return False

    @classmethod
    def seed_everything(cls, seed: int | None = None) -> None:
        if seed is not None:
            import random

            import numpy as np

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    # ---- memory -------------------------------------------------------------------
    @classmethod
    @lru_cache(maxsize=1)
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        return _HBM_GIB.get(_platform_target(), 24) * (1 << 30)

    @classmethod
    def get_current_memory_usage(cls, device=None) -> float:
        return 0.0

    @classmethod
    def get_available_gpu_memory(
        cls,
        device_id: int | None = None,
        distributed: bool = False,
        empty_cache: bool = True,
        cpu_group: Any = None,
    ) -> float:
        """Free HBM of the worker's core in GiB: the core's total minus torch-neuronx's reserved.

        The launcher calls this too (SGLang's server-args auto-tuner), before any worker owns a
        core, and a worker can call it before torch-neuronx's process group has started the
        runtime. Neither may open the device, so both get the core's total, which also keeps
        SGLang's auto-offload heuristics off. ``distributed`` takes the minimum over
        ``cpu_group``, as SGLang's CUDA platform does.
        """
        free = float(_HBM_GIB.get(_platform_target(), 24))
        reserved = _worker_reserved_bytes(device_id)
        if reserved is not None:
            free -= reserved / (1 << 30)
        if distributed:
            import torch.distributed as dist

            if dist.is_initialized():
                value = torch.tensor(free, dtype=torch.float32)
                dist.all_reduce(value, op=dist.ReduceOp.MIN, group=cpu_group)
                free = float(value.item())
        return free

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return f"NeuronCore ({_platform_target()})"

    # ---- distributed / attention --------------------------------------------------
    def get_torch_distributed_backend_str(self) -> str:
        # torch-neuronx sets up the device runtime (rank / world size, CCOM root,
        # nrt_init) in its ``neuron`` process group; Gloo keeps the CPU collectives.
        return "cpu:gloo,neuron:neuron"

    @classmethod
    def get_all_to_all_communicator_cls(cls) -> str:
        return (
            "sglang.multimodal_gen.runtime.distributed.device_communicators."
            "cpu_communicator.CpuCommunicator"
        )

    @classmethod
    def get_attn_backend_cls_str(
        cls,
        selected_backend: AttentionBackendEnum | None,
        head_size: int,
        dtype: torch.dtype,
    ) -> str:
        # The Neuron pipelines run their own attention kernels; SGLang modules that
        # still build an attention layer get SDPA.
        return "sglang.multimodal_gen.runtime.layers.attention.backends.sdpa.SDPABackend"
