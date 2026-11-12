# SPDX-License-Identifier: Apache-2.0
"""The plugin's one interface to torch-neuronx's framework services.

torch-neuronx, AWS Neuron's native PyTorch backend (the ``neuron`` PrivateUse1 device and the ``neuron`` dynamo
backend: FX -> StableHLO -> neuronx-cc), owns the device, the executor and the replica-mesh registry
(``torch_neuronx.distributed.mesh_registry``). This module wraps the services most of the plugin uses: the NKI kernel
wrapper, the NKI dtype names, the platform target, the replica-mesh registry, the dynamo backend name and the compiler
environment. A few modules talk to torch-neuronx directly for what they own: the device copies and memory queries
(``backend/device_io.py``), the executor's binding cache (``backend/executor_cache.py``), and the DeepSeek-V2 MLA
kernels and the logical-core count in ``nki/``. A CPU-mode process (``SGLANG_NEURON_CPU_MODE=1``) does not load
torch-neuronx and gets host stand-ins here. Nothing here is called inside a compiled forward except the NKI wrapper's
returned callers.
"""

from __future__ import annotations

import contextlib
import functools
import threading
from typing import Any, Iterator, List

from sglang_neuron.backend import envs

# Canonical target families: a revision-suffixed target string maps to its family prefix.
_TARGET_FAMILIES = ("trn1n", "trn1", "trn2", "trn3", "inf2")


# compile backends
def compile_backend_name(cpu_mode: bool = False) -> str:
    """The dynamo backend that compiles the serving graphs: torch-neuronx's ``neuron`` in a device process; a CPU-mode
    process (torch-neuronx has no CPU mode) traces with ``aot_eager``, which runs the same functionalized graph on the
    host."""
    return "aot_eager" if cpu_mode else "neuron"


# NKI kernels
def wrap_nki(kernel: Any, **kwargs: Any) -> Any:
    """Wrap an NKI kernel so a traced graph calls it as one custom call (the caller supports ``kernel[grid](...)``).

    A wrap may run at module import or inside a forward that torch.compile traces: the wrapper runs at trace time (it
    inspects the kernel's signature and registers the kernel), and the traced graph holds only the custom call, so a
    compiled graph's executions do not wrap again. The ops wrap where they pick the kernel for the traced shape
    (sampling_topk.py, ops/attention, ops/moe)."""
    if envs.SGLANG_NEURON_CPU_MODE:
        return _HostProcessKernel(kernel)
    from torch_neuronx.nki_hop import wrap_nki as _wrap

    return _wrap(_as_nki_jit(kernel), **kwargs)


class _HostProcessKernel:
    """An NKI kernel of a CPU-mode process. Such a process does not load torch-neuronx (importing it registers its
    neuron device module over the CPU-mode stand-in), and torch-neuronx has no host dispatch of NKI kernels, so the ops
    take their torch fallback and a call of the kernel raises."""

    def __init__(self, kernel):
        self.kernel = kernel

    def __getitem__(self, grid):
        return self

    def __call__(self, *args, **kwargs):
        name = getattr(getattr(self.kernel, "func", self.kernel), "__name__", repr(self.kernel))
        raise RuntimeError(f"NKI kernel {name} called in a CPU-mode process: torch-neuronx runs NKI kernels on the "
                           "device only")


def _as_nki_jit(kernel: Any) -> Any:
    """torch-neuronx wraps NKI kernel objects (``nki.jit``): a plain kernel function is jitted first."""
    if hasattr(kernel, "func"):
        return kernel
    import nki

    return nki.jit(kernel)


_TORCH_TO_NKI_DTYPE_NAMES = {
    "bfloat16": "bfloat16", "float32": "float32", "float16": "float16", "float64": "float64",
    "int64": "int64", "uint64": "uint64", "int32": "int32", "uint32": "uint32", "int16": "int16",
    "uint16": "uint16", "int8": "int8", "uint8": "uint8", "bool": "bool",
    "float8_e4m3fn": "float8_e4m3fn", "float8_e5m2": "float8_e5m2",
}


def torch_to_nki_dtype(dtype) -> str:
    """The ``nki.language`` dtype name of a torch dtype. Trn2 has no OCP e4m3fn: there torch's ``float8_e4m3fn``
    maps to NKI's ``float8_e4m3``."""
    import torch

    name = str(dtype).removeprefix("torch.")
    if dtype == torch.float8_e4m3fn and normalize_target_family(get_platform_target()) == "trn2":
        return "float8_e4m3"
    try:
        return _TORCH_TO_NKI_DTYPE_NAMES[name]
    except KeyError:
        raise ValueError(f"Unsupported torch dtype for NKI: {dtype}") from None


# platform target
def get_platform_target() -> str:
    """The NKI / compiler platform target (``NEURON_PLATFORM_TARGET_OVERRIDE`` first, then the runtime's instance
    family), as torch-neuronx reports it."""
    import os

    override = os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE")
    if override:
        return override
    if envs.SGLANG_NEURON_CPU_MODE:
        raise RuntimeError("a CPU-mode process has no Neuron runtime to read the instance family from: set "
                           "NEURON_PLATFORM_TARGET_OVERRIDE")
    from torch_neuronx.utils import get_platform_target as _target

    return str(_target())


def normalize_target_family(target: str) -> str:
    """Map a (possibly revision-suffixed) target string to its family (``trn3-rev2`` -> ``trn3``)."""
    for family in _TARGET_FAMILIES:
        if target.startswith(family):
            return family
    raise RuntimeError(f"platform target {target!r} does not map to a known target family {list(_TARGET_FAMILIES)}")


# replica meshes
# A traced collective names its process group; torch-neuronx legalizes it into the replica groups of the group's whole
# partition (every rank's cell), the same on every rank, so all ranks compile one SPMD NEFF. torch-neuronx keeps that
# partition in its registry, keyed by the process group's name: torch's own, which every member of the group holds.
class _HostMeshRegistry:
    """The replica-mesh registry of a CPU-mode process, which does not load torch-neuronx
    (TORCH_DEVICE_BACKEND_AUTOLOAD=0; importing it would register a second "neuron" device module), for graphs that run
    on the host. It models the parts of torch_neuronx.distributed.mesh_registry the plugin uses, and only those: the
    _MESH_REGISTRY table and _register_replica_groups (a conflicting re-register raises)."""

    def __init__(self):
        self._MESH_REGISTRY = {}

    def _register_replica_groups(self, name, groups):
        existing = self._MESH_REGISTRY.get(name)
        if existing is not None and existing != groups:
            raise RuntimeError(f"process group {name!r} is registered with replica groups {existing}, not {groups}")
        self._MESH_REGISTRY[name] = groups


@functools.lru_cache(maxsize=1)
def _tnx_registry():
    from sglang_neuron.backend import envs

    if envs.SGLANG_NEURON_CPU_MODE:
        return _HostMeshRegistry()
    from torch_neuronx.distributed import mesh_registry

    return mesh_registry


def register_mesh(pg_name: str, mesh: List[List[int]]) -> None:
    """Register ``mesh`` (the full partition) as the replica groups of process group ``pg_name``."""
    cells = [[int(r) for r in cell] for cell in mesh]
    # raises if the name is already registered with a different partition: two groups under one name would compile one
    # of them against the other's replica groups
    _tnx_registry()._register_replica_groups(str(pg_name), cells)


def resolve_full_mesh(group) -> List[List[int]]:
    """The full partition registered for a GroupCoordinator's device group, which an NKI collective takes as its
    replica groups. A group with no registered partition raises: torch-neuronx's own lookup would fall back to this
    rank's cell, and every cell would then compile a different collective instead of one SPMD NEFF."""
    device_group = getattr(group, "device_group", None)
    if device_group is None:
        raise ValueError(f"resolve_full_mesh needs a GroupCoordinator (a device_group); got {type(group).__name__}")
    name = str(device_group.group_name)
    cells = _tnx_registry()._MESH_REGISTRY.get(name)
    if cells is None:
        raise RuntimeError(f"process group {name!r} has no registered partition: create it with "
                           "parallel.state._make_group")
    return [[int(r) for r in cell] for cell in cells]


def mesh_logger_names() -> tuple:
    """torch-neuronx's replica-mesh loggers. SGLANG_NEURON_DEBUG_MESH=1 raises them to DEBUG, where they log every
    registered and every resolved process group's replica groups; a group with no registered partition is logged at
    ERROR whatever the level."""
    return ("torch_neuronx.distributed.mesh_registry",
            "torch_neuronx.neuron_dynamo_backend.fx.passes.collective_legalization")


# compiler arguments
# torch-neuronx compiles a torch.compile graph with "--model-type transformer --auto-cast=none", then NEURON_CC_FLAGS,
# at the optimization level NEURON_COMPILER_OPT_LEVEL (default -O2). It ignores a graph's own compiler arguments and
# reads the two variables when the graph is first called (the call that compiles it); both are part of the NEFF's
# persistent cache key, and a compiled graph keeps its NEFF whatever the variables hold later. So the plugin sets them
# around the call that compiles each graph family (compile_scope): every family compiles with its own arguments, and a
# restart with the same arguments finds the same NEFFs.
#
# Two defaults of torch-neuronx's compile differ from the program the plugin's argument list describes, and both are
# overridden:
# * the transformer model type deadlocks the GPT-OSS packed-FP8 decode under DP attention (the compiler places MoE work
#   between the attention kernel's cross-core exchange and the reduce-scatter, and the two physical cores of a logical
#   core wait on each other); "--model-type=generic", after torch-neuronx's own on the command line, wins. A list that
#   names a model type keeps it (a model's own list may name the type its programs are measured at);
# * its StableHLO carries the module hierarchy, which turns neuronx-cc's modular flow to the hierarchy splitter; the
#   plugin's program uses the split finder ("--hierarchy-splitter=false" in the hlo2tensorizer options), unless a
#   graph family's arguments name the splitter (a prefill family whose layers are submodules may choose the hierarchy
#   splitter, which compiles each distinct layer module once).
GENERIC_MODEL_TYPE_ARG = "--model-type=generic"
SPLITTER_OPTION = "--hierarchy-splitter="
SPLIT_FINDER_OPTION = SPLITTER_OPTION + "false"
_MODEL_TYPE_ARG = "--model-type"
_HLO2TENSORIZER_ARG = "--internal-hlo2tensorizer-options="
_OPT_LEVELS = ("-O0", "-O1", "-O2", "-O3")
_COMPILER_ENV = ("NEURON_COMPILER_OPT_LEVEL", "NEURON_CC_FLAGS")
# The values the names hold outside a compile_scope: the plugin's, not torch-neuronx's defaults (its torch.compile
# graphs default to -O2, only its eager XLA kernels to -O1). No serving graph compiles outside a scope: warm-up
# compiles every graph in one, and serving runs under fail_on_recompile.
_COMPILER_ENV_DEFAULTS = {"NEURON_COMPILER_OPT_LEVEL": "-O1", "NEURON_CC_FLAGS": ""}
_COMPILE_ENV_LOCK = threading.RLock()  # a scoped call may compile inside a warm-up's scope


def with_hlo_options(compiler_args: List[str], *, drop: tuple = (), add: tuple = ()) -> List[str]:
    """``compiler_args`` with the hlo2tensorizer options starting with any of ``drop`` removed and ``add`` appended
    (a graph family's change to the plugin's argument list)."""
    import shlex

    args = []
    for arg in compiler_args:
        if arg.startswith(_HLO2TENSORIZER_ARG):
            options = [o for o in shlex.split(arg[len(_HLO2TENSORIZER_ARG):]) if not o.startswith(drop)]
            arg = _HLO2TENSORIZER_ARG + shlex.join(options + list(add))
        args.append(arg)
    return args


def torchneuronx_compiler_env(compiler_args: List[str]) -> dict:
    """The NEURON_COMPILER_OPT_LEVEL / NEURON_CC_FLAGS values that compile ``compiler_args`` (the plugin's compiler
    argument list) on torch-neuronx: the optimization level apart, the split finder in the hlo2tensorizer options, the
    generic model type last unless the list names one."""
    import shlex

    level, args = "-O1", []
    for arg in compiler_args:
        if arg in _OPT_LEVELS:
            level = arg
        else:
            args.append(arg)
    for i, arg in enumerate(args):
        if arg.startswith(_HLO2TENSORIZER_ARG):
            # a family that names the splitter itself keeps its choice
            if not any(o.startswith(SPLITTER_OPTION) for o in shlex.split(arg[len(_HLO2TENSORIZER_ARG):])):
                args[i] = f"{arg} {SPLIT_FINDER_OPTION}"
            break
    else:
        args.append(_HLO2TENSORIZER_ARG + SPLIT_FINDER_OPTION)
    if not any(arg.startswith(_MODEL_TYPE_ARG) for arg in args):
        args.append(GENERIC_MODEL_TYPE_ARG)
    return {"NEURON_COMPILER_OPT_LEVEL": level, "NEURON_CC_FLAGS": shlex.join(args)}


# torch-neuronx's ShardIndexInjection pass (neuron_dynamo_backend/fx/passes/vocab_sharding_spmd.py) gives every graph
# an int64 vocab-shard-index input that its executor creates on the device at each call (torch.tensor(..., device=)):
# a host-to-device copy on the graphs' stream between two graphs, which torch-neuronx's scheduler starts only once the
# graph before it has retired, and the next graph only once the copy has, so every prefill forward started late after
# the one before it. The plugin's graphs need no injected index: a shard offset is computed from a group's
# rank tensor, a graph input (SPMD check), so the pass is off in every process.
VOCAB_SHARDING_PASS = "TORCH_NEURONX_VOCAB_SHARDING_SPMD_DISABLE"


def disable_vocab_sharding_pass(environ=None) -> None:
    """Turn torch-neuronx's vocab-sharding SPMD passes off in this process (``VOCAB_SHARDING_PASS=1``, read at each
    graph's compile); a launch environment that sets another value is refused."""
    import os

    environ = os.environ if environ is None else environ
    value = environ.get(VOCAB_SHARDING_PASS)
    if value is not None and value != "1":
        raise ValueError(f"{VOCAB_SHARDING_PASS}={value!r}: the vocab-sharding pass gives every graph a per-call input "
                         "copied between graphs, which delays each graph until the one before it has retired; remove "
                         "it from the launch environment")
    environ[VOCAB_SHARDING_PASS] = "1"


NKI_TRACE_CACHE_OFF = "NKI_DISABLE_TRACE_CACHE"


def disable_nki_trace_cache(environ=None) -> None:
    """Keep NKI's trace cache off in this process (``NKI_DISABLE_TRACE_CACHE=1``, which overrides the
    NKI_ENABLE_TRACE_CACHE=1 torch-neuronx sets): it keys a kernel trace without the compiler arguments, and each graph
    family compiles with its own, so a trace of one family would serve another. A launch environment that sets another
    value is refused (backend/envs sets it when it is first imported, before any kernel is traced)."""
    import os

    environ = os.environ if environ is None else environ
    value = environ.get(NKI_TRACE_CACHE_OFF)
    if value is not None and value != "1":
        raise ValueError(f"{NKI_TRACE_CACHE_OFF}={value!r}: NKI's trace cache keys a kernel trace without the compiler "
                         "arguments, so a trace of one graph family would serve another; remove it from the launch "
                         "environment")
    environ[NKI_TRACE_CACHE_OFF] = "1"


def refuse_launch_compiler_env(environ=None) -> None:
    """The plugin owns NEURON_CC_FLAGS and NEURON_COMPILER_OPT_LEVEL (compile_scope): a launch environment that sets
    either would compile torch-neuronx's own ops with it and is refused, naming it."""
    import os

    environ = os.environ if environ is None else environ
    # _COMPILER_ENV_DEFAULTS (the plugin's outside-scope values, not torch-neuronx's defaults) are accepted:
    # reserve_compiler_env sets them and a spawned process inherits them
    preset = sorted(k for k in _COMPILER_ENV if environ.get(k, _COMPILER_ENV_DEFAULTS[k]) != _COMPILER_ENV_DEFAULTS[k])
    if preset:
        raise ValueError(f"{', '.join(preset)} in the launch environment: the plugin sets the compiler arguments per "
                         f"graph family from its own list; unset {'them' if len(preset) > 1 else 'it'}")


def reserve_compiler_env(environ=None) -> None:
    """Give NEURON_COMPILER_OPT_LEVEL and NEURON_CC_FLAGS a value (_COMPILER_ENV_DEFAULTS) in the process environment
    where they are unset, before its runtime starts, so that compile_scope only ever replaces their values.
    torch-neuronx's runtime worker reads both (getenv) while it loads a NEFF, concurrently with the scheduler thread's
    scopes, and adding or removing a name reallocates the C environment array under that reader (glibc frees the old
    array): the load segfaults in nrt_load's getenv. Replacing a value swaps one pointer and frees nothing."""
    import os

    environ = os.environ if environ is None else environ
    for name, value in _COMPILER_ENV_DEFAULTS.items():
        environ.setdefault(name, value)


@contextlib.contextmanager
def compile_scope(compiler_args: List[str]) -> Iterator[None]:
    """Hold the compiler environment that compiles ``compiler_args`` for the duration of the block (the warm-up calls
    that compile a graph family), then restore the previous values."""
    with _held_compiler_env(torchneuronx_compiler_env(compiler_args)):
        yield


@contextlib.contextmanager
def _held_compiler_env(wanted: dict) -> Iterator[None]:
    """Hold ``wanted`` (torchneuronx_compiler_env's values) in the process environment for the block, then restore the
    previous values: a name reserve_compiler_env reserved gets its value back and is never removed (a removal
    reallocates the environment array under the runtime's readers); only a process that reserved nothing (CPU mode)
    has a name to remove."""
    import os

    with _COMPILE_ENV_LOCK:
        previous = {k: os.environ.get(k) for k in wanted}
        os.environ.update(wanted)
        try:
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
