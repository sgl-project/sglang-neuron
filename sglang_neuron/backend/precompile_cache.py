# SPDX-License-Identifier: Apache-2.0
"""Keep dynamo's trace of each serving graph across launches (``SGLANG_NEURON_PRECOMPILE_CACHE=1``).

A launch on a warm NEFF cache still traces every warmed graph: dynamo runs the model's Python over every layer, then
AOTAutograd and torch-neuronx lower the FX graph to StableHLO, which only then names the cached NEFF. On a large model
that is minutes per graph (Hy3 at TP64: about 150 s for each decode bucket and context), so a restart takes most of an
hour with nothing to compile. torch's caching precompile saves each compiled frame as a package (its guards and the
backend's output: torch-neuronx's ``NeuronOutputCode`` embeds the graph's StableHLO), and a later launch restores the
package in place of the trace; torch-neuronx then finds the NEFF by that StableHLO as before.

Here:

* ``enable`` turns on torch's ``caching_precompile`` and AOTAutograd cache and applies torch-neuronx's precompile
  patches (``cache_artifact._apply_precompile_patches``, which also gives each rank its own package directory by its
  ``NEURON_RT_VISIBLE_CORES`` core), before the first compile (backend/init).
* ``configure`` points ``TORCHINDUCTOR_CACHE_DIR`` at ``<root>/<fingerprint>`` before a runtime builds its compiled
  graph (graph_runner.build_compiled). The root is ``precompile`` beside torch-neuronx's NEFF cache root. The
  fingerprint covers what a restored package does not check itself, so a launch that differs in any of it traces into
  a new directory: the plugin's Python sources and the kernel-source digest (backend/compile_cache: the NKI kernels and
  kernel library version a graph's custom calls were traced from, which dynamo's source checks do not see), the
  server arguments that can shape a graph, every ``SGLANG_NEURON_*`` variable but the internal ones, and the torch,
  torch-neuronx, NKI and neuronx-cc versions.
* ``save`` writes the packages after each runtime's warm-up (graph_runner._warmup_all) and logs dynamo's precompile
  hit and miss counts, so a launch shows whether it restored its graphs. Saving is cumulative: the last save of a
  process holds every family it compiled.

Off by default. A package restore skips the trace, so whatever a trace would have run on the host (a model's
trace-time checks and logs) does not run on a warm launch."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

logger = logging.getLogger(__name__)

ENV = "SGLANG_NEURON_PRECOMPILE_CACHE"
_CACHE_DIR_ENV = "TORCHINDUCTOR_CACHE_DIR"
_FINGERPRINT_LEN = 32
# server arguments that only address, log, report or authorize: they shape no graph
_IGNORED_SERVER_ARGS = frozenset({
    "host", "port", "nccl_port", "dist_init_addr", "dist_timeout", "api_key", "admin_api_key", "served_model_name",
    "log_level", "log_level_http", "log_requests", "log_requests_level", "log_requests_format", "enable_metrics",
    "enable_metrics_for_all_schedulers", "enable_cache_report", "show_time_cost", "decode_log_interval",
    "bucket_time_to_first_token", "bucket_inter_token_latency", "bucket_e2e_request_latency",
    "collect_tokens_histogram",
    "disaggregation_bootstrap_port", "disaggregation_ib_device", "file_storage_path", "crash_dump_folder",
    "watchdog_timeout", "random_seed", "prompt_tokens_details", "enable_request_time_stats_logging",
    # the scheduler's policy alone: which batch runs next, not what a graph computes
    "prefill_decode_interval", "prefill_decode_interval_min_bs", "schedule_policy", "schedule_conservativeness",
})
# environment switches of the same kind: the scheduler's choices and what a launch records (SGLANG_NEURON_PROFILE_*)
_IGNORED_ENV_PREFIXES = ("SGLANG_NEURON_PROFILE_", "SGLANG_NEURON_PREFILL_YIELD", "SGLANG_NEURON_HOST_TRACE")

_enabled = False
_configured: Optional[str] = None


def requested(environ: Mapping[str, str] = os.environ) -> bool:
    """Whether the launch asks for the precompile cache: ``1`` on, ``0`` or unset off, any other value refused."""
    value = environ.get(ENV)
    if value is None or value == "0":
        return False
    if value != "1":
        raise ValueError(f"{ENV}={value!r}: set it to 0 or 1, or leave it unset")
    return True


def enable() -> bool:
    """Turn on torch's caching precompile and AOTAutograd cache, and torch-neuronx's precompile patches, when the
    launch asks for them. Before the first compile; idempotent. Returns whether the cache is on."""
    global _enabled
    if _enabled or not requested():
        return _enabled
    import torch._dynamo.config
    import torch._functorch.config
    from torch_neuronx.neuron_dynamo_backend import cache_artifact

    torch._dynamo.config.caching_precompile = True
    torch._dynamo.config.enable_aot_compile = True
    torch._functorch.config.enable_autograd_cache = True
    # torch-neuronx applies these at its import only when caching_precompile was already on then
    cache_artifact._apply_precompile_patches()
    _no_autograd_cache_hits()
    _enabled = True
    return True


def _no_autograd_cache_hits() -> None:
    """Give every graph its own AOTAutograd cache key, so no graph is served another's compiled output.

    The key hashes the FX graph, and an NKI kernel call in it (torch-neuronx's nki_hop) holds only the integers its
    kernel and constant arguments have in a process-global registry, numbered in the order the process traced them.
    Two graphs that call different kernels (or one kernel with other constants) under the same numbers hash alike: a
    MiMo-V2 launch's prefill warm-up was served a decode graph's output ("Output 0 has shape (9, 4) but the input it
    mutates has shape (1, 3)"), and ranks that traced their shapes in other orders were served different programs for
    one shape (the SPMD check refused them). A package restore does not look the cache up (it installs the backend
    output the package holds), so a key of its own per graph, the nonce torch already falls back to when a graph
    cannot be hashed, loses nothing the precompile cache needs."""
    import uuid

    from torch._functorch._aot_autograd import autograd_cache

    if getattr(autograd_cache.autograd_cache_key, "_sglang_neuron_nonce", False):
        return

    def autograd_cache_key(*args, **kwargs):
        return autograd_cache.AOTAUTOGRAD_CACHE_PREFIX + "nonce" + uuid.uuid4().hex, []

    autograd_cache_key._sglang_neuron_nonce = True
    autograd_cache.autograd_cache_key = autograd_cache_key


def _plain(value: Any) -> Any:
    """``value`` as JSON when it is made of plain values only, else None (an object's repr can name its address, which
    differs between processes)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        items = [_plain(v) for v in value]
        return None if any(i is None and v is not None for i, v in zip(items, value)) else items
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        items = {k: _plain(v) for k, v in value.items()}
        return None if any(items[k] is None and value[k] is not None for k in value) else items
    return None


def _server_args() -> Dict[str, Any]:
    """The launch's server arguments that can shape a graph, those of plain values (_plain)."""
    try:
        from sglang.srt.runtime_context import get_server_args

        args = get_server_args()
    except Exception:  # noqa: BLE001 - no server (a test's runtime): the fingerprint keeps the other inputs
        return {}
    items = vars(args) if args is not None and hasattr(args, "__dict__") else {}
    out = {}
    for key, value in sorted(items.items()):
        plain = _plain(value)
        if key not in _IGNORED_SERVER_ARGS and (plain is not None or value is None):
            out[key] = plain
    return out


def _version(dist: str) -> str:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return "-"


def _plugin_sources_digest() -> str:
    """The plugin's Python sources, every file: a traced function's source is checked by dynamo only for the frames it
    inlined, and the code around a graph (its inputs' layout, its wrappers) shapes it too."""
    from sglang_neuron.backend.compile_cache import PLUGIN_ROOT

    digest = hashlib.sha256()
    package = PLUGIN_ROOT / "sglang_neuron"
    for path in sorted(package.rglob("*.py")):
        digest.update(str(path.relative_to(package)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def fingerprint_inputs(environ: Mapping[str, str] = os.environ) -> Dict[str, Any]:
    """Every input of the cache directory's fingerprint (module docstring)."""
    import torch

    from sglang_neuron import env_registry
    from sglang_neuron.backend import compile_cache

    return {
        "plugin_sources": _plugin_sources_digest(),
        "kernel_sources": compile_cache.kernel_sources_digest(),
        "server_args": _server_args(),
        # the internal ones name this launch's ports and rendezvous, not its graphs
        "env": {k: v for k, v in sorted(environ.items())
                if k.startswith("SGLANG_NEURON_") and env_registry.KNOBS.get(k) != env_registry.INTERNAL
                and not k.startswith(_IGNORED_ENV_PREFIXES)},
        "versions": {"torch": torch.__version__, "torch-neuronx": _version("torch-neuronx"), "nki": _version("nki"),
                     "neuronx-cc": _version("neuronx-cc")},
    }


def fingerprint(inputs: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:_FINGERPRINT_LEN]


def cache_root(environ: Mapping[str, str] = os.environ) -> Path:
    """``precompile`` beside torch-neuronx's NEFF cache root (compile_cache.CACHE_DIRS), so the packages live and go
    with the NEFFs they name."""
    from sglang_neuron.backend.compile_cache import CACHE_DIRS

    name, default = CACHE_DIRS[0]
    return Path(environ.get(name) or default).resolve().parent / "precompile"


def configure() -> Optional[str]:
    """Point TORCHINDUCTOR_CACHE_DIR at this launch's package directory, before the first compiled frame; once per
    process. Returns the directory, or None with the cache off."""
    global _configured
    if not _enabled:
        return None
    if _configured is None:
        inputs = fingerprint_inputs()
        directory = cache_root() / fingerprint(inputs)
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[_CACHE_DIR_ENV] = str(directory)
        _configured = str(directory)
        # each input's own digest: two launches (or ranks) in different directories show which input differs
        parts = " ".join(f"{k}={fingerprint(v)[:8]}" for k, v in inputs.items())
        logger.info("[neuron] precompile cache at %s (%s)", directory, parts)
    return _configured


def _counters() -> Dict[str, int]:
    from torch._dynamo.utils import counters

    keys = (("dynamo_cache", "dynamo_cache_hit"), ("dynamo_cache", "dynamo_cache_miss"),
            ("dynamo_cache", "dynamo_cache_error"), ("aot_autograd", "autograd_cache_hit"),
            ("aot_autograd", "autograd_cache_miss"), ("aot_autograd", "autograd_cache_bypass"))
    return {f"{group}.{key}": int(counters[group][key]) for group, key in keys}


def save(family: str) -> int:
    """Save the packages of every graph this process compiled so far, after a runtime's warm-up, and log the restore
    counts. A failed save is logged, not raised: the server serves either way. Returns the packages saved."""
    if not _enabled:
        return 0
    try:
        import torch
        from torch._dynamo.precompile_context import PrecompileContext

        torch.compiler.save_cache_artifacts()
        saved = len(PrecompileContext.save_to_dynamo_cache().get("dynamo", ()))
    except Exception:  # noqa: BLE001
        logger.warning("[neuron] precompile cache: saving the %s packages failed", family, exc_info=True)
        return 0
    logger.info("[neuron] precompile cache: %s saved %d package(s) to %s; %s", family, saved,
                os.environ.get(_CACHE_DIR_ENV, "?"), _counters())
    return saved


def import_model_modules(model) -> None:
    """Import every module of the model packages ``model``'s submodules come from, before a package restore.

    A restored package installs its guards at ``torch.compile``, and a guard can read a module global through a
    package attribute (``G['__import_sglang_neuron_dot_models_dot_mimo_v2_dot_kernels'].attention_block``): the
    submodule is set on its package only once imported, which a traced launch did lazily inside the forward a restore
    does not run, so the guard fails with an AttributeError. Importing each ``sglang_neuron.models.<model>`` package
    whole sets every such attribute."""
    import importlib
    import pkgutil

    packages = set()
    for module in model.modules():
        parts = type(module).__module__.split(".")
        if len(parts) >= 3 and parts[:2] == ["sglang_neuron", "models"]:
            packages.add(".".join(parts[:3]))
    for name in sorted(packages):
        package = importlib.import_module(name)
        for info in pkgutil.walk_packages(getattr(package, "__path__", []), prefix=name + "."):
            try:
                importlib.import_module(info.name)
            except Exception:  # noqa: BLE001 - a module that cannot import here is one no graph read
                logger.debug("[neuron] precompile cache: %s did not import", info.name, exc_info=True)
