# SPDX-License-Identifier: Apache-2.0
"""Environment variables of the SGLang Neuron diffusion plugin, read when accessed.

- ``NEURON_LOGICAL_NC_CONFIG``: logical NeuronCore size (LNC). It drives the compiler
  (set by the model's environment profile) and the NKI kernel launch degree.
- ``SGLANG_NEURON_REQUIRE_NKI_KERNELS``: ``1`` turns a torch fallback in a reporting NKI
  kernel gate (``kernels/kernel_choice.py``) into ``NkiKernelRequiredError``.
- ``SGLANG_NEURON_DISABLE_NKI_KERNELS``: ``1`` makes every NKI kernel gate take its torch
  fallback (``vn/utils/neuron_utils.can_run_kernel``); the kill switch for the NKI kernels.
- ``SGLANG_NEURON_WAN_VAE_PATCH_PARALLEL_SIZE``: the Wan2.2 VAE's patch-parallel group size
  (ranks ``0..size-1``, clamped to the world; 1, 2, 4, 8, 16 or a multiple of 32). Unset: the
  pipeline takes the reference degree for the task at the load-time resolution (the first
  ``--warmup-resolutions`` entry, else 480 x 832): 16 / 32 / 64 for T2V and 16 / 32 / 32 for I2V
  at 480p / 720p / 1080p.
- ``SGLANG_NEURON_LINGBOT_TIMING_FILE``: a path; the LingBot-World stage appends one JSON row per
  served chunk there (its engine-side timing, read by the serving e2e test).
- ``SGLANG_NEURON_WAN_SAVE_VIDEO_PT``: a ``.pt`` path; the Wan2.2 pipelines save each request's
  decoded frames there (``[T, 3, H, W]`` bf16 in [0, 1], the goldens' layout, which is
  bf16-exact), for the accuracy gate. ``{index}`` in the path numbers one file per request.
  Unset: nothing is saved.
- ``SGLANG_NEURON_WAN_SAVE_VIDEO_PT_INDICES``: comma list of the 0-based request indices (per
  process) to save, e.g. ``6`` for the last of 2 warm-ups + 5 runs. Unset: every request.

The Neuron layers under ``sglang_neuron.diffusion.vn`` read them here.
"""

import os
from collections.abc import Callable
from typing import Any


def _int_or_none(value: str | None) -> int | None:
    return None if value is None else int(value)


def _bool(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes", "on")


environment_variables: dict[str, Callable[[], Any]] = {
    "NEURON_LOGICAL_NC_CONFIG": lambda: _int_or_none(os.getenv("NEURON_LOGICAL_NC_CONFIG")),
    "SGLANG_NEURON_REQUIRE_NKI_KERNELS": lambda: _bool("SGLANG_NEURON_REQUIRE_NKI_KERNELS"),
    "SGLANG_NEURON_DISABLE_NKI_KERNELS": lambda: _bool("SGLANG_NEURON_DISABLE_NKI_KERNELS"),
    "SGLANG_NEURON_WAN_VAE_PATCH_PARALLEL_SIZE": lambda: _int_or_none(
        os.getenv("SGLANG_NEURON_WAN_VAE_PATCH_PARALLEL_SIZE") or None
    ),
    "SGLANG_NEURON_WAN_SAVE_VIDEO_PT": lambda: os.getenv("SGLANG_NEURON_WAN_SAVE_VIDEO_PT") or None,
    "SGLANG_NEURON_LINGBOT_TIMING_FILE": lambda: os.getenv("SGLANG_NEURON_LINGBOT_TIMING_FILE")
    or None,
    "SGLANG_NEURON_WAN_SAVE_VIDEO_PT_INDICES": lambda: _index_set(
        os.getenv("SGLANG_NEURON_WAN_SAVE_VIDEO_PT_INDICES")
    ),
}


def _index_set(value: str | None) -> frozenset[int] | None:
    """``"0,6"`` -> ``{0, 6}``; unset or empty -> ``None`` (every index)."""
    if value is None or not value.strip():
        return None
    return frozenset(int(part) for part in value.split(",") if part.strip())


def __getattr__(name: str) -> Any:
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(environment_variables.keys())


def compile_kwargs() -> dict:
    """``backend`` and ``options`` for ``torch.compile`` of the pipeline's graphs.

    One place for the pipeline and the module tests, so the tests compile what production
    compiles.
    """
    from sglang_neuron.diffusion import compile_backend

    return {"backend": compile_backend.register(), "options": {}}
