# SPDX-License-Identifier: Apache-2.0
"""The one place the Kimi-K3 decode imports its NKI kernels and host layout builders from.

The kernels are vendored under ``KERNEL_ROOT`` (``models/kimi_k3/kernels/decode/``, its MANIFEST.json). ``KERNELS`` maps
each device kernel to its module under the root and its entry; ``HOST_MODULES`` maps each host module the decode imports
by name to its module. Importing this module never imports ``nki``. The first accessor call outside a traced graph (or
``prepare()``) imports every kernel, and raises ``KernelUnavailable`` naming the module when one does not import. Inside
a traced forward the accessors only read the loaded kernels from a dict, which dynamo reads as a constant; a kernel that
was not loaded before the trace raises ``KernelUnavailable`` there. Most host builders import ``nki`` too (through the
kernel modules whose constants they use), so they need a host with ``nki``.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import torch

KERNEL_ROOT = "sglang_neuron.models.kimi_k3.kernels.decode"

KERNELS = {
    "verify": ("multinode.verify_model_2srv", "kimi_k3_verify_model_2srv"),
    "draft_mla_step": ("draft_mla", "kimi_k3_draft_mla_step"),
    "full_vocab_sample": ("sampling.full_vocab_sample", "full_vocab_sample"),
    "kda_a_exp_init": ("tkg_kda", "kimi_k3_kda_a_exp_init_kernel"),
}
"""key -> (module under ``KERNEL_ROOT``, entry) of each device kernel of the decode programs."""

HOST_MODULES = {
    "draft_mla_torch": "draft_mla_torch",
}
"""name -> module under ``KERNEL_ROOT`` of each host module the decode imports by name (the weight layout builders
are ``weights_host.HOST_MODULES``)."""


class KernelUnavailable(ImportError):
    """A Kimi-K3 decode kernel or host module does not import on this host."""


_LOADED: dict[str, Any] = {}
"""key -> kernel entry, filled by ``prepare``. A plain dict, not a function cache: dynamo traces through the body of a
cached function (and its imports), but reads a dict entry as a constant."""
_HOST: dict[str, Any] = {}
"""name -> host module, filled by ``host_module``."""


def kernel_root_dir() -> Path:
    """The directory of ``KERNEL_ROOT`` (``models/kimi_k3/kernels/decode/``)."""
    return Path(__file__).resolve().parents[1] / "kernels" / "decode"


def _import(module: str, what: str) -> Any:
    path = f"{KERNEL_ROOT}.{module}"
    try:
        return importlib.import_module(path)
    except ImportError as err:
        raise KernelUnavailable(f"The Kimi-K3 decode {what} {path} is not available ({err!r}). The kernels are "
                                f"vendored under {KERNEL_ROOT}; run from a checkout that has them, on a host with "
                                "nki.") from err


def _load(key: str) -> Any:
    module, name = KERNELS[key]
    entry = getattr(_import(module, "kernel"), name, None)
    if entry is None:
        raise KernelUnavailable(f"The Kimi-K3 decode kernel module {KERNEL_ROOT}.{module} has no {name}.")
    return entry


def prepare() -> None:
    """Import every kernel of ``KERNELS``; call outside any traced forward.

    Raises:
        KernelUnavailable: A kernel module does not import, or has no such entry.
    """
    for key in KERNELS:
        if key not in _LOADED:
            _LOADED[key] = _load(key)


def import_kernel(key: str) -> Any:
    """The entry of kernel ``key`` (``KERNELS``); outside a trace, the first call runs ``prepare``.

    Raises:
        KernelUnavailable: A kernel does not import, or a traced forward asks for one before ``prepare`` ran.
    """
    kernel = _LOADED.get(key)
    if kernel is not None:
        return kernel
    if torch.compiler.is_compiling():
        raise KernelUnavailable(f"The Kimi-K3 decode kernel {key} was not loaded before the trace: call "
                                "kernel_imports.prepare() when the decode programs are built.")
    prepare()
    return _LOADED[key]


def host_module(name: str) -> Any:
    """The host module ``name`` (``HOST_MODULES``), imported on first use; call outside any traced forward.

    Raises:
        KernelUnavailable: The module does not import.
    """
    module = _HOST.get(name)
    if module is None:
        module = _HOST[name] = _import(HOST_MODULES[name], "host module")
    return module


def kernels_available(key: str = "verify") -> bool:
    """True when the module of kernel ``key`` imports."""
    try:
        importlib.import_module(f"{KERNEL_ROOT}.{KERNELS[key][0]}")
    except ImportError:
        return False
    return True


def verify_entry() -> Any:
    """``kimi_k3_verify_model_2srv``: the 2-server verify (93 layers, tail, drafter aux states) of one replica."""
    return import_kernel("verify")


def draft_mla_step() -> Any:
    """``kimi_k3_draft_mla_step``: the DSpark MLA drafter step (context rows, 5 MLA layers, tail, draft sampling)."""
    return import_kernel("draft_mla_step")


def full_vocab_sample() -> Any:
    """``full_vocab_sample``: the vocab-sharded target sampler (collectives over the TP group inside the kernel)."""
    return import_kernel("full_vocab_sample")


def kda_a_exp_init() -> Any:
    """``kimi_k3_kda_a_exp_init_kernel``: the wide ``kda_params`` stack ``[L, 128, W + Hl]`` of a plain stack."""
    return import_kernel("kda_a_exp_init")
