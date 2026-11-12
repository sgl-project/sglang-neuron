# SPDX-License-Identifier: Apache-2.0
"""The neuronx-cc toolchain this process compiles with, and its digest.

torch-neuronx compiles with the ``neuronx-cc`` first on PATH but keys the cached NEFF on the version of the neuronxcc
package this process imported. The key is right only when both are one install, so the toolchain records both sides
and ``Toolchain.same_install`` checks that the PATH compiler's interpreter imports the same package.

The digest covers the compiler path, its package, and the neuronx-cc, nki and kernel library versions. It reads the
environment it is given, so compute it with the PATH and PYTHONPATH the launch compiles with.

``tools/neuron_release.py digest`` prints the digest, ``--show`` the fields as well."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shlex
import shutil
import subprocess
from typing import Mapping, Optional

# what the compiler's interpreter prints: the directory of the neuronxcc package it imports
_PACKAGE_PROBE = "import os, neuronxcc; print(os.path.dirname(os.path.realpath(neuronxcc.__file__)))"
# the most of the compiler file read for its interpreter line
_HEAD_BYTES = 1 << 12


@dataclasses.dataclass(frozen=True)
class Toolchain:
    # the neuronx-cc PATH resolves first, which torch-neuronx runs ("" when there is none)
    neuronx_cc: str
    # the neuronxcc package directory that compiler runs (its own interpreter's import of it, in this environment);
    # None when the PATH entry is no Python console script, or its interpreter cannot import the package
    compiler_package: Optional[str]
    # the neuronxcc package directory this process imports (None when it is not installed)
    imported_package: Optional[str]
    # the imported neuronxcc package's __version__, the string torch-neuronx keys its NEFFs on
    neuronxcc_version: str
    nki: str
    nkilib: str

    @property
    def same_install(self) -> bool:
        """Whether the compiler on PATH runs the neuronxcc package this process imports, the one the NEFF cache key
        names."""
        return self.compiler_package is not None and self.compiler_package == self.imported_package


def interpreter(path: str) -> Optional[str]:
    """The interpreter of a console script at ``path`` (its ``#!`` line), or None for a file without one: pip writes
    every console script, neuronx-cc's included, with the interpreter it installed for on that line. When that path is
    too long for a ``#!`` line or holds a space, pip writes ``#!/bin/sh`` and runs the interpreter from the next line,
    ``'''exec' "<python>" "$0" "$@"``."""
    try:
        with open(path, "rb") as f:
            head = f.read(_HEAD_BYTES).decode("utf-8", "replace")
    except OSError:
        return None
    if not head.startswith("#!"):
        return None
    lines = head[2:].splitlines()
    words = lines[0].split() if lines else []
    if words == ["/bin/sh"] and len(lines) > 1 and lines[1].startswith("'''exec' "):
        try:
            words = shlex.split(lines[1])[1:]
        except ValueError:
            return None
    if not words:
        return None
    # ``#!/usr/bin/env python3`` names its interpreter through env
    if os.path.basename(words[0]) == "env" and len(words) > 1:
        return shutil.which(words[1])
    return words[0]


def compiler_package(compiler: str, environ: Mapping[str, str]) -> Optional[str]:
    """The neuronxcc package directory the console script ``compiler`` runs: its interpreter, started with ``environ``
    (the PYTHONPATH a compile inherits), imports it. None for a file that is no Python console script, or whose
    interpreter cannot import the package. A probe that does not run to its end (an interpreter that cannot start, or
    no answer in 120 s on a loaded host) raises RuntimeError: it says nothing about the install, and a None in its place
    would read as another toolchain, on a prefill rank as a fleet mismatch that names no cause."""
    python = interpreter(compiler)
    if python is None:
        return None
    try:
        done = subprocess.run([python, "-c", _PACKAGE_PROBE], env=dict(environ), capture_output=True, text=True,
                              timeout=120)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(f"could not probe the neuronxcc package the compiler {compiler} runs ({python}): {e}") from e
    lines = done.stdout.strip().splitlines()
    return lines[-1] if done.returncode == 0 and lines else None


def _imported(module: str) -> tuple:
    """``(package directory, __version__)`` of ``module`` as this process imports it, or ``(None, "-")`` where it is
    not installed. A package that is installed but fails to import (a missing library or dependency) raises its own
    error: a launch's toolchain check would otherwise call it another toolchain and send the operator to install one."""
    import importlib

    try:
        package = importlib.import_module(module)
    except ModuleNotFoundError as e:
        if e.name != module:
            raise
        return None, "-"
    return os.path.dirname(os.path.realpath(package.__file__)), str(package.__version__)


def compiler_toolchain(environ: Mapping[str, str] = os.environ) -> Toolchain:
    """The toolchain a compile of this process runs: the neuronx-cc on ``environ``'s PATH and the package it runs, the
    package this process imports, and the versions."""
    from sglang_neuron.backend import compile_cache

    compiler = shutil.which("neuronx-cc", path=environ.get("PATH", "")) or ""
    imported, version = _imported("neuronxcc")
    return Toolchain(
        neuronx_cc=compiler,
        compiler_package=compiler_package(compiler, environ) if compiler else None,
        imported_package=imported,
        neuronxcc_version=version,
        nki=_imported("nki")[1],
        nkilib=";".join(compile_cache.kernel_library_versions()),
    )


def toolchain_digest(tc: Optional[Toolchain] = None) -> str:
    """sha256[:16] of every field of ``tc`` (this process's toolchain when None)."""
    tc = compiler_toolchain() if tc is None else tc
    text = json.dumps(dataclasses.asdict(tc), sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16]

