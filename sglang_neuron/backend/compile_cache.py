# SPDX-License-Identifier: Apache-2.0
"""Refuse a compile cache built from other NKI kernel sources.

torch-neuronx keys a cached NEFF by its graph's StableHLO and its own build, not by the NKI kernel sources the graph
calls or the kernel library (nkilib) version, so a cache built before a kernel changed would serve the old kernel. Each
cache root records the digest of the kernel sources that built it (``KERNEL_STAMP``, written by the first launch on an
empty cache). A launch with other sources is refused, as is a non-empty cache with no stamp. The roots are
torch-neuronx's ``TORCH_NEURONX_NEFF_CACHE_DIR`` and ``TORCH_NEURONX_HLO_CACHE_DIR``.

The kernel sources (``kernel_sources``) are each plugin file that mentions nki or a kernel library (nkilib); each
plugin module a kernel's code imports, directly or through other plugin modules (its code runs inside the trace; the
imports of a file that only calls kernels are host code and are not followed); and every file of a vendored kernel
tree (a model's ``kernels/`` package, ``VENDORED_KERNELS``) but the repository's records beside it
(``is_kernel_record``: its Markdown and file manifests, which the wheel does not ship, so a checkout and an install of
the same commit stamp alike). The match errs wide on purpose: a file
that names a kernel library may set what the library reads when it traces a kernel (a model's kernel build settings),
which no key covers.

The compiler needs no stamp: torch-neuronx's key holds neuronx-cc's version, and the release compiles with the pinned
neuronx-cc only (release/neuron_sdk.toml). The key is true only while the compiler on PATH runs the package whose
version it names, which a model's launch rules check before the first compile
(release.toolchain.Toolchain.same_install)."""

from __future__ import annotations

import ast
import hashlib
import logging
import os
import re
from importlib import metadata
from pathlib import Path
from typing import Iterable, Iterator, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

KERNEL_STAMP = "KERNEL_SOURCES"
# torch-neuronx's compile caches: (variable, default when unset)
CACHE_DIRS = (("TORCH_NEURONX_NEFF_CACHE_DIR", "/tmp/neff_cache"), ("TORCH_NEURONX_HLO_CACHE_DIR", "/tmp/hlo_cache"))
# a file that defines an NKI kernel (an nki.jit kernel, or code written in nki's language)
_DEFINES_KERNEL = rb"nki\.jit|@nki\.|nki\.language as|^\s*(?:import|from) nki\b"
NKI_KERNEL = re.compile(_DEFINES_KERNEL, re.M)
# a file that defines a kernel or calls one; it also counts when it names a kernel library at all (upstream's wide match)
NKI_SOURCE = re.compile(_DEFINES_KERNEL + rb"|\bnkilib(?:_src)?\b", re.M)
# the vendored kernel trees, a model's own kernels/ package (sglang_neuron/models/<model>/kernels): every file counts,
# whatever its type, but the repository's records beside the kernels
VENDORED_KERNELS = "sglang_neuron/models/*/kernels"
# those records: the kernel trees' file manifests (tools/<model>/kernel_manifests.py), which pyproject.toml's
# exclude-package-data keeps out of the wheel, and their Markdown (provenance notes, READMEs; .md in any case), which no
# package data takes. The two file names are reserved for the manifests, in any package (the wheel leaves out every
# file of these names); every such file sits at a kernel copy's root
KERNEL_RECORDS = ("MANIFEST.json", "SHARED_MANIFEST.json")
# this module, in any checkout: it names nki in its own match, and builds no kernel
THIS_MODULE = Path("sglang_neuron") / "backend" / "compile_cache.py"
PLUGIN_ROOT = Path(__file__).resolve().parents[2]


def kernel_library_versions() -> List[str]:
    """``<distribution>=<version>`` for each installed distribution that ships nkilib (neuronx-cc on the serving image),
    or ``nkilib=-`` when none does."""
    owners = sorted(set(metadata.packages_distributions().get("nkilib", [])))
    return [f"{dist}={metadata.version(dist)}" for dist in owners] or ["nkilib=-"]


def _module_file(root: Path, dotted: str) -> Optional[Path]:
    """The file under ``root`` of the plugin module or package ``dotted``, or None for a module outside the plugin (or
    a name that is not a module)."""
    parts = dotted.split(".")
    if parts[0] != "sglang_neuron":
        return None
    path = Path(root).joinpath(*parts)
    for candidate in (path / "__init__.py", path.with_suffix(".py")):
        if candidate.is_file():
            return candidate
    return None


def _plugin_imports(root: Path, path: Path) -> Iterator[Path]:
    """The files of the plugin modules that the module at ``path`` imports, at module level or inside a function,
    absolutely or relatively. ``from <package> import <name>`` is the submodule ``<name>`` where there is one, and
    otherwise the package's own ``__init__``."""
    parts = path.relative_to(root).with_suffix("").parts
    package = parts[:-1]  # a module's package, and an __init__'s own
    for node in ast.walk(ast.parse(path.read_bytes(), filename=str(path))):
        if isinstance(node, ast.Import):
            found = [_module_file(root, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level > len(package):
                continue
            base = package[:len(package) - node.level + 1] if node.level else ()
            module = ".".join(base + tuple(node.module.split(".") if node.module else ()))
            found = [_module_file(root, f"{module}.{alias.name}") or _module_file(root, module)
                     for alias in node.names]
        else:
            continue
        yield from (file for file in found if file is not None)


def _imports_past_vendored(tree: Path, path: Path, data: bytes) -> bool:
    """Whether the module at ``path`` in the vendored tree ``tree`` may import a plugin module outside it: it names the
    plugin package, or a relative import climbs out of the tree (one dot is the module's own package, so it takes one
    more dot than the module's depth below the tree). Its imports inside the tree need no parse, since every file there
    counts."""
    dots = len(path.parent.relative_to(tree).parts) + 2
    return b"sglang_neuron" in data or re.search(rb"^\s*from\s+\.{%d}" % dots, data, re.M) is not None


def is_kernel_record(path: Path) -> bool:
    """Whether a file of a vendored kernel tree is one of the repository's records of it, not a source."""
    return path.suffix.lower() == ".md" or path.name in KERNEL_RECORDS


def kernel_sources(root: Path = PLUGIN_ROOT) -> List[Path]:
    """The NKI kernel sources under ``root``/sglang_neuron (the module docstring's three kinds), sorted."""
    root = Path(root)
    trees = [tree for tree in sorted(root.glob(VENDORED_KERNELS)) if tree.is_dir()]

    def vendored(path: Path) -> Optional[Path]:
        return next((tree for tree in trees if tree in path.parents), None)

    sources, kernels = set(), []
    for path in (root / "sglang_neuron").rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts or path.relative_to(root) == THIS_MODULE:
            continue
        tree = vendored(path)
        if tree is not None:
            if not is_kernel_record(path):
                sources.add(path)
                if path.suffix == ".py" and _imports_past_vendored(tree, path, path.read_bytes()):
                    kernels.append(path)
        elif path.suffix == ".py":
            data = path.read_bytes()
            if NKI_SOURCE.search(data):
                sources.add(path)
                if NKI_KERNEL.search(data):
                    kernels.append(path)
    # every plugin module a kernel's code reaches through its imports (a vendored one already counts, and is parsed
    # above when it may import past its tree)
    reached = set(kernels)
    while kernels:
        for module in _plugin_imports(root, kernels.pop()):
            if module not in reached and module.relative_to(root) != THIS_MODULE:
                reached.add(module)
                if vendored(module) is None:
                    kernels.append(module)
    return sorted(sources | reached)


def kernel_sources_digest(root: Path = PLUGIN_ROOT, libraries: Optional[Iterable[str]] = None) -> str:
    """md5 over the NKI kernel sources under ``root``/sglang_neuron (``kernel_sources``: path and content of each) and
    the kernel library versions (``kernel_library_versions``)."""
    digest = hashlib.md5()
    for path in kernel_sources(root):
        digest.update(str(path.relative_to(root)).encode() + b"\0" + hashlib.md5(path.read_bytes()).digest())
    for entry in (kernel_library_versions() if libraries is None else libraries):
        digest.update(f"{entry}\n".encode())
    return digest.hexdigest()


def _held_file(root: Path) -> Optional[Path]:
    """A file under the cache root outside hidden entries (bookkeeping, not compiler output), or None."""
    if not root.is_dir():
        return None
    for entry in sorted(root.iterdir()):
        if entry.name.startswith(".") or entry.name == KERNEL_STAMP:
            continue
        if entry.is_file():
            return entry
        found = next((f for f in entry.rglob("*") if f.is_file()), None) if entry.is_dir() else None
        if found is not None:
            return found
    return None


def check_cache(root: Path, digest: str) -> str:
    """Refuse a root stamped with other kernel sources, or holding files without a stamp; stamp an empty one atomically,
    since ranks start together. Returns the line to log."""
    stamp = root / KERNEL_STAMP
    if stamp.exists():
        words = stamp.read_text().split()
        built = words[0] if words else ""
        if built != digest:
            raise RuntimeError(f"the compile cache {root} was built from other NKI kernel sources or another kernel "
                               f"library ({KERNEL_STAMP} {built}, this launch {digest}): a NEFF's cache key covers "
                               "neither, so its NEFFs would run the old kernels; launch with another cache directory, "
                               "or remove this one")
        return f"compile cache {root}: kernel sources {digest}"
    found = _held_file(root)
    if found is not None:
        raise RuntimeError(f"the compile cache {root} holds {found} but no {KERNEL_STAMP}: nothing records the NKI "
                           "kernel sources or kernel library it was built from, and a NEFF's cache key covers neither, "
                           "so its NEFFs could run old kernels; launch with another cache directory, or remove this one")
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / f".{KERNEL_STAMP}.{os.getpid()}"
    tmp.write_text(digest + "\n")
    os.replace(tmp, stamp)
    return f"compile cache {root}: kernel sources {digest} (stamped)"


def cache_roots(environ: Mapping[str, str] = os.environ) -> List[Path]:
    """torch-neuronx's compile cache roots for this process, each once."""
    roots: List[Path] = []
    for name, default in CACHE_DIRS:
        root = Path(environ.get(name) or default)
        if root not in roots:
            roots.append(root)
    return roots


def check_compile_caches(environ: Mapping[str, str] = os.environ,
                         digest: Optional[str] = None) -> List[Tuple[Path, str]]:
    """Check (and stamp where empty) every compile cache root this process compiles into, before its first compile."""
    digest = digest or kernel_sources_digest()
    checked = []
    for root in cache_roots(environ):
        line = check_cache(root, digest)
        logger.info("[neuron] %s", line)
        checked.append((root, line))
    return checked


def main(argv=None) -> int:
    """``python -m sglang_neuron.backend.compile_cache stamp <root>...``: stamp existing roots with this checkout's
    kernel sources, for a cache the operator knows they built (e.g. one filled before stamps existed). A launch only
    ever stamps an empty root."""
    import argparse

    parser = argparse.ArgumentParser(prog="python -m sglang_neuron.backend.compile_cache")
    sub = parser.add_subparsers(dest="command", required=True)
    stamp = sub.add_parser("stamp", help="stamp cache roots built from this checkout's NKI kernel sources")
    stamp.add_argument("roots", nargs="+", type=Path)
    args = parser.parse_args(argv)
    digest = kernel_sources_digest()
    for root in args.roots:
        if not root.is_dir():
            parser.error(f"{root} is not a directory")
        tmp = root / f".{KERNEL_STAMP}.{os.getpid()}"
        tmp.write_text(digest + "\n")
        os.replace(tmp, root / KERNEL_STAMP)
        print(f"{root}: {KERNEL_STAMP} {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
