"""Report SGLang requirements the current environment does not satisfy.

Usage: python tools/check_sglang_deps.py <sglang>/python/pyproject.toml [overlay_dir]

Prints only requirements that are missing or out of range, first for the
installed environment and then with ``overlay_dir`` searched first (the
``pip install --no-deps --target`` overlay used with PYTHONPATH). Imports
nothing from torch, torch_neuronx or SGLang.
"""
import importlib.metadata as md
import sys
import tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def installed(name, paths):
    for dist in md.distributions(path=paths):
        if canonicalize_name(dist.metadata["Name"]) == canonicalize_name(name):
            return dist.version
    return None


def satisfied(req, version):
    return version is not None and req.specifier.contains(version, prereleases=True)


def main():
    pyproject = sys.argv[1]
    overlay = sys.argv[2:3]
    base = [p for p in sys.path if p]
    reqs = [Requirement(r) for r in tomllib.load(open(pyproject, "rb"))["project"]["dependencies"]]
    for req in reqs:
        if req.marker and not req.marker.evaluate():
            continue
        plain = installed(req.name, base)
        layered = installed(req.name, overlay + base)
        if satisfied(req, plain) and satisfied(req, layered):
            continue
        status = "OK with overlay" if satisfied(req, layered) else "UNMET"
        print(f"{req.name:26s} want {str(req.specifier) or '*':20s} "
              f"installed {plain or '-':16s} overlay {layered or '-':12s} {status}")


if __name__ == "__main__":
    main()
