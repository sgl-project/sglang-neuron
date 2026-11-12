# SPDX-License-Identifier: Apache-2.0
"""The GLM-5.2 kernel copies' file manifests, a maintainer's tool run from a checkout.

Each copy under sglang_neuron/models/glm52/kernels/ is a fork of the GLM-5.2 kernels the plugin was built from, edited
in place for the released toolchain (PROVENANCE.md there lists every edit). A copy's ``MANIFEST.json`` records the
sha256 of each of its files, so an edit that is not also recorded there fails ``--check`` instead of reaching a NEFF
unnoticed; ``SHARED_MANIFEST.json`` records the sources both copies trace (``shared``).
The repository keeps these records beside the kernels; the wheel ships none of them. After an intended edit, list it
in PROVENANCE.md and rewrite the manifest:

    python tools/glm52/kernel_manifests.py --write decode
    python tools/glm52/kernel_manifests.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

#: The kernel copies the served points run (models/glm52/points.py and ops/).
COPIES = ("decode", "prefill")
MANIFEST = "MANIFEST.json"
#: Where the copies are, in this checkout.
ROOT = Path(__file__).resolve().parents[2]
KERNELS = ROOT / "sglang_neuron" / "models" / "glm52" / "kernels"
#: The record of the sources both copies trace (``SHARED_SOURCES``), beside them.
SHARED = "shared"
SHARED_MANIFEST = "SHARED_MANIFEST.json"


def _files(copy_dir: Path) -> dict[str, str]:
    """``{relative path: sha256}`` of every file of a copy but its manifest and Python's caches."""
    out = {}
    for path in sorted(copy_dir.rglob("*")):
        rel = path.relative_to(copy_dir)
        if not path.is_file() or rel.as_posix() == MANIFEST or "__pycache__" in rel.parts:
            continue
        out[rel.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def _shared_files(kernels: Path) -> dict[str, str]:
    """``{name: sha256}`` of the shared kernel sources."""
    from sglang_neuron.models.glm52.kernels import SHARED_SOURCES

    return {name: hashlib.sha256((kernels / name).read_bytes()).hexdigest()
            for name in SHARED_SOURCES if (kernels / name).is_file()}


def problems(copy_dir: Path) -> list[str]:
    """What of a copy its manifest does not record, one line each: a file changed, missing or not listed. The copy
    ``shared`` is the shared sources, recorded in ``SHARED_MANIFEST.json`` beside them."""
    if copy_dir.name == SHARED:
        manifest, name, have = copy_dir.parent / SHARED_MANIFEST, SHARED, _shared_files(copy_dir.parent)
    else:
        manifest, name, have = copy_dir / MANIFEST, copy_dir.name, None
    if not manifest.is_file():
        return [f"{name}: no {manifest.name}"]
    want = json.loads(manifest.read_text())["files"]
    have = _files(copy_dir) if have is None else have
    out = [f"{name}/{rel}: changed" for rel in sorted(want.keys() & have.keys()) if want[rel] != have[rel]]
    out += [f"{name}/{rel}: missing" for rel in sorted(want.keys() - have.keys())]
    out += [f"{name}/{rel}: not in {manifest.name}" for rel in sorted(have.keys() - want.keys())]
    return out


def write(copy_dir: Path) -> None:
    """Record every file of a copy (or the shared sources, for ``shared``) as it is now."""
    if copy_dir.name == SHARED:
        body = {"note": "sha256 of the kernel sources both copies trace; PROVENANCE.md lists the edits",
                "files": _shared_files(copy_dir.parent)}
        target = copy_dir.parent / SHARED_MANIFEST
    else:
        body = {"note": "sha256 of every file of this kernel copy; ../PROVENANCE.md lists the edits to the fork",
                "files": _files(copy_dir)}
        target = copy_dir / MANIFEST
    target.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(ROOT))  # the checkout's sglang_neuron, for SHARED_SOURCES
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="exit 1 if a copy differs from its manifest")
    mode.add_argument("--write", action="store_true", help="rewrite the named copies' manifests")
    names = (*COPIES, SHARED)
    parser.add_argument("copies", nargs="*", metavar="copy", help=f"one of {', '.join(names)} (default: all)")
    args = parser.parse_args(argv)
    unknown = sorted(set(args.copies) - set(names))
    if unknown:
        parser.error(f"not a served kernel copy: {', '.join(unknown)}")
    copies = args.copies or list(names)
    if args.write:
        for name in copies:
            write(KERNELS / name)
        return 0
    found = [line for name in copies for line in problems(KERNELS / name)]
    print("\n".join(found) or "every copy matches its manifest")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
