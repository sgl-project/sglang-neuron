# SPDX-License-Identifier: Apache-2.0
"""The Qwen3.5 kernel copies' file manifests.

Each copy under this package (``giga/``, ``serve/``, ``weights/``) is maintained here (PROVENANCE.md says where it was
forked from and lists every edit since). A copy's ``MANIFEST.json`` records the sha256 of each of its files, so an edit
that is not also recorded there fails the manifest check instead of reaching a NEFF
unnoticed. After an intended edit, list it in PROVENANCE.md and rewrite the manifest:

    python -m sglang_neuron.models.qwen35.kernels.manifests --write giga
    python -m sglang_neuron.models.qwen35.kernels.manifests --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

#: The kernel copies the Qwen3.5 decode and prefill run, and the packer of their weights.
COPIES = ("giga", "serve", "weights")
MANIFEST = "MANIFEST.json"
_HERE = Path(__file__).resolve().parent


def _files(copy_dir: Path) -> dict[str, str]:
    """``{relative path: sha256}`` of every file of a copy but its manifest and Python's caches."""
    out = {}
    for path in sorted(copy_dir.rglob("*")):
        rel = path.relative_to(copy_dir)
        if not path.is_file() or rel.as_posix() == MANIFEST or "__pycache__" in rel.parts:
            continue
        out[rel.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def recorded(copy_dir: Path) -> dict[str, str]:
    """The ``{relative path: sha256}`` a copy's manifest records."""
    return json.loads((copy_dir / MANIFEST).read_text())["files"]


def problems(copy_dir: Path) -> list[str]:
    """What of a copy its manifest does not record, one line each: a file changed, missing or not listed."""
    name = copy_dir.name
    if not (copy_dir / MANIFEST).is_file():
        return [f"{name}: no {MANIFEST}"]
    want, have = recorded(copy_dir), _files(copy_dir)
    out = [f"{name}/{rel}: changed" for rel in sorted(want.keys() & have.keys()) if want[rel] != have[rel]]
    out += [f"{name}/{rel}: missing" for rel in sorted(want.keys() - have.keys())]
    out += [f"{name}/{rel}: not in {MANIFEST}" for rel in sorted(have.keys() - want.keys())]
    return out


def write(copy_dir: Path) -> None:
    """Record every file of a copy as it is now."""
    body = {"note": "sha256 of every file of this kernel copy; ../PROVENANCE.md lists the edits since the fork",
            "files": _files(copy_dir)}
    (copy_dir / MANIFEST).write_text(json.dumps(body, indent=1, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="exit 1 if a copy differs from its manifest")
    mode.add_argument("--write", action="store_true", help="rewrite the named copies' manifests")
    parser.add_argument("copies", nargs="*", metavar="copy", help=f"one of {', '.join(COPIES)} (default: all)")
    args = parser.parse_args(argv)
    unknown = sorted(set(args.copies) - set(COPIES))
    if unknown:
        parser.error(f"not a Qwen3.5 kernel copy: {', '.join(unknown)}")
    copies = args.copies or list(COPIES)
    if args.write:
        for name in copies:
            write(_HERE / name)
        return 0
    found = [line for name in copies for line in problems(_HERE / name)]
    print("\n".join(found) or "every copy matches its manifest")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
