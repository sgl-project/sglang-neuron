# SPDX-License-Identifier: Apache-2.0
"""The Kimi-K3 kernel copies' file manifests, a maintainer's tool run from a checkout.

Each copy under sglang_neuron/models/kimi_k3/kernels/ (``prefill/``, ``decode/``) is a fork of Kimi-K3 NKI kernels,
maintained there (PROVENANCE.md there lists every edit). A copy's ``MANIFEST.json`` records the sha256 of each of its
files, so an edit that is not also recorded there fails ``--check`` instead of reaching a NEFF unnoticed. The repository keeps these records beside the kernels; the wheel ships none of them.
After an intended edit, list it in PROVENANCE.md and rewrite the manifest:

    python tools/kimi_k3/kernel_manifests.py --write decode
    python tools/kimi_k3/kernel_manifests.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

#: The kernel copies the Kimi-K3 prefill and decode run.
COPIES = ("decode", "prefill")
MANIFEST = "MANIFEST.json"
#: Where the copies are, in this checkout.
ROOT = Path(__file__).resolve().parents[2]
KERNELS = ROOT / "sglang_neuron" / "models" / "kimi_k3" / "kernels"


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
    body = {"note": "sha256 of every file of this kernel copy; ../PROVENANCE.md lists the edits to the fork",
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
        parser.error(f"not a Kimi-K3 kernel copy: {', '.join(unknown)}")
    copies = args.copies or list(COPIES)
    if args.write:
        for name in copies:
            write(KERNELS / name)
        return 0
    found = [line for name in copies for line in problems(KERNELS / name)]
    print("\n".join(found) or "every copy matches its manifest")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
