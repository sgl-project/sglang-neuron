#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Write or check ``sglang_neuron/sglang_patches/upstream_pins.json``: the md5 of each upstream SGLang source that the
modules of ``source_guard.PINNED_MODULES`` patch or rely on, computed from the SGLang tree on ``PYTHONPATH``.

    python tools/pin_upstream_sources.py --check   # exit 1 when a pin is missing, stale or differs from this SGLang
    python tools/pin_upstream_sources.py --write   # after re-validating those patches against a new SGLang revision

Run it in the serving environment (SGLang importable, ``SGLANG_NEURON_CPU_MODE=1`` on a host without a device).
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys


def targets() -> set:
    """Every (module, qualified name) the pinned modules name."""
    from sglang_neuron.sglang_patches import source_guard

    found = set()
    for module_name, tables in source_guard.PINNED_MODULES.items():
        module = importlib.import_module(module_name)
        for table in tables:
            found.update(getattr(module, table))
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    from sglang_neuron.sglang_patches import source_guard

    current = {f"{m}:{q}": source_guard.source_md5(source_guard.resolve(m, q)) for m, q in sorted(targets())}
    if args.write:
        recorded = json.loads(source_guard.PINS_FILE.read_text())
        recorded["targets"] = current
        source_guard.PINS_FILE.write_text(json.dumps(recorded, indent=1) + "\n")
        print(f"wrote {len(current)} pins to {source_guard.PINS_FILE}")
        return 0
    recorded = dict(source_guard.pins())
    problems = [f"{k}: pinned {recorded.get(k)}, this SGLang {v}" for k, v in current.items() if recorded.get(k) != v]
    problems += [f"{k}: pinned, but no module names it" for k in sorted(set(recorded) - set(current))]
    print("\n".join(problems) if problems else f"{len(current)} pins match this SGLang")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
