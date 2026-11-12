#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The DMA-order schedule of the GPT-OSS giga-kernel decode graph: find the kernel's call scope in a compiled decode
NEFF, rewrite a schedule to it, and report how many of the schedule's constraints name an instruction of the NEFF.

The compiler binds each constraint of a schedule (sglang_neuron/models/gpt_oss/giga/dma_order.json) to a backend
instruction by name. A name ends in the scope of the kernel call in the decode graph, ``inst__I-<K>-0``, where ``K`` is
the call's index in the graph, so it depends on every operation the graph traces before the kernel. The compiler skips
a constraint whose name it doesn't find without an error, so a schedule for another graph compiles to an unscheduled
kernel. The kernel library builds some DMAs only under the flags the schedule was derived with (the giga decode
settings in models/catalog.py), so a NEFF compiled without them lacks those names; ``check`` lists the missing names by
prefix. It exits 0 when the NEFF names at least the share of constraints the server requires (MIN_BOUND_SHARE in
models/gpt_oss/giga/schedule.py): the compiler folds some small loads into one DMA under one of their names.

usage: python3 tools/giga_schedule.py scope <decode.neff>
       python3 tools/giga_schedule.py rewrite <schedule.json> <K> <out.json>
       python3 tools/giga_schedule.py check <schedule.json> <decode.neff>
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sglang_neuron.models.gpt_oss.giga.schedule import (  # noqa: E402
    MIN_BOUND_SHARE,
    bound,
    kernel_scope as _blob_scope,
    neff_blob as _neff_blob_bytes,
    rewrite,
)


def neff_blob(path: str) -> bytes:
    """Every member of a NEFF file, concatenated."""
    with open(path, "rb") as f:
        return _neff_blob_bytes(f.read())


def kernel_scope(neff: str) -> int:
    """The kernel-call scope of a NEFF file."""
    return _blob_scope(neff_blob(neff))


def check(schedule: dict, neff: str):
    """(constraints, constraints whose instruction the NEFF names, missing names grouped by their prefix)."""
    return bound(schedule, neff_blob(neff))


def main(argv) -> int:
    if len(argv) == 2 and argv[0] == "scope":
        print(kernel_scope(argv[1]))
        return 0
    if len(argv) == 4 and argv[0] == "rewrite":
        with open(argv[1]) as f:
            out = rewrite(json.load(f), int(argv[2]))
        with open(argv[3], "w") as f:
            json.dump(out, f)
        print(argv[3])
        return 0
    if len(argv) == 3 and argv[0] == "check":
        with open(argv[1]) as f:
            total, named, groups = check(json.load(f), argv[2])
        print(f"{named} of {total} constraints name an instruction of {argv[2]} (the server requires "
              f"{MIN_BOUND_SHARE:.0%})")
        for name, n in groups.most_common(12):
            print(f"  missing {n:5d} {name}")
        return 0 if named >= MIN_BOUND_SHARE * total else 1
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
