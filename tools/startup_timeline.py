#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Where a launch's start-up time went, from its server log (rank 0's lines and the server's own).

    python3 tools/startup_timeline.py server.log [more.log ...]

Prints one row per phase: process start to the weight load, the weight load (and the converted-experts cache), the
KV pool, each graph family's trace and warm-up (and whether the precompile cache restored it), and the rest up to
"fired up". Every number is wall time between two log lines, so it reads any launch: single node, a PD role, cold
or warm."""

from __future__ import annotations

import re
import sys
from datetime import datetime

STAMP = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)(?: (TP\d+)[^\]]*)?\] (.*)")
EVENTS = (
    ("load_begin", re.compile(r"Load weight begin")),
    ("load_end", re.compile(r"Load weight end\. elapsed=([\d.]+) s")),
    ("pool_end", re.compile(r"Memory pool end")),
    ("restored", re.compile(r"\[neuron\] (\w+) graphs: dynamo package restored")),
    ("not_restored", re.compile(r"\[neuron\] (\w+) graphs: the precompile package did not restore")),
    ("traced", re.compile(r"\[neuron\] traced (\d+) shapes in ([\d.]+)s")),
    ("warm", re.compile(r"\[neuron\] warmup done in ([\d.]+)s \((\d+) shapes\)")),
    ("saved", re.compile(r"\[neuron\] precompile cache: (\w+) saved (\d+) package")),
    ("ready", re.compile(r"fired up and ready to roll")),
)


def parse(path: str):
    first, events = None, []
    with open(path, errors="replace") as f:
        for line in f:
            m = STAMP.match(line)
            if not m:
                continue
            when = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            first = first or when
            rank, text = m.group(2), m.group(3)
            if rank not in (None, "TP0"):
                continue
            for name, pattern in EVENTS:
                hit = pattern.search(text)
                if hit:
                    events.append((when, name, hit.groups()))
                    break
    return first, events


def timeline(path: str) -> list:
    first, events = parse(path)
    if first is None:
        return [("(no timestamped lines)", 0.0, "")]
    rows, last, family_t0 = [], first, None
    by_name = {}
    for when, name, groups in events:
        by_name.setdefault(name, []).append((when, groups))
    load_begin = by_name.get("load_begin", [(None,)])[0][0]
    if load_begin:
        rows.append(("process start -> weight load", (load_begin - first).total_seconds(), ""))
        last = load_begin
    if "load_end" in by_name:
        when, groups = by_name["load_end"][0]
        rows.append(("weight load", (when - last).total_seconds(), f"loader {groups[0]} s"))
        last = when
    if "pool_end" in by_name:
        when, _ = by_name["pool_end"][0]
        rows.append(("KV pool", (when - last).total_seconds(), ""))
        last = when
    # each warm-up closes a family: its rows run from the end of the previous phase
    restored = [g[0] for _, g in by_name.get("restored", [])]
    families = [g[0] for _, g in by_name.get("saved", [])]
    traced = list(by_name.get("traced", []))
    for index, (when, groups) in enumerate(by_name.get("warm", [])):
        family = families[index] if index < len(families) else f"family {index + 1}"
        trace = next((g for t, g in traced if last <= t <= when), None)
        note = "restored (no trace)" if family in restored else (
            f"traced {trace[0]} shapes in {float(trace[1]):.0f} s" if trace else "")
        rows.append((f"{family} graphs: build + warm-up ({groups[1]} shapes)", (when - last).total_seconds(), note))
        last = when
    if "ready" in by_name:
        when, _ = by_name["ready"][0]
        rows.append(("-> fired up", (when - last).total_seconds(), ""))
        rows.append(("TOTAL", (when - first).total_seconds(), ""))
    else:
        rows.append(("(not ready in this log)", 0.0, ""))
    return rows


def main(argv) -> int:
    for path in argv[1:]:
        print(f"== {path}")
        for name, seconds, note in timeline(path):
            print(f"  {name:<44} {seconds:8.0f} s  {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
