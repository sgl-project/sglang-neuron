#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Where a MiMo-V2 decode step's device time goes, from a serve-window device profile converted to Parquet
(docs/guides/profiling.md: ``neuron-explorer view -d serve_rank0 --output-format parquet --ingest-only``):

    python3 tools/mimo_v2/profile_step.py <parquet session dir>

Prints each execution's span; for the median one, its kernel calls (HLO custom calls) in order with their start
offsets, the period of the decoder-layer kernels, and for one mid-step layer each engine's busy time, the top opcodes
and the DMA bytes moved, on physical core 0."""

import sys


def main(path: str) -> int:
    import pyarrow.parquet as pq

    ins = pq.read_table(f"{path}/Instruction.parquet",
                        columns=["execution_index", "start_ts", "end_ts", "engine", "pcore_idx", "opcode",
                                 "duration_ns"]).to_pandas()
    spans = {e: (g.start_ts.min(), g.end_ts.max()) for e, g in ins.groupby("execution_index")}
    for e, (a, b) in sorted(spans.items()):
        print(f"execution {e}: {(b - a) / 1e3:.0f} us")
    order = sorted(spans, key=lambda e: spans[e][1] - spans[e][0])
    pick = order[len(order) // 2]
    t0, t1 = spans[pick]
    h = pq.read_table(f"{path}/HloInstruction.parquet").to_pandas()
    h = h[(h.start_ts >= t0) & (h.end_ts <= t1)].sort_values("start_ts")
    calls = h[h.hlo_name.str.contains("custom-call")]
    # the step's XLA ops (everything but the kernel calls and the containers of the step's body)
    import re

    other = h[~h.hlo_name.str.contains(r"custom-call|= tuple\(|region_", regex=True)]
    kinds = other.hlo_name.map(lambda n: (re.search(r"= (\S+?)\(", n) or [None, n[:16]])[1])
    dur = (other.end_ts - other.start_ts) / 1e3
    print(f"XLA ops: {len(other)}, {dur.sum():.0f} us summed (by op: "
          + ", ".join(f"{k} {v:.0f}" for k, v in dur.groupby(kinds).sum().sort_values(ascending=False).head(8).items())
          + ")")
    starts = [(r.start_ts - t0) / 1e3 for r in calls.itertuples()]
    print(f"execution {pick}: {len(calls)} kernel calls")
    # the decoder layers: the long run of calls at a steady period
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    if gaps:
        gaps_sorted = sorted(gaps)
        period = gaps_sorted[len(gaps_sorted) // 2]
        print(f"median gap between kernel calls {period:.0f} us; calls at {[round(s) for s in starts[:6]]} ... "
              f"{[round(s) for s in starts[-6:]]}")
        mid = len(starts) // 2
        a, b = t0 + starts[mid] * 1e3, t0 + starts[mid + 1] * 1e3
        w = ins[(ins.execution_index == pick) & (ins.start_ts >= a) & (ins.start_ts < b) & (ins.pcore_idx == 0)]
        print(f"one mid-step kernel call: {(b - a) / 1e3:.0f} us")
        for eng, x in w.groupby("engine"):
            print(f"  {eng:<8} busy {x.duration_ns.sum() / 1e3:7.1f} us  ({len(x)} instructions)")
        top = w.groupby(["engine", "opcode"]).duration_ns.agg(["count", "sum"]).sort_values("sum", ascending=False)
        for (eng, op), row in top.head(12).iterrows():
            print(f"  {eng:<8} {op:<28} {int(row['count']):5d}  {row['sum'] / 1e3:7.1f} us")
        try:
            du = pq.read_table(f"{path}/DmaUsage.parquet").to_pandas()
            d = du[(du.timestamp >= a) & (du.timestamp < b) & (du.pcore_idx == 0)]
            print(f"  DMA mean throughput {d.current_throughput.mean() / 1e9:.0f} GB/s (core 0)")
        except Exception as e:  # noqa: BLE001 - a session without the table
            print(f"  (no DMA usage: {e})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
