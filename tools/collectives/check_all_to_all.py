# SPDX-License-Identifier: Apache-2.0
"""Byte-compare the outputs of all_to_all_check.py's executions with what one all_to_all over all ranks delivers.

Usage: python tools/collectives/check_all_to_all.py <infer_result dir> [--ranks 128] [--rows 4096] [--cols 7168]
[--runs 8]

Rank r's output of every execution (``output_worker_<r>/out``, ``out.2`` ... ``out.<runs>``) must hold, as its block s
(rows ``[s * B, (s + 1) * B)``, ``B = rows / ranks``), rows ``[r * B, (r + 1) * B)`` of sender s's input,
``default_rng(1000 + s).integers(0, 256, (rows, cols), uint8)``. Prints how many rank-executions are bad, the first 64
bad (receiver <- sender) blocks, and how many bad bytes are 0xA5: in an ``A2A_POISON=1`` run, the fill of a byte the
exchange never wrote (about 1 in 256 other bad bytes match it by chance). Exits 0 when every byte of every execution
is right, 1 on any bad byte, and 2 when a rank's outputs are missing, misnumbered or of the wrong size.
"""
import argparse
import os
import re
import sys

import numpy as np

POISON = 0xA5
_OUTPUT = re.compile(r"out(?:\.(\d+))?")


def _source(rank, rows, cols):
    return np.random.default_rng(1000 + rank).integers(0, 256, size=(rows, cols), dtype=np.uint8)


def _outputs(directory, runs):
    """A rank's output files in execution order (out, out.2, ... out.<runs>), or None unless exactly those exist."""
    numbered = {}
    for name in os.listdir(directory) if os.path.isdir(directory) else ():
        match = _OUTPUT.fullmatch(name)
        if match:
            numbered[int(match.group(1) or 1)] = os.path.join(directory, name)
    if sorted(numbered) != list(range(1, runs + 1)):
        return None
    return [numbered[i] for i in range(1, runs + 1)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("result_dir")
    parser.add_argument("--ranks", type=int, default=128)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=7168)
    parser.add_argument("--runs", type=int, default=8, help="executions per rank (all_to_all_check.py's A2A_RUNS)")
    args = parser.parse_args(argv)
    n, rows, cols = args.ranks, args.rows, args.cols
    if rows % n:
        parser.error(f"{rows} rows do not split into {n} blocks")
    block = rows // n
    sources = [_source(s, rows, cols) for s in range(n)]
    executions = bad_executions = bad_bytes = poison_bytes = 0
    bad_pairs = set()
    for r in range(n):
        directory = os.path.join(args.result_dir, f"output_worker_{r}")
        files = _outputs(directory, args.runs)
        if files is None:
            print(f"rank {r}: {directory} does not hold exactly out, out.2 ... out.{args.runs}", file=sys.stderr)
            return 2
        expected = np.concatenate([sources[s][r * block:(r + 1) * block] for s in range(n)])
        for path in files:
            got = np.fromfile(path, dtype=np.uint8)
            if got.size != rows * cols:
                print(f"{path}: {got.size} bytes, not {rows * cols}", file=sys.stderr)
                return 2
            executions += 1
            got = got.reshape(rows, cols)
            wrong = got != expected
            if not wrong.any():
                continue
            bad_executions += 1
            bad_bytes += int(wrong.sum())
            poison_bytes += int((got[wrong] == POISON).sum())
            bad_pairs.update((r, int(s)) for s in np.unique(np.nonzero(wrong)[0] // block))
    print(f"{n} ranks, {executions} rank-executions: {bad_executions} bad, {bad_bytes} bad bytes "
          f"({poison_bytes} of them 0xA5)")
    if bad_pairs:
        print("bad (receiver <- sender) blocks:", sorted(bad_pairs)[:64], "..." if len(bad_pairs) > 64 else "")
    return 1 if bad_executions else 0


if __name__ == "__main__":
    sys.exit(main())
