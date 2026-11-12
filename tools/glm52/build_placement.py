#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a static MoE expert placement for the GLM-5.2 prefill from a routing histogram.

Each MoE layer waits for the EP rank with the most routed-expert work, and the identity placement leaves every expert
of a hot expert's rank on that rank. A placement gives each rank ``e_local`` slots (9 at PP4: 288 slots for 256
experts) and places experts hottest first on the least-loaded rank with a free slot, so a hot expert's rank holds fewer
experts. An expert's cost follows the routed kernel: its expected rows in a ``--chunk-tokens`` forward, in 128-row
blocks, the first block streaming all its weights and each further one the gate and up projections again.

Input histogram, each token counted once per top-k expert::

    {"tokens": 1298432, "layers": {"3": [rows routed to expert 0, expert 1, ...], "4": [...], ...}}

The Neuron prefill routes inside its kernel and records nothing, so the counts come from a server that routes through
SGLang's own MoE layer (the same checkpoint on GPU SGLang over the same traffic, ``--expert-distribution-recorder-mode
stat``): sum ``logical_count`` over the passes per layer; ``tokens`` is one layer's sum divided by top-k (8). Layers
with no routed rows stay at the identity.

Output, the file the prefill loaders read, EP rank q owning slots [q * e_local, (q + 1) * e_local), -1 an empty slot::

    {"e_local": 9, "calibration": {"workload": ..., "tokens": ..., "objective": ...}, "layers": {"3": [...], ...}}

Only PP4 serves a placement (PP16 runs the identity). The served file is pinned by md5 (placements.PLACEMENT_MD5)
because it moves the stages' device time; a new one replaces the pin together with new stage measurements. Another
``--e-local`` means other stage programs, since the kernel's expert tables are ``e_local`` columns wide.

Usage: python tools/glm52/build_placement.py HISTOGRAM OUT --workload TEXT [--ep 32] [--e-local 9] [--chunk-tokens 4096]
"""

from __future__ import annotations

import argparse
import json
import sys

BLOCK_ROWS = 128
# The PP4 routed experts' block times (EP32, TP1, intermediate 2048), which are weight-stream bound: the first block of
# an expert streams its 21.2 MB of weights, each further block the 12.6 MB of gate and up again.
FIRST_BLOCK_US = 59.0
NEXT_BLOCK_US = 44.0


def expert_us(rows: float) -> float:
    """The routed kernel's time for an expert of ``rows`` rows in a forward: nothing for no row, else its first block
    and its rows past the first block in further blocks."""
    if rows <= 0:
        return 0.0
    return FIRST_BLOCK_US + NEXT_BLOCK_US * max(0.0, rows / BLOCK_ROWS - 1.0)


def _plain_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def read_histogram(path: str) -> tuple:
    """``(tokens, {layer: [rows of each expert]})`` of a histogram file, the layers with no routed row left out.

    Raises:
        ValueError: If the file is not a histogram of the documented form, or no layer routed a row.
    """
    with open(path) as f:
        raw = json.load(f)
    if not isinstance(raw, dict) or not isinstance(raw.get("layers"), dict):
        raise ValueError(f'{path}: no "layers" map of checkpoint layer -> rows of each expert')
    tokens = raw.get("tokens")
    if not _plain_int(tokens) or tokens < 1:
        raise ValueError(f'{path}: "tokens" is {tokens!r}, not the count of prompt tokens the rows were counted over')
    layers = {}
    for key, rows in raw["layers"].items():
        if not (key.isascii() and key.isdigit() and str(int(key)) == key):
            raise ValueError(f"{path}: the layer key {key!r} is not a checkpoint layer index")
        if not isinstance(rows, list) or not rows or not all(_plain_int(r) and r >= 0 for r in rows):
            raise ValueError(f"{path}: layer {key} is not a list of row counts, one per expert")
        if any(rows):
            layers[int(key)] = rows
    if not layers:
        raise ValueError(f"{path}: no layer routed a row")
    if len({len(rows) for rows in layers.values()}) != 1:
        raise ValueError(f"{path}: the layers count different numbers of experts")
    return tokens, dict(sorted(layers.items()))


def place_layer(costs: list, ep: int, e_local: int) -> list:
    """The slot -> expert list of one layer whose experts cost ``costs``: the experts hottest first (the lower id on a
    tie), each on the rank with the least cost (the lower rank on a tie) that has a free slot; a rank's experts in
    ascending order, then its empty slots (-1)."""
    ranks = [[] for _ in range(ep)]
    load = [0.0] * ep
    for expert in sorted(range(len(costs)), key=lambda e: (-costs[e], e)):
        rank = min((q for q in range(ep) if len(ranks[q]) < e_local), key=lambda q: (load[q], q))
        ranks[rank].append(expert)
        load[rank] += costs[expert]
    return [slot for experts in ranks for slot in sorted(experts) + [-1] * (e_local - len(experts))]


def layer_costs(rows: list, tokens: int, chunk_tokens: int) -> list:
    """Each expert's cost in a forward of ``chunk_tokens`` tokens, from its rows over ``tokens`` histogram tokens."""
    return [expert_us(r * chunk_tokens / tokens) for r in rows]


def busiest_rank_us(slots: list, costs: list, e_local: int) -> float:
    """The cost of the busiest rank of one layer's slot list."""
    return max(sum(costs[e] for e in slots[q:q + e_local] if e >= 0) for q in range(0, len(slots), e_local))


def build_placement(tokens: int, layers: dict, workload: str, ep: int = 32, e_local: int = 9,
                    chunk_tokens: int = 4096) -> dict:
    """The placement of every layer of a histogram (read_histogram), with its calibration.

    Raises:
        ValueError: If the experts do not split evenly over ``ep`` ranks (SGLang's EP gives each rank E / ep), or
            ``ep`` ranks of ``e_local`` slots cannot hold them.
    """
    experts = len(next(iter(layers.values())))
    if ep < 1 or e_local < 1 or chunk_tokens < 1:
        raise ValueError(f"ep {ep}, e_local {e_local} and chunk_tokens {chunk_tokens} must be at least 1")
    if experts % ep:
        raise ValueError(f"{experts} experts do not split evenly over {ep} EP ranks")
    if ep * e_local < experts:
        raise ValueError(f"{ep} EP ranks x {e_local} slots hold fewer than {experts} experts")
    objective = (f"the least routed-expert time on the busiest of {ep} EP ranks, {e_local} slots a rank; an expert's "
                 f"expected rows in a {chunk_tokens:,}-token forward cost {FIRST_BLOCK_US:g} us for the first "
                 f"{BLOCK_ROWS}-row block and {NEXT_BLOCK_US:g} us a further block")
    placed = {layer: place_layer(layer_costs(rows, tokens, chunk_tokens), ep, e_local)
              for layer, rows in layers.items()}
    return {"e_local": e_local,
            "calibration": {"workload": workload, "tokens": f"{tokens:,} prompt tokens", "objective": objective},
            "layers": placed}


def placement_text(placement: dict) -> str:
    """A placement's file text: its other fields first, then one line a layer, so a diff names the layers that moved."""
    head = [f"  {json.dumps(key)}: {json.dumps(value)}," for key, value in placement.items() if key != "layers"]
    layers = [f"    {json.dumps(str(layer))}: {json.dumps(slots)}" for layer, slots in placement["layers"].items()]
    return "\n".join(["{", *head, '  "layers": {', ",\n".join(layers), "  }", "}"]) + "\n"


def main(argv) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("histogram", help="the routing histogram (JSON, see the module docstring)")
    parser.add_argument("out", help="the placement file to write")
    parser.add_argument("--workload", required=True, help="what the histogram was counted on, for the calibration")
    parser.add_argument("--ep", type=int, default=32, help="EP ranks of a stage (default 32, the PP4 point)")
    parser.add_argument("--e-local", type=int, default=9, help="slots a rank (default 9)")
    parser.add_argument("--chunk-tokens", type=int, default=4096, help="tokens of a prefill forward (default 4096)")
    args = parser.parse_args(argv)
    try:
        tokens, layers = read_histogram(args.histogram)
        placement = build_placement(tokens, layers, args.workload, args.ep, args.e_local, args.chunk_tokens)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    with open(args.out, "w") as f:
        f.write(placement_text(placement))
    per_rank = len(next(iter(layers.values()))) // args.ep
    identity, placed = [], []
    for layer, rows in layers.items():
        costs = layer_costs(rows, tokens, args.chunk_tokens)
        identity.append(busiest_rank_us(list(range(len(rows))), costs, per_rank))
        placed.append(busiest_rank_us(placement["layers"][layer], costs, args.e_local))
    print(f"{args.out}: {len(layers)} layers placed; the busiest rank's routed experts, mean over the layers: "
          f"{sum(identity) / len(identity):.0f} us at the identity placement, {sum(placed) / len(placed):.0f} us "
          "placed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
