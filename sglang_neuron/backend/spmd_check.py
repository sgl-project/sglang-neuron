# SPDX-License-Identifier: Apache-2.0
"""The SPMD check: every rank of a tensor-parallel group runs the same NEFF for each warmed
graph shape, or the server refuses to start.

On torch-neuronx a rank's graph is lowered from its own trace. The ranks lower identical StableHLO by construction (the
rank is a graph input, and the mesh registry gives every rank the whole partition), so one neuronx-cc run serves them
all through torch-neuronx's per-key compile lock and NEFF cache. Nothing in that path verifies it: a rank-dependent
constant reaching a graph compiles a different NEFF on one rank, whose collectives then run against the other ranks'
programs (a hang, or wrong outputs).

How it works. A NEFF's cache key (``Executor.cache_key``) is the sha256 of its serialized StableHLO, with rank- and
run-dependent attributes stripped, plus the compiler's version hash.

1. The graph runner says which shape it is warming and on which gloo group (``warming``).
2. The executor binding cache reports each executor's first call (``note_first_call``), which happens while
   torch-neuronx prepares the inputs, before the NEFF is submitted. During a warm-up the ranks all-gather the key
   there, and a mismatch is refused before any rank submits a program whose collectives would hang against the
   others'. This works because every rank warms the same shapes in the same order, and a shape's executors first run
   in the same order on every rank.
3. After the warm-up the ranks all-gather their graph -> keys maps (``check``) and refuse a graph that some ranks
   didn't warm, or that warmed without a recorded NEFF.

A graph that first runs outside a warm-up isn't checked, but the runtime always warms."""

from __future__ import annotations

import contextlib
import contextvars
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# the graph a block warms, and the TP group whose ranks compare each of its NEFFs before the first submit
_WARMING: contextvars.ContextVar[Optional[Tuple[str, object, int]]] = contextvars.ContextVar(
    "sglang_neuron_warmed_graph", default=None)
# graph -> the NEFF cache keys of the executors whose first call ran while it warmed
_KEYS: Dict[str, set] = {}


@contextlib.contextmanager
def warming(graph: str, group=None, world_size: int = 1):
    """The block warms ``graph``: an executor that first runs inside it belongs to it, and with ``group`` (the TP
    group's gloo group, of ``world_size`` ranks) its NEFF is compared across the group before it is submitted."""
    token = _WARMING.set((graph, group, int(world_size)))
    _KEYS.setdefault(graph, set())
    try:
        yield
    finally:
        _WARMING.reset(token)


def note_first_call(executor) -> None:
    """Called by the executor binding cache at an executor's first call, before torch-neuronx submits it."""
    warming = _WARMING.get()
    if warming is None:
        return
    graph, group, world_size = warming
    key = str(executor.cache_key)
    _KEYS.setdefault(graph, set()).add(key)
    if group is not None and world_size > 1:
        compare(graph, key, group, world_size)


def compare(graph: str, key: str, group, world_size: int) -> None:
    """Refuse, before the submit, an executor whose NEFF is not the one every other rank of the group runs at this
    point of the warm-up."""
    import torch.distributed as dist

    keys: List[Optional[str]] = [None] * world_size
    dist.all_gather_object(keys, key, group=group)
    if len(set(keys)) > 1:
        variants: Dict[str, List[int]] = {}
        for rank, k in enumerate(keys):
            variants.setdefault(str(k), []).append(rank)
        parts = "; ".join(f"ranks {_ranks(ranks)}: {k[:16]}" for k, ranks in sorted(variants.items(),
                                                                                   key=lambda item: item[1]))
        raise RuntimeError(
            f"SPMD check: the ranks of the tensor-parallel group compiled different programs for {graph} ({parts}), so "
            "their collectives would run against mismatched NEFFs; refused before any rank submits it. A "
            "rank-dependent value reaches the graph's trace; the server is refused")


def warmed_keys() -> Dict[str, List[str]]:
    return {graph: sorted(keys) for graph, keys in _KEYS.items()}


def _ranks(ranks: List[int]) -> str:
    return ",".join(str(r) for r in ranks)


def mismatches(per_rank: List[Dict[str, List[str]]]) -> List[str]:
    """One line per graph whose keys are not the same on every rank (a graph a rank did not warm has no keys there)."""
    problems = []
    for graph in sorted(set().union(*per_rank)):
        variants: Dict[tuple, List[int]] = {}
        for rank, keys in enumerate(per_rank):
            variants.setdefault(tuple(keys.get(graph, ())), []).append(rank)
        if len(variants) > 1:
            parts = [f"ranks {_ranks(ranks)}: {', '.join(k[:16] for k in keys) or 'not warmed'}"
                     for keys, ranks in sorted(variants.items(), key=lambda item: item[1])]
            problems.append(f"{graph} ({'; '.join(parts)})")
    return problems


def check(group, world_size: int) -> None:
    """Refuse a TP group whose ranks warmed different NEFFs for a graph (group: the group's gloo process group)."""
    import torch.distributed as dist

    gathered: List[Optional[Dict[str, List[str]]]] = [None] * world_size
    dist.all_gather_object(gathered, warmed_keys(), group=group)
    problems = mismatches(gathered)
    # a graph warmed with no NEFF recorded on any rank: the check would pass without having seen it
    unseen = sorted(graph for graph, keys in (gathered[0] or {}).items() if not keys)
    if unseen:
        raise RuntimeError(f"SPMD check: no NEFF was recorded while warming {', '.join(unseen)} (the executor binding "
                           "cache records each executor's first call); the server is refused")
    if problems:
        raise RuntimeError(
            "SPMD check: the ranks of the tensor-parallel group compiled different programs for "
            f"{len(problems)} warmed graph(s), so their collectives would run against mismatched NEFFs: "
            + " | ".join(problems)
            + ". A rank-dependent value reaches the graph's trace; the server is refused")
    logger.info("[neuron] SPMD check: %d warmed graph(s), one NEFF set on all %d ranks", len(gathered[0] or {}),
                world_size)
