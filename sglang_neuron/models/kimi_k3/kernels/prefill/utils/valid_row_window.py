# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The valid-row window of the Kimi-K3 prefill blocks: the option and its runtime helpers.

A prefix-cache hit that starts inside a segment gives the cached rows before ``valid_start`` as pad rows. The block
option ``valid_row_window`` names the parts of a block that then skip the work of their whole row tiles before
``valid_start``. Each part is its own item, so each one can be compared alone.

The contract of every item: the valid rows of each output, the cache pages, the recurrent state and the logits keep
their bits. A pad row of a block output may change (for example a partial sum of a skipped tile becomes +0.0 where the
work of the cached row gave another value), and no valid output reads it; each item proves that with a NaN poison
control on the pad rows.

Items:
    KDA block: ``"scan"`` (the scan chunks before valid_start), ``"oproj"`` (the o_proj groups before it). Both also
    keep the pad rows of y (+0.0, as without them).
    MoE block: ``"shared_expert"`` (the shared-expert tiles before it).

An empty window keeps the program, and so does a window without ``valid_start``. The default of the KDA block is
``("scan",)`` on Trn2 and the empty window on Trn3; the MoE block's default is the empty window.

The helpers give the runtime predicates of such skips as 0/1 int32 [1, 1] tiles in SBUF (VectorE integer
operations on small values, which are exact), run a 0/1-trip loop on one of them, and give the first row of
this rank's row block in the group.
"""

import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl

from .kernel_assert import kernel_assert

KDA_ITEMS = ("scan", "oproj")
MOE_ITEMS = ("shared_expert",)

_I32 = nl.int32
_VEC = nisa.engine.vector
_DMA_SYNC = {"dge_mode": nisa.dge_mode.hwdge, "engine": nisa.engine.sync}


def window_items(valid_row_window, allowed, built, block):
    """Check ``valid_row_window`` and return its items as a frozenset.

    Args:
        valid_row_window: A tuple (or list) of item names, or None / empty.
        allowed: The item names of the block.
        built: The items this tree builds; the others are refused.
        block: The block name, for the messages.

    Example:
        >>> sorted(window_items(("oproj", "scan"), KDA_ITEMS, KDA_ITEMS, "kda"))
        ['oproj', 'scan']
    """
    items = tuple(valid_row_window or ())
    for item in items:
        kernel_assert(
            isinstance(item, str) and item in allowed,
            f"{block}: valid_row_window item {item!r} is not one of {allowed}",
        )
    kernel_assert(len(set(items)) == len(items), f"{block}: valid_row_window repeats an item: {items}")
    missing = [item for item in items if item not in built]
    kernel_assert(not missing, f"{block}: valid_row_window items {missing} are not built in this version")
    return frozenset(items)


def load_scalar(src, name, dma=None):
    """Load an int32 [1, 1] HBM tensor into a new [1, 1] SBUF tile (Sync hardware DGE by default)."""
    dst = nl.ndarray((1, 1), dtype=_I32, buffer=nl.sbuf, name=name)
    nisa.dma_copy(dst=dst, src=src.ap(pattern=[[1, 1], [1, 1]], offset=0), **(dma or _DMA_SYNC))
    return dst


def flag_at_least(dst, src, threshold):
    """``dst = 1 if src >= threshold else 0`` for int32 [1, 1] tiles: min(max(src - threshold + 1, 0), 1)."""
    nisa.tensor_scalar(
        dst=dst, data=src, op0=nl.subtract, operand0=int(threshold) - 1, op1=nl.maximum, operand1=0, engine=_VEC
    )
    nisa.tensor_scalar(dst=dst, data=dst, op0=nl.minimum, operand0=1, engine=_VEC)


def flag_below(dst, src, threshold):
    """``dst = 1 if src < threshold else 0`` for int32 [1, 1] tiles: min(max(threshold - src, 0), 1)."""
    nisa.tensor_scalar(
        dst=dst, data=src, op0=nl.multiply, operand0=-1, op1=nl.add, operand1=int(threshold), engine=_VEC
    )
    nisa.tensor_scalar(dst=dst, data=dst, op0=nl.maximum, operand0=0, op1=nl.minimum, operand1=1, engine=_VEC)


def run_if(flag, body):
    """Run ``body()`` once if the int32 [1, 1] SBUF tile ``flag`` holds 1, and not at all if it holds 0.

    It is a 0/1-trip loop. Its trip count goes into a register right at the loop, so the register is live only
    there: each engine sequencer holds 64 registers, and the compiler does not spill them. A loop boundary ends
    with all-engine barriers, so a loop has a fixed cost even with 0 trips. No collective may run in
    ``body``.
    """
    reg = nisa.register_alloc()
    nisa.register_load(reg, flag)
    nl.fori_loop(0, reg, lambda _: body())


def own_row_start(world_ranks, rows, name):
    """Return the first row of this rank's row block in the group, ``index * rows``, as an int32 [1, 1] SBUF tile.

    ``index`` is the position of this rank in its rank list of ``world_ranks``: the position at which the group's
    all-gathers put the rows of this rank, and the row block that a reduce-scatter gives it. The kernel has no
    rank input, so the value comes from a lookup: a [1, R] table (GpSimd iota, stored to HBM) read at
    ``scalar_offset = rank``. Entry ``r`` of the table holds ``index(r) * rows``, and ``R`` is the number of ranks
    in ``world_ranks``.

    NKI 0.7.0 has no ``ncc.self_index_in_replica_group``. Thus the lookup uses the global rank from
    ``ncc.rank_id``, and the table maps each rank to its row start. For one group of ranks 0..W-1, the table is the
    same [1, W] iota of ``i * rows``.

    Example:
        >>> row0 = own_row_start(((0, 1, 2, 3),), 128, "kb_row0")  # rank 2: [[256]]
    """
    ranks = [int(rank) for group in world_ranks for rank in group]
    R = len(ranks)
    kernel_assert(
        sorted(ranks) == list(range(R)),
        f"own_row_start: world_ranks {world_ranks} must list each rank 0..{R - 1} one time",
    )
    starts = [0] * R
    for group in world_ranks:
        for position, rank in enumerate(group):
            starts[int(rank)] = position * int(rows)
    table = nl.ndarray((1, R), dtype=_I32, buffer=nl.sbuf, name=f"{name}_tab")
    # Write each run of ranks whose row starts have one step with one iota.
    first = 0
    while first < R:
        step = starts[first + 1] - starts[first] if first + 1 < R else 0
        end = first + 1
        while end < R and starts[end] - starts[end - 1] == step:
            end += 1
        nisa.iota(dst=table[:, first:end], pattern=[[step, end - first]], offset=starts[first])
        first = end
    table_hbm = nl.ndarray((1, R), dtype=_I32, buffer=nl.private_hbm, name=f"{name}_tab_hbm")
    nisa.dma_copy(dst=table_hbm, src=table, **_DMA_SYNC)
    index = nl.ndarray((1, 1), dtype=_I32, buffer=nl.sbuf, name=f"{name}_rank")
    nisa.register_store(dst=index, src=ncc.rank_id(name=f"{name}_rank_id"))
    row0 = nl.ndarray((1, 1), dtype=_I32, buffer=nl.sbuf, name=f"{name}_row0")
    nisa.dma_copy(
        dst=row0, src=table_hbm.ap(pattern=[[1, 1], [1, 1]], offset=0, scalar_offset=index, indirect_dim=1), **_DMA_SYNC
    )
    return row0
