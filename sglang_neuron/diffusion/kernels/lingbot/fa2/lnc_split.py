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

"""How the paged FA2 kernel splits one launch's work over the cores of a logical NeuronCore.

Pure trace-time integer math (no NKI), so the split is unit-testable on a CPU host. At LNC 1
every function returns the whole range on the one core, in the kernel's original order.

At LNC 2 (two physical cores per logical rank):

* **Attention** (:func:`attention_items`): core 0 takes heads ``[0, N // 2)``, core 1 heads
  ``[N // 2, 2 * (N // 2))``; with an odd head count the last head's query rows are split at a
  128-row tile boundary (core 0 the first part), so both cores run the same number of history
  pipelines and about the same work. Query rows are independent and both cores read that head's
  whole history and chunk, so the split is exact and every row's arithmetic is the LNC 1 one.
  A head of a single query tile cannot be split and stays on core 0.
* **CP gather** (:func:`gather_slots`): a collective is a logical-rank operation, so both cores
  issue every all-gather, in the same order. Slot ``j`` lists the heads whose gathers a core issues
  from its ``j``-th history pipeline: the heads of the ``j``-th item of either core, so each core's
  first chunk-phase head is gathered first.
* **Staging, commit** (:func:`split_range`): the shard's head-major staging, the gathered chunk's
  reorder and the commit write are split contiguously over flat item ranges, so every pool slot is
  written exactly once, by one core.
"""

TILE = 128


def split_range(n, num_shards, shard):
    """``[start, stop)`` of ``shard``'s contiguous share of ``n`` items (the first shards take one
    more when ``n`` does not divide)."""
    base = n // num_shards
    extra = n - base * num_shards
    start = shard * base + min(shard, extra)
    stop = start + base
    if shard < extra:
        stop = stop + 1
    return start, stop


def split_rows(sq, tile=TILE):
    """Rows of the first core's part of a split head: the multiple of ``tile`` that best balances
    the two parts (the smaller one on a tie), or ``None`` when ``sq`` is a single tile (the head
    cannot be split)."""
    tiles = (sq + tile - 1) // tile
    if tiles < 2:
        return None
    best = tile
    best_gap = max(sq - 2 * tile, 2 * tile - sq)
    for t in range(2, tiles):
        gap = max(sq - 2 * t * tile, 2 * t * tile - sq)
        if gap < best_gap:
            best = t * tile
            best_gap = gap
    return best


def attention_items(num_heads, sq, num_shards, shard, tile=TILE):
    """This core's attention work: ``(head, row_start, rows)`` items, in issue order."""
    items = []
    if num_shards == 1:
        for h in range(num_heads):
            items.append((h, 0, sq))
        return items
    pairs = num_heads // num_shards
    for h in range(shard * pairs, (shard + 1) * pairs):
        items.append((h, 0, sq))
    if num_heads % num_shards != 0:
        last = num_heads - 1
        cut = split_rows(sq, tile)
        if cut is None:
            if shard == 0:
                items.append((last, 0, sq))
        elif shard == 0:
            items.append((last, 0, cut))
        else:
            items.append((last, cut, sq - cut))
    return items


def gather_slots(num_heads, sq, num_shards, tile=TILE):
    """The CP gathers per history-pipeline slot (the same on every core): a list of head lists,
    each in ascending order.

    Every head appears in exactly one slot. A core with fewer items than slots issues the rest
    after its history pipelines.
    """
    per_core = []
    depth = 0
    for s in range(num_shards):
        items = attention_items(num_heads, sq, num_shards, s, tile)
        per_core.append(items)
        depth = max(depth, len(items))
    taken = [False] * num_heads
    slots = []
    for j in range(depth):
        heads = []
        for h in range(num_heads):
            for s in range(num_shards):
                if j < len(per_core[s]) and per_core[s][j][0] == h and not taken[h]:
                    taken[h] = True
                    heads.append(h)
        slots.append(heads)
    return slots
