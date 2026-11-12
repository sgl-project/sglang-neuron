# SPDX-License-Identifier: Apache-2.0
"""The per-step inputs of the Qwen3.5 giga decode, built on the host from SGLang's batch.

A decode step of a replica runs ``rows`` rows (the seqs bucket; ``n`` of them real, in SGLang's batch order). Every
rank of the replica builds the same replica-wide row tables (the graph takes its own token rows, rank position ``t``
owning rows ``t * rows / replica ...``):

- ``input_ids [rows]``: each row's current token (host ids; a device sampler's future replaces it);
- ``pos [rows, 1]``: each row's position (the token's offset in its KV slot);
- ``kv_slot [rows, 1]``: each row's KV slot, the SGLang page holding its sequence (``out_cache_loc // S``);
- ``gdn_slot [rows, 1]``: each row's GDN state slot (SGLang's mamba slot index);
- ``row_valid [world_rows, 1]``: 1 for a real row, in EP-gathered order over every replica (replica ``d``'s rows at
  ``d * rows ...``), from the token counts SGLang's DP attention gathers for every replica; identical on every rank.

Padding rows read and write only what no request owns: position 0 of SGLang's reserved page 0 and the state pool's
sink slot (both hold garbage nobody reads), with token 0, and are routed to no expert.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class DecodeRows:
    input_ids: np.ndarray  # [rows] int32
    pos: np.ndarray  # [rows, 1] int32
    kv_slot: np.ndarray  # [rows, 1] int32
    gdn_slot: np.ndarray  # [rows, 1] int32
    row_valid: np.ndarray  # [dp * rows, 1] int32


def build_decode_rows(*, token_ids: Sequence[int], positions: Sequence[int], out_cache_loc: Sequence[int],
                      state_slots: Sequence[int], rows: int, replica: int, slot_tokens: int, sink_slot: int,
                      replica_tokens: Sequence[int], paged: bool = False) -> DecodeRows:
    """The step's inputs of a replica (module docstring). ``replica_tokens``: every replica's real decode rows this
    step (SGLang's gathered ``global_num_tokens``), in replica order. ``paged``: ``slot_tokens`` is a KV page of
    the request's context, so a token's cache slot is at its position's offset in a page (the block table names the
    page; ``kv_slot`` then holds the written page)."""
    n = len(token_ids)
    if not (len(positions) == len(out_cache_loc) == len(state_slots) == n):
        raise ValueError(f"decode rows: {n} tokens, {len(positions)} positions, {len(out_cache_loc)} cache slots and "
                         f"{len(state_slots)} state slots")
    if n > rows or rows % replica:
        raise ValueError(f"{n} decode rows in a bucket of {rows} rows over a {replica}-rank replica")
    if any(int(t) > rows for t in replica_tokens):
        raise ValueError(f"a replica's {max(replica_tokens)} decode rows exceed the step's {rows}-row bucket")
    pos = np.zeros((rows, 1), np.int32)
    kv_slot = np.zeros((rows, 1), np.int32)
    gdn_slot = np.full((rows, 1), sink_slot, np.int32)
    tokens = np.zeros(rows, np.int32)
    if n:
        loc = np.asarray(out_cache_loc, np.int64)
        p = np.asarray(positions, np.int64)
        if np.any(loc % slot_tokens != (p % slot_tokens if paged else p)):
            bad = int(np.flatnonzero(loc % slot_tokens != p)[0])
            raise RuntimeError(f"decode row {bad}: cache slot {int(loc[bad])} is offset {int(loc[bad] % slot_tokens)} "
                               f"of its {slot_tokens}-token page, the token's position is {int(p[bad])}; the giga "
                               "attention needs one page per request (--page-size = the KV slot)")
        if np.any(loc // slot_tokens == 0):
            raise RuntimeError("a decode row's KV slot is SGLang's reserved page 0")
        s = np.asarray(state_slots, np.int64)
        if np.any((s < 1) | (s >= sink_slot)):
            raise RuntimeError(f"decode state slots {s.tolist()} outside the pool's slots [1, {sink_slot - 1}]")
        pos[:n, 0] = p
        kv_slot[:n, 0] = loc // slot_tokens
        gdn_slot[:n, 0] = s
        tokens[:n] = np.asarray(token_ids, np.int64)
    counts = np.asarray(replica_tokens, np.int64)
    row_valid = (np.arange(rows)[None, :] < counts[:, None]).astype(np.int32).reshape(-1, 1)
    return DecodeRows(input_ids=tokens, pos=pos, kv_slot=kv_slot, gdn_slot=gdn_slot, row_valid=row_valid)
