# SPDX-License-Identifier: Apache-2.0
"""A rank's index of its remote slots: SGLang page key -> slot, in LRU order, with a record per slot.

Each slot has a generation (bumped at every assignment), a state (free, writing, ready), the digest of the bytes
written, and a pin count. A key is readable only once its write completed (``ready``). A read snapshots
``(slot, gen, digest)`` and pins the slot, so this rank does not reassign it under the read; the reader then accepts
a page only if the slot still holds that key and generation and the bytes match the digest. Slots are handed out in
runs of consecutive ids where possible, so a batch of pages moves in few descriptors.

The index lives in the backend of one rank and is lost on restart; every slot then counts as empty, which is safe.
"""

from __future__ import annotations

import bisect
import collections
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

FREE, WRITING, READY = 0, 1, 2


@dataclass(frozen=True)
class Snapshot:
    """A slot as a read found it."""
    key: str
    slot: int
    gen: int
    digest: bytes


class RunAllocator:
    """Free slots as runs ``start -> length``; first fit for a run of n, else the longest runs."""

    def __init__(self, num_slots: int):
        self._len: Dict[int, int] = {0: num_slots} if num_slots > 0 else {}
        self._end: Dict[int, int] = {num_slots: 0} if num_slots > 0 else {}  # end (exclusive) -> start
        self._starts: List[int] = [0] if num_slots > 0 else []
        self.free = num_slots

    def _remove(self, start: int) -> int:
        length = self._len.pop(start)
        del self._end[start + length]
        self._starts.pop(bisect.bisect_left(self._starts, start))
        return length

    def _add(self, start: int, length: int) -> None:
        self._len[start] = length
        self._end[start + length] = start
        bisect.insort(self._starts, start)

    def take(self, n: int) -> List[int]:
        """Up to ``n`` free slots, consecutive where possible (fewer if the allocator runs out)."""
        out: List[int] = []
        while len(out) < n and self._starts:
            want = n - len(out)
            start = next((s for s in self._starts if self._len[s] >= want), None)
            if start is None:
                start = max(self._starts, key=lambda s: self._len[s])
            length = self._remove(start)
            used = min(length, want)
            out.extend(range(start, start + used))
            if used < length:
                self._add(start + used, length - used)
        self.free -= len(out)
        return out

    def give(self, slots: Sequence[int]) -> None:
        for slot in sorted(slots):
            start, length = slot, 1
            prev = self._end.get(start)
            if prev is not None:
                length += self._remove(prev)
                start = prev
            if start + length in self._len:
                length += self._remove(start + length)
            self._add(start, length)
            self.free += 1


class SlotIndex:
    def __init__(self, num_slots: int):
        if num_slots <= 0:
            raise ValueError(f"{num_slots} slots")
        self.num_slots = num_slots
        self._lru: "collections.OrderedDict[str, int]" = collections.OrderedDict()  # oldest first, ready keys only
        self._key: List[Optional[str]] = [None] * num_slots
        self._gen = [0] * num_slots
        self._state = [FREE] * num_slots
        self._digest: List[Optional[bytes]] = [None] * num_slots
        self._pins = [0] * num_slots
        self._writing: Dict[str, int] = {}  # key -> slot of a write in flight
        self._free = RunAllocator(num_slots)
        self._lock = threading.Lock()
        self.evictions = 0

    # -- reads of the index --
    def ready_prefix(self, keys: Sequence[str]) -> int:
        """How many leading keys are ready."""
        with self._lock:
            for i, key in enumerate(keys):
                if key not in self._lru:
                    return i
            return len(keys)

    # -- writes --
    def begin_write(self, keys: Sequence[str]) -> Tuple[List[Optional[Tuple[int, int]]], List[bool]]:
        """Plan a batch: per key either ``(slot, gen)`` to write, or None. ``present[i]`` is True for a key that is
        ready already (touched, nothing to write). A key with neither could not get a slot (all pinned) or has a
        write in flight."""
        plan: List[Optional[Tuple[int, int]]] = [None] * len(keys)
        present = [False] * len(keys)
        with self._lock:
            need, seen = [], set()
            for i, key in enumerate(keys):
                if key in self._lru:
                    self._lru.move_to_end(key)
                    present[i] = True
                elif key not in self._writing and key not in seen:
                    seen.add(key)
                    need.append(i)
            slots = self._take(len(need))
            for i, slot in zip(need, slots):
                self._gen[slot] += 1
                self._state[slot] = WRITING
                self._key[slot] = keys[i]
                self._digest[slot] = None
                self._writing[keys[i]] = slot
                plan[i] = (slot, self._gen[slot])
        return plan, present

    def finish_write(self, slot: int, gen: int, digest: Optional[bytes]) -> bool:
        """End a write: ready with ``digest``, or freed when ``digest`` is None (it failed). False if the slot was
        reassigned meanwhile (cleared)."""
        with self._lock:
            if self._gen[slot] != gen or self._state[slot] != WRITING:
                return False
            key = self._key[slot]
            self._writing.pop(key, None)
            if digest is None:
                self._release(slot)
                return False
            self._state[slot] = READY
            self._digest[slot] = digest
            self._lru[key] = slot
            return True

    # -- reads --
    def begin_read(self, keys: Sequence[str]) -> List[Snapshot]:
        """Snapshot and pin the leading ready keys."""
        out: List[Snapshot] = []
        with self._lock:
            for key in keys:
                slot = self._lru.get(key)
                if slot is None:
                    break
                self._lru.move_to_end(key)
                self._pins[slot] += 1
                out.append(Snapshot(key, slot, self._gen[slot], self._digest[slot]))
        return out

    def still_holds(self, snap: Snapshot) -> bool:
        with self._lock:
            return (self._gen[snap.slot] == snap.gen and self._state[snap.slot] == READY
                    and self._key[snap.slot] == snap.key)

    def unpin(self, snaps: Sequence[Snapshot]) -> None:
        """End a read; a slot `drop`-ped while pinned is freed by its last unpin."""
        with self._lock:
            for snap in snaps:
                self._pins[snap.slot] -= 1
                if (self._pins[snap.slot] == 0 and self._state[snap.slot] == FREE and self._key[snap.slot] is None
                        and self._gen[snap.slot] == snap.gen):
                    self._free.give([snap.slot])

    def drop(self, snap: Snapshot) -> None:
        """Forget a slot whose bytes failed a check (it will be written again)."""
        with self._lock:
            if self._gen[snap.slot] == snap.gen and self._state[snap.slot] == READY:
                self._lru.pop(snap.key, None)
                if self._pins[snap.slot] == 0:
                    self._release(snap.slot)
                else:
                    self._state[snap.slot] = FREE  # freed by the last unpin
                    self._key[snap.slot] = None

    def clear(self) -> None:
        with self._lock:
            for slot in range(self.num_slots):
                if self._state[slot] == READY and self._pins[slot] == 0:
                    self._lru.pop(self._key[slot], None)
                    self._release(slot)

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {"ready": len(self._lru), "writing": len(self._writing), "free": self._free.free,
                    "evictions": self.evictions}

    # -- under the lock --
    def _release(self, slot: int) -> None:
        self._state[slot] = FREE
        self._key[slot] = None
        self._digest[slot] = None
        self._free.give([slot])

    def _take(self, n: int) -> List[int]:
        slots = self._free.take(n)
        if len(slots) < n:
            # evict unpinned ready slots, oldest first, iterating the LRU without copying it
            evict = []
            for key in self._lru:
                if self._free.free + len(evict) >= n - len(slots):
                    break
                if self._pins[self._lru[key]]:
                    continue
                evict.append(key)
            for key in evict:
                self._release(self._lru.pop(key))
                self.evictions += 1
            slots += self._free.take(n - len(slots))
        return slots
