# SPDX-License-Identifier: Apache-2.0
"""The Kimi-K3 prompt cache: exact END entries of finished prompts over the prefill's pages and KDA state slots.

The policy: END entries, ``keep_previous``, eviction in least-recent order. The chunk-keyed radix tree hits only at a
whole prefill chunk, because the KDA state is kept at chunk ends only. An END entry holds the KDA state of its own last
computed token, so a hit can end at any token.

* An END entry keeps the first ``N`` tokens of a finished request: the KV pages ``[0, ceil(N / page))`` and the KDA
  state slot of the request (the cache ADOPTS the slot, with no copy).
* The key of an entry is the chain hash of the full 4096-token units of its tokens, then the tail and the length
  (``key``).
* A lookup gives the longest entry that is a prefix of the prompt and leaves one token or more to compute.
* An entry that a running request hit is locked: eviction skips it, so its pages and its state slot stay intact
  until the request copies them.
* ``keep_previous``: a request that extends its hit (the next turn of a session) keeps the old entry in the group of
  the new entry, in place of the first place in the eviction order.

The cache decides with host state only and a logical clock, so each PP stage that runs the same requests in the same
order keeps the same entries. The module does not import SGLang or torch: ``kimi_prompt_backend`` binds it to SGLang's
scheduler and to the Kimi prefill pool.

Example:
    >>> freed = []
    >>> cache = KimiPromptCache(page_tokens=4, resources=Resources(free_pages=freed.extend))
    >>> tokens = list(range(10))
    >>> cache.keep("r1", tokens, b"s", 10, pages=[7, 8, 9], slot_owner=None) is not None
    True
    >>> hit = cache.lookup("r2", tokens + [10, 11], b"s")
    >>> hit.num_tokens, hit.pages
    (10, (7, 8, 9))
    >>> cache.evict(3), sorted(freed)
    (3, [7, 8, 9])
"""

from __future__ import annotations

import array
import dataclasses
import hashlib
import logging
import sys
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

HASH_UNIT = 4096
_LOG = "[kimi_prompt_cache]"
# The environment variables that set the host slots (kimi_prompt_backend.HOST_SLOTS_ENV and HOST_STATE_SLOTS_ENV);
# named in a log line only.
_HOST_SLOTS_KNOB = "SGLANG_NEURON_KIMI_PROMPT_HOST_SLOTS"
_HOST_STATE_SLOTS_KNOB = "SGLANG_NEURON_KIMI_PROMPT_HOST_STATE_SLOTS"


def _token_bytes(tokens: Sequence[int]) -> bytes:
    """The bytes of ``tokens`` as the key hashes them (``np.int64`` little-endian).

    Example:
        >>> _token_bytes([1, -1]).hex()
        '0100000000000000ffffffffffffffff'
    """
    data = array.array("q", tokens)
    if sys.byteorder != "little":
        data.byteswap()
    return data.tobytes()


def request_salt(lora_id: object = 0, cache_salt: Optional[str] = None, extra_key: Optional[str] = None) -> bytes:
    """The key salt of a request: ``fpc|<lora id>|<cache salt>``, with SGLang's extra key when it is set.

    Example:
        >>> request_salt()
        b'fpc|0|'
        >>> request_salt(3, "a", "k")
        b'fpc|3|a|k'
    """
    salt = f"fpc|{lora_id or 0}|{cache_salt or ''}"
    return (salt if not extra_key else f"{salt}|{extra_key}").encode()


@dataclasses.dataclass
class Resources:
    """What the cache does with the pages and slots of a dropped entry, and how it adopts a request's slot.

    ``free_pages(pages)`` gets the pages that no entry holds any more. ``adopt_slot(request_id, owner)`` moves the
    request's KDA state slot to ``owner`` and returns it (the ``StateSlots.transfer``). ``free_slot(owner)`` gives
    back the slot of a dropped entry. ``on_drop(entry, freed)`` sees each dropped entry before its pages go back (the
    host tier copies it there).
    """

    free_pages: Callable[[List[int]], None]
    adopt_slot: Callable[[str, str], Optional[int]] = lambda request_id, owner: None
    free_slot: Callable[[str], None] = lambda owner: None
    on_drop: Callable[["Entry", List[int]], None] = lambda entry, freed: None


@dataclasses.dataclass
class Entry:
    """The kept pages and KDA state of one finished prompt."""

    key: bytes
    num_tokens: int
    pages: Tuple[int, ...]
    boundary: bytes
    # The owner name of the adopted state slot, and the slot (None: the entry keeps no state).
    slot_owner: Optional[str] = None
    slot: Optional[int] = None
    created: int = 0
    hits: int = 0
    request_id: str = ""
    kind: str = "end"
    # With keep_previous: the key of the newest END entry whose group holds this entry, else None.
    owner: Optional[bytes] = None
    # The running requests that hit the entry and did not release it yet.
    locks: int = 0
    # With the host tier: the content id of each whole page (the key of the tokens up to the page end), None for the
    # tail page. Entries of a session share the pages of one content id, on the device and on the host.
    cids: Tuple[Optional[bytes], ...] = ()


@dataclasses.dataclass(frozen=True)
class Hit:
    """What a request reuses of an entry: ``tokens`` of it, the pages ``[0, tokens // page)`` as they are, and the
    page that holds the tokens past the last whole page (``tail_page``, None when ``tokens`` ends on a page)."""

    entry: Entry
    tokens: int
    shared_pages: Tuple[int, ...]
    tail_page: Optional[int]


@dataclasses.dataclass(frozen=True)
class HostOp:
    """One copy between the device and the host tier. ``kind`` is ``"mla"`` (``device`` is a KV page id, ``host_slot``
    a host page slot) or ``"state"`` (``device`` is a KDA state slot, ``host_slot`` a host state slot). ``spill``
    copies the device to the host slot; else (a restore) the host slot to the device."""

    spill: bool
    kind: str
    device: int
    host_slot: int


@dataclasses.dataclass
class HostEntry:
    """An END entry that the host tier keeps: the host page slot of each page, and the host state slot of its KDA state
    (empty when the entry keeps no state)."""

    key: bytes
    num_tokens: int
    boundary: bytes
    cids: Tuple[Optional[bytes], ...]
    slots: Tuple[int, ...]
    state: Tuple[int, ...]
    request_id: str = ""


class HostTier:
    """The host-DRAM victim tier of the END entries (synchronous form).

    * Store: an END entry that the device drops goes to the host. Each whole page of the entry that the host does not
      hold yet (by content id) and each private page (the tail) gets a free page slot and a spill op; its KDA state
      gets a free state slot and a spill op. The least recent host entries go first when the slots are short; an entry
      that does not fit in the whole tier is not stored.
    * A host entry whose tokens the device holds again (a restore) goes to the front of the host eviction order.

    The counts are the same on every rank, so every PP stage keeps the same host entries: a page slot holds this rank's
    rows of one KV page, a state slot this rank's KDA state of one request.

    Args:
        num_slots: The host page slots of a rank.
        num_state_slots: The host state slots of a rank.

    Example:
        >>> host = HostTier(num_slots=8, num_state_slots=2)
        >>> host.free_slots, host.free_state_slots
        (8, 2)
    """

    def __init__(self, num_slots: int, num_state_slots: int = 0) -> None:
        self.num_slots, self.num_state_slots = int(num_slots), int(num_state_slots)
        self._free = list(range(self.num_slots))
        self._free_state = list(range(self.num_state_slots))
        self._entries: "OrderedDict[bytes, HostEntry]" = OrderedDict()
        self._lengths_by_boundary: Dict[bytes, Dict[int, int]] = {}
        # {content id: [host slot, entry count]} of the whole pages.
        self._pages: Dict[bytes, List[int]] = {}
        self.num_stored = self.num_restored = self.num_skipped = self.num_evicted = 0
        # The host entries that a restore of this step reads: a store does not evict them.
        self.pinned: Set[bytes] = set()

    @property
    def free_slots(self) -> int:
        return len(self._free)

    @property
    def free_state_slots(self) -> int:
        return len(self._free_state)

    @property
    def num_entries(self) -> int:
        return len(self._entries)

    def keys(self) -> List[bytes]:
        return list(self._entries)

    def get(self, key: bytes) -> Optional[HostEntry]:
        return self._entries.get(key)

    def lengths(self, boundary: bytes) -> Dict[int, int]:
        return self._lengths_by_boundary.get(boundary, {})

    def _take(self) -> int:
        self._free.sort()
        return self._free.pop(0)

    def _take_state(self) -> int:
        self._free_state.sort()
        return self._free_state.pop(0)

    def _need(self, entry: Entry) -> Tuple[int, int]:
        """The page slots and the state slots that a store of ``entry`` takes."""
        new_pages = sum(1 for cid in entry.cids if cid is None or cid not in self._pages)
        return new_pages, 1 if entry.slot is not None else 0

    def _fits(self, entry: Entry) -> bool:
        pages, states = self._need(entry)
        return pages <= len(self._free) and states <= len(self._free_state)

    def _evict_one(self, keep: Optional[bytes]) -> bool:
        for key, old in self._entries.items():
            if key != keep and key not in self.pinned:
                self._drop(old)
                return True
        return False

    def _drop(self, entry: HostEntry) -> None:
        del self._entries[entry.key]
        lengths = self._lengths_by_boundary[entry.boundary]
        lengths[entry.num_tokens] -= 1
        if not lengths[entry.num_tokens]:
            del lengths[entry.num_tokens]
            if not lengths:
                del self._lengths_by_boundary[entry.boundary]
        for cid, slot in zip(entry.cids, entry.slots):
            if cid is None:
                self._free.append(slot)
                continue
            ref = self._pages[cid]
            ref[1] -= 1
            if not ref[1]:
                del self._pages[cid]
                self._free.append(slot)
        self._free_state.extend(entry.state)
        self.num_evicted += 1

    def store(self, entry: Entry, ops: List[HostOp]) -> bool:
        """Keep a dropped device END entry; append its spill ops to ``ops``. False when it does not fit."""
        if entry.key in self._entries:
            self._entries.move_to_end(entry.key)
            return True
        if not entry.cids or len(entry.cids) != len(entry.pages):
            return False
        while not self._fits(entry) and self._evict_one(None):
            pass
        if not self._fits(entry):
            self.num_skipped += 1
            return False
        slots = []
        for cid, page in zip(entry.cids, entry.pages):
            if cid is not None and cid in self._pages:
                self._pages[cid][1] += 1
                slots.append(self._pages[cid][0])
                continue
            slot = self._take()
            if cid is not None:
                self._pages[cid] = [slot, 1]
            slots.append(slot)
            ops.append(HostOp(True, "mla", page, slot))
        state = []
        if entry.slot is not None:
            slot = self._take_state()
            state.append(slot)
            ops.append(HostOp(True, "state", entry.slot, slot))
        host = HostEntry(key=entry.key, num_tokens=entry.num_tokens, boundary=entry.boundary, cids=entry.cids,
                         slots=tuple(slots), state=tuple(state), request_id=entry.request_id)
        self._entries[host.key] = host
        lengths = self._lengths_by_boundary.setdefault(host.boundary, {})
        lengths[host.num_tokens] = lengths.get(host.num_tokens, 0) + 1
        self.num_stored += 1
        return True

    def restored(self, key: bytes) -> None:
        """The device holds the entry again: its host copy goes to the front of the host eviction order."""
        if key in self._entries:
            self._entries.move_to_end(key, last=False)
            self.num_restored += 1


class _Chain:
    __slots__ = ("salt", "digests")

    def __init__(self, salt: bytes, tokens: Sequence[int]) -> None:
        self.salt, self.digests = salt, []


class KimiPromptCache:
    """The END entries of finished prompts, their pages and their KDA state slots (see the module docstring).

    Args:
        page_tokens: The tokens of one KV page (the allocator's page: 2048 on the Kimi-K3 prefill, 128 tokens on each
            of its 16 DCP ranks).
        resources: The callbacks for pages and slots (``Resources``).
        hash_unit: The tokens of one chain hash (4096).
        keep_previous: Keep the entry that a request extended in the group of its new entry.
        host: The host tier (``HostTier``), or None.

    Example:
        >>> cache = KimiPromptCache(page_tokens=2048, resources=Resources(free_pages=lambda pages: None))
        >>> cache.num_entries
        0
    """

    def __init__(self, *, page_tokens: int, resources: Resources, hash_unit: int = HASH_UNIT,
                 keep_previous: bool = True, host: Optional[HostTier] = None) -> None:
        if page_tokens <= 0 or hash_unit <= 0:
            raise ValueError(f"The page ({page_tokens} tokens) and the hash unit ({hash_unit} tokens) must be "
                             "positive.")
        self.page_tokens, self.hash_unit, self.keep_previous = int(page_tokens), int(hash_unit), bool(keep_previous)
        self.resources = resources
        self._entries: "OrderedDict[bytes, Entry]" = OrderedDict()
        # {boundary chain hash: {entry length: entry count}}: a lookup hashes a tail only for these lengths.
        self._lengths_by_boundary: Dict[bytes, Dict[int, int]] = {}
        # {key of the newest END entry of a group: the keys of the entries that its group keeps}.
        self._groups: Dict[bytes, Set[bytes]] = {}
        # {page: the count of entries that hold it}.
        self._refs: Dict[int, int] = {}
        self._chains: Dict[str, _Chain] = {}
        # {request id: the key of the entry that it hit} (keep_previous), and the entries that it locks.
        self._hit_keys: Dict[str, bytes] = {}
        self._locked_by: Dict[str, bytes] = {}
        # {page: the running requests that write it past the end of the entries that hold it} (tail_extendable)
        self._claims: Dict[int, Set[str]] = {}
        self._clock = 0
        # The host tier (None: off), the copies that the adapter runs before the next forward, and the device page of
        # each content id that an entry holds (a restore reuses it).
        self.host = host
        self._host_ops: List[HostOp] = []
        self._device_cids: Dict[bytes, int] = {}
        self.num_host_hits = self.num_host_hit_tokens = 0
        self.num_hits = self.num_misses = self.num_hit_tokens = self.num_evictions = 0
        self.num_previous_kept = self.num_previous_hits = 0

    # -- state --------------------------------------------------------------------------------------------------------

    @property
    def num_entries(self) -> int:
        """The count of kept entries."""
        return len(self._entries)

    def entries(self) -> List[Entry]:
        """The entries from the least recent (the next to evict) to the most recent."""
        return list(self._entries.values())

    def keys(self) -> List[bytes]:
        """The keys in eviction order (``entries``)."""
        return list(self._entries)

    def get(self, key: bytes) -> Optional[Entry]:
        """The device entry of ``key``, or None."""
        return self._entries.get(key)

    def held_pages(self) -> int:
        """The pages that the entries hold."""
        return len(self._refs)

    def evictable_pages(self) -> int:
        """The pages that an eviction of every unlocked entry frees: the pages that no locked entry holds."""
        locked = {p for e in self._entries.values() if e.locks for p in e.pages}
        return sum(1 for p in self._refs if p not in locked)

    def held_slots(self) -> int:
        """The KDA state slots that the entries hold."""
        return sum(1 for e in self._entries.values() if e.slot_owner is not None)

    # -- keys ---------------------------------------------------------------------------------------------------------

    def _chain(self, request_id: str, tokens: Sequence[int], salt: bytes, num_tokens: int) -> List[bytes]:
        """The chain hashes of the first full units of ``num_tokens``, kept for each request. A request's tokens only
        grow (its prompt, then its outputs)."""
        chain = self._chains.get(request_id)
        if chain is None or chain.salt != salt:
            chain = self._chains[request_id] = _Chain(salt, ())
        unit, need = self.hash_unit, num_tokens // self.hash_unit
        while len(chain.digests) < need:
            k = len(chain.digests)
            parent = chain.digests[-1] if chain.digests else salt
            chain.digests.append(hashlib.sha256(parent + _token_bytes(tokens[k * unit:(k + 1) * unit])).digest())
        return chain.digests

    def boundary(self, request_id: str, tokens: Sequence[int], salt: bytes, num_tokens: int) -> bytes:
        """The chain hash of the full hash units of the first ``num_tokens`` tokens (the salt if none)."""
        full = num_tokens // self.hash_unit
        digests = self._chain(request_id, tokens, salt, num_tokens)
        return digests[full - 1] if full else salt

    def key(self, request_id: str, tokens: Sequence[int], salt: bytes, num_tokens: int) -> bytes:
        """The key of the first ``num_tokens`` tokens: the chain hash of the full units, the tail and the length.

        Example:
            >>> c = KimiPromptCache(page_tokens=4, resources=Resources(free_pages=lambda p: None), hash_unit=2)
            >>> len(c.key("r", [1, 2, 3], b"s", 3))
            32
        """
        parent = self.boundary(request_id, tokens, salt, num_tokens)
        full = num_tokens // self.hash_unit
        tail = _token_bytes(tokens[full * self.hash_unit:num_tokens])
        return hashlib.sha256(b"end" + parent + tail + int(num_tokens).to_bytes(8, "little")).digest()

    # -- lookup and hits ----------------------------------------------------------------------------------------------

    def lookup(self, request_id: str, tokens: Sequence[int], salt: Optional[bytes],
               max_tokens: Optional[int] = None, accept: Optional[Callable[[int], bool]] = None) -> Optional[Entry]:
        """The longest entry that is a prefix of ``tokens`` and leaves one token or more to compute (and has at most
        ``max_tokens`` tokens, when set), or None. The entry becomes the most recent. ``accept(length)`` (when set) can
        refuse an entry: the lookup then takes a shorter one.

        Example:
            >>> c = KimiPromptCache(page_tokens=4, resources=Resources(free_pages=lambda p: None))
            >>> c.lookup("r9", [5, 6], b"s") is None
            True
        """
        if salt is None or not self._entries:
            return None
        max_tokens = len(tokens) - 1 if max_tokens is None else min(int(max_tokens), len(tokens) - 1)
        if max_tokens <= 0:
            return None
        unit = self.hash_unit
        digests = self._chain(request_id, tokens, salt, max_tokens)
        # From the longest boundary to the shortest: an entry at boundary k has a length in [k * unit, (k + 1) * unit).
        for full in range(max_tokens // unit, -1, -1):
            lengths = self._lengths_by_boundary.get(digests[full - 1] if full else salt)
            if not lengths:
                continue
            for length in sorted(lengths, reverse=True):
                if length > max_tokens:
                    continue
                entry = self._entries.get(self.key(request_id, tokens, salt, length))
                if entry is not None and (accept is None or accept(length)):
                    self._entries.move_to_end(entry.key)
                    return entry
        return None

    def hit(self, request_id: str, entry: Entry) -> Hit:
        """Lock ``entry`` for ``request_id`` and return what the request reuses. One hit a request: a second hit
        releases the first.

        Example:
            >>> c = KimiPromptCache(page_tokens=4, resources=Resources(free_pages=lambda p: None))
            >>> e = c.keep("r1", list(range(10)), b"s", 10, pages=[7, 8, 9], slot_owner=None)
            >>> h = c.hit("r2", c.lookup("r2", list(range(12)), b"s"))
            >>> h.tokens, h.shared_pages, h.tail_page, e.locks
            (10, (7, 8), 9, 1)
        """
        hit = self.take_hit(request_id, entry)
        self.release(request_id)
        entry.locks += 1
        self._locked_by[request_id] = entry.key
        return hit

    def lock(self, key: bytes) -> bool:
        """Lock the entry of ``key`` for one holder (SGLang's tree lock, which names no request). False when the cache
        has no such entry.

        Example:
            >>> c = KimiPromptCache(page_tokens=4, resources=Resources(free_pages=lambda p: None))
            >>> e = c.keep("r1", list(range(10)), b"s", 10, pages=[7, 8, 9], slot_owner=None)
            >>> c.lock(e.key), c.evict(3), c.unlock(e.key), c.evict(3)
            (True, 0, True, 3)
        """
        entry = self._entries.get(key)
        if entry is None:
            return False
        entry.locks += 1
        return True

    def unlock(self, key: bytes) -> bool:
        """Give back one ``lock`` of ``key``. False when the entry is gone or not locked."""
        entry = self._entries.get(key)
        if entry is None or entry.locks <= 0:
            return False
        entry.locks -= 1
        return True

    def take_hit(self, request_id: str, entry: Entry) -> Hit:
        """Record that ``request_id`` reuses ``entry`` (the counters, and the entry that its END entry extends for
        keep_previous), with no lock: the caller holds one (``lock``). Returns what the request reuses."""
        if entry.key not in self._entries:
            raise KeyError(f"{_LOG} The entry of {entry.num_tokens} tokens is not in the cache. Look it up again.")
        self._hit_keys[request_id] = entry.key
        entry.hits += 1
        self.num_hits += 1
        self.num_hit_tokens += entry.num_tokens
        if entry.owner is not None:
            self.num_previous_hits += 1
        whole = entry.num_tokens // self.page_tokens
        tail = entry.pages[whole] if entry.num_tokens % self.page_tokens else None
        return Hit(entry=entry, tokens=entry.num_tokens, shared_pages=tuple(entry.pages[:whole]), tail_page=tail)

    # -- the last page of a hit ---------------------------------------------------------------------------------------

    def tail_extendable(self, entry: Entry) -> bool:
        """Whether a request that hits ``entry`` can write its new rows into the entry's last page in place: the hit
        ends inside the page, no running request writes the page past its own rows (a claim), and no other entry holds
        more tokens of the page. The entry keeps its rows, because the request writes only the rows past the hit. Else
        the request writes into a copy of the page.

        Example:
            >>> c = KimiPromptCache(page_tokens=4, resources=Resources(free_pages=lambda p: None))
            >>> e = c.keep("r1", list(range(10)), b"s", 10, pages=[7, 8, 9], slot_owner=None)
            >>> c.tail_extendable(e), c.claim("r2", 9), c.tail_extendable(e)
            (True, None, False)
        """
        if not entry.num_tokens % self.page_tokens:
            return False
        page = entry.pages[entry.num_tokens // self.page_tokens]
        if self._claims.get(page):
            return False
        return not any(other is not entry and page in other.pages and other.num_tokens > entry.num_tokens
                       for other in self._entries.values())

    def claim(self, request_id: str, page: int) -> None:
        """Record that running ``request_id`` writes rows of ``page`` past the end of the entries that hold it (a hit
        that extends the page in place, or the last page of a running request's chunk-end entry). ``forget`` drops the
        claims of the request."""
        self._claims.setdefault(int(page), set()).add(request_id)

    def miss(self) -> None:
        """Count a request that hits no entry."""
        self.num_misses += 1

    def release(self, request_id: str) -> None:
        """Unlock the entry that ``request_id`` hit. The key of the hit stays for ``keep``."""
        key = self._locked_by.pop(request_id, None)
        entry = self._entries.get(key) if key is not None else None
        if entry is not None and entry.locks > 0:
            entry.locks -= 1

    def forget(self, request_id: str) -> None:
        """Drop every record of a request that ends (after ``keep``, or without it)."""
        self.release(request_id)
        self._hit_keys.pop(request_id, None)
        self._chains.pop(request_id, None)
        for page in [p for p, rids in self._claims.items() if request_id in rids]:
            self._claims[page].discard(request_id)
            if not self._claims[page]:
                del self._claims[page]

    # -- host tier ----------------------------------------------------------------------------------------------------

    def lookup_host(self, request_id: str, tokens: Sequence[int], salt: Optional[bytes], longer_than: int,
                    max_tokens: Optional[int] = None,
                    accept: Optional[Callable[[int], bool]] = None) -> Optional[HostEntry]:
        """The longest host entry that is a prefix of ``tokens``, leaves one token or more to compute (at most
        ``max_tokens`` tokens, when set), and is longer than ``longer_than`` (the device hit), or None. A host entry
        that the device holds is skipped."""
        if self.host is None or salt is None or not self.host.num_entries:
            return None
        max_tokens = len(tokens) - 1 if max_tokens is None else min(int(max_tokens), len(tokens) - 1)
        if max_tokens <= longer_than:
            return None
        unit = self.hash_unit
        digests = self._chain(request_id, tokens, salt, max_tokens)
        for full in range(max_tokens // unit, longer_than // unit - 1, -1):
            lengths = self.host.lengths(digests[full - 1] if full else salt)
            for length in sorted(lengths, reverse=True):
                if not longer_than < length <= max_tokens:
                    continue
                key = self.key(request_id, tokens, salt, length)
                found = self.host.get(key)
                if found is not None and key not in self._entries and (accept is None or accept(length)):
                    return found
        return None

    def restore_pages(self, host_entry: HostEntry) -> int:
        """The new device pages that a restore of ``host_entry`` needs (its pages that no device entry holds). An
        eviction can drop a device page that the restore would share, so read it again after each eviction."""
        return sum(1 for cid in host_entry.cids if cid is None or cid not in self._device_cids)

    def restore(self, request_id: str, host_entry: HostEntry, new_pages: Sequence[int],
                slot: Optional[Tuple[str, int]]) -> Entry:
        """Make ``host_entry`` a device entry again and lock it for ``request_id`` (a hit follows with ``hit``).

        Args:
            request_id: The request that restores it.
            host_entry: The host entry (``lookup_host``).
            new_pages: ``restore_pages(host_entry)`` new device pages; a page whose content a device entry holds is
                shared, the others get the copies.
            slot: ``(owner, slot)`` of a free KDA state slot that the entry takes, or None when the entry has no state.

        Returns:
            The device entry (the most recent). The restore copies are in ``take_host_ops``.
        """
        if len(new_pages) != self.restore_pages(host_entry):
            raise ValueError(f"{_LOG} A restore of {host_entry.num_tokens} tokens needs "
                             f"{self.restore_pages(host_entry)} new pages, not {len(new_pages)}.")
        if host_entry.state and slot is None:
            raise ValueError(f"{_LOG} The host entry of {host_entry.num_tokens} tokens holds a KDA state. Give the "
                             "restore a free state slot.")
        fresh = iter(new_pages)
        pages = []
        for cid, host_slot in zip(host_entry.cids, host_entry.slots):
            if cid is not None and cid in self._device_cids:
                pages.append(self._device_cids[cid])
                continue
            page = next(fresh)
            pages.append(page)
            self._host_ops.append(HostOp(False, "mla", page, host_slot))
        entry = Entry(key=host_entry.key, num_tokens=host_entry.num_tokens, pages=tuple(pages),
                      boundary=host_entry.boundary, created=self._clock, request_id=host_entry.request_id,
                      cids=host_entry.cids)
        if slot is not None and host_entry.state:
            entry.slot_owner, entry.slot = slot
            for host_slot in host_entry.state:
                self._host_ops.append(HostOp(False, "state", entry.slot, host_slot))
        self._hold(entry)
        self._insert(entry)
        self.host.restored(entry.key)
        self.host.pinned.discard(entry.key)
        self.num_host_hits += 1
        self.num_host_hit_tokens += entry.num_tokens
        logger.info("%s Restore an entry of %d tokens from the host tier for request %s (%d of its %d pages copied "
                    "to the device). Host hits %d (%d tokens); %d of %d host page slots used.", _LOG,
                    entry.num_tokens, request_id, len(new_pages), len(pages), self.num_host_hits,
                    self.num_host_hit_tokens, self.host.num_slots - self.host.free_slots, self.host.num_slots)
        return entry

    def take_host_ops(self) -> List[HostOp]:
        """The host copies since the last call, in order: run them before the next forward (a spill reads a page that
        an allocation of this step can take; a restore fills the pages that the forward reads)."""
        ops, self._host_ops = self._host_ops, []
        return ops

    # -- keep ---------------------------------------------------------------------------------------------------------

    def keep(self, request_id: str, tokens: Sequence[int], salt: Optional[bytes], num_tokens: int,
             pages: Sequence[int], slot_owner: Optional[str], kind: str = "end") -> Optional[Entry]:
        """Keep the first ``num_tokens`` tokens of a finished request as an END entry. With ``kind="chunk_end"`` it
        keeps the KDA state at a chunk end of a running request instead: the entry goes first in the eviction order,
        it does not take the request's hit for ``keep_previous``, and it stays on the device only.

        Args:
            request_id: The request.
            tokens: Its prompt (at least ``num_tokens`` tokens).
            salt: Its key salt (``request_salt``); None keeps nothing.
            num_tokens: The tokens that the request computed.
            pages: The pages of its row that hold them, ``ceil(num_tokens / page_tokens)`` or more (the rest stay
                with the request).
            slot_owner: The owner name of the request's KDA state slot, or None to keep no state. The cache adopts
                the slot through ``Resources.adopt_slot`` only for a new entry.

        Returns:
            The new entry, or None when the cache keeps nothing new (an entry with the same tokens exists: it becomes
            the most recent, and the request keeps its pages and its slot).
        """
        need = -(-int(num_tokens) // self.page_tokens)
        if salt is None or num_tokens <= 0 or num_tokens > len(tokens) or len(pages) < need:
            return None
        key = self.key(request_id, tokens, salt, num_tokens)
        self._clock += 1
        existing = self._entries.get(key)
        if existing is not None:
            self._entries.move_to_end(key)
            existing.kind = "end"
            if existing.request_id != request_id:
                existing.request_id = request_id
                logger.info("%s Refresh the entry of %d tokens of request %s. An entry with the same tokens exists.",
                            _LOG, num_tokens, request_id)
            return None
        entry = Entry(key=key, num_tokens=int(num_tokens), pages=tuple(int(p) for p in pages[:need]),
                      boundary=self.boundary(request_id, tokens, salt, num_tokens), created=self._clock,
                      request_id=request_id)
        if slot_owner is not None:
            owner = f"kimi-prompt:{key.hex()[:16]}"
            entry.slot = self.resources.adopt_slot(slot_owner, owner)
            if entry.slot is None:
                raise RuntimeError(f"{_LOG} Request {request_id} gave no KDA state slot to adopt. Give the cache the "
                                   "slot of each request (StateSlots.transfer), or keep the entry without state.")
            entry.slot_owner = owner
        entry.kind = kind
        if self.host is not None:
            entry.cids = self._page_cids(request_id, tokens, salt, entry.num_tokens)
        self._hold(entry)
        if kind != "end":
            self._insert(entry, last=False)
            logger.info("%s Keep the chunk-end state of %d tokens of request %s. Entries %d.", _LOG, num_tokens,
                        request_id, len(self._entries))
            return entry
        self._insert(entry)
        extended = self._hit_keys.pop(request_id, None)
        if extended in self._entries and self.keep_previous:
            self._keep_previous(extended, key)
        elif extended in self._entries:
            # The new entry extends the old one (the next turn): evict the old entry first. The new entry holds the
            # shared pages, so that eviction frees only the old tail page and the old state.
            self._entries.move_to_end(extended, last=False)
        logger.info("%s Keep the cache of %d tokens of request %s (%d pages). Entries %d, held pages %d.", _LOG,
                    num_tokens, request_id, len(entry.pages), len(self._entries), len(self._refs))
        return entry

    def _page_cids(self, request_id: str, tokens: Sequence[int], salt: bytes, num_tokens: int):
        whole = num_tokens // self.page_tokens
        cids = [self.key(request_id, tokens, salt, (p + 1) * self.page_tokens) for p in range(whole)]
        return tuple(cids + ([None] if num_tokens % self.page_tokens else []))

    def _hold(self, entry: Entry) -> None:
        for p in entry.pages:
            self._refs[p] = self._refs.get(p, 0) + 1
        for cid, p in zip(entry.cids, entry.pages):
            if cid is not None:
                self._device_cids.setdefault(cid, p)

    def _insert(self, entry: Entry, last: bool = True) -> None:
        self._entries[entry.key] = entry
        if not last:
            self._entries.move_to_end(entry.key, last=False)
        lengths = self._lengths_by_boundary.setdefault(entry.boundary, {})
        lengths[entry.num_tokens] = lengths.get(entry.num_tokens, 0) + 1

    def _keep_previous(self, extended: bytes, key: bytes) -> None:
        """``keep_previous``: ``extended`` goes into the group of the new entry ``key``. The other members of its old
        group go to the front of the eviction order."""
        entry = self._entries[extended]
        if entry.kind != "end":
            self._entries.move_to_end(extended, last=False)
            return
        old_owner = entry.owner if entry.owner is not None else extended
        old_members = [old_owner, *sorted(self._groups.pop(old_owner, set()))]
        if extended not in old_members:
            old_members.append(extended)
        self._groups.pop(extended, None)
        for member in old_members:
            old = self._entries.get(member)
            if old is not None:
                old.owner = None
        entry.owner = key
        self._groups[key] = {extended}
        self._entries.move_to_end(extended)
        self._entries.move_to_end(key)
        for member in reversed(old_members):
            if member != extended and member in self._entries:
                self._entries.move_to_end(member, last=False)
        self.num_previous_kept += 1

    # -- eviction -----------------------------------------------------------------------------------------------------

    def _drop(self, entry: Entry, spill: bool = True) -> List[int]:
        host_note = ""
        if spill and self.host is not None and entry.kind == "end":
            host_note = self._host_note(self.host.store(entry, self._host_ops))
        del self._entries[entry.key]
        lengths = self._lengths_by_boundary[entry.boundary]
        lengths[entry.num_tokens] -= 1
        if not lengths[entry.num_tokens]:
            del lengths[entry.num_tokens]
            if not lengths:
                del self._lengths_by_boundary[entry.boundary]
        self._unlink(entry)
        freed = []
        for p in entry.pages:
            count = self._refs[p] - 1
            if count:
                self._refs[p] = count
            else:
                del self._refs[p]
                freed.append(p)
        for cid, p in zip(entry.cids, entry.pages):
            if cid is not None and p in freed and self._device_cids.get(cid) == p:
                del self._device_cids[cid]
        self.resources.on_drop(entry, freed)
        if entry.slot_owner is not None:
            self.resources.free_slot(entry.slot_owner)
        if freed:
            self.resources.free_pages(freed)
        self.num_evictions += 1
        logger.info("%s Evict an entry of %d tokens (%d hits, %d pages freed). Entries left: %d.%s", _LOG,
                    entry.num_tokens, entry.hits, len(freed), len(self._entries), host_note)
        return freed

    def _host_note(self, stored: bool) -> str:
        """The sentence of an eviction log line about the host tier: it keeps the entry, or it cannot."""
        host = self.host
        used = host.num_slots - host.free_slots
        states = host.num_state_slots - host.free_state_slots
        if stored:
            return (f" The host tier keeps it: {used} of {host.num_slots} host page slots and {states} of "
                    f"{host.num_state_slots} host state slots used, {host.num_entries} host entries, "
                    f"{host.num_evicted} host evictions.")
        return (f" The host tier cannot keep it ({used} of {host.num_slots} host page slots and {states} of "
                f"{host.num_state_slots} host state slots used, {host.num_skipped} skipped entries). Give the host "
                f"tier more slots ({_HOST_SLOTS_KNOB}, {_HOST_STATE_SLOTS_KNOB}).")

    def _unlink(self, entry: Entry) -> None:
        if entry.owner is not None:
            members = self._groups.get(entry.owner)
            if members is not None:
                members.discard(entry.key)
                if not members:
                    del self._groups[entry.owner]
            entry.owner = None
        for member in self._groups.pop(entry.key, ()):
            kept = self._entries.get(member)
            if kept is not None and kept.owner == entry.key:
                kept.owner = None

    def _victims(self):
        return [e for e in self._entries.values() if not e.locks]

    def evict(self, num_pages: int) -> int:
        """Drop the least recent unlocked entries until ``num_pages`` pages are free, or no unlocked entry is left.
        Returns the pages freed.

        Example:
            >>> c = KimiPromptCache(page_tokens=4, resources=Resources(free_pages=lambda p: None))
            >>> _ = c.keep("r1", list(range(10)), b"s", 10, pages=[7, 8, 9], slot_owner=None)
            >>> c.evict(1), c.num_entries
            (3, 0)
        """
        freed = 0
        for entry in self._victims():
            if freed >= num_pages:
                break
            freed += len(self._drop(entry))
        return freed

    def evict_until(self, done: Callable[[], bool]) -> int:
        """Drop the least recent unlocked entries one at a time until ``done()`` is true (an allocation fits), as
        an allocation does. Returns the count of dropped entries.

        Example:
            >>> c = KimiPromptCache(page_tokens=4, resources=Resources(free_pages=lambda p: None))
            >>> _ = c.keep("r1", list(range(10)), b"s", 10, pages=[7, 8, 9], slot_owner=None)
            >>> c.evict_until(lambda: c.num_entries == 0)
            1
        """
        dropped = 0
        for entry in self._victims():
            if done():
                break
            self._drop(entry)
            dropped += 1
        return dropped

    def drop(self, key: bytes) -> bool:
        """Drop the unlocked entry of ``key`` now: its state slot and the pages that no other entry holds go back.
        False when the cache has no such entry or a request holds it.

        Example:
            >>> c = KimiPromptCache(page_tokens=4, resources=Resources(free_pages=lambda p: None))
            >>> e = c.keep("r1", list(range(10)), b"s", 10, pages=[7, 8, 9], slot_owner=None)
            >>> c.drop(e.key), c.num_entries
            (True, 0)
        """
        entry = self._entries.get(key)
        if entry is None or entry.locks:
            return False
        self._drop(entry)
        return True

    def free_state_slot(self) -> bool:
        """Drop the least recent unlocked entry that holds a KDA state slot (an admission needs the slot). False when
        no such entry exists."""
        for entry in self._victims():
            if entry.slot_owner is not None:
                self._drop(entry)
                return True
        return False

    def reset(self, free_pages: bool = True) -> None:
        """Drop every entry (locked ones too) and every request record. With ``free_pages`` False the pages do not go
        back through ``Resources.free_pages`` (SGLang clears its allocator after a flush); the state slots do."""
        callback = self.resources.free_pages
        if not free_pages:
            self.resources.free_pages = lambda pages: None
        try:
            for entry in list(self._entries.values()):
                entry.locks = 0
                self._drop(entry, spill=False)
        finally:
            self.resources.free_pages = callback
        self._host_ops.clear()
        self._hit_keys.clear()
        self._locked_by.clear()
        self._chains.clear()
        self._claims.clear()
