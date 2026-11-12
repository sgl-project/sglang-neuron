# SPDX-License-Identifier: Apache-2.0
"""Radix-cache backend ``kimi-prompt``: the Kimi-K3 prompt cache (mem_cache/kimi_prompt_cache) as SGLang's prefix
cache on the Kimi-K3 PD prefill.

The backend is not a tree. It keeps END entries of finished prompts and gives a new prompt the longest entry that is
its prefix, at any token (``KimiPromptCache``). The chunk-keyed tree (mem_cache/chunk_keyed_tree) hits only at a whole
4096-token chunk, so each next turn of a session computes up to 4,095 tokens of the turn before again; here a next turn
starts at the exact end of the turn before.

The pools are SGLang's, as the Kimi-K3 stage graphs read them (models/kimi_k3/sglang_model.bind_kv_cache):

* KV pages: SGLang's allocator pages, under DCP a virtual page of ``128 x --dcp-size`` tokens (each rank keeps its
  128-row block of it). A page holds this rank's MLA rows of every MLA layer of the stage and, on the drafter writer
  stage, the drafter rows (the MLA pool's ``page_companions``). So an entry's pages also keep its drafter rows.
* KDA state: SGLang's state slots (the request table's mamba pool, ``NeuronStatePool``). The backend reports
  ``supports_mamba``: it gives back a request's state slots itself, and it reports its slots to the scheduler.

SGLang drives it through ``BasePrefixCache``:

* ``match_prefix`` looks up (no lock, no allocation: the scheduler repeats it for each waiting request). The match
  handle names the entry. ``cache_protected_len`` is the whole pages of the hit, so the request owns its copy of the
  page that holds the hit's last tokens.
* ``lock`` (at admission) locks the entry until ``release_kv_cache`` unlocks it, so its pages and its KDA state stay.
  A handle of a host entry pins it on the host.
* ``prefix_device_indices`` (at allocation, once) restores a host entry first. The page that holds the hit's last
  tokens stays shared when no one else uses its rows past the hit (``KimiPromptCache.tail_extendable``: the request
  then writes its rows into it in place); else it is copied into a page of the request (``PageCopies``, before the
  request's first forward). The request gets its own state slot with upstream's deferred copy-on-write of the entry's state (``req.kv.mamba_cow_src_index``, which
  the model runner copies before the forward), and returns the row prefix: the shared pages, then the copy.
* ``checkpoint`` of a finished request (``release_kv_cache`` after the KV transfer) keeps the END entry of its
  ``up_to`` computed tokens: the entry takes the row pages and adopts the request's state slot (the state of its last
  computed token), and ``cache_protected_len`` covers the row, so the release frees none of it.
* With chunk ends on, each forward that does not end the prompt also writes its end state into the request's tracking
  slot (upstream's extra buffer, ``track_entry``), and ``checkpoint`` of the running request keeps a chunk-end entry
  that takes that slot (``donate_tracking_slot``): no device copy.
* ``evict`` drops the least recent unlocked entries. With the host tier on, a dropped entry is copied to the host
  first (``HostCopies``), and the copies complete before the allocator or the state slots get the pages and the slot
  back.

Each PP stage runs its own backend on the same requests in the same order, with the same page, slot and host counts, so
the stages keep the same entries.

Example:
    >>> BACKEND
    'kimi-prompt'
"""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    MatchPrefixParams,
    MatchResult,
)

from sglang_neuron.mem_cache.kimi_prompt_cache import HostOp, HostTier, KimiPromptCache, Resources, request_salt

logger = logging.getLogger(__name__)

BACKEND = "kimi-prompt"
HOST_SLOTS_ENV = "SGLANG_NEURON_KIMI_PROMPT_HOST_SLOTS"
"""The host page slots of each rank (0 or unset: no host tier). A slot holds this rank's rows of one KV page."""
HOST_STATE_SLOTS_ENV = "SGLANG_NEURON_KIMI_PROMPT_HOST_STATE_SLOTS"
"""The host state slots of each rank (default: a twentieth of the page slots, at least 1 with a host tier). A slot
holds this rank's KDA state of one entry."""
KEEP_PREVIOUS_ENV = "SGLANG_NEURON_KIMI_PROMPT_KEEP_PREVIOUS"
CHUNK_ENDS_ENV = "SGLANG_NEURON_KIMI_PROMPT_CHUNK_ENDS"
"""``1`` (the default): keep the KDA state at each chunk end of a running request as a chunk-end entry. ``0``: END
entries only."""
_LOG = "[kimi_prompt]"


class _Handle:
    """A match handle: the key of a device entry, of a host entry (``host``), or None (a miss)."""

    __slots__ = ("key", "host", "locks")

    def __init__(self, key: Optional[bytes], host: bool = False) -> None:
        self.key, self.host, self.locks = key, host, 0

    def __repr__(self) -> str:
        where = "host" if self.host else "device"
        return "kimi-prompt:miss" if self.key is None else f"kimi-prompt:{where}:{self.key.hex()[:12]}"


def _page_view(tensor: torch.Tensor, num_pages: int) -> torch.Tensor:
    """``tensor``'s first ``num_pages`` pages as ``[num_pages, page bytes]``, one row a page (a view)."""
    flat = tensor.reshape(-1) if tensor.is_contiguous() else None
    if flat is None:
        raise ValueError("a KV page view needs a contiguous pool tensor")
    per_page = flat.numel() // num_pages if tensor.shape[0] != num_pages else tensor[0].numel()
    return flat[:num_pages * per_page].view(num_pages, per_page)


VERIFY_COPY_ENV = "SGLANG_NEURON_KIMI_PROMPT_VERIFY_COPY"
"""Debug only: ``1`` reads back each page copy of a hit and compares it with its source, and raises on a difference.
The read-back waits for the device, so take no performance numbers with it."""


class PageCopies:
    """The tensors of one KV page on this rank, and the copy of a page into another one before the next forward.

    ``page_tensors`` is one ``[pages, page bytes]`` view a tensor: each MLA layer of the stage, then the tensors the
    model keeps at the same pages (the drafter rows). ``copy(src, dst)`` moves every tensor's page ``src`` into page
    ``dst`` through one host page (two runtime batch DMAs on host_pool.TransferEngine, torch copies in CPU mode), and
    returns when the bytes have landed. Each DMA waits for the graphs launched before it, so the copy lands after the
    zero fill of the fresh page and before the request's first forward. An eager device copy of a page needs a
    compiled program over the whole pool, and the device has no memory for it beside the stage graphs."""

    def __init__(self, page_tensors: Sequence[torch.Tensor], verify: Optional[bool] = None) -> None:
        from sglang_neuron.mem_cache.host_pool import TransferEngine, pinned_host_like

        self.page_tensors = list(page_tensors)
        self.verify = os.environ.get(VERIFY_COPY_ENV, "0") == "1" if verify is None else bool(verify)
        on_neuron = bool(self.page_tensors) and self.page_tensors[0].device.type == "neuron"
        self.host = [pinned_host_like(t, 1, on_neuron) for t in self.page_tensors]
        self.engine = TransferEngine(self.page_tensors, self.host, [t[0].numel() * t.element_size()
                                                                    for t in self.page_tensors])
        self.engine.name, self.engine.verify = "kimi-prompt-copy", False
        self.copies = 0

    def copy(self, src: int, dst: int) -> None:
        if src == dst:
            raise ValueError(f"{_LOG} A page copy from page {src} into itself.")
        self.engine.wait(self.engine.submit(True, [int(src)], [0]))
        self.engine.wait(self.engine.submit(False, [int(dst)], [0]))
        self.copies += 1
        if self.verify:
            for i, view in enumerate(self.page_tensors):
                if not torch.equal(view[int(src)].cpu(), view[int(dst)].cpu()):
                    raise RuntimeError(f"{_LOG} The copy of page {src} into page {dst} differs in page tensor {i}. "
                                       "The page copy is not exact on this runtime.")


def pool_page_tensors(kv_pool: Any) -> List[torch.Tensor]:
    """The ``[pages, page bytes]`` views of one KV page on this rank: each layer of the full-attention leaf of
    ``kv_pool`` (SGLang's hybrid linear pool's MLA leaf), then its ``page_companions`` (the drafter rows on the drafter
    writer stage)."""
    from sglang_neuron.mem_cache.memory_pool import full_attention_pool

    leaf = full_attention_pool(kv_pool)
    pages = int(leaf.num_blocks)
    views = [_page_view(rows, pages) for rows in leaf.kv_buffer]
    views += [_page_view(t, pages) for t in getattr(leaf, "page_companions", ())]
    return views


def state_slot_tensors(state_pool: Any) -> List[torch.Tensor]:
    """The ``[slots, slot bytes]`` views of one KDA state slot on this rank: each state component's tensor of each
    layer (``NeuronStatePool.tensors``, without the working bank)."""
    out = []
    for name, tensor in state_pool.tensors.items():
        if name.startswith("work."):
            continue
        for layer in range(tensor.shape[0]):
            out.append(_page_view(tensor[layer], int(tensor.shape[1])))
    return out


class HostCopies:
    """The device side of the host tier: run ``HostOp`` copies to the end before returning (the synchronous form).

    ``EngineHostCopies`` runs them on host_pool.TransferEngine; a test gives its own."""

    def run(self, ops: Sequence[HostOp]) -> None:
        raise NotImplementedError


class EngineHostCopies(HostCopies):
    """Host regions beside the KV pages and the state slots of this rank, and the copies between them, on
    host_pool.TransferEngine (the runtime's batch DMA on a NeuronCore, torch copies in CPU mode). Each engine waits
    for the graphs launched before its submit, so a spill reads what they wrote and a restore lands after the fresh
    page's zero fill. A host page slot holds one KV page of every page tensor; a host state slot one state slot of
    every state tensor."""

    def __init__(self, page_tensors: Sequence[torch.Tensor], state_tensors: Sequence[torch.Tensor], num_slots: int,
                 num_state_slots: int) -> None:
        from sglang_neuron.mem_cache.host_pool import TransferEngine, pinned_host_like

        on_neuron = page_tensors[0].device.type == "neuron"
        self.pages = list(page_tensors)
        self.states = list(state_tensors)
        self.host_pages = [pinned_host_like(t, int(num_slots), on_neuron) for t in self.pages]
        self.host_states = [pinned_host_like(t, int(num_state_slots), on_neuron) for t in self.states]
        self.bytes = sum(r.numel() * r.element_size() for r in self.host_pages + self.host_states)
        self.page_engine = TransferEngine(self.pages, self.host_pages, [t[0].numel() * t.element_size()
                                                                        for t in self.pages])
        self.state_engine = (TransferEngine(self.states, self.host_states, [t[0].numel() * t.element_size()
                                                                            for t in self.states])
                             if self.states and num_state_slots else None)
        self.page_engine.name, self.page_engine.verify = "kimi-prompt-pages", False
        if self.state_engine is not None:
            self.state_engine.name, self.state_engine.verify = "kimi-prompt-states", False

    def run(self, ops: Sequence[HostOp]) -> None:
        for spill in (True, False):  # every spill first: a restore can fill a page that a spill of this step reads
            for kind, engine in (("mla", self.page_engine), ("state", self.state_engine)):
                part = [op for op in ops if op.spill == spill and op.kind == kind]
                if not part:
                    continue
                if engine is None:
                    raise RuntimeError(f"{_LOG} The host tier has no state slots for {len(part)} state copies.")
                engine.wait(engine.submit(spill, [op.device for op in part], [op.host_slot for op in part]))


class KimiPromptBackend(BasePrefixCache):
    """SGLang's prefix cache on the Kimi-K3 prefill: END entries over SGLang's KV pages and state slots (see the module
    docstring).

    Args:
        params: SGLang's ``CacheInitParams``.
        keep_previous: Keep the entry that a request extended in the group of its new entry.
        chunk_ends: Keep a chunk-end entry at each chunk end of a running request (``track_entry`` must be on).
        host_slots: The host page slots of the host tier (0: off).
        host_state_slots: The host state slots of the host tier.
        page_copies: The device page copy (default: ``PageCopies`` over the KV pool's page tensors).
        host_copies: The device side of the host tier (default ``EngineHostCopies`` when ``host_slots`` is set).
    """

    def __init__(self, params: Any, *, keep_previous: bool = True, chunk_ends: bool = True, host_slots: int = 0,
                 host_state_slots: int = 0, page_copies: Optional[PageCopies] = None,
                 host_copies: Optional[HostCopies] = None) -> None:
        from sglang.srt.mem_cache.static_kv_storage import kv_storage_of

        from sglang_neuron.mem_cache.state_pool import state_pool_of

        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = int(self.token_to_kv_pool_allocator.page_size)
        self.logical_page_size = self.page_size
        self.disable = False
        self.sliding_window_size = None
        self.state_pool = state_pool_of(self.req_to_token_pool)
        if self.state_pool is None:
            raise ValueError(f"{_LOG} The request table has no Neuron state pool. The backend keeps the KDA state of "
                             "the Kimi-K3 prefill in SGLang's state slots. Use it on a Kimi-K3 prefill server only.")
        self.storage = kv_storage_of(self.req_to_token_pool, self.token_to_kv_pool_allocator)
        self.state_sub_pool = self.storage.tree_sub_pool("MAMBA")
        self.chunk_ends = bool(chunk_ends)
        self.root_node = _Handle(None)
        self._chunk_locks: Dict[str, List[bytes]] = {}
        # {owner name of an entry's slot: the state slot}, and the slot that the next keep adopts
        self._owned: Dict[str, int] = {}
        self._adopting: Optional[int] = None
        host = HostTier(int(host_slots), int(host_state_slots)) if host_slots else None
        self.core = KimiPromptCache(page_tokens=self.page_size, keep_previous=keep_previous, host=host,
                                    resources=Resources(free_pages=self._free_pages, adopt_slot=self._adopt_slot,
                                                        free_slot=self._free_owned_slot))
        self._page_copies = page_copies
        self.host_copies = host_copies
        if host is not None and host_copies is None:
            kv_pool = self.token_to_kv_pool_allocator.get_kvcache()
            self.host_copies = EngineHostCopies(pool_page_tensors(kv_pool), state_slot_tensors(self.state_pool),
                                                int(host_slots), int(host_state_slots))
            logger.info("%s the host tier holds %d page slots and %d state slots on this rank (%.2f GiB)", _LOG,
                        int(host_slots), int(host_state_slots), self.host_copies.bytes / (1 << 30))
        self.restores_refused = 0  # requests that found a host entry but had no device room to restore it
        self.tail_in_place = self.tail_copies = 0  # hits inside a page: the page extended in place, or copied
        self.chunk_end_reclaims = 0  # older chunk-end entries of running requests that gave their state slot back
        if getattr(params, "enable_metrics", False):
            self.init_metrics_collector()

    @property
    def page_copies(self) -> PageCopies:
        """The device page copy, built at its first use: the drafter row pools join the MLA pool's pages when the model
        binds its pools."""
        if self._page_copies is None:
            self._page_copies = PageCopies(pool_page_tensors(self.token_to_kv_pool_allocator.get_kvcache()))
        return self._page_copies

    # -- pages --------------------------------------------------------------------------------------------------------

    def _free_pages(self, pages: List[int]) -> None:
        # Spill before free: the spill copies of a dropped entry read its pages and its state slot, so they complete
        # before the pages and the slot go back. The allocator zeroes a page as it hands it out again, and a copy that
        # ran after that would read the zeros.
        self._run_host_ops()
        alloc = self.token_to_kv_pool_allocator
        alloc.free(torch.tensor([int(p) * self.page_size for p in pages], dtype=torch.int64, device=alloc.device))

    def _alloc_pages(self, count: int) -> List[int]:
        """``count`` free pages, after an eviction when the allocator is short. Raises when they do not fit."""
        if count <= 0:
            return []
        alloc = self.token_to_kv_pool_allocator
        if alloc.available_size() < count * self.page_size:
            self.evict(EvictParams(num_tokens=count * self.page_size - alloc.available_size()))
        self._run_host_ops()
        slots = alloc.alloc(count * self.page_size)
        if slots is None:
            raise RuntimeError(f"{_LOG} {count} KV pages are not free after the eviction of every unlocked entry. "
                               "Admit fewer requests at once, or give the prefill more KV pages.")
        return [int(s) // self.page_size for s in slots[::self.page_size].tolist()]

    def _run_host_ops(self) -> None:
        ops = self.core.take_host_ops()
        if ops:
            if self.host_copies is None:
                raise RuntimeError(f"{_LOG} The host tier made {len(ops)} copies, but no device side runs them.")
            self.host_copies.run(ops)

    # -- state slots --------------------------------------------------------------------------------------------------

    def _take_slot(self) -> Optional[int]:
        """A free state slot, after the eviction of an entry's slot when none is free; None when no slot can be freed."""
        slot = self.storage.take_state_slots(self.state_sub_pool, 1)
        if slot is None and self.free_state_slot():
            slot = self.storage.take_state_slots(self.state_sub_pool, 1)
        return None if slot is None else int(slot.reshape(-1)[0])

    def _slot_tensor(self, slot: int) -> torch.Tensor:
        alloc = self.token_to_kv_pool_allocator.state_slots
        device = getattr(alloc, "device", "cpu")
        return torch.tensor([int(slot)], dtype=torch.int64, device=device)

    def _give_slot(self, slot: int) -> None:
        self._run_host_ops()  # the spill of a dropped entry's state reads the slot (see _free_pages)
        self.storage.free_state_slots(self.state_sub_pool, self._slot_tensor(slot))

    def _adopt_slot(self, request_id: str, owner: str) -> Optional[int]:
        slot, self._adopting = self._adopting, None
        if slot is None:
            return None
        self._owned[owner] = int(slot)
        return int(slot)

    def _free_owned_slot(self, owner: str) -> None:
        slot = self._owned.pop(owner, None)
        if slot is not None:
            self._give_slot(slot)

    def _release_request_state(self, req: Any, *, keep_working: bool) -> None:
        """Give back the request's state slots: all of them, or (``keep_working``) all but its working slot, which an
        END entry adopted. The request then holds no state slot."""
        kv = req.kv
        if kv.mamba_pool_idx is None:
            return
        if not keep_working:
            self.storage.release_state(self.state_sub_pool, req)
            return
        # upstream's StateStorage.release without the working slot
        buffer = getattr(kv, "mamba_ping_pong_track_buffer", None)
        if buffer is not None:
            empty = kv.mamba_ping_pong_empty or [False] * int(buffer.numel())
            for position, is_empty in enumerate(empty):
                if not is_empty:
                    self.storage.free_state_slots(self.state_sub_pool, buffer[position:position + 1].clone())
            kv.mamba_ping_pong_track_buffer = None
            kv.mamba_ping_pong_empty = None
            kv.mamba_next_track_idx = None
            kv.mamba_last_track_idx = None
            kv.mamba_last_track_seqlen = None
            kv.mamba_prev_track_seqlen = None
        kv.mamba_cow_src_index = None
        kv.mamba_needs_clear = False
        kv.mamba_pool_idx = None

    # -- match --------------------------------------------------------------------------------------------------------

    @staticmethod
    def _salt(req: Any, key: Any) -> bytes:
        lora = getattr(req, "lora_id", None) if req is not None else None
        return request_salt(lora or 0, key.cache_salt or getattr(req, "cache_salt", None), key.extra_key)

    def reset(self) -> None:
        """Drop every entry with no page free (SGLang clears the allocator and the state slots after this)."""
        self.core.reset(free_pages=False)
        self._owned.clear()
        self._chunk_locks.clear()

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        key, req = params.key, params.req
        tokens = key.token_ids
        limit = len(key)
        rid = req.rid if req is not None else "kimi-prompt-probe"
        salt = self._salt(req, key)
        entry = self.core.lookup(rid, tokens, salt, max_tokens=limit)
        length, handle = (entry.num_tokens, _Handle(entry.key)) if entry is not None else (0, self.root_node)
        if self.core.host is not None:
            found = self.core.lookup_host(rid, tokens, salt, length, max_tokens=limit)
            if found is not None and self._restore_fits(found):
                length, handle = found.num_tokens, _Handle(found.key, host=True)
            elif found is not None and req is not None:
                self.restores_refused += 1
                logger.info("%s request %s cannot restore the host entry of %d tokens: the device has no room for "
                            "its pages or no KDA state slot. It takes the device hit of %d tokens. %d requests have "
                            "this.", _LOG, rid, found.num_tokens, length, self.restores_refused)
        if req is None:
            self.core.forget(rid)
        return MatchResult(device_prefix_len=length, last_device_node=handle, last_host_node=handle,
                           best_match_node=handle, cache_protected_len=length // self.page_size * self.page_size)

    def _restore_fits(self, host_entry: Any) -> bool:
        need = self.core.restore_pages(host_entry) + 1  # and the copy of the last page
        free = self.token_to_kv_pool_allocator.available_size() // self.page_size
        return need <= free + self.core.evictable_pages() and self._slot_obtainable(2)

    def _slot_obtainable(self, count: int = 1) -> bool:
        """True when ``count`` state slots are free or can be freed: unlocked entries hold them, or running requests
        have older chunk-end entries (``_older_chunk_end``). A locked entry's slot does not count."""
        free = int(self.token_to_kv_pool_allocator.state_slots.available_size())
        free += sum(1 for e in self.core.entries() if e.slot_owner is not None and not e.locks)
        if free >= count:
            return True
        return free + (1 if self._older_chunk_end() is not None else 0) >= count

    def _older_chunk_end(self) -> Optional[Tuple[str, bytes]]:
        """``(request id, key)`` of a chunk-end entry of a running request that a newer chunk-end entry of the same
        request covers, and that no other request holds, or None. The newer entry holds every page of the older one
        (both are the request's row up to their ends), so the older one can go with its slot only. It takes the
        request with the most chunk-end entries, and its newest older entry, so that the early entries stay: sessions
        share prefixes there."""
        best = None
        for rid, keys in self._chunk_locks.items():
            if len(keys) < 2 or (best is not None and len(keys) <= len(self._chunk_locks[best[0]])):
                continue
            for key in reversed(keys[:-1]):
                entry = self.core.get(key)
                if entry is not None and entry.locks == 1 and entry.slot_owner is not None:
                    best = (rid, key)
                    break
        return best

    def _reclaim_chunk_end_slot(self) -> bool:
        """Free a state slot from a running request's older chunk-end entry (``_older_chunk_end``). False when no such
        entry exists."""
        found = self._older_chunk_end()
        if found is None:
            return False
        rid, key = found
        self._chunk_locks[rid].remove(key)
        self.core.unlock(key)
        if not self.core.drop(key):
            return False
        self.chunk_end_reclaims += 1
        return True

    def supports_prefix_sharing(self) -> bool:
        return True

    def supports_mamba(self) -> bool:
        return True

    def dfs_weight_order(self, node_handles: Sequence[Any]) -> list:
        return list(range(len(node_handles)))

    def root_node_handle(self, extra_key: Optional[str] = None) -> Any:
        return self.root_node

    # -- locks --------------------------------------------------------------------------------------------------------

    def inc_lock_ref(self, node: Any) -> IncLockRefResult:
        if node is not None and node.key is not None:
            if node.host:
                self.core.host.pinned.add(node.key)
            elif not self.core.lock(node.key):
                raise RuntimeError(f"{_LOG} The entry of {node} is gone before its lock. Match the request again.")
            node.locks += 1
        return IncLockRefResult(delta=0)

    def dec_lock_ref(self, node: Any, params: Any = None) -> DecLockRefResult:
        if node is not None and node.key is not None and node.locks > 0:
            node.locks -= 1
            if node.host:
                if not node.locks:
                    self.core.host.pinned.discard(node.key)
            else:
                self.core.unlock(node.key)
        return DecLockRefResult(delta=0)

    # -- allocation ---------------------------------------------------------------------------------------------------

    def _restore(self, req: Any, node: _Handle) -> None:
        """Make the host entry of ``node`` a device entry; the handle's locks move to it."""
        host_entry = self.core.host.get(node.key)
        if host_entry is None:
            raise RuntimeError(f"{_LOG} The host entry of request {req.rid} is gone before its restore.")
        slot = None
        if host_entry.state:
            taken = self._take_slot()
            if taken is None:
                raise RuntimeError(f"{_LOG} No state slot is free for the restore of request {req.rid}. Admit fewer "
                                   "requests at once, or give the prefill more state slots (--max-mamba-cache-size).")
            owner = f"kimi-prompt-restore:{node.key.hex()[:16]}"
            self._owned[owner] = taken
            slot = (owner, taken)
        count = self.core.restore_pages(host_entry)
        pages = self._alloc_pages(count)
        if self.core.restore_pages(host_entry) != count:  # an eviction dropped a page that the restore would share
            pages += self._alloc_pages(self.core.restore_pages(host_entry) - count)
        entry = self.core.restore(req.rid, host_entry, pages, slot)
        self._run_host_ops()
        for _ in range(node.locks):
            self.core.lock(entry.key)
        self.core.host.pinned.discard(node.key)
        node.host = False

    def prefix_device_indices(self, req: Any) -> torch.Tensor:
        node, length = req.last_node, int(req.prefix_len)
        if node is None or node.key is None:
            raise RuntimeError(f"{_LOG} Request {req.rid} has a prefix of {length} tokens but no cache entry.")
        if node.host:
            self._restore(req, node)
        entry = self.core.get(node.key)
        if entry is None or entry.num_tokens != length:
            raise RuntimeError(f"{_LOG} Request {req.rid} reuses {length} tokens, but its entry holds "
                               f"{entry.num_tokens if entry else 'no'} tokens. The KDA state is exact only at the "
                               "entry's end. Do not shorten a kimi-prompt hit.")
        hit = self.core.take_hit(req.rid, entry)
        ps = self.page_size
        rows = [torch.arange(p * ps, (p + 1) * ps, dtype=torch.int64) for p in hit.shared_pages]
        protected = len(hit.shared_pages) * ps
        if hit.tail_page is not None and self.core.tail_extendable(entry):
            # no one else writes or reads the page past the hit: the request writes its rows into it in place, and the
            # page stays the entry's (the request frees only the pages after it)
            self.core.claim(req.rid, hit.tail_page)
            rows.append(torch.arange(hit.tail_page * ps, hit.tail_page * ps + length % ps, dtype=torch.int64))
            protected += ps
            self.tail_in_place += 1
        elif hit.tail_page is not None:
            private = self._alloc_pages(1)[0]
            self.page_copies.copy(hit.tail_page, private)
            rows.append(torch.arange(private * ps, private * ps + length % ps, dtype=torch.int64))
            self.tail_copies += 1
        req.kv.protect_prefix(protected)
        if entry.slot is not None and not req.kv.holds_mamba:
            slot = self._take_slot()
            if slot is None:
                raise RuntimeError(f"{_LOG} No state slot is free for request {req.rid}. Admit fewer requests at "
                                   "once, or give the prefill more state slots (--max-mamba-cache-size).")
            req.kv.mamba_pool_idx = self._slot_tensor(slot)[0]
            # upstream's deferred copy-on-write: the model runner copies the entry's state into the slot before the
            # request's first forward
            req.kv.mamba_cow_src_index = self._slot_tensor(entry.slot)
            req.kv.mamba_needs_clear = False
        out = torch.cat(rows) if rows else torch.empty(0, dtype=torch.int64)
        return out.to(self.token_to_kv_pool_allocator.device).contiguous()

    # -- the end of a request -----------------------------------------------------------------------------------------

    def _row_pages(self, req: Any, num_tokens: int) -> List[int]:
        ps = self.page_size
        row = self.req_to_token_pool.req_to_token[req.kv.req_pool_idx]
        return [int(row[i * ps]) // ps for i in range(-(-int(num_tokens) // ps))]

    @staticmethod
    def _req_key(req: Any):
        tokens = list(req.origin_input_ids) + list(getattr(req, "output_ids", ()) or ())
        salt = request_salt(getattr(req, "lora_id", None) or 0, getattr(req, "cache_salt", None),
                            getattr(req, "extra_key", None))
        return tokens, salt

    def _keep_chunk_end(self, req: Any, up_to: int) -> None:
        """Keep a chunk-end entry at ``up_to`` (the end of the forward that ran): the row's pages and the request's
        tracking slot, which holds the state at ``up_to`` (``track_entry``). The request takes a fresh tracking slot in
        its place. The entry is locked until the request releases its row (the request still writes the rows after
        ``up_to`` into the last page), and the request's protected prefix covers its pages."""
        kv = req.kv
        n = len(req.origin_input_ids)
        if not 0 < up_to < n or kv.req_pool_idx is None or kv.mamba_pool_idx is None:
            return
        if getattr(kv, "mamba_last_track_seqlen", None) != up_to or getattr(kv, "mamba_last_track_idx", None) is None:
            return  # the forward that ended at up_to wrote no tracking slot
        fresh = self._take_slot()
        if fresh is None:
            return  # no slot for the request's next tracking: keep no entry here
        self._run_host_ops()
        donated = int(self.storage.donate_tracking_slot(self.state_sub_pool, req, self._slot_tensor(fresh))
                      .reshape(-1)[0])
        kv.mamba_last_track_seqlen = None
        pages = self._row_pages(req, up_to)
        tokens, salt = self._req_key(req)
        self._adopting = donated
        entry = self.core.keep(req.rid, tokens, salt, up_to, pages, slot_owner=req.rid, kind="chunk_end")
        if self._adopting is not None:  # an entry of these tokens exists: the donated slot is no entry's
            self._adopting = None
            self._give_slot(donated)
        if entry is None:
            return
        self.core.lock(entry.key)
        self._chunk_locks.setdefault(req.rid, []).append(entry.key)
        if up_to % self.page_size:
            # the request writes its next rows into the entry's last page: a hit on the entry copies that page
            self.core.claim(req.rid, pages[-1])
        kv.protect_prefix(max(int(kv.cache_protected_len), len(pages) * self.page_size))

    def checkpoint(self, req: Any, *, up_to: int, **kwargs) -> None:
        if not req.finished():
            if self.chunk_ends:
                self._keep_chunk_end(req, int(up_to))
            return  # a running request keeps nothing else (END entries at the end)
        num_tokens = int(up_to)
        if num_tokens <= 0 or req.kv.req_pool_idx is None:
            return
        pages = self._row_pages(req, num_tokens)
        tokens, salt = self._req_key(req)
        working = req.kv.mamba_pool_idx
        self._adopting = None if working is None else int(working)
        entry = self.core.keep(req.rid, tokens, salt, num_tokens, pages,
                               slot_owner=req.rid if working is not None else None)
        adopted = working is not None and self._adopting is None and entry is not None
        self._adopting = None
        if entry is not None:
            req.kv.protect_prefix(num_tokens)
        req.kimi_prompt_adopted = adopted

    def release_insert_len(self, req: Any) -> int:
        """The tokens that the release of ``req`` keeps as its END entry: every token that the prefill computed (its
        prompt; with the first token on the decode, the prompt that the bootstrap cut to N - 1). An in-flight hold of a
        next turn reads it."""
        return len(req.origin_input_ids)

    def on_release(self, req: Any, *, checkpointed: bool) -> None:
        for key in self._chunk_locks.pop(req.rid, ()):
            self.core.unlock(key)
        self.core.forget(req.rid)
        adopted = bool(getattr(req, "kimi_prompt_adopted", False))
        req.kimi_prompt_adopted = False
        self._release_request_state(req, keep_working=adopted)

    # -- eviction and sizes -------------------------------------------------------------------------------------------

    def evict(self, params: EvictParams) -> EvictResult:
        need = -(-int(params.num_tokens) // self.page_size)
        freed = self.core.evict(need) if need > 0 else 0
        slots = 0
        for _ in range(int(getattr(params, "mamba_num", 0) or 0)):
            if not self.free_state_slot():
                break
            slots += 1
        self._run_host_ops()  # before the allocator gives the pages to anyone
        result = EvictResult(num_tokens_evicted=freed * self.page_size)
        if hasattr(result, "mamba_num_evicted"):
            result.mamba_num_evicted = slots
        return result

    def free_state_slot(self) -> bool:
        """Free a state slot (a hit, a host restore and a chunk-end entry need one): drop the least recent unlocked
        entry that holds a slot, else a running request's older chunk-end entry (``_older_chunk_end``)."""
        done = self.core.free_state_slot() or self._reclaim_chunk_end_slot()
        self._run_host_ops()
        return done

    def evictable_size(self) -> int:
        return self.core.evictable_pages() * self.page_size

    def protected_size(self) -> int:
        return (self.core.held_pages() - self.core.evictable_pages()) * self.page_size

    def full_evictable_size(self) -> int:
        return self.evictable_size()

    def full_protected_size(self) -> int:
        return self.protected_size()

    def mamba_evictable_size(self) -> int:
        """The state slots that unlocked entries hold."""
        return sum(1 for e in self.core.entries() if e.slot_owner is not None and not e.locks)

    def mamba_protected_size(self) -> int:
        """The state slots that locked entries hold."""
        return sum(1 for e in self.core.entries() if e.slot_owner is not None and e.locks)

    def total_size(self) -> int:
        return self.core.held_pages() * self.page_size

    def all_values_flatten(self) -> torch.Tensor:
        """The KV token ids of every page that an entry holds."""
        ps = self.page_size
        pages = sorted({p for e in self.core.entries() for p in e.pages})
        return torch.tensor([p * ps + i for p in pages for i in range(ps)], dtype=torch.int64)

    def all_mamba_values_flatten(self) -> torch.Tensor:
        """The state slots that the entries hold."""
        return torch.tensor(sorted(self._owned.values()), dtype=torch.int64)

    def sanity_check(self) -> None:
        held = sorted(e.slot for e in self.core.entries() if e.slot is not None)
        if held != sorted(self._owned.values()):
            raise RuntimeError(f"{_LOG} The entries hold state slots {held}, the backend records "
                               f"{sorted(self._owned.values())}.")

    def pretty_print(self) -> None:
        host = self.core.host
        print(f"{_LOG} entries {self.core.num_entries}, held pages {self.core.held_pages()} (evictable "
              f"{self.core.evictable_pages()}), state slots {self.core.held_slots()}, hits {self.core.num_hits}, "
              f"misses {self.core.num_misses}, evictions {self.core.num_evictions}"
              + (f", host entries {host.num_entries}, free host page slots {host.free_slots}, free host state slots "
                 f"{host.free_state_slots}" if host else ""))


def host_slots_of(environ=os.environ) -> Tuple[int, int]:
    """``(page slots, state slots)`` of the host tier from ``SGLANG_NEURON_KIMI_PROMPT_HOST_SLOTS`` and
    ``SGLANG_NEURON_KIMI_PROMPT_HOST_STATE_SLOTS``. A value that is not a non-negative integer is refused by name.

    Example:
        >>> host_slots_of({}), host_slots_of({HOST_SLOTS_ENV: "100"}), host_slots_of({HOST_SLOTS_ENV: "100",
        ...                                                                           HOST_STATE_SLOTS_ENV: "7"})
        ((0, 0), (100, 5), (100, 7))
    """
    from sglang_neuron import env_registry

    pages = env_registry.non_negative_int(HOST_SLOTS_ENV, environ) if environ.get(HOST_SLOTS_ENV) else 0
    if not pages:
        return 0, 0
    if environ.get(HOST_STATE_SLOTS_ENV):
        return pages, env_registry.non_negative_int(HOST_STATE_SLOTS_ENV, environ)
    return pages, max(1, pages // 20)


def build(ctx: Any) -> KimiPromptBackend:
    """The backend factory (``register_radix_cache_backend``): the Kimi prompt cache over SGLang's pools of
    ``ctx.params``. Refuses a disabled radix cache and a request table without the Neuron state pool.

    Example:
        >>> build  # doctest: +ELLIPSIS
        <function build at ...>
    """
    if ctx.disable_radix_cache:
        raise ValueError(f"--radix-cache-backend {BACKEND} with the radix cache disabled: the backend is a prefix "
                         "cache, and the launch asks for none. Remove --disable-radix-cache.")
    host_slots, host_state_slots = host_slots_of()
    keep_previous = os.environ.get(KEEP_PREVIOUS_ENV, "1") != "0"
    chunk_ends = os.environ.get(CHUNK_ENDS_ENV, "1") != "0"
    page = int(ctx.params.token_to_kv_pool_allocator.page_size)
    logger.info("%s the prefix cache keeps END entries over %d-token pages (keep_previous %s, chunk ends %s, host "
                "page slots %d, host state slots %d)", _LOG, page, keep_previous, chunk_ends, host_slots,
                host_state_slots)
    return KimiPromptBackend(ctx.params, keep_previous=keep_previous, chunk_ends=chunk_ends, host_slots=host_slots,
                             host_state_slots=host_state_slots)


def register_backend() -> None:
    """Register ``build`` as backend ``kimi-prompt`` (once per process: SGLang refuses a second registration)."""
    from sglang.srt.mem_cache.registry import get_radix_cache_factory, register_radix_cache_backend

    if get_radix_cache_factory(BACKEND) is None:
        register_radix_cache_backend(BACKEND, build)


# -- the tracking rule of the chunk ends ------------------------------------------------------------------------------

# (module, qualified name): md5 of the upstream source the backend relies on. The tracking rule replaces the first
# (it never calls it); the deferred copy-on-write, the state release, the donation of a tracking slot and the release
# and checkpoint of a request's KV are the paths the backend's state slots go through.
_SOURCES_MD5 = {
    ("sglang.srt.managers.schedule_batch", "ScheduleBatch._mamba_radix_cache_v2_req_prepare_for_extend"):
        "e3ed52bc38f326783b3826960c46bda5",
    ("sglang.srt.managers.schedule_batch", "ScheduleBatch._collect_deferred_mamba_cow_and_clear"):
        "0caa8118127ff2949d0d57cc806d9f97",
    ("sglang.srt.mem_cache.kv.state_storage", "StateStorage.release"): "b4c4ec3e9ff45fb5ffe356a2ba00b3b7",
    ("sglang.srt.mem_cache.memory_pool", "HybridReqToTokenPool.donate_mamba_ping_pong_slot"):
        "2543274c7031788f9e9704d4b26fa2c4",
    ("sglang.srt.mem_cache.common", "release_kv_cache"): "288de87ca4a5923c41b5dd0557935694",
    ("sglang.srt.mem_cache.common", "checkpoint_kv_cache"): "5449f01fe3e2e37a7d63c29de1b48623",
}


def track_entry(original_fn, self, req):
    """AROUND ``ScheduleBatch._mamba_radix_cache_v2_req_prepare_for_extend`` on a launch with this backend: the
    tracking rule of the chunk ends. A forward that does not end the prompt writes its end state into the request's
    next tracking slot as well (the stage graph's ``state_track_slot``), and that slot becomes the last tracked one at
    the forward's end. A forward that ends the prompt tracks nothing: its END entry adopts the working slot. Upstream's
    rule asks for checkpoints on its page grid, inside a forward too, which the stage graphs cannot write."""
    from sglang.srt.managers.schedule_batch import _MambaRadixCacheV2TrackEntry

    kv = req.kv
    position = kv.mamba_next_track_idx
    end = int(req.extend_end)
    chunk_ends = os.environ.get(CHUNK_ENDS_ENV, "1") != "0"
    if not chunk_ends or end >= len(req.origin_input_ids) or position is None:
        return _MambaRadixCacheV2TrackEntry(track_mask=False, track_position=position or 0, track_seqlen=-1)
    kv.mamba_last_track_idx = position
    kv.mamba_next_track_idx = self.req_to_token_pool.get_mamba_ping_pong_other_idx(position)
    kv.mamba_last_track_seqlen = end
    return _MambaRadixCacheV2TrackEntry(track_mask=True, track_position=position, track_seqlen=end)


def register_hooks(registry: Any, around: Any) -> None:
    """On a launch that selects the backend (sglang_patches/kimi_prompt): check the pins, then the tracking rule of the
    chunk ends."""
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_SOURCES_MD5, __name__)
    registry.register("sglang.srt.managers.schedule_batch.ScheduleBatch._mamba_radix_cache_v2_req_prepare_for_extend",
                      track_entry, around)
