# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""SGLang's paged KV allocator with faster host paths.

Without Triton, SGLang's paged allocator runs its torch fallbacks (``alloc_extend_naive``, ``alloc_decode_naive``),
which build many small tensors on the scheduler's critical path. This subclass keeps the allocator's behaviour (page 0
reserved, the ``last_loc`` partial-page contract, free groups, need_sort's staged pages) and does the work on the host
instead: ``alloc_extend`` goes page by page from the host lengths, ``alloc_decode`` counts a one-row batch's new page
without a tensor reduction, and frees use a bitmap rather than torch.unique's sort (page_set.py). The bookkeeping
tensors are on the CPU, SGLang's device on this platform.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator, alloc_decode_naive
from sglang.srt.utils import get_num_new_pages

from sglang_neuron.mem_cache.page_set import PageSetFreeMixin


def alloc_extend_pages(prefix_lens_cpu: torch.Tensor, seq_lens_cpu: torch.Tensor, last_loc: torch.Tensor,
                       free_pages: torch.Tensor, out_indices: torch.Tensor, page_size: int) -> None:
    """Write the KV slots of an extend of every request into ``out_indices`` (the requests' new tokens back to back), by
    the paged allocator's contract: a request's first new tokens fill the rest of its prefix's last page (from
    ``last_loc + 1``), then each further page is the next page of ``free_pages``, in request order. The work is per
    page, from the lengths' host copies: a request's slots are its partial page's run and one (pages x page_size)
    block of its new pages. SGLang's alloc_extend_naive builds and indexes some twenty tensors as long as the extend,
    on the scheduler's critical path."""
    slot = torch.arange(page_size, dtype=out_indices.dtype, device=out_indices.device)
    pos = page = 0
    for i, (prefix, seq) in enumerate(zip(prefix_lens_cpu.tolist(), seq_lens_cpu.tolist())):
        n = seq - prefix
        if n <= 0:
            continue
        # the tokens that complete the prefix's last page (none when the prefix ends on a page boundary)
        first = min(seq, -(-prefix // page_size) * page_size) - prefix
        if first > 0:
            out_indices[pos:pos + first] = last_loc[i] + 1 + slot[:first]
        rest = n - first
        if rest > 0:
            pages = -(-rest // page_size)
            block = (free_pages[page:page + pages, None] * page_size + slot).reshape(-1)
            out_indices[pos + first:pos + n] = block[:rest]
            page += pages
        pos += n


# SGLang's staging, which the staged batches' launch counts follow entry by entry: every staged batch comes from
# _release_page_ids, clear empties the stage, a merge returns all of it, and every free and allocation reaches the stage
# through these methods. free_segment (into free_page_ids) is what every row release takes, and under a radix tree also
# every eviction (StaticKVStorage.free_units) and duplicate insert (StaticKVStorage.free_token_rows)
_SOURCES_MD5 = {("sglang.srt.mem_cache.allocator.paged", f"PagedTokenToKVPoolAllocator.{name}"): md5 for name, md5 in (
    ("_release_page_ids", "4357cb7cdbc0694031a80e13e6c83d30"),
    ("clear", "77125e295b13c089351fc865ff3cc48b"),
    ("merge_and_sort_free", "248077b82757a563810b27b28125195b"),
    ("get_all_free_pages", "026fe2e58e3882a28ddf805b88651b32"),
    ("available_size", "7b25ae8028912ba7b5d287546249ac49"),
    ("free", "19dfe9c55416704ee3b52f25a83d1777"),
    ("free_segment", "b547543cc840350c4ff9a96d6c4b1200"),
    ("free_page_ids", "8ac1a1fadb565b9ff3e66b8bb56832e7"),
    ("free_group_end", "0fc62bf59dd401a68d68020ad4cf1686"),
    ("alloc", "9b7361963b09cca714d8a93840ff688f"),
)}


def _check_dcp_widening(allocator) -> None:
    """Under DCP of size D, SGLang builds this allocator over D x the pool's tokens in pages of D x its block
    (kv_cache_configurator), so each page is a virtual page with one block on every DCP rank. Refuse any other size:
    it would admit KV the pool does not hold or address pages no rank keeps. The pool is the full-attention one (a
    hybrid linear-attention pool's leaf)."""
    from sglang_neuron.mem_cache.memory_pool import full_attention_pool

    pool = allocator.get_kvcache()
    pool = None if pool is None else full_attention_pool(pool)
    if pool is None or not hasattr(pool, "dcp"):
        return
    size = 1 if pool.dcp is None else pool.dcp.size
    want = (pool.size * size, pool.page_size * size)
    if (allocator.size, allocator.page_size) != want:
        raise RuntimeError(f"SGLang built the KV allocator over {allocator.size} tokens in pages of "
                           f"{allocator.page_size}, not over {want[0]} in pages of {want[1]}: the pool's {pool.size} "
                           f"tokens and {pool.page_size}-token blocks times its DCP size ({size})")


# The zero-page contract (mem_cache/mla_pool.py): every page the allocator hands out reads zero. A pool under it
# declares when its pages are zeroed (``zero_policy``) and offers one fill, ``zero_pages(page_ids)``; this allocator
# runs that fill on one path, at the point the policy names:
#   on_free  - when a page returns to the free list (_release_page_ids; every free path reaches it). The default: the
#              MLA / DSA pools. Decode-safe: a decode's freed pages are staged behind the launches issued up to the
#              free (merge_and_sort_free), so the fill runs before a NIXL receive can target the page again.
#   on_alloc - when a page is handed out (alloc, alloc_extend, alloc_decode; every allocation takes its pages from
#              free_pages). For a pool that zeroes pages for kernels reading whole blocks past a sequence's end only on
#              the pages it hands out (the FP8 MLA row pool, KVLayout.zero_fresh_pages). Refused on a PD decode: a
#              decode's pages receive their KV over NIXL right after the allocation, which the device does not order
#              behind the fill.
# A device fill rewrites more of a buffer than the pages it names (measured with the HiCache host tier: bytes a
# concurrent HiCache load DMA wrote into other pages went back), so whichever fill runs first waits on the host for the
# HiCache loads in flight (host_pool.wait_for_loads; free with no HiCache) unless the pool declares its fill
# load-safe (``zero_fill_load_safe``).
ZERO_ON_FREE = "on_free"
ZERO_ON_ALLOC = "on_alloc"
ZERO_POLICIES = (ZERO_ON_FREE, ZERO_ON_ALLOC)


def zeroing_pool(kvcache):
    """The pool under the zero-page contract that ``kvcache`` is or holds (a hybrid pool's full-attention leaf,
    ``full_kv_pool``), or None: the first with a callable ``zero_pages`` and a ``zero_policy`` other than None."""
    for pool in (kvcache, getattr(kvcache, "full_kv_pool", None)):
        if pool is None or not callable(getattr(pool, "zero_pages", None)):
            continue
        policy = getattr(pool, "zero_policy", ZERO_ON_FREE)
        if policy is None:
            continue
        if policy not in ZERO_POLICIES:
            raise ValueError(f"{type(pool).__name__}.zero_policy {policy!r} is not one of {ZERO_POLICIES}")
        return pool
    return None


def zero_policy(kvcache) -> Optional[str]:
    """When ``kvcache``'s pages are zeroed (ZERO_ON_FREE / ZERO_ON_ALLOC), or None outside the contract."""
    pool = zeroing_pool(kvcache)
    return None if pool is None else getattr(pool, "zero_policy", ZERO_ON_FREE)


def page_zeroer(kvcache, policy: str = ZERO_ON_FREE):
    """The fill of a pool under the zero-page contract whose policy is ``policy`` (the pool's ``zero_pages``, wrapped
    to wait for HiCache loads in flight unless the pool declares its fill load-safe), or None."""
    pool = zeroing_pool(kvcache)
    if pool is None or getattr(pool, "zero_policy", ZERO_ON_FREE) != policy:
        return None
    zero = pool.zero_pages
    if getattr(pool, "zero_fill_load_safe", False):
        return zero

    def fill(pages):
        from sglang_neuron.mem_cache import host_pool

        host_pool.wait_for_loads()
        zero(pages)

    return fill


def _refuse_decode_on_alloc(allocator) -> None:
    from sglang_neuron import serving_policy

    if (zero_policy(getattr(allocator, "_kvcache", None)) == ZERO_ON_ALLOC
            and serving_policy.current().role == "decode"):
        raise ValueError("the KV pool zeroes its pages as they are handed out (zero_policy on_alloc) on a PD decode, "
                         "whose pages receive their KV over NIXL, which the device does not order behind that fill; "
                         "serve the pool with zero_policy on_free on a decode")


def _refuse_unserved_fresh_pages(allocator) -> None:
    """A model that declares fresh pages zero (model_executor.kv_layout.KVLayout.zero_fresh_pages) needs a pool whose
    pages are zeroed as they are handed out (zero_policy on_alloc): the Neuron MLA pool declares it then."""
    from sglang_neuron.mem_cache.memory_pool import full_attention_pool
    from sglang_neuron.model_executor import kv_layout

    kvcache = getattr(allocator, "_kvcache", None)
    if kv_layout.current().zero_fresh_pages and zero_policy(kvcache) != ZERO_ON_ALLOC:
        pool = None if kvcache is None else full_attention_pool(kvcache)
        raise ValueError(f"the model declares zero_fresh_pages and its KV pool ({type(pool).__name__}) does not zero "
                         "its pages as they are handed out: serve it on the MLA pool (NeuronMLATokenToKVPool)")


class NeuronPagedTokenToKVPoolAllocator(PageSetFreeMixin, PagedTokenToKVPoolAllocator):
    @property
    def zeroes_freed_pages(self) -> bool:
        """Whether every page this allocator hands out again reads zero: its pool (or a hybrid pool's full-attention
        leaf) is under the zero-page contract, on either policy. A megakernel whose pools are zero_on_free reads whole
        pages and is refused without it (megakernel/built.py)."""
        return zero_policy(getattr(self, "_kvcache", None)) is not None

    def _zero_handed_out(self, pages: torch.Tensor) -> None:
        """The on_alloc fill of the pages this allocation takes from free_pages (nothing under another policy)."""
        if not pages.numel():
            return
        zero = page_zeroer(getattr(self, "_kvcache", None), ZERO_ON_ALLOC)
        if zero is not None:
            zero(pages)

    def __init__(self, *args, **kwargs):
        from sglang_neuron import serving_policy
        from sglang_neuron.sglang_patches import source_guard

        source_guard.check_targets(_SOURCES_MD5, __name__)
        super().__init__(*args, **kwargs)
        _check_dcp_widening(self)
        # a disaggregated decode's staged pages wait for the launches that may still write them (merge_and_sort_free);
        # a prefill's go back as upstream's do, for the reason merge_and_sort_free gives
        self.launch_guarded = self.need_sort and serving_policy.current().role == "decode"
        _refuse_decode_on_alloc(self)
        _refuse_unserved_fresh_pages(self)

    def clear(self):
        # this thread's launch count (device_io.launches) when each staged batch was freed, in staged_pages' order
        self._staged_launches = []
        super().clear()

    def _release_page_ids(self, *page_ids: torch.Tensor):
        # every free path reaches here, so a pool zeroing on free zeroes a page exactly when it returns to the free list
        zero = page_zeroer(getattr(self, "_kvcache", None), ZERO_ON_FREE)
        if zero is not None:
            for pages in page_ids:
                zero(pages)
        # getattr: a bare instance (a unit test's object.__new__) has no launch bookkeeping
        launch_guarded = getattr(self, "launch_guarded", False)
        if launch_guarded:
            import threading

            from sglang_neuron.backend import device_io

            # the launch counts are the launching thread's: another thread's would read 0 and stamp the batch safe
            launcher = device_io.launching_thread()
            if launcher is not None and launcher != threading.get_ident():
                raise RuntimeError(f"KV pages freed on a thread that launches no graph (thread {threading.get_ident()}; "
                                   f"the graphs launch on thread {launcher}): {__name__} stamps a freed batch with the "
                                   "launching thread's launch count, so it would hand the pages out while launches "
                                   "may still write them")
        super()._release_page_ids(*page_ids)
        if launch_guarded:
            self._staged_launches.extend([device_io.launches()] * len(page_ids))

    def merge_and_sort_free(self, need_pages: Optional[int] = None):
        """Return the staged pages to the free list.

        On a disaggregated decode a page goes back only once every launch that may still write it has finished: a
        finished request's row stays in the steps already launched (one, or the result lag's in-flight bound), and the
        page's next owner may get its KV over NIXL, which the device doesn't order behind them. Each staged batch
        carries this thread's launch count at its free and is safe once device_io.retired_launches, a lower bound,
        reaches it. The safe batches go back without a wait; the rest wait for this thread's launches only when the
        safe ones leave an allocation of ``need_pages`` short, or when the caller names no need (SGLang's own merges,
        which take every staged page). The decode radix cache's evictions and duplicate inserts free through
        free_segment too, so their pages are staged the same way.

        A prefill's staged pages go back as upstream's do (launch_guarded). A prefill frees a request's pages once its
        KV transfer has ended, so its own forward has finished before the transfer starts; the launches still in flight
        are other requests' forwards, which never touch those pages; and the next owner's KV comes from its own
        forward, a later launch on the same stream.

        The counts are the launching thread's: _release_page_ids refuses a free on any other thread, and the merges and
        allocations rely on SGLang running them on the scheduler thread (a merge on another thread would see no
        launches and wait for none)."""
        if not self.launch_guarded:
            return super().merge_and_sort_free()
        if not self.staged_pages:
            return
        if len(self._staged_launches) != len(self.staged_pages):
            # SGLang staged a batch past _release_page_ids, so the counts no longer say which batch is safe
            raise RuntimeError(f"{len(self.staged_pages)} staged page batches but {len(self._staged_launches)} launch "
                               f"counts: SGLang staged pages another way than {__name__} was written against; "
                               "re-validate it")
        from sglang_neuron.backend import device_io

        retired = device_io.retired_launches()
        safe = [freed <= retired for freed in self._staged_launches]
        if not all(safe):
            safe_pages = sum(pages.numel() for pages, ok in zip(self.staged_pages, safe) if ok)
            if need_pages is not None and len(self.free_pages) + safe_pages >= need_pages:
                self._merge_staged(safe)
                return
            device_io.wait_launches()
        super().merge_and_sort_free()
        self._staged_launches = []

    def _merge_staged(self, take: list) -> None:
        """Move the staged batches ``take`` marks to the free list, sorted as SGLang's merge sorts it."""
        taken = [pages for pages, moved in zip(self.staged_pages, take) if moved]
        self.staged_pages = [pages for pages, moved in zip(self.staged_pages, take) if not moved]
        self._staged_launches = [freed for freed, moved in zip(self._staged_launches, take) if not moved]
        self.num_staged_pages = sum(pages.numel() for pages in self.staged_pages)
        self.free_pages, _ = torch.sort(torch.cat((self.free_pages, *taken)))

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        num_new_pages: int = None,
    ):
        if self.debug_mode:
            assert torch.all((last_loc + 1) % self.page_size == prefix_lens % self.page_size)

        if num_new_pages is None:
            num_new_pages = get_num_new_pages(
                seq_lens=seq_lens_cpu, page_size=self.page_size, prefix_lens=prefix_lens_cpu
            )
        num_new_pages_item = int(num_new_pages)

        if self.need_sort and num_new_pages_item > len(self.free_pages):
            self.merge_and_sort_free(num_new_pages_item)
        if num_new_pages_item > len(self.free_pages):
            return None

        out_indices = torch.empty((extend_num_tokens,), dtype=torch.int64, device=self.device)
        alloc_extend_pages(prefix_lens_cpu, seq_lens_cpu, last_loc, self.free_pages, out_indices, self.page_size)

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        self._zero_handed_out(self.free_pages[:num_new_pages_item])
        self.free_pages = self.free_pages[num_new_pages_item:]
        return out_indices

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ):
        """SGLang's alloc_decode_naive slots (a row that starts a page takes the next free page, every other row
        ``last_loc + 1``), with a one-row batch's new page counted on the host and the staged pages merged only when
        the free pages fall short."""
        if self.debug_mode:
            assert torch.all((last_loc + 2) % self.page_size == seq_lens % self.page_size)

        if seq_lens_cpu.numel() == 1 and seq_lens_cpu.dtype in (torch.int32, torch.int64):
            # one row: whether it starts a page is read on the host, with no tensor reduction (get_num_new_pages
            # also takes host lengths only)
            assert seq_lens_cpu.device == torch.device("cpu")
            num_new_pages = int((int(seq_lens_cpu.item()) - 1) % self.page_size == 0)
        else:
            num_new_pages = int(get_num_new_pages(seq_lens=seq_lens_cpu, page_size=self.page_size, decode=True))

        if self.need_sort and num_new_pages > len(self.free_pages):
            self.merge_and_sort_free(num_new_pages)
        if num_new_pages > len(self.free_pages):
            return None

        out_indices = torch.empty((len(seq_lens),), dtype=torch.int64, device=self.device)
        alloc_decode_naive(seq_lens, last_loc, self.free_pages, out_indices, self.page_size, num_new_pages)

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        self._zero_handed_out(self.free_pages[:num_new_pages])
        self.free_pages = self.free_pages[num_new_pages:]
        return out_indices

    def alloc(self, need_size: int):
        out = super().alloc(need_size)
        if out is not None:
            self._zero_handed_out(out[::self.page_size] // self.page_size)
        return out
