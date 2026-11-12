# SPDX-License-Identifier: Apache-2.0
"""Page sets for the paged KV allocators without a sort.

SGLang's paged allocator reduces a freed index to the pages it touches with ``torch.unique(index // page_size)``
(PagedTokenToKVPoolAllocator.free). That is a sort, O(n log n) on one host thread (the schedulers run torch with few
intra-op threads), paid by every caller: request finish, radix-tree eviction, disaggregation frees. With many requests
at a large page it can outlast a decode step.

Page ids are bounded by the pool, so the page set is a bitmap: mark[index >> log2(page_size)], then the marked ids
in ascending order. That is O(n + pages) and the same tensor torch.unique returns, and it is faster than the sort from
a few thousand slots up. The allocator's free depends on its index only through its page set, so the mixin below
hands SGLang's own method one slot per touched page (the page's first slot). SGLang's logic, free groups and
release_pages included, runs unchanged, and its torch.unique then sorts a few hundred page ids. Indices that are
small, not on the CPU, or outside the pool go to SGLang as they are."""
from __future__ import annotations

from typing import Optional

import torch

# below this many indices, or an eighth of the pool's pages (the bitmap's fixed cost), the sort is about as fast
MIN_INDICES = 4096


def page_heads(index: torch.Tensor, page_size: int, num_pages: int) -> Optional[torch.Tensor]:
    """The first slot of every page ``index`` touches, ascending: ``torch.unique(index // page_size) * page_size``,
    in O(n + num_pages). None when the sort is the better tool or the input is not a slot index of a
    ``num_pages``-page pool (pages 0..num_pages, page 0 SGLang's reserved one): small, not on the CPU, not an integer
    tensor, or out of range."""
    if (page_size <= 1 or index.device.type != "cpu"
            or index.dtype not in (torch.int32, torch.int64)
            or index.numel() < max(MIN_INDICES, num_pages // 8)):
        return None
    # integer division is most of the cost: page sizes are powers of two in practice, and a shift is exact for them
    shift = page_size.bit_length() - 1 if page_size & (page_size - 1) == 0 else None
    pages = index >> shift if shift is not None else torch.div(index, page_size, rounding_mode="floor")
    lo, hi = torch.aminmax(pages)
    lo, hi = int(lo), int(hi)
    if lo < 0 or hi > num_pages:
        return None
    mark = torch.zeros(hi + 1, dtype=torch.bool)
    mark[pages] = True
    heads = torch.nonzero(mark).view(-1).to(index.dtype)
    return heads << shift if shift is not None else heads * page_size


class PageSetFreeMixin:
    """For a PagedTokenToKVPoolAllocator subclass (list it first): free() through :func:`page_heads`."""

    def free(self, free_index: torch.Tensor):
        heads = page_heads(free_index, self.page_size, self.num_pages)
        super().free(free_index if heads is None else heads)
