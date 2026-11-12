# SPDX-License-Identifier: Apache-2.0
"""The KV cache layout a model declares while it is built, which the Neuron KV pools and the paged allocator then
follow (SGLang builds them after the model). One declaration per process: a scheduler builds one model, then its pool."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class KVLayout:
    # token context parallelism: every rank holds all KV heads for its own block-cyclic token slices, so an SGLang
    # page of cp_size * (page / cp_size) tokens is one (page / cp_size)-token block on each rank (mem_cache/cp_pages)
    cp_size: int = 1
    # this rank's position in the KV context-parallel group
    cp_rank: int = 0
    # the KV heads each rank holds when they differ from SGLang's split (KV heads / attention TP): under token context
    # parallelism every rank holds every KV head; None keeps SGLang's
    kv_heads_per_rank: Optional[int] = None
    # the pool's K pages are token-major ([block, head_dim], like its V pages), not the dim-major transpose: the
    # Qwen3.5 kv_cp decode's pool and the kv_cp=1 prefill serving its scatter (sglang_patches/kv_cp_scatter). A PD
    # hand-off needs one K layout on both roles (sglang_patches/pd_kv_cp_layout carries it in the layout frame).
    k_token_major: bool = False
    # layers 2j and 2j + 1 share one K tensor and one V tensor of twice the blocks, layer 2j in the first half
    paired_layers: bool = False
    # every page the paged allocator hands out reads zero on this rank, for kernels that read whole blocks past a
    # sequence's end; served by the MLA pool (NeuronMLATokenToKVPool: zero_policy on_alloc, mem_cache.allocator)
    zero_fresh_pages: bool = False
    # a model's own paged allocator, returned by this zero-argument callable: a NeuronPagedTokenToKVPoolAllocator
    # subclass that decides which pages a request takes (platform.get_paged_allocator_cls); None: the platform's
    paged_allocator_class: Optional[Callable[[], type]] = None

    def physical_block_size(self, page_size: int) -> int:
        """This rank's block of an SGLang page under token context parallelism."""
        if page_size % self.cp_size:
            raise ValueError(f"page size {page_size} is not a multiple of the KV context-parallel degree "
                             f"{self.cp_size}")
        return page_size // self.cp_size


_LAYOUT = KVLayout()


def declare(layout: KVLayout) -> None:
    global _LAYOUT
    _LAYOUT = layout


def current() -> KVLayout:
    return _LAYOUT
