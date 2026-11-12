# SPDX-License-Identifier: Apache-2.0
"""HiCache L2 host copies of a hybrid MLA + recurrent-state model's Neuron pools: the platform MLA pool's pages and the
recurrent-state pool's slots. Each moves through host_pool.TransferEngine (the runtime's batch DMA, one op per region
and run) and completes through host_pool.XferEvent, as the MHA host pool does (mem_cache/host_pool.py).

* MLA (``NeuronHostMLAPool``): one host region per MLA layer of the stage, ``[host blocks, block rows, row width]``
  in the pool's store dtype. The unit of a transfer is SGLang's allocator page. Under DCP that page is widened
  (``DcpLayout``: virtual page v is block v on every rank, each rank's stripes of it in its own block), so a transfer
  of whole allocator pages moves this rank's whole blocks: host block ``index // page`` <-> device block
  ``index // page``. A HiCache page holds whole allocator pages (the chunk-keyed tree's page is the chunk). Upstream's
  MLA host pool translates DCP indices for token-interleaved placement (slot % dcp == rank), not for these stripes,
  so it is not used for this pool. Tensors the model keeps at the same pages (the pool's ``page_companions``, which
  the model sets when it binds the pool, e.g. drafter rows) get regions of their own and move with the pages.
* State (``NeuronHostStatePool``): one host region per state component and layer, ``[host slots, *slot shape]``; a
  slot is one page (upstream's MambaPoolHost keeps page size 1 too). SGLang names device and host slots; slot 0 is the
  pool's padding slot and the trailing sink is no request's, so neither is ever moved.

Host regions are layer-first, pinned and page-locked, like the MHA host pool's (the runtime DMAs into them after the
call returns). A load is submitted whole at layer 0: the compiled stage forward reads every layer at once.
"""

from __future__ import annotations

import logging
from typing import List

import torch

from sglang.srt.mem_cache.pool_host.base import HostKVCache

from sglang_neuron.mem_cache import host_pool

logger = logging.getLogger(__name__)


class _NeuronHostPages(HostKVCache):
    """A host copy of a list of device tensors ``[pages, ...]`` (one host region each, page for page), moved by the
    engine. Subclasses set ``device_tensors``, ``device_pages`` (the device's page count), ``reserved`` (device pages
    below it are never moved) and ``unit`` (logical tokens of one device page; a HiCache page holds whole ones, e.g.
    the chunk-keyed tree's chunk of two allocator pages) before calling ``_start``."""

    # the transfer engine class: the runtime's batch DMA on the device
    engine_class = None

    def _start(self, *, dtype, device_capacity_tokens, page_size, host_to_device_ratio, host_size, start_layer,
               end_layer, dcp_size=1, dcp_rank=0, allocator_type="default", pool_label="kv"):
        # L2 only: the remote DRAM tier is validated on the MHA host pool alone (its check, before any buffer)
        host_pool.NeuronPagedHostPool.check_allocator(allocator_type, pool_label)
        self.page_bytes = [t[0].numel() * t.element_size() for t in self.device_tensors]
        for t in self.device_tensors:
            if not t.is_contiguous() or t.shape[0] != self.device_pages:
                raise ValueError("Neuron HiCache needs contiguous [pages, ...] device tensors of one page count")
        self.layer_num = self.target_layer_num = len(self.device_tensors)
        # bytes of one of this rank's token rows over every region: a device page holds unit / dcp_size of them
        super().__init__(dtype=dtype, device_capacity_tokens=device_capacity_tokens,
                         size_per_token=sum(self.page_bytes) // (self.unit // dcp_size), start_layer=start_layer,
                         end_layer=end_layer, host_to_device_ratio=host_to_device_ratio, host_size=host_size,
                         page_size=page_size, layout="layer_first", pin_memory=True, device="cpu",
                         allocator_type="default", dcp_size=dcp_size, dcp_rank=dcp_rank, pool_label=pool_label)
        engine_class = type(self).engine_class or host_pool.TransferEngine
        self.engine = engine_class(self.device_tensors, self.host_regions, self.page_bytes)
        self.engine.name, self.engine.verify = pool_label, host_pool._VERIFY
        host_pool._ENGINES.append(self.engine)
        logger.info("[neuron hicache] %s host pool: %d pages x %d regions (%.2f GiB)", pool_label, self.page_num,
                    len(self.device_tensors),
                    sum(r.numel() * r.element_size() for r in self.host_regions) / (1 << 30))

    def get_size_per_token(self):
        return self.size_per_token

    @property
    def host_units(self) -> int:
        """Device pages the host regions hold: ``unit`` logical tokens each, a host page holding whole units."""
        return self.page_num * (self.logical_page_size // self.unit)

    def init_kv_buffer(self):
        if self.logical_page_size % self.unit:
            raise ValueError(f"a HiCache page of {self.logical_page_size} tokens is not whole device pages of "
                             f"{self.unit} tokens")
        regions = []
        for t in self.device_tensors:
            region = torch.empty((self.host_units, *t.shape[1:]), dtype=t.dtype, pin_memory=self.engine_pins())
            if self.engine_pins():
                host_pool._mlock(region)
            regions.append(region)
        self.host_regions = regions
        return regions

    @classmethod
    def engine_pins(cls) -> bool:
        """Whether the host regions are pinned and page-locked: the runtime's DMA engine writes them after its call
        returns (an engine that copies on the host needs neither)."""
        return getattr(cls.engine_class or host_pool.TransferEngine, "pins_host", True)

    def _pages(self, indices: torch.Tensor, limit: int, low: int) -> List[int]:
        """Whole, aligned device pages (``unit`` logical tokens) of ``indices`` as page numbers in ``[low, limit)``."""
        page = self.unit
        idx = indices.to("cpu", torch.int64).reshape(-1)
        if idx.numel() % page:
            raise ValueError("HiCache transfers move whole pages")
        rows = idx.reshape(-1, page)
        starts = rows[:, 0]
        if torch.any(starts % page) or not torch.equal(rows, starts[:, None] + torch.arange(page)):
            raise ValueError("HiCache indices must be whole, aligned pages")
        pages = (starts // page).tolist()
        if pages and (min(pages) < low or max(pages) >= limit):
            raise ValueError(f"HiCache page out of range [{low}, {limit})")
        return pages

    def _submit(self, to_host: bool, host_indices, device_indices) -> None:
        host = self._pages(host_indices, self.host_units, 0)
        dev = self._pages(device_indices, self.device_pages, self.reserved)
        if len(host) != len(dev):
            raise ValueError("HiCache host and device page counts differ")
        if host:
            self.engine.submit(to_host, dev, host)

    def backup_from_device_all_layer(self, device_pool, host_indices, device_indices, io_backend):
        self._submit(True, host_indices, device_indices)

    def load_to_device_per_layer(self, device_pool, host_indices, device_indices, layer_id, io_backend, *,
                                 is_draft=False):
        """The whole load at layer 0: the compiled forward reads every layer at once."""
        if is_draft:
            raise ValueError("Neuron HiCache keeps no draft KV")
        if layer_id == 0:
            self._submit(False, host_indices, device_indices)

    def get_data_page(self, index, flat=True):
        raise NotImplementedError("Neuron HiCache is L2 only (no storage backend)")

    def get_dummy_flat_data_page(self):
        raise NotImplementedError("Neuron HiCache is L2 only (no storage backend)")

    def set_from_flat_data_page(self, index, data_page):
        raise NotImplementedError("Neuron HiCache is L2 only (no storage backend)")


class NeuronHostMLAPool(_NeuronHostPages):
    """The host copy of the platform MLA pool (mem_cache.memory_pool.NeuronMLATokenToKVPool, standalone or the leaf of
    SGLang's hybrid pool) of one stage rank. ``page_size`` is the allocator's page (widened by ``dcp_size`` under
    DCP); device page 0 is the reserved page."""

    def __init__(self, kv_pool, host_to_device_ratio, host_size, page_size, *, dcp_size=1, dcp_rank=0,
                 device_capacity_tokens=None, allocator_type="default", pool_label="kv"):
        self.device_tensors = [kv_pool.get_block_buffer(kv_pool.start_layer + i) for i in range(kv_pool.layer_num)]
        # tensors the model keeps at the same pages (``page_companions``, e.g. drafter rows) move with them
        self.device_tensors += list(getattr(kv_pool, "page_companions", ()))
        # one allocator page: one block on every rank
        self.device_pages, self.reserved, self.unit = kv_pool.num_blocks, 1, kv_pool.page_size * dcp_size
        self._start(dtype=kv_pool.store_dtype, page_size=page_size, host_to_device_ratio=host_to_device_ratio,
                    host_size=host_size, start_layer=kv_pool.start_layer,
                    end_layer=kv_pool.start_layer + kv_pool.layer_num - 1,
                    device_capacity_tokens=kv_pool.size if device_capacity_tokens is None else device_capacity_tokens,
                    dcp_size=dcp_size, dcp_rank=dcp_rank, allocator_type=allocator_type, pool_label=pool_label)


class NeuronHostStatePool(_NeuronHostPages):
    """The host copy of the recurrent-state pool (mem_cache.state_pool.NeuronStatePool) of one stage rank: every
    component of every layer, slot for slot. Device slots 1..size are a request's or a checkpoint's."""

    def __init__(self, state_pool, host_to_device_ratio, host_size, *, device_capacity=None, allocator_type="default",
                 pool_label="mamba"):
        stage = state_pool.spec_stage
        self.device_tensors = [state_pool.tensors[c.name][i] for c in stage.components
                               for i in range(state_pool.num_mamba_layers)]
        # slots 0..size; the trailing sink lies past them and is never moved
        self.device_pages, self.reserved, self.unit = state_pool.size + 1, 1, 1
        self.device_tensors = [t[:self.device_pages] for t in self.device_tensors]
        dtypes = {t.dtype for t in self.device_tensors}
        self._start(dtype=dtypes.pop() if len(dtypes) == 1 else torch.uint8, page_size=1,
                    host_to_device_ratio=host_to_device_ratio, host_size=host_size,
                    start_layer=min(state_pool.mamba_layer_ids), end_layer=max(state_pool.mamba_layer_ids),
                    device_capacity_tokens=state_pool.size if device_capacity is None else device_capacity,
                    allocator_type=allocator_type, pool_label=pool_label)


# ---- SGLang's hybrid host stack --------------------------------------------------------------------------------


def kv_host_pool(original, *, kv_pool, page_size, use_mla=None, override_kv_cache_dim=None, host_size=None,
                 mtp_draft_device_pools=(), pool_label="kv"):
    """hybrid_pool_assembler.build_kv_host_pool: the Neuron MLA host pool for the platform MLA pool, SGLang's host
    pool for any other. ``page_size`` is the HiCache page, widened under DCP like the allocator's."""
    from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler
    from sglang.srt.runtime_context import get_memory, get_parallel

    from sglang_neuron.mem_cache.memory_pool import NeuronMLATokenToKVPool

    if not isinstance(kv_pool, NeuronMLATokenToKVPool):
        return original(kv_pool=kv_pool, page_size=page_size, use_mla=use_mla,
                        override_kv_cache_dim=override_kv_cache_dim, host_size=host_size,
                        mtp_draft_device_pools=mtp_draft_device_pools, pool_label=pool_label)
    if mtp_draft_device_pools:
        raise ValueError("Neuron HiCache keeps no draft KV")
    memory, parallel = get_memory(), get_parallel()
    dcp = int(parallel.attn_dcp_size) if parallel.dcp_enabled else 1
    return NeuronHostMLAPool(kv_pool, memory.hicache_ratio, memory.hicache_size if host_size is None else host_size,
                             page_size, dcp_size=dcp, dcp_rank=int(parallel.attn_dcp_rank) if dcp > 1 else 0,
                             device_capacity_tokens=kv_pool.size if kv_pool.host_capacity_tokens is None
                             else kv_pool.host_capacity_tokens,
                             allocator_type=hybrid_pool_assembler._get_allocator_type(), pool_label=pool_label)


def state_host_pool(original, mamba_pool, pool_info, *, host_size, layout, mtp_draft_device_pools=()):
    """hybrid_pool_assembler.build_mamba_host_pool: the Neuron state host pool for the Neuron recurrent-state pool
    (whatever SGLang's global host layout: its regions are layer-first), SGLang's MambaPoolHost for any other."""
    from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler
    from sglang.srt.runtime_context import get_memory

    from sglang_neuron.mem_cache.state_pool import NeuronStatePool

    if not isinstance(mamba_pool, NeuronStatePool):
        return original(mamba_pool, pool_info, host_size=host_size, layout=layout,
                        mtp_draft_device_pools=mtp_draft_device_pools)
    if mtp_draft_device_pools:
        raise ValueError("Neuron HiCache keeps no draft state")
    capacity = mamba_pool.host_capacity_tokens
    return NeuronHostStatePool(mamba_pool, get_memory().hicache_ratio, host_size,
                               device_capacity=mamba_pool.size if capacity is None else capacity,
                               allocator_type=hybrid_pool_assembler._get_allocator_type())


_ASSEMBLER = "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler"
# Wrapped: the two builders; relied on: the hybrid mamba stack calling both by module name with these arguments and
# taking the KV pool as the hybrid pool's full-attention leaf, and the builders' _get_allocator_type naming the storage
# backend's allocator (as mla_host_pool reads it). Pinned by md5 against SGLang 43881a7.
_SOURCES_MD5 = {
    (_ASSEMBLER, "build_kv_host_pool"): "3733320973aea2035d48663febbb4e65",
    (_ASSEMBLER, "build_mamba_host_pool"): "397a80fba7c7b4e0842816c2941f7f9a",
    (_ASSEMBLER, "build_hybrid_mamba_stack"): "f040e1e472580df544b24cf5552daa50",
}


def register_hooks(registry, hook_type) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_SOURCES_MD5, __name__)
    registry.register(f"{_ASSEMBLER}.build_kv_host_pool", kv_host_pool, hook_type.AROUND)
    registry.register(f"{_ASSEMBLER}.build_mamba_host_pool", state_host_pool, hook_type.AROUND)
