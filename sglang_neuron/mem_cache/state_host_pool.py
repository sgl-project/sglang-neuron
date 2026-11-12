# SPDX-License-Identifier: Apache-2.0
"""Host copies of a recurrent-state pool's slots: HiCache L2 of state (``NeuronStateHostPool``) and the slot-granular
host round trip upstream's decode retraction backup calls (``NeuronStatePool.get_cpu_copy`` / ``load_cpu_copy``).

SGLang drives state offload the way it drives KV offload: its hybrid stack builder (``build_hybrid_mamba_stack``) asks
the state pool for its buffers (``get_device_pool_infos``: upstream's ``MambaStateBufferInfo``) and builds the host
pool with ``build_mamba_host_pool``; the radix tree, the host slot allocator and ``HybridCacheController`` decide what
moves. Upstream's ``MambaPoolHost`` moves slots with CUDA JIT kernels (or an NPU operator), which Neuron has none of,
so for a Neuron state pool the builder returns this module's host pool instead (``build_mamba_host_pool``, wrapped).

What moves is a whole slot: every component (the recurrent state and each conv window) of every state layer, in the
pool's own physical layout and dtype, so a state reloaded from the host is the bytes the kernel left, bit for bit (no
layout change, no cast). A working-bank pool moves its canonical slots only; the working rows are scratch a
forward fills from them (``load_rows``).

Layout. The device tensors are ``[layers, slots + 2, *shape]`` per component (padding slot 0, slots, sink); one slot
of one layer is contiguous. The host keeps the same order, ``[layers, host slots, *shape]`` per component, so slots
consecutive on both sides are one contiguous span per (component, layer), and a transfer of n slots is at most
(components x layers x runs) ops. The copies run with the ``TransferEngine`` (mem_cache/host_pool: the runtime's
``nrt_tensor_read_batch`` / ``nrt_tensor_write_batch``, one transfer thread per direction, fenced on the graphs
launched before it) with one region per (component, layer); its jobs complete the same HiCache events (``XferEvent``)
as the KV pool's, so a forward that reads a loaded slot waits for it (``set_consumer_waits``). In CPU mode the same
regions and op records run as host memmoves (``HostTransferEngine``).

Host-only checkpoints (upstream's ``SGLANG_OPT_MAMBA_HOST_ONLY_CHECKPOINT``) store through the same pool and wait on
the same events. Refused: upstream's MTP draft state (no Neuron speculative state), side slot states (refused by the
pool already), a layout other than layer_first (the platform sets it, platform.resolve_hicache) and an L3 storage
backend (refused at launch)."""

from __future__ import annotations

import logging
from math import prod
from typing import List, Sequence

import torch

from sglang.srt.mem_cache.pool_host.base import HostKVCache

from sglang_neuron.mem_cache import host_pool
from sglang_neuron.mem_cache.host_pool import page_runs

logger = logging.getLogger(__name__)


# ---- the slot regions of a state pool ---------------------------------------------------------------------------


class SlotRegions:
    """The (component, layer) regions of a state pool's slot tensors: views ``[layers, slots, *shape]`` (upstream's
    ``MambaStateBufferInfo`` temporal then conv tensors, or the pool's canonical tensors) whose one slot of one layer is
    contiguous. Region r = component c, layer l starts at byte ``offsets[r]`` of the tensor's storage; slot s of it at
    ``offsets[r] + s * slot_bytes[c]``."""

    def __init__(self, views: Sequence[torch.Tensor]):
        if not views:
            raise ValueError("a state pool's slot regions need at least one tensor")
        layers, slots = views[0].shape[:2]
        self.views = list(views)
        self.layers, self.slots = int(layers), int(slots)
        self.slot_bytes, self.tensors, self.offsets, self.components, self.layer_of = [], [], [], [], []
        for c, view in enumerate(views):
            if view.ndim < 2 or tuple(view.shape[:2]) != (self.layers, self.slots):
                raise ValueError(f"state tensor {c} {tuple(view.shape)} is not [layers {self.layers}, slots "
                                 f"{self.slots}, ...] like the first")
            inner = prod(view.shape[2:])
            if view.stride(1) != inner or not view[0, 0].is_contiguous():
                raise ValueError(f"state tensor {c} {tuple(view.shape)} strides {view.stride()}: one slot of one "
                                 "layer is not contiguous, so it is not one span the runtime can copy")
            elem = view.element_size()
            self.slot_bytes.append(inner * elem)
            for layer in range(self.layers):
                self.tensors.append(view)
                self.offsets.append((view.storage_offset() + layer * view.stride(0)) * elem)
                self.components.append(c)
                self.layer_of.append(layer)

    @property
    def bytes_per_slot(self) -> int:
        """One slot of every component and layer."""
        return sum(self.slot_bytes) * self.layers

    def page_bytes(self) -> List[int]:
        return [self.slot_bytes[c] for c in self.components]

    def host_bases(self, host_tensors: Sequence[torch.Tensor]) -> List[int]:
        """Each region's host address in host tensors laid out ``[layers, host slots, *shape]`` per component."""
        return [host_tensors[c].data_ptr() + layer * host_tensors[c].shape[1] * self.slot_bytes[c]
                for c, layer in zip(self.components, self.layer_of)]

    def check_host(self, host_tensors: Sequence[torch.Tensor], slots: int) -> None:
        for c, (view, host) in enumerate(zip(self.views, host_tensors)):
            want = (self.layers, slots, *view.shape[2:])
            if tuple(host.shape) != want or host.dtype != view.dtype or not host.is_contiguous():
                raise ValueError(f"state host copy {c}: {tuple(host.shape)} {host.dtype} is not the contiguous "
                                 f"{want} {view.dtype} the pool's slots copy into")


def copy_slots_now(regions: SlotRegions, host_tensors: Sequence[torch.Tensor], device_slots: Sequence[int],
                   host_slots: Sequence[int], to_host: bool) -> None:
    """Copy whole slots between the pool's tensors and host tensors ``[layers, host slots, *shape]``, synchronously:
    the device's queued graphs finish first (they may write these slots), and the copy has landed on return. For the
    retraction backup and its restore, which upstream runs between two device synchronizations too."""
    runs = page_runs(list(device_slots), list(host_slots))
    if not runs:
        return
    cols = (host_pool._column(regions.offsets), host_pool._column(regions.host_bases(host_tensors)),
            host_pool._column(regions.page_bytes()))
    ops = host_pool.batch_ops(runs, *cols)
    if regions.tensors[0].device.type == "cpu":
        host_pool.run_host_batch([t.untyped_storage().data_ptr() for t in regions.tensors], ops, to_host)
        return
    import torch_neuronx

    torch_neuronx.synchronize()
    found = {}
    for t in regions.tensors:
        if id(t) not in found:
            found[id(t)] = host_pool._nrt_handle(t)
    host_pool.run_nrt_batch([found[id(t)] for t in regions.tensors], ops, to_host)


def host_tensors_like(regions: SlotRegions, slots: int, *, pin: bool) -> List[torch.Tensor]:
    """Host tensors ``[layers, slots, *shape]`` per component in the pool's dtypes. ``pin``: pinned and page-locked,
    since the runtime's DMA reads and writes them directly (as the KV host pool's)."""
    out = []
    for view in regions.views:
        t = torch.empty((regions.layers, slots, *view.shape[2:]), dtype=view.dtype, pin_memory=pin)
        if pin and t.numel():
            host_pool._mlock(t)
        out.append(t)
    return out


# ---- the HiCache host pool --------------------------------------------------------------------------------------


class NeuronStateHostPool(HostKVCache):
    """The host copy of a Neuron state pool (``NeuronStatePool``) for HiCache L2: ``build_mamba_host_pool``'s answer
    for that pool, with MambaPoolHost's constructor arguments. One host slot holds one device slot (page size 1)."""

    def __init__(self, buffer_info, host_to_device_ratio: float, host_size: float, *, device_capacity: int,
                 pin_memory: bool = True, device: str = "cpu", allocator_type: str = "default",
                 layout: str = "layer_first", mtp_draft_buffer_infos=()):
        if mtp_draft_buffer_infos:
            raise ValueError("Neuron state HiCache keeps no MTP draft state (Neuron serves no speculative decoding)")
        if layout != "layer_first":
            raise ValueError(f"Neuron state HiCache keeps a layer_first host pool, not --hicache-mem-layout {layout}")
        if allocator_type not in (None, "default"):
            # the remote DRAM tier ("dynamic") is validated on the MHA host pool alone
            raise ValueError(f"Neuron state HiCache is L2 only: no --hicache-storage-backend ({allocator_type} "
                             "allocator; no remote DRAM tier for the state host pool)")
        buffer_info.validate()
        if buffer_info.extra_checkpoint_buffers:
            raise ValueError("Neuron state HiCache moves the pool's own components only; side slot states would be "
                             "left behind on a reload")
        self.buffer_info = buffer_info
        self.regions = SlotRegions([buffer_info.temporal, *buffer_info.conv])
        self.num_mamba_layers = self.layer_num = self.target_layer_num = self.regions.layers
        # device slots the transfers may name: 1..size (slot 0 is upstream's padding slot and stays zero)
        self.device_slot_num = self.regions.slots
        self.on_device = buffer_info.temporal.device.type != "cpu"
        self.device_pool = None
        super().__init__(dtype=buffer_info.temporal.dtype, device_capacity_tokens=int(device_capacity),
                         size_per_token=self.regions.bytes_per_slot, start_layer=0, end_layer=self.layer_num,
                         host_to_device_ratio=host_to_device_ratio, host_size=host_size, page_size=1, layout=layout,
                         pin_memory=pin_memory and self.on_device, device=device, allocator_type="default",
                         pool_label="state")
        engine_cls = host_pool.TransferEngine if self.on_device else host_pool.HostTransferEngine
        self.engine = engine_cls(self.regions.tensors, self.host_regions, self.regions.page_bytes(),
                                 dev_offsets=self.regions.offsets,
                                 host_bases=self.regions.host_bases(self.host_regions))
        host_pool._ENGINES.append(self.engine)
        logger.info("[neuron hicache] state host pool: %d slots x %d layers x %d components (%.2f GiB, %.1f MB a "
                    "slot)", self.size, self.layer_num, len(self.regions.views),
                    self.size * self.size_per_token / (1 << 30), self.size_per_token / 1e6)

    def get_size_per_token(self):
        return self.size_per_token

    def init_kv_buffer(self):
        """One host tensor per component, ``[layers, host slots, *shape]`` in the component's dtype; pinned and
        page-locked on a device (the runtime DMAs into it directly)."""
        self.host_regions = host_tensors_like(self.regions, self.size, pin=self.pin_memory)
        return self.host_regions

    def _slots(self, indices, limit: int, low: int, what: str) -> List[int]:
        idx = indices.reshape(-1).to("cpu", torch.int64) if isinstance(indices, torch.Tensor) else \
            torch.as_tensor(list(indices), dtype=torch.int64)
        slots = idx.tolist()
        bad = [s for s in slots if not low <= s < limit]
        if bad:
            raise ValueError(f"state HiCache {what} slots {bad} are outside [{low}, {limit})")
        return slots

    def _submit(self, to_host: bool, host_indices, device_indices) -> None:
        host = self._slots(host_indices, self.size, 0, "host")
        dev = self._slots(device_indices, self.device_slot_num, 1, "device")
        if len(host) != len(dev):
            raise ValueError(f"state HiCache: {len(host)} host slots for {len(dev)} device slots")
        if len(set(dev)) != len(dev) or len(set(host)) != len(host):
            raise ValueError("state HiCache slots repeat in one transfer")
        if dev:
            self.engine.submit(to_host, dev, host)

    def backup_from_device_all_layer(self, device_pool, host_indices, device_indices, io_backend):
        self._submit(True, host_indices, device_indices)

    def load_to_device_per_layer(self, device_pool, host_indices, device_indices, layer_id, io_backend, *,
                                 is_draft=False):
        """The whole slot at the first state layer: the compiled forward reads every layer at once, and waits for the
        load through the HiCache events (host_pool.set_consumer_waits)."""
        if is_draft:
            raise ValueError("Neuron state HiCache keeps no draft state")
        if layer_id == 0:
            self._submit(False, host_indices, device_indices)

    def get_data_page(self, index, flat=True):
        raise NotImplementedError("Neuron state HiCache is L2 only (no storage backend)")

    def get_dummy_flat_data_page(self):
        raise NotImplementedError("Neuron state HiCache is L2 only (no storage backend)")

    def set_from_flat_data_page(self, index, data_page):
        raise NotImplementedError("Neuron state HiCache is L2 only (no storage backend)")


# ---- the hook ---------------------------------------------------------------------------------------------------


def build_mamba_host_pool(original, mamba_pool, pool_info, *, host_size, layout, mtp_draft_device_pools=()):
    """hybrid_pool_assembler.build_mamba_host_pool: a Neuron state pool's host pool is NeuronStateHostPool, sized as
    upstream sizes MambaPoolHost (``--hicache-size``'s state share, else ``--hicache-ratio`` x the pool's
    ``host_capacity_tokens``, its slot count); any other pool keeps upstream's."""
    from sglang_neuron.mem_cache.state_pool import NeuronStatePool

    if not isinstance(mamba_pool, NeuronStatePool):
        return original(mamba_pool, pool_info, host_size=host_size, layout=layout,
                        mtp_draft_device_pools=mtp_draft_device_pools)
    if mtp_draft_device_pools:
        raise ValueError("Neuron state HiCache keeps no MTP draft state (Neuron serves no speculative decoding)")
    from sglang.srt.runtime_context import get_memory

    pool_info.buffer_info.validate(layer_ids=pool_info.layer_ids)
    capacity = mamba_pool.host_capacity_tokens
    from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler

    # the launch's storage backend reaches the pool's refusal (upstream's builder passes none)
    return NeuronStateHostPool(pool_info.buffer_info, get_memory().hicache_ratio, host_size,
                               device_capacity=mamba_pool.size if capacity is None else capacity, layout=layout,
                               allocator_type=hybrid_pool_assembler._get_allocator_type())


# Upstream functions the hook wraps or relies on, pinned by md5 against SGLang 43881a7 (sglang_patches.source_guard); a
# changed source stops the launch.
_SOURCES_MD5 = {
    # wrapped: builds MambaPoolHost (CUDA/NPU transfer kernels) from the pool, its info, host size and layout
    ("sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler", "build_mamba_host_pool"):
        "397a80fba7c7b4e0842816c2941f7f9a",
    # passes the state pool itself (not a view), its state-slot allocator's alloc/free, and splits --hicache-size
    # by _split_hicache_size; the host pool's load is per layer through the stage-local mamba layer mapping
    ("sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler", "build_hybrid_mamba_stack"):
        "f040e1e472580df544b24cf5552daa50",
    # weighs each pool's host share by host_capacity_bytes (the pool's canonical slot bytes), else get_kv_size_bytes
    ("sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler", "_split_hicache_size"):
        "731b24a0ca1abfef21fe60f66a7b5947",
    # the pool's get_device_pool_infos validates its buffers against it
    ("sglang.srt.mem_cache.device_pool_info", "MambaStateBufferInfo.validate"): "afa5a3c89b4b53773dc349977dece6e6",
    # the decode retraction backup: get_cpu_copy(mamba_pool_idx.unsqueeze(0)), and load_cpu_copy of that tuple into the
    # request's (new) slot
    ("sglang.srt.managers.schedule_batch", "Req.offload_kv_cache"): "c7e3ac57ef0e57182782dd7cee356583",
    ("sglang.srt.managers.schedule_batch", "Req.load_kv_cache"): "11d12a2123d347a2358c8036edaf7b91",
}


def register_hooks(registry, hook_type) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_SOURCES_MD5, __name__)
    registry.register("sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler.build_mamba_host_pool",
                      build_mamba_host_pool, hook_type.AROUND)
