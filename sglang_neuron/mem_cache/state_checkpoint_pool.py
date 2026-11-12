# SPDX-License-Identifier: Apache-2.0
"""A separately sized store for the radix cache's recurrent-state checkpoints, so cached prefixes do not take the
slots live requests run in.

Without it, upstream's radix cache keeps each checkpoint in a slot of the request pool (the donated or copied request
slot): a Qwen3.5 rank's slot is 23.6 MB, and every cached prefix is one slot fewer for running requests, so the
operator sizes one pool for both (upstream's ``MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO`` 3x the running requests)
and an agentic workload with heavy prefix reuse evicts checkpoints under load. Upstream separates them with its
``MambaCheckpointPool`` (``--enable-int8-mamba-checkpoint``): the radix commits each checkpoint into a pool of its
own (``KVStorage.store_checkpoint`` -> ``store_from_active``), frees the request's slot, and on a hit copies the
checkpoint into the request's fresh slot (``ModelRunner._maybe_execute_deferred_mamba_cow_and_clear`` ->
``load_to_active``); a full store evicts the least recently used checkpoint (``MambaComponent._alloc_int8_ckpt_slot``)
and a radix reset clears it (``StateStorage.clear``). This module is that store for a Neuron state pool, built in its
place by the same constructor (``maybe_init_int8_mamba_checkpoint_pool``, wrapped) and driven by the same calls.

Exact, not int8. Upstream quantizes the temporal state per (head, k-channel) over the d_v axis of ``[H, d_v, d_k]``.
On Neuron the store keeps the pool's own bytes, every component in the pool's dtype and physical layout, and a hit
restores them bit for bit:

* the Neuron radix cache's contract is that a hit replays a cold run's forwards over the same state bits
  (sglang_patches/chunk_keyed_tree); a once-rounded state is a different continuation, which upstream accepts and the
  continuation goldens here would not;
* the kernels keep the temporal state in their own layouts (GDN ``[K, vh, V]``, state_spec ``physical_shape``), so
  upstream's scale axis is a different axis per model, and a quantize/dequantize per layout is a kernel per model to
  compile and qualify; a copy is one shared graph (``CompiledStateOps.copy_across``) per tensor shape and bucket.

So ``--enable-int8-mamba-checkpoint`` on a Neuron state pool selects this exact store (named in the log), and
``--int8-mamba-ckpt-size`` sizes it in checkpoints (default: twice the request pool's slots, upstream's default).

Memory. Upstream allocates its store from the ``--mem-fraction-static`` headroom at its default size, where the
programs' instructions and scratch load after the pools: on a rank with 19.1 MB checkpoints, the default 128 take
2.31 GiB of it and leave 0.56 GB, and the prefill NEFF's load fails on every rank (it loads with ~2.3 GB free).
Counting the store in the KV budget does not help a launch whose KV pool is stated (``--max-total-tokens``, a compiled
dimension). So the store is sized once the configurator built every pool (``size_deferred_stores``, from
state_pool_class's ``init_pools``), from the device memory then free less a reserve for the programs' loads
(``SGLANG_NEURON_STATE_CKPT_RESERVE_GB``, 2 by default):

* the default size is the most checkpoints that leave the reserve, at most twice the request pool's slots (logged when
  fewer), and a launch where not one fits is refused;
* an explicit ``--int8-mamba-ckpt-size`` that would leave less than the reserve is refused, naming the most that fit.

The chosen size, its bytes and what is left are logged. Upstream's launch refusals hold (``handle_int8_mamba_checkpoint``: not with HiCache, LMCache, unified memory, or a custom radix
backend). With HiCache, upstream's host-only checkpoints (``SGLANG_OPT_MAMBA_HOST_ONLY_CHECKPOINT``) keep checkpoints
off the device instead (mem_cache/state_host_pool).

Layout. Banks of ``[layers, bank slots + 2, *shape]`` per component, each a zero slot 0, its slots and a trailing sink,
like the pool's tensors (so ``copy_across`` pads the same way), each bank under the 4 GiB an indexed DMA addresses
(``state_ops.MAX_TENSOR_BYTES``), all banks the same shape so they share one compiled graph per bucket. Checkpoint id g
(1..num_slots, upstream's ``MambaSlotAllocator`` ids) is slot ``(g - 1) % bank slots + 1`` of bank ``(g - 1) // bank
slots``. A working-bank pool's checkpoints are its canonical slots; the working rows are never stored.

The store binds to the request pool it serves when that pool is built (sglang_patches/megakernel_pool_layout's
``init_mamba_pool``): it checks the two agree on the state spec and warms ``copy_across`` both ways, so a radix insert
or hit never compiles on one rank while the others wait. A failed copy quarantines what it may have half written: the
store on a store, the request pool on a hit."""

from __future__ import annotations

import logging
import os
from math import ceil
from typing import Dict, List, Optional

import torch

from sglang_neuron.mem_cache.state_ops import MAX_TENSOR_BYTES, oversized_tensors
from sglang_neuron.mem_cache.state_spec import StateSpec

logger = logging.getLogger(__name__)


class StateCheckpointStoreQuarantined(RuntimeError):
    pass


def bank_geometry(spec: StateSpec, num_slots: int, max_tensor_bytes: int = MAX_TENSOR_BYTES) -> tuple:
    """(banks, slots per bank) holding ``num_slots`` checkpoints with every bank tensor (its slots plus the zero and
    sink slots) under ``max_tensor_bytes``, the banks as few and as even as that allows."""
    largest = max(c.slot_bytes for c in spec.components) * spec.num_layers  # one slot of the largest tensor
    per_bank = max_tensor_bytes // largest - 2
    if per_bank < 1:
        raise ValueError(f"{spec.family} checkpoint store: one slot of a component is {largest} bytes, so no bank of "
                         f"one checkpoint fits the {max_tensor_bytes}-byte tensor an indexed DMA addresses")
    banks = ceil(num_slots / per_bank)
    return banks, ceil(num_slots / banks)


class NeuronStateCheckpointPool:
    """Upstream ``MambaCheckpointPool``'s surface (alloc / free / available_size / clear / num_slots,
    store_from_active / load_to_active, mem_usage_bytes) over exact copies of a Neuron state pool's slots."""

    def __init__(self, *, spec: StateSpec, num_slots: Optional[int], ops, device,
                 max_tensor_bytes: int = MAX_TENSOR_BYTES, explicit=None, default: int = 0):
        """``num_slots`` None: sized and allocated later (``allocate``, from size_deferred_stores) for ``explicit``
        (--int8-mamba-ckpt-size) or at most ``default`` checkpoints."""
        if num_slots is not None and num_slots < 1:
            raise ValueError(f"{spec.family} checkpoint store: needs at least one checkpoint slot")
        if not callable(getattr(ops, "copy_across", None)):
            raise TypeError(f"{spec.family} checkpoint store: the model's state ops {type(ops).__name__} have no "
                            "copy_across (mem_cache/state_ops.StateOps), so a checkpoint cannot move between the "
                            "request pool and the store; launch without --enable-int8-mamba-checkpoint")
        self.spec = spec
        self.ops = ops
        self.device = device
        self.max_tensor_bytes = max_tensor_bytes
        self.explicit, self.default = explicit, int(default)
        self._active = None
        self._failed: Optional[BaseException] = None
        self.num_slots, self.banks, self.allocator = 0, [], None
        if num_slots is not None:
            self.allocate(num_slots)

    @property
    def allocated(self) -> bool:
        return self.allocator is not None

    def allocate(self, num_slots: int) -> None:
        """Allocate the banks of ``num_slots`` checkpoints (once), and bind the request pool if one waits."""
        from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator

        spec, max_tensor_bytes, device = self.spec, self.max_tensor_bytes, self.device
        if self.allocated:
            raise RuntimeError(f"{spec.family} checkpoint store: allocated already")
        if num_slots < 1:
            raise ValueError(f"{spec.family} checkpoint store: needs at least one checkpoint slot")
        self.num_slots = int(num_slots)
        self.num_banks, self.bank_slots = bank_geometry(spec, self.num_slots, max_tensor_bytes)
        self.banks: List[Dict[str, torch.Tensor]] = [
            {c.name: torch.zeros((spec.num_layers, self.bank_slots + 2, *c.shape), dtype=getattr(torch, c.dtype),
                                 device=device) for c in spec.components}
            for _ in range(self.num_banks)]
        problems = oversized_tensors(self.banks[0])
        if problems:  # bank_geometry keeps each under the limit; a guard against a changed limit
            raise ValueError(f"{spec.family} checkpoint store: " + "; ".join(problems))
        # ids 1..num_slots on the host, as the request pool's allocator keeps them (slot 0 is never handed out)
        self.allocator = MambaSlotAllocator(size=self.num_slots, device="cpu")
        logger.info("[neuron] %s radix checkpoint store: %d checkpoints in %d bank(s) of %d, exact (%s), %.2f GiB "
                    "(%.1f MB a checkpoint); --enable-int8-mamba-checkpoint keeps the pool's dtypes on Neuron",
                    spec.family, self.num_slots, self.num_banks, self.bank_slots,
                    ", ".join(f"{c.name} {c.dtype}" for c in spec.components), self.mem_usage_bytes() / (1 << 30),
                    spec.slot_bytes() / 1e6)
        if self._active is not None:
            self._warm(self._active)

    # -- lifecycle (upstream's allocator) ------------------------------------------------------------------------
    def alloc(self, n: int = 1):
        return self.allocator.alloc(n)

    def free(self, slots: torch.Tensor) -> None:
        self.allocator.free(slots.reshape(-1).to("cpu", torch.int64))

    def available_size(self) -> int:
        return self.allocator.available_size()

    def clear(self) -> None:
        """Release every checkpoint (radix reset). The bytes stay; a store overwrites every component and layer of
        its slot before a hit can read it."""
        self.allocator.clear()

    # -- binding -------------------------------------------------------------------------------------------------
    def bind_active(self, pool) -> None:
        """The request pool this store serves: the same state spec (the copy is per component and layer), and
        ``copy_across`` compiled both ways now (one bank stands for all: they share shapes)."""
        from sglang_neuron.mem_cache.state_pool import NeuronStatePool

        if not isinstance(pool, NeuronStatePool):
            raise TypeError(f"{self.spec.family} checkpoint store: serves a NeuronStatePool, not {type(pool).__name__}")
        if pool.spec_stage.digest() != self.spec.digest():
            raise ValueError(f"{self.spec.family} checkpoint store: built for layers {list(self.spec.layer_ids)}, "
                             f"and the request pool holds another state ({pool.spec_stage.family}, layers "
                             f"{list(pool.spec_stage.layer_ids)})")
        self._active = pool
        if self.allocated:  # else warmed once allocated
            self._warm(pool)

    def _warm(self, pool) -> None:
        warm = getattr(self.ops, "warm_across", None)
        if callable(warm):
            warm(pool.canonical_tensors(), self.banks[0])

    # -- moving checkpoints --------------------------------------------------------------------------------------
    def store_from_active(self, active_mamba_pool, active_slots, ckpt_slots) -> None:
        """Copy the request pool's slots ``active_slots`` into checkpoints ``ckpt_slots`` (the radix insert)."""
        pool = self._pool(active_mamba_pool)
        src = pool._slots(_flat(active_slots), "checkpoint source")
        dst = self._ids(ckpt_slots, "checkpoint destination")
        if len(src) != len(dst):
            raise ValueError(f"checkpoint store: {len(src)} sources for {len(dst)} checkpoints")
        if len(set(dst)) != len(dst):
            raise ValueError("checkpoint store destinations must be unique")
        pool._check()
        self._check()
        try:
            for bank, pairs in self._by_bank(dst, src).items():
                self.ops.copy_across(pool.canonical_tensors(), self.banks[bank], [s for _, s in pairs],
                                     [local for local, _ in pairs])
        except BaseException as exc:
            self._failed = exc
            raise

    def load_to_active(self, active_mamba_pool, ckpt_slots, active_slots) -> None:
        """Copy checkpoints ``ckpt_slots`` into the request pool's slots ``active_slots`` (the copy-on-write of a
        hit); the checkpoints are unchanged."""
        pool = self._pool(active_mamba_pool)
        src = self._ids(ckpt_slots, "checkpoint source")
        dst = pool._slots(_flat(active_slots), "checkpoint restore")
        if len(src) != len(dst):
            raise ValueError(f"checkpoint restore: {len(src)} checkpoints for {len(dst)} slots")
        if len(set(dst)) != len(dst):
            raise ValueError("checkpoint restore destinations must be unique")
        self._check()
        for bank, pairs in self._by_bank(src, dst).items():
            pool._run(self.ops.copy_across, self.banks[bank], pool.canonical_tensors(),
                      [local for local, _ in pairs], [d for _, d in pairs])

    # -- sizing --------------------------------------------------------------------------------------------------
    def mem_usage_bytes(self) -> int:
        return sum(t.nbytes for bank in self.banks for t in bank.values())

    @property
    def quarantined(self) -> bool:
        return self._failed is not None

    # -- helpers -------------------------------------------------------------------------------------------------
    def _pool(self, pool):
        if not self.allocated:
            raise RuntimeError(f"{self.spec.family} checkpoint store: used before it was sized (size_deferred_stores)")
        if self._active is None:
            self.bind_active(pool)
        elif pool is not self._active:
            raise ValueError(f"{self.spec.family} checkpoint store: serves the request pool it was bound to, not "
                             "another")
        return pool

    def _check(self) -> None:
        if self._failed is not None:
            raise StateCheckpointStoreQuarantined(
                f"{self.spec.family} checkpoint store is quarantined after a failed copy "
                f"({type(self._failed).__name__}: {self._failed})") from self._failed

    def _ids(self, values, what: str) -> List[int]:
        from sglang_neuron.mem_cache.state_pool import NeuronStatePool

        ids = NeuronStatePool._host(_flat(values))
        bad = [g for g in ids if not 1 <= g <= self.num_slots]
        if bad:
            raise ValueError(f"{what}: checkpoints {bad} are outside [1, {self.num_slots}]")
        return ids

    def _by_bank(self, ids: List[int], others: List[int]) -> Dict[int, list]:
        """{bank: [(slot in the bank, the paired request slot)]}, in call order within a bank."""
        out: Dict[int, list] = {}
        for g, other in zip(ids, others):
            out.setdefault((g - 1) // self.bank_slots, []).append(((g - 1) % self.bank_slots + 1, other))
        return out


def _flat(values):
    return values.reshape(-1) if isinstance(values, torch.Tensor) else values


# ---- the store's size: the free HBM after the pools, less the program-load reserve ------------------------------

_GIB = 1 << 30
RESERVE_ENV = "SGLANG_NEURON_STATE_CKPT_RESERVE_GB"
RESERVE_GB = 2.0  # its default: the Memory note's rank loads its programs with ~2.3 GB free and fails at 0.56 GB
# stores built while their request pool's configurator runs, sized once the pools are built (size_deferred_stores)
_DEFERRED: list = []


def store_bytes(spec: StateSpec, num_slots: int, max_tensor_bytes: int = MAX_TENSOR_BYTES) -> int:
    """The HBM of a store of ``num_slots`` checkpoints: every bank tensor, its zero and sink slots included."""
    banks, per_bank = bank_geometry(spec, num_slots, max_tensor_bytes)
    slot = sum(_prod(c.shape) * getattr(torch, c.dtype).itemsize for c in spec.components) * spec.num_layers
    return banks * (per_bank + 2) * slot


def _prod(shape) -> int:
    out = 1
    for d in shape:
        out *= int(d)
    return out


def reserve_bytes() -> int:
    """The device memory the store leaves free for the programs' loads (their instruction staging, scratchpads and
    DMA rings): SGLANG_NEURON_STATE_CKPT_RESERVE_GB, 2 GiB by default."""
    raw = os.environ.get(RESERVE_ENV, "").strip()
    try:
        gb = float(raw) if raw else RESERVE_GB
    except ValueError:
        raise ValueError(f"{RESERVE_ENV}={raw!r} is not a number of GiB") from None
    if gb < 0:
        raise ValueError(f"{RESERVE_ENV}={gb}: a number of GiB, 0 or more")
    return int(gb * _GIB)


def size_store(spec: StateSpec, *, free: int, reserve: int, explicit=None, default: int = 0,
               max_tensor_bytes: int = MAX_TENSOR_BYTES) -> int:
    """The checkpoints the store holds out of ``free`` device bytes (what the built pools left) less ``reserve`` (the
    programs' loads): ``explicit`` (--int8-mamba-ckpt-size) if it leaves the reserve, else refused naming the most that
    do; else the most that fit, at most ``default`` (twice the request pool's slots); not one fitting is refused."""

    def fits(n):
        return free - store_bytes(spec, n, max_tensor_bytes) >= reserve

    def most(upper):
        lo, hi = 0, int(upper)  # the largest fitting n in [0, upper]
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if fits(mid):
                lo = mid
            else:
                hi = mid - 1
        return lo

    have = (f"{free / _GIB:.2f} GiB of device memory free after the pools, {reserve / _GIB:.2f} GiB of it reserved "
            f"for loading the programs ({RESERVE_ENV})")
    if explicit:
        explicit = int(explicit)
        if fits(explicit):
            return explicit
        raise ValueError(f"--int8-mamba-ckpt-size {explicit} does not fit: {explicit} {spec.family} checkpoints take "
                         f"{store_bytes(spec, explicit, max_tensor_bytes) / _GIB:.2f} GiB, with {have}; at most "
                         f"{most(explicit)} fit. Lower --int8-mamba-ckpt-size, --max-total-tokens or "
                         "--mem-fraction-static")
    n = most(int(default))
    if n < 1:
        raise ValueError(f"--enable-int8-mamba-checkpoint: not one {spec.family} checkpoint "
                         f"({store_bytes(spec, 1, max_tensor_bytes) / 1e6:.1f} MB) fits, with {have}; lower "
                         "--max-total-tokens or --mem-fraction-static, or launch without --enable-int8-mamba-checkpoint")
    if n < default:
        logger.warning("[neuron] %s radix checkpoint store: %d checkpoints of the default %d (twice the request pool) "
                       "fit with %s", spec.family, n, default, have)
    return n


def _free_device_bytes() -> Optional[int]:
    """This rank's free device memory now (the platform's total less what the runtime holds), or None without a device
    to read (CPU mode, the host: nothing is probed, no neuron-ls)."""
    from sglang_neuron.backend import device_io

    if not device_io.on_device():
        return None
    from sglang.srt.platforms import current_platform

    total = int(current_platform.get_device_total_memory(0))
    return max(total - int(current_platform.get_current_memory_usage()), 0)


def size_deferred_stores() -> None:
    """Once the configurator built every pool (state_pool_class.init_pools): size each store built meanwhile from the
    free device memory left (size_store), allocate it and bind it to its request pool."""
    stores, _DEFERRED[:] = list(_DEFERRED), []
    for store in stores:
        free, reserve = _free_device_bytes(), reserve_bytes()
        if free is None:  # no device memory to read: the requested size, else the default
            store.allocate(int(store.explicit or store.default))
            continue
        n = size_store(store.spec, free=free, reserve=reserve, explicit=store.explicit, default=store.default)
        store.allocate(n)
        logger.info("[neuron] %s radix checkpoint store sized after the pools: %d checkpoints (%s), %.2f GiB of %.2f "
                    "GiB free, %.2f GiB left for the programs' loads", store.spec.family, n,
                    "--int8-mamba-ckpt-size" if store.explicit else f"default at most {store.default}",
                    store.mem_usage_bytes() / _GIB, free / _GIB, (free - store.mem_usage_bytes()) / _GIB)


# ---- the hook ---------------------------------------------------------------------------------------------------


def _requested() -> tuple:
    """(--enable-int8-mamba-checkpoint, --int8-mamba-ckpt-size); off without a server-args context, as upstream."""
    from sglang.srt.runtime_context import get_exec

    try:
        mamba = get_exec().mamba
    except ValueError:
        return False, None
    if mamba is None:
        return False, None
    return bool(mamba.enable_int8_mamba_checkpoint), mamba.int8_mamba_ckpt_size


def maybe_init_checkpoint_pool(original, *, mamba_size, cache_params, mamba_layer_ids, device):
    """mamba_checkpoint_pool.maybe_init_int8_mamba_checkpoint_pool: for a Neuron state pool (the class the megakernel
    pool-layout seam is building), this module's exact store, on the rank's Neuron device; any other pool keeps
    upstream's."""
    from sglang_neuron.mem_cache.state_pool import NeuronStatePool
    from sglang_neuron.sglang_patches import state_pool_class as seam

    cls = seam._STATE_POOL_CLS.get()
    if cls is None or not (isinstance(cls, type) and issubclass(cls, NeuronStatePool)):
        return original(mamba_size=mamba_size, cache_params=cache_params, mamba_layer_ids=mamba_layer_ids,
                        device=device)
    enabled, size = _requested()
    if not enabled:
        return None
    spec = cls.spec
    ids = tuple(int(i) for i in mamba_layer_ids)
    stage = spec if ids == spec.layer_ids else spec.stage(min(ids), max(ids) + 1)
    from sglang_neuron.model_executor.runtime_init import neuron_device

    default = 2 * int(mamba_size)
    if _free_device_bytes() is None:
        # no device memory to read (CPU mode, the host): the requested size, else twice the request pool, now
        return NeuronStateCheckpointPool(spec=stage, num_slots=int(size or default), ops=cls.ops,
                                         device=neuron_device())
    # sized once every pool is built (size_deferred_stores): --int8-mamba-ckpt-size, else at most twice the request
    # pool's slots, either leaving the programs' load reserve free
    store = NeuronStateCheckpointPool(spec=stage, num_slots=None, ops=cls.ops, device=neuron_device(),
                                      explicit=size, default=default)
    _DEFERRED.append(store)
    return store


# Upstream functions the store wraps or relies on, pinned by md5 against SGLang 43881a7 (sglang_patches.source_guard);
# a changed source stops the launch.
_SOURCES_MD5 = {
    # wrapped: built inside HybridReqToTokenPool._init_mamba_pool (pinned by the seam) after the request pool
    ("sglang.srt.mem_cache.mamba_checkpoint_pool", "maybe_init_int8_mamba_checkpoint_pool"):
        "600aeb7855395b74783acd6b9e6d3d9c",
    # the radix insert: store_from_active(state.pool, active.view(-1), checkpoint)
    ("sglang.srt.mem_cache.kv.storage", "KVStorage.store_checkpoint"): "603bfc09ce044e066cb206e54a55917e",
    ("sglang.srt.mem_cache.static_kv_storage", "StaticKVStorage.store_checkpoint"):
        "603bfc09ce044e066cb206e54a55917e",
    # takes one checkpoint slot per insert, evicting a cached checkpoint when the store is full
    ("sglang.srt.mem_cache.unified_cache.components.mamba", "MambaComponent._alloc_int8_ckpt_slot"):
        "2f6e0934a6734dd27bfdfa4926b660ac",
    # the hit: load_to_active(state.pool, cow sources, cow destinations), host index tensors
    ("sglang.srt.model_executor.model_runner", "ModelRunner._maybe_execute_deferred_mamba_cow_and_clear"):
        "e6563c974d16e638dc56944e655ebb95",
    # the launch refusals the store relies on (HiCache, LMCache, unified memory, a custom radix backend)
    ("sglang.srt.arg_groups.mamba_hook", "handle_int8_mamba_checkpoint"): "0828d30abbf63df6ce4e5742ea55fb71",
}


def register_hooks(registry, around) -> None:
    """Registered with the megakernel pool-layout seam (sglang_patches/megakernel_pool_layout.register), which builds
    the Neuron state pools this store serves."""
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_SOURCES_MD5, __name__)
    registry.register("sglang.srt.mem_cache.mamba_checkpoint_pool.maybe_init_int8_mamba_checkpoint_pool",
                      maybe_init_checkpoint_pool, around)

