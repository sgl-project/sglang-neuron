# SPDX-License-Identifier: Apache-2.0
"""The recurrent-state pool: upstream SGLang's ``MambaPool`` over a model's declared state components, its slot
operations delegated to the model's kernels.

SGLang keeps every piece of recurrent-state bookkeeping: ``HybridReqToTokenPool`` maps a request to a state slot,
``MambaSlotAllocator`` hands slots out, the radix cache's mamba component (``UnifiedRadixCache`` +
``MambaComponent``) owns checkpoint slots on tree nodes, chooses copy-on-write sources and evicts, and the PD
connector registers and sends the pool's state entries. The platform supplies the pool class only
(``HybridReqToTokenPool.mamba_pool_cls``, which sglang_patches/state_pool_class sets per model: a megakernel model's
``PoolLayout.state_pool_class``, else the model's ``NeuronModel.state_pool_class``), built here by
``NeuronStatePool.bind(spec, ops)``:

* storage: on this rank's Neuron device (``neuron_device()``, not the host device SGLang passes the request table), one
  tensor per component, ``[state layers, slots, *physical shape]``, physical slots = ``size + 1`` (slot
  0 is upstream's padding slot and stays zero; the allocator hands out 1..size) plus one trailing sink a kernel's
  padded bucket may write; ``mamba_cache`` is upstream's ``MambaPool.State`` over slots [0, size], so every upstream
  reader (layer views, PD entries, sizing) sees upstream's layout;
* conv slicing: components in upstream's per-slot shapes keep upstream's conv slice axis and q/k/v shard groups
  (``cache_params.shape``: KDA's window-first conv slices on axis 1), so SGLang's PD transfer slices the state between
  attention TP sizes (a TP16 prefill to a TP32 decode) as it slices its own pool's;
* an optional working bank (``StateSpec.working_rows``): ``work.<component>`` ``[layers, rows, *shape]``, row 0
  scratch; the kernel advances these and ``load_rows`` / ``publish_rows`` move state between canonical slots and rows;
  a model without one updates its canonical slots in place (Qwen3.5);
* ``clear_slots`` / ``copy_from`` (upstream's names, which SGLang calls for fresh slots and copy-on-write) validate
  host slot ids and call the model's ``StateOps``; ``drain`` waits for every submitted operation;
* PD across attention TPs: a component stored in a kernel layout (``physical_shape``) has its entries marked not
  head-sliceable, so a cross-TP pairing is refused at registration (disaggregation/state_transfer); at equal TP
  upstream sends each slot whole;
* PD registration: the entries' addresses are the Neuron runtime's device addresses
  (``backend.device_io.device_address``, what NIXL registers and a WRITE targets), not torch-neuronx's placeholder
  ``data_ptr()``;
* host copies of whole slots (every component of every layer, canonical slots only): HiCache L2 builds its host pool
  from ``get_device_pool_infos`` (upstream's ``MambaStateBufferInfo`` over ``mamba_cache``) and sizes it by
  ``host_capacity_tokens`` (the slot count) and ``host_capacity_bytes`` (the canonical slots' bytes, not the working
  bank or the padding/sink slots), through mem_cache/state_host_pool; ``get_cpu_copy`` / ``load_cpu_copy`` are
  upstream's retraction backup and restore of a request's slot, synchronous and slot-granular;
* a separate radix checkpoint store (mem_cache/state_checkpoint_pool) binds to the pool it serves at construction
  (``HybridReqToTokenPool._init_mamba_pool``, through sglang_patches/megakernel_pool_layout) and copies slots into and
  out of it with the model's ``copy_across``;
* a failed operation or drain quarantines the pool: every later operation raises with the first failure, and slots
  are never handed to new owners over state an operation may still write;
* upstream options the pool does not implement (speculative state, ReplaySSM rings, the envelope layout, memory
  saver, sibling slot states) are refused at construction, with the option named, and so is a radix cache whose
  checkpoints this kernel cannot make exact (``radix_refusal`` over SGLang's resolved ``RadixSettings``: the page
  grid, or the chunk-keyed tree's chunk grid).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from operator import index as _int
from typing import Dict, Optional, Sequence

import torch

from sglang.srt.mem_cache.memory_pool import MambaPool

from sglang_neuron.mem_cache.state_ops import (CompiledStateOps, TorchStateOps, check_ops, oversized_tensors,
                                               working_bank_ops)
from sglang_neuron.mem_cache.state_spec import StateSpec

logger = logging.getLogger(__name__)

# MambaPool constructor options and the value that means "off"; a truthy value is refused
_OFF_OPTIONS = ("enable_memory_saver", "speculative_num_draft_tokens", "enable_linear_replayssm", "envelope_layout",
                "enable_linear_replayssm_spec", "replayssm_request_ring")


def _dtype(name: str) -> torch.dtype:
    return getattr(torch, name)


@dataclass(frozen=True)
class RadixSettings:
    """SGLang's resolved settings that decide whether a radix checkpoint of the state can be exact."""

    disabled: bool  # --disable-radix-cache
    extra_buffer: bool  # the mamba tracking buffer (--mamba-radix-cache-strategy extra_buffer)
    chunk_keyed: bool  # the chunk-keyed tree (--radix-cache-backend neuron-chunk)
    page: int  # the tree's page: the allocator's (widened under DCP)
    chunk: int  # --chunked-prefill-size
    cache_chunk: int  # upstream's mamba cache chunk
    # the Kimi-K3 prompt cache (--radix-cache-backend kimi-prompt): END entries at forward ends, tracks at forward ends
    kimi_prompt: bool = False

    @property
    def checkpoint_interval(self) -> int:
        """Upstream's checkpoint grid on SGLang's paged tree (runtime_context.mamba_checkpoint_grid)."""
        return math.lcm(self.cache_chunk, self.page)


def radix_settings() -> RadixSettings:
    """The launch's RadixSettings, read from SGLang's runtime context and the exported serving policy in the scheduler
    process."""
    from sglang.srt import runtime_context

    from sglang_neuron import serving_policy
    from sglang_neuron.distributed.decode_context_parallel import runtime_layout

    page_size = int(runtime_context.get_schedule().page_size or 1)  # unset: upstream's default 1
    dcp = runtime_layout(page_size)
    disabled = bool(runtime_context.get_memory().disable_radix_cache)
    # upstream's mamba cache chunk resolves from the model's config; without a tree nothing reads it (radix_refusal)
    cache_chunk = 0 if disabled else int(runtime_context.mamba_cache_chunk_size())
    return RadixSettings(disabled=disabled,
                         extra_buffer=bool(runtime_context.get_exec().mamba.enable_mamba_extra_buffer),
                         chunk_keyed=bool(serving_policy.current().chunk_keyed_tree),
                         kimi_prompt=bool(serving_policy.current().kimi_prompt_cache),
                         page=page_size * (1 if dcp is None else dcp.size),
                         chunk=int(runtime_context.get_schedule().chunked_prefill_size or 0),
                         cache_chunk=cache_chunk)


def radix_refusal(spec: StateSpec, settings: RadixSettings) -> Optional[str]:
    """Why SGLang's radix cache cannot keep exact checkpoints of ``spec``'s state under ``settings``, or None: on the
    chunk-keyed tree its chunk grid (state_checkpoint.chunk_grid_problem), else upstream's page grid
    (state_checkpoint.radix_problem)."""
    from sglang_neuron.mem_cache.state_checkpoint import chunk_grid_problem, radix_problem

    if settings.disabled:
        return None
    if settings.kimi_prompt:
        # an END entry adopts the state a forward ends with, and a chunk end the tracking slot that the same forward
        # writes (mem_cache/kimi_prompt_backend): no state inside a forward, so any kernel serves it
        if not settings.extra_buffer:
            return (f"{spec.family} state on the kimi-prompt prefix cache: its chunk ends take upstream's tracking "
                    "slots; launch with the mamba tracking buffer (--mamba-radix-cache-strategy extra_buffer)")
        return None
    if settings.chunk_keyed:
        problem = chunk_grid_problem(spec, chunk=settings.chunk, cache_chunk=settings.cache_chunk,
                                     extra_buffer=settings.extra_buffer)
    else:
        problem = radix_problem(spec, checkpoint_interval=settings.checkpoint_interval, page_size=settings.page,
                                extra_buffer=settings.extra_buffer)
    return None if problem is None else problem + "; or launch with --disable-radix-cache"


class StatePoolQuarantined(RuntimeError):
    pass


class NeuronStatePool(MambaPool):
    """Bind with ``NeuronStatePool.bind(spec, ops)``; SGLang constructs the bound class with MambaPool's arguments."""

    spec: Optional[StateSpec] = None
    ops = None
    conv_window_axis = -1
    _slot_siblings = ()

    @classmethod
    def bind(cls, spec: StateSpec, ops=None, *, name: Optional[str] = None) -> type:
        """A pool class for this model's state: ``spec`` the full model's declaration (each pipeline stage keeps the
        layers SGLang passes it), ``ops`` the model's kernel slot operations (when None: the eager TorchStateOps
        without a working bank; with one, state_ops.working_bank_ops: the model's row programs, the TorchStateOps reference when
        None, with CompiledStateOps on the canonical slots unless the ops carry them)."""
        if not isinstance(spec, StateSpec):
            raise TypeError("a state pool binds a StateSpec")
        # a kernel-layout component is not refused here: it is marked not head-sliceable (slice axis None), so a PD
        # transfer across attention TP sizes is refused at registration instead
        if spec.working_rows is not None:
            # a working-bank pool: its load / publish are its kernels' own programs (the reference when it has none),
            # and its canonical slots, plain tensors, take the compiled slot operations, copy_across included, unless
            # the model's ops carry them (state_ops.working_bank_ops)
            ops = working_bank_ops(ops)
        elif ops is None:
            # a model that names no slot operations keeps the eager TorchStateOps (Kimi-K3's prefill state pool);
            # a model whose kernels advance the canonical slots in place passes its compiled ops (Qwen3.5's
            # GdnStateOps, a CompiledStateOps)
            ops = TorchStateOps()
        check_ops(ops, working=spec.working_rows is not None)
        return type(name or f"NeuronStatePool_{spec.family}", (cls,), {"spec": spec, "ops": ops})

    def __init__(self, *, size: int, spec_state_size: int, cache_params, mamba_layer_ids: Sequence[int], device,
                 speculative_eagle_topk: Optional[int] = None, linear_replayssm_cache_len: int = 16, **options):
        spec = type(self).spec
        if spec is None:
            raise TypeError("NeuronStatePool is unbound: build the pool class with NeuronStatePool.bind(spec, ops)")
        unknown = sorted(set(options) - set(_OFF_OPTIONS))
        on = sorted(k for k in _OFF_OPTIONS if options.get(k))
        if speculative_eagle_topk not in (None, 0, 1):
            on.append("speculative_eagle_topk")
        if unknown or on:
            raise ValueError(f"{spec.family} state pool: upstream options {on + unknown} have no Neuron state "
                             "implementation (speculative state, ReplaySSM rings, envelope layout, memory saver); "
                             "launch without them")
        ids = tuple(_int(i) for i in mamba_layer_ids)
        missing = [i for i in ids if i not in spec.layer_ids]
        if missing:
            raise ValueError(f"{spec.family} state pool: SGLang asks for state layers {missing} the model did not "
                             f"declare (declared {list(spec.layer_ids)})")
        stage = spec if ids == spec.layer_ids else spec.stage(min(ids), max(ids) + 1)
        stage.check_cache_params(cache_params, ids)
        if size < 1:
            raise ValueError("a state pool needs at least one slot")

        problem = radix_refusal(stage, radix_settings())
        if problem is not None:
            raise ValueError(problem)

        self.spec_stage = stage
        self.size = int(size)
        # the state lives on this rank's Neuron device, where the kernels read it: SGLang builds its request tables on
        # the host under the platform, so the device it passes (``device``) is the host's (the KV pools likewise
        # allocate on neuron_device(), mem_cache/memory_pool.py)
        from sglang_neuron.model_executor.runtime_init import neuron_device

        device = neuron_device()
        self.device = device
        self.mamba_layer_ids = list(ids)
        self.num_mamba_layers = len(ids)
        self.debug_memory_pool = False
        self.enable_custom_mem_pool, self.custom_mem_pool = False, None
        self.enable_linear_replayssm = self.enable_linear_replayssm_spec = False
        # upstream's ReplaySSM ring attributes, read by HybridReqToTokenPool.alloc and the radix cache: none here
        self.replayssm_write_pos = self.replayssm_spec_write_pos = None
        self.replayssm_cache_base = self.replayssm_is_flush = None
        self.replayssm_spec_fold = False
        # upstream's PD slicing of the state between attention TP sizes reads these as its own pool sets them: the
        # cache parameters' conv slice axis (1 for KDA's window-first conv, 0 for GDN's channels-first) and q/k/v shard
        # groups (the spec's declared groups when upstream's shape names none). A kernel layout other than upstream's
        # per-slot shape has no head-sliceable conv axis: its entries carry slice axis None.
        upstream_layout = all(c.physical_shape is None for c in stage.convs)
        # a spec whose conv orders its axes unlike upstream's declares its own channel axis (state_spec's
        # conv_slice_axis: KDA channel-major is 0); the cache parameters' axis otherwise
        if stage.conv_slice_axis is not None:
            self.conv_slice_axis = int(stage.conv_slice_axis)
        else:
            self.conv_slice_axis = int(getattr(cache_params.shape, "conv_slice_axis", 0) or 0) if upstream_layout else 0
        groups = getattr(cache_params.shape, "conv_shard_groups", None)
        if groups is None:
            groups = dict(stage.extra).get("conv_shard_groups")
        self.conv_shard_groups = list(groups) if groups is not None and upstream_layout else None
        self.physical_slots = self.size + 2  # padding slot 0, slots 1..size, trailing sink
        self.sink_slot = self.size + 1
        self.tensors: Dict[str, torch.Tensor] = {}
        for c in stage.components:
            self.tensors[c.name] = torch.zeros((self.num_mamba_layers, self.physical_slots, *c.shape),
                                               dtype=_dtype(c.dtype), device=device)
            if stage.working_rows is not None:
                self.tensors["work." + c.name] = torch.zeros((self.num_mamba_layers, stage.working_rows, *c.shape),
                                                             dtype=_dtype(c.dtype), device=device)
        view = slice(0, self.size + 1)
        self.mamba_cache = MambaPool.State(
            temporal=self.tensors[stage.recurrent.name][:, view],
            conv=[self.tensors[c.name][:, view] for c in stage.convs])
        self.mem_usage = sum(t.nbytes for t in self.tensors.values()) / (1 << 30)
        # HiCache sizing (hybrid_pool_assembler): the host pool holds slots, so its device capacity is the slot count,
        # and --hicache-size is split by the bytes of those slots, not by the working bank and padding/sink slots
        self.host_capacity_tokens = self.size
        self.host_capacity_bytes = self.size * stage.slot_bytes()
        problems = oversized_tensors(self.tensors)
        if problems:
            raise ValueError(f"{stage.family} state pool: " + "; ".join(problems) + " (fewer state slots per rank: a "
                             "lower --max-running-requests / --max-mamba-cache-size or --mem-fraction-static)")
        # the slot operations' compiled shapes, now on every rank at the same point: one first compiled at a live
        # request's finish would hold that replica's ranks while the others wait in the next step's collectives
        warm = getattr(self.ops, "warm", None)
        if callable(warm):
            warm(self.tensors)
        self._failed: Optional[BaseException] = None
        logger.info("[neuron] %s state pool: %d slots x %d layers, %.3f GiB (ops %s)", stage.family, self.size,
                    self.num_mamba_layers, self.mem_usage, getattr(self.ops, "name", type(self.ops).__name__))

    # -- upstream MambaPool surface ----------------------------------------------------------------------------
    def register_slot_state(self, state) -> None:
        raise ValueError(f"{self.spec_stage.family} state pool: a sibling slot state ({type(state).__name__}) would "
                         "ride along on clear/copy/transfer through kernels that do not know it; not supported")

    def clear_slots(self, indices) -> None:
        slots = self._slots(indices, "clear")
        if slots:
            self._run(self.ops.clear_slots, self.tensors, slots)

    def copy_from(self, src_indices, dst_indices) -> None:
        src, dst = self._slots(src_indices, "copy source"), self._slots(dst_indices, "copy destination")
        if len(src) != len(dst):
            raise ValueError(f"state copy: {len(src)} sources for {len(dst)} destinations")
        if len(set(dst)) != len(dst) or set(src) & set(dst):
            raise ValueError("state copy destinations must be unique and disjoint from the sources (copy-on-write "
                             "never writes the checkpoint it reads)")
        if dst:
            self._run(self.ops.copy_slots, self.tensors, src, dst)

    def _transfer_order(self):
        """(component, upstream field, ``[layers, slots + 1, *shape]`` tensor) in upstream's PD entry order: each conv
        tensor, then the temporal tensor."""
        stage = self.spec_stage
        order = [(c, "conv", self.mamba_cache.conv[i]) for i, c in enumerate(stage.convs)]
        order.append((stage.recurrent, "temporal", self.mamba_cache.temporal))
        return order

    def _iter_transfer_state_entries(self):
        """Upstream's PD state entries in upstream's order (each conv tensor's layers, then the temporal tensor's),
        with slice axis None (upstream's "not head-sliceable" marker, dimension 0) for a component stored in a kernel
        layout other than upstream's: a transfer between different attention TPs is then refused at registration
        (disaggregation/state_transfer) instead of slicing the wrong axis."""
        for component, field, tensor in self._transfer_order():
            axis = None if component.physical_shape is not None else (self.conv_slice_axis if field == "conv" else 0)
            for layer_index, layer_id in enumerate(self.mamba_layer_ids):
                yield field, tensor[layer_index], axis, layer_id

    def get_contiguous_buf_infos(self):
        """Upstream's PD registration of the state entries (``MambaPool.get_contiguous_buf_infos``: each entry's
        address, bytes and one slot's bytes, in ``_iter_transfer_state_entries`` order) with the Neuron runtime's device
        addresses: torch-neuronx's ``data_ptr()`` is a placeholder that NIXL cannot register and a WRITE would miss
        (backend.device_io.device_address; on the host it is ``data_ptr()``)."""
        from sglang_neuron.backend.device_io import device_address

        data_ptrs, data_lens, item_lens = [], [], []
        for _, state_tensor, _, _ in self._iter_transfer_state_entries():
            data_ptrs.append(device_address(state_tensor))
            data_lens.append(state_tensor.nbytes)
            item_lens.append(state_tensor[0].nbytes)
        return data_ptrs, data_lens, item_lens

    def get_kv_size_bytes(self):
        return sum(t.nbytes for t in self.tensors.values())

    def canonical_tensors(self) -> Dict[str, torch.Tensor]:
        """{component: [layers, slots + 2, *shape]}: the slots requests and checkpoints own, without the working bank."""
        return {name: t for name, t in self.tensors.items() if not name.startswith("work.")}

    def get_device_pool_infos(self):
        """Upstream's state buffers for HiCache's host pool (hybrid_pool_assembler.build_mamba_host_pool, answered by
        mem_cache/state_host_pool for this pool): ``mamba_cache``'s temporal and conv views ``[layers, slots + 1,
        *shape]`` in the pool's physical layout, global layer ids. The working bank is scratch the forward refills,
        never a host copy; the sink is outside the views."""
        from sglang.srt.mem_cache.device_pool_info import DevicePoolInfo, MambaStateBufferInfo
        from sglang.srt.mem_cache.hicache_storage import PoolName

        info = MambaStateBufferInfo(temporal=self.mamba_cache.temporal, conv=tuple(self.mamba_cache.conv))
        layer_ids = tuple(self.mamba_layer_ids)
        try:
            info.validate(layer_ids=layer_ids)
        except ValueError as exc:
            raise ValueError(f"{self.spec_stage.family} state pool: its tensors are not upstream's state buffers "
                             f"({exc}); HiCache cannot hold this model's state") from exc
        return (DevicePoolInfo(pool_name=PoolName.MAMBA, indices_from_pool=PoolName.MAMBA, layer_ids=layer_ids,
                               buffer_info=info),)

    def _slot_regions(self):
        from sglang_neuron.mem_cache.state_host_pool import SlotRegions

        regions = getattr(self, "_regions", None)
        if regions is None:
            regions = self._regions = SlotRegions([self.mamba_cache.temporal, *self.mamba_cache.conv])
        return regions

    def get_cpu_copy(self, indices):
        """Upstream's retraction backup (``Req.offload_kv_cache``): slots ``indices`` of every component and layer on
        the host, as upstream's ``(conv list, temporal)`` with each ``[layers, n, *shape]``. Synchronous, like
        upstream's: the queued graphs finish first and the bytes have landed on return."""
        from sglang_neuron.mem_cache import state_host_pool

        slots = self._slots(_flat(indices), "host backup")
        regions = self._slot_regions()
        host = state_host_pool.host_tensors_like(regions, len(slots), pin=self.device.type != "cpu")
        self._check()  # a read leaves the slots as they were: a failure raises without quarantining the pool
        state_host_pool.copy_slots_now(regions, host, slots, range(len(slots)), True)
        return host[1:], host[0]

    def load_cpu_copy(self, mamba_cache_cpu, indices):
        """Upstream's retraction restore (``Req.load_kv_cache``): ``get_cpu_copy``'s tuple into slots ``indices`` (the
        request's slot, which may differ from the one backed up), bit for bit."""
        from sglang_neuron.mem_cache import state_host_pool

        if len(mamba_cache_cpu) != 2:
            raise ValueError(f"{self.spec_stage.family} state pool: a host copy is (conv list, temporal); got "
                             f"{len(mamba_cache_cpu)} parts (side slot states are refused at construction)")
        conv_cpu, temporal_cpu = mamba_cache_cpu
        slots = self._slots(_flat(indices), "host restore")
        if len(set(slots)) != len(slots):
            raise ValueError("state host restore destinations must be unique")
        regions = self._slot_regions()
        host = [temporal_cpu, *conv_cpu]
        if len(host) != len(regions.views):
            raise ValueError(f"{self.spec_stage.family} state pool: a host copy of {len(host)} components for a pool "
                             f"of {len(regions.views)}")
        regions.check_host(host, len(slots))
        self._run(state_host_pool.copy_slots_now, regions, host, slots, range(len(slots)), False)

    # -- working bank -------------------------------------------------------------------------------------------
    def load_rows(self, slots, rows) -> None:
        """Canonical slots -> working rows, before a forward of a working-bank model (slot 0 loads zero state)."""
        slots, rows = self._slots(slots, "load", zero=True), self._rows(rows)
        if slots:
            self._run(self.ops.load_rows, self.tensors, slots, rows)

    def publish_rows(self, slots, rows) -> None:
        """Working rows -> canonical slots, after the forward that advanced them was accepted."""
        slots, rows = self._slots(slots, "publish"), self._rows(rows)
        if len(set(slots)) != len(slots):
            raise ValueError("state publication destinations must be unique")
        if slots:
            self._run(self.ops.publish_rows, self.tensors, slots, rows)

    def capture_tracks(self, slots, track_slots, prefix_lens, extend_lens, track_seqlens, track_mask) -> list:
        """After an accepted extend forward: upstream's tracked checkpoints that land on a forward's end become copies
        of the request's slot into its tracking slot (upstream's extra buffer; a checkpoint the radix tree may later
        adopt). Returns the (request index, boundary) captures inside a forward, which only the model's kernel can
        have written; one this kernel cannot capture is refused (state_checkpoint.capture_boundary)."""
        from sglang_neuron.mem_cache.state_checkpoint import forward_end_tracks

        tracks = forward_end_tracks(prefix_lens, extend_lens, track_seqlens, track_mask, self.spec_stage.track_chunk)
        slots, track_slots = self._host(slots), self._host(track_slots)
        final = [i for i, _, at_end in tracks if at_end]
        if final:
            self.copy_from([slots[i] for i in final], [track_slots[i] for i in final])
        return [(i, boundary) for i, boundary, at_end in tracks if not at_end]

    # -- completion ---------------------------------------------------------------------------------------------
    def drain(self) -> None:
        """Wait for every submitted slot operation; called before slots change owner (StateStorage.clear through
        sglang_patches/state_pool_class, the PD decode table's clear through sglang_patches/decode_state_drain)."""
        self._check()
        try:
            self.ops.drain()
        except BaseException as exc:
            self._failed = exc
            raise

    @property
    def quarantined(self) -> bool:
        return self._failed is not None

    # -- helpers ------------------------------------------------------------------------------------------------
    def _check(self):
        if self._failed is not None:
            raise StatePoolQuarantined(f"{self.spec_stage.family} state pool is quarantined after a failed slot "
                                       f"operation ({type(self._failed).__name__}: {self._failed})") from self._failed

    def _run(self, fn, *args):
        self._check()
        from sglang_neuron.mem_cache import host_pool

        # a slot operation's compiled program rewrites the whole state tensor, and a HiCache load in flight DMAs into
        # other slots of it outside the device's stream order: the operation waits for those loads
        host_pool.wait_for_loads()
        try:
            fn(*args)
        except BaseException as exc:
            self._failed = exc
            raise

    @staticmethod
    def _host(values) -> list:
        if isinstance(values, torch.Tensor):
            if values.ndim != 1 or values.dtype not in (torch.int32, torch.int64):
                raise ValueError("state slot ids must be a 1-D integer tensor")
            values = values.tolist() if values.device.type == "cpu" else values.cpu().tolist()
        return [_int(v) for v in values]

    def _slots(self, values, what: str, zero: bool = False) -> list:
        slots = self._host(values)
        low = 0 if zero else 1
        bad = [s for s in slots if not low <= s <= self.size]
        if bad:
            raise ValueError(f"state {what}: slots {bad} are outside [{low}, {self.size}]")
        return slots

    def _rows(self, values) -> list:
        rows_total = self.spec_stage.working_rows
        if rows_total is None:
            raise ValueError(f"{self.spec_stage.family} state pool has no working bank")
        rows = self._host(values)
        if len(set(rows)) != len(rows) or any(not 1 <= r < rows_total for r in rows):
            raise ValueError(f"working rows {rows} repeat or are outside [1, {rows_total})")
        return rows


def _flat(values):
    """Upstream passes a request's slot as ``mamba_pool_idx.unsqueeze(0)``; any shape of slot ids, flattened."""
    return values.reshape(-1) if isinstance(values, torch.Tensor) else values


# -- the model's pool ---------------------------------------------------------------------------------------------
def state_pool_of(req_to_token_pool) -> Optional["NeuronStatePool"]:
    """The Neuron state pool SGLang built into a request table, or None (no state, or SGLang's own MambaPool)."""
    pool = getattr(req_to_token_pool, "mamba_pool", None)
    return pool if isinstance(pool, NeuronStatePool) else None


def check_model_state(contract, spec, req_to_token_pool) -> None:
    """After SGLang built the pools: the state pool SGLang built is a NeuronStatePool bound to the model's
    ``state_spec``. ``contract`` is the model's derived contract (graph-runner build: its recurrent layers must also be
    exactly the spec's) or, for a model that hosts its own runtime, its architecture name. A model that declared no
    state pool class (``PoolLayout.state_pool_class`` / ``NeuronModel.state_pool_class``) gets SGLang's MambaPool in
    upstream's layout on the host, which its kernels cannot read: refused here, naming the state layers (rule 2)."""
    if isinstance(contract, str):
        architecture = contract
    else:
        contract.check_state_spec(spec)
        architecture = contract.architecture
    pool = getattr(req_to_token_pool, "mamba_pool", None)
    if spec is None:
        if pool is not None:
            raise ValueError(f"{architecture}: SGLang built a recurrent-state pool and the model declares no state spec")
        return
    if not isinstance(pool, NeuronStatePool):
        raise ValueError(f"{architecture}: layers {list(spec.layer_ids)} carry {spec.family} state and SGLang built "
                         f"{type(pool).__name__ if pool is not None else 'no state pool'}: the model must declare its "
                         "pool class (PoolLayout.state_pool_class or NeuronModel.state_pool_class = "
                         "NeuronStatePool.bind(spec, ops))")
    if type(pool).spec.digest() != spec.digest():
        raise ValueError(f"{architecture}: the model declares a {spec.family} state spec and the state pool SGLang "
                         f"built is bound to another ({type(pool).spec.family}, layers "
                         f"{list(type(pool).spec.layer_ids)})")


def layout_parts(spec: StateSpec, ops=None, *, name: Optional[str] = None, base: Optional[type] = None):
    """A model's state in its megakernel PoolLayout (megakernel/pools.py): the bound pool class
    (``PoolLayout.state_pool_class``, which sglang_patches/state_pool_class hands SGLang's request table) and the
    pool entries its kernel binds: each component's whole physical tensor ``[layers, slots + 2, *shape]`` (padding slot,
    slots, sink; the kernel's padded buckets write the sink) and, for a working-bank kernel, each ``work.<component>``.
    Kinds: ``recurrent_state`` for the recurrent component, ``conv_state`` for a conv one; layer ids are global.
    ``base``: the model's ``NeuronStatePool`` subclass to bind (NeuronStatePool itself when None)."""
    from sglang_neuron.megakernel.pools import PoolEntry

    base = NeuronStatePool if base is None else base
    if not issubclass(base, NeuronStatePool):
        raise TypeError(f"a state pool layout binds a NeuronStatePool subclass, not {base.__name__}")
    cls = base.bind(spec, ops, name=name)
    names = [c.name for c in spec.components]
    if spec.working_rows is not None:
        names += ["work." + n for n in names]
    roles = {c.name: c.role for c in spec.components}
    entries = []
    for tensor_name in names:
        role = roles[tensor_name.removeprefix("work.")]

        def resolve(kv_pool, req_to_token_pool, _name=tensor_name):
            pool = state_pool_of(req_to_token_pool)
            if pool is None:
                raise RuntimeError("the request table holds no Neuron state pool (PoolLayout.state_pool_class)")
            return [pool.tensors[_name]]

        entries.append(PoolEntry(name=f"state.{tensor_name}", kind="recurrent_state" if role == "recurrent"
                                 else "conv_state", layers=spec.layer_ids, resolve=resolve, buffers=1,
                                 per_buffer=len(spec.layer_ids)))
    return cls, tuple(entries)
