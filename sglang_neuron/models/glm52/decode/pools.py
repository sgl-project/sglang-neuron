# SPDX-License-Identifier: Apache-2.0
"""The GLM-5.2-family GK decode's KV: the giga kernel's own caches hold it, SGLang's pool only counts it.

The GK keeps each sequence's KV in caches of its own, rank-private and CP-sharded (models/glm52/layout/decode/gk_kv_cache.py):
one sequence slot per token group, a static identity block table, a fixed depth of ``context`` positions per slot. It
reads no SGLang page. SGLang still schedules by its token allocator, so the decode server keeps SGLang's request and
token pools, sized for the kernel's slots at its context, over a KV pool that holds one layer of a minimal head: the
allocator's accounting, not the KV (``configure_kv_cache``). A request keeps its kernel slot for its whole life:
``slot = req_pool_idx - 1``, SGLang's request row (row 0 is its padding row), so the request pool has exactly the
kernel's slots, and none beyond them: SGLang's decode would otherwise pre-allocate rows for requests still in transfer
(``--disaggregation-decode-extra-slots``), but a request holds its kernel slot from its transfer on.

The PD decode registers the GK's own caches as its NIXL KV regions, so the prefill's WRITEs land in them
(``Glm52DecodeAccountingPool``, ``gk_kv_regions``): one region a latent layer, then one a full layer's indexer keys
(the global ids ``pd.placement.decode_entry_ids``, by which SGLang pairs them with each prefill stage's), each of the
rank's blocks (``geo.num_blocks`` items of one 32-row block: 1,024 at 128k, 8,192 at 1M), and the pool publishes the GLM layout tag
(``pd.placement.GLM52_KV_LAYOUT``), which only a GLM prefill matches. The accounting pool's own buffers are not
registered: no peer writes them.

The GK's weights come from its weight store (each rank's stacks in the kernel's layout, ``layout.weight_loaders_mxfp4.
build_rank_store``), built once from the checkpoint for the geometry, at ``SGLANG_NEURON_GLM52_WEIGHT_STORE``; the
value ``zeros`` runs the kernel over zero weights of the store's shapes (its ``zero_rank_weights``), a decode whose
device time is the kernel's and whose tokens mean nothing, for a bring-up before a store exists."""

from __future__ import annotations

import functools
import logging
import os

logger = logging.getLogger(__name__)

# the accounting pool's head: one layer of one KV head of this width; the kernel never reads it
_ACCOUNTING_HEAD_DIM = 8
WEIGHT_STORE_ENV = "SGLANG_NEURON_GLM52_WEIGHT_STORE"
ZERO_WEIGHTS = "zeros"


def weight_store(geo) -> str:
    """The GK weight store of ``geo`` (``SGLANG_NEURON_GLM52_WEIGHT_STORE``)."""
    path = os.environ.get(WEIGHT_STORE_ENV, "").strip()
    if not path:
        raise ValueError(f"the GLM-5.2-family GK decode loads its weights from the GK weight store built for its "
                         f"geometry (cp {geo.cp_degree}, t_shard {geo.t_shard}, context {geo.context}); set "
                         f"{WEIGHT_STORE_ENV} to it")
    if path != ZERO_WEIGHTS and not os.path.isdir(path):
        raise ValueError(f"{WEIGHT_STORE_ENV}={path}: no such weight store")
    return path


def extra_rows_problem(extra_slots) -> str | None:
    """Why a launch with ``extra_slots`` decode request rows beyond the kernel's slots is not served, or None."""
    extra = int(extra_slots or 0)
    if not extra:
        return None
    return (f"{extra} decode request rows beyond the kernel's slots for requests in transfer (a request holds its "
            "kernel slot from its transfer on; launch with --disaggregation-decode-extra-slots 0)")


def accounting_kv_spec(config):
    """The graph's KV spec: no layer reads SGLang's pool."""
    from sglang_neuron.mem_cache.kv_cache import KVSpec

    return KVSpec(layers=[])


def gk_kv_regions(caches, geo, address_of, mtp_caches=None):
    """(ptrs, lens, item_lens) of the decode's PD KV regions over the GK's caches (the module docstring), in
    ``pd.placement.decode_entry_ids`` order: every latent layer of each latent stack, then every full layer's indexer
    keys, then with ``mtp_caches`` (the async MTP draft's ``(latent, keys)``, one layer each) the MTP layer's latent and
    keys. ``address_of(tensor)`` is a tensor's device address (``device_io.device_address``)."""
    from sglang_neuron.models.glm52.pd import placement as pd_placement

    latent_item, key_item = pd_placement.entry_bytes(geo)
    ptrs, lens, items = [], [], []
    stacks = [(stack, latent_item) for stack in caches.latent_stacks] + [(caches.idx_k_cache_all, key_item)]
    if mtp_caches is not None:
        latent, keys = mtp_caches
        # one layer each, as a stack of one
        stacks += [(latent.unsqueeze(0), latent_item), (keys.unsqueeze(0), key_item)]
    for stack, item in stacks:
        base, layer_bytes = int(address_of(stack)), stack.stride(0) * stack.element_size()
        if layer_bytes % item or not stack.is_contiguous():
            raise ValueError(f"a GK cache {tuple(stack.shape)} {stack.dtype} is not whole contiguous {item}-byte "
                             "blocks a layer")
        for layer in range(stack.shape[0]):
            ptrs.append(base + layer * layer_bytes)
            lens.append(layer_bytes)
            items.append(item)
    ids = pd_placement.decode_entry_ids(geo, mtp=mtp_caches is not None)
    if len(ptrs) != len(ids):
        raise ValueError(f"the GK's caches hold {len(ptrs)} layer regions; the decode registers {len(ids)}")
    if len({n // i for n, i in zip(lens, items)}) != 1:
        raise ValueError("the GK's latent and key caches hold different block counts; a prepared NIXL list needs one")
    return ptrs, lens, items


@functools.lru_cache(maxsize=None)
def accounting_pool_cls():
    """The decode's accounting pool class: the platform's MHA pool kind, publishing the GK's caches as its PD
    regions (one class a process: SGLang and the PD path check the pool's type)."""
    from sglang_neuron.mem_cache.memory_pool import NeuronMHATokenToKVPoolKind
    from sglang_neuron.models.glm52.pd import placement as pd_placement

    class Glm52DecodeAccountingPool(NeuronMHATokenToKVPoolKind):
        """SGLang's token accounting over one minimal layer, whose PD registration is the GK's caches
        (``gk_kv_regions``), their global layer ids and the GLM layout tag."""

        # sglang_patches.pd_kv_layer_ids publishes get_kv_layer_ids; the GLM KV hand-off names pd_kv_layout
        publishes_kv_layer_ids = True
        pd_kv_layout = pd_placement.GLM52_KV_LAYOUT
        # (caches, geo, mtp_caches) once bind_gk has run
        _gk = None

        def _create_buffers(self) -> None:
            """One block of the one layer's K and V, on the host: the pool only counts tokens (its PD regions are the
            GK's caches, get_contiguous_buf_infos, and nothing reads or writes its buffers), so a [blocks, ...] buffer
            pair would hold the slots' whole context in HBM for nothing (1.07 GB a rank at 32 x 1M)."""
            import torch

            k_shape, shape = self.get_kv_buffer_shape()
            self.k_buffer = [torch.zeros((1, *k_shape[1:]), dtype=self.store_dtype, device="cpu")]
            self.v_buffer = [torch.zeros((1, *shape[1:]), dtype=self.store_dtype, device="cpu")]

        def bind_gk(self, caches, geo, rank: int, mtp_caches=None) -> None:
            """Register the GK's ``caches`` of GK rank ``rank`` (and an async MTP draft's ``mtp_caches``) as this
            pool's PD regions, and publish the rank's placement for the PD hand-off (pd.placement.publish_decode_layout,
            which models/glm52/pd/handoff carries to the KV manager)."""
            self._gk = (caches, geo, mtp_caches)
            pd_placement.publish_decode_layout(self, geo, rank)

        def get_contiguous_buf_infos(self):
            from sglang_neuron.backend.device_io import device_address

            gk = self._gk
            if gk is None:
                raise RuntimeError("the GLM GK decode registers the kernel's own caches for its PD KV transfer, and "
                                   "none are bound to the pool (decode_pools.configure_kv_cache binds the graph's)")
            return gk_kv_regions(gk[0], gk[1], device_address, gk[2])

        def get_kv_layer_ids(self):
            return pd_placement.decode_entry_ids(self._gk[1], mtp=self._gk[2] is not None)

    return Glm52DecodeAccountingPool


class GkSlotFence:
    """A finished request's kernel slot goes back to SGLang's request pool only once every decode step that may still
    write it has finished on the device.

    A finished request stays in the steps already launched (the result lag's in-flight bound), and each of them writes
    the request's next row into its slot's caches. The slot's next owner takes its prompt KV over NIXL into those
    caches, which the device does not order behind those steps: a stale step landing after the WRITE would overwrite a
    row of the new prompt. So a freed row is stamped with this thread's launch count (``device_io.launches``) and is
    handed out again only once ``device_io.retired_launches``, a lower bound, reaches it: the platform's own rule for
    a disaggregated decode's KV pages (mem_cache/allocator.merge_and_sort_free), here for the GK's slots, which hold the
    KV in place of SGLang's pages. A request is then held in SGLang's prealloc queue (``available_size`` counts only
    the rows it may take) until the steps that may write its slot have retired; no destination is published before,
    so no WRITE can race them. The pool waits for its launches only when nothing else will retire them: no request of
    the newest launched step still runs (every decision reads scheduler state and launch counts, which every rank
    holds alike).

    It wraps the request pool's ``free``, ``alloc``, ``available_size`` and ``clear`` on the instance. The rows stay in
    ``free_slots`` while fenced, so SGLang's own count of its rows (the invariant checker's) is unchanged."""

    def __init__(self, req_pool, rank=None):
        self.pool = req_pool
        # GK rank 0 logs each request's slot, which the slot-dependent rounding of a greedy decode follows
        self.logs_slots = rank == 0
        self.stamps = {}  # fenced row -> this thread's launch count at its free
        self.live = frozenset()  # the kernel slots of the newest launched step
        self._free, self._alloc, self._clear = req_pool.free, req_pool.alloc, req_pool.clear
        req_pool.free, req_pool.alloc = self.free, self.alloc
        req_pool.available_size, req_pool.clear = self.available_size, self.clear

    def note_step(self, slots) -> None:
        """The kernel slots of the step about to launch (models/glm52/decode/runtime.py)."""
        self.live = frozenset(int(s) for s in slots)

    def free(self, req) -> None:
        from sglang_neuron.backend import device_io

        row = int(req.kv.req_pool_idx)
        self._free(req)
        self.stamps[row] = device_io.launches()

    def _safe(self) -> list:
        """The free rows a request may take: the unfenced ones, after those whose launches have retired leave the
        fence; when none is and no request of the newest step still runs, after waiting for every launch."""
        from sglang_neuron.backend import device_io

        retired = device_io.retired_launches()
        self.stamps = {row: stamp for row, stamp in self.stamps.items() if stamp > retired}
        safe = [row for row in self.pool.free_slots if row not in self.stamps]
        if not safe and self.stamps:
            held = set(range(1, self.pool.size + 1)) - set(self.pool.free_slots)
            if not any(slot + 1 in held for slot in self.live):
                device_io.wait_launches()
                self.stamps.clear()
                safe = list(self.pool.free_slots)
        return safe

    def available_size(self) -> int:
        return len(self._safe())

    def alloc(self, reqs):
        safe = self._safe()
        need = sum(1 for r in reqs if not r.kv.holds_kv)
        if need > len(safe):
            return None
        # the safe rows first, so the pool's own alloc takes only them
        self.pool.free_slots = safe + [row for row in self.pool.free_slots if row in self.stamps]
        rows = self._alloc(reqs)
        if rows is not None and self.logs_slots:
            for req, row in zip(reqs, rows):
                logger.info("[glm52] request %s takes GK slot %d", req.rid, row - 1)
        return rows

    def clear(self) -> None:
        from sglang_neuron.backend import device_io

        device_io.wait_launches()
        self.stamps.clear()
        self._clear()


_SLOT_FENCE = {"fence": None}


def slot_fence():
    """This process's GK slot fence (configure_kv_cache installs it), or None."""
    return _SLOT_FENCE["fence"]


def pool_tokens(geo, page_size: int) -> int:
    """Tokens of SGLang's pool: every kernel slot at the kernel's context, in whole pages."""
    tokens = geo.dp_size * geo.num_seqs * geo.context
    return tokens // page_size * page_size


def configure_kv_cache(configurator, geo, *, pre_model_load_memory: int, caches, rank: int, mtp_caches=None):
    """SGLang's request pool of the kernel's slots and token pool at their context, over the accounting pool and the
    platform's allocator; the pool holds one layer of one ``_ACCOUNTING_HEAD_DIM``-wide head, and registers GK rank
    ``rank``'s ``caches`` (the graph's, which its weight load allocates before SGLang builds the pools) and an async MTP
    draft's ``mtp_caches`` as its PD regions (the module docstring)."""
    import torch
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigResult
    from sglang.srt.platforms import current_platform
    from sglang.srt.runtime_context import get_schedule

    if caches is None:
        raise RuntimeError("the GLM GK decode registers the kernel's own caches as its PD KV regions, and the graph "
                           "holds none: its weights load (which allocates them) before SGLang builds the pools")
    schedule = get_schedule()
    page = int(schedule.page_size)
    slots = geo.dp_size * geo.num_seqs
    tokens = pool_tokens(geo, page)
    stated = schedule.max_total_tokens
    if stated is not None and int(stated) != tokens:
        raise ValueError(f"--max-total-tokens {stated}: the GLM GK decode's pool is its {slots} slots at its "
                         f"{geo.context}-token context, {tokens} tokens; launch with --max-total-tokens {tokens}")
    running = schedule.max_running_requests
    if running is not None and int(running) != slots:
        raise ValueError(f"--max-running-requests {running}: the GLM GK decode scores {slots} sequence slots; launch "
                         f"with --max-running-requests {slots}")
    req_pool = configurator._build_req_to_token_pool(max_num_reqs=slots)
    rows = int(req_pool.size) + int(req_pool.pre_alloc_size)
    if rows != slots:
        raise ValueError(f"SGLang sized the request pool at {rows} rows for {slots} kernel slots; the kernel's slots "
                         "are the request rows")
    _SLOT_FENCE["fence"] = GkSlotFence(req_pool, rank)
    pool = accounting_pool_cls()(
        tokens, page_size=page, dtype=torch.bfloat16, head_num=1, head_dim=_ACCOUNTING_HEAD_DIM, layer_num=1,
        device=configurator.device, enable_memory_saver=False, start_layer=0, end_layer=0)
    pool.bind_gk(caches, geo, rank, mtp_caches)
    allocator = current_platform.get_paged_allocator_cls()(
        tokens, page_size=page, dtype=torch.bfloat16, device=configurator.device, kvcache=pool, need_sort=True)
    return KVCacheConfigResult(
        max_total_num_tokens=tokens, max_running_requests=slots, full_max_total_num_tokens=None,
        swa_max_total_num_tokens=None, req_to_token_pool=req_pool, token_to_kv_pool=pool,
        token_to_kv_pool_allocator=allocator, memory_pool_config=None)


def slot_of_request_row(req_pool_idx: int, slots: int) -> int:
    """The kernel slot of SGLang request row ``req_pool_idx`` (row 0 is SGLang's padding row)."""
    slot = int(req_pool_idx) - 1
    if not 0 <= slot < slots:
        raise ValueError(f"request row {req_pool_idx} has no kernel slot among {slots}")
    return slot
