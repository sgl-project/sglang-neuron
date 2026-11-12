# SPDX-License-Identifier: Apache-2.0
"""KV pool of one GLM-5.2 prefill stage rank, in the layout the prefill giga kernel reads and writes in place.

The kernel shards a request's context over the stage's ``cp`` ranks in runs of 32 tokens: token t belongs to rank
(t // 32) % cp. SGLang sees pages of ``cp * rows`` tokens (1024 at CP32), and its page p is block p on every rank: rank
r keeps tokens [32r, 32r + 32) of the page in its rows [rows * p, rows * (p + 1)). So SGLang's own paged allocator hands
out the pages unchanged. Page 0 is the null page that padding tokens write and no request reads. The pool has no
``dcp`` attribute, so the platform's allocator runs over SGLang's own token count.

Per rank and stage:

* ``kv_stacks[j]`` ``[L_j, n_slots, 320]`` bf16: launch j's MLA latent rows. Each layer's cache, the region a PD
  transfer registers, is a view of its stack, so the kernel, the graph and NIXL share one copy of the rows.
* ``idx_stacks[j]`` ``[F_j, n_slots, 128]`` bf16: the indexer keys of launch j's full layers, scored and written in
  place by the kernel. The 128k decode reads bf16 keys, so there these views are the registered key regions.
* ``fp8_keys[g]`` ``[num_pages, 1, rows, 128]`` uint8 per full layer: the keys the 1M decode reads, written by the
  graph after each launch and registered in place of the bf16 views. The graph binds them as ``[n_slots, 64]`` bf16
  words: an ``index_put_`` of uint8 rows lowers to a scatter with a whole-cache temporary, of bf16 words it does
  not.
* on the last stage, the MTP layer's latent rows and keys in the target layers' formats. They are registered only for
  a decode that drafts with MTP (``pd.placement.PD_MTP_ENV``): upstream's prepared send refuses a source entry that no
  decode entry pairs with.

The PD regions follow the graph's ``get_kv_spec`` order, every MLA cache in layer order and then every key cache, each
with its global layer id, so SGLang pairs every entry with the decode's entry of the same layer and kind.

The caches the graph binds are created once and never replaced, so the executor's binding cache holds, and
``graph_written_tensors`` names them for the warm-up's in-place check: a cache written through a copy-back would cost a
whole-cache device copy every step.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from sglang.srt.mem_cache.memory_pool import GB, KVCache

from sglang_neuron.mem_cache.memory_pool import NeuronMLATokenToKVPool
from sglang_neuron.models.glm52 import points
from sglang_neuron.models.glm52.pd import placement as pd_placement
from sglang_neuron.models.glm52.layout import glm_kv_row

if TYPE_CHECKING:
    from sglang_neuron.models.glm52.prefill.stage_layout import StageLayout

logger = logging.getLogger(__name__)

# bf16 words of the kernel's static latent row: 64 bf16 k_pe pre-divided by the static scale, then 512 fp8 latent bytes
# in the decode kernel's channel order (layout.glm_kv_row)
ROW_WORDS = glm_kv_row.STATIC_ROW_BYTES // 2
# the indexer's key width (GLM-5.2 family index_head_dim): bf16 values in the kernel's stacks, fp8 bytes in the caches
KEY_BYTES = 128
# the MLA geometry the static row packs (kv_lora_rank fp8 latent + qk_rope_head_dim bf16 k_pe), which the checkpoint
# must have
KV_LORA_RANK, QK_ROPE_HEAD_DIM = 512, 64
assert KV_LORA_RANK + 2 * QK_ROPE_HEAD_DIM == glm_kv_row.STATIC_ROW_BYTES

# Upstream SGLang code this module mirrors (pins: sglang_patches/upstream_pins.json): MLATokenToKVPool.__init__'s
# attributes (its constructor would allocate token-major rows), KVCacheConfigurator.configure and its sizing helpers
_CONFIGURATOR = "sglang.srt.mem_cache.kv_cache_configurator"
_UPSTREAM_TARGETS = (
    ("sglang.srt.mem_cache.memory_pool", "MLATokenToKVPool.__init__"),
    (_CONFIGURATOR, "KVCacheConfigurator.configure"),
    (_CONFIGURATOR, "KVCacheConfigurator._profile_available_bytes"),
    (_CONFIGURATOR, "KVCacheConfigurator._apply_token_constraints"),
    (_CONFIGURATOR, "KVCacheConfigurator.resolve_max_num_reqs"),
    (_CONFIGURATOR, "KVCacheConfigurator._req_pool_rows"),
    (_CONFIGURATOR, "KVCacheConfigurator._build_req_to_token_pool"),
    (_CONFIGURATOR, "KVCacheConfigurator._build_token_to_kv_pool_allocator"),
)


def bytes_per_page(layout: "StageLayout", pd_mtp: bool = False) -> int:
    """Device bytes one page takes on each rank of the stage: ``rows`` rows of every MLA latent row (640 B), bf16
    indexer key (256 B) and fp8 key (128 B) the stage holds, MTP layer included (with ``pd_mtp`` its bf16 key copy too,
    which only a drafting decode at a bf16-key point reads).
    442,368 / 471,040 / 495,616 B on PP4 stages 0 / 1-2 / 3 at 32 rows (served pools of 12,289 or 14,336 pages:
    5.4-7.1 GB per rank)."""
    full = sum(len(group.full) for group in layout.groups)
    row = layout_layers(layout) * 2 * ROW_WORDS + full * (2 * KEY_BYTES + KEY_BYTES)
    if layout.mtp_rows:
        row += 2 * ROW_WORDS + KEY_BYTES + (2 * KEY_BYTES if pd_mtp else 0)
    return layout.kv_page_rows * row


def layout_layers(layout: "StageLayout") -> int:
    """The stage's decoder layers, which its launch groups cover exactly and in order."""
    edges = [(group.lo, group.hi) for group in layout.groups]
    n = layout.end_layer - layout.start_layer
    if not edges or edges[0][0] != 0 or edges[-1][1] != n or any(a[1] != b[0] for a, b in zip(edges, edges[1:])):
        raise ValueError(f"the launch groups {edges} do not cover the stage's {n} layers [{layout.start_layer}, "
                         f"{layout.end_layer}) in order")
    return n


class Glm52PrefillKVPool(NeuronMLATokenToKVPool):
    """The KV caches of one prefill stage rank, as SGLang's MLA pool kind.

    SGLang sees ``(num_pages - 1) * page`` tokens in pages of ``cp * rows`` and one ``kv_buffer`` per layer. The class
    is a ``NeuronMLATokenToKVPool`` so the platform's pool check and SGLang's PD path treat it as MLA; its rows are 320
    words, not upstream's 576, so the constructor sets the MLA attributes itself. The pool's own writes, host copies and
    every read addressed by SGLang slot are refused: this rank holds only its runs of each page, so no upstream reader
    would address these rows correctly."""

    # sglang_patches.pd_kv_layer_ids publishes get_kv_layer_ids for the PD registration
    publishes_kv_layer_ids = True
    # the GLM KV hand-off publishes it (pd.handoff.get_pd_kv_layout), and SGLang's layout check pairs it with a GLM decode
    pd_kv_layout = pd_placement.GLM52_KV_LAYOUT
    # outside the allocator's zero-page contract: the kernel reads only rows below each request's context and the null
    # page, and the platform MLA pool's page fill addresses whole SGLang pages, which this rank does not hold
    zero_policy = None

    def __init__(self, layout: "StageLayout", *, num_pages: int, device, fp8_key_regions: bool = False):
        if num_pages < 2:
            raise ValueError(f"a pool of {num_pages} pages has none to allocate: page 0 is reserved")
        layers = layout_layers(layout)
        self.layout = layout
        self.num_pages = int(num_pages)
        self.rows = int(layout.kv_page_rows)
        self.n_slots = self.num_pages * self.rows
        page = layout.cp * self.rows
        # KVCache's constructor by name, as NeuronMHATokenToKVPool does: MLATokenToKVPool's allocates token-major rows.
        # SGLang's device string on this platform is the host's; the rows live on ``device``
        KVCache.__init__(self, (self.num_pages - 1) * page, page, torch.bfloat16, layers, "cpu", False,
                         layout.start_layer, layout.end_layer)
        # the attributes MLATokenToKVPool.__init__ sets (its consumers: SGLang's MLA checks and PD path)
        self.kv_lora_rank, self.qk_rope_head_dim = KV_LORA_RANK, QK_ROPE_HEAD_DIM
        self.use_dsa = self.dcp_replicated = self.dsa_kv_cache_store_fp8 = False
        self.kv_cache_dim = ROW_WORDS
        self.model_layer_ids = tuple(range(self.start_layer, self.start_layer + self.layer_num))
        self.kv_scale_buffer = None
        self.kv_device = torch.device(device)
        # the MTP layer's rows as two PD regions, for a decode that drafts with them (pd.placement.PD_MTP_ENV)
        self.pd_mtp = bool(layout.mtp_rows and pd_placement.pd_mtp())
        # the key regions in the decode point's format: fp8 (the 1M point) or bf16 (the 128k point)
        self.fp8_key_regions = bool(fp8_key_regions)
        self._allocate(layout)
        names = self.region_names()
        expected = list(layout.kv_layer_names(mtp=self.pd_mtp))
        if names != expected:
            raise ValueError(f"the pool's PD regions {names} are not the stage layout's KV layers {expected}")
        self._finalize_allocation_log(self.size)
        logger.info("GLM-5.2 prefill KV pool, stage layers [%d, %d)%s: %d pages of %d rows per rank (SGLang page %d), "
                    "%d launch stacks, %d fp8 key caches, %d B per page, %.2f GB",
                    self.start_layer, self.end_layer, " + MTP" if layout.mtp_rows else "", self.num_pages, self.rows,
                    page, len(self.kv_stacks), len(self.fp8_keys), bytes_per_page(layout, self.mtp_key_pages is not None),
                    self.get_kv_size_bytes() / GB)

    def _allocate(self, layout: "StageLayout") -> None:
        zeros = self._zeros
        stacks, idx_stacks, launch_keys = [], [], []
        self._latent_pages: Dict[int, torch.Tensor] = {}
        self._key_pages: Dict[int, torch.Tensor] = {}
        self._key_words: Dict[int, torch.Tensor] = {}
        self.fp8_keys: Dict[int, torch.Tensor] = {}
        for group in layout.groups:
            kv = zeros((group.hi - group.lo, self.n_slots, ROW_WORDS), torch.bfloat16)
            idx = zeros((len(group.full), self.n_slots, KEY_BYTES), torch.bfloat16)
            for k in range(group.hi - group.lo):
                self._latent_pages[self.start_layer + group.lo + k] = kv[k].view(self.num_pages, 1, self.rows,
                                                                                 ROW_WORDS)
            words = []
            for k, local in enumerate(group.full):
                g = self.start_layer + local
                self._key_pages[g] = idx[k].view(self.num_pages, 1, self.rows, KEY_BYTES)
                self.fp8_keys[g] = zeros((self.num_pages, 1, self.rows, KEY_BYTES), torch.uint8)
                self._key_words[g] = _words(self.fp8_keys[g])
                words.append(self._key_words[g])
            stacks.append(kv)
            idx_stacks.append(idx)
            launch_keys.append(tuple(words))
        self.kv_stacks: Tuple[torch.Tensor, ...] = tuple(stacks)
        self.idx_stacks: Tuple[torch.Tensor, ...] = tuple(idx_stacks)
        self._launch_keys = tuple(launch_keys)
        self.mtp_layer: Optional[int] = int(layout.mtp_layer) if layout.mtp_rows else None
        self.mtp_latent: Optional[torch.Tensor] = None
        self.mtp_keys: Optional[torch.Tensor] = None
        self._mtp_flat: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.mtp_key_pages: Optional[torch.Tensor] = None
        self._mtp_pd_keys: Optional[torch.Tensor] = None
        if self.mtp_layer is not None:
            self.mtp_latent = zeros((self.num_pages, 1, self.rows, ROW_WORDS), torch.bfloat16)
            self.mtp_keys = zeros((self.num_pages, 1, self.rows, KEY_BYTES), torch.uint8)
            self._latent_pages[self.mtp_layer] = self.mtp_latent
            self._key_words[self.mtp_layer] = _words(self.mtp_keys)
            self._mtp_flat = (self.mtp_latent.view(self.n_slots, ROW_WORDS), self._key_words[self.mtp_layer])
            if self.pd_mtp and not self.fp8_key_regions:
                # the bf16 keys of the MTP rows, the decode's draft indexer format at a bf16-key point (the target
                # layers' key regions); at an fp8-key point the draft reads the fp8 keys the rows already write
                self.mtp_key_pages = zeros((self.num_pages, 1, self.rows, KEY_BYTES), torch.bfloat16)
                self._mtp_pd_keys = self.mtp_key_pages.view(self.n_slots, KEY_BYTES)
        # SGLang's MLA kv_buffer: each stage layer's rows [n_slots, 1, row], a view of its launch stack
        self.kv_buffer = [self._latent_pages[g].view(self.n_slots, 1, ROW_WORDS) for g in self.model_layer_ids]

    def _zeros(self, shape, dtype) -> torch.Tensor:
        # straight on the device: a host staging copy would hold about 4-7 GB of host RAM per rank, 128 ranks to a host
        return torch.zeros(shape, dtype=dtype, device=self.kv_device)

    # each layer's caches by global layer id
    def latent_cache(self, global_layer: int) -> torch.Tensor:
        """The ``[num_pages, 1, rows, 320]`` bf16 latent cache of a stage layer or the MTP layer, a view of its
        storage: the per-layer cache, which the PD registration and the decode-side layout name."""
        return self._lookup(self._latent_pages, global_layer, "latent cache")

    def key_cache(self, global_layer: int) -> torch.Tensor:
        """The ``[num_pages, 1, rows, 128]`` bf16 key cache of a full stage layer, a view of its launch's idx stack: the
        key region the PD registration names at a bf16-key decode point (without ``fp8_key_regions``)."""
        return self._lookup(self._key_pages, global_layer, "bf16 key cache")

    def key_words(self, global_layer: int) -> torch.Tensor:
        """The fp8 key cache of a full layer or the MTP layer as the ``[n_slots, 64]`` bf16 words the graph binds and
        writes (one object per layer, created with the pool)."""
        return self._lookup(self._key_words, global_layer, "fp8 key cache")

    # what the graph binds
    def launch_caches(self, j: int) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Launch ``j``'s ``(kv_stack [L_j, n_slots, 320] bf16, idx_stack [F_j, n_slots, 128] bf16, the key words
        [n_slots, 64] bf16 of its full layers in layer order)``: the kernel's ``kv_caches`` and ``idx_k_caches``, and
        the caches the graph's fp8 key store writes after the launch."""
        return self.kv_stacks[j], self.idx_stacks[j], self._launch_keys[j]

    def mtp_caches(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """The MTP layer's ``(latent rows [n_slots, 320] bf16, key words [n_slots, 64] bf16)``, the flat caches the
        carry-path MTP rows write by slot (prefill.stage._write_mtp_rows); None off the stage that writes them."""
        return self._mtp_flat

    def mtp_pd_keys(self) -> Optional[torch.Tensor]:
        """The MTP rows' bf16 keys ``[n_slots, 128]``, the second MTP PD region's flat view, which the graph writes with
        the other MTP rows; None without pd_mtp."""
        return self._mtp_pd_keys

    def graph_written_tensors(self) -> Dict[str, torch.Tensor]:
        """Every object the stage graph binds and writes, by name: the launch stacks, the key words of each full layer
        and the MTP caches. The warm-up proves each is written in place by the compiled graph (model_executor
        graph_runner.check_in_place)."""
        out = {}
        for j, (kv, idx) in enumerate(zip(self.kv_stacks, self.idx_stacks)):
            out[f"kv_stack{j}"] = kv
            if idx.shape[0]:
                # a launch of shared indexer layers only (a PP16 stage that starts on one) has no key stack to write
                out[f"idx_stack{j}"] = idx
        for g, words in self._key_words.items():
            out[f"layers.{g}.self_attn.indexer"] = words
        if self._mtp_flat is not None:
            out[f"layers.{self.mtp_layer}.self_attn"] = self._mtp_flat[0]
        if self._mtp_pd_keys is not None:
            out[f"layers.{self.mtp_layer}.self_attn.indexer.bf16"] = self._mtp_pd_keys
        return out

    # the PD registration
    def region_names(self) -> List[str]:
        """The KV layer of every PD region, in registration order: the MLA caches of the stage's layers in global
        layer order, then the bf16 key caches of its full layers in layer order, with pd_mtp the MTP layer last in each
        kind (StageLayout.kv_layer_names, the stage graph's get_kv_spec)."""
        return ([f"layers.{g}.self_attn" for g in self._mla_layers()]
                + [f"layers.{g}.self_attn.indexer" for g in self._key_layers()])

    def _region_caches(self) -> List[torch.Tensor]:
        """The caches of :meth:`region_names`, in its order: the key caches in the decode point's format (the fp8 keys
        with ``fp8_key_regions``, else the bf16 views)."""
        latent = [self._latent_pages[g] for g in self._mla_layers()]
        keys = [(self.fp8_keys if self.fp8_key_regions else self._key_pages)[g] for g in sorted(self._key_pages)]
        mtp_keys = self.mtp_keys if self.fp8_key_regions else self.mtp_key_pages
        return latent + keys + ([mtp_keys] if self.pd_mtp else [])

    def get_contiguous_buf_infos(self):
        """(ptrs, lens, item_lens) of every region in :meth:`region_names` order: the cache's device address
        (``data_ptr()`` is a placeholder on torch-neuronx), its bytes, and the bytes of one page of it, ``rows`` rows,
        at ``ptr + p x item_len`` for page ``p``."""
        from sglang_neuron.backend.device_io import device_address

        caches = self._region_caches()
        ptrs = [device_address(t) for t in caches]
        lens = [t.numel() * t.element_size() for t in caches]
        return ptrs, lens, [n // self.num_pages for n in lens]

    def get_kv_layer_ids(self) -> List[int]:
        """The global layer of every region of :meth:`get_contiguous_buf_infos` (pd.placement.stage_entry_ids)."""
        return self._mla_layers() + self._key_layers()

    def get_kv_size_bytes(self) -> int:
        tensors = [*self.kv_stacks, *self.idx_stacks, *self.fp8_keys.values()]
        if self.mtp_latent is not None:
            tensors += [self.mtp_latent, self.mtp_keys]
        if self.mtp_key_pages is not None:
            tensors.append(self.mtp_key_pages)
        return sum(t.numel() * t.element_size() for t in tensors)

    def _mla_layers(self) -> List[int]:
        return list(self.model_layer_ids) + ([self.mtp_layer] if self.pd_mtp else [])

    def _key_layers(self) -> List[int]:
        return sorted(self._key_pages) + ([self.mtp_layer] if self.pd_mtp else [])

    def _lookup(self, table: dict, layer: int, what: str) -> torch.Tensor:
        try:
            return table[int(layer)]
        except KeyError:
            raise KeyError(f"layer {layer} has no {what} on the stage of layers [{self.start_layer}, "
                           f"{self.end_layer}){' + MTP ' + str(self.mtp_layer) if self.mtp_layer is not None else ''}: "
                           f"{sorted(table)}") from None

    # SGLang's slot-addressed reads and writes: SGLang's slot is a virtual-page slot, and this rank keeps only its runs
    # of each page
    def get_key_buffer(self, layer_id: int):
        _refuse_slot_access("get_key_buffer")

    def get_value_buffer(self, layer_id: int):
        _refuse_slot_access("get_value_buffer", "the static latent row has no value slice (its fp8 latent and bf16 "
                            "k_pe are packed in 320 words)")

    def get_kv_buffer(self, layer_id: int):
        _refuse_slot_access("get_kv_buffer", "the static latent row has no value slice")

    def get_mla_kv_buffer(self, *args, **kwargs):
        _refuse_slot_access("get_mla_kv_buffer")

    def set_mla_kv_buffer_prefix_valid(self, *args, **kwargs):
        raise RuntimeError("the MLA rows are written by the compiled graphs, not by the pool")

    def move_kv_cache(self, *args, **kwargs):
        _refuse_slot_access("move_kv_cache")

    def get_block_buffer(self, layer_id: int):
        _refuse_slot_access("get_block_buffer")

    def zero_pages(self, page_ids) -> None:
        _refuse_slot_access("zero_pages")


def _refuse_slot_access(name: str, why: str = "") -> None:
    raise NotImplementedError(f"Glm52PrefillKVPool.{name}: SGLang's MLA pool addresses its rows by token slot, and a "
                              "GLM-5.2 prefill rank holds only its 32-token runs of each page, in the kernel's "
                              f"layout{'; ' + why if why else ''}")


def _words(keys: torch.Tensor) -> torch.Tensor:
    """A ``[..., 128]`` uint8 key cache as its flat ``[n_slots, 64]`` bf16 words (the same bytes)."""
    return keys.view(-1, KEY_BYTES).view(torch.bfloat16)


def configure_kv_cache(configurator, layout: "StageLayout", *, pre_model_load_memory: int, cp_rank: int):
    """Build the stage's KV pools in place of ``KVCacheConfigurator.configure``: exactly the stated
    ``--max-total-tokens`` in whole pages, the platform's paged allocator and SGLang's request table.

    The page count is a dimension of every compiled graph, so every stage builds the same count instead of sizing its
    own from free memory. The stated pool must fit this rank's budget as SGLang computes it (the world's least free HBM
    after the weights, less ``--mem-fraction-static``). Every rank reaches SGLang's MIN sync of the token count before
    any refusal, so a stage short of HBM refuses the launch on every rank instead of leaving the others in a
    collective."""
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigResult
    from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig
    from sglang.srt.platforms import current_platform
    from sglang.srt.runtime_context import get_disagg, get_schedule

    from sglang_neuron.model_executor.runtime_init import neuron_device
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_pinned(_UPSTREAM_TARGETS, __name__)
    schedule = get_schedule()
    page = layout.cp * layout.kv_page_rows
    _check_launch(configurator, layout, page, schedule)
    stated = int(schedule.max_total_tokens)
    num_pages = stated // page + 1
    # the launch's decode point with its indexer pool (both roles read SGLANG_NEURON_GLM52_POINT): the key regions take
    # its key format, and only a bf16-key point's drafting decode needs the MTP rows' bf16 key copy
    geo = points.current().decode_geometry()
    per_page = bytes_per_page(layout, bool(layout.mtp_rows and pd_placement.pd_mtp() and not geo.idx_fp8_keys))
    budget = int(configurator._profile_available_bytes(pre_model_load_memory))
    # SGLang's MIN sync of the tokens each stage's budget holds (its whole pages past the reserved one). It returns at
    # most the stated count, and the plugin's hook on it refuses a smaller one on every rank after the same sync
    configurator._apply_token_constraints(max(budget // per_page - 1, 0) * page)
    # this rank's pages on the GK decode's slots: it is the stage's kernel CP rank ``cp_rank``, the GK decode ``geo``
    pool = Glm52PrefillKVPool(layout, num_pages=num_pages, device=neuron_device(), fp8_key_regions=geo.idx_fp8_keys)
    pd_placement.publish_prefill_layout(pool, geo, cp_rank)
    # the paged branch of KVCacheConfigurator._build_token_to_kv_pool_allocator: SGLang's own token count and page (no
    # DCP widening: the pool's page is already the stage's virtual page)
    allocator = current_platform.get_paged_allocator_cls()(
        stated, page_size=page, dtype=configurator.kv_cache_dtype, device=configurator.device, kvcache=pool,
        need_sort=get_disagg().disaggregation_mode in ("decode", "prefill"))
    max_running_requests = configurator.resolve_max_num_reqs(stated)
    req_to_token_pool = configurator._build_req_to_token_pool(
        max_num_reqs=configurator._req_pool_rows(max_running_requests))
    config = MemoryPoolConfig(max_total_num_tokens=stated, max_running_requests=max_running_requests,
                              mem_fraction_static=schedule.mem_fraction_static)
    return KVCacheConfigResult(max_total_num_tokens=stated, max_running_requests=max_running_requests,
                               full_max_total_num_tokens=None, swa_max_total_num_tokens=None,
                               req_to_token_pool=req_to_token_pool, token_to_kv_pool=pool,
                               token_to_kv_pool_allocator=allocator, memory_pool_config=config)


def _check_launch(configurator, layout: "StageLayout", page: int, schedule) -> None:
    """Refuse a launch whose KV settings the stage pool cannot serve, naming the setting: the server's geometry only
    (one SGLang page = the stage's virtual page, ``--page-size``), a stated ``--max-total-tokens`` in whole pages, the
    bf16 pool dtype (the kernel's row is bf16 words; its fp8 is the row's own format), SGLang's layer range of the stage
    and the checkpoint's MLA and indexer widths, which the kernel's rows are."""
    problems = []
    if schedule.page_size != page:
        problems.append(f"--page-size {schedule.page_size}: the stage's KV page is its CP degree ({layout.cp}) x "
                        f"{layout.kv_page_rows} rows = {page} tokens, the virtual page {layout.virtual_page} of the "
                        f"server's geometry; launch with --page-size {layout.virtual_page}")
    stated = schedule.max_total_tokens
    if stated is None or int(stated) <= 0 or int(stated) % page:
        problems.append(f"--max-total-tokens {stated}: state the KV pool's size, a positive multiple of the "
                        f"{page}-token page, since its page count is a dimension of every compiled graph")
    if configurator.kv_cache_dtype != torch.bfloat16:
        problems.append(f"--kv-cache-dtype {configurator.kv_cache_dtype}: the pool stores the kernel's fp8 static rows "
                        "as bf16 words; launch with --kv-cache-dtype auto (or bf16)")
    info = configurator.layer_info
    if (int(info.start_layer), int(info.end_layer)) != (layout.start_layer, layout.end_layer):
        problems.append(f"SGLang's stage holds layers [{info.start_layer}, {info.end_layer}), the stage layout "
                        f"[{layout.start_layer}, {layout.end_layer}): set SGLANG_PP_LAYER_PARTITION to the layout's "
                        "split")
    hf = configurator.model_config.hf_config
    widths = {"kv_lora_rank": KV_LORA_RANK, "qk_rope_head_dim": QK_ROPE_HEAD_DIM, "index_head_dim": KEY_BYTES}
    for name, want in widths.items():
        if getattr(hf, name, None) != want:
            problems.append(f"the checkpoint's {name} is {getattr(hf, name, None)}, and the kernel's KV rows hold "
                            f"{want}")
    if problems:
        raise ValueError("the GLM-5.2 prefill stage's KV pool cannot be built:\n  - " + "\n  - ".join(problems))
