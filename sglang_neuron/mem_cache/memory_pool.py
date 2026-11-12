# SPDX-License-Identifier: Apache-2.0
"""Neuron KV cache pool in the block layout the NKI attention kernels consume.

SGLang addresses KV by *token slot* (``req_to_token[req, pos] -> slot``) with
``slot // page_size`` = physical page. The Neuron decode/prefill kernels take a
block table and a 4-D per-layer cache ``[num_blocks, kv_heads, block_size, head_dim]``.
With ``page_size == block_size`` an SGLang page *is* a kernel block, so:

  * this pool allocates K and V per layer directly in the kernel layout on the
    Neuron device (device tensors; SGLang's bookkeeping stays on CPU);
  * ``get_key_buffer/get_value_buffer`` return the 4-D block tensors (the
    plugin's attention path and the model consume them; SGLang core never reads
    KV contents itself);
  * ``get_contiguous_buf_infos`` reports page-granular contiguous regions in
    SGLang's fixed order (k_0..k_{L-1}, v_0..v_{L-1}) for PD transfer — a page
    is contiguous in this layout (kv_heads*block_size*head_dim elements);
  * ``get_kv_layer_ids`` gives each region's global layer, so a pipelined
    stage's entries pair with the peer's entries of the same layers.

Row/page 0 is SGLang's reserved padding page (never allocated); the pool holds
``size + page_size`` slots == ``size/page_size + 1`` blocks.

A model that declares token context parallelism (``model_executor.kv_layout.KVLayout`` with ``cp_size`` C above 1)
gets every KV head on every rank of its attention-TP group and, of each SGLang page of P tokens, one P / C-token block:
rank r holds the page's tokens [r x P / C, (r + 1) x P / C). The pool publishes ``kv_cp_size``, ``kv_cp_rank`` and
``kv_cp_interleave``; a model that declares nothing (GPT-OSS) keeps one whole-page block per rank and ``kv_cp_size`` 1.

A model that declares paired layers (``model_executor.kv_layout``, the GPT-OSS giga-kernel decode) gets layers 2j and
2j + 1 as the two halves of one K tensor and one V tensor of twice the blocks (``layer_pairs``); each layer's buffer is
the view of its half, so every per-layer path above, the PD registration among them, is unchanged.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch

from sglang.srt.mem_cache.memory_pool import GB, KVCache, MHATokenToKVPool, MLATokenToKVPool

logger = logging.getLogger(__name__)


class NeuronMHATokenToKVPool(KVCache):
    # Each page stores its heads contiguously ([blocks, kv_heads, block, head_dim]; packed fp8 K
    # [blocks, kv_heads, block // 2, head_dim, 2]), so a KV transfer between different attention-TP
    # sizes slices a page by head range (sglang_patches.nixl_head_major reads this flag).
    kv_head_major = True
    # the token context-parallel layout the pool holds (set in __init__ from model_executor.kv_layout): none by default
    kv_cp_size = 1
    kv_cp_rank = 0
    # the PD registration names each entry's global layer (get_kv_layer_ids; sglang_patches.pd_kv_layer_ids)
    publishes_kv_layer_ids = True

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        **kwargs,
    ):
        # KVCache's constructor by name: on upstream SGLang the pool kind's next class is MHATokenToKVPool
        # (NeuronMHATokenToKVPoolKind), whose constructor allocates token-major buffers
        KVCache.__init__(self, size, page_size, dtype, layer_num, device, False, start_layer, end_layer)
        # SGLang stores fp8 caches as uint8 and reinterprets on access; the Neuron kernels take
        # the raw buffers, so allocate them in the real float8 dtype (same bytes, zero-copy for
        # SGLang's byte-level paths such as the disaggregation transfer).
        if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            self.store_dtype = dtype
        assert size % page_size == 0, (size, page_size)
        _refuse_other_head_dims(head_dim, kwargs)
        from sglang_neuron.model_executor import kv_layout

        layout = kv_layout.current()
        # token context parallelism (kv_layout.KVLayout.cp_size): this rank holds every KV head for its page / cp_size
        # tokens of each page. The pool publishes the layout it holds (kv_cp_size, kv_cp_rank, kv_cp_interleave), and
        # the runtime refuses a pool whose layout differs from the model's (NeuronModelRuntime.__init__)
        self.kv_cp_size = int(layout.cp_size)
        self.kv_cp_rank = int(layout.cp_rank)
        self.head_num = layout.kv_heads_per_rank or head_num
        self.head_dim = head_dim
        # SGLang's MHATokenToKVPool exposes these (the unified radix cache and the disaggregation read them)
        self.k_head_dim = head_dim
        self.v_head_dim = head_dim
        # one kernel block per SGLang page: the whole page, or this rank's page / cp_size tokens of it
        self.block_size = layout.physical_block_size(page_size)
        self.kv_cp_interleave = self.block_size
        # DCP: an allocator page is a virtual page with one block on each DCP rank, and this rank keeps its block of
        # virtual page i in block i, so the pool's shape is unchanged (distributed.decode_context_parallel).
        # dcp_interleave is the run of consecutive tokens one rank keeps, which a PD transfer follows: a block (1
        # without DCP).
        from sglang_neuron.distributed.decode_context_parallel import runtime_layout

        self.dcp = runtime_layout(page_size)
        if self.dcp is not None and self.kv_cp_size > 1:
            raise ValueError("decode context parallelism over a context-parallel KV layout is not supported")
        self.dcp_interleave = page_size if self.dcp is not None else 1
        # +1 block: SGLang's reserved page 0 (slots [0, page_size))
        self.num_blocks = size // page_size + 1
        from sglang_neuron.model_executor.runtime_init import neuron_device

        self.kv_device = neuron_device()
        self.k_buffer: List[torch.Tensor] = []
        self.v_buffer: List[torch.Tensor] = []
        self.paired_layers = layout.paired_layers
        # the model that declares the pairing refuses an odd layer count first (GptOssModel)
        assert not (self.paired_layers and self.layer_num % 2), self.layer_num
        if self.paired_layers and self.kv_cp_size > 1:
            raise ValueError("a paired-layer KV pool holds whole pages per rank; it runs no token context parallelism")
        # the per-pair K and V tensors of a paired pool (pair j: layer 2j's blocks, then layer 2j + 1's), else None
        self.layer_pairs: Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]] = None
        self._create_buffers()
        # SGLang's metrics and /get_load report the pool's size from mem_usage, as its own pools set it (in GB);
        # a kind answering upstream's MHA contract (K bytes, V bytes) reports their sum
        size = self.get_kv_size_bytes()
        self.mem_usage = (sum(size) if isinstance(size, tuple) else size) / GB
        logger.info(
            "Neuron KV pool: layers=%d blocks=%d page=%d block=%d kv_cp=%d kv_heads=%d head_dim=%d dtype=%s (%.2f GB)",
            layer_num, self.num_blocks, page_size, self.block_size, self.kv_cp_size, self.head_num, head_dim,
            self.store_dtype, self.mem_usage,
        )

    # alloc
    @property
    def fp8_packed_k(self) -> bool:
        """fp8 caches use the kernels' packed K layout [nb, kv_heads, block//2, d, 2] (two consecutive
        tokens per bf16-sized slot; V stays [nb, kv_heads, block, d]). The nkilib qkv / segmented /
        decode kernels only support fp8 KV in this layout."""
        return self.store_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)

    def _create_buffers(self):
        k_shape, shape = self.get_kv_buffer_shape()
        if self.paired_layers:
            nb = self.num_blocks
            k_pairs = [torch.zeros((2 * nb, *k_shape[1:]), dtype=self.store_dtype, device=self.kv_device)
                       for _ in range(self.layer_num // 2)]
            v_pairs = [torch.zeros((2 * nb, *shape[1:]), dtype=self.store_dtype, device=self.kv_device)
                       for _ in range(self.layer_num // 2)]
            self.k_buffer = [half for pair in k_pairs for half in (pair[:nb], pair[nb:])]
            self.v_buffer = [half for pair in v_pairs for half in (pair[:nb], pair[nb:])]
            self.layer_pairs = (k_pairs, v_pairs)
            return
        for _ in range(self.layer_num):
            self.k_buffer.append(torch.zeros(k_shape, dtype=self.store_dtype, device=self.kv_device))
            self.v_buffer.append(torch.zeros(shape, dtype=self.store_dtype, device=self.kv_device))

    def _clear_buffers(self):
        del self.k_buffer, self.v_buffer
        self.k_buffer, self.v_buffer, self.layer_pairs = [], [], None

    # access
    def _idx(self, layer_id: int) -> int:
        return layer_id - self.start_layer

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self.k_buffer[self._idx(layer_id)]

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        return self.v_buffer[self._idx(layer_id)]

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def get_kv_layer_ids(self) -> List[int]:
        """The global layer of every entry ``get_contiguous_buf_infos`` registers, K entries then V. A pipelined stage
        holds only its own layers, and SGLang's PD transfer pairs entries by layer (sglang_patches.pd_kv_layer_ids)."""
        layers = list(range(self.start_layer, self.start_layer + self.layer_num))
        return layers + layers

    def set_kv_buffer(self, *args, **kwargs):
        raise RuntimeError("the K/V pages are written by the compiled graphs, not by the pool")

    # bookkeeping
    def get_kv_size_bytes(self) -> int:
        per = self.num_blocks * self.head_num * self.block_size * self.head_dim * self.store_dtype.itemsize
        return per * 2 * self.layer_num

    def get_kv_buffer_shape(self):
        """Per-layer K and V buffer shapes."""
        shape = (self.num_blocks, self.head_num, self.block_size, self.head_dim)
        k_shape = ((self.num_blocks, self.head_num, self.block_size // 2, self.head_dim, 2) if self.fp8_packed_k
                   else shape)
        return torch.Size(k_shape), torch.Size(shape)

    def get_contiguous_buf_infos(self):
        """(ptrs, lens, item_lens): item_len = bytes per page; page p at ptr + p*item_len."""
        item_len = self.head_num * self.block_size * self.head_dim * self.store_dtype.itemsize
        from sglang_neuron.backend.device_io import device_address

        ptrs = [device_address(t) for t in self.k_buffer] + [device_address(t) for t in self.v_buffer]
        lens = [t.numel() * t.element_size() for t in self.k_buffer] + [
            t.numel() * t.element_size() for t in self.v_buffer
        ]
        return ptrs, lens, [item_len] * (2 * self.layer_num)

    # SGLang's KVCache signatures, for its allocators' host copies (Req.offload_kv_cache); a PD decode keeps no
    # retraction backup ("none", disaggregation.apply_launch_settings), so serving reaches neither
    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        raise NotImplementedError("KV host offload is not supported by sglang-neuron")

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None):
        raise NotImplementedError("KV host offload is not supported by sglang-neuron")


class NeuronMHATokenToKVPoolKind(NeuronMHATokenToKVPool, MHATokenToKVPool):
    """The Neuron pool registered as SGLang's MHA pool kind (``get_kv_pool_cls(kind="mha")``), which SGLang requires
    to subclass ``MHATokenToKVPool``.

    Only the type comes from ``MHATokenToKVPool``. Its constructor never runs, and any of its methods the Neuron pool
    doesn't override raises (``_refuse_mha_methods``), since those address SGLang's token-major buffers. Pool options
    this pool doesn't implement are refused."""

    def __init__(self, *args, **kwargs):
        _refuse_unsupported_pool_options(kwargs)
        NeuronMHATokenToKVPool.__init__(self, *args, **kwargs)

    @property
    def post_capture_backed_bytes(self) -> int:
        return 0  # sized before graph capture: SGLang's post-capture sizing never resizes it

    def get_device_pool_infos(self):
        """The pool's K and V tensors for HiCache's host pool (mem_cache/host_pool.py, which SGLang's assembler picks
        for these buffers): per layer [blocks, ...] in the kernels' layout, page p at row p. There are no pointer
        tables; the Neuron host pool moves whole rows. A token context-parallel pool is refused: each rank's row is
        page / kv_cp_size tokens, which SGLang's host pool, sized by the page, does not address."""
        self._refuse_cp_hicache()
        return self._mha_device_pool_infos()

    def _refuse_cp_hicache(self):
        """The HiCache refusal of a token context-parallel pool: a subclass that serves HiCache under CP calls this
        itself when its opt-in is off, and skips it when on (its host pool is mem_cache/host_pool.py's CP one)."""
        if self.kv_cp_size > 1:
            raise ValueError(f"HiCache over a token context-parallel KV pool (kv_cp_size {self.kv_cp_size}) is not "
                             "supported: each rank holds page / kv_cp_size tokens of a page")

    def _mha_device_pool_infos(self):
        """The MHABufferInfo of get_device_pool_infos, without the CP refusal."""
        from sglang.srt.mem_cache.device_pool_info import DevicePoolInfo, KVBufferLayout, KVBuffers, MHABufferInfo
        from sglang.srt.mem_cache.hicache_storage import PoolName

        buffers = KVBuffers(k=tuple(self.k_buffer), v=tuple(self.v_buffer), k_data_ptrs=None, v_data_ptrs=None,
                            layout=KVBufferLayout.HND)
        info = MHABufferInfo(buffers=buffers, page_size=self.page_size, head_num=self.head_num,
                             head_dim=self.head_dim, v_head_dim=self.head_dim, dtype=self.store_dtype,
                             device=self.k_buffer[0].device)
        return (DevicePoolInfo(pool_name=PoolName.KV, indices_from_pool=PoolName.KV,
                               layer_ids=tuple(range(self.start_layer, self.start_layer + self.layer_num)),
                               buffer_info=info),)


# MHATokenToKVPool keyword arguments whose features this pool does not have, with the value that means "off"
_POOL_OPTIONS_OFF = {"post_capture_active": False, "quant_method": None, "kv_cache_layout": None,
                     "enable_kv_cache_copy": False}


def _refuse_other_head_dims(head_dim: int, kwargs: dict) -> None:
    """Refuse a V or sliding-window head dim other than ``head_dim``: the pool's K and V pages, on every layer, are
    [kv_heads, page, head_dim] alike."""
    other = {k: kwargs[k] for k in ("v_head_dim", "swa_head_dim", "swa_v_head_dim")
             if kwargs.get(k) not in (None, head_dim)}
    if other:
        raise ValueError(f"the Neuron KV pool holds K and V of one head dim ({head_dim}) on every layer, not "
                         f"{', '.join(f'{k}={v}' for k, v in other.items())}")


def _refuse_unsupported_pool_options(kwargs: dict, options_off=None) -> None:
    """Refuse a pool option the Neuron pool does not implement: for the MHA pool, post-capture sizing, a KV quantization
    method, another cache layout and KV copies for speculative decoding; alternate streams have no effect on Neuron, and
    the platform refuses --enable-memory-saver at launch. The pool refuses head dims other than its own itself
    (_refuse_other_head_dims)."""
    options_off = _POOL_OPTIONS_OFF if options_off is None else options_off
    on = sorted(k for k, off in options_off.items() if k in kwargs and kwargs[k] not in (off, None, False))
    if on:
        raise ValueError(f"the Neuron KV pool does not support {', '.join(f'{k}={kwargs[k]!r}' for k in on)}")


def _refuse_mha_methods(cls, own, base) -> None:
    """On ``cls``, replace each method of ``base`` that ``own`` (the Neuron pool) doesn't define, and that ``KVCache``
    has no generic version of, with one that raises: it would address ``base``'s buffer layout, which the Neuron pool
    doesn't have."""
    generic = {name for name, value in vars(KVCache).items() if callable(value)}
    for name, value in list(vars(base).items()):
        if (name.startswith("__") or not callable(value) or name in vars(cls) or name in vars(own)
                or name in generic):
            continue

        def refuse(self, *args, _name=name, **kwargs):
            raise NotImplementedError(f"{type(self).__name__}.{_name}: SGLang's MHATokenToKVPool method addresses its "
                                      "token-major buffers, which the Neuron block-layout pool does not have")

        refuse.__name__ = name
        setattr(cls, name, refuse)


_refuse_mha_methods(NeuronMHATokenToKVPoolKind, NeuronMHATokenToKVPool, MHATokenToKVPool)


class NeuronMLATokenToKVPool(MLATokenToKVPool):
    """SGLang's MLA pool (the platform's ``get_kv_pool_cls(kind="mla")``) with its buffers on this rank's NeuronCore:
    per layer [slots + page, 1, kv_lora_rank + qk_rope_head_dim], one compact row per token slot, slot 0's page
    reserved, which the MLA kernels read by token slot. The compiled graphs write the rows, and a PD transfer moves
    whole pages of them; the pool's own writes, host copies and the pool options this layout does not have are
    refused.

    - FP8 (``--kv-cache-dtype fp8_e4m3``): stored as uint8, as upstream stores it (``store_dtype``); kernels take a
      layer's rows as blocks ``[pages + 1, page, kv_lora_rank + qk_rope_head_dim]`` of the stored bytes
      (``get_block_buffer``) and read them as ``dtype``.
    - DCP: this rank's placement (``dcp``, distributed.decode_context_parallel.DcpLayout), as on the MHA pool: block i
      holds this rank's tokens of the allocator's virtual page i, so the shape is unchanged, the attention metadata
      localizes slots, and the allocator checks SGLang's widened size.
    - A model that declares ``KVLayout.zero_fresh_pages`` puts the pool under mem_cache.allocator's zero-page contract
      with ``zero_policy`` on_alloc: every page is zeroed (``zero_pages``) as the allocator hands it out. Otherwise the
      pool declares no policy and is outside the contract. Page 0 is zero when the pool is built, and stays zero as
      long as no write targets slot 0.
    - PD: a standalone pool names each entry's global layer (``get_kv_layer_ids``); inside SGLang's hybrid linear
      pool, the hybrid pool names them."""

    # the PD registration names each entry's global layer (get_kv_layer_ids; sglang_patches.pd_kv_layer_ids)
    publishes_kv_layer_ids = True

    def __init__(self, *args, **kwargs):
        _refuse_unsupported_pool_options(kwargs, _MLA_POOL_OPTIONS_OFF)
        super().__init__(*args, **kwargs)
        from sglang_neuron.distributed.decode_context_parallel import runtime_layout
        from sglang_neuron.model_executor import kv_layout

        self.dcp = runtime_layout(self.page_size)
        self.dcp_interleave = self.page_size if self.dcp is not None else 1
        self.zero_fresh_pages = kv_layout.current().zero_fresh_pages
        # mem_cache.allocator's zero-page contract: zeroed as handed out, or outside it (None)
        from sglang_neuron.mem_cache.allocator import ZERO_ON_ALLOC

        self.zero_policy = ZERO_ON_ALLOC if self.zero_fresh_pages else None
        if self.zero_fresh_pages:
            # before any page is handed out
            self.warm_zero_pages()

    def _create_buffers(self):
        from sglang_neuron.model_executor.runtime_init import neuron_device

        device = neuron_device()
        self.kv_buffer = [torch.zeros((self.size + self.page_size, 1, self.kv_cache_dim), dtype=self.store_dtype,
                                      device=device) for _ in range(self.layer_num)]

    @property
    def num_blocks(self) -> int:
        """Pages of the pool on this rank, the reserved page 0 included."""
        return self.size // self.page_size + 1

    def get_block_buffer(self, layer_id: int) -> torch.Tensor:
        """Layer ``layer_id``'s rows (the pool's own layer id: global for a standalone pool, the full-attention index
        inside SGLang's hybrid pool) as blocks ``[num_blocks, page, kv_cache_dim]`` of ``store_dtype``, a view."""
        if self.size % self.page_size:
            raise ValueError(f"an MLA pool of {self.size} tokens is not whole {self.page_size}-token pages, so its rows "
                             "are no block view")
        return self.kv_buffer[layer_id - self.start_layer].view(self.num_blocks, self.page_size, self.kv_cache_dim)

    def get_kv_layer_ids(self) -> List[int]:
        """The global layer of every entry ``get_contiguous_buf_infos`` registers (one per layer), so a pipelined
        stage's entries pair with the peer's of the same layers (sglang_patches.pd_kv_layer_ids)."""
        return list(self.model_layer_ids)

    def _fill_view(self, layer: int) -> torch.Tensor:
        """Layer ``layer``'s pages, one row of bytes each, as a 2-byte dtype where the page's bytes allow: zero bytes
        read zero in any dtype, and the fill then runs as a bf16 pool's does."""
        rows = self.kv_buffer[layer]
        pages = rows.shape[0] // self.page_size
        flat = rows[:pages * self.page_size].view(pages, -1).view(torch.uint8)
        return flat.view(torch.bfloat16) if flat.shape[1] % 2 == 0 else flat

    def warm_zero_pages(self) -> None:
        """Compile the fill of ``zero_pages`` on this rank now, before serving: zero the reserved page 0, which is
        zero already. Every layer buffer has the fill's one shape. A pool that zeroes its fresh pages warms the fill as
        it is built. A caller that zeroes pages on demand calls it before serving (a PD decode zeroes the tail blocks
        of each new request, models/kimi_k3/decode/spec_worker.py), so that no fill compiles at the first request."""
        if self.layer_num:
            fill_pages(self._fill_view(0), [0])

    def zero_pages(self, page_ids) -> None:
        """Zero pages ``page_ids`` of this rank's blocks on every layer (page 0 included when named; ids past the pool
        are refused). Fills of one shape per layer buffer (fill_pages), queued on the device before the launches that
        follow, without a host wait for the graph in flight."""
        ids = sorted({int(p) for p in (page_ids.tolist() if hasattr(page_ids, "tolist") else page_ids)})
        if not ids:
            return
        if ids[0] < 0 or ids[-1] >= self.num_blocks:
            raise ValueError(f"pages {ids} are outside this MLA pool's {self.num_blocks} pages")
        for layer in range(self.layer_num):
            fill_pages(self._fill_view(layer), ids)

    def nonzero_reserved_layers(self) -> List[int]:
        """The layers whose reserved page 0 holds a nonzero byte (a diagnostic: it reads the page back)."""
        return [self.start_layer + i for i in range(self.layer_num)
                if bool(self._fill_view(i)[0].ne(0).any().cpu())]

    def get_contiguous_buf_infos(self):
        """(ptrs, lens, item_lens) of each layer's rows, the regions a PD transfer registers with NIXL: the device
        address (upstream's data_ptr() is a placeholder on torch-neuronx), and per page its rows [p x page, (p + 1) x
        page), one run at ptr + p x item_len, as upstream's MLA sender indexes them."""
        from sglang_neuron.backend.device_io import device_address

        ptrs = [device_address(t) for t in self.kv_buffer]
        lens = [t.numel() * t.element_size() for t in self.kv_buffer]
        return ptrs, lens, [t[0].numel() * t.element_size() * self.page_size for t in self.kv_buffer]

    def set_kv_buffer(self, *args, **kwargs):
        raise RuntimeError("the MLA rows are written by the compiled graphs, not by the pool")

    def set_mla_kv_buffer(self, *args, **kwargs):
        raise RuntimeError("the MLA rows are written by the compiled graphs, not by the pool")

    def get_cpu_copy(self, *args, **kwargs):
        raise NotImplementedError("KV host offload is not supported by sglang-neuron")

    def load_cpu_copy(self, *args, **kwargs):
        raise NotImplementedError("KV host offload is not supported by sglang-neuron")

    def get_device_pool_infos(self):
        raise NotImplementedError(f"{type(self).__name__}.get_device_pool_infos: host KV pools (HiCache, host-pool "
                                  "retraction backup) are not supported by sglang-neuron")


# MLATokenToKVPool keyword arguments whose features the Neuron MLA pool does not have, with the value that means "off"
_MLA_POOL_OPTIONS_OFF = {"use_dsa": False, "override_kv_cache_dim": None, "dcp_replicated": False}

# page ids per fill: a fill compiles once per input shape, so every fill of a buffer takes one
ZERO_CHUNK = 64


def _on_neuron(pages: torch.Tensor) -> bool:
    return pages.device.type == "neuron"


def fill_pages(pages: torch.Tensor, ids, chunk: int = ZERO_CHUNK) -> int:
    """Zero rows ``ids`` of ``pages`` (one row per page) in fills of exactly ``chunk`` ids, the last padded by
    repeating its last id (a fill is idempotent); returns the number of fills.

    On a NeuronCore each fill is a compiled in-place graph (mla_pool._row_fill) whose ids go up on the side stream
    (device_io.upload_many). torch-neuronx starts an operation on another execution unit than the operation before it
    only after that operation has retired, and it waits on the host for that. An eager fill, or an id copy on the
    graphs' stream, thus holds the scheduler until the forward in flight ends, and the next forward starts late (a
    Kimi-K3 prefill stage zeroes the fresh pages of every chunk, so every forward). A compiled fill runs on the same
    unit as the forwards, and its submit waits on the device for its ids. Elsewhere (the host) an eager
    ``index_fill_``."""
    ids = [int(i) for i in ids]
    fills = 0
    for start in range(0, len(ids), chunk):
        part = ids[start:start + chunk]
        part += [part[-1]] * (chunk - len(part))
        rows = torch.tensor(part, dtype=torch.long)
        if _on_neuron(pages):
            from sglang_neuron.backend import device_io
            from sglang_neuron.mem_cache import mla_pool

            (index,) = device_io.upload_many([rows], pages.device)
            mla_pool._row_fill(pages, chunk)(pages, index)
        else:
            pages.index_fill_(0, rows.to(pages.device), 0)
        fills += 1
    return fills


def full_attention_pool(pool):
    """The pool that holds a KV pool's full-attention layers: SGLang's hybrid linear-attention pool's leaf
    (``full_kv_pool``), else the pool itself."""
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

    return pool.full_kv_pool if isinstance(pool, HybridLinearKVPool) else pool

