# SPDX-License-Identifier: Apache-2.0
"""Neuron MLA latent and DSA index-K pools, in the paged token-major layout the MLA / DSA kernels consume.

An MLA model (``kv_lora_rank`` set: Kimi-K3, GLM-5.2) caches one latent vector per token, ``kv_lora_rank``
compressed KV plus ``qk_rope_head_dim`` RoPE key (576 = 512 + 64; 512 for a NoPE latent, ``qk_rope_head_dim`` 0),
shared by every head. A DSA model also caches index keys (``index_head_dim`` = 128) for its lightning indexer: one per
token, or with KPool (``index_kpool`` k > 1, ``index_kpool_compress``) one learned compression of every k consecutive
tokens' keys, written when the k-token pool completes. The pools hold both per layer in one tensor each, in the
kernels' layout:

* latent ``[layers, blocks, block, kv_lora_rank + qk_rope_head_dim]``;
* index keys ``[layers, blocks, index_block / index_kpool, index_head_dim]``, bf16 (upstream's FP8-with-scale index rows
  are a CUDA layout);
* with KPool, the request tails ``[layers, max_running_requests + 1, index_kpool + tail_extra_slots, index_head_dim]``,
  once for the raw index keys and once for the compression gate scores of the tokens past a request's last complete
  pool: upstream's per-request ring (the token at position ``p`` in slot ``p mod (index_kpool + tail_extra_slots)``,
  disaggregation/utils.get_dsa_tail_state_indices), held by every CP rank.

Block ``b`` is SGLang's page ``b`` on this rank, so a block table is SGLang's page ids (block 0 = SGLang's reserved page,
never allocated) and a P/D transfer moves whole blocks. Under context parallelism (mem_cache/cp_pages.CpPages: decode
context parallelism or a declared token-CP layout) one native page is one block per CP rank, so a rank's pool holds
``block = native page / cp_size`` tokens of every page. The index keys are either sharded the same way
(``index_sharding="owner"``, the default: each rank scores its own tokens, and the selection merges across the
group) or replicated whole on every rank (``"replicated"``, as upstream does: each rank selects over the full
context locally; ``index_block`` = the native page).

The zero-page contract. Kernels gather whole blocks and mask positions past a sequence's length in the scores; the
masked keys still enter the value matmul with probability 0, and ``0 * Inf = NaN``. So the pools guarantee:

1. every block is zero when the pool is built, and block 0 is never written: write slots of padding rows and of tokens
   another CP rank owns are ``cp_pages.DROP_SLOT`` (skipped), never a slot of block 0, and block-table padding reads
   block 0 (all zeros);
2. a freed page is zeroed before it is handed out again (``zero_pages``, called by the Neuron paged allocators' release
   path, mem_cache/allocator.py), so the tail of a reused page never carries a previous request's bytes, finite or not.
   That is the pool's ``zero_policy`` "on_free"; a pool may declare "on_alloc" instead (its pages zeroed as the
   allocator hands them out, refused on a PD decode), and either fill first waits for the HiCache loads in flight
   (host_pool.wait_for_loads) unless the pool declares ``zero_fill_load_safe`` (mem_cache/allocator.py).

This contract is what keeps the outputs finite, not the operands: an MLA decode kernel that reads whole blocks turns
a NaN anywhere in a listed block, past the row's position bound, into a NaN output (measured on trn3), and a decode
whose token opens a new block lists that whole block (attention/mla_operands.decode_operands covers the prior tokens
and the decoded one). Finite stale bytes are masked exactly.

The zeroing runs on the scheduler thread at every free, so it is one compiled in-place fill per buffer and chunk over
every layer at once (``fill_pages``: the buffer's flat view, the input aliased to the output, no transient copy), not
an eager op per layer: a GLM-5.2 8K request's free is 4 fills, not 312 eager ones, in less time, on trn3. A compiled
graph exists per input shape, and one first compiled while serving would hold the ranks of the replica that freed a
request of a new length while every other rank waits in the next step's collectives, which the runtime aborts after
its 30 s timeout. So ``fill_pages`` fills in a fixed ladder of sizes (``ZERO_LADDER``: 1, 8, 16 and 64 ids, a remainder
padded by repeating an id up to the next size: the fill is idempotent), and each pool compiles every size once when it
builds its buffers (``warm_fill``), on every rank at the same point. One 64-id shape would make every free, of one page
or one request row, rewrite 64 ids' worth of the buffer (GLM-5.2 on trn3: one 32K request row, 12 GB, much device
time).

A model declares its tensors (``buffers()``, a ``PoolBuffer`` each: the latent, the index keys, any MTP latent, request
rows or tails); the pool allocates them, applies the zero-page contract to every paged one (``page_dim``) and counts
them all, so a new model's pool lists its buffers instead of re-implementing that life cycle.

The pools are model-written: the compiled graphs' kernels write latent and index keys through the operands of
attention/mla_operands.py; ``set_kv_buffer`` and SGLang's other token-major writers refuse.

Host copies move whole pages of every paged buffer, each rank its own block of each page (a native page id is the
block id on every CP rank, so no index translation): HiCache L2 through the host pools of mem_cache/mla_host_pool.py,
which SGLang's assembler builds from ``get_device_pool_infos``, and SGLang's retraction backup through ``get_cpu_copy``
/ ``load_cpu_copy``, which also carry the request's row of every request-row buffer (``row_dim``: the KPool tails).
"""

from __future__ import annotations

import itertools
import logging
import types
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

# page ids per compiled fill: a fill takes exactly one of these many ids (a ladder of compiled shapes per pool buffer,
# the module docstring); the largest is the chunk a long free is split into
ZERO_LADDER = (1, 8, 16, 64)
ZERO_CHUNK = ZERO_LADDER[-1]


def _fill_rows(flat: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    flat.index_fill_(0, rows, 0)
    return flat


# one compiled in-place fill per (flat shape, dtype, device, index length), each over its own copy of the body's code
# (_distinct)
_COMPILED_FILLS: dict = {}
_DISTINCT = itertools.count()


def _distinct(fn):
    """``fn`` under a code object of its own. Dynamo keys its guard cache and its recompile limit (8) on the code
    object, not on the torch.compile wrapper, so the fills of several buffer shapes over one body would share one
    cache, and past 8 shapes a ``fullgraph`` compile raises FailOnRecompileLimitHit (measured on trn3 with the state
    pools' compiled slot operations)."""
    code = fn.__code__.replace(co_name=f"{fn.__name__}_{next(_DISTINCT)}")
    return types.FunctionType(code, fn.__globals__, code.co_name, fn.__defaults__, fn.__closure__)


def compile_fill(fn=None):
    """A compiled fill (on a NeuronCore): the fill body over its own code (_distinct), static, whole, under dynamo's
    tensor guards only. A pool warms its fills as it is built (warm_fill), outside SGLang's scheduler loop, which runs
    under DynamicGradMode (inference mode): with dynamo's global-state guards (grad mode) every warmed fill would compile
    again at its first call while serving, a new executor and NEFF at the first request. A fill takes only tensors (the
    buffer and its row ids), none of which requires grad, so grad mode changes nothing it computes."""
    from sglang_neuron.backend import stack

    return torch.compile(_distinct(fn or _fill_rows), backend=stack.compile_backend_name(), dynamic=False,
                         fullgraph=True, options={"guard_filter_fn": torch.compiler.keep_tensor_guards_unsafe})


def _row_fill(flat: torch.Tensor, rows: int):
    """The fill of ``rows`` flat rows of ``flat``: on a NeuronCore a compiled graph that writes the zeros in place
    (the input aliased to the output: no transient copy of the buffer, and one graph launch per fill instead of an
    eager op per layer), elsewhere (the host, the meta device) an eager ``index_fill_``."""
    if flat.device.type != "neuron":
        return _fill_rows
    key = (tuple(flat.shape), flat.dtype, str(flat.device), rows)
    fn = _COMPILED_FILLS.get(key)
    if fn is None:
        fn = _COMPILED_FILLS[key] = compile_fill()
    return fn


def fill_chunks(n: int, ladder=ZERO_LADDER) -> list:
    """The compiled fill sizes ``n`` ids are zeroed in: as many of the largest rung as fit, then the smallest rung that
    holds the rest (padded by repeating an id: the fill is idempotent)."""
    top = ladder[-1]
    sizes = [top] * (n // top)
    rest = n - top * len(sizes)
    if rest:
        sizes.append(next(c for c in ladder if c >= rest))
    return sizes


def fill_pages(buffer: torch.Tensor, dim: int, ids, ladder=ZERO_LADDER) -> int:
    """Zero ``buffer``'s slices ``ids`` along ``dim`` (the page dimension: every leading index, e.g. every layer of
    a buffer stacked over the layers, is zeroed at those pages) in fills of a ladder size each (fill_chunks: a fill
    writes what its ids name, so a short free does not rewrite a whole 64-id chunk); returns the number of fills.

    The buffer is filled through its flat view ``[lead * pages, rest]``, one compiled in-place fill of ``lead * size``
    rows per fill (_row_fill): on trn3, zeroing a GLM-5.2 request's 128 latent pages over 78 layers takes 2 compiled
    fills, in less scheduler time than 156 eager per-layer fills."""
    ids = [int(i) for i in ids]
    if not ids:
        return 0
    if not buffer.is_contiguous():
        raise ValueError("fill_pages zeroes a contiguous pool buffer through its flat view")
    lead = 1
    for size in buffer.shape[:dim]:
        lead *= int(size)
    pages = int(buffer.shape[dim])
    if not all(0 <= i < pages for i in ids):
        raise IndexError(f"page ids outside the buffer's {pages} pages")
    flat = buffer.view(lead * pages, -1)
    offsets = torch.arange(lead, dtype=torch.long).unsqueeze(1) * pages
    fills, i = 0, 0
    for size in fill_chunks(len(ids), ladder):
        part = ids[i:i + size]
        i += len(part)
        part += [part[-1]] * (size - len(part))
        rows = (offsets + torch.tensor(part, dtype=torch.long).unsqueeze(0)).reshape(-1)
        fill = _row_fill(flat, lead * size)
        index = _row_index(flat, lead * size)
        if index is None:
            fill(flat, rows)
        else:
            # the ids go into the compiled fill's one fixed device tensor: a fresh device tensor per fill gives each
            # execution a new input address, for which the runtime builds a new DMA ring out of HBM a server sized to
            # its pool does not keep free (trn3, GLM-5.2 under an abort storm: "Failed to allocate DEVICE memory" for
            # the ring, every rank's scheduler down). Fills run in submission order, so reusing it is safe.
            index.copy_(rows, non_blocking=True)
            fill(flat, index)
        fills += 1
    return fills


# one fixed device row-id tensor per compiled fill (_row_fill's key), reused by every fill of that shape
_ROW_INDEX: dict = {}


def _row_index(flat: torch.Tensor, rows: int):
    """The fixed device tensor of ``rows`` row ids a compiled fill of ``flat`` reads, or None off a NeuronCore (the
    eager fill takes the host ids)."""
    if flat.device.type != "neuron":
        return None
    key = (tuple(flat.shape), flat.dtype, str(flat.device), rows)
    index = _ROW_INDEX.get(key)
    if index is None:
        index = _ROW_INDEX[key] = torch.zeros(rows, dtype=torch.long, device=flat.device)
    return index


def warm_fill(buffer: torch.Tensor, dim: int, page: int = 1, ladder=ZERO_LADDER) -> None:
    """Compile every ladder shape of ``fill_pages`` for ``buffer`` by zeroing ``page`` (repeated to each size), which a
    freshly built pool holds as zeros already: no shape compiles while serving."""
    for size in ladder:
        fill_pages(buffer, dim, [page] * size, ladder)

from sglang.srt.mem_cache.memory_pool import KVCache

from sglang_neuron.mem_cache.cp_pages import CpPages

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PoolBuffer:
    """One tensor of an MLA / DSA pool, in the layout the model's kernels take (NeuronMLATokenToKVPool.buffers).

    ``name``: the pool attribute that holds it (``latent`` is required; ``index_keys`` is SGLang's DSA index-K
    buffer). ``shape`` / ``dtype`` (None: the pool's KV dtype). ``page_dim``: the dimension indexed by SGLang page id
    (this rank's block id), which puts the buffer under the zero-page contract: zeroed when built, its freed pages
    zeroed before reuse, block 0 never written; every index before ``page_dim`` (the layers of a stacked buffer) is
    zeroed at those pages. None: state the model addresses otherwise (request rows, tails), which the pool only
    allocates zero-filled and counts. ``row_dim``: for such state, the dimension indexed by SGLang's request-pool row
    (``req_pool_index``: the KPool tails), whose row of a request get_cpu_copy / load_cpu_copy carry with its pages
    (SGLang's retraction backup); None: state no host copy carries. ``row_width``: the indices of ``row_dim`` one row
    spans (row ``r`` is ``[r * row_width, (r + 1) * row_width)``: GLM-5.2's index keys hold a row's 128 features
    there). ``zero_rows``: the zero-row contract, a row's slice zeroed when SGLang hands the row to a new request
    (attach_row_zeroing), for state a kernel reads past the request's own positions."""

    name: str
    shape: Tuple[int, ...]
    page_dim: Optional[int] = None
    dtype: Optional[torch.dtype] = None
    row_dim: Optional[int] = None
    row_width: int = 1
    zero_rows: bool = False

    def __post_init__(self):
        for what, dim in (("page_dim", self.page_dim), ("row_dim", self.row_dim)):
            if dim is not None and not 0 <= dim < len(self.shape):
                raise ValueError(f"pool buffer {self.name}: {what} {dim} is not a dimension of {self.shape}")
        if self.page_dim is not None and self.row_dim is not None:
            raise ValueError(f"pool buffer {self.name}: a buffer is indexed by page or by request row, not both")
        if self.row_dim is None and (self.row_width != 1 or self.zero_rows):
            raise ValueError(f"pool buffer {self.name}: row_width and zero_rows describe request rows (row_dim)")
        if self.row_dim is not None and (self.row_width < 1 or self.shape[self.row_dim] % self.row_width):
            raise ValueError(f"pool buffer {self.name}: rows of {self.row_width} do not split dimension {self.row_dim} "
                             f"of {self.shape}")

    def row_view(self, tensor: torch.Tensor) -> torch.Tensor:
        """``tensor`` with ``row_dim`` as request rows: split into [rows, row_width] when a row spans several indices."""
        if self.row_width == 1:
            return tensor
        shape = list(tensor.shape)
        d = self.row_dim
        return tensor.view(*shape[:d], shape[d] // self.row_width, self.row_width, *shape[d + 1:])


# SGLang 43881a7's ReqToTokenPool.alloc_rows, which calls _on_alloc_rows with the rows it hands out: the hook the
# zero-row contract rides on (attach_row_zeroing), pinned so an upstream change cannot silently skip the zeroing
_ALLOC_ROWS_MD5 = {("sglang.srt.mem_cache.memory_pool", "ReqToTokenPool.alloc_rows"): "fd044aee8a73169da3a0d6d8f83d9118"}


def attach_row_zeroing(kv_pool, req_to_token_pool) -> bool:
    """Zero a pool's zero-row buffers (PoolBuffer.zero_rows) at every row SGLang allocates: chained onto the request
    pool's ``_on_alloc_rows`` (ReqToTokenPool.alloc_rows calls it with the rows it hands out, before a request writes
    them). A disaggregated decode receives a request's whole row from the prefill (REQUEST_ROW state) into the row it
    allocated, so it zeroes nothing: an asynchronous fill there could land after the transfer. Returns whether it
    attached."""
    zero = getattr(kv_pool, "zero_request_rows", None)
    rows_of = getattr(kv_pool, "zero_row_buffers", None)
    if zero is None or rows_of is None or req_to_token_pool is None or not rows_of():
        return False
    from sglang_neuron import serving_policy

    if serving_policy.current().role == "decode":
        return False
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_ALLOC_ROWS_MD5, __name__)
    previous = getattr(req_to_token_pool, "_on_alloc_rows", None)
    if getattr(previous, "_neuron_row_zeroing", False):
        return True

    def on_alloc_rows(rows):
        zero(rows)
        if previous is not None:
            previous(rows)

    on_alloc_rows._neuron_row_zeroing = True
    req_to_token_pool._on_alloc_rows = on_alloc_rows
    return True

# The DSA index key width the index-K layout and the indexer kernels take.
INDEX_K_DIM = 128
INDEX_SHARDINGS = ("owner", "replicated")


def resolve_cp(page_size: int):
    """(CpPages, cp_rank, source) of this rank's KV pool: decode context parallelism (SGLang's ``--dcp-size``: the
    pool's page is the block, the allocator's widened page the native page), a token-CP layout the model declared
    (model_executor/kv_layout: SGLang's page is the native page, ``page / cp_size`` the block), or neither."""
    from sglang_neuron.distributed.decode_context_parallel import runtime_layout
    from sglang_neuron.model_executor import kv_layout

    layout = kv_layout.current()
    dcp = runtime_layout(page_size)
    if dcp is not None and layout.cp_size > 1:
        raise ValueError("decode context parallelism over a context-parallel KV layout is not supported")
    if dcp is not None:
        return CpPages(int(dcp.size), int(page_size)), int(dcp.rank), "dcp"
    if layout.cp_size > 1:
        return CpPages(int(layout.cp_size), layout.physical_block_size(page_size)), int(layout.cp_rank), "cp"
    return CpPages(1, int(page_size)), 0, "none"


class NeuronMLATokenToKVPool(KVCache):
    """The latent pool (and, with ``index_head_dim``, the DSA index-K pool) of an MLA model on Neuron."""

    kv_head_major = True  # one "head": a latent page is contiguous, so a P/D transfer moves it whole
    is_neuron_latent_pool = True
    # What SGLang's HiCache assembler reads of a KV pool (hybrid_pool_assembler.build_kv_host_pool): the host pool is
    # sized from the pool's own token count, every rank holds every layer (CP shards tokens, not layers), and the
    # Neuron host pool stages nothing on the device (it DMAs whole pages, mem_cache/host_pool.py).
    host_capacity_tokens = None
    layer_shard_enabled = False
    hicache_write_back_staging = None
    # the zero-page contract's policy (mem_cache/allocator.ZERO_POLICIES): zeroed on free, the default; a model whose
    # pool must zero the pages it hands out sets "on_alloc" (not served on a PD decode)
    zero_policy = "on_free"
    # whether zero_pages' device fill leaves bytes outside its pages alone while a HiCache load DMA writes the buffer;
    # False: the allocator waits for the loads in flight first (mem_cache/allocator)
    zero_fill_load_safe = False

    def __init__(self, size: int, page_size: int, dtype: torch.dtype, kv_lora_rank: int, qk_rope_head_dim: int,
                 layer_num: int, device: str, enable_memory_saver: bool, start_layer: Optional[int] = None,
                 end_layer: Optional[int] = None, *, index_head_dim: Optional[int] = None,
                 index_sharding: str = "owner", index_kpool: int = 1, index_kpool_compress: bool = False,
                 tail_extra_slots: int = 0, max_running_requests: Optional[int] = None, **kwargs):
        _refuse_unsupported_options(kwargs)
        if enable_memory_saver:
            raise ValueError("the Neuron MLA pool does not support the memory saver")
        KVCache.__init__(self, size, page_size, dtype, layer_num, device, False, start_layer, end_layer)
        if dtype not in (torch.bfloat16, torch.float16, torch.float32):
            # an fp8 latent (upstream's 656-byte DSA rows with scales) is a CUDA layout no Neuron kernel reads
            raise ValueError(f"the Neuron MLA pool stores {dtype}; the MLA kernels take a bf16 latent")
        if kv_lora_rank <= 0 or qk_rope_head_dim < 0:  # qk_rope_head_dim 0: a NoPE latent
            raise ValueError(f"invalid MLA latent split kv_lora_rank={kv_lora_rank}, "
                             f"qk_rope_head_dim={qk_rope_head_dim}")
        if size % page_size:
            raise ValueError(f"pool size {size} is not whole pages of {page_size}")
        self.store_dtype = dtype
        self.kv_lora_rank = int(kv_lora_rank)
        self.qk_rope_head_dim = int(qk_rope_head_dim)
        self.kv_cache_dim = self.kv_lora_rank + self.qk_rope_head_dim
        self.use_dsa = index_head_dim is not None
        self.dsa_kv_cache_store_fp8 = False
        self.dcp_replicated = False
        if self.use_dsa and int(index_head_dim) != INDEX_K_DIM:
            raise ValueError(f"index_head_dim {index_head_dim}: the Neuron index-K layout is {INDEX_K_DIM} wide")
        if index_sharding not in INDEX_SHARDINGS:
            raise ValueError(f"index_sharding {index_sharding!r} is not one of {INDEX_SHARDINGS}")
        self.index_head_dim = int(index_head_dim) if self.use_dsa else None
        self.index_sharding = index_sharding

        self.cp, self.cp_rank, self.cp_source = resolve_cp(page_size)
        self.block_size = self.cp.block
        # SGLang's reserved page 0 is block 0 of every rank. paged_blocks: the blocks SGLang's allocator hands out
        # (and frees) plus block 0; a model's layout may hold more (num_blocks: e.g. a scratch block), never freed
        self.num_blocks = self.paged_blocks = size // page_size + 1
        # the attributes the MHA pool publishes for the runtime's and the P/D transfer's layout checks
        self.kv_cp_size = self.cp.cp_size if self.cp_source == "cp" else 1
        self.kv_cp_rank = self.cp_rank if self.cp_source == "cp" else 0
        self.kv_cp_interleave = self.block_size
        self.dcp = None
        self.dcp_interleave = page_size if self.cp_source == "dcp" else 1
        self.index_block = (self.cp.page if self.index_sharding == "replicated" else self.block_size)
        self._init_kpool(int(index_kpool), bool(index_kpool_compress), int(tail_extra_slots or 0),
                         max_running_requests)
        self.model_layer_ids = tuple(range(self.start_layer, self.start_layer + layer_num))
        from sglang_neuron.model_executor.runtime_init import neuron_device

        self.kv_device = neuron_device()
        self._create_buffers()
        self.layer_transfer_counter = None
        logger.info("Neuron MLA pool: layers=%d blocks=%d block=%d cp=%d(%s) latent=%d index=%s(%s) dtype=%s "
                    "(%.2f GiB)", layer_num, self.num_blocks, self.block_size, self.cp.cp_size, self.cp_source,
                    self.kv_cache_dim, self.index_head_dim, self.index_sharding, dtype,
                    self.get_kv_size_bytes() / 2**30)

    def _init_kpool(self, kpool: int, compress: bool, extra: int, max_running_requests: Optional[int]) -> None:
        """KPool geometry: ``index_rows`` index keys per block (one per ``kpool`` tokens) and, with compression, the
        request tails' rows and slots. Only the compressed form is held: plain pooled keys are a layout no Neuron
        kernel reads."""
        if kpool < 1 or (kpool > 1) != compress:
            raise ValueError(f"index_kpool={kpool} with index_kpool_compress={compress}: the Neuron index-K pool holds "
                             "per-token keys or compressed pooled keys")
        if kpool > 1:
            if not self.use_dsa:
                raise ValueError(f"index_kpool={kpool} needs the DSA index keys (index_head_dim)")
            if self.index_block % kpool:
                raise ValueError(f"index_kpool={kpool} does not divide the {self.index_block}-token index block: a "
                                 "pool would straddle two blocks (two CP owners)")
            if max_running_requests is None or int(max_running_requests) < 1 or extra < 0:
                raise ValueError(f"the compressed index-K tails need max_running_requests (got {max_running_requests}) "
                                 f"and tail_extra_slots >= 0 (got {extra})")
        self.index_kpool, self.index_kpool_compress = kpool, compress
        self.kpool_use_compress = kpool > 1
        self.tail_extra_slots = extra if kpool > 1 else 0
        # upstream's request-pool rows: the scheduler's requests and the sentinel row
        self.tail_rows = int(max_running_requests) + 1 if kpool > 1 else 0
        self.tail_slots = kpool + self.tail_extra_slots if kpool > 1 else 0
        self.index_rows = self.index_block // kpool
        self.tail_keys = self.tail_scores = None

    # buffers: the model declares them (buffers()), the pool owns their life cycle and the zero-page contract
    def latent_shape(self) -> torch.Size:
        return torch.Size((self.layer_num, self.num_blocks, self.block_size, self.kv_cache_dim))

    def index_shape(self) -> Optional[torch.Size]:
        if not self.use_dsa:
            return None
        return torch.Size((self.layer_num, self.num_blocks, self.index_rows, self.index_head_dim))

    def tail_shape(self) -> Optional[torch.Size]:
        if not self.kpool_use_compress:
            return None
        return torch.Size((self.layer_num, self.tail_rows, self.tail_slots, self.index_head_dim))

    def buffers(self) -> Tuple[PoolBuffer, ...]:
        """The pool's tensors (PoolBuffer). A model whose kernels hold more, or other, tensors overrides this and
        nothing else of the life cycle: the pool allocates every buffer zero-filled, warms the zero fill of each paged
        one, zeroes the freed pages of every paged one, checks their block 0 and counts every one in its size."""
        out = [PoolBuffer("latent", tuple(self.latent_shape()), page_dim=1)]
        if self.use_dsa:
            out.append(PoolBuffer("index_keys", tuple(self.index_shape()), page_dim=1))
        if self.kpool_use_compress:
            # request-row state (one ring row per request), not pages: the scheduler's row reuse owns it
            out += [PoolBuffer("tail_keys", tuple(self.tail_shape()), row_dim=1),
                    PoolBuffer("tail_scores", tuple(self.tail_shape()), row_dim=1)]
        return tuple(out)

    def _bind_views(self) -> None:
        """The per-layer views SGLang's accessors return (kv_buffer, index_buffers) of the stacked buffers."""
        self.kv_buffer = [self.latent[i] for i in range(self.layer_num)]
        index_keys = getattr(self, "index_keys", None)
        self.index_buffers = [index_keys[i] for i in range(self.layer_num)] if index_keys is not None else []

    def _create_buffers(self):
        specs = self.buffers()
        names = [b.name for b in specs]
        if len(set(names)) != len(names) or "latent" not in names:
            raise ValueError(f"pool buffers {names}: names must be unique and include the latent")
        for spec in specs:
            setattr(self, spec.name, torch.zeros(spec.shape, dtype=spec.dtype or self.store_dtype,
                                                 device=self.kv_device))
        self._buffer_specs = specs
        self._bind_views()
        # every ladder shape of the zero fill per paged buffer, compiled now on every rank (fill_pages)
        if self.paged_blocks > 1:
            for spec in self.paged_buffers():
                warm_fill(getattr(self, spec.name), spec.page_dim)
        # and of every zero-row buffer (row 0 is SGLang's padding row, which no request reads)
        for spec in self.zero_row_buffers():
            warm_fill(spec.row_view(getattr(self, spec.name)), spec.row_dim, page=0)

    def paged_buffers(self) -> Tuple[PoolBuffer, ...]:
        """The buffers under the zero-page contract (indexed by SGLang page id along ``page_dim``)."""
        return tuple(b for b in getattr(self, "_buffer_specs", ()) if b.page_dim is not None)

    def zero_row_buffers(self) -> Tuple[PoolBuffer, ...]:
        """The request-row buffers under the zero-row contract (PoolBuffer.zero_rows)."""
        return tuple(b for b in getattr(self, "_buffer_specs", ()) if b.zero_rows)

    def zero_request_rows(self, rows) -> None:
        """Zero the zero-row buffers' slices of SGLang rows ``rows`` (attach_row_zeroing): one compiled fill per buffer
        and ladder size, as fill_pages zeroes pages."""
        ids = sorted({int(r) for r in (rows.tolist() if hasattr(rows, "tolist") else rows)})
        if not ids:
            return
        for spec in self.zero_row_buffers():
            fill_pages(spec.row_view(getattr(self, spec.name)), spec.row_dim, ids)

    def buffer_tensors(self) -> List[torch.Tensor]:
        return [t for t in (getattr(self, b.name, None) for b in getattr(self, "_buffer_specs", ())) if t is not None]

    def _clear_buffers(self):
        self.kv_buffer, self.index_buffers = [], []
        for spec in getattr(self, "_buffer_specs", ()):
            setattr(self, spec.name, None)
        self.latent = self.index_keys = self.tail_keys = self.tail_scores = None

    # access
    def _idx(self, layer_id: int) -> int:
        local = layer_id - self.start_layer
        if not 0 <= local < self.layer_num:
            raise IndexError(f"layer {layer_id} is outside this pool's layers [{self.start_layer}, "
                             f"{self.start_layer + self.layer_num})")
        return local

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        """Layer ``layer_id``'s latent blocks ``[blocks, block, kv_lora_rank + qk_rope_head_dim]``."""
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self.kv_buffer[self._idx(layer_id)]

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        """MLA values are the latent's first ``kv_lora_rank`` channels."""
        return self.get_key_buffer(layer_id)[..., : self.kv_lora_rank]

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def get_index_k_buffer(self, layer_id: int) -> torch.Tensor:
        """Layer ``layer_id``'s index keys ``[blocks, index_block, index_head_dim]``."""
        if not self.use_dsa:
            raise RuntimeError("this MLA pool holds no DSA index keys")
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self.index_buffers[self._idx(layer_id)]

    # SGLang's DSA name for the index-K buffer (no scale rows on Neuron)
    get_index_k_with_scale_buffer = get_index_k_buffer

    def get_compress_tail_buffers(self, layer_id: int):
        """Layer ``layer_id``'s request tails ``([rows, slots, 128] keys, [rows, slots, 128] gate scores)``."""
        if not self.kpool_use_compress:
            raise RuntimeError("this pool holds no compressed index-K tails (index_kpool 1)")
        i = self._idx(layer_id)
        return self.tail_keys[i], self.tail_scores[i]

    @property
    def index_k_with_scale_buffer(self):
        return self.index_buffers

    # writes are the kernels'
    def set_kv_buffer(self, *args, **kwargs):
        raise RuntimeError("the Neuron MLA pool is model-written: the compiled graphs' kernels write the latent and "
                           "index keys through their write slots (attention/mla_operands.py)")

    set_mla_kv_buffer = set_kv_buffer
    set_index_k_scale_buffer = set_kv_buffer
    set_index_k_buffer = set_kv_buffer

    def move_kv_cache(self, tgt_loc, src_loc):
        raise NotImplementedError("the Neuron MLA pool does not move KV between slots (no speculative decoding)")

    # the zero-page contract
    def zero_pages(self, page_ids) -> None:
        """Zero the blocks of freed SGLang pages (the zero-page contract: a reused page's tail reads as zeros). Block
        0 is never freed and never touched; ids outside the pool are ignored. Eager fills of one shape per buffer
        (fill_pages)."""
        ids = sorted({int(p) for p in (page_ids.tolist() if hasattr(page_ids, "tolist") else page_ids)
                      if 0 < int(p) < self.paged_blocks})
        if not ids:
            return
        for spec in self.paged_buffers():
            fill_pages(getattr(self, spec.name), spec.page_dim, ids)

    def zero_page_violations(self) -> list:
        """(buffer, layer) pairs whose block 0 is not all zeros, for every paged buffer (layer: the global layer id of
        a buffer stacked over the pool's layers, else the leading index): empty while the contract holds. A
        diagnostic: it reads the pool back."""
        bad = []
        for spec in self.paged_buffers():
            buf = getattr(self, spec.name)
            block0 = buf.select(spec.page_dim, 0)
            lead = block0.reshape(block0.shape[0], -1) if spec.page_dim > 0 else block0.reshape(1, -1)
            nonzero = lead.ne(0).any(dim=1).cpu()
            stacked = spec.page_dim > 0 and buf.shape[0] == self.layer_num
            bad.extend((spec.name, self.start_layer + i if stacked else i)
                       for i in torch.nonzero(nonzero).flatten().tolist())
        return bad

    # bookkeeping
    def get_kv_size_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.buffer_tensors())

    def latent_page_bytes(self) -> int:
        return self.block_size * self.kv_cache_dim * self.store_dtype.itemsize

    def index_page_bytes(self) -> int:
        return self.index_rows * (self.index_head_dim or 0) * self.store_dtype.itemsize

    def get_contiguous_buf_infos(self):
        """(ptrs, lens, item_lens) of the latent, one entry per layer: SGLang page ``p`` is this rank's block ``p`` at
        ``ptr + p * item_len``."""
        from sglang_neuron.backend.device_io import device_address

        return ([device_address(t) for t in self.kv_buffer], [t.numel() * t.element_size() for t in self.kv_buffer],
                [self.latent_page_bytes()] * self.layer_num)

    def get_state_buf_infos(self):
        """The index keys, which SGLang's P/D transfer registers as the DSA state beside the latent: one entry per
        layer, a page per item like the latent's."""
        if not self.use_dsa:
            return [], [], []
        from sglang_neuron.backend.device_io import device_address

        return ([device_address(t) for t in self.index_buffers],
                [t.numel() * t.element_size() for t in self.index_buffers],
                [self.index_page_bytes()] * self.layer_num)

    def get_compress_tail_buf_infos(self):
        """The request tails, which SGLang's P/D transfer registers as the DSA tail state: every layer's key tails,
        then every layer's score tails, one request row per item (upstream's order); none without KPool."""
        if not self.kpool_use_compress:
            return [], [], []
        from sglang_neuron.backend.device_io import device_address

        views = [self.tail_keys[i] for i in range(self.layer_num)] + [self.tail_scores[i] for i in range(self.layer_num)]
        return ([device_address(t) for t in views], [t.numel() * t.element_size() for t in views],
                [t[0].numel() * t.element_size() for t in views])

    def get_kv_layer_ids(self):
        """GLOBAL layer id of each get_contiguous_buf_infos entry (P/D pairs entries by it under PP)."""
        return list(self.model_layer_ids)

    def register_layer_transfer_counter(self, counter):
        self.layer_transfer_counter = counter

    # host copies: HiCache L2 (mem_cache/mla_host_pool.py) and SGLang's retraction backup
    def page_views(self) -> List[Tuple[str, tuple, torch.Tensor]]:
        """(buffer name, leading index, view) of every paged buffer, split over the dimensions before its ``page_dim``
        (the layers of a stacked buffer) into contiguous page-first views: a host copy moves row ``p`` of each, this
        rank's block of SGLang page ``p``. In ``buffers()`` order, so a model's extra paged buffers ride along."""
        return [(spec.name, idx, view) for spec in self.paged_buffers()
                for idx, view in _split_leading(getattr(self, spec.name), spec.page_dim)]

    def row_views(self) -> List[Tuple[str, tuple, torch.Tensor]]:
        """(buffer name, leading index, view) of every request-row buffer (``row_dim``), row-first."""
        return [(spec.name, idx, view) for spec in getattr(self, "_buffer_specs", ()) if spec.row_dim is not None
                for idx, view in _split_leading(spec.row_view(getattr(self, spec.name)), spec.row_dim)]

    def request_pages(self, indices) -> List[int]:
        """The SGLang pages of a request's token slots ``indices`` (positions 0 .. n - 1 in order, as ``req_to_token``
        holds them; a partial last page moves whole): the native page of each, which is this rank's block id under
        every CP layout, so no slot is translated. Refused unless position ``i`` sits at offset ``i mod page`` of its
        page, on the reserved page 0, and past the pool's pages."""
        idx = torch.as_tensor(indices).reshape(-1).to("cpu", torch.int64)
        page = self.cp.page
        if idx.numel() == 0:
            return []
        pages = idx[::page] // page
        expected = pages.repeat_interleave(page)[: idx.numel()] * page + torch.arange(idx.numel()) % page
        if not torch.equal(idx, expected):
            raise ValueError(f"a host copy moves whole {page}-token pages: the request's slots must fill its pages in "
                             "order from a page start")
        pages = pages.tolist()
        if len(set(pages)) != len(pages) or min(pages) < 1 or max(pages) >= self.paged_blocks:
            raise ValueError(f"request pages must be distinct and in [1, {self.paged_blocks}): page 0 is the reserved "
                             "zero page")
        return pages

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        """SGLang's retraction backup (Req.offload_kv_cache): a host copy of the request's pages of every paged buffer,
        this rank's block of each, and with ``req_pool_index`` its row of every request-row buffer (the KPool tails,
        as upstream's DSA pool carries them). The runtime's DMA copies after the launched graphs
        (host_pool.copy_pages_now): no graph of its own, so nothing compiles while serving. No Mamba state here."""
        from sglang_neuron.mem_cache.host_pool import copy_pages_now, pinned_host_like

        pages = self.request_pages(indices)
        on_neuron = self.latent.device.type == "neuron"
        page_views = self.page_views()
        data = [pinned_host_like(view, len(pages), on_neuron) for _, _, view in page_views]
        copy_pages_now([v for _, _, v in page_views], pages, data, to_host=True)
        row_views = self.row_views() if req_pool_index is not None else []
        rows = []
        if row_views:
            row = _request_row(req_pool_index, row_views)
            rows = [pinned_host_like(view, 1, on_neuron) for _, _, view in row_views]
            copy_pages_now([v for _, _, v in row_views], [row], rows, to_host=True)
        return PoolCpuCopy(tokens=int(torch.as_tensor(indices).numel()), pages=len(pages),
                           regions=tuple((name, idx) for name, idx, _ in page_views), data=tuple(data),
                           row_regions=tuple((name, idx) for name, idx, _ in row_views), rows=tuple(rows))

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None):
        """Write a get_cpu_copy back to the request's new token slots ``indices`` (whole pages fresh from the allocator,
        so zeroed: the copied tail of a partial last page is the request's own bytes or zeros), and its request rows to
        row ``req_pool_index``; without ``req_pool_index`` the rows are not restored, as upstream's DSA pool does."""
        from sglang_neuron.mem_cache.host_pool import copy_pages_now

        if not isinstance(kv_cache_cpu, PoolCpuCopy):
            raise TypeError(f"{type(self).__name__}.load_cpu_copy takes its own get_cpu_copy, not "
                            f"{type(kv_cache_cpu).__name__}")
        page_views = self.page_views()
        if kv_cache_cpu.regions != tuple((name, idx) for name, idx, _ in page_views) or any(
                tuple(d.shape[1:]) != tuple(v.shape[1:]) or d.dtype != v.dtype
                for d, (_, _, v) in zip(kv_cache_cpu.data, page_views)):
            raise ValueError("the host copy was taken from a pool of other buffers")
        tokens = int(torch.as_tensor(indices).numel())
        if tokens != kv_cache_cpu.tokens:
            raise ValueError(f"the host copy holds {kv_cache_cpu.tokens} tokens, not {tokens}")
        pages = self.request_pages(indices)
        copy_pages_now([v for _, _, v in page_views], pages, list(kv_cache_cpu.data), to_host=False)
        if kv_cache_cpu.rows and req_pool_index is not None:
            row_views = self.row_views()
            if kv_cache_cpu.row_regions != tuple((name, idx) for name, idx, _ in row_views):
                raise ValueError("the host copy's request rows were taken from a pool of other buffers")
            copy_pages_now([v for _, _, v in row_views], [_request_row(req_pool_index, row_views)],
                           list(kv_cache_cpu.rows), to_host=False)

    def get_device_pool_infos(self):
        """The pool's buffers for SGLang's HiCache assembler (hybrid_pool_assembler; the host pools are
        mem_cache/mla_host_pool.py's): the latent as the KV pool and, on a DSA pool, the index keys as the INDEXER side
        pool, which takes the KV pool's host indices. Each layer's buffer is this rank's rows, page ``p`` at rows
        ``[p * rows, (p + 1) * rows)``: the latent ``[blocks * block, 1, kv_cache_dim]``, the index keys ``[blocks *
        index_rows, 1, 128]`` (KEY_ROWS; ``page_size / compress_ratio`` is the index rows of a page, so under token CP
        the ratio also counts the tokens other ranks hold). Layer ids are global. No pointer tables: the Neuron host
        pools DMA whole pages. A model's other paged buffers ride with the latent's host pool."""
        from sglang.srt.mem_cache.device_pool_info import (DevicePoolInfo, IndexKeyBufferInfo, IndexKeyLayout,
                                                           MLABufferInfo)
        from sglang.srt.mem_cache.hicache_storage import PoolName

        layer_ids = tuple(self.model_layer_ids)
        latent = MLABufferInfo(buffers=tuple(t.view(-1, 1, self.kv_cache_dim) for t in self.kv_buffer),
                               data_ptrs=None, layer_indices=tuple(range(self.layer_num)),
                               device_layer_num=self.layer_num, page_size=self.page_size,
                               kv_lora_rank=self.kv_lora_rank, qk_rope_head_dim=self.qk_rope_head_dim,
                               kv_cache_dim=self.kv_cache_dim, dtype=self.store_dtype, device=self.latent.device)
        infos = [DevicePoolInfo(pool_name=PoolName.KV, indices_from_pool=PoolName.KV, layer_ids=layer_ids,
                                buffer_info=latent)]
        if self.use_dsa:
            if self.page_size % self.index_rows:
                raise ValueError(f"the index keys hold {self.index_rows} rows a page, which do not divide SGLang's "
                                 f"{self.page_size}-token page (replicated index keys under DCP): no host pool "
                                 "describes them")
            keys = IndexKeyBufferInfo(page_size=self.page_size, compress_ratio=self.page_size // self.index_rows,
                                      key_layout=IndexKeyLayout.KEY_ROWS,
                                      buffers=tuple(t.view(-1, 1, self.index_head_dim) for t in self.index_buffers))
            infos.append(DevicePoolInfo(pool_name=PoolName.INDEXER, indices_from_pool=PoolName.KV,
                                        layer_ids=layer_ids, buffer_info=keys))
        return tuple(infos)


def _split_leading(buffer: torch.Tensor, dim: int) -> list:
    """(leading index, view) of ``buffer`` over every index of its dimensions before ``dim``: contiguous views whose
    first dimension is ``dim``."""
    if not buffer.is_contiguous():
        raise ValueError("a host copy moves rows of a contiguous pool buffer")
    return [(idx, buffer[idx]) for idx in itertools.product(*(range(int(n)) for n in buffer.shape[:dim]))]


def _request_row(req_pool_index, row_views) -> int:
    """The request-pool row ``req_pool_index`` names (an int or a one-element tensor), checked against the rows."""
    index = torch.as_tensor(req_pool_index).reshape(-1)
    rows = int(row_views[0][2].shape[0])
    if index.numel() != 1 or not 0 <= int(index[0]) < rows:
        raise ValueError(f"req_pool_index {req_pool_index!r} is not one request row of [0, {rows})")
    return int(index[0])


@dataclass(frozen=True)
class PoolCpuCopy:
    """A request's host copy (NeuronMLATokenToKVPool.get_cpu_copy): ``data[i]`` holds ``pages`` pages of region
    ``regions[i]`` (buffer name, leading index), ``rows[i]`` the request's row of request-row region
    ``row_regions[i]``; ``tokens`` is the number of slots it was taken for."""

    tokens: int
    pages: int
    regions: tuple
    data: tuple
    row_regions: tuple = ()
    rows: tuple = ()


# Upstream pool options whose features the Neuron latent pool does not have, with their "off" values.
_OPTIONS_OFF = {"use_dsa": None, "override_kv_cache_dim": None, "dcp_replicated": False, "index_buf_size": None,
                "skip_topk_layers": None, "kv_cache_dim": None}


def _refuse_unsupported_options(kwargs: dict) -> None:
    unknown = sorted(set(kwargs) - set(_OPTIONS_OFF))
    if unknown:
        raise TypeError(f"the Neuron MLA pool takes no {', '.join(unknown)}")
    on = []
    for key, value in kwargs.items():
        if key in ("use_dsa", "kv_cache_dim", "index_buf_size"):
            continue  # informational, or checked by the kinds
        if key == "skip_topk_layers":
            continue  # a shared-indexer layer reuses the previous selection: its index keys are allocated, never read
        if value not in (_OPTIONS_OFF[key], None):
            on.append(f"{key}={value!r}")
    if on:
        raise ValueError(f"the Neuron MLA pool does not support {', '.join(on)}")


def _upstream_kinds():
    from sglang.srt.mem_cache import memory_pool as upstream

    return getattr(upstream, "MLATokenToKVPool", None), getattr(upstream, "DSATokenToKVPool", None)


_MLA_BASE, _DSA_BASE = _upstream_kinds()


def _refuse_inherited(cls, own, base) -> None:
    """Every method ``base`` (SGLang's pool) defines that ``own`` (the Neuron pool) does not, and KVCache has no generic
    form of, raises naming itself on ``cls``: it addresses SGLang's token-major buffers, which this pool does not
    have."""
    generic = {name for name, value in vars(KVCache).items() if callable(value)}
    for klass in base.__mro__:
        if klass in (KVCache, object) or not issubclass(klass, KVCache):
            continue
        for name, value in list(vars(klass).items()):
            if (name.startswith("__") or not callable(value) or name in vars(cls) or hasattr(own, name)
                    or name in generic):
                continue

            def refuse(self, *args, _name=name, _base=klass.__name__, **kwargs):
                raise NotImplementedError(f"{type(self).__name__}.{_name}: SGLang's {_base} method addresses its "
                                          "token-major buffers, which the Neuron latent pool does not have")

            refuse.__name__ = name
            setattr(cls, name, refuse)


if _MLA_BASE is not None:

    class NeuronMLATokenToKVPoolKind(NeuronMLATokenToKVPool, _MLA_BASE):
        """The Neuron latent pool as upstream SGLang's MLA pool kind (``get_kv_pool_cls(kind="mla")``), constructed
        with ``MLATokenToKVPool``'s keyword arguments (standalone and as the hybrid-linear pool's full-attention
        leaf)."""

        def __init__(self, size, page_size, dtype, kv_lora_rank, qk_rope_head_dim, layer_num, device,
                     enable_memory_saver, start_layer=None, end_layer=None, **kwargs):
            if kwargs.get("use_dsa"):
                raise ValueError("a DSA model's pool is the dsa kind (NeuronDSATokenToKVPoolKind)")
            NeuronMLATokenToKVPool.__init__(self, size, page_size, dtype, kv_lora_rank, qk_rope_head_dim, layer_num,
                                            device, enable_memory_saver, start_layer, end_layer, **kwargs)

    _refuse_inherited(NeuronMLATokenToKVPoolKind, NeuronMLATokenToKVPool, _MLA_BASE)
else:  # an SGLang without the MLA pool kind
    NeuronMLATokenToKVPoolKind = None


if _DSA_BASE is not None:

    class NeuronDSATokenToKVPoolKind(NeuronMLATokenToKVPool, _DSA_BASE):
        """The Neuron latent + index-K pool as upstream SGLang's DSA pool kind (``get_kv_pool_cls(kind="dsa")``),
        constructed with ``DSATokenToKVPool``'s keyword arguments. The index keys are sharded like the latent unless the
        model declares them replicated (``index_sharding``, bound with ``bind``)."""

        index_sharding_default = "owner"

        @classmethod
        def bind(cls, *, index_sharding: str):
            """The kind with the model's index-K sharding (the model's pool declaration, before SGLang builds it)."""
            if index_sharding not in INDEX_SHARDINGS:
                raise ValueError(f"index_sharding {index_sharding!r} is not one of {INDEX_SHARDINGS}")
            return type(cls.__name__, (cls,), {"index_sharding_default": index_sharding})

        def __init__(self, size, page_size, kv_lora_rank, dtype, qk_rope_head_dim, layer_num, device, index_head_dim,
                     enable_memory_saver, kv_cache_dim, start_layer=None, end_layer=None, **kwargs):
            if kv_cache_dim != kv_lora_rank + qk_rope_head_dim:
                raise ValueError(f"kv_cache_dim {kv_cache_dim} is not kv_lora_rank + qk_rope_head_dim "
                                 f"({kv_lora_rank} + {qk_rope_head_dim}): an FP8 DSA latent is a CUDA layout")
            NeuronMLATokenToKVPool.__init__(self, size, page_size, dtype, kv_lora_rank, qk_rope_head_dim, layer_num,
                                            device, enable_memory_saver, start_layer, end_layer,
                                            index_head_dim=index_head_dim,
                                            index_sharding=self.index_sharding_default, **kwargs)
            self.index_page_size = self.index_rows
            self.index_buf_size = (self.num_blocks - 1) * self.index_rows
            skip = kwargs.get("skip_topk_layers")
            self.skip_topk_layers = list(skip) if skip is not None else [False] * layer_num
            if len(self.skip_topk_layers) != layer_num:
                raise ValueError(f"{len(self.skip_topk_layers)} skip_topk_layers flags for {layer_num} layers")
            self.kv_scale_buffer = None

    _refuse_inherited(NeuronDSATokenToKVPoolKind, NeuronMLATokenToKVPool, _DSA_BASE)
else:
    NeuronDSATokenToKVPoolKind = None


# The DSA kind the platform answers: the model may bind its index-K sharding before SGLang builds the pool.
_DSA_KIND = NeuronDSATokenToKVPoolKind


def declare_dsa_index_sharding(index_sharding: str) -> None:
    """A DSA model declares where its index keys live (``owner``: sharded like the latent; ``replicated``: whole on
    every CP rank) during construction, before SGLang builds the pool."""
    global _DSA_KIND
    if NeuronDSATokenToKVPoolKind is None:
        raise RuntimeError("this SGLang has no DSA pool kind")
    _DSA_KIND = NeuronDSATokenToKVPoolKind.bind(index_sharding=index_sharding)


def dsa_kind():
    return _DSA_KIND
