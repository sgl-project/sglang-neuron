# MLA and DSA pools

## Overview

An MLA model (`kv_lora_rank` set) caches one latent vector per token, shared by every head; a DSA model also caches
the index keys of its lightning indexer. The platform's paged latent pool (`sglang_neuron/mem_cache/mla_pool.py`) holds
both on the NeuronCore in the paged, token-major layout the MLA and DSA kernels read, as upstream SGLang's `mla` and
`dsa` pool kinds. The compiled graphs' kernels write the pool; SGLang's allocator, radix cache and PD transfer address
it by page.

SGLang's `mla` kind gets this pool when the launch sets `SGLANG_NEURON_MLA_POOL=paged`. The default (`rows`) is
SGLang's MLA pool on the NeuronCore with one row per token slot (`mem_cache/memory_pool.NeuronMLATokenToKVPool`), which
GLM-5.2's prefill pool and Kimi-K3's MLA layers build on.

## Key concepts

- **Latent**: `kv_lora_rank` compressed KV plus `qk_rope_head_dim` RoPE key per token (576 = 512 + 64; 512 for a
  NoPE latent, whose `qk_rope_head_dim` is 0).
- **Index keys**: the DSA indexer's keys, `index_head_dim` = 128, bf16. With KPool (`index_kpool` k > 1 and
  `index_kpool_compress`) one learned compression of every k consecutive tokens' keys.
- **Block**: SGLang's page `b` on this rank is block `b`, so a block table is SGLang's page ids and block 0 is SGLang's
  reserved page, never allocated.
- **Index sharding**: `owner` (the index keys sharded like the latent, each rank scoring its own tokens) or
  `replicated` (whole on every CP rank, each rank selecting over the full context, as upstream's DSA pool keeps them).

## How it works

### Layout

The pool holds each kind of tensor for every layer in one tensor:

- latent `[layers, blocks, block, kv_lora_rank + qk_rope_head_dim]`;
- index keys `[layers, blocks, index_block / index_kpool, index_head_dim]`;
- with KPool, the request tails `[layers, max_running_requests + 1, index_kpool + tail_extra_slots, index_head_dim]`,
  once for the raw index keys and once for the compression gate scores of the tokens past a request's last complete
  pool: upstream's per-request ring, held by every CP rank.

Under context parallelism (`mem_cache/cp_pages.CpPages`: decode context parallelism or a declared token-CP layout)
one native page is one block per CP rank, so a rank's pool holds `native page / cp_size` tokens of every page
([Context parallelism](context-parallelism.md)). A P/D transfer moves whole blocks, one page per item.

### The zero-page contract

The kernels gather whole blocks and mask the positions past a sequence's length in the scores; the masked keys still
enter the value matmul with probability 0, and `0 * Inf` is NaN. The pool therefore guarantees that:

1. every block is zero when the pool is built, and block 0 is never written: the write slots of padding rows and of
   tokens another CP rank owns are `cp_pages.DROP_SLOT` (skipped), and block-table padding reads block 0;
2. a freed page is zeroed before it is handed out again (`zero_pages`, called from the release path of the Neuron paged
   allocators in `mem_cache/allocator.py`).

The zeroing is one compiled in-place fill per buffer over every layer at once (`fill_pages`), in chunks of exactly
`ZERO_CHUNK` page ids (the last chunk padded by repeating an id, since the fill is idempotent). Each buffer has one
compiled shape, which every rank compiles when it builds the pool (`warm_fill`), so a free never compiles while serving.

### Declared buffers

A model declares its tensors (`buffers()`, one `PoolBuffer` each: the latent, the index keys, an MTP latent, request
rows or tails). The pool allocates them zero-filled, applies the zero-page contract to every paged one (`page_dim`) and
counts them all in its size, so a model's pool lists its buffers instead of re-implementing their life cycle. The base
pool's own `buffers()` (`mem_cache/mla_pool.py`) is the default set: the latent, the index keys of a DSA model, and the
tail keys and scores when the pool keeps compressed tails.

## Configuration

There is no launch setting. A model selects the pool kind through SGLang's pool configurator and, for a DSA model,
declares its index sharding during construction (`mla_pool.declare_dsa_index_sharding`), before SGLang builds the pool.

## Supported configurations

DeepSeek-V2 serves on the paged pool with `SGLANG_NEURON_MLA_POOL=paged`: one block a page, no context parallelism
and no index keys, so its graphs read the blocks as SGLang's token slots (`DeepseekV2ForCausalLM.bind_kv_cache`). The
pool's other features have no model in the catalog that uses them: the DSA kind
(`NeuronDSATokenToKVPoolKind`, which the platform's `get_kv_pool_cls` does not answer, so a model selects it through
its own pool module or a megakernel pool layout), both index shardings, the KPool tails, the NoPE latent, and the
token-CP and DCP blocks.

The paged pool serves SGLang's radix cache and HiCache L2 (`--enable-hierarchical-cache`): `get_device_pool_infos` hands
SGLang's hybrid pool assembler the latent as the KV pool and, on a DSA pool, the index keys as the INDEXER side pool.
The Neuron host pools of `mem_cache/mla_host_pool.py` (`NeuronLatentHostPool`, `NeuronIndexKeyHostPool`) copy whole
pages, each rank its own block of each page, and SGLang's retraction backup (`get_cpu_copy` / `load_cpu_copy`) also
carries the request's row of every request-row buffer.

## Limitations

The pool refuses:

- a latent dtype other than bf16, fp16 or fp32 (an FP8 latent with scales is a CUDA layout), and an index width other
  than 128;
- the memory saver, and the upstream options `override_kv_cache_dim` and `dcp_replicated`;
- SGLang's token-major writers (`set_kv_buffer` and the others): the pool is model-written;
- HiCache over the rows pool (no pages to copy), and MTP draft KV in the host pools;
- KPool without compression, KPool that does not divide the index block, and the compressed tails without
  `max_running_requests`;
- a prefill without DCP as the PD peer of an MLA or DSA DCP decode (`sglang_patches/dcp_page_transfer.py`,
  `refuse_mla_page_peer`): upstream's MLA plan sends tokens round-robin and no index keys, while this pool keeps whole
  blocks per DCP rank, which `cp_pages.latent_page_plan` maps and no transfer executes. (`--dcp-size` itself is gated
  by the model's catalog entry, `decode_context_parallel`, in `distributed/decode_context_parallel.dcp_launch_problems`.)

## Upstream SGLang interactions

`NeuronMLATokenToKVPoolKind` and `NeuronDSATokenToKVPoolKind` subclass upstream's `MLATokenToKVPool` and
`DSATokenToKVPool` and take their constructor arguments, so SGLang's pool configurator builds them standalone and as the
hybrid-linear pool's full-attention leaf. Every inherited upstream method that addresses upstream's token-major buffers,
and has no generic `KVCache` form, raises naming itself.

## Implementation

| File | Contents |
|---|---|
| `sglang_neuron/mem_cache/mla_pool.py` | the pool, its upstream kinds, the zero fills and the index-sharding declaration |
| `sglang_neuron/mem_cache/mla_host_pool.py` | the HiCache host pools of the latent and index keys, and the assembler hooks |
| `sglang_neuron/mem_cache/cp_pages.py` | block-cyclic page ownership under context parallelism |
| `sglang_neuron/mem_cache/allocator.py` | the paged allocators that zero freed pages |
| `sglang_neuron/attention/mla_operands.py` | the operands through which the kernels write the latent and index keys |

`tools/check_mla_hicache.py` checks the host pools' page round trips, the index-key side pool on the latent's host
indices and the retraction backup on hardware.
