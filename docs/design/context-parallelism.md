# Context Parallelism

## Overview

Context parallelism (CP) splits the token dimension of one request across ranks; tensor parallelism splits attention by
heads. The plugin has three CP mechanisms, each with its own scope:

- **Decode context parallelism (DCP)**, SGLang's `--dcp-size` on a GPT-OSS PD decode, and on Kimi-K3's PD prefill and
  PD decode: every sequence's KV is split by page over a group of attention-TP ranks that hold the same KV head (on
  Kimi-K3, the MLA latent). It is specified in [Decode context parallelism](decode-context-parallelism.md), and
  Kimi-K3's layout is in its [recipe](../model-recipes/kimi-k3.md#the-parallel-layout).
- **Token context parallelism of the KV**, a layout a model declares: every rank of the attention-TP group holds every
  KV head for its own share of each page. The MHA pool and the platform runtime address it on a PD prefill, and on the
  PD decode of a model whose decode graph attends over the CP blocks itself: Qwen3.5's kv_cp decode, which a
  whole-page prefill feeds through the [scatter](#a-whole-page-prefill-into-a-cp-decode). The paged MLA and DSA pools
  (`mem_cache/mla_pool.py`) can hold it too, and no model in the catalog declares it on them. This document specifies
  it.
- **Traced CP helpers** (`parallel/context_parallel.py`) that a model's graph or kernel host uses over block-cyclic KV:
  prefill row splits, window gathers and the exact merge of partial attention.

SGLang's attention context parallelism (`--attn-cp-size`) is not served: the platform refuses it at launch
(`platform.refuse_unserved`). Token context parallelism uses the attention-TP group, not SGLang's attention-CP group.

## Key concepts

**KV CP degree, CP rank, interleave.** With degree `C` a page of `P` tokens is one `P / C`-token block (the interleave)
on each rank of the attention-TP group; CP rank `r` (the attention-TP rank) holds the page's tokens
`[r x P / C, (r + 1) x P / C)`. SGLang's page, allocator, radix tree and block tables are unchanged and identical on
every rank; only the tokens a rank keeps of each page differ.

**Native page and block.** `mem_cache/cp_pages.CpPages(cp_size, block)`: a native page of `cp_size x block` tokens is
one `block`-token kernel block on each rank. Under token CP the native page is SGLang's page and the block its
interleave; under DCP the native page is SGLang's widened page and the block the pool's page.

**Partial attention and the exact merge.** A rank's attention over its KV share gives, per query and head, an output
normalized over its own keys and the log-sum-exp of its scores. With `M = max_r lse_r` the exact output is
`sum_r out_r exp(lse_r - M) / sum_r exp(lse_r - M)` (`merge_partials`; `lse_of` forms the log-sum-exp from a flash
kernel's row max and row sum).

## How it works

### The declared layout

A model declares the layout while it is built, before SGLang builds its KV pool:
`model_executor/kv_layout.declare(KVLayout(cp_size=C, cp_rank=r, kv_heads_per_rank=H))`. The pool reads
`kv_layout.current()`:

- `NeuronMHATokenToKVPool` holds `H` KV heads (SGLang's split when `kv_heads_per_rank` is None) and blocks of
  `P / C` tokens, and publishes `kv_cp_size`, `kv_cp_rank` and `kv_cp_interleave`. A page that `C` does not divide is
  refused.
- `NeuronMLATokenToKVPool` (`mem_cache/mla_pool.resolve_cp`) holds the latent and, for DSA, the index keys in the same
  block-cyclic pages, and publishes the same three attributes.

A model that declares nothing (GPT-OSS) keeps one whole-page block per rank and publishes degree 1.

### The platform runtime

`NeuronModel.kv_cp_size` states the degree the model's graphs address. `NeuronModelRuntime.__init__` reads it and:

- refuses a degree that does not divide the page or does not span the attention-TP group;
- serves it on the PD prefill role, and on the PD decode role of a model that serves its own CP decode
  (`NeuronModel.kv_cp_decode`): a plain decode graph addresses a page as one block of the rank's, and an aggregated
  server's decode reads the pages its own prefill writes;
- refuses the hybrid SWA pool, since the CP prefill indexes the sliding layers with full-pool pages;
- refuses a KV pool (or a hybrid pool's full sub-pool) whose published degree, CP rank or interleave is not the
  model's, a pool that publishes none counting as degree 1;
- adds the degree to the contract digest every rank of the engine compares before it builds a graph.

On a CP prefill step `build_model_inputs` gives the graph this rank's slots (`_cp_rank_slots`: the chunk's tokens at
in-page offsets `[r x I, (r + 1) x I)`, compacted to its blocks; padding stays on the reserved block 0; a chunk off the
page grid is refused) and the interleave as the kernel block. A sliding layer reads its prior through a static table of
whole pages (`_cp_prefill_swa_window`): the pages that hold the `W - 1` keys before the page-aligned prior and the
first active page, rounded up to whole kernel row tiles, with the frame offset of its first page. The model's prefill
attention must read and write only its rank's share of each page; the runtime cannot check that.

A CP decode graph (Qwen3.5's) reads every page through the block table, which is the same on every rank. Each rank
all-gathers the query heads over the attention-TP group and attends over its own block of every page. The kernel then
merges the partials over the group: an all-gather of each rank's softmax statistics, and a reduce-scatter that gives
each rank the merged output of its own heads. Only the rank whose block holds the new token writes its K and V. The
other ranks write them to a sink row after the pool's pages, which the allocator never hands out and no read sees.

A model whose runtime does not run `NeuronModelRuntime.__init__` checks its pools itself.

### The PD hand-off

SGLang's PD transfer reads no `kv_cp_*` attribute: at equal attention TP it copies each prefill rank's pages into the
decode rank of the same attention-TP rank, block for block. That is the exact hand-off when the decode holds the same
layout, and silent corruption otherwise, so `sglang_patches/pd_kv_cp_layout` publishes the decode's layout (degree, CP
rank, block) in its registration and the prefill refuses a decode at another layout or CP rank.

### A whole-page prefill into a CP decode

A CP decode rank holds one block of every page, in every KV head. A prefill without token CP holds whole pages, and its
KV heads split over its attention-TP group. So a decode rank's block of a page is on several prefill ranks, one for
each KV head shard, and SGLang's copy of whole pages between equal attention-TP ranks cannot move it.
`sglang_patches/kv_cp_scatter.py` serves this one asymmetric pairing (`pd_kv_cp_layout.scatter_pair`): the prefill
holds whole pages, the decode's frame names the KV heads it holds, and the decode's degree times its block is the
prefill's page. Qwen3.5's kv_cp decode is the model that uses it today
([Qwen3.5 PD](../model-recipes/qwen35-pd.md#256k-context-kv_cp16-decode-context-parallelism)).

**The layout frame.** The decode's registration frame is `CpLayout(cp_size, cp_rank, block, heads, k_tm)`: the degree,
the CP rank, the block, the KV heads the rank holds of its block (0: SGLang's split), and whether its K pages are
token-major (`KVLayout.k_token_major`, which the model declares). A frame with neither heads nor token-major K encodes
the first three fields, and a frame with heads and dim-major K the first four, so a peer of an older plugin reads the
same fields.

**What the prefill refuses.** At registration (`kv_cp_scatter.geometry_problem`, `pd_kv_cp_layout.cp_layout_problem`)
the prefill refuses a decode rank, which then fails its rooms with the reason, for:

- an MLA pool: the scatter moves head-sharded MHA pages;
- a prefill attention TP other than the decode's degree;
- KV heads that do not split the degree evenly;
- a KV entry whose prefill page is not the degree's number of whole blocks, or whose decode page is not the heads
  times the block, and KV entries that do not pair one per layer (a decode-only speculative layer, for example);
- K pages of different layouts on the two roles. A prefill with token-major K also refuses a decode without the frame
  or with whole pages, since the whole-page transfer copies bytes without regard to the layout.

**The rank mapping.** Decode rank `e` pulls from the prefill ranks that hold its KV heads: rank `e mod G + h x G` for
each head shard `h`, with `G = degree / heads` (`CommonKVManager._resolve_rank_mapping`, wrapped). It waits for one
response from each source. Each prefill rank serves `heads` decode ranks.

**The send.** At registration the prefill builds two strided descriptor lists for each decode peer
(`NixlKVManager._prepare_payload_xfer`, wrapped). The source list runs over this rank's KV stacks at the peer's block of
each page, and the destination list over the decode's stacks at this rank's head columns. A send then moves one block of
each page to the peer over the peer's own source list (`send_kvcache`, wrapped). A page id names the same page on both
roles, so the allocator, the radix tree and the block tables need no translation.

A V page is `[tokens, head_dim]`, so the block of a V page is one contiguous run. A K page that the pool keeps as the
dim-major transpose `[head_dim, tokens]` gives one run for each dim row. A token-major K page gives one run, as V does,
so a token-major pair sends two runs for each layer of a page. The frame's K field must be the same on both roles.

**The bounded send and the drain.** A posted batch stays outstanding on the EFA queue pair until the send settles, and a
queue pair with too much outstanding work fails the transfer. So the prefill splits a send into batches of whole pages,
each of at most `SGLANG_NEURON_QWEN35_SCATTER_MAX_DESCS` descriptors (default 32768). With
`SGLANG_NEURON_QWEN35_SCATTER_DRAIN` (on by default), it settles the previous batch before it posts the next. One lock
for each transfer manager makes this one step across its transfer workers, so a source rank keeps one batch outstanding
across its peers. `SGLANG_NEURON_QWEN35_SCATTER_DUMP=1` logs each peer's descriptor lists and each send.

**The recurrent state.** Recurrent state splits by attention heads, not by CP
([Context parallelism](recurrent-state.md#context-parallelism)), so a decode rank's state shard is on one prefill
rank: the rank at its own attention-TP position. Every KV source of a decode rank reaches the last-chunk state send, and
SGLang pairs state slots by position, so without a gate each source writes the decode rank's one state slot. The scatter
gates `maybe_send_extra`: only the source at the decode rank's own position sends the state, and the other sources send
the KV and the aux data only. A pairing that gives a decode rank more than one KV source needs this gate on each
positional one-to-one transfer of the same send path.

Token CP and not `--dcp-size` serves such a model because the DCP page transfer from a whole-page prefill sends no
state, and refuses a pool with state components (`dcp_page_transfer.check_page_peer`). Under token CP the SGLang page
stays the allocation unit, and the recurrent state uses SGLang's equal-TP path.

### The host tier

HiCache's host (L2) tier ([Prefix (radix) cache](runtime.md#prefix-radix-cache)) over a token-CP MHA pool keeps each
rank's own shard of a page. A row of a CP pool holds the rank's block of the page in each KV head the rank holds. The
page id is the token page on every rank, since the allocator and the tree are the same on every rank. So a host page
mirrors one device page id and holds this rank's shard of it, and the cache controller's indices pass untranslated, as
on the MLA pool under token CP.

`mem_cache/host_pool.host_pool_cls` gives `NeuronCpHostKVPool` to a Neuron MHA pool whose page row holds fewer tokens
than the page; only a pool that serves HiCache under CP publishes such buffers. Its transfer unit is the page
(`unit_tokens`), not the row: the base class counts a row's tokens as one device page, which would spread one tree page
over several page ids. Its bytes per token are the shard's page bytes over the page's tokens, so `--hicache-ratio` keeps
its meaning (the host holds that multiple of the device pool's tokens), and `--hicache-size` buys pages at the shard's
size. A load runs whole at the pool's first layer, and the forward that reads it waits for it, as on every Neuron host
pool. A transfer that names the reserved page 0 is refused.

The base MHA pool refuses HiCache under CP (`_refuse_cp_hicache`). A pool kind that serves it publishes its rows
instead: Qwen3.5's `Qwen35SlotKVPool` publishes them with `SGLANG_NEURON_QWEN35_HICACHE=1`, and keeps the refusal
without it.
The CP host pool takes no remote DRAM tier: that tier moves whole host pages by pointer, and a rank's host page is its
shard of a page. So the remote DRAM backend (`--hicache-storage-backend dynamic`) is refused. The recurrent state goes
to the state pool's own host pool ([Host copies](recurrent-state.md#host-copies-hicache-l2-retraction-backup)), since
CP does not split it.

A PD decode runs HiCache only when its model's catalog entry names a switch that opts it in
(`ModelSpec.decode_hicache_env`) and the launch sets that switch to 1. The platform refuses
`--enable-hierarchical-cache` on every other decode.

### The traced helpers

Every helper in `parallel/context_parallel.py` takes the CP rank as a `[1]` int64 tensor and selects with a one-hot
mask, never a rank-indexed slice, so all ranks of a group trace one graph, and neuronx-cc never sees a rank-dependent
slice feeding a collective. The collective is passed in as a sum all-reduce, so the same code runs over a device group
or a CPU simulation of the ranks.

- `pcp_rows` / `pcp_gather`: a replicated sub-block's rows split contiguously over the group and gathered back with one
  sum all-reduce. Attention itself is not split this way: a contiguous causal split is not load-balanced.
- `cp_window`: a sequence's KV window at native slots, assembled on every rank from the block-cyclic shards (each rank
  contributes the slots it owns, zero elsewhere, then one sum all-reduce).
- `merge_partials` / `lse_of`: the exact merge of per-rank partial attention.

## Configuration

Token CP has no platform launch flag: the model declares it. Qwen3.5's decode reads its degree from
`SGLANG_NEURON_QWEN35_KV_CP`. `--page-size` must be a multiple of the degree (Qwen3.5's kv_cp=16 pages at 2048, one
128-token block per rank). `--attn-cp-size` stays 1.

The scatter's settings, which the prefill reads once at import:

| Setting | Default | What it does |
|---|---|---|
| `SGLANG_NEURON_QWEN35_SCATTER_MAX_DESCS` | 32768 | the descriptors of one posted batch, in whole pages |
| `SGLANG_NEURON_QWEN35_SCATTER_DRAIN` | 1 | settle the previous batch before the next one posts; `0` posts without a settle |
| `SGLANG_NEURON_QWEN35_SCATTER_DUMP` | unset | log each peer's descriptor lists and each send |

## Supported configurations

| Layout | Pool | Runtime | Role |
|---|---|---|---|
| Token CP on the paged MLA and DSA pools | `mem_cache/mla_pool.NeuronMLATokenToKVPool` | `NeuronModelRuntime` (its layout checks), or a model runtime that checks the pool itself | no model in the tree declares it |
| Token CP on the MHA pool | `NeuronMHATokenToKVPool` | `NeuronModelRuntime` | PD prefill; no model in the tree declares it |
| Qwen3.5 kv_cp=16 (`cp_size` = attention TP 16, both KV heads on every rank) | `Qwen35SlotKVPool`, an MHA pool kind | `NeuronModelRuntime`, the model's CP decode graphs | PD decode, behind a whole-page prefill through the scatter |

## Limitations

| Combination | Behaviour | Reason |
|---|---|---|
| `--attn-cp-size` above 1 | refused at launch (`platform.refuse_unserved`) | no runtime of the plugin serves SGLang's attention-CP group |
| `NeuronModel.kv_cp_size` above 1 with a page it does not divide or a degree other than the attention-TP size, on an aggregated runtime, on a decode runtime of a model without `kv_cp_decode`, over the hybrid SWA pool, or over a pool that publishes another layout | `ValueError` when the runtime is built | the runtime addresses token CP on a PD prefill, or on a model's own CP decode, over a pool of the same layout only |
| A token-CP layout on the MHA pool with DCP, with paired layers, or under HiCache | `ValueError` when the pool is built or when HiCache asks for its buffers | the DCP placement, the paired-buffer kernel and SGLang's host pool address whole pages per rank; a pool kind that serves the [host tier](#the-host-tier) lifts the HiCache refusal behind its model's switch |
| A token-CP prefill into a decode of another degree, CP rank or block | the prefill refuses the decode peer at registration (`sglang_patches/pd_kv_cp_layout`) | SGLang's transfer copies blocks to the decode rank of the same attention-TP rank |
| A whole-page prefill into a CP decode outside the scatter's geometry, or with another K layout | the prefill refuses the decode peer at registration (`kv_cp_scatter.geometry_problem`, `pd_kv_cp_layout.cp_layout_problem`) | the scatter needs equal attention TP, whole blocks and one K layout ([the refusals](#a-whole-page-prefill-into-a-cp-decode)) |
| An MHA token-CP prefill into an MHA decode | refused: a decode runtime without `kv_cp_decode` refuses the layout, and the prefill refuses a decode without it | the MHA decode graphs address whole pages; no model in the tree runs an MHA CP prefill (Qwen3.5 refuses `SGLANG_NEURON_QWEN35_KV_CP` on its prefill) |
| An L3 storage backend under the CP host pool | refused when the host pool is built | a rank's host page is its shard of a page, which the remote DRAM tier does not move |

## Upstream SGLang interactions

- SGLang builds the KV pool after the model, through the platform's pool class; the model's declaration
  (`kv_layout.declare`) is how the layout reaches the pool.
- SGLang's PD transfer copies pages block for block between ranks of the same attention-TP rank;
  `sglang_patches/pd_kv_cp_layout` pins the functions it wraps against SGLang 43881a7.
- The scatter wraps SGLang's NIXL `_prepare_payload_xfer`, `send_kvcache` and `maybe_send_extra` on the prefill and
  `CommonKVManager._resolve_rank_mapping` on the decode, and pins them and the sources it calls against SGLang 43881a7
  (`sglang_patches/kv_cp_scatter`). A peer that is not a scatter pair takes upstream's paths.

## Implementation

| Module | Responsibility |
|---|---|
| `sglang_neuron/model_executor/kv_layout.py` | `KVLayout` (`cp_size`, `cp_rank`, `kv_heads_per_rank`, `k_token_major`), `declare`, `current` |
| `sglang_neuron/mem_cache/memory_pool.py` | the MHA pool's token-CP blocks and published layout, its refusals |
| `sglang_neuron/mem_cache/cp_pages.py`, `mem_cache/mla_pool.py` | block-cyclic page ownership; the MLA and DSA pools under token CP or DCP |
| `sglang_neuron/model_executor/graph_runner.py` | the runtime's layout checks, `_cp_rank_slots`, `_cp_prefill_swa_window`, the CP prefill inputs |
| `sglang_neuron/models/base.py` | `NeuronModel.kv_cp_size` (1 by default) |
| `sglang_neuron/parallel/context_parallel.py` | `pcp_rows`, `pcp_gather`, `cp_window`, `merge_partials`, `lse_of` |
| `sglang_neuron/sglang_patches/pd_kv_cp_layout.py` | the layout frame (`CpLayout`), `own_layout`, `scatter_pair`, the registration refusals |
| `sglang_neuron/sglang_patches/kv_cp_scatter.py` | the scatter's geometry check, its descriptor lists, the bounded send and the drain, the state owner gate, the decode's rank mapping |
| `sglang_neuron/mem_cache/host_pool.py` | `NeuronCpHostKVPool` and the host pool choice (`host_pool_cls`) |
| `sglang_neuron/models/qwen35/kv_pool.py` | `Qwen35SlotKVPool`: the kv_cp blocks of both KV heads, the token-major K pages, the CP sink row, the host-tier opt-in |
