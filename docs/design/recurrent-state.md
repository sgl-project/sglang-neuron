# Recurrent State

How the platform holds, reuses and transfers the per-request state of recurrent layers: Kimi-K3's Kimi delta
attention (KDA) layers, Qwen3.5's gated delta net (GDN) layers and any later SSM.

## Overview

A recurrent layer carries a fixed-size state per request instead of a growing KV cache: a recurrent (temporal) matrix
and a short convolution window per layer. Upstream SGLang already owns all of the bookkeeping for such state:
`HybridReqToTokenPool` maps each request to a state slot, `MambaSlotAllocator` hands slots out, the radix cache's
mamba component keeps checkpoint slots on tree nodes and copies one into a request on a match, and the PD connector
registers the state tensors and sends one slot per request. The platform does not add a second allocator, tree or
transfer protocol. It supplies four things:

* a declaration of the state each model holds (`StateSpec`), checked against upstream's resolved cache parameters and
  against the model contract's layer kinds;
* the pool class SGLang builds (`NeuronStatePool`, a `MambaPool`), whose storage is the model kernel's layout and whose
  slot operations are the model kernel's (`StateOps`), with an eager torch reference for CPU mode;
* the rules that make a radix checkpoint exact (which token boundaries hold exact state, and under which identity);
* refusals, at launch and at PD registration, for every pairing that upstream would corrupt silently or fail per
  request.

One framework serves every recurrent family. A model chooses its family, its components and its kernels; nothing in
the framework is specific to KDA or GDN beyond two convenience constructors.

## Key concepts

**State component.** One per-slot tensor of every state layer: the recurrent `temporal` component and one or more
`conv` components, in upstream's order (`MambaPool.State`: `temporal`, `conv[i]`). A component has a logical shape
(upstream's per-slot shape from `cache_params.shape`), a dtype, and optionally a physical shape: the layout the
model's kernel reads, with the same element count, for example a conv window of `(3, taps, heads * dim)` where
upstream says `(taps, 3 * heads * dim)`. Kimi-K3 declares its conv window channel-major, `(3 * heads * dim, taps)`
(`kda_spec(conv_layout=CHANNEL_MAJOR)`), which still slices by channel; Qwen3.5's GDN uses upstream's layout.

**State spec.** `StateSpec(family, components, layer_ids, working_rows, track_chunk, tp_size, numerics)`: the family
(`gdn`, `kda`, `mamba2`), its components, the global ids of its state layers in pool buffer order, whether the kernel
advances state in a separate working bank, the token grid on which the prefill kernel can capture state inside a
forward, and the numerics profile. Its digest covers everything a PD peer or a cached checkpoint must agree on and
nothing rank-local.

**Canonical slot.** A row of the pool's tensors that owns one request's or one tree node's state. Slot 0 is upstream's
padding slot and stays zero; slots 1..size are allocated by `MambaSlotAllocator`. The pool adds one trailing sink slot
that a kernel's padded bucket may write and that no reader ever sees.

**Working bank.** For a kernel that advances state in rows separate from the canonical slots, `working_rows` rows per
layer (row 0 scratch). A forward loads canonical slots into rows, advances the rows, and publishes rows back into
canonical slots once the forward is accepted. A kernel without a working bank (Qwen3.5, Kimi-K3's prefill) advances
canonical slots in place.

**Checkpoint.** State at a token boundary held in a slot the radix tree owns. Only a copy taken at a boundary is a
checkpoint; a live request slot has already moved past every earlier boundary.

**State ops.** The five slot operations a pool runs on its tensors: `clear_slots`, `copy_slots`, `load_rows`,
`publish_rows`, `drain`. The model's kernel package provides them, or a model whose kernels advance the canonical slots
in place (no working bank) shares `CompiledStateOps`, the default for such a pool: one compiled graph per (operation,
state tensor, bucket of 8 / 16 / 32 / 64 / 128 / 256 slots), which the pool warms when it builds its tensors, so a live step never
compiles. `TorchStateOps` is the reference (eager torch; CPU mode). A state tensor past 4 GiB is refused at
construction: an indexed DMA cannot address past it.

## How it works

### Declaring state

The model contract (`models/contract.py`) gives every layer a kind. HF `layer_types` value `linear_attention`
(Qwen3.5) becomes a layer with state family `gdn`; `kda`, or a Kimi-Linear config's 1-based
`linear_attn_config.kda_layers`, becomes family `kda`. Other SSM kinds (`mamba`, `mamba2`, `ssm`) stay refused until a
model declares components for them. A model on the declared path names each layer's family in `state_families`.
Recurrent layers hash into the fleet digest with their family, so every rank must resolve the same layer kinds;
attention layers keep the identity they always had.

The model declares its `StateSpec` (`model.state_spec`) built from upstream's cache parameters
(`state_spec.from_cache_params`) or from the constructors `gdn_spec` / `kda_spec`, which reproduce upstream's
`Mamba2StateShape.create` and `KimiLinearStateShape.create` per rank. At graph-runner build
`state_pool.check_model_state` proves that the contract's recurrent layers are exactly the spec's layers, in one
family, and that the pool SGLang built is bound to the same spec digest.

### Pool factory

SGLang builds the state pool in `HybridReqToTokenPool._init_mamba_pool` from the class attribute
`HybridReqToTokenPool.mamba_pool_cls`, for the target, draft and PD decode request tables alike. One seam owns that
hook for every model (`sglang_patches/state_pool_class`): while SGLang's KV configurator builds a model's pools it sets
`mamba_pool_cls` to the class the model declares, a megakernel model's `PoolLayout.state_pool_class` (the Neuron
platform answers `get_megakernel_state_pool_cls` with it) or else a platform-runtime model's
`NeuronModel.state_pool_class` (`NeuronStatePool.bind(spec, ops)` set on the model object before SGLang builds the
pools). Two different declarations on one model are refused. A megakernel model declares its state in its layout:

```python
cls, entries = state_pool.layout_parts(spec, ops)   # NeuronStatePool.bind(spec, ops) and its pool entries
layout = PoolLayout(family=..., entries=(*attention_entries, *entries), state_pool_class=cls, ...)
```

`layout_parts` also gives the pool entries the model's kernel binds: each component's whole physical tensor
(`[layers, slots + 2, *shape]`, padding slot and sink included) and the working bank, kinds `recurrent_state` and
`conv_state`, global layer ids. A model whose request table got SGLang's own `MambaPool` (no layout, or a layout
without a state class) is refused at graph-runner build (`state_pool.check_model_state`), naming its state layers.

`NeuronStatePool.__init__` takes `MambaPool`'s keyword arguments and:

1. refuses every upstream option it does not implement when it is on: speculative state, ReplaySSM rings, the
   envelope layout, the memory saver, and any option it does not know (a newer SGLang's new switch is refused, not
   ignored);
2. takes the stage's layers from `mamba_layer_ids` (global ids), refusing one the model did not declare;
3. checks upstream's resolved `cache_params` against the spec: layers, temporal and conv element counts, dtypes
   (upstream resolves them from the checkpoint, `--mamba-ssm-dtype` and `SGLANG_MAMBA_CONV_DTYPE`; the refusal names
   the flag) and family (`is_kda`); a mismatch is a launch refusal, never a cast; and refuses a radix cache whose
   checkpoints the kernel cannot make exact (`state_pool.radix_refusal` over SGLang's resolved `RadixSettings`);
4. allocates each component as `[layers, size + 2, *physical shape]` (padding slot, slots, sink) and, with a working
   bank, `work.<component>` as `[layers, working_rows, *physical shape]`, on the rank's Neuron device
   (`runtime_init.neuron_device()`: SGLang builds its request tables on the host under the platform, so the device it
   passes is the host's);
5. presents `mamba_cache = MambaPool.State(temporal=..., conv=[...])` over slots `[0, size]`, so upstream's layer views
   (`mamba2_layer_cache`), sizing (`mem_usage_bytes`) and PD entries (`get_contiguous_buf_infos`,
   `get_state_layer_ids`) read upstream's layout unchanged;
6. takes upstream's conv slice axis and q/k/v shard groups from `cache_params.shape` (KDA's window-first conv slices
   on axis 1, GDN's channels-first on axis 0) for components in upstream's per-slot shape, so SGLang's PD transfer
   slices the state between attention TP sizes (a TP16 prefill to a TP32 decode) as it slices its own pool's; a
   kernel-layout conv has none (its entries carry slice axis None).

### Slot operations

`clear_slots(indices)` and `copy_from(src, dst)` keep upstream's names, so SGLang's own callers (fresh-slot clears,
copy-on-write on a match, the deferred COW/clear before a forward) reach the model's kernels. Each validates host slot
ids (in `[1, size]`; a load may read slot 0, the zero state), requires copy destinations unique and disjoint from
sources, and calls the model's `StateOps`. The reference semantics (`TorchStateOps`):

| Operation | Effect |
|---|---|
| `clear_slots(slots)` | every component of every layer of each slot reads zero |
| `copy_slots(src, dst)` | slot `dst[i]` becomes a bit copy of slot `src[i]` (NaN payloads, signed zero, subnormals included); sources unchanged |
| `load_rows(slots, rows)` | working row `rows[i]` takes canonical slot `slots[i]` |
| `publish_rows(slots, rows)` | canonical slot `slots[i]` takes working row `rows[i]` |
| `drain()` | every operation submitted so far has completed or raised |

A kernel implementation pads to its compiled buckets itself: padding reads slot 0 and writes the sink. Device
implementations come from `kernels/<model>/` (fused NKI clears, compiled per-tensor copies, precompiled
copy/load/publish programs); the framework never runs an eager device index write on a Neuron tensor.

### Working and checkpoint slots

Without a working bank, the request's canonical slot is the kernel's state operand: a forward reads and writes it in
place, and a checkpoint is a copy of it into a tree-owned slot.

With a working bank, canonical slots are never the kernel's mutable operand. Before a forward the request's
canonical slot is loaded into its working row; the forward advances the row; after every rank accepts the forward's
output (finite, no overflow, the forward's own checks) the row is published back. A rejected forward publishes nothing
and the canonical checkpoint stays intact. A request keeps one stable working row from admission to its last phase, so
a later phase never overwrites an earlier request's unpublished row. Transfer descriptors and cache plans describe
canonical slots only, never working rows.

### Copy-on-write on a radix branch

The radix cache's mamba component keeps a slot per tree node. On a match it gives the request a fresh slot and copies
the node's checkpoint into it (`copy_from(node_slot, request_slot)`): copy-on-write, so the tree's checkpoint is never
the slot a forward advances. The copy runs through the model's `copy_slots`; it may be deferred into the next forward
as long as it reads the checkpoint slot, not a live row.

The alternative, a copy-on-match prologue inside the decode kernel that copies a donor request's row into the new
request's row, is valid only when the donor's row holds the branch position. A live donor's state summarizes its
current position, which is past the branch point once it has decoded. The framework therefore copies from tree-owned
checkpoint slots (upstream's native COW) and admits an in-kernel copy only as an implementation of `copy_slots` whose
source is such a slot.

### Checkpoint boundaries

A forward over tokens `[prefix, end)` leaves exact state at `end`. Inside the forward the kernel can capture state only
on its own tile grid, relative to the forward's start (`track_chunk`; 0 for a kernel that cannot capture inside a
forward). Upstream's scheduler requests an intermediate capture with a tracking hint of `aligned + 1`
(`mamba_track_seqlens`); the checkpoint depth is the aligned length, `((track - prefix) // chunk) * chunk`
(`state_checkpoint.aligned_track_length`, the arithmetic of upstream's `ForwardBatch.mamba_track_aligned_lens`, on host
integers). Treating the hint as the depth would attach the snapshot to the wrong token.

| Boundary | Exact when |
|---|---|
| a forward's end | always (the state the forward leaves in its slot) |
| `prefix + k * track_chunk` inside a forward | the kernel takes the capture control and `track_chunk > 0` |
| any other boundary | never; it needs another forward that ends there |

On the page grid `state_checkpoint.radix_problem` states the launch rule (the pool refuses at construction): with upstream's tracking buffer the checkpoint interval must be a whole number of
`track_chunk` tiles (and `track_chunk > 0`); without it upstream keeps forward-end state and needs page size 1.

A tracked capture that lands on a forward's end is the state the forward leaves, copied after the accepted forward
into the request's tracking slot (`NeuronStatePool.capture_tracks`, a `copy_slots`); one inside a forward is the
kernel's own, and is refused for a kernel without capture (`state_checkpoint.forward_end_tracks`).

### The chunk-keyed tree

With the chunk-keyed radix tree (`sglang_patches/chunk_keyed_tree`: tree page = prefill chunk) and one request per
prefill forward on the chunk grid (`sglang_patches/prefill_chunk_grid`), recurrent state rides the chunk keys through
upstream's tracking buffer, with every checkpoint at a forward end:

* upstream's checkpoint grid is `lcm(mamba cache chunk, tree page)`; with the chunk a whole number of mamba cache
  chunks it is the chunk itself;
* a forward covers exactly one chunk from an on-grid prefix, so the tracked depth upstream's scheduler picks
  (`prefix + floor(extend / grid) * grid`) is the forward's end, and a radix branching point, which lies on the grid,
  is never strictly inside a forward;
* the prompt's last partial chunk tracks nothing (`extend < grid`), and the tree's inserts stop at the prompt's last
  whole chunk, so no checkpoint off the chunk grid enters the tree.

So a kernel without in-forward capture (`track_chunk 0`, Qwen3.5's GDN) serves the chunk-keyed tree with forward-end
copies alone. The tree build accepts SGLang's hybrid SSM pools only for a model whose catalog entry keeps chunk state
checkpoints (`ModelSpec.chunk_state_checkpoints`, exported as `ServingPolicy.chunk_state_checkpoints`, which also has
the platform serve upstream's tracking buffer, `support_mamba_cache_extra_buffer`); such a model is refused SGLang's
default page-keyed tree, and a pipelined launch of it needs `--max-mamba-cache-size` (stages size their state pools
from their own free memory). The pool refuses at construction a launch on the chunk keys without upstream's tracking
buffer (its no-buffer strategy keys per-token pages) or with a chunk that is not a whole number of mamba cache chunks
(`state_checkpoint.chunk_grid_problem`, through `state_pool.radix_refusal`). A forward that spans several chunks (off the chunk grid)
would track an inner boundary and is refused at capture for such a kernel.

### A chunk from a cached prefix

A hybrid model whose prefill runs one request's chunk per forward over a paged KV pool resumes a radix hit as it resumes
a cold prompt's next chunk. Qwen3.5's giga prefill does this today (`sglang_neuron/models/qwen35/prefix.py`). A hit at
chunk `k` copies the node's checkpoint into the request's slot (the copy-on-write above), and the forward then runs the
cold prompt's own chunk `k` over the shared pages. Each forward takes these runtime inputs:

- the request's state slot, which the kernel reads as the initial state and writes with the final state, and the
  tracking slot that also takes the final state: SGLang's `mamba_track_indices` when the batch asks for a checkpoint,
  else the pool's sink;
- the pool page of each page the chunk writes, and the request's pages before the chunk (its prior) with the prior's
  length, both padded with page 0;
- the chunk's valid length in its bucket.

The prefill program is keyed by its token bucket and its prior tier: the number of prior tokens its attention can read,
a fixed set of tiers below the context and then the context in whole pages. The cached length is a runtime input, so a
cold chunk and a hit of the same bucket and tier run one program.

The bucket pads the last chunk of a prompt. The recurrent kernel must leave the state of the last valid row: the pad
rows add nothing to the state, and the conv window ends at the last valid row. The pad rows' K and V go to the
request's own last page or to page 0, and the attention masks both. The first output token comes from the last valid
row, not the last row of the bucket.

The host refuses a chunk that does not start on a whole page, a chunk longer than its bucket, a prior longer than the
program's tier, and a checkpoint that the batch asks for inside the forward (`state_checkpoint.capture_boundary` at
`track_chunk 0`). Under DP attention every rank runs one program in a step, since the MoE and the LM head span every
rank. So the prefilling replicas agree on the bucket and the tier from the DP-attention cohort snapshot
([Serving a whole-model megakernel](megakernel.md#serving-a-whole-model-megakernel)).

### Checkpoint identity and restore

Exact bytes are exact only for the numerics that produced them. A checkpoint (`StateCheckpoint`) is keyed by a
`CacheIdentity`: model revision, weights revision, numerics profile, state-layout digest, stage and shard, plus a digest
of its token prefix. A collective accumulation order is part of the numerics profile: a checkpoint written under one
order is a miss under another. A match is clamped to the deepest checkpoint at or below the radix match that is ready
(its producing operation completed) and has this identity and prefix (`longest_checkpoint`); with none, the request
starts from zero state and recomputes the prefix.

Restore is the copy-on-write above: the checkpoint slot is copied into the request's slot, and for a working-bank
kernel then loaded into the request's working row. A restored request continues from the boundary with its KV pages
for the same prefix; state and KV must come from the same boundary.

### Drain before reset

A device implementation may return before its copies and publications have completed. Before SGLang resets the state
slots, the pool is drained: `HybridReqToTokenPool.clear` resets through `StateStorage.clear`, which
`sglang_patches/state_pool_class` drains; the PD decode's `HybridMambaDecodeReqToTokenPool.clear` resets its slot
allocator directly, which `sglang_patches/decode_state_drain` drains. A drain that raises stops the reset: no slot goes to a new owner while an operation that failed may still
write it. Any failed operation or drain quarantines the pool; every later operation raises with the first failure, and
a later successful fence cannot hide an earlier failure.

### PD transfer of state components

Upstream registers the pool's state as one `StateType.MAMBA` component: one entry per (tensor, layer), field-major
(each conv tensor's layers, then the temporal tensor's layers), each with its global layer id, item length (one slot of
one layer) and sliceable head dimension. At a request's last chunk the prefill pairs its entries with the decode's by
layer id, in order of occurrence, and at equal attention TP writes `item_len` bytes from `src + item_len * src_slot` to
`dst + item_len * dst_slot`; at different attention TPs it slices each entry by its head dimension.

The prefill refuses, when a decode rank registers (`NixlKVManager._add_remote_peer`, through
`sglang_patches/nixl_state_components` and `disaggregation/state_transfer.state_entries_problem`):

| Pairing | Upstream behaviour | Here |
|---|---|---|
| decode registered fewer state components than the prefill sends | every request fails at its last chunk | refused at registration |
| a paired entry with a different slot size (another dtype or shape) | WRITE with the prefill's stride into the decode's slot grid, silently | refused at registration |
| a prefill state layer the decode has no entry for | every request fails at its last chunk | refused at registration |
| different attention TPs and an entry stored in a kernel layout | slices the wrong axis, or fails per request | refused at registration |

A component stored in a kernel layout reports upstream's "not head-sliceable" marker (dimension 0), which is what
makes the last refusal possible. The state travels with the request's last KV chunk and carries the state after the
whole prompt; the decode request runs only when both have landed. `state_transfer.slot_runs` is the equal-TP byte plan.

**Quarantine.** SGLang frees a decode request's state slot only in the release of its KV row (`release_kv_cache`,
through the radix cache's mamba component), which `DecodeTransferQueue._release_request` is the only caller of for a
transfer. A failed or aborted transfer that the KV quarantine holds (`sglang_patches/pd_decode_kv_quarantine`) is not
released until every prefill rank acked its drain, so its state slot stays allocated exactly as long as its KV pages
and no late WRITE of the state component lands in a slot another request owns. A request that fails in the prealloc
queue, before its decode sent the prefill its slot ids, frees its slot at once: nothing can be written into it. The
next owner of a freed slot gets it cleared before its first forward.

No platform path re-lays a kernel-layout entry out across attention TPs: such a pairing is refused, with the reason
and the prefill's TP to pair instead. Qwen3.5's PD pairs one attention TP on both roles (its replica), which upstream's
equal-TP send serves whole slot by whole slot.

**Addresses.** The pool registers its entries with the Neuron runtime's device addresses
(`NeuronStatePool.get_contiguous_buf_infos`, `backend.device_io.device_address`): torch-neuronx's `data_ptr()` is a
placeholder that NIXL cannot register and a WRITE would miss.

### Host copies (HiCache L2, retraction backup)

A host copy moves whole slots: every component of every state layer, in the pool's physical layout and dtype, so a
reloaded slot is the bytes the kernel left. A working-bank pool moves its canonical slots only.

- `NeuronStatePool.get_device_pool_infos` gives upstream's `MambaStateBufferInfo` over `mamba_cache` (slots
  [0, size], no sink, no working bank), validated, with global layer ids; `host_capacity_tokens` is the slot count and
  `host_capacity_bytes` the canonical slots' bytes, which is what `--hicache-size` is split by.
- `build_mamba_host_pool` (wrapped, md5 pinned) builds `NeuronStateHostPool` (mem_cache/state_host_pool) for a Neuron
  pool in place of upstream's `MambaPoolHost`, whose transfers are CUDA/NPU kernels. The host keeps
  `[layers, host slots, *shape]` per component; a transfer is one `nrt_tensor_{read,write}_batch` call through main's
  `TransferEngine` with one region per (component, layer), so slots consecutive on both sides are one op per region.
  Its jobs complete the same HiCache events as the KV pool's: a store waits for the graphs launched before it, a load
  is submitted whole at the first state layer, and the forward that reads it waits (`set_consumer_waits`).
- `get_cpu_copy` / `load_cpu_copy` (upstream's decode retraction backup) copy a request's slot to and from host
  tensors synchronously through the same op records, in upstream's `(conv list, temporal)` form.

Upstream's host-only checkpoints (`SGLANG_OPT_MAMBA_HOST_ONLY_CHECKPOINT`, with HiCache) store through the same host
pool and keep radix checkpoints off the device entirely.

### Checkpoint store

Upstream keeps a radix checkpoint in a slot of the request pool, so cached prefixes and running requests share one
slot budget (a Qwen3.5 rank's slot is about 12 MB at a 16-rank replica, `prefix.checkpoint_slot_bytes`). `--enable-int8-mamba-checkpoint` gives the radix a store of its own;
for a Neuron state pool the wrapped `maybe_init_int8_mamba_checkpoint_pool` builds `NeuronStateCheckpointPool`
(mem_cache/state_checkpoint_pool) with upstream's `MambaCheckpointPool` surface, driven by upstream's own calls: an
insert commits the request's slot into the store (`store_from_active`) and keeps no request slot, a hit copies the
checkpoint into the request's fresh slot (`load_to_active`, upstream's deferred COW), a full store evicts the least
recently used checkpoint, and a radix reset clears it. `--int8-mamba-ckpt-size` sizes it in checkpoints (default
twice the request pool). The store is exact: every component in the pool's dtype and layout, copied with the model's
`copy_across` (`CompiledStateOps`: one compiled graph per tensor shape and bucket 1/4/16, warmed when the store binds to
its pool), so a hit restores the bytes bit for bit, as the Neuron radix cache's bit-identical-hit contract needs;
upstream's int8 rounding is not applied (its scale axis is a different physical axis per kernel layout). Banks keep
each tensor under the 4 GiB an indexed DMA addresses. On a device it is sized once the pools are built, from the HBM
they left less a reserve for loading the programs (`SGLANG_NEURON_STATE_CKPT_RESERVE_GB`, 2 GiB): at most twice the
request pool's slots, an explicit `--int8-mamba-ckpt-size` that would eat the reserve refused.

### Pipeline stages

Each stage builds a pool for the state layers it holds; SGLang passes the stage's `mamba_layer_ids` (global ids) and
the pool keeps them (`StateSpec.stage`). Every key derived from state (buffer index lookup, PD pairing, checkpoint
identity) uses the global id; only module paths are stage-local. A PD pairing between a PP prefill stage and a decode
holding more layers pairs by layer id.

### Context parallelism

State is never sharded by context parallelism. Under `--dcp-size D` (Kimi-K3's PD prefill) or a declared token-CP KV
layout (Qwen3.5's kv_cp decode) the KV pages are split across the CP group, but a request's state is sharded only by
attention heads, over attention TP, as without CP: every rank of a DCP group holds its own heads' state for the whole
sequence, advanced by the recurrent kernel over all positions. CP touches the pool in one place: under DCP the radix
tree's page is the allocator's widened `D x page` (`StateSpec` checkpoint grid, `state_pool.py`), so checkpoint
boundaries fall on whole virtual pages.

A PD pairing of a CP prefill with a CP decode at another attention TP moves state by heads, decode rank `g` taking its
heads from the prefill rank that held them; the one such path is the DCP stripe transfer of Kimi-K3's PD
(`disaggregation/dcp_stripes.py`: a prefill of `Cp` DCP ranks to a decode of a larger multiple `Cd`, decode rank `g`'s
state from prefill rank `g // (Cd / Cp)`, by the page rule the decode announces). Every other cross-attention-TP
pairing of a kernel-layout component keeps the refusal below.

A whole-page prefill that feeds a token-CP decode at equal attention TP (Qwen3.5's kv_cp scatter) moves the state by
SGLang's equal-TP path, slot by slot. Each decode rank then has one KV source for each KV head shard, but its state
shard is on one of them, the prefill rank at its own attention-TP position. So only that source sends the state, and
the others send the KV and the aux data only
([The recurrent state](context-parallelism.md#a-whole-page-prefill-into-a-cp-decode)).

### Rank agreement

Every decision the ranks of an attention-TP group make about state is made from state that advances at the same
scheduler points on every rank: slot allocation and release are upstream's allocator calls in scheduler order; clears,
copies, loads and publications are issued from the forward that every rank runs; a checkpoint becomes matchable at the
scheduler step that processes its producing forward's result, not when a rank's device happens to finish; the PD
refusals read only registration data every rank receives. A rank whose operation fails raises and stops serving; no
rank takes a different branch on its own completion timing.

## Configuration

There are no environment variables. A model opts in by declaring `state_spec`, its layer kinds (derived or
`state_families`) and its state pool class: a megakernel model's `PoolLayout.state_pool_class`
(`state_pool.layout_parts(spec, ops)`, or the model's own form of it, as Qwen3.5's `state.layout_parts`), or a
platform-runtime model's `NeuronModel.state_pool_class` (`NeuronStatePool.bind(spec, ops)`, as Kimi-K3's prefill).
Upstream's own switches stay upstream's: `--mamba-ssm-dtype` / `SGLANG_MAMBA_{CONV,SSM}_DTYPE` must resolve to the
dtypes the model declares, and `--max-mamba-cache-size` sizes the slot count.

## Supported configurations

| Family | Model | Components (per rank) | Working bank | In-forward capture |
|---|---|---|---|---|
| `gdn` | Qwen3.5 | temporal `(v_heads/tp, head_v, head_k)` FP32; conv `(conv_dim/tp, kernel - 1)` BF16 | no | no (`track_chunk 0`) |
| `kda` | Kimi-K3 (PD prefill) | temporal `(heads/tp, dim, dim)` BF16; conv channel-major `(3 * heads/tp * dim, kernel - 1)` BF16 (`--mamba-ssm-dtype bfloat16`) | no | no (`track_chunk 0`) |

The Kimi-K3 PD decode keeps its KDA state in pages of its MLA buffers, which its verify program reads and writes. Its
request table's state pool (`models/kimi_k3/decode/sgl_pools.state_pool_class`) is a `MambaPool` with SGLang's slots
and no state tensors of its own, not a `NeuronStatePool`.

## Limitations

- Speculative decoding state, ReplaySSM rings, the envelope layout, the memory saver and sibling slot states (PLE) are
  refused. State HiCache is L2 only (no storage backend) and keeps no MTP draft state.
- The checkpoint store keeps exact copies (no int8), and upstream's launch rules for it hold: not with HiCache,
  LMCache, unified memory or a `--radix-cache-backend` (so not with the chunk-keyed tree).
- The pool converts device slot tensors to host lists; a model should keep SGLang's request-to-slot mapping on the host
  so that no clear or copy reads device memory.
- PD across attention TPs is refused at registration for a component in a kernel layout; the DCP stripe transfer
  (`disaggregation/dcp_stripes.py`) carries state across attention TPs on a path of its own (Context parallelism).
- State is not sharded by context parallelism (Context parallelism): a CP group's ranks each hold their heads' state for
  the whole sequence.
- Decode-side radix reuse of recurrent state (`--disaggregation-decode-enable-radix-cache`) keeps upstream's behaviour:
  a decode receives only the state after the whole prompt, which is a checkpoint at the prompt's end only.
- Graph-runner metadata for recurrent layers (which layers get attention metadata) is the model's; the framework only
  proves the declaration agrees with the contract.
- Exact state transport does not make a continuation numerically identical to an uninterrupted run: a continuation
  that repacks tokens across ranks changes collective accumulation order. A model's cached serving is qualified with
  its continuation golden under a fixed numerics profile, not by byte-exact restore alone.

## Upstream SGLang interactions

The served SGLang tree is `43881a7`.

| Upstream piece | Use | How the platform reaches it |
|---|---|---|
| `MambaPool` | base class; constructor kwargs; `State`; `clear_slots`, `copy_from`, `_iter_transfer_state_entries`, `get_contiguous_buf_infos` | subclassed; a constructor option the pool does not implement is refused when on |
| `HybridReqToTokenPool.mamba_pool_cls`, `_init_mamba_pool`, `StateStorage.clear` | pool selection and drain (the state pool class seam) | md5 pinned by the seam |
| `HybridMambaDecodeReqToTokenPool.clear` | the PD decode's reset, drained (`decode_state_drain`) | md5 pinned |
| `MambaSlotAllocator` | slot allocation | through `StateStorage` |
| radix mamba component | checkpoints, COW, eviction | `MambaComponent(cache, params, slot)` through `StateStorage` (`take_state_slots`, `copy_state`, `release_state`) |
| `ForwardBatch.mamba_track_aligned_lens` | tracking-hint arithmetic | the same arithmetic on host integers (`state_checkpoint.aligned_track_length`) |
| `setup_state_kv_args`, `StateType.MAMBA`, `build_transfer_entry_pairs`, `NixlKVManager._send_mamba_state(_slice)` | PD state entries and pairing | `setup_state_kv_args(kv_args, storage, ...)` over `storage.wire_entries()` |

SGLang has no platform hook for the state pool class. The state pool class seam stands in for an upstream candidate: a
`mamba_pool_cls` argument, the platform's `get_megakernel_state_pool_cls`, and a `MambaPool.drain` call from
`StateStorage.clear`.

## Implementation

| File | Contents |
|---|---|
| `sglang_neuron/mem_cache/state_spec.py` | `StateComponent`, `StateSpec`, `from_cache_params`, `gdn_spec`, `kda_spec`, upstream agreement check |
| `sglang_neuron/mem_cache/state_ops.py` | the `StateOps` hook, `CompiledStateOps` (the shared device implementation), `TorchStateOps` reference, `check_ops`, `oversized_tensors` |
| `sglang_neuron/mem_cache/state_pool.py` | `NeuronStatePool` (bind, storage, slot operations, quarantine, PD entries, forward-end captures), `RadixSettings` / `radix_settings` / `radix_refusal` (the radix launch check), `layout_parts`, `state_pool_of`, `check_model_state` |
| `sglang_neuron/sglang_patches/state_pool_class.py` | the one state pool class seam (megakernel layout or `NeuronModel.state_pool_class`) and the drain before `StateStorage.clear` |
| `sglang_neuron/mem_cache/state_host_pool.py` | `SlotRegions`, `copy_slots_now`, `NeuronStateHostPool`, the `build_mamba_host_pool` hook |
| `sglang_neuron/mem_cache/state_checkpoint_pool.py` | `NeuronStateCheckpointPool`, `bank_geometry`, the `maybe_init_int8_mamba_checkpoint_pool` hook |
| `sglang_neuron/mem_cache/state_checkpoint.py` | tracking-hint alignment, exact boundaries, `CacheIdentity`, `StateCheckpoint`, `longest_checkpoint`, `radix_problem`, `chunk_grid_problem`, `forward_end_tracks` |
| `sglang_neuron/disaggregation/state_transfer.py` | `state_entries_problem`, `slot_runs` |
| `sglang_neuron/sglang_patches/chunk_keyed_tree.py`, `sglang_neuron/mem_cache/chunk_keyed_tree.py` | a model that keeps chunk state checkpoints: refused the default tree, PP needs `--max-mamba-cache-size`, the tree accepts its SSM pool |
| `sglang_neuron/sglang_patches/decode_state_drain.py` | the PD decode table's reset drains the state pool |
| `sglang_neuron/sglang_patches/nixl_state_components.py` | the registration refusals |
| `sglang_neuron/models/contract.py` | recurrent layer kinds and `check_state_spec` |

## Lessons

Each of the framework's rules guards against a failure or follows from a design choice:

- **Native SGLang bookkeeping, model-owned bytes**: the pool uses SGLang's request table, slot allocator and native
  copy and reset, with no second allocator or LRU. The pool is per model, chosen from the model's declaration, and it
  checks the global layer order, the shapes and the dtypes before it allocates. A dtype that differs from the
  declaration is refused, not cast.
- **Immutable zero and a sink**: padding must read a state that is always zero and write somewhere no reader sees.
  Slot 0 stays zero and the sink trails the visible slots, so upstream's views need no remap of the slot ids.
- **Drain before reset; quarantine on failure** (the `StateStorage.clear` drain): an asynchronous publication must not
  land after its slot changes owner, and a later successful fence must not hide an earlier failure.
- **Checkpoints are dedicated slots, never live rows**: a live row cannot serve an earlier prefix; a radix split
  reuses an existing checkpoint and never invents one.
- **Tracking hint is not depth**: upstream encodes an intermediate capture as `aligned + 1`; the depth is the aligned
  length. Hint 3073 for prefix 2048 + 1535 tokens is boundary 3072.
- **Exact boundaries follow the kernel's tile grid**: a kernel captures state inside a forward only on its own tile
  grid (`track_chunk`); an unsupported grid needs a matching kernel, not rounded depths.
- **Arithmetic state selection corrupts sources**: `(working - snapshot) * flag + snapshot` returned 0 for a selected 1
  when the unused value was 1e20 and propagated unused NaNs; slot selection must be bit-exact (masking or indexing),
  which is the contract `copy_slots` / `load_rows` state.
- **Byte-exact restore is not exact continuation**: a hybrid image restored byte for byte on all ranks can still give
  a continuation whose hidden state differs from the uninterrupted run. A repacked continuation moves tokens to other
  sequence-parallel ranks and changes the reduce-scatter accumulation order. A fixed-order FP32 reduction makes the
  continuation byte-exact, but it also changes the uninterrupted baseline. Hence the numerics profile in the checkpoint
  identity, and qualification by continuation golden rather than restore bytes.
- **Compiled state-owner capacity**: another model's profiled state capacity is not this model's request; the pool takes
  its size from SGLang's request table for its own owner.
- **Host-side slot ids** (Qwen3.5): a free list and a request mapping on host tensors avoid variable-shape lazy graphs
  at admission. Clears and copies run as one fixed-shape compiled op per state tensor, padded to buckets and warmed at
  startup. The pool keeps ids on the host and leaves bucketing to the kernel ops.
- **Native COW over in-kernel copy-on-match**: a donor-row copy fused into the decode prologue is bit-identical, but
  valid only when the donor's row holds the branch position. The framework copies from tree-owned checkpoints through
  upstream's COW ([Copy-on-write on a radix branch](#copy-on-write-on-a-radix-branch)).
- **GDN state over PD** (Qwen3.5): the kernel's rec layout `[K, vh, V]` differs from upstream's `[vh, V, K]`; sending it
  raw gives 128 runs per head, so a staging rewrite into a head-major host copy made each decode row one 64 KiB run per
  layer; descriptor lengths are checked against row bytes; a request's attention group must nest in its GDN group. Hence
  the kernel-layout marker, the cross-TP refusal and the equal-TP `slot_runs`.
- **Refuse at registration, not per request** (`sglang_patches/nixl_state_components`, `pd_decode_kv_quarantine`): a
  pairing that cannot work fails once when the roles meet, naming both sides; a failed transfer's destination is held
  until no WRITE can land.
- **Global layer identity under PP** (`sglang_patches/pd_kv_layer_ids`): PD pairs state entries by global layer id;
  every state key here is global.
