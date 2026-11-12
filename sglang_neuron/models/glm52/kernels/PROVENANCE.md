# GLM-5.2 NKI kernels: provenance and edits

| path | what | run by |
| --- | --- | --- |
| `decode/` | the decode giga kernel `glm52_decode_giga_kernel` (`giga_kernel.py`; 78 layers: MLA attention with the DSA indexer, the MXFP4 all-expert MoE, the LM-head tail and sampler) and the MTP draft chain (`mtp/`) | every decode point (`models/glm52/points.py`), through `models/glm52/ops/gk_decode.py` and `ops/mtp_draft.py` |
| `prefill/` | the prefill giga kernel `glm52_prefill_giga_kernel` (`giga_kernel.py`): a launch group of a stage's layers in one kernel, context parallel over the stage | the prefill stages (`models/glm52/prefill/stage.py`), through `ops/prefill_kernel.py` |
| `topk_select.py` | the NKI top-k select with a carried payload, and the wide rounds of a chunked key shard (recorded in `SHARED_MANIFEST.json`, and part of both kernels' digests) | both kernels' indexer top-k merges |
| `__init__.py`, `compile_knobs.py` | the kernel digest that keys the NKI compile cache, and the registry of the knobs the kernels read at import or trace time | |

Both kernels run at one rank per physical NeuronCore (LNC1).

## Where the kernels come from

Both were forked on 2026-11-07 from GLM-5.2 kernels written in the Neuron kernel library's layout (an `nkilib` tree of
`core/`, `experimental/` and `models/` packages). Here they take GLM-5.2's own layout and names: `decode/` and
`prefill/` with their sub-packages by function (`attention/`, `moe/`, `mlp/`, `router/`, `utils/`, ...), the library's
folder names and model-family names dropped. The edits below name each file by its path in the fork, as it was when the
edit was made; one commit moved the files to these paths (`git log --follow` gives a file's history across it). They are
maintained here, not refreshed from elsewhere: each kernel's `MANIFEST.json` records every file's sha256,
`tests/unit/test_kernel_manifests.py` holds the kernel to it, and every change since the fork is listed below. After an
edit, list it here and rewrite the manifest (`python tools/glm52/kernel_manifests.py --write
<decode|prefill>`).

The kernels run on the pinned Neuron SDK 2.33.0 (`sglang_neuron/release/neuron_sdk.toml`, read by
`models/glm52/toolchains.RELEASED_TOOLCHAINS`) with the stock compiler's own options. Neither kernel ships a prebuilt
device library or depends on a private one: `tools/neuron_release.py guard` (run by `tests/unit/release/test_guard.py`)
checks the package and the served NEFFs.

## Edits to the decode kernel (fork paths)

Under the fork's GLM model package (in the library's `models/` folder):

- `core/attention/indexer_tkg/indexer_topk_local.py`: the packed local stage takes a dealt key shard only, and writes
  each winner's payload code (its shard-local key) into the packed tile's padding words, which the NKI merge selects
  from.
- `core/attention/indexer_tkg/indexer_topk_merge_carry_index.py`: the packed CP merge selects in NKI
  (`topk_select.py`: two `nisa.topk` calls, the second carrying each winner's code),
  loading its score and code snakes straight from the gathered blocks, and refuses the wide merge: on a chunked
  shard a CP rank's key shard is the keys it holds, so it takes its own share of the global top-k as its selection.
- `core/attention/indexer_tkg/indexer_topk_local_carry_index.py`: a chunked shard's phase 2 runs `topk_select`'s
  wide rounds over the key group; a shard one select ranks, and the split hand-off, are refused.
- `core/attention/mla_tkg/prior_index.py`: `build_resident_index` writes each token's keep row through an access
  pattern whose partition step is the tile's free size (its reshape gave the size-1 partition dimension a step of 0,
  which neuronx-cc's BIR verifier refuses). The same bytes in the same order; the NEFF's instruction bytes are
  unchanged.
- `experimental/glm52_decode_giga_kernel.py`: a fourth latent stack `kv_cache_hi3`, so each stack of a 32 x 1M latent
  cache stays below the device's 2^32-byte allocation.
- `experimental/dsa_attention_decode_layer.py`: the indexer merge's NKI select and the wide rounds (as above), a CP
  rank's own top-k share as its selection, and `GLM52K_ATTN_COMBINE_RECV_QUEUE` (off by default), the CP combine's
  receive on a queue of its own.
- `experimental/moe_decode_layer_dp_tp_ep.py`: `GLM52K_MOE_SHARED_ROWS_FROM_EP` (off by default), the shared expert
  reading its group's rows out of the EP row gather.
- `core/moe/moe_tkg/`, `core/router/rmsnorm_mx_quantize.py`, `experimental/moe_decode_layer_dp_tp_ep.py` and the giga
  kernel's MoE call: the MoE keeps only the schedule the served points run. The compacted-token routed path
  (`compact_routed_tokens`, the "DLoC" MoE, slower than the all-expert MoE on the released toolchain) is removed with
  the giga kernel's zero-maintained output ring, as are the shared expert's SBUF-to-SBUF gather (an SBUF collective
  cannot be pinned at long context, and that gather measured stale bytes) and the measurement-only forced routings.
  The knobs every served point set the same way (the tile stores, the weight-DMA order, the next expert's gate/up
  prefetch, the bank-major row unpack, the one-transpose row pack on Sync, the fused swizzle, the shared expert's unit
  affinities) are the kernel's code, and the row pack and the swizzle refuse a token count above one tile. The EP row
  gather always lands in the layer's own collective destination: the caller-owned one (`rows_ep_dst`, with the giga
  kernel's `NKILIB_MOE_ROWS_EP_DST` and `NKILIB_MOE_ROWS_EP_RING`) is removed, as is the shared ReduceScatter hook
  (`set_shared_rs_hook`) that only the removed token-sharded merge set.
- `mtp/core/attention/indexer_tkg/indexer_topk_local.py`,
  `mtp/core/attention/indexer_tkg/indexer_topk_merge_carry_index.py`,
  `mtp/experimental/dsa_attention_decode_layer.py`: the MTP draft layer's indexer takes the same NKI select and wide
  rounds as the target's; it merges the shards' wide shares (`wide=True`). Its split carry-index local refuses.
- `mtp/core/utils/cc_pin.py`: the draft chain pins each collective operand once, to the kernel end; the giga
  kernel's attention window does not apply to it.
- The attention core reads no schedule knob any more (`core/attention/mla_tkg/` and its `mtp/` copy, with their callers
  in `experimental/dsa_attention_decode_layer.py`, `experimental/glm52_decode_giga_kernel.py`,
  `mtp/experimental/dsa_attention_decode_layer.py`). The schedule the served points ran is
  now the copy's own code: the cache write after the gathers, the eviction-folded softmax, the pipelined pair loop,
  six-tile key evictions, the fenced resident fallback with its SBUF hand-over, rank-major queries, direct
  down-projection stores, the spread hidden and q-norm K-tiles, the early kv path, the banded `wq_b` on ACT, bank
  evictions on Sync and the trimmed snake table. The paths no served point traced are gone: the DMA-transpose key path,
  the bias-row share, the resident layer filters and HBM hand-over, the row-slot and query-copy variants, the KV
  prefetch, the decode core's KVShare form (`attend_new_keys=False`) and DMA-transpose arguments with the two
  unimported draft modules that used them (`experimental/glm52_mtp_draft_chain.py`,
  `experimental/glm52_mtp_decode_layer.py`), the decode core's token-sharded entry (`prior_rows_sb`,
  `query_sib_index`, `prior_rows_ready`, the context-only exit and `NKILIB_TS_CTX_SWDGE`, `NKILIB_TS_BIAS_SHARE`,
  `NKILIB_TS_ROWS_WAIT_ALL`), the o-projection's ACT bank evictions (`NKILIB_OPROJ_EVICT_ACT` in the MTP layer, an
  argument no caller passed in the decode copy), in the MTP core the picked-query path, the `kv_free=1` form and the
  unfused exit, and the DMA-transpose arguments the MTP draft chain and layer still threaded to a core that refused
  them. The served decode programs trace to the same live MLIR as before: what changes is only the never-used
  `[128, 1]` fp32 tile the unfused softmax allocated. The MTP draft programs add six memsets a chain, one a layer
  call: the softmax weights tile is zeroed where an odd token count leaves its second half unwritten.
- `mtp/`: the draft chain keeps only the program the served draft runs. It always takes the FP8 experts with one
  factor per weight, the greedy head, IndexShare, the column-tiled MLPs, the draft rows and the rows layout over a
  column-sharded `eh_proj`, the step-0 indexer's live-chunk loop with draft-group scoring, hardware-DGE key reads
  and its query stores ordered after the candidate gather, and the schedule its pins served (the late step-0 helper
  tables, the exit table, the front pick of the gathered rows, the KV-free draft attention with early keys, the
  attention ring, the early `hnorm` and
  embedding rows, the quiet head, the blocked residual adds, the shared weight ring, the fused shared gate/up, the
  banded argmax, the tight lm_head tiles and the HBM collectives) as code. Gone with the other options: the top-k head
  (`mtp/experimental/glm52_lm_head_tail.py`, `mtp/core/lm_head_tail/`), the all-slot MoE layer and its block-scale and
  bf16 experts, the replicated prologue, the all-step indexer, and the step-0 draft-row core, exit, tail and carry
  variants no point set; with them the chain's shared-expert affinity, shared rank and TP group inputs, the expert
  MLP's (`mtp/core/mlp_bf16/`) block scales, unscaled weights, untiled matmuls, own weight ring and HBM output, and the
  attention layer's query-row variants (`query_rows`, `query_exit_rows`) and caller-owned output
  (`mtp/experimental/dsa_attention_decode_layer.py`). The rows MoE's affinity slicing is
  `mtp/core/router/affinity_masking.py`. The served draft programs trace to the same MLIR.
- `core/attention/mla_tkg/mla_decode_attention.py` and its `mtp/` copy: the softmax weights tile is zeroed whenever
  the token count is odd, not only below 64 heads. A pair's absent second token leaves partitions 64-127 unwritten,
  and the weight transposes read all 128 rows; at 64 heads a one-token draft step moved uninitialized bytes into
  columns nothing reads, which the BIR simulator refuses.

Removed from the copy: a prebuilt top-k device library with its loader, a toolchain probe, and the fork's refresh tool
with its upstream manifest. Code no served point reaches: the first MTP draft chain and the modules only it used
(`experimental/glm52_mtp_decode_layer.py`, `experimental/glm52_mtp_draft_chain.py`,
`experimental/moe_decode_layer_dp_tp_ep_bf16.py`, `core/mtp_prologue/`, `core/mtp_tail/`, `core/moe_bf16/`),
`core/utils/common_types.py`, the library's top-k kernel body (its configuration and the helpers `chunked_topk.py` calls
stay, renamed `decode/lm_head/topk_layout.py` with `TopkConfig`, `create_topk_config` and `_validate_topk`), uncalled
helpers of the reachable modules, the indexer top-k's device-library selects with their staging knobs
(`NKILIB_IDX_MERGE_2Q`, `NKILIB_IDX_MERGE_NOSYNC`) and its contiguous-shard arms (`NKILIB_IDX_PACK_VECTOR`), and the
speculative-sampling stages: the LM-head tail's speculative verify and the draft's sampled head
(`mtp/core/mtp_tail/mtp_sampled_tail.py`); the block-cyclic indexer pool, which the chunk-dealt pool replaced; the
token-sharded attention with its direct-RDMA KV merge (`NKILIB_MLA_TOKEN_SHARD_ATTN`, `NKILIB_TS_*`,
`NKILIB_MLA_KVMERGE_*`); and the debug-only outputs (the layer taps, the first-layer selection, the per-layer trace).

## Edits to the prefill kernel (fork paths)

Under `nkilib/experimental/mla/glm/`:

- `glm_topk_merge.py` and `glm_indexer_topk.py`: the chunk merge selects in NKI
  (`topk_select.emit_select_nki`, two `nisa.topk` calls carrying each candidate's chunk-local position).
- `glm_prefill_gk.py`: a full layer's picks on the halved indexer path are attended without a live gather, so the
  stage runs without the runtime's experimental per-block collectives.
- `SkipMode` moves into `nkilib/core`, so the core modules import nothing from `experimental/`.
- The sub-kernels keep only the arguments the prefill giga kernel passes: the V-up kernel's batch-major, wide and
  head-subset outputs, the o_proj partial's column groups, the QKV kernel's resident weights and stage-1 preload
  argument, the sparse attention's group-major output, the blockwise MoE's zero-init fence and early dynamic setup,
  and the indexer config's `pack_rows` are gone, with `subkernels/indexed_flatten.py`, which nothing calls.

Removed from the copy: the prebuilt top-k device library and a toolchain probe. The live-plan KV gather
and prefetch stay, though the release turns them off at trace time (they need the runtime's experimental per-block
collectives): with the recipe's `GLM_GK_KV_PLANS`, the stage graph's KV allocation keeps one spare row for them, so
removing them changes every served prefill program the hardware gates ran.

Cut to the served prefill points (PP4 x 32 and PP16 x 8 ranks), removed as paths no launch of either point reaches: the
kernel-entry trace (`core/utils/entry_trace.py`, `NKILIB_TRACE_KERNEL_ENTRY`); every trace-time status print; the old
merge's body, its candidate cap and the hoisted, copy and wide variants of its two-level select
(`NKILIB_GK_IDX_MERGE_HOIST`, `GLM_GK_IDX_MERGE_COPY`, `GLM_GK_IDX_MERGE_WIDE`) and the streaming top-k
(`stream_topk`), whose select no trace could emit; the indexer key split
(`glm_indexer_ksplit.py`, `q_pos_g`); the MoE expert replicas (`moe_slot_tab`, `GLM_GK_MOE_REPLICA`); the expert-count
dump (`emit_moe_counts`); the cross-launch weight and KV hand-overs (`next_*` / `pre_*`, `xkv_next_cache` / `xkv_pre`,
`GLM_GK_X_LAUNCH`, `GLM_GK_KV_PF_XL`) with the full-layer KV prefetch (`GLM_GK_KV_PF_FULL`); the packed prefill (`q_lo`,
`q_row_offset`); and the no-mask attention graph (`attn_nomask`, `GLM_GK_ATTN_NOMASK`) with its k_pe tile path
(`GLM_GK_ATTN_KPETILE`). The recipe drops the two knobs the cut left unread. The comments and docstrings lose the fork's
local-patch tags and internal commit ids.

Knobs: each knob both points set to one value is code at that value (`GLM_GK_CC_SYNC=oproj,moe`, `GLM_GK_KV_FLAT_BASE`,
`GLM_GK_IDX_HALVES`, `GLM_GK_MOE_ROWS`, `GLM_GK_MOE_ROUTING_PLAIN`, `GLM_GK_ATTN_MASKROW`, `NKILIB_GK_IDX_RELU_PAIR=2`,
the head-TP attention), and the branches of knobs no point sets, or sets to a value that does nothing there, are gone
with their readers (`up_scope.py` among them). Only the knobs whose value differs between PP4 and PP16 stay. Every
served prefill program traces to the same MLIR as before.

Names: `sbm_name_scope` no longer patches the NKI tracer's private location builder (`nki._backends`). Each call site
passes its name through `scoped_name`, which adds the same scope prefix, so every name a call site gives traces as
before. The two names NKI makes itself, the shared identity matrix (`_nki_identity_<dtype>`) and a shared constant's
location (`shared_const_<n>`), now carry no scope prefix; no instruction or address changes.

The copy's own knob is `GLM52K_GK_ATTN_UNROLL_MAX_S` (it was `NKILIB_GK_ATTN_UNROLL_MAX_S`): the copies' knobs take a
`GLM52K_` prefix of the same length, and `NKILIB_` stays nkilib's (`NKILIB_LOG_LEVEL`).

The copy's SBUF allocator module is `utils/sbuf_scoped_alloc.py` (it was `utils/allocator.py`), a name that says what
it is and no longer clashes with SGLang's page allocator, `mem_cache/allocator.py`.

Comments no longer point at sources outside this repository (another model's loader, a design note) or at a runtime
feature by its internal name.

## Import paths after the plugin's layout split

The plugin's layout package split by role: the decode modules the kernels import (`gk_geometry`, `idx_pool`,
`weight_loaders_mxfp4`) are now under `models/glm52/layout/decode/`. Both kernels' imports and docstrings name the new
paths; no other line changed.
The per-token KV row (one dequantization factor per token, `set_kv_row(0)`) is gone with its quantizer and every per-key
factor step of the attention: nothing selected it, so every served program writes and reads the decode kernel's static
row. Two buffers that only it read stay allocated, so the served programs keep their SBUF layout. The input checks and
the comments no longer describe it.
The plugin's GLM-5.2 unit tests moved into folders that mirror the package: `prefill/attention/common.py` names the
static KV scale's test at its new path, `tests/unit/models/glm52/layout/test_static_kv_scale.py`; no other line moved.
The comments and docstrings of both kernels (`decode/`, its `mtp/` copy, and `prefill/`) no longer state measured
performance numbers (times, bandwidths, utilization, speedups and run results); each keeps its reason without the
number.
The comments and docstrings of both kernels also no longer state internal references (internal library paths and
model names, run and test names, test tools, design notes and test files that are not in this repository) or the
history of experiments, fixes and design changes; they state the current behavior and its reason in the present
tense, name the current files, functions and knobs, or describe them in generic terms, and some typos are fixed.
`prefill/utils/kernel_helpers.py` loses two commented-out functions. No code changed: only comment and docstring
lines changed, and some of them moved.
