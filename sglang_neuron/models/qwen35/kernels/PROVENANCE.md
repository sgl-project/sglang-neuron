# Qwen3.5 NKI kernels: provenance and edits

| path | what | run by |
| --- | --- | --- |
| `giga/` | the Qwen3.5-397B-A17B giga kernels: the whole-model decode `qwen35_hybrid_multistep` (`multistep.py`; 45 GDN + 15 gated attention layers, MoE in every layer, the LM head and greedy argmax, the kv_cp decode), the chunked prefill stack (`prefill_stack.py`), the MTP verify and low-concurrency stacks, their layer bodies, the torch references the kernels and tests import and the library modules they import (`core/`) | the decode graph (`models/qwen35/graph.py`), through `ops/qwen35_giga.py` (`KERNEL_PACKAGE`) |
| `serve/` | `qwen35_serve_prefill` (`serve_prefill.py`): one segment of prefill layers with per-layer weights over the prefill stack's layer body, writing the attention KV into the decode's pool, and its paged chunk mode | the prefill graph, through `ops/qwen35_giga.py` (`PREFILL_MODULE`) |
| `weights/` | the kernel packer `pack_model.py` (the per-rank stacked operands of `qwen35_hybrid_multistep(per_layer_weights=True)`) | `tests/unit/models/qwen35/test_weights.py` reads its `Layout` for the kernel operand shapes. The server does not read its output: `models/qwen35/weights.py` reads the serving pack of `tools/qwen35/build_pack` |
| `__init__.py`, `manifests.py` | this package, and the copies' file manifests | |

The kernels run at one rank per physical NeuronCore (LNC1).

## Where the kernels come from

The giga kernels were forked from Qwen3.5 kernels written in the Neuron kernel library's layout: a model package
(`nkilib/models/qwen35`) that reaches the library's core by relative imports. The fork includes the multistep's
`QWEN35_MS_IDX_ONCE` gate and `row_valid` (padding rows are routed to no expert). The library modules whose bytes
differ from the serving image's nkilib were forked with them (`core/attention/attention_cte.py`, `core/utils/{allocator,
kernel_helpers,modular_allocator}.py` and their package files). The other library modules they import
(`core/utils/{kernel_assert,logging,tree_logger,common_types,interleave_copy,stream_shuffle_broadcast}`) are the
serving image's `nkilib`, imported by absolute name; the compile cache's digest records the nkilib version
(`backend/compile_cache.kernel_library_versions`). `weights/pack_model.py` comes from the fork's `tools/`, and
`serve/serve_prefill.py` from the fork's prefill serving module (its module docstring lists the adaptation).

They are maintained here, not refreshed from elsewhere: each copy's `MANIFEST.json` records every file's sha256,
`tests/unit/models/qwen35/test_kernel_manifests.py` holds the copies to it (and checks that the compile cache's kernel
sources cover them), and every change since the fork is listed below. After an edit, list it here and rewrite the
manifest (`python -m sglang_neuron.models.qwen35.kernels.manifests --write <giga|serve|weights>`).

Until this package, the kernels lived outside the plugin package in a top-level `kernels/qwen35/` tree, vendored by a
script from a kernel library checkout plus the patch files below; the tree also held the library's harness tests
(`tests/`), the persistence probes (`probes/`) and the per-layer GDN kernels of the dense host (`gdn/`). Nothing in the
plugin runs those, so they are not carried here: they stay in this repository's history (the commit that moved the
kernels here deletes them).

## Layout

| Here | Fork path |
| --- | --- |
| `giga/*.py` | `models/qwen35/` (the overlay `kernels/qwen35/src/qwen35/`) |
| `giga/core/attention/`, `giga/core/utils/` | the library's `core/attention/`, `core/utils/` |
| `serve/serve_prefill.py` | `giga/qwen35/kernels/qwen35_serve/serve_prefill.py` |
| `weights/` | `tools/` |

`giga/` drops the library's `models/qwen35/` package level: the model modules sit at its root next to `core/`.

## Edits to the fork

The rows 0001 to 0018 were patch files applied to the fork at vendoring (`kernels/qwen35/patches/`); the files here
carry them. Each adds an argument or a gate whose default traces the kernel as before, unless the row says otherwise.
Files are named by their path here. They serve the radix prefix cache and chunked prefill on a paged KV pool
(docs/design/recurrent-state.md), the long-context kv_cp decode and the PD transfer (docs/design/context-parallelism.md).

| Edit | Files | What it changes | Why |
| --- | --- | --- | --- |
| 0001 | `giga/gdn_prefill.py`, `giga/gdn_prefill_torch.py` | `valid_len` ([1, 1] int32, one sequence): pad rows get g = 0, beta = 0, k = 0, so the state passes them unchanged, and the conv window ends at the last valid row (one-hot matmuls over the token-major projections, exact); `state_row`: the initial-state row to read (no HBM slice). The reference skips pad rows. | a prompt's last chunk is padded to its bucket; the GDN state handed on must be the state after the last real token |
| 0002 | `giga/attn_prefill.py`, `giga/attn_prefill_torch.py` | paged mode (`write_pages`, `prior_pages`, `prior_len`): the caches are page pools in `attn_decode_tkg`'s "dp" layout, the chunk's K / V go to its pages (page id as each DMA's scalar offset), the prior is gathered from the request's pages into a static-capacity scratch with a run-time `prior_used_len`. `attn_prefill_paged_torch` is its reference. | the radix cache shares pages between requests, and a chunk attends to the prefix already in the pool |
| 0003 | `giga/prefill_stack.py` | `prefill_layer_body`: with `st["from_state"]` the first chunk starts from the hand-off rows; `st["paged"]` runs the paged attention; `st["valid"]` passes `valid_len` to the GDN. | chunked prefill from a prefix: a chunk continues the request's state rather than zeros |
| 0004 | `giga/multistep.py`, `giga/layer_decode.py` | `block_table` ([B_attn, MP] int32, state modes 2 / 3) in place of `kv_slot`: each attention layer's table offset by its pool rows, passed to `attn_decode_tkg(block_table=...)` (its paged prototype, unchanged). | the decode reads the same page pool the prefill writes |
| 0005 | `giga/attn_decode.py` | split-K for spans past `QWEN35_ATTN_SPLIT_MAX` (16384): two passes over blocks of `QWEN35_ATTN_SPLIT_BLOCK` (4096) positions (running max / sum, then the normalized probabilities and P.V accumulated per row in fp32). Shorter spans keep the single pass. | SBUF holds one block's scores instead of a row's whole `[B * nq, S]` fp32 span, which fails allocation past about 16-24K |
| 0006 | `giga/attn_decode.py` | split-K in one pass over the blocks (online softmax, `QWEN35_ATTN_SPLIT_PASSES=1`, the default): each block folds into the row's running max / sum and its unnormalized bf16 `exp(s - M)` V into an fp32 accumulator rescaled by `exp(M_old - M_new)`, divided by the sum at the end; the rescale moves to the accumulator's head partitions through one exact 0/1-selector matmul. `QWEN35_ATTN_SPLIT_PASSES=2` keeps 0005. | each block's K and V are read once instead of twice |
| 0007 | `giga/attn_decode.py` | the paged one-pass split-K iterates its blocks in a run-time loop (`nl.while_loop`, `QWEN35_ATTN_BLOCK_LOOP=1`, the default) instead of one unrolled copy per block: the pass's block-table columns staged by one dynamic-offset DMA, the block base and pass counter loop-carried; `pool_off` (0008) added in the loop. Gates on the "dp" layout and whole-block spans (`block_loop_engages`); bit-identical to the unrolled form in simulation. Needs `enable-linear-scan-allocation=false` (the decode's `NKI_COMPILE_MODE=enable-alloc=legacy`; NCC_IBIR853). | program size and compile time independent of the context length |
| 0008 | `giga/multistep.py`, `giga/layer_decode.py` | when the layer takes 0007's loop, the multistep hands it the raw shared block table plus the layer's pool-row offset (`pool_off = a * Spa`, applied per pass) instead of a staged per-layer copy; the unrolled path keeps the copy. | the loop's dynamic-offset DMA must read a kernel input, not an HBM intermediate |
| 0009 | `giga/attn_decode.py`, `giga/layer_decode.py`, `giga/multistep.py` | decode context parallelism (kv_cp) for the paged decode: `attn_cp_front` (per-rank q / k / v), `attn_cp_core` (the online split-K over the rank's CPB-token block of every page, emitting unnormalized `(M, L, acc)` and the owner-only new-token write, non-owners into a trailing sink row), `attn_cp_combine` (the cross-rank merge after the `(M, L)` all-gather), `attn_cp_tail` (o_proj partial for the reduce-scatter); `_attn_mixer_cp` wires the rank-as-data collectives so one NEFF serves every rank. The core's span runs in a run-time block loop past one block (`cp_block_loop_engages`), with an unrolled tail. The KV pool assert allows the sink row. | 256K contexts: one rank's KV and attention work shrink by the kv_cp degree |
| 0010 | `giga/attn_decode.py` | `attn_cp_core` takes an inert `probe=None` (serving never passes it; the default traces byte-identical): a probe harness passes a [1, 4] fp32 tensor that records the blocks run and the gather DMAs. | per-block TPOT attribution on silicon |
| 0011 | `giga/attn_decode.py` | a run-time trip bound for `attn_cp_core`'s block loop derived from `pos`: ceil(max_b u_b / (SCH * G)) passes (an all-masked pass is an exact no-op), the partial tail only when the bound reaches the last whole block; bit-identical in the cp16 sims. Padding rows (pos 0) never extend it. | the loop runs to the cohort's longest live row, not the block-table width |
| 0012 | `giga/attn_prefill.py`, `giga/prefill_stack.py` | token-major K pages (`attn_prefill_tkg(k_tm=)`, the kv_cp=1 PD prefill's pool layout under `SGLANG_NEURON_QWEN35_KV_TM`): K is [.., S, d] like V; every K DMA moves to a strided pattern landing the same values in the same SBUF tiles, bit-identical in simulation; `prefill_stack.py` threads `st["k_tm"]`, `serve/serve_prefill.py` takes `k_tm`. | the PD transfer's K block of a page is one contiguous run per layer |
| 0013 | `giga/attn_decode.py` | token-major K pages on the kv_cp decode (`attn_cp_core`'s `k_cache_t` is `[NP, nkv, CPB, d]` like `v_cache`): the K gather reads it with a strided DMA of the same count and tile shapes; the owner-only K write takes V's form. `attn_decode_tkg` (kv_cp=1) keeps the dim-major layout. Bit-identical cp16 sims. | 3,855 -> 30 NIXL descriptors per page in the PD transfer |
| 0014 | `giga/attn_decode.py` | the token-major K gather reads V's contiguous burst form into a staging tile and transposes each 128-token chunk on-chip (`nisa.nc_transpose`) into the same tiles; same DMA count, bit-identical. | 0013's 2-byte-strided read could not burst (much slower on silicon at b16 / 220K) |
| 0015 | `giga/attn_decode.py` | pipelines the token-major K gather: the span's whole run is staged up front per (row, head) and the PSUM transpose tiles alternate by chunk parity; same DMA count and tiles, bit-identical. | the DMA queue overlaps the transposes instead of serializing (0014 measured slower than the dim-major gather) |
| 0016 | `giga/attn_prefill.py` | the token-major K prefill drops every 2-byte-strided DMA: pool K rows written from the post-norm / RoPE tile in V's burst form, both prior-K reads take V's contiguous form and transpose on-chip into the same tiles; `k_tm=False` paths byte-identical. | the 0014 hazard in the prefill |
| 0017 | `giga/attn_decode.py` | head-combined K / V gathers: a page's two KV heads are one contiguous 64KB run, so one per-page DMA stages both heads (1024 -> 512 seg DMAs per trip at B16); same tiles downstream, bit-identical. The CP probe's seg-DMA expectation follows. | halves the gather's per-DMA fixed cost |
| 0018 | `giga/attn_decode.py` | the head-combined gather DMAs run on the hardware descriptor engine (`dge_mode=hwdge`); timing only, bit-identical. | the 64KB DMA measured a higher fixed cost than a 32KB one on the software DGE path |
| prune | `giga/`, `weights/` | removed the modules nothing imports (import closure of `ops/qwen35_giga.py`'s two entries, `tools/kernels/compile_smoke.py` and the unit tests): the collective and DMA benchmarks `coll_bench`, `dma_bench`, `lowc_coll_bench`; the torch references `lowc_layer_torch`, `moe_front_torch`, `moe_lowc_torch`, `moe_prefill_torch`, `mtp_torch`; `mx_dense`, `serve_layout` and `weights/accuracy_check`. They stay in this repository's history (the commit that prunes them deletes them). | review: keep only what serving and the tests reach |
| move | every file | the files moved from `kernels/qwen35/` into this package (layout above): `giga/attn_prefill.py` imports `.core.attention.attention_cte` (was `...core`), `serve/serve_prefill.py` imports `..giga.prefill_stack` (was `sglang_neuron.kernels.qwen35.giga.models.qwen35.prefill_stack`); `giga/__init__.py`'s docstring. No kernel arithmetic, kernel name or environment variable name changed. | inside the plugin package the compile cache's kernel digest covers the kernels and the wheel ships them |
| perf text | `giga/attn_decode.py`, `giga/attn_prefill.py`, `giga/core/attention/attention_cte.py`, `giga/gdn_decode.py`, `giga/gdn_prefill.py`, `giga/layer_decode.py`, `giga/lowc_layer.py`, `giga/moe_compact.py`, `giga/moe_experts_mx.py`, `giga/moe_prefill.py`, `giga/multistep.py`, `giga/verify_moe.py`, this file | the comments and docstrings no longer state measured performance numbers (times, rates, speedups, measured run results); each keeps its reason without the number. No code changed, and every line stays where it was. | review: no performance numbers in the repository text |
| comments | `giga/attn_decode.py`, `giga/core/attention/attention_cte.py`, `giga/core/utils/allocator.py`, `giga/core/utils/kernel_helpers.py`, `giga/gdn_decode.py`, `giga/gdn_decode_torch.py`, `giga/gdn_prefill.py`, `giga/gdn_prefill_torch.py`, `giga/gdn_verify.py`, `giga/layer_decode.py`, `giga/lowc_layer.py`, `giga/moe_lowc.py`, `giga/moe_prefill.py`, `giga/multistep.py`, `giga/prefill_stack.py`, `serve/serve_prefill.py`, `weights/pack_model.py`, this file | the comments and docstrings no longer state internal references (development branches, PR numbers, run and test names, test tools, design notes and test files that are not in this repository) or the history of experiments; they name the current files, functions and commands, or describe them in generic terms, and some typos are fixed. The `QWEN35_LOWC_DGATHER` comment in `giga/moe_lowc.py` states what the path supports, and `giga/core/utils/kernel_helpers.py` loses two commented-out functions. No code changed: only comment and docstring lines changed, and some of them moved. | review: no internal references or experiment history in the repository text |

Earlier edits, at vendoring: the imports of library modules that the image's nkilib provides with the same bytes name
`nkilib.` instead of the relative path. The comments in `weights/pack_model.py` refer to the fork's accuracy
reference, dense MX packer and low-concurrency torch reference: `weights/accuracy_check`, `mx_dense` and
`lowc_layer_torch` (see the prune row).

## Hosting

`giga/multistep.py` `qwen35_hybrid_multistep` (58 operands) is the whole-model decode for `num_steps` tokens: embedding
table and token shard in, stacked GDN operands (`w_proj_p`, `conv_w`, `a_log`, `dt_bias`, `norm_w`, `w_out`),
attention operands (`w_qkv`, `q_norm_w`, `k_norm_w`, `w_gate_p`, `w_o`), MoE operands with this rank's MX experts,
`final_ln_w`, `w_lm_p`, `vbase`; the state it updates: `conv_state`, `rec_state` (GDN, slot-indexed by `gdn_slot` under
`state_mode`), `k_cache_t`, `v_cache` (`kv_slot` or `block_table`, `kv_write`, `pos`); `tp_group`, `atp_group`,
`ep_group` as `nki.collectives.ReplicaGroup` with their sizes and `tp_rank`, `atp_rank`, `ep_rank`; `row_valid` masks
padding rows; `per_layer_weights=True` reads stacked per-layer weights. The persistent mode (`persistent`, `ctl`,
`max_steps`) is blocked on hardware (collectives in device loops). `giga/prefill_stack.py` `qwen35_prefill_stack` is
the chunked prefill, and `serve/serve_prefill.py` the segment the plugin serves. Weight layout: `weights/pack_model.py`
writes the per-rank stacked operands of `qwen35_hybrid_multistep(per_layer_weights=True)`.

## Tests

`tests/unit/models/qwen35/` runs on CPU: the manifests and compile-cache coverage (`test_kernel_manifests.py`), the
packer's layout (`test_weights.py`) and the prefill torch references (`test_prefix.py`). The kernels' device gates are
the served pair's e2e test (`tests/e2e/models/qwen35/disaggregated/test_accuracy.py`) and the `tools/kernels/` compile
probes.
