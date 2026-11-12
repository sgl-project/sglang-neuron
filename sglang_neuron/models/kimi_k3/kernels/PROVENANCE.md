# Kimi-K3 NKI kernels: provenance and edits

| path | what | run by |
| --- | --- | --- |
| `prefill/` | the prefill kernels of one pipeline stage rank (sequence-parallel form, LNC1): the KDA block, the MLA flash and its cross-rank merge, AttnRes, the MXFP8 linears, the blockwise MXFP4 MoE, the MoE token map and the MLA page I/O | the prefill stages (`models/kimi_k3/prefill/`), through `prefill/kernel_imports.py` |
| `decode/` | the two-node decode programs: the verify (93 layers, the tail and the drafter's aux states in one call), the MLA drafter step, the full-vocab sampler and the KDA init kernel, with the host layout builders of their weights | the decode (`models/kimi_k3/decode/`), through `decode/kernel_imports.py` and `decode/kernel_calls.py` |
| `__init__.py` | this package | |

## Where the kernels come from

Both parts were forked from Kimi-K3 kernels written in the Neuron kernel library's layout (an `nkilib` tree of `core/`,
`experimental/` and `models/` packages). Here they take this package's layout: each part's sub-packages by function
(below), the library's folder names and model-family names dropped. They are maintained here, not refreshed from
elsewhere: each part's `MANIFEST.json` records every file's sha256, `tests/unit/models/kimi_k3/test_kernel_manifests.py`
holds the kernels to it, and every change since the fork is listed below. After an edit, list it here and rewrite the
manifest (`python tools/kimi_k3/kernel_manifests.py --write <decode|prefill>`).

The kernels run on the pinned Neuron SDK 2.33.0 (`sglang_neuron/release/neuron_sdk.toml`) with the compiler options of
the served programs. Neither part ships a prebuilt device library or depends on a private one: `tools/neuron_release.py
guard` (run by `tests/unit/release/test_guard.py`) checks the package and the served NEFFs. Nothing here imports the
serving image's `nkilib`: the library modules the kernels need are vendored with them, and the modules import each
other by relative imports.

## Layout

| Part | Here | Fork path |
| --- | --- | --- |
| prefill | `attention/` (`mla_flash_latent`, `mla_dcp_merge`) | the part's root (`mla_flash_latent_kernel`, `mla_dcp_merge`) |
| prefill | `attn_res/` | `experimental/kimi_k3/attn_res/` |
| prefill | `kda/` | `experimental/kimi_k3/kda_block_prefill/` and `kda_prefill_v2/` |
| prefill | `linear/` (`mx_linear`, `mx_head_linear`) | `experimental/kimi_k3/prefill_mx_linear/` |
| prefill | `moe/` (the blockwise MoE, `token_map`) and its library copy `moe/mlp/`, `moe/quantization/`, `moe/utils/` | the separate library copy the blockwise MoE was built against, `.../moe_cte/` and its `core/` packages; `moe_token_map` |
| prefill | `utils/` (with `trn_target`, `valid_row_window`) | `core/utils/`, `experimental/kimi_k3/` |
| prefill | `page_io` | the part's root |
| decode | the root: the model's decode kernels and host builders | `models/kimi_k3/` |
| decode | `multinode/` | `models/kimi_k3/multinode/` |
| decode | `moe/` (the library MoE token-generation modules) | `core/moe/moe_tkg/`, `core/moe_block/` |
| decode | `moe/routed/`, `moe/sel_load_tp/`, `moe/selective_load/`, `moe/selective_load_t32/` | the model's routed-expert bodies under `experimental/` |
| decode | `norm/`, `quantization/`, `utils/` (with `trace_names`), `sampling/` | `core/subkernels/`, `core/quantization/`, `core/utils/`, `experimental/sampling/` |

A module whose fork name carried the model name drops it (`kimi_k3_tkg_kda` is `tkg_kda`, `kimi_k3_verify_kda` is
`verify_kda`). The prefill and decode copies of a library module are separate modules.

## Kernels

| Part | Kernel (entry) | Module | Use |
| --- | --- | --- | --- |
| prefill | `kimi_k3_kda_block_prefill` | `kda/kda_block_prefill.py` (with `kda_front`, `in_proj_conv`, `block_cc` and the scan `kda_prefill_v2_kernel`) | one call per KDA layer (gather, MXFP8 q/k/v/g/b, short convs, scan, norm and gate, o_proj, reduce-scatter) |
| prefill | `mx_quantize`, `pack_from_mx` | `kda/kda_front_mx_torch.py` | host MXFP8 packer of the KDA block front weights |
| prefill | `attn_res_cte`, `attn_res_bank_scores` | `attn_res/` | the AttnRes mixture of every layer; the bank-score table once a stage |
| prefill | `prefill_mx_linear`, `prefill_mx_head_linear` | `linear/` | the MXFP8 linears (MLA q_b, o_proj, MoE latent and shared); the MLA W_UK absorb |
| prefill | `bwmm_shard_on_block_mx` | `moe/bwmm_shard_on_block_mx.py` | the blockwise MXFP4 routed experts (SiTU activation) |
| prefill | `mla_flash_latent` | `attention/mla_flash_latent.py` | the MLA flash of a query tile (W_V before the merge, MXFP8 latent query) |
| prefill | `kimi_k3_mla_dcp_merge` | `attention/mla_dcp_merge.py` | the cross-rank merge of the flash partials |
| prefill | `kimi_k3_rows_write_after`, `kimi_k3_page_read`, `kimi_k3_page_write` | `page_io.py` | the MLA FP8 row writes and the drafter context rows |
| prefill | `kimi_k3_moe_token_runs` | `moe/token_map.py` | the MoE token map (two calls of at most 32 local experts) |
| decode | `kimi_k3_verify_model_2srv` | `multinode/verify_model_2srv.py` | the verify program |
| decode | `kimi_k3_draft_mla_step` | `draft_mla.py` | the draft program (the DSpark MLA drafter) |
| decode | `full_vocab_sample` | `sampling/full_vocab_sample.py` | target sampling over the TP32 vocab shards |
| decode | `kimi_k3_kda_a_exp_init_kernel` | `tkg_kda.py` | the init programs that widen `kda_params` |
| decode | host builders | the modules `decode/weights_host.py` and `decode/kernel_imports.py` name | per-rank weight layouts and constants |

The decode part is the import closure (top-level and function-level imports) of its entries and host builders.

Not forked, because the served prefill does not run them: the model's own `kda_front`, `kda_post`, `kda_prep`,
`kda_short_conv`, `moe_front`, `mla_merge_scale` and `quad_page_io`; the library's `mla_block_prefill`,
`moe_block_prefill`, `mla_flash_latent` (the prefill runs the model's own flash) and `moe_sel_load_tp`; the decode
kernels `attn_res_tkg`, `mla_decode_v2` and `moe_tkg_decode`; the scan plan `kimi_k3_prefill_kda_scan_plan` (the plugin
builds the scan constants itself); and the other blockwise MoE forms (`moe_cte.py`, `bwmm_shard_on_block.py`,
`bwmm_shard_on_block_v2`).

## Edits to the fork

- Layout. The files moved to the paths above, and each relative import names the module's new path. A module the fork
  imported from a package by its old name (`from . import kimi_k3_tkg_moe_2srv as moe_2srv_module`) is imported by its
  new name under the same local name.
- Alias imports. The fork's `experimental/kimi_k3/__init__.py` and `models/kimi_k3/__init__.py` installed the old
  module names of renamed files as aliases at import time (prefill: `trn_target`, `valid_row_window`,
  `kda_block_prefill.block_cc`, `block_front` and the blockwise MoE's `down_projection_mx`, `gate_up_projection_mx`,
  `moe_cte_mx_utils`, `moe_cte_utils`; decode: `mla_decode`, `kda_decode`, `moe_decode`, `dma_gate`, `explicit_sbuf`,
  `mx_layout`, `multinode.moe_decode_2srv`, `moe_tkg_decode.moe_tkg_decode`, `trace_names`, ...). Here each import names
  the real module, and those package files are gone.
- Package files. The prefill packages `attn_res/`, `kda/`, `moe/` and `linear/` keep a docstring and import nothing, so
  a host module such as `kda_front_mx_torch` imports without `nki`; `decode/moe/__init__.py` no longer imports
  `moe_tkg`. Import a kernel by its module.
- Names. Comments and docstrings that named another framework's modules, compiler builds or issue numbers are
  reworded; the KDA block torch reference's CPU rounding mode is `mode="torch"` (`_short_conv_torch`, `_l2norm_torch`,
  `_body_torch`).
- `page_io.kimi_k3_rows_write_after` takes a pool in the dtype of the rows (the uint8 MLA row pool with uint8 rows) as
  well as a bf16 pool; the bf16 pool keeps its behavior (bf16 rows, or FP8 rows into its bytes).
- `decode/multinode/tkg_moe_2srv.py`: the column out projection (`out_proj_form` `column`, which the served decode
  takes) compiles on neuronx-cc 2.28. `column_out_projection` records the load of its output all-gather as a gate
  consumer (`release_gate_consumer`), as the reduced form records the load of its output all-reduce. In
  `_load_mx_codes` and `_load_chunk_mx_tiles`, only the first MXFP8 code DMA of a series waits for `after` (a runtime
  edge), and each later code DMA follows that first DMA in the schedule (`_after_once`). The dependency-reduction pass
  of neuronx-cc 2.28 stops when one producer has several runtime edges. The kernel library has the same two edits.
  Without `after` (the reduced form) the code DMAs do not change.
- Decode comments. The comments and docstrings of the decode part no longer state measured performance numbers
  (times, rates, speedups and the step results of profiles). Each sentence keeps its meaning without the number, and
  each line stays in its place. No code changed.
- Prefill comments. The comments and docstrings of the prefill part no longer state measured performance numbers, as
  in the decode part. Each line stays in its place. No code changed.
- Comment references. The comments and docstrings of both parts no longer state internal references (run, profile
  and build tags, deployment tags, experiment names, test names and repository names) or the history of experiments
  (dates and tuning rounds). A comment names each module by its name here, not by its fork name, and describes in
  words a module or a function that this package does not have. The same edits correct typos. Each line stays in
  its place. No code changed.
- Comment wording. The comments and docstrings of both parts no longer describe an option by the history of the
  program ("today's program", "the program of before", "as before"). Each sentence names the program without the
  option, or the default, and tells when the off state is not the default. A sentence that said what an option "no
  longer" does now says, in the present tense, what the code does. A sentence that kept a form off "until a
  measurement" now states the current default. A name that this package does not have is
  described in words, and a mention of another model is replaced by a description of the feature. The commented-out
  code is removed: `is_sbuf_tensor` and `is_psum_tensor` in the three `kernel_helpers.py` copies, and an assert in
  `prefill/moe/bwmm_shard_on_I.py`. Four attribute docstrings move to right after the assignment that they document
  (`decode/tkg_moe.py` `ROUTER_DTYPES`; `decode/tkg_mla_paged.py` `_DEEP_SUMS`, `_WGATE_EARLY_FORMS` and
  `_MX_SCALE_PLAN_SLOT`). Lines are added, removed and moved. No code changed.

No kernel arithmetic, kernel name or environment variable name is changed.
