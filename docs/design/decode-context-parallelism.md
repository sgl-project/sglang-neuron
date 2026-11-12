# Decode context parallelism

## Overview

Decode context parallelism (DCP, SGLang's `--dcp-size D`) splits every sequence's KV cache across D ranks of an
attention-TP group: each holds the same KV head and a different share of the tokens, attends over its share, and merges
the partial results across the group. A decode server then holds D times the context in the same KV pool. The cost is
three collectives per layer over the group: gather the query heads, gather the softmax statistics, and merge the
outputs (a reduce-scatter below four ranks, an all-to-all from four).

The plugin serves DCP on two PD decodes: GPT-OSS's per-layer decode, as one TP group, and Kimi-K3's two-node decode,
DCP32 inside each of its eight DP-attention groups ([Kimi-K3](../model-recipes/kimi-k3.md#pd-decode)). A GPT-OSS
prefill runs without DCP and sends each DCP rank the whole pages it keeps. The Kimi-K3 PD prefill keeps its MLA rows in
DCP placement too, DCP16 over the TP group of each pipeline stage, and sends them to the decode in whole page stripes.

## Key concepts

- **Virtual page.** Under DCP SGLang's allocator page is D pool pages (P tokens each), one block on each rank of the
  group. The allocator holds D times the pool's tokens and SGLang admits against that; the pool keeps its shape, one
  block per virtual page.
- **Page interleave.** Rank r keeps tokens `[r * P, (r + 1) * P)` of each virtual page, in its block of that page's
  index: SGLang's page-interleaved DCP layout (`layers/dcp/layout.py` at `interleave_size = P`), as on its NPU path.
  Block tables are the same on every rank; only the slots a rank writes, how many tokens it keeps and whether it owns
  the decoded token differ.
- **Owner.** The decoded token belongs to the rank whose share of its virtual page holds its position. Only the owner
  writes its K and V and counts it in attention; the other ranks write it to SGLang's reserved slot and mask it.
- **Group sinks.** GPT-OSS adds a learned sink to each head's softmax denominator, and the kernel adds it on every
  rank, so rank 0 holds the real sinks and the others a large negative value. The merged softmax counts each sink
  once.

## How it works

1. **Launch.** `distributed.decode_context_parallel.validate_dcp_launch` serves `--dcp-size` above 1 only on a PD decode
   of a model that declares DCP (`ModelSpec.decode_context_parallel`), as one TP group (no `--dp-size`, no DP attention)
   or in a decode layout the model declares, with that layout's DP-attention groups and DCP size
   (`dcp_inside_dp_attention`), with an attention TP size D divides, and without PP, another `--dcp-comm-backend` or the
   decode radix cache, naming every reason it refuses. It also serves the PD prefill of a model that declares
   `prefill_context_parallel` ([Limitations](#limitations)). It runs in the platform's defaults, before any scheduler
   admits. The GPT-OSS host then checks that each rank of a group holds the same single KV head (`validate_dcp_heads`).
2. **Groups.** SGLang builds DCP groups only on CUDA, HIP, NPU and XPU, so on a DCP launch the plugin rebuilds
   `initialize_model_parallel` with that condition extended to the platform's declaration
   (`sglang_patches/dcp_groups.py`). It mirrors each DCP group as a plugin group
   (`parallel.groups.get_neuron_dcp_group`), over which the kernel's collectives compile.
3. **Pool, allocator and metadata.** The pool keeps its place in the group and its page interleave
   (`NeuronMHATokenToKVPool.dcp`, `dcp_interleave`). The allocator must be the pool widened by D on both axes, or the
   launch is refused. The attention metadata indexes virtual pages and writes each new token to its owner's local
   slot (`attention/backend.py`).
4. **Decode inputs.** Each rank's decode table must be whole `2 x LNC x P_MAX`-token tiles, the unit at which the
   kernel reads the packed FP8 K cache in place, and end at least one virtual page past every row's last token, since
   the kernel keeps the decoded token in the table's last row. `decode_context_parallel.decode_context` rounds the
   context up to whole virtual pages, adds one, then rounds up to whole tiles per rank. Each step uploads, with its
   other inputs, each row's prior length on this rank (`dcp_kv_end`), the owner gate (`dcp_active_owner`) and, for the
   sliding layers, the window's bounds in the trimmed table (`dcp_kv_start`), which also ends one virtual page past the
   decoded token's page.
5. **Attention.** Each rank's mask covers the group's gathered heads, ends at its own prior and gates the decoded token
   to its owner. nkilib's `attention_block_tkg` runs with `CP = D` over the DCP partition: it gathers the query heads,
   attends over this rank's share and merges the partial softmaxes, each rank getting its own heads back
   (`ops/attention/attention_decode.py`). The torch path does the same with an all-gather and a reduce-scatter.
6. **PD transfer.** Prefill page k (P tokens) belongs to DCP rank `k mod D`, at its block of virtual page `k // D`.
   The decode announces its interleave in a trailing registration frame, and every PD prefill accepts an MHA DCP
   decode whose interleave is its page and sends each DCP rank the pages it owns, whole or as a head slice
   (`sglang_patches/dcp_page_transfer.py`).

Steps 3 to 6 are the per-layer path over an MHA pool. The Kimi-K3 decode attends over the MLA rows of its DCP groups in
its own programs (`sglang_neuron/models/kimi_k3/decode/`), and its prefill sends them in stripes
([Limitations](#limitations)).

## Configuration

| Setting | Value |
| --- | --- |
| `--dcp-size D` | the DCP group's ranks: on the decode server of one TP group (no DP), D divides the TP size, which is a multiple of D x the model's KV heads; in a declared decode layout, the layout's DCP size (Kimi-K3: 32, one DCP group a DP-attention group); on the PD prefill of a model with `prefill_context_parallel`, the stage's TP size (Kimi-K3: 16) |
| `--max-total-tokens` | the KV pool of one rank; SGLang admits D times it |
| `--context-length` | any; the runtime covers one virtual page past the context rounded up to whole virtual pages, then rounds each rank's table up to whole packed tiles |
| decode batch | at most `128 / (query heads per rank x D)` rows when a rank holds more than one query head (the kernel's DCP tile) |
| the prefill server | without DCP, at an attention TP that pairs by whole pages or head slices; a prefill that already replicates KV heads (attention TP above the KV head count) cannot feed a decode of larger attention TP, with or without DCP, and refuses it when the decode registers. A DCP prefill (Kimi-K3) feeds a decode whose DCP size is a larger multiple of its own, by stripes |

## Supported configurations

- GPT-OSS-120B's per-layer MXFP4 decode, on the PD decode role, with an FP8 KV cache of 128-token pages.
- A prefill of attention TP4 or TP8 into a decode of attention TP16 with `--dcp-size 2`, EP16. The head slice of a
  TP4 prefill page is shared by both ranks of a DCP group.
- The DCP sizes validated: 2 and 4 (the all-to-all merge from 4).
- Kimi-K3's two-node PD decode: TP256 as eight DP-attention groups of attention TP32, EP256 and `--dcp-size 32`, one
  DCP group a DP-attention group, with an FP8 MLA cache of 128-token pages. Its PD prefill runs `--dcp-size 16` over
  each of its eight TP16 pipeline stages, and sends each MLA stripe whole to the decode rank that keeps it
  ([Kimi-K3](../model-recipes/kimi-k3.md#serving-a-decode-over-nixl)).

## Limitations

- **Prefill and aggregated serving.** Not served: a prefill over a prior sharded across the group would need nkilib's
  KV-parallel segmented prefill kernel to take the attention sink, which it does not. A model whose own prefill stores
  and attends over the group's shards sets `prefill_context_parallel` in its catalog entry (Kimi-K3); its PD prefill
  role is then served DCP, pipelined or not (each stage's DCP group is ranks of that stage's TP group), on the same
  placement and allocator check, the MLA pool included (`NeuronMLATokenToKVPool.dcp`). Such a prefill hands its MLA
  pages to a decode of a larger DCP size by whole page stripes (`disaggregation/dcp_stripes.py`, through
  `sglang_patches/dcp_page_transfer.py`): stripe `j` goes from prefill rank `j % Cp` to decode rank `j % Cd`, and the
  recurrent state by attention heads as SGLang slices it between TP sizes.
- **The giga-kernel decode.** No DCP; it attends over each rank's whole KV in one kernel call.
- **PP.** Not on a DCP decode: each pipelined decode stage pairs its own pool with the prefill's, which the page
  interleave does not follow. The DCP prefill of a model with `prefill_context_parallel` is served pipelined.
- **MLA models on the decode role.** Kimi-K3 is the one MLA model whose catalog entry declares
  `decode_context_parallel` (the launch gate, `dcp_launch_problems`), on its declared two-node decode. A prefill
  without DCP is refused as the peer of an MLA or DSA DCP decode at registration
  (`dcp_page_transfer.refuse_mla_page_peer`): upstream's MLA plan places tokens round-robin and sends no DSA index
  keys, while the Neuron MLA pool keeps whole blocks per DCP rank (`mem_cache/cp_pages.py`). A DCP prefill's stripe
  hand-off (above) is not refused. Context parallelism outside `--dcp-size` is the model's own: a declared token-CP
  layout (`model_executor/kv_layout.KVLayout.cp_size`, which Qwen3.5's KV-CP decode declares over its own pool), and a
  megakernel that shards its attention over its TP group itself (GLM-5.2: `--dcp-size 1`).
- **DP attention and DP replicas.** Refused (`--dp-size`, `--enable-dp-attention`, in `dcp_launch_problems`), except in
  a decode layout the model declares (`dcp_inside_dp_attention`): Kimi-K3's, whose DCP groups are its eight
  DP-attention groups of 32 ranks. No run covers another DCP decode of several DP groups.
- **The decode radix cache.** Refused (`--disaggregation-decode-enable-radix-cache`, in `dcp_launch_problems`): no run
  covers the two together. SGLang pages a DCP decode's tree at the virtual page, so a reused prefix would be whole
  virtual pages, and the prefill's page plan sends each DCP rank its pages after it.
- **Per-step cost.** Three collectives per layer per step, so a step costs more device time than at the same attention
  TP without DCP. DCP pays where the context or the number of running sequences needs the capacity.

## Upstream SGLang interactions

- `layers/dcp/layout.py`: the placement uses upstream's `get_dcp_lens` and `localize_dcp_indices` at
  `interleave_size = P`.
- `distributed/parallel_state.initialize_model_parallel` (rebuilt, `sglang_patches/dcp_groups.py`): the DCP groups
  are upstream's. Only the platform condition takes the platform's declaration.
- `mem_cache/kv_cache_configurator.py`: SGLang builds the platform's paged allocator over `size x D` tokens in pages of
  `page x D`. The allocator refuses any other size.
- PD (`sglang_patches/dcp_page_transfer.py`, against upstream's NIXL connection): the decode's registration frame and
  pairing (`try_ensure_parallel_info` rebuilt with its MHA check extended), the prefill's relayout decision, peer
  check, payload preparation, chunk plan and send. A peer without DCP takes upstream's paths unchanged.

## Implementation

| Module | Role |
| --- | --- |
| `distributed/decode_context_parallel.py` | `DcpLayout`, `decode_context`, `group_sinks`, the launch, head and batch checks |
| `sglang_patches/dcp_groups.py` | SGLang's DCP groups on this platform |
| `parallel/state.py`, `parallel/groups.py` | the plugin's DCP group |
| `mem_cache/memory_pool.py`, `mem_cache/allocator.py`, `attention/backend.py` | the pool's placement, the allocator check, the metadata |
| `model_executor/graph_runner.py` | the DCP decode contexts (`BucketConfig.kv_shards`) and each step's bounds (`_dcp_bounds`) |
| `ops/attention/attention_decode.py`, `attention_decode_mask.py` | the kernel's CP path, the torch merge, the owner gate |
| `models/gpt_oss` | the group sinks, the op's group, the masks of the gathered heads, the launch checks |
| `sglang_patches/dcp_page_transfer.py` | the PD page plan |
| `disaggregation/dcp_stripes.py` | the stripe plan and rank pairing of a DCP prefill's hand-off to a decode of a larger DCP size |
| `models/kimi_k3/decode/capability.py`, `models/kimi_k3/decode/` | the declared two-node decode layout, and the decode programs over each DCP group's MLA rows |

## Validation

- `tools/check_dcp_decode_attention.py` (hardware, torchrun): the kernel's CP path on real collectives, D = 2
  (reduce-scatter) and 4 (all-to-all), for full and sliding masks and plain and packed FP8 K. Every rank's heads must
  equal the single-rank kernel decode and the torch path, and the decoded token's K and V land in its owner's slot;
  with the real sinks on every rank, or without the owner gate, the result must be off by several times the decode's
  own error.
