# MiMo-V2 on Neuron

`MiMoV2ForCausalLM` (XiaomiMiMo/MiMo-V2.6-Flash-RL, MiMo-V2.5) runs on the [runtime](../design/runtime.md) as an
aggregated server on one trn2.48xlarge, the text decoder only. The host is `sglang_neuron/models/mimo_v2/sglang_model.py`;
the target is InferenceX AgentX at 256K and 1M contexts. The checkpoint conversion follows the vllm-neuron port
([vllm-project/vllm-neuron#67](https://github.com/vllm-project/vllm-neuron/pull/67)); the attention does not, because
that port's eager fp32 attention gathers the whole context every step and cannot serve 256K.

**Status:** served on a trn2.48xlarge at 8K, 256K and 1M contexts with FP8 experts (the measurements below; Neuron SDK
2.33.0, the release image `pytorch-inference-sglang-neuron:v0.1.0` with this tree on `PYTHONPATH`), and with
`--speculative-algorithm NEURON_MTP` (the checkpoint's 3 MTP layers, [below](#mtp-neuron_mtp)) and the FP8 KV cache
(`--kv-cache-dtype fp8_e4m3`, [below](#fp8-kv-cache)) and DCP (`--dcp-size 8`, [below](#dcp---dcp-size-8)). PD runs over
NIXL on two nodes ([below](#pd-two-trn248xlarge-nixl-over-efa)). Neither the decode nor the prefill graphs hold an XLA
compute op (NKI kernels only), and the model imports nothing from other models' packages (`kernels/PROVENANCE.md` names
what was copied). The formal AgentX 1M points are [below](#agentx-1m-conc-8).

## The model

| Parameter | MiMo-V2.6-Flash-RL |
| --- | --- |
| hidden size, vocabulary | 4096, 152576 (untied LM head) |
| layers | 48: `hybrid_layer_pattern` 0 = full attention (9 layers), 1 = sliding window (39 layers) |
| full attention | 64 query / 4 KV heads, Q/K head 192, V head 128, RoPE base 1e7, no sink |
| sliding attention | 64 query / 8 KV heads, Q/K 192, V 128, RoPE base 1e4, window 128, a learned sink per head |
| RoPE | partial: the first 64 of 192 channels, as two contiguous halves (`rotate_half`) |
| V | times `attention_value_scale` (0.707) before it is cached |
| MLP | layer 0 dense (intermediate 16384); layers 1-47 MoE: 256 experts, top 8, intermediate 2048, no shared expert |
| router | sigmoid; noaux_tc with one group: top 8 of score + `e_score_correction_bias`, weights from the scores, renormalized |
| MTP | 3 layers (`model.mtp.layers.*`): sliding attention (the sliding kind's heads and RoPE, window 128, sinks) and a dense MLP (16384) each |
| context | `max_position_embeddings` 1048576 |

## The checkpoint and the weights

| Tensors | Stored as | Served as |
| --- | --- | --- |
| `self_attn.qkv_proj` | FP8 e4m3, 128 x 128 blocks, **pre-sharded**: 4 shards (the index's `metadata.tp_size`) of `[Q | K | V]`, a scale grid per section | bf16, each rank's heads re-sharded for any attention TP |
| dense MLP | FP8 e4m3, 128 x 128 blocks (`weight_scale_inv`) | bf16 |
| routed experts | **MXFP4** (`store_dtype: mxfp4`): uint8 `[R, C / 2]`, e8m0 `weight_scale` `[R, C / 32]` | **Trn2 FP8** (e4m3, max 240) with a power-of-two scale per output row, **without loss** |
| `o_proj`, router, sinks, embedding, LM head | bf16 | bf16 |

Each rank converts only its own shards as it reads them (`sglang_neuron/models/mimo_v2/weights.py`, `experts_fp8.py`).

**The experts in FP8, losslessly.** Trn2 has no MX matmul, so MXFP4 cannot run as stored. Each expert matrix is
dequantized and divided by the smallest power of two that brings its row's largest value to 240 or below. A power of
two only shifts exponents and an e2m1 value has two significant bits, so every quotient is an e4m3 number: over
all 36,096 expert matrices of the published checkpoint the quotients lie in [2^-6, 192] (2^-6 is e4m3's smallest
normal) and every round trip is bit-equal (`tools/mimo_v2/check_fp8_experts.py`; a `rowmax / 240` scale, which is not a
power of two, leaves about 7% of the values inexact, relative L2 8e-3 to 1e-2). A rank holds about 5 GB of weights (4.7 GB of experts) instead of 10 GB in bf16, and the full-attention
KV pool grows from 1.02M to 1.53M tokens a rank. The experts are always served in FP8.

## Attention

nkilib's attention kernels take one head dim for Q, K and V, and above 128 only multiples of 128 (`attention_tkg`:
`d_head % 128 == 0 or d_head <= 128`). The graphs run every head at **256 channels** (`config.KV_HEAD_DIM`):

- Q and K: the 192 channels re-ordered so the checkpoint's rotary pairs `(j, j + 32)` land on the kernels' `(j, j + 128)`,
  the 128 un-rotated channels after them in each half, then 64 zero channels. Q and K are permuted alike, so every dot
  product is the checkpoint's.
- V: its 128 channels times `attention_value_scale`, then 128 zero channels. The output projection reads the first 128.
- The softmax scale stays `192 ** -0.5`.

The prefill runs a chunk's input RMSNorm, QKV projection, RoPE, FP8 scaling and K/V write in one NKI kernel
(`kernels/prefill.prefill_front`), then nkilib's segmented attention over the prior and the chunk, with the window (it loads only a window's pages) and
the sinks. The KV pool is sized for the same 256 channels: `sglang_patches/mimo_v2_launch.py` sets SGLang's
`head_dim`, `v_head_dim`, `swa_head_dim` and `swa_v_head_dim` to 256 after `ModelConfig` derives them, and lets SGLang's
MiMoV2 fused-QKV TP check through, since the graphs re-shard the fused QKV themselves.

## The decode

The decode runs one form, with no switch back to XLA (the earlier per-layer-graph and half-kernel forms, measured
below, are gone):

| Form | Per layer |
| --- | --- |
| **one kernel a layer** | `kernels/layer.py`: both halves and both all-reduces in one call, no XLA op in a layer or between layers; the MoE layers keep the residual stream and both all-reduces in SBUF (SBUF-to-SBUF all-reduce, the program halves crossed by sendrecv); the dense layer 0 and the 3 MTP draft layers with nkilib's `mlp`; spans (NEURON_MTP) and the FP8 cache |

Measured on a trn2.48xlarge, TP64 / EP64, FP8 experts, `--context-length 8192 --chunked-prefill-size 2048
--max-running-requests 8`, 1000 tokens in and 256 out (`examples/mimo_v2/bench_decode.py`, medians):

| Form | TPOT c=1 | TPOT c=8 | output tok/s c=8 |
| --- | --- | --- | --- |
| per-layer graph | 22.17 ms | 25.36 ms | 296.0 |
| **two half kernels** | **19.45 ms** | **21.32 ms** | **347.8** |
| one kernel a layer | 20.55 ms | 22.51 ms | 330.7 |

With NEURON_MTP and the FP8 cache, `--context-length 8192 --max-running-requests 8`, 1000/4000-token prompts, a
decode step (TPOT x accept length) and GSM8K/1319:

| Form | step c=1 | step c=4 | step c=8 | GSM8K |
| --- | --- | --- | --- | --- |
| two half kernels, the all-reduces in the graph | 23.0 ms | 26.2 ms | 30.6 ms | 91.6% |
| one kernel a layer, the all-reduces through HBM | 24.0 ms | 26.2 ms | 31.1 ms | 90.8% |
| **one kernel a layer, residual and all-reduces in SBUF** | **23.2 ms** | **26.6 ms** | **29.6 ms** | **91.4%** |

`tools/mimo_v2/bench_cc.py` times the all-reduce alone, 96 of [32, 4096] bf16 over 64 ranks: 37.5 us each in the
graph, 45.1 us through HBM with the core barriers, 36.1 us SBUF to SBUF.

The first table is the older bf16-cache, no-MTP measurement, from before the one-kernel layer kept its residual and
all-reduces in SBUF; the second is that change, which made the one-kernel layer the only form.

**The layer kernels since** (the same 8K MTP FP8 server, step = TPOT x accept):

| Change | step c=1 | step c=4 | step c=8 |
| --- | --- | --- | --- |
| the zero-XLA step (sampling, token relay and the MTP tail in kernels) | 19.7 ms | 21.8 ms | 24.2 ms |
| `moe_half`: the 4 local experts split between the 2 programs (no per-expert sendrecv), whole-row down tiles; RMSNorm, FP8 input and each expert's activation in a transposed `[128, n x T]` layout; the norm weight transposed on the PE. `attn_half`: O-proj before the K/V write, W_o loaded early | 15.7 ms | 18.35 ms | 20.75 ms |
| `moe_half`: per-token scalars broadcast by stream shuffle; the programs' exchange overlaps the last expert | 15.7 ms | 17.8 ms | 19.95 ms |
| **`attn_half`: the sliding layers' mask vectorized** (the default) | **15.7 ms** | **17.75 ms** | **19.0 ms** |

GSM8K (400) 0.907-0.912 and greedy accept 3.42 throughout. The MoE half at 32 rows went from 259 to 177 us a layer; its
middle (every local expert's weights through the PE each step) is PE-bound, and the PE is HAM-throttled to about 60%
of its peak, so only fewer PE instructions help there. A decode row uses T of the 128 partitions: a `[T, H]`
elementwise op costs about H cycles, `[128, H x T / 128]` about T, which is what the transposed layouts buy.

Two switches (`env_registry.py`):

- `SGLANG_NEURON_MIMO_SWA_MASK_VEC` (default `1`): `attn_half.install_swa_mask_patch` replaces nkilib's
  `gen_mask_tkg._create_batch_masks_swa` (about 9 instructions a row and token) with the same mask built over
  `[128, tiles, rows, span]` in about 9 instructions a call; bit-equal at 1 / 2 / 8 rows and 4- / 64-block tables. `0`
  keeps nkilib's.
- `SGLANG_NEURON_MIMO_FA_EXIT` (default `0`): pass nkilib's attention a `max_context_len` (the batch's longest row, a
  rank's own rows under KVDP), so its flash-attention loop stops after that row's tiles. nkilib's resized-block-table
  load reads tile 0's blocks for every dynamic tile (sgl-project/sglang-neuron#4); `install_fa_exit_patch` fixes it
  (bit-equal from 20K to 150K, needles 12/12 and 8/8). It is off because the dynamic loop is slower than the unrolled
  one at AgentX's mix: P90 interactivity 46.0 with it, 48.6 without, on one node; 1P1D measured the same direction.

Tried and dropped: two layers fused into one kernel (zero gain: the layer kernels already run back to back in one
NEFF), a router weight prefetch, alternating DMA queues, DVE work moved to GpSimd, and a mask built once a step for
the sliding layers (0.9 ms a step at 8K, but it hung long-context servers at two rows or more; removed).
`tools/mimo_v2/bench_moe_half.py` times the MoE half on one core (a chain of R calls in one kernel, R=1 against R=9).

## Accuracy

GSM8K, all 1319 questions, 5-shot, greedy (SGLang's `sglang.test.few_shot_gsm8k`), on the trn2.48xlarge:

| Weights, decode | Context | GSM8K |
| --- | --- | --- |
| bf16 experts, per-layer graph | 8K | 88.9% |
| FP8 experts, per-layer graph | 1M | 89.3% |
| FP8 experts, two half kernels | 8K | 89.5% |
| FP8 experts, one kernel a layer | 8K | 89.8% |

All within the run's sampling error (about 0.85%). Needles (`examples/mimo_v2/run_needle.py`, a number in a repeated
filler text, found by the 9 full-attention layers alone): 9/9 from 1.5K to 6.5K tokens on each form; 18 of 20 from
119K to 249K tokens on the per-layer graph at 256K, the two misses at depth 0.1 of 119K and 179K (the same depth found
with other seeds); the long-context runs of the default form are in [Long contexts](#long-contexts).

### Checks

| Tool | Where | What |
| --- | --- | --- |
| `tools/mimo_v2/check_layout.py` | CPU, real checkpoint tensors (`fetch_tensors.py`) | the fused-QKV loader bit-equal to an independent dequantization at TP 64 / 16 / 8; the padded attention equal to HF eager attention to 1e-7 at matched rounding; MXFP4 and FP8-block dequantization bit-equal |
| `tools/mimo_v2/check_fp8_experts.py` | CPU, the whole checkpoint | all 36,096 expert matrices' power-of-two FP8 conversion bit-equal, quotients in [2^-6, 192] |
| `tools/mimo_v2/check_moe_half.py` | one NeuronCore | the MoE half kernel vs a torch reference, 1 / 4 / 8 rows: relative L2 1.7e-3; a reference without the selection bias fails |
| `tools/mimo_v2/check_attn_half.py` | one NeuronCore | the attention half kernel vs a torch reference and the per-layer graph's kernel, full and sliding (sinks): 1.7e-3 to the latter; the K/V rows in their slots, no other row written; a reference without sinks fails |
| `tools/mimo_v2/check_layer.py` | 4 NeuronCores | the one-kernel MoE layer bit-equal to the half kernels with host all-reduces at 4 and 8 rows; the dense layer 3.9e-3 to a torch MLP |

## Serving

```bash
CONTEXT=8192    bash examples/mimo_v2/run_server.sh                                         # bring-up
CONTEXT=262144  bash examples/mimo_v2/run_server.sh                                         # AgentX 256K
CONTEXT=1048576 MAX_RUNNING=8 SWA_RATIO=0.04 bash examples/mimo_v2/run_server.sh            # AgentX 1M
URL=http://127.0.0.1:30000 bash examples/mimo_v2/run_client.sh
python3 examples/mimo_v2/run_needle.py --tokens 8000 32000 120000 250000
python3 examples/mimo_v2/bench_decode.py --concurrency 1 8
```

`run_server.sh` is `--tp-size 64 --ep-size 64 --chunked-prefill-size 8192 --reasoning-parser mimo --tool-call-parser
mimo` with the radix cache and:

- **Hybrid SWA memory** (`catalog.ModelSpec.hybrid_swa_memory`): the sliding layers' KV lives in SGLang's window pool,
  each running request's last 128 tokens and a prefill chunk, so only the 9 full layers hold the whole context. Under the
  radix cache it needs `SGLANG_NEURON_HYBRID_SWA_RADIX=1`; without it every layer holds the whole context.
- **`--swa-full-tokens-ratio`** 0.06 (0.04 at 1M): SGLang's 0.8 gives the window pool most of the KV (a full-pool token
  is 9 layers x 1 KiB a rank, a window token 39 x 1 KiB). The plugin's step-0 check refuses a ratio the running requests
  can run short of and names the one that fits.
- **Past 256K, `NEURON_RT_IO_RING_CACHE_SIZE=4`** (the script sets it): with the runtime's default ring of 32 IO sets a
  program, the first 1M decode steps failed to allocate their DMA rings ("Failed to allocate DEVICE memory").
- **`--chunked-prefill-size` at most 8192**: the segmented prefill's KV segment tiles (`utils/bucket_utils`).
- **`SGLANG_NEURON_SAMPLER=topk_topp`**: a request that names no sampling takes the checkpoint's `generation_config`
  (temperature 1.0, top-p 0.95), as AgentX's do; the plugin's default greedy sampler answers those with a 400.

### MTP (`NEURON_MTP`)

`MTP=1 bash examples/mimo_v2/run_server.sh` adds `--speculative-algorithm NEURON_MTP` (`speculative/neuron_mtp.py`):
one decode step verifies the previous step's 3 drafts and drafts 3 more on the device (`models/mimo_v2/mtp.py`,
`mtp_step.py`, `runtime.py`: a device carry of each request slot's next tokens and length, the host's per-row control,
modelled on Hy3's step and copied, not imported; `model.MiMoV2Graph._mtp_decode`):

- **Verify**: every target layer runs each row's 4 tokens at once, nkilib's `attention_block_tkg` at `S_tkg = 4` over a
  span mask (`attention.span_mask`: the prior ends at the row's committed length, the sliding layers' window starts at
  each token's own position) and the 4 tokens' K/V written after it (`attention.span_slots`). The target's sample at
  each token checks the next; the row keeps the drafts up to the first mismatch and the target's token after them, so
  the output is the target's own sample run (greedy, or `topk_topp` keyed by position).
- **Draft**: MTP layer k reads the target's state before its final norm and the token k + 1 ahead
  (`eh_proj(cat(enorm(embed), hnorm(h)))`, as SGLang's multi-layer EAGLE runs MiMo's MTP: not chain-style, every layer
  reads the target's state); its draft is its greedy pick at the row's last kept position, and its tokens past that
  position are the drafts before it.
- **KV**: the MTP layers' KV are the pool's layers 48-50, in the window pool (`sglang_patches/mimo_v2_launch`: an
  AROUND hook on `ModelConfig._derive_hybrid_model` lists them among the sliding layers; the host sizes the pool 3
  layers deeper). The prefill writes them over each chunk, the target's greedy pick and the layers' drafts after its
  last token. The decode's sliding tables are one page wider (a step reaches 2 x 4 positions past the host's length).

Two traps a step must avoid, both of which collapse every later answer to one repeated token at concurrency above 1
and never show at concurrency 1 (`MiMoV2Runtime._allocated_table`, `mtp_step.span_inputs`; Hy3's step has the same fixes):

- the step's block table reaches past a request's allocation, where its request-table row still holds an earlier
  request's slots (now another request's or the radix tree's pages): those columns are -1;
- a padding row reads the padding slot's carry, whose length its own steps advance by one each step: its masks then
  opened the whole table and the K/V it wrote to the reserved slot carried what it read there, which every later
  attention over that page (a masked window page maps to it) took in. A padding row now decodes token 0 at position 0.

Measured at 8K (`CONTEXT=8192 MAX_RUNNING=16 MTP=1`), against the same server without MTP:

| | no MTP | NEURON_MTP, 3 layers |
| --- | --- | --- |
| GSM8K (5-shot, 1319, parallel 8) | 0.895 / 619 s | **0.895 / 287 s** (accepted length 3.39) |
| decode TPOT, c=1 (GSM8K prompts, 256 tokens) | 19.45 ms | **6.67 ms** (accepted 3.41) |
| decode TPOT, c=8 | 21.32 ms | **9.18 ms** (accepted 3.24), 743 tok/s |
| decode TPOT, c=16 | | **11.53 ms** (accepted 3.33), 1075 tok/s |

A step (4 tokens a row through the target, 3 MTP layer passes) costs 22.7 ms at c=1 and 29.7 ms at c=8, against the
one-token step's 19.45 and 21.32: the verify runs nkilib's attention and the MoE half kernel, not the attention half
kernel (`kernels/attn_half.py` takes one token a row).

### FP8 KV cache

`--kv-cache-dtype fp8_e4m3` (`models/mimo_v2/attention.py`): K and V pages in Trn2's e4m3 (at most 240), each value
times `SGLANG_NEURON_MIMO_KV_SCALE` (1.0 by default; the checkpoint has no KV scales), both token-major `[page, 256]`.
The platform pool packs FP8 K (two tokens a bf16-sized slot) for kernels that read it so up to 128 channels; MiMo
declares `KVLayout(fp8_k_packed=False)` because nkilib's decode attention reads FP8 heads of 256 only token-major.

- **Decode**: `attention_block_tkg` with `k_scale` / `v_scale` quantizes the new token's K/V itself (clip(x * scale)),
  the fused attention half kernel writes them as they come, 1 / k_scale goes into the softmax scale and 1 / v_scale
  into the output projection (`tools/mimo_v2/check_attn_half.py --fp8`).
- **Prefill**: nkilib's segmented attention reads FP8 K only packed, so the prefill packs the pages it reads
  (`attention.pack_prior`: a sliding layer its window's few pages, a full layer its table), by an FP8 gather and
  transpose (a uint8 view of the cache fails the compiler's type check: `Type mismatch uint8 vs float8_e4m3`)
  (`tools/mimo_v2/check_prefill_fp8.py`: equal to the bf16 kernel over the dequantized cache).

Measured at 8K with `MTP=1` (the same server as the MTP table, bf16 cache in brackets): GSM8K 0.914 (0.895), decode TPOT
6.54 / 8.84 / 11.48 ms at c = 1 / 8 / 16 (6.67 / 9.18 / 11.53); the pool holds 2.79M full-layer tokens a rank (1.53M).

### DCP (`--dcp-size 8`)

```bash
SGLANG_NEURON_HYBRID_SWA_RADIX=0 bash examples/mimo_v2/run_server.sh --dcp-size 8 --disable-hybrid-swa-memory
```

At TP64 every KV head is replicated on 8 (sliding) or 16 (full) consecutive ranks; `--dcp-size 8` keeps each head's
sequence in 8-way block-cyclic shards over those replicas (`distributed/decode_context_parallel.py`; at most 8: the
sliding layers' 8 KV heads), served aggregated (`catalog`: `decode_context_parallel`, `prefill_context_parallel`), on
one KV pool (`--disable-hybrid-swa-memory`). `models/mimo_v2/attention.py`:

- **Decode**: every layer's `attention_block_tkg` with `CP = 8` over its shard (the runtime's per-row bounds, the decoded
  token counted on its owner), the sliding layers' sinks as the group's (`group_sinks`: real on rank 0). nkilib's CP
  path holds Q and the output as `[d_head, rows]` tiles (heads of at most 128 channels); the vendored copy in
  `kernels/cp` gathers Q and merges the output one 128-channel d-tile at a time with nkilib's own collectives
  (`tools/mimo_v2/check_dcp_decode.py`, 8 ranks: every rank's head equals full attention over the unsharded
  sequence, with and without sinks).
- **Prefill, full layers** (no sinks): nkilib's KV-parallel segmented attention over the shards (Hy3's).
- **Prefill, sliding layers**: a chunk starts at a whole virtual page, so the keys before it its window reaches are the
  last block of the previous virtual page, on the group's rank 7; every rank takes it (an all-gather) and attends
  locally over it and the chunk's own K/V with the sinks (nkilib's KV-parallel kernel takes no sinks).

Measured at 8K (bf16 cache, no MTP): GSM8K 0.901 (1319, parallel 8); decode TPOT 33.87 / 46.48 / 53.94 ms at
c = 1 / 8 / 16 (19.45 / 21.32 ms without DCP: the DCP step runs nkilib's attention kernel with its collectives on all 48
layers, not the fused attention half). The pool holds 2620 pages of 128 tokens a rank for every layer, 2.68M tokens of
context over the group: DCP is the capacity point (1M contexts, more of them), not the latency one. With the FP8 cache
(`--kv-cache-dtype fp8_e4m3`): GSM8K 0.910, TPOT 34.96 / 46.88 ms at c = 1 / 8, 5240 pages a rank (5.37M tokens). Not yet with
NEURON_MTP or the hybrid window pool.

### PD (two trn2.48xlarge, NIXL over EFA)

A TP64 prefill on one node in front of a TP64 decode on the other, the plugin's balancer in front. The form AgentX ran
([below](#agentx-1m-conc-8)): NEURON_MTP on the decode, the FP8 cache and the hybrid window pool on both roles:

```bash
# both: the container with /dev/infiniband (--privileged), --network host and --init
export SGLANG_NEURON_PD_HYBRID_SWA=1 SGLANG_NEURON_MIMO_EXPERT_CACHE=1 SGLANG_NEURON_PRECOMPILE_CACHE=1
export CONTEXT=1048576 MAX_RUNNING=8 MEM_FRACTION=0.66 SWA_RATIO=0.04 MAX_TOTAL_TOKENS=2106752
PD="--kv-cache-dtype fp8_e4m3 --disaggregation-transfer-backend nixl"
# prefill node
SGLANG_NEURON_MIMO_PD_MTP_KV=1 MTP=0 HOST=<prefill ip> PORT=30001 bash examples/mimo_v2/run_server.sh $PD \
  --disaggregation-mode prefill --disaggregation-bootstrap-port 8998
# decode node
SGLANG_NEURON_HYBRID_SWA_RADIX=0 MTP=1 HOST=<decode ip> PORT=30002 bash examples/mimo_v2/run_server.sh $PD \
  --disaggregation-mode decode
python3 -m sglang_neuron.disaggregation.balancer --prefill http://<prefill ip>:30001 \
  --decode http://<decode ip>:30002 --bootstrap-port 8998 --port 8000
```

`SGLANG_NEURON_PD_HYBRID_SWA=1` sizes the decode's window pool without a prefill chunk in flight and adds the MTP
layers to the prefill's window layer ids; without it both roles take `--disable-hybrid-swa-memory` (one KV pool).
A container without `--init` keeps a crashed server's children as zombies, and the next launch waits on them.

Measured at a 64K context on one pool (bf16 cache, no MTP): GSM8K 0.900 (1319, parallel 8) through the balancer; decode
TPOT 20.92 / 23.09 / 26.23 ms at c = 1 / 8 / 16. With the hybrid window pool, MTP and the FP8 cache: GSM8K 0.916, accept
3.43, TPOT 6.86 / 8.20 / 8.23 ms at c = 1 / 8 / 16, the same as on one pool. Without EFA NIXL's LIBFABRIC backend
registers no device memory (`registerMem: no available backends for mem type 'VRAM_SEG'`), so a one-node pair
(`examples/mimo_v2/run_pd_one_node.sh`) needs a node with EFA too.

**NEURON_MTP and the FP8 cache on a PD pair**: the prefill takes `SGLANG_NEURON_MIMO_PD_MTP_KV=1` (it holds the MTP
layers and writes their KV over each chunk, so both pools hold 51 layers and the transfer carries the MTP layers' KV)
and the decode `--speculative-algorithm NEURON_MTP` (`MTP=1`); both `--kv-cache-dtype fp8_e4m3`. Measured at 64K:
GSM8K 0.920; accepted length 3.43 (the aggregated server's 3.39); decode TPOT 6.85 / 8.23 / 9.70 ms at c = 1 / 8 / 16,
1146 tok/s at c = 16 (the aggregated server's 11.53 ms there: the decode no longer pauses for prefill chunks). A pool of
51 layers holds 4917 pages of 128 tokens a rank (629K tokens).

**DP attention on the decode** (`--dp-size 8 --enable-dp-attention --enable-dp-attention-local-control-broadcast`, the
decode's `MAX_RUNNING=64`: a DP batch is a multiple of the attention TP, 8 rows a group): eight groups of attention TP 8
(8 query heads a rank over its one KV head), the experts over EP64 across them (`moe.MiMoV2MoE.serves_dp_attention`):
each MoE step all-gathers the groups' rows over the world, runs the rank's 4 experts over all of them, all-reduces over
the world and keeps its group's rows. The world, not the DP columns: on a Trn2 64-rank layout the columns (ranks 8
apart) form no collective group the runtime can build (`ENC:enc_init_comm ... no_hier no_mesh`). The TP64 prefill
hands its KV to the DP decode as is. Measured: GSM8K 0.904; decode TPOT 41.8 ms at every concurrency from 1 to 64
(the groups step in lockstep at 8 rows each), 181 / 643 / 1110 tok/s at c = 8 / 32 / 64, against the TP64 decode's
23.1 ms and 317 tok/s at c = 8 and 530 tok/s at its c = 16 ceiling: the throughput layout.

### HiCache over the window pool

`SGLANG_NEURON_HYBRID_SWA_HICACHE=1` with `--enable-hierarchical-cache --hicache-size <GB a rank>` keeps evicted radix
pages in host memory with hybrid SWA memory on: SGLang's hybrid pool assembler builds one host pool for the full pool
and one for the window pool (`build_hybrid_swa_group`), each the Neuron MHA host pool (`mem_cache/host_pool.py`), and
its controller moves a request's full pages and its window's pages between them. `--hicache-size` is per rank: 64
ranks x 8 GB is 512 GB of the node's 2 TB. Checked at 8K with MTP and the FP8 cache and a 32K-token device pool
(`--max-total-tokens 32768 --swa-full-tokens-ratio 1.0`): 60 distinct 2.3K-token prompts twice, the second pass hits
2257 tokens a prompt on average (mostly from the host); GSM8K 0.927 / 0.930 over two passes.

### Pools, measured (`--mem-fraction-static 0.85`, bf16 KV, a rank)

| Experts | full-attention pool | window pool |
| --- | --- | --- |
| bf16 | 7978 pages = 1,021,184 tokens (8.76 GB) | 479 pages, 61,184 tokens (2.28 GB), at ratio 0.06 |
| FP8 | 11981 pages = 1,533,568 tokens (13.16 GB) | at ratio 0.04 |

### Long contexts

| Context | Measured |
| --- | --- |
| 256K, per-layer graph, bf16 experts | first compile 26 min; needles above |
| 1M, per-layer graph, FP8 experts | first start 54 min (the decode graphs' trace is about 22 min of every start); GSM8K above |
| 256K, two half kernels, FP8 experts | needles 7/9 at 32K / 119K / 249K (32K 3/3; one miss at depth 0.5 of 119K and one at 0.1 of 249K); TPOT 20.83 ms at c=1, 29.71 ms at c=8 (1000 tokens in); a 100K-token prompt: TTFT 8.4 s, TPOT 22.4 ms |
| 1M, two half kernels, FP8 experts | needles 4/4 at 497K and 895K tokens (depths 0.1 and 0.9), the four requests in about 3.5 min |

The needle misses move with the decode's numerics (the per-layer graph missed others with the same seeds) and no
depth is lost every time; the filler text repeats two sentences, which a model can skim. Reading them against a GPU
reference on the same prompts is open.

### AgentX 1M, conc 8

InferenceX AgentX (agentx-bench `client/run_point.sh`, the full 1M corpus, 3600 s, a fresh server for the point),
`CONTEXT=1048576 MAX_RUNNING=8 SWA_RATIO=0.04`, NEURON_MTP, the FP8 cache, the layer kernels' defaults:

| Deployment | Tree | P50 / P90 interactivity | ITL P50 / P90 | TTFT P90 | output tok/s | a chip |
| --- | --- | --- | --- | --- | --- | --- |
| one trn2.48xlarge (16 chips) | 2f27e9a | 80.9 / 48.6 | 12.3 / 20.5 ms | 3.40 s | 283.4 | 17.71 |
| 1P1D, two trn2.48xlarge (32 chips), [above](#pd-two-trn248xlarge-nixl-over-efa) | 6eaa5cd (FA exit on) | 101.9 / 65.1 | 9.8 / 15.4 ms | 3.12 s | 300.6 | 9.39 |

The H100 SGLang reference on the same point: P90 interactivity 135.3, 47.2 output tok/s a GPU. On the 1P1D pair the
FA exit off (this tree's default) measured P90 interactivity 68.3 against 67.2 over the first 40 minutes (a partial
run). A decode step at long contexts is bound by the 9 full-attention layers (the K transposes on the PE: about 2.6 ms
a step at 100K rows) and by the experts' weights streaming through the PE every step; the per-step all-reduces wait
about 1.3 ms.

### Compile times and caches

A first start compiles every graph: about 13 min at 8K, 26 min at 256K, 54 min at 1M. The compile cache keys hash
the NKI kernel sources (`backend/compile_cache.py`), so a cache directory serves one kernel tree: after a kernel edit,
start with another `CACHE_DIR`.

## The reference GPU launch, flag by flag

| GPU flag | Here | Why |
| --- | --- | --- |
| `--tp 8 --dp 2 --enable-dp-attention` | `--tp-size 64 --ep-size 64` | The runtime refuses DP attention on an aggregated server (its DP ranks would mix prefill and decode graphs in a step) and serves it on a PD pair only. At TP64 each rank holds one KV head, replicated |
| `--enable-dp-lm-head` | not needed | the LM head is vocabulary-sharded over all 64 ranks |
| `--mm-enable-dp-encoder` | refused | the text decoder only; `--enable-multimodal` is refused, and a launch without it is served text-only |
| `--mem-fraction-static 0.65` | 0.85 | a NeuronCore's HBM, with the graphs' scratchpad outside the fraction |
| `--chunked-prefill-size 16384` | 8192 | the segmented prefill's KV segment tiles go up to 8192 |
| `--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --enable-multi-layer-eagle` | `--speculative-algorithm NEURON_MTP` (`MTP=1`) | the same chain (3 top-1 drafts, one per MTP layer, each layer reading the target's state before its norm), verified and drafted inside the decode graph; the chain-shape flags are filled in |
| `--reasoning-parser mimo --tool-call-parser mimo` | kept | SGLang's host-side parsers, which AgentX's tool calls need |
| `--trust-remote-code` | kept | |

## What a launch refuses

- DP attention but on a PD decode with `--ep-size` equal to `--tp-size`, `--pp-size`; `--dcp-size` above 8, or with the hybrid window pool or NEURON_MTP; a layout whose heads, KV heads, experts or intermediate sizes do not shard
  (every rank holds exactly one KV head: `--tp-size` a multiple of 8).
- A `--kv-cache-dtype` other than bf16 or `fp8_e4m3`, a dtype other than bfloat16.
- The split q/k/v projection layout (MiMo-V2-Flash), a scaled RoPE, grouped routing, shared experts, attention biases.
- `--enable-multimodal`, `--mm-enable-dp-encoder`.
- `NEURON_MTP` on a PD prefill (it takes `SGLANG_NEURON_MIMO_PD_MTP_KV=1` instead and writes the MTP layers' KV for the
  decode), with DCP or DP attention, with other chain-shape flags than 3 / 1 / 4, or with logprobs.

## Next

- **Long-context decode**, where AgentX spends its steps: the full layers' K transposes off the PE (a DMA transpose
  needs heads of at most 128 channels: K split into two 128-channel halves, or V stored at its 128), and block-sparse
  attention on the 9 full layers (per-page K summaries, a page selection every few layers, as Hy3's decode; an
  approximation, so gated on needles and AgentX accuracy).
- **The all-reduces**: about 1.3 ms a step waiting; the lever left is prefetching the next layer's weights across them.
- **One call a step** ([design](../design/mimo-v2-decode-megakernel.md), stage 3): the 48 layers looped in one kernel.
  Fusing two layers measured zero gain, so this is for launch overhead and cross-layer prefetch, not per-layer cost.
- **An attention at Q/K 192 and V 128** (no padding: 1.6x fewer KV bytes and reads).
- **Deployment**: a bigger decode batch behind the PD prefill, replicas, DP attention on the PD decode (measured as the
  throughput layout above).
- **FP8 KV cache**: per-layer KV scales from a calibration.
- **DCP**: with NEURON_MTP and over the hybrid window pool.
