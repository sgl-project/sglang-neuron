# MiMo-V2 on Neuron

`MiMoV2ForCausalLM` (XiaomiMiMo/MiMo-V2.6-Flash-RL, MiMo-V2.5) runs on the [runtime](../design/runtime.md) as an
aggregated server on one trn2.48xlarge, the text decoder only. The host is `sglang_neuron/models/mimo_v2/sglang_model.py`;
the target is InferenceX AgentX at 256K and 1M contexts. The checkpoint conversion follows the vllm-neuron port
([vllm-project/vllm-neuron#67](https://github.com/vllm-project/vllm-neuron/pull/67)); the attention does not, because
that port's eager fp32 attention gathers the whole context every step and cannot serve 256K.

**Status:** served on a trn2.48xlarge at 8K, 256K and 1M contexts with FP8 experts (the measurements below; Neuron SDK
2.33.0, the release image `pytorch-inference-sglang-neuron:v0.1.0` with this tree on `PYTHONPATH`). Not yet: MTP, an FP8
KV cache, DP attention / DCP, an AgentX run.

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
| MTP | 3 layers (`model.mtp.layers.*`), not loaded yet |
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
KV pool grows from 1.02M to 1.53M tokens a rank. `SGLANG_NEURON_MIMO_EXPERTS=bf16` keeps the bf16 experts (also exact),
the reference.

## Attention

nkilib's attention kernels take one head dim for Q, K and V, and above 128 only multiples of 128 (`attention_tkg`:
`d_head % 128 == 0 or d_head <= 128`). The graphs run every head at **256 channels** (`config.KV_HEAD_DIM`):

- Q and K: the 192 channels re-ordered so the checkpoint's rotary pairs `(j, j + 32)` land on the kernels' `(j, j + 128)`,
  the 128 un-rotated channels after them in each half, then 64 zero channels. Q and K are permuted alike, so every dot
  product is the checkpoint's.
- V: its 128 channels times `attention_value_scale`, then 128 zero channels. The output projection reads the first 128.
- The softmax scale stays `192 ** -0.5`.

The prefill runs the QKV projection and RoPE in the graph, the chunk's K/V written by an NKI scatter (`kv_write.py`),
then nkilib's segmented attention over the prior and the chunk, with the window (it loads only a window's pages) and
the sinks. The KV pool is sized for the same 256 channels: `sglang_patches/mimo_v2_launch.py` sets SGLang's
`head_dim`, `v_head_dim`, `swa_head_dim` and `swa_v_head_dim` to 256 after `ModelConfig` derives them, and lets SGLang's
MiMoV2 fused-QKV TP check through, since the graphs re-shard the fused QKV themselves.

## The decode

Three forms, chosen by environment (the default is the fastest measured):

| Form | Per layer | Switch |
| --- | --- | --- |
| per-layer graph | nkilib `attention_block_tkg` (`W_out=None, update_cache=False` above 128 channels), the K/V scatter, the output projection, the router and nkilib's `moe_tkg` in the graph | `SGLANG_NEURON_MIMO_DECODE_ATTN_FUSED=0 SGLANG_NEURON_MIMO_DECODE_MOE_FUSED=0` |
| **two half kernels** (default) | `kernels/attn_half.py`: the attention block, the K/V write and the output projection in one call; `kernels/moe_half.py`: the RMSNorm, the noaux router, the 4 local FP8 experts in one call; the two all-reduces in the graph | the default |
| one kernel a layer | `kernels/layer.py`: both halves and both all-reduces in one call, no XLA op in a layer (the dense layer 0 with nkilib's `mlp`) | `SGLANG_NEURON_MIMO_DECODE_LAYER_KERNEL=1` |

Measured on a trn2.48xlarge, TP64 / EP64, FP8 experts, `--context-length 8192 --chunked-prefill-size 2048
--max-running-requests 8`, 1000 tokens in and 256 out (`examples/mimo_v2/bench_decode.py`, medians):

| Form | TPOT c=1 | TPOT c=8 | output tok/s c=8 |
| --- | --- | --- | --- |
| per-layer graph | 22.17 ms | 25.36 ms | 296.0 |
| **two half kernels** | **19.45 ms** | **21.32 ms** | **347.8** |
| one kernel a layer | 20.55 ms | 22.51 ms | 330.7 |

The one-kernel layer is correct (below) but about 1.1 ms a step slower than the half kernels with the graph's
all-reduces: its 96 in-kernel all-reduces of 64 ranks, each with its fences, cost more than the graph's. Making them
cheaper (SBUF-to-SBUF all-reduce and residual, as nkilib's `transformer_tkg` `sbuf_residual_and_cc`) and running every
layer in one call are the next steps ([the design](../design/mimo-v2-decode-megakernel.md)).

`SGLANG_NEURON_MIMO_FULL_DECODE=posid` runs the per-layer graph's full layers in the kernel's position mode (no early
exit: nkilib's is wrong past 8K, sgl-project/sglang-neuron#4). `SGLANG_NEURON_MIMO_KV_WRITE_TORCH=1` writes the
per-layer graph's K/V with `index_put_`, to rule the scatter kernel out.

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
| `--speculative-algorithm EAGLE` (multi-layer, 3 steps) | not yet | the plugin serves only `NEURON_MTP`, a model's own in-graph MTP step; MiMo's 3 MTP layers are a next step |
| `--reasoning-parser mimo --tool-call-parser mimo` | kept | SGLang's host-side parsers, which AgentX's tool calls need |
| `--trust-remote-code` | kept | |

## What a launch refuses

- DP attention, `--dcp-size`, `--pp-size`; a layout whose heads, KV heads, experts or intermediate sizes do not shard
  (every rank holds exactly one KV head: `--tp-size` a multiple of 8).
- Any `--kv-cache-dtype` but bf16 (for now), a dtype other than bfloat16.
- The split q/k/v projection layout (MiMo-V2-Flash), a scaled RoPE, grouped routing, shared experts, attention biases.
- `--enable-multimodal`, `--mm-enable-dp-encoder`.

## Next

- **The decode megakernel** ([design](../design/mimo-v2-decode-megakernel.md)): cheaper in-kernel collectives, then
  every layer in one call, then an attention at Q/K 192 and V 128 (no padding: 1.6x fewer KV bytes and reads).
- **NEURON_MTP** with the 3 MTP layers (`mtp_host`, as Hy3's `models/hy_v3/mtp_step.py`).
- **FP8 KV cache**: heads of 256 need the unpacked FP8 layout (nkilib packs FP8 K only up to 128 channels) and KV scales
  from a calibration, as Hy3's (vllm-project/vllm-neuron#73).
- **PD with DP attention / DCP**: at TP64 a full layer's KV head is replicated on 16 ranks, which DP attention (attention
  TP 8: 2 ranks a head) or DCP removes; both run on a PD pair only, where the window pool is not served yet.
