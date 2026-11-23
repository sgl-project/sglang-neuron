# Hy4-Flash on Neuron

`HYV4FlashForConditionalGeneration` ([tencent/Hy4-flash-FP8](https://huggingface.co/tencent/Hy4-flash-FP8)) runs its
text decoder on the [runtime](../design/runtime.md) as an aggregated server on attention TP32 with the 512 routed
experts over EP32: half a trn2.48xlarge, 32 logical NeuronCores. The model is in `sglang_neuron/models/hy_v4_flash/`.

## The model

Hy4-Flash is 42 layers of gated MLA (`q_lora_rank` 1024, `kv_lora_rank` 512, Q/K heads of 192 + 64 RoPE channels,
V heads of 256) with a learned sink logit per head and an output gated by `sigmoid(gate_proj(x))`, closed by
`post_proj_norm`. Layers interleave two kinds:

| | sparse layers (12) | sliding layers (30) |
| --- | --- | --- |
| heads (per rank at TP32) | 32 (1) | 64 (2) |
| keys a query attends | the IndexPool's 2048 tokens | the 512 positions ending at its own |
| RoPE base | 1e7 | 1e4 |

- **IndexPool.** A sparse layer pools every 4 tokens' indexer keys into one gated, RMS-normed group key (written
  when the group completes, one row per 4 KV slots); a query scores every complete group before its own with 16
  relu heads and keeps the 511 best, plus its own group. The top-511 is a bisection on the threshold
  (`ops.top_count_mask`), ties kept in group order. A decode gathers each row's 2048 tokens (`ops.compact_groups`); a
  prefill up to a 16K table masks the table's scores, above it gathers per block of 256 queries.
- **iHC.** The residual is 4 streams; each block reads their sigmoid-weighted mix and adds its output to each stream
  times a per-stream gate (fp32, as the reference).
- **LatentMoE.** 16 of 512 experts a token, in a 1536-wide latent space between `pre_expert_proj` and
  `post_expert_proj`, their SwiGLU clamped at 10, beside a shared expert. The experts load as Trn2 FP8 (e4m3, a
  power-of-two scale per output row, requantized from the checkpoint's 128 x 128 blocks) and run in nkilib's
  `moe_tkg` (decode) and `moe_cte` hybrid I-sharded form (prefill, empty blocks skipped).
- **Engram.** Before 4 layers a hashed bigram and trigram lookup (16 hash heads, ~165M rows x 96 a layer) is gated per
  stream and added to the streams. The tables split by rows over the TP (about 1 GB a rank a layer in bf16; under the
  checkpoint's `engram_fp8_embedding`, the reference's FP8 round trip as e4m3fn bytes and a scale a row). The hash is
  int32 arithmetic on 7-bit chunks of each token's product, precomputed on the host: a NeuronCore's int64 multiply,
  XOR and remainder are not exact. Each KV slot keeps its token id, so a step reads its n-grams' earlier tokens
  through the block table.

The KV cache is the platform's MHA pool holding each layer's 576-wide MLA row as one KV head of 288 channels (its
first half in K, its second in V), the sliding layers' in SGLang's hybrid SWA window pool
(`sglang_patches/hy4_flash_launch.py` gives SGLang 43881a7 the config type and this geometry). The config class is a
text-only copy of upstream SGLang's (`models/hy_v4_flash/hf_config.py`).

## Serving

The commands below run inside the container with the checkpoint mounted at `/models`; the
[container guide](../guides/container.md) has the `docker run`.

```bash
NEURON_RT_VISIBLE_CORES=0-31 SGLANG_NEURON_PRECOMPILE_CACHE=1 \
python3 -m sglang.launch_server --model-path /models/Hy4-flash-FP8 --served-model-name hy4-flash \
  --tp-size 32 --ep-size 32 --context-length 9216 --chunked-prefill-size 2048 \
  --max-running-requests 64 --max-total-tokens 200000 --cuda-graph-bs-decode 1 16 64 \
  --disable-radix-cache --host 0.0.0.0 --port 30000
```

The launch fixes the layout: `--tp-size 32` (one KV head a rank) with `--ep-size` equal to it, no DP attention, PP or
DCP, the text decoder only (`--enable-multimodal` and `--speculative-algorithm` are refused), a bf16 KV cache. Both
graph families take the context in rungs (`SGLANG_NEURON_HY4F_CONTEXTS`, 16K, 64K, 256K and 1M by default, capped at
`--context-length`): a short request on a long-context server runs the short graphs. A cold start compiles every
token bucket and rung; rank 0 traces and queues each family's compiles before the other ranks trace
(`runtime.py`), so each graph compiles once.

## Checks

`tools/hy4_flash/cpu_check.py` runs the graphs on the CPU at TP 1 against golden outputs of the transformers
reference (`modeling_hy_v4_flash`, transformers#50903): per-layer residual streams, the prefill's logits and
teacher-forced decode steps. `tools/hy4_flash/mini_compile.py` compiles (and with `--bench` / `--profile` times or
profiles) the first layers on one NeuronCore at a TP's per-rank shapes. On a trn2.48xlarge the server's greedy
continuations of a 35-token chat prompt and of a 3063-token retrieval prompt (sparse selection active) match the
reference's first 32 tokens.

## Performance

The random 8192 / 1024 benchmark (`python3 -m sglang.bench_serving --dataset-name random --random-input-len 8192
--random-output-len 1024 --random-range-ratio 1.0`) at concurrency 1, 10 prompts: TTFT 2292 ms, TPOT 51.75 ms, 167
tokens/s. The graphs are still largely torch; the decode attention, the IndexPool and the norms are next to move to
kernels.
