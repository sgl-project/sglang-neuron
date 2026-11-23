# Qwen3.8 on Neuron

`Qwen3_5ForConditionalGeneration` (the text model of `Qwen/Qwen3.8-27B`) runs on the [runtime](../design/runtime.md) in
bf16 as an aggregated server, TP8 on one trn2.3xlarge. The host is `sglang_neuron/models/qwen38/sglang_model.py` and the
graphs are `sglang_neuron/models/qwen38/graph.py`. The graphs serve text only; the checkpoint's vision tower is not
loaded.

## The model

Qwen3.8-27B is the dense member of the Qwen3.5 family: 64 decoder layers in a pattern of three Gated DeltaNet (GDN)
layers then one gated full-attention layer, a SwiGLU MLP in every layer, and an untied LM head.

| | GDN layers (48) | Attention layers (16) |
| --- | --- | --- |
| Heads | 16 key heads, 48 value heads, head dim 128 | 24 query heads, 4 KV heads, head dim 256 |
| Mixer | causal conv (kernel 4), gated delta rule, gated RMSNorm | partial RoPE (64 of 256 dims, theta 1e7), sigmoid output gate |
| Per-request memory | a recurrent state of fixed size | paged KV, growing with the context |

The layer norms, the final norm and the attention query and key norms are zero-centred (`1 + w`); the GDN gated norm
is plain. The attention's output gate is folded into `q_proj` (per head, the query rows then the gate rows).

Both layer kinds run the Qwen3.5 per-layer NKI kernels unchanged (`sglang_neuron/models/qwen35/kernels/giga`, pinned by
their manifest): `gdn_decode_tkg` and `gdn_prefill_tkg` for the GDN layers, `attn_decode_tkg` and `attn_prefill_tkg` in
their paged mode for the attention layers. The graph stitches them with the norms, the MLP and the all-reduces.

### Layout

The kernels run one rank per physical NeuronCore (LNC 1, which the catalog entry sets), so a trn2.3xlarge serves the
model at TP8 over its eight cores. Per rank: 6 GDN value heads, 2 GDN key heads, 3 query heads padded to 4 with a zero
head (the attention kernel takes an even count), 1 KV head, and an eighth of the MLP. The weights take 6.8 GB of each
core's 12 GB.

### State and KV

- **GDN state.** The platform's state pool (`mem_cache/state_spec.gdn_spec`) with a working bank: before a step the
  batch's state slots are loaded into contiguous bank rows, the kernels advance the rows in place, and the rows are
  published back after the step (`models/qwen38/runtime.py`). A load or publish covers a power-of-two bucket of rows,
  compiled when the pool is bound, so no step compiles a new copy while serving. The publish is an indirect-DMA NKI
  write in place (`models/qwen38/bank_ops.py`): as eager torch indexing it rewrote the whole state pool, 6.1 ms per
  step for one row into the radix cache's 84-slot pool, and the kernel takes 0.7 ms.
- **Attention KV.** A page pool of the 16 attention layers (`models/qwen38/kv_pool.py`), read through SGLang's block
  table. A prompt is prefilled in chunks of whole pages; each chunk carries the GDN state and reads the request's
  earlier pages as its prior.

## Serving

Aggregated, on the eight NeuronCores of a trn2.3xlarge, with the validated flags:

```bash
python3 -m sglang.launch_server --model-path /models/Qwen3.8-27B --tp-size 8 \
  --context-length 16384 --page-size 512 --chunked-prefill-size 2048 --cuda-graph-bs-prefill 512 \
  --max-running-requests 4 --max-total-tokens 65536 --mem-fraction-static 0.85 \
  --radix-cache-backend neuron-chunk --mamba-radix-cache-strategy extra_buffer --mamba-track-interval 2048 \
  --dtype bfloat16 \
  --host 0.0.0.0 --port 30000
```

`--host 0.0.0.0` listens inside the container; publish the port on the host's loopback only
(`-p 127.0.0.1:30000:30000`), as the [container guide](../guides/container.md#3-start-a-container) explains.
[examples/qwen38_27b](../../examples/qwen38_27b/) has the server, a client and the accuracy check as scripts.

- **Context.** `--context-length 32768` with `--max-running-requests` 2 or 4 was also run (results below).
- **Memory.** After the weights (6.8 GB of a core's 12 GB), a core holds the KV pool, the state slots and the loaded
  programs, and more rows compile more decode graphs. Run on a trn2.3xlarge:

  | `--max-running-requests` | `--max-total-tokens` | Free after the pools | Result |
  | --- | --- | --- | --- |
  | 4 | 65536 | 2.9 GB | serves (the flags above; also at a 32k context) |
  | 8 | 65536 | 2.8 GB | serves |
  | 8 | 131072 | 1.8 GB | the prefill program fails to load (TDRV "Failed to allocate DEVICE memory") |
  | 16 | 65536 | 2.5 GB | the prefill program fails to load |
  | 16 | 32768 | 3.0 GB | serves |

  The KV costs about 16 KB per token per core, a request's GDN state about 40 MB per core.
- **Prefill tiles.** Below the 2048-token chunk the prefill compiles a graph of each KV segment tile it is given, and
  a prompt's last piece runs the smallest that holds it, so a short prompt does not pay for a whole padded chunk. Both
  prefill kernels take any multiple of 512 tokens. Without `--cuda-graph-bs-prefill` every tile below the chunk is
  compiled (512 and 1024); the flag lists the ones to compile. The tiles compete for device memory with the KV pool,
  the rows and the radix cache's state slots. With the radix cache on at the flags above, the state pool holds 85
  slots (1.65 GB a core) and 1.47 GB is free after the pools:

  | GDN state | Radix slots | Prefill tiles | Result |
  | --- | --- | --- | --- |
  | float32 | 85 | 512, 1024 (no flag) | the prefill program fails to load ("Failed to allocate resource") |
  | float32 | 85 | 512 (`--cuda-graph-bs-prefill 512`, the flags above) | serves |
  | float32 | 85 | none (`--cuda-graph-bs-prefill 2048`) | serves: the chunk alone, as before the tiles |
  | bfloat16 | 168 (the pool's bytes unchanged) | 512, 1024 | the prefill program fails to load |
  | bfloat16 | 85 (`--max-mamba-cache-size 85`) | 512, 1024 | serves |

  Without the radix cache the state pool is small: 16 rows with a 32768-token pool serve the full ladder in float32.
- **GDN state dtype.** `--mamba-ssm-dtype bfloat16` stores the GDN state in bf16 (the recurrence still runs in fp32 on
  chip; the kernels' state DMAs cast on the way in and out), half the bytes a slot. SGLang sizes the radix cache's state
  pool by bytes, so on its own the flag doubles the slots rather than freeing memory; with `--max-mamba-cache-size 85`
  it keeps the float32 server's checkpoints in half the memory, which lets the full ladder load beside them:

  ```bash
  ... --mamba-ssm-dtype bfloat16 --max-mamba-cache-size 85    # in place of --cuda-graph-bs-prefill 512
  ```

  The state dtype is `float32` (the default) or `bfloat16`; another value is refused at launch.
- **Page and chunk.** The page is a power of two of at least 128 tokens (512 validated); the chunk is whole pages
  (2048 validated).
- **Prefix cache.** The radix cache keeps the GDN state at the end of each prefill chunk (the chunk-keyed tree,
  [A chunk from a cached prefix](../design/recurrent-state.md#a-chunk-from-a-cached-prefix)): the three flags in the
  command above, the track interval equal to the chunk. `--disable-radix-cache` serves without it. A hit covers the
  prompt's whole chunks in the tree, so prompts share a prefix only in 2048-token steps. With the publish kernel the cache
  costs a decode step nothing measurable; a chat whose turns repeat a long history saves about 0.64 s of prefill per
  cached chunk.
- The Trainium2 core takes the GDN prefill's widest matmul at 512 moving columns; the model's launch rules set the
  kernel's `QWEN35_GDNP_WIDE=512` (`models/qwen38/launch.py`).

## Compiled shapes

- **Decode.** A graph per batch bucket up to `--max-running-requests` (the powers of two and the value itself), each at
  a quarter, a half and the whole of the context; a step runs the smallest that holds it. With the flags above that is
  9 decode graphs: the first launch on an empty compile cache took 22 minutes on a trn2.3xlarge, a relaunch with the
  same flags and cache about 6 minutes.
- **Prefill.** One graph of the chunk and one of each tile `--cuda-graph-bs-prefill` lists below it (512 with the
  flags above, see Prefill tiles); without the flag, one of each KV segment tile below the chunk (512 and 1024).
- `--max-total-tokens` sizes the KV pool, whose page count is a dimension of every graph; state it, so a relaunch finds
  its graphs in the compile cache.

## Measured results

One trn2.3xlarge, TP8, bf16, the flags above with `--disable-radix-cache` unless a result says the cache was on,
greedy, an idle host. The results after this first part were measured before the prefill tiles, with the chunk alone.

**Prefill tiles and the GDN state dtype.** One server at a time on the same host. On the recipe flags (radix cache
on, 4 rows, 65536 tokens), a streaming client with random-token prompts and 64 new tokens, 2 runs each:

| Prefill tiles, GDN state | TTFT at 64 to 512 / 1024 / 3072 tokens | TPOT | GSM8K | Radix hits | Batches | Needle |
| --- | --- | --- | --- | --- | --- | --- |
| the chunk alone, float32 | 0.65 / 0.65 / 1.27 s | 28.2 ms | 0.990 | 4 of 4 | 16 of 16 | 6 of 6 |
| 512, float32 (the flags above) | 0.20 / 0.65 / 1.27 s | 28.2 ms | 0.990 | 4 of 4 | not run | not run |
| 512 and 1024, bfloat16, 85 slots | 0.19 / 0.36 / 0.98 s | 27.7 ms | 0.990 | 4 of 4 | 16 of 16 | 6 of 6 |

Radix hits: the four cases of the prefix cache table below, each the cold run's exact tokens. Needle: a code at 10, 50
and 90 percent of filler at 2183 and 12158 tokens of prompt. Teacher-forced against Hugging Face on CPU (the five
prompts below, 24 steps each): 120 of 120 with margin 0 with both tiles and the float32 state (16 rows, 32768 tokens,
radix off), and 120 of 120 with margin 0 with both tiles and the bfloat16 state on the recipe flags (85 slots).

At 16 rows, a 32768-token pool and the radix cache off (`--context-length 16384 --max-running-requests 16
--max-total-tokens 32768 --disable-radix-cache`), `sglang.bench_serving` with `random-ids` and `--random-range-ratio
1`: 4 prompts at 1 client, 64 at 16, 96 at 32. Output tokens/s at 16 clients unless stated:

| Input / output tokens | The chunk alone | 512 and 1024, float32 | 512 and 1024, bfloat16 |
| --- | --- | --- | --- |
| 128 / 128, TTFT p50 at 1 client | 646 ms | 199 ms | 192 ms |
| TPOT p50 at 1 client | 31.7 ms | 31.7 ms | 30.5 ms |
| 128 / 128 | 125.1 | 222.4 | 243.0 |
| 128 / 512 | 229.3 | 286.7 | 318.7 |
| 1024 / 512 | 197.4 | 197.7 | 213.8 |
| 2048 / 512 | 128.2 | 143.4 | 155.8 |
| 1024 / 512 at 32 clients (16 queue) | 208.2 | 209.0 | 227.1 |
| mixed, 410 to 4096 in / 51 to 512 out, 128 prompts | 104.3 | 111.8 | 119.6 |
| mixed, TPOT p50 / p99 | 89.2 / 139.6 ms | 82.6 / 126.8 ms | 76.7 / 120.6 ms |

`bench_serving`'s 1024-token `random-ids` prompts reach the server as 1046 to 1070 tokens (its prefill log), above
the 1024 tile, so they run the 2048 chunk on every server; the streaming client's 1024-token prompt above runs the
1024 tile.

A soak of 20 minutes at 20 clients on each (prompts of 100 to 8000 tokens, 32 to 256 new tokens, 15 percent of the
streams closed midway): 562, 625 and 661 requests completed, no error on any, median TPOT 117 to 121, 104 to 103 and
103 to 103 ms from the first to the last 10 minutes, and each server answered correctly afterwards.

With the bfloat16 state, 32 rows at `--context-length 4096 --max-total-tokens 32768` (both tiles) load, at 32
clients: 355.6, 438.9, 239.8 and 164.4 output tokens/s at 128 / 128, 128 / 512, 1024 / 512 and 2048 / 512, TPOT p50
69.5, 54.4, 49.3 and 51.0 ms. A 30-minute soak at 40 clients (prompts up to 3000 tokens): 1861 completed, 342
cancelled, no error; the median TPOT was 108.8 ms in the first 10 minutes and 113.7 ms in the last 10 (not
investigated), and the server answered correctly afterwards.

Agent-like sessions on the recipe flags (radix on): a 6000-token start, then 10 turns that each append 100 to 500
tokens and take 64 greedy tokens, the history growing. TTFT of the later turns, median / p90: one session at a time
0.66 / 1.28 s with the chunk alone and 0.65 / 0.83 s with both tiles and the bfloat16 state (85 slots); 8 sessions 4 at
a time 1.89 / 3.14 s and 1.26 / 2.50 s (first turns 3.45 and 2.82 s; 129 and 112 s for all 8).

**Accuracy.** Teacher-forced against a Hugging Face reference on CPU in bf16, 24 steps on each of five prompts (5, 867,
2165, 7779 and 14582 tokens; the last crosses eight prefill chunks): 120 of 120 steps with the same argmax. GSM8K's
first 100 questions, five-shot at temperature 0: 0.990, none unparsed. A code hidden at 10, 50 and 90 percent of filler
text is answered exactly at 2183, 6383 and 12158 tokens of prompt and, on a 32k server, at 25283.

**Batches.** Eight prompts of 50 to 5010 tokens, each with its own answer length, give exactly the tokens they give
alone when run together and when started 0.7 s apart (16 of 16), with the batch growing and shrinking under the queue.

**Prefix cache.** On a server with the radix cache, a hit gives exactly the cold run's greedy tokens:

| Case | Prompt tokens | Cached tokens | Cold / hit, TTFT + 63 decode steps |
| --- | --- | --- | --- |
| the same prompt | 6013 | 4096 | 4.79 s / 3.49 s |
| the same prompt | 12013 | 10240 | 6.79 s / 3.58 s |
| the same document, another question | 12010 | 10240 | 6.80 s / 3.58 s |
| a second chat turn | 8151 | 6144 | 5.46 s / 3.54 s |

GSM8K with the cache on: 0.990.

**Decode time per token**, one client streaming 64 tokens after a random-token prompt: 28.2 ms at 64 and 2048 tokens of
prompt, 30.3 ms at 8192 and 16000, the same with the radix cache on and off; TTFT 0.65 s up to 2048 tokens, 2.52 s at
8192, 5.02 s at 16000. Concurrency at 128 new tokens: 30.3, 42.0 and 67.8 output tokens/s at 1, 2 and 4 clients. With
this build GSM8K scores 0.990 and the teacher-forced check above is 120 of 120.

The tables below were measured before the publish kernel (TorchStateOps rows, about 13 ms more per decode step); the
correctness results above hold for both builds.

**One client**, random-token prompts (`sglang.bench_serving` with `random-ids`, and a streaming client):

| Input / output tokens | TTFT p50 | TPOT p50 | TPOT p99 |
| --- | --- | --- | --- |
| 1024 / 512 | 0.66 s | 40.6 ms | 41.6 ms |
| 8000 / 1024 | 3.24 s | 44.0 ms | 44.0 ms |
| 14000 / 1024 | 5.27 s | 46.2 ms | 46.5 ms |
| 30000 / 1024 (32k server) | 11.17 s | 50.2 ms | 50.8 ms |

Decode time per token from the streaming client in that build: 41.5 ms at 64 tokens of prompt, 43.7 at 8192, 43.6 at
16000; on the 32k server, 49.0 at 32000. The GDN layers' cost does not depend on the context and the attention layers read only the
pages a request holds. Prefill runs at about 3.1k tokens/s (0.64 s per 2048-token chunk).

**Several clients.** Prefills run one request's chunk at a time, so TTFT includes the queue and a decode step waits for
a chunk that runs between steps.

| Input / output tokens | Clients | Output tokens/s | TTFT p50 | TPOT p50 | TPOT p99 |
| --- | --- | --- | --- | --- | --- |
| 1024 / 512 | 4 | 69.3 | 2.26 s | 53.4 ms | 55.3 ms |
| 1024 / 512 | 8 (8 rows) | 121.6 | 2.90 s | 60.2 ms | 64.7 ms |
| 4096 / 512 | 8 (8 rows) | 90.8 | 8.64 s | 71.4 ms | 84.9 ms |
| 8000 / 512 | 2 | 29.9 | 5.20 s | 56.8 ms | 59.4 ms |
| 8000 / 512 | 4 | 46.2 | 8.75 s | 69.6 ms | 79.2 ms |
| 14000 / 512 | 2 | 26.9 | 8.11 s | 58.7 ms | 63.2 ms |
| 14000 / 512 | 4 | 40.6 | 13.54 s | 72.3 ms | 87.3 ms |

**One chip's capacity** at a 16k context, `bench_serving` 1024 in / 512 out: 69.3 output tokens/s at 4 clients
(4 rows), 121.6 at 8 (8 rows), 136.1 at 16 (16 rows, a 32768-token pool), with TPOT p50 53.4, 60.2 and 78.7 ms.

**Steadiness.** On the radix server, 90 minutes at 6 clients against 4 rows (prompts of 100 to 8000 tokens, 32 to 256
new tokens, 15 percent of the streams closed by the client midway): 1415 requests completed, 259 cancelled, no error;
the median TPOT was 89.6 ms in the first 10 minutes and 89.8 ms in the last 10 (6 clients share 4 rows, so requests
queue), and the server answered correctly afterwards with one KV page in use.

## What is refused

A launch the graphs can't serve fails at startup, naming the setting at fault:

- a `--tp-size` other than 8 (the kernels run one rank per physical core of a trn2.3xlarge);
- a page size that is not a power of two of at least 128, and a chunk that is not whole pages;
- the radix cache on any tree but the chunk-keyed one, and a `--mamba-track-interval` other than the chunk;
- a KV cache dtype other than `auto`, `bf16` or `bfloat16`;
- the features every model refuses ([What a launch refuses](../design/runtime.md#what-a-launch-refuses)).
