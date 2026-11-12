# Serving Qwen3.5-397B with the radix prefix cache (paged KV, chunked prefill)

The design is in [A chunk from a cached prefix](../design/recurrent-state.md#a-chunk-from-a-cached-prefix) and
[Compiling the kernel graphs](../design/megakernel.md#compiling-the-kernel-graphs). Status on one trn3 node:

- warm-up compiles every NEFF with the serving defaults: decode 20 MB, prefill segments 1.5–3 MB;
- the server answers correctly with async execution on;
- a turn-2 radix hit (8192 cached tokens) gives the same greedy output as a cold run.

## Build the inputs

The server needs two inputs that the public checkpoint does not have: a text-only view of the FP8 checkpoint and a
per-rank weight pack. Build both on a CPU host, from the repository root, in the venv of the
[setup guide](../getting-started/setup-guide.md).

1. Download the public checkpoint `Qwen/Qwen3.5-397B-A17B-FP8` to `<Qwen3.5-397B-A17B-FP8>` (about 406 GB).
2. Make the text-only view. The tool writes `config.json` (the `text_config` as `Qwen3_5MoeForCausalLM`) and
   links every other file. It does not change the source directory:

   ```bash
   python tools/qwen35/make_text_view.py <Qwen3.5-397B-A17B-FP8> <Qwen3.5-397B-A17B-FP8-text>
   ```

3. Make the weight pack (about 410 GB in 1475 files). The pack holds the decode and prefill layouts. The dense
   weights are the FP8 blocks dequantized to bf16, and the routed experts are MXFP8. A run that stops can start
   again: it skips the finished jobs.

   ```bash
   python -m tools.qwen35.build_pack.pack --ckpt <Qwen3.5-397B-A17B-FP8> --out <qwen35-pack> --workers 96 --verify 4
   ```

4. Set `SGLANG_NEURON_QWEN35_PACK=<qwen35-pack>` and `--model-path <Qwen3.5-397B-A17B-FP8-text>` in the launches
   below.

## Launch (aggregated, tp128, dp8, one trn3 node): agentic and long-context default

The commands are in the container image's form ([container guide](../guides/container.md); mount the checkpoint at
`/models` and the weight pack beside it). In a development environment ([setup
guide](../getting-started/setup-guide.md#build-a-development-environment)), also export `SGLANG_PLATFORM=neuron` and
`PYTHONPATH=<sglang-checkout>/python`.


```bash
export NEURON_RT_MAP_HBM=1 TORCH_NEURONX_ENABLE_STREAMS=1
export TORCH_NEURONX_NEFF_CACHE_DIR=<cache>/neff TORCH_NEURONX_HLO_CACHE_DIR=<cache>/hlo NKI_COMPILE_CACHE_URL=<cache>/nki
export SGLANG_NEURON_QWEN35_PACK=<qwen35-pack>                # the 397B pack (gdn16, attn8 split per rank)
export QWEN35_ATTN_PAGE_Q=auto QWEN35_ATTN_SLOT_DGE=hw        # paged decode DMAs on hardware DGE
export TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT=10800             # the decode family's compiles outlast the 1200 s default
python3 -m sglang.launch_server --model-path /models/<Qwen3.5-397B-A17B-FP8-text> \
  --device cpu --attention-backend neuron --sampling-backend pytorch \
  --tp-size 128 --dp-size 8 --enable-dp-attention --ep-size 128 --enable-nvidia-megakernel \
  --page-size 2048 --context-length 16384 --chunked-prefill-size 16384 \
  --max-running-requests 512 --cuda-graph-bs-decode 16 32 64 --max-total-tokens 65536 \
  --radix-cache-backend neuron-chunk --mamba-radix-cache-strategy extra_buffer --mamba-track-interval 2048 \
  --max-mamba-cache-size 2880 --mem-fraction-static 0.8 --disable-custom-all-reduce --trust-remote-code \
  --host 0.0.0.0 --port 30000
```

`--host 0.0.0.0` listens inside the container; publish the port on the host's loopback only
(`-p 127.0.0.1:30000:30000`), as the [container guide](../guides/container.md#3-start-a-container) explains.

- **Model path.** Use the text-only view of the checkpoint (`config.json` = the `text_config` as
  `Qwen3_5MoeForCausalLM`, every other file symlinked). The catalog does not serve the multimodal architecture.
- **Paging.** A launch is paged when `--context-length` exceeds `--page-size`. The catalog serves pages of 2048 (or
  256 / 512 / 1024) tokens at LNC 1. 2048 is the default: the KV page is then the tree's chunk, and with hardware-DGE
  page DMAs the paged decode attention costs what the slot pool's does, less than at 512 with spread queues (64 rows /
  16K). At 512, keep `QWEN35_ATTN_PAGE_Q=spread`: there hardware DGE and the compiler's queue are both slower. An
  aggregated DP-attention server is served because the megakernel plans each
  step for the whole DP cohort.
- **Chunk.** `--chunked-prefill-size 16384` is C = 2048 per replica (SGLang splits it over the 8 DP groups). It is the
  radix tree's page and the GDN checkpoint grid (`--mamba-track-interval 2048`). It must be whole KV pages and whole
  512-token blocks.
- **`--max-mamba-cache-size`** is the total over the DP replicas. SGLang reserves 5 state slots per running request
  with the extra buffer, so 2880 = 8 × (64 × 5 + 40 checkpoints). One slot is 12 MB per rank, so 360 slots per
  replica use 4.0 GiB. Raising it costs KV tokens (`--max-total-tokens`, per replica, 15 KiB/token/rank). A rank has
  about 10.6 GB after the weights. Each cached 2048-token chunk holds one checkpoint slot.
- **Decode batch.** `--cuda-graph-bs-decode 16 32 64` gives decode graphs of 16, 32 and 64 rows per replica (the
  kernel needs a multiple of 16), each at every context rung. The decode family compiles in nki's legacy mode
  (`runtime.DECODE_NKI_COMPILE_MODE`, NKI's instruction scheduling off), as the prefill family does. Under NKI's
  scheduling the 16-row graph exhausts the Pool engine's registers (NCC_IGCA094). The legacy mode costs the 32-row
  graph no step time (at 25 rows per replica).
- **Capacity at 2048-token pages.** With `--max-total-tokens 65536` per replica (32 pages), a request holds whole
  2048-token pages, so a replica runs at most 32 / ceil((prompt + output) / 2048) requests and the 64-row graph is not
  reached. For many short requests, use the high-concurrency recipe below.
- **Async.** The serving policy logs `async_execution: true`, with SGLang's overlap schedule.
- **Sampling.** Greedy only, no logprobs. Set `temperature: 0` (or `top_k: 1`) in each request. A request without a
  sampling field takes the checkpoint's `generation_config.json` values (temperature 0.6, top_k 20), and the server
  refuses it with HTTP 400. The chat template turns thinking on by default. For a chat eval, turn thinking off
  (`chat_template_kwargs {"enable_thinking": false}`), or give each answer 16K or more tokens and remove the stop
  strings that thinking text can contain (for example `Question:` and `Q:`).

## Launch for high-concurrency short prompts (pages of 512)

The same launch with `--page-size 512`, `QWEN35_ATTN_PAGE_Q=spread` (no `QWEN35_ATTN_SLOT_DGE`), and everything else
unchanged. A request then holds 512-token pages:

| Request (prompt + output) | Requests per replica, PG 2048 | Requests per replica, PG 512 |
|---|---:|---:|
| 512 + 512 | 32 | 64 (the 64-row graph) |
| 2048 + 512 | 16 | 25 |
| 8192 + 1024 | 6 | 7 |

The trade: at PG 512 the paged decode attention costs more per layer (at 64 rows / 16K), and at low concurrency TPOT
is higher (2k/512 and 8k/1k, both with the 32-row graph). At PG 2048, 2k/512 at c256 or c512 only queues.

## Routing (radix hits need replica affinity)

Each attention-DP replica keeps its own radix tree. A session's turns must reach the same replica to hit: use a
router with a cache-aware policy, or pin with `routed_dp_rank` in `/generate`. The in-server round-robin spreads a
session's turns over the replicas, so they never hit.

## Measured on one trn3 node

- **Numerics.**
  - GSM8K (250 questions, 5-shot, greedy, 64 in flight): 98.4% host-ids, 97.6% relay, 97.2% relay + direct submit.
  - Needles at about 12.8K tokens: 8/8 found, and each radix hit gives the same output.
- **Hit vs cold identity holds**, which is what caching needs. Radix check, 9.4K tokens: equal. AgentX session 0:
  6/6 turns equal.
- **Greedy outputs depend on what runs in the same step (a known property).**
  - The same server, one request at a time, twice: 40/40 identical.
  - One at a time vs 64 in flight: 23/40 identical.
  - A fresh server vs the previous one, both one request at a time: 26/40 identical.
  - Every step's MoE spans all 128 ranks, and the DP cohort's step mixes the replicas' rows (padding included), so
    the MXFP8 and accumulation order follow the step's composition, and which replica serves a request.
- **TPOT is device-bound** (8k/1k, 64-row decode over the full 16K span): it is the same at c1 and c8, the host
  trace puts nearly all of it in the decode program with the relay, and the host overhead with relay + direct submit
  is small. The decode attention over 16K pages is a large share of it, which the context ladder and the 32-row
  bucket target.
- **AgentX** (16 sessions × 6 turns, +512 per turn, c8, pinned): every later turn hits the whole 2048-token chunks of
  the previous turn. TTFT is lower hot than cold.
- **HiCache (L2 host) on the chunk-keyed tree.** Add `--enable-hierarchical-cache --hicache-ratio 2
  --hicache-write-policy write_through`. A HiCache page is the tree's 2048-token chunk, so the host KV pool holds each
  chunk's four 512-token device pages (1.90 GiB per rank); the GDN chunk-state checkpoints go to the state host pool.
  The prefill waits for the whole load (the KV and the state engines) before it reads the restored pages.
  Host memory: at `--hicache-ratio 2` with `--max-mamba-cache-size 2880`, each rank pins about 10 GiB (1.90 GiB of KV
  and 8.06 GiB of GDN state), about 1.3 TiB for the 128 ranks of a node. SGLang lets the host pools use all of the
  node's free memory but 10 GiB. On a trn3 node with 2 TB, this leaves about 145 GB free. Make sure that no other
  large host process runs on the node, or lower `--hicache-ratio` or `--max-mamba-cache-size`.
  Measured with sessions pinned to one replica (DP rank 2, 3 sessions of about 9.3K tokens, 9 fill prompts that evict them from
  HBM):

  | Path | Cached tokens | Greedy identical to cold |
  |---|---:|---:|
  | Cold | 0 | — |
  | HBM hit | 8192 | 3/3 |
  | Host hit | 8192 | 3/3 |

  With `SGLANG_NEURON_HICACHE_VERIFY=1`, every load is checked against the device copy: 0 of 240 pages differ.

- **Decode context ladder and the 32-row bucket.** The cohort picks one context rung per step (4K / 8K / 16K pages,
  from the snapshot's longest decoding sequence). `--cuda-graph-bs-decode 32 64` adds a 32-row bucket. Measured at
  PG 512, before the 16-row graph (below), at 2k/512, 6k/1k and 8k/1k: TPOT is equal at c1 and c8, and at 8k/1k it is
  lower than with the 64-row decode over the full 16K span.

- **Replica position matters.** The same replica, one request at a time: 20/20 identical. Replica 0 vs replica 5:
  14/20. A request's replica changes its rows' place in the step, which explains the cross-server difference.

- **KV pages of 2048 tokens with hardware DGE** (the default above, in place of pages of 512 with spread queues). Same
  ladder, 32/64-row buckets, fresh compile: TPOT is lower than at PG 512 at 2k/512, 6k/1k and 8k/1k, at c1 and c8.

  - GSM8K n=250: 97.2%.
  - Determinism: the same replica 20/20, replica 0 vs replica 5 14/20 (as at 512).
  - The radix check passes.
  - HiCache: greedy output is identical on both hit paths (3/3), and
    `SGLANG_NEURON_HICACHE_VERIFY=1` finds 0 differing pages (a host load is now 4 pages of 2048 tokens).

- **The 16-row graph** (legacy nki compile mode, PG 2048, hardware DGE; one request per replica, c1 and c8 equal):
  TPOT is lower than with the 32-row graph at 2k/512, 6k/1k and 8k/1k, and at 512/512 at c64 / c128 (8 / 16 rows
  per replica). GSM8K n=250 98.0%.
  Determinism: the same replica 20/20, replica 0 vs replica 5 13/20. The radix check passes.
  The grid-edge probe passes: prompts one short of the grid decoding across it.

- **32K context** (`--context-length 65536 --max-total-tokens 69632`, same flags otherwise, HiCache on). The pool
  must hold the context plus two pages: the KV allocator keeps one page back, so with `--max-total-tokens 65536` a
  prompt near 64K tokens gets no tokens or is never scheduled, and the launch refuses that pool. The decode compiles 3 buckets × 3
  rungs (16K / 32K / 64K; the 32K and 64K rungs run single-pass split-K in the 60-layer decode). Prefill compiles
  3 prior tiers (0 / 16K / 64K).
  - Needles at about 27.3K tokens: 8/8 found, and every hit repeats the cold answer (26624 cached).
  - TPOT 32k/1k is equal at c1 and at c8.
  - AgentX: 16 sessions on 8 replicas, 7 turns, first prompt 8K, +4K per turn to about 33.5K. Later turns hit most
    of the reusable prefix, and their TTFT p50 is lower than a cold run's. Session 0 replayed cold matches its hit
    outputs on all 7 turns. 0 errors.
  - HiCache at about 31K prompts (2 sessions, then 2 fills that evict them from HBM but fit the 2× host pool):
    30720 tokens cached on both paths, greedy output identical 2/2.
    With 9 fills the host pool (133K tokens per replica) evicts them too, and the turn is cold.

## Known limits

- **Context past 64K** is not compiled. Contexts are powers of two, and 64K is the largest served and measured.
- **Prefill overlap.** A paged prefill forward runs its chunk as one micro-batch; the 1024-token overlap is a
  follow-up.

## Checks

Smoke, then a 9.4K-token session pinned to one replica:

| Request | Cached tokens | Output |
|---|---:|---|
| turn 1, cold | 0 | — |
| turn 1, repeat | 8192 | same as cold |
| turn 2 | 8192 | — |
| turn 2, cold after `/flush_cache` | 0 | same as the hit |

The greedy outputs are equal and correct.
