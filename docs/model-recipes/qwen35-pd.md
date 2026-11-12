# Serving Qwen3.5-397B in P/D (1 prefill node + 1 decode node), radix cache on the paged pool

Status: **run on hardware**, 1P1D on two trn3 nodes, KV and GDN state over NIXL / EFA. Results
below.

- The aggregated server this builds on is in [Qwen3.5 radix](qwen35-radix.md).

## What each role runs

| Role | Graph families (`megakernel.ROLE_FAMILIES`) | Programs (`phase_programs`) |
|---|---|---|
| prefill | `prefill` | prefill, idle (an idle replica pads the cohort's prefill), no direct submit |
| decode | `decode` | decode, idle, direct submit |

What moves from prefill to decode over NIXL (SGLang's PD transfer, `--disaggregation-transfer-backend nixl`):

- **KV:** the 15 full-attention layers' pages. `Qwen35SlotKVPool.get_contiguous_buf_infos` reports one entry per
  layer for K and one for V, and a page per item. K^T `[d, PG]` and V `[PG, d]` are each one contiguous 1 MiB page
  at PG 2048, so whole pages move by page id.
- **GDN state:** the request's recurrent and conv state of the 45 GDN layers, each slot whole through upstream's
  equal-TP path. Both nodes must run the same attention TP (16 per replica): the temporal state is stored in the
  kernels' `[k, h, v]` layout, which upstream cannot head-slice, so a pairing across attention TPs is refused at
  registration (`disaggregation/state_transfer.state_entries_problem`).

## Launch

Both nodes run the container image ([container guide](../guides/container.md): the node's network, the Neuron and EFA
devices, the checkpoint at `/models` and the weight pack mounted), with the same environment. In a development
environment ([setup guide](../getting-started/setup-guide.md#build-a-development-environment)), also export `SGLANG_PLATFORM=neuron` and
`PYTHONPATH=<sglang-checkout>/python`, with the NIXL overlay `tools/build_nixl.sh` builds first (`NIXL_PLUGIN_DIR`
unset).


```bash
export NEURON_RT_MAP_HBM=1 TORCH_NEURONX_ENABLE_STREAMS=1
export TORCH_NEURONX_NEFF_CACHE_DIR=<cache>/neff TORCH_NEURONX_HLO_CACHE_DIR=<cache>/hlo NKI_COMPILE_CACHE_URL=<cache>/nki
export SGLANG_NEURON_QWEN35_PACK=<qwen35-pack>   # the packed per-rank weights (models/qwen35/weights.py)
export QWEN35_ATTN_PAGE_Q=auto QWEN35_ATTN_SLOT_DGE=hw
export TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT=10800
# EFA / NIXL at 128 ranks per node. The plugin selects the LIBFABRIC backend (table below).
export FI_EFA_USE_HUGE_PAGE=0 FI_EFA_USE_DEVICE_RDMA=1  # hugepages: 128 ranks x 16 rails drained them (SIGBUS)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1  # host oversubscription slows launches
COMMON="--model-path /models/<Qwen3.5-397B-A17B-FP8-text> --device cpu --attention-backend neuron --sampling-backend pytorch \
  --tp-size 128 --dp-size 8 --enable-dp-attention --ep-size 128 --enable-nvidia-megakernel \
  --page-size 2048 --context-length 65536 --chunked-prefill-size 16384 --max-total-tokens 69632 \
  --radix-cache-backend neuron-chunk --mamba-radix-cache-strategy extra_buffer --mamba-track-interval 2048 \
  --max-mamba-cache-size 2880 --mem-fraction-static 0.8 --disable-custom-all-reduce --trust-remote-code \
  --disaggregation-transfer-backend nixl --disaggregation-bootstrap-port 8998 --host <this node address> --port 30000 \
  --enable-request-time-stats-logging"   # per-request phase durations (bootstrap / forward / transfer) in the log
```

Prefill node:

```bash
python3 -m sglang.launch_server $COMMON --disaggregation-mode prefill --max-running-requests 64
```

Decode node:

```bash
python3 -m sglang.launch_server $COMMON --disaggregation-mode decode --disaggregation-decode-enable-radix-cache \
  --max-running-requests 512 --cuda-graph-bs-decode 16 32 64
```

Front end, on the prefill node: the plugin's balancer, which the validation ran and the image includes (the
[example](../../examples/qwen35/) has it as `run_balancer.sh`):

```bash
python3 -m sglang_neuron.disaggregation.balancer --prefill http://<prefill-ip>:30000 \
  --decode http://<decode-ip>:30000
```

It listens on 127.0.0.1:8000 and picks each request's prefill replica by load.

**Replica affinity needs another front end.** Both servers place a request on replica `bootstrap_room % dp`
(`follow_bootstrap_room`). The balancer picks the prefill replica by load, and SGLang's router (sglang_router 0.3.2,
not in the image) draws the room at random: it drops `routed_dp_rank`, rejects a client `bootstrap_room` (422), and
forwards `data_parallel_rank`, which the servers refuse as a room conflict; `--dp-aware` fails the same way
(502/503). Through either, a session's turns land on different replicas, so most of the reusable prefix is not
reused. A front end that pins a session to one replica on both servers (posting
each request to both with `bootstrap_room ≡ replica (mod 8)`) reuses most of it. This repository does not
ship that session-affinity proxy.

Launch notes:

- **Chunk.** The decode keys its radix tree on its own chunk (`--chunked-prefill-size` over its 8 DP groups), so
  both nodes take 16384, which is C = 2048 per replica. The decode's chunk must divide the prefill's
  (`sglang_patches.decode_radix_prefill_chunks`).
- **Radix on the decode** needs `--disaggregation-decode-enable-radix-cache`. Without it, `--radix-cache-backend
  neuron-chunk` is refused on the decode.
- **Replica affinity.** Each attention-DP replica keeps its own tree on both nodes. A session's turns must reach
  the same replica on both nodes to hit, which the proxy provides.
- **Restart both servers together.** A stale NIXL peer breaks transfers.
- **Sampling.** Greedy only, no logprobs, as on the [aggregated server](qwen35-radix.md). Set `temperature: 0` (or
  `top_k: 1`) in each request: a request without a sampling field takes the checkpoint's temperature 0.6 and is
  refused.
- **Pool size.** Each role's `--max-total-tokens` must hold the context plus two pages (the KV allocator keeps one
  page back): 266240 on the prefill and 299008 on the decode at 262144. The launch refuses a smaller pool.
- **Compile.** Each role compiles only its family. Prefill: three prior tiers × the 2048 bucket (its warm-up reads
  the aggregated server's cache). Decode: 3 buckets × 3 context rungs (16K / 32K / 64K).

## NIXL and PD environment

The PD environment both roles must agree on (the mandated table is `docs/design/disaggregated-inference.md`; the
launch refuses a conflicting setting before the server comes up):

| Setting | Value | Roles | Why |
|---|---|---|---|
| `SGLANG_DISAGGREGATION_NIXL_BACKEND` | `LIBFABRIC` (SGLang's default is `UCX`) | both | the KV moves over EFA, and a failed transfer reads settled only once LIBFABRIC reports it |
| `NIXL_LIBFABRIC_EFA_UNSOLICITED_WRITE_RECV` | `1` | both | patch `patches/nixl/0006-libfabric-efa-unsolicited-write-recv-knob.patch`'s switch; unset, a decode rank exhausts its posted receives and its QP dies at ~2.5M cumulative write-imms (~three 220K-token KV transfers), and the decode server stops |

The image's NIXL is built from this repo's `patches/nixl/` series, and a development environment's overlay must be too
(`tools/build_nixl.sh`): patch 0006 is what makes the knob above exist (unset, the plugin behaves as upstream:
unsolicited write receive off). A stock NIXL serves short prompts and then fails under sustained long-context load (the
~2.5M write-imm ceiling above), so in a development environment check that every node loads the overlay (`python -c
"import nixl; print(nixl.__file__)"` names `<nixl-overlay>`).

The validated 256K pair (needles 12/12 at ~194K tokens, GSM8K n=250 97.6%) ran the launch above with
`--context-length 262144 --watchdog-timeout 7200` on both roles; on the prefill `--max-total-tokens 266240
--max-mamba-cache-size 128 --max-running-requests 64`, `SGLANG_NEURON_QWEN35_SCATTER_MAX_DESCS=8192` and
`SGLANG_NEURON_QWEN35_KV_TM=1` (the kv_cp=16 decode's K pages are token-major, and the scatter refuses a dim-major
prefill K pool at registration); on the decode
`SGLANG_NEURON_QWEN35_KV_CP=16 --max-total-tokens 299008 --max-mamba-cache-size 2880 --max-running-requests 512`
(section [256K context](#256k-context-kv_cp16-decode-context-parallelism)); and `SGLANG_NEURON_CCOM_PORT_OFFSET=<0..899>` on both so a
relaunch never meets a port a dying predecessor still holds.

The kv_cp scatter (the prefill's send of each page's blocks to the kv_cp decode,
[its design](../design/context-parallelism.md#a-whole-page-prefill-into-a-cp-decode)) reads its three knobs once at
import, and the decode runtime reads the state dump. All four are registered in `sglang_neuron/env_registry.py`:

| Setting | Default | Why |
|---|---|---|
| `SGLANG_NEURON_QWEN35_SCATTER_MAX_DESCS` | 32768 | the EFA descriptors of one posted batch, in whole pages; with the drain a prefill rank keeps one batch outstanding across its peers (the 256K prefill launch sets 8192) |
| `SGLANG_NEURON_QWEN35_SCATTER_DRAIN` | on by default | settle the previous posted batch before the next is posted; `0` posts without settling |
| `SGLANG_NEURON_QWEN35_SCATTER_DUMP` | unset (off) | dump each scatter's descriptor layout |
| `SGLANG_NEURON_QWEN35_STATE_DUMP` | unset (off) | dump the received GDN state slots |

## Measured

**Correctness, through the proxy:**
- Radix check: a turn-2 hit of 8192 tokens gives the same output as cold.
- Track probe: prompts one short of the grid, decoding across 1-2 grid points. Every repeat hits the prompt's last
  whole chunk and matches cold.
- Needles 8/8 at about 12.8K tokens. GSM8K n=250: 97.6%.

**Transfer per request (KV pages + GDN state):** the decode's transfer duration minus the prefill's forward. The
prefill's logged `transfer_speed` divides by the whole forward.

**Latency through the router** (bench_serving, random prompts: 8k/1k at c8, 32k/1k at c1 and c8): TPOT equals the
aggregated server's, since the decode node runs only decode steps.

**Stability:** 8k/1k at c64, 192 requests. All completed. No traceback on either server, and no NRT collective
timeout.

**AgentX**: 16 sessions, 8 turns, first 8192 tokens, growth 2048, max-new 256, think 3 s, through the router and
through the proxy at concurrency 8 and 16. 0 errors in every run.

**Warm-up gap (open).** On the prefill role, four GDN state-slot programs first ran while serving, outside the
warm-up's SPMD check: `clear_slots` at the first request and `copy_slots` at the first radix hit. The pool warms both operations (2 tensors
× buckets 8 / 64 / 256) at construction. The decode role logs none.

## Known gaps

- HiCache on the PD decode is implemented and CPU-tested behind `SGLANG_NEURON_QWEN35_HICACHE=1`
  ([The host tier](../design/context-parallelism.md#the-host-tier)) but not checked on the nodes (the aggregated
  server's HiCache is). Set the switch on the decode with `--enable-hierarchical-cache`; without it the platform
  refuses `--enable-hierarchical-cache` on the decode. The kv_cp decode's host tier refuses an L3 storage backend
  (`--hicache-storage-backend`).
- The GDN state pairing is proven at equal attention TP only.
- The prefill role's state-slot warm-up gap (above).

## 256K context: kv_cp=16 decode context parallelism

At 256K the decode's KV no longer fits one rank's whole pages (2 MB/page/layer × 15 attention layers × 128 pages
per request). The decode role then context-parallels the KV over the 16-rank replica: every rank holds both KV
heads of its 128-token block of each 2048-token page (1,920 B/token/rank), the PD transfer scatters each page's
block to the rank that attends over it (128 KB/page/layer per peer instead of 2 MB; 0.47 GB/request at 256K), and
the block tables, allocator and radix cache stay identical on every rank. The prefill is unchanged (kv_cp=1).

Decode node, as above plus:

```bash
SGLANG_NEURON_QWEN35_KV_CP=16 python -m sglang.launch_server \
  ... --context-length 262144 --max-total-tokens 299008 ...
```

Prefill node: no `SGLANG_NEURON_QWEN35_KV_CP` (the launch refuses it on the prefill and aggregated roles), and
`SGLANG_NEURON_QWEN35_KV_TM=1`, as in the validated pair above. The launch refuses `SGLANG_NEURON_QWEN35_KV_TM=1` on
any role but a kv_cp=1 PD prefill, and on a prefill without the paged pool (`--context-length` past `--page-size`).

- **Pool.** At 1,920 B/token/rank, 512K tokens would be ≈ 0.94 GiB/rank of KV (vs 15 GiB whole pages), next to
  weights+graphs 7.35 GB, GDN state 4.35 GB (TP16-sharded, unchanged) and 67 MB of RoPE tables. HBM is tight at
  that size; the validated launch sizes the pool at `--max-total-tokens 299008`, which still covers one 256K
  request (128 pages of 2048 = 262,144 tokens). The pool must also stay within the kernel's flat K/V write, or the
  launch refuses it: 15 attention layers x the pages of a layer (page 0 included) x 256 x the rank's block of a page
  must not pass 2^31 elements. A kv_cp=16 rank's block is 128 tokens. A kv_cp=1 pool of 2048-token pages, as on the
  prefill, holds at most 273 pages of a layer with page 0, 557,056 tokens.
- **Attention.** At kv_cp=16 a rank's share of 256K is 16K tokens, so the decode runs the unrolled attention path;
  the split-K block loop is the kv_cp=1 fallback and the >256K-per-rank case.
- **Transfer.** Each prefill rank sends only its KV head shard's block of each page to the two decode ranks that
  read it (sglang_patches/kv_cp_scatter); the equal-TP geometry (prefill attn TP = the decode CP group) is checked
  at peer registration ([the scatter](../design/context-parallelism.md#a-whole-page-prefill-into-a-cp-decode)).
- **State owner gate.** The GDN state is attention-TP sharded (kv_cp does not touch it), so a decode rank's state
  shard is on one prefill rank, its own in-replica index. The scatter gives every decode rank two KV sources
  ({e % 8, e % 8 + 8} at kv_cp=16), and only the source at the decode rank's own index sends the state; the other
  sends KV and aux only ([the state owner](../design/context-parallelism.md#a-whole-page-prefill-into-a-cp-decode)).
- **Prefill state pool.** In-flight 256K prompt KV is 3.84 GB/rank whole pages; `--max-mamba-cache-size 128` at
  dp8 bounds the mamba state pool accordingly.
