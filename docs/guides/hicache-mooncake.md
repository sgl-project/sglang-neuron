# HiCache L3 on Mooncake Store

A KV tier below HiCache's L2 host pool, held in a [Mooncake Store](https://github.com/kvcache-ai/Mooncake) pool: prefix
KV pages written there by one server are read back by any server of the same model and layout attached to the same
pool, and by the same server after it restarts. It is an SGLang HiCache storage backend
(`--hicache-storage-backend dynamic`, `sglang_neuron/mem_cache/mooncake_store.py`) over the Mooncake Python client.

Against the [remote DRAM tier](hicache-remote-dram.md), whose index is private to each rank and starts empty when the
rank starts, the Mooncake pool is shared and outlives the serving engines.

## How it works

- **Pool:** a `mooncake_master` keeps the object index; one or more storage services (`mooncake_client`) contribute
  host DRAM to the pool. The serving ranks are pure clients (`global_segment_size_gib` 0), so the pool and its pages
  survive a server restart. A rank may also contribute memory of its own, but the pages on it go away with the rank.
- **Objects:** one object per host page of a rank, `<key prefix>_<page key>_tp<r>of<n>_pp<r>of<n>_dp<r>`. The key prefix
  defaults to the model path, so models sharing one pool do not collide; servers of the same model and the same
  TP / PP / DP layout share their pages.
- **Layout:** the Neuron host pool is layer-first (one region per K and V of every layer). With `layout` `page_major`
  (the default) a page is gathered into a registered staging buffer, contiguous, and put or got as one buffer; the
  copies run on `copy_threads` threads. One contiguous object per page moves an order of magnitude faster than the
  page's 2 x layers rows as separate buffers: on trn2 over EFA, 64 pages of 4.5 MiB, put 31 GB/s against 0.95 and get
  75 GB/s against 22 (`region_multi_buffer`).
- **Semantics** follow the remote DRAM tier: `batch_exists` and reads report the longest held prefix, writes skip
  pages the pool already holds, and a failed transfer is a miss.

## Launch

The Mooncake master and a storage service, on a host every serving node reaches (here over EFA; `tcp` works too):

```bash
mooncake_master --rpc_port=50051 --enable_http_metadata_server=false
mooncake_client --host=<storage host ip> --port=50052 --master_server_address=<master ip>:50051 \
  --metadata_server=P2PHANDSHAKE --protocol=efa --global_segment_size='64 GB'
```

Both ship in `mooncake-transfer-engine-non-cuda` (0.3.14 or later, the `mooncake` package directory). A container
running them needs the EFA devices (`/dev/infiniband/uverbs*`), `--cap-add IPC_LOCK` and `--ulimit memlock=-1`.

On every serving node, with HiCache on:

```bash
python -m sglang.launch_server ... \
  --enable-hierarchical-cache --hicache-write-policy write_through \
  --hicache-storage-backend dynamic --hicache-storage-prefetch-policy wait_complete \
  --hicache-storage-backend-extra-config '{"backend_name": "mooncake_neuron",
    "module_path": "sglang_neuron.mem_cache.mooncake_store", "class_name": "NeuronMooncakeStore",
    "master_server_address": "<master ip>:50051", "protocol": "efa", "global_segment_size_gib": 0}'
```

Settings (`DEFAULTS` in `mooncake_store.py`): `master_server_address` (required), `metadata_server` (`P2PHANDSHAKE`),
`protocol` (`tcp` or `efa`), `device_name`, `global_segment_size_gib` (host memory the rank contributes, default 4),
`local_buffer_size_mib`, `local_hostname` (default: the address towards the master), `key_prefix`, `layout`,
`staging_pages`, `copy_threads`, `batch_pages`. The platform sets `interface_v1`.

## Measured

Qwen3-8B TP4 on one trn2.48xlarge, two servers (cores 0-3 and 4-7) attached to one pool (64 GB storage service, EFA),
16 prompts of 3,005 tokens with distinct 3,000-token prefixes:

| Run | Cached tokens | TTFT p50 | Greedy output |
|---|---|---|---|
| server A, cold | 0 | 194 ms | reference |
| server B, prefixes A wrote (cross-engine) | 2,944 of 3,005 | 182 ms | identical to A's |
| server A restarted, same prompts | 2,944 | 179 ms | identical to A's |
| either server again (device hit) | 2,944 | 75 ms | identical |

An L3 hit reads 2,944 tokens of KV (about 100 MiB per rank on TP4) from the pool in a few milliseconds, far under
the 105 ms it adds over a device hit; the rest is the load from the host pool to the device and its scheduling, the same path an L2 hit takes.

InferenceX AgentX (agentx-harness 1.0.13, `semianalysis_cc_traces_weka_062126_256k`, concurrency 4, 900 s, every run
`submission_valid`) on one server of Qwen3-30B-A3B-Instruct-2507, TP8 EP8, `--context-length 262144`,
`--max-running-requests 4`, `SGLANG_NEURON_SAMPLER=topk_topp` (the model's sampling defaults), with the paged decode
not passing `max_context_len` (sglang-neuron#4: the nkilib flash-attention early exit is wrong past 8K tokens). The
three servers ran at the same time, each on its own two trn2 devices; HiCache L2 is `--hicache-ratio 2`:

| HiCache | Requests | Output tok/s | Total tok/s | Interactivity p50 | TTFT p50 / p90 | ITL p50 | Prompt cache read |
|---|---|---|---|---|---|---|---|
| off | 77 | 50.66 | 3,936 | 15.41 | 24.4 s / 47.1 s | 64.9 ms | 89.3% |
| L2 | 77 | 50.82 | 3,999 | 15.26 | 23.3 s / 47.9 s | 65.5 ms | 90.4% |
| L2 + Mooncake L3 | 77 | 50.84 | 4,001 | 15.30 | 23.1 s / 47.8 s | 65.4 ms | 90.4% |

The device pool (524K tokens) and the host pool hold this single server's working set: L2 serves 0.45 to 0.51 million
prompt tokens (prefill computes 1.13 instead of 1.43 million), and L3 is written (9,151 pages) but never read. On one
server the Mooncake tier costs nothing; it pays where the host pool cannot serve a prefix: another server of the same
model (the cross-engine row above), a restart, or a working set beyond the host pool.

## Limits

- The MHA host pool only (as the remote DRAM tier); the MLA / DSA host pools stay L2 only.
- Aggregated serving only; the disaggregated prefill role is not wired.
- `clear()` does nothing: the pool is shared, so one rank's clear would drop pages other servers still serve. Remove a
  model's pages with the Mooncake master's tools.
- Pages are shared only between servers of the same model and the same TP / PP / DP layout (the key carries the rank).
