# HiCache remote DRAM tier

A KV tier below HiCache's L2 host pool: prefix KV pages that leave HBM and L2 are kept in the host DRAM of another
trn3 node and read back over NIXL LIBFABRIC (EFA) on a later hit, instead of being prefilled again. It is an SGLang
HiCache storage backend (`--hicache-storage-backend dynamic`) plus a lease server on the lending node.

## How it works

- **Lease server** (`python -m sglang_neuron.mem_cache.remote_dram.server`, on the lending node): fills a region of
  host DRAM in transparent huge pages, registers it with one NIXL agent, and answers lease requests on a TCP port. A
  lease is a byte range of the region and the agent metadata to reach it; the same key gets the same lease again.
- **Backend** (`sglang_neuron/mem_cache/remote_dram/backend.py`, one per scheduler rank): takes a lease at start,
  splits it into slots of one host page each, and keeps a private index from SGLang's page hash to a slot. A write
  copies the pages into a staging arena, digests them (xxh3-128) and writes them with NIXL; a key is readable only
  once its write completed. A read is accepted only if the slot still holds that key and generation and the bytes
  match the digest; any failure is a miss, never a wrong page. SGLang takes the minimum hit over the ranks.
- **Layout:** a slot is page-major (all 2 x layers K/V pages of one host page in one row), so a run of pages is one
  NIXL descriptor. The host pool stays layer-first; the copy into staging changes the layout.

## Launch

On the lending node (one address, the serving node's address and IAM identity allowed, TLS):

```bash
mkdir -m 0700 /path/tls
python -m sglang_neuron.mem_cache.remote_dram.server --gib 1024 --host <lending node ip> --port 5590 \
  --allow-ips <serving node ip> --auth sts --allow-arns <serving node's caller ARN> --tls-dir /path/tls
```

Copy `/path/tls/ca.pem` to the serving node. On the serving node, with HiCache on:

```bash
FI_EFA_IFACE=$(ls /sys/class/infiniband | paste -sd, -) python -m sglang.launch_server ... \
  --enable-hierarchical-cache --hicache-write-policy write_through \
  --hicache-storage-backend dynamic --hicache-storage-prefetch-policy wait_complete \
  --hicache-storage-backend-extra-config '{"backend_name": "remote_dram",
    "module_path": "sglang_neuron.mem_cache.remote_dram.backend", "class_name": "RemoteDramStorage",
    "servers": ["<lending node ip>:5590"], "gib_per_server": 128, "engine_id": "p0",
    "auth": "sts", "tls_ca": "/path/ca.pem"}'
```

- `gib_per_server` is split over the ranks that share a server; `servers` may list several lending nodes.
- `FI_EFA_IFACE`: the platform sets each rank's to its Neuron device's one EFA interface. libfabric reads it once
  per process, so the tier's transfers then use that interface only (on trn3 the 8 TP ranks share 2 of the 16
  interfaces). A comma list at launch gives every rank all of them; the backend warns when a rank has one.
- Requirements on both nodes: the NIXL overlay that `tools/build_nixl.sh` builds, first on `PYTHONPATH`; on the
  serving node `botocore` (the lease token), and `ninja` on `PATH` with the OpenSSL headers (SGLang JIT-builds its
  storage page hash).
- Other backend settings (`DEFAULTS` in `backend.py`): `layout` (`page_major` or `region_major`), `copy_threads`,
  `staging_pages`, `read_deadline_ms`, `read_deadline_ms_per_page`, `suspect_s`, `io_timeout_s`, `max_reconnects`.

## Security

- The lease server binds one address (a wildcard address is refused) and closes a connection from an address
  outside `--allow-ips` before reading a byte.
- Off loopback the server requires `--auth sts`, `--tls-dir` and LIBFABRIC, and the client requires TLS and a token:
  - **TLS 1.3** with a CA made for each run. The client trusts only that CA and sends no token before the handshake
    passes.
  - **Token:** a presigned `sts:GetCallerIdentity` request that signs the lease key, a nonce and the server's
    `ip:port`. The server forwards it to STS, accepts each token once, and checks the caller ARN against
    `--allow-arns`.
- Without auth the server serves loopback only (CPU mode and single-node use).
- No per-lease isolation on the data plane: a leaseholder can address the whole registered chunk. The allow lists
  are the trust boundary.
- The server exits at `--max-lifetime-s` (default 4 h). When the host's MemAvailable falls under
  `--shrink-floor-gib` it frees chunks from the end of the region and revokes their leases; under
  `--stop-floor-gib` it stops.

## Limits

- Each rank's index is private and starts empty when the rank starts: a restarted server cannot read what it
  wrote before. There is no sharing across engines.
- The lease server keeps the NIXL connection state of every client agent it has seen.
- Aggregated serving only; the disaggregated prefill role is not wired.
- A token context-parallel KV pool's host tier takes no remote DRAM tier: each rank's host page is its shard of a page
  ([The host tier](../design/context-parallelism.md#the-host-tier)).
