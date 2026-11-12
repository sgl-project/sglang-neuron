# SPDX-License-Identifier: Apache-2.0
"""Remote DRAM tier of Neuron HiCache: prefix KV pages in host DRAM of other nodes, below the L2 host pool.

A scheduler rank stores and loads its host pages in a byte range (a lease) of a region that a lease server on
another node registered with NIXL, and moves them with NIXL READs and WRITEs. SGLang drives it as a HiCache storage
backend (``--hicache-storage-backend dynamic``, `backend.RemoteDramStorage`); the lease server (`server`) is a
standalone process and imports no SGLang.

Modules:

- `fill`: memory checks and the region fill (standard library only).
- `lease`: the lease channel's wire format, its STS token and TLS client (standard library, botocore for a token).
- `server`: the lease server CLI.
- `index`: a rank's key -> slot index, with generations, records and pins.
- `transport`: the staging arena and the stores (in-process for tests, NIXL).
- `backend`: the HiCache storage backend.

Every failure ends as a miss, and SGLang prefills the page again; a page that does not check out is never returned.
"""
