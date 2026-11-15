# SPDX-License-Identifier: Apache-2.0
"""Mooncake Store as the HiCache L3 tier of the Neuron host pool (``--hicache-storage-backend dynamic``).

SGLang's own Mooncake backend (``sglang.srt.mem_cache.storage.mooncake_store``) reads the host pool through
``kv_buffer`` and ``get_page_buffer_meta``, and serves an MHA model only from a page-first host buffer. The Neuron host
pool (mem_cache/host_pool.py) is layer-first: ``host_regions`` holds one pinned host tensor per device region (for the
MHA pool, K of every layer then V; for MLA / DSA, one per layer of each paged buffer), each ``[host pages, page
bytes]``. This backend keeps that layout:

- one host page of this rank is one Mooncake object, ``<key prefix>_<page key>_<rank>``;
- ``layout`` ``page_major`` (the default): a page is gathered from its row in every region into a registered staging
  buffer, contiguous, and put or got as one buffer; the copies run on ``copy_threads`` threads. One contiguous object
  per page moves an order of magnitude faster than the page's rows as separate buffers (on trn2 over EFA, 64 pages of
  4.5 MiB: put 31 against 0.95 GB/s, get 75 against 22 GB/s);
- ``layout`` ``region_multi_buffer``: every region is registered once and a page is put or got straight from the host
  pages as one multi-buffer transfer over its rows (``batch_put_from_multi_buffers`` /
  ``batch_get_into_multi_buffers``), no copy;
- like the remote DRAM tier (mem_cache/remote_dram/backend.py), ``batch_exists`` and ``batch_get_v1`` report the
  longest held prefix, ``batch_set_v1`` skips keys the store already holds, and a failed transfer is a miss.

The rank suffix carries TP, PP and DP ranks: the objects are this rank's shard of a page, so ranks of one server, and
servers of the same layout and model, share pages through the store.

Launch (with the platform's HiCache settings):

    --enable-hierarchical-cache --hicache-storage-backend dynamic --hicache-storage-backend-extra-config
    '{"backend_name": "mooncake_neuron", "module_path": "sglang_neuron.mem_cache.mooncake_store",
      "class_name": "NeuronMooncakeStore", "master_server_address": "10.0.0.5:50051",
      "metadata_server": "P2PHANDSHAKE", "protocol": "efa", "global_segment_size_gib": 32}'
"""

from __future__ import annotations

import logging
import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import numpy as np
import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheStorage, HiCacheStorageConfig, HiCacheStorageExtraInfo
from sglang_neuron.mem_cache.storage_tiers import MOONCAKE

logger = logging.getLogger(__name__)

BACKEND_NAME, MODULE_PATH, CLASS_NAME = MOONCAKE["backend_name"], MOONCAKE["module_path"], MOONCAKE["class_name"]

DEFAULTS = {
    # the Mooncake master (mooncake_master --rpc_port), host:port
    "master_server_address": None,
    # the transfer engine's metadata: P2PHANDSHAKE, or an etcd / http metadata server
    "metadata_server": "P2PHANDSHAKE",
    # Mooncake's transport for host memory: efa on EFA hosts, else tcp
    "protocol": "tcp",
    # EFA / RDMA devices to use, "" for Mooncake's own discovery
    "device_name": "",
    # host memory this rank contributes to the store's pool, split from the server's share by the operator
    "global_segment_size_gib": 4.0,
    # the client's own staging buffer; zero-copy puts and gets do not use it
    "local_buffer_size_mib": 16,
    # this host's address as the other clients reach it; default: the address the master is reached from
    "local_hostname": None,
    # isolates models and deployments sharing one store; default: "<model name>"
    "key_prefix": None,
    # objects per Mooncake batch call
    "batch_pages": 256,
    # page_major: one contiguous object per page through a staging buffer; region_multi_buffer: no copy, one buffer
    # per region row
    "layout": "page_major",
    # pages the staging buffer holds (page_major); a larger batch goes through it in parts
    "staging_pages": 64,
    # threads for the copies between the host pages and the staging buffer (page_major)
    "copy_threads": 8,
}


class NeuronMooncakeStore(HiCacheStorage):
    def __init__(self, storage_config: HiCacheStorageConfig, kwargs: Optional[dict] = None):
        raw = dict(storage_config.extra_config or {})
        unknown = set(raw) - set(DEFAULTS) - {"backend_name", "module_path", "class_name", "interface_v1"}
        if unknown:
            raise ValueError(f"Mooncake L3: unknown settings {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **raw}
        if not self.cfg["master_server_address"]:
            raise ValueError("Mooncake L3: give master_server_address (the mooncake_master's host:port)")
        if self.cfg["layout"] not in ("page_major", "region_multi_buffer"):
            raise ValueError(f"Mooncake L3: layout {self.cfg['layout']!r} (page_major or region_multi_buffer)")
        self.config = storage_config
        c = storage_config
        model = "-".join((c.model_name or "model").split("/"))
        self.prefix = self.cfg["key_prefix"] or model
        self.rank_suffix = f"tp{c.tp_rank}of{c.tp_size}_pp{c.pp_rank}of{c.pp_size}_dp{c.dp_rank}"
        self.store = None
        self.regions: List[torch.Tensor] = []
        self.row_bytes: List[int] = []
        self.page_size = 1
        self.page_major = self.cfg["layout"] == "page_major"
        self.host_pages: List[np.ndarray] = []  # each region as [host pages, row bytes] uint8
        self.offsets: List[int] = []  # each region's byte offset inside a staged page
        self.staging: Optional[np.ndarray] = None  # [staging pages, page bytes] uint8, registered
        self._pool: Optional[ThreadPoolExecutor] = None
        self.stats = {"set_pages": 0, "set_present": 0, "set_failed": 0, "get_pages": 0, "get_failed": 0}
        self._metrics = {"prefetch_pgs": [], "backup_pgs": [], "prefetch_bandwidth": [], "backup_bandwidth": []} \
            if storage_config.enable_storage_metrics else None
        self._connect()

    # -- setup --
    def _connect(self) -> None:
        try:
            from mooncake.store import MooncakeDistributedStore
        except ImportError as exc:
            raise ImportError("Mooncake L3: install mooncake-transfer-engine-non-cuda (0.3.14 or later)") from exc
        master = self.cfg["master_server_address"]
        hostname = self.cfg["local_hostname"] or _local_address_towards(master)
        segment = int(float(self.cfg["global_segment_size_gib"]) * (1 << 30))
        local = int(self.cfg["local_buffer_size_mib"]) * (1 << 20)
        store = MooncakeDistributedStore()
        rc = store.setup(hostname, self.cfg["metadata_server"], segment, local, self.cfg["protocol"],
                         self.cfg["device_name"], master)
        if rc != 0:
            raise RuntimeError(f"Mooncake L3: setup against {master} failed ({rc})")
        self.store = store
        logger.info("[neuron hicache] Mooncake L3 rank %s: %s over %s, %.1f GiB contributed, key prefix %s",
                    self.rank_suffix, master, self.cfg["protocol"], segment / (1 << 30), self.prefix)

    def register_mem_pool_host(self, mem_pool_host) -> None:
        super().register_mem_pool_host(mem_pool_host)
        regions = getattr(mem_pool_host, "host_regions", None)
        if not regions or getattr(mem_pool_host, "mode", None) != "batch":
            raise ValueError("Mooncake L3: needs the Neuron layer-first host pool (mem_cache/host_pool.py)")
        self.page_size = mem_pool_host.page_size
        self.regions = list(regions)
        self.row_bytes = [int(r[0].numel() * r.element_size()) for r in self.regions]
        if self.page_major:
            self.host_pages = [r.view(r.shape[0], -1).view(torch.uint8).numpy() for r in self.regions]
            self.offsets = np.cumsum([0] + self.row_bytes[:-1]).tolist()
            pages = max(1, int(self.cfg["staging_pages"]))
            self.staging = np.zeros((pages, sum(self.row_bytes)), dtype=np.uint8)
            self._register(self.staging.ctypes.data, self.staging.nbytes, "staging buffer")
            self._pool = ThreadPoolExecutor(max(1, int(self.cfg["copy_threads"])), thread_name_prefix="mooncake-l3")
        else:
            for r in self.regions:
                self._register(int(r.data_ptr()), int(r.numel() * r.element_size()), "host region")
        logger.info("[neuron hicache] Mooncake L3 rank %s: %s, %d host regions, %d B per page",
                    self.rank_suffix, self.cfg["layout"], len(self.regions), sum(self.row_bytes))

    def _register(self, ptr: int, nbytes: int, what: str) -> None:
        rc = self.store.register_buffer(ptr, nbytes)
        if rc != 0:
            raise RuntimeError(f"Mooncake L3: registering a {nbytes} B {what} failed ({rc})")

    # -- helpers --
    def _key(self, key: str) -> str:
        return f"{self.prefix}_{key}_{self.rank_suffix}"

    def _pages(self, host_indices: torch.Tensor, count: int) -> List[int]:
        idx = host_indices.reshape(-1)[::self.page_size][:count]
        return (idx // self.page_size).tolist()

    def _buffers(self, pages: List[int]):
        ptrs = [[int(r.data_ptr()) + p * rb for r, rb in zip(self.regions, self.row_bytes)] for p in pages]
        sizes = [list(self.row_bytes) for _ in pages]
        return ptrs, sizes

    def _batches(self, n: int):
        step = max(1, int(self.cfg["staging_pages"] if self.page_major else self.cfg["batch_pages"]))
        return [range(i, min(n, i + step)) for i in range(0, n, step)]

    def _stage(self, pages: List[int], to_staging: bool) -> None:
        """Copy ``pages`` of the host pool into staging rows 0..n-1 (or back), the regions split over the threads."""
        n = len(pages)
        idx = np.asarray(pages, dtype=np.int64)

        def part(rs):
            for r in rs:
                lo, hi = self.offsets[r], self.offsets[r] + self.row_bytes[r]
                if to_staging:
                    self.staging[:n, lo:hi] = self.host_pages[r][idx]
                else:
                    self.host_pages[r][idx] = self.staging[:n, lo:hi]

        threads = self._pool._max_workers
        list(self._pool.map(part, [range(i, len(self.regions), threads) for i in range(threads)]))

    def _put(self, names: List[str], pages: List[int]) -> List[int]:
        if not self.page_major:
            ptrs, sizes = self._buffers(pages)
            return self.store.batch_put_from_multi_buffers(names, ptrs, sizes)
        self._stage(pages, to_staging=True)
        width = self.staging.shape[1]
        base = self.staging.ctypes.data
        return self.store.batch_put_from(names, [base + k * width for k in range(len(pages))], [width] * len(pages))

    def _get(self, names: List[str], pages: List[int]) -> List[int]:
        """Bytes read per name, or a negative code; page_major copies the leading good pages back."""
        if not self.page_major:
            ptrs, sizes = self._buffers(pages)
            return self.store.batch_get_into_multi_buffers(names, ptrs, sizes)
        width = self.staging.shape[1]
        base = self.staging.ctypes.data
        got = self.store.batch_get_into(names, [base + k * width for k in range(len(pages))], [width] * len(pages))
        good = 0
        for n in got:
            if n <= 0:
                break
            good += 1
        if good:
            self._stage(pages[:good], to_staging=False)
        return got

    def _count(self, **inc) -> None:
        for k, v in inc.items():
            self.stats[k] += v

    def _record(self, op: str, pages: int, seconds: float) -> None:
        if self._metrics is None or pages == 0 or seconds <= 0:
            return
        gb = pages * sum(self.row_bytes) / 1e9
        self._metrics[f"{op}_pgs"].append(pages)
        self._metrics[f"{op}_bandwidth"].append(gb / seconds)

    # -- SGLang's interface (interface_v1) --
    def batch_exists(self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None) -> int:
        if not keys or self.store is None:
            return 0
        present = self.store.batch_is_exist([self._key(k) for k in keys])
        held = 0
        for p in present:
            if p != 1:
                break
            held += 1
        return held

    def batch_set_v1(self, keys: List[str], host_indices: torch.Tensor,
                     extra_info: Optional[HiCacheStorageExtraInfo] = None) -> List[bool]:
        result = [False] * len(keys)
        if not keys or self.store is None or not self.regions:
            return result
        t0 = time.perf_counter()
        names = [self._key(k) for k in keys]
        present = self.store.batch_is_exist(names)
        todo = [i for i, p in enumerate(present) if p != 1]
        for i, p in enumerate(present):
            result[i] = p == 1
        self._count(set_present=len(keys) - len(todo))
        if todo:
            pages = self._pages(host_indices, len(keys))
            stored = failed = 0
            for part in self._batches(len(todo)):
                idx = [todo[j] for j in part]
                rcs = self._put([names[i] for i in idx], [pages[i] for i in idx])
                for i, rc in zip(idx, rcs):
                    result[i] = rc == 0
                    stored += rc == 0
                    failed += rc != 0
            self._count(set_pages=stored, set_failed=failed)
            if failed:
                logger.warning("Mooncake L3 rank %s: %d of %d page puts failed", self.rank_suffix, failed, len(todo))
            self._record("backup", stored, time.perf_counter() - t0)
        return result

    def batch_get_v1(self, keys: List[str], host_indices: torch.Tensor,
                     extra_info: Optional[HiCacheStorageExtraInfo] = None) -> List[bool]:
        """The leading pages read whole into the host pages; reading stops at the first page that fails."""
        result = [False] * len(keys)
        if not keys or self.store is None or not self.regions:
            return result
        t0 = time.perf_counter()
        pages = self._pages(host_indices, len(keys))
        good = 0
        for part in self._batches(len(keys)):
            idx = list(part)
            got = self._get([self._key(keys[i]) for i in idx], [pages[i] for i in idx])
            # bytes read on success, a negative code on error (Mooncake's pybind_client.h)
            for i, n in zip(idx, got):
                if n <= 0:
                    break
                result[i] = True
                good += 1
            if good < idx[-1] + 1:
                break
        self._count(get_pages=good, get_failed=len(keys) - good)
        self._record("prefetch", good, time.perf_counter() - t0)
        return result

    def exists(self, key: str) -> bool:
        return self.batch_exists([key]) == 1

    # the generic (copying) interface: SGLang uses batch_*_v1 with interface_v1
    def get(self, key, target_location=None, target_sizes=None):
        return None

    def batch_get(self, keys, target_locations=None, target_sizes=None):
        return [None] * len(keys)

    def set(self, key, value=None, target_location=None, target_sizes=None) -> bool:
        return False

    def batch_set(self, keys, values=None, target_locations=None, target_sizes=None) -> bool:
        return False

    def clear(self) -> None:
        """Nothing: the store is shared, so one rank's clear would drop pages other servers still serve."""

    def close(self) -> None:
        if self.store is not None:
            self.store.close()
            self.store = None
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def counters(self) -> dict:
        return dict(self.stats)

    def get_stats(self):
        if self._metrics is None:
            return None
        from sglang.srt.observability.metrics_collector import StorageMetrics

        drained = {k: list(v) for k, v in self._metrics.items()}
        for v in self._metrics.values():
            v.clear()
        return StorageMetrics(**drained)


def _local_address_towards(master: str) -> str:
    """This host's address on the route to ``master`` (no packet is sent)."""
    host = master.rsplit(":", 1)[0]
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect((host, 9))
            return s.getsockname()[0]
        except OSError:
            return os.environ.get("MOONCAKE_LOCAL_HOSTNAME", socket.gethostname())
