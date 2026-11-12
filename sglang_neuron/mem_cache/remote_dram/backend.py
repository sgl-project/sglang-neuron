# SPDX-License-Identifier: Apache-2.0
"""The remote DRAM tier as a SGLang HiCache storage backend (``--hicache-storage-backend dynamic``).

One backend per scheduler rank. At start it takes a lease on ``servers[rank % len(servers)]`` (`lease`), sized
``gib_per_server`` over the ranks sharing that server, and splits it into slots of one host page each. SGLang calls:

- ``batch_exists``: how many leading keys this rank holds, ready. SGLang takes the MIN over the ranks, so every rank
  serves the same prefix from its private index.
- ``batch_set_v1``: a key already held is touched and reported stored; the others get slots (in runs where
  possible), are copied from the host pages into staging, digested (xxh3-128), written, and become readable only once
  the write completed.
- ``batch_get_v1``: the leading held keys are pinned and read into staging; a page is accepted only if its slot still
  holds that key and generation and its bytes match the digest, then copied into the host pages. The result is the
  longest good prefix.

Every failure is a miss: a write that fails frees its slot, a page that fails a check is dropped, and a read that
misses its deadline (200 ms + 0.1 ms per page by default) makes the rank report no hits for ``suspect_s``.

Launch (with the platform's HiCache settings):

    --enable-hierarchical-cache --hicache-storage-backend dynamic --hicache-storage-backend-extra-config
    '{"backend_name": "remote_dram", "module_path": "sglang_neuron.mem_cache.remote_dram.backend",
      "class_name": "RemoteDramStorage", "servers": ["10.0.0.12:5590"], "gib_per_server": 288,
      "engine_id": "p0", "auth": "sts", "tls_ca": "/path/ca.pem"}'
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, List, Optional, Sequence

import numpy as np
import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheStorage, HiCacheStorageConfig, HiCacheStorageExtraInfo

from sglang_neuron.mem_cache.remote_dram import lease as lease_mod
from sglang_neuron.mem_cache.remote_dram.index import SlotIndex
from sglang_neuron.mem_cache.remote_dram.transport import Arena, HostStore, Layout, NixlStore, TransferTimeout, runs

logger = logging.getLogger(__name__)

BACKEND_NAME = "remote_dram"
MODULE_PATH = "sglang_neuron.mem_cache.remote_dram.backend"
CLASS_NAME = "RemoteDramStorage"
#: The counters are logged at most this often, when they changed.
LOG_EVERY_S = 30.0
DEFAULTS = {
    "servers": [],
    "gib_per_server": 0.0,
    "ranks_per_server": 0,  # 0: tp x pp ranks spread over the servers
    "engine_id": "e0",
    "auth": "none",
    "tls_ca": "",
    "sts_region": lease_mod.DEFAULT_STS_REGION,
    "store": "nixl",  # "host": the in-process store (CPU tests)
    "nixl_backend": "LIBFABRIC",
    "staging_pages": 512,
    "layout": "page_major",  # or "region_major" (transport.Layout)
    "copy_threads": 8,  # host page <-> staging copies and digests (numpy and xxhash drop the GIL)
    "io_timeout_s": 5.0,
    "read_deadline_ms": 200.0,
    "read_deadline_ms_per_page": 0.1,
    "suspect_s": 30.0,
    "max_reconnects": 3,
    "lease_timeout_s": 10.0,
    "host_slots": 0,  # store "host": slots of the in-process lease
}


def _digest(parts: Sequence[np.ndarray]) -> bytes:
    import xxhash

    h = xxhash.xxh3_128()
    for part in parts:
        h.update(part)
    return h.digest()


class RemoteDramStorage(HiCacheStorage):
    def __init__(self, storage_config: HiCacheStorageConfig, kwargs: Optional[dict] = None):
        raw = dict(storage_config.extra_config or {})
        unknown = set(raw) - set(DEFAULTS) - {"backend_name", "module_path", "class_name", "interface_v1"}
        if unknown:
            raise ValueError(f"remote DRAM: unknown settings {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **raw}
        if self.cfg["auth"] not in lease_mod.AUTH_MODES:
            raise ValueError(f"remote DRAM: auth {self.cfg['auth']!r}")
        if self.cfg["store"] not in ("nixl", "host"):
            raise ValueError(f"remote DRAM: store {self.cfg['store']!r}")
        if self.cfg["layout"] not in ("page_major", "region_major"):
            raise ValueError(f"remote DRAM: layout {self.cfg['layout']!r}")
        if self.cfg["store"] == "nixl" and (not self.cfg["servers"] or self.cfg["gib_per_server"] <= 0):
            raise ValueError("remote DRAM: give servers and gib_per_server")
        self.config = storage_config
        c = storage_config
        self.rank_index = (c.dp_rank * c.pp_size + c.pp_rank) * c.tp_size + c.tp_rank
        self.lease_key = f"{self.cfg['engine_id']}-tp{c.tp_rank}-pp{c.pp_rank}-dp{c.dp_rank}"
        self.index: Optional[SlotIndex] = None
        self.arena: Optional[Arena] = None
        self.store = None
        self._suspect_until = 0.0
        self.stats = {"set_pages": 0, "set_present": 0, "set_failed": 0, "get_pages": 0, "get_failed": 0,
                      "bad_digest": 0, "stale": 0, "deadline_misses": 0}
        self._stats_lock = threading.Lock()
        self._logged_at, self._logged = time.monotonic(), dict(self.stats)
        # per call (pages, GB/s) for SGLang's storage metrics (--enable-metrics), drained by get_stats
        self._metrics = {"prefetch_pgs": [], "backup_pgs": [], "prefetch_bandwidth": [], "backup_bandwidth": []} \
            if storage_config.enable_storage_metrics else None

    # -- setup --
    def register_mem_pool_host(self, mem_pool_host) -> None:
        super().register_mem_pool_host(mem_pool_host)
        regions = getattr(mem_pool_host, "host_regions", None)
        if not regions or getattr(mem_pool_host, "mode", None) != "batch":
            raise ValueError("remote DRAM: needs the Neuron layer-first host pool (mem_cache/host_pool.py)")
        self.page_size = mem_pool_host.page_size
        self.host_pages = [r.view(r.shape[0], -1).view(torch.uint8).numpy() for r in regions]
        layout = Layout(tuple(int(t.shape[1]) for t in self.host_pages), self.cfg["layout"] == "page_major")
        slots = self._num_slots(layout)
        self.arena = Arena(layout, int(self.cfg["staging_pages"]))
        if self.cfg["store"] == "host":
            self.store = HostStore(layout, slots, self.arena)
        else:
            server = self.cfg["servers"][self.rank_index % len(self.cfg["servers"])]
            nbytes = layout.nbytes(slots)

            def take_lease():
                return lease_mod.request_lease(server, self.lease_key, nbytes, self.cfg["lease_timeout_s"],
                                               self.cfg["auth"], self.cfg["tls_ca"], self.cfg["sts_region"])

            self.store = NixlStore(layout, slots, self.arena, take_lease, self.lease_key, self.cfg["nixl_backend"],
                                   self.cfg["io_timeout_s"], max_reconnects=int(self.cfg["max_reconnects"]))
        # libfabric reads FI_EFA_IFACE once per process (the Neuron runtime loads it first), and the platform sets
        # each rank's to its device's one interface: on trn3, 8 TP ranks then share 2 of 16 EFA interfaces and reach
        # a much lower bandwidth together than all 16 give (M3)
        iface = os.environ.get("FI_EFA_IFACE", "")
        if self.cfg["store"] == "nixl" and iface and "," not in iface:
            logger.warning("remote DRAM %s: FI_EFA_IFACE=%s gives this rank's transfers one EFA interface; launch "
                           "with FI_EFA_IFACE set to a comma list of the node's interfaces to use them all",
                           self.lease_key, iface)
        self.index = SlotIndex(slots)
        threads = max(1, int(self.cfg["copy_threads"]))
        self._pool = ThreadPoolExecutor(threads, thread_name_prefix="remote-dram-copy")
        self._threads = threads
        logger.info("remote DRAM %s: %d slots of %d B (%.1f GiB), store %s", self.lease_key, slots, layout.slot_bytes,
                    slots * layout.slot_bytes / (1 << 30), self.cfg["store"])

    def _num_slots(self, layout: Layout) -> int:
        """The slots of this rank's lease: ``gib_per_server`` split over the ranks that share the server."""
        if self.cfg["store"] == "host":
            return int(self.cfg["host_slots"]) or 1024
        sharing = int(self.cfg["ranks_per_server"]) or math.ceil(
            self.config.tp_size * self.config.pp_size / len(self.cfg["servers"]))
        budget = int(self.cfg["gib_per_server"] * (1 << 30)) // sharing
        slots = budget // layout.slot_bytes
        while slots > 0 and layout.nbytes(slots) > budget:  # each region is 4 KiB aligned
            slots -= 1
        if slots <= 0:
            raise ValueError(f"remote DRAM: {self.cfg['gib_per_server']} GiB per server holds no slot of "
                             f"{layout.slot_bytes} B for {sharing} ranks")
        return slots

    def _pages(self, host_indices: torch.Tensor, count: int) -> List[int]:
        idx = host_indices.reshape(-1)[::self.page_size][:count]
        return (idx // self.page_size).tolist()

    def _split(self, fn: Callable[[Sequence[int]], Any], count: int) -> List[Any]:
        """``fn`` over ``range(count)`` split into strided parts, one per copy thread; the parts' results in order."""
        parts = [range(i, count, self._threads) for i in range(min(self._threads, count))]
        return list(self._pool.map(fn, parts))

    def _copy(self, dst: Sequence[np.ndarray], dst_idx: Sequence[int], src: Sequence[np.ndarray],
              src_idx: Sequence[int]) -> None:
        """``dst[r][dst_idx[k]] = src[r][src_idx[k]]`` for every region r, the regions split over the copy threads.

        A few runs consecutive on both sides are slice copies; otherwise one gather per region (np.take into a
        contiguous destination range, else a fancy-index copy). Indices are checked first: np.take runs with
        mode="clip", which would not raise.
        """
        n = len(src_idx)
        si, di = np.asarray(src_idx, dtype=np.int64), np.asarray(dst_idx, dtype=np.int64)
        if n == 0:
            return
        if si.min() < 0 or si.max() >= src[0].shape[0] or di.min() < 0 or di.max() >= dst[0].shape[0]:
            raise IndexError("remote DRAM: a page index is out of range")
        pairs = runs(si.tolist(), di.tolist())
        lo = int(di[0])
        dst_range = bool((di == np.arange(lo, lo + n)).all())

        def part(regions):
            for r in regions:
                d, s = dst[r], src[r]
                if len(pairs) <= 8:
                    for a, b, k in pairs:
                        d[b:b + k] = s[a:a + k]
                elif dst_range:
                    np.take(s, si, axis=0, out=d[lo:lo + n], mode="clip")
                else:
                    d[di] = s[si]

        self._split(part, len(dst))

    def _digests(self, rows: Sequence[int]) -> List[bytes]:
        parts = self._split(lambda ks: [_digest(self.arena.page_parts(rows[k])) for k in ks], len(rows))
        out: List[bytes] = [b""] * len(rows)
        for i, got in enumerate(parts):
            out[i::self._threads] = got
        return out

    @property
    def suspect(self) -> bool:
        return time.monotonic() < self._suspect_until

    def _count(self, **inc) -> None:
        with self._stats_lock:
            for k, v in inc.items():
                self.stats[k] += v
            now = time.monotonic()
            if now - self._logged_at < LOG_EVERY_S or self.stats == self._logged:
                return
            self._logged_at, self._logged = now, dict(self.stats)
        logger.info("remote DRAM %s: %s", self.lease_key, self.counters())

    def _record(self, op: str, pages: int, seconds: float) -> None:
        if self._metrics is None or pages <= 0:
            return
        with self._stats_lock:
            self._metrics[f"{op}_pgs"].append(pages)
            self._metrics[f"{op}_bandwidth"].append(pages * self.store.layout.slot_bytes / 1e9 / max(seconds, 1e-9))

    # -- SGLang's interface --
    def batch_exists(self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None) -> int:
        if self.index is None or self.suspect:
            return 0
        if getattr(self.store, "down", False):
            return 0
        return self.index.ready_prefix(keys)

    def batch_set_v1(self, keys: List[str], host_indices: torch.Tensor,
                     extra_info: Optional[HiCacheStorageExtraInfo] = None) -> List[bool]:
        result = [False] * len(keys)
        if not keys or self.index is None:
            return result
        t0 = time.perf_counter()
        pages = self._pages(host_indices, len(keys))
        plan, present = self.index.begin_write(keys)
        todo = [i for i, p in enumerate(plan) if p is not None]
        for i, held in enumerate(present):
            result[i] = held
        self._count(set_present=sum(present))
        stored = failed = 0
        # one staging chunk at a time: a batch bigger than the arena still goes through
        for base in range(0, len(todo), self.arena.rows):
            part = todo[base:base + self.arena.rows]
            rows = self.arena.take(len(part))
            if rows is None:
                logger.warning("remote DRAM %s: no staging rows for %d pages; skipping them",
                               self.lease_key, len(part))
                for i in part:
                    self.index.finish_write(plan[i][0], plan[i][1], None)
                failed += len(part)
                continue
            self._copy(self.arena.regions, rows, self.host_pages, [pages[i] for i in part])
            digests = self._digests(rows)
            slots = [plan[i][0] for i in part]
            order = sorted(range(len(part)), key=lambda k: slots[k])
            transfer = runs([slots[k] for k in order], [rows[k] for k in order])
            try:
                self.store.write(transfer)
                ok = True
            except TransferTimeout:
                ok, rows = False, None  # the store holds the rows
            except Exception as exc:  # noqa: BLE001 - a failed write is a miss
                logger.warning("remote DRAM %s: write of %d pages failed: %s", self.lease_key, len(part), exc)
                ok = False
            if rows is not None:
                self.arena.give(rows)
            for k, i in enumerate(part):
                result[i] = self.index.finish_write(plan[i][0], plan[i][1], digests[k] if ok else None)
            stored += len(part) if ok else 0
            failed += 0 if ok else len(part)
        self._count(set_pages=stored, set_failed=failed)
        if stored:
            self._record("backup", stored, time.perf_counter() - t0)
        return result

    def batch_get_v1(self, keys: List[str], host_indices: torch.Tensor,
                     extra_info: Optional[HiCacheStorageExtraInfo] = None) -> List[bool]:
        result = [False] * len(keys)
        if not keys or self.index is None or self.suspect:
            return result
        t0 = time.perf_counter()
        snaps = self.index.begin_read(keys)
        if not snaps:
            return result
        try:
            good_total, pages = 0, None
            # one staging chunk at a time; hits stay a prefix: stop at the first chunk not fully good
            for base in range(0, len(snaps), self.arena.rows):
                part = snaps[base:base + self.arena.rows]
                rows = self.arena.take(len(part))
                if rows is None:
                    logger.warning("remote DRAM %s: no staging rows for %d pages; reporting no more hits",
                                   self.lease_key, len(part))
                    self._count(get_failed=len(snaps) - good_total)
                    break
                order = sorted(range(len(part)), key=lambda k: part[k].slot)
                transfer = runs([part[k].slot for k in order], [rows[k] for k in order])
                deadline = (self.cfg["read_deadline_ms"] + self.cfg["read_deadline_ms_per_page"] * len(part)) / 1e3
                try:
                    self.store.read(transfer, deadline)
                except TransferTimeout as exc:
                    self._suspect_until = time.monotonic() + self.cfg["suspect_s"]
                    logger.warning("remote DRAM %s: %s; reporting no hits for %.0f s", self.lease_key, exc,
                                   self.cfg["suspect_s"])
                    self._count(deadline_misses=1, get_failed=len(snaps) - good_total)
                    return result  # the store holds the rows until the read ends
                except Exception as exc:  # noqa: BLE001 - a failed read (StoreDown included) is a miss
                    logger.warning("remote DRAM %s: read of %d pages failed: %s", self.lease_key, len(part), exc)
                    self.arena.give(rows)
                    self._count(get_failed=len(snaps) - good_total)
                    break
                digests = self._digests(rows)
                good = 0
                for k, snap in enumerate(part):
                    if not self.index.still_holds(snap):
                        self._count(stale=1)
                        break
                    if digests[k] != snap.digest:
                        self._count(bad_digest=1)
                        self.index.drop(snap)
                        break
                    good += 1
                if good:
                    if pages is None:
                        pages = self._pages(host_indices, len(keys))
                    self._copy(self.host_pages, pages[good_total:good_total + good], self.arena.regions,
                               rows[:good])
                    for i in range(good_total, good_total + good):
                        result[i] = True
                self.arena.give(rows)
                good_total += good
                self._count(get_pages=good)
                if good < len(part):
                    self._count(get_failed=len(snaps) - good_total)
                    break
            self._record("prefetch", good_total, time.perf_counter() - t0)
            return result
        finally:
            self.index.unpin(snaps)

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
        if self.index is not None:
            self.index.clear()

    def close(self) -> None:
        if self.store is not None:
            self.store.close()
            self.store = None
        if self.arena is not None:
            self.arena.close()
            self.arena = None
        if getattr(self, "_pool", None) is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def counters(self) -> dict:
        """This rank's counters and its index's slot states."""
        with self._stats_lock:
            out = dict(self.stats)
        if self.index is not None:
            out.update(self.index.stats())
        return out

    def get_stats(self):
        """SGLang's storage metrics since the last call (a StorageMetrics), or None without --enable-metrics."""
        if self._metrics is None:
            return None
        from sglang.srt.observability.metrics_collector import StorageMetrics

        with self._stats_lock:
            drained = {k: list(v) for k, v in self._metrics.items()}
            for v in self._metrics.values():
                v.clear()
        return StorageMetrics(**drained)
