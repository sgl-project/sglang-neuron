# SPDX-License-Identifier: Apache-2.0
"""HiCache L2 for the Neuron KV pools: a host copy of the device pages, and the copies between them.

SGLang's HiCache (``--enable-hierarchical-cache``) keeps evicted radix-cache pages in host memory and copies them back
on a later hit. SGLang drives it (the radix tree, the host allocator, ``HybridCacheController``, ``L2TransferEngine``);
the platform supplies the host pool, which owns the copies. On this platform SGLang's device module is the host
(``torch.cpu``, process_init), so SGLang's own transfer events complete at once; the host pool runs the copies on the
Neuron device and reports their completion through the events this module puts in their place (``XferEvent``).

The host buffer is layer-first and page-granular, one region per device tensor whose first dimension is the page:
for the MHA pool ``[2 x layers][host pages][page bytes]``, K of every layer then V; for the MLA / DSA pools
(mla_host_pool.py) one region per layer of each paged buffer. A device page of one region is contiguous in the Neuron
block layout (the packed fp8 K too), so a run of pages consecutive on both sides is one contiguous span per region.

A transfer moves whole pages with the runtime's ``nrt_tensor_read_batch`` / ``nrt_tensor_write_batch``: one call per
transfer, one op per (region, run), on a transfer thread per direction. A store waits on the host for the graphs
launched before it (they may still be writing its pages), and so does a load on an aggregated server; the forward that
reads a load waits for it on the host (``LayerDoneCounter.set_consumer``). On an aggregated server the scheduler holds
that forward's batch back while the load runs and the running batch decodes (hicache_hold.py). The runtime DMAs into
the host buffer after the call returns, so the buffer is pinned and page-locked (``mlock``). In CPU mode
(``SGLANG_NEURON_CPU_MODE=1``, the unit tests) the pool's tensors are host tensors and the same jobs copy with torch.
"""

from __future__ import annotations

import collections
import ctypes
import functools
import logging
import os
import threading
import time
from typing import List

import numpy as np
import torch

from sglang.srt.mem_cache.pool_host.base import HostKVCache

logger = logging.getLogger(__name__)

_STATS = os.environ.get("SGLANG_NEURON_HICACHE_STATS", "0") == "1"
_VERIFY = os.environ.get("SGLANG_NEURON_HICACHE_VERIFY", "0") == "1"


# ---- runtime library --------------------------------------------------------------------------------------------


class _BatchOp(ctypes.Structure):
    _fields_ = [("offset", ctypes.c_uint64), ("size", ctypes.c_uint64), ("buffer", ctypes.c_void_p)]


class _Batch(ctypes.Structure):
    _fields_ = [("tensor", ctypes.c_void_p), ("ops", ctypes.POINTER(_BatchOp)), ("num_ops", ctypes.c_uint32)]


# _BatchOp's layout as a numpy record, so one array holds every tensor's ops and each _Batch points at its row
_BATCH_OP = np.dtype([("offset", "<u8"), ("size", "<u8"), ("buffer", "<u8")])
assert _BATCH_OP.itemsize == ctypes.sizeof(_BatchOp)
_BATCH_OP_PTR = ctypes.POINTER(_BatchOp)


def _column(values) -> np.ndarray:
    """Per-tensor constants as a [tensors, 1] uint64 column, to broadcast against a job's runs."""
    return np.asarray(values, dtype=np.uint64).reshape(-1, 1)


def batch_ops(runs: list, dev_off: np.ndarray, host_base: np.ndarray, page_bytes: np.ndarray) -> np.ndarray:
    """The [tensors, runs] _BatchOp records of one batch transfer: for tensor r and run (device page d, host page h,
    count n), device offset dev_off[r] + d * page_bytes[r], size n * page_bytes[r], host address
    host_base[r] + h * page_bytes[r]. The per-tensor arguments are _column()s."""
    a = np.asarray(runs, dtype=np.uint64).reshape(-1, 3)
    ops = np.empty((len(dev_off), len(a)), dtype=_BATCH_OP)
    ops["offset"] = dev_off + a[:, 0] * page_bytes
    ops["size"] = a[:, 2] * page_bytes
    ops["buffer"] = host_base + a[:, 1] * page_bytes
    return ops


@functools.lru_cache(maxsize=1)
def _libnrt():
    lib = ctypes.CDLL("libnrt.so.1")
    lib.nrt_tensor_read_batch.restype = ctypes.c_int
    lib.nrt_tensor_read_batch.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    lib.nrt_tensor_write_batch.restype = ctypes.c_int
    lib.nrt_tensor_write_batch.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    lib.nrt_tensor_read.restype = ctypes.c_int
    lib.nrt_tensor_read.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64]
    get = ctypes.pythonapi.PyCapsule_GetPointer
    get.restype, get.argtypes = ctypes.c_void_p, [ctypes.py_object, ctypes.c_char_p]
    return lib, get


def _nrt_handle(tensor) -> int:
    import torch_neuronx

    _, get = _libnrt()
    return int(get(torch_neuronx._C._get_nrt_tensor_capsule(tensor), b"nrt_tensor_t"))


def nrt_batch_copy(handles: list, op_cols: tuple, runs: list, to_host: bool) -> int:
    """One ``nrt_tensor_read_batch`` (``to_host``) or ``nrt_tensor_write_batch`` over ``runs`` (device page, host page,
    count) of every tensor ``handles`` names (op_cols: batch_ops' per-tensor columns); returns the number of ops."""
    run_nrt_batch(handles, batch_ops(runs, *op_cols), to_host)  # the ops array lives until the call returns
    return len(handles) * len(runs)


def torch_page_copy(device_tensors: list, host_tensors: list, runs: list, to_host: bool) -> int:
    """The same page runs as torch copies, for host tensors standing in for the device (CPU mode)."""
    for dev, host in zip(device_tensors, host_tensors):
        for d, h, n in runs:
            if to_host:
                host[h:h + n].copy_(dev[d:d + n])
            else:
                dev[d:d + n].copy_(host[h:h + n])
    return len(device_tensors) * len(runs)


def _op_columns(device_tensors: list, host_tensors: list, page_bytes: List[int]) -> tuple:
    """batch_ops' per-tensor columns: each device view's byte offset in its storage's NRT tensor, each host tensor's
    address, each page's bytes."""
    base_off = [t.storage_offset() * t.element_size() for t in device_tensors]
    return _column(base_off), _column([h.data_ptr() for h in host_tensors]), _column(page_bytes)


def page_bytes_of(tensors: list) -> List[int]:
    """Bytes of one page (first-dimension row) of each tensor."""
    return [t[0].numel() * t.element_size() for t in tensors]


def copy_pages_now(device_tensors: list, device_pages: List[int], host_tensors: list, to_host: bool) -> None:
    """Copy ``device_pages`` of each page-first device tensor to (``to_host``) or from the matching host tensor's pages
    ``0 .. len(device_pages) - 1``, and return when the bytes have landed: the host copies of a pool's pages outside
    HiCache (get_cpu_copy / load_cpu_copy). On a NeuronCore the runtime's batch DMA moves them after every launched
    graph has finished (one may still write or read those pages), so the copy compiles no graph of its own; host tensors
    stand in for the device in CPU mode."""
    if not device_pages:
        return
    runs = page_runs(list(device_pages), list(range(len(device_pages))))
    if device_tensors[0].device.type != "neuron":
        torch_page_copy(device_tensors, host_tensors, runs, to_host)
        return
    import torch_neuronx

    torch_neuronx.synchronize()  # the graphs launched so far, and every pool allocation bound to device memory
    nrt_batch_copy([_nrt_handle(t) for t in device_tensors],
                   _op_columns(device_tensors, host_tensors, page_bytes_of(device_tensors)), runs, to_host)


def pinned_host_like(tensor, pages: int, on_neuron: bool):
    """A host tensor of ``pages`` pages of ``tensor``'s page shape: pinned and page-locked when the runtime DMAs into
    it (the module docstring), a plain host tensor in CPU mode."""
    region = torch.empty((pages, *tensor.shape[1:]), dtype=tensor.dtype, pin_memory=on_neuron)
    if on_neuron:
        _mlock(region)
    return region


def _mlock(tensor) -> None:
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    nbytes = tensor.numel() * tensor.element_size()
    if libc.mlock(ctypes.c_void_p(tensor.data_ptr()), ctypes.c_size_t(nbytes)) != 0:
        raise OSError(ctypes.get_errno(), f"mlock of the {nbytes}-byte HiCache host buffer failed (RLIMIT_MEMLOCK)")


# ---- transfers --------------------------------------------------------------------------------------------------


class _Job:
    __slots__ = ("seq", "to_host", "runs", "pages", "fence", "done", "error", "t_queued", "t_start", "t_end", "ops",
                 "dev_runs", "host_runs")

    def __init__(self, seq, to_host, runs, pages, fence, dev_runs=0, host_runs=0):
        self.seq = seq
        self.to_host = to_host
        self.runs = runs            # [(device page, host page, count)]
        self.pages = pages
        self.dev_runs = dev_runs    # stats: runs of consecutive device pages alone (what a device-side gather would leave)
        self.host_runs = host_runs  # stats: runs of consecutive host pages alone
        self.fence = fence          # a torch-neuronx event the DMAs wait for (stores; loads on an aggregated server)
        self.done = threading.Event()
        self.error = None
        self.t_queued = time.perf_counter()
        self.t_start = self.t_end = None
        self.ops = 0


def page_runs(device_pages: List[int], host_pages: List[int]) -> list:
    """(device page, host page, count) runs over pairs consecutive on both sides, in the given order."""
    runs = []
    for d, h in zip(device_pages, host_pages):
        if runs and runs[-1][0] + runs[-1][2] == d and runs[-1][1] + runs[-1][2] == h:
            runs[-1][2] += 1
        else:
            runs.append([d, h, 1])
    return [tuple(r) for r in runs]


def _count_runs(pages: List[int]) -> int:
    return sum(1 for i, p in enumerate(pages) if i == 0 or p != pages[i - 1] + 1)


def run_nrt_batch(handles: list, ops: np.ndarray, to_host: bool) -> None:
    """One ``nrt_tensor_read_batch`` (to the host) or ``nrt_tensor_write_batch`` call over the [tensors, runs]
    _BatchOp records ``ops`` (batch_ops), row r against the NRT tensor ``handles[r]``. ``ops`` must stay alive until
    the call returns."""
    lib, _ = _libnrt()
    batches = (_Batch * len(handles))()
    row, stride = ops.ctypes.data, ops.strides[0]
    for r, handle in enumerate(handles):
        batches[r].tensor = handle
        batches[r].ops = ctypes.cast(row + r * stride, _BATCH_OP_PTR)
        batches[r].num_ops = ops.shape[1]
    f = lib.nrt_tensor_read_batch if to_host else lib.nrt_tensor_write_batch
    rc = f(batches, len(handles))
    if rc != 0:
        raise RuntimeError(f"nrt_tensor_{'read' if to_host else 'write'}_batch: NRT status {rc}")


def run_host_batch(bases: list, ops: np.ndarray, to_host: bool) -> None:
    """``run_nrt_batch`` for tensors in host memory (CPU mode, tests): row r's offsets are into the storage at address
    ``bases[r]``; each op is one memmove. It runs the same op records the runtime call would get, so the host path
    checks their arithmetic byte for byte."""
    for r, base in enumerate(bases):
        for offset, size, buffer in ops[r].tolist():
            if to_host:
                ctypes.memmove(buffer, base + offset, size)
            else:
                ctypes.memmove(base + offset, buffer, size)


class TransferEngine:
    """Runs the host pool's copies with the runtime's batch DMA and tracks each as a job with a sequence number; the
    HiCache events (XferEvent) complete with a job.

    A region r is a span of one device tensor addressed in pages: device page d at byte ``dev_offsets[r] + d *
    page_bytes[r]`` of ``device_tensors[r]``'s storage, host page h at address ``host_bases[r] + h * page_bytes[r]``.
    By default a region is a whole tensor (its storage offset) and a whole host region (its address), as the KV pool
    uses it; the state host pool (mem_cache/state_host_pool) passes one region per (component, layer) of the same
    tensors, so a slot of every layer moves in one call."""

    # the host pool's label in logs, and whether loads are read back and compared (SGLANG_NEURON_HICACHE_VERIFY)
    name = "kv"
    verify = False

    def __init__(self, device_tensors: list, host_regions: list, page_bytes: List[int], *, dev_offsets=None,
                 host_bases=None):
        self.devt = device_tensors
        self.host = host_regions
        self.page_bytes = page_bytes
        self.device = device_tensors[0].device
        self._seq = 0
        self._jobs = {}
        self._lock = threading.Lock()
        self.stats = collections.Counter()
        self._attach()
        if self.handles is None and (dev_offsets is not None or host_bases is not None):
            raise ValueError("a CPU-mode transfer engine moves whole tensors; regions within a tensor need the "
                             "HostTransferEngine")
        if dev_offsets is None:
            dev_offsets = [t.storage_offset() * t.element_size() for t in device_tensors]
        if host_bases is None:
            host_bases = [h.data_ptr() for h in host_regions]
        if not len(device_tensors) == len(dev_offsets) == len(host_bases) == len(page_bytes):
            raise ValueError("a transfer region needs one device tensor, device offset, host base and page size")
        self._op_cols = (_column(dev_offsets), _column(host_bases), _column(page_bytes))
        # one thread per direction: a store waits on the host for the graphs launched before it, and a load (which the
        # next forward waits for) must not queue behind that wait
        self._queues = {True: collections.deque(), False: collections.deque()}
        self._cvs = {True: threading.Condition(), False: threading.Condition()}
        for to_host, name in ((True, "store"), (False, "load")):
            threading.Thread(target=self._loop, args=(to_host,), name=f"neuron-hicache-{name}", daemon=True).start()

    def _attach(self) -> None:
        """The NRT tensor of every region's device tensor (a tensor named by several regions is looked up once); in CPU
        mode (host tensors stand in for the device, nothing launches graphs) none."""
        if self.device.type != "neuron":
            self.tnx, self.index, self.handles = None, None, None
            return
        import torch_neuronx

        self.tnx = torch_neuronx
        self.index = self.device.index if self.device.index is not None else torch_neuronx.current_device()
        torch_neuronx.synchronize()  # every pool allocation bound to device memory
        found = {}
        for t in self.devt:
            if id(t) not in found:
                found[id(t)] = _nrt_handle(t)
        self.handles = [found[id(t)] for t in self.devt]

    def _fence(self):
        return self.tnx.default_stream(self.index).record_event() if self.tnx is not None else None

    def _execute(self, to_host: bool, ops: np.ndarray) -> None:
        run_nrt_batch(self.handles, ops, to_host)

    # -- submission (scheduler thread) --
    def submit(self, to_host: bool, device_pages: List[int], host_pages: List[int]) -> int:
        runs = page_runs(device_pages, host_pages)
        with self._lock:
            self._seq += 1
            seq = self._seq
        # a copy waits for the graphs launched up to now (overlap scheduling: possibly still running). A store reads the
        # KV they write; a load may write a page one of them still writes, since an aggregated server frees a finished
        # request's pages while an earlier step may still write its over-allocated slot, and a prefill frees an aborted
        # chunked request's pages while its last chunk still writes them (Scheduler.process_pending_chunked_abort).
        # The forward that reads a load queues behind those graphs anyway.
        fence = self._fence()
        # the pairs can move in any order: the fewest runs one side alone allows is its sorted pages' runs (stats only)
        job = _Job(seq, to_host, runs, len(device_pages), fence,
                   _count_runs(sorted(device_pages)) if _STATS else 0, _count_runs(sorted(host_pages)) if _STATS else 0)
        with self._lock:
            self._jobs[seq] = job
        cv = self._cvs[to_host]
        with cv:
            self._queues[to_host].append(job)
            cv.notify()
        return seq

    # -- the transfer threads --
    def _loop(self, to_host: bool) -> None:
        cv, queue = self._cvs[to_host], self._queues[to_host]
        while True:
            with cv:
                while not queue:
                    cv.wait()
                job = queue.popleft()
            try:
                if job.fence is not None:
                    job.fence.synchronize()
                job.t_start = time.perf_counter()
                if self.verify:
                    cid, before, t0 = _call_begin(f"{self.name}:{'store' if job.to_host else 'load'}#{job.seq}")
                self._run_batch(job)
                job.t_end = time.perf_counter()
                if self.verify:
                    during = _call_end(cid)
                    if not job.to_host:
                        logger.info("[neuron hicache] verify %s load #%d t=%.6f..%.6f runs=%s concurrent_before=%s "
                                    "concurrent_during=%s", self.name, job.seq, t0, time.time(), job.runs, before,
                                    during)
                        self._verify_load(job, "returned")
            except Exception as exc:  # noqa: BLE001 - reported where the job is waited for
                job.error = exc
                logger.exception("[neuron hicache] transfer %d failed", job.seq)
            job.done.set()

    def _run_batch(self, job: _Job) -> None:
        if self.handles is None:  # CPU mode: host tensors stand in for the device
            job.ops = torch_page_copy(self.devt, self.host, job.runs, job.to_host)
            return
        ops = batch_ops(job.runs, *self._op_cols)  # kept alive by this frame until the call returns
        self._execute(job.to_host, ops)
        job.ops = len(self.devt) * len(job.runs)

    # -- completion (any thread) --
    def done(self, seq: int) -> bool:
        job = self._jobs.get(seq)
        return job is None or job.done.is_set()

    def wait(self, seq: int) -> None:
        job = self._jobs.get(seq)
        if job is None:
            return
        job.done.wait()
        self._retire(job)

    def wait_on_device(self, seq: int) -> None:
        """Make the forward that follows wait for load ``seq``: the graphs are submitted from this thread, so waiting
        here on the host orders the load before them."""
        job = self._jobs.get(seq)
        if job is None:
            return
        t0 = time.perf_counter()
        if _STATS:
            idle = self._wait_device_idle(job)
        else:
            job.done.wait()
        if self.verify and not job.to_host and job.error is None:
            # the device work this thread queued since the load (the step's eager page fills) has run
            self.tnx.synchronize()
            self._verify_load(job, "before-forward")
        waited = 1e3 * (time.perf_counter() - t0)
        self.stats["load_wait_ms"] += waited
        if _STATS:
            self.stats["load_idle_ms"] += idle
            logger.info("[neuron hicache] forward waited %.2f ms for load #%d, device idle %.2f ms", waited, seq, idle)
        self._retire(job)

    @staticmethod
    def _wait_device_idle(job: _Job) -> float:
        """Wait on the host for load ``job`` and return the part of the wait the device sat idle (ms): the time after
        the newest graph this thread launched had finished, all of it when none was running. The rest of the wait was
        hidden behind that graph."""
        from sglang_neuron.backend import device_io

        inflight = device_io._inflight()
        launch = inflight[-1] if inflight else None
        idle_since = None if launch is not None and not launch.query() else time.perf_counter()
        while not job.done.wait(0.0005):
            if idle_since is None and launch.query():
                idle_since = time.perf_counter()
        return 0.0 if idle_since is None else 1e3 * (time.perf_counter() - idle_since)

    def _retire(self, job: _Job) -> None:
        if job.error is not None:
            raise RuntimeError(f"HiCache transfer {job.seq} failed: {job.error!r}") from job.error
        if _STATS and job.t_end is not None:
            kind = "store" if job.to_host else "load"
            logger.info("[neuron hicache] %s #%d pages=%d runs=%d ops=%d queued->start %.2f ms, run %.2f ms "
                        "dev_runs=%d host_runs=%d", kind, job.seq, job.pages, len(job.runs), job.ops,
                        1e3 * ((job.t_start or job.t_queued) - job.t_queued),
                        1e3 * (job.t_end - (job.t_start or job.t_end)), job.dev_runs, job.host_runs)
        with self._lock:
            self._jobs.pop(job.seq, None)

    def latest(self) -> int:
        return self._seq

    def wait_all(self) -> None:
        for seq in sorted(self._jobs):
            self.wait(seq)

    def unfinished_loads(self) -> List[int]:
        with self._lock:
            jobs = list(self._jobs.values())
        return [j.seq for j in jobs if not j.to_host and not j.done.is_set()]

    def wait_loads(self) -> None:
        """Wait on the host until every load submitted so far has finished (its batch call returned). The loads stay
        tracked: the forward that reads one still retires it (wait_on_device) and reports its error."""
        with self._lock:
            jobs = [j for j in self._jobs.values() if not j.to_host]
        for job in jobs:
            job.done.wait()

    # -- debug: SGLANG_NEURON_HICACHE_VERIFY --
    def _verify_load(self, job: _Job, when: str) -> None:
        """Read the device pages of load ``job`` back (``nrt_tensor_read``, synchronous, one call per tensor and run)
        and compare them with their host pages in 256-byte chunks; log each differing page's chunk runs and whether
        the device bytes there are zero."""
        lib, _ = _libnrt()
        bad, report = 0, []
        for r, (handle, region, pb) in enumerate(zip(self.handles, self.host, self.page_bytes)):
            host = region.reshape(region.shape[0], -1).view(torch.uint8).numpy()
            base = int(self._op_cols[0][r, 0])
            for d, h, n in job.runs:
                got = np.empty((n, pb), dtype=np.uint8)
                rc = lib.nrt_tensor_read(ctypes.c_void_p(handle), got.ctypes.data_as(ctypes.c_void_p),
                                         ctypes.c_uint64(base + d * pb), ctypes.c_uint64(n * pb))
                if rc != 0:
                    raise RuntimeError(f"nrt_tensor_read: NRT status {rc}")
                want = host[h:h + n]
                for j in np.nonzero((got != want).any(axis=1))[0]:
                    bad += 1
                    chunks = np.nonzero((got[j] != want[j]).reshape(-1, 256).any(axis=1))[0]
                    runs = _index_runs(chunks.tolist())
                    zero = all(not got[j, a * 256:b * 256].any() for a, b in runs)
                    if len(report) < 24:
                        report.append(f"t{r} dev{d + j} host{h + j} chunks{runs} zero={zero}")
        logger.info("[neuron hicache] verify %s load #%d %s: %d bad pages %s", self.name, job.seq, when, bad, report)


def _index_runs(values: List[int]) -> list:
    """[start, end) runs of consecutive ints."""
    runs = []
    for v in values:
        if runs and runs[-1][1] == v:
            runs[-1][1] += 1
        else:
            runs.append([v, v + 1])
    return runs


# the runtime batch calls in flight in this process (debug, SGLANG_NEURON_HICACHE_VERIFY): which overlap a load
_CALLS = {"lock": threading.Lock(), "next": 0, "active": {}, "recent": collections.deque(maxlen=256)}


def _call_begin(label: str):
    with _CALLS["lock"]:
        cid = _CALLS["next"]
        _CALLS["next"] += 1
        before = sorted(_CALLS["active"].values())
        _CALLS["active"][cid] = label
        _CALLS["recent"].append((cid, label))
    return cid, before, time.time()


def _call_end(cid: int) -> list:
    with _CALLS["lock"]:
        _CALLS["active"].pop(cid, None)
        return [label for c, label in _CALLS["recent"] if c > cid]


class HostTransferEngine(TransferEngine):
    """The TransferEngine over device tensors in host memory (CPU mode and the unit tests): the same regions, runs,
    op records, threads and jobs, with each op a memmove (run_host_batch) and no device fence."""

    def _attach(self) -> None:
        if any(t.device.type != "cpu" for t in self.devt):
            raise ValueError("the host transfer engine moves host tensors only")
        self.handles = [t.untyped_storage().data_ptr() for t in self.devt]

    def _fence(self):
        return None

    def _execute(self, to_host: bool, ops: np.ndarray) -> None:
        run_host_batch(self.handles, ops, to_host)


# ---- the events SGLang's HiCache waits on ----------------------------------------------------------------------

_ENGINES: list = []


def loads_in_flight() -> list:
    """(engine index, seq) of every HiCache load of this process that has not finished."""
    return [(i, s) for i, e in enumerate(_ENGINES) for s in e.unfinished_loads()]


def wait_for_loads() -> None:
    """Wait on the host until every HiCache load of this process has finished (mem_cache/allocator: a device page fill
    does not run while a load DMA writes the same buffer). Free without HiCache: no engine."""
    for engine in list(_ENGINES):
        engine.wait_loads()


class XferEvent:
    """Stands in for SGLang's device events in HiCache (l2_transfer's transfer acks and cache_controller's load
    events): record() binds to the newest transfer job any Neuron host pool has submitted, and query / synchronize
    follow that job. An event recorded before any job, or after its job retired, is complete."""

    def __init__(self, enable_timing: bool = False):
        self._binds = None
        self._t = None

    def record(self, stream=None):
        self._binds = [(e, e.latest()) for e in _ENGINES]
        self._t = time.perf_counter()

    def query(self) -> bool:
        return all(e.done(seq) for e, seq in (self._binds or ()))

    def synchronize(self) -> None:
        for e, seq in self._binds or ():
            e.wait(seq)

    def wait(self, stream=None) -> None:
        return None

    def wait_on_device(self) -> None:
        for e, seq in self._binds or ():
            e.wait_on_device(seq)

    def elapsed_time(self, end) -> float:
        return 1e3 * ((end._t or 0) - (self._t or 0))


def make_timing_event_pair():
    """l2_transfer.make_timing_event_pair with XferEvents (no device timing)."""
    return XferEvent(), XferEvent(), False


def layer_loading_event_init(original, self, num_layers: int):
    original(self, num_layers)
    self.load_events = [XferEvent() for _ in range(num_layers)]
    self.start_event = XferEvent()


def set_consumer_waits(original, self, index: int):
    """LayerDoneCounter.set_consumer, then the next forward waits for that load: a compiled forward reads every
    layer's pages at once and never calls the per-layer wait_until."""
    original(self, index)
    if index is not None and index >= 0:
        ev = self.events[index].finish_event
        if isinstance(ev, XferEvent):
            ev.wait_on_device()


# ---- the host pools ---------------------------------------------------------------------------------------------


class NeuronPagedHostPool(HostKVCache):
    """A host copy of a Neuron pool's page-first device tensors for HiCache L2: one host region per device tensor,
    ``[host pages, *page shape]``, and a TransferEngine that moves whole pages between them. The pools of each layout
    (NeuronHostKVPool for the MHA pool, mla_host_pool.py for the MLA / DSA pools) only name their tensors.

    ``page_size`` is SGLang's (the allocator's) page and ``dcp_size`` widens it as HostKVCache does: a host page holds
    one device page of every region, whatever number of tokens of it this rank stores. Host and device indices are
    whole aligned pages of ``logical_page_size`` tokens; the page id is the row of every region. A load runs whole at
    ``load_layer`` (the first layer the pool holds): the compiled forward reads every layer at once.

    ``side_page_bytes``: the bytes a page of the side pools that mirror this pool's host pages hold (the DSA index
    keys, mla_host_pool.NeuronIndexKeyHostPool). They count in ``size_per_token``, so a ``host_size`` (SGLang's
    --hicache-size share of the device pool, whose bytes include the side buffers) buys pages for the side pools too.

    ``unit``: the tokens of one device page when SGLang's HiCache page is several of them (the chunk-keyed tree's page
    is the prefill chunk, a whole number of KV pages: Qwen3.5's 2048-token chunk of 512-token pages). A host page then
    holds ``page_size // unit`` device pages of every region, and a transfer moves each whole, aligned device page of
    its indices on its own (a chunk's KV pages need not be consecutive on the device)."""

    # whether the plugin's storage tiers (allocator "dynamic": mem_cache/remote_dram, mem_cache/mooncake_store) may
    # sit below this host pool
    remote_dram_tier = False

    @classmethod
    def check_allocator(cls, allocator_type, pool_label) -> None:
        """SGLang's get_allocator_type() names the storage backend: "dynamic" is one of the plugin's storage tiers, the
        remote DRAM tier or Mooncake Store (the platform refuses every other backend), which read and write these host
        pages themselves (batch_*_v1); validated on the MHA pool only (``remote_dram_tier``), so the other layouts'
        host pools stay L2 only."""
        allowed = (None, "default", "dynamic") if cls.remote_dram_tier else (None, "default")
        if allocator_type not in allowed:
            raise ValueError(f"Neuron HiCache allocates its own host memory, not a {allocator_type} allocator"
                             + ("" if cls.remote_dram_tier else f" (no remote DRAM tier for a {pool_label} host pool)"))

    def __init__(self, device_tensors, *, dtype, page_size: int, layer_num: int, device_capacity_tokens: int,
                 start_layer: int, end_layer: int, host_to_device_ratio: float, host_size, layout: str,
                 allocator_type, pool_label: str, dcp_size: int = 1, dcp_rank: int = 0, host_token_capacity=None,
                 device_page_limit=None, load_layer: int = 0, side_page_bytes: int = 0, unit=None):
        if layout != "layer_first":
            raise ValueError(f"Neuron HiCache keeps a layer_first host pool, not --hicache-mem-layout {layout}")
        self.check_allocator(allocator_type, pool_label)
        self.mode = "batch"  # the only transfer mode; the remote DRAM tier's backend checks it
        self.device_tensors = list(device_tensors)
        if not self.device_tensors:
            raise ValueError("a Neuron HiCache host pool needs at least one device tensor")
        self.device_page_num = int(self.device_tensors[0].shape[0])
        self.page_bytes = page_bytes_of(self.device_tensors)
        for t, pb in zip(self.device_tensors, self.page_bytes):
            if not t.is_contiguous() or t.shape[0] != self.device_page_num:
                raise ValueError("Neuron HiCache needs contiguous [pages, ...] device tensors of one page count")
            if pb == 0:
                raise ValueError("empty KV page")
        # the pages SGLang's allocator hands out (a layout may hold more rows, e.g. a scratch block, never cached)
        self.device_page_limit = self.device_page_num if device_page_limit is None else int(device_page_limit)
        self.on_neuron = self.device_tensors[0].device.type == "neuron"
        self.layer_num = self.target_layer_num = layer_num
        self.load_layer = load_layer
        self.unit = int(unit) if unit else int(page_size)
        if self.unit <= 0 or page_size % self.unit:
            raise ValueError(f"a HiCache page of {page_size} tokens is not whole device pages of {self.unit} tokens")
        self.units_per_page = page_size // self.unit
        physical_page = self.unit // dcp_size
        super().__init__(dtype=dtype, device_capacity_tokens=device_capacity_tokens,
                         size_per_token=-(-(sum(self.page_bytes) + int(side_page_bytes)) // physical_page),
                         start_layer=start_layer,
                         end_layer=end_layer, host_to_device_ratio=host_to_device_ratio, host_size=host_size,
                         page_size=page_size, layout=layout, pin_memory=self.on_neuron, device="cpu",
                         allocator_type="default", dcp_size=dcp_size, dcp_rank=dcp_rank, pool_label=pool_label,
                         host_token_capacity=host_token_capacity)
        self.engine = TransferEngine(self.device_tensors, self.host_regions, self.page_bytes)
        _ENGINES.append(self.engine)
        logger.info("[neuron hicache] %s host pool: %d pages x %d regions (%.2f GiB)", pool_label, self.page_num,
                    len(self.device_tensors), sum(r.numel() * r.element_size() for r in self.host_regions) / (1 << 30))

    def get_size_per_token(self):
        return self.size_per_token

    def init_kv_buffer(self):
        """One host region per device tensor: [host pages, *page shape] in the tensor's dtype, pinned and page-locked
        on a NeuronCore (the runtime DMAs into it after its call returns)."""
        self.host_regions = [pinned_host_like(t, self.page_num * self.units_per_page, self.on_neuron)
                             for t in self.device_tensors]
        return self.host_regions

    def _pages(self, indices: torch.Tensor, limit: int, reserved_zero: bool) -> List[int]:
        page = self.unit if getattr(self, "units_per_page", 1) > 1 else self.logical_page_size
        idx = indices.to("cpu", torch.int64) if indices.device.type != "cpu" else indices.to(torch.int64)
        if idx.numel() % page:
            raise ValueError("HiCache transfers move whole pages")
        rows = idx.reshape(-1, page)
        starts = rows[:, 0]
        if torch.any(starts % page) or not torch.equal(rows, starts[:, None] + torch.arange(page)):
            raise ValueError("HiCache indices must be whole, aligned pages")
        pages = (starts // page).tolist()
        lo = 1 if reserved_zero else 0
        if pages and (min(pages) < lo or max(pages) >= limit):
            raise ValueError(f"HiCache page out of range [{lo}, {limit})")
        return pages

    def _submit(self, to_host, host_indices, device_indices):
        host = self._pages(host_indices, self.page_num * self.units_per_page, False)
        dev = self._pages(device_indices, self.device_page_limit, True)
        if len(host) != len(dev):
            raise ValueError("HiCache host and device page counts differ")
        if not host:
            return
        seq = self.engine.submit(to_host, dev, host)
        if _VERIFY:
            self._verify(seq, to_host, dev, host)

    def _verify(self, seq, to_host, dev, host) -> None:
        self.engine.wait(seq)
        bad = 0
        for t, region in zip(self.device_tensors, self.host_regions):
            got = t.to("cpu")
            for d, h in zip(dev, host):
                if not torch.equal(got[d].reshape(-1).view(torch.uint8), region[h].reshape(-1).view(torch.uint8)):
                    bad += 1
        self.engine.stats["verify_pages"] += len(dev)
        self.engine.stats["verify_bad"] += bad
        logger.info("[neuron hicache] verify %s #%d: %d pages x %d regions, %d differ (totals %s)",
                    "store" if to_host else "load", seq, len(dev), len(self.device_tensors), bad,
                    dict(self.engine.stats))

    def backup_from_device_all_layer(self, device_pool, host_indices, device_indices, io_backend):
        self._submit(True, host_indices, device_indices)

    def load_to_device_per_layer(self, device_pool, host_indices, device_indices, layer_id, io_backend, *,
                                 is_draft=False):
        """The whole load at ``load_layer``: the compiled forward reads every layer at once."""
        if is_draft:
            raise ValueError("Neuron HiCache keeps no draft KV")
        if layer_id == self.load_layer:
            self._submit(False, host_indices, device_indices)

    def get_data_page(self, index, flat=True):
        raise NotImplementedError("Neuron HiCache storage is the remote DRAM tier's batch_*_v1 only")

    def get_dummy_flat_data_page(self):
        raise NotImplementedError("Neuron HiCache storage is the remote DRAM tier's batch_*_v1 only")

    def set_from_flat_data_page(self, index, data_page):
        raise NotImplementedError("Neuron HiCache storage is the remote DRAM tier's batch_*_v1 only")


class NeuronHostKVPool(NeuronPagedHostPool):
    """The host copy of a Neuron MHA pool (NeuronMHATokenToKVPoolKind), for HiCache L2: every layer's K, then every
    layer's V."""

    remote_dram_tier = True

    def __init__(self, buffer_info, host_to_device_ratio, host_size, page_size, layout, pin_memory=True,
                 device="cpu", allocator_type="default", *, device_capacity_tokens, start_layer, end_layer,
                 draft_buffer_infos=(), write_back_staging=None, pool_label: str = "kv", head_group_num: int = 1,
                 unit_tokens=None):
        if draft_buffer_infos or write_back_staging is not None:
            raise ValueError("Neuron HiCache keeps no draft KV and no write-back staging")
        self.check_allocator(allocator_type, pool_label)
        k, v = list(buffer_info.buffers.k), list(buffer_info.buffers.v)
        self.head_num = buffer_info.head_num
        self.head_dim = buffer_info.head_dim
        # tokens of one device page: a page row holds them for every head of the pool (a HiCache page may hold
        # several, the chunk-keyed tree's chunk). A pool whose page id spans more than its row's tokens passes
        # unit_tokens itself (the CP pool's page id is the token page on every rank; its row holds the rank's
        # block of it, NeuronCpHostKVPool).
        if unit_tokens is not None:
            unit = int(unit_tokens)
        else:
            unit = k[0][0].numel() // max(1, int(self.head_num) * int(self.head_dim)) if len(k) else page_size
        super().__init__(k + v, dtype=buffer_info.dtype, page_size=page_size, layer_num=len(k),
                         device_capacity_tokens=device_capacity_tokens, start_layer=start_layer, end_layer=end_layer,
                         host_to_device_ratio=host_to_device_ratio, host_size=host_size, layout=layout,
                         allocator_type=allocator_type, pool_label=pool_label,
                         unit=unit if unit and page_size % unit == 0 and unit < page_size else None)


def _row_tokens(buffer_info) -> int:
    """Tokens of one page row of a Neuron MHA pool's buffers: the whole page for a whole-page pool; the rank's
    page / kv_cp-token block of the page under token CP (the row then holds the rank's every KV head of that block,
    memory_pool.py)."""
    k = buffer_info.buffers.k
    if not len(k):
        return 0
    return k[0][0].numel() // max(1, int(buffer_info.head_num) * int(buffer_info.head_dim))


class NeuronCpHostKVPool(NeuronHostKVPool):
    """The host copy of a token context-parallel Neuron MHA pool (kv_cp > 1; the Qwen3.5 DCP decode's
    Qwen35SlotKVPool, models/qwen35/kv_pool.py). Page p's row of every region holds this rank's page /
    kv_cp-token block of page p, in the rank's every KV head, and the page id is the token page on every rank —
    the allocator and the tree are CP-unaware — so a host page mirrors one device page id and the controller's
    indices pass untranslated, each rank storing its own shard of the page (the MLA CP precedent,
    mem_cache/mla_host_pool.py). The transfer page counts the page's tokens, not the row's, so the unit is the
    pool's page_size, where the base class would take the row's tokens and scatter one tree page over kv_cp page
    ids. L2 only: the remote DRAM tier moves whole host pages by pointer, which a rank's shard of a page is not
    (docs/design/context-parallelism.md#the-host-tier)."""

    remote_dram_tier = False

    def __init__(self, buffer_info, *args, **kwargs):
        row_tokens = _row_tokens(buffer_info)
        if not 0 < row_tokens < buffer_info.page_size:
            raise ValueError(f"a CP host pool takes a pool whose page row holds the rank's block of the page "
                             f"({row_tokens} tokens of a {buffer_info.page_size}-token page)")
        kwargs["unit_tokens"] = buffer_info.page_size
        super().__init__(buffer_info, *args, **kwargs)


def host_pool_cls(original, buffer_info):
    """hybrid_pool_assembler.get_mha_host_pool_cls: the Neuron host pool for a Neuron pool's buffers."""
    k = buffer_info.buffers.k
    if len(k) and k[0].device.type == "neuron":
        if 0 < _row_tokens(buffer_info) < buffer_info.page_size:
            # token CP: the row holds the rank's block of the page; only a pool that serves HiCache under CP
            # publishes such buffers (memory_pool.py's refusal gates the rest)
            return NeuronCpHostKVPool
        return NeuronHostKVPool
    return original(buffer_info)


def _without_stream_fence(original, self, *args, **kwargs):
    """HiCacheController.start_writing / start_loading without the scheduler's forward-stream fence: SGLang's L2
    streams are host streams here, which cannot wait on the Neuron forward stream, and the Neuron host pool orders
    the copies itself: a store and a load each wait for the graphs launched before them (TransferEngine.submit)."""
    saved = self.write_fence_stream, self.load_fence_stream
    self.write_fence_stream = self.load_fence_stream = None
    try:
        return original(self, *args, **kwargs)
    finally:
        self.write_fence_stream, self.load_fence_stream = saved


# Upstream functions whose logic these hooks skip, replace or rely on, pinned by md5 against SGLang 43881a7
# (sglang_patches.source_guard); a changed source stops the launch.
_SOURCES_MD5 = {
    # replaced: XferEvents in place of the device module's (torch.cpu) events
    ("sglang.srt.mem_cache.l2_transfer", "make_timing_event_pair"): "ca588f6728d4ae3787c35257f6c2b715",
    # run without the forward-stream fence: the host pool orders the copies itself
    ("sglang.srt.managers.cache_controller", "HiCacheController.start_writing"): "2c55f2e379859333da51f64479e82790",
    ("sglang.srt.managers.cache_controller", "HiCacheController.start_loading"): "83d37fb82508b0e4c4892fc3e0ad92db",
    # its events are replaced with XferEvents after it runs
    ("sglang.srt.managers.cache_controller", "LayerLoadingEvent.__init__"): "2275fcfd6548ffb93114b800d98dfa11",
    # NeuronHostKVPool overrides the methods these forward to; an upstream body of their own would bypass it
    ("sglang.srt.mem_cache.pool_host.base", "HostKVCache.backup_from_device_all_layer_physical"):
        "1460920848050c18fc84199b1a01fb3e",
    ("sglang.srt.mem_cache.pool_host.base", "HostKVCache.load_to_device_per_layer_physical"):
        "6e4a31e3066f9a48b520133b2b51e8a0",
}


def register_hooks(registry, hook_type) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_SOURCES_MD5, __name__)
    for name in ("start_writing", "start_loading"):
        registry.register(f"sglang.srt.managers.cache_controller.HiCacheController.{name}", _without_stream_fence,
                          hook_type.AROUND)
    registry.register("sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler.get_mha_host_pool_cls",
                      host_pool_cls, hook_type.AROUND)
    registry.register("sglang.srt.mem_cache.l2_transfer.make_timing_event_pair", make_timing_event_pair,
                      hook_type.REPLACE)
    registry.register("sglang.srt.managers.cache_controller.LayerLoadingEvent.__init__", layer_loading_event_init,
                      hook_type.AROUND)
    registry.register("sglang.srt.managers.cache_controller.LayerDoneCounter.set_consumer", set_consumer_waits,
                      hook_type.AROUND)
    from sglang_neuron.mem_cache import hicache_hold, mla_host_pool, state_host_pool

    hicache_hold.register_hooks(registry, hook_type)
    # a Neuron state pool's host pool (hybrid_pool_assembler.build_mamba_host_pool)
    state_host_pool.register_hooks(registry, hook_type)
    # the MLA / DSA pools' host pools, which SGLang's assembler builds without a per-platform class choice
    mla_host_pool.register_hooks(registry, hook_type)
