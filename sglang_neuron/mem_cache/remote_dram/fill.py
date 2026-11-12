# SPDX-License-Identifier: Apache-2.0
"""Memory checks and the fill of a lease server's region. Standard library only.

A region is a list of chunks, each its own anonymous mapping between guard pages, so no two chunks merge into one VMA:
`/proc/self/smaps` then tells how much of each chunk is in transparent huge pages, and the lease server registers,
and can free, each chunk alone.

EFA maps a registration in the page size it finds. A chunk with 4 KiB pages costs many more translation entries, and
on trn3 the registration of a large total of 4 KiB-page memory fails with ENOMEM (measured at 64 GiB chunks of a
1 TB region; one 256 GiB chunk of 4 KiB pages registered). So the server asks for huge
pages, faults every chunk in from threads pinned to one NUMA node (where the kernel compacts for a huge page fault),
tries MADV_COLLAPSE on what stayed small, and refuses a region whose 4 KiB-page chunks are over a budget.
"""

from __future__ import annotations

import ctypes
import logging
import mmap
import os
import queue
import threading
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

GIB = 1 << 30
HUGE = 2 << 20  # one transparent huge page (x86_64 and aarch64 with 4 KiB pages)
MADV_HUGEPAGE = getattr(mmap, "MADV_HUGEPAGE", 14)
MADV_POPULATE_WRITE = 23  # Linux 5.14
MADV_COLLAPSE = 25  # Linux 6.1
MPOL_PREFERRED = 1
SYS_MBIND = {"x86_64": 237, "aarch64": 235}
#: The most bytes of 4 KiB-page chunks a region may register (one 256 GiB chunk registered on trn3 EFA in the
#: reference work; more failed with ENOMEM).
SMALL_PAGE_BUDGET = 256 * GIB


class FillError(RuntimeError):
    """The region cannot be filled or would not register: the server does not start."""


# -- host memory ------------------------------------------------------------------------------------------------------

def meminfo(path: str = "/proc/meminfo") -> Dict[str, int]:
    """MemTotal, MemAvailable, MemFree, CommitLimit and Committed_AS, in bytes."""
    out: Dict[str, int] = {}
    with open(path) as handle:
        for line in handle:
            name, _, rest = line.partition(":")
            if name in ("MemTotal", "MemAvailable", "MemFree", "CommitLimit", "Committed_AS"):
                out[name] = int(rest.split()[0]) * 1024
    return out


def check_floors(nbytes: int, floor_avail: int, floor_free: int, info: Dict[str, int]) -> str:
    """Why a region of ``nbytes`` would leave the host under a floor, or "".

    ``floor_free`` 0 turns the MemFree check off: a serving host's page cache keeps MemFree low while MemAvailable,
    which counts the cache the kernel takes back, is the real room.

    >>> G = GIB
    >>> check_floors(900 * G, 150 * G, 0, {"MemAvailable": 1600 * G, "MemFree": 900 * G})
    ''
    >>> check_floors(900 * G, 800 * G, 0, {"MemAvailable": 1600 * G, "MemFree": 900 * G})[:12]
    'MemAvailable'
    """
    avail = info["MemAvailable"] - nbytes
    if avail < floor_avail:
        return (f"MemAvailable after the region is {avail / GIB:.1f} GiB, under the floor {floor_avail / GIB:.1f} GiB; "
                "lower --gib")
    free = info["MemFree"] - nbytes
    if floor_free > 0 and free < floor_free:
        return (f"MemFree after the region is {free / GIB:.1f} GiB, under the floor {floor_free / GIB:.1f} GiB; "
                "lower --gib or drop the page cache first")
    return ""


# -- NUMA -------------------------------------------------------------------------------------------------------------

def parse_cpulist(text: str) -> Set[int]:
    """The CPUs of a sysfs list such as ``0-3,8,10-11``.

    >>> sorted(parse_cpulist("0-2,5"))
    [0, 1, 2, 5]
    """
    out: Set[int] = set()
    for part in text.strip().split(","):
        if not part:
            continue
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


def node_cpus(node: int, sysfs: str = "/sys") -> Set[int]:
    with open(os.path.join(sysfs, "devices/system/node", f"node{node}", "cpulist")) as handle:
        return parse_cpulist(handle.read())


def efa_node(sysfs: str = "/sys") -> Optional[int]:
    """The NUMA node most of the host's EFA devices are on, or None when no device reports one."""
    base = os.path.join(sysfs, "class/infiniband")
    counts: Dict[int, int] = {}
    try:
        names = sorted(os.listdir(base))
    except OSError:
        return None
    for name in names:
        try:
            with open(os.path.join(base, name, "device/numa_node")) as handle:
                node = int(handle.read().strip())
        except (OSError, ValueError):
            continue
        if node >= 0:
            counts[node] = counts.get(node, 0) + 1
    if not counts:
        return None
    return max(sorted(counts), key=lambda n: counts[n])


# -- madvise and mbind ------------------------------------------------------------------------------------------------

def _libc() -> ctypes.CDLL:
    return ctypes.CDLL(None, use_errno=True)


def madvise(addr: int, nbytes: int, advice: int) -> None:
    """madvise(2) through ctypes, which drops the GIL (``mmap.madvise`` holds it while a fault runs)."""
    if _libc().madvise(ctypes.c_void_p(addr), ctypes.c_size_t(nbytes), ctypes.c_int(advice)) != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"madvise({advice}) of {nbytes} B: {os.strerror(err)}")


def mbind_preferred(addr: int, nbytes: int, node: int) -> None:
    """Prefer ``node`` for the mapping's pages (MPOL_PREFERRED falls back to other nodes when it is full, where
    MPOL_BIND would call the OOM killer)."""
    import platform

    number = SYS_MBIND.get(platform.machine())
    if number is None:
        raise OSError(0, f"mbind: no syscall number for {platform.machine()}")
    mask = (ctypes.c_ulong * ((node // 64) + 1))()
    mask[node // 64] = 1 << (node % 64)
    rc = _libc().syscall(ctypes.c_long(number), ctypes.c_void_p(addr), ctypes.c_ulong(nbytes),
                         ctypes.c_int(MPOL_PREFERRED), mask, ctypes.c_ulong(64 * len(mask) + 1), ctypes.c_uint(0))
    if rc != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"mbind(node {node}): {os.strerror(err)}")


# -- chunks -----------------------------------------------------------------------------------------------------------

def plan_ranges(nbytes: int, step: int) -> List[Tuple[int, int]]:
    """``[lo, hi)`` ranges of ``step`` bytes covering ``nbytes`` (the last one shorter).

    >>> plan_ranges(5 * HUGE, 2 * HUGE)
    [(0, 4194304), (4194304, 8388608), (8388608, 10485760)]
    """
    if nbytes <= 0 or step <= 0:
        raise ValueError(f"plan_ranges({nbytes}, {step})")
    return [(lo, min(lo + step, nbytes)) for lo in range(0, nbytes, step)]


_PROT_NONE = 0x0
_PROT_RW = 0x1 | 0x2
_MAP_PRIVATE_ANON_NORESERVE = 0x02 | 0x20 | 0x4000
_MAP_FAILED = ctypes.c_void_p(-1).value


class Chunk:
    """One anonymous private mapping of ``nbytes`` (its own VMA), 2 MiB aligned, asking for huge pages, not yet
    faulted in. Mapped and unmapped through libc, so nothing in Python holds a view of it when it is unmapped.

    The kernel merges adjacent anonymous mappings of the same flags into one VMA (measured on trn3: two 32 GiB
    chunks read as one VMA in smaps), so a chunk sits inside a PROT_NONE reservation with a guard on each side.
    """

    def __init__(self, nbytes: int):
        if nbytes <= 0 or nbytes % HUGE:
            raise ValueError(f"a chunk of {nbytes} B: give a positive multiple of {HUGE} B")
        libc = _libc()
        libc.mmap.restype = ctypes.c_void_p
        libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                              ctypes.c_long]
        reserved = nbytes + 2 * HUGE
        base = libc.mmap(None, reserved, _PROT_NONE, _MAP_PRIVATE_ANON_NORESERVE, -1, 0)
        if base is None or base == _MAP_FAILED:
            err = ctypes.get_errno()
            raise FillError(f"mmap of {reserved} B: {os.strerror(err)}")
        # at least one huge page of guard below, and at least one 4 KiB page above (base is page aligned)
        addr = (base + HUGE + HUGE - 1) // HUGE * HUGE
        self._base, self._reserved = base, reserved
        self.nbytes = nbytes
        self.addr: Optional[int] = addr
        if libc.mprotect(ctypes.c_void_p(addr), ctypes.c_size_t(nbytes), ctypes.c_int(_PROT_RW)) != 0:
            err = ctypes.get_errno()
            self.close()
            raise FillError(f"mprotect of {nbytes} B: {os.strerror(err)}")
        try:
            madvise(addr, nbytes, MADV_HUGEPAGE)
        except OSError as exc:
            logger.warning("chunk at %#x: MADV_HUGEPAGE refused (%s)", addr, exc)

    def view(self) -> memoryview:
        """A writable view of the chunk (tests, and the in-process store). Do not keep it past `close`."""
        if self.addr is None:
            raise ValueError("the chunk is unmapped")
        return memoryview((ctypes.c_char * self.nbytes).from_address(self.addr)).cast("B")

    def close(self) -> None:
        """Unmap: the pages go back to the host. Deregister the chunk first."""
        addr, self.addr = self.addr, None
        if addr is not None:
            _libc().munmap(ctypes.c_void_p(self._base), ctypes.c_size_t(self._reserved))


def fill(chunks: Sequence[Chunk], threads: int = 8, step: int = 1 * GIB, cpus: Optional[Iterable[int]] = None,
         node: Optional[int] = None, log_every_s: float = 10.0) -> float:
    """Fault every chunk in (MADV_POPULATE_WRITE) from ``threads`` threads pinned to ``cpus``, preferring ``node`` for
    the pages. Returns the seconds it took. Raises FillError when a range fails."""
    total = sum(c.nbytes for c in chunks)
    work: "queue.Queue[Tuple[int, int]]" = queue.Queue()
    for chunk in chunks:
        if node is not None:
            try:
                mbind_preferred(chunk.addr, chunk.nbytes, node)
            except OSError as exc:
                logger.warning("fill: mbind to node %d refused (%s); the pages go where the threads run", node, exc)
        for lo, hi in plan_ranges(chunk.nbytes, step):
            work.put((chunk.addr + lo, hi - lo))
    done = [0]
    errors: List[BaseException] = []
    lock = threading.Lock()
    cpu_set = set(cpus) if cpus else None

    def run() -> None:
        if cpu_set:
            os.sched_setaffinity(0, cpu_set)
        while not errors:
            try:
                addr, nbytes = work.get_nowait()
            except queue.Empty:
                return
            try:
                madvise(addr, nbytes, MADV_POPULATE_WRITE)
            except OSError as exc:
                errors.append(exc)
                return
            with lock:
                done[0] += nbytes

    t0 = time.monotonic()
    pool = [threading.Thread(target=run, name=f"remote-dram-fill-{i}", daemon=True) for i in range(max(1, threads))]
    for thread in pool:
        thread.start()
    last = t0
    while any(t.is_alive() for t in pool):
        for thread in pool:
            thread.join(timeout=0.5)
        now = time.monotonic()
        if now - last >= log_every_s:
            last = now
            logger.info("fill: %.1f of %.1f GiB in %.0f s", done[0] / GIB, total / GIB, now - t0)
    if errors:
        raise FillError(f"the fill failed: {errors[0]}")
    seconds = time.monotonic() - t0
    logger.info("fill: %.1f GiB in %.1f s (%.1f GiB/s) on %d threads", total / GIB, seconds,
                total / GIB / max(seconds, 1e-9), len(pool))
    return seconds


# -- huge-page coverage -----------------------------------------------------------------------------------------------

def vma_huge_bytes(ranges: Iterable[Tuple[int, int]], path: str = "/proc/self/smaps") -> Dict[int, int]:
    """AnonHugePages (bytes) of each ``(addr, nbytes)`` range, keyed by addr: the sum over the VMAs inside it (a
    range madvise or a registration split stays one range). A VMA that crosses a range's edge counts for none."""
    spans = sorted(ranges)
    out = {a: 0 for a, _ in spans}
    current: Optional[int] = None
    with open(path) as handle:
        for line in handle:
            head = line.split(None, 1)[0]
            if "-" in head and ":" not in head:
                lo, hi = (int(x, 16) for x in head.split("-", 1))
                current = next((a for a, n in spans if a <= lo and hi <= a + n), None)
            elif current is not None and line.startswith("AnonHugePages:"):
                out[current] += int(line.split()[1]) * 1024
    return out


def check_huge(chunks: Sequence[Chunk], allow_small: bool = False, budget: int = SMALL_PAGE_BUDGET,
               huge_of: Callable[[Iterable[Tuple[int, int]]], Dict[int, int]] = vma_huge_bytes,
               collapse: Callable[[int, int], None] = lambda a, n: madvise(a, n, MADV_COLLAPSE)) -> List[int]:
    """The indices of the chunks that are not fully in huge pages, after one MADV_COLLAPSE on each of them.

    Refuses (FillError) when those chunks add up to more than ``budget`` bytes, unless ``allow_small``.
    """
    huge = huge_of([(c.addr, c.nbytes) for c in chunks])
    small = [i for i, c in enumerate(chunks) if huge.get(c.addr, 0) < c.nbytes]
    if small:
        for i in small:
            try:
                collapse(chunks[i].addr, chunks[i].nbytes)
            except OSError as exc:
                logger.info("fill: MADV_COLLAPSE of chunk %d: %s", i, exc)
        huge = huge_of([(c.addr, c.nbytes) for c in chunks])
        small = [i for i, c in enumerate(chunks) if huge.get(c.addr, 0) < c.nbytes]
    at_4k = sum(chunks[i].nbytes for i in small)
    in_huge = sum(min(huge.get(c.addr, 0), c.nbytes) for c in chunks)
    total = sum(c.nbytes for c in chunks)
    logger.info("fill: %.1f of %.1f GiB in huge pages; %d of %d chunks (%.1f GiB) not fully", in_huge / GIB,
                total / GIB, len(small), len(chunks), at_4k / GIB)
    if at_4k > budget and not allow_small:
        raise FillError(f"{len(small)} chunks ({at_4k / GIB:.1f} GiB) are not fully in huge pages, over the "
                        f"{budget / GIB:.0f} GiB that EFA registers in 4 KiB pages; start again when the host has "
                        "free 2 MiB blocks, lower --gib, or pass --allow-small-pages")
    return small
