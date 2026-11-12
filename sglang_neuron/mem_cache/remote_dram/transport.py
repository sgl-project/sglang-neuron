# SPDX-License-Identifier: Apache-2.0
"""How a rank's pages move between its staging arena and its remote lease.

Layout: the Neuron host pool is layer-first, one host region per device KV tensor (2 x layers regions; ``page_bytes[r]``
is a page of region r). The lease and the arena are page-major by default: a slot (an arena row) is one row holding
the page of every region, so a run of k pages consecutive in arena rows and in slots is one descriptor per direction.
The copy between the host pages and the arena changes the layout. Region-major (region r is ``[slots][page bytes r]``)
makes a run 2 x layers descriptors (72 for GPT-OSS 120B), whatever k is; on trn3 over LIBFABRIC that reads 256
scattered pages on one rank much more slowly than page-major reads them (M3).

Every remote transfer goes through the arena, never straight to SGLang's host pages: a transfer that times out may
still land later, and only the backend's own rows can be held back for that (quarantined) while SGLang reuses its
pages.

Stores: `HostStore` keeps the lease in this process (CPU tests, with fault injection); `NixlStore` reaches a lease
server's region over NIXL (LIBFABRIC off loopback).
"""

from __future__ import annotations

import contextlib
import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from sglang_neuron.mem_cache.remote_dram import fill as fill_mod
from sglang_neuron.mem_cache.remote_dram.index import RunAllocator
from sglang_neuron.mem_cache.remote_dram.lease import SLOT_ALIGN, Lease

logger = logging.getLogger(__name__)

#: A run of a transfer: ``(first slot, first arena row, pages)``.
Run = Tuple[int, int, int]


def _align(n: int, a: int = SLOT_ALIGN) -> int:
    return (n + a - 1) // a * a


class StoreDown(RuntimeError):
    """The store refuses transfers now (its connection broke and no reconnect is due, or it gave up)."""


class TransferTimeout(TimeoutError):
    """A transfer passed its deadline; its arena rows stay with the store until the transfer ends."""


@dataclass(frozen=True)
class Layout:
    """Placement of ``n`` pages of the host pool's regions (``page_bytes``) in one buffer.

    Page-major (the default): page i is one 4 KiB-aligned row ``[stride]`` holding every region's page bytes, so a
    run of k pages is one descriptor. Region-major: region r is ``[n][page_bytes[r]]``, 4 KiB aligned, so a run is
    one descriptor per region.
    """
    page_bytes: Tuple[int, ...]
    page_major: bool = True

    @property
    def slot_bytes(self) -> int:
        return sum(self.page_bytes)

    @property
    def stride(self) -> int:
        """Bytes per page-major row."""
        return _align(self.slot_bytes)

    def region_offsets(self, n: int) -> List[int]:
        """Region-major: each region's offset in a buffer of ``n`` pages."""
        out, off = [], 0
        for pb in self.page_bytes:
            out.append(off)
            off += _align(n * pb)
        return out

    def nbytes(self, n: int) -> int:
        if self.page_major:
            return n * self.stride
        return sum(_align(n * pb) for pb in self.page_bytes)

    def views(self, whole: np.ndarray, n: int) -> List[np.ndarray]:
        """``[n][page_bytes[r]]`` uint8 views of each region in ``whole`` (a buffer of ``n`` pages)."""
        if self.page_major:
            rows = whole[:n * self.stride].reshape(n, self.stride)
            offs = np.cumsum((0,) + self.page_bytes[:-1])
            return [rows[:, off:off + pb] for off, pb in zip(offs.tolist(), self.page_bytes)]
        return [whole[off:off + n * pb].reshape(n, pb) for off, pb in zip(self.region_offsets(n), self.page_bytes)]

    def spans(self, base: int, n: int, first: int, count: int) -> List[Tuple[int, int]]:
        """``(addr, nbytes)`` descriptors of pages ``[first, first + count)`` of a buffer of ``n`` pages at ``base``."""
        if self.page_major:
            return [(base + first * self.stride, count * self.stride)]
        return [(base + off + first * pb, count * pb) for off, pb in zip(self.region_offsets(n), self.page_bytes)]


def runs(slots: Sequence[int], rows: Sequence[int]) -> List[Run]:
    """Maximal runs of pairs consecutive in both slots and rows, in order.

    >>> runs([4, 5, 6, 9], [0, 1, 2, 3])
    [(4, 0, 3), (9, 3, 1)]
    """
    out: List[List[int]] = []
    for s, r in zip(slots, rows):
        if out and out[-1][0] + out[-1][2] == s and out[-1][1] + out[-1][2] == r:
            out[-1][2] += 1
        else:
            out.append([s, r, 1])
    return [tuple(x) for x in out]


class Arena:
    """The rank's staging rows, region-major (`Layout`), in one mapping a NIXL agent registers once.

    ``take`` hands out rows in runs; ``give`` returns them; ``quarantine`` keeps rows a late DMA may still write
    out of use for good.
    """

    def __init__(self, layout: Layout, rows: int):
        self.layout = layout
        self.rows = rows
        self.nbytes = layout.nbytes(rows)
        self.chunk = fill_mod.Chunk(_align(self.nbytes, fill_mod.HUGE))
        self.addr = self.chunk.addr
        self._whole = np.frombuffer(self.chunk.view(), dtype=np.uint8)
        self.regions = layout.views(self._whole, rows)
        self._free = RunAllocator(rows)
        self._cv = threading.Condition()
        self.quarantined = 0

    def take(self, n: int, timeout: float = 1.0) -> Optional[List[int]]:
        """``n`` rows (consecutive where possible), waiting up to ``timeout`` s; None if they do not free up."""
        if n > self.rows:
            return None
        deadline = time.monotonic() + timeout
        with self._cv:
            while self._free.free < n:
                left = deadline - time.monotonic()
                if left <= 0 or not self._cv.wait(left):
                    if self._free.free < n:
                        return None
            return self._free.take(n)

    def give(self, rows: Sequence[int]) -> None:
        with self._cv:
            self._free.give(rows)
            self._cv.notify_all()

    def quarantine(self, rows: Sequence[int]) -> None:
        with self._cv:
            self.quarantined += len(rows)
        logger.warning("remote DRAM: %d staging rows quarantined for good (%d in all): a late transfer may still "
                       "land in them", len(rows), self.quarantined)

    def spans(self, row: int, count: int) -> List[Tuple[int, int]]:
        return self.layout.spans(self.addr, self.rows, row, count)

    def page_parts(self, row: int) -> List[np.ndarray]:
        """Row ``row``'s page bytes in region order, as few contiguous arrays as the layout allows (one when
        page-major), for its digest."""
        if self.layout.page_major:
            start = row * self.layout.stride
            return [self._whole[start:start + self.layout.slot_bytes]]
        return [r[row] for r in self.regions]

    def close(self) -> None:
        self.regions = []
        self._whole = None
        self.chunk.close()


# -- stores -----------------------------------------------------------------------------------------------------------

class HostStore:
    """The lease in this process (CPU tests). ``faults`` injects failures: "write" / "read" raise, "late" times a
    read out (its rows stay held until `release_late`), "corrupt" flips a byte of the next read."""

    def __init__(self, layout: Layout, num_slots: int, arena: Arena):
        self.layout = layout
        self.num_slots = num_slots
        self.arena = arena
        self.chunk = fill_mod.Chunk(_align(layout.nbytes(num_slots), fill_mod.HUGE))
        self.regions = layout.views(np.frombuffer(self.chunk.view(), dtype=np.uint8), num_slots)
        self.faults: List[str] = []
        self.late: List[List[int]] = []
        self.writes = self.reads = 0
        self._lock = threading.Lock()

    def _fault(self, kind: str) -> bool:
        with self._lock:
            if kind in self.faults:
                self.faults.remove(kind)
                return True
        return False

    def write(self, transfer: Sequence[Run]) -> None:
        if self._fault("write") or self._fault("down"):
            raise RuntimeError("injected write failure")
        for slot, row, n in transfer:
            for remote, staging in zip(self.regions, self.arena.regions):
                remote[slot:slot + n] = staging[row:row + n]
        self.writes += 1

    def read(self, transfer: Sequence[Run], deadline_s: float) -> None:
        if self._fault("down"):
            raise StoreDown("injected: store down")
        if self._fault("read"):
            raise RuntimeError("injected read failure")
        if self._fault("late"):
            self.late.append([row + i for _, row, n in transfer for i in range(n)])
            raise TransferTimeout("injected: read missed its deadline")
        for slot, row, n in transfer:
            for remote, staging in zip(self.regions, self.arena.regions):
                staging[row:row + n] = remote[slot:slot + n]
        if self._fault("corrupt"):
            self.arena.regions[0][transfer[0][1], 0] ^= 0xFF
        self.reads += 1

    def release_late(self) -> None:
        for rows in self.late:
            self.arena.give(rows)
        self.late = []

    def close(self) -> None:
        self.regions = []
        self.chunk.close()


class NixlStore:
    """The lease in a lease server's region, over NIXL.

    The rank leases at start but connects at its first write: on trn3 a NIXL agent that registers Neuron HBM later on
    the node can break a connection made before it. A failed transfer breaks the connection; the next transfer after
    ``reconnect_gap_s`` makes a new agent (same key, same lease), at most ``max_reconnects`` times; meanwhile a
    transfer raises StoreDown at once. A read past its deadline keeps its rows until it ends (`reap_late`); past
    ``io_timeout_s`` its rows are quarantined.
    """

    POLL_S, MAX_POLL_S = 0.0002, 0.002

    def __init__(self, layout: Layout, num_slots: int, arena: Arena, lease_fn: Callable[[], Lease], name: str,
                 backend: str = "LIBFABRIC", io_timeout_s: float = 5.0, reconnect_gap_s: float = 5.0,
                 max_reconnects: int = 3, progress_delay_us: int = 250):
        self.layout = layout
        self.num_slots = num_slots
        self.arena = arena
        self.lease_fn = lease_fn
        self.lease = lease_fn()
        if self.lease.nbytes < layout.nbytes(num_slots):
            raise ValueError(f"lease of {self.lease.nbytes} B for {layout.nbytes(num_slots)} B of slots")
        self.name = name
        self.backend = backend
        self.io_timeout_s = io_timeout_s
        self.reconnect_gap_s = reconnect_gap_s
        self.max_reconnects = max_reconnects
        self.progress_delay_us = progress_delay_us
        # agent names are unique per process: the lease server's agent keeps the connection of every agent name it
        # has seen, and a restarted rank's agent of a name it knows got no handshake (on trn3 its first WRITE failed
        # with NIXL_ERR_REMOTE_DISCONNECT after 60 s)
        self._agent_tag = f"{os.getpid()}-{secrets.token_hex(3)}"
        self._conn: Optional[dict] = None
        self._old: List[dict] = []
        self._late: List[Tuple[dict, object, List[int], float]] = []
        self._lock = threading.Lock()
        self._broken_at: Optional[float] = None
        self._cause = ""
        self.connects = 0

    @property
    def down(self) -> bool:
        return self._broken_at is not None and self.connects > self.max_reconnects

    def _agent(self, agent_name: str):
        from nixl._api import nixl_agent, nixl_agent_config, nixl_thread_sync_t, nixlBind

        agent_cls = nixlBind.nixlAgent
        delay = self.progress_delay_us

        def with_delay(name, config):
            config.pthrDelay = delay
            return agent_cls(name, config)

        nixlBind.nixlAgent = with_delay
        try:
            return nixl_agent(agent_name, nixl_agent_config(backends=[self.backend], enable_listen_thread=False,
                                                            sync_mode=nixl_thread_sync_t.NIXL_THREAD_SYNC_RW))
        finally:
            nixlBind.nixlAgent = agent_cls

    def _connect(self) -> dict:
        if self.connects:
            lease = self.lease_fn()  # same key, same lease; new agent metadata if the server restarted its agent
            if lease.addr != self.lease.addr:
                raise RuntimeError(f"remote DRAM: lease {self.name} moved on reconnect")
            self.lease = lease
        agent = self._agent(f"remote-dram-{self.name}-{self._agent_tag}-c{self.connects}")
        reg = agent.get_reg_descs([(self.arena.addr, self.arena.chunk.nbytes, 0, "")], "DRAM")
        agent.register_memory(reg, backends=[self.backend])
        remote = agent.add_remote_agent(self.lease.metadata)
        return {"agent": agent, "reg": reg, "remote": remote}

    def _take_conn(self) -> dict:
        with self._lock:
            if self._conn is not None and self._broken_at is None:
                return self._conn
            if self.down:
                raise StoreDown(f"remote DRAM {self.name}: down ({self._cause})")
            if self._broken_at is not None and time.monotonic() - self._broken_at < self.reconnect_gap_s:
                raise StoreDown(f"remote DRAM {self.name}: broken ({self._cause}), reconnect not due")
            if self._conn is not None:
                self._old.append(self._conn)
                self._conn = None
            try:
                conn = self._connect()
            except Exception as exc:  # noqa: BLE001 - retried after the gap
                self._broken_at = time.monotonic()
                self._cause = f"connect: {type(exc).__name__}: {exc}"[:200]
                self.connects += 1
                raise StoreDown(self._cause) from exc
            self._conn = conn
            self._broken_at = None
            self.connects += 1
            logger.info("remote DRAM %s: connected to %s (lease %d B at %#x)", self.name, self.lease.server,
                        self.lease.nbytes, self.lease.addr)
            return conn

    def _break(self, conn: dict, cause: str) -> None:
        with self._lock:
            if self._conn is conn and self._broken_at is None:
                self._broken_at = time.monotonic()
                self._cause = cause[:200]
                logger.error("remote DRAM %s: connection to %s broke: %s", self.name, self.lease.server, cause)

    def _descs(self, agent, transfer: Sequence[Run]):
        local, far = [], []
        for slot, row, n in transfer:
            for (laddr, size), (raddr, _) in zip(self.arena.spans(row, n),
                                                 self.layout.spans(self.lease.addr, self.num_slots, slot, n)):
                local.append((laddr, size, 0))
                far.append((raddr, size, self.lease.dev))
        return (agent.get_xfer_descs(np.array(local, dtype=np.int64), "DRAM"),
                agent.get_xfer_descs(np.array(far, dtype=np.int64), "DRAM"))

    def _xfer(self, op: str, transfer: Sequence[Run], deadline_s: Optional[float]) -> None:
        conn = self._take_conn()
        agent = conn["agent"]
        try:
            local, far = self._descs(agent, transfer)
            handle = agent.initialize_xfer(op, local, far, conn["remote"])
            state = agent.transfer(handle)
        except Exception as exc:
            self._break(conn, f"{op} post: {type(exc).__name__}: {exc}")
            raise
        t0 = time.monotonic()
        limit = self.io_timeout_s if deadline_s is None else min(deadline_s, self.io_timeout_s)
        poll = self.POLL_S
        while state == "PROC":
            if time.monotonic() - t0 > limit:
                rows = [row + i for _, row, n in transfer for i in range(n)]
                if limit < self.io_timeout_s:
                    with self._lock:
                        self._late.append((conn, handle, rows, t0))
                    raise TransferTimeout(f"remote DRAM {op} of {len(rows)} pages missed {limit * 1e3:.0f} ms")
                self._break(conn, f"{op} timed out")
                self.arena.quarantine(rows)
                raise TransferTimeout(f"remote DRAM {op} of {len(rows)} pages timed out")
            time.sleep(poll)
            poll = min(2 * poll, self.MAX_POLL_S)
            state = agent.check_xfer_state(handle)
        with contextlib.suppress(Exception):
            agent.release_xfer_handle(handle)
        if state != "DONE":
            self._break(conn, f"{op} ended {state}")
            raise RuntimeError(f"remote DRAM {op} ended {state}")

    def write(self, transfer: Sequence[Run]) -> None:
        self._xfer("WRITE", transfer, None)

    def read(self, transfer: Sequence[Run], deadline_s: float) -> None:
        self.reap_late()
        self._xfer("READ", transfer, deadline_s)

    def reap_late(self) -> int:
        """Return the rows of late reads that ended; quarantine those past ``io_timeout_s``. Returns the rows held."""
        with self._lock:
            late, self._late = self._late, []
        held, keep = 0, []
        for conn, handle, rows, t0 in late:
            try:
                state = conn["agent"].check_xfer_state(handle)
            except Exception:  # noqa: BLE001 - an agent error ends the read
                state = "ERR"
            if state == "PROC":
                if time.monotonic() - t0 <= self.io_timeout_s:
                    keep.append((conn, handle, rows, t0))
                    held += len(rows)
                    continue
                self._break(conn, "late READ timed out")
                self.arena.quarantine(rows)
                continue
            with contextlib.suppress(Exception):
                conn["agent"].release_xfer_handle(handle)
            if state != "DONE":
                self._break(conn, f"late READ ended {state}")
            self.arena.give(rows)
        with self._lock:
            self._late.extend(keep)
        return held

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.reap_late()
        with self._lock:
            conns = [c for c in [self._conn, *self._old] if c is not None]
            self._conn, self._old = None, []
        for conn in conns:
            agent = conn["agent"]
            for release in (lambda: agent.remove_remote_agent(conn["remote"]),
                            lambda: agent.deregister_memory(conn["reg"])):
                with contextlib.suppress(Exception):
                    release()
