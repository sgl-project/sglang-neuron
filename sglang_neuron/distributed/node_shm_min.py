# SPDX-License-Identifier: Apache-2.0
"""The MIN (or MAX) of a short integer vector over the ranks of one node, through a shared-memory segment (x86-64,
Python 3.13 or later). Used by the poll reduces in poll_reduce.py, dp_identity's fleet rendezvous and
program_fence's rank meeting. A group over several nodes reduces through each node's segment, then a gloo all-reduce
of the node leaders, whose result each leader publishes to its node (group_reduce).

A gloo all_reduce plus broadcast over a node's ranks is a TCP ring through the loopback. Each rank adds a socket round
trip to the critical path, and the ring releases the ranks one after another. Here every rank writes its row and then
publishes the round number; the node leader reduces the rows and publishes the result for the others to read.

Each call is one round, and every rank of the node makes the calls in the same order. A rank rewrites its row only
after reading the previous result, and the leader publishes a result only after reading every row, so one segment
serves every round. x86-64 keeps stores and loads in program order, so a row is complete once its round number is
visible. Other architectures would need memory fences, which this module does not issue, so they are refused. A
vector longer than the segment runs as several rounds. Every rank also posts the lengths of its row and of its whole
vector, and the leader refuses a round whose lengths differ (a rank that polled another queue). The leader publishes
the refusal, so no rank waits out the timeout.

The segment's name exists only while the ranks attach. The leader creates it, every rank attaches between two node
barriers, and the leader then unlinks the name. Each mapping outlives the name, and Linux frees the memory with the
last mapping, so nothing is left in /dev/shm however the processes end. Ranks map the segment with
``SharedMemory(track=False)``, so that Python's multiprocessing resource tracker, which deletes a segment when the
process that created it exits, leaves it alone. Every rank posts its pid, and a waiting
rank checks about once a second that the ranks it waits for still exist. A rank killed mid-collective then fails the
round within a second rather than after ``timeout_s``.

Layout: int64 seq[local_size], counts[local_size] (this round's length), totals[local_size] (the whole vector's
length), pids[local_size], out_seq, out_n, then T rows[local_size][capacity] and T out[capacity], where T is the
element type (uint8 by default, int64 for counters). Every array starts 8-byte aligned."""
from __future__ import annotations

import os
import platform
import sys
import threading
import time

import numpy as np

# A wait busy-polls SPIN times, then sleeps NAP_S, and repeats. The busy polls hold the GIL and the sleep releases it.
# The scheduler process runs Python threads beside the event loop (on a PD decode, the device-output copy thread and
# the KV receiver's heartbeat and listener threads), and a rank that waits here every iteration must let them run. A
# spin of a few thousand polls would hold the GIL for milliseconds at a time. At 64, the common case (all ranks
# arriving together) still ends within the spin, and otherwise the GIL is released after a short hold.
SPIN = 64
LIVENESS_S = 1.0  # how often a waiting rank checks that the ranks it waits for are still alive
NAP_S = 20e-6
# A nap can overrun its deadline by the thread's timer slack. The kernel's default of 50 us would stretch each 20 us
# nap to up to 70 us, so a waiting thread sets a 1 us slack before its first nap.
TIMER_SLACK_NS = 1000
_PR_SET_TIMERSLACK = 29
_slack = threading.local()


def set_timer_slack(ns: int = TIMER_SLACK_NS) -> None:
    """PR_SET_TIMERSLACK on the calling thread, so its short timed sleeps end within ``ns`` of their deadline instead
    of the kernel's default 50 us."""
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    ulong = ctypes.c_ulong
    if libc.prctl(_PR_SET_TIMERSLACK, ulong(ns), ulong(0), ulong(0), ulong(0)) != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"prctl(PR_SET_TIMERSLACK, {ns}): {os.strerror(err)}")
    _slack.ns = ns


class NodeShmMin:
    def __init__(self, name: str, local_rank: int, local_size: int, barrier, capacity: int = 4096,
                 timeout_s: float = 1800.0, dtype=np.uint8, op: str = "min", spin_s: float = 0.0):
        """All ranks of the node construct it together. ``barrier()`` is a node-local barrier (one gloo barrier on the
        node group), called twice: the first lets every rank attach to the segment the leader created, and the second
        tells the leader that every rank has attached, so it can unlink the name. The checks below fail on every rank
        alike, before any barrier. ``op`` is "min" or "max" over the ranks' rows, whose elements have type ``dtype``.
        ``spin_s``: each wait first polls without sleeping for up to that long (backend.device_io.spin_until), for a
        rare wait on the device's critical path such as the program-switch fence; 0, for a wait every iteration,
        starts the SPIN / NAP_S loop at once."""
        from multiprocessing import shared_memory

        if platform.machine() != "x86_64":
            raise RuntimeError(f"NodeShmMin relies on x86-64 store / load ordering, not {platform.machine()}")
        if sys.version_info < (3, 13):
            raise RuntimeError("NodeShmMin needs Python >= 3.13 for "
                               "SharedMemory(track=False): the segment's name is unlinked once every rank has "
                               "attached, and no resource tracker may own it. This is Python "
                               "%d.%d" % sys.version_info[:2])
        if op not in ("min", "max"):
            raise ValueError(f"NodeShmMin: op must be 'min' or 'max', not {op!r}")
        self.local_rank, self.local_size, self.capacity, self.timeout_s = local_rank, local_size, capacity, timeout_s
        self.spin_s = float(spin_s)
        self.dtype, self.op = np.dtype(dtype), op
        item = self.dtype.itemsize
        rows_bytes = -(-local_size * capacity * item // 8) * 8  # keeps `out` 8-byte aligned
        head = 8 * (4 * local_size + 2)
        size = head + rows_bytes + capacity * item
        if local_rank == 0:
            self._shm = shared_memory.SharedMemory(name=name, create=True, size=size, track=False)
            try:
                self._shm.buf[:size] = b"\0" * size
                self._map(head, rows_bytes)
                barrier()  # the others attach
                barrier()  # every rank has attached and posted its pid
            finally:
                self._shm.unlink()
        else:
            barrier()
            self._shm = shared_memory.SharedMemory(name=name, track=False)
            self._map(head, rows_bytes)
            barrier()
        self.round = 0

    def _map(self, head: int, rows_bytes: int) -> None:
        """Map the segment's arrays (layout in the module docstring) and post this rank's pid for the others' liveness
        checks."""
        L, buf = self.local_size, self._shm.buf
        self.seq = np.ndarray((L,), dtype=np.int64, buffer=buf, offset=0)
        self.counts = np.ndarray((L,), dtype=np.int64, buffer=buf, offset=8 * L)
        self.totals = np.ndarray((L,), dtype=np.int64, buffer=buf, offset=16 * L)
        self.pids = np.ndarray((L,), dtype=np.int64, buffer=buf, offset=24 * L)
        self.out_seq = np.ndarray((1,), dtype=np.int64, buffer=buf, offset=32 * L)
        self.out_n = np.ndarray((1,), dtype=np.int64, buffer=buf, offset=32 * L + 8)
        self.rows = np.ndarray((L, self.capacity), dtype=self.dtype, buffer=buf, offset=head)
        self.out = np.ndarray((self.capacity,), dtype=self.dtype, buffer=buf, offset=head + rows_bytes)
        self.pids[self.local_rank] = os.getpid()

    def _wait(self, done, awaited):
        """Spin until ``done()``. Every LIVENESS_S, check that the ranks ``awaited()`` returns (those this rank still
        waits for) are alive. If a node rank died (SIGKILL, OOM), this raises within a second instead of after
        timeout_s, as a gloo collective would raise on the dead rank's closed socket. With ``spin_s`` the wait polls
        without sleeping first; the liveness and timeout checks start after it."""
        if self.spin_s > 0:
            from sglang_neuron.backend.device_io import spin_until

            if spin_until(done, self.spin_s):
                return
        deadline = checked = None
        n = 0
        while not done():
            n += 1
            if n % SPIN == 0:
                now = time.monotonic()
                deadline = deadline or now + self.timeout_s
                checked = checked or now
                if now - checked >= LIVENESS_S:
                    checked = now
                    for r in awaited():
                        if not _alive(int(self.pids[r])):
                            raise RuntimeError(f"NodeShmMin round {self.round + 1}: node rank {r} "
                                               f"(pid {int(self.pids[r])}) has exited")
                if now > deadline:
                    raise TimeoutError(f"NodeShmMin round {self.round + 1}: a node rank did not arrive in "
                                       f"{self.timeout_s:.0f} s")
                if getattr(_slack, "ns", None) != TIMER_SLACK_NS:
                    set_timer_slack()
                time.sleep(NAP_S)

    def reduce(self, values, leader_fn=None) -> list:
        """The node MIN (or MAX) of every rank's ``values``, which must have the same length on every rank. A vector
        longer than the capacity is reduced in consecutive rounds of ``capacity`` values. On the node leader,
        ``leader_fn`` (numpy vector -> numpy vector) turns each round's node result into the published one: the reduce
        over the node leaders of a group that spans nodes (leader_all_reduce)."""
        values = np.asarray(values, dtype=self.dtype)
        total = len(values)
        if total <= self.capacity:
            return self._round(values, total, leader_fn)
        out = []
        for i in range(0, total, self.capacity):
            out += self._round(values[i : i + self.capacity], total, leader_fn)
        return out

    def _publish(self, g: int, n: int) -> None:
        self.out_n[0] = n
        self.out_seq[0] = g  # after the result and its length (x86-64 store order)
        self.round = g

    def _round(self, values, total: int, leader_fn=None) -> list:
        n, g = len(values), self.round + 1
        self.rows[self.local_rank, :n] = values
        self.counts[self.local_rank] = n
        self.totals[self.local_rank] = total
        self.seq[self.local_rank] = g  # published after the row and its lengths (x86-64 store order)
        if self.local_rank == 0:
            try:
                self._wait(lambda: bool((self.seq == g).all()), lambda: np.flatnonzero(self.seq != g).tolist())
            except RuntimeError:
                self._publish(g, _RANK_DIED)  # the live ranks fail now too, instead of waiting on the leader
                raise
            if not bool((self.counts == n).all()) or not bool((self.totals == total).all()):
                self._publish(g, _REFUSED)
                raise ValueError(f"NodeShmMin round {g}: the node's ranks posted vectors of different lengths "
                                 f"{sorted(set(self.totals.tolist()))} (a poll reduce is a collective: every rank must "
                                 "reduce the same queue)")
            rows = self.rows[:, :n]
            m = rows.min(axis=0) if self.op == "min" else rows.max(axis=0)
            if leader_fn is not None:
                try:
                    m = np.asarray(leader_fn(m), dtype=self.dtype)
                except BaseException:
                    self._publish(g, _LEADER_FAILED)  # the node's ranks fail now too, instead of waiting on the leader
                    raise
            self.out[:n] = m
            self._publish(g, n)
            return m.tolist()
        self._wait(lambda: int(self.out_seq[0]) == g, lambda: [0])
        got = int(self.out_n[0])
        if got != n:
            self.round = g
            if got == _RANK_DIED:
                raise RuntimeError(f"NodeShmMin round {g}: a node rank has exited (seen by the node leader)")
            if got == _LEADER_FAILED:
                raise RuntimeError(f"NodeShmMin round {g}: the node leader's reduce over the other nodes failed")
            raise ValueError(f"NodeShmMin round {g}: the node leader refused the round (vector lengths differ; this "
                             f"rank posted {total})")
        self.round = g
        return self.out[:n].tolist()


_REFUSED = -1  # out_n of a round the leader refused (the ranks' vector lengths differ)
_RANK_DIED = -3  # out_n of a round the leader abandoned: a node rank it waited for has exited
_LEADER_FAILED = -2  # out_n of a round whose cross-node stage (leader_fn) raised on the leader


def _alive(pid: int) -> bool:
    """Whether a node rank's process is still running. A killed rank stays a zombie until its parent reaps it, and a
    zombie still answers kill(pid, 0), so its /proc state is read too. A process of another user counts as alive: it
    can only be a reused pid, held by a process this one cannot signal."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        with open(f"/proc/{pid}/stat") as f:
            state = f.read().rsplit(")", 1)[1].split()[0]  # the field after the parenthesised command name
    except (OSError, IndexError):
        return False
    return state not in ("Z", "X")


def shm_supported() -> bool:
    """Whether this process can run a NodeShmMin: x86-64, Python >= 3.13 and a usable /dev/shm."""
    return (platform.machine() == "x86_64" and sys.version_info >= (3, 13)
            and os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK))


def group_segment_name(group, leader: int, group_key: str) -> str:
    """A segment name every rank of ``group`` agrees on. The group leader (global rank ``leader``) builds it and
    broadcasts it over ``group``: one gloo collective at construction, which every rank of the group reaches together.
    The name carries the launcher's pid, so two servers on one node never attach to each other's segment. It also
    carries the leader's own pid: a server with plain --dp-size N runs N TP worlds under one data-parallel controller,
    and their node groups have the same ranks in every world."""
    import torch.distributed as dist

    name = [f"sgn_pollmin_{os.getppid()}_{leader}_{group_key}_{os.getpid()}" if dist.get_rank() == leader else None]
    dist.broadcast_object_list(name, src=leader, group=group)
    return name[0]


def node_of(rank: int, world_size: int, nnodes: int) -> int:
    """The node of global rank ``rank`` in a world of ``world_size`` ranks over ``nnodes`` nodes. SGLang places a
    server's ranks node-major, ``world_size // nnodes`` per node (engine._calculate_rank_ranges without pipeline
    stages)."""
    return int(rank) // (int(world_size) // max(int(nnodes), 1))


def within_one_node(ranks, world_size: int, nnodes: int) -> bool:
    """Whether every rank of ``ranks`` runs on one node, where one shared-memory segment can reach them all."""
    return int(nnodes) <= 1 or len({node_of(r, world_size, nnodes) for r in ranks}) == 1


def leader_all_reduce(leader_group, op: str):
    """The cross-node stage of a reduce over nodes, as NodeShmMin.reduce's ``leader_fn``: the node result all-reduced
    with ``op`` ("min" or "max") over ``leader_group`` (the first rank of each node). None on a rank that leads no node
    (``leader_group`` None), whose reduce never runs the stage."""
    if op not in ("min", "max"):
        raise ValueError(f"leader_all_reduce: op {op!r} is not 'min' or 'max'")
    if leader_group is None:
        return None
    import torch
    import torch.distributed as dist

    reduce_op = dist.ReduceOp.MIN if op == "min" else dist.ReduceOp.MAX

    def leader_fn(node_result):
        t = torch.from_numpy(np.ascontiguousarray(node_result).copy())
        dist.all_reduce(t, op=reduce_op, group=leader_group)
        return t.numpy()

    return leader_fn


_NODE_SPLITS: dict = {}  # id(group) -> (node group, leader group or None)


def node_split(group, nnodes: int):
    """(this rank's node group, the node-leader group) of ``group``, a gloo group of the whole world over ``nnodes``
    nodes. The node group holds the ranks of this rank's node; the leader group holds the first rank of each node and is
    given to those ranks only (None elsewhere). torch's new_group is collective over the world, so the caller builds
    the split at a point every rank of the world reaches together, the first collective of ``group``; it is built once
    per group."""
    import torch.distributed as dist

    key = id(group)
    if key in _NODE_SPLITS:
        return _NODE_SPLITS[key]
    ranks = sorted(dist.get_process_group_ranks(group))
    world = dist.get_world_size()
    if ranks != list(range(world)) or world % int(nnodes):
        raise ValueError(f"a node split needs the whole world ({world} ranks) over {nnodes} nodes, not a group of "
                         f"{len(ranks)} ranks")
    nodes = [[r for r in ranks if node_of(r, world, nnodes) == n] for n in range(int(nnodes))]
    node_groups = [dist.new_group(ranks=n, backend="gloo") for n in nodes]
    leaders = dist.new_group(ranks=[n[0] for n in nodes], backend="gloo")
    me = dist.get_rank()
    mine = node_of(me, world, nnodes)
    _NODE_SPLITS[key] = (node_groups[mine], leaders if me == nodes[mine][0] else None)
    return _NODE_SPLITS[key]


class SpanningReduce:
    """A reduce over a group that spans nodes: each node's ranks reduce through their node's segment, the node leaders
    reduce the node results over their gloo group, and each leader publishes the result to its node. ``reduce`` has
    NodeShmMin.reduce's contract."""

    def __init__(self, segment: "NodeShmMin", leader_fn):
        self.segment, self.leader_fn = segment, leader_fn

    def reduce(self, values) -> list:
        return self.segment.reduce(values, leader_fn=self.leader_fn)


def group_reduce(group, nnodes: int, what: str, **kwargs):
    """The shared-memory reduce over ``group`` (an object with NodeShmMin's ``reduce``), built at the group's first
    collective, which every rank of the group reaches together: one node segment for a group inside one node, else a
    SpanningReduce over the whole world's node split (node_split). None for a group that spans nodes without being the
    whole world, which only a gloo all-reduce reaches. ``kwargs`` go to NodeShmMin."""
    import torch.distributed as dist

    ranks = dist.get_process_group_ranks(group)
    world = dist.get_world_size()
    if within_one_node(ranks, world, nnodes):
        name = group_segment_name(group, ranks[0], f"{what}{ranks[0]}x{len(ranks)}")
        return NodeShmMin(name, ranks.index(dist.get_rank()), len(ranks), lambda: dist.barrier(group=group), **kwargs)
    if sorted(ranks) != list(range(world)):
        return None
    node_group, leader_group = node_split(group, nnodes)
    node_ranks = dist.get_process_group_ranks(node_group)
    name = group_segment_name(node_group, node_ranks[0], f"{what}{ranks[0]}x{len(ranks)}")
    segment = NodeShmMin(name, node_ranks.index(dist.get_rank()), len(node_ranks),
                         lambda: dist.barrier(group=node_group), **kwargs)
    return SpanningReduce(segment, leader_all_reduce(leader_group, kwargs.get("op", "min")))
