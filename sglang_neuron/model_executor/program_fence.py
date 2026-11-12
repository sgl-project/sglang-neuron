# SPDX-License-Identifier: Apache-2.0
"""The program-switch fence.

Different compiled programs share the runtime's collective buffers and semaphores. If one rank starts a new program
while a slower peer is still running the previous one, it corrupts the peer's logits and KV. The runtime's own
per-execution barrier would prevent this, but it only covers synchronous submission and torch-neuronx submits
asynchronously, so the plugin runs with it off (runtime_init.runtime_defaults) and fences the ranks itself.

Only a program switch needs the fence: an aggregated server alternating prefill and decode, or a runtime with several
compiled shapes (prefill segment tiles, decode batch buckets). A process with one program never fences, and neither
does steady decode. When a rank's next model execution runs a different NEFF than its last one, the fence, from the
pre-execute hook right before the submit:

1. waits for this rank's submitted executions to finish on the device (device_io.wait_launches: the newest launch
   event, since the graphs run in launch order on the default stream);
2. meets every rank of the TP group on the host (rank_meeting: the node's shared memory, each node's and then the
   node leaders' on a server over several nodes, or a gloo all-reduce; never the device collective buffers).

After the meeting every rank has finished the old program, so none can start the new one early.

A switch leaves the device idle until the new program starts, so the fence keeps everything else off that path: it
orders the step's uploads before the meeting (device_io.wait_uploads), so only the submit remains after it; it runs
ahead of the other pre-execute hooks, so on a PP stage it overlaps the wait for the previous stage's boundary; and both
of its waits poll for up to SPIN_S before blocking, since a blocked thread wakes through a busy host's run queue.

Under PP each stage has its own TP group (SGLang builds one per stage), so a stage meets only its own ranks. A
pipelined aggregated server's stages switch between prefill and decode at different moments, as their micro-batches
arrive, so a server-wide meeting would deadlock: a stage would wait for a peer that is waiting for its boundary.

The NEFF's identity is its cache key, which the executor binding cache hands over once the inputs are prepared
(backend.executor_cache.submitting), so the fence never walks the call's arguments. Only model executions count (the
compiled graphs wrapped by ``model_call``). Every rank runs the same NEFF sequence under the same keys (SPMD), and the
meeting all-reduces the new key, so ranks that would switch to different programs fail the step instead of running on
each other's collectives. NEFFs outside a model call carry no collectives on an aggregated server and don't move the
fence.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_LOG_FIRST = 16  # fences logged one by one; after these, every _LOG_EVERY-th
_LOG_EVERY = 100
# the fence's waits (the device sync, the shared-memory meeting) poll for up to this long before blocking: a meeting's
# ranks arrive close together, inside the spin, and an old program that runs longer is waited for by blocking
SPIN_S = 2e-3


def enabled() -> bool:
    from sglang_neuron import serving_policy

    return serving_policy.current().program_switch_fence


@functools.lru_cache(maxsize=256)
def neff_key(neff_id: str) -> int:
    """A 63-bit integer of ``neff_id``: the value every rank must agree on at a fence (cached: a process has a few
    NEFFs, and a switch should not hash)."""
    return int.from_bytes(hashlib.sha256(neff_id.encode()).digest()[:8], "little") & ((1 << 63) - 1)


class ProgramSwitchFence:
    """Fences this rank's model executions at program switches.

    ``sync()`` returns once every execution this rank submitted has finished on its device (None:
    await_newest_execution, the rank's launch events); ``meet(key)`` returns once every rank of the group has called it,
    and raises if their keys differ."""

    def __init__(self, sync: Optional[Callable[[], None]], meet: Callable[[int], None]):
        self._sync = sync if sync is not None else self.await_newest_execution
        self._meet = meet
        self._local = threading.local()
        self._last: Optional[str] = None
        self.fences = 0
        self.fence_seconds = 0.0

    @property
    def last_neff(self) -> Optional[str]:
        return self._last

    def model_call(self, fn: Callable) -> Callable:
        """``fn`` (a runtime's compiled graph) with its executions counted as model executions: the call runs inside
        device_io.pre_execute_scope, so the fence's hook runs before every graph the call submits, warm-up calls
        included."""
        from sglang_neuron.backend import device_io, executor_cache

        local = self._local

        def call(*args, **kwargs):
            depth = getattr(local, "depth", 0)
            local.depth = depth + 1
            try:
                executor_cache.submitting()  # a key no submit took (an executor called outside a model call) is stale
                with device_io.pre_execute_scope():
                    return fn(*args, **kwargs)
            finally:
                local.depth = depth

        call.__wrapped__ = fn
        call.program_switch_fence = self
        return call

    def before_execute(self, neff_id: str) -> None:
        """The pre-execute hook's body: runs right before an execution of ``neff_id`` is submitted."""
        if not getattr(self._local, "depth", 0):
            return
        if not neff_id:
            # without an identity every program reads as the last one and no switch would ever fence
            raise RuntimeError("the runtime reported no NEFF id for a model execution, so the program-switch fence cannot "
                               "tell its programs apart")
        previous = self._last
        if previous is not None and neff_id != previous:
            self._fence(previous, neff_id)
        self._last = neff_id

    def _fence(self, previous: str, neff_id: str) -> None:
        from sglang_neuron.backend import device_io
        from sglang_neuron.model_executor import host_trace

        t0, w0 = time.perf_counter(), time.time_ns()
        self._sync()
        t1 = time.perf_counter()
        # the step's upload ordering now, so after the meeting only the submit itself remains
        device_io.wait_uploads()
        self._meet(neff_key(neff_id))
        t2 = time.perf_counter()
        self.fences += 1
        self.fence_seconds += t2 - t0
        tr = host_trace.trace()
        if tr is not None:
            tr.add("fence", w0, time.time_ns(), f"{previous[:16]}->{neff_id[:16]}")
        if self.fences <= _LOG_FIRST or self.fences % _LOG_EVERY == 0:
            logger.info("[neuron] program switch fence %d: %s -> %s, %.2f ms (device sync %.2f ms, rank meeting "
                        "%.2f ms); %.1f ms over %d fences", self.fences, previous[:16], neff_id[:16],
                        (t2 - t0) * 1e3, (t1 - t0) * 1e3, (t2 - t1) * 1e3, self.fence_seconds * 1e3, self.fences)

    def await_newest_execution(self) -> None:
        """The default sync: until this rank's newest launch, and with it every launch before, has finished on the
        device (device_io.wait_launches, polled for up to SPIN_S first)."""
        from sglang_neuron.backend import device_io

        device_io.wait_launches(spin_s=SPIN_S)


def _check_agreement(key: int, largest: int, smallest: int) -> None:
    if largest != key or smallest != key:
        raise RuntimeError(f"program switch fence: the ranks switch to different programs (this rank's key {key}, "
                           f"the group's largest {largest} and smallest {smallest}): their model executions are out of "
                           "step")


def gloo_meeting(group) -> Callable[[int], None]:
    """``meet`` over ``group`` (a gloo group of every rank of the TP group): one all-reduce of [key, -key] with MAX,
    which returns [the largest key, -the smallest key], so every rank learns whether all of them agree."""
    import torch
    import torch.distributed as dist

    def meet(key: int) -> None:
        t = torch.tensor([key, -key], dtype=torch.int64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
        _check_agreement(key, int(t[0]), -int(t[1]))

    return meet


def shm_meeting(group, nnodes: int = 1) -> Callable[[int], None]:
    """``meet`` over ``group`` through shared memory (NodeShmMin): every rank posts [key, -key] and the leader publishes
    the reduction. A gloo all-reduce is a TCP ring, a network round trip per rank; this reduce pays none. A group over
    ``nnodes`` nodes meets through each node's segment and a gloo all-reduce of the node leaders
    (node_shm_min.group_reduce).

    The segment is built at the first meeting, because its name is agreed on with a collective and every rank reaches
    the first program switch at the same point."""
    state = {}

    def meet(key: int) -> None:
        if not state:
            import numpy as np

            from sglang_neuron.distributed.node_shm_min import group_reduce

            state["shm"] = group_reduce(group, nnodes, "fence", capacity=2, dtype=np.int64, op="max", spin_s=SPIN_S)
        largest, neg_smallest = state["shm"].reduce([key, -key])
        _check_agreement(key, int(largest), -int(neg_smallest))

    return meet


def rank_meeting(group, nnodes: Optional[int] = None):
    """(meet, how) for the TP group's gloo ``group``: shm_meeting where the node supports it and the group lies in one
    node or is the whole world, else gloo_meeting. ``nnodes`` defaults to the server's (serving_policy.nnodes)."""
    from sglang_neuron.distributed.node_shm_min import shm_supported, within_one_node

    if nnodes is None:
        from sglang_neuron import serving_policy

        nnodes = serving_policy.current().nnodes
    if shm_supported() and nnodes <= 1:
        return shm_meeting(group), "through the node's shared memory"
    if shm_supported():
        import torch.distributed as dist

        ranks = dist.get_process_group_ranks(group)
        world = dist.get_world_size()
        if within_one_node(ranks, world, nnodes):
            return shm_meeting(group), "through the node's shared memory"
        if sorted(ranks) == list(range(world)):
            return shm_meeting(group, nnodes), (f"through each node's shared memory and the {nnodes} node leaders' "
                                                "gloo all-reduce")
    return gloo_meeting(group), "on a gloo all-reduce over the TP group"


_FENCE: Optional[ProgramSwitchFence] = None
_LOCK = threading.Lock()


def install_hook(fence: ProgramSwitchFence) -> None:
    """Run ``fence`` from backend.device_io's pre-execute hook, ahead of the other hooks, keyed by the cache key of
    the NEFF being submitted (backend.executor_cache.submitting)."""
    from sglang_neuron.backend import device_io, executor_cache

    if device_io.on_device() and not executor_cache._installed:
        # every submit would find no key, and no switch would ever fence
        raise RuntimeError("the program-switch fence keys a model execution by the NEFF cache key the executor binding "
                           "cache hands to its submit (backend.executor_cache), which is not installed")

    def pre_execute():
        key = executor_cache.submitting()
        if key is not None:  # None: not a compiled graph's submit (an eager kernel's, which has no collectives)
            fence.before_execute(key)

    # ahead of the other hooks: the drain and the meeting overlap the PP seam's landing wait (module docstring)
    device_io.add_pre_execute(pre_execute, first=True)


def ensure_installed(group_fn: Optional[Callable] = None, programs: int = 1) -> Optional[ProgramSwitchFence]:
    """The process's fence, installed by the first call that needs one: an aggregated server (prefill and decode
    alternate on one device) or a runtime with ``programs`` > 1 compiled shapes. Returns None for a disaggregated role
    with a single program. ``group_fn`` returns the TP group's gloo group (default: SGLang's TP cpu_group)."""
    global _FENCE
    if not enabled() and programs <= 1:
        return None
    with _LOCK:
        if _FENCE is None:
            if group_fn is None:
                from sglang.srt.distributed import parallel_state as sgl_ps

                group_fn = lambda: sgl_ps.get_tp_group().cpu_group  # noqa: E731
            meet, how = rank_meeting(group_fn())
            fence = ProgramSwitchFence(None, meet)
            install_hook(fence)
            _FENCE = fence
            logger.info("[neuron] program switch fence installed: the ranks meet %s before a model execution whose "
                        "NEFF differs from the last one", how)
        return _FENCE


def wrap_compiled(compiled: Callable, programs: int = 1) -> Callable:
    """A runtime's compiled graph of ``programs`` shapes, its executions fenced at program switches on an aggregated
    server and wherever the runtime has more than one shape."""
    fence = ensure_installed(programs=programs)
    return compiled if fence is None else fence.model_call(compiled)
