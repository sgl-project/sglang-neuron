# SPDX-License-Identifier: Apache-2.0
"""The host<->device copies of the serving graphs' per-step inputs and outputs, ordered so the device runs the graphs
back to back.

On torch-neuronx every operation goes into a stream, and a stream starts a device operation on another
execution unit than the operation before it only once that operation has retired (torch-neuronx's scheduler gates it
on the host). A graph runs on the compute unit and a copy on the tensor-op unit, so one copy between two graphs on one
stream makes the second graph wait for the copy and the copy for the first graph: the device idles between every two
steps. torch-neuronx has one stream besides the default stream, so this module gives them fixed roles:

* the default stream carries the graphs;
* the side stream carries the host copies:
  - an upload (a step's host input) is enqueued on it before the step's graph launches, and the default stream waits
    for the uploads by one event right before the graph's submit; the upload runs while the graph before it is still
    running. The wait is part of the submit itself (``install`` wraps torch-neuronx's graph submit at backend init), so
    every graph, a warm-up or profiling launch too, runs after the uploads its thread enqueued before it, whichever
    call site launches it;
  - a readback (a launched graph's output) is enqueued on it only once its graph has finished (the graph's launch
    event, waited for on the host), so a readback never holds the side stream behind a running graph, where the next
    step's uploads would queue behind it. A small contiguous output (a step's sampled tokens) skips the stream: once
    the graph has finished, the host reads it from its runtime tensor (``_direct_read``).

Every uploaded tensor is also recorded on the default stream, so its memory is not reused before the graph that reads
it has run. A thread keeps at most SGLANG_NEURON_MAX_INFLIGHT_LAUNCHES launches in flight (2 when unset), the plugin's
in-flight bound: a launch past it first waits for the oldest one, which the scheduler's overlap loop has normally
finished with already. The side stream needs ``TORCH_NEURONX_ENABLE_STREAMS=1`` in the launch environment
(torch-neuronx reads it when torch imports it, and without it every stream is the default stream); ``check_streams``
refuses a process without it.
"""

from __future__ import annotations

import collections
import contextlib
import functools
import os
import threading
import time
from typing import Any, Callable, Optional

# per thread: the device index of uploads the next graph submit has not waited for yet, and the launches in flight
_STATE = threading.local()
_SIDE = {}  # device index -> torch-neuronx side stream
_DEFAULT = {}  # device index -> torch-neuronx default stream (a constant per device)


@functools.lru_cache(maxsize=1)
def on_device() -> bool:
    """Whether this process runs on the Neuron device (not ``SGLANG_NEURON_CPU_MODE=1``), read once: the copies run
    on every step. A CPU-mode process copies with plain ``.to()`` calls and launches no events."""
    from sglang_neuron.backend import envs

    return not envs.SGLANG_NEURON_CPU_MODE


# the hooks a fenced graph call runs right before the runtime submits it (add_pre_execute, pre_execute_scope)
_PRE_EXECUTE: list = []


def install() -> None:
    """Wrap torch-neuronx's graph submit (``torch_neuronx._C.execute_compiled_graph``) once per process; backend/init.py
    calls it. Every submit then makes the default stream wait for the uploads its thread enqueued since the last
    submit, right before the runtime gets the graph, and a submit inside ``pre_execute_scope`` first runs the
    ``add_pre_execute`` hooks.

    torch-neuronx's ``Executor.__call__`` calls ``execute_compiled_graph`` by module attribute; a build that bound it
    at import would never run the wrapper, and a graph would read its inputs before their uploads landed, so the
    executor's call protocol is checked against the pinned source first (backend/executor_cache), and another source
    is refused."""
    import torch_neuronx

    if getattr(torch_neuronx._C.execute_compiled_graph, "_sgn_submit", False):
        return
    from sglang_neuron.backend import executor_cache

    executor_cache.check_call_protocol()
    submit = torch_neuronx._C.execute_compiled_graph

    def execute_compiled_graph(*args, **kwargs):
        if getattr(_STATE, "pre_execute", False):
            for pre in _PRE_EXECUTE:
                pre()
        wait_uploads()  # after the hooks: the event covers every upload up to the submit
        return submit(*args, **kwargs)

    execute_compiled_graph._sgn_submit = True
    torch_neuronx._C.execute_compiled_graph = execute_compiled_graph


def add_pre_execute(hook, *, first: bool = False) -> None:
    """Run ``hook()`` right before torch-neuronx submits a graph execution inside a ``pre_execute_scope`` (the graph
    runner's forward, and every compiled graph the program-switch fence wraps).

    The hook runs after all the Python work of the call (dynamo's guards, the AOT wrapper, input preparation and output
    allocation), immediately before ``torch_neuronx._C.execute_compiled_graph``. That is where the program-switch fence
    meets the ranks. Executions outside a scope (eager copies, unfenced warm-up graphs) are not affected. Hooks run in
    the order they were added; a ``first`` hook runs ahead of the others."""
    install()
    if first:
        _PRE_EXECUTE.insert(0, hook)
    else:
        _PRE_EXECUTE.append(hook)


@contextlib.contextmanager
def pre_execute_scope():
    """The graph executions this thread submits inside run the add_pre_execute hooks first (a fenced model call)."""
    outer = getattr(_STATE, "pre_execute", False)
    _STATE.pre_execute = True
    try:
        yield
    finally:
        _STATE.pre_execute = outer


def check_streams() -> None:
    """Refuse a torch-neuronx process whose streams are aliased to the default stream (TORCH_NEURONX_ENABLE_STREAMS
    unset): every step's copies would then sit between its graphs. The launch bound is checked here too, so a bad one
    refuses the runtime's start instead of its first served step."""
    if not on_device():
        return
    _max_inflight()
    check_io_ring()
    from torch_neuronx.utils import is_streams_enabled

    if not is_streams_enabled():
        raise RuntimeError("TORCH_NEURONX_ENABLE_STREAMS is not 1 in the launch environment: torch-neuronx then runs "
                           "every copy on the graphs' stream, between two graphs; set TORCH_NEURONX_ENABLE_STREAMS=1 "
                           "(read when torch is imported)")


def _side(device_index: int):
    side = _SIDE.get(device_index)
    if side is None:
        import torch_neuronx

        side = _SIDE[device_index] = torch_neuronx.Stream(device=device_index)
    return side


def _default(device_index: int):
    default = _DEFAULT.get(device_index)
    if default is None:
        import torch_neuronx

        default = _DEFAULT[device_index] = torch_neuronx.default_stream(device_index)
    return default


def upload(host, dev):
    """``host`` on ``dev`` for the next graph launch on this thread."""
    return upload_many((host,), dev)[0]


def upload_many(hosts, dev) -> list:
    """``hosts`` on ``dev`` for the next graph launch on this thread, in order: one switch to the side stream for all
    of them (a step's host inputs), each recorded on the default stream.

    A caller may free or overwrite a host tensor as soon as this returns. For a ``non_blocking`` copy from a tensor
    its host allocator didn't allocate (any ordinary CPU tensor), torch-neuronx copies the bytes into the operation's
    own bounce buffer on this thread before the copy is queued (``copy_cpu_to_neuron`` in csrc/core/utils/
    CopyUtils.cpp, ``WriteDirectKernelExecution::AllocateBounceBuffer``). Only a tensor from its pinned host allocator
    would be read later; the platform reports no pinned memory to SGLang (``device.is_pin_memory_available``)."""
    if not on_device() or getattr(dev, "type", str(dev)) == "cpu":
        return [host.to(dev) for host in hosts]
    import torch_neuronx

    if not getattr(torch_neuronx._C.execute_compiled_graph, "_sgn_submit", False):
        raise RuntimeError("device_io.upload before device_io.install(): no graph submit would wait for the upload, "
                           "so a graph could read it before it lands")
    index = dev.index if dev.index is not None else torch_neuronx.current_device()
    default = _default(index)
    with torch_neuronx.stream(_side(index)):
        out = [host.to(dev, non_blocking=True) for host in hosts]
    for tensor in out:
        tensor.record_stream(default)
    _STATE.pending = index
    return out


class UploadRing:
    """Fixed device tensors one graph argument's host values go into, call after call: a small graph run many times
    with fresh small inputs (a pool's slot or page ids) reads the same few device addresses instead of a new tensor's
    each time, for which the runtime would build a new DMA ring (HBM a server sized to its pools does not keep free).

    ``put`` writes into the next buffer on the side stream like ``upload`` (asynchronous; the next submit waits for it),
    after the device finished the last graph that read that buffer (``used`` records it), so a write never races a
    pending read. On the host (CPU mode, tests) it returns the values on the device directly."""

    def __init__(self, shape, dtype, dev, size: int = 2):
        self.dev = dev
        self.slots = [None] * size if (on_device() and getattr(dev, "type", str(dev)) != "cpu") else None
        self.shape, self.dtype = tuple(shape), dtype
        self.events = [None] * size
        self.next = 0
        self.current = None

    def put(self, host):
        if self.slots is None:
            return host.to(self.dev)
        import torch
        import torch_neuronx

        if tuple(host.shape) != self.shape:
            raise ValueError(f"upload ring of {self.shape} given {tuple(host.shape)}")
        i = self.next
        self.next = (i + 1) % len(self.slots)
        index = self.dev.index if self.dev.index is not None else torch_neuronx.current_device()
        side = _side(index)
        with torch_neuronx.stream(side):
            if self.slots[i] is None:
                # allocated on the stream that writes it, as upload_many's outputs are: a buffer allocated on the
                # default stream is bound lazily there, and the side stream's copy could run before that allocation
                # ("Tensor not bound: lazy alloc kAlloc may not have executed" at a Kimi-K3 PP2 x TP16 warm-up)
                self.slots[i] = torch.empty(self.shape, dtype=self.dtype, device=self.dev)
            if self.events[i] is not None:
                side.wait_event(self.events[i])
            self.slots[i].copy_(host, non_blocking=True)
        self.slots[i].record_stream(_default(index))
        _STATE.pending = index
        self.current = (i, index)
        return self.slots[i]

    def used(self) -> None:
        """Right after the graph that read the last ``put`` was launched: its completion frees that buffer."""
        if self.slots is None or self.current is None:
            return
        i, index = self.current
        self.events[i] = _default(index).record_event()
        self.current = None


MAX_INFLIGHT = "SGLANG_NEURON_MAX_INFLIGHT_LAUNCHES"


@functools.lru_cache(maxsize=1)
def _max_inflight() -> int:
    """SGLANG_NEURON_MAX_INFLIGHT_LAUNCHES, the plugin's bound on in-flight launches (2 when unset); a value that is not
    an integer of 1 or more is refused naming it (the runtime's start checks it, check_streams)."""
    raw = os.environ.get(MAX_INFLIGHT, "2")
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        raise ValueError(f"{MAX_INFLIGHT}={raw!r}: set it to an integer of 1 or more")
    return value


@functools.lru_cache(maxsize=1)
def _io_ring() -> int:
    """NEURON_RT_IO_RING_CACHE_SIZE, the ring of IO tensor sets a program's executions cycle through
    (runtime_init.IO_RING_CACHE_SIZE when unset)."""
    from sglang_neuron.model_executor import runtime_init

    raw = os.environ.get("NEURON_RT_IO_RING_CACHE_SIZE", str(runtime_init.IO_RING_CACHE_SIZE))
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        raise ValueError(f"NEURON_RT_IO_RING_CACHE_SIZE={raw!r}: set it to an integer of 1 or more")
    return value


def check_io_ring() -> None:
    """Refuse an IO ring no deeper than the in-flight bound: a step's outputs are read up to the bound's launches of
    their program later (the decode's result lag, sglang_patches.pd_decode_result_lag), and a ring that shallow may by
    then hold another step's."""
    from sglang_neuron.model_executor import runtime_init

    ring, bound = _io_ring(), _max_inflight()
    if ring <= bound:
        raise ValueError(
            f"NEURON_RT_IO_RING_CACHE_SIZE={ring} is not above {MAX_INFLIGHT}={bound}: a "
            f"step's outputs are read on the host up to {bound} launches of their program later, by when a ring of "
            f"{ring} may hold another step's; set the ring above the bound (the default is "
            f"{runtime_init.IO_RING_CACHE_SIZE}) or lower the bound")


def _inflight() -> collections.deque:
    q = getattr(_STATE, "inflight", None)
    if q is None:
        q = _STATE.inflight = collections.deque()
    return q


def before_launch() -> None:
    """Right before a graph launch counted against the in-flight bound (``after_launch``): wait for this
    thread's oldest launch while the bound is reached. The launch's wait for its uploads is the submit's own
    (``install``)."""
    if not on_device():
        return
    inflight = _inflight()
    while len(inflight) >= _max_inflight():
        inflight.popleft().synchronize()


def wait_uploads() -> None:
    """Make the default stream (the graphs) wait, with one event, for the uploads this thread put on the side stream
    since the last submit. A readback the copy thread queued in between only waits on a finished graph, so it costs
    microseconds. Every submit runs it after its hooks (``install``); a hook may also run it earlier.

    It's always the default stream, not the submit's current stream: a submit on the side stream is already behind the
    uploads, but the graph after it still has to wait for them."""
    index = getattr(_STATE, "pending", None)
    if index is None:
        return
    _STATE.pending = None
    _default(index).wait_event(_side(index).record_event())


def spin_until(done: Callable[[], bool], spin_s: float) -> bool:
    """Poll ``done()`` for up to ``spin_s`` seconds, yielding the core (and the GIL) between polls; whether it came true.
    A spinning thread stays runnable, where a blocked one waits on the scheduler's run queue to wake."""
    if done():
        return True
    if spin_s <= 0:
        return False
    deadline = time.perf_counter() + spin_s
    while True:
        os.sched_yield()
        if done():
            return True
        if time.perf_counter() >= deadline:
            return False


def wait_launches(spin_s: float = 0.0) -> None:
    """Until every graph this thread launched has finished on the device (the graphs run in launch order on the
    default stream, so the newest launch's event covers them all). With ``spin_s``, the newest launch's event is polled
    for up to that long (``spin_until``) before the blocking wait."""
    if not on_device():
        return
    inflight = _inflight()
    if not inflight:
        return
    newest = inflight[-1]
    if not (spin_s > 0 and spin_until(newest.query, spin_s)):
        newest.synchronize()
    inflight.clear()


def last_launch_event() -> Optional[Any]:
    """The event of this thread's newest graph launch that may still be running, or None (nothing pending, or CPU mode).
    Graphs run in launch order on the default stream, so it covers every earlier launch. A PP KV sender waits on it
    before sending a chunk the stage's forward just wrote."""
    if not on_device():
        return None
    inflight = _inflight()
    return inflight[-1] if inflight else None


def after_launch() -> Optional[Any]:
    """Right after a graph launch: the event a readback of its outputs waits for (None in a CPU-mode process). The
    launch counts against the in-flight bound."""
    if not on_device():
        return None
    import torch_neuronx

    event = torch_neuronx.current_stream().record_event()
    _inflight().append(event)
    _STATE.launched = launches() + 1
    if _LAUNCHER["thread"] is None:
        _LAUNCHER["thread"] = threading.get_ident()
    return event


def read_landed(tensor):
    """A host copy of ``tensor``, a buffer no queued graph writes (a PP seam landing slot the network wrote, or a send
    slot whose forward the caller already waited for); safe from any thread. On torch-neuronx the read runs on the side
    stream, so it does not wait for the graphs queued on the default stream."""
    if not on_device() or tensor.device.type == "cpu":
        return tensor.to("cpu", copy=True)
    import torch_neuronx

    side = _side(tensor.device.index)
    with torch_neuronx.stream(side):
        tensor.record_stream(side)
        out = tensor.to("cpu", non_blocking=True)
        done = side.record_event()
    done.synchronize()
    return out


# the thread that launches this process's graphs (the scheduler's), recorded at the first launch
_LAUNCHER = {"thread": None}


def launching_thread() -> Optional[int]:
    """The ident of the thread that launched this process's first graph, or None before any launch (and in CPU mode)."""
    return _LAUNCHER["thread"]


def launches() -> int:
    """How many graphs this thread has launched (``after_launch``); 0 in a CPU-mode process."""
    return getattr(_STATE, "launched", 0)


def retired_launches() -> int:
    """A lower bound on this thread's launches that have finished on the device: all but those in its in-flight window.
    The graphs run in launch order, and a launch leaves the window only through a wait that covers it (before_launch's
    for the oldest, wait_launches' for the newest)."""
    return launches() - len(_inflight())


def readback(tensor, launch_event):
    """``tensor`` (an output of the graph whose ``after_launch`` returned ``launch_event``) on the host; blocks until
    the graph has finished. Any thread."""
    return readback_many((tensor,), launch_event)[0]


def readback_many(tensors, launch_event) -> list:
    """``tensors`` (outputs of the graph whose ``after_launch`` returned ``launch_event``, on one device) on the host,
    in order; blocks until the graph has finished. One wait for the graph and one for the reads, whatever the count:
    every read is queued on the side stream before the host waits for them. Any thread."""
    if launch_event is None:
        return [tensor.to("cpu") for tensor in tensors]
    devices = {tensor.device.index for tensor in tensors if tensor.device.type != "cpu"}
    if not devices:
        return [tensor.to("cpu") for tensor in tensors]
    if len(devices) > 1:
        raise ValueError(f"readback_many reads one device's tensors, got devices {sorted(devices)}")
    import torch_neuronx

    launch_event.synchronize()
    direct = _direct_read(tensors)
    if direct is not None:
        return direct
    side = _side(devices.pop())
    out = []
    with torch_neuronx.stream(side):
        side.wait_event(launch_event)  # finished: orders the reads in the stream at no wait
        for tensor in tensors:
            if tensor.device.type != "cpu":
                tensor.record_stream(side)
            out.append(tensor.to("cpu", non_blocking=True))
        done = side.record_event()
    done.synchronize()
    return out


# A readback of at most this many bytes per tensor (a step's sampled tokens and logprobs) is a direct runtime read.
DIRECT_READ_MAX_BYTES = 1 << 16


def _direct_read(tensors) -> Optional[list]:
    """The finished graph's small contiguous outputs read straight from their runtime tensors (``nrt_tensor_read``)
    into host tensors, or None when one is not such an output. A copy queued on the side stream would be a second
    device operation, started and retired through torch-neuronx's scheduler after the graph's own completion: two
    completion latencies on the path from the end of a step to its result on the host, which bounds how far ahead
    of the device the host can submit the step after next."""
    import ctypes

    import torch
    import torch_neuronx

    sized = [t for t in tensors if t.device.type != "cpu"]
    if any(not t.is_contiguous() or t.numel() * t.element_size() > DIRECT_READ_MAX_BYTES for t in sized):
        return None
    lib, get = _libnrt()
    out = []
    for tensor in tensors:
        if tensor.device.type == "cpu":
            out.append(tensor)
            continue
        host = torch.empty(tensor.shape, dtype=tensor.dtype)
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes:
            handle = get(torch_neuronx._C._get_nrt_tensor_capsule(tensor), b"nrt_tensor_t")
            status = lib.nrt_tensor_read(handle, ctypes.c_void_p(host.data_ptr()),
                                         tensor.storage_offset() * tensor.element_size(), nbytes)
            if status != 0:
                raise RuntimeError(f"nrt_tensor_read of a {tuple(tensor.shape)} {tensor.dtype} graph output failed: "
                                   f"NRT status {status}")
        out.append(host)
    return out


@functools.lru_cache(maxsize=1)
def _libnrt():
    import ctypes

    lib = ctypes.CDLL("libnrt.so.1")
    lib.nrt_tensor_get_va.restype = ctypes.c_void_p
    lib.nrt_tensor_get_va.argtypes = [ctypes.c_void_p]
    lib.nrt_tensor_get_size.restype = ctypes.c_size_t
    lib.nrt_tensor_get_size.argtypes = [ctypes.c_void_p]
    lib.nrt_tensor_read.restype = ctypes.c_int
    lib.nrt_tensor_read.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
    lib.nrt_get_vnc_memory_stats.restype = ctypes.c_int
    lib.nrt_get_vnc_memory_stats.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    get = ctypes.pythonapi.PyCapsule_GetPointer
    get.restype, get.argtypes = ctypes.c_void_p, [ctypes.py_object, ctypes.c_char_p]
    return lib, get


def device_address(tensor) -> int:
    """The device virtual address of ``tensor``'s first element, the address NIXL registers and an RDMA write
    targets (the KV pools). Call it where the region is registered, not per step.

    torch-neuronx's allocator binds device memory when the
    allocation reaches the stream, and ``data_ptr()`` is a placeholder; the address is the storage's NRT tensor's
    (``nrt_tensor_get_va``), bound once every queued allocation has run, plus the view's storage offset."""
    if not on_device() or tensor.device.type != "neuron":
        return int(tensor.data_ptr())
    import torch_neuronx

    torch_neuronx.synchronize()
    lib, get = _libnrt()
    handle = get(torch_neuronx._C._get_nrt_tensor_capsule(tensor), b"nrt_tensor_t")
    base = lib.nrt_tensor_get_va(handle)
    if not base:
        raise RuntimeError(f"the Neuron runtime has no device address for a {tuple(tensor.shape)} {tensor.dtype} "
                           "tensor")
    offset = tensor.storage_offset() * tensor.element_size()
    end = offset + tensor.numel() * tensor.element_size()
    size = int(lib.nrt_tensor_get_size(handle))
    if end > size:
        raise RuntimeError(f"a {tuple(tensor.shape)} {tensor.dtype} view ends at byte {end} of its {size}-byte NRT "
                           "tensor")
    return int(base) + offset


def vnc_bytes_used() -> Optional[int]:
    """The bytes the Neuron runtime holds on this rank's core (NEFFs, scratchpads and the allocator's segments):
    ``total - free`` of ``torch.neuron.mem_get_info()``. None in a CPU-mode process or before the runtime started: the
    query would start the runtime."""
    if not on_device():
        return None
    import torch
    import torch_neuronx

    if not torch_neuronx.is_neuron_runtime_initialized():
        return None
    mem_get_info = getattr(torch.neuron, "mem_get_info", None)
    if mem_get_info is not None:
        free, total = mem_get_info()
        return int(total) - int(free)
    # a torch-neuronx build without torch.neuron.mem_get_info: read libnrt's count for VNC 0
    import ctypes

    class _Stats(ctypes.Structure):
        _fields_ = [("bytes_used", ctypes.c_size_t), ("bytes_limit", ctypes.c_size_t)]

    lib, _ = _libnrt()
    stats, out = _Stats(), ctypes.c_size_t(0)
    status = lib.nrt_get_vnc_memory_stats(ctypes.c_uint32(0), ctypes.byref(stats), ctypes.sizeof(stats),
                                          ctypes.byref(out))
    if status != 0:
        raise RuntimeError(f"nrt_get_vnc_memory_stats failed with status {status}")
    return int(stats.bytes_used)
