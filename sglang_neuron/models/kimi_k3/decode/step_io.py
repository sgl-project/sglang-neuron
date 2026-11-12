# SPDX-License-Identifier: Apache-2.0
"""The per-step host inputs of the Kimi-K3 decode programs written straight into persistent device buffers.

A step's host inputs (about 23 small tensors, 114 KB at k3) change every step. Uploaded as copies, even on
device_io's side stream, each step's graphs wait for them through a stream event that torch-neuronx gates on the host:
measured on the 256-rank decode, the verify slot of a step is longer than with one fixed set of inputs.
``StepInputRing`` writes them with the runtime's ``nrt_tensor_write`` into one of ``depth`` sets of device buffers
instead, so no stream operation is queued for them at all. A set is written only after the last step that read it
has finished (its launch event), and the step's graphs are submitted after the write returns.
"""

from __future__ import annotations

import ctypes
import functools
import time
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor


@functools.lru_cache(maxsize=1)
def _libnrt():
    lib = ctypes.CDLL("libnrt.so.1")
    lib.nrt_tensor_write.restype = ctypes.c_int
    lib.nrt_tensor_write.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
    lib.nrt_tensor_get_size.restype = ctypes.c_size_t
    lib.nrt_tensor_get_size.argtypes = [ctypes.c_void_p]
    get = ctypes.pythonapi.PyCapsule_GetPointer
    get.restype, get.argtypes = ctypes.c_void_p, [ctypes.py_object, ctypes.c_char_p]
    return lib, get


def _handle(tensor: Tensor) -> int:
    import torch_neuronx

    lib, get = _libnrt()
    return get(torch_neuronx._C._get_nrt_tensor_capsule(tensor), b"nrt_tensor_t")


def writable(device_tensor: Tensor) -> bool:
    """Whether ``device_tensor`` is a contiguous tensor of its own runtime tensor's bytes from offset 0, which a raw
    write of the host tensor's bytes fills exactly."""
    lib, _ = _libnrt()
    nbytes = device_tensor.numel() * device_tensor.element_size()
    return (device_tensor.is_contiguous() and device_tensor.storage_offset() == 0
            and int(lib.nrt_tensor_get_size(_handle(device_tensor))) >= nbytes)


def write(device_tensor: Tensor, host: Tensor) -> None:
    """Write ``host``'s bytes into ``device_tensor`` (same shape and dtype) with ``nrt_tensor_write``; returns when the
    bytes are on the device."""
    if tuple(host.shape) != tuple(device_tensor.shape) or host.dtype != device_tensor.dtype:
        raise ValueError(f"A {host.dtype} {tuple(host.shape)} host input does not fit its {device_tensor.dtype} "
                         f"{tuple(device_tensor.shape)} device buffer.")
    host = host.contiguous()
    nbytes = host.numel() * host.element_size()
    if not nbytes:
        return
    lib, _ = _libnrt()
    status = lib.nrt_tensor_write(_handle(device_tensor), ctypes.c_void_p(host.data_ptr()), 0, nbytes)
    if status != 0:
        raise RuntimeError(f"nrt_tensor_write of a {tuple(host.shape)} {host.dtype} step input failed: NRT status "
                           f"{status}")


RING_DEPTH = 3
"""The input ring's sets, and the graphs' rotating output sets (``backend.output_buffers``): a step's IO set repeats
every ``RING_DEPTH`` steps, so the runtime reuses each graph's IO DMA rings."""


class StepInputRing:
    """``depth`` sets of persistent device buffers for the per-step host inputs of one runtime.

    ``write(host)`` takes the next set: it waits for the launch event of the last step that read the set (normally
    long finished), writes every host tensor that differs from what its buffer holds (a step's mask, tables and
    sampler rows mostly repeat) and returns the set's device tensors. The step that reads
    them gives its launch event to ``used(event)``. The first ``write`` of a name allocates its buffers with a plain
    copy (bound device tensors). A name whose buffer cannot take a raw write (not a plain contiguous runtime tensor)
    stays with ``fallback`` (the runtime's own upload) for every step.
    """

    def __init__(self, device: Any, fallback, depth: int = RING_DEPTH) -> None:
        if depth < 2:
            raise ValueError("The input ring needs 2 sets or more: the next step's inputs are written while a step runs.")
        self.device = torch.device(device)
        self.fallback = fallback
        self.depth = int(depth)
        self.sets: list[dict[str, Tensor]] = [{} for _ in range(self.depth)]
        # the host value each buffer of a set holds: an unchanged input is not written again
        self.held: list[dict[str, Tensor]] = [{} for _ in range(self.depth)]
        self.events: list[Any] = [None] * self.depth
        self.slow: set[str] = set()
        self.index = -1
        self.counts = {"writes": 0, "unchanged": 0, "fallback": 0, "steps": 0}
        # the last write's host times in ms (a probe): the wait for the set's last step, the raw writes, the fallback
        self.last_times = {"wait": 0.0, "writes": 0.0, "fallback": 0.0}

    def write(self, host: Mapping[str, Tensor]) -> dict[str, Tensor]:
        start = time.perf_counter()
        self.index = (self.index + 1) % self.depth
        event = self.events[self.index]
        if event is not None:
            event.synchronize()
            self.events[self.index] = None
        waited = time.perf_counter()
        writing = 0.0
        buffers, held = self.sets[self.index], self.held[self.index]
        out: dict[str, Tensor] = {}
        late: dict[str, Tensor] = {}
        self.counts["steps"] += 1
        for name, value in host.items():
            if value.device.type != "cpu":
                out[name] = value
                continue
            if name in self.slow:
                late[name] = value
                continue
            buffer = buffers.get(name)
            if buffer is None or tuple(buffer.shape) != tuple(value.shape) or buffer.dtype != value.dtype:
                buffer = value.to(self.device)
                torch_sync(buffer)
                if not writable(buffer):
                    self.slow.add(name)
                    late[name] = value
                    continue
                buffers[name] = buffer
                held[name] = value.clone()
                out[name] = buffer
                continue
            previous = held.get(name)
            if previous is not None and torch.equal(previous, value):
                self.counts["unchanged"] += 1
            else:
                t_write = time.perf_counter()
                write(buffer, value)
                writing += time.perf_counter() - t_write
                held[name] = value.clone()
                self.counts["writes"] += 1
            out[name] = buffer
        t_fallback = time.perf_counter()
        if late:
            self.counts["fallback"] += len(late)
            out.update(self.fallback(late))
        self.last_times = {"wait": (waited - start) * 1e3, "writes": writing * 1e3,
                           "fallback": (time.perf_counter() - t_fallback) * 1e3}
        return out

    def used(self, event: Any) -> None:
        """The launch event of the step that reads the set ``write`` returned last."""
        if self.index >= 0:
            self.events[self.index] = event


def torch_sync(tensor: Tensor) -> None:
    """Wait until a freshly copied device tensor is bound and written (a blocking read of it waits for its copy)."""
    tensor.to("cpu")
