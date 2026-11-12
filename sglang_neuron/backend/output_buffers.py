# SPDX-License-Identifier: Apache-2.0
"""Rotating output buffers for compiled-graph calls, so a graph's IO buffers repeat and the runtime reuses its IO rings.

The Neuron runtime builds a NEFF's IO DMA rings for each set of input and output buffers it executes with, and caches
``NEURON_RT_IO_RING_CACHE_SIZE`` such sets per NEFF. torch-neuronx's ``Executor`` allocates every output with
``torch.empty`` on every call, so a graph whose outputs feed its next calls (a decode step's carry) never executes with
the same set twice, and the runtime rebuilds the rings inside each submit. The device waits for that submit whenever
the graph is submitted after the work queued before it has ended.

Inside ``rotating(depth)``, this thread's executor calls take their outputs from ``depth`` sets of buffers kept per
executor, call ``n`` from set ``n % depth``, instead of fresh ones; a caller whose inputs come from the same number of
rotating sets (the decode's input ring) then repeats one IO set every ``depth`` calls.

Reuse is safe by construction:

- A buffer is handed out again only when nothing but its set holds it: neither the tensor object (its Python
  reference count) nor its storage (a view of it, as the executor's 64-bit outputs are). Otherwise that call allocates
  a new buffer in the slot, as torch-neuronx would, and the old one stays with whoever holds it (``fresh`` counts it).
- The graphs run on one stream in launch order, so an execution that writes a reused buffer runs after every graph
  launched before it that reads it, as with any buffer the caching allocator hands out again.
- An output aliased to an input (an in-place mutation) is the executor's own: its input's prepared tensor.
- Every reader of a graph output blocks until its read has finished: the next graph in launch order, or
  backend.device_io's readbacks (``readback_many``, ``read_landed``), which wait on the host for their side-stream
  copies before they return. A slot never goes back to the caching allocator, so ``record_stream`` does not hold it: a
  read that returned before its copy finished (a side-stream copy left in flight) would race the call that takes the
  slot next. No other module copies a device tensor to the host without blocking.

Measured on the Kimi-K3 two-node decode (k = 3, 1M context, 4 requests a replica): a shorter step with the rotating
outputs than without, on the same machines.

``Executor._allocate_output`` is wrapped (once, on the first ``rotating``) after its source is checked against the
torch-neuronx build this was written for (``torch_pins.TORCH_NEURONX_BUILD``, whose source backend.executor_cache
pins); another build is refused.
"""

from __future__ import annotations

import contextlib
import functools
import sys
import threading

import torch

_LOCAL = threading.local()
_installed = False
# every executor's outputs under ``rotating``, reused and freshly allocated (``totals``)
_TOTALS = {"reused": 0, "fresh": 0}


class _Sets:
    """One executor's ``depth`` output sets: per set, output index -> buffer."""

    def __init__(self, depth: int) -> None:
        self.depth = int(depth)
        self.sets: list[dict[int, torch.Tensor]] = [{} for _ in range(self.depth)]
        self.index = -1
        self.calls = 0
        self.reused = 0
        self.fresh = 0

    def advance(self) -> dict[int, torch.Tensor]:
        self.index = (self.index + 1) % self.depth
        self.calls += 1
        return self.sets[self.index]

    def current(self) -> dict[int, torch.Tensor]:
        return self.sets[self.index]


def _refs(slots: dict, i: int) -> tuple[int, int]:
    """(Python references, storage uses) of ``slots[i]``, as seen from here; equal to ``_baseline()`` when only the
    slot holds it."""
    buffer = slots[i]
    return sys.getrefcount(buffer), torch._C._storage_Use_Count(buffer.untyped_storage()._cdata)


@functools.lru_cache(maxsize=1)
def _baseline() -> tuple[int, int]:
    """``_refs`` of a buffer nothing else holds (measured, since the interpreter's own count of a local varies)."""
    return _refs({0: torch.empty(1)}, 0)


def _unheld(slots: dict, i: int) -> bool:
    """Whether only the slot holds ``slots[i]``: no other reference to the tensor, no view of its storage."""
    refs, uses = _refs(slots, i)
    base_refs, base_uses = _baseline()
    return refs <= base_refs and uses <= base_uses


def _fits(buffer: torch.Tensor, out, device_str: str) -> bool:
    return (tuple(buffer.shape) == tuple(out.shape) and buffer.dtype == out.dtype
            and str(buffer.device) == device_str)


def _wrap(original):
    @functools.wraps(original)
    def allocate(self, i, out, neuron_inputs, device_str):
        depth = getattr(_LOCAL, "depth", 0)
        if not depth:
            return original(self, i, out, neuron_inputs, device_str)
        sets = self.__dict__.get("_sgn_output_sets")
        if sets is None or sets.depth != depth:
            sets = self._sgn_output_sets = _Sets(depth)
        # outputs are allocated in index order, so index 0 starts a call
        slots = sets.advance() if i == 0 else sets.current()
        if self.alias_buffer_map and i in self.alias_buffer_map:
            return original(self, i, out, neuron_inputs, device_str)
        if i in slots and _fits(slots[i], out, device_str) and _unheld(slots, i):
            sets.reused += 1
            _TOTALS["reused"] += 1
            return slots[i]
        buffer = original(self, i, out, neuron_inputs, device_str)
        slots[i] = buffer
        sets.fresh += 1
        _TOTALS["fresh"] += 1
        return buffer

    allocate._sgn_output_buffers = True
    return allocate


def install() -> None:
    """Wrap ``Executor._allocate_output`` (idempotent); refuse a torch-neuronx whose source differs from the pin."""
    global _installed
    if _installed:
        return
    from torch_neuronx.neuron_dynamo_backend.executor import Executor

    from sglang_neuron.backend import executor_cache, torch_pins

    current = Executor.__dict__.get("_allocate_output")
    if not getattr(current, "_sgn_output_buffers", False):
        if executor_cache._md5(Executor, "_allocate_output") != executor_cache._SOURCES["_allocate_output"]:
            raise RuntimeError("torch-neuronx's Executor._allocate_output differs from the build the plugin's rotating "
                               f"output buffers were written for (torch-neuronx {torch_pins.TORCH_NEURONX_BUILD}); "
                               "re-validate sglang_neuron/backend/output_buffers.py against this torch-neuronx")
        Executor._allocate_output = _wrap(current)
    _installed = True


@contextlib.contextmanager
def rotating(depth: int):
    """This thread's compiled-graph calls inside take their outputs from ``depth`` rotating sets per executor (the
    module docstring); ``depth`` 0 leaves torch-neuronx's allocation as it is."""
    depth = int(depth)
    if depth < 0:
        raise ValueError(f"rotating output buffers: depth {depth}; give 0 (off) or more")
    if depth:
        install()
    outer = getattr(_LOCAL, "depth", 0)
    _LOCAL.depth = depth
    try:
        yield
    finally:
        _LOCAL.depth = outer


def stats(executor) -> dict[str, int]:
    """An executor's calls under ``rotating`` and how many outputs were reused or freshly allocated."""
    sets = getattr(executor, "__dict__", {}).get("_sgn_output_sets")
    if sets is None:
        return {"calls": 0, "reused": 0, "fresh": 0}
    return {"calls": sets.calls, "reused": sets.reused, "fresh": sets.fresh}


def totals() -> dict[str, int]:
    """All executors' outputs under ``rotating`` so far: reused and freshly allocated."""
    return dict(_TOTALS)
