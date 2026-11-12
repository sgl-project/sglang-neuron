# SPDX-License-Identifier: Apache-2.0
"""The slot operations a recurrent-state pool runs on its tensors, as the hook a model's kernels provide.

The framework never decides how state bytes move on the device: a model hands the pool factory a ``StateOps`` whose
methods its kernel package implements (``kernels/<model>/``: fused NKI clears, precompiled copy/load/publish
programs) or this module's ``CompiledStateOps``. ``TorchStateOps`` is the reference implementation, eager torch
indexing, which the CPU mode and the tests run and which defines what every implementation must do:

* ``clear_slots(tensors, slots)``: every component of every layer of each slot reads zero.
* ``copy_slots(tensors, src, dst)``: slot ``dst[i]`` becomes a bit copy of slot ``src[i]`` in every component and
  layer; sources are unchanged (copy-on-write on a radix match).
* ``load_rows`` / ``publish_rows`` (working-bank models only): canonical slot -> working row before a forward, working
  row -> canonical slot after an accepted forward.
* ``drain()``: every operation submitted so far has completed (or raised). A pool calls it before its slots change
  owner (drain-before-reset); a stream-ordered implementation that completes in submission order may leave it a no-op.
* ``copy_across(src_tensors, dst_tensors, src, dst)`` (only for a separate checkpoint store,
  mem_cache/state_checkpoint_pool): slot ``dst[i]`` of every ``dst_tensors`` component becomes a bit copy of slot
  ``src[i]`` of the same component in ``src_tensors``, a tensor of the same per-slot shape and dtype with its own slot
  count (the store's checkpoint on a radix insert, the request's slot on a hit). Both sides keep a zero slot 0 and a
  trailing sink, so padding reads slot 0 and writes the sink as the in-pool copy does.

``CompiledStateOps`` is the device implementation of a model without a working bank whose kernels advance the canonical
slots in place (a GDN or KDA state): one compiled graph per state tensor and operation (an
``index_fill`` along the slot axis, or a gather then a scatter), padded to fixed buckets so that a live step never
compiles. It reads nothing of the model but the tensors' slot axis, so every such model shares it.

``WorkingBankOps`` serves a working-bank pool: the model's row programs (``load_rows`` / ``publish_rows``, which
its kernels own) with ``CompiledStateOps`` on the canonical slots (``clear_slots`` / ``copy_slots`` / ``copy_across``).
The canonical slots are plain ``[layers, slots + 2, *shape]`` tensors, so the generic compiled graphs serve them, and
the radix copy-on-write and the checkpoint store's copies never run as eager device ops that compile at a live step.

Slot index lists are host integers, validated by the pool before the call; ``tensors`` maps a component name to its
``[layers, slots, *shape]`` tensor (the working bank under ``work.<name>``). An implementation pads to its compiled
buckets itself; padding reads slot 0, which stays zero, and writes only a sink the pool reserves.
"""

from __future__ import annotations

import bisect
import itertools
import logging
from typing import Dict, Protocol, Sequence

import torch

logger = logging.getLogger(__name__)


class StateOps(Protocol):
    def clear_slots(self, tensors: Dict[str, torch.Tensor], slots: Sequence[int]) -> None: ...

    def copy_across(self, src_tensors: Dict[str, torch.Tensor], dst_tensors: Dict[str, torch.Tensor],
                    src: Sequence[int], dst: Sequence[int]) -> None: ...

    def copy_slots(self, tensors: Dict[str, torch.Tensor], src: Sequence[int], dst: Sequence[int]) -> None: ...

    def load_rows(self, tensors: Dict[str, torch.Tensor], slots: Sequence[int], rows: Sequence[int]) -> None: ...

    def publish_rows(self, tensors: Dict[str, torch.Tensor], slots: Sequence[int], rows: Sequence[int]) -> None: ...

    def drain(self) -> None: ...


def _index(values, device) -> torch.Tensor:
    return torch.as_tensor(list(values), dtype=torch.int64, device=device)


class TorchStateOps:
    """Eager torch reference: synchronous index writes, so ``drain`` has nothing to wait for."""

    name = "torch"

    def clear_slots(self, tensors, slots):
        if not len(slots):
            return
        for name, tensor in tensors.items():
            if not name.startswith("work."):
                tensor[:, _index(slots, tensor.device)] = 0

    def copy_slots(self, tensors, src, dst):
        if not len(dst):
            return
        for name, tensor in tensors.items():
            if not name.startswith("work."):
                # gather first: a dst that is also a later src still reads the pre-copy bytes
                tensor[:, _index(dst, tensor.device)] = tensor[:, _index(src, tensor.device)].clone()

    def copy_across(self, src_tensors, dst_tensors, src, dst):
        if not len(dst):
            return
        for name, tensor in dst_tensors.items():
            source = src_tensors[name]
            tensor[:, _index(dst, tensor.device)] = source[:, _index(src, source.device)]

    def load_rows(self, tensors, slots, rows):
        for name, tensor in tensors.items():
            if not name.startswith("work."):
                tensors["work." + name][:, _index(rows, tensor.device)] = tensor[:, _index(slots, tensor.device)]

    def publish_rows(self, tensors, slots, rows):
        for name, tensor in tensors.items():
            if not name.startswith("work."):
                tensor[:, _index(slots, tensor.device)] = tensors["work." + name][:, _index(rows, tensor.device)]

    def drain(self):
        return None


class WorkingBankOps:
    """A working-bank pool's slot operations (the module docstring): ``rows`` the model's row programs (load_rows /
    publish_rows; TorchStateOps, the reference, when the model has none of its own), ``canonical`` the canonical-slot
    operations (CompiledStateOps when None). Both run on the device's default queue in submission order with the
    forwards that read the slots; ``drain`` drains both."""

    def __init__(self, rows, canonical=None):
        missing = [n for n in ("load_rows", "publish_rows", "drain") if not callable(getattr(rows, n, None))]
        if missing:
            raise TypeError(f"working-bank row ops {type(rows).__name__} lack {missing}")
        self.rows = rows
        self.canonical = canonical if canonical is not None else CompiledStateOps()
        self.name = f"{getattr(rows, 'name', type(rows).__name__)} rows + {getattr(self.canonical, 'name', 'canonical')}"

    def clear_slots(self, tensors, slots):
        self.canonical.clear_slots(tensors, slots)

    def copy_slots(self, tensors, src, dst):
        self.canonical.copy_slots(tensors, src, dst)

    def copy_across(self, src_tensors, dst_tensors, src, dst):
        self.canonical.copy_across(src_tensors, dst_tensors, src, dst)

    def load_rows(self, tensors, slots, rows):
        self.rows.load_rows(tensors, slots, rows)

    def publish_rows(self, tensors, slots, rows):
        self.rows.publish_rows(tensors, slots, rows)

    def warm(self, tensors) -> None:
        for ops in (self.canonical, self.rows):
            warm = getattr(ops, "warm", None)
            if callable(warm):
                warm(tensors)

    def warm_across(self, a_tensors, b_tensors) -> None:
        warm = getattr(self.canonical, "warm_across", None)
        if callable(warm):
            warm(a_tensors, b_tensors)

    def drain(self):
        self.rows.drain()
        self.canonical.drain()


def working_bank_ops(ops=None):
    """The ops a working-bank pool runs: ``ops`` as given when it carries every operation the pool and its checkpoint
    store call (a model whose kernels own its canonical-slot copies too, or a test's recording ops); otherwise
    ``WorkingBankOps`` over ``ops`` as the row programs (TorchStateOps when None), the canonical slots on the compiled
    graphs. A model's precompiled ops that cover only load / publish so get the canonical-slot operations, and
    with them a ``copy_across``: --enable-int8-mamba-checkpoint is served for its pool."""
    if ops is None:
        return WorkingBankOps(TorchStateOps())
    if isinstance(ops, WorkingBankOps):
        return ops
    if all(callable(getattr(ops, n, None)) for n in ("clear_slots", "copy_slots", "copy_across", "drain")):
        return ops
    return WorkingBankOps(ops)


def check_ops(ops, *, working: bool) -> None:
    """Refuse an ops object missing a method the pool will call (at pool construction, not at the first request)."""
    needed = ["clear_slots", "copy_slots", "drain"] + (["load_rows", "publish_rows"] if working else [])
    missing = [n for n in needed if not callable(getattr(ops, n, None))]
    if missing:
        raise TypeError(f"state ops {type(ops).__name__} lack {missing}: a model's kernel package must provide every "
                        "slot operation its pool runs (mem_cache/state_ops.StateOps)")


# CompiledStateOps' buckets: a call of n slots runs the smallest bucket >= n (larger calls run in chunks of the largest).
# Every padding row rewrites the sink slot, a whole slot of every layer (18 MB on the rank measured here), so the steps
# stay at most 2x apart: with (8, 64, 256) a 16-slot clear wrote 64 slots and took longer than eager on trn3.
OP_BUCKETS = (8, 16, 32, 64, 128, 256)

# copy_across's buckets: the checkpoint store copies one slot per radix insert (upstream takes one checkpoint slot per
# insert) and one per prefill request on a hit (a forward runs one request's prefill), so a call is almost always one
# slot; a padding row rewrites a whole slot of every layer (23.6 MB per Qwen3.5 rank), so the small sizes are exact
ACROSS_BUCKETS = (1, 4, 16)

# a state tensor's bytes past which the indexed copies the slot operations and the kernels compile have DMA static
# offsets the descriptor's 32-bit immediate cannot encode (neuronx-cc NCC_IEIL015)
MAX_TENSOR_BYTES = 1 << 32


def oversized_tensors(tensors: Dict[str, torch.Tensor]) -> list:
    """The state tensors whose one allocation exceeds ``MAX_TENSOR_BYTES``."""
    return [f"{name} {tuple(t.shape)} is {t.nbytes / 2**30:.2f} GiB, past the 4 GiB an indexed DMA addresses"
            for name, t in tensors.items() if t.nbytes > MAX_TENSOR_BYTES]


def _clear_body(tensor, index):
    """Every listed slot of every layer reads zero (index rows past the call's slots name the sink)."""
    tensor.index_fill_(1, index, 0)
    return tensor.reshape(-1)[-1:].clone()


def _copy_body(tensor, src, dst):
    """Slot ``dst[i]`` takes slot ``src[i]``'s bytes in every layer; the gather completes before the scatter."""
    tensor.index_copy_(1, dst, tensor.index_select(1, src))
    return tensor.reshape(-1)[-1:].clone()


def _across_body(source, tensor, src, dst):
    """Slot ``dst[i]`` of ``tensor`` takes slot ``src[i]`` of ``source`` in every layer."""
    tensor.index_copy_(1, dst, source.index_select(1, src))
    return tensor.reshape(-1)[-1:].clone()


_DISTINCT = itertools.count()


def compile_op(body):
    """A slot operation's compiled graph (on a NeuronCore): ``body`` over its own code (_distinct), static, whole, with
    no eager fallback, under dynamo's tensor guards only. The pool warms its operations as it is built, outside SGLang's
    scheduler loop, which runs under DynamicGradMode (inference mode): with dynamo's global-state guards (grad mode)
    every warmed graph would compile again at its first call while serving, a new executor and NEFF at the first
    request. An operation takes only tensors, none of which requires grad, so grad mode changes nothing it computes."""
    from sglang_neuron.backend import stack

    return torch.compile(_distinct(body), backend=stack.compile_backend_name(), fullgraph=True, dynamic=False,
                         options={"fallback_execution": False,
                                  "guard_filter_fn": torch.compiler.keep_tensor_guards_unsafe})


def _distinct(fn):
    """``fn`` under a code object of its own. Dynamo keys its guard cache and its recompile limit (8) on the code
    object, not on the torch.compile wrapper, so graphs of one body for several tensor shapes and buckets would share
    one cache: past 8 shapes a ``fullgraph`` compile raises FailOnRecompileLimitHit (measured on trn3: a second state
    pool's warm in one process). A copy of the code per graph gives each its own cache."""
    import types

    code = fn.__code__.replace(co_name=f"{fn.__name__}_{next(_DISTINCT)}")
    return types.FunctionType(code, fn.__globals__, code.co_name, fn.__defaults__, fn.__closure__)


class CompiledStateOps:
    """Compiled slot operations of a state pool without a working bank (the module docstring): padding clears and
    writes the sink slot and reads the zero slot 0. They run on the default device queue in submission order with the
    forwards that read the slots, so ``drain`` has nothing to wait for."""

    name = "compiled"

    def __init__(self, buckets: Sequence[int] = OP_BUCKETS, across_buckets: Sequence[int] = ACROSS_BUCKETS):
        self.buckets = tuple(sorted(int(b) for b in buckets))
        self.across_buckets = tuple(sorted(int(b) for b in across_buckets))
        self._graphs = {}
        # one device_io.UploadRing of slot ids per (graph, argument, length): _call
        self._fixed = {}
        self.calls = 0

    def _graph(self, kind: str, tensor: torch.Tensor, bucket: int, source: torch.Tensor = None):
        """The compiled graph of ``kind`` for this tensor's shape (and the source's, for a copy across tensors) and the
        bucket, each over its own copy of the body's code (_distinct), so no number of tensors, buckets or pools in a
        process reaches dynamo's recompile limit."""
        device = tensor.device
        key = (kind, tuple(tensor.shape), tensor.dtype, str(device),
               None if source is None else tuple(source.shape), int(bucket))
        graph = self._graphs.get(key)
        if graph is None:
            body = {"clear": _clear_body, "copy": _copy_body, "across": _across_body}[kind]
            if device.type != "neuron":
                graph = body  # the host (CPU mode, tests) and the meta device run the body eagerly
            else:
                graph = compile_op(body)
            self._graphs[key] = graph
        return graph

    def _call(self, graph, tensor, *ids, source=None) -> None:
        """Run ``graph`` on ``tensor`` (after ``source``, a copy across pools' source, when given) with each id list as
        a device tensor. On a NeuronCore each argument's ids go
        through its own device_io.UploadRing (keyed by the graph, the argument and the length): a fresh device tensor
        per call gives each execution a new input address, for which the runtime builds a new DMA ring out of HBM a
        server sized to its pools does not keep free (the same failure can stop a GLM-5.2 server under an abort storm
        through the zero fills); the ring's few fixed buffers are written asynchronously, each after the graph that
        last read it."""
        device = tensor.device
        lead = () if source is None else (source,)
        if device.type != "neuron":
            graph(*lead, tensor, *(torch.tensor(list(v), dtype=torch.int64).to(device) for v in ids))
            return
        from sglang_neuron.backend import device_io

        rings = []
        for arg, values in enumerate(ids):
            key = (id(graph), arg, len(values))
            ring = self._fixed.get(key)
            if ring is None:
                ring = self._fixed[key] = device_io.UploadRing((len(values),), torch.int64, device)
            rings.append(ring)
        args = [ring.put(torch.tensor(list(v), dtype=torch.int64)) for ring, v in zip(rings, ids)]
        graph(*lead, tensor, *args)
        for ring in rings:
            ring.used()

    def _chunks(self, n: int, buckets=None):
        buckets = self.buckets if buckets is None else buckets
        top = buckets[-1]
        for start in range(0, n, top):
            stop = min(n, start + top)
            yield start, stop, buckets[min(bisect.bisect_left(buckets, stop - start), len(buckets) - 1)]

    def clear_slots(self, tensors, slots):
        slots = list(slots)
        for name, tensor in tensors.items():
            if name.startswith("work."):
                continue
            sink = tensor.shape[1] - 1
            for start, stop, bucket in self._chunks(len(slots)):
                part = slots[start:stop]
                self._call(self._graph("clear", tensor, bucket), tensor, part + [sink] * (bucket - len(part)))
                self.calls += 1

    def copy_slots(self, tensors, src, dst):
        src, dst = list(src), list(dst)
        for name, tensor in tensors.items():
            if name.startswith("work."):
                continue
            sink = tensor.shape[1] - 1
            for start, stop, bucket in self._chunks(len(dst)):
                pad = bucket - (stop - start)
                self._call(self._graph("copy", tensor, bucket), tensor, src[start:stop] + [0] * pad,
                           dst[start:stop] + [sink] * pad)
                self.calls += 1

    def copy_across(self, src_tensors, dst_tensors, src, dst):
        src, dst = list(src), list(dst)
        for name, tensor in dst_tensors.items():
            source = src_tensors[name]
            sink = tensor.shape[1] - 1
            for start, stop, bucket in self._chunks(len(dst), self.across_buckets):
                pad = bucket - (stop - start)
                self._call(self._graph("across", tensor, bucket, source), tensor, src[start:stop] + [0] * pad,
                           dst[start:stop] + [sink] * pad, source=source)
                self.calls += 1

    def warm_across(self, a_tensors, b_tensors) -> None:
        """Compile copy_across both ways between two tensor sets at every bucket at startup (each reads slot 0 and
        writes the sink only), so a radix insert or hit never compiles on one rank while the others wait."""
        for x, y in ((a_tensors, b_tensors), (b_tensors, a_tensors)):
            for name, tensor in y.items():
                sink = tensor.shape[1] - 1
                for bucket in self.across_buckets:
                    self._call(self._graph("across", tensor, bucket, x[name]), tensor, [0] * bucket, [sink] * bucket,
                               source=x[name])
        logger.info("[neuron] state copy across tensors warmed: %d tensors x buckets %s", len(b_tensors),
                    self.across_buckets)

    def load_rows(self, tensors, slots, rows):
        raise NotImplementedError("the compiled slot operations serve a pool with no working bank (the kernels "
                                  "advance slots in place)")

    publish_rows = load_rows

    def drain(self):
        # submitted on the default device queue, in order with the forwards that read the slots
        return None

    def warm(self, tensors) -> None:
        """Compile every bucket of both operations at startup (each touches the sink and reads slot 0 only)."""
        for name, tensor in tensors.items():
            if name.startswith("work."):
                continue
            sink = tensor.shape[1] - 1
            for bucket in self.buckets:
                self._call(self._graph("clear", tensor, bucket), tensor, [sink] * bucket)
                self._call(self._graph("copy", tensor, bucket), tensor, [0] * bucket, [sink] * bucket)
        logger.info("[neuron] state slot operations warmed: %d tensors x buckets %s", len(tensors), self.buckets)
