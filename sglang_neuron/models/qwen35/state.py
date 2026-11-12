# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5's GDN recurrent state on the platform's recurrent-state framework (docs/design/recurrent-state.md).

- ``state_spec``: the per-rank state of the 45 GDN layers, family ``gdn``, as upstream's
  ``Mamba2StateShape.create`` sizes it at attention TP = the replica (``config.Qwen35Layout``): temporal
  ``(v_heads/tp, head_v, head_k)`` float32, stored in the giga kernels' layout ``(head_k, v_heads/tp, head_v)``
  (``rec_state [rows, 128, vh, 128]``, token-major), and the conv window ``(conv_dim/tp, kernel - 1)`` bfloat16 in
  upstream's layout (q | k | v sub-blocks of the rank's heads, which is the kernels' ``conv_state [rows, Cc, 3]``).
  The kernels advance the canonical slots in place (no working bank) and capture state only at a forward's end
  (``track_chunk`` 0).
- ``Qwen35StatePool``: the framework's ``NeuronStatePool`` (its tensors on the Neuron device) with the kernels' views:
  each component's ``[layers, slots + 2, *shape]`` tensor is also held as ``[layers * (slots + 2), *shape]``
  (``kernel_tensors``), the layer-major pool ``qwen35_hybrid_multistep`` takes, in which GDN layer ``g``'s slot ``s``
  is row ``g * (slots + 2) + s``. Both are views of one allocation.
- ``GdnStateOps``: the pool's slot operations as compiled per-tensor graphs (one ``index_fill`` / gather-scatter per
  state tensor), padded to fixed buckets so that a live step never compiles: padding clears and writes the sink slot
  and reads the zero slot 0. They run on the default device queue in submission order with the forwards that read the
  slots, so ``drain`` has nothing to wait for.
- PD: a prefill and a decode of one attention TP (the served PD pairing: both roles run the same replica) send each
  slot whole, through upstream's equal-TP path. The temporal state's kernel layout marks its entries not
  head-sliceable, so a decode of another attention TP is refused at registration (disaggregation/state_transfer)
  with the reason.
"""
from __future__ import annotations

import bisect
import dataclasses
import logging
from typing import Dict, Sequence

import torch

from sglang_neuron.mem_cache.state_pool import NeuronStatePool
from sglang_neuron.mem_cache.state_spec import StateSpec, gdn_spec

from .config import Qwen35Dims, Qwen35Layout

logger = logging.getLogger(__name__)

# slot-operation buckets: a call of n slots runs the smallest bucket >= n (larger calls run in chunks of the largest)
OP_BUCKETS = (8, 64, 256)


def state_spec(dims: Qwen35Dims, layout: Qwen35Layout) -> StateSpec:
    """The GDN state one rank holds (module docstring)."""
    spec = gdn_spec(layer_ids=dims.gdn_layers, tp_size=layout.replica, num_k_heads=dims.num_k_heads,
                    num_v_heads=dims.num_v_heads, head_k_dim=dims.head_k_dim, head_v_dim=dims.head_v_dim,
                    conv_kernel=dims.conv_kernel, temporal_dtype="float32", conv_dtype="bfloat16",
                    numerics="qwen35-giga")
    vh = dims.num_v_heads // layout.replica
    temporal = dataclasses.replace(spec.recurrent, physical_shape=(dims.head_k_dim, vh, dims.head_v_dim))
    return dataclasses.replace(spec, components=(temporal, *spec.convs))


# a state tensor's bytes past which the indexed copies the slot operations and the kernels compile have DMA static
# offsets the descriptor's 32-bit immediate cannot encode (neuronx-cc NCC_IEIL015)
MAX_TENSOR_BYTES = 1 << 32


def oversized_tensors(tensors: Dict[str, torch.Tensor]) -> list:
    """The state tensors whose one allocation exceeds ``MAX_TENSOR_BYTES``."""
    return [f"{name} {tuple(t.shape)} is {t.nbytes / 2**30:.2f} GiB, past the 4 GiB an indexed DMA addresses"
            for name, t in tensors.items() if t.nbytes > MAX_TENSOR_BYTES]


class Qwen35StatePool(NeuronStatePool):
    """The GDN state pool (module docstring). Bind with ``Qwen35StatePool.bind(spec, ops)``."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        problems = oversized_tensors(self.tensors)
        if problems:
            raise ValueError("the Qwen3.5 GDN state pool: " + "; ".join(problems) + " (fewer state slots per rank: "
                             "a lower --max-running-requests or --mem-fraction-static)")
        self.kernel_tensors: Dict[str, torch.Tensor] = {}
        for c in self.spec_stage.components:
            t = self.tensors[c.name]
            self.kernel_tensors[c.name] = t.view(t.shape[0] * t.shape[1], *t.shape[2:])

    @property
    def kernel_slots(self) -> int:
        """Rows per layer block of the kernel's pools (``Sp``): padding slot 0, the slots, the sink."""
        return self.physical_slots


def kernel_pool(name: str):
    """A ``PoolEntry`` resolver of the state pool's layer-major kernel tensor of component ``name``."""

    def resolve(kv_pool, req_to_token_pool):
        pool = getattr(req_to_token_pool, "mamba_pool", None)
        if not isinstance(pool, Qwen35StatePool):
            raise RuntimeError(f"the request table holds {type(pool).__name__}, not the Qwen3.5 state pool "
                               "(PoolLayout.state_pool_class)")
        return [pool.kernel_tensors[name]]

    return resolve


def _clear_body(tensor, index):
    """Every listed slot of every layer reads zero (index rows past the call's slots name the sink)."""
    tensor.index_fill_(1, index, 0)
    return tensor.reshape(-1)[-1:].clone()


def _copy_body(tensor, src, dst):
    """Slot ``dst[i]`` takes slot ``src[i]``'s bytes in every layer; the gather completes before the scatter."""
    tensor.index_copy_(1, dst, tensor.index_select(1, src))
    return tensor.reshape(-1)[-1:].clone()


class GdnStateOps:
    """Compiled slot operations of the Qwen3.5 state pool (module docstring)."""

    name = "qwen35-compiled"

    def __init__(self, buckets: Sequence[int] = OP_BUCKETS):
        self.buckets = tuple(sorted(int(b) for b in buckets))
        self._graphs = {}
        self.calls = 0

    def _graph(self, kind: str, device):
        graph = self._graphs.get((kind, device.type))
        if graph is None:
            body = _clear_body if kind == "clear" else _copy_body
            if device.type == "cpu":
                graph = body
            else:
                import torch._dynamo

                from sglang_neuron.backend import stack

                # one dynamo entry per (state tensor, bucket) of each operation: 2 tensors x the buckets, and every
                # runtime of an aggregated server warms them (above dynamo's default limit of 8 entries)
                torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit,
                                                           4 * len(self.buckets) * 4)
                graph = torch.compile(body, backend=stack.compile_backend_name(), fullgraph=True, dynamic=False,
                                      options={"fallback_execution": False})
            self._graphs[(kind, device.type)] = graph
        return graph

    @staticmethod
    def _index(values, device):
        host = torch.tensor(list(values), dtype=torch.int64)
        if device.type == "cpu":
            return host
        from sglang_neuron.backend import device_io

        return device_io.upload(host, device)

    def _chunks(self, n: int):
        top = self.buckets[-1]
        for start in range(0, n, top):
            stop = min(n, start + top)
            yield start, stop, self.buckets[min(bisect.bisect_left(self.buckets, stop - start),
                                                 len(self.buckets) - 1)]

    def clear_slots(self, tensors, slots):
        slots = list(slots)
        for name, tensor in tensors.items():
            if name.startswith("work."):
                continue
            sink = tensor.shape[1] - 1
            for start, stop, bucket in self._chunks(len(slots)):
                part = slots[start:stop]
                index = self._index(part + [sink] * (bucket - len(part)), tensor.device)
                self._graph("clear", tensor.device)(tensor, index)
                self.calls += 1

    def copy_slots(self, tensors, src, dst):
        src, dst = list(src), list(dst)
        for name, tensor in tensors.items():
            if name.startswith("work."):
                continue
            sink = tensor.shape[1] - 1
            for start, stop, bucket in self._chunks(len(dst)):
                pad = bucket - (stop - start)
                s = self._index(src[start:stop] + [0] * pad, tensor.device)
                d = self._index(dst[start:stop] + [sink] * pad, tensor.device)
                self._graph("copy", tensor.device)(tensor, s, d)
                self.calls += 1

    def load_rows(self, tensors, slots, rows):
        raise NotImplementedError("the Qwen3.5 state pool has no working bank (the kernels advance slots in place)")

    publish_rows = load_rows

    def drain(self):
        # submitted on the default device queue, in order with the forwards that read the slots
        return None

    def warm(self, tensors) -> None:
        """Compile every bucket of both operations at startup (each touches the sink and reads slot 0 only); a tensor
        another runtime of the server already warmed is skipped."""
        warmed = self.__dict__.setdefault("_warmed", set())
        for name, tensor in tensors.items():
            if name.startswith("work.") or (name, tensor.data_ptr()) in warmed:
                continue
            warmed.add((name, tensor.data_ptr()))
            sink = tensor.shape[1] - 1
            for bucket in self.buckets:
                self._graph("clear", tensor.device)(tensor, self._index([sink] * bucket, tensor.device))
                self._graph("copy", tensor.device)(tensor, self._index([0] * bucket, tensor.device),
                                                   self._index([sink] * bucket, tensor.device))
        logger.info("[qwen35] state slot operations warmed: %d tensors x buckets %s", len(tensors), self.buckets)


def layout_parts(spec: StateSpec, ops=None):
    """The state pool class and the pool entries of the kernels' layer-major state tensors (``state.<component>``,
    kinds ``recurrent_state`` and ``conv_state``, the GDN layers' global ids)."""
    from sglang_neuron.megakernel.pools import PoolEntry

    cls = Qwen35StatePool.bind(spec, ops if ops is not None else GdnStateOps(), name="Qwen35StatePool_gdn")
    entries = tuple(
        PoolEntry(name=f"state.{c.name}", kind="recurrent_state" if c.role == "recurrent" else "conv_state",
                  layers=spec.layer_ids, resolve=kernel_pool(c.name), buffers=1, per_buffer=len(spec.layer_ids))
        for c in spec.components)
    return cls, entries
