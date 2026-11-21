# SPDX-License-Identifier: Apache-2.0
"""Host weight copies backed by a file on local disk instead of anonymous process memory.

On one trn2 device, MiniMax-H3 keeps host copies of weights that do not fit in HBM (the streamed
conditioner's blocks, and what moves out around the video decode). In process memory those pages are
anonymous: under memory pressure the kernel can only write them to swap and read them back slowly.
``model_config.host_weights_dir`` (a local NVMe instance store, e.g. ``/mnt/nvme/h3``) writes each
copy to a file there and memory-maps it read-only instead:

* The pages are page cache, not process memory: the kernel drops them under pressure without writing
  anything, and re-reads them from the file on the next use.
* Ranks whose copies are identical (the video decoder is replicated on every rank) map one file, so the
  page cache holds one copy for all of them. Content-addressed by a caller-provided key.
* On an idle host nothing changes: the pages stay cached and reads are memory speed.

The copy to the device reads straight from the mapped pages (`tensor.to(device)`); no extra host buffer
is kept.
"""

from __future__ import annotations

import contextlib
import fcntl
import gc
import json
import logging
import os

import torch
from torch import nn

logger = logging.getLogger(__name__)

_ALIGN = 4096


def to_mapped(tensors: dict, key: str, directory: str) -> dict:
    """Return read-only CPU views of ``tensors`` backed by ``<directory>/<key>.bin``.

    ``tensors`` maps any hashable to a CPU tensor. The first process to arrive writes the file (others
    with the same ``key`` wait on a lock and reuse it). The returned tensors share memory with the file
    mapping; the inputs can be released.
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{key}.bin")
    names = list(tensors)
    layout, offset = [], 0
    for name in names:
        t = tensors[name]
        nbytes = t.numel() * t.element_size()
        layout.append((offset, nbytes, t.dtype, tuple(t.shape)))
        offset = (offset + nbytes + _ALIGN - 1) // _ALIGN * _ALIGN
    total = max(offset, _ALIGN)
    meta = json.dumps([(o, n, str(d), s) for o, n, d, s in layout])
    with open(path + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        done = path + ".done"
        if not (os.path.exists(done) and open(done).read() == meta):
            with open(path + ".tmp", "wb") as handle:
                handle.truncate(total)
                for name, (o, n, _, _) in zip(names, layout):
                    handle.seek(o)
                    handle.write(
                        tensors[name].contiguous().view(torch.uint8).reshape(-1).numpy().tobytes()
                    )
            os.replace(path + ".tmp", path)
            with open(done, "w") as handle:
                handle.write(meta)
        fcntl.flock(lock, fcntl.LOCK_UN)
    storage = torch.UntypedStorage.from_file(path, shared=False, nbytes=total)
    flat = torch.empty(0, dtype=torch.uint8).set_(storage)
    out = {}
    for name, (o, n, dtype, shape) in zip(names, layout):
        out[name] = flat[o : o + n].view(dtype).view(shape)
    logger.info("MiniMax-H3 host weights: %s, %.2f GB mapped from %s.", key, total / 1e9, path)
    return out


def release_device_cache() -> None:
    """Return the device memory freed tensors leave in torch-neuronx's caching allocator to the
    runtime.

    The allocator caches freed device blocks, so dropping the device copies of weights does not
    make room for a differently shaped allocation (the next module's weights) until the cache is
    emptied. A no-op before the Neuron runtime starts (CPU mode, the unit tests)."""
    import sys

    torch_neuronx = sys.modules.get("torch_neuronx")
    if torch_neuronx is None or not torch_neuronx.is_neuron_runtime_initialized():
        return
    # Device tensors held only by reference cycles (the VAE's tile bookkeeping) are freed first.
    gc.collect()
    torch_neuronx.empty_cache()


class HostSwap:
    """Move ``modules``' parameters and buffers between the device and a host copy taken once.

    ``evict`` points them at the host copy, which needs no device-to-host copy since the weights
    never change; ``load`` copies the host copy to ``device``. Each tensor is replaced the way
    ``Module.to`` replaces it across device types (a new ``Parameter``), which compiled graphs
    that take the weights as inputs tolerate. With ``directory`` the host copy is file-backed
    (``to_mapped``), keyed by ``key``."""

    def __init__(self, modules, device, directory: str | None, key: str, on_device: bool = True):
        """``on_device``: whether ``modules`` are on ``device`` now. If they are, only the tensors
        that are (not, e.g., ``adaln_on_host``'s host-resident projections) take part; if not,
        all of them, and ``load`` brings them in."""
        self.device = device
        self._slots = []
        seen = set()
        for module in modules:
            for owner in module.modules():
                for kind in ("_parameters", "_buffers"):
                    for name, value in getattr(owner, kind).items():
                        if value is None or (id(owner), name) in seen:
                            continue
                        if on_device and value.device.type == "cpu":
                            continue
                        seen.add((id(owner), name))
                        self._slots.append((owner, kind, name))
        host = {
            index: getattr(owner, kind)[name].detach().to("cpu").clone()
            for index, (owner, kind, name) in enumerate(self._slots)
        }
        self._host = to_mapped(host, key, directory) if directory else host
        self.on_device = on_device

    def _set(self, device) -> None:
        for index, (owner, kind, name) in enumerate(self._slots):
            value = self._host[index] if device is None else self._host[index].to(device)
            if kind == "_parameters":
                value = nn.Parameter(value, requires_grad=False)
            getattr(owner, kind)[name] = value
        self.on_device = device is not None

    def evict(self) -> None:
        if self.on_device:
            self._set(None)

    def load(self) -> None:
        if not self.on_device:
            self._set(self.device)

    @staticmethod
    @contextlib.contextmanager
    def exchange(out: "HostSwap", into: "HostSwap | None"):
        """``out`` leaves the device and ``into`` arrives for the duration; then back."""
        out.evict()
        release_device_cache()
        if into is not None:
            into.load()
        try:
            yield
        finally:
            if into is not None:
                into.evict()
            release_device_cache()
            out.load()
