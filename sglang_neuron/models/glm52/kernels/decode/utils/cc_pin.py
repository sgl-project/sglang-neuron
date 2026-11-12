# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pinning of collective operands for a whole kernel, turned on by the caller for long contexts (the giga kernel and
the MTP draft chain, when a rank holds at least 65536 indexer keys).

The allocator reuses a collective buffer's HBM offsets in later layers, and a peer's write into a reused destination
is ordered by the collective, not by this rank's use of the previous occupant. Pinning gives each recorded buffer its
own addresses for the whole kernel: ``pin`` records it, and ``close_pins`` folds an exact read of each (``a | (v &
0)``) into a kernel output so it stays live. Without it the 1M decode stack went NaN in most runs.

Ring slots (``cc_dst_ring``) already have their own addresses and are not pinned. SBUF operands cannot be pinned, so a
pinning kernel keeps its collectives in HBM. The attention operands are pinned too unless the caller or
``GLM52K_CC_PIN_ATTN=0`` says otherwise, either to the kernel end or for a window of ``attn_window`` more layers
(the giga kernel uses 2 from 262144 keys a rank, where full-depth pins do not fit in HBM). With pinning off nothing
is emitted, so the trace is the unpinned one.
"""

import math
import os

import nki.isa as nisa
import nki.language as nl

from .cc_dst_ring import ring_slot
from .kernel_assert import kernel_assert

PINNED = []
_ENABLED = [False]
_ATTN = [False]
_WINDOW = [0]
_LAYERS = []  # attention operands of the layers still inside the window, oldest first
_ACC = [None]  # SBUF int32 [1, 1] that a windowed release folds its reads into
_HOME = {}  # id(tensor) -> the list that records it (PINNED or one window layer): each tensor is recorded once
PIN_COUNT = [0]  # buffers pinned in this process, so a trace can prove the knob applied


def open_pins(default: bool = False, attn_default: bool | None = None, attn_window: int = 0) -> bool:
    """Start a kernel's pin list at kernel start: pinning is on if ``default``.

    The attention operands are pinned while pinning is on: if GLM52K_CC_PIN_ATTN=1, not if =0, else if
    ``attn_default`` (None: whenever pinning is on). They stay live ``attn_window`` more layers; 0 = to the kernel end.
    """
    PINNED.clear()
    _ENABLED[0] = bool(default)
    env_attn = os.environ.get("GLM52K_CC_PIN_ATTN")
    attn = env_attn == "1" if env_attn in ("0", "1") else (True if attn_default is None else bool(attn_default))
    _ATTN[0] = _ENABLED[0] and attn
    _WINDOW[0] = attn_window
    _LAYERS.clear()
    _ACC[0] = None
    _HOME.clear()
    return _ENABLED[0]


def pin(tensor: nl.NkiTensor) -> nl.NkiTensor:
    """Record an HBM collective operand while pinning is on (see ``open_pins``). Returns ``tensor``."""
    if _ENABLED[0] and _wanted(tensor):
        _record(tensor, PINNED)
    return tensor


def pin_attn(*tensors: nl.NkiTensor) -> None:
    """``pin`` for attention collective operands: kept for the window while pinning is on (see ``open_pins``)."""
    if not _ATTN[0]:
        return
    if _WINDOW[0] <= 0:
        for tensor in tensors:
            pin(tensor)
        return
    if not _LAYERS:
        _LAYERS.append([])
    for tensor in tensors:
        if _wanted(tensor):
            _record(tensor, _LAYERS[-1])


def next_layer() -> None:
    """Open a layer's attention pins; release the layer that leaves the window by reading it here."""
    if not _ATTN[0] or _WINDOW[0] <= 0:
        return
    _LAYERS.append([])
    while len(_LAYERS) > _WINDOW[0] + 1:
        _release(_LAYERS.pop(0))


def _wanted(tensor: nl.NkiTensor) -> bool:
    """Whether an operand of a collective takes a pin: not a ring slot."""
    kernel_assert(
        getattr(tensor, "buffer", None) != nl.sbuf,
        "cc_pin: an SBUF collective operand cannot be pinned (close_pins reads HBM); keep the collectives of a "
        "pinned kernel in HBM",
    )
    return not ring_slot(tensor)


def _record(tensor: nl.NkiTensor, home: list) -> None:
    """Record ``tensor`` in ``home`` once. One already pinned to the kernel end stays there; one in an older window
    layer moves to ``home``, so it is released after its newest use."""
    held = _HOME.get(id(tensor))
    if held is home or held is PINNED:
        return
    if held is None:
        PIN_COUNT[0] += 1
    else:
        held[:] = [other for other in held if other is not tensor]
    home.append(tensor)
    _HOME[id(tensor)] = home


def _release(layer: list) -> None:
    """Fold a window layer's operands into the running accumulator, which keeps each live up to this read."""
    if _ACC[0] is None:
        _ACC[0] = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(dst=_ACC[0][0:1, 0:1], value=0)
    _fold_batch(_ACC[0], layer)
    for tensor in layer:
        _HOME.pop(id(tensor), None)


def _fold_batch(acc: nl.NkiTensor, tensors: list) -> None:
    """``acc |= (first word of every tensor) & 0`` with one tile: a 4-byte DMA per tensor into its own column, one
    ``& 0`` over the tile, one OR of its first column into ``acc``. The ``& 0`` reads every column, so it waits for
    every DMA, and each tensor stays live up to its read. One tile, not a chain of two Vector ops per operand on one
    accumulator: ~2,200 operands at 1M made that chain a long serial tail at the kernel end."""
    if not tensors:
        return
    n = len(tensors)
    words = nl.ndarray((1, 4 * n), dtype=nl.uint8, buffer=nl.sbuf)
    for i, tensor in enumerate(tensors):
        head = tensor.reshape((1, math.prod(tensor.shape))).view(nl.uint8)
        nisa.dma_copy(dst=words[0:1, 4 * i : 4 * i + 4], src=head[0:1, 0:4])
    zero = nl.ndarray((1, n), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=zero[0:1, 0:n], data=words.view(nl.int32)[0:1, 0:n], op0=nl.bitwise_and, operand0=0)
    nisa.tensor_tensor(dst=acc[0:1, 0:1], data1=acc[0:1, 0:1], data2=zero[0:1, 0:1], op=nl.bitwise_or)


def close_pins(anchor: nl.NkiTensor) -> None:
    """Read 4 bytes of every pinned tensor and fold them, as zero, into ``anchor[0, 0]`` (SBUF, int32-viewable)."""
    windowed = [tensor for layer in _LAYERS for tensor in layer]
    if not PINNED and not windowed and _ACC[0] is None:
        _ENABLED[0] = _ATTN[0] = False
        _HOME.clear()
        return
    acc = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=acc[0:1, 0:1], value=0)
    _fold_batch(acc, PINNED + windowed)
    if _ACC[0] is not None:
        nisa.tensor_tensor(dst=acc[0:1, 0:1], data1=acc[0:1, 0:1], data2=_ACC[0][0:1, 0:1], op=nl.bitwise_or)
    bits = anchor.view(nl.int32)
    nisa.tensor_tensor(dst=bits[0:1, 0:1], data1=bits[0:1, 0:1], data2=acc[0:1, 0:1], op=nl.bitwise_or)
    PINNED.clear()
    _LAYERS.clear()
    _HOME.clear()
    _ACC[0] = None
    _ENABLED[0] = _ATTN[0] = False


__all__ = ["PIN_COUNT", "close_pins", "next_layer", "open_pins", "pin", "pin_attn"]
