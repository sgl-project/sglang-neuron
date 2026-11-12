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

"""
Pinning of collective operands for a whole kernel. The caller turns it on (the giga kernel and the MTP draft chain:
when a rank holds idx_keys >= 65536, i.e. long context).

The allocator reuses the HBM offsets of a layer's collective buffers in later layers (in a 4-layer NEFF,
two consecutive MoE layers' EP gather destinations share one offset). A peer's write into a reused
destination, or its read of a reused source, is ordered by the collective, not by this rank's local
use of the previous occupant. Pinning gives every pinned buffer its own addresses: ``pin`` records it,
and ``close_pins`` folds a bit-exact read of each (``a | (v & 0)``) into a kernel output tile, so each
stays live to the end. Without it the 128-rank 1M decode reads reused addresses and goes NaN.

RING SLOTS. ``cc_dst_ring.py`` already gives each ringed destination (the AllGather and all_to_all destinations) an
address that only its own call site uses for the whole trace, so a ring slot is not pinned (``ring_slot``). That
leaves every source, the ReduceScatter destinations (never ringed) and the destinations that bypass the ring. A tensor is recorded once.

SBUF operands cannot be pinned: ``close_pins`` reads HBM, and a kernel-lifetime SBUF tile is what the SBUF
collectives' stale-byte defect needs. A kernel that pins keeps its collectives in HBM (``pin`` refuses SBUF).

While pinning is on, the attention collective operands are pinned too (``pin_attn``) unless GLM52K_CC_PIN_ATTN=0.
At 16 x 1M (131072) the attention pins are needed: without them MTP decoding with 16 requests gives NaN rows, which the MoE
pins alone do not prevent.

The draft chain has no layer boundary to release a window at, so every pinned operand stays live to the kernel end:
the giga kernel's attention window does not apply here.

Everything here except ``_fold`` and ``close_pins`` is Python only: with pinning off nothing is emitted
or allocated, so the trace is the unpinned one.
"""

import math
import os

import nki.isa as nisa
import nki.language as nl

from ...utils.kernel_assert import kernel_assert
from .cc_dst_ring import ring_slot

PINNED = []
_ENABLED = [False]
_ATTN = [False]
_SEEN = set()  # id() of every tensor in PINNED: each tensor is recorded once
PIN_COUNT = [0]  # buffers pinned in this process, so a trace can prove the knob applied


def open_pins(default: bool = False) -> bool:
    """Start a kernel's pin list at kernel start: pinning is on if ``default``. The attention operands are pinned
    while pinning is on, unless GLM52K_CC_PIN_ATTN=0.
    """
    PINNED.clear()
    _ENABLED[0] = bool(default)
    _ATTN[0] = _ENABLED[0] and os.environ.get("GLM52K_CC_PIN_ATTN") != "0"
    _SEEN.clear()
    return _ENABLED[0]


def pin(tensor: nl.NkiTensor) -> nl.NkiTensor:
    """Record an HBM collective operand, once, while pinning is on (see ``open_pins``). Returns ``tensor``."""
    if _ENABLED[0] and _wanted(tensor) and id(tensor) not in _SEEN:
        _SEEN.add(id(tensor))
        PINNED.append(tensor)
        PIN_COUNT[0] += 1
    return tensor


def pin_attn(*tensors: nl.NkiTensor) -> None:
    """``pin`` for attention collective operands, while the attention pins are on (see ``open_pins``)."""
    if _ATTN[0]:
        for tensor in tensors:
            pin(tensor)


def _wanted(tensor: nl.NkiTensor) -> bool:
    """Whether an operand of a collective takes a pin: not a ring slot."""
    kernel_assert(
        getattr(tensor, "buffer", None) != nl.sbuf,
        "cc_pin: an SBUF collective operand cannot be pinned (close_pins reads HBM); keep the collectives of a "
        "pinned kernel in HBM",
    )
    return not ring_slot(tensor)


def _fold(acc: nl.NkiTensor, tensor: nl.NkiTensor) -> None:
    """``acc |= tensor's first 4 bytes & 0``: exact, and it keeps ``tensor`` live up to this read."""
    head = tensor.reshape((1, math.prod(tensor.shape))).view(nl.uint8)
    word = nl.ndarray((1, 4), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.dma_copy(dst=word[0:1, 0:4], src=head[0:1, 0:4])
    zero = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=zero[0:1, 0:1], data=word.view(nl.int32)[0:1, 0:1], op0=nl.bitwise_and, operand0=0)
    nisa.tensor_tensor(dst=acc[0:1, 0:1], data1=acc[0:1, 0:1], data2=zero[0:1, 0:1], op=nl.bitwise_or)


def close_pins(anchor: nl.NkiTensor) -> None:
    """Read 4 bytes of every pinned tensor and fold them, as zero, into ``anchor[0, 0]`` (SBUF, int32-viewable)."""
    if not PINNED:
        _ENABLED[0] = _ATTN[0] = False
        _SEEN.clear()
        return
    acc = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=acc[0:1, 0:1], value=0)
    for tensor in PINNED:
        _fold(acc, tensor)
    bits = anchor.view(nl.int32)
    nisa.tensor_tensor(dst=bits[0:1, 0:1], data1=bits[0:1, 0:1], data2=acc[0:1, 0:1], op=nl.bitwise_or)
    PINNED.clear()
    _SEEN.clear()
    _ENABLED[0] = _ATTN[0] = False


__all__ = ["PIN_COUNT", "close_pins", "open_pins", "pin", "pin_attn"]
