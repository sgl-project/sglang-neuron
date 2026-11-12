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

"""Kernel-lifetime ring buffers for the destinations of a multi-layer trace's per-layer collectives.

The compiler labels a collective trigger handshake-skippable (``unique_tensors``) only when no other
tensor occupies its destination's address between it and the previous collective of its replica group.
A per-layer destination is a fresh allocation every layer, and the allocator is free to place the
neighbouring layers' short-lived tensors in the same range; when it does, the runtime runs a
pre-collective handshake with every peer of the group. Which layers that hits
depends on the allocation pattern around the site -- every other layer, or the layer after each
indexer layer -- not on the collective.

While a ring is open, :func:`cc_dst` hands each call site a fixed set of ``depth`` buffers allocated on
first use and reused in rotation, so nothing but that site's own collective ever lives at those
addresses. Outside an open ring (single-layer kernels, tests) it allocates per call as before.
"""

import nki.language as nl

from ...utils.trace_names import unique_name

CC_DST_RING = 2
"""Ring depth for the stack's collective destinations: each call site's destinations rotate over this many buffers
instead of one allocation per layer."""

_STATE = [None]


def open_cc_dst_ring(depth: int = CC_DST_RING) -> None:
    """Start rotating collective destinations for the rest of this trace (``depth`` 0 is a no-op)."""
    _STATE[0] = {"depth": depth, "uses": {}, "pool": {}} if depth > 0 else None


def close_cc_dst_ring() -> None:
    """Stop rotating; later calls allocate per call again. Drops every pooled handle."""
    _STATE[0] = None


def ring_slot(tensor: nl.NkiTensor) -> bool:
    """Whether ``tensor`` is a slot of the open ring: a kernel-lifetime buffer that only its own call site uses.

    Python only (an identity test against the pool), so asking it emits nothing. ``utils/cc_pin.py``
    leaves ring slots unpinned by default: their address already belongs to the site for the whole trace.
    """
    state = _STATE[0]
    return state is not None and any(slot is tensor for slot in state["pool"].values())


def cc_dst(shape: tuple, dtype, buffer, name: str, reduce_scatter: bool = False) -> nl.NkiTensor:
    """A collective destination: the next ring slot of this call site while a ring is open, else a fresh tensor.

    A call site is identified by ``(name, shape, dtype, buffer)``, so a helper reached from several sites
    must pass a distinct ``name`` per site (the debug names already are).

    Args:
        shape (tuple): Tensor shape.
        dtype: Element dtype.
        buffer: Memory class (an HBM class; SBUF tiles are scheduled by their own allocator).
        name (str): Debug name stem, also the call-site key.
        reduce_scatter (bool): A ReduceScatter destination, which is never ringed (AllGather and all_to_all
            destinations are).

    Returns:
        nl.NkiTensor: The destination.
    """
    state = _STATE[0]
    if state is None or reduce_scatter:
        return nl.ndarray(shape, dtype=dtype, buffer=buffer, name=unique_name(name))
    site = (name, tuple(shape), str(dtype), str(buffer))
    use = state["uses"].get(site, 0)
    state["uses"][site] = use + 1
    slot = (site, use % state["depth"])
    if slot not in state["pool"]:
        state["pool"][slot] = nl.ndarray(
            shape, dtype=dtype, buffer=buffer, name=unique_name(f"{name}_ring{use % state['depth']}")
        )
    return state["pool"][slot]
