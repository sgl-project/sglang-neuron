# SPDX-License-Identifier: Apache-2.0
"""An all_to_all over every rank of a Trainium3 node, for the MoE token exchange of an NKI kernel, that stays exact on
the Neuron runtime 2.35 the containers ship.

The runtime builds an ``nki.collectives.all_to_all`` whose replica group spans several devices with more than one rank
on each, such as all 128 ranks of a node at LNC1, on its latency-optimized algorithm. That algorithm relays every block
bound for another device through a slot of a relay rank's channel buffer, one slot per (sender, destination). It
refills a slot for the next tile with no read-completion sync in between, and for the next collective with none unless
the all_to_all is the last collective of the kernel's basic block. A relay that reaches the collective late is then
still forwarding a slot when its sender refills it, and the receiver gets the sender's next tile, or its next
all_to_all's block, in place of the current one; an all_to_all that is not the last collective also reports completion
before its relayed blocks land. Only an all_to_all of one tile (at most 72,817 bytes
per rank pair at the default 80 MiB channel buffer) that is the last collective of its basic block is exact, and no
fence inside a kernel protects any other.

``ep_all_to_all`` exchanges the same blocks in two all_to_alls whose groups never take that path:

1. inside each device, over its ``device_ranks`` contiguous ranks (8 at LNC1): the runtime copies each block straight
   into its peer's output;
2. across the devices, over the ranks with the same position on their devices (8 groups of 16 at LNC1, one rank per
   device): the runtime writes each block into its receiver's output, or, where the sender has no direct route to the
   receiver, into the receiver's proxy buffer, which alternates halves between consecutive collectives and which the
   receiver copies out once every write has landed.

Block copies before the first exchange and between the two put every block where the one all_to_all would: rank ``d``
receives sender ``s``'s block ``d`` as its block ``s``. The groups are rank-ordered: the two hops of the GPT-OSS
giga-kernel decode's fused tail (``hop_meshes(world, (device_ranks, devices))`` in
``sglang_neuron/ops/gpt_oss_giga_tail.py``).

Each call allocates the three ``nl.private_hbm`` tensors it stages through, named from its ``name`` (give every call
site its own), and fences them. The runtime reports an all_to_all that is not the last collective of its basic block
complete once this rank's incoming blocks land, while its own sends may still be reading its source (its
``sync_required`` is false), and the compiler may hand a tensor's HBM to another after
its last use. So after the two exchanges come two small all_to_alls over the same groups, each between the first rows of
two staging tensors whose contents nothing reads any more: the in-device group's from the first exchange's input into
its output, the across group's from that output into the second exchange's input. Writing a tensor an exchange read or
wrote orders a fence after that exchange, and reading or writing a staging tensor keeps it allocated until the fence. A
rank passes a fence's handshake only once every rank of its group has reached it, which each does only after its
exchange received every block, so this rank's sends of the exchange are done by then, inside a kernel and at its end
alike, and no rank writes a fence's rows before. A fence's own sends carry only bytes nothing reads.

``src`` is an HBM tensor ``[n_ranks * B, C]``, block ``b`` being rows ``[b * B, (b + 1) * B)``; ``dst`` is an
``nl.private_hbm`` tensor of its shape and dtype, since a collective cannot write a kernel's output. Everything here
runs inside the kernel's trace, so it builds its replica groups there (a ReplicaGroup cannot cross the kernel's
boundary) and uses no comprehensions, which NKI's tracer rejects. Validated on Trainium3 at LNC1 (128 ranks, 16 devices
of 8) by ``tools/collectives/all_to_all_check.py``.
"""
from __future__ import annotations

from typing import List, Tuple

try:
    import nki.collectives as ncc
    import nki.isa as nisa
    import nki.language as nl

    _NKI_IMPORT_ERR = None
except Exception as exc:  # noqa: BLE001  # pragma: no cover - Neuron image only
    _NKI_IMPORT_ERR = exc


def _devices(n_ranks: int, device_ranks: int) -> int:
    if device_ranks < 1 or n_ranks % device_ranks:
        raise ValueError(f"{n_ranks} ranks do not split into devices of {device_ranks} ranks")
    return n_ranks // device_ranks


def replica_groups(n_ranks: int, device_ranks: int) -> Tuple[List[List[int]], List[List[int]]]:
    """The two exchanges' replica groups: the ``device_ranks`` contiguous ranks of each device, then the ranks with the
    same position on each device, one per device. Each group is rank-ordered."""
    devices = _devices(n_ranks, device_ranks)
    in_device = []
    for d in range(devices):
        in_device.append(list(range(d * device_ranks, (d + 1) * device_ranks)))
    across = []
    for p in range(device_ranks):
        across.append(list(range(p, n_ranks, device_ranks)))
    return in_device, across


def first_layout(n_ranks: int, device_ranks: int) -> List[int]:
    """The source block at each block of the first exchange's input: the block for in-device peer ``p`` holds the blocks
    for ranks ``(e, p)`` of every device ``e``, in device order, so it reaches the rank that forwards them."""
    devices = _devices(n_ranks, device_ranks)
    layout = []
    for p in range(device_ranks):
        for e in range(devices):
            layout.append(e * device_ranks + p)
    return layout


def second_layout(n_ranks: int, device_ranks: int) -> List[int]:
    """The first exchange's output block at each block of the second exchange's input: the block for device ``e`` holds
    what this rank received for rank ``(e, p)`` from each in-device peer, in peer order, so the second exchange's
    output is in sender order."""
    devices = _devices(n_ranks, device_ranks)
    layout = []
    for e in range(devices):
        for q in range(device_ranks):
            layout.append(q * devices + e)
    return layout


def _permute_blocks(dst, src, layout: List[int], block_rows: int) -> None:
    """dst block ``i`` = src block ``layout[i]``, one HBM-to-HBM DMA per block."""
    for i in range(len(layout)):
        j = layout[i]
        nisa.dma_copy(dst=dst[nl.ds(i * block_rows, block_rows), :], src=src[nl.ds(j * block_rows, block_rows), :])


def _fence(group, src, dst, rows: int) -> None:
    """One all_to_all over ``group`` of the first ``rows`` rows of ``src`` into the first ``rows`` rows of ``dst``, two
    staging tensors whose contents nothing reads any more (the module docstring says why)."""
    ncc.all_to_all(srcs=[src[nl.ds(0, rows), :]], dsts=[dst[nl.ds(0, rows), :]], replica_group=group, collective_dim=0)


def ep_all_to_all(src, dst, *, n_ranks: int, device_ranks: int, name: str) -> None:
    """``dst`` = the all_to_all of ``src`` over all ``n_ranks`` ranks of ``n_ranks / device_ranks`` devices, in the two
    exchanges of the module docstring, staged through three tensors this call allocates, named ``name_*``, each exchange
    followed by its fence.

    Refuses a destination of another shape or dtype than ``src`` (a DMA converts dtypes silently), and a node of one
    device or of one rank per device, whose single all_to_all never takes the relayed path."""
    if _NKI_IMPORT_ERR is not None:  # pragma: no cover - Neuron image only
        raise RuntimeError(f"ep_all_to_all needs NKI, the Neuron image's ({_NKI_IMPORT_ERR!r})")
    devices = _devices(n_ranks, device_ranks)
    if devices == 1 or device_ranks == 1:
        raise ValueError(f"{n_ranks} ranks on {devices} devices of {device_ranks}: one nki.collectives.all_to_all over "
                         "them never takes the relayed path, so call it directly")
    rows, cols = src.shape
    if rows % n_ranks:
        raise ValueError(f"{rows} rows do not split into {n_ranks} blocks")
    if tuple(dst.shape) != (rows, cols) or dst.dtype != src.dtype:
        raise ValueError(f"ep_all_to_all: a {tuple(dst.shape)} {dst.dtype} destination where the source is "
                         f"{(rows, cols)} {src.dtype}")
    # the first exchange's input and output, and the second exchange's input
    staged = nl.ndarray((rows, cols), dtype=src.dtype, buffer=nl.private_hbm, name=f"{name}_in_device_src")
    landed = nl.ndarray((rows, cols), dtype=src.dtype, buffer=nl.private_hbm, name=f"{name}_in_device_dst")
    forward = nl.ndarray((rows, cols), dtype=src.dtype, buffer=nl.private_hbm, name=f"{name}_across_src")
    in_device, across = replica_groups(n_ranks, device_ranks)
    block = rows // n_ranks
    _permute_blocks(staged, src, first_layout(n_ranks, device_ranks), block)
    ncc.all_to_all(srcs=[staged], dsts=[landed], replica_group=ncc.ReplicaGroup(in_device), collective_dim=0)
    _permute_blocks(forward, landed, second_layout(n_ranks, device_ranks), block)
    ncc.all_to_all(srcs=[forward], dsts=[dst], replica_group=ncc.ReplicaGroup(across), collective_dim=0)
    # after the second exchange, so that neither fence delays it
    _fence(ncc.ReplicaGroup(in_device), staged, landed, device_ranks)
    _fence(ncc.ReplicaGroup(across), landed, forward, devices)
