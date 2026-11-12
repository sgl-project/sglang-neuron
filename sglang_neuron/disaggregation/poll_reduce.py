# SPDX-License-Identifier: Apache-2.0
"""The disaggregation poll and flag reduces through shared memory. They change how the host waits, never a transferred
byte.

Every iteration SGLang MIN-reduces the bootstrap, transfer and prealloc poll states over the attention-TP group with a
gloo all_reduce (``_all_reduce_polls``), and its decode agrees on per-request flags the same way (``all_reduce_flags``,
MIN or MAX). A gloo ring lets the ranks out one after another, so the last ones reach the next forward's first
collective late. An attention-TP group runs on one node (a server over several nodes keeps each inside one), so each
reduce can go through a shared-memory segment of the group instead (one per group and operation), and the ranks leave
together. The serving policy turns this on for a decode; elsewhere, for a group over nodes, or on a host without
shared-memory support, SGLang's reduces run.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_SEGMENTS: dict = {}  # (id(group), what) -> its shared-memory segment
# the wrapped upstream functions, md5-pinned against upstream SGLang 43881a7 (source_guard): the hooks rely on their
# contracts, a MIN of a list of ints (_all_reduce_polls) and a MIN or MAX of a same-length list of small ints, returned
# unchanged when empty or on one rank (all_reduce_flags, which _all_reduce_polls calls)
_SOURCES_MD5 = {
    ("sglang.srt.disaggregation.utils", "_all_reduce_polls"): "c4240054bd74dfd3885756f7953f7427",
    ("sglang.srt.disaggregation.utils", "all_reduce_flags"): "7a6299fea1c3571aba1d517d3feca38e",
}


def _shm_reduce() -> bool:
    from sglang_neuron import serving_policy
    from sglang_neuron.distributed.node_shm_min import shm_supported

    return serving_policy.current().node_local_poll_shm and shm_supported()


def _on_one_node(ranks) -> bool:
    """Whether a group's ranks run on one node, where its segment reaches them all. On a server over several nodes an
    attention-TP group lies inside a node (capability.DecodeLaunchLayout); a group over nodes keeps upstream's
    reduce."""
    from sglang_neuron import serving_policy

    nnodes = serving_policy.current().nnodes
    if nnodes <= 1:
        return True
    import torch.distributed as dist

    from sglang_neuron.distributed.node_shm_min import within_one_node

    return within_one_node(ranks, dist.get_world_size(), nnodes)


def _local_shm(group, ranks, what="poll", **kwargs):
    """The group's segment for ``what``, created at its first reduce: a reduce is a collective every rank of the group
    reaches in the same order, so they create it together. ``kwargs`` go to NodeShmMin."""
    import torch.distributed as dist

    from sglang_neuron.distributed.node_shm_min import NodeShmMin, group_segment_name

    key = (id(group), what)
    if key not in _SEGMENTS:
        name = group_segment_name(group, ranks[0], f"{what}{ranks[0]}x{len(ranks)}")
        _SEGMENTS[key] = NodeShmMin(name, ranks.index(dist.get_rank()), len(ranks), lambda: dist.barrier(group=group),
                                    **kwargs)
    return _SEGMENTS[key]


def _all_reduce_polls(original_fn, polls, group):
    """AROUND disaggregation.utils._all_reduce_polls: the MIN through the group's shared-memory segment. An empty list,
    which only the pipelined decode passes (Scheduler.get_rids polls its queues unguarded), keeps upstream's: every rank
    passes it together, so a segment round would only hold each rank until the last arrives."""
    import torch.distributed as dist

    if group is None or not polls or not _shm_reduce():
        return original_fn(polls, group)
    ranks = dist.get_process_group_ranks(group)
    if len(ranks) == 1:
        return [int(p) for p in polls]
    if not _on_one_node(ranks):
        return original_fn(polls, group)
    return _local_shm(group, ranks).reduce(polls)


def _all_reduce_flags(original_fn, flags, group, op):
    """AROUND disaggregation.utils.all_reduce_flags: upstream's int32 MIN or MAX through the group's segment for that
    operation, in rounds when the list exceeds its capacity. Empty lists, one-rank groups and other ops keep
    upstream's."""
    import numpy as np
    import torch.distributed as dist

    which = None
    if flags:
        which = "min" if op == dist.ReduceOp.MIN else "max" if op == dist.ReduceOp.MAX else None
    if group is None or which is None or not _shm_reduce():
        return original_fn(flags=flags, group=group, op=op)
    ranks = dist.get_process_group_ranks(group)
    if len(ranks) == 1 or not _on_one_node(ranks):
        return original_fn(flags=flags, group=group, op=op)
    segment = _local_shm(group, ranks, f"flags{which}", op=which, dtype=np.int32)
    return [int(v) for v in segment.reduce([int(f) for f in flags])]


def register_hooks(registry, around) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_SOURCES_MD5, __name__)
    registry.register("sglang.srt.disaggregation.utils._all_reduce_polls", _all_reduce_polls, around)
    registry.register("sglang.srt.disaggregation.utils.all_reduce_flags", _all_reduce_flags, around)
