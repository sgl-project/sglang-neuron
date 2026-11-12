# SPDX-License-Identifier: Apache-2.0
"""HiCache host memory: the ranks of a node share one budget, read once before any of them builds a host pool.

SGLang checks each host pool, as it allocates, against ``pool_host.base.host_memory_budget_bytes``: (MemAvailable -
reserve) / ranks on the node, read at that moment. The ranks of a node build their host pools at the same time, so a
rank that allocates after others counts their pools as used memory, and with many ranks on a node an explicit tier
(``--hicache-ratio``, ``--hicache-size``) well under the node's free memory is refused on the ranks that allocate
last. An equal share a rank is also too small for a rank that needs more than the others (a pipelined model's
drafter stage keeps the drafter rows in its host pool too), while the others leave theirs unused.

For an explicit tier, ``node_budget_scope`` (around ``hicache_auto_size.auto_size_hicache``, the scope SGLang builds
every host pool in) reads the node's free memory on every rank before any rank builds a host pool (after the weights
and device pools), takes the world's minimum, and puts it in one counter per node (a file in ``/dev/shm`` that every
rank of the node holds open; it is unlinked once all have opened it, so nothing is left behind). Each host pool's
check (``node_budget_bytes``, around ``host_memory_budget_bytes``) then books its bytes against that counter under a
lock: a pool fits while the node has room, whichever rank it is on. A tier that does not fit is refused by SGLang's
own check, which names the bytes the node has left. A query without bytes returns the rank's equal share of what is
left, as upstream's. If the counter cannot be set up on some node, every rank books against an equal share of the
same snapshot instead. Auto-sizing (a host memory fraction) and launches without HiCache keep upstream's paths.

The replicas of a ``--dp-size`` server (without DP attention) share the node, but each runs in its own world, whose
counter does not see the other replicas' pools. So each replica books against an equal part of the node: the free
memory it reads divided by the number of replicas. A replica that reads after another one has built its pools counts
them as used memory, so it gets less than an equal part, never more."""

from __future__ import annotations

import fcntl
import logging
import os
import secrets
import struct
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_AUTO_SIZE = "sglang.srt.mem_cache.hicache_auto_size"
_BASE = "sglang.srt.mem_cache.pool_host.base"
_SHM_DIR = "/dev/shm"
_COUNTER = struct.Struct("<q")

# (counter fd, ranks on this host) while this rank builds its host pools for an explicit tier; None otherwise
_NODE: ContextVar[Optional[Tuple[int, int]]] = ContextVar("neuron_hicache_node_budget", default=None)


def snapshot_host_budget(original, *args, enabled, **kwargs):
    """AROUND hicache_auto_size.auto_size_hicache: with HiCache on and an explicit size (auto-sizing off: no host
    memory fraction), the node budget scope; else upstream's."""
    from sglang.srt.runtime_context import get_memory

    if not enabled or get_memory().hicache_host_memory_fraction is not None:
        return original(*args, enabled=enabled, **kwargs)
    return node_budget_scope()


def node_budget_bytes(original, requested_bytes: int = 0, *, auto_size: bool = False) -> int:
    """AROUND pool_host.base.host_memory_budget_bytes. Inside the node budget scope: book ``requested_bytes`` against
    the node's counter when they fit, and return what the node had before (the caller refuses a pool larger than
    that); a query (no bytes) returns this rank's equal share of what is left. Elsewhere: upstream's."""
    node = _NODE.get()
    if node is None or auto_size:
        return original(requested_bytes, auto_size=auto_size)
    fd, per_host = node
    requested = int(requested_bytes)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        (left,) = _COUNTER.unpack(os.pread(fd, _COUNTER.size, 0))
        if requested <= 0:
            return left // per_host
        if requested <= left:
            os.pwrite(fd, _COUNTER.pack(left - requested), 0)
        return left
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)


def _world_group():
    from sglang.srt.runtime_context import get_parallel

    return get_parallel().world_group.cpu_group


def _replicas() -> int:
    """The server's --dp-size replicas, each its own world (DP attention runs inside one replica)."""
    from sglang_neuron import sglang_tree

    return max(int(sglang_tree.dp_widths(sglang_tree.parallel_config())[0]), 1)


def _all(flag: bool, group) -> bool:
    """Whether ``flag`` holds on every rank (one collective)."""
    import torch

    value = torch.tensor([1 if flag else 0], dtype=torch.int64)
    torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.MIN, group=group)
    return bool(value.item())


def _create(path: str, free: int) -> bool:
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            os.write(fd, _COUNTER.pack(free))
        finally:
            os.close(fd)
        return True
    except OSError as e:
        logger.warning("[neuron hicache] cannot create the node's host budget counter %s: %s", path, e)
        return False


def _open(path: str) -> Optional[int]:
    try:
        return os.open(path, os.O_RDWR)
    except OSError as e:
        logger.warning("[neuron hicache] cannot open the node's host budget counter %s: %s", path, e)
        return None


@contextmanager
def node_budget_scope():
    import torch

    from sglang.srt.mem_cache.pool_host import base

    free = int(base.available_host_memory_bytes(allow_cgroup_fallback=True)) - base.HICACHE_HOST_MEMORY_RESERVE_BYTES
    # this replica's part of the node: the other replicas' worlds book against counters of their own
    replicas = _replicas()
    free //= replicas
    dist = torch.distributed
    if not dist.is_initialized():
        with base.host_memory_budget_scope(max(free, 0)):
            yield
        return
    group = _world_group()
    rank = int(dist.get_rank())
    per_host = max(int(base.ranks_per_host()), 1)
    node, local = divmod(rank, per_host)
    # one collective before any host pool is built, as auto_size_hicache: every rank has read the node's free memory
    # before any rank allocates; the world books against the smallest, under a name rank 0 draws for this launch
    token = secrets.randbits(62) + 1 if rank == 0 else 0
    value = torch.tensor([free, -token], dtype=torch.int64)
    dist.all_reduce(value, op=dist.ReduceOp.MIN, group=group)
    free, token = max(int(value[0].item()), 0), -int(value[1].item())
    path = os.path.join(_SHM_DIR, f"sglang-neuron-hicache-budget-{token:x}-{node}")
    created = _create(path, free) if local == 0 else True
    fd = _open(path) if _all(created, group) else None
    shared = _all(fd is not None, group)
    if local == 0 and created:
        try:
            os.unlink(path)  # every rank of the node holds it open by now
        except OSError:
            pass
    if not shared:
        if fd is not None:
            os.close(fd)
        if rank == 0:
            logger.warning("[neuron hicache] no shared host budget on some node; each rank books against an equal "
                           "share, %.1f GiB", free / per_host / 1024**3)
        with base.host_memory_budget_scope(free // per_host):
            yield
        return
    if local == 0:
        logger.info("[neuron hicache] the %d ranks on this host share %.1f GiB for their host pools (read before any "
                    "host pool is built%s)", per_host, free / 1024**3,
                    f"; 1/{replicas} of the node, this --dp-size replica's part" if replicas > 1 else "")
    reset = _NODE.set((fd, per_host))
    try:
        yield
    finally:
        _NODE.reset(reset)
        os.close(fd)


# Wrapped: auto_size_hicache, host_memory_budget_bytes; relied on: SGLang building every HiCache host pool inside
# auto_size_hicache (build_kv_cache), its return to the caller for an explicit size, a host pool checking its bytes
# against host_memory_budget_bytes (refused when the bytes exceed what it returns), the free-memory reading and the
# ranks a host the budget is split by. Pinned by md5 against SGLang 43881a7.
_SOURCES_MD5 = {
    (_AUTO_SIZE, "auto_size_hicache"): "dc3bbd20bf9b34578ab4a9965ed87c41",
    ("sglang.srt.mem_cache.kv_cache_builder", "build_kv_cache"): "838c1a921a6565db59b783e06d2861dc",
    (_BASE, "host_memory_budget_bytes"): "851f20eab5519a385cd3979e6190fcf6",
    (_BASE, "host_memory_budget_scope"): "bcfc19fc73a940fe039ff935090a4702",
    (_BASE, "HostKVCache._allocate_host_buffer"): "99f775ea554d0ca042eab30bebefe679",
    (_BASE, "ranks_per_host"): "fe9967cc1eba605696f8c08ef028fd96",
    ("sglang.srt.mem_cache.host_memory", "available_host_memory_bytes"): "adbb53ee4bf5f3c2b5cd1eb54838a798",
}


def register_hooks(registry, hook_type) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_SOURCES_MD5, __name__)
    registry.register(f"{_AUTO_SIZE}.auto_size_hicache", snapshot_host_budget, hook_type.AROUND)
    registry.register(f"{_BASE}.host_memory_budget_bytes", node_budget_bytes, hook_type.AROUND)
