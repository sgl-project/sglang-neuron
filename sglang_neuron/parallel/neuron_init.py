# SPDX-License-Identifier: Apache-2.0
"""Rank and device setup of a torch-neuronx process.

torch-neuronx loads a NEFF that holds collectives with this process's global rank and the world size
(``nrt_load_collectives``), and a traced collective finds this rank's replica group by the global rank. It takes both,
and this process's place among the node's ranks, from its rank state, which a ``neuron`` process group sets when the
first one is created (``torch_neuronx.distributed.backend._neuron_runtime_setup``). SGLang-Neuron's process groups are
gloo, so the plugin sets the same state itself, once, before anything starts the Neuron runtime:

1. the world size and global rank: SGLang's TP world (its TP size and this process's TP rank);
2. the node's local world size and this process's local rank, which is also its torch device index
   (``runtime_init.neuron_device``: ``neuron:<local rank>``);
3. the Neuron runtime start (``torch_neuronx._lazy_init``), with ``NEURON_RT_VISIBLE_CORES`` (this rank's one logical
   core) and ``NEURON_RT_ROOT_COMM_ID`` already in the environment (``runtime_init``);
4. a check that the process sees exactly one logical core and that its current device is ``neuron:<local rank>``.

torch-neuronx's own setup also pins the process to the NUMA node of its device. SGLang binds each scheduler to its
``--numa-node`` itself, so the plugin leaves CPU affinity to SGLang.

torch-neuronx's rank setters accept one value per process and its runtime starts once, so a second call with the same
values does nothing and a call with other values, or after the runtime started, raises.
"""

from __future__ import annotations

import logging
from typing import NamedTuple, Optional

logger = logging.getLogger(__name__)


class NeuronRanks(NamedTuple):
    rank: int
    world_size: int
    local_rank: int
    local_world_size: int


_RANKS: Optional[NeuronRanks] = None


def _validate(ranks: NeuronRanks) -> None:
    rank, world, local_rank, local_world = ranks
    if world < 1 or not 0 <= rank < world:
        raise ValueError(f"rank {rank} of world size {world}: the rank must lie in [0, world size)")
    if local_world < 1 or local_world > world or not 0 <= local_rank < local_world:
        raise ValueError(f"local rank {local_rank} of local world size {local_world} (world size {world}): the local "
                         "rank must lie in [0, local world size) and the local world size in [1, world size]")


def init_neuron_ranks(rank: int, world_size: int, local_rank: int, local_world_size: int) -> None:
    """Set torch-neuronx's rank state for this process and start the Neuron runtime on ``neuron:<local_rank>``."""
    global _RANKS
    ranks = NeuronRanks(int(rank), int(world_size), int(local_rank), int(local_world_size))
    _validate(ranks)
    if _RANKS is not None:
        if _RANKS != ranks:
            raise RuntimeError(f"torch-neuronx's ranks are already set to {_RANKS._asdict()} in this process; "
                               f"{ranks._asdict()} cannot replace them")
        return

    import torch_neuronx

    if torch_neuronx.is_neuron_runtime_initialized():
        raise RuntimeError("the Neuron runtime started before torch-neuronx's ranks were set (something used the "
                           "neuron device before ModelRunner.init_torch_distributed): its collective NEFFs would load "
                           "without this process's rank")
    c = torch_neuronx._C
    c._reset_vnc_count()
    c._set_world_size(ranks.world_size)
    c._set_rank(ranks.rank)
    c._set_local_world_size(ranks.local_world_size)
    c._set_local_device_start_index(ranks.local_rank)
    torch_neuronx._lazy_init()
    cores = int(c._vnc_count())
    if cores != 1:
        raise RuntimeError(f"the Neuron runtime of rank {ranks.rank} sees {cores} logical cores; a rank runs on one "
                           "(NEURON_RT_VISIBLE_CORES must name a single core)")
    current = int(torch_neuronx.current_device())
    if current != ranks.local_rank:
        raise RuntimeError(f"torch-neuronx's current device is neuron:{current} on local rank {ranks.local_rank}; "
                           f"the rank's device is neuron:{ranks.local_rank}")
    _RANKS = ranks
    logger.info("torch-neuronx ranks: rank %d of %d, local rank %d of %d, device neuron:%d", ranks.rank,
                ranks.world_size, ranks.local_rank, ranks.local_world_size, ranks.local_rank)


def check_torch_world(rank: int, world_size: int) -> None:
    """Raise unless the torch world SGLang created is the one ``init_neuron_ranks`` set (call it after the world is
    up): a NEFF whose replica groups name torch world ranks would otherwise run its collectives on other ranks."""
    if _RANKS is None:
        raise RuntimeError("init_neuron_ranks has not run in this process")
    if (int(rank), int(world_size)) != (_RANKS.rank, _RANKS.world_size):
        raise RuntimeError(f"torch-neuronx runs as rank {_RANKS.rank} of {_RANKS.world_size} but the torch world is "
                           f"rank {rank} of {world_size}")


def runtime_barrier() -> None:
    """A Neuron runtime barrier over the world (``nrt_barrier``), the one torch-neuronx's own setup ends with: every
    rank's runtime has joined the world's collectives bootstrap when it returns."""
    if _RANKS is None:
        raise RuntimeError("init_neuron_ranks has not run in this process")
    import torch_neuronx

    torch_neuronx._C._nrt_barrier(0, _RANKS.rank, _RANKS.world_size)


def _reset_for_tests() -> None:
    global _RANKS
    _RANKS = None
