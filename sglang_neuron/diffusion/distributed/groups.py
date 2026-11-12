# SPDX-License-Identifier: Apache-2.0
"""Process groups the Neuron diffusion models use: world, TP, CP and CFG.

They are SGLang Diffusion's own groups (``sglang.multimodal_gen.runtime.distributed.
parallel_state``): the world, TP, SP (used as CP) and CFG ``GroupCoordinator``s that SGLang's
worker builds. The plugin builds no second set. On trn2 SGLang builds them on the
physical-mesh partitions (``parallel_state.mesh_tp_cp_cfg_groups``), because the worker setup
renames SGLang's rank order onto the mesh first (``sglang_patches.diffusion_mesh_rank_order``);
:func:`bind_groups` checks each group's ranks before the models use them.

- ``device_group``: the process group for collectives on device tensors. It has the world's
  backend (the platform's ``cpu:gloo,neuron:neuron``): torch-neuronx's ``neuron`` backend for
  device tensors, Gloo for CPU tensors. Collectives inside compiled graphs are traced into the
  NEFF, and torch-neuronx resolves their replica groups through its mesh registry
  (``parallel_state.register_replica_groups``).
- ``cpu_group``: a separate Gloo group for host-side collectives.
- ``ranks``, ``world_size``, ``rank_in_group``, ``rank``, ``local_rank``, ``all_reduce``,
  ``all_gather``, ``barrier``: from ``GroupCoordinator``; ``all_reduce`` and ``all_gather``
  call ``torch.distributed`` on ``device_group`` (SGLang's custom all-reduce is CUDA-only).
- ``rank_in_group_tensor`` (added here): the on-device rank, stamped after binding, so a shard
  offset is a graph input and every rank traces one shared graph.

Rank order. torch-neuronx records and lowers a ``new_group``'s ranks in ascending order, while
``rank_in_group`` (and the rank tensor stamped from it) follows the group's declared order; the
plugin's LLM path refuses an unsorted group for that reason (``parallel/state.py``,
``_make_group``). Half of the trn2 CP groups are not ascending (mesh column 4 is
``[12, 8, 28, 24, 44, 40, 60, 56]``). ``parallel_state.register_replica_groups`` writes every
partition into torch-neuronx's mesh registry in declared order, and only compiled graphs read
that registry, so a compiled collective agrees with ``rank_in_group``. Every order-sensitive
collective on these groups (the CP all-gather, the ring attention's permute) must therefore run
inside a compiled graph: an eager collective on a CP group would use the ascending order.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from sglang.multimodal_gen.runtime.distributed.group_coordinator import GroupCoordinator

_WORLD: GroupCoordinator | None = None
_TP: GroupCoordinator | None = None
_CP: GroupCoordinator | None = None
_CFG: GroupCoordinator | None = None


def _require(group: GroupCoordinator | None, name: str) -> GroupCoordinator:
    if group is None:
        raise AssertionError(f"Neuron {name} group is not initialized")
    return group


def get_world_group() -> GroupCoordinator:
    return _require(_WORLD, "world")


def get_tp_group() -> GroupCoordinator:
    return _require(_TP, "tensor-parallel")


def get_cp_group() -> GroupCoordinator:
    return _require(_CP, "context-parallel")


def get_cfg_group() -> GroupCoordinator:
    return _require(_CFG, "cfg-parallel")


def get_tensor_model_parallel_world_size() -> int:
    return get_tp_group().world_size


def get_tensor_model_parallel_rank() -> int:
    return get_tp_group().rank_in_group


def get_classifier_free_guidance_world_size() -> int:
    return get_cfg_group().world_size


def get_classifier_free_guidance_rank() -> int:
    return get_cfg_group().rank_in_group


def is_cfg_group_initialized() -> bool:
    return _CFG is not None


def model_parallel_is_initialized() -> bool:
    return _TP is not None


def bind_groups(
    *,
    tp_groups: list[list[int]],
    cp_groups: list[list[int]],
    cfg_groups: list[list[int]],
) -> None:
    """Use SGLang's world, TP, SP and CFG groups as the models' world, TP, CP and CFG groups.

    Call once per rank, after SGLang's worker built its groups. Each group must hold exactly
    this rank's group of the given partitions, members in the same order (``rank_in_group``
    follows it); otherwise the ``sglang_patches.diffusion_mesh_rank_order`` hook did not apply.
    """
    from sglang.multimodal_gen.runtime.distributed import parallel_state as sglang_groups

    global _WORLD, _TP, _CP, _CFG
    if _TP is not None:
        raise RuntimeError("Neuron diffusion groups are already bound")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    world = sglang_groups.get_world_group()
    tp = sglang_groups.get_tp_group()
    cp = sglang_groups.get_sp_group()
    cfg = sglang_groups.get_cfg_group()
    if list(world.ranks) != list(range(world_size)):
        raise RuntimeError(f"SGLang's world group {world.ranks} is not ranks 0..{world_size - 1}")
    for name, group, partition in (
        ("tp", tp, tp_groups),
        ("cp", cp, cp_groups),
        ("cfg", cfg, cfg_groups),
    ):
        expected = next((g for g in partition if rank in g), None)
        if expected is None or list(group.ranks) != list(expected):
            raise RuntimeError(
                f"SGLang's {name} group {group.ranks} is not the Neuron layout's {expected}. "
                "SGLang must build its groups with tp x sp = world (no --dp-size or other "
                "--sp-degree), with the sglang_patches.diffusion_mesh_rank_order hook applied, in the "
                "rank order it expects (tp-sp-pp-cfg-dp)"
            )
    for group in (world, tp, cp, cfg):
        if not hasattr(group, "rank_in_group_tensor"):
            group.rank_in_group_tensor = None
    _WORLD, _TP, _CP, _CFG = world, tp, cp, cfg


def stamp_rank_tensors(device) -> list[torch.Tensor]:
    """Put each group's rank on the device (see module docstring)."""
    stamped = []
    for group in (_WORLD, _TP, _CP, _CFG):
        if group is not None and getattr(group, "rank_in_group_tensor", None) is None:
            group.rank_in_group_tensor = torch.tensor(
                [group.rank_in_group], dtype=torch.int32, device=device
            )
            stamped.append(group.rank_in_group_tensor)
    return stamped
