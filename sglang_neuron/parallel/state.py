# SPDX-License-Identifier: Apache-2.0
"""Build the plugin's process groups (parallel.groups) from SGLang's world, TP and attention-TP groups.

Each group's partition is registered as its replica groups, so the collectives compiled over it lower to one SPMD
program for every rank; the rank inside a group stays a graph input (its rank tensor).

Under PP each stage is its own TP group (PP-major: stage s holds world ranks s * tp .. s * tp + tp - 1), and every
group pattern is laid over every stage, so a stage's collectives never span another stage's ranks. torch's new_group is
collective over the whole world, so every rank creates every stage's groups in the same order and keeps its own. The
DCP group is not laid over stages, which is safe only because a DCP launch refuses --pp-size
(decode_context_parallel.dcp_launch_problems).

Component DP is built only for the sizes a model host declares (ModelHost.component_dp): one TP supergroup and DP
column per size (groups.component_group_ranks), shared by the components of that size.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Mapping, Optional

import torch
import torch.distributed as dist

from sglang.srt.distributed import parallel_state as sgl_ps
from sglang.srt.distributed.parallel_state import GroupCoordinator

logger = logging.getLogger(__name__)

_INITIALIZED = False


@dataclass
class ParallelState:
    ep_degree: int = 1
    dp_size: int = 1
    # the declared component DP sizes above 1, as (component, size) pairs in groups.COMPONENTS order
    component_dp: tuple = ()


STATE = ParallelState()


def _communicators_off() -> dict:
    """Every device communicator and message queue SGLang's GroupCoordinator offers, off: the plugin's groups are gloo,
    and the device collectives are lowered inside the compiled graphs. Read from the constructor's own ``use_*``
    parameters, so every communicator SGLang adds is turned off too."""
    import inspect

    return {name: False for name in inspect.signature(GroupCoordinator.__init__).parameters if name.startswith("use_")}


def _make_group(group_ranks: List[List[int]], local_rank: int, group_name: str) -> GroupCoordinator:
    from sglang_neuron.backend import stack

    # torch-neuronx records and lowers a group's ranks in ascending order, while SGLang's rank_in_group (and the rank
    # tensor stamped from it) follows the declared order: a cell in another order would give a rank two positions in
    # one group. Refused before the group exists, so every rank raises at the same point.
    unordered = [list(cell) for cell in group_ranks if list(cell) != sorted(cell)]
    if unordered:
        raise ValueError(
            f"group {group_name!r} has ranks out of ascending order ({unordered[0]}): torch-neuronx orders a "
            "group's ranks ascending while SGLang's rank_in_group follows the declared order, so the two would "
            "disagree on each rank's position in the group")
    coordinator = GroupCoordinator(
        group_ranks=group_ranks,
        local_rank=local_rank,
        torch_distributed_backend="gloo",
        group_name=group_name,
        **_communicators_off(),
    )
    stack.register_mesh(coordinator.device_group.group_name, group_ranks)
    return coordinator


def initialize_neuron_parallel_state(*, ep_degree: int, dp_size: int,
                                     component_dp: Mapping[str, int] | None = None) -> None:
    """Build the plugin's groups for a world of SGLang's TP ranks in ``dp_size`` DP-attention replicas, with experts
    over ``ep_degree`` rows, and the component DP groups the model host declares in ``component_dp``. Idempotent for
    the same arguments."""
    global _INITIALIZED
    requested = ParallelState(ep_degree=ep_degree, dp_size=dp_size,
                              component_dp=declared_component_dp(component_dp or {}, dp_size))
    if _INITIALIZED:
        if requested != STATE:
            raise RuntimeError(f"the Neuron groups are already built for {STATE}, requested {requested}")
        return

    from sglang_neuron.backend.stack import register_mesh
    from sglang_neuron.model_executor.runtime_init import neuron_device
    from sglang_neuron.parallel import groups as nps

    tp = sgl_ps.get_tp_group()
    world = sgl_ps.get_world_group()
    attn_tp = sgl_ps.get_attn_tp_group()
    local_rank = tp.local_rank
    tp_size = tp.world_size
    if dp_size < 1 or tp_size % dp_size:
        raise ValueError(f"the DP size {dp_size} must divide the TP size {tp_size}")
    attn_tp_size = tp_size // dp_size
    if attn_tp.world_size != attn_tp_size:
        raise ValueError(f"SGLang's attention TP has {attn_tp.world_size} ranks, but a TP size of {tp_size} over "
                         f"{dp_size} DP replicas needs {attn_tp_size}: launch with --enable-dp-attention --dp-size "
                         f"{dp_size}")
    stages = stage_tp_ranks(tp, sgl_ps.get_pp_group())

    def over_stages(pattern):
        """A partition of TP positions (0 .. tp_size - 1), one cell per position list and stage, in world ranks."""
        return [[stage[r] for r in cell] for cell in pattern for stage in stages]

    attn_tp_ranks = over_stages([list(range(i, i + attn_tp_size)) for i in range(0, tp_size, attn_tp_size)])

    register_mesh(world.device_group.group_name, [list(world.ranks)])
    register_mesh(tp.device_group.group_name, stages)
    register_mesh(attn_tp.device_group.group_name, attn_tp_ranks)

    g = {"_NEURON_ATTN_TP": _make_group(attn_tp_ranks, local_rank, "neuron_attn_tp")}
    if ep_degree > 1:
        ep_tp_ranks, ep_ranks = nps.ep_group_ranks(tp_size, ep_degree)
        g["_NEURON_EP_TP"] = _make_group(over_stages(ep_tp_ranks), local_rank, "neuron_ep_tp")
        g["_NEURON_EP"] = _make_group(over_stages(ep_ranks), local_rank, "neuron_ep")
        if dp_size > 1:
            g["_WIDE_EP"] = _make_group(stages, local_rank, "wide_ep")
    if dp_size > 1:
        g["_DP"] = _make_group(over_stages(nps.dp_column_ranks(tp_size, dp_size)), local_rank, "neuron_dp")
    dcp = sgl_ps.get_dcp_group_no_assert()
    if dcp is not None and dcp.world_size > 1:
        g["_NEURON_DCP"] = _make_group(dcp_ranks(tp_size, attn_tp_size, dcp.world_size, list(dcp.ranks), stages),
                                       local_rank, "neuron_dcp")
    # component DP: one TP supergroup and DP column per declared size; a component without one shards over attention TP
    sizes = dict(requested.component_dp)
    layouts = {1: (g["_NEURON_ATTN_TP"], None)}
    for component in nps.COMPONENTS:
        size = sizes.get(component, 1)
        if size not in layouts:
            supergroups, columns = nps.component_group_ranks(tp_size, dp_size, size)
            layouts[size] = (_make_group(over_stages(supergroups), local_rank, f"neuron_component_tp{size}"),
                             _make_group(over_stages(columns), local_rank, f"neuron_component_dp{size}"))
        g[f"_NEURON_{component.upper()}_TP"], g[f"_NEURON_{component.upper()}_DP"] = layouts[size]
    nps.install_groups(g)

    # the rank tensor of every group a model may shard over (groups.shard_select)
    device = neuron_device()
    for grp in {id(x): x for x in (tp, world, attn_tp, *g.values()) if x is not None}.values():
        grp.rank_in_group_tensor = torch.tensor([grp.rank_in_group], dtype=torch.int32, device=device)

    for field, value in vars(requested).items():
        setattr(STATE, field, value)
    _INITIALIZED = True
    logger.info("[neuron] parallel groups: world %d, attention TP %d x DP %d, EP %d%s", tp_size, attn_tp_size, dp_size,
                ep_degree, "".join(f", {component} DP {size}" for component, size in requested.component_dp))


def declared_component_dp(declared: Mapping[str, int], dp_size: int) -> tuple:
    """A model host's component DP declaration as ParallelState keeps it: (component, size) pairs above 1, in
    groups.COMPONENTS order. Unknown components, sizes that do not divide the DP size and sizes that do not nest
    (dp_transition needs one to divide the other) are refused before any group exists."""
    from sglang_neuron.parallel.groups import COMPONENTS

    unknown = sorted(set(declared) - set(COMPONENTS))
    if unknown:
        raise ValueError(f"component DP is declared for {unknown}: the components that take a DP size are "
                         f"{', '.join(COMPONENTS)}")
    for component, size in declared.items():
        if size < 1 or dp_size % size:
            raise ValueError(f"the {component} component DP size {size} must divide the DP size {dp_size}: its TP "
                             "supergroup and DP column span whole DP replicas")
    sizes = sorted(set(declared.values()))
    for i, small in enumerate(sizes):
        for large in sizes[i + 1:]:
            if large % small:
                raise ValueError(f"component DP sizes {small} and {large} do not nest: a component's rows move into "
                                 "another's layout (dp_transition) only where one size divides the other")
    return tuple((component, declared[component]) for component in COMPONENTS if declared.get(component, 1) != 1)


def dcp_ranks(tp_size: int, attn_tp_size: int, size: int, sglang_group: List[int],
              stages: Optional[List[List[int]]] = None) -> List[List[int]]:
    """The partition of SGLang's DCP groups: runs of ``size`` consecutive TP ranks, each inside one attention-TP group,
    whose ranks share the KV heads, in world ranks of every pipeline stage (``stages``: each stage's TP ranks,
    stage_tp_ranks; TP positions when None). ``sglang_group`` is this rank's SGLang DCP group, which must be one of the
    runs."""
    if attn_tp_size % size:
        raise ValueError(f"--dcp-size {size} must divide the attention TP size {attn_tp_size}: a DCP group's ranks "
                         "share their KV heads, which an attention-TP group holds")
    runs = [list(range(i, i + size)) for i in range(0, tp_size, size)]
    if stages is not None:
        runs = [[stage[r] for r in run] for run in runs for stage in stages]
    if sglang_group not in runs:
        raise RuntimeError(f"SGLang's DCP group {sglang_group} is not one of the runs of {size} consecutive ranks of "
                           f"the TP group")
    return runs


def stage_tp_ranks(tp, pp) -> List[List[int]]:
    """Every pipeline stage's TP ranks, in stage order (one list, this TP group's, without PP). SGLang's pp group holds
    this TP position's rank in every stage, and a stage's TP ranks are consecutive world ranks."""
    ranks = list(tp.ranks)
    if pp is None or pp.world_size <= 1:
        return [ranks]
    position = tp.rank_in_group
    stages = [[int(first) - position + i for i in range(tp.world_size)] for first in pp.ranks]
    if stages[pp.rank_in_group] != ranks:
        raise RuntimeError(f"the TP group {ranks} is not stage {pp.rank_in_group}'s consecutive ranks "
                           f"{stages[pp.rank_in_group]} (pp group {list(pp.ranks)})")
    return stages


def tp_barrier() -> None:
    dist.barrier(group=sgl_ps.get_tp_group().cpu_group)


def world_barrier() -> None:
    """Every rank of the server, all its pipeline stages."""
    dist.barrier(group=sgl_ps.get_world_group().cpu_group)
