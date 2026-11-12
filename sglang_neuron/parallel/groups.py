# SPDX-License-Identifier: Apache-2.0
"""The plugin's process groups, which the models' compiled collectives run over.

parallel.state builds them from SGLang's world, TP and attention-TP groups; each is a gloo GroupCoordinator whose device
collectives are lowered inside the compiled graphs.

Component DP: a model host may shard its embedding, LM head or MLP over the attention-TP ranks of several consecutive DP
replicas (ModelHost.component_dp). Its weights split over that TP supergroup, and it runs on the rows of every replica
in its DP column, which dp_transition gathers and splits back.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from sglang.srt.distributed.parallel_state import GroupCoordinator

# this rank's DP replica's attention-TP ranks: attention, embedding, LM head and the dense layers shard over it
_NEURON_ATTN_TP: GroupCoordinator | None = None
# the expert-parallel mesh: rows of world_size / ep_degree consecutive ranks (EP-TP) and its columns (EP)
_NEURON_EP_TP: GroupCoordinator | None = None
_NEURON_EP: GroupCoordinator | None = None
# every rank of the server, for the MoE output's all-reduce when the experts span the DP replicas
_WIDE_EP: GroupCoordinator | None = None
# the same attention-TP position in every DP replica
_DP: GroupCoordinator | None = None
# SGLang's DCP group (--dcp-size): consecutive attention-TP ranks that hold the same KV head and different tokens of
# each sequence; the decode kernel merges its attention over it
_NEURON_DCP: GroupCoordinator | None = None
# component DP: each component's TP supergroup and DP column; without a declared size, attention TP and None
_NEURON_EMBEDDING_TP: GroupCoordinator | None = None
_NEURON_EMBEDDING_DP: GroupCoordinator | None = None
_NEURON_LM_HEAD_TP: GroupCoordinator | None = None
_NEURON_LM_HEAD_DP: GroupCoordinator | None = None
_NEURON_MLP_TP: GroupCoordinator | None = None
_NEURON_MLP_DP: GroupCoordinator | None = None
COMPONENTS = ("embedding", "lm_head", "mlp")

_GROUP_GLOBAL_NAMES = ("_NEURON_ATTN_TP", "_NEURON_EP_TP", "_NEURON_EP", "_WIDE_EP", "_DP", "_NEURON_DCP",
                       "_NEURON_EMBEDDING_TP", "_NEURON_EMBEDDING_DP", "_NEURON_LM_HEAD_TP", "_NEURON_LM_HEAD_DP",
                       "_NEURON_MLP_TP", "_NEURON_MLP_DP")


def _is_trn2() -> bool:
    """Whether this host's platform target is in the Trainium2 family (targets carry revision suffixes)."""
    from sglang_neuron.backend.stack import get_platform_target, normalize_target_family

    return normalize_target_family(get_platform_target()) == "trn2"


def ep_group_ranks(world_size: int, ep_degree: int) -> tuple[list[list[int]], list[list[int]]]:
    """(EP-TP rows, EP columns) of the expert-parallel mesh: world_size=8, ep_degree=2 gives the rows
    [[0, 1, 2, 3], [4, 5, 6, 7]] and the columns [[0, 4], [1, 5], [2, 6], [3, 7]].

    A Trainium2 host (any revision) with a 64-rank 8x8 setup is refused: its 2D torus places a row's ranks
    non-contiguously, so its columns (e.g. [12, 8, 28, ...]) are not ascending, and torch-neuronx orders a group's
    ranks ascending while SGLang's rank_in_group follows the declared order (parallel.state._make_group)."""
    if ep_degree < 1 or world_size % ep_degree:
        raise ValueError(f"the EP degree {ep_degree} must divide the world size {world_size}")
    row = world_size // ep_degree
    if world_size == 64 and row == 8 and _is_trn2():
        raise ValueError(
            "the Trainium2 64-rank 8x8 EP mesh is not served: its torus columns are not in ascending rank order, "
            "and torch-neuronx orders a group's ranks ascending while SGLang's rank_in_group follows the declared "
            "order, so the two would disagree on each rank's position in the group (choose another EP degree)")
    rows = [list(range(r * row, (r + 1) * row)) for r in range(ep_degree)]
    return rows, [[rows[r][c] for r in range(ep_degree)] for c in range(row)]


def dp_column_ranks(world_size: int, dp_size: int) -> list[list[int]]:
    """The ranks at each attention-TP position across the DP replicas: world_size=8, dp_size=2 gives
    [[0, 4], [1, 5], [2, 6], [3, 7]]."""
    if dp_size < 1 or world_size % dp_size:
        raise ValueError(f"the DP size {dp_size} must divide the world size {world_size}")
    tp = world_size // dp_size
    return [[d * tp + p for d in range(dp_size)] for p in range(tp)]


def component_group_ranks(world_size: int, dp_size: int, component_dp: int) -> tuple[list[list[int]], list[list[int]]]:
    """(TP supergroups, DP columns) of a component over runs of ``component_dp`` consecutive DP replicas: a supergroup
    is one run's attention-TP ranks, replica after replica, and a column the run's ranks at one attention-TP position.
    world_size=64, dp_size=8, component_dp=4 gives the supergroups [0..31] and [32..63] and the columns [0, 8, 16, 24],
    [1, 9, 17, 25], ..., [32, 40, 48, 56], ..., [39, 47, 55, 63]."""
    if dp_size < 1 or world_size % dp_size:
        raise ValueError(f"the DP size {dp_size} must divide the world size {world_size}")
    if component_dp < 1 or dp_size % component_dp:
        raise ValueError(f"the component DP size {component_dp} must divide the DP size {dp_size}: a component's "
                         "supergroup spans whole DP replicas")
    tp = world_size // dp_size
    starts = range(0, world_size, tp * component_dp)
    return ([list(range(s, s + tp * component_dp)) for s in starts],
            [[s + d * tp + p for d in range(component_dp)] for s in starts for p in range(tp)])


def install_groups(groups: dict) -> None:
    """Install the groups parallel.state built, keyed by this module's global names."""
    unknown = set(groups) - set(_GROUP_GLOBAL_NAMES)
    if unknown:
        raise KeyError(f"unknown group globals {sorted(unknown)}; expected {_GROUP_GLOBAL_NAMES}")
    globals().update(groups)


def shard_select(x: "torch.Tensor", dim: int, group: "GroupCoordinator", n_shards: int | None = None) -> "torch.Tensor":
    """``x``'s shard for this rank of ``group`` along ``dim``, one of ``n_shards`` (default: the group's size).

    The offset comes from the group's rank tensor, a graph input, so every rank traces the same graph; a Python rank
    would bake a per-rank constant into each rank's NEFF. The rank tensor stays 1-D: under chunked prefill
    torch.compile folds a 0-D one into a Python scalar."""
    import torch

    n = n_shards if n_shards is not None else group.world_size
    if x.shape[dim] % n:
        raise ValueError(f"shard_select: dim {dim} has size {x.shape[dim]}, not divisible into {n} shards")
    chunk = x.shape[dim] // n
    offset = group.rank_in_group_tensor.to(torch.int64) * chunk
    return x.index_select(dim, torch.arange(chunk, device=x.device, dtype=torch.int64) + offset)


def dp_transition(x: "torch.Tensor", source: "GroupCoordinator | None", target: "GroupCoordinator | None",
                  dim: int = 0) -> "torch.Tensor":
    """Move ``x``'s rows along ``dim`` between layouts. A layout is a component's DP column (get_neuron_*_dp_group),
    whose ranks each hold every replica's rows in replica order, or None: this replica's own rows, as in attention.

    Narrowing keeps the rows of the narrower column, which nests in the wider one; widening keeps this replica's rows
    and all-gathers them over the target column. The offsets come from the source column's rank tensor, a graph input,
    as in shard_select."""
    import torch

    from sglang_neuron.parallel import collectives

    width = 1 if source is None else source.world_size
    target_width = 1 if target is None else target.world_size
    if width == target_width:
        return x
    if max(width, target_width) % min(width, target_width):
        raise ValueError(f"dp_transition: DP columns of {width} and {target_width} replicas do not nest; one "
                         "component's DP size must divide the other's")
    if x.shape[dim] % width:
        raise ValueError(f"dp_transition: dim {dim} has size {x.shape[dim]}, not {width} replicas' rows")
    if width < target_width:
        return collectives.all_gather(target, x if source is None else shard_select(x, dim, source), dim=dim)
    rows = x.shape[dim] // width * target_width
    start = source.rank_in_group_tensor.to(torch.int64) // target_width * rows
    return x.index_select(dim, torch.arange(rows, device=x.device, dtype=torch.int64) + start)


# The getters run inside traced model forwards, so each reads its module global directly: dynamo traces a global load,
# but not a call of globals().
def _built(group, name: str, when: str = "") -> GroupCoordinator:
    if group is None:
        raise RuntimeError(f"the plugin group {name} is not built{when}: parallel.state.initialize_neuron_parallel_state "
                           "builds it")
    return group


def get_neuron_attn_tp_group() -> GroupCoordinator:
    return _built(_NEURON_ATTN_TP, "_NEURON_ATTN_TP")


def get_neuron_ep_tp_group() -> GroupCoordinator:
    return _built(_NEURON_EP_TP, "_NEURON_EP_TP", " (it exists with an EP degree above 1)")


def get_neuron_ep_group() -> GroupCoordinator:
    return _built(_NEURON_EP, "_NEURON_EP", " (it exists with an EP degree above 1)")


def get_neuron_ep_rank() -> int:
    """This rank's row of the expert-parallel mesh (0 without EP)."""
    return 0 if _NEURON_EP is None else _NEURON_EP.rank_in_group


def get_wide_ep_group() -> GroupCoordinator:
    return _built(_WIDE_EP, "_WIDE_EP", " (it exists with an EP degree and a DP size above 1)")


def get_dp_group() -> GroupCoordinator | None:
    """The DP column group, or None with one DP replica."""
    return _DP


def get_neuron_dcp_group() -> GroupCoordinator | None:
    """This rank's DCP group, or None without decode context parallelism."""
    return _NEURON_DCP


# component DP: a component's TP supergroup and its DP column (None without a declared size)
def get_neuron_embedding_tp_group() -> GroupCoordinator:
    return _built(_NEURON_EMBEDDING_TP, "_NEURON_EMBEDDING_TP")


def get_neuron_embedding_dp_group() -> GroupCoordinator | None:
    return _NEURON_EMBEDDING_DP


def get_neuron_lm_head_tp_group() -> GroupCoordinator:
    return _built(_NEURON_LM_HEAD_TP, "_NEURON_LM_HEAD_TP")


def get_neuron_lm_head_dp_group() -> GroupCoordinator | None:
    return _NEURON_LM_HEAD_DP


def get_neuron_mlp_tp_group() -> GroupCoordinator:
    return _built(_NEURON_MLP_TP, "_NEURON_MLP_TP")


def get_neuron_mlp_dp_group() -> GroupCoordinator | None:
    return _NEURON_MLP_DP


def destroy_neuron_parallel_state() -> None:
    """Destroy the installed groups and reset parallel.state, for tests that build several worlds in one process."""
    g = globals()
    for group in {id(g[n]): g[n] for n in _GROUP_GLOBAL_NAMES if g[n] is not None}.values():
        group.destroy()
    g.update(dict.fromkeys(_GROUP_GLOBAL_NAMES))
    from sglang_neuron.parallel import state

    state._INITIALIZED = False
    for field, value in vars(state.ParallelState()).items():
        setattr(state.STATE, field, value)
