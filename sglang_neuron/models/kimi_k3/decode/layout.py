# SPDX-License-Identifier: Apache-2.0
"""The parallel layout of the Kimi-K3 2-server decode (``dp8``).

One decode engine runs on 2 nodes of 128 LNC1 ranks: 8 replicas of attention TP32 (MLA CP32), 4 on each node, and one
world of 256 ranks for the MoE (TP and EP) and the dense MLP. Global rank ``g`` has TP rank ``g % 32``, DP rank
``g // 32`` and node ``g // 128``, so each replica is inside one node.

The decode kernels take their collective groups as rank tuples (``rank_groups``): TP group ``d`` holds the ranks of
replica ``d``, DP group ``t`` holds TP rank ``t`` of each replica, and the world group holds every rank. The kernel
wrappers build the ``ReplicaGroup`` of each from these tuples.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

RANKS_PER_NODE = 128
"""The LNC1 ranks of one trn3 decode node (16 devices of 8 cores)."""

RankGroups = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class DecodeLayout:
    """The group sizes of one decode layout.

    Attributes:
        name: The layout name.
        attention_tp: The attention TP size (KDA, MLA projections, lm_head, embedding).
        mla_cp: The MLA context-parallel size.
        moe_tp: The TP size of the MoE projections and the dense MLP.
        ep: The expert-parallel size.
        dp: The replicas.
    """

    name: str
    attention_tp: int
    mla_cp: int
    moe_tp: int
    ep: int
    dp: int

    @property
    def world_size(self) -> int:
        return self.attention_tp * self.dp

    @property
    def nnodes(self) -> int:
        """The nodes of the layout (``RANKS_PER_NODE`` ranks on each)."""
        return self.world_size // RANKS_PER_NODE

    @property
    def replicas_per_node(self) -> int:
        return RANKS_PER_NODE // self.attention_tp

    def tp_rank(self, global_rank: int) -> int:
        return global_rank % self.attention_tp

    def dp_rank(self, global_rank: int) -> int:
        return global_rank // self.attention_tp

    def node_of_rank(self, global_rank: int) -> int:
        return global_rank // RANKS_PER_NODE

    def rank_groups(self) -> tuple[RankGroups, RankGroups, RankGroups]:
        """The TP, DP and world rank groups, as global ranks.

        Example:
            >>> tp, dp, world = DP8_LAYOUT.rank_groups()
            >>> tp[7][0], tp[7][-1], dp[1][:3], len(world[0])
            (224, 255, (1, 33, 65), 256)
        """
        tp, dp = self.attention_tp, self.dp
        world = tp * dp
        tp_groups = tuple(tuple(range(d * tp, (d + 1) * tp)) for d in range(dp))
        dp_groups = tuple(tuple(t + d * tp for d in range(dp)) for t in range(tp))
        if dp == 1:
            dp_groups = (tuple(range(world)),)
        return tp_groups, dp_groups, (tuple(range(world)),)

    def groups_of_rank(self, global_rank: int) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """The TP, DP and world groups that global rank ``global_rank`` belongs to."""
        tp_groups, dp_groups, world_groups = self.rank_groups()
        dp_group = dp_groups[0] if self.dp == 1 else dp_groups[self.tp_rank(global_rank)]
        return tp_groups[self.dp_rank(global_rank)], dp_group, world_groups[0]


DP8_LAYOUT = DecodeLayout("dp8", attention_tp=32, mla_cp=32, moe_tp=256, ep=256, dp=8)
"""The 2-server decode: 8 replicas of TP32 / CP32 over 2 nodes, MoE TP and EP over the 256-rank world."""


def check_layout_rank_groups(layout: DecodeLayout, tp_groups: Sequence[Sequence[int]],
                             dp_groups: Sequence[Sequence[int]], world_groups: Sequence[Sequence[int]]) -> None:
    """Stop when rank groups do not cover the layout or a replica spans two nodes.

    The kernels run their world collectives over all ranks and their replica collectives inside one node: one world
    group of all ranks; TP groups that cover all ranks once, each of ``attention_tp`` ranks on one node; DP groups
    that cover all ranks once.

    Raises:
        ValueError: A group does not follow the layout; the message names the group.

    Example:
        >>> check_layout_rank_groups(DP8_LAYOUT, *DP8_LAYOUT.rank_groups())
    """
    ranks = list(range(layout.world_size))
    if [sorted(group) for group in world_groups] != [ranks]:
        raise ValueError(f"The Kimi-K3 decode layout {layout.name} needs one world group of the {layout.world_size} "
                         f"ranks of {layout.nnodes} node(s).")
    for name, groups in (("TP", tp_groups), ("DP", dp_groups)):
        if sorted(rank for group in groups for rank in group) != ranks:
            raise ValueError(f"The {name} groups of the Kimi-K3 decode layout {layout.name} do not cover the "
                             f"{layout.world_size} ranks one time (TP rank g % TP, DP rank g // TP).")
    for group in tp_groups:
        nodes = sorted({layout.node_of_rank(rank) for rank in group})
        if len(group) != layout.attention_tp or len(nodes) != 1:
            raise ValueError(f"A TP group of the Kimi-K3 decode layout {layout.name} has {len(group)} ranks on the "
                             f"nodes {nodes}; each replica needs {layout.attention_tp} ranks on one node of "
                             f"{RANKS_PER_NODE} ranks.")


def check_rank_groups_of_rank(layout: DecodeLayout, global_rank: int, tp_ranks: Sequence[int],
                              dp_ranks: Sequence[int], world_ranks: Sequence[int]) -> None:
    """Stop when the process groups of one rank are not the groups of the layout.

    Raises:
        ValueError: A group differs from the layout; the message names the group.

    Example:
        >>> check_rank_groups_of_rank(DP8_LAYOUT, 130, range(128, 160), range(2, 256, 32), range(256))
    """
    check_layout_rank_groups(layout, *layout.rank_groups())
    expected = dict(zip(("TP", "DP", "world"), layout.groups_of_rank(global_rank)))
    got = {"TP": tp_ranks, "DP": dp_ranks, "world": world_ranks}
    for name, ranks in got.items():
        ranks = tuple(int(rank) for rank in ranks)
        if ranks != expected[name]:
            raise ValueError(f"The {name} group of rank {global_rank} starts with {list(ranks[:4])} and has "
                             f"{len(ranks)} ranks; the Kimi-K3 decode layout {layout.name} needs "
                             f"{len(expected[name])} ranks from {expected[name][0]}.")
