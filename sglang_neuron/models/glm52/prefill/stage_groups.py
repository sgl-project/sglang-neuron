# SPDX-License-Identifier: Apache-2.0
"""The process groups a GLM-5.2 prefill stage graph runs its out-of-kernel collectives over.

The giga kernel's own collectives name static replica groups (StageLayout.replica_groups). Outside it the stage graph
runs three collectives of its own over the stage's 32 ranks: the embedding's reduce-scatter, the sampled row's
all-reduce and the greedy selection's all-gather over the LM head's vocabulary shards. It also reads two rank tensors,
the CP rank and the EP rank, as graph inputs, so every rank traces one graph. The groups are the plugin's (from_plugin),
injected so the CPU tests can give the graph their own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator


@dataclass(frozen=True)
class StageGroups:
    """The stage's CP group (its 32 ranks: the embedding, the sampled row, the LM head's greedy selection) and its EP
    group (the MoE's expert-parallel ranks, whose rank tensor is the kernel's ep_rank). At EP32 over a 32-rank stage the
    two hold the same ranks; they stay two groups with two rank tensors because the server's plugin groups are two
    (attention TP and EP), and a graph handed one tensor for both would trace one input fewer than the server's."""

    cp: "GroupCoordinator"
    ep: "GroupCoordinator"

    @classmethod
    def from_plugin(cls) -> "StageGroups":
        """The server's: the plugin's attention-TP group (a stage's TP group under PP) and its EP group, which
        parallel.state.initialize_neuron_parallel_state built with their rank tensors on the device."""
        from sglang_neuron.parallel import groups

        return cls(cp=groups.get_neuron_attn_tp_group(), ep=groups.get_neuron_ep_group())
