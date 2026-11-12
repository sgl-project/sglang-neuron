# SPDX-License-Identifier: Apache-2.0
"""Qwen3-MoE's sparse MoE block on attention TP with expert parallelism: every rank routes every token (the router is
replicated: softmax over all experts, top-k, renormalized under norm_topk_prob), runs its share of the experts and the
TP all-reduce sums the ranks' shares. A decode runs them with the bf16 moe_tkg kernel; a prefill with moe_cte on blocks
of each expert's tokens. With fewer EP ranks than TP ranks, an expert's intermediate dimension shards over the rest."""

import torch
import torch.nn.functional as F
from torch import nn

from sglang_neuron.models.transformer.config import DTYPE
from sglang_neuron.ops.moe.moe_cte_bf16 import moe_experts_blockwise
from sglang_neuron.ops.moe.moe_tkg_bf16 import moe_experts
from sglang_neuron.parallel import collectives as neuron_collectives
from sglang_neuron.parallel import groups
from sglang_neuron.parallel.state import STATE
from sglang_neuron.utils.weight_loader import SafetensorsWeightLoader, set_weight_loader


class SparseMoE(nn.Module):
    def __init__(self, *, hidden_size: int, num_experts: int, top_k: int, intermediate_size: int, norm_topk_prob: bool,
                 prefill: bool):
        super().__init__()
        self.prefill = prefill
        self.attn_tp_group = groups.get_neuron_attn_tp_group()
        if STATE.dp_size > 1:
            raise ValueError("Qwen3-MoE's experts are served on attention TP without DP attention; launch without "
                             "--enable-dp-attention")
        ep = STATE.ep_degree
        tp = self.attn_tp_group.world_size
        if num_experts % ep or tp % ep:
            raise ValueError(f"{num_experts} experts do not split over an EP size of {ep} within an attention TP of {tp}")
        expert_tp = tp // ep
        if intermediate_size % expert_tp:
            raise ValueError(f"an expert intermediate size of {intermediate_size} does not shard over {expert_tp} ranks")
        self.num_experts, self.top_k, self.norm_topk_prob = num_experts, top_k, norm_topk_prob
        self.num_local_experts = num_experts // ep
        rows = intermediate_size // expert_tp
        if ep > 1:
            # held for its rank tensor, a graph input, so every rank traces one graph
            self.ep_group = groups.get_neuron_ep_group()
            ep_rank = self.ep_group.rank_in_group
            expert_tp_rank = groups.get_neuron_ep_tp_group().rank_in_group if expert_tp > 1 else 0
        else:  # no EP: every expert splits its intermediate dimension over the whole attention TP
            self.ep_group, ep_rank, expert_tp_rank = None, 0, self.attn_tp_group.rank_in_group
        self.local_experts = range(ep_rank * self.num_local_experts, (ep_rank + 1) * self.num_local_experts)
        self.router_weight = nn.Parameter(torch.empty(num_experts, hidden_size, dtype=DTYPE))
        self.gate_up_weight = nn.Parameter(torch.empty(self.num_local_experts, hidden_size, 2, rows, dtype=DTYPE))
        self.down_weight = nn.Parameter(torch.empty(self.num_local_experts, rows, hidden_size, dtype=DTYPE))
        set_weight_loader(self.gate_up_weight, _gate_up_loader(self.num_local_experts, rows, expert_tp_rank))
        set_weight_loader(self.down_weight, _down_loader(rows, expert_tp_rank))

    def _ep_rank_id(self) -> torch.Tensor:
        if self.ep_group is None:
            return torch.zeros((1, 1), dtype=torch.int32, device=self.gate_up_weight.device)
        return self.ep_group.rank_in_group_tensor.reshape(1, 1).to(torch.int32)

    def forward(self, x, real=None):
        """The block's output for x [T, H]. ``real`` [T] bool marks a prefill chunk's real tokens: moe_cte maps
        only those (ops/moe/moe_cte_bf16.moe_experts_blockwise). The decode ignores it."""
        probs = F.softmax((x @ self.router_weight.t()).float(), dim=-1)
        weights, experts = torch.topk(probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        affinities = torch.zeros(x.shape[0], self.num_experts, dtype=torch.float32, device=x.device)
        affinities = affinities.scatter(1, experts, weights.to(x.dtype).float())
        if self.prefill:
            # this rank's experts' columns, offset by the rank tensor so that every rank traces one graph
            local = (torch.arange(self.num_local_experts, device=x.device, dtype=torch.int32)
                     + self._ep_rank_id().reshape(1) * self.num_local_experts)
            out = moe_experts_blockwise(x, gate_up=self.gate_up_weight, down=self.down_weight,
                                        affinities=affinities.index_select(1, local), top_k=self.top_k,
                                        padding_mask=real)
        else:
            out = moe_experts(x, gate_up=self.gate_up_weight, down=self.down_weight, affinities=affinities,
                              index=experts.to(torch.int32), rank_id=self._ep_rank_id())
        return neuron_collectives.all_reduce(self.attn_tp_group, out)

    def weight_mappings(self, src: str, dst: str) -> dict:
        experts = f"{src}.mlp.experts"
        return {
            f"{dst}.router_weight": f"{src}.mlp.gate.weight",
            f"{dst}.gate_up_weight": ([f"{experts}.{e}.gate_proj.weight" for e in self.local_experts]
                                      + [f"{experts}.{e}.up_proj.weight" for e in self.local_experts]),
            f"{dst}.down_weight": [f"{experts}.{e}.down_proj.weight" for e in self.local_experts],
        }


def _gate_up_loader(local: int, rows: int, expert_tp_rank: int) -> SafetensorsWeightLoader:
    """The local experts' gate_proj and up_proj ([I, H] each) as [E_local, H, 2, I / expert_tp], this rank's rows."""

    def transform(slices: list, rank: int) -> torch.Tensor:
        shard = slice(expert_tp_rank * rows, (expert_tp_rank + 1) * rows)
        gate = [s[shard].t() for s in slices[:local]]
        up = [s[shard].t() for s in slices[local:]]
        return torch.stack([torch.stack([g, u], dim=1) for g, u in zip(gate, up)])

    return SafetensorsWeightLoader(transform=transform)


def _down_loader(rows: int, expert_tp_rank: int) -> SafetensorsWeightLoader:
    """The local experts' down_proj ([H, I] each) as [E_local, I / expert_tp, H], this rank's columns."""

    def transform(slices: list, rank: int) -> torch.Tensor:
        shard = slice(expert_tp_rank * rows, (expert_tp_rank + 1) * rows)
        return torch.stack([s[:, shard].t() for s in slices])

    return SafetensorsWeightLoader(transform=transform)
