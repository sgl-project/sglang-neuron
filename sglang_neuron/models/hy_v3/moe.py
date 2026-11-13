# SPDX-License-Identifier: Apache-2.0
"""Hy3's MoE block: Qwen3-MoE's expert-parallel experts (models/qwen3_moe/moe.py) behind Hy3's router, plus one
shared expert.

The router is replicated and runs in fp32: scores = sigmoid(x W_r^T); the top-k experts are chosen on scores +
expert_bias (the bias selects only), and their weights are the chosen scores, renormalized to sum to one (route_norm)
and scaled by router_scaling_factor. The shared expert is a SiLU-gated MLP on every token, its intermediate dimension
sharded over the attention TP; its partial output joins the routed experts' before the one all-reduce of the block.

With the FP8 checkpoint the routed experts keep their per-tensor FP8 weights (experts_fp8.py): the decode runs
nkilib's static-FP8 moe_tkg, and the prefill dequantizes this rank's experts in the graph for the bf16 moe_cte."""

import torch
from torch import nn

from sglang_neuron.models.hy_v3 import experts_fp8
from sglang_neuron.models.hy_v3.fp8 import dequantizing, with_scales
from sglang_neuron.models.hy_v3.mlp import ShardedMLP
from sglang_neuron.models.qwen3_moe.moe import SparseMoE
from sglang_neuron.ops.moe.moe_cte_bf16 import moe_experts_blockwise
from sglang_neuron.ops.moe.moe_tkg_bf16 import moe_experts
from sglang_neuron.parallel import collectives as neuron_collectives
from sglang_neuron.utils.weight_loader import get_weight_loader, set_weight_loader


def hy3_route(x: torch.Tensor, router_weight: torch.Tensor, expert_bias: torch.Tensor, top_k: int, route_norm: bool,
              scaling: float) -> tuple:
    """Dense [T, E] fp32 affinities (zero off the chosen experts) and the [T, K] chosen experts."""
    scores = torch.sigmoid(x.float() @ router_weight.float().t())
    _, experts = torch.topk(scores + expert_bias, top_k, dim=-1)
    chosen = scores.gather(1, experts)
    if route_norm:
        chosen = chosen / (chosen.sum(dim=-1, keepdim=True) + 1e-20)
    affinities = torch.zeros_like(scores).scatter(1, experts, chosen * scaling)
    return affinities, experts


class HyV3MoE(SparseMoE):
    def __init__(self, *, hidden_size: int, num_experts: int, top_k: int, intermediate_size: int, route_norm: bool,
                 scaling: float, shared_intermediate_size: int, fp8: bool, prefill: bool):
        super().__init__(hidden_size=hidden_size, num_experts=num_experts, top_k=top_k,
                         intermediate_size=intermediate_size, norm_topk_prob=False, prefill=prefill)
        self.route_norm, self.scaling, self.fp8 = route_norm, scaling, fp8
        self.expert_bias = nn.Parameter(torch.empty(num_experts, dtype=torch.float32))
        self.shared = ShardedMLP(hidden_size, shared_intermediate_size)
        if fp8:
            for p in (self.shared.gate_up_weight, self.shared.down_weight):
                set_weight_loader(p, dequantizing(get_weight_loader(p)))
            experts_fp8.make_fp8(self)

    def forward(self, x, real=None):
        affinities, experts = hy3_route(x, self.router_weight, self.expert_bias, self.top_k, self.route_norm,
                                        self.scaling)
        affinities = affinities.to(x.dtype).float()
        if self.prefill:
            local = (torch.arange(self.num_local_experts, device=x.device, dtype=torch.int32)
                     + self._ep_rank_id().reshape(1) * self.num_local_experts)
            gate_up, down = experts_fp8.bf16_weights(self) if self.fp8 else (self.gate_up_weight, self.down_weight)
            out = moe_experts_blockwise(x, gate_up=gate_up, down=down,
                                        affinities=affinities.index_select(1, local), top_k=self.top_k,
                                        padding_mask=real, reload_weights=True)
        elif self.fp8:
            out = experts_fp8.decode_experts(self, x, affinities, experts.to(torch.int32))
        else:
            out = moe_experts(x, gate_up=self.gate_up_weight, down=self.down_weight, affinities=affinities,
                              index=experts.to(torch.int32), rank_id=self._ep_rank_id())
        out = out + self.shared.partial(x)
        return neuron_collectives.all_reduce(self.attn_tp_group, out)

    def weight_mappings(self, src: str, dst: str) -> dict:
        mlp, experts = f"{src}.mlp", f"{src}.mlp.experts"
        local = self.local_experts
        scaled = with_scales if self.fp8 else list
        return {
            f"{dst}.router_weight": f"{mlp}.router.gate.weight",
            f"{dst}.expert_bias": f"{mlp}.expert_bias",
            f"{dst}.gate_up_weight": ([f"{experts}.{e}.gate_proj.weight" for e in local]
                                      + [f"{experts}.{e}.up_proj.weight" for e in local]),
            f"{dst}.down_weight": [f"{experts}.{e}.down_proj.weight" for e in local],
            **(experts_fp8.scale_mappings(experts, local, dst) if self.fp8 else {}),
            f"{dst}.shared.gate_up_weight": scaled([f"{mlp}.shared_mlp.gate_proj.weight",
                                                    f"{mlp}.shared_mlp.up_proj.weight"]),
            f"{dst}.shared.down_weight": scaled([f"{mlp}.shared_mlp.down_proj.weight"]),
        }
