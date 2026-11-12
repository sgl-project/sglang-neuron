# SPDX-License-Identifier: Apache-2.0
"""The GPT-OSS BF16 graphs (SGLANG_NEURON_GPT_OSS_EXPERTS=bf16): the MXFP4 graphs with BF16 experts dequantized on the
host from the checkpoint's MXFP4 blocks (weight_loaders_bf16). Everything but the experts' weights and kernels is the
MXFP4 graphs' own, in the same shuffled hidden order. The decode MoE is moe_block_tkg without scales; the prefill
routes in torch, then runs moe_cte's BF16 block-sharded form with GPT-OSS's SwiGLU."""

import torch
from torch import nn

import sglang_neuron.ops as NF
from sglang_neuron.models.gpt_oss.config import DTYPE
from sglang_neuron.models.gpt_oss.mxfp4 import common, model_decode, model_prefill
from sglang_neuron.models.gpt_oss.weight_loaders_bf16 import (
    bf16_down_weight_loader,
    bf16_gate_up_bias_loader,
    bf16_gate_up_weight_loader,
)
from sglang_neuron.models.gpt_oss.weight_loaders_mxfp4 import decode_router_weight_loader, mxfp4_down_bias_loader
from sglang_neuron.ops.moe.moe_cte_bf16 import BLOCK_SIZE, moe_experts_blockwise_swiglu
from sglang_neuron.parallel import collectives as neuron_collectives
from sglang_neuron.utils.weight_loader import (
    ShardSpec,
    expert_parallel_tensor_dim_loader,
    set_weight_loader,
    with_rank_override,
)


class Bf16ExpertWeights:
    """The router ([H, E], shuffled hidden order) and BF16 expert weights: gate/up [E_local, H, 2, I] with biases
    [E_local, 2, I], down [E_local, I, H] with biases [E_local, H]."""

    def _init_expert_weights(self):
        E, H, I = self.num_local_experts, self.hidden_size, self.intermediate_size_per_rank
        self.router_weight = nn.Parameter(torch.empty(H, self.total_num_experts, dtype=DTYPE))
        self.router_bias = nn.Parameter(torch.zeros(self.total_num_experts, dtype=DTYPE))
        self.gate_up_proj_weight = nn.Parameter(torch.empty(E, H, 2, I, dtype=DTYPE))
        self.gate_up_proj_bias = nn.Parameter(torch.zeros(E, 2, I, dtype=DTYPE))
        self.down_proj_weight = nn.Parameter(torch.empty(E, I, H, dtype=DTYPE))
        self.down_proj_bias = nn.Parameter(torch.zeros(E, H, dtype=DTYPE))

        set_weight_loader(self.router_weight, decode_router_weight_loader(self.total_num_experts, H))
        set_weight_loader(self.post_attention_layernorm.weight, common._shuffled_hidden_loader(H))
        local_experts = list(range(self.ep_rank * E, (self.ep_rank + 1) * E))
        # sharded by expert TP rank, like the MXFP4 experts
        gate_up = ShardSpec(dim=1, size=I * 2, num_shards=self.tp_degree)
        down = ShardSpec(dim=2, size=I // 32, num_shards=self.tp_degree)
        loaders = {
            self.gate_up_proj_weight: bf16_gate_up_weight_loader(gate_up, (E, H, 2, I)),
            # GPT-OSS's SwiGLU multiplies by up + 1; the 1 is folded into the up bias
            self.gate_up_proj_bias: bf16_gate_up_bias_loader(gate_up, (E, 2, I), hidden_act_bias=1.0),
            self.down_proj_weight: bf16_down_weight_loader(down, (E, I, H)),
            # summed over the expert TP ranks, so each holds a share of the bias
            self.down_proj_bias: mxfp4_down_bias_loader((E, H), scale=self.tp_degree),
        }
        for param, loader in loaders.items():
            set_weight_loader(param, with_rank_override(expert_parallel_tensor_dim_loader(local_experts, loader),
                                                        rank=self.expert_tp_rank))

    def hf_weight_mappings(self, prefix: str, key: str) -> dict:
        experts, key_experts = f"{prefix}.mlp.experts", f"{key}.mlp.experts"
        mx = {proj: [f"{key_experts}.{proj}_blocks", f"{key_experts}.{proj}_scales"]
              for proj in ("gate_up_proj", "down_proj")}
        return {
            f"{experts}.post_attention_layernorm.weight": f"{key}.post_attention_layernorm.weight",
            f"{experts}.router_weight": f"{key}.mlp.router.weight",
            f"{experts}.router_bias": f"{key}.mlp.router.bias",
            f"{experts}.gate_up_proj_weight": mx["gate_up_proj"],
            f"{experts}.gate_up_proj_bias": f"{key_experts}.gate_up_proj_bias",
            f"{experts}.down_proj_weight": mx["down_proj"],
            f"{experts}.down_proj_bias": f"{key_experts}.down_proj_bias",
        }


class PrefillExperts(Bf16ExpertWeights, model_prefill.GptOssExperts):
    # the BF16 kernel's block (ops/moe/moe_cte_bf16.py)
    block_size = BLOCK_SIZE

    def forward(self, hidden_states, positions, residual):
        """The MoE of this rank's sequence shard; returns ``(out, residual + hidden_states)``."""
        norm = self.post_attention_layernorm
        normed, summed = NF.rmsnorm_residual_prefill(
            hidden_states.to(DTYPE), gamma=norm.weight, residual=residual, eps=norm.variance_epsilon,
            unpadded_hidden_size=norm.unpadded_hidden_size,
        )
        affinities = self._route(normed)
        if self.tp_group.world_size > 1:
            normed = neuron_collectives.all_gather(self.tp_group, normed, dim=0)
            affinities = neuron_collectives.all_gather(self.tp_group, affinities, dim=0)
        # the padding mask, taken before the replicas' tokens are concatenated as in the MXFP4 prefill
        real = torch.arange(positions.shape[0], device=positions.device) <= torch.argmax(positions)
        if self.cross_dp_ep:
            normed = neuron_collectives.all_gather(self.dp_group, normed, dim=0)
            affinities = neuron_collectives.all_gather(self.dp_group, affinities, dim=0)
            real = neuron_collectives.all_gather(self.dp_group, real, dim=0)
        affinities = self._local_affinities(affinities, self._ep_rank_id())
        token_position_to_id, block_to_expert, _ = self._blockwise_mapping(affinities, real)
        out = moe_experts_blockwise_swiglu(
            normed,
            gate_up=self.gate_up_proj_weight,
            gate_up_bias=self.gate_up_proj_bias,
            down=self.down_proj_weight,
            down_bias=self.down_proj_bias,
            affinities=affinities,
            token_position_to_id=token_position_to_id,
            block_to_expert=block_to_expert,
            **self._clamps(),
        )
        return self._combine(out), summed

    def _route(self, normed):
        """[T, E] fp32: each token's softmax over its top-k router logits, zero elsewhere. Routes this rank's sequence
        shard before any gather, as Qwen3-MoE does."""
        logits = (normed @ self.router_weight).float() + self.router_bias.float()
        top, experts = torch.topk(logits, self.num_experts_per_token, dim=-1)
        return torch.zeros_like(logits).scatter(1, experts, torch.softmax(top, dim=-1))


class DecodeExperts(Bf16ExpertWeights, model_decode.GptOssExperts):
    def _decode_expert_weights(self) -> dict:
        """The BF16 experts for moe_block_tkg: no scales, and the gate/up bias as it is held."""
        return dict(expert_gate_up_weights=self.gate_up_proj_weight, expert_gate_up_bias=self.gate_up_proj_bias,
                    expert_down_weights=self.down_proj_weight, expert_down_bias=self.down_proj_bias)


class _PrefillMLP(model_prefill.GptOssMLP):
    experts_type = PrefillExperts


class _PrefillLayer(model_prefill.GptOssDecoderLayer):
    mlp_type = _PrefillMLP


class _PrefillModel(model_prefill.GptOssModel):
    layer_type = _PrefillLayer


class PrefillGraph(model_prefill.GptOssForCausalLM):
    model_type = _PrefillModel


class _DecodeMLP(model_decode.GptOssMLP):
    experts_type = DecodeExperts


class _DecodeLayer(model_decode.GptOssDecoderLayer):
    mlp_type = _DecodeMLP


class _DecodeModel(model_decode.GptOssModel):
    layer_type = _DecodeLayer


class DecodeGraph(model_decode.GptOssForCausalLM):
    model_type = _DecodeModel
