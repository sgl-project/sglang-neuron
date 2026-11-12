# SPDX-License-Identifier: Apache-2.0
"""Qwen3MoeForCausalLM: the host SGLang builds for a Qwen3-MoE checkpoint (models/host.py), on the shared dense
transformer with a sparse MoE in every layer."""

from dataclasses import dataclass

import torch
from torch import nn

from sglang_neuron.models.host import ModelHost
from sglang_neuron.models.qwen3.sglang_model import qwen3_config
from sglang_neuron.models.qwen3_moe.moe import SparseMoE
from sglang_neuron.models.transformer.config import TransformerConfig, check_bf16_launch
from sglang_neuron.models.transformer.model import TransformerForCausalLM


@dataclass(frozen=True)
class MoEConfig:
    num_experts: int
    top_k: int
    intermediate_size: int
    norm_topk_prob: bool


def qwen3_moe_config(hf_config) -> tuple:
    """The dense part (as Qwen3's) and the MoE of a Qwen3-MoE checkpoint, refusing dense MLP layers among the sparse
    ones, which the graphs do not serve."""
    if int(getattr(hf_config, "decoder_sparse_step", 1)) != 1 or getattr(hf_config, "mlp_only_layers", None):
        raise ValueError("Qwen3-MoE: dense MLP layers (decoder_sparse_step, mlp_only_layers) are not served; every "
                         "layer is sparse")
    moe = MoEConfig(num_experts=hf_config.num_experts, top_k=hf_config.num_experts_per_tok,
                    intermediate_size=hf_config.moe_intermediate_size, norm_topk_prob=bool(hf_config.norm_topk_prob))
    return qwen3_config(hf_config), moe


class Qwen3MoeGraph(TransformerForCausalLM):
    def __init__(self, config: TransformerConfig, moe: MoEConfig, phase: str):
        self.moe = moe
        super().__init__(config, phase)

    def build_mlp(self, config: TransformerConfig, layer_idx: int) -> nn.Module:
        return SparseMoE(hidden_size=config.hidden_size, num_experts=self.moe.num_experts, top_k=self.moe.top_k,
                         intermediate_size=self.moe.intermediate_size, norm_topk_prob=self.moe.norm_topk_prob,
                         prefill=self.phase == "prefill")

    @torch.no_grad()
    def forward(self, input_ids, positions, attn_metadata, sampling_positions, rank):
        """The dense core's forward. A prefill also hands each layer's MoE the chunk's real-token mask, so that moe_cte
        maps only the real tokens: a padding row's hidden state changes from run to run, and its routing would change
        the real tokens' sums. A prefill runs one request, and the runtime pads a chunk with the last real position
        (model_executor/graph_runner.build_model_inputs), so the last real row is the first row at the largest position,
        as GPT-OSS's prefill finds it (models/gpt_oss/bf16.PrefillExperts)."""
        if self.phase != "prefill":
            return super().forward(input_ids, positions, attn_metadata, sampling_positions, rank)
        real = torch.arange(positions.shape[0], device=positions.device) <= torch.argmax(positions)
        hidden = self.embed_tokens(input_ids, scatter_tokens=False, rank=rank)
        cos, sin = self.rotary_emb(positions)
        for layer in self.layers:
            # DecoderLayer.forward, with the mask for the MoE
            md = attn_metadata[layer.self_attn.name]
            h = layer.self_attn.prefill(layer.input_layernorm(hidden), cos, sin, md)
            if layer.post_self_attn_layernorm is not None:
                h = layer.post_self_attn_layernorm(h)
            hidden = hidden + h
            h = layer.mlp(layer.post_attention_layernorm(hidden), real=real)
            if layer.post_mlp_layernorm is not None:
                h = layer.post_mlp_layernorm(h)
            hidden = hidden + h
        hidden = self.norm(hidden.index_select(0, sampling_positions))
        return self.lm_head(hidden)

    def mlp_weight_mappings(self, src: str, dst: str) -> dict:
        layer = int(dst.split(".")[1])
        return self.layers[layer].mlp.weight_mappings(src, f"{dst}.mlp")


class Qwen3MoeForCausalLM(ModelHost):
    name = "Qwen3-MoE"
    decode_context_ladder = True

    def check_launch(self, config) -> None:
        check_bf16_launch(config, self.name)
        qwen3_moe_config(config)

    def build_graph(self, phase: str, config) -> nn.Module:
        return Qwen3MoeGraph(*qwen3_moe_config(config), phase)


EntryClass = [Qwen3MoeForCausalLM]
