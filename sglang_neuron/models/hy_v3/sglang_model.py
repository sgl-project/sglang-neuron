# SPDX-License-Identifier: Apache-2.0
"""HYV3ForCausalLM: the host SGLang builds for a Hy3 checkpoint (models/host.py), on the shared dense transformer with
Hy3's MoE (moe.py) in every layer after the first ``first_k_dense_replace`` dense ones.

Hy3's attention is the dense transformer's: GQA with a per-head RMSNorm of Q and K before a plain RoPE. With more
attention-TP ranks than its 8 KV heads each KV head is replicated (models/transformer/weights.kv_rank). Every FP8
linear weight is dequantized to bf16 as it loads (fp8.py); the graphs compute in bf16. The checkpoint's MTP layer
(``num_nextn_predict_layers``) is not loaded."""

from dataclasses import dataclass

import torch
from torch import nn

from sglang_neuron.models.host import ModelHost
from sglang_neuron.models.hy_v3.fp8 import dequantizing, with_scales
from sglang_neuron.models.hy_v3.moe import HyV3MoE
from sglang_neuron.models.qwen3_moe.sglang_model import Qwen3MoeGraph
from sglang_neuron.models.transformer.config import TransformerConfig, check_bf16_launch, rope_theta_of
from sglang_neuron.models.transformer.model import MLP
from sglang_neuron.utils.weight_loader import get_weight_loader, set_weight_loader

NAME = "Hy3"


@dataclass(frozen=True)
class HyV3MoEConfig:
    num_experts: int
    top_k: int
    intermediate_size: int
    shared_intermediate_size: int
    first_k_dense_replace: int
    route_norm: bool
    scaling: float


def hy3_config(hf_config) -> tuple:
    """The dense part and the MoE of a Hy3 checkpoint, refusing what the graphs do not serve."""
    quant = getattr(hf_config, "quantization_config", None) or {}
    quant = quant if isinstance(quant, dict) else quant.to_dict()
    if quant and (quant.get("quant_method") != "fp8" or quant.get("weight_block_size")):
        raise ValueError(f"{NAME}: the graphs read bf16 or per-tensor FP8 weights, not {quant}")
    if not getattr(hf_config, "moe_router_use_sigmoid", True) or not getattr(hf_config,
                                                                          "moe_router_enable_expert_bias", True):
        raise ValueError(f"{NAME}: the router is served as a sigmoid with a selection bias "
                         "(moe_router_use_sigmoid, moe_router_enable_expert_bias)")
    if getattr(hf_config, "attention_bias", False):
        raise ValueError(f"{NAME}: attention biases (attention_bias) are not served")
    if not getattr(hf_config, "qk_norm", True):
        raise ValueError(f"{NAME}: the attention is served with its QK-norm (qk_norm)")
    head_dim = getattr(hf_config, "head_dim", None) or hf_config.hidden_size // hf_config.num_attention_heads
    dense = TransformerConfig(
        vocab_size=hf_config.vocab_size,
        hidden_size=hf_config.hidden_size,
        intermediate_size=hf_config.intermediate_size,
        num_hidden_layers=hf_config.num_hidden_layers,
        num_attention_heads=hf_config.num_attention_heads,
        num_key_value_heads=hf_config.num_key_value_heads,
        head_dim=head_dim,
        rms_norm_eps=hf_config.rms_norm_eps,
        rope_theta=rope_theta_of(hf_config, NAME),
        rotary_dim=head_dim,
        tie_word_embeddings=bool(hf_config.tie_word_embeddings),
        qk_norm=True,
    )
    moe = HyV3MoEConfig(
        num_experts=hf_config.num_experts,
        top_k=hf_config.num_experts_per_tok,
        intermediate_size=hf_config.moe_intermediate_size,
        shared_intermediate_size=hf_config.moe_intermediate_size * int(getattr(hf_config, "num_shared_experts", 1)),
        first_k_dense_replace=int(getattr(hf_config, "first_k_dense_replace", 1)),
        route_norm=bool(getattr(hf_config, "route_norm", True)),
        scaling=float(getattr(hf_config, "router_scaling_factor", 1.0)),
    )
    return dense, moe, bool(quant)


def check_layout(dense: TransformerConfig, moe: HyV3MoEConfig, tp: int, ep: int, attn_dp: int) -> None:
    """Refuse a parallel layout the graphs cannot shard, before any weight loads."""
    problems = []
    if attn_dp > 1:
        problems.append("DP attention is not served; the experts run on attention TP")
    if dense.num_attention_heads % tp:
        problems.append(f"{dense.num_attention_heads} attention heads do not shard over --tp-size {tp}")
    kv = dense.num_key_value_heads
    if kv % tp and tp % kv:
        problems.append(f"{kv} KV heads neither shard over nor replicate across --tp-size {tp}")
    if moe.num_experts % ep or tp % ep:
        problems.append(f"{moe.num_experts} experts do not split over --ep-size {ep} within --tp-size {tp}")
    elif moe.intermediate_size % (tp // ep):
        problems.append(f"an expert intermediate size of {moe.intermediate_size} does not shard over the {tp // ep} "
                        "ranks of each expert")
    for what, size in (("dense", dense.intermediate_size), ("shared-expert", moe.shared_intermediate_size)):
        if size % tp:
            problems.append(f"a {what} intermediate size of {size} does not shard over --tp-size {tp}")
    if problems:
        raise ValueError(f"{NAME}: " + "; ".join(problems))


class DenseMLP(MLP):
    """A dense layer's MLP; it takes the prefill's real-token mask the MoE layers read, and ignores it."""

    def forward(self, x, real=None):
        return super().forward(x)


class HyV3Graph(Qwen3MoeGraph):
    def __init__(self, config: TransformerConfig, moe: HyV3MoEConfig, fp8: bool, phase: str):
        self.fp8 = fp8
        super().__init__(config, moe, phase)
        if fp8:
            # the MoE blocks wrap their own loaders
            for layer in self.layers:
                params = [layer.self_attn.qkv_weight, layer.self_attn.o_weight]
                if isinstance(layer.mlp, DenseMLP):
                    params += [layer.mlp.gate_up_weight, layer.mlp.down_weight]
                for p in params:
                    set_weight_loader(p, dequantizing(get_weight_loader(p)))

    def build_mlp(self, config: TransformerConfig, layer_idx: int) -> nn.Module:
        if layer_idx < self.moe.first_k_dense_replace:
            return DenseMLP(config, fused_gate_up=False)
        return HyV3MoE(hidden_size=config.hidden_size, num_experts=self.moe.num_experts, top_k=self.moe.top_k,
                       intermediate_size=self.moe.intermediate_size, route_norm=self.moe.route_norm,
                       scaling=self.moe.scaling, shared_intermediate_size=self.moe.shared_intermediate_size,
                       fp8=self.fp8, prefill=self.phase == "prefill")

    def hf_weight_mappings(self) -> dict:
        mappings = super().hf_weight_mappings()
        if self.fp8:
            for i in range(len(self.layers)):
                for name in ("qkv_weight", "o_weight"):
                    key = f"layers.{i}.self_attn.{name}"
                    mappings[key] = with_scales(mappings[key])
        return mappings

    def mlp_weight_mappings(self, src: str, dst: str) -> dict:
        mlp = self.layers[int(dst.split(".")[1])].mlp
        if isinstance(mlp, HyV3MoE):
            return mlp.weight_mappings(src, f"{dst}.mlp")
        mappings = {f"{dst}.mlp.gate_up_weight": [f"{src}.mlp.gate_proj.weight", f"{src}.mlp.up_proj.weight"],
                    f"{dst}.mlp.down_weight": f"{src}.mlp.down_proj.weight"}
        return {k: with_scales(v) for k, v in mappings.items()} if self.fp8 else mappings


class HYV3ForCausalLM(ModelHost):
    name = NAME
    decode_context_ladder = True

    def check_launch(self, config) -> None:
        from sglang_neuron import sglang_tree

        check_bf16_launch(config, self.name)
        dense, moe, _ = hy3_config(config)
        pc = sglang_tree.parallel_config()
        check_layout(dense, moe, int(pc.tp_size), int(pc.ep_size or 1), sglang_tree.attention_dp_size(pc))

    def build_graph(self, phase: str, config) -> nn.Module:
        return HyV3Graph(*hy3_config(config), phase)


EntryClass = [HYV3ForCausalLM]
