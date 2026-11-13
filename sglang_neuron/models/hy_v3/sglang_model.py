# SPDX-License-Identifier: Apache-2.0
"""HYV3ForCausalLM: the host SGLang builds for a Hy3 checkpoint (models/host.py), on the shared dense transformer with
Hy3's MoE (moe.py) in every layer after the first ``first_k_dense_replace`` dense ones.

Hy3's attention is the dense transformer's: GQA with a per-head RMSNorm of Q and K before a plain RoPE. With more
attention-TP ranks than its 8 KV heads each KV head is replicated (models/transformer/weights.kv_rank). Every FP8
linear weight is dequantized to bf16 as it loads (fp8.py); the graphs compute in bf16. The checkpoint's MTP layer
(``num_nextn_predict_layers``) is not loaded."""

import os
from dataclasses import dataclass

import torch
from torch import nn

from sglang_neuron.models.host import ModelHost
from sglang_neuron.models.hy_v3.attention import DcpDecodeInputs, Fp8Attention, dcp_group
from sglang_neuron.models.hy_v3.fp8 import dequantizing, with_scales
from sglang_neuron.models.hy_v3.kv_scales import load_kv_scales
from sglang_neuron.models.hy_v3.launch import FP8_CAST
from sglang_neuron.models.hy_v3.mlp import ShardedMLP
from sglang_neuron.models.hy_v3.moe import HyV3MoE
from sglang_neuron.models.qwen3_moe.sglang_model import Qwen3MoeGraph
from sglang_neuron.models.transformer.config import TransformerConfig, check_bf16_launch, rope_theta_of
from sglang_neuron.utils.weight_loader import get_weight_loader, set_weight_loader

NAME = "Hy3"
# neuronx-cc compiles a graph's float8_e4m3fn tensors (the FP8 KV cache) on Trn2 only as its own e4m3
FP8_FN_AS_E4M3 = "--experimental-unsafe-fp8e4m3fn-as-fp8e4m3"


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


def fp8_kv_cache() -> bool:
    """Whether the launch's KV cache is FP8 (--kv-cache-dtype fp8_e4m3)."""
    from sglang.srt.runtime_context import get_server_args

    return str(getattr(get_server_args(), "kv_cache_dtype", "auto")).startswith("fp8")


class HyV3Graph(Qwen3MoeGraph):
    def __init__(self, config: TransformerConfig, moe: HyV3MoEConfig, fp8: bool, phase: str):
        self.fp8 = fp8
        super().__init__(config, moe, phase)
        self.fp8_kv = fp8_kv_cache()
        if self.fp8_kv:
            for i, layer in enumerate(self.layers):
                layer.self_attn = Fp8Attention(config, i)
        if fp8:
            # the MoE blocks wrap their own loaders
            for layer in self.layers:
                params = [layer.self_attn.qkv_weight, layer.self_attn.o_weight]
                if isinstance(layer.mlp, ShardedMLP):
                    params += [layer.mlp.gate_up_weight, layer.mlp.down_weight]
                for p in params:
                    set_weight_loader(p, dequantizing(get_weight_loader(p)))

    def build_mlp(self, config: TransformerConfig, layer_idx: int) -> nn.Module:
        if layer_idx < self.moe.first_k_dense_replace:
            return ShardedMLP(config.hidden_size, config.intermediate_size)
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

    @torch.no_grad()
    def forward(self, input_ids, positions, attn_metadata, sampling_positions, rank):
        """The dense core's forward; a DCP decode (an FP8 cache under --dcp-size) builds the group's decode mask once
        for every layer (attention.DcpDecodeInputs)."""
        attn = self.layers[0].self_attn
        if self.phase != "decode" or getattr(attn, "dcp_size", 1) == 1:
            return super().forward(input_ids, positions, attn_metadata, sampling_positions, rank)
        from sglang_neuron.models.gpt_oss.attention_masks import decode_attention_masks

        md = attn_metadata[attn.name]
        rows = md["block_table_tensor"].shape[0]
        _, mask = decode_attention_masks(attn_metadata, positions, sliding_window=0,
                                         q_heads=attn.num_heads * attn.dcp_size, swa_layer=None, full_layer=attn.name)
        inputs = DcpDecodeInputs(table=md["block_table_tensor"].to(torch.int32),
                                 update_idx=md["slot_mapping"].view(rows, 1).to(torch.uint32), mask=mask)
        hidden = self.embed_tokens(input_ids, scatter_tokens=False, rank=rank)
        cos, sin = self.rotary_emb(positions)
        for layer in self.layers:
            hidden = layer(hidden, lambda x, a=layer.self_attn: a.decode(x, cos, sin, inputs))
        hidden = self.norm(hidden.index_select(0, sampling_positions))
        return self.lm_head(hidden)

    def load_weights(self, checkpoint_path: str, device: torch.device) -> None:
        super().load_weights(checkpoint_path, device)
        if self.fp8_kv:
            for layer, (k, v) in zip(self.layers, load_kv_scales(checkpoint_path, len(self.layers))):
                layer.self_attn.set_kv_scales(k, v)

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
        dcp = int(getattr(pc, "attn_dcp_size", 1) or 1)
        if dcp > 1:
            from sglang_neuron.distributed.decode_context_parallel import validate_dcp_heads

            if not fp8_kv_cache():
                raise ValueError(f"{NAME}: --dcp-size {dcp} is served with the FP8 KV cache (--kv-cache-dtype fp8_e4m3)")
            validate_dcp_heads(NAME, dense.num_attention_heads, dense.num_key_value_heads, int(pc.tp_size), dcp)
            # the KV-parallel prefill walks its chunk in segments of whole virtual pages (D blocks), so every prefill
            # runs the whole chunk
            self.prefill_ladder = False

    def build_graph(self, phase: str, config) -> nn.Module:
        return HyV3Graph(*hy3_config(config), phase)

    def neuron_compiler_args(self, phase: str, default_args: list[str]) -> list[str]:
        """An FP8 KV cache on Trn2 compiles its float8_e4m3fn tensors as Trn2's e4m3 (launch.py): a frontend option,
        passed through the hlo2tensorizer options."""
        args = super().neuron_compiler_args(phase, default_args)
        if not (fp8_kv_cache() and os.environ.get(FP8_CAST) == "1"):
            return args
        prefix = "--internal-hlo2tensorizer-options="
        for i, arg in enumerate(args):
            if arg.startswith(prefix):
                if FP8_FN_AS_E4M3 not in arg:
                    args = [*args[:i], f"{arg} {FP8_FN_AS_E4M3}", *args[i + 1:]]
                return args
        return [*args, prefix + FP8_FN_AS_E4M3]


EntryClass = [HYV3ForCausalLM]
