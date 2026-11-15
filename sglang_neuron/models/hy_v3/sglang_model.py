# SPDX-License-Identifier: Apache-2.0
"""HYV3ForCausalLM: the host SGLang builds for a Hy3 checkpoint (models/host.py), on the shared dense transformer with
Hy3's MoE (moe.py) in every layer after the first ``first_k_dense_replace`` dense ones.

Hy3's attention is the dense transformer's: GQA with a per-head RMSNorm of Q and K before a plain RoPE. With more
attention-TP ranks than its 8 KV heads each KV head is replicated (models/transformer/weights.kv_rank). Every FP8
linear weight is dequantized to bf16 as it loads (fp8.py); the graphs compute in bf16. Under
``--speculative-algorithm NEURON_MTP`` the checkpoint's MTP layer (``num_nextn_predict_layers``) is loaded too (mtp.py)
and the KV pool holds its layer after the target's."""

import os
from dataclasses import dataclass

import torch
from torch import nn

from sglang_neuron.models.host import ModelHost
from sglang_neuron.models.hy_v3 import mtp, sparse
from sglang_neuron.models.hy_v3.attention import DcpDecodeInputs, Fp8Attention, dcp_group
from sglang_neuron.models.hy_v3.fp8 import dequantizing, with_scales
from sglang_neuron.models.hy_v3.kv_scales import load_kv_scales
from sglang_neuron.models.hy_v3.launch import FP8_CAST
from sglang_neuron.models.hy_v3.mlp import ShardedMLP
from sglang_neuron.models.hy_v3.moe import HyV3MoE
from sglang_neuron.models.qwen3_moe.sglang_model import Qwen3MoeGraph
from sglang_neuron.models.transformer.config import TransformerConfig, check_bf16_launch, rope_theta_of
from sglang_neuron.models.transformer.model import DecoderLayer
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
        self.mtp = None
        if mtp.enabled():
            n = config.num_hidden_layers
            layer = DecoderLayer(config, n, self.build_mlp(config, n))
            if self.fp8_kv:
                layer.self_attn = Fp8Attention(config, n)
                # the draft attends densely (mtp.py)
                layer.self_attn.sparse = None
            self.mtp = mtp.HyV3Mtp(config, layer)
            if phase == "decode":
                from sglang.srt.runtime_context import get_server_args

                from sglang_neuron.models.hy_v3 import mtp_step

                # the decode samples in its own forward: every step is the MTP step (mtp_step.py), whose carry is
                # each request slot's next tokens and committed length, one more row for the padding rows
                self.samples_in_forward = True
                self.mtp_slots = int(get_server_args().max_running_requests)
                self.mtp_width = self.mtp_slots
                self.register_buffer("mtp_carry", torch.zeros(self.mtp_slots + 1, mtp_step.SPEC, dtype=torch.int32))
                self.register_buffer("mtp_len", torch.zeros(self.mtp_slots + 1, dtype=torch.int32))
        if fp8:
            # the MoE blocks wrap their own loaders
            for layer in self.decoder_layers():
                params = [layer.self_attn.qkv_weight, layer.self_attn.o_weight]
                if getattr(layer.self_attn, "sparse", None) is not None:
                    params.append(layer.self_attn.sparse.wq)
                if isinstance(layer.mlp, ShardedMLP):
                    params += [layer.mlp.gate_up_weight, layer.mlp.down_weight]
                for p in params:
                    set_weight_loader(p, dequantizing(get_weight_loader(p)))

    def decoder_layers(self) -> list:
        """The target's decoder layers, then the MTP layer's under NEURON_MTP."""
        return [*self.layers, *([self.mtp.layer] if self.mtp is not None else [])]

    def greedy_token(self, logits: torch.Tensor) -> torch.Tensor:
        """The greedy token of each row of the LM head's ``logits`` (its vocabulary shard on the device sampler)."""
        from sglang_neuron import sampling

        if self.lm_head.gather_output:
            return torch.argmax(logits, dim=-1).to(torch.int32)
        group = self.lm_head_tp_group
        return sampling.greedy_sharded_logits(logits, group, group.rank_in_group_tensor, self.config.vocab_size,
                                              top_logprobs=0)["token_ids"]

    def build_mlp(self, config: TransformerConfig, layer_idx: int) -> nn.Module:
        if layer_idx < self.moe.first_k_dense_replace:
            return ShardedMLP(config.hidden_size, config.intermediate_size)
        return HyV3MoE(hidden_size=config.hidden_size, num_experts=self.moe.num_experts, top_k=self.moe.top_k,
                       intermediate_size=self.moe.intermediate_size, route_norm=self.moe.route_norm,
                       scaling=self.moe.scaling, shared_intermediate_size=self.moe.shared_intermediate_size,
                       fp8=self.fp8, prefill=self.phase == "prefill")

    def hf_weight_mappings(self) -> dict:
        mappings = super().hf_weight_mappings()
        for i, layer in enumerate(self.layers):
            if getattr(layer.self_attn, "sparse", None) is not None:
                attn = f"model.layers.{i}.self_attn"
                q = [f"{attn}.q_proj.weight"]
                mappings[f"layers.{i}.self_attn.sparse.wq"] = with_scales(q) if self.fp8 else q
                mappings[f"layers.{i}.self_attn.sparse.q_norm"] = f"{attn}.q_norm.weight"
        if self.fp8:
            for i in range(len(self.layers)):
                for name in ("qkv_weight", "o_weight"):
                    key = f"layers.{i}.self_attn.{name}"
                    mappings[key] = with_scales(mappings[key])
        if self.mtp is not None:
            mappings.update(self.mtp.weight_mappings(self._mtp_layer_mappings()))
        return mappings

    def _mtp_layer_mappings(self) -> dict:
        """The MTP decoder layer's checkpoint keys, as a target layer's (TransformerForCausalLM.hf_weight_mappings)."""
        src, dst = f"model.layers.{self.mtp.layer_idx}", "mtp.layer"
        attn = f"{src}.self_attn"
        scaled = with_scales if self.fp8 else list
        return {
            f"{dst}.self_attn.qkv_weight": scaled([f"{attn}.{n}_proj.weight" for n in "qkv"]),
            f"{dst}.self_attn.o_weight": scaled([f"{attn}.o_proj.weight"]),
            f"{dst}.self_attn.q_norm.weight": f"{attn}.q_norm.weight",
            f"{dst}.self_attn.k_norm.weight": f"{attn}.k_norm.weight",
            f"{dst}.input_layernorm.weight": f"{src}.input_layernorm.weight",
            f"{dst}.post_attention_layernorm.weight": f"{src}.post_attention_layernorm.weight",
            **self.mtp.layer.mlp.weight_mappings(src, f"{dst}.mlp"),
        }

    @torch.no_grad()
    def forward(self, input_ids, positions, attn_metadata, sampling_positions, rank, mtp_row_slots=None,
                mtp_ctrl=None):
        """The dense core's forward; a DCP decode (an FP8 cache under --dcp-size) builds the group's decode mask once
        for every layer (attention.DcpDecodeInputs). Under NEURON_MTP the decode is the MTP step (mtp_step.py)."""
        attn = self.layers[0].self_attn
        if self.mtp is not None and self.phase == "prefill":
            return self._prefill_with_mtp(input_ids, positions, attn_metadata, sampling_positions, rank)
        if self.mtp is not None:
            return self._mtp_decode(attn_metadata, rank, mtp_row_slots, mtp_ctrl)
        if self.phase != "decode" or getattr(attn, "dcp_size", 1) == 1:
            if getattr(attn, "sparse", None) is not None:
                # the sparse summaries of a prefill chunk need the request's length so far
                end = (positions.max() + 1).reshape(1)
                attn_metadata = {k: dict(v, positions_end=end) for k, v in attn_metadata.items()}
            return super().forward(input_ids, positions, attn_metadata, sampling_positions, rank)
        from sglang_neuron.models.gpt_oss.attention_masks import decode_attention_masks

        md = attn_metadata[attn.name]
        rows = md["block_table_tensor"].shape[0]
        _, mask = decode_attention_masks(attn_metadata, positions, sliding_window=0,
                                         q_heads=attn.num_heads * attn.dcp_size, swa_layer=None, full_layer=attn.name)
        table = md["block_table_tensor"].to(torch.int32)
        inputs = DcpDecodeInputs(table=table, update_idx=md["slot_mapping"].view(rows, 1).to(torch.uint32), mask=mask,
                                 owner=md["dcp_active_owner"], local_prior=md["dcp_kv_end"].view(rows),
                                 positions=positions, n_keep=sparse.kept_pages(table.shape[1]))
        hidden = self.embed_tokens(input_ids, scatter_tokens=False, rank=rank)
        cos, sin = self.rotary_emb(positions)
        for layer in self.layers:
            hidden = layer(hidden, lambda x, a=layer.self_attn: a.decode(x, cos, sin, inputs))
        hidden = self.norm(hidden.index_select(0, sampling_positions))
        return self.lm_head(hidden)

    def _prefill_with_mtp(self, input_ids, positions, attn_metadata, sampling_positions, rank):
        """The target's prefill (TransformerForCausalLM.forward), then the MTP layer's KV over the chunk."""
        if getattr(self.layers[0].self_attn, "sparse", None) is not None:
            end = (positions.max() + 1).reshape(1)
            attn_metadata = {k: dict(v, positions_end=end) for k, v in attn_metadata.items()}
        hidden = self.embed_tokens(input_ids, scatter_tokens=False, rank=rank)
        cos, sin = self.rotary_emb(positions)
        for layer in self.layers:
            md = attn_metadata[layer.self_attn.name]
            hidden = layer(hidden, lambda x, a=layer.self_attn, md=md: a.prefill(x, cos, sin, md))
        logits = self.lm_head(self.norm(hidden.index_select(0, sampling_positions)))
        self.mtp.prefill(self, hidden, input_ids, positions, logits, attn_metadata, cos, sin, rank)
        return logits

    def _mtp_decode(self, attn_metadata, rank, row_slots, ctrl):
        """One MTP step (mtp_step.py): ``{"token_ids": [width, SPEC]}`` int32, each row's kept run, -1 after it, at the
        decode width."""
        from sglang_neuron.models.hy_v3 import mtp_step
        from sglang_neuron.ops.attention.attention_decode_mask import gen_attention_decode_mask

        attn = self.layers[0].self_attn
        if attn.dcp_size == 1:
            raise NotImplementedError("Hy3's MTP step runs under DCP (--dcp-size) with the FP8 KV cache")
        md = attn_metadata[attn.name]
        table = md["block_table_tensor"].to(torch.int32)
        width, spec, page = table.shape[0], mtp_step.SPEC, md["block_size"]
        slots = row_slots.to(torch.int64)
        span = mtp_step.span_inputs(self.mtp_carry.index_select(0, slots), self.mtp_len.index_select(0, slots), ctrl)
        share = mtp_step.dcp_span(span["positions"], span["valid"], span["base"], table, attn.dcp_size,
                                  attn.dcp_group.rank_in_group_tensor, page)
        prior = share["local_prior"].reshape(width, 1).expand(width, spec).reshape(1, width * spec)
        mask = gen_attention_decode_mask(prior.to(torch.float32), bs=width, q_head=attn.num_heads * attn.dcp_size,
                                         s_active=spec, s_prior=table.shape[1] * page, block_len=page,
                                         active_gate=share["owner"])
        inputs = DcpDecodeInputs(table=table, update_idx=share["update_idx"].to(torch.uint32), mask=mask,
                                 owner=share["owner"], local_prior=share["local_prior"], positions=span["positions"],
                                 n_keep=sparse.kept_pages(table.shape[1]), span=spec, local_pos=share["local_pos"])
        positions = span["positions"].reshape(-1)
        hidden = self.embed_tokens(span["tokens"].reshape(-1).to(torch.int32), scatter_tokens=False, rank=rank)
        cos, sin = self.rotary_emb(positions)
        for layer in self.layers:
            hidden = layer(hidden, lambda x, a=layer.self_attn: a.decode(x, cos, sin, inputs))
        normed = self.norm(hidden)
        sampled = self.greedy_token(self.lm_head(normed)).reshape(width, spec).to(torch.int64)
        out, kept = mtp_step.accept(span["tokens"], sampled, span)
        bonus = sampled.gather(1, kept.reshape(-1, 1))
        drafts = self._mtp_drafts(span, share, sampled, normed, kept, table, page, mask, inputs, cos, sin, rank)
        self.mtp_carry.index_copy_(0, slots, torch.cat([bonus, drafts], dim=1).to(torch.int32))
        self.mtp_len.index_copy_(0, slots, (span["base"] + kept + 1).to(torch.int32))
        if self.mtp_width > width:
            out = torch.cat([out, torch.full((self.mtp_width - width, spec), -1, dtype=out.dtype, device=out.device)])
        return {"token_ids": out.to(torch.int32)}

    def _mtp_drafts(self, span, share, sampled, normed, kept, table, page, mask, inputs, cos, sin, rank):
        """[W, NUM_STEPS] the MTP layer's drafts after each row's kept run (mtp.py): its pass over the span (the token
        after each position is the target's sample there) writes its KV and gives the first draft at the row's last
        kept position; each further draft is a one-token pass at the next position, from the previous draft and the
        previous pass's state."""
        from sglang_neuron.models.hy_v3 import mtp_step
        from sglang_neuron.ops.attention.attention_decode_mask import gen_attention_decode_mask

        width, spec = sampled.shape
        attn = self.layers[0].self_attn
        state = self.mtp.decode(self, sampled.reshape(-1), normed, span["positions"].reshape(-1), inputs, cos, sin,
                                rank).view(width, spec, -1)
        prev = state.gather(1, kept.reshape(width, 1, 1).expand(width, 1, state.shape[-1])).reshape(width, -1)
        draft = self.greedy_token(self.lm_head(prev)).to(torch.int64)
        drafts = [draft]
        live = span["live"].reshape(width, 1)
        for i in range(1, mtp_step.NUM_STEPS):
            q = span["base"] + kept + i
            one = mtp_step.dcp_span(q.reshape(width, 1), live, q, table, attn.dcp_size,
                                    attn.dcp_group.rank_in_group_tensor, page)
            step_mask = gen_attention_decode_mask(one["local_prior"].reshape(1, width).to(torch.float32), bs=width,
                                                  q_head=attn.num_heads * attn.dcp_size, s_active=1,
                                                  s_prior=table.shape[1] * page, block_len=page,
                                                  active_gate=one["owner"])
            step_inputs = DcpDecodeInputs(table=table, update_idx=one["update_idx"].to(torch.uint32), mask=step_mask,
                                          owner=one["owner"], local_prior=one["local_prior"],
                                          positions=q.reshape(width, 1), n_keep=table.shape[1], span=1,
                                          local_pos=one["local_pos"])
            q_cos, q_sin = self.rotary_emb(q)
            prev = self.mtp.decode(self, draft, prev, q, step_inputs, q_cos, q_sin, rank)
            draft = self.greedy_token(self.lm_head(prev)).to(torch.int64)
            drafts.append(draft)
        return torch.stack(drafts, dim=1)

    def get_kv_spec(self):
        spec = super().get_kv_spec()
        if self.mtp is not None:
            from sglang_neuron.mem_cache.kv_cache import KVSpec, LayerSpec

            spec = KVSpec(layers=[*spec.layers, LayerSpec(name=self.mtp.attn.name)])
        return spec

    def bind_kv_cache(self, kv_caches: dict) -> None:
        super().bind_kv_cache(kv_caches)
        if self.mtp is not None:
            self.mtp.attn.k_cache, self.mtp.attn.v_cache = kv_caches[self.mtp.attn.name]
        for layer in self.layers:
            if getattr(layer.self_attn, "sparse", None) is not None:
                layer.self_attn.sparse.bind(layer.self_attn.name, layer.self_attn.k_cache)

    def load_weights(self, checkpoint_path: str, device: torch.device) -> None:
        super().load_weights(checkpoint_path, device)
        if self.fp8_kv:
            layers = self.decoder_layers()
            for layer, (k, v) in zip(layers, load_kv_scales(checkpoint_path, len(layers))):
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

    def _set_stage(self, config) -> None:
        super()._set_stage(config)
        if mtp.enabled():
            # the KV pool's layer after the target's is the MTP layer's (SGLang sizes the pool from end_layer)
            self.end_layer += 1

    @property
    def runtime_class(self):
        from sglang_neuron.models.hy_v3.runtime import HyV3Runtime

        return HyV3Runtime

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

    def build_graph(self, phase: str, config) -> nn.Module:
        return HyV3Graph(*hy3_config(config), phase)

    def neuron_compiler_args(self, phase: str, default_args: list[str]) -> list[str]:
        """FP8 experts or an FP8 KV cache on Trn2 compile their float8_e4m3fn tensors as Trn2's e4m3 (launch.py): a frontend option,
        passed through the hlo2tensorizer options."""
        args = super().neuron_compiler_args(phase, default_args)
        if os.environ.get(FP8_CAST) != "1":
            return args
        prefix = "--internal-hlo2tensorizer-options="
        for i, arg in enumerate(args):
            if arg.startswith(prefix):
                if FP8_FN_AS_E4M3 not in arg:
                    args = [*args[:i], f"{arg} {FP8_FN_AS_E4M3}", *args[i + 1:]]
                return args
        return [*args, prefix + FP8_FN_AS_E4M3]


EntryClass = [HYV3ForCausalLM]
