# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2's role graphs (one class, ``phase`` prefill or decode), on attention TP with the experts over EP.

Each layer is pre-norm: ``x + attention(input_layernorm(x))``, then ``x + mlp(post_attention_layernorm(x))``, its
attention the layer kind's (attention.MiMoAttention), its MLP the dense SiLU MLP in layer 0
(models/hy_v3/mlp.ShardedMLP, its 16384 rows sharded over attention TP) and the MoE block after it (moe.MiMoV2MoE).
The two layer kinds have their own RoPE base, so the graph keeps one table of each.

The graph declares its layers (``layer_names``, ``sliding_windows``: the config has ``hybrid_layer_pattern``, not
``layer_types``), so the runtime trims each sliding layer's decode table to its window and, with hybrid SWA memory,
reads it from the window pool (models/contract.py, model_executor/graph_runner.py)."""

from __future__ import annotations

import torch
from sglang.srt.distributed.parallel_state import get_attn_tp_group
from torch import nn

from sglang_neuron.layers import ColumnParallelLinear
from sglang_neuron.layers.embedding import VocabDimShardedEmbedding
from sglang_neuron.mem_cache.kv_cache import KVSpec, LayerSpec
from sglang_neuron.models.mimo_v2 import weights
from sglang_neuron.models.mimo_v2.attention import MiMoAttention, decode_masks
from sglang_neuron.models.mimo_v2.config import MiMoV2Config
from sglang_neuron.models.mimo_v2.moe import MiMoV2MoE
from sglang_neuron.models.mimo_v2.rope import RotaryTables
from sglang_neuron.models.transformer import weights as dense_weights
from sglang_neuron.models.transformer.config import DTYPE
from sglang_neuron.models.transformer.model import RMSNorm
from sglang_neuron.parallel import collectives as neuron_collectives
from sglang_neuron.parallel import groups
from sglang_neuron.utils.checkpoints import SafetensorsCheckpoint
from sglang_neuron.utils.weight_loader import SafetensorsWeightLoader, set_weight_loader

# the scale key of a 128 x 128 FP8 weight; MXFP4 experts carry weight_scale (expert_scale_suffix)
FP8_SCALE = "weight_scale_inv"
# each MoE layer's decode in one kernel with its all-reduces inside (kernels/layer.py): 1 or 0 (the default, until it is
# validated on hardware) for the attention and MoE halves as separate kernels with the all-reduces in the graph
LAYER_KERNEL_ENV = "SGLANG_NEURON_MIMO_DECODE_LAYER_KERNEL"


def layer_kernel_enabled() -> bool:
    import os

    value = os.environ.get(LAYER_KERNEL_ENV, "0").strip()
    if value not in ("0", "1"):
        raise ValueError(f"{LAYER_KERNEL_ENV}={value!r}: set 1 (each MoE layer's decode in one kernel) or 0")
    return value == "1"


def expert_scale_suffix(hf_config) -> str:
    """The routed experts' scale key: weight_scale for MXFP4 (``store_dtype: mxfp4``), weight_scale_inv for FP8
    blocks, "" for a bf16 checkpoint."""
    quant = getattr(hf_config, "quantization_config", None) or {}
    quant = quant if isinstance(quant, dict) else quant.to_dict()
    if not quant:
        return ""
    return "weight_scale" if str(quant.get("store_dtype") or "").lower() == "mxfp4" else FP8_SCALE


class DenseMLP(nn.Module):
    """Layer 0's SiLU-gated MLP, its intermediate dimension sharded over attention TP, with gate, up and down as their
    own [H, I / tp], [H, I / tp] and [I / tp, H] tensors: the layout nkilib's mlp takes in the decode layer kernel
    (kernels/layer.dense_layer), which the prefill's torch path reads too, so one device copy serves both."""

    def __init__(self, hidden_size: int, intermediate_size: int, quantized: bool):
        super().__init__()
        self.attn_tp_group = groups.get_neuron_attn_tp_group()
        tp = self.attn_tp_group.world_size
        if intermediate_size % tp:
            raise ValueError(f"a dense intermediate size of {intermediate_size} does not shard over {tp} ranks")
        rows = intermediate_size // tp
        self.gate_weight = nn.Parameter(torch.empty(hidden_size, rows, dtype=DTYPE))
        self.up_weight = nn.Parameter(torch.empty(hidden_size, rows, dtype=DTYPE))
        self.down_weight = nn.Parameter(torch.empty(rows, hidden_size, dtype=DTYPE))
        column = SafetensorsWeightLoader(transform=lambda slices, rank: slices[0][rank * rows:(rank + 1) * rows].t())
        row = SafetensorsWeightLoader(transform=lambda slices, rank: slices[0][:, rank * rows:(rank + 1) * rows].t())
        for param, loader in ((self.gate_weight, column), (self.up_weight, column), (self.down_weight, row)):
            set_weight_loader(param, weights.dequantizing(loader) if quantized else loader)

    def forward(self, x, real=None):
        out = (torch.nn.functional.silu(x @ self.gate_weight) * (x @ self.up_weight)) @ self.down_weight
        return neuron_collectives.all_reduce(self.attn_tp_group, out)


def _fp32_rows_loader() -> SafetensorsWeightLoader:
    """A norm's weight [H] as the fp32 [1, H] row nkilib's attention block reads it in."""
    return SafetensorsWeightLoader(transform=lambda slices, rank: slices[0][...].to(torch.float32).reshape(1, -1))


class MiMoDecoderLayer(nn.Module):
    def __init__(self, config: MiMoV2Config, layer_idx: int, mlp: nn.Module):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        # the input norm's weight in fp32, as the decode kernels read it (once at load, not every step)
        self.input_norm_f32 = nn.Parameter(torch.empty(1, config.hidden_size, dtype=torch.float32),
                                           requires_grad=False)
        set_weight_loader(self.input_norm_f32, _fp32_rows_loader())
        self.self_attn = MiMoAttention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = mlp
        self.is_swa = config.swa_layers[layer_idx]


class MiMoV2Graph(nn.Module):
    def __init__(self, config: MiMoV2Config, phase: str, *, quantized: bool, expert_suffix: str):
        super().__init__()
        if phase not in ("prefill", "decode"):
            raise ValueError(f"a role graph is a prefill or a decode, not {phase!r}")
        self.config = config
        self.phase = phase
        self.quantized = quantized
        self.expert_suffix = expert_suffix
        self.tp_group = get_attn_tp_group()
        self.lm_head_tp_group = groups.get_neuron_attn_tp_group()
        group = self.lm_head_tp_group
        self.embed_tokens = VocabDimShardedEmbedding(config.vocab_size, config.hidden_size, dtype=DTYPE,
                                                     tp_group=group.device_group)
        self.layers = nn.ModuleList([MiMoDecoderLayer(config, i, self._build_mlp(i))
                                     for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.full_rope = RotaryTables(config.full.rope_theta, config.rotary_dim)
        self.swa_rope = RotaryTables(config.swa.rope_theta, config.rotary_dim)
        self.lm_head = ColumnParallelLinear(config.hidden_size, config.vocab_size, dtype=DTYPE, gather_output=True,
                                            tp_group=group.device_group)
        set_weight_loader(self.embed_tokens.weight,
                          dense_weights.vocab_rows_loader(self.embed_tokens.vocab_size_per_rank, config.hidden_size))
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        else:
            set_weight_loader(self.lm_head.weight,
                              dense_weights.vocab_rows_loader(self.lm_head.out_features_per_rank, config.hidden_size))
        # the runtime's layer contract (models/contract.declared_layer_kinds)
        self.layer_names = [config.layer_name(i) for i in range(config.num_hidden_layers)]
        self.sliding_windows = config.sliding_windows

    def _build_mlp(self, layer_idx: int) -> nn.Module:
        config = self.config
        if not config.moe_layers[layer_idx]:
            return DenseMLP(config.hidden_size, config.intermediate_size, self.quantized)
        return MiMoV2MoE(hidden_size=config.hidden_size, num_experts=config.num_experts, top_k=config.top_k,
                         intermediate_size=config.moe_intermediate_size, norm_topk_prob=config.norm_topk_prob,
                         scaling=config.routed_scaling_factor, expert_scale_suffix=self.expert_suffix,
                         prefill=self.phase == "prefill", layer_key=f"layers.{layer_idx}")

    def _tables(self, positions: torch.Tensor) -> dict:
        """Each layer kind's cos / sin [T, KV_HEAD_DIM / 2] of the step's positions."""
        return {True: self.swa_rope(positions), False: self.full_rope(positions)}

    @torch.no_grad()
    def forward(self, input_ids, positions, attn_metadata, sampling_positions, rank):
        hidden = self.embed_tokens(input_ids, scatter_tokens=False, rank=rank)
        tables = self._tables(positions)
        if self.phase == "prefill":
            # the chunk's real tokens, which the MoE maps alone (models/qwen3_moe/sglang_model.Qwen3MoeGraph.forward):
            # the runtime pads a chunk with its last real position
            real = torch.arange(positions.shape[0], device=positions.device) <= torch.argmax(positions)
            for layer in self.layers:
                cos, sin = tables[layer.is_swa]
                attn = layer.self_attn
                hidden = hidden + attn.prefill(layer.input_layernorm(hidden), cos, sin, attn_metadata[attn.name])
                hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden), real=real)
        else:
            heads = self.layers[0].self_attn.num_heads
            inputs = decode_masks(attn_metadata, positions, self.config, heads)
            step = None
            for layer in self.layers:
                cos, sin = tables[layer.is_swa]
                attn = layer.self_attn
                md = attn_metadata[attn.name]
                kind_inputs = inputs["swa" if layer.is_swa else "full"]
                if kind_inputs.mask is not None and layer_kernel_enabled() and self._layer_kernel_serves(layer):
                    if step is None:
                        step = self._step_inputs(tables, attn_metadata, layer)
                    hidden = self._layer_kernel(layer, hidden, md, kind_inputs, step)
                    continue
                hidden = hidden + attn.decode(hidden, layer.input_layernorm, cos, sin, md, kind_inputs)
                norm = layer.post_attention_layernorm
                if getattr(layer.mlp, "fused", False):
                    if step is None:
                        step = self._step_inputs(tables, attn_metadata, layer)
                    hidden = hidden + layer.mlp.fused_decode(hidden, norm.weight, norm.eps, step["ebase"])
                else:
                    hidden = hidden + layer.mlp(norm(hidden))
        hidden = self.norm(hidden.index_select(0, sampling_positions))
        return self.lm_head(hidden)

    @staticmethod
    def _layer_kernel_serves(layer) -> bool:
        """Whether ``layer``'s decode runs as one kernel: a dense layer, or an MoE layer whose experts the MoE half
        kernel takes (moe.MiMoV2MoE.fused)."""
        return isinstance(layer.mlp, DenseMLP) or getattr(layer.mlp, "fused", False)

    def _step_inputs(self, tables, attn_metadata, layer) -> dict:
        """The kernels' inputs a decode step shares across its layers, built once: each kind's cos / sin in the
        kernel's [d / 2, B, 1] layout, each kind's slots as int32 [B, 1] (the sliding layers' are the window pool's),
        this rank's first local expert and the attention-TP group's mesh."""
        from sglang_neuron.backend.stack import resolve_full_mesh

        step = {"mesh": tuple(tuple(int(r) for r in cell)
                              for cell in resolve_full_mesh(layer.self_attn.attn_tp_group))}
        for is_swa, (cos, sin) in tables.items():
            step[("cos", is_swa)] = cos.t().unsqueeze(-1).contiguous()
            step[("sin", is_swa)] = sin.t().unsqueeze(-1).contiguous()
        for other in self.layers:
            key = ("slots", other.is_swa)
            if key not in step:
                slots = attn_metadata[other.self_attn.name]["slot_mapping"]
                step[key] = slots.view(slots.shape[0], 1).to(torch.int32)
        moe = next((m.mlp for m in self.layers if isinstance(m.mlp, MiMoV2MoE)), None)
        step["ebase"] = None if moe is None else (moe._ep_rank_id().to(torch.float32)
                                                  * moe.num_local_experts).reshape(1, 1)
        return step

    def _layer_kernel(self, layer, hidden, md, inputs, step):
        """One layer's decode in one kernel (kernels/layer.py), its all-reduces inside."""
        from sglang_neuron.backend.stack import wrap_nki
        from sglang_neuron.models.mimo_v2.kernels.layer import dense_layer, moe_layer
        from sglang_neuron.utils.hardware_config import get_logical_nc_config

        attn, mlp = layer.self_attn, layer.mlp
        rows = hidden.shape[0]
        attn_args = (hidden.view(rows, 1, -1), layer.input_norm_f32, attn.qkv_weight, step[("cos", layer.is_swa)],
                     step[("sin", layer.is_swa)], md["block_table_tensor"], attn.k_cache, attn.v_cache, inputs.mask,
                     attn._sink_column(), step[("slots", layer.is_swa)], attn.o_weight)
        common = dict(eps=layer.input_layernorm.eps, softmax_scale=attn.scaling, v_head_dim=attn.v_head_dim)
        gamma = layer.post_attention_layernorm.weight.view(1, -1)
        if isinstance(mlp, DenseMLP):
            return wrap_nki(dense_layer)[get_logical_nc_config()](
                *attn_args, gamma, mlp.gate_weight, mlp.up_weight, mlp.down_weight, step["mesh"], **common)
        return wrap_nki(moe_layer)[get_logical_nc_config()](
            *attn_args, gamma, mlp.router_wT, mlp.expert_bias.view(1, -1), step["ebase"], mlp.gate_up_weight,
            mlp.down_weight, mlp.gate_up_scale, mlp.down_scale, step["mesh"], route_scale=float(mlp.scaling),
            top_k=mlp.top_k, **common)

    def get_kv_spec(self) -> KVSpec:
        return KVSpec(layers=[LayerSpec(name=layer.self_attn.name) for layer in self.layers])

    def bind_kv_cache(self, kv_caches: dict) -> None:
        for layer in self.layers:
            attn = layer.self_attn
            attn.k_cache, attn.v_cache = kv_caches[attn.name]

    def hf_weight_mappings(self) -> dict:
        """Parameter name to its checkpoint key, or the keys (weights, then their scales) it is built from."""
        fp8 = (lambda keys: weights.with_scales(keys, FP8_SCALE)) if self.quantized else list
        mappings = {"embed_tokens.weight": "model.embed_tokens.weight", "norm.weight": "model.norm.weight"}
        if not self.config.tie_word_embeddings:
            mappings["lm_head.weight"] = "lm_head.weight"
        for i, layer in enumerate(self.layers):
            src, dst = f"model.layers.{i}", f"layers.{i}"
            attn = f"{src}.self_attn"
            mappings.update({
                f"{dst}.self_attn.qkv_weight": fp8([f"{attn}.qkv_proj.weight"]),
                f"{dst}.self_attn.o_weight": f"{attn}.o_proj.weight",
                f"{dst}.input_layernorm.weight": f"{src}.input_layernorm.weight",
                f"{dst}.post_attention_layernorm.weight": f"{src}.post_attention_layernorm.weight",
            })
            if layer.self_attn.sink is not None:
                mappings[f"{dst}.self_attn.sink"] = f"{attn}.attention_sink_bias"
            mappings[f"{dst}.input_norm_f32"] = f"{src}.input_layernorm.weight"
            if isinstance(layer.mlp, MiMoV2MoE):
                mappings.update(layer.mlp.weight_mappings(src, f"{dst}.mlp"))
            else:
                for proj in ("gate", "up", "down"):
                    mappings[f"{dst}.mlp.{proj}_weight"] = fp8([f"{src}.mlp.{proj}_proj.weight"])
        return mappings

    def load_weights(self, checkpoint_path: str, device: torch.device) -> None:
        """Read this rank's shard of every mapped parameter onto ``device``: the fused QKV by the checkpoint's own
        shard count (weights.qkv_disk_tp), each quantized weight dequantized to bf16 as it is read."""
        disk_tp = weights.qkv_disk_tp(checkpoint_path)
        for layer in self.layers:
            layer.self_attn.set_qkv_disk_tp(disk_tp)
        checkpoint = SafetensorsCheckpoint(checkpoint_path)
        state = checkpoint.load_sharded_pipelined(self.tp_group.rank_in_group, self.tp_group.world_size, self,
                                                  self.hf_weight_mappings(), device).state_dict
        self.load_state_dict(state, strict=False, assign=True)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
