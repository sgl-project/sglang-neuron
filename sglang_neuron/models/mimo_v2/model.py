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
from sglang_neuron.models.mimo_v2 import mtp, weights
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
# each decoder layer's decode in one kernel with its all-reduces inside (kernels/layer.py), the target's and the MTP
# draft layers': 1 (the default: no XLA op between layers, the residual and collectives in SBUF, measured equal or faster
# than the half kernels) or 0 for the attention and MoE halves as separate kernels with the all-reduces in the graph
LAYER_KERNEL_ENV = "SGLANG_NEURON_MIMO_DECODE_LAYER_KERNEL"


# the step's embedding and greedy LM head in NKI kernels (kernels/head.py): 1 (the default) or 0 for torch's
HEAD_KERNEL_ENV = "SGLANG_NEURON_MIMO_HEAD_KERNEL"


def head_kernel_enabled() -> bool:
    import os

    value = os.environ.get(HEAD_KERNEL_ENV, "1").strip()
    if value not in ("0", "1"):
        raise ValueError(f"{HEAD_KERNEL_ENV}={value!r}: set 1 (the embedding and greedy LM head as kernels) or 0")
    return value == "1"


# the decode experts' activations in FP8, per token (kernels/moe_half.py fp8_act): 1 (the default) or 0 for bf16
MOE_W8A8_ENV = "SGLANG_NEURON_MIMO_MOE_W8A8"


def moe_w8a8() -> bool:
    import os

    value = os.environ.get(MOE_W8A8_ENV, "1").strip()
    if value not in ("0", "1"):
        raise ValueError(f"{MOE_W8A8_ENV}={value!r}: set 1 (FP8 activations, double-row matmuls) or 0 (bf16)")
    return value == "1"


# the full layers' decode attention over KV data parallelism (MiMoV2Graph._kvdp): 1 (the default) or 0
KVDP_ENV = "SGLANG_NEURON_MIMO_KVDP"


def layer_kernel_enabled() -> bool:
    import os

    value = os.environ.get(LAYER_KERNEL_ENV, "1").strip()
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
    def __init__(self, config: MiMoV2Config, layer_idx: int, mlp: nn.Module, kind=None):
        """``kind`` overrides the layer's attention kind (the MTP layer's, past the target's layers)."""
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        # the input norm's weight in fp32, as the decode kernels read it (once at load, not every step)
        self.input_norm_f32 = nn.Parameter(torch.empty(1, config.hidden_size, dtype=torch.float32),
                                           requires_grad=False)
        set_weight_loader(self.input_norm_f32, _fp32_rows_loader())
        self.self_attn = MiMoAttention(config, layer_idx, kind)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = mlp
        self.is_swa = config.swa_layers[layer_idx] if kind is None else kind.sliding_window is not None


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
        # the LM head transposed, [H, vocabulary shard]: the columns the greedy kernel streams (kernels/head.greedy)
        rows = self.lm_head.out_features_per_rank
        self.lm_head_T = nn.Parameter(torch.empty(config.hidden_size, rows, dtype=DTYPE), requires_grad=False)
        rows_loader = dense_weights.vocab_rows_loader(rows, config.hidden_size)
        set_weight_loader(self.lm_head_T, SafetensorsWeightLoader(
            transform=lambda slices, rank: rows_loader.transform(slices, rank).t().contiguous()))
        # the runtime's layer contract (models/contract.declared_layer_kinds)
        self.layer_names = [config.layer_name(i) for i in range(config.num_hidden_layers)]
        self.sliding_windows = config.sliding_windows
        # under NEURON_MTP the checkpoint's MTP layers the draft runs (mtp.py); the decode is then the MTP step
        # (mtp_step.py)
        self.mtp = None
        if mtp.enabled():
            self.mtp = nn.ModuleList([mtp.MiMoMtp(config, quantized, k) for k in range(mtp.num_layers())])
            if phase == "decode":
                from sglang.srt.runtime_context import get_server_args

                from sglang_neuron import sampling as neuron_sampling
                from sglang_neuron.models.mimo_v2 import mtp_step

                # the decode samples in its own forward; its carry is each request slot's next tokens and committed
                # length, one more row for the padding rows
                self.samples_in_forward = True
                self.sampler_modes = neuron_sampling.SAMPLER_MODES
                self.mtp_slots = mtp_step.request_rows(get_server_args())
                self.mtp_width = self.mtp_slots
                self.register_buffer("mtp_carry", torch.zeros(self.mtp_slots + 1, mtp_step.SPEC, dtype=torch.int32))
                self.register_buffer("mtp_len", torch.zeros(self.mtp_slots + 1, dtype=torch.int32))

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

    def greedy_token(self, logits: torch.Tensor) -> torch.Tensor:
        """The greedy token of each row of the LM head's ``logits`` (its vocabulary shard on the device sampler)."""
        from sglang_neuron import sampling

        if self.lm_head.gather_output:
            return torch.argmax(logits, dim=-1).to(torch.int32)
        group = self.lm_head_tp_group
        return sampling.greedy_sharded_logits(logits, group, group.rank_in_group_tensor, self.config.vocab_size,
                                              top_logprobs=0)["token_ids"]

    def _head_mesh(self):
        from sglang_neuron.backend.stack import resolve_full_mesh

        return tuple(tuple(int(r) for r in cell) for cell in resolve_full_mesh(self.lm_head_tp_group))

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        """The embeddings [T, H] of ``ids`` [T] (T <= 128): the vocabulary-sharded lookup and its sum over the ranks in
        one kernel (kernels/head.embed), or the torch embedding with SGLANG_NEURON_MIMO_HEAD_KERNEL=0."""
        if not head_kernel_enabled() or ids.shape[0] > 128:
            return self.embed_tokens(ids.to(torch.int32), scatter_tokens=False,
                                     rank=self.lm_head_tp_group.rank_in_group_tensor)
        from sglang_neuron.backend.stack import wrap_nki
        from sglang_neuron.models.mimo_v2.kernels.head import embed
        from sglang_neuron.utils.hardware_config import get_logical_nc_config

        return wrap_nki(embed)[get_logical_nc_config()](ids.reshape(-1, 1).to(torch.int32), self.embed_tokens.weight,
                                                        self.lm_head_tp_group.rank_in_group_tensor, self._head_mesh())

    def greedy(self, states: torch.Tensor, norm) -> torch.Tensor:
        """[T] int32 the greedy token of each of ``states`` [T, H] (T <= 128) before the final ``norm``: the norm, the
        sharded LM head and the argmax over the ranks in one kernel (kernels/head.greedy)."""
        if not head_kernel_enabled() or states.shape[0] > 128:
            return self.greedy_token(self.lm_head(norm(states))).reshape(-1)
        from sglang_neuron.backend.stack import wrap_nki
        from sglang_neuron.models.mimo_v2.kernels.head import greedy
        from sglang_neuron.utils.hardware_config import get_logical_nc_config

        return wrap_nki(greedy)[get_logical_nc_config()](
            states, norm.weight.view(1, -1), self.lm_head_T, self.lm_head_tp_group.rank_in_group_tensor,
            self._head_mesh(), eps=norm.eps).reshape(-1)

    def _first_layer(self, swa: bool) -> str:
        return next(self.config.layer_name(i) for i, s in enumerate(self.config.swa_layers) if s == swa)

    @torch.no_grad()
    def forward(self, input_ids, positions, attn_metadata, sampling_positions, rank, mtp_row_slots=None,
                mtp_ctrl=None, sampling_temperatures=None, sampling_seeds=None, sampling_filters=None):
        if self.mtp is not None and self.phase == "decode":
            return self._mtp_decode(attn_metadata, rank, mtp_row_slots, mtp_ctrl,
                                    (sampling_temperatures, sampling_seeds, sampling_filters))
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
            attn0 = self.layers[0].self_attn
            # under DCP the decode attends the group's gathered query heads
            inputs = decode_masks(attn_metadata, positions, self.config, attn0.num_heads * attn0.dcp_size)
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
        logits = self.lm_head(self.norm(hidden.index_select(0, sampling_positions)))
        if self.mtp is not None:
            self._mtp_prefill(hidden, input_ids, logits, sampling_positions, attn_metadata[self._first_layer(True)],
                              tables[True], rank)
        return logits

    # -- NEURON_MTP (mtp_step.py) -------------------------------------------------------------------------------------

    def _mtp_prefill(self, hidden, input_ids, logits, sampling_positions, md, table, rank) -> None:
        """Every MTP layer's KV over the chunk (mtp.py), from the target's states before its final norm: layer k reads
        the chunk's tokens k + 1 ahead, the target's greedy pick and the layers' drafts after its last token."""
        at = sampling_positions.reshape(-1)[-1:].to(torch.int64)
        ids = input_ids.reshape(1, -1)
        token = self.greedy_token(logits).reshape(-1)[-1:]
        for k, layer in enumerate(self.mtp):
            ids = mtp.rotate(ids, at, token)
            states = layer.prefill(self, hidden, ids.reshape(-1), md, *table, rank)
            if k + 1 < len(self.mtp):
                token = layer.draft(self, states.index_select(0, at))

    def _span_sample(self, logits, positions, params, width: int, spec: int) -> torch.Tensor:
        """[width, spec] the target's draw at each span token under its row's sampling parameters ``params``
        (temperatures, seeds, filters [width] rows; greedy without them): each token's noise is keyed by its row's
        seed and its position, as the decode without MTP draws (models/hy_v3/sglang_model.HyV3Graph._span_sample)."""
        from sglang_neuron import sampling

        temperatures, seeds, filters = params
        mode = sampling.sampler_mode()
        if mode == "greedy" or temperatures is None:
            return self.greedy_token(logits).reshape(width, spec).to(torch.int64)
        group = self.lm_head_tp_group
        rank = group.rank_in_group_tensor
        per_token = lambda t: t.reshape(width, 1, *t.shape[1:]).expand(width, spec, *t.shape[1:]).reshape(  # noqa: E731
            width * spec, *t.shape[1:])
        keys = sampling.noise_row_keys(per_token(seeds), positions.reshape(-1))
        if mode == "topk_topp":
            out = sampling.sample_filtered_sharded_logits(logits, group, rank, self.config.vocab_size,
                                                          per_token(temperatures), keys, per_token(filters),
                                                          top_logprobs=0)
        else:
            out = sampling.sample_sharded_logits(logits, group, rank, self.config.vocab_size,
                                                 per_token(temperatures), keys, top_logprobs=0)
        return out["token_ids"].reshape(width, spec).to(torch.int64)

    def _mtp_decode(self, attn_metadata, rank, row_slots, ctrl, params):
        """One MTP step (mtp_step.py): ``{"token_ids": [width, SPEC]}`` int32, each row's kept run, -1 after it, at the
        decode width. Every layer runs the rows' spans (attention.MiMoAttention.span_decode), each kind over its own
        span mask and slots, built once."""
        from sglang_neuron.backend.stack import resolve_full_mesh
        from sglang_neuron.models.mimo_v2 import mtp_step
        from sglang_neuron.models.mimo_v2.attention import span_mask, span_slots

        md = {True: attn_metadata[self._first_layer(True)], False: attn_metadata[self._first_layer(False)]}
        width, spec = md[False]["block_table_tensor"].shape[0], mtp_step.SPEC
        slots = row_slots.to(torch.int64)
        span = mtp_step.span_inputs(self.mtp_carry.index_select(0, slots), self.mtp_len.index_select(0, slots), ctrl)
        positions = span["positions"]
        prior = span["base"].reshape(width, 1).expand(width, spec)
        heads = self.layers[0].self_attn.num_heads
        windows = {True: self.config.swa.sliding_window, False: None}
        # the full layers' decode under the layer kernels masks by position (MiMoV2Graph._layer_call): no full mask
        full_by_position = layer_kernel_enabled() and all(self._layer_kernel_serves(layer)
                                                          for layer in self.decoder_layers())
        # the sliding layers' too, but where the one-layer MTP draft runs its own passes on their span mask
        by_position = {False: full_by_position, True: full_by_position and len(self.mtp) > 1}
        masks = {k: (None if by_position[k] else
                     span_mask(md[k], prior, positions, window=windows[k], num_heads=heads)) for k in md}
        kv_slots = {k: span_slots(md[k], positions, span["valid"], window=windows[k]) for k in md}
        tables = self._tables(positions.reshape(-1))
        hidden = self.embed(span["tokens"].reshape(-1))
        ebase = self._expert_base()
        # the one-kernel layer (kernels/layer.py) over the spans: its inputs in the kernel's layouts, built once a step
        kernel_step = None
        if layer_kernel_enabled() and all(self._layer_kernel_serves(layer) for layer in self.decoder_layers()):
            layout = lambda t: t.reshape(width, spec, -1).permute(2, 0, 1).contiguous()  # noqa: E731
            kernel_step = {"mesh": tuple(tuple(int(r) for r in cell) for cell in
                                         resolve_full_mesh(self.layers[0].self_attn.attn_tp_group)),
                           "ebase": ebase,
                           "kv_scales": self.layers[0].self_attn._scale_tensors(hidden.device)}
            for k, (cos, sin) in tables.items():
                kernel_step[("cos", k)], kernel_step[("sin", k)] = layout(cos), layout(sin)
                kernel_step[("slots", k)] = kv_slots[k].reshape(-1, 1).to(torch.int32)
            # each span token's prior length and a sliding layer's window start, the layers' position masks (the
            # span masks' prior_end and start: span_mask), in each kind's table frame
            kernel_step[("pos", False)] = prior.to(torch.float32).contiguous()
            offset = md[True]["swa_kv_pos_offset"].to(torch.int64).reshape(width, 1)
            end = (prior.to(torch.int64) - offset).clamp(min=0)
            start = (positions.to(torch.int64) - offset - windows[True] + 1).clamp(min=0)
            kernel_step[("pos", True)] = end.to(torch.float32).contiguous()
            kernel_step[("start", True)] = torch.minimum(start, end).to(torch.float32).contiguous()
        for layer in self.layers:
            if kernel_step is not None:
                k = layer.is_swa
                hidden = self._layer_call(layer, hidden.view(width, spec, -1),
                                          attn_metadata[layer.self_attn.name]["block_table_tensor"], masks[k],
                                          kernel_step[("cos", k)], kernel_step[("sin", k)], kernel_step[("slots", k)],
                                          kernel_step)
                continue
            cos, sin = tables[layer.is_swa]
            attn = layer.self_attn
            out, _, _ = attn.span_decode(hidden, layer.input_layernorm, cos, sin, attn_metadata[attn.name],
                                         masks[layer.is_swa], kv_slots[layer.is_swa], spec)
            hidden = hidden + out
            norm = layer.post_attention_layernorm
            if getattr(layer.mlp, "fused", False):
                hidden = hidden + layer.mlp.fused_decode(hidden, norm.weight, norm.eps, ebase)
            else:
                hidden = hidden + layer.mlp(norm(hidden))
        from sglang_neuron import sampling

        if params[0] is None or sampling.sampler_mode() == "greedy":
            # the greedy target: the final norm, the LM head and the argmax in one kernel (kernels/head.greedy)
            sampled = self.greedy(hidden, self.norm).reshape(width, spec).to(torch.int64)
        else:
            sampled = self._span_sample(self.lm_head(self.norm(hidden)), positions, params, width, spec)
        out, kept = mtp_step.accept(span["tokens"], sampled, span)
        bonus = sampled.gather(1, kept.reshape(-1, 1))
        kernel = None if kernel_step is None else (kernel_step[("cos", True)], kernel_step[("sin", True)],
                                                   kernel_step[("slots", True)], kernel_step)
        drafts = self._mtp_drafts(span, sampled, hidden, kept, md[True], masks[True], kv_slots[True], tables[True],
                                  rank, kernel)
        self.mtp_carry.index_copy_(0, slots, torch.cat([bonus, drafts], dim=1).to(torch.int32))
        self.mtp_len.index_copy_(0, slots, (span["base"] + kept + 1).to(torch.int32))
        if self.mtp_width > width:
            out = torch.cat([out, torch.full((self.mtp_width - width, spec), -1, dtype=out.dtype, device=out.device)])
        return {"token_ids": out.to(torch.int32)}

    def _mtp_drafts(self, span, sampled, hidden, kept, md, mask, kv_slots, table, rank, kernel=None):
        """[W, NUM_STEPS] the MTP layer's drafts after each row's kept run (mtp.py): its pass over the span (the token
        after each position is the target's sample there) writes its KV and gives the first draft at the row's last
        kept position; each further draft is a one-token pass at the next position, from the previous draft and the
        previous pass's state."""
        from sglang_neuron.models.mimo_v2 import mtp_step
        from sglang_neuron.models.mimo_v2.attention import span_mask, span_slots

        width, spec = sampled.shape
        size = hidden.shape[-1]
        at_kept = lambda state: state.view(width, spec, size).gather(  # noqa: E731
            1, kept.reshape(width, 1, 1).expand(width, 1, size)).reshape(width, size)
        if len(self.mtp) > 1:
            # one draft a layer, each layer's tokens one position further ahead (mtp.rotate)
            tokens, drafts = sampled, []
            for k, layer in enumerate(self.mtp):
                state, _ = layer.decode(self, tokens.reshape(-1), hidden, *table, md, mask, kv_slots, spec, None, rank,
                                        kernel)
                drafts.append(layer.draft(self, at_kept(state)))
                if k + 1 < len(self.mtp):
                    tokens = mtp.rotate(tokens, kept, drafts[-1])
            return torch.stack(drafts, dim=1)
        layer = self.mtp[0]
        window = self.config.swa.sliding_window
        heads = layer.attn.num_heads
        state, caches = layer.decode(self, sampled.reshape(-1), hidden, *table, md, mask, kv_slots, spec, None, rank)
        prev = at_kept(state)
        draft = layer.draft(self, prev)
        drafts = [draft]
        live = span["live"].reshape(width, 1)
        for i in range(1, mtp_step.NUM_STEPS):
            q = (span["base"] + kept + i).reshape(width, 1)
            step_mask = span_mask(md, q, q, window=window, num_heads=heads)
            step_slots = span_slots(md, q, live, window=window)
            prev, caches = layer.decode(self, draft, prev, *self.swa_rope(q.reshape(-1)), md, step_mask, step_slots,
                                        1, caches, rank)
            draft = layer.draft(self, prev)
            drafts.append(draft)
        return torch.stack(drafts, dim=1)

    def _expert_base(self):
        """[1, 1] fp32 this rank's first local expert id (the MoE half kernel's ``ebase``), None without MoE."""
        moe = next((m.mlp for m in self.layers if isinstance(m.mlp, MiMoV2MoE)), None)
        return None if moe is None else (moe._ep_rank_id().to(torch.float32) * moe.num_local_experts).reshape(1, 1)

    @staticmethod
    def _layer_kernel_serves(layer) -> bool:
        """Whether ``layer``'s decode runs as one kernel: a dense layer, or an MoE layer whose experts the MoE half
        kernel takes (moe.MiMoV2MoE.fused)."""
        if layer.self_attn.dcp_size > 1:
            return False  # its attention half attends this rank's own KV, without DCP
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
        step["ebase"] = self._expert_base()
        step["kv_scales"] = layer.self_attn._scale_tensors(layer.self_attn.qkv_weight.device)
        return step

    def _layer_kernel(self, layer, hidden, md, inputs, step):
        """One layer's decode in one kernel (kernels/layer.py), its all-reduces inside."""
        rows = hidden.shape[0]
        return self._layer_call(layer, hidden.view(rows, 1, -1), md["block_table_tensor"], inputs.mask,
                                step[("cos", layer.is_swa)], step[("sin", layer.is_swa)],
                                step[("slots", layer.is_swa)], step)

    def _kvdp(self, layer, rows: int) -> int:
        """The KV data parallelism of ``layer``'s decode attention at ``rows`` rows (kernels/attn_half.py ``kvdp``):
        the full layers' (no sinks, a long prior) over groups of up to 8 ranks holding the same KV head, at batches of
        at least 4 (nkilib's all-to-all takes 4 or more ranks), 1 elsewhere (SGLANG_NEURON_MIMO_KVDP=0: never)."""
        import os

        if os.environ.get(KVDP_ENV, "1").strip() == "0" or layer.is_swa or layer.self_attn.sink is not None:
            return 1
        replicas = self.layers[0].self_attn.attn_tp_group.world_size // max(1, self.config.full.num_kv_heads)
        k = 1
        while k * 2 <= min(8, rows, replicas) and rows % (k * 2) == 0:
            k *= 2
        return k if k >= 4 else 1

    def _kvdp_mesh(self, layer, k: int):
        from sglang_neuron.backend.stack import resolve_full_mesh

        cells = []
        for cell in resolve_full_mesh(layer.self_attn.attn_tp_group):
            ranks = [int(r) for r in cell]
            cells += [tuple(ranks[i:i + k]) for i in range(0, len(ranks), k)]
        return tuple(cells)

    def _layer_call(self, layer, x, table, mask, cos, sin, slots, step):
        """kernels/layer.py's moe_layer / dense_layer over ``x`` [B, S, H] (S tokens a row), cos / sin [d / 2, B, S],
        slots [B x S, 1] int32. Returns the layer's output [B x S, H]. A full layer's attention runs over KV data
        parallelism (``_kvdp``) where ``step`` holds the rows' prior lengths (``("pos", False)`` [B, S] fp32)."""
        from sglang_neuron.backend.stack import wrap_nki
        from sglang_neuron.models.mimo_v2.kernels.layer import dense_layer, moe_layer
        from sglang_neuron.utils.hardware_config import get_logical_nc_config

        attn, mlp = layer.self_attn, layer.mlp
        rows, span = x.shape[0], x.shape[1]
        posid = ("pos", layer.is_swa) in step
        kvdp = self._kvdp(layer, rows) if posid else 1
        if posid:
            # the span's active part alone (the prior is masked by position): token q sees the span's tokens k <= q
            k_idx = torch.arange(span, device=x.device).reshape(span, 1, 1, 1)
            q_idx = torch.arange(span, device=x.device).reshape(1, 1, 1, span)
            mask = (k_idx <= q_idx).expand(span, rows // kvdp, attn.num_heads * kvdp, span).to(torch.uint8)
        attn_args = (x, layer.input_norm_f32, attn.qkv_weight, cos, sin, table, attn.k_cache, attn.v_cache, mask,
                     attn._sink_column(), slots, attn._kernel_o_weight())
        k_scale, v_scale = step.get("kv_scales", (None, None))
        common = dict(k_scale=k_scale, v_scale=v_scale, eps=layer.input_layernorm.eps,
                      softmax_scale=attn.scaling / attn.kv_scale, v_head_dim=attn.v_head_dim)
        if kvdp > 1:
            common.update(pos=step[("pos", False)], rank=attn.attn_tp_group.rank_in_group_tensor, kvdp=kvdp,
                          kvdp_mesh=self._kvdp_mesh(layer, kvdp))
        elif posid:
            common.update(pos=step[("pos", layer.is_swa)], start=step.get(("start", layer.is_swa)))
        gamma = layer.post_attention_layernorm.weight.view(1, -1)
        if isinstance(mlp, DenseMLP):
            return wrap_nki(dense_layer)[get_logical_nc_config()](
                *attn_args, gamma, mlp.gate_weight, mlp.up_weight, mlp.down_weight, step["mesh"], **common)
        return wrap_nki(moe_layer)[get_logical_nc_config()](
            *attn_args, gamma, mlp.router_wT, mlp.expert_bias.view(1, -1), step["ebase"], mlp.gate_up_weight,
            mlp.down_weight, mlp.gate_up_scale, mlp.down_scale, step["mesh"], route_scale=float(mlp.scaling),
            top_k=mlp.top_k, fp8_act=moe_w8a8(), **common)

    def decoder_layers(self) -> list:
        """The target's decoder layers, then the MTP layers' under NEURON_MTP (the KV pool's layers after them)."""
        return [*self.layers, *(layer.layer for layer in (self.mtp or ()))]

    def get_kv_spec(self) -> KVSpec:
        return KVSpec(layers=[LayerSpec(name=layer.self_attn.name) for layer in self.decoder_layers()])

    def bind_kv_cache(self, kv_caches: dict) -> None:
        for layer in self.decoder_layers():
            attn = layer.self_attn
            attn.k_cache, attn.v_cache = kv_caches[attn.name]

    def hf_weight_mappings(self) -> dict:
        """Parameter name to its checkpoint key, or the keys (weights, then their scales) it is built from."""
        fp8 = (lambda keys: weights.with_scales(keys, FP8_SCALE)) if self.quantized else list
        mappings = {"embed_tokens.weight": "model.embed_tokens.weight", "norm.weight": "model.norm.weight"}
        if not self.config.tie_word_embeddings:
            mappings["lm_head.weight"] = "lm_head.weight"
        mappings["lm_head_T"] = "model.embed_tokens.weight" if self.config.tie_word_embeddings else "lm_head.weight"
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
            if layer.self_attn.dcp_sink is not None:
                mappings[f"{dst}.self_attn.dcp_sink"] = f"{attn}.attention_sink_bias"
            mappings[f"{dst}.input_norm_f32"] = f"{src}.input_layernorm.weight"
            if isinstance(layer.mlp, MiMoV2MoE):
                mappings.update(layer.mlp.weight_mappings(src, f"{dst}.mlp"))
            else:
                for proj in ("gate", "up", "down"):
                    mappings[f"{dst}.mlp.{proj}_weight"] = fp8([f"{src}.mlp.{proj}_proj.weight"])
        for k, layer in enumerate(self.mtp or ()):
            mappings.update(layer.weight_mappings(f"mtp.{k}", fp8))
        return mappings

    def load_weights(self, checkpoint_path: str, device: torch.device) -> None:
        """Read this rank's shard of every mapped parameter onto ``device``: the fused QKV by the checkpoint's own
        shard count (weights.qkv_disk_tp), each quantized weight dequantized to bf16 as it is read."""
        disk_tp = weights.qkv_disk_tp(checkpoint_path)
        for layer in self.decoder_layers():
            layer.self_attn.set_qkv_disk_tp(disk_tp)
        from sglang_neuron.models.mimo_v2 import experts_fp8

        # the converted experts' disk cache is named by the checkpoint (experts_fp8._ExpertCache)
        experts_fp8.set_checkpoint(checkpoint_path)
        checkpoint = SafetensorsCheckpoint(checkpoint_path)
        state = checkpoint.load_sharded_pipelined(self.tp_group.rank_in_group, self.tp_group.world_size, self,
                                                  self.hf_weight_mappings(), device).state_dict
        self.load_state_dict(state, strict=False, assign=True)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
