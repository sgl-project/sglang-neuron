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
from sglang_neuron.utils.neuron_utils import can_run_kernel
from sglang_neuron.utils.weight_loader import SafetensorsWeightLoader, set_weight_loader

# the scale key of a 128 x 128 FP8 weight; MXFP4 experts carry weight_scale (expert_scale_suffix)
FP8_SCALE = "weight_scale_inv"
# the full layers' decode attention over KV data parallelism (MiMoV2Graph._kvdp): 1 (the default) or 0
KVDP_ENV = "SGLANG_NEURON_MIMO_KVDP"




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

    def prefill_residual(self, x, normed, **_):
        """A prefill chunk's residual stream after the MLP, x + MLP(normed): on a device the MLP, its all-reduce and
        the add in one kernel (kernels/prefill.prefill_mlp)."""
        if not can_run_kernel(x):
            return x + self(normed)
        from sglang_neuron.backend.stack import resolve_full_mesh, wrap_nki
        from sglang_neuron.models.mimo_v2.kernels.prefill import prefill_mlp
        from sglang_neuron.utils.hardware_config import get_logical_nc_config

        mesh = tuple(tuple(int(r) for r in cell) for cell in resolve_full_mesh(self.attn_tp_group))
        return wrap_nki(prefill_mlp)[get_logical_nc_config()](normed, x, self.gate_weight, self.up_weight,
                                                              self.down_weight, mesh)


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
        if not mtp.enabled():
            from sglang_neuron import sampling as neuron_sampling

            # without MTP each graph samples its rows in its own forward too (_sample_rows): kernels on a device
            self.samples_in_forward = True
            self.sampler_modes = neuron_sampling.SAMPLER_MODES
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
            else:
                from sglang_neuron import sampling as neuron_sampling

                # the prefill samples its chunk's row in its own forward too (_sample_rows): kernels on a device
                self.samples_in_forward = True
                self.sampler_modes = neuron_sampling.SAMPLER_MODES

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
        one kernel (kernels/head.embed); torch past 128 rows (a prefill chunk)."""
        if ids.shape[0] > 128:
            return self.embed_tokens(ids.to(torch.int32), scatter_tokens=False,
                                     rank=self.lm_head_tp_group.rank_in_group_tensor)
        from sglang_neuron.backend.stack import wrap_nki
        from sglang_neuron.models.mimo_v2.kernels.head import embed
        from sglang_neuron.utils.hardware_config import get_logical_nc_config

        return wrap_nki(embed)[get_logical_nc_config()](ids.reshape(-1, 1).to(torch.int32), self.embed_tokens.weight,
                                                        self.lm_head_tp_group.rank_in_group_tensor, self._head_mesh())

    def greedy(self, states: torch.Tensor, norm, rows: torch.Tensor = None) -> torch.Tensor:
        """[T] int32 the greedy token of each of ``states`` [T, H] (T <= 128) before the final ``norm``: the norm, the
        sharded LM head and the argmax over the ranks in one kernel (kernels/head.greedy); with ``rows`` [T', 1] int32
        of states' rows at them, gathered in the kernel."""
        if (states.shape[0] if rows is None else rows.shape[0]) > 128 or not can_run_kernel(states):
            if rows is not None:
                states = states.index_select(0, rows.reshape(-1).to(torch.int64))
            return self.greedy_token(self.lm_head(norm(states))).reshape(-1)
        from sglang_neuron.backend.stack import wrap_nki
        from sglang_neuron.models.mimo_v2.kernels.head import greedy
        from sglang_neuron.utils.hardware_config import get_logical_nc_config

        return wrap_nki(greedy)[get_logical_nc_config()](
            states, norm.weight.view(1, -1), self.lm_head_T, self.lm_head_tp_group.rank_in_group_tensor,
            self._head_mesh(), eps=norm.eps, rows=rows).reshape(-1)

    def _step_kernel(self, md, row_slots, ctrl, width: int, spec: int):
        """The MTP step's inputs by kernels/step.py: (span, the layer kernels' step inputs). ``span`` holds what the
        accept and the drafts read (mtp_step.span_inputs' tokens, positions, valid, base, fresh and live)."""
        from sglang_neuron.backend.stack import wrap_nki
        from sglang_neuron.models.mimo_v2.kernels.step import step_inputs
        from sglang_neuron.utils.hardware_config import get_logical_nc_config

        outs = wrap_nki(step_inputs)[get_logical_nc_config()](
            self.mtp_carry, self.mtp_len, ctrl.to(torch.int32), row_slots.to(torch.int32),
            md[False]["block_table_tensor"].to(torch.int32), md[True]["block_table_tensor"].to(torch.int32),
            md[True]["swa_kv_pos_offset"].to(torch.int32), self.full_rope.inv_freq.view(1, -1),
            self.swa_rope.inv_freq.view(1, -1), page_shift=int(md[False]["block_size"]).bit_length() - 1,
            window=int(self.config.swa.sliding_window), mask_heads=self._mask_heads())
        (tokens, positions, full_slots, swa_slots, valid, fresh, live, base, full_pos, swa_pos, swa_start,
         full_cos, full_sin, swa_cos, swa_sin, *masks) = outs
        grid = lambda t: t.reshape(width, spec)  # noqa: E731
        span = {"tokens": grid(tokens).to(torch.int64), "positions": grid(positions).to(torch.int64),
                "valid": grid(valid) > 0, "base": grid(base)[:, 0].to(torch.int64),
                "fresh": grid(fresh)[:, 0] > 0, "live": grid(live)[:, 0] > 0,
                # the kernel's own int32 [W x S, 1] tensors, which the verify and commit kernels take as they are
                "raw": {"tokens": tokens, "positions": positions, "valid": valid, "fresh": fresh, "live": live,
                        "base": base}}
        layout = lambda t: t.reshape(t.shape[0], width, spec)  # noqa: E731
        step = {("cos", False): layout(full_cos), ("sin", False): layout(full_sin), ("cos", True): layout(swa_cos),
                ("sin", True): layout(swa_sin), ("slots", False): full_slots, ("slots", True): swa_slots,
                ("pos", False): grid(full_pos), ("pos", True): grid(swa_pos), ("start", True): grid(swa_start)}
        step.update({("mask", h): m for h, m in zip(self._mask_heads(), masks)})
        return span, step

    def _mask_heads(self) -> tuple:
        """The decoder layers' distinct per-rank query head counts: one span active mask from the step kernel each."""
        return tuple(sorted({int(layer.self_attn.num_heads * layer.self_attn.dcp_size)
                             for layer in self.decoder_layers()}))

    def _sample_kernel(self, hidden, positions, params, width: int, spec: int) -> torch.Tensor:
        """[width, spec] the target's draw at each span token by kernels/sample.py, each token with its row's
        temperature, seed and filters (the platform's per-row sampling inputs) and its own position's draw."""
        from sglang_neuron.backend.stack import wrap_nki
        from sglang_neuron.models.mimo_v2.kernels.sample import sample
        from sglang_neuron.utils.hardware_config import get_logical_nc_config

        temperatures, seeds, filters = params
        out = wrap_nki(sample)[get_logical_nc_config()](
            hidden, self.norm.weight.view(1, -1), self.lm_head_T, self.lm_head_tp_group.rank_in_group_tensor,
            self._head_mesh(), temperatures.reshape(width, 1).to(torch.float32), seeds.reshape(width, 1).to(torch.int32),
            filters.reshape(width, 3).to(torch.float32), positions.reshape(-1, 1).to(torch.int32), eps=self.norm.eps,
            span=spec)
        return out[0]

    def _prefill_inputs(self, input_ids, positions) -> tuple:
        """A prefill chunk's embeddings and both kinds' RoPE tables in one kernel (kernels/prefill.prefill_inputs)."""
        from sglang_neuron.backend.stack import wrap_nki
        from sglang_neuron.models.mimo_v2.kernels.prefill import prefill_inputs
        from sglang_neuron.utils.hardware_config import get_logical_nc_config

        hidden, fc, fs, sc, ss = wrap_nki(prefill_inputs)[get_logical_nc_config()](
            input_ids.reshape(-1, 1).to(torch.int32), positions.reshape(-1, 1).to(torch.int32),
            self.embed_tokens.weight, self.lm_head_tp_group.rank_in_group_tensor, self._head_mesh(),
            self.full_rope.inv_freq.view(1, -1), self.swa_rope.inv_freq.view(1, -1))
        return hidden, {True: (sc, ss), False: (fc, fs)}

    def _first_layer(self, swa: bool) -> str:
        return next(self.config.layer_name(i) for i, s in enumerate(self.config.swa_layers) if s == swa)

    @torch.no_grad()
    def forward(self, input_ids, positions, attn_metadata, sampling_positions, rank, mtp_row_slots=None,
                mtp_ctrl=None, sampling_temperatures=None, sampling_seeds=None, sampling_filters=None,
                token_control=None, token_width=0):
        if self.mtp is not None and self.phase == "decode":
            return self._mtp_decode(attn_metadata, rank, mtp_row_slots, mtp_ctrl,
                                    (sampling_temperatures, sampling_seeds, sampling_filters))
        if self.phase == "decode" and can_run_kernel(positions):
            return self._decode_kernels(input_ids, positions, attn_metadata, sampling_temperatures, sampling_seeds,
                                        sampling_filters, token_control, token_width)
        if token_control is not None:
            # the CPU reference of the token relay (model_executor/token_io.FixedDecodeTokenIO)
            source = token_control[1]
            gathered = torch.index_select(input_ids, 0, source.clamp(min=0).to(torch.int64))
            input_ids = torch.where(source >= 0, gathered, token_control[0])[:positions.shape[0]]
        if self.phase == "prefill" and can_run_kernel(positions) and positions.shape[0] % 128 == 0:
            hidden, tables = self._prefill_inputs(input_ids, positions)
        else:
            hidden = self.embed_tokens(input_ids, scatter_tokens=False, rank=rank)
            tables = self._tables(positions)
        if self.phase == "prefill":
            # the chunk's real tokens, which the MoE maps alone (models/qwen3_moe/sglang_model.Qwen3MoeGraph.forward):
            # the runtime pads a chunk with its last real position
            if can_run_kernel(positions):
                # the kernels tell the padding by its repeated position (kernels/prefill.prefill_route)
                real, ids32 = None, positions.reshape(-1, 1).to(torch.int32)
            else:
                real, ids32 = torch.arange(positions.shape[0], device=positions.device) <= torch.argmax(positions), None
            for layer in self.layers:
                cos, sin = tables[layer.is_swa]
                attn = layer.self_attn
                hidden, normed = attn.prefill(hidden, cos, sin, attn_metadata[attn.name], layer.input_layernorm,
                                              layer.post_attention_layernorm)
                hidden = layer.mlp.prefill_residual(hidden, normed, real=real, positions=ids32)
        else:
            inputs = decode_masks(positions)
            for layer in self.layers:
                cos, sin = tables[layer.is_swa]
                attn = layer.self_attn
                hidden = hidden + attn.decode(hidden, layer.input_layernorm, cos, sin, attn_metadata[attn.name],
                                              inputs["swa" if layer.is_swa else "full"])
                hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        if getattr(self, "samples_in_forward", False):
            # the graph samples its rows itself (a CPU decode without MTP reaches here: the device's is _decode_kernels)
            out = self._sample_rows(hidden, positions, sampling_positions, sampling_temperatures, sampling_seeds,
                                    sampling_filters)
            if self.mtp is not None and self.phase == "prefill":
                self._mtp_prefill(hidden, input_ids, None, sampling_positions,
                                  attn_metadata[self._first_layer(True)], tables[True], rank)
            if token_width and out["token_ids"].shape[0] < token_width:
                out = dict(out)
                out["token_ids"] = torch.nn.functional.pad(out["token_ids"],
                                                           (0, token_width - out["token_ids"].shape[0]), value=0)
            return out
        logits = self.lm_head(self.norm(hidden.index_select(0, sampling_positions)))
        if self.mtp is not None:
            self._mtp_prefill(hidden, input_ids, logits, sampling_positions, attn_metadata[self._first_layer(True)],
                              tables[True], rank)
        return logits

    def _sample_rows(self, hidden, positions, sampling_positions, temperatures, seeds, filters, out_width=0) -> dict:
        """The sampled rows' tokens and logprobs, as host.SampledExecution draws them from the logits: on a device the
        final norm, the LM head, the draw and the logprobs in one kernel (kernels/sample.py, the rows gathered in it,
        ``sampling_positions`` None for every row of ``hidden``); the torch below is the CPU reference."""
        from sglang_neuron import sampling

        mode = sampling.sampler_mode()
        if can_run_kernel(hidden):
            from sglang_neuron.backend.stack import wrap_nki
            from sglang_neuron.models.mimo_v2.kernels.sample import K_POOL, sample
            from sglang_neuron.utils.hardware_config import get_logical_nc_config

            if mode not in ("greedy", "topk_topp"):
                raise ValueError(f"MiMo-V2 samples in kernels/sample.py: SGLANG_NEURON_SAMPLER=topk_topp or greedy, "
                                 f"not {mode}")
            k = int(sampling.MAX_TOP_LOGPROBS)
            if k > K_POOL:
                raise ValueError(f"SGLANG_NEURON_MAX_TOP_LOGPROBS={k}: the sampler kernel keeps {K_POOL} candidates")
            rows = None if sampling_positions is None else sampling_positions.reshape(-1, 1).to(torch.int32)
            n = hidden.shape[0] if rows is None else rows.shape[0]
            dev = hidden.device
            if mode == "greedy" or temperatures is None:
                # temperature 0 everywhere: candidate 0, the argmax
                temperatures = torch.zeros(n, dtype=torch.float32, device=dev)
                seeds = torch.zeros(n, dtype=torch.int32, device=dev)
                filters = None
            if filters is None:
                filters = torch.tensor([[float(sampling.TOP_K_ALL), 1.0, 0.0]], dtype=torch.float32,
                                       device=dev).expand(n, 3).contiguous()
            tok, lp, tv, ti = wrap_nki(sample)[get_logical_nc_config()](
                hidden, self.norm.weight.view(1, -1), self.lm_head_T, self.lm_head_tp_group.rank_in_group_tensor,
                self._head_mesh(), temperatures.reshape(n, 1).to(torch.float32), seeds.reshape(n, 1).to(torch.int32),
                filters.reshape(n, 3).to(torch.float32), positions.reshape(-1, 1).to(torch.int32), eps=self.norm.eps,
                rows=rows, top_logprobs=k, original_logprob=sampling.return_original_logprob(), out_width=out_width)
            out = {"token_ids": tok.reshape(-1), "logprobs": lp.reshape(-1)}
            if k:
                out.update(top_vals=tv, top_ids=ti)
            return out
        states = hidden if sampling_positions is None else hidden.index_select(0, sampling_positions)
        logits = self.lm_head(self.norm(states))
        group = self.lm_head_tp_group
        rank = group.rank_in_group_tensor
        if mode == "greedy" or temperatures is None:
            return sampling.greedy_sharded_logits(logits, group, rank, self.config.vocab_size)
        at = positions if sampling_positions is None else sampling.sampled_positions(positions, sampling_positions)
        keys = sampling.noise_row_keys(seeds, at)
        if mode == "topk_topp":
            return sampling.sample_filtered_sharded_logits(logits, group, rank, self.config.vocab_size, temperatures,
                                                           keys, filters)
        return sampling.sample_sharded_logits(logits, group, rank, self.config.vocab_size, temperatures, keys)

    def _decode_kernels(self, input_ids, positions, attn_metadata, temperatures, seeds, filters, token_control=None,
                        token_width=0) -> dict:
        """One decode step without MTP on a device, as kernels: the step's inputs (kernels/step.decode_inputs: both
        kinds' RoPE tables, slots and position-mask inputs), the embedding (head.embed), one kernel a layer
        (kernels/layer.py, masked by position, the full layers over KV data parallelism) and the sampler with its
        logprobs (kernels/sample.py)."""
        from sglang_neuron.backend.stack import resolve_full_mesh, wrap_nki
        from sglang_neuron.models.mimo_v2.kernels.step import decode_inputs
        from sglang_neuron.utils.hardware_config import get_logical_nc_config

        if not all(self._layer_kernel_serves(layer) for layer in self.layers):
            raise ValueError("MiMo-V2 decodes on the one-kernel layers (kernels/layer.py): every MoE layer's experts "
                             "in the MoE half kernel")
        md = {True: attn_metadata[self._first_layer(True)], False: attn_metadata[self._first_layer(False)]}
        rows = positions.shape[0]
        half = self.full_rope.inv_freq.shape[0] + self.full_rope.pass_through
        attn0 = self.layers[0].self_attn
        dcp = {}
        if attn0.dcp_size > 1:
            # DCP: this rank's share's bounds and the rows whose new token it keeps (the runtime's)
            i32 = lambda t: t.reshape(-1).to(torch.int32)  # noqa: E731
            dcp = dict(dcp_full_end=i32(md[False]["dcp_kv_end"]), dcp_swa_end=i32(md[True]["dcp_kv_end"]),
                       dcp_swa_start=i32(md[True]["dcp_kv_start"]), dcp_owner=i32(md[True]["dcp_active_owner"]))
        outs = wrap_nki(decode_inputs)[get_logical_nc_config()](
            positions.reshape(-1).to(torch.int32), md[False]["slot_mapping"].reshape(-1).to(torch.int32),
            md[True]["slot_mapping"].reshape(-1).to(torch.int32), md[True]["swa_kv_pos_offset"].reshape(-1).to(
                torch.int32), self.full_rope.inv_freq.view(1, -1), self.swa_rope.inv_freq.view(1, -1),
            window=int(self.config.swa.sliding_window), half=half, mask_heads=self._mask_heads(), **dcp)
        full_slots, swa_slots, full_pos, swa_pos, swa_start, fc, fs, sc, ss, *masks = outs
        layout = lambda t: t.reshape(t.shape[0], rows, 1)  # noqa: E731
        step = {("cos", False): layout(fc), ("sin", False): layout(fs), ("cos", True): layout(sc),
                ("sin", True): layout(ss), ("slots", False): full_slots, ("slots", True): swa_slots,
                ("pos", False): full_pos, ("pos", True): swa_pos, ("start", True): swa_start,
                "mesh": tuple(tuple(int(r) for r in cell) for cell in resolve_full_mesh(attn0.attn_tp_group)),
                "ebase": self._expert_base(), "kv_scales": attn0._scale_tensors(positions.device)}
        step.update({("mask", h): m for h, m in zip(self._mask_heads(), masks)})
        if token_control is None:
            hidden = self.embed(input_ids.reshape(-1))
        else:
            # the relay's ids, assembled in the embedding kernel (kernels/head.embed's control)
            from sglang_neuron.models.mimo_v2.kernels.head import embed

            hidden = wrap_nki(embed)[get_logical_nc_config()](
                input_ids.reshape(-1, 1).to(torch.int32), self.embed_tokens.weight,
                self.lm_head_tp_group.rank_in_group_tensor, self._head_mesh(), control=token_control, rows=rows)
        for layer in self.layers:
            k = layer.is_swa
            hidden = self._layer_call(layer, hidden.view(rows, 1, -1),
                                      attn_metadata[layer.self_attn.name]["block_table_tensor"], None,
                                      step[("cos", k)], step[("sin", k)], step[("slots", k)], step)
        return self._sample_rows(hidden, positions, None, temperatures, seeds, filters, out_width=token_width)

    # -- NEURON_MTP (mtp_step.py) -------------------------------------------------------------------------------------

    def _mtp_prefill(self, hidden, input_ids, logits, sampling_positions, md, table, rank) -> None:
        """Every MTP layer's KV over the chunk (mtp.py), from the target's states before its final norm: layer k reads
        the chunk's tokens k + 1 ahead, the target's greedy pick and the layers' drafts after its last token."""
        if can_run_kernel(hidden) and hidden.shape[0] % 128 == 0:
            # on a device: the target's pick at the chunk's last real row (head.greedy gathering it), then per layer
            # the rotation and projection (kernels/prefill.prefill_mtp_project), its attention and MLP kernels and its
            # draft (head.greedy)
            at32 = sampling_positions.reshape(-1)[-1:].reshape(1, 1).to(torch.int32)
            token = self.greedy(hidden, self.norm, at32).reshape(1, 1)
            ids = input_ids.reshape(-1, 1).to(torch.int32)
            for k, layer in enumerate(self.mtp):
                x, ids = layer.project_chunk(self, ids, at32, token, hidden, rank)
                states = layer.prefill(self, hidden, None, md, *table, rank, x=x)
                if k + 1 < len(self.mtp):
                    token = self.greedy(states, layer.final_layernorm, at32).reshape(1, 1)
            return
        if logits is None:
            logits = self.lm_head(self.norm(hidden.index_select(0, sampling_positions)))
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
        from sglang_neuron import sampling
        from sglang_neuron.backend.stack import resolve_full_mesh
        from sglang_neuron.models.mimo_v2 import mtp_step
        from sglang_neuron.models.mimo_v2.attention import span_mask, span_slots

        md = {True: attn_metadata[self._first_layer(True)], False: attn_metadata[self._first_layer(False)]}
        width, spec = md[False]["block_table_tensor"].shape[0], mtp_step.SPEC
        slots = row_slots.to(torch.int64)
        heads = self.layers[0].self_attn.num_heads
        windows = {True: self.config.swa.sliding_window, False: None}
        ebase = self._expert_base()
        on_device = can_run_kernel(row_slots)
        if on_device and not all(self._layer_kernel_serves(layer) for layer in self.decoder_layers()):
            raise ValueError("NEURON_MTP decodes on the one-kernel layers (kernels/layer.py), which do not take DCP: "
                             "launch the MTP decode without --dcp-size")
        if on_device and sampling.sampler_mode() not in ("greedy", "topk_topp"):
            raise ValueError(f"NEURON_MTP samples in kernels/sample.py: SGLANG_NEURON_SAMPLER=topk_topp or greedy, "
                             f"not {sampling.sampler_mode()}")
        kernel_step = None
        if on_device:
            # the step's inputs in one kernel (kernels/step.py): the span's tokens and positions, both kinds' slots,
            # position masks and RoPE tables in the layer kernels' layouts
            span, kernel_step = self._step_kernel(md, row_slots, ctrl, width, spec)
            kernel_step.update({"mesh": tuple(tuple(int(r) for r in cell) for cell in
                                              resolve_full_mesh(self.layers[0].self_attn.attn_tp_group)),
                                "ebase": ebase, "kv_scales": self.layers[0].self_attn._scale_tensors(
                                    md[False]["block_table_tensor"].device)})
            positions = span["positions"]
            masks = {True: None, False: None}
            kv_slots = {True: kernel_step[("slots", True)].reshape(-1), False: kernel_step[("slots", False)].reshape(-1)}
            tables = None
            hidden = self.embed(span["tokens"].reshape(-1))
        else:
            # the CPU reference (torch): the same step, built from the carry in the graph
            span = mtp_step.span_inputs(self.mtp_carry.index_select(0, slots), self.mtp_len.index_select(0, slots),
                                        ctrl)
            positions = span["positions"]
            prior = span["base"].reshape(width, 1).expand(width, spec)
            masks = {k: span_mask(md[k], prior, positions, window=windows[k], num_heads=heads) for k in md}
            kv_slots = {k: span_slots(md[k], positions, span["valid"], window=windows[k]) for k in md}
            tables = self._tables(positions.reshape(-1))
            hidden = self.embed(span["tokens"].reshape(-1))
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
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        # the kernels' int32 draws stay int32 into the step's tail kernels ("raw")
        keep_i32 = "raw" in span
        if params[0] is None or sampling.sampler_mode() == "greedy":
            # the greedy target: the final norm, the LM head and the argmax in one kernel (kernels/head.greedy)
            sampled = self.greedy(hidden, self.norm)
            sampled = sampled if keep_i32 else sampled.reshape(width, spec).to(torch.int64)
        elif on_device:
            # the drawing target: the final norm, the LM head and the tail rule in one kernel (kernels/sample.py)
            sampled = self._sample_kernel(hidden, span["raw"]["positions"] if keep_i32 else positions, params, width,
                                          spec)
            sampled = sampled if keep_i32 else sampled.reshape(width, spec).to(torch.int64)
        else:
            sampled = self._span_sample(self.lm_head(self.norm(hidden)), positions, params, width, spec)
        if "raw" in span:
            return {"token_ids": self._mtp_tail_kernels(span, hidden, sampled, md[True], kernel_step, row_slots,
                                                        width, spec, rank)}
        out, kept = mtp_step.accept(span["tokens"], sampled, span)
        bonus = sampled.gather(1, kept.reshape(-1, 1))
        kernel = None if kernel_step is None else (kernel_step[("cos", True)], kernel_step[("sin", True)],
                                                   kernel_step[("slots", True)], kernel_step)
        drafts = self._mtp_drafts(span, sampled, hidden, kept, md[True], masks[True], kv_slots[True],
                                  tables[True] if tables is not None else (None, None), rank, kernel)
        self.mtp_carry.index_copy_(0, slots, torch.cat([bonus, drafts], dim=1).to(torch.int32))
        self.mtp_len.index_copy_(0, slots, (span["base"] + kept + 1).to(torch.int32))
        if self.mtp_width > width:
            out = torch.cat([out, torch.full((self.mtp_width - width, spec), -1, dtype=out.dtype, device=out.device)])
        return {"token_ids": out.to(torch.int32)}

    def _mtp_tail_kernels(self, span, hidden, sampled, md, step, row_slots, width: int, spec: int, rank):
        """The step after the target's draws, in kernels (kernels/accept.py): the verify, each MTP layer's pass and
        greedy draft at the row's last kept token (head.greedy gathering that row), the tokens rotated for the next
        layer, and the carry commit, whose result is the step's output."""
        from sglang_neuron.backend.stack import wrap_nki
        from sglang_neuron.models.mimo_v2.kernels.accept import accept, commit, rotate
        from sglang_neuron.utils.hardware_config import get_logical_nc_config

        grid = get_logical_nc_config()
        raw = span["raw"]
        drawn = sampled.reshape(-1, 1) if sampled.dtype == torch.int32 else sampled.reshape(-1, 1).to(torch.int32)
        out, kept, bonus, kept_row = wrap_nki(accept)[grid](raw["tokens"], drawn, raw["valid"], raw["fresh"],
                                                            raw["live"], span=spec, out_width=self.mtp_width)
        kernel = (step[("cos", True)], step[("sin", True)], step[("slots", True)], step)
        tokens, drafts = drawn, []
        for k, layer in enumerate(self.mtp):
            state, _ = layer.decode(self, tokens.reshape(-1), hidden, None, None, md, None, None, spec, None, rank,
                                    kernel)
            draft = self.greedy(state, layer.final_layernorm, kept_row).reshape(width, 1)
            drafts.append(draft)
            if k + 1 < len(self.mtp):
                tokens = wrap_nki(rotate)[grid](tokens, kept, draft)
        return wrap_nki(commit)[grid](self.mtp_carry, self.mtp_len, row_slots.to(torch.int32), bonus, *drafts,
                                      raw["base"], kept, out)

    def _mtp_drafts(self, span, sampled, hidden, kept, md, mask, kv_slots, table, rank, kernel=None):
        """[W, NUM_STEPS] the MTP layers' drafts after each row's kept run (mtp.py), the CPU reference: each layer's
        pass over the span writes its KV and gives its draft at the row's last kept position."""
        width, spec = sampled.shape
        size = hidden.shape[-1]
        at_kept = lambda state: state.view(width, spec, size).gather(  # noqa: E731
            1, kept.reshape(width, 1, 1).expand(width, 1, size)).reshape(width, size)
        # one draft a layer, each layer's tokens one position further ahead (mtp.rotate)
        tokens, drafts = sampled, []
        for k, layer in enumerate(self.mtp):
            state, _ = layer.decode(self, tokens.reshape(-1), hidden, *table, md, mask, kv_slots, spec, None, rank,
                                    kernel)
            drafts.append(layer.draft(self, at_kept(state)))
            if k + 1 < len(self.mtp):
                tokens = mtp.rotate(tokens, kept, drafts[-1])
        return torch.stack(drafts, dim=1)

    def _expert_base(self):
        """The MoE half kernel's ``ebase``: this rank's EP rank [1] int32 (the kernel multiplies it by its local
        expert count), None without MoE."""
        moe = next((m.mlp for m in self.layers if isinstance(m.mlp, MiMoV2MoE)), None)
        if moe is None:
            return None
        if moe.ep_group is None:
            return torch.zeros(1, dtype=torch.int32, device=moe.gate_up_weight.device)
        return moe.ep_group.rank_in_group_tensor

    @staticmethod
    def _layer_kernel_serves(layer) -> bool:
        """Whether ``layer``'s decode runs as one kernel: a dense layer, or an MoE layer whose experts the MoE half
        kernel takes (moe.MiMoV2MoE.fused)."""
        return isinstance(layer.mlp, DenseMLP) or getattr(layer.mlp, "fused", False)

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
        dcp = attn.dcp_size
        kvdp = self._kvdp(layer, rows) if posid and dcp == 1 else 1
        heads_attn = attn.num_heads * dcp  # the heads its attention sees: the DCP group's gathered
        if posid:
            # the span's active part alone (the prior is masked by position): token q sees the span's tokens k <= q
            built = step.get(("mask", int(heads_attn)))
            if built is not None and tuple(built.shape) == (span, rows * heads_attn, span):
                # the step kernel's (kernels/step.py), the same memory for any kvdp: a view, no XLA op
                mask = built.view(span, rows // kvdp, heads_attn * kvdp, span)
            else:
                k_idx = torch.arange(span, device=x.device).reshape(span, 1, 1, 1)
                q_idx = torch.arange(span, device=x.device).reshape(1, 1, 1, span)
                mask = (k_idx <= q_idx).expand(span, rows // kvdp, heads_attn * kvdp, span).to(torch.uint8)
        sink = attn._sink_column() if dcp == 1 else (None if attn.dcp_sink is None else attn.dcp_sink.reshape(-1, 1))
        attn_args = (x, layer.input_norm_f32, attn.qkv_weight, cos, sin, table, attn.k_cache, attn.v_cache, mask,
                     sink, slots, attn._kernel_o_weight())
        k_scale, v_scale = step.get("kv_scales", (None, None))
        common = dict(k_scale=k_scale, v_scale=v_scale, eps=layer.input_layernorm.eps,
                      softmax_scale=attn.scaling / attn.kv_scale, v_head_dim=attn.v_head_dim)
        if dcp > 1:
            from sglang_neuron.ops.attention.attention_decode import dcp_collective_mode

            common.update(pos=step[("pos", layer.is_swa)], start=step.get(("start", layer.is_swa)), dcp=dcp,
                          dcp_mesh=attn._mesh(), dcp_mode=dcp_collective_mode(dcp))
        elif kvdp > 1:
            common.update(pos=step[("pos", False)], rank=attn.attn_tp_group.rank_in_group_tensor, kvdp=kvdp,
                          kvdp_mesh=self._kvdp_mesh(layer, kvdp))
        elif posid:
            common.update(pos=step[("pos", layer.is_swa)], start=step.get(("start", layer.is_swa)))
        gamma = layer.post_attention_layernorm.weight.view(1, -1)
        if isinstance(mlp, DenseMLP):
            return wrap_nki(dense_layer)[get_logical_nc_config()](
                *attn_args, gamma, mlp.gate_weight, mlp.up_weight, mlp.down_weight, step["mesh"], **common)
        if mlp.dp_size > 1:
            # DP attention (a PD decode): the experts over every DP group's rows (kernels/layer._dp_moe)
            from sglang_neuron.backend.stack import resolve_full_mesh

            common.update(dp=mlp.dp_size, dp_rank=mlp.dp_group.rank_in_group_tensor,
                          world_mesh=tuple(tuple(int(r) for r in cell)
                                           for cell in resolve_full_mesh(mlp.wide_ep_group)))
        return wrap_nki(moe_layer)[get_logical_nc_config()](
            *attn_args, gamma, mlp.router_wT, mlp.expert_bias.view(1, -1), step["ebase"], mlp.gate_up_weight,
            mlp.down_weight, mlp.gate_up_scale, mlp.down_scale, step["mesh"], route_scale=float(mlp.scaling),
            top_k=mlp.top_k, fp8_act=True, **common)

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
