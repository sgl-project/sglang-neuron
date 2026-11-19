# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2's MTP layers (``model.mtp.layers.<k>`` of the checkpoint, ``num_nextn_predict_layers`` of them: 3 on
V2.6-Flash), the draft of ``--speculative-algorithm NEURON_MTP``.

MTP layer k at position i drafts the token at i + 2 + k from the target's state at i and the token at i + 1 + k
(SGLang's MiMoV2MTP, models/mimo_v2_nextn.py, as its multi-layer EAGLE runs it):

    h = eh_proj_k(cat(enorm_k(embed(t[i + 1 + k])), hnorm_k(h_i)))   h_i: the target's last layer output, before its
    h = layer_k(h);  logits = lm_head(final_layernorm_k(h))           final norm (return_hidden_states_before_norm)

with the target's embedding and LM head. Every layer reads the target's states (MiMo is not one of SGLang's
chain-style MTP models, whose layer k reads layer k - 1's). Each layer is a sliding-window attention layer (the sliding
kind's heads, window, sinks and RoPE base) and a dense MLP, its norms ``input_layernorm`` and ``pre_mlp_layernorm``.

``SGLANG_NEURON_MIMO_MTP_LAYERS`` sets the layers the draft runs (``LAYERS_ENV``):

- ``3`` (the default, the checkpoint's): a step's three drafts are the three layers' at the row's last kept position,
  each layer run over the step's span with the tokens one more position ahead than the layer before it; past the
  row's kept run those tokens are the drafts before it (SGLang's ``rotate_input_ids``);
- ``1``: the first layer alone, chained: each further draft is its pass at the next position from the previous draft
  and its previous pass's state (the EAGLE chain).

Each layer's KV is one pool layer after the target's, in the window pool (the host sizes the pool deeper and
sglang_patches/mimo_v2_launch lists the layers among the sliding ones), so it reads a target sliding layer's tables and
slots. The prefill writes every layer's KV over each chunk; the tokens past the chunk's last one are the target's
greedy pick and the layers' greedy drafts after it (the sampled token is not in the graph), which only gives a draft a
slightly different state there, never a different output. A layer's KV at the positions whose tokens were drafts stays
as written (SGLang's draft extend rewrites a few of them with the accepted tokens), which can only lower acceptance."""

from __future__ import annotations

import os

import torch
from torch import nn

from sglang_neuron.layers import ColumnParallelLinear
from sglang_neuron.models.transformer.config import DTYPE
from sglang_neuron.models.transformer.model import RMSNorm
from sglang_neuron.parallel import groups
from sglang_neuron.utils.weight_loader import SafetensorsWeightLoader, set_weight_loader

LAYERS_ENV = "SGLANG_NEURON_MIMO_MTP_LAYERS"
# SGLANG_NEURON_MIMO_PD_MTP_KV=1 on a --disaggregation-mode prefill: build the MTP layers and write their KV over each
# chunk, for a NEURON_MTP decode (whose pool holds the layers; the prefill itself runs no speculative step)
PD_MTP_KV_ENV = "SGLANG_NEURON_MIMO_PD_MTP_KV"


def enabled() -> bool:
    """Whether the launch holds the MTP layers: it serves NEURON_MTP (an aggregated server or a PD decode), or it is the
    PD prefill of such a decode (``PD_MTP_KV_ENV``)."""
    from sglang.srt.runtime_context import get_server_args

    from sglang_neuron.speculative.neuron_mtp import is_neuron_mtp

    args = get_server_args()
    if is_neuron_mtp(getattr(args, "speculative_algorithm", None)):
        return True
    if os.environ.get(PD_MTP_KV_ENV, "0") == "1":
        if str(getattr(args, "disaggregation_mode", "null")) != "prefill":
            raise ValueError(f"{PD_MTP_KV_ENV}=1 is for a --disaggregation-mode prefill; a decode or an aggregated "
                             "server holds the MTP layers with --speculative-algorithm NEURON_MTP")
        return True
    return False


def num_layers(environ=os.environ) -> int:
    """The MTP layers the draft runs (module docstring): 3 or 1."""
    value = environ.get(LAYERS_ENV, "3").strip()
    if value not in ("1", "3"):
        raise ValueError(f"{LAYERS_ENV}={value!r}: set 3 (the checkpoint's MTP layers, one a draft) or 1 (the first "
                         "layer, chained)")
    return int(value)


def rotate(tokens: torch.Tensor, at: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
    """Each row of ``tokens`` [R, S] one position ahead: its next token at every place before ``at`` [R], ``token`` [R]
    at ``at``, the rest unchanged (SGLang's rotate_input_ids, per row)."""
    width = tokens.shape[1]
    ahead = torch.cat([tokens[:, 1:], tokens[:, -1:]], dim=1)
    j = torch.arange(width, device=tokens.device).reshape(1, width)
    at = at.reshape(-1, 1)
    out = torch.where(j < at, ahead, tokens)
    return torch.where(j == at, token.reshape(-1, 1).to(tokens.dtype), out)


class MiMoMtp(nn.Module):
    """MTP layer ``index``: its KV is the pool's layer ``num_hidden_layers + index``."""

    def __init__(self, config, quantized: bool, index: int):
        from sglang_neuron.models.mimo_v2.model import DenseMLP, MiMoDecoderLayer

        super().__init__()
        hidden = config.hidden_size
        self.index = index
        self.src = f"model.mtp.layers.{index}"
        self.layer_idx = config.num_hidden_layers + index
        self.enorm = RMSNorm(hidden, config.rms_norm_eps)
        self.hnorm = RMSNorm(hidden, config.rms_norm_eps)
        self.final_layernorm = RMSNorm(hidden, config.rms_norm_eps)
        group = groups.get_neuron_attn_tp_group()
        self.eh_proj = ColumnParallelLinear(2 * hidden, hidden, dtype=DTYPE, gather_output=True,
                                            tp_group=group.device_group)
        rows = self.eh_proj.out_features_per_rank
        set_weight_loader(self.eh_proj.weight, SafetensorsWeightLoader(
            transform=lambda slices, rank: slices[0][rank * rows:(rank + 1) * rows]))
        self.layer = MiMoDecoderLayer(config, self.layer_idx,
                                      DenseMLP(hidden, config.intermediate_size, quantized), kind=config.swa)

    @property
    def attn(self):
        return self.layer.self_attn

    def weight_mappings(self, dst: str, fp8) -> dict:
        """This layer's parameters (under ``dst``) to their checkpoint keys; ``fp8`` gives a quantized weight's keys."""
        src, attn = self.src, f"{self.src}.self_attn"
        mappings = {
            f"{dst}.enorm.weight": f"{src}.enorm.weight", f"{dst}.hnorm.weight": f"{src}.hnorm.weight",
            f"{dst}.final_layernorm.weight": f"{src}.final_layernorm.weight",
            f"{dst}.eh_proj.weight": f"{src}.eh_proj.weight",
            f"{dst}.layer.self_attn.qkv_weight": fp8([f"{attn}.qkv_proj.weight"]),
            f"{dst}.layer.self_attn.o_weight": f"{attn}.o_proj.weight",
            f"{dst}.layer.input_layernorm.weight": f"{src}.input_layernorm.weight",
            f"{dst}.layer.input_norm_f32": f"{src}.input_layernorm.weight",
            f"{dst}.layer.post_attention_layernorm.weight": f"{src}.pre_mlp_layernorm.weight",
        }
        if self.attn.sink is not None:
            mappings[f"{dst}.layer.self_attn.sink"] = f"{attn}.attention_sink_bias"
        for proj in ("gate", "up", "down"):
            mappings[f"{dst}.layer.mlp.{proj}_weight"] = fp8([f"{src}.mlp.{proj}_proj.weight"])
        return mappings

    def project(self, embeds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """The layer's input from the next tokens' embeddings and the states before a final norm, [T, H]."""
        return self.eh_proj(torch.cat([self.enorm(embeds), self.hnorm(target.to(embeds.dtype))], dim=-1))

    def prefill(self, graph, hidden: torch.Tensor, next_ids: torch.Tensor, md: dict, cos, sin, rank) -> torch.Tensor:
        """Write this layer's KV over a prefill chunk from the target's states ``hidden`` [T, H] and the tokens
        ``next_ids`` [T] it reads at each position (``md``: a target sliding layer's metadata); returns its states
        before the final norm [T, H]."""
        layer = self.layer
        x = self.project(graph.embed_tokens(next_ids, scatter_tokens=False, rank=rank), hidden)
        x = x + self.attn.prefill(layer.input_layernorm(x), cos, sin, md)
        return x + layer.mlp(layer.post_attention_layernorm(x))

    def decode(self, graph, next_ids, prev, cos, sin, md, mask, slots, span: int, caches, rank, kernel=None) -> tuple:
        """One decode pass over ``span`` tokens a row: the next tokens ``next_ids`` [N] with ``prev`` [N, H] the states
        it reads, into the layer's KV at ``slots`` [N]. Returns (its states before the final norm [N, H], the caches as
        written). ``kernel`` (cos, sin, slots and the step's inputs in the one-kernel layer's layouts,
        model.MiMoV2Graph._layer_call) runs the layer as one kernel (kernels/layer.dense_layer)."""
        layer = self.layer
        x = self.project(graph.embed_tokens(next_ids.to(torch.int32), scatter_tokens=False, rank=rank), prev)
        if kernel is not None and caches is None:
            k_cos, k_sin, k_slots, step = kernel
            rows = x.shape[0] // span
            out = graph._layer_call(layer, x.view(rows, span, -1), md["block_table_tensor"], mask, k_cos, k_sin,
                                    k_slots, step)
            return out, (self.attn.k_cache, self.attn.v_cache)
        out, k_cache, v_cache = self.attn.span_decode(x, layer.input_layernorm, cos, sin, md, mask, slots, span,
                                                      caches)
        x = x + out
        x = x + layer.mlp(layer.post_attention_layernorm(x))
        return x, (k_cache, v_cache)

    def draft(self, graph, states: torch.Tensor) -> torch.Tensor:
        """The greedy draft of each of ``states`` [R, H] (before the final norm), int64 [R]."""
        return graph.greedy_token(graph.lm_head(self.final_layernorm(states))).reshape(-1).to(torch.int64)
