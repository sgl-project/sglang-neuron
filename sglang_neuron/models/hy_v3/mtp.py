# SPDX-License-Identifier: Apache-2.0
"""Hy3's MTP layer (``num_nextn_predict_layers`` 1: ``model.layers.<num_hidden_layers>`` of the checkpoint), the draft
of ``--speculative-algorithm NEURON_MTP``.

One pass at position i predicts the token at i + 2 from the target's state at i and the token at i + 1:

    h = eh_proj(cat(enorm(embed(t[i + 1])), hnorm(norm(h_i))))     h_i: the target's last layer output, norm its final norm
    h = final_layernorm(decoder_layer(h));  logits = lm_head(h)

with the target's embedding, final norm and LM head, and the embedding zeroed at position 0. The decoder layer is a
target layer (attention, MoE and shared expert) at index ``num_hidden_layers``, so its attention is the KV pool's layer
after the target's (the host sizes the pool one layer deeper). Its attention reads the whole table, never the
block-sparse one: the draft only proposes, the target checks every token.

The prefill writes the layer's KV over the chunk: the token after each position is the chunk's next input token, and
after the chunk's last one the target's own greedy pick (the sampled token is not in the graph). That pick, and the
chunk boundary of a long prompt, give the draft a slightly different state there, never a different output."""

import torch
from torch import nn

from sglang_neuron.layers import ColumnParallelLinear
from sglang_neuron.models.transformer.config import DTYPE
from sglang_neuron.models.transformer.model import DecoderLayer, RMSNorm
from sglang_neuron.parallel import groups
from sglang_neuron.utils.weight_loader import SafetensorsWeightLoader, set_weight_loader


# SGLANG_NEURON_HY3_PD_MTP_KV=1 on a --disaggregation-mode prefill: build the MTP layer and write its KV over each chunk,
# for a NEURON_MTP decode (whose pool holds the layer; the prefill itself runs no speculative step)
PD_MTP_KV_ENV = "SGLANG_NEURON_HY3_PD_MTP_KV"


def enabled() -> bool:
    """Whether the launch holds the MTP layer: it serves NEURON_MTP (the decode verifies and drafts with the layer), or
    it is the PD prefill of such a decode (``SGLANG_NEURON_HY3_PD_MTP_KV``)."""
    import os

    from sglang.srt.runtime_context import get_server_args

    from sglang_neuron.speculative.neuron_mtp import is_neuron_mtp

    args = get_server_args()
    if is_neuron_mtp(getattr(args, "speculative_algorithm", None)):
        return True
    if os.environ.get(PD_MTP_KV_ENV, "0") == "1":
        if str(getattr(args, "disaggregation_mode", "null")) != "prefill":
            raise ValueError(f"{PD_MTP_KV_ENV}=1 is for a --disaggregation-mode prefill; a decode serves the layer with "
                             "--speculative-algorithm NEURON_MTP")
        return True
    return False


class HyV3Mtp(nn.Module):
    def __init__(self, config, layer: DecoderLayer):
        super().__init__()
        hidden = config.hidden_size
        self.layer_idx = config.num_hidden_layers
        self.enorm = RMSNorm(hidden, config.rms_norm_eps)
        self.hnorm = RMSNorm(hidden, config.rms_norm_eps)
        self.final_layernorm = RMSNorm(hidden, config.rms_norm_eps)
        group = groups.get_neuron_attn_tp_group()
        self.eh_proj = ColumnParallelLinear(2 * hidden, hidden, dtype=DTYPE, gather_output=True,
                                            tp_group=group.device_group)
        rows = self.eh_proj.out_features_per_rank
        set_weight_loader(self.eh_proj.weight,
                          SafetensorsWeightLoader(transform=lambda slices, rank: slices[0][rank * rows:(rank + 1) * rows]))
        self.layer = layer

    @property
    def attn(self):
        return self.layer.self_attn

    def weight_mappings(self, layer_mappings: dict) -> dict:
        """This layer's checkpoint keys; ``layer_mappings`` the decoder layer's, as the target's layer mappings give them
        under ``mtp.layer``."""
        src = f"model.layers.{self.layer_idx}"
        return {"mtp.enorm.weight": f"{src}.enorm.weight", "mtp.hnorm.weight": f"{src}.hnorm.weight",
                "mtp.final_layernorm.weight": f"{src}.final_layernorm.weight",
                "mtp.eh_proj.weight": f"{src}.eh_proj.weight", **layer_mappings}

    def project(self, embeds: torch.Tensor, target: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """The decoder layer's input from the next tokens' embeddings and the target's normed states, [T, H]."""
        embeds = embeds * (positions != 0).unsqueeze(-1).to(embeds.dtype)
        return self.eh_proj(torch.cat([self.enorm(embeds), self.hnorm(target.to(embeds.dtype))], dim=-1))

    def prefill(self, graph, hidden: torch.Tensor, input_ids: torch.Tensor, positions: torch.Tensor, logits,
                attn_metadata: dict, cos, sin, rank, real: torch.Tensor) -> None:
        """Write this layer's KV over a prefill chunk: ``hidden`` [T, H] the target's last layer output, ``logits`` the
        target's at the sampling positions, ``real`` [T] the chunk's real rows. The target's greedy pick follows the
        chunk's last real token, which a padded chunk has before its last row."""
        from sglang_neuron.models.hy_v3.sglang_model import prefill_layer

        # the layer reads the target's tables and slots (the runtime builds metadata for the target's layers)
        md = attn_metadata[graph.layers[0].self_attn.name]
        pick = graph.greedy_token(logits).reshape(1).to(input_ids.dtype)
        last = torch.argmax(positions).reshape(1)
        next_ids = torch.cat([input_ids[1:], input_ids[-1:]]).index_copy(0, last, pick)
        x = self.project(graph.embed_tokens(next_ids, scatter_tokens=False, rank=rank), graph.norm(hidden), positions)
        prefill_layer(self.layer, x, lambda h: self.attn.prefill(h, cos, sin, md), real)

    def decode(self, graph, next_ids: torch.Tensor, prev: torch.Tensor, positions: torch.Tensor, inputs, cos, sin,
               rank) -> torch.Tensor:
        """One decode pass of this layer over ``positions`` [N] (a step's span, or one recurrent draft a row): the next
        tokens ``next_ids`` [N] with ``prev`` [N, H] the target's normed states (or the previous pass's output), into
        its KV under the target's DCP ``inputs``; returns its final-normed states [N, H]."""
        x = self.project(graph.embed_tokens(next_ids.to(torch.int32), scatter_tokens=False, rank=rank), prev,
                         positions)
        x = graph.decode_layer(self.layer, x, lambda h: self.attn.decode(h, cos, sin, inputs))
        return self.final_layernorm(x)
