# SPDX-License-Identifier: Apache-2.0
"""The Llama role graphs: the shared dense transformer, its rotary tables over Llama 3.1's rescaled frequencies when the
checkpoint has a ``llama3`` RoPE scaling."""

from __future__ import annotations

import math

import torch

from sglang_neuron.models.llama.config import Llama3Scaling
from sglang_neuron.models.transformer.config import TransformerConfig
from sglang_neuron.models.transformer.model import RotaryEmbedding, TransformerForCausalLM


def llama3_inv_freq(inv_freq: torch.Tensor, scaling: Llama3Scaling) -> torch.Tensor:
    """The ``llama3`` rescaling of plain inverse frequencies (transformers' ``_compute_llama3_parameters``): by
    wavelength ``2 pi / f``, above ``original / low_freq_factor`` divide by ``factor``, below ``original /
    high_freq_factor`` keep, and between the two blend linearly in ``original / wavelength``. The cos/sin tables are not
    rescaled (the llama3 attention factor is 1)."""
    low_wavelen = scaling.original_max_position_embeddings / scaling.low_freq_factor
    high_wavelen = scaling.original_max_position_embeddings / scaling.high_freq_factor
    wavelen = 2 * math.pi / inv_freq
    scaled = torch.where(wavelen > low_wavelen, inv_freq / scaling.factor, inv_freq)
    if scaling.high_freq_factor == scaling.low_freq_factor:
        # no band between the two wavelengths
        return scaled
    smooth = ((scaling.original_max_position_embeddings / wavelen - scaling.low_freq_factor)
              / (scaling.high_freq_factor - scaling.low_freq_factor))
    blended = (1 - smooth) * scaled / scaling.factor + smooth * scaled
    medium = (wavelen >= high_wavelen) & (wavelen <= low_wavelen)
    return torch.where(medium, blended, scaled)


class Llama3RotaryEmbedding(RotaryEmbedding):
    """The shared rotary tables over the llama3-rescaled frequencies: the same positions, shapes and dtype, from a
    rescaled ``inv_freq``."""

    def __init__(self, config: TransformerConfig, scaling: Llama3Scaling):
        super().__init__(config)
        # assigning a tensor to a registered buffer's name keeps it a (non-persistent) buffer
        self.inv_freq = llama3_inv_freq(self.inv_freq, scaling)


class LlamaGraph(TransformerForCausalLM):
    """One role graph of a Llama checkpoint: the dense core, its rotary tables rescaled for a llama3 RoPE."""

    def __init__(self, config: TransformerConfig, phase: str, scaling: Llama3Scaling | None):
        super().__init__(config, phase)
        if scaling is not None:
            self.rotary_emb = Llama3RotaryEmbedding(config, scaling)
