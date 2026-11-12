# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V2's checkpoint loaders: each builds this attention-TP rank's shard of a parameter from the checkpoint's
Hugging Face tensors, in the layout the graphs read.

The checkpoint rotates its RoPE channels as interleaved pairs; the graphs rotate contiguous halves, so the loaders
reorder the RoPE output rows of the query and of the KV down-projection (``rope_halves``), which leaves every product
unchanged. Attention heads shard over the attention TP; the routed experts are split into contiguous blocks, one per
rank."""

import torch

from sglang_neuron.models.deepseek_v2.config import DeepseekV2Config
from sglang_neuron.utils.weight_loader import SafetensorsWeightLoader


def rope_halves(dim: int) -> torch.Tensor:
    """The order that takes RoPE channels from interleaved pairs (0, 1), (2, 3), ... to the halves [0, 2, ...] and
    [1, 3, ...] the graphs rotate."""
    return torch.cat([torch.arange(0, dim, 2), torch.arange(1, dim, 2)])


def q_loader(config: DeepseekV2Config, heads: int) -> SafetensorsWeightLoader:
    """q_proj [heads x (nope + rope), H] as this rank's [H, heads / tp x (nope + rope)], each head's RoPE rows in
    halves."""
    d, nope = config.q_head_dim, config.qk_nope_head_dim
    order = torch.cat([torch.arange(nope), nope + rope_halves(config.qk_rope_head_dim)])

    def transform(slices, rank):
        rows = slices[0][rank * heads * d:(rank + 1) * heads * d]
        return rows.view(heads, d, -1)[:, order].reshape(heads * d, -1).t()

    return SafetensorsWeightLoader(transform=transform)


def kv_a_loader(config: DeepseekV2Config) -> SafetensorsWeightLoader:
    """kv_a_proj_with_mqa [latent + rope, H], replicated, as [H, latent + rope] with its RoPE rows in halves."""
    order = torch.cat([torch.arange(config.kv_lora_rank),
                       config.kv_lora_rank + rope_halves(config.qk_rope_head_dim)])
    return SafetensorsWeightLoader(transform=lambda slices, rank: slices[0][:][order].t())


def kv_b_loader(config: DeepseekV2Config, heads: int, values: bool) -> SafetensorsWeightLoader:
    """This rank's heads of kv_b_proj [heads x (nope + v), latent]: the key part [heads / tp, nope, latent], which the
    absorbed attention folds into the query, or the value part [heads / tp, v, latent]."""
    nope, v = config.qk_nope_head_dim, config.v_head_dim

    def transform(slices, rank):
        rows = slices[0][rank * heads * (nope + v):(rank + 1) * heads * (nope + v)].view(heads, nope + v, -1)
        return rows[:, nope:] if values else rows[:, :nope]

    return SafetensorsWeightLoader(transform=transform)


def experts_gate_up_loader() -> SafetensorsWeightLoader:
    """The rank's experts' gate and up projections ([I, H] each, the mapping lists gate then up per expert) as [E,
    H, 2, I]."""
    return SafetensorsWeightLoader(transform=lambda slices, rank: torch.stack(
        [torch.stack([gate[:].t(), up[:].t()], dim=1) for gate, up in zip(slices[0::2], slices[1::2])]))


def experts_down_loader() -> SafetensorsWeightLoader:
    """The rank's experts' down projections ([H, I] each) as [E, I, H]."""
    return SafetensorsWeightLoader(transform=lambda slices, rank: torch.stack([down[:].t() for down in slices]))
