# SPDX-License-Identifier: Apache-2.0
"""Choose a host-side latent gain without changing the static KV row scale."""

import math

import torch

from sglang_neuron.models.glm52.kernels.decode.attention.mla.kv_row_layout import (  # noqa: E501
    E4M3FN_MAX,
)
from sglang_neuron.models.glm52.layout.glm_kv_row import STATIC_SCALE
from sglang_neuron.utils.weight_loader import SafetensorsWeightLoader


def static_kv_latent_gain(max_abs_gamma: float, latent_width: int) -> float:
    """Boost small RMSNorm outputs while leaving larger outputs unchanged.

    RMSNorm's absolute output is at most sqrt(width) * max(abs(gamma)).
    Keep that bound below half the static row's FP8 range (E4M3FN_MAX * STATIC_SCALE). A power-of-two gain can
    fold into gamma and its inverse into both latent projections.
    """
    if not math.isfinite(max_abs_gamma) or max_abs_gamma < 0:
        raise ValueError("KV normalization weights must have finite magnitudes")
    if latent_width <= 0:
        raise ValueError("KV latent width must be positive")
    if max_abs_gamma == 0:
        return 1.0
    bound = math.sqrt(latent_width) * max_abs_gamma
    ratio = (E4M3FN_MAX * STATIC_SCALE / 2) / bound
    if ratio < 2:
        return 1.0
    # Limit the fold so tiny projection weights retain ample dtype range.
    exponent = min(math.floor(math.log2(ratio)), 16)
    return math.ldexp(1.0, exponent)


def with_static_kv_balance(
    loader: SafetensorsWeightLoader, projection: bool = False
) -> SafetensorsWeightLoader:
    """Fold a latent gain into a norm, or its inverse into a projection.

    A projection mapping must append the raw KV normalization weight. Norm
    mappings still take one slice.
    """
    def transform(slices, rank):
        gamma = torch.as_tensor(slices[-1][:])
        gain = static_kv_latent_gain(float(gamma.float().abs().max()), gamma.numel())
        weight = loader.load(slices[:-1] if projection else slices, rank)
        return weight / gain if projection else weight * gain

    return SafetensorsWeightLoader(transform=transform)


def balance_static_kv_stacks(
    tensors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Fold the same gain into loaded decode weights without editing the store."""
    norms = tensors["kv_norm_w_all"]
    rows = norms.reshape(norms.shape[0], -1)
    gains = torch.tensor(
        [
            static_kv_latent_gain(float(row.float().abs().max()), row.numel())
            for row in rows
        ],
        dtype=torch.float32,
    )
    balanced = dict(tensors)
    for name in ("kv_norm_w_all", "q_absorb_w_all", "out_absorb_w_all"):
        weight = tensors[name]
        gain = gains.to(weight.dtype).reshape(
            (gains.numel(),) + (1,) * (weight.ndim - 1)
        )
        balanced[name] = weight * gain if name == "kv_norm_w_all" else weight / gain
    return balanced
