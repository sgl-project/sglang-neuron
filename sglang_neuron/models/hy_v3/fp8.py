# SPDX-License-Identifier: Apache-2.0
"""Per-tensor FP8 weights read as bf16.

tencent/Hy3-FP8 stores every linear weight but the embedding, the LM head and the routers as ``float8_e4m3fn`` with
one ``weight_scale`` per tensor (``W = weight * weight_scale``; its ``input_scale`` is for FP8 activations, which the
bf16 graphs do not use). A parameter built from such weights maps its weight keys and, after them, their scales
(``with_scales``); ``dequantizing`` wraps the parameter's loader so that it sees bf16 slices, and the loaders of the
bf16 models (fused QKV, gate/up, row-parallel, the per-expert stacks) shard them unchanged. Only the shard a loader
reads is converted."""

import torch

from sglang_neuron.models.transformer.config import DTYPE
from sglang_neuron.utils.weight_loader import SafetensorsWeightLoader


class Fp8Slice:
    """A checkpoint slice of an FP8 weight that reads as bf16: ``weight[key] * scale``."""

    def __init__(self, weight, scale: torch.Tensor):
        self._weight = weight
        self._scale = scale.to(torch.float32)

    def get_shape(self):
        return self._weight.get_shape()

    def __getitem__(self, key):
        return (self._weight[key].to(torch.float32) * self._scale).to(DTYPE)


def with_scales(keys) -> list:
    """The checkpoint keys of an FP8 parameter: its weight keys, then each one's ``weight_scale``."""
    keys = [keys] if isinstance(keys, str) else list(keys)
    if not all(k.endswith(".weight") for k in keys):
        raise ValueError(f"FP8 weight keys end in .weight: {keys[:2]}")
    return keys + [k.removesuffix(".weight") + ".weight_scale" for k in keys]


def dequantizing(loader: SafetensorsWeightLoader) -> SafetensorsWeightLoader:
    """``loader`` over the bf16 view of FP8 weights, for a mapping built with ``with_scales``."""

    def transform(slices: list, rank: int) -> torch.Tensor:
        n = len(slices) // 2
        weights = [Fp8Slice(w, s[...]) for w, s in zip(slices[:n], slices[n:])]
        return loader.load(weights, rank)

    return SafetensorsWeightLoader(transform=transform)
