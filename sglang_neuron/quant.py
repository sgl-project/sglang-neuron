# SPDX-License-Identifier: Apache-2.0
"""The quantization configs SGLang gets for a quantized checkpoint on Neuron whose model loads the quantized weights
into its own kernels: an MXFP4 checkpoint and a Quark export. Each config quantizes nothing: it gives no layer a
quantization method."""

from __future__ import annotations

from typing import Any, List, Optional

import torch
from sglang.srt.layers.quantization.base_config import QuantizationConfig


class NeuronPassthroughQuantConfig(QuantizationConfig):
    def __init__(self, quant_method: str = "mxfp4"):
        super().__init__()
        self.quant_method = quant_method

    @classmethod
    def get_name(cls) -> str:
        return "mxfp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: dict) -> "NeuronPassthroughQuantConfig":
        # the model's graphs read the quantized weights; every model's host refuses a method it does not serve
        return cls(quant_method=str(config.get("quant_method", cls.get_name())))

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> Optional[Any]:
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class NeuronQuarkPassthroughQuantConfig(NeuronPassthroughQuantConfig):
    """A Quark export's config: SGLang knows the checkpoint's method by this name, and resolves its config from it."""

    @classmethod
    def get_name(cls) -> str:
        return "quark"


_PASSTHROUGH = {config.get_name(): config
                for config in (NeuronPassthroughQuantConfig, NeuronQuarkPassthroughQuantConfig)}


def passthrough_config(name: str) -> type:
    """The inert config class SGLang gets for the quantization method ``name`` (mxfp4 or quark)."""
    return _PASSTHROUGH[name]
