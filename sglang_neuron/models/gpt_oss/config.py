# SPDX-License-Identifier: Apache-2.0
"""The GPT-OSS architecture the Neuron graphs build: the checkpoint's config with the hidden size and the expert
intermediate size padded to the kernels' tiling."""

import os
from dataclasses import dataclass, field, fields

import torch

# the graphs compute in bfloat16
DTYPE = torch.bfloat16
# the hidden size the kernels tile, padded from the checkpoint's
PADDED_HIDDEN_SIZE = 3072
EXPERT_FORMATS = ("mxfp4", "bf16")


def expert_format(environ=os.environ) -> str:
    """SGLANG_NEURON_GPT_OSS_EXPERTS: mxfp4 (default) runs the checkpoint's MXFP4 tiles in the MXFP4 MoE kernels, whose
    prefill quantizes activations to MX FP8; bf16 dequantizes the same tiles on the host into BF16 experts
    (models/gpt_oss/bf16.py), at about 4x the memory, as a BF16-activation reference. Not --quantization, which names
    the checkpoint's format (MXFP4 for both)."""
    value = environ.get("SGLANG_NEURON_GPT_OSS_EXPERTS") or "mxfp4"
    if value not in EXPERT_FORMATS:
        raise ValueError(f"SGLANG_NEURON_GPT_OSS_EXPERTS={value!r}: set it to {' or '.join(EXPERT_FORMATS)}, or "
                         "leave it unset for mxfp4")
    return value


@dataclass
class GptOssConfig:
    """GPT-OSS-120B by default; from_hf_config takes every field the checkpoint's config carries."""

    vocab_size: int = 201088
    hidden_size: int = PADDED_HIDDEN_SIZE
    unpadded_hidden_size: int = 2880
    num_hidden_layers: int = 36
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 64
    intermediate_size: int = 3072
    rms_norm_eps: float = 1e-5
    num_local_experts: int = 128
    num_experts_per_tok: int = 4
    sliding_window: int = 128
    rope_parameters: dict = field(default_factory=lambda: {
        "rope_type": "yarn", "rope_theta": 150000.0, "factor": 32.0, "beta_fast": 32.0, "beta_slow": 1.0,
        "original_max_position_embeddings": 4096, "truncate": False,
    })
    swiglu_limit: float = 7.0

    @classmethod
    def from_hf_config(cls, hf_config, intermediate_size: int) -> "GptOssConfig":
        """The config of the MXFP4 checkpoint ``hf_config``, with the expert intermediate size padded to
        ``intermediate_size`` (sglang_model.padded_intermediate_size)."""
        quantization = getattr(hf_config, "quantization_config", None)
        method = (quantization.get("quant_method") if isinstance(quantization, dict)
                  else getattr(quantization, "quant_method", None))
        if method != "mxfp4":
            raise ValueError(f"the Neuron GPT-OSS graphs serve MXFP4 checkpoints; this checkpoint's quant_method is "
                             f"{method!r}")
        # transformers names the checkpoint dtype "dtype"; older releases "torch_dtype"
        dtype = getattr(hf_config, "dtype", None) or getattr(hf_config, "torch_dtype", None)
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype, dtype)
        if dtype not in (None, DTYPE):
            raise ValueError(f"GPT-OSS checkpoint dtype {dtype}: the Neuron graphs compute in bfloat16 only")
        names = {f.name for f in fields(cls)}
        values = {name: getattr(hf_config, name) for name in names if hasattr(hf_config, name)}
        values["unpadded_hidden_size"] = values.get("hidden_size", cls.unpadded_hidden_size)
        values["hidden_size"] = PADDED_HIDDEN_SIZE
        values["intermediate_size"] = intermediate_size
        return cls(**values)
