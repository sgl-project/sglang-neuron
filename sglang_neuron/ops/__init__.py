# SPDX-License-Identifier: Apache-2.0
"""The NKI kernel calls of the model graphs; this namespace exports GPT-OSS's. All but the MXFP4 expert kernels
(moe_cte, moe_block_tkg) have a torch path for the CPU tests.

The exports load lazily: the kernels import ``nki``/``nkilib``, which only exist on the device, and a model whose
host code is importable off the device (models/qwen35's graphs and weight loader are, for the CPU tests) still
imports this package. Accessing an export without ``nki`` raises its ModuleNotFoundError there, as importing it
here did.
"""

import importlib

# first: the backend's environment, which nkilib's attention kernels read when they are imported
from sglang_neuron.backend import envs as _backend_envs  # noqa: F401

_EXPORTS = {
    "gen_attention_decode_mask": ".attention.attention_decode_mask",
    "segmented_attention": ".attention.attention_segmented_cte",
    "o_proj": ".attention.o_proj",
    "qkv_proj": ".attention.qkv",
    "swa_fused_attention": ".attention.swa_fused",
    "moe_block_tkg": ".moe.moe_block_tkg",
    "build_blockwise_mapping": ".moe.moe_blockwise",
    "moe_cte": ".moe.moe_cte",
    "rmsnorm_mx_prefill": ".moe.rmsnorm_mx_prefill",
    "slice_affinities_from_packed": ".moe.rmsnorm_mx_prefill",
    "rmsnorm_residual_prefill": ".rmsnorm_residual_prefill",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value
    return value
