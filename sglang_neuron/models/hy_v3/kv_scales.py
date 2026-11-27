# SPDX-License-Identifier: Apache-2.0
"""The per-layer K and V scales of Hy3's FP8 KV cache.

tencent/Hy3-FP8 ships no KV scales, and a unit scale puts V (about 0.1 in the early layers) into the FP8 subnormals, so
the FP8 cache needs a calibrated profile: ``$SGLANG_NEURON_HY3_KV_SCALES``, else ``<checkpoint>/kv_scales.safetensors``.
A profile holds, per decoder layer ``i``, ``model.layers.{i}.self_attn.k_scale`` and ``v_scale``: fp32 dequantization
scales (``x ~= fp8 * scale``) of K after the QK-norm and RoPE and of V, for an FP8 range of ``target_fp8_max`` (its
metadata; 240 when absent). vllm-neuron's ``contrib/models/hy_v3/kv_profiles/Hy3-FP8_neuron.safetensors`` is one. A
profile made for another range is rescaled to the platform's clamp."""

import json
import os

from safetensors import safe_open

from sglang_neuron.utils.dtype_utils import FP8_CLAMP_MAX

PROFILE_ENV = "SGLANG_NEURON_HY3_KV_SCALES"
PROFILE_FILE = "kv_scales.safetensors"


def profile_path(checkpoint_dir: str) -> str:
    path = os.environ.get(PROFILE_ENV) or os.path.join(checkpoint_dir, PROFILE_FILE)
    if not os.path.isfile(path):
        raise ValueError(f"Hy3's FP8 KV cache needs a KV scale profile: set {PROFILE_ENV} or put {PROFILE_FILE} in the "
                         f"checkpoint directory (vllm-neuron's contrib/models/hy_v3/kv_profiles/Hy3-FP8_neuron"
                         f".safetensors is one); {path} does not exist")
    return path


def load_kv_scales(checkpoint_dir: str, num_layers: int) -> list:
    """[(k_scale, v_scale)] per layer, dequantization scales for the platform's FP8 clamp."""
    path = profile_path(checkpoint_dir)
    with safe_open(path, "pt") as f:
        meta = f.metadata() or {}
        target = float(json.loads(meta["target_fp8_max"]) if "target_fp8_max" in meta else 240.0)
        rescale = target / FP8_CLAMP_MAX
        scales = []
        for i in range(num_layers):
            pair = []
            for name in ("k_scale", "v_scale"):
                key = f"model.layers.{i}.self_attn.{name}"
                if key not in f.keys():
                    raise ValueError(f"the KV scale profile {path} has no {key}")
                pair.append(float(f.get_tensor(key).float().reshape(-1)[0]) * rescale)
            scales.append(tuple(pair))
    return scales
