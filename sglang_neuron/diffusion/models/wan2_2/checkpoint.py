# SPDX-License-Identifier: Apache-2.0
"""Which checkpoints the Neuron Wan2.2 pipelines take: A14B, with BF16 weights.

JSON reads only, so the launcher can ask before torch-neuronx loads (to pick the environment
profile), as the pipeline routing does.
"""

from __future__ import annotations

from sglang_neuron.diffusion.model_index import read_component_config, read_model_index

WAN22_A14B_CLASS_NAMES = ("WanPipeline", "WanImageToVideoPipeline")


def is_wan22_a14b(model_path: str | None, class_name: str | None = None) -> bool:
    """Whether ``model_path`` is a Wan2.2 A14B checkpoint (of ``class_name``, when given).

    A14B is the only Wan2.2 family with two experts: ``model_index.json`` lists a
    ``transformer_2`` and a ``boundary_ratio``. Wan2.1 has no second expert; Wan2.2 TI2V-5B
    lists ``transformer_2`` as ``[null, null]`` and sets ``expand_timesteps``.
    """
    index = read_model_index(model_path)
    entry = index.get("transformer_2")
    has_second_expert = isinstance(entry, (list, tuple)) and len(entry) == 2 and entry[1]
    return (
        bool(has_second_expert)
        and index.get("boundary_ratio") is not None
        and not index.get("expand_timesteps", False)
        and index.get("_class_name") in WAN22_A14B_CLASS_NAMES
        and (class_name is None or index.get("_class_name") == class_name)
        and not is_quantized(model_path)
    )


def is_quantized(model_path: str) -> bool:
    """Whether the checkpoint's DiT ships pre-quantized weights (e.g. the NVFP4 export), for a
    local directory or a Hub ID.

    The Neuron pipelines load BF16 weights only.
    """
    return any(
        (read_component_config(model_path, subfolder) or {}).get("quantization_config")
        for subfolder in ("transformer", "transformer_2")
    )
