# SPDX-License-Identifier: Apache-2.0
"""Neuron platform and models for SGLang Diffusion on torch-neuronx."""

from importlib.util import find_spec


def activate() -> str | None:
    """``sglang.multimodal_gen.platforms`` entry point.

    Selects the Neuron diffusion platform when torch-neuronx is installed. Import-safe: it
    only checks that the package exists, so the HTTP process and the launcher never import
    SGLang internals or torch_neuronx, or open a NeuronCore here.
    """
    if find_spec("torch_neuronx") is None:
        return None
    return "sglang_neuron.diffusion.platform.NeuronDiffusionPlatform"
