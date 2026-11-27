# SPDX-License-Identifier: Apache-2.0
"""Inkling-2-Mini (Inkling2LoopForConditionalGeneration, text) on SGLang Neuron.

Importing the package registers the checkpoint's config classes (``inkling2_loop``) with transformers, whose release
in the image predates them: every process of a launch imports the package (general_plugin._register_model_package)
before SGLang reads the checkpoint's config."""

from sglang_neuron.models.inkling2.config import register_hf_configs

register_hf_configs()
