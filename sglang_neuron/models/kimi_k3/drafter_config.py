# SPDX-License-Identifier: Apache-2.0
"""The Kimi-K3 DSPARK drafter's config type, for Transformers' AutoConfig.

The drafter checkpoint's ``config.json`` names ``model_type`` ``k3_dspark`` and ships no configuration code, so
``AutoConfig.from_pretrained`` does not know it, and SGLang reads the draft model's config with it while it resolves
the speculative arguments (``arg_groups.speculative_hook``, the draft architecture check) and again for the draft
model config. ``register`` adds a plain config class under that type, in every process of a Neuron launch
(``general_plugin``). The decode reads the drafter's own fields from the file and checks them
(``decode.weights_draft``); this class only carries them.
"""

from __future__ import annotations

from transformers import AutoConfig, PretrainedConfig

MODEL_TYPE = "k3_dspark"


class K3DSparkConfig(PretrainedConfig):
    """The drafter's ``config.json`` fields as attributes."""

    model_type = MODEL_TYPE


def register() -> None:
    """Register ``K3DSparkConfig`` under ``k3_dspark`` with AutoConfig; again is a no-op."""
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    if MODEL_TYPE in CONFIG_MAPPING:
        return
    AutoConfig.register(MODEL_TYPE, K3DSparkConfig)
