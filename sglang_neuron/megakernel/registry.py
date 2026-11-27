# SPDX-License-Identifier: Apache-2.0
"""The megakernel specs the plugin ships, by (HF architecture, device target), registered in SGLang's spec registry at
plugin load on their target only through the shared SGLang registry."""
from __future__ import annotations

# each ("<HF architecture>", "<device target>"): "<module>:<attribute>"
SPECS: dict = {
    # Qwen3.5's giga-kernel decode and prefill: aggregated or either PD role on trn3
    ("Qwen3_5MoeForCausalLM", "trn3"): "sglang_neuron.models.qwen35.megakernel:QWEN35_GIGA_SPEC",
}


def register_specs() -> list:
    """Register every shipped spec path on its target; the (architecture, target) pairs registered."""
    from sglang_neuron.megakernel import upstream

    for (architecture, target), path in SPECS.items():
        upstream.require().register(architecture=architecture, path=path, target=target)
    return list(SPECS)
