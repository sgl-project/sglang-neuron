# SPDX-License-Identifier: Apache-2.0
"""The megakernel specs the plugin ships, by (HF architecture, device target), registered in SGLang's spec registry at
plugin load on their target only (``spec.register(..., isa=...)``, the target registry of
``sglang_patches/megakernel_platform.py``), so a CUDA spec of the same architecture is never shadowed."""
from __future__ import annotations

# each ("<HF architecture>", "<device target>"): "<module>:<attribute>"
SPECS: dict = {
    # Qwen3.5's giga-kernel decode and prefill: aggregated or either PD role on trn3
    ("Qwen3_5MoeForCausalLM", "trn3"): "sglang_neuron.models.qwen35.megakernel:QWEN35_GIGA_SPEC",
}


def register_specs() -> list:
    """Register every shipped spec path on its target; the (architecture, target) pairs registered."""
    from sglang_neuron.sglang_patches.megakernel_platform import register_spec

    for (architecture, target), path in SPECS.items():
        # with a target the seam keeps its own registry and never calls upstream's register
        register_spec(None, architecture=architecture, path=path, isa=target)
    return list(SPECS)
