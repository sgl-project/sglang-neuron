# SPDX-License-Identifier: Apache-2.0
"""Give SGLang's quantization registry an MXFP4 entry on Neuron, where upstream registers none.

GPT-OSS checkpoints declare ``quant_method: mxfp4``. Upstream registers its ``Mxfp4Config`` only on CPU, CUDA, gfx95 and
XPU, and ``get_quantization_config`` refuses a name missing from ``QUANTIZATION_METHODS`` before it asks the platform,
so on Neuron the name would be refused outright. The entry added here is the inert ``NeuronPassthroughQuantConfig``,
which the platform's ``get_quantization_config`` also returns. A registry that already maps ``mxfp4`` to another
class is refused rather than replaced.

The relied-on target, ``layers.quantization.get_quantization_config`` (the registry check comes before the platform
lookup), is md5-pinned against upstream SGLang 43881a7.

Upstream: SGLang could ask the platform before refusing a quantization name, or let a platform register one; this module
then goes away.
"""

from __future__ import annotations

TARGET = "sglang.srt.layers.quantization.get_quantization_config"
_TARGET_MD5 = "0e857a25d35b20808eaa15920fada97b"
NAME = "mxfp4"


def register() -> None:
    """Add the inert MXFP4 entry to SGLang's registry; idempotent."""
    from sglang.srt.layers import quantization

    from sglang_neuron.quant import NeuronPassthroughQuantConfig
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check(quantization.get_quantization_config, _TARGET_MD5, TARGET, __name__)
    have = quantization.QUANTIZATION_METHODS.get(NAME)
    if have is not None and have is not NeuronPassthroughQuantConfig:
        raise RuntimeError(f"{__name__}: SGLang's quantization registry already maps {NAME!r} to "
                           f"{have.__module__}.{have.__qualname__}, which the Neuron model does not run")
    quantization.QUANTIZATION_METHODS[NAME] = NeuronPassthroughQuantConfig
