# SPDX-License-Identifier: Apache-2.0
"""Let SGLang build its decode-context-parallel groups on Neuron.

Upstream ``distributed.parallel_state.initialize_model_parallel`` builds the DCP groups (``--dcp-size D``: D
consecutive ranks of each TP group) only on CUDA, HIP, NPU and XPU, and refuses ``decode_context_parallel_size > 1``
anywhere else. It is rebuilt from source with that one platform check also accepting
``NeuronSRTPlatform.supports_decode_context_parallel`` (``GATE`` -> ``NEURON_GATE``); the groups and every other check
stay upstream's. distributed/decode_context_parallel.validate_dcp_launch refuses the DCP launches Neuron does not serve.
Registered on DCP launches only (serving_policy.dcp_size).

Upstream: ``initialize_model_parallel`` would ask the platform instead of listing device types.

Target md5-pinned against upstream SGLang 43881a7: ``initialize_model_parallel``."""

from __future__ import annotations

_MODULE = "sglang.srt.distributed.parallel_state"
TARGET = f"{_MODULE}.initialize_model_parallel"
_TARGET_MD5 = "21e20fecfcd644667b83639d52fef24f"
GATE = """    if decode_context_parallel_size > 1 and not (
        is_hip() or is_cuda() or _is_npu or _is_xpu
    ):
"""
NEURON_GATE = """    if decode_context_parallel_size > 1 and not (
        is_hip()
        or is_cuda()
        or _is_npu
        or _is_xpu
        or platforms.current_platform.supports_decode_context_parallel()
    ):
"""


def initialize_model_parallel():
    """Upstream's initialize_model_parallel with its DCP platform condition extended by the platform's declaration."""
    from sglang_neuron.sglang_patches import source_guard

    original = source_guard.resolve(_MODULE, "initialize_model_parallel")
    source_guard.check(original, _TARGET_MD5, TARGET, __name__)
    return source_guard.with_substitutions(original, [(GATE, NEURON_GATE)])


def register(registry, replace) -> None:
    registry.register(TARGET, initialize_model_parallel(), replace)
