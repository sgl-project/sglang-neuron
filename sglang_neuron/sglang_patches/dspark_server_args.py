# SPDX-License-Identifier: Apache-2.0
"""Let SGLang resolve DSPARK speculative decoding on Neuron for a model whose own worker serves it.

Upstream ``arg_groups.speculative_hook._handle_dspark`` refuses every device but CUDA, NPU and XPU, and the platform
sets SGLang's device to ``cpu`` (its tensors live on the host) before SGLang resolves the speculative arguments. The
name DSPARK is a built-in algorithm, which SGLang's registry does not let a plugin register again. So
``_handle_dspark`` is rebuilt from source with that one device check also accepting an out-of-tree platform that
serves DSPARK (``GATE`` -> ``NEURON_GATE``); every other check and resolution stays upstream's, the DP-attention
ones included, whose flags platform.apply_speculative_settings sets. NeuronSRTPlatform.supports_speculative_algorithm
answers yes only for the algorithm the launch's model declares (catalog.ModelSpec.speculative_algorithms), and
platform.refuse_unserved refuses any other launch first. SpeculativeAlgorithm.handle_server_args imports the function
from its module when it runs, so the rebuilt one is the one it calls. A launch hook: SGLang resolves the arguments in
the launching process.

Upstream: ``_handle_dspark`` would ask an out-of-tree platform, as ``_handle_dflash`` does.

Target md5-pinned against upstream SGLang 43881a7: ``_handle_dspark``."""

from __future__ import annotations

_MODULE = "sglang.srt.arg_groups.speculative_hook"
TARGET = f"{_MODULE}._handle_dspark"
_TARGET_MD5 = "3f04597c9c46327f865921ef50988f96"
GATE = """    if not cfg.device.startswith(("cuda", "npu", "xpu")):
"""
NEURON_GATE = """    if not (
        cfg.device.startswith(("cuda", "npu", "xpu"))
        or (
            current_platform.is_out_of_tree()
            and current_platform.supports_speculative_algorithm("DSPARK")
        )
    ):
"""


def handle_dspark():
    """Upstream's _handle_dspark with its device check extended by the out-of-tree platform's declaration."""
    from sglang_neuron.sglang_patches import source_guard

    original = source_guard.resolve(_MODULE, "_handle_dspark")
    source_guard.check(original, _TARGET_MD5, TARGET, __name__)
    return source_guard.with_substitutions(original, [(GATE, NEURON_GATE)])


def register(registry, replace) -> None:
    registry.register(TARGET, handle_dspark(), replace)
