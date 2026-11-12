# SPDX-License-Identifier: Apache-2.0
"""Build DP attention's MLP-sync token counts without pinned memory.

``ForwardBatch.prepare_mlp_sync_batch`` pins its ``global_num_tokens`` tensor unless SGLang itself runs on CPU
(``forward_batch_info._is_cpu``). On this platform the batch lives on the host, and torch sends the pinned-allocator
lookup to the renamed PrivateUse1 backend, which raises. The method is rebuilt from upstream's source with the pin
turned off. Everything else stays upstream's, including the padding fill value: ``_is_cpu`` also picks between the
CPU-graph and the CUDA-graph fill value, and the Neuron attention backend returns the same number for both.

Target md5-pinned against upstream SGLang 43881a7: ``ForwardBatch.prepare_mlp_sync_batch``.

Upstream: SGLang could decide the pin with the platform's is_pin_memory_available() instead of forward_batch_info's
_is_cpu; the module then goes away."""

from __future__ import annotations

_MODULE = "sglang.srt.model_executor.forward_batch_info"
_QUALNAME = "ForwardBatch.prepare_mlp_sync_batch"
TARGET = f"{_MODULE}.{_QUALNAME}"
_TARGET_MD5 = "5d1af57a8341ab1a75c086e70a6786c4"
UPSTREAM_PIN = "self.use_pin_memory = not _is_cpu"
NEURON_PIN = "self.use_pin_memory = False"


def prepare_mlp_sync_batch():
    """Upstream's ForwardBatch.prepare_mlp_sync_batch, rebuilt with its token-count tensor left unpinned."""
    from sglang_neuron.sglang_patches import source_guard

    original = source_guard.resolve(_MODULE, _QUALNAME)
    source_guard.check(original, _TARGET_MD5, TARGET, __name__)
    return source_guard.with_substitutions(original, [(UPSTREAM_PIN, NEURON_PIN)])


def register(registry, replace) -> None:
    registry.register(TARGET, prepare_mlp_sync_batch(), replace)
