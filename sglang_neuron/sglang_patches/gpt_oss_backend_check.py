# SPDX-License-Identifier: Apache-2.0
"""Let SGLang's GPT-OSS and Gemma 4 attention-backend checks accept the Neuron backend.

``model_hook.handle_model_specific_adjustments`` asserts that a ``GptOssForCausalLM`` or Gemma 4 server's prefill and
decode attention backends are on a fixed list of SGLang's own backends, and ``neuron`` is not on it. For such a launch
whose two backends are both ``neuron``, the module swaps ``attention_backends_of`` (the function the adjustment reads
the backends through) for one that answers a listed backend, for that one call only. Any other launch runs the
adjustment unchanged. The server arguments are never modified.

The wrapped target is md5-pinned against upstream SGLang 43881a7.

Upstream: SGLang's GPT-OSS and Gemma 4 checks could accept the attention backend an out-of-tree platform registers; this module
then goes away.
"""

from __future__ import annotations

import contextlib
import importlib

TARGET = "sglang.srt.arg_groups.model_hook.handle_model_specific_adjustments"
_TARGET_MD5 = "a735fb8d579c5fb17e107828e2546b4b"
# the architectures whose adjustment asserts a listed backend (both lists contain triton)
_CHECKED = ("GptOssForCausalLM", "Gemma4ForConditionalGeneration")
# a listed backend; on such a launch the adjustment reads the backends for this check only
_LISTED = ("triton", "triton")


@contextlib.contextmanager
def _listed_backends(model_hook):
    original = model_hook.attention_backends_of
    model_hook.attention_backends_of = lambda cfg: _LISTED
    try:
        yield
    finally:
        model_hook.attention_backends_of = original


def handle_model_specific_adjustments(original_fn, server_args, *args, **kwargs):
    from sglang.srt.arg_groups import model_hook

    from sglang_neuron.general_plugin import ATTN_BACKEND_NAME
    from sglang_neuron.models import catalog

    # other models' adjustments read the backends for checks of their own, which a stand-in backend would mislead
    if not any(arch in catalog.architectures(server_args) for arch in _CHECKED):
        return original_fn(server_args, *args, **kwargs)
    try:
        pair = tuple(model_hook.attention_backends_of(model_hook.resolved_view(server_args)))
    except Exception:  # noqa: BLE001 - not resolved yet: the adjustment runs unchanged and reports it
        pair = ()
    if pair == (ATTN_BACKEND_NAME, ATTN_BACKEND_NAME):
        with _listed_backends(model_hook):
            return original_fn(server_args, *args, **kwargs)
    return original_fn(server_args, *args, **kwargs)


def register(registry, around) -> None:
    from sglang_neuron.sglang_patches import source_guard

    module_name, _, name = TARGET.rpartition(".")
    source_guard.check(getattr(importlib.import_module(module_name), name), _TARGET_MD5, TARGET, __name__)
    registry.register(TARGET, handle_model_specific_adjustments, around)
