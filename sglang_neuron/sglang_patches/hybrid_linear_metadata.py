# SPDX-License-Identifier: Apache-2.0
"""A hybrid linear-attention model on the Neuron platform gets no host-side linear-attention metadata.

For a model with linear-attention layers (KDA, GDN) SGLang wraps the attention backend in
``HybridLinearAttnBackend``: the platform's backend for the full-attention layers and SGLang's own linear-attention
backend (``KDAAttnBackend``) for the others, and the eager runner calls the wrapper's ``init_forward_metadata`` before
every forward, which builds both backends' metadata. A Neuron model's graph runs its linear attention itself from the
recurrent-state pool's tensors (mem_cache/state_pool.py), so the linear backend's metadata is never read; building it on the
host costs a step and reads attributes of SGLang's own ``MambaPool`` (the replay ring flags) that the platform's
state pool does not have. When the wrapper's full-attention backend is the platform's, only that backend's metadata
is built; any other wrapper runs upstream's code.

The target is md5-pinned against upstream SGLang 43881a7.

Upstream: the hybrid wrapper could skip a linear-attention backend whose layers the model runs in its own graph;
this module then goes away."""

from __future__ import annotations

import importlib

TARGET = "sglang.srt.layers.attention.hybrid_linear_attn_backend.HybridLinearAttnBackend.init_forward_metadata"
_TARGET_MD5 = "7b6674984aa5de769224f85a30e2e892"


def init_forward_metadata(original_fn, self, forward_batch, *args, **kwargs):
    from sglang_neuron.attention.backend import NeuronAttnBackend

    if isinstance(self.full_attn_backend, NeuronAttnBackend):
        return self.full_attn_backend.init_forward_metadata(forward_batch)
    return original_fn(self, forward_batch, *args, **kwargs)


def register(registry, around) -> None:
    from sglang_neuron.sglang_patches import source_guard

    module_name, cls, name = TARGET.rsplit(".", 2)
    target = getattr(getattr(importlib.import_module(module_name), cls), name)
    source_guard.check(target, _TARGET_MD5, TARGET, __name__)
    registry.register(TARGET, init_forward_metadata, around)
