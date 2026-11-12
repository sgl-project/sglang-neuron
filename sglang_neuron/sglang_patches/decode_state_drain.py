# SPDX-License-Identifier: Apache-2.0
"""A PD decode's request table drains its recurrent-state pool before its state slots change owner.

sglang_patches/state_pool_class drains a model's state pool before ``StateStorage.clear``, the reset of
``HybridReqToTokenPool.clear``. The PD decode's request table, ``HybridMambaDecodeReqToTokenPool``, overrides ``clear``
and resets its slot allocator directly (``self.state.slots.clear()``), past that drain. A Neuron state pool
(mem_cache/state_pool) may still hold a copy or a publication its kernel submitted, and a slot handed to a new request
while it lands would carry the old request's state. So ``HybridMambaDecodeReqToTokenPool.clear`` (wrapped) drains the
pool first; a drain that raises stops the reset. Registered on every decode role.

Upstream: the decode table's clear would reset through ``StateStorage.clear``, which the pool's drain already wraps.

Target md5-pinned against upstream SGLang 43881a7: ``HybridMambaDecodeReqToTokenPool.clear`` (wrapped)."""

from __future__ import annotations

TARGET = ("sglang.srt.disaggregation.decode", "HybridMambaDecodeReqToTokenPool.clear")
_TARGETS_MD5 = {
    TARGET: "cb4e4e16bd8d9db88cd2fbe589a5b25c",
}


def clear(original_fn, self, *args, **kwargs):
    """AROUND HybridMambaDecodeReqToTokenPool.clear: drain the state pool's pending operations first."""
    drain = getattr(getattr(self, "mamba_pool", None), "drain", None)
    if drain is not None:
        drain()
    return original_fn(self, *args, **kwargs)


def register(registry, around) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
    registry.register(".".join(TARGET), clear, around)
