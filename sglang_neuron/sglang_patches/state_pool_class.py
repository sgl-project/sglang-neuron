# SPDX-License-Identifier: Apache-2.0
"""SGLang's request tables build a model's recurrent-state pool from the class the model declares.

A model with recurrent-state layers (KDA, GDN, Mamba-2) declares its pool class (a ``MambaPool`` subclass, normally
mem_cache/state_pool.NeuronStatePool bound to the model's spec and kernels) in one of two places:

- a megakernel model: its built object's ``pool_layout.state_pool_class`` (megakernel/pools.PoolLayout), which the
  platform answers through ``get_megakernel_state_pool_cls(pool_layout=...)``
  (sglang_patches/megakernel_pool_layout.pool_layout reads the layout);
- a platform-runtime model: ``NeuronModel.state_pool_class`` on the model object SGLang built.

While SGLang's KV cache configurator builds that model's pools (``_init_pools``), every request table with Mamba-style
state it builds (``HybridReqToTokenPool``, the PD decode's ``HybridMambaDecodeReqToTokenPool``, a draft's
``clone_with_new_mamba``) builds its state pool with the declared class (``_init_mamba_pool``). Before SGLang's state
storage hands its slots to new owners (``StateStorage.clear``), the pool's pending slot operations finish
(``drain``), so a queued state write cannot land after a slot changed owner. A model that declares no class keeps
SGLang's ``MambaPool`` on every path; one that declares two different classes is refused.

The class is keyed by the configurator's own model, never a process-global setting, so a draft model's configurator
builds its own tables. Targets are md5-pinned against upstream SGLang 43881a7.

Upstream: generic; upstream's request tables take a class attribute (``mamba_pool_cls``) this sets per model."""

from __future__ import annotations

import contextvars
from typing import Optional

_CONFIGURATOR = "sglang.srt.mem_cache.kv_cache_configurator"
_POOL = "sglang.srt.mem_cache.memory_pool"
_STATE = "sglang.srt.mem_cache.kv.state_storage"
# (module, qualified name): md5 of the source
_TARGETS_MD5 = {
    # builds every request-to-token pool (target, PD decode, draft clone) of the configurator's model
    (_CONFIGURATOR, "KVCacheConfigurator._init_pools"): "f6ad670db321a275d67d74467a74c3ef",
    # builds self.mamba_pool from self.mamba_pool_cls
    (_POOL, "HybridReqToTokenPool._init_mamba_pool"): "007258087fe596033454195e7c683c32",
    (_STATE, "StateStorage.clear"): "4ab627245a0a9c9b4994aceceb22874e",
}
# the state pool class of the model whose pools _init_pools builds
_STATE_POOL_CLS: contextvars.ContextVar[Optional[type]] = contextvars.ContextVar("neuron_state_pool_cls", default=None)


def _platform():
    from sglang.srt.platforms import current_platform

    return current_platform


def _checked(cls, where: str) -> Optional[type]:
    if cls is None:
        return None
    from sglang.srt.mem_cache.memory_pool import MambaPool

    if not isinstance(cls, type) or not issubclass(cls, MambaPool):
        raise TypeError(f"{where} is {cls!r}: a recurrent-state pool class subclasses MambaPool "
                        "(mem_cache/state_pool.NeuronStatePool.bind(spec, ops))")
    return cls


def layout_state_pool_cls(model) -> Optional[type]:
    """The platform's state pool class for ``model``'s built megakernel layout, or None (no layout, or a layout whose
    platform answers none)."""
    from sglang_neuron.sglang_patches.megakernel_pool_layout import pool_layout

    layout = pool_layout(model)
    if layout is None:
        return None
    resolve = getattr(_platform(), "get_megakernel_state_pool_cls", None)
    return _checked(None if resolve is None else resolve(pool_layout=layout),
                    "the megakernel layout's state pool class")


def state_pool_cls(model) -> Optional[type]:
    """The state pool class ``model`` declares (its megakernel layout's, else ``model.state_pool_class``), or None for
    the request table's own; anything but a MambaPool subclass, or two different declarations, is refused."""
    if model is None:
        return None
    from_layout = layout_state_pool_cls(model)
    declared = _checked(getattr(model, "state_pool_class", None), f"{type(model).__name__}.state_pool_class")
    if from_layout is not None and declared is not None and from_layout is not declared:
        raise TypeError(f"{type(model).__name__} declares two state pool classes: {from_layout.__name__} (its "
                        f"megakernel pool layout) and {declared.__name__} (state_pool_class); declare one")
    return from_layout if from_layout is not None else declared


def init_pools(original_fn, self, *args, **kwargs):
    """AROUND KVCacheConfigurator._init_pools: the model's state pool class for the request tables it builds."""
    token = _STATE_POOL_CLS.set(state_pool_cls(getattr(self, "model", None)))
    try:
        pools = original_fn(self, *args, **kwargs)
    finally:
        _STATE_POOL_CLS.reset(token)
    # a radix checkpoint store built with these pools takes its size from the device memory they left
    # (mem_cache/state_checkpoint_pool.size_deferred_stores)
    from sglang_neuron.mem_cache import state_checkpoint_pool

    state_checkpoint_pool.size_deferred_stores()
    return pools


def init_mamba_pool(original_fn, self, *args, **kwargs):
    """AROUND HybridReqToTokenPool._init_mamba_pool. The original builds the state pool and then, under
    ``--enable-int8-mamba-checkpoint``, the radix checkpoint store (mem_cache/state_checkpoint_pool for a Neuron state
    pool); a store that binds to its request pool (``bind_active``: spec agreement, its copies warmed) is bound here,
    the first point both exist."""
    cls = _STATE_POOL_CLS.get()
    if cls is not None:
        self.mamba_pool_cls = cls
    result = original_fn(self, *args, **kwargs)
    bind = getattr(getattr(self, "mamba_ckpt_pool", None), "bind_active", None)
    if callable(bind):
        bind(self.mamba_pool)
    return result


def clear_state(original_fn, self, *args, **kwargs):
    """AROUND StateStorage.clear: drain the state pool's pending operations before its slots change owner."""
    drain = getattr(getattr(self, "pool", None), "drain", None)
    if drain is not None:
        drain()
    return original_fn(self, *args, **kwargs)


def register(registry, around) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
    registry.register(f"{_CONFIGURATOR}.KVCacheConfigurator._init_pools", init_pools, around)
    registry.register(f"{_POOL}.HybridReqToTokenPool._init_mamba_pool", init_mamba_pool, around)
    registry.register(f"{_STATE}.StateStorage.clear", clear_state, around)
    # the radix checkpoint store a Neuron state pool gets in place of upstream's int8 one
    from sglang_neuron.mem_cache import state_checkpoint_pool

    state_checkpoint_pool.register_hooks(registry, around)
