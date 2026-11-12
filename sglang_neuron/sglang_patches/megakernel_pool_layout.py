# SPDX-License-Identifier: Apache-2.0
"""SGLang's KV cache configurator builds a megakernel model's pools in the physical layout its built object declares.

A built megakernel may declare a ``pool_layout`` (megakernel_platform.py hands it to the platform when it is built). The
configurator of that model (the target's or a draft's, each from its own model's built object, never a process-global
active layout) then takes its pool classes from the platform:

- KV pools: ``_resolve_kv_pool_class`` asks ``platform.get_megakernel_kv_pool_cls(kind=..., pool_layout=...)`` instead
  of ``get_kv_pool_cls``, for the attention kinds the model instantiates only: MHA without an MLA backend, MLA with an
  MLA backend unless the model is DSA (a hybrid-SWA DSA model still needs MLA for its SWA component), DSA with an MLA
  backend on a DSA model. A layout need not supply a kind its model never builds; the configurator keeps that kind's
  in-tree default, which it does not construct. A used kind the platform answers None for is refused, and the class
  must subclass the kind's in-tree pool as upstream checks.
- Recurrent state pools: the layout's ``state_pool_class`` reaches the request tables through the one state pool seam
  every model shares (sglang_patches/state_pool_class: a megakernel layout's class, else
  ``NeuronModel.state_pool_class``), which also drains the pool before a ``StateStorage`` reset re-owns its slots.

A model without a declared layout keeps every upstream path. Targets are md5-pinned against upstream SGLang 43881a7.

Upstream: part of the platform megakernel seams (sgl-project/sglang#46355 adaptation); upstream candidates."""

from __future__ import annotations

import contextvars
from typing import Any, Optional

_CONFIGURATOR = "sglang.srt.mem_cache.kv_cache_configurator"
_TARGETS_MD5 = {
    (_CONFIGURATOR, "KVCacheConfigurator._resolve_kv_pool_class"): "6385977604aedaf3221b7cc20017ba8f",
    # calls _resolve_kv_pool_class for mha, mla and dsa with is_dsa_model as its argument
    (_CONFIGURATOR, "KVCacheConfigurator._build_token_to_kv_pool"): "e1cb83a0b76fe827a4a1cb1c2ad77193",
}
# the attention pool kinds the configurator's model builds, set while _build_token_to_kv_pool runs
_USED_KINDS: contextvars.ContextVar[Optional[frozenset]] = contextvars.ContextVar("megakernel_used_kinds",
                                                                                 default=None)


def pool_layout(model) -> Any:
    """The pool layout ``model``'s built megakernel declares, or None."""
    built = None if model is None else model.__dict__.get("_megakernel")
    return getattr(built, "pool_layout", None)


def used_kinds(*, use_mla_backend: bool, is_dsa_model: bool, is_hybrid_swa: bool) -> frozenset:
    """The attention pool kinds a model instantiates."""
    if not use_mla_backend:
        return frozenset({"mha"})
    kinds = {"dsa"} if is_dsa_model else set()
    if not is_dsa_model or is_hybrid_swa:
        kinds.add("mla")
    return frozenset(kinds)


def _platform():
    from sglang.srt.platforms import current_platform

    return current_platform


def build_token_to_kv_pool(original_fn, self, *args, **kwargs):
    """AROUND KVCacheConfigurator._build_token_to_kv_pool: records the kinds the model builds."""
    if pool_layout(self.model) is None:
        return original_fn(self, *args, **kwargs)
    token = _USED_KINDS.set(used_kinds(use_mla_backend=self.use_mla_backend,
                                       is_dsa_model=bool(kwargs["is_dsa_model"]),
                                       is_hybrid_swa=self.is_hybrid_swa))
    try:
        return original_fn(self, *args, **kwargs)
    finally:
        _USED_KINDS.reset(token)


def resolve_kv_pool_class(original_fn, self, *, kind, default):
    """AROUND KVCacheConfigurator._resolve_kv_pool_class."""
    layout = pool_layout(self.model)
    if layout is None:
        return original_fn(self, kind=kind, default=default)
    kinds = _USED_KINDS.get()
    if kinds is not None and kind not in kinds:
        # a kind the model never constructs: the layout need not supply it
        return default
    platform = _platform()
    resolve = getattr(platform, "get_megakernel_kv_pool_cls", None)
    pool_cls = None if resolve is None else resolve(kind=kind, pool_layout=layout)
    if pool_cls is None:
        raise TypeError(f"{type(platform).__name__} supplies no {kind} pool class for the megakernel pool layout "
                        f"{layout!r}; a declared layout requires an explicit class for every kind its model builds")
    from sglang.srt.mem_cache.kv_cache_configurator import _KV_POOL_BASE_FOR_KIND

    base = _KV_POOL_BASE_FOR_KIND[kind]
    if not (isinstance(pool_cls, type) and issubclass(pool_cls, base)):
        raise TypeError(f"{type(platform).__name__}.get_megakernel_kv_pool_cls(kind={kind!r}) returned {pool_cls!r}; "
                        f"expected a subclass of {base.__name__}")
    return pool_cls


def register(registry, around) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
    registry.register(f"{_CONFIGURATOR}.KVCacheConfigurator._build_token_to_kv_pool", build_token_to_kv_pool, around)
    registry.register(f"{_CONFIGURATOR}.KVCacheConfigurator._resolve_kv_pool_class", resolve_kv_pool_class, around)
