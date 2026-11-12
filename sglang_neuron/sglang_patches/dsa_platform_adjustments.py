# SPDX-License-Identifier: Apache-2.0
"""Let SGLang's DSA (sparse-attention) model adjustments run on the Neuron platform.

For a DSA model, ``model_hook.handle_model_specific_adjustments`` treats every platform that is neither NPU nor XPU as
a GPU: it reads the CUDA device capability, which raises on Neuron, and runs two passes that pick a GPU KV dtype and
GPU DSA attention backends. On Neuron, as on NPU, the platform owns a DSA model's KV dtype and attention, so both passes
declare nothing and the unused capability read returns (0, 0), which no GPU branch accepts.

A model whose catalog entry owns its DSA attention (``ModelSpec.owns_dsa_attention``) also keeps SGLang's DSA override
provider from replacing its attention backend and page with the GPU ones (``DROPPED``); everything else the provider
declares, and every other model and platform, is unchanged. A provider a catalog entry skips
(``ModelSpec.skipped_override_providers``) declares nothing for that architecture on Neuron and is not run there. The
server arguments are never modified here."""

from __future__ import annotations

import contextlib
import functools

ADJUST = "sglang.srt.arg_groups.model_hook.handle_model_specific_adjustments"
KV_DTYPE_PASS = "sglang.srt.arg_groups.overrides._dsa_kv_cache_dtype_default"
BACKEND_PASS = "sglang.srt.arg_groups.overrides._dsa_split_backend_resolution"
# the wrapped passes, as (module, qualified name), pinned in upstream_pins.json
_TARGETS = tuple(tuple(name.rsplit(".", 1)) for name in (ADJUST, KV_DTYPE_PASS, BACKEND_PASS))
# a capability below every GPU branch's threshold; the adjustment itself discards it
NO_GPU_CAPABILITY = (0, 0)

_PROVIDER_MODULE = "sglang.srt.arg_groups.model_overrides.deepseek_v2"
_BASE = "sglang.srt.arg_groups.model_override_base"
_OVERRIDE_TARGETS = (
    (_PROVIDER_MODULE, "_deepseek_family_overrides"),
    # the providers' registry, appended to at import
    (_BASE, "register_model_override"),
    # calls each registered provider from the registry, recording it under its __qualname__
    ("sglang.srt.arg_groups.overrides", "collect_model_override_declarations"),
)
# what the family provider declares that a model owning its DSA attention chooses itself on Neuron
DROPPED = ("attention_backend", "page_size", "dsa_prefill_backend", "dsa_decode_backend")
_MARK = "_sglang_neuron_dsa_dropped"


def _skipped_targets() -> tuple:
    from sglang_neuron.models import catalog

    return tuple(sorted({tuple(p) for spec in catalog.MODELS.values() for p in spec.skipped_override_providers}))


# the providers the catalog entries skip (ModelSpec.skipped_override_providers), pinned in upstream_pins.json
_SKIPPED_TARGETS = _skipped_targets()


@contextlib.contextmanager
def _no_cuda_capability():
    """Within the block, torch.cuda.get_device_capability returns NO_GPU_CAPABILITY instead of probing a driver."""
    import torch

    original = torch.cuda.get_device_capability
    torch.cuda.get_device_capability = lambda *args, **kwargs: NO_GPU_CAPABILITY
    try:
        yield
    finally:
        torch.cuda.get_device_capability = original


def handle_model_specific_adjustments(original_fn, server_args, *args, **kwargs):
    with _no_cuda_capability():
        return original_fn(server_args, *args, **kwargs)


def dsa_pass_on_neuron(original_fn, view, *args, **kwargs):
    """A DSA override pass on the Neuron platform: nothing declared (the platform owns DSA's KV dtype and attention)."""
    return {}


def _platform():
    """SGLang's current platform (the tests' seam)."""
    from sglang.srt.platforms import current_platform

    return current_platform


def dropping(provider, dropped=DROPPED):
    """``provider``, declaring nothing of ``dropped`` on the Neuron platform; ``dropped`` None: nothing at all, without
    running the provider (one that asserts on the platform's own attention backend)."""

    @functools.wraps(provider)
    def wrapper(server_args, hf_config):
        on_neuron = getattr(_platform(), "device_name", None) == "neuron"
        if on_neuron and dropped is None:
            return {}
        declared = provider(server_args, hf_config)
        if on_neuron:
            return {key: value for key, value in declared.items() if key not in dropped}
        return declared

    setattr(wrapper, _MARK, True)
    return wrapper


def owning_architectures() -> list:
    """The catalog's architectures whose model chooses its own DSA attention backend and page."""
    from sglang_neuron.models import catalog

    return sorted(arch for arch, spec in catalog.MODELS.items() if spec.owns_dsa_attention)


def drop_overrides(architectures, module: str, name: str, dropped=DROPPED) -> None:
    """Wrap upstream's provider ``module.name`` in each architecture's registry entry with dropping(); idempotent.

    The registry (``model_override_base._MODEL_OVERRIDE_FNS``) holds each provider as it was registered at import, and
    ``overrides.collect_model_override_declarations`` calls it from there, so the entry is replaced (patching the module
    attribute would reach nothing); the wrapper keeps the provider's name, which the declarations are recorded under."""
    import importlib

    provider = getattr(importlib.import_module(module), name)
    registry = importlib.import_module(_BASE)._MODEL_OVERRIDE_FNS
    for arch in architectures:
        providers = registry.get(arch, [])
        if any(getattr(fn, _MARK, False) and getattr(fn, "__wrapped__", None) is provider for fn in providers):
            continue
        at = [i for i, fn in enumerate(providers) if fn is provider]
        if len(at) != 1:
            raise RuntimeError(f"{__name__}: SGLang's override registry holds {module}.{name} {len(at)} times for "
                               f"{arch}, not once; re-validate {__name__} against this SGLang")
        providers[at[0]] = dropping(provider, dropped)


def drop_family_overrides(architectures) -> None:
    """Wrap the DeepSeek-family provider in each architecture's registry entry with dropping(); idempotent."""
    drop_overrides(architectures, _PROVIDER_MODULE, "_deepseek_family_overrides")


def skip_providers() -> None:
    """Make each provider a catalog entry skips (ModelSpec.skipped_override_providers) declare nothing for its
    architecture on Neuron; idempotent."""
    from sglang_neuron.models import catalog

    for arch, spec in sorted(catalog.MODELS.items()):
        for module, name in spec.skipped_override_providers:
            drop_overrides([arch], module, name, dropped=None)


def register(registry, around) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_pinned(_TARGETS + _OVERRIDE_TARGETS + _SKIPPED_TARGETS, __name__)
    registry.register(ADJUST, handle_model_specific_adjustments, around)
    registry.register(KV_DTYPE_PASS, dsa_pass_on_neuron, around)
    registry.register(BACKEND_PASS, dsa_pass_on_neuron, around)
    drop_family_overrides(owning_architectures())
    skip_providers()
