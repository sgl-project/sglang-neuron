# SPDX-License-Identifier: Apache-2.0
"""The platform seams of upstream SGLang's megakernel binding, so an out-of-tree platform serves a MegakernelSpec.

Upstream SGLang 43881a7 ships the megakernel contract (``sglang.kernels.plugin_kernels.nvidia.spec``: MegakernelSpec,
BuiltMegakernel, the lazy per-architecture registry) and its binding (``sglang.srt.kernel_plugins.nvidia.binding``)
behind ``--enable-nvidia-megakernel``, but the binding is CUDA-only: it matches specs on a ``torch.cuda`` ISA probe,
requires the sgl-nvidia-mega entry point, reads ``torch.cuda.mem_get_info`` and the SM count, and always replaces
``model.forward`` with its CUDA-graph dispatch. This module makes the binding platform-dispatched, as the upstream RFC
(sgl-project/sglang#46355) does in tree, without editing the served tree:

- ``device_isa`` answers the platform's ``get_megakernel_target()`` when the platform has one (the Neuron platform
  answers its compiler target, a static string, so the tokenizer process may ask too), else upstream's CUDA probe.
- ``plugin_reasons`` is empty on an out-of-tree platform: its own plugin registers its specs, not sgl-nvidia-mega.
- ``install_megakernel`` (rebuilt from upstream's source) reads free device memory only on a CUDA-alike platform,
  hands a built object's optional ``pool_layout`` to the platform's ``activate_megakernel_pool_layout`` right after the
  build (before KV sizing; a platform without it, or one that raises NotImplementedError, refuses the launch), and lets
  the platform's ``install_megakernel_dispatch`` own the dispatch: a callable it returns is the whole-step eager dispatch
  (kept in ``model.__dict__["_megakernel_eager_dispatch"]``, read by ``eager_dispatch``; megakernel_eager_dispatch.py
  routes EagerRunner through it), True means the platform's own runner dispatches; either way ``model.forward`` and the
  CUDA-graph capture check are left alone. False (or no such method) keeps upstream's ``model.forward`` wrapper.
- ``_model_disable_reasons`` (rebuilt) checks the CUDA SM count only for a spec that states ``resident_blocks``, and
  refuses such a spec on a platform that is not CUDA-alike.
- The registry takes an optional device target (``register(..., isa=...)``): an exact (architecture, target)
  registration wins over the architecture's default, so a Neuron spec and a CUDA spec of one architecture coexist
  without either importing the other's SDK. ``find_spec``, ``registered_architectures`` and ``spec_registered`` are
  upstream's, rebuilt to read both registries.

``PlatformMegakernelSpec`` is upstream's MegakernelSpec with the optional fields the other seams read (each None keeps
SGLang's own path): ``request_disable_reasons`` (megakernel_request_contract.py) and ``mlp_sync_cohort_metadata``
(mlp_sync_cohort_snapshot.py). Seams read them with getattr, so a plain MegakernelSpec works.

The switch stays upstream's ``--enable-nvidia-megakernel``: the RFC's platform-neutral ``--enable-megakernel`` is a new
ExecKernel field, which a runtime patch cannot add to the msgspec struct. Every rank decides with upstream's own rank
agreement (``_find_spec_on_every_rank``, ``megakernel_agree``); the seams add no per-rank state. Targets are md5-pinned
against upstream SGLang 43881a7. docs/design/megakernel.md (Seams) states the interface the megakernel host builds on.

Upstream: the in-tree form is sgl-project/sglang#46355 (platform target, platform-owned dispatch, pool layout); the
target registry and the eager dispatch callable are SGLang-Neuron's additions to it, upstream candidates."""

from __future__ import annotations

import functools
import importlib
from typing import Any, Callable, Optional

from sglang.kernels.plugin_kernels.nvidia.spec import MegakernelSpec

_BINDING = "sglang.srt.kernel_plugins.nvidia.binding"
_SPEC = "sglang.kernels.plugin_kernels.nvidia.spec"
# (module, qualified name): md5 of the source
_TARGETS_MD5 = {
    (_BINDING, "device_isa"): "3e1e81e06fb8c906338387b4c5a3ed78",
    (_BINDING, "plugin_reasons"): "293c2f9cb27a27e48c98952e1fe554bf",
    (_BINDING, "install_megakernel"): "6eeaf0893bfd302ab39a3cba83d88dc2",
    (_BINDING, "_model_disable_reasons"): "c419e273b24bb32d863584ce9afbde68",
    (_SPEC, "register"): "fc33df6c9942a244abef2c84b66008b9",
    (_SPEC, "registered_architectures"): "a9c7af0a096ae66eed86746b3dee5833",
    (_SPEC, "spec_registered"): "d5ee837bb5691ecd1fea7e255fcb5edb",
    (_SPEC, "find_spec"): "12082ccccb964ad9aec9451c2bbfe4c7",
}
# where install_megakernel keeps a platform's whole-step eager dispatch (not setattr: a dispatch that is an nn.Module
# must not join the model's state dict, as upstream keeps the built object)
EAGER_DISPATCH_KEY = "_megakernel_eager_dispatch"
_IMPORT = "    from sglang_neuron.sglang_patches import megakernel_platform as _mk\n\n"
INSTALL_SUBSTITUTIONS = (
    # free device memory where torch.cuda can tell
    ("    start, free_before = time.perf_counter(), torch.cuda.mem_get_info()[0]\n",
     _IMPORT +
     "    start, free_before = time.perf_counter(), _mk.device_free_bytes()\n"),
    # the built object's physical pool layout goes to the platform before KV sizing
    ('    model.__dict__["_megakernel"] = built\n',
     '    model.__dict__["_megakernel"] = built\n'
     "    _mk.activate_pool_layout(spec=spec, built=built, refuse=_refuse)\n"),
    ("    used_mib = (free_before - torch.cuda.mem_get_info()[0]) / 2**20\n",
     "    used_mib = (free_before - _mk.device_free_bytes()) / 2**20\n"),
    # a platform that owns the dispatch keeps model.forward and the capture check
    ("    dispatch = _Dispatch(built=built, original=model.forward, binding=binding)\n",
     "    _mk.agree_mlp_sync_cohort(spec=spec, built=built, agree=megakernel_agree)\n"
     "    if _mk.install_platform_dispatch(model=model, built=built, binding=binding):\n"
     "        return\n"
     "    dispatch = _Dispatch(built=built, original=model.forward, binding=binding)\n"),
)
DISABLE_REASONS_SUBSTITUTIONS = (
    ("    sms = torch.cuda.get_device_properties(\n",
     "    from sglang_neuron.sglang_patches import megakernel_platform as _mk\n\n"
     "    if not spec.requires.resident_blocks:\n"
     "        return reasons\n"
     "    if not _mk.is_cuda_alike():\n"
     "        reasons.append(\n"
     '            f"resident_blocks {spec.requires.resident_blocks} is a CUDA SM count; this platform checks "\n'
     '            "residency in the spec\'s model_disable_reasons"\n'
     "        )\n"
     "        return reasons\n"
     "    sms = torch.cuda.get_device_properties(\n"),
)
_ISA_IMPORT = "    from sglang_neuron.sglang_patches.megakernel_platform import ISA_REGISTRY\n\n"
FIND_SPEC_SUBSTITUTIONS = (
    ("        path = _REGISTRY.get(arch)\n",
     "        # an exact (architecture, target) registration wins over the architecture's default\n"
     "        path = ISA_REGISTRY.get((arch, isa), _REGISTRY.get(arch))\n"),
    ('    why = []\n', _ISA_IMPORT + '    why = []\n'),
)
REGISTERED_SUBSTITUTIONS = (
    ("    return sorted(_REGISTRY)\n",
     _ISA_IMPORT +
     "    return sorted(set(_REGISTRY) | {arch for arch, _ in ISA_REGISTRY})\n"),
)
SPEC_REGISTERED_SUBSTITUTIONS = (
    ("    return any(arch in _REGISTRY for arch in architectures or ())\n",
     "    registered = set(registered_architectures())\n"
     "    return any(arch in registered for arch in architectures or ())\n"),
)

# (HF architecture, device target) -> the "module:attribute" path of its spec
ISA_REGISTRY: dict[tuple[str, str], str] = {}


class PlatformMegakernelSpec(MegakernelSpec, frozen=True):
    """Upstream's MegakernelSpec with the optional fields of the platform seams; None keeps SGLang's own path.

    request_disable_reasons: (tokenized request) -> list of reasons, host-only checks in the frontend before the request
    reaches a scheduler (megakernel_request_contract.py).
    mlp_sync_cohort_metadata: the megakernel reads the DP-attention cohort snapshot
    (ForwardBatch.mlp_sync_batch_metadata, mlp_sync_cohort_snapshot.py); a built object or its pool_layout may also
    declare it. Off, the MLP sync exchange is upstream's."""

    request_disable_reasons: Optional[Callable[[Any], list]] = None
    mlp_sync_cohort_metadata: bool = False


def _platform():
    from sglang.srt.platforms import current_platform

    return current_platform


def is_cuda_alike() -> bool:
    return bool(_platform().is_cuda_alike())


def device_free_bytes() -> int:
    """Free device memory for install_megakernel's build log; 0 where torch.cuda cannot tell."""
    if not is_cuda_alike():
        return 0
    import torch

    return torch.cuda.mem_get_info()[0]


def activate_pool_layout(*, spec, built, refuse) -> None:
    """Hand ``built.pool_layout`` (when not None) to the platform before the KV cache is sized; ``refuse`` (the
    binding's _refuse) raises when the platform cannot build pools in it."""
    layout = getattr(built, "pool_layout", None)
    if layout is None:
        return
    activate = getattr(_platform(), "activate_megakernel_pool_layout", None)
    if activate is None:
        refuse([f"{spec.name} declares a pool_layout and {type(_platform()).__name__} cannot build pools in one"])
    try:
        activate(layout)
    except NotImplementedError as e:
        refuse([f"{spec.name} declares a pool_layout: {e}"])


def needs_mlp_sync_cohort(*, spec, built) -> bool:
    """Whether the spec, its built object or the built object's pool layout reads the MLP sync cohort snapshot."""
    return any(bool(getattr(owner, "mlp_sync_cohort_metadata", False))
               for owner in (spec, built, getattr(built, "pool_layout", None)))


def agree_mlp_sync_cohort(*, spec, built, agree) -> None:
    """Turn the MLP sync cohort snapshot on when the megakernel reads it. Every rank of the exchange must gather the
    same columns, so the ranks agree on the decision first (``agree``: the binding's megakernel_agree over the TP
    group, which spans the attention-DP participants); a rank never turns it off once on (a draft without the need
    leaves the target's decision)."""
    needs = needs_mlp_sync_cohort(spec=spec, built=built)
    agree(needs, what=f"whether {spec.name} reads the MLP sync cohort snapshot")
    if needs:
        from sglang_neuron.sglang_patches import mlp_sync_cohort_snapshot

        mlp_sync_cohort_snapshot.enable(spec.name)


def install_platform_dispatch(*, model, built, binding) -> bool:
    """The platform's install_megakernel_dispatch: True when the platform owns the dispatch (a callable it returns is
    the whole-step eager dispatch, kept on the model for EagerRunner), False to keep upstream's model.forward wrapper."""
    install = getattr(_platform(), "install_megakernel_dispatch", None)
    if install is None:
        return False
    dispatch = install(model=model, built=built, binding=binding)
    if callable(dispatch):
        model.__dict__[EAGER_DISPATCH_KEY] = dispatch
        return True
    return bool(dispatch)


def eager_dispatch(model) -> Optional[Callable]:
    """The platform's whole-step eager dispatch installed on ``model``, or None for SGLang's standard path."""
    return None if model is None else model.__dict__.get(EAGER_DISPATCH_KEY)


def device_isa(original_fn) -> Optional[str]:
    """AROUND binding.device_isa: the platform's megakernel target, else upstream's CUDA probe."""
    target = getattr(_platform(), "get_megakernel_target", None)
    return original_fn() if target is None else target()


def plugin_reasons(original_fn) -> list:
    """AROUND binding.plugin_reasons: an out-of-tree platform registers its specs from its own plugin."""
    if _platform().is_out_of_tree():
        return []
    return original_fn()


def register_spec(original_fn, *, architecture: str, path: str, isa: Optional[str] = None) -> None:
    """AROUND spec.register: with ``isa``, register ``path`` for that device target only; without, upstream's."""
    if isa is None:
        return original_fn(architecture=architecture, path=path)
    if not isinstance(isa, str) or not isa:
        raise ValueError("a megakernel spec's device target must be a nonempty string")
    key = (architecture, isa)
    if ISA_REGISTRY.get(key, path) != path:
        raise ValueError(f"a megakernel spec for {key} is already registered: {ISA_REGISTRY[key]}")
    ISA_REGISTRY[key] = path


def _function(module: str, qualname: str):
    return functools.reduce(getattr, qualname.split("."), importlib.import_module(module))


def _rebuilt(module: str, qualname: str, substitutions):
    from sglang_neuron.sglang_patches import source_guard

    return source_guard.with_substitutions(_function(module, qualname), substitutions)


def install_megakernel():
    return _rebuilt(_BINDING, "install_megakernel", INSTALL_SUBSTITUTIONS)


def model_disable_reasons():
    return _rebuilt(_BINDING, "_model_disable_reasons", DISABLE_REASONS_SUBSTITUTIONS)


def find_spec():
    return _rebuilt(_SPEC, "find_spec", FIND_SPEC_SUBSTITUTIONS)


def registered_architectures():
    return _rebuilt(_SPEC, "registered_architectures", REGISTERED_SUBSTITUTIONS)


def spec_registered():
    return _rebuilt(_SPEC, "spec_registered", SPEC_REGISTERED_SUBSTITUTIONS)


def register(registry, around, replace) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
    registry.register(f"{_BINDING}.device_isa", device_isa, around)
    registry.register(f"{_BINDING}.plugin_reasons", plugin_reasons, around)
    registry.register(f"{_BINDING}.install_megakernel", install_megakernel(), replace)
    registry.register(f"{_BINDING}._model_disable_reasons", model_disable_reasons(), replace)
    registry.register(f"{_SPEC}.register", register_spec, around)
    registry.register(f"{_SPEC}.registered_architectures", registered_architectures(), replace)
    registry.register(f"{_SPEC}.spec_registered", spec_registered(), replace)
    registry.register(f"{_SPEC}.find_spec", find_spec(), replace)
