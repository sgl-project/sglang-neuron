# SPDX-License-Identifier: Apache-2.0
"""The whole-step dispatch of a Neuron megakernel.

SGLang's binding installs a megakernel under ``--enable-nvidia-megakernel``: it finds the spec registered for the
model's architecture on the platform's device target, checks it against the resolved configuration, builds it after the
weights load (before the KV cache is sized) and asks the platform for the dispatch
(``NeuronSRTPlatform.install_megakernel_dispatch``, the platform seams of ``sglang_patches/megakernel_platform.py``).
The platform calls ``host.create_dispatch``, which lands here:

- ``attach`` refuses a built object that is not a ``NeuronBuiltMegakernel``, makes the ranks of the TP group agree on
  its digest (programs, widths, pool layout; SGLang's own agreement covers the widths and draft entries only) and
  returns the ``NeuronMegakernelDispatch``, which SGLang keeps on the model (``megakernel_platform.EAGER_DISPATCH_KEY``).
- ``prepare`` runs once the pools exist: SGLang's ``EagerRunner`` calls it, and so does the platform's graph runner
  (``NeuronModel._ensure_runtime``); the second call with the same runner is a no-op. The built object binds its pool
  layout and builds, compiles and warms its runtimes.
- Every forward runs through the dispatch: SGLang's ``EagerRunner.execute`` calls it for every mode, the platform's
  ``NeuronGraphRunner`` plans in ``load_batch`` and runs in ``execute``, and ``NeuronModel.forward`` calls it. ``plan``
  refuses a step no program covers with the reason; there is no other path.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

from .built import NeuronBuiltMegakernel

logger = logging.getLogger(__name__)

# SGLang's attributes on the model: the built object (kernel_plugins.nvidia.binding.built_megakernel) and the platform's
# whole-step dispatch (sglang_patches/megakernel_platform.EAGER_DISPATCH_KEY)
_BUILT = "_megakernel"
_DISPATCH = "_megakernel_eager_dispatch"


def dispatch_of(model) -> Optional["NeuronMegakernelDispatch"]:
    """The model's installed Neuron megakernel dispatch, or None (the model's own host serves it)."""
    dispatch = None if model is None else getattr(model, "__dict__", {}).get(_DISPATCH)
    return dispatch if isinstance(dispatch, NeuronMegakernelDispatch) else None


def built_megakernel(model):
    return None if model is None else getattr(model, "__dict__", {}).get(_BUILT)


def _gather_tp(value) -> list:
    """``value`` from every rank of the TP group (its CPU group); one rank's alone without a group."""
    try:
        from sglang.srt.distributed import parallel_state

        group = parallel_state.get_tp_group()
    except Exception:  # noqa: BLE001 -- no initialized group (a single-process harness)
        return [value]
    if group.world_size <= 1:
        return [value]
    import torch.distributed as dist

    out = [None] * group.world_size
    dist.all_gather_object(out, value, group=group.cpu_group)
    return out


def refuse(reasons: list, what: str = "the megakernel") -> None:
    if reasons:
        raise ValueError(f"{what} cannot serve this setup:\n  - " + "\n  - ".join(reasons))


def agree(value, *, what: str, gather: Callable = _gather_tp) -> None:
    values = gather(value)
    if any(v != value for v in values):
        raise RuntimeError(f"the ranks' {what} differ: " + "; ".join(f"rank {r}: {v}" for r, v in enumerate(values)))


def attach(model, built, *, binding=None, gather: Callable = _gather_tp) -> "NeuronMegakernelDispatch":
    """The dispatch of a built megakernel, after the ranks agree on the build."""
    name = getattr(built, "name", type(built).__name__)
    if not isinstance(built, NeuronBuiltMegakernel):
        refuse([f"its build returned {type(built).__name__}; a Neuron megakernel is a "
                "sglang_neuron.megakernel.NeuronBuiltMegakernel (plan / forward over the platform's runtimes)"],
               what=name)
    if built.model is not model:
        refuse(["it was built for another model instance"], what=name)
    agree(built.digest(), what=f"{name} builds (programs, widths, pool layout)", gather=gather)
    logger.info("[megakernel] %s", built.summary())
    return NeuronMegakernelDispatch(model=model, built=built, binding=binding)


def check_declared(model) -> None:
    """Refuse a model that declares a megakernel spec (``NeuronModel.megakernel_spec``) without its dispatch installed,
    or with a dispatch of another spec: the model built graphs only that spec serves."""
    declare = getattr(model, "megakernel_spec", None)
    spec = declare() if callable(declare) else None
    dispatch = dispatch_of(model)
    if spec is None:
        if dispatch is not None:
            raise RuntimeError(f"{type(model).__name__} declares no megakernel spec in this configuration, yet "
                               f"{dispatch.built.name} was installed on it")
        return
    if dispatch is None:
        from .upstream import FLAG

        raise ValueError(f"{type(model).__name__} serves this configuration through its megakernel spec {spec.name}, "
                         f"which SGLang installs under {FLAG}: launch with {FLAG}")
    if dispatch.built.name != spec.name:
        raise RuntimeError(f"{type(model).__name__} declares the megakernel spec {spec.name}, and SGLang installed "
                           f"{dispatch.built.name} from its registry")


class NeuronMegakernelDispatch:
    """Plans and runs a model's steps through its built megakernel (the module docstring). Callable with
    ``model.forward``'s signature, as SGLang's ``EagerRunner`` calls a platform dispatch."""

    def __init__(self, *, model, built: NeuronBuiltMegakernel, binding=None):
        self.model, self.built, self.binding = model, built, binding
        self.name = built.name
        self._runner = None

    @property
    def prepared(self) -> bool:
        return self._runner is not None

    def prepare(self, model_runner) -> dict:
        """Once the pools exist: the built object binds them and builds, compiles and warms its runtimes. A second call
        with the same runner (SGLang's EagerRunner and the platform's graph runner both prepare) returns the runtimes."""
        if self._runner is not None:
            if model_runner is not self._runner:
                raise RuntimeError(f"{self.name} is prepared for another model runner")
            return self.built.runtimes
        runtimes = self.built.prepare(model_runner)
        self._runner = model_runner
        return runtimes

    def plan(self, input_ids, forward_batch, *args, **kwargs):
        if self._runner is None:
            raise RuntimeError(f"{self.name}: a step arrived before the megakernel was prepared")
        step = self.built.plan(input_ids, forward_batch, *args, **kwargs)
        if step is None:
            mode = getattr(forward_batch, "forward_mode", None)
            raise RuntimeError(
                f"{self.name} covers no {mode} step of batch size {int(getattr(forward_batch, 'batch_size', 0))} "
                f"(it covers {sorted(self.built.covered_modes)}, widths {list(self.built.widths)}); a megakernel step "
                "has no other path")
        return step

    def run(self, step, input_ids, positions, forward_batch, *args, **kwargs):
        import msgspec

        step = msgspec.structs.replace(step, view=self.built.view(step, forward_batch))
        return self.built.forward(step, self.built.no_eager_path, input_ids, positions, forward_batch, *args,
                                  **kwargs)

    def __call__(self, input_ids, positions, forward_batch, *args, **kwargs):
        step = self.plan(input_ids, forward_batch, *args, **kwargs)
        return self.run(step, input_ids, positions, forward_batch, *args, **kwargs)
