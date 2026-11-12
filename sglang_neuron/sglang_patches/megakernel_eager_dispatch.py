# SPDX-License-Identifier: Apache-2.0
"""SGLang's EagerRunner runs a platform's whole-step megakernel dispatch, when one is installed, instead of the model's
per-mode eager forward.

A platform whose ``install_megakernel_dispatch`` returns a callable (megakernel_platform.py) owns every covered step of
the model: the callable takes ``model.forward``'s arguments and runs the whole step, with the device input packing,
context-parallel preparation and phase selection of its precompiled programs. Upstream's EagerRunner would instead
allocate and fill its CUDA-style shared input-copy bank and run the model's forward per mode. With the dispatch
installed on the runner's model:

- ``EagerRunner.__init__`` runs BaseRunner's init only, allocates no input-copy bank (``_eager_registry`` is None),
  runs no eager warmup, and calls the dispatch's optional ``prepare(model_runner)`` once (the pools exist by then).
- ``EagerRunner.load_batch`` returns a shallow copy of the live batch, as upstream does under
  SGLANG_EAGER_INPUT_NO_COPY.
- ``EagerRunner.execute`` calls the dispatch with the loaded batch's input ids and positions, the batch and the
  pipeline proxy kwargs, under upstream's device timer ("megakernel") and forward context, for every forward mode.

Without an installed dispatch every function is upstream's. Targets are md5-pinned against upstream SGLang 43881a7.

Upstream: part of the platform megakernel seams (sgl-project/sglang#46355 adaptation); an upstream candidate."""

from __future__ import annotations

from dataclasses import replace

_RUNNER = "sglang.srt.model_executor.runner.eager_runner"
_TARGETS_MD5 = {
    (_RUNNER, "EagerRunner.__init__"): "6f22867c32c165e57686419a5d085423",
    (_RUNNER, "EagerRunner.load_batch"): "c47b448c79da1e4b92fd52fd389fda13",
    (_RUNNER, "EagerRunner.execute"): "55340bfd7b60eb609ddac77386459798",
}


def _dispatch_of(model_runner):
    from sglang_neuron.sglang_patches.megakernel_platform import eager_dispatch

    return eager_dispatch(getattr(model_runner, "model", None))


def init(original_fn, self, model_runner, *args, **kwargs):
    """AROUND EagerRunner.__init__."""
    dispatch = _dispatch_of(model_runner)
    if dispatch is None:
        self._platform_dispatch = None
        return original_fn(self, model_runner, *args, **kwargs)
    from sglang.srt.model_executor.runner.base_runner import BaseRunner

    BaseRunner.__init__(self, model_runner)
    self._platform_dispatch = dispatch
    # the dispatch packs its own device inputs: no shared input-copy bank
    self._eager_registry = None
    prepare = getattr(dispatch, "prepare", None)
    if prepare is not None:
        prepare(model_runner)


def load_batch(original_fn, self, forward_batch, pp_proxy_tensors=None, **kwargs):
    """AROUND EagerRunner.load_batch."""
    if getattr(self, "_platform_dispatch", None) is None:
        return original_fn(self, forward_batch, pp_proxy_tensors, **kwargs)
    batch = replace(forward_batch)
    # dataclasses.replace copies the fields only: the MLP sync cohort snapshot, which the platform sets beside them
    # (sglang_patches/mlp_sync_cohort_snapshot), rides along, or a cohort planner would read None
    from sglang_neuron.sglang_patches.mlp_sync_cohort_snapshot import ATTR

    if ATTR in vars(forward_batch):
        setattr(batch, ATTR, vars(forward_batch)[ATTR])
    return batch


def execute(original_fn, self, forward_batch, pp_proxy_tensors=None, **kwargs):
    """AROUND EagerRunner.execute."""
    dispatch = getattr(self, "_platform_dispatch", None)
    if dispatch is None:
        return original_fn(self, forward_batch, pp_proxy_tensors, **kwargs)
    from sglang.srt.model_executor.forward_context import forward_context, get_forward_context
    from sglang.srt.utils.device_timer import device_timer_ctx

    with self.model_context():
        batch = self.load_batch(forward_batch, pp_proxy_tensors)
        with (device_timer_ctx(self.model_runner.device_timer, "megakernel"),
              forward_context(get_forward_context(), forward_batch=batch)):
            return dispatch(batch.input_ids, batch.positions, batch,
                            **self.model_runner._pp_kwargs(pp_proxy_tensors))


def register(registry, around) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
    registry.register(f"{_RUNNER}.EagerRunner.__init__", init, around)
    registry.register(f"{_RUNNER}.EagerRunner.load_batch", load_batch, around)
    registry.register(f"{_RUNNER}.EagerRunner.execute", execute, around)
