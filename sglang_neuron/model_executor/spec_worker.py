# SPDX-License-Identifier: Apache-2.0
"""Speculative decoding on Neuron: the served model's own speculative worker, through SGLang's out-of-tree platform
seam (``Platform.get_speculative_worker_cls`` and ``supports_speculative_algorithm``).

SGLang's built-in speculative workers keep their bookkeeping in device tensors and Triton or sgl_kernel ops, and run a
draft model of their own; on Neuron each eager device op is a compiled NEFF. So a model that serves speculative
decoding declares the algorithms in its catalog entry (catalog.ModelSpec.speculative_algorithms), and its model class
names its worker (``speculative_worker_cls(algorithm)``, a class with the interface of SGLang's ``BaseSpecWorker``):

- the launching process lifts the refusal of ``--speculative-algorithm`` for a declared algorithm alone
  (platform.refuse_unserved), and the serving policy records it (serving_policy.speculative_algorithm);
- ``NeuronSpeculativeWorker`` is the class the platform returns to SGLang. SGLang's scheduler constructs it with the
  built-in worker's keyword arguments once the target is loaded, and it builds the loaded model's own worker;
- ``model_class`` finds a served architecture's class in SGLang's model registry, which the plugin points at the
  model's package (general_plugin._register_model_package).
"""

from __future__ import annotations

from typing import Optional, Sequence


def model_class(architectures: Sequence[str]) -> Optional[type]:
    """The class SGLang's model registry builds for the first of ``architectures`` it holds, or None."""
    from sglang.srt.models.registry import ModelRegistry

    return next((ModelRegistry.models[name] for name in architectures or () if name in ModelRegistry.models), None)


def worker_of(cls, algorithm: Optional[str]) -> Optional[type]:
    """The worker class model class ``cls`` serves SGLang's ``algorithm`` with, or None."""
    offer = getattr(cls, "speculative_worker_cls", None)
    if cls is None or offer is None or not algorithm or str(algorithm).upper() == "NONE":
        return None
    return offer(str(algorithm).upper())


def speculative_worker_cls(architectures: Sequence[str], algorithm: Optional[str]) -> Optional[type]:
    return worker_of(model_class(architectures), algorithm)


class NeuronSpeculativeWorker:
    """What ``NeuronSRTPlatform.get_speculative_worker_cls`` returns: constructed by SGLang's scheduler with the
    built-in worker's keyword arguments, it returns the loaded target model's own worker, or refuses."""

    def __new__(cls, *, server_args, gpu_id, nccl_port, target_worker, **kwargs):
        from sglang_neuron import serving_policy

        model = target_worker.model_runner.model
        algorithm = getattr(server_args, "speculative_algorithm", None)
        served = serving_policy.current().speculative_algorithm
        worker_cls = worker_of(type(model), algorithm) if str(algorithm).upper() == served else None
        if worker_cls is None:
            raise ValueError(f"speculative decoding ({algorithm}) is not served on Neuron for {type(model).__name__}: "
                             "its catalog entry declares no such algorithm, or its model class names no worker for it")
        return worker_cls(server_args=server_args, gpu_id=gpu_id, nccl_port=nccl_port, target_worker=target_worker,
                          **kwargs)
