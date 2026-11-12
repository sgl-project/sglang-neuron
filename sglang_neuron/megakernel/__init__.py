# SPDX-License-Identifier: Apache-2.0
"""Whole-model kernels behind SGLang's megakernel contract (``MegakernelSpec`` / ``BuiltMegakernel``), executed on
torch-neuronx compiled graphs. docs/design/megakernel.md describes the design."""
from .built import MODES, NeuronBuiltMegakernel, NeuronForwardView, PhaseProgram, step_class
from .dispatch import NeuronMegakernelDispatch, attach, built_megakernel, check_declared, dispatch_of
from .pools import POOL_KINDS, BoundPools, PoolEntry, PoolLayout, paired_kv, per_layer, state_pool

__all__ = [
    "MODES", "NeuronBuiltMegakernel", "NeuronForwardView", "PhaseProgram", "step_class",
    "NeuronMegakernelDispatch", "attach", "built_megakernel", "check_declared", "dispatch_of",
    "POOL_KINDS", "BoundPools", "PoolEntry", "PoolLayout", "paired_kv", "per_layer", "state_pool",
]
