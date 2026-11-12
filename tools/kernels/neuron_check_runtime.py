# SPDX-License-Identifier: Apache-2.0
"""The Neuron runtime and graph compilation of the Qwen3.5 compile probes (qwen35_decode_compile.py,
qwen35_prefill_compile.py).

* ``start_rank_runtime(rank, world_size)`` starts this process's Neuron runtime on ``neuron:<rank>`` (one process per
  core of one node): the rank state torch-neuronx loads collective NEFFs with (parallel/neuron_init.py), then its
  runtime.
* ``CheckGraph(fn)`` compiles ``fn`` for the check with the serving graphs' compiler arguments. ``precompile(*inputs)``
  compiles it without running it (torch-neuronx's compile-only mode under the compiler environment of those
  arguments), so a check can make sure every rank has its NEFF before any rank enters a graph's collectives; calling
  the instance runs it, from the NEFF precompile built.

Call them after the check has set its Neuron environment and run sglang_neuron._init_neuron_stack().
"""
from __future__ import annotations


def start_rank_runtime(rank: int, world_size: int) -> None:
    from sglang_neuron.parallel.neuron_init import init_neuron_ranks

    init_neuron_ranks(rank, world_size, rank, world_size)


class CheckGraph:
    def __init__(self, fn, compiler_args=None):
        import torch

        from sglang_neuron.backend import stack
        from sglang_neuron.model_executor.graph_runner import COMPILER_ARGS

        self.compiler_args = list(COMPILER_ARGS if compiler_args is None else compiler_args)
        # Two dynamo entries of one function: the compile-only one never runs, the serving one loads the NEFF it built.
        # The compile-only one goes through its own backend callable: dynamo shares one entry between torch.compile
        # wrappers of the same backend name and settings, so the serving call would get the compile-only handle
        # (placeholder outputs).
        neuron = torch._dynamo.lookup_backend(stack.compile_backend_name())

        def compile_only_backend(gm, example_inputs):
            return neuron(gm, example_inputs)

        self._precompiled = torch.compile(fn, backend=compile_only_backend, fullgraph=True, dynamic=False)
        self._compiled = torch.compile(fn, backend=stack.compile_backend_name(), fullgraph=True, dynamic=False)

    def precompile(self, *inputs) -> None:
        """Compile the graph of ``inputs`` without running it."""
        from torch_neuronx.compiler import compile_only

        from sglang_neuron.backend import stack

        with compile_only(), stack.compile_scope(self.compiler_args):
            self._precompiled(*inputs)

    def __call__(self, *inputs):
        from sglang_neuron.backend import stack

        with stack.compile_scope(self.compiler_args):
            return self._compiled(*inputs)
