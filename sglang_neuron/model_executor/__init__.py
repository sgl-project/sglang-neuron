# SPDX-License-Identifier: Apache-2.0
"""Model execution on Neuron: the runtime that compiles and launches each step's model graph behind SGLang's graph
runner (graph_runner), its per-rank device initialization (runtime_init), the asynchronous launch and deferred token
materialization (async_execution, token_io), the program-switch
fence of an aggregated server (program_fence), and the device and host captures of a running server (profiling,
host_trace)."""
