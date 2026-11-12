# SPDX-License-Identifier: Apache-2.0
"""Bind each scheduler to the NUMA node the launch resolved for it.

A scheduler process looks up its node with ``numa_utils.get_numa_node_if_available``, which reads the raw argument
``server_args.numa_node``. A platform declares its defaults as resolution results instead, and SGLang publishes those
into each process's runtime context (``get_device().numa_node``). Without this module, the node the platform picks for
each scheduler (cpu_placement.bind_to_numa) never reaches the scheduler, which then runs on every CPU of the host. Here
the scheduler's lookup reads the published value; a launch that publishes none keeps upstream's lookup.

Target md5-pinned against upstream SGLang 43881a7: ``numa_utils.get_numa_node_if_available``, wrapped where the
scheduler module imported it.

Upstream: get_numa_node_if_available could read the published device configuration; this module then goes away.
"""

from __future__ import annotations

TARGET = "sglang.srt.managers.scheduler.get_numa_node_if_available"
_SOURCE = ("sglang.srt.utils.numa_utils", "get_numa_node_if_available")
_SOURCE_MD5 = "25551fe1e87821bda8cf7b6bd0a1c98c"


def get_numa_node_if_available(original_fn, server_args, gpu_id):
    from sglang.srt.runtime_context import get_device

    nodes = get_device().numa_node
    if nodes is None:
        return original_fn(server_args, gpu_id)
    return nodes[gpu_id]


def register(registry, around) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check(source_guard.resolve(*_SOURCE), _SOURCE_MD5, ".".join(_SOURCE), __name__)
    registry.register(TARGET, get_numa_node_if_available, around)
