# SPDX-License-Identifier: Apache-2.0
"""CPU placement for a launch's processes, set in the launching process before any scheduler starts.

At each step's collectives every rank of a server waits for the slowest one, in decode and prefill alike. A scheduler
that loses its CPU for a moment therefore delays the whole server. Two causes are handled here:

* On a trn3 host the hypervisor periodically takes vCPU 2 and its hyperthread siblings. The launching process drops
  them from its CPU affinity, and every process it starts inherits that.
* The kernel's NUMA balancing migrates an unbound scheduler's pages, and the hint faults it raises land inside the
  step's submit. An unbound prefill scheduler took many of them while serving, and one rank's late
  launch held up a forward's collectives for the others. So every launch binds each scheduler to the NUMA node of its
  own NeuronCore (SGLang's --numa-node, applied in process through libnuma; sglang_patches.numa_node_published). The
  node comes from the driver's device map, since NeuronCore numbering does not follow PCI order. A server whose cores
  span both nodes thus binds each rank next to its core.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_STEAL_CPU = 2


def parse_cpu_list(spec: str) -> set:
    """A kernel CPU list ('2,98', '2-3')."""
    cpus = set()
    for part in filter(None, (p.strip() for p in spec.split(","))):
        lo, _, hi = part.partition("-")
        cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


def reserved_cpus(sys_root: Path = Path("/")) -> set:
    """The CPUs the hypervisor takes from this host: vCPU 2 and its hyperthread siblings on a trn3 host, else none."""
    try:
        product = (sys_root / "sys/devices/virtual/dmi/id/product_name").read_text().strip()
    except OSError:
        return set()
    if not product.startswith("trn3"):
        return set()
    siblings = sys_root / f"sys/devices/system/cpu/cpu{_STEAL_CPU}/topology/thread_siblings_list"
    try:
        return parse_cpu_list(siblings.read_text()) | {_STEAL_CPU}
    except OSError:
        return {_STEAL_CPU}


def leave_reserved_cpus_free(sys_root: Path = Path("/")) -> None:
    allowed = os.sched_getaffinity(0)
    left = allowed - reserved_cpus(sys_root)
    if left and left != allowed:
        os.sched_setaffinity(0, left)
        logger.info("[neuron] the server leaves CPUs %s free", sorted(allowed - left))


def core_numa_nodes(devices: list, cores=None) -> list:
    """The NUMA node of each of ``cores``, by default every NeuronCore of the node in order, from neuron-ls's device
    list."""
    nodes = {int(core): int(d["numa_node"]) for d in devices for core in d["neuroncore_ids"]}
    return [nodes[core] for core in (range(len(nodes)) if cores is None else cores)]


def server_cores():
    """The server's NeuronCores as the launch names them: NEURON_VISIBLE_DEVICES, else NEURON_RT_VISIBLE_CORES (a range
    list such as ``32-35``). Each rank picks its core the same way (runtime_init.prepare_process_env). None when
    neither is set, meaning every core of the node in order."""
    from sglang_neuron.utils.hardware_config import parse_range_list

    for name in ("NEURON_VISIBLE_DEVICES", "NEURON_RT_VISIBLE_CORES"):
        if os.environ.get(name):
            return parse_range_list(os.environ[name])
    return None


def bind_to_numa(server_args, devices=None) -> None:
    """Default --numa-node to each scheduler's NeuronCore node, for any server role, and have SGLang apply any
    --numa-node in process."""
    if server_args.numa_node is None:
        # SGLang indexes the list by each scheduler's gpu_id, its place in the server's cores
        if devices is None:
            from sglang_neuron.device import neuron_ls

            devices = neuron_ls()
        server_args.numa_node = core_numa_nodes(devices, server_cores())
    # SGLang's default path re-executes each scheduler under numactl, which binds nothing on an image without it
    from sglang_neuron import serving_policy

    serving_policy.set_launch_setting("SGLANG_NUMA_BIND_V2", "0", "--numa-node binds each scheduler in process "
                                      "(libnuma); SGLang's numactl path needs numactl in the image")
