# SPDX-License-Identifier: Apache-2.0
"""Per-rank Neuron runtime initialization.

An AROUND hook on ModelRunner.init_torch_distributed, each step once per process. SGLang's scheduler has already built
the gloo world and groups (bootstrap.init_parallel_runtime); the hooked method's memory probe first touches the device.
  Before that probe:
    1. NEURON_VISIBLE_DEVICES = the server's cores on this node: the launch's NEURON_RT_VISIBLE_CORES, else the first
       ones
    2. NEURON_RT_VISIBLE_CORES = this rank's single core (torch device neuron:<local rank>)
    3. NEURON_RT_MAP_HBM=1 (+ queue/ring/barrier tunables, with the model's own over them: its launch hooks'
       rank_runtime_env), the torch CPU thread cap, EFA interface affinity
    4. NEURON_RT_ROOT_COMM_ID: on one node every rank derives the same CCOM bootstrap address without a
       collective (root host = dist-init host; port keyed on the first visible device, or on
       the replica's first device under plain --dp-size); over several nodes every rank takes rank 0's, broadcast
       over the gloo world (ccom_rendezvous)
    5. the runtime start, which latches COMM_ID: parallel.neuron_init.init_neuron_ranks (torch-neuronx's global rank,
       world size, local rank and local world size, then its runtime start on neuron:<local rank>)
  After it:
    6. the torch world must be the one torch-neuronx was set up with, then a Neuron runtime barrier over the world
"""

from __future__ import annotations

import errno
import logging
import os
import socket
from typing import NamedTuple, Optional

import torch

from sglang_neuron import envs
from sglang_neuron.backend import envs as backend_envs
from sglang_neuron.serving_policy import SCRATCHPAD_PAGE_MB

logger = logging.getLogger(__name__)

_CCOM_PORT_BASE = 61234
_DONE = False
_LOCAL_RANK = 0


def neuron_device() -> torch.device:
    """This rank's Neuron device, neuron:<local rank>: torch-neuronx numbers a process's one device by its local rank
    (parallel.neuron_init). In CPU mode
    (SGLANG_NEURON_CPU_MODE=1: kernels fall back to torch) everything lives on the host."""
    if backend_envs.SGLANG_NEURON_CPU_MODE:
        return torch.device("cpu")
    return torch.device("neuron", _LOCAL_RANK)


# The runtime's ring of IO tensor sets that a program's executions cycle through (NEURON_RT_IO_RING_CACHE_SIZE), unless
# the launch sets it; it must stay above the in-flight bound (backend.device_io.check_io_ring)
IO_RING_CACHE_SIZE = 32


def runtime_defaults(role: str, giga_decode: bool = False, scratchpad_page_mb: int = SCRATCHPAD_PAGE_MB,
                     decode_layout=(), megakernel: bool = False) -> dict:
    """The Neuron runtime settings a rank of ``role`` starts with, unless the launch sets them.

    The runtime skips its per-execution barrier: the platform synchronizes the ranks where back-to-back executions could
    race, at the switches between programs (model_executor.program_fence: an aggregated server's, and a role's of more
    than one compiled shape); back-to-back executions of one program need none. A decode queues up to 63 executions per
    core and completes them without the runtime's low-latency polling thread, which competes with the scheduler for the
    host; a prefill forward is long enough that 32 queued executions cover it. The giga-kernel decode runs every
    layer's collectives inside its one execution a step, and the runtime serves their DMA queues at service level 3
    (lower is higher), the level GPT-OSS's giga-kernel decode is measured at; a model's rank hook may state its own
    decode's (_model_rank_env). The scratchpad page is the launch's (``scratchpad_page_mb``,
    serving_policy.ServingPolicy), which the graphs compile with too (graph_runner). The in-flight bound is the plugin's
    own (SGLANG_NEURON_MAX_INFLIGHT_LAUNCHES, backend.device_io), not a runtime setting.

    A whole-model megakernel (``megakernel``: serving_policy.ServingPolicy.megakernel) runs the same regime in every
    role it serves: each kernel call carries every layer's collectives over all the server's ranks, its programs were
    validated with the disaggregated roles' DMA settings, and a host per core competes with the runtime's polling
    thread. So every role of it gets the giga-kernel decode's settings: the DMA settings, 63 queued executions, no
    low-latency completion thread and the highest DMA service level.

    A model's declared decode layout (``decode_layout``: its ``runtime_env`` pairs, ServingPolicy.decode_runtime_env)
    runs the model's own programs, compiled for the layout's settings: its ranks start with those settings and with
    none of the role's defaults above, which the platform's own graphs are compiled and validated with (their
    scratchpad page among them)."""
    if decode_layout:
        return dict(decode_layout)
    settings = {
        "NEURON_SCRATCHPAD_PAGE_SIZE": str(int(scratchpad_page_mb)),
        "NEURON_RT_DISABLE_EXECUTION_BARRIER": "1",
        "NEURON_RT_IO_RING_CACHE_SIZE": str(IO_RING_CACHE_SIZE),
        "NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS": "63" if role == "decode" or megakernel else "32",
    }
    if role in ("prefill", "decode") or megakernel:
        # the disaggregated roles' DMA settings, which their graphs were validated with
        settings["NEURON_RT_DBG_VECTOR_DGE_SKIP_NOTIF"] = "1"
        settings["NEURON_RT_DBG_INTRA_RDH_CHANNEL_BUFFER_SIZE"] = str(80 * 2**20)
    if role == "decode" or megakernel:
        settings["NEURON_RT_XU_WORKER_ENABLE_LOW_LATENCY_COMPLETION"] = "0"
    if giga_decode or megakernel:
        settings["NEURON_RT_DBG_CC_DMA_QOS_SERVICE_LEVEL"] = "3"
    return settings


def prepare_process_env(*, local_rank: int, local_world_size: int, visible=None, block_start: int = 0) -> None:
    """Pin this rank to one NeuronCore and set runtime env (before dist init).

    The server's cores are the ones the launch names in NEURON_RT_VISIBLE_CORES (a range list such as ``32-35``), which
    every process of the launch inherits whole; ``block_start`` is this process's offset in them (a DP replica of a
    plain --dp-size launch). Without it the cores are ``visible``, this process's own block, else the first
    local_world_size. Each rank's NEURON_RT_VISIBLE_CORES then becomes its one core, so the server's cores are kept in
    NEURON_VISIBLE_DEVICES, and every rank keys the CCOM root on its world's first core (_WORLD_FIRST_CORE). A
    launch that sets NEURON_VISIBLE_DEVICES itself names the server's cores the same way."""
    if backend_envs.SGLANG_NEURON_CPU_MODE:
        return
    source = "NEURON_VISIBLE_DEVICES"
    if source not in os.environ:
        if os.environ.get("NEURON_RT_VISIBLE_CORES"):
            source = "NEURON_RT_VISIBLE_CORES"
            os.environ["NEURON_VISIBLE_DEVICES"] = os.environ["NEURON_RT_VISIBLE_CORES"]
        else:
            cores = list(visible) if visible is not None else list(range(local_world_size))
            os.environ["NEURON_VISIBLE_DEVICES"] = ",".join(str(i) for i in cores)
            block_start = 0
    from sglang_neuron.utils.hardware_config import parse_range_list

    listed = parse_range_list(os.environ["NEURON_VISIBLE_DEVICES"])
    if len(listed) < block_start + local_world_size:
        raise RuntimeError(
            f"{source} names {len(listed)} cores ({os.environ['NEURON_VISIBLE_DEVICES']}) but this process's block "
            f"needs cores [{block_start}, {block_start + local_world_size}) of them (local_world_size="
            f"{local_world_size})"
        )
    visible = listed[block_start:]
    core = visible[local_rank]
    global _LOCAL_RANK, _WORLD_FIRST_CORE
    _LOCAL_RANK = int(local_rank)
    # the world's first core, on which every rank keys the CCOM root port (root_comm_id)
    _WORLD_FIRST_CORE = int(visible[0])
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(core)
    os.environ["NEURON_RT_MAP_HBM"] = "1"  # EFA's RDMA into device memory (NIXL's KV transfers) needs HBM mapped
    from sglang_neuron import serving_policy

    policy = serving_policy.current()
    settings = runtime_defaults(policy.role, policy.giga_decode, scratchpad_page_mb=policy.scratchpad_page_mb,
                                decode_layout=policy.decode_runtime_env, megakernel=policy.megakernel)
    settings.update(_model_rank_env(policy.role))
    for name, value in settings.items():
        os.environ.setdefault(name, value)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    # SGLANG_NEURON_PROFILE_RANKS="0" (a comma list of local ranks): the Neuron runtime's device and system profile of
    # those ranks from process start, written under SGLANG_NEURON_PROFILE_DIR in a directory per NeuronCore (the local
    # ranks of plain --dp-size replicas repeat)
    if int(local_rank) in envs.profile_ranks():
        out = os.path.join(envs.profile_dir(), f"core{core}")
        os.makedirs(out, exist_ok=True)
        os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
        os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = out
        # the runtime's variable spells model mode 1
        os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "session" if envs.profile_device_mode() == "session" else "1"
        os.environ["NEURON_RT_INSPECT_SYSTEM_PROFILE"] = "1"
        logger.info("neuron profiling enabled for local_rank=%d -> %s", local_rank, out)
    _cap_torch_threads()
    # this core's EFA interface, which the disaggregated KV transfer runs on; a lookup that fails, a node without EFA
    # included, warns, since the rank's transfers then run without the affinity
    if "FI_EFA_IFACE" not in os.environ:
        try:
            from sglang_neuron.utils.hardware_config import get_efa_interface

            iface = get_efa_interface(local_rank, visible)
            if iface:
                os.environ["FI_EFA_IFACE"] = iface
        except Exception as e:  # noqa: BLE001 - any lookup failure is reported, and the rank runs without the affinity
            logger.warning("EFA interface affinity for local rank %d failed (%r): its transfers run without it; set "
                           "FI_EFA_IFACE to choose the interface", local_rank, e)
    logger.info(
        "neuron rank env: local_rank=%d core=%s visible=%s FI_EFA_IFACE=%s",
        local_rank, core, os.environ["NEURON_VISIBLE_DEVICES"], os.environ.get("FI_EFA_IFACE"),
    )


def _model_rank_env(role: str) -> dict:
    """The runtime settings the launch's model gives each rank of ``role`` over runtime_defaults
    (catalog.ModelSpec.launch_hooks rank_runtime_env), for the model whose package the launch names
    (catalog.MODEL_PACKAGE_ENV); a launch that sets one keeps its own value, and a model without launch rules states
    none. It reads the catalog alone: a rank may not have imported SGLang yet, before which the platform does not
    import."""
    from sglang_neuron.models import catalog

    hooks = catalog.served_launch_hooks()
    return {} if hooks is None else dict(hooks.rank_runtime_env(role))


def _cap_torch_threads() -> None:
    """Cap torch's CPU intra-op and inter-op pools at two threads per rank.

    torch sized its pools to the host when it was imported, before prepare_process_env exported OMP_NUM_THREADS. A
    scheduler's host ops are small, but each fans out over the whole pool, and one pool per rank oversubscribes the
    host's CPUs many times over."""
    n = 2
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(n)
    except RuntimeError:
        pass  # interop pool already started; intra-op cap is what matters
    logger.info("torch cpu threads capped: intra-op=%d", torch.get_num_threads())


def _root_host(dist_init_addr: str | None) -> str:
    if dist_init_addr:
        host = dist_init_addr.rsplit(":", 1)[0].strip("[]")
        if host and host not in ("0.0.0.0", "::"):
            return host
    return os.environ.get("MASTER_ADDR") or socket.gethostbyname(socket.gethostname())


# The world's first core (prepare_process_env): the replica's first core under plain --dp-size
_WORLD_FIRST_CORE: Optional[int] = None
# the offsets past the base a launch's CCOM root ports may take
_CCOM_OFFSET_SPAN = 900


def root_comm_id(dist_init_addr: str | None = None) -> str:
    """The CCOM bootstrap address every rank of the world computes without a collective: the dist-init host and the
    port of the world's first core, _CCOM_PORT_BASE + core + the launch's offset (SGLANG_NEURON_CCOM_PORT_OFFSET,
    ccom_port_offset)."""
    off = int(os.environ.get("SGLANG_NEURON_CCOM_PORT_OFFSET", "0") or 0)
    if _WORLD_FIRST_CORE is None:
        raise RuntimeError("the CCOM root is keyed on the world's first core, which prepare_process_env sets first")
    return f"{_root_host(dist_init_addr)}:{_CCOM_PORT_BASE + _WORLD_FIRST_CORE + off}"


def ccom_port_offset(cores, node_cores: int, host: str, free=None) -> int:
    """A launch's CCOM port offset: the smallest multiple of the node's core count at which the root port of each of
    ``cores``, _CCOM_PORT_BASE + core + offset, binds free on the root ``host``. A world keys its root on its first
    core, so two launches on disjoint cores of a node never share a port at any of these offsets, and a relaunch steps
    past a port a dying predecessor's root still holds, where the bootstrap ring would fail with "connection closed by
    remote peer"."""
    free = free or _port_free
    for offset in range(0, _CCOM_OFFSET_SPAN, int(node_cores)):
        if all(free(host, _CCOM_PORT_BASE + int(core) + offset) for core in cores):
            return offset
    raise RuntimeError(f"no CCOM root port offset below {_CCOM_OFFSET_SPAN} binds free on {host} for the launch's "
                       f"cores {list(cores)}; stop the processes holding ports from {_CCOM_PORT_BASE}")


def launch_ccom_port_offset(server_args) -> int:
    """The CCOM port offset of a launch on this node: ccom_port_offset over the launch's cores (cpu_placement.
    server_cores, else every core of the node) on its root host. The launching process of node 0 decides it, since the
    root host is node 0 (--dist-init-addr); on a server over several nodes the other nodes' ranks take node 0's root
    when their runtime starts (ccom_rendezvous)."""
    if backend_envs.SGLANG_NEURON_CPU_MODE:
        return 0
    from sglang_neuron import cpu_placement
    from sglang_neuron.device import neuron_ls

    node_cores = sum(len(device["neuroncore_ids"]) for device in neuron_ls())
    cores = cpu_placement.server_cores() or range(node_cores)
    return ccom_port_offset(cores, node_cores, _root_host(getattr(server_args, "dist_init_addr", None)))


def _port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
        except OSError as error:
            if error.errno == errno.EADDRINUSE:
                return False
            raise RuntimeError(f"the CCOM root host {host} (--dist-init-addr, else MASTER_ADDR or this node) is not "
                               f"an IPv4 address of this node ({error}); launch with --dist-init-addr <this node's "
                               "address>") from error
    return True


def set_root_comm_id(dist_init_addr: str | None = None, addr: str | None = None) -> None:
    """Export NEURON_RT_ROOT_COMM_ID *before* anything touches NRT (NRT latches it at nrt_init; SGLang's own dist init
    already probes device memory): ``addr`` (rank 0's root, ccom_rendezvous), else this rank's root_comm_id."""
    if backend_envs.SGLANG_NEURON_CPU_MODE:
        return
    addr = addr or root_comm_id(dist_init_addr)
    os.environ["NEURON_RT_ROOT_COMM_ID"] = addr
    logger.info("CCOM bootstrap: NEURON_RT_ROOT_COMM_ID=%s", addr)


def ccom_rendezvous(dist_init_addr: str | None = None) -> Optional[str]:
    """The CCOM root every rank of a server over several nodes starts its runtime with: rank 0's root_comm_id, whose
    port offset node 0's launching process chose on the root host (launch_ccom_port_offset), broadcast over SGLang's
    gloo world, which is up before the runtime starts. Rank 0 sends its serving policy with it, and a rank whose own
    launching process resolved another policy is refused: ranks whose mechanisms differ would run different collectives
    and hang without a word. None in CPU mode."""
    if backend_envs.SGLANG_NEURON_CPU_MODE:
        return None
    import torch.distributed as dist

    from sglang_neuron import serving_policy

    policy = os.environ.get(serving_policy.ENV_KEY)
    sent = [(root_comm_id(dist_init_addr), policy) if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(sent, src=0, device=torch.device("cpu"))
    root, root_policy = sent[0]
    if root_policy != policy:
        raise RuntimeError(f"serving policy mismatch: rank {dist.get_rank()} runs {policy} but rank 0 runs "
                           f"{root_policy} (the nodes' launch arguments or environments differ)")
    logger.info("CCOM rendezvous: rank %d of %d starts from rank 0's root %s", dist.get_rank(), dist.get_world_size(),
                root)
    return root


def _sigterm_to_exit(signum, frame) -> None:
    # once: a second TERM must not interrupt the exit the first one started (SIGKILL still ends the process)
    import signal

    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise SystemExit(128 + signum)


def install_clean_exit() -> None:
    """A rank stopped with SIGTERM exits through the interpreter's normal exit, which closes the Neuron runtime.

    torch-neuronx closes the runtime (nrt_close, from its C extension) at the interpreter's normal exit only; Python's
    default SIGTERM action ends the process without it and leaves the devices unclosed for the next server, whose
    collectives can then stall. The handler turns the first SIGTERM into SystemExit. Installed from the main thread
    only (signal handlers live there); idempotent."""
    import signal
    import threading

    if (threading.current_thread() is threading.main_thread()
            and signal.getsignal(signal.SIGTERM) is not _sigterm_to_exit):
        signal.signal(signal.SIGTERM, _sigterm_to_exit)


def node_local_width(tp_size: int, pp_size: int, nnodes: int) -> int:
    """The ranks SGLang spawns on one node (engine._calculate_rank_ranges): ``pp_size // nnodes`` stages (at least one)
    of the TP ranks a node holds, ``tp_size`` over the nodes one TP group spans. ``tp_size * pp_size`` on one node."""
    pp_per_node = max(int(pp_size) // int(nnodes), 1)
    nnodes_per_tp_group = max(int(nnodes) // int(pp_size), 1)
    return pp_per_node * (int(tp_size) // nnodes_per_tp_group)


class TorchWorld(NamedTuple):
    """This process's place in SGLang's torch world (distributed.bootstrap._init_parallel_groups) and on its node."""
    rank: int
    world_size: int
    local_rank: int
    local_world_size: int


def torch_world(ps, local_rank: int, local_world_size: int) -> TorchWorld:
    """The torch world rank and size SGLang's dist init will create, from the same inputs (``ps``: SGLang's resolved
    parallel ranks and widths): ``tp_size * pp_size`` ranks, PP-major, rank ``pp_rank * tp_size + tp_rank``
    (runtime_context.spawn_world_rank)."""
    tp_size, pp_size = int(ps.tp_size), int(getattr(ps, "pp_size", 1) or 1)
    rank = int(getattr(ps, "pp_rank", 0) or 0) * tp_size + int(ps.tp_rank)
    return TorchWorld(rank, tp_size * pp_size, int(local_rank), int(local_world_size))


def runtime_initialize(world: TorchWorld) -> None:
    """Start this rank's Neuron runtime (step 5)."""
    if backend_envs.SGLANG_NEURON_CPU_MODE:
        return
    from sglang_neuron.backend import device_io
    from sglang_neuron.parallel.neuron_init import init_neuron_ranks

    device_io.check_streams()
    # no retry: torch-neuronx's rank setters accept one value per process
    init_neuron_ranks(world.rank, world.world_size, world.local_rank, world.local_world_size)
    install_clean_exit()
    from sglang_neuron import serving_policy

    if serving_policy.current().hbm_tail_guard:
        # before any NEFF loads: the arena's top is held before an instruction stream can end there
        from sglang_neuron.backend import hbm_tail_guard

        hbm_tail_guard.install()


def check_overlap_schedule() -> None:
    """Check that the scheduler's final overlap setting matches serving_policy.ServingPolicy.overlap_schedule (async
    execution, off on a pipelined PD prefill).

    SGLang's passes after the platform hook could flip it (its PP override turns it off, model-specific passes could
    too), leaving the async executor under a synchronous loop or the reverse. Read it from the schedule
    config the scheduler uses (runtime_context.get_schedule), where those passes land; server_args still holds the
    platform's value."""
    from sglang.srt.runtime_context import get_schedule

    from sglang_neuron import serving_policy

    policy = serving_policy.current()
    disabled = bool(get_schedule().disable_overlap_schedule)
    if disabled == policy.overlap_schedule:
        raise ValueError(f"disable_overlap_schedule={disabled} with serving_policy.async_execution="
                         f"{policy.async_execution} and pp_size={policy.pp_size}: the scheduler's overlap "
                         "schedule and the executor disagree")


def init_torch_distributed_hook(original_fn, self, *args, **kwargs):
    """AROUND ModelRunner.init_torch_distributed."""
    global _DONE
    sa = self.server_args
    check_overlap_schedule()
    from sglang_neuron import sglang_tree

    ps = sglang_tree.parallel_of(self)
    tp_size = int(ps.tp_size)
    nnodes = int(getattr(sa, "nnodes", 1) or 1)
    replicas = sglang_tree.dp_widths(sa)[0]
    gpu_id = int(getattr(self, "gpu_id", 0))
    visible = None
    block_start = 0
    # the node's ranks: on one node every PP stage's TP ranks, which SGLang gives gpu_id pp_rank * tp + tp_rank
    # (weight_cache.protocol.compute_local_gpu_id), so each stage gets its own cores; over several nodes the ranks
    # SGLang spawns on each node, whose gpu_id is their place on it
    local_world_size = node_local_width(tp_size, int(getattr(ps, "pp_size", 1) or 1), nnodes)
    local_rank = gpu_id % local_world_size
    if replicas > 1:
        # independent DP engines (plain --dp-size): SGLang hands replica d the gpu_ids [d * tp, (d + 1) * tp). Each
        # replica gets its own core block, and its first core keys its CCOM root port, so replicas never collide.
        base = (gpu_id // tp_size) * tp_size
        visible = range(base, base + tp_size)
        block_start = base
    dist_init_addr = getattr(sa, "dist_init_addr", None)
    world = torch_world(ps, local_rank, local_world_size)
    if not _DONE:
        prepare_process_env(local_rank=local_rank, local_world_size=local_world_size, visible=visible,
                            block_start=block_start)
        # NRT reads NEURON_RT_ROOT_COMM_ID at nrt_init, and the hooked method's
        # memory probe starts NRT (an implicit nrt_init): set it and start first.
        if nnodes > 1:
            set_root_comm_id(dist_init_addr, ccom_rendezvous(dist_init_addr))
        else:
            set_root_comm_id(dist_init_addr)
        runtime_initialize(world)
    ret = original_fn(self, *args, **kwargs)
    if not _DONE:
        check_runtime_world(world)
        _DONE = True
    return ret


def check_runtime_world(world: TorchWorld) -> None:
    """Step 6: the torch world SGLang created is the one torch-neuronx runs as, and every rank's runtime has joined the
    world's collectives bootstrap."""
    if backend_envs.SGLANG_NEURON_CPU_MODE:
        return
    import torch.distributed as dist

    from sglang_neuron.parallel import neuron_init

    neuron_init.check_torch_world(dist.get_rank(), dist.get_world_size())
    neuron_init.runtime_barrier()
    logger.info("Neuron runtime barrier passed (rank %d of %d)", world.rank, world.world_size)
