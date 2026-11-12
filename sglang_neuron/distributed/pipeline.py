# SPDX-License-Identifier: Apache-2.0
"""Pipeline-parallel stages: the layer split, the launch checks, and the stage ring's connection.

SGLang runs the pipeline itself: one scheduler per (pp_rank, tp_rank), the micro-batches, the proxy dict that carries
each stage boundary, and KV pools sized from the model's start_layer and end_layer. The plugin moves the boundary
tensors HBM to HBM over its NIXL seam (distributed/pp_seam.py). This module owns:

* ``current_stage_spec``: the stage's layers, from SGLang's own split (``get_pp_indices``, which honours
  SGLANG_PP_LAYER_PARTITION), so the plugin's model and SGLang's pools agree on the layers;
* ``validate_pp_launch``: one node only. The PD roles run asynchronously (a prefill's last stage runs async; a decode
  runs under SGLang's overlap schedule and its last stage relays its sampled tokens to the first). An aggregated server
  runs SGLang's own PP loop synchronously. Everything else is refused at launch, with every reason named;
* ``connect_pp_ring``: connect the stage ring's gloo pairs before the PP loop's first send.

Ranks are PP-major, as SGLang numbers them: world rank = pp_rank * tp_size + tp_rank.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StageSpec:
    """The layers one pipeline stage builds: global layer ids ``[start_layer, end_layer)`` of ``num_hidden_layers``."""

    pp_rank: int
    pp_size: int
    start_layer: int
    end_layer: int
    num_hidden_layers: int

    def __post_init__(self) -> None:
        if not 0 <= self.pp_rank < self.pp_size:
            raise ValueError(f"pp_rank {self.pp_rank} is outside [0, {self.pp_size})")
        if not 0 <= self.start_layer < self.end_layer <= self.num_hidden_layers:
            raise ValueError(f"stage {self.pp_rank} of {self.pp_size} holds layers [{self.start_layer}, "
                             f"{self.end_layer}) of {self.num_hidden_layers}: every stage needs at least one layer "
                             "inside the stack")

    @property
    def is_first(self) -> bool:
        return self.pp_rank == 0

    @property
    def is_last(self) -> bool:
        return self.pp_rank == self.pp_size - 1


def current_stage_spec(num_hidden_layers: int) -> Optional[StageSpec]:
    """The stage this process serves, or None without PP; a model's host calls it when it builds its layers.

    The split is SGLang's own (``sglang.srt.distributed.utils.get_pp_indices``), the same call that sizes SGLang's KV
    pools and picks the layers its PD transfer sends."""
    from sglang_neuron import sglang_tree

    pc = sglang_tree.parallel_config()
    pp_size = int(getattr(pc, "pp_size", 1) or 1)
    if pp_size <= 1:
        return None
    from sglang.srt.distributed.utils import get_pp_indices

    pp_rank = int(pc.pp_rank)
    start, end = get_pp_indices(num_hidden_layers, pp_rank, pp_size)
    return StageSpec(pp_rank=pp_rank, pp_size=pp_size, start_layer=start, end_layer=end,
                     num_hidden_layers=num_hidden_layers)


def _partition_problems(pp_size: int) -> list:
    """What is wrong with SGLANG_PP_LAYER_PARTITION for ``pp_size`` stages, parsed as SGLang's get_pp_indices parses it.
    SGLang checks the sum when the model builds, but accepts an empty stage, which the plugin refuses."""
    raw = os.environ.get("SGLANG_PP_LAYER_PARTITION")
    if raw is None:
        return []
    try:
        counts = [int(x) for x in raw.split(",")]
    except ValueError:
        return [f"SGLANG_PP_LAYER_PARTITION={raw!r}: set it to a comma list of integers, one per stage"]
    problems = []
    if len(counts) != pp_size:
        problems.append(f"SGLANG_PP_LAYER_PARTITION={raw!r} has {len(counts)} entries for --pp-size {pp_size}")
    if any(c <= 0 for c in counts):
        problems.append(f"SGLANG_PP_LAYER_PARTITION={raw!r}: every stage needs at least one layer")
    return problems


def pp_launch_problems(server_args, policy=None) -> list:
    """Why this pipelined launch cannot run, or []. Each reason names the argument or variable to change.

    Served: one node, no async batch depth. A PD role needs async execution: a prefill's last stage runs async so the
    stage with the longer forward never waits on the host, and a decode runs under SGLang's overlap schedule with the
    sampled-token relay (``pp_decode_problems``). An aggregated server runs SGLang's own PP loop synchronously
    (serving_policy.resolve) and pays a host wait per micro-batch. --nnodes, --attn-cp-size and speculative decoding
    stay refused by platform.refuse_unserved, and --dcp-size with PP by distributed.decode_context_parallel. Without
    ``policy`` the overlap schedule is read from ``server_args``."""
    from sglang.srt.environ import envs as sgl_envs

    from sglang_neuron import sglang_tree

    pp_size = int(getattr(server_args, "pp_size", 1) or 1)
    problems = []
    mode = str(getattr(server_args, "disaggregation_mode", "null") or "null")
    pd_role = mode in ("prefill", "decode")
    if mode == "decode":
        problems += [f"a pipelined PD decode: {p}" for p in pp_decode_problems(server_args, policy)]
    async_execution = (policy.async_execution if policy is not None
                       else not getattr(server_args, "disable_overlap_schedule", False))
    if pd_role and not async_execution:
        problems.append("--disable-overlap-schedule: it turns off async execution, which a pipelined PD role runs on "
                        "(a prefill's asynchronous last stage, a decode's PP loop under SGLang's overlap schedule)")
    depth = int(getattr(server_args, "pp_async_batch_depth", 0) or 0)
    if depth:
        loop = ("a pipelined PD role runs its PP loop in the overlap order" if pd_role else
                "a pipelined aggregated server is served on SGLang's PP loop, and no run covers the last stage's "
                "buffered outputs")
        problems.append(f"--pp-async-batch-depth {depth}: {loop}, with pp_size micro-batches")
    replicas, attention_dp = sglang_tree.dp_widths(server_args)
    if replicas > 1:
        problems.append(f"--dp-size {replicas}: no run covers DP replicas of a pipelined server")
    if attention_dp > 1:
        problems.append("DP attention (--enable-dp-attention, --attn-dp-size): the stage graphs and the seam carry "
                        "one TP group's batch")
    if getattr(server_args, "enable_nvidia_megakernel", False):
        problems.append("--enable-nvidia-megakernel: a whole-model kernel holds every layer, not a stage's")
    if sgl_envs.SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM.get():
        problems.append("SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM: the seam sends every stage boundary in order, and a "
                        "skipped one would put the stages' slots out of step")
    if getattr(server_args, "enable_dynamic_chunking", False):
        problems.append("--enable-dynamic-chunking: SGLang profiles its chunk sizes on the first stage alone, at sizes "
                        "off the compiled prefill")
    for name in ("SGLANG_REQ_WAITING_TIMEOUT", "SGLANG_REQ_RUNNING_TIMEOUT"):
        value = getattr(sgl_envs, name).get()
        if value is not None and float(value) > 0:
            problems.append(f"{name}={value}: each stage would time a request out by its own clock, at another loop "
                            "iteration, and the stages would schedule different batches")
    workers = int(getattr(server_args, "tokenizer_worker_num", 1) or 1)
    if workers > 1:
        problems.append(f"--tokenizer-worker-num {workers}: no PP run covers a tokenizer worker's pause and flush "
                        "paths")
    running = getattr(server_args, "max_running_requests", None)
    floor = pp_prefill_admission_floor(pp_size)
    if (mode == "prefill" and running is not None and int(running) <= floor and policy is not None
            and policy.prefill_compute_ahead):
        problems.append(f"--max-running-requests {running}: a {pp_size}-stage PD prefill that computes ahead holds up "
                        f"to {floor} requests at once (pp_prefill_admission_floor) and computes ahead only in the rows "
                        f"above them; launch with more than {floor}")
    interval = int(getattr(server_args, "scheduler_recv_interval", 1) or 1)
    if interval != 1:
        # a later stage takes the requests the first stage relays only when it receives, and the PP loop's consensus
        # needs every stage to see each iteration's requests and aborts at the same point
        problems.append(f"--scheduler-recv-interval {interval}: every stage must receive every iteration")
    problems += _partition_problems(pp_size)
    return problems


def pp_decode_problems(server_args, policy=None) -> list:
    """Why a pipelined PD decode cannot run, or []. Its stages run SGLang's PD decode PP loop in overlap order, the last
    stage relays its sampled tokens to the first over the seam (``ServingPolicy.pp_token_relay``), and each decode graph
    writes its boundary into the seam slots in place. That needs one decode batch size equal to the micro-batch
    (sglang_patches/pp_decode_micro_batch_bound holds each micro-batch to ``pp_max_micro_batch_size``), the per-layer
    decode graph, and sampling in the step's own graph. The stages must also keep the same free KV pages
    (sglang_patches/pp_decode_held_release), so the decode radix cache is refused: each stage would match a new request
    against its own tree."""
    from sglang.srt.environ import envs as sgl_envs

    from sglang_neuron import serving_policy

    problems = []
    if policy is not None and policy.giga_decode:
        problems.append("the giga-kernel decode runs every layer in one kernel call, not a stage's")
    if getattr(server_args, "disaggregation_decode_enable_radix_cache", False):
        # DecodePreallocQueue.pop_preallocated matches each agreed request in _match_prefix_and_lock against the stage's
        # own tree
        problems.append("--disaggregation-decode-enable-radix-cache: each stage matches a new request's prefix against "
                        "its own radix tree when it preallocates the request, and SGLang's PD decode PP loop agrees on "
                        "the requests the stages preallocate, transfer and retract but not on that match, which sets "
                        "the KV pages each stage allocates; the stages must keep the same free pages "
                        "(sglang_patches/pp_decode_held_release)")
    if sgl_envs.SGLANG_ENABLE_DELAY_SAMPLE.get():
        # a delayed sample happens in a later launch the overlap-order loop never runs, so the last stage would relay
        # nothing
        problems.append("SGLANG_ENABLE_DELAY_SAMPLE: the relay sends the tokens each decode graph samples itself")
    sizes = serving_policy.listed_decode_batch_sizes(server_args)
    running = getattr(server_args, "max_running_requests", None)
    micro = micro_batch_size(server_args)
    if len(sizes) != 1 or micro is None or int(sizes[0]) != micro:
        # one decode graph, whose boundary fills the seam's slots: SGLang would otherwise pick smaller batch graphs
        want = "B" if micro is None else micro
        problems.append(f"one decode batch size, the micro-batch: launch with --max-running-requests and "
                        f"--cuda-graph-bs-decode {want}, the micro-batch SGLang runs (--pp-max-micro-batch-size, or "
                        f"--max-running-requests // --pp-size) (listed {list(sizes) or 'none'}, "
                        f"--max-running-requests {running})")
    return problems


def micro_batch_size(server_args) -> Optional[int]:
    """The requests one micro-batch holds, as SGLang's scheduler sets it: --pp-max-micro-batch-size, or
    max(--max-running-requests // --pp-size, 1); None without either."""
    explicit = getattr(server_args, "pp_max_micro_batch_size", None)
    if explicit:
        return int(explicit)
    running = getattr(server_args, "max_running_requests", None)
    if running is None:
        return None
    return max(int(running) // int(getattr(server_args, "pp_size", 1) or 1), 1)


def validate_pp_launch(server_args, policy=None) -> None:
    """Refuse a pipelined launch the plugin does not serve, naming every reason (``pp_launch_problems``); a launch
    without PP passes. A prefill below the admission floor gets a warning, an aggregated server a log line about its
    host waits."""
    pp_size = int(getattr(server_args, "pp_size", 1) or 1)
    if pp_size <= 1:
        return
    problems = pp_launch_problems(server_args, policy)
    if problems:
        raise ValueError(f"--pp-size {pp_size} cannot launch:\n  - " + "\n  - ".join(problems))
    mode = str(getattr(server_args, "disaggregation_mode", ""))
    if mode not in ("prefill", "decode"):
        logger.info("[neuron] pipelined aggregated server accepted: --pp-size %d. SGLang's PP loop waits on the host "
                    "for each micro-batch's outputs before it schedules the micro-batch again: served, with a stage's "
                    "device idle while its host waits", pp_size)
        return
    running = getattr(server_args, "max_running_requests", None)
    floor = pp_prefill_admission_floor(pp_size)
    if mode == "prefill" and running is not None and int(running) < floor:
        logger.warning("[neuron] --pp-size %d with --max-running-requests %d: the stages hold up to %d requests at once "
                       "(pp_prefill_admission_floor), so fewer leave a stage idle between forwards", pp_size,
                       int(running), floor)
    logger.info("[neuron] pipelined PD %s accepted: --pp-size %d", mode, pp_size)


def default_pp_prefill_admission(server_args, computes_ahead: bool = False) -> Optional[int]:
    """A pipelined PD prefill launched without --max-running-requests admits ``4 x --pp-size`` requests (32 at PP8):
    above pp_prefill_admission_floor, so each stage has a chunk to run while earlier requests wait for their transfer.
    A model whose prefill computes ahead (``computes_ahead``, catalog.ModelSpec.prefill_compute_ahead) gets no default:
    its launch states the bound, which sizes its pool (serving_policy._prefill_compute_ahead). Returns the value set, or
    None (not a pipelined PD prefill, a model that computes ahead, or the launch sets it)."""
    pp_size = int(getattr(server_args, "pp_size", 1) or 1)
    if pp_size <= 1 or str(getattr(server_args, "disaggregation_mode", "")) != "prefill" or computes_ahead:
        return None
    if getattr(server_args, "max_running_requests", None) is not None:
        return None
    server_args.max_running_requests = 4 * pp_size
    logger.info("[neuron] pipelined PD prefill: --max-running-requests %d (4 x --pp-size %d)",
                server_args.max_running_requests, pp_size)
    return server_args.max_running_requests


def pp_prefill_admission_floor(pp_size: int) -> int:
    """The requests a pipelined PD prefill holds at once with its stages back to back: pp_size + 2 scheduled chunks not
    yet processed, plus up to pp_size + 1 transferred requests whose release consensus is still going round the ring. A
    request also keeps its slot while its KV goes to the decode, so admit well above this."""
    return 2 * int(pp_size) + 3


def connect_pp_ring(scheduler) -> None:
    """Connect the gloo pairs the PP loop sends on, before its first send.

    Under TORCH_GLOO_LAZY_INIT=1 a pair connects at first use and a first send waits for the peer to use the pair too,
    so a stage that sends its proxy before the objects its successor waits for would deadlock on the first batch. One
    receive from the previous stage and one send to the next, on the pp group's gloo groups and (for the ranks that
    carry stage objects) the world group's pair, connects them all up front. Without lazy init this is just one round of
    the ring."""
    import torch
    import torch.distributed as dist

    from sglang_neuron import sglang_tree

    pp, ps = scheduler.pp_group, sglang_tree.parallel_of(scheduler)
    if pp.world_size <= 1:
        return
    prev, nxt = pp.ranks[(pp.rank_in_group - 1) % pp.world_size], pp.ranks[(pp.rank_in_group + 1) % pp.world_size]
    groups = {id(g): g for g in (pp.cpu_group, pp.device_group) if g is not None and dist.get_backend(g) == "gloo"}
    pairs = [(group, prev, nxt) for group in groups.values()]
    if ps.attn_tp_rank == 0 and ps.attn_cp_rank == 0:
        offset = ps.attn_dp_rank * ps.attn_cp_size * ps.attn_tp_size
        pairs.append((scheduler.world_group.cpu_group, ((ps.pp_rank - 1) % ps.pp_size) * ps.tp_size + offset,
                      ((ps.pp_rank + 1) % ps.pp_size) * ps.tp_size + offset))
    for group, src, dst in pairs:
        works = [dist.irecv(torch.zeros(1), src=src, group=group), dist.isend(torch.zeros(1), dst=dst, group=group)]
        for work in works:
            work.wait()
