# SPDX-License-Identifier: Apache-2.0
"""The host-path mechanisms a launch runs, resolved once from its role and server arguments.

Each mechanism here takes host work off the device's critical path without changing a token. None is a user switch:
a mechanism runs wherever its prerequisites hold. When they do not, a DP-attention decode is refused with the missing
conditions named, because its fallback is a per-step exchange that costs every step. Elsewhere the launch logs the
missing conditions and SGLang's own path runs. The policy also records the PP and DCP sizes, the radix tree choice and
whether a pipelined PD prefill computes ahead.

Resolution happens in the launching process, the only one that has the server arguments before it starts a scheduler
(platform.apply_server_args_defaults). The result is exported as SGLANG_NEURON_SERVING_POLICY, which every SGLang
subprocess inherits, because plugin registration runs in each process before that process sees its arguments.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os

logger = logging.getLogger(__name__)

ENV_KEY = "SGLANG_NEURON_SERVING_POLICY"

# The plugin's HBM scratchpad page, in MiB, for a role whose model names none (catalog.ModelSpec.scratchpad_page_mb).
# The compiler's --hbm-scratchpad-page-size (graph_runner.COMPILER_ARGS) and the runtime's NEURON_SCRATCHPAD_PAGE_SIZE
# must match, except under LNC2 contiguous scratchpad, where the runtime ignores the page size.
SCRATCHPAD_PAGE_MB = 3584


@dataclasses.dataclass(frozen=True)
class ServingPolicy:
    """The host-path mechanisms of one launch."""

    role: str  # "aggregated" | "prefill" | "decode"
    # SGLang's overlap schedule with the device token relay
    async_execution: bool
    # model_executor.program_fence: an aggregated server's ranks meet whenever their next model execution switches
    # between the prefill and decode programs. A runtime with more than one compiled shape fences in any role
    # (graph_runner.build_compiled).
    program_switch_fence: bool = False
    # the DP identity's fence rendezvous of every rank runs through the node's shared memory
    dp_rendezvous_shm: bool = False
    # the disaggregation poll's reductions over a node-local group run through shared memory
    node_local_poll_shm: bool = False
    # the scheduler hands its TP group the requests over SGLang's shared-memory message queues instead of a gloo
    # broadcast (distributed.request_broadcast)
    request_broadcast_shm: bool = False
    # managers.dp_identity: on a DP-attention decode the per-step MLP-sync result is a known constant, so every rank
    # skips that exchange and always launches. Control messages are delivered at an iteration all ranks share.
    dp_identity: bool = False
    # sglang_patches.detokenizer_sender: attention-TP rank 1 of each DP group streams the group's outputs, so the DP
    # leader only keeps the request bookkeeping
    detokenizer_sender_rank1: bool = False
    # a giga-kernel decode (giga_decode_layout): the decode runs every layer in one kernel call per step over
    # the node's 128 physical NeuronCores, one rank each (NEURON_LOGICAL_NC_CONFIG=1, platform.set_logical_nc_config)
    giga_decode: bool = False
    # the HBM scratchpad page the graphs compile with and the runtime bills in (SCRATCHPAD_PAGE_MB, or the model's for
    # the role, catalog.ModelSpec.scratchpad_page_mb)
    scratchpad_page_mb: int = SCRATCHPAD_PAGE_MB
    # sglang_patches.pd_decode_result_lag: the decode processes a step's result only after launching the in-flight bound
    # of later steps, so a rank whose host runs late keeps its device fed instead of holding every rank at the giga
    # kernel's barrier; on the giga-kernel decode only, the row it is measured on
    decode_result_lag: bool = False
    # pipeline stages, 1 without PP (resolve says how each role runs them)
    pp_size: int = 1
    # ranks per decode-context-parallel group, 1 without DCP (SGLang builds the groups, sglang_patches.dcp_groups)
    dcp_size: int = 1
    # --radix-cache-backend neuron-chunk: the prefix cache keys whole prefill chunks (sglang_patches.chunk_keyed_tree).
    # Kept here because every process registers the backend before it sees the server arguments.
    chunk_keyed_tree: bool = False
    # --radix-cache-backend kimi-prompt: the Kimi-K3 prompt cache on the PD prefill (sglang_patches.kimi_prompt), END
    # entries at any token. Kept here because every process registers the backend before it sees the server arguments.
    kimi_prompt_cache: bool = False
    # backend.hbm_tail_guard: the last MiB of the runtime's small-allocation arena, at the top of the HBM aperture, is
    # held from the runtime start for the life of the process, so no NEFF instruction stream ends at the engines'
    # prefetch bound (the NRT status 1200 "Stuck in collectives op" workaround; opt-in, SGLANG_NEURON_HBM_TAIL_GUARD=1)
    hbm_tail_guard: bool = False
    # sglang_patches.pp_prefill_compute_ahead: a pipelined PD prefill computes a request before its decode grants the
    # seat, parks it with its KV and pushes the whole KV once the decode publishes the destination, every stage deciding
    # such a request at the bootstrap queue's consensus point (on a pipelined PD prefill of a model that declares it,
    # _prefill_compute_ahead; scheduling only)
    prefill_compute_ahead: bool = False
    # the name of the model-declared PD decode layout the launch runs (declared_decode_layout), "" for none
    decode_layout: str = ""
    # that layout's Neuron runtime settings, name -> value, which every rank starts with in place of its role's
    # defaults (runtime_init.runtime_defaults); empty without a declared layout
    decode_runtime_env: dict = dataclasses.field(default_factory=dict)
    # the nodes the server spans (--nnodes): above 1 only in a declared decode layout
    nnodes: int = 1
    # every rank's NEURON_LOGICAL_NC_CONFIG (platform.set_logical_nc_config): 1 on the giga-kernel decode, the declared
    # decode layout's degree, else the model's (catalog.ModelSpec.logical_nc)
    logical_nc: int = 2
    # the --speculative-algorithm the model's own speculative worker serves (catalog.ModelSpec.speculative_algorithms,
    # model_executor.spec_worker), "" without speculative decoding
    speculative_algorithm: str = ""
    # a PD role on SGLang's fake transfer backend, which moves no KV and builds no NIXL agent: a prefill without a
    # decode peer (sglang_patches.pd_fake_prefill), or a decode-only benchmark of a declared decode layout
    fake_kv_transfer: bool = False
    # the chunk-keyed tree also keeps the model's recurrent-state checkpoints (ModelSpec.chunk_state_checkpoints), so
    # its factory accepts SGLang's hybrid SSM pools (mem_cache.chunk_keyed_tree)
    chunk_state_checkpoints: bool = False
    # the launch serves the model's whole-model megakernel (catalog.ModelSpec.megakernel with --enable-nvidia-megakernel):
    # every role runs the giga runtime defaults (runtime_init.runtime_defaults), and under async execution its decode
    # takes each step's ids on the host from the previous step's lagged readback (async_execution's host-ids mode)
    megakernel: bool = False

    @property
    def pp_token_relay(self) -> bool:
        """A pipelined PD decode: the last stage sends each forward's sampled tokens to the first over the PP seam (kind
        "tk"), whose next graph builds its input ids from them, so no host waits for sampling."""
        return self.pp_size > 1 and self.role == "decode"

    @property
    def decode_radix_prefill_chunks(self) -> bool:
        """A PD decode on the chunk-keyed tree (which needs --disaggregation-decode-enable-radix-cache): the tree reuses
        only KV its prefills computed for whole chunks of the decode's own per-rank --chunked-prefill-size, and the
        decode pairs only with a prefill whose chunk divides it (sglang_patches.decode_radix_prefill_chunks)."""
        return self.role == "decode" and self.chunk_keyed_tree

    @property
    def overlap_schedule(self) -> bool:
        """The overlap schedule every scheduler must end with (runtime_init.check_overlap_schedule): on with async
        execution, except on a pipelined PD prefill, where SGLang turns it off for PP after this policy resolves and
        the plugin's async last stage takes its place."""
        return self.async_execution and not (self.pp_size > 1 and self.role == "prefill")

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "ServingPolicy":
        return cls(**json.loads(text))


def launch_role(server_args) -> str:
    """The serving role of a launch: "prefill" or "decode" for a PD server, "aggregated" for any other."""
    mode = str(getattr(server_args, "disaggregation_mode", "null") or "null")
    return mode if mode in ("prefill", "decode") else "aggregated"


def _width(server_args, name: str) -> int:
    return int(getattr(server_args, name, 1) or 1)


def dp_identity_blockers(server_args, async_execution: bool) -> list:
    """The DP identity's conditions this launch misses; empty when the identity can run.

    The identity replaces a per-step exchange with its constant result. That result holds only for one launch shape,
    with every rank launching every iteration. Its control fence also needs every DP leader to receive control
    messages directly and every rank to poll every iteration (dp_identity.fence_checks). dp_identity.proof_checks
    asserts the runtime side of the same conditions at the first step."""
    from sglang_neuron.managers import dp_identity

    checks = {
        **dp_identity.fence_checks(server_args),
        "no two-batch overlap": not getattr(server_args, "enable_two_batch_overlap", False),
        "async execution (no --disable-overlap-schedule)": async_execution,
        "--max-running-requests": getattr(server_args, "max_running_requests", None) is not None,
        "one decode batch size (no decode size below the per-DP batch in --cuda-graph-bs-decode or "
        "--cuda-graph-config)": _one_decode_batch_size(server_args),
    }
    return [name for name, ok in checks.items() if not ok]


def runs_dp_identity(server_args, role: str, dp_steps_from_exchange: bool = False) -> bool:
    """Whether the launch is a role the DP identity serves: a decode with DP attention over more than one DP rank, of a
    model whose steps are not planned from the per-step exchange the identity skips (a whole-model megakernel's
    MegakernelServing.dp_steps_from_exchange: its ranks derive each step's kernel calls from every replica's
    counts)."""
    from sglang_neuron import sglang_tree

    return role == "decode" and sglang_tree.attention_dp_size(server_args) > 1 and not dp_steps_from_exchange


def listed_decode_batch_sizes(server_args) -> tuple:
    """The decode batch sizes the launch lists, by --cuda-graph-bs-decode or as decode bs in --cuda-graph-config (an
    empty tuple when it lists none)."""
    listed = getattr(server_args, "cuda_graph_bs_decode", None)
    if listed is None:
        from sglang.srt.model_executor.cuda_graph_config import Phase

        if (Phase.DECODE, "bs") in getattr(server_args, "_cuda_graph_config_locked", ()):
            listed = server_args.cuda_graph_config.decode.bs
    return tuple(int(b) for b in listed or ())


def _one_decode_batch_size(server_args) -> bool:
    """Whether every listed decode batch size is at least the per-DP batch (sglang_tree.scheduler_batch), so the
    decode runs at one size, the batch. Without --max-running-requests there is no per-DP batch to compare;
    dp_identity_blockers names that condition separately."""
    from sglang_neuron import sglang_tree

    batch = sglang_tree.scheduler_batch(server_args)
    return batch is None or all(b >= batch for b in listed_decode_batch_sizes(server_args))


def detokenizer_sender_blockers(server_args, dp_identity: bool, environ=os.environ) -> list:
    """The conditions this launch misses for streaming outputs from attention-TP rank 1; empty when none are missing.

    Rank 1 can stand in for the DP leader only if every rank of the DP group holds the same requests in the same state,
    which the DP identity's decode guarantees. The stream also must not carry anything only the leader has: per-request
    time stats, tracing, the Rust server, or metrics recorded by the leader alone. detokenizer_sender.runtime_blockers
    checks the runtime side when the scheduler starts."""
    from sglang_neuron import sglang_tree

    groups = sglang_tree.attention_dp_size(server_args)
    tp = int(getattr(server_args, "tp_size", 1) or 1)
    cp = int(getattr(server_args, "attn_cp_size", 1) or 1)
    checks = {
        "the DP identity": dp_identity,
        "an attention-TP width of at least 2": tp // max(groups, 1) // max(cp, 1) >= 2,
        "no --enable-request-time-stats-logging": not getattr(server_args, "enable_request_time_stats_logging", False),
        "no --enable-metrics without --enable-metrics-for-all-schedulers":
            not getattr(server_args, "enable_metrics", False)
            or bool(getattr(server_args, "enable_metrics_for_all_schedulers", False)),
        "no --enable-trace": not getattr(server_args, "enable_trace", False),
        "SGLANG_RUST_SERVER off": not _sglang_bool(environ.get("SGLANG_RUST_SERVER")),
        # rank 1 pushes to its group's detokenizer worker over the launching node's IPC endpoints: a multi-node launch
        # with several workers has no such endpoint on rank 1's node, so the leader keeps the push
        "one node (nnodes 1) with --detokenizer-worker-num above 1":
            int(getattr(server_args, "detokenizer_worker_num", 1) or 1) <= 1
            or int(getattr(server_args, "nnodes", 1) or 1) == 1,
    }
    return [name for name, ok in checks.items() if not ok]


def refuse_unserved_layouts(server_args, role: str, prefill_holds_prior: bool,
                            aggregated_dp_attention: bool = False) -> None:
    """Refuse the DP-attention layouts the compiled graphs cannot serve, naming why.

    The DP ranks all run the same compiled graph each step, sized from the token counts SGLang gathers across the
    group, so the launch cannot skip that gather (SGLANG_SCHEDULER_SKIP_ALL_GATHER). An aggregated server would mix
    one rank's prefill with another's decode in the same step, except on a model whose ranks plan every step's kernel
    calls from those counts alike (``aggregated_dp_attention``: a whole-model megakernel's
    MegakernelServing.dp_steps_from_exchange), where a prefill replica and a decode replica run the same calls. A
    prefill whose window also holds the request's cached prior (the tokens already in its KV cache) sizes its graph by
    its own request's prior, which the group's token counts do not carry."""
    from sglang_neuron import sglang_tree

    if sglang_tree.attention_dp_size(server_args) == 1:
        return
    from sglang.srt.environ import envs as sglang_envs

    if sglang_envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.get():
        raise ValueError("SGLANG_SCHEDULER_SKIP_ALL_GATHER with --enable-dp-attention: the DP ranks agree on one graph a "
                         "step from the token counts SGLang gathers, which the skipped all-gather leaves unset; launch "
                         "without it")
    if role == "aggregated" and not aggregated_dp_attention:
        raise ValueError("--enable-dp-attention on an aggregated server: its DP ranks would run prefill and decode "
                         "graphs in the same step; serve DP attention disaggregated (--disaggregation-mode prefill or "
                         "decode)")
    if prefill_holds_prior and role == "prefill":
        raise ValueError("--enable-dp-attention on a prefill of a model whose prefill window holds the request's "
                         "cached prior: each DP rank would size its graph by its own request's prior; launch the "
                         "prefill without --enable-dp-attention")


# the giga-kernel decode's world: every physical NeuronCore of the node is a rank, in 16 DP-attention groups of 8, with
# the experts over all 128 (the kernel's mesh)
GIGA_DECODE_RANKS = 128
GIGA_DECODE_DP = 16


def giga_decode_layout(server_args, role: str, model_has_giga_decode: bool, dp: int = GIGA_DECODE_DP) -> bool:
    """Whether the launch is a giga-kernel decode: the decode of a model that has one (catalog.ModelSpec.giga_decode)
    over the kernel's world, ``--tp-size 128 --dp-size 16 --enable-dp-attention --ep-size 128`` (or ``--attn-dp-size
    16``, SGLang's current spelling), in one server replica; a kernel that runs its own attention DP inside
    (``dp`` 1, catalog.ModelSpec.giga_decode_dp) takes ``--tp-size 128`` without DP attention. 128 ranks fit a node
    only at one physical core per logical core, and the kernel is traced for exactly that mesh, so such a model at 128
    ranks in any other layout is refused, naming what differs."""
    from sglang_neuron import sglang_tree

    tp = int(getattr(server_args, "tp_size", 1) or 1)
    if not model_has_giga_decode or tp != GIGA_DECODE_RANKS:
        return False
    replicas, attention_dp = sglang_tree.dp_widths(server_args)
    batch = sglang_tree.scheduler_batch(server_args)
    if dp == 1:
        dp_need = {"no DP attention, in one server replica": attention_dp == 1 and replicas == 1}
    else:
        attention_tp = GIGA_DECODE_RANKS // dp
        dp_need = {
            f"--dp-size {dp} --enable-dp-attention (or --attn-dp-size {dp}) in one server replica":
                attention_dp == dp and replicas == 1,
            # the kernel shards each DP group's decode rows over the group's attention ranks
            f"--max-running-requests giving each DP group a batch that is a multiple of its {attention_tp} attention "
            f"ranks (a multiple of {GIGA_DECODE_RANKS} gives one)": batch is None or batch % attention_tp == 0,
        }
    needs = {
        "--disaggregation-mode decode": role == "decode",
        **dp_need,
        f"--ep-size {GIGA_DECODE_RANKS}": int(getattr(server_args, "ep_size", 1) or 1) == GIGA_DECODE_RANKS,
    }
    missing = [flag for flag, met in needs.items() if not met]
    if missing:
        raise ValueError(f"--tp-size {GIGA_DECODE_RANKS}: the one layout served on {GIGA_DECODE_RANKS} ranks is the "
                         f"giga-kernel decode, which also needs {', '.join(missing)}")
    return True


HBM_TAIL_GUARD_ENV = "SGLANG_NEURON_HBM_TAIL_GUARD"


def _hbm_tail_guard(environ=os.environ) -> bool:
    """SGLANG_NEURON_HBM_TAIL_GUARD=1 (opt-in): hold the top of the HBM small-allocation arena from the runtime start
    (backend.hbm_tail_guard), the NRT status 1200 workaround. Unset: False. Any other value, and the opt-in off the
    device (CPU mode, where no arena exists), are refused."""
    value = environ.get(HBM_TAIL_GUARD_ENV)
    if value is None:
        return False
    if value != "1":
        raise ValueError(f"{HBM_TAIL_GUARD_ENV}={value!r}: set it to 1 or leave it unset")
    from sglang_neuron.backend import envs as backend_envs

    if backend_envs.SGLANG_NEURON_CPU_MODE:
        raise ValueError(f"{HBM_TAIL_GUARD_ENV}=1 cannot apply: SGLANG_NEURON_CPU_MODE=1 runs no Neuron device, so "
                         "there is no HBM arena to guard")
    return True


def _prefill_compute_ahead(server_args, role: str, pp_size: int, model_computes_ahead: bool = False) -> bool:
    """Compute ahead on a pipelined PD prefill (sglang_patches.pp_prefill_compute_ahead) for a model whose catalog
    entry declares it (ModelSpec.prefill_compute_ahead); off on every other launch. It changes only the scheduling.

    A request computed ahead holds its pages, locked in the radix tree, until its seat is granted, so the pool must hold
    the full-context prompts of every request that may be pending at once (--max-running-requests less the reserve)
    plus one seat-granted request; otherwise parked requests could fill the pool while the decode's free seats wait for
    requests that need pages. The launch must therefore state --max-running-requests, --max-total-tokens and
    --context-length, and one that misses them or does not fit runs without compute ahead and logs why. A pipelined
    server never has a hierarchical cache (platform.refuse_unserved)."""
    if not model_computes_ahead or role != "prefill" or pp_size <= 1:
        return False

    def without(reason: str) -> bool:
        logger.warning("[neuron] this pipelined PD prefill runs without compute ahead (sglang_patches."
                       "pp_prefill_compute_ahead): %s", reason)
        return False

    from sglang_neuron.distributed.pipeline import pp_prefill_admission_floor

    sizes = {flag: getattr(server_args, field, None) for flag, field in (
        ("--max-running-requests", "max_running_requests"), ("--max-total-tokens", "max_total_tokens"),
        ("--context-length", "context_length"))}
    missing = [flag for flag, size in sizes.items() if size is None]
    if missing:
        return without(f"without {', '.join(missing)}: a request computed ahead holds its pages until its seat is "
                       "granted, so the pool must hold every pending request's prompt and one seat-granted request's "
                       "at the full context, which the policy checks from these; launch with --max-running-requests, "
                       "--max-total-tokens and --context-length")
    # pipeline.pp_launch_problems refuses a prefill without rows above the floor, so ``running - floor`` is positive
    running, total, context = (int(size) for size in sizes.values())
    floor = pp_prefill_admission_floor(pp_size)
    page = int(getattr(server_args, "page_size", None) or 1)
    prompt = -(-context // page) * page  # a full-context prompt, in whole pages
    held = running - floor + 1  # every request that may be pending at once, and one seat-granted request
    if held * prompt > total:
        most = floor - 1 + total // prompt
        return without(f"with --max-running-requests {running} at --pp-size {pp_size}: up to {running - floor} "
                       f"requests computed ahead hold their pages until their seats are granted, and a seat-granted "
                       f"request then needs its own, {held} prompts of --context-length {context} in {page}-token "
                       f"pages, {held * prompt} tokens, above --max-total-tokens {total}; parked requests could fill "
                       f"the pool while the decode's free seats wait for requests that need pages. Launch with "
                       f"--max-running-requests {most} or less, or --max-total-tokens {held * prompt} or more")
    return True




def megakernel_async_blockers(server_args, megakernel) -> list:
    """The conditions a launch of a whole-model megakernel (``megakernel``: the model's catalog.MegakernelServing)
    misses for async execution; empty when its steps can run under SGLang's overlap schedule.

    The megakernel host runs a step's kernel calls without a read in between and returns one pending result, and the
    next decode takes its ids on the host from that result's readback (async_execution's host-ids mode), which the
    model's programs must serve. A speculative worker (NEXTN/MTP) verifies each step's draft on the host before the
    next step, and SGLang's overlap schedule runs no speculative worker on this platform."""
    algorithm = getattr(server_args, "speculative_algorithm", None)
    checks = {
        "programs that serve async steps (catalog.MegakernelServing.async_steps)": bool(megakernel.async_steps),
        "no --speculative-algorithm (the speculative worker verifies each step on the host)":
            algorithm in (None, "", "NONE", "none"),
    }
    return [name for name, ok in checks.items() if not ok]


def declared_decode_layout(server_args, role: str, layouts=()):
    """The model-declared PD decode layout (capability.DecodeLaunchLayout, catalog.ModelSpec.decode_layouts) the launch
    runs, or None. A layout is selected by its node count and --tp-size; a launch with both is refused naming every
    other flag of the layout it misses. A launch over more than one node selects a layout or is refused."""
    from sglang_neuron import sglang_tree

    nnodes, tp = _width(server_args, "nnodes"), _width(server_args, "tp_size")
    layout = next((each for each in layouts if (each.nnodes, each.tp_size) == (nnodes, tp)), None)
    if layout is None:
        if nnodes > 1:
            served = "; ".join(each.flags() for each in layouts) or "none"
            raise ValueError(f"--nnodes {nnodes} --tp-size {tp}: a server spans several nodes only in a decode layout "
                             f"its model declares ({served})")
        return None
    replicas, attention_dp = sglang_tree.dp_widths(server_args)
    needs = {
        "--disaggregation-mode decode": role == "decode",
        f"--dp-size {layout.dp_size} --enable-dp-attention (or --attn-dp-size {layout.dp_size}) in one server replica":
            attention_dp == layout.dp_size and replicas == 1,
        f"--ep-size {layout.ep_size}": _width(server_args, "ep_size") == layout.ep_size,
        f"--dcp-size {layout.dcp_size}": _width(server_args, "dcp_size") == layout.dcp_size,
        "--pp-size 1": _width(server_args, "pp_size") == 1,
        # every node's ranks start their collectives from node 0's address (runtime_init.root_comm_id)
        "--dist-init-addr <node 0's address>:<port>": nnodes == 1 or bool(getattr(server_args, "dist_init_addr", None)),
    }
    missing = [flag for flag, met in needs.items() if not met]
    if missing:
        raise ValueError(f"--nnodes {nnodes} --tp-size {tp} is the {layout.name} ({layout.flags()}), which also needs "
                         f"{', '.join(missing)}")
    return layout


def speculative_algorithm(server_args, served=()) -> str:
    """The launch's --speculative-algorithm in upper case if the model's own worker serves it (``served``,
    catalog.ModelSpec.speculative_algorithms), else ""; platform.refuse_unserved refuses any other algorithm."""
    algorithm = str(getattr(server_args, "speculative_algorithm", None) or "").upper()
    return algorithm if algorithm in served else ""


def resolve(server_args, prefill_holds_prior: bool = False, model_has_giga_decode: bool = False,
            giga_decode_dp: int = GIGA_DECODE_DP, scratchpad_page_mb: tuple = (),
            model_computes_ahead: bool = False, decode_layouts=(), speculative_algorithms=(),
            chunk_state_checkpoints: bool = False, logical_nc_config: int = 2, megakernel=None) -> ServingPolicy:
    """Resolve the launch's mechanisms.

    * Async execution is SGLang's overlap schedule, on unless the launch passes --disable-overlap-schedule or is a
      pipelined aggregated server.
    * A disaggregated decode runs the shared-memory forms of the request broadcast, the poll reductions and the DP
      rendezvous; their gloo forms delay the next launch behind a short decode step. A prefill server keeps the gloo
      forms, since its DP replicas share the node-local ranks.
    * Every decode with DP attention runs the DP identity, and is refused if it misses the identity's conditions. On
      top of the identity, attention-TP rank 1 streams the outputs when its own conditions hold.
    * An aggregated server fences its program switches, and so does any role whose runtime warms more than one
      shape.
    * A model's decode over its giga-kernel layout is the giga-kernel decode (giga_decode_layout).
    * The scratchpad page is the model's for the role (``scratchpad_page_mb``, ((role, MiB), ...) as
      catalog.ModelSpec.scratchpad_page_mb), else the plugin's.
    * A pipelined PD role runs async: a prefill in the overlap order with an async last stage and SGLang's overlap
      schedule off (sglang_patches.pp_prefill_overlap_loop, pp_last_stage_async), a decode with its sampled tokens
      relayed from the last stage to the first (pp_token_relay). A pipelined aggregated server runs synchronously on
      SGLang's own PP loop, which waits on the host for each micro-batch's outputs. validate_pp_launch
      (distributed.pipeline) refuses any other pipelined launch before the policy is exported.
    * A launch of a model's whole-model megakernel (``megakernel``: its catalog.MegakernelServing, given when the launch
      passes --enable-nvidia-megakernel) runs the giga runtime defaults in every role. A model whose ranks plan each
      step from the per-step DP-attention exchange (dp_steps_from_exchange) serves DP attention on an aggregated server
      and keeps that exchange on a decode instead of the DP identity. It runs async execution in the host-ids mode
      when megakernel_async_blockers is empty, and a launch that asks for async execution otherwise is refused.
    * --radix-cache-backend neuron-chunk selects the chunk-keyed prefix cache, which
      sglang_patches.chunk_keyed_tree.validate_launch checked first; it keeps recurrent-state checkpoints for a model
      that declares them (``chunk_state_checkpoints``). --radix-cache-backend kimi-prompt selects the Kimi-K3 prompt
      cache (``kimi_prompt_cache``, sglang_patches.kimi_prompt.validate_launch checked first).
    * SGLANG_NEURON_HBM_TAIL_GUARD=1 (opt-in, on the device only) holds the top of the HBM small-allocation arena from
      the runtime start (backend.hbm_tail_guard).
    * Compute ahead on a pipelined PD prefill of a model whose catalog entry declares it (``model_computes_ahead``,
      _prefill_compute_ahead).
    * A PD decode in one of its model's declared layouts (``decode_layouts``, declared_decode_layout) runs over the
      layout's nodes at its logical NC degree; a launch over several nodes in no declared layout is refused. Every
      other launch runs the model's degree (``logical_nc_config``), the giga-kernel decode 1.
    * A --speculative-algorithm the model's own worker serves (``speculative_algorithms``) is recorded for the
      platform's speculative-worker seam and the DP identity.
    * A prefill or a declared-layout decode on SGLang's fake transfer backend builds no NIXL agent
      (``fake_kv_transfer``)."""
    from sglang_neuron import sglang_tree
    from sglang_neuron.sglang_patches import chunk_keyed_tree, kimi_prompt

    role = launch_role(server_args)
    steps_from_exchange = megakernel is not None and bool(megakernel.dp_steps_from_exchange)
    refuse_unserved_layouts(server_args, role, prefill_holds_prior, aggregated_dp_attention=steps_from_exchange)
    giga_decode = giga_decode_layout(server_args, role, model_has_giga_decode, giga_decode_dp)
    layout = declared_decode_layout(server_args, role, decode_layouts)
    decode = role == "decode"
    pp_size = int(getattr(server_args, "pp_size", 1) or 1)
    async_execution = (not getattr(server_args, "disable_overlap_schedule", False)
                       and not (pp_size > 1 and role == "aggregated"))
    if megakernel is not None and async_execution:
        blockers = megakernel_async_blockers(server_args, megakernel)
        if blockers:
            raise ValueError(f"async execution (SGLang's overlap schedule) of this model's whole-model megakernel needs "
                             f"{'; '.join(blockers)}; launch with --disable-overlap-schedule")
    dp_identity = runs_dp_identity(server_args, role, steps_from_exchange)
    if dp_identity:
        blockers = dp_identity_blockers(server_args, async_execution)
        if blockers:
            raise ValueError(f"a decode with DP attention runs its ranks in lockstep without a per-step exchange, "
                             f"which needs {'; '.join(blockers)}")
    sender_blockers = detokenizer_sender_blockers(server_args, dp_identity)
    if dp_identity and sender_blockers:
        logger.warning("[neuron] the DP leader streams the decode's outputs (not met: %s)", "; ".join(sender_blockers))
    page = dict(scratchpad_page_mb).get(role)
    return ServingPolicy(
        role=role,
        async_execution=async_execution,
        program_switch_fence=role == "aggregated",
        dp_rendezvous_shm=decode,
        node_local_poll_shm=decode,
        # the shared-memory broadcast hands the whole TP group the leader's requests, which under DP attention only the
        # DP identity's fence may do (each DP group has its own requests)
        request_broadcast_shm=decode and (dp_identity or sglang_tree.attention_dp_size(server_args) == 1),
        dp_identity=dp_identity,
        detokenizer_sender_rank1=not sender_blockers,
        giga_decode=giga_decode,
        **({} if page is None else {"scratchpad_page_mb": int(page)}),
        decode_result_lag=giga_decode,
        pp_size=pp_size,
        dcp_size=int(getattr(server_args, "dcp_size", 1) or 1),
        chunk_keyed_tree=chunk_keyed_tree.selected(server_args),
        kimi_prompt_cache=kimi_prompt.selected(server_args),
        hbm_tail_guard=_hbm_tail_guard(),
        prefill_compute_ahead=_prefill_compute_ahead(server_args, role, pp_size, model_computes_ahead),
        decode_layout=layout.name if layout is not None else "",
        decode_runtime_env=dict(layout.runtime_env) if layout is not None else {},
        nnodes=_width(server_args, "nnodes"),
        logical_nc=1 if giga_decode else layout.logical_nc if layout is not None else logical_nc_config,
        speculative_algorithm=speculative_algorithm(server_args, speculative_algorithms),
        fake_kv_transfer=(getattr(server_args, "disaggregation_transfer_backend", None) == "fake"
                          and (role == "prefill" or layout is not None)),
        chunk_state_checkpoints=chunk_keyed_tree.selected(server_args) and chunk_state_checkpoints,
        megakernel=megakernel is not None,
    )


def _sglang_bool(value: str | None) -> bool | None:
    """An environment value as SGLang's EnvBool reads it; None when unset or not a boolean."""
    value = (value or "").lower()
    return True if value in ("true", "1", "yes", "y") else False if value in ("false", "0", "no", "n") else None


def set_launch_setting(name: str, value: str, reason: str, environ=os.environ) -> None:
    """Set an environment variable every scheduler reads, here in the launching process so each scheduler inherits
    it. An unset variable takes the value and one already set to it is accepted; any other value is refused with the
    reason. Booleans compare as SGLang's EnvBool reads them, so "1" matches "true"."""
    have = environ.get(name)
    if have is not None:
        want_bool, have_bool = _sglang_bool(value), _sglang_bool(have)
        same = have_bool is want_bool if want_bool is not None else have.strip() == value
        if not same:
            raise ValueError(f"{name}={have}: {reason}; the platform sets it to {value} for this launch, so remove it "
                             "from the launch environment")
    environ[name] = value


_CURRENT: ServingPolicy | None = None


def export(policy: ServingPolicy) -> None:
    """Publish the resolved policy to this process and to every subprocess it starts."""
    global _CURRENT
    os.environ[ENV_KEY] = policy.to_json()
    _CURRENT = policy
    logger.info("[neuron] serving policy: %s", policy.to_json())


def current() -> ServingPolicy:
    """The policy the launching process resolved, read once per process. A process the platform did not launch (a
    unit test, an offline tool) runs none of the mechanisms."""
    global _CURRENT
    if _CURRENT is None:
        text = os.environ.get(ENV_KEY)
        _CURRENT = ServingPolicy.from_json(text) if text else ServingPolicy(role="aggregated", async_execution=False)
    return _CURRENT
