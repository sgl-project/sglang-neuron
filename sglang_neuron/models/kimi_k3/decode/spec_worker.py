# SPDX-License-Identifier: Apache-2.0
"""The Kimi-K3 DSPARK speculative worker: SGLang's spec-v2 worker of the Kimi-K3 PD decode over the decode runtime's
verify and draft programs (``K3DecodeRuntime.step``).

SGLang keeps the DSPARK bookkeeping around the worker: ``DFlashDraftInputV2`` reserves the ``2 (k + 1)`` KV slots of
each request before a step and the PD join builds a request's first draft input from the prefill's token
(``build_dspark_disagg_draft_input``); the result processor commits ``accept_lens`` tokens of each request. One step of
the runtime verifies ``Q = k + 1`` rows of each request (the bonus token and ``k`` drafts), accepts on the device,
samples the next bonus token, and drafts the next step's ``k`` drafts; so the drafts never reach the host, and the
next step takes them, with each request's row 0 and accepted count, from the step's device outputs.

Per DECODE batch of the replica (``K3StepDriver.step``), in batch order, then pad rows up to the served batch ``B``:

- row 0 on the host: a request whose last step SGLang has not processed yet (one step in flight under the overlap
  schedule) has committed length ``L`` = the true row 0 of that step, so its next row 0 is in ``[L + 1, L + Q]`` and
  the host gives ``L + Q``; the program computes the true row 0 on the device and the host window ``[N - k, N + k]``
  covers it. Every other request gives its committed length, exactly;
- carried and fresh rows (``DecodeSpecHostState.plan_step``): a request of the last step takes its state from that
  step's device outputs; any other request is fresh: a new request (just joined from the prefill) verifies its bonus
  token (the prefill's sample) with no draft, at its committed length, with the KDA state the prefill sent; an older
  fresh request takes its harvested bonus and verifies no draft;
- the lowered tables (lowering.py) from SGLang's ``req_to_token``, the KDA state pages of the request's state slot
  and its rider ring (sgl_pools.py), the host draws and sampling params, and the last step's device outputs;
- a new request's MLA blocks after its last transferred stripe are zeroed first (``tail_zero_blocks``).

The step's results reach the host when SGLang processes the step (``K3PendingStep.fill_result``, through the
platform's deferred result copy): the tokens ``[n, Q]`` (the accepted drafts, then the new bonus), ``accept_lens``,
the new sequence lengths, and the host-state harvest that checks each request's row 0, accepted count, tokens and the
sampler's row flags (``spec_math.ROW_FLAG_*``: a NaN or +inf logit, no allowed token). A request that fails a check
gets none of the step's tokens and finishes with its error at that step's commit; the other requests keep running.

Draws: each request draws from its own generator, seeded with its request seed (``sampling.request_seed``: its
``sampling_seed``, else a hash of its id), so its draws do not depend on the other requests of its batch.

An IDLE batch (a replica with no request) runs the same programs on pad rows: the EP256 collectives span every
replica. The model's launch rules refuse at the tokenizer what the programs' sampler does not serve (launch
.validate_request: structured output, logprobs, hidden states, penalties, min-p, logit bias, custom logit
processors); a request that still reaches the decode with one (``unserved_reasons``) is aborted with the reason.

``SGLANG_NEURON_KIMI_K3_FORCED_AL`` (benchmark only, env_registry BENCHMARK) forces a mean acceptance length in
``[1, k + 1]``: each step takes forced accepted counts drawn for that mean, so the served program and the token
accounting run as in a real run; the answers are then not the target's, and the worker logs a warning.

``SGLANG_NEURON_KIMI_K3_STEP_LOG=1`` (debug only, env_registry DEBUG, off by default) logs one ``K3STEP`` line for each
harvested verify step, on the first DCP rank of each replica (``step_log_line``). The perf tools read the step period,
the busy slots and the accepted rows of each slot from these lines.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch

from sglang_neuron.models.kimi_k3.decode import host_state as hs
from sglang_neuron.models.kimi_k3.decode import lowering as lw
from sglang_neuron.models.kimi_k3.decode import sgl_pools as sp
from sglang_neuron.models.kimi_k3.decode import spec_math as sm
from sglang_neuron.models.kimi_k3.decode.programs import DecodeProgramConfig

logger = logging.getLogger(__name__)

FORCED_AL_ENV = "SGLANG_NEURON_KIMI_K3_FORCED_AL"
PAD_ID = 0
"""The id of a pad draft and of a pad row."""

LOWERED_INPUTS = ("input_ids", "mla_block_table", "mla_slot_window", "kda_state_table_0", "kda_state_table_1",
                  "kda_state_table_2", "spec_draft_rider_ring", "sampling_params")
"""The step inputs the worker builds beside the host state's ``spec_*`` inputs, under the programs' names
(``programs.VERIFY_INPUTS``): the request-major ids ``[B Q]``, the MLA host-page table ``[B, 128]`` int32
(``lowering.MlaTables.device``), the MLA slot window ``[B, 2k + 1]`` int64, the KDA state table ``[B, 1 + Q]`` int32 of
each state group, the rider ring kernel pages ``[B, 14]`` int32 and the sampler params ``[B Q, 4]`` fp32."""
CARRY_INPUTS = ("spec_prev_row0", "spec_prev_accepted", "spec_prev_next_ids", "spec_prev_q_probs", "spec_prev_q_ids")
"""The step inputs from the last step's device outputs (``programs.StepOutputs.carry_inputs``), zeros without a last
step."""


ROW_FLAG_REASONS = ((sm.ROW_FLAG_NAN, "a NaN or +inf logit"), (sm.ROW_FLAG_NO_TOKEN, "no allowed token"),
                    (sm.ROW_FLAG_LAST_ID, "a greedy token at the last id, the clamp target of a NaN row"))
"""The sampler's row flags (``StepOutputs.flags``, the OR over a request's rows) and what each one reports."""


def flag_reasons(flags: int) -> list[str]:
    """The reasons of a request's row flags (``ROW_FLAG_REASONS``); an unknown bit is named by its value."""
    reasons = [text for bit, text in ROW_FLAG_REASONS if flags & bit]
    unknown = flags & ~sum(bit for bit, _ in ROW_FLAG_REASONS)
    if unknown:
        reasons.append(f"flag bits {unknown:#x}")
    return reasons


@dataclass(frozen=True)
class RowInput:
    """One request of a step, as SGLang's batch holds it.

    Attributes:
        rid: the request id.
        pool_index: its ``req_to_token`` row.
        committed: its committed length (``req.kv.kv_committed_len``).
        allocated: its positions with a KV slot (``req.kv.kv_allocated_len``).
        slot: its state slot (SGLang's mamba slot, 1-based).
        bonus: the bonus token the draft input relays (used for a new request: the prefill's sample).
        temperature, top_p, top_k: its sampling params.
        seed: its request seed (``sampling.request_seed``), which seeds its draws; None draws from the replica's
            generator.
    """

    rid: str
    pool_index: int
    committed: int
    allocated: int
    slot: int
    bonus: int = PAD_ID
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    seed: Optional[int] = None


@dataclass
class HostStep:
    """A step's results on the host: per request (batch order) the tokens ``[n, Q]`` int64 (the accepted tokens, then
    -1), ``accept_lens`` ``[n]`` int32, the new committed lengths ``[n]`` int64; the requests that failed a check, whose
    rows hold no token (all -1, accepted 0, the length of before the step)."""

    tokens: torch.Tensor
    accept_lens: torch.Tensor
    new_seq_lens: torch.Tensor
    failures: dict[str, str] = field(default_factory=dict)


@dataclass
class StepHandle:
    """A submitted step: its host-state record (None for an idle step), its device outputs, the request ids of its
    rows and each request's host row 0."""

    record: Optional[hs.StepRecord]
    outputs: Any
    rids: list[str]
    row0: np.ndarray
    host: Optional[HostStep] = None
    # the step clock's timed events right before the verify launch and after the verify and the draft launch
    # (StepClock), or None
    marks: Optional[list] = None


STEP_LOG_ENV = "SGLANG_NEURON_KIMI_K3_STEP_LOG"
STEP_LOG_EMPTY = "-"
"""The three fields of a slot without a request in a ``K3STEP`` line."""
STEP_LOG_SKIPPED = "x"
"""The three fields of a slot whose request the harvest did not store (a failed check or a dropped request)."""


def step_log_enabled(environ: Mapping[str, str] = os.environ) -> bool:
    """Return True when ``SGLANG_NEURON_KIMI_K3_STEP_LOG`` is 1. Unset, empty or 0 gives False.

    Raises:
        ValueError: the variable has another value.

    Example:
        >>> step_log_enabled({"SGLANG_NEURON_KIMI_K3_STEP_LOG": "1"})
        True
    """
    value = environ.get(STEP_LOG_ENV, "0").strip()
    if value not in ("", "0", "1"):
        raise ValueError(f"{STEP_LOG_ENV} is {value!r}. Set it to 0 or 1.")
    return value == "1"


def step_log_line(replica: int, step: int, context: int, queries: int, slots: Sequence[tuple],
                  t: Optional[float] = None) -> str:
    """Return the ``K3STEP`` line of one harvested verify step. ``slots`` holds the ``(P, acc, rid)`` fields of each
    batch row: row 0, the accepted rows and the last 8 characters of the request id.

    The fields are ``t`` (epoch seconds), ``dp`` (the replica), ``step``, ``ctx`` (the context of the step in tokens),
    ``q`` (k + 1), ``b`` (the batch rows), then one comma list each for ``P``, ``acc`` and ``rid``.

    Example:
        >>> step_log_line(2, 7, 4096, 4, [("130", "3", "abcd1234"), ("-", "-", "-")], t=1.0)
        'K3STEP t=1.0000 dp=2 step=7 ctx=4096 q=4 b=2 P=130,- acc=3,- rid=abcd1234,-'
    """
    t = time.time() if t is None else t
    p_list, acc_list, rid_list = (",".join(column) for column in zip(*slots)) if slots else ("", "", "")
    return (f"K3STEP t={t:.4f} dp={replica} step={step} ctx={context} q={queries} b={len(slots)} P={p_list} "
            f"acc={acc_list} rid={rid_list}")


STEP_TIMES_ENV = "SGLANG_NEURON_KIMI_K3_STEP_TIMES"


class StepClock:
    """Host times of the worker's steps (a probe, ``SGLANG_NEURON_KIMI_K3_STEP_TIMES=N``: one log line per ``N``
    decode steps, on the first DCP rank of each replica). Per step, in ms: ``period`` (one decode step's entry to
    the next), ``between`` (SGLang's time between the worker's steps), ``rows`` (reading the batch), ``plan`` (host
    state, lowering, draws), ``upload``, ``submit`` (the runtime's step: the in-flight bound's wait and the two graph
    calls), ``read`` (the result read when SGLang processes the step).

    Example:
        >>> clock = StepClock(2, enabled=True)
        >>> clock.add("plan", 1.0); clock.add("plan", 3.0)
        >>> clock.means()["plan"]
        2.0
    """

    # ``plan_cpu`` / ``upload_cpu``: this thread's CPU time in plan and upload; a wall time well above it waits for
    # the GIL or the CPU (the result read runs on another thread)
    # ``up_wait`` / ``up_writes`` / ``up_fallback``: the input ring's wait for its set's last step, its raw writes
    # and its fallback uploads (``StepInputRing.last_times``)
    # ``dev_*``: device times from timed events right before the verify launch and after each graph launch (read
    # once the step's results are read): ``dev_period`` one draft's end to the next, ``dev_idle`` the last draft's
    # end to the event before the verify (device time lost to this rank's host), ``dev_verify`` the verify (with its
    # wait in the device barrier for the latest rank), ``dev_draft`` the draft
    # ``v_*`` / ``d_*``: inside the submit, the verify's and the draft's in-flight bound wait and graph call, wall
    # and this thread's CPU (``K3DecodeRuntime._launch``); ``read_cpu``: the result read's CPU on its thread
    NAMES = ("period", "between", "rows", "plan", "plan_cpu", "upload", "upload_cpu", "up_wait", "up_writes",
             "up_fallback", "submit", "v_wait", "v_wait_cpu", "v_call", "v_call_cpu", "d_wait", "d_wait_cpu",
             "d_call", "d_call_cpu", "read", "read_cpu", "dev_period", "dev_idle", "dev_verify", "dev_draft")

    def __init__(self, every: int, enabled: bool = True):
        self.every = int(every)
        self.enabled = bool(enabled) and self.every > 0
        self.sums: dict[str, list[float]] = {name: [] for name in self.NAMES}
        self.steps = 0
        self.last_entry: Optional[float] = None
        self.last_exit: Optional[float] = None
        self.prev_draft_mark = None
        # the monotonic times (s) of the newest step's in-flight wait releases (``*_at``), logged with the step
        # number: the ranks of a host run the same steps, so a line's times compare across them
        self.at: dict[str, float] = {}

    def device_step(self, marks: Optional[list]) -> None:
        """Add a read step's device times from its three timed events (right before its verify launch, after its
        verify and after its draft launch)."""
        if not self.enabled or not marks or len(marks) != 3:
            self.prev_draft_mark = None
            return
        verify_start, verify_end, draft_end = marks
        try:
            self.add("dev_verify", verify_start.elapsed_time(verify_end))
            self.add("dev_draft", verify_end.elapsed_time(draft_end))
            if self.prev_draft_mark is not None:
                self.add("dev_period", self.prev_draft_mark.elapsed_time(draft_end))
                self.add("dev_idle", self.prev_draft_mark.elapsed_time(verify_start))
        except Exception:  # noqa: BLE001 - a probe; a runtime without timed events leaves the device times empty
            pass
        self.prev_draft_mark = draft_end

    @classmethod
    def from_env(cls, enabled: bool, environ: Mapping[str, str] = os.environ) -> "StepClock":
        """``N``: a line every ``N`` steps where ``enabled`` (the first DCP rank of each replica); ``N,all``: on every
        rank (the latest rank of a step shows in its device times)."""
        every, _, ranks = environ.get(STEP_TIMES_ENV, "").strip().partition(",")
        every = every.strip()
        return cls(int(every) if every.isdigit() else 0, enabled or ranks.strip() == "all")

    def add(self, name: str, ms: float) -> None:
        if self.enabled:
            self.sums[name].append(float(ms))

    def means(self) -> dict[str, float]:
        return {name: (sum(v) / len(v) if v else 0.0) for name, v in self.sums.items()}

    def maxes(self) -> dict[str, float]:
        return {name: (max(v) if v else 0.0) for name, v in self.sums.items()}

    def step_done(self) -> None:
        if not self.enabled:
            return
        self.steps += 1
        if self.steps % self.every == 0:
            means, maxes = self.means(), self.maxes()
            at = "".join(f", {name} {value:.6f}" for name, value in sorted(self.at.items()))
            from sglang_neuron.backend import output_buffers

            # the graphs' rotating outputs so far (backend.output_buffers): a fresh count that keeps growing means a
            # holder keeps them past their set's turn, and the IO sets stop repeating
            buffers = output_buffers.totals()
            logger.info("Kimi-K3 step times over %d steps (ms, mean/max): %s; outputs reused %d fresh %d; step %d%s",
                        self.every, ", ".join(f"{n} {means[n]:.2f}/{maxes[n]:.2f}" for n in self.NAMES),
                        buffers["reused"], buffers["fresh"], self.steps, at)
            self.sums = {name: [] for name in self.NAMES}


def forced_acceptance(k: int, replica_index: int, seed: Optional[int] = None,
                      environ: Mapping[str, str] = os.environ) -> Optional[hs.SyntheticAcceptance]:
    """The forced acceptance of ``SGLANG_NEURON_KIMI_K3_FORCED_AL``, or None when it is unset."""
    value = environ.get(FORCED_AL_ENV)
    if not value:
        return None
    try:
        rates = hs.forced_rates(value, k)
    except ValueError as error:
        raise ValueError(f"{FORCED_AL_ENV}={value}: {error}") from None
    logger.warning("%s=%s: the decode forces this mean acceptance length (a benchmark); its answers are not the "
                   "target's", FORCED_AL_ENV, value)
    return hs.SyntheticAcceptance(rates, seed, replica_index)


class K3StepDriver:
    """The host side of the Kimi-K3 DSPARK step of one decode rank (the module docstring), over a runtime with
    ``step(inputs) -> StepOutputs``.

    Args:
        runtime: the decode runtime (``K3DecodeRuntime`` or a stand-in).
        config: the served programs' settings (``programs.DecodeProgramConfig``: k, B, the vocab and its TP shards,
            the k3 form's live rows).
        layout: the page space (``sgl_pools.DecodePageLayout``).
        replica_index: the DP replica, which seeds the host draws (the same on every rank of the replica).
        cp_rank: this rank's DCP rank (its stripes of the MLA rows).
        device: where the step inputs go.
        synthetic: forced acceptance counts (``forced_acceptance``), or None.
        zero_blocks: zeroes this rank's MLA blocks (allocator pages) on every MLA layer, or None.
        seed: the launch's random seed.
        max_model_len: the context limit of the launch (positions clamp to it); the page geometry is the programs'.
        logits_dump: the debug dump of the verify's logits (models/kimi_k3/logits_dump.py) over a runtime whose
            outputs carry them (``dumped_logits``), or None.
        step_log: log a ``K3STEP`` line for each harvested step (``step_log_line``) on the first DCP rank; None reads
            ``SGLANG_NEURON_KIMI_K3_STEP_LOG``.
    """

    def __init__(self, runtime, config: DecodeProgramConfig, layout: sp.DecodePageLayout, replica_index: int, *,
                 cp_rank: int = 0, device=None, synthetic: Optional[hs.SyntheticAcceptance] = None,
                 zero_blocks: Optional[Callable[[np.ndarray], None]] = None, seed: Optional[int] = None,
                 max_model_len: Optional[int] = None, logits_dump=None, step_log: Optional[bool] = None):
        if layout.queries != config.queries:
            raise ValueError(f"the step verifies {config.queries} rows a request; the page layout has {layout.queries}")
        self.runtime = runtime
        self.config = config
        self.geometry = dataclasses.replace(config.geometry, cp_rank=int(cp_rank))
        self.layout = layout
        self.device = device
        self.zero_blocks = zero_blocks
        self.host = hs.DecodeSpecHostState(config.k, int(max_model_len or self.geometry.max_model_len), config.vocab,
                                           synthetic=synthetic, forced=True, live_rows=config.live_rows)
        self.draws = hs.HostDraws(seed, replica_index)
        # rid -> the generator of its draws, seeded with its request seed (RowInput.seed)
        self.generators: dict[str, torch.Generator] = {}
        self.last_record: Optional[hs.StepRecord] = None
        self.last_outputs = None
        self.draft_lo: dict[str, int] = {}
        # rid -> (its request row, the positions whose slots a step checked)
        self.checked: dict[str, tuple[int, int]] = {}
        self._static: Optional[dict[str, torch.Tensor]] = None
        self.clock = StepClock.from_env(enabled=int(cp_rank) == 0)
        # rid -> the committed length its last harvested step gives (row 0 + accepted). The result of a step can be
        # read and harvested (on the platform's copy thread) before SGLang commits it, so the next step's host row 0
        # takes it over SGLang's committed length, which still lags
        self.harvested_len: dict[str, int] = {}
        # the host state is planned on the scheduler thread and harvested on the copy thread
        self._lock = threading.Lock()
        self.logits_dump = logits_dump
        self.replica_index = int(replica_index)
        self.cp_rank = int(cp_rank)
        self.step_log = (step_log_enabled() if step_log is None else bool(step_log)) and self.cp_rank == 0

    # -- requests -------------------------------------------------------------------------------------------------
    def drop(self, rid: str) -> None:
        """Forget a finished request."""
        with self._lock:
            self.host.drop_requests((), finished=(rid,))
            self.draft_lo.pop(rid, None)
            self.checked.pop(rid, None)
            self.harvested_len.pop(rid, None)
            self.generators.pop(rid, None)

    def generator(self, row: RowInput) -> Optional[torch.Generator]:
        """The generator of a request's draws: its own, seeded once with its request seed; None without a seed (the
        replica's generator)."""
        if row.seed is None:
            return None
        generator = self.generators.get(row.rid)
        if generator is None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(row.seed) % (1 << 64))
            self.generators[row.rid] = generator
        return generator

    def is_new(self, rid: str) -> bool:
        return rid not in self.host.requests

    def host_row0(self, row: RowInput) -> int:
        """The host's row 0 of a request's next step (the module docstring): the exact committed length once its last
        step is harvested (SGLang's, or the harvest's while SGLang has not committed it yet), else SGLang's committed
        length plus Q while that step is in flight."""
        state = self.host.requests.get(row.rid)
        in_flight = state is not None and state.submitted > state.harvested
        if in_flight:
            return int(row.committed) + self.config.queries
        return max(int(row.committed), self.harvested_len.get(row.rid, 0))

    # -- the step -------------------------------------------------------------------------------------------------
    def _static_inputs(self) -> dict[str, torch.Tensor]:
        """The zero last-step outputs (the carry without a last step), on the device once."""
        if self._static is None:
            cfg = self.config
            values = hs.null_step_outputs(cfg.batch, cfg.queries, True)
            self._static = {name: self._to_device(value) for name, value in values.items() if name in CARRY_INPUTS}
        return self._static

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor if self.device is None else tensor.to(self.device)

    def _carry(self) -> dict[str, torch.Tensor]:
        if self.last_outputs is None:
            return {name: value for name, value in self._static_inputs().items() if name in CARRY_INPUTS}
        return self.last_outputs.carry_inputs()

    def step(self, rows: Sequence[RowInput], req_to_token: np.ndarray) -> StepHandle:
        """Build and run one step of ``rows`` (batch order; none: an idle step on pad rows)."""
        with self._lock:
            return self._step(rows, req_to_token)

    def _step(self, rows: Sequence[RowInput], req_to_token: np.ndarray) -> StepHandle:
        t_plan, c_plan = time.perf_counter(), time.thread_time()
        cfg = self.config
        k, q, batch = cfg.k, cfg.queries, cfg.batch
        n = len(rows)
        if n > batch:
            raise ValueError(f"a step of {n} requests; the replica serves {batch}")
        new = [self.is_new(r.rid) for r in rows]
        row0 = np.zeros(batch, dtype=np.int64)
        row0[:n] = [self.host_row0(r) for r in rows]
        req_ids: list[Optional[str]] = [r.rid for r in rows] + [None] * (batch - n)
        for r, fresh in zip(rows, new):
            if fresh:
                self.draft_lo[r.rid] = sp.draft_window_start(r.committed)
        # a carried row verifies the drafts its last step drafted; a fresh row verifies its bonus token only
        last_rows = {} if self.last_record is None else self.last_record.rows()
        known = {}
        for r in rows:
            state = self.host.requests.get(r.rid)
            carried = (state is not None and self.last_record is not None
                       and (r.rid, state.generation) in last_rows and self.last_record.step == state.submitted)
            known[r.rid] = k if carried else 0
        self.host.known_drafts, self.host.scheduled_drafts = known, dict(known)
        draft_lo = [self.draft_lo[r.rid] for r in rows] + [0] * (batch - n)
        plan = self.host.plan_step(req_ids, row0, ring_requests={r.rid for r in rows}, last=self.last_record,
                                   async_step=True, draft_lo=draft_lo)
        self.host.check_carried_rows(plan, self.last_record)

        # the slots of a request's committed positions do not change while it runs in its request row, so each step
        # checks them only from the page of its last committed length (lowering.request_blocks)
        checked = []
        for r in rows:
            row_index, done = self.checked.get(r.rid, (None, 0))
            checked.append(done if row_index == r.pool_index else 0)
        lowered = lw.lower_decode_batch(req_to_token, [r.pool_index for r in rows], row0[:n].tolist(),
                                        [r.allocated for r in rows], [self.layout.kda_base(r.slot) for r in rows],
                                        batch, self.geometry, checked=checked)
        for r in rows:
            self.checked[r.rid] = (r.pool_index, int(r.committed))
        generators = [self.generator(r) for r in rows]
        tensors = self.host.step_tensors(plan, generators)
        window = lw.mla_slot_window(lowered.mla.raw, tensors["spec_position_window"].numpy(), n, self.geometry)
        ids = torch.full((batch, q), PAD_ID, dtype=torch.int32)
        for i, (r, fresh) in enumerate(zip(rows, new)):
            if fresh:
                ids[i, 0] = int(r.bonus)
        spec_draws, row_draws = hs.step_draws(self.draws, generators, batch, k)
        params = hs.sampling_params_rows([r.temperature for r in rows], [r.top_p for r in rows],
                                         [r.top_k for r in rows], batch, q, cfg.vocab, row_draws)
        tensors.update({
            "input_ids": plan.input_ids(ids.reshape(-1)),
            "mla_block_table": torch.from_numpy(lowered.mla.device),
            "mla_slot_window": torch.from_numpy(window),
            **{f"kda_state_table_{g}": torch.from_numpy(table) for g, table in enumerate(lowered.kda)},
            "spec_draft_rider_ring": sp.ring_table(self.layout, [r.slot for r in rows], batch),
            "sampling_params": params,
            "spec_draws": spec_draws,
        })
        t_upload, c_upload = time.perf_counter(), time.thread_time()
        upload = getattr(self.runtime, "to_device", None)
        if self.device is not None and upload is not None:
            # the runtime's uploads (device_io's side stream on a Neuron device): they do not wait for a queued graph
            inputs = upload(tensors, step=True)
        else:
            inputs = {name: self._to_device(value) for name, value in tensors.items()}
        inputs.update(self._carry())

        if self.zero_blocks is not None:
            for r, fresh in zip(rows, new):
                if fresh:
                    blocks = sp.tail_zero_blocks(req_to_token[r.pool_index], r.committed, r.allocated,
                                                 self.geometry.cp_rank, self.geometry.cp_size)
                    if blocks.size:
                        self.zero_blocks(blocks)
        t_submit, c_submit = time.perf_counter(), time.thread_time()
        probe = self.clock.enabled and self.device is not None and bool(n)
        marks, launch_marks, times = ([], [], {}) if probe else (None, None, None)
        outputs = (self.runtime.step(inputs, marks=marks, launch_marks=launch_marks, times=times) if probe
                   else self.runtime.step(inputs))
        if n:
            done = time.perf_counter()
            self.clock.add("plan", (t_upload - t_plan) * 1e3)
            self.clock.add("plan_cpu", (c_upload - c_plan) * 1e3)
            self.clock.add("upload", (t_submit - t_upload) * 1e3)
            self.clock.add("upload_cpu", (c_submit - c_upload) * 1e3)
            ring_times = getattr(self.runtime, "input_ring_times", None)
            for name, value in (ring_times() if callable(ring_times) else {}).items():
                self.clock.add(f"up_{name}", value)
            for name, value in (times or {}).items():
                if name.endswith("_at"):
                    self.clock.at[name] = value
                else:
                    self.clock.add(name, value)
            self.clock.add("submit", (done - t_submit) * 1e3)
        if n == 0:
            # an idle step: no request carries its state into the next step
            self.last_record, self.last_outputs = None, None
            self.clock.prev_draft_mark = None
            return StepHandle(None, outputs, [], row0[:0])
        plan.record.outputs = (outputs.accepted, outputs.next_ids, outputs.row0)
        self.last_record, self.last_outputs = plan.record, outputs
        return StepHandle(plan.record, outputs, [r.rid for r in rows], row0[:n],
                          marks=(launch_marks + marks) if probe and launch_marks and marks else None)

    def finish(self, handle: StepHandle) -> HostStep:
        """Read a step's results to the host and harvest them (once)."""
        if handle.host is not None:
            return handle.host
        if handle.record is None:
            handle.host = HostStep(torch.zeros((0, self.config.queries), dtype=torch.int64),
                                   torch.zeros(0, dtype=torch.int32), torch.zeros(0, dtype=torch.int64))
            return handle.host
        out = handle.outputs
        read = getattr(self.runtime, "read", None)
        t_read, c_read = time.perf_counter(), time.thread_time()
        if read is not None and getattr(out, "launch_event", None) is not None:
            # waits for this step's graphs only (its launch event), not for the step launched after it
            values = {name: torch.from_numpy(value) for name, value in read(out).items()}
        else:
            names = ("accepted", "row0", "next_ids", "tokens", "flags")
            values = {name: getattr(out, name).to("cpu") for name in names}
        self.clock.add("read", (time.perf_counter() - t_read) * 1e3)
        self.clock.add("read_cpu", (time.thread_time() - c_read) * 1e3)
        self.clock.device_step(handle.marks)
        if self.logits_dump is not None:
            self._dump_logits(handle, values)
        with self._lock:
            return self._harvest(handle, values)

    def _dump_logits(self, handle: StepHandle, values: Mapping[str, torch.Tensor]) -> None:
        """Write the step's verify rows of the wanted requests (models/kimi_k3/logits_dump.py), after the step's read
        on the same thread (one host waiter on the step)."""
        logits = getattr(handle.outputs, "dumped_logits", None)
        if logits is None:
            raise RuntimeError("The logits dump is on, and the step's outputs carry no logits.")
        cfg = self.config
        self.logits_dump.after_step(
            handle.rids, logits, values["row0"].reshape(-1).tolist(), values["accepted"].reshape(-1).tolist(),
            values["tokens"].to(torch.int64).tolist(), queries=cfg.queries, rank=self.cp_rank, world=cfg.tp,
            replica=self.replica_index, event=getattr(handle.outputs, "launch_event", None))

    def _harvest(self, handle: StepHandle, values: Mapping[str, torch.Tensor]) -> HostStep:
        if handle.host is not None:
            return handle.host
        n = len(handle.rids)
        accepted = values["accepted"].reshape(-1).to(torch.int64)
        row0 = values["row0"].reshape(-1).to(torch.int64)
        next_ids = values["next_ids"].to(torch.int64)
        tokens = values["tokens"].to(torch.int64)[:n].clone()
        flags = values["flags"].reshape(-1).to(torch.int64)
        self.host.harvest(handle.record, row0.numpy(), accepted.numpy(), next_ids.numpy())
        failures = self.host.take_failures()
        lens = accepted[:n].clone()
        vocab = int(self.config.vocab)
        for i, rid in enumerate(handle.rids):
            if rid in failures:
                continue
            run = tokens[i, : int(lens[i])]
            error = None
            if int(flags[i]):
                reasons = ", ".join(flag_reasons(int(flags[i])))
                error = f"Request {rid}: the step's sampler flagged its rows ({reasons}); its tokens are dropped."
            elif bool(((run < 0) | (run >= vocab)).any()):
                error = (f"Request {rid}: the step accepted {int(lens[i])} rows and returned the tokens "
                         f"{tokens[i].tolist()}; the accepted rows must hold ids below the vocab size {vocab}.")
            if error is not None:
                self.host.fail_request(rid, error)
                failures[rid] = error
        # a failed request gets none of the step's tokens
        for i, rid in enumerate(handle.rids):
            if rid in failures:
                tokens[i] = -1
                lens[i] = 0
        new_lens = row0[:n] + lens
        for i, rid in enumerate(handle.rids):
            if rid in self.host.requests:
                self.harvested_len[rid] = int(new_lens[i])
        handle.host = HostStep(tokens, lens.to(torch.int32), new_lens, failures)
        if self.step_log:
            self._log_step(handle.record, row0, accepted, failures)
        return handle.host

    def _log_step(self, record: hs.StepRecord, row0: torch.Tensor, accepted: torch.Tensor,
                  failures: Mapping[str, str]) -> None:
        """Log the ``K3STEP`` line of a harvested step; a step that stored no request logs none."""
        slots, stored = [], False
        for row, rid in enumerate(record.req_ids):
            if rid is None:
                slots.append((STEP_LOG_EMPTY,) * 3)
                continue
            state = self.host.requests.get(rid)
            if rid in failures or state is None or state.harvested != record.step:
                slots.append((STEP_LOG_SKIPPED,) * 3)
                continue
            slots.append((str(int(row0[row])), str(int(accepted[row])), str(rid)[-8:]))
            stored = True
        if stored:
            logger.info("%s", step_log_line(self.replica_index, record.step, record.context_bucket,
                                            self.config.queries, slots))


class K3PendingStep:
    """A step whose results SGLang reads when it processes the step: the platform's deferred result copy
    (model_executor.async_execution) calls ``fill_result`` on the scheduler thread, right before SGLang commits the
    step (the decode reads its results inline, decode.host), so every earlier step of the batch's requests is
    committed then."""

    def __init__(self, driver: K3StepDriver, handle: StepHandle, reqs: Sequence[Any]):
        self.driver, self.handle, self.reqs = driver, handle, list(reqs)

    def fill_result(self, result) -> None:
        host = self.driver.finish(self.handle)
        tokens, lens, new_lens = host.tokens.clone(), host.accept_lens.clone(), host.new_seq_lens.clone()
        for i, rid in enumerate(self.handle.rids):
            error = host.failures.get(rid)
            if error is None:
                continue
            finish_failed(self.reqs[i], error)
            # SGLang commits accept_lens[i] tokens of a row and counts accept_lens[i] - 1 accepted drafts, so a failed
            # row hands it one pad token, which finish_failed keeps out of the request's output
            tokens[i, 0], lens[i], new_lens[i] = PAD_ID, 1, new_lens[i] + 1
        result.next_token_ids = tokens.reshape(-1)
        result.accept_lens = lens
        result.new_seq_lens = new_lens


def finish_failed(req, error: str) -> None:
    """Finish a request whose step failed a host check at that step's commit, with none of the step's tokens: SGLang
    appends the step's row to ``output_ids`` and finishes the request there (``request_finish``), and the output it
    streams ends at ``finished_len`` (``Req.output_ids_through_stop``), the length before the step. A request that
    has finished already is left as it is."""
    from sglang.srt.managers.schedule_batch import FINISH_ABORT

    if req.finished():
        return
    req.finished_len = len(req.output_ids)
    req.request_finish(FINISH_ABORT(error, 500, "InternalServerError"))


def _logits_with(pending) -> Any:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput

    from sglang_neuron.model_executor.async_execution import mark_pending

    return mark_pending(LogitsProcessorOutput(next_token_logits=None), pending)


def unserved_reasons(req, return_logprob: bool = False, return_hidden_states: bool = False) -> list[str]:
    """What the decode programs do not serve of a request: their sampler takes temperature, top-p and top-k only."""
    reasons = []
    if getattr(req, "grammar", None) is not None:
        reasons.append("structured output (grammar)")
    if return_logprob and getattr(req, "return_logprob", False):
        reasons.append("logprobs")
    if return_hidden_states and getattr(req, "return_hidden_states", False):
        reasons.append("hidden states")
    params = getattr(req, "sampling_params", None)
    if params is not None:
        for name, off in (("frequency_penalty", 0.0), ("presence_penalty", 0.0), ("repetition_penalty", 1.0),
                          ("min_p", 0.0), ("min_new_tokens", 0)):
            value = getattr(params, name, off)
            if value is not None and value != off:
                reasons.append(f"{name}={value}")
        if getattr(params, "logit_bias", None):
            reasons.append("logit_bias")
        if getattr(params, "custom_params", None) or getattr(req, "custom_logit_processor", None):
            reasons.append("a custom logit processor")
    return reasons


def _worker_class():
    from sglang.srt.managers.utils import GenerationBatchResult
    from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
    from sglang.srt.speculative.dspark_components.dspark_draft import make_next_draft_input

    class KimiK3DsparkWorker(BaseSpecWorker):
        """SGLang's spec-v2 worker of the Kimi-K3 DSPARK decode (the module docstring), built by the platform's
        speculative-worker seam (model_executor.spec_worker) from the decode host's ``speculative_worker_cls``."""

        def __init__(self, server_args, gpu_id: int, nccl_port: int, target_worker, **kwargs):
            super().__init__()
            self._target_worker = target_worker
            self._draft_worker = None  # the drafter is the runtime's draft program
            self.server_args = server_args
            self.model = target_worker.model_runner.model
            reasons = worker_reasons(server_args, self.model)
            if reasons:
                raise ValueError("the Kimi-K3 DSPARK worker cannot serve this launch:\n  - " + "\n  - ".join(reasons))
            self.queries = int(server_args.speculative_num_draft_tokens)
            self.overlap = not bool(getattr(server_args, "disable_overlap_schedule", False))
            self.speculative_num_steps, self.speculative_num_draft_tokens, self.topk = 1, self.queries, 1
            logger.info("Kimi-K3 DSPARK worker: k = %d, %s schedule", self.queries - 1,
                        "overlap" if self.overlap else "synchronous")

        def init_cuda_graphs(self):
            return None

        def init_attention_backends(self):
            return None

        @property
        def primary_draft_kv_pool(self):
            """The rider pool, whose 5 drafter entries a PD decode registers after the target's."""
            return self.model.rider_pool

        def note_request_finished(self, *, rid: str, natural_stop: bool) -> None:
            driver = self.model.step_driver(required=False)
            if driver is not None:
                driver.drop(rid)

        # What SGLang's scheduler asks a DSPARK draft worker while it serves: the decode-stats log line
        # (metrics_reporter.report_decode_stats, every --decode-log-interval steps) and the server info
        # (internal_state). Upstream's worker answers from its block-accept recorder and its info records; the K3
        # worker keeps neither (the acceptance is in SGLang's own accept-length stats).
        def block_accept_estimate_log_suffix(self) -> Optional[str]:
            return None

        def dump_info_records(self) -> Optional[dict]:
            return None

        def clear_info_records(self) -> None:
            return None

        def forward_batch_generation(self, batch, on_publish=None, grammar_barrier=None, pp_proxy_tensors=None,
                                     **kwargs):
            if pp_proxy_tensors is not None:
                raise ValueError("the Kimi-K3 DSPARK worker runs no pipeline stage")
            mode = batch.forward_mode
            if mode.is_idle() or (mode.is_decode() and batch.batch_size() == 0):
                result = self._idle(batch)
            elif mode.is_decode():
                result = self._decode(batch)
            else:
                raise RuntimeError(f"the Kimi-K3 decode runs decode and idle steps only, not {mode}")
            if on_publish is not None:
                on_publish(result.new_seq_lens)
            return result

        def _idle(self, batch):
            driver = self.model.step_driver()
            driver.step([], np.zeros((1, 1), dtype=np.int64))
            empty = torch.zeros(0, dtype=torch.int64)
            return GenerationBatchResult(
                logits_output=None, next_token_ids=empty, accept_lens=torch.zeros(0, dtype=torch.int32),
                next_draft_input=make_next_draft_input(bonus_tokens=empty, new_seq_lens=empty),
                speculative_num_draft_tokens=self.queries, new_seq_lens=empty, can_run_cuda_graph=False)

        def _rows(self, batch) -> list[RowInput]:
            from sglang.srt.managers.schedule_batch import FINISH_ABORT

            from sglang_neuron.sampling import request_seed

            driver = self.model.step_driver()
            bonus = getattr(batch.spec_info, "bonus_tokens", None)
            bonus = None if bonus is None else bonus.to("cpu").reshape(-1).tolist()
            rows = []
            for i, req in enumerate(batch.reqs):
                if driver.is_new(req.rid):
                    reasons = unserved_reasons(req, batch.return_logprob, batch.return_hidden_states)
                    if reasons:
                        req.request_finish(FINISH_ABORT(
                            f"the Kimi-K3 decode serves temperature, top-p and top-k sampling only, not "
                            f"{', '.join(reasons)}", 400, "BadRequestError"))
                params = req.sampling_params
                kv = req.kv
                rows.append(RowInput(rid=req.rid, pool_index=int(kv.req_pool_idx), committed=int(kv.kv_committed_len),
                                     allocated=int(kv.kv_allocated_len), slot=int(kv.mamba_pool_idx),
                                     bonus=PAD_ID if bonus is None else int(bonus[i]),
                                     temperature=float(params.temperature), top_p=float(params.top_p),
                                     top_k=int(params.top_k), seed=request_seed(req)))
            return rows

        def _decode(self, batch):
            driver = self.model.step_driver()
            clock = driver.clock
            entry = time.perf_counter()
            if clock.last_entry is not None:
                clock.add("period", (entry - clock.last_entry) * 1e3)
            if clock.last_exit is not None:
                clock.add("between", (entry - clock.last_exit) * 1e3)
            clock.last_entry = entry
            rows = self._rows(batch)
            clock.add("rows", (time.perf_counter() - entry) * 1e3)
            req_to_token = batch.req_to_token_pool.req_to_token
            handle = driver.step(rows, req_to_token.numpy() if isinstance(req_to_token, torch.Tensor) else req_to_token)
            # the host's bound of each request's committed length after the step (its row 0 plus Q): the next step
            # reads its own row 0 from SGLang's committed lengths, and the result fills the exact lengths
            upper = torch.from_numpy(handle.row0 + self.queries).to(torch.int64)
            n = len(rows)
            pending = K3PendingStep(driver, handle, batch.reqs)
            result = GenerationBatchResult(
                logits_output=_logits_with(pending),
                next_token_ids=torch.zeros(n * self.queries, dtype=torch.int64),
                accept_lens=torch.ones(n, dtype=torch.int32),
                next_draft_input=make_next_draft_input(bonus_tokens=torch.zeros(n, dtype=torch.int64),
                                                       new_seq_lens=upper),
                speculative_num_draft_tokens=self.queries, new_seq_lens=upper, can_run_cuda_graph=False)
            if not self.overlap:
                pending.fill_result(result)
                result.logits_output.neuron_pending_sample = None
            clock.last_exit = time.perf_counter()
            clock.step_done()
            return result

    return KimiK3DsparkWorker


_WORKER = None


def worker_class() -> type:
    """The worker class (a ``BaseSpecWorker``), built once SGLang is importable (in the scheduler processes)."""
    global _WORKER
    if _WORKER is None:
        _WORKER = _worker_class()
    return _WORKER


def worker_reasons(server_args, model) -> list[str]:
    """Why the Kimi-K3 DSPARK worker cannot serve this launch (empty: it can)."""
    reasons = []
    if not hasattr(model, "step_driver"):
        return [f"the Kimi-K3 DSPARK worker serves the Kimi-K3 decode host, not {type(model).__name__}"]
    queries = int(getattr(server_args, "speculative_num_draft_tokens", 0) or 0)
    if queries not in (4, 8):
        reasons.append(f"--speculative-num-draft-tokens {queries}: the decode programs verify k + 1 = 4 or 8 rows")
    if int(getattr(server_args, "speculative_num_steps", 1) or 1) != 1:
        reasons.append("--speculative-num-steps above 1: one draft program call drafts every draft")
    if int(getattr(server_args, "speculative_eagle_topk", 1) or 1) != 1:
        reasons.append("--speculative-eagle-topk above 1: the drafts are one chain")
    if int(getattr(server_args, "pp_size", 1) or 1) > 1:
        reasons.append("pipeline stages: the decode programs run the whole model")
    return reasons
