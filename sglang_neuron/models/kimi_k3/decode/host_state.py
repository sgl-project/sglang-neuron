# SPDX-License-Identifier: Apache-2.0
"""Host state of the Kimi-K3 DSpark decode step: per-request carry state, step plans, step tensors and host draws.

One verify program runs the whole speculative step of a DP replica; the host keeps a small state for each request id
(``DecodeSpecHostState``):

- ``replay_len``: the verify rows that the last verify step of the request accepted (0 before its first step). The
  verify kernel loads the KDA state after row ``replay_len - 1`` from the request's state slots.
- ``row0``: the row-0 position of the last verify step. The next verify step must start at ``row0 + replay_len``.
- ``ctx_start``: the first drafter context position of a request without a transferred drafter context.
- ``generation``: a number that drops the late output of a request that finished, was preempted or resumed.

The state belongs to the request id, not to the batch row. Async steps: the host's row 0 of a carried request is
optimistic, so the program computes row 0 on the device and returns ``row0``, the capped ``accepted`` count and the
``next_ids`` of each row; the next step takes them as inputs. Only a fresh row (a new, absent or resumed request,
padding, or any row of a sync step) takes host values. The host reads the outputs one step late (``harvest``) and
checks them; a failed check fails that request only (``failures``).

Every decode step verifies ``1 + k`` rows of each request; a request with fewer known drafts gets pad drafts and the
program accepts at most the known drafts.

Host inputs that are traced shapes: the window positions ``[B, 2k + 1]``, the drafter rope rows ``[B, W, 128]``, the
draws ``[B, 2k + 2]`` (``u_accept [k]``, ``u_final``, ``u_draft [k]``, ``u_out``) and the sampling params
``[B * Q, 4]`` (``[top_k, top_p, temperature, draw]``).
"""

from __future__ import annotations

import itertools
import logging
import math
import threading
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

logger = logging.getLogger(__name__)

PAD_DRAW = 0.5
"""The draw of a pad row or a warmup row; it has no effect on a real request."""
SAMPLING_EPS = 1e-5
"""A temperature below this samples greedily."""
_SYNTHETIC_SEED_SALT = 0x5EED_ACC7
"""Mixed into the replica seed so the synthetic draws differ from the sampling draws."""
_MIX_GOLDEN = 0x9E3779B97F4A7C15
_MIX_A = 0xBF58476D1CE4E5B9
_MIX_B = 0x94D049BB133111EB
_MASK_63 = (1 << 63) - 1
_MASK_64 = (1 << 64) - 1

SPEC_STEP_INPUTS = (
    "fresh",
    "prev_row",
    "kda_replay_len",
    "kda_state_position",
    "known",
    "window_base",
    "position_window",
    "draft_lo",
    "rope_window",
    "prev_row0",
    "prev_accepted",
    "prev_next_ids",
)
"""The per-step inputs of the speculative program; the host passes each one as ``spec_<name>``."""
SPEC_FORCED_INPUT = "forced_accept"
SPEC_SYNTHETIC_INPUT = "synthetic_accept"
SPEC_LIVE_ROWS_INPUT = "live_rows"


# Host draws.


def replica_seed(seed: int | None, replica_index: int) -> int:
    """Return the seed of the default draw generator of one DP replica (SplitMix64 of the seed and the replica).

    The same on each TP rank of the replica and different for each replica; None means 0.
    """
    value = ((seed or 0) + _MIX_GOLDEN * (replica_index + 1)) & _MASK_64
    value = ((value ^ (value >> 30)) * _MIX_A) & _MASK_64
    value = ((value ^ (value >> 27)) * _MIX_B) & _MASK_64
    return (value ^ (value >> 31)) & _MASK_63


class HostDraws:
    """Give host draws in [0, 1): a request with a seed draws from its own CPU generator, the others from the
    replica's generator (``replica_seed``), so every TP rank of a replica makes the same draws."""

    def __init__(self, seed: int | None, replica_index: int) -> None:
        self.replica_index = int(replica_index)
        self._default = torch.Generator(device="cpu")
        self._default.manual_seed(replica_seed(seed, self.replica_index))

    def draw(
        self,
        generators: Sequence[torch.Generator | None],
        num_rows: int | None = None,
        rows_per_request: int = 1,
    ) -> torch.Tensor:
        """Return ``[num_rows]`` float32: draw ``j`` of request ``i`` at row ``i * rows_per_request + j``; rows after
        the requests get ``PAD_DRAW``."""
        if rows_per_request < 1:
            raise ValueError("Give one sampled row or more for each request.")
        needed = len(generators) * rows_per_request
        rows = needed if num_rows is None else int(num_rows)
        if rows < needed:
            raise ValueError(f"The draw has {rows} rows for {needed} sampled rows.")
        out = torch.full((rows,), PAD_DRAW, dtype=torch.float32)
        for index, generator in enumerate(generators):
            source = self._default if generator is None else generator
            start = index * rows_per_request
            # FP32 by name: SGLang's model loader sets torch's default dtype to the model dtype (bf16).
            out[start : start + rows_per_request] = torch.rand((rows_per_request,), dtype=torch.float32,
                                                                 generator=source)
        return out


def spec_step_draw_count(num_speculative_tokens: int) -> int:
    """Return the host draws of a request in one step: ``u_accept [k]``, ``u_final``, ``u_draft [k]``, ``u_out``."""
    return 2 * num_speculative_tokens + 2


def step_draws(
    draws: HostDraws, generators: Sequence[torch.Generator | None], batch: int, num_speculative_tokens: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(spec_draws [B, 2k + 2], row_draws [R * Q])`` of one step of ``R = len(generators)`` requests.

    ``spec_draws`` pads the rows after the requests with ``PAD_DRAW``; ``row_draws`` is the draw column of the
    sampling params: the first ``Q = k + 1`` draws of each request.
    """
    width = spec_step_draw_count(num_speculative_tokens)
    num_reqs = len(generators)
    full = draws.draw(generators, rows_per_request=width).reshape(num_reqs, width)
    out = torch.full((batch, width), PAD_DRAW, dtype=torch.float32)
    out[:num_reqs] = full[:batch]
    return out, full[:, : num_speculative_tokens + 1].reshape(-1)


def sampling_params_rows(
    temperature: Sequence[float],
    top_p: Sequence[float],
    top_k: Sequence[int],
    batch: int,
    queries: int,
    vocab_size: int,
    row_draws: torch.Tensor | None = None,
    force_greedy: bool = False,
) -> torch.Tensor:
    """Return the sampler params ``[B * Q, 3]`` (or ``[B * Q, 4]`` with the draw column) of a verify step, fp32.

    Each of the ``R`` requests gives its temperature, top_p and top_k. A greedy request (temperature below 1e-5) has
    temperature 0, top_p 1 and no top_k; a top_k outside ``(0, vocab)`` means none and takes the vocab size. A column
    that every request leaves off takes its default (top_k -1, top_p 1, temperature 0). The rows after the requests
    are 0, each request row repeats for its Q verify rows (row ``b * Q + j``), and the draw column (``row_draws``,
    ``[R * Q]``) has ``PAD_DRAW`` after the requests.
    """
    num_reqs = len(temperature)
    if len(top_p) != num_reqs or len(top_k) != num_reqs:
        raise ValueError("Give one temperature, top_p and top_k for each request.")
    greedy = [float(t) < SAMPLING_EPS for t in temperature]
    temps = [0.0 if g else float(t) for g, t in zip(greedy, temperature)]
    tops_p = [1.0 if g else float(p) for g, p in zip(greedy, top_p)]
    tops_k = []
    for g, k in zip(greedy, top_k):
        k = 0 if g else int(k)
        tops_k.append(k if 0 < k < vocab_size else vocab_size)
    has_top_k = any(0 < (0 if g else int(k)) < vocab_size for g, k in zip(greedy, top_k))
    has_top_p = any(p < 1 for p in tops_p)
    all_greedy = all(greedy)
    columns = [
        torch.tensor(tops_k, dtype=torch.int32).float() if has_top_k else torch.full((num_reqs,), -1.0),
        torch.tensor(tops_p, dtype=torch.float32) if has_top_p else torch.ones(num_reqs, dtype=torch.float32),
        torch.tensor(temps, dtype=torch.float32)
        if not all_greedy and not force_greedy
        else torch.zeros(num_reqs, dtype=torch.float32),
    ]
    params = torch.stack(columns, dim=1)
    if params.shape[0] < batch:
        params = torch.nn.functional.pad(params, (0, 0, 0, batch - params.shape[0]), value=0)
    index = torch.arange(params.shape[0]).repeat_interleave(queries)
    params = torch.index_select(params, 0, index)
    if row_draws is None:
        return params
    if row_draws.shape[0] > params.shape[0]:
        raise ValueError(f"The draw column has {row_draws.shape[0]} rows for {params.shape[0]} sampled rows.")
    column = torch.full((params.shape[0],), PAD_DRAW, dtype=torch.float32)
    column[: row_draws.shape[0]] = row_draws.to(torch.float32)
    return torch.cat([params.to(torch.float32), column.unsqueeze(1)], dim=1)


# Forced and synthetic acceptance counts (test and benchmark only).


def unconditional_to_conditional_rates(rates: Sequence[float]) -> list[float]:
    """Convert per-position unconditional acceptance rates ``p`` to the conditional rates ``c_i = p_i / p_{i-1}`` of
    the early-terminating accept loop."""
    rates = list(rates)
    return [p / q if q > 0.0 else 0.0 for p, q in zip(rates, [1.0, *rates[:-1]])]


def acceptance_length_to_rates(length: float, n: int) -> list[float]:
    """Return the minimum-variance unconditional rates of ``n`` draft positions for a mean acceptance length
    ``length`` (1 .. n + 1): ``floor(length - 1)`` positions always accept, the next one with the fractional rest."""
    num_drafts = length - 1
    num_full = int(num_drafts)
    return ([1.0] * num_full + [num_drafts - num_full] + [0.0] * (n - num_full - 1))[:n]


def forced_rates(mean_length: str | float, num_speculative_tokens: int) -> list[float]:
    """Return the forced-acceptance rates of a mean acceptance length in ``[1, k + 1]``.

    Raises:
        ValueError: the length is not a number in ``[1, k + 1]``.
    """
    try:
        length = float(mean_length)
    except ValueError:
        length = float("nan")
    k = int(num_speculative_tokens)
    if not 1.0 <= length <= float(k + 1):
        raise ValueError(f"The forced mean acceptance length is {mean_length!r}; give a number in [1, {k + 1}].")
    return acceptance_length_to_rates(length, k)


class SyntheticAcceptance:
    """Accept the draft at position ``i`` while its uniform draw is below the conditional rate of the unconditional
    rates ``rates``, up to the first reject, whatever the target token (test and benchmark only).

    A request with a seed draws from its own generator; the others from the replica generator
    (``replica_seed ^ salt``).
    """

    def __init__(self, rates: Sequence[float], seed: int | None, replica_index: int) -> None:
        self.rates = np.asarray(unconditional_to_conditional_rates(list(rates)), dtype=np.float64)
        self._default = torch.Generator(device="cpu")
        self._default.manual_seed(replica_seed(seed, replica_index) ^ _SYNTHETIC_SEED_SALT)

    def accepted(self, known: np.ndarray, generators: Sequence[torch.Generator | None] = ()) -> np.ndarray:
        """Return ``[B]`` int32: the leading drafts of each row whose draw is below its rate (one float64 draw for each
        of its ``known`` drafts; a row without drafts draws nothing)."""
        out = np.zeros(len(known), dtype=np.int32)
        for row, count in enumerate(int(n) for n in known):
            if count <= 0:
                continue
            generator = generators[row] if row < len(generators) else None
            draws = torch.rand(
                (count,), dtype=torch.float64, generator=self._default if generator is None else generator
            ).numpy()
            rejected = np.flatnonzero(draws >= self.rates[:count])
            out[row] = rejected[0] if rejected.size else count
        return out


# Drafter rope window (DSpark drafter YaRN: theta 50000, factor 32, original 32768, beta 32 / 1, mscale 1 / 1).

DRAFT_ROPE_DIM = 64
DRAFT_ROPE_THETA = 50000.0
DRAFT_YARN_FACTOR = 32.0
DRAFT_YARN_ORIGINAL = 32768
DRAFT_YARN_BETA_FAST = 32.0
DRAFT_YARN_BETA_SLOW = 1.0
DRAFT_YARN_MSCALE = 1.0
DRAFT_YARN_MSCALE_ALL_DIM = 1.0


def _yarn_mscale(factor: float, mscale: float) -> float:
    return 1.0 if factor <= 1 else 0.1 * mscale * math.log(factor) + 1.0


def draft_yarn_inv_freq() -> torch.Tensor:
    """Return the FP32 YaRN inverse frequencies ``[32]`` of the drafter rope."""
    dim = DRAFT_ROPE_DIM
    pos_freqs = DRAFT_ROPE_THETA ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    extrapolation = 1.0 / pos_freqs
    interpolation = 1.0 / (DRAFT_YARN_FACTOR * pos_freqs)

    def correction_dim(rotations: float) -> float:
        return (dim * math.log(DRAFT_YARN_ORIGINAL / (rotations * 2 * math.pi))) / (2 * math.log(DRAFT_ROPE_THETA))

    low = max(math.floor(correction_dim(DRAFT_YARN_BETA_FAST)), 0)
    high = min(math.ceil(correction_dim(DRAFT_YARN_BETA_SLOW)), dim - 1)
    if low == high:
        high += 0.001
    ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
    mask = 1 - ramp
    return interpolation * (1 - mask) + extrapolation * mask


def draft_rope_rows(positions: np.ndarray) -> torch.Tensor:
    """Return the FP32 YaRN rows ``[N, 128] = [cos_x | sin_x]`` of the drafter for integer positions (host).

    Elements ``2i`` and ``2i + 1`` of ``cos_x`` both hold ``cos_i`` (and the same for ``sin_x``), so the kernel rotates
    a GPT-J pair row as ``x * cos_x + rot(x) * sin_x``. A negative position is clamped to 0.
    """
    with torch.device("cpu"):
        pos = torch.as_tensor(np.asarray(positions).reshape(-1), dtype=torch.int64).clamp(min=0)
        scale = _yarn_mscale(DRAFT_YARN_FACTOR, DRAFT_YARN_MSCALE) / _yarn_mscale(
            DRAFT_YARN_FACTOR, DRAFT_YARN_MSCALE_ALL_DIM
        )
        freqs = pos.to(torch.float32).reshape(-1, 1) * draft_yarn_inv_freq()
        cos, sin = freqs.cos() * scale, freqs.sin() * scale
        return torch.cat([cos.repeat_interleave(2, dim=-1), sin.repeat_interleave(2, dim=-1)], dim=-1)


def rope_window_width(num_speculative_tokens: int) -> int:
    """Return the rope window, ``max(3k + 2, 2k + 8)``: the last drafter block row is at offset ``2k + 7`` at most."""
    return max(3 * num_speculative_tokens + 2, 2 * num_speculative_tokens + 8)


def host_rope_window(window_base: np.ndarray, width: int) -> torch.Tensor:
    """Return the fp32 YaRN rows ``[B, width, 128]`` of the positions ``window_base + i`` (host)."""
    base = np.asarray(window_base).reshape(-1, 1)
    rows = draft_rope_rows(base + np.arange(width))
    return rows.reshape(base.shape[0], width, rows.shape[-1])


# Inputs of steps without a last step, and of the sampled rows.

DRAFT_SAMPLE_WIDTH = 256
"""The q-list width of the drafter."""


def null_step_outputs(batch: int, queries: int, probabilistic: bool) -> dict[str, torch.Tensor]:
    """Return zero last-step outputs for a step with fresh rows only (the first step, a warmup or an idle step).

    Each input is its own tensor: the compile merges one tensor that two inputs share. With probabilistic drafts the
    q lists are zeros (a fresh row verifies no draft) and the draws ``PAD_DRAW``.
    """
    outputs = {
        "spec_prev_row0": torch.zeros(batch, dtype=torch.int32),
        "spec_prev_accepted": torch.zeros(batch, dtype=torch.int32),
        "spec_prev_next_ids": torch.zeros((batch, queries), dtype=torch.int32),
    }
    if probabilistic:
        k = queries - 1
        outputs.update(
            {
                "spec_prev_q_probs": torch.zeros((batch, k, DRAFT_SAMPLE_WIDTH), dtype=torch.float32),
                "spec_prev_q_ids": torch.zeros((batch, k, DRAFT_SAMPLE_WIDTH), dtype=torch.int32),
                "spec_draws": torch.full((batch, spec_step_draw_count(k)), PAD_DRAW, dtype=torch.float32),
            }
        )
    return outputs


# Per-request state and step plans.


def slot_window_width(num_speculative_tokens: int) -> int:
    """Return the candidate positions of the verify rows: ``N - k .. N + k`` for the host's row 0 ``N``."""
    return 2 * num_speculative_tokens + 1


@dataclass
class RequestState:
    generation: int
    replay_len: int = 0
    ctx_start: int | None = None
    # Row 0 and the next ids of the last harvested verify step; None before the first one.
    row0: int | None = None
    next_ids: np.ndarray | None = None
    # Step ids of the last submitted and the last harvested verify step (-1: none).
    submitted: int = -1
    harvested: int = -1


@dataclass
class StepRecord:
    """The host view of one submitted speculative step, for the bookkeeping of its outputs.

    ``outputs`` holds the device outputs ``(accepted, next_ids, row0)`` of an async step; ``context_bucket`` the
    context of the step in tokens.
    """

    step: int
    req_ids: list[str | None]
    generations: list[int]
    prev_steps: list[int]
    sched_row0: np.ndarray
    known: np.ndarray
    fresh: np.ndarray
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    context_bucket: int = 0

    def rows(self) -> dict[tuple[str, int], int]:
        """Return the row of each (request id, generation) of the step."""
        return {
            (req_id, generation): row
            for row, (req_id, generation) in enumerate(zip(self.req_ids, self.generations))
            if req_id is not None
        }


@dataclass
class StepPlan:
    """The host inputs of one speculative step (int32 ``[B]`` arrays unless given).

    ``fresh`` is 1 for a row with host values; any other row takes its state from row ``prev_row`` of the last step.
    ``window_base`` is the host's row 0 minus k (int64). ``host_ids`` maps a fresh row to the ids ``[Q]`` of its last
    step.
    """

    record: StepRecord
    fresh: np.ndarray
    prev_row: np.ndarray
    replay: np.ndarray
    state_position: np.ndarray
    known: np.ndarray
    window_base: np.ndarray
    draft_lo: np.ndarray
    host_ids: dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def carried(self) -> bool:
        """Return True when a row takes its state from the last step."""
        return bool((self.fresh == 0).any())

    def position_window(self, width: int, max_model_len: int) -> np.ndarray:
        """Return the int32 positions ``[B, width]`` of the window, clamped to the context (0 for padding)."""
        positions = self.window_base.reshape(-1, 1) + np.arange(width)
        positions = np.clip(positions, 0, max_model_len - 1)
        real = np.array([req_id is not None for req_id in self.record.req_ids])
        return np.where(real.reshape(-1, 1), positions, 0).astype(np.int32)

    def input_ids(self, host_ids: torch.Tensor) -> torch.Tensor:
        """Return the request-major host ids ``[B * Q]`` with the ids of the last step of each mapped row."""
        if not self.host_ids:
            return host_ids
        ids = host_ids.clone().reshape(len(self.record.req_ids), -1)
        for row, values in self.host_ids.items():
            ids[row] = torch.from_numpy(values).to(ids.dtype)
        return ids.reshape(-1)


class DecodeSpecHostState:
    """Keep the per-request host state of the Kimi-K3 DSpark step of one DP replica and build its host inputs.

    The output thread and the step thread both use the state, so a lock guards it.

    Args:
        num_speculative_tokens: k, the drafts of each verify step.
        max_model_len: the largest position plus one.
        vocab_size: the vocab; a device draft outside it is counted in ``bad_drafts``.
        synthetic: the synthetic or forced acceptance counts of a test run, or None.
        forced: give each step the forced counts input (``spec_forced_accept``): the counts of ``synthetic``, or -1
            (the served result) without it. A real and a forced run then run the same program.
        live_rows: give each step ``spec_live_rows`` ``[B, 1]`` int32 (1 for a request row, 0 for a pad row).
    """

    def __init__(
        self,
        num_speculative_tokens: int,
        max_model_len: int,
        vocab_size: int | None = None,
        synthetic: SyntheticAcceptance | None = None,
        forced: bool = False,
        live_rows: bool = False,
    ) -> None:
        self.num_speculative_tokens = num_speculative_tokens
        self.queries = num_speculative_tokens + 1
        self.max_model_len = max_model_len
        self.vocab_size = vocab_size
        self.synthetic = synthetic
        self.forced = forced
        self.live_rows = live_rows
        self.requests: dict[str, RequestState] = {}
        # The known drafts (the acceptance cap) and the scheduled drafts of each request of the current step.
        self.known_drafts: dict[str, int] = {}
        self.scheduled_drafts: dict[str, int] = {}
        self.stats = {"steps": 0, "rows": 0, "tokens": 0, "drafts": 0}
        self.carry_mismatches = 0
        self.bad_drafts = 0
        # Rows of each (context bucket, known drafts): [a - 1] counts the rows with a accepted rows.
        self.accept_hist: dict[tuple[int, int], list[int]] = {}
        self.failures: dict[str, str] = {}
        self._unreported: dict[str, str] = {}
        self._lock = threading.Lock()
        self._generations = itertools.count()
        self._steps = itertools.count()

    def drop_requests(self, dropped: Collection[str], finished: Collection[str] = ()) -> None:
        """Drop the host state of finished, preempted and resumed requests; a resumed request starts again fresh.
        ``finished`` also clears their failures."""
        finished = set(finished)
        with self._lock:
            for req_id in set(dropped) | finished:
                self.requests.pop(req_id, None)
            for req_id in finished:
                self.failures.pop(req_id, None)
                self._unreported.pop(req_id, None)

    def fail_request(self, req_id: str, error: str) -> None:
        """Fail one request: drop its host state and keep its error for the report."""
        with self._lock:
            self._fail(req_id, error)

    def take_failures(self) -> dict[str, str]:
        """Return the errors of the failed requests that no output has reported yet, once."""
        with self._lock:
            failures, self._unreported = self._unreported, {}
        return failures

    def _fail(self, req_id: str, error: str) -> None:
        """Drop the host state of a failed request and keep its first error; hold the lock."""
        self.requests.pop(req_id, None)
        if req_id in self.failures:
            return
        logger.warning("Kimi-K3 DSpark: request %s failed a host check; the other requests keep running. %s", req_id,
                       error)
        self.failures[req_id] = error
        self._unreported[req_id] = error

    def plan_step(
        self,
        req_ids: Sequence[str | None],
        row0_positions: np.ndarray,
        ring_requests: Collection[str | None],
        last: StepRecord | None = None,
        async_step: bool = False,
        draft_lo: Sequence[int] | None = None,
        context_bucket: int | None = None,
    ) -> StepPlan:
        """Return the host inputs of the speculative step of the replica and record the step.

        A request of ``last`` (the last step of an async host) takes its state from that step on the device. Every
        other request is fresh: a first verify step replays 0 rows from the host's row 0, and a later one takes its
        harvested state. ``spec_kda_state_position`` of a fresh row is verify row 0 of its previous step. A padding
        row gets 0 for each.

        Args:
            req_ids: the request of each batch row of the replica (B rows); None for padding.
            row0_positions: ``[B]`` the host's position of verify row 0 of each row (0 for padding): exact for a fresh
                row, up to k past the true row 0 otherwise.
            ring_requests: the requests with a drafter context row before this step; a request without one restarts
                its drafter context at verify row 0.
            last: the record of the last step, whose device outputs the rows of its requests take, or None.
            async_step: the host has no drafts of the last step of a request, so a fresh row takes the ids of its
                last step from the host.
            draft_lo: ``[B]`` the first valid drafter context position of each row; None keeps ``ctx_start``.
            context_bucket: the context of the step in tokens; None gives ``max_model_len``.

        A fresh request that does not start at the row 0 of its last step plus its replay length, or whose last step
        has no output on the host, fails (``failures``); its row replays 0 rows from the host's row 0.
        """
        k = self.num_speculative_tokens
        batch = len(req_ids)
        fresh = np.ones(batch, dtype=np.int32)
        prev_row = np.zeros(batch, dtype=np.int32)
        replay = np.zeros(batch, dtype=np.int32)
        state_position = np.zeros(batch, dtype=np.int32)
        known = np.zeros(batch, dtype=np.int32)
        lower = None if draft_lo is None else np.asarray(draft_lo, dtype=np.int64)
        draft_lo = np.zeros(batch, dtype=np.int32)
        sched_row0 = np.asarray(row0_positions, dtype=np.int64).reshape(-1).copy()
        last_rows = last.rows() if last is not None else {}
        generations, prev_steps = [-1] * batch, [-1] * batch
        host_ids: dict[int, np.ndarray] = {}
        with self._lock:
            step = next(self._steps)
            for row, req_id in enumerate(req_ids):
                if req_id is None:
                    continue
                state = self.requests.get(req_id)
                if state is None:
                    state = RequestState(next(self._generations))
                    self.requests[req_id] = state
                generations[row], prev_steps[row] = state.generation, state.submitted
                known[row] = self.known_drafts.get(req_id, k)
                source = last_rows.get((req_id, state.generation))
                if source is not None and last.step == state.submitted:
                    fresh[row], prev_row[row] = 0, source
                else:
                    position = int(sched_row0[row])
                    replay[row], state_position[row], error = self._fresh_inputs(req_id, state, position)
                    if error is not None:
                        self._fail(req_id, error)
                    if state.ctx_start is None or req_id not in ring_requests:
                        state.ctx_start = position
                    if async_step and state.next_ids is not None:
                        host_ids[row] = state.next_ids
                draft_lo[row] = state.ctx_start if lower is None else lower[row]
                state.submitted = step
        record = StepRecord(
            step,
            list(req_ids),
            generations,
            prev_steps,
            sched_row0,
            known.copy(),
            fresh.copy(),
            context_bucket=int(context_bucket or self.max_model_len),
        )
        return StepPlan(record, fresh, prev_row, replay, state_position, known, sched_row0 - k, draft_lo, host_ids)

    def check_carried_rows(self, plan: StepPlan, last: StepRecord | None) -> int:
        """Count the carried rows of a step whose source row in the last step is not their request."""
        req_ids = plan.record.req_ids
        last_ids = [] if last is None else list(last.req_ids)
        bad = [
            row
            for row, req_id in enumerate(req_ids)
            if req_id is not None
            and not plan.fresh[row]
            and (req_id not in last_ids or int(plan.prev_row[row]) != last_ids.index(req_id))
        ]
        if not bad:
            return 0
        with self._lock:
            first = self.carry_mismatches == 0
            self.carry_mismatches += len(bad)
        if first:
            logger.warning(
                "Kimi-K3 DSpark: the carried rows %s take their state from a row of another request. req_ids=%s "
                "last.req_ids=%s prev_row=%s fresh=%s",
                bad,
                list(req_ids),
                last_ids if last is not None else None,
                plan.prev_row.tolist(),
                plan.fresh.tolist(),
            )
        return len(bad)

    def step_tensors(
        self, plan: StepPlan, generators: Sequence[torch.Generator | None] = ()
    ) -> dict[str, torch.Tensor]:
        """Return the host ``spec_*`` inputs of a step; the outputs of the last step come from the device.

        ``generators`` holds the generator of each request row with a seed, else None (forced / synthetic counts).
        """
        k = self.num_speculative_tokens
        values = {
            "fresh": plan.fresh,
            "prev_row": plan.prev_row,
            "kda_replay_len": plan.replay,
            "kda_state_position": plan.state_position,
            "known": plan.known,
            "window_base": plan.window_base.astype(np.int32),
            "position_window": plan.position_window(slot_window_width(k), self.max_model_len),
            "draft_lo": plan.draft_lo,
        }
        tensors = {f"spec_{name}": torch.from_numpy(v) for name, v in values.items()}
        tensors["spec_rope_window"] = host_rope_window(plan.window_base, rope_window_width(k))
        if self.live_rows:
            tensors[f"spec_{SPEC_LIVE_ROWS_INPUT}"] = torch.tensor(
                [[int(req_id is not None)] for req_id in plan.record.req_ids], dtype=torch.int32
            ).reshape(-1, 1)
        if self.forced:
            counts = (
                np.full(len(plan.known), -1, dtype=np.int32)
                if self.synthetic is None
                else self.synthetic.accepted(plan.known, generators)
            )
            tensors[f"spec_{SPEC_FORCED_INPUT}"] = torch.from_numpy(counts)
        elif self.synthetic is not None:
            tensors[f"spec_{SPEC_SYNTHETIC_INPUT}"] = torch.from_numpy(self.synthetic.accepted(plan.known, generators))
        return tensors

    def _fresh_inputs(self, req_id: str, state: RequestState, position: int) -> tuple[int, int, str | None]:
        """Return the replay length and the state position of a fresh row, and the error of its row 0 check; a row
        that fails the check replays 0 rows from ``position``."""
        if state.submitted < 0:
            return 0, position, None
        if state.harvested != state.submitted:
            return (
                0,
                position,
                f"Request {req_id} has no output of its last verify step (step {state.submitted}) on the host. A "
                "fresh row needs that output. Read the output of the last step back before the step.",
            )
        expected = state.row0 + state.replay_len
        if position != expected:
            return (
                0,
                position,
                f"Request {req_id} starts a verify step at position {position}. Its KDA state needs position "
                f"{expected}: row 0 of the last step ({state.row0}) plus its {state.replay_len} accepted rows. The "
                "scheduler and the decode count different accepted tokens.",
            )
        return state.replay_len, state.row0, None

    def harvest(self, record: StepRecord, row0: np.ndarray, accepted: np.ndarray, next_ids: np.ndarray) -> None:
        """Store the outputs of a verify step for each request and check them.

        Args:
            record: the record of the step (``plan_step``).
            row0: ``[B]`` row 0 of each row, from the device.
            accepted: ``[B]`` the capped accepted rows of each row.
            next_ids: ``[B, Q]`` the bonus token and the next drafts of each row.

        A request whose row 0 is not row 0 of the last step plus its accepted rows, whose row 0 is outside the verify
        window, or whose accepted count is out of range fails; its output changes no state.
        """
        with self._lock:
            rows = self.stats["rows"]
            for row, req_id in enumerate(record.req_ids):
                if req_id is None:
                    continue
                state = self.requests.get(req_id)
                if state is None or state.generation != record.generations[row] or record.step <= state.harvested:
                    continue
                error = self._check_output(req_id, state, record, row, int(row0[row]), int(accepted[row]))
                if error is not None:
                    self._fail(req_id, error)
                    continue
                state.row0, state.replay_len = int(row0[row]), int(accepted[row])
                state.next_ids = np.array(next_ids[row], dtype=np.int32)
                if self.vocab_size is not None:
                    drafts = state.next_ids[1:]
                    self.bad_drafts += int(((drafts < 0) | (drafts >= self.vocab_size)).sum())
                state.harvested = record.step
                known = int(record.known[row])
                self.stats["rows"] += 1
                self.stats["tokens"] += state.replay_len
                self.stats["drafts"] += known
                counts = self.accept_hist.setdefault((record.context_bucket, known), [0] * (known + 1))
                counts[state.replay_len - 1] += 1
            # A second harvest of a record adds no step.
            if self.stats["rows"] != rows:
                self.stats["steps"] += 1

    def _check_output(
        self, req_id: str, state: RequestState, record: StepRecord, row: int, row0: int, accepted: int
    ) -> str | None:
        """Return the error when an output of the device disagrees with the host state of the request."""
        lead = int(record.sched_row0[row]) - row0
        # A fresh row has the exact row 0 of the host; a carried row lags it by its rejected drafts.
        limit = 0 if record.fresh[row] else self.num_speculative_tokens
        if not 0 <= lead <= limit:
            return (
                f"Request {req_id} ran verify step {record.step} at row 0 {row0}, which is {lead} rows below the "
                f"host's row 0 {int(record.sched_row0[row])}. The row may be 0 to {limit} rows below it. The "
                "scheduler and the decode count different accepted tokens."
            )
        if not 1 <= accepted <= 1 + int(record.known[row]):
            return (
                f"Request {req_id} accepted {accepted} rows in verify step {record.step} for "
                f"{int(record.known[row])} known drafts. The program must cap the acceptance."
            )
        previous = record.prev_steps[row]
        if previous < 0 or state.harvested != previous:
            return None
        expected = state.row0 + state.replay_len
        if row0 != expected:
            return (
                f"Request {req_id} ran verify step {record.step} at row 0 {row0}. Its KDA state needs row 0 "
                f"{expected}: row 0 of the last step ({state.row0}) plus its {state.replay_len} accepted rows. The "
                "device replay length is wrong."
            )
        return None
