# SPDX-License-Identifier: Apache-2.0
"""Logit test mode for GLM-5.2: teacher forcing and raw logits on the served 1P1D (test only).

The logit test compares the served pair's logits for 50 teacher-forced tokens per prompt against the HF
GPU goldens. With ``LOGIT_TEST_DIR_ENV`` set:

* The decode builds ``GlmGkMtpLogitTestGraph``. A request that names golden tokens keeps every draft, emits and
  carries the golden tokens, and its slot's full-vocabulary logits are written under the directory, one file a step.
* The prefill's last stage writes the logits of each request's sampled row.

A forced request's prefill emits golden token 0; the decode's first step scores row 1 and each following step the
next 8 rows, so 50 tokens take the prefill and 7 decode steps.

Without the setting nothing here is traced or read, and a request naming golden tokens is refused at admission.
"""

from __future__ import annotations

import logging
import os
import pathlib
from typing import Optional

import torch

logger = logging.getLogger(__name__)

#: Directory where both servers of a logit test pair write their logits; unset on a served launch.
LOGIT_TEST_DIR_ENV = "SGLANG_NEURON_GLM52_LOGIT_TEST_DIR"
#: ``custom_params`` key holding a request's golden token ids.
FORCED_TOKENS_PARAM = "neuron_forced_token_ids"
#: Fill value for unforced slots.
UNFORCED = -1


def test_dir() -> Optional[pathlib.Path]:
    """The launch's logit test directory, or None for a served launch."""
    raw = os.environ.get(LOGIT_TEST_DIR_ENV, "").strip()
    return pathlib.Path(raw) if raw else None


def warn_on(what: str) -> None:
    """Log that this launch runs the logit test mode, so an inherited setting is never mistaken for a served launch."""
    logger.warning("%s=%s: the GLM-5.2 logit test mode is on: %s. A served launch never sets it", LOGIT_TEST_DIR_ENV,
                   test_dir(), what)


def _custom(sampling_params) -> dict:
    custom = (sampling_params.get("custom_params") if isinstance(sampling_params, dict)
              else getattr(sampling_params, "custom_params", None))
    return custom if isinstance(custom, dict) else {}


def forced_tokens(sampling_params, vocab_size: int) -> Optional[list]:
    """The request's golden tokens, or None. ``sampling_params`` is the raw dict or SGLang's SamplingParams.

    Raises:
        ValueError: The tokens are not a list of at least 2 ids in ``[0, vocab_size)``.
    """
    tokens = _custom(sampling_params).get(FORCED_TOKENS_PARAM)
    if tokens is None:
        return None
    if (not isinstance(tokens, list) or len(tokens) < 2
            or not all(isinstance(t, int) and not isinstance(t, bool) and 0 <= t < vocab_size for t in tokens)):
        raise ValueError(f"custom_params[{FORCED_TOKENS_PARAM!r}] is a list of 2 or more token ids in "
                         f"[0, {vocab_size}): the golden tokens a logit test forces, token 0 the prefill's")
    return list(tokens)


def validate_request(obj, preferred, server_args, vocab_size: int) -> None:
    """Admit a request that names golden tokens only on a logit test server, unbatched, greedy, ignoring EOS and
    asking for exactly its golden tokens. The decode reads one forced request per step.
    """
    from sglang_neuron import sampling

    raw = getattr(obj, "sampling_params", None)
    if isinstance(raw, (list, tuple)):
        # a batch has one sampling params per request; the decode reads one forced request per step
        if any(_custom(params).get(FORCED_TOKENS_PARAM) is not None for params in raw):
            raise ValueError(f"custom_params[{FORCED_TOKENS_PARAM!r}] in a batched request: a logit test sends each "
                             "forced request alone")
        return
    tokens = forced_tokens(raw if raw is not None else {}, vocab_size)
    if tokens is None:
        return
    if test_dir() is None:
        raise ValueError(f"custom_params[{FORCED_TOKENS_PARAM!r}] forces a logit test's tokens, which only a server "
                         f"launched with {LOGIT_TEST_DIR_ENV} serves")
    get = sampling._request_params(obj, preferred)
    problems = []
    if not sampling.request_is_greedy(obj, preferred):
        problems.append("a sampled request (send temperature 0)")
    if get("max_new_tokens") != len(tokens):
        problems.append(f"max_new_tokens {get('max_new_tokens')} (ask for the {len(tokens)} forced tokens)")
    if not get("ignore_eos", False):
        problems.append("ignore_eos unset (a forced EOS would end the request early)")
    if get("stop") or get("stop_token_ids") or getattr(obj, "return_logprob", False):
        problems.append("stop strings, stop tokens or logprobs")
    if problems:
        raise ValueError("a teacher-forced logit test request is not served with " + "; ".join(problems))


class ForcedTokens:
    """Host-side schedule of the forced requests: each slot's golden tokens and the next golden row it scores."""

    def __init__(self, spec: int):
        self.spec = int(spec)
        self._slots = {}

    def start(self, slot: int, tokens: list) -> None:
        """Start a forced request in ``slot``."""
        self._slots[int(slot)] = {"tokens": list(tokens), "row": 1}

    def stop(self, slot: int) -> None:
        self._slots.pop(int(slot), None)

    def forced(self, slot: int) -> bool:
        return int(slot) in self._slots

    def step(self, slot: int, fresh: bool) -> tuple:
        """``(first_row, rows, window)`` for the slot's next step: the first golden row it scores, how many of its rows
        are golden (fewer or none on steps launched past the request's end), and the next ``2 * spec`` golden tokens.
        """
        state = self._slots[int(slot)]
        first, count = state["row"], 1 if fresh else self.spec
        if fresh != (first == 1):
            raise RuntimeError(f"slot {slot}'s {'fresh' if fresh else 'verify'} step at golden row {first}: a forced "
                               "request's fresh step scores row 1 and only it")
        tokens = state["tokens"]
        window = [tokens[first + i] if first + i < len(tokens) else 0 for i in range(2 * self.spec)]
        state["row"] = first + count
        return first, max(0, min(count, len(tokens) - first)), window


# graph functions, traced only in the logit test graph


def forced_outputs(sampled: torch.Tensor, carry: torch.Tensor, slot: dict, forced: torch.Tensor,
                   row_slots: torch.Tensor):
    """Replace forced slots' emitted tokens and carry with their golden window; unforced slots keep the graph's own."""
    spec = carry.shape[1]
    on = (forced[:, 0] >= 0) & slot["live"]
    last = torch.clamp(slot["count"].to(torch.int64) - 1, min=0).reshape(-1, 1)
    j = torch.arange(spec, dtype=torch.int64, device=forced.device).reshape(1, spec)
    window = forced.to(torch.int64)
    table = torch.where(j <= last, window[:, :spec], torch.full_like(window[:, :spec], -1))
    nxt = torch.gather(window, 1, last + j)
    rows = row_slots.to(torch.int64)
    on_rows = on.index_select(0, rows).reshape(-1, 1)
    emitted = torch.where(on_rows, table.index_select(0, rows), sampled.to(torch.int64))
    carried = torch.where(on.reshape(-1, 1), nxt, carry.to(torch.int64))
    return emitted.to(torch.int32), carried.to(torch.int32)


def forced_keep(forced: torch.Tensor) -> torch.Tensor:
    """``[num_slots]`` bool: the forced slots, which keep every draft."""
    return forced[:, 0] >= 0


def probe_rows(hidden: torch.Tensor, forced: torch.Tensor, my_slot: torch.Tensor, own_rows: torch.Tensor,
               offsets: torch.Tensor, gamma: torch.Tensor, w_t: torch.Tensor, group, eps: float) -> torch.Tensor:
    """The forced slot's ``spec`` logit rows over this rank's vocabulary shard, gathered in rank order.

    Each rank places its own rows of the forced slot and zeros elsewhere; one all-reduce then gives every rank the
    slot's rows exactly. The placement is a select rather than a multiply, so a NaN in a dead slot's row cannot
    leak in. The final norm rounds as the tail does (``bf16(bf16(x * rms) * gamma)``).
    """
    from sglang_neuron.parallel import collectives

    mine = (forced[:, 0] >= 0).index_select(0, my_slot.to(torch.int64)).reshape(1, 1)
    # [spec, t_shard]: which of this rank's rows sits at each slot row
    match = offsets.to(torch.int64).reshape(-1, 1) == own_rows.to(torch.int64).reshape(1, -1)
    local = torch.arange(own_rows.shape[0], dtype=torch.int64, device=hidden.device).reshape(1, -1)
    pick = (match.to(torch.int64) * local).sum(dim=1)  # integer: the row placed at each place (0 where none)
    held = match.any(dim=1).reshape(-1, 1) & mine
    rows = hidden.to(torch.float32).index_select(0, pick)
    x = collectives.all_reduce(group, torch.where(held, rows, torch.zeros_like(rows)))
    rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    normed = ((x * rms).to(torch.bfloat16).to(torch.float32) * gamma).to(torch.bfloat16).to(torch.float32)
    return collectives.all_gather(group, normed @ w_t.to(torch.float32), dim=0)


def probe_tokens(target: torch.Tensor, forced: torch.Tensor, spec: int) -> torch.Tensor:
    """The tail's tokens ``[spec]`` for the forced slot's rows (zeros when none is forced)."""
    table = target.to(torch.int64).reshape(-1, spec)
    on = (forced[:, 0] >= 0).reshape(-1, 1)
    return torch.where(on, table, torch.zeros_like(table)).sum(dim=0).to(torch.int32)


# host helpers


def outputs_of(res) -> dict:
    """A step's graph outputs: an async step's pending outputs, or a synchronous step's host sample."""
    pending = getattr(res, "neuron_pending_sample", None)
    if pending is not None:
        return pending.outputs
    return getattr(res, "neuron_sample", None) or {}


def vocab_rows(gathered: torch.Tensor, ranks: int, rows: int) -> torch.Tensor:
    """Reshape ``[ranks * rows, vocab_shard]`` into ``[rows, vocab]``."""
    shard = gathered.shape[-1]
    return gathered.reshape(ranks, rows, shard).permute(1, 0, 2).reshape(rows, ranks * shard)


def safe_rid(rid) -> str:
    """A request id as one path component."""
    text = str(rid)
    if not text or text in (".", "..") or "/" in text or "\0" in text:
        raise ValueError(f"the logit test names each request's files by its rid, and {text!r} is not a file name")
    return text


def write(directory: pathlib.Path, rid, name: str, payload: dict) -> pathlib.Path:
    """Write ``payload`` to ``directory/<rid>/<name>`` atomically."""
    folder = pathlib.Path(directory) / safe_rid(rid)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    partial = folder / f".{name}.{os.getpid()}.partial"
    torch.save(payload, partial)
    os.replace(partial, path)
    return path


def decode_name(first_row: int) -> str:
    return f"decode_r{int(first_row):02d}.pt"


PREFILL_NAME = "prefill.pt"


def write_decode_rows(directory, rid, first_row: int, count: int, gathered: torch.Tensor, tail_tokens: torch.Tensor,
                      ranks: int, spec: int, slot: int) -> None:
    """Write one forced step's logits ``[count, vocab]``, tail tokens and GK slot (decode rank 0)."""
    rows = vocab_rows(gathered.to("cpu").to(torch.float32), ranks, spec)[:count]
    write(directory, rid, decode_name(first_row),
          {"first_row": int(first_row), "logits": rows.clone(), "tail_tokens": tail_tokens.to("cpu")[:count].clone(),
           "slot": int(slot)})


def write_prefill_row(directory, rid, end: int, logits: torch.Tensor, vocab_size: int) -> None:
    """Write the prefill's logits ``[vocab]`` for a request's sampled row; each chunk overwrites it."""
    row = logits.to("cpu").to(torch.float32).reshape(-1)[:vocab_size]
    write(directory, rid, PREFILL_NAME, {"end": int(end), "logits": row.clone()})
