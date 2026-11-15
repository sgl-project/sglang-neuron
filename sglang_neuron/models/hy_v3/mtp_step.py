# SPDX-License-Identifier: Apache-2.0
"""Hy3's NEURON_MTP decode step (catalog.ModelSpec.mtp_host; speculative/neuron_mtp.py): one device step verifies a
row's previous drafts, accepts, and drafts the next ones, which stay on the device.

A step decodes ``SPEC = NUM_STEPS + 1`` tokens a row: its next input token (the previous step's bonus) and the
previous step's drafts. Only the device knows how many it accepted, so the decode graph keeps each request slot's
carry (``mtp_carry [slots + 1, SPEC]``, the next step's tokens, and ``mtp_len [slots + 1]``, the committed length) and
builds every position, cache slot and mask from it. The host sends, a batch row each (``CTRL_*``), only what it knows:
whether the row is live, and for a request new to its slot (fresh: its first step after its prefill) the position and
the token its first step scores, which it scores alone, like the decode without MTP. The last carry row is the
padding rows' (``row_slots`` sends them there).

Under DCP a span's tokens can sit on different ranks: each is gated by its own owner, and the span shares the prior of
its first token (gen_attention_decode_mask's per-token gate). Every token of a row is checked against the target's
sample at the position before it; the row keeps the drafts up to the first mismatch and the target's next token after
them, so the output is the target's own sample run.

TODO(mtp): the MTP layer drafts (mtp.py); until then every draft is 0 and a step keeps one token a row."""

from __future__ import annotations

import torch

NUM_STEPS = 3
SPEC = NUM_STEPS + 1
PREFILL_SETTING = None
# an aggregated server: the target's prefill graph writes the MTP layer's KV (mtp.py)
AGGREGATED = True
CTRL_LIVE, CTRL_FRESH, CTRL_BASE, CTRL_TOKEN = range(4)
CTRL_COLUMNS = 4


def launch_problems(server_args) -> list:
    return []


def launch_warnings(server_args) -> list:
    return []


def slot_of(req_pool_idx: int, slots: int) -> int:
    """The carry slot of SGLang request row ``req_pool_idx`` (row 0 is SGLang's padding row)."""
    slot = int(req_pool_idx) - 1
    if not 0 <= slot < slots:
        raise ValueError(f"request row {req_pool_idx} has no MTP slot among {slots}")
    return slot


def host_rows(batch, slots_of_rows: list, fresh: list, width: int, num_slots: int) -> tuple:
    """(``row_slots [width]``, ``ctrl [width, CTRL_COLUMNS]``) int32 of a step: each batch row's carry slot (the padding
    slot ``num_slots`` past the batch) and control. A fresh row's token is the prefill's sample at the position after
    its prompt: from the host once its prefill's result was processed, else from the token relay."""
    from sglang_neuron.model_executor.async_execution import relay_prefill_token

    row_slots = torch.full((width,), num_slots, dtype=torch.int32)
    ctrl = torch.zeros(width, CTRL_COLUMNS, dtype=torch.int32)
    generations = batch.req_to_token_pool.req_generation
    indices = batch.req_pool_indices.to("cpu").reshape(-1).tolist()
    for row, (req, slot, new) in enumerate(zip(batch.reqs, slots_of_rows, fresh)):
        row_slots[row] = slot
        ctrl[row, CTRL_LIVE] = 1
        if new:
            ctrl[row, CTRL_FRESH] = 1
            if len(req.output_ids):
                ctrl[row, CTRL_BASE] = len(req.origin_input_ids) + len(req.output_ids) - 1
                ctrl[row, CTRL_TOKEN] = int(req.output_ids[-1])
            else:
                index = int(indices[row])
                token = relay_prefill_token(index, int(generations[index]))
                if token is None:
                    raise RuntimeError(f"no first token for request {req.rid}: its prefill relayed none")
                ctrl[row, CTRL_BASE] = len(req.origin_input_ids)
                ctrl[row, CTRL_TOKEN] = token
    return row_slots, ctrl


def span_inputs(carry: torch.Tensor, length: torch.Tensor, ctrl: torch.Tensor) -> dict:
    """Each row's tokens, positions and live tokens this step: ``tokens``, ``positions`` and ``valid`` [W, SPEC],
    ``base``, ``count``, ``fresh`` and ``live`` [W]. A token past a row's count reuses the row's first position."""
    width = ctrl.shape[0]
    live = ctrl[:, CTRL_LIVE] > 0
    fresh = (ctrl[:, CTRL_FRESH] > 0) & live
    j = torch.arange(SPEC, dtype=torch.int64, device=ctrl.device).reshape(1, SPEC)
    host = torch.where(j == 0, ctrl[:, CTRL_TOKEN:CTRL_TOKEN + 1].to(torch.int64), torch.zeros_like(j))
    tokens = torch.where(fresh.reshape(-1, 1), host, carry.to(torch.int64))
    base = torch.where(fresh, ctrl[:, CTRL_BASE].to(torch.int64), length.to(torch.int64))
    count = torch.where(fresh, torch.ones_like(base), torch.full_like(base, SPEC))
    count = torch.where(live, count, torch.zeros_like(count))
    valid = j < count.reshape(width, 1)
    positions = torch.where(valid, base.reshape(width, 1) + j, base.reshape(width, 1).expand(width, SPEC))
    return {"tokens": tokens, "positions": positions, "valid": valid, "base": base, "count": count,
            "fresh": fresh, "live": live}


def dcp_span(positions: torch.Tensor, valid: torch.Tensor, base: torch.Tensor, table: torch.Tensor, size: int,
             rank: torch.Tensor, block: int) -> dict:
    """This rank's share of each row's span under DCP (distributed.decode_context_parallel.DcpLayout): ``owner``
    [W, SPEC] 1 where it keeps the token, ``update_idx`` [W, SPEC] its pool slot (NULL_SLOT elsewhere), ``local_pos``
    [W, SPEC] the token's place among this rank's tokens and ``local_prior`` [W] the prior tokens it keeps. ``table``
    [W, pages] holds each row's virtual pages, which are this rank's block ids."""
    from sglang_neuron.distributed.decode_context_parallel import NULL_SLOT
    from sglang_neuron.models.hy_v3.sparse import local_count

    vpage = size * block
    col = torch.clamp(torch.div(positions, vpage, rounding_mode="floor"), max=table.shape[1] - 1)
    page = table.to(torch.int64).gather(1, col)
    rank = rank.to(torch.int64).reshape(-1)[0]
    owner = (torch.div(positions, block, rounding_mode="floor") % size == rank) & valid & (page >= 0)
    slot = page * block + positions % block
    update_idx = torch.where(owner, slot, torch.full_like(slot, NULL_SLOT))
    rank_t = rank.reshape(1)
    local_pos = local_count(positions.reshape(-1), size, rank_t, block).reshape(positions.shape)
    local_prior = local_count(base, size, rank_t, block)
    return {"owner": owner.to(torch.int32), "update_idx": update_idx, "local_pos": local_pos,
            "local_prior": local_prior}


def accept(tokens: torch.Tensor, sampled: torch.Tensor, span: dict) -> tuple:
    """(``out [W, SPEC]``: each row's kept run, -1 after it; ``kept [W]``: the drafts it kept). A draft is kept while it
    equals the target's sample at the position before it."""
    match = (tokens[:, 1:] == sampled[:, :-1]) & span["valid"][:, 1:] & ~span["fresh"].reshape(-1, 1)
    kept = torch.cumprod(match.to(torch.int64), dim=1).sum(dim=1)
    j = torch.arange(SPEC, dtype=torch.int64, device=tokens.device).reshape(1, SPEC)
    out = torch.where(j <= kept.reshape(-1, 1), sampled.to(torch.int64), torch.full_like(sampled, -1, dtype=torch.int64))
    out = torch.where(span["live"].reshape(-1, 1), out, torch.full_like(out, -1))
    return out, kept


class StepHost:
    """The host side of a step (speculative/neuron_mtp.NeuronMtpWorker): the decode runtime's ``run_mtp_step``."""

    def __init__(self, target_worker):
        self.target_worker = target_worker

    def _runtime(self):
        model = self.target_worker.model_runner.model
        runtimes = getattr(model, "phase_runtimes", None)
        runtime = runtimes["decode"] if runtimes is not None else getattr(model, "runtime", None)
        if runtime is None or not hasattr(runtime, "run_mtp_step"):
            raise RuntimeError(f"Hy3's MTP step runs on its decode runtime, not {type(runtime).__name__}")
        return runtime

    def slot(self, req_pool_idx: int) -> int:
        return slot_of(req_pool_idx, self._runtime().mtp_num_slots)

    def step(self, batch, slots: list, fresh: list):
        return self._runtime().run_mtp_step(batch, slots, fresh)
