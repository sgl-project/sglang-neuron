# SPDX-License-Identifier: Apache-2.0
"""Device-side inputs and state of one async MTP decode step (see mtp/graph.py).

A step scores ``spec`` rows per kernel slot: the slot's next token and the previous step's 7 drafts. Only the device
knows how many drafts were accepted, so everything that depends on it (row positions, cache write rows, the residency
cut) is built here from per-slot device state, and the host sends only what it already knows:

- ``carry_tokens [slots, spec]``: each slot's next input tokens, the bonus or correction token and then the drafts.
- ``slot_len [slots]``: each slot's committed length. Rejected rows are not rolled back; the next step overwrites them.
- ``ctrl [slots, CTRL_COLUMNS]`` (from the host): whether a slot is fresh and, if so, its first token and position. A
  fresh slot's first step scores that one token, exactly like the decode without MTP.

Because the length lives on the device, the host never needs it and can run any number of steps ahead.
"""

from __future__ import annotations

import torch

# host control columns; CTRL_SYNTH_KEPT carries the benchmark's kept-draft count (mtp.graph.SYNTHETIC_AL_ENV)
CTRL_FRESH, CTRL_BASE, CTRL_TOKEN, CTRL_SYNTH_KEPT = range(4)
CTRL_COLUMNS = 4
# a fresh slot's first step scores one row, like the decode without MTP
FRESH_ROWS = 1


def slot_inputs(slot_rows: torch.Tensor, ctrl: torch.Tensor, carry_tokens: torch.Tensor, slot_len: torch.Tensor,
                context: int) -> dict:
    """Each slot's rows for this step.

    ``slot_rows`` is each slot's batch row, -1 when empty. A live slot that isn't fresh verifies its carry from its
    device length; a fresh slot scores the host's token at the host's position. Rows at or past ``context`` are
    not scored.

    Returns ``tokens [S, spec]`` and ``base``, ``count``, ``verify``, ``live`` (each ``[S]``).
    """
    spec = carry_tokens.shape[1]
    live = slot_rows >= 0
    fresh = (ctrl[:, CTRL_FRESH] > 0) & live
    j = torch.arange(spec, dtype=torch.int64, device=ctrl.device).reshape(1, spec)
    host_tokens = torch.where(j == 0, ctrl[:, CTRL_TOKEN:CTRL_TOKEN + 1].to(torch.int64), torch.zeros_like(j))
    tokens = torch.where(fresh.reshape(-1, 1), host_tokens, carry_tokens.to(torch.int64))
    base = torch.where(fresh, ctrl[:, CTRL_BASE].to(torch.int64), slot_len.to(torch.int64))
    count = torch.where(fresh, torch.full_like(base, FRESH_ROWS), torch.full_like(base, spec))
    count = torch.minimum(count, torch.clamp(context - base, min=0))
    count = torch.where(live, count, torch.zeros_like(count))
    return {"tokens": tokens.to(torch.int32), "base": base.to(torch.int32), "count": count.to(torch.int32),
            "verify": live & ~fresh, "live": live & (count > 0)}


def synthetic_keep(kept: torch.Tensor, spec: int) -> torch.Tensor:
    """Expand each slot's kept-draft count ``kept [S]`` into the ``[S * spec]`` mask ``acceptance`` takes."""
    k = kept.to(torch.int64).reshape(-1, 1)
    j = torch.arange(spec, dtype=torch.int64, device=kept.device).reshape(1, spec)
    return (j < k).reshape(-1).to(torch.int32)


def row_positions(base: torch.Tensor, count: torch.Tensor, spec_offsets: torch.Tensor):
    """Row ``j`` sits at ``base + j`` while ``j < count``; rows past the count reuse row 0's position."""
    valid = spec_offsets < count
    return base + spec_offsets * valid.to(spec_offsets.dtype), valid


def rope_rows(cos_table: torch.Tensor, sin_table: torch.Tensor, rows: torch.Tensor):
    """RoPE rows ``[spec, rope / 2]`` at each row's position, clamped to the tables."""
    index = torch.clamp(rows.to(torch.int64), 0, cos_table.shape[0] - 1)
    return cos_table.index_select(0, index), sin_table.index_select(0, index)


def latent_slots(geo, rows: torch.Tensor, valid: torch.Tensor, live: torch.Tensor, cp_rank: torch.Tensor):
    """Latent cache row of each row's key, ``[num_seqs_local, spec]``, -1 for none.

    Uses the static identity block table. Only the CP owner of a position writes it, and only for real rows
    inside the slot's blocks, so a padding row never writes a key that the next step would score.
    """
    logical = torch.div(rows, geo.block_size * geo.cp_degree, rounding_mode="floor")
    owner = torch.div(rows, geo.block_size, rounding_mode="floor") % geo.cp_degree
    mine = (owner == cp_rank) & valid & live & (logical < geo.local_blocks_per_seq)
    slot = logical * geo.block_size + rows % geo.block_size
    written = torch.where(mine, slot, torch.full_like(slot, -1))
    return written.to(torch.int32).reshape(geo.num_seqs_local, geo.spec)


def indexer_slots(geo, rows: torch.Tensor, valid: torch.Tensor, live: torch.Tensor, key_rank: torch.Tensor):
    """Indexer cache row of each row's key, ``[spec]``, -1 for none.

    Global block ``g`` lives on key rank ``g % idx_shard_degree`` at local block ``g // idx_shard_degree``.
    """
    bs = geo.idx_block_size
    g = torch.div(rows, bs, rounding_mode="floor")
    owned = (g % geo.idx_shard_degree) == key_rank
    local = torch.div(g, geo.idx_shard_degree, rounding_mode="floor") * bs + rows % bs
    inside = owned & (rows >= 0) & (local < geo.idx_keys) & valid & live
    return torch.where(inside, local, torch.full_like(local, -1)).to(torch.int32)


def threshold(base: torch.Tensor, live: torch.Tensor, spec: int):
    """Residency cut ``[spec]``: the step's first position, or -1 for an empty slot.

    The indexer only ranks keys written by earlier steps; this step's own keys reach each row through the causal
    sibling band.
    """
    return torch.where(live, base, torch.full_like(base, -1)).expand(spec).contiguous().to(torch.int32)


def acceptance(slot: dict, target: torch.Tensor, synthetic_keep: torch.Tensor | None = None,
               forced: torch.Tensor | None = None) -> dict:
    """Accept drafts and build the draft chain's inputs.

    ``target [S * spec]`` is the target's token for every row (greedy, or the row's sampled token). Draft ``j + 1``
    is kept while every earlier draft was kept and it equals the target's token of row ``j``. A fresh slot keeps
    its single row. ``last`` is a slot's last kept row and ``next_token`` the target's token there.

    ``synthetic_keep`` (benchmark only) keeps the host's count of drafts whatever their tokens, and ``forced``
    (logit test only) keeps every draft of the marked slots.

    Returns the chain inputs ``mtp_tokens``, ``slot_base``, ``slot_last`` and ``slot_live`` (zero for slots that
    did not run), plus ``target``, ``last``, ``verify``, ``live``, ``next_token`` and ``drafts_in`` for
    ``step_outputs`` and ``commit``.
    """
    if forced is not None and synthetic_keep is not None:
        raise ValueError("the acceptance takes the benchmark's synthetic keep or the logit test's forced slots, not "
                         "both")
    tokens = slot["tokens"].to(torch.int64)
    num_slots, spec = tokens.shape
    live, count = slot["live"], slot["count"].to(torch.int64)
    verify = slot["verify"] & live
    t = target.to(torch.int64).reshape(num_slots, spec)
    j = torch.arange(spec, dtype=torch.int64, device=tokens.device).reshape(1, spec)
    drafts_in = torch.cat([tokens[:, 1:], torch.zeros_like(tokens[:, :1])], dim=1)
    agree = drafts_in == t
    if forced is not None:
        agree = agree | forced.reshape(-1, 1)
    if synthetic_keep is not None:
        agree = synthetic_keep.to(torch.int64).reshape(num_slots, spec) > 0
    match = (agree & (j + 1 < count.reshape(-1, 1))).to(torch.float32)
    accepted = torch.cumprod(match[:, : spec - 1], dim=1).sum(dim=1).to(torch.int64)
    last = torch.clamp(torch.where(verify, accepted, count - 1), 0, spec - 1)
    next_token = torch.gather(t, 1, last.reshape(-1, 1)).reshape(-1)
    at = last.reshape(-1, 1)
    mtp_tokens = torch.where(j < at, drafts_in, torch.where(j == at, next_token.reshape(-1, 1), torch.zeros_like(t)))
    zero = torch.zeros_like(last)
    return {"mtp_tokens": torch.where(live.reshape(-1, 1), mtp_tokens, torch.zeros_like(mtp_tokens)).to(torch.int32),
            "slot_base": torch.where(live, slot["base"].to(torch.int64), zero).to(torch.int32),
            "slot_last": torch.where(live, last, zero).to(torch.int32), "slot_live": live.to(torch.int32),
            "target": t, "last": last, "verify": verify, "live": live, "next_token": next_token,
            "drafts_in": drafts_in}


def commit(state: dict, slot: dict, slot_len: torch.Tensor) -> torch.Tensor:
    """Each slot's new committed length: ``base + last + 1`` if it ran, otherwise unchanged."""
    ran = state["live"]
    return torch.where(ran, slot["base"].to(torch.int64) + state["last"].to(torch.int64) + 1,
                       slot_len.to(torch.int64)).to(torch.int32)


__all__ = ["CTRL_FRESH", "CTRL_BASE", "CTRL_TOKEN", "CTRL_SYNTH_KEPT", "CTRL_COLUMNS", "FRESH_ROWS", "slot_inputs",
           "synthetic_keep", "row_positions", "rope_rows", "latent_slots", "indexer_slots", "threshold", "acceptance",
           "commit"]
