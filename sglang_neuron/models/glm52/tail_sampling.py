# SPDX-License-Identifier: Apache-2.0
"""On-device sampling for the GK decode: the ``[top_k, top_p, temperature, u]`` rows its LM-head tail samples with.

The decode samples inside the GK's LM-head tail (top_p rule ``keep_crossing``, 256 candidates a row);
``sampling.tail_rule_sample`` is the same rule in torch, used as the tests' reference. Sampling on the device keeps
the decode asynchronous: the per-row parameters are already staged on the device, and ``u`` is derived from the
request seed and the row's input position, so a step needs no host work or readback.

Because the draw for position ``p + 1`` depends only on ``(seed, p)``, a seeded request repeats under any batch order
or preemption, the prefill draws the first token with the same function, and with MTP every verify row ``j`` draws
with key ``(seed, base + j)``, so the accepted tokens have the same distribution as one-token-at-a-time sampling.
"""

from __future__ import annotations

import torch

from sglang_neuron import sampling


def filler_rows(rows: int, k: int) -> torch.Tensor:
    """Rows that sample nothing: greedy ``[k, 1, 0, 0]``."""
    return torch.tensor([[float(k), 1.0, 0.0, 0.0]] * rows, dtype=torch.float32)


def request_params(row: torch.Tensor, temperatures: torch.Tensor, seeds: torch.Tensor, filters: torch.Tensor,
                   columns: torch.Tensor, k: int):
    """One request's ``(top_k, top_p, temperature, seed)`` from the platform's per-batch-row inputs, top_k clipped
    to ``[1, k]``. ``row`` is the batch row (-1 for an empty slot, which then must not be live).
    """
    r = torch.clamp(row, min=0).to(torch.int64)
    temperature = temperatures.index_select(0, r).to(torch.float32).reshape(1, 1)
    seed = seeds.index_select(0, r).to(torch.int32).reshape(1)
    picked = filters.index_select(0, r).to(torch.float32)  # [1, 3]
    top_k = torch.clamp(picked.index_select(1, columns[0:1]), min=1.0, max=float(k))
    top_p = picked.index_select(1, columns[1:2])
    return top_k, top_p, temperature, seed


def gk_tail_rows_at(positions: torch.Tensor, live: torch.Tensor, top_k: torch.Tensor, top_p: torch.Tensor,
                    temperature: torch.Tensor, seed: torch.Tensor, filler: torch.Tensor) -> torch.Tensor:
    """Tail rows ``[n, 4]`` for one request at the given input positions; rows that are not ``live`` take ``filler``.

    Row ``i`` samples the token at ``positions[i] + 1`` with key ``(seed, positions[i])``, so an MTP verify row
    draws exactly what a step without the draft would.
    """
    n = filler.shape[0]
    draws = sampling.row_draws(sampling.noise_row_keys(seed.expand(n), positions.to(torch.int32))).reshape(-1, 1)
    request = torch.cat([top_k.expand(n, 1), top_p.expand(n, 1), temperature.expand(n, 1), draws], dim=1)
    return torch.where(live.to(torch.float32) > 0, request, filler)


def gk_tail_rows(row: torch.Tensor, live: torch.Tensor, positions: torch.Tensor, spec_rows: torch.Tensor,
                 temperatures: torch.Tensor, seeds: torch.Tensor, filters: torch.Tensor, filler: torch.Tensor,
                 columns: torch.Tensor, k: int) -> torch.Tensor:
    """One rank's tail rows ``[t_shard, 4]`` for the decode without MTP.

    ``live`` marks the rank's spec row that carries the sequence's token, and ``row`` the sequence's batch row (-1
    when the slot is empty). ``temperatures``, ``seeds`` and ``filters`` are the platform's per-batch-row sampling
    inputs; ``columns`` picks top_k and top_p out of ``filters`` with an index_select, since a strided column view
    does not lower on Neuron.
    """
    top_k, top_p, temperature, seed = request_params(row, temperatures, seeds, filters, columns, k)
    r = torch.clamp(row, min=0).to(torch.int64)
    position = positions.index_select(0, r).to(torch.int32) + spec_rows.to(torch.int32)  # [t_shard]
    use = live.to(torch.float32) * (row >= 0).to(torch.float32).reshape(1, 1)
    return gk_tail_rows_at(position, use, top_k, top_p, temperature, seed, filler)


#: Columns of the MTP step's per-slot request table: top_k, top_p, temperature.
SLOT_PARAM_COLUMNS = 3


def slot_request_row(sampling_params, seed: int) -> tuple[list[float], int]:
    """A request's row of the MTP step's per-slot table: ``([top_k, top_p, temperature], folded seed)``.

    SGLang's greedy rule (top_k 1, or a temperature below GREEDY_TEMPERATURE) maps to temperature 0, and an
    unrestricted top_k to TOP_K_ALL.
    """
    top_k = int(getattr(sampling_params, "top_k", -1) or -1)
    temperature = float(getattr(sampling_params, "temperature", 1.0))
    if top_k == 1 or temperature < sampling.GREEDY_TEMPERATURE:
        temperature = 0.0
    if top_k < 1:
        top_k = sampling.TOP_K_ALL
    top_p = float(getattr(sampling_params, "top_p", 1.0))
    folded = int(sampling.fold_seeds(torch.tensor([int(seed)], dtype=torch.int64))[0])
    return [float(top_k), top_p, temperature], folded


def slot_params(table: torch.Tensor, seeds: torch.Tensor, slot: torch.Tensor, columns: torch.Tensor, k: int):
    """The MTP graph's version of ``request_params``: read this rank's slot from the per-slot table and seeds."""
    picked = table.index_select(0, slot.to(torch.int64)).to(torch.float32)  # [1, 3]
    top_k = torch.clamp(picked.index_select(1, columns[0:1]), min=1.0, max=float(k))
    return (top_k, picked.index_select(1, columns[1:2]), picked.index_select(1, columns[2:3]),
            seeds.index_select(0, slot.to(torch.int64)).to(torch.int32).reshape(1))
