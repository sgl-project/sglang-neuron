# SPDX-License-Identifier: Apache-2.0
"""Model-independent device sampling and the compiled sampled-output contract.

A compiled model may return a dict with ``token_ids``, ``logprobs``,
``top_vals`` and ``top_ids``. Token IDs remain device tensors until the runtime
chooses to materialize them, so the same contract supports a future-token relay.

The sampler mode is a launch constant (``SGLANG_NEURON_SAMPLER``):

* ``greedy`` (default): exact argmax; requests that are not greedy (temperature
  at or above GREEDY_TEMPERATURE and top_k != 1) are rejected at admission.
* ``temperature``: exact categorical sampling from ``softmax(logits / T)`` over
  the full vocabulary (Gumbel-max), per request; temperature 0 rows select the
  same tokens as ``greedy`` (their logprobs agree to fp32 rounding). Top-k /
  top-p / min-p filtering is not applied on device, so such requests are
  rejected at admission.
* ``topk_topp``: ``temperature`` plus SGLang's top-k / top-p / min-p filters,
  applied to a pool of the global top TOP_K_MAX tokens:
  exact for top_k <= that bound and for nucleus / min-p sets inside the pool
  (a larger nucleus is truncated to the pool).
  A giga-kernel decode may serve the mode in the kernel's own LM-head tail, which samples next to the LM-head
  matmul so no logits leave the device; that tail's rule is ``tail_rule_sample``.
* ``host``: SGLang's own sampler on the full logits the graphs return, so penalties,
  logit bias, grammars, any top-k and logprobs work; each step waits for its logits,
  so the launch runs without async execution (platform.check_host_sampler).

The Gumbel noise of a draw is a pure function of (request seed, position,
global token id): reproducible for a seed, identical on every rank of a
vocabulary group, independent of batch row order and of the vocabulary
sharding.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping

import torch

from sglang_neuron.parallel import collectives as neuron_collectives

MAX_TOP_LOGPROBS = int(os.environ.get("SGLANG_NEURON_MAX_TOP_LOGPROBS", "0"))
# the device sampler's modes; the drawing ones take each row's temperature and seed as graph inputs
SAMPLER_MODES = ("greedy", "temperature", "topk_topp")
DRAWING_MODES = ("temperature", "topk_topp")
# the host sampler: SGLang's own, on the logits the graphs return
HOST_SAMPLER = "host"
# The topk_topp filter pool: the global top TOP_K_MAX tokens per row. 256 holds any top_k a client commonly sends and
# the nucleus of a peaked distribution exactly; its all-gather carries 5 + 2 * 256 values per row per shard.
TOP_K_MAX = 256
# SGLang's normalized "no top-k" value (sampling_params.TOP_K_ALL) and greedy threshold.
TOP_K_ALL = 1 << 30
GREEDY_TEMPERATURE = 1e-6

# Noise hash on 31-bit lanes: odd multipliers (murmur3's fmix32 constants below bit 31) and a
# golden-ratio salt. Every value stays in [0, 2**31), so right shifts never see a negative
# int32: Neuron lowers >> of a negative int32 as a truncating division, torch as an
# arithmetic shift.
_LANE = 0x7FFFFFFF
_MIX_C1 = 0x05EBCA6B
_MIX_C2 = 0x42B2AE35
_COLUMN_SALT = 0x1E3779B9


def sampler_mode() -> str:
    """This server's sampler mode (SGLANG_NEURON_SAMPLER), validated."""
    mode = os.environ.get("SGLANG_NEURON_SAMPLER", "greedy").strip().lower() or "greedy"
    modes = SAMPLER_MODES + (HOST_SAMPLER,)
    if mode not in modes:
        raise ValueError(f"SGLANG_NEURON_SAMPLER={mode!r} is not one of {modes}")
    return mode


def return_original_logprob() -> bool:
    """SGLang's SGLANG_RETURN_ORIGINAL_LOGPROB: return logprobs of the untempered distribution.

    SGLang parses this variable in two places that disagree: its sampler uses get_bool_env_var (only true / 1 are on)
    and its logprob processor uses EnvBool (yes / y are on too). A value like "yes" would turn on one and not the
    other, so only true / 1 / false / 0 (any case) are accepted and anything else raises; platform._check_plugin_env
    refuses it at launch."""
    raw = os.environ.get("SGLANG_RETURN_ORIGINAL_LOGPROB", "false")
    value = raw.lower()
    if value not in ("true", "1", "false", "0"):
        raise ValueError(f"SGLANG_RETURN_ORIGINAL_LOGPROB={raw!r}: set it to true, 1, false or 0 (SGLang's sampler and "
                         "logprob processor parse other values differently), or leave it unset")
    return value in ("true", "1")


def _mix31(h: torch.Tensor) -> torch.Tensor:
    """murmur3-style finalizer on 31-bit int32 lanes: a bijection of [0, 2**31).

    Products wrap modulo 2**32 on both torch and Neuron; masking to the lane keeps them
    modulo 2**31 and non-negative."""
    h = h ^ (h >> 16)
    h = (h * _MIX_C1) & _LANE
    h = h ^ (h >> 13)
    h = (h * _MIX_C2) & _LANE
    return h ^ (h >> 16)


def noise_row_keys(seeds: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Per-row noise key from the request seed (fold_seeds) and the position of the row's input token.

    For one seed the key is a bijection of the position, so no two draws of a request share
    a noise field."""
    return _mix31(_mix31(seeds.to(torch.int32) & _LANE) ^ (positions.to(torch.int32) & _LANE))


def gumbel_noise(row_keys: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Standard Gumbel noise [R, V'] for rows' keys [R] at global token ids (int32): [V'] shared
    by every row, or [R, V'] per row. A token's noise does not depend on which other ids are asked.

    Within a row the hash is a bijection of the token id; its low 23 bits give a uniform in
    (0, 1), bounded away from 0 and 1."""
    columns = token_ids if token_ids.ndim == 2 else token_ids.reshape(1, -1)
    h = _mix31(row_keys.reshape(-1, 1) ^ _mix31(columns ^ _COLUMN_SALT))
    u = ((h & 0x7FFFFF).to(torch.float32) + 0.5) * (1.0 / 8388608.0)
    return -torch.log(-torch.log(u))


# A giga-kernel LM-head tail samples from GK_TAIL_SAMPLE_K candidates a row, and treats a temperature below
# GK_TAIL_GREEDY_TEMPERATURE as greedy.
GK_TAIL_SAMPLE_K = 256
GK_TAIL_GREEDY_TEMPERATURE = 1e-5
# salt so the inverse-transform draw is independent of the Gumbel noise from the same key
_DRAW_SALT = 0x2545F491


def row_draws(row_keys: torch.Tensor) -> torch.Tensor:
    """One uniform draw per row in (0, 1) from its noise key, for a kernel tail's inverse-transform sampler.

    Draws are odd multiples of 2**-24, exact in float32 and never 0 or 1. The key is the request seed and the row's
    input position, so the prefill's first token and every decode step draw the same way under any batch order,
    sharding or MTP acceptance.
    """
    h = _mix31((row_keys.to(torch.int32) & _LANE) ^ _DRAW_SALT)
    return ((h & 0x7FFFFF) * 2 + 1).to(torch.float32) * (1.0 / 16777216.0)


def fold_seeds(seeds: torch.Tensor) -> torch.Tensor:
    """int64 request seeds -> 31-bit int32 lanes: the low word xor the high word, top bit folded in."""
    seeds = seeds.to(torch.int64)
    word = (seeds ^ (seeds >> 32)) & 0xFFFFFFFF
    return ((word ^ (word >> 31)) & _LANE).to(torch.int32)


def request_seed(req) -> int:
    """The int64 seed of a request's draws: its sampling_seed, else a stable hash of its rid (distinct per request and
    per n > 1 copy, each of which has its own rid, and the same on every rank, unlike Python's salted hash()). Under
    --enable-deterministic-inference SGLang seeds rows itself (42)."""
    seed = getattr(req.sampling_params, "sampling_seed", None)
    if seed is not None:
        return int(seed)
    digest = hashlib.blake2b(str(req.rid).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def _max_smallest_id(values, ids, last_id):
    """Row maximum of ``values`` [R, N] and the smallest of ``ids`` holding it, both [R, 1].

    ``ids`` ([N] or [R, N]) are exact float32 integers, none above ``last_id``. Ties are settled by a max and a min
    reduction, which are exact in any reduction order. An index-returning max (``torch.max`` with a dim, ``argmax``)
    cannot settle them on the device: its Neuron lowering does not return the first index of a tie, nor the same index
    in every program. A row with no maximum (NaN) takes ``last_id``, so the result is always a valid id."""
    best = values.amax(dim=-1, keepdim=True)
    smallest = ids.expand_as(values).masked_fill(values != best, float(last_id)).amin(dim=-1, keepdim=True)
    return best, smallest


def _first_max(values):
    """Row maximum of ``values`` [R, N] and the first column holding it (int64), both [R, 1]: torch.max over the last
    dimension with the first-index tie rule on every backend."""
    width = values.shape[-1]
    columns = torch.arange(width, dtype=torch.float32, device=values.device)
    best, first = _max_smallest_id(values, columns, width - 1)
    return best, first.to(torch.int64)


def _shard_winner(per_rank, vocabulary_size):
    """Each row's winning shard from gathered payloads ``per_rank`` [R, world, P] whose columns 0 and 1 hold every
    shard's best score and its global token id: (best score [R, 1], token id [R, 1] float32, winner mask
    [R, world]). Equal scores go to the smallest token id, compared directly rather than through the gather order."""
    best, token = _max_smallest_id(per_rank[:, :, 0], per_rank[:, :, 1], vocabulary_size - 1)
    return best, token, per_rank[:, :, 1] == token


def _exact_topk(logits, k, dim=-1):
    """Small sorted top-k using reductions, with deterministic tie breaking.

    Keep selection explicit instead of lowering torch.topk to AwsNeuronTopK,
    which returns incorrect global candidates after an all-gather.
    Each iteration takes the maximum and the first column holding it by max / min
    reductions (_max_smallest_id, never an index-returning max) and removes only
    that column, so equal-valued tokens remain eligible and the smallest index wins
    a tie.
    """
    if dim not in (-1, logits.ndim - 1):
        raise ValueError("Sampler top-k reduces the final dimension")
    width = logits.shape[-1]
    columns = torch.arange(width, dtype=torch.float32, device=logits.device)
    remaining = logits
    values, selected = [], []
    for _ in range(k):
        value, index = _max_smallest_id(remaining, columns, width - 1)
        values.append(value)
        selected.append(index)
        remaining = remaining.masked_fill(columns == index, float("-inf"))
    return torch.cat(values, dim=-1), torch.cat(selected, dim=-1).to(torch.int64)


def greedy_sharded_logits(logits, group, rank, vocabulary_size, top_logprobs=MAX_TOP_LOGPROBS):
    """Exact greedy selection plus normalized logprobs from vocabulary shards.

    One small all-gather carries the local maximum, ID, log-normalizer and
    top-logprob candidates. No full-vocabulary gather or host sampling occurs.
    Shards must be contiguous and identically padded. Ties choose the smallest
    vocabulary ID, matching torch.argmax on the concatenated logits: within a
    shard and across shards they are settled by max / min reductions
    (_first_max, _shard_winner), whatever index the device's argmax would return.
    """
    if vocabulary_size >= 2**24:
        raise ValueError("Packed float32 token IDs require a vocabulary smaller than 2**24")
    if not 0 <= top_logprobs <= logits.shape[-1]:
        raise ValueError("top_logprobs must fit in a local vocabulary shard")
    width = logits.shape[-1]
    ids = torch.arange(width, dtype=torch.int64, device=logits.device) + rank.to(torch.int64) * width
    logits = logits.float().masked_fill(ids.unsqueeze(0) >= vocabulary_size, float("-inf"))
    value, local_id = _first_max(logits)
    token = local_id + rank.to(torch.int64) * width
    lse = torch.logsumexp(logits, dim=-1, keepdim=True)
    payload = [value, token.float(), lse]
    if top_logprobs:
        top_vals, top_ids = _exact_topk(logits, top_logprobs, dim=-1)
        payload.extend([top_vals, (top_ids + rank.to(torch.int64) * width).float()])
    gathered = neuron_collectives.all_gather(group, torch.cat(payload, dim=-1), dim=-1)
    candidates = gathered.reshape(logits.shape[0], group.world_size, 3 + 2 * top_logprobs)
    best, token, _ = _shard_winner(candidates, vocabulary_size)
    global_lse = torch.logsumexp(candidates[:, :, 2], dim=-1)
    result = {
        "token_ids": token.squeeze(1).to(torch.int32),
        "logprobs": best.squeeze(1) - global_lse,
    }
    if top_logprobs:
        all_vals = candidates[:, :, 3:3 + top_logprobs].reshape(logits.shape[0], -1)
        all_ids = candidates[:, :, 3 + top_logprobs:].reshape(logits.shape[0], -1)
        top_vals, order = _exact_topk(all_vals, top_logprobs, dim=-1)
        result["top_vals"] = top_vals - global_lse.unsqueeze(-1)
        result["top_ids"] = all_ids.gather(1, order).to(torch.int32)
    return result


def sampled_positions(positions: torch.Tensor, sampling_positions: torch.Tensor) -> torch.Tensor:
    """Position of each sampled row's input token (the key of its draw)."""
    return torch.index_select(positions.reshape(-1), 0, sampling_positions.reshape(-1).to(torch.int64))


def sample_sharded_logits(logits, group, rank, vocabulary_size, temperatures, row_keys,
                          top_logprobs=MAX_TOP_LOGPROBS):
    """Exact temperature sampling from vocabulary shards (Gumbel-max), plus logprobs.

    Rows with temperature > 0 take ``argmax(logits / T + G)`` with G the
    (seed, position, token)-keyed Gumbel noise of ``gumbel_noise``: an exact draw
    from ``softmax(logits / T)`` over the whole vocabulary. Rows with temperature 0
    are greedy: the tokens of ``greedy_sharded_logits``, and its logprobs to fp32
    rounding (bitwise on the host; a device program may order the log-normalizer
    reduction differently). One small all-gather
    carries each shard's best score, its token and raw logit, the shard's
    log-normalizer and its top-logprob candidates.

    Logprobs follow SGLang's sampler: ``log_softmax(logits / T)`` for sampled rows,
    ``log_softmax(logits)`` for greedy rows, and ``log_softmax(logits)`` for every row
    under SGLANG_RETURN_ORIGINAL_LOGPROB. Temperature preserves order, so the top
    logprobs are selected on raw logits and rescaled.
    """
    if vocabulary_size >= 2**24:
        raise ValueError("Packed float32 token IDs require a vocabulary smaller than 2**24")
    if not 0 <= top_logprobs <= logits.shape[-1]:
        raise ValueError("top_logprobs must fit in a local vocabulary shard")
    rows, width = logits.shape
    logits, sampled, inverse, score = _tempered_scores(logits, rank, vocabulary_size, temperatures, row_keys)
    value, local_id = _first_max(score)
    token = local_id + rank.to(torch.int64) * width
    raw = logits.gather(1, local_id)
    scale = torch.ones_like(inverse) if return_original_logprob() else inverse
    lse = torch.logsumexp(logits * scale, dim=-1, keepdim=True)
    payload = [value, token.float(), raw, lse]
    if top_logprobs:
        top_vals, top_ids = _exact_topk(logits, top_logprobs, dim=-1)
        payload.extend([top_vals, (top_ids + rank.to(torch.int64) * width).float()])
    gathered = neuron_collectives.all_gather(group, torch.cat(payload, dim=-1), dim=-1)
    candidates = gathered.reshape(rows, group.world_size, 4 + 2 * top_logprobs)
    _, token, winner = _shard_winner(candidates, vocabulary_size)
    raw = candidates[:, :, 2].masked_fill(~winner, float("-inf")).amax(dim=-1)
    global_lse = torch.logsumexp(candidates[:, :, 3], dim=-1)
    scale = scale.reshape(rows)
    result = {
        "token_ids": token.squeeze(1).to(torch.int32),
        "logprobs": raw * scale - global_lse,
    }
    if top_logprobs:
        all_vals = candidates[:, :, 4:4 + top_logprobs].reshape(rows, -1)
        all_ids = candidates[:, :, 4 + top_logprobs:].reshape(rows, -1)
        top_vals, order = _exact_topk(all_vals, top_logprobs, dim=-1)
        result["top_vals"] = top_vals * scale.unsqueeze(-1) - global_lse.unsqueeze(-1)
        result["top_ids"] = all_ids.gather(1, order).to(torch.int32)
    return result


def _tempered_scores(logits, rank, vocabulary_size, temperatures, row_keys):
    """Shard-local stage of the sampled modes: padded-vocabulary mask, per-row inverse temperature
    (1 for greedy rows) and the Gumbel-perturbed tempered scores (raw logits for greedy rows)."""
    rows, width = logits.shape
    ids = torch.arange(width, dtype=torch.int64, device=logits.device) + rank.to(torch.int64) * width
    logits = logits.float().masked_fill(ids.unsqueeze(0) >= vocabulary_size, float("-inf"))
    temperature = temperatures.reshape(rows, 1).to(torch.float32)
    sampled = temperature > 0
    inverse = torch.where(sampled, 1.0 / torch.where(sampled, temperature, torch.ones_like(temperature)),
                          torch.ones_like(temperature))
    noise = gumbel_noise(row_keys, ids.to(torch.int32))
    score = torch.where(sampled, logits * inverse + noise, logits)
    return logits, sampled, inverse, score


def _pool_topk(values, k):
    """Sorted top-k of the filter pool on device: the NKI kernel where one fits the shape, else
    the reduction-based _exact_topk. Never torch.topk on a device tensor: it lowers to the
    AwsNeuronTopK custom call, which returns wrong global candidates after an all-gather. CPU tensors (tests) take
    torch.topk, which is exact."""
    from sglang_neuron.sampling_topk import select_topk_method, topk as select_topk

    if values.device.type != "cpu" and select_topk_method(values, k, -1) is None:
        return _exact_topk(values, k, dim=-1)
    return select_topk(values, k, dim=-1, gather_dim=-1)


def sample_filtered_sharded_logits(logits, group, rank, vocabulary_size, temperatures, row_keys, filters,
                                   top_logprobs=MAX_TOP_LOGPROBS, candidates=None):
    """``sample_sharded_logits`` plus SGLang's top-k / top-p / min-p filters (``topk_topp`` mode).

    ``filters`` [R, 3] float32 holds each row's normalized (top_k, top_p, min_p); top_k is
    TOP_K_ALL when unrestricted. Every shard also sends its top ``candidates`` (default
    TOP_K_MAX) raw logits; their
    union holds the global top ``candidates``, sorted here into the filter pool. On it the filters
    follow top_k_top_p_min_p_sampling_from_probs_torch: keep pool rank < top_k, exclusive
    cumulative probability <= top_p and probability >= min_p * the top probability, with the
    probabilities of the tempered full-vocabulary distribution (global log-normalizer). The draw is
    the Gumbel-max over the kept tokens with the same token-keyed noise as the unfiltered draw, so
    a row whose filters keep the unfiltered winner draws that winner; rows without filters (and
    greedy rows) take the unfiltered full-vocabulary result. Logprobs and top logprobs are those of
    the unfiltered distribution, as in SGLang; the top logprobs come from the pool.

    Ties: pool order among equal logits follows the rotational top-k kernel (torch.topk on CPU),
    not _exact_topk's smallest-index rule — upstream's own descending sort is also unstable, so a
    tie's kept-set boundary is deterministic per build but unspecified, matching upstream. The draw
    itself, over the shards and over the kept pool, takes the smallest token id among equal scores.
    """
    if vocabulary_size >= 2**24:
        raise ValueError("Packed float32 token IDs require a vocabulary smaller than 2**24")
    rows, width = logits.shape
    candidates = TOP_K_MAX if candidates is None else candidates
    if not 0 <= top_logprobs <= candidates <= width:
        raise ValueError(f"need top_logprobs ({top_logprobs}) <= candidates ({candidates}) <= shard width ({width})")
    logits, sampled, inverse, score = _tempered_scores(logits, rank, vocabulary_size, temperatures, row_keys)
    # Shard candidates via the NKI rotational top-k, on the fp32 masked logits (the values the merge
    # uses). fp32 selects the rotational kernel deliberately: the GpSIMD nisa.topk kernel deadlocks
    # when co-resident with this op's vector work in one NEFF (hangs the GPSIMD engine mid-NEFF,
    # 30 s execution timeout), while the rotational kernel in the same graph is exact against
    # torch.topk values.
    pool_vals, pool_ids = _pool_topk(logits, candidates)
    pool_ids = pool_ids.to(torch.int64)
    value, local_id = _first_max(score)
    token = local_id + rank.to(torch.int64) * width
    raw = logits.gather(1, local_id)
    original = return_original_logprob()
    scale = torch.ones_like(inverse) if original else inverse
    lse = torch.logsumexp(logits * scale, dim=-1, keepdim=True)
    tempered_lse = torch.logsumexp(logits * inverse, dim=-1, keepdim=True) if original else lse
    payload = [value, token.float(), raw, lse, tempered_lse, pool_vals,
               (pool_ids + rank.to(torch.int64) * width).float()]
    gathered = neuron_collectives.all_gather(group, torch.cat(payload, dim=-1), dim=-1)
    per_rank = gathered.reshape(rows, group.world_size, 5 + 2 * candidates)
    _, unfiltered_token, winner = _shard_winner(per_rank, vocabulary_size)
    unfiltered_raw = per_rank[:, :, 2].masked_fill(~winner, float("-inf")).amax(dim=-1, keepdim=True)
    global_lse = torch.logsumexp(per_rank[:, :, 3], dim=-1)
    global_tempered_lse = torch.logsumexp(per_rank[:, :, 4], dim=-1, keepdim=True)
    vals, order = _pool_topk(per_rank[:, :, 5:5 + candidates].reshape(rows, -1).contiguous(), candidates)
    order = order.to(torch.int64)
    pool_tokens = per_rank[:, :, 5 + candidates:].reshape(rows, -1).gather(1, order)
    probs = torch.exp(vals * inverse - global_tempered_lse)
    strictly_before = torch.ones(candidates, candidates, dtype=torch.float32, device=logits.device).triu(1)
    exclusive = probs @ strictly_before
    top_k, top_p, min_p = filters[:, 0:1], filters[:, 1:2], filters[:, 2:3]
    pool_rank = torch.arange(candidates, dtype=torch.float32, device=logits.device).unsqueeze(0)
    keep = (pool_rank < top_k) & (exclusive <= top_p) & (probs >= min_p * probs[:, :1])
    pool_score = torch.where(keep, vals * inverse + gumbel_noise(row_keys, pool_tokens.to(torch.int32)),
                             torch.full_like(vals, float("-inf")))
    _, pool_token = _max_smallest_id(pool_score, pool_tokens, vocabulary_size - 1)
    pool_raw = vals.masked_fill(pool_tokens != pool_token, float("-inf")).amax(dim=-1, keepdim=True)
    restricted = sampled & ((top_k < TOP_K_ALL) | (top_p < 1) | (min_p > 0))
    token_ids = torch.where(restricted, pool_token, unfiltered_token).squeeze(1)
    raw_token = torch.where(restricted, pool_raw, unfiltered_raw).squeeze(1)
    scale = scale.reshape(rows)
    result = {"token_ids": token_ids.to(torch.int32), "logprobs": raw_token * scale - global_lse}
    if top_logprobs:
        result["top_vals"] = vals[:, :top_logprobs] * scale.unsqueeze(-1) - global_lse.unsqueeze(-1)
        result["top_ids"] = pool_tokens[:, :top_logprobs].to(torch.int32)
    return result


# the kernel tail's finite mask value below the top_k cut
_TAIL_MASK = -3000.0


def tail_rule_sample(values: torch.Tensor, ids: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """Torch reference of a giga-kernel LM-head tail's sampler, for the tests and the prefill's first token: each row of
    ``values`` / ``ids`` ([R, k], logits sorted descending) takes the token its ``params`` row ``[top_k, top_p,
    temperature, u]`` picks. Candidates below the top_k-th are masked, top-p keeps the crossing candidate (as Hugging
    Face does), and ``u`` (``row_draws``) inverts the kept cumulative mass; a temperature below
    GK_TAIL_GREEDY_TEMPERATURE takes candidate 0. Returns [R] token ids.
    """
    k = values.shape[1]
    values = values.to(torch.float32)
    columns = torch.arange(k, dtype=torch.float32, device=values.device).unsqueeze(0)
    top_k, top_p, temperature = (params[:, i:i + 1].to(torch.float32) for i in range(3))
    kth = torch.clamp(torch.clamp(top_k, max=float(k)) - 1.0, min=0.0)
    threshold = torch.where(columns == kth, values, torch.zeros_like(values)).sum(dim=-1, keepdim=True)
    masked = torch.where(values < threshold, torch.full_like(values, _TAIL_MASK), values)
    greedy = temperature < GK_TAIL_GREEDY_TEMPERATURE
    inverse = 1.0 / torch.where(greedy, torch.ones_like(temperature), temperature)
    weights = torch.exp((masked - masked.amax(dim=-1, keepdim=True)) * inverse)
    inclusive = torch.ones(k, k, dtype=torch.float32, device=values.device).triu()
    cumulative = weights @ inclusive
    before = torch.cat([torch.zeros_like(cumulative[:, :1]), cumulative[:, :-1]], dim=-1)
    kept = torch.where((before >= top_p * cumulative[:, -1:]) & (columns > 0), torch.zeros_like(weights), weights)
    kept_cumulative = kept @ inclusive
    draw = params[:, 3:4].to(torch.float32)
    position = (kept_cumulative < draw * kept_cumulative[:, -1:]).sum(dim=-1, keepdim=True)
    position = torch.where(greedy, torch.zeros_like(position), position)
    return ids.gather(1, position.to(torch.int64)).squeeze(1)


def tail_sample_sharded_logits(logits, group, rank, vocabulary_size, params, k=GK_TAIL_SAMPLE_K):
    """``tail_rule_sample`` on vocabulary-sharded logits, for a prefill whose decode samples in a kernel tail, so the
    first token is drawn the way the decode draws the rest.

    Each shard's top ``k`` logits are all-gathered and the global top ``k`` becomes each row's candidate set. The
    decode samples from bf16 logits, so the two agree in distribution, not bit for bit. Exact ties among greedy
    maxima are broken by the pool order; callers that need the smallest id use ``greedy_sharded_logits`` for greedy
    rows.

    Returns:
        {"token_ids": [R] int32}.
    """
    if vocabulary_size >= 2**24:
        raise ValueError("Packed float32 token IDs require a vocabulary smaller than 2**24")
    rows, width = logits.shape
    if not 1 <= k <= width:
        raise ValueError(f"need 1 <= k ({k}) <= shard width ({width})")
    ids = torch.arange(width, dtype=torch.int64, device=logits.device) + rank.to(torch.int64) * width
    logits = logits.float().masked_fill(ids.unsqueeze(0) >= vocabulary_size, float("-inf"))
    shard_vals, shard_ids = _pool_topk(logits, k)
    shard_ids = shard_ids.to(torch.int64) + rank.to(torch.int64) * width
    gathered = neuron_collectives.all_gather(group, torch.cat([shard_vals, shard_ids.float()], dim=-1), dim=-1)
    per_rank = gathered.reshape(rows, group.world_size, 2 * k)
    vals, order = _pool_topk(per_rank[:, :, :k].reshape(rows, -1).contiguous(), k)
    cand_ids = per_rank[:, :, k:].reshape(rows, -1).gather(1, order.to(torch.int64))
    # a padding id's -inf would make the softmax NaN; put it below the mask value instead
    vals = torch.where(torch.isfinite(vals), vals, torch.full_like(vals, 2 * _TAIL_MASK))
    return {"token_ids": tail_rule_sample(vals, cand_ids, params).to(torch.int32)}


def _as_dict(params) -> dict:
    if params is None:
        return {}
    if isinstance(params, Mapping):
        return dict(params)
    fields = getattr(params, "__struct_fields__", None)  # a msgspec SamplingParams
    if fields is not None:
        return {name: getattr(params, name) for name in fields}
    return dict(vars(params))


def _request_params(obj, preferred=None):
    """Parameter lookup over the request merged onto the server's --preferred-sampling-params,
    exactly as SGLang builds the request's SamplingParams ({**preferred, **request}: a key the
    request sets wins, an explicit None included, and None means the default)."""
    merged = {**_as_dict(preferred), **_as_dict(getattr(obj, "sampling_params", None))}

    def get(name, default=None):
        value = merged.get(name)
        return default if value is None else value

    return get


def validate_pd_request(obj, server_args) -> None:
    """Reject, on a PD role, a request SGLang fans out to n > 1 copies, whatever the sampler. SGLang's tokenizer manager
    (TokenizerManager._handle_batch_request) sends the fan-out's common-prefix warmup and every copy as copies of the
    request's one tokenized request, so all of them carry its single bootstrap room, and their KV handoffs between the
    prefill and the decode meet in that one room."""
    role = str(getattr(server_args, "disaggregation_mode", "null") or "null")
    if role not in ("prefill", "decode"):
        return
    # the request's own n, as SGLang reads it (GenerateReqInput.parallel_sample_num); a preferred n fans out nothing
    n = int(_as_dict(getattr(obj, "sampling_params", None)).get("n") or 1)
    if n > 1:
        raise ValueError(
            f"n={n} is not supported on a PD {role} server: SGLang sends the n copies of a request, and its "
            "common-prefix warmup, under the request's one bootstrap room, so their KV transfers between the prefill "
            "and the decode collide; send the copies as n separate requests")


def request_is_greedy(obj, preferred=None) -> bool:
    """Whether a raw request is greedy by SGLang's rule (top_k 1, or temperature below GREEDY_TEMPERATURE), or
    generates no token (max_new_tokens 0), so that no temperature or filter applies to it: SGLang's frontend caches a
    shared prompt prefix that way at the default temperature (RuntimeEndpoint.cache_prefix, the first request of
    sglang.test.few_shot_gsm8k)."""
    get = _request_params(obj, preferred)
    return get("top_k", -1) == 1 or get("temperature", 1.0) < GREEDY_TEMPERATURE or get("max_new_tokens") == 0


def validate_device_sampling_request(obj, mode: str, preferred=None, deterministic=False, host_serves=False):
    """Reject options the sampler of ``mode`` cannot serve, before scheduling.

    Checks the raw request merged over the server's preferred sampling params (before
    SamplingParams.normalize): greedy is top_k == 1 or a temperature below GREEDY_TEMPERATURE, SGLang's rule (a
    negative one is left to SamplingParams.verify). A request at exactly GREEDY_TEMPERATURE keeps its temperature and
    its filters, so it is a sampled request. ``deterministic`` is the server's --enable-deterministic-inference, under
    which SGLang seeds every row that sets no seed. The host sampler applies every option but a seed and beam search.
    ``host_serves``: this launch could run the host sampler instead, so a refusal of an option it serves says so."""
    get = _request_params(obj, preferred)
    hint = _ON_HOST if host_serves else ""
    top_k = get("top_k", -1)
    greedy = request_is_greedy(obj, preferred)
    if mode == HOST_SAMPLER:
        _refuse_beam_search(get)
        if not greedy and get("sampling_seed") is not None:
            # SGLang seeds a batch only under --enable-deterministic-inference (SamplingBatchInfo.from_schedule_batch,
            # md5-pinned in platform), which it refuses on the neuron attention backend
            raise ValueError("sampling_seed is not applied by SGLang's sampler on this Neuron server "
                             f"(SGLANG_NEURON_SAMPLER={HOST_SAMPLER}); use a temperature or topk_topp server for "
                             "seeded sampling, or omit the seed")
        return
    if not greedy:
        if top_k is not None and top_k != -1 and top_k < 1:
            # SamplingParams.verify's own refusal, in its words: the mode checks below only see valid values
            raise ValueError(f"top_k must be -1 (disable) or at least 1, got {top_k}.")
        if mode == "greedy":
            raise ValueError(
                "This Neuron device-sampling server requires temperature=0 or top_k=1: it is launched with "
                f"SGLANG_NEURON_SAMPLER=greedy{hint}"
            )
        unrestricted_k = top_k == -1 or top_k >= TOP_K_ALL
        if mode == "temperature" and (not unrestricted_k or get("top_p", 1.0) != 1.0 or get("min_p", 0.0) != 0.0):
            raise ValueError(
                "top_k / top_p / min_p filtering is not applied by this Neuron device sampler "
                f"(SGLANG_NEURON_SAMPLER={mode} samples the full vocabulary); use a topk_topp server for filtered "
                f"sampling{hint}"
            )
        if mode == "topk_topp" and not (unrestricted_k or 1 <= top_k <= TOP_K_MAX):
            raise ValueError(
                f"top_k={top_k} exceeds this Neuron device sampler's filter pool "
                f"({TOP_K_MAX} tokens); use top_k <= {TOP_K_MAX}{hint}"
            )
        if mode == "topk_topp" and get("min_p", 0.0) != 0.0:
            from sglang_neuron import serving_policy

            if serving_policy.current().giga_decode:
                # a giga-kernel decode samples in its kernel's LM-head tail, which has no min_p
                raise ValueError("min_p is not applied by the giga-kernel decode's sampler, which applies top_k, "
                                 f"top_p and temperature only; send min_p=0{hint}")
    seed = get("sampling_seed")
    if seed is not None and not -(2**63) <= int(seed) < 2**63:
        raise ValueError("sampling_seed must fit in a signed 64-bit integer")
    # SGLang fans a request out on its own n only
    if (not greedy and (seed is not None or deterministic)
            and int(_as_dict(getattr(obj, "sampling_params", None)).get("n") or 1) > 1):
        # TokenizerManager._handle_batch_request gives every copy a fresh uuid4 rid and the request's one
        # SamplingParams, so the scheduler sees n requests with one seed: their noise keys, and so their draws, would
        # be identical
        what = ("sampling_seed with n > 1" if seed is not None
                else "n > 1 under --enable-deterministic-inference (which seeds every row alike)")
        fix = ("send the copies as separate requests with distinct seeds, or omit the seed" if seed is not None
               else "send the copies as separate requests with distinct seeds")
        raise ValueError(
            f"{what} would return n identical samples on the Neuron device sampler: SGLang gives the copies of an "
            f"n > 1 fan-out the request's one SamplingParams, so they share one seed and their noise keys match at "
            f"every position; {fix}")
    _validate_device_sampler_features(obj, get, hint)


# appended to a device-sampler refusal of an option the host sampler serves, on a launch that could run it
_ON_HOST = f" (a server launched with SGLANG_NEURON_SAMPLER={HOST_SAMPLER} and --disable-overlap-schedule serves it)"


def _refuse_beam_search(get):
    if (get("beam_width") or 1) > 1:  # SGLang serves a width of one as an ordinary request
        raise ValueError("Beam search is not supported on Neuron")


def _validate_device_sampler_features(obj, get, hint=""):
    """Options no device sampler mode applies (penalties, processors, beam, logprob forms)."""
    if (get("frequency_penalty", 0) != 0 or get("presence_penalty", 0) != 0
            or get("repetition_penalty", 1) != 1):
        raise ValueError(f"Penalties are not applied by the Neuron device sampler{hint}")
    if get("min_new_tokens", 0):
        raise ValueError(f"Minimum-length token masking (min_new_tokens={get('min_new_tokens')}) is not applied by "
                         f"the Neuron device sampler{hint}")
    _refuse_beam_search(get)
    constrained = [name for name in ("logit_bias", "json_schema", "regex", "ebnf", "structural_tag") if get(name)]
    if getattr(obj, "custom_logit_processor", None):
        constrained.append("custom_logit_processor")
    if constrained:
        raise ValueError(f"Logit processors and constrained generation ({', '.join(constrained)}) are not applied by "
                         f"the Neuron device sampler{hint}")
    # Counted with or without return_logprob: a batch with any logprob request carries every row's
    # top_logprobs_num (ScheduleBatch.top_logprobs_nums), so the sampler output must hold them all.
    requested = getattr(obj, "top_logprobs_num", 0) or 0
    if requested > MAX_TOP_LOGPROBS:
        raise ValueError(f"This Neuron server supports at most {MAX_TOP_LOGPROBS} top logprobs "
                         f"(top_logprobs_num={requested}; the cap is SGLANG_NEURON_MAX_TOP_LOGPROBS){hint}")
    # An empty list counts: a batch's sub-request keeps its [] (only a single request's becomes None), and the
    # scheduler treats any non-None list as a token-ID logprob request
    if getattr(obj, "token_ids_logprob", None) is not None:
        raise ValueError("Arbitrary token-ID logprobs (token_ids_logprob, an empty list included) are not produced "
                         f"by the Neuron device sampler{hint}")
    if getattr(obj, "return_sampling_mask", False):
        raise ValueError("The sampling mask (return_sampling_mask) is not produced by the Neuron device sampler"
                         f"{hint}")
    returns = getattr(obj, "return_logprob", False)
    if any(returns) if isinstance(returns, (list, tuple)) else bool(returns):
        from sglang_neuron import serving_policy

        if serving_policy.current().giga_decode:
            # the fused tail picks tokens inside the kernel, greedy or sampled, so no log-normalizer leaves it
            raise ValueError("Logprobs (return_logprob) are not produced by the giga-kernel decode: its fused tail "
                             "picks each token inside the kernel, and no log-normalizer leaves it")
