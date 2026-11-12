# SPDX-License-Identifier: Apache-2.0
"""Inputs of one GLM-5.2 prefill stage step: the host tensors the runtime uploads and the per-rank index math the
stage graph runs on them.

Every rank gets the whole step and keeps its context-parallel shard: rank r owns the r-th run of ``block`` tokens in
every ``cp * block``. A slot is always the owner rank's slot: token p lives on rank (p // block) % cp at local row
(p // (cp * block)) * block + p % block, in page table[local // page] at row local % page. The server, the stage
benchmark and the CPU tests call the same functions, so they compute the same inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import Tensor

from sglang_neuron import sampling
from sglang_neuron.models.glm52.launch import MAX_CONTEXT
from sglang_neuron.models.glm52.layout import rope_split
from sglang_neuron.parallel import collectives

#: The query position of a bucket padding row: larger than any real position, so no real query
#: scores a padding key; a power of two below 2^24, so fp32 holds it and the indexer's -2^102 mask scale stays finite.
PAD_POSITION = 1 << 23


def nonneg_div(x: Tensor, d: int) -> Tensor:
    """``x // d`` for a non-negative integer tensor ``x``, where floor and truncation agree: a plain integer divide.
    The floor form (``//``) adds sign-fix selects, which the image's neuronx-cc 2.28 tensorizer cannot split inside a
    stage graph (LegalizeSundaMacro, NCC_ILSM901 "Cannot split")."""
    return torch.div(x, d, rounding_mode="trunc")


def nonneg_mod(x: Tensor, d: int) -> Tensor:
    """``x % d`` for a non-negative integer tensor ``x``, without the floor form's sign-fix selects (nonneg_div)."""
    return x - nonneg_div(x, d) * d


def owned_index(seq_len: int, block: int, cp: int, cp_rank: Tensor, device) -> Tensor:
    """The indices of the tokens CP rank ``cp_rank`` owns, in order: the
    ``cp_rank``-th run of ``block`` tokens of every virtual block of ``cp * block``. ``cp_rank`` is a runtime tensor, so
    every rank traces one graph."""
    n_vblock = seq_len // (cp * block)
    vblock = torch.arange(n_vblock, device=device).view(n_vblock, 1) * (cp * block)
    within = torch.arange(block, device=device).view(1, block)
    offset = cp_rank.to(torch.int64) * block
    return (vblock + within + offset).reshape(-1)


def rank_major_order(seq_len: int, block: int, cp: int, device) -> Tensor:
    """The token order whose chunk r is owned_index(.., r), the rank-major interleaved order, so a
    reduce-scatter of the reordered tokens gives each rank its own shard."""
    n_vblock = seq_len // (cp * block)
    order = torch.arange(seq_len, device=device).view(n_vblock, cp, block)
    return order.permute(1, 0, 2).reshape(-1)


def query_positions(positions: Tensor) -> Tensor:
    """Each row's query position: its position, or PAD_POSITION on a bucket padding row. One
    sequence runs per forward, so the real positions increase strictly, and a row whose position does not exceed the
    row before it is padding (the runtime repeats the last real position there). RoPE keeps the true positions."""
    is_pad = torch.cat([torch.zeros_like(positions[:1], dtype=torch.bool), positions[1:] <= positions[:-1]])
    return torch.where(is_pad, torch.full_like(positions, PAD_POSITION), positions)


def rope_tables(theta: float, dim: int) -> dict[str, Tensor]:
    """The default (non-YaRN) RoPE of a ``dim``-wide rope slice as the 1M decode keeps it: the two-level float32 tables
    of float64 angles (layout.rope_split, 512 KiB), over every position a stage graph addresses (launch.MAX_CONTEXT).
    Angles computed in float32 drift by up to 0.05 rad at 1M, and a full table pair would take 256 MiB a rank."""
    return rope_split.split_rope_tables(theta, dim // 2, MAX_CONTEXT)


def rope_rows(tables: dict[str, Tensor], positions: Tensor) -> tuple[Tensor, Tensor]:
    """``(cos, sin)`` fp32 rows at ``positions``, the same products as the 1M decode's lookup, so prefill keys and
    decode queries carry bit-identical angles."""
    index = torch.clamp(positions.to(torch.int64), 0, MAX_CONTEXT - 1)
    hi = nonneg_div(index, rope_split.LO_ROWS)
    lo = index - hi * rope_split.LO_ROWS
    cos_hi, sin_hi, cos_lo, sin_lo = (tables[name] for name in rope_split.TABLE_NAMES)
    ch, sh = cos_hi.index_select(0, hi), sin_hi.index_select(0, hi)
    cl, sl = cos_lo.index_select(0, lo), sin_lo.index_select(0, lo)
    return ch * cl - sh * sl, sh * cl + ch * sl


def kv_context(slot_mapping: Tensor, block_table: Tensor, cached_seq_len: Tensor, *, t_local: int, cp: int,
               cp_rank: Tensor, block: int, page: int) -> tuple[Tensor, dict]:
    """This rank's ``[t_local]`` KV write slots and the kernel's prefix arguments.

    A padding token writes its natural slot, or the null page when it falls past the table. The kernel's static
    ``kv_ctx_local`` is the table's width, so every prior runs the same graph.

    Raises:
        ValueError: If ``t_local`` is not a multiple of ``block``.
    """
    if t_local % block:
        raise ValueError(f"the local token count {t_local} is not a multiple of the CP interleave {block}: use prefill "
                         f"buckets that are multiples of {cp * block}")
    table = block_table.reshape(-1).clamp(min=0).to(torch.int32)
    device = slot_mapping.device
    owned = owned_index(cp * t_local, block, cp, cp_rank.reshape(()), device)
    slots = slot_mapping.reshape(-1).index_select(0, owned).to(torch.int32)
    local_pos = torch.arange(t_local, dtype=torch.int32, device=device)
    prior = nonneg_div(cached_seq_len.reshape(1, 1).to(torch.int32), cp).to(torch.int32)
    local_pos = local_pos + prior.reshape(())
    prefix_args = {"block_table": table, "prior_tokens": prior, "kv_ctx_local": table.shape[0] * page,
                   "kv_block_size": page}
    ext = torch.cat([table, table.new_zeros(-(-t_local // page))])
    natural = ext.index_select(0, nonneg_div(local_pos, page)) * page + nonneg_mod(local_pos, page)
    return torch.where(slots < page, natural, slots), prefix_args


def group_tables(positions: Tensor, slots: Tensor, *, seq_len: int, block: int, cp: int, cp_rank: Tensor,
                 wo_group: int, rope) -> dict:
    """The hybrid head TP's tables of this rank's wo group: ``cos_g`` / ``sin_g`` of the group's ``wo_group * t_local``
    tokens in rank order, and ``slot_mapping_g``, this rank's own slots at its position in the group and -1 elsewhere,
    so each rank writes only its own cache rows. ``rope`` maps positions to ``(cos, sin)``."""
    src = cp_rank.reshape(())
    base = src - src % wo_group
    device = positions.device
    owned_g = torch.cat([owned_index(seq_len, block, cp, base + j, device) for j in range(wo_group)])
    cos_g, sin_g = rope(positions.index_select(0, owned_g))
    out = {"cos_g": cos_g.to(torch.float32), "sin_g": sin_g.to(torch.float32)}
    t_local = seq_len // cp
    p = (src % wo_group).to(torch.int64)
    own = nonneg_div(torch.arange(wo_group * t_local, device=device), t_local) == p
    none = torch.full((wo_group * t_local,), -1, dtype=torch.int32, device=device)
    out["slot_mapping_g"] = torch.where(own, slots.repeat(wo_group), none).to(torch.int32)
    return out


def select_sampled_rows(hidden: Tensor, sampling_positions: Tensor, *, block: int, cp: int, cp_rank: Tensor,
                        group) -> Tensor:
    """The ``[N, H]`` rows at the sampling positions on every rank: each rank keeps the rows it owns, zeros the rest
    (a select, so a NaN elsewhere cannot leak in) and a sum all-reduce combines them."""
    positions = sampling_positions.to(torch.int64)
    owner = nonneg_mod(nonneg_div(positions, block), cp)
    local_row = nonneg_div(positions, cp * block) * block + nonneg_mod(positions, block)
    rows = hidden.index_select(0, local_row)
    owned = (owner == cp_rank.reshape(()).to(torch.int64)).unsqueeze(-1)
    rows = torch.where(owned, rows, torch.zeros_like(rows))
    return collectives.all_reduce(group, rows.contiguous(), "sum")


def first_token_params(positions: Tensor, sampling_positions: Tensor, temperatures: Tensor, seeds: Tensor,
                       filters: Tensor, k: int) -> Tensor:
    """The decode tail's ``[R, 4]`` ``(top_k, top_p, temperature, u)`` rows, with ``u`` drawn from the request seed at
    the prompt's last position, so the first token and the decode tokens share one stream of draws."""
    rows = sampling_positions.reshape(-1).shape[0]
    columns = torch.arange(2, dtype=torch.int64, device=filters.device)
    picked = filters[:rows].to(torch.float32)
    top_k = torch.clamp(picked.index_select(1, columns[0:1]), min=1.0, max=float(k))
    top_p = picked.index_select(1, columns[1:2])
    temperature = temperatures[:rows].to(torch.float32).reshape(rows, 1)
    keys = sampling.noise_row_keys(seeds[:rows].to(torch.int32), sampling.sampled_positions(positions, sampling_positions))
    draws = sampling.row_draws(keys).reshape(rows, 1)
    return torch.cat([top_k, top_p, temperature, draws], dim=1)


def sample_first_token(logits: Tensor, group, cp_rank: Tensor, vocabulary_size: int, *, positions: Tensor,
                       sampling_positions: Tensor, sampling_temperatures: Tensor | None = None,
                       sampling_seeds: Tensor | None = None, sampling_filters: Tensor | None = None) -> dict:
    """Pick each sampled row's first token from the vocabulary-sharded ``logits``: exact greedy without sampling
    inputs, otherwise the decode tail's draw with the same keys. ``logprobs`` is over the whole vocabulary.

    Returns:
        ``{"token_ids": [R] int32, "logprobs": [R] float32}``.
    """
    if sampling_temperatures is not None and (sampling_seeds is None or sampling_filters is None):
        raise ValueError("the first token's draw takes the temperatures, seeds and (top_k, top_p, min_p) filters of "
                         "SGLANG_NEURON_SAMPLER=topk_topp")
    greedy = sampling.greedy_sharded_logits(logits, group, cp_rank, vocabulary_size, top_logprobs=0)
    if sampling_temperatures is None:
        return greedy
    k = sampling.GK_TAIL_SAMPLE_K
    params = first_token_params(positions, sampling_positions, sampling_temperatures, sampling_seeds,
                                sampling_filters, k)
    drawn = sampling.tail_sample_sharded_logits(logits, group, cp_rank, vocabulary_size, params, k=k)["token_ids"]
    greedy_rows = params[:, 2] < sampling.GK_TAIL_GREEDY_TEMPERATURE
    token = torch.where(greedy_rows, greedy["token_ids"], drawn)
    return {"token_ids": token, "logprobs": _token_logprobs(logits, token, group, cp_rank, vocabulary_size)}


def _token_logprobs(logits: Tensor, token: Tensor, group, cp_rank: Tensor, vocabulary_size: int) -> Tensor:
    """log p(token) over the whole vocabulary from the shards: each shard's logit of the token (-inf off the shard)
    and its log-normalizer, one all-gather over ``group``."""
    rows, width = logits.shape
    ids = torch.arange(width, dtype=torch.int64, device=logits.device) + cp_rank.to(torch.int64).reshape(()) * width
    masked = logits.float().masked_fill(ids.unsqueeze(0) >= vocabulary_size, float("-inf"))
    hit = ids.unsqueeze(0) == token.to(torch.int64).reshape(rows, 1)
    value = torch.where(hit, masked, torch.full_like(masked, float("-inf"))).amax(dim=-1, keepdim=True)
    lse = torch.logsumexp(masked, dim=-1, keepdim=True)
    gathered = collectives.all_gather(group, torch.cat([value, lse], dim=-1), dim=-1).reshape(rows, -1, 2)
    return gathered[:, :, 0].amax(dim=-1) - torch.logsumexp(gathered[:, :, 1], dim=-1)


def owner_slots(virtual_slots: Tensor, *, virtual_page: int, interleave: int) -> Tensor:
    """Map SGLang's virtual slots to owner-local slots, (vs // virtual_page) * interleave + vs % interleave. SGLang's
    paged allocator keeps slot % page equal to position % page, so this is the owner-slot rule; slot 0 stays 0."""
    vs = virtual_slots.to(torch.int64)
    return (nonneg_div(vs, virtual_page) * interleave + nonneg_mod(vs, interleave)).to(torch.int32)


@dataclass(frozen=True)
class Glm52HostInputs:
    """The host tensors of one step, in upload order (``tensors``): all int32 but ``sampling_positions`` (int64)."""

    input_ids: Tensor  # [T], pad 0
    positions: Tensor  # [T], pad = the last real position (query_positions reads the padding off it)
    slot_mapping: Tensor  # [T], every token's owner slot, pad 0
    block_table: Tensor  # [1, W], the request's page ids on every rank of the stage, -1 past the request
    cached_seq_len: Tensor  # [1, 1], the global prior, a whole number of virtual pages
    sampling_positions: Tensor  # [1] = n - 1, the step's last real row

    def tensors(self) -> list[Tensor]:
        return [getattr(self, f.name) for f in fields(self)]


#: The stage graph's step inputs in order (Glm52HostInputs' fields, then the rank tensor), and its boundary names.
STEP_INPUTS = tuple(f.name for f in fields(Glm52HostInputs)) + ("rank",)
BOUNDARY = ("pipeline_hidden_states", "pipeline_residual")
#: The boundary's third tensor between stages where the next starts on a shared indexer layer (StageLayout.picks_out):
#: each rank's ``[tokens / cp, index_topk]`` int32 top-k picks of the stage's last full layer.
PICKS = "pipeline_picks"


def host_inputs(*, token_ids: Tensor, positions: Tensor, slot_mapping: Tensor, pages: Tensor, cached: int, bucket: int,
                table_width: int, virtual_page: int) -> Glm52HostInputs:
    """One request's step padded to ``bucket``.

    Raises:
        ValueError: An empty or oversized step, a bucket or prior off the virtual page, more pages than the table
            holds, or inputs of different lengths.
    """
    n = int(token_ids.shape[0])
    if not 0 < n <= bucket or positions.shape[0] != n or slot_mapping.shape[0] != n:
        raise ValueError(f"a prefill step of {n} tokens ({positions.shape[0]} positions, {slot_mapping.shape[0]} "
                         f"slots) in a bucket of {bucket}")
    if bucket % virtual_page or cached % virtual_page:
        raise ValueError(f"the bucket {bucket} and the prior {cached} must be whole {virtual_page}-token virtual "
                         "pages: the stage splits both over its ranks in whole runs of the CP interleave")
    pages = pages.reshape(-1)
    if pages.shape[0] > table_width:
        raise ValueError(f"the request holds {pages.shape[0]} pages, more than the block table's {table_width}")
    input_ids = torch.zeros(bucket, dtype=torch.int32)
    input_ids[:n] = token_ids.to(torch.int32)
    padded_positions = torch.full((bucket,), int(positions[n - 1]), dtype=torch.int32)
    padded_positions[:n] = positions.to(torch.int32)
    slots = torch.zeros(bucket, dtype=torch.int32)
    slots[:n] = slot_mapping.to(torch.int32)
    block_table = torch.full((1, table_width), -1, dtype=torch.int32)
    block_table[0, :pages.shape[0]] = pages.to(torch.int32)
    return Glm52HostInputs(input_ids=input_ids, positions=padded_positions, slot_mapping=slots,
                           block_table=block_table, cached_seq_len=torch.tensor([[cached]], dtype=torch.int32),
                           sampling_positions=torch.tensor([n - 1], dtype=torch.int64))


def step_kwargs(dev, rank: Tensor, *, stage_in: tuple | None = None, stage_out: tuple | None = None) -> dict:
    """The stage graph's keyword arguments in the runtime's order: the step inputs, the rank tensor, and the
    boundary in and out."""
    tensors = dev.tensors() if isinstance(dev, Glm52HostInputs) else list(dev)
    if len(tensors) != len(STEP_INPUTS) - 1:
        raise ValueError(f"{len(tensors)} step tensors; the stage graph takes {STEP_INPUTS[:-1]}")
    out = dict(zip(STEP_INPUTS, [*tensors, rank]))
    names = (*BOUNDARY, PICKS)
    for given, suffix in ((stage_in, ""), (stage_out, "_out")):
        if given is None:
            continue
        if len(given) not in (2, 3):
            raise ValueError(f"a boundary of {len(given)} tensors: the stages hand (h, r) or (h, r, picks)")
        out.update(zip((f"{name}{suffix}" for name in names), given))
    return out
