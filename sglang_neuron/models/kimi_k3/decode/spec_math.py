# SPDX-License-Identifier: Apache-2.0
"""In-graph torch glue of one Kimi-K3 DSpark decode step: carry, windows, embedding, acceptance, drafter inputs.

These functions are traced into the verify and draft device programs, so each keeps its torch ops, dtypes, shapes and
op order: they decide the HLO around the kernels. Every rank value and collective comes in as an argument.

Row order: the host keeps the Q = 1 + k rows of a request together (request-major, row ``b * Q + j``); the kernels
take token-major rows (row ``j * B + b``) (``to_token_major``, ``to_request_major``).

Async steps: the host's row 0 ``N`` of a carried request runs ahead of the true row 0 by the drafts that the last step
rejected (0 to k), so the host gives each request a window of candidate positions from ``N - k`` and the graph takes
its rows at the true row 0 (``carry_spec_state``, ``window_rows``). A fresh row takes its replay length and state
position from the host; any other row takes them from the outputs of the last step.

Probabilistic drafts (``accept_probabilistic_drafts``): the drafter draws each draft of a sampled request from its
distribution q over at most 256 candidates S of the row; the verify takes one joint accept length from the cumulative
ratio ``c_i = min(1, c_{i-1} p(d_i) / q(d_i))`` (``block``) or tests each draft (``standard``), and draws the stop row
from ``max(c p - q, 0)`` as a mixture of its part on S and the target outside S. The shapes are static.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor

ROW_FLAG_NAN = 1
"""Row flag: a logit of the row is NaN or +inf."""
ROW_FLAG_NO_TOKEN = 2
"""Row flag: every logit of the row is -inf (no token is allowed)."""
ROW_FLAG_LAST_ID = 4
"""Row flag: a greedy kernel token is the last id, the clamp target of a NaN row."""
SAMPLING_EPS = 1e-5
"""A temperature below this samples greedily."""
MLA_BLOCK = 128
"""Tokens of one MLA kernel page, which is also the CP interleave."""
DRAFTER_SAMPLE_SLOTS = 7
"""The draft slots of the MLA drafter kernel: ``u_draft`` is ``[B, 7]`` whatever k is."""
DRAFT_SAMPLE_WIDTH = 256
"""The q-list width of the drafter: the top ``256 / tp`` candidates of each of the ``tp`` vocab shards."""
PROBABILISTIC_METHODS = ("standard", "block")
"""The rejection rules of probabilistic drafts."""

_TINY = torch.finfo(torch.float32).tiny
_EPS = torch.finfo(torch.float32).eps
_BELOW_ONE = 1.0 - _EPS
_LAST_KEY = 2**31 - 1
"""The order key of an id below 0: above every vocab id, so it goes last. The largest int32 value (the compiler
refuses a 64-bit constant outside the 32-bit range)."""
_ORDER_CHUNK = 32
"""The columns of one compare chunk of ``_ascending_order`` (its temporaries are ``[B, K, 32]``)."""


# Row order and windows.


def to_token_major(values: Tensor, queries: int) -> Tensor:
    """Reorder request-major rows ``[B * Q, ...]`` (row ``b * Q + j``) to token-major (row ``j * B + b``)."""
    batch = values.shape[0] // queries
    return values.reshape(batch, queries, *values.shape[1:]).transpose(0, 1).reshape(values.shape)


def to_request_major(values: Tensor, queries: int) -> Tensor:
    """Invert ``to_token_major``."""
    batch = values.shape[0] // queries
    return values.reshape(queries, batch, *values.shape[1:]).transpose(0, 1).reshape(values.shape)


def slot_window_width(num_speculative_tokens: int) -> int:
    """Return the candidate positions of the verify rows: ``N - k .. N + k`` for the host's row 0 ``N``."""
    return 2 * num_speculative_tokens + 1


def window_rows(window: Tensor, offset: Tensor, count: int) -> Tensor:
    """Return ``count`` rows of each window from its offset: ``[B, W, ...]`` to ``[B, count, ...]``."""
    index = offset.to(torch.int64).reshape(-1, 1) + torch.arange(count, dtype=torch.int64, device=offset.device)
    # The gather needs an explicit bound on the device; the callers clamp the offset too.
    index = index.clamp(0, window.shape[1] - 1)
    if window.ndim == 2:
        return torch.gather(window, 1, index)
    index = index.reshape(*index.shape, *([1] * (window.ndim - 2)))
    return torch.gather(window, 1, index.expand(-1, -1, *window.shape[2:]))


class SpecCarry(NamedTuple):
    """The int32 per-request state of a speculative step.

    Attributes:
        replay: ``[B]`` the verify rows of the previous step that the step builds on (the KDA state slot it loads).
        state_position: ``[B]`` the position of the committed KDA state (row 0 of the previous step).
        row0: ``[B]`` the position of verify row 0 of this step.
        input_ids: ``[B, Q]`` the bonus token and the drafts of each request.
    """

    replay: Tensor
    state_position: Tensor
    row0: Tensor
    input_ids: Tensor


def carry_spec_state(
    fresh: Tensor,
    prev_row: Tensor,
    host_replay: Tensor,
    host_state_position: Tensor,
    host_ids: Tensor,
    prev_accepted: Tensor,
    prev_row0: Tensor,
    prev_next_ids: Tensor,
) -> SpecCarry:
    """Take the state of each request from the host (fresh) or from row ``prev_row`` of the last step.

    The last step gives ``accepted`` (the replay length), ``row0`` (the committed state position) and ``next_ids``,
    so ``row0 = state_position + replay`` needs no host read of the last step.

    Args:
        fresh: ``[B]`` 1 for a row with host values (sync, a new or absent request, padding), else 0.
        prev_row: ``[B]`` the row of the request in the last step (any value for a fresh row).
        host_replay, host_state_position: ``[B]`` the host values of a fresh row.
        host_ids: ``[B * Q]`` request-major host input ids.
        prev_accepted, prev_row0: ``[B]`` outputs of the last step.
        prev_next_ids: ``[B, Q]`` output of the last step.
    """
    batch, queries = prev_next_ids.shape
    # The gather needs an explicit bound on the device; a fresh row may hold any prev_row.
    index = prev_row.to(torch.int64).clamp(0, batch - 1)
    host = fresh.to(torch.int32) != 0
    replay = torch.where(host, host_replay.to(torch.int32), torch.index_select(prev_accepted.to(torch.int32), 0, index))
    state_position = torch.where(
        host,
        host_state_position.to(torch.int32),
        torch.index_select(prev_row0.to(torch.int32), 0, index),
    )
    input_ids = torch.where(
        host.reshape(-1, 1),
        host_ids.to(torch.int32).reshape(batch, queries),
        torch.index_select(prev_next_ids.to(torch.int32), 0, index),
    )
    return SpecCarry(replay, state_position, state_position + replay, input_ids)


def step_positions(row0: Tensor, window_base: Tensor, position_window: Tensor, queries: int) -> tuple[Tensor, Tensor]:
    """Return ``(offset [B], positions [B * Q])``: the window offset of the true row 0 and the request-major positions.

    The clamp keeps the gather in range; the host check reports a row 0 outside the window.
    """
    offset = (row0 - window_base.to(torch.int32)).clamp(0, position_window.shape[1] - queries)
    positions = window_rows(position_window, offset, queries).reshape(-1)
    return offset, positions


# Embedding.


def masked_vocab_embedding(
    input_ids: Tensor, weight: Tensor, tp_rank: Tensor, all_reduce: Callable[[Tensor], Tensor]
) -> Tensor:
    """Return the embedding rows of ``input_ids`` from a TP vocab shard ``weight [V / tp, H]``, summed over the group.

    Each rank looks up the ids of its shard (``tp_rank * V / tp`` on) and gives zeros for the others; ``all_reduce``
    (the TP-group sum) adds the shards.
    """
    local_vocab = weight.shape[0]
    start = tp_rank.to(input_ids.dtype)
    start = start * local_vocab
    owned = (input_ids >= start) & (input_ids < start + local_vocab)
    local_ids = (input_ids - start).clamp(min=0, max=local_vocab - 1)
    local = F.embedding(local_ids, weight)
    local = torch.where(owned.unsqueeze(-1), local, torch.zeros_like(local))
    return all_reduce(local)


# KDA state pages and MLA slots.


def kda_state_blocks(queries: int) -> int:
    """Return the KDA state pages of each request of a verify step of Q rows: the committed state and Q row slots."""
    return 1 + queries


def valid_pages(pages: Tensor, num_pages: int, null_page: int = 0) -> Tensor:
    """Give ``null_page`` for each invalid page (a negative entry, page 0, or a page outside the pool)."""
    pages = pages.to(torch.int32)
    valid = (pages > 0) & (pages < num_pages)
    if null_page == 0:
        return torch.where(valid, pages, torch.zeros_like(pages))
    if not 0 < null_page < num_pages:
        raise ValueError(f"The KDA null page {null_page} is not in the pool of {num_pages} pages.")
    return torch.where(valid, pages, torch.full_like(pages, null_page))


def mask_pad_rank_pages(pages: Tensor, tp_rank: Tensor | None, kda_ranks: int | None, null_page: int = 0) -> Tensor:
    """Give the null page to every entry of a pad rank (``tp_rank >= kda_ranks``); None keeps the pages."""
    if kda_ranks is None:
        return pages
    if tp_rank is None:
        raise ValueError("Give the TP rank with kda_ranks.")
    real = tp_rank.reshape(-1)[0].to(torch.int32) < kda_ranks
    return torch.where(real, pages, torch.full_like(pages, null_page))


def verify_kda_slot_pages(
    block_table: Tensor, num_pages: int, replay_len: Tensor, queries: int, null_page: int = 0
) -> tuple[Tensor, Tensor, Tensor]:
    """Return the KDA state page ``[B, 1]`` and the state slots of a verify step.

    The verify kernel writes the state after scored row ``j`` to the slot of row ``j``. The next step loads the state
    after its ``replay_len`` accepted rows from the slot of row ``replay_len - 1``, so it replays no rows. A step reads
    its slot before it writes the new slots, so each step uses the same Q slot pages. A rejected row's slot is never
    read.

    Args:
        block_table: ``[B, 1 + Q]`` state pages of each request: column 0 the committed state (the prefill state goes
            there), column ``1 + j`` the slot of row ``j``.
        num_pages: the page count of the pool of the layer.
        replay_len: ``[B]`` or ``[B, 1]`` accepted rows of the previous step, 0 to Q.
        queries: Q, the verify rows of each request.
        null_page: the page of an invalid state or slot write (0, or a separate KDA null page). An invalid slot read
            stays page 0 (off).

    Returns:
        ``(state [B, 1], state_slot_page [B, 1], write_state_slots [B, Q])`` int32. ``state_slot_page`` is the slot of
        row ``replay_len - 1``, or page 0 when ``replay_len`` is 0 (the kernel then reads the committed state).
    """
    blocks = kda_state_blocks(queries)
    if block_table.ndim != 2 or block_table.shape[1] != blocks:
        raise ValueError(
            f"The KDA state table has shape {tuple(block_table.shape)}. The Kimi-K3 verify step needs {blocks} state "
            "pages for each request."
        )
    pages = valid_pages(block_table, num_pages)
    replay = replay_len.to(torch.int64).reshape(-1, 1)
    read = torch.gather(pages, 1, replay.clamp(0, queries))
    read = torch.where(replay > 0, read, torch.zeros_like(read))
    if null_page:
        pages = valid_pages(block_table, num_pages, null_page)
    return pages[:, 0:1], read, pages[:, 1:]


def mla_kernel_slots(
    slot_mapping: Tensor,
    positions: Tensor,
    cp_rank: Tensor,
    num_pages: int,
    *,
    cp_size: int = 32,
    block: int = MLA_BLOCK,
) -> Tensor:
    """Return the kernel-page slot of each current MLA row on this rank, ``[N, 1]`` int32 (0 if not the owner).

    The owner of position ``p`` is CP rank ``(p // block) % cp_size``. A slot in the null page or outside the pool of
    ``num_pages`` kernel pages also gets 0.
    """
    slots = slot_mapping.reshape(-1).to(torch.int32)
    positions = positions.reshape(-1).to(torch.int32)
    if slots.shape != positions.shape:
        raise ValueError("The MLA slots need one slot for each position.")
    owner = torch.div(positions, block, rounding_mode="floor").remainder(cp_size)
    owned = owner == cp_rank.reshape(1).to(torch.int32)
    valid = owned & (slots >= block) & (slots < num_pages * block)
    return torch.where(valid, slots, torch.zeros_like(slots)).reshape(-1, 1)


def verify_mla_slots(
    slot_mapping: Tensor, positions: Tensor, cp_rank: Tensor, num_pages: int, queries: int, cp_size: int = 32
) -> Tensor:
    """Return the MLA slots ``[Q, B, 1]`` of the Q tokens of each request on this rank (0 if not the owner).

    ``slot_mapping`` and ``positions`` are request-major ``[B * Q]``.
    """
    slots = mla_kernel_slots(slot_mapping, positions, cp_rank, num_pages, cp_size=cp_size)
    return to_token_major(slots, queries).reshape(queries, -1, 1)


# Greedy, synthetic and forced acceptance.


def accept_drafts(target_tokens: Tensor, drafts: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Accept the matching prefix of the drafts; return ``(output [B, Q], num_accepted [B], bonus [B])`` int32.

    ``target_tokens [B, Q]`` holds the target token of each verify row (row ``j`` predicts position ``p + j + 1``);
    ``drafts [B, Q - 1]`` the inputs of rows 1..Q-1. ``output`` holds the target tokens of rows
    ``0 .. num_accepted - 1`` and -1 after them.
    """
    match = (drafts.to(torch.int64) == target_tokens[:, :-1].to(torch.int64)).to(torch.int64)
    num_accepted = torch.cumprod(match, dim=1).sum(dim=1) + 1
    rows = torch.arange(target_tokens.shape[1], device=target_tokens.device).reshape(1, -1)
    output = torch.where(rows < num_accepted.reshape(-1, 1), target_tokens.to(torch.int64), torch.full_like(rows, -1))
    bonus = torch.gather(target_tokens.to(torch.int64), 1, (num_accepted - 1).reshape(-1, 1)).reshape(-1)
    return output.to(torch.int32), num_accepted.to(torch.int32), bonus.to(torch.int32)


def accept_known_drafts(target_tokens: Tensor, drafts: Tensor, known: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Accept the matching prefix of the drafts, at most the ``known`` drafts of each request.

    A pad draft is not a known draft, so a match there is not accepted. Returns ``(output, num_accepted, bonus)`` as
    ``accept_drafts``.
    """
    _, accepted, _ = accept_drafts(target_tokens, drafts)
    # A known count below 0 would give the bonus gather the index -1.
    known = known.to(torch.int32).reshape(-1).clamp(0, target_tokens.shape[1] - 1)
    accepted = torch.minimum(accepted, known + 1)
    rows = torch.arange(target_tokens.shape[1], device=target_tokens.device).reshape(1, -1)
    target = target_tokens.to(torch.int64)
    output = torch.where(rows < accepted.reshape(-1, 1), target, torch.full_like(target, -1))
    bonus = torch.gather(target, 1, (accepted.to(torch.int64) - 1).reshape(-1, 1))
    return output.to(torch.int32), accepted.to(torch.int32), bonus.reshape(-1).to(torch.int32)


def accept_synthetic_drafts(
    target_tokens: Tensor, drafts: Tensor, known: Tensor, synthetic: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Accept the first ``synthetic`` drafts of each request (at most ``known``), whatever the target tokens.

    The output holds the accepted drafts, then the target token of the first rejected row (or of the bonus row), and
    -1 after them. Returns ``(output, num_accepted, bonus)`` as ``accept_drafts``.
    """
    # The host counts are the gather index; the bonus gather needs [0, Q - 1].
    accepted = (
        torch.minimum(synthetic.to(torch.int64).reshape(-1), known.to(torch.int64).reshape(-1))
        .clamp(0, target_tokens.shape[1] - 1)
        .reshape(-1, 1)
    )
    rows = torch.arange(target_tokens.shape[1], device=target_tokens.device).reshape(1, -1)
    target = target_tokens.to(torch.int64)
    # Column Q - 1 is never an accepted draft (accepted <= k).
    tokens = torch.cat([drafts.to(torch.int64), target[:, -1:]], dim=1)
    output = torch.where(rows < accepted, tokens, torch.where(rows == accepted, target, torch.full_like(target, -1)))
    bonus = torch.gather(target, 1, accepted).reshape(-1)
    return output.to(torch.int32), (accepted.reshape(-1) + 1).to(torch.int32), bonus.to(torch.int32)


def select_forced_drafts(
    served: tuple[Tensor, Tensor, Tensor], target_tokens: Tensor, drafts: Tensor, known: Tensor, forced: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Keep the ``served`` result of a row with ``forced`` -1, else give the row its first ``forced`` drafts.

    A forced row follows ``accept_synthetic_drafts`` over the same target tokens. Every row runs both decisions and a
    select picks one, so the program does not depend on the values of ``forced``, and a -1 row keeps the served result
    bit for bit.
    """
    count = forced.to(torch.int32).reshape(-1)
    synthetic = accept_synthetic_drafts(target_tokens, drafts, known, count.clamp(min=0))
    use = count >= 0
    return (
        torch.where(use.reshape(-1, 1), synthetic[0], served[0]),
        torch.where(use, synthetic[1], served[1]),
        torch.where(use, synthetic[2], served[2]),
    )


def accept_forced_drafts(
    target_tokens: Tensor, drafts: Tensor, known: Tensor, forced: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Accept the matching drafts of a row with ``forced`` -1, else its first ``forced`` drafts (greedy drafts)."""
    return select_forced_drafts(
        accept_known_drafts(target_tokens, drafts, known), target_tokens, drafts, known, forced
    )


# Probabilistic drafts.


def spec_step_draw_count(num_speculative_tokens: int) -> int:
    """Return the host draws of a request in one step: ``u_accept [k]``, ``u_final``, ``u_draft [k]``, ``u_out``."""
    return 2 * num_speculative_tokens + 2


def sampled_spec_rows(sampling_params: Tensor, batch: int, queries: int) -> Tensor:
    """Return ``[B]`` bool: the requests whose target samples (temperature >= 1e-5 and a top_k other than 1).

    ``sampling_params`` is ``[B * Q, C]`` request-major with the columns ``[top_k, top_p, temperature, ...]``.
    """
    params = sampling_params.reshape(batch, queries, -1)[:, 0]
    return (params[:, 2] >= SAMPLING_EPS) & (params[:, 0] != 1)


def drafter_sampling_inputs(
    sampling_params: Tensor, draws: Tensor, sampled: Tensor, batch: int, queries: int
) -> tuple[Tensor, Tensor]:
    """Return the drafter kernel inputs ``(sample_params [B, 3], u_draft [B, 7])`` fp32 of a sampled step.

    ``sample_params`` holds ``[top_k, top_p, temperature]`` of each request; a request that does not sample gets
    temperature 0, so the kernel gives it the greedy drafts. ``u_draft`` holds the draws ``u_draft [k]`` of
    ``draws [B, 2k + 2]``, padded with 0.5 to the 7 kernel slots.
    """
    k = queries - 1
    params = sampling_params.reshape(batch, queries, -1)[:, 0, :3].to(torch.float32)
    temperature = torch.where(sampled.reshape(-1), params[:, 2], torch.zeros_like(params[:, 2]))
    params = torch.stack([params[:, 0], params[:, 1], temperature], dim=1)
    u_draft = draws[:, k + 1 : 2 * k + 1].to(torch.float32)
    if k < DRAFTER_SAMPLE_SLOTS:
        u_draft = torch.nn.functional.pad(u_draft, (0, DRAFTER_SAMPLE_SLOTS - k), value=0.5)
    else:
        # At k = 7 the pad has no width, so u_draft would be a strided slice of the draws input copied to an output of
        # the verify program, which neuronx-cc 2.28.4568 stops on ([NCC_IBIR632]). The clamp makes it a computed copy;
        # each host draw is in [0, 1), so the values do not change.
        u_draft = u_draft.clamp(0.0, 1.0)
    return params.contiguous(), u_draft.contiguous()


def cumsum_matmul(tensor: Tensor) -> Tensor:
    """Return the cumulative sum of a 2D tensor along its last axis as a matmul with an upper triangle of ones."""
    size = tensor.shape[-1]
    triu = torch.triu(torch.ones(size, size, dtype=tensor.dtype, device=tensor.device))
    return tensor @ triu


def multinomial(probs: Tensor, uniform: Tensor, cumsum: Callable[[Tensor], Tensor] = cumsum_matmul) -> Tensor:
    """Return the inverse-CDF draw of each row of ``probs [B, K]`` at ``uniform [B]``.

    ``cumsum`` is the last-axis cumulative sum of a 2D tensor: ``cumsum_matmul`` on the CPU, the NKI cumsum kernel in a
    device program.
    """
    if probs.ndim != 2:
        raise ValueError(f"The draw takes 2D probabilities, not shape {tuple(probs.shape)}.")
    cdf = cumsum(probs)
    # The last CDF value is 1 after the division, and set to 1 so no rounding draws past the last column.
    cdf = cdf / cdf[..., -1:]
    cdf[..., -1:] = 1.0
    rand = uniform.to(probs.dtype)
    # A zero draw must not select a masked tie before the first kept token.
    rand = rand.clamp(min=torch.finfo(probs.dtype).tiny)
    return torch.sum(rand.unsqueeze(-1) > cdf, dim=-1)


def _prob_of(probs: Tensor, ids: Tensor, tokens: Tensor) -> Tensor:
    """Return the probability of each token in its list (0 outside it): ``[..., K]`` lists, ``[...]`` tokens."""
    return (probs * (ids == tokens.unsqueeze(-1)).to(probs.dtype)).sum(dim=-1)


def _ascending_order(keys: Tensor, *values: Tensor, shards: int = 1) -> tuple[Tensor, ...]:
    """Return each of ``values [B, K]`` in the stable ascending order of ``keys [B, K]``, with no sort op.

    The rank of column j is the count of the columns with a smaller key, or with the same key and a smaller column.
    Slot r takes the value of the column of rank r: a sum of that value and zeros. With ``shards`` above 1 the order
    is shard-major (``_shard_ascending_order``).
    """
    if shards > 1:
        return _shard_ascending_order(keys, shards, *values)
    width = keys.shape[-1]
    column = torch.arange(width, device=keys.device)
    chunks = [column[start : start + _ORDER_CHUNK] for start in range(0, width, _ORDER_CHUNK)]
    ranks = []
    for part in chunks:
        other = keys.unsqueeze(-1)
        mine = keys[:, part].unsqueeze(-2)
        tie = (other == mine) & (column.reshape(-1, 1) < part.reshape(1, -1))
        ranks.append(((other < mine) | tie).to(torch.int32).sum(dim=-2))
    rank = torch.cat(ranks, dim=-1).unsqueeze(-1)
    ordered = []
    for value in values:
        parts = [
            torch.where(rank == part, value.unsqueeze(-1), torch.zeros_like(value).unsqueeze(-1)).sum(dim=-2)
            for part in chunks
        ]
        ordered.append(torch.cat(parts, dim=-1))
    return tuple(ordered)


def _shard_ascending_order(keys: Tensor, shards: int, *values: Tensor) -> tuple[Tensor, ...]:
    """Return each of ``values [B, K]`` in shard-major order: per vocab shard, its columns in the stable ascending order
    of their keys.

    Column ``j * shards + s`` is candidate ``j`` of shard ``s``. The rank of a column counts the columns of its own
    shard only, so the compares are ``[B, shards, J, J]`` with ``J = K / shards``. When the ids of each shard lie in
    that shard's vocab range, shard-major order is the global ascending id order; a key above the vocab (the -1
    sentinel) goes last inside its own shard.
    """
    batch, width = keys.shape
    if width % shards:
        raise ValueError(f"Width {width} is not a multiple of {shards} shards.")
    per = width // shards

    def blocks(x: Tensor) -> Tensor:
        return x.reshape(batch, per, shards).transpose(1, 2)

    column = torch.arange(per, device=keys.device)
    shard_keys = blocks(keys)
    other = shard_keys.unsqueeze(-1)
    mine = shard_keys.unsqueeze(-2)
    tie = (other == mine) & (column.reshape(-1, 1) < column.reshape(1, -1))
    rank = ((other < mine) | tie).to(torch.int32).sum(dim=-2)
    slot = rank.unsqueeze(-2) == column.reshape(-1, 1)
    ordered = []
    for value in values:
        shard_values = blocks(value).unsqueeze(-2)
        picked = torch.where(slot, shard_values, torch.zeros_like(shard_values))
        ordered.append(picked.sum(dim=-1).reshape(batch, width))
    return tuple(ordered)


def _draw_list(probs: Tensor, ids: Tensor, uniform: Tensor, cumsum: Callable[[Tensor], Tensor]) -> Tensor:
    """Return ``[B]`` int64: the inverse-CDF draw over each list ``probs [B, K]`` with one uniform for each row.

    The index is clamped to the last column with mass, so a rounding of the CDF never picks a column without mass.
    """
    columns = torch.arange(probs.shape[-1], device=probs.device)
    last = torch.where(probs > 0, columns, torch.zeros_like(columns)).amax(dim=-1)
    index = torch.minimum(multinomial(probs, uniform, cumsum), last).clamp(min=0)
    return torch.gather(ids, -1, index.reshape(-1, 1)).reshape(-1)


def _accept_lengths(
    method: str, p_draft: Tensor, q_draft: Tensor, scale: Tensor, mass: Tensor, limit: Tensor, u_accept: Tensor
) -> Tensor:
    """Return the accepted drafts ``[B]`` int64 of the rule ``method`` (at most ``limit`` for each row)."""
    batch, k = p_draft.shape
    lengths = torch.arange(1, k + 1, device=p_draft.device).reshape(1, -1)
    if method == "standard":
        # A draft with q 0 (a zeroed non-finite row) is rejected; 0 < p would accept it untested.
        accept = (q_draft > 0) & (u_accept * q_draft < p_draft)
        run = torch.cumprod(accept.to(torch.int64), dim=1).sum(dim=1)
        return torch.minimum(run, limit)
    # Block length i < limit has h_i = R_i / (R_i + 1 - c_i) (1 if the denominator is 0); length limit has c_limit.
    denom = mass[:, 1:] + 1.0 - scale[:, 1:k]
    h = torch.where(denom > 0, mass[:, 1:] / denom.clamp(min=_TINY), torch.ones_like(denom))
    h = torch.cat([h, torch.ones(batch, 1, device=h.device, dtype=h.dtype)], dim=1)
    h = torch.where(lengths < limit.reshape(-1, 1), h, scale[:, 1:])
    hits = (u_accept <= h) & (lengths <= limit.reshape(-1, 1))
    return torch.where(hits, lengths, torch.zeros_like(lengths)).amax(dim=1)


def accept_probabilistic_drafts(
    p_at: Tensor,
    plain_tokens: Tensor,
    outside_tokens: Tensor,
    q_probs: Tensor,
    q_ids: Tensor,
    drafts: Tensor,
    known: Tensor,
    u_accept: Tensor,
    u_final: Tensor,
    method: str = "block",
    vocab: int | None = None,
    shards: int = 1,
    cumsum: Callable[[Tensor], Tensor] = cumsum_matmul,
) -> tuple[Tensor, Tensor, Tensor]:
    """Verify probabilistic drafts; return ``(output [B, Q], num_accepted [B], bonus [B])`` int32.

    Args:
        p_at: ``[B, k, K]`` the target probability of each q id of each draft row.
        plain_tokens: ``[B, Q]`` a target draw of each verify row whose uniform is independent of the accept draws of
            the rows before it (the sampler's tokens).
        outside_tokens: ``[B, k]`` a draw of each draft row from the target without the q ids of that row, with its
            own uniform; -1 when the q ids hold all the target mass.
        q_probs, q_ids: ``[B, k, K]`` the q list of each draft (fp32 probs over distinct global ids).
        drafts: ``[B, k]`` the drafts of the step (the verify inputs 1..k).
        known: ``[B]`` the known drafts (0 to k); the op tests no draft after them.
        u_accept, u_final: ``[B, k]`` and ``[B]`` uniform draws in [0, 1).
        method: ``"standard"`` or ``"block"``.
        vocab: the vocab size. With it, an id at or above it has no mass, and the last token is clamped to the vocab.
        shards: the vocab shards of the q-list layout (``verify_candidate_layout``); above 1 the residual draw orders S
            shard by shard.
        cumsum: the last-axis cumulative sum of the residual draw (``multinomial``).

    At the stop row the residual is the mixture of its part on S (an inverse-CDF draw over S in ascending id order,
    the mass ``sum max(c p - q, 0)``) and the target outside S (``outside_tokens``, the mass ``c (1 - p(S))``);
    ``u_final`` chooses the part and draws inside S. When all known drafts are accepted the last token is
    ``plain_tokens`` of row ``known``. Clamps: p and q to [0, 1], each residual to [0, 1]; a residual part below
    ``(K + 2) * eps`` is 0, and so is the outside part of a row whose ``outside_tokens`` is -1. A q row with a NaN or
    infinite entry is all 0, so the block stops at that row.
    """
    if method not in PROBABILISTIC_METHODS:
        raise ValueError(f"Probabilistic drafts take the rejection rule {PROBABILISTIC_METHODS}, not {method!r}.")
    batch, k, width = q_probs.shape
    queries = k + 1
    device = drafts.device
    drafts = drafts.to(torch.int64)
    q_ids = q_ids.to(torch.int64)
    valid = (q_ids >= 0) if vocab is None else (q_ids >= 0) & (q_ids < vocab)
    q_ids = torch.where(valid, q_ids, torch.full_like(q_ids, -1))
    # A q row with a non-finite entry has no mass (q 0 truncates the block there); a NaN mass would give h = 1.
    q_probs = q_probs.to(torch.float32)
    q_probs = torch.where(
        torch.isfinite(q_probs).all(dim=-1, keepdim=True) & valid, q_probs.clamp(0.0, 1.0), torch.zeros_like(q_probs)
    )
    p_at = p_at.to(torch.float32)
    p_at = torch.where(torch.isfinite(p_at) & valid, p_at.clamp(0.0, 1.0), torch.zeros_like(p_at))
    # A residual below the FP32 rounding of a sum of K terms is no mass.
    floor = (width + 2) * _EPS
    # A zero draw would pass u <= h at h = 0 (a zero-probability draft); the draws are clamped to the smallest normal.
    u_accept = u_accept.to(torch.float32).clamp(min=_TINY)
    u_final = u_final.to(torch.float32).clamp(min=_TINY)
    p_draft = _prob_of(p_at, q_ids, drafts)
    q_draft = _prob_of(q_probs, q_ids, drafts)
    ratio = torch.where(q_draft > 0, p_draft / q_draft.clamp(min=_TINY), torch.zeros_like(p_draft))
    # FP32 by name: SGLang's model loader sets torch's default dtype to the model dtype (bf16).
    scales = [torch.ones(batch, dtype=torch.float32, device=device)]
    for i in range(k):
        scales.append(torch.clamp(scales[-1] * ratio[:, i], max=1.0) if method == "block" else scales[0])
    scale = torch.stack(scales, dim=1)
    inside = torch.clamp(scale[:, :k].unsqueeze(-1) * p_at - q_probs, 0.0, 1.0)
    inside = torch.where((inside.sum(dim=-1) < floor).unsqueeze(-1), torch.zeros_like(inside), inside)
    outside_p = torch.clamp(1.0 - p_at.sum(dim=-1), 0.0, 1.0)
    # No target mass outside the q ids (outside token -1): the FP32 sum of p can round below 1, so the part is 0.
    no_outside = outside_tokens < 0
    if vocab is not None:
        no_outside = no_outside | (outside_tokens >= vocab)
    outside_p = torch.where((outside_p < floor) | no_outside, torch.zeros_like(outside_p), outside_p)
    outside = scale[:, :k] * outside_p
    mass = inside.sum(dim=-1) + outside
    limit = known.to(torch.int64).reshape(-1).clamp(0, k)
    # A draft whose q row has no mass has no test (the standard rule would accept it at q 0): stop before it.
    with_q = torch.cumprod((q_probs.sum(dim=-1) > 0).to(torch.int64), dim=1).sum(dim=1)
    limit = torch.minimum(limit, with_q)
    accepted = _accept_lengths(method, p_draft, q_draft, scale, mass, limit, u_accept).clamp(0, k)

    u = u_final.to(torch.float32).reshape(-1)
    row = accepted.clamp(0, k - 1)
    pick = row.reshape(-1, 1, 1).expand(batch, 1, width)
    in_res = torch.gather(inside, 1, pick).reshape(batch, width)
    in_ids = torch.gather(q_ids, 1, pick).reshape(batch, width)
    out_mass = torch.gather(outside, 1, row.reshape(-1, 1)).reshape(-1)
    in_mass = in_res.sum(dim=-1)
    total = in_mass + out_mass
    plain = torch.gather(plain_tokens.to(torch.int64), 1, accepted.reshape(-1, 1)).reshape(-1)
    outside_token = torch.gather(outside_tokens.to(torch.int64), 1, row.reshape(-1, 1)).reshape(-1)
    # Ascending id order with no device sort; an id below 0 has no mass and goes last.
    key = torch.where(in_ids >= 0, in_ids, torch.full_like(in_ids, _LAST_KEY))
    in_res, sorted_ids = _ascending_order(key, in_res, in_ids, shards=shards)
    in_probs = torch.where(
        in_mass.reshape(-1, 1) > 0,
        in_res / in_mass.clamp(min=_TINY).reshape(-1, 1),
        torch.full_like(in_res, 1.0 / width),
    )
    scaled = u * total
    u_in = (scaled / in_mass.clamp(min=_TINY)).clamp(0.0, _BELOW_ONE)
    inside_token = _draw_list(in_probs, sorted_ids, u_in, cumsum)
    take_inside = (scaled < in_mass) | (out_mass <= 0)
    # A residual with no mass has probability 0 (it needs p = q); it takes the plain draw.
    residual = torch.where(total > 0, torch.where(take_inside, inside_token, outside_token), plain)
    final = torch.where(accepted == limit, plain, residual)
    if vocab is not None:
        final = final.clamp(0, vocab - 1)

    columns = torch.arange(queries, device=device).reshape(1, -1)
    padded = torch.cat([drafts, torch.zeros(batch, 1, dtype=torch.int64, device=device)], dim=1)
    output = torch.where(columns < accepted.reshape(-1, 1), padded, torch.full_like(padded, -1))
    output = torch.where(columns == accepted.reshape(-1, 1), final.reshape(-1, 1), output)
    return output.to(torch.int32), (accepted + 1).to(torch.int32), final.to(torch.int32)


def verify_candidate_layout(values: Tensor, tp_size: int) -> Tensor:
    """Return the drafter q lists ``[..., K]`` (rank-major blocks) in the candidate layout of the full-vocab verify.

    The drafter gathers the top ``K / tp_size`` candidates of each vocab shard in rank order; the full-vocab kernel
    takes column ``j * tp_size + s`` = candidate ``j`` of shard ``s``.
    """
    *lead, width = values.shape
    per_shard = width // tp_size
    return values.reshape(*lead, tp_size, per_shard).transpose(-1, -2).reshape(*lead, width).contiguous()


def merge_sampled_rows(
    greedy: tuple[Tensor, Tensor, Tensor], probabilistic: tuple[Tensor, Tensor, Tensor], sampled: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``(output, num_accepted, bonus)``: the probabilistic result for a sampled row, else the greedy one."""
    pick = sampled.reshape(-1)
    return (
        torch.where(pick.reshape(-1, 1), probabilistic[0], greedy[0]),
        torch.where(pick, probabilistic[1], greedy[1]),
        torch.where(pick, probabilistic[2], greedy[2]),
    )


# q lists of the verify and the drafter.


def spec_verify_candidates(
    prev_row: Tensor,
    prev_q_probs: Tensor,
    prev_q_ids: Tensor,
    fresh: Tensor,
    draws: Tensor,
    batch: int,
    queries: int,
    vocab: int,
    tp_size: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return the verify inputs of probabilistic drafts: ``(q_probs, q_ids [B, k, K], candidate_ids, excl_u)``.

    The q lists of each row come from its row of the last step (``prev_row``), in the candidate layout of
    ``verify_candidate_layout``. ``candidate_ids [B * Q, K]`` holds them for the k draft rows and -1 for the bonus row;
    ``excl_u [B * Q, 1]`` is the request's ``u_out`` (the last draw column) on each of its rows. A fresh row has no q
    lists (the host gives zeros), so its ids take the -1 sentinel and no q mass.
    """
    index = prev_row.to(torch.int64).clamp(0, prev_q_ids.shape[0] - 1)
    q_probs, q_ids = (torch.index_select(lists, 0, index) for lists in (prev_q_probs, prev_q_ids))
    valid = (q_ids >= 0) & (q_ids < vocab) & (fresh.reshape(-1, 1, 1) == 0)
    q_ids = torch.where(valid, q_ids, torch.full_like(q_ids, -1))
    q_probs = torch.where(valid, q_probs, torch.zeros_like(q_probs))
    q_probs, q_ids = (verify_candidate_layout(t, tp_size) for t in (q_probs, q_ids))
    bonus_row = torch.full_like(q_ids[:, :1], -1)
    candidate_ids = torch.cat([q_ids, bonus_row], dim=1).reshape(batch * queries, -1)
    excl_u = draws[:, -1:].repeat_interleave(queries, dim=0)
    return q_probs, q_ids, candidate_ids.to(torch.int32), excl_u.to(torch.float32)


def next_q_lists(q_lists: tuple[Tensor, ...] | list[Tensor], batch: int, queries: int) -> tuple[Tensor, ...]:
    """Return the q lists ``[B, k, width]`` of the next drafts from the drafter q lists of its 7 slots."""
    return tuple(q.reshape(batch, DRAFTER_SAMPLE_SLOTS, -1)[:, : queries - 1].contiguous() for q in q_lists)


def request_flags(flags: Tensor, batch: int) -> Tensor:
    """Return the bitwise OR ``[B]`` of the request-major row flags ``[B * Q]`` of each request."""
    rows = flags.reshape(batch, -1)
    bits = [ROW_FLAG_NAN, ROW_FLAG_NO_TOKEN, ROW_FLAG_LAST_ID]
    return sum(torch.bitwise_and(rows, bit).amax(dim=1) for bit in bits).to(torch.int32)


# Step compositions, in the op order of the served step.


class AcceptOutputs(NamedTuple):
    """The acceptance of a verify step: ``tokens [B, Q]`` (-1 after the accepted prefix), ``accepted [B]``,
    ``bonus [B]`` int32, and the drafter sampling inputs ``(sample_params [B, 3], u_draft [B, 7])`` (sampled steps)."""

    tokens: Tensor
    accepted: Tensor
    bonus: Tensor
    drafter_sampling: tuple[Tensor, Tensor] | None


def accept_spec_step(
    target: Tensor,
    input_ids: Tensor,
    known: Tensor,
    *,
    batch: int,
    queries: int,
    vocab: int,
    probabilistic: bool,
    forced: Tensor | None = None,
    synthetic: Tensor | None = None,
    sampled_inputs: dict[str, Tensor] | None = None,
    method: str = "block",
    tp_size: int = 32,
    cumsum: Callable[[Tensor], Tensor] = cumsum_matmul,
) -> AcceptOutputs:
    """Accept the drafts of one verify step from the target tokens, in the op order of the served step.

    Args:
        target: ``[B * Q]`` request-major target tokens (the sampler draw of each row); a probabilistic step clamps
            them to the vocab first.
        input_ids: ``[B, Q]`` the carried input ids (``SpecCarry.input_ids``); columns 1.. are the drafts.
        known: ``[B]`` the known drafts of each request.
        probabilistic: the step takes probabilistic drafts (a sampled step with the full-vocab verify).
        forced: ``[B]`` forced counts (-1 keeps the served result), or None.
        synthetic: ``[B]`` synthetic counts (a greedy step without forced counts), or None.
        sampled_inputs: with ``probabilistic``: ``p_at [B * Q, K]`` and ``excl_token [B * Q]`` of the sampler,
            ``q_probs`` / ``q_ids`` ``[B, k, K]`` (``spec_verify_candidates``), ``sampling_params [B * Q, C]``,
            ``fresh [B]`` and ``draws [B, 2k + 2]``.
        cumsum: the cumulative sum of the residual draw: ``cumsum_matmul`` on the CPU; the verify program passes the
            NKI cumsum kernel (grid 1 at LNC1), which the served program runs there.
    """
    if probabilistic:
        # The sampled tokens feed the acceptance and the next step, so they stay in the vocab.
        target = target.clamp(0, vocab - 1)
    known_drafts = known.clamp(0, queries - 1) if probabilistic else known
    if forced is not None and not probabilistic:
        tokens, accepted, bonus = accept_forced_drafts(
            target.reshape(batch, queries), input_ids[:, 1:], known_drafts, forced
        )
    elif synthetic is not None:
        tokens, accepted, bonus = accept_synthetic_drafts(
            target.reshape(batch, queries), input_ids[:, 1:], known_drafts, synthetic
        )
    else:
        tokens, accepted, bonus = accept_known_drafts(target.reshape(batch, queries), input_ids[:, 1:], known_drafts)
    drafter_sampling = None
    if probabilistic:
        if sampled_inputs is None:
            raise ValueError("A probabilistic step takes the sampler and q-list inputs (sampled_inputs).")
        k = queries - 1
        sampling_params = sampled_inputs["sampling_params"]
        sampled = sampled_spec_rows(sampling_params, batch, queries)
        # A fresh row has no q lists of its drafts on the device, so its sampled request verifies none.
        known = torch.where(
            sampled_inputs["fresh"].to(torch.int32) != 0, torch.zeros_like(known_drafts), known_drafts
        )
        draws = sampled_inputs["draws"]
        tokens, accepted, bonus = merge_sampled_rows(
            (tokens, accepted, bonus),
            accept_probabilistic_drafts(
                sampled_inputs["p_at"].reshape(batch, queries, -1)[:, :k],
                target.reshape(batch, queries),
                sampled_inputs["excl_token"].reshape(batch, queries)[:, :k],
                sampled_inputs["q_probs"],
                sampled_inputs["q_ids"],
                input_ids[:, 1:],
                known,
                draws[:, :k],
                draws[:, k],
                method=method,
                vocab=vocab,
                shards=tp_size,
                cumsum=cumsum,
            ),
            sampled,
        )
        if forced is not None:
            # The served verify runs, then a row with a count takes its first forced drafts; a -1 row keeps the
            # served result bit for bit.
            tokens, accepted, bonus = select_forced_drafts(
                (tokens, accepted, bonus), target.reshape(batch, queries), input_ids[:, 1:], known, forced
            )
        drafter_sampling = drafter_sampling_inputs(sampling_params, draws, sampled, batch, queries)
    return AcceptOutputs(tokens, accepted, bonus, drafter_sampling)


def draft_next_ids(
    bonus: Tensor, block: Tensor, q_lists: tuple[Tensor, ...] | None, batch: int, queries: int, vocab: int
) -> tuple[Tensor, tuple[Tensor, ...]]:
    """Return ``(next_ids [B, Q], next q lists)`` from the drafter outputs: the bonus token, then the first k drafts of
    the drafter block ``[B, 7]``, clamped to the vocab; the q lists of the first k slots (sampled steps)."""
    next_q: tuple[Tensor, ...] = ()
    if q_lists is not None:
        next_q = next_q_lists(q_lists, batch, queries)
    block = block.clamp(0, vocab - 1)
    next_ids = torch.cat([bonus.reshape(-1, 1), block[:, : queries - 1].to(torch.int32)], dim=1)
    return next_ids, next_q
