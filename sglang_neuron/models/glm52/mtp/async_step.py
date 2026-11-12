# SPDX-License-Identifier: Apache-2.0
"""Output side of an async MTP decode step: the token table the host reads, and the next device carry.

With overlap scheduling, step N+1 launches before the host reads step N's tokens, so the drafts stay on the device in
a carry indexed by kernel slot. A request keeps its slot for life while its batch row can move, so the carry needs no
remapping when the batch changes. The slot's committed length lives in ``mtp.step.commit``.
"""

from __future__ import annotations

import torch


def step_outputs(
    target: torch.Tensor,
    last: torch.Tensor,
    verify: torch.Tensor,
    live: torch.Tensor,
    next_token: torch.Tensor,
    drafts: torch.Tensor,
    row_slots: torch.Tensor,
    carry_tokens: torch.Tensor,
    vocab_size: int,
    kept_drafts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the host token table ``[rows, spec]`` and the next carry ``[num_slots, spec]``.

    A verifying row emits the target's tokens for rows ``0 .. last`` (the accepted drafts, then the bonus or
    correction token); any other row emits its last row's token in column 0. Everything else is -1.

    Under synthetic acceptance a kept draft need not match the target's token, so ``kept_drafts`` (the verified
    drafts, ``acceptance``'s ``drafts_in``) replaces rows ``0 .. last - 1``. ``row_slots`` maps each batch row to
    its slot (0 for padding). A slot that did not run keeps its carry, and a draft outside the vocabulary becomes
    token 0, which the next step simply rejects.
    """
    spec = target.shape[1]
    t = target.to(torch.int64)
    last = last.to(torch.int64)
    j = torch.arange(spec, dtype=torch.int64, device=t.device).reshape(1, spec)
    minus_one = torch.full_like(t, -1)
    at_last = torch.gather(t, 1, last.reshape(-1, 1))
    kept = t
    if kept_drafts is not None:
        kept = torch.where(j < last.reshape(-1, 1), kept_drafts.to(torch.int64), t)
    table = torch.where(
        verify.reshape(-1, 1),
        torch.where(j <= last.reshape(-1, 1), kept, minus_one),
        torch.where(j == 0, at_last.expand_as(t), minus_one),
    )
    sampled = table.index_select(0, row_slots.to(torch.int64))
    d = drafts.to(torch.int64)
    d = torch.where((d < 0) | (d >= vocab_size), torch.zeros_like(d), d)
    fresh = torch.cat([next_token.to(torch.int64).reshape(-1, 1), d], dim=1)
    tokens = torch.where(live.reshape(-1, 1), fresh, carry_tokens.to(torch.int64))
    return sampled.to(torch.int32), tokens.to(torch.int32)


__all__ = ["step_outputs"]
