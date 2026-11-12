# SPDX-License-Identifier: Apache-2.0
"""One Kimi-K3 prefill pipeline stage on one rank of its group, in the sequence-parallel form.

Stage 0 embeds the step's tokens into this rank's row block; a later stage reads the row block from the PP seam.
Every stage zeroes the pad rows of its input stream and bank, scores the bank once for all its AttnRes calls (when
the backend takes a table), runs its layers, and adds the drafter fc partials of its consumer layers.

Seam payload to the next stage (per rank):

* ``hidden`` ``[Tl, H]`` bf16: the stream row block;
* ``bank`` ``[entries * Tl, H]`` bf16, entry-major: the bank row block, one ``Tl``-row block per entry;
* ``draft_acc`` ``[T, H / world]`` fp32: the drafter accumulator, all rows, this rank's columns.

The last stage writes the drafter rows and returns the logits of the sampled rows: it mixes their stream with their
bank (``output_attn_res``), applies the final norm and the vocab-parallel LM head.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from sglang_neuron.models.kimi_k3.prefill.attn_res import (
    TorchAttnRes,
    attn_res_mix,
    bank_from_seam,
    bank_to_seam,
    build_bank_scores,
    score_weight,
)
from sglang_neuron.models.kimi_k3.prefill.comm import StageComm, gather_owned_rows, valid_row_mask
from sglang_neuron.models.kimi_k3.prefill.geometry import KimiK3Geometry
from sglang_neuron.models.kimi_k3.prefill.layer import KimiK3PrefillLayer, RmsNorm, ScoreProj, StepRows
from sglang_neuron.models.kimi_k3.prefill.stage_plan import StagePlan

TAIL_FFN_ROWS = 128
"""Rows of the last layer's FFN part on the last stage: the sampled rows, then copies of the last one (the MoE and
the AttnRes kernels tile rows by 128)."""


@dataclass
class SeamPayload:
    """The tensors one rank sends to the same rank of the next stage (``entries`` is static)."""

    hidden: torch.Tensor
    bank: torch.Tensor
    entries: int
    draft_acc: torch.Tensor | None = None


@dataclass
class StageStep:
    """The inputs of one stage call.

    Args:
        positions: int64 ``[T]`` global positions.
        valid_len: int64 ``[1]``; rows at or after it are pad rows.
        input_ids: int64 ``[T]`` on stage 0 (every rank).
        seam: The payload of the previous stage on a later stage.
        attn: Layer -> its KDA state or MLA cache.
        sampled_rows: int64 ``[n]`` rows whose logits the last stage returns (None: the last valid row).
        draft_rope: fp32 ``[T, 128]`` drafter rope rows (writer stage).
        draft_slots: int32 ``[T]`` drafter row write slots, -1 for rows this rank does not own (writer stage).
        draft_pools: One uint8 ``[blocks, 128, 576]`` drafter row pool per drafter layer (writer stage).
        state_io: The KDA states read from and written back to their pools inside the stage call (an object with
            ``read() -> {layer: KdaState}`` and ``write({layer: state})``), or None: ``attn`` holds them.
    """

    positions: torch.Tensor
    valid_len: torch.Tensor
    input_ids: torch.Tensor | None = None
    seam: SeamPayload | None = None
    attn: Mapping[int, Any] = field(default_factory=dict)
    sampled_rows: torch.Tensor | None = None
    draft_rope: torch.Tensor | None = None
    draft_slots: torch.Tensor | None = None
    draft_pools: Sequence[torch.Tensor] | None = None
    state_io: Any = None


@dataclass
class StageOutput:
    """``seam`` on a stage with a successor, ``logits`` ``[n, V]`` fp32 on the last stage; ``attn`` holds each
    layer's new KDA state (MLA caches are written in place and returned as given)."""

    seam: SeamPayload | None
    logits: torch.Tensor | None
    attn: dict[int, Any]


class StageTail(nn.Module):
    """The last stage's output head (checkpoint names ``model.output_attn_res_norm``, ``model.output_attn_res_proj``,
    ``model.norm``, ``lm_head``); ``lm_head.weight`` is this rank's vocab rows ``[V / world, H]``. With
    ``gather_output`` False the logits stay this rank's vocabulary shard (a sampler that reads the shards)."""

    gather_output = True

    def __init__(self, hidden: int, vocab_local: int, eps: float, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.output_attn_res_norm = RmsNorm(hidden, eps, dtype)
        self.output_attn_res_proj = ScoreProj(hidden, dtype)
        self.norm = RmsNorm(hidden, eps, dtype)
        self.lm_head = nn.Linear(hidden, vocab_local, bias=False, dtype=dtype)
        self.lm_head.weight.requires_grad_(False)

    def forward(self, stream: torch.Tensor, bank: torch.Tensor, comm: StageComm) -> torch.Tensor:
        """``stream [n, H]`` and ``bank [n, C, H]`` of the sampled rows (every rank) -> logits ``[n, V]`` fp32
        (``[n, V / world]`` without ``gather_output``)."""
        score_w = score_weight(self.output_attn_res_norm.weight, self.output_attn_res_proj.weight)
        mixed = attn_res_mix(stream, bank, score_w, self.output_attn_res_norm.eps)
        logits = self.lm_head(self.norm(mixed))
        return (comm.all_gather(logits, dim=-1) if self.gather_output else logits).float()


def zero_pad_rows(hidden: torch.Tensor, bank: torch.Tensor | None,
                  keep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Set the pad rows of a row block and its bank to 0 by select, so a NaN in a pad row from an earlier stage
    cannot reach a valid row through a kernel that multiplies pad rows by 0; valid rows keep their bits."""
    zero = torch.zeros((), dtype=hidden.dtype, device=hidden.device)
    hidden = torch.where(keep.reshape(-1, 1), hidden, zero)
    if bank is not None:
        bank = torch.where(keep.reshape(-1, 1, 1), bank, zero.to(bank.dtype))
    return hidden, bank


def tail_ffn_rows(sampled: torch.Tensor, num_rows: int, world: int, tail: int = TAIL_FFN_ROWS) -> torch.Tensor | None:
    """The rows of the last layer's FFN part: the sampled rows, then copies of the last one up to ``tail`` rows.
    None (all rows) when the step has no more than ``tail`` rows, more samples than that, or ``tail`` does not split
    over the group."""
    index = sampled.reshape(-1).to(torch.int64)
    n = index.numel()
    if tail <= 0 or num_rows <= tail or n == 0 or n > tail or tail % world:
        return None
    return torch.cat([index, index[-1:].expand(tail - n)])


class KimiK3PrefillStage(nn.Module):
    """One prefill stage rank.

    Args:
        geometry: The model geometry.
        plan: The stage plan.
        modules: ``load_stage_rank``'s modules: ``embed`` (stage 0), ``layers.<i>.attn``, ``layers.<i>.ffn``,
            ``layers.<i>.norms`` (each stage layer), ``final`` (last stage), ``draft`` (when the plan has a drafter).
        attn_res: The AttnRes backend (``TorchAttnRes`` when None).
        tail_rows: Rows of the last layer's FFN part on the last stage (0: all rows).
        draft_row_writer: The drafter row writer of the writer stage (``DraftContext.write``'s ``writer``; None: the
            torch scatter). The kernel backend gives ``page_io.kernel_row_writer``.
    """

    def __init__(self, geometry: KimiK3Geometry, plan: StagePlan, modules: Mapping[str, nn.Module],
                 attn_res: Any = None, tail_rows: int = TAIL_FFN_ROWS, draft_row_writer: Any = None):
        super().__init__()
        self.geometry, self.plan = geometry, plan
        self.draft_row_writer = draft_row_writer
        self.attn_res = TorchAttnRes() if attn_res is None else attn_res
        self.tail_rows = tail_rows
        self.embed = modules["embed"] if plan.is_first else None
        self.final = modules["final"] if plan.is_last else None
        self.draft = modules.get("draft") if plan.drafter else None
        if plan.drafter and self.draft is None:
            raise ValueError("The stage plan has a drafter but the modules have no 'draft'")
        self.layers = nn.ModuleList([
            KimiK3PrefillLayer(p, modules[f"layers.{p.index}.attn"], modules[f"layers.{p.index}.ffn"],
                               modules[f"layers.{p.index}.norms"], self.attn_res) for p in plan.layer_plans])

    def _score_ws(self) -> tuple[torch.Tensor, float] | None:
        """The score vectors ``[L, H]`` fp32 of the stage's AttnRes calls in call order, and their one eps."""
        if not self.plan.calls:
            return None
        by_index = {layer.plan.index: layer for layer in self.layers}
        sites = [by_index[idx].norms.site(site) for idx, site, _ in self.plan.calls]
        eps = {e for _, e in sites}
        if len(eps) != 1:
            raise ValueError(f"The AttnRes norms of the stage have eps values {sorted(eps)}; one bank-score table "
                             "needs one")
        return torch.stack([w for w, _ in sites]), eps.pop()

    def forward(self, step: StageStep, comm: StageComm) -> StageOutput:
        plan = self.plan
        num_rows = step.positions.shape[0]
        if num_rows % comm.world:
            raise ValueError(f"{num_rows} rows do not split into {comm.world} row blocks")
        local_rows = num_rows // comm.world
        if plan.is_first:
            hidden, bank = self.embed(step.input_ids, comm), None
        else:
            seam = step.seam
            if seam is None:
                raise ValueError(f"Stage {plan.pp_rank} needs the seam payload of the previous stage")
            if seam.entries != plan.bank_in:
                raise ValueError(f"The seam carries {seam.entries} bank entries; stage {plan.pp_rank} reads "
                                 f"{plan.bank_in}")
            hidden, bank = seam.hidden, bank_from_seam(seam.bank, plan.bank_in, local_rows)
        hidden, bank = zero_pad_rows(hidden, bank, valid_row_mask(step.valid_len, num_rows, comm))

        draft_acc = None
        if self.draft is not None:
            draft_acc = self.draft.empty_acc(num_rows, hidden.device) if plan.is_first else step.seam.draft_acc

        scores = None
        score_ws = self._score_ws()
        if score_ws is not None:
            scores = build_bank_scores(self.attn_res, plan.calls, score_ws[0], hidden, bank, score_ws[1])

        sampled = step.sampled_rows
        if plan.is_last and sampled is None:
            sampled = (step.valid_len.reshape(-1)[:1] - 1).to(torch.int64)
        ffn_rows = tail_ffn_rows(sampled, num_rows, comm.world, self.tail_rows) if plan.is_last else None

        rows = StepRows(step.positions, step.valid_len)
        attn = step.attn
        if step.state_io is not None:
            # the state pool reads and writes are this module's ops: neuronx-cc's hierarchy splitter partitions the
            # stage program by module, and a last stage whose wrapper module holds them fails its MacroGeneration
            attn = {**attn, **step.state_io.read()}
        new_attn: dict[int, Any] = {}
        for layer in self.layers:
            index = layer.plan.index
            if draft_acc is not None and layer.plan.draft_consumer:
                aux = comm.all_gather(hidden.to(torch.bfloat16), dim=0)
                draft_acc = self.draft.add(draft_acc, index, aux)
            hidden, bank, new_attn[index] = layer(hidden, bank, attn.get(index), rows, comm, scores,
                                                  ffn_rows=ffn_rows if index == plan.last_layer else None)
        if scores is not None:
            scores.check_done()
        if step.state_io is not None:
            step.state_io.write(new_attn)

        if not plan.is_last:
            return StageOutput(SeamPayload(hidden, bank_to_seam(bank), plan.bank_out, draft_acc), None, new_attn)

        if self.draft is not None and self.draft.writer:
            # row 0 is always a valid row, so its position is the chunk's start
            self.draft.write(draft_acc, step.draft_rope, step.draft_slots, step.draft_pools, comm,
                             writer=self.draft_row_writer, start=step.positions.reshape(-1)[:1].to(torch.int64))
        n = sampled.reshape(-1).shape[0]
        hidden = hidden[:n] if ffn_rows is not None else gather_owned_rows(hidden, sampled, comm)
        bank = gather_owned_rows(bank, sampled, comm)
        return StageOutput(None, self.final(hidden, bank, comm), new_attn)
