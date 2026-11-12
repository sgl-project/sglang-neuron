# SPDX-License-Identifier: Apache-2.0
"""One Kimi-K3 decoder layer of a prefill stage in the sequence-parallel (SP) form.

Each rank holds one row block of the stream and of the AttnRes bank: rows ``rank * Tl .. rank * Tl + Tl - 1`` of
the step's ``T`` rows (``Tl = T / world``). The two AttnRes mixtures, the bank append, the norms and the residual
adds run on the row block. The attention and the FFN take the normed row block and return their output row block;
they gather the rows of all ranks themselves and reduce-scatter their output projections by row.

The layer (``model.py`` order of the checkpoint modeling code)::

    mix = attn_res(prefix, bank)            # when the bank is not empty
    if layer % block == 0: bank += [prefix]; prefix = 0
    prefix += attention(input_layernorm(mix))
    mix = attn_res(prefix, bank)
    prefix += ffn(post_attention_layernorm(mix))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from sglang_neuron.models.kimi_k3.prefill.attn_res import BankScores, score_weight
from sglang_neuron.models.kimi_k3.prefill.comm import StageComm, gather_owned_rows, local_row_block
from sglang_neuron.models.kimi_k3.prefill.stage_plan import ATTN_SITE, FFN_SITE, MOE, LayerPlan


class RmsNorm(nn.Module):
    """The Kimi RMSNorm: normalize in fp32, round to the input dtype, then multiply by the weight."""

    def __init__(self, dim: int, eps: float, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype), requires_grad=False)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * xf.to(x.dtype)).to(x.dtype)


class ScoreProj(nn.Module):
    """An AttnRes score projection ``nn.Linear(H, 1)`` (weight ``[1, H]``); only its weight is read."""

    def __init__(self, dim: int, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, dim, dtype=dtype), requires_grad=False)


class LayerNorms(nn.Module):
    """The norms and AttnRes projections of one layer, under the checkpoint names (``model.layers.<i>.<name>``)."""

    def __init__(self, hidden: int, eps: float, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.input_layernorm = RmsNorm(hidden, eps, dtype)
        self.post_attention_layernorm = RmsNorm(hidden, eps, dtype)
        self.self_attention_res_norm = RmsNorm(hidden, eps, dtype)
        self.mlp_res_norm = RmsNorm(hidden, eps, dtype)
        self.self_attention_res_proj = ScoreProj(hidden, dtype)
        self.mlp_res_proj = ScoreProj(hidden, dtype)

    def site(self, site: str) -> tuple[torch.Tensor, float]:
        """``(score vector [H] fp32, eps)`` of the AttnRes call at ``site``."""
        norm, proj = ((self.self_attention_res_norm, self.self_attention_res_proj) if site == ATTN_SITE else
                      (self.mlp_res_norm, self.mlp_res_proj))
        return score_weight(norm.weight, proj.weight), norm.eps


@dataclass
class StepRows:
    """The row facts of one step: ``positions`` int64 ``[T]`` (global), ``valid_len`` int64 ``[1]`` (rows at or
    after it are pad rows)."""

    positions: torch.Tensor
    valid_len: torch.Tensor

    @property
    def num_rows(self) -> int:
        return self.positions.shape[0]


class KimiK3PrefillLayer(nn.Module):
    """One layer in the SP form, around an attention module and an FFN module.

    Args:
        plan: The layer's plan (kind, bank append).
        attn: ``KdaAttention`` (``forward(x_rows, state, valid_len, comm) -> (y, state)``) or ``MlaAttention``
            (``forward(x_rows, positions, cache, valid_len, comm) -> y``, cache written in place).
        ffn: ``LatentMoe`` (``forward(x_rows, comm, valid_len) -> y``) or ``DenseMlp`` (``forward(x_rows, comm)``).
        norms: The layer's ``LayerNorms``.
        attn_res: The AttnRes backend (``attn_res.TorchAttnRes``, ``KernelAttnRes``, ...).
    """

    def __init__(self, plan: LayerPlan, attn: nn.Module, ffn: nn.Module, norms: LayerNorms, attn_res: Any):
        super().__init__()
        self.plan = plan
        self.attn = attn
        self.ffn = ffn
        self.norms = norms
        self.attn_res = attn_res

    def _mix(self, prefix: torch.Tensor, bank: torch.Tensor, site: str, scores: BankScores | None) -> torch.Tensor:
        score_w, eps = self.norms.site(site)
        if scores is None:
            return self.attn_res(prefix, bank, score_w, eps)
        row = scores.take(self.plan.index, site, bank.shape[1])
        return self.attn_res(prefix, bank, score_w, eps, bank_scores=row)

    def _ffn(self, x: torch.Tensor, comm: StageComm, valid_len: torch.Tensor | None) -> torch.Tensor:
        """The FFN of the row block; the MoE gives the pad rows (at or after ``valid_len``) no routed expert."""
        if self.plan.ffn == MOE:
            return self.ffn(x, comm, valid_len=valid_len)
        return self.ffn(x, comm)

    def forward(self, stream: torch.Tensor, bank: torch.Tensor | None, attn_state: Any, rows: StepRows,
                comm: StageComm, scores: BankScores | None = None, ffn_rows: torch.Tensor | None = None):
        """Run the layer on this rank's row block.

        Args:
            stream: ``[Tl, H]`` the input stream (the running prefix of the block).
            bank: ``[Tl, C, H]`` the bank, or None before the first entry.
            attn_state: The KDA state or the MLA cache of the layer.
            rows: The step's positions and ``valid_len``.
            comm: The stage collectives.
            scores: The stage's bank-score table, or None.
            ffn_rows: ``[n]`` global rows (``n % world == 0``): the FFN part runs on these rows only, and the layer
                returns their stream ``[n, H]`` on every rank (the tail of the last layer). The bank stays the row
                block.

        Returns:
            ``(stream [Tl, H] or [n, H], bank [Tl, C', H], new attention state)``.
        """
        prefix = stream
        mix = prefix if bank is None else self._mix(prefix, bank, ATTN_SITE, scores)
        if self.plan.appends_bank:
            snapshot = prefix.unsqueeze(1)
            bank = snapshot if bank is None else torch.cat([bank, snapshot], dim=1)
            prefix = None
        if bank is None:
            raise ValueError(f"Layer {self.plan.index} has no AttnRes bank at its FFN mixture: the stage input must "
                             "carry the bank of the earlier layers")
        x = self.norms.input_layernorm(mix)
        if self.plan.is_mla:
            out = self.attn(x, rows.positions, attn_state, rows.valid_len, comm)
            new_state = attn_state
        else:
            out, new_state = self.attn(x, attn_state, rows.valid_len, comm)
        prefix = out if prefix is None else prefix + out
        mix = self._mix(prefix, bank, FFN_SITE, scores)
        if ffn_rows is None:
            prefix = prefix + self._ffn(self.norms.post_attention_layernorm(mix), comm, rows.valid_len)
            return prefix, bank, new_state
        hidden = prefix.shape[-1]
        both = gather_owned_rows(torch.cat([prefix, mix.to(prefix.dtype)], dim=-1), ffn_rows, comm)
        prefix, mix = both[:, :hidden], both[:, hidden:]
        # The selected rows are real rows: no pad mask.
        part = self._ffn(self.norms.post_attention_layernorm(local_row_block(mix, comm)), comm, None)
        return prefix + comm.all_gather(part, dim=0), bank, new_state
