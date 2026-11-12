# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 attention residuals (AttnRes) on a prefill row block: the bank, the mixture and its backends.

Each layer mixes the live stream with the bank of earlier stream snapshots twice: before its attention (when the
bank is not empty) and before its FFN. A mixture scores each candidate ``v`` (bank entries, then the stream) as
``dot(rmsnorm(v), w)`` with ``w = norm.weight * proj.weight`` and returns the softmax-weighted sum of the RAW
candidates in fp32, rounded once to the input dtype.

Backends share one call contract, ``backend(stream, bank, score_w, eps, bank_scores=None) -> [T, H]``:

* ``TorchAttnRes``: the reference numerics in torch (the CPU backend). It takes no bank scores.
* ``TorchScoredAttnRes``: the kernel's factored algebra in torch with a torch bank-score table, so the scored path
  of a stage runs on a host.
* ``KernelAttnRes``: the ``attn_res_cte`` NKI kernel with the stage's bank-score table from
  ``attn_res_bank_scores`` (the device backend).

Bank layout: inside a stage the bank is token-major ``[Tl, C, H]`` (the kernel layout). At a PP seam it is
entry-major ``[C * Tl, H]``, so the first ``C * Tl`` rows of a seam buffer hold a ``C``-entry bank.
"""

from __future__ import annotations

import inspect

import torch

from sglang_neuron.models.kimi_k3.prefill import kernel_imports
from sglang_neuron.models.kimi_k3.prefill.stage_plan import ATTN_SITE, FFN_SITE  # noqa: F401 (re-export)

# attn_res_cte tiles rows by 128 and the hidden axis by 512, and takes 1..15 bank entries.
TOKEN_TILE = 128
HIDDEN_CHUNK = 512
MAX_BANK = 15
MAX_TABLE_CALLS = 128
KERNEL_DTYPES = (torch.bfloat16, torch.float32)


def score_weight(norm_weight: torch.Tensor, proj_weight: torch.Tensor) -> torch.Tensor:
    """The fp32 score vector ``norm.weight * proj.weight`` (``[H]``; ``proj_weight`` is ``[1, H]`` or ``[H]``)."""
    return (norm_weight.float() * proj_weight.reshape(-1).float()).contiguous()


def attn_res_mix(stream: torch.Tensor, bank: torch.Tensor, score_w: torch.Tensor, eps: float) -> torch.Tensor:
    """The reference AttnRes mixture.

    Args:
        stream: ``[T, H]`` the live stream.
        bank: ``[T, C, H]`` the bank entries (``C >= 1``).
        score_w: ``[H]`` fp32 (``score_weight``).
        eps: The RMSNorm eps of the scores.

    Returns:
        ``[T, H]`` in the promoted dtype of the candidates.
    """
    v = torch.cat((bank, stream.unsqueeze(1)), dim=1)
    v_float = v.float()
    k = v_float * torch.rsqrt(v_float.pow(2).mean(-1, keepdim=True) + eps)
    scores = (k * score_w.float()).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    return torch.matmul(probs, v_float).squeeze(1).to(v.dtype)


def _factored_scores(x: torch.Tensor, score_w: torch.Tensor, eps: float) -> torch.Tensor:
    """``dot(x, w) * rsqrt(sum(x ** 2) / H + eps)`` over the last axis, fp32."""
    x = x.float()
    return (x * score_w.float()).sum(-1) * torch.rsqrt(x.pow(2).sum(-1) / x.shape[-1] + eps)


def attn_res_kernel_algebra(stream: torch.Tensor, bank: torch.Tensor, score_w: torch.Tensor, eps: float,
                            bank_scores: torch.Tensor | None = None) -> torch.Tensor:
    """The algebra of ``attn_res_cte`` in fp32 torch: factored scores, columns ``0 .. C - 1`` of ``bank_scores`` in
    place of the bank scores when given, the softmax over ``C + 1`` candidates, the mixture of the raw candidates."""
    dtype = torch.promote_types(stream.dtype, bank.dtype)
    bank_f, stream_f = bank.to(dtype).float(), stream.to(dtype).float()
    c = bank_f.shape[1]
    bank_part = _factored_scores(bank_f, score_w, eps) if bank_scores is None else bank_scores[:, :c].float()
    probs = torch.cat((bank_part, _factored_scores(stream_f, score_w, eps).unsqueeze(-1)), dim=-1).softmax(-1)
    mixed = (probs[:, :-1].unsqueeze(-1) * bank_f).sum(1) + probs[:, -1:] * stream_f
    return mixed.to(dtype)


def _bf16_trunc(x: torch.Tensor) -> torch.Tensor:
    """fp32 ``x`` with the low 16 mantissa bits cleared: fp32 tensors of bf16 values."""
    return (x.view(torch.int32) & -65536).view(torch.float32)


def _bf16_parts(x: torch.Tensor) -> torch.Tensor:
    """fp32 ``x`` split into 3 parts of bf16 values stacked on dim 0; their sum is ``x`` to 2^-24 of ``x``. The
    split uses bit masks, which the compiler keeps (it folds an fp32 -> bf16 -> fp32 convert pair)."""
    hi = _bf16_trunc(x)
    rest = x - hi
    mid = _bf16_trunc(rest)
    return torch.cat([hi, mid, _bf16_trunc(rest - mid)], dim=0)


def entry_scores(entries: torch.Tensor, score_ws: torch.Tensor, eps: float) -> torch.Tensor:
    """Scores ``[L, T, n]`` fp32 of the entries ``[T, n, H]`` under each score vector of ``score_ws [L, H]``.

    The device runs an fp32 matmul on bf16 operands, so every matmul operand here holds bf16 values only: the score
    vectors (and fp32 entries) are split into 3 bf16 parts stacked on one axis, which keeps one matmul whose products
    are exact.
    """
    v = entries.float()
    inv_rms = torch.rsqrt(v.pow(2).mean(-1) + eps)
    num, (t, n, _) = score_ws.shape[0], v.shape
    w3 = _bf16_parts(score_ws.float())
    if entries.dtype == torch.bfloat16:
        dot = torch.einsum("tnh,lh->ltn", v, w3).reshape(3, num, t, n).sum(0)
    else:
        v3 = _bf16_parts(v.transpose(0, 1)).transpose(0, 1)
        dot = torch.einsum("tnh,lh->ltn", v3, w3).reshape(3, num, t, 3, n).sum((0, 3))
    return dot * inv_rms


def torch_bank_score_table(stream: torch.Tensor, bank: torch.Tensor | None, score_ws: torch.Tensor,
                           eps: float) -> torch.Tensor:
    """The bank-score table ``[L, T, C + 1]`` in torch: the bank entries, then the stage input as the last column
    (the entry the first layer of a stage appends)."""
    parts = [] if bank is None else [entry_scores(bank, score_ws, eps)]
    parts.append(entry_scores(stream.unsqueeze(1), score_ws, eps))
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=2)


def kernel_shape_ok(rows: int, width: int, hidden: int, dtype: torch.dtype, grid: int = 1) -> bool:
    """``attn_res_cte`` accepts a ``[rows, width, hidden]`` bank of ``dtype``."""
    return (rows > 0 and rows % (TOKEN_TILE * grid) == 0 and hidden % HIDDEN_CHUNK == 0 and 1 <= width <= MAX_BANK
            and dtype in KERNEL_DTYPES)


class BankScores:
    """The bank-score table of one stage: row ``i`` (``[T, S]`` fp32) holds the bank scores of AttnRes call ``i``.

    ``take`` gives the rows in call order and checks each call's key ``(layer, site, width)``, so a row never goes to
    another call; ``check_done`` checks that the stage took every row.
    """

    def __init__(self, table: torch.Tensor, calls):
        calls = list(calls)
        if table.dim() != 3 or table.shape[0] != len(calls):
            raise ValueError(f"The bank-score table {tuple(table.shape)} does not hold one [T, S] row for each of "
                             f"the {len(calls)} calls")
        self.table, self.calls, self._next = table, calls, 0

    def take(self, layer: int, site: str, width: int) -> torch.Tensor:
        i, call = self._next, (layer, site, width)
        if i >= len(self.calls) or self.calls[i] != call:
            want = self.calls[i] if i < len(self.calls) else "no row"
            raise RuntimeError(f"AttnRes call {i} is {call}, but the bank-score table has {want}: build the table "
                               f"from the stage plan that runs")
        if width > self.table.shape[2]:
            raise RuntimeError(f"AttnRes call {call} needs {width} score columns; the table has "
                               f"{self.table.shape[2]}")
        self._next = i + 1
        return self.table[i]

    def check_done(self) -> None:
        if self._next != len(self.calls):
            raise RuntimeError(f"The stage took {self._next} of {len(self.calls)} bank-score rows")


class TorchAttnRes:
    """The reference AttnRes in torch (``attn_res_mix``). It builds no bank-score table."""

    scored = False

    def __call__(self, stream, bank, score_w, eps, bank_scores=None):
        if bank_scores is not None:
            raise ValueError("TorchAttnRes takes no bank scores; use TorchScoredAttnRes or KernelAttnRes")
        return attn_res_mix(stream, bank, score_w, eps)

    def takes_table(self, rows: int, width: int, hidden: int, dtype: torch.dtype) -> bool:
        return False

    def bank_score_table(self, stream, bank, score_ws, eps):
        return None


class TorchScoredAttnRes:
    """The kernel's algebra in torch, with the stage's bank scores from a torch table."""

    scored = True

    def __call__(self, stream, bank, score_w, eps, bank_scores=None):
        return attn_res_kernel_algebra(stream, bank, score_w, eps, bank_scores)

    def takes_table(self, rows: int, width: int, hidden: int, dtype: torch.dtype) -> bool:
        return width >= 1

    def bank_score_table(self, stream, bank, score_ws, eps):
        return torch_bank_score_table(stream, bank, score_ws, eps)


def _kernel_params(kernel) -> set:
    try:
        return set(inspect.signature(getattr(kernel, "func", kernel)).parameters)
    except (TypeError, ValueError):
        return set()


class KernelAttnRes:
    """``attn_res_cte`` through ``wrap_nki``, with the stage's bank scores from ``attn_res_bank_scores``.

    A shape the kernel does not take is refused (ValueError): no served shape is one, and the reference math inside a
    compiled stage program would change its numerics without a word. The kernel gets no ``valid_len``: its
    ``valid_len`` path gives wrong rows on Trn3. ``ocp_fp8_sinks`` types the kernel's two scoring sinks as
    ``float8_e4m3fn``, so a stage program that also holds OCP FP8 data keeps the OCP FP8 converter mode.

    Args:
        grid: The SPMD program count of the row axis (1 at LNC1).
        ocp_fp8_sinks: Give ``ocp_fp8_sinks=True`` to each call.
        table_kernel: Build the bank-score table with ``attn_res_bank_scores``; False builds it in torch.
    """

    scored = True

    def __init__(self, grid: int = 1, ocp_fp8_sinks: bool = True, table_kernel: bool = True):
        from sglang_neuron.backend.stack import wrap_nki

        self.grid = grid
        self._mix = kernel_imports.attn_res_cte()
        params = _kernel_params(self._mix)
        wanted = {"bank_scores"} | ({"ocp_fp8_sinks"} if ocp_fp8_sinks else set())
        if params and not wanted <= params:
            raise kernel_imports.KernelUnavailable(f"The vendored attn_res_cte has no {sorted(wanted - params)} "
                                                   f"input")
        self._options = {"ocp_fp8_sinks": True} if ocp_fp8_sinks else {}
        self._mix_call = wrap_nki(self._mix)[grid]
        self._table_call = wrap_nki(kernel_imports.attn_res_bank_scores())[grid] if table_kernel else None

    def takes_table(self, rows: int, width: int, hidden: int, dtype: torch.dtype) -> bool:
        return kernel_shape_ok(rows, width, hidden, dtype, self.grid)

    def __call__(self, stream, bank, score_w, eps, bank_scores=None):
        rows, width, hidden = bank.shape
        dtype = torch.promote_types(stream.dtype, bank.dtype)
        if not kernel_shape_ok(rows, width, hidden, dtype, self.grid):
            raise ValueError(f"attn_res_cte does not take a [{rows}, {width}, {hidden}] {dtype} bank at grid "
                             f"{self.grid} (rows a multiple of {TOKEN_TILE * self.grid}, hidden of {HIDDEN_CHUNK}, "
                             f"1 to {MAX_BANK} entries, {' or '.join(map(str, KERNEL_DTYPES))})")
        extra = {} if bank_scores is None else {"bank_scores": bank_scores.float().contiguous()}
        return self._mix_call(bank=bank.to(dtype).contiguous(), stream=stream.to(dtype).contiguous(),
                              score_w=score_w.float().contiguous(), eps=float(eps), **extra, **self._options)

    def bank_score_table(self, stream, bank, score_ws, eps):
        rows, hidden = stream.shape
        width = 0 if bank is None else bank.shape[1]
        dtype = stream.dtype if bank is None else torch.promote_types(stream.dtype, bank.dtype)
        if self._table_call is None:
            return torch_bank_score_table(stream, bank, score_ws, eps)
        if (rows % (TOKEN_TILE * self.grid) or hidden % TOKEN_TILE or width > MAX_BANK
                or not 1 <= score_ws.shape[0] <= MAX_TABLE_CALLS or dtype not in KERNEL_DTYPES):
            raise ValueError(f"attn_res_bank_scores does not take {score_ws.shape[0]} calls over a [{rows}, {width}, "
                             f"{hidden}] {dtype} bank at grid {self.grid}")
        extra = {} if bank is None else {"bank": bank.to(dtype).contiguous()}
        return self._table_call(stream=stream.to(dtype).contiguous(), score_ws=score_ws.float().contiguous(),
                                eps=float(eps), **extra)


def build_bank_scores(backend, calls, score_ws: torch.Tensor, stream: torch.Tensor, bank: torch.Tensor | None,
                      eps: float) -> BankScores | None:
    """The bank-score table of a stage, or None when the backend takes none or the plan has no static call list.

    Args:
        backend: The AttnRes backend of the stage.
        calls: The stage plan's ``calls`` (``(layer, site, width)`` in call order), or None.
        score_ws: ``[L, H]`` fp32 score vectors of the calls, in call order.
        stream: ``[Tl, H]`` the stage input.
        bank: ``[Tl, C, H]`` the input bank, or None.
        eps: The eps of every AttnRes norm of the stage.
    """
    if not getattr(backend, "scored", False) or not calls:
        return None
    width = max(c for _, _, c in calls)
    dtype = stream.dtype if bank is None else torch.promote_types(stream.dtype, bank.dtype)
    if not backend.takes_table(stream.shape[0], width, stream.shape[1], dtype):
        return None
    table = backend.bank_score_table(stream, bank, score_ws, eps)
    return None if table is None else BankScores(table.contiguous(), calls)


def bank_from_seam(rows: torch.Tensor, entries: int, local_rows: int) -> torch.Tensor | None:
    """The token-major bank ``[Tl, C, H]`` of the first ``C * Tl`` rows of an entry-major seam buffer; None for an
    empty bank."""
    if entries == 0:
        return None
    hidden = rows.shape[-1]
    return rows[: entries * local_rows].reshape(entries, local_rows, hidden).transpose(0, 1).contiguous()


def bank_to_seam(bank: torch.Tensor) -> torch.Tensor:
    """The entry-major seam rows ``[C * Tl, H]`` of a token-major bank ``[Tl, C, H]``."""
    return bank.transpose(0, 1).reshape(-1, bank.shape[-1]).contiguous()
