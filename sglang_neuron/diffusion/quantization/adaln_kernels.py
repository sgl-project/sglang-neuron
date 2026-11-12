# SPDX-License-Identifier: Apache-2.0
"""Fused adaptive-LayerNorm entry point for the FLUX DiT.

Replaces the separate ``LayerNorm(x) * (1 + scale) + shift`` (and plain affine ``LayerNorm``) before
every attention / FFN / output projection with a single vendored NKI kernel
(:func:`adaln_quant_kernel`, run without quantization).

Dispatch mirrors :func:`o_proj`: :func:`_can_use_adaln_kernel` bundles every runnability
condition, and the caller falls back to the unfused path when it returns False.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sglang_neuron.diffusion.kernels.kernel_choice import device_reason, record_kernel_choice

# The AdaLN kernel launches at NEURON_LOGICAL_NC_CONFIG (unset: 2). It shards over grid dims
# (0, 1), so it also runs on one core at LNC 1 (the LingBot DiT's block norm sites).
# Maximum hidden dimension the kernel supports (matches AdaLNQuantConstants.MAX_H).
_MAX_HIDDEN = 16384
_SUPPORTED_NORMS = ("layer_norm", "rms_norm", "none")


def _can_use_adaln_kernel(hidden: torch.Tensor, norm_type: str) -> bool:
    """Validate the inputs and report whether the fused AdaLN NKI kernel can run.

    Raises on an unsupported ``norm_type`` (an error on both the kernel and fallback paths).
    Otherwise returns False (take the unfused fallback) when NKI kernels can't run here
    (CPU tensors, fake-tensor tracing, kernels disabled — via ``can_run_kernel``), ``hidden`` has
    fewer than 2 dims, or the hidden dimension exceeds ``_MAX_HIDDEN``.
    """
    if norm_type not in _SUPPORTED_NORMS:
        raise ValueError(f"unsupported norm_type {norm_type!r}")

    from sglang_neuron.diffusion.vn.utils.neuron_utils import can_run_kernel

    if not can_run_kernel(hidden):
        return record_kernel_choice("adaln", False, device_reason(hidden))
    if hidden.dim() < 2 or hidden.shape[-1] > _MAX_HIDDEN:
        return record_kernel_choice("adaln", False, "shape outside the kernel limits")
    return record_kernel_choice("adaln", True, "eligible")


def _eager_adaln(
    hidden, mul, add, *, eps, norm_type, residual=None, gate=None, round_gated_product=False
):
    """Unfused fallback: gated residual add, then fp32 norm and modulation.

    When ``residual`` is given, computes ``x = residual + gate * hidden`` (``gate`` may be ``None``
    for a plain add) in fp32 first and returns ``(residual_out, modulated)``; otherwise normalizes
    ``hidden`` directly and returns ``modulated``.
    """
    x = hidden.float()
    residual_out = None
    if residual is not None:
        if gate is not None:
            x = x * gate.float()
            if round_gated_product:
                x = x.type_as(hidden).float()
        x = x + residual.float()
        # Round to bf16 before normalizing (matches the unfused model's ``.type_as`` and the kernel).
        residual_out = x.type_as(residual)
        x = residual_out.float()
    if norm_type == "layer_norm":
        norm = F.layer_norm(x, (x.shape[-1],), None, None, eps)
    elif norm_type == "rms_norm":
        norm = x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    else:  # "none"
        norm = x
    if mul is not None:
        norm = norm * mul.float()
    if add is not None:
        norm = norm + add.float()
    modulated = norm.type_as(hidden)
    return (residual_out, modulated) if residual is not None else modulated


def adaln_modulate(
    hidden: torch.Tensor,
    mul: torch.Tensor | None = None,
    add: torch.Tensor | None = None,
    *,
    eps: float,
    norm_type: str = "layer_norm",
    ocp: bool = True,
    residual: torch.Tensor | None = None,
    gate: torch.Tensor | None = None,
    fold_rows: bool = False,
    round_gated_product: bool = False,
    fold_mul: bool = True,
    row_indices: torch.Tensor | None = None,
):
    """Fused ``Norm(residual + gate * hidden) * mul + add`` with an unfused fallback.

    ``fold_rows`` selects the kernel's bit-identical few-row LayerNorm layout and
    ``round_gated_product`` rounds ``gate * hidden`` to the hidden dtype before the residual add
    (``AdaLNQuantKernelArgs``).

    ``hidden`` is ``[B, S, H]`` or ``[T, H]``; ``mul`` / ``add`` are ``[B, 1, H]`` (per-batch AdaLN
    ``1 + scale`` / ``shift``), ``[H]`` (shared affine gamma / beta), ``[B, S, H]`` / ``[S, H]``
    (one row per token), or ``None``.

    ``row_indices`` (``[S]``, ``[S, 1]`` or ``[B, S, 1]``) turns a multi-row ``mul`` / ``add`` /
    ``gate`` into a TABLE of distinct modulation rows that the kernel gathers per token. A model
    whose modulation takes only a few distinct values per step (one row per timestep and modality,
    say) then passes the table and the indices instead of an ``index_select`` expansion of it — the
    expansion is the DRAM cost the gather removes, not the gather itself.

    When ``residual`` is ``None`` (the plain case), normalizes ``hidden`` directly and returns the
    modulated tensor, with the same leading dims as ``hidden``.

    When ``residual`` is given (the fused gated-residual case), ``hidden`` is the sub-layer output
    and the kernel first computes ``residual + gate * hidden`` in the compute dtype (``gate`` — same
    per-batch/shared layout as ``mul`` — may be ``None`` for a plain add). It returns
    ``(residual_out, modulated)``: ``residual_out`` is that bf16 sum (the block's carried residual)
    and ``modulated`` is the normalized activation.
    """
    have_residual = residual is not None
    if not _can_use_adaln_kernel(hidden, norm_type):
        if row_indices is not None:
            # The fallback has no gather: expand the tables it was given. Same test as the
            # kernel's (_is_table), so a shared [1, H] vector and a per-row [B, S, H] one are
            # left alone.
            flat = row_indices.reshape(-1)
            mul, add, gate = (
                vec.index_select(0, flat)
                if vec is not None and vec.dim() == 2 and vec.shape[0] > 1
                else vec
                for vec in (mul, add, gate)
            )
        return _eager_adaln(
            hidden,
            mul,
            add,
            eps=eps,
            norm_type=norm_type,
            residual=residual,
            gate=gate,
            round_gated_product=round_gated_product,
        )

    from nkilib.core.utils.common_types import (
        DtypeMode,
        NormType,
        QuantizationType,
    )
    from sglang_neuron.diffusion.kernels.flux.adaln_quant import (
        AdaLNQuantKernelArgs,
        adaln_quant_kernel,
    )

    kargs = AdaLNQuantKernelArgs(
        norm_type={
            "layer_norm": NormType.LAYER_NORM,
            "rms_norm": NormType.RMS_NORM,
            "none": NormType.NO_NORM,
        }[norm_type],
        quantization_type=QuantizationType.NONE,
        eps=eps,
        fold_rows=fold_rows,
        round_gated_product=round_gated_product,
        fold_mul=fold_mul,
    )
    from sglang_neuron.diffusion.neuron_api import wrap_nki_kernel

    launch = wrap_nki_kernel(adaln_quant_kernel, supported_lnc=(1, 2))
    dtype_mode = DtypeMode.OCP if ocp else DtypeMode.NON_OCP

    hidden_3d = hidden.unsqueeze(0) if hidden.dim() == 2 else hidden
    hidden_dim = hidden_3d.shape[-1]
    residual_3d = None
    if have_residual:
        residual_3d = residual.unsqueeze(0) if residual.dim() == 2 else residual

    # Modulate in the hidden (compute) dtype: the PE broadcast-matmul consumes mul/add in the
    # compute dtype, and the normalized value is rounded to the output dtype at the store. The
    # gate stays fp32 (see _row) so the gated add matches the unfused model's add (bf16
    # sub-layer output * fp32 gate, accumulated in fp32); a per-row gate keeps its dtype, since
    # the kernel accumulates in fp32 either way.
    mod_dtype = hidden_3d.dtype
    gate_dtype = (
        gate.dtype if gate is not None and gate.dim() > 1 and gate.shape[-2] > 1 else torch.float32
    )

    def _row(vec, b, dtype=None):
        # mul/add/gate are [B, 1, H] (per-batch), [H]/[1, H] (shared), [B, S, H]/[S, H] (per row),
        # or a [T, H] table to gather (row_indices); return the [1, H] slice, the [S, H] rows or
        # the table unchanged.
        if vec is None:
            return None
        return (vec[b] if vec.dim() == 3 else vec).reshape(-1, hidden_dim).to(dtype or mod_dtype)

    def _indices(b):
        # One int32 row index per token of batch b, as the kernel's [S, 1] index column (the
        # indirect DMA takes int32 offsets; this is one column, not one row per token).
        if row_indices is None:
            return None
        rows = row_indices[b] if row_indices.dim() == 3 else row_indices
        return rows.reshape(-1, 1).to(torch.int32).contiguous()

    outs = []
    residual_outs = []
    for b in range(hidden_3d.shape[0]):
        if have_residual:
            residual_out_b, out_b = launch(
                hidden=hidden_3d[b].contiguous(),
                kargs=kargs,
                mul=_row(mul, b),
                add=_row(add, b),
                dtype_mode=dtype_mode,
                residual=residual_3d[b].contiguous(),
                gate=_row(gate, b, dtype=gate_dtype),
                row_indices=_indices(b),
            )
            residual_outs.append(residual_out_b.unsqueeze(0))
            outs.append(out_b.unsqueeze(0))
        else:
            outs.append(
                launch(
                    hidden=hidden_3d[b].contiguous(),
                    kargs=kargs,
                    mul=_row(mul, b),
                    add=_row(add, b),
                    dtype_mode=dtype_mode,
                    row_indices=_indices(b),
                ).unsqueeze(0)
            )

    out = torch.cat(outs, dim=0)
    out = out.squeeze(0) if hidden.dim() == 2 else out
    if not have_residual:
        return out
    residual_out = torch.cat(residual_outs, dim=0)
    residual_out = residual_out.squeeze(0) if hidden.dim() == 2 else residual_out
    return residual_out, out
