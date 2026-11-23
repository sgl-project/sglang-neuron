# SPDX-License-Identifier: Apache-2.0
"""Hy4-Flash's checkpoint read as the graphs' parameters, each rank reading only its shard.

The FP8 checkpoint (``quantization_config``: e4m3, 128 x 128 blocks with one ``*_scale_inv`` each, stored bf16)
quantizes the sparse-and-sliding layers' ``q_b_proj``, ``linear_gate`` and ``o_proj`` (not layer 0's) and the routed
experts; every other tensor is bf16 or fp32. A 2-D FP8 weight is read through fp8.DequantSlice,
which dequantizes only the blocks a shard covers. The routed experts are fused 3-D tensors ([E, 2 I, H] gate then up,
[E, H, I] down) and convert per expert to Trn2 FP8 rows (experts.py).

Attention shards by head over the attention TP: rows of ``q_b_proj``, ``linear_gate`` and ``kv_b_proj`` (a head is
192 nope + 64 RoPE Q channels, 192 nope K + 256 V channels of ``kv_b``), columns of ``o_proj``, and the sinks. The
low-rank down projections, the norms, the IndexPool and the iHC and Engram weights are replicated."""

from __future__ import annotations

import torch

from sglang_neuron.models.hy_v4_flash.fp8 import DequantSlice
from sglang_neuron.models.transformer.config import DTYPE
from sglang_neuron.utils.weight_loader import SafetensorsWeightLoader

SCALE = "weight_scale_inv"


def _view(slices: list, quantized: bool):
    """The parameter's single checkpoint tensor, a dequantizing view when the mapping carries its scale too."""
    return DequantSlice(slices[0], slices[1]) if quantized else slices[0]


def keys(name: str, quantized: bool) -> list:
    """The checkpoint keys of a weight: the weight, then its block scale for an FP8 one."""
    return [name, name.removesuffix(".weight") + "." + SCALE] if quantized else [name]


def replicated(transposed: bool = True, dtype=DTYPE, quantized: bool = False) -> SafetensorsWeightLoader:
    """The whole tensor, as [in, out] for a linear weight (``transposed``), in ``dtype``."""

    def transform(slices, rank):
        t = _view(slices, quantized)[...]
        t = t.t() if transposed else t
        return t.to(dtype)

    return SafetensorsWeightLoader(transform=transform)


def head_rows(rows_per_rank: int, quantized: bool) -> SafetensorsWeightLoader:
    """A column-parallel weight [out, in] split by heads: this rank's out rows, as [in, rows]."""
    return SafetensorsWeightLoader(transform=lambda slices, rank: _view(slices, quantized)[
        rank * rows_per_rank:(rank + 1) * rows_per_rank, :].to(DTYPE).t())


def head_cols(cols_per_rank: int, quantized: bool) -> SafetensorsWeightLoader:
    """A row-parallel weight [H, in] split by heads: this rank's in columns, as [cols, H]."""
    return SafetensorsWeightLoader(transform=lambda slices, rank: _view(slices, quantized)[
        :, rank * cols_per_rank:(rank + 1) * cols_per_rank].to(DTYPE).t())


def kv_b(heads: int, nope: int, v: int, part: str) -> SafetensorsWeightLoader:
    """This rank's heads of ``kv_b_proj`` [heads x (nope + v), latent] as W_uk [heads, nope, latent] (``part``
    "k") or W_uv [heads, v, latent] ("v")."""

    def transform(slices, rank):
        per = nope + v
        w = slices[0][rank * heads * per:(rank + 1) * heads * per, :].to(DTYPE).view(heads, per, -1)
        return (w[:, :nope] if part == "k" else w[:, nope:]).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def head_vector(per_rank: int) -> SafetensorsWeightLoader:
    """A per-head vector (the sinks) as this rank's heads, fp32."""
    return SafetensorsWeightLoader(transform=lambda slices, rank: slices[0][
        rank * per_rank:(rank + 1) * per_rank].to(torch.float32))


def vector(dtype=None) -> SafetensorsWeightLoader:
    """A whole vector or small tensor, in ``dtype`` (None: as stored)."""

    def transform(slices, rank):
        t = slices[0][...]
        return t if dtype is None else t.to(dtype)

    return SafetensorsWeightLoader(transform=transform)


def stacked(dtype=None, transposed: bool = False) -> SafetensorsWeightLoader:
    """Several same-shaped tensors stacked on a new first dimension (the Engram's per-stream projections)."""

    def transform(slices, rank):
        parts = [s[...] for s in slices]
        if transposed:
            parts = [p.t() for p in parts]
        out = torch.stack(parts)
        return out if dtype is None else out.to(dtype)

    return SafetensorsWeightLoader(transform=transform)


def mlp_gate_up(rows: int) -> SafetensorsWeightLoader:
    """A SiLU MLP's gate and up ([I, H] each) as [H, 2 x I / tp]: this rank's gate rows, then its up rows."""
    return SafetensorsWeightLoader(transform=lambda slices, rank: torch.cat(
        [s[rank * rows:(rank + 1) * rows] for s in slices], dim=0).to(DTYPE).t())


def mlp_down(cols: int) -> SafetensorsWeightLoader:
    """A SiLU MLP's down [H, I] as [I / tp, H]: this rank's columns."""
    return SafetensorsWeightLoader(transform=lambda slices, rank: slices[0][
        :, rank * cols:(rank + 1) * cols].to(DTYPE).t())


def fp8_rows(x: torch.Tensor) -> tuple:
    """(e4m3fn bytes [N, D] uint8, bf16 scales [N, 1]) of bf16 rows [N, D]: the reference's ``_fp8_round_trip``
    quantization, in bf16 as it computes it (a scale of each row's largest magnitude over e4m3fn's 448). The graph
    reads a row back as its bytes' e4m3fn values times its scale, which is the round trip's value."""
    info = torch.finfo(torch.float8_e4m3fn)
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / info.max
    q = (x / scale).clamp(-info.max, info.max).to(torch.float8_e4m3fn)
    return q.view(torch.uint8), scale


def e4m3fn_values() -> torch.Tensor:
    """[256] bf16 the e4m3fn value of each byte (its two NaN codes read as 0)."""
    values = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn).to(torch.float32)
    return torch.nan_to_num(values, nan=0.0).to(DTYPE)


def engram_rows(rows_per_rank: int, part: str, memo, key) -> SafetensorsWeightLoader:
    """This rank's rows of a layer's flat Engram table [N, D] (``rows_per_rank`` each, the last shard zero-padded) as
    FP8 bytes (``part`` "bytes") and their row scales ("scales"), the reference's FP8 round trip (fp8_rows); both
    parameters share one read (``memo``, a fp8.Memo, under ``key``)."""

    def convert(table, rank):
        n, d = (int(s) for s in table.get_shape())
        lo, hi = min(rank * rows_per_rank, n), min((rank + 1) * rows_per_rank, n)
        q = torch.zeros(rows_per_rank, d, dtype=torch.uint8)
        sc = torch.zeros(rows_per_rank, 1, dtype=DTYPE)
        chunk = 1 << 22
        for i in range(lo, hi, chunk):
            j = min(i + chunk, hi)
            q[i - lo:j - lo], sc[i - lo:j - lo] = fp8_rows(table[i:j].to(DTYPE))
        return {"bytes": q, "scales": sc}

    return SafetensorsWeightLoader(transform=lambda slices, rank: memo.take((key, rank), part,
                                                                            lambda: convert(slices[0], rank)))


def engram_rows_bf16(rows_per_rank: int) -> SafetensorsWeightLoader:
    """This rank's rows of a layer's flat Engram table [N, D] as stored (bf16), the last shard zero-padded."""

    def transform(slices, rank):
        (table,) = slices
        n, d = (int(s) for s in table.get_shape())
        lo, hi = min(rank * rows_per_rank, n), min((rank + 1) * rows_per_rank, n)
        out = torch.zeros(rows_per_rank, d, dtype=DTYPE)
        if hi > lo:
            out[:hi - lo] = table[lo:hi].to(DTYPE)
        return out

    return SafetensorsWeightLoader(transform=transform)
