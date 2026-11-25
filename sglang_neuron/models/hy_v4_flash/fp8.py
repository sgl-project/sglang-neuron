# SPDX-License-Identifier: Apache-2.0
"""Hy4-Flash's FP8 weights at load: the checkpoint's 128 x 128 blockwise ``float8_e4m3fn`` (one ``*_scale_inv`` per
block, stored bf16) dequantized for the bf16 graphs, and the routed experts requantized to Trn2's FP8.

Trn2's e4m3 tops out at 240, not e4m3fn's 448, and the decode MoE kernel takes one scale per output row, not per block.
``requant_rows`` gives each output row a power-of-two scale, the smallest that brings its largest value to 240 or
below: a power of two shifts exponents only, so the requantization rounds no more than the block scale's own fit. The
tensors stay ``float8_e4m3fn`` holding at most 240, which the graphs compile as Trn2's e4m3 (launch.py)."""

from __future__ import annotations

import threading

import torch

from sglang_neuron.models.transformer.config import DTYPE

BLOCK = 128
FP8 = torch.float8_e4m3fn
FP8_MAX = 240.0


def _ranges(key, shape) -> tuple:
    """(r0, r1, c0, c1) of a 2-D basic index (unit-step slices, or Ellipsis) over ``shape``."""
    rows, cols = shape
    if key is Ellipsis:
        key = (slice(None), slice(None))
    if not isinstance(key, tuple):
        key = (key,)
    if len(key) == 1:
        key = (key[0], slice(None))
    if len(key) != 2 or not all(isinstance(k, slice) for k in key):
        raise IndexError(f"a dequantizing slice takes a 2-D slice index, not {key!r}")
    (r0, r1, rs), (c0, c1, cs) = key[0].indices(rows), key[1].indices(cols)
    if rs != 1 or cs != 1:
        raise IndexError(f"a dequantizing slice takes unit steps, not {key!r}")
    return r0, max(r0, r1), c0, max(c0, c1)


def dequant_blocks(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """fp32 of a blockwise FP8 tensor [..., R, C] with its block scales [..., ceil(R / 128), ceil(C / 128)]."""
    rows, cols = w.shape[-2:]
    scale = s.to(torch.float32).repeat_interleave(BLOCK, -2)[..., :rows, :].repeat_interleave(BLOCK, -1)[..., :cols]
    return w.to(torch.float32) * scale


class DequantSlice:
    """A checkpoint slice of a 2-D blockwise FP8 weight with its scale that reads as bf16: ``get_shape`` and unit-step
    2-D slicing, reading only the blocks a window meets."""

    def __init__(self, weight, scale):
        self._weight, self._scale = weight, scale
        self._shape = tuple(int(d) for d in weight.get_shape())

    def get_shape(self):
        return self._shape

    @property
    def shape(self):
        return self._shape

    def __getitem__(self, key):
        r0, r1, c0, c1 = _ranges(key, self._shape)
        if r1 <= r0 or c1 <= c0:
            return torch.zeros((r1 - r0, c1 - c0), dtype=DTYPE)
        b0, b1 = r0 // BLOCK, (r1 - 1) // BLOCK + 1
        k0, k1 = c0 // BLOCK, (c1 - 1) // BLOCK + 1
        w = self._weight[b0 * BLOCK:min(b1 * BLOCK, self._shape[0]), k0 * BLOCK:min(k1 * BLOCK, self._shape[1])]
        s = self._scale[b0:b1, k0:k1]
        out = dequant_blocks(w, s)
        return out[r0 - b0 * BLOCK:r1 - b0 * BLOCK, c0 - k0 * BLOCK:c1 - k0 * BLOCK].to(DTYPE)


def requant_rows(w: torch.Tensor) -> tuple:
    """fp32 [out, in] -> (FP8 [out, in], fp32 [out]) with a power-of-two scale per output row (module docstring)."""
    amax = w.abs().amax(dim=1).clamp(min=1e-30)
    scale = torch.exp2(torch.ceil(torch.log2(amax / FP8_MAX)))
    q = (w / scale[:, None]).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    return q, scale


def fp8_t(q: torch.Tensor) -> torch.Tensor:
    """An FP8 matrix transposed through its bytes (the CPU has no FP8 transpose kernels)."""
    return q.view(torch.uint8).t().contiguous().view(FP8)


class Memo:
    """One conversion per key shared by several parameters whose loaders may run on the checkpoint loader's threads
    at once; each part is handed out once, so a reload converts again."""

    def __init__(self):
        self._lock = threading.Lock()
        self._slots = {}

    def take(self, key, part: str, compute):
        with self._lock:
            slot = self._slots.setdefault(key, {"lock": threading.Lock(), "parts": None})
        with slot["lock"]:
            if slot["parts"] is None:
                slot["parts"] = compute()
            value = slot["parts"].pop(part)
            if not slot["parts"]:
                with self._lock:
                    self._slots.pop(key, None)
        return value
