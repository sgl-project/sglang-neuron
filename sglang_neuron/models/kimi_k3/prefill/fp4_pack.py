# SPDX-License-Identifier: Apache-2.0
"""MXFP4 routed-expert weights: the checkpoint dequant and the packers of the blockwise MoE kernel.

The checkpoint holds each expert matrix as E2M1 codes ``[N, K / 2]`` uint8 (two a byte, the low nibble at the even
column) and E8M0 scales ``[N, K / 32]`` uint8 (``2^(s - 127)`` for each 32 consecutive columns of a row).

The blockwise kernel reads gate/up ``[128, 2, D / 512, I]`` and down ``[128, I / 512, D]`` uint16 words (four E2M1
codes a word, viewed as ``float4_e2m1fn_x4`` in the trace) and ``[16, ...]`` uint8 scale tables of the same tiling.
The kernel contracts nibble ``q`` of the word at partition ``p`` of 512-tile ``t`` with the input value that its own
MX quantization puts at slot ``(p, t, q)``, and one kernel scale at row ``p // 8`` covers 8 partitions by 4 nibbles.

The packers keep the moves value-exact with two choices:

1. The latent columns go to the kernel permuted (``permute_hidden_bwmm``): position ``q (K / 4) + t 128 + s 8 + d``
   holds natural column ``t 512 + s 32 + q 8 + d``. The hidden quantization of the kernel puts position
   ``q (K / 4) + t 128 + p`` at slot ``(p, t, q)``, so slot ``(p = 8 s + d, t, q)`` holds natural column
   ``t 512 + s 32 + q 8 + d``, and each kernel scale group is one checkpoint 32-run.
2. The gate/up output axis stores intermediate row ``t 512 + s 32 + q 8 + d`` at position ``t 512 + q 128 + s 8 + d``.
   The down projection contracts slot ``(p, t, q)`` with intermediate position ``t 512 + q 128 + p`` (the gate/up
   output tiling of the kernel), so the same rule maps its slots to natural intermediate columns, and its scale groups
   align as well. The down output columns stay natural.
"""

from __future__ import annotations

import torch
from torch import Tensor

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_TRANSPOSE_TILE = 64
MX_GROUP = 32
"""Values per E8M0 scale."""


def e2m1_values(nibbles: Tensor) -> Tensor:
    """Return the fp32 value of each E2M1 code (bits 0..2 the magnitude index, bit 3 the sign)."""
    lut = _E2M1.to(nibbles.device)
    magnitude = lut[(nibbles & 0x7).to(torch.int64)]
    return torch.where((nibbles & 0x8) != 0, -magnitude, magnitude)


def fp4_dequant(codes: Tensor, scales: Tensor) -> Tensor:
    """Return the fp32 values ``[N, K]`` of checkpoint MXFP4 ``(codes [N, K / 2], scales [N, K / 32])``.

    Each value is exact in fp32, and exact in bf16 (an E2M1 value times a power of 2).
    """
    n, half = codes.shape
    c = codes.to(torch.int32)
    values = torch.empty(n, 2 * half, dtype=torch.float32, device=codes.device)
    values[:, 0::2] = e2m1_values(c & 0xF)
    values[:, 1::2] = e2m1_values(c >> 4)
    scale = torch.exp2(scales.to(torch.int32).to(torch.float32) - 127.0)
    return (values.view(n, -1, MX_GROUP) * scale[:, :, None]).view(n, 2 * half)


def _transpose_bytes(t: Tensor) -> Tensor:
    """Return the contiguous transpose of a 2-D uint8 tensor, copied in 64 x 64 tiles for the cache."""
    rows, cols = t.shape
    tile = _TRANSPOSE_TILE
    if rows % tile or cols % tile:
        return t.t().contiguous()
    v = t.contiguous().view(rows // tile, tile, cols // tile, tile)
    return v.permute(2, 3, 0, 1).contiguous().view(cols, rows)


def _bwmm_words_into(bt: Tensor, out: Tensor) -> None:
    """Write the kernel words of the transposed codes ``bt`` into ``out``.

    ``bt`` is ``[K / 512, 16, 4, 4, N]`` uint8: byte ``(t, s, q, m, n)`` holds the codes of columns
    ``t 512 + s 32 + q 8 + 2 m`` (low nibble) and ``... + 1`` (high nibble) of row ``n``. ``out`` is a
    ``[16, 4, 2, K / 512, N, 2]`` uint8 view of the words: byte ``(s, m, h, t, n, b)`` is byte ``b`` (low first) of
    the word at partition ``p = 8 s + 2 m + h``, tile ``t``, row ``n``. The low byte holds nibbles ``q = 0, 1``, the
    high byte nibbles ``q = 2, 3``.
    """
    b0, b1, b2, b3 = (bt[:, :, q].permute(1, 2, 0, 3) for q in range(4))
    torch.bitwise_or(b0 & 0xF, b1 << 4, out=out[:, :, 0, :, :, 0])
    torch.bitwise_or(b0 >> 4, b1 & 0xF0, out=out[:, :, 1, :, :, 0])
    torch.bitwise_or(b2 & 0xF, b3 << 4, out=out[:, :, 0, :, :, 1])
    torch.bitwise_or(b2 >> 4, b3 & 0xF0, out=out[:, :, 1, :, :, 1])


def _check_matrix(codes: Tensor, scales: Tensor) -> tuple[int, int]:
    if codes.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise ValueError(f"The MXFP4 codes and scales are uint8; got {codes.dtype} and {scales.dtype}.")
    n, half = codes.shape
    k = 2 * half
    if k % 512 or tuple(scales.shape) != (n, k // MX_GROUP):
        raise ValueError(f"The MXFP4 matrix has codes {tuple(codes.shape)} and scales {tuple(scales.shape)}; the "
                         f"kernel layout needs a contraction width that is a multiple of 512 and scales [N, K / 32].")
    return n, k


def _pack_contraction(codes: Tensor, scales: Tensor) -> tuple[Tensor, Tensor]:
    """Pack one matrix ``(codes [N, K / 2], scales [N, K / 32])`` into ``([128, K / 512, N] u16, [16, K / 512, N])``.

    Nibble ``q`` of the word at ``(p = 8 s + d, t, n)`` is the code of column ``t 512 + s 32 + q 8 + d`` of row
    ``n``; the scale at ``(s, t, n)`` is the checkpoint scale of column group ``16 t + s``.
    """
    n, k = _check_matrix(codes, scales)
    kt = k // 512
    bt = _transpose_bytes(codes).reshape(kt, 16, 4, 4, n)
    out = torch.empty((16, 4, 2, kt, n, 2), dtype=torch.uint8, device=codes.device)
    _bwmm_words_into(bt, out)
    data = out.view(torch.int16).reshape(128, kt, n).view(torch.uint16)
    return data, scales.reshape(n, kt, 16).permute(2, 1, 0).contiguous()


def pack_expert_gate_up_bwmm(w1_codes: Tensor, w1_scales: Tensor, w3_codes: Tensor,
                             w3_scales: Tensor) -> tuple[Tensor, Tensor]:
    """Pack one expert's gate (w1) and up (w3) ``[I, D / 2]`` / ``[I, D / 32]`` for the blockwise kernel.

    Returns:
        ``(weights [128, 2, D / 512, I] uint16, scales [16, 2, D / 512, I] uint8)``: the contraction layout of
        ``_pack_contraction`` with the output rows in the gate/up free-axis order (module docstring, rule 2).
    """
    n, k = _check_matrix(w1_codes, w1_scales)
    _check_matrix(w3_codes, w3_scales)
    if n % 512 or tuple(w3_codes.shape) != tuple(w1_codes.shape):
        raise ValueError(f"The gate and up codes are {tuple(w1_codes.shape)} and {tuple(w3_codes.shape)}; the kernel "
                         "needs equal shapes and an intermediate width that is a multiple of 512.")
    half = k // 2
    kt, ti = k // 512, n // 512
    words = torch.empty((16, 4, 2, 2, kt, n, 2), dtype=torch.uint8, device=w1_codes.device)
    scale = torch.empty((16, 2, kt, n), dtype=torch.uint8, device=w1_codes.device)
    for j, (codes, scales) in enumerate(((w1_codes, w1_scales), (w3_codes, w3_scales))):
        # Free-axis position t 512 + q 128 + s 8 + d takes row t 512 + s 32 + q 8 + d.
        rows = codes.reshape(ti, 16, 4, 8, half).permute(0, 2, 1, 3, 4)
        bt = _transpose_bytes(rows.reshape(n, half)).reshape(kt, 16, 4, 4, n)
        _bwmm_words_into(bt, words[:, :, :, j])
        sc = scales.reshape(ti, 16, 4, 8, kt, 16).permute(5, 4, 0, 2, 1, 3)
        scale[:, j].copy_(sc.reshape(16, kt, n))
    data = words.view(torch.int16).reshape(128, 2, kt, n).view(torch.uint16)
    return data, scale


def pack_expert_down_bwmm(w2_codes: Tensor, w2_scales: Tensor) -> tuple[Tensor, Tensor]:
    """Pack one expert's down (w2) ``[D, I / 2]`` / ``[D, I / 32]``: ``([128, I / 512, D] u16, [16, I / 512, D])``."""
    return _pack_contraction(w2_codes, w2_scales)


def bwmm_free_order(n: int) -> Tensor:
    """Return the natural row held at each gate/up free-axis position (module docstring, rule 2).

    Example:
        >>> bwmm_free_order(512)[:3].tolist(), int(bwmm_free_order(512)[8])
        ([0, 1, 2], 32)
    """
    if n % 512:
        raise ValueError(f"The free axis {n} must tile by 512.")
    return torch.arange(n).reshape(n // 512, 16, 4, 8).permute(0, 2, 1, 3).reshape(n)


def permute_hidden_bwmm(hidden: Tensor) -> Tensor:
    """Give the latent columns in the kernel contraction order (module docstring, rule 1)."""
    rows, k = hidden.shape
    if k % 512:
        raise ValueError(f"The latent width {k} must tile by 512.")
    return hidden.reshape(rows, k // 512, 16, 4, 8).permute(0, 3, 1, 2, 4).reshape(rows, k)


def bwmm_hidden_order(k: int) -> Tensor:
    """Return the natural column of each position of ``permute_hidden_bwmm``.

    ``permute_hidden_bwmm(h) == h[:, bwmm_hidden_order(k)]``, so a linear layer with the weight rows
    ``W[bwmm_hidden_order(k)]`` gives its output in the kernel order with no permutation op.

    Example:
        >>> bwmm_hidden_order(512)[:3].tolist()
        [0, 1, 2]
    """
    return permute_hidden_bwmm(torch.arange(k).reshape(1, k)).reshape(k)


def unpermute_hidden_bwmm(hidden: Tensor) -> Tensor:
    """Undo ``permute_hidden_bwmm``: give the columns in the natural order."""
    order = bwmm_hidden_order(hidden.shape[-1]).to(hidden.device)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.shape[0], device=hidden.device)
    return hidden.index_select(-1, inverse)


def _word_nibbles(data: Tensor) -> Tensor:
    """Return the 4 nibbles of each uint16 word, ``[4, *data.shape]`` int32 (nibble ``q`` = bits ``4 q .. 4 q + 3``)."""
    words = data.view(torch.int16).to(torch.int32) & 0xFFFF
    return torch.stack([(words >> (4 * q)) & 0xF for q in range(4)])


def _unpack_contraction(data: Tensor, scale: Tensor) -> tuple[Tensor, Tensor]:
    """Return ``(codes [N, K / 2], scales [N, K / 32])`` of one ``_pack_contraction`` matrix."""
    _, kt, n = data.shape
    nib = _word_nibbles(data).reshape(4, 16, 8, kt, n)  # [q, s, d, t, n]
    natural = nib.permute(4, 3, 1, 0, 2).reshape(n, 512 * kt)  # column t 512 + s 32 + q 8 + d
    codes = (natural[:, 0::2] | (natural[:, 1::2] << 4)).to(torch.uint8)
    return codes, scale.permute(2, 1, 0).reshape(n, 16 * kt).contiguous()


def unpack_expert_gate_up_bwmm(data: Tensor, scale: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return the checkpoint ``(w1_codes, w1_scales, w3_codes, w3_scales)`` of ``pack_expert_gate_up_bwmm``."""
    n = data.shape[-1]
    order = bwmm_free_order(n)
    out = []
    for j in range(2):
        codes, scales = _unpack_contraction(data[:, j], scale[:, j])
        natural_codes, natural_scales = torch.empty_like(codes), torch.empty_like(scales)
        natural_codes[order], natural_scales[order] = codes, scales
        out += [natural_codes, natural_scales]
    return tuple(out)


def unpack_expert_down_bwmm(data: Tensor, scale: Tensor) -> tuple[Tensor, Tensor]:
    """Return the checkpoint ``(w2_codes, w2_scales)`` of ``pack_expert_down_bwmm``."""
    return _unpack_contraction(data, scale)


def kernel_slot_values(data: Tensor, scale: Tensor) -> Tensor:
    """Return the weight value at each kernel slot of one packed matrix, ``[4, 128, K / 512, N]`` fp32.

    Entry ``(q, p, t, n)`` is nibble ``q`` of the word at partition ``p``, tile ``t``, column ``n``, times
    ``2^(scale[p // 8, t, n] - 127)``. This is the value that the kernel contracts at slot ``(p, t, q)``.
    """
    values = e2m1_values(_word_nibbles(data))
    factor = torch.exp2(scale.to(torch.int32).to(torch.float32) - 127.0).repeat_interleave(8, dim=0)
    return values * factor[None]
