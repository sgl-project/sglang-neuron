# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Host MXFP8 / MXFP4 weights of the Kimi-K3 decode projections: the quantizers, the group orders and the packing.

A linear ``weight[out, in]`` quantizes per output row in groups of 32 input rows. Each group gets one E8M0 scale
by the ``nisa.quantize_mx`` rule, ``s = max(FP32 exponent field of the group) - 7`` (0 for an all-zero group), and
each value becomes ``e4m3(x / 2^(s - 127))``, rounded to nearest even. The group maximum lands in [128, 256), so no
value clips. The orders give the 32 input rows of each group (scale index ``b``):

* ``consecutive``: rows ``32 b .. 32 b + 31``.
* ``hT_quad``: ``b = 16 g + s`` holds rows ``{128 (4 g + q) + 8 s + r : q < 4, r < 8}``, for an hT activation input.
* ``shard_quad``: with ``C = in / 128``, ``b = 16 u + s`` holds rows ``{C p + 4 u + q : p = 8 s + r, r < 8, q < 4}``,
  for an input whose partition ``p`` holds rows ``C p .. C p + C - 1``.

The quantization is per (output row, group), so a slice of the output rows has the bits of the whole weight.
"""

import ml_dtypes
import numpy as np
import torch

from .tkg_mx_quant import MX_GROUP_ROWS, MX_LANES, MX_SCALE_ROWS, MX_SCALE_SLOTS, P_MAX, mx_hT_shapes

MX_ORDERS = ("consecutive", "hT_quad", "shard_quad")
MX_BLOCK = 32
_E4M3_MAX_EXP = 7


def mx_group_rows(rows: int, order: str) -> np.ndarray:
    """Return ``[rows // 32, 32]``: the input rows of each MX group, in scale-index order."""
    if order not in MX_ORDERS:
        raise ValueError(f"MX order {order!r} is not one of {MX_ORDERS}")
    index = np.arange(rows)
    if order == "consecutive":
        if rows % MX_BLOCK:
            raise ValueError(f"the consecutive order needs in % {MX_BLOCK} == 0, got {rows}")
        return index.reshape(-1, MX_BLOCK)
    if rows % MX_GROUP_ROWS:
        raise ValueError(f"the {order} order needs in % {MX_GROUP_ROWS} == 0, got {rows}")
    if order == "hT_quad":
        # [g, q, s, r] -> [g, s, q, r]
        return index.reshape(-1, MX_LANES, P_MAX // 8, 8).transpose(0, 2, 1, 3).reshape(-1, MX_BLOCK)
    # shard_quad: [s, r, u, q] -> [u, s, r, q]
    return index.reshape(P_MAX // 8, 8, -1, MX_LANES).transpose(2, 0, 1, 3).reshape(-1, MX_BLOCK)


def _as_float32(weight) -> np.ndarray:
    if isinstance(weight, torch.Tensor):
        weight = weight.detach().float().numpy()
    return np.asarray(weight, dtype=np.float32)


def mx_quantize(weight, order: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(data, scale)`` of ``weight[out, in]``: e4m3 bits uint8 ``[out, in]`` and E8M0 uint8 ``[out, in / 32]``."""
    values = _as_float32(weight)
    if not np.isfinite(values).all():
        raise ValueError("an MXFP8 weight must be finite")
    groups = mx_group_rows(values.shape[1], order)
    grouped = values[:, groups]
    exponent = ((grouped.view(np.uint32) >> 23) & 0xFF).max(axis=2).astype(np.int32)
    scale = np.clip(exponent - _E4M3_MAX_EXP, 0, 255).astype(np.uint8)
    step = np.ldexp(np.float64(1.0), scale.astype(np.int32) - 127)[:, :, None]
    quantized = (grouped / step).astype(np.float32).astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    data = np.empty(values.shape, dtype=np.uint8)
    data[:, groups] = quantized
    return data, scale


def mx_dequantize(data: np.ndarray, scale: np.ndarray, order: str) -> np.ndarray:
    """Return the float64 values ``[out, in]`` of ``(data, scale)`` from ``mx_quantize``."""
    groups = mx_group_rows(data.shape[1], order)
    values = np.asarray(data, dtype=np.uint8).view(ml_dtypes.float8_e4m3fn).astype(np.float64)
    step = np.ldexp(np.float64(1.0), np.asarray(scale, dtype=np.int32) - 127)
    out = np.empty(values.shape, dtype=np.float64)
    out[:, groups] = values[:, groups] * step[:, :, None]
    return out


_E2M1_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def mx4_quantize(weight, order: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the MXFP4 ``(codes, scale)`` of ``weight[out, in]``: e2m1 codes uint8 ``[out, in]`` and E8M0 ``[out, in / 32]``.

    The scale of a group is ``2^k`` with the smallest ``k`` such that ``6 * 2^k >= amax`` (no value clips). Each value
    rounds to the nearest e2m1 value, ties to the even code. Code bits: sign 3, exponent 2..1, mantissa 0.
    """
    values = _as_float32(weight).astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("an MXFP4 weight must be finite")
    groups = mx_group_rows(values.shape[1], order)
    grouped = values[:, groups]
    amax = np.abs(grouped).max(axis=2)
    mantissa, exponent = np.frexp(amax / 6.0)
    k = np.where(mantissa == 0.5, exponent - 1, exponent)
    k = np.where(amax > 0, k, -127)
    k = np.clip(k, -127, 127)
    scaled = np.abs(grouped) / np.ldexp(1.0, k)[:, :, None]
    distance = np.abs(scaled[..., None] - _E2M1_VALUES)
    best = distance.min(axis=-1, keepdims=True)
    # Of two equally near values, take the even code.
    candidates = np.where(distance == best, np.arange(8), 99)
    even = np.where(candidates % 2 == 0, candidates, 99).min(axis=-1)
    code = np.where(even < 99, even, candidates.min(axis=-1)).astype(np.uint8)
    code = np.where(grouped < 0, code | 8, code).astype(np.uint8)
    data = np.empty(values.shape, dtype=np.uint8)
    data[:, groups] = code
    return data, (k + 127).astype(np.uint8)


def mx4_dequantize(codes: np.ndarray, scale: np.ndarray, order: str) -> np.ndarray:
    """Return the float64 values ``[out, in]`` of ``(codes, scale)`` from ``mx4_quantize``."""
    groups = mx_group_rows(codes.shape[1], order)
    codes = np.asarray(codes, dtype=np.uint8)
    values = _E2M1_VALUES[codes & 7] * np.where(codes & 8, -1.0, 1.0)
    step = np.ldexp(np.float64(1.0), np.asarray(scale, dtype=np.int32) - 127)
    out = np.empty(values.shape, dtype=np.float64)
    out[:, groups] = values[:, groups] * step[:, :, None]
    return out


def pack_mx4_hT(codes_in_out: np.ndarray, scale_groups_out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pack an hT_quad MXFP4 kernel weight ``[in, out]`` (e2m1 codes) into ``(data [G, 128, out] uint16, scales)``.

    Element ``(g, p, n)`` of the data holds the codes of rows ``128 (4 g + q) + p`` of column ``n`` in its nibble ``q``
    (bits ``4 q .. 4 q + 3``), the ``float4_e2m1fn_x4`` packing. The scales have the layout of ``pack_mx_hT``.
    """
    rows, cols = codes_in_out.shape
    _, scales = pack_mx_hT(np.zeros((rows, cols), dtype=np.uint8), scale_groups_out)
    groups = rows // MX_GROUP_ROWS
    lanes = np.asarray(codes_in_out, dtype=np.uint16).reshape(groups, MX_LANES, P_MAX, cols)
    data = lanes[:, 0] | (lanes[:, 1] << 4) | (lanes[:, 2] << 8) | (lanes[:, 3] << 12)
    return np.ascontiguousarray(data.astype(np.uint16)), scales


def pack_mx_hT(data_in_out: np.ndarray, scale_groups_out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pack an hT_quad MXFP8 kernel weight ``[in, out]`` (scales ``[in / 32, out]``) into the ``tkg_mx_quant`` tensors.

    Returns:
        ``(data [G, 128, 4 out] uint8, scales [128, S, out] uint8)``.
    """
    rows, cols = data_in_out.shape
    data_shape, scale_shape = mx_hT_shapes(rows, cols)
    groups = data_shape[0]
    # row 128 (4 g + q) + p -> [g, p, n, q]
    data = np.asarray(data_in_out, dtype=np.uint8).reshape(groups, MX_LANES, P_MAX, cols).transpose(0, 2, 3, 1)
    scale_rows = P_MAX // 8
    by_group = np.asarray(scale_groups_out, dtype=np.uint8).reshape(groups, scale_rows, cols)
    scales = np.zeros(scale_shape, dtype=np.uint8)
    g = np.arange(groups)[:, None]
    s = np.arange(scale_rows)[None, :]
    partition = 32 * (s // MX_SCALE_ROWS) + MX_SCALE_ROWS * (g % MX_SCALE_SLOTS) + s % MX_SCALE_ROWS
    scales[partition, np.broadcast_to(g // MX_SCALE_SLOTS, partition.shape)] = by_group
    return np.ascontiguousarray(data.reshape(data_shape)), scales


def mx_rows(weights: dict, name: str, rows, order: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(data, scale)`` of the output rows ``rows`` of the linear ``name``.

    The loader gives ``weights[name]`` (uint8 e4m3 bits) and ``weights[name + "_mx_scale"]``. A BF16 weight with no
    scale quantizes here (tests, and the slow load path).
    """
    key = name + "_mx_scale"
    if key in weights:
        data = np.asarray(weights[name].view(torch.uint8) if isinstance(weights[name], torch.Tensor) else weights[name])
        scale = np.asarray(weights[key].view(torch.uint8) if isinstance(weights[key], torch.Tensor) else weights[key])
        if data.dtype != np.uint8 or scale.dtype != np.uint8:
            raise TypeError(f"{name}: MXFP8 data and scales must be uint8, got {data.dtype} and {scale.dtype}")
        return data[rows], scale[rows]
    return mx_quantize(weights[name][rows], order)


MX_QUADRANT_ROWS = 32
"""Contraction partitions of one MX quadrant row tile."""
MX_COMPACT_ROWS = (P_MAX // MX_QUADRANT_ROWS) * MX_SCALE_ROWS
"""Scale rows of one MX group (4 in each quadrant): the compact scales ``[16, N]`` of a one-group weight."""


def mx_compact_scales(scales: np.ndarray) -> np.ndarray:
    """Return the compact scales ``[16, N]`` of the one-group scales ``[128, 1, N]`` of ``pack_mx_hT``.

    Row ``4 c + i`` is partition ``32 c + i`` (``tkg_mx_quant.load_mx_compact_scales`` loads it back there); the other
    partitions of a one-group weight hold no scale.
    """
    scales = np.asarray(scales, dtype=np.uint8)
    if scales.ndim != 3 or scales.shape[:2] != (P_MAX, 1):
        raise ValueError(f"compact MX scales need one-group [128, 1, N] scales, got {scales.shape}")
    rows = [MX_QUADRANT_ROWS * (s // MX_SCALE_ROWS) + s % MX_SCALE_ROWS for s in range(MX_COMPACT_ROWS)]
    return np.ascontiguousarray(scales[rows, 0, :])


def pack_mx_hT_compact(weight_in_out, order: str = "hT_quad") -> tuple[np.ndarray, np.ndarray]:
    """Return the one-group MXFP8 weight ``(data [1, 128, 4 out] uint8, compact scales [16, out] uint8)`` of
    ``weight_in_out`` ``[in, out]`` with ``in <= 512``.

    The rows past ``in`` are zero (the padded-lane form: 3 heads of 128 rows in lanes 0 .. 2, a zero lane 3), so
    their lanes add nothing.
    """
    values = _as_float32(weight_in_out)
    rows, cols = values.shape
    if rows > MX_GROUP_ROWS:
        raise ValueError(f"a one-group MX weight needs in <= {MX_GROUP_ROWS}, got {rows}")
    padded = np.zeros((MX_GROUP_ROWS, cols), dtype=np.float32)
    padded[:rows] = values
    data, scale = mx_quantize(padded.T, order)
    packed, scales = pack_mx_hT(data.T, scale.T)
    return packed, mx_compact_scales(scales)


def _quadrant_order() -> np.ndarray:
    """Return the 128 head rows in quadrant scale-group order: index ``32 s + 8 l + r`` holds row ``32 l + 8 s + r``."""
    return np.arange(P_MAX).reshape(MX_LANES, MX_SCALE_ROWS, 8).transpose(1, 0, 2).reshape(-1)


def mx_quadrant_quantize(weight) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(data [out, 128], scale [out, 4])`` of ``weight[out, 128]`` in the quadrant order of one head.

    The 128 input rows of a head fill one 32-partition quadrant: partition ``p``, lane ``l`` holds row ``32 l + p``,
    so scale ``s`` of a column holds the rows ``{32 l + 8 s + r : l < 4, r < 8}`` (the hT_quad rule on 32-row tiles).
    Each group gets the ``mx_quantize`` rule.
    """
    values = _as_float32(weight)
    if values.ndim != 2 or values.shape[1] != P_MAX:
        raise ValueError(f"a quadrant MX weight needs [out, {P_MAX}], got {values.shape}")
    order = _quadrant_order()
    data_c, scale = mx_quantize(values[:, order], "consecutive")
    data = np.empty_like(data_c)
    data[:, order] = data_c
    return data, scale


def mx_quadrant_dequantize(data: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Return the float64 values ``[out, 128]`` of ``(data, scale)`` from ``mx_quadrant_quantize``."""
    order = _quadrant_order()
    values = mx_dequantize(np.asarray(data, dtype=np.uint8)[:, order], scale, "consecutive")
    out = np.empty_like(values)
    out[:, order] = values
    return out


def pack_mx_quadrants(heads_out_in) -> tuple[np.ndarray, np.ndarray]:
    """Return the quadrant MXFP8 weight ``(data [128, 4 N] uint8, compact scales [16, N] uint8)`` of up to 4 heads.

    ``heads_out_in`` is ``[H, N, 128]`` (head ``h``: ``weight[out, in]`` with a 128-row contraction). Byte
    ``(32 h + p, 4 n + l)`` holds row ``32 l + p`` of column ``n`` of head ``h`` (``mx_quadrant_quantize``), so head
    ``h`` is the nc_matmul_mx row tile at PE rows ``32 h``. Compact row ``4 h + s`` holds scale ``s`` of head ``h``;
    the quadrants past the heads are zero.
    """
    heads = np.asarray(heads_out_in)
    count, cols, rows = heads.shape
    if count * MX_QUADRANT_ROWS > P_MAX or rows != P_MAX:
        raise ValueError(f"quadrant MX weights need <= 4 heads of [N, {P_MAX}], got {heads.shape}")
    data = np.zeros((P_MAX, cols, MX_LANES), dtype=np.uint8)
    compact = np.zeros((MX_COMPACT_ROWS, cols), dtype=np.uint8)
    for head in range(count):
        head_data, head_scale = mx_quadrant_quantize(heads[head])
        # [n, 32 l + p] -> [p, n, l]
        lanes = head_data.reshape(cols, MX_LANES, MX_QUADRANT_ROWS).transpose(2, 0, 1)
        data[head * MX_QUADRANT_ROWS : (head + 1) * MX_QUADRANT_ROWS] = lanes
        compact[head * MX_SCALE_ROWS : (head + 1) * MX_SCALE_ROWS] = head_scale.T
    return np.ascontiguousarray(data.reshape(P_MAX, cols * MX_LANES)), compact
