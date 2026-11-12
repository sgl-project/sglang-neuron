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

"""Host MXFP8 weights of the Kimi-K3 KDA front projections: the quantizer and the kernel packing.

The front projects a bf16 hidden ``x [T, HID]`` by the linears ``W_q``, ``W_k``, ``W_v`` (the conv tiles) and
``W_gb = cat(W_g, W_b)`` (the g|b tile). The MXFP8 form quantizes the hidden once per 512-row group and keeps
each weight in the same group order, so one ``nc_matmul_mx`` contracts 512 hidden rows (128 partitions x 4 x4
lanes) in place of 4 bf16 matmuls.

A linear ``W[out, in]`` quantizes per output row in groups of 32 input rows (the ``nisa.quantize_mx`` rule:
``s = max(FP32 exponent field of the group) - 7``, then ``e4m3(x / 2^(s - 127))`` rounded to nearest even). The
``hT_quad`` group order matches an hT activation whose partition ``p`` holds hidden row ``512 g + 128 q + p`` of x4
group ``g``: scale index ``b = 16 g + s`` holds the 32 rows ``{128 (4 g + q) + 8 s + r : q < 4, r < 8}``.

The kernel packing (``pack_*``) puts each group's data and scales in their own slot with the scale rows at the
quadrant partitions ``{32 (s // 4) + s % 4}`` (first partition 0), the layout ``nisa.quantize_mx`` writes for the
activation and ``nisa.nc_matmul_mx`` reads for both operands.
"""

import ml_dtypes
import numpy as np

P = 128
MX_LANES = 4
MX_GROUP_ROWS = P * MX_LANES  # 512 hidden rows per MX group (one nc_matmul_mx)
MX_BLOCK = 32  # input rows per scale
MX_SCALE_ROWS = P // 8  # 16 scale rows per group (one per 8 partitions)
_E4M3_MAX_EXP = 7


def _as_float32(weight) -> np.ndarray:
    try:
        import torch

        if isinstance(weight, torch.Tensor):
            weight = weight.detach().float().numpy()
    except ImportError:
        pass
    return np.asarray(weight, dtype=np.float32)


def mx_group_rows(rows: int) -> np.ndarray:
    """Return ``[rows // 32, 32]``: the input rows of each hT_quad MX group, in scale-index order."""
    if rows % MX_GROUP_ROWS:
        raise ValueError(f"the hT_quad order needs in % {MX_GROUP_ROWS} == 0, got {rows}")
    index = np.arange(rows)
    # [g, q, s, r] -> [g, s, q, r]
    return index.reshape(-1, MX_LANES, P // 8, 8).transpose(0, 2, 1, 3).reshape(-1, MX_BLOCK)


def mx_quantize(weight) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(data, scale)`` of ``weight[out, in]``: e4m3 bits uint8 ``[out, in]`` and E8M0 uint8 ``[out, in / 32]``."""
    values = _as_float32(weight)
    if not np.isfinite(values).all():
        raise ValueError("an MXFP8 weight must be finite")
    groups = mx_group_rows(values.shape[1])
    grouped = values[:, groups]
    exponent = ((grouped.view(np.uint32) >> 23) & 0xFF).max(axis=2).astype(np.int32)
    scale = np.clip(exponent - _E4M3_MAX_EXP, 0, 255).astype(np.uint8)
    step = np.ldexp(np.float64(1.0), scale.astype(np.int32) - 127)[:, :, None]
    quantized = (grouped / step).astype(np.float32).astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    data = np.empty(values.shape, dtype=np.uint8)
    data[:, groups] = quantized
    return data, scale


def mx_dequantize(data: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Return the float64 values ``[out, in]`` of ``(data, scale)`` from ``mx_quantize``."""
    groups = mx_group_rows(data.shape[1])
    values = np.asarray(data, dtype=np.uint8).view(ml_dtypes.float8_e4m3fn).astype(np.float64)
    step = np.ldexp(np.float64(1.0), np.asarray(scale, dtype=np.int32) - 127)
    out = np.empty(values.shape, dtype=np.float64)
    out[:, groups] = values[:, groups] * step[:, :, None]
    return out


def _scale_partition() -> np.ndarray:
    """Return ``[16]``: the quadrant partition of each scale row ``s`` (first partition 0)."""
    s = np.arange(MX_SCALE_ROWS)
    return 32 * (s // 4) + (s % 4)


def pack_from_mx(data_out_in, scale_out_groups) -> tuple[np.ndarray, np.ndarray]:
    """Pack pre-quantized hT_quad bytes into the kernel MX tensors ``(data [G, 128, 4 out], scale [128, G, out])``.

    ``data_out_in`` is the e4m3 bytes ``[out, in]`` uint8 and ``scale_out_groups`` the E8M0 scales ``[out, in / 32]``
    uint8, as ``mx_weights.mx_quantize`` returns them in the hT_quad order. ``data[g, p, 4 n + q]`` is the byte of
    ``W[n, 512 g + 128 q + p]``; ``scale[part, g, n]`` the scale of group ``g``, scale row ``s`` at partition
    ``32 (s // 4) + s % 4``, output column ``n``.
    """
    data_out_in = np.asarray(data_out_in, dtype=np.uint8)
    scale_out_groups = np.asarray(scale_out_groups, dtype=np.uint8)
    out, hid = data_out_in.shape
    groups = hid // MX_GROUP_ROWS
    # in = 512 g + 128 q + p -> [n, g, q, p] -> [g, p, n, q] -> [g, p, 4 n + q]
    d = data_out_in.reshape(out, groups, MX_LANES, P).transpose(1, 3, 0, 2)
    data = np.ascontiguousarray(d.reshape(groups, P, MX_LANES * out))
    sc = scale_out_groups.reshape(out, groups, MX_SCALE_ROWS)  # [n, g, s]
    scale = np.zeros((P, groups, out), dtype=np.uint8)
    for s, part in enumerate(_scale_partition()):
        scale[part] = sc[:, :, s].T  # [g, n]
    return data, scale


def pack_linear_mx(weight) -> tuple[np.ndarray, np.ndarray]:
    """Quantize a linear ``W[out, in]`` and pack it into the kernel MX tensors (``pack_from_mx`` of ``mx_quantize``)."""
    return pack_from_mx(*mx_quantize(weight))


def pack_qkv_mx(w_q, w_k, w_v) -> tuple[np.ndarray, np.ndarray]:
    """Pack the 9 conv weight tiles (q, k, v x 3 heads of 128) into ``(data [9, G, 128, 4*128], scale [9, 128, G, 128])``.

    Tile ``ci`` is output rows ``128 ci .. 128 ci + 127`` of ``cat(W_q, W_k, W_v) [1152, HID]``.
    """
    w_cat = np.concatenate([_as_float32(w) for w in (w_q, w_k, w_v)], axis=0)  # [1152, HID]
    n_tiles = w_cat.shape[0] // P
    datas, scales = [], []
    for ci in range(n_tiles):
        data, scale = pack_linear_mx(w_cat[P * ci : P * (ci + 1)])  # data [G,128,512], scale [128,G,128]
        datas.append(data)
        scales.append(scale)
    return np.stack(datas, 0), np.stack(scales, 0)


def pack_gb_mx(w_g, w_b) -> tuple[np.ndarray, np.ndarray]:
    """Pack ``cat(W_g, W_b) [387, HID]`` into ``(data [G, 128, 4*387], scale [128, G, 387])``."""
    w_gb = np.concatenate([_as_float32(w_g), _as_float32(w_b)], axis=0)  # [387, HID]
    return pack_linear_mx(w_gb)


FP8_E4M3FN_MAX = 448.0
FP8_PAIR_ROWS = 2 * P  # 256 contraction rows per nc_matmul in double_row mode


def quantize_per_channel_fp8(weight) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(data [out, in] e4m3fn, scale [out] f32)`` of a per-output-channel static FP8 quantization.

    The scale ``s_c`` is the smallest power of two at or above ``amax_c / 448`` (so ``|w / s_c| <= 448``, no clip).
    """
    wf = _as_float32(weight)
    if not np.isfinite(wf).all():
        raise ValueError("an FP8 weight must be finite")
    amax = np.abs(wf).max(axis=1)
    scale = np.where(amax > 0, np.exp2(np.ceil(np.log2(np.maximum(amax, 1e-30) / FP8_E4M3FN_MAX))), 1.0).astype(
        np.float32
    )
    data = (wf / scale[:, None]).astype(ml_dtypes.float8_e4m3fn)
    return data, scale


def pack_qkv_fp8(w_q, w_k, w_v, tile: int = P) -> tuple[np.ndarray, np.ndarray]:
    """Pack the 9 conv tiles into ``(data [9, 128, KC, 128] e4m3fn, scale [9, 128] f32)`` for the ``mx=2`` W phase.

    ``data[t, p, c, j]`` is the e4m3 byte of ``W_cat[128 t + j, 128 c + p] / scale[t, j]`` for
    ``W_cat = cat(W_q, W_k, W_v)``; ``scale[t, j]`` is the per-output-channel static scale.
    """
    w_cat = np.concatenate([_as_float32(w) for w in (w_q, w_k, w_v)], axis=0)  # [1152, HID]
    n_tiles, KC = w_cat.shape[0] // tile, w_cat.shape[1] // tile
    datas, scales = [], []
    for t in range(n_tiles):
        data, scale = quantize_per_channel_fp8(w_cat[tile * t : tile * t + tile])  # [128 j, HID], [128]
        datas.append(data.view(np.uint8).reshape(tile, KC, tile).transpose(2, 1, 0))  # [p, c, j]
        scales.append(scale)
    data_u8 = np.stack(datas, 0).view(ml_dtypes.float8_e4m3fn)  # [9, 128, KC, 128]
    return np.ascontiguousarray(data_u8), np.ascontiguousarray(np.stack(scales, 0))


def fp8_static_project(x, weight) -> np.ndarray:
    """Return the ``mx=2`` form of ``x @ weight.t()``: static e4m3 x (scale 1.0), per-channel e4m3 weight, fp32 pairs.

    Each ``nc_matmul`` in ``double_row`` mode contracts 256 rows (128 partitions x 2) in fp32; the pairs accumulate
    in fp32 PSUM, and the per-output-channel weight scale is applied once at the end.
    """
    xf = _as_float32(x)
    wf = _as_float32(weight)
    x_q = xf.astype(ml_dtypes.float8_e4m3fn).astype(np.float64)
    data, scale = quantize_per_channel_fp8(wf)
    w_q = data.astype(np.float64)
    pairs = xf.shape[1] // FP8_PAIR_ROWS
    acc = np.zeros((xf.shape[0], wf.shape[0]), dtype=np.float32)
    for pc in range(pairs):
        cols = slice(FP8_PAIR_ROWS * pc, FP8_PAIR_ROWS * pc + FP8_PAIR_ROWS)
        acc = acc + (x_q[:, cols] @ w_q[:, cols].T).astype(np.float32)
    return acc * scale[None, :].astype(np.float32)


def mx3_project(x, weight) -> np.ndarray:
    """Return the ``mx=3`` form of ``x @ weight.t()``: static e4m3 x (per-tensor scale 1.0), MXFP8 weight, fp32 groups.

    Same as the mx=1 form but x is a plain e4m3 cast with a constant E8M0 scale (127 = 2^0), no ``quantize_mx``.
    """
    xf = _as_float32(x)
    wf = _as_float32(weight)
    x_deq = xf.astype(ml_dtypes.float8_e4m3fn).astype(np.float64)
    w_deq = mx_dequantize(*mx_quantize(wf))
    groups = xf.shape[1] // MX_GROUP_ROWS
    acc = np.zeros((xf.shape[0], wf.shape[0]), dtype=np.float32)
    for g in range(groups):
        cols = slice(g * MX_GROUP_ROWS, (g + 1) * MX_GROUP_ROWS)
        acc = acc + (x_deq[:, cols] @ w_deq[:, cols].T).astype(np.float32)
    return acc


def mx_project(x, weight) -> np.ndarray:
    """Return the MXFP8 form of ``x @ weight.t()``: an fp32 sum over the 512-row groups of the dequantized operands.

    Both ``x [T, HID]`` and ``weight [out, HID]`` quantize per hT_quad group. Each group's dequantized matmul is one
    ``nc_matmul_mx`` (fp32), and the groups accumulate in fp32 PSUM, so this is the kernel value up to the order of
    the fp32 sum inside one 512-row group.
    """
    xf = _as_float32(x)
    wf = _as_float32(weight)
    x_deq = mx_dequantize(*mx_quantize(xf))
    w_deq = mx_dequantize(*mx_quantize(wf))
    groups = xf.shape[1] // MX_GROUP_ROWS
    acc = np.zeros((xf.shape[0], wf.shape[0]), dtype=np.float32)
    for g in range(groups):
        cols = slice(g * MX_GROUP_ROWS, (g + 1) * MX_GROUP_ROWS)
        acc = acc + (x_deq[:, cols] @ w_deq[:, cols].T).astype(np.float32)
    return acc
