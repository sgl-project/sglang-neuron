# SPDX-License-Identifier: Apache-2.0
"""MXFP8 weights of the Kimi-K3 prefill: the quantizer, the kernel packers, the torch references and the MX linears.

MXFP8 is float8_e4m3fn values with one E8M0 scale for each group of 32 values along the group axis. The scale
exponent of a group is the largest fp32 exponent field of its values minus 7 (clipped to 0..255), and each value is
``x / 2^(s - 127)`` rounded to the nearest e4m3fn value, ties to even (the hardware ``quantize_mx`` rule). The largest
value of a group lands in [128, 256), so no value clips. Each (row, group) is quantized alone, so a slice of the rows
gives the bytes of the same slice of the whole tensor.

Group orders (which 32 values of the group axis form one group, and the index of its scale):

- ``consecutive``: group ``b`` holds the values ``32 b .. 32 b + 31``; scale index ``b``.
- ``hT_quad``: the axis is ``512 G`` values. Group ``(g, s)`` holds the values ``128 (4 g + q) + 8 s + r`` for ``q``
  in 0..3 and ``r`` in 0..7; scale index ``16 g + s``.

The prefill MX linear kernel takes a weight ``W [in, out]`` (the checkpoint weight transposed) as:

* ``w_mx`` ``[R, 4 out]`` uint8, byte ``4 n + q`` of row ``r``: order hT_quad, ``r = 128 g + p`` holds input
  ``128 (4 g + q) + p`` (``R = 128 G``, ``G = ceil(in / 512)``); order consecutive, row ``r`` holds input ``4 r + q``.
* ``w_scale`` ``[S, out]`` uint8: row ``16 g + s`` is the scale of group ``s`` of input tile ``g``.

With order hT_quad, an input width that is not a multiple of 512 gets zero input rows up to the next multiple: a zero
row adds 0 to each sum and does not change the scale of its group. The kernel quantizes its input in the same groups
and sums in fp32. ``MxLinear`` without a kernel backend runs ``mx_linear_reference``: the MXFP8 value of each weight
and of each input, multiplied in fp32.

The KDA front projections take another MX layout (``pack_kda_front_mx``): one 512-value group per slot, with the
scale rows at the quadrant partitions ``32 (s // 4) + s % 4``.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor

GROUP = 32
"""Values per scale."""

MX_LINEAR_ORDERS = ("consecutive", "hT_quad")
"""The group orders of the prefill MX linear kernel."""

AXES = ("in", "out")

P_MAX = 128
MX_LANES = 4
MX_GROUP_ROWS = P_MAX * MX_LANES
"""Input rows of one hT_quad tile (one MX matmul contraction)."""
MX_SCALE_ROWS = MX_GROUP_ROWS // GROUP
"""Scale rows of one hT_quad tile."""
MX_LINEAR_MAX_ROWS = 8192
"""The most rows of one MX linear kernel call."""

_E4M3_EXPONENT_OFFSET = 7
_SPAN = {"consecutive": GROUP, "hT_quad": MX_GROUP_ROWS}
_CHUNK_VALUES = 1 << 20


def _check_order(order: str) -> None:
    if order not in MX_LINEAR_ORDERS:
        raise ValueError(f"The MX order {order!r} is unknown. Use one of {MX_LINEAR_ORDERS}.")


def _group_view(order: str, rows: int, length: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The 5-D view of a ``[rows, length]`` block and the dims of one group in it."""
    if order == "consecutive":
        return (rows, length // GROUP, 1, GROUP, 1), (2, 3, 4)
    return (rows, length // MX_GROUP_ROWS, MX_LANES, MX_SCALE_ROWS, 8), (2, 4)


def mx_quantize(weight: Tensor, order: str, axis: str = "in", what: str = "the weight") -> tuple[Tensor, Tensor]:
    """Quantize a BF16 weight ``[out, in]`` to MXFP8.

    Args:
        weight: A 2-D BF16 tensor.
        order: ``consecutive`` or ``hT_quad``.
        axis: The group axis: ``in`` (the last axis) or ``out`` (the first axis).
        what: The name of the weight, for the error text.

    Returns:
        ``(data, scale)``: uint8 e4m3fn bytes ``[out, in]`` and uint8 E8M0 scales, ``[out, in / 32]`` (axis in) or
        ``[out / 32, in]`` (axis out), in the scale order of ``order``.

    Raises:
        ValueError: The weight is not 2-D BF16, its group axis does not fit the order, or it has a non-finite value.

    Example:
        >>> data, scale = mx_quantize(torch.ones(2, 512, dtype=torch.bfloat16), "hT_quad")
        >>> data.shape, scale.shape, int(scale[0, 0]), int(data[0, 0])
        (torch.Size([2, 512]), torch.Size([2, 16]), 120, 112)
    """
    _check_order(order)
    if axis not in AXES:
        raise ValueError(f"The MX group axis {axis!r} is unknown. Use one of {AXES}.")
    if weight.dtype != torch.bfloat16 or weight.dim() != 2:
        raise ValueError(f"MX quantization takes a 2-D BF16 weight. {what} is {weight.dtype} {tuple(weight.shape)}.")
    if axis == "out":
        data, scale = mx_quantize(weight.t().contiguous(), order, "in", what)
        return data.t().contiguous(), scale.t().contiguous()
    rows, length = weight.shape
    if length % _SPAN[order]:
        raise ValueError(f"The MX {order} order needs a group axis that is a multiple of {_SPAN[order]}. "
                         f"{what} has {length}.")
    weight = weight.detach().cpu().contiguous()
    data = torch.empty((rows, length), dtype=torch.uint8)
    scale = torch.empty((rows, length // GROUP), dtype=torch.uint8)
    step = max(1, _CHUNK_VALUES // max(length, 1))
    for start in range(0, rows, step):
        stop = min(rows, start + step)
        block = weight[start:stop]
        shape, dims = _group_view(order, stop - start, length)
        # The bf16 exponent field is the fp32 exponent field.
        exponent = torch.bitwise_right_shift(block.view(torch.int16), 7).bitwise_and_(0xFF).view(shape)
        group_max = exponent.amax(dim=dims, keepdim=True)
        if bool((group_max == 0xFF).any()):
            raise ValueError(f"{what}: the weight has a non-finite value; MX quantization needs finite values.")
        exp = (group_max - _E4M3_EXPONENT_OFFSET).clamp_(0, 255)
        # 2^(127 - s) from its fp32 bits: s <= 247, so the exponent field 254 - s is normal.
        multiplier = ((254 - exp.to(torch.int32)) << 23).view(torch.float32)
        wide = block.view(shape).to(torch.float32).mul_(multiplier)
        data[start:stop].view(torch.float8_e4m3fn).view(shape).copy_(wide)
        scale[start:stop].copy_(exp.to(torch.uint8).reshape(stop - start, length // GROUP))
    return data, scale


def mx_dequantize(data: Tensor, scale: Tensor, order: str, axis: str = "in") -> Tensor:
    """Return the fp32 values of MXFP8 bytes and scales (the inverse of ``mx_quantize``)."""
    _check_order(order)
    if axis == "out":
        return mx_dequantize(data.t().contiguous(), scale.t().contiguous(), order, "in").t()
    rows, length = data.shape
    shape, _ = _group_view(order, rows, length)
    if order == "hT_quad":
        factor = scale.reshape(rows, length // MX_GROUP_ROWS, 1, MX_SCALE_ROWS, 1)
    else:
        factor = scale.reshape(rows, length // GROUP, 1, 1, 1)
    values = data.view(torch.float8_e4m3fn).to(torch.float32).view(shape)
    return (values * torch.exp2(factor.to(torch.float32) - 127.0)).reshape(rows, length)


def padded_in_features(in_features: int, order: str) -> int:
    """Return the input width of the quantized weight: the next multiple of 512 (hT_quad) or the width itself.

    Example:
        >>> padded_in_features(384, "hT_quad"), padded_in_features(384, "consecutive")
        (512, 384)
    """
    _check_order(order)
    if order == "consecutive":
        if in_features % GROUP:
            raise ValueError(f"The MX order consecutive needs an input width that is a multiple of {GROUP}, "
                             f"got {in_features}.")
        return in_features
    return math.ceil(in_features / MX_GROUP_ROWS) * MX_GROUP_ROWS


def mx_linear_shapes(in_features: int, out_features: int, order: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return the ``(w_mx, w_scale)`` shapes of the kernel tensors of a weight ``[out, in]``.

    Example:
        >>> mx_linear_shapes(1024, 512, "hT_quad")
        ((256, 2048), (32, 512))
    """
    padded = padded_in_features(in_features, order)
    return (padded // MX_LANES, MX_LANES * out_features), (padded // GROUP, out_features)


def pack_mx_linear(data: Tensor, scale: Tensor, order: str) -> tuple[Tensor, Tensor]:
    """Pack the ``mx_quantize`` output of a weight ``[out, in]`` (axis in) into the MX linear kernel tensors.

    Returns:
        ``(w_mx [in / 4, 4 out], w_scale [in / 32, out])``, both contiguous uint8.
    """
    _check_order(order)
    cols, rows = data.shape
    if rows != padded_in_features(rows, order):
        raise ValueError(f"The MX {order} weight has {rows} input rows; pad them to "
                         f"{padded_in_features(rows, order)}.")
    if tuple(scale.shape) != (cols, rows // GROUP):
        raise ValueError(f"The MX scales have shape {tuple(scale.shape)}; a weight [{cols}, {rows}] needs "
                         f"[{cols}, {rows // GROUP}].")
    data_in_out = data.to(torch.uint8).t()
    if order == "hT_quad":
        # Input 128 (4 g + q) + p -> [g, p, n, q].
        lanes = data_in_out.reshape(rows // MX_GROUP_ROWS, MX_LANES, P_MAX, cols).permute(0, 2, 3, 1)
    else:
        # Input 4 r + q -> [r, n, q].
        lanes = data_in_out.reshape(rows // MX_LANES, MX_LANES, cols).permute(0, 2, 1)
    w_mx = lanes.reshape(rows // MX_LANES, MX_LANES * cols).contiguous()
    return w_mx, scale.to(torch.uint8).t().contiguous()


def unpack_mx_linear(w_mx: Tensor, w_scale: Tensor, order: str) -> tuple[Tensor, Tensor]:
    """Return ``(data [out, in], scale [out, in / 32])`` of the kernel tensors (the inverse of ``pack_mx_linear``)."""
    _check_order(order)
    w_mx = w_mx.reshape(-1, w_mx.shape[-1])
    quads, width = w_mx.shape
    cols, rows = width // MX_LANES, quads * MX_LANES
    lanes = w_mx.reshape(quads, cols, MX_LANES)
    if order == "hT_quad":
        data_in_out = lanes.reshape(rows // MX_GROUP_ROWS, P_MAX, cols, MX_LANES).permute(0, 3, 1, 2)
    else:
        data_in_out = lanes.permute(0, 2, 1)
    return data_in_out.reshape(rows, cols).t().contiguous(), w_scale.t().contiguous()


def _as_bf16(weight: Tensor, what: str) -> Tensor:
    weight = weight.detach()
    if weight.dtype == torch.bfloat16:
        return weight
    narrow = weight.to(torch.bfloat16)
    if not torch.equal(narrow.to(weight.dtype), weight):
        raise ValueError(f"{what}: the MXFP8 load quantizes BF16 weights, and a value of the {weight.dtype} weight "
                         "has no BF16 value.")
    return narrow


def quantize_mx_linear(weight: Tensor, order: str, what: str = "the weight") -> tuple[Tensor, Tensor]:
    """Return the kernel tensors ``(w_mx, w_scale)`` of a BF16 linear weight ``[out, in]``.

    With order hT_quad, the input axis gets zero rows up to ``padded_in_features(in, order)``. A meta weight gives
    meta tensors of the kernel shapes.

    Example:
        >>> w_mx, w_scale = quantize_mx_linear(torch.ones(512, 1024, dtype=torch.bfloat16), "hT_quad")
        >>> w_mx.shape, w_scale.shape
        (torch.Size([256, 2048]), torch.Size([32, 512]))
    """
    out_features, in_features = weight.shape
    padded = padded_in_features(in_features, order)
    if weight.is_meta:
        mx_shape, scale_shape = mx_linear_shapes(in_features, out_features, order)
        return (torch.empty(mx_shape, dtype=torch.uint8, device="meta"),
                torch.empty(scale_shape, dtype=torch.uint8, device="meta"))
    weight = _as_bf16(weight, what).cpu()
    if padded != in_features:
        weight = torch.nn.functional.pad(weight, (0, padded - in_features))
    return pack_mx_linear(*mx_quantize(weight, order, "in", what), order)


def dequantize_mx_linear(w_mx: Tensor, w_scale: Tensor, order: str) -> Tensor:
    """Return the fp32 weight ``[out, in_padded]`` of the kernel tensors."""
    data, scale = unpack_mx_linear(w_mx.cpu(), w_scale.cpu(), order)
    return mx_dequantize(data, scale, order)


def mx_linear_reference(x: Tensor, w_mx: Tensor, w_scale: Tensor, order: str) -> Tensor:
    """Return ``x @ W`` of the MXFP8 weight, with the MXFP8 value of each input, in the dtype of ``x``.

    ``x`` ``[..., in]`` gets zero columns up to the padded width of the weight. Each row takes the groups of
    ``order`` along the input axis (the input quantize of the kernel). The products sum in fp32.
    """
    weight = dequantize_mx_linear(w_mx, w_scale, order)
    padded = weight.shape[1]
    lead = x.shape[:-1]
    rows = x.detach().reshape(-1, x.shape[-1]).to(torch.bfloat16).cpu()
    if rows.shape[1] != padded:
        rows = torch.nn.functional.pad(rows, (0, padded - rows.shape[1]))
    values = mx_dequantize(*mx_quantize(rows, order, "in", "the MX linear input"), order)
    out = values @ weight.t()
    return out.reshape(*lead, weight.shape[0]).to(device=x.device, dtype=x.dtype)


class MxLinear(nn.Module):
    """A bias-free linear layer with an MXFP8 weight in the MX linear kernel layout.

    It holds the buffers ``w_mx`` and ``w_scale`` (``pack_mx_linear``) and no BF16 weight. ``linear_fn`` is the
    kernel backend, ``linear_fn(x, layer, feature_major=False) -> y``; None runs ``mx_linear_reference``, or gives an
    empty output of the right shape for a meta input. ``forward_feature_major(x_t [in, L])`` gives the same
    ``[L, out]`` as ``forward(x_t.t())``; the kernel takes the feature-major input as it is.

    Example:
        >>> lin = MxLinear.from_weight(torch.ones(768, 1024, dtype=torch.bfloat16), "hT_quad", "shared_gate_up")
        >>> lin.in_features, lin.out_features, lin.order
        (1024, 768, 'hT_quad')
    """

    def __init__(self, name: str, in_features: int, out_features: int, order: str, w_mx: Tensor, w_scale: Tensor):
        super().__init__()
        self.name = name
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.order = order
        shapes = mx_linear_shapes(self.in_features, self.out_features, order)
        if (tuple(w_mx.shape), tuple(w_scale.shape)) != shapes:
            raise ValueError(f"{name}: the MX tensors have shapes {tuple(w_mx.shape)} and {tuple(w_scale.shape)}; a "
                             f"{order} weight [{self.out_features}, {self.in_features}] needs {shapes[0]} and "
                             f"{shapes[1]}.")
        self.register_buffer("w_mx", w_mx)
        self.register_buffer("w_scale", w_scale)
        self.linear_fn: Callable[..., Tensor] | None = None

    @classmethod
    def from_weight(cls, weight: Tensor, order: str, name: str) -> "MxLinear":
        """Quantize a BF16 (or meta) weight ``[out, in]`` into an ``MxLinear``."""
        out_features, in_features = weight.shape
        return cls(name, in_features, out_features, order, *quantize_mx_linear(weight, order, name))

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"{self.name}: the input has {x.shape[-1]} columns; the layer takes {self.in_features}.")
        if self.linear_fn is not None:
            return self.linear_fn(x, self)
        if x.is_meta or self.w_mx.is_meta:
            return torch.empty((*x.shape[:-1], self.out_features), dtype=x.dtype, device=x.device)
        return mx_linear_reference(x, self.w_mx, self.w_scale, self.order)

    def forward_feature_major(self, x_t: Tensor) -> Tensor:
        """Return ``x @ W`` ``[L, out]`` for a feature-major ``x_t [in, L]``."""
        if x_t.dim() != 2 or x_t.shape[0] != self.in_features:
            raise ValueError(f"{self.name}: the feature-major input has shape {tuple(x_t.shape)}; the layer takes "
                             f"[{self.in_features}, L].")
        if self.linear_fn is not None:
            return self.linear_fn(x_t, self, True)
        return self.forward(x_t.t())

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, order={self.order}"


def quantize_mx_head_linear(weight: Tensor, what: str = "the weight") -> tuple[Tensor, Tensor]:
    """Return the kernel tensors of G per-head weights ``W [G, in, out]`` (``x_h @ W[h]``), consecutive groups.

    Each head is the consecutive ``pack_mx_linear`` layout: ``w_mx [G, in / 4, 4 out]``, ``w_scale [G, in / 32,
    out]``, one E8M0 scale for each 32 consecutive inputs of one output. A meta weight gives meta tensors.

    Example:
        >>> w_mx, w_scale = quantize_mx_head_linear(torch.ones(2, 128, 512, dtype=torch.bfloat16))
        >>> w_mx.shape, w_scale.shape
        (torch.Size([2, 32, 2048]), torch.Size([2, 4, 512]))
    """
    heads, in_features, out_features = weight.shape
    mx_shape, scale_shape = mx_linear_shapes(in_features, out_features, "consecutive")
    if weight.is_meta:
        return (torch.empty((heads, *mx_shape), dtype=torch.uint8, device="meta"),
                torch.empty((heads, *scale_shape), dtype=torch.uint8, device="meta"))
    packs = [
        pack_mx_linear(*mx_quantize(_as_bf16(weight[head], what).t().contiguous().cpu(), "consecutive", "in",
                                    f"{what} head {head}"), "consecutive")
        for head in range(heads)
    ]
    return torch.stack([w for w, _ in packs]), torch.stack([s for _, s in packs])


def mx_head_linear_reference(x_t: Tensor, w_mx: Tensor, w_scale: Tensor) -> Tensor:
    """Return ``out[h] = (Q(x_t[h]^T) @ W[h])^T`` ``[G, out, M]`` in fp32 math and the dtype of ``x_t [G, in, M]``."""
    return torch.stack([mx_linear_reference(x_t[head].t(), w_mx[head], w_scale[head], "consecutive").t()
                        for head in range(x_t.shape[0])])


class MxHeadLinear(nn.Module):
    """G bias-free per-head linears with MXFP8 weights, feature-major: ``out[h] = W[h]^T x_t[h]``.

    It holds ``w_mx [G, in / 4, 4 out]`` and ``w_scale [G, in / 32, out]`` (``quantize_mx_head_linear``) and no
    BF16 weight. ``linear_fn(x_t, layer) -> out`` is the kernel backend; None runs ``mx_head_linear_reference``.

    Example:
        >>> lin = MxHeadLinear.from_weight(torch.ones(2, 128, 512, dtype=torch.bfloat16), "w_uk")
        >>> lin(torch.ones(2, 128, 4, dtype=torch.bfloat16)).shape
        torch.Size([2, 512, 4])
    """

    def __init__(self, name: str, w_mx: Tensor, w_scale: Tensor):
        super().__init__()
        self.name = name
        self.heads, quads, width = w_mx.shape
        self.in_features = quads * MX_LANES
        self.out_features = width // MX_LANES
        if tuple(w_scale.shape) != (self.heads, self.in_features // GROUP, self.out_features):
            raise ValueError(f"{name}: the MX scales have shape {tuple(w_scale.shape)}; the data {tuple(w_mx.shape)} "
                             f"needs [{self.heads}, {self.in_features // GROUP}, {self.out_features}].")
        self.order = "consecutive"
        self.register_buffer("w_mx", w_mx)
        self.register_buffer("w_scale", w_scale)
        self.linear_fn: Callable[..., Tensor] | None = None

    @classmethod
    def from_weight(cls, weight: Tensor, name: str) -> "MxHeadLinear":
        """Quantize BF16 (or meta) per-head weights ``[G, in, out]`` into an ``MxHeadLinear``."""
        return cls(name, *quantize_mx_head_linear(weight, name))

    def forward(self, x_t: Tensor) -> Tensor:
        if x_t.dim() != 3 or tuple(x_t.shape[:2]) != (self.heads, self.in_features):
            raise ValueError(f"{self.name}: the input has shape {tuple(x_t.shape)}; the layer takes "
                             f"[{self.heads}, {self.in_features}, M].")
        if self.linear_fn is not None:
            return self.linear_fn(x_t, self)
        if x_t.is_meta or self.w_mx.is_meta:
            return torch.empty((self.heads, self.out_features, x_t.shape[2]), dtype=x_t.dtype, device=x_t.device)
        return mx_head_linear_reference(x_t, self.w_mx, self.w_scale)

    def extra_repr(self) -> str:
        return f"heads={self.heads}, in_features={self.in_features}, out_features={self.out_features}"


def install_mx_linear_backend(module: nn.Module, linear_fn: Callable[..., Tensor] | None,
                              head_linear_fn: Callable[..., Tensor] | None) -> int:
    """Set the backends of every ``MxLinear`` and ``MxHeadLinear`` under ``module``; return the count."""
    count = 0
    for sub in module.modules():
        if isinstance(sub, MxLinear):
            sub.linear_fn = linear_fn
            count += 1
        elif isinstance(sub, MxHeadLinear):
            sub.linear_fn = head_linear_fn
            count += 1
    return count


def _kda_scale_partitions() -> Tensor:
    """The partition of each of the 16 scale rows of one 512-value group: ``32 (s // 4) + s % 4``."""
    s = torch.arange(MX_SCALE_ROWS)
    return 32 * (s // 4) + s % 4


def pack_kda_front_mx(data: Tensor, scale: Tensor) -> tuple[Tensor, Tensor]:
    """Pack hT_quad MXFP8 bytes of a weight ``[out, in]`` into the KDA front kernel tensors.

    Args:
        data: e4m3fn bytes ``[out, in]`` uint8 (``mx_quantize`` order hT_quad).
        scale: E8M0 scales ``[out, in / 32]`` uint8 in the hT_quad scale order.

    Returns:
        ``(data [G, 128, 4 out], scale [128, G, out])`` uint8 with ``G = in / 512``: ``data[g, p, 4 n + q]`` is the
        byte of ``W[n, 512 g + 128 q + p]``; ``scale[32 (s // 4) + s % 4, g, n]`` is the scale of group ``s`` of tile
        ``g`` of column ``n``, and the other partitions hold 0.
    """
    out, hidden = data.shape
    groups = hidden // MX_GROUP_ROWS
    if hidden % MX_GROUP_ROWS or tuple(scale.shape) != (out, hidden // GROUP):
        raise ValueError(f"The KDA front MX pack takes data [out, 512 G] and scales [out, 16 G]; got "
                         f"{tuple(data.shape)} and {tuple(scale.shape)}.")
    # in = 512 g + 128 q + p -> [n, g, q, p] -> [g, p, n, q]
    lanes = data.to(torch.uint8).reshape(out, groups, MX_LANES, P_MAX).permute(1, 3, 0, 2)
    packed = lanes.reshape(groups, P_MAX, MX_LANES * out).contiguous()
    table = torch.zeros((P_MAX, groups, out), dtype=torch.uint8)
    table[_kda_scale_partitions()] = scale.to(torch.uint8).reshape(out, groups, MX_SCALE_ROWS).permute(2, 1, 0)
    return packed, table


def kda_front_mx_tensors(qkv: Tensor, gb: Tensor) -> dict[str, Tensor]:
    """Return the MXFP8 KDA front weights in the kernel layout.

    Args:
        qkv: ``[3 p, in]`` BF16, the q, k and v projection rows (group-major when the front runs per head group).
        gb: ``[p + hc, in]`` BF16, the g and b projection rows (group-major likewise).

    Returns:
        ``w_qkv_mx [3 p / 128, G, 128, 512]``, ``w_qkv_scale [3 p / 128, 128, G, 128]`` (one tile per 128 output
        rows), ``w_gb_mx [G, 128, 4 (p + hc)]`` and ``w_gb_scale [128, G, p + hc]``, all uint8, ``G = in / 512``.
    """
    data, scale = mx_quantize(qkv, "hT_quad", "in", "the KDA q/k/v weight")
    tiles = [pack_kda_front_mx(data[P_MAX * t:P_MAX * (t + 1)], scale[P_MAX * t:P_MAX * (t + 1)])
             for t in range(data.shape[0] // P_MAX)]
    gb_data, gb_scale = pack_kda_front_mx(*mx_quantize(gb, "hT_quad", "in", "the KDA g/b weight"))
    return {
        "w_qkv_mx": torch.stack([d for d, _ in tiles]),
        "w_qkv_scale": torch.stack([s for _, s in tiles]),
        "w_gb_mx": gb_data,
        "w_gb_scale": gb_scale,
    }
