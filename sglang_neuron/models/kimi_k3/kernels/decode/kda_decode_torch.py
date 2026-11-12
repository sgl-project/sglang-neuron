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

"""Torch reference and host layout functions of the Kimi-K3 KDA decode attention.

The global weights use the serving orientation ``weight[out, in]`` and these names:

| Name | Shape | Dtype |
|---|---|---|
| ``q_proj``, ``k_proj``, ``v_proj``, ``g_proj`` | ``[W, H]`` | bf16 |
| ``o_proj`` | ``[H, W]`` | bf16 |
| ``f_a_proj`` | ``[128, H]`` | bf16 |
| ``f_b_proj`` | ``[W, 128]`` | bf16 |
| ``b_proj`` | ``[heads, H]`` | bf16 |
| ``q_conv1d``, ``k_conv1d``, ``v_conv1d`` | ``[W, 1, 4]`` | fp32 |
| ``a_log`` | ``[heads]`` | fp32 |
| ``dt_bias`` | ``[W]`` | fp32 |
| ``o_norm`` | ``[128]`` | fp32 |

Here ``W = heads * 128``. The global states are the recurrent state ``[heads, 128, 128]`` (head,
key, value) and the conv state ``[3, W, 3]`` (q, k, v; channel; taps oldest first), both bf16.
"""

import ml_dtypes
import numpy as np
import torch
import torch.nn.functional as F

from .tkg_kda import (
    CONV_TAPS,
    GATE_LOWER_BOUND,
    HEAD_DIM,
    KDA_WEIGHT_NAMES,
    L2_EPS,
    NORM_EPS,
    kda_in_columns,
    kda_in_head_major_permutation,
    kda_page_offsets,
    kda_small_columns,
)
from .tkg_mx_quant import MX_LANES
from .mx_weights_torch import MX_BLOCK, mx_quantize, mx_rows, pack_mx_hT

_CONV_NAMES = ("q_conv1d", "k_conv1d", "v_conv1d")

MX_WEIGHT_CLASSES = ("kda_w_in", "kda_w_out")
"""MXFP8 weight classes of the KDA builders (``mx_weights``). A loader reads the class names: a tuple, not a map, so the
loader keeps its own list of the input weights of each class."""
MX_WEIGHT_ORDERS = {"kda_w_in": ("hT_quad",), "kda_w_out": ("hT_quad",)}
"""The group orders of each class of ``MX_WEIGHT_CLASSES``."""
MX_FROM_BF16 = ("kda_w_out",)
"""Classes that the builders quantize from the BF16 weights (the padded-lane ``kda_w_out`` of ``kda_w_out_form``):
a loader gives them the BF16 linears and the order."""


def to_numpy_bf16(tensor: torch.Tensor) -> np.ndarray:
    """Return a bf16 torch tensor as a numpy ``ml_dtypes.bfloat16`` array."""
    return tensor.detach().to(torch.bfloat16).contiguous().view(torch.int16).numpy().view(ml_dtypes.bfloat16)


def from_numpy_bf16(array: np.ndarray) -> torch.Tensor:
    """Return a numpy ``ml_dtypes.bfloat16`` array as a bf16 torch tensor."""
    return torch.from_numpy(np.ascontiguousarray(array).view(np.int16)).view(torch.bfloat16)


def _linear_bf16(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Return ``bf16(x @ weight.T)`` with fp32 accumulation."""
    return (x.float() @ weight.float().T).to(torch.bfloat16)


def _l2_norm(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.square().sum(dim=-1, keepdim=True) + L2_EPS)


def _decode_conv(values: torch.Tensor, state: torch.Tensor, weight: torch.Tensor, conv_dtype: str = "bf16"):
    """Run the depthwise conv of one projection for one token.

    Args:
        values: bf16 ``[W]`` new projection values.
        state: bf16 ``[W, 3]`` conv state, oldest tap first.
        weight: fp32 ``[W, 1, 4]`` conv weight.
        conv_dtype: ``bf16`` (bf16 weights, products and output) or ``fp32`` (fp32 weights, products, sum
            and output), see ``tkg_kda.CONV_DTYPES``.

    Returns:
        The ``[W]`` conv output (bf16, or fp32 for ``fp32``) and the bf16 ``[W, 3]`` next state.
    """
    window = torch.cat((state.to(torch.bfloat16), values.unsqueeze(-1)), dim=-1)
    if conv_dtype == "fp32":
        summed = (window.float() * weight.reshape(window.shape).float()).sum(dim=-1)
        return F.silu(summed), window[:, 1:]
    product = window * weight.reshape(window.shape).to(torch.bfloat16)
    summed = product.float().sum(dim=-1).to(torch.bfloat16)
    return F.silu(summed.float()).to(torch.bfloat16), window[:, 1:]


def kda_decode_torch_ref(
    x: torch.Tensor,
    weights: dict,
    recurrent_state: torch.Tensor,
    conv_state: torch.Tensor,
    reset: bool,
    o_proj_shards: int = 1,
    conv_dtype: str = "bf16",
    state_dtype: str = "bf16",
):
    """Run the global KDA decode attention for one token.

    The reference follows the bf16 and fp32 boundaries of the serving decode stack. Every linear
    accumulates in fp32 and rounds its output to bf16.

    Args:
        x: bf16 ``[1, H]`` normalized hidden.
        weights: Global weights, see the module docstring.
        recurrent_state: bf16 ``[heads, 128, 128]``.
        conv_state: bf16 ``[3, W, 3]``.
        reset: Use zero states (null page or position 0).
        o_proj_shards: Round the ``o_proj`` partial of each of this many head shards to bf16 and
            sum the partials in fp32, as a TP all-reduce of bf16 partials does.
        conv_dtype: Form of the short convolution, see ``_decode_conv``.
        state_dtype: Form of the recurrent state (``tkg_kda.KDA_STATE_DTYPES``). ``bf16`` rounds the next state
            to bf16; the fp32 forms return it in fp32.

    Returns:
        A tuple of the bf16 ``[1, H]`` output, the ``[heads, 128, 128]`` next recurrent state (bf16, or fp32)
        and the bf16 ``[3, W, 3]`` next conv state.

    Example:
        >>> out, rec, conv = kda_decode_torch_ref(x, weights, rec, conv, reset=False)
    """
    heads = weights["a_log"].shape[0]
    width = heads * HEAD_DIM
    xb = x.to(torch.bfloat16).reshape(1, -1)
    if reset:
        recurrent_state = torch.zeros_like(recurrent_state)
        conv_state = torch.zeros_like(conv_state)

    conv_outputs = []
    next_conv = []
    for p, name in enumerate(("q_proj", "k_proj", "v_proj")):
        values = _linear_bf16(xb, weights[name])[0]
        out, nxt = _decode_conv(values, conv_state[p], weights[_CONV_NAMES[p]], conv_dtype)
        conv_outputs.append(out)
        next_conv.append(nxt)
    q, k, v = (t.float().reshape(heads, HEAD_DIM) for t in conv_outputs)

    q_n = _l2_norm(q) * HEAD_DIM**-0.5
    k_n = _l2_norm(k)
    gate = _linear_bf16(_linear_bf16(xb, weights["f_a_proj"]), weights["f_b_proj"])[0]
    gate = gate.float().reshape(heads, HEAD_DIM)
    a_exp = torch.exp(weights["a_log"].float()).reshape(heads, 1)
    dt_bias = weights["dt_bias"].float().reshape(heads, HEAD_DIM)
    decay = torch.exp(GATE_LOWER_BOUND * torch.sigmoid(a_exp * (gate + dt_bias)))
    beta = torch.sigmoid(_linear_bf16(xb, weights["b_proj"])[0].float()).reshape(heads, 1)

    state = recurrent_state.float()
    prediction = torch.einsum("hkv,hk->hv", state, k_n * decay)
    difference = (v - prediction) * beta
    state_new = state * decay.unsqueeze(-1) + k_n.unsqueeze(-1) * difference.unsqueeze(-2)
    out = torch.einsum("hkv,hk->hv", state_new, q_n).to(torch.bfloat16).float()

    out_gate = torch.sigmoid(_linear_bf16(xb, weights["g_proj"])[0].float()).reshape(heads, HEAD_DIM)
    normed = out * torch.rsqrt(out.square().mean(dim=-1, keepdim=True) + NORM_EPS)
    o = (normed * weights["o_norm"].float() * out_gate).to(torch.bfloat16).reshape(1, width)

    o_proj = weights["o_proj"]
    shard = width // o_proj_shards
    y = torch.zeros(1, o_proj.shape[0], dtype=torch.float32)
    for s in range(o_proj_shards):
        cols = slice(s * shard, (s + 1) * shard)
        y += _linear_bf16(o[:, cols], o_proj[:, cols]).float()
    state_out = state_new.to(torch.bfloat16) if state_dtype == "bf16" else state_new.float()
    return y.to(torch.bfloat16), state_out, torch.stack(next_conv)


def build_kda_rank_weights(
    weights: dict,
    tp_size: int,
    tp_rank: int,
    conv_dtype: str = "bf16",
    mx_weights: dict = None,
    a_exp_params=False,
    w_in_head_major: bool = False,
) -> dict:
    """Build the kernel weights of one TP rank from the global weights.

    Args:
        weights: Global weights, see the module docstring.
        tp_size: TP degree.
        tp_rank: TP rank.
        conv_dtype: Form of the short convolution (``tkg_kda.CONV_DTYPES``). ``fp32`` also writes the fp32
            conv weights to ``kda_params`` at ``kda_small_columns(Hl, "fp32")["conv32"]``, for a kernel call with
            the same ``conv_dtype``.
        a_exp_params: False (default) keeps the plain ``kda_params`` width. True, or a function of the fp32 A_log
            values (a ``torch.Tensor``) that returns their exp, also writes ``exp(A_log)`` to the last ``Hl`` columns
            of ``kda_params`` (``kda_small_columns`` ``a_exp``), so the kernel bodies read it in place of computing it.
            True takes ``torch.exp``; its bits equal the device's only where the two exps agree (torch and numpy
            differ in the last bit of ~40 % of fp32 inputs). A served pack takes the device's own values
            (``kda_a_exp_init``).
        mx_weights: None, or ``{class: order}`` of ``MX_WEIGHT_ORDERS``. ``{"kda_w_in": "hT_quad"}`` builds the MXFP8
            ``kda_w_in`` and ``kda_w_in_scale`` (``tkg_mx_quant``) from the MXFP8 linears of the loader
            (``weights[name]`` uint8 and ``weights[name + "_mx_scale"]``, ``mx_weights_torch``) or from BF16 ones.
            ``{"kda_w_out": "hT_quad"}`` builds the MXFP8 padded-lane ``kda_w_out`` and ``kda_w_out_scale``
            (``tkg_kda.kda_w_out_form``) from the BF16 ``o_proj``.
        w_in_head_major: False (default) keeps the original column order. True gives the MXFP8 ``kda_w_in`` and its
            scales in the head-major order of ``kda_verify_attention`` ``head_pipeline`` "heads"
            (``kda_w_in_head_major``); needs ``mx_weights`` ``kda_w_in``.

    Returns:
        A dict from each name of ``KDA_WEIGHT_NAMES`` to a contiguous numpy array, in the layouts of
        ``tkg_kda``, and ``kda_w_in_scale`` with an MXFP8 ``kda_w_in``.

    Example:
        >>> rank_weights = build_kda_rank_weights(weights, tp_size=32, tp_rank=5)
        >>> rank_weights["kda_w_in"].shape
        (7168, 1672)
    """
    heads = weights["a_log"].shape[0]
    if heads % tp_size:
        raise ValueError(f"{heads} KDA heads do not divide by TP {tp_size}. Use a divisor of {heads}.")
    local_heads = heads // tp_size
    width = local_heads * HEAD_DIM
    c0 = tp_rank * width
    h0 = tp_rank * local_heads
    channels = slice(c0, c0 + width)
    hidden = weights["q_proj"].shape[1]

    cols = kda_in_columns(local_heads)
    mx = _mx_classes(mx_weights)
    in_parts = [(name, cols[start], channels) for name, start in _IN_PROJ_PARTS]
    in_parts += [("f_a_proj", cols["f_a"], slice(None)), ("b_proj", cols["b"], slice(h0, h0 + local_heads))]
    if "kda_w_in" in mx:
        w_in_mx = _mx_w_in(weights, in_parts, hidden, cols["width"], mx["kda_w_in"])
    else:
        w_in = torch.zeros(hidden, cols["width"], dtype=torch.bfloat16)
        for name, start, rows in in_parts:
            part = weights[name][rows]
            w_in[:, start : start + part.shape[0]] = part.T

    w_out = weights["o_proj"][:, channels].T.contiguous()

    small = kda_small_columns(local_heads, conv_dtype)
    w_small = torch.zeros(HEAD_DIM, small["small_width"], dtype=torch.bfloat16)
    w_small[:, small["f_b"] : small["f_b"] + width] = weights["f_b_proj"][channels].T
    for p, name in enumerate(_CONV_NAMES):
        # [W, 1, 4] -> [heads, 128, 4] -> [128 d, heads, 4]
        conv = weights[name][channels].reshape(local_heads, HEAD_DIM, CONV_TAPS).permute(1, 0, 2)
        start = small["conv"] + p * local_heads * CONV_TAPS
        w_small[:, start : start + local_heads * CONV_TAPS] = conv.reshape(HEAD_DIM, -1).to(torch.bfloat16)

    params = torch.zeros(HEAD_DIM, small["params_width"] + (local_heads if a_exp_params else 0), dtype=torch.float32)
    params[:, small["dt_bias"] : small["dt_bias"] + local_heads] = (
        weights["dt_bias"][channels].float().reshape(local_heads, HEAD_DIM).T
    )
    params[:, small["a_log"] : small["a_log"] + local_heads] = weights["a_log"][h0 : h0 + local_heads].float()
    if a_exp_params:
        exp_fn = torch.exp if a_exp_params is True else a_exp_params
        params[:, small["a_exp"] : small["a_exp"] + local_heads] = exp_fn(
            weights["a_log"][h0 : h0 + local_heads].float()
        )
    params[:, small["o_norm"]] = weights["o_norm"].float()
    if conv_dtype == "fp32":
        for p, name in enumerate(_CONV_NAMES):
            conv = weights[name][channels].reshape(local_heads, HEAD_DIM, CONV_TAPS).permute(1, 0, 2)
            start = small["conv32"] + p * local_heads * CONV_TAPS
            params[:, start : start + local_heads * CONV_TAPS] = conv.reshape(HEAD_DIM, -1).float()

    arrays = {
        "kda_w_out": to_numpy_bf16(w_out),
        "kda_w_small": to_numpy_bf16(w_small),
        "kda_params": np.ascontiguousarray(params.numpy()),
    }
    if "kda_w_in" in mx:
        arrays["kda_w_in"], arrays["kda_w_in_scale"] = w_in_mx
    else:
        arrays["kda_w_in"] = to_numpy_bf16(w_in)
    if w_in_head_major:
        # The head-major column order of kda_verify_attention head_pipeline "heads" (an order of the MXFP8 kda_w_in).
        if "kda_w_in" not in mx:
            raise ValueError("w_in_head_major is an order of the MXFP8 kda_w_in: give mx_weights kda_w_in")
        arrays["kda_w_in"], arrays["kda_w_in_scale"] = kda_w_in_head_major(
            arrays["kda_w_in"], arrays["kda_w_in_scale"], local_heads
        )
    if "kda_w_out" in mx:
        arrays["kda_w_out"], arrays["kda_w_out_scale"] = mx_kda_w_out(w_out, mx["kda_w_out"])
    return {name: arrays[name] for name in KDA_WEIGHT_NAMES + _mx_outputs(mx)}


def mx_kda_w_out(w_out, order: str = "hT_quad") -> tuple[np.ndarray, np.ndarray]:
    """Return the MXFP8 padded-lane ``kda_w_out`` ``[1, 128, 4 H]`` and its scales ``[128, 1, H]`` from ``[128 Hl, H]``.

    Lane ``q`` holds the rows of local head ``q``; the rows past the heads are zero, so their lanes add nothing.
    """
    width, hidden = w_out.shape
    padded = np.zeros((MX_LANES * HEAD_DIM, hidden), dtype=np.float32)
    padded[:width] = w_out.float().numpy() if isinstance(w_out, torch.Tensor) else np.asarray(w_out, np.float32)
    data, scale = mx_quantize(padded.T, order)
    return pack_mx_hT(data.T, scale.T)


_IN_PROJ_PARTS = (("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v"), ("g_proj", "g"))


def _mx_classes(mx_weights: dict) -> dict:
    """Check ``mx_weights`` against ``MX_WEIGHT_ORDERS``; return it (``{}`` for None)."""
    mx = dict(mx_weights or {})
    for name, order in mx.items():
        if order not in MX_WEIGHT_ORDERS.get(name, ()):
            raise ValueError(f"MXFP8 class {name!r} with order {order!r} is not in {MX_WEIGHT_ORDERS}")
    return mx


def _mx_outputs(mx: dict) -> tuple:
    return tuple(f"{name}_scale" for name in KDA_WEIGHT_NAMES if name in mx)


def _mx_w_in(weights: dict, parts: list, hidden: int, n_cols: int, order: str) -> tuple:
    """Return the packed MXFP8 ``kda_w_in`` and its scales from the MXFP8 (or BF16) input linears."""
    data = np.zeros((hidden, n_cols), dtype=np.uint8)
    scale = np.zeros((hidden // MX_BLOCK, n_cols), dtype=np.uint8)
    for name, start, rows in parts:
        part_data, part_scale = mx_rows(weights, name, rows, order)
        data[:, start : start + part_data.shape[0]] = part_data.T
        scale[:, start : start + part_scale.shape[0]] = part_scale.T
    return pack_mx_hT(data, scale)


def build_kda_rank_weights_tp128(
    weights: dict, rank: int, tp_size: int = 128, conv_dtype: str = "bf16", mx_weights: dict = None
) -> dict:
    """Build one rank's kernel weights of the TP128 layout, with pad ranks above the head count.

    The 96 KDA heads are padded to the 128 ranks of the DP1 world: rank ``g < 96`` owns head ``g``
    with one local head, and ranks ``96 .. 127`` run the same body with zero weights. A pad rank's
    output projection is zero, so its all-reduce partial is zero: the world all-reduce reproduces the
    96-head reference. A pad rank reads and writes page 0 (the null page), so it needs no state page.

    Args:
        weights: Global weights, see the module docstring. ``a_log`` carries the head count.
        rank: World rank, 0 to ``tp_size - 1``.
        tp_size: TP degree (128 for the DP1 world).
        conv_dtype: Form of the short convolution, see ``build_kda_rank_weights``.
        mx_weights: MXFP8 classes, see ``build_kda_rank_weights``.

    Returns:
        A dict from each name of ``KDA_WEIGHT_NAMES`` to a contiguous numpy array, in the 1-head
        layouts of ``tkg_kda``. Rank ``g < heads`` holds head ``g``; a pad rank holds zeros.

    Example:
        >>> rank_weights = build_kda_rank_weights_tp128(weights, rank=100)
        >>> rank_weights["kda_w_out"].shape
        (128, 7168)
    """
    heads = weights["a_log"].shape[0]
    if rank < heads:
        return build_kda_rank_weights(weights, heads, rank, conv_dtype, mx_weights)
    mx = _mx_classes(mx_weights)
    hidden = weights["q_proj"].shape[1]
    cols = kda_in_columns(1)
    small = kda_small_columns(1, conv_dtype)
    arrays = {
        "kda_w_in": np.zeros((hidden, cols["width"]), dtype=ml_dtypes.bfloat16),
        "kda_w_out": np.zeros((HEAD_DIM, hidden), dtype=ml_dtypes.bfloat16),
        "kda_w_small": np.zeros((HEAD_DIM, small["small_width"]), dtype=ml_dtypes.bfloat16),
        "kda_params": np.zeros((HEAD_DIM, small["params_width"]), dtype=np.float32),
    }
    if "kda_w_in" in mx:
        arrays["kda_w_in"], arrays["kda_w_in_scale"] = pack_mx_hT(
            np.zeros((hidden, cols["width"]), dtype=np.uint8),
            np.zeros((hidden // MX_BLOCK, cols["width"]), dtype=np.uint8),
        )
    if "kda_w_out" in mx:
        arrays["kda_w_out"], arrays["kda_w_out_scale"] = mx_kda_w_out(np.zeros((HEAD_DIM, hidden), np.float32))
    return {name: arrays[name] for name in KDA_WEIGHT_NAMES + _mx_outputs(mx)}


def kda_w_in_head_major(w_in: np.ndarray, w_in_scale: np.ndarray, local_heads: int, inverse: bool = False) -> tuple:
    """Return the MXFP8 ``kda_w_in`` and its scales in the head-major column order (``kda_in_columns_head_major``).

    ``w_in`` is the hT_quad data ``[G, 128, 4 N]`` (4 bytes a column) and ``w_in_scale`` the scales ``[128, S, N]``
    (``tkg_mx_quant``). The same column permutation applies to both, so each output block keeps its weights and scales and
    the projection keeps its bits (``kda_verify_attention`` ``head_pipeline`` ``heads`` reads this order). ``inverse``
    returns the original order from the head-major one. The arrays keep their dtype and shape.
    """
    perm = np.asarray(kda_in_head_major_permutation(local_heads))
    n_cols = perm.size
    if w_in.shape[-1] != 4 * n_cols or w_in_scale.shape[-1] != n_cols:
        raise ValueError(
            f"kda_w_in {w_in.shape} / kda_w_in_scale {w_in_scale.shape} do not match the {n_cols}-column layout"
        )
    if inverse:
        perm = np.argsort(perm)
    data = np.ascontiguousarray(w_in.reshape(*w_in.shape[:-1], n_cols, 4)[..., perm, :].reshape(w_in.shape))
    scale = np.ascontiguousarray(w_in_scale[..., perm])
    return data, scale


def write_kda_rank_page(
    page: np.ndarray,
    recurrent_state: torch.Tensor,
    conv_state: torch.Tensor,
    tp_size: int,
    tp_rank: int,
    state_dtype: str = "bf16",
) -> np.ndarray:
    """Write the KDA states of one TP rank into one cache page.

    Args:
        page: bf16 ``[PAGE_WIDTH]`` page (at least ``kda_page_offsets(Hl, state_dtype)["end"]`` elements). The
            function does not change the other elements.
        recurrent_state: Global ``[heads, 128, 128]``, bf16 (or any float dtype for an fp32 ``state_dtype``).
        conv_state: Global bf16 ``[3, W, 3]``.
        tp_size: TP degree.
        tp_rank: TP rank.
        state_dtype: Form of the recurrent state (``tkg_kda.KDA_STATE_DTYPES``). The fp32 forms write the fp32
            bits of the state, two bf16 elements per value (little-endian, as the kernel's fp32 view reads them).

    Returns:
        A new bf16 page with the rank's conv state and recurrent state.

    Example:
        >>> page = write_kda_rank_page(np.zeros(73728, ml_dtypes.bfloat16), rec, conv, 32, 0)
    """
    heads = recurrent_state.shape[0]
    local_heads = heads // tp_size
    width = local_heads * HEAD_DIM
    offsets = kda_page_offsets(local_heads, state_dtype)
    out = np.array(page, copy=True).reshape(-1)
    conv = conv_state[:, tp_rank * width : (tp_rank + 1) * width, :]
    rec = recurrent_state[tp_rank * local_heads : (tp_rank + 1) * local_heads]
    out[offsets["conv"] : offsets["recurrent"]] = to_numpy_bf16(conv).reshape(-1)
    if state_dtype == "bf16":
        out[offsets["recurrent"] : offsets["end"]] = to_numpy_bf16(rec).reshape(-1)
    else:
        bits = np.ascontiguousarray(rec.detach().float().numpy().reshape(-1)).view(np.uint16)
        out[offsets["recurrent"] : offsets["end"]] = bits.view(ml_dtypes.bfloat16)
    return out


def read_kda_rank_state(page: np.ndarray, local_heads: int, state_dtype: str = "bf16") -> torch.Tensor:
    """Return the recurrent state ``[Hl, 128, 128]`` of one rank's cache page (fp32 values of either form).

    Example:
        >>> state = read_kda_rank_state(page, 3, "fp32")
    """
    offsets = kda_page_offsets(local_heads, state_dtype)
    values = np.ascontiguousarray(np.asarray(page).reshape(-1)[offsets["recurrent"] : offsets["end"]])
    if state_dtype == "bf16":
        values = values.astype(ml_dtypes.bfloat16).astype(np.float32)
    else:
        values = values.astype(ml_dtypes.bfloat16).view(np.uint16).view(np.float32)
    return torch.from_numpy(values.copy()).reshape(local_heads, HEAD_DIM, HEAD_DIM)
