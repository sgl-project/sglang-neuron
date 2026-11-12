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

"""Torch reference and host layouts of the Kimi-K3 MLA decode attention body.

The reference is global: all 96 heads, the full-context cache, and one token. The host functions
build the kernel weights and the paged cache of one TP32 rank from the global tensors.
"""

from typing import Sequence

import ml_dtypes
import numpy as np
import torch

from .mx_weights_torch import mx_rows, pack_mx_hT, pack_mx_hT_compact, pack_mx_quadrants

HIDDEN = 7168
TP = 32
HEADS = 96
LOCAL_HEADS = HEADS // TP
Q_LORA = 1536
KV_LORA = 512
NOPE = 128
ROPE = 64
V_HEAD = 128
Q_HEAD = NOPE + ROPE
CACHE_WIDTH = KV_LORA + ROPE
BLOCK = 128
LOCAL_BLOCKS = 4
PAGE_WIDTH = BLOCK * CACHE_WIDTH
LORA_NORM_EPS = 1e-5
SOFTMAX_SCALE = Q_HEAD**-0.5
FP8_E4M3FN_MAX = 448.0
"""Largest finite float8_e4m3fn value."""

GLOBAL_WEIGHT_SHAPES = {
    "q_a_proj": (Q_LORA, HIDDEN),
    "q_a_layernorm": (Q_LORA,),
    "q_b_proj": (HEADS * Q_HEAD, Q_LORA),
    "kv_a_proj": (CACHE_WIDTH, HIDDEN),
    "kv_a_layernorm": (KV_LORA,),
    "kv_b_proj": (HEADS * (NOPE + V_HEAD), KV_LORA),
    "g_proj": (HEADS * V_HEAD, HIDDEN),
    "o_proj": (HIDDEN, HEADS * V_HEAD),
}
"""Global weights in the serving (``weight[out, in]``) orientation, keyed by projection name."""

TensorLike = torch.Tensor | np.ndarray

MX_WEIGHT_CLASSES = ("mla_in_proj", "mla_q_b_proj", "mla_kv_b_proj", "mla_o_proj")
"""MXFP8 weight classes of ``mla_decode_rank_weights`` (``mx_weights``). A loader reads the class names: a tuple, not a map, so the
loader keeps its own list of the input weights of each class."""
MX_WEIGHT_ORDERS = {
    "mla_in_proj": ("hT_quad",),
    "mla_q_b_proj": ("hT_quad",),
    "mla_kv_b_proj": ("hT_quad",),
    "mla_o_proj": ("hT_quad",),
}
"""The group orders of each class of ``MX_WEIGHT_CLASSES``."""
MX_FROM_BF16 = ("mla_in_proj", "mla_q_b_proj", "mla_kv_b_proj", "mla_o_proj")
"""Classes quantized from the BF16 checkpoint at load (no MXFP8 checkpoint). A loader with no ``_mx_scale`` tensor of a
class in this tuple quantizes the BF16 weight here (``mx_rows``)."""
_IN_PROJ_NAMES = ("q_a_proj", "kv_a_proj", "g_proj")


def _to_torch(value: TensorLike) -> torch.Tensor:
    """Return ``value`` as a torch tensor and keep BF16 and FP8 values exact."""
    if isinstance(value, torch.Tensor):
        return value
    if value.dtype == ml_dtypes.bfloat16:
        return torch.from_numpy(np.asarray(value, dtype=np.float32)).to(torch.bfloat16)
    if value.dtype == ml_dtypes.float8_e4m3fn:
        return torch.from_numpy(np.asarray(value, dtype=np.float32))
    return torch.from_numpy(np.ascontiguousarray(value))


def quantize_fp8_e4m3fn(value: TensorLike) -> np.ndarray:
    """Return ``value`` as the float8_e4m3fn cache rows that the cache write stores, with a scale of 1.0.

    The values are clamped to +-448 and rounded to the nearest even float8_e4m3fn value. A cast without
    the clamp gives NaN from 464 on.

    Example:
        >>> quantize_fp8_e4m3fn(np.array([1.1, 500.0], dtype=np.float32)).astype(np.float32)
        array([  1.125, 448.   ], dtype=float32)
    """
    clamped = _to_torch(value).float().clamp(-FP8_E4M3FN_MAX, FP8_E4M3FN_MAX)
    raw = clamped.to(torch.float8_e4m3fn).view(torch.uint8).numpy()
    return np.ascontiguousarray(raw).view(ml_dtypes.float8_e4m3fn)


def _to_cache_dtype(value: TensorLike, dtype) -> np.ndarray:
    """Return ``value`` as cache rows of ``dtype`` (BF16, or FP8 as :func:`quantize_fp8_e4m3fn` writes it)."""
    if np.dtype(dtype) == np.dtype(ml_dtypes.float8_e4m3fn):
        return quantize_fp8_e4m3fn(value)
    return _to_bf16_numpy(value)


def _to_bf16_numpy(value: TensorLike) -> np.ndarray:
    """Return ``value`` as a contiguous BF16 numpy array."""
    if isinstance(value, torch.Tensor):
        value = value.detach().float().numpy()
    return np.ascontiguousarray(np.asarray(value).astype(ml_dtypes.bfloat16))


def _bf16(x: torch.Tensor) -> torch.Tensor:
    """Round to BF16 and return FP32 values."""
    return x.to(torch.bfloat16).float()


def _rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Return ``bf16(bf16(x32 * rsqrt(mean(x32^2) + eps)) * w32)`` as FP32 values."""
    x32 = x.float()
    normed = _bf16(x32 * torch.rsqrt(x32.square().mean(dim=-1, keepdim=True) + eps))
    return _bf16(normed * weight.float())


def _linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Return the BF16-rounded ``x @ weight.T`` with an FP32 accumulation."""
    return _bf16(x.float() @ weight.float().T)


def to_hT_order(vector: np.ndarray) -> np.ndarray:
    """Return a ``[..., N]`` vector in hT order: element ``p * (N/128) + j`` is element ``j * 128 + p``.

    Example:
        >>> to_hT_order(np.arange(256.0))[:3]
        array([  0., 128.,   1.])
    """
    width = vector.shape[-1]
    lead = vector.shape[:-1]
    swapped = vector.reshape(*lead, width // BLOCK, BLOCK).swapaxes(-1, -2)
    return np.ascontiguousarray(swapped).reshape(*lead, width)


def mla_build_cache_row(
    kv_a: TensorLike, kv_a_layernorm: TensorLike, lora_norm_eps: float = LORA_NORM_EPS
) -> torch.Tensor:
    """Return the FP32 values of the BF16 cache rows ``[rms(latent) | k_pe]`` of ``[..., 576]`` projections,
    with the norm epsilon ``lora_norm_eps``.

    Example:
        >>> mla_build_cache_row(torch.ones(2, 576), torch.ones(512)).shape
        torch.Size([2, 576])
    """
    kv_a = _bf16(_to_torch(kv_a).float())
    latent = _rms(kv_a[..., :KV_LORA], _to_torch(kv_a_layernorm), lora_norm_eps)
    return torch.cat((latent, kv_a[..., KV_LORA:]), dim=-1)


def mla_decode_attention_torch_ref(
    x: TensorLike,
    weights: dict[str, TensorLike],
    cache: TensorLike,
    position: int,
    fp8_cache: bool = False,
    lora_norm_eps: float = LORA_NORM_EPS,
) -> dict[str, torch.Tensor]:
    """Return the global Kimi-K3 MLA decode attention output of one token.

    The function follows the serving decode step at the BF16 and FP32 boundaries of the TP32 path:
    the projections round to BF16, the norms use ``eps=lora_norm_eps`` (1e-5 by default), and the softmax
    and its merge are FP32.

    Args:
        x: ``[1, 7168]`` BF16 normalized hidden state.
        weights: global weights in the ``weight[out, in]`` orientation, see ``GLOBAL_WEIGHT_SHAPES``.
        cache: ``[S, 576]`` BF16 or FP8 cache rows ``[normalized latent | k_pe]`` by global position, with
            ``S > position``. The rows at ``position`` and later are not read.
        position: global position of the current token.
        fp8_cache: the cache is FP8, so the current row is stored and attended as
            :func:`quantize_fp8_e4m3fn` writes it.
        lora_norm_eps: the epsilon of the q_a and kv_a norms (``lora_norm_eps`` of the kernel).

    Returns:
        A dict with ``out`` (FP32 values of the BF16 ``[1, 7168]`` output, before the TP all-reduce
        rounding) and ``cache_row`` (FP32 values of the ``[1, 576]`` cache row of the current token, in
        the cache dtype).

    Example:
        >>> # out = mla_decode_attention_torch_ref(x, weights, cache, 4000)["out"]
    """
    w = {name: _to_torch(value).float() for name, value in weights.items()}
    x32 = _bf16(_to_torch(x).float().reshape(1, HIDDEN))

    q_a = _linear(x32, w["q_a_proj"])
    q_latent = _rms(q_a, w["q_a_layernorm"], lora_norm_eps)
    q = _linear(q_latent, w["q_b_proj"]).reshape(HEADS, Q_HEAD)
    q_nope, q_pe = q[:, :NOPE], q[:, NOPE:]
    kv_b = w["kv_b_proj"].reshape(HEADS, NOPE + V_HEAD, KV_LORA)
    q_abs = _bf16(torch.einsum("hd,hdc->hc", q_nope, kv_b[:, :NOPE, :]))

    kv_a = _linear(x32, w["kv_a_proj"])
    cache_row = mla_build_cache_row(kv_a, w["kv_a_layernorm"], lora_norm_eps)
    if fp8_cache:
        cache_row = torch.from_numpy(quantize_fp8_e4m3fn(cache_row).astype(np.float32))

    prior = _to_torch(cache).float()[:position]
    keys = torch.cat((prior, cache_row), dim=0)
    scores = (q_abs @ keys[:, :KV_LORA].T + q_pe @ keys[:, KV_LORA:].T) * SOFTMAX_SCALE
    probs = torch.softmax(scores, dim=-1)
    latent = _bf16(probs @ keys[:, :KV_LORA])

    value = _bf16(torch.einsum("hc,hvc->hv", latent, kv_b[:, NOPE:, :]))
    gate = _linear(x32, w["g_proj"]).reshape(HEADS, V_HEAD)
    o = _bf16(value * _bf16(torch.sigmoid(gate)))
    out = o.reshape(1, HEADS * V_HEAD) @ w["o_proj"].T
    return {"out": out, "cache_row": cache_row}


def mla_decode_rank_weights(
    weights: dict[str, TensorLike], tp_rank: int, mx_weights: dict = None
) -> dict[str, np.ndarray]:
    """Build the kernel weights of one TP32 rank from the global ``weight[out, in]`` weights.

    The result has the names and layouts of ``MLA_WEIGHT_NAMES`` in ``tkg_mla_paged.py``. Rank ``t`` owns the
    heads ``3t .. 3t+2`` and the q_a columns ``48t ..``. Its kv_a column ``c`` is kv_a feature ``32c + t``.
    ``mx_weights`` ``{"mla_in_proj": "hT_quad"}`` builds the MXFP8 ``mla_in_proj`` and ``mla_in_proj_scale``
    (``tkg_mx_quant``) from the MXFP8 linears of the loader (``weights[name]`` uint8 and ``weights[name + "_mx_scale"]``)
    or from BF16 ones. ``{"mla_q_b_proj": "hT_quad"}`` builds the MXFP8 ``mla_q_b_proj`` and ``mla_q_b_proj_scale`` the
    same way (the rank q_b ``[1536, 576]`` is 3 clean hT_quad groups). ``{"mla_o_proj": "hT_quad"}`` builds the
    padded-lane MXFP8 ``mla_o_proj`` ``[1, 128, 4 * 7168]`` (heads in lanes 0 .. 2, lane 3 zero) with the compact
    ``mla_o_proj_scale`` ``[16, 7168]`` (``pack_mx_hT_compact``). ``{"mla_kv_b_proj": "hT_quad"}`` builds the MXFP8
    ``mla_kv_b_proj`` ``[128, 4 * (512 + 384)]``: the absorb part in quadrant row tiles (head ``h`` at partitions
    ``32 h ..``, ``pack_mx_quadrants``), then the value part as one hT_quad group, and the compact
    ``mla_kv_b_proj_scale`` ``[16, 512 + 384]``. The three are quantized from BF16. A class not in ``mx_weights``
    stays BF16.

    Example:
        >>> # rank_weights = mla_decode_rank_weights(global_weights, tp_rank=5)
    """
    t = tp_rank
    mx = dict(mx_weights or {})
    for name, order in mx.items():
        if order not in MX_WEIGHT_ORDERS.get(name, ()):
            raise ValueError(f"MXFP8 class {name!r} with order {order!r} is not in {MX_WEIGHT_ORDERS}")
    skip = set(_IN_PROJ_NAMES) if "mla_in_proj" in mx else set()
    w = {
        name: np.asarray(_to_bf16_numpy(value))
        for name, value in weights.items()
        if name not in skip and not name.endswith("_mx_scale")
    }
    q_lora_local = Q_LORA // TP
    cache_local = CACHE_WIDTH // TP
    v_local = LOCAL_HEADS * V_HEAD
    kv_a_rows = [c * TP + t for c in range(cache_local)]
    in_rows = {
        "q_a_proj": slice(t * q_lora_local, (t + 1) * q_lora_local),
        "kv_a_proj": kv_a_rows,
        "g_proj": slice(t * v_local, (t + 1) * v_local),
    }
    if "mla_in_proj" in mx:
        parts = [mx_rows(weights, name, in_rows[name], mx["mla_in_proj"]) for name in _IN_PROJ_NAMES]
        in_mx = pack_mx_hT(
            np.concatenate([data for data, _ in parts], axis=0).T,
            np.concatenate([scale for _, scale in parts], axis=0).T,
        )
    else:
        in_mx = None

    in_proj = (
        None
        if in_mx is not None
        else np.concatenate(
            (
                w["q_a_proj"][in_rows["q_a_proj"]],
                w["kv_a_proj"][in_rows["kv_a_proj"]],
                w["g_proj"][in_rows["g_proj"]],
            ),
            axis=0,
        ).T
    )
    if in_proj is not None:
        # SBUF order: row p holds the qkv columns of the input rows 128 j + p for every j, then their gate
        # columns, so each partition of a weight DMA reads long contiguous runs.
        qkv_local = q_lora_local + cache_local
        in_tiles = in_proj.reshape(HIDDEN // BLOCK, BLOCK, in_proj.shape[1]).transpose(1, 0, 2)
        in_proj = np.concatenate(
            (in_tiles[:, :, :qkv_local].reshape(BLOCK, -1), in_tiles[:, :, qkv_local:].reshape(BLOCK, -1)), axis=1
        )

    q_b_rows = slice(t * LOCAL_HEADS * Q_HEAD, (t + 1) * LOCAL_HEADS * Q_HEAD)
    if "mla_q_b_proj" in mx:
        # Rank q_b ``[1536, 576]`` (in, out): the 1536 input rows are 3 clean hT_quad groups, no padding. The same
        # host quantize and pack as ``mla_in_proj``; the weight is the moving operand of ``nc_matmul_mx`` in the body.
        data, scale = mx_rows(weights, "q_b_proj", q_b_rows, mx["mla_q_b_proj"])
        q_b_mx = pack_mx_hT(data.T, scale.T)
        q_b = q_b_mx[0]
    else:
        q_b = w["q_b_proj"][q_b_rows].T
        q_b = q_b.reshape(Q_LORA // BLOCK, BLOCK, q_b.shape[1]).transpose(1, 0, 2).reshape(BLOCK, -1)

    kv_b = w["kv_b_proj"][t * LOCAL_HEADS * (NOPE + V_HEAD) : (t + 1) * LOCAL_HEADS * (NOPE + V_HEAD)]
    kv_b = kv_b.reshape(LOCAL_HEADS, NOPE + V_HEAD, KV_LORA)
    # Absorb part: element (d, h * 512 + c) is key[h, d, c].
    absorb = kv_b[:, :NOPE, :].transpose(1, 0, 2).reshape(NOPE, LOCAL_HEADS * KV_LORA)
    # Value part: element (p, (k * 3 + h) * 128 + v) is value[h, v, k * 128 + p].
    value = kv_b[:, NOPE:, :].reshape(LOCAL_HEADS, V_HEAD, KV_LORA // BLOCK, BLOCK)
    value = value.transpose(3, 2, 0, 1).reshape(BLOCK, KV_LORA // BLOCK * LOCAL_HEADS * V_HEAD)

    o_proj = w["o_proj"][:, t * v_local : (t + 1) * v_local].T
    kv_b_mx = o_mx = None
    if "mla_kv_b_proj" in mx:
        # Absorb part: head h ``key[h]`` ``[128 (nope), 512]`` in quadrant h, the weight moving; value part: the
        # ``[512, 384]`` latent-by-(head, dim) rows as one hT_quad group, the weight stationary.
        key = np.asarray(kv_b[:, :NOPE, :], np.float32).transpose(0, 2, 1)
        absorb_mx = pack_mx_quadrants(key)
        value_in_out = np.asarray(kv_b[:, NOPE:, :], np.float32).reshape(v_local, KV_LORA).T
        value_mx = pack_mx_hT_compact(value_in_out, mx["mla_kv_b_proj"])
        kv_b_mx = (
            np.concatenate((absorb_mx[0], value_mx[0][0]), axis=1),
            np.concatenate((absorb_mx[1], value_mx[1]), axis=1),
        )
    if "mla_o_proj" in mx:
        # The local o_proj rows ``[384, 7168]`` (in, out) as one padded-lane group: lane h holds head h, lane 3 zero.
        o_mx = pack_mx_hT_compact(np.asarray(o_proj, np.float32), mx["mla_o_proj"])

    out = {
        "mla_in_proj": np.ascontiguousarray(in_proj if in_mx is None else in_mx[0]),
        "mla_q_a_norm": to_hT_order(w["q_a_layernorm"]),
        "mla_q_b_proj": np.ascontiguousarray(q_b),
        "mla_kv_a_norm": np.ascontiguousarray(w["kv_a_layernorm"]),
        "mla_kv_b_proj": np.ascontiguousarray(
            np.concatenate((absorb, value), axis=1) if kv_b_mx is None else kv_b_mx[0]
        ),
        "mla_o_proj": np.ascontiguousarray(o_proj if o_mx is None else o_mx[0]),
    }
    if in_mx is not None:
        out["mla_in_proj_scale"] = in_mx[1]
    if "mla_q_b_proj" in mx:
        out["mla_q_b_proj_scale"] = q_b_mx[1]
    if kv_b_mx is not None:
        out["mla_kv_b_proj_scale"] = np.ascontiguousarray(kv_b_mx[1])
    if o_mx is not None:
        out["mla_o_proj_scale"] = o_mx[1]
    return out


def mla_decode_rank_cache(
    cache: np.ndarray,
    position: int,
    tp_rank: int,
    block_pages: Sequence[int],
    pool: np.ndarray,
    copy: bool = True,
    cp: int = TP,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lay out the paged cache of one CP rank from a global cache.

    Local block ``j`` of rank ``t`` holds global block ``cp j + t``, so its 128 rows are the global rows
    ``(cp j + t) * 128 ..``. The function copies each local block into the page ``block_pages[j]``. A page of
    0 is the null page: the block is not allocated and the function copies nothing.

    Args:
        cache: ``[S, 576]`` global cache rows by position, in the pool dtype. ``S`` covers every
            allocated block.
        position: global position of the current token.
        tp_rank: CP rank ``t``.
        block_pages: page of each local block, 0 for an unallocated block.
        pool: ``[P, 73728]`` BF16 or FP8 page pool.
        copy: write a copy of ``pool`` (default). With ``False`` the function writes ``pool`` in place, so
            the sequences of a batch can share one pool.
        cp: CP ranks (32, or 128 for one replica on all ranks).

    Returns:
        ``(pool, block_table, slot)``: the pool with the local blocks written, the ``[1, NB]`` int32
        block table, and the ``[1, 1]`` int32 scheduler slot of the current row (0 on a non-owner rank).

    Example:
        >>> # pool, table, slot = mla_decode_rank_cache(cache, 4000, 31, [3, 0, 0, 0], pool)
    """
    if copy:
        pool = pool.copy()
    rows = pool.reshape(pool.shape[0] * BLOCK, CACHE_WIDTH)
    for local_block, page in enumerate(block_pages):
        if page == 0:
            continue
        start = (local_block * cp + tp_rank) * BLOCK
        rows[page * BLOCK : (page + 1) * BLOCK] = cache[start : start + BLOCK]
    owner = (position // BLOCK) % cp
    slot = 0
    if owner == tp_rank:
        slot = block_pages[position // (BLOCK * cp)] * BLOCK + position % BLOCK
    block_table = np.asarray([list(block_pages)], dtype=np.int32)
    return pool, block_table, np.asarray([[slot]], dtype=np.int32)


def mla_decode_write_row(pool: np.ndarray, slot: np.ndarray, cache_row: TensorLike) -> np.ndarray:
    """Return a copy of ``pool`` with the current row at the scheduler slot, as the owner rank writes it.

    A slot in the null page 0 is not written, so a non-owner rank (slot 0) returns the pool unchanged. An
    FP8 pool gets the row as :func:`quantize_fp8_e4m3fn` writes it.

    Example:
        >>> # expected_pool = mla_decode_write_row(pool, slot, ref["cache_row"])
    """
    pool = pool.copy()
    row_index = int(np.asarray(slot).reshape(-1)[0])
    if row_index >= BLOCK:
        rows = pool.reshape(pool.shape[0] * BLOCK, CACHE_WIDTH)
        rows[row_index] = _to_cache_dtype(cache_row, pool.dtype).reshape(CACHE_WIDTH)
    return pool
