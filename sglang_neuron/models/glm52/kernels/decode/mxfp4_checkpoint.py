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

"""Convert an MXFP4 checkpoint's MoE expert weights into this kernel's MX layouts.

The kernel owns its weight layout, so this converter ships with the kernel: the weight store
(``models/glm52/layout/decode/weight_loaders_mxfp4.py``) has to produce exactly these layouts, and a second
implementation of the rules below would be a second place for them to be wrong.

THE PACKING IS PLAIN ``(2i, 2i+1)``, AND THE CONFIG FIELD THAT SAYS OTHERWISE IS MISLEADING
------------------------------------------------------------------------------------------
Element ``k`` lives in byte ``k // 2``, low nibble first. That is the naive reading, and it is right.

The checkpoint's ``quantization_config`` says otherwise. It advertises ``quant_method: quark`` with
``export.pack_method: "reorder"``, and quark's 4-bit packer does have a reorder mode that puts eight
values in one int32 as ``[0, 2, 4, 6, 1, 3, 5, 7]``. These bytes are not packed that way, and two
checks that look like proof cannot tell the orders apart: unpacking with quark's own reorder inverse
matches a reorder implementation bit for bit (both sides assume reorder), and a smoothness test over
four schemes gives mean|diff|/std within 0.002 of each other (an MLP's input channels have no local
ordering).

A reference settles it. Scoring each candidate by the layer-6 MoE output it produces against a
captured GPU reference decode step, with every other input pinned (the bf16 dense MLP of layer 0
reproduces the same capture at 3.7e-3), gives:

    plain (2i, 2i+1)     cosine +0.9932415   rel_l2 0.117   ||ours|| 0.1127 vs 0.1151
    quark reorder        cosine +0.0987080   rel_l2 1.019   ||ours|| 0.0366
    reorder inverse      cosine +0.1061782   rel_l2 1.015
    lo4|hi4 per word     cosine +0.1061782   rel_l2 1.015
    hi4|lo4 per word     cosine +0.0158634   rel_l2 1.036
    byte-half swapped    cosine -0.0122275   rel_l2 1.045

``pack_method: "reorder"`` appears to be inherited boilerplate on the ``weight_format:
real_quantized`` MX export path rather than a description of these bytes.

Why the earlier levels did not catch it: levels 1 and 2 compare a converted weight against
``dequant_mxfp4`` of the *same bytes*, so an error in the shared unpack cancels on both sides, and level
3 compares the kernel against a reference that does the same. They validated layout plumbing, which is
what they were for; none of them decoded the checkpoint against anything external.

WHICH CONTRACTION ELEMENT EACH SLOT HOLDS, MEASURED RATHER THAN ASSUMED
----------------------------------------------------------------------
The MX matmul pairs the two operands by their *slot* ``(partition, tile, x4_nibble)``; the label attached
to a slot is whatever the producer of the other operand chose. So the weight's mapping is dictated by the
ACTIVATION, and the activation's was recovered by running the MoE reference against a plain numpy MLP
under every candidate (9 combinations, one expert, one token, affinity 1, synthetic weights so neither
side has any quantization difference):

    gate/up   slot (p, kt, n) -> h = n*(H/4) + kt*128 + p     cosine 0.9991 -- everything else <= 0.22
    down      slot (p, ti, n) -> i = ti*512 + n*128 + p

The first is set by ``rmsnorm_mx_quantize``'s swizzle (``reshape(T, 4, H/512, 128).transpose(3,2,0,1)``,
mirrored in ``_moe_tkg_mx_ref``); the second by ``gate_up_proj_mx_torch_ref``'s output write
(``cur.reshape(4, 128, BxS).permute(1, 2, 0)``). Note this contradicts ``down_proj_mx_torch_ref``'s
docstring, which claims ``512*tile + 4p + q``: that ordering scores cosine 0.045 and is stale.

It also differs from the production GPT-OSS MXFP4 loader, whose weights use ``h = kt*512 + 4p + n``. That
is not a disagreement about hardware -- GPT-OSS reaches its convention by PRE-SHUFFLING the activation in
HBM (``[T, 4_H, H/512, 16_H, 8_H]``), a step GLM-5.2 does not do.

LOSSLESSNESS IS A PROPERTY OF THE SCALE GROUPS, NOT OF THE ELEMENT PLACEMENT
---------------------------------------------------------------------------
Any bijection places every code somewhere, so element placement alone can never lose information. What
can is the E8M0 exponent: a kernel scale group is ``{fixed tile, fixed j = p//8, all 8 r = p%8, all 4 n}``
-- 32 elements -- and the checkpoint's exponent transfers unchanged only if that set is exactly one
checkpoint 32-block. Under both mappings above it is not; it is **4 runs of 8**, so the group straddles 4
checkpoint blocks and the three sub-blocks with smaller exponents lose mantissa bits. Measured on layer 6:

    gate 0.0300   up 0.0296   down 0.0277     (relative L2 over 12,582,912 elements each)

:data:`TAU` fixes it. ``k = tile*512 + (p//8)*32 + n*8 + (p%8)`` makes a group 32 CONSECUTIVE indices, so
one checkpoint block lands in one kernel group -- verified loss exactly 0.0. It is still just a
permutation of the contraction axis, so it costs nothing at run time; what it costs is agreement with the
producer of the other operand:

* **down** -- its contraction axis is the intermediate, which OUR gate/up kernel produces, and gate/up's
  free axis carries no scale group at all. So :func:`intermediate_permutation` reorders gate/up's ``I``
  axis and down becomes lossless with NO kernel change. This loader always does that.
* **gate/up** -- its contraction axis is the hidden, which ``rmsnorm_mx_quantize`` produces. Going lossless
  needs that swizzle changed to TAU (``lossless_hidden=True`` here). Both orders are 4-element strided
  gathers per partition, so the change should be layout-neutral, but it is a kernel change and is
  therefore opt-in rather than the default.

End-to-end on the MoE reference with real layer-6 weights, versus a numpy MLP built from the exact
dequantization (16 tokens, top_k 2):

    default kernel                      cosine 0.9979596  rel_l2 0.0641
    default kernel + intermediate perm  cosine 0.9983376  rel_l2 0.0578
    + TAU hidden (fully lossless)       cosine 0.9990382  rel_l2 0.0438

The 0.0438 floor is not the weights: it is the down projection quantizing the intermediate to mxfp8, and a
synthetic control with no weight quantization difference at all sits at 0.0427.
"""

from __future__ import annotations

import nki.language as nl
import numpy as np

from .utils.mx_torch_common import quantize_to_mx

_PMAX = 128
_Q_WIDTH = 4
_K_TILE = _PMAX * _Q_WIDTH  # 512 contraction elements per tile
_MX_BLOCK = 32
_SCALE_P = _K_TILE // _MX_BLOCK  # 16 scale rows per 512-element contraction tile
_Q_HEIGHT = 8  # partitions per MX scale group
_E8M0_BIAS = 127

QUARK_REORDER = (0, 2, 4, 6, 1, 3, 5, 7)
"""Quark's ``reorder`` map: what the checkpoint's config advertises and these bytes do not use (the module docstring)."""

FP4_E2M1_LUT = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)
"""E2M1 code -> value. Codes 8..15 are the negatives of 0..7."""


# =============================================================================
# Slot -> contraction-index mappings
# =============================================================================


def glm_hidden(p, tile: int, n: int, k_dim: int):
    """What ``rmsnorm_mx_quantize`` puts in gate/up slot ``(p, tile, n)``: ``n*(K/4) + tile*128 + p``."""
    return n * (k_dim // _Q_WIDTH) + tile * _PMAX + p


def tau(p, tile: int, n: int, k_dim: int):
    """The alignment-preserving relabeling: ``tile*512 + (p//8)*32 + n*8 + (p % 8)``.

    A scale group (fixed ``tile``, fixed ``j = p//8``) becomes ``[tile*512 + 32j, +32)`` -- 32 consecutive
    indices, i.e. exactly checkpoint block ``tile*16 + j``.
    """
    return tile * _K_TILE + (p // _Q_HEIGHT) * _MX_BLOCK + n * _Q_HEIGHT + (p % _Q_HEIGHT)


def groups_align(k_dim: int, mapping) -> bool:
    """Does every kernel scale group under ``mapping`` coincide with one checkpoint 32-block?"""
    rows = rows_per_tile(k_dim)
    p = np.arange(rows)
    for tile in range(max(1, k_dim // _K_TILE)):
        for j in range(rows // _Q_HEIGHT):
            ks = np.concatenate([mapping(p[_Q_HEIGHT * j : _Q_HEIGHT * (j + 1)], tile, n, k_dim) for n in range(4)])
            if np.unique(ks // _MX_BLOCK).size != 1:
                return False
    return True


def rows_per_tile(k_dim: int) -> int:
    """Partitions one contraction tile occupies: 128 when the tile is full, ``k_dim / 4`` when narrower.

    The shared expert's ``I_sh`` is 256, so its down projection contracts over a PARTIAL tile -- 64
    partitions, 8 scale rows -- and both the layout and the intermediate ordering change with it.
    """
    return min(_PMAX, k_dim // _Q_WIDTH)


def intermediate_permutation(i_dim: int) -> np.ndarray:
    """Reorder for gate/up's ``I`` free axis so the down projection can use :func:`tau` losslessly.

    gate/up free slot ``l = ti*512 + n*rows + p`` -- the position its output write lands in, and hence the
    position down reads as its contraction slot ``(p, ti, n)`` -- must carry checkpoint row
    ``tau(p, ti, n)``. ``rows`` is :func:`rows_per_tile`, because the reference writes
    ``cur.reshape(4, rows_padded // 4, BxS).permute(1, 2, 0)`` and ``rows_padded`` is the tile's real
    width. Applied to the gate and up weights, their scales and their bias; ``I`` is internal to the MLP
    and the activation is elementwise in it, so any permutation is free.
    """
    perm = np.empty(i_dim, np.int64)
    rows = rows_per_tile(i_dim)
    p = np.arange(rows)
    for ti in range(max(1, i_dim // _K_TILE)):
        for n in range(_Q_WIDTH):
            perm[ti * _K_TILE + n * rows + p] = tau(p, ti, n, i_dim)
    return perm


# =============================================================================
# Checkpoint decoding
# =============================================================================


def unpack_fp4_codes(packed_u8: np.ndarray) -> np.ndarray:
    """Unpack ``[O, K/2]`` uint8 into ``[O, K]`` E2M1 codes: element ``k`` at byte ``k // 2``, low first.

    See the module docstring for why this is the plain reading and not quark's reorder, despite the
    config's ``pack_method`` field.

    Args:
        packed_u8 (np.ndarray): ``[O, K/2]`` uint8 as stored in the checkpoint.

    Returns:
        np.ndarray: ``[O, K]`` uint8 codes in 0..15.
    """
    o, k_half = packed_u8.shape
    return np.stack([packed_u8 & 0xF, packed_u8 >> 4], axis=-1).reshape(o, k_half * 2)


def dequant_mxfp4(packed_u8: np.ndarray, scale_u8: np.ndarray) -> np.ndarray:
    """Dequantize a checkpoint MXFP4 tensor to fp32.

    Exact, not approximate: an E2M1 code times ``2 ** (e8m0 - 127)`` is representable in fp32, so this
    is the reference every layout check compares against.

    Args:
        packed_u8 (np.ndarray): ``[O, K/2]`` uint8 packed codes.
        scale_u8 (np.ndarray): ``[O, K/32]`` uint8 E8M0 exponents.

    Returns:
        np.ndarray: ``[O, K]`` fp32.
    """
    codes = unpack_fp4_codes(packed_u8)
    scale = np.power(2.0, scale_u8.astype(np.float32) - _E8M0_BIAS)
    return FP4_E2M1_LUT[codes] * np.repeat(scale, _MX_BLOCK, axis=1)


# =============================================================================
# Re-tiling into the kernel's layout
# =============================================================================


def _place_aligned(codes: np.ndarray, scale_u8: np.ndarray, mapping) -> tuple[np.ndarray, np.ndarray]:
    """Pure permutation into ``([128, K/512, O] uint16 words, [16, K/512, O] scale)``.

    Requires ``mapping`` to align the scale groups; the caller checks. Because the group under an aligned
    mapping is exactly checkpoint block ``tile*16 + j``, the scale is the same transpose for every aligned
    mapping and only the code placement varies.
    """
    o_dim, k_dim = codes.shape
    n_tile = max(1, k_dim // _K_TILE)
    rows = rows_per_tile(k_dim)
    p = np.arange(rows)
    words = np.zeros((rows, n_tile, o_dim), dtype=np.uint16)
    for tile in range(n_tile):
        for n in range(_Q_WIDTH):
            k = mapping(p, tile, n, k_dim)
            words[:, tile, :] |= (codes[:, k].T.astype(np.uint16) & 0xF) << (4 * n)
    dense_scale = np.ascontiguousarray(scale_u8.reshape(o_dim, n_tile, rows // _Q_HEIGHT).transpose(2, 1, 0))
    return words, dense_scale


def _place_requantized(dense: np.ndarray, mapping, x4_dtype) -> tuple[np.ndarray, np.ndarray]:
    """Re-quantize ``[O, K]`` fp32 into ``([128, K/512, O] x4, [16, K/512, O] scale)`` under ``mapping``.

    For a mapping whose groups straddle checkpoint blocks there is no permutation that preserves the
    exponents, so the values are re-blocked and re-quantized. The kernel's own ``quantize_to_mx`` does it,
    with the free axis laid out as ``(tile, o, n)`` so its 8-partition x 4-free blocks coincide exactly
    with the kernel's scale groups.
    """
    o_dim, k_dim = dense.shape
    n_tile = max(1, k_dim // _K_TILE)
    rows = rows_per_tile(k_dim)
    p = np.arange(rows)
    buf = np.empty((rows, n_tile, o_dim, _Q_WIDTH), dtype=np.float32)
    for tile in range(n_tile):
        for n in range(_Q_WIDTH):
            buf[:, tile, :, n] = dense[:, mapping(p, tile, n, k_dim)].T
    packed, scale = quantize_to_mx(np.ascontiguousarray(buf.reshape(rows, -1)), x4_dtype)
    return packed.reshape(rows, n_tile, o_dim), scale.reshape(rows // _Q_HEIGHT, n_tile, o_dim)


def _to_kernel(packed_u8, scale_u8, mapping, x4_dtype) -> tuple[np.ndarray, np.ndarray]:
    """One checkpoint tensor -> ``([rows, ceil(K/512), O] x4, [rows/8, ceil(K/512), O] dense scale)``.

    ``rows`` is :func:`rows_per_tile`, so a contraction narrower than one 512 tile (the shared expert's
    ``I_sh = 256``) yields the 64-partition / 8-scale-row shape the kernel expects there.
    """
    k_dim = packed_u8.shape[1] * 2
    if k_dim % _MX_BLOCK or (k_dim > _K_TILE and k_dim % _K_TILE):
        raise ValueError(f"contraction {k_dim} must be a multiple of {_K_TILE} (or of {_MX_BLOCK} if smaller)")
    if groups_align(k_dim, mapping):
        words, scale = _place_aligned(unpack_fp4_codes(packed_u8), scale_u8, mapping)
        return np.ascontiguousarray(words).view(x4_dtype), scale
    return _place_requantized(dequant_mxfp4(packed_u8, scale_u8), mapping, x4_dtype)


# =============================================================================
# PACK64 scale remaps (inverses of the kernel's own readers)
# =============================================================================


def pack64_gate_up_scale(dense: np.ndarray, i_dim: int) -> np.ndarray:
    """Dense ``[16, 2, H/512, I]`` -> PACK64 ``[2, 64, H/512, I/4]``.

    Inverts ``moe_tkg_torch._unpack64_gate_up_scale``'s relation
    ``packed[g, 16q + 4k + r, h, ti*128 + j] == dense[4q + r, g, h, ti*512 + k*128 + j]``.
    """
    n_h = dense.shape[2]
    n_ti = i_dim // _K_TILE
    out = np.zeros((2, 64, n_h, i_dim // _Q_WIDTH), dtype=np.uint8)
    for q in range(4):
        for kk in range(4):
            for r in range(4):
                src = dense[4 * q + r]  # [2, H/512, I]
                for ti in range(n_ti):
                    lo = ti * _K_TILE + kk * _PMAX
                    out[:, 16 * q + 4 * kk + r, :, ti * _PMAX : (ti + 1) * _PMAX] = src[:, :, lo : lo + _PMAX]
    return out


def pack64_down_scale(dense: np.ndarray) -> np.ndarray:
    """Dense ``[16, I/512, H]`` -> PACK64 ``[64, H]``.

    Inverts ``_unpack64_down_scale``'s relation ``packed[16q + 4ti + r, h] == dense[4q + r, ti, h]``.
    """
    n_i512, h = dense.shape[1], dense.shape[2]
    out = np.zeros((64, h), dtype=np.uint8)
    for q in range(4):
        for ti in range(n_i512):
            for r in range(4):
                out[16 * q + 4 * ti + r] = dense[4 * q + r, ti]
    return out


# =============================================================================
# Public conversion entry points
# =============================================================================


def convert_gate_up(
    gate_w, gate_s, up_w, up_s, x4_dtype, *, lossless_hidden: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """One expert's gate and up projections -> ``([128, 2, H/512, I], [2, 64, H/512, I/4])``.

    The ``I`` axis is reordered by :func:`intermediate_permutation` so the down projection can be loaded
    losslessly; a caller supplying a gate/up bias must apply the same reorder to it.

    Args:
        gate_w, up_w (np.ndarray): ``[I, H/2]`` uint8 packed codes from the checkpoint.
        gate_s, up_s (np.ndarray): ``[I, H/32]`` uint8 E8M0 scales.
        x4_dtype: The ``float4_e2m1fn_x4`` numpy dtype to view the packed words as.
        lossless_hidden (bool): True if ``rmsnorm_mx_quantize`` emits the hidden in :func:`tau` order, in
            which case the checkpoint's exponents transfer unchanged. False (default) matches the kernel
            with its default swizzle and re-quantizes, costing ~0.030 relative L2 on these two tensors.

    Returns:
        tuple: weights and PACK64 scales for one expert.
    """
    i_dim = gate_w.shape[0]
    perm = intermediate_permutation(i_dim)
    mapping = tau if lossless_hidden else glm_hidden
    gw, gsc = _to_kernel(gate_w[perm], gate_s[perm], mapping, x4_dtype)
    uw, usc = _to_kernel(up_w[perm], up_s[perm], mapping, x4_dtype)
    return (
        np.ascontiguousarray(np.stack([gw, uw], axis=1)),
        pack64_gate_up_scale(np.stack([gsc, usc], axis=1), i_dim),
    )


def convert_down(down_w, down_s, x4_dtype) -> tuple[np.ndarray, np.ndarray]:
    """One expert's down projection -> ``([128, I/512, H], [64, H])``, losslessly.

    Uses :func:`tau`, which is available here without any kernel change because the contraction axis is
    the intermediate and :func:`intermediate_permutation` has already put gate/up's output in that order.

    Args:
        down_w (np.ndarray): ``[H, I/2]`` uint8 packed codes.
        down_s (np.ndarray): ``[H, I/32]`` uint8 E8M0 scales.
        x4_dtype: The ``float4_e2m1fn_x4`` numpy dtype.

    Returns:
        tuple: weights and PACK64 scales for one expert.
    """
    w, sc = _to_kernel(down_w, down_s, tau, x4_dtype)
    return np.ascontiguousarray(w), pack64_down_scale(sc)


def convert_shared(
    gate_w, gate_s, up_w, up_s, down_w, down_s, x4_dtype, *, tp_rank: int, tp_size: int, lossless_hidden: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The shared expert, TP-sharded, in the UNPACKED scale layout the MoE layer kernel wants there.

    PACK64 does not apply: its gate/up pack needs ``I % 512 == 0`` and its down pack needs a full 16-row
    scale, and the shared expert's ``I_sh = I / tp_size`` (2048/8 = 256) satisfies neither. The kernel
    detects the layout from the tensor shape, so the unpacked form is selected automatically.

    ``base_model_tp_plan`` shards ``gate_proj`` / ``up_proj`` column-wise and ``down_proj`` row-wise, so
    this rank takes rows ``[tp_rank * I_sh, +I_sh)`` of gate/up and the matching COLUMNS of down. Each
    rank's output is then a partial sum over its ``I`` slice, which the layer's exit collective completes.

    Returns:
        tuple: ``([1, 128, 2, H/512, I_sh], [1, 16, 2, H/512, I_sh], [1, I_sh/4, 1, H], [1, I_sh/32, 1, H])``
    """
    i_full = gate_w.shape[0]
    i_sh = i_full // tp_size
    lo = tp_rank * i_sh
    sl = slice(lo, lo + i_sh)
    perm = intermediate_permutation(i_sh)
    mapping = tau if lossless_hidden else glm_hidden
    gw, gsc = _to_kernel(gate_w[sl][perm], gate_s[sl][perm], mapping, x4_dtype)
    uw, usc = _to_kernel(up_w[sl][perm], up_s[sl][perm], mapping, x4_dtype)
    # down: [H, I_full] -> this rank's I columns. The packed axis is I, so slice on the CODE axis and
    # re-pack, rather than on the stored half-width bytes.
    dn_codes = unpack_fp4_codes(down_w)[:, sl]
    dn_scale = down_s[:, lo // _MX_BLOCK : (lo + i_sh) // _MX_BLOCK]
    dw, dsc = _place_aligned(dn_codes, dn_scale, tau)
    return (
        np.ascontiguousarray(np.stack([gw, uw], axis=1))[None],
        np.stack([gsc, usc], axis=1)[None],
        np.ascontiguousarray(dw).view(x4_dtype)[None],
        dsc[None],
    )


def quantize_hidden(hidden: np.ndarray, mapping) -> tuple[np.ndarray, np.ndarray]:
    """Quantize ``[T, H]`` fp32 into the pre-quantized MX hidden ``moe_tkg`` consumes, under ``mapping``.

    ``mapping`` is not defaulted on purpose: silently assuming one is the bug this whole module exists to
    document. Pass :func:`glm_hidden` to reproduce exactly what the kernel's own internal swizzle would
    have produced (verified identical to ``_moe_tkg_mx_ref``'s
    ``reshape(T, 4, H/512, 128).transpose(3, 2, 0, 1)``), or :func:`tau` for the lossless path -- which is
    what ``rmsnorm_mx_quantize`` would have to emit for it.

    Returns:
        tuple: ``([128, H/512, T]`` float8_e4m3fn_x4, ``[128, H/512, T]`` uint8 scale in the hardware
        quadrant layout ``moe_tkg`` reads and its reference un-scatters).
    """
    t_dim, h_dim = hidden.shape
    n_h512 = h_dim // _K_TILE
    p = np.arange(_PMAX)
    buf = np.empty((_PMAX, n_h512, t_dim, _Q_WIDTH), np.float32)
    for kt in range(n_h512):
        for n in range(_Q_WIDTH):
            buf[:, kt, :, n] = hidden[:, mapping(p, kt, n, h_dim)].T
    packed, scale = quantize_to_mx(np.ascontiguousarray(buf.reshape(_PMAX, -1)), nl.float8_e4m3fn_x4)
    scale = scale.reshape(_SCALE_P, n_h512, t_dim)
    hw = np.zeros((_PMAX, n_h512, t_dim), np.uint8)  # inverse of moe_tkg_torch._unscatter_mx_scale
    for q in range(4):
        hw[32 * q : 32 * q + 4] = scale[4 * q : 4 * q + 4]
    return packed.reshape(_PMAX, n_h512, t_dim), hw
