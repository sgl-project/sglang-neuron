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

"""One GLM-5.2 sparse-layer MoE block with its EP and TP collectives fused into a single kernel.

    RMSNorm -> MX quantize -> AllGather[MLP rows] -> AllGather[EP rows] -> router -> AllGather[affinity]
        -> shared expert -> routed experts -> ReduceScatter

The row gathers are issued before the router: nothing they carry depends on the routing, so the router's PE work
overlaps them. Everything runs in one ``@nki.jit`` trace, so the collectives interleave with the expert matmuls.

Sharding (128 ranks per replica, DP16 x TP8):

* Entry and exit are token-sharded 128 ways: rank ``r`` owns slots ``[T_shard * r, T_shard * (r + 1))``, the layout
  the attention exit's ReduceScatter produces.
* Routed experts are EP128 (``E_L = 2`` local experts), so every rank gathers every token, ``T_ep = T_shard *
  ep_size`` slots.
* The shared expert is column-sharded over the TP group (or a wider ``mlp_replica_group``) and reads its group's
  ``T_sh`` rows through a smaller gather.
* ``router_tp`` shards the router by expert over the TP group, with one all_to_all handing each rank all logits of
  its rows; it matches the replicated router bit for bit while ``T_shard * tp_size <= 32``.

Two ReduceScatters, not an AllReduce: the routed partial is summed over the EP group and the shared partial over its
TP group, and both land on this rank's ``T_shard`` slots, so the two results add element-wise. That needs
``ep_replica_group`` in rank order and consecutive TP blocks, both asserted.

The MX pair is token-innermost, but an HBM collective only concatenates on dim 0, so the pair is packed into
``[T_shard, H + H/4]`` per-token rows (data and micro-scales side by side) for the gather and unpacked after it.
Quantizing before the gather halves the bytes compared with bf16.
"""

import os

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from .attention.mla.mla_proj_primitives import load_gamma_broadcast, rmsnorm_inplace
from .mlp.projection_mx_constants import _q_width
from .mlp.mlp_bf16 import mlp_bf16
from .moe.all_expert_mx_impl import MOE_GU_PREFETCH, prefetch_expert_gate_up
from .moe.moe_tkg import moe_tkg
from .router.rmsnorm_mx_quantize import take_part_loads
from .router.rmsnorm_router_quantize import rmsnorm_mx_quantize_for_router, router_topk_from_normed
from .router.router_topk import router_bias_broadcast, router_logits, router_select
from .utils.cc_dst_ring import cc_dst
from .utils.cc_pin import pin
from .utils.common_types import ActFnType, ExpertAffinityScaleMode
from .utils.kernel_assert import kernel_assert
from .utils.row_layout import spread_rows_shape
from .utils.scratch_hbm import scratch_hbm
from .utils.trace_names import unique_name

_ERROR_PREFIX = "[GLM-5.2 MoE Decode Layer DP/TP/EP Kernel]"

_P_MAX = 128
"""SBUF partition count. The fused norm emits the unquantized hidden states as [128, T, H/128]."""

_MX_H_TILE = 512
"""H elements per MX tile: 128 partitions x 4 x4-packed elements."""

# Buffer class for every HBM tensor that is an operand of a collective. At LNC=1 the RUNTIME rejects a
# collective whose sources are shared_hbm, and ``_validate_tensors_buffer_types`` still asserts that
# every src AND dst of a reduce_scatter share one buffer type, so both ends are private_hbm.
# ``router_topk_from_normed`` returns its gathered affinities in private_hbm for the same reason.
_COLLECTIVE_HBM = nl.private_hbm

MOE_EP_AG_EARLY = os.environ.get("GLM52K_MOE_EP_AG_EARLY", "0") == "1"
"""router_tp: issue the 128-rank EP row AllGather right after the router's TP row gather (before the router matmul and
the logits all_to_all), and trace the router's selection (the logits read-back, sigmoid / biased top-k, affinities)
and the affinity AllGather AFTER the shared expert and the routed row unpack. The logits all_to_all completes behind
the row gather (one collective stream, issue order), and a top-k waiting for it would hold every later Vector
instruction -- the shared expert's and the row unpack's evictions -- behind it. Traced after them, the top-k blocks
nothing the routed matmuls need: the
affinities are consumed only at expert 0's post-scale, well after the first routed matmul, so the routed experts
start earlier. Bit-identical. Off by default."""

LAT_ROUTER_AG_FIRST = os.environ.get("GLM52K_LAT_ROUTER_AG_FIRST", "0") == "1"
"""router_tp: issue the router's TP row AllGather right after the norm, BEFORE the MX row pack and the shared
expert's row AllGather. Collectives complete in issue order on one stream, so the router gather (whose logits gate
the logits all_to_all and the 128-rank EP row gather after it) otherwise waits behind the pack's DMAs and the
shared-expert row gather. Bit-identical. Off by default."""

MOE_SHARED_ROWS_FROM_EP = os.environ.get("GLM52K_MOE_SHARED_ROWS_FROM_EP", "0") == "1"
"""The shared expert reads its MLP group's rows out of the EP row gather instead of gathering them again over the
group. Both gathers carry the same packed rows (``rows_src``): the EP gather puts every rank's rows in rank order, so a
group of ``mlp_tp`` consecutive ranks finds its ``T_sh`` rows at row ``(rank // mlp_tp) * T_sh`` of ``rows_ep``, one
dynamic-offset load, which saves one collective ahead of the EP gather the routed experts wait on. Bit-identical.
Needs consecutive MLP groups of a power-of-two size. Off by default."""

# The gathered row is a byte container: 1-byte fp8 so its width is expressed in bytes. fp8 is not
# a legal collective dtype (runtime error 1202), so the collective sees a bf16 bitcast of it.
_ROW_DTYPE = nl.float8_e4m3fn
_COLLECTIVE_VIEW_DTYPE = nl.bfloat16

_X4_BYTES = 4
"""Bytes per x4-packed MX element."""

# Transpose views. The Tensor Engine needs dst.dtype == src.dtype and has no uint8 input, so the
# x4-packed MX data rides a 4-byte float view (whole containers move, values are untouched) and
# the uint8 micro-scales ride a 1-byte float view.
_X4_TP_VIEW_DTYPE = nl.float32
_U8_TP_VIEW_DTYPE = nl.float8_e5m2

# PSUM bank budget for the unpack transposes: 2 KB per partition, each bank of transposes evicted with one copy.
_PSUM_BANK_BYTES = 2048


def _numel(shape) -> int:
    """Element count of a shape tuple."""
    n = 1
    for d in shape:
        n *= int(d)
    return n


def _mx_row_width(H: int) -> int:
    """Width in bytes of one token's packed MX row: H bytes of data + H/4 uint8 micro-scales."""
    return H + H // _q_width


def _pack_prequant_rows(quant_hbm, scale_hbm, rows_hbm, T: int, H: int) -> None:
    """``[128, H/512, T]`` MX (quant, scale) pair -> ``[T, H + H/4]`` per-token rows in HBM, the token on the gather's
    dim 0. Byte-for-byte the inverse of :func:`_unpack_prequant_rows`.

    ONE Tensor Engine transpose per half (the ``[128, H/512 * T]`` x4 words and the micro-scales, token-major on the
    free axis) and one Vector eviction each, instead of ``H/512`` transposes and their evictions of ``[T, 512+]``
    (Vector cost follows the free size). Two integer Vector copies first make the pair token-major, so transposed
    partition ``t * H/512 + h`` holds h-tile ``h`` of token ``t``: each token's 512-byte x4 runs and 128-byte scale
    runs are contiguous partitions, and one DMA per (token, half) lays them along that token's row, on the Sync
    HWDGE queue rather than on ACT between the router's ACT ops.

    Args:
        quant_hbm (nl.NkiTensor): [128, H/512, T] x4-packed MX data, in HBM or already in SBUF.
        scale_hbm (nl.NkiTensor): [128, H/512, T] uint8 MX micro-scales, in the same place.
        rows_hbm (nl.NkiTensor): [T, H + H/4] 1-byte destination in HBM.
        T (int): Token count, with T * H/512 <= 128 (the transposes put every (token, h-tile) on a partition).
        H (int): Hidden dimension, a multiple of 512.

    Returns:
        None
    """
    H0 = nl.tile_size.pmax
    n_h512 = H // (H0 * _q_width)
    H_pack = _mx_row_width(H)
    TH = T * n_h512
    kernel_assert(
        TH <= H0,
        f"{_ERROR_PREFIX} the row pack puts T * H/512 = {TH} (token, h-tile) pairs on partitions, at most {H0}",
    )

    if quant_hbm.buffer == nl.sbuf:
        quant_sb, scale_sb = quant_hbm, scale_hbm
    else:
        quant_sb = nl.ndarray((H0, n_h512, T), dtype=quant_hbm.dtype, buffer=nl.sbuf)
        scale_sb = nl.ndarray((H0, n_h512, T), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.dma_copy(dst=quant_sb, src=quant_hbm)
        nisa.dma_copy(dst=scale_sb, src=scale_hbm)

    # Token-major free axis first (the PE stationary takes one free dim): integer copies move the bits untouched.
    q_tm = nl.ndarray((H0, T, n_h512), dtype=nl.uint32, buffer=nl.sbuf)
    s_tm = nl.ndarray((H0, T, n_h512), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.tensor_copy(dst=q_tm, src=quant_sb.view(nl.uint32).permute((0, 2, 1)), engine=nisa.vector_engine)
    nisa.tensor_copy(dst=s_tm, src=scale_sb.view(nl.uint8).permute((0, 2, 1)), engine=nisa.vector_engine)
    q_psum = nl.ndarray((TH, H0), dtype=_X4_TP_VIEW_DTYPE, buffer=nl.psum)
    nisa.nc_transpose(dst=q_psum, data=q_tm.view(_X4_TP_VIEW_DTYPE).reshape((H0, TH)))
    # A 1-byte PSUM destination needs the stride-2 layout.
    s_psum = nl.ndarray((TH, H0, 2), dtype=_U8_TP_VIEW_DTYPE, buffer=nl.psum)
    nisa.nc_transpose(dst=s_psum[:, :, 0], data=s_tm.view(_U8_TP_VIEW_DTYPE).reshape((H0, TH)))
    q_rows = nl.ndarray((TH, H0 * _X4_BYTES), dtype=rows_hbm.dtype, buffer=nl.sbuf)
    s_rows = nl.ndarray((TH, H0), dtype=rows_hbm.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=q_rows.view(_X4_TP_VIEW_DTYPE), src=q_psum, engine=nisa.vector_engine)
    nisa.tensor_copy(dst=s_rows.view(_U8_TP_VIEW_DTYPE), src=s_psum[:, :, 0], engine=nisa.vector_engine)
    q_run = H0 * _X4_BYTES
    dma_kw = dict(dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
    for t in range(T):
        p0 = t * n_h512
        nisa.dma_copy(
            dst=rows_hbm.ap(pattern=[[q_run, n_h512], [1, q_run]], offset=t * H_pack),
            src=q_rows[p0 : p0 + n_h512, :],
            **dma_kw,
        )
        nisa.dma_copy(
            dst=rows_hbm.ap(pattern=[[H0, n_h512], [1, H0]], offset=t * H_pack + H),
            src=s_rows[p0 : p0 + n_h512, :],
            **dma_kw,
        )


def _unpack_prequant_rows(rows_hbm, T: int, H: int, quant_dtype, row_offset_sb=None):
    """``[T, H + H/4]`` gathered rows -> ``moe_tkg``'s ``[128, H/512, T]`` MX pair in SBUF.

    Inverse of :func:`_pack_prequant_rows`. The pair is left in SBUF because ``moe_tkg`` aliases
    an SBUF pre-quantized input with no copy when ``T % 4 == 0``.

    Args:
        rows_hbm (nl.NkiTensor): [T, H + H/4] gathered rows in HBM.
        T (int): Token count, a multiple of 4.
        H (int): Hidden dimension, a multiple of 512.
        quant_dtype: MX packed dtype of the data half (``nl.float8_e4m3fn_x4``).
        row_offset_sb (nl.NkiTensor): Optional [1, 1] uint32 SBUF row index: the T rows start at that row of
            ``rows_hbm`` (a runtime value: one dynamic-offset load per T tile, ``MOE_SHARED_ROWS_FROM_EP``). None
            (default): row 0.

    Returns:
        Tuple[nl.NkiTensor, nl.NkiTensor]: (quant [128, H/512, T], scale [128, H/512, T]) in SBUF.
    """
    H0 = nl.tile_size.pmax
    n_h512 = H // (H0 * _q_width)
    H_pack = _mx_row_width(H)

    quant_sb = nl.ndarray((H0, n_h512, T), dtype=quant_dtype, buffer=nl.sbuf)
    scale_sb = nl.ndarray((H0, n_h512, T), dtype=nl.uint8, buffer=nl.sbuf)
    quant_dst = quant_sb.view(_X4_TP_VIEW_DTYPE)
    scale_dst = scale_sb.view(_U8_TP_VIEW_DTYPE)

    if T > H0:
        kernel_assert(row_offset_sb is None, "a row offset needs the T-tile-major unpack (T <= 128)")
        _unpack_rows_bank_major(rows_hbm, quant_dst, scale_dst, T, H)
        return quant_sb, scale_sb

    # T <= 128 rides the transpose's partition axis on load and its free axis on the way out
    rows_sb = nl.ndarray((T, H_pack), dtype=rows_hbm.dtype, buffer=nl.sbuf)
    if row_offset_sb is None:
        nisa.dma_copy(dst=rows_sb, src=rows_hbm[0:T, 0:H_pack])
    else:
        nisa.dma_copy(
            dst=rows_sb,
            src=rows_hbm.ap(pattern=[[H_pack, T], [1, H_pack]], offset=0, scalar_offset=row_offset_sb, indirect_dim=0),
            dge_mode=nisa.dge_mode.swdge,
        )
    quant_src = rows_sb[:, 0:H].view(_X4_TP_VIEW_DTYPE).reshape((T, n_h512, H0))
    scale_src = rows_sb[:, H:H_pack].view(_U8_TP_VIEW_DTYPE).reshape((T, n_h512, H0))
    # one Vector eviction per PSUM bank of transposes, not per transpose: the evictions queue on Vector behind the
    # MoE's other copies; at T = 128 a bank holds 4 quant tiles or 8 scale tiles
    q_per_bank = max(1, _PSUM_BANK_BYTES // (T * 4))
    s_per_bank = max(1, _PSUM_BANK_BYTES // (T * 2))
    for h0 in range(0, n_h512, q_per_bank):
        g = min(q_per_bank, n_h512 - h0)
        quant_psum = nl.ndarray((H0, g, T), dtype=_X4_TP_VIEW_DTYPE, buffer=nl.psum)
        for i in range(g):
            nisa.nc_transpose(data=quant_src[:, h0 + i, :], dst=quant_psum[:, i, :])
        nisa.tensor_copy(dst=quant_dst[:, h0 : h0 + g, 0:T], src=quant_psum[:, 0:g, :])
    for h0 in range(0, n_h512, s_per_bank):
        g = min(s_per_bank, n_h512 - h0)
        scale_psum = nl.ndarray((H0, g, T, 2), dtype=_U8_TP_VIEW_DTYPE, buffer=nl.psum)
        for i in range(g):
            nisa.nc_transpose(data=scale_src[:, h0 + i, :], dst=scale_psum[:, i, :, 0])
        nisa.tensor_copy(dst=scale_dst[:, h0 : h0 + g, 0:T], src=scale_psum[:, 0:g, :, 0])

    return quant_sb, scale_sb


def _shared_rows_offset(rank_id, mlp_tp: int, T_sh: int) -> nl.NkiTensor:
    """[1, 1] uint32 SBUF: the first row of this rank's MLP group in the EP row gather, ``(rank_id // mlp_tp) * T_sh``
    (``MOE_SHARED_ROWS_FROM_EP``; both powers of two, so two shifts). ``rank_id`` is the [1, 1] uint32 HBM EP rank."""
    rank_sb = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.dma_copy(dst=rank_sb[0:1, 0:1], src=rank_id[0:1, 0:1])
    offset_sb = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=offset_sb[0:1, 0:1],
        data=rank_sb[0:1, 0:1],
        op0=nl.right_shift,
        operand0=mlp_tp.bit_length() - 1,
        op1=nl.left_shift,
        operand1=T_sh.bit_length() - 1,
    )
    return offset_sb


def _unpack_rows_bank_major(rows_hbm, quant_dst, scale_dst, T: int, H: int) -> None:
    """``_unpack_prequant_rows`` in consumption order: bank-major over T tiles.

    Each T tile's row tile is filled by column-slice DMAs, one per PSUM bank of quant tiles plus one for all
    the scales, issued slice-major: (quant bank 0 of every T tile), (scales of every T tile), (quant bank 1),
    ... The transposes and their Vector evictions then run bank-major in the order the gate/up matmuls read
    the h-tiles: quant bank 0, scale bank 0, quant bank 1, scale bank 1, ... Byte-for-byte the same result.
    The first gate/up matmul needs h-tile 0 of all T tokens and their scales, which a T-tile-major unpack
    produces last.

    Args:
        rows_hbm (nl.NkiTensor): [T, H + H/4] gathered rows in HBM.
        quant_dst (nl.NkiTensor): [128, H/512, T] fp32 view of the quant destination in SBUF.
        scale_dst (nl.NkiTensor): [128, H/512, T] fp8e5m2 view of the scale destination in SBUF.
        T (int): Token count, a multiple of 128.
        H (int): Hidden dimension, a multiple of 512.

    Returns:
        None
    """
    H0 = nl.tile_size.pmax
    n_h512 = H // (H0 * _q_width)
    H_pack = _mx_row_width(H)
    t_starts = list(range(0, T, H0))
    q_per_bank = max(1, _PSUM_BANK_BYTES // (H0 * 4))
    s_per_bank = max(1, _PSUM_BANK_BYTES // (H0 * 2))
    q_banks = [(h0, min(q_per_bank, n_h512 - h0)) for h0 in range(0, n_h512, q_per_bank)]
    s_banks = [(h0, min(s_per_bank, n_h512 - h0)) for h0 in range(0, n_h512, s_per_bank)]

    # Column slices (byte ranges of a row): quant bank 0, all scales, quant banks 1..
    q_cols = [(h0 * H0 * _X4_BYTES, (h0 + g) * H0 * _X4_BYTES) for h0, g in q_banks]
    rows_sbs = [nl.ndarray((H0, H_pack), dtype=rows_hbm.dtype, buffer=nl.sbuf) for _ in t_starts]
    for c0, c1 in q_cols[:1] + [(H, H_pack)] + q_cols[1:]:
        for rows_sb, t0 in zip(rows_sbs, t_starts):
            nisa.dma_copy(dst=rows_sb[:, c0:c1], src=rows_hbm[t0 : t0 + H0, c0:c1])

    # Transposes + evictions, bank-major: q0, s0, q1, s1, ... (each bank across every T tile)
    order = []
    for b in range(max(len(q_banks), len(s_banks))):
        order += [("q",) + q_banks[b]] if b < len(q_banks) else []
        order += [("s",) + s_banks[b]] if b < len(s_banks) else []
    for kind, h0, g in order:
        for rows_sb, t0 in zip(rows_sbs, t_starts):
            if kind == "q":
                quant_src = rows_sb[:, 0:H].view(_X4_TP_VIEW_DTYPE).reshape((H0, n_h512, H0))
                quant_psum = nl.ndarray((H0, g, H0), dtype=_X4_TP_VIEW_DTYPE, buffer=nl.psum)
                for i in range(g):
                    nisa.nc_transpose(data=quant_src[:, h0 + i, :], dst=quant_psum[:, i, :])
                nisa.tensor_copy(
                    dst=quant_dst[:, h0 : h0 + g, t0 : t0 + H0], src=quant_psum[:, 0:g, :], engine=nisa.vector_engine
                )
            else:
                scale_src = rows_sb[:, H:H_pack].view(_U8_TP_VIEW_DTYPE).reshape((H0, n_h512, H0))
                # A 1-byte PSUM destination needs the stride-2 layout.
                scale_psum = nl.ndarray((H0, g, H0, 2), dtype=_U8_TP_VIEW_DTYPE, buffer=nl.psum)
                for i in range(g):
                    nisa.nc_transpose(data=scale_src[:, h0 + i, :], dst=scale_psum[:, i, :, 0])
                nisa.tensor_copy(
                    dst=scale_dst[:, h0 : h0 + g, t0 : t0 + H0],
                    src=scale_psum[:, 0:g, :, 0],
                    engine=nisa.vector_engine,
                )


@nki.jit
def glm52_moe_decode_layer_dp_tp_ep_kernel(
    # ── input: this rank's 128-way token shard of the post-attention residual ──
    x_shard: nl.NkiTensor,  # [T_shard, H]
    gamma: nl.NkiTensor,  # [1, H] post_attention_layernorm weight
    # ── router: replicated, or column-sharded over the TP group ──
    router_weights: nl.NkiTensor,  # [H, E] bf16, or [H, E / tp_size] (this TP position's experts)
    router_bias: nl.NkiTensor,  # [1, E] e_score_correction_bias
    # ── routed experts: this rank's E_L of the E global experts ──
    routed_gate_up_weights: nl.NkiTensor,  # [E_L, 128, 2, H/512, I]
    routed_gate_up_weights_scale: nl.NkiTensor,  # [E_L, 2, 64, H/512, I/4] (PACK64)
    routed_down_weights: nl.NkiTensor,  # [E_L, 128, I/512, H]
    routed_down_weights_scale: nl.NkiTensor,  # [E_L, 64, H] (PACK64)
    # ── shared expert: one dense MLP, column/row-sharded over TP (or over the MLP group) ──
    shared_gate_up_weights: nl.NkiTensor,  # [1, 128, 2, H/512, I_sh]
    shared_gate_up_weights_scale: nl.NkiTensor,  # [1, 16, 2, H/512, I_sh]
    shared_down_weights: nl.NkiTensor,  # [1, I_sh/4, 1, H]
    shared_down_weights_scale: nl.NkiTensor,  # [1, I_sh/32, 1, H]
    shared_affinities: nl.NkiTensor,  # [T_sh, 1] ones -- the shared expert is unweighted
    # ── rank selectors ──
    rank_id: nl.NkiTensor,  # [1, 1] uint32, EP rank -> routed affinity columns
    shared_rank_id: nl.NkiTensor,  # [1, 1] uint32, always 0 -> the single shared expert
    # ── groups + scalars ──
    ep_replica_group: ReplicaGroup,
    tp_replica_group: ReplicaGroup,
    ep_size: int,
    tp_size: int,
    top_k: int,
    eps: float,
    routed_scaling_factor: float,
    dense_only: bool = False,
    # Dense-layer weights, bf16 and UNPACKED: [H, I] gate/up and [I, H] down, this rank's TP slice.
    # Required exactly when dense_only is set, and unused otherwise.
    dense_gate_weights: nl.NkiTensor = None,
    dense_up_weights: nl.NkiTensor = None,
    dense_down_weights: nl.NkiTensor = None,
    return_affinities: bool = True,
    output: nl.NkiTensor = None,
    gu_prefetch: int = None,
    # The consecutive ranks that split this layer's dense MLP or shared expert, and their count. None
    # and 0 take the TP group. A wider group holds fewer weight columns per rank and gathers more rows.
    mlp_replica_group: ReplicaGroup = None,
    mlp_tp_size: int = 0,
    x_parts: tuple = None,
):
    """One rank's GLM-5.2 MoE block, collectives included (see the module docstring).

    Dimensions:
        T_shard: Token slots this rank owns on entry and exit (``ceil(C * spec / ep_size)``)
        T_ep: ``T_shard * ep_size`` -- global token slots, what the routed experts see
        T_sh: ``T_shard * mlp_tp_size`` -- the MLP group's slots (``T_shard * tp_size``, this DP replica's,
            by default), what the shared expert or the dense MLP sees
        H: Hidden dimension, a multiple of 512
        E: Number of global routed experts
        E_L: ``E // ep_size`` -- routed experts local to this rank
        I: Routed-expert intermediate dimension
        I_sh: Shared-expert intermediate dimension after the TP (or MLP group) column shard

    Args:
        x_shard (nl.NkiTensor): [T_shard, H] fp16/bf16 post-attention residual in HBM.
        gamma (nl.NkiTensor): [1, H] ``post_attention_layernorm`` weight in HBM.
        router_weights (nl.NkiTensor): [H, E] bf16 router projection in HBM. Must be bf16: the
            router matmul contracts in bf16 regardless of the activation dtype. ``[H, E / tp_size]``
            selects the TP-sharded router (``router_tp``): TP position ``p`` holds experts
            ``[p * E / tp_size, ...)``, computes their logits for the TP group's ``T_shard * tp_size``
            rows, and one all_to_all over the TP group gives each rank all ``E`` logits of its own rows
            (``_router_topk_tp``). The expert choice and the affinities are bit-identical to the
            replicated router's while ``T_shard * tp_size <= 32``; above that they match to fp32 rounding
            only (see the module docstring).
        router_bias (nl.NkiTensor): [1, E] ``e_score_correction_bias`` in HBM, every expert under
            either router form; ``E`` is read from it. Added to the sigmoid scores for expert SELECTION
            only, never to the returned affinities. Loaded into fp32 whatever its HBM dtype.
        routed_gate_up_weights (nl.NkiTensor): [E_L, 128, 2, H/512, I] MX-packed fused gate/up.
        routed_gate_up_weights_scale (nl.NkiTensor): [E_L, 2, 64, H/512, I/4] uint8 PACK64 scales.
        routed_down_weights (nl.NkiTensor): [E_L, 128, I/512, H] MX-packed down projection.
        routed_down_weights_scale (nl.NkiTensor): [E_L, 64, H] uint8 PACK64 scales.
        shared_gate_up_weights (nl.NkiTensor): [1, 128, 2, H/512, I_sh] MX-packed fused gate/up.
        shared_gate_up_weights_scale (nl.NkiTensor): [1, 16, 2, H/512, I_sh] uint8 scales. The
            PACK64 layout needs ``I % 512 == 0``, which ``I_sh = 256`` is not, so this half of
            the kernel uses the unpacked layout; both are detected from the tensor shape.
        shared_down_weights (nl.NkiTensor): [1, I_sh/4, 1, H] MX-packed down projection.
        shared_down_weights_scale (nl.NkiTensor): [1, I_sh/32, 1, H] uint8 scales.
        shared_affinities (nl.NkiTensor): [T_sh, 1] all-ones affinity column in HBM. The shared
            expert's contribution is unweighted, and POST_SCALE is the only affinity mode the
            expert kernel supports, so it is scaled by 1. A dense layer does not read it.
        rank_id (nl.NkiTensor): [1, 1] uint32 in HBM. Selects routed affinity columns
            ``[E_L * rank_id, E_L * (rank_id + 1))``.
        shared_rank_id (nl.NkiTensor): [1, 1] uint32 zero in HBM. The shared expert is expert 0
            of a one-column affinity table on every rank, so it needs its own zero selector.
        ep_replica_group (ReplicaGroup): One group listing all ``ep_size`` ranks in order.
        tp_replica_group (ReplicaGroup): ``ep_size / tp_size`` groups of ``tp_size`` CONSECUTIVE
            ranks (``[[0..7], [8..15], ...]``).
        ep_size (int): Expert-parallel degree; also the total rank count.
        tp_size (int): Tensor-parallel degree of the shared expert.
        top_k (int): Routed experts selected per token.
        eps (float): RMSNorm epsilon.
        routed_scaling_factor (float): ``config.routed_scaling_factor`` (2.5 for GLM-5.2).
        return_affinities (bool): Copy the gathered affinity table into a shared_hbm output. The
            standalone test reads it; the giga stack passes False and gets None. (default: True)
        output (nl.NkiTensor): Optional [T_shard, H] destination for ``out``. An SBUF tile receives the
            combined sum with no HBM store (the giga stack's residual add reads it there); a shared_hbm
            tensor is written as the kernel's own allocation would be. None allocates one. (default: None)
        gu_prefetch (int): Overrides ``GLM52K_MOE_GU_PREFETCH`` for this layer (None: the knob). (default: None)
        mlp_replica_group (ReplicaGroup): Groups of ``mlp_tp_size`` CONSECUTIVE ranks that split the
            dense MLP (dense layer) or the shared expert (sparse layer): the MLP's row gather and its
            ReduceScatter run over it. None selects ``tp_replica_group``. The router keeps the TP group
            either way. (default: None)
        mlp_tp_size (int): Ranks in one ``mlp_replica_group`` group, dividing ``ep_size``; 0 selects
            ``tp_size``. ``shared_affinities`` then has ``T_shard * mlp_tp_size`` rows. (default: 0)
        x_parts (tuple): Optional ``(residual, delta)``, two [T_shard, H] HBM tensors whose bf16 sum IS ``x_shard``
            (the attention exit publishes them): the norm loads both and adds them with the residual add's own op, so it does
            not wait for ``x_shard``'s store. Sparse layers only. (default: None)

    Returns:
        Tuple[nl.NkiTensor, nl.NkiTensor]:
            out: [T_shard, H] in ``x_shard``'s dtype -- the MoE block's contribution to this
                rank's token shard, ready for the layer's residual add.
            expert_affinities_gathered: [T_ep, E] the gathered dense affinity table, or None for a
                dense layer or when ``return_affinities`` is False. Returned for bring-up
                observability: it isolates RMSNorm + router + AllGather from the expert matmuls
                when a whole-layer comparison fails.

    Pseudocode:
        normed, quant, scale = rmsnorm_mx_quantize(x_shard, gamma)
        rows_replica = AllGather[MLP](pack(quant, scale))          # MLP group = TP group by default
        rows_all = AllGather[EP](pack(quant, scale))
        affinity = router(normed, router_w, router_b)
        #   router_tp: logits = a2a[TP](AllGather[TP](normed) @ router_w[:, my experts])
        affinity_all = AllGather[EP](affinity)
        shared = moe(unpack(rows_replica), shared_weights, ones)
        routed = moe(unpack(rows_all), routed_weights, affinity_all[:, this_rank])
        out = ReduceScatter[EP](routed) + ReduceScatter[MLP](shared)
    """
    T_shard, H = x_shard.shape
    # Both are routed quantities, and a dense layer has no router or routed experts. Zero keeps the shared-expert path's
    # arithmetic below unchanged -- it never reads E or E_L -- while making the dense branch's absent inputs explicit
    # rather than a NoneType dereference. E comes from the bias, which holds every expert under both router forms.
    E = 0 if dense_only else router_bias.shape[-1]
    router_shards = 1 if dense_only else E // router_weights.shape[-1]
    E_L = 0 if dense_only else routed_gate_up_weights.shape[0]
    T_ep = T_shard * ep_size
    mlp_tp = mlp_tp_size or tp_size
    mlp_group = tp_replica_group if mlp_replica_group is None else mlp_replica_group
    T_sh = T_shard * mlp_tp
    H_pack = _mx_row_width(H)

    _validate_moe_decode_layer(
        T_shard=T_shard,
        H=H,
        E=E,
        E_L=E_L,
        T_ep=T_ep,
        T_sh=T_sh,
        ep_size=ep_size,
        tp_size=tp_size,
        top_k=top_k,
        ep_replica_group=ep_replica_group,
        tp_replica_group=tp_replica_group,
        shared_experts=shared_gate_up_weights.shape[0],
        shared_affinity_rows=shared_affinities.shape[0],
        dense_only=dense_only,
        router_shards=router_shards,
        router_columns=0 if dense_only else router_weights.shape[-1],
        mlp_tp=mlp_tp,
        mlp_replica_group=mlp_group,
    )

    # ═══════ RMSNorm -> router -> MX quantize, on this rank's own tokens only ═══════
    # Per-token work, so doing it before the gather costs 1/ep_size of doing it after.
    if dense_only:
        """
        A DENSE layer is a plain ``GlmMoeDsaMLP`` in bf16, and it shares NOTHING with the MX pipeline
        below -- not the router, not the routed experts, not even the quantize.

        The checkpoint is why. ``quantization_config.exclude`` keeps
        ``model.layers.{0,1,2}.mlp.{gate,up,down}_proj`` in bf16, so those layers have no micro-scales to
        carry and no packed rows to move: this branch norms in bf16, AllGathers the bf16 rows over the TP
        group, and runs one dense MLP. Routing the dense layers through the MX path instead would quantize
        weights the checkpoint ships unquantized, and would understate weight traffic by 42.4 MB per rank
        per layer (56.6 MB bf16 against 14.2 MB MXFP4).

        The single TP ReduceScatter after this is the layer's ONLY collective -- one of a sparse layer's
        three -- which is the other reason the dense layers are much cheaper.
        """
        normed_shard = nl.ndarray(
            (T_shard, H), dtype=x_shard.dtype, buffer=_COLLECTIVE_HBM, name=unique_name("dense_normed_shard")
        )
        x_sb = nl.ndarray((_P_MAX, H), dtype=x_shard.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_sb[0:T_shard, 0:H], src=x_shard[0:T_shard, 0:H])
        rmsnorm_inplace(x_sb[0:_P_MAX, 0:H], load_gamma_broadcast(gamma, T_shard), T_shard, eps)
        nisa.dma_copy(dst=normed_shard[0:T_shard, 0:H], src=x_sb[0:T_shard, 0:H])

        # The dense MLP runs over the whole MLP group's slots (the DP replica's by default), so that group
        # gathers its rows first -- the same group whose ReduceScatter completes the row-sharded down
        # projection afterwards.
        normed_tp = cc_dst((T_sh, H), x_shard.dtype, _COLLECTIVE_HBM, "dense_normed_tp")
        pin(normed_shard), pin(normed_tp)  # long context only; see utils/cc_pin.py
        ncc.all_gather(srcs=[normed_shard], dsts=[normed_tp], replica_group=mlp_group, collective_dim=0)

        # [1, T_sh, H]: the kernel's input contract is 3-D, and B = 1 with a flat token axis is what the
        # rest of this layer already hands its sub-kernels.
        shared_out = mlp_bf16(
            hidden_tensor=normed_tp.reshape((1, T_sh, H)),
            gate_weights=dense_gate_weights,
            up_weights=dense_up_weights,
            down_weights=dense_down_weights,
            activation_fn=ActFnType.SiLU,
            output_dtype=nl.bfloat16,
        )
        routed_out = None
        affinities_ep = None
    else:
        # ═══════ RMSNorm -> MX quantize, on this rank's own tokens only ═══════
        # The pair stays in SBUF: it is packed into the collective's rows from there, and the HBM spill
        # a fused router kernel makes (then reloads for the pack) is avoided.
        n_h512 = H // _MX_H_TILE
        hidden_quant = nl.ndarray((_P_MAX, n_h512, T_shard), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
        hidden_scale = nl.ndarray((_P_MAX, n_h512, T_shard), dtype=nl.uint8, buffer=nl.sbuf)
        _, _, router_in = rmsnorm_mx_quantize_for_router(
            x_shard.reshape((1, T_shard, H)),
            gamma,
            eps,
            output_quant=hidden_quant,
            output_scale=hidden_scale,
            input_parts=None if x_parts is None else tuple(p.reshape((1, T_shard, H)) for p in x_parts),
        )
        # The norm's part loads read the attention exit RS's destination (they wait for it): routed expert 0's gate
        # weight DMA is held behind them below.
        norm_part_loads = [x for x in take_part_loads() if x is not None]
        # GLM52K_LAT_ROUTER_AG_FIRST: the router's rows go first on the collective stream (see the knob).
        router_rows_tp = None
        router_ag_first = LAT_ROUTER_AG_FIRST and router_shards > 1
        if router_ag_first:
            router_rows_tp = _router_tp_gather(router_in, tp_replica_group, tp_size)

        # ═══════ Dispatch the rows BEFORE the router runs ═══════
        # Nothing the row gathers carry depends on the routing, so they are in flight while the router runs;
        # the affinity gather follows the router and is consumed only at the first routed post-scale.
        rows_src = nl.ndarray(
            (T_shard, H_pack), dtype=_ROW_DTYPE, buffer=_COLLECTIVE_HBM, name=unique_name("mx_rows_src")
        )
        _pack_prequant_rows(hidden_quant, hidden_scale, rows_src, T_shard, H)

        rows_ep = cc_dst((T_ep, H_pack), _ROW_DTYPE, _COLLECTIVE_HBM, "mx_rows_ep")

        # Collectives complete in issue order, so each gather is issued in the order its consumer runs:
        # the MLP group's rows first (the shared expert, which runs next, needs only this gather and starts
        # while the 128-rank gathers are in flight), then the EP rows (the routed matmuls), and the affinities
        # last (consumed only at the post-scale after each expert's down projection, so they have expert 0's
        # gate/up/down to hide behind).
        # Long context pins the row gathers' operands (utils/cc_pin.py): the source, the HBM TP rows and the EP
        # rows, which are a ring slot by default and then left out.
        pin(rows_src), pin(rows_ep)
        if MOE_SHARED_ROWS_FROM_EP:
            groups = [list(range(g * mlp_tp, (g + 1) * mlp_tp)) for g in range(ep_size // mlp_tp)]
            kernel_assert(
                mlp_tp & (mlp_tp - 1) == 0 and T_sh & (T_sh - 1) == 0 and ep_size % mlp_tp == 0
                and mlp_group._value == groups,
                f"{_ERROR_PREFIX} GLM52K_MOE_SHARED_ROWS_FROM_EP needs consecutive MLP groups of a power-of-two size "
                f"(mlp_tp={mlp_tp}, T_sh={T_sh}), got {mlp_group._value}",
            )
        else:
            rows_tp = nl.ndarray(
                (T_sh, H_pack), dtype=_ROW_DTYPE, buffer=_COLLECTIVE_HBM, name=unique_name("mx_rows_tp")
            )
            pin(rows_tp)
            ncc.all_gather(
                srcs=[rows_src.view(_COLLECTIVE_VIEW_DTYPE)],
                dsts=[rows_tp.view(_COLLECTIVE_VIEW_DTYPE)],
                replica_group=mlp_group,
                collective_dim=0,
            )

        ep_rows_gathered = []

        def _gather_ep_rows():
            ep_rows_gathered.append(True)
            ncc.all_gather(
                srcs=[rows_src.view(_COLLECTIVE_VIEW_DTYPE)],
                dsts=[rows_ep.view(_COLLECTIVE_VIEW_DTYPE)],
                replica_group=ep_replica_group,
                collective_dim=0,
            )

        # MOE_EP_AG_EARLY: the EP rows go early, and the router's selection is traced after the row unpack
        ep_early = MOE_EP_AG_EARLY and router_shards > 1

        # router_tp: the TP group's normalized rows, which the sharded router's matmul reads. Issued before
        # the 128-rank EP row gather, or it would complete only after that one (collectives complete in issue
        # order); its source is the norm's output, so it does not wait for the row pack.
        if router_shards > 1 and not router_ag_first:
            router_rows_tp = _router_tp_gather(router_in, tp_replica_group, tp_size)

        _gather_ep_rows()

        # ═══════ Router: sigmoid top-k on the normalized copy, then the affinity gather ═══════
        if router_shards == 1:
            expert_index, affinities, _ = router_topk_from_normed(
                router_in, router_weights, router_bias, top_k=top_k, routed_scaling_factor=routed_scaling_factor
            )
        else:
            router_out = _router_topk_tp(
                router_rows_tp,
                router_weights,
                router_bias,
                tp_replica_group=tp_replica_group,
                tp_size=tp_size,
                T_shard=T_shard,
                top_k=top_k,
                routed_scaling_factor=routed_scaling_factor,
                defer_select=ep_early,
            )
            if ep_early:
                expert_index, router_select_later = router_out
            else:
                expert_index, affinities = router_out

        def _gather_affinities(affinities):
            affinities_ep = cc_dst((T_ep, E), affinities.dtype, _COLLECTIVE_HBM, "affinities_ep")
            pin(affinities), pin(affinities_ep)  # long context only; see utils/cc_pin.py
            ncc.all_gather(srcs=[affinities], dsts=[affinities_ep], replica_group=ep_replica_group, collective_dim=0)
            return affinities_ep

        if not ep_early:
            affinities_ep = _gather_affinities(affinities)

        # ═══════ Shared expert: one dense MLP over this DP replica's token slots ═══════
        # BEFORE the routed experts: it depends only on the 8-rank TP gather, so its weight load and
        # matmuls run while the 128-rank EP gathers are still in flight, instead of after the routed
        # experts where they sat in the layer's tail.
        routed_gu0 = None
        gu_prefetch = MOE_GU_PREFETCH if gu_prefetch == None else gu_prefetch
        if gu_prefetch == 2:
            # Routed expert 0's gate/up loads ahead of the shared expert (see MOE_GU_PREFETCH)
            routed_gu0 = prefetch_expert_gate_up(
                routed_gate_up_weights, routed_gate_up_weights_scale, routed_down_weights, T=T_ep
            )
        if MOE_SHARED_ROWS_FROM_EP:
            # the EP gather precedes this read in program order on every path (see MOE_SHARED_ROWS_FROM_EP)
            kernel_assert(ep_rows_gathered, f"{_ERROR_PREFIX} the shared expert's rows come from the EP gather, not issued yet")
            quant_tp_sb, scale_tp_sb = _unpack_prequant_rows(
                rows_ep, T_sh, H, hidden_quant.dtype, row_offset_sb=_shared_rows_offset(rank_id, mlp_tp, T_sh)
            )
        else:
            quant_tp_sb, scale_tp_sb = _unpack_prequant_rows(rows_tp, T_sh, H, hidden_quant.dtype)
        shared_out = moe_tkg(
            hidden_input=quant_tp_sb,
            hidden_input_scale=scale_tp_sb,
            expert_gate_up_weights=shared_gate_up_weights,
            expert_down_weights=shared_down_weights,
            expert_gate_up_weights_scale=shared_gate_up_weights_scale,
            expert_down_weights_scale=shared_down_weights_scale,
            expert_affinities=shared_affinities,
            expert_index=expert_index,
            rank_id=shared_rank_id,
            mask_unselected_experts=False,
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            activation_fn=ActFnType.SiLU,
            output_dtype=nl.bfloat16,
            # unit affinities: the post-scale multiplies by an immediate 1.0, with no table to load
            unit_affinities=True,
        )

        # ═══════ Routed experts: this rank's E_L experts over every global token slot ═══════
        # expert_index is this rank's own [T_shard, top_k] table, not the gathered one: with
        # mask_unselected_experts=False the expert kernel reads only its top_k width, because the
        # affinity table the router produced is already zero outside each token's top_k.
        if gu_prefetch == 1:
            # Expert 0's gate/up loads ahead of the row unpack: G0 / U0 cannot inherit its scratch
            routed_gu0 = prefetch_expert_gate_up(
                routed_gate_up_weights,
                routed_gate_up_weights_scale,
                routed_down_weights,
                T=T_ep,
                next_too=True,
            )
            if norm_part_loads:
                # hold expert 0's gate load behind the norm's read-back so it streams during the MoE front rather
                # than through the attention exit's stores and ReduceScatter
                routed_gu0.dmas[0].depends_on(*norm_part_loads)
        quant_ep_sb, scale_ep_sb = _unpack_prequant_rows(rows_ep, T_ep, H, hidden_quant.dtype)
        if ep_early:
            # MOE_EP_AG_EARLY: the selection and the affinity gather behind the shared expert and the row unpack
            affinities = router_select_later()
            affinities_ep = _gather_affinities(affinities)
        routed_out = moe_tkg(
            hidden_input=quant_ep_sb,
            hidden_input_scale=scale_ep_sb,
            expert_gate_up_weights=routed_gate_up_weights,
            expert_down_weights=routed_down_weights,
            expert_gate_up_weights_scale=routed_gate_up_weights_scale,
            expert_down_weights_scale=routed_down_weights_scale,
            expert_affinities=affinities_ep,
            expert_index=expert_index,
            rank_id=rank_id,
            mask_unselected_experts=False,
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            activation_fn=ActFnType.SiLU,
            output_dtype=nl.bfloat16,
            prefetched_gate_up=routed_gu0,
        )

    # ═══════ Combine: both ReduceScatters land on the same T_shard token slots ═══════
    # The MLP group's ReduceScatter hands rank r its group's local slots [T_shard*p, ...) with p = r mod mlp_tp,
    # i.e. global [T_shard*r, ...): the same slots as the EP one, for any consecutive-block group.
    shared_shard = cc_dst((T_shard, H), shared_out.dtype, _COLLECTIVE_HBM, "shared_shard", reduce_scatter=True)
    pin(shared_out), pin(shared_shard)  # long context only; see utils/cc_pin.py
    ncc.reduce_scatter(srcs=[shared_out], dsts=[shared_shard], op=nl.add, replica_group=mlp_group, collective_dim=0)

    # The sum is formed in SBUF. An SBUF ``output`` IS that tile (the giga stack's residual add reads it
    # from there, saving the 24 KB store and reload per layer); otherwise it is stored to the HBM output.
    out = output
    if out is None:
        out = nl.ndarray((T_shard, H), dtype=x_shard.dtype, buffer=nl.shared_hbm, name=unique_name("moe_layer_out"))
    kernel_assert(
        (tuple(out.shape) == (T_shard, H) or (out.buffer == nl.sbuf and _numel(out.shape) == T_shard * H))
        and out.dtype == x_shard.dtype,
        f"{_ERROR_PREFIX} output must hold [{T_shard}, {H}] {x_shard.dtype}, got {out.shape} {out.dtype}",
    )
    # The adds run on the partition-spread view of the rows (see utils/row_layout.py): at T_shard = 2
    # the token-major tile would put 6144 elements on each of two partitions.
    sb_shape = tuple(out.shape) if out.buffer == nl.sbuf else (spread_rows_shape(T_shard, H) or (T_shard, H))
    out_sb = out if out.buffer == nl.sbuf else nl.ndarray(sb_shape, dtype=out.dtype, buffer=nl.sbuf)
    shared_sb = nl.ndarray(sb_shape, dtype=shared_shard.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=shared_sb, src=shared_shard.reshape(sb_shape))
    if dense_only:
        # A dense layer's block output is the MLP alone. There is no routed branch to add, and the
        # single TP ReduceScatter above is this layer's ONLY collective -- one of three, which is why
        # the three dense layers are much cheaper than a sparse one.
        nisa.tensor_copy(dst=out_sb, src=shared_sb)
    else:
        routed_shard = cc_dst((T_shard, H), routed_out.dtype, _COLLECTIVE_HBM, "routed_shard", reduce_scatter=True)
        pin(routed_out), pin(routed_shard)  # long context only; see utils/cc_pin.py
        ncc.reduce_scatter(
            srcs=[routed_out], dsts=[routed_shard], op=nl.add, replica_group=ep_replica_group, collective_dim=0
        )
        routed_sb = nl.ndarray(sb_shape, dtype=routed_shard.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=routed_sb, src=routed_shard.reshape(sb_shape))
        nisa.tensor_tensor(dst=out_sb, data1=routed_sb, data2=shared_sb, op=nl.add)
    if out_sb is not out:
        nisa.dma_copy(dst=out.reshape(sb_shape), src=out_sb)

    # The gathered affinity table is copied into its own output rather than returned directly:
    # the compiler rejects a collective whose destination is a kernel IO tensor
    # ("Collective instruction cannot write IO tensors").
    if dense_only or not return_affinities:
        # No router ran (dense), or the caller does not want the table (the giga stack discards it, and
        # the 128 KB HBM->HBM copy per layer is traffic for nothing). The standalone MoE test is the
        # only consumer and it never runs a dense layer.
        affinities_out = None
    else:
        affinities_out = nl.ndarray(
            (T_ep, E), dtype=affinities_ep.dtype, buffer=nl.shared_hbm, name=unique_name("affinities_gathered_out")
        )
        nisa.dma_copy(dst=affinities_out, src=affinities_ep)

    return out, affinities_out


def _router_tp_gather(router_in, tp_replica_group: ReplicaGroup, tp_size: int):
    """AllGather[TP] of the normalized bf16 rows that the TP-sharded router reads (``router_tp``).

    The gather concatenates on dim 0, so rank ``r``'s ``[128, T_shard * H/128]`` block lands at rows
    ``[128 r, 128 (r + 1))`` of the destination; ``_router_topk_tp`` reads it back at tokens
    ``[T_shard r, T_shard (r + 1))`` with one strided DMA. The router's H1-major layout (``x[p, t, h1] ==
    normed[t, h1 * 128 + p]``) rides through unchanged, because each partition's row moves whole.

    Args:
        router_in (nl.NkiTensor): [128, T_shard, H/128] bf16 normalized rows in SBUF
            (``rmsnorm_mx_quantize_for_router``).
        tp_replica_group (ReplicaGroup): The TP groups.
        tp_size (int): Ranks in one TP group.

    Returns:
        nl.NkiTensor: [128 * tp_size, T_shard * H/128] bf16 gathered rows in private HBM.
    """
    _, T_shard, H_free = router_in.shape
    row_free = T_shard * H_free
    rows_src = nl.ndarray(
        (_P_MAX, row_free), dtype=nl.bfloat16, buffer=_COLLECTIVE_HBM, name=unique_name("router_rows_src")
    )
    nisa.dma_copy(dst=rows_src, src=router_in.reshape((_P_MAX, row_free)))
    rows_tp = cc_dst((_P_MAX * tp_size, row_free), nl.bfloat16, _COLLECTIVE_HBM, "router_rows_tp")
    pin(rows_src), pin(rows_tp)  # long context only; see utils/cc_pin.py
    ncc.all_gather(srcs=[rows_src], dsts=[rows_tp], replica_group=tp_replica_group, collective_dim=0)
    return rows_tp


def _router_topk_tp(
    rows_tp,
    router_weights,
    router_bias,
    tp_replica_group: ReplicaGroup,
    tp_size: int,
    T_shard: int,
    top_k: int,
    routed_scaling_factor: float,
    defer_select: bool = False,
):
    """``router_topk_from_normed`` with the router column-sharded over the TP group (``router_tp``).

    ``defer_select``: stop after the all_to_all and return ``(expert_index, select)``; ``select()`` traces step 4
    (the logits read-back and ``router_select``) and returns ``expert_affinities`` (``MOE_EP_AG_EARLY``).

    A replicated router is ``H * E`` bf16 on every rank (3 MiB a layer). Here TP position ``p`` holds the
    columns of experts ``[p * E_w, (p + 1) * E_w)``, ``E_w = E / tp_size``:

        1. (``_router_tp_gather``) AllGather[TP] of the normalized bf16 rows, in the router's SBUF layout.
        2. Logits of this rank's ``E_w`` experts for the group's ``T_shard * tp_size`` rows (``router_logits``).
        3. all_to_all[TP]: block ``q`` (the rows rank ``q`` owns) goes to rank ``q``, so each rank receives its
           own rows' logits for every expert block, in expert order.
        4. Sigmoid, biased top-K and the affinities on the ``[T_shard, E]`` logits (``router_select``), with the
           fp32 bias of all ``E`` experts, as the replicated router does.

    Each logit is the same bf16 dot product over the full ``H`` in the same PSUM order, and the fp32 logits
    move as bytes, so the expert choice (ties included) and the affinities are bit-identical to the replicated
    router's. That order holds while the group's rows take the same PE column-tile width as one rank's
    (``router_logits`` sizes it from the row count): ``T_shard * tp_size <= 32``, as at the production
    ``T_shard = 2`` (16 rows). Above that bound exact equivalence does NOT hold: a wider tile regroups the fp32
    partial sums, so the logits match to fp32 rounding only and a top-k tie can resolve differently.

    Args:
        rows_tp (nl.NkiTensor): [128 * tp_size, T_shard * H/128] bf16 gathered rows (``_router_tp_gather``).
        router_weights (nl.NkiTensor): [H, E / tp_size] bf16, this TP position's router columns.
        router_bias (nl.NkiTensor): [1, E] selection bias of every expert, in HBM.
        tp_replica_group (ReplicaGroup): The TP groups.
        tp_size (int): Ranks in one TP group.
        T_shard (int): Token slots this rank owns.
        top_k (int): Experts selected per token.
        routed_scaling_factor (float): Final multiplier on the L1-normalized affinities.

    Returns:
        tuple: (expert_index [T_shard, top_k] int32, expert_affinities [T_shard, E] bf16 in private HBM -- a
            collective source), as ``router_topk_from_normed`` returns its first two.
    """
    H_free = rows_tp.shape[1] // T_shard
    E_w = router_weights.shape[-1]
    E = E_w * tp_size
    T_sh = T_shard * tp_size
    kernel_assert(
        router_weights.dtype == nl.bfloat16 and tuple(router_weights.shape) == (H_free * _P_MAX, E_w),
        f"{_ERROR_PREFIX} the TP-sharded router contracts the gathered bf16 rows over H={H_free * _P_MAX}, so "
        f"router_weights must be bf16 [H, E / tp_size]; got {router_weights.dtype} {router_weights.shape}",
    )

    # 1-2. The group's rows, then this rank's experts' logits for all of them.
    x_tp = _router_tp_rows(rows_tp, T_shard, tp_size)
    logits_tp, _ = router_logits(x_tp, router_weights)  # [T_sh, E_w] fp32

    # 3. fp32 is moved as a bf16 view: the payload is bytes, and every collective here moves bf16.
    logits_src = nl.ndarray(
        (T_sh, E_w), dtype=nl.float32, buffer=_COLLECTIVE_HBM, name=unique_name("router_logits_src")
    )
    nisa.dma_copy(dst=logits_src, src=logits_tp)
    logits_recv = cc_dst((T_sh, E_w), nl.float32, _COLLECTIVE_HBM, "router_logits_recv")
    pin(logits_src), pin(logits_recv)  # long context only; see utils/cc_pin.py
    ncc.all_to_all(
        srcs=[logits_src.view(_COLLECTIVE_VIEW_DTYPE)],
        dsts=[logits_recv.view(_COLLECTIVE_VIEW_DTYPE)],
        replica_group=tp_replica_group,
        collective_dim=0,
    )
    if defer_select:
        expert_index = nl.ndarray((T_shard, top_k), dtype=nl.int32, buffer=scratch_hbm())
        # The bias broadcast does not wait for the logits; traced here so the scheduler cannot hoist its matmul
        # above the routed row unpack
        bias_bc_sb = router_bias_broadcast(router_bias, T_shard, E)

        def _select():
            logits_sb = _router_tp_logits(logits_recv, T_shard, tp_size)
            expert_affinities = nl.ndarray((T_shard, E), dtype=nl.bfloat16, buffer=nl.private_hbm)
            router_select(
                router_logits_sb=logits_sb,
                bias_broadcast_sb=bias_bc_sb,
                expert_affinities=expert_affinities,
                expert_index=expert_index,
                k=top_k,
                routed_scaling_factor=routed_scaling_factor,
                output_dtype=nl.bfloat16,
            )
            return expert_affinities

        return expert_index, _select
    logits_sb = _router_tp_logits(logits_recv, T_shard, tp_size)

    # 4. The same output buffers as router_topk_from_normed: private_hbm affinities (the EP AllGather reads
    # them, and at LNC=1 the runtime rejects a shared_hbm collective source).
    expert_index = nl.ndarray((T_shard, top_k), dtype=nl.int32, buffer=scratch_hbm())
    expert_affinities = nl.ndarray((T_shard, E), dtype=nl.bfloat16, buffer=nl.private_hbm)
    router_select(
        router_logits_sb=logits_sb,
        bias_broadcast_sb=router_bias_broadcast(router_bias, T_shard, E),
        expert_affinities=expert_affinities,
        expert_index=expert_index,
        k=top_k,
        routed_scaling_factor=routed_scaling_factor,
        output_dtype=nl.bfloat16,
    )
    return expert_index, expert_affinities


def _router_tp_rows(rows_tp, T_shard: int, tp_size: int):
    """``_router_tp_gather``'s destination -> the router's ``[128, T_shard * tp_size, H/128]`` SBUF input.

    Rank ``r``'s block (rows ``[128 r, 128 (r + 1))``) becomes tokens ``[T_shard r, T_shard (r + 1))``, so token
    ``t`` of the group is the TP group's global slot ``t``, in rank order: one strided DMA.

    Args:
        rows_tp (nl.NkiTensor): [128 * tp_size, T_shard * H/128] bf16 gathered rows in HBM.
        T_shard (int): Token slots per rank.
        tp_size (int): Ranks in the gather.

    Returns:
        nl.NkiTensor: [128, T_shard * tp_size, H/128] bf16 in SBUF, partition dim holding consecutive H.
    """
    row_free = rows_tp.shape[1]
    x_tp = nl.ndarray((_P_MAX, T_shard * tp_size, row_free // T_shard), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=x_tp.reshape((_P_MAX, tp_size, row_free)),
        src=rows_tp.ap(pattern=[[row_free, _P_MAX], [_P_MAX * row_free, tp_size], [1, row_free]], offset=0),
    )
    return x_tp


def _router_tp_logits(logits_recv, T_shard: int, tp_size: int):
    """The logits all_to_all's destination -> this rank's ``[T_shard, E]`` fp32 logits of every expert, in SBUF.

    Block ``r`` of the receive (rows ``[T_shard r, T_shard (r + 1))``) holds this rank's rows for experts
    ``[r * E_w, (r + 1) * E_w)``, so one strided DMA puts every block at its expert columns.

    Args:
        logits_recv (nl.NkiTensor): [T_shard * tp_size, E_w] fp32 received logits in HBM.
        T_shard (int): Token slots per rank.
        tp_size (int): Ranks in the exchange.

    Returns:
        nl.NkiTensor: [T_shard, E_w * tp_size] fp32 in SBUF.
    """
    E_w = logits_recv.shape[1]
    logits_sb = nl.ndarray((T_shard, E_w * tp_size), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=logits_sb.reshape((T_shard, tp_size, E_w)),
        src=logits_recv.ap(pattern=[[E_w, T_shard], [T_shard * E_w, tp_size], [1, E_w]], offset=0),
    )
    return logits_sb


def _validate_moe_decode_layer(
    T_shard: int,
    H: int,
    E: int,
    E_L: int,
    T_ep: int,
    T_sh: int,
    ep_size: int,
    tp_size: int,
    top_k: int,
    ep_replica_group: ReplicaGroup,
    tp_replica_group: ReplicaGroup,
    shared_experts: int,
    shared_affinity_rows: int,
    dense_only: bool = False,
    router_shards: int = 1,
    router_columns: int = 0,
    mlp_tp: int = 0,
    mlp_replica_group: ReplicaGroup = None,
) -> None:
    """Reject the shardings and shapes this kernel does not handle.

    Args:
        T_shard (int): Token slots per rank on entry/exit.
        H (int): Hidden dimension.
        E (int): Global routed expert count.
        E_L (int): Routed experts local to this rank.
        T_ep (int): Global token slots.
        T_sh (int): Token slots of the MLP group (per DP replica by default).
        ep_size (int): Expert-parallel degree / total rank count.
        tp_size (int): Shared-expert tensor-parallel degree.
        top_k (int): Routed experts per token.
        ep_replica_group (ReplicaGroup): The EP collective group.
        tp_replica_group (ReplicaGroup): The TP collective groups.
        shared_experts (int): Leading dim of the shared-expert weights; must be 1.
        shared_affinity_rows (int): Rows of ``shared_affinities``; must cover all ``T_sh`` slots.
        dense_only (bool): A dense layer, which has no router, no routed experts and no affinity read.
        router_shards (int): 1 for a replicated router, tp_size for a TP-sharded one.
        router_columns (int): Expert columns of this rank's router weights.
        mlp_tp (int): Ranks that split the dense MLP or the shared expert (0: tp_size).
        mlp_replica_group (ReplicaGroup): Their collective groups.

    Returns:
        None
    """
    # Single NeuronCore: every sub-kernel (norm, router, both expert calls) rejects a
    # multi-program launch, so fail here with a message about the layer rather than a sub-kernel.
    kernel_assert(
        nl.program_ndim() == 0 or nl.program_ndim() == 1,
        f"{_ERROR_PREFIX} only no specialization or specialization along one axis is supported, got "
        f"nl.program_ndim()={nl.program_ndim()}",
    )
    n_programs = nl.num_programs(axes=0) if nl.program_ndim() != 0 else 1
    kernel_assert(
        n_programs == 1,
        f"{_ERROR_PREFIX} runs on a single NeuronCore (LNC=1), got {n_programs} programs",
    )

    # The router emits one token per SBUF partition and does not tile T.
    kernel_assert(
        T_shard <= nl.tile_size.pmax,
        f"{_ERROR_PREFIX} T_shard must be at most {nl.tile_size.pmax}, got T_shard={T_shard}",
    )
    kernel_assert(
        H % (nl.tile_size.pmax * _q_width) == 0,
        f"{_ERROR_PREFIX} H must be a multiple of {nl.tile_size.pmax * _q_width}, got H={H}",
    )

    # The pre-quantized expert path needs T % 4 for the matmul free dim; the unquantized path's
    # T % 32 does not apply because the swizzle it comes from is skipped.
    kernel_assert(
        T_ep % _q_width == 0 and T_sh % _q_width == 0,
        f"{_ERROR_PREFIX} gathered token counts must be multiples of {_q_width}, got T_ep={T_ep}, T_sh={T_sh}",
    )

    # EP must partition the global experts exactly, or rank_id's affinity slice runs off the end.
    # Routed-only invariants. A dense layer has no router and no routed experts, so E and E_L are 0
    # by construction and neither check means anything -- `top_k <= E` would in fact reject every
    # dense layer for having no experts to route to.
    if not dense_only:
        kernel_assert(
            E_L * ep_size == E,
            f"{_ERROR_PREFIX} E_L * ep_size must equal E, got E_L={E_L}, ep_size={ep_size}, E={E}",
        )
        kernel_assert(
            top_k <= E,
            f"{_ERROR_PREFIX} top_k must be at most E, got top_k={top_k}, E={E}",
        )
        kernel_assert(
            router_shards in (1, tp_size) and router_columns * router_shards == E,
            f"{_ERROR_PREFIX} router_weights must hold all E={E} expert columns, or E / tp_size="
            f"{E // tp_size} (sharded over the TP group); got {router_columns}",
        )
        # The sharded router selects on [T_shard, E] logits assembled in SBUF (the same max8 groups of 8 as the
        # replicated one), and its matmul puts the TP group's rows on PSUM partitions.
        kernel_assert(
            router_shards == 1 or (E % 8 == 0 and T_shard * tp_size <= nl.tile_size.pmax),
            f"{_ERROR_PREFIX} the TP-sharded router needs E % 8 == 0 and T_shard * tp_size <= "
            f"{nl.tile_size.pmax}, got E={E}, T_shard * tp_size={T_shard * tp_size}",
        )
    kernel_assert(
        shared_experts == 1,
        f"{_ERROR_PREFIX} the shared expert is a single dense MLP, got {shared_experts} shared experts",
    )
    # The shared expert's affinity table is read as [T_sh, 1] by an indirect DMA that takes its row
    # count from T, not from the tensor, so an undersized table would read past the end silently.
    # A dense layer reads no affinity, and its MLP group may be wider than the shared expert's.
    kernel_assert(
        dense_only or shared_affinity_rows == T_sh,
        f"{_ERROR_PREFIX} shared_affinities must have T_sh={T_sh} rows, got {shared_affinity_rows}",
    )

    # The two ReduceScatter destinations only coincide when the TP groups are consecutive blocks
    # of tp_size ranks tiling one in-order EP group. Getting this wrong silently adds the shared
    # expert's output to the wrong tokens, so check the groups rather than trusting the caller.
    kernel_assert(
        ep_size % tp_size == 0,
        f"{_ERROR_PREFIX} ep_size must be a multiple of tp_size, got ep_size={ep_size}, tp_size={tp_size}",
    )
    kernel_assert(
        ep_replica_group._value == [list(range(ep_size))],
        f"{_ERROR_PREFIX} ep_replica_group must be one in-order group of all {ep_size} ranks, got "
        f"{ep_replica_group._value}",
    )
    kernel_assert(
        tp_replica_group._value == [list(range(d * tp_size, (d + 1) * tp_size)) for d in range(ep_size // tp_size)],
        f"{_ERROR_PREFIX} tp_replica_group must be consecutive blocks of {tp_size} ranks, got "
        f"{tp_replica_group._value}",
    )
    # The MLP group's ReduceScatter must also land on this rank's own T_shard slots.
    mlp_tp = mlp_tp or tp_size
    kernel_assert(
        ep_size % mlp_tp == 0
        and (
            mlp_replica_group is None
            or mlp_replica_group._value == [list(range(d * mlp_tp, (d + 1) * mlp_tp)) for d in range(ep_size // mlp_tp)]
        ),
        f"{_ERROR_PREFIX} mlp_replica_group must be consecutive blocks of {mlp_tp} ranks, got "
        f"{None if mlp_replica_group is None else mlp_replica_group._value}",
    )
