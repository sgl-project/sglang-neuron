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

"""One rank's GLM-5.2 MTP-layer MoE block on the draft's rows, collectives included.

The draft chain runs its MoE on one row per sequence that every rank already holds (``glm52_moe_rows_bf16``): every
rank norms the rows and routes them, runs its routed experts and its TP shard of the shared expert over all of them
with the tokens stationary, and one ``all_reduce(add)`` over the EP group completes both sums. The experts are FP8
with one fp32 factor per weight, which ``mlp_bf16`` folds into instructions it issues anyway; tokens stay bf16 and
accumulation fp32.

``router_topk`` reads ``x`` as ``[128, T, H/128]`` with consecutive H down the partition axis and uses it as the
stationary operand, so the layout matters. It is built from the normed ``[T, H]`` row with a single
``nisa.dma_transpose``; a load with the opposite H split would pass every assert and give wrong logits.

``router_bias`` stays fp32: the correction bias reaches ~15.6, and rounding it to bf16 changes which experts are
selected.
"""

import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from ..utils.common_types import ActFnType
from ..utils.kernel_assert import kernel_assert
from ..utils.trace_names import unique_name
from .attention.mla.mla_proj_primitives import load_gamma_broadcast, rmsnorm_into
from .mlp.mlp_bf16 import mlp_bf16
from .mlp.mlp_tkg_constants import MLPSharedWeightRing, weight_ring_geometry
from .router.affinity_masking import mask_expert_affinities
from .router.router_topk import router_topk
from .utils.sbuf_collectives import all_reduce_hbm

_ERROR_PREFIX = "[GLM5.2 MTP MoE Decode Layer DP/TP/EP bf16 Kernel]"

_P_MAX = 128
"""SBUF partition count, and the H tile the router's x layout puts on partitions."""

_LIVE_MAX_TOKENS = 32
"""Row ceiling of the token-stationary MLPs. ``mlp_bf16`` itself accepts up to 128 tokens, but above 32 it widens its
PE column tile (64) or drops the tiling; the chain holds one row per sequence."""

_RING_SLOTS = 8
"""The block's one weight ring (:class:`_RowsWeightRing`) cuts the default ring's bytes into this many slots: 8 slots
of 18 KB keep the 144 KB of a depth-3 ring in loads of 8 H tiles (gate/up) and 2 I tiles (down). Every load is still
a multiple of the 4-way column tiling, so the accumulation order is unchanged, and the PE starts on a load, and frees
its slot, a quarter of the way into what was one 48 KB load."""

# Buffer class for every HBM tensor that is an operand of a collective. At LNC=1 the RUNTIME rejects a
# collective whose sources are shared_hbm, so they are private_hbm.
_COLLECTIVE_HBM = nl.private_hbm


def _local_experts_on_rows(
    hidden_live: nl.NkiTensor,
    aff_live: nl.NkiTensor,
    routed_gate_up_weights: nl.NkiTensor,
    routed_down_weights: nl.NkiTensor,
    rank_id: nl.NkiTensor,
    out_bf16: nl.NkiTensor,
    fp8_scales: nl.NkiTensor,
    weight_ring,
) -> nl.NkiTensor:
    """This rank's local experts over ``T_live`` rows, tokens stationary, each scaled by its affinity.

    Each expert's ``mlp_bf16`` writes its rows straight into an SBUF tile, which the affinity scale reads in
    place: no HBM write and reload per expert.

    Args:
        hidden_live (nl.NkiTensor): [1, T_live, H] bf16 normed rows in HBM (``mlp_bf16``'s 3-D contract).
        aff_live (nl.NkiTensor): [T_live, E] bf16 the rows' dense affinity table in HBM.
        routed_gate_up_weights (nl.NkiTensor): [E_L, H, 2, I] float8_e4m3fn, gate and up interleaved on axis 2.
        routed_down_weights (nl.NkiTensor): [E_L, I, H] float8_e4m3fn.
        rank_id (nl.NkiTensor): [1, 1] uint32 EP rank, selecting affinity columns ``[E_L * r, E_L * (r + 1))``.
        out_bf16 (nl.NkiTensor): [T_live, H] bf16 SBUF tile. The last expert's accumulate writes its bf16 result
            here instead of into the fp32 sum (the same rounding as the fp32 sum followed by a cast, in one
            instruction fewer).
        fp8_scales (nl.NkiTensor): [E_L, 3] fp32 per-weight factors of the experts.
        weight_ring (_RowsWeightRing): The caller's weight ring, threaded through the experts' MLPs.

    Returns:
        nl.NkiTensor: ``out_bf16``, the sum over local experts.
    """
    T_live, H = hidden_live.shape[1], hidden_live.shape[2]
    E_L, _, _, I = routed_gate_up_weights.shape
    # [T_live, E_L]: this rank's expert columns, selected on device by rank_id. Widened to fp32: a
    # per-partition scalar operand must be fp32.
    aff_local_bf = mask_expert_affinities(aff_live, rank_id, E_L, T_live)
    aff_local = nl.ndarray((_P_MAX, E_L), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=aff_local[0:T_live, 0:E_L], src=aff_local_bf[0:T_live, 0:E_L])

    # ═══════ Each local expert as a dense MLP over the live tokens, POST_SCALE by its affinity ═══════
    acc = nl.ndarray((_P_MAX, H), dtype=nl.float32, buffer=nl.sbuf)
    for e in range(E_L):
        # The last expert's sum goes straight to bf16.
        dst = out_bf16[0:T_live, 0:H] if e == E_L - 1 else acc[0:T_live, 0:H]
        # The fused [H, 2, I] expert as a [H, 2I] row: gate is columns [0, I), up is [I, 2I).
        w2 = routed_gate_up_weights[e].reshape((H, 2 * I))
        part_sb = nl.ndarray((T_live, H), dtype=nl.bfloat16, buffer=nl.sbuf)
        mlp_bf16(
            hidden_tensor=hidden_live,
            gate_weights=w2[:, 0:I],
            up_weights=w2[:, I : 2 * I],
            down_weights=routed_down_weights[e],
            activation_fn=ActFnType.SiLU,
            output_dtype=nl.bfloat16,
            output=part_sb,
            tensor_scales=fp8_scales[e : e + 1, 0:3],
            weight_ring=weight_ring.next_call(),
        )
        if e == 0:
            nisa.tensor_scalar(dst=dst, data=part_sb[0:T_live, 0:H], op0=nl.multiply, operand0=aff_local[0:T_live, 0:1])
        else:
            nisa.scalar_tensor_tensor(
                dst=dst,
                data=part_sb[0:T_live, 0:H],
                op0=nl.multiply,
                operand0=aff_local[0:T_live, e : e + 1],
                op1=nl.add,
                operand1=acc[0:T_live, 0:H],
            )
    return out_bf16


class _RowsWeightRing:
    """The rows MoE's one weight ring of ``mlp_bf16``'s slots and its weight DMAs, threaded through the block's MLP
    calls in trace order, its rotation continued from call to call: each call appends its weight DMAs to ``dmas``, so
    the next call starts at ring position ``len(dmas)`` and its first load takes the slot that frees first. A ring per
    call made an expert's first load wait for the slot the previous expert frees last."""

    def __init__(self, weight_dtype):
        depth, slot_elems = weight_ring_geometry(weight_dtype)
        slot_elems = depth * slot_elems // _RING_SLOTS
        self.slots = [nl.ndarray((_P_MAX, slot_elems), dtype=weight_dtype, buffer=nl.sbuf) for _ in range(_RING_SLOTS)]
        self.dmas = []

    def next_call(self) -> MLPSharedWeightRing:
        """The ``weight_ring`` of the next ``mlp_bf16`` call."""
        return MLPSharedWeightRing(slots=self.slots, position=len(self.dmas), dmas=self.dmas)


def group_rows_index(rank_id: nl.NkiTensor, tp_size: int, rows_per_group: int) -> nl.NkiTensor:
    """[rows_per_group, 1] int32 SBUF: rows ``[g * R, (g + 1) * R)`` of group ``g = rank // tp_size``, for any
    group of ``tp_size`` consecutive ranks (the shared expert's TP group, a token slice, an o_proj block).

    Built on device from ``rank_id``, because one NEFF serves every rank (see the MTP draft chain).
    """
    kernel_assert(tp_size & (tp_size - 1) == 0, f"{_ERROR_PREFIX} tp_size must be a power of two, got {tp_size}")
    R = rows_per_group
    rank_i = nl.ndarray((_P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
    # uint32 -> int32 has the same bits for any rank id, so the DMA's reinterpretation is exact.
    nisa.dma_copy(dst=rank_i[0:R, 0:1], src=rank_id[0:1, 0:1].view(nl.int32).broadcast(0, R))
    nisa.tensor_scalar(
        dst=rank_i[0:R, 0:1], data=rank_i[0:R, 0:1], op0=nl.right_shift, operand0=tp_size.bit_length() - 1
    )
    group_f = nl.ndarray((_P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=group_f[0:R, 0:1], src=rank_i[0:R, 0:1])
    lane_i = nl.ndarray((_P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(lane_i[0:R, 0:1], pattern=[[0, 1]], offset=0, channel_multiplier=1)
    lane_f = nl.ndarray((_P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=lane_f[0:R, 0:1], src=lane_i[0:R, 0:1])
    # g * R + lane, exact in fp32 for every row index this layer can hold.
    nisa.scalar_tensor_tensor(
        dst=lane_f[0:R, 0:1],
        data=group_f[0:R, 0:1],
        op0=nl.multiply,
        operand0=float(R),
        op1=nl.add,
        operand1=lane_f[0:R, 0:1],
    )
    rows_i = nl.ndarray((_P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=rows_i[0:R, 0:1], src=lane_f[0:R, 0:1])
    return rows_i


def glm52_moe_rows_bf16(
    h_rows: nl.NkiTensor,
    gamma: nl.NkiTensor,
    router_weights: nl.NkiTensor,
    router_bias: nl.NkiTensor,
    routed_gate_up_weights: nl.NkiTensor,
    routed_down_weights: nl.NkiTensor,
    shared_gate_up_weights: nl.NkiTensor,
    shared_down_weights: nl.NkiTensor,
    rank_id: nl.NkiTensor,
    ep_replica_group: ReplicaGroup,
    ep_size: int,
    tp_size: int,
    top_k: int,
    eps: float,
    routed_scaling_factor: float,
    routed_fp8_scales: nl.NkiTensor,
    shared_fp8_scales: nl.NkiTensor,
    weight_dmas_out: list,
) -> nl.NkiTensor:
    """The MoE block over ``S`` rows that EVERY rank already holds, with one AllReduce and no other collective.

    For the MTP draft chain, where only one row per sequence is live (``S`` = 32 sequences), this replaces
    the token-sharded layer's two ``[T_ep, H]`` AllGathers, its TP AllGather and its two ReduceScatters:

    * every rank norms the ``S`` rows and runs the router on them itself (both are per-row and cheap);
    * every rank runs its ``E_L`` routed experts over all ``S`` rows, tokens stationary, each scaled by its affinity
      column (zero where the row did not pick the expert);
    * every rank runs its TP shard of the shared expert over all ``S`` rows too, and keeps only the rows
      ``[g * R, (g + 1) * R)`` of its TP group ``g`` (``tp_size`` consecutive ranks), ``R = S / (ep_size /
      tp_size)``, so each row's shared expert is summed over exactly one TP group. The MLP keeps the tokens
      stationary, so 32 rows cost the same instructions as 2, and a 0/1 row mask replaces the gather of
      the group's rows and the scatter of their sum;
    * one ``all_reduce(add)`` over the EP group of the ``[S, H]`` partials completes both sums.

    The rows stay on chip up to the AllReduce: the input is the residual add's SBUF tile and each expert's MLP
    writes into SBUF. The normed rows also go to HBM once, because the router's transposing load and the MLP's input
    load read HBM.

    Args:
        h_rows (nl.NkiTensor): [S, H] bf16 the post-attention residual rows, the SAME on every rank, in SBUF
            (or HBM).
        gamma (nl.NkiTensor): [1, H] ``post_attention_layernorm.weight``.
        router_weights (nl.NkiTensor): [H, E] bf16 ``mlp.gate.weight`` transposed, whole on every rank.
        router_bias (nl.NkiTensor): [1, E] fp32 ``e_score_correction_bias``.
        routed_gate_up_weights (nl.NkiTensor): [E_L, H, 2, I] float8_e4m3fn this rank's routed experts.
        routed_down_weights (nl.NkiTensor): [E_L, I, H] float8_e4m3fn.
        shared_gate_up_weights (nl.NkiTensor): [1, H, 2, I_sh] float8_e4m3fn this rank's TP shard.
        shared_down_weights (nl.NkiTensor): [1, I_sh, H] float8_e4m3fn this rank's TP shard.
        rank_id (nl.NkiTensor): [1, 1] uint32 EP rank.
        ep_replica_group (ReplicaGroup): All ``ep_size`` ranks in order.
        ep_size, tp_size, top_k, eps, routed_scaling_factor: The EP and TP degrees, the router's top-k, the norm's
            epsilon and the routed experts' scaling factor.
        routed_fp8_scales, shared_fp8_scales (nl.NkiTensor): [E_L, 3] / [1, 3] fp32 FP8 per-weight factors.
        weight_dmas_out (list): Receives the experts' weight DMA instructions in trace order, the last one the shared
            expert's down load: the caller orders its lm_head prefetch after it, so the prefetch stops sharing the
            weight queue with the experts.

    Returns:
        nl.NkiTensor: [S, H] bf16 private HBM, the AllReduce's destination (``all_reduce_hbm``): the MoE output of
        every row, the same on every rank, for the caller to load where and how it consumes it.
    """
    S, H = h_rows.shape
    E = router_weights.shape[-1]
    E_L = routed_gate_up_weights.shape[0]
    I_sh = shared_gate_up_weights.shape[3]
    H1 = H // _P_MAX
    n_groups = ep_size // tp_size
    R = S // n_groups
    kernel_assert(
        S <= _LIVE_MAX_TOKENS and E == E_L * ep_size and S % n_groups == 0 and H % _P_MAX == 0,
        f"{_ERROR_PREFIX} rows MoE needs S <= {_LIVE_MAX_TOKENS}, E == E_L * ep_size and S divisible by the "
        f"{n_groups} TP groups, got S={S}, E={E}, E_L={E_L}, ep_size={ep_size}",
    )

    # ═══════ RMSNorm and the router, on every row, on every rank ═══════
    if h_rows.buffer == nl.sbuf:
        h_sb = h_rows
    else:
        h_sb = nl.ndarray((S, H), dtype=h_rows.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=h_sb, src=h_rows[0:S, 0:H])
    # Out of place: the caller adds this block's output back onto h_rows.
    x_sb = nl.ndarray((S, H), dtype=h_rows.dtype, buffer=nl.sbuf)
    rmsnorm_into(x_sb, h_sb, load_gamma_broadcast(gamma, S), S, eps)
    normed = nl.ndarray((1, S, H), dtype=h_rows.dtype, buffer=_COLLECTIVE_HBM, name=unique_name("moe_rows_normed"))
    nisa.dma_copy(dst=normed[0], src=x_sb)
    normed_2d = normed.reshape((S, H))

    # The router's [128, S, H/128] x layout, by one dma_transpose (the module docstring).
    x_router_sb = nl.ndarray((_P_MAX, S, H1), dtype=nl.bfloat16, buffer=nl.sbuf)
    normed_transpose_src = normed_2d.reshape((S * H1, _P_MAX)).expand_dim(dim=1).expand_dim(dim=1)
    x_router_dst = x_router_sb.flatten_dims(start_dim=1, end_dim=2).expand_dim(dim=1).expand_dim(dim=1)
    nisa.dma_transpose(dst=x_router_dst, src=normed_transpose_src)
    affinities = nl.ndarray((S, E), dtype=nl.bfloat16, buffer=_COLLECTIVE_HBM, name=unique_name("moe_rows_aff"))
    expert_index = nl.ndarray((S, top_k), dtype=nl.int32, buffer=nl.shared_hbm, name=unique_name("moe_rows_eidx"))
    router_topk(
        x=x_router_sb,
        w=router_weights,
        w_bias=router_bias,
        expert_affinities=affinities,
        expert_index=expert_index,
        k=top_k,
        routed_scaling_factor=routed_scaling_factor,
    )

    ring = _RowsWeightRing(routed_down_weights.dtype)

    # ═══════ Routed experts: this rank's E_L experts over all S rows, summed into bf16 ═══════
    routed_sb = nl.ndarray((S, H), dtype=nl.bfloat16, buffer=nl.sbuf)
    _local_experts_on_rows(
        normed,
        affinities,
        routed_gate_up_weights,
        routed_down_weights,
        rank_id,
        out_bf16=routed_sb,
        fp8_scales=routed_fp8_scales,
        weight_ring=ring,
    )

    # ═══════ Shared expert: this rank's TP shard over all S rows, kept on this TP group's R rows ═══════
    w2 = shared_gate_up_weights[0].reshape((H, 2 * I_sh))
    sh_sb = nl.ndarray((S, H), dtype=nl.bfloat16, buffer=nl.sbuf)
    mlp_bf16(
        hidden_tensor=normed,
        gate_weights=w2[:, 0:I_sh],
        up_weights=w2[:, I_sh : 2 * I_sh],
        down_weights=shared_down_weights[0],
        activation_fn=ActFnType.SiLU,
        output_dtype=nl.bfloat16,
        output=sh_sb,
        tensor_scales=shared_fp8_scales[0:1, 0:3],
        weight_ring=ring.next_call(),
        # gate and up from the fused [H, 2 I_sh] weight: each partition reads whole 512-element rows, one contiguous
        # run per load, instead of two loads of 256-element runs at a 512 stride
        gate_up_fused=w2,
    )
    # Every load appended its DMA (the ring's positions depend on it).
    kernel_assert(len(ring.dmas) > 0, f"{_ERROR_PREFIX} the rows MoE recorded no weight DMA")
    weight_dmas_out.extend(ring.dmas)
    # partial = mask * shared + routed, in fp32 with one bf16 rounding: on the group's rows the sum the
    # gather / add / scatter formed, elsewhere exactly the routed partial (+0 changes nothing).
    keep = _group_rows_mask(rank_id, tp_size, R, S)
    partial_sb = nl.ndarray((S, H), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=partial_sb,
        data=sh_sb,
        op0=nl.multiply,
        operand0=keep[0:S, 0:1],
        op1=nl.add,
        operand1=routed_sb,
    )

    # ═══════ One AllReduce completes the routed sum over EP and the shared sum over TP ═══════
    return all_reduce_hbm(partial_sb, ep_replica_group, nl.add, "moe_rows_out")


def _group_rows_mask(rank_id: nl.NkiTensor, tp_size: int, rows_per_group: int, rows: int) -> nl.NkiTensor:
    """[P_MAX, 1] fp32 SBUF, rows ``[0, rows)``: 1.0 on the rows of group ``g = rank // tp_size`` (``group_rows_index``'s
    rows ``[g * R, (g + 1) * R)``), 0.0 elsewhere. Built on device from ``rank_id``."""
    R = rows_per_group
    kernel_assert(tp_size & (tp_size - 1) == 0, f"{_ERROR_PREFIX} tp_size must be a power of two, got {tp_size}")
    # g on every row, as in group_rows_index: uint32 -> int32 keeps the bits of any rank id.
    group_i = nl.ndarray((_P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=group_i[0:rows, 0:1], src=rank_id[0:1, 0:1].view(nl.int32).broadcast(0, rows))
    nisa.tensor_scalar(
        dst=group_i[0:rows, 0:1], data=group_i[0:rows, 0:1], op0=nl.right_shift, operand0=tp_size.bit_length() - 1
    )
    lo_f = nl.ndarray((_P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=lo_f[0:rows, 0:1], src=group_i[0:rows, 0:1])
    lane_i = nl.ndarray((_P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(lane_i[0:rows, 0:1], pattern=[[0, 1]], offset=0, channel_multiplier=1)
    lane_f = nl.ndarray((_P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=lane_f[0:rows, 0:1], src=lane_i[0:rows, 0:1])
    # d = lane - g * R, exact in fp32; the row is the group's iff 0 <= d < R.
    d_f = nl.ndarray((_P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=d_f[0:rows, 0:1],
        data=lo_f[0:rows, 0:1],
        op0=nl.multiply,
        operand0=-float(R),
        op1=nl.add,
        operand1=lane_f[0:rows, 0:1],
    )
    ge = nl.ndarray((_P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=ge[0:rows, 0:1], data=d_f[0:rows, 0:1], op0=nl.greater_equal, operand0=0.0)
    keep = nl.ndarray((_P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=keep[0:rows, 0:1], data=d_f[0:rows, 0:1], op0=nl.less, operand0=float(R))
    nisa.tensor_tensor(dst=keep[0:rows, 0:1], data1=keep[0:rows, 0:1], data2=ge[0:rows, 0:1], op=nl.multiply)
    return keep
