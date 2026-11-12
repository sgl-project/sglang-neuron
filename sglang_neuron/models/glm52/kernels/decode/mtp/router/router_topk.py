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

"""
This file implements the router top-K kernel for the GLM-5.2 Mixture of Experts (MoE) block.

It computes the router logits (x @ w), applies a sigmoid, adds the expert-score correction bias to a
separate copy used for SELECTION ONLY, selects the top-K experts per token, then gathers those
experts' PRE-bias scores, L1-normalizes them, scales by routed_scaling_factor and writes the dense
[T, E] affinity tensor.
"""

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa import engine

from ...utils.kernel_assert import kernel_assert

P_MAX = 128
F_MAX = 512
PE_COLUMN_TILE_32 = 32  # PE array column tile size for T <= 32
PE_COLUMN_TILE_64 = 64  # PE array column tile size for 32 < T <= 64
PE_COLUMN_TILE_128 = 128  # PE array column tile size (full width, disables tiling)

# Epsilon added to the top-K score sum before the reciprocal, matching the reference router's
# `denominator = topk_weights.sum(-1, keepdim=True) + 1e-20`. Guards a token whose selected scores
# all underflow to zero from producing inf affinities.
_L1_NORM_EPS = 1e-20

_ERROR_PREFIX = "[Router TopK Kernel]"


@nki.jit
def router_topk(
    x: nl.NkiTensor,
    w: nl.NkiTensor,
    w_bias: nl.NkiTensor,
    expert_affinities: nl.NkiTensor,
    expert_index: nl.NkiTensor,
    k: int,
    routed_scaling_factor: float = 1.0,
):
    """
    Router top-K kernel for the GLM-5.2 Mixture of Experts (MoE) block.

    Computes the router logits (x @ w), applies a sigmoid to get the expert scores, adds w_bias to a
    separate copy used only to choose the top-K experts, gathers the chosen experts' PRE-bias scores,
    L1-normalizes them, scales by routed_scaling_factor and writes the dense [T, E] affinities.

    The kernel runs on a single NeuronCore (LNC=1); launching it with more than one program is
    rejected because there is no token shard and every program would write the same output rows.
    'x' is always read from SBUF as [128, T, H/128] with consecutive H elements down the partition
    dim, which is what the fused MX RMSNorm emits, and both outputs are always written to HBM.

    Intended for token generation: T <= 128 (one token per SBUF partition), expert counts E <= 512,
    hidden dimensions H that are multiples of 128, and K <= 8 top experts per token.

    Dimensions:
        T: Number of tokens
        H: Hidden dimension size
        E: Number of experts
        K: Number of top experts to select per token

    Args:
        x (nl.NkiTensor): Input tensor [128, T, H/128] in SBUF, partition dim holding consecutive H.
        w (nl.NkiTensor): Weight tensor [H, E] in HBM.
        w_bias (nl.NkiTensor): Bias tensor [1, E] or [E] in HBM. Required.
        expert_affinities (nl.NkiTensor): Output expert affinities [T, E] in HBM. Written in
                        full: the selected experts hold their normalized, scaled affinity and every
                        other expert column is zero.
        expert_index (nl.NkiTensor): Output expert indices [T, K] in HBM.
        k (int): Number of top experts to select (must be in [1, 8]).
        routed_scaling_factor (float): Final multiplier on the L1-normalized affinities. Defaults to
                        1.0 (no scaling).

    Returns:
        outputs (list): [expert_index, expert_affinities, expert_affinities_sb] -- the two HBM outputs
            and the [T, E] SBUF tile the affinities were stored from.

    Notes:
        - w_bias biases the SELECTION ONLY. It is added to the sigmoid scores, not to the logits, and
          the returned weights are gathered from the PRE-bias scores. This is what makes the
          load-balancing bias affect which experts win without distorting how much they contribute.
        - The gather and the scatter are one step: multiplying the pre-bias scores by a one-hot mask
          over the selected experts leaves the selected scores in place and zeroes every other
          column, which is already the dense [T, E] output.
        - The L1 normalization divides by (sum + 1e-20); the epsilon and routed_scaling_factor are
          both folded into the [T, 1] reciprocal so the full-width multiply stays single-op.
        - expert_index holds the K selected experts in nisa.max8 / nisa.nc_find_index8 order, which
          is not sorted by score. Consumers must treat it as a set.

    Pseudocode:
        # Load inputs
        w_sb = load_w_from_hbm(w)
        bias_sb = load_and_broadcast_bias(w_bias)

        # Compute router logits
        router_logits_psum = zeros()
        for h_tile in range(num_h_tiles):
            router_logits_psum += matmul(x[..., h_tile], w_sb[:, h_tile, :])
        router_logits_sb = merge_column_tiles(router_logits_psum)

        # ACT1: scores, plus a biased copy for selection only
        scores = sigmoid(router_logits_sb)
        scores_for_choice = scores + bias_sb

        # Top-K selection -- indices only; the biased values are discarded
        top8_values = max8(scores_for_choice)
        expert_index = find_index8(scores_for_choice, top8_values)[:k]

        # Gather the PRE-bias scores of the selected experts, which is also the scatter
        onehot = sum(expert_ids == expert_index[i] for i in range(k))
        gathered = scores * onehot

        # Norm and scale
        recip = routed_scaling_factor / (reduce_sum(gathered) + 1e-20)
        expert_affinities = gathered * recip

        return [expert_index, expert_affinities]
    """
    # The following code is split into sections correlated to the pseudo-code above

    """Get input shapes and validate."""

    H, E = w.shape

    # Single NeuronCore only: there is no token shard, so a second program would recompute the whole
    # result and race with the first one on the expert_affinities zero-fill / scatter in shared HBM.
    n_programs = nl.num_programs(axes=0) if nl.program_ndim() != 0 else 1
    kernel_assert(
        n_programs == 1,
        f"{_ERROR_PREFIX} router_topk runs on a single NeuronCore (LNC=1), got {n_programs} programs",
    )

    # Buffer residency is pinned: 'x' is read from SBUF, and both outputs are written to HBM. The
    # HBM-'x' loader and the SBUF-output assembly are gone, so reject those tensors here rather than
    # letting them reach a matmul / indirect DMA that cannot address them.
    kernel_assert(x.buffer == nl.sbuf, f"{_ERROR_PREFIX} 'x' must be in SBUF, got buffer {x.buffer}")
    kernel_assert(
        expert_index.buffer != nl.sbuf,
        f"{_ERROR_PREFIX} expert_index must be in HBM, got buffer {expert_index.buffer}",
    )
    kernel_assert(
        expert_affinities.buffer != nl.sbuf,
        f"{_ERROR_PREFIX} expert_affinities must be in HBM, got buffer {expert_affinities.buffer}",
    )

    # Validate 'x'. It is always in SBUF with layout [128, T, H/128].
    kernel_assert(len(x.shape) == 3, f"Expect the input 'x' SBUF tensor to be 3-D, got shape of {x.shape}")
    kernel_assert(x.shape[0] == P_MAX, f"Expect input x.shape[0] to be partition-dim max {P_MAX} but got {x.shape[0]}")
    # Sanity check 'x' dim-2 (3rd dim) is H/pmax
    kernel_assert(x.shape[2] == H // P_MAX, f"Expect x.shape[2] to be H/pmax ({H // P_MAX}) but got {x.shape[2]}")
    T = x.shape[1]

    # Every tensor below keeps one token per SBUF partition, so T cannot exceed the partition dim.
    kernel_assert(T <= P_MAX, f"{_ERROR_PREFIX} T ({T}) must be <= pmax ({P_MAX})")

    # E is expected to be less than max moving free-dim
    kernel_assert(E <= F_MAX, f"E ({E}) must be <= gemm_moving_fmax ({F_MAX})")
    # max8 / nc_find_index8 operate on chunks of 8, so E must fill whole chunks.
    kernel_assert(E % 8 == 0, f"{_ERROR_PREFIX} E ({E}) must be a multiple of 8")

    # The bias is always fused into the router logits.
    kernel_assert(w_bias != None, f"{_ERROR_PREFIX} w_bias is required")
    # w_bias should be [1,E], this will assert if we cannot do the reshape
    w_bias = w_bias.reshape((1, E))

    # We tile on H, which is the contraction dim = partition dim.
    kernel_assert(H % P_MAX == 0, f"H ({H}) must be a multiple of pmax ({P_MAX})")
    num_h_tiles = H // P_MAX

    # Check that input data types match
    kernel_assert(x.dtype == w.dtype, f"x dtype ({x.dtype}) must match w dtype ({w.dtype})")

    input_dtype = x.dtype

    """Compute router-logits (x.T @ w) + w_bias."""

    """
    TensorEngine/PEArray column tiling setup.

    The 'x' tensor's T is on the free-dim in SBUF. If T<128 then some of the PEArray columns
    are unused. Hence we engage the column-tiling feature to split the array in multiple tiles
    (column-wise), each of which can execute an independent matmul in parallel.
    """

    # Column tile width must be multiple of 32 so round up to the nearest 32 depending on T.
    if T <= PE_COLUMN_TILE_32:
        pe_array_column_tiling_size = PE_COLUMN_TILE_32
    elif T <= PE_COLUMN_TILE_64:
        pe_array_column_tiling_size = PE_COLUMN_TILE_64
    else:
        pe_array_column_tiling_size = PE_COLUMN_TILE_128

    num_pe_array_column_tiles = PE_COLUMN_TILE_128 // pe_array_column_tiling_size
    tile_size = (P_MAX, pe_array_column_tiling_size)

    # Result of the x@w matmul, shape [T,E]
    router_logits_sb = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.sbuf)

    """Load from HBM."""

    # Load 'w'. See the comments in router_topk_input_w_load() to understand the HBM-to-SBUF
    # layout/access pattern; it mirrors the H-stride of 'x' in SBUF.
    w_sb = router_topk_input_w_load(w)

    """
    Load the bias vector, shape [1,E]. Broadcast into a tensor of shape [T,E] (i.e. every row
    contains a copy of w_bias) using TensorE/matmul against a row of ones.
    """
    router_logits_bias_vector_sb = nl.ndarray((1, E), dtype=nl.float32, buffer=nl.sbuf)
    router_logits_bias_broadcasted_sb = nl.ndarray(
        shape=(T, E), dtype=router_logits_bias_vector_sb.dtype, buffer=nl.sbuf
    )
    nisa.dma_copy(
        dst=router_logits_bias_vector_sb,
        src=w_bias,
    )
    ones_mask = nl.ndarray((1, T), dtype=router_logits_bias_vector_sb.dtype, buffer=nl.sbuf)
    nisa.memset(dst=ones_mask, value=1.0, engine=engine.gpsimd)
    bias_bc_psum = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(
        dst=bias_bc_psum,
        stationary=ones_mask[...],
        moving=router_logits_bias_vector_sb[...],
        is_stationary_onezero=True,
    )
    nisa.tensor_copy(dst=router_logits_bias_broadcasted_sb, src=bias_bc_psum[...], engine=engine.scalar)

    """Tiled matmul."""

    """
    Initialize PSUM buffer.

    We allocate [pmax, E] (i.e. the full pmax) because each column-tile writes into
    a separate p-dim range. If T < pmax we'll use an access pattern to get the correct data.
    """
    router_logits_psum = nl.ndarray((P_MAX, E), nl.float32, buffer=nl.psum)

    for h_tile_idx in range(num_h_tiles):
        # Logically we are multiplying x @ w = [T,H] @ [H,E] = [Hp,Tf] @ [Hp,Ef] (using P/F notation).
        w_tile_sb = w_sb[:, h_tile_idx, :]
        # x = [128, T, num_h_tiles], so one H-tile of all T tokens is a [128, T] slice.
        x_tile_sb = x.select(dim=2, index=h_tile_idx)

        # column-tiling setup
        current_column_tile_index = h_tile_idx % num_pe_array_column_tiles
        # Calculate the starting column number for this slot (it's used to set the tile_position in nc_matmul)
        current_column_tile_column_offset = current_column_tile_index * pe_array_column_tiling_size
        tile_position = (0, current_column_tile_column_offset)

        # matmul ([P,F] notation): x_tile_sb [pmax, T] @ w_tile_sb [pmax, E] = router_logits_psum [T, E]
        nisa.nc_matmul(
            dst=router_logits_psum[nl.ds(current_column_tile_column_offset, T), :],
            stationary=x_tile_sb,
            moving=w_tile_sb,
            tile_position=tile_position,
            tile_size=tile_size,
        )

    """Evict the logits from PSUM."""

    """
    Copy the router logits from PSUM to SBUF. The bias is NOT applied here: e_score_correction_bias
    biases only the expert SELECTION, and it is added to the sigmoid SCORES below, not to the logits.
    Tensor shapes:
    - router_logits_sb: [T, E]
    - router_logits_psum: [pmax, E]
    """
    nisa.tensor_copy(dst=router_logits_sb, src=router_logits_psum[:T, :])

    """Merge column tiles."""
    # Accumulate the remaining column-tile results. This loop does not execute when column-tiling is
    # disabled (i.e. num_pe_array_column_tiles==1). Only accumulate column tiles that received matmul
    # data: when num_h_tiles < num_pe_array_column_tiles, some column tile slots are empty (never
    # written by a matmul), so we must not accumulate them.
    num_active_column_tiles = min(num_h_tiles, num_pe_array_column_tiles)
    for column_tile_idx in range(1, num_active_column_tiles):
        current_column_tile_column_offset = column_tile_idx * pe_array_column_tiling_size
        nisa.tensor_tensor(
            dst=router_logits_sb,
            data1=router_logits_sb[...],
            data2=router_logits_psum[nl.ds(current_column_tile_column_offset, T), :],
            op=nl.add,
        )

    """ACT1 -- sigmoid the router logits, then bias a SEPARATE copy for selection."""

    # scores = sigmoid(logits). 'Full' means the full E. These are the PRE-bias scores, and they are
    # what the returned affinities are gathered from.
    expert_affinities_full_sb = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=expert_affinities_full_sb, op=nl.sigmoid, data=router_logits_sb[...])

    # scores_for_choice = scores + e_score_correction_bias. This drives the top-K SELECTION ONLY --
    # aux-loss-free load balancing biases which experts win, never how much they contribute, so the
    # returned weights come from expert_affinities_full_sb above and never from this tensor.
    scores_for_choice_sb = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=scores_for_choice_sb,
        data1=expert_affinities_full_sb[...],
        data2=router_logits_bias_broadcasted_sb[...],
        op=nl.add,
    )

    """topK."""

    """
    Top-K operation finds the largest K values in each partition along with their indexes.

    Selection reads scores_for_choice_sb (biased); only the INDICES are kept. The values that come
    back from max8 are biased and are deliberately discarded.
    """

    # Simplify by setting max K=8.
    kernel_assert(k >= 1, f"K ({k}) must be >= 1")
    kernel_assert(k <= 8, f"K ({k}) must be <= 8")

    # The nisa instructions for topK operate in chunks of 8.
    # Get the top 8 values, in each partition (each token). Keep this around for nc_find_index8.
    router_indexes_topk_sb = nl.ndarray((T, k), dtype=nl.uint32, buffer=nl.sbuf)

    tmp_buffer = nl.ndarray((T, 8), dtype=router_indexes_topk_sb.dtype, buffer=nl.sbuf)

    scores_for_choice_top8_sb = nl.ndarray((T, 8), dtype=scores_for_choice_sb.dtype, buffer=nl.sbuf)
    nisa.max8(dst=scores_for_choice_top8_sb, src=scores_for_choice_sb[...])  # returns [T,8]

    # Now get the indexes (locations) of those top-8 values.
    # returns [T,8], take the [T,K] slice
    nisa.nc_find_index8(
        dst=tmp_buffer,
        data=scores_for_choice_sb[...],
        vals=scores_for_choice_top8_sb,
    )
    nisa.tensor_copy(dst=router_indexes_topk_sb, src=tmp_buffer[:, :k])

    """Store expert_index."""

    # copy from router_indexes_topk_sb (SBUF [T,k]) to expert_index (HBM [T,k])
    nisa.dma_copy(src=router_indexes_topk_sb, dst=expert_index)

    """Gather + Norm + Scale -- build the dense affinities from the PRE-bias scores."""

    """
    The returned affinity of a selected expert is its PRE-bias sigmoid score, L1-normalized over the
    token's selected experts and scaled by routed_scaling_factor.

    A one-hot mask over E does the gather and the scatter in one step: multiplying the pre-bias
    scores by a mask that is 1 exactly at the K selected experts leaves the selected scores in place
    and zeroes every other expert column, which is precisely the dense [T, E] output shape. That
    removes the need for a separate indirect-DMA scatter (and its HBM zero-fill).
    """
    topk_idx_fp32_sb = nl.ndarray((T, k), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=topk_idx_fp32_sb, src=router_indexes_topk_sb, engine=nisa.scalar_engine)

    # expert_ids[t, e] == e, so an equality test against a selected index marks that expert's column.
    expert_ids_sb = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=expert_ids_sb, pattern=[[1, E]], offset=0, channel_multiplier=0)

    onehot_sb = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=onehot_sb, value=0.0)
    for k_idx in range(k):
        # Fuse the equality test and the running accumulate into ONE Vector op:
        # onehot = (expert_ids == topk_idx[k]) + onehot.
        nisa.scalar_tensor_tensor(
            dst=onehot_sb,
            data=expert_ids_sb,
            op0=nl.equal,
            operand0=topk_idx_fp32_sb[:, k_idx : k_idx + 1],
            op1=nl.add,
            operand1=onehot_sb,
        )

    # gathered[t, e] = scores[t, e] if e is selected else 0 -- the gather AND the scatter.
    gathered_sb = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=gathered_sb, data1=expert_affinities_full_sb[...], data2=onehot_sb, op=nl.multiply)

    # L1 normalize over the selected experts, then apply routed_scaling_factor. Both the epsilon and
    # the scale fold into the [T, 1] reciprocal, so the full-width affinity multiply stays single-op.
    sum_of_scores_sb = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=sum_of_scores_sb, op=nl.add, data=gathered_sb[...], axis=1, keepdims=True)
    nisa.tensor_scalar(
        dst=sum_of_scores_sb,
        data=sum_of_scores_sb[...],
        op0=nl.add,
        operand0=_L1_NORM_EPS,
        engine=nisa.scalar_engine,
    )
    nisa.reciprocal(dst=sum_of_scores_sb, data=sum_of_scores_sb[...])
    nisa.tensor_scalar(
        dst=sum_of_scores_sb,
        data=sum_of_scores_sb[...],
        op0=nl.multiply,
        operand0=float(routed_scaling_factor),
        engine=nisa.scalar_engine,
    )

    expert_affinities_sb = nl.ndarray((T, E), dtype=input_dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=expert_affinities_sb,
        data=gathered_sb[...],
        op0=nl.multiply,
        operand0=sum_of_scores_sb[...],
    )
    nisa.dma_copy(dst=expert_affinities, src=expert_affinities_sb)

    """Return. The SBUF copy of the affinities is returned too: a caller can hang on-chip work off it
    (a true data dependency on the router's end) without waiting for the HBM store."""

    return [expert_index, expert_affinities, expert_affinities_sb]


def router_topk_input_w_load(w: nl.NkiTensor) -> nl.NkiTensor:
    """
    Load weight tensor w from HBM to SBUF with the layout that matches x's [128, T, H/128].

    Performs DMA transfer from HBM to SBUF with a layout conversion that matches the H-dimension
    stride pattern of the x tensor in SBUF (consecutive H elements down the partition dim), enabling
    efficient matmul operations.

    Args:
        w (nl.NkiTensor): Weight tensor [H, E] in HBM

    Returns:
        w_sb (nl.NkiTensor): Weight tensor in SBUF [128, H/128, E], each SBUF column holding
                          consecutive H elements.

    Notes:
        - H must be a multiple of 128
        - The layout must match the x tensor layout for a correct matmul contraction
    """
    H, E = w.shape
    num_h_tiles = H // P_MAX

    # HBM tensor shape [H,E] reshaped (new view) as [num_h_tiles, 128, E].
    # Divide the H dimension into num_h_tiles groups of 128 rows.
    #
    # H-dim ↓     E columns →
    # ┌─────────────────────────────────┐
    # │ H0          H-tile 0            │
    # │                                 │
    # │                                 │
    # │ H127                            │
    # ├─────────────────────────────────┤
    # │ H128        H-tile 1            │
    # │                                 │
    # │                                 │
    # │ H255                            │
    # ├─────────────────────────────────┤
    # │ H256        H-tile 2            │
    # │                                 │
    # │                                 │
    # │ H383                            │
    # ├─────────────────────────────────┤
    # │              ...                │
    # ├─────────────────────────────────┤
    # │ H_(num_h_tiles-1)*128           │
    # │         H-tile num_h_tiles-1    │
    # │                                 │
    # │ H-1                             │
    # └─────────────────────────────────┘
    #
    # SBUF layout [128, num_h_tiles, E]
    #
    #   ←─────E─────→     ←─────E─────→     ←─────E─────→             ←─────E─────→
    # ┌─────────────────┬─────────────────┬─────────────────┬───┬─────────────────────────┐
    # │ H0              │ H128            │ H256            │   │ H_(num_h_tiles-1)*128   │
    # │                 │                 │                 │   │                         │
    # │   H-tile 0      │   H-tile 1      │   H-tile 2      │...│ H-tile num_h_tiles-1    │ ← 128 rows
    # │                 │                 │                 │   │                         │   (P-dim)
    # │ H127            │ H255            │ H383            │   │ H-1                     │
    # └─────────────────┴─────────────────┴─────────────────┴───┴─────────────────────────┘
    #
    # H-tiles arranged left to right.
    # Therefore each column of SBUF contains H data with a stride of 1 (i.e. consecutive H data).

    w_sb = nl.ndarray((P_MAX, num_h_tiles, E), dtype=w.dtype, buffer=nl.sbuf)

    nisa.dma_copy(
        src=(w.reshape_dim(dim=0, shape=(num_h_tiles, P_MAX)).permute((1, 0, 2))),
        dst=w_sb,
    )

    return w_sb
