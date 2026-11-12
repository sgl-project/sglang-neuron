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
GLM-5.2's prefill MoE: the kernel library's blockwise-MX MoE kernel (``bwmm_shard_on_block_mx``) specialised to one
logical core (LNC1).

Same math: blockwise matrix multiplication for Mixture of Experts (MoE) layers using MXFP4 or
MXFP8 quantization. Tokens are pre-assigned to fixed-size blocks; each block runs the full
gate/up -> activation -> down pipeline for its expert and accumulates into the output.

Why this fork exists
--------------------
The shared kernel is hard-wired to two logical Neuron cores: ``check_kernel_compatibility``
asserts ``num_shards == 2``, blocks are split across the two shards, output writes carry a
``shard_id * T * H`` offset into a ``[2, T, H]`` buffer, and the kernel closes with a
``core_barrier`` plus a cross-shard reduce that merges the two halves. The GLM-5.2 prefill runs at
``NEURON_LOGICAL_NC_CONFIG=1`` (one logical core), where that machinery is dead weight at best and wrong at worst, so
this kernel (with ``block_moe_utils.py``) is the shared one specialised to a single core.

What differs from the shared kernel
-----------------------------------
This is a MECHANICAL specialisation at ``num_shards == 1`` / ``shard_id == 0``, not a redesign:

* The entry asserts a single logical core (``get_program_sharding_info()`` -> ``n_prgs == 1``).
* All block-distribution arithmetic collapses: every block is processed by the one core, the
  static loop walks ``0 .. num_static_blocks - 1``, and the dynamic loops step the block index
  by 1 instead of by ``num_shards``.
* The odd-block-count "remainder + dummy block" load-balancing split is gone: with one core
  there is no peer to hand a dummy block to.
* The output is a plain ``[T, H]`` buffer written with no shard offset, and there is no closing
  barrier or shard reduce -- the single core's accumulation IS the final result.

Single-core performance structure
---------------------------------
One core has no peer to hide latency behind, so the per-block pipeline is reorganised around the
DMA queues and the dynamic-loop back-edges, where every engine drains:

* Routing tables (``token_position_to_id``, ``block_to_expert``) are loaded into SBUF once. Per-block
  lookups are on-chip dynamic-offset copies instead of tiny DMAs that would queue behind
  multi-MB weight transfers.
* Hidden states are gathered as whole token rows (one software-DGE DMA per B128 tile) and
  PE-transposed into the MX moving layout.
* Weights are always reused across consecutive same-expert blocks (OOB-skipped reloads), with the
  full gate/up weights resident in SBUF when they fit.
* Standard-layout MX scales are repacked once into the packed layout, so each expert's scales load
  with one contiguous DMA.
* The output accumulates with a scatter read-modify-write DMA, deferred to the next block so no
  DMA is in flight at a loop entry or back-edge. The last static block prefetches the first dynamic
  block, so the dynamic loop starts warm.
* The activation and gate * up multiply evacuate the gate/up PSUM directly, and the expert affinity
  is applied while evacuating the down-projection PSUM.

MUST NOT be used at LNC2. With two cores both would redundantly compute every block and race on
the same un-offset output rows. Use the kernel library's ``bwmm_shard_on_block_mx`` there.
"""

from typing import Any, Optional

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import dge_mode, oob_mode

from ..mlp.mlp_parameters import MLPQuantizationParameters
from .down_projection_mx import _down_proj_prep_inter_and_weights, down_projection_mx
from .gate_up_projection_mx import _matmul_one_H512_tile, gate_up_projection_mx_tp
from .moe_cte_mx_utils import (
    SBUF_QUADRANT_SIZE,
    BWMMMXConfigs,
    BWMMMXDimensionSizes,
    InputTensors,
    ProjConfig,
    SharedBuffers,
    _generate_expert_index_vector,
    _pmax,
    _q_height,
    _q_width,
    apply_clamp,
    convert_to_mxfp_dtype,
    load_fp8_hidden_states_mx,
    quantize_block_hidden_state_T,
    quantize_block_hidden_state_T_static_mx,
    transpose_fp8_hidden_states,
)
from ..utils.sbuf_scoped_alloc import SbufManager, scoped_name, sizeinbytes
from ..utils.common_types import ActFnType, ExpertAffinityScaleMode, QuantizationType
from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import (
    NUM_HW_PSUM_BANKS,
    PSUM_BANK_SIZE,
    _sbm_alloc,
    get_nl_act_fn_from_type,
    get_program_sharding_info,
)
from ..utils.logging import get_logger
from ..utils.stream_shuffle_broadcast import stream_shuffle_broadcast
from .block_moe_utils import (
    OutputTensors,
    PSUM_SIZE,
    SkipMode,
    calculate_expert_affinities,
    div_ceil,
    output_initialization,
)

# Packed-scale geometry: each MX scale tile occupies a 4-partition stripe per
# 32-partition quadrant; up to 4 such tiles (16 partitions of the top half) fit
# per packed buffer. Used when computing the packed SBUF buffer shape.
SLOTS_PER_PACKED_BUFFER = 4

# I-tile size for tiled gate/up projection (512 = _pmax * _q_width = one I512 tile)
_I_TILE_SZ = _pmax * _q_width

# Reserve scratchpad buffer space for internally created ops, ex: identity for hidden transpose
SBUF_SCRATCHPAD_RESERVE = 1024

# ---------------------------------------------------------------------------
# Chunked dynamic-loop parameters
# ---------------------------------------------------------------------------
# _DYN_INNER = number of consecutive blocks the core processes per outer iter.
#   - The outer loop runs n_active // _DYN_STEP times; leftover 0.._DYN_STEP-1
#     active blocks fall through to the remainder loop (1 block per iter).
# _DYN_STEP = blocks consumed per outer iter across all cores. Upstream this is
# num_shards * _DYN_INNER; LNC1 has a single core so it equals _DYN_INNER. The two
# names are kept distinct to mirror the shared kernel's structure.

_DYN_INNER = 8
_DYN_STEP = _DYN_INNER
_DYN_STEP_LOG2 = _DYN_STEP.bit_length() - 1  # log2(_DYN_STEP); used as right-shift count log2(8) = 3

logger = get_logger("block_moe_mx")


@nki.jit
def block_moe_mx(
    hidden_states,
    expert_affinities_masked,
    gate_up_proj_weight,
    down_proj_weight,
    token_position_to_id,
    block_to_expert,
    # [1, 1] int32 expert-parallel rank. Packed-affinity path only: the affinity tail is dense over
    # every GLOBAL expert while block_to_expert holds LOCAL ids, so the kernel reads column
    # block_expert + ep_rank * E_local. None treats the local id as global (EP=1 only).
    ep_rank: nl.NkiTensor = None,
    # dynamic-loop variables
    conditions: nl.NkiTensor = None,
    gate_and_up_proj_bias: nl.NkiTensor = None,
    down_proj_bias: nl.NkiTensor = None,
    # quantize scales
    gate_up_proj_scale: nl.NkiTensor = None,
    down_proj_scale: nl.NkiTensor = None,
    # Non-tensor args
    block_size=None,
    n_static_blocks: int = -1,
    n_dynamic_blocks: int = -1,
    # Routing shape for the auto-computed best-case static-block estimate.
    # top_k tokens per expert, sharded across ep_degree EP ranks.
    top_k: int = 1,
    ep_degree: int = 1,
    gate_up_activations_T=None,
    down_activations=None,
    # Meta parameters
    activation_function: ActFnType = ActFnType.SiLU,
    skip_dma: SkipMode = SkipMode(False, False),
    compute_dtype=nl.bfloat16,
    weight_dtype: Any = None,  # Target dtype for weight conversion (e.g., nl.float8_e4m3fn_x4, nl.float8_e5m2_x4)
    is_tensor_update_accumulating=True,
    expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    use_packed_scales: bool = False,
    weight_skip_expert: int = None,
    # When quantization_type == QuantizationType.STATIC_MX, takes the STATIC_MX path.
    # Otherwise (default NONE / any non-STATIC_MX value), behaves as MX (existing path, bit-identical).
    # STATIC_MX reuses the existing gate_up_proj_scale / down_proj_scale tensors to carry the
    # per-expert weight scales (gate_up_proj_scale: fp32 [E, 2, 1], down_proj_scale: fp32 [E, 1]);
    # the per-tensor input scales come in via gate_up_in_scale / down_in_scale.
    quantization_type: QuantizationType = QuantizationType.NONE,
    gate_up_in_scale: Optional[nl.NkiTensor] = None,
    down_in_scale: Optional[nl.NkiTensor] = None,
    # Caller-provided [T, H] output buffer (any HBM kind). Lets an inlining caller point the
    # accumulation at e.g. a private_hbm collective source directly, instead of copying the
    # kernel-allocated shared_hbm output into one.
    output_tensor: nl.NkiTensor = None,
    # Load the routing tables with a strided dma_copy instead of a dma_transpose. An in-flight
    # in-trace collective holds later HWDGE transposes but not plain copies, so a call that should
    # run under a collective (a small table, e.g. the GK shared expert) sets this.
    routing_plain_dma: bool = False,
    # The routing as the nonzero rows of find_nonzero_indices ([E, T] int32, -1 past each count) and the [N] int32 row block
    # (e * T / B + j: block j of expert e) of each block, instead of token_position_to_id. The token ids are the same, and the
    # caller skips the flatten of the rows. A row block index of E * T / B reads padding (-1).
    token_rows: nl.NkiTensor = None,
    block_rows: nl.NkiTensor = None,
    # False: output_tensor already holds zeros (the caller zeroed it), so the kernel issues no zero-init and only
    # accumulates. True (default): the kernel zeroes its output before the first block.
    init_output: bool = True,
    # (GLM_GK_TC_MOE_AG=split) [T, E_global] rows of the dense affinities, gathered on their own: hidden_states
    # then holds only [hidden | scale] and each block's gather fills the concat row's affinity tail from these rows, so
    # the gathered concat rows are byte for byte the fused-row ones. None (default): the affinities ride in the row.
    affinity_rows: nl.NkiTensor = None,
    # (GK GLM_GK_MOE_AFF8) the packed row tail is the router's compact top-8 [ids (8 fp32) | affinities (8 fp32)]
    # (rmsnorm_mx_prefill pack_topk8) instead of the dense [E_global] fp32 affinities: each B128 tile's affinity for the
    # block's global expert g is max_k (ids[k] == g) * vals[k], the dense value bit for bit. Needs ep_rank. With
    # affinity_rows (GLM_GK_TC_MOE_AG=split) those rows are the [T, 16] compact tails. False (default): the dense tail.
    compact_affinities: bool = False,
    # ``(tile_major, priority, second_queue)`` of the gate/up and down weight DMAs: ``tile_major`` 1 / 2 = the
    # gate/up weights are stored I-tile-major (``_gup_tile_src``; 2 also loads each 512-wide I tile with one DMA instead of the
    # H-chunked streams), ``priority`` = their DMA QoS class (0 highest .. 3 lowest), ``second_queue`` = odd I-tiles' weight
    # DMAs on the Scalar sequencer's HWDGE queue. The same bytes land in the same SBUF buffers; the matmuls are unchanged.
    # None (default): the H-chunked streams.
    weight_stream: tuple = None,
    # (GK GLM_GK_MOE_PF_TILES) gate/up I tiles of the NEXT block prefetched across the block boundary (tiled
    # gate/up path only): 1 (default) = tile 0 only; k = 2 / 3 = tiles 0 .. k-1 into k persistent tile
    # buffers (tile i of a block uses buffer i % k), so the weight queue streams the next block's tiles 1 .. k-1 under this
    # block's down projection instead of idling behind the next down-weight load's WAR wait. The prefetches whose block is
    # dynamic are skipped (OOB sentinel) when that block is not live. Same bytes into the same matmuls: bit-identical.
    prefetch_tiles: int = 1,
):
    """
    Blockwise MXFP MoE kernel, SINGLE LOGICAL CORE (LNC1) only. Use as standalone kernel.

    The blockwise matrix multiplication (matmul) kernel implements a Mixture of Experts (MoE)
    layer at a block granularity, offering an alternative to token dropping approaches.
    This method assumes that tokens have already been assigned to blocks, as specified
    by the user through the token_position_to_id parameter.
    This kernel loops over all blocks, without considering they are padded or non-padded blocks.
    Supports both MXFP4 and MXFP8 weight quantization.

    LNC1 ONLY: the kernel asserts a single logical Neuron core at entry. There is no
    block distribution across shards, no shard offset on the output (which is a plain
    [T, H] buffer), and no closing core_barrier / cross-shard reduce -- the one core's
    accumulation is the final result. Launching this at LNC2 fails the entry assert; use
    the kernel library's bwmm_shard_on_block_mx for two cores.

    Intended Usage:
        - Block size B: 128-1024 tokens
        - Total tokens T: 32k
        - Hidden dimension H: 512-8192
        - Intermediate dimension I_TP: 384-3072
        - Number of experts E: 8-128

    Dimensions:
        H: Hidden dimension size
        T: Total number of input tokens (after linearizing across the batch dimension)
        B: Number of tokens per block
        N: Total number of blocks
        E: Number of experts
        I: Intermediate size / tp degree

    Args:
        hidden_states (nl.NkiTensor): Tensor of input hidden states on HBM of size (T+1, H). The reason it is T+1 is because padding token position is set to T.
                                       with skip_dma, id will be set to -1, so this shape can be (T, H). Similarly for expert_affinities_masked, output
        expert_affinities_masked (nl.NkiTensor): Tensor of expert affinities corresponding to each token of size ((T+1) * E, 1).
                                        TODO: cannot refactor to (T+1, E) as we currently don't support dynamic slice on both axis.
        gate_up_proj_weight (nl.NkiTensor): Tensor of concatenated gate and up projection weights on HBM (E, H, 2, I).
                                          Supports MXFP4 (nl.float4_e2m1fn_x4) and MXFP8 (nl.float8_e4m3fn_x4, nl.float8_e5m2_x4).
        down_proj_weight (nl.NkiTensor): Tensor of down projection weights on HBM (E, I, H).
                                       Supports MXFP4 (nl.float4_e2m1fn_x4) and MXFP8 (nl.float8_e4m3fn_x4, nl.float8_e5m2_x4).
        block_size (int): Number of tokens per block
        token_position_to_id (nl.NkiTensor): Tensor of block index of the corresponding tokens on HBM (N * B,)
                                          Note that we include tokens included for padding purposes and N * B >= T.
                                          For padding token, id is set to T. with skip_dma, id will be set to -1.
        block_to_expert (nl.NkiTensor): Tensor of expert indices of corresponding blocks on HBM (N, 1)

        num_static_block (int): Optional. Number of non-padded blocks if known (default: -1).
        n_dynamic_blocks (int): Number of blocks to process with dynamic loop when n_static_blocks
            is not specified (default: 55, empirically tuned for GPT-OSS).
        gate_and_up_proj_bias: nl.NkiTensor = None, Optional. A tensor of shape [E, 2, I].
                              Note that if activation function is Swiglu, we expect up_bias = up_bias + 1
        down_proj_bias: nl.NkiTensor = None. Optional argument. A tensor of shape [E, H]

        # Arguments for quantization scales
        gate_up_proj_scale: nl.NkiTensor = None. uint8 MX scales of shape
                            [E, _pmax // _q_height, 2, n_H512_tile, I] (standard layout) or
                            [E, _pmax, n_packed_gup, 2, I] (use_packed_scales=True).
                            Under STATIC_MX it instead carries the per-expert gate/up weight
                            scales as fp32 [E, 2, 1] (idx 0 = gate, idx 1 = up).
        down_proj_scale: nl.NkiTensor = None. uint8 MX scales of shape
                            [E, p_I // _q_height, n_total_I512_tile, H] (standard layout) or
                            [E, _pmax, n_packed_down, H] (use_packed_scales=True).
                            Under STATIC_MX it instead carries the per-expert down weight
                            scale as fp32 [E, 1].

        # Unsupported output tensors. Please set to None.
        gate_up_activations_T: nl.NkiTensor = None. Currently not supported.
        down_activations: nl.NkiTensor = None. Currently not supported

        # meta parameters
        activation_function: one of the Enum in nkilib.core.utils.common_types.ActFnType.
                              Indicate what activation function to use in the MLP block
        skip_dma: SkipMode = SkipMode(False, False). skip_token is honored; weight reuse across
            same-expert blocks (what skip_weight enables upstream) is always on in this kernel.
        weight_skip_expert: int = None. The expert index a same-expert block loads its weights with,
            so the reload is OOB-skipped and the previous block's weights stay resident. It must be
            out of bounds of the whole HBM tensor the weights live in, not just of the view passed
            here: for a view ``w[i]`` into a stacked ``[L, E, ...]`` tensor, index E is still in
            bounds (it reads layer i + 1's expert 0) unless i is the last layer, so pass L * E.
            Defaults to E.
        compute_dtype=nl.bfloat16,
        weight_dtype: Target dtype for weight conversion when weights are passed as uint/int/float types.
                     For MXFP4: nl.float4_e2m1fn_x4
                     For MXFP8: nl.float8_e4m3fn_x4 or nl.float8_e5m2_x4
                     If None, auto-detects (defaults to e4m3fn for MXFP8)
        is_tensor_update_accumulating: bool. Indicate whether we need to accumulate the results over multiple blocks
        expert_affinities_scaling_mode: one of the Enum in nkilib.core.utils.common_types.ExpertAffinityScaleMode.
                                        Indicate if the kernel is doing post or pre scaling.
        n_block_per_iter: int. Currently unsupported

        #parameters for clipping the MLP projections
        gate_clamp_upper_limit: Optional[float] = None,
        gate_clamp_lower_limit: Optional[float] = None,
        up_clamp_lower_limit: Optional[float] = None,
        up_clamp_upper_limit: Optional[float] = None

        skip_dma (bool): Whether to skip DMA operations (default: False)

    Returns:
        output (nl.NkiTensor): Tensor of output hidden states on HBM of size (T+1, H).

    Notes:
        - All input/output tensors must have the same floating point dtype
        - token_position_to_id and block_to_expert must be np.int32 tensors

    Pseudocode:
        assert num_logical_cores == 1   # LNC1 only

        if expert_affinities_scaling_mode == PRE_SCALE_DELAYED:
            expert_affinities_scaling_mode = PRE_SCALE

        T, H = hidden_states.shape
        B = block_size
        E, _, _, _, I = gate_up_proj_weight.shape
        N = token_position_to_id.shape[0] // B
        dims = BWMMMXDimensionSizes(T, H, B, E, N, I, cond_vec_len)
        prj_cfg = ProjConfig(H, I, B, force_lnc1=True, n_prgs=1, prg_id=0)
        configs = BWMMMXConfigs(...)

        allocate reused buffers: p_gup_idx_vector, p_down_idx_vector, gup_scales_sb, activation_bias
        inps = InputTensors(...)

        check_kernel_compatibility(dims, configs)

        output = allocate [T, H] in HBM
        if is_tensor_update_accumulating:
            output_initialization(output, dims)

        allocate shared buffers: block_hidden_states, block_hidden_states_T, hidden_qtz_sb, hidden_scale_sb
        allocate down_weight_qtz, cond, index

        if use_dynamic_while:
            n_dynamic_blocks = N - n_static_blocks (padded to even)
            n_static_blocks = N - n_dynamic_blocks
            process_static_blocks(n_static_blocks)
            process_dynamic_blocks(n_dynamic_blocks)
        else:
            process_static_blocks(N)

        # LNC1: single core -> nothing to synchronise and nothing to reduce.
        # The accumulated output IS the result.
        return output
    """
    _WEIGHT_SKIP_EXPERT[0] = weight_skip_expert
    _AFF_ROWS[0] = affinity_rows
    _ws = tuple(weight_stream) if weight_stream is not None else (0, None, False)
    kernel_assert(
        len(_ws) == 3 and _ws[0] in (0, 1, 2) and (_ws[1] is None or _ws[1] in (0, 1, 2, 3)),
        f"weight_stream {weight_stream!r}: (tile_major 0 / 1 / 2, priority None / 0..3, second_queue)",
    )
    _WSTREAM[0], _WSTREAM[1], _WSTREAM[2] = _ws[0], _ws[1], bool(_ws[2])
    kernel_assert(prefetch_tiles in (1, 2, 3), f"prefetch_tiles {prefetch_tiles!r}: 1 (tile 0 only), 2 or 3")
    _PFT[0] = prefetch_tiles
    _PF_BUFS[0] = None
    _PF_LIVE[0] = None
    kernel_assert(
        not compact_affinities
        or (ep_rank != None and expert_affinities_masked == None and (affinity_rows == None or affinity_rows.shape[-1] == 2 * _AFF8_K)),
        "compact_affinities needs ep_rank and the compact tail in the rows (or [T, 16] affinity_rows); no expert_affinities_masked",
    )
    _AFF8[0] = (ep_rank, gate_up_proj_weight.shape[0]) if compact_affinities else None
    # LNC1-only kernel: fail loudly if launched on more than one logical core. All the
    # block-distribution / shard-offset / cross-core-reduce logic of the shared LNC2 kernel
    # has been specialised away here, so a multi-core launch would have every core redundantly
    # compute every block and race on the same un-offset output rows.
    _, _n_logical_cores, _ = get_program_sharding_info()
    kernel_assert(
        _n_logical_cores == 1,
        f"block_moe_mx is an LNC1-only kernel, but was launched on {_n_logical_cores} logical cores. "
        f"Use moe/bwmm_shard_on_block_mx.py for LNC2.",
    )

    if expert_affinities_scaling_mode == ExpertAffinityScaleMode.PRE_SCALE_DELAYED:
        expert_affinities_scaling_mode = ExpertAffinityScaleMode.PRE_SCALE

    # Always reuse weights across consecutive same-expert blocks. The weight-skip path
    # OOB-skips a reload when the next block's expert matches the current one, which is
    # correct for any routing, so it does not wait on the caller's skip_weight opt-in.
    skip_dma = SkipMode(skip_dma.skip_token, True)

    # Convert weights to MXFP dtype first
    gate_up_proj_weight, target_dtype = convert_to_mxfp_dtype(gate_up_proj_weight, weight_dtype)
    down_proj_weight, _ = convert_to_mxfp_dtype(down_proj_weight, target_dtype)

    # Pre-quantized fp8 hidden states (real MX): hidden_states arrives as a concatenated
    # [T, H_concat] tensor = [hidden_quant (H fp8) | hidden_scale (H/4 uint8)] from a prior
    # QMX layer. We skip on-device quantization and gather+transpose both regions. The true
    # hidden dim H is recovered from the gate/up weight (concat dim hides it on the input).
    # A missing affinity tensor means the affinities travel inside the pre-quantized row. The
    # producer then hands the row over in the affinity (container) dtype, not fp8, so the dtype test
    # alone would miss it and read H from the whole packed width.
    is_affinities_packed_arg = expert_affinities_masked == None
    expert_affinities_dtype = hidden_states.dtype  # affinity tail dtype (meaningful only when packed)
    if affinity_rows != None:
        kernel_assert(is_affinities_packed_arg, "affinity_rows replaces the fused affinity tail: pass expert_affinities_masked=None")
        expert_affinities_dtype = affinity_rows.dtype
    is_fp8_hidden = is_affinities_packed_arg or hidden_states.dtype in (nl.float8_e4m3fn, nl.float8_e5m2)

    T = hidden_states.shape[0]
    B = block_size
    E, _Hp, _, _n_H512, I = gate_up_proj_weight.shape
    # When the producer fused expert affinities into the row, the concat is
    # [hidden (H fp8) | scale (scale_region uint8) | affinities (E_global x affinity dtype) | pad].
    # affinities_col_offset is the fp8 column where the affinity region starts (= the end of the
    # hidden+scale region).
    is_affinities_packed = False
    affinities_col_offset = 0
    if is_fp8_hidden:
        H = _Hp * _n_H512 * _q_width  # real hidden dimension that's not concatenated with scales
        kernel_assert(
            quantization_type == QuantizationType.MX,
            f"fp8 pre-quantized hidden states only support QuantizationType.MX, got {quantization_type}",
        )
        # Measure the row in fp8 columns (a no-op view when it is already fp8): a container-dtype
        # row counts containers, which would put the affinity offset outside the row.
        hidden_states = hidden_states.view(nl.float8_e4m3fn)
        hidden_scale_width = H + div_ceil(_n_H512, SLOTS_PER_PACKED_BUFFER) * _pmax
        # A missing affinity tensor declares the packed layout; a wider row confirms it.
        is_affinities_packed = is_affinities_packed_arg or hidden_states.shape[-1] > hidden_scale_width
        affinities_col_offset = hidden_scale_width
        if affinity_rows != None:
            kernel_assert(
                hidden_states.shape[-1] == hidden_scale_width and affinity_rows.shape[0] == T,
                f"affinity_rows: hidden_states must be [T, {hidden_scale_width}] fp8 (no tail) and affinity_rows [T, E_global]; "
                f"got {hidden_states.shape} and {affinity_rows.shape}",
            )
    else:
        _, H = hidden_states.shape
    cond_vec_len = conditions.shape[0] if conditions != None else 0

    N = block_rows.shape[0] if token_rows != None else token_position_to_id.shape[0] // B
    dims = BWMMMXDimensionSizes(T=T, H=H, B=B, E=E, N=N, I=I, cond_vec_len=cond_vec_len)

    is_static_quant = quantization_type == QuantizationType.STATIC_MX

    quant_params = None
    if is_static_quant:
        # STATIC_MX folds the per-block MX scale tables into the dummy-127 path, so the
        # per-block packed-scale layout is unused here; reject it explicitly.
        kernel_assert(
            not use_packed_scales,
            "STATIC_MX does not support use_packed_scales; per-block scales are not used on this path.",
        )
        # Reuse the existing weight-scale tensors: gate_up_proj_scale carries the packed
        # [E, 2, 1] gate/up weight scales, down_proj_scale carries the [E, 1] down weight scale.
        # Split the packed gate/up tensor into separate [E, 1] gate and up views here so the
        # setup below consumes gate_w_scale / up_w_scale symmetrically (idx 0 = gate, idx 1 = up).
        gate_up_w_view = gate_up_proj_scale.reshape((dims.E, 2, 1))
        quant_params = MLPQuantizationParameters(
            quantization_type=quantization_type,
            gate_w_scale=gate_up_w_view.slice(dim=1, start=0, end=1).reshape((dims.E, 1)),
            up_w_scale=gate_up_w_view.slice(dim=1, start=1, end=2).reshape((dims.E, 1)),
            down_w_scale=down_proj_scale,
            gate_up_in_scale=gate_up_in_scale,
            down_in_scale=down_in_scale,
            clipping_bound=0.0,
        )

    # When use_packed_scales is True, both scale tensors are in packed HBM layout:
    #   gate_up_proj_scale: uint8[E, _pmax, n_packed_gup, 2, I]
    #   down_proj_scale:    uint8[E, _pmax, n_packed_down, H]
    # Producers always emit both packed or both standard, so a single flag covers both.

    # Skip unused partition zeroing when both gate and up have at least one clamp,
    # since the clamp op writes all partitions (including unused ones).
    has_gate_clamp = gate_clamp_upper_limit != None or gate_clamp_lower_limit != None
    has_up_clamp = up_clamp_upper_limit != None or up_clamp_lower_limit != None
    zero_unused_partitions = not (has_gate_clamp and has_up_clamp)

    prj_cfg = ProjConfig(
        H=dims.H,
        I=dims.I,
        BxS=dims.B,
        force_lnc1=True,
        n_prgs=1,
        prg_id=0,
        use_stream_shuffle_broadcast=False,
        sharding_config="H",
        zero_unused_partitions=zero_unused_partitions,
    )

    # Standard-layout MX scales spend 16 of every 128 partitions and need a 16-row gather per block.
    # Repack them once into the packed layout so every block loads its expert's scales with one
    # contiguous 128-partition DMA into a ~4x smaller SBUF buffer.
    if (
        not is_static_quant
        and not use_packed_scales
        and gate_up_proj_scale.shape[1] == _pmax // _q_height
        and down_proj_scale.shape[1] == _pmax // _q_height
    ):
        gate_up_proj_scale, down_proj_scale = _repack_scales_to_packed(gate_up_proj_scale, down_proj_scale)
        use_packed_scales = True

    # subtract 1024 for scratchpad buffer space for internally creates ops, ex: identity for hidden transpose
    sb_upper_bound = nl.tile_size.total_available_sbuf_size - SBUF_SCRATCHPAD_RESERVE
    sbm = SbufManager(0, sb_upper_bound, logger)
    sbm.open_scope(name="top_level_scope")

    # reused buffers
    p_gup_idx_vector = sbm.alloc_stack((_pmax, 1), dtype=nl.float32, name="p_idx_vector", align=SBUF_QUADRANT_SIZE)
    nisa.memset(dst=p_gup_idx_vector, value=-1.0)

    p_gup_idx_vector_int32 = sbm.alloc_stack(
        (_pmax, 1), dtype=nl.int32, name="p_idx_vector_int32", align=SBUF_QUADRANT_SIZE
    )

    p_down_idx_vector = sbm.alloc_stack(
        (_pmax, 1), dtype=nl.float32, name="p_down_idx_vector", align=SBUF_QUADRANT_SIZE
    )
    nisa.memset(dst=p_down_idx_vector, value=-1.0)

    """
    When packing weight scales, allocate a smaller [_pmax, n_packed, 2, I] buffer instead
    of [_pmax, 2, n_H512_tile_sharded, I]. Since n_packed = ceil(n_H512_tile / 4), we save
    roughly 4x SBUF on this buffer when n_H512_tile is a multiple of 4 (and 3x for n_H512_tile=6).
    Unpacked path: scale DMAs use vector_offset which forces SWDGE; SWDGE permits
    src/dst dtype mismatch, so we can allocate as uint32 (a faster memset on the upfront
    zero-pad) and consume via .view(nl.uint8). The packed path uses scalar_offset HWDGE,
    which requires src/dst dtype match, so it stays on uint8.
    """
    _gup_alloc_as_u32 = (not use_packed_scales) and (dims.I % 4 == 0)
    if is_static_quant:
        # STATIC_MX: single shared [_pmax, max_free] all-127 dummy reused across
        # gup_scales_sb (cur_I128_tile_sz ≤ 128), hidden_scale_sb / down inter
        # (cur_BxS ≤ BxS_tile_sz), etc. BxS_tile_sz = min(B, _psum_fmax*2/_q_width)
        # = min(B, 256), so the buffer must fit the widest tile.
        _bxs_tile_sz = min(dims.B, PSUM_SIZE * 2 // _q_width)
        _static_dummy_free = max(_pmax, _bxs_tile_sz)
        n_packed_gup = 0
        static_dummy_scale_sb = sbm.alloc_stack(
            (_pmax, _static_dummy_free),
            dtype=nl.uint8,
            name="static_dummy_scale_sb",
            align=SBUF_QUADRANT_SIZE,
        )
        nisa.memset(static_dummy_scale_sb, value=127)
        gup_scales_sb = static_dummy_scale_sb
    elif use_packed_scales:
        n_packed_gup = (prj_cfg.n_H512_tile_sharded + SLOTS_PER_PACKED_BUFFER - 1) // SLOTS_PER_PACKED_BUFFER
        gup_scales_sb = sbm.alloc_stack(
            (_pmax, n_packed_gup, 2, dims.I),
            dtype=nl.uint8,
            name="gup_scales_packed_sb",
            align=SBUF_QUADRANT_SIZE,
        )
    elif _gup_alloc_as_u32:
        n_packed_gup = 0
        gup_scales_sb = sbm.alloc_stack(
            (_pmax, 2, prj_cfg.n_H512_tile_sharded, dims.I // 4),
            dtype=nl.uint32,
            name="gup_scales_sb",
            align=SBUF_QUADRANT_SIZE,
        )
        gup_scales_sb = gup_scales_sb.view(nl.uint8)
    else:
        n_packed_gup = 0
        gup_scales_sb = sbm.alloc_stack(
            (_pmax, 2, prj_cfg.n_H512_tile_sharded, dims.I),
            dtype=nl.uint8,
            name="gup_scales_sb",
            align=SBUF_QUADRANT_SIZE,
        )

    activation_bias = sbm.alloc_stack((_pmax, 1), dtype=nl.float32, name="activation_bias", align=SBUF_QUADRANT_SIZE)
    nisa.memset(activation_bias, value=0)

    # ── STATIC_MX top-level setup ──
    # Pack per-expert quant constants into two SBUF tables, broadcast across all 128 partitions.
    # Per-block: one scalar_offset gather per table — gup pulls [_pmax, 3], down pulls [_pmax, 2].
    #   gup_scale_lut_sb [_pmax, E, 3]: [1/in_scale[e], in_scale[e]*gate_w_scale[e], in_scale[e]*up_w_scale[e]]
    #   down_scale_lut_sb [_pmax, E, 2]: [1/down_in_scale[e],   down_in_scale[e]*down_w_scale[e]]
    gup_scale_lut_sb = None
    down_scale_lut_sb = None
    if is_static_quant:
        # ── gup table: [_pmax, E, 3] ──
        gup_scale_lut_sb = sbm.alloc_stack(
            (_pmax, dims.E, 3), dtype=nl.float32, name="gup_scale_lut_sb", align=SBUF_QUADRANT_SIZE
        )
        gup_p0 = gup_scale_lut_sb.slice(dim=0, start=0, end=1)  # [1, E, 3]
        gup_slot0 = gup_p0.slice(dim=2, start=0, end=1)  # [1, E, 1] = in_scale per expert
        gup_slot1 = gup_p0.slice(dim=2, start=1, end=2)  # gate w_scale → gate combined (after fuse)
        gup_slot2 = gup_p0.slice(dim=2, start=2, end=3)  # up w_scale → up combined (after fuse)
        gup_slots12 = gup_p0.slice(dim=2, start=1, end=3)  # gate+up slots, fused together below
        gup_slot0_bcast2 = gup_p0.slice(dim=2, start=0, end=1).broadcast(dim=2, size=2)

        # Load per-expert in_scale into slot 0; gate_w_scale into slot 1, up_w_scale into slot 2.
        nisa.dma_copy(
            dst=gup_slot0,
            src=quant_params.gate_up_in_scale.reshape((1, dims.E, 1)),
            dge_mode=nisa.dge_mode.hwdge,
        )
        nisa.dma_copy(
            dst=gup_slot1,
            src=quant_params.gate_w_scale.reshape((1, dims.E, 1)),
            dge_mode=nisa.dge_mode.hwdge,
        )
        nisa.dma_copy(
            dst=gup_slot2,
            src=quant_params.up_w_scale.reshape((1, dims.E, 1)),
            dge_mode=nisa.dge_mode.hwdge,
        )
        # combined = w_scale * in_scale on slots 1 and 2, (broadcast slot 0 across the 2 inner slots).
        nisa.tensor_tensor(dst=gup_slots12, data1=gup_slots12, data2=gup_slot0_bcast2, op=nl.multiply)
        # Replace slot 0 with reciprocal (1 / in_scale[e]) for the bf16→fp8 quant tensor_scalar.
        nisa.reciprocal(dst=gup_slot0, data=gup_slot0)
        # Broadcast to all 128 partitions
        gup_table_flat = gup_scale_lut_sb.reshape((_pmax, dims.E * 3))
        stream_shuffle_broadcast(src=gup_table_flat, dst=gup_table_flat)

        # ── down table: [_pmax, E, 2] ──
        down_scale_lut_sb = sbm.alloc_stack(
            (_pmax, dims.E, 2), dtype=nl.float32, name="down_scale_lut_sb", align=SBUF_QUADRANT_SIZE
        )
        down_p0 = down_scale_lut_sb.slice(dim=0, start=0, end=1)  # [1, E, 2]
        down_slot0 = down_p0.slice(dim=2, start=0, end=1)
        down_slot1 = down_p0.slice(dim=2, start=1, end=2)

        nisa.dma_copy(
            dst=down_slot0,
            src=quant_params.down_in_scale.reshape((1, dims.E, 1)),
            dge_mode=nisa.dge_mode.hwdge,
        )
        nisa.dma_copy(
            dst=down_slot1,
            src=quant_params.down_w_scale.reshape((1, dims.E, 1)),
            dge_mode=nisa.dge_mode.hwdge,
        )
        # Slot 1 ← in_scale[e] * w_scale[e].
        nisa.tensor_tensor(dst=down_slot1, data1=down_slot1, data2=down_slot0, op=nl.multiply)
        # Slot 0 ← 1 / in_scale[e] for intermediate fp8 quant.
        nisa.reciprocal(dst=down_slot0, data=down_slot0)
        # Broadcast to all 128 partitions.
        down_table_flat = down_scale_lut_sb.reshape((_pmax, dims.E * 2))
        stream_shuffle_broadcast(src=down_table_flat, dst=down_table_flat)

    # Hoist [0, 1, 2, 3] H-fold offset vector once for all blocks. Used for calculated hidden state indices
    arange_4H = sbm.alloc_stack((1, _q_width), dtype=nl.float32, name="arange_4H", align=SBUF_QUADRANT_SIZE)
    nisa.iota(arange_4H, [[1, _q_width]], offset=0)

    # Packed-affinity path: this rank's global expert base (ep_rank * E_local), derived once.
    local_expert_start_idx_sb = None
    n_global_experts = E
    if is_affinities_packed and ep_rank != None:
        local_expert_start_idx_sb = sbm.alloc_stack(
            (1, 1), dtype=nl.int32, name="local_expert_start_idx_sb", align=SBUF_QUADRANT_SIZE
        )
        nisa.dma_copy(dst=local_expert_start_idx_sb[0, 0], src=ep_rank.ap(pattern=[[1, 1], [1, 1]], offset=0))
        e_local_sb = sbm.alloc_stack((1, 1), dtype=nl.int32, name="e_local_sb", align=SBUF_QUADRANT_SIZE)
        nisa.memset(dst=e_local_sb, value=E)
        nisa.tensor_tensor(
            dst=local_expert_start_idx_sb, data1=local_expert_start_idx_sb, data2=e_local_sb, op=nl.multiply
        )
        n_global_experts = E * ep_degree
        _aff_width = (
            hidden_states.shape[-1] // sizeinbytes(expert_affinities_dtype)
            if affinity_rows == None
            else affinities_col_offset // sizeinbytes(expert_affinities_dtype) + affinity_rows.shape[-1]
        )
        kernel_assert(
            affinities_col_offset // sizeinbytes(expert_affinities_dtype) + (2 * _AFF8_K if _AFF8[0] != None else n_global_experts)
            <= _aff_width,
            f"The packed row has no room for {n_global_experts} global expert affinities. "
            f"Check ep_degree ({ep_degree}) against E_local ({E}).",
        )

    # fp8 path keeps the raw concat [T, H_concat] view; the gather helper does its own .ap().
    # bf16 path reshapes to the H-folded layout consumed by load_hidden_states_mx.
    _hidden_states_view = (
        hidden_states if is_fp8_hidden else hidden_states.reshape((T, _q_width, prj_cfg.n_H512_tile, _pmax))
    )

    inps = InputTensors(
        hidden_states=_hidden_states_view,
        gate_up_proj_weight=gate_up_proj_weight,
        gate_and_up_proj_bias=gate_and_up_proj_bias,
        down_proj_bias=down_proj_bias,
        down_proj_weight=down_proj_weight,
        gate_up_proj_scale=gate_up_proj_scale,
        down_proj_scale=down_proj_scale,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        expert_affinities_masked=expert_affinities_masked,
        p_gup_idx_vector=p_gup_idx_vector,
        p_gup_idx_vector_int32=p_gup_idx_vector_int32,
        p_down_idx_vector=p_down_idx_vector,
        gup_scales_sb=gup_scales_sb,
        activation_bias=activation_bias,
        conditions=conditions,
        arange_4H=arange_4H,
        local_expert_start_idx=local_expert_start_idx_sb,
        gup_scale_lut_sb=gup_scale_lut_sb,
        down_scale_lut_sb=down_scale_lut_sb,
    )

    routing = _load_resident_routing(
        inps, dims, sbm, n_global_experts=n_global_experts, plain_dma=routing_plain_dma, rows=token_rows, row_blocks=block_rows,
    )

    # Full-persistent gup scheme: only available if SBUF budget allows.
    # In that case, the persistent gup buffer holds the full (gate+up, full I) weights across blocks, and the cross-block
    # prefetch OOB-skips on same-expert blocks. All other configs keep the per-tile streaming.
    _gup_full_persistent = skip_dma.skip_weight and dims.I > _I_TILE_SZ

    # If the largest SBUF buffers — persistent (full-persistent gup, down weight,
    # hidden_qtz, hidden pre/post-transpose double buffers) plus the
    # biggest per-block coexisting buffer (dp_out_sb) would exceed 85% of
    # per-partition SBUF, fall back to per-tile streaming so the remaining allocs don't OOM.
    if _gup_full_persistent:
        _gup_full_bytes = 2 * prj_cfg.n_H512_tile_sharded * dims.I * sizeinbytes(gate_up_proj_weight.dtype)
        _down_w_bytes = prj_cfg.n_total_I512_tile * prj_cfg.H_sharded * sizeinbytes(down_proj_weight.dtype)
        _hidden_qtz_bytes = prj_cfg.n_H512_tile * dims.B * sizeinbytes(nl.float8_e4m3fn_x4)
        _block_hs_bytes = (dims.B // SBUF_QUADRANT_SIZE) * prj_cfg.n_H512_tile * _pmax * sizeinbytes(compute_dtype)
        _dp_out_bytes = 2 * dims.H * sizeinbytes(compute_dtype)
        # Persistent uint8 weight-scale buffers (gup_scales_sb + down_scale_sb). Their size varies by
        # path and, on the standard MX path, is the single largest term of this estimate (an estimate
        # calibrated for STATIC_MX only misses it):
        #   - STATIC_MX: a tiny shared all-127 dummy is reused -> ~128 B.
        #   - packed:    4 H512/I512 tiles fold into one 128-wide block -> ~4x smaller.
        #   - standard:  full per-tile scales (uint8, 1 B/elt) co-resident with gup_full.
        # Count them so full-resident gup falls back to per-tile streaming when they don't fit.
        if is_static_quant:
            _scale_bytes = _pmax
        elif use_packed_scales:
            _n_packed_down = div_ceil(prj_cfg.n_total_I512_tile, SLOTS_PER_PACKED_BUFFER)
            _scale_bytes = n_packed_gup * 2 * dims.I + _n_packed_down * prj_cfg.H_sharded
        else:
            _scale_bytes = 2 * prj_cfg.n_H512_tile_sharded * dims.I + prj_cfg.n_total_I512_tile * prj_cfg.H_sharded
        _big_bufs_bytes = (
            _gup_full_bytes + _down_w_bytes + _hidden_qtz_bytes + 2 * _block_hs_bytes + _dp_out_bytes + _scale_bytes
        )
        _sbuf_threshold = (nl.tile_size.total_available_sbuf_size * 85) // 100
        if _big_bufs_bytes > _sbuf_threshold:
            logger.debug(
                f"Disabling gup_full_persistent: big buffers={_big_bufs_bytes} B > 85% per-partition SBUF "
                f"({_sbuf_threshold} B). Falling back to per-tile-0 weight skipping."
            )
            _gup_full_persistent = False

    kernel_assert(
        not _WSTREAM[0]
        or (not _gup_full_persistent and dims.I % _I_TILE_SZ == 0 and prj_cfg.n_H512_tile_sharded == prj_cfg.n_H512_tile),
        f"weight_stream tile_major needs the I-tiled gate/up path (I {dims.I} a multiple of {_I_TILE_SZ} > {_I_TILE_SZ}, no "
        f"full-resident gate/up, unsharded H)",
    )

    configs = BWMMMXConfigs(
        skip_dma=skip_dma,
        compute_dtype=compute_dtype,
        scaling_mode=expert_affinities_scaling_mode,
        weight_dtype=gate_up_proj_weight.dtype,
        io_dtype=hidden_states.dtype,
        is_tensor_update_accumulating=is_tensor_update_accumulating,
        use_dynamic_while=conditions != None,
        n_static_blocks=n_static_blocks,
        linear_bias=(gate_and_up_proj_bias != None and down_proj_bias != None),
        activation_function=activation_function,
        fuse_gate_and_up_load=True,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        qtz_dtype=nl.float8_e4m3fn_x4,
        use_packed_scales=use_packed_scales,
        is_static_quant=is_static_quant,
        has_gate_clamp=has_gate_clamp,
        gup_full_persistent=_gup_full_persistent,
        is_fp8_hidden=is_fp8_hidden,
        is_affinities_packed=is_affinities_packed,
        affinities_col_offset=affinities_col_offset,
        expert_affinities_dtype=expert_affinities_dtype,
    )

    check_kernel_compatibility(dims, configs)

    # Output is the dequantized result, not fp8. For pre-quantized fp8 hidden, hidden_states.dtype
    # is fp8, so use compute_dtype (bf16) for the output instead of mirroring the input dtype.
    output_dtype = compute_dtype if is_fp8_hidden else hidden_states.dtype
    # LNC1: a single [T, H] buffer. The shared LNC2 kernel allocates [2, T, H] under
    # tensor-update accumulation so each core owns a private half that a closing reduce
    # merges; with one core there is nothing to merge, so accumulation happens in place.
    kernel_assert(init_output or output_tensor is not None, "init_output=False needs a caller-zeroed output_tensor")
    if output_tensor is not None:
        kernel_assert(
            output_tensor.shape[0] == dims.T and output_tensor.shape[1] == dims.H and output_tensor.dtype == output_dtype,
            f"output_tensor must be [{dims.T}, {dims.H}] {output_dtype}, got {output_tensor.shape} {output_tensor.dtype}",
        )
        output = output_tensor
    else:
        output = nl.ndarray((dims.T, dims.H), dtype=output_dtype, buffer=nl.shared_hbm)

    outs = OutputTensors(
        gate_up_activations_T=gate_up_activations_T,
        down_activations=down_activations,
        output=output,
    )
    """
    Allocate buffers for prefetching and current block processing.
    """

    hidden_qtz_sb = sbm.alloc_stack(
        (_pmax, prj_cfg.n_H512_tile, dims.B // SBUF_QUADRANT_SIZE, SBUF_QUADRANT_SIZE),
        dtype=configs.qtz_dtype,
        name="hidden_qtz_sb",
        align=SBUF_QUADRANT_SIZE,
    )
    if is_static_quant:
        # STATIC_MX: reuse the shared all-127 dummy [_pmax, _pmax] buffer.
        hidden_scale_sb = static_dummy_scale_sb
    else:
        # fp8 pre-quantized path: scales arrive packed (4 H512 tiles folded into one 128-wide block),
        # so the scale buffer holds n_packed blocks instead of n_H512_tile. The online-quant bf16 path
        # keeps one scale tile per H512 tile.
        n_scale_dim = div_ceil(prj_cfg.n_H512_tile, 4) if is_fp8_hidden else prj_cfg.n_H512_tile
        hidden_scale_sb = sbm.alloc_stack(
            (_pmax, n_scale_dim, dims.B // SBUF_QUADRANT_SIZE, SBUF_QUADRANT_SIZE),
            dtype=nl.uint8,
            name="hidden_scale_sb",
            align=SBUF_QUADRANT_SIZE,
        )

    # fp8 pre-quantized path: persistent buffer holding one block's gathered concat rows
    # ([hidden_quant (H fp8) | hidden_scale (packed: n_packed*128 uint8)] [| affinities (E bf16) | pad]
    # when affinities are fused) between the prefetch (DMA) and transpose (PE) stages. H and the packed
    # scale region are both multiples of 4, so the row is fp32-aligned for the transpose's fp8->fp32
    # reinterpret; the affinity tail (when present) is the full remaining concat width.
    block_hidden_concat = None
    if is_fp8_hidden:
        # Hold the FULL concat row so the gather pulls the (optional) affinity tail too.
        concat_free_size = hidden_states.shape[-1]
        if affinity_rows != None:
            concat_free_size = affinities_col_offset + affinity_rows.shape[-1] * sizeinbytes(affinity_rows.dtype)
        block_hidden_concat = sbm.alloc_stack(
            (_pmax, dims.B // _pmax, concat_free_size),
            dtype=hidden_states.dtype,
            name="block_hidden_concat",
            align=SBUF_QUADRANT_SIZE,
        )

        if is_affinities_packed and skip_dma.skip_token:
            nisa.memset(block_hidden_concat[:, :, affinities_col_offset:concat_free_size], value=0)

    down_weight_qtz = sbm.alloc_stack(
        (_pmax, prj_cfg.n_total_I512_tile, prj_cfg.H_sharded),
        dtype=inps.down_proj_weight.dtype,
        name="down_weight_qtz",
        align=SBUF_QUADRANT_SIZE,
    )

    # Memset weight if input weight HBM does not pad on par dim
    if dims.p_I != _pmax:
        nisa.memset(down_weight_qtz[:, prj_cfg.n_total_I512_tile - 1, :], value=0)

    if is_static_quant:
        # STATIC_MX: dummy 127 moving_scale for down matmul. Per-tile matmul
        # reads [_pmax, H_tile_size] per call. Caller's matmul site slices 2D under static_quant instead of 3D with n_I512 tiles.
        # n_packed_down only used by MX path; keep unset under STATIC_MX (use_packed_scales=False enforced).
        n_packed_down = 0
        down_scale_sb = sbm.alloc_stack(
            (_pmax, prj_cfg.H_tile_size),
            dtype=nl.uint8,
            name="down_scale_sb",
            align=SBUF_QUADRANT_SIZE,
        )
        nisa.memset(down_scale_sb, value=127)
    elif use_packed_scales:
        n_packed_down = (prj_cfg.n_total_I512_tile + SLOTS_PER_PACKED_BUFFER - 1) // SLOTS_PER_PACKED_BUFFER
        down_scale_sb = sbm.alloc_stack(
            (_pmax, n_packed_down, prj_cfg.H_sharded),
            dtype=nl.uint8,
            name="down_scale_packed_sb",
            align=SBUF_QUADRANT_SIZE,
        )
    else:
        n_packed_down = 0
        down_scale_sb = sbm.alloc_stack(
            (_pmax, prj_cfg.n_total_I512_tile, prj_cfg.H_sharded),
            dtype=nl.uint8,
            name="down_scale_sb",
            align=SBUF_QUADRANT_SIZE,
        )
        if dims.p_I != _pmax:
            nisa.memset(down_scale_sb[:, prj_cfg.n_total_I512_tile - 1, :], value=0)

    # STATIC_MX: dummy 127 stationary_scale for the down matmul intermediate.
    # Reuse the shared [_pmax, _pmax] all-127 dummy buffer (gup_scales_sb /
    # hidden_scale_sb point to the same backing). Matmul site reads as a 2D
    # per-tile view under static_quant.
    dummy_inter_scale_sb = static_dummy_scale_sb if is_static_quant else None

    # init counters
    # in shard-on-block we can move independently
    cond = (
        sbm.alloc_stack((1, 1), dtype=nl.int32, name="cond", align=SBUF_QUADRANT_SIZE)
        if configs.use_dynamic_while
        else None
    )
    index = (
        sbm.alloc_stack((1, 1), dtype=nl.int32, name="index", align=SBUF_QUADRANT_SIZE)
        if configs.use_dynamic_while
        else None
    )

    # Pre-allocate persistent gup tile buffer for cross-block prefetching.
    # Two layouts:
    #   - configs.gup_full_persistent: hold full gup weights (gate+up, full I)
    #     persistent across blocks. Cross-block prefetch loads the entire gup; same-expert
    #     blocks OOB-skip the prefetch Per-I-tile compute reads slices directly from this persistent buffer (no per-tile streaming).
    #   - Otherwise: small "tile 0 prefetch" buffer; per-I-tile compute streams
    #     subsequent tiles via gup_wt_a/gup_wt_b ping-pong.
    gup_tile_prefetch_buf = None
    if dims.I > _I_TILE_SZ:
        if configs.gup_full_persistent:
            gup_tile_prefetch_buf = sbm.alloc_stack(
                (_pmax, 2, prj_cfg.n_H512_tile_sharded, dims.I),
                dtype=inps.gate_up_proj_weight.dtype,
                name="gup_full_persistent",
                align=SBUF_QUADRANT_SIZE,
            )
        else:
            gup_tile_prefetch_buf = sbm.alloc_stack(
                (_pmax, 2, prj_cfg.n_H512_tile_sharded, _I_TILE_SZ),
                dtype=inps.gate_up_proj_weight.dtype,
                name="gup_tile_prefetch",
                align=SBUF_QUADRANT_SIZE,
            )
            if _PFT[0] > 1:
                # (prefetch_tiles) k persistent tile buffers: tile i of every block uses buffer i % k, and the
                # cross-block prefetch fills buffers 0 .. k-1 with the next block's tiles 0 .. k-1.
                kernel_assert(
                    div_ceil(dims.I, _I_TILE_SZ) >= _PFT[0] and prj_cfg.n_H512_tile_sharded == prj_cfg.n_H512_tile,
                    f"prefetch_tiles {_PFT[0]} needs at least that many {_I_TILE_SZ}-wide I tiles (I {dims.I}) and unsharded H",
                )
                _extra = tuple(
                    sbm.alloc_stack(
                        (_pmax, 2, prj_cfg.n_H512_tile_sharded, _I_TILE_SZ),
                        dtype=inps.gate_up_proj_weight.dtype,
                        name=f"gup_tile_prefetch{t}",
                        align=SBUF_QUADRANT_SIZE,
                    )
                    for t in range(1, _PFT[0])
                )
                _PF_BUFS[0] = (gup_tile_prefetch_buf,) + _extra
                if conditions != None:
                    _PF_LIVE[0] = _build_pf_live(inps, routing, dims, sbm, dims.E)

    buffers = SharedBuffers(
        block_hidden_states=None,
        block_hidden_states_T=None,
        hidden_qtz_sb=hidden_qtz_sb,
        hidden_scale_sb=hidden_scale_sb,
        # Blocks accumulate into the output with a DMA read-modify-write, so no SBUF copy
        # of the previous output is kept.
        block_old=None,
        down_weight_qtz=down_weight_qtz,
        down_scale_sb=down_scale_sb,
        cond=cond,
        index=index,
        token_4_H_indices_on_p=None,
        gup_tile_buf_a=gup_tile_prefetch_buf,
        dummy_inter_scale_sb=dummy_inter_scale_sb,
        block_hidden_concat=block_hidden_concat,
    )

    """
    END OF PREPARING DIMS, CONFIGS, SHARED_BUFFERS

    MAIN COMPUTATION STARTS
    """

    """Weight skipping: pre-compute skip mask and hoist buffers."""
    # When tiling is active, gup_tile_prefetch_buf is allocated. In tiled path we
    # don't hoist full weight buffers (weights are tile-loaded per I-tile) —
    # skip_weight hoists scales, bias, and down weights for OOB-skip reuse.
    _tiling_active = gup_tile_prefetch_buf != None
    if skip_dma.skip_weight:
        # Determine the actual number of static blocks that process_static_blocks will iterate
        # over, so the skip mask lines up with the block iteration order (LNC1: blocks
        # 0 .. _num_static_for_mask - 1, in order, on the one core).
        if configs.use_dynamic_while:
            if configs.n_static_blocks > 0:
                # Must match the n_static_blocks computed below (LNC1: taken as-is, with no
                # even-parity padding, since there is no second core to balance against).
                _num_static_for_mask = configs.n_static_blocks
            else:
                if conditions != None and (n_dynamic_blocks < 0 or n_dynamic_blocks > dims.N):
                    # Must match the auto-computed n_static_blocks below so the skip
                    # mask aligns with the block iteration order.
                    _num_static_for_mask = max(1, div_ceil(div_ceil(dims.T * top_k, ep_degree), dims.B))
                else:
                    _num_static_for_mask = dims.N - n_dynamic_blocks
        else:
            _num_static_for_mask = N

        if _num_static_for_mask > 0:
            """
            LNC1: the one core walks static blocks 0 .. _num_static_for_mask - 1 in order,
            so the mask is just block_to_expert[0 : _num_static_for_mask] with one extra
            tail slot.

            The tail slot exists because the loop's last iteration looks up
            all_experts_for_weights[i + 1] for its speculative next-block prefetch. Its value
            is left at E (memset) so that speculative prefetch is OOB-skipped, saving a DMA --
            the block it would prefetch for does not exist. (The shared LNC2 kernel puts the
            odd-N remainder block's real expert here for shard 1; LNC1 has no remainder block.)
            """
            n_static_mask_blocks = _num_static_for_mask
            n_mask_len = n_static_mask_blocks + 1
            n_mask_alloc = n_static_mask_blocks + 1

            all_experts = sbm.alloc_stack((1, n_mask_alloc), dtype=nl.int32, name="all_experts")
            nisa.memset(dst=all_experts, value=_skip_expert(E))
            # When dynamic blocks follow, the tail slot holds the first dynamic block's expert: the
            # last static block prefetches that block, skipping the weights if the expert repeats.
            n_mask_copy = n_static_mask_blocks + (1 if conditions != None and n_static_mask_blocks < N else 0)
            nisa.dma_copy(
                dst=all_experts[0:1, 0:n_mask_copy],
                src=block_to_expert.reshape((N, 1)).ap(pattern=[[1, 1], [1, n_mask_copy]], offset=0),
            )

            # Build weight-expert array: E (skip) where same expert as previous block.
            # Non-tiling: modify all_experts in-place (original not needed after).
            # Tiling: need original mapping for gup weight loads
            if _tiling_active:
                all_experts_for_weights = sbm.alloc_stack(
                    (1, n_mask_alloc), dtype=nl.int32, name="all_experts_for_weights"
                )
                nisa.tensor_copy(dst=all_experts_for_weights, src=all_experts)
                if n_mask_len > 1:
                    _compute_weight_skip_mask(sbm, all_experts, all_experts_for_weights, n_mask_len, _skip_expert(E))
            else:
                all_experts_for_weights = all_experts
                if n_mask_len > 1:
                    _compute_weight_skip_mask(sbm, all_experts, all_experts, n_mask_len, _skip_expert(E))
        else:
            all_experts_for_weights = None

        # Hoist weight/bias buffers so they persist across iterations. In tiled path, full gup weights don't persist (tile-loaded per I-tile) so we
        # only hoist scales, bias, and down weights for OOB-skip reuse.
        if _tiling_active:
            logger.debug("Weight skipping on tiling path: only skipping scales, bias, and down weight")
            hoisted_gup_weights = None
            # Hoist bias buffers so bias DMAs can be OOB-skipped on same-expert blocks
            # (weights stay in tile buffers and are reloaded per I-tile). Conditionally allocate based on if there is a bias a or not.
            hoisted_gup_bias = (
                sbm.alloc_stack(
                    (_pmax, 2, prj_cfg.n_total_I512_tile, _q_width),
                    dtype=inps.gate_and_up_proj_bias.dtype,
                    name="hoisted_gup_bias",
                    align=SBUF_QUADRANT_SIZE,
                )
                if gate_and_up_proj_bias != None
                else None
            )
            if hoisted_gup_bias != None and dims.I < _pmax * _q_width:
                # hoisted_gup_bias: [128_I, 2, n_total_I512, 4_I]
                nisa.memset(dst=hoisted_gup_bias[:, :, 0, :], value=0.0)

            hoisted_down_bias = (
                sbm.alloc_stack(
                    (1, dims.H),
                    dtype=inps.down_proj_bias.dtype,
                    name="hoisted_down_bias",
                    align=SBUF_QUADRANT_SIZE,
                )
                if down_proj_bias != None
                else None
            )
        else:
            hoisted_gup_weights = sbm.alloc_stack(
                (_pmax, 2, prj_cfg.n_H512_tile_sharded, dims.I),
                dtype=inps.gate_up_proj_weight.dtype,
                name="hoisted_gup_weights",
                align=SBUF_QUADRANT_SIZE,
            )
            hoisted_gup_bias = (
                sbm.alloc_stack(
                    (_pmax, 2, prj_cfg.n_total_I512_tile, _q_width),
                    dtype=inps.gate_and_up_proj_bias.dtype,
                    name="hoisted_gup_bias",
                    align=SBUF_QUADRANT_SIZE,
                )
                if gate_and_up_proj_bias != None
                else None
            )
            if hoisted_gup_bias != None and dims.I < _pmax * _q_width:
                nisa.memset(dst=hoisted_gup_bias[:, :, 0, :], value=0.0)
            hoisted_down_bias = (
                sbm.alloc_stack(
                    (1, dims.H),
                    dtype=inps.down_proj_bias.dtype,
                    name="hoisted_down_bias",
                    align=SBUF_QUADRANT_SIZE,
                )
                if down_proj_bias != None
                else None
            )
    else:
        # not doing any skip_weight
        all_experts_for_weights = None
        hoisted_gup_weights = None
        hoisted_gup_bias = None
        hoisted_down_bias = None

    # Deferred-accumulate buffers: each block's output is added into HBM at the start of the next
    # block (see compute_one_block) and the last one after the loops. A DMA still in flight at a
    # dynamic-loop entry or back-edge would make every engine drain there.
    pending_out = sbm.alloc_stack(
        (_pmax, dims.n_B128_tiles, dims.H), dtype=nl.bfloat16, name="pending_out", align=SBUF_QUADRANT_SIZE
    )

    if configs.use_dynamic_while:
        if configs.n_static_blocks > 0:
            kernel_assert(
                configs.n_static_blocks < dims.N,
                f"Cannot have more static blocks than total number of blocks. Got ({configs.n_static_blocks}) > N = {dims.N}",
            )
            # LNC1: no parity padding. The shared LNC2 kernel rounds the dynamic block count
            # up to an even number so the two cores get equal work; one core needs no such
            # alignment, so the split is exactly as requested.
            n_dynamic_blocks = dims.N - configs.n_static_blocks
            n_static_blocks = configs.n_static_blocks
            n_dynamic_blocks_local = n_dynamic_blocks
            auto_computed = False
        else:
            # If invalid n_dynamic_blocks is passed, auto-calculate best case combination.
            # Always process at least one static block
            if conditions != None and (n_dynamic_blocks < 0 or n_dynamic_blocks > dims.N):
                # Best case: tokens spread top_k ways, sharded across ep_degree EP ranks,
                # packed into B-sized blocks. ceil(ceil(T*top_k/ep_degree)/B).
                n_static_blocks = max(1, div_ceil(div_ceil(dims.T * top_k, ep_degree), dims.B))
                n_dynamic_blocks_local = dims.N - n_static_blocks
                auto_computed = True
            else:
                n_dynamic_blocks_local = n_dynamic_blocks
                n_static_blocks = dims.N - n_dynamic_blocks_local
                auto_computed = False

        # LNC1: process_dynamic_blocks steps the block index by 1, so any positive dynamic
        # block count is processable. (The shared LNC2 kernel steps by num_shards == 2 and
        # therefore has to push a 1-block residual into the static path here.)

        if configs.n_static_blocks <= 0 and auto_computed:
            logger.debug(
                f"n_dynamic_blocks={n_dynamic_blocks} out of range, auto-computing from T={dims.T}, B={dims.B}: "
                f"{n_static_blocks} static, {n_dynamic_blocks_local} dynamic"
            )

        # base index for dynamic blocks to start
        nisa.memset(dst=buffers.index[0, 0], value=n_static_blocks)

        logger.debug(f"Processing {n_static_blocks} static blocks, {n_dynamic_blocks_local} dynamic blocks")
        # When n_static_blocks==0, process_static_blocks is skipped but output_initialization
        # (which zeros the output tensor for accumulation) lives inside it. Initialize here
        # so dynamic-only paths don't read uninitialized shared DRAM.
        if n_static_blocks == 0 and configs.is_tensor_update_accumulating:
            if init_output:
                H = dims.H
                zeros = sbm.alloc_heap(
                    (_pmax, H), dtype=nl.bfloat16, name="output_init_zeros_dyn", align=SBUF_QUADRANT_SIZE
                )
                nisa.memset(zeros, value=0.0)
                output_initialization(outs.output, dims, sbm=sbm, zeros=zeros)
                sbm.pop_heap()  # free zeros
            # The first dynamic block flushes a "previous block" (clamped to block 0) that never
            # ran; a zero pending output makes that flush a no-op.
            nisa.memset(dst=pending_out, value=0.0)

        def _dyn_setup():
            # the dynamic loop setup, issued after the static blocks
            """
            Precompute dynamic-for iteration counts at top level

            Chunked scheme (LNC1: one block per remainder iter, _DYN_STEP == _DYN_INNER):
              n_outer_iters = n_active // _DYN_STEP   (each outer iter = _DYN_INNER blocks)
              n_rem_iters   = n_active - n_outer_iters * _DYN_INNER   (= n_active % _DYN_STEP)
            The chunked outer loop consumes n_outer_iters * _DYN_INNER blocks; the remainder
            loop walks the leftover 0 .. _DYN_STEP-1 blocks one at a time.
            """
            dyn_conds_sb = sbm.alloc_stack(
                (1, n_dynamic_blocks_local), dtype=nl.int32, name="dyn_conds", align=SBUF_QUADRANT_SIZE
            )
            nisa.dma_copy(
                dst=dyn_conds_sb,
                src=inps.conditions.reshape((inps.conditions.shape[0], 1)).ap(
                    pattern=[[1, 1], [1, n_dynamic_blocks_local]], offset=n_static_blocks
                ),
            )
            # number of active dynamic blocks at runtime
            n_active_sb = sbm.alloc_stack((1, 1), dtype=nl.int32, name="n_active", align=SBUF_QUADRANT_SIZE)
            nisa.tensor_reduce(dst=n_active_sb, data=dyn_conds_sb, op=nl.add, axis=1)

            # n_outer_iters = n_active // _DYN_STEP (shift-by-log2 since _DYN_STEP is a power of 2)
            n_outer_iters_sb = sbm.alloc_stack((1, 1), dtype=nl.int32, name="n_outer_iters", align=SBUF_QUADRANT_SIZE)

            # floor division by _DYN_STEP
            nisa.tensor_scalar(
                dst=n_outer_iters_sb,
                data=n_active_sb,
                op0=nl.right_shift,
                operand0=_DYN_STEP_LOG2,
            )

            # n_rem_iters = n_active - n_outer_iters * _DYN_INNER
            # LNC1: the remainder loop does one block per iter, so the total iteration count
            # is n_active itself. (The shared LNC2 kernel does 2 blocks per iter and therefore
            # uses n_half = (n_active + 1) >> 1 here.)
            n_outer_scaled_sb = sbm.alloc_stack((1, 1), dtype=nl.int32, name="n_outer_scaled", align=SBUF_QUADRANT_SIZE)
            nisa.tensor_scalar(dst=n_outer_scaled_sb, data=n_outer_iters_sb, op0=nl.multiply, operand0=_DYN_INNER)
            n_rem_iters_sb = sbm.alloc_stack((1, 1), dtype=nl.int32, name="n_rem_iters", align=SBUF_QUADRANT_SIZE)
            nisa.tensor_tensor(dst=n_rem_iters_sb, data1=n_active_sb, data2=n_outer_scaled_sb, op=nl.subtract)

            outer_reg = nisa.register_alloc()
            nisa.register_load(outer_reg, n_outer_iters_sb)
            rem_reg = nisa.register_alloc()
            nisa.register_load(rem_reg, n_rem_iters_sb)

            """
            Weight skipping for dynamic blocks: build two independent masks, one
            for the chunked region and one for the remainder region.
            Each mask gets its own _compute_weight_skip_mask pass, so position 0
            of each mask holds the raw expert (no predecessor to compare against)
            and the first block of each region always loads weights fresh. The
            chunked->rem boundary is therefore handled implicitly without any
            runtime seam comparison.
            """
            _use_dyn_weight_skip = configs.skip_dma.skip_weight
            chunked_experts_mask = None
            rem_experts_mask = None
            dyn_weight_expert_sb = None
            chunked_iter_sb = None
            rem_iter_sb = None
            n_chunked_alloc = 0
            n_rem_alloc = 0
            if _use_dyn_weight_skip:
                # Worst-case sizing (compile-time).
                # Chunked: (n_dynamic_blocks_local // _DYN_STEP) * _DYN_INNER blocks.
                n_chunks_max = n_dynamic_blocks_local // _DYN_STEP
                n_chunked_alloc = n_chunks_max * _DYN_INNER
                # Remainder: after chunking, at most (_DYN_STEP - 1) active blocks remain,
                # and LNC1's remainder loop takes one block per iter.
                n_rem_alloc = min(n_dynamic_blocks_local, _DYN_STEP - 1)

                # --- Chunked mask ---
                if n_chunked_alloc > 0:
                    chunked_experts_mask = _sbm_alloc(
                        sbm,
                        (1, n_chunked_alloc),
                        dtype=nl.int32,
                        name="chunked_experts_mask",
                        align=SBUF_QUADRANT_SIZE,
                    )
                    nisa.memset(dst=chunked_experts_mask, value=_skip_expert(dims.E))

                    sbm.open_scope(name="chunked_weight_skip_mask")
                    prev_prefix = sbm.get_name_prefix()
                    sbm.set_name_prefix(f"{prev_prefix}chunk_")

                    # LNC1: the chunked block at position k (0..n_chunked_alloc-1) is
                    #   n_static_blocks + (k // _DYN_INNER) * _DYN_STEP + (k % _DYN_INNER)
                    # which, since _DYN_STEP == _DYN_INNER on one core, is simply
                    # n_static_blocks + k. The outer-chunk x inner pattern is kept (it degenerates
                    # to a contiguous read) so this mirrors the shared kernel's structure.
                    nisa.dma_copy(
                        dst=chunked_experts_mask[0:1, 0:n_chunked_alloc],
                        src=inps.block_to_expert.reshape((1, dims.N)).ap(
                            pattern=[
                                [1, 1],
                                [_DYN_STEP, n_chunks_max],
                                [1, _DYN_INNER],
                            ],
                            offset=n_static_blocks,
                        ),
                    )

                    if n_chunked_alloc > 1:
                        if _tiling_active:
                            chunked_all_experts = _sbm_alloc(
                                sbm,
                                (1, n_chunked_alloc),
                                dtype=nl.int32,
                                name="chunked_all_experts",
                                align=SBUF_QUADRANT_SIZE,
                            )
                            nisa.tensor_copy(dst=chunked_all_experts, src=chunked_experts_mask)
                            _compute_weight_skip_mask(
                                sbm, chunked_all_experts, chunked_experts_mask, n_chunked_alloc, _skip_expert(dims.E)
                            )
                        else:
                            _compute_weight_skip_mask(
                                sbm, chunked_experts_mask, chunked_experts_mask, n_chunked_alloc, _skip_expert(dims.E)
                            )

                    sbm.set_name_prefix(prev_prefix)
                    sbm.close_scope()

                # --- Remainder mask ---
                """
                The remainder region starts at a runtime-computed offset
                (n_static_blocks + n_outer_iters * _DYN_STEP). DMA-load using a
                scalar_offset derived from n_outer_iters at runtime.
                Allocate one extra sentinel slot at the tail so the prefetch's
                next-expert lookup at the last rem iter (rem_iter+1
                walking past n_rem_alloc - 1) reads E (skip) safely. The
                prefetch result at the last rem iter is never consumed
                (next_block_idx is clamped to N-1 with no further compute), so
                the sentinel value is functionally a don't-care.
                """
                # Records how many rem iters the rem_base clamp consumed,
                # so the rem loop's iter counters start at that shift and
                # iter 0 indexes the first real rem-region block.
                rem_iter_shift_sb = None
                if n_rem_alloc > 0:
                    rem_experts_mask = _sbm_alloc(
                        sbm,
                        (1, n_rem_alloc + 1),
                        dtype=nl.int32,
                        name="rem_experts_mask",
                        align=SBUF_QUADRANT_SIZE,
                    )
                    nisa.memset(dst=rem_experts_mask, value=_skip_expert(dims.E))
                    rem_iter_shift_sb = _sbm_alloc(
                        sbm,
                        (1, 1),
                        dtype=nl.uint32,
                        name="rem_iter_shift",
                        align=SBUF_QUADRANT_SIZE,
                    )

                    sbm.open_scope(name="rem_weight_skip_mask")
                    prev_prefix = sbm.get_name_prefix()
                    sbm.set_name_prefix(f"{prev_prefix}rem_")

                    # rem_base_sb (uint32) = n_static_blocks + n_outer_iters * _DYN_STEP
                    rem_base_sb = _sbm_alloc(sbm, (1, 1), dtype=nl.uint32, name="rem_base", align=SBUF_QUADRANT_SIZE)
                    nisa.tensor_scalar(
                        dst=rem_base_sb,
                        data=n_outer_iters_sb,
                        op0=nl.multiply,
                        operand0=_DYN_STEP,
                        op1=nl.add,
                        operand1=n_static_blocks,
                    )

                    """
                    The mask DMA reads n_rem_alloc CONSECUTIVE entries starting at rem_base (LNC1
                    processes every block, so the stride is 1). If any lane goes past N-1,
                    oob_mode.skip aborts the whole DMA — so clamp rem_base down to a safe start.
                    With a unit stride every base is a block this core processes, so no alignment
                    correction is needed (the shared LNC2 kernel must additionally re-align the
                    clamped base onto its shard's parity).
                    """
                    _max_safe_base = dims.N - 1 - (n_rem_alloc - 1)
                    # rem_iter_shift_sb tracks how many iters the clamp consumed,
                    # so iter 0 of the rem loop still indexes the first real rem block.
                    rem_clamped_base_sb = _sbm_alloc(
                        sbm,
                        (1, 1),
                        dtype=nl.uint32,
                        name="rem_clamped_base",
                        align=SBUF_QUADRANT_SIZE,
                    )
                    nisa.tensor_scalar(
                        dst=rem_clamped_base_sb,
                        data=rem_base_sb,
                        op0=nl.minimum,
                        operand0=_max_safe_base,
                    )
                    # rem_iter_shift_sb = rem_base - clamped_base (unit stride -> no divide)
                    nisa.tensor_tensor(
                        dst=rem_iter_shift_sb,
                        data1=rem_base_sb,
                        data2=rem_clamped_base_sb,
                        op=nl.subtract,
                    )

                    nisa.dma_copy(
                        dst=rem_experts_mask[0:1, 0:n_rem_alloc],
                        src=inps.block_to_expert.ap(
                            pattern=[
                                [1, 1],
                                [1, n_rem_alloc],
                            ],
                            offset=0,
                            scalar_offset=rem_clamped_base_sb,
                            indirect_dim=0,
                        ),
                        oob_mode=oob_mode.skip,
                    )

                    if n_rem_alloc > 1:
                        # Mirror the +1 sentinel slot in the tiling shadow
                        # buffer so tensor_copy shapes match. Skip-mask range
                        # stays n_rem_alloc; the sentinel slot keeps its E.
                        if _tiling_active:
                            rem_all_experts = _sbm_alloc(
                                sbm,
                                (1, n_rem_alloc + 1),
                                dtype=nl.int32,
                                name="rem_all_experts",
                                align=SBUF_QUADRANT_SIZE,
                            )
                            nisa.tensor_copy(dst=rem_all_experts, src=rem_experts_mask)
                            _compute_weight_skip_mask(sbm, rem_all_experts, rem_experts_mask, n_rem_alloc, _skip_expert(dims.E))
                        else:
                            _compute_weight_skip_mask(sbm, rem_experts_mask, rem_experts_mask, n_rem_alloc, _skip_expert(dims.E))

                    sbm.set_name_prefix(prev_prefix)
                    sbm.close_scope()

                # Per-block scratch for current weight-expert lookup.
                dyn_weight_expert_sb = _sbm_alloc(
                    sbm, (1, 1), dtype=nl.int32, name="dyn_weight_expert_sb", align=SBUF_QUADRANT_SIZE
                )
                # Iter counters for chunked and remainder masks.
                chunked_iter_sb = _sbm_alloc(
                    sbm, (1, 1), dtype=nl.uint32, name="chunked_iter", align=SBUF_QUADRANT_SIZE
                )
                nisa.memset(dst=chunked_iter_sb, value=0)
                rem_iter_sb = _sbm_alloc(sbm, (1, 1), dtype=nl.uint32, name="rem_iter", align=SBUF_QUADRANT_SIZE)
                if n_rem_alloc > 0:
                    # Init iter counter to the rem-base clamp shift so iter 0
                    # of the rem loop indexes the slot holding the actual
                    # first rem-region block's expert.
                    nisa.tensor_copy(dst=rem_iter_sb, src=rem_iter_shift_sb)
                else:
                    nisa.memset(dst=rem_iter_sb, value=0)
            return dict(
                outer_reg=outer_reg, rem_reg=rem_reg, n_outer_iters_sb=n_outer_iters_sb, chunked_experts_mask=chunked_experts_mask,
                rem_experts_mask=rem_experts_mask, dyn_weight_expert_sb=dyn_weight_expert_sb, chunked_iter_sb=chunked_iter_sb,
                rem_iter_sb=rem_iter_sb, n_chunked_alloc=n_chunked_alloc, n_rem_alloc=n_rem_alloc,
            )

        _dyn_kw = None

        if n_static_blocks > 0:
            process_static_blocks(
                dims=dims,
                configs=configs,
                prj_cfg=prj_cfg,
                inps=inps,
                outs=outs,
                buffers=buffers,
                num_static_blocks=n_static_blocks,
                sbm=sbm,
                routing=routing,
                pending_out=pending_out,
                all_experts_for_weights=all_experts_for_weights,
                hoisted_gup_weights=hoisted_gup_weights,
                hoisted_gup_bias=hoisted_gup_bias,
                hoisted_down_bias=hoisted_down_bias,
                is_tensor_update_accumulating=configs.is_tensor_update_accumulating,
                prefetch_dynamic=n_dynamic_blocks_local > 0,
                init_output=init_output,
            )

        if n_dynamic_blocks_local > 0:
            if _dyn_kw is None:
                _dyn_kw = _dyn_setup()
            process_dynamic_blocks(
                dims=dims,
                configs=configs,
                prj_cfg=prj_cfg,
                inps=inps,
                outs=outs,
                buffers=buffers,
                num_static_blocks=n_static_blocks,
                num_dynamic_blocks=n_dynamic_blocks_local,
                sbm=sbm,
                routing=routing,
                pending_out=pending_out,
                hoisted_gup_weights=hoisted_gup_weights,
                hoisted_gup_bias=hoisted_gup_bias,
                hoisted_down_bias=hoisted_down_bias,
                **_dyn_kw,
                first_prefetched=n_static_blocks > 0,
            )

    else:
        """
        STATIC LOOP OVER ALL BLOCKS
        """
        process_static_blocks(
            dims=dims,
            configs=configs,
            prj_cfg=prj_cfg,
            inps=inps,
            outs=outs,
            buffers=buffers,
            num_static_blocks=dims.N,
            sbm=sbm,
            routing=routing,
            pending_out=pending_out,
            all_experts_for_weights=all_experts_for_weights,
            hoisted_gup_weights=hoisted_gup_weights,
            hoisted_gup_bias=hoisted_gup_bias,
            hoisted_down_bias=hoisted_down_bias,
            is_tensor_update_accumulating=configs.is_tensor_update_accumulating,
            init_output=init_output,
        )

    # Accumulate the last block's deferred output. After the dynamic loops buffers.index is one past
    # the last processed block.
    if configs.use_dynamic_while:
        last_idx = _sbm_alloc(sbm, (1, 1), dtype=nl.int32, name="last_block_idx", align=SBUF_QUADRANT_SIZE)
        nisa.tensor_scalar(dst=last_idx, data=buffers.index, op0=nl.subtract, operand0=1, op1=nl.maximum, operand1=0)
    else:
        last_idx = dims.N - 1
    _flush_pending_output(routing, last_idx, pending_out, outs, dims, sbm)

    # LNC1: no cross-core epilogue. The shared LNC2 kernel ends with a core_barrier plus a
    # split cross-shard reduce that folds output[1] into output[0]; with a single core the
    # accumulated [T, H] buffer already IS the final result.

    sbm.close_scope()

    return output


def _extract_block_affinity(
    block_hidden_concat,
    block_expert,
    dims,
    kernel_cfg,
    sbm,
    name_prefix="aff",
    local_expert_start_idx=None,
    n_global_experts=None,
):
    """Extract this block's expert-affinity column from the gathered fp8 concat rows (packed path).

    When the producer fused the dense [T, E] affinities into the row (is_affinities_packed), each
    gathered row is [hidden | scale | affinities (E_global x kernel_cfg.expert_affinities_dtype) | pad].
    This is a pure on-chip op (one dynamic-offset tensor_copy per B128 tile) -- NOT a DMA -- so it
    replaces the indirect affinity gather (calculate_expert_affinities).

    The affinity region is dense over every GLOBAL expert because the producer's router runs before the
    EP shard, while block_expert is a LOCAL id. With local_expert_start_idx (== ep_rank * E_local) the
    column is block_expert + local_expert_start_idx; without it the local id is the column (EP=1).

    block_hidden_concat is [_pmax, n_B_tiles, concat_free_size] fp8. The affinity region starts at fp8
    column kernel_cfg.affinities_col_offset, i.e. affinity-dtype column affinities_col_offset //
    sizeinbytes(dtype).

    Returns a list of n_B_tiles tensors, each [_pmax, 1] fp32 -- the same shape contract as
    calculate_expert_affinities, so the per-block compute consumes expert_affinity[n] identically.
    """
    affin_dtype = kernel_cfg.expert_affinities_dtype
    n_B_tiles = dims.B // _pmax
    aff_col = kernel_cfg.affinities_col_offset // sizeinbytes(affin_dtype)
    # Static window width: every global expert. Slicing to end-of-row instead makes an AP whose
    # partition step equals the row width, which the ap() verifier rejects.
    aff_col_end = aff_col + (n_global_experts if n_global_experts != None else dims.E)

    concat_view = block_hidden_concat.view(affin_dtype)

    # Dynamic select requires a uint32 index (TensorCopyDynamicSrc verifier); expert ids are >= 0.
    if local_expert_start_idx != None:
        global_expert = _sbm_alloc(
            sbm, (1, 1), dtype=nl.int32, name=f"{name_prefix}_global_expert", align=SBUF_QUADRANT_SIZE
        )
        nisa.tensor_tensor(dst=global_expert, data1=block_expert, data2=local_expert_start_idx, op=nl.add)
        select_index_u32 = global_expert.view(nl.uint32)
    else:
        select_index_u32 = block_expert.view(nl.uint32)

    expert_affinity = []
    for n in range(n_B_tiles):
        affinity_f32 = _sbm_alloc(
            sbm, (_pmax, 1), dtype=nl.float32, name=f"{name_prefix}_affinity_t{n}", align=SBUF_QUADRANT_SIZE
        )
        # [_pmax, n_B_tiles, free] -> B-tile n -> affinity window -> this block's expert column -> [_pmax, 1]
        aff_tv = (
            concat_view.slice(1, n, n + 1)
            .squeeze_dim(1)
            .slice(1, aff_col, aff_col_end)
            .select(dim=1, index=select_index_u32)
            .expand_dim(1)
        )
        nisa.tensor_copy(dst=affinity_f32, src=aff_tv, engine=nisa.vector_engine)
        expert_affinity.append(affinity_f32)
    return expert_affinity


def _gather_gup_in_quant_recip(inps, block_expert, dims, sbm, name_prefix=""):
    """STATIC_MX: pull 1/in_scale[block_expert] from slot 0 of gup_scale_lut_sb."""
    # scalar_offset requires uint32; reinterpret block_expert (int32 [1,1]) in place (same-size bitcast).
    block_expert_u32 = block_expert.view(nl.uint32)
    in_quant_recip = _sbm_alloc(
        sbm, (_pmax, 1), dtype=nl.float32, name=f"{name_prefix}in_quant_recip", align=SBUF_QUADRANT_SIZE
    )
    nisa.tensor_copy(
        dst=in_quant_recip,
        src=inps.gup_scale_lut_sb.ap(
            pattern=[[dims.E * 3, _pmax], [3, 1], [1, 1]],
            offset=0,
            scalar_offset=block_expert_u32,
            indirect_dim=1,
        ),
    )
    return in_quant_recip


_WEIGHT_SKIP_EXPERT = [None]  # this call's weight_skip_expert (set at kernel entry)
_AFF_ROWS = [None]  # this call's affinity_rows (set at kernel entry)
# (``weight_stream``) this call's gate/up weight-stream options (set at kernel entry): [tile-major HBM layout
# (0 off, 1 = the H-chunked streams read it, 2 = one DMA per 512-wide I tile), the weight DMAs' QoS priority (None = the default
# class), odd I-tiles' weight DMAs on the second HWDGE queue (the Scalar sequencer)].
_WSTREAM = [0, None, False]
# (``prefetch_tiles``) this call's cross-block prefetch depth k, its k persistent gate/up tile
# buffers (tuple, buffer 0 = gup_tile_buf_a; None when k = 1 or the tiled path is off), the [1, N] int32 prefetch expert of
# every block (its expert when live, else the OOB skip sentinel; None without conditions).
_PFT = [1]
_PF_BUFS = [None]
_PF_LIVE = [None]


def _pf_bufs():
    """(prefetch_tiles) The k persistent gate/up tile buffers of this call, or None (the tile-0-only scheme)."""
    return _PF_BUFS[0]


def _pf_expert_live(next_block_idx, dims, sbm, name):
    """(prefetch_tiles) ``[1, 1]`` int32 expert for the next block's gate/up prefetch: its expert when the block is live, the
    OOB skip sentinel when it is not (the prefetch's data would never be read). None without the live table."""
    if _PF_LIVE[0] is None:
        return None
    e = _sbm_alloc(sbm, (1, 1), dtype=nl.int32, name=name, align=SBUF_QUADRANT_SIZE)
    if isinstance(next_block_idx, int):
        nisa.tensor_copy(dst=e, src=_PF_LIVE[0][0:1, next_block_idx : next_block_idx + 1])
    else:
        nisa.tensor_copy(
            dst=e,
            src=_PF_LIVE[0].ap(pattern=[[dims.N, 1], [1, 1]], offset=0, scalar_offset=next_block_idx.view(nl.uint32), indirect_dim=1),
        )
    return e


def _build_pf_live(inps, routing, dims, sbm, E):
    """(prefetch_tiles) ``[1, N]`` int32: ``block_expert[n]`` where ``conditions[n]``, else the skip sentinel, by integer DVE
    arithmetic (``(be - S) * c + S``; every value < 2^21, exact on the vector engine)."""
    S = _skip_expert(E)
    c = sbm.alloc_stack((1, dims.N), dtype=nl.int32, name="pf_live_cond", align=SBUF_QUADRANT_SIZE)
    nisa.dma_copy(dst=c, src=inps.conditions.reshape((inps.conditions.shape[0], 1)).ap(pattern=[[1, 1], [1, dims.N]], offset=0))
    t = sbm.alloc_stack((1, dims.N), dtype=nl.int32, name="pf_live_expert", align=SBUF_QUADRANT_SIZE)
    ve = dict(engine=nisa.engine.vector)
    nisa.tensor_scalar(dst=t, data=routing.block_expert, op0=nl.subtract, operand0=S, **ve)
    nisa.tensor_tensor(dst=t, data1=t, data2=c, op=nl.multiply, **ve)
    nisa.tensor_scalar(dst=t, data=t, op0=nl.add, operand0=S, **ve)
    return t


def _prefetch_gup_tiles_rest(inps, expert, prj_cfg, skip_dma, dims, sbm):
    """(prefetch_tiles) Prefetch I tiles 1 .. k-1 of ``expert``'s gate/up weights into persistent tile buffers 1 .. k-1 (tile 0
    and the scales go through ``_prefetch_gup_tile0``). OOB-skipped for the skip sentinel."""
    bufs = _pf_bufs()
    if bufs is None:
        return
    for t in range(1, len(bufs)):
        _load_gup_weight_tile(
            inps=inps, block_expert=expert, prj_cfg=prj_cfg, skip_dma=skip_dma, dst_weight=bufs[t], dst_scale=None,
            dst_bias=None, token_indices_on_p=None, I_offset=t * _I_TILE_SZ, dims=dims, sbm=sbm,
        )


def _wkw(i_tile=0):
    """``nisa.dma_copy`` keyword arguments of a gate/up / down weight DMA of I-tile ``i_tile`` under ``weight_stream``:
    the QoS priority, and the Scalar sequencer's HWDGE queue for odd I-tiles. {} when unset (unchanged)."""
    kw = {}
    if _WSTREAM[1] != None:
        kw["priority"] = _WSTREAM[1]
    if _WSTREAM[2] and i_tile % 2 == 1:
        kw["engine"] = nisa.engine.scalar
    return kw


def _gup_tile_src(inps, expert, i_tile: int):
    """(tile-major ``weight_stream``) ``[_pmax, 2, n_H512, 512]`` HBM view of I-tile ``i_tile`` of ``expert``'s
    gate/up weights. The host stored ``[E, 128, 2, n_H512, I]`` in the byte order ``[E][I / 512][128][2][n_H512][512]``
    (``layout/prefill/weight_loaders._gate_up_tile_major``), so the ``[E, I / 512, 128, 2, n_H512, 512]`` view of the same
    bytes holds at ``[e, t, p, g, h, i]`` the value the default layout holds at ``[e, p, g, h, 512 t + i]``: the SBUF image of
    every load is unchanged, and a partition's part of a tile is one 24 KiB run (a 3-H512-tile stream chunk one 3 KiB run)
    instead of 1 KiB pieces 4 KiB apart."""
    w = inps.gate_up_proj_weight
    E, P, two, nh, I = w.shape
    w6 = w.reshape((E, I // _I_TILE_SZ, P, two, nh, _I_TILE_SZ))
    return w6.select(dim=0, index=expert).select(dim=0, index=i_tile)

# (compact_affinities) this call's (ep_rank HBM [1, 1], E_local), or None (set at kernel entry).
_AFF8 = [None]
_AFF8_K = 8


def _aff8_global_table(inps, dims, sbm):
    """(compact_affinities) ``[_pmax, N]`` fp32: the global expert id of every block (block_to_expert + ep_rank * E_local)
    on every partition, by two broadcast loads of the HBM tables (partition stride 0) once per call."""
    ep_hbm, e_local = _AFF8[0]
    N = dims.N
    bte = sbm.alloc_stack((_pmax, N), dtype=nl.int32, name="aff8_bte", align=SBUF_QUADRANT_SIZE)
    nisa.dma_copy(dst=bte, src=inps.block_to_expert.reshape((1, N)).ap(pattern=[[0, _pmax], [1, N]], offset=0))
    ep_bc = sbm.alloc_stack((_pmax, 1), dtype=nl.int32, name="aff8_ep", align=SBUF_QUADRANT_SIZE)
    nisa.dma_copy(dst=ep_bc, src=ep_hbm.reshape((1, 1)).ap(pattern=[[0, _pmax], [1, 1]], offset=0))
    nisa.tensor_scalar(dst=ep_bc, data=ep_bc, op0=nl.multiply, operand0=e_local, engine=nisa.vector_engine)
    # A tensor operand of an arithmetic tensor_scalar must be fp32 (MLIR verifier): add in fp32 (ids < 2^24, exact).
    ep_f = sbm.alloc_stack((_pmax, 1), dtype=nl.float32, name="aff8_ep_f", align=SBUF_QUADRANT_SIZE)
    nisa.tensor_copy(dst=ep_f, src=ep_bc)
    gtab = sbm.alloc_stack((_pmax, N), dtype=nl.float32, name="aff8_gtab", align=SBUF_QUADRANT_SIZE)
    nisa.tensor_copy(dst=gtab, src=bte)
    nisa.tensor_scalar(dst=gtab, data=gtab, op0=nl.add, operand0=ep_f[0:_pmax, 0:1], engine=nisa.vector_engine)
    return gtab


def _extract_block_affinity_compact(block_hidden_concat, block_idx, routing, dims, kernel_cfg, sbm, name_prefix="aff8"):
    """(compact_affinities) ``_extract_block_affinity`` for the compact top-8 tail: per B128 tile, the affinity of the
    block's global expert g is ``max_k (ids[k] == g) * vals[k]`` ([_pmax, 1] fp32 each, the same contract)."""
    n_B_tiles = dims.B // _pmax
    aff_col = kernel_cfg.affinities_col_offset // sizeinbytes(nl.float32)
    concat_view = block_hidden_concat.view(nl.float32)
    g_bc = _sbm_alloc(sbm, (_pmax, 1), dtype=nl.float32, name=f"{name_prefix}_g", align=SBUF_QUADRANT_SIZE)
    if isinstance(block_idx, int):
        nisa.tensor_copy(dst=g_bc, src=routing.gtab[0:_pmax, block_idx : block_idx + 1])
    else:
        nisa.tensor_copy(
            dst=g_bc,
            src=routing.gtab.ap(
                pattern=[[dims.N, _pmax], [1, 1]], offset=0, scalar_offset=block_idx.view(nl.uint32), indirect_dim=1
            ),
        )
    expert_affinity = []
    for n in range(n_B_tiles):
        tile = concat_view.slice(1, n, n + 1).squeeze_dim(1)
        ids = tile.slice(1, aff_col, aff_col + _AFF8_K)
        vals = tile.slice(1, aff_col + _AFF8_K, aff_col + 2 * _AFF8_K)
        eq = _sbm_alloc(sbm, (_pmax, _AFF8_K), dtype=nl.float32, name=f"{name_prefix}_eq{n}", align=SBUF_QUADRANT_SIZE)
        nisa.tensor_scalar(dst=eq, data=ids, op0=nl.equal, operand0=g_bc[0:_pmax, 0:1], engine=nisa.vector_engine)
        nisa.tensor_tensor(dst=eq, data1=eq, data2=vals, op=nl.multiply, engine=nisa.vector_engine)
        affinity_f32 = _sbm_alloc(
            sbm, (_pmax, 1), dtype=nl.float32, name=f"{name_prefix}_affinity_t{n}", align=SBUF_QUADRANT_SIZE
        )
        nisa.tensor_reduce(dst=affinity_f32, op=nl.maximum, data=eq, axis=1, keepdims=True)
        expert_affinity.append(affinity_f32)
    return expert_affinity


def _load_fp8_rows(inps, dims, skip_dma, token_indices_on_p, block_hidden_concat):
    """load_fp8_hidden_states_mx, plus (affinity_rows) the affinity tail of each gathered row from its own [T, E_global] rows:
    one more indirect DMA per B128 tile into the concat row's tail columns, so the row matches the fused one byte for byte."""
    load_fp8_hidden_states_mx(
        inps, dims, skip_dma, token_indices_on_p=token_indices_on_p, block_hidden_concat=block_hidden_concat
    )
    aff = _AFF_ROWS[0]
    if aff == None:
        return
    c0 = inps.hidden_states.shape[-1]
    n_aff = aff.shape[-1]
    tail = block_hidden_concat.view(aff.dtype)
    t0 = c0 // sizeinbytes(aff.dtype)
    n_B_tiles = div_ceil(dims.B, _pmax)
    blk_T = min(dims.B, _pmax)
    for b_tile_idx in range(n_B_tiles):
        nisa.dma_copy(
            src=aff.ap(
                pattern=[[n_aff, blk_T], [1, n_aff]],
                offset=0,
                vector_offset=token_indices_on_p.ap(pattern=[[n_B_tiles, blk_T], [1, 1]], offset=b_tile_idx),
                indirect_dim=0,
            ),
            dst=tail[0:blk_T, b_tile_idx, t0 : t0 + n_aff],
            oob_mode=nisa.oob_mode.skip if skip_dma.skip_token else nisa.oob_mode.error,
            dge_mode=nisa.dge_mode.swdge,
        )


def _skip_expert(E):
    """The OOB-skip expert index of the weight-reuse masks (see weight_skip_expert)."""
    return E if _WEIGHT_SKIP_EXPERT[0] == None else _WEIGHT_SKIP_EXPERT[0]


def _compute_weight_skip_mask(sbm, all_experts, all_experts_for_weights, n_mask_blocks, E):
    """Compare consecutive block experts and set weight expert to E (OOB/skip) where same.

    n_mask_blocks is the number of mask entries to compare (LNC1: all blocks in the region,
    since the one core processes them all consecutively).
    """
    sbm.open_scope(name="weight_skip_mask")
    is_same = _sbm_alloc(sbm, (1, n_mask_blocks - 1), dtype=nl.uint8, name="is_same", align=SBUF_QUADRANT_SIZE)
    nisa.tensor_tensor(
        data1=all_experts[0:1, 1:n_mask_blocks],
        data2=all_experts[0:1, 0 : n_mask_blocks - 1],
        op=nl.equal,
        dst=is_same,
    )
    on_false = _sbm_alloc(sbm, (1, n_mask_blocks - 1), dtype=nl.int32, name="on_false", align=SBUF_QUADRANT_SIZE)
    nisa.memset(dst=on_false, value=E)
    nisa.tensor_copy_predicated(
        dst=all_experts_for_weights[0:1, 1:n_mask_blocks],
        src=on_false,
        predicate=is_same,
    )
    sbm.close_scope()


def _repack_scales_to_packed(gate_up_proj_scale, down_proj_scale):
    """Copy standard-layout MX scales into the packed layout in kernel-private HBM.

    Standard: gate/up [E, 16, 2, n_H512, I], down [E, 16, n_I512, H]; scale row 4q + r of tile k
    belongs at partition 32q + r. Packed: gate/up [E, _pmax, n_packed, 2, I], down
    [E, _pmax, n_packed, H] with n_packed = ceil(n_tiles / 4); tile k = 4b + s sits in packed
    buffer b at partition 32q + 4s + r. One static HBM -> HBM DMA per (q, s) moves all experts.
    Partitions the layout does not use are never read by the matmul.
    """
    n_q = _pmax // SBUF_QUADRANT_SIZE
    rows_per_q = _pmax // _q_height // n_q

    E, _, _, n_H512, I = gate_up_proj_scale.shape

    # A DRAM access pattern holds at most 5 dims including the implicit partition dim, so a
    # single-expert copy drops its size-1 expert dim to stay in range.
    def _e_dim(stride):
        return [[stride, E]] if E > 1 else []

    n_pk_gup = div_ceil(n_H512, SLOTS_PER_PACKED_BUFFER)
    gup_pk = nl.ndarray((E, _pmax, n_pk_gup, 2, I), dtype=gate_up_proj_scale.dtype, buffer=nl.private_hbm)
    for q in range(n_q):
        for s in range(min(SLOTS_PER_PACKED_BUFFER, n_H512)):
            n_b = div_ceil(n_H512 - s, SLOTS_PER_PACKED_BUFFER)
            nisa.dma_copy(
                dst=gup_pk.ap(
                    pattern=_e_dim(_pmax * n_pk_gup * 2 * I)
                    + [
                        [n_pk_gup * 2 * I, rows_per_q],
                        [2 * I, n_b],
                        [I, 2],
                        [1, I],
                    ],
                    offset=(q * SBUF_QUADRANT_SIZE + s * _q_width) * n_pk_gup * 2 * I,
                ),
                src=gate_up_proj_scale.ap(
                    pattern=_e_dim(16 * 2 * n_H512 * I)
                    + [
                        [2 * n_H512 * I, rows_per_q],
                        [SLOTS_PER_PACKED_BUFFER * I, n_b],
                        [n_H512 * I, 2],
                        [1, I],
                    ],
                    offset=q * rows_per_q * 2 * n_H512 * I + s * I,
                ),
            )

    _, _, n_I512, H = down_proj_scale.shape
    n_pk_down = div_ceil(n_I512, SLOTS_PER_PACKED_BUFFER)
    down_pk = nl.ndarray((E, _pmax, n_pk_down, H), dtype=down_proj_scale.dtype, buffer=nl.private_hbm)
    for q in range(n_q):
        for s in range(min(SLOTS_PER_PACKED_BUFFER, n_I512)):
            n_b = div_ceil(n_I512 - s, SLOTS_PER_PACKED_BUFFER)
            nisa.dma_copy(
                dst=down_pk.ap(
                    pattern=_e_dim(_pmax * n_pk_down * H) + [[n_pk_down * H, rows_per_q], [H, n_b], [1, H]],
                    offset=(q * SBUF_QUADRANT_SIZE + s * _q_width) * n_pk_down * H,
                ),
                src=down_proj_scale.ap(
                    pattern=_e_dim(16 * n_I512 * H)
                    + [
                        [n_I512 * H, rows_per_q],
                        [SLOTS_PER_PACKED_BUFFER * H, n_b],
                        [1, H],
                    ],
                    offset=q * rows_per_q * n_I512 * H + s * H,
                ),
            )
    return gup_pk, down_pk


def _rows_scale_loadable(scale_hbm):
    """True when a standard-layout MX scale tensor can use _load_expert_scale_rows.

    That needs 16 scale rows per expert (4 per SBUF quadrant), i.e. 128 contraction partitions.
    """
    return scale_hbm.shape[1] == _pmax // _q_height


def _load_expert_scale_rows(dst_sb, scale_hbm, expert, skip_weight):
    """Load one expert's standard-layout MX scales with hardware-DGE DMAs.

    scale_hbm is uint8 [E, 16, ...]: 16 scale rows per expert, rows 4q..4q+3 belonging at
    partitions 32q..32q+3 of dst_sb (quadrant q). Each of the four DMAs copies all 16 rows of the
    expert to 16 consecutive partitions starting at 28q, which puts rows 4q..4q+3 exactly at
    32q..32q+3. The other rows land in the unused partitions of the quadrant layout, which the
    matmul never reads. A scalar-offset DMA must cover 1 or a multiple of 16 partitions, which a
    direct 4-partition copy per quadrant would violate.

    Compared with the vector-offset gather this replaces, it needs no per-expert index vector,
    runs on hardware DGE instead of the GpSimd queue, and reads 4x16 rows instead of 128.
    With skip_weight the expert may be E (same expert as the previous block): the DMAs are then
    OOB-skipped and the previous expert's scales stay resident.

    Args:
        dst_sb: uint8 SBUF scale buffer [_pmax, ...] in the matmul quadrant layout.
        scale_hbm: uint8 HBM scales [E, 16, ...].
        expert: [1, 1] int32 expert index.
        skip_weight: OOB-skip instead of erroring when expert == E.
    """
    n_experts, n_rows = scale_hbm.shape[0], scale_hbm.shape[1]
    kernel_assert(n_rows == _pmax // _q_height, f"expected {_pmax // _q_height} scale rows per expert, got {n_rows}")
    row_elems = 1
    for d in scale_hbm.shape[2:]:
        row_elems = row_elems * d
    src = scale_hbm.reshape((n_experts, n_rows * row_elems))
    dst_flat = dst_sb.reshape((_pmax, row_elems))
    rows_per_quadrant = n_rows // (_pmax // SBUF_QUADRANT_SIZE)
    for q in range(_pmax // SBUF_QUADRANT_SIZE):
        p0 = q * (SBUF_QUADRANT_SIZE - rows_per_quadrant)
        nisa.dma_copy(
            dst=dst_flat[p0 : p0 + n_rows, 0:row_elems],
            src=src.ap(pattern=[[row_elems, n_rows], [1, row_elems]], offset=0, scalar_offset=expert, indirect_dim=0),
            oob_mode=oob_mode.skip if skip_weight else oob_mode.error,
            dge_mode=dge_mode.hwdge,
        )


def _prefetch_gup_tile0(
    inps,
    expert,
    prj_cfg,
    buffers,
    dims,
    skip_dma,
    sbm,
    name_prefix="pf",
    scale_expert=None,
    use_packed_scales=False,
    skip_scales=False,
    full_I_load=False,
):
    """Prefetch gate/up weights and scales into persistent buffers.

    scale_expert: optional OOB-aware expert used only for the scale expert-index vector
        (enables scale-skip across same-expert blocks). Defaults to `expert`.
    use_packed_scales: caller-supplied flag selecting packed vs. standard HBM scale layout.
    skip_scales: STATIC_MX path skips scale DMAs entirely — inps.gup_scales_sb holds dummy 127.
    full_I_load: when True, prefetch the FULL I dim of gup weights (used by the
        full-resident gup scheme). When False, prefetch one I-tile worth.
    """
    scale_shape = inps.gate_up_proj_scale.shape if not skip_scales else None
    use_row_scales = not skip_scales and not use_packed_scales and _rows_scale_loadable(inps.gate_up_proj_scale)
    if use_row_scales:
        # Scales first: they are small and gate/up needs them together with tile 0.
        _load_expert_scale_rows(
            inps.gup_scales_sb,
            inps.gate_up_proj_scale,
            scale_expert if scale_expert != None else expert,
            skip_dma.skip_weight,
        )
    # Packed scales address via scalar_offset=block_expert, so the per-expert index vector is never consumed.
    if not skip_scales and not use_packed_scales and not use_row_scales:
        token_indices = _generate_expert_index_vector(
            expert_index=scale_expert if scale_expert != None else expert,
            dst_idx_vector=inps.p_gup_idx_vector,
            scale_factor=scale_shape[1],
            n_quadrants_needed=prj_cfg.H0 // SBUF_QUADRANT_SIZE,
            n_remaining_partition=0,
            name_prefix=f"{name_prefix}_gup_eiv",
            sbm=sbm,
            dst_int32=inps.p_gup_idx_vector_int32,
        )
    # Weight extent: full I for the full-resident scheme; one I-tile otherwise.
    load_I = dims.I if full_I_load else min(_I_TILE_SZ, dims.I)
    if _WSTREAM[0]:
        # tile-major gate/up: I-tile 0 is one [128][2][n_H512][512] block.
        kernel_assert(not full_I_load, "weight_stream tile_major: no full-I gate/up prefetch")
        gup_weight_view = _gup_tile_src(inps, expert, 0)
    else:
        gup_weight_view = (
            inps.gate_up_proj_weight.select(dim=0, index=expert)
            .slice(dim=2, start=0, end=prj_cfg.n_H512_tile_sharded)
            .slice(dim=3, start=0, end=load_I)
        )
    nisa.dma_copy(
        dst=buffers.gup_tile_buf_a[:_pmax, :2, : prj_cfg.n_H512_tile_sharded, :load_I],
        src=gup_weight_view,
        oob_mode=oob_mode.skip if skip_dma.skip_weight else oob_mode.error,
        dge_mode=dge_mode.hwdge,
        **_wkw(0),
    )
    if skip_scales or use_row_scales:
        # STATIC_MX: scale operand is the constant all-127 dummy, nothing to load per block.
        # Row-loaded scales were issued above.
        return
    if use_packed_scales:
        # Packed prefetch: mirror production's gate/up split for parity.
        # When skip_weight: one combined DMA (single OOB-skip evaluation).
        # Otherwise: two DMAs (gate + up) for better DMA scheduling.
        n_packed_gup = scale_shape[2]
        _scale_expert = scale_expert if scale_expert != None else expert
        if skip_dma.skip_weight:
            nisa.dma_copy(
                dst=inps.gup_scales_sb[:_pmax, :n_packed_gup, :2, : prj_cfg.I],
                src=inps.gate_up_proj_scale.ap(
                    pattern=[
                        [n_packed_gup * 2 * prj_cfg.I, _pmax],
                        [2 * prj_cfg.I, n_packed_gup],
                        [prj_cfg.I, 2],
                        [1, prj_cfg.I],
                    ],
                    offset=0,
                    scalar_offset=_scale_expert,
                    indirect_dim=0,
                ),
                oob_mode=oob_mode.skip,
                dge_mode=dge_mode.hwdge,
                **_wkw(0),
            )
        else:
            # Gate scales
            nisa.dma_copy(
                dst=inps.gup_scales_sb[:_pmax, :n_packed_gup, 0:1, : prj_cfg.I],
                src=inps.gate_up_proj_scale.ap(
                    pattern=[
                        [n_packed_gup * 2 * prj_cfg.I, _pmax],
                        [2 * prj_cfg.I, n_packed_gup],
                        [1, prj_cfg.I],
                    ],
                    offset=0,
                    scalar_offset=_scale_expert,
                    indirect_dim=0,
                ),
                oob_mode=oob_mode.error,
                dge_mode=dge_mode.hwdge,
            )
            # Up scales
            nisa.dma_copy(
                dst=inps.gup_scales_sb[:_pmax, :n_packed_gup, 1:2, : prj_cfg.I],
                src=inps.gate_up_proj_scale.ap(
                    pattern=[
                        [n_packed_gup * 2 * prj_cfg.I, _pmax],
                        [2 * prj_cfg.I, n_packed_gup],
                        [1, prj_cfg.I],
                    ],
                    offset=prj_cfg.I,
                    scalar_offset=_scale_expert,
                    indirect_dim=0,
                ),
                oob_mode=oob_mode.error,
                dge_mode=dge_mode.hwdge,
            )
        return
    # Full scales — one trigger for weight_skipping (simpler OOB-skip), two triggers otherwise (better DMA scheduling)
    gup_scale_view = inps.gate_up_proj_scale.reshape(
        (scale_shape[0] * scale_shape[1], scale_shape[2], scale_shape[3], scale_shape[4])
    )
    full_n_H512 = scale_shape[3]
    stride_dim0 = 2 * full_n_H512 * prj_cfg.I
    if skip_dma.skip_weight:
        # Single trigger: load gate+up scales together (matches load_gup_weights_scales_mx pattern)
        nisa.dma_copy(
            src=gup_scale_view.ap(
                pattern=[
                    [stride_dim0, _pmax],
                    [full_n_H512 * prj_cfg.I, 2],
                    [prj_cfg.I, prj_cfg.n_H512_tile_sharded],
                    [1, prj_cfg.I],
                ],
                offset=0,
                vector_offset=token_indices.ap([[1, _pmax], [1, 1]], offset=0),
                indirect_dim=0,
            ),
            dst=inps.gup_scales_sb[:_pmax, :2, : prj_cfg.n_H512_tile_sharded, : prj_cfg.I],
            oob_mode=oob_mode.skip,
        )
    else:
        # Two triggers: split gate and up for better DMA scheduling
        # Gate scales
        nisa.dma_copy(
            src=gup_scale_view.ap(
                pattern=[
                    [stride_dim0, _pmax],
                    [prj_cfg.I, prj_cfg.n_H512_tile_sharded],
                    [1, prj_cfg.I],
                ],
                offset=0,
                vector_offset=token_indices.ap([[1, _pmax], [1, 1]], offset=0),
                indirect_dim=0,
            ),
            dst=inps.gup_scales_sb[:_pmax, 0:1, : prj_cfg.n_H512_tile_sharded, : prj_cfg.I],
            oob_mode=oob_mode.skip,
        )
        # Up scales
        nisa.dma_copy(
            src=gup_scale_view.ap(
                pattern=[
                    [stride_dim0, _pmax],
                    [prj_cfg.I, prj_cfg.n_H512_tile_sharded],
                    [1, prj_cfg.I],
                ],
                offset=full_n_H512 * prj_cfg.I,
                vector_offset=token_indices.ap([[1, _pmax], [1, 1]], offset=0),
                indirect_dim=0,
            ),
            dst=inps.gup_scales_sb[:_pmax, 1:2, : prj_cfg.n_H512_tile_sharded, : prj_cfg.I],
            oob_mode=oob_mode.skip,
        )


def check_kernel_compatibility(dims: BWMMMXDimensionSizes, configs: BWMMMXConfigs):
    """
    Validate kernel configuration and dimension compatibility.

    Performs comprehensive validation of kernel parameters to ensure they meet
    hardware constraints and implementation requirements before execution.

    Args:
        dims (BWMMMXDimensionSizes): Dimension configuration containing B, H, I, N,
            num_shards (must be 1 for this LNC1 kernel), and cond_vec_len.
        configs (BWMMMXConfigs): Kernel configuration containing is_tensor_update_accumulating
            and use_dynamic_while flags.

    Returns:
        None: Raises assertion errors if validation fails.

    Notes:
        - Block size (B) must be multiple of 128 for efficient tiling
        - Hidden dimension (H) must be in range [512, 8192] and divisible by PSUM_SIZE (512)
        - Intermediate dimension (I) must be divisible by 16 for quantization alignment
        - LNC1 only: exactly one logical core
        - Dynamic loop requires condition vector of length N+2
        - Only supports topK > 1 (tensor update accumulating mode)

    Pseudocode:
        assert B % 128 == 0
        assert 512 <= H <= 8192
        assert H % PSUM_SIZE == 0
        assert I % 16 == 0
        assert is_tensor_update_accumulating == True
        assert num_shards == 1   # LNC1 only
        if use_dynamic_while:
            assert cond_vec_len == N + 2
    """
    kernel_assert(dims.B % _pmax == 0, f"Blocksize must be a multiple of 128")
    kernel_assert(512 <= dims.H <= 8192, f"Hidden dims must be between 512 and 8192, found {dims.H}")
    kernel_assert(dims.H % PSUM_SIZE == 0, f"Hidden dim size must be multiples of {PSUM_SIZE}, found {dims.H} ")

    kernel_assert(dims.I % 16 == 0, f"down_proj_weight I must be divisible by 16, found {dims.I} . Please pad it")
    kernel_assert(configs.is_tensor_update_accumulating, "Only support topK > 1 at the moment.")

    # LNC1-only fork: all sharding logic has been specialised away, so a multi-core launch
    # is unsupported. dims.num_shards mirrors get_program_sharding_info()'s n_prgs, which the
    # entry already asserts is 1; re-check here so direct helper users also fail loudly.
    kernel_assert(
        dims.num_shards == 1,
        f"block_moe_mx is LNC1-only and requires exactly 1 logical core, got {dims.num_shards}",
    )

    if configs.use_dynamic_while:
        kernel_assert(
            dims.cond_vec_len == dims.N + 2,
            f"condition vector must have exactly N+2 elements, got {dims.cond_vec_len} != N + 2 ({dims.N} + 2)",
        )


def load_gup_weights_scales_mx(
    inps: InputTensors,
    block_expert: nl.NkiTensor,
    dims: BWMMMXDimensionSizes,
    prj_cfg: ProjConfig,
    skip_dma: SkipMode,
    sbm=None,
    dst_weight=None,
    dst_bias=None,
    name_prefix="gup",
    use_packed_scales: bool = False,
    skip_scales: bool = False,  # STATIC_MX: skip per-block scale DMA (inps.gup_scales_sb holds dummy 127)
):
    """
    Load gate and up projection weights, scales, and biases for current expert.

    Loads MXFP4/MXFP8 quantized weights, uint8 scales, and biases for both gate and up
    projections from HBM to SBUF for the expert assigned to the current block.

    Args:
        inps (InputTensors): Input tensors containing gate_up_proj_weight of shape
            [E, 128, 2, n_H512_tile, I], gate_up_proj_scale, gate_and_up_proj_bias,
            and buffers for scales and index vectors.
        block_expert (nl.NkiTensor): Expert index for current block, shape [1, 1].
        dims (BWMMMXDimensionSizes): Dimension configuration with I, H.
        prj_cfg (ProjConfig): Projection configuration with n_H512_tile_sharded, I.
        skip_dma (SkipMode): DMA skip configuration for weight loading.

    Returns:
        tuple: (gup_weights_qtz_sb, gup_scales_sb, gup_bias_sb)
            - gup_weights_qtz_sb (nl.NkiTensor): Quantized weights [128, 2, n_H512_tile_sharded, I]
            - gup_scales_sb (nl.NkiTensor): Dequantization scales [128, 2, n_H512_tile_sharded, I]
            - gup_bias_sb (nl.NkiTensor): Bias values [128, 2, n_total_I512_tile, 128]

    Notes:
        - Uses indirect DGE with block_expert for expert selection
        - Generates index vectors on-the-fly for scale loading
        - Pads bias to 512 when I < 512 for alignment
        - Scales are loaded with zero-padding for out-of-bounds partitions
        - Gate and up projections share weight buffer (dimension 1 has size 2)

    Pseudocode:
        gup_weights_qtz_sb = allocate [128, 2, n_H512_tile_sharded, I] in SBUF
        dma_copy gate_up_proj_weight[block_expert, :, :, :, :] to gup_weights_qtz_sb

        gup_scale_view = reshape gate_up_proj_scale to [E*16, 2, n_H512_tile, I]
        token_indices_on_p = generate_expert_index_vector(block_expert)
        dma_copy gup_scale_view[token_indices_on_p, :, :, :] to gup_scales_sb

        gup_bias_sb = allocate [128, 2, n_total_I512_tile, 128] in SBUF
        if I < 512:
            memset gup_bias_sb to 0
            dma_copy gate_and_up_proj_bias[block_expert, :I//4, :, :, :] to gup_bias_sb[:I//4, :, :, :]
        else:
            dma_copy gate_and_up_proj_bias[block_expert, :, :, :, :] to gup_bias_sb

        return gup_weights_qtz_sb, gup_scales_sb, gup_bias_sb
    """
    if dst_weight != None:
        gup_weights_qtz_sb = dst_weight
    else:
        gup_weights_qtz_sb = _sbm_alloc(
            sbm,
            (_pmax, 2, prj_cfg.n_H512_tile_sharded, dims.I),
            dtype=inps.gate_up_proj_weight.dtype,
            name="gup_weights_qtz_sb",
            align=SBUF_QUADRANT_SIZE,
        )
    """
    Load gate/up weight for current expert.

    gate_up_proj_weight shape: (E, 128, 2, n_H512_tile, I)
    select expert -> (128, 2, n_H512_tile, I)
    slice H512 tiles -> (128, 2, n_H512_tile_sharded, I)
    """
    gup_weight_view = inps.gate_up_proj_weight.select(dim=0, index=block_expert).slice(
        dim=2, start=0, end=prj_cfg.n_H512_tile_sharded
    )
    nisa.dma_copy(
        dst=gup_weights_qtz_sb,
        src=gup_weight_view,
        oob_mode=oob_mode.skip if skip_dma.skip_weight else oob_mode.error,
        dge_mode=dge_mode.hwdge,
    )

    """
    GATE UP SCALES
    """
    # gup_n_quadrants_needed is returned to callers; depends only on H0 / quadrant
    # size, so it's the same regardless of scale layout.
    gup_n_quadrants_needed = prj_cfg.H0 // SBUF_QUADRANT_SIZE
    token_indices_on_p = None
    scale_shape = None

    # STATIC_MX skips entirely: inps.gup_scales_sb holds persistent dummy 127 from top-level memset.
    if not skip_scales:
        scale_shape = inps.gate_up_proj_scale.shape

        if use_packed_scales:
            # Packed scale path: HBM is [E, _pmax, n_packed_gup, 2, I]; SBUF is the
            # same shape. Single dma_copy with scalar_offset=block_expert.
            n_packed_gup = scale_shape[2]
            kernel_assert(
                scale_shape == (dims.E, _pmax, n_packed_gup, 2, prj_cfg.I),
                f"Packed gate_up_proj_scale shape mismatch: got {scale_shape}, "
                f"expected ({dims.E}, {_pmax}, n_packed_gup, 2, {prj_cfg.I})",
            )
            nisa.dma_copy(
                dst=inps.gup_scales_sb[:_pmax, :n_packed_gup, :2, : prj_cfg.I],
                src=inps.gate_up_proj_scale.ap(
                    pattern=[
                        [n_packed_gup * 2 * prj_cfg.I, _pmax],
                        [2 * prj_cfg.I, n_packed_gup],
                        [prj_cfg.I, 2],
                        [1, prj_cfg.I],
                    ],
                    offset=0,
                    scalar_offset=block_expert,
                    indirect_dim=0,
                ),
                oob_mode=oob_mode.skip if skip_dma.skip_weight else oob_mode.error,
                dge_mode=dge_mode.hwdge,
            )
        elif _rows_scale_loadable(inps.gate_up_proj_scale):
            _load_expert_scale_rows(inps.gup_scales_sb, inps.gate_up_proj_scale, block_expert, skip_dma.skip_weight)
        else:
            # fold E * 16 together
            gup_scale_view = inps.gate_up_proj_scale.reshape(
                (scale_shape[0] * scale_shape[1], scale_shape[2], scale_shape[3], scale_shape[4])
            )

            """
            Construct a vector DGE index to index into E*16
                if block_expert == 0, we want something like this (transposed to the P dimension)
                [0 1 2 3 -1 -1 -1 ..... 4 5 6 7 -1 -1 -1 .... 8 9 10 11 -1 -1 -1 .... 12 13 14 15 -1 -1 -1... -1]

                if block_expert == 3, we want something like this
                [48 49 50 51 -1 -1 -1 ..... 52 53 54 55 -1 -1 -1 .... 56 57 58 59 -1 -1 -1 .... 60 61 62 63 -1 -1 -1... -1]
                i.e, basically the same as above, with offset 16*3 = 48
            """
            token_indices_on_p = _generate_expert_index_vector(
                expert_index=block_expert,
                dst_idx_vector=inps.p_gup_idx_vector,
                scale_factor=scale_shape[1],
                n_quadrants_needed=gup_n_quadrants_needed,
                n_remaining_partition=0,
                name_prefix=f"{name_prefix}_expert_index_vector",
                sbm=sbm,
                dst_int32=inps.p_gup_idx_vector_int32,
            )
            # gup_scale_view shape: (E*16, 2, n_H512_tile, I) - use FULL source tensor dimensions for strides.
            # The source tensor has full n_H512_tile, we only load n_H512_tile_sharded elements.
            full_n_H512_tile_scale = scale_shape[3]
            stride_dim0 = 2 * full_n_H512_tile_scale * prj_cfg.I
            nisa.dma_copy(
                src=gup_scale_view.ap(
                    pattern=[
                        [stride_dim0, _pmax],
                        [full_n_H512_tile_scale * prj_cfg.I, 2],
                        [prj_cfg.I, prj_cfg.n_H512_tile_sharded],
                        [1, prj_cfg.I],
                    ],
                    offset=0,
                    vector_offset=token_indices_on_p.ap(
                        [[1, _pmax], [1, 1]],
                        offset=0,
                    ),
                    indirect_dim=0,
                ),
                dst=inps.gup_scales_sb[:_pmax, :2, : prj_cfg.n_H512_tile_sharded, : prj_cfg.I],
                oob_mode=oob_mode.skip,
            )

    """
    GATE UP BIAS
    """
    gup_bias_sb = None
    if inps.gate_and_up_proj_bias:
        if dst_bias != None:
            gup_bias_sb = dst_bias
        else:
            gup_bias_sb = _sbm_alloc(
                sbm,
                (_pmax, 2, prj_cfg.n_total_I512_tile, _q_width),
                dtype=inps.gate_and_up_proj_bias.dtype,
                name="gup_bias_sb",
                align=SBUF_QUADRANT_SIZE,
            )

        if dims.I < _pmax * _q_width:  # when I<512, gate/up bias HBM is not padded so pad it here
            if not (skip_dma.skip_weight and dst_bias != None):
                nisa.memset(dst=gup_bias_sb[:, :, 0, :], value=0.0)
            # gate_and_up_proj_bias shape: (E, I_par_dim, 2, n_total_I512_tile, _q_width) where I_par_dim = I//4
            I_par_dim = dims.I // 4
            bias_stride_dim0 = 2 * prj_cfg.n_total_I512_tile * _q_width  # stride for I_par_dim
            bias_stride_dim1 = prj_cfg.n_total_I512_tile * _q_width  # stride for gate/up (2)
            bias_stride_dim2 = _q_width  # stride for n_total_I512_tile
            nisa.dma_copy(
                dst=gup_bias_sb[:I_par_dim, :, :, :],
                src=inps.gate_and_up_proj_bias.ap(
                    pattern=[
                        [bias_stride_dim0, I_par_dim],
                        [bias_stride_dim1, 2],
                        [bias_stride_dim2, prj_cfg.n_total_I512_tile],
                        [1, _q_width],
                    ],
                    offset=0,
                    scalar_offset=block_expert,
                    indirect_dim=0,
                ),
                oob_mode=oob_mode.skip if skip_dma.skip_weight else oob_mode.error,
                dge_mode=dge_mode.hwdge,
            )
        else:
            # gate_and_up_proj_bias shape: (E, _pmax, 2, n_total_I512_tile, _q_width)
            # Strides: dim1=2*n_total_I512_tile*_q_width, dim2=n_total_I512_tile*_q_width, dim3=_q_width, dim4=1
            bias_stride_dim1 = 2 * prj_cfg.n_total_I512_tile * _q_width
            bias_stride_dim2 = prj_cfg.n_total_I512_tile * _q_width
            nisa.dma_copy(
                dst=gup_bias_sb,
                src=inps.gate_and_up_proj_bias.ap(
                    pattern=[
                        [bias_stride_dim1, _pmax],
                        [bias_stride_dim2, 2],
                        [_q_width, prj_cfg.n_total_I512_tile],
                        [1, _q_width],
                    ],
                    offset=0,
                    scalar_offset=block_expert,
                    indirect_dim=0,
                ),
                oob_mode=oob_mode.skip if skip_dma.skip_weight else oob_mode.error,
                dge_mode=dge_mode.hwdge,
            )

    return gup_weights_qtz_sb, inps.gup_scales_sb, gup_bias_sb, token_indices_on_p, gup_n_quadrants_needed


def _load_gup_weight_tile(
    inps,
    block_expert,
    prj_cfg,
    skip_dma,
    dst_weight,
    dst_scale,
    dst_bias,
    token_indices_on_p,
    I_offset,
    dims,
    sbm=None,
):
    """Load a single I-tile of gate/up weights, scales, and bias from HBM to SBUF.

    Args:
        dst_weight: SBUF buffer (_pmax, 2, n_H512_tile_sharded, _I_TILE_SZ).
        dst_scale: SBUF buffer (_pmax, 2, n_H512_tile_sharded, _I_TILE_SZ).
        dst_bias: SBUF buffer (_pmax, 2, 1, _q_width).
        token_indices_on_p: Pre-computed expert index vector for scale DGE.
        I_offset: Starting offset in I dimension.
    """
    cur_I_load_sz = min(_I_TILE_SZ, dims.I - I_offset)

    # --- WEIGHTS ---
    if _WSTREAM[0]:
        # tile-major gate/up: the I-tile is one [128][2][n_H512][512] block.
        gup_weight_view = _gup_tile_src(inps, block_expert, I_offset // _I_TILE_SZ)
    else:
        gup_weight_view = (
            inps.gate_up_proj_weight.select(dim=0, index=block_expert)
            .slice(dim=2, start=0, end=prj_cfg.n_H512_tile_sharded)
            .slice(dim=3, start=I_offset, end=I_offset + cur_I_load_sz)
        )
    nisa.dma_copy(
        dst=dst_weight[:_pmax, :2, : prj_cfg.n_H512_tile_sharded, :cur_I_load_sz],
        src=gup_weight_view,
        oob_mode=oob_mode.skip if skip_dma.skip_weight else oob_mode.error,
        dge_mode=dge_mode.hwdge,
        **_wkw(I_offset // _I_TILE_SZ),
    )

    # --- SCALES ---
    if dst_scale != None:
        scale_shape = inps.gate_up_proj_scale.shape
        gup_scale_view = inps.gate_up_proj_scale.reshape(
            (scale_shape[0] * scale_shape[1], scale_shape[2], scale_shape[3], scale_shape[4])
        )
        full_n_H512_tile_scale = scale_shape[3]
        stride_dim0 = 2 * full_n_H512_tile_scale * dims.I
        nisa.dma_copy(
            src=gup_scale_view.ap(
                pattern=[
                    [stride_dim0, _pmax],
                    [full_n_H512_tile_scale * dims.I, 2],
                    [dims.I, prj_cfg.n_H512_tile_sharded],
                    [1, cur_I_load_sz],
                ],
                offset=I_offset,
                vector_offset=token_indices_on_p.ap(
                    [[1, _pmax], [1, 1]],
                    offset=0,
                ),
                indirect_dim=0,
            ),
            dst=dst_scale[:_pmax, :2, : prj_cfg.n_H512_tile_sharded, :cur_I_load_sz],
            oob_mode=oob_mode.skip,
        )

    # --- BIAS ---
    if dst_bias == None:
        return

    i_I512_tile = I_offset // _I_TILE_SZ
    full_n_total_I512_tile = prj_cfg.n_total_I512_tile

    if dims.I < _I_TILE_SZ:
        # I < 512: bias HBM has I_par_dim = I//4 on p-dim due to QMX
        I_par_dim = dims.I // 4
        bias_stride_dim0 = 2 * full_n_total_I512_tile * _q_width
        bias_stride_dim1 = full_n_total_I512_tile * _q_width
        nisa.dma_copy(
            dst=dst_bias[:I_par_dim, :, :, :],
            src=inps.gate_and_up_proj_bias.ap(
                pattern=[
                    [bias_stride_dim0, I_par_dim],
                    [bias_stride_dim1, 2],
                    [_q_width, 1],
                    [1, _q_width],
                ],
                offset=i_I512_tile * _q_width,
                scalar_offset=block_expert,
                indirect_dim=0,
            ),
            oob_mode=oob_mode.skip if skip_dma.skip_weight else oob_mode.error,
            dge_mode=dge_mode.hwdge,
        )
    else:
        # I >= 512: bias HBM has _pmax on p-dim
        bias_stride_dim1 = 2 * full_n_total_I512_tile * _q_width
        bias_stride_dim2 = full_n_total_I512_tile * _q_width
        nisa.dma_copy(
            dst=dst_bias,
            src=inps.gate_and_up_proj_bias.ap(
                pattern=[
                    [bias_stride_dim1, _pmax],
                    [bias_stride_dim2, 2],
                    [_q_width, 1],
                    [1, _q_width],
                ],
                offset=i_I512_tile * _q_width,
                scalar_offset=block_expert,
                indirect_dim=0,
            ),
            oob_mode=oob_mode.skip if skip_dma.skip_weight else oob_mode.error,
            dge_mode=dge_mode.hwdge,
        )


def load_down_proj_weights_mx(
    inps: InputTensors,
    block_expert: nl.NkiTensor,
    dst_weight: nl.NkiTensor,
    dims: BWMMMXDimensionSizes,
    prj_cfg: ProjConfig,
    skip_dma: SkipMode,
    gup_token_indices_on_p: nl.NkiTensor = None,
    gup_n_quadrants_needed: int = None,
    dst_scale: nl.NkiTensor = None,
    sbm=None,
    dst_bias=None,
    use_packed_scales: bool = False,
    skip_scales: bool = False,  # STATIC_MX: skip per-block scale DMA (caller pre-memsets dummy 127)
):
    """
    Load down projection weights, scales, and biases for current expert.

    Loads MXFP4/MXFP8 quantized weights and uint8 scales for down projection from HBM
    to SBUF, constructing partition index vectors for proper expert selection.

    Args:
        inps (InputTensors): Input tensors containing down_proj_weight [E, p_I, n_total_I512_tile, H],
            down_proj_scale, down_proj_bias, and index vector buffer.
        block_expert (nl.NkiTensor): Expert index for current block, shape [1, 1].
        dst_weight (nl.NkiTensor): Destination buffer for weights in SBUF.
        dims (BWMMMXDimensionSizes): Dimension configuration with I, H, p_I.
        prj_cfg (ProjConfig): Projection configuration with sharding info.
        skip_dma (SkipMode): DMA skip configuration.

    Returns:
        tuple: (down_weight_hbm, down_scale_sb, down_bias_sb)
            - down_weight_hbm: Reference to weight tensor in HBM
            - down_scale_sb: Scales in SBUF [128, n_total_I512_tile, H_sharded]
            - down_bias_sb: Bias in SBUF [1, H]

    Notes:
        - Loads only sharded portion of H dimension per program
        - Constructs partition index with quadrant-based addressing
        - Handles remainder partitions when p_I not divisible by 32
        - Zero-pads scales when p_I < 128

    Pseudocode:
        dma_copy down_proj_weight[block_expert, :, :, :] to dst_weight

        down_scale_sb = allocate [128, n_total_I512_tile, H_sharded] in SBUF
        if p_I != 128:
            memset down_scale_sb[:, -1, :] to 0

        down_scale_view = reshape down_proj_scale to [E*16, n_total_I512_tile, H]
        construct p_down_idx_vector: [block_expert*16+0, ..., block_expert*16+15, -1, ...]
        dma_copy down_scale_view[p_down_idx_vector, :, :] to down_scale_sb

        down_bias_sb = allocate [1, H] in SBUF
        dma_copy down_proj_bias[block_expert, :] to down_bias_sb

        return down_scale_sb, down_bias_sb
    """
    """
    Load down projection weights from HBM to SBUF.
    
    down_proj_weight shape: (E, p_I, n_total_I512_tile, H)
    Load directly into dst_weight with scalar AP.
    scalar_offset=block_expert with indirect_dim=0 means access starts at 
    block_expert * (p_I * n_total_I512_tile * H)
    """

    """
    Load down projection weights from HBM to SBUF.

    down_proj_weight shape: (E, p_I, n_total_I512_tile, H)
    select expert -> (p_I, n_total_I512_tile, H)
    slice H for sharding -> (p_I, n_total_I512_tile, H_sharded)
    """
    down_weight_view = inps.down_proj_weight.select(dim=0, index=block_expert).slice(
        dim=2, start=prj_cfg.prg_id * prj_cfg.H_sharded, end=(prj_cfg.prg_id + 1) * prj_cfg.H_sharded
    )
    nisa.dma_copy(
        src=down_weight_view,
        dst=dst_weight[: dims.p_I, :, :],
        oob_mode=oob_mode.skip if skip_dma.skip_weight else oob_mode.error,
        dge_mode=dge_mode.hwdge,
        **_wkw(0),
    )

    """
    DOWN SCALES
    """
    # STATIC_MX skips entirely: caller pre-fills dst_scale (the shared [_pmax, 1] all-127 buffer
    # under STATIC_MX, with [p_I:_pmax] zeroed when p_I < _pmax).
    if skip_scales:
        kernel_assert(dst_scale is not None, "skip_scales=True requires caller-provided dst_scale (dummy 127 buffer)")
        down_scale_sb = dst_scale
        # Skip directly to bias load.
        scale_shape = None
        n_packed_down = None
    else:
        scale_shape = inps.down_proj_scale.shape

        # n_packed_down is needed by the packed DMA below regardless of whether the
        # buffer was allocated locally or passed in via dst_scale, so derive it once up front.
        n_packed_down = scale_shape[2] if use_packed_scales else None

        # Alloc and load weight scale, which needs zero padding in sbuf
        if dst_scale != None:
            down_scale_sb = dst_scale
        elif use_packed_scales:
            down_scale_sb = _sbm_alloc(
                sbm,
                (_pmax, n_packed_down, prj_cfg.H_sharded),
                dtype=nl.uint8,
                name="down_scale_packed_sb_local",
                align=SBUF_QUADRANT_SIZE,
            )
            if dims.p_I != _pmax:
                nisa.memset(down_scale_sb[:, n_packed_down - 1, :], value=0)
        else:
            down_scale_sb = _sbm_alloc(
                sbm,
                (_pmax, prj_cfg.n_total_I512_tile, prj_cfg.H_sharded),
                dtype=nl.uint8,
                name="down_scale_sb_local",
                align=SBUF_QUADRANT_SIZE,
            )
            # Memset weight scale if input weight scale HBM does not pad on par dim
            if dims.p_I != _pmax:
                nisa.memset(down_scale_sb[:, prj_cfg.n_total_I512_tile - 1, :], value=0)

        if not use_packed_scales:
            kernel_assert(
                down_scale_sb.shape == (_pmax, prj_cfg.n_total_I512_tile, prj_cfg.H_sharded),
                f"Got {down_scale_sb.shape}",
            )

    if skip_scales:
        pass  # No scale DMA under STATIC_MX
    elif use_packed_scales:
        # Packed scale path: HBM is [E, _pmax, n_packed_down, H]; SBUF strips the
        # leading E dim (one expert at a time, via scalar_offset=block_expert).
        kernel_assert(
            scale_shape == (dims.E, _pmax, n_packed_down, dims.H),
            f"Packed down_proj_scale shape mismatch: got {scale_shape}, "
            f"expected ({dims.E}, {_pmax}, n_packed_down, {dims.H})",
        )
        nisa.dma_copy(
            dst=down_scale_sb[:_pmax, :n_packed_down, : prj_cfg.H_sharded],
            src=inps.down_proj_scale.ap(
                pattern=[
                    [n_packed_down * dims.H, _pmax],
                    [dims.H, n_packed_down],
                    [1, prj_cfg.H_sharded],
                ],
                offset=prj_cfg.prg_id * prj_cfg.H_sharded,
                scalar_offset=block_expert,
                indirect_dim=0,
            ),
            oob_mode=oob_mode.skip if skip_dma.skip_weight else oob_mode.error,
            dge_mode=dge_mode.hwdge,
            **_wkw(0),
        )
    elif _rows_scale_loadable(inps.down_proj_scale) and prj_cfg.H_sharded == dims.H:
        _load_expert_scale_rows(down_scale_sb, inps.down_proj_scale, block_expert, skip_dma.skip_weight)
    else:
        """
        Construct a vector DGE index to index into E*16
            if block_expert == 0, we want something like this (transposed to the P dimension)
            [0 1 2 3 -1 -1 -1 ..... 4 5 6 7 -1 -1 -1 .... 8 9 10 11 -1 -1 -1 .... 12 13 14 15 -1 -1 -1... -1]

            if block_expert == 3, we want something like this
            [48 49 50 51 -1 -1 -1 ..... 52 53 54 55 -1 -1 -1 .... 56 57 58 59 -1 -1 -1 .... 60 61 62 63 -1 -1 -1... -1]
            i.e, basically the same as above, with offset 16*3 = 48
        """
        down_scale_view = inps.down_proj_scale.reshape(
            (scale_shape[0] * scale_shape[1], scale_shape[2], scale_shape[3])
        )

        down_n_quadrants_needed, n_remaining_partition = divmod(dims.p_I, SBUF_QUADRANT_SIZE)
        n_remaining_partition = n_remaining_partition // _q_height

        if (
            gup_token_indices_on_p != None
            and gup_n_quadrants_needed != None
            and gup_n_quadrants_needed == down_n_quadrants_needed
        ):
            token_indices_on_p = gup_token_indices_on_p
        else:
            token_indices_on_p = _generate_expert_index_vector(
                expert_index=block_expert,
                dst_idx_vector=inps.p_down_idx_vector,
                scale_factor=scale_shape[1],
                n_quadrants_needed=down_n_quadrants_needed,
                n_remaining_partition=n_remaining_partition,
                name_prefix="down_expert_index_vector",
                sbm=sbm,
            )

        # down_scale_view shape: (E*16, n_total_I512_tile, H)
        # accumulated shape to right of dim 0: n_total_I512_tile * H
        down_scale_stride_dim0 = prj_cfg.n_total_I512_tile * dims.H

        # Copy only p_I valid partitions (padding partitions already zeroed by memset outside loop)
        nisa.dma_copy(
            src=down_scale_view.ap(
                pattern=[
                    [down_scale_stride_dim0, dims.p_I],
                    [dims.H, prj_cfg.n_total_I512_tile],
                    [1, prj_cfg.H_sharded],
                ],
                offset=prj_cfg.prg_id * prj_cfg.H_sharded,
                vector_offset=token_indices_on_p.ap(
                    [[1, dims.p_I], [1, 1]],
                    offset=0,
                ),
                indirect_dim=0,
            ),
            dst=down_scale_sb[: dims.p_I, : prj_cfg.n_total_I512_tile, : prj_cfg.H_sharded],
            oob_mode=oob_mode.skip,
        )

    # load bias
    # down_proj_bias shape: (E, H)
    down_bias_sb = None
    if inps.down_proj_bias:
        if dst_bias != None:
            down_bias_sb = dst_bias
        else:
            down_bias_sb = _sbm_alloc(
                sbm,
                (1, dims.H),
                dtype=inps.down_proj_bias.dtype,
                name="down_bias_sb",
                align=SBUF_QUADRANT_SIZE,
            )
        nisa.dma_copy(
            src=inps.down_proj_bias.ap(
                pattern=[[dims.H, 1], [1, dims.H]], offset=0, scalar_offset=block_expert, indirect_dim=0
            ),
            dst=down_bias_sb,
            oob_mode=oob_mode.skip if skip_dma.skip_weight else oob_mode.error,
            dge_mode=dge_mode.hwdge,
        )

    return down_scale_sb, down_bias_sb


def _down_projection_affinity_scaled(
    inter_sb, weight, weight_scale, expert_affinity, cfg, sbm, psum_bank_offset, out_sb, is_packed_scale
):
    """MX down projection whose PSUM evacuation also applies the expert affinity.

    Same math as down_projection_mx without bias (MX quantization only): out[b, :] = W^T inter[:, b]
    then scaled by affinity[b]. Scaling while copying PSUM -> SBUF replaces a separate full-width
    multiply pass over the [B, H] output.

    Args:
        expert_affinity: list of n_B128_tiles fp32 [_pmax, 1] per-token affinities.
        out_sb: optional pre-allocated bf16 [_pmax, n_B128_tiles, H] output.
    Returns:
        bf16 [_pmax, n_B128_tiles, H] SBUF tensor.
    """
    BxS, H = cfg.BxS, cfg.H
    n_BxS_tile = div_ceil(BxS, _pmax)
    inter_qtz, inter_qtz_scale, weight_qtz, weight_qtz_scale = _down_proj_prep_inter_and_weights(
        inter_sb, weight, weight_scale, cfg, sbm=sbm, name_prefix="dp"
    )
    if out_sb == None:
        out_sb = _sbm_alloc(sbm, (_pmax, n_BxS_tile, H), dtype=nl.bfloat16, name="dp_out_sb", align=SBUF_QUADRANT_SIZE)
    for i_H_tile in nl.affine_range(cfg.n_H_tile_sharded):
        h_slice = nl.ds(i_H_tile * cfg.H_tile_size, cfg.H_tile_size)
        for i_b in nl.affine_range(n_BxS_tile):
            cur_b = min(_pmax, BxS - i_b * _pmax)
            b_slice = nl.ds(i_b * _pmax, cur_b)
            bank = (psum_bank_offset + i_H_tile * n_BxS_tile + i_b) % NUM_HW_PSUM_BANKS
            psum = nl.ndarray(
                (cur_b, cfg.H_tile_size), dtype=nl.bfloat16, buffer=nl.psum, address=(0, bank * PSUM_BANK_SIZE)
            )
            for i_I512 in nl.affine_range(cfg.n_total_I512_tile):
                if is_packed_scale:
                    slot = (i_I512 % SLOTS_PER_PACKED_BUFFER) * _q_width
                    moving_scale = weight_qtz_scale[slot:, i_I512 // SLOTS_PER_PACKED_BUFFER, h_slice]
                else:
                    moving_scale = weight_qtz_scale[:, i_I512, h_slice]
                nisa.nc_matmul_mx(
                    dst=psum,
                    stationary=inter_qtz[:, i_I512, b_slice],
                    moving=weight_qtz[:, i_I512, h_slice],
                    stationary_scale=inter_qtz_scale[:, i_I512, b_slice],
                    moving_scale=moving_scale,
                )
            nisa.tensor_scalar(
                dst=out_sb[0:cur_b, i_b, h_slice],
                data=psum,
                op0=nl.multiply,
                operand0=expert_affinity[i_b][0:cur_b, 0:1],
                engine=nisa.scalar_engine if i_b % 2 == 0 else nisa.vector_engine,
            )
    return out_sb


def _accumulate_output_rows(src_data, token_indices_2D, n, outs, dims):
    """Add one B128 tile of block results into the output rows of its tokens.

    A single scatter read-modify-write DMA (``output[idx] += src``) replaces the gather of the
    previous output, the SBUF add, and the scatter write. Token ids are unique within a block,
    which the RMW requires; padding ids are out of bounds and skipped. The output is zeroed
    before the first block, so the first contribution to a row needs no special case.

    LNC1: the output is a plain [T, H] buffer, so the scatter carries no shard offset.

    Args:
        src_data: SBUF tensor (_pmax, n_B128_tiles, H) with this block's scaled results.
        token_indices_2D: Token index mapping (_pmax, n_B128_tiles).
        n: B128 tile index.
        outs: OutputTensors with output HBM tensor.
        dims: Dimension sizes.
    """
    T = outs.output.shape[-2]
    output_ap = outs.output.reshape((T, 1, dims.H)).ap(
        pattern=[[dims.H, _pmax], [1, 1], [1, dims.H]],
        offset=0,
        vector_offset=token_indices_2D.ap([[dims.n_B128_tiles, _pmax], [1, 1]], offset=n),
        indirect_dim=0,
    )
    nisa.dma_compute(
        dst=output_ap,
        srcs=[output_ap, src_data[0:_pmax, n : n + 1, 0 : dims.H]],
        scales=[1.0, 1.0],
        reduce_op=nl.add,
        oob_mode=oob_mode.skip,
    )


def _flush_pending_output(routing, prev_block_idx, pending_out, outs, dims, sbm):
    """Accumulate the deferred output of block prev_block_idx (int or [1, 1] int32 SBUF)."""
    prev_token_ids = _select_block_token_ids(routing, prev_block_idx, dims, sbm, name="prev_token_ids")
    for n in range(dims.n_B128_tiles):
        _accumulate_output_rows(pending_out, prev_token_ids, n, outs, dims)


class ResidentRouting(nl.NKIObject):
    """Block routing tables kept in SBUF for the whole kernel.

    Per-block lookups become on-chip copies instead of tiny DMAs. A tiny DMA sits in a DMA queue
    behind multi-MB weight and hidden-state transfers, so it stalls the block that needs it.

    Attributes:
        token_ids (nl.NkiTensor): int32 [_pmax, N * n_B128_tiles]. Column
            ``block * n_B128_tiles + t`` holds the token ids of B128 tile ``t`` of ``block``,
            one token per partition.
        block_expert (nl.NkiTensor): int32 [1, N], the expert of every block.
        n_global_experts (int): experts covered by a packed affinity row (E_local * ep_degree
            with ep_rank, else E_local).
    """

    token_ids: nl.NkiTensor
    block_expert: nl.NkiTensor
    n_global_experts: int
    # (token_rows) [1, N] int32: the row block of token_ids that holds each block; None: token_ids is in block order.
    block_row: nl.NkiTensor
    n_row_blocks: int
    # (compact_affinities) [_pmax, N] fp32 global expert id of every block on every partition; None otherwise.
    gtab: nl.NkiTensor


def _load_resident_routing(inps, dims, sbm, n_global_experts, plain_dma=False, rows=None, row_blocks=None):
    """Load token_position_to_id (or the nonzero rows) and block_to_expert into SBUF once."""
    if rows != None:
        # [E, T] rows -> [E * T / B + 1] row blocks; the last one is padding.
        flat = rows.reshape((rows.shape[0] * rows.shape[1],))
        n_row_blocks = rows.shape[0] * rows.shape[1] // dims.B + 1
        n_src_tiles = (n_row_blocks - 1) * dims.n_B128_tiles
    else:
        flat = inps.token_position_to_id
        n_row_blocks = dims.N
        n_src_tiles = dims.N * dims.n_B128_tiles
    n_tiles_all = n_row_blocks * dims.n_B128_tiles
    token_ids = sbm.alloc_stack(
        (_pmax, n_tiles_all), dtype=nl.int32, name="resident_token_ids", align=SBUF_QUADRANT_SIZE
    )
    if rows != None:
        nisa.memset(dst=token_ids[0:_pmax, n_src_tiles:n_tiles_all], value=-1)
    # [N * n_B128_tiles, _pmax] rows -> one token per partition, in chunks the DMA transpose accepts.
    for c0 in range(0, n_src_tiles, _pmax):
        n_cols = min(_pmax, n_src_tiles - c0)
        if plain_dma:
            # Element-strided copy: partition p, column c reads row c, element p.
            nisa.dma_copy(
                dst=token_ids[0:_pmax, c0 : c0 + n_cols],
                src=flat.ap(pattern=[[1, _pmax], [_pmax, n_cols]], offset=c0 * _pmax),
            )
            continue
        nisa.dma_transpose(
            dst=token_ids.ap(pattern=[[n_tiles_all, _pmax], [1, 1], [1, 1], [1, n_cols]], offset=c0),
            src=flat.ap(pattern=[[_pmax, n_cols], [1, 1], [1, 1], [1, _pmax]], offset=c0 * _pmax),
        )
    block_expert = sbm.alloc_stack((1, dims.N), dtype=nl.int32, name="resident_block_expert", align=SBUF_QUADRANT_SIZE)
    nisa.dma_copy(dst=block_expert, src=inps.block_to_expert.reshape((1, dims.N)), dge_mode=dge_mode.hwdge)
    block_row = None
    if rows != None:
        block_row = sbm.alloc_stack((1, dims.N), dtype=nl.int32, name="resident_block_row", align=SBUF_QUADRANT_SIZE)
        nisa.dma_copy(dst=block_row, src=row_blocks.reshape((1, dims.N)), dge_mode=dge_mode.hwdge)
    gtab = _aff8_global_table(inps, dims, sbm) if _AFF8[0] != None else None
    return ResidentRouting(
        token_ids=token_ids,
        block_expert=block_expert,
        n_global_experts=n_global_experts,
        block_row=block_row,
        n_row_blocks=n_row_blocks,
        gtab=gtab,
    )


def _select_block_expert(routing, block_idx, dims, sbm, name="block_expert"):
    """[1, 1] int32 expert of block_idx (a Python int or a [1, 1] int32 SBUF scalar)."""
    block_expert = _sbm_alloc(sbm, (1, 1), dtype=nl.int32, name=name, align=SBUF_QUADRANT_SIZE)
    if isinstance(block_idx, int):
        nisa.tensor_copy(dst=block_expert, src=routing.block_expert[0:1, block_idx : block_idx + 1])
    else:
        nisa.tensor_copy(
            dst=block_expert,
            src=routing.block_expert.ap(
                pattern=[[dims.N, 1], [1, 1]], offset=0, scalar_offset=block_idx.view(nl.uint32), indirect_dim=1
            ),
        )
    return block_expert


def _select_block_token_ids(routing, block_idx, dims, sbm, name="token_ids"):
    """[_pmax, n_B128_tiles] int32 token ids of block_idx, one token per partition."""
    n_tiles = dims.n_B128_tiles
    table = routing.token_ids
    token_ids = _sbm_alloc(sbm, (_pmax, n_tiles), dtype=nl.int32, name=name, align=SBUF_QUADRANT_SIZE)
    if routing.block_row != None:
        # token_rows: look the block's row block up first, then copy it like a block.
        row = _sbm_alloc(sbm, (1, 1), dtype=nl.int32, name=f"{name}_row", align=SBUF_QUADRANT_SIZE)
        if isinstance(block_idx, int):
            nisa.tensor_copy(dst=row, src=routing.block_row[0:1, block_idx : block_idx + 1])
        else:
            nisa.tensor_copy(
                dst=row,
                src=routing.block_row.ap(
                    pattern=[[dims.N, 1], [1, 1]], offset=0, scalar_offset=block_idx.view(nl.uint32), indirect_dim=1
                ),
            )
        nr = routing.n_row_blocks
        nisa.tensor_copy(
            dst=token_ids,
            src=table.reshape((_pmax, nr, n_tiles)).ap(
                pattern=[[nr * n_tiles, _pmax], [n_tiles, 1], [1, n_tiles]],
                offset=0,
                scalar_offset=row.view(nl.uint32),
                indirect_dim=1,
            ),
        )
    elif isinstance(block_idx, int):
        nisa.tensor_copy(dst=token_ids, src=table[:, block_idx * n_tiles : (block_idx + 1) * n_tiles])
    else:
        nisa.tensor_copy(
            dst=token_ids,
            # scalar_offset scales by the tensor-shape stride of indirect_dim, so view the
            # table as [_pmax, N, n_tiles] to step one block per index.
            src=table.reshape((_pmax, dims.N, n_tiles)).ap(
                pattern=[[dims.N * n_tiles, _pmax], [n_tiles, 1], [1, n_tiles]],
                offset=0,
                scalar_offset=block_idx.view(nl.uint32),
                indirect_dim=1,
            ),
        )
    return token_ids


def _gather_block_hidden_rows(inps, dims, skip_dma, token_ids, block_hidden_states):
    """Gather the block's hidden-state rows, one token per partition.

    One indirect DMA per B128 tile moves whole [H] rows, so a block costs n_B128_tiles software
    DGE instructions with full-row descriptors.

    Args:
        token_ids: int32 [_pmax, n_B128_tiles] token ids (padding ids are out of bounds).
        block_hidden_states: bf16 SBUF buffer viewed as [_pmax, n_B128_tiles, H].
    """
    hidden_rows = inps.hidden_states.reshape((dims.T, dims.H))
    rows_sb = block_hidden_states.reshape((_pmax, dims.n_B128_tiles, dims.H))
    for t in range(dims.n_B128_tiles):
        nisa.dma_copy(
            dst=rows_sb[0:_pmax, t, 0 : dims.H],
            src=hidden_rows.ap(
                pattern=[[dims.H, _pmax], [1, dims.H]],
                offset=0,
                vector_offset=token_ids.ap([[dims.n_B128_tiles, _pmax], [1, 1]], offset=t),
                indirect_dim=0,
            ),
            oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
        )


def _transpose_hidden_rows(block_hidden_states, block_hidden_states_T, dims, prj_cfg, sbm):
    """PE-transpose gathered token rows into the MX moving layout.

    Produces the layout quantize_block_hidden_state_T expects: for H512 tile h, partition j and
    token b, ``T[j, h, 4 * b + f] = hidden[b, f * H / 4 + h * 128 + j]``. Each 128x128 transpose
    takes one token tile and one (h, f) column block; the PSUM -> SBUF copy interleaves the four
    f blocks of a bank into the x4 element order.
    """
    n_tiles = dims.n_B128_tiles
    n_h = prj_cfg.n_H512_tile
    h_quarter = dims.H // _q_width
    rows_sb = block_hidden_states.reshape((_pmax, n_tiles, dims.H))
    dst_flat = block_hidden_states_T.reshape((_pmax, n_h, dims.B * _q_width))

    # One bank holds 1024 bf16 per partition: two token tiles of four f blocks each.
    tiles_per_bank = min(n_tiles, PSUM_SIZE * 2 // (_q_width * _pmax))
    bank_cols = tiles_per_bank * _q_width * _pmax
    n_bank_groups = div_ceil(n_tiles, tiles_per_bank)
    # Evacuating a bank (a strided read) is slower than filling it (8 transposes), so
    # rotate over 4 banks and 2 evacuating engines. Banks 4-7: gate/up holds banks 0-3 until the
    # transposes start.
    n_psum_bufs = 4
    psum_bufs = []
    for i_buf in range(n_psum_bufs):
        psum_bufs.append(
            nl.ndarray(
                (_pmax, bank_cols),
                dtype=block_hidden_states.dtype,
                buffer=nl.psum,
                address=(0, (NUM_HW_PSUM_BANKS - n_psum_bufs + i_buf) * PSUM_BANK_SIZE),
            )
        )

    for i_h in nl.affine_range(n_h):
        for i_grp in range(n_bank_groups):
            psum_buf = psum_bufs[(i_h * n_bank_groups + i_grp) % n_psum_bufs]
            grp_tiles = min(tiles_per_bank, n_tiles - i_grp * tiles_per_bank)
            for i_t in range(grp_tiles):
                t = i_grp * tiles_per_bank + i_t
                for f in range(_q_width):
                    col = (i_t * _q_width + f) * _pmax
                    src_col = f * h_quarter + i_h * _pmax
                    nisa.nc_transpose(
                        dst=psum_buf[0:_pmax, col : col + _pmax],
                        data=rows_sb[0:_pmax, t, src_col : src_col + _pmax],
                    )
            # psum [j, (t, f, b)] -> dst [j, h, (t, b, f)]. The strided PSUM read costs more than a
            # contiguous copy (a strided SBUF write costs even more), so alternate the evacuating engine.
            nisa.tensor_copy(
                dst=dst_flat.ap(
                    pattern=[
                        [n_h * dims.B * _q_width, _pmax],
                        [_q_width * _pmax, grp_tiles],
                        [_q_width, _pmax],
                        [1, _q_width],
                    ],
                    offset=i_h * dims.B * _q_width + i_grp * bank_cols,
                ),
                src=psum_buf.ap(
                    pattern=[[bank_cols, _pmax], [_q_width * _pmax, grp_tiles], [1, _pmax], [_pmax, _q_width]],
                    offset=0,
                ),
                engine=nisa.vector_engine if (i_h * n_bank_groups + i_grp) % 2 == 0 else nisa.scalar_engine,
            )


def _load_transpose_hidden_rows(inps, dims, prj_cfg, skip_dma, token_ids, buffers, sbm):
    """Gather + transpose one block's hidden rows into buffers.block_hidden_states_T."""
    _gather_block_hidden_rows(inps, dims, skip_dma, token_ids, buffers.block_hidden_states)
    _transpose_hidden_rows(buffers.block_hidden_states, buffers.block_hidden_states_T, dims, prj_cfg, sbm)


def _gate_up_act_mul_tile(
    hidden_qtz_sb,
    hidden_scale_sb,
    gate_weight,
    gate_scale,
    up_weight,
    up_scale,
    cfg,
    out_sb,
    act_op,
    act_bias,
    gate_bank,
    up_bank,
    is_packed_scale,
    is_packed_moving_scale,
):
    """MX gate and up projections for one I tile, with act(gate) * up fused into PSUM evacuation.

    Same math as two gate_up_projection_mx_tp calls (no bias, no clamp, MX quantization) followed by
    the activation and the gate * up multiply, but the activation evacuates the gate PSUM directly
    and the multiply reads up straight from PSUM. That drops the up evacuation copy and the separate
    activation pass over SBUF.

    Args:
        gate_weight / up_weight: SBUF mxfp_x4 [_pmax, n_H512_tile, I_tile] weights.
        gate_scale / up_scale: SBUF uint8 scales for the same tile (standard or packed layout).
        out_sb: bf16 [_pmax, n_I512_tile, BxS, _q_width] output (the intermediate state tile).
        act_bias: fp32 [_pmax, 1] zero bias for nisa.activation.
        gate_bank / up_bank: first PSUM bank for gate / up.
    """
    BxS, I = cfg.BxS, cfg.I
    BxS_tile_sz = min(BxS, PSUM_SIZE * 2 // _q_width)
    if cfg.r_I512_tile > 0 and cfg.zero_unused_partitions:
        nisa.memset(dst=out_sb[:, cfg.n_total_I512_tile - 1, :, :], value=0.0)
    for i_b in range(div_ceil(BxS, BxS_tile_sz)):
        b_off = i_b * BxS_tile_sz
        b_sz = min(BxS_tile_sz, BxS - b_off)
        gate_psum = []
        up_psum = []
        for i_I512 in range(cfg.n_total_I512_tile):
            gate_psum.append(
                nl.ndarray(
                    (_pmax, _q_width, b_sz),
                    dtype=nl.bfloat16,
                    buffer=nl.psum,
                    address=(0, ((gate_bank + i_I512) % NUM_HW_PSUM_BANKS) * PSUM_BANK_SIZE),
                )
            )
            up_psum.append(
                nl.ndarray(
                    (_pmax, _q_width, b_sz),
                    dtype=nl.bfloat16,
                    buffer=nl.psum,
                    address=(0, ((up_bank + i_I512) % NUM_HW_PSUM_BANKS) * PSUM_BANK_SIZE),
                )
            )
        for weight, scale, psum_lst in ((gate_weight, gate_scale, gate_psum), (up_weight, up_scale, up_psum)):
            for i_h in range(cfg.n_H512_tile_sharded):
                _matmul_one_H512_tile(
                    i_H512_tile=i_h,
                    hidden_qtz_sb=hidden_qtz_sb,
                    hidden_scale_sb=hidden_scale_sb,
                    weight_qtz_tv=weight,
                    weight_scale_tv=scale,
                    out_psum_lst=psum_lst,
                    cfg=cfg,
                    cur_BxS_tile_offset=b_off,
                    cur_BxS_tile_sz=b_sz,
                    is_packed_scale=is_packed_scale,
                    is_packed_moving_scale=is_packed_moving_scale,
                )
        for i_I512 in range(cfg.n_total_I512_tile):
            n_p = min(_pmax, I // _q_width - i_I512 * _pmax)
            dst = out_sb[:n_p, i_I512, b_off : b_off + b_sz, :]
            # PSUM [p, q, b] -> [p, b, q]
            psum_pattern = [[_q_width * b_sz, n_p], [1, b_sz], [b_sz, _q_width]]
            nisa.activation(
                dst=dst, op=act_op, data=gate_psum[i_I512].ap(psum_pattern), scale=1.0, bias=act_bias[:n_p, :]
            )
            nisa.tensor_tensor(dst=dst, data1=dst, data2=up_psum[i_I512].ap(psum_pattern), op=nl.multiply)


def compute_one_block(
    block_idx: int,
    next_block_idx: int,
    buffers: SharedBuffers,
    dims: BWMMMXDimensionSizes,
    inps: InputTensors,
    outs: OutputTensors,
    kernel_cfg: BWMMMXConfigs,
    prj_cfg: ProjConfig,
    is_dynamic: bool = False,
    is_first_block: bool = False,
    sbm=None,
    block_expert_for_weights=None,
    next_block_expert_for_weights=None,
    hoisted_gup_weights=None,
    hoisted_gup_bias=None,
    hoisted_down_bias=None,
    name_tag=None,
    routing=None,
    pending_out=None,
    next_is_dynamic=False,
):
    """
    Process one block through complete MoE MLP pipeline.

    Executes gate projection, up projection, activation, and down projection
    for a single block with MXFP4 quantization and expert routing.

    Args:
        block_idx (int): Current block index.
        next_block_idx (int): Next block index for prefetching (None if last).
        buffers (SharedBuffers): Shared computation buffers.
        dims (BWMMMXDimensionSizes): Dimension configuration.
        inps (InputTensors): Input tensors.
        outs (OutputTensors): Output tensors.
        kernel_cfg (BWMMMXConfigs): Kernel configuration.
        prj_cfg (ProjConfig): Projection configuration.
        is_dynamic (bool): Whether from dynamic loop.

    Returns:
        None: Writes results to outs.output.

    Notes:
        - Loads expert weights and scales
        - Prefetches next block hidden states if next_block_idx provided
        - Applies gate/up projections with optional clamping
        - Applies activation function (SiLU or Swish)
        - Computes down projection
        - Scales by expert affinity and accumulates
        - LNC1: no dummy/load-balancing block variant (the shared LNC2 kernel has one so the
          two cores stay in lockstep on an odd block count); every block here is real
        - LNC1: the output write carries no shard offset

    Pseudocode:
        block_expert = routing.block_expert[block_idx]            # on-chip select, no DMA

        if not is_dynamic:
            quantize_block_hidden_state_T(buffers, prj_cfg, dims)  # this block, prefetched earlier

        token_indices_2D = routing.token_ids[block_idx]           # on-chip select, no DMA
        expert_affinity = calculate_expert_affinities(...)

        if next_block_idx != None:
            gather hidden rows of next_block_idx                  # overlaps gate/up
        if pending_out != None and not is_first_block:
            output[ids(block_idx - 1)] += pending_out             # deferred scatter RMW DMA

        for i_tile in range(n_I_tiles):                           # gate/up weights SBUF-resident
            inter[i_tile] = act(hidden @ gate_w[i_tile]) * (hidden @ up_w[i_tile])

        if next_block_idx != None:
            transpose (PE) + quantize the next block's hidden rows
            prefetch next block's gate/up weights + scales (OOB-skipped if same expert)

        block_new = down_proj_mx(inter) * expert_affinity         # affinity in PSUM evacuation
        if pending_out == None:
            output[token_indices_2D] += block_new                 # scatter RMW DMA
        # else block_new is pending_out; the next block (or the final flush) accumulates it
    """
    # Per-block disambiguator for op/alloc names. In the dynamic path block_idx is a
    # runtime SBUF tensor (not a Python int), so f"b{block_idx}" is identical across
    # iterations; callers pass a distinct name_tag (e.g. "chunk_0", "rem") for uniqueness.
    tag = name_tag if name_tag else f"b{block_idx}"
    if sbm != None:
        sbm.open_scope(name="compute_block_scope")
        prev_prefix = sbm.get_name_prefix()
        sbm.set_name_prefix(f"{prev_prefix}{tag}_")

    block_expert = _select_block_expert(routing, block_idx, dims, sbm)

    # Use weight-skip expert if provided (set to E when same as previous block's expert)
    weight_expert = block_expert_for_weights if block_expert_for_weights != None else block_expert

    # ── STATIC_MX: per-block scale-LUT gather ──
    # One scalar_offset gather per LUT pulls all per-expert constants for this block:
    #   gup_scale_lut [_pmax, E, 3] → [_pmax, 3] = [in_recip, gate_combined, up_combined]
    #   down_scale_lut [_pmax, E, 2] → [_pmax, 2] = [down_in_recip, down_combined]
    gate_in_quant_recip = None
    gate_combined_dequant = None
    up_combined_dequant = None
    down_in_quant_recip = None
    down_combined_dequant_per_block = None
    if kernel_cfg.is_static_quant:
        # scalar_offset requires uint32; reinterpret block_expert (int32 [1,1]).
        block_expert_u32 = block_expert.view(nl.uint32)

        # gup gather: 3D pattern walks _pmax × 3-slot expert row.
        # indirect_dim=1 has stride 3, so scalar_offset=block_expert shifts by 3*expert (one expert's slot triple).
        gup_per_block = _sbm_alloc(sbm, (_pmax, 3), dtype=nl.float32, name="gup_per_block", align=SBUF_QUADRANT_SIZE)
        nisa.tensor_copy(
            dst=gup_per_block,
            src=inps.gup_scale_lut_sb.ap(
                pattern=[[dims.E * 3, _pmax], [3, 1], [1, 3]],
                offset=0,
                scalar_offset=block_expert_u32,
                indirect_dim=1,
            ),
        )
        gate_in_quant_recip = gup_per_block[:, 0:1]
        gate_combined_dequant = gup_per_block[:, 1:2]
        up_combined_dequant = gup_per_block[:, 2:3]

        # down gather: same shape, count=2 on the inner slot dim.
        down_per_block = _sbm_alloc(sbm, (_pmax, 2), dtype=nl.float32, name="down_per_block", align=SBUF_QUADRANT_SIZE)
        nisa.tensor_copy(
            dst=down_per_block,
            src=inps.down_scale_lut_sb.ap(
                pattern=[[dims.E * 2, _pmax], [2, 1], [1, 2]],
                offset=0,
                scalar_offset=block_expert_u32,
                indirect_dim=1,
            ),
        )
        down_in_quant_recip = down_per_block[:, 0:1]
        down_combined_dequant_per_block = down_per_block[:, 1:2]

    # fp8 pre-quantized hidden: the load/transpose happens in-block (after token indices are
    # available) with no online quantize and no bf16 prefetch buffers, so skip the bf16 pipeline.
    if not kernel_cfg.is_fp8_hidden:
        # quantize prefetched data. Note that online quantize can only quantize to fp8
        # only quantize here if it is a static block. for dynamic block we quantize immediately after fetching
        if not is_dynamic:
            if kernel_cfg.is_static_quant:
                quantize_block_hidden_state_T_static_mx(buffers, prj_cfg, dims, gate_in_quant_recip)
            else:
                quantize_block_hidden_state_T(buffers, prj_cfg, dims)

        _free_hidden_bufs(sbm, buffers.block_hidden_states, buffers.block_hidden_states_T)
        buffers.block_hidden_states_T = None
        buffers.block_hidden_states = None

    """
    Alloc block_hidden_states and start DMA load early to overlap with up proj.
    block_hidden_states_T alloc + transpose deferred to after activation+multiply
    to prevent compiler from scheduling nc_transpose during up proj.
    """

    buffers.hidden_qtz_sb = buffers.hidden_qtz_sb.reshape((_pmax, prj_cfg.n_H512_tile, dims.B))
    if not kernel_cfg.is_static_quant:
        # STATIC_MX: hidden_scale_sb is the small dummy [_pmax, _pmax] all-127 buffer; matmul
        # site reads it as a 2D view (per-tile shape), so we skip the per-block reshape.
        # fp8 path: scales are packed (n_packed scale blocks); bf16 path: one tile per H512 tile.
        n_scale_dim = div_ceil(prj_cfg.n_H512_tile, 4) if kernel_cfg.is_fp8_hidden else prj_cfg.n_H512_tile
        buffers.hidden_scale_sb = buffers.hidden_scale_sb.reshape((_pmax, n_scale_dim, dims.B))

    flatten_free_dim = prj_cfg.n_total_I512_tile * dims.B * _q_width
    # Tile by default when I > 512
    use_tiled_gup = dims.I > _I_TILE_SZ
    if use_tiled_gup:
        logger.debug(f"Tiling gate/up weights: I={dims.I} > {_I_TILE_SZ}")

    # will be used in non-tiled path if down projection output can't be created. We will re-use address
    # of the gup weight qtz sb which should be finished, but was loaded outside of gup proj scope.
    _gup_wt_addr = None

    if use_tiled_gup:
        # Skip index vector computation if tile 0 was prefetched during previous block's down proj
        _skip_tile0 = not is_first_block and buffers.gup_tile_buf_a != None
        # Packed scales address via scalar_offset=block_expert, so the per-expert index
        # vector is never consumed, static mx has fixed scales
        _row_scales = (
            not kernel_cfg.is_static_quant
            and not kernel_cfg.use_packed_scales
            and _rows_scale_loadable(inps.gate_up_proj_scale)
        )
        if _skip_tile0 or kernel_cfg.is_static_quant or kernel_cfg.use_packed_scales or _row_scales:
            # Tile 0 weights, scales, and index vector already cached from prefetch, or the
            # scale load needs no index vector.
            gup_n_quadrants_needed = prj_cfg.H0 // SBUF_QUADRANT_SIZE
            gup_token_indices_on_p = inps.p_gup_idx_vector_int32
        else:
            scale_shape = inps.gate_up_proj_scale.shape
            gup_n_quadrants_needed = prj_cfg.H0 // SBUF_QUADRANT_SIZE
            # Able to reuse index vector calculated here because I_TILE_SIZE >= 512, which means we have the same layout as gup proj
            gup_token_indices_on_p = _generate_expert_index_vector(
                expert_index=weight_expert,
                dst_idx_vector=inps.p_gup_idx_vector,
                scale_factor=scale_shape[1],
                n_quadrants_needed=gup_n_quadrants_needed,
                n_remaining_partition=0,
                name_prefix="gup_expert_index_vector",
                sbm=sbm,
                dst_int32=inps.p_gup_idx_vector_int32,
            )
    else:
        # Skip gup weight/scale/bias load if prefetched during previous block's down proj.
        # Works for both static and dynamic paths: block N-1's down proj prefetches block N's
        # weights into hoisted_gup_weights, and block N can skip its main load.
        _skip_gup_load = not is_first_block and hoisted_gup_weights != None
        # Save stack addr before gup weight alloc for potential reuse by down proj output
        _gup_wt_addr = sbm.stack_curr_addr if sbm != None and hoisted_gup_weights == None else None
        if _skip_gup_load:
            # Weights/scales/bias and index vector already cached from prefetch.
            gate_and_up_weights = hoisted_gup_weights
            gup_bias = hoisted_gup_bias
            gup_n_quadrants_needed = prj_cfg.H0 // SBUF_QUADRANT_SIZE
            gup_token_indices_on_p = inps.p_gup_idx_vector_int32
            gate_and_up_scales = inps.gup_scales_sb
        else:
            gate_and_up_weights, gate_and_up_scales, gup_bias, gup_token_indices_on_p, gup_n_quadrants_needed = (
                load_gup_weights_scales_mx(
                    inps,
                    weight_expert,
                    dims,
                    prj_cfg=prj_cfg,
                    skip_dma=kernel_cfg.skip_dma,
                    sbm=sbm,
                    dst_weight=hoisted_gup_weights,
                    dst_bias=hoisted_gup_bias,
                    use_packed_scales=kernel_cfg.use_packed_scales,
                    skip_scales=kernel_cfg.is_static_quant,
                )
            )

    # For non-tiled path, load down weights early to overlap with gup compute
    # For tiled path, defer to after gup scope to avoid DMA contention with tile prefetches
    if not use_tiled_gup:
        down_scale_sb, down_bias_sb = load_down_proj_weights_mx(
            inps,
            weight_expert,
            buffers.down_weight_qtz,
            dims,
            prj_cfg,
            kernel_cfg.skip_dma,
            gup_token_indices_on_p,
            gup_n_quadrants_needed,
            dst_scale=buffers.down_scale_sb,
            sbm=sbm,
            dst_bias=hoisted_down_bias,
            use_packed_scales=kernel_cfg.use_packed_scales,
            skip_scales=kernel_cfg.is_static_quant,
        )
    down_weight_qtz_viewed = buffers.down_weight_qtz

    token_indices_2D = _select_block_token_ids(routing, block_idx, dims, sbm, name="token_indices_2D")

    kernel_assert(
        token_indices_2D.shape == (_pmax, dims.n_B128_tiles),
        f"Expect token_indices_2D to have shape (128, {dims.n_B128_tiles}), got {token_indices_2D.shape}",
    )

    # fp8 pre-quantized hidden: this block's hidden was already gathered + transposed into
    # hidden_qtz_sb / hidden_scale_sb during the previous block's deferred stage (or driver init for
    # the first block). The next block's gather (DMA) is issued below during gate/up, and its
    # transpose (PE) is deferred until after gate/up so it overlaps down projection.

    if kernel_cfg.is_affinities_packed and _AFF8[0] != None:
        # compact_affinities: the compact top-8 tail.
        expert_affinity = _extract_block_affinity_compact(
            buffers.block_hidden_concat, block_idx, routing, dims, kernel_cfg, sbm=sbm
        )
    elif kernel_cfg.is_affinities_packed:
        # Extract expert affinities for this block expert
        expert_affinity = _extract_block_affinity(
            buffers.block_hidden_concat,
            block_expert,
            dims,
            kernel_cfg,
            sbm=sbm,
            local_expert_start_idx=inps.local_expert_start_idx,
            n_global_experts=routing.n_global_experts,
        )
    else:
        expert_affinity = calculate_expert_affinities(
            inps.expert_affinities_masked,
            token_indices_2D,
            block_expert,
            dims.E,
            dims.B // _pmax,
            nl.float32,
            kernel_cfg.skip_dma,
            sbm=sbm,
        )

    if next_block_idx != None and not kernel_cfg.is_fp8_hidden:
        _alloc_hidden_src_buf(sbm, buffers, dims, prj_cfg, kernel_cfg, tag=f"nb{next_block_idx}_")
        _nb_token_ids = _select_block_token_ids(routing, next_block_idx, dims, sbm, name="nb_token_ids")
        _gather_block_hidden_rows(inps, dims, kernel_cfg.skip_dma, _nb_token_ids, buffers.block_hidden_states)

    # Deferred accumulate: add the PREVIOUS block's output now. Issuing it at the end of its own
    # block would leave a DMA in flight at a dynamic-loop entry or back-edge, where every engine
    # drains. It goes after the next block's hidden gather: both use the software DGE queue, and
    # the gather is on the critical path while the accumulate has a full block of slack.
    if pending_out != None and not is_first_block:
        if isinstance(block_idx, int):
            prev_idx = block_idx - 1
        else:
            prev_idx = _sbm_alloc(sbm, (1, 1), dtype=nl.int32, name="prev_block_idx", align=SBUF_QUADRANT_SIZE)
            nisa.tensor_scalar(dst=prev_idx, data=block_idx, op0=nl.subtract, operand0=1, op1=nl.maximum, operand1=0)
        _flush_pending_output(routing, prev_idx, pending_out, outs, dims, sbm)

    # fp8: gather (DMA only) the next block's concat rows here so it overlaps gate/up compute.
    # The matching transpose (PE) is deferred until after gate/up (see below).
    if next_block_idx != None and kernel_cfg.is_fp8_hidden:
        _prev_pfx = sbm.get_name_prefix() if sbm != None else None
        if sbm != None:
            sbm.set_name_prefix(f"{_prev_pfx}fp8pf_")
        _nb_tok = _select_block_token_ids(routing, next_block_idx, dims, sbm, name="fp8_nb_token_ids")
        _load_fp8_rows(
            inps,
            dims,
            kernel_cfg.skip_dma,
            token_indices_on_p=_nb_tok,
            block_hidden_concat=buffers.block_hidden_concat,
        )
        if sbm != None:
            sbm.set_name_prefix(_prev_pfx)
    """
    GATE/UP PROJECTIONS + ACTIVATION + MULTIPLY
    
    Scoped so that gate/up weights, bias, gate_proj_out_sb, up_proj_out_sb,
    and all internal projection allocations are freed after producing intermediate_state_sb.
    """
    # intermediate_state_sb is allocated outside the gate/up scope so it survives for down projection
    intermediate_state_sb = _sbm_alloc(
        sbm,
        (_pmax, flatten_free_dim),
        dtype=nl.bfloat16,
        name="intermediate_state_sb",
        align=SBUF_QUADRANT_SIZE,
    )

    if sbm != None:
        sbm.open_scope(name="gup_proj")

    if use_tiled_gup:
        """
        Tiled gate/up projection: tile along I dimension with double-buffering.

        When full gate/up weights don't fit in SBUF, we tile along the I (intermediate)
        dimension in chunks of _I_TILE_SZ (512). Two weight buffers (A, B) alternate
        so DMA load of the next tile overlaps with compute on the current tile.

        Timeline (3 I-tiles example):
            buf_A: [load T0]──────[compute T0 gate]─[compute T0 up]──────────────────[load T2]──[compute T2 gate]─[compute T2 up]
            buf_B: ───────────────[load T1]──────────────────────────[compute T1 gate]─[compute T1 up]

        Memory layout per tile:
            weight buf: (_pmax, 2, n_H512_tile_sharded, _I_TILE_SZ)  -- gate+up interleaved
            scales:     loaded once for full I, sliced per tile
            bias:       loaded once for full I, sliced per tile
            output:     (_pmax, n_I_tiles, B, _q_width) -- one slice per tile

        Per-tile pipeline:
            1. gate_proj  = hidden @ weight[gate_tile] + bias[tile]
            2. (prefetch next tile weight into alternate buffer)
            3. up_proj    = hidden @ weight[up_tile] + bias[tile]
            4. clamp → activate → gate * up → intermediate_state[tile]
        """
        n_I_tiles = div_ceil(dims.I, _I_TILE_SZ)
        wt_dtype = inps.gate_up_proj_weight.dtype
        last_tile_partial = (dims.I % _I_TILE_SZ) != 0
        tile_buf_shape = (_pmax, 2, prj_cfg.n_H512_tile_sharded, _I_TILE_SZ)

        """
        Use persistent tile buffer + one scope-local buffer for double buffering.
        If persistent buffer wasn't pre-allocated (SBUF budget too tight to keep it
        alive through down proj), allocate a scope-local buffer instead — loses
        cross-block prefetch but still enables double-buffering within the I-tile loop.       
        """

        gup_wt_a = (
            buffers.gup_tile_buf_a
            if buffers.gup_tile_buf_a != None
            else _sbm_alloc(
                sbm,
                tile_buf_shape,
                dtype=wt_dtype,
                name="gup_wt_a",
                align=SBUF_QUADRANT_SIZE,
            )
        )

        # Single-buffering aliases both ping-pong slots, which is only correct when the
        # regime-3 inter-tile prefetch (the sole nxt_buf write, gated `not _use_h_chunked`)
        # is disabled — i.e. the H-chunked regime (H >= 3072). Allocate the second buffer
        # unless we're both in that regime AND out of SBUF room.
        # weight_stream tile_major 2: every I tile is one DMA into the ping-pong buffer (regime 3).
        _h_chunked_active = dims.H >= 3072 and not kernel_cfg.gup_full_persistent and _WSTREAM[0] != 2
        _tile_buf_bytes = 2 * prj_cfg.n_H512_tile_sharded * _I_TILE_SZ * sizeinbytes(wt_dtype)
        _can_double_buffer = sbm == None or sbm.get_free_space() >= _tile_buf_bytes
        if _pf_bufs() is not None:
            # (prefetch_tiles) the k persistent tile buffers; tiles 0 .. k-1 came from the cross-block prefetch
            # unless this is the first block.
            gup_wt_bufs = list(_pf_bufs())
        elif _h_chunked_active and not _can_double_buffer:
            logger.debug(
                f"Single-buffering gate/up tile weights (free={sbm.get_free_space()} B, "
                f"need={_tile_buf_bytes} B for gup_wt_b)."
            )
            gup_wt_bufs = [gup_wt_a, gup_wt_a]
        else:
            gup_wt_b = _sbm_alloc(
                sbm,
                (tile_buf_shape),
                dtype=wt_dtype,
                name="gup_wt_b",
                align=SBUF_QUADRANT_SIZE,
            )
            gup_wt_bufs = [gup_wt_a, gup_wt_b]

        # Load full scales once into pre-allocated inps.gup_scales_sb (skip if prefetched).
        # STATIC_MX skips entirely: inps.gup_scales_sb holds persistent dummy 127 from top-level memset.
        if not _skip_tile0 and not kernel_cfg.is_static_quant:
            if kernel_cfg.use_packed_scales:
                # Packed scale load: HBM is [E, _pmax, n_packed_gup, 2, I]; mirror
                # production's gate/up split per skip_weight.
                scale_shape = inps.gate_up_proj_scale.shape
                n_packed_gup = scale_shape[2]
                if kernel_cfg.skip_dma.skip_weight:
                    nisa.dma_copy(
                        dst=inps.gup_scales_sb[:_pmax, :n_packed_gup, :2, : prj_cfg.I],
                        src=inps.gate_up_proj_scale.ap(
                            pattern=[
                                [n_packed_gup * 2 * prj_cfg.I, _pmax],
                                [2 * prj_cfg.I, n_packed_gup],
                                [prj_cfg.I, 2],
                                [1, prj_cfg.I],
                            ],
                            offset=0,
                            scalar_offset=block_expert,
                            indirect_dim=0,
                        ),
                        oob_mode=oob_mode.skip,
                        dge_mode=dge_mode.hwdge,
                        name=scoped_name(f"dma_gup_scales_packed_tile0_{tag}"),
                        **_wkw(0),
                    )
                else:
                    nisa.dma_copy(
                        dst=inps.gup_scales_sb[:_pmax, :n_packed_gup, 0:1, : prj_cfg.I],
                        src=inps.gate_up_proj_scale.ap(
                            pattern=[
                                [n_packed_gup * 2 * prj_cfg.I, _pmax],
                                [2 * prj_cfg.I, n_packed_gup],
                                [1, prj_cfg.I],
                            ],
                            offset=0,
                            scalar_offset=block_expert,
                            indirect_dim=0,
                        ),
                        oob_mode=oob_mode.error,
                        dge_mode=dge_mode.hwdge,
                        name=scoped_name(f"dma_gate_scales_packed_tile0_{tag}"),
                    )
                    nisa.dma_copy(
                        dst=inps.gup_scales_sb[:_pmax, :n_packed_gup, 1:2, : prj_cfg.I],
                        src=inps.gate_up_proj_scale.ap(
                            pattern=[
                                [n_packed_gup * 2 * prj_cfg.I, _pmax],
                                [2 * prj_cfg.I, n_packed_gup],
                                [1, prj_cfg.I],
                            ],
                            offset=prj_cfg.I,
                            scalar_offset=block_expert,
                            indirect_dim=0,
                        ),
                        oob_mode=oob_mode.error,
                        dge_mode=dge_mode.hwdge,
                        name=scoped_name(f"dma_up_scales_packed_tile0_{tag}"),
                    )
            elif _row_scales:
                _load_expert_scale_rows(
                    inps.gup_scales_sb, inps.gate_up_proj_scale, block_expert, kernel_cfg.skip_dma.skip_weight
                )
            else:
                scale_shape = inps.gate_up_proj_scale.shape
                gup_scale_view = inps.gate_up_proj_scale.reshape(
                    (scale_shape[0] * scale_shape[1], scale_shape[2], scale_shape[3], scale_shape[4])
                )
                full_n_H512_tile_scale = scale_shape[3]
                stride_dim0 = 2 * full_n_H512_tile_scale * prj_cfg.I
                nisa.dma_copy(
                    src=gup_scale_view.ap(
                        pattern=[
                            [stride_dim0, _pmax],
                            [full_n_H512_tile_scale * prj_cfg.I, 2],
                            [prj_cfg.I, prj_cfg.n_H512_tile_sharded],
                            [1, prj_cfg.I],
                        ],
                        offset=0,
                        vector_offset=gup_token_indices_on_p.ap(
                            [[1, _pmax], [1, 1]],
                            offset=0,
                        ),
                        indirect_dim=0,
                    ),
                    dst=inps.gup_scales_sb[:_pmax, :2, : prj_cfg.n_H512_tile_sharded, : prj_cfg.I],
                    oob_mode=oob_mode.skip,
                )

        # Load full bias once. If hoisted_gup_bias is provided (skip_weight path), reuse
        # it across blocks and use weight_expert so OOB-skip preserves previous expert's bias.
        gup_bias_full = None
        if inps.gate_and_up_proj_bias:
            if hoisted_gup_bias != None:
                gup_bias_full = hoisted_gup_bias
                _bias_expert = weight_expert
            else:
                gup_bias_full = _sbm_alloc(
                    sbm,
                    (_pmax, 2, prj_cfg.n_total_I512_tile, _q_width),
                    dtype=inps.gate_and_up_proj_bias.dtype,
                    name="gup_bias_full",
                    align=SBUF_QUADRANT_SIZE,
                )
                _bias_expert = block_expert
            if dims.I < _pmax * _q_width:
                if hoisted_gup_bias == None:
                    nisa.memset(dst=gup_bias_full[:, :, 0, :], value=0.0)
                # This is due to needing 4_I for QMX
                I_par_dim = dims.I // 4
                bias_stride_dim0 = 2 * prj_cfg.n_total_I512_tile * _q_width
                bias_stride_dim1 = prj_cfg.n_total_I512_tile * _q_width
                nisa.dma_copy(
                    dst=gup_bias_full[:I_par_dim, :, :, :],
                    src=inps.gate_and_up_proj_bias.ap(
                        pattern=[
                            [bias_stride_dim0, I_par_dim],
                            [bias_stride_dim1, 2],
                            [_q_width, prj_cfg.n_total_I512_tile],
                            [1, _q_width],
                        ],
                        offset=0,
                        scalar_offset=_bias_expert,
                        indirect_dim=0,
                    ),
                    oob_mode=oob_mode.skip if kernel_cfg.skip_dma.skip_weight else oob_mode.error,
                    dge_mode=dge_mode.hwdge,
                )
            else:
                bias_stride_dim1 = 2 * prj_cfg.n_total_I512_tile * _q_width
                bias_stride_dim2 = prj_cfg.n_total_I512_tile * _q_width
                nisa.dma_copy(
                    dst=gup_bias_full,
                    src=inps.gate_and_up_proj_bias.ap(
                        pattern=[
                            [bias_stride_dim1, _pmax],
                            [bias_stride_dim2, 2],
                            [_q_width, prj_cfg.n_total_I512_tile],
                            [1, _q_width],
                        ],
                        offset=0,
                        scalar_offset=_bias_expert,
                        indirect_dim=0,
                    ),
                    oob_mode=oob_mode.skip if kernel_cfg.skip_dma.skip_weight else oob_mode.error,
                    dge_mode=dge_mode.hwdge,
                )

        # Single scratch buffer for up projection output. Gate writes directly to
        # intermediate_state_sb, then gate * up overwrites it in place.
        up_tile_sb = _sbm_alloc(
            sbm,
            (_pmax, 1, dims.B, _q_width),
            dtype=nl.bfloat16,
            name="up_tile_sb",
            align=SBUF_QUADRANT_SIZE,
        )

        # Pre-build ProjConfig for full tiles; update I-fields for last tile if partial
        tile_prj_cfg = ProjConfig(
            H=dims.H,
            I=_I_TILE_SZ,
            BxS=dims.B,
            force_lnc1=True,
            n_prgs=1,
            prg_id=0,
            use_stream_shuffle_broadcast=False,
            sharding_config="H",
            zero_unused_partitions=prj_cfg.zero_unused_partitions,
        )

        # Initial weight load:
        #   - full_resident path: load FULL gup (gate+up, full I) into the persistent
        #     buffer when no prior cross-block prefetch supplied them (first block of a shard).
        #   - tile-streaming path: load just tile 0; subsequent tiles stream during the loop.
        if not _skip_tile0:
            if kernel_cfg.gup_full_persistent:
                _prefetch_gup_tile0(
                    inps,
                    block_expert,
                    prj_cfg,
                    buffers,
                    dims,
                    kernel_cfg.skip_dma,
                    sbm,
                    name_prefix=f"first_b{block_idx}",
                    use_packed_scales=kernel_cfg.use_packed_scales,
                    skip_scales=kernel_cfg.is_static_quant,
                    full_I_load=True,
                )
            else:
                _load_gup_weight_tile(
                    inps=inps,
                    block_expert=block_expert,
                    prj_cfg=prj_cfg,
                    skip_dma=kernel_cfg.skip_dma,
                    dst_weight=gup_wt_bufs[0],
                    dst_scale=None,
                    dst_bias=None,
                    token_indices_on_p=gup_token_indices_on_p,
                    I_offset=0,
                    dims=dims,
                    sbm=sbm,
                )

        # Pre-build a flat view of the full-resident gup buffer for slicing per I-tile.
        if kernel_cfg.gup_full_persistent:
            _gup_resident_flat = buffers.gup_tile_buf_a.flatten_dims(1, 2)

        # Use H-chunked weight loading for tiles 1+ when H is large (tile-streaming path only)
        _use_h_chunked = dims.H >= 3072 and not kernel_cfg.gup_full_persistent and _WSTREAM[0] != 2
        kernel_assert(_WSTREAM[0] != 2 or gup_wt_bufs[0] is not gup_wt_bufs[1], "weight_stream tile_major 2 needs both gate/up tile buffers")

        # STATIC_MX/SW-dequant path.
        # gup_scales layout: standard [_pmax, 2, n_H512_tile_sharded, I] flattened
        # to [_pmax, 2*n_H512, I] for gate/up slicing. Packed layout
        # [_pmax, n_packed_gup, 2, I] needs no flatten. STATIC_MX uses a shared
        # all-127 dummy buffer that the sub-kernel reads directly.
        if kernel_cfg.use_packed_scales or kernel_cfg.is_static_quant:
            gup_scales_flat = None
        else:
            gup_scales_flat = inps.gup_scales_sb.reshape((_pmax, 2 * prj_cfg.n_H512_tile_sharded, dims.I))
        gup_bias_flat = (
            gup_bias_full.reshape((_pmax, 2 * prj_cfg.n_total_I512_tile, _q_width)) if gup_bias_full != None else None
        )
        intermediate_state_tiled = intermediate_state_sb.reshape((_pmax, prj_cfg.n_total_I512_tile, dims.B, _q_width))

        # Loop-invariant: gate activation op depends only on kernel_cfg, not the tile.
        # STATIC_MX fuses silu into the gate projection, but only when there is no gate
        # clamp (the clamp must run between dequant+bias and the activation).
        _gate_act_op = (
            get_nl_act_fn_from_type(kernel_cfg.activation_function)
            if kernel_cfg.is_static_quant and not kernel_cfg.has_gate_clamp
            else None
        )

        # Regime 1 without bias or clamps fuses act(gate) * up into the PSUM evacuation.
        _fuse_gup_epilogue = (
            kernel_cfg.gup_full_persistent
            and not kernel_cfg.is_static_quant
            and not inps.gate_and_up_proj_bias
            and kernel_cfg.gate_clamp_upper_limit == None
            and kernel_cfg.gate_clamp_lower_limit == None
            and kernel_cfg.up_clamp_upper_limit == None
            and kernel_cfg.up_clamp_lower_limit == None
        )

        # ── Per-I-tile weight sourcing: three regimes, selected by the if/elif/else below ──
        # Each iteration computes one 512-wide I-tile (we are here because I > _I_TILE_SZ).
        # The regimes differ only in WHERE this tile's gate/up weights come from:
        #   1. Full-resident (gup_full_persistent): the entire gate+up (all I-tiles) is
        #      resident in one big SBUF buffer that persists across blocks; each tile is a
        #      slice at cur_I_offset, no per-tile DMA. Enabled only for STATIC_MX +
        #      skip_weight + I>512 that fits the SBUF budget (see _gup_full_persistent).
        #   2. H-chunked (_use_h_chunked and i_tile > 0; requires H >= 3072): weights are
        #      streamed from HBM via gate_up_projection_mx_tp's own dst_weight_sb DMA,
        #      interleaved with matmul. Tile 0 still falls to regime 3.
        #   3. Ping-pong tile (else): the default streaming scheme — the buffer holds one
        #      I-tile (gup_wt_a/gup_wt_b alternate). Tile 0 is the cross-block prefetch
        #      already in SBUF; tiles 1+ are staged into the alternate buffer by the
        #      _load_gup_weight_tile call below (gated on not _use_h_chunked). Handles tile 0
        #      always, and every tile when H < 3072.
        # (prefetch_tiles) k tile buffers (2 = the default ping-pong); tiles 0 .. _n_pref - 1 are already in SBUF from
        # the cross-block prefetch (by default tile 0 only, or none on the first block).
        _nbuf = len(gup_wt_bufs) if _pf_bufs() is not None else 2
        _n_pref = (_nbuf if _pf_bufs() is not None else 1) if _skip_tile0 else 0
        for i_tile in nl.affine_range(n_I_tiles):
            cur_buf = i_tile % _nbuf
            nxt_buf = (i_tile + 1) % _nbuf
            cur_I_offset = i_tile * _I_TILE_SZ
            cur_I_tile_sz = min(_I_TILE_SZ, dims.I - cur_I_offset)
            is_last_tile = i_tile == n_I_tiles - 1

            # Update ProjConfig I-fields for last partial tile
            if is_last_tile and last_tile_partial:
                tile_prj_cfg.I = cur_I_tile_sz
                tile_prj_cfg._generate_H_shard_config()

            gate_bias_view = None
            if inps.gate_and_up_proj_bias:
                gate_bias_view = gup_bias_flat.slice(dim=1, start=i_tile, end=i_tile + 1)
            up_bias_view = None
            if inps.gate_and_up_proj_bias:
                up_bias_view = gup_bias_flat.slice(
                    dim=1, start=prj_cfg.n_total_I512_tile + i_tile, end=prj_cfg.n_total_I512_tile + i_tile + 1
                )

            if kernel_cfg.gup_full_persistent:
                # Regime 1 (full-resident): slice this I-tile from the persistent buffer; no DMA.
                gate_wt_view = _gup_resident_flat.slice(1, 0, prj_cfg.n_H512_tile_sharded).slice(
                    2, cur_I_offset, cur_I_offset + cur_I_tile_sz
                )
                up_wt_view = _gup_resident_flat.slice(
                    1, prj_cfg.n_H512_tile_sharded, 2 * prj_cfg.n_H512_tile_sharded
                ).slice(2, cur_I_offset, cur_I_offset + cur_I_tile_sz)

                if kernel_cfg.is_static_quant:
                    gate_weight_scale_arg = inps.gup_scales_sb
                    up_weight_scale_arg = inps.gup_scales_sb
                elif kernel_cfg.use_packed_scales:
                    gate_weight_scale_arg = inps.gup_scales_sb[:, :, 0, cur_I_offset : cur_I_offset + cur_I_tile_sz]
                    up_weight_scale_arg = inps.gup_scales_sb[:, :, 1, cur_I_offset : cur_I_offset + cur_I_tile_sz]
                else:
                    gate_weight_scale_arg = gup_scales_flat.slice(1, 0, prj_cfg.n_H512_tile_sharded).slice(
                        2, cur_I_offset, cur_I_offset + cur_I_tile_sz
                    )
                    up_weight_scale_arg = gup_scales_flat.slice(
                        1, prj_cfg.n_H512_tile_sharded, 2 * prj_cfg.n_H512_tile_sharded
                    ).slice(2, cur_I_offset, cur_I_offset + cur_I_tile_sz)

                if _fuse_gup_epilogue:
                    _gate_up_act_mul_tile(
                        hidden_qtz_sb=buffers.hidden_qtz_sb,
                        hidden_scale_sb=buffers.hidden_scale_sb,
                        gate_weight=gate_wt_view,
                        gate_scale=gate_weight_scale_arg,
                        up_weight=up_wt_view,
                        up_scale=up_weight_scale_arg,
                        cfg=tile_prj_cfg,
                        out_sb=intermediate_state_tiled[:_pmax, i_tile : i_tile + 1, : dims.B, :_q_width],
                        act_op=get_nl_act_fn_from_type(kernel_cfg.activation_function),
                        act_bias=inps.activation_bias,
                        gate_bank=0 if i_tile % 2 == 0 else 1,
                        up_bank=2 if i_tile % 2 == 0 else 3,
                        is_packed_scale=kernel_cfg.use_packed_scales,
                        is_packed_moving_scale=kernel_cfg.is_fp8_hidden,
                    )
                else:
                    gate_up_projection_mx_tp(
                        hidden_qtz_sb=buffers.hidden_qtz_sb,
                        hidden_scale_sb=buffers.hidden_scale_sb,
                        weight_qtz=gate_wt_view,
                        weight_scale=gate_weight_scale_arg,
                        bias_sb=gate_bias_view,
                        cfg=tile_prj_cfg,
                        sbm=sbm,
                        psum_bank_offset=0 if i_tile % 2 == 0 else 1,
                        name_prefix=f"gate_t{i_tile}",
                        out_sb=intermediate_state_tiled[:_pmax, i_tile : i_tile + 1, : dims.B, :_q_width],
                        is_packed_scale=kernel_cfg.use_packed_scales,
                        is_packed_moving_scale=kernel_cfg.is_fp8_hidden,
                        w_dequant_scale=gate_combined_dequant,
                        activation_op=_gate_act_op,
                        is_static_quant=kernel_cfg.is_static_quant,
                    )

                    gate_up_projection_mx_tp(
                        hidden_qtz_sb=buffers.hidden_qtz_sb,
                        hidden_scale_sb=buffers.hidden_scale_sb,
                        weight_qtz=up_wt_view,
                        weight_scale=up_weight_scale_arg,
                        bias_sb=up_bias_view,
                        cfg=tile_prj_cfg,
                        sbm=sbm,
                        psum_bank_offset=2 if i_tile % 2 == 0 else 3,
                        name_prefix=f"up_t{i_tile}",
                        out_sb=up_tile_sb[:_pmax, 0:1, : dims.B, :_q_width],
                        is_packed_scale=kernel_cfg.use_packed_scales,
                        is_packed_moving_scale=kernel_cfg.is_fp8_hidden,
                        w_dequant_scale=up_combined_dequant,
                        activation_op=None,
                        is_static_quant=kernel_cfg.is_static_quant,
                    )
            elif _use_h_chunked and i_tile > 0 and i_tile >= _n_pref:
                # (prefetch_tiles: tiles below _n_pref were prefetched into gup_wt_bufs and take regime 3.)
                # Regime 2 (H-chunked): stream this I-tile's weight from HBM in chunks,
                # interleaving DMA with matmul. HBM weight view (gate half, then up).
                if _WSTREAM[0]:
                    # tile-major gate/up: [128, n_H512, 512] per gate / up, a 3-H512 chunk = 3 KiB runs.
                    _gup_t = _gup_tile_src(inps, block_expert, i_tile)
                    gate_wt_hbm = (
                        _gup_t.select(dim=1, index=0)
                        .slice(dim=1, start=0, end=prj_cfg.n_H512_tile_sharded)
                        .slice(dim=2, start=0, end=cur_I_tile_sz)
                    )
                    up_wt_hbm = (
                        _gup_t.select(dim=1, index=1)
                        .slice(dim=1, start=0, end=prj_cfg.n_H512_tile_sharded)
                        .slice(dim=2, start=0, end=cur_I_tile_sz)
                    )
                else:
                    gate_wt_hbm = (
                        inps.gate_up_proj_weight.select(dim=0, index=block_expert)
                        .select(dim=1, index=0)
                        .slice(dim=1, start=0, end=prj_cfg.n_H512_tile_sharded)
                        .slice(dim=2, start=cur_I_offset, end=cur_I_offset + cur_I_tile_sz)
                    )
                    up_wt_hbm = (
                        inps.gate_up_proj_weight.select(dim=0, index=block_expert)
                        .select(dim=1, index=1)
                        .slice(dim=1, start=0, end=prj_cfg.n_H512_tile_sharded)
                        .slice(dim=2, start=cur_I_offset, end=cur_I_offset + cur_I_tile_sz)
                    )

                # Build flattened (gate/up × n_H512) gate/up dst views via
                # nl.NkiTensor so partition stride is preserved through to the AP with base tensor AP stride
                cur_wt_flat = gup_wt_bufs[cur_buf].flatten_dims(1, 2)
                gate_dst = cur_wt_flat.slice(1, 0, prj_cfg.n_H512_tile_sharded).slice(2, 0, _I_TILE_SZ)
                up_dst = cur_wt_flat.slice(1, prj_cfg.n_H512_tile_sharded, 2 * prj_cfg.n_H512_tile_sharded).slice(
                    2, 0, _I_TILE_SZ
                )

                if kernel_cfg.is_static_quant:
                    gate_weight_scale_arg = inps.gup_scales_sb
                    up_weight_scale_arg = inps.gup_scales_sb
                elif kernel_cfg.use_packed_scales:
                    gate_weight_scale_arg = inps.gup_scales_sb[:, :, 0, cur_I_offset : cur_I_offset + cur_I_tile_sz]
                    up_weight_scale_arg = inps.gup_scales_sb[:, :, 1, cur_I_offset : cur_I_offset + cur_I_tile_sz]
                else:
                    gate_weight_scale_arg = gup_scales_flat.slice(1, 0, prj_cfg.n_H512_tile_sharded).slice(
                        2, cur_I_offset, cur_I_offset + cur_I_tile_sz
                    )
                    up_weight_scale_arg = gup_scales_flat.slice(
                        1, prj_cfg.n_H512_tile_sharded, 2 * prj_cfg.n_H512_tile_sharded
                    ).slice(2, cur_I_offset, cur_I_offset + cur_I_tile_sz)

                gate_up_projection_mx_tp(
                    hidden_qtz_sb=buffers.hidden_qtz_sb,
                    hidden_scale_sb=buffers.hidden_scale_sb,
                    weight_qtz=gate_wt_hbm,
                    weight_scale=gate_weight_scale_arg,
                    dst_weight_sb=gate_dst,
                    bias_sb=gate_bias_view,
                    cfg=tile_prj_cfg,
                    skip_dma=kernel_cfg.skip_dma,
                    sbm=sbm,
                    psum_bank_offset=0 if i_tile % 2 == 0 else 1,
                    name_prefix=f"gate_t{i_tile}",
                    out_sb=intermediate_state_tiled[:_pmax, i_tile : i_tile + 1, : dims.B, :_q_width],
                    is_packed_scale=kernel_cfg.use_packed_scales,
                    is_packed_moving_scale=kernel_cfg.is_fp8_hidden,
                    w_dequant_scale=gate_combined_dequant,
                    activation_op=_gate_act_op,
                    is_static_quant=kernel_cfg.is_static_quant,
                    stream_dma_kwargs=_wkw(i_tile),
                )

                gate_up_projection_mx_tp(
                    hidden_qtz_sb=buffers.hidden_qtz_sb,
                    hidden_scale_sb=buffers.hidden_scale_sb,
                    weight_qtz=up_wt_hbm,
                    weight_scale=up_weight_scale_arg,
                    dst_weight_sb=up_dst,
                    bias_sb=up_bias_view,
                    cfg=tile_prj_cfg,
                    skip_dma=kernel_cfg.skip_dma,
                    sbm=sbm,
                    psum_bank_offset=2 if i_tile % 2 == 0 else 3,
                    name_prefix=f"up_t{i_tile}",
                    out_sb=up_tile_sb[:_pmax, 0:1, : dims.B, :_q_width],
                    is_packed_scale=kernel_cfg.use_packed_scales,
                    is_packed_moving_scale=kernel_cfg.is_fp8_hidden,
                    w_dequant_scale=up_combined_dequant,
                    activation_op=None,
                    is_static_quant=kernel_cfg.is_static_quant,
                    stream_dma_kwargs=_wkw(i_tile),
                )
            else:
                # Regime 3 (ping-pong tile): weights already in the SBUF tile buffer —
                # tile 0 from the cross-block prefetch, or any tile when H < 3072.
                cur_wt_flat = gup_wt_bufs[cur_buf].flatten_dims(1, 2)

                if kernel_cfg.is_static_quant:
                    gate_weight_scale_arg = inps.gup_scales_sb
                elif kernel_cfg.use_packed_scales:
                    gate_weight_scale_arg = inps.gup_scales_sb[:, :, 0, cur_I_offset : cur_I_offset + cur_I_tile_sz]
                else:
                    gate_weight_scale_arg = gup_scales_flat.slice(1, 0, prj_cfg.n_H512_tile_sharded).slice(
                        2, cur_I_offset, cur_I_offset + cur_I_tile_sz
                    )
                gate_up_projection_mx_tp(
                    hidden_qtz_sb=buffers.hidden_qtz_sb,
                    hidden_scale_sb=buffers.hidden_scale_sb,
                    weight_qtz=cur_wt_flat.slice(1, 0, prj_cfg.n_H512_tile_sharded).slice(2, 0, cur_I_tile_sz),
                    weight_scale=gate_weight_scale_arg,
                    bias_sb=gate_bias_view,
                    cfg=tile_prj_cfg,
                    sbm=sbm,
                    psum_bank_offset=0 if i_tile % 2 == 0 else 1,
                    name_prefix=f"gate_t{i_tile}",
                    out_sb=intermediate_state_tiled[:_pmax, i_tile : i_tile + 1, : dims.B, :_q_width],
                    is_packed_scale=kernel_cfg.use_packed_scales,
                    is_packed_moving_scale=kernel_cfg.is_fp8_hidden,
                    w_dequant_scale=gate_combined_dequant,
                    activation_op=_gate_act_op,
                    is_static_quant=kernel_cfg.is_static_quant,
                )

                # Prefetch next tile between gate and up (only for standard path, and not a tile that
                # the cross-block prefetch already brought)
                if i_tile < n_I_tiles - 1 and not _use_h_chunked and i_tile + 1 >= _n_pref:
                    nxt_I_offset = (i_tile + 1) * _I_TILE_SZ
                    _load_gup_weight_tile(
                        inps=inps,
                        block_expert=block_expert,
                        prj_cfg=prj_cfg,
                        skip_dma=kernel_cfg.skip_dma,
                        dst_weight=gup_wt_bufs[nxt_buf],
                        dst_scale=None,
                        dst_bias=None,
                        token_indices_on_p=gup_token_indices_on_p,
                        I_offset=nxt_I_offset,
                        dims=dims,
                        sbm=sbm,
                    )

                if kernel_cfg.is_static_quant:
                    up_weight_scale_arg = inps.gup_scales_sb
                elif kernel_cfg.use_packed_scales:
                    up_weight_scale_arg = inps.gup_scales_sb[:, :, 1, cur_I_offset : cur_I_offset + cur_I_tile_sz]
                else:
                    up_weight_scale_arg = gup_scales_flat.slice(
                        1, prj_cfg.n_H512_tile_sharded, 2 * prj_cfg.n_H512_tile_sharded
                    ).slice(2, cur_I_offset, cur_I_offset + cur_I_tile_sz)
                gate_up_projection_mx_tp(
                    hidden_qtz_sb=buffers.hidden_qtz_sb,
                    hidden_scale_sb=buffers.hidden_scale_sb,
                    weight_qtz=cur_wt_flat.slice(1, prj_cfg.n_H512_tile_sharded, 2 * prj_cfg.n_H512_tile_sharded).slice(
                        2, 0, cur_I_tile_sz
                    ),
                    weight_scale=up_weight_scale_arg,
                    bias_sb=up_bias_view,
                    cfg=tile_prj_cfg,
                    sbm=sbm,
                    psum_bank_offset=2 if i_tile % 2 == 0 else 3,
                    name_prefix=f"up_t{i_tile}",
                    out_sb=up_tile_sb[:_pmax, 0:1, : dims.B, :_q_width],
                    is_packed_scale=kernel_cfg.use_packed_scales,
                    is_packed_moving_scale=kernel_cfg.is_fp8_hidden,
                    w_dequant_scale=up_combined_dequant,
                    activation_op=None,
                    is_static_quant=kernel_cfg.is_static_quant,
                )

            # Per-tile: clip, activate gate (in intermediate_state_sb), clip up, then gate * up → intermediate_state_sb
            # (already done inside _gate_up_act_mul_tile on the fused path).
            if not _fuse_gup_epilogue:
                gate_tile = intermediate_state_tiled[:_pmax, i_tile : i_tile + 1, : dims.B, :_q_width]
                up_tile = up_tile_sb[:_pmax, 0:1, : dims.B, :_q_width]

                apply_clamp(gate_tile, kernel_cfg.gate_clamp_upper_limit, kernel_cfg.gate_clamp_lower_limit)
                apply_clamp(up_tile, kernel_cfg.up_clamp_upper_limit, kernel_cfg.up_clamp_lower_limit)
                # STATIC_MX with no gate clamp: silu was already fused into the gate projection above.
                if not (kernel_cfg.is_static_quant and not kernel_cfg.has_gate_clamp):
                    nisa.activation(
                        dst=gate_tile,
                        op=get_nl_act_fn_from_type(kernel_cfg.activation_function),
                        data=gate_tile,
                        scale=1.0,
                        bias=inps.activation_bias,
                    )
                nisa.tensor_tensor(gate_tile, gate_tile, up_tile, op=nl.multiply)

    else:
        # ── Non-tiled path ──
        # Build the flattened (gate/up × n_H512) view via flatten_dims rather than reshape.
        gup_weights_flat = gate_and_up_weights.flatten_dims(1, 2)
        if kernel_cfg.use_packed_scales or kernel_cfg.is_static_quant:
            # STATIC_MX: gate_and_up_scales is the shared [_pmax, _pmax] all-127 dummy;
            # the flatten is unused (call sites pass the buffer directly).
            gup_scales_flat = None
        else:
            gup_scales_flat = gate_and_up_scales.flatten_dims(1, 2)
        gate_bias_view = None
        up_bias_view = None
        if gup_bias:
            gup_bias_flat = gup_bias.flatten_dims(1, 2)
            gate_bias_view = gup_bias_flat.slice(dim=1, start=0, end=prj_cfg.n_total_I512_tile)

        if kernel_cfg.is_static_quant:
            gate_weight_scale_arg = inps.gup_scales_sb
        elif kernel_cfg.use_packed_scales:
            gate_weight_scale_arg = inps.gup_scales_sb[:, :, 0, :]
        else:
            gate_weight_scale_arg = gup_scales_flat.slice(1, 0, prj_cfg.n_H512_tile_sharded)
        # STATIC_MX silu fusion: only when no gate clamp.
        _gate_act_op = (
            get_nl_act_fn_from_type(kernel_cfg.activation_function)
            if kernel_cfg.is_static_quant and not kernel_cfg.has_gate_clamp
            else None
        )
        gate_proj_out_sb = gate_up_projection_mx_tp(
            hidden_qtz_sb=buffers.hidden_qtz_sb,
            hidden_scale_sb=buffers.hidden_scale_sb,
            weight_qtz=gup_weights_flat.slice(1, 0, prj_cfg.n_H512_tile_sharded),
            weight_scale=gate_weight_scale_arg,
            bias_sb=gate_bias_view,
            cfg=prj_cfg,
            sbm=sbm,
            psum_bank_offset=0,
            name_prefix="gate",
            is_packed_scale=kernel_cfg.use_packed_scales,
            is_packed_moving_scale=kernel_cfg.is_fp8_hidden,
            w_dequant_scale=gate_combined_dequant,
            activation_op=_gate_act_op,
            is_static_quant=kernel_cfg.is_static_quant,
        )

        gate_proj_out_sb = gate_proj_out_sb.reshape((_pmax, flatten_free_dim))

        if gup_bias:
            up_bias_view = gup_bias_flat.slice(
                dim=1, start=prj_cfg.n_total_I512_tile, end=2 * prj_cfg.n_total_I512_tile
            )

        if kernel_cfg.is_static_quant:
            up_weight_scale_arg = inps.gup_scales_sb
        elif kernel_cfg.use_packed_scales:
            up_weight_scale_arg = inps.gup_scales_sb[:, :, 1, :]
        else:
            up_weight_scale_arg = gup_scales_flat.slice(1, prj_cfg.n_H512_tile_sharded, 2 * prj_cfg.n_H512_tile_sharded)
        up_proj_out_sb = gate_up_projection_mx_tp(
            hidden_qtz_sb=buffers.hidden_qtz_sb,
            hidden_scale_sb=buffers.hidden_scale_sb,
            weight_qtz=gup_weights_flat.slice(1, prj_cfg.n_H512_tile_sharded, 2 * prj_cfg.n_H512_tile_sharded),
            weight_scale=up_weight_scale_arg,
            bias_sb=up_bias_view,
            cfg=prj_cfg,
            sbm=sbm,
            psum_bank_offset=4,
            name_prefix="up",
            is_packed_scale=kernel_cfg.use_packed_scales,
            is_packed_moving_scale=kernel_cfg.is_fp8_hidden,
            w_dequant_scale=up_combined_dequant,
            activation_op=None,
            is_static_quant=kernel_cfg.is_static_quant,
        )

        up_proj_out_sb = up_proj_out_sb.reshape((_pmax, flatten_free_dim))

    # clipping gate (non-tiled path only; tiled path does this per-tile)
    if not use_tiled_gup:
        apply_clamp(
            gate_proj_out_sb[0:_pmax, 0:flatten_free_dim],
            kernel_cfg.gate_clamp_upper_limit,
            kernel_cfg.gate_clamp_lower_limit,
        )

    # clipping up (non-tiled path only)
    if not use_tiled_gup:
        apply_clamp(
            up_proj_out_sb[0:_pmax, 0:flatten_free_dim],
            kernel_cfg.up_clamp_upper_limit,
            kernel_cfg.up_clamp_lower_limit,
        )

    # bf16 path reshapes the persistent hidden buffers back to the quantize layout [.., B//32, 32]
    # for the next-block online quantize below. fp8 path has no online quantize (gather fills the
    # buffers in-block), so it leaves them in the [.., n_H512_tile, B] matmul layout.
    if not kernel_cfg.is_fp8_hidden:
        buffers.hidden_qtz_sb = buffers.hidden_qtz_sb.reshape((_pmax, prj_cfg.n_H512_tile, dims.B // 32, 32))
        if not kernel_cfg.is_static_quant:
            buffers.hidden_scale_sb = buffers.hidden_scale_sb.reshape((_pmax, prj_cfg.n_H512_tile, dims.B // 32, 32))

    # activation and multiply (non-tiled path only; tiled path does this per-tile).
    # STATIC_MX with no gate clamp: silu was already fused into the gate projection.
    if not use_tiled_gup:
        if not (kernel_cfg.is_static_quant and not kernel_cfg.has_gate_clamp):
            nisa.activation(
                dst=gate_proj_out_sb[0:_pmax, 0:flatten_free_dim],
                op=get_nl_act_fn_from_type(kernel_cfg.activation_function),
                data=gate_proj_out_sb[0:_pmax, 0:flatten_free_dim],
                scale=1.0,
                bias=inps.activation_bias,
            )

        nisa.tensor_tensor(
            intermediate_state_sb[:_pmax, :flatten_free_dim],
            gate_proj_out_sb[:_pmax, :flatten_free_dim],
            up_proj_out_sb[:_pmax, :flatten_free_dim],
            op=nl.multiply,
        )

    if sbm != None:
        sbm.close_scope()  # frees gate/up weights, bias, gate_proj_out_sb, up_proj_out_sb, and projection internals

    intermediate_state_sb = intermediate_state_sb.reshape((_pmax, prj_cfg.n_total_I512_tile, dims.B, _q_width))

    """
    TRANSPOSE AND QUANTIZE NEXT BLOCK HIDDEN STATES
    DMA load was started earlier (during up proj). Now allocate block_hidden_states_T
    and transpose the gathered rows. Deferred to here so nc_transpose doesn't contend
    with up projection on the tensor engine.
    """
    # Hoist next-block expert load: shared by static_mx prefetch quant (recip) and gup-weight prefetch below.
    _pf_expert = None
    if next_block_idx != None:
        _pf_expert = _select_block_expert(routing, next_block_idx, dims, sbm, name="pf_block_expert")

        # fp8 pre-quantized hidden: the next block was gathered (DMA) during gate/up above. Transpose
        # it (PE) into hidden_qtz_sb / hidden_scale_sb now — deferred here so nc_transpose overlaps
        # down projection instead of contending with gate/up. No online quantize.
        if kernel_cfg.is_fp8_hidden:
            transpose_fp8_hidden_states(
                dims,
                prj_cfg,
                buffers.block_hidden_concat,
                hidden_qtz_sb=buffers.hidden_qtz_sb,
                hidden_scale_sb=buffers.hidden_scale_sb,
                sbm=sbm,
            )
        else:
            # PE transpose: the gathered rows are already in block_hidden_states; alloc _T and transpose.
            _alloc_hidden_T_buf(sbm, buffers, dims, prj_cfg, kernel_cfg, tag=f"nb{next_block_idx}_")
            _transpose_hidden_rows(buffers.block_hidden_states, buffers.block_hidden_states_T, dims, prj_cfg, sbm)

        # Dynamic blocks expect their hidden quantized on entry, so quantize right after the
        # transpose when the next block is dynamic.
        if (is_dynamic or next_is_dynamic) and not kernel_cfg.is_fp8_hidden:
            if kernel_cfg.is_static_quant:
                # Reuse hoisted _pf_expert to gather next block's 1/in_scale[expert] from gup_scale_lut_sb.
                _nb_in_quant_recip = _gather_gup_in_quant_recip(inps, _pf_expert, dims, sbm, name_prefix="nb_")
                quantize_block_hidden_state_T_static_mx(buffers, prj_cfg, dims, _nb_in_quant_recip)
            else:
                quantize_block_hidden_state_T(buffers, prj_cfg, dims)
            # Quantized data is in persistent hidden_qtz_sb/hidden_scale_sb;
            # pop blk_hs_T (and blk_hs if present) off heap to reclaim space for down_proj.
            sbm.pop_heap()  # blk_hs_T
            buffers.block_hidden_states_T = None
            sbm.pop_heap()  # blk_hs
            buffers.block_hidden_states = None

    """
    DOWN PROJECTION
    """
    # Tiled path: load down weights here (deferred from before gup to avoid DMA contention)
    if use_tiled_gup:
        down_scale_sb, down_bias_sb = load_down_proj_weights_mx(
            inps,
            weight_expert,  # skip-aware: OOB-skips weight/scale/bias DMAs for same-expert consecutive blocks (down_weight_qtz is persistent)
            buffers.down_weight_qtz,
            dims,
            prj_cfg,
            kernel_cfg.skip_dma,
            gup_token_indices_on_p,
            gup_n_quadrants_needed,
            dst_scale=buffers.down_scale_sb,
            sbm=sbm,
            dst_bias=hoisted_down_bias,
            use_packed_scales=kernel_cfg.use_packed_scales,
            skip_scales=kernel_cfg.is_static_quant,
        )

    # Prefetch next block's gup weights/scales into persistent buffers (overlaps with down proj)
    if next_block_idx != None:
        # Skip-aware expert for the prefetch must be precomputed by the caller.
        # Static and dynamic paths both pass `next_block_expert_for_weights`
        # built from per-region weight-skip masks; the dynamic chunked->rem
        # boundary hoists a 4-op compare in the loop body.
        _pf_skip_expert = next_block_expert_for_weights if kernel_cfg.skip_dma.skip_weight else None

        if use_tiled_gup and buffers.gup_tile_buf_a != None:
            # Full-resident scheme: pass skip-aware expert as `expert` so the WEIGHT
            # DMA OOB-skips on same-expert (preserves the resident buffer across blocks).
            # The default tile-0-only scheme passes the real expert (the buffer is reloaded
            # per-block anyway, so the skip-aware expert only matters for scales).
            if kernel_cfg.gup_full_persistent:
                _pf_weight_expert = _pf_skip_expert if _pf_skip_expert is not None else _pf_expert
            else:
                _pf_weight_expert = _pf_expert
                if _pf_bufs() is not None and (is_dynamic or next_is_dynamic) and kernel_cfg.skip_dma.skip_weight:
                    # (prefetch_tiles) a dynamic next block may not be live (the last iteration's lookahead, or a
                    # static -> dynamic hand-over with no live dynamic block): skip its k tile loads, never read.
                    _live = _pf_expert_live(next_block_idx, dims, sbm, name="pf_live_expert_sel")
                    if _live is not None:
                        _pf_weight_expert = _live
            _prefetch_gup_tile0(
                inps,
                _pf_weight_expert,
                prj_cfg,
                buffers,
                dims,
                kernel_cfg.skip_dma,
                sbm,
                name_prefix=f"pf_b{next_block_idx}",
                scale_expert=_pf_skip_expert,
                use_packed_scales=kernel_cfg.use_packed_scales,
                skip_scales=kernel_cfg.is_static_quant,
                full_I_load=kernel_cfg.gup_full_persistent,
            )
            if not kernel_cfg.gup_full_persistent:
                # (prefetch_tiles) the next block's tiles 1 .. k-1 (no-op when k = 1).
                _prefetch_gup_tiles_rest(inps, _pf_weight_expert, prj_cfg, kernel_cfg.skip_dma, dims, sbm)
        elif not use_tiled_gup and hoisted_gup_weights != None:
            if sbm != None:
                sbm.open_scope(name="pf_gup_full")
            load_gup_weights_scales_mx(
                inps,
                _pf_skip_expert or _pf_expert,
                dims,
                prj_cfg=prj_cfg,
                skip_dma=kernel_cfg.skip_dma,
                sbm=sbm,
                dst_weight=hoisted_gup_weights,
                dst_bias=hoisted_gup_bias,
                name_prefix="pf_gup",
                use_packed_scales=kernel_cfg.use_packed_scales,
                skip_scales=kernel_cfg.is_static_quant,
            )
            if sbm != None:
                sbm.close_scope()

    if sbm != None:
        sbm.open_scope(name="down_proj")

    # If not enough space for dp_out_sb, reuse dead gup weight buffer at same address
    _dp_out_sb = pending_out
    if _dp_out_sb == None and _gup_wt_addr != None and sbm != None:
        n_BxS_tile = dims.B // _pmax
        dp_out_bytes = n_BxS_tile * dims.H * sizeinbytes(nl.bfloat16)
        # check if we can allocate dp_out_sb, otherwise reuse memory location
        if sbm.heap_curr_addr - sbm.stack_curr_addr < dp_out_bytes:
            _dp_out_sb = nl.ndarray(
                (_pmax, n_BxS_tile, dims.H),
                dtype=nl.bfloat16,
                buffer=nl.sbuf,
                name=scoped_name(f"dp_out_sb_reuse_{tag}"),
                address=(0, _gup_wt_addr),
            )

    # MX without down bias: fold the affinity into the PSUM evacuation.
    _fuse_affinity = not kernel_cfg.is_static_quant and down_bias_sb == None
    if _fuse_affinity:
        block_new = _down_projection_affinity_scaled(
            inter_sb=intermediate_state_sb,
            weight=down_weight_qtz_viewed,
            weight_scale=down_scale_sb,
            expert_affinity=expert_affinity,
            cfg=prj_cfg,
            sbm=sbm,
            # Start past the hidden-transpose banks: the fp8 transpose uses banks 0-3, the bf16
            # one banks 4-7.
            psum_bank_offset=4 if kernel_cfg.is_fp8_hidden else 0,
            out_sb=_dp_out_sb,
            is_packed_scale=kernel_cfg.use_packed_scales,
        )
    else:
        block_new = down_projection_mx(
            inter_sb=intermediate_state_sb,
            weight=down_weight_qtz_viewed,
            weight_scale=down_scale_sb,
            bias_sb=down_bias_sb,
            cfg=prj_cfg,
            sbm=sbm,
            # Start past the banks the hidden transpose used: the fp8 transpose occupies
            # banks 0-3, the bf16 transpose banks 4-7.
            psum_bank_offset=4 if kernel_cfg.is_fp8_hidden else 0,
            name_prefix="dp",
            out_sb=_dp_out_sb,
            is_packed_scale=kernel_cfg.use_packed_scales,
            w_dequant_scale=down_combined_dequant_per_block,
            inter_quant_recip=down_in_quant_recip if kernel_cfg.is_static_quant else None,
            dummy_inter_scale=buffers.dummy_inter_scale_sb if kernel_cfg.is_static_quant else None,
        )

    if sbm != None:
        sbm.close_scope()

    for n in range(dims.B // _pmax):
        # LNC1: no dummy-block affinity zeroing -- every block computed here is real.
        if not _fuse_affinity:
            nisa.tensor_scalar(
                dst=block_new[0:_pmax, n, 0 : dims.H],
                data=block_new[0:_pmax, n, 0 : dims.H],
                op0=nl.multiply,
                operand0=expert_affinity[n][0:_pmax, 0:1],
                engine=nisa.scalar_engine,
            )
        if pending_out == None:
            _accumulate_output_rows(block_new, token_indices_2D, n, outs, dims)
    # With pending_out, block_new IS pending_out: the next block (or the final flush) accumulates it,
    # re-deriving the token ids from the resident routing table. Copying the ids into a pending
    # buffer here would wait on the in-flight flush that still reads them.

    if sbm != None:
        sbm.set_name_prefix(prev_prefix)
        sbm.close_scope()


def _alloc_hidden_bufs(sbm, buffers, dims, prj_cfg, configs, tag=""):
    """Heap-allocate hidden state buffers.

    Allocates both block_hidden_states (gathered rows) and block_hidden_states_T (transposed).
    """
    _alloc_hidden_src_buf(sbm, buffers, dims, prj_cfg, configs, tag=tag)
    _alloc_hidden_T_buf(sbm, buffers, dims, prj_cfg, configs, tag=tag)


def _alloc_hidden_src_buf(sbm, buffers, dims, prj_cfg, configs, tag=""):
    """Heap-allocate block_hidden_states (pre-transpose) on demand."""
    prev = sbm.get_name_prefix()
    sbm.set_name_prefix(f"{prev}{tag}")
    # Gathered token rows, [_pmax, n_B128_tiles, H] (one token per partition).
    buffers.block_hidden_states = sbm.alloc_heap(
        (_pmax, dims.n_B128_tiles, dims.H),
        dtype=configs.compute_dtype,
        name="blk_hs",
        align=SBUF_QUADRANT_SIZE,
    )
    sbm.set_name_prefix(prev)


def _alloc_hidden_T_buf(sbm, buffers, dims, prj_cfg, configs, tag=""):
    """Heap-allocate block_hidden_states_T (post-transpose) on demand."""
    prev = sbm.get_name_prefix()
    sbm.set_name_prefix(f"{prev}{tag}")
    buffers.block_hidden_states_T = sbm.alloc_heap(
        (_pmax, prj_cfg.n_H512_tile, dims.B // SBUF_QUADRANT_SIZE, SBUF_QUADRANT_SIZE * _q_width),
        dtype=configs.compute_dtype,
        name="blk_hs_T",
        align=SBUF_QUADRANT_SIZE,
    )
    sbm.set_name_prefix(prev)


def _free_hidden_bufs(sbm, block_hidden_states, block_hidden_states_T=None):
    """Free heap-allocated block_hidden_states_T (and block_hidden_states if present)."""
    if block_hidden_states_T != None:
        sbm.pop_heap()  # block_hidden_states_T (allocated last)
    if block_hidden_states != None:
        sbm.pop_heap()  # block_hidden_states


def process_static_blocks(
    dims: BWMMMXDimensionSizes,
    configs: BWMMMXConfigs,
    prj_cfg: ProjConfig,
    inps: InputTensors,
    outs: OutputTensors,
    buffers: SharedBuffers,
    num_static_blocks: int,
    sbm=None,
    routing=None,
    pending_out=None,
    all_experts_for_weights=None,
    hoisted_gup_weights=None,
    hoisted_gup_bias=None,
    hoisted_down_bias=None,
    is_tensor_update_accumulating=True,
    prefetch_dynamic=False,
    init_output=True,
):
    """
    Process static (non-padded) blocks with prefetching optimization.

    With prefetch_dynamic the last static block prefetches (hidden states, weights) the first
    dynamic block, block num_static_blocks, so the dynamic loop starts without a serial load.

    Iterates through known non-padded blocks with double-buffering to overlap
    computation and data loading.

    Args:
        dims (BWMMMXDimensionSizes): Dimension configuration.
        configs (BWMMMXConfigs): Kernel configuration.
        prj_cfg (ProjConfig): Projection configuration.
        inps (InputTensors): Input tensors.
        outs (OutputTensors): Output tensors.
        buffers (SharedBuffers): Shared buffers.
        num_static_blocks (int): Number of static blocks to process.

    Returns:
        None: Processes blocks and writes to outs.output.

    Notes:
        - LNC1: the one core processes ALL static blocks, in index order, exactly once
        - Prefetches next block while processing current
        - Last block has no prefetch
        - LNC1: no odd/even split and no dummy block -- those exist upstream only to keep
          two cores balanced when the block count doesn't divide evenly

    Pseudocode:
        gather + transpose block 0's hidden rows

        for block_idx in range(num_static_blocks - 1):
            compute_one_block(block_idx, block_idx + 1, ...)
        compute_one_block(num_static_blocks - 1, num_static_blocks if prefetch_dynamic else None, ...)
    """
    # LNC1: one core owns every static block; the first one to prefetch is block 0.
    first_block_idx = 0

    # fp8 pre-quantized hidden gathers + transposes in-block (no bf16 prefetch buffers, no
    # init quantize). Only the persistent hidden_qtz_sb/hidden_scale_sb view shape is set up.
    if not configs.is_fp8_hidden:
        # Heap-allocate hidden state buffers for first block load
        _alloc_hidden_bufs(sbm, buffers, dims, prj_cfg, configs, tag=f"sb{first_block_idx}_")

    if configs.is_fp8_hidden:
        # fp8: persistent hidden buffers stay in the [_pmax, n_H512_tile, B] matmul layout.
        # Prefetch (gather) the first block here; compute_one_block transposes it and prefetches
        # the next block, so the gather DMA always overlaps the prior block's compute.
        buffers.hidden_qtz_sb = buffers.hidden_qtz_sb.reshape((_pmax, prj_cfg.n_H512_tile, dims.B))
        # fp8 scales are packed: n_packed scale blocks (4 H512 tiles each), not n_H512_tile.
        buffers.hidden_scale_sb = buffers.hidden_scale_sb.reshape((_pmax, div_ceil(prj_cfg.n_H512_tile, 4), dims.B))
        # First block: gather + transpose now so hidden_qtz_sb is ready before the loop. Each
        # compute_one_block then gathers the next block (during gate/up) and transposes it (deferred
        # after gate/up), so steady-state gather/transpose always overlap compute.
        _fb_tok = _select_block_token_ids(routing, first_block_idx, dims, sbm, name="fp8_fb_token_ids")
        _load_fp8_rows(
            inps,
            dims,
            configs.skip_dma,
            token_indices_on_p=_fb_tok,
            block_hidden_concat=buffers.block_hidden_concat,
        )
        transpose_fp8_hidden_states(
            dims,
            prj_cfg,
            buffers.block_hidden_concat,
            hidden_qtz_sb=buffers.hidden_qtz_sb,
            hidden_scale_sb=buffers.hidden_scale_sb,
            sbm=sbm,
        )
    else:
        sbm.open_scope(name="init_hidden")
        _fb_token_ids = _select_block_token_ids(routing, first_block_idx, dims, sbm, name="fb_token_ids")
        _load_transpose_hidden_rows(inps, dims, prj_cfg, configs.skip_dma, _fb_token_ids, buffers, sbm)
        sbm.close_scope()

        buffers.hidden_qtz_sb = buffers.hidden_qtz_sb.reshape((_pmax, prj_cfg.n_H512_tile, dims.B // 32, 32))
        if not configs.is_static_quant:
            buffers.hidden_scale_sb = buffers.hidden_scale_sb.reshape((_pmax, prj_cfg.n_H512_tile, dims.B // 32, 32))
        # NOTE: we do not quantize here because we will do it in the beginning of each static block

    # Zero the output after the first block's hidden gather is issued: the 25 MB-class init then
    # overlaps the first block instead of delaying its gather. Only the first accumulate needs it.
    # init_output=False: the caller zeroed the output already.
    if is_tensor_update_accumulating and init_output:
        zeros = sbm.alloc_heap((_pmax, dims.H), dtype=nl.bfloat16, name="output_init_zeros", align=SBUF_QUADRANT_SIZE)
        nisa.memset(zeros, value=0.0)
        output_initialization(outs.output, dims, sbm=sbm, zeros=zeros)
        sbm.pop_heap()  # zeros (hidden bufs stay)

    """
    LNC1: one flat pass over every static block. The shared LNC2 kernel splits here into an
    even-N / odd-N pair of paths so the two cores get equal work (with the odd leftover block
    given to shard 1 for real and to shard 0 as a zero-affinity dummy). With one core there is
    nothing to balance: blocks 0 .. num_static_blocks-2 each prefetch their successor, and the
    final block runs with no prefetch.
    """
    for block_idx in nl.sequential_range(num_static_blocks - 1):
        _block_expert_for_weights = None
        _next_block_expert_for_weights = None
        if all_experts_for_weights != None:
            _block_expert_for_weights = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_copy(
                dst=_block_expert_for_weights,
                src=all_experts_for_weights[0:1, block_idx : block_idx + 1],
            )
            _next_block_expert_for_weights = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_copy(
                dst=_next_block_expert_for_weights,
                src=all_experts_for_weights[0:1, block_idx + 1 : block_idx + 2],
            )

        compute_one_block(
            block_idx,
            block_idx + 1,
            buffers,
            dims,
            inps,
            outs,
            kernel_cfg=configs,
            routing=routing,
            pending_out=pending_out,
            prj_cfg=prj_cfg,
            is_first_block=(block_idx == 0),
            sbm=sbm,
            block_expert_for_weights=_block_expert_for_weights,
            next_block_expert_for_weights=_next_block_expert_for_weights,
            hoisted_gup_weights=hoisted_gup_weights,
            hoisted_gup_bias=hoisted_gup_bias,
            hoisted_down_bias=hoisted_down_bias,
        )

    last_block_idx = num_static_blocks - 1

    _last_expert_for_weights = None
    _next_expert_for_weights = None
    if all_experts_for_weights != None:
        _last_expert_for_weights = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=_last_expert_for_weights,
            src=all_experts_for_weights[0:1, last_block_idx : last_block_idx + 1],
        )
        if prefetch_dynamic:
            _next_expert_for_weights = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_copy(
                dst=_next_expert_for_weights,
                src=all_experts_for_weights[0:1, last_block_idx + 1 : last_block_idx + 2],
            )

    compute_one_block(
        last_block_idx,
        num_static_blocks if prefetch_dynamic else None,
        buffers,
        dims,
        inps,
        outs,
        kernel_cfg=configs,
        routing=routing,
        pending_out=pending_out,
        prj_cfg=prj_cfg,
        is_first_block=(num_static_blocks == 1),
        sbm=sbm,
        block_expert_for_weights=_last_expert_for_weights,
        next_block_expert_for_weights=_next_expert_for_weights,
        hoisted_gup_weights=hoisted_gup_weights,
        hoisted_gup_bias=hoisted_gup_bias,
        hoisted_down_bias=hoisted_down_bias,
        next_is_dynamic=prefetch_dynamic,
    )


def process_dynamic_blocks(
    dims: BWMMMXDimensionSizes,
    configs: BWMMMXConfigs,
    prj_cfg: ProjConfig,
    inps: InputTensors,
    outs: OutputTensors,
    buffers: SharedBuffers,
    num_static_blocks: int,
    num_dynamic_blocks: int,
    sbm=None,
    routing=None,
    pending_out=None,
    hoisted_gup_weights=None,
    hoisted_gup_bias=None,
    hoisted_down_bias=None,
    outer_reg=None,
    rem_reg=None,
    n_outer_iters_sb=None,
    chunked_experts_mask=None,
    rem_experts_mask=None,
    dyn_weight_expert_sb=None,
    chunked_iter_sb=None,
    rem_iter_sb=None,
    n_chunked_alloc: int = 0,
    n_rem_alloc: int = 0,
    first_prefetched: bool = False,
):
    """
    Process dynamic (potentially padded) blocks using condition vector with
    a chunked outer loop + single-block remainder loop.

    LNC1: one core walks every dynamic block, in index order, exactly once.

    Structure:
      - Outer loop (step=_DYN_STEP=_DYN_INNER=8): each iter processes 8 contiguous
          blocks (index+0..+7). index += 8 per outer iter.
      - Remainder loop (step=1): handles the 0.._DYN_STEP-1 active blocks left
          over after chunking, one per iter.

    Iteration counts (computed at caller, passed via outer_reg / rem_reg):
        n_active      = sum(conditions[n_static_blocks : n_static_blocks + num_dynamic_blocks])
        n_outer_iters = n_active // _DYN_STEP
        n_rem_iters   = n_active - n_outer_iters * _DYN_INNER

    Args:
        outer_reg: register holding n_outer_iters (drives chunked outer loop)
        rem_reg:   register holding n_rem_iters (drives the remainder loop)
        n_outer_iters_sb: SBUF scalar holding n_outer_iters
        chunked_experts_mask: weight-skip mask for chunked path (block order)
        rem_experts_mask: weight-skip mask for remainder path (block order)
        dyn_weight_expert_sb: per-block scratch for current weight-expert lookup.
        chunked_iter_sb / rem_iter_sb: mask iter counters.
            Next-block expert is read with the same counter and
            `offset=1` on the .ap() pattern.
        n_chunked_alloc / n_rem_alloc: compile-time mask lengths
    """
    kernel_assert(
        num_static_blocks + num_dynamic_blocks == dims.N,
        f"num_static_blocks + num_dynamic_blocks must equal N, got {num_static_blocks} + {num_dynamic_blocks}!= {dims.N} ",
    )

    logger.debug(f"Start looping over dynamic blocks {num_static_blocks} to {dims.cond_vec_len} - 1")

    # Compile-time flags: does the outer / remainder loop possibly run at runtime?
    _outer_may_run = num_dynamic_blocks >= _DYN_STEP

    INNER = _DYN_INNER  # blocks processed per outer iter (module constant)
    STEP = _DYN_STEP  # blocks consumed per outer iter across all cores; == INNER at LNC1

    # --- First-block prefetch -------------------------------------------------
    # LNC1: the first dynamic block is num_static_blocks regardless of whether the chunked
    # outer loop runs, because both the chunked and the remainder scheme start there. (The
    # shared LNC2 kernel needs a runtime computation here: shard 1's first block is
    # num_static_blocks + INNER when the outer loop runs but num_static_blocks + 1 when it
    # doesn't.) Kept as an SBUF scalar because the loops below index it at runtime.
    logger.debug("Prefetch first dynamic block")
    first_block_idx_sb = _sbm_alloc(sbm, (1, 1), dtype=nl.int32, name="first_block_idx", align=SBUF_QUADRANT_SIZE)
    nisa.memset(dst=first_block_idx_sb, value=num_static_blocks)

    if first_prefetched:
        # The last static block already loaded this block's hidden states and weights.
        pass
    elif configs.is_fp8_hidden:
        # fp8: persistent hidden buffers stay in the [_pmax, n_H512_tile, B] matmul layout.
        # Prefetch (gather) the first dynamic block; compute_one_block transposes it and prefetches
        # the next block so the gather DMA overlaps the prior block's compute.
        buffers.hidden_qtz_sb = buffers.hidden_qtz_sb.reshape((_pmax, prj_cfg.n_H512_tile, dims.B))
        # fp8 scales are packed: n_packed scale blocks (4 H512 tiles each), not n_H512_tile.
        buffers.hidden_scale_sb = buffers.hidden_scale_sb.reshape((_pmax, div_ceil(prj_cfg.n_H512_tile, 4), dims.B))
        # First block: gather + transpose now so hidden_qtz_sb is ready before the loop. Each
        # compute_one_block then gathers the next block (during gate/up) and transposes it (deferred).
        _prev_pfx = sbm.get_name_prefix() if sbm != None else None
        if sbm != None:
            sbm.set_name_prefix(f"{_prev_pfx}fp8pf_dyninit_")
        _fb_tok = _select_block_token_ids(routing, num_static_blocks, dims, sbm, name="fp8_dyn_fb_token_ids")
        _load_fp8_rows(
            inps,
            dims,
            configs.skip_dma,
            token_indices_on_p=_fb_tok,
            block_hidden_concat=buffers.block_hidden_concat,
        )
        if sbm != None:
            sbm.set_name_prefix(_prev_pfx)
        transpose_fp8_hidden_states(
            dims,
            prj_cfg,
            buffers.block_hidden_concat,
            hidden_qtz_sb=buffers.hidden_qtz_sb,
            hidden_scale_sb=buffers.hidden_scale_sb,
            sbm=sbm,
        )
    else:
        # Heap-allocate hidden state buffers for first dynamic block load
        _alloc_hidden_bufs(sbm, buffers, dims, prj_cfg, configs, tag="dyn_init_")

        if sbm != None:
            sbm.open_scope(name="dyn_block_load_hidden_quant")
        # STATIC_MX: gather first block's per-expert recip for the software quant.
        _dyn_init_in_quant_recip = None
        if configs.is_static_quant:
            _dyn_init_expert = _select_block_expert(routing, num_static_blocks, dims, sbm, name="dyn_init_expert")
            _dyn_init_in_quant_recip = _gather_gup_in_quant_recip(
                inps, _dyn_init_expert, dims, sbm, name_prefix="dyn_init_"
            )
        _fb_token_ids = _select_block_token_ids(routing, num_static_blocks, dims, sbm, name="dyn_fb_token_ids")
        _load_transpose_hidden_rows(inps, dims, prj_cfg, configs.skip_dma, _fb_token_ids, buffers, sbm)
        buffers.hidden_qtz_sb = buffers.hidden_qtz_sb.reshape((_pmax, prj_cfg.n_H512_tile, dims.B // 32, 32))
        if configs.is_static_quant:
            quantize_block_hidden_state_T_static_mx(buffers, prj_cfg, dims, _dyn_init_in_quant_recip)
        else:
            buffers.hidden_scale_sb = buffers.hidden_scale_sb.reshape((_pmax, prj_cfg.n_H512_tile, dims.B // 32, 32))
            quantize_block_hidden_state_T(buffers, prj_cfg, dims)
        if sbm != None:
            sbm.close_scope()

    # Prefetch first-dynamic-block's gup weights into persistent buffer.
    # Under full-resident scheme: load the entire gup. Otherwise: just tile 0.
    if buffers.gup_tile_buf_a != None and not first_prefetched:
        sbm.open_scope(name="pf_dyn_init")
        _pf_expert = _select_block_expert(routing, num_static_blocks, dims, sbm, name="pf_dyn_expert")
        _prefetch_gup_tile0(
            inps,
            _pf_expert,
            prj_cfg,
            buffers,
            dims,
            configs.skip_dma,
            sbm,
            name_prefix="pf_dyn_init",
            use_packed_scales=configs.use_packed_scales,
            skip_scales=configs.is_static_quant,
            full_I_load=configs.gup_full_persistent,
        )
        if not configs.gup_full_persistent:
            # (prefetch_tiles) the first dynamic block's tiles 1 .. k-1 (no-op when k = 1).
            _prefetch_gup_tiles_rest(inps, _pf_expert, prj_cfg, configs.skip_dma, dims, sbm)
        sbm.close_scope()

    # For non-tiled dynamic path, allocate persistent gup weight/bias buffers and prefetch
    # the first block's weights so the load can be skipped inside compute_one_block.
    _nontiled_gup_prefetch = dims.I <= _I_TILE_SZ
    if _nontiled_gup_prefetch:
        if hoisted_gup_weights == None:
            hoisted_gup_weights = _sbm_alloc(
                sbm,
                (_pmax, 2, prj_cfg.n_H512_tile_sharded, dims.I),
                dtype=inps.gate_up_proj_weight.dtype,
                name="dyn_hoisted_gup_weights",
                align=SBUF_QUADRANT_SIZE,
            )
        if hoisted_gup_bias == None and inps.gate_and_up_proj_bias:
            hoisted_gup_bias = _sbm_alloc(
                sbm,
                (_pmax, 2, prj_cfg.n_total_I512_tile, _q_width),
                dtype=inps.gate_and_up_proj_bias.dtype,
                name="dyn_hoisted_gup_bias",
                align=SBUF_QUADRANT_SIZE,
            )
            if dims.I < _pmax * _q_width:
                nisa.memset(dst=hoisted_gup_bias[:, :, 0, :], value=0.0)
        if not first_prefetched:
            # Prefetch first dynamic block's full gup weights/scales/bias
            sbm.open_scope(name="pf_dyn_gup_init")
            _pf_expert = _select_block_expert(routing, num_static_blocks, dims, sbm, name="pf_dyn_expert")
            load_gup_weights_scales_mx(
                inps,
                _pf_expert,
                dims,
                prj_cfg=prj_cfg,
                skip_dma=configs.skip_dma,
                sbm=sbm,
                dst_weight=hoisted_gup_weights,
                dst_bias=hoisted_gup_bias,
                use_packed_scales=configs.use_packed_scales,
                skip_scales=configs.is_static_quant,
            )
            sbm.close_scope()

    # Weight-skip flags (compile-time, driven by presence of precomputed masks)
    _use_chunked_skip = chunked_experts_mask != None and n_chunked_alloc > 0
    _use_rem_skip = rem_experts_mask != None and n_rem_alloc > 0

    # Block-idx scratch shared by chunked and rem loops; compute_one_block
    # gets a unique scope prefix from the name_tag passed by each caller.
    dyn_block_idx_sb = _sbm_alloc(
        sbm,
        (1, 1),
        dtype=nl.int32,
        name="dyn_block_idx",
        align=SBUF_QUADRANT_SIZE,
    )
    dyn_next_block_idx_sb = _sbm_alloc(
        sbm,
        (1, 1),
        dtype=nl.int32,
        name="dyn_next_block_idx",
        align=SBUF_QUADRANT_SIZE,
    )

    # --- Chunked outer loop (step=_DYN_STEP=8) ------------------------------
    if _outer_may_run:
        """
        LNC1 needs no "is this the last outer iter?" countdown. At inner == INNER-1 the
        prefetch target is index + STEP whether the next region is another chunk (whose
        first block is index + STEP) or the remainder loop (whose first block is also
        index + STEP), so the two formulas coincide. (Upstream they only coincide for
        shard 0; shard 1 needs a runtime correction on the last outer iter.)
        """
        # Boundary scratch for chunked->rem skip-aware expert compute.
        # boundary_skip_oob_sb holds the constant E (set once); the others are
        # overwritten each boundary iter.
        boundary_skip_oob_sb = None
        boundary_skip_expert_sb = None
        boundary_is_same_sb = None
        if _use_chunked_skip:
            boundary_skip_oob_sb = _sbm_alloc(
                sbm, (1, 1), dtype=nl.int32, name="boundary_skip_oob", align=SBUF_QUADRANT_SIZE
            )
            nisa.memset(dst=boundary_skip_oob_sb, value=_skip_expert(dims.E))
            boundary_skip_expert_sb = _sbm_alloc(
                sbm, (1, 1), dtype=nl.int32, name="boundary_skip_expert", align=SBUF_QUADRANT_SIZE
            )
            boundary_is_same_sb = _sbm_alloc(
                sbm, (1, 1), dtype=nl.uint8, name="boundary_is_same", align=SBUF_QUADRANT_SIZE
            )

        def _process_outer_block(_outer):
            for inner in nl.sequential_range(INNER):
                _bi_sb = dyn_block_idx_sb
                _nbi_sb = dyn_next_block_idx_sb

                # LNC1: block_idx = index + inner (one core owns all INNER blocks of the chunk)
                nisa.tensor_scalar(
                    dst=_bi_sb,
                    data=buffers.index,
                    op0=nl.add,
                    operand0=inner,
                )

                """
                next_block_idx:
                inner 0..INNER-2: block_idx + 1 (within-chunk)
                inner == INNER-1: the block processed NEXT is index + STEP -- the first block
                    of either the next chunk or (on the last outer iter) the remainder loop.
                    At LNC1 those are the same index, so no runtime last-iter correction is
                    needed. Clamped to N-1 for safety.
                """
                if inner < INNER - 1:
                    nisa.tensor_scalar(
                        dst=_nbi_sb,
                        data=_bi_sb,
                        op0=nl.add,
                        operand0=1,
                    )
                else:
                    nisa.tensor_scalar(
                        dst=_nbi_sb,
                        data=buffers.index,
                        op0=nl.add,
                        operand0=STEP,
                        op1=nl.minimum,
                        operand1=dims.N - 1,
                    )

                _dyn_expert_for_weights = None
                _dyn_next_expert_for_weights = None
                if _use_chunked_skip:
                    # chunked_iter advances by 1 per inner block.
                    nisa.tensor_copy(
                        dst=dyn_weight_expert_sb,
                        src=chunked_experts_mask.ap(
                            pattern=[[n_chunked_alloc, 1], [1, 1]],
                            offset=0,
                            scalar_offset=chunked_iter_sb,
                            indirect_dim=1,
                        ),
                    )
                    _dyn_expert_for_weights = dyn_weight_expert_sb
                    if inner < INNER - 1:
                        # Precompute the prefetch's skip-aware expert via
                        # indirect tensor_copy from chunked_experts_mask at
                        # chunked_iter + 1 (compile-time offset).
                        _dyn_next_expert_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
                        nisa.tensor_copy(
                            dst=_dyn_next_expert_sb,
                            src=chunked_experts_mask.ap(
                                pattern=[[n_chunked_alloc, 1], [1, 1]],
                                offset=1,
                                scalar_offset=chunked_iter_sb,
                                indirect_dim=1,
                            ),
                        )
                        _dyn_next_expert_for_weights = _dyn_next_expert_sb
                    else:
                        # Chunked->rem boundary: next block is runtime-variable,
                        # derive skip-aware expert from raw block_to_expert reads.
                        # Load next-expert directly into boundary_skip_expert_sb
                        boundary_curr_expert_sb = _select_block_expert(
                            routing, _bi_sb, dims, sbm, name="boundary_curr_expert"
                        )
                        nisa.tensor_copy(
                            dst=boundary_skip_expert_sb,
                            src=routing.block_expert.ap(
                                pattern=[[dims.N, 1], [1, 1]],
                                offset=0,
                                scalar_offset=_nbi_sb.view(nl.uint32),
                                indirect_dim=1,
                            ),
                        )
                        nisa.tensor_tensor(
                            data1=boundary_skip_expert_sb,
                            data2=boundary_curr_expert_sb,
                            op=nl.equal,
                            dst=boundary_is_same_sb,
                        )
                        nisa.tensor_copy_predicated(
                            dst=boundary_skip_expert_sb,
                            src=boundary_skip_oob_sb,
                            predicate=boundary_is_same_sb,
                        )
                        _dyn_next_expert_for_weights = boundary_skip_expert_sb

                compute_one_block(
                    _bi_sb,
                    _nbi_sb,
                    buffers,
                    dims,
                    inps,
                    outs,
                    kernel_cfg=configs,
                    routing=routing,
                    pending_out=pending_out,
                    prj_cfg=prj_cfg,
                    is_dynamic=True,
                    is_first_block=False,
                    sbm=sbm,
                    block_expert_for_weights=_dyn_expert_for_weights,
                    next_block_expert_for_weights=_dyn_next_expert_for_weights,
                    hoisted_gup_weights=hoisted_gup_weights,
                    hoisted_gup_bias=hoisted_gup_bias,
                    hoisted_down_bias=hoisted_down_bias,
                    name_tag=f"chunk_{inner}",
                )

                if _use_chunked_skip:
                    nisa.tensor_scalar(
                        dst=chunked_iter_sb,
                        data=chunked_iter_sb,
                        op0=nl.add,
                        operand0=1,
                    )

            # End of outer iter: advance index by STEP (8)
            nisa.tensor_scalar(dst=buffers.index, data=buffers.index, op0=nl.add, operand0=STEP)

        nl.fori_loop(0, outer_reg, _process_outer_block)

        # Outer -> Remainder transition: inner=INNER-1 already prefetched index + STEP,
        # which is exactly the remainder loop's first block, so no corrective prefetch here.

    def _process_remainder_block(_rem):
        # LNC1: block_idx = index (one block per remainder iter)
        nisa.tensor_copy(dst=dyn_block_idx_sb, src=buffers.index)
        # next_block_idx = min(block_idx + 1, N - 1)
        nisa.tensor_scalar(
            dst=dyn_next_block_idx_sb,
            data=dyn_block_idx_sb,
            op0=nl.add,
            operand0=1,
            op1=nl.minimum,
            operand1=dims.N - 1,
        )

        _dyn_expert_for_weights = None
        _dyn_next_expert_for_weights = None
        if _use_rem_skip:
            # Mask is sized n_rem_alloc + 1 with the +1 sentinel slot at E,
            # so curr/next lookups use _n_rem_mask_len as the AP pattern bound.
            _n_rem_mask_len = n_rem_alloc + 1
            nisa.tensor_copy(
                dst=dyn_weight_expert_sb,
                src=rem_experts_mask.ap(
                    pattern=[[_n_rem_mask_len, 1], [1, 1]],
                    offset=0,
                    scalar_offset=rem_iter_sb,
                    indirect_dim=1,
                ),
            )
            _dyn_expert_for_weights = dyn_weight_expert_sb
            # Precompute prefetch's skip-aware expert via rem_iter + 1.
            # On the last rem iter, this walks into the sentinel slot (= E)
            # which is functionally a don't-care since the prefetch's output
            # is never consumed (no compute follows the last rem iter).
            _dyn_next_expert_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_copy(
                dst=_dyn_next_expert_sb,
                src=rem_experts_mask.ap(
                    pattern=[[_n_rem_mask_len, 1], [1, 1]],
                    offset=1,
                    scalar_offset=rem_iter_sb,
                    indirect_dim=1,
                ),
            )
            _dyn_next_expert_for_weights = _dyn_next_expert_sb

        compute_one_block(
            dyn_block_idx_sb,
            dyn_next_block_idx_sb,
            buffers,
            dims,
            inps,
            outs,
            kernel_cfg=configs,
            routing=routing,
            pending_out=pending_out,
            prj_cfg=prj_cfg,
            is_dynamic=True,
            is_first_block=False,
            sbm=sbm,
            block_expert_for_weights=_dyn_expert_for_weights,
            next_block_expert_for_weights=_dyn_next_expert_for_weights,
            hoisted_gup_weights=hoisted_gup_weights,
            hoisted_gup_bias=hoisted_gup_bias,
            hoisted_down_bias=hoisted_down_bias,
            name_tag="rem",
        )

        # LNC1: advance index by 1 block; advance rem_iter by 1.
        nisa.tensor_scalar(dst=buffers.index, data=buffers.index, op0=nl.add, operand0=1)
        if _use_rem_skip:
            nisa.tensor_scalar(dst=rem_iter_sb, data=rem_iter_sb, op0=nl.add, operand0=1)

    nl.fori_loop(0, rem_reg, _process_remainder_block)
