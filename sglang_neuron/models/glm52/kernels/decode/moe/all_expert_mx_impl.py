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

"""All-expert MoE token generation implementation with MX (microscaling) quantization.

Every local expert runs its MLP over all ``T`` tokens, and the per-token affinity (zero for tokens not routed to the
expert) scales its contribution.
"""

import os

import nki
import nki.isa as nisa
import nki.language as nl

from ..utils.kernel_assert import kernel_assert
from .all_expert_mx_utils import (
    AllExpertMXDimensions,
    AllExpertMXInputTensors,
    AllExpertMXKernelConfig,
    ExpertWeightsSBUF,
    PrefetchedGateUp,
    init_all_expert_mx_configs,
    validate_all_expert_mx_inputs,
)
from .down_projection_mx import down_projection_mx, load_broadcast_down_weight_scale_bias
from .gate_up_projection_mx import gate_up_projection_mx, load_gate_up_weight_scale_bias
from .mlp_parameters import MLPParameters
from .projection_mx_constants import GATE_FUSED_IDX, MX_SCALE_DTYPE, UP_FUSED_IDX

# Minimum free size of a dynamic-token gate/up MX matmul (see _load_token_count)
_MIN_DYNAMIC_MATMUL_TOKENS = 128

MOE_GU_PREFETCH = int(os.environ.get("GLM52K_MOE_GU_PREFETCH", "0"))
"""The GLM-5.2 MoE layer traces routed expert 0's gate and up weight loads (``prefetch_expert_gate_up``) ahead of
the routed row unpack (1: right before it, after the shared expert; 2: before the shared expert), and ``moe_tkg``
takes them (``prefetched_gate_up``). Only two of the three expert-0 weight tiles find SBUF free before the rows
AllGather ends, and the allocator gives the late slot by instruction order; traced first, G0 and U0 reuse earlier
space and the late slot goes to D0, which is consumed last. Off by default."""

MOE_AFF_SLICE_LATE = os.environ.get("GLM52K_MOE_AFF_SLICE_LATE", "0") == "1"
"""``moe_tkg`` hands the routed affinity slice (rank-id load, expert offset, the indirect slice DMA that reads the
gathered affinity table) to this kernel, which traces it right before expert 0's down projection, its first consumer,
instead of ahead of everything. Traced first, its [1, 1] rank-id / offset tiles die before expert 0's down weights (D0)
are allocated, the allocator recycles their bytes for D0, and D0 (and expert 1's gate weights behind it) waits for
the affinity AllGather. Bit-identical. Off by default."""

_AFF_SLICE = [None]


def set_affinity_slice(fn) -> None:
    """``MOE_AFF_SLICE_LATE``: the next expert kernel traces ``fn()`` (-> masked affinities) before expert 0's down
    projection and uses its result as the masked affinities."""
    _AFF_SLICE[0] = fn


class _LateAffinities:
    """A pending ``MOE_AFF_SLICE_LATE`` slice: ``get()`` traces it once and returns the masked affinities."""

    def __init__(self, fn):
        self._fn, self._value = fn, None

    def get(self):
        if self._value is None:
            self._value = self._fn()
        return self._value


def _take_affinity_slice():
    """The pending ``MOE_AFF_SLICE_LATE`` slice (a ``_LateAffinities``), or None."""
    fn, _AFF_SLICE[0] = _AFF_SLICE[0], None
    return None if fn is None else _LateAffinities(fn)


@nki.jit
def _all_expert_moe_tkg_mx(
    mlp_params: MLPParameters,
    output: nl.NkiTensor,
    token_count: nl.NkiTensor = None,
    prefetched_gate_up=None,
) -> nl.NkiTensor:
    """
    Perform all-expert MoE MLP on input using microscaling format (MX) weights.

    Dimensions:
        T: Total number of input tokens
        H: Hidden dimension size of the model
        I: Intermediate dimension size of the model after tensor parallelism
        E_L: Number of local experts after expert parallelism

    Args:
        mlp_params (MLPParameters): MLPParameters containing all input tensors and configuration.
        output (nl.NkiTensor): [T, H] output tensor in HBM.
        token_count (nl.NkiTensor): Optional [1, 1] int32 in HBM, runtime number of valid tokens
            (1 <= token_count <= T). Only the first token_count tokens are computed by the gate/up
            projections; output rows past it are not meaningful.
        prefetched_gate_up (Optional[PrefetchedGateUp]): One expert's gate/up loads, traced earlier by the
            caller (``prefetch_expert_gate_up``).

    Returns:
        output (nl.NkiTensor): [T, H] in HBM, Output tensor with MoE results.

    Pseudocode:
        # Step 1: Load and quantize input (skipped if hidden_input_scale provided)
        input_quant, input_scale = layout_adapter(input)

        # Step 2: Process each expert sequentially
        for expert_idx in range(E_L):
            # Load expert weights
            gate_w, up_w, down_w = load_one_expert(expert_idx)

            # Compute gate/up projection and activation
            act = gate_up_projection(input_quant, gate_w, up_w)

            # Compute down projection with affinity scaling
            expert_out = down_projection(act, down_w)
            if affinity_scaling_mode == POST_SCALE:
                expert_out *= expert_affinities[expert_idx]

            # Accumulate results
            if expert_idx == 0:
                output = expert_out
            else:
                output += expert_out
    """

    # Initialize configs and validate inputs
    input_tensors, kernel_cfg, dims = init_all_expert_mx_configs(
        mlp_params=mlp_params,
        output=output,
    )
    validate_all_expert_mx_inputs(input_tensors, kernel_cfg, dims)

    _all_expert_mx(
        input_tensors=input_tensors,
        kernel_cfg=kernel_cfg,
        dims=dims,
        token_count=token_count,
        prefetched_gate_up=prefetched_gate_up,
    )

    return output


def _all_expert_mx(
    input_tensors: AllExpertMXInputTensors,
    kernel_cfg: AllExpertMXKernelConfig,
    dims: AllExpertMXDimensions,
    token_count: nl.NkiTensor = None,
    prefetched_gate_up=None,
) -> nl.NkiTensor:
    """
    All-expert MoE computation.

    Processes all experts sequentially, computing MLP(all tokens) for each expert
    before moving to the next.

    Args:
        input_tensors (AllExpertMXInputTensors): Tensor parameters.
        kernel_cfg (AllExpertMXKernelConfig): Scalar parameters.
        dims (AllExpertMXDimensions): Dimension parameters.
        token_count (nl.NkiTensor): Optional [1, 1] int32 runtime token count in HBM.
        prefetched_gate_up (Optional[PrefetchedGateUp]): One expert's gate/up loads traced by the caller.

    Returns:
        nl.NkiTensor: Output tensor with MoE computation results.
    """

    # Step 0: Runtime token counts -> registers driving the gate/up shape registers (one pair per expert)
    n_tokens_lst = [None] * dims.E_L
    n_matmul_tokens_lst = [None] * dims.E_L
    if token_count != None:
        n_tokens, n_matmul_tokens = _load_token_count(token_count, dims.T)
        n_tokens_lst = [n_tokens] * dims.E_L
        n_matmul_tokens_lst = [n_matmul_tokens] * dims.E_L

    # Step 1: Optional load + swizzle + QMX hidden states
    if input_tensors.hidden_input_scale != None:
        # Input is already swizzled + MX quantized upstream (SBUF or pre-quantized HBM)
        if input_tensors.hidden_input.buffer != nl.sbuf:
            # Pre-quantized HBM: DMA copy T_physical to padded SBUF buffer
            n_h512 = input_tensors.hidden_input.shape[1]
            input_quant_sb = nl.ndarray(
                (dims.pmax, n_h512, dims.T), dtype=input_tensors.hidden_input.dtype, buffer=nl.sbuf
            )
            input_scale_sb = nl.ndarray(
                (dims.pmax, n_h512, dims.T),
                dtype=input_tensors.hidden_input_scale.dtype,
                buffer=nl.sbuf,
            )
            nisa.dma_copy(dst=input_quant_sb[:, :, : dims.T_physical], src=input_tensors.hidden_input)
            nisa.dma_copy(dst=input_scale_sb[:, :, : dims.T_physical], src=input_tensors.hidden_input_scale)
        else:
            if dims.T_physical < dims.T:
                n_h512 = input_tensors.hidden_input.shape[1]
                input_quant_sb = nl.ndarray(
                    (dims.pmax, n_h512, dims.T), dtype=input_tensors.hidden_input.dtype, buffer=nl.sbuf
                )
                input_scale_sb = nl.ndarray(
                    (dims.pmax, n_h512, dims.T), dtype=input_tensors.hidden_input_scale.dtype, buffer=nl.sbuf
                )
                nisa.tensor_copy(dst=input_quant_sb[:, :, : dims.T_physical], src=input_tensors.hidden_input)
                nisa.tensor_copy(dst=input_scale_sb[:, :, : dims.T_physical], src=input_tensors.hidden_input_scale)
            else:
                input_quant_sb = input_tensors.hidden_input
                input_scale_sb = input_tensors.hidden_input_scale

        # Zero-pad trailing T rows if T was padded
        if dims.T_physical < dims.T:
            nisa.memset(input_quant_sb[:, :, dims.T_physical :], 0)
            nisa.memset(input_scale_sb[:, :, dims.T_physical :], 0)
    else:
        input_quant_sb, input_scale_sb = _layout_adapter_qmx_hbm(
            input=input_tensors.hidden_input,
            dims=dims,
        )

    # Step 1.1: Zero-pad expert affinities when T was padded up
    if kernel_cfg.unit_affinities:
        expert_affinities_masked_sb = None  # post-scale by an immediate 1.0
    elif dims.T_physical < dims.T:
        affinities_padded = nl.ndarray(
            (dims.T, input_tensors.expert_affinities_masked.shape[1]),
            dtype=input_tensors.expert_affinities_masked.dtype,
            buffer=nl.sbuf,
        )
        nisa.memset(affinities_padded, 0)
        nisa.dma_copy(dst=affinities_padded[: dims.T_physical, :], src=input_tensors.expert_affinities_masked)
        expert_affinities_masked_sb = affinities_padded
    else:
        expert_affinities_masked_sb = input_tensors.expert_affinities_masked
    late_slice = _take_affinity_slice()
    if late_slice is not None:
        # MOE_AFF_SLICE_LATE: resolved (traced) in expert 0's _compute_expert_mlp, right before its down projection
        kernel_assert(
            not kernel_cfg.unit_affinities and dims.T_physical == dims.T,
            "GLM52K_MOE_AFF_SLICE_LATE is for routed affinities and an exact T",
        )
        expert_affinities_masked_sb = late_slice

    # Step 2: Allocate output
    output_shape = (dims.tile_T, dims.n_tiles_in_T, dims.H)
    output_sb = nl.ndarray(output_shape, dtype=nl.bfloat16, buffer=nl.sbuf)

    # Step 3: Compute expert MLPs sequentially
    prev_down_dma = None
    next_gate_up = [prefetched_gate_up]
    for expert_idx in nl.sequential_range(dims.E_L):
        # Step 3.1: Load weights for this expert
        weights = _load_expert(
            input_tensors=input_tensors,
            dims=dims,
            expert_idx=expert_idx,
            prev_down_dma=prev_down_dma,
            prefetched=next_gate_up[0],
        )
        prev_down_dma = weights.down_weight_dma

        # Step 3.2: Compute MLP for this expert
        _compute_expert_mlp(
            input_quant=input_quant_sb,
            input_scale=input_scale_sb,
            weights=weights,
            kernel_cfg=kernel_cfg,
            expert_affinities_masked=expert_affinities_masked_sb,
            output_sb=output_sb[...],
            output_hbm=input_tensors.output,
            expert_idx=expert_idx,
            is_first_expert=(expert_idx == 0),
            is_last_expert=(expert_idx == dims.E_L - 1),
            T_physical=dims.T_physical if dims.T_physical < dims.T else None,
            n_tokens=n_tokens_lst[expert_idx],
            n_matmul_tokens=n_matmul_tokens_lst[expert_idx],
            after_gate_up=(
                (lambda e=expert_idx: next_gate_up.__setitem__(0, _prefetch_next(input_tensors, dims, e + 1)))
                if prefetched_gate_up != None and prefetched_gate_up.next_too and expert_idx + 1 < dims.E_L
                else None
            ),
        )

    return input_tensors.output


def _load_token_count(token_count: nl.NkiTensor, T: int):
    """
    Load the runtime token count into the registers used by the dynamic gate/up projections.

    Args:
        token_count (nl.NkiTensor): [1, 1] int32 in HBM, number of valid tokens (1 <= count <= T).
        T (int): Static token capacity.

    Returns:
        n_tokens (VirtualRegister): Token count, the free size of the PSUM evictions.
        n_matmul_tokens (VirtualRegister): max(token count, min(T, _MIN_DYNAMIC_MATMUL_TOKENS)), the
            free size of the MX matmuls. A narrower MX matmul cannot hide the next LoadStationary
            behind it and costs as much as a 256-token one; the extra PSUM columns are never evicted.
    """
    n_tokens_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=n_tokens_sb, src=token_count)
    n_tokens = nisa.register_alloc()
    nisa.register_load(n_tokens, n_tokens_sb)

    n_matmul_tokens_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=n_matmul_tokens_sb,
        data=n_tokens_sb,
        op0=nl.maximum,
        operand0=min(T, _MIN_DYNAMIC_MATMUL_TOKENS),
    )
    n_matmul_tokens = nisa.register_alloc()
    nisa.register_load(n_matmul_tokens, n_matmul_tokens_sb)
    return n_tokens, n_matmul_tokens


def _layout_adapter_qmx_hbm(
    input: nl.NkiTensor,
    dims: AllExpertMXDimensions,
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """
    Load input from HBM, transform tensor into swizzled layout, and perform quantization to MXFP8.

    Args:
        input (nl.NkiTensor): [T, 4_H * H/512 * 16_H * 8_H], Input tensor in HBM.
        dims (AllExpertMXDimensions): Dimension parameters.

    Returns:
        output_quant_sb (nl.NkiTensor): [16_H * 8_H, H/512, T], Quantized output in SBUF
            (4_H packed in x4 dtype).
        output_scale_sb (nl.NkiTensor): [16_H * 8_H, H/512, T], Scales in SBUF
            (located in leading 4P of each SBUF quadrant).
    """

    # Extract shapes
    n_T32_load_tiles = dims.n_T32_tiles
    T_load = dims.T
    T_x4_load = dims.T32_H4

    # Shapes + allocations
    input_hbm_shape = (n_T32_load_tiles, T_x4_load, dims.n_H512_tiles, dims.tile_H)
    input_sb_shape = (T_x4_load, n_T32_load_tiles, dims.n_H512_tiles, dims.tile_H)
    swizzle_shape = (dims.tile_H, dims.n_H512_tiles, n_T32_load_tiles, T_x4_load)
    out_quantized_shape = (dims.tile_H, dims.n_H512_tiles, T_load)
    input_sb = nl.ndarray(input_sb_shape, dtype=input.dtype, buffer=nl.sbuf)
    input_swizzled_sb = nl.ndarray(swizzle_shape, dtype=input_sb.dtype, buffer=nl.sbuf)
    output_quant_sb = nl.ndarray(out_quantized_shape, dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
    output_scale_sb = nl.ndarray(out_quantized_shape, dtype=MX_SCALE_DTYPE, buffer=nl.sbuf)

    # Reshape input
    input = input.reshape(input_hbm_shape)

    # Load interleaved T * 4_H, then transpose to achieve swizzled layout
    for t32_tile_idx in nl.affine_range(n_T32_load_tiles):
        nisa.dma_copy(
            src=input[t32_tile_idx, :, :, :],
            dst=input_sb[:, t32_tile_idx, :, :],
        )
        for h512_tile_idx in nl.affine_range(dims.n_H512_tiles):
            input_transposed_psum = nl.ndarray((dims.tile_H, T_x4_load), dtype=input_sb.dtype, buffer=nl.psum)
            nisa.nc_transpose(data=input_sb[:, t32_tile_idx, h512_tile_idx, :], dst=input_transposed_psum[...])
            nisa.tensor_copy(src=input_transposed_psum[...], dst=input_swizzled_sb[:, h512_tile_idx, t32_tile_idx, :])

    # Quantize to MXFP8
    nisa.quantize_mx(
        src=input_swizzled_sb,
        dst=output_quant_sb,
        dst_scale=output_scale_sb,
    )

    return output_quant_sb, output_scale_sb


def _load_expert(
    input_tensors: AllExpertMXInputTensors,
    dims: AllExpertMXDimensions,
    expert_idx: int,
    prev_down_dma=None,
    prefetched=None,
) -> ExpertWeightsSBUF:
    """
    Load gate, up, and down projection weight, scale, and bias input_tensors for one expert.

    The down projection bias is broadcast to [tile_T, H].

    Args:
        input_tensors (AllExpertMXInputTensors): Tensor parameters.
        dims (AllExpertMXDimensions): Dimension parameters.
        expert_idx (int): Expert index to load.
        prev_down_dma: The previous expert's down weight DMA, which this expert's gate weight DMA follows.
        prefetched (Optional[PrefetchedGateUp]): Gate/up loads already traced by the caller; used when they are
            this expert's.

    Returns:
        ExpertWeightsSBUF: Expert weights, scales, and biases in SBUF.
    """
    if prefetched != None and prefetched.expert_idx == expert_idx:
        return _load_expert_down(
            input_tensors, dims, expert_idx, prefetched.gate, prefetched.up, prefetched.dmas, prev_down_dma
        )

    dma_handles = []

    # Load gate projection
    gate_weight_sb, gate_weight_scale_sb, gate_bias_sb = load_gate_up_weight_scale_bias(
        weight=input_tensors.gate_up_weights,
        scale=input_tensors.gate_up_weights_scale,
        bias=input_tensors.gate_up_weights_bias,
        expert_idx=expert_idx,
        gate_or_up_idx=GATE_FUSED_IDX,
        H=dims.H,
        n_I512_tiles=dims.n_I512_tiles,
        I=dims.I,
        I_padded=dims.I_padded,
        dma_handles=dma_handles,
    )

    # Load up projection
    up_weight_sb, up_weight_scale_sb, up_bias_sb = load_gate_up_weight_scale_bias(
        weight=input_tensors.gate_up_weights,
        scale=input_tensors.gate_up_weights_scale,
        bias=input_tensors.gate_up_weights_bias,
        expert_idx=expert_idx,
        gate_or_up_idx=UP_FUSED_IDX,
        H=dims.H,
        n_I512_tiles=dims.n_I512_tiles,
        I=dims.I,
        I_padded=dims.I_padded,
        dma_handles=dma_handles,
    )

    return _load_expert_down(
        input_tensors,
        dims,
        expert_idx,
        (gate_weight_sb, gate_weight_scale_sb, gate_bias_sb),
        (up_weight_sb, up_weight_scale_sb, up_bias_sb),
        dma_handles,
        prev_down_dma,
    )


def _load_expert_down(input_tensors, dims, expert_idx, gate, up, dma_handles, prev_down_dma) -> ExpertWeightsSBUF:
    """Load one expert's down projection (weight, scale, broadcast bias) next to its loaded gate/up tiles.

    ``gate`` / ``up`` are (weight, scale, bias) SBUF tiles and ``dma_handles`` their two weight DMAs. The weight DMAs
    are triggered in consumption order on the in-order weight queue, G(e) U(e) D(e) G(e+1): the down weight follows
    the up one and the next expert's gate weight follows this down weight (each edge waits for the producer to land).
    The scheduler otherwise orders G, D, U on some layers and G1 ahead of D0 on others, so a weight needed first
    arrives behind one needed later.
    """
    gate_weight_sb, gate_weight_scale_sb, gate_bias_sb = gate
    up_weight_sb, up_weight_scale_sb, up_bias_sb = up
    dma_handles = list(dma_handles)

    # Load down projection, broadcast down projection bias
    down_weight_sb, down_weight_scale_sb, down_bias_sb = load_broadcast_down_weight_scale_bias(
        weight=input_tensors.down_weights,
        scale=input_tensors.down_weights_scale,
        bias=input_tensors.down_weights_bias,
        expert_idx=expert_idx,
        H=dims.H,
        tile_I=nl.tile_size.pmax,
        n_I512_tiles=dims.n_I512_tiles,
        tile_T=dims.tile_T,
        dma_handles=dma_handles,
    )

    gate_dma, up_dma, down_weight_dma = dma_handles
    down_weight_dma.depends_on(up_dma)
    if prev_down_dma is not None:
        gate_dma.depends_on(prev_down_dma)

    return ExpertWeightsSBUF(
        gate_weight_sb=gate_weight_sb,
        up_weight_sb=up_weight_sb,
        down_weight_sb=down_weight_sb,
        gate_weight_scale_sb=gate_weight_scale_sb,
        up_weight_scale_sb=up_weight_scale_sb,
        down_weight_scale_sb=down_weight_scale_sb,
        gate_bias_sb=gate_bias_sb,
        up_bias_sb=up_bias_sb,
        down_bias_sb=down_bias_sb,
        down_weight_dma=down_weight_dma,
    )


def _prefetch_next(input_tensors: AllExpertMXInputTensors, dims: AllExpertMXDimensions, expert_idx: int):
    """Trace expert ``expert_idx``'s gate/up loads now (a prefetch with ``next_too``)."""
    return prefetch_expert_gate_up(
        input_tensors.gate_up_weights,
        input_tensors.gate_up_weights_scale,
        input_tensors.down_weights,
        T=dims.T_physical,
        gate_up_bias=input_tensors.gate_up_weights_bias,
        expert_idx=expert_idx,
    )


def prefetch_expert_gate_up(
    gate_up_weights: nl.NkiTensor,
    gate_up_weights_scale: nl.NkiTensor,
    down_weights: nl.NkiTensor,
    T: int,
    gate_up_bias: nl.NkiTensor = None,
    expert_idx: int = 0,
    next_too: bool = False,
) -> PrefetchedGateUp:
    """
    Trace one expert's gate and up weight/scale/bias loads now, for ``moe_tkg(prefetched_gate_up=...)``.

    Args:
        gate_up_weights (nl.NkiTensor): [E_L, 128, 2, H/512, I] MX gate/up weights in HBM (as for ``moe_tkg``).
        gate_up_weights_scale (nl.NkiTensor): Their MX micro-scales.
        down_weights (nl.NkiTensor): [E_L, I_p, I/512, H] MX down weights (only H is read).
        T (int): Tokens of the ``moe_tkg`` call.
        gate_up_bias (nl.NkiTensor, optional): Gate/up bias.
        expert_idx (int): Local expert to load.
        next_too (bool): The expert kernel also traces expert e+1's gate/up loads right after expert e's gate/up
            projection, before its down projection, instead of after it.

    Returns:
        PrefetchedGateUp: The SBUF tiles and the weight DMAs.
    """
    E_L, I, H = gate_up_weights.shape[0], gate_up_weights.shape[-1], down_weights.shape[-1]
    dims = AllExpertMXDimensions(T=T, E_L=E_L, I=I, H=H)
    dmas = []
    loads = []
    for gate_or_up_idx in (GATE_FUSED_IDX, UP_FUSED_IDX):
        loads.append(
            load_gate_up_weight_scale_bias(
                weight=gate_up_weights,
                scale=gate_up_weights_scale,
                bias=gate_up_bias,
                expert_idx=expert_idx,
                gate_or_up_idx=gate_or_up_idx,
                H=dims.H,
                n_I512_tiles=dims.n_I512_tiles,
                I=dims.I,
                I_padded=dims.I_padded,
                dma_handles=dmas,
            )
        )
    return PrefetchedGateUp(
        expert_idx=expert_idx,
        gate=loads[0],
        up=loads[1],
        dmas=dmas,
        next_too=next_too,
    )


def _compute_expert_mlp(
    input_quant: nl.NkiTensor,
    input_scale: nl.NkiTensor,
    weights: ExpertWeightsSBUF,
    kernel_cfg: AllExpertMXKernelConfig,
    expert_affinities_masked: nl.NkiTensor,
    output_sb: nl.NkiTensor,
    output_hbm: nl.NkiTensor,
    expert_idx: int,
    is_first_expert: bool,
    is_last_expert: bool,
    T_physical: int = None,
    n_tokens=None,
    n_matmul_tokens=None,
    after_gate_up=None,
) -> nl.NkiTensor:
    """
    Compute expert MLP for one expert over all tokens.

    Args:
        input_quant (nl.NkiTensor): Quantized input tensor.
        input_scale (nl.NkiTensor): Input scale tensor.
        weights (ExpertWeightsSBUF): Expert weights, scales, and biases in SBUF.
        kernel_cfg (AllExpertMXKernelConfig): Kernel config parameters.
        expert_affinities_masked (nl.NkiTensor): Masked expert affinities.
        output_sb (nl.NkiTensor): Output tensor in SBUF.
        output_hbm (nl.NkiTensor): Output tensor in HBM.
        expert_idx (int): Expert index.
        is_first_expert (bool): Whether the current expert is the first expert.
        is_last_expert (bool): Whether the current expert is the last expert.
        T_physical (int): Number of real (unpadded) tokens, when T is padded. None when T is exact.
        n_tokens (Optional[VirtualRegister]): Runtime token count for the gate/up projections.
        n_matmul_tokens (Optional[VirtualRegister]): Free size of the gate/up matmuls, >= n_tokens.
        after_gate_up (Optional[callable]): Traced between the gate/up and the down projection.

    Returns:
        output_sb: Output tensor in SBUF.
    """

    # Step 1: Compute gate/up projection, projection clamping, activation function, and QMX
    act_quant_sb, act_scale_sb = gate_up_projection_mx(
        input_quant_sb=input_quant,
        input_scale_sb=input_scale,
        gate_weight_sb=weights.gate_weight_sb,
        up_weight_sb=weights.up_weight_sb,
        gate_weight_scale_sb=weights.gate_weight_scale_sb,
        up_weight_scale_sb=weights.up_weight_scale_sb,
        gate_bias_sb=weights.gate_bias_sb,
        up_bias_sb=weights.up_bias_sb,
        gate_clamp_upper_limit=kernel_cfg.gate_clamp_upper_limit,
        gate_clamp_lower_limit=kernel_cfg.gate_clamp_lower_limit,
        up_clamp_upper_limit=kernel_cfg.up_clamp_upper_limit,
        up_clamp_lower_limit=kernel_cfg.up_clamp_lower_limit,
        hidden_act_fn=kernel_cfg.hidden_act_fn,
        n_tokens=n_tokens,
        n_matmul_tokens=n_matmul_tokens,
    )
    if after_gate_up != None:
        after_gate_up()
    if isinstance(expert_affinities_masked, _LateAffinities):
        # MOE_AFF_SLICE_LATE: the affinity slice is traced here, after this expert's weight loads and gate/up
        expert_affinities_masked = expert_affinities_masked.get()

    # Step 2: Compute down projection, expert affinity scaling, expert add, and SB->HBM spill
    down_projection_mx(
        act_sb=act_quant_sb[...],
        act_scale_sb=act_scale_sb[...],
        weight_sb=weights.down_weight_sb,
        weight_scale_sb=weights.down_weight_scale_sb,
        bias_sb=weights.down_bias_sb,
        expert_affinities_masked_sb=expert_affinities_masked,
        expert_idx=expert_idx,
        out_sb=output_sb,
        out_hbm=output_hbm,
        expert_affinities_scaling_mode=kernel_cfg.expert_affinities_scaling_mode,
        is_first_expert=is_first_expert,
        is_last_expert=is_last_expert,
        T_physical=T_physical,
    )

    return output_sb
