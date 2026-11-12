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

"""Dense MLP token generation implementation with bf16 weights."""

import nki.isa as nisa
import nki.language as nl

from ...utils.kernel_assert import kernel_assert
from .mlp_parameters_bf16 import MLPParameters
from .mlp_tkg_constants import MLPTKGConstants
from .mlp_tkg_down_projection import process_down_projection
from .mlp_tkg_gate_up_projection import process_gate_up_projection
from .mlp_tkg_utils import input_load


def mlp_tkg_bf16_impl(
    params: MLPParameters,
    output_sb: nl.NkiTensor,
) -> nl.NkiTensor:
    """
    Dense MLP token generation with bf16 weights.

    Computes ``down(act_fn(gate(x)) * up(x))``. There is no normalization, no bias, no clamping
    and no fused residual add: with tensor parallelism the surrounding model applies the norm
    before this kernel and the residual add after the cross-rank reduction, so fusing either in
    here would break the math.

    ``down_proj`` is split along its contraction dimension I, so the result is this rank's PARTIAL
    SUM over its I slice; the reduction across tensor-parallel ranks happens outside this kernel.

    Dimensions:
        T: Total number of input tokens (B * S)
        H: Hidden dimension size
        I: Intermediate dimension size on this tensor-parallel rank

    Args:
        params (MLPParameters): MLP configuration with bf16 weights.
        output_sb (nl.NkiTensor): [T, H] SBUF tile of the output dtype, which the down projection writes directly.

    Returns:
        output_sb (nl.NkiTensor): [T, H], holding this rank's partial sum.

    Pseudocode:
        # Step 1: Load the hidden states into [H0, T, H1]
        input_sb = input_load(hidden)

        # Step 2: Gate and up projections, then the activation and the element-wise product
        inter = transpose(act_fn(gate(input_sb)) * up(input_sb))

        # Step 3: Down projection
        output = down(inter)
    """

    io_dtype = params.hidden_tensor.dtype

    # ---------------- Compute Kernel Dimensions ----------------
    dims = MLPTKGConstants.calculate_constants(params)

    # ---------------- Weight ring, the caller's, shared by all three projections ----------------
    # Fixed [128, slot_elems] slots that gate, up and down stream through in turn, the rotation continued from the
    # caller's previous call, so the next projection's first load is in flight while the previous one's last matmuls
    # run. A gate/up load must hold a multiple of the column-tiling factor of H tiles, so every tile keeps its PSUM
    # row window (k % factor).
    w_dtype = params.down_proj_weights_tensor.dtype
    kernel_assert(
        params.gate_proj_weights_tensor.dtype == w_dtype and params.up_proj_weights_tensor.dtype == w_dtype,
        "gate, up and down weights must share one dtype to share the weight ring, got "
        f"{params.gate_proj_weights_tensor.dtype}, {params.up_proj_weights_tensor.dtype}, {w_dtype}",
    )
    shared_ring = params.weight_ring
    weight_ring = shared_ring.slots
    ring = MLPTKGConstants.calculate_weight_ring(dims, (len(weight_ring), weight_ring[0].shape[1]))
    kernel_assert(
        all(s.dtype == w_dtype and list(s.shape) == [dims._pmax, ring.slot_elems] for s in weight_ring)
        and (ring.gate_up_loads == 1 or ring.gate_up_h1_per_load % dims.column_tiling_factor == 0),
        f"weight_ring must hold [{dims._pmax}, {ring.slot_elems}] {w_dtype} slots whose gate/up loads keep the "
        f"PSUM row windows, got h1={ring.gate_up_h1_per_load} per load at a {dims.column_tiling_factor}-way tile",
    )
    ring_start = shared_ring.position
    weight_dmas = shared_ring.dmas

    # ---------------- Input Load ----------------
    input_sb = input_load(params.hidden_tensor, dims, params.output_dtype)

    # FP8 tensor scales: (gate, up, down) on every token partition, the per-partition scalars the folds take.
    factors_sb = nl.ndarray((dims.T, 3), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=factors_sb, src=params.tensor_scales[0:1, 0:3].broadcast(0, dims.T))

    # ---------- Process gate/up projection, activation, gate/up multiplication ----------
    gate_up_sb = nl.ndarray(
        (dims.I0, dims.I1, dims.T),
        dtype=io_dtype,
        buffer=nl.sbuf,
    )
    gate_up_loads = process_gate_up_projection(
        hidden=input_sb,
        output=gate_up_sb,
        params=params,
        dims=dims,
        weight_ring=weight_ring,
        ring=ring,
        factors_sb=factors_sb,
        ring_start=ring_start,
        weight_dmas=weight_dmas,
    )

    # ---------- Process down projection ----------
    # The SBUF output IS the down projection's tile, so the result never leaves the chip.
    kernel_assert(
        output_sb.dtype == io_dtype,
        f"the SBUF output must be {io_dtype}, the down projection's tile dtype, got {output_sb.dtype}",
    )
    process_down_projection(
        hidden=gate_up_sb,
        output=output_sb,
        params=params,
        dims=dims,
        weight_ring=weight_ring,
        ring=ring,
        factors_sb=factors_sb,
        ring_start=ring_start,
        gate_up_loads=gate_up_loads,
        weight_dmas=weight_dmas,
    )

    return output_sb
