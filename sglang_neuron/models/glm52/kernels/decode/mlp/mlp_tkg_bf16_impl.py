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

from ..utils.kernel_assert import kernel_assert
from .mlp_parameters_bf16 import MLPParameters
from .mlp_tkg_constants import MLPTKGConstants
from .mlp_tkg_down_projection import process_down_projection
from .mlp_tkg_gate_up_projection import process_gate_up_projection
from .mlp_tkg_utils import input_load


def mlp_tkg_bf16_impl(
    params: MLPParameters,
    output_tensor_hbm: nl.NkiTensor,
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
        output_tensor_hbm (nl.NkiTensor): [T, H], Output tensor in shared HBM.

    Returns:
        output_tensor_hbm (nl.NkiTensor): [T, H] in shared HBM, holding this rank's partial sum.

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

    # ---------------- Weight ring, shared by all three projections ----------------
    # One double-buffer of fixed [128, slot_elems] slots that gate, up and down stream through in
    # turn, so every weight DMA is slot-sized (8 MB / 6 MB at GLM-5.2's shape) and the next
    # projection's first load is in flight while the previous one's last matmuls run.
    w_dtype = params.down_proj_weights_tensor.dtype
    kernel_assert(
        params.gate_proj_weights_tensor.dtype == w_dtype and params.up_proj_weights_tensor.dtype == w_dtype,
        "gate, up and down weights must share one dtype to share the weight ring, got "
        f"{params.gate_proj_weights_tensor.dtype}, {params.up_proj_weights_tensor.dtype}, {w_dtype}",
    )
    ring = MLPTKGConstants.calculate_weight_ring(dims, w_dtype)
    weight_ring = []
    for _ in range(ring.depth):
        weight_ring.append(nl.ndarray((dims._pmax, ring.slot_elems), dtype=w_dtype, buffer=nl.sbuf))

    # ---------------- Input Load ----------------
    input_sb = input_load(params.hidden_tensor, dims, params.output_dtype)

    # ---------- Process gate/up projection, activation, gate/up multiplication ----------
    gate_up_sb = nl.ndarray(
        (dims.I0, dims.I1, dims.T),
        dtype=io_dtype,
        buffer=nl.sbuf,
    )
    process_gate_up_projection(
        hidden=input_sb,
        output=gate_up_sb,
        params=params,
        dims=dims,
        weight_ring=weight_ring,
        ring=ring,
    )

    # ---------- Process down projection ----------
    down_sb = nl.ndarray(
        (dims.T, dims.H),
        dtype=io_dtype,
        buffer=nl.sbuf,
    )
    process_down_projection(
        hidden=gate_up_sb,
        output=down_sb,
        params=params,
        dims=dims,
        weight_ring=weight_ring,
        ring=ring,
    )

    # ---------- Return output ----------
    nisa.dma_copy(
        dst=output_tensor_hbm,
        src=down_sb,
    )

    return output_tensor_hbm
