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

"""This rank's columns of the expert affinities, for the MTP draft's rows MoE (``moe_layer_bf16.glm52_moe_rows_bf16``)."""

import nki.isa as nisa
import nki.language as nl

from ...utils.kernel_helpers import div_ceil

# Hardware constants
_DGE_ALIGNMENT = 16  # DMA gather engine alignment requirement


def mask_expert_affinities(
    expert_affinities: nl.NkiTensor,
    rank_id: nl.NkiTensor,
    E_L: int,
    T: int,
) -> nl.NkiTensor:
    """
    Slice the global expert affinities down to this rank's local experts.

    The columns this rank owns are [E_L * rank_id, E_L * (rank_id + 1)). `rank_id` is read from
    the device, so one traced kernel serves every rank.

    Args:
        expert_affinities (nl.NkiTensor): [T, E], Global expert affinities in HBM.
        rank_id (nl.NkiTensor): [1, 1] uint32 in HBM, specifies which experts this rank processes.
        E_L (int): Number of local experts.
        T (int): Number of tokens.

    Returns:
        expert_affinities_masked (nl.NkiTensor): [T, E_L]@SBUF or [128_T, ceil(T/128), E_L]@SBUF

    Notes:
        - For T <= 128, returns a 2D tensor [T, E_L]
        - For T > 128, returns a 3D tiled tensor [T_par, n_T128_tiles, E_L]
        - Uses indirect DMA with scalar_offset for rank-based expert slicing
        - The caller's affinity table must already be zero at every expert a token did not select;
          this kernel applies no per-token masking. See the public entry point's docstring.
    """
    # Load rank_id to SBUF
    rank_id_sbuf = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.dma_copy(src=rank_id[0:1, 0:1], dst=rank_id_sbuf[0:1, 0:1])

    # Calculate expert offset: expert_offset = rank_id * E_L
    expert_offset_sbuf = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=expert_offset_sbuf[0:1, 0:1],
        data=rank_id_sbuf[0:1, 0:1],
        op0=nl.multiply,
        operand0=E_L,
    )

    # Slice expert affinities based on rank_id, load to SBUF
    return _load_slice_affinities_sbuf(
        expert_affinities=expert_affinities,
        expert_offset_sbuf=expert_offset_sbuf,
        E_L=E_L,
        T=T,
        io_dtype=expert_affinities.dtype,
    )


def _load_slice_affinities_sbuf(expert_affinities, expert_offset_sbuf, E_L, T, io_dtype):
    """Load and slice expert affinities from HBM to SBUF based on rank offset."""
    E = expert_affinities.shape[1]

    if T <= 128:
        # 2D output [T, E_L]
        expert_affinities_masked = nl.ndarray((T, E_L), dtype=io_dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            src=expert_affinities.ap(
                pattern=[[E, T], [1, E_L]],
                offset=0,
                scalar_offset=expert_offset_sbuf,
                indirect_dim=1,
            ),
            dst=expert_affinities_masked[0:T, 0:E_L],
            dge_mode=nisa.dge_mode.unknown if T % _DGE_ALIGNMENT == 0 else nisa.dge_mode.swdge,
        )
    else:
        # 3D tiled output [T_par, n_T128_tiles, E_L] for T > 128
        T_par = nl.tile_size.pmax
        n_T128_tiles = div_ceil(T, T_par)
        last_tile_T = T % T_par
        has_partial_tile = last_tile_T != 0
        n_full_tiles = n_T128_tiles - 1 if has_partial_tile else n_T128_tiles

        expert_affinities_masked = nl.ndarray((T_par, n_T128_tiles, E_L), dtype=io_dtype, buffer=nl.sbuf)

        if n_full_tiles > 0:
            nisa.dma_copy(
                src=expert_affinities.ap(
                    pattern=[[E, T_par], [T_par * E, n_full_tiles], [1, E_L]],
                    offset=0,
                    scalar_offset=expert_offset_sbuf,
                    indirect_dim=1,
                ),
                dst=expert_affinities_masked[0:T_par, 0:n_full_tiles, 0:E_L],
                dge_mode=nisa.dge_mode.unknown,
            )

        if has_partial_tile:
            partial_tile_offset = n_full_tiles * T_par * E
            nisa.dma_copy(
                src=expert_affinities.ap(
                    pattern=[[E, last_tile_T], [T_par * E, 1], [1, E_L]],
                    offset=partial_tile_offset,
                    scalar_offset=expert_offset_sbuf,
                    indirect_dim=1,
                ),
                dst=expert_affinities_masked[0:last_tile_T, n_full_tiles : n_full_tiles + 1, 0:E_L],
                dge_mode=nisa.dge_mode.unknown if last_tile_T % _DGE_ALIGNMENT == 0 else nisa.dge_mode.swdge,
            )

    return expert_affinities_masked
