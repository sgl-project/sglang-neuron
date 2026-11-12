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
GLM-5.2 DSA indexer: the column-sharded query up-projection.

The first of four indexer kernels, and the only one before the all-to-all. Column-sharding ``wq_b``
rather than head-sharding it is the whole reason this kernel exists separately: ``q`` is 4096 wide and
only needs GATHERING afterwards, whereas the head axis is REDUCED by the per-head weighted sum, so a
head shard would force an AllReduce of a 32768-wide score instead.

No RoPE here. RoPE rotates the leading 64 of each head's 128, and a column shard need not respect
head boundaries, so it runs after the gather -- exactly as the MLA path defers its norms.
"""

import nki
import nki.isa as nisa
import nki.language as nl

from ..mla.mla_proj_primitives import linear, transpose_into_k_tiles
from .indexer_constants import get_indexer_q_proj_shape


@nki.jit
def indexer_q_proj(
    q_resid: nl.NkiTensor,
    wq_b: nl.NkiTensor,
) -> nl.NkiTensor:
    """Project the normed query latent up to this rank's column slice of the indexer query.

    Args:
        q_resid: [T_q, q_lora_rank] the complete, normed query latent (``mla_decode_q_kv_up``'s ``q_latent_normed``,
            not the down-projection's unnormed shard).
        wq_b: [q_lora_rank, q_shard] this rank's columns of the indexer query projection.

    Returns:
        [T_q, q_shard], a column slice the caller gathers to full width before scoring. No RoPE here: a column shard need
        not fall on head boundaries, so the score kernel rotates after the gather.
    """

    shapes = get_indexer_q_proj_shape(q_resid, wq_b)
    T, q_lora, q_shard = shapes.num_tokens, shapes.q_lora_rank, shapes.q_shard

    q_resid_sbuf = nl.ndarray((nl.tile_size.pmax, q_lora), dtype=shapes.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=q_resid_sbuf[0:T, 0:q_lora], src=q_resid[0:T, 0:q_lora])

    q_resid_tiles_sbuf = transpose_into_k_tiles(q_resid_sbuf[0 : nl.tile_size.pmax, 0:q_lora], T)
    q_shard_sbuf = linear(q_resid_tiles_sbuf, wq_b, T)

    # private_hbm: the layer AllGathers this column shard directly, and at LNC=1 the runtime rejects a
    # collective whose sources are shared_hbm.
    q_out = nl.ndarray((T, q_shard), dtype=shapes.dtype, buffer=nl.private_hbm)
    nisa.dma_copy(dst=q_out[0:T, 0:q_shard], src=q_shard_sbuf[0:T, 0:q_shard])

    return q_out
