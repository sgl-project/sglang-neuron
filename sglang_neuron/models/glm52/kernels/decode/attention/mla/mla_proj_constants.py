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
Shapes for the three GLM-5.2 MLA decode projection kernels around the attention core.

No shard degree is a parameter. Each kernel reads its shard width off the weight it is given, so a
sharded and an unsharded run are the same kernel, and the caller alone picks the collective that
follows.
"""

from dataclasses import dataclass

import nki.language as nl

from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import div_ceil
from .mla_decode_constants import P_MAX


@dataclass
class DownProjShape(nl.NKIObject):
    """Extents for the two column-sharded down-projections that open the layer."""

    num_tokens: int
    """Query tokens this rank projects. One per partition."""

    hidden: int
    """Model hidden width, the contraction of both down-projections. Not sharded."""

    q_lora_shard: int
    """This rank's slice of the query latent, ``q_lora_rank / degree``."""

    kv_row_shard: int
    """This rank's slice of the cache row, ``kv_row_dim / degree``."""

    dtype: nl.DType
    """Kernel compute dtype."""

    @property
    def num_hidden_tiles(self) -> int:
        """Partition tiles the hidden contraction spans."""
        return div_ceil(self.hidden, P_MAX)


@dataclass
class QKvUpShape(nl.NKIObject):
    """Extents for the post-AllGather query up-projection, RoPE, and W_UK absorb."""

    num_tokens: int
    """Query tokens. One per partition."""

    num_heads: int
    """Query heads this rank attends over (total heads / head shard)."""

    q_lora_rank: int
    """Complete query latent width, after the AllGather. The RMSNorm reduces over all of it."""

    qk_nope_dim: int
    """Per-head QK width RoPE does not rotate. Also the absorb's contraction."""

    qk_rope_dim: int
    """Per-head QK width RoPE rotates. Must be even."""

    kv_lora_dim: int
    """Latent width C, the absorb's output width and a cache row's kv_latent field."""

    dtype: nl.DType
    """Kernel compute dtype."""

    @property
    def qk_head_dim(self) -> int:
        """One head's packed QK width."""
        return self.qk_nope_dim + self.qk_rope_dim

    @property
    def qk_packed_dim(self) -> int:
        """
        Packed QK width of this rank's heads, each head's ``[q_nope | q_pe]`` concatenated.

        Head ``h`` occupies ``[h * qk_head_dim, (h + 1) * qk_head_dim)``. A head shard hands each rank
        whole heads, so the per-head offsets do not change and the shard needs no collective.
        """
        return self.num_heads * self.qk_head_dim

    @property
    def kv_row_dim(self) -> int:
        """One logical cache row, ``cat([k_pe(R), kv_latent(C)])``, unquantized."""
        return self.qk_rope_dim + self.kv_lora_dim

    @property
    def num_latent_tiles(self) -> int:
        """Partition tiles the latent spans. Feature ``c`` lands at ``[c % P_MAX, c // P_MAX]``."""
        return div_ceil(self.kv_lora_dim, P_MAX)

    @property
    def num_q_lora_tiles(self) -> int:
        """Partition tiles the query latent spans: the up-projection's contraction tiles."""
        return div_ceil(self.q_lora_rank, P_MAX)

    @property
    def num_nope_tiles(self) -> int:
        """
        Partition tiles the nope width spans, which the W_UK absorb accumulates over.

        GLM-5.2's ``qk_nope_head_dim`` is 192, so this is 2: the absorb's contraction crosses the
        128-partition boundary and both tiles accumulate into one PSUM region.
        """
        return div_ceil(self.qk_nope_dim, P_MAX)


@dataclass
class OutProjShape(nl.NKIObject):
    """Extents for the W_UV absorb fused with the row-parallel output projection."""

    num_tokens: int
    """Query tokens this rank projects: after the CP ReduceScatter, T / cp_degree of the tokens the core scored."""

    num_heads: int
    """Query heads on this rank. o_proj's contraction is sharded by exactly this."""

    kv_lora_dim: int
    """Latent width C, the W_UV absorb's contraction."""

    v_head_dim: int
    """Per-head value width the absorb emits."""

    hidden_shard: int
    """
    This rank's slice of the output hidden width.

    ``hidden`` for pure row parallelism (the exit is one ReduceScatter), or ``hidden / d`` for the 2-D
    scheme (a ReduceScatter over heads and an AllGather over hidden). The kernel only sees the weight's
    width.
    """

    dtype: nl.DType
    """Kernel compute dtype."""

    @property
    def absorbed_dim(self) -> int:
        """o_proj's contraction on this rank: its own heads' value widths concatenated."""
        return self.num_heads * self.v_head_dim

    @property
    def num_latent_tiles(self) -> int:
        """Partition tiles C spans, which the absorb accumulates over."""
        return div_ceil(self.kv_lora_dim, P_MAX)

    @property
    def num_v_tiles(self) -> int:
        """
        Partition tiles the value width spans.

        GLM-5.2's ``v_head_dim`` is 256, so this is 2: the W_UV absorb's output crosses the 128-partition
        boundary, and since o_proj contracts that output, o_proj walks ``num_heads * num_v_tiles`` tiles.
        """
        return div_ceil(self.v_head_dim, P_MAX)


def get_down_proj_shape(
    hidden_states: nl.NkiTensor,
    wq_a: nl.NkiTensor,
    wkv_a: nl.NkiTensor,
) -> DownProjShape:
    """Derive the down-projection extents. The weight widths imply the column-shard degree."""

    shapes = DownProjShape(
        num_tokens=hidden_states.shape[0],
        hidden=hidden_states.shape[1],
        q_lora_shard=wq_a.shape[1],
        kv_row_shard=wkv_a.shape[1],
        dtype=hidden_states.dtype,
    )

    kernel_assert(
        shapes.num_tokens <= P_MAX,
        f"num_tokens={shapes.num_tokens} must be <= {P_MAX}, one token per partition",
    )
    kernel_assert(
        wq_a.shape[0] == shapes.hidden,
        f"wq_a.shape[0]={wq_a.shape[0]} must equal hidden={shapes.hidden}; the column shard splits "
        f"wq_a's OUTPUT, so its contraction stays whole",
    )
    kernel_assert(
        wkv_a.shape[0] == shapes.hidden,
        f"wkv_a.shape[0]={wkv_a.shape[0]} must equal hidden={shapes.hidden}",
    )
    kernel_assert(shapes.q_lora_shard > 0, f"q_lora_shard={shapes.q_lora_shard} must be positive")
    kernel_assert(shapes.kv_row_shard > 0, f"kv_row_shard={shapes.kv_row_shard} must be positive")
    return shapes


def get_q_kv_up_shape(
    q_latent: nl.NkiTensor,
    kv_row: nl.NkiTensor,
    wq_b: nl.NkiTensor,
    q_absorb_w: nl.NkiTensor,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
) -> QKvUpShape:
    """
    Derive the up-projection extents.

    ``num_heads`` and the per-head widths are arguments because ``wq_b`` flattens every head's
    ``[q_nope | q_pe]`` into one axis, so no operand carries them.
    """

    shapes = QKvUpShape(
        num_tokens=q_latent.shape[0],
        num_heads=num_heads,
        q_lora_rank=q_latent.shape[1],
        qk_nope_dim=qk_nope_head_dim,
        qk_rope_dim=qk_rope_head_dim,
        kv_lora_dim=kv_lora_rank,
        dtype=q_latent.dtype,
    )

    kernel_assert(
        shapes.num_tokens <= P_MAX,
        f"num_tokens={shapes.num_tokens} must be <= {P_MAX}, one token per partition",
    )
    kernel_assert(
        kv_row.shape[0] == shapes.num_tokens,
        f"kv_row.shape[0]={kv_row.shape[0]} must equal num_tokens={shapes.num_tokens}",
    )
    # Both inputs come after the AllGather and must be complete: the norm divides by this width, so a
    # forgotten AllGather would scale it by the shard degree and still give plausible numbers.
    kernel_assert(
        kv_row.shape[1] == shapes.kv_row_dim,
        f"kv_row.shape[1]={kv_row.shape[1]} must equal the COMPLETE kv_row_dim={shapes.kv_row_dim}; "
        f"this kernel runs after the AllGather because RMSNorm reduces over the sharded axis",
    )
    kernel_assert(
        wq_b.shape[0] == shapes.q_lora_rank,
        f"wq_b.shape[0]={wq_b.shape[0]} must equal the COMPLETE q_lora_rank={shapes.q_lora_rank}",
    )
    # A wq_b narrower than this rank's heads is a further column shard the caller gathers. It must hold
    # whole heads (RoPE and the W_UK absorb index per head) and divide qk_packed_dim (the gather
    # concatenates equal blocks).
    kernel_assert(
        shapes.qk_packed_dim % wq_b.shape[1] == 0 and wq_b.shape[1] % shapes.qk_head_dim == 0,
        f"wq_b.shape[1]={wq_b.shape[1]} must be a whole number of heads (qk_head_dim="
        f"{shapes.qk_head_dim}) and must divide qk_packed_dim={shapes.qk_packed_dim} "
        f"(num_heads={shapes.num_heads} x qk_head_dim={shapes.qk_head_dim})",
    )
    # The rank absorbs only the heads whose wq_b columns it holds, and gathers the result.
    degree = shapes.qk_packed_dim // wq_b.shape[1]
    kernel_assert(
        q_absorb_w.shape[0] * degree == shapes.num_heads,
        f"q_absorb_w.shape[0]={q_absorb_w.shape[0]} must equal this rank's wq_b heads: num_heads="
        f"{shapes.num_heads} / wq_b column degree {degree}",
    )
    kernel_assert(
        q_absorb_w.shape[1] == shapes.qk_nope_dim and q_absorb_w.shape[2] == shapes.kv_lora_dim,
        f"q_absorb_w must be (num_heads, qk_nope_dim={shapes.qk_nope_dim}, "
        f"kv_lora_dim={shapes.kv_lora_dim}), got {q_absorb_w.shape}",
    )
    # qk_nope_dim is unbounded because the absorb tiles its contraction (192 does not fit one tile).
    # qk_rope_dim is one transpose's partition dim, so it must fit.
    kernel_assert(
        shapes.qk_rope_dim <= P_MAX,
        f"qk_rope_dim={shapes.qk_rope_dim} must be <= {P_MAX}, it is one transpose's partition dim",
    )
    kernel_assert(
        shapes.qk_rope_dim % 2 == 0,
        f"qk_rope_dim={shapes.qk_rope_dim} must be even; interleaved RoPE pairs 2i with 2i+1",
    )
    kernel_assert(
        shapes.kv_lora_dim % P_MAX == 0,
        f"kv_lora_dim={shapes.kv_lora_dim} must be a multiple of {P_MAX}",
    )
    return shapes


def get_out_proj_shape(
    ctx: nl.NkiTensor,
    out_absorb_w: nl.NkiTensor,
    o_proj_w: nl.NkiTensor,
) -> OutProjShape:
    """Derive the output-projection extents. Every width comes off an operand."""

    shapes = OutProjShape(
        num_tokens=ctx.shape[0],
        num_heads=ctx.shape[1],
        kv_lora_dim=ctx.shape[2],
        v_head_dim=out_absorb_w.shape[2],
        hidden_shard=o_proj_w.shape[1],
        dtype=ctx.dtype,
    )

    kernel_assert(
        shapes.num_tokens <= P_MAX,
        f"num_tokens={shapes.num_tokens} must be <= {P_MAX}, one token per partition",
    )
    # v_head_dim is unbounded because the absorb tiles its output (256 does not fit one tile).
    kernel_assert(
        shapes.kv_lora_dim % P_MAX == 0,
        f"kv_lora_dim={shapes.kv_lora_dim} must be a multiple of {P_MAX}",
    )
    # out_absorb_w is latent-major, [C, num_heads, v_head_dim] (see mla_decode_out_proj).
    kernel_assert(
        out_absorb_w.shape[0] == shapes.kv_lora_dim,
        f"out_absorb_w.shape[0]={out_absorb_w.shape[0]} must equal kv_lora_dim={shapes.kv_lora_dim} "
        f"(latent-major layout [C, num_heads, v_head_dim])",
    )
    kernel_assert(
        out_absorb_w.shape[1] == shapes.num_heads,
        f"out_absorb_w.shape[1]={out_absorb_w.shape[1]} must equal ctx's num_heads={shapes.num_heads}",
    )
    # This pins the scheme: o_proj contracts num_heads * v_head_dim, so it is row-parallel on the head
    # shard and its output is a partial sum the caller reduces. A full-width o_proj fails here.
    kernel_assert(
        o_proj_w.shape[0] == shapes.absorbed_dim,
        f"o_proj_w.shape[0]={o_proj_w.shape[0]} must equal absorbed_dim={shapes.absorbed_dim} "
        f"(num_heads={shapes.num_heads} x v_head_dim={shapes.v_head_dim}); o_proj is row-parallel on "
        f"this rank's heads, so its output is a partial sum the caller reduces",
    )
    return shapes
