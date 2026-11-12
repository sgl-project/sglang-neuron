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

"""Shapes and hardware constants for the GLM-5.2 absorbed-MLA decode attention kernel."""

from dataclasses import dataclass

import nki.language as nl

from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import div_ceil
from .kv_row_layout import LATENT_BYTES, ROPE_BYTES, ROW_BYTES

P_MAX = 128
"""Partition-dim max. An int literal, because nl.tile_size.pmax is not usable in a shape."""

F_MAX = 512
"""Moving-free / PSUM-free max per bank."""

MASK_BOUND = 1.0e30
"""
Magnitude of the additive causal/residency mask. The attention core adds ``keep * MASK_BOUND - MASK_BOUND`` (0 for a
kept key, -MASK_BOUND for a dropped one, in bf16) to every score through the rope matmul's ones row. It survives bf16
rounding and swamps any finite fp32 score, so a dropped key's ``exp`` is exactly 0, yet a rank that keeps no key of a
token still gets a finite max and lse.
"""

NON_RESIDENT_POS = 1 << 30
"""
Key position written for a slot this rank does not hold, or that the selector left empty.

The mask keeps a slot iff ``key_pos <= positions[t]``, so a dropped slot needs a position that fails
the test; -1 would pass it and the slot would be scored. 2**30 is exact in fp32 and leaves int32
headroom after a real position is added to it, so none of its readers (the mask's subtract, the prior
index's position fill and resident keep table) can overflow.
"""


@dataclass
class MlaDecodeShape(nl.NKIObject):
    """
    Compile-time extents for one decode step on one rank. Every dimension is per-rank.

    The token axis is flat: ``num_tokens = num_seqs * spec``, each token with its own absolute
    position and selected-key row, so there is no separate speculation axis.
    """

    num_seqs: int
    """Sequences on this rank."""

    spec: int
    """Speculative tokens per sequence. 1 is plain decode."""

    num_heads: int
    """Query heads on this rank (H / KVDP)."""

    kv_lora_dim: int
    """Latent width C. Also the width of a cache row's kv_latent field."""

    qk_rope_dim: int
    """RoPE width R. Also the width of a cache row's k_pe field."""

    num_prior: int
    """Selected prior keys per token: the global top-k budget, padded to a multiple of P_MAX."""

    num_blocks: int
    """Blocks in this rank's local paged cache."""

    block_size: int
    """Cache rows per block. Must be a power of two; the row resolve shifts and masks by it."""

    max_blocks: int
    """Logical blocks per sequence in this rank's local block table."""

    block_shift: int
    """log2(block_size)."""

    cp_degree: int
    """Context-parallel degree. This rank holds every cp_degree'th logical block."""

    cp_rank: int
    """This rank's index within its CP group, in [0, cp_degree)."""

    dtype: nl.DType
    """Kernel compute dtype."""

    @property
    def num_tokens(self) -> int:
        """Query tokens this rank scores, flat over sequences and speculation."""
        return self.num_seqs * self.spec

    @property
    def num_new_rows(self) -> int:
        """New (this-step) cache rows this rank quantizes and appends: every sibling of every sequence."""
        return self.num_seqs * self.spec

    @property
    def num_keys(self) -> int:
        """
        Key positions scored per token: the selected prior plus this step's ``spec`` sibling keys.

        The siblings are appended in band rather than gathered, so no sibling is read back through
        HBM from a row this kernel just wrote, which avoids a write-then-gather ordering hazard.
        """
        return self.num_prior + self.spec

    @property
    def kv_row_dim(self) -> int:
        """
        Logical element count of one KV vector, cat([k_pe, kv_latent]).

        This is the width of the caller's unquantized bf16 row, not of a packed cache row; use
        ``kv_row_bytes`` for anything that addresses the cache.
        """
        return self.qk_rope_dim + self.kv_lora_dim

    @property
    def kv_row_bytes(self) -> int:
        """Physical width of one packed cache row, in bytes. See kv_row_layout."""
        return ROW_BYTES

    @property
    def rope_bytes(self) -> int:
        """Byte offset at which the packed row's latent field starts."""
        return ROPE_BYTES

    @property
    def num_prior_tiles(self) -> int:
        """Partition tiles the selected prior spans. Key ``t * P_MAX + p`` lives at ``[p, t]``."""
        return div_ceil(self.num_prior, P_MAX)

    @property
    def num_key_tiles(self) -> int:
        """Partition tiles all num_keys span. The sibling block shares the tile after the prior."""
        return div_ceil(self.num_keys, P_MAX)

    @property
    def num_latent_tiles(self) -> int:
        """Partition tiles the latent spans, which is the score matmul's contraction count."""
        return div_ceil(self.kv_lora_dim, P_MAX)

    @property
    def num_cache_rows(self) -> int:
        """Rows in this rank's flattened local cache."""
        return self.num_blocks * self.block_size

    @property
    def oob_row(self) -> int:
        """
        Row index that makes an indirect gather skip a slot.

        The gather's bound is ``src._tensor_shape[indirect_dim]``, exactly ``num_cache_rows`` for the
        flattened cache. The smallest out-of-bounds value keeps the sentinel within the uint16 staging
        of the snake table.
        """
        return self.num_cache_rows


def get_mla_decode_shape(
    kv_cache: nl.NkiTensor,
    block_table: nl.NkiTensor,
    gather_indices: nl.NkiTensor,
    q_absorbed_sb: nl.NkiTensor,
    positions: nl.NkiTensor,
    kv_row_new: nl.NkiTensor,
    kv_lora_rank: int,
    spec: int,
    cp_degree: int,
    cp_rank: int,
) -> MlaDecodeShape:
    """Derive every extent from the operands, so a caller passes only the four true scalars."""

    block_size = kv_cache.shape[2]
    num_tokens = positions.shape[0]
    kernel_assert(
        num_tokens % spec == 0,
        f"num_tokens={num_tokens} must be divisible by spec={spec}",
    )
    num_seqs = num_tokens // spec

    shapes = MlaDecodeShape(
        num_seqs=num_seqs,
        spec=spec,
        num_heads=q_absorbed_sb.shape[3],
        kv_lora_dim=kv_lora_rank,
        # From the logical bf16 row, not the cache: the packed row would give 640 - 512 = 128, which
        # passes the qk_rope_dim assert below and silently builds a wrong kernel. The packed width is
        # checked against ROW_BYTES further down.
        qk_rope_dim=kv_row_new.shape[2] - kv_lora_rank,
        num_prior=gather_indices.shape[2],
        num_blocks=kv_cache.shape[0],
        block_size=block_size,
        max_blocks=block_table.shape[1],
        block_shift=_log2(block_size),
        cp_degree=cp_degree,
        cp_rank=cp_rank,
        dtype=q_absorbed_sb.dtype,
    )

    kernel_assert(
        shapes.num_tokens <= P_MAX,
        f"num_tokens={shapes.num_tokens} must be <= {P_MAX}, one query token per partition",
    )
    kernel_assert(
        shapes.num_heads <= P_MAX,
        f"num_heads={shapes.num_heads} must be <= {P_MAX}, it is a matmul free axis",
    )
    kernel_assert(
        shapes.kv_lora_dim % P_MAX == 0,
        f"kv_lora_dim={shapes.kv_lora_dim} must be a multiple of {P_MAX}",
    )
    kernel_assert(
        0 < shapes.qk_rope_dim <= P_MAX,
        f"qk_rope_dim={shapes.qk_rope_dim}, derived from kv_row_dim={kv_cache.shape[3]} less "
        f"kv_lora_dim={shapes.kv_lora_dim}, must be in (0, {P_MAX}]",
    )
    # A ragged prior would put the sibling block in a tile whose leading partitions are prior keys,
    # scoring its tail against a key_pos the index build never wrote. Whole tiles keep the sibling
    # block on a tile of its own.
    kernel_assert(
        shapes.num_prior % P_MAX == 0,
        f"num_prior={shapes.num_prior} must be a multiple of {P_MAX}; pad the selection instead of shortening it",
    )
    kernel_assert(
        shapes.spec <= P_MAX,
        f"spec={shapes.spec} must be <= {P_MAX}, the sibling block occupies one partition tile",
    )
    kernel_assert(
        shapes.block_size & (shapes.block_size - 1) == 0,
        f"block_size={shapes.block_size} must be a power of two, the row resolve shifts by it",
    )
    kernel_assert(
        shapes.cp_degree & (shapes.cp_degree - 1) == 0,
        f"cp_degree={shapes.cp_degree} must be a power of two, the owner test masks by it",
    )
    kernel_assert(
        0 <= shapes.cp_rank < shapes.cp_degree,
        f"cp_rank={shapes.cp_rank} must be in [0, cp_degree={shapes.cp_degree})",
    )
    # The snake index table is built by a hardware transpose, which needs a 2-byte dtype, so a row
    # index travels as two uint16 halves recombined as hi * 65536 + lo through fp32, which is exact
    # only below 2**24.
    kernel_assert(
        shapes.num_cache_rows < (1 << 24),
        f"num_cache_rows={shapes.num_cache_rows} must be < {1 << 24}: the snake table recombines "
        f"two uint16 halves through an fp32 intermediate, which is exact only below 2**24",
    )
    # The cache must be uint8 packed rows. The dtype check matters: the cache write is a 1-D indirect
    # scatter, which, unlike the batched 2-D form, does not require matching dtypes and silently casts
    # through fp32. An fp8 or bf16 cache would get an unscaled, unclamped cast that first shows up as
    # wrong gathers.
    kernel_assert(
        kv_cache.dtype == nl.uint8,
        f"kv_cache must be uint8 (a packed {ROW_BYTES}-byte row), got {kv_cache.dtype}",
    )
    kernel_assert(
        kv_cache.shape[3] == ROW_BYTES,
        f"kv_cache row must be {ROW_BYTES} packed bytes ({ROPE_BYTES} bf16 k_pe + "
        f"{LATENT_BYTES} fp8 latent), got {kv_cache.shape[3]}",
    )
    kernel_assert(
        kv_row_new.dtype == nl.bfloat16,
        f"kv_row_new is the LOGICAL bf16 row; the kernel quantizes it, got {kv_row_new.dtype}",
    )
    kernel_assert(
        shapes.kv_lora_dim == LATENT_BYTES,
        f"the packed row reserves {LATENT_BYTES} bytes for the latent at one byte per element, so "
        f"kv_lora_dim must be {LATENT_BYTES}, got {shapes.kv_lora_dim}",
    )
    kernel_assert(
        shapes.qk_rope_dim * 2 == ROPE_BYTES,
        f"the packed row reserves {ROPE_BYTES} bytes for bf16 k_pe, so qk_rope_dim must be "
        f"{ROPE_BYTES // 2}, got {shapes.qk_rope_dim}",
    )
    kernel_assert(
        block_table.shape[0] == shapes.num_seqs,
        f"block_table.shape[0]={block_table.shape[0]} must equal num_seqs={shapes.num_seqs}",
    )
    # The reference's own [B, S, topk] shape: selection is per (sequence, speculative token).
    kernel_assert(
        gather_indices.shape[0] == shapes.num_seqs and gather_indices.shape[1] == spec,
        f"gather_indices must be (num_seqs={shapes.num_seqs}, spec={spec}, num_prior), got {gather_indices.shape}",
    )
    return shapes


def _log2(n: int) -> int:
    shift = 0
    while n > 1:
        n >>= 1
        shift += 1
    return shift
