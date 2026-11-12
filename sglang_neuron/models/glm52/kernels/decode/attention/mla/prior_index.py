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
Resolve global selected key positions to local paged-cache rows, and gather them.

The selector picks the top-k over the whole context, so the positions arriving here are global, but
under context parallelism a rank holds only every ``cp_degree``'th logical block. Each slot gets:

* a real in-range row (a nearby owned block's for a non-resident slot, a spread row for an empty
  one), so the gather always moves finite bytes, and
* for a slot this rank cannot serve, a sentinel key position, so the core's causal mask drops its
  score to -MASK_BOUND and the fetched row is weighted by exactly zero.

The sentinel makes it correct; the row choice keeps it finite. Skipping the transfer instead would
leave the destination untouched, and the context matmul reads that tile directly, where
``0 * NaN`` is NaN.
"""

import os

import nki.isa as nisa
import nki.language as nl
from nki.isa import dge_mode

from ...utils.kernel_assert import kernel_assert
from ...utils.scratch_hbm import scratch_hbm
from ...utils.trace_names import unique_name
from .mla_decode_constants import NON_RESIDENT_POS, P_MAX, MlaDecodeShape

_BLOCK_TABLE_RESIDENT_BUDGET = 4096
"""Max int32 columns of block table to hold resident per partition, which bounds the seq grouping."""

_HWDGE_2D_PDIM = 16
"""
Partition dim a hardware-DGE dma_transpose requires of a 2-D source, exactly.

nki checks this only when ``dge_mode`` is passed explicitly; otherwise an illegal shape is lowered to
hwdge anyway and the result is silently wrong.
"""

_DGE_DST_ALIGN_BYTES = 32
"""
Destination alignment a DGE crossbar transpose requires per destination group.

nki does not check it, and a violation shows up only as an opaque backend "exit code 70". With an
innermost extent of ``inner`` 2-byte elements, groups start ``inner * 2`` bytes apart, so the rule is
``inner * 2 % 32 == 0``.
"""

_AFFINE_TABLE = os.environ.get("GLM52K_PRIOR_INDEX_AFFINE_TABLE", "")
"""An integer ``k`` declares the block table affine, ``table[s, j] = s * max_blocks + j + k``, as the static identity
table with its CP guard column is (``Glm52GkGeometry.block_table``: column 0 holds -1 and column ``j + 1`` holds ``j``,
so ``k = -1``). The block lookup is then ``block_l + k`` on Vector instead of a broadcast of the whole table and a
GpSimd ``nc_n_gather`` over it, which is slow at long context. Bit-identical with that table only: never set it for
a paged or engine-managed table. Empty (default) runs the general lookup."""

SNAKE_DIRECT = os.environ.get("GLM52K_PRIOR_INDEX_SNAKE_DIRECT", "0") == "1"
"""Build the snake table from the resolved rows as exact fp32 values (every row is below 2**24): one staging copy to
HBM, one contiguous read back as a [128, 128] tile, and one PE transpose (bit-accurate on fp32 values), instead of
two uint16 halves each staged through HBM and DMA-transposed per token. Bit-identical. Needs
``num_tokens * num_prior_tiles == 128``. Off by default."""

KM_SBUF = os.environ.get("GLM52K_PRIOR_INDEX_KM_SBUF", "0") == "1"
"""Read the key-major selection from the SBUF tile the carry-index merge built (``register_positions_km``) instead
of loading back the HBM copy it wrote, which takes an HBM round trip off the selection-to-gather chain.
Bit-identical. Off by default."""

_POSITIONS_KM_SBUF = {}
"""``KM_SBUF`` registry: ``id(positions_km)`` maps to ``(positions_km, its [P_MAX, T * n_tiles] int32 SBUF tile)``."""


def register_positions_km(positions_km: nl.NkiTensor, km_sbuf: nl.NkiTensor) -> None:
    """Record that ``positions_km`` (HBM, ``[P_MAX, T, n_tiles]``) is a copy of ``km_sbuf`` (SBUF), for ``KM_SBUF``."""
    _POSITIONS_KM_SBUF[id(positions_km)] = (positions_km, km_sbuf)


def _positions_km_sbuf(positions_km: nl.NkiTensor):
    """The SBUF tile registered for ``positions_km``, or None."""
    if not KM_SBUF or positions_km is None:
        return None
    entry = _POSITIONS_KM_SBUF.get(id(positions_km))
    if entry is None or entry[0] is not positions_km:
        return None
    return entry[1]


_SPREAD_BIG = 1 << 23
"""Pushes an empty slot's row below every real one in the spread select of ``_resolve_local_rows``. The cache
must hold no more rows than this so ``cache_row - _SPREAD_BIG`` stays exact in fp32."""

# When every cache row fits 16 bits, only the low half of the snake table is built (the high half is all zeros).
# The two int32 adds on the resolve chain (live-slot offset, block lookup) are pinned to Vector, since left to the
# scheduler they land on the slower GpSimd. Exact: every value is an integer below 2**24.


def build_prior_index(
    gather_indices: nl.NkiTensor,
    block_table: nl.NkiTensor,
    mla: MlaDecodeShape,
    pos_offset: nl.NkiTensor = None,
    positions_km: nl.NkiTensor = None,
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """
    Resolve every selected global position to a local cache row, and publish the positions key-major.

    Args:
        gather_indices (nl.NkiTensor): [num_seqs, spec, num_prior] int32 @ HBM, the global positions
            the selector chose for each (sequence, speculative token). Negative means not selected.
        block_table (nl.NkiTensor): [num_seqs, max_blocks] int32 @ HBM, this rank's local table:
            ``[s, i]`` is the physical block holding the sequence's ``i``'th local logical block, or
            negative if unallocated.
        mla (MlaDecodeShape): Shapes, plus the CP degree and this rank's position in its CP group.
        pos_offset (nl.NkiTensor): [1, 1] int32 @ HBM, or None. Added to every non-negative position
            (a -1 slot stays -1); folding the per-rank shift in here keeps a separate pass off the
            selection's critical path.
        positions_km (nl.NkiTensor): [num_seqs * P_MAX, spec * num_prior_tiles] int32 @ HBM, or None.
            The same selection already key-major, as ``indexer_topk_merge_carry_index_packed`` publishes
            it: ``[s * P_MAX + p, t * tiles + i]`` is slot ``i * P_MAX + p`` of token ``(s, t)``. When
            given, positions load from it with one contiguous DMA per partition and ``gather_indices``
            only sizes the shapes. Its slot order may be any fixed permutation, since the tables stay
            consistent with each other and every consumer treats the slots as a set.

    Returns:
        snake_rows (nl.NkiTensor): [P_MAX, num_tokens, num_prior_tiles] uint32 @ SBUF, local cache rows
            permuted into the order a batched indirect gather consumes them.
        key_pos_f (nl.NkiTensor): [P_MAX, num_tokens, num_prior_tiles] fp32 @ SBUF, the absolute
            position of the key the gather places at ``[p, t]``, or NON_RESIDENT_POS, partition-major
            as the core's mask consumes it.
        natural_rows (nl.NkiTensor): [P_MAX, num_tokens, num_prior_tiles] uint32 @ SBUF, the same rows
            in identity key order: ``[p, tok, i]`` is the row of key ``i * P_MAX + p``.
    """
    num_prior = mla.num_prior
    num_seqs, spec = mla.num_seqs, mla.spec
    num_tokens = mla.num_tokens
    n_tiles = mla.num_prior_tiles
    # Each pass holds seq_group sequences' block tables on every partition. The group counts whole
    # sequences because a table row is per sequence and its iota can ramp but not divide, so a
    # sequence's speculative tokens resolve in the same pass.
    seq_group = max(1, min(num_seqs, _BLOCK_TABLE_RESIDENT_BUDGET // mla.max_blocks))

    rows_i32 = nl.ndarray((P_MAX, num_seqs, spec, n_tiles), dtype=nl.int32, buffer=nl.sbuf)
    key_pos_f = nl.ndarray((P_MAX, num_seqs, spec, n_tiles), dtype=nl.float32, buffer=nl.sbuf)
    for seq_start in range(0, num_seqs, seq_group):
        seq_count = min(seq_group, num_seqs - seq_start)
        group_rows, group_pos = _resolve_local_rows(
            gather_indices, block_table, mla, seq_start, seq_count, pos_offset, positions_km
        )
        nisa.tensor_copy(
            dst=rows_i32[0:P_MAX, seq_start : seq_start + seq_count, 0:spec, 0:n_tiles],
            src=group_rows[0:P_MAX, 0:seq_count, 0:spec, 0:n_tiles],
        )
        nisa.tensor_copy(
            dst=key_pos_f[0:P_MAX, seq_start : seq_start + seq_count, 0:spec, 0:n_tiles],
            src=group_pos[0:P_MAX, 0:seq_count, 0:spec, 0:n_tiles],
        )
    # (seq, spec) is the flat query-token axis from here on; the core indexes by token.
    rows_i32 = rows_i32.reshape((P_MAX, num_tokens, n_tiles))
    key_pos_f = key_pos_f.reshape((P_MAX, num_tokens, n_tiles))

    # The snake table, the batched gather's index. The hardware flattens the gather's 2-D vector_offset
    # column-major while it walks the SBUF destination row-major, so the table is pre-permuted:
    # snake[p, t, i] == rows_flat[t * P_MAX * n_tiles + i * P_MAX + p], the transpose of each token's
    # [n_tiles, P_MAX] index block.
    #
    # dma_transpose needs a 2-byte dtype, and a local row index can exceed 16 bits at long context, so
    # the general path transposes the low and high uint16 halves separately and recombines them. A
    # negative index cannot be left to wrap: a dtype-changing tensor_copy saturates through fp32, so -1
    # would become row 0 and gather live KV. Every sentinel is an explicit positive value.
    block = P_MAX * n_tiles
    snake_rows = nl.ndarray((P_MAX, num_tokens, n_tiles), dtype=nl.uint32, buffer=nl.sbuf)

    # Two DGE rules must hold, and n_tiles == 16 satisfies both:
    #   (a) a hwdge 2-D source needs src.shape[0] == 16, so n_tiles == 16;
    #   (b) each sequence's destination slice starts at s * n_tiles * 2 bytes, which must be 32-byte
    #       aligned, so n_tiles % 16 == 0.
    # nki never checks (b), so it is asserted on its own in case (a) is relaxed later.
    if SNAKE_DIRECT and num_tokens * n_tiles == P_MAX and P_MAX % n_tiles == 0 and mla.num_cache_rows < (1 << 24):
        _snake_direct(rows_i32, snake_rows, num_tokens, n_tiles)
    elif n_tiles == _HWDGE_2D_PDIM:
        kernel_assert(
            (n_tiles * 2) % _DGE_DST_ALIGN_BYTES == 0,
            f"DGE transpose dst must be {_DGE_DST_ALIGN_BYTES}-byte aligned, but the per-sequence "
            f"stride is {n_tiles * 2} B (n_tiles={n_tiles} uint16)",
        )
        half_sb = []
        low_only = mla.num_cache_rows <= (1 << 16)
        n_halves = 1 if low_only else 2
        for half in range(n_halves):
            # Low half = row & 0xFFFF, high half = row >> 16. Both are non-negative, since every row is
            # already in range.
            if low_only:
                part_i32 = rows_i32  # already < 2**16: the mask is the identity
            else:
                part_i32 = nl.ndarray((P_MAX, num_tokens, n_tiles), dtype=nl.int32, buffer=nl.sbuf)
                if half == 0:
                    nisa.tensor_scalar(
                        part_i32[0:P_MAX, 0:num_tokens, 0:n_tiles],
                        rows_i32[0:P_MAX, 0:num_tokens, 0:n_tiles],
                        op0=nl.bitwise_and,
                        operand0=0xFFFF,
                    )
                else:
                    nisa.tensor_scalar(
                        part_i32[0:P_MAX, 0:num_tokens, 0:n_tiles],
                        rows_i32[0:P_MAX, 0:num_tokens, 0:n_tiles],
                        op0=nl.right_shift,
                        operand0=16,
                    )
            part_u16 = nl.ndarray((P_MAX, num_tokens, n_tiles), dtype=nl.uint16, buffer=nl.sbuf)
            nisa.tensor_copy(
                dst=part_u16[0:P_MAX, 0:num_tokens, 0:n_tiles],
                src=part_i32[0:P_MAX, 0:num_tokens, 0:n_tiles],
            )
            snake_hbm = nl.ndarray((num_tokens, n_tiles, P_MAX), dtype=nl.uint16, buffer=nl.private_hbm)
            nisa.dma_copy(
                dst=snake_hbm.reshape((num_tokens * block,)).ap(
                    pattern=[[n_tiles, P_MAX], [block, num_tokens], [1, n_tiles]], offset=0
                ),
                src=part_u16[0:P_MAX, 0:num_tokens, 0:n_tiles],
            )
            out_u16 = nl.ndarray((P_MAX, num_tokens, n_tiles), dtype=nl.uint16, buffer=nl.sbuf)
            for tok in range(num_tokens):
                nisa.dma_transpose(
                    dst=out_u16[0:P_MAX, tok, 0:n_tiles],
                    src=snake_hbm[tok, 0:n_tiles, 0:P_MAX],
                    dge_mode=dge_mode.hwdge,
                )
            out_i32 = nl.ndarray((P_MAX, num_tokens, n_tiles), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_copy(
                dst=out_i32[0:P_MAX, 0:num_tokens, 0:n_tiles],
                src=out_u16[0:P_MAX, 0:num_tokens, 0:n_tiles],
            )
            half_sb.append(out_i32)
        # Recombine: row = high * 65536 + low.
        if low_only:
            combined = half_sb[0]
        else:
            combined = nl.ndarray((P_MAX, num_tokens, n_tiles), dtype=nl.int32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(
                dst=combined[0:P_MAX, 0:num_tokens, 0:n_tiles],
                data=half_sb[1][0:P_MAX, 0:num_tokens, 0:n_tiles],
                op0=nl.multiply,
                operand0=float(1 << 16),
                op1=nl.add,
                operand1=half_sb[0][0:P_MAX, 0:num_tokens, 0:n_tiles],
            )
        nisa.tensor_copy(
            dst=snake_rows[0:P_MAX, 0:num_tokens, 0:n_tiles],
            src=combined[0:P_MAX, 0:num_tokens, 0:n_tiles],
        )
    else:
        # Fallback for any n_tiles: stage int32 through HBM and read the permutation back, at the
        # cost of a read-back that scatters 4 bytes per partition.
        snake_hbm_i32 = nl.ndarray((num_tokens * block,), dtype=nl.int32, buffer=nl.private_hbm)
        nisa.dma_copy(
            dst=snake_hbm_i32.ap(pattern=[[n_tiles, P_MAX], [block, num_tokens], [1, n_tiles]], offset=0),
            src=rows_i32[0:P_MAX, 0:num_tokens, 0:n_tiles],
        )
        snake_i32 = nl.ndarray((P_MAX, num_tokens, n_tiles), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=snake_i32[0:P_MAX, 0:num_tokens, 0:n_tiles],
            src=snake_hbm_i32.ap(pattern=[[1, P_MAX], [block, num_tokens], [P_MAX, n_tiles]], offset=0),
        )
        nisa.tensor_copy(
            dst=snake_rows[0:P_MAX, 0:num_tokens, 0:n_tiles],
            src=snake_i32[0:P_MAX, 0:num_tokens, 0:n_tiles],
        )

    kernel_assert(num_prior == n_tiles * P_MAX, "num_prior must fill whole partition tiles")
    # The natural-order table: [p, tok, i] is the local row of key i * P_MAX + p. The snake order only
    # suits a batched 2-D vector_offset; other consumers (the resident index) want this one. Given the
    # snake table they would permute the key axis, which the position mask is not invariant to.
    natural_rows = nl.ndarray((P_MAX, num_tokens, n_tiles), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_copy(
        dst=natural_rows[0:P_MAX, 0:num_tokens, 0:n_tiles],
        src=rows_i32[0:P_MAX, 0:num_tokens, 0:n_tiles],
    )

    return snake_rows, key_pos_f, natural_rows


def _snake_direct(rows_i32: nl.NkiTensor, snake_rows: nl.NkiTensor, num_tokens: int, n_tiles: int) -> None:
    """
    The snake table of ``build_prior_index`` with one staging copy and one PE transpose (``SNAKE_DIRECT``).

    ``snake[p, t, i] = rows[g * i + p // n_tiles, t, p % n_tiles]`` with ``g = P_MAX // n_tiles``, the transpose of each
    token's flat ``[n_tiles, P_MAX]`` index block. The rows go to HBM as exact fp32 values (below 2**24) in the general
    path's staging layout and come back as one ``[P_MAX, P_MAX]`` tile whose partition ``t * n_tiles + r`` is token
    ``t``'s block row ``r``. Its PE transpose, bit-accurate on fp32 values, is ``snake`` with ``(t, i)`` on the free axis.
    """
    T = num_tokens
    block = P_MAX * n_tiles
    rows_f = nl.ndarray((P_MAX, T, n_tiles), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=rows_f[0:P_MAX, 0:T, 0:n_tiles], src=rows_i32[0:P_MAX, 0:T, 0:n_tiles], engine=nisa.vector_engine)
    stage = nl.ndarray((T * block,), dtype=nl.float32, buffer=nl.private_hbm, name=unique_name("snake_stage"))
    nisa.dma_copy(
        dst=stage.ap(pattern=[[n_tiles, P_MAX], [block, T], [1, n_tiles]], offset=0),
        src=rows_f[0:P_MAX, 0:T, 0:n_tiles],
    )
    x = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=x[0:P_MAX, 0:P_MAX], src=stage.reshape((P_MAX, P_MAX))[0:P_MAX, 0:P_MAX])
    s_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=s_psum[0:P_MAX, 0:P_MAX], data=x[0:P_MAX, 0:P_MAX], engine=nisa.engine.tensor)
    nisa.tensor_copy(
        dst=snake_rows.reshape((P_MAX, T * n_tiles))[0:P_MAX, 0 : T * n_tiles],
        src=s_psum[0:P_MAX, 0:P_MAX],
        engine=nisa.vector_engine,
    )


def export_prior_index(
    snake_rows: nl.NkiTensor,
    key_pos_f: nl.NkiTensor,
    natural_rows: nl.NkiTensor,
    name: str = "prior_index",
) -> tuple[nl.NkiTensor, nl.NkiTensor, nl.NkiTensor]:
    """
    Stage the three index tables to HBM so they can be handed to a later ``mla_decode_attention``.

    The tables depend only on the selection, the block table and the CP shift, not on the layer's
    cache, so every layer of an index-share group has the same tables. Importing them costs three
    small DMAs per consumer instead of the whole resolve-and-snake chain.

    Args:
        snake_rows (nl.NkiTensor): [P_MAX, num_tokens, num_prior_tiles] uint32 @ SBUF.
        key_pos_f (nl.NkiTensor): [P_MAX, num_tokens, num_prior_tiles] fp32 @ SBUF.
        natural_rows (nl.NkiTensor): [P_MAX, num_tokens, num_prior_tiles] uint32 @ SBUF.
        name (str): Debug-name stem for the HBM tensors.

    Returns:
        (snake_hbm, key_pos_hbm, natural_hbm), the same three tables in ``shared_hbm``.
    """
    shape = snake_rows.shape
    out = []
    for sb, dtype, tag in (
        (snake_rows, nl.uint32, "snake"),
        (key_pos_f, nl.float32, "keypos"),
        (natural_rows, nl.uint32, "natural"),
    ):
        hbm = nl.ndarray(shape, dtype=dtype, buffer=scratch_hbm(), name=unique_name(f"{name}_{tag}"))
        nisa.dma_copy(dst=hbm, src=sb)
        out.append(hbm)
    return out[0], out[1], out[2]


def import_prior_index(
    snake_hbm: nl.NkiTensor,
    key_pos_hbm: nl.NkiTensor,
    natural_hbm: nl.NkiTensor,
    mla: MlaDecodeShape,
    need_natural: bool,
    **dma_kw,
) -> tuple[nl.NkiTensor, nl.NkiTensor, nl.NkiTensor]:
    """
    Load prebuilt index tables from HBM into the SBUF layout ``build_prior_index`` would have produced.

    Args:
        snake_hbm (nl.NkiTensor): [P_MAX, num_tokens, num_prior_tiles] uint32 @ HBM.
        key_pos_hbm (nl.NkiTensor): [P_MAX, num_tokens, num_prior_tiles] fp32 @ HBM.
        natural_hbm (nl.NkiTensor): [P_MAX, num_tokens, num_prior_tiles] uint32 @ HBM, or None when
            ``need_natural`` is False.
        mla (MlaDecodeShape): Shapes, for the assert.
        need_natural (bool): Whether the caller reads ``natural_rows``: the resident core's index build
            (``build_resident_index``) does. When False that table is not loaded and None is returned.
        dma_kw: Extra ``nisa.dma_copy`` arguments of the three loads (a DGE mode inside a dynamic loop).

    Returns:
        (snake_rows, key_pos_f, natural_rows) in SBUF; ``natural_rows`` is None if not needed.
    """
    shape = (P_MAX, mla.num_tokens, mla.num_prior_tiles)
    kernel_assert(
        tuple(snake_hbm.shape) == shape and tuple(key_pos_hbm.shape) == shape,
        f"prebuilt prior-index tables must be {shape}, got {snake_hbm.shape} / {key_pos_hbm.shape}",
    )
    snake_rows = nl.ndarray(shape, dtype=nl.uint32, buffer=nl.sbuf)
    nisa.dma_copy(dst=snake_rows, src=snake_hbm, **dma_kw)
    key_pos_f = nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=key_pos_f, src=key_pos_hbm, **dma_kw)
    natural_rows = None
    if need_natural:
        kernel_assert(
            natural_hbm is not None and tuple(natural_hbm.shape) == shape,
            f"the resident index needs the natural-order table, shape {shape}",
        )
        natural_rows = nl.ndarray(shape, dtype=nl.uint32, buffer=nl.sbuf)
        nisa.dma_copy(dst=natural_rows, src=natural_hbm, **dma_kw)
    return snake_rows, key_pos_f, natural_rows


def gather_prior_rows(
    rows_sb: nl.NkiTensor,
    slot: int,
    kv_cache: nl.NkiTensor,
    snake_rows: nl.NkiTensor,
    tok: int,
    mla: MlaDecodeShape,
) -> nisa.NkiInstruction:
    """
    Gather one query token's selected prior rows into a caller-owned slot of ``rows_sb``, with one
    batched indirect DMA for all num_prior rows (per-tile DMAs were descriptor-bound).

    ``snake_rows`` must be the snake-permuted table, which cancels the mismatch between the column-major
    flatten of the 2-D vector_offset and the row-major walk of the destination. The natural-order table
    would land key ``p * n_tiles + t`` where key ``t * P_MAX + p`` belongs. That is a pure permutation of
    the key axis, which softmax and the weighted sum ignore but the position mask does not, and it can
    still pass a loose tolerance, so check a change here against a known-good result, not by pass/fail.

    Every row index is in range, including non-resident slots, whose real rows the score mask discards
    (see the module docstring).

    Args:
        rows_sb (nl.NkiTensor): [P_MAX, n_slots, num_key_tiles, kv_row_bytes] uint8 @ SBUF, written.
        slot (int): Which slot of ``rows_sb`` to fill.
        kv_cache (nl.NkiTensor): [num_blocks, 1, block_size, kv_row_bytes] uint8 @ HBM, this rank's
            local cache of packed rows.
        snake_rows (nl.NkiTensor): [P_MAX, num_tokens, num_prior_tiles] uint32 @ SBUF.
        tok (int): Which query token's selection to gather.
        mla (MlaDecodeShape): Shapes.

    Returns:
        The gather's instruction handle, for schedule edges.
    """
    # The gather moves packed bytes: the batched 2-D indirect form allows no cast, so both sides are uint8.
    num_prior, kv_row = mla.num_prior, mla.kv_row_bytes
    n_tiles = mla.num_prior_tiles
    cache_rows = kv_cache.reshape((mla.num_cache_rows, kv_row))

    return nisa.dma_copy(
        dst=rows_sb[0:P_MAX, slot, 0:n_tiles, 0:kv_row],
        src=cache_rows.ap(
            pattern=[[kv_row, num_prior], [1, kv_row]],
            offset=0,
            vector_offset=snake_rows[0:P_MAX, tok, 0:n_tiles],
            indirect_dim=0,
        ),
    )


def _resolve_local_rows(
    gather_indices: nl.NkiTensor,
    block_table: nl.NkiTensor,
    mla: MlaDecodeShape,
    seq_start: int,
    seq_count: int,
    pos_offset: nl.NkiTensor = None,
    positions_km: nl.NkiTensor = None,
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """
    Map a group of sequences' global selected positions to local cache rows plus masked-out sentinels.

    Returns ``(cache_row, key_pos)``, both [P_MAX, seq_count, spec, num_prior_tiles] int32, key-major.
    """
    num_prior, block_size = mla.num_prior, mla.block_size
    max_blocks, block_shift = mla.max_blocks, mla.block_shift
    cp_degree = mla.cp_degree
    cp_shift = _log2_int(cp_degree)
    n_tiles = mla.num_prior_tiles
    spec = mla.spec
    tok_count = seq_count * spec

    # Load this group's global positions, key-major.
    pos = nl.ndarray((P_MAX, seq_count, spec, n_tiles), dtype=nl.int32, buffer=nl.sbuf)
    km_sb = _positions_km_sbuf(positions_km) if (seq_start == 0 and seq_count == mla.num_seqs == 1) else None
    if km_sb is not None:
        # Copy the merge's own SBUF tile, since the resolve below writes ``pos`` in place.
        nisa.tensor_copy(
            dst=pos.reshape((P_MAX, spec * n_tiles))[0:P_MAX, 0 : spec * n_tiles],
            src=km_sb[0:P_MAX, 0 : spec * n_tiles],
            engine=nisa.vector_engine,
        )
    elif positions_km is not None:
        # Already key-major, one [P_MAX, spec, n_tiles] block per sequence: a contiguous
        # spec * n_tiles run per partition per sequence.
        km = positions_km.reshape((mla.num_seqs, P_MAX, spec, n_tiles))
        for s in range(seq_count):
            nisa.dma_copy(
                dst=pos[0:P_MAX, s, 0:spec, 0:n_tiles],
                src=km[seq_start + s, 0:P_MAX, 0:spec, 0:n_tiles],
            )
    else:
        # gather_indices is [num_seqs, spec, num_prior], so a group of seq_count sequences covers
        # seq_count * spec contiguous rows. Key ``i * P_MAX + p`` lands on partition p: a 4-byte read
        # per partition per (token, tile).
        flat = gather_indices.reshape((mla.num_tokens * num_prior, 1))
        nisa.dma_copy(
            dst=pos[0:P_MAX, 0:seq_count, 0:spec, 0:n_tiles],
            src=flat.ap(
                pattern=[[1, P_MAX], [num_prior * spec, seq_count], [num_prior, spec], [P_MAX, n_tiles]],
                offset=seq_start * spec * num_prior,
            ),
        )
    # Every op below is elementwise over the free axes, so they all work on the flat view.
    pos = pos.reshape((P_MAX, tok_count, n_tiles))

    if pos_offset is not None:
        # The per-rank position shift on live slots only, pos += (pos >= 0) * offset: a few Vector ops on
        # the full tile instead of a separate pass. A dead slot (-1) must stay -1 for the validity tests.
        # The operand tile is fp32 (tensor_scalar's per-partition operand dtype); the DMA converts it.
        off_sb = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=off_sb[0:P_MAX, 0:1], src=pos_offset[0:1, 0:1].broadcast(0, P_MAX))
        live = nl.ndarray((P_MAX, tok_count, n_tiles), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            live[0:P_MAX, 0:tok_count, 0:n_tiles],
            pos[0:P_MAX, 0:tok_count, 0:n_tiles],
            op0=nl.greater_equal,
            operand0=0,
        )
        nisa.tensor_scalar(
            live[0:P_MAX, 0:tok_count, 0:n_tiles],
            live[0:P_MAX, 0:tok_count, 0:n_tiles],
            op0=nl.multiply,
            operand0=off_sb[0:P_MAX, 0:1],
        )
        nisa.tensor_tensor(
            pos[0:P_MAX, 0:tok_count, 0:n_tiles],
            pos[0:P_MAX, 0:tok_count, 0:n_tiles],
            live[0:P_MAX, 0:tok_count, 0:n_tiles],
            nl.add,
            engine=nisa.engine.vector,
        )

    # The global logical block of each position, and the offset inside it.
    block_g = nl.ndarray((P_MAX, tok_count, n_tiles), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        block_g[0:P_MAX, 0:tok_count, 0:n_tiles],
        pos[0:P_MAX, 0:tok_count, 0:n_tiles],
        op0=nl.right_shift,
        operand0=block_shift,
    )
    intra = nl.ndarray((P_MAX, tok_count, n_tiles), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        intra[0:P_MAX, 0:tok_count, 0:n_tiles],
        pos[0:P_MAX, 0:tok_count, 0:n_tiles],
        op0=nl.bitwise_and,
        operand0=block_size - 1,
    )

    # CP residency: blocks are dealt round-robin, so this rank owns global block g iff
    # g % cp_degree == cp_rank and stores it at local index g / cp_degree. The selector favors recent
    # positions, so a contiguous split would put most picks on the rank holding the newest chunk;
    # round-robin balances them. The table stays local-width because a global-width one would not fit
    # on every partition at long context.
    owner = nl.ndarray((P_MAX, tok_count, n_tiles), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        owner[0:P_MAX, 0:tok_count, 0:n_tiles],
        block_g[0:P_MAX, 0:tok_count, 0:n_tiles],
        op0=nl.bitwise_and,
        operand0=cp_degree - 1,
    )
    block_l = nl.ndarray((P_MAX, tok_count, n_tiles), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        block_l[0:P_MAX, 0:tok_count, 0:n_tiles],
        block_g[0:P_MAX, 0:tok_count, 0:n_tiles],
        op0=nl.right_shift,
        operand0=cp_shift,
    )
    # Clamp the local block into the table before the lookup. Out-of-window slots alias onto the last
    # column, and the residency and sign tests below reject them anyway.
    nisa.tensor_scalar(
        block_l[0:P_MAX, 0:tok_count, 0:n_tiles],
        block_l[0:P_MAX, 0:tok_count, 0:n_tiles],
        op0=nl.minimum,
        operand0=max_blocks - 1,
    )
    nisa.tensor_scalar(
        block_l[0:P_MAX, 0:tok_count, 0:n_tiles],
        block_l[0:P_MAX, 0:tok_count, 0:n_tiles],
        op0=nl.maximum,
        operand0=0,
    )

    # Local logical block to physical block through this rank's table. The table is small and any
    # partition may index any entry, so it is broadcast once and looked up on chip; an indirect DMA per
    # lookup would move 4 bytes per descriptor.
    if _AFFINE_TABLE:
        return _resolve_rows_tail(
            pos, intra, owner, _affine_block_id(block_l, mla, seq_start, seq_count, int(_AFFINE_TABLE)), mla, tok_count
        )
    table_flat = block_table.reshape((mla.num_seqs * max_blocks, 1))
    group_width = seq_count * max_blocks
    table_sb = nl.ndarray((P_MAX, group_width), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=table_sb[0:P_MAX, 0:group_width],
        src=table_flat.ap(pattern=[[0, P_MAX], [1, group_width]], offset=seq_start * max_blocks),
    )
    # Each sequence indexes its own row: lookup = s * max_blocks + local_block. The ramp is flat across
    # the spec and tile axes because a sequence's speculative tokens share one table row.
    seq_offset_4d = nl.ndarray((P_MAX, seq_count, spec, n_tiles), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(
        seq_offset_4d[0:P_MAX, 0:seq_count, 0:spec, 0:n_tiles],
        pattern=[[max_blocks, seq_count], [0, spec], [0, n_tiles]],
        offset=0,
        channel_multiplier=0,
    )
    tok_offset = seq_offset_4d.reshape((P_MAX, tok_count, n_tiles))
    lookup = nl.ndarray((P_MAX, tok_count, n_tiles), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        lookup[0:P_MAX, 0:tok_count, 0:n_tiles],
        block_l[0:P_MAX, 0:tok_count, 0:n_tiles],
        tok_offset[0:P_MAX, 0:tok_count, 0:n_tiles],
        nl.add,
        engine=nisa.engine.vector,
    )
    lookup_u32 = nl.ndarray((P_MAX, tok_count, n_tiles), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_copy(
        dst=lookup_u32[0:P_MAX, 0:tok_count, 0:n_tiles],
        src=lookup[0:P_MAX, 0:tok_count, 0:n_tiles],
    )
    block_id = nl.ndarray((P_MAX, tok_count, n_tiles), dtype=nl.int32, buffer=nl.sbuf)
    nisa.nc_n_gather(
        dst=block_id[0:P_MAX, 0:tok_count, 0:n_tiles],
        data=table_sb[0:P_MAX, 0:group_width],
        indices=lookup_u32[0:P_MAX, 0:tok_count, 0:n_tiles],
    )

    return _resolve_rows_tail(pos, intra, owner, block_id, mla, tok_count)


def _affine_block_id(block_l, mla: MlaDecodeShape, seq_start: int, seq_count: int, k: int):
    """``table[s, j] = s * max_blocks + j + k`` without the table, one Vector op per sequence (``_AFFINE_TABLE``)."""
    n_tiles, spec = mla.num_prior_tiles, mla.spec
    tok_count = seq_count * spec
    block_id = nl.ndarray((P_MAX, tok_count, n_tiles), dtype=nl.int32, buffer=nl.sbuf)
    for s in range(seq_count):
        t0, t1 = s * spec, (s + 1) * spec
        nisa.tensor_scalar(
            block_id[0:P_MAX, t0:t1, 0:n_tiles],
            block_l[0:P_MAX, t0:t1, 0:n_tiles],
            op0=nl.add,
            operand0=(seq_start + s) * mla.max_blocks + k,
            engine=nisa.vector_engine,
        )
    return block_id


def _resolve_rows_tail(pos, intra, owner, block_id, mla: MlaDecodeShape, tok_count: int):
    """The validity, cache-row and key-position half of ``_resolve_local_rows``, from the looked-up physical blocks."""
    cp_rank, block_size = mla.cp_rank, mla.block_size
    n_tiles, spec = mla.num_prior_tiles, mla.spec
    seq_count = tok_count // spec
    # A slot is invalid if this rank does not own the block (owner != cp_rank), the block is not
    # allocated (block_id < 0), or the selector left it empty (pos < 0). pos is tested directly rather
    # than through the shifted block, whose arithmetic shift of a negative value is not reliable.
    off_rank = nl.ndarray((P_MAX, tok_count, n_tiles), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        off_rank[0:P_MAX, 0:tok_count, 0:n_tiles],
        owner[0:P_MAX, 0:tok_count, 0:n_tiles],
        op0=nl.not_equal,
        operand0=cp_rank,
    )
    unallocated = nl.ndarray((P_MAX, tok_count, n_tiles), dtype=nl.int32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=unallocated[0:P_MAX, 0:tok_count, 0:n_tiles],
        data=block_id[0:P_MAX, 0:tok_count, 0:n_tiles],
        op0=nl.minimum,
        operand0=0.0,
        op1=nl.minimum,
        operand1=pos[0:P_MAX, 0:tok_count, 0:n_tiles],
    )
    # unallocated < 0 iff block_id < 0 or pos < 0. The 0/1 ``empty`` flag stays separate for the row
    # spread below, then is OR'd with the owner test into ``invalid``.
    empty = nl.ndarray((P_MAX, tok_count, n_tiles), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        empty[0:P_MAX, 0:tok_count, 0:n_tiles],
        unallocated[0:P_MAX, 0:tok_count, 0:n_tiles],
        op0=nl.less,
        operand0=0.0,
    )
    invalid = nl.ndarray((P_MAX, tok_count, n_tiles), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        invalid[0:P_MAX, 0:tok_count, 0:n_tiles],
        empty[0:P_MAX, 0:tok_count, 0:n_tiles],
        off_rank[0:P_MAX, 0:tok_count, 0:n_tiles],
        nl.maximum,
    )

    # Physical block and offset to a flat cache row.
    cache_row = nl.ndarray((P_MAX, tok_count, n_tiles), dtype=nl.int32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=cache_row[0:P_MAX, 0:tok_count, 0:n_tiles],
        data=block_id[0:P_MAX, 0:tok_count, 0:n_tiles],
        op0=nl.multiply,
        operand0=float(block_size),
        op1=nl.add,
        operand1=intra[0:P_MAX, 0:tok_count, 0:n_tiles],
    )
    nisa.tensor_scalar(
        cache_row[0:P_MAX, 0:tok_count, 0:n_tiles],
        cache_row[0:P_MAX, 0:tok_count, 0:n_tiles],
        op0=nl.minimum,
        operand0=mla.num_cache_rows - 1,
    )
    # Every slot gathers a real, in-range row, and empty slots are spread over the cache.
    #
    # Skipping out-of-bounds slots would move no bytes for the many non-resident ones, but a skipped
    # indirect DMA leaves its destination untouched, and the context matmul reads that tile directly,
    # where 0 * NaN is NaN whatever the mask says. Two of the 256 float8_e4m3fn encodings are NaN, so
    # stale SBUF bytes poison output dims (seen on hardware as run-to-run varying NaNs that only the
    # determinism check caught). A one-time memset does not help: the gather looks like a full write,
    # so the memset can be eliminated, and a later group's slot still holds an earlier gather's row.
    #
    # A non-resident slot already has a real row: block_l floors onto the nearest owned block below,
    # so the table returns a real block and intra a real offset. An empty slot (pos < 0, fewer than
    # num_prior candidates, or block_id < 0, an unallocated entry) has none, and clamping it to row 0
    # would send every empty slot to the same row and serialize those reads on one HBM channel. Its
    # own gather index, an iota clamped into the cache, becomes its row instead, so a step's empty
    # slots read distinct consecutive rows. Cache rows always hold finite values.
    #
    # Either way the key_pos sentinel below makes the mask drop the slot, so the row it fetched is
    # weighted by exactly 0; the row choice only decides where the DMA traffic lands.
    #
    # spread[p, t, i] = t * num_prior + i * P_MAX + p, the slot's flat key-major index. The min into
    # the cache is emitted only when the step's slots outnumber the rows.
    spread = nl.ndarray((P_MAX, tok_count, n_tiles), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(
        spread[0:P_MAX, 0:tok_count, 0:n_tiles],
        pattern=[[P_MAX * n_tiles, tok_count], [P_MAX, n_tiles]],
        offset=0,
        channel_multiplier=1,
    )
    if tok_count * P_MAX * n_tiles > mla.num_cache_rows:
        nisa.tensor_scalar(
            spread[0:P_MAX, 0:tok_count, 0:n_tiles],
            spread[0:P_MAX, 0:tok_count, 0:n_tiles],
            op0=nl.minimum,
            operand0=mla.num_cache_rows - 1,
        )
    # The select is max(cache_row - BIG * empty, empty * spread): a non-empty slot compares its row
    # (non-negative by construction) against 0 and keeps it exactly; an empty one pushes its row below
    # every real one and takes the spread. Only two instructions then sit behind cache_row, and both
    # are pinned to Vector: left to the scheduler they land on GpSimd behind the gathers' descriptor
    # generation. The integer math runs through fp32, so BIG must keep cache_row - BIG exact, which
    # 2**23 does for a cache of at most 2**23 rows (asserted below).
    kernel_assert(
        mla.num_cache_rows <= _SPREAD_BIG,
        f"num_cache_rows={mla.num_cache_rows} must be <= {_SPREAD_BIG} for the empty-slot spread select",
    )
    empty_spread = nl.ndarray((P_MAX, tok_count, n_tiles), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        empty_spread[0:P_MAX, 0:tok_count, 0:n_tiles],
        empty[0:P_MAX, 0:tok_count, 0:n_tiles],
        spread[0:P_MAX, 0:tok_count, 0:n_tiles],
        nl.multiply,
        engine=nisa.vector_engine,
    )
    nisa.scalar_tensor_tensor(
        dst=cache_row[0:P_MAX, 0:tok_count, 0:n_tiles],
        data=empty[0:P_MAX, 0:tok_count, 0:n_tiles],
        op0=nl.multiply,
        operand0=-float(_SPREAD_BIG),
        op1=nl.add,
        operand1=cache_row[0:P_MAX, 0:tok_count, 0:n_tiles],
    )
    nisa.tensor_tensor(
        cache_row[0:P_MAX, 0:tok_count, 0:n_tiles],
        cache_row[0:P_MAX, 0:tok_count, 0:n_tiles],
        empty_spread[0:P_MAX, 0:tok_count, 0:n_tiles],
        nl.maximum,
        engine=nisa.vector_engine,
    )

    # An invalid slot gets a key position the mask cannot keep. A valid one is unchanged exactly
    # (0 * NON_RESIDENT_POS + pos == pos, with pos well inside fp32's exact-integer range), so a rank
    # that owns everything matches no localization byte for byte.
    key_pos = nl.ndarray((P_MAX, tok_count, n_tiles), dtype=nl.int32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=key_pos[0:P_MAX, 0:tok_count, 0:n_tiles],
        data=invalid[0:P_MAX, 0:tok_count, 0:n_tiles],
        op0=nl.multiply,
        operand0=float(NON_RESIDENT_POS),
        op1=nl.add,
        operand1=pos[0:P_MAX, 0:tok_count, 0:n_tiles],
    )

    return (
        cache_row.reshape((P_MAX, seq_count, spec, n_tiles)),
        key_pos.reshape((P_MAX, seq_count, spec, n_tiles)),
    )


_RESIDENT_KEEP_POS = -float(1 << 29)
"""Key position of a kept compacted key in ``build_resident_index``'s mask table: ``key_pos - pos < 0.5`` holds for
every query position, so the core's mask keeps it. A padding slot gets NON_RESIDENT_POS."""

_GPSIMD_CORE_PARTS = 16
"""Partitions per GpSimd core. ``nisa.nonzero_with_count`` reads and writes only partition ``16 * c``, so token ``t``
of a resident index lives on partition ``16 * t``."""


def build_resident_index(
    key_pos_f: nl.NkiTensor,
    natural_rows: nl.NkiTensor,
    positions: nl.NkiTensor,
    mla: MlaDecodeShape,
    n_tiles_c: int,
    dge: dge_mode = dge_mode.unknown,
) -> tuple[nl.NkiTensor, nl.NkiTensor, nl.NkiTensor]:
    """
    Compact each query token's kept prior keys to the front, for the resident-only attention core.

    A kept key is one the core's mask keeps: selected, resident on this CP rank and causal
    (``key_pos <= positions[tok]``). Only about 1/cp_degree of the selection is kept, so the resident-only
    core gathers and scores ``n_tiles_c`` key tiles per token instead of ``num_prior_tiles``.

    Steps: build keep and ``row + 1`` key-major; move token ``t``'s values onto partition ``16 t`` through HBM;
    ``nonzero_with_count`` lists the kept keys in order on each GpSimd core and ``nc_n_gather`` picks their rows;
    padding slots past the count get distinct in-range rows (finite bytes the mask discards); then ``n_tiles_c``
    PE transposes (bit-accurate) put compacted key ``j`` of token ``t`` at ``[j % 128, t, j // 128]``. The counts
    come from a ones matmul over the key-major keep.

    Args:
        key_pos_f (nl.NkiTensor): [P_MAX, T, num_prior_tiles] fp32 @ SBUF, from ``build_prior_index``.
        natural_rows (nl.NkiTensor): [P_MAX, T, num_prior_tiles] uint32 @ SBUF, from ``build_prior_index``.
        positions (nl.NkiTensor): [T] int32 @ HBM, the query positions the core masks with.
        mla (MlaDecodeShape): Shapes of the full selection.
        n_tiles_c (int): Compacted key tiles per token; ``n_c = n_tiles_c * P_MAX`` keys.
        dge (dge_mode): DGE mode of the plain DMAs (hwdge inside a dynamic loop).

    Returns:
        rows_c (nl.NkiTensor): [P_MAX, T, n_tiles_c] uint32 @ SBUF, the compacted gather table.
        key_pos_c (nl.NkiTensor): [P_MAX, T, n_tiles_c] fp32 @ SBUF, the core's mask table for it: a kept key
            carries a position below every query, a padding slot NON_RESIDENT_POS.
        flag (nl.NkiTensor): [1, 1] int32 @ SBUF, 1 when some token keeps more than ``n_c`` keys (the resident-only
            core is then incomplete and the full core must run), else 0.
    """
    T, n_t = mla.num_tokens, mla.num_prior_tiles
    n_keys = n_t * P_MAX
    n_c = n_tiles_c * P_MAX
    G = _GPSIMD_CORE_PARTS
    kernel_assert(
        T * G <= P_MAX and 0 < n_tiles_c <= n_t and mla.num_cache_rows < (1 << 24),
        f"resident index: {T} tokens x {G} partitions must fit {P_MAX}, 0 < n_tiles_c={n_tiles_c} <= {n_t}, "
        f"and {mla.num_cache_rows} cache rows < 2**24",
    )

    pos_i = nl.ndarray((P_MAX, T), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=pos_i[0:P_MAX, 0:T], src=positions[0:T].reshape((1, T)).ap(pattern=[[0, P_MAX], [1, T]], offset=0), dge_mode=dge
    )
    pos_f = nl.ndarray((P_MAX, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pos_f[0:P_MAX, 0:T], src=pos_i[0:P_MAX, 0:T])

    # keep, exactly the core's test (key_pos - pos < 0.5), and row + 1 for the kept keys (0 for the rest).
    keep = nl.ndarray((P_MAX, T, n_t), dtype=nl.float32, buffer=nl.sbuf)
    for tok in range(T):
        nisa.tensor_scalar(
            dst=keep[0:P_MAX, tok, 0:n_t],
            data=key_pos_f[0:P_MAX, tok, 0:n_t],
            op0=nl.subtract,
            operand0=pos_f[0:P_MAX, tok : tok + 1],
            op1=nl.less,
            operand1=0.5,
        )
    rk = nl.ndarray((P_MAX, T, n_t), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=rk[0:P_MAX, 0:T, 0:n_t],
        data=natural_rows[0:P_MAX, 0:T, 0:n_t],
        op0=nl.add,
        operand0=1.0,
        op1=nl.multiply,
        operand1=keep[0:P_MAX, 0:T, 0:n_t],
    )

    # Kept keys per (token, tile), replicated on every partition by a ones matmul (exact: counts <= 128), then per
    # token; the overflow flag is their max against the capacity.
    keep_bf = nl.ndarray((P_MAX, T * n_t), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=keep_bf[0:P_MAX, 0 : T * n_t], src=keep.reshape((P_MAX, T * n_t))[0:P_MAX, 0 : T * n_t])
    ones_bf = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=ones_bf[0:P_MAX, 0:P_MAX], value=1.0)
    cnt_psum = nl.ndarray((P_MAX, T * n_t), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=cnt_psum[0:P_MAX, 0 : T * n_t], stationary=ones_bf[0:P_MAX, 0:P_MAX], moving=keep_bf[0:P_MAX, 0 : T * n_t])
    n_kept = nl.ndarray((P_MAX, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(n_kept[0:P_MAX, 0:T], nl.add, cnt_psum.reshape((P_MAX, T, n_t))[0:P_MAX, 0:T, 0:n_t], 2)
    n_max = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(n_max[0:1, 0:1], nl.maximum, n_kept[0:1, 0:T], 1)
    flag_f = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=flag_f[0:1, 0:1], data=n_max[0:1, 0:1], op0=nl.greater, operand0=float(n_c))
    flag = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=flag[0:1, 0:1], src=flag_f[0:1, 0:1])

    # Token-major: token t's row + 1 values on partition G * t, in key order p * n_t + i (any fixed order works).
    rk_hbm = nl.ndarray((P_MAX, T, n_t), dtype=nl.float32, buffer=nl.private_hbm, name=unique_name("resident_rk"))
    nisa.dma_copy(dst=rk_hbm[0:P_MAX, 0:T, 0:n_t], src=rk[0:P_MAX, 0:T, 0:n_t], dge_mode=dge)
    rk_tm = nl.ndarray((P_MAX, n_keys + 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=rk_tm[0:P_MAX, 0 : n_keys + 1], value=0.0)
    rk_flat = rk_hbm.reshape((P_MAX * T * n_t,))
    for tok in range(T):
        nisa.dma_copy(
            # One partition's n_keys values as [P_MAX, n_t]. The explicit access pattern gives the size-1 partition
            # dim a real step; a reshape gives it 0, which the BIR verifier refuses. Same bytes, same order.
            dst=rk_tm[G * tok : G * tok + 1, 0:n_keys].ap(pattern=[[n_keys + 1, 1], [n_t, P_MAX], [1, n_t]]),
            src=rk_flat.ap(pattern=[[0, 1], [T * n_t, P_MAX], [1, n_t]], offset=tok * n_t),
            dge_mode=dge,
        )

    # The kept keys' indices in order, padded with n_keys (the zero column), and the count in the last column.
    idx = nl.ndarray((P_MAX, n_keys + 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=idx[0:P_MAX, 0 : n_keys + 1], value=0)
    nisa.nonzero_with_count(dst=idx[0:P_MAX, 0 : n_keys + 1], src=rk_tm[0:P_MAX, 0:n_keys], index_offset=0, padding_val=n_keys)
    idx_u = nl.ndarray((P_MAX, n_c), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=idx_u[0:P_MAX, 0:n_c], src=idx[0:P_MAX, 0:n_c])
    picked = nl.ndarray((P_MAX, n_c), dtype=nl.float32, buffer=nl.sbuf)
    nisa.nc_n_gather(dst=picked[0:P_MAX, 0:n_c], data=rk_tm[0:P_MAX, 0 : n_keys + 1], indices=idx_u[0:P_MAX, 0:n_c])

    # rows_c = j < count ? picked - 1 : spread, where spread gives token t its own distinct rows t * n_c + j, like
    # the empty-slot spread, so padding gathers interleave over the cache. Integers below 2**24, so exact.
    count_f = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=count_f[0:P_MAX, 0:1], src=idx[0:P_MAX, n_keys : n_keys + 1])
    j_f = nl.ndarray((P_MAX, n_c), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(j_f[0:P_MAX, 0:n_c], pattern=[[1, n_c]], offset=0, channel_multiplier=0)
    valid = nl.ndarray((P_MAX, n_c), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=valid[0:P_MAX, 0:n_c], data=j_f[0:P_MAX, 0:n_c], op0=nl.less, operand0=count_f[0:P_MAX, 0:1])
    spread = nl.ndarray((P_MAX, n_c), dtype=nl.float32, buffer=nl.sbuf)
    kernel_assert(n_c % G == 0, f"n_c={n_c} must be a multiple of {G}")
    nisa.iota(spread[0:P_MAX, 0:n_c], pattern=[[1, n_c]], offset=0, channel_multiplier=n_c // G)
    nisa.tensor_scalar(dst=spread[0:P_MAX, 0:n_c], data=spread[0:P_MAX, 0:n_c], op0=nl.minimum, operand0=float(mla.num_cache_rows - 1))
    rows_tm = nl.ndarray((P_MAX, n_c), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=rows_tm[0:P_MAX, 0:n_c], data=picked[0:P_MAX, 0:n_c], op0=nl.add, operand0=-1.0, op1=nl.subtract,
        operand1=spread[0:P_MAX, 0:n_c],
    )
    nisa.tensor_tensor(rows_tm[0:P_MAX, 0:n_c], rows_tm[0:P_MAX, 0:n_c], valid[0:P_MAX, 0:n_c], nl.multiply)
    nisa.tensor_tensor(rows_tm[0:P_MAX, 0:n_c], rows_tm[0:P_MAX, 0:n_c], spread[0:P_MAX, 0:n_c], nl.add)

    # Key-major: chunk b of token t's list, transposed on the PE (bit-accurate), is column G * t of the result.
    rows_c = nl.ndarray((P_MAX, T, n_tiles_c), dtype=nl.uint32, buffer=nl.sbuf)
    for b in range(n_tiles_c):
        t_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(t_psum[0:P_MAX, 0:P_MAX], rows_tm[0:P_MAX, b * P_MAX : (b + 1) * P_MAX])
        nisa.tensor_copy(
            dst=rows_c[0:P_MAX, 0:T, b : b + 1].reshape((P_MAX, T)),
            src=t_psum.slice(1, start=0, end=G * (T - 1) + 1, step=G)[0:P_MAX],
            engine=nisa.engine.vector,
        )

    # The mask table: compacted key j = p * n_tiles_c + i is kept iff j < its token's count.
    jk = nl.ndarray((P_MAX, n_tiles_c), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(jk[0:P_MAX, 0:n_tiles_c], pattern=[[1, n_tiles_c]], offset=0, channel_multiplier=n_tiles_c)
    keep_c = nl.ndarray((P_MAX, T, n_tiles_c), dtype=nl.float32, buffer=nl.sbuf)
    for tok in range(T):
        nisa.tensor_scalar(
            dst=keep_c[0:P_MAX, tok, 0:n_tiles_c], data=jk[0:P_MAX, 0:n_tiles_c], op0=nl.less,
            operand0=n_kept[0:P_MAX, tok : tok + 1],
        )
    key_pos_c = nl.ndarray((P_MAX, T, n_tiles_c), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=key_pos_c[0:P_MAX, 0:T, 0:n_tiles_c],
        data=keep_c[0:P_MAX, 0:T, 0:n_tiles_c],
        op0=nl.multiply,
        operand0=_RESIDENT_KEEP_POS - float(NON_RESIDENT_POS),
        op1=nl.add,
        operand1=float(NON_RESIDENT_POS),
    )
    return rows_c, key_pos_c, flag


def _log2_int(n: int) -> int:
    shift = 0
    while n > 1:
        n >>= 1
        shift += 1
    return shift
