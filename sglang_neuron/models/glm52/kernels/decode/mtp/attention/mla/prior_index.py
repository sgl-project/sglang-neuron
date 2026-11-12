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

The selector picks the top-k over the whole context, so positions arrive global. Under context
parallelism this rank holds only every ``cp_degree``'th logical block, so most are not resident. For a
slot this rank cannot serve:

* the row is still a real in-range one (a nearby owned block's for a non-resident slot, an
  iota-spread row for an empty one), so the gather always moves finite bytes, and
* the key position is replaced by a huge sentinel, so the core's causal mask drops the score to
  -MASK_BOUND and the fetched row is multiplied by exactly zero.

The sentinel is what makes it correct; the row choice only keeps the data finite and spreads the
traffic. Skipping the transfer instead would leave the destination untouched, and the context matmul
reads that tile directly, where ``0 * NaN`` is NaN (see ``_resolve_local_rows``).
"""

import nki.isa as nisa
import nki.language as nl
from nki.isa import dge_mode

from ....attention.mla.mla_decode_constants import NON_RESIDENT_POS, P_MAX, MlaDecodeShape
from ....utils.kernel_assert import kernel_assert
from ....utils.scratch_hbm import scratch_hbm
from ....utils.trace_names import unique_name

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
Destination alignment a DGE crossbar transpose requires, per destination group.

nki does not validate this; a violation surfaces as an opaque backend "exit code 70". With an
innermost dst extent of ``inner`` 2-byte elements, the rule is ``inner * 2 % 32 == 0``.
"""

_SPREAD_BIG = 1 << 23
"""Pushes an empty slot's row below every real one in the spread select of ``_resolve_local_rows``. The
cache must hold fewer rows than this so ``cache_row - _SPREAD_BIG`` stays exact in fp32."""

# When every cache row fits 16 bits only the low half of the snake table is built, since the high half is all zero.
# The two int32 adds on the resolve chain (live-slot offset, block lookup) are pinned to Vector; unpinned they land
# on the slower GpSimd. Both are exact since every value is an integer < 2**24.


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
        gather_indices (nl.NkiTensor): [num_seqs, spec, num_prior] int32 @ HBM, global positions chosen
            by the selector, one row per (sequence, speculative token). Negative means not selected.
        block_table (nl.NkiTensor): [num_seqs, max_blocks] int32 @ HBM, this rank's local table: entry
            ``[s, i]`` is the physical block of local logical block ``i``, negative if unallocated.
        mla (MlaDecodeShape): Shapes, plus the CP degree and this rank's index in its CP group.
        pos_offset (nl.NkiTensor): [1, 1] int32 @ HBM, or None. Added to every non-negative position
            (a -1 slot stays -1), so the caller's per-rank shift needs no separate pass.
        positions_km (nl.NkiTensor): [num_seqs * P_MAX, spec * num_prior_tiles] int32 @ HBM, or None.
            The same selection already key-major, as ``indexer_topk_merge_carry_index_packed`` publishes
            it: ``[s * P_MAX + p, t * tiles + i]`` is slot ``i * P_MAX + p`` of token ``(s, t)``. When
            given, positions load with one contiguous DMA per partition and ``gather_indices`` only sizes
            the shapes. Any fixed slot permutation is fine: the tables stay mutually consistent and every
            consumer treats the slots as an unordered set.

    Returns:
        snake_rows (nl.NkiTensor): [P_MAX, num_tokens, num_prior_tiles] uint32 @ SBUF, local cache rows
            permuted into the order a batched indirect gather consumes them.
        key_pos_f (nl.NkiTensor): [P_MAX, num_tokens, num_prior_tiles] fp32 @ SBUF, the position of the
            key the gather places at ``[p, t]``, or NON_RESIDENT_POS. Partition-major, as the mask reads it.
        natural_rows (nl.NkiTensor): [P_MAX, num_tokens, num_prior_tiles] uint32 @ SBUF, the same rows in
            identity key order (entry [p, tok, i] is key ``i * P_MAX + p``), for ``nisa.dma_transpose``.
    """
    num_prior = mla.num_prior
    num_seqs, spec = mla.num_seqs, mla.spec
    num_tokens = mla.num_tokens
    n_tiles = mla.num_prior_tiles
    # Each pass holds seq_group sequences' block tables resident on every partition. It counts whole
    # sequences because a table row is per sequence and the iota addressing it can ramp but not divide,
    # so a sequence's speculative siblings must resolve in the same pass.
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

    # Snake table: the batched gather's index, built with a hardware transpose. The gather flattens its
    # 2-D vector_offset column-major while walking the SBUF destination row-major, so the table is
    # pre-permuted: ``snake[p, t, i] == rows_flat[t * P_MAX * n_tiles + i * P_MAX + p]``, the
    # [n_tiles, P_MAX] -> [P_MAX, n_tiles] transpose of each token's index block.
    #
    # dma_transpose needs a 2-byte dtype, but a row index can exceed 16 bits at long context (128K over
    # CP8 with 16 sequences is 262,144 local rows), so the high and low halves are transposed separately
    # and recombined. That is still cheaper than the HBM round trip of the fallback.
    #
    # A negative index cannot be left to wrap to a large unsigned value: a dtype-changing tensor_copy goes
    # through fp32 and saturates, so -1 becomes row 0 and its live KV would be scored. Every sentinel is
    # therefore an explicit positive value.
    block = P_MAX * n_tiles
    snake_rows = nl.ndarray((P_MAX, num_tokens, n_tiles), dtype=nl.uint32, buffer=nl.sbuf)

    # Two DGE rules must hold, and n_tiles == 16 satisfies both: the hwdge 2-D source needs
    # src.shape[0] == 16, and the 32-byte destination alignment needs n_tiles * 2 % 32 == 0. nki has no
    # check for the second, so it is asserted separately in case the first is ever relaxed.
    if n_tiles == _HWDGE_2D_PDIM:
        kernel_assert(
            (n_tiles * 2) % _DGE_DST_ALIGN_BYTES == 0,
            f"DGE transpose dst must be {_DGE_DST_ALIGN_BYTES}-byte aligned, but the per-sequence "
            f"stride is {n_tiles * 2} B (n_tiles={n_tiles} uint16)",
        )
        half_sb = []
        low_only = mla.num_cache_rows <= (1 << 16)
        for half in range(1 if low_only else 2):
            # low half = row & 0xFFFF, high half = row >> 16; both nonnegative since every row is in range.
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
        # recombine: row = high * 65536 + low
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
        # Fallback for any n_tiles: stage int32 through HBM and read the permutation back (a read-back
        # that scatters 4 bytes per partition).
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
    # The natural-order table: entry [p, tok, i] is the row of key i * P_MAX + p. A consumer that walks
    # its destination differently (nisa.dma_transpose, whose flatten is already the identity) needs this
    # one; the snake table would permute the key axis, which the position mask is not invariant to.
    natural_rows = nl.ndarray((P_MAX, num_tokens, n_tiles), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_copy(
        dst=natural_rows[0:P_MAX, 0:num_tokens, 0:n_tiles],
        src=rows_i32[0:P_MAX, 0:num_tokens, 0:n_tiles],
    )

    return snake_rows, key_pos_f, natural_rows


def export_prior_index(
    snake_rows: nl.NkiTensor,
    key_pos_f: nl.NkiTensor,
    natural_rows: nl.NkiTensor,
    name: str = "prior_index",
) -> tuple[nl.NkiTensor, nl.NkiTensor, nl.NkiTensor]:
    """
    Stage the three index tables to HBM so they can be handed to a later ``mla_decode_attention``.

    The tables depend only on the selection, the block table and the CP shift, not on cache contents,
    so every layer in an index-share group has identical tables. Importing them costs three small DMAs
    per consumer instead of the whole resolve-and-snake chain.

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
) -> tuple[nl.NkiTensor, nl.NkiTensor, nl.NkiTensor]:
    """
    Load prebuilt index tables from HBM into the SBUF layout ``build_prior_index`` would have produced.

    Args:
        snake_hbm (nl.NkiTensor): [P_MAX, num_tokens, num_prior_tiles] uint32 @ HBM.
        key_pos_hbm (nl.NkiTensor): [P_MAX, num_tokens, num_prior_tiles] fp32 @ HBM.
        natural_hbm (nl.NkiTensor): [P_MAX, num_tokens, num_prior_tiles] uint32 @ HBM, or None when
            ``need_natural`` is False.
        mla (MlaDecodeShape): Shapes, for the assert.
        need_natural (bool): Whether the caller reads ``natural_rows`` (no draft core does); if False it
            is not loaded and None is returned.

    Returns:
        (snake_rows, key_pos_f, natural_rows) in SBUF; ``natural_rows`` is None if not needed.
    """
    shape = (P_MAX, mla.num_tokens, mla.num_prior_tiles)
    kernel_assert(
        tuple(snake_hbm.shape) == shape and tuple(key_pos_hbm.shape) == shape,
        f"prebuilt prior-index tables must be {shape}, got {snake_hbm.shape} / {key_pos_hbm.shape}",
    )
    snake_rows = nl.ndarray(shape, dtype=nl.uint32, buffer=nl.sbuf)
    nisa.dma_copy(dst=snake_rows, src=snake_hbm)
    key_pos_f = nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=key_pos_f, src=key_pos_hbm)
    natural_rows = None
    if need_natural:
        kernel_assert(
            natural_hbm is not None and tuple(natural_hbm.shape) == shape,
            f"the caller needs the natural-order table, shape {shape}",
        )
        natural_rows = nl.ndarray(shape, dtype=nl.uint32, buffer=nl.sbuf)
        nisa.dma_copy(dst=natural_rows, src=natural_hbm)
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
    Gather one query token's selected prior rows into a caller-owned slot of ``rows_sb``.

    One batched indirect DMA moves all num_prior rows; one DMA per tile would be bound by descriptor
    issue.

    ``snake_rows`` must be the snake-permuted table, since the hardware flattens the 2-D vector_offset
    column-major while walking the destination row-major. The natural-order table would land key
    ``p * n_tiles + t`` where ``t * P_MAX + p`` belongs: a pure permutation of the key axis that the
    mask is not invariant to, yet one that stays in range and can pass a loose tolerance. Check changes
    here against a known-good baseline, not by pass/fail.

    Every row index is in range, including non-resident slots, which point at a real row the mask then
    discards (see ``_resolve_local_rows``).

    Args:
        rows_sb (nl.NkiTensor): [P_MAX, n_slots, num_key_tiles, kv_row_bytes] uint8 @ SBUF, mutated.
        slot (int): Which slot of ``rows_sb`` to fill.
        kv_cache (nl.NkiTensor): [num_blocks, 1, block_size, kv_row_bytes] uint8 @ HBM, this rank's
            local cache of packed rows.
        snake_rows (nl.NkiTensor): [P_MAX, num_tokens, num_prior_tiles] uint32 @ SBUF.
        tok (int): Which query token's selection to gather.
        mla (MlaDecodeShape): Shapes.

    Returns:
        The gather's instruction handle, for schedule edges.
    """
    # The gather moves packed bytes: the batched 2-D indirect form forbids a cast, so both sides are uint8.
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
    Map a group of sequences' global selected positions to local cache rows and masked-out sentinels.

    Returns ``(cache_row, key_pos)``, both [P_MAX, seq_count, spec, num_prior_tiles] int32, key-major.
    """
    num_prior, block_size = mla.num_prior, mla.block_size
    max_blocks, block_shift = mla.max_blocks, mla.block_shift
    cp_degree, cp_rank = mla.cp_degree, mla.cp_rank
    cp_shift = _log2_int(cp_degree)
    n_tiles = mla.num_prior_tiles
    spec = mla.spec
    tok_count = seq_count * spec

    # Load this group's global positions, key-major.
    pos = nl.ndarray((P_MAX, seq_count, spec, n_tiles), dtype=nl.int32, buffer=nl.sbuf)
    if positions_km is not None:
        # One contiguous spec * n_tiles run per partition per sequence.
        km = positions_km.reshape((mla.num_seqs, P_MAX, spec, n_tiles))
        for s in range(seq_count):
            nisa.dma_copy(
                dst=pos[0:P_MAX, s, 0:spec, 0:n_tiles],
                src=km[seq_start + s, 0:P_MAX, 0:spec, 0:n_tiles],
            )
    else:
        # The group is seq_count * spec contiguous rows of gather_indices; key ``i * P_MAX + p`` lands on
        # partition p, a 4-byte read per partition per (token, tile).
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
        # The per-rank position shift on live slots only, pos += (pos >= 0) * offset: a few ops on the full
        # 128-partition tile instead of a DMA round trip. A dead slot must stay -1 for the validity tests.
        # The operand tile is fp32 (tensor_scalar's per-partition operand dtype); the DMA converts.
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

    # Global position -> global logical block, and the offset inside it.
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
    # g % cp_degree == cp_rank, stored at local index g / cp_degree. Round-robin because selections
    # cluster on recent positions: a contiguous split would put nearly every pick on the rank holding the
    # newest chunk, and that rank sets the barrier. A local-width table also keeps the resident table
    # small at long context; a global-width one would collapse seq_group to 1.
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
    # Clamp the local slot into the table. Out-of-window slots alias onto the last column and are
    # rejected by the residency and sign tests below.
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

    # Local logical block -> physical block. The table is tiny, so it is broadcast to every partition
    # once and looked up on chip; an indirect DMA per lookup would move 4 bytes per descriptor.
    table_flat = block_table.reshape((mla.num_seqs * max_blocks, 1))
    group_width = seq_count * max_blocks
    table_sb = nl.ndarray((P_MAX, group_width), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=table_sb[0:P_MAX, 0:group_width],
        src=table_flat.ap(pattern=[[0, P_MAX], [1, group_width]], offset=seq_start * max_blocks),
    )
    # lookup = s * max_blocks + local_block. The ramp is flat across spec and tiles because a sequence's
    # speculative tokens share one table row.
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

    # A slot is invalid if this rank does not own the block (owner != cp_rank), the block is unallocated
    # (block_id < 0), or the selector left it empty (pos < 0). ``pos`` is tested directly rather than
    # through the shifted block, whose arithmetic shift of a negative value is not relied on.
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
    # unallocated < 0 iff (block_id < 0 or pos < 0). ``empty`` is kept separately for the row spread
    # below, then OR'd with the owner test into ``invalid``.
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

    # Physical block + offset -> flat cache row.
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
    # Every slot gathers a real in-range row, never an out-of-bounds skip. A skipped indirect DMA leaves
    # its destination untouched, and the context matmul reads the gathered tile directly, where
    # 0 * NaN is NaN whatever the mask says: arbitrary SBUF bytes include the fp8 e4m3 NaN encodings
    # (0x7F, 0xFF), and on hardware skipping produced NaN in a varying set of latent dims while accuracy
    # checks still passed. A one-time memset does not help: the gather looks like a full write, so the
    # memset can be eliminated, and a later slot would still hold an earlier token's row.
    #
    # A non-resident slot needs nothing extra: ``block_l`` floors onto the nearest owned block below,
    # so it reads a real row of this rank's cache. An empty slot (pos < 0: fewer than num_prior
    # candidates, as in a sequence's first 2048 tokens; or block_id < 0) has no such row, and clamping
    # it to 0 would pile every empty slot onto one row and serialize on one HBM channel. Its gather
    # index ``t * num_prior + i * P_MAX + p`` becomes its row instead, so a step's empty slots read
    # distinct consecutive rows spread over every channel. Cache rows always hold finite data.
    #
    # Correctness comes from the key_pos sentinel below, which masks the slot so its row is multiplied
    # by exactly 0; the row choice only decides where the DMA traffic lands.
    # spread[p, t, i] = t * num_prior + i * P_MAX + p, the slot's flat key-major index. The min into the
    # cache is only emitted when the step's slots outnumber the rows.
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
    # Select max(cache_row - _SPREAD_BIG * empty, empty * spread): a non-empty slot keeps its row (>= 0) exactly,
    # an empty one takes its spread row. Only two instructions sit on the chain behind cache_row, and both
    # are pinned to Vector; on GpSimd they would queue behind the gathers' descriptor generation at
    # each time. The math runs through fp32, so the cache must have at most 2**23 rows.
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

    # An invalid slot gets a key position the mask cannot keep. A valid slot is unchanged bit-exactly
    # (0 * NON_RESIDENT_POS + pos == pos, exact in fp32), so a rank that owns everything matches no
    # localization byte for byte.
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


def _log2_int(n: int) -> int:
    shift = 0
    while n > 1:
        n >>= 1
        shift += 1
    return shift
