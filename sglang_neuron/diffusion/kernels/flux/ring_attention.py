# SPDX-License-Identifier: Apache-2.0
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

"""Const-max ring attention forward for FLUX.1-dev, with the LNC2 split on query rows.

Same math, prologue and ring as ``ring_attention_const_max_fwd`` (per-row Cauchy-Schwarz bound
c_i = softmax_scale * ||q_i|| * max_j ||k_j||, pure-addition ring merge, one final normalize), and
the same helpers, imported from that module. Only the division of the attention compute across the
two physical cores of an LNC2 launch differs.

``ring_attention_const_max_fwd`` gives each core whole heads, so FLUX at TP8 (3 heads per rank)
runs 2 heads on core 0 and 1 on core 1. Here every core computes ALL heads for its own contiguous
range of query rows: core 0 rows [0, split), core 1 rows [split, seqlen), with split the 128-row
boundary nearest seqlen / 2. The prologue (K/Q transpose, per-token norms, V copy, c_i assembly)
stays head-sharded and writes shared HBM, followed by core barriers, so each core reads the full
Q^T, c, K and V buffers it needs. Both cores issue the identical ring collectives on the identical
buffers. The SBUF accumulator holds [heads x own query groups], and each core normalizes and
stores only its own rows of every head. Inference only: the LSE output is not produced.
"""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.kernel_helpers import div_ceil
from nkilib.core.utils.modular_allocator import ModularAllocator
from sglang_neuron.diffusion.kernels.flux.attention_const_max import (
    _attention_const_max_partial,
    _max_sq_block_size,
)
from sglang_neuron.diffusion.kernels.flux.ring_attention_const_max_fwd import (
    _assemble_c_row,
    _copy_batch_sharded,
    _head_chunks,
    _heads_per_chunk,
    _issue_kv_permute,
    _normalize_one_batch,
    _shard_batch_runs,
    _transpose_qk_pipelined,
    _zero_kv_pad,
)

_P = nl.tile_size.pmax
_NORM_SCRATCH_BYTES = 16 * 1024
_UNIT_REGION_BYTES = 32 * 1024
_NUM_UNIT_REGIONS = 2


def _query_row_range(seqlen, shard_id, num_shards):
    """Query rows [start, end) this LNC shard computes; the split is the 128-row boundary nearest seqlen/2."""
    if num_shards == 1:
        return 0, seqlen
    split = _P * ((seqlen + _P) // (2 * _P))
    kernel_assert(0 < split < seqlen, f"LNC2 query-row split needs seqlen > {_P}, got {seqlen}")
    if shard_id == 0:
        return 0, split
    return split, seqlen


def _rotated(bases, offset):
    """``bases`` rotated left by ``offset``; None stays None."""
    if bases is None:
        return None
    return [bases[(idx + offset) % len(bases)] for idx in range(len(bases))]


def _run_partial_step(
    q_rows,
    k_buf,
    v_buf,
    o_aug_acc_sb,
    c_rows,
    chunk,
    softmax_scale,
    is_accumulate_step,
    kv_pad_rows,
    prologue_depth,
    tile_allocator,
    unit_bases,
):
    """Fold one ring step's partial attention for every head of `chunk` over this core's query rows."""
    _attention_const_max_partial(
        q_rows,
        k_buf[chunk.start : chunk.end],
        v_buf[chunk.start : chunk.end],
        o_aug_acc_sb,
        c_rows,
        softmax_scale=softmax_scale,
        aug_col_base=0,
        is_accumulate_step=is_accumulate_step,
        kv_pad_rows=kv_pad_rows,
        prologue_depth=prologue_depth,
        sbuf_allocator=tile_allocator,
        sbuf_unit_bases=unit_bases,
        sbuf_unit_bytes=_UNIT_REGION_BYTES,
    )


def _run_last_step_and_normalize(
    q_rows,
    k_buf,
    v_buf,
    o_aug_acc_sb,
    c_rows,
    o_out,
    chunk,
    row_start,
    row_end,
    head_dim,
    softmax_scale,
    is_accumulate_step,
    kv_pad_rows,
    lse_dtype,
    allocator,
    prologue_depth,
    tile_allocator,
    unit_bases,
    norm_base,
):
    """Last ring step: per head, fold its partial attention, then normalize and store its own rows.

    Interleaving per head lets head h's normalize and output DMA overlap head h+1's matmuls.
    """
    own_rows = row_end - row_start
    own_groups = div_ceil(own_rows, _P)
    units_per_head = div_ceil(own_rows, _max_sq_block_size())
    for local_head in range(chunk.count):
        head_idx = chunk.start + local_head
        aug_col_base = local_head * own_groups * (head_dim + 1)
        _attention_const_max_partial(
            q_rows[local_head : local_head + 1],
            k_buf[head_idx : head_idx + 1],
            v_buf[head_idx : head_idx + 1],
            o_aug_acc_sb,
            c_rows[local_head : local_head + 1],
            softmax_scale=softmax_scale,
            aug_col_base=aug_col_base,
            is_accumulate_step=is_accumulate_step,
            kv_pad_rows=kv_pad_rows,
            prologue_depth=prologue_depth,
            sbuf_allocator=tile_allocator,
            sbuf_unit_bases=_rotated(unit_bases, local_head * units_per_head),
            sbuf_unit_bytes=_UNIT_REGION_BYTES,
        )
        if tile_allocator is not None:
            allocator.set_current_address(norm_base)
        _normalize_one_batch(
            o_aug_acc_sb,
            aug_col_base,
            o_out[head_idx, row_start:row_end, :],
            None,
            head_dim,
            own_groups,
            own_rows,
            False,
            lse_dtype,
            allocator,
            None,
        )
        if tile_allocator is not None:
            norm_end = allocator.get_current_address()
            kernel_assert(
                norm_end <= norm_base + _NORM_SCRATCH_BYTES,
                f"normalize scratch ends at SBUF byte {norm_end}, past its "
                f"{_NORM_SCRATCH_BYTES}-byte region at {norm_base}",
            )


@nki.jit
def flux_ring_attention_fwd(
    q: nl.NkiTensor,
    k: nl.NkiTensor,
    v: nl.NkiTensor,
    replica_groups: tuple = None,
    num_workers: int = 1,
    softmax_scale: float = None,
    training: bool = False,
    lse_dtype: nki.dtype = nl.float32,
    kv_prefetch_depth: int = 2,
    prologue_depth: int = 3,
    pin_tile_sbuf: bool = True,
):
    """Fixed-max ring attention forward (non-causal) with LNC work split on query rows.

    Args:
        q (nl.NkiTensor): Query, token-major [b, seqlen, h, d], d == 128.
        k (nl.NkiTensor): Key, token-major [b, seqlen, h, d].
        v (nl.NkiTensor): Value, token-major [b, seqlen, h, d].
        replica_groups (tuple): Replica groups of the CP rings.
        num_workers (int): Ring size (context-parallel degree).
        softmax_scale (float): Softmax scale. Default 1/sqrt(d).
        training (bool): Must be False; the LSE output is not supported.
        lse_dtype (nki.dtype): Unused; kept for signature parity with ring_attention_const_max_fwd.
        kv_prefetch_depth (int): Ring steps ahead to launch the K/V permutes, clamped to
            [1, num_workers-1].
        prologue_depth (int): KV tiles of MM1 and exp issued before the first MM2 of each Q-block.
        pin_tile_sbuf (bool): Allocate every ring-step tile buffer at fixed SBUF addresses (two
            rotating Q-block unit regions above the accumulator), so the kernel occupies a fixed
            low SBUF range and the compiler's overlapping ops get the rest without waiting on it.

    Returns:
        o (nl.NkiTensor): [b, h, seqlen, d] attention output.
    """
    kernel_assert(
        nisa.get_nc_version() > nisa.nc_version.gen2,
        "flux_ring_attention_fwd not supported on trn1",
    )
    kernel_assert(not training, "flux_ring_attention_fwd is inference only (training=False)")

    batch, seqlen, q_heads, head_dim = q.shape
    _, k_seqlen, k_heads, k_head_dim = k.shape
    kernel_assert(q_heads == k_heads, "expects q_heads == kv_heads, broadcast before calling")
    kernel_assert(head_dim == _P, f"head_dim must be 128, got {head_dim}")
    kernel_assert(
        k_seqlen == seqlen and k_head_dim == head_dim,
        f"q and k must be token-major [b, seqlen, h, d]; got q={q.shape}, k={k.shape}",
    )
    if replica_groups is None:
        replica_groups = ()

    batch_heads = batch * q_heads
    num_token_groups = div_ceil(seqlen, _P)
    seq_padded = num_token_groups * _P
    kv_pad_rows = seq_padded - seqlen
    softmax_scale = softmax_scale or (1.0 / float(head_dim**0.5))

    q = q.reshape((batch, seqlen, q_heads * head_dim))
    k = k.reshape((batch, seqlen, k_heads * head_dim))
    v = v.reshape((batch, seqlen, q_heads * head_dim))
    o = nl.ndarray((batch_heads, seqlen, head_dim), dtype=q.dtype, buffer=nl.shared_hbm)

    program_ndims = nl.program_ndim()
    shard_id = nl.program_id(0) if program_ndims == 1 else 0
    num_shards = nl.num_programs(0) if program_ndims == 1 else 1
    kernel_assert(num_shards <= 2, f"supports LNC1 and LNC2, got {num_shards} shards")
    num_heads_per_shard = batch_heads // num_shards
    head_offset = shard_id * num_heads_per_shard
    has_remainder = (batch_heads % num_shards) != 0
    runs = _shard_batch_runs(batch_heads, num_heads_per_shard, head_offset, has_remainder, shard_id)

    row_start, row_end = _query_row_range(seqlen, shard_id, num_shards)
    own_groups = div_ceil(row_end - row_start, _P)

    prefetch_depth = min(max(kv_prefetch_depth, 1), max(num_workers - 1, 1))
    num_kv_buf = prefetch_depth + 1
    k_bufs = []
    v_bufs = []
    for buf_idx in range(num_kv_buf):
        k_bufs.append(
            nl.ndarray(
                (batch_heads, head_dim, seq_padded),
                dtype=k.dtype,
                buffer=nl.shared_hbm,
                name=f"fx_k_buf_{buf_idx}",
            )
        )
        v_bufs.append(
            nl.ndarray(
                (batch_heads, seq_padded, head_dim),
                dtype=v.dtype,
                buffer=nl.shared_hbm,
                name=f"fx_v_buf_{buf_idx}",
            )
        )
    replica_group = ReplicaGroup(replica_groups)

    chunk_width = min(batch_heads, _heads_per_chunk(own_groups, head_dim))
    allocator = ModularAllocator(initial_address=0)
    o_aug_acc_sb = allocator.alloc_sbuf_tensor(
        shape=(_P, chunk_width * own_groups * (head_dim + 1)), dtype=nl.float32
    )

    local_k_maxsq = nl.ndarray(
        (batch_heads, 1), dtype=nl.float32, buffer=nl.shared_hbm, name="fx_local_k_maxsq"
    )
    reduce_k_max = num_workers > 1
    if reduce_k_max:
        global_k_maxsq = nl.ndarray(
            (batch_heads, 1), dtype=nl.float32, buffer=nl.shared_hbm, name="fx_global_k_maxsq"
        )
    c_row = nl.ndarray(
        (batch_heads, seq_padded), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="fx_c_row"
    )
    num_head_iters = num_heads_per_shard + (1 if (has_remainder and shard_id == 0) else 0)
    q_sumsq_ckpt = allocator.get_current_address()
    q_sumsq_sb = allocator.alloc_sbuf_tensor(
        shape=(_P, num_token_groups),
        dtype=nl.float32,
        block_dim=[num_head_iters],
        num_free_tiles=[num_head_iters],
    )

    _transpose_qk_pipelined(
        k_bufs[0],
        k,
        batch_heads,
        head_dim,
        seqlen,
        num_heads_per_shard,
        head_offset,
        has_remainder,
        shard_id,
        allocator,
        k_heads,
        k_maxsq_out=local_k_maxsq,
        dst_stride=seq_padded,
    )
    if reduce_k_max:
        if num_shards > 1:
            nisa.core_barrier(local_k_maxsq, (0, 1))
        ncc.all_reduce(
            dsts=[global_k_maxsq], srcs=[local_k_maxsq], op=nl.maximum, replica_group=replica_group
        )
        k_maxsq = global_k_maxsq
    else:
        k_maxsq = local_k_maxsq

    q_transposed = nl.ndarray(
        (batch_heads, head_dim, seqlen), dtype=q.dtype, buffer=nl.shared_hbm, name="fx_q_t"
    )
    _transpose_qk_pipelined(
        q_transposed,
        q,
        batch_heads,
        head_dim,
        seqlen,
        num_heads_per_shard,
        head_offset,
        has_remainder,
        shard_id,
        allocator,
        q_heads,
        q_sumsq_sb=q_sumsq_sb,
    )
    _copy_batch_sharded(
        v_bufs[0][:, :seqlen, :],
        v,
        batch_heads,
        num_heads_per_shard,
        head_offset,
        has_remainder,
        shard_id,
        q_heads,
        head_dim,
        seqlen,
    )
    if kv_pad_rows > 0:
        _zero_kv_pad(k_bufs[0], v_bufs[0], seqlen, seq_padded, head_dim, runs, allocator)

    if num_shards > 1:
        nisa.core_barrier(q_transposed, (0, 1))
        nisa.core_barrier(k_bufs[0], (0, 1))
        nisa.core_barrier(v_bufs[0], (0, 1))

    _assemble_c_row(q_sumsq_sb, k_maxsq, c_row, softmax_scale, seqlen, runs, allocator)
    allocator.set_current_address(q_sumsq_ckpt)
    if num_shards > 1:
        nisa.core_barrier(c_row, (0, 1))

    tile_allocator = None
    unit_bases = None
    norm_base = allocator.get_current_address()
    units_per_step = 0
    if pin_tile_sbuf:
        tile_allocator = allocator
        unit_bases = [
            norm_base + _NORM_SCRATCH_BYTES + idx * _UNIT_REGION_BYTES
            for idx in range(_NUM_UNIT_REGIONS)
        ]
        kernel_assert(
            unit_bases[-1] + _UNIT_REGION_BYTES <= nl.tile_size.sbuf_fmax_bytes,
            "pinned ring tile regions exceed SBUF",
        )
        units_per_step = chunk_width * div_ceil(row_end - row_start, _max_sq_block_size())

    for chunk in _head_chunks(batch_heads, chunk_width):
        q_rows = q_transposed[chunk.start : chunk.end, :, row_start:row_end]
        c_rows = c_row[chunk.start : chunk.end, row_start:row_end]

        if num_workers > 1:
            for prefetch_step in range(1, prefetch_depth + 1):
                _issue_kv_permute(
                    k_bufs[(prefetch_step - 1) % num_kv_buf][chunk.start : chunk.end],
                    k_bufs[prefetch_step % num_kv_buf][chunk.start : chunk.end],
                    v_bufs[(prefetch_step - 1) % num_kv_buf][chunk.start : chunk.end],
                    v_bufs[prefetch_step % num_kv_buf][chunk.start : chunk.end],
                    replica_group,
                )

        if num_workers == 1:
            _run_last_step_and_normalize(
                q_rows,
                k_bufs[0],
                v_bufs[0],
                o_aug_acc_sb,
                c_rows,
                o,
                chunk,
                row_start,
                row_end,
                head_dim,
                softmax_scale,
                False,
                kv_pad_rows,
                lse_dtype,
                allocator,
                prologue_depth,
                tile_allocator,
                unit_bases,
                norm_base,
            )
        else:
            _run_partial_step(
                q_rows,
                k_bufs[0],
                v_bufs[0],
                o_aug_acc_sb,
                c_rows,
                chunk,
                softmax_scale,
                False,
                kv_pad_rows,
                prologue_depth,
                tile_allocator,
                unit_bases,
            )
            step_idx = 0
            for ring_step in nl.sequential_range(1, num_workers):
                step_idx += 1
                consume_idx = step_idx % num_kv_buf
                produce_step = step_idx + prefetch_depth
                if produce_step <= num_workers - 1:
                    _issue_kv_permute(
                        k_bufs[(produce_step - 1) % num_kv_buf][chunk.start : chunk.end],
                        k_bufs[produce_step % num_kv_buf][chunk.start : chunk.end],
                        v_bufs[(produce_step - 1) % num_kv_buf][chunk.start : chunk.end],
                        v_bufs[produce_step % num_kv_buf][chunk.start : chunk.end],
                        replica_group,
                    )
                if step_idx == num_workers - 1:
                    _run_last_step_and_normalize(
                        q_rows,
                        k_bufs[consume_idx],
                        v_bufs[consume_idx],
                        o_aug_acc_sb,
                        c_rows,
                        o,
                        chunk,
                        row_start,
                        row_end,
                        head_dim,
                        softmax_scale,
                        True,
                        kv_pad_rows,
                        lse_dtype,
                        allocator,
                        prologue_depth,
                        tile_allocator,
                        _rotated(unit_bases, step_idx * units_per_step),
                        norm_base,
                    )
                else:
                    _run_partial_step(
                        q_rows,
                        k_bufs[consume_idx],
                        v_bufs[consume_idx],
                        o_aug_acc_sb,
                        c_rows,
                        chunk,
                        softmax_scale,
                        True,
                        kv_pad_rows,
                        prologue_depth,
                        tile_allocator,
                        _rotated(unit_bases, step_idx * units_per_step),
                    )

    return o.reshape((batch, q_heads, seqlen, head_dim))
