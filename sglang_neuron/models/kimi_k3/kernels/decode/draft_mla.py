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

"""Kimi-K3 DSpark MLA drafter kernels (TP32 = CP32, LNC1, Trn3).

The drafter has 5 dense MLA layers, a rank-256 Markov head, and the target embedding and lm_head. One
draft step drafts 7 tokens for each of B sequences of a DP replica, on one TP32 rank. The pieces:

- ``draft_context``: from the verify rows' aux states, build ``x = context_norm(context_proj(cat
  aux))`` and owner-write each layer's cache rows into the 5 rider pools, in the target MLA page format.
- ``draft_layer``: one dense MLA layer. ``h += mla_block_attention(input_norm(h))`` in draft mode
  (``write_cache=False``), then ``h += silu_mlp(post_attention_norm(h))`` with the intermediate TP32
  sharded (448 per rank).
- ``draft_tail``: ``final_norm``, the lm_head shard, and the sequential Markov greedy. Row ``i``'s
  bias is ``markov_w2 @ markov_w1[prev]`` with ``prev`` the bonus for row 0, then the previous draft.
- ``kimi_k3_draft_mla_step``: the context stage, the embed gather, 5 drafter layers, then the tail, in one call.

The references and the BF16 rounding points are ``draft_mla_torch``; the host weight layouts are
``draft_mla_host``.
"""

import math
from contextlib import contextmanager, nullcontext

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from .utils.kernel_assert import kernel_assert
from .common import (
    P_MAX,
    PSUM_BANK_FP32,
    check_pair_group,
    fp8_page_dtype,
    fp8_page_max,
    is_trn3,
    load_vector_hT,
    new_ones,
    pair_shared_buffer,
    rmsnorm_hT,
    store_hT_rows,
)
from .tkg_sbuf_alloc import AUTO_SBUF, ExplicitSbuf, SbufTiles, explicit_ones, install_identities
from .tkg_mla_block import MLA_PASS_ROWS, mla_block_attention, mla_block_collective_hbm, mla_block_collective_sources
from .mla_decode_torch import CACHE_WIDTH, HIDDEN, KV_LORA, PAGE_WIDTH, ROPE, TP

_H_TILES = HIDDEN // P_MAX
_AUX_LAYERS = 5
_AUX_TILES = _AUX_LAYERS * _H_TILES
_AUX_VERIFY = 8
"""Verify rows per sequence that the context stage writes (the anchor and 7 drafts of the previous step)."""
_H_SHARD = HIDDEN // TP
_DRAFT_SHARD = 14336 // TP
_DRAFT_HEADS = 64
"""Drafter attention heads. At CP32 one rank owns 2 (``32 * 2``); at CP128 rank g owns head g (g < 64)."""
_BLOCK = 7
_LOOP_FP8_GROUPS = 4
"""Prior-key groups per loop iteration on the CP32 drafter loop path. Half the FP8 default caps the MLA
loop-group buffers so the explicit SBUF region fits at B up to 8. The groups attend in the same order,
only fewer per iteration, so the output is bitwise identical. Only the CP32 loop path sets it; the shared
default and all other callers keep ``loop_group_cap=0`` (the pool-dtype default)."""
_MASK_TOKEN = 163837
_EPS = 1e-5
_BF16_KEY_MAX = 65536.0
"""Clamp bound of a stored BF16 cache row, as ``tkg_mla_paged``: finite rows are far below it."""
_PAGE_BLOCK = 128
# One local block spans TP * 128 = 4096 global positions (_WIN_SHIFT = log2(4096)).
_WIN_BLOCK = TP * _PAGE_BLOCK
_WIN_SHIFT = 12
# One tkg_mla_block key group spans 4 local blocks, so 4 window columns (``win_trim``).
_GROUP_COLS = 4
_K1_HAS_KEY_START = True
"""``mla_block_attention`` takes ``key_start`` as an optional trailing argument."""
_KV_A_SHARD = CACHE_WIDTH // TP
"""kv_a output columns of one rank: 576 over 32 ranks is 18. The TP all-gather rebuilds the 576 row."""

_DTYPE_BYTES = {nl.bfloat16: 2, nl.float32: 4}

DRAFT_PAIR_SHARED_CC_DST_SITES = ("draft_ar",)
"""The collective destinations that the draft step's ``pair_shared_cc_dsts`` can put in LNC1 shared HBM: ``draft_ar``,
the HBM destination arena of its all-reduces over the TP group (``draft_hbm_ar_dst``: the dense MLP, the Markov rows
and the vocab-shard block embedding). The TP group holds both cores of every HBM stack, so the two cores get the same
bytes, and an all-reduce is ordered with the other core of the stack by its data (``common.PAIR_SHARED_CC_DST_SITES``).
"""


def check_draft_pair_shared(sites) -> bool:
    """Return whether the draft step's ``pair_shared_cc_dsts`` ``sites`` name ``draft_ar``; fail on any other or a
    repeated name, and off trn2 (the trn2 serving form is the one with the shared destination)."""
    sites = tuple(sites or ())
    kernel_assert(
        all(site in DRAFT_PAIR_SHARED_CC_DST_SITES for site in sites) and len(set(sites)) == len(sites),
        f"pair_shared_cc_dsts {sites} of the draft step must be distinct names of {DRAFT_PAIR_SHARED_CC_DST_SITES}",
    )
    kernel_assert(not sites or not is_trn3(), f"pair_shared_cc_dsts {sites} of the draft step is for trn2 only")
    return bool(sites)


class _ArenaPlan:
    """Column ranges of two BF16 SBUF arenas: arena 1 for the collective destinations (a tile name that
    contains ``_dst``), arena 0 for the sources and other tiles.

    ``tile`` records the next range in its arena and returns a slot index; ``view`` returns the slot's view
    from the ``(source, destination)`` arena pair. A collective's source and destination then land in
    different tensors, and each arena is reserved to its last use (the tail argmax).
    """

    def __init__(self):
        self.slots = []
        self.widths = [0, 0]

    def tile(self, shape, dtype, name=None):
        """Record a tile at the next 32-byte-aligned columns of its arena (``_dst`` -> arena 1); return slot."""
        arena = 1 if name and "_dst" in name else 0
        self.slots.append((arena, self.widths[arena], tuple(shape), dtype))
        self.widths[arena] += -(-math.prod(shape[1:]) * _DTYPE_BYTES[dtype] // 32) * 16
        return len(self.slots) - 1

    def columns(self, slot):
        """Return the ``[start, end)`` BF16 columns of a slot in its arena, with the alignment padding."""
        _, offset, shape, dtype = self.slots[slot]
        return offset, offset + -(-math.prod(shape[1:]) * _DTYPE_BYTES[dtype] // 32) * 16

    def view(self, arenas, slot):
        """Return the arena view of a slot (``arenas`` is the ``(source, destination)`` tensor pair)."""
        arena, offset, shape, dtype = self.slots[slot]
        base = arenas[arena] if dtype == nl.bfloat16 else arenas[arena].view(dtype)
        start = offset * 2 // _DTYPE_BYTES[dtype]
        view = base[0 : shape[0], start : start + math.prod(shape[1:])]
        return view.reshape_dim(1, shape[1:]) if len(shape) > 2 else view


def _matmul_tok(x_hT, w_hbm, out_cols, tag, sb, k_tiles_per_dma=14, out=None, ring=False):
    """Return ``x @ w`` token-major ``[T, out_cols]`` BF16 for hT ``x`` ``[128, T, K]`` and HBM ``w`` ``[128, K, out]``.

    ``x_hT`` is the stationary operand (T columns), so the result is token-major and each matmul output
    stays inside one PSUM bank. The weight loads in K-tile chunks to bound the live SBUF. ``out`` writes
    into a caller tile (a persistent collective source) instead of a scratch tile.

    ``ring`` reuses a 2-deep ring of weight-chunk tiles across all chunks, in place of one tile per chunk
    (which the explicit stack allocator holds to the layer frame end). The DMA, matmul and accumulation
    order do not change, so the result is bitwise identical. The caller sets it only on the drafter loop
    path, where the per-chunk tiles would otherwise fill the explicit SBUF region.
    """
    _, total_rows, k_tiles = x_hT.shape
    hwdge = nisa.dge_mode.hwdge
    sync = nisa.engine.sync
    out = sb.tile((total_rows, out_cols), nl.bfloat16) if out is None else out
    ring_bufs = None
    if ring:
        ring_width = min(PSUM_BANK_FP32, out_cols)
        ring_bufs = [sb.tile((P_MAX, k_tiles_per_dma, ring_width), nl.bfloat16) for _ in range(2)]
    ri = 0
    for c0 in range(0, out_cols, PSUM_BANK_FP32):
        width = min(PSUM_BANK_FP32, out_cols - c0)
        psum = nl.ndarray((total_rows, width), dtype=nl.float32, buffer=nl.psum)
        for k0 in range(0, k_tiles, k_tiles_per_dma):
            chunk = min(k_tiles_per_dma, k_tiles - k0)
            if ring_bufs is None:
                w_sb = sb.tile((P_MAX, chunk, width), nl.bfloat16)
            else:
                w_sb = ring_bufs[ri % 2][0:P_MAX, 0:chunk, 0:width]
                ri += 1
            nisa.dma_copy(
                dst=w_sb,
                src=w_hbm[0:P_MAX, k0 : k0 + chunk, c0 : c0 + width],
                dge_mode=hwdge,
                engine=sync,
                name=f"{tag}_w_{c0}_{k0}",
            )
            for dk in range(chunk):
                k = k0 + dk
                nisa.nc_matmul(
                    dst=psum,
                    stationary=x_hT[0:P_MAX, 0:total_rows, k],
                    moving=w_sb[0:P_MAX, dk, 0:width],
                    accumulate=k > 0,
                )
        nisa.tensor_copy(dst=out[0:total_rows, c0 : c0 + width], src=psum)
    return out


def _rows_to_hT(rows_sb, sb):
    """Transpose a token-major SBUF tile ``[T, W]`` into the hT tile ``[128, T, W/128]`` (W a multiple of 128)."""
    total_rows, width = rows_sb.shape
    n_tiles = width // P_MAX
    out = sb.tile((P_MAX, total_rows, n_tiles), rows_sb.dtype)
    for j in range(n_tiles):
        psum = nl.ndarray((P_MAX, total_rows), dtype=rows_sb.dtype, buffer=nl.psum)
        nisa.nc_transpose(dst=psum, data=rows_sb[0:total_rows, j * P_MAX : (j + 1) * P_MAX])
        nisa.tensor_copy(dst=out[0:P_MAX, 0:total_rows, j], src=psum)
    return out


def _apply_rope(x, cos_x, sin_x, rows, sb):
    """Rotate the GPT-J pairs of ``x`` ``[rows, 64]`` BF16 in place, in FP32.

    ``cos_x`` and ``sin_x`` are ``[rows, 64]`` FP32, pre-expanded for the pairs. The rotation is
    ``x * cos + rot(x) * sin`` with ``rot(x)[2i] = -x[2i+1]`` and ``rot(x)[2i+1] = x[2i]``.
    """
    fp32 = nl.float32
    xf = sb.tile((rows, ROPE), fp32)
    nisa.tensor_copy(dst=xf, src=x)
    pairs = xf.reshape((rows, ROPE // 2, 2))
    rot = sb.tile((rows, ROPE // 2, 2), fp32)
    nisa.tensor_scalar(dst=rot[:, :, 0:1], data=pairs[:, :, 1:2], op0=nl.multiply, operand0=-1.0)
    nisa.tensor_copy(dst=rot[:, :, 1:2], src=pairs[:, :, 0:1])
    rot_flat = rot.reshape((rows, ROPE))
    out = sb.tile((rows, ROPE), fp32)
    nisa.tensor_tensor(dst=out, data1=xf, data2=cos_x, op=nl.multiply)
    nisa.tensor_tensor(dst=rot_flat, data1=rot_flat, data2=sin_x, op=nl.multiply)
    nisa.tensor_tensor(dst=out, data1=out, data2=rot_flat, op=nl.add)
    nisa.tensor_copy(dst=x, src=out)


def _own_positions(pos_row, q_rows, tokens, sb):
    """Return ``[rows, 1]`` INT32 own positions ``pos_row[0, b] + j`` for row ``r = j * tokens + b``."""
    int32 = nl.int32
    fp32 = nl.float32
    total_rows = q_rows * tokens
    j3 = sb.tile((1, q_rows, tokens), int32)
    nisa.iota(dst=j3, pattern=[[1, q_rows], [0, tokens]], offset=0)
    own3 = sb.tile((1, q_rows, tokens), int32)
    nisa.tensor_tensor(dst=own3, data1=j3, data2=pos_row.reshape((1, 1, tokens)).broadcast(1, q_rows), op=nl.add)
    own_f = sb.tile((1, total_rows), fp32)
    nisa.tensor_copy(dst=own_f, src=own3.reshape((1, total_rows)))
    own_psum = nl.ndarray((total_rows, 1), dtype=fp32, buffer=nl.psum)
    nisa.nc_transpose(dst=own_psum, data=own_f)
    own = sb.tile((total_rows, 1), int32)
    nisa.tensor_copy(dst=own, src=own_psum)
    return own


def _gather_window(window, flat_idx, total_rows, tag, sb):
    """Gather ``window_flat[flat_idx[r]]`` into ``[rows, 128]`` FP32. ``window`` is ``[B, W, 128]`` FP32 HBM."""
    hwdge = nisa.dge_mode.hwdge
    sync = nisa.engine.sync
    width = window.shape[2]
    flat = window.reshape((window.shape[0] * window.shape[1], width))
    out = sb.tile((total_rows, width), nl.float32)
    for r in range(total_rows):
        nisa.dma_copy(
            dst=out[r : r + 1, 0:width],
            src=flat.ap(
                pattern=[[width, 1], [1, width]], offset=0, scalar_offset=flat_idx[r : r + 1, 0:1], indirect_dim=0
            ),
            dge_mode=hwdge,
            engine=sync,
            name=f"{tag}_rope_gather_{r}",
        )
    return out


def _write_rows(total_rows, tokens, q_rows, slot, position, rank_info, tag, sb, cp=TP):
    """Return the per-row pool write row ``[T, 1]`` INT32: the slot on the owner rank, else 0 (null page).

    The owner of query row ``r = j * B + b`` is rank ``((position[b] + j) >> 7) & (cp - 1)``. The write
    row is the slot when the rank owns the row and the slot is a real page row (>= 128), else row 0.
    """
    fp32 = nl.float32
    int32 = nl.int32
    hwdge = nisa.dge_mode.hwdge
    sync = nisa.engine.sync
    pos_row = sb.tile((1, tokens), int32)
    nisa.dma_copy(
        dst=pos_row,
        src=position.ap(pattern=[[tokens, 1], [1, tokens]], offset=0),
        dge_mode=hwdge,
        engine=sync,
        name=f"{tag}_pos",
    )
    own_pos = _own_positions(pos_row, q_rows, tokens, sb)
    owner_block = sb.tile((total_rows, 1), int32)
    nisa.tensor_scalar(
        dst=owner_block, data=own_pos, op0=nl.right_shift, operand0=7, op1=nl.bitwise_and, operand1=cp - 1
    )
    slot_i = sb.tile((total_rows, 1), int32)
    nisa.dma_copy(dst=slot_i, src=slot[0:total_rows, 0:1], dge_mode=hwdge, engine=sync, name=f"{tag}_slot")
    rank_i = sb.tile((total_rows, 1), int32)
    nisa.dma_copy(
        dst=rank_i, src=rank_info[0:1, 2:3].broadcast(0, total_rows), dge_mode=hwdge, engine=sync, name=f"{tag}_rank"
    )
    token_f = sb.tile((total_rows, 3), fp32)
    nisa.tensor_copy(dst=token_f[0:total_rows, 0:1], src=owner_block)
    nisa.tensor_copy(dst=token_f[0:total_rows, 1:2], src=slot_i)
    nisa.tensor_copy(dst=token_f[0:total_rows, 2:3], src=rank_i)
    owner = sb.tile((total_rows, 1), fp32)
    nisa.tensor_scalar(dst=owner, data=token_f[0:total_rows, 0:1], op0=nl.equal, operand0=token_f[0:total_rows, 2:3])
    write_ok = sb.tile((total_rows, 1), fp32)
    nisa.tensor_scalar(
        dst=write_ok,
        data=token_f[0:total_rows, 1:2],
        op0=nl.greater_equal,
        operand0=float(_PAGE_BLOCK),
        op1=nl.multiply,
        operand1=owner,
    )
    write_mask = sb.tile((total_rows, 1), int32)
    nisa.tensor_scalar(dst=write_mask, data=write_ok, op0=nl.multiply, operand0=-1.0)
    write_row = sb.tile((total_rows, 1), int32)
    nisa.tensor_tensor(dst=write_row, data1=slot_i, data2=write_mask, op=nl.bitwise_and)
    return write_row


def _context_proj_streamed(aux, context_proj, total_rows, out_cols, tag, sb, out=None):
    """Context projection ``cat(aux) @ context_proj`` without materializing the full aux hT tile.

    Equivalent to ``_matmul_tok(_build_aux_hT(aux, total_rows), context_proj, out_cols)``: the matmul
    accumulates over the same ``_AUX_TILES`` input tiles in aux-layer-major order, so the result is
    bitwise identical. The difference is that only one aux layer's hT tile ``[128, total_rows, 56]`` is
    live at a time instead of the full ``[128, total_rows, 280]`` tile, which keeps the context stage
    off the SBUF peak. Used for large batch (tokens > 4), where the full aux tile overflows SBUF
    (NCC_EGCA111). ``aux`` is ``[5, total_rows, H]`` BF16 HBM; ``context_proj`` is ``[128, 280, out]``.
    """
    bf16 = nl.bfloat16
    fp32 = nl.float32
    hwdge = nisa.dge_mode.hwdge
    sync = nisa.engine.sync
    aux_flat = aux.reshape((_AUX_LAYERS * total_rows, HIDDEN))
    out = sb.tile((total_rows, out_cols), bf16) if out is None else out
    psum = nl.ndarray((total_rows, out_cols), dtype=fp32, buffer=nl.psum)
    for a in range(_AUX_LAYERS):
        aux_a = sb.tile((P_MAX, total_rows, _H_TILES), bf16)
        _rows_into_hT(aux_flat, a * total_rows, total_rows, aux_a, 0, f"{tag}_aux_{a}", sb)
        for j0 in range(0, _H_TILES, 14):
            chunk = min(14, _H_TILES - j0)
            w_sb = sb.tile((P_MAX, chunk, out_cols), bf16)
            nisa.dma_copy(
                dst=w_sb,
                src=context_proj[0:P_MAX, a * _H_TILES + j0 : a * _H_TILES + j0 + chunk, 0:out_cols],
                dge_mode=hwdge,
                engine=sync,
                name=f"{tag}_w_{a}_{j0}",
            )
            for dj in range(chunk):
                k = a * _H_TILES + j0 + dj
                nisa.nc_matmul(
                    dst=psum,
                    stationary=aux_a[0:P_MAX, 0:total_rows, j0 + dj],
                    moving=w_sb[0:P_MAX, dj, 0:out_cols],
                    accumulate=k > 0,
                )
    nisa.tensor_copy(dst=out[0:total_rows, 0:out_cols], src=psum)
    return out


def draft_context(
    aux_hT,
    context_proj,
    context_norm,
    kv_a_proj,
    kv_a_norm,
    pools,
    slot,
    position,
    rope_rows,
    rank_info,
    tp_group,
    tag,
    layers=5,
    fp8_pages=False,
    sb=None,
    sources=None,
    cp=TP,
    aux_raw=None,
):
    """Context stage: build the context feature and owner-write each layer's cache rows.

    Args:
        aux_hT: ``[128, 8B, 280]`` BF16 SBUF cat(aux) in hT, token-major (row ``j * B + b``).
        context_proj: ``[128, 280, 224]`` BF16 HBM ``[in, out]`` hT-tiled output shard of this rank.
        context_norm: ``[7168]`` norm weight HBM.
        kv_a_proj: ``[5, 128, 56, 576]`` BF16 HBM ``[in, out]`` hT-tiled kv_a projection (replicated).
        kv_a_norm: ``[5, 512]`` HBM latent norm weight (replicated).
        pools: list of 5 rider pools ``[P, 73728]`` BF16 or FLOAT8_E4M3FN HBM.
        slot: ``[8B, 1]`` INT32 HBM verify write slots.
        position: ``[B, 1]`` INT32 HBM verify row-0 position of each sequence.
        rope_rows: ``[8B, 128]`` FP32 HBM GPT-J ``[cos | sin]`` of the verify positions.
        rank_info, tp_group, tag: the rank id, the TP group, and a unique name prefix.
        layers, fp8_pages, sb: the layer count, the pool dtype flag, and the SBUF allocator.
    """
    sb = AUTO_SBUF if sb is None else sb
    fp32 = nl.float32
    bf16 = nl.bfloat16
    hwdge = nisa.dge_mode.hwdge
    sync = nisa.engine.sync
    total_rows = aux_hT.shape[1] if aux_hT is not None else aux_raw.shape[1]
    tokens = position.shape[0]
    verify_rows = total_rows // tokens
    ones = new_ones()
    # CP-dependent shard sizes. At CP32 the context columns and the kv_a row split are contiguous
    # (CACHE_WIDTH // 32 = 18). At CP128 the kv_a shard is the target split: four kv latent features
    # (128 c + g, c = 0..3) and one k_pe feature (g for g < 64, a zero column above), so five columns.
    h_shard = HIDDEN // cp
    kv_latent_local = KV_LORA // cp
    kv_a_shard = CACHE_WIDTH // cp if cp == TP else kv_latent_local + 1

    # Context feature x = context_norm(context_proj(aux)); the output columns gather over the world. The AG
    # source and destination are persistent tiles (sources) when the step provides them, so a later
    # body cannot overwrite the source while a peer still reads it.
    ctx_src = sources["ctx_src"] if sources is not None else None
    if aux_raw is not None:
        fc_local = _context_proj_streamed(
            aux_raw, context_proj, total_rows, h_shard, f"{tag}_ctx_proj", sb, out=ctx_src
        )
    else:
        fc_local = _matmul_tok(aux_hT, context_proj, h_shard, f"{tag}_ctx_proj", sb, out=ctx_src)
    fc_full = sources["ctx_dst"] if sources is not None else sb.tile((total_rows, HIDDEN), bf16)
    ncc.all_gather(srcs=[fc_local], dsts=[fc_full], replica_group=tp_group, collective_dim=1, name=f"{tag}_ctx_gather")
    norm_hT = load_vector_hT(context_norm, dtype=fp32)
    x_hT = rmsnorm_hT(_rows_to_hT(fc_full, sb), norm_hT, _EPS, ones, sb=sb)

    rope_sb = sb.tile((total_rows, 2 * ROPE), fp32)
    nisa.dma_copy(
        dst=rope_sb, src=rope_rows[0:total_rows, 0 : 2 * ROPE], dge_mode=hwdge, engine=sync, name=f"{tag}_ctx_rope"
    )
    cos_x = rope_sb[0:total_rows, 0:ROPE]
    sin_x = rope_sb[0:total_rows, ROPE : 2 * ROPE]

    write_row = _write_rows(total_rows, tokens, verify_rows, slot, position, rank_info, f"{tag}_ctx", sb, cp=cp)
    write_index = _distinct_null_rows(write_row, total_rows, sb)
    # Batch the 5 layers' kv_a projections into ONE all-gather. Each layer's shard is packed layer-major
    # in the source; the gather concatenates the world ranks, so the destination reads as
    # [T, rank, layer, shard] and each layer's 576 row de-interleaves with one local copy. A single world
    # collective (not five) means no later layer's weight DMA reuses an earlier layer's gather source.
    kva_src = sources["kva_src"] if sources is not None else sb.tile((total_rows, layers * kv_a_shard), bf16)
    for layer in range(layers):
        _matmul_tok(
            x_hT,
            kv_a_proj[layer],
            kv_a_shard,
            f"{tag}_kv_a_{layer}",
            sb,
            out=kva_src[0:total_rows, layer * kv_a_shard : (layer + 1) * kv_a_shard],
        )
    kva_dst = sources["kva_dst"] if sources is not None else sb.tile((total_rows, cp * layers * kv_a_shard), bf16)
    ncc.all_gather(srcs=[kva_src], dsts=[kva_dst], replica_group=tp_group, collective_dim=1, name=f"{tag}_kv_a_gather")
    gathered = kva_dst.reshape((total_rows, cp, layers, kv_a_shard))
    for layer in range(layers):
        kv_a = sb.tile((total_rows, CACHE_WIDTH), bf16)
        if cp == TP:
            nisa.tensor_copy(
                dst=kv_a.reshape((total_rows, cp, kv_a_shard)),
                src=gathered[0:total_rows, 0:cp, layer, 0:kv_a_shard],
            )
        else:
            # De-interleave the target split: kv latent feature cp*c + r is column c of rank r, and k_pe
            # feature r is column kv_latent_local of rank r (r < 64). Rows at and above 64 hold a zero k_pe.
            g_layer = gathered[0:total_rows, 0:cp, layer, 0:kv_a_shard]
            nisa.tensor_copy(
                dst=kv_a[0:total_rows, 0:KV_LORA].reshape((total_rows, kv_latent_local, cp)),
                src=g_layer[0:total_rows, 0:cp, 0:kv_latent_local].permute((0, 2, 1)),
            )
            nisa.tensor_copy(
                dst=kv_a[0:total_rows, KV_LORA:CACHE_WIDTH].reshape((total_rows, 1, ROPE)),
                src=g_layer[0:total_rows, 0:ROPE, kv_latent_local:kv_a_shard].permute((0, 2, 1)),
            )
        cache_row = _cache_row(kv_a, kv_a_norm, layer, cos_x, sin_x, total_rows, sb)
        pool = pools[layer]
        pool_rows = pool.reshape((pool.shape[0] * _PAGE_BLOCK, CACHE_WIDTH))
        # Clamp the stored row (FP8 to its largest finite value in a new tile, BF16 in place), so it is stored finite.
        bf16_pool = pool.dtype == bf16
        write_rows = cache_row if bf16_pool else sb.tile((total_rows, CACHE_WIDTH), pool.dtype)
        bound = _BF16_KEY_MAX if bf16_pool else fp8_page_max()
        nisa.tensor_scalar(
            dst=write_rows, data=cache_row, op0=nl.minimum, operand0=bound, op1=nl.maximum, operand1=-bound
        )
        _write_context_rows(pool_rows, write_index, write_rows, total_rows, f"{tag}_ctx_write_{layer}")


def _distinct_null_rows(write_row, total_rows, sb):
    """Return the ``[T, 1]`` INT32 write rows of one vector-indirect write: ``write_row`` (``_write_rows``), with row ``r``
    of the null page for each row ``r`` that writes no owned slot (row 0), so no two rows of the write share a target.
    The null page holds other bytes; no read takes them (``batched_cache_write`` of ``tkg_mla_block``)."""
    fp32 = nl.float32
    kernel_assert(total_rows <= _PAGE_BLOCK, f"one context write takes at most {_PAGE_BLOCK} rows, got {total_rows}")
    row_f = sb.tile((total_rows, 1), fp32)
    nisa.tensor_copy(dst=row_f, src=write_row)
    null_flag = sb.tile((total_rows, 1), fp32)
    nisa.tensor_scalar(dst=null_flag, data=row_f, op0=nl.equal, operand0=0.0)
    null_row = sb.tile((total_rows, 1), nl.int32)
    nisa.iota(dst=null_row, pattern=[[0, 1]], offset=0, channel_multiplier=1)
    null_off = sb.tile((total_rows, 1), fp32)
    nisa.tensor_tensor(dst=null_off, data1=null_flag, data2=null_row, op=nl.multiply)
    write_index = sb.tile((total_rows, 1), nl.int32)
    nisa.tensor_tensor(dst=write_index, data1=write_row, data2=null_off, op=nl.add)
    return write_index


def _write_context_rows(pool_rows, write_index, write_rows, total_rows, name):
    """Write the ``[T, 576]`` rows ``write_rows`` at the pool rows ``write_index`` (``_distinct_null_rows``) of the HBM
    ``[rows, 576]`` pool ``pool_rows`` with one vector-indirect DMA (SWDGE). Per-row dynamic-offset writes of one pool
    run one after another (the compiler orders them), each with its own cost."""
    kernel_assert(pool_rows.shape[0] < 1 << 24, "the context write needs fewer than 2**24 pool rows")
    nisa.dma_copy(
        dst=pool_rows.ap(
            pattern=[[CACHE_WIDTH, total_rows], [1, CACHE_WIDTH]],
            offset=0,
            vector_offset=write_index[0:total_rows, 0:1],
            indirect_dim=0,
        ),
        src=write_rows[0:total_rows, 0:CACHE_WIDTH],
        dge_mode=nisa.dge_mode.swdge,
        name=name,
    )


def _cache_row(kv_a, kv_a_norm, layer, cos_x, sin_x, total_rows, sb):
    """Return the ``[T, 576]`` BF16 cache row ``[rms(latent) | rope(k_pe)]`` from a kv_a projection ``[T, 576]``."""
    fp32 = nl.float32
    bf16 = nl.bfloat16
    hwdge = nisa.dge_mode.hwdge
    sync = nisa.engine.sync
    cache_row = sb.tile((total_rows, CACHE_WIDTH), bf16)
    sq = sb.tile((total_rows, 1), fp32)
    nisa.scalar_tensor_tensor(
        dst=sb.tile((total_rows, KV_LORA), fp32),
        data=kv_a[0:total_rows, 0:KV_LORA],
        op0=nl.multiply,
        operand0=1.0,
        op1=nl.multiply,
        operand1=kv_a[0:total_rows, 0:KV_LORA],
        reduce_op=nl.add,
        reduce_res=sq,
    )
    rstd = sb.tile((total_rows, 1), fp32)
    nisa.activation(dst=rstd, op=nl.rsqrt, data=sq, scale=1.0 / float(KV_LORA), bias=_EPS)
    normed = sb.tile((total_rows, KV_LORA), bf16)
    nisa.tensor_scalar(dst=normed, data=kv_a[0:total_rows, 0:KV_LORA], op0=nl.multiply, operand0=rstd)
    norm_w = sb.tile((total_rows, KV_LORA), bf16)
    nisa.dma_copy(
        dst=norm_w,
        src=kv_a_norm.ap(pattern=[[0, total_rows], [1, KV_LORA]], offset=layer * KV_LORA),
        dge_mode=hwdge,
        engine=sync,
        name=f"ctx_norm_w_{layer}",
    )
    nisa.tensor_tensor(dst=cache_row[0:total_rows, 0:KV_LORA], data1=normed, data2=norm_w, op=nl.multiply)
    nisa.tensor_copy(dst=cache_row[0:total_rows, KV_LORA:CACHE_WIDTH], src=kv_a[0:total_rows, KV_LORA:CACHE_WIDTH])
    _apply_rope(cache_row[0:total_rows, KV_LORA:CACHE_WIDTH], cos_x, sin_x, total_rows, sb)
    return cache_row


_MARKOV_RANK = 256
_MARKOV_TILES = _MARKOV_RANK // P_MAX
_NO_TOKEN = float(1 << 22)
"""A vocab id above every real id. FP32 holds the ids and this value exactly."""
_FINITE_MAX = 3.4e38
"""Above every finite candidate value and q of a drafter row, below +inf: the bound of the row's finite test."""


def _global_argmax(logits, offset_val, tp_group, tp_size, tag, sb, ag_src=None, ag_dst=None):
    """Return the global greedy token ``[B, 1]`` INT32 of the local shard logits ``[B, V_s]`` BF16.

    The id of a candidate is its local index plus ``offset_val`` (this rank's vocab base). The gather
    carries ``(value, global id)`` of each rank; the largest value wins, the smallest id breaks ties.
    ``ag_src``/``ag_dst`` are the persistent gather source/destination. The token is in ``[0, tp_size * V_s)``;
    a row with NaN logits also gives an id in the vocab.
    """
    fp32 = nl.float32
    rows, v_s = logits.shape
    top8 = sb.tile((rows, 8), logits.dtype)
    nisa.max8(dst=top8, src=logits)
    idx8 = sb.tile((rows, 8), nl.uint32)
    nisa.nc_find_index8(dst=idx8, data=logits, vals=top8)
    local = ag_src if ag_src is not None else sb.tile((rows, 2), fp32)
    nisa.tensor_copy(dst=local[0:rows, 0:1], src=top8[0:rows, 0:1])
    gid = sb.tile((rows, 1), fp32)
    nisa.tensor_copy(dst=gid, src=idx8[0:rows, 0:1])
    nisa.tensor_scalar(dst=local[0:rows, 1:2], data=gid, op0=nl.add, operand0=offset_val)
    gathered = ag_dst if ag_dst is not None else sb.tile((rows, tp_size * 2), fp32)
    ncc.all_gather(srcs=[local], dsts=[gathered], replica_group=tp_group, collective_dim=1, name=f"{tag}_gather")
    pairs = gathered.reshape((rows, tp_size, 2))
    values = pairs[0:rows, 0:tp_size, 0]
    ids = sb.tile((rows, tp_size), fp32)
    nisa.tensor_copy(dst=ids, src=pairs[0:rows, 0:tp_size, 1])
    best = sb.tile((rows, 1), fp32)
    nisa.tensor_reduce(dst=best, data=values, op=nl.maximum, axis=(1,))
    is_best = sb.tile((rows, tp_size), fp32)
    nisa.tensor_scalar(dst=is_best, data=values, op0=nl.equal, operand0=best)
    penalty = sb.tile((rows, tp_size), fp32)
    nisa.tensor_scalar(dst=penalty, data=is_best, op0=nl.multiply, operand0=-_NO_TOKEN, op1=nl.add, operand1=_NO_TOKEN)
    nisa.tensor_tensor(dst=ids, data1=ids, data2=penalty, op=nl.add)
    token_f = sb.tile((rows, 1), fp32)
    nisa.tensor_reduce(dst=token_f, data=ids, op=nl.minimum, axis=(1,))
    token = sb.tile((rows, 1), nl.int32)
    # A row with no match (NaN logits) has the penalty on every id; the clamp keeps it in the vocab.
    nisa.tensor_scalar(
        dst=token, data=token_f, op0=nl.minimum, operand0=float(tp_size * v_s - 1), op1=nl.maximum, operand1=0.0
    )
    return token


_SAMPLE_K = 256
"""Candidates of a sampled draft slot (the sampler's ``max_top_k``) at CP32 and CP128: the top ``256 // cp`` pairs of
each rank. ``_sample_width`` gives the count at other group sizes."""
_DST_ANCHOR = 16
"""``sample_unique_dst``: the FP32 anchor columns after the 7 per-slot gather destinations (the only local write of the
tile, before the lm_head)."""
_SAMPLING_EPS = 1e-5
"""A temperature below this is greedy (the sampler rule)."""
_U_MIN = 1.1754944e-38
"""The smallest normal FP32. A draw ``u`` below it is clamped to it (the sampler rule)."""


def _const(shape, value, sb):
    """Return an FP32 SBUF tile filled with ``value`` (``tensor_copy_predicated`` takes no scalar source)."""
    tile = sb.tile(shape, nl.float32)
    nisa.memset(dst=tile, value=value, engine=nisa.engine.vector)
    return tile


def _sample_width(cp):
    """Return the candidates of a sampled slot: ``cp`` ranks times ``min(8, 256 // cp)`` pairs (``max8`` gives 8)."""
    return cp * min(8, _SAMPLE_K // cp)


def _sampling_setup(sample_params, u_draft, tokens, k, tag, sb):
    """Load the per-row sampler inputs of a sampled draft tail once; return a dict of SBUF tiles (``k`` candidates).

    ``sample_params`` ``[B, 3]`` FP32 HBM holds the sampler columns ``[top_k, top_p, temperature]``.
    ``u_draft`` ``[B, 7]`` FP32 HBM holds the host draw of each slot (the request's seeded stream).
    """
    fp32 = nl.float32
    u8 = nl.uint8
    hwdge = nisa.dge_mode.hwdge
    sync = nisa.engine.sync
    params = sb.tile((tokens, 3), fp32)
    nisa.dma_copy(dst=params, src=sample_params[0:tokens, 0:3], dge_mode=hwdge, engine=sync, name=f"{tag}_params")
    u = sb.tile((tokens, _BLOCK), fp32)
    nisa.dma_copy(dst=u, src=u_draft[0:tokens, 0:_BLOCK], dge_mode=hwdge, engine=sync, name=f"{tag}_u")
    nisa.tensor_scalar(dst=u, data=u, op0=nl.maximum, operand0=_U_MIN, engine=nisa.engine.vector)
    top_k, top_p, temp = params[0:tokens, 0:1], params[0:tokens, 1:2], params[0:tokens, 2:3]
    sampled_f = sb.tile((tokens, 1), fp32)
    nisa.tensor_scalar(
        dst=sampled_f, data=temp, op0=nl.greater_equal, operand0=_SAMPLING_EPS, engine=nisa.engine.vector
    )
    sampled = sb.tile((tokens, 1), u8)
    nisa.tensor_copy(dst=sampled, src=sampled_f, engine=nisa.engine.vector)
    inv_t = sb.tile((tokens, 1), fp32)
    nisa.tensor_scalar(dst=inv_t, data=temp, op0=nl.maximum, operand0=_SAMPLING_EPS, engine=nisa.engine.vector)
    nisa.reciprocal(dst=inv_t, data=inv_t)
    # The descending position of the top_k cut; a top_k of 0 or less keeps all candidates.
    kth = sb.tile((tokens, 1), fp32)
    nisa.tensor_scalar(
        dst=kth, data=top_k, op0=nl.minimum, operand0=float(k), op1=nl.subtract, operand1=1.0, engine=nisa.engine.vector
    )
    no_cut = sb.tile((tokens, 1), u8)
    nisa.tensor_scalar(dst=no_cut, data=top_k, op0=nl.less_equal, operand0=0.0, engine=nisa.engine.vector)
    nisa.tensor_copy_predicated(dst=kth, src=_const((tokens, 1), float(k - 1), sb), predicate=no_cut)
    # A top_p of 1 or more keeps all candidates; the largest candidate always stays.
    tp = sb.tile((tokens, 1), fp32)
    nisa.tensor_scalar(dst=tp, data=top_p, op0=nl.maximum, operand0=1e-6, engine=nisa.engine.vector)
    full = sb.tile((tokens, 1), u8)
    nisa.tensor_scalar(dst=full, data=top_p, op0=nl.greater_equal, operand0=1.0, engine=nisa.engine.vector)
    nisa.tensor_copy_predicated(dst=tp, src=_const((tokens, 1), 2.0, sb), predicate=full)
    iota = sb.tile((tokens, k), fp32)
    nisa.iota(dst=iota, pattern=[[1, k]], offset=0)
    sampled_k = sb.tile((tokens, k), u8)
    nisa.tensor_scalar(
        dst=sampled_k,
        data=iota,
        op0=nl.multiply,
        operand0=0.0,
        op1=nl.add,
        operand1=sampled_f,
        engine=nisa.engine.vector,
    )
    onehot0 = sb.tile((tokens, k), fp32)
    nisa.tensor_scalar(dst=onehot0, data=iota, op0=nl.equal, operand0=0.0, engine=nisa.engine.vector)
    return {
        "u": u,
        "sampled": sampled,
        "sampled_k": sampled_k,
        "inv_t": inv_t,
        "kth": kth,
        "top_p": tp,
        "iota": iota,
        "onehot0": onehot0,
    }


def _draw_candidates(values, ids, setup, slot, vocab, q_probs, q_ids, sb):
    """Return the draft ``[B, 1]`` INT32 of one slot from its candidates, and write the slot's q list.

    ``values``/``ids`` ``[B, K]`` FP32 (K a multiple of 8) are the candidate logits and global ids. A greedy row (temperature below
    1e-5) takes the largest value with the smallest id. A sampled row follows the sampler rule: the top_k and
    top_p cuts are value thresholds from the descending sort of the values, then the temperature softmax, then an
    inverse-CDF draw with ``u`` over the kept candidates in gather order. ``q_probs``/``q_ids`` ``[B, K]`` FP32
    receive the distribution of the draw (a greedy row: one-hot at column 0). All operations are row-local, so a
    NaN row changes no other row; its draft is clamped into the vocab. A non-finite row (a NaN or infinite candidate
    value, or a non-finite q such as a 0/0) writes q = 0 on its whole row, not NaN: a block verify then stops at this
    draft, which keeps the target law. The test uses compares only (no invalid FP op), so a finite row keeps its bits.
    """
    fp32 = nl.float32
    rows, k = values.shape
    iota = setup["iota"]
    best = sb.tile((rows, 1), fp32)
    nisa.tensor_reduce(dst=best, data=values, op=nl.maximum, axis=(1,))
    is_best = sb.tile((rows, k), fp32)
    nisa.tensor_scalar(dst=is_best, data=values, op0=nl.equal, operand0=best, engine=nisa.engine.vector)
    penalty = sb.tile((rows, k), fp32)
    nisa.tensor_scalar(
        dst=penalty,
        data=is_best,
        op0=nl.multiply,
        operand0=-_NO_TOKEN,
        op1=nl.add,
        operand1=_NO_TOKEN,
        engine=nisa.engine.vector,
    )
    nisa.tensor_tensor(dst=penalty, data1=ids, data2=penalty, op=nl.add, engine=nisa.engine.vector)
    token_f = sb.tile((rows, 1), fp32)
    nisa.tensor_reduce(dst=token_f, data=penalty, op=nl.minimum, axis=(1,))

    # Descending sort of the values: 8 per pass, each pass replaces its 8 with -inf.
    work = sb.tile((rows, k), fp32)
    nisa.tensor_copy(dst=work, src=values, engine=nisa.engine.vector)
    desc = sb.tile((rows, k), fp32)
    for p in range(k // 8):
        nisa.max8(dst=desc[0:rows, 8 * p : 8 * p + 8], src=work)
        if p + 1 < k // 8:
            nisa.nc_match_replace8(dst=work, data=work, vals=desc[0:rows, 8 * p : 8 * p + 8], imm=float("-inf"))
    scratch = sb.tile((rows, k), fp32)
    t_k = sb.tile((rows, 1), fp32)
    nisa.range_select(
        dst=scratch,
        on_true_tile=desc,
        comp_op0=nl.greater,
        comp_op1=nl.greater_equal,
        bound0=setup["kth"],
        bound1=setup["kth"],
        reduce_op=nl.maximum,
        reduce_res=t_k,
        reduce_cmd=nisa.reduce_cmd.reset_reduce,
        range_start=1,
    )
    bias = sb.tile((rows, 1), fp32)
    nisa.tensor_scalar(
        dst=bias,
        data=best,
        op0=nl.multiply,
        operand0=setup["inv_t"],
        op1=nl.multiply,
        operand1=-1.0,
        engine=nisa.engine.vector,
    )
    e_desc = sb.tile((rows, k), fp32)
    nisa.activation(dst=e_desc, op=nl.exp, data=desc, scale=setup["inv_t"], bias=bias, engine=nisa.engine.scalar)
    kept_k = sb.tile((rows, k), fp32)
    nisa.scalar_tensor_tensor(
        dst=kept_k, data=desc, op0=nl.greater_equal, operand0=t_k, op1=nl.multiply, operand1=e_desc
    )
    total = sb.tile((rows, 1), fp32)
    nisa.tensor_reduce(dst=total, data=kept_k, op=nl.add, axis=(1,))
    cum = sb.tile((rows, k), fp32)
    nisa.tensor_scalar_cumulative(
        dst=cum, src=kept_k, op0=nl.add, op1=nl.add, imm0=0.0, reduce_cmd=nisa.reduce_cmd.reset_reduce
    )
    # top_p keeps the descending prefix whose exclusive mass is below top_p * total.
    nisa.tensor_tensor(dst=cum, data1=cum, data2=kept_k, op=nl.subtract, engine=nisa.engine.vector)
    mass = sb.tile((rows, 1), fp32)
    nisa.tensor_tensor(dst=mass, data1=setup["top_p"], data2=total, op=nl.multiply, engine=nisa.engine.vector)
    inside = sb.tile((rows, k), fp32)
    nisa.tensor_scalar(dst=inside, data=cum, op0=nl.less, operand0=mass, engine=nisa.engine.vector)
    last_p = sb.tile((rows, 1), fp32)
    nisa.tensor_reduce(dst=last_p, data=inside, op=nl.add, axis=(1,))
    nisa.tensor_scalar(dst=last_p, data=last_p, op0=nl.subtract, operand0=1.0, engine=nisa.engine.vector)
    t_p = sb.tile((rows, 1), fp32)
    nisa.range_select(
        dst=scratch,
        on_true_tile=desc,
        comp_op0=nl.greater,
        comp_op1=nl.greater_equal,
        bound0=last_p,
        bound1=last_p,
        reduce_op=nl.maximum,
        reduce_res=t_p,
        reduce_cmd=nisa.reduce_cmd.reset_reduce,
        range_start=1,
    )
    nisa.tensor_tensor(dst=t_p, data1=t_p, data2=t_k, op=nl.maximum, engine=nisa.engine.vector)

    # The draw over the kept candidates in gather order.
    e_cand = sb.tile((rows, k), fp32)
    nisa.activation(dst=e_cand, op=nl.exp, data=values, scale=setup["inv_t"], bias=bias, engine=nisa.engine.scalar)
    weight = sb.tile((rows, k), fp32)
    nisa.scalar_tensor_tensor(
        dst=weight, data=values, op0=nl.greater_equal, operand0=t_p, op1=nl.multiply, operand1=e_cand
    )
    cdf = sb.tile((rows, k), fp32)
    nisa.tensor_scalar_cumulative(
        dst=cdf, src=weight, op0=nl.add, op1=nl.add, imm0=0.0, reduce_cmd=nisa.reduce_cmd.reset_reduce
    )
    threshold = sb.tile((rows, 1), fp32)
    nisa.tensor_tensor(
        dst=threshold,
        data1=setup["u"][0:rows, slot : slot + 1],
        data2=cdf[0:rows, k - 1 : k],
        op=nl.multiply,
        engine=nisa.engine.vector,
    )
    below = sb.tile((rows, k), fp32)
    nisa.tensor_scalar(dst=below, data=cdf, op0=nl.less, operand0=threshold, engine=nisa.engine.vector)
    position = sb.tile((rows, 1), fp32)
    nisa.tensor_reduce(dst=position, data=below, op=nl.add, axis=(1,))
    pick = sb.tile((rows, k), fp32)
    nisa.scalar_tensor_tensor(dst=pick, data=iota, op0=nl.equal, operand0=position, op1=nl.multiply, operand1=ids)
    drawn = sb.tile((rows, 1), fp32)
    nisa.tensor_reduce(dst=drawn, data=pick, op=nl.add, axis=(1,))

    nisa.tensor_copy_predicated(dst=token_f, src=drawn, predicate=setup["sampled"])
    # A row with no match (NaN logits) has an id outside the vocab; the clamp keeps it in.
    nisa.tensor_scalar(
        dst=token_f,
        data=token_f,
        op0=nl.minimum,
        operand0=float(vocab - 1),
        op1=nl.maximum,
        operand1=0.0,
        engine=nisa.engine.vector,
    )
    token = sb.tile((rows, 1), nl.int32)
    nisa.tensor_copy(dst=token, src=token_f, engine=nisa.engine.vector)
    _write_q_list(values, ids, weight, cdf, token_f, setup, q_probs, q_ids, rows, k, iota, sb)
    return token


def _write_q_list(values, ids, weight, cdf, token_f, setup, q_probs, q_ids, rows, k, iota, sb):
    """Write the q list ``q_probs`` / ``q_ids`` of one slot of ``_draw_candidates`` (0 on a non-finite row)."""
    fp32 = nl.float32
    inv_total = sb.tile((rows, 1), fp32)
    nisa.reciprocal(dst=inv_total, data=cdf[0:rows, k - 1 : k])
    q = sb.tile((rows, k), fp32)
    nisa.tensor_scalar(dst=q, data=weight, op0=nl.multiply, operand0=inv_total, engine=nisa.engine.vector)
    nisa.tensor_copy(dst=q_probs, src=setup["onehot0"], engine=nisa.engine.vector)
    nisa.tensor_copy_predicated(dst=q_probs, src=q, predicate=setup["sampled_k"])
    nisa.tensor_scalar(
        dst=q_ids, data=iota, op0=nl.multiply, operand0=0.0, op1=nl.add, operand1=token_f, engine=nisa.engine.vector
    )
    nisa.tensor_copy_predicated(dst=q_ids, src=ids, predicate=setup["sampled_k"])
    # q = 0 on a non-finite row: 1 where a candidate value and its q are finite (x == x is 0 for NaN, the bounds are 0
    # for +-inf), the row minimum, then a select of zeros over the whole row.
    vector = nisa.engine.vector
    finite = sb.tile((rows, k), fp32)
    nisa.tensor_tensor(dst=finite, data1=values, data2=values, op=nl.equal, engine=vector)
    q_ok = sb.tile((rows, k), fp32)
    nisa.tensor_tensor(dst=q_ok, data1=q_probs, data2=q_probs, op=nl.equal, engine=vector)
    nisa.scalar_tensor_tensor(
        dst=finite, data=values, op0=nl.less, operand0=_FINITE_MAX, op1=nl.multiply, operand1=finite
    )
    nisa.scalar_tensor_tensor(
        dst=finite, data=values, op0=nl.greater, operand0=-_FINITE_MAX, op1=nl.multiply, operand1=finite
    )
    nisa.scalar_tensor_tensor(dst=q_ok, data=q_probs, op0=nl.less, operand0=_FINITE_MAX, op1=nl.multiply, operand1=q_ok)
    nisa.tensor_tensor(dst=finite, data1=finite, data2=q_ok, op=nl.multiply, engine=vector)
    row_finite = sb.tile((rows, 1), fp32)
    nisa.tensor_reduce(dst=row_finite, data=finite, op=nl.minimum, axis=(1,))
    row_bad = sb.tile((rows, k), fp32)
    nisa.tensor_scalar(
        dst=row_bad, data=iota, op0=nl.multiply, operand0=0.0, op1=nl.add, operand1=row_finite, engine=vector
    )
    bad = sb.tile((rows, k), nl.uint8)
    nisa.tensor_scalar(dst=bad, data=row_bad, op0=nl.less, operand0=0.5, engine=vector)
    zeros = sb.tile((rows, k), fp32)
    nisa.memset(dst=zeros, value=0.0, engine=vector)
    nisa.tensor_copy_predicated(dst=q_probs, src=zeros, predicate=bad)


def _sample_slot(
    logits,
    offset_val,
    tp_group,
    cp,
    setup,
    slot,
    q_probs,
    q_ids,
    tag,
    sb,
    ag_src,
    ag_dst,
    window_dst=False,
):
    """Return the draft ``[B, 1]`` INT32 of one slot of the shard logits ``[B, V_s]`` BF16 (sampled tail).

    Each rank gives its top ``K / cp`` (value, global id) pairs (``K = _sample_width(cp)``); one gather in rank order
    gives the K candidates of ``_draw_candidates``. They hold the global top ``K / cp``, so a greedy row gets the
    ``_global_argmax`` token. ``ag_src`` ``[B, 2K / cp]`` and ``ag_dst`` ``[B, 2K]`` are the persistent gather
    source and destination.

    ``window_dst`` (``sample_unique_dst``): the pairs view of ``ag_dst`` splits its free axis in place of a reshape, as
    ``ag_dst`` can be a column window of a wider tile.
    """
    fp32 = nl.float32
    k = _sample_width(cp)
    per_rank = k // cp
    rows, v_s = logits.shape
    top8 = sb.tile((rows, 8), logits.dtype)
    nisa.max8(dst=top8, src=logits)
    idx8 = sb.tile((rows, 8), nl.uint32)
    nisa.nc_find_index8(dst=idx8, data=logits, vals=top8)
    nisa.tensor_copy(dst=ag_src[0:rows, 0:per_rank], src=top8[0:rows, 0:per_rank], engine=nisa.engine.vector)
    gid = sb.tile((rows, per_rank), fp32)
    nisa.tensor_copy(dst=gid, src=idx8[0:rows, 0:per_rank], engine=nisa.engine.vector)
    nisa.tensor_scalar(
        dst=ag_src[0:rows, per_rank : 2 * per_rank],
        data=gid,
        op0=nl.add,
        operand0=offset_val,
        engine=nisa.engine.vector,
    )
    ncc.all_gather(srcs=[ag_src], dsts=[ag_dst], replica_group=tp_group, collective_dim=1, name=f"{tag}_gather")
    pairs = ag_dst.reshape_dim(1, (cp, 2 * per_rank)) if window_dst else ag_dst.reshape((rows, cp, 2 * per_rank))
    values = sb.tile((rows, cp, per_rank), fp32)
    nisa.tensor_copy(dst=values, src=pairs[0:rows, 0:cp, 0:per_rank], engine=nisa.engine.vector)
    ids = sb.tile((rows, cp, per_rank), fp32)
    nisa.tensor_copy(dst=ids, src=pairs[0:rows, 0:cp, per_rank : 2 * per_rank], engine=nisa.engine.vector)
    return _draw_candidates(
        values.reshape((rows, k)), ids.reshape((rows, k)), setup, slot, cp * v_s, q_probs, q_ids, sb
    )


def _lm_head_base(x_hT, lm_head_weight, tag, sb):
    """Return the lm_head shard logits ``[T, V_s]`` BF16 of the final-normed hT stream ``[128, T, 56]``."""
    fp32 = nl.float32
    hwdge = nisa.dge_mode.hwdge
    sync = nisa.engine.sync
    _, total_rows, n_tiles = x_hT.shape
    v_s = lm_head_weight.shape[1]
    w_tiles = lm_head_weight.reshape((n_tiles, P_MAX, v_s))
    base = sb.tile((total_rows, v_s), nl.bfloat16)
    for c0 in range(0, v_s, PSUM_BANK_FP32):
        width = min(PSUM_BANK_FP32, v_s - c0)
        psum = nl.ndarray((total_rows, width), dtype=fp32, buffer=nl.psum)
        for k in range(n_tiles):
            w = sb.tile((P_MAX, width), nl.bfloat16)
            nisa.dma_copy(
                dst=w, src=w_tiles[k, 0:P_MAX, c0 : c0 + width], dge_mode=hwdge, engine=sync, name=f"{tag}_lm_{c0}_{k}"
            )
            nisa.nc_matmul(dst=psum, stationary=x_hT[0:P_MAX, 0:total_rows, k], moving=w, accumulate=k > 0)
        nisa.tensor_copy(dst=base[0:total_rows, c0 : c0 + width], src=psum)
    return base


def draft_tail(
    h_hT,
    final_norm,
    lm_head_weight,
    markov_w1,
    markov_w2,
    bonus,
    rank_info,
    tp_group,
    tag,
    sb=None,
    hidden_out=None,
    sources=None,
    cp=TP,
    tail_rows=_BLOCK,
    sampling=None,
    markov_w1_local=False,
    unique_dst_heap=None,
):
    """Tail: ``final_norm``, the lm_head shard, and the sequential Markov greedy. Return ``[B, 7]`` INT32.

    With ``sampling`` (``sample_params`` and ``u_draft`` HBM, see ``_sampling_setup``), a sampled row draws each
    draft (``_sample_slot``) and the return is ``(drafts, q_probs, q_ids)`` with ``[B, 7 * 256]`` FP32 and INT32 q
    lists (``[B, 7 * K]`` with ``K = _sample_width(cp)``). Without it, the tail is the greedy tail.

    Args:
        h_hT: ``[128, 7B, 56]`` BF16 SBUF block stream after the 5 layers (before ``final_norm``).
        final_norm: ``[7168]`` HBM norm weight.
        lm_head_weight: ``[7168, V_s]`` BF16 HBM ``[in, out]`` vocab shard of this rank.
        markov_w1: ``[163840, 256]`` BF16 HBM full Markov ``w1``; a device gather reads row ``prev``.
        markov_w2: ``[256, V_s]`` BF16 HBM ``[in, out]`` vocab shard of this rank's Markov ``w2``.
        bonus: ``[B, 1]`` INT32 HBM anchor token of each sequence (``prev`` of row 0).
        rank_info, tp_group, tag, sb: the rank id, the TP group, a name prefix, and the SBUF allocator.
        hidden_out: optional ``[7B, 7168]`` HBM that receives the final-normed hidden (the lm_head input),
            for the tie-tolerance accuracy check.
        tail_rows: the greedy rows to run (1 to 7). Row j reads only row j - 1, so rows below ``tail_rows`` keep their
            bits; the rows from ``tail_rows`` on are 0.
        sampling: optional dict with ``sample_params`` ``[B, 3]`` and ``u_draft`` ``[B, 7]`` FP32 HBM. Only the slots
            below ``tail_rows`` draw; the q lists of the other slots are 0.
        markov_w1_local: ``markov_w1`` is the full ``[V, 256]`` table on every rank (``cp * V_s`` rows), so each row
            gathers ``markov_w1[prev]`` itself, in place of the owner's masked gather and the TP all-reduce. The
            all-reduce adds zeros to the owner's row, so the row keeps its values. The default False keeps the program.
        unique_dst_heap: with ``sampling`` and ``sources``, the region-heap allocator of the step
            (``sample_unique_dst``). The candidate gather of each slot then writes its own destination: a column window
            of one heap tile that the tail allocates before the lm_head, after every earlier heap tile, so no earlier
            tile moves. A write to anchor columns after the windows makes the tile live from there, so no tile of the
            tail shares its bytes. None (default) keeps the program.
    """
    sb = AUTO_SBUF if sb is None else sb
    fp32 = nl.float32
    hwdge = nisa.dge_mode.hwdge
    sync = nisa.engine.sync
    _, total_rows, _ = h_hT.shape
    v_s = lm_head_weight.shape[1]
    tokens = total_rows // _BLOCK
    ones = new_ones()
    # ``unique_dst_heap``: one gather destination a slot, so no gather writes bytes that an earlier instruction wrote.
    slot_dsts = None
    if unique_dst_heap is not None and sampling is not None and sources is not None:
        cols = sources["tail_dst"].shape[1]
        dsts = unique_dst_heap.tile((tokens, _BLOCK * cols + _DST_ANCHOR), fp32, "tail_ag_dsts")
        anchor = dsts[0:tokens, _BLOCK * cols : _BLOCK * cols + _DST_ANCHOR]
        nisa.memset(dst=anchor, value=0.0, engine=nisa.engine.vector)
        slot_dsts = [dsts[0:tokens, i * cols : (i + 1) * cols] for i in range(_BLOCK)]

    norm_hT = load_vector_hT(final_norm, dtype=fp32)
    x_hT = rmsnorm_hT(h_hT, norm_hT, _EPS, ones, sb=sb)
    if hidden_out is not None:
        store_hT_rows(hidden_out, x_hT, sb=sb)
    base = _lm_head_base(x_hT, lm_head_weight, f"{tag}_base", sb)

    rank_i = sb.tile((tokens, 1), nl.int32)
    nisa.dma_copy(
        dst=rank_i, src=rank_info[0:1, 2:3].broadcast(0, tokens), dge_mode=hwdge, engine=sync, name=f"{tag}_rank"
    )
    rank_f = sb.tile((tokens, 1), fp32)
    nisa.tensor_copy(dst=rank_f, src=rank_i)
    offset_val = sb.tile((tokens, 1), fp32)
    nisa.tensor_scalar(dst=offset_val, data=rank_f, op0=nl.multiply, operand0=float(v_s))
    shard_base = sb.tile((tokens, 1), nl.int32)
    nisa.tensor_copy(dst=shard_base, src=offset_val)

    prev = sb.tile((tokens, 1), nl.int32)
    nisa.dma_copy(dst=prev, src=bonus[0:tokens, 0:1], dge_mode=hwdge, engine=sync, name=f"{tag}_bonus")
    drafts = sb.tile((tokens, _BLOCK), nl.int32)
    if tail_rows < _BLOCK:
        nisa.memset(dst=drafts, value=0)
    width = _sample_width(cp)
    if sampling is not None:
        setup = _sampling_setup(sampling["sample_params"], sampling["u_draft"], tokens, width, f"{tag}_sample", sb)
        q_probs = sb.tile((tokens, _BLOCK * width), fp32)
        q_ids = sb.tile((tokens, _BLOCK * width), fp32)
        if tail_rows < _BLOCK:
            nisa.memset(dst=q_probs, value=0.0, engine=nisa.engine.vector)
            nisa.memset(dst=q_ids, value=0.0, engine=nisa.engine.vector)
        # Without the persistent sources, one gather source and destination serve every slot.
        if sources is None:
            sample_src = sb.tile((tokens, 2 * width // cp), fp32)
            sample_dst = sb.tile((tokens, 2 * width), fp32)
    if markov_w1_local:
        kernel_assert(
            markov_w1.shape[0] == cp * v_s,
            f"markov_w1_local needs the full markov_w1 [{cp * v_s}, {_MARKOV_RANK}], got {tuple(markov_w1.shape)}",
        )
    for j in range(tail_rows):
        if markov_w1_local:
            # The full table on every rank: gather row prev (clamped to the vocab), no all-reduce.
            full_idx = sb.tile((tokens, 1), nl.int32)
            nisa.tensor_scalar(
                dst=full_idx,
                data=prev,
                op0=nl.minimum,
                operand0=float(cp * v_s - 1),
                op1=nl.maximum,
                operand1=0.0,
            )
            w1g = sb.tile((tokens, _MARKOV_RANK), nl.bfloat16)
            for b in range(tokens):
                nisa.dma_copy(
                    dst=w1g[b : b + 1, 0:_MARKOV_RANK],
                    src=markov_w1.ap(
                        pattern=[[_MARKOV_RANK, 1], [1, _MARKOV_RANK]],
                        offset=0,
                        scalar_offset=full_idx[b : b + 1, 0:1],
                        indirect_dim=0,
                    ),
                    dge_mode=hwdge,
                    engine=sync,
                    name=f"{tag}_w1g_{j}_{b}",
                )
        else:
            # markov_w1 is vocab-sharded: the owner of prev gathers its row, others gather 0, a tiny TP
            # all-reduce broadcasts markov_w1[prev] to every rank. clamp bounds the gather index in range.
            local_idx = sb.tile((tokens, 1), nl.int32)
            nisa.tensor_tensor(dst=local_idx, data1=prev, data2=shard_base, op=nl.subtract)
            clamp = sb.tile((tokens, 1), nl.int32)
            nisa.tensor_scalar(
                dst=clamp, data=local_idx, op0=nl.minimum, operand0=float(v_s - 1), op1=nl.maximum, operand1=0.0
            )
            li_f = sb.tile((tokens, 1), fp32)
            cl_f = sb.tile((tokens, 1), fp32)
            nisa.tensor_copy(dst=li_f, src=local_idx)
            nisa.tensor_copy(dst=cl_f, src=clamp)
            is_owner = sb.tile((tokens, 1), fp32)
            nisa.tensor_scalar(dst=is_owner, data=li_f, op0=nl.equal, operand0=cl_f)
            w1g_local = sb.tile((tokens, _MARKOV_RANK), nl.bfloat16)
            for b in range(tokens):
                nisa.dma_copy(
                    dst=w1g_local[b : b + 1, 0:_MARKOV_RANK],
                    src=markov_w1.ap(
                        pattern=[[_MARKOV_RANK, 1], [1, _MARKOV_RANK]],
                        offset=0,
                        scalar_offset=clamp[b : b + 1, 0:1],
                        indirect_dim=0,
                    ),
                    dge_mode=hwdge,
                    engine=sync,
                    name=f"{tag}_w1g_{j}_{b}",
                )
            nisa.tensor_scalar(dst=w1g_local, data=w1g_local, op0=nl.multiply, operand0=is_owner)
            # The Markov all-reduce runs in persistent HBM. Store the local masked feature, reduce
            # HBM->HBM over TP, load the result. The HBM tensors are reused across the greedy rows, not cleared.
            w1g = sb.tile((tokens, _MARKOV_RANK), nl.bfloat16)
            if sources is not None:
                nisa.dma_copy(
                    dst=sources["markov_src_hbm"],
                    src=w1g_local,
                    dge_mode=hwdge,
                    engine=sync,
                    name=f"{tag}_w1_ar_store_{j}",
                )
                check_pair_group(sources["markov_dst_hbm"], tp_group, f"{tag}_w1_all_reduce_{j}")
                ncc.all_reduce(
                    srcs=[sources["markov_src_hbm"]],
                    dsts=[sources["markov_dst_hbm"]],
                    replica_group=tp_group,
                    op=nl.add,
                    name=f"{tag}_w1_all_reduce_{j}",
                )
                nisa.dma_copy(
                    dst=w1g, src=sources["markov_dst_hbm"], dge_mode=hwdge, engine=sync, name=f"{tag}_w1_ar_load_{j}"
                )
            else:
                ncc.all_reduce(
                    srcs=[w1g_local], dsts=[w1g], replica_group=tp_group, op=nl.add, name=f"{tag}_w1_all_reduce_{j}"
                )
        w1g_hT = _rows_to_hT(w1g, sb)
        # Dedicated bias output tile (not a fresh scratch) so the matmul output never takes the markov
        # all-reduce source bytes while a peer still reads them.
        bias = sources["markov_bias"] if sources is not None else sb.tile((tokens, v_s), nl.bfloat16)
        _matmul_tok(w1g_hT, markov_w2, v_s, f"{tag}_bias_{j}", sb, k_tiles_per_dma=_MARKOV_TILES, out=bias)
        base_j = sb.tile((tokens, v_s), nl.bfloat16)
        nisa.dma_copy(
            dst=base_j,
            src=base[j * tokens : (j + 1) * tokens, 0:v_s],
            dge_mode=hwdge,
            engine=sync,
            name=f"{tag}_base_{j}",
        )
        logits = sb.tile((tokens, v_s), nl.bfloat16)
        nisa.tensor_tensor(dst=logits, data1=base_j, data2=bias, op=nl.add)
        tail_src = sources["tail_src"] if sources is not None else None
        if sources is not None and j % 2 and "tail_src_1" in sources:
            # ag_parity: the odd slots take the second gather source.
            tail_src = sources["tail_src_1"]
        tail_dst = sources["tail_dst"] if sources is not None else None
        if sampling is None:
            token = _global_argmax(
                logits, offset_val, tp_group, cp, f"{tag}_argmax_{j}", sb, ag_src=tail_src, ag_dst=tail_dst
            )
        else:
            if tail_src is None:
                tail_src, tail_dst = sample_src, sample_dst
            if slot_dsts is not None:
                tail_dst = slot_dsts[j]
            c0, c1 = j * width, (j + 1) * width
            token = _sample_slot(
                logits,
                offset_val,
                tp_group,
                cp,
                setup,
                j,
                q_probs[0:tokens, c0:c1],
                q_ids[0:tokens, c0:c1],
                f"{tag}_sample_{j}",
                sb,
                tail_src,
                tail_dst,
                window_dst=slot_dsts is not None,
            )
        nisa.tensor_copy(dst=drafts[0:tokens, j : j + 1], src=token)
        prev = token
    if sampling is None:
        return drafts
    # A NaN row can hold ids outside the vocab; the clamp keeps every q id in.
    q_ids_i = sb.tile((tokens, _BLOCK * width), nl.int32)
    nisa.tensor_scalar(
        dst=q_ids_i,
        data=q_ids,
        op0=nl.minimum,
        operand0=float(cp * v_s - 1),
        op1=nl.maximum,
        operand1=0.0,
        engine=nisa.engine.vector,
    )
    return drafts, q_probs, q_ids_i


def _silu_mlp(x_hT, gate_up, down, tp_group, tag, sb, ar_src_hbm=None, ar_dst_hbm=None, cp=TP):
    """Return the dense SwiGLU MLP output ``[128, T, 56]`` BF16 hT for the normalized stream ``x_hT``.

    The intermediate is TP32 sharded (448 per rank). The down partial sums over TP in FP32, then rounds
    once, which matches the single BF16 round of ``draft_mla_torch.draft_layer``. ``ar_src_hbm`` and
    ``ar_dst_hbm`` are the persistent per-rank HBM all-reduce source and destination:
    the local partial is stored there, reduced HBM->HBM over TP, and the result loaded back. ``None`` runs
    the all-reduce in SBUF (the standalone path).
    """
    fp32 = nl.float32
    bf16 = nl.bfloat16
    hwdge = nisa.dge_mode.hwdge
    sync = nisa.engine.sync
    _, total_rows, _ = x_hT.shape
    shard = 14336 // cp
    down_tiles = (shard + P_MAX - 1) // P_MAX
    full_tiles = shard // P_MAX
    tail = shard - full_tiles * P_MAX
    padded = total_rows + total_rows % 2
    # Above 4 tokens the down weight loads one output-column chunk at a time, to keep the B=8 SBUF peak down.
    stream_down = total_rows > _BLOCK * 4
    # CP32 explicit (loop) path only: ring the gate|up weight chunks and scope the gate|up so its ring and
    # the SwiGLU intermediates free before the down projection. Drafter-only; the DMA, matmul and
    # accumulation order do not change, so the output is bitwise identical. The static and CP128 forms keep
    # the base program (ring_gu False -> the allocation sequence matches the base).
    ring_gu = sb.explicit and cp == TP

    # The SwiGLU activation transposed for the down matmul; it outlives the gate|up scope on the ring path.
    act_t = sb.tile((P_MAX, down_tiles, total_rows), bf16) if ring_gu else None
    down_sb = None
    gu_scope = sb.scope(f"{tag}_gu") if ring_gu else nullcontext()
    with gu_scope:
        gate_up_bf16 = _matmul_tok(x_hT, gate_up, 2 * shard, f"{tag}_gu", sb, ring=ring_gu)
        gate = gate_up_bf16[0:total_rows, 0:shard]
        up = gate_up_bf16[0:total_rows, shard : 2 * shard]
        sig = sb.tile((total_rows, shard), fp32)
        nisa.activation(dst=sig, op=nl.sigmoid, data=gate)
        act = sb.tile((total_rows, shard), bf16)
        nisa.tensor_tensor(dst=act, data1=gate, data2=sig, op=nl.multiply)
        act_up = sb.tile((total_rows, shard), bf16)
        nisa.tensor_tensor(dst=act_up, data1=act, data2=up, op=nl.multiply)

        if not stream_down and not ring_gu:
            down_sb = sb.tile((P_MAX, down_tiles, HIDDEN), bf16)
            if full_tiles:
                nisa.dma_copy(
                    dst=down_sb[:, 0:full_tiles, :],
                    src=down.ap(pattern=[[HIDDEN, P_MAX], [P_MAX * HIDDEN, full_tiles], [1, HIDDEN]], offset=0),
                    dge_mode=hwdge,
                    engine=sync,
                    name=f"{tag}_down",
                )
            if tail:
                nisa.dma_copy(
                    dst=down_sb[0:tail, full_tiles, :],
                    src=down[full_tiles * P_MAX : shard, :],
                    dge_mode=hwdge,
                    engine=sync,
                    name=f"{tag}_down_tail",
                )

        act_t_psum = nl.ndarray((P_MAX, down_tiles, padded), dtype=bf16, buffer=nl.psum)
        for i in range(down_tiles):
            i_rows = min(P_MAX, shard - i * P_MAX)
            nisa.nc_transpose(
                dst=act_t_psum[0:i_rows, i, 0:total_rows], data=act_up[0:total_rows, i * P_MAX : i * P_MAX + i_rows]
            )
        if not ring_gu:
            act_t = sb.tile((P_MAX, down_tiles, total_rows), bf16)
        nisa.tensor_copy(dst=act_t, src=act_t_psum[0:P_MAX, 0:down_tiles, 0:total_rows])

    # Ring path with a resident down weight (B<=4): load it after the gate|up scope frees its ring.
    if not stream_down and ring_gu:
        down_sb = sb.tile((P_MAX, down_tiles, HIDDEN), bf16)
        if full_tiles:
            nisa.dma_copy(
                dst=down_sb[:, 0:full_tiles, :],
                src=down.ap(pattern=[[HIDDEN, P_MAX], [P_MAX * HIDDEN, full_tiles], [1, HIDDEN]], offset=0),
                dge_mode=hwdge,
                engine=sync,
                name=f"{tag}_down",
            )
        if tail:
            nisa.dma_copy(
                dst=down_sb[0:tail, full_tiles, :],
                src=down[full_tiles * P_MAX : shard, :],
                dge_mode=hwdge,
                engine=sync,
                name=f"{tag}_down_tail",
            )

    tiles_per_bank = min(_H_TILES, PSUM_BANK_FP32 // total_rows)
    partial = sb.tile((P_MAX, total_rows, _H_TILES), fp32)
    for first in range(0, _H_TILES, tiles_per_bank):
        n = min(tiles_per_bank, _H_TILES - first)
        # Explicit CP32 path: scope each streamed down-weight chunk so it frees per iteration instead of
        # accumulating on the region stack. The base stream_down allocates a fresh chunk per output block,
        # which the compiler frees by liveness on the placed path but the explicit stack holds to frame end.
        chunk_scope = sb.scope(f"{tag}_dw_{first}") if (ring_gu and stream_down) else nullcontext()
        with chunk_scope:
            if stream_down:
                cols = n * P_MAX
                col0 = first * P_MAX
                down_w = sb.tile((P_MAX, down_tiles, cols), bf16)
                # At CP128 the shard (112) has no full tile, only the tail.
                if full_tiles:
                    nisa.dma_copy(
                        dst=down_w[:, 0:full_tiles, :],
                        src=down.ap(pattern=[[HIDDEN, P_MAX], [P_MAX * HIDDEN, full_tiles], [1, cols]], offset=col0),
                        dge_mode=hwdge,
                        engine=sync,
                        name=f"{tag}_down_{first}",
                    )
                if tail:
                    nisa.dma_copy(
                        dst=down_w[0:tail, full_tiles, :],
                        src=down[full_tiles * P_MAX : shard, col0 : col0 + cols],
                        dge_mode=hwdge,
                        engine=sync,
                        name=f"{tag}_down_tail_{first}",
                    )
                col_base = first
            else:
                down_w = down_sb
                col_base = 0
            psum = nl.ndarray((P_MAX, n, total_rows), dtype=fp32, buffer=nl.psum)
            for off in range(n):
                hj = first + off
                col = (hj - col_base) * P_MAX
                for i in range(down_tiles):
                    i_rows = min(P_MAX, shard - i * P_MAX)
                    nisa.nc_matmul(
                        dst=psum[0:P_MAX, off, 0:total_rows],
                        stationary=down_w[0:i_rows, i, col : col + P_MAX],
                        moving=act_t[0:i_rows, i, 0:total_rows],
                        accumulate=i > 0,
                    )
            nisa.tensor_copy(dst=partial[0:P_MAX, 0:total_rows, first : first + n], src=psum.permute((0, 2, 1)))
    summed = sb.tile((P_MAX, total_rows, _H_TILES), fp32)
    if ar_src_hbm is not None:
        # All-reduce in persistent HBM (placeable scratch). Store the local partial, reduce
        # HBM->HBM over TP, load the reduced result back to SBUF. The HBM tensors are not cleared at start.
        nisa.dma_copy(
            dst=ar_src_hbm,
            src=partial.reshape((P_MAX, total_rows * _H_TILES)),
            dge_mode=hwdge,
            engine=sync,
            name=f"{tag}_ar_store",
        )
        check_pair_group(ar_dst_hbm, tp_group, f"{tag}_mlp_all_reduce")
        ncc.all_reduce(
            srcs=[ar_src_hbm], dsts=[ar_dst_hbm], replica_group=tp_group, op=nl.add, name=f"{tag}_mlp_all_reduce"
        )
        nisa.dma_copy(
            dst=summed.reshape((P_MAX, total_rows * _H_TILES)),
            src=ar_dst_hbm,
            dge_mode=hwdge,
            engine=sync,
            name=f"{tag}_ar_load",
        )
    else:
        ncc.all_reduce(
            srcs=[partial.reshape((P_MAX, total_rows * _H_TILES))],
            dsts=[summed.reshape((P_MAX, total_rows * _H_TILES))],
            replica_group=tp_group,
            op=nl.add,
            name=f"{tag}_mlp_all_reduce",
        )
    out = sb.tile((P_MAX, total_rows, _H_TILES), bf16)
    nisa.tensor_copy(dst=out, src=summed)
    return out


def _layer_mla_sources(mla_sources, layer: int):
    """Return the MLA collective sources of drafter layer ``layer``: with the ``qkv_p1`` and ``qkv_dst_p1`` keys
    (``ag_parity``), an odd layer takes them as its qkv all-gather pair; else ``mla_sources`` as it is.

    Example:
        >>> _layer_mla_sources({"qkv": 0, "qkv_dst": 1, "qkv_p1": 2, "qkv_dst_p1": 3}, 1)
        {'qkv': 2, 'qkv_dst': 3}
        >>> _layer_mla_sources({"qkv": 0, "qkv_dst": 1, "qkv_p1": 2, "qkv_dst_p1": 3}, 2)
        {'qkv': 0, 'qkv_dst': 1}
    """
    if mla_sources is None or "qkv_p1" not in mla_sources:
        return mla_sources
    out = {key: value for key, value in mla_sources.items() if key not in ("qkv_p1", "qkv_dst_p1")}
    if layer % 2:
        out["qkv"], out["qkv_dst"] = mla_sources["qkv_p1"], mla_sources["qkv_dst_p1"]
    return out


def draft_layer(
    h_hT,
    mla_in_proj,
    mla_q_a_norm,
    mla_q_b_proj,
    mla_kv_a_norm,
    mla_kv_b_proj,
    mla_o_proj,
    input_norm,
    post_norm,
    dense_gate_up,
    dense_down,
    kv_pool,
    block_table,
    slot,
    position,
    rope_rows,
    rank_info,
    tp_group,
    tag,
    softmax_scale,
    key_start=None,
    window_base=None,
    loop_path=False,
    fp8_pages=False,
    sb=None,
    out=None,
    ar_src_hbm=None,
    ar_dst_hbm=None,
    ones=None,
    mla_sources=None,
    cp=TP,
    heads=None,
    rows_attention=False,
    q_hbm=False,
    trimmed_window=False,
    query_rows=0,
    out_ag_call=False,
    stack_rows=False,
    pass_q_prefetch=0,
    q_gather_buffers=1,
    q_prefetch_after=(),
    pass_tail_overlap=False,
    seq_passes=False,
    state_store_split=False,
    prior_hoist=0,
):
    """Drafter layer: ``h += mla(input_norm(h))`` (draft mode) then ``h += silu_mlp(post_norm(h))``.

    Args:
        h_hT: ``[128, 7B, 56]`` BF16 SBUF block stream.
        mla_in_proj .. mla_o_proj: this rank's ``MLA_WEIGHT_NAMES`` weights (``local_heads=2``, no gate).
        input_norm, post_norm: ``[7168]`` HBM norm weights.
        dense_gate_up: ``[128, 56, 896]`` BF16 HBM ``[in, out]`` hT-tiled gate|up shard.
        dense_down: ``[448, 7168]`` BF16 HBM ``[in, out]`` down shard.
        kv_pool, block_table, slot, position, rope_rows, rank_info, tp_group: the MLA draft inputs. The
            prior keys are the pool rows below ``position[b]``; ``write_cache=False`` keeps the block rows
            in SBUF.
        tag, softmax_scale, loop_path, fp8_pages, sb: a name prefix, the drafter softmax scale, the long
            context flag, the pool dtype flag, and the SBUF allocator.
        rows_attention: the 7 rows of a sequence attend each prior-key group together, with the keys as the
            stationary operand (``transposed_scores`` of ``mla_block_attention``). The rows share the prior keys of
            the block, so the result differs from the per-row form only by rounding.
        q_hbm: gather the queries of all rows with one all-gather into the HBM pair of ``mla_sources``
            (``q_hbm`` of ``mla_block_attention``), in place of one gather for each 4-token tile of each pass. The
            gathered bytes are the same.
        trimmed_window: ``block_table`` is the 12-column trimmed window (``trimmed_window`` of
            ``mla_block_attention``).
        query_rows: attend only the passes of the token-major rows below this count (``query_rows`` of
            ``mla_block_attention``); 0 attends every row.
        stack_rows: attend each prior-key group for 2 rows of a sequence at once (``stack_rows`` of
            ``mla_block_attention``); the outputs do not change.
        pass_q_prefetch, q_gather_buffers: the query gather tiles of the later passes (``mla_block_attention``, with
            ``q_hbm``); the outputs do not change.
        q_prefetch_after: instructions that the ``pass_q_prefetch`` loads follow in the schedule (no hardware wait).
        pass_tail_overlap: run the output tail of each MLA pass after the attention of the next pass
            (``pass_tail_overlap`` of ``mla_block_attention``; ``mla_sources`` with the second state pair); the outputs
            do not change.
        seq_passes: attend the MLA passes on whole sequences (``seq_passes`` of ``mla_block_attention``, with
            ``q_hbm``); the outputs do not change.
        state_store_split: store the FP32 MLA state in two column halves on the SP and ScalarE queues
            (``state_store_split`` of ``mla_block_attention``); the output does not change.
        prior_hoist: trace the page loads and key transposes of the first this-many key groups of each sequence before
            each pass's query loads (``prior_hoist`` of ``mla_block_attention``); the output does not change.

    Returns:
        ``[128, 7B, 56]`` BF16 SBUF block stream after the layer.
    """
    sb = AUTO_SBUF if sb is None else sb
    bf16 = nl.bfloat16
    ones = new_ones() if ones is None else ones
    input_hT = load_vector_hT(input_norm, dtype=nl.float32)
    x_hT = rmsnorm_hT(h_hT, input_hT, _EPS, ones, sb=sb)
    mla_kwargs = dict(
        local_heads=2 if cp == TP else 1,
        q_rows=_BLOCK,
        rope_rows=rope_rows,
        causal=False,
        output_gate=False,
        write_cache=False,
        softmax_scale=softmax_scale,
        loop_path=loop_path,
        fp8_pages=fp8_pages,
        persistent_rs=True,
    )
    if rows_attention:
        mla_kwargs["transposed_scores"] = True
    if q_hbm:
        mla_kwargs["q_hbm"] = True
    if trimmed_window:
        mla_kwargs["trimmed_window"] = True
    if query_rows:
        mla_kwargs["query_rows"] = query_rows
    if out_ag_call:
        mla_kwargs["out_ag_call"] = True
    if stack_rows:
        mla_kwargs["stack_rows"] = True
    if pass_q_prefetch:
        mla_kwargs["pass_q_prefetch"] = pass_q_prefetch
    if q_prefetch_after:
        mla_kwargs["q_prefetch_after"] = q_prefetch_after
    if pass_tail_overlap:
        mla_kwargs["pass_tail_overlap"] = True
    if seq_passes:
        mla_kwargs["seq_passes"] = True
    if q_gather_buffers != 1:
        mla_kwargs["q_gather_buffers"] = q_gather_buffers
    if state_store_split:
        mla_kwargs["state_store_split"] = True
    if prior_hoist:
        mla_kwargs["prior_hoist"] = prior_hoist
    if cp != TP:
        # CP128: one query and output head on each rank, 64 gathered heads (ranks 64..127 are zero pad).
        mla_kwargs["cp"] = cp
        mla_kwargs["heads"] = heads
    if _K1_HAS_KEY_START and key_start is not None:
        mla_kwargs["key_start"] = key_start
    if window_base is not None:
        mla_kwargs["window_base"] = window_base
    if loop_path:
        # The loop path needs the MLA body's tiles at explicit addresses (this layer's region frame).
        mla_kwargs["sbuf"] = sb
        if cp == TP:
            # CP32 only: cap the FP8 loop groups so the MLA attention fits the region at B up to 8.
            mla_kwargs["loop_group_cap"] = _LOOP_FP8_GROUPS
        if mla_sources is not None:
            # CP128: the persistent HBM tensors of the MLA collectives.
            mla_kwargs["sources"] = mla_sources
    elif mla_sources is not None:
        # Static path: pass the kernel's persistent q, qkv and tail-AG source tiles so the MLA body does not build
        # a per-call arena that dies after the layer (its tail-AG source would be reused by the next
        # layer after only the local completion). The sources are reserved to the kernel end.
        mla_kwargs["sources"] = mla_sources
    attn = mla_block_attention(
        x_hT,
        mla_in_proj,
        mla_q_a_norm,
        mla_q_b_proj,
        mla_kv_a_norm,
        mla_kv_b_proj,
        mla_o_proj,
        kv_pool,
        block_table,
        slot,
        position,
        rank_info,
        tp_group,
        f"{tag}_mla",
        **mla_kwargs,
    )
    h_attn = sb.tile(tuple(h_hT.shape), bf16)
    nisa.tensor_tensor(dst=h_attn, data1=h_hT, data2=attn, op=nl.add)
    post_hT = load_vector_hT(post_norm, dtype=nl.float32)
    x_hT = rmsnorm_hT(h_attn, post_hT, _EPS, ones, sb=sb)
    mlp = _silu_mlp(
        x_hT, dense_gate_up, dense_down, tp_group, f"{tag}_mlp", sb, ar_src_hbm=ar_src_hbm, ar_dst_hbm=ar_dst_hbm, cp=cp
    )
    h_out = out if out is not None else sb.tile(tuple(h_hT.shape), bf16)
    nisa.tensor_tensor(dst=h_out, data1=h_attn, data2=mlp, op=nl.add)
    return h_out


draft_layer.token_major = True


_ROWS_PER_LOAD = 16
_ACT_ROW_LOADS = []
"""``_record_act_row_loads``: a stack of lists; while one is open, ``_rows_into_hT`` appends each of its ScalarE-queue
row loads to the innermost list."""


@contextmanager
def _record_act_row_loads():
    """Record the ScalarE-queue row loads of ``_rows_into_hT`` (the context-stage and block-embed rows) while open."""
    loads = []
    _ACT_ROW_LOADS.append(loads)
    try:
        yield loads
    finally:
        _ACT_ROW_LOADS.pop()


"""The rows of one DMA of ``_rows_into_hT``: a ``[56, 16, 128]`` BF16 staging tile is 4 KB a partition."""


def _rows_into_hT(src_rows, row0, nrows, dst, col0, tag, sb):
    """Write the rows ``row0 .. row0 + nrows`` of the HBM ``[N, 7168]`` tensor ``src_rows`` into the hT columns
    ``col0 .. col0 + 56`` of ``dst`` ``[128, nrows, *]``: element ``(p, r, col0 + t)`` is ``src_rows[row0 + r, 128 t + p]``.

    One DMA loads up to ``_ROWS_PER_LOAD`` rows into a ``[56, rows, 128]`` staging tile, on the SP and ACT queues in
    turn; then each row takes one PE transpose and one copy. One DMA a row costs one DMA latency a row (each waits for
    the one before it). The bytes and the transposes do not change.
    """
    hwdge = nisa.dge_mode.hwdge
    for load, r0 in enumerate(range(0, nrows, _ROWS_PER_LOAD)):
        count = min(_ROWS_PER_LOAD, nrows - r0)
        stage = sb.tile((_H_TILES, count, P_MAX), src_rows.dtype)
        row_load = nisa.dma_copy(
            dst=stage,
            src=src_rows.ap(pattern=[[P_MAX, _H_TILES], [HIDDEN, count], [1, P_MAX]], offset=(row0 + r0) * HIDDEN),
            dge_mode=hwdge,
            engine=(nisa.engine.sync, nisa.engine.scalar)[load % 2],
            name=f"{tag}_{r0}",
        )
        if load % 2 and _ACT_ROW_LOADS:
            _ACT_ROW_LOADS[-1].append(row_load)
        for row in range(count):
            psum = nl.ndarray((P_MAX, _H_TILES), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=psum, data=stage[0:_H_TILES, row, 0:P_MAX])
            nisa.tensor_copy(dst=dst[0:P_MAX, r0 + row, col0 : col0 + _H_TILES], src=psum)


def _load_rows_hT(src, total_rows, sb):
    """Load an HBM ``[T, 7168]`` tensor into the token-major hT tile ``[128, T, 56]`` BF16."""
    out = sb.tile((P_MAX, total_rows, _H_TILES), nl.bfloat16)
    _rows_into_hT(src, 0, total_rows, out, 0, "blk_emb", sb)
    return out


def _build_aux_hT(aux, total_rows, sb):
    """Build ``[128, 8B, 280]`` BF16 aux hT from ``aux`` ``[5, 8B, H]`` (cat over the 5 aux states)."""
    aux_flat = aux.reshape((_AUX_LAYERS * total_rows, HIDDEN))
    aux_hT = sb.tile((P_MAX, total_rows, _AUX_TILES), nl.bfloat16)
    for a in range(_AUX_LAYERS):
        _rows_into_hT(aux_flat, a * total_rows, total_rows, aux_hT, a * _H_TILES, f"aux_load_{a}", sb)
    return aux_hT


def _embed_block(embed, bonus, mask_token, tokens, tp_group, tag, sb, sources=None):
    """Gather the block embedding ``[128, 7B, 56]`` BF16 hT: row 0 is ``embed[bonus]``, rows 1..6 ``embed[mask]``.

    ``embed`` is ``[V, 224]`` BF16 HBM (this rank's hidden shard); the TP all-gather reconstructs the
    full hidden before the hT transpose. ``sources`` provides the persistent AG source/destination.
    """
    bf16 = nl.bfloat16
    hwdge = nisa.dge_mode.hwdge
    sync = nisa.engine.sync
    h_shard = embed.shape[1]
    total_rows = _BLOCK * tokens
    bonus_i = sb.tile((tokens, 1), nl.int32)
    nisa.dma_copy(dst=bonus_i, src=bonus[0:tokens, 0:1], dge_mode=hwdge, engine=sync, name=f"{tag}_bonus")
    local = sources["emb_src"] if sources is not None else sb.tile((total_rows, h_shard), bf16)
    for b in range(tokens):
        nisa.dma_copy(
            dst=local[b : b + 1, 0:h_shard],
            src=embed.ap(
                pattern=[[h_shard, 1], [1, h_shard]], offset=0, scalar_offset=bonus_i[b : b + 1, 0:1], indirect_dim=0
            ),
            dge_mode=hwdge,
            engine=sync,
            name=f"{tag}_bonus_emb_{b}",
        )
    nisa.dma_copy(
        dst=local[tokens:total_rows, 0:h_shard],
        src=embed[mask_token : mask_token + 1, 0:h_shard].broadcast(0, total_rows - tokens),
        dge_mode=hwdge,
        engine=sync,
        name=f"{tag}_mask_emb",
    )
    full = sources["emb_dst"] if sources is not None else sb.tile((total_rows, HIDDEN), bf16)
    ncc.all_gather(srcs=[local], dsts=[full], replica_group=tp_group, collective_dim=1, name=f"{tag}_emb_gather")
    return _rows_to_hT(full, sb)


def _pad_verify_aux(aux, tokens, queries, sb):
    """Return the aux ``[5, Q*B, H]`` of a step with Q verify rows as ``[5, 8B, H]`` HBM, with zero rows j >= Q."""
    hwdge = nisa.dge_mode.hwdge
    sync = nisa.engine.sync
    rows = queries * tokens
    total = _AUX_VERIFY * tokens
    out = nl.ndarray((_AUX_LAYERS, total, HIDDEN), dtype=aux.dtype, buffer=nl.private_hbm, name="aux_pad")
    zeros = sb.tile((total - rows, HIDDEN), aux.dtype)
    nisa.memset(dst=zeros, value=0.0)
    for a in range(_AUX_LAYERS):
        nisa.dma_copy(
            dst=out[a, 0:rows, 0:HIDDEN], src=aux[a, 0:rows, 0:HIDDEN], dge_mode=hwdge, engine=sync, name=f"aux_pad_{a}"
        )
        nisa.dma_copy(dst=out[a, rows:total, 0:HIDDEN], src=zeros, dge_mode=hwdge, engine=sync, name=f"aux_zero_{a}")
    return out


def _guard_rows(src, rows, total, limit, tag, sb, engine=None):
    """Return INT32 ``src`` ``[rows, W]`` as HBM ``[total, W]``: values outside ``[0, limit)`` and rows past ``rows`` are 0.

    ``limit`` None keeps the values. The values stay below 2^24, so the FP32 compares and products are exact. ``engine``
    pins every compute op of the guard to that engine; None leaves the choice to the compiler.
    """
    eng = {} if engine is None else {"engine": engine}
    hwdge = nisa.dge_mode.hwdge
    sync = nisa.engine.sync
    fp32 = nl.float32
    width = src.shape[1]
    kernel_assert(total <= P_MAX, f"the guarded rows must fit one tile ({P_MAX}), got {total}")
    raw = sb.tile((rows, width), nl.int32)
    nisa.dma_copy(dst=raw, src=src[0:rows, 0:width], dge_mode=hwdge, engine=sync, name=f"{tag}_load")
    out_i = sb.tile((total, width), nl.int32)
    nisa.memset(dst=out_i, value=0, **eng)
    if limit is None:
        nisa.tensor_copy(dst=out_i[0:rows, 0:width], src=raw, **eng)
    else:
        val = sb.tile((rows, width), fp32)
        nisa.tensor_copy(dst=val, src=raw, **eng)
        valid = sb.tile((rows, width), fp32)
        nisa.tensor_scalar(dst=valid, data=val, op0=nl.greater_equal, operand0=0.0, **eng)
        below = sb.tile((rows, width), fp32)
        nisa.tensor_scalar(dst=below, data=val, op0=nl.less, operand0=float(limit), **eng)
        nisa.tensor_tensor(dst=valid, data1=valid, data2=below, op=nl.multiply, **eng)
        nisa.tensor_tensor(dst=val, data1=val, data2=valid, op=nl.multiply, **eng)
        nisa.tensor_copy(dst=out_i[0:rows, 0:width], src=val, **eng)
    out = nl.ndarray((total, width), dtype=nl.int32, buffer=nl.private_hbm, name=f"{tag}_guarded")
    nisa.dma_copy(dst=out, src=out_i, dge_mode=hwdge, engine=sync, name=f"{tag}_store")
    return out


def _embed_vocab_rows(embed_vocab, bonus, mask_token, tokens, rank_info, tp_group, src_hbm, dst_hbm, sb):
    """Return the block embedding ``[7B, H]`` BF16 HBM from this rank's vocab shard ``embed_vocab`` ``[V/TP, H]``.

    As the target's sharded lookup: each rank takes the rows it owns (+0 elsewhere) of [bonus, mask], and one TP
    all-reduce adds them. Each sum is one owner value plus zeros, so the bits do not depend on the reduce order.
    Rows: B bonus rows, then the mask row for each of the 6B mask rows (token-major).
    """
    hwdge = nisa.dge_mode.hwdge
    sync = nisa.engine.sync
    fp32 = nl.float32
    bf16 = nl.bfloat16
    rows = tokens + 1
    v_s = embed_vocab.shape[0]
    ids = sb.tile((rows, 1), nl.int32)
    nisa.memset(dst=ids, value=mask_token)
    nisa.dma_copy(dst=ids[0:tokens, 0:1], src=bonus[0:tokens, 0:1], dge_mode=hwdge, engine=sync, name="vemb_bonus")
    rank_i = sb.tile((rows, 1), nl.int32)
    nisa.dma_copy(dst=rank_i, src=rank_info[0:1, 2:3].broadcast(0, rows), dge_mode=hwdge, engine=sync, name="vemb_rank")
    local = sb.tile((rows, 1), fp32)
    nisa.tensor_copy(dst=local, src=ids)
    start = sb.tile((rows, 1), fp32)
    nisa.tensor_copy(dst=start, src=rank_i)
    nisa.tensor_scalar(dst=start, data=start, op0=nl.multiply, operand0=float(v_s))
    nisa.tensor_tensor(dst=local, data1=local, data2=start, op=nl.subtract)
    owned = sb.tile((rows, 1), fp32)
    nisa.tensor_scalar(dst=owned, data=local, op0=nl.greater_equal, operand0=0.0)
    below = sb.tile((rows, 1), fp32)
    nisa.tensor_scalar(dst=below, data=local, op0=nl.less, operand0=float(v_s))
    nisa.tensor_tensor(dst=owned, data1=owned, data2=below, op=nl.multiply)
    nisa.tensor_tensor(dst=local, data1=local, data2=owned, op=nl.multiply)
    idx = sb.tile((rows, 1), nl.int32)
    nisa.tensor_copy(dst=idx, src=local)
    gathered = sb.tile((rows, HIDDEN), bf16)
    for r in range(rows):
        nisa.dma_copy(
            dst=gathered[r : r + 1, 0:HIDDEN],
            src=embed_vocab.ap(
                pattern=[[HIDDEN, 1], [1, HIDDEN]], offset=0, scalar_offset=idx[r : r + 1, 0:1], indirect_dim=0
            ),
            dge_mode=hwdge,
            engine=sync,
            name=f"vemb_row_{r}",
        )
    # x * owned + 0 gives the owner's value, or +0 (also for a -0 owner value, as the sharded lookup's sum gives).
    masked = sb.tile((rows, HIDDEN), bf16)
    nisa.tensor_scalar(dst=masked, data=gathered, op0=nl.multiply, operand0=owned, op1=nl.add, operand1=0.0)
    nisa.dma_copy(dst=src_hbm, src=masked, dge_mode=hwdge, engine=sync, name="vemb_ar_store")
    check_pair_group(dst_hbm, tp_group, "vemb_ar")
    ncc.all_reduce(srcs=[src_hbm], dsts=[dst_hbm], replica_group=tp_group, op=nl.add, name="vemb_ar")
    out = nl.ndarray((_BLOCK * tokens, HIDDEN), dtype=bf16, buffer=nl.private_hbm, name="vemb_block")
    nisa.dma_copy(
        dst=out[0:tokens, 0:HIDDEN],
        src=dst_hbm[0:tokens, 0:HIDDEN],
        dge_mode=hwdge,
        engine=sync,
        name="vemb_bonus_rows",
    )
    nisa.dma_copy(
        dst=out[tokens : _BLOCK * tokens, 0:HIDDEN],
        src=dst_hbm[tokens:rows, 0:HIDDEN].broadcast(0, (_BLOCK - 1) * tokens),
        dge_mode=hwdge,
        engine=sync,
        name="vemb_mask_rows",
    )
    return out


def _compute_key_start(draft_lo, p_tile, window, tokens, tag, sb):
    """Return an HBM ``key_start`` ``[B, 1]`` INT32 = ``max(draft_lo, P - window)`` (``draft_lo`` when ``window<=0``).

    ``mla_block_attention`` loads ``key_start`` with ``.ap()``, so it must be an HBM tensor (not an SBUF
    tile whose free dim is 1, which fails the partition-step check at B > 1).
    """
    hwdge = nisa.dge_mode.hwdge
    sync = nisa.engine.sync
    out = sb.tile((tokens, 1), nl.int32)
    nisa.dma_copy(dst=out, src=draft_lo[0:tokens, 0:1], dge_mode=hwdge, engine=sync, name=f"{tag}_lo")
    if window > 0:
        floor = sb.tile((tokens, 1), nl.int32)
        nisa.tensor_scalar(dst=floor, data=p_tile, op0=nl.subtract, operand0=float(window))
        nisa.tensor_tensor(dst=out, data1=out, data2=floor, op=nl.maximum)
    key_hbm = nl.ndarray((tokens, 1), dtype=nl.int32, buffer=nl.hbm, name=f"{tag}_hbm")
    nisa.dma_copy(dst=key_hbm, src=out, dge_mode=hwdge, engine=sync, name=f"{tag}_store")
    return key_hbm


def _window_nb(window, nb):
    """Number of CP32 block_table columns the window keeps, or ``nb`` when they span the whole table.

    The window ``[P - window, P)`` touches up to ``ceil(window / 4096) + 1`` columns when ``P`` is not on a
    column edge. tkg_mla_block takes 1 to 4 columns or a power of two of 8 to 256, so the count rounds up to
    the next such width (16 for a 32K window). key_start then masks the keys below ``P - window``, so the
    drafter attends exactly the window, as the CP128 path and the model reference do.
    """
    if window <= 0:
        return nb
    need = -(-window // _WIN_BLOCK) + 1
    nbw = need if need <= 4 else 1 << (need - 1).bit_length()
    return nbw if nbw < nb else nb


def _window_cp128_params(window, nb, cp):
    """CP128 window-slice sizes, or ``None`` when the window keeps the whole table.

    A block_table column holds ``cp * 128`` positions (``span``). The slice keeps
    ``NBW = ceil(window / span) + 1`` columns, so an unaligned window of ``window`` positions stays
    covered: two columns can miss up to ``span - 1`` keys, the third column holds them. Returns
    ``(span, shift, nbw)`` with ``shift = log2(span)`` and ``nbw < nb``.
    """
    if window <= 0:
        return None
    span = cp * _PAGE_BLOCK
    nbw = -(-window // span) + 1
    if nbw >= nb:
        return None
    return span, span.bit_length() - 1, nbw


def _window_block_table(
    block_table,
    p_tile,
    key_start_hbm,
    window,
    tokens,
    tag,
    sb,
    win_block=_WIN_BLOCK,
    win_shift=_WIN_SHIFT,
    nbw=None,
    trim=False,
):
    """Slice ``block_table`` to the window and shift ``key_start`` into the window frame.

    Returns ``(windowed_bt [B, NBW] INT32 HBM, key_start_rel [B, 1] INT32 HBM, window_base [B, 1] INT32
    HBM)``. One column spans ``win_block`` positions (CP32: 4096; CP128: ``cp * 128``). ``NBW`` is
    ``nbw`` when given, else ``ceil(window / win_block)``; ``window_base = first_block * win_block`` is
    subtracted in tkg_mla_block's mask path. The kept columns are the ``NBW`` blocks ending at the block of
    the last prior key (position ``P - 1``), so ``tkg_mla_block`` sees ``NBW`` blocks and runs its static
    path. Because ``tkg_mla_block`` numbers prior keys from block column 0 (``iota`` offset 0), ``key_start``
    is made relative to the first kept block (``key_start_abs - first_block * win_block``). ``position``
    stays absolute: its only use is the future bound ``slot < position - 128 t``, which every window key
    satisfies.

    With ``trim`` (``win_trim``), the table keeps 3 of the 4 key groups of the ``NBW``-column slice: the last 3 when
    the block of ``P - 1`` is at slice column ``NBW - 4`` or later (group 0 is then below ``key_start``), else the
    first 3 (group 3 is then at or above ``P``). The kept groups keep their columns and their order.
    """
    hwdge = nisa.dge_mode.hwdge
    sync = nisa.engine.sync
    int32 = nl.int32
    nb = block_table.shape[1]
    if nbw is None:
        nbw = _window_nb(window, nb)
    # base[b] = clamp(floor((P_b - 1) / win_block) - (NBW - 1), 0, NB - NBW) * win_block.
    pm1 = sb.tile((tokens, 1), int32)
    nisa.tensor_scalar(dst=pm1, data=p_tile, op0=nl.subtract, operand0=1.0)
    low = sb.tile((tokens, 1), int32)
    nisa.tensor_scalar(dst=low, data=pm1, op0=nl.bitwise_and, operand0=float(win_block - 1))
    base = sb.tile((tokens, 1), int32)
    nisa.tensor_tensor(dst=base, data1=pm1, data2=low, op=nl.subtract)
    nisa.tensor_scalar(
        dst=base, data=base, op0=nl.subtract, operand0=float((nbw - 1) * win_block), op1=nl.maximum, operand1=0.0
    )
    nisa.tensor_scalar(dst=base, data=base, op0=nl.minimum, operand0=float((nb - nbw) * win_block))
    if trim:
        # Drop key group 0 (shift the base by one group) when the block of P - 1 is at slice column NBW - 4 or later.
        kernel_assert(
            nbw % _GROUP_COLS == 0 and window <= (nbw - 2 * _GROUP_COLS) * win_block,
            f"win_trim needs a slice of whole key groups and a window of at most {nbw - 2 * _GROUP_COLS} columns; "
            f"got {nbw} columns for a window of {window}",
        )
        shift = sb.tile((tokens, 1), int32)
        nisa.tensor_scalar(
            dst=shift,
            data=pm1,
            op0=nl.greater_equal,
            operand0=float((nbw - _GROUP_COLS) * win_block),
            op1=nl.multiply,
            operand1=float(_GROUP_COLS * win_block),
        )
        nisa.tensor_tensor(dst=base, data1=base, data2=shift, op=nl.add)
        nbw -= _GROUP_COLS
    first_block = sb.tile((tokens, 1), int32)
    nisa.tensor_scalar(dst=first_block, data=base, op0=nl.right_shift, operand0=win_shift)
    # Gather from a flat (B*nb, 1) view: the DGE bounds-checks the indirect index against B*nb, not B.
    bt_flat = block_table.reshape((tokens * nb, 1))
    bt_sb = sb.tile((tokens, nbw), int32)
    for b in range(tokens):
        nisa.dma_copy(
            dst=bt_sb[b : b + 1, 0:nbw],
            src=bt_flat.ap(
                pattern=[[1, 1], [1, nbw]], offset=b * nb, scalar_offset=first_block[b : b + 1, 0:1], indirect_dim=0
            ),
            dge_mode=hwdge,
            engine=sync,
            name=f"{tag}_bt_{b}",
        )
    windowed_bt = nl.ndarray((tokens, nbw), dtype=int32, buffer=nl.hbm, name=f"{tag}_bt")
    nisa.dma_copy(dst=windowed_bt, src=bt_sb, dge_mode=hwdge, engine=sync, name=f"{tag}_bt_store")
    # tkg_mla_block subtracts window_base in the mask; the position stays absolute for RoPE, owner and slot.
    wb_hbm = nl.ndarray((tokens, 1), dtype=int32, buffer=nl.hbm, name=f"{tag}_wb_hbm")
    nisa.dma_copy(dst=wb_hbm, src=base, dge_mode=hwdge, engine=sync, name=f"{tag}_wb_store")
    # key_start_rel = max(0, key_start_abs - base).
    ks = sb.tile((tokens, 1), int32)
    nisa.dma_copy(dst=ks, src=key_start_hbm[0:tokens, 0:1], dge_mode=hwdge, engine=sync, name=f"{tag}_ks")
    nisa.tensor_tensor(dst=ks, data1=ks, data2=base, op=nl.subtract)
    nisa.tensor_scalar(dst=ks, data=ks, op0=nl.maximum, operand0=0.0)
    ks_hbm = nl.ndarray((tokens, 1), dtype=int32, buffer=nl.hbm, name=f"{tag}_ks_hbm")
    nisa.dma_copy(dst=ks_hbm, src=ks, dge_mode=hwdge, engine=sync, name=f"{tag}_ks_store")
    return windowed_bt, ks_hbm, wb_hbm


def _context_stage(
    aux, row0, rope_window, row0_row, bw_minus_base, context_proj, context_norm, kv_a_proj, kv_a_norm, pools,
    verify_slots, rank_info, tp_group, layers, fp8_pages, sb, sources, cp,
):  # fmt: skip
    """Context stage of the step: the verify rows' RoPE rows, then the owner writes of the 5 layers' cache rows."""
    int32 = nl.int32
    tokens = row0.shape[0]
    total_v = _AUX_VERIFY * tokens
    # flat_idx[j*B+b] = (position[b] + bW - base[b]) + j, folded into the per-token row before the +j.
    ctx_pos = sb.tile((1, tokens), int32)
    nisa.tensor_tensor(dst=ctx_pos, data1=row0_row, data2=bw_minus_base, op=nl.add)
    ctx_idx = _own_positions(ctx_pos, _AUX_VERIFY, tokens, sb)
    ctx_rope = _gather_window(rope_window, ctx_idx, total_v, "ctx", sb)
    # Above 4 tokens the context projection streams per aux layer, so one small aux tile is live.
    stream_ctx = tokens > 4
    aux_hT = None if stream_ctx else _build_aux_hT(aux, total_v, sb)
    draft_context(
        aux_hT,
        context_proj,
        context_norm,
        [kv_a_proj[layer] for layer in range(layers)],
        kv_a_norm,
        pools,
        verify_slots,
        row0,
        ctx_rope,
        rank_info,
        tp_group,
        "k3",
        layers=layers,
        fp8_pages=fp8_pages,
        sb=sb,
        sources=sources,
        aux_raw=aux if stream_ctx else None,
        cp=cp,
    )


@nki.jit
def kimi_k3_draft_mla_step(
    aux: nl.NkiTensor,
    bonus: nl.NkiTensor,
    accepted: nl.NkiTensor,
    row0: nl.NkiTensor,
    draft_lo: nl.NkiTensor,
    rider_pool: nl.NkiTensor,
    block_table: nl.NkiTensor,
    verify_slots: nl.NkiTensor,
    rope_window: nl.NkiTensor,
    rope_base: nl.NkiTensor,
    context_proj: nl.NkiTensor,
    context_norm: nl.NkiTensor,
    kv_a_proj: nl.NkiTensor,
    kv_a_norm: nl.NkiTensor,
    mla_in_proj: nl.NkiTensor,
    mla_q_a_norm: nl.NkiTensor,
    mla_q_b_proj: nl.NkiTensor,
    mla_kv_a_norm: nl.NkiTensor,
    mla_kv_b_proj: nl.NkiTensor,
    mla_o_proj: nl.NkiTensor,
    input_norm: nl.NkiTensor,
    post_norm: nl.NkiTensor,
    dense_gate_up: nl.NkiTensor,
    dense_down: nl.NkiTensor,
    final_norm: nl.NkiTensor,
    lm_head: nl.NkiTensor,
    markov_w1: nl.NkiTensor,
    markov_w2: nl.NkiTensor,
    embed: nl.NkiTensor = None,
    block_embed: nl.NkiTensor = None,
    embed_vocab: nl.NkiTensor = None,
    rank_info: nl.NkiTensor = None,
    tp_group: ReplicaGroup = None,
    softmax_scale: float = None,
    layers: int = 5,
    window: int = 0,
    mask_token: int = _MASK_TOKEN,
    loop_path: bool = False,
    return_hidden: bool = False,
    return_layer_hidden: bool = False,
    cp: int = TP,
    window_read: bool = False,
    queries: int = _AUX_VERIFY,
    guard_indices: bool = False,
    tail_rows: int = _BLOCK,
    context_done: bool = False,
    sample_params: nl.NkiTensor = None,
    u_draft: nl.NkiTensor = None,
    rows_attention: bool = False,
    pe_moves: bool = False,
    mla_state_rows: int = 0,
    pair_shared_cc_dsts: tuple = (),
    win_trim: bool = False,
    last_layer_trim: bool = False,
    merged_out_ag: bool = False,
    cc_pad256: bool = False,
    stack_rows: bool = False,
    pass_q_prefetch: int = 0,
    q_gather_buffers: int = 1,
    pass_tail_overlap: bool = False,
    seq_passes: bool = False,
    q_call_one_pass: bool = False,
    markov_w1_local: bool = False,
    state_store_split: bool = False,
    sample_unique_dst: bool = False,
    prior_hoist: int = 0,
    ag_parity: bool = False,
):
    """Run one Kimi-K3 DSpark draft step for B sequences on one rank; return ``[B, 7]`` INT32 drafts.

    The per-layer weights (``mla_*``, ``input_norm``, ``post_norm``, ``dense_*``) are stacked ``[5, ...]``
    and sliced per layer. See ``draft_mla_host.draft_step_rank_weights`` for the host layout.

    ``cp`` is the collective group size: 32 is the TP32 = CP32 replica (the unchanged CP32 path), 128 is the
    DP1 replica on all ranks (one head per rank, ranks 64..127 are zero pad; the kv_a, context, dense MLP,
    lm_head and Markov shards split over 128). The weights must match ``cp`` (``draft_mla_host``).

    Args:
        aux: ``[5, 8B, 7168]`` BF16 HBM verify aux states (token-major ``j*B+b``).
        bonus, accepted, row0, draft_lo: ``[B, 1]`` INT32 HBM anchor token, accepted count, verify row-0
            position, and prior-key lower bound.
        rider_pool: ``[5P, 73728]`` BF16 or FLOAT8_E4M3FN HBM, the 5 layers' pools stacked.
        block_table: ``[B, NB]`` INT32 HBM draft block table.
        verify_slots: ``[8B, 1]`` INT32 HBM context-stage write slots.
        rope_window: ``[B, W, 128]`` FP32 HBM GPT-J ``[cos | sin]`` rows; ``rope_base`` ``[B, 1]`` INT32
            is each window's start position.
        context_proj .. embed: the per-rank weights (``draft_mla_host``).
        rank_info, tp_group, softmax_scale: the rank id, TP group, and drafter softmax scale.
        layers, window, mask_token, loop_path: the layer count, the context window (0 = full), the mask
            token id, and the MLA long-context flag.
        window_read: at CP128 (``cp != TP``) slice the block table to the window columns, in place of
            masking the keys over the full table. A column holds ``cp * 128`` positions, so the slice
            keeps ``ceil(window / (cp * 128)) + 1`` columns and runs tkg_mla_block's static path. Off by
            default; the CP32 path slices the window regardless. The sliced read is bitwise identical to
            the masked full read, since the skipped columns are fully masked.
        embed_vocab: the target's vocab shard ``[V/TP, H]`` BF16 HBM (rows of TP rank ``r`` start at
            ``r * V/TP``). The kernel builds the block embedding from it with one TP all-reduce, bitwise equal
            to ``block_embed`` from the target's sharded lookup. Exclusive with ``embed`` and ``block_embed``.
        queries: the verify rows Q of the step (2 to 8). Below 8, ``aux`` is ``[5, Q*B, H]`` and ``verify_slots``
            ``[Q*B, 1]``; the kernel pads them to 8B rows with zero aux and slot 0, as the host padding does.
        guard_indices: map ``block_table`` entries outside ``[0, pages)`` and ``verify_slots`` outside
            ``[0, pages * 128)`` to 0 (the null page) in the kernel, as the host clamps do.
        tail_rows: the Markov greedy rows of the tail (1 to 7). A k-draft step needs only k: rows below
            ``tail_rows`` keep their bits (each row reads only the row before), and the drafts from ``tail_rows`` on
            are 0. The default 7 keeps the program.
        context_done: skip the context stage, because ``kimi_k3_draft_mla_context`` already wrote the verify rows'
            context rows into ``rider_pool`` (the same inputs give the same rows). The default False keeps the program.
        sample_params, u_draft: optional ``[B, 3]`` (``[top_k, top_p, temperature]``) and ``[B, 7]`` FP32 HBM. With
            them, a row with a temperature of at least 1e-5 draws its drafts (``draft_tail`` sampling), and the
            outputs end with ``q_probs`` ``[B, 7, 256]`` FP32 and ``q_ids`` ``[B, 7, 256]`` INT32. Without them
            the kernel is the greedy drafter. Only the slots below ``tail_rows`` draw; the q lists of the
            other slots are 0.
        rows_attention: attend the prior keys of the 7 block rows of a sequence in one pass (``draft_layer``). The
            default False keeps the program.
        pe_moves: move work into the TensorE waits of the draft MLA, without changing any output. On the CP32
            static path above ``MLA_PASS_ROWS`` rows (B >= 3), each layer gathers the queries of all rows at its
            start with one all-gather in HBM, so the queries of the second pass do not wait on the CCs of the first
            pass's tail. Other forms keep their program. The default False keeps the program.
        mla_state_rows: CP32 static path: the FP32 MLA state reduce-scatter of each layer runs on the persistent HBM
            pair of this many rows (``mla_block_collective_hbm`` ``reduce_rows``), at most ``MLA_PASS_ROWS``. The
            runtime picks the reduce-scatter algorithm, and so the summation order, by the message size: the one
            pass of 7 or 14 rows (0.9 / 1.8 MB) at B = 1 / 2 takes the single-hop mesh and the 16-row passes of
            B >= 3 (2.1 MB) recursive halving, so the drafts of a sequence differ between the buckets. 16 gives
            every bucket the 16-row message. The rows past the pass's rows are not stored, read or used. A step of
            more than 16 rows already runs 16-row passes and keeps its program. 0 (default) keeps the program.
        pair_shared_cc_dsts: names of ``DRAFT_PAIR_SHARED_CC_DST_SITES`` (``check_draft_pair_shared``, trn2 only):
            ``draft_ar`` puts the destination arena of the HBM all-reduces (``draft_hbm_ar_dst``) in LNC1 shared HBM,
            one buffer for the two cores of an HBM stack. It needs the neuronx-cc option --lnc1-with-shared-mem and a
            runtime with pair-shared HBM; the sources and every other tensor stay private and the values do not
            change. () (default) keeps the program.
        win_trim: CP32 window slice: attend 3 key groups (12 columns), not the 4 groups of the 16 columns of a 32K
            window. The cut group holds no window key: group 0 is below ``key_start`` once the block of ``P - 1`` is
            at slice column 12 or later, else group 3 is at or above ``P``. Its keys take exact zero weights, so the
            outputs do not change. Not with ``rows_attention``: its paired group sums would change their order. The
            default False keeps the program.
        last_layer_trim: with ``tail_rows`` below 7, the last layer attends only the passes of the rows ``j <
            tail_rows`` (token-major rows below ``tail_rows * B``), the only rows that the tail reads; the later rows
            get a zero attention output. The drafts, the q lists and the rider pool do not change; the hidden rows
            from ``tail_rows * B`` on do (``return_hidden``). CP32 static path, not with ``rows_attention``. The
            default False keeps the program.
        merged_out_ag: each layer's MLA passes write their output reduce-scatter slices to one ``ag_call`` tile, and
            the layer runs one output all-gather in place of one per pass (``mla_block_attention`` ``out_ag_call``).
            The reduce-scatters do not change and the all-gather is a copy, so the outputs do not change. CP32 static
            path. The default False keeps the program.
        cc_pad256: the FP32 HBM pair of the MLA state reduce-scatter has padded columns, so that each rank's part is a
            multiple of 256 bytes (``mla_block_collective_hbm`` ``cc_pad256``; the pad columns are not read, so the
            outputs do not change). CP32 static path. The default False keeps the program.
        stack_rows: CP32 static path: each layer attends each prior-key group for 2 rows of a sequence of the pass
            at once (``mla_block_attention`` ``stack_rows``): one set of score matmuls, cap, exp, weight transposes and
            value matmuls for the unit, with row ``u`` on the stationary columns and PSUM partitions ``64 u ..`` (64 heads a row), in place
            of one set for each row. Each row gets the numerator, the weight sum and the state adds of the per-row form, so
            the outputs do not change. Not with ``rows_attention``. The default False keeps the program.
        pass_q_prefetch, q_gather_buffers: with ``pe_moves`` above one pass (the layer's query gather in HBM), the query
            gather tiles of the later passes (``mla_block_attention``): ``pass_q_prefetch`` (at most 2) of them load into
            their own tiles on the ScalarE hardware DGE queue right after the layer's query gather, so they do not wait
            for the collectives of the pass before, and ``q_gather_buffers`` 2 loads the others into two tiles in turn.
            The same bytes and instructions, so the outputs do not change. A step of one pass (16 rows or fewer) keeps its
            program. Layer 0's prefetch loads follow the last ScalarE-queue row load of the context stage and the block
            embed in the schedule (a schedule edge, no hardware wait), so those rows keep their place on the ScalarE
            queue. The defaults (0, 1) keep the program.
        pass_tail_overlap: CP32 static path above one MLA pass (more than 16 rows, B >= 3): each layer runs the output
            tail of a pass (the state load, the value and output projections, the output reduce-scatter and
            all-gather) after the attention of the next pass, so the state store and reduce-scatter of the pass run
            under that attention (``mla_block_attention`` ``pass_tail_overlap``; the odd passes take a second persistent
            HBM state pair). The same instructions and collective sizes, so the outputs do not change. A step of one
            pass keeps its program. The default False keeps the program.
        seq_passes: with ``pe_moves`` on the CP32 static path above one MLA pass, each layer attends its passes on
            whole sequences: 2 sequences (14 rows) a pass in place of 16 token-major rows of every sequence, so each
            sequence's prior keys are loaded and prepared once a layer, not once in every pass (``mla_block_attention``
            ``seq_passes``). Each row takes the same arithmetic in the same order and the collectives keep their
            sizes, so the outputs do not change. B must be even. Not with ``last_layer_trim``. A
            step of one pass keeps its program. The default False keeps the program.
        q_call_one_pass: CP32 static path, a step of one MLA pass (16 rows or fewer, B <= 2): each layer gathers the
            queries of all rows with one all-gather in HBM (``mla_block_attention`` ``q_hbm``, the gather of
            ``pe_moves`` above one pass), in place of one all-gather for each 4-row query tile. The tile gathers run
            one after another on one collective stream, so the attention of each later tile waits for its gather. The
            gathered bytes are the same and only their layout changes, so the outputs do not change. The query tile
            options of the later passes (``pass_q_prefetch``, ``q_gather_buffers``) stay off: one pass has no later
            pass. A step of more than one pass keeps its program. The default False keeps the program.
        markov_w1_local: ``markov_w1`` is the full ``[V, 256]`` Markov w1 on every rank
            (``draft_mla_host.draft_step_rank_weights(markov_w1_full=True)``, 84 MB a rank at V = 163840). Each tail row
            gathers ``markov_w1[prev]`` itself and runs no w1 all-reduce (``draft_tail`` ``markov_w1_local``). The default
            False keeps the program (the vocab shard and the all-reduce).
        state_store_split: each layer stores its FP32 MLA state (1.84 MB a rank at B = 2) as two column halves on the
            SP and ScalarE HWDGE queues, in place of one DMA (``draft_layer`` ``state_store_split``). The outputs do not
            change. The default False keeps the program.
        sample_unique_dst: with the sampled tail, the candidate all-gather of each slot writes its own FP32 destination
            (a column window of one region-heap tile that the tail allocates before the lm_head, 14.4 KB a partition) in
            place of the destination that all 7 slots share. A write to anchor columns makes the tile live from there, so
            no gather writes bytes that an earlier instruction wrote, and the compiler can skip their start handshakes.
            The tile comes after every earlier heap tile, so no earlier tile moves. The same values, so the outputs do not change. Without the sampled tail, it changes nothing. False
            (default) keeps the program.
        prior_hoist: with ``stack_rows``, each layer traces the page loads and the TensorE key transposes of the first
            this-many prior-key groups of each sequence before each MLA pass's query loads (``draft_layer``
            ``prior_hoist``), so they run while the query gather is in flight; the page loads take the ScalarE HWDGE
            queue. The same loads and transposes on the same values, so the outputs do not change. 0 (default) keeps the
            program.
        ag_parity: the all-gather sources that one step uses again take two persistent tiles, which alternate: the
            tail gather source of the 7 slots (slot j takes tile j mod 2), and on the CP32 static path the qkv
            all-gather pair of the 5 layers (layer n takes pair n mod 2). The next use then writes the other source,
            not the source that a peer of the last gather can still read. The values do not change. False (default)
            keeps the program. Test only: a released runtime can give wrong values when one persistent SBUF tile is the
            source of many collectives in one execution.
    """
    draft_ar_shared = check_draft_pair_shared(pair_shared_cc_dsts)
    sb = AUTO_SBUF
    sampled = sample_params is not None
    tail_pairs = 2 * _sample_width(cp) // cp if sampled else 2
    int32 = nl.int32
    hwdge = nisa.dge_mode.hwdge
    sync = nisa.engine.sync
    tokens = row0.shape[0]
    total_v = _AUX_VERIFY * tokens
    total_q = _BLOCK * tokens
    # Trn2 reads the FP8 pages in its legacy FP8 format (``fp8_page_dtype``); the aliased output keeps the input view.
    page_pool = (
        rider_pool.view(fp8_page_dtype()) if rider_pool.dtype == nl.float8_e4m3fn and not is_trn3() else rider_pool
    )
    layer_pages = page_pool.shape[0] // layers
    pools = [page_pool[layer * layer_pages : (layer + 1) * layer_pages, 0:PAGE_WIDTH] for layer in range(layers)]
    kernel_assert(2 <= queries <= _AUX_VERIFY, f"queries must be 2 to {_AUX_VERIFY}, got {queries}")
    kernel_assert(1 <= tail_rows <= _BLOCK, f"tail_rows must be 1 to {_BLOCK}, got {tail_rows}")
    kernel_assert(
        not pass_tail_overlap or (cp == TP and not loop_path),
        f"pass_tail_overlap is for the CP32 static path; got cp={cp}, loop_path={loop_path}",
    )
    # Only a step of more than one MLA pass changes its program.
    tail_overlap = pass_tail_overlap and total_q > MLA_PASS_ROWS
    kernel_assert(
        not seq_passes or (cp == TP and not loop_path and pe_moves and not last_layer_trim),
        f"seq_passes is for the CP32 static path with pe_moves, not last_layer_trim; got cp={cp}, "
        f"loop_path={loop_path}, pe_moves={pe_moves}, last_layer_trim={last_layer_trim}",
    )
    seq_on = seq_passes and total_q > MLA_PASS_ROWS
    kernel_assert(
        0 <= mla_state_rows <= MLA_PASS_ROWS and (mla_state_rows == 0 or (cp == TP and not loop_path)),
        f"mla_state_rows is 0 to {MLA_PASS_ROWS} rows of the HBM MLA state reduce-scatter of the CP32 static path; "
        f"got {mla_state_rows} at cp={cp}, loop_path={loop_path}",
    )
    # The sampled tail takes both sampling inputs, in the shapes and dtypes that it loads, and returns 256
    # candidates per slot; any other form is refused rather than traced as the greedy tail.
    kernel_assert(sampled == (u_draft is not None), "sample_params and u_draft go together")
    if sampled:
        kernel_assert(
            tuple(sample_params.shape) == (tokens, 3) and sample_params.dtype == nl.float32,
            f"sample_params must be [B, 3] = [{tokens}, 3] FP32, got {tuple(sample_params.shape)} {sample_params.dtype}",
        )
        kernel_assert(
            tuple(u_draft.shape) == (tokens, _BLOCK) and u_draft.dtype == nl.float32,
            f"u_draft must be [B, {_BLOCK}] = [{tokens}, {_BLOCK}] FP32, got {tuple(u_draft.shape)} {u_draft.dtype}",
        )
        kernel_assert(
            _sample_width(cp) == _SAMPLE_K,
            f"the sampled tail returns {_SAMPLE_K} candidates per slot; cp={cp} gives {_sample_width(cp)}",
        )
    kernel_assert(
        embed_vocab is None or (embed is None and block_embed is None),
        "embed_vocab replaces embed and block_embed: pass only one of them",
    )
    if queries < _AUX_VERIFY:
        aux = _pad_verify_aux(aux, tokens, queries, sb)
    # ``seq_passes``: the guard's ops stay on VectorE (the compiler moved its first copy to GpSimd in a seq_passes
    # program); other steps keep the compiler's choice, and so their programs.
    guard_engine = nisa.engine.vector if seq_on else None
    if queries < _AUX_VERIFY or guard_indices:
        slot_limit = layer_pages * _PAGE_BLOCK if guard_indices else None
        verify_slots = _guard_rows(
            verify_slots, queries * tokens, total_v, slot_limit, "slots", sb, engine=guard_engine
        )
    if guard_indices:
        block_table = _guard_rows(block_table, tokens, tokens, layer_pages, "table", sb, engine=guard_engine)
    fp8_pages = page_pool.dtype != nl.bfloat16
    # CP-dependent shards. At CP32 the context columns and the kv_a row split are contiguous; at CP128 the
    # kv_a shard is the four interleaved kv latent features plus one k_pe column (see ``draft_context``).
    h_shard = HIDDEN // cp
    kv_a_shard = CACHE_WIDTH // cp if cp == TP else KV_LORA // cp + 1
    mla_local_heads = 2 if cp == TP else 1
    mla_coll_kwargs = {} if cp == TP else {"cp": cp, "heads": _DRAFT_HEADS}

    # The CP128 windowed read keeps ceil(W/span)+1 columns (3 for a 32K window), which fit tkg_mla_block's
    # static unrolled path, so the step drops the loop region (its per-layer release-heap stack is for the
    # full-table block loop). Off at CP32 and when window_read is False, so those programs do not change.
    window_cp128 = (
        _K1_HAS_KEY_START
        and cp != TP
        and window_read
        and _window_cp128_params(window, block_table.shape[1], cp) is not None
    )
    loop_path = loop_path and not window_cp128
    # pe_moves: one query all-gather a layer in HBM on the CP32 static path above one pass (B >= 3).
    # q_call_one_pass: the same gather at one pass (B <= 2).
    kernel_assert(
        not q_call_one_pass or (cp == TP and not loop_path),
        f"q_call_one_pass is for the CP32 static path; got cp={cp}, loop_path={loop_path}",
    )
    multi_pass = total_q > MLA_PASS_ROWS
    q_hbm = cp == TP and not loop_path and ((pe_moves and multi_pass) or (q_call_one_pass and not multi_pass))
    # The query tile options of the later passes act only above one pass.
    q_multi = q_hbm and multi_pass
    q_kw = {"q_hbm": True} if q_hbm else {}

    # The CP32 static path does not fit SBUF above 8 sequences at any context.
    kernel_assert(
        loop_path or cp != TP or tokens <= 8,
        f"The CP32 static drafter supports at most 8 sequences, got {tokens}: it overflows SBUF. "
        "Use loop_path=True, cp=128, or at most 8 sequences.",
    )

    # The explicit region holds the TensorE transpose identities (install_identities).
    region = ExplicitSbuf()
    install_identities(region)
    bf16 = nl.bfloat16
    v_s = lm_head.shape[1]
    kernel_assert(
        embed_vocab is None or embed_vocab.shape == (v_s, HIDDEN),
        f"embed_vocab must be the [V/TP, H] = [{v_s}, {HIDDEN}] vocab shard of the lm_head's TP group, "
        f"got {None if embed_vocab is None else tuple(embed_vocab.shape)}",
    )
    heap = SbufTiles(region, heap=True)
    # Persistent collective tiles, one per site for the whole kernel: a peer can read a gather source after the
    # local output lands, and write a destination before the local rank reaches the gather. On the static path
    # the sources and the ``*_dst`` destinations are views of two SBUF arenas (``_ArenaPlan`` routes each tile by
    # name), both reserved to the tail argmax gather, so a source and its destination are never in one tensor.
    # The all-reduces (dense MLP, Markov) run in persistent per-rank HBM tensors with no start clear. On the loop
    # path the block-end barriers separate the uses, so the tiles are region heap tiles.
    plan = heap if loop_path else _ArenaPlan()
    tiles = {
        "ctx_src": plan.tile((total_v, h_shard), bf16, "ctx_ag_src"),
        # One batched kv_a gather over all layers (layer-major source, rank-major gather output).
        "kva_src": plan.tile((total_v, layers * kv_a_shard), bf16, "kva_ag_src"),
        "emb_src": plan.tile((total_q, h_shard), bf16, "emb_ag_src"),
        # The tail argmax source anchors the source arena to the kernel end (it is the last collective's
        # source), so the gather sources stay reserved past the last layer. It is a gather source, never
        # reshaped, so it has no single-partition access-pattern issue (unlike the destination below).
        "tail_src": plan.tile((tokens, tail_pairs), nl.float32, "tail_ag_src"),
    }
    if ag_parity:
        # The second tail gather source (ag_parity): slot j of the tail takes tail_src or tail_src_1 by j mod 2.
        tiles["tail_src_1"] = plan.tile((tokens, tail_pairs), nl.float32, "tail_ag_src_1")
    early_shapes = {
        "ctx_dst": ((total_v, HIDDEN), "ctx_ag_dst"),
        "kva_dst": ((total_v, cp * layers * kv_a_shard), "kva_ag_dst"),
        "emb_dst": ((total_q, HIDDEN), "emb_ag_dst"),
    }
    # Above 4 sequences (CP32 static) the early gather destinations are plain tiles, free after the embed stage.
    early_dsts = tuple(early_shapes) if not loop_path and cp == TP and tokens > 4 else ()
    tiles.update(
        {key: plan.tile(shape, bf16, name) for key, (shape, name) in early_shapes.items() if key not in early_dsts}
    )
    # The MLA body's gather sources (qkv, ag) and destination (qkv_dst), one set reused across the 5
    # layers. At CP32 they pack into the two collective-safe arenas (the CP32 path). At CP128 the output
    # reduce-scatter tile is one partition per rank, and a one-partition sub-view of the explicit packed
    # arena is an illegal partition step, so the MLA sources are standalone region-heap tiles (each its own
    # tensor, so a source and its destination are still never one tensor).
    mla_in_arena = not loop_path and cp == TP
    # with_rs reserves the CP32 output reduce-scatter sources; the CP128 drafter reduces in HBM.
    mla_tiles = (
        mla_block_collective_sources(
            total_q,
            mla_local_heads,
            plan,
            with_rs=True,
            out_ag_call=merged_out_ag,
            **mla_coll_kwargs,
            **q_kw,
            # ``pass_tail_overlap``: one output all-gather destination a pass, in the destination arena.
            **({"y_dst_passes": -(-total_q // MLA_PASS_ROWS)} if tail_overlap and not merged_out_ag else {}),
        )
        if mla_in_arena
        else {}
    )
    if ag_parity and mla_in_arena:
        # The second qkv all-gather pair of the layers (ag_parity), in the same two arenas: layer n takes pair n mod 2.
        for key, name in (("qkv", "mla_qkv_src_p1"), ("qkv_dst", "mla_qkv_dst_p1")):
            _, _, shape, dtype = plan.slots[mla_tiles[key]]
            mla_tiles[f"{key}_p1"] = plan.tile(shape, dtype, name)
    # CP128 static path: the qkv gather pair in the reserved arenas, the one-partition ag tiles in the region heap.
    cp_static = not loop_path and cp != TP
    cp_tiles = (
        mla_block_collective_sources(total_q, mla_local_heads, heap, gather_sb=plan, **mla_coll_kwargs)
        if cp_static
        else {}
    )
    # The context, kv_a and embed gathers and the tail gathers write every byte of their destinations, so the start
    # clear leaves those destinations out: a local write in a gather's window would cost the gather its handshake skip.
    clear_skip = {}
    if loop_path:
        sources = dict(tiles)
        # CP128: one persistent set of MLA HBM collectives; per-layer tensors can alias a source a peer still reads.
        mla_sources = None if cp == TP else mla_block_collective_hbm(total_q, mla_local_heads, **mla_coll_kwargs)
        bases = list(sources.values())
    else:
        arenas = tuple(region.heap((P_MAX, w), bf16, f"draft_arena_{i}") for i, w in enumerate(plan.widths))
        sources = {key: plan.view(arenas, slot) for key, slot in tiles.items()}
        bases = list(arenas)
        clear_skip[id(arenas[1])] = [plan.columns(tiles[key]) for key in early_shapes if key in tiles]
        for key in early_dsts:
            shape, name = early_shapes[key]
            sources[key] = sb.tile(shape, bf16, name)
        if mla_in_arena:
            mla_sources = {
                key: [plan.view(arenas, s) for s in slot] if isinstance(slot, list) else plan.view(arenas, slot)
                for key, slot in mla_tiles.items()
            }
        else:
            mla_sources = {key: plan.view(arenas, cp_tiles[key]) for key in ("qkv", "qkv_dst")}
            mla_sources["ag"] = cp_tiles["ag"]
            # Clear the standalone ag tiles at the kernel start, as the arena memset does for the other tiles.
            bases += list(cp_tiles["ag"])
        # The MLA state reduce-scatter runs in persistent per-rank HBM (not cleared at start). At CP128 it
        # also returns the per-pass query gather pair (the query gather runs in HBM, not SBUF).
        state_kw = {"reduce_rows": mla_state_rows} if mla_state_rows else {}
        if cc_pad256:
            state_kw["cc_pad256"] = True
        if tail_overlap:
            state_kw["state_pairs"] = 2
        mla_sources.update(mla_block_collective_hbm(total_q, mla_local_heads, **mla_coll_kwargs, **q_kw, **state_kw))
    # The dense-MLP and Markov all-reduces run in persistent per-rank HBM. All HBM collective SOURCES are
    # disjoint slices of one private_hbm arena and all DESTINATIONS slices of a second, so each slice's live
    # range runs to the last HBM collective (the Markov AR) and no site loses its bytes to another. Source
    # and destination are different tensors. Neither arena is cleared at the kernel start. (The MLA state
    # reduce-scatter HBM stays in its own mla_block_collective_hbm tensors: that helper takes no plan, so it
    # cannot slice this arena.)
    # Two FLAT (1D) per-rank HBM arenas, one for the AR sources, one for the destinations. The dense-MLP
    # slice (FP32) and the Markov slice (BF16) are laid end to end, so each is a CONTIGUOUS reshaped view
    # (a 2D column-slice would give the collective a strided partition access -- NCC_INLA001). Each arena
    # lives to the last HBM collective (the Markov AR), so no site aliases another. No start clear.
    _mlp_bf16 = P_MAX * total_q * _H_TILES * 2  # FP32 MLP slice measured in BF16 elements
    _mk_bf16 = tokens * _MARKOV_RANK
    # The vocab-shard block embedding all-reduce takes a third slice (only with embed_vocab).
    _emb_bf16 = (tokens + 1) * HIDDEN if embed_vocab is not None else 0
    _arena = _mlp_bf16 + _mk_bf16 + _emb_bf16
    hbm_src = nl.ndarray((_arena,), dtype=bf16, buffer=nl.private_hbm, name="draft_hbm_ar_src")
    hbm_dst = nl.ndarray((_arena,), dtype=bf16, buffer=pair_shared_buffer(draft_ar_shared), name="draft_hbm_ar_dst")
    sources["mlp_src_hbm"] = hbm_src[0:_mlp_bf16].view(nl.float32).reshape((P_MAX, total_q * _H_TILES))
    sources["mlp_dst_hbm"] = hbm_dst[0:_mlp_bf16].view(nl.float32).reshape((P_MAX, total_q * _H_TILES))
    sources["markov_src_hbm"] = hbm_src[_mlp_bf16 : _mlp_bf16 + _mk_bf16].reshape((tokens, _MARKOV_RANK))
    sources["markov_dst_hbm"] = hbm_dst[_mlp_bf16 : _mlp_bf16 + _mk_bf16].reshape((tokens, _MARKOV_RANK))
    if embed_vocab is not None:
        _e0 = _mlp_bf16 + _mk_bf16
        sources["emb_src_hbm"] = hbm_src[_e0 : _e0 + _emb_bf16].reshape((tokens + 1, HIDDEN))
        sources["emb_dst_hbm"] = hbm_dst[_e0 : _e0 + _emb_bf16].reshape((tokens + 1, HIDDEN))
    # The Markov bias is the bias matmul output, not a collective tile, so it is its own region-heap tile.
    sources["markov_bias"] = heap.tile((tokens, v_s), bf16, "markov_bias")
    bases.append(sources["markov_bias"])
    # The tail argmax DESTINATION is its own region-heap tile, not an arena view: the argmax reshapes it
    # ([tokens, TP, 2]), and at B=1 (tokens=1) a reshape of an arena view builds an illegal partition-step
    # access pattern (NCC_INLA001). A standalone tile has its own row and a valid step. It is written only
    # by its own gather (destinations are not reused as sources), so it needs no arena anchor.
    sources["tail_dst"] = heap.tile((tokens, tail_pairs * cp), nl.float32, "tail_ag_dst")
    # Clear the SBUF bases at the start, so no earlier tile shares the bytes of a collective source or destination.
    for _base in bases:
        skip = sorted(clear_skip.get(id(_base), ()))
        if not skip:
            nisa.memset(dst=_base, value=0.0)
            continue
        start = 0
        for c0, c1 in skip + [(_base.shape[1], _base.shape[1])]:
            if c0 > start:
                nisa.memset(dst=_base[0:P_MAX, start:c0], value=0.0)
            start = max(start, c1)

    # Device scalar rows on partition 0: row0, accepted, P = row0 + accepted, and b*W - base.
    row0_row = sb.tile((1, tokens), int32)
    acc_row = sb.tile((1, tokens), int32)
    base_row = sb.tile((1, tokens), int32)
    nisa.dma_copy(
        dst=row0_row,
        src=row0.ap(pattern=[[tokens, 1], [1, tokens]], offset=0),
        dge_mode=hwdge,
        engine=sync,
        name="row0_row",
    )
    nisa.dma_copy(
        dst=acc_row,
        src=accepted.ap(pattern=[[tokens, 1], [1, tokens]], offset=0),
        dge_mode=hwdge,
        engine=sync,
        name="acc_row",
    )
    nisa.dma_copy(
        dst=base_row,
        src=rope_base.ap(pattern=[[tokens, 1], [1, tokens]], offset=0),
        dge_mode=hwdge,
        engine=sync,
        name="base_row",
    )
    p_row = sb.tile((1, tokens), int32)
    nisa.tensor_tensor(dst=p_row, data1=row0_row, data2=acc_row, op=nl.add)
    bw = sb.tile((1, tokens), int32)
    nisa.iota(dst=bw, pattern=[[rope_window.shape[1], tokens]], offset=0)
    bw_minus_base = sb.tile((1, tokens), int32)
    nisa.tensor_tensor(dst=bw_minus_base, data1=bw, data2=base_row, op=nl.subtract)

    # One reserved arena keeps every source live to the kernel end. With the query prefetch of the later passes, record
    # the ScalarE-queue row loads of the context stage and the block embed: layer 0's prefetch loads take a schedule edge
    # after the last of them, so the scheduler cannot put them ahead of those rows on the ScalarE queue.
    prefetch_order = pass_q_prefetch and q_multi
    with _record_act_row_loads() if prefetch_order else nullcontext() as act_row_loads:
        if not context_done:
            _context_stage(
                aux, row0, rope_window, row0_row, bw_minus_base, context_proj, context_norm, kv_a_proj, kv_a_norm,
                pools, verify_slots, rank_info, tp_group, layers, fp8_pages, sb, sources, cp,
            )  # fmt: skip

        # block_embed skips the device gather; else gather [bonus, mask x 6] and all-gather over TP.
        if embed_vocab is not None:
            block_embed = _embed_vocab_rows(
                embed_vocab,
                bonus,
                mask_token,
                tokens,
                rank_info,
                tp_group,
                sources["emb_src_hbm"],
                sources["emb_dst_hbm"],
                sb,
            )
        if block_embed is not None:
            h_hT = _load_rows_hT(block_embed, total_q, sb)
        else:
            h_hT = _embed_block(embed, bonus, mask_token, tokens, tp_group, "emb", sb, sources=sources)
    blk_pos = sb.tile((1, tokens), int32)
    nisa.tensor_tensor(dst=blk_pos, data1=p_row, data2=bw_minus_base, op=nl.add)
    blk_idx = _own_positions(blk_pos, _BLOCK, tokens, sb)
    blk_rope = _gather_window(rope_window, blk_idx, total_q, "blk", sb)

    p_tile = sb.tile((tokens, 1), int32)
    nisa.dma_copy(dst=p_tile, src=row0[0:tokens, 0:1], dge_mode=hwdge, engine=sync, name="p_row0")
    acc_tile = sb.tile((tokens, 1), int32)
    nisa.dma_copy(dst=acc_tile, src=accepted[0:tokens, 0:1], dge_mode=hwdge, engine=sync, name="p_acc")
    nisa.tensor_tensor(dst=p_tile, data1=p_tile, data2=acc_tile, op=nl.add)
    position = nl.ndarray((tokens, 1), dtype=int32, buffer=nl.hbm, name="draft_position")
    nisa.dma_copy(dst=position, src=p_tile, dge_mode=hwdge, engine=sync, name="p_store")
    zero_slot = nl.ndarray((total_q, 1), dtype=int32, buffer=nl.hbm, name="draft_zero_slot")
    zero_sb = sb.tile((total_q, 1), int32)
    nisa.memset(dst=zero_sb, value=0)
    nisa.dma_copy(dst=zero_slot, src=zero_sb, dge_mode=hwdge, engine=sync, name="zero_slot_store")

    key_start = None
    if _K1_HAS_KEY_START:
        key_start = _compute_key_start(draft_lo, p_tile, window, tokens, "key_start", sb)

    # The window passes tkg_mla_block only the blocks that end at the current position (static path).
    bt_for_layers = block_table
    window_base = None
    trimmed = False
    if key_start is not None and cp == TP:
        # CP32: a column holds 4096 positions; keep the columns of _window_nb (16 for a 32K window).
        if _window_nb(window, block_table.shape[1]) < block_table.shape[1]:
            bt_for_layers, key_start, window_base = _window_block_table(
                block_table, p_tile, key_start, window, tokens, "win", sb, trim=win_trim
            )
            trimmed = win_trim
    elif window_cp128:
        # CP128: a column holds cp*128 positions; keep ceil(window / span) + 1 columns (window_read).
        span, shift, nbw = _window_cp128_params(window, block_table.shape[1], cp)
        bt_for_layers, key_start, window_base = _window_block_table(
            block_table, p_tile, key_start, window, tokens, "win", sb, win_block=span, win_shift=shift, nbw=nbw
        )
    kernel_assert(
        trimmed == win_trim and not (win_trim and loop_path),
        "win_trim needs the CP32 window slice on the static path (window > 0, a table wider than the slice)",
    )
    kernel_assert(
        not (rows_attention and win_trim), "win_trim keeps the per-row draft attention: not with rows_attention"
    )
    kernel_assert(
        not last_layer_trim or (tail_rows < _BLOCK and cp == TP and not loop_path and not rows_attention),
        f"last_layer_trim needs tail_rows below {_BLOCK} on the CP32 static path without rows_attention; got "
        f"tail_rows={tail_rows}, cp={cp}, loop_path={loop_path}, rows_attention={rows_attention}",
    )
    last_rows = tail_rows * tokens if last_layer_trim else 0
    kernel_assert(
        not merged_out_ag or (cp == TP and not loop_path),
        f"merged_out_ag is for the CP32 static path; got cp={cp}, loop_path={loop_path}",
    )
    kernel_assert(
        not cc_pad256 or (cp == TP and not loop_path),
        f"cc_pad256 is for the CP32 static path; got cp={cp}, loop_path={loop_path}",
    )
    if total_q > MLA_PASS_ROWS:
        # Above one pass the query tile options need the layer query gather; one pass keeps its program either way.
        kernel_assert(
            (pass_q_prefetch == 0 and q_gather_buffers == 1) or q_hbm,
            f"pass_q_prefetch {pass_q_prefetch} and q_gather_buffers {q_gather_buffers} need the layer query gather "
            "(pe_moves on the CP32 static path above one pass)",
        )
    kernel_assert(
        not stack_rows or (cp == TP and not loop_path and not rows_attention),
        f"stack_rows is for the CP32 static path without rows_attention; got cp={cp}, loop_path={loop_path}, "
        f"rows_attention={rows_attention}",
    )

    # The loop path needs the MLA body's SBUF tiles at explicit addresses (NCC_IIGCA118 otherwise). The region
    # (already created, with install_identities) holds the residual streams, the block rope, the
    # all-ones, and the dense-MLP all-reduce source across the 5 layers; each layer's MLA body runs in a
    # release-heap frame. The context stage and the tail have no runtime loop, so they stay on the automatic allocator.
    layer_ones = None
    streams = None
    mlp_src_hbm = sources["mlp_src_hbm"]
    mlp_dst_hbm = sources["mlp_dst_hbm"]
    if loop_path:
        layer_ones = explicit_ones(region)
        streams = [region.heap((P_MAX, total_q, _H_TILES), nl.bfloat16, f"draft_stream_{i}") for i in range(2)]
        nisa.tensor_copy(dst=streams[0], src=h_hT)
        blk_rope_heap = region.heap((total_q, 2 * ROPE), nl.float32, "draft_blk_rope")
        nisa.tensor_copy(dst=blk_rope_heap, src=blk_rope)
        blk_rope = blk_rope_heap

    # Test-only diagnostic: the residual stream after each layer (before final_norm), token-major
    # [7B, H]. The test teacher-forces each layer with the kernel's previous-layer output to tell a
    # real per-layer bug from accumulated rounding.
    layer_hidden = None
    if return_layer_hidden:
        layer_hidden = nl.ndarray(
            (layers, total_q, HIDDEN), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="layer_hidden"
        )
    for layer in range(layers):
        layer_in = streams[layer % 2] if loop_path else h_hT
        layer_out = streams[(layer + 1) % 2] if loop_path else None
        layer_sb = SbufTiles(region) if loop_path else sb
        frame = region.frame(f"L{layer}", release_heap=True) if loop_path else nullcontext()
        with frame:
            h_hT = draft_layer(
                layer_in,
                mla_in_proj[layer],
                mla_q_a_norm[layer],
                mla_q_b_proj[layer],
                mla_kv_a_norm[layer],
                mla_kv_b_proj[layer],
                mla_o_proj[layer],
                input_norm[layer],
                post_norm[layer],
                dense_gate_up[layer],
                dense_down[layer],
                pools[layer],
                bt_for_layers,
                zero_slot,
                position,
                blk_rope,
                rank_info,
                tp_group,
                f"L{layer}",
                softmax_scale=softmax_scale,
                key_start=key_start,
                window_base=window_base,
                loop_path=loop_path,
                fp8_pages=fp8_pages,
                sb=layer_sb,
                out=layer_out,
                ar_src_hbm=mlp_src_hbm,
                ar_dst_hbm=mlp_dst_hbm,
                ones=layer_ones,
                mla_sources=_layer_mla_sources(mla_sources, layer),
                cp=cp,
                heads=_DRAFT_HEADS,
                rows_attention=rows_attention,
                q_hbm=q_hbm,
                trimmed_window=trimmed,
                query_rows=last_rows if layer == layers - 1 else 0,
                out_ag_call=merged_out_ag,
                stack_rows=stack_rows,
                pass_q_prefetch=pass_q_prefetch if q_multi else 0,
                q_gather_buffers=q_gather_buffers if q_multi else 1,
                q_prefetch_after=tuple(act_row_loads[-1:]) if (prefetch_order and layer == 0) else (),
                pass_tail_overlap=tail_overlap,
                seq_passes=seq_on,
                state_store_split=state_store_split,
                prior_hoist=prior_hoist,
            )
        if return_layer_hidden:
            store_hT_rows(layer_hidden[layer], h_hT, sb=sb)

    hidden_out = None
    if return_hidden:
        hidden_out = nl.ndarray((total_q, HIDDEN), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="hidden")
    tail = draft_tail(
        h_hT,
        final_norm,
        lm_head,
        markov_w1,
        markov_w2,
        bonus,
        rank_info,
        tp_group,
        "k4",
        sb=sb,
        hidden_out=hidden_out,
        sources=sources,
        cp=cp,
        tail_rows=tail_rows,
        markov_w1_local=markov_w1_local,
        unique_dst_heap=heap if sample_unique_dst else None,
        sampling=(
            {"sample_params": sample_params, "u_draft": u_draft}
            if sampled
            else None
        ),
    )
    drafts = tail[0] if sampled else tail
    out = nl.ndarray((tokens, _BLOCK), dtype=int32, buffer=nl.shared_hbm, name="drafts")
    nisa.dma_copy(dst=out, src=drafts, dge_mode=hwdge, engine=sync, name="drafts_store")
    q_outputs = []
    if sampled:
        for name, tile, dtype in (("q_probs", tail[1], nl.float32), ("q_ids", tail[2], int32)):
            q_hbm = nl.ndarray((tokens, _BLOCK, _sample_width(cp)), dtype=dtype, buffer=nl.shared_hbm, name=name)
            nisa.dma_copy(
                dst=q_hbm.reshape((tokens, _BLOCK * _sample_width(cp))),
                src=tile,
                dge_mode=hwdge,
                engine=sync,
                name=f"{name}_store",
            )
            q_outputs.append(q_hbm)
    # The aliased rider_pool output stays second (its original position); the optional hidden and
    # per-layer hidden follow in that order, matching the test's output-key order, so no output name
    # is registered twice.
    outputs = [out, rider_pool]
    if return_hidden:
        outputs.append(hidden_out)
    if return_layer_hidden:
        outputs.append(layer_hidden)
    outputs += q_outputs
    return tuple(outputs)


@nki.jit
def kimi_k3_draft_mla_context(
    aux: nl.NkiTensor,
    row0: nl.NkiTensor,
    rider_pool: nl.NkiTensor,
    verify_slots: nl.NkiTensor,
    rope_window: nl.NkiTensor,
    rope_base: nl.NkiTensor,
    context_proj: nl.NkiTensor,
    context_norm: nl.NkiTensor,
    kv_a_proj: nl.NkiTensor,
    kv_a_norm: nl.NkiTensor,
    rank_info: nl.NkiTensor = None,
    tp_group: ReplicaGroup = None,
    layers: int = 5,
    cp: int = TP,
    queries: int = _AUX_VERIFY,
    guard_indices: bool = False,
):
    """Run only the context stage of ``kimi_k3_draft_mla_step``; return ``rider_pool`` (aliased) with its rows written.

    The context stage reads the verify rows (``aux``, ``verify_slots``, ``row0``, the RoPE window) and not the accept
    (``accepted``, ``bonus``), so a decode graph can run it before the sampler. Then ``kimi_k3_draft_mla_step`` with
    ``context_done=True`` on the returned pool gives the bits of one full step: the same inputs give the same rows.
    The arguments are those of ``kimi_k3_draft_mla_step``; ``guard_indices`` guards only the slots here.
    """
    sb = AUTO_SBUF
    int32 = nl.int32
    bf16 = nl.bfloat16
    hwdge = nisa.dge_mode.hwdge
    sync = nisa.engine.sync
    tokens = row0.shape[0]
    total_v = _AUX_VERIFY * tokens
    page_pool = (
        rider_pool.view(fp8_page_dtype()) if rider_pool.dtype == nl.float8_e4m3fn and not is_trn3() else rider_pool
    )
    layer_pages = page_pool.shape[0] // layers
    pools = [page_pool[layer * layer_pages : (layer + 1) * layer_pages, 0:PAGE_WIDTH] for layer in range(layers)]
    kernel_assert(2 <= queries <= _AUX_VERIFY, f"queries must be 2 to {_AUX_VERIFY}, got {queries}")
    if queries < _AUX_VERIFY:
        aux = _pad_verify_aux(aux, tokens, queries, sb)
    if queries < _AUX_VERIFY or guard_indices:
        slot_limit = layer_pages * _PAGE_BLOCK if guard_indices else None
        verify_slots = _guard_rows(verify_slots, queries * tokens, total_v, slot_limit, "slots", sb)
    fp8_pages = page_pool.dtype != bf16
    h_shard = HIDDEN // cp
    kv_a_shard = CACHE_WIDTH // cp if cp == TP else KV_LORA // cp + 1

    region = ExplicitSbuf()
    install_identities(region)
    heap = SbufTiles(region, heap=True)
    # The two gathers' sources and destinations are region-heap tiles for the whole kernel, as in the step.
    sources = {
        "ctx_src": heap.tile((total_v, h_shard), bf16, "ctx_ag_src"),
        "kva_src": heap.tile((total_v, layers * kv_a_shard), bf16, "kva_ag_src"),
        "ctx_dst": heap.tile((total_v, HIDDEN), bf16, "ctx_ag_dst"),
        "kva_dst": heap.tile((total_v, cp * layers * kv_a_shard), bf16, "kva_ag_dst"),
    }
    for tile in sources.values():
        nisa.memset(dst=tile, value=0.0)

    row0_row = sb.tile((1, tokens), int32)
    base_row = sb.tile((1, tokens), int32)
    nisa.dma_copy(
        dst=row0_row,
        src=row0.ap(pattern=[[tokens, 1], [1, tokens]], offset=0),
        dge_mode=hwdge,
        engine=sync,
        name="row0_row",
    )
    nisa.dma_copy(
        dst=base_row,
        src=rope_base.ap(pattern=[[tokens, 1], [1, tokens]], offset=0),
        dge_mode=hwdge,
        engine=sync,
        name="base_row",
    )
    bw = sb.tile((1, tokens), int32)
    nisa.iota(dst=bw, pattern=[[rope_window.shape[1], tokens]], offset=0)
    bw_minus_base = sb.tile((1, tokens), int32)
    nisa.tensor_tensor(dst=bw_minus_base, data1=bw, data2=base_row, op=nl.subtract)
    _context_stage(
        aux, row0, rope_window, row0_row, bw_minus_base, context_proj, context_norm, kv_a_proj, kv_a_norm, pools,
        verify_slots, rank_info, tp_group, layers, fp8_pages, sb, sources, cp,
    )  # fmt: skip
    return rider_pool
