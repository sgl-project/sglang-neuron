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

"""GLM-5.2 MTP greedy head in one trace: ``shared_head.norm`` -> lm_head (vocabulary shard) -> global argmax.

A draft token is the next draft step's input, so the head sits on a serial path that runs once per draft step. It
keeps the vocabulary sharded: each rank takes the argmax of its own shard, and one AllGather of every rank's ``[T,
2]`` (max, index) pair resolves the global argmax, with no top-k, sampler or full logits.

Ties: a candidate wins when its value equals the global max exactly, and its key is ``vocab_size - index``, so the
largest key is the smallest index among ties, matching ``torch.argmax``. fp32 holds every index exactly.

The lm_head uses column bands (``mla_proj_primitives.linear``) because the stationary operand is only ``T = 32`` rows
wide; each element keeps its fp32 accumulation order, so the logits are bit-identical to an unbanded matmul.

``vocab_base`` is an int32 tensor, not a Python int, because one NEFF serves every rank. ``hidden_states`` must be
replicated across the vocabulary group, which the draft chain's rows layout already gives. The weight shard is
streamed rather than held in SBUF, since each byte is read once either way. LNC=1 only.
"""

from typing import Tuple

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from ..attention.mla.mla_decode_constants import F_MAX, P_MAX
from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import div_ceil
from ..utils.trace_names import unique_name
from .attention.mla.mla_proj_primitives import pe_column_band
from .utils.cc_pin import pin

_ERROR_PREFIX = "[GLM-5.2 MTP Greedy Tail Kernel]"

_NORM_DMA_BLOCK = 6 * P_MAX
"""Column block for the input load and its sum of squares: each block's squares run on the Scalar Engine while
later blocks are still in flight."""

_FIND_INDEX8_MAX = 16384
"""``nisa.nc_find_index8`` accepts between 8 and 16,384 elements per partition, so one call covers
a whole vocab shard only while ``V_s`` stays inside that. At ``VTP = 128`` it is 1210."""

_LOWEST_FP32 = -3.4028234663852886e38
"""Fills the columns past a narrower band's chain: below any finite logit, so neither the max nor the index sees it."""


@nki.jit
def glm52_mtp_greedy_tail(
    hidden_states: nl.NkiTensor,
    gamma: nl.NkiTensor,
    lm_head_weight_t: nl.NkiTensor,
    vocab_base: nl.NkiTensor,
    vocab_replica_group: ReplicaGroup,
    vocab_size: int,
    pairs_dst: nl.NkiTensor,
    eps: float = 1e-5,
    tokens_out: nl.NkiTensor = None,
    tokens_sb_out: nl.NkiTensor = None,
    lm_head_weights_after=None,
    handles_out: dict = None,
    row_index_out: tuple = None,
    store_tokens: bool = True,
) -> Tuple[nl.NkiTensor, ...]:
    """
    Greedy-sample one token per sequence from the MTP layer's hidden state.

    Usage:
        Tuned for: GLM-5.2 MTP layer 78, C32 decode, vocab-parallel bf16 lm_head, LNC=1
        Applicable to: any greedy decode head whose vocab is column-parallel over a replica group

    Dimensions:
        T: Rows needing a token -- one per sequence (C32 = 32 in production)
        H: Hidden dimension (6144)
        V: Global vocab (154880)
        V_s: This rank's vocab shard, ``V / VTP`` (1210 at VTP = 128)

    Args:
        hidden_states (nl.NkiTensor): [T, H] @ HBM or SBUF, fp32/fp16/bf16. The MTP block's output,
            REPLICATED across ``vocab_replica_group`` -- this rank holds a slice of the vocab and
            so needs every row. A 16-bit input is widened through an explicit on-chip copy, which
            is exact and halves this load; feeding 16-bit data straight into the square and gamma
            chains would instead make those engines compute at input precision.
        gamma (nl.NkiTensor): [1, H] @ HBM. ``shared_head.norm`` weight, i.e. the final RMSNorm the
            reference applies before the head. bf16 in the checkpoint; loaded into fp32 SBUF.
        lm_head_weight_t (nl.NkiTensor): [H, V_s] @ HBM, **bf16, ALREADY TRANSPOSED** from the
            checkpoint's [V, H] and C-contiguous. ``nc_matmul`` contracts the partition dim, so H
            must present on partitions; a transposed VIEW gives none of the benefit because a DMA
            descriptor is built from strides. bf16 is the checkpoint's own precision, so fp32 would
            not be more accurate than the model -- it would just double the dominant HBM read.
            GLM-5.2 ties ``shared_head.head`` to ``lm_head``, so one shard serves the target's head
            and the MTP's.
        vocab_base (nl.NkiTensor): [1, 1] int32 @ HBM. This rank's ``rank * V_s``. A TENSOR because
            one NEFF serves all ranks -- see THE PER-RANK VOCAB BASE MUST BE A TENSOR.
        vocab_replica_group (ReplicaGroup): one group listing all ``VTP`` ranks that share the
            vocab. Order does not matter: a max is commutative
            and the global id comes from ``vocab_base``, not from position in the group.
        vocab_size (int): ``V``, the GLOBAL vocab. Rank-invariant, so a Python int is safe.
        eps (float): RMSNorm epsilon. GLM-5.2's ``rms_norm_eps`` is 1e-5.
        tokens_out (nl.NkiTensor): Optional [T, 1] uint32 @ HBM the tokens are written to (a slice of the
            caller's per-step output, say) instead of a fresh shared-HBM tensor.
        tokens_sb_out (nl.NkiTensor): Optional [P_MAX, 1] uint32 SBUF tile that also receives the tokens
            on rows ``[0, T)``, for an on-chip consumer (the draft chain's embedding lookup).
        lm_head_weights_after: Optional instruction (a weight DMA of the caller's) that every lm_head weight
            DMA is ordered after (``depends_on``), so the prefetch does not share the weight queue with it.
        handles_out (dict): Optional; receives the instruction handles a caller can order its own work after to
            keep it off this head's critical path: ``"pair_copy"`` (Vector, the last write of the AllGather's
            source) and ``"all_gather"`` (its trigger, on GpSimd).
        row_index_out (tuple): Optional ``(idx_sb, n_rows)``: [P_MAX, 1] int32 SBUF that receives, on rows ``[0, T)``,
            ``clamp(token - vocab_base + 1, 0, n_rows - 1)`` -- the row of each token in a vocab-sharded table with
            one zero row before and after this rank's ``V_s`` rows (the draft chain's embedding) -- computed from
            the winning key, ahead of the token's own cast.
        store_tokens (bool): False leaves the HBM token store to the caller (``tokens_sb_out`` holds them).
        pairs_dst (nl.NkiTensor): [VTP * T, 2] fp32 private HBM destination of the (max, index) AllGather. The draft
            chain's steps rotate a fixed pair, so each destination's addresses are that collective's alone and the
            runtime skips the per-call handshake.

    Returns:
        Tuple[nl.NkiTensor]: ``(tokens,)``, [T, 1] uint32 GLOBAL vocab ids, identical on every rank of the group
        (``tokens_out`` when given).

    Notes:
        DEFERRED RMS: the per-row scale is applied on the GEMM's PSUM reads
        rather than on its input. Exact algebra -- ``rms`` is a per-row scalar and the GEMM is
        linear -- and it removes the dependency that made every matmul wait on the whole row's
        sum of squares. The bf16 rounding lands on ``x * gamma`` instead of ``x * rms * gamma``,
        which is the same one-rounding-per-element bound because bf16's relative error is
        scale-invariant.

        The index path runs in fp32 throughout and never in the logit dtype. bf16 represents
        integers exactly only to 256; near a 1210-wide shard its spacing already exceeds 1, so a
        true argmax would round to a different -- possibly out-of-range -- token.

    Pseudocode:
        x    = rmsnorm(hidden_states, gamma, eps)
        lg   = x @ lm_head_weight_t                       # [T, V_s], this rank's shard
        m    = max(lg, axis=1);  i = argmax(lg, axis=1) + vocab_base
        gm   = all_reduce_max(m)
        key  = (m == gm) * (vocab_size - i)
        tok  = vocab_size - all_reduce_max(key)
    """

    T, H = hidden_states.shape
    v_s = lm_head_weight_t.shape[1]

    _validate_mtp_greedy_tail(hidden_states, gamma, lm_head_weight_t, vocab_base, vocab_size, T, H, v_s)

    gamma_sb = _load_gamma_broadcast(gamma, H, T)
    row_index = None
    if row_index_out is not None:
        # Off the critical path: (V + 1) - vocab_base, the row of token t being that minus the winning key.
        base_i = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=base_i[0:T, 0:1], src=vocab_base[0:1, 0:1].broadcast(0, T))
        row_base = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=row_base[0:T, 0:1], src=base_i[0:T, 0:1])
        nisa.tensor_scalar(
            dst=row_base[0:T, 0:1],
            data=row_base[0:T, 0:1],
            op0=nl.multiply,
            operand0=-1.0,
            op1=nl.add,
            operand1=float(vocab_size + 1),
        )
        row_index = (row_index_out[0], row_index_out[1], row_base)

    # ═══════ RMSNorm, blocked so the squares overlap the load ═══════
    xt, rms = _norm_stage(hidden_states, gamma_sb, eps, T, H)

    # ═══════ Transposed activations, hoisted: every vocab chunk reuses them ═══════
    xt_chunks = _transpose_activations(xt, gamma_sb, T, H)

    if _banded_argmax_fits(T, v_s):
        # ═══════ lm_head in PE column bands, then the argmax on the bands' partitions ═══════
        # The argmax runs on the PE bands' own partitions instead of moving the bands to rows [0, T) by DMA first:
        # the same logits, the same (max, lowest index) pair.
        pair_sb, pair_copy = _banded_argmax(
            xt_chunks, lm_head_weight_t, rms, vocab_base, vocab_size, T, H, v_s, lm_head_weights_after
        )
        if handles_out is not None:
            handles_out["pair_copy"] = pair_copy
        tokens = _global_argmax(
            None,
            None,
            vocab_replica_group,
            vocab_size // v_s,
            vocab_size,
            T,
            tokens_out,
            tokens_sb_out,
            handles_out,
            pair_sb=pair_sb,
            row_index=row_index,
            store_tokens=store_tokens,
            pairs_dst=pairs_dst,
        )
        return (tokens,)

    # ═══════ lm_head in PE column bands, into an SBUF logit tile ═══════
    logits_sb = _banded_lm_head(xt_chunks, lm_head_weight_t, rms, T, H, v_s, lm_head_weights_after)

    # ═══════ Local argmax over this rank's shard ═══════
    local_max, local_idx = _local_argmax(logits_sb, vocab_base, T, v_s)

    # ═══════ Global argmax: one all_gather of the (max, index) pairs over the vocab group ═══════
    tokens = _global_argmax(
        local_max,
        local_idx,
        vocab_replica_group,
        vocab_size // v_s,
        vocab_size,
        T,
        tokens_out,
        tokens_sb_out,
        handles_out,
        row_index=row_index,
        store_tokens=store_tokens,
        pairs_dst=pairs_dst,
    )

    return (tokens,)


def _vocab_chains(v_s: int, n_bands: int) -> list:
    """Column ranges ``[c0, c1)`` of the shard, one PE-band chain each, a multiple of ``n_bands`` long.

    Each chain fits one PSUM bank row (``F_MAX`` fp32), and the count is rounded up to fill every band
    of the last bank: 4 chains of ~303 columns at ``V_s = 1210``, 20 of ~484 at ``V_s = 9680``.
    """
    n_chains = div_ceil(div_ceil(v_s, F_MAX), n_bands) * n_bands
    width = div_ceil(v_s, n_chains)
    chains = []
    for c in range(n_chains):
        c0 = c * width
        c1 = min(v_s, c0 + width)
        if c0 < c1:
            chains.append((c0, c1))
    return chains


def _banded_lm_head(
    xt_chunks: list,
    lm_head_weight_t: nl.NkiTensor,
    rms: nl.NkiTensor,
    rows: int,
    hidden: int,
    v_s: int,
    weights_after=None,
) -> nl.NkiTensor:
    """``logits[T, V_s] = rms * (x * gamma) @ W``, the GEMM in PE column bands. See PE COLUMN BANDS.

    The chains go in groups of ``n_bands``, one PSUM bank per group: for each 128-row k-tile, ONE DMA
    streams the group's columns of the weight and the group's bands each take their slice. Band ``b``
    accumulates on PSUM partitions ``[b * band, b * band + T)``; eviction copies each band out on those
    partitions and one SBUF->SBUF DMA moves it to rows ``[0, T)`` (the DMA is what crosses partitions).
    The deferred rms is then one multiply over the whole tile.

    Args:
        xt_chunks (list): ``H / 128`` bf16 ``[128, T]`` tiles from ``_transpose_activations``.
        lm_head_weight_t (nl.NkiTensor): [H, V_s] bf16 @ HBM.
        rms (nl.NkiTensor): [P_MAX, 1] fp32 per-row ``rsqrt(mean(x^2) + eps)``.
        rows (int): T.
        hidden (int): H.
        v_s (int): V_s.
        weights_after: Optional instruction every weight DMA is ordered after.

    Returns:
        nl.NkiTensor: [P_MAX, V_s] fp32 SBUF, rows ``[0, T)`` valid.
    """

    band = pe_column_band(rows)
    n_bands = P_MAX // band
    chains = _vocab_chains(v_s, n_bands)
    logits_sb = nl.ndarray((P_MAX, v_s), dtype=nl.float32, buffer=nl.sbuf)
    n_k_tiles = div_ceil(hidden, P_MAX)

    for g0 in range(0, len(chains), n_bands):
        group = chains[g0 : g0 + n_bands]
        lo, hi = group[0][0], group[-1][1]
        acc_ps = nl.ndarray((P_MAX, F_MAX), dtype=nl.float32, buffer=nl.psum)
        for c in range(n_k_tiles):
            h0 = c * P_MAX
            h_len = min(P_MAX, hidden - h0)
            # Streamed, not resident: with a single row tile each byte is read exactly once either way,
            # and streaming keeps SBUF free -- see STREAMED, NOT RESIDENT.
            # sized to the shard's columns (2420 B a partition at V_s = 1210), not a PSUM bank group, so more tiles
            # fit in flight
            w_sb = nl.ndarray((P_MAX, hi - lo), dtype=nl.bfloat16, buffer=nl.sbuf)
            w_dma = nisa.dma_copy(
                dst=w_sb[0:h_len, 0 : hi - lo],
                src=lm_head_weight_t[h0 : h0 + h_len, lo:hi],
                dge_mode=nisa.dge_mode.none,
            )
            if weights_after is not None:
                w_dma.depends_on(weights_after)
            for b, (c0, c1) in enumerate(group):
                nisa.nc_matmul(
                    dst=acc_ps[b * band : b * band + rows, 0 : c1 - c0],
                    stationary=xt_chunks[c][0:h_len, 0:rows],
                    moving=w_sb[0:h_len, c0 - lo : c1 - lo],
                    accumulate=c > 0,
                    tile_position=(0, b * band),
                    tile_size=(P_MAX, band),
                )
        for b, (c0, c1) in enumerate(group):
            band_rows = slice(b * band, b * band + rows)
            if b == 0:
                nisa.tensor_copy(dst=logits_sb[0:rows, c0:c1], src=acc_ps[0:rows, 0 : c1 - c0])
            else:
                band_sb = nl.ndarray((P_MAX, F_MAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=band_sb[band_rows, 0 : c1 - c0], src=acc_ps[band_rows, 0 : c1 - c0])
                nisa.dma_copy(dst=logits_sb[0:rows, c0:c1], src=band_sb[band_rows, 0 : c1 - c0])

    # The deferred per-row rms: one multiply over the whole shard.
    nisa.tensor_scalar(
        dst=logits_sb[0:rows, 0:v_s], data=logits_sb[0:rows, 0:v_s], op0=nl.multiply, operand0=rms[0:rows, 0:1]
    )
    return logits_sb


def _banded_argmax_fits(rows: int, v_s: int) -> bool:
    """Whether :func:`_banded_argmax` applies: the bands fill all 128 partitions, one chain per band."""
    band = pe_column_band(rows)
    n_bands = P_MAX // band
    return rows == band and n_bands > 1 and len(_vocab_chains(v_s, n_bands)) == n_bands


def _banded_argmax(
    xt_chunks: list,
    lm_head_weight_t: nl.NkiTensor,
    rms: nl.NkiTensor,
    vocab_base: nl.NkiTensor,
    vocab_size: int,
    rows: int,
    hidden: int,
    v_s: int,
    weights_after=None,
) -> tuple:
    """``_banded_lm_head`` then ``_local_argmax`` without moving the bands: the (max, index) pair per row.

    ``_banded_lm_head`` evicts each band on the partitions it accumulated on and moves bands 1.. to rows ``[0, T)``
    by SBUF -> SBUF DMA before the rms multiply and the argmax over the whole shard. Here every partition holds one
    band row: the rms multiply reads the PSUM on all 128 partitions at once (``rms`` replicated per band, early),
    each partition takes the max and its first index over its band's columns, and the four band candidates of a
    row meet on one partition through exact fp32 PE transposes, where the global tail's rule picks the winner:
    the max, and the lowest index among equal maxima. The bands are contiguous column ranges in order, so that
    is the first occurrence of the max over the shard -- ``_local_argmax``'s pair. The logits are the same fp32
    products (``acc * rms`` on the Vector engine).

    Returns:
        tuple: ``(pair_sb, pair_copy)``: [T, 2] fp32 SBUF ``(max, global index)`` per row, and its last write.
    """
    band = pe_column_band(rows)
    n_bands = P_MAX // band
    chains = _vocab_chains(v_s, n_bands)
    lo, hi = chains[0][0], chains[-1][1]
    width = max(c1 - c0 for c0, c1 in chains)
    n_k_tiles = div_ceil(hidden, P_MAX)

    # ---- off the critical path: rms and each band's first global index, on every band's partitions ----
    rms_rep = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    for b in range(n_bands):
        nisa.dma_copy(dst=rms_rep[b * band : b * band + rows, 0:1], src=rms[0:rows, 0:1])
    base_i = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=base_i[0:P_MAX, 0:1], src=vocab_base[0:1, 0:1].broadcast(0, P_MAX))
    band_base = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=band_base[0:P_MAX, 0:1], src=base_i[0:P_MAX, 0:1])
    chain0 = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    for b, (c0, _c1) in enumerate(chains):
        nisa.memset(dst=chain0[b * band : (b + 1) * band, 0:1], value=float(c0))
    # Integers below 2**24: exact in fp32 in any order.
    nisa.tensor_tensor(
        dst=band_base[0:P_MAX, 0:1], data1=band_base[0:P_MAX, 0:1], data2=chain0[0:P_MAX, 0:1], op=nl.add
    )

    # ---- the GEMM, as _banded_lm_head ----
    acc_ps = nl.ndarray((P_MAX, F_MAX), dtype=nl.float32, buffer=nl.psum)
    for c in range(n_k_tiles):
        h0 = c * P_MAX
        h_len = min(P_MAX, hidden - h0)
        w_sb = nl.ndarray((P_MAX, hi - lo), dtype=nl.bfloat16, buffer=nl.sbuf)
        w_dma = nisa.dma_copy(
            dst=w_sb[0:h_len, 0 : hi - lo],
            src=lm_head_weight_t[h0 : h0 + h_len, lo:hi],
            dge_mode=nisa.dge_mode.none,
        )
        if weights_after is not None:
            w_dma.depends_on(weights_after)
        for b, (c0, c1) in enumerate(chains):
            nisa.nc_matmul(
                dst=acc_ps[b * band : b * band + rows, 0 : c1 - c0],
                stationary=xt_chunks[c][0:h_len, 0:rows],
                moving=w_sb[0:h_len, c0 - lo : c1 - lo],
                accumulate=c > 0,
                tile_position=(0, b * band),
                tile_size=(P_MAX, band),
            )

    # ---- the deferred rms on the eviction, every band at once (runs of equal-width bands) ----
    logits4 = nl.ndarray((P_MAX, width), dtype=nl.float32, buffer=nl.sbuf)
    b0 = 0
    while b0 < n_bands:
        w = chains[b0][1] - chains[b0][0]
        b1 = b0 + 1
        while b1 < n_bands and chains[b1][1] - chains[b1][0] == w:
            b1 += 1
        p0, p1 = b0 * band, b1 * band
        nisa.tensor_scalar(
            dst=logits4[p0:p1, 0:w],
            data=acc_ps[p0:p1, 0:w],
            op0=nl.multiply,
            operand0=rms_rep[p0:p1, 0:1],
            engine=nisa.engine.vector,
        )
        if w < width:
            nisa.memset(dst=logits4[p0:p1, w:width], value=_LOWEST_FP32)
        b0 = b1

    # ---- each band row's max and the global index of its first occurrence ----
    band_max = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(op=nl.maximum, data=logits4[0:P_MAX, 0:width], dst=band_max[0:P_MAX, 0:1], axis=1)
    idx8 = nl.ndarray((P_MAX, 8), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.nc_find_index8(
        data=logits4[0:P_MAX, 0:width], vals=band_max[0:P_MAX, 0:1].broadcast(1, 8), dst=idx8[0:P_MAX, 0:8]
    )
    band_idx = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=band_idx[0:P_MAX, 0:1], src=idx8[0:P_MAX, 0:1])
    nisa.tensor_tensor(
        dst=band_idx[0:P_MAX, 0:1], data1=band_idx[0:P_MAX, 0:1], data2=band_base[0:P_MAX, 0:1], op=nl.add
    )

    # ---- the band candidates of every row onto one partition: [1, 2 * 128], (value | index) at 32 b + t ----
    cand_ps = nl.ndarray((1, 2 * P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=cand_ps[0:1, 0:P_MAX], data=band_max[0:P_MAX, 0:1], engine=nisa.tensor_engine)
    nisa.nc_transpose(dst=cand_ps[0:1, P_MAX : 2 * P_MAX], data=band_idx[0:P_MAX, 0:1], engine=nisa.tensor_engine)
    cand_sb = nl.ndarray((1, 2 * P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=cand_sb[0:1, 0 : 2 * P_MAX], src=cand_ps[0:1, 0 : 2 * P_MAX])
    # [1, T, n_bands] views: element (t, b) at 32 b + t.
    p_stride = 2 * P_MAX
    values = cand_sb.ap(pattern=[[p_stride, 1], [1, rows], [band, n_bands]], offset=0)
    indices = cand_sb.ap(pattern=[[p_stride, 1], [1, rows], [band, n_bands]], offset=P_MAX)

    # ---- the winner per row: the global tail's rule over n_bands candidates ----
    res = nl.ndarray((1, 2 * rows), dtype=nl.float32, buffer=nl.sbuf)  # (max | index), t on the free axis
    nisa.tensor_reduce(op=nl.maximum, data=values, dst=res[0:1, 0:rows], axis=2)
    won = nl.ndarray((1, rows * n_bands), dtype=nl.float32, buffer=nl.sbuf)
    won_tb = won.ap(pattern=[[rows * n_bands, 1], [n_bands, rows], [1, n_bands]], offset=0)
    nisa.tensor_tensor(
        dst=won_tb,
        data1=values,
        data2=res.ap(pattern=[[2 * rows, 1], [1, rows], [0, n_bands]], offset=0),
        op=nl.equal,
    )
    key = nl.ndarray((1, rows * n_bands), dtype=nl.float32, buffer=nl.sbuf)
    key_tb = key.ap(pattern=[[rows * n_bands, 1], [n_bands, rows], [1, n_bands]], offset=0)
    nisa.tensor_scalar(
        dst=key_tb, data=indices, op0=nl.subtract, operand0=float(vocab_size), op1=nl.multiply, operand1=-1.0
    )
    nisa.tensor_tensor(dst=key_tb, data1=key_tb, data2=won_tb, op=nl.multiply)
    best = nl.ndarray((1, rows), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(op=nl.maximum, data=key_tb, dst=best[0:1, 0:rows], axis=2)
    nisa.tensor_scalar(
        dst=res[0:1, rows : 2 * rows],
        data=best[0:1, 0:rows],
        op0=nl.subtract,
        operand0=float(vocab_size),
        op1=nl.multiply,
        operand1=-1.0,
    )

    # ---- back to one row per partition ----
    pair_ps = nl.ndarray((P_MAX, 2), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=pair_ps[0:rows, 0:1], data=res[0:1, 0:rows], engine=nisa.tensor_engine)
    nisa.nc_transpose(dst=pair_ps[0:rows, 1:2], data=res[0:1, rows : 2 * rows], engine=nisa.tensor_engine)
    pair_sb = nl.ndarray((rows, 2), dtype=nl.float32, buffer=nl.sbuf)
    pair_copy = nisa.tensor_copy(dst=pair_sb[0:rows, 0:2], src=pair_ps[0:rows, 0:2])
    return pair_sb, pair_copy


def _load_gamma_broadcast(gamma_hbm: nl.NkiTensor, hidden: int, rows: int) -> nl.NkiTensor:
    """Loads ``gamma [1, H]`` broadcast to ``[P_MAX, H]`` fp32.

    A PARTITION-dim broadcast of an on-chip tensor is illegal, so it happens at the DMA boundary
    with an HBM-side ``.broadcast`` view. fp32 regardless of the HBM dtype: the DMA casts, and the
    normalization multiply then runs in fp32 whatever the checkpoint stores.
    """

    gamma_sb = nl.ndarray((P_MAX, hidden), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=gamma_sb[0:rows, 0:hidden], src=gamma_hbm[0:1, 0:hidden].broadcast(0, rows))
    return gamma_sb


def _norm_stage(x_hbm: nl.NkiTensor, gamma_sb: nl.NkiTensor, eps: float, rows: int, hidden: int):
    """Loads the hidden state and returns ``(xt fp32 [P_MAX, H], rms fp32 [P_MAX, 1])``.

    The row loads in ``_NORM_DMA_BLOCK`` column blocks and each block's ``sum(x^2)`` accumulates
    into the Scalar Engine's ``reduce_regs`` while later blocks are still in flight. ``rsqrt(ssq / H + eps)`` is ONE activation, because the Scalar Engine applies
    ``op(data * scale + bias)`` natively.

    The hidden size is not padded, so the divisor is plain ``H``.

    Args:
        x_hbm (nl.NkiTensor): [T, H] hidden states, in HBM or SBUF.
        gamma_sb (nl.NkiTensor): unused by the norm itself; kept out of this stage deliberately --
            gamma applies per GEMM chunk in ``_transpose_activations``.
        eps (float): RMSNorm epsilon.
        rows (int): T.
        hidden (int): H.

    Returns:
        tuple: ``(xt, rms)``, both SBUF.
    """
    del gamma_sb  # applied per chunk downstream, not here; named for call-site clarity

    widen = x_hbm.dtype != nl.float32
    on_chip = x_hbm.buffer == nl.sbuf
    xt = nl.ndarray((P_MAX, hidden), dtype=nl.float32, buffer=nl.sbuf)
    ssq = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    sq_scratch = nl.ndarray((P_MAX, _NORM_DMA_BLOCK), dtype=nl.float32, buffer=nl.sbuf)

    n_blk = div_ceil(hidden, _NORM_DMA_BLOCK)
    if on_chip:
        # Already in SBUF (the draft chain's residual add): the widening copy reads it in place.
        x_raw = x_hbm if widen else None
        if not widen:
            nisa.tensor_copy(dst=xt[0:rows, 0:hidden], src=x_hbm[0:rows, 0:hidden])
    else:
        x_raw = nl.ndarray((P_MAX, hidden), dtype=x_hbm.dtype, buffer=nl.sbuf) if widen else None
        for b in range(n_blk):
            b0 = b * _NORM_DMA_BLOCK
            b_len = min(_NORM_DMA_BLOCK, hidden - b0)
            nisa.dma_copy(
                dst=(x_raw if widen else xt)[0:rows, b0 : b0 + b_len],
                src=x_hbm[0:rows, b0 : b0 + b_len],
                dge_mode=nisa.dge_mode.none,
            )
    if widen:
        # An explicit conversion COPY is what makes a 16-bit input bitwise-equivalent to a
        # pre-widened fp32 one: fp16/bf16 are subsets of fp32, whereas feeding 16-bit data
        # straight into the chains below makes the engines compute at the input's precision.
        for b in range(n_blk):
            b0 = b * _NORM_DMA_BLOCK
            b_len = min(_NORM_DMA_BLOCK, hidden - b0)
            nisa.tensor_copy(dst=xt[0:rows, b0 : b0 + b_len], src=x_raw[0:rows, b0 : b0 + b_len])

    for b in range(n_blk):
        b0 = b * _NORM_DMA_BLOCK
        b_len = min(_NORM_DMA_BLOCK, hidden - b0)
        nisa.activation(
            dst=sq_scratch[0:rows, 0:b_len],
            op=nl.square,
            data=xt[0:rows, b0 : b0 + b_len],
            reduce_op=nl.add,
            reduce_res=ssq[0:rows, 0:1] if b == n_blk - 1 else None,
            reduce_cmd=nisa.reduce_cmd.reset_reduce if b == 0 else nisa.reduce_cmd.reduce,
        )

    rms = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=rms[0:rows, 0:1], op=nl.rsqrt, data=ssq[0:rows, 0:1], scale=1.0 / float(hidden), bias=eps)
    return xt, rms


def _transpose_activations(xt: nl.NkiTensor, gamma_sb: nl.NkiTensor, rows: int, hidden: int) -> list:
    """Scales by gamma and transposes, one 128-column chunk at a time.

    Returns ``H / P_MAX`` bf16 tiles, chunk ``c`` holding ``[128, T]`` with H on PARTITIONS --
    which is what ``nc_matmul`` needs of a stationary operand. Hoisted out of the vocab loop
    because every vocab chunk contracts against all of them: recomputing per chunk would repeat
    ``H / 128`` transposes ``V_s / F_MAX`` times. At the production shape the whole set is 12 KB
    per partition.

    RMS is deliberately NOT in this chain -- see DEFERRED RMS in the entry's Notes.
    """

    xt_chunks = []
    for c in range(div_ceil(hidden, P_MAX)):
        h0 = c * P_MAX
        h_len = min(P_MAX, hidden - h0)

        scaled = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=scaled[0:rows, 0:h_len],
            data1=xt[0:rows, h0 : h0 + h_len],
            data2=gamma_sb[0:rows, h0 : h0 + h_len],
            op=nl.multiply,
        )
        t_ps = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=t_ps[0:h_len, 0:rows], data=scaled[0:rows, 0:h_len], engine=nisa.tensor_engine)
        t_sb = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=t_sb[0:h_len, 0:rows], src=t_ps[0:h_len, 0:rows])
        xt_chunks.append(t_sb)
    return xt_chunks


def _local_argmax(logits_sb: nl.NkiTensor, vocab_base: nl.NkiTensor, rows: int, v_s: int):
    """This rank's per-row ``(max value, GLOBAL index)`` over its vocab shard.

    ``nc_find_index8`` returns the index of the FIRST occurrence of each probed value, so probing
    with the row max yields the lowest local index on a tie -- ``torch.argmax``'s rule. Only slot
    0 is read; the instruction's other seven are unused here.

    The index is cast to fp32 by ``tensor_copy`` (a true numeric cast, unlike a dma_copy across
    dtypes, which bit-reinterprets) and stays fp32 through both reduces. fp32 holds every index
    below 2**24 exactly.

    Args:
        logits_sb (nl.NkiTensor): [P_MAX, V_s] fp32 logits for this rank's shard.
        vocab_base (nl.NkiTensor): [1, 1] int32 ``rank * V_s``.
        rows (int): T.
        v_s (int): V_s.

    Returns:
        tuple: ``(local_max [P_MAX, 1] fp32, local_idx [P_MAX, 1] fp32 global ids)``.
    """

    local_max = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(op=nl.maximum, data=logits_sb[0:rows, 0:v_s], dst=local_max[0:rows, 0:1], axis=1)

    idx8 = nl.ndarray((P_MAX, 8), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.nc_find_index8(
        data=logits_sb[0:rows, 0:v_s], vals=local_max[0:rows, 0:1].broadcast(1, 8), dst=idx8[0:rows, 0:8]
    )

    local_idx = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=local_idx[0:rows, 0:1], src=idx8[0:rows, 0:1])

    # Partition-broadcast the rank's base at the DMA boundary, as with gamma.
    base_i = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=base_i[0:rows, 0:1], src=vocab_base[0:1, 0:1].broadcast(0, rows))
    base_f = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=base_f[0:rows, 0:1], src=base_i[0:rows, 0:1])

    nisa.tensor_tensor(dst=local_idx[0:rows, 0:1], data1=local_idx[0:rows, 0:1], data2=base_f[0:rows, 0:1], op=nl.add)
    return local_max, local_idx


def _global_argmax(
    local_max: nl.NkiTensor,
    local_idx: nl.NkiTensor,
    group: ReplicaGroup,
    vtp: int,
    vocab_size: int,
    rows: int,
    tokens_out: nl.NkiTensor = None,
    tokens_sb_out: nl.NkiTensor = None,
    handles_out: dict = None,
    pair_sb: nl.NkiTensor = None,
    row_index: tuple = None,
    store_tokens: bool = True,
    *,
    pairs_dst: nl.NkiTensor,
) -> nl.NkiTensor:
    """Resolves per-rank ``(max, index)`` to one global token id per row. See ONE GATHER.

    The gather runs through HBM (:func:`_all_gather_pairs_hbm`), which leaves the ``[T, VTP]`` candidate layout --
    row ``t`` on partition ``t``, the ``VTP`` candidates on the free axis -- so the resolution is free-axis reduces
    only.

    Args:
        local_max (nl.NkiTensor): [P_MAX, 1] fp32 this rank's max logit per row.
        local_idx (nl.NkiTensor): [P_MAX, 1] fp32 the global vocab id of that max.
        group (ReplicaGroup): all VTP ranks sharing the vocab.
        vtp (int): Ranks in ``group``.
        vocab_size (int): V, used as the key's complement base.
        rows (int): T.
        tokens_out (nl.NkiTensor): Optional [T, 1] uint32 HBM destination; a fresh shared-HBM tensor if None.
        tokens_sb_out (nl.NkiTensor): Optional [P_MAX, 1] uint32 SBUF tile that also receives the tokens.
        handles_out (dict): Optional; receives ``"pair_copy"`` and ``"all_gather"`` (see the entry).
        pair_sb (nl.NkiTensor): Optional [T, 2] fp32 SBUF ``(max, index)`` pairs already built (``_banded_argmax``),
            in place of ``local_max`` and ``local_idx``.
        row_index (tuple): Optional ``(idx_sb, n_rows, row_base)``: see the entry's ``row_index_out``; ``row_base``
            [P_MAX, 1] fp32 holds ``vocab_size + 1 - vocab_base``.
        store_tokens (bool): False skips the HBM token store.
        pairs_dst (nl.NkiTensor): The AllGather's destination (see the entry).

    Returns:
        nl.NkiTensor: [T, 1] uint32 in HBM, identical on every rank.
    """

    # ---- the one collective: every rank's (max, index) pairs, through HBM ----
    pair_copy = None
    if pair_sb is None:
        pair_sb = nl.ndarray((rows, 2), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=pair_sb[0:rows, 0:1], src=local_max[0:rows, 0:1])
        pair_copy = nisa.tensor_copy(dst=pair_sb[0:rows, 1:2], src=local_idx[0:rows, 0:1])
    values, indices = _all_gather_pairs_hbm(pair_sb, group, vtp, rows, pairs_dst, handles_out)
    if handles_out is not None and pair_copy is not None:
        handles_out["pair_copy"] = pair_copy

    # ---- 1: the global max VALUE of each row ----
    global_max = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(op=nl.maximum, data=values, dst=global_max[0:rows, 0:1], axis=1)

    # ---- 2: the winners' complement keys; the largest is the SMALLEST id among ties ----
    # Exact equality is right: a max returns one of its inputs bitwise, so the winning rank's own value
    # matches it exactly and a tolerance would only create ties.
    won = nl.ndarray((P_MAX, vtp), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=won[0:rows, 0:vtp], data=values, op0=nl.equal, operand0=global_max[0:rows, 0:1])
    key = nl.ndarray((P_MAX, vtp), dtype=nl.float32, buffer=nl.sbuf)
    # vocab_size - idx, which is >= 1 for any valid index, so a winner always outranks a loser's 0.
    nisa.tensor_scalar(
        dst=key[0:rows, 0:vtp],
        data=indices,
        op0=nl.subtract,
        operand0=float(vocab_size),
        op1=nl.multiply,
        operand1=-1.0,
    )
    nisa.tensor_tensor(dst=key[0:rows, 0:vtp], data1=key[0:rows, 0:vtp], data2=won[0:rows, 0:vtp], op=nl.multiply)
    best_key = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(op=nl.maximum, data=key[0:rows, 0:vtp], dst=best_key[0:rows, 0:1], axis=1)

    if row_index is not None:
        # token - base + 1 = (V + 1 - base) - key, clamped onto the table's zero rows: integers below 2**24, exact.
        idx_sb, n_rows, row_base = row_index
        row_f = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=row_f[0:rows, 0:1],
            data=best_key[0:rows, 0:1],
            op0=nl.multiply,
            operand0=-1.0,
            op1=nl.add,
            operand1=row_base[0:rows, 0:1],
        )
        nisa.tensor_scalar(
            dst=row_f[0:rows, 0:1],
            data=row_f[0:rows, 0:1],
            op0=nl.maximum,
            operand0=0.0,
            op1=nl.minimum,
            operand1=float(n_rows - 1),
        )
        nisa.tensor_copy(dst=idx_sb[0:rows, 0:1], src=row_f[0:rows, 0:1])

    token_f = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=token_f[0:rows, 0:1],
        data=best_key[0:rows, 0:1],
        op0=nl.subtract,
        operand0=float(vocab_size),
        op1=nl.multiply,
        operand1=-1.0,
    )

    # tensor_copy value-casts; a dma_copy across dtypes would bit-reinterpret the float.
    token_u = tokens_sb_out if tokens_sb_out is not None else nl.ndarray((P_MAX, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=token_u[0:rows, 0:1], src=token_f[0:rows, 0:1])

    tokens = tokens_out if tokens_out is not None else nl.ndarray((rows, 1), dtype=nl.uint32, buffer=nl.shared_hbm)
    if store_tokens:
        nisa.dma_copy(dst=tokens[0:rows, 0:1], src=token_u[0:rows, 0:1])
    return tokens


def _all_gather_pairs_hbm(
    pair_sb: nl.NkiTensor, group: ReplicaGroup, vtp: int, rows: int, dst_hbm: nl.NkiTensor, handles_out: dict
) -> tuple:
    """The (max, index) AllGather through HBM: ``(values, indices)``, each [T, VTP].

    Rank ``r``'s pairs land at rows ``[T r, T (r + 1))`` of the gathered tensor, so one load puts rank ``r`` on
    partition ``r`` with its ``2 T`` elements contiguous, and one exact fp32 PE transpose per field moves them to
    the ``[T, VTP]`` layout: row ``t`` on partition ``t``, rank ``r`` at column ``r``. Over 128 ranks this is
    faster than an SBUF free-axis gather.
    """
    kernel_assert(vtp <= P_MAX, f"{_ERROR_PREFIX} the HBM pair gather needs VTP <= {P_MAX}, got {vtp}")
    src_hbm = nl.ndarray((rows, 2), dtype=nl.float32, buffer=nl.private_hbm, name=unique_name("mtp_tail_pair_src"))
    # on the Sync HWDGE queue, which is faster here than the compiler's pick (ACT)
    nisa.dma_copy(
        dst=src_hbm[0:rows, 0:2], src=pair_sb[0:rows, 0:2], dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync
    )
    pin(src_hbm), pin(dst_hbm)  # long context only; see utils/cc_pin.py
    cc = ncc.all_gather(srcs=[src_hbm], dsts=[dst_hbm], replica_group=group, collective_dim=0)
    if handles_out is not None:
        handles_out["all_gather"] = cc
    by_rank = nl.ndarray((P_MAX, 2 * rows), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=by_rank[0:vtp, 0 : 2 * rows], src=dst_hbm.reshape((vtp, 2 * rows))[0:vtp, 0 : 2 * rows])
    fields = []
    for f in range(2):
        field_ps = nl.ndarray((P_MAX, vtp), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(
            dst=field_ps[0:rows, 0:vtp],
            data=by_rank.ap(pattern=[[2 * rows, vtp], [2, rows]], offset=f),
            engine=nisa.tensor_engine,
        )
        fields.append(field_ps[0:rows, 0:vtp])
    return fields[0], fields[1]


def _validate_mtp_greedy_tail(
    hidden_states: nl.NkiTensor,
    gamma: nl.NkiTensor,
    lm_head_weight_t: nl.NkiTensor,
    vocab_base: nl.NkiTensor,
    vocab_size: int,
    rows: int,
    hidden: int,
    v_s: int,
) -> None:
    """
    Rejects the launches and shapes this single-configuration kernel does not handle.

    Args:
        hidden_states (nl.NkiTensor): [T, H] replicated hidden states.
        gamma (nl.NkiTensor): [1, H] final-norm weight.
        lm_head_weight_t (nl.NkiTensor): [H, V_s] transposed bf16 vocab shard.
        vocab_base (nl.NkiTensor): [1, 1] int32 rank vocab offset.
        vocab_size (int): global vocab V.
        rows (int): T.
        hidden (int): H.
        v_s (int): V_s.

    Returns:
        None
    """

    # Single NeuronCore: the stages carry no core_barrier, so two programs would race on the same
    # shared_hbm collective buffers.
    kernel_assert(
        nl.program_ndim() == 0 or nl.program_ndim() == 1,
        f"{_ERROR_PREFIX} only no specialization or specialization along one axis is supported, "
        f"got nl.program_ndim()={nl.program_ndim()}",
    )
    n_programs = nl.num_programs(axes=0) if nl.program_ndim() != 0 else 1
    kernel_assert(n_programs == 1, f"{_ERROR_PREFIX} runs on a single NeuronCore (LNC=1), got {n_programs} programs")

    kernel_assert(
        nisa.get_nc_version() >= nisa.nc_version.gen4,
        f"{_ERROR_PREFIX} only supports gen4+ (Trn3+) but got {nisa.get_nc_version()=}",
    )

    kernel_assert(len(hidden_states.shape) == 2, f"{_ERROR_PREFIX} hidden_states must be [T, H]")
    kernel_assert(
        hidden_states.dtype in (nl.float32, nl.float16, nl.bfloat16),
        f"{_ERROR_PREFIX} hidden_states must be fp32/fp16/bf16 (16-bit widens exactly), got {hidden_states.dtype}",
    )
    kernel_assert(
        rows <= P_MAX,
        f"{_ERROR_PREFIX} T must be at most {P_MAX}, one row per sequence on one PE tile, got T={rows}",
    )
    kernel_assert(
        hidden % P_MAX == 0,
        f"{_ERROR_PREFIX} H must be a multiple of {P_MAX} so the contraction tiles evenly, got H={hidden}",
    )
    kernel_assert(
        gamma.shape == (1, hidden),
        f"{_ERROR_PREFIX} gamma must be (1, {hidden}), got {gamma.shape}",
    )
    kernel_assert(
        len(lm_head_weight_t.shape) == 2 and lm_head_weight_t.shape[0] == hidden,
        f"{_ERROR_PREFIX} lm_head_weight_t must be [H={hidden}, V_s] -- the checkpoint's [V, H] "
        f"TRANSPOSED -- got {lm_head_weight_t.shape}",
    )
    kernel_assert(
        lm_head_weight_t.dtype == nl.bfloat16,
        f"{_ERROR_PREFIX} lm_head_weight_t must be bf16, the checkpoint's own precision, got {lm_head_weight_t.dtype}",
    )
    kernel_assert(
        8 <= v_s <= _FIND_INDEX8_MAX,
        f"{_ERROR_PREFIX} V_s={v_s} must be in [8, {_FIND_INDEX8_MAX}] for one nc_find_index8 pass; "
        f"a smaller VTP needs the argmax chunked over the shard",
    )
    kernel_assert(
        vocab_size % v_s == 0,
        f"{_ERROR_PREFIX} vocab_size={vocab_size} must be a whole number of {v_s}-wide shards",
    )
    kernel_assert(
        vocab_base.shape == (1, 1) and vocab_base.dtype == nl.int32,
        f"{_ERROR_PREFIX} vocab_base must be a (1, 1) int32 TENSOR -- a Python int would fold to "
        f"rank 0's value on every rank -- got shape {vocab_base.shape} dtype {vocab_base.dtype}",
    )
    kernel_assert(
        vocab_size < 2**24,
        f"{_ERROR_PREFIX} vocab_size={vocab_size} must stay below 2**24 so fp32 carries every index exactly",
    )
