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
"""AttnRes --- Kimi-K3's cross-layer residual mixture, as a NKI kernel.

Twice in every decoder layer, the plain residual add is replaced by a softmax mixture
over the current residual stream plus ``C`` snapshots of it banked at earlier layers.
The kernel computes the mixture with a fixed, small instruction count per token tile.

The operator, per token ``t``, over candidates ``v_c = bank[t, c]`` (``c < C``) and
``v_C = stream[t]``::

    ss_c  = sum_h v_c[h] ** 2
    dot_c = sum_h v_c[h] * score_w[h]
    s_c   = dot_c * rsqrt(ss_c / H + eps)      # == rmsnorm(v_c) . score_w
    p     = softmax_c(s)                        # over the C+1 candidates
    out   = sum_c p_c * v_c                     # mixes the RAW candidates

Two facts drive the layout and the low instruction count:

1. ``rsqrt(ss_c / H + eps)`` is one scalar per (token, candidate), so it factors out of
   the dot product: only two free-axis reductions per candidate are needed
   (``sum v^2`` and ``sum v * score_w``), never a normalised copy of ``v``.
2. The mixture is over the RAW candidates, so the ``v_c`` loaded for scoring are the
   same tensors reused for the weighted sum --- there is no second, normalised tensor
   in play.

Layout: tokens on the partition axis (tiles of up to 128), ``H`` on the free axis.
``C`` and ``H`` never sit on partitions. Each candidate's ``H`` row is loaded from HBM
once and reduced by one ScalarE instruction (``sum v^2``) and one VectorE instruction
(``sum v * score_w``), which do not depend on each other; one ``rsqrt`` per tile then
gives the ``[tok, C+1]`` scores and their softmax. The mixture runs on TensorE as
``sum_c diag(exp_c) @ v_c``, accumulated in PSUM over 1536-column chunks of the resident
candidates, and the PSUM evictions divide by ``sum_c exp_c``. The resident candidates
rotate through a ring of SBUF slots a few candidates larger than one tile, and the
token tiles are software-pipelined in program order: tile ``i``'s scoring runs on
ScalarE/VectorE while tile ``i-1``'s mixture runs on TensorE, with its PSUM evictions
interleaved between tile ``i``'s scoring passes. One to three candidates per tile (by
bank width) are scored on TensorE instead, from a DMA-transposed copy: one
accumulation over 128-row blocks of H gives their Gram matrix, whose diagonal is
``sum v^2``, and their dot with ``score_w`` (split as bf16 hi + lo). The ring takes
the SBUF that the other buffers leave free, counted in the input dtype. When the
``C+1`` rows do not fit (at ``H = 7168`` on trn3: ``C > 11`` in bf16, ``C > 4`` in
fp32), the mixture re-loads them chunk by chunk instead, without pipelining.

Given bank scores: within a prefill stage the bank only grows, and its ~23 calls
re-score the same snapshot with their own ``score_w``. The caller can score the bank
entries once per stage, for all of them, and pass the result as ``bank_scores``. The
kernel then scores only the stream and streams the bank entries through the mixture,
without keeping a tile's candidates resident: every entry is read once, from a ring of
column chunks whose loads run a whole ring ahead of TensorE. That takes the kernel to
its streaming floor of ``(C + 2) * T * H`` input and output bytes (at ``C = 8``,
``T = 4096``: 589 MB, DMA-bound), and it needs neither the
DMA-transposed copies nor the candidate ring of the path below.

Pad rows: given the optional ``valid_len``, the kernel runs the pipeline over just the
row groups that hold a real row or the first pad row, and copies the output of that pad
row to the other rows. Each such prefix of the segment is its own copy of the pipeline
in a 0- or 1-trip dynamic loop, so no pipeline is cut at a group boundary.

LNC: a 1-D SPMD grid shards the token axis across the programs (the grid size must
divide ``T``), as in the MoE kernels. Works with 1 program too. Every
candidate is independent per token, so the shards never communicate.
"""

import nki
import nki.isa as nisa
import nki.language as nl

from ..utils.allocator import sizeinbytes
from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import div_ceil, get_verified_program_sharding_info, is_trn3_b1

# Partition (token) tile size and the free-axis chunk contract.
_PMAX = 128
_HCHUNK = 512
# H columns per TensorE mixture PSUM accumulator (three fp32 PSUM banks, leaving two
# for the TensorE scores), and per mixture matmul (one bank).
_MIX_CHUNK = 1536
_PSUM_BANK = 512
# Score of a softmax padding column: its exp is exactly 0.
_NEG_BIG = -1.0e30
# SBUF bytes per partition that the kernel leaves free for the compiler.
_SBUF_RESERVE = 8 * 1024
# Candidates per token tile scored on TensorE instead of ScalarE/VectorE. Two balance
# TensorE (mixture plus these scores) against ScalarE/VectorE, and their Gram
# accumulators fill the two PSUM banks the mixture leaves free.
_PE_SCORED = 2
# Per-block width of a transposed candidate: 128 tokens, score_w hi and lo, padded so
# the DMA transpose writes 32-byte aligned blocks.
_T_BLOCK = 144
# Practical bank width is 1..8; the kernel supports up to this many snapshots.
_MAX_C = 15
# Rows of the first two tile groups of the valid-length path; each later group is
# as long as all groups before it (1024, 1024, 2048, ... rows). The pipeline runs over
# the groups up to the one that holds the first pad row.
_VALID_GROUP_ROWS = 1024
# Bank widths at which a valid-length variant transposes its second TensorE candidate
# from the loaded copy instead of HBM (measured on trn3).
_LOOP_SLOT_T = (4,)
# DMA options of a transfer put on the Sync hardware DGE, off the loads' queue.
_SIDE = {"dge_mode": nisa.dge_mode.hwdge, "engine": nisa.engine.sync}
# Stream slots of the path with given bank scores: a tile's stream is loaded and
# scored while the two tiles before it are mixed.
_STREAM_SLOTS = 3
# Bytes per partition of one bank load of that path, at least: a ring entry spans
# as many mixture chunks as this allows.
_ENTRY_BYTES = 24 * 1024
# Bank widths at which the path with given bank scores puts everything but the bank
# loads (the streams, the scores and the output stores) on the Sync hardware DGE queue.
# Measured on trn3 at T=4096: faster at C=1. From C=2 on the bank loads need
# the whole DMA bandwidth, and the split costs more than the parallelism gains
# (slower at C=2 and C=3).
_SCORED_SIDE_C = (1,)
# Bank widths at which that path evicts a tile's mixture into one [tok, H] buffer and
# stores it in one DMA (14 KB per partition) instead of one DMA per mixture chunk (2-3
# KB). Measured on trn3 at T=4096: faster at every C but 2, where the chunked
# stores interleave better with the bank loads.
_WIDE_OUT_C = (1, 3, 4, 5, 6, 7, 8)


def _pe_scored(C):
    """TensorE-scored candidates per token tile at bank width ``C``, measured on trn3.

    The count moves the point where the TensorE-scoring chain (Gram, the next
    candidate's transpose into the shared buffer, its Gram) and the slot reloads
    meet. At C=1 and C=2 one TensorE candidate is faster: a TensorE candidate is
    loaded twice, natural and transposed, and there DMA is the bottleneck. At C=6
    one is faster too: with two, the second one's chain is on the critical path. At
    C=7 three are faster than two (at T=4096); everywhere else two
    are best."""
    return {1: 1, 2: 1, 6: 1, 7: 3}.get(C, _PE_SCORED)


def _ring_fixed_bytes(H, Cp1, dsize):
    """SBUF bytes per partition of the pipelined path, without the candidate slots.

    The ring of slots gets the rest of the budget. If the kernel asks for more
    SBUF than the target has, the compiler spills the buffers to HBM, and the
    results become wrong. Count every buffer in its real dtype: a fp32 call needs
    twice the bytes of a bf16 call."""
    n_hb = H // _PMAX
    # ss, dot, r, scores, expv (and its bf16 copy); row_max etc.
    scalars = (5 * (Cp1 + 1) + 4) * 4 + ((Cp1 + 1) * dsize if dsize < 4 else 0)
    return (
        H * 4  # sw (fp32)
        + 2 * 4  # sq and prod
        + 2 * _PMAX * dsize  # ones and ident
        + (_PMAX if dsize == 4 else 0)  # ident_sel (uint8, fp32 calls only)
        + 2 * _MIX_CHUNK * dsize  # out_ring
        + n_hb * _T_BLOCK * dsize  # v_t
        + n_hb * 4  # sw_t (fp32)
        + _PMAX  # diag_junk (fp8)
        # The pipeline keeps the mixture weights and the scalars of two tiles.
        + 2 * (Cp1 * _PMAX * dsize + scalars)
    )


def _scored_fixed_bytes(H, Cp1, dsize, n_tiles):
    """SBUF bytes per partition of the path with given bank scores, without its ring
    of bank chunks. Counted in the real dtypes, as in ``_ring_fixed_bytes``."""
    # scores, expv (and its bf16 copy); ss, dot, r, row_max etc.
    scalars = (2 * (Cp1 + 1) + 8) * 4 + ((Cp1 + 1) * dsize if dsize < 4 else 0)
    return (
        H * 4  # sw (fp32)
        + n_tiles * (Cp1 - 1) * 4  # the bank scores of every tile
        + 2  # sq and prod
        + 2 * _PMAX * dsize  # ones and ident
        + (_PMAX if dsize == 4 else 0)  # ident_sel (uint8, fp32 calls only)
        + 2 * _MIX_CHUNK * dsize  # out_ring
        + _STREAM_SLOTS * H * dsize  # the stream slots
        # The mixture weights and the scalars of three tiles are live.
        + 3 * (Cp1 * _PMAX * dsize + scalars)
    )


def _dge(in_loop):
    """DMA options: a DMA in a dynamic loop cannot use static descriptors."""
    return {"dge_mode": nisa.dge_mode.swdge} if in_loop else {}


def _load_candidate(dst, bank, stream, c, t_off, tok, h0, hw, in_loop=False):
    """DMA columns ``[h0, h0 + hw)`` of candidate ``c`` for tokens ``[t_off, t_off + tok)``."""
    T, C, H = bank.shape
    if c < C:
        nisa.dma_copy(
            dst=dst[0:tok, 0:hw],
            src=bank.ap(pattern=[[C * H, tok], [1, hw]], offset=t_off * C * H + c * H + h0),
            **_dge(in_loop),
        )
    else:
        nisa.dma_copy(
            dst=dst[0:tok, 0:hw],
            src=stream.ap(pattern=[[H, tok], [1, hw]], offset=t_off * H + h0),
            **_dge(in_loop),
        )


def _after(ins, producers):
    """Order ``ins`` after ``producers`` with a hardware wait. Some neuronx-cc builds
    drop the dependency of a read on an earlier write by another engine, e.g.
    when a later instruction writes another part of the same tensor; then nothing
    orders the pair."""
    if producers:
        ins.depends_on(*producers, kind=nisa.dependency.runtime)
    return ins


def _score_candidate(v_c, c, tok, H, sw, sq, prod, ss, dot):
    """Column ``c`` of ``ss`` gets ``sum_h v_c ** 2`` (ScalarE) and of ``dot`` gets
    ``sum_h v_c * score_w`` (VectorE). The two passes read the same tile and do not
    depend on each other; their elementwise results go to discarded tiles."""
    nisa.activation_reduce(
        dst=sq.ap([[1, tok], [0, H]]),
        op=nl.square,
        data=v_c[0:tok, 0:H],
        reduce_op=nl.add,
        reduce_res=ss[0:tok, nl.ds(c, 1)],
    )
    nisa.scalar_tensor_tensor(
        dst=prod.ap([[1, tok], [0, H]]),
        data=v_c[0:tok, 0:H],
        op0=nl.multiply,
        operand0=1.0,
        op1=nl.multiply,
        operand1=sw[0:tok, 0:H],
        reduce_op=nl.add,
        reduce_res=dot[0:tok, nl.ds(c, 1)],
    )


def _scores_width(Cp1, vector_exp):
    """Columns of a score tile: the fused exp-sum of trn3 needs an even count."""
    return Cp1 + Cp1 % 2 if vector_exp else Cp1


def _mixture_weights(ss, dot, Cp1, tok, inv_h, eps, vector_exp, ident_sel, dtype, split_weights=False):
    """Softmax of ``dot * rsqrt(ss / H + eps)`` over the C+1 candidates, returned as
    the diag(exp_c) stationaries of the TensorE mixture and the ``1 / sum_c exp_c``
    that the PSUM evictions apply."""
    Cp1_even = _scores_width(Cp1, vector_exp)
    # One rsqrt and one multiply per tile for all C+1 candidates.
    r = nl.ndarray((_PMAX, Cp1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=r[0:tok, 0:Cp1], data=ss[0:tok, 0:Cp1], op=nl.rsqrt, scale=inv_h, bias=eps)
    scores = nl.ndarray((_PMAX, Cp1_even), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=scores[0:tok, 0:Cp1], data1=dot[0:tok, 0:Cp1], data2=r[0:tok, 0:Cp1], op=nl.multiply)
    return _softmax_weights(scores, Cp1, tok, vector_exp, ident_sel, dtype, split_weights)


def _softmax_weights(scores, Cp1, tok, vector_exp, ident_sel, dtype, split_weights=False):
    """Softmax of the ``[tok, C+1]`` scores, as the diag(exp_c) stationaries and the
    ``1 / sum_c exp_c`` of ``_mixture_weights``. ``scores`` has
    ``_scores_width(Cp1, vector_exp)`` columns.

    With ``split_weights`` (a 2-byte dtype) each candidate's stationary is the pair
    ``(diag(hi_c), diag(lo_c))``, ``hi_c = dtype(exp_c)`` and ``lo_c = dtype(exp_c - hi_c)``:
    their sum keeps about 16 significant bits of the fp32 weight."""
    Cp1_even = _scores_width(Cp1, vector_exp)
    row_max = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=row_max[0:tok, 0:1], data=scores[0:tok, 0:Cp1], op=nl.maximum, axis=(1,))
    expv = nl.ndarray((_PMAX, Cp1_even), dtype=nl.float32, buffer=nl.sbuf)
    denom = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    if vector_exp:
        # exp(s - max) and its row sum in one VectorE instruction. With the exp off
        # ScalarE, ScalarE keeps one activation table (square and rsqrt) for the whole
        # kernel. The fused sum needs an even element count, so a padding score column
        # holds _NEG_BIG, whose exp is 0.
        if Cp1_even > Cp1:
            nisa.memset(dst=scores[0:tok, Cp1:Cp1_even], value=_NEG_BIG)
        nisa.exponential(
            dst=expv[0:tok, 0:Cp1_even],
            src=scores[0:tok, 0:Cp1_even],
            max_value=row_max[0:tok, 0:1],
            reduce_res=denom[0:tok, 0:1],
            reduce_cmd=nisa.reduce_cmd.reset_reduce,
        )
    else:
        neg_max = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=neg_max[0:tok, 0:1], data=row_max[0:tok, 0:1], op0=nl.multiply, operand0=-1.0)
        nisa.activation(dst=expv[0:tok, 0:Cp1], data=scores[0:tok, 0:Cp1], op=nl.exp, bias=neg_max[0:tok, 0:1])
        nisa.tensor_reduce(dst=denom[0:tok, 0:1], data=expv[0:tok, 0:Cp1], op=nl.add, axis=(1,))
    inv_denom = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=inv_denom[0:tok, 0:1], data=denom[0:tok, 0:1])

    # diag(exp_c): the stationary that scales row t of v_c by exp_c[t] in the TensorE
    # mixture. The unnormalized weight of the top candidate is exactly 1, so a bf16
    # stationary rounds only the smaller weights; a normalized p_c in bf16 would round
    # every weight, the largest one included. The diagonal holds exp_c[t] and every
    # other element an exact 0: a multiply of the identity would make a non-finite
    # exp_c[t] fill row t (0 * NaN), and the matmul contracts over the rows, so one
    # token would reach every token of the tile. In bf16 the diagonal is a bitwise AND
    # of exp_c's bits with ident_sel, all ones on the diagonal and 0 elsewhere; in fp32
    # it is a select (predicate ident_sel). Either takes exp_c in the stationary's
    # dtype: rounded once, as the multiply's output was.
    if sizeinbytes(dtype) == 4:
        expv_lp = expv
    else:
        expv_lp = nl.ndarray((_PMAX, Cp1_even), dtype=dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=expv_lp[0:tok, 0:Cp1], src=expv[0:tok, 0:Cp1], engine=nisa.engine.vector)
    split = split_weights and sizeinbytes(dtype) == 2
    if split:
        # lo = dtype(exp - hi): the fp32 difference is exact (hi is exp rounded to nearest), so hi + lo is exp
        # to about 2^-17 relative. Both halves become diagonals the same way.
        expv_lo = nl.ndarray((_PMAX, Cp1_even), dtype=dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=expv_lo[0:tok, 0:Cp1],
            data1=expv[0:tok, 0:Cp1],
            data2=expv_lp[0:tok, 0:Cp1],
            op=nl.subtract,
            engine=nisa.engine.vector,
        )

    def diag_of(w, c):
        diag_c = nl.ndarray((_PMAX, _PMAX), dtype=dtype, buffer=nl.sbuf)
        if sizeinbytes(dtype) == 2:
            nisa.tensor_scalar(
                dst=diag_c.view(nl.uint16)[0:tok, 0:tok],
                data=ident_sel[0:tok, 0:tok],
                op0=nl.bitwise_and,
                operand0=w.view(nl.uint16)[0:tok, c : c + 1],
                engine=nisa.engine.vector,
            )
        else:
            nisa.select_reduce(
                dst=diag_c[0:tok, 0:tok],
                predicate=ident_sel[0:tok, 0:tok],
                on_true=w.ap(pattern=[[Cp1_even, tok], [0, tok]], offset=c),
                on_false=0.0,
            )
        return diag_c

    diags = []
    for c in range(Cp1):
        diag_c = diag_of(expv_lp, c)
        diags.append((diag_c, diag_of(expv_lo, c)) if split else diag_c)
    return diags, inv_denom


def _parts(diag):
    """The stationaries of one candidate's weight: one diagonal, or the hi and lo diagonals (``split_weights``)."""
    return diag if isinstance(diag, tuple) else (diag,)


def _mixture_matmuls(acc, diags, moving, tok, hw):
    """``acc[:, 0:hw] = sum_c diag(p_c) @ moving[c]`` on TensorE, accumulated in PSUM."""
    for c in range(len(diags)):
        for k, diag in enumerate(_parts(diags[c])):
            # One matmul per PSUM bank: wider matmuls that span banks give nondeterministic
            # results past the first bank on hardware.
            for b0 in range(0, hw, _PSUM_BANK):
                nisa.nc_matmul(
                    dst=acc[0:tok, b0 : b0 + _PSUM_BANK],
                    stationary=diag[0:tok, 0:tok],
                    moving=moving[c][0:tok, b0 : b0 + _PSUM_BANK],
                    accumulate=c > 0 or k > 0,
                )


def _load_candidate_t(dst, bank, stream, c, t_off, tok):
    """DMA-transpose candidate ``c`` for tokens ``[t_off, t_off + tok)`` into
    ``dst[h, j, t] = v_c[t, j * 128 + h]``: H on partitions, in 128-column blocks."""
    T, C, H = bank.shape
    n_hb = H // _PMAX
    src, row, off = (bank, C * H, t_off * C * H + c * H) if c < C else (stream, H, t_off * H)
    nisa.dma_transpose(
        dst=dst.ap([[n_hb * _T_BLOCK, _PMAX], [1, 1], [_T_BLOCK, n_hb], [1, tok]]),
        src=src.ap([[row, tok], [1, 1], [_PMAX, n_hb], [1, _PMAX]], offset=off),
    )


def _transpose_slot(dst, v_c, tok, H, row):
    """DMA-transpose a resident candidate ``v_c [tok, H]`` within SBUF into
    ``dst[h, j, t] = v_c[t, j * 128 + h]``. A dynamic loop admits static DMAs only
    from internal tensors, so this transposes the loaded copy instead of HBM.
    ``v_c`` is a view of a tensor with ``row`` elements per partition."""
    n_hb = H // _PMAX
    nisa.dma_transpose(
        dst=dst.ap([[n_hb * _T_BLOCK, _PMAX], [1, 1], [_T_BLOCK, n_hb], [1, tok]]),
        src=v_c.ap([[row, tok], [1, 1], [_PMAX, n_hb], [1, _PMAX]]),
    )


# Tokens per row group of a hardware-DGE transpose: it transposes at most 16 rows.
_HW_T_ROWS = 16


def _load_candidate_t_hw(dst, bank, stream, c, t_off, tok):
    """``_load_candidate_t`` on the hardware DGE: a dynamic loop admits no static DMA
    from a kernel input. One DMA moves the ``tok // 16`` groups of 16 tokens, as the
    outer dimension of its access pattern."""
    T, C, H = bank.shape
    n_hb = H // _PMAX
    src, row, off = (bank, C * H, t_off * C * H + c * H) if c < C else (stream, H, t_off * H)
    n_rg = tok // _HW_T_ROWS
    nisa.dma_transpose(
        dst=dst.ap([[n_hb * _T_BLOCK, _PMAX], [_HW_T_ROWS, n_rg], [_T_BLOCK, n_hb], [1, _HW_T_ROWS]]),
        src=src.ap([[row, _HW_T_ROWS], [_HW_T_ROWS * row, n_rg], [_PMAX, n_hb], [1, _PMAX]], offset=off),
        dge_mode=nisa.dge_mode.hwdge,
    )


def _score_on_pe(v_t, gram, tok, n_hb, after=()):
    """Accumulate ``gram[t, :] = sum_h v[t, h] * [v[:, h] | sw_hi[h] | sw_lo[h]]`` on
    TensorE from a transposed candidate: the diagonal of the first ``tok`` columns is
    ``sum_h v ** 2`` and the last two columns sum to ``sum_h v * score_w``. The
    first matmul waits for the instructions ``after``."""
    for j in range(n_hb):
        mm = nisa.nc_matmul(
            dst=gram[0:tok, 0 : _PMAX + 2],
            stationary=v_t[0:_PMAX, j, 0:tok],
            moving=v_t[0:_PMAX, j, 0 : _PMAX + 2],
            accumulate=j > 0,
        )
        if j == 0:
            _after(mm, after)


def _row_elems(t):
    """Elements per partition of an SBUF tensor (the partition step of its access patterns)."""
    n = 1
    for d in t.shape[1:]:
        n *= d
    return n


def _gram_to_scores(gram, ident_sel, diag_junk, ss, dot, c, tok):
    """Read ``ss`` (the Gram diagonal) and ``dot`` for candidate ``c`` out of PSUM.

    The diagonal comes from a select (predicate ``ident_sel``, the identity) and a max
    over the row: the off-diagonal Gram entries pair this token with the others, so a
    multiply by the identity would turn another token's non-finite entry into this
    token's NaN (0 * NaN). The diagonal ``sum v ** 2`` is >= 0, so the max of it and
    the selected zeros is the diagonal itself. ``diag_junk`` is a write-only fp32
    element per partition (a zero-stride destination)."""
    nisa.select_reduce(
        dst=diag_junk.ap(pattern=[[_row_elems(diag_junk), tok], [0, tok]], offset=0),
        predicate=ident_sel[0:tok, 0:tok],
        on_true=gram[0:tok, 0:tok],
        on_false=0.0,
        reduce_res=ss[0:tok, nl.ds(c, 1)],
        reduce_cmd=nisa.reduce_cmd.reset_reduce,
        reduce_op=nl.maximum,
    )
    nisa.tensor_reduce(dst=dot[0:tok, nl.ds(c, 1)], data=gram[0:tok, _PMAX : _PMAX + 2], op=nl.add, axis=(1,))


def _evict_chunk(
    acc, inv_denom, out_bf, out, t_off, tok, h0, hw, H, engine=nisa.engine.unknown, in_loop=False, store_dge=None
):
    """Normalize one PSUM mixture chunk by ``inv_denom``, round it to the output dtype
    on ``engine``, and store it (with ``store_dge``'s DMA options if given)."""
    nisa.tensor_scalar(
        dst=out_bf[0:tok, 0:hw],
        data=acc[0:tok, 0:hw],
        op0=nl.multiply,
        operand0=inv_denom[0:tok, 0:1],
        engine=engine,
    )
    nisa.dma_copy(
        dst=out.ap(pattern=[[H, tok], [1, hw]], offset=t_off * H + h0),
        src=out_bf[0:tok, 0:hw],
        **(_dge(in_loop) if store_dge is None else store_dge),
    )


def _variant_flags(valid_len, shard_off, seq_per_shard, group_starts, H):
    """The 0/1 trip registers of the valid-length variants, and the element offset in
    ``out`` of the shard's first pad row as a ``[1, 1]`` int32 tile.

    Variant k runs the pipeline over the tile groups ``0..k``: it is the one that runs
    if row ``valid_len`` falls in group k of the first shard. The last variant also
    runs when there is no pad row. Every shard runs the same variant: the cores of a
    logical core hang if they take different loops (their hardware DGE queues are
    shared). That variant also covers each later shard's first pad row, which is its
    first row unless the first shard has no pad row. The registers are ``run_k`` for
    each of the K variants, then (for K > 2) ``run_0 + ... + run_{K-2}``."""
    vl = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    # The registers gate the first instruction of the pipeline, so the load goes first
    # and on its own queue.
    nisa.dma_copy(dst=vl, src=valid_len.ap(pattern=[[1, 1], [1, 1]], offset=0), **_SIDE)
    vlf = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=vlf, src=vl, engine=nisa.engine.vector)
    # The first pad row of this shard: row valid_len, or the shard's first row if
    # valid_len falls before the shard. Without pad rows it is clamped to the last
    # row, which the last variant computes. row * H is exact in fp32: H % 512 == 0, so
    # it is (row * H / 512) * 512 with row * H / 512 < 2^24.
    pad_off = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    pad_rowf = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=pad_rowf,
        data=vlf,
        op0=nl.maximum,
        operand0=float(shard_off),
        op1=nl.minimum,
        operand1=float(shard_off + seq_per_shard - 1),
    )
    nisa.tensor_scalar(dst=pad_off, data=pad_rowf, op0=nl.multiply, operand0=float(H))
    K = len(group_starts)
    # f[g] = (first row of group g <= valid_len), with f[0] = 1 and f[K] = 0; variant k
    # runs if f[k] - f[k + 1] = 1.
    first_row = nl.ndarray((1, K + 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=first_row[0:1, 0:1], value=-1.0e9)
    for g in range(1, K):
        nisa.memset(dst=first_row[0:1, nl.ds(g, 1)], value=float(group_starts[g]))
    nisa.memset(dst=first_row[0:1, nl.ds(K, 1)], value=1.0e9)
    f = nl.ndarray((1, K + 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=f, data1=first_row, data2=vlf.ap([[1, 1], [0, K + 1]]), op=nl.less_equal)
    runs = nl.ndarray((1, K + 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=runs[0:1, 0:K], data1=f[0:1, 0:K], data2=f[0:1, 1 : K + 1], op=nl.subtract)
    nisa.tensor_tensor(dst=runs[0:1, K : K + 1], data1=f[0:1, 0:1], data2=f[0:1, K - 1 : K], op=nl.subtract)
    regs = []
    for k in range(K + (1 if K > 2 else 0)):
        reg = nisa.register_alloc()
        nisa.register_load(reg, runs[0:1, nl.ds(k, 1)])
        regs.append(reg)
    return regs, pad_off


@nki.jit
def attn_res_cte(
    bank: nl.NkiTensor,
    stream: nl.NkiTensor,
    score_w: nl.NkiTensor,
    eps: float,
    valid_len: nl.NkiTensor = None,
    bank_scores: nl.NkiTensor = None,
    split_weights: bool = False,
    ocp_fp8_sinks: bool = False,
    stream_sync_dge: bool = False,
):
    """Softmax mixture of the residual stream with its banked snapshots.

    Args:
        bank: ``[T, C, H]`` bf16 or fp32 --- ``C`` snapshot candidates per token
            (``1 <= C <= 15``). Row-major, ``H`` contiguous.
        stream: ``[T, H]`` in ``bank``'s dtype --- the current residual stream
            (the ``(C+1)``-th candidate).
        score_w: ``[H]`` fp32 --- the fused score vector
            (``norm.weight * proj.weight``), precomputed by the caller.
        eps: RMSNorm epsilon.
        valid_len: optional ``[1]`` int32 --- the count of real rows; the rows from
            ``valid_len`` on are pad rows, which the caller guarantees are all
            identical (the prefill pads with copies of one pad token). The kernel
            then computes the rows up to the end of the group of rows (1024, 1024,
            2048, ... rows long) that holds the first pad row, and copies that pad
            row's output to the rows after it: the output equals the full
            computation. ``None`` (the default) computes every row, as does the
            re-load path (fp32 at large ``C``). Without pad rows (``valid_len`` =
            ``T``) the cost is about that of ``None``.
        bank_scores: optional ``[T, S]`` fp32 with ``S >= C`` --- the scores of the
            bank entries for this call's ``score_w``:
            ``bank_scores[t, c] = dot_c * rsqrt(ss_c / H + eps)`` for ``c < C``, as
            in the operator below; columns from ``C`` on are not read. Within a
            prefill stage the bank only grows, and its scores against every
            ``score_w`` of the stage can be computed once per stage. The kernel then
            scores only the stream, and streams the bank entries through the
            mixture without keeping them resident. ``None`` (the default) scores
            every candidate.
        split_weights: opt-in, a precision change (default False: the kernel is
            unchanged). With a bf16 input, the TensorE mixture takes each weight
            ``exp_c`` as two bf16 diagonals, ``hi = bf16(exp_c)`` and
            ``lo = bf16(exp_c - hi)``, so the products and their fp32 PSUM sums see
            the fp32 weight to about 2^-17 instead of one bf16 rounding (2^-9). The
            output then has the error of an fp32 mixture rounded once to bf16, as the
            GPU's fp32 AttnRes. The mixture matmuls double. An fp32 input already
            mixes with fp32 weights, so it ignores the option.
        ocp_fp8_sinks: opt-in (default False: the kernel is unchanged). Type the two
            write-only scoring destinations as ``float8_e4m3fn`` instead of
            ``float8_e4m3``. The compiler puts the FP8 converters of a whole
            program in the ``float8_e4m3`` mode (maximum 240, exponent 1111 read
            as Inf or NaN) when any ``float8_e4m3`` tensor exists. The option
            keeps this kernel free of such tensors, so a program that holds
            ``float8_e4m3fn`` data stays in the OCP mode (maximum 448). The kernel's
            instructions and outputs are the same; only the program-wide mode can
            change.
        stream_sync_dge: opt-in (default False: the kernel is unchanged). With
            ``bank_scores``, load the stream tiles on the Sync hardware DGE queue at
            every bank width, as the path does at ``C = 1``; the bank loads and the
            output stores keep their queues. The loaded values are the same, so the
            outputs do not change.

    Returns:
        ``out: [T, H]`` in ``stream``'s dtype.

    Notes:
        - All reductions and the softmax run in fp32; only the loaded candidates, the
          mixture weights of the TensorE stationaries (each rounded once, the largest
          one exact) and the stored output are in the input dtype.
        - ``T`` must be a multiple of 128 and of the SPMD grid size; ``H`` a multiple
          of 512. Both are checked with ``kernel_assert``.
        - A non-finite score of one row (from its candidates or its ``bank_scores``)
          stays in that row's output. A non-finite candidate element ``(t, j)`` reaches
          column ``j`` of every row of its 128-row tile, because the TensorE mixture
          contracts the candidates over the rows (``0 * NaN``): the caller keeps the
          candidates finite (for example zeroed pad rows).
    """
    T, C, H = bank.shape
    Cp1 = C + 1
    dtype = stream.dtype

    kernel_assert(T % _PMAX == 0, f"attn_res_cte requires T % {_PMAX} == 0, got T={T}")
    kernel_assert(H % _HCHUNK == 0, f"attn_res_cte requires H % {_HCHUNK} == 0, got H={H}")
    kernel_assert(C <= _MAX_C, f"attn_res_cte supports C <= {_MAX_C} candidates, got C={C}")
    kernel_assert(
        bank.dtype == dtype,
        f"attn_res_cte requires bank and stream to share a dtype, got {bank.dtype} vs {dtype}",
    )
    kernel_assert(
        score_w.dtype == nl.float32,
        f"attn_res_cte requires score_w in float32, got {score_w.dtype}",
    )
    scored = bank_scores is not None
    if scored:
        kernel_assert(
            bank_scores.dtype == nl.float32,
            f"attn_res_cte requires bank_scores in float32, got {bank_scores.dtype}",
        )
        kernel_assert(
            len(bank_scores.shape) == 2 and bank_scores.shape[0] == T and bank_scores.shape[1] >= C,
            f"attn_res_cte requires bank_scores of shape [T, S >= C], got {bank_scores.shape}",
        )

    _, num_shards, shard_id = get_verified_program_sharding_info("attn_res_cte", (0, 1))
    kernel_assert(
        T % num_shards == 0,
        f"attn_res_cte requires T % num_shards == 0, got T={T}, num_shards={num_shards}",
    )
    seq_per_shard = T // num_shards
    tile = min(_PMAX, seq_per_shard)
    n_tiles = div_ceil(seq_per_shard, tile)
    shard_off = shard_id * seq_per_shard

    out = nl.ndarray((T, H), dtype=dtype, buffer=nl.shared_hbm)

    # Tile groups of the valid-length path, as [first tile, last tile + 1).
    first_tiles = [0]
    nxt = max(1, _VALID_GROUP_ROWS // tile)
    while nxt < n_tiles:
        first_tiles.append(nxt)
        nxt *= 2
    groups = list(zip(first_tiles, first_tiles[1:] + [n_tiles]))
    n_groups = len(groups)
    n_hb = H // _PMAX
    n_pe = min(_pe_scored(C), Cp1)
    budget = nl.tile_size.total_available_sbuf_size - _SBUF_RESERVE
    dsize = sizeinbytes(dtype)
    n_slots = min(2 * Cp1, (budget - _ring_fixed_bytes(H, Cp1, dsize)) // (H * dsize))
    # The pipelined paths: the ring of resident candidates, or with given bank scores
    # the stream of bank chunks. The valid-length variants need one of them; the
    # re-load path computes every row.
    pipelined = scored or n_slots >= Cp1
    variants = valid_len is not None and n_groups > 1 and pipelined
    if variants:
        var_regs, pad_off = _variant_flags(valid_len, shard_off, seq_per_shard, [t0 * tile for t0, _ in groups], H)

    # score_w broadcast to every partition once (it does not depend on the token).
    # Partition stride 0 in the access pattern replicates the same H-vector to all
    # 128 partitions. The valid-length variants load it in their loop (below).
    sw = nl.ndarray((_PMAX, H), dtype=nl.float32, buffer=nl.sbuf)
    if not variants:
        # With given bank scores the bank loads start at once; the broadcast takes the
        # Sync queue, off theirs.
        nisa.dma_copy(dst=sw, src=score_w.ap(pattern=[[0, _PMAX], [1, H]], offset=0), **(_SIDE if scored else {}))

    inv_h = 1.0 / float(H)
    vector_exp = is_trn3_b1()
    chunks = [(h0, min(_MIX_CHUNK, H - h0)) for h0 in range(0, H, _MIX_CHUNK)]

    # Write-only destinations of the scoring passes, one per engine and shared by all
    # candidates: the values are never read, and an engine runs its own instructions
    # in order, so sharing costs no synchronization. Each pass writes all its
    # elementwise results to one fp8 element per partition (a zero-stride access
    # pattern), which leaves 14 KB per partition to the ring: the fused sums
    # accumulate in fp32 before the output cast, and the discarded values (and any
    # overflow in them) are never read. Their FP8 type sets the program-wide FP8
    # mode (ocp_fp8_sinks).
    sink_dtype = nl.float8_e4m3fn if ocp_fp8_sinks else nl.float8_e4m3
    sq = nl.ndarray((_PMAX, 1), dtype=sink_dtype, buffer=nl.sbuf)
    prod = nl.ndarray((_PMAX, 1), dtype=sink_dtype, buffer=nl.sbuf)

    # The diagonal of the diag(p_c) stationaries of the TensorE mixture: in bf16 the
    # tile's own bits as uint16, 0xFFFF on the diagonal and 0 elsewhere (the mask of a
    # bitwise AND, and a predicate of the selects); in fp32 an identity and a uint8
    # copy, the predicate of the selects.
    ones = nl.ndarray((_PMAX, _PMAX), dtype=dtype, buffer=nl.sbuf)
    ident = nl.ndarray((_PMAX, _PMAX), dtype=dtype, buffer=nl.sbuf)
    sel_bits = dsize == 2
    ident_sel = ident.view(nl.uint16) if sel_bits else nl.ndarray((_PMAX, _PMAX), dtype=nl.uint8, buffer=nl.sbuf)

    def make_ident():
        sel = nisa.affine_select(
            dst=ident,
            pattern=[[-1, _PMAX]],
            channel_multiplier=1,
            on_true_tile=ones,
            on_false_value=0.0,
            cmp_op=nl.equal,
        )
        # The affine select runs on GpSimd, and the compiler can leave its readers
        # on other engines unordered. Rewrite the identity on VectorE once the select
        # is done, so every reader waits for a VectorE write: in bf16 into the 0xFFFF
        # diagonal mask, in fp32 as x != 0 (which keeps its 0 and 1).
        if sel_bits:
            rewrite = nisa.tensor_scalar(
                dst=ident_sel,
                data=ident,
                op0=nl.not_equal,
                operand0=0.0,
                op1=nl.multiply,
                operand1=65535.0,
                engine=nisa.engine.vector,
            )
        else:
            rewrite = nisa.tensor_scalar(
                dst=ident, data=ident, op0=nl.not_equal, operand0=0.0, engine=nisa.engine.vector
            )
        _after(rewrite, [sel])
        if not sel_bits:
            nisa.tensor_scalar(dst=ident_sel, data=ident, op0=nl.not_equal, operand0=0.0, engine=nisa.engine.vector)

    if not variants:
        nisa.memset(dst=ones, value=1.0)
        make_ident()

    # Resident candidates rotate through a ring of n_slots [tok, H] tiles. A ring
    # larger than one tile's C+1 candidates lets the next tile's first n_extra
    # candidates load while this tile is still mixed. When one tile's candidates do
    # not fit the budget (at H = 7168 on trn3: C > 11 in bf16, C > 4 in fp32), the
    # mixture re-loads them chunk by chunk instead.
    if pipelined:
        if not scored:
            # The last n_pe candidates of each tile are scored on TensorE from a
            # transposed copy (one shared buffer), which takes n_pe of the C+1 scoring
            # passes off ScalarE and VectorE.
            v_t = nl.ndarray((_PMAX, n_hb, _T_BLOCK), dtype=dtype, buffer=nl.sbuf)
            # fp32 (the select keeps its input dtype), as many bytes as the fp8 [128, 128] it was.
            diag_junk = nl.ndarray((_PMAX, _PMAX // 4), dtype=nl.float32, buffer=nl.sbuf)
            # Columns 128 and 129 of every block hold score_w split as bf16 hi + lo, so
            # the TensorE dot keeps about 16 mantissa bits. The transposes only write
            # columns [0, tok).
            sw_t = nl.ndarray((_PMAX, n_hb), dtype=nl.float32, buffer=nl.sbuf)

        # The writes of the score_w columns of v_t. The first Gram waits for them
        # explicitly: the transposes write the other columns of v_t, and the
        # compiler can drop the Gram's dependency on these writes.
        sw_src = []

        def load_sw_t(**dge):
            nisa.dma_copy(dst=sw_t, src=score_w.ap(pattern=[[1, _PMAX], [_PMAX, n_hb]], offset=0), **dge)
            hi = nisa.tensor_copy(dst=v_t[0:_PMAX, 0:n_hb, _PMAX], src=sw_t[0:_PMAX, 0:n_hb])
            lo = nisa.tensor_tensor(
                dst=v_t[0:_PMAX, 0:n_hb, _PMAX + 1],
                data1=sw_t[0:_PMAX, 0:n_hb],
                data2=v_t[0:_PMAX, 0:n_hb, _PMAX],
                op=nl.subtract,
            )
            sw_src[:] = [hi, lo]

        if not variants and not scored:
            nisa.memset(dst=v_t[0:_PMAX, 0:n_hb, 0:_PMAX], value=0.0)
            load_sw_t()
        # With given bank scores the slots hold the streams only. The slots, and the
        # bank ring below, are views of one tensor, so no load writes a whole SBUF
        # tensor. Some neuronx-cc builds drop a load that writes a whole tensor
        # when another tensor still holds the same HBM bytes, and then sends every
        # later read of the dropped tensor to that one, even past the next load into
        # it. With two calls in one program that read the same rows, a later call's
        # tile then mixes rows that an earlier call left in SBUF. The compiler leaves
        # loads into part of a tensor alone.
        n_ring_slots = _STREAM_SLOTS if scored else n_slots
        slot_ring = nl.ndarray((_PMAX, n_ring_slots, H), dtype=dtype, buffer=nl.sbuf)
        slots = [slot_ring[0:_PMAX, k, 0:H] for k in range(n_ring_slots)]
        wide_out = scored and C in _WIDE_OUT_C
        out_w = H if wide_out else _MIX_CHUNK
        out_ring = [nl.ndarray((_PMAX, out_w), dtype=dtype, buffer=nl.sbuf) for _ in range(2)]
        if scored:
            # The bank chunks of the mixture rotate through a ring of [tok, cg, chunk]
            # entries, one per tile, chunk and group of cg bank entries, in the order
            # the mixture reads them. The ring takes the rest of the budget.
            ring_bytes = budget - _scored_fixed_bytes(H, Cp1, dsize, n_tiles) - 2 * (out_w - _MIX_CHUNK) * dsize
            n_cg = div_ceil(C, max(1, min(C, ring_bytes // (2 * _MIX_CHUNK * dsize))))
            cg = div_ceil(C, n_cg)
            # An entry spans jw chunks, so that each load moves about _ENTRY_BYTES per
            # partition: small loads waste DMA time on fixed costs.
            jw = max(1, min(len(chunks), _ENTRY_BYTES // (cg * _MIX_CHUNK * dsize)))
            n_jg = div_ceil(len(chunks), jw)
            ring_w = min(jw * _MIX_CHUNK, H)
            n_ring = min(max(3, 2 * n_jg * n_cg), ring_bytes // (cg * ring_w * dsize))
            kernel_assert(n_ring >= 2, f"attn_res_cte: no SBUF for the bank chunks at C={C}, H={H}")
            bank_ring = nl.ndarray((_PMAX, n_ring, cg, ring_w), dtype=dtype, buffer=nl.sbuf)
            ring = [bank_ring[0:_PMAX, e, 0:cg, 0:ring_w] for e in range(n_ring)]

        def tile_rows(i):
            return min(tile, seq_per_shard - i * tile), shard_off + i * tile

        class _Mixture:
            """The previous tile's TensorE mixture, issued in step with its evictions.

            Chunk j accumulates in psum_ring[j % 2], so chunk j + 2's matmuls are
            issued right after chunk j is evicted. ``after_last_read`` runs once the
            last matmul reading the candidates is issued: the next tile's loads into
            the same slots must come after it in program order."""

            def __init__(self, psum_ring, tok, t_off, weights, cands, after_last_read, in_loop):
                self.psum_ring = psum_ring
                self.tok, self.t_off, self.cands = tok, t_off, cands
                self.diags, self.inv_denom = weights
                self.after_last_read = after_last_read
                self.in_loop = in_loop
                self.n_evicted = 0
                for j in range(min(2, len(chunks))):
                    self._matmuls(j)

            def _matmuls(self, j):
                h0, hw = chunks[j]
                moving = [v[0 : self.tok, h0 : h0 + hw] for v in self.cands]
                _mixture_matmuls(self.psum_ring[j % 2], self.diags, moving, self.tok, hw)
                if j == len(chunks) - 1:
                    self.after_last_read()

            def done(self):
                return self.n_evicted == len(chunks)

            def evict_next(self):
                j = self.n_evicted
                h0, hw = chunks[j]
                # Alternate the evictions between VectorE and ScalarE: VectorE is
                # the busier engine, and both are otherwise loaded by the scoring.
                engine = nisa.engine.vector if j % 2 == 0 else nisa.engine.scalar
                _evict_chunk(
                    self.psum_ring[j % 2],
                    self.inv_denom,
                    out_ring[j % 2],
                    out,
                    self.t_off,
                    self.tok,
                    h0,
                    hw,
                    H,
                    engine,
                    self.in_loop,
                )
                self.n_evicted += 1
                if j + 2 < len(chunks):
                    self._matmuls(j + 2)

        def run_scored(n, in_loop, after_first_loads=None):
            """The pipeline over the shard's first ``n`` tiles, with given bank scores.

            Only a tile's stream is scored (ScalarE and VectorE), during the previous
            tile's mixture, in column pieces between its PSUM evictions; the softmax
            takes the bank entries' scores from ``bank_scores``, all of a run's tiles
            in one DMA. A stream is loaded two tiles ahead of its mixture. The mixture
            reads the resident stream first, then the bank entries from the ring: each
            ring entry is loaded into the slot that the mixture freed ``n_ring``
            entries before, so the loads run ahead of TensorE by up to the whole ring.
            The bank loads keep the other paths' queue. At ``C`` in
            ``_SCORED_SIDE_C`` the streams, the scores and the output stores take the
            Sync hardware DGE instead, which pays off only while the bank loads do not
            need the whole DMA bandwidth."""
            psum_ring = [nl.ndarray((_PMAX, _MIX_CHUNK), dtype=nl.float32, buffer=nl.psum) for _ in range(2)]
            Cp1_even = _scores_width(Cp1, vector_exp)
            Cs = bank_scores.shape[1]
            entries = [(i, jg, g) for i in range(n) for jg in range(n_jg) for g in range(n_cg)]
            side_stream = _SIDE if (C in _SCORED_SIDE_C or stream_sync_dge) else _dge(in_loop)
            side_store = _SIDE if C in _SCORED_SIDE_C else None
            # The stream is scored in n_pieces column pieces, one after each of the
            # first n_pieces evictions; the softmax follows the last piece.
            n_pieces = max(1, len(chunks) - 1)
            pieces = [(k * H // n_pieces, (k + 1) * H // n_pieces) for k in range(n_pieces)]

            def load_entry(e):
                i, jg, g = entries[e]
                tok, t_off = tile_rows(i)
                h0 = chunks[jg * jw][0]
                h1 = sum(chunks[min(len(chunks), (jg + 1) * jw) - 1])
                c0, nc = g * cg, min(cg, C - g * cg)
                nisa.dma_copy(
                    dst=ring[e % n_ring][0:tok, 0:nc, 0 : h1 - h0],
                    src=bank.ap(pattern=[[C * H, tok], [H, nc], [1, h1 - h0]], offset=t_off * C * H + c0 * H + h0),
                    **_dge(in_loop),
                )

            def load_stream(i):
                tok, t_off = tile_rows(i)
                nisa.dma_copy(
                    dst=slots[i % _STREAM_SLOTS][0:tok, 0:H],
                    src=stream.ap(pattern=[[H, tok], [1, H]], offset=t_off * H),
                    **side_stream,
                )
                ss = nl.ndarray((_PMAX, n_pieces), dtype=nl.float32, buffer=nl.sbuf)
                dot = nl.ndarray((_PMAX, n_pieces), dtype=nl.float32, buffer=nl.sbuf)
                return ss, dot

            def score_piece(i, state, k):
                tok, _ = tile_rows(i)
                ss, dot = state
                h0, h1 = pieces[k]
                v = slots[i % _STREAM_SLOTS]
                nisa.activation_reduce(
                    dst=sq.ap([[1, tok], [0, h1 - h0]]),
                    op=nl.square,
                    data=v[0:tok, h0:h1],
                    reduce_op=nl.add,
                    reduce_res=ss[0:tok, nl.ds(k, 1)],
                )
                nisa.scalar_tensor_tensor(
                    dst=prod.ap([[1, tok], [0, h1 - h0]]),
                    data=v[0:tok, h0:h1],
                    op0=nl.multiply,
                    operand0=1.0,
                    op1=nl.multiply,
                    operand1=sw[0:tok, h0:h1],
                    reduce_op=nl.add,
                    reduce_res=dot[0:tok, nl.ds(k, 1)],
                )

            def finish_weights(i, state):
                # The stream's score completes the bank entries' scores of the tile.
                tok, _ = tile_rows(i)
                ss, dot = state
                # The tile's bank scores, then the stream's in column C.
                scores = nl.ndarray((_PMAX, Cp1_even), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=scores[0:tok, 0:C], src=all_scores[0:tok, i, 0:C])
                if n_pieces > 1:
                    ss1 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                    dot1 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_reduce(dst=ss1[0:tok, 0:1], data=ss[0:tok, 0:n_pieces], op=nl.add, axis=(1,))
                    nisa.tensor_reduce(dst=dot1[0:tok, 0:1], data=dot[0:tok, 0:n_pieces], op=nl.add, axis=(1,))
                    ss, dot = ss1, dot1
                r = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=r[0:tok, 0:1], data=ss[0:tok, 0:1], op=nl.rsqrt, scale=inv_h, bias=eps)
                nisa.tensor_tensor(dst=scores[0:tok, C:Cp1], data1=dot[0:tok, 0:1], data2=r[0:tok, 0:1], op=nl.multiply)
                return _softmax_weights(scores, Cp1, tok, vector_exp, ident_sel, dtype, split_weights)

            def matmuls(acc, diag, moving, tok, hw, first):
                for k, part in enumerate(_parts(diag)):
                    for b0 in range(0, hw, _PSUM_BANK):
                        nisa.nc_matmul(
                            dst=acc[0:tok, b0 : b0 + _PSUM_BANK],
                            stationary=part[0:tok, 0:tok],
                            moving=moving[0:tok, b0 : b0 + _PSUM_BANK],
                            accumulate=not first or k > 0,
                        )

            # The bank scores of all n tiles in one DMA, after the first stream.
            all_scores = nl.ndarray((_PMAX, n, C), dtype=nl.float32, buffer=nl.sbuf)
            tok0, _ = tile_rows(0)

            def load_scores():
                nisa.dma_copy(
                    dst=all_scores[0:tok0, 0:n, 0:C],
                    src=bank_scores.ap(pattern=[[Cs, tok0], [tile * Cs, n], [1, C]], offset=shard_off * Cs),
                    **side_stream,
                )

            states = {}
            for i in range(min(_STREAM_SLOTS - 1, n)):
                if i == 1:
                    load_scores()
                states[i] = load_stream(i)
            if n == 1:
                load_scores()
            for e in range(min(n_ring, len(entries))):
                load_entry(e)
            if after_first_loads is not None:
                after_first_loads()
            for k in range(n_pieces):
                score_piece(0, states[0], k)
            w = finish_weights(0, states.pop(0))
            for i in range(n):
                # The slot of tile i - 1's stream is free once its mixture is issued.
                if i + _STREAM_SLOTS - 1 < n:
                    states[i + _STREAM_SLOTS - 1] = load_stream(i + _STREAM_SLOTS - 1)
                tok, t_off = tile_rows(i)
                diags, inv_denom = w
                st = slots[i % _STREAM_SLOTS]
                e0 = i * n_jg * n_cg
                for j, (h0, hw) in enumerate(chunks):
                    acc = psum_ring[j % 2]
                    jg, jo = j // jw, (j % jw) * _MIX_CHUNK
                    # The resident stream first: its matmuls wait for no load.
                    matmuls(acc, diags[C], st[0:tok, h0 : h0 + hw], tok, hw, True)
                    for g in range(n_cg):
                        ent = ring[(e0 + jg * n_cg + g) % n_ring]
                        for k in range(min(cg, C - g * cg)):
                            matmuls(acc, diags[g * cg + k], ent[0:tok, k, jo : jo + hw], tok, hw, False)
                    if j == len(chunks) - 1 or (j + 1) % jw == 0:
                        # The entries of this chunk group are read: refill their slots.
                        for g in range(n_cg):
                            e = e0 + jg * n_cg + g
                            if e + n_ring < len(entries):
                                load_entry(e + n_ring)
                    # Alternate the evictions between VectorE and ScalarE, which also
                    # score the next tile's stream.
                    engine = nisa.engine.vector if j % 2 == 0 else nisa.engine.scalar
                    if wide_out:
                        # One store per tile: the chunks land in one [tok, H] buffer.
                        ob = out_ring[i % 2]
                        nisa.tensor_scalar(
                            dst=ob[0:tok, h0 : h0 + hw],
                            data=acc[0:tok, 0:hw],
                            op0=nl.multiply,
                            operand0=inv_denom[0:tok, 0:1],
                            engine=engine,
                        )
                        if j == len(chunks) - 1:
                            nisa.dma_copy(
                                dst=out.ap(pattern=[[H, tok], [1, H]], offset=t_off * H),
                                src=ob[0:tok, 0:H],
                                **(_dge(in_loop) if side_store is None else side_store),
                            )
                    else:
                        _evict_chunk(
                            acc, inv_denom, out_ring[j % 2], out, t_off, tok, h0, hw, H, engine, in_loop, side_store
                        )
                    if i + 1 < n and j < n_pieces:
                        score_piece(i + 1, states[i + 1], j)
                        if j == n_pieces - 1:
                            w = finish_weights(i + 1, states.pop(i + 1))

        def run(n, in_loop, after_first_loads=None):
            """The software pipeline over the shard's first ``n`` tiles.

            Tile i's scoring runs on ScalarE/VectorE while tile i-1's mixture runs on
            TensorE, and VectorE evicts tile i-1's mixture chunks between tile i's
            scoring passes. The order is not pinned with nl.no_reorder(): when the
            kernel is inlined into a graph that keeps tensors in SBUF across it, the
            SBUF allocator spills them, and it cannot place a reload at the end of a
            no_reorder region. In a dynamic loop every load and store takes the
            software DGE, and no static DMA may read HBM (below for the transposes).
            The PSUM accumulators are allocated per run: a PSUM tensor cannot be live
            across a dynamic loop. ``after_first_loads`` issues the loop's setup after
            the first tile's loads."""
            if scored:
                return run_scored(n, in_loop, after_first_loads)
            # The ring sizes are measured on trn3 at H=7168. With two TensorE
            # candidates, a ring of C+5 slots puts the second one's transpose and Gram
            # on the critical path, so the run uses one slot fewer (C=3).
            r_pe = n_pe
            r_slots = n_slots
            if r_pe == 2 and n_slots == Cp1 + 4:
                r_slots -= 1
            r_extra = r_slots - Cp1
            r_cands = list(range(Cp1 - r_pe, Cp1))

            def slot(i, c):
                return slots[(i * Cp1 + c) % r_slots]

            # Each TensorE-scored candidate has its own Gram accumulator in PSUM, next
            # to the two mixture accumulators.
            gram_ring = [nl.ndarray((_PMAX, _PMAX + 2), dtype=nl.float32, buffer=nl.psum) for _ in range(r_pe)]
            psum_ring = [nl.ndarray((_PMAX, _MIX_CHUNK), dtype=nl.float32, buffer=nl.psum) for _ in range(2)]

            def load(li, cs):
                tok, t_off = tile_rows(li)
                for c in cs:
                    _load_candidate(slot(li, c), bank, stream, c, t_off, tok, 0, H, in_loop)

            def transpose(li, c):
                tok, t_off = tile_rows(li)
                if not in_loop:
                    _load_candidate_t(v_t, bank, stream, c, t_off, tok)
                elif c in slot_t:
                    _transpose_slot(v_t, slot(li, c), tok, H, n_ring_slots * H)
                else:
                    _load_candidate_t_hw(v_t, bank, stream, c, t_off, tok)

            def score_on_pe(li):
                # The transposed copies of tile li's TensorE-scored candidates share
                # v_t, so each next transpose waits for the previous TensorE pass. The
                # first one was prefetched with the previous tile.
                tok, _ = tile_rows(li)
                for k, c in enumerate(r_cands):
                    if k > 0:
                        transpose(li, c)
                    # Every later Gram runs after the first one on TensorE.
                    _score_on_pe(v_t, gram_ring[k], tok, n_hb, sw_src if li == 0 and k == 0 else ())

            def after_mixture_reads(li):
                # These slots held the previous tile's candidates, so the loads wait
                # for its mixture.
                load(li, range(r_extra, Cp1))
                score_on_pe(li)

            # In a dynamic loop the TensorE candidates are transposed from HBM on the
            # hardware DGE, and the first one still a tile ahead. At C=4 the second
            # one is transposed from its copy, which is loaded a tile ahead there:
            # that is faster than the hardware DGE behind the first one's Gram.
            slot_t = [c for c in r_cands[1:] if c < r_extra] if in_loop and C in _LOOP_SLOT_T else []
            load(0, range(Cp1))
            transpose(0, r_cands[0])
            if after_first_loads is not None:
                after_first_loads()
            score_on_pe(0)
            if n > 1:
                load(1, range(r_extra))

            prev = None
            for li in range(n):
                mix = None
                if prev is not None:
                    mix = _Mixture(psum_ring, *prev, lambda li=li: after_mixture_reads(li), in_loop)
                tok, t_off = tile_rows(li)
                cands = [slot(li, c) for c in range(Cp1)]

                ss = nl.ndarray((_PMAX, Cp1), dtype=nl.float32, buffer=nl.sbuf)
                dot = nl.ndarray((_PMAX, Cp1), dtype=nl.float32, buffer=nl.sbuf)
                for c in range(Cp1 - r_pe):
                    if mix is not None and c == r_extra:
                        # The next candidates wait for the previous mixture anyway, so
                        # finish its evictions first.
                        while not mix.done():
                            mix.evict_next()
                    _score_candidate(cands[c], c, tok, H, sw, sq, prod, ss, dot)
                    if mix is not None and c < r_extra and not mix.done():
                        mix.evict_next()
                while mix is not None and not mix.done():
                    mix.evict_next()
                for k, c in enumerate(r_cands):
                    _gram_to_scores(gram_ring[k], ident_sel, diag_junk, ss, dot, c, tok)

                weights = _mixture_weights(ss, dot, Cp1, tok, inv_h, eps, vector_exp, ident_sel, dtype, split_weights)
                if li + 1 < n:
                    if li > 0:
                        # Tile 1's first candidates were loaded before the loop.
                        load(li + 1, range(r_extra))
                    transpose(li + 1, r_cands[0])
                prev = (tok, t_off, weights, cands)

            mix = _Mixture(psum_ring, *prev, lambda: None, in_loop)
            while not mix.done():
                mix.evict_next()

        if not variants:
            run(n_tiles, False)
        else:
            # Variant k runs the pipeline over the tile groups 0..k, in a 0- or 1-trip
            # dynamic loop: exactly one variant runs. A variant that stops before the
            # last tile copies the output of the first pad row to the rows after its
            # groups: they are pad rows, and so is every row from the first pad row
            # on, so their output equals it. Each variant is its own pipeline, so no
            # pipeline drains at a group boundary. The full-length variant comes
            # first; the others share one loop, so the full length skips only one.

            def setup():
                # Issued after the first tile's loads, which they would delay. The
                # score_w broadcast takes the Sync queue, off the loads' one.
                nisa.dma_copy(dst=sw, src=score_w.ap(pattern=[[0, _PMAX], [1, H]], offset=0), **_SIDE)
                if not scored:
                    load_sw_t(**_dge(True))
                make_ident()

            def fill(e):
                # The first pad row's output, on every partition of a free slot.
                pad_out = slots[0]
                nisa.dma_copy(
                    dst=pad_out[0:_PMAX, 0:H],
                    src=out.ap(pattern=[[0, _PMAX], [1, H]], offset=0, scalar_offset=pad_off, indirect_dim=1),
                    dge_mode=nisa.dge_mode.swdge,
                )
                for t in range(e, n_tiles):
                    tok, t_off = tile_rows(t)
                    nisa.dma_copy(
                        dst=out.ap(pattern=[[H, tok], [1, H]], offset=t_off * H),
                        src=pad_out[0:tok, 0:H],
                        dge_mode=nisa.dge_mode.swdge,
                    )

            def variant(k):
                e = groups[k][1]
                if not scored:
                    nisa.memset(dst=v_t[0:_PMAX, 0:n_hb, 0:_PMAX], value=0.0)
                nisa.memset(dst=ones, value=1.0)
                run(e, True, setup)
                if e < n_tiles:
                    # A nested loop of the same trip count orders the fill after the
                    # pipeline's stores: a dynamic loop starts once they complete.
                    nl.fori_loop(0, var_regs[k], lambda _: fill(e))

            last = n_groups - 1
            nl.fori_loop(0, var_regs[last], lambda _: variant(last))
            if n_groups == 2:
                nl.fori_loop(0, var_regs[0], lambda _: variant(0))
            else:

                def short(_):
                    for k in range(last):
                        nl.fori_loop(0, var_regs[k], lambda _, k=k: variant(k))

                nl.fori_loop(0, var_regs[n_groups], short)
    else:
        for i in range(n_tiles):
            tok = min(tile, seq_per_shard - i * tile)
            t_off = shard_off + i * tile
            ss = nl.ndarray((_PMAX, Cp1), dtype=nl.float32, buffer=nl.sbuf)
            dot = nl.ndarray((_PMAX, Cp1), dtype=nl.float32, buffer=nl.sbuf)
            for c in range(Cp1):
                v_c = nl.ndarray((_PMAX, H), dtype=dtype, buffer=nl.sbuf)
                _load_candidate(v_c, bank, stream, c, t_off, tok, 0, H)
                _score_candidate(v_c, c, tok, H, sw, sq, prod, ss, dot)
            diags, inv_denom = _mixture_weights(
                ss, dot, Cp1, tok, inv_h, eps, vector_exp, ident_sel, dtype, split_weights
            )
            for h0, hw in chunks:
                acc = nl.ndarray((_PMAX, _MIX_CHUNK), dtype=nl.float32, buffer=nl.psum)
                moving = []
                for c in range(Cp1):
                    v_c = nl.ndarray((_PMAX, _MIX_CHUNK), dtype=dtype, buffer=nl.sbuf)
                    _load_candidate(v_c, bank, stream, c, t_off, tok, h0, hw)
                    moving.append(v_c)
                _mixture_matmuls(acc, diags, moving, tok, hw)
                out_bf = nl.ndarray((_PMAX, _MIX_CHUNK), dtype=dtype, buffer=nl.sbuf)
                _evict_chunk(acc, inv_denom, out_bf, out, t_off, tok, h0, hw, H)

    return out
