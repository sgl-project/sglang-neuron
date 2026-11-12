# SPDX-License-Identifier: Apache-2.0
"""A top-k select that carries a payload, in released NKI (nki 0.7) instructions only.

The GLM-5.2 kernels select the top ``k`` of ``n`` candidates per query and need, for each winner, a payload that is
data rather than arithmetic (the indexer merge's global key position). ``nisa.topk`` returns only the winners' values
and their indices within the query, and an index cannot be turned back into a position that was gathered from
elsewhere, so the payload has to travel through the select itself. This module does that with two ``nisa.topk``
calls:

1. ``nisa.topk`` over the scores gives every query's ``k``-th largest score, ``tau``.
2. Membership is decided in place, on the candidate tile itself: every candidate above ``tau``, then the first
   ``k - #(score > tau)`` candidates equal to ``tau`` (in partition-major order), so a query has exactly ``k``.
3. Each member's payload is written as the bit pattern of a finite bf16 (a non-member gets the lowest finite bf16),
   and a second ``nisa.topk`` over that tile returns exactly the ``k`` members, their payload as its values.

Step 3 is the point: ``nisa.topk`` moves values across the 16 partitions of a query, which no other released
instruction does at element granularity, so the payload rides through the select as the value.
The payload must be below ``CODE_LIMIT`` (just under 16 bits: its code is any normal bf16 bit pattern but the lowest);
the caller recovers anything more arithmetically from the winner's snake index (its column block or partition).

Tile convention (as ``nisa.topk``): a query owns 16 partitions; candidate ``e`` of a query sits at partition
``e % 16``, column ``e // 16``; result ``j`` is written at partition ``j % 16``, column ``j // 16``.
"""

from __future__ import annotations

import ml_dtypes
import nki.isa as nisa
import nki.language as nl
import numpy as np

SNAKE = 16
"""Partitions of one query in ``nisa.topk``."""

#: Bit patterns: payload ``x``'s code is ``x + 128`` (positive normal bf16) below ``_NEG_FROM``, ``x + 384``
#: (negative normal bf16) from there; a non-member is the lowest finite bf16, below every code.
_NEG_FROM = 32512
_NON_MEMBER = 0xFF7F
CODE_LIMIT = 2 * _NEG_FROM - 1
"""Payloads are ``0 .. CODE_LIMIT - 1`` (``CODE_LIMIT`` itself would encode to the non-member pattern)."""


def _require(condition: bool, message: str) -> None:
    """Refuse a call whose trace-time arguments the select cannot serve, naming what is wrong."""
    if not condition:
        raise ValueError(message)


def _log2(x: int) -> int:
    _require(x > 0 and x & (x - 1) == 0, f"{x} is not a power of two")
    return x.bit_length() - 1


def code_of(x: int) -> int:
    """The bit pattern a payload ``x`` (a trace-time int) is carried as: what :func:`encode_codes` writes."""
    _require(0 <= x < CODE_LIMIT, f"payload {x} is outside the codes 0 .. {CODE_LIMIT - 1}")
    return x + 128 + (256 if x >= _NEG_FROM else 0)


def _sbuf(alloc, shape, dtype, name):
    """An SBUF tile from the caller's allocator, or from ``nl.ndarray`` when there is none."""
    return nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf) if alloc is None else alloc(shape, dtype, name)


def _group_consts(P: int, alloc=None):
    """``same[k, m] = 1`` when partitions k and m belong to one query, ``lower[k, m] = same and k < m``: one bf16
    [P, 2P] tile (``same`` then ``lower``), from a constant embedded in the NEFF (one DMA, no compute)."""
    q = np.arange(P) // SNAKE
    same = (q[:, None] == q[None, :])
    lower = same & (np.arange(P)[:, None] < np.arange(P)[None, :])
    const = nl.shared_constant(np.concatenate([same, lower], axis=1).astype(ml_dtypes.bfloat16))
    both = _sbuf(alloc, (P, 2 * P), nl.bfloat16, "pt_grp")
    nisa.dma_copy(dst=both, src=const)
    return both[:, 0:P], both[:, P : 2 * P]


def _query_tau(val1, k_cols: int, P: int, alloc=None):
    """Each query's k-th largest score (the smallest of its k winners), on all 16 of its partitions, fp32 [P, 1].

    Each partition's negated minimum, written twice (neuronx-cc 2.28's TopK check wants an even 16-bit free size), feeds
    a tiny unsorted ``nisa.topk`` (n = 32, k = 2) per query: the query's maximum is in it twice, so both results are
    the maximum whatever their order, and result 0 lands on the query's first partition; one stream shuffle copies it
    to the other 15. (With distinct inputs an unsorted k = 2 result 0 could be the second largest.)
    """
    vec = nisa.engine.vector
    neg = _sbuf(alloc, (P, 2), nl.bfloat16, "pt_neg")
    nisa.tensor_reduce(dst=neg[:, 0:1], op=nl.min, data=val1[0:P, 0:k_cols], axis=1, negate=True)
    nisa.tensor_copy(dst=neg[:, 1:2], src=neg[:, 0:1], engine=vec)
    tv = _sbuf(alloc, (P, 2), nl.bfloat16, "pt_tv")
    ti = _sbuf(alloc, (P, 2), nl.uint32, "pt_ti")
    nisa.topk(val_dst=tv[0:P, 0:2], idx_dst=ti[0:P, 0:2], src=neg, n=2 * SNAKE)
    tb = _sbuf(alloc, (P, 2), nl.bfloat16, "pt_tb")
    nisa.nc_stream_shuffle(dst=tb, src=tv, shuffle_mask=[(i // SNAKE) * SNAKE for i in range(32)])
    tau = _sbuf(alloc, (P, 1), nl.float32, "pt_tau")
    nisa.tensor_scalar(dst=tau, data=tb[:, 0:1], op0=nl.multiply, operand0=-1.0, engine=vec)
    return tau


def select_topk_payload(scores, codes, n: int, k: int, alloc=None):
    """The top ``k`` of ``n`` candidates per query, each winner's payload code carried through.

    Args:
        scores: [P, n // 16] bf16 SBUF, the candidates in the snake layout (P a multiple of 16, one query per 16).
        codes: [P, n // 16] uint16 SBUF, each candidate's payload ``x`` (``0 .. CODE_LIMIT - 1``) as its bf16 bit
            pattern (:func:`encode_codes`, :func:`code_of`), same layout.
        n, k: candidates and winners per query: ``n`` below 65,536, ``k`` a multiple of 16, at most ``n`` and 4,096.
        alloc: ``alloc(shape, dtype, name)`` -> an SBUF tile, for a kernel that places its own SBUF tiles (the prefill's
            SbufManager); None allocates them with ``nl.ndarray``. The two PSUM tiles always come from ``nl.ndarray``.

    Returns:
        (code_out [P, k] uint16 SBUF, idx_out [P, k] uint32 SBUF): result ``j`` of a query at partition ``j % 16``,
        column ``j // 16`` (columns ``0 .. k/16 - 1`` hold the results): its bit pattern (:func:`decode_codes` gives
        ``x`` back) and its snake index ``e``. Ties at the k-th score are taken in partition-major order, so the
        winners are a valid top ``k``.
    """
    P, cols = scores.shape
    k_cols = k // SNAKE
    _require(cols * SNAKE == n and n < 65536,
             f"{cols} columns are not {n} candidates, or {n} is more than one nisa.topk ranks")
    # a partition's count above tau (at most k/16) goes through bf16, which holds the integers to 256 exactly
    _require(k % SNAKE == 0 and 0 < k <= min(n, 256 * SNAKE),
             f"k = {k} must be a multiple of {SNAKE}, at most n = {n} and at most {256 * SNAKE}")
    vec = nisa.engine.vector
    f32 = nl.float32

    # 1. tau
    val1 = _sbuf(alloc, (P, k), nl.bfloat16, "pt_val1")
    idx1 = _sbuf(alloc, (P, k), nl.uint32, "pt_idx1")
    nisa.topk(val_dst=val1[0:P, 0:k], idx_dst=idx1[0:P, 0:k], src=scores, n=n)
    same, lower = _group_consts(P, alloc)
    tau = _query_tau(val1, k_cols, P, alloc)

    # 2. membership in place: score > tau, plus the first (k - #above) ties in partition-major order. Every score above
    # tau is among the k winners, so #above is counted on the winners' tile, a quarter of the candidates. The group
    # sums are bf16 matmuls on the tie and above counts, exact: every count operand is at most 256 (bf16 holds the
    # integers to 256 exactly; #above <= k/16, and a partition's tie count, at most its `cols`, goes in parts of 256),
    # and the sums are fp32.
    u16, bf16 = nl.uint16, nl.bfloat16
    gt1 = _sbuf(alloc, (P, k_cols), bf16, "pt_gt1")
    n_gt = _sbuf(alloc, (P, 1), f32, "pt_n_gt")
    nisa.tensor_scalar_reduce(dst=gt1, data=val1[0:P, 0:k_cols], op0=nl.greater, operand0=tau, reduce_op=nl.add,
                              reduce_res=n_gt)
    eq = _sbuf(alloc, (P, cols), bf16, "pt_eq")
    n_eq = _sbuf(alloc, (P, 1), f32, "pt_n_eq")
    nisa.tensor_scalar_reduce(dst=eq, data=scores, op0=nl.equal, operand0=tau, reduce_op=nl.add, reduce_res=n_eq)
    n_parts = -(-cols // 256)
    cnt = _sbuf(alloc, (P, 1 + n_parts), bf16, "pt_cnt")   # [#above (<= k/16), the tie count in parts of <= 256]
    nisa.tensor_copy(dst=cnt[:, 0:1], src=n_gt, engine=vec)
    part = _sbuf(alloc, (P, 1), f32, "pt_part")
    for i in range(n_parts):
        # part i = min(max(n_eq - 256 i, 0), 256)
        nisa.tensor_scalar(dst=part, data=n_eq, op0=nl.subtract, operand0=256.0 * i, op1=nl.maximum, operand1=0.0,
                           engine=vec)
        nisa.tensor_scalar(dst=cnt[:, 1 + i : 2 + i], data=part, op0=nl.minimum, operand0=256.0, engine=vec)
    above = nl.ndarray((P, 1), dtype=f32, buffer=nl.psum)
    nisa.nc_matmul(dst=above, stationary=same, moving=cnt[:, 0:1])
    before = nl.ndarray((P, n_parts), dtype=f32, buffer=nl.psum)
    nisa.nc_matmul(dst=before, stationary=lower, moving=cnt[:, 1 : 1 + n_parts])
    # a tie is in when its rank among the query's ties (ties on lower partitions + its inclusive rank in its row)
    # is at most k - above, i.e. its inclusive row rank < k + 1 - above - before
    room = _sbuf(alloc, (P, 1), f32, "pt_room")
    nisa.tensor_scalar(dst=room, data=above, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=float(k + 1),
                       engine=vec)
    if n_parts == 1:
        nisa.tensor_tensor(dst=room, data1=room, data2=before[:, 0:1], op=nl.subtract, engine=vec)
    else:
        n_before = _sbuf(alloc, (P, 1), f32, "pt_n_before")
        nisa.tensor_reduce(dst=n_before, op=nl.add, data=before[:, 0:n_parts], axis=1)
        nisa.tensor_tensor(dst=room, data1=room, data2=n_before, op=nl.subtract, engine=vec)
    rank = _sbuf(alloc, (P, cols), u16, "pt_rank")
    nisa.tensor_scalar_cumulative(dst=rank, src=eq, op0=nl.add, op1=nl.add, imm0=0.0)
    tie_in = _sbuf(alloc, (P, cols), u16, "pt_tie_in")
    nisa.scalar_tensor_tensor(dst=tie_in, data=rank, op0=nl.less, operand0=room, op1=nl.multiply, operand1=eq)
    member = _sbuf(alloc, (P, cols), u16, "pt_member")
    nisa.scalar_tensor_tensor(dst=member, data=scores, op0=nl.greater, operand0=tau, op1=nl.add, operand1=tie_in)

    # 3. the members' codes as bf16 bit patterns, everything else below them; the second select returns the codes
    tile2 = _sbuf(alloc, (P, cols), nl.uint16, "pt_tile2")
    nisa.memset(dst=tile2, value=_NON_MEMBER, engine=vec)
    nisa.tensor_copy_predicated(dst=tile2, src=codes, predicate=member)
    val2 = _sbuf(alloc, (P, k), nl.bfloat16, "pt_val2")
    idx2 = _sbuf(alloc, (P, k), nl.uint32, "pt_idx2")
    nisa.topk(val_dst=val2[0:P, 0:k], idx_dst=idx2[0:P, 0:k], src=tile2.view(nl.bfloat16), n=n)
    return val2.view(nl.uint16), idx2


def encode_codes(x, dst_codes, P: int, cols: int, alloc=None):
    """Payload ``x`` ([P, cols] int32, ``0 .. CODE_LIMIT - 1``) -> its bf16 bit pattern code ([P, cols] uint16)."""
    vec = nisa.engine.vector
    hi = _sbuf(alloc, (P, cols), nl.int32, "pt_hi")
    nisa.tensor_scalar(dst=hi, data=x, op0=nl.greater_equal, operand0=_NEG_FROM, op1=nl.multiply, operand1=256,
                       engine=vec)
    nisa.scalar_tensor_tensor(dst=dst_codes, data=x, op0=nl.add, operand0=128, op1=nl.add, operand1=hi)


def decode_codes(code, dst_x, P: int, cols: int, alloc=None):
    """Inverse of :func:`encode_codes`: code ([P, cols] uint16) -> payload ([P, cols] int32)."""
    vec = nisa.engine.vector
    b = _sbuf(alloc, (P, cols), nl.int32, "pt_b")
    nisa.tensor_copy(dst=b, src=code, engine=vec)
    hi = _sbuf(alloc, (P, cols), nl.int32, "pt_hi")
    nisa.tensor_scalar(dst=hi, data=b, op0=nl.greater_equal, operand0=0x8000, op1=nl.multiply, operand1=256,
                       engine=vec)
    nisa.scalar_tensor_tensor(dst=dst_x, data=b, op0=nl.subtract, operand0=128, op1=nl.subtract, operand1=hi)


DEAD_KEY = 1 << 15
"""The CP-deal payload of a dead candidate: one past every shard-local key (a shard holds at most 32768 keys)."""


def encode_local_codes(v, vals, dst_codes, P: int, cols: int, live_floor: float, alloc=None):
    """The local stage's half of the CP-deal carry: each winner's payload code from its shard-local key ``v``
    ([P, cols] int32), ``DEAD_KEY``'s code where its score ``vals`` ([P, cols] bf16) is at or below ``live_floor``.
    Writes ``dst_codes`` ([P, cols] uint16)."""
    vec = nisa.engine.vector
    encode_codes(v, dst_codes, P, cols, alloc)
    dead = _sbuf(alloc, (P, cols), nl.uint16, "pl_dead")
    nisa.tensor_scalar(dst=dead, data=vals, op0=nl.less_equal, operand0=live_floor, engine=vec)
    nisa.tensor_copy_predicated(dst=dst_codes, src=code_of(DEAD_KEY), predicate=dead)


def select_cp_deal(scores, codes, n_cand: int, topk: int, *, block_size: int, shard_degree: int, alloc=None):
    """The decode indexer's packed CP merge select, in NKI: each query's top ``topk`` of its ``shard_degree`` key
    shards' candidates, as gated global positions.

    ``scores`` / ``codes`` ([P, shard_degree * topk/16] bf16 / uint16, the snake layout, shard ``s`` in column block
    ``s``) are the gathered local winners and the payload codes the local stage wrote (:func:`encode_local_codes`). A
    winner's position is ``v + (shard_degree - 1) * block_size * (v // block_size) + s * block_size`` for its
    shard-local key ``v`` and its shard ``s``, the column block of its snake index.

    Returns:
        [P, topk/16] int32 SBUF: the positions in the snake layout (result ``j`` of a query at partition ``j % 16``,
        column ``j // 16``), -1 for a dead candidate.
    """
    P, cols = scores.shape
    k_cols = topk // SNAKE
    vec = nisa.engine.vector
    i32 = nl.int32
    _require(block_size > 0 and block_size & (block_size - 1) == 0 and k_cols & (k_cols - 1) == 0,
             f"block_size {block_size} and topk / 16 = {k_cols} must be powers of two")
    _require(cols == shard_degree * k_cols and cols * SNAKE == n_cand,
             f"{cols} columns are not {shard_degree} shards of {k_cols}, or not {n_cand} candidates")
    shift_lo = _log2(block_size)
    col_shift = _log2(SNAKE * k_cols)  # e >> log2(16 * k_cols) = column // k_cols = s
    # the block term w = (v >> shift_lo) * (shard_degree - 1) + s of a live key (v < DEAD_KEY) is on a uint16 tile
    w_max = ((DEAD_KEY - 1) >> shift_lo) * (shard_degree - 1) + shard_degree - 1
    _require(w_max < 1 << 16, f"block_size {block_size} at shard_degree {shard_degree}: a live key's block term "
             f"reaches {w_max}, past the 16 bits the deal decodes on")

    code_out, idx_out = select_topk_payload(scores, codes, n_cand, topk, alloc)

    # v back from the code, then position = block_size * w + v, all on 16-bit tiles but the position (v <= DEAD_KEY,
    # a live key's w <= w_max, a dead one's position replaced by -1): no widening copy. Bit shifts and arithmetic do
    # not share one instruction.
    u16 = nl.uint16
    c16 = code_out[0:P, 0:k_cols]
    hi = _sbuf(alloc, (P, k_cols), u16, "pd_hi")
    nisa.tensor_scalar(dst=hi, data=c16, op0=nl.greater_equal, operand0=0x8000, op1=nl.multiply, operand1=256,
                       engine=vec)
    v = _sbuf(alloc, (P, k_cols), u16, "pd_v")
    nisa.scalar_tensor_tensor(dst=v, data=c16, op0=nl.subtract, operand0=128, op1=nl.subtract, operand1=hi)
    s_t = _sbuf(alloc, (P, k_cols), nl.uint32, "pd_s")  # a bit shift keeps its operand's dtype
    nisa.tensor_scalar(dst=s_t, data=idx_out[0:P, 0:k_cols], op0=nl.right_shift, operand0=col_shift, engine=vec)
    h = _sbuf(alloc, (P, k_cols), u16, "pd_h")
    nisa.tensor_scalar(dst=h, data=v, op0=nl.right_shift, operand0=shift_lo, engine=vec)
    w = _sbuf(alloc, (P, k_cols), u16, "pd_w")
    nisa.scalar_tensor_tensor(dst=w, data=h, op0=nl.multiply, operand0=shard_degree - 1, op1=nl.add, operand1=s_t)
    pos = _sbuf(alloc, (P, k_cols), i32, "pd_pos")
    nisa.scalar_tensor_tensor(dst=pos, data=w, op0=nl.multiply, operand0=block_size, op1=nl.add, operand1=v)
    dead = _sbuf(alloc, (P, k_cols), u16, "pd_dead")
    nisa.tensor_scalar(dst=dead, data=v, op0=nl.equal, operand0=DEAD_KEY, engine=vec)
    nisa.tensor_copy_predicated(dst=pos, src=-1, predicate=dead)
    return pos


def emit_select_nki(scores, pos, out_pos, n_cols: int, k: int, block_bases, block_widths, alloc=None):
    """The GLM-5.2 prefill chunk merge's select (``glm_topk_merge.merge_batch``): each query's top
    ``k`` candidates' global key positions, ``k / 16`` per partition.

    The merge's candidates are ``len(block_bases)`` chunks of ``k`` (stage 1's per-chunk winners), chunk ``c`` in
    columns ``[c k/16, (c + 1) k/16)``, its positions ``block_bases[c] .. block_bases[c] + block_widths[c] - 1``. The
    select carries a candidate's position less its chunk's base as the payload (a chunk is at most ``CODE_LIMIT`` keys
    wide; the served geometries are at most 61,440) and adds the base of the winner's chunk back, the chunk being the
    winner's column block: ``e >> log2(k)`` of its snake index ``e``.

    Args:
        scores: [P, >= n_cols] bf16 SBUF, the candidates' scores in the snake layout.
        pos: [P, >= n_cols] fp32 SBUF, their global positions, same layout (a dead candidate's may be anything).
        out_pos: [P, k / 16] int32 SBUF: partition ``16 j + q`` gets ``k / 16`` of query ``j``'s positions, so 8 flat
            ``[k]`` rows reshaped to ``[128, k / 16]`` is the destination.
        n_cols: candidate columns per partition, ``len(block_bases) * k / 16`` (below 4,096: ``n < 65,536``).
        block_bases, block_widths: each chunk's first position and width, trace-time ints.
        alloc: as in :func:`select_topk_payload`.

    A query needs at least ``k`` candidates above -inf for its picks to be scores only; the merge always has them
    (chunk 0 holds ``k`` finite ones).
    """
    P = scores.shape[0]
    k_cols = k // SNAKE
    n_blk = len(block_bases)
    vec = nisa.engine.vector
    i32 = nl.int32
    _require(k & (k - 1) == 0, f"k = {k} must be a power of two: a winner's chunk is its snake index >> log2(k)")
    _require(n_cols == n_blk * k_cols, f"{n_cols} columns are not {n_blk} chunks of {k_cols}")
    _require(SNAKE * n_cols < 65536, f"{SNAKE * n_cols} candidates is more than one nisa.topk ranks")
    _require(max(block_widths) <= CODE_LIMIT,
             f"a chunk of {max(block_widths)} keys does not fit the {CODE_LIMIT} payload codes")

    # payload: the chunk-local position, clamped into the code range (a dead candidate never becomes a member)
    x = _sbuf(alloc, (P, n_cols), i32, "pn_x")
    for c, b in enumerate(block_bases):
        blk = slice(c * k_cols, (c + 1) * k_cols)
        nisa.tensor_scalar(dst=x[:, blk], data=pos[:, blk], op0=nl.subtract, operand0=float(b), op1=nl.maximum,
                           operand1=0.0, engine=vec)
    nisa.tensor_scalar(dst=x, data=x, op0=nl.minimum, operand0=CODE_LIMIT - 1, engine=vec)
    codes = _sbuf(alloc, (P, n_cols), nl.uint16, "pn_codes")
    encode_codes(x, codes, P, n_cols, alloc)

    code_out, idx_out = select_topk_payload(scores[:, 0:n_cols], codes, SNAKE * n_cols, k, alloc)

    # position = payload + the base of the winner's chunk, chunk = e >> log2(k) (16 snake partitions x k/16 columns)
    decode_codes(code_out[0:P, 0:k_cols], out_pos[0:P, 0:k_cols], P, k_cols, alloc)
    if n_blk == 1:
        if block_bases[0]:
            nisa.tensor_scalar(dst=out_pos, data=out_pos, op0=nl.add, operand0=block_bases[0], engine=vec)
        return
    ch = _sbuf(alloc, (P, k_cols), i32, "pn_ch")
    nisa.tensor_scalar(dst=ch, data=idx_out[0:P, 0:k_cols].view(i32), op0=nl.right_shift,
                       operand0=k.bit_length() - 1, engine=vec)
    steps = [block_bases[c] - block_bases[c - 1] for c in range(1, n_blk)]
    if block_bases[0] == 0 and len(set(steps)) == 1:
        nisa.scalar_tensor_tensor(dst=out_pos, data=ch, op0=nl.multiply, operand0=steps[0], op1=nl.add,
                                  operand1=out_pos)
        return
    # base(c) = base[0] + sum over c' >= 1 of [c >= c'] (base[c'] - base[c' - 1])
    step_c = _sbuf(alloc, (P, k_cols), i32, "pn_step")
    for c in range(1, n_blk):
        nisa.tensor_scalar(dst=step_c, data=ch, op0=nl.greater_equal, operand0=c, op1=nl.multiply,
                           operand1=steps[c - 1], engine=vec)
        nisa.tensor_tensor(dst=out_pos, data1=out_pos, data2=step_c, op=nl.add, engine=vec)
    if block_bases[0]:
        nisa.tensor_scalar(dst=out_pos, data=out_pos, op0=nl.add, operand0=block_bases[0], engine=vec)


# Shards wider than one select (more than 32,768 keys a rank, as at the 1M point).
#
# Above 32,768 keys a rank the indexer's local stage ranks the shard in chunks (phase 1: a builtin nisa.topk of each
# 32,768-key chunk), and both later selects must carry a shard-local key of up to 18 bits (262,144 keys at 1M), more
# than one bf16 code holds. So the payload is split: a select carries part of the key as its code and the winner's
# place in the tile (its snake index) carries the rest, and the two merges run in two rounds:
#
# 1. each rank ranks its phase-1 candidates (wide_local_values) and the key group all-gathers those top-k scores;
# 2. every rank finds the global cut from them (wide_global_cut) and selects, among its own candidates, exactly its
#    share of the global top k (wide_members): above the cut, and its quota of the ties at it, shard by shard;
# 3. it compacts those members with their in-chunk key as the code, the chunk being the winner's column block
#    (wide_publish), and publishes their global positions, -1 elsewhere;
# 4. the merge, on the gathered lists, compacts the k live entries with the shard-local key in 5 column blocks per
#    shard, code = key // 4 in blocks 0..3 by key % 4 and key - OVERFLOW_FROM above (merge_wide).
#
# Exact: the k selected keys are a top k of the sequence (ties at the cut resolve shard-major, partition-major).

#: Keys from here on take the overflow block of merge_wide: key // 4 would reach CODE_LIMIT.
OVERFLOW_FROM = 4 * CODE_LIMIT


def shard_local_key(pos, dst, P: int, cols: int, shard_degree: int, block_size: int, alloc=None):
    """The shard-local key ``v`` of each global position under the interleaved deal (``deal_positions``' inverse):
    ``v = ((p >> log2(deg * bs)) << log2(bs)) | (p & (bs - 1))``, the rank term ``rank * bs`` dropping out.
    ``pos`` / ``dst``: [P, cols] int32 (``dst`` may be ``pos``'s own tile only if ``pos`` is not read again)."""
    hi = _sbuf(alloc, (P, cols), nl.int32, "pw_hi")
    nisa.tensor_scalar(dst=hi, data=pos, op0=nl.right_shift, operand0=_log2(shard_degree * block_size))
    nisa.tensor_scalar(dst=hi, data=hi, op0=nl.left_shift, operand0=_log2(block_size))
    lo = _sbuf(alloc, (P, cols), nl.int32, "pw_lo")
    nisa.tensor_scalar(dst=lo, data=pos, op0=nl.bitwise_and, operand0=block_size - 1)
    nisa.tensor_tensor(dst=dst, data1=hi, data2=lo, op=nl.bitwise_or)


def deal_global(v, dst, P: int, cols: int, shard_degree: int, block_size: int, rank_term, alloc=None):
    """``deal_positions`` plus the rank term: ``p = deg * bs * (v >> log2(bs)) + (v & (bs - 1)) + rank_term``, with
    ``rank_term`` a [P, 1] fp32 tile or a [P, cols] int32 tile. ``v`` / ``dst``: [P, cols] int32."""
    blk = _sbuf(alloc, (P, cols), nl.int32, "pd_blk")
    nisa.tensor_scalar(dst=blk, data=v, op0=nl.right_shift, operand0=_log2(block_size))
    intra = _sbuf(alloc, (P, cols), nl.int32, "pd_intra")
    nisa.tensor_scalar(dst=intra, data=v, op0=nl.bitwise_and, operand0=block_size - 1)
    nisa.scalar_tensor_tensor(dst=dst, data=blk, op0=nl.multiply, operand0=float(shard_degree * block_size),
                              op1=nl.add, operand1=intra)
    if rank_term.shape[1] == 1:
        nisa.tensor_scalar(dst=dst, data=dst, op0=nl.add, operand0=rank_term, engine=nisa.engine.vector)
    else:
        nisa.tensor_tensor(dst=dst, data1=dst, data2=rank_term, op=nl.add, engine=nisa.engine.vector)


def wide_local_values(scores, n: int, k: int, alloc=None):
    """Round 1: this rank's top ``k`` scores of its ``n`` phase-1 candidates (any order), [P, k/16] bf16: what the key
    group all-gathers for wide_global_cut."""
    P = scores.shape[0]
    val = _sbuf(alloc, (P, k), nl.bfloat16, "pw_val1")
    idx = _sbuf(alloc, (P, k), nl.uint32, "pw_idx1")
    nisa.topk(val_dst=val[0:P, 0:k], idx_dst=idx[0:P, 0:k], src=scores, n=n)
    return val[0:P, 0 : k // SNAKE]


def _group_counts(mask, P: int, same, name: str, alloc=None):
    """Each query's count of the set entries of ``mask`` ([P, cols] float 0/1, at most 256 a partition), on every one of
    its 16 partitions: [P, 1] fp32 (a bf16 matmul against ``same``, exact)."""
    f32 = nl.float32
    part = _sbuf(alloc, (P, 1), f32, name + "_p")
    nisa.tensor_reduce(dst=part, op=nl.add, data=mask, axis=1)
    part_b = _sbuf(alloc, (P, 1), nl.bfloat16, name + "_b")
    nisa.tensor_copy(dst=part_b, src=part, engine=nisa.engine.vector)
    total = nl.ndarray((P, 1), dtype=f32, buffer=nl.psum)
    nisa.nc_matmul(dst=total, stationary=same, moving=part_b)
    out = _sbuf(alloc, (P, 1), f32, name)
    nisa.tensor_copy(dst=out, src=total, engine=nisa.engine.vector)
    return out


def wide_global_cut(gathered, k: int, degree: int, own_shard, alloc=None):
    """Round 2a: the global cut and this rank's share of its ties, from the key group's round-1 values.

    Args:
        gathered: [P, degree * k/16] bf16, shard ``s``'s top-``k`` scores in column block ``s``.
        k: winners per query, a multiple of 16 and at most 4,096.
        degree: key shards in the group.
        own_shard: [P, 1] fp32, this rank's shard index (its key rank).
        alloc: as in :func:`select_topk_payload`.

    Returns:
        (tau [P, 1] fp32, the k-th largest score of the union; room [P, 1] fp32, how many of this rank's candidates
        equal to tau are in the top k: ``k - #above(all) - #ties(shards before this one)``, clamped to
        ``[0, #ties(own)]``, so the ties at the cut go shard-major.)
    """
    P, cols = gathered.shape
    k_cols = k // SNAKE
    # each shard's per-partition counts (at most k/16) go through bf16, which holds the integers to 256 exactly
    _require(k % SNAKE == 0 and 0 < k <= 256 * SNAKE and cols == degree * k_cols,
             f"{cols} columns are not {degree} shards of k / 16 = {k_cols}, or k = {k} is not a multiple of {SNAKE} "
             f"of at most {256 * SNAKE}")
    vec = nisa.engine.vector
    f32, bf16 = nl.float32, nl.bfloat16
    val = _sbuf(alloc, (P, k), bf16, "pc_val")
    idx = _sbuf(alloc, (P, k), nl.uint32, "pc_idx")
    nisa.topk(val_dst=val[0:P, 0:k], idx_dst=idx[0:P, 0:k], src=gathered, n=degree * k)
    tau = _query_tau(val, k_cols, P, alloc)
    same, _ = _group_consts(P, alloc)
    above_all = _sbuf(alloc, (P, 1), f32, "pc_above")
    nisa.memset(dst=above_all, value=0.0, engine=vec)
    ties_before = _sbuf(alloc, (P, 1), f32, "pc_before")
    nisa.memset(dst=ties_before, value=0.0, engine=vec)
    ties_own = _sbuf(alloc, (P, 1), f32, "pc_own")
    nisa.memset(dst=ties_own, value=0.0, engine=vec)
    m = _sbuf(alloc, (P, k_cols), bf16, "pc_m")
    sel = _sbuf(alloc, (P, 1), f32, "pc_sel")
    for s in range(degree):
        blk = gathered[:, s * k_cols : (s + 1) * k_cols]
        nisa.tensor_scalar(dst=m, data=blk, op0=nl.greater, operand0=tau, engine=vec)
        gt = _group_counts(m, P, same, f"pc_gt{s}", alloc)
        nisa.tensor_tensor(dst=above_all, data1=above_all, data2=gt, op=nl.add, engine=vec)
        nisa.tensor_scalar(dst=m, data=blk, op0=nl.equal, operand0=tau, engine=vec)
        eq = _group_counts(m, P, same, f"pc_eq{s}", alloc)
        # ties of the shards before this one: eq * (s < own); this one's: eq * (s == own)
        nisa.tensor_scalar(dst=sel, data=own_shard, op0=nl.greater, operand0=float(s), engine=vec)
        nisa.scalar_tensor_tensor(dst=ties_before, data=eq, op0=nl.multiply, operand0=sel, op1=nl.add,
                                  operand1=ties_before)
        nisa.tensor_scalar(dst=sel, data=own_shard, op0=nl.equal, operand0=float(s), engine=vec)
        nisa.scalar_tensor_tensor(dst=ties_own, data=eq, op0=nl.multiply, operand0=sel, op1=nl.add,
                                  operand1=ties_own)
    room = _sbuf(alloc, (P, 1), f32, "pc_room")
    # k - above_all - ties_before, clamped to [0, ties_own]
    nisa.tensor_tensor(dst=room, data1=above_all, data2=ties_before, op=nl.add, engine=vec)
    nisa.tensor_scalar(dst=room, data=room, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=float(k),
                       engine=vec)
    nisa.tensor_scalar(dst=room, data=room, op0=nl.maximum, operand0=0.0, engine=vec)
    nisa.tensor_tensor(dst=room, data1=room, data2=ties_own, op=nl.minimum, engine=vec)
    return tau, room


def wide_members(scores, tau, room, alloc=None):
    """Round 2b: this rank's share of the global top k among its candidates ([P, cols] bf16): every score above
    ``tau`` and the first ``room`` of those equal to it in partition-major order. [P, cols] uint16 0/1."""
    P, cols = scores.shape
    vec = nisa.engine.vector
    f32, bf16, u16 = nl.float32, nl.bfloat16, nl.uint16
    _, lower = _group_consts(P, alloc)
    eq = _sbuf(alloc, (P, cols), bf16, "pm_eq")
    n_eq = _sbuf(alloc, (P, 1), f32, "pm_n_eq")
    nisa.tensor_scalar_reduce(dst=eq, data=scores, op0=nl.equal, operand0=tau, reduce_op=nl.add, reduce_res=n_eq)
    n_parts = -(-cols // 256)
    cnt = _sbuf(alloc, (P, n_parts), bf16, "pm_cnt")
    part = _sbuf(alloc, (P, 1), f32, "pm_part")
    for i in range(n_parts):
        nisa.tensor_scalar(dst=part, data=n_eq, op0=nl.subtract, operand0=256.0 * i, op1=nl.maximum, operand1=0.0,
                           engine=vec)
        nisa.tensor_scalar(dst=cnt[:, i : i + 1], data=part, op0=nl.minimum, operand0=256.0, engine=vec)
    before = nl.ndarray((P, n_parts), dtype=f32, buffer=nl.psum)
    nisa.nc_matmul(dst=before, stationary=lower, moving=cnt[:, 0:n_parts])
    # a tie is in when its rank among this rank's ties (ties on lower partitions + its inclusive rank in the row) is at
    # most room, i.e. its inclusive row rank < room + 1 - before
    lim = _sbuf(alloc, (P, 1), f32, "pm_lim")
    nisa.tensor_scalar(dst=lim, data=room, op0=nl.add, operand0=1.0, engine=vec)
    n_before = _sbuf(alloc, (P, 1), f32, "pm_n_before")
    nisa.tensor_reduce(dst=n_before, op=nl.add, data=before[:, 0:n_parts], axis=1)
    nisa.tensor_tensor(dst=lim, data1=lim, data2=n_before, op=nl.subtract, engine=vec)
    rank = _sbuf(alloc, (P, cols), u16, "pm_rank")
    nisa.tensor_scalar_cumulative(dst=rank, src=eq, op0=nl.add, op1=nl.add, imm0=0.0)
    tie_in = _sbuf(alloc, (P, cols), u16, "pm_tie_in")
    nisa.scalar_tensor_tensor(dst=tie_in, data=rank, op0=nl.less, operand0=lim, op1=nl.multiply, operand1=eq)
    member = _sbuf(alloc, (P, cols), u16, "pm_member")
    nisa.scalar_tensor_tensor(dst=member, data=scores, op0=nl.greater, operand0=tau, op1=nl.add, operand1=tie_in)
    return member


def wide_publish(scores, pos, member, n: int, k: int, chunks: int, chunk_keys: int, *, shard_degree: int,
                 block_size: int, rank_term, live_floor: float, alloc=None):
    """Round 2c: this rank's members' global positions, compacted into k slots (the snake layout, -1 for an empty or
    dead slot), [P, k/16] int32.

    The candidates are phase 1's, chunk ``c`` in snake partitions ``[c * 16/chunks, (c + 1) * 16/chunks)`` of each
    query (the blocked layout ``load_snake`` reads), so a winner's chunk is its snake index's partition over
    ``16 / chunks``; the code carries its key in the chunk (``chunk_keys`` wide, a power of two of at most 32,768).
    The shard, ``chunks * chunk_keys`` keys, is at most what merge_wide carries (``OVERFLOW_FROM + CODE_LIMIT``).
    ``pos``: [P, cols] int32 global positions (phase 1's, the rank term folded); ``rank_term``: [P, 1] fp32, this
    rank's ``rank * block_size``; a candidate scored at or below ``live_floor`` is dead.
    """
    P, cols = scores.shape
    k_cols = k // SNAKE
    vec = nisa.engine.vector
    i32, u16 = nl.int32, nl.uint16
    _require(chunk_keys <= DEAD_KEY, f"a chunk of {chunk_keys} keys is wider than the {DEAD_KEY} codes below DEAD_KEY")
    # the in-chunk key is the shard key's low bits (a mask of the width), so the width is a power of two
    _require(chunk_keys & (chunk_keys - 1) == 0, f"a chunk of {chunk_keys} keys: the wide select carries each "
             "winner's in-chunk key as the shard key masked by the chunk width, which takes a power-of-two width")
    _require(chunks * chunk_keys <= OVERFLOW_FROM + CODE_LIMIT, f"a shard of {chunks * chunk_keys} keys is wider "
             f"than the {OVERFLOW_FROM + CODE_LIMIT} keys merge_wide's five column blocks carry")
    _require(SNAKE % chunks == 0 and cols * SNAKE == n and n == chunks * k,
             f"{chunks} chunks of {k} candidates are not the {n} candidates of {cols} columns")
    # the in-chunk key of each candidate as its payload, DEAD_KEY's where its score is dead
    v = _sbuf(alloc, (P, cols), i32, "pp_v")
    shard_local_key(pos, v, P, cols, shard_degree, block_size, alloc)
    nisa.tensor_scalar(dst=v, data=v, op0=nl.bitwise_and, operand0=chunk_keys - 1)
    codes = _sbuf(alloc, (P, cols), u16, "pp_codes")
    encode_codes(v, codes, P, cols, alloc)
    dead = _sbuf(alloc, (P, cols), u16, "pp_dead")
    nisa.tensor_scalar(dst=dead, data=scores, op0=nl.less_equal, operand0=live_floor, engine=vec)
    nisa.tensor_copy_predicated(dst=codes, src=code_of(DEAD_KEY), predicate=dead)
    tile2 = _sbuf(alloc, (P, cols), u16, "pp_tile2")
    nisa.memset(dst=tile2, value=_NON_MEMBER, engine=vec)
    nisa.tensor_copy_predicated(dst=tile2, src=codes, predicate=member)
    val2 = _sbuf(alloc, (P, k), nl.bfloat16, "pp_val2")
    idx2 = _sbuf(alloc, (P, k), nl.uint32, "pp_idx2")
    nisa.topk(val_dst=val2[0:P, 0:k], idx_dst=idx2[0:P, 0:k], src=tile2.view(nl.bfloat16), n=n)
    c16 = val2.view(u16)[0:P, 0:k_cols]
    x = _sbuf(alloc, (P, k_cols), i32, "pp_x")
    decode_codes(c16, x, P, k_cols, alloc)
    # chunk = (e & 15) >> log2(16 / chunks): its snake partition's chunk block
    ch = _sbuf(alloc, (P, k_cols), nl.uint32, "pp_ch")
    nisa.tensor_scalar(dst=ch, data=idx2[0:P, 0:k_cols], op0=nl.bitwise_and, operand0=SNAKE - 1)
    nisa.tensor_scalar(dst=ch, data=ch, op0=nl.right_shift, operand0=_log2(SNAKE // chunks))
    lv = _sbuf(alloc, (P, k_cols), i32, "pp_lv")
    nisa.scalar_tensor_tensor(dst=lv, data=ch, op0=nl.multiply, operand0=float(chunk_keys), op1=nl.add, operand1=x)
    out = _sbuf(alloc, (P, k_cols), i32, "pp_out")
    deal_global(lv, out, P, k_cols, shard_degree, block_size, rank_term, alloc)
    # empty: a dead key's code or a non-member's pattern (which decodes to CODE_LIMIT), both from DEAD_KEY on
    bad = _sbuf(alloc, (P, k_cols), u16, "pp_bad")
    nisa.tensor_scalar(dst=bad, data=x, op0=nl.greater_equal, operand0=DEAD_KEY, engine=vec)
    nisa.tensor_copy_predicated(dst=out, src=-1, predicate=bad)
    return out


def merge_wide(pos, k: int, degree: int, block_size: int, alloc=None):
    """Round 3: the global top k from the key group's published lists ([P, degree * k/16] int32 global positions,
    shard ``s`` in column block ``s``, -1 for an empty slot; exactly k live entries a query when the sequence has k
    live keys): [P, k/16] int32 in the snake layout, -1 for an empty slot.

    The live entries are compacted by one nisa.topk over 5 column blocks a shard: shard-local key ``v`` below
    OVERFLOW_FROM in block ``b * degree + s`` for ``b = v % 4`` with code ``v // 4``, from OVERFLOW_FROM on in block
    ``4 * degree + s`` with code ``v - OVERFLOW_FROM``; the winner's block gives ``s`` and ``b`` back.
    """
    P, cols = pos.shape
    k_cols = k // SNAKE
    vec = nisa.engine.vector
    i32, u16 = nl.int32, nl.uint16
    _require(cols == degree * k_cols and degree & (degree - 1) == 0,
             f"{cols} columns are not {degree} shards of {k_cols}, or {degree} is not a power of two")
    n_blocks = 5 * degree
    n = n_blocks * k
    _require(n < 65536, f"{n} candidates is more than one nisa.topk ranks")
    v = _sbuf(alloc, (P, cols), i32, "pg_v")
    shard_local_key(pos, v, P, cols, degree, block_size, alloc)
    live = _sbuf(alloc, (P, cols), u16, "pg_live")
    nisa.tensor_scalar(dst=live, data=pos, op0=nl.greater_equal, operand0=0, engine=vec)
    low = _sbuf(alloc, (P, cols), u16, "pg_low")      # live and below the overflow
    nisa.tensor_scalar(dst=low, data=v, op0=nl.less, operand0=OVERFLOW_FROM, engine=vec)
    nisa.tensor_tensor(dst=low, data1=low, data2=live, op=nl.multiply, engine=vec)
    over = _sbuf(alloc, (P, cols), u16, "pg_over")    # live and at or above it
    nisa.tensor_tensor(dst=over, data1=live, data2=low, op=nl.subtract, engine=vec)
    q = _sbuf(alloc, (P, cols), i32, "pg_q")
    nisa.tensor_scalar(dst=q, data=v, op0=nl.right_shift, operand0=2)
    code_q = _sbuf(alloc, (P, cols), u16, "pg_code_q")
    encode_codes(q, code_q, P, cols, alloc)
    r = _sbuf(alloc, (P, cols), i32, "pg_r")
    nisa.tensor_scalar(dst=r, data=v, op0=nl.bitwise_and, operand0=3)
    ov = _sbuf(alloc, (P, cols), i32, "pg_ov")
    # ov in the code range for every entry (a live key's already is: wide_publish refuses a wider shard)
    nisa.tensor_scalar(dst=ov, data=v, op0=nl.subtract, operand0=OVERFLOW_FROM, op1=nl.maximum, operand1=0,
                       engine=vec)
    nisa.tensor_scalar(dst=ov, data=ov, op0=nl.minimum, operand0=CODE_LIMIT - 1, engine=vec)
    code_ov = _sbuf(alloc, (P, cols), u16, "pg_code_ov")
    encode_codes(ov, code_ov, P, cols, alloc)
    tile = _sbuf(alloc, (P, n_blocks * k_cols), u16, "pg_tile")
    nisa.memset(dst=tile, value=_NON_MEMBER, engine=vec)
    sel = _sbuf(alloc, (P, k_cols), u16, "pg_sel")
    for s in range(degree):
        src = slice(s * k_cols, (s + 1) * k_cols)
        for b in range(4):
            dst = slice((b * degree + s) * k_cols, (b * degree + s + 1) * k_cols)
            nisa.tensor_scalar(dst=sel, data=r[:, src], op0=nl.equal, operand0=b, engine=vec)
            nisa.tensor_tensor(dst=sel, data1=sel, data2=low[:, src], op=nl.multiply, engine=vec)
            nisa.tensor_copy_predicated(dst=tile[:, dst], src=code_q[:, src], predicate=sel)
        dst = slice((4 * degree + s) * k_cols, (4 * degree + s + 1) * k_cols)
        nisa.tensor_copy_predicated(dst=tile[:, dst], src=code_ov[:, src], predicate=over[:, src])
    val2 = _sbuf(alloc, (P, k), nl.bfloat16, "pg_val2")
    idx2 = _sbuf(alloc, (P, k), nl.uint32, "pg_idx2")
    nisa.topk(val_dst=val2[0:P, 0:k], idx_dst=idx2[0:P, 0:k], src=tile.view(nl.bfloat16), n=n)
    x = _sbuf(alloc, (P, k_cols), i32, "pg_x")
    decode_codes(val2.view(u16)[0:P, 0:k_cols], x, P, k_cols, alloc)
    # the winner's block (16 snake partitions x k_cols columns each):
    # s = block & (degree - 1), b = block >> log2(degree)
    blk = _sbuf(alloc, (P, k_cols), nl.uint32, "pg_blk")
    nisa.tensor_scalar(dst=blk, data=idx2[0:P, 0:k_cols], op0=nl.right_shift, operand0=_log2(SNAKE * k_cols))
    s_t = _sbuf(alloc, (P, k_cols), nl.uint32, "pg_s")
    nisa.tensor_scalar(dst=s_t, data=blk, op0=nl.bitwise_and, operand0=degree - 1)
    b_t = _sbuf(alloc, (P, k_cols), nl.uint32, "pg_b")
    nisa.tensor_scalar(dst=b_t, data=blk, op0=nl.right_shift, operand0=_log2(degree))
    # v = x * 4 + b (b < 4), else x + OVERFLOW_FROM
    vq = _sbuf(alloc, (P, k_cols), i32, "pg_vq")
    nisa.scalar_tensor_tensor(dst=vq, data=x, op0=nl.multiply, operand0=4.0, op1=nl.add, operand1=b_t)
    is_ov = _sbuf(alloc, (P, k_cols), u16, "pg_is_ov")
    nisa.tensor_scalar(dst=is_ov, data=b_t, op0=nl.greater_equal, operand0=4, engine=vec)
    vo = _sbuf(alloc, (P, k_cols), i32, "pg_vo")
    nisa.tensor_scalar(dst=vo, data=x, op0=nl.add, operand0=OVERFLOW_FROM, engine=vec)
    nisa.tensor_copy_predicated(dst=vq, src=vo, predicate=is_ov)
    rank_term = _sbuf(alloc, (P, k_cols), i32, "pg_rank")
    nisa.tensor_scalar(dst=rank_term, data=s_t, op0=nl.multiply, operand0=float(block_size), engine=vec)
    out = _sbuf(alloc, (P, k_cols), i32, "pg_out")
    deal_global(vq, out, P, k_cols, degree, block_size, rank_term, alloc)
    # an empty slot: a non-member's pattern, which decodes to CODE_LIMIT (every code is below it)
    bad = _sbuf(alloc, (P, k_cols), u16, "pg_bad")
    nisa.tensor_scalar(dst=bad, data=x, op0=nl.greater_equal, operand0=CODE_LIMIT, engine=vec)
    nisa.tensor_copy_predicated(dst=out, src=-1, predicate=bad)
    return out
