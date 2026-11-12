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

"""NKI kernel for the KDA (Kimi Delta Attention) prefill scan, version 2.

The kernel is a clean-room chunked implementation of the gated delta rule
with a per-channel decay.

Layouts (fp32, ``H`` heads, ``K = V = 128``, ``Tp`` a multiple of 128):

* ``qT``, ``kT``, ``kbT``: ``[H, K, Tp]``. These are the scaled normalized
  query, the normalized key, and ``beta * key``, with the channel axis first.
* ``vb``: ``[H, Tp, V]``, ``beta * value``.
* ``gtok``: ``[H, Tp, K]``, the log gate in ``[lower_bound, 0]``.
* ``S0``: ``[H, K, V]``, the initial state.
* Constants from ``kda_scan_constants`` (``prefill/kda.py``).

Outputs: ``o [H, Tp, V]`` and ``S_final [H, K, V]``, both fp32.

All matmuls use fp32 operands. The kernel keeps the state ``S`` of each
head in SBUF for the full sequence. The stages of all heads run round
robin, so the engines overlap the serial chains of the heads, and steps 1-2
of each chunk run spread over steps 3-5 of the previous chunk of the same
head. The work tiles of each head are its own, for the full sequence, so
the SBUF allocator cannot place the tiles of two heads at one address,
which would order the heads' chains behind each other.

Trailing zero chunks. A serving segment pads the prompt to the bucket
length with rows whose ``q``, ``k`` and ``g`` are zero. Such a chunk leaves
the state unchanged (its decay is ``exp(0) = 1`` and it writes ``k = 0``)
and its output is zero, for any state. The kernel finds the last chunk
that holds a nonzero ``q``, ``k`` or ``g`` value on the device (an exact
abs-max test, so ``-0.0`` counts as zero), and it runs only the chunks up
to that one. The first ``N_STATIC`` chunks run as static code while the
test runs. The other chunks run in a loop with a runtime trip count, two
chunks per iteration, and the last chunk of an odd count runs in a
0-or-1-trip tail loop. The output rows of the chunks after the last one are
written as zeros. With the optional ``valid_len`` input the caller gives
the count of the valid rows instead, and the kernel skips the zero test.
It then also makes the pad rows (the rows at or after ``valid_len``) of
each loaded chunk exact zeros (``g = -0.0``), whatever the pad inputs hold:
the scan contracts over the chunk's tokens, so a NaN or Inf in a pad row
would otherwise reach every row of the chunk and the state (``0 * NaN``).

Leading zero chunks. A prefix-cache hit that starts inside a segment gives
the cached rows before ``valid_start`` as zero rows, and ``S0`` is the state
after them. Such a chunk leaves the state unchanged and has a zero output,
like a trailing zero chunk. With the optional ``valid_start`` input, the
loop starts at the first chunk that holds a row at or after it (at least
chunk ``N_STATIC``), so the whole chunks before it do not run, and their
output rows keep the zeros of the loop's zero fill.

Worst-case exp() arguments (``|lower_bound| <= 8``):

* ``E1 = exp(G[t])``: ``[-1024, 0]``.
* ``E2 = exp(G[C-1] - G[s])``: ``[-1016, 0]``.
* ``E3 = exp(G[C-1])``: ``[-1024, 0]``.
* ``E4 = exp(G[t] - G[r_b])``, ``t`` in sub-block ``b``: ``[0, 56]``.
* ``E5 = exp(SUF[s])``, the in-sub-block part of ``G[r_b] - G[s]``:
  ``[-56, 0]``.
* ``F = exp(BS[b', b])``, the sum of the whole sub-blocks between:
  ``[-1024, 0]``.

``r_b = 8b + 7`` is the last token of sub-block ``b``. Each pre-mask entry
of the pair matrices is at most ``exp(56)``. The fp32 limit is
``exp(88.7)``.

Wide sub-blocks. With the static ``lower_bound`` argument at most 5.86 in
magnitude (the Kimi-K3 gate law gives -5), the sub-blocks hold 16 tokens:
``E4`` and the pre-mask entries are then at most ``exp(15 * |lower_bound|)``
(``exp(75)`` at -5), and the pair matrices need half the matmuls.
"""

from __future__ import annotations

import weakref

import nki
import nki.isa as nisa
import nki.language as nl
from nki.compiler._builtins import get_backend

from ..utils.trn_target import dma_priority, is_trn2

C = 128  # chunk length (tokens)
SUB = 8  # sub-block length for the pair-decay split
NSUB = C // SUB
LEVELS = 7  # log2(C) block doubling levels for the inverse
NPAIR = NSUB * (NSUB + 1) // 2  # sub-block pairs b' <= b
WIDE_SUB = 16  # sub-block length when the gate bound allows it
_EXP_LIMIT = 88.0  # fp32 exp() overflows above 88.7

_F32 = nl.float32
_I32 = nl.int32


# The calls of this kernel in each trace, keyed by the trace context (see _new_trace_id).
_TRACE_CALLS = weakref.WeakKeyDictionary()
_END = object()  # the value of next() on a finished stage generator
SPREAD = 4  # stages of the previous chunk per stage of the next chunk
N_STATIC = 2  # leading chunks that always run, as static code
DET_ELEMS = 2048  # fp32 elements per partition of one zero-test buffer (8 KB)
DET_BUFS = 2  # zero-test buffers in flight for each DMA queue
DET_R0 = 15  # driver round of the first zero-test reduction
DET_GAP = 4  # driver rounds between the reductions of two zero-test loads
DET_QK_SPLIT = 2  # chunks per q/k zero-test reduction
DET_G_SPLIT = 4  # g zero-test reductions per load


class _Namer:
    """Give each buffer a name that is unique in one kernel trace.

    The trace frontend needs a unique name for each op. The name has three
    parts: a prefix for the call in the trace, a scope, and a counter. Use one
    ``_Namer`` per scope. A ``fori_loop`` body is traced one time, so the
    names of a body are unique.
    """

    def __init__(self, trace_id, scope):
        self.prefix = f"k{trace_id}_{scope}"
        self.n = 0

    def __call__(self, name):
        self.n += 1
        return f"{self.prefix}_{name}_{self.n}"


class _Once:
    """Call ``fn`` on the first call only, and return its result each time."""

    def __init__(self, fn):
        self.fn = fn
        self.val = None

    def __call__(self):
        if self.val is None:
            self.val = self.fn()
        return self.val


def _new_trace_id():
    """Number this call among the calls of this kernel in the current trace: 1, 2, ...

    The numbers restart in each trace, so the names of a trace depend only on
    its own calls, not on what the process traced before. The calls that one
    trace inlines (a program of several layers) still get distinct numbers.
    """
    ctx = get_backend().get_current_context()
    n = _TRACE_CALLS.get(ctx, 0) + 1
    _TRACE_CALLS[ctx] = n
    return n


def _sb(shape, name, dtype=_F32):
    return nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf, name=name)


def _ps(shape, name):
    return nl.ndarray(shape, dtype=_F32, buffer=nl.psum, name=name)


def _dma(dst, src, priority=0):
    """DMA on the Sync engine's hardware DGE queue.

    A software DGE DMA keeps GpSimd busy, and VectorE
    instructions stall while it runs. A hardware DGE DMA issues
    quickly on the otherwise idle Sync engine. ``priority`` is the DMA QoS
    class, 0 the highest. Trn2 has no DMA QoS, so its DMA gets no priority.
    """
    nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync, **dma_priority(priority))


def _run(gens):
    """Run stage generators round robin until all of them finish."""
    gens = list(gens)
    while gens:
        for g in list(gens):
            if next(g, _END) is _END:
                gens.remove(g)


def _pipe(items):
    """Run ``(back, front)`` generator pairs of one head in order.

    One stage of ``front`` (steps 1-2 of the next chunk) runs after every
    ``SPREAD`` stages of ``back`` (steps 3-5 of the current chunk). Either
    one can be None.
    """
    for back, front in items:
        if back is not None:
            for k, _ in enumerate(back, start=1):
                yield
                if front is not None and k % SPREAD == 0 and next(front, _END) is _END:
                    front = None
        while front is not None:
            if next(front, _END) is _END:
                front = None
            yield


def _run_items(items_by_head, n_groups):
    """Run the ``(back, front)`` items of all heads of one loop body.

    With one group, each head's items run as one ``_pipe`` and all heads round robin. With more, the items run
    item-major: for each item, the heads run as ``n_groups`` contiguous groups, one after the other, and every group
    ends item k before any group starts item k + 1. Item k + 1's front loads the v tile that item k's back reads
    (one tile for all heads), so this order keeps each load after every head's read. Each head's instructions and
    their order do not change.
    """
    if n_groups == 1:
        _run([_pipe(items) for items in items_by_head])
        return
    n = len(items_by_head)
    for k in range(len(items_by_head[0])):
        for g in range(n_groups):
            _run([_pipe([items[k]]) for items in items_by_head[n * g // n_groups : n * (g + 1) // n_groups]])


def _inverse_head_hook(isa, nmx, a_t2, tt, tm, block_inverse):
    """The step-3 inverse levels below log2(block_size) through ``block_inverse``, a stage generator.

    ``tt`` gets A ([t, s], one PE transpose of A^T), the hook writes T at its block size into ``tm``, and ``tt`` then
    gets T^T.
    """
    ps_a = _ps((C, C), nmx("ps_a"))
    isa.nc_transpose(dst=ps_a, data=a_t2)
    yield
    isa.tensor_copy(dst=tt, src=ps_a, engine=nisa.engine.scalar)
    block_inverse(tm, tt, isa)
    yield
    ps_t = _ps((C, C), nmx("ps_t"))
    isa.nc_transpose(dst=ps_t, data=tm)
    yield
    isa.tensor_copy(dst=tt, src=ps_t, engine=nisa.engine.scalar)
    yield


@nki.jit
def kda_prefill_v2(
    qT,
    kT,
    kbT,
    vb,
    gtok,
    S0,
    lcat,
    wcat,
    mask_a3,
    mask_b3,
    mask_lvl,
    ident,
    valid_len=None,
    lower_bound=-8.0,
    o_private=False,
    pad_rows_zero=False,
    pair_psum_zero="vector",
    chunks_per_iter=2,
    decay_onezero=False,
    own_count_tile=False,
    valid_start=None,
    lead_state=False,
    head_groups=1,
    block_inverse=None,
):
    """Run the chunked KDA prefill scan for all heads.

    Args:
        qT: ``[H, K, Tp]`` fp32 scaled normalized queries (channel first).
        kT: ``[H, K, Tp]`` fp32 normalized keys (channel first).
        kbT: ``[H, K, Tp]`` fp32 ``beta * key`` (channel first).
        vb: ``[H, Tp, V]`` fp32 ``beta * value``.
        gtok: ``[H, Tp, K]`` fp32 log gates.
        S0: ``[H, K, V]`` fp32 initial state.
        lcat: ``[C, 2C]`` prefix and strict-suffix sum matrix.
        wcat: ``[C, C + NPAIR]`` 0/1 sum matrix for the in-sub-block
            suffix sums and the sub-block pair sums.
        mask_a3: ``[C, NSUB, SUB]`` strict causal mask (``s < t``).
        mask_b3: ``[C, NSUB, SUB]`` causal mask (``s <= t``).
        mask_lvl: ``[C, LEVELS, C]`` doubling-level quadrant masks of
            ``A_l T``, layout ``[t, level, x]``.
        ident: ``[C, C]`` identity.
        valid_len: Optional ``[1, 1]`` int32 count of the valid rows. The
            rows at or after it are pad rows: the kernel runs only the
            chunks with a valid row, with no zero test, and zeroes the pad
            rows of each loaded chunk (``q = k = kb = v = 0``,
            ``g = -0.0``), so their inputs may hold anything, NaN and Inf
            included; their output rows are 0. None (the default) finds the
            chunks to run from the data, exact for any input; its pad rows
            must then be zero.
        lower_bound: Static bound of the log gates, ``gtok >= lower_bound``.
            With ``15 * |lower_bound| < 88`` (the Kimi-K3 gate law gives
            -5), the pair matrices use sub-blocks of 16 tokens instead of
            8, which halves their matmuls. The default -8 keeps 8.

        o_private: Static. Allocate ``o`` in ``private_hbm``. Use it when a caller kernel nests this kernel and
            reads ``o`` as its own scratch: with ``NKI_LNC1_SHARED_HBM=1``, a ``shared_hbm`` scratch tensor is one
            buffer for the two cores of an HBM stack, and the per-core outputs would then race. The default keeps
            ``shared_hbm`` (the output of a standalone call).
        pad_rows_zero: Static. The caller guarantees that every pad row of the five inputs is an exact zero (as
            the KDA block kernel makes them with selects). The kernel then skips its own pad-row zeroing of each
            loaded chunk. The default False keeps the zeroing, which also protects against NaN and Inf in the
            pad rows of the inputs.
        pair_psum_zero: Static. How each chunk's pair-matrix PSUM tile becomes zero before its sub-block
            matmuls, which write only its causal part (the masks then multiply the rest by 0). ``"vector"``
            (the default) is a VectorE memset at the sub-block stage, which TensorE waits for. ``"early"`` is
            the same memset two stages earlier, with the decay-sum matmul. ``"scalar"`` is a ScalarE copy of
            a zero tile at the sub-block stage. All three write +0.0, so the outputs are the same bits.
        chunks_per_iter: Static. Chunks per iteration of the main chunk loop: 2 (the default) or 4. Each loop
            iteration ends with all-engine barriers, after which TensorE restarts at a throttled rate. With 4,
            a loop of four-chunk iterations runs first, then the 0-or-1-trip pair loop and the 0-or-1-trip tail
            loop take the rest. Each chunk runs the same steps in the same order, so the outputs are the same
            bits.
        decay_onezero: Static. Mark ``lcat`` and ``wcat``, the 0/1 moving operands of the two decay-sum matmuls
            (steps 1 and 2), as ones/zeros tiles (``is_moving_onezero``). For an fp32 moving operand, the
            matmul then skips the pass of its low half, which is exactly 0 for a 0/1 tile: only zero products
            are dropped, and every consumer of these sums is an ``exp``. The default False keeps both passes.
            The NKI release in use does not pass ``is_moving_onezero`` to the backend, so the
            compiled program is the same with either value.
        own_count_tile: Static. With ``valid_len``, compute the loop counts in a tile of their own and only read
            the tile that ``valid_len`` is loaded into. A caller that loads ``valid_len`` too (the KDA block kernel)
            can then share that load with this kernel: the compiler may merge the two loads into one tile, and the
            in-place count arithmetic of the default would wait until every earlier reader of the caller's tile is
            done. The same integer operations run, so the counts and the outputs are the same bits.
        valid_start: Optional ``[1, 1]`` int32, the first row of a prefix-cache hit inside the segment. Needs
            ``valid_len`` and ``pad_rows_zero``. The caller gives the rows before it as zero rows
            (``q = k = kb = v = 0``, ``g = 0``) and ``S0`` as the state after them. The loop then starts at chunk
            ``min(max(max(valid_start, 0) // 128, N_STATIC), n_eff)``, where ``n_eff`` is the count of chunks to
            run from ``valid_len``: the whole chunks before ``valid_start`` would leave the state unchanged and
            write zero output rows, so they do not run, and their output rows are zero. The first ``N_STATIC``
            chunks run in every case. None (the default) runs every chunk up to ``n_eff``.
        lead_state: Static. Needs ``valid_len``. Also return ``S_lead``, the state after the rows ``[0, g)`` with
            ``g = max(vs, ((max(valid_len, 1) - 1) >> 7) << 7)`` and ``vs = clamp(valid_start, 0, valid_len)`` (0
            without ``valid_start``): with a ``valid_start`` that is a multiple of 128 (or none), ``g`` is the
            first row of the last chunk with a valid row. A later call that starts its rows at ``g`` (its
            ``valid_start``) with ``S0 = S_lead`` then gives the bits of this call's rows from ``g`` on. Each chunk
            copies the state into a staging tile before its update (the static chunks only while they start
            before ``valid_len``), and one store writes the last copy. With ``valid_start`` too, a static chunk
            whose rows all lie before ``valid_start`` keeps the state bit for bit (its update is undone from the
            staging tile), so its identity steps do not rest on ``exp(0) = 1`` in hardware. The default False
            gives the program without ``lead_state``.
        head_groups: Static. The main chunk loop runs the heads in this many contiguous groups, one group after the
            other (item by item, see ``_run_items``), instead of all heads round robin. Each head runs the same
            instructions in the same order, so the outputs are the same bits. On trn3 with 12 heads (a TP8 rank),
            the scan is faster with 2 groups, and more so with 6 groups, at T 2048 and at
            T 4096. With 3 heads, 2 groups make it slower. 1 (the default) runs all heads round robin (no
            head groups).
        block_inverse: Static. None (the default) gives the program without the hook. Else a callable
            ``block_inverse(dst, src, isa)`` with an int attribute ``block_size`` (a power of two, 2 to 64) that
            replaces the step-3 inverse levels below ``log2(block_size)``. ``src`` [128, 128] fp32 SBUF holds the
            strictly lower triangular A [t, s] of a chunk and head. The call writes into ``dst`` [128, 128] fp32 SBUF
            the inverse of I + A on each diagonal ``block_size`` block, and 0 outside those blocks. ``isa`` is the
            ``nisa`` form that the scan uses for that head (it adds the scheduling hints); use it for the nisa calls
            of the hook. The outputs keep the default's bits only if the hook gives the bits of those levels. The
            hook runs with any ``head_groups``. On trn3 with 12 heads at T 2048, the GpSimd hook at block 16 made
            the scan faster at 1 group and at 6 groups than the default at 6 groups.

    Returns:
        ``(o [H, Tp, V], S_final [H, K, V])`` fp32 HBM tensors, then ``S_lead [H, K, V]`` fp32 with
        ``lead_state``.

    Example:
        >>> # o, s = nki.simulate(kda_prefill_v2[1])(*numpy_inputs)
    """
    H, K, Tp = qT.shape
    V = vb.shape[2]
    if K != 128 or V != 128:
        raise ValueError(f"kda_prefill_v2 supports K = V = 128 only, got K={K}, V={V}")
    if Tp % C != 0:
        raise ValueError(f"Tp must be a multiple of {C}, got Tp={Tp}")
    if chunks_per_iter not in (2, 4):
        raise ValueError(f"chunks_per_iter must be 2 or 4, got {chunks_per_iter!r}")
    if pair_psum_zero not in ("vector", "early", "scalar"):
        raise ValueError(f"pair_psum_zero must be 'vector', 'early' or 'scalar', got {pair_psum_zero!r}")
    if valid_start is not None and (valid_len is None or not pad_rows_zero):
        raise ValueError("valid_start needs valid_len and pad_rows_zero")
    if lead_state and valid_len is None:
        raise ValueError("lead_state needs valid_len")
    if not isinstance(head_groups, int) or not 1 <= head_groups <= max(H, 1):
        raise ValueError(f"head_groups must be an int in [1, {H}], got {head_groups!r}")
    if block_inverse is not None:
        block = getattr(block_inverse, "block_size", None)
        if not callable(block_inverse) or not isinstance(block, int) or not 2 <= block <= C // 2 or block & (block - 1):
            raise ValueError(
                f"block_inverse must be a callable with a power-of-two int block_size in [2, {C // 2}], "
                f"got {block_inverse!r}"
            )
    NCH = Tp // C
    # The pair matrices split each chunk into sub-blocks of `sub` tokens.
    # The query-side factor exp(G[t] - G[r_b]) is at most
    # exp((sub - 1) * |lower_bound|), which must stay below the fp32 limit.
    wide = (WIDE_SUB - 1) * abs(float(lower_bound)) < _EXP_LIMIT
    sub = WIDE_SUB if wide else SUB
    nsub = C // sub
    npair = nsub * (nsub + 1) // 2
    n_static = min(NCH, N_STATIC)
    # decay_onezero: the hint on the decay-sum matmuls (given only when set, so the default call is unchanged).
    onezero = {"is_moving_onezero": True} if decay_onezero else {}

    tid = _new_trace_id()
    nm = _Namer(tid, "const")

    o = nl.ndarray((H, Tp, V), dtype=_F32, buffer=nl.private_hbm if o_private else nl.shared_hbm, name=nm("o"))
    s_fin = nl.ndarray((H, K, V), dtype=_F32, buffer=nl.shared_hbm, name=nm("s_fin"))
    pad = None if valid_len is None or pad_rows_zero else _PadMask(tid, valid_len)

    # The constants, loaded once, in the order of first use (the first
    # chunk loads before them).
    lcat_sb = _sb((C, 2 * C), nm("lcat_sb"))
    wcat_sb = _sb((C, C + NPAIR), nm("wcat_sb"))
    # The 16-token sub-block sums, built from the 8-token ones of wcat.
    wcat16 = _sb((C, C + npair), nm("wcat16")) if wide else None
    ma_sb = _sb((C, NSUB, SUB), nm("ma_sb"))
    mb_sb = _sb((C, NSUB, SUB), nm("mb_sb"))
    ml_sb = _sb((C, LEVELS, C), nm("ml_sb"))
    id_sb = _sb((C, C), nm("id_sb"))
    ml0_sb = _sb((C, C), nm("ml0_sb"))
    # The state of all heads, [K, H, V], in SBUF for the full sequence.
    S_all = _sb((K, H, V), nm("S_all"))
    # With lead_state: the staging tile of the state at the start of the last chunk that ran (see _LeadState).
    lead = _LeadState(tid, valid_len, valid_start, n_static, K, H, V) if lead_state else None
    # With pair_psum_zero="scalar": the zeros that ScalarE copies into each pair-matrix PSUM tile.
    zero_ab = None
    if pair_psum_zero == "scalar":
        zero_ab = _sb((C, nsub, 2 * sub), nm("zero_ab"))
        nisa.memset(dst=zero_ab, value=0.0, engine=nisa.engine.vector)

    def load_sums():
        """The decay-sum matrices, which steps 1-2 of the first chunk need."""
        _dma(lcat_sb, lcat)
        _dma(wcat_sb, wcat)
        if wide:
            _wide_wcat(wcat16, wcat_sb, nsub)

    def load_constants():
        _dma(ma_sb, mask_a3)
        _dma(mb_sb, mask_b3)
        _dma(ml_sb, mask_lvl)
        _dma(id_sb, ident)
        _dma(S_all, S0.ap(pattern=[[V, K], [K * V, H], [1, V]]))
        if lead is not None:
            _dma(lead.tile, S0.ap(pattern=[[V, K], [K * V, H], [1, V]]))

    def level0_mask():
        """The level-0 mask in the [s, t] layout of A^T, before the first step 3."""
        for _ in range(8):
            yield
        ps_m0 = _ps((C, C), nm("ps_m0"))
        nisa.nc_transpose(dst=ps_m0, data=ml_sb[:, 0, :], engine=nisa.engine.tensor)
        nisa.tensor_copy(dst=ml0_sb, src=ps_m0)

    # Steps 1-2 of a chunk write a buffer set, and steps 3-5 of the chunk
    # read it. Chunk c uses set c % 2, so steps 1-2 of chunk c + 1 can run
    # while steps 3-5 of chunk c run. The sets live outside the chunk loop.
    # A set also holds the work tiles of steps 3-5 of each head, so steps
    # 3-5 of chunk c + 1 can start before those of chunk c end.
    def new_back_ws(i, h):
        return {
            "tt": _sb((C, C), nm(f"wb{i}_tt_{h}")),
            "tm": _sb((C, C), nm(f"wb{i}_tm_{h}")),
            "p_sb": _sb((C, C), nm(f"wb{i}_p_{h}")),
            "kend": _sb((C, K), nm(f"wb{i}_kend_{h}")),
            "r_sb": _sb((C, V), nm(f"wb{i}_r_{h}")),
            "e_sb": _sb((C, V), nm(f"wb{i}_e_{h}")),
        }

    def new_set(i):
        return {
            "qg": [_sb((K, C), nm(f"qg{i}_{h}")) for h in range(H)],
            "kbg": [_sb((K, C), nm(f"kbg{i}_{h}")) for h in range(H)],
            "kend": [_sb((K, C), nm(f"kend{i}_{h}")) for h in range(H)],
            "dec": [_sb((K, 1), nm(f"dec{i}_{h}")) for h in range(H)],
            "a": [_sb((C, NSUB, SUB), nm(f"a{i}_{h}")) for h in range(H)],
            "b": [_sb((C, NSUB, SUB), nm(f"b{i}_{h}")) for h in range(H)],
            "v": _sb((C, H, V), nm(f"v{i}")),
            "wb": [new_back_ws(i, h) for h in range(H)],
        }

    sets = [new_set(i) for i in range(min(NCH, 2))]

    # The work tiles of steps 1-2 of each head, which all its chunks reuse.
    # All work tiles live for the whole kernel, so no two heads share an
    # address. When the caller keeps so much SBUF live that the allocator
    # falls back to graph coloring with spills, it packs short-lived tiles
    # with disjoint lifetimes at one address; the write-after-read order
    # between such tiles of two heads then serializes the heads' chains
    # (a loop iteration took longer).
    def new_front_ws(h):
        return {
            "e_g": _sb((K, C), nm(f"wf_eg_{h}")),
            "e_s": _sb((K, C), nm(f"wf_es_{h}")),
            "e5": _sb((K, C), nm(f"wf_e5_{h}")),
            "e4": _sb((K, C), nm(f"wf_e4_{h}")),
            "f_bs": _sb((K, npair), nm(f"wf_fbs_{h}")),
            "k5": _sb((K, nsub, sub), nm(f"wf_k5_{h}")),
            "x4": _sb((K, 2, C), nm(f"wf_x4_{h}")),
            "y": [_sb((K, b + 1, sub), nm(f"wf_y{b}_{h}")) for b in range(nsub)],
        }

    wsf = [new_front_ws(h) for h in range(H)]
    # With head groups the main loop runs one group after the other (``_run_items``), so there a head's work
    # tiles need not outlive its group's turn: the heads at one position of their groups share one set of loop
    # work tiles, which a group writes after the group before it has read them. The loop then keeps the work
    # tiles of one group live instead of those of all heads. The static chunks run all heads round robin and
    # keep the per-head tiles; with one group the loop takes them too.
    if head_groups > 1:
        starts = [H * g // head_groups for g in range(head_groups + 1)]
        slot = [h - starts[g] for g in range(head_groups) for h in range(starts[g], starts[g + 1])]
        n_slot = max(b - a for a, b in zip(starts, starts[1:]))
        for i, st in enumerate(sets):
            wbl = [new_back_ws(f"{i}l", s) for s in range(n_slot)]
            st["wbl"] = [wbl[slot[h]] for h in range(H)]
        wsfl_slots = [new_front_ws(f"l{s}") for s in range(n_slot)]
        wsfl = [wsfl_slots[slot[h]] for h in range(H)]
    else:
        for st in sets:
            st["wbl"] = st["wb"]
        wsfl = wsf
    order = _FrontBackOrder(H)

    def chan_ap(x, t_off):
        """``x [H, K, Tp]`` at the chunk at token ``t_off``, as ``[K, H, C]``."""
        pattern = [[Tp, K], [K * Tp, H], [1, C]]
        if isinstance(t_off, int):
            return x.ap(pattern=pattern, offset=t_off)
        reg, extra = t_off
        return x.ap(pattern=pattern, offset=extra, scalar_offset=reg, indirect_dim=2)

    def tok_ap(x, t_off):
        """``x [H, Tp, D]`` at the chunk at token ``t_off``, as ``[C, H, D]``."""
        D = x.shape[2]
        pattern = [[D, C], [Tp * D, H], [1, D]]
        if isinstance(t_off, int):
            return x.ap(pattern=pattern, offset=t_off * D)
        reg, extra = t_off
        return x.ap(pattern=pattern, offset=extra * D, scalar_offset=reg, indirect_dim=1)

    def loader(nmx, t_off, v3, after_g=None, pad_rel=None, ready=None):
        """Load one chunk of all heads (one DMA per input), on first use.

        ``v3`` is the ``[C, H, V]`` tile for ``vb``, which steps 3-5 read.
        ``after_g`` runs right after the ``gtok`` load, which step 1 needs
        first. ``.issue()`` issues the DMAs only (the static chunks issue
        theirs at the start). With ``valid_len``, the pad rows become zero
        at their first use (see ``_PadMask``): the first call builds the
        chunk's predicates and zeroes ``v3``, and each head's steps 1-2
        zero its slices of g, q, k and beta * k. ``pad_rel`` is the chunk's
        first token relative to the loop's current chunk (a loop chunk only).
        ``ready`` (a list) guards a load whose offset register is written in
        the trace: the DMAs issue only once the list is not empty.
        """

        def issue():
            if ready is not None and not ready:
                raise RuntimeError("kda_prefill_v2: a chunk load issued before its offset register was written")
            g3 = _sb((C, H, K), nmx("g3"))
            _dma(g3, tok_ap(gtok, t_off))
            if after_g is not None:
                after_g()
            q3 = _sb((K, H, C), nmx("q3"))
            _dma(q3, chan_ap(qT, t_off))
            k3 = _sb((K, H, C), nmx("k3"))
            _dma(k3, chan_ap(kT, t_off))
            kb3 = _sb((K, H, C), nmx("kb3"))
            _dma(kb3, chan_ap(kbT, t_off))
            _dma(v3, tok_ap(vb, t_off))
            return q3, k3, kb3, g3

        issue_once = _Once(issue)

        def get():
            q3, k3, kb3, g3 = issue_once()
            pm = None
            if pad is not None:
                pm = pad.predicates(nmx, t_off if isinstance(t_off, int) else None, pad_rel)
                pad.zero_v(pm, H, v3)
            return q3, k3, kb3, g3, pm

        ld = _Once(get)
        ld.issue = issue_once
        return ld

    class _Out:
        """The output tile of one chunk of all heads, stored after the last head.

        With ``tile``, the chunk writes that tile and the caller stores it
        later (``t_off`` is None).
        """

        def __init__(self, nmx, t_off, tile=None):
            self.o3 = _sb((C, H, V), nmx("o3")) if tile is None else tile
            self.t_off = t_off
            self.left = H

        def done(self):
            self.left -= 1
            if self.left == 0 and self.t_off is not None:
                _dma(tok_ap(o, self.t_off), self.o3)

    def front(h, ld, cs, nmx, loop=False):
        """Steps 1-2 of one chunk of head ``h``, a stage generator. ``loop``: a main-loop chunk."""
        isa = order.isa(h, front=True)
        q3, k3, kb3, g3, pm = ld()
        if pm is not None:
            pad.zero_head(isa, pm, h, g3, (q3, k3, kb3))
        yield
        q_c, k_c, kb_c, g_c = q3[:, h, :], k3[:, h, :], kb3[:, h, :], g3[:, h, :]
        wf = (wsfl if loop else wsf)[h]

        # ---- Step 1: prefix sums G[t] and strict suffix sums. ----
        # Each output adds one contiguous run of gates with one sign.
        ps_g = _ps((K, 2 * C), nmx("ps_g"))
        isa.nc_matmul(dst=ps_g, stationary=g_c, moving=lcat_sb, **onezero)
        yield
        e_g = wf["e_g"]
        isa.activation(dst=e_g, op=nl.exp, data=ps_g[:, 0:C])  # E1
        isa.tensor_tensor(dst=cs["qg"][h], data1=q_c, data2=e_g, op=nl.multiply, engine=nisa.engine.vector)
        isa.tensor_tensor(dst=cs["kbg"][h], data1=kb_c, data2=e_g, op=nl.multiply, engine=nisa.engine.vector)
        e_s = wf["e_s"]
        isa.activation(dst=e_s, op=nl.exp, data=ps_g[:, C : 2 * C])  # E2
        isa.tensor_tensor(dst=cs["kend"][h], data1=k_c, data2=e_s, op=nl.multiply, engine=nisa.engine.vector)
        isa.activation(dst=cs["dec"][h], op=nl.exp, data=ps_g[:, C - 1 : C])  # E3
        yield

        # ---- Step 2: pair matrices A^T and B^T, layout [s, t]. ----
        # Sub-block b uses the reference r_b = sub * b + sub - 1, its last
        # token (sub = 8, or 16 with a small gate bound). For s
        # in sub-block b' <= b, G[r_b] - G[s] = SUF[s] + BS[b', b]: SUF[s]
        # sums the gates after s in its own sub-block, and BS[b', b] sums the
        # whole sub-blocks b'+1..b. Both are runs of gates with one sign, so
        # exp(G[r_b] - G[s]) = E5[s] * F[b', b]. The query side is
        # exp(G[t] - G[r_b]) = exp(-SUF[t]) (E4). One matmul gives SUF and
        # all BS.
        ps_d = _ps((K, C + npair), nmx("ps_d"))
        isa.nc_matmul(dst=ps_d, stationary=g_c, moving=wcat16 if wide else wcat_sb, **onezero)
        ps_ab = None
        if pair_psum_zero == "early":
            # The pair-matrix PSUM tile (below), zeroed here so that the sub-block matmuls do not wait for it.
            ps_ab = _ps((C, nsub, 2 * sub), nmx("ps_ab"))
            isa.memset(dst=ps_ab, value=0.0, engine=nisa.engine.vector)
        yield
        e5 = wf["e5"]
        isa.activation(dst=e5, op=nl.exp, data=ps_d[:, 0:C])
        e4 = wf["e4"]
        isa.activation(dst=e4, op=nl.exp, data=ps_d[:, 0:C], scale=-1.0)
        f_bs = wf["f_bs"]
        isa.activation(dst=f_bs, op=nl.exp, data=ps_d[:, C : C + npair])
        k5 = wf["k5"]
        isa.tensor_tensor(dst=k5.reshape((K, C)), data1=k_c, data2=e5, op=nl.multiply, engine=nisa.engine.vector)
        # Query-side factors, kb then q for each t.
        x4 = wf["x4"]
        isa.tensor_tensor(dst=x4[:, 0, :], data1=kb_c, data2=e4, op=nl.multiply, engine=nisa.engine.vector)
        isa.tensor_tensor(dst=x4[:, 1, :], data1=q_c, data2=e4, op=nl.multiply, engine=nisa.engine.vector)
        yield
        # Sub-block b only needs key columns s < sub * (b + 1). The rest of
        # each column block is masked, so the PSUM tile starts at zero.
        if ps_ab is None:
            ps_ab = _ps((C, nsub, 2 * sub), nmx("ps_ab"))
            if pair_psum_zero == "scalar":
                isa.tensor_copy(dst=ps_ab, src=zero_ab, engine=nisa.engine.scalar)
            else:
                isa.memset(dst=ps_ab, value=0.0, engine=nisa.engine.vector)
        ps_ab4 = ps_ab.reshape((C, nsub, 2, sub))
        per_stage = nsub // 4
        for b in range(nsub):
            t0 = b * sub
            n_b = t0 + sub
            off = b * (b + 1) // 2
            # Key-side factor: k[s] exp(G[r_b] - G[s]) for s < n_b.
            y = wf["y"][b]
            isa.tensor_tensor(
                dst=y,
                data1=k5[:, 0 : b + 1, :],
                data2=f_bs[:, off : off + b + 1].expand_dim(2).broadcast(2, sub),
                op=nl.multiply,
                engine=nisa.engine.vector,
            )
            if is_trn2():
                # Trn2 does not accumulate onto PSUM that a memset or copy wrote, so each region starts the
                # accumulation itself. Each matmul writes its own region once.
                isa.nc_matmul(
                    dst=ps_ab4[0:n_b, b, :, :],
                    stationary=y.reshape((K, n_b)),
                    moving=x4[:, :, t0 : t0 + sub],
                    accumulate=False,
                )
            else:
                isa.nc_matmul(
                    dst=ps_ab4[0:n_b, b, :, :], stationary=y.reshape((K, n_b)), moving=x4[:, :, t0 : t0 + sub]
                )
            if b % per_stage == per_stage - 1:
                yield
        # The masks set each out-of-range entry to 0. These entries are
        # finite (at most exp((sub - 1) * |lower_bound|)), so 0 * entry is
        # 0. The masks are [s, t] matrices, so they reshape to any
        # sub-block length.
        isa.tensor_tensor(
            dst=cs["a"][h].reshape((C, nsub, sub)),
            data1=ps_ab[:, :, 0:sub],
            data2=ma_sb.reshape((C, nsub, sub)),
            op=nl.multiply,
            engine=nisa.engine.vector,
        )
        isa.tensor_tensor(
            dst=cs["b"][h].reshape((C, nsub, sub)),
            data1=ps_ab[:, :, sub : 2 * sub],
            data2=mb_sb.reshape((C, nsub, sub)),
            op=nl.multiply,
            engine=nisa.engine.vector,
        )

    def back(h, cs, v3, out, nmx, lead_c=None, loop=False):
        """Steps 3-5 of one chunk of head ``h``, a stage generator.

        ``lead_c`` (with ``lead_state``): ``"loop"`` for a loop chunk, else the index of the static chunk.
        ``loop``: a main-loop chunk (its work tiles).
        """
        isa = order.isa(h, front=False)
        S = S_all[:, h, :]
        if lead is not None:
            lead.stage(isa, h, S, lead_c)
        wb = cs["wbl" if loop else "wb"][h]
        a_t2 = cs["a"][h].reshape((C, C))
        b_t2 = cs["b"][h].reshape((C, C))

        # ---- Step 3: T = (I + A)^-1 by block forward substitution. ----
        # T_2l = T_l - T_l A_l T_l, with A_l the off-diagonal quadrants of
        # level l. T_l is block diagonal, so A_l T_l is A T_l restricted to
        # the quadrants: every level uses the same stationary A^T and masks
        # the product as it leaves PSUM. Keep T and T^T. |T[t, s]| <= 1.
        # Level 0 (T_0 = I): T_1^T = I - A_0^T and T_1 = (T_1^T)^T.
        tt = wb["tt"]
        tm = wb["tm"]
        first = 1
        if block_inverse is not None:
            # The hook gives T at its block size, so the levels below log2(block_size) are skipped.
            yield from _inverse_head_hook(isa, nmx, a_t2, tt, tm, block_inverse)
            first = block_inverse.block_size.bit_length() - 1
        else:
            isa.tensor_tensor(dst=tt, data1=a_t2, data2=ml0_sb, op=nl.multiply, engine=nisa.engine.vector)
            isa.tensor_tensor(dst=tt, data1=id_sb, data2=tt, op=nl.subtract, engine=nisa.engine.vector)
            yield
            ps_t = _ps((C, C), nmx("ps_t"))
            isa.nc_transpose(dst=ps_t, data=tt)
            yield
            isa.tensor_copy(dst=tm, src=ps_t, engine=nisa.engine.scalar)
            yield
        for lv in range(first, LEVELS):
            ps_p = _ps((C, C), nmx("ps_p"))
            isa.nc_matmul(dst=ps_p, stationary=a_t2, moving=tm)
            yield
            p_sb = wb["p_sb"]
            isa.tensor_tensor(dst=p_sb, data1=ps_p, data2=ml_sb[:, lv, :], op=nl.multiply, engine=nisa.engine.vector)
            yield
            if lv < LEVELS - 1:
                # T <- T - T P, then T^T is an exact PE transpose of T.
                ps_q = _ps((C, C), nmx("ps_q"))
                isa.nc_matmul(dst=ps_q, stationary=tt, moving=p_sb)
                yield
                isa.tensor_tensor(dst=tm, data1=tm, data2=ps_q, op=nl.subtract, engine=nisa.engine.vector)
                yield
                ps_t = _ps((C, C), nmx("ps_t"))
                isa.nc_transpose(dst=ps_t, data=tm)
                yield
                isa.tensor_copy(dst=tt, src=ps_t, engine=nisa.engine.scalar)
                yield
            else:
                # The last level only needs T^T: T^T <- T^T - P^T T^T.
                ps_qt = _ps((C, C), nmx("ps_qt"))
                isa.nc_matmul(dst=ps_qt, stationary=p_sb, moving=tt)
                yield
                isa.tensor_tensor(dst=tt, data1=tt, data2=ps_qt, op=nl.subtract, engine=nisa.engine.vector)
                yield

        # ---- Step 4: kend in the [s, K] layout of the state update. ----
        ps_tr2 = _ps((C, K), nmx("ps_tr2"))
        isa.nc_transpose(dst=ps_tr2, data=cs["kend"][h])
        kend = wb["kend"]
        isa.tensor_copy(dst=kend, src=ps_tr2)
        yield

        # ---- Step 5: chunk recurrence on the state. ----
        # e = U - W S = T (beta v - (beta k E1) S): T applies once, after
        # the state term, so U and W are never formed.
        ps_r = _ps((C, V), nmx("ps_r"))
        isa.nc_matmul(dst=ps_r, stationary=cs["kbg"][h], moving=S)
        r_sb = wb["r_sb"]
        isa.tensor_tensor(dst=r_sb, data1=v3[:, h, :], data2=ps_r, op=nl.subtract, engine=nisa.engine.vector)
        yield
        ps_e = _ps((C, V), nmx("ps_e"))
        isa.nc_matmul(dst=ps_e, stationary=tt, moving=r_sb)
        yield
        e_sb = wb["e_sb"]
        isa.tensor_copy(dst=e_sb, src=ps_e, engine=nisa.engine.scalar)
        yield
        ps_o = _ps((C, V), nmx("ps_o"))
        isa.nc_matmul(dst=ps_o, stationary=cs["qg"][h], moving=S, accumulate=False)
        isa.nc_matmul(dst=ps_o, stationary=b_t2, moving=e_sb, accumulate=True)
        isa.tensor_copy(dst=out.o3[:, h, :], src=ps_o)
        out.done()
        ps_s = _ps((K, V), nmx("ps_s"))
        isa.nc_matmul(dst=ps_s, stationary=kend, moving=e_sb)
        isa.scalar_tensor_tensor(dst=S, data=S, op0=nl.multiply, operand0=cs["dec"][h], op1=nl.add, operand1=ps_s)
        if lead is not None:
            lead.keep(isa, h, S, lead_c)

    # ---- The static chunks 0 .. n_static - 1. ----
    # The heads are independent. Each engine runs its instructions in
    # program order, so the driver interleaves the stages of the heads:
    # while one head waits on a dependency (the inverse levels and the
    # state update are serial PE <-> DVE chains), the engines run the
    # stages of the other heads. The sub-block matmuls of step 2 load a new
    # stationary every few columns. Run as one burst for all heads, they
    # keep the TensorE activity throttle on, so steps 1-2 of chunk n run
    # spread over steps 3-5 of chunk n - 1 instead.
    nms = _Namer(tid, "static")
    dynamic = NCH > n_static
    # The static chunks load into their own v tiles, so all their loads
    # (and those of the first loop chunk) issue at the start.
    v_static = [_sb((C, H, V), nms("v_st")) for _ in range(n_static)]
    if dynamic:
        v_static.append(sets[n_static % 2]["v"])
    # With valid_start, the first loop chunk starts at a runtime token, t0_sb: its steps 1-2 run in the static part
    # as for chunk n_static, from a load that the loop-count generator issues once it has written t0_sb.
    lead_skip = dynamic and valid_start is not None
    t0_sb = _sb((1, 1), nm("t0_sb"), _I32) if lead_skip else None
    t0_ready = []
    loads = [loader(nms, c * C, v_static[c], after_g=load_sums if c == 0 else None) for c in range(n_static)]
    if dynamic:
        if lead_skip:
            loads.append(loader(nms, (t0_sb, 0), v_static[n_static], ready=t0_ready))
        else:
            loads.append(loader(nms, n_static * C, v_static[n_static]))
    # With a loop, the output of the last static chunk stays pending in
    # o_pend: the loop stores it while it runs the next chunk.
    o_pend = _sb((C, H, V), nm("o_pend")) if dynamic else None
    outs = [_Out(nms, c * C) for c in range(n_static)]
    if dynamic:
        outs[-1] = _Out(nms, None, tile=o_pend)

    def static_items(h):
        def nmx(tag):
            return _Namer(tid, f"s_h{h}_{tag}")

        items = [(None, front(h, loads[0], sets[0], nmx("f0")))]
        for c in range(n_static):
            nxt = None
            if c + 1 < NCH:
                nxt = front(h, loads[c + 1], sets[(c + 1) % 2], nmx(f"f{c + 1}"))
            items.append((back(h, sets[c % 2], v_static[c], outs[c], nmx(f"b{c}"), lead_c=c), nxt))
        return items

    zero = _sb((C, H * ZF_CHUNKS * V), nm("zero")) if dynamic else None

    def first_load():
        loads[0].issue()
        load_constants()
        for ld in loads[1 : n_static if lead_skip else len(loads)]:
            ld.issue()
        if dynamic:
            nisa.memset(dst=zero, value=0.0, engine=nisa.engine.gpsimd)
        yield

    gens = [first_load(), level0_mask()]
    if lead is not None:
        gens.append(lead.setup())
    gens += [_pipe(static_items(h)) for h in range(H)]
    regs = {}
    if dynamic and valid_len is None:
        gens.append(_zero_test_gen(tid, qT, kT, gtok, n_static, regs, quads=chunks_per_iter == 4))
    elif dynamic and lead_skip:

        def first_loop_load():
            t0_ready.append(True)
            loads[n_static].issue()

        gens.append(
            _pairs_from_valid_len(
                tid,
                valid_len,
                n_static,
                NCH,
                regs,
                quads=chunks_per_iter == 4,
                own_tile=bool(own_count_tile),
                lead=(valid_start, t0_sb, first_loop_load),
            )
        )
    elif dynamic:
        gens.append(
            _pairs_from_valid_len(
                tid, valid_len, n_static, NCH, regs, quads=chunks_per_iter == 4, own_tile=bool(own_count_tile)
            )
        )
    _run(gens)

    order.reset()
    if dynamic:
        # ---- The chunks n_static .. n_eff - 1, two per loop iteration. ----
        # An odd count runs its last chunk in a 0-or-1-trip tail loop.
        _zero_fill(o, n_static, zero)
        if pad is not None:
            pad.loop_init(n_static * C)
        npair_reg = nisa.register_alloc()
        nisa.register_load(npair_reg, regs["npair"])
        ntail_reg = nisa.register_alloc()
        nisa.register_load(ntail_reg, regs["ntail"])
        if chunks_per_iter == 4:
            nquad_reg = nisa.register_alloc()
            nisa.register_load(nquad_reg, regs["nquad"])
        # t_sb is the first token of the next chunk to run, tf_sb the
        # first token of the chunk after the pair, clamped to the last
        # chunk (its steps 1-2 run and are not used when the loop ends),
        # and tq_sb the first token of the chunk pending in o_pend.
        t_sb = _sb((1, 1), nm("t_sb"), _I32)
        if lead_skip:
            # The first loop chunk, the one whose steps 1-2 ran in the static part.
            nisa.tensor_copy(dst=t_sb, src=t0_sb, engine=nisa.engine.vector)
        else:
            nisa.memset(dst=t_sb, value=n_static * C, engine=nisa.engine.vector)
        tf_sb = _sb((1, 1), nm("tf_sb"), _I32)
        tq_sb = _sb((1, 1), nm("tq_sb"), _I32)
        nisa.memset(dst=tq_sb, value=(n_static - 1) * C, engine=nisa.engine.vector)
        par = n_static % 2

        def pair_body(_):
            order.reset()
            nmp = _Namer(tid, "pair")
            _dma(tok_ap(o, (tq_sb, 0)), o_pend)
            nisa.tensor_scalar(
                dst=tf_sb,
                data=t_sb,
                op0=nl.add,
                operand0=2 * C,
                op1=nl.minimum,
                operand1=Tp - C,
                engine=nisa.engine.vector,
            )
            ld2 = loader(nmp, (t_sb, C), sets[1 - par]["v"], pad_rel=C)
            ld3 = loader(nmp, (tf_sb, 0), sets[par]["v"], pad_rel=2 * C)
            out1 = _Out(nmp, (t_sb, 0))
            out2 = _Out(nmp, None, tile=o_pend)
            pair_items = []
            for h in range(H):

                def nmx(tag, h=h):
                    return _Namer(tid, f"p_h{h}_{tag}")

                items = [
                    (
                        back(h, sets[par], sets[par]["v"], out1, nmx("b1"), lead_c="loop", loop=True),
                        front(h, ld2, sets[1 - par], nmx("f2"), loop=True),
                    ),
                    (
                        back(h, sets[1 - par], sets[1 - par]["v"], out2, nmx("b2"), lead_c="loop", loop=True),
                        front(h, ld3, sets[par], nmx("f3"), loop=True),
                    ),
                ]
                pair_items.append(items)
            _run_items(pair_items, head_groups)
            nisa.activation(dst=tq_sb, op=nl.copy, data=t_sb, scale=1.0, bias=float(C), engine=nisa.engine.scalar)
            nisa.activation(dst=t_sb, op=nl.copy, data=t_sb, scale=1.0, bias=float(2 * C), engine=nisa.engine.scalar)
            if pad is not None:
                pad.loop_advance(2 * C)

        def tail_body(_):
            order.reset()
            # The last chunk of an odd count. Its steps 1-2 ran in the last
            # pair (or in the static part), into set `par`.
            nmt = _Namer(tid, "tail")
            _dma(tok_ap(o, (tq_sb, 0)), o_pend)
            out1 = _Out(nmt, None, tile=o_pend)
            tail_items = []
            for h in range(H):
                item = (
                    back(h, sets[par], sets[par]["v"], out1, _Namer(tid, f"t_h{h}_b1"), lead_c="loop", loop=True),
                    None,
                )
                tail_items.append([item])
            _run_items(tail_items, head_groups)
            nisa.tensor_copy(dst=tq_sb, src=t_sb)

        def quad_body(_):
            # Four chunks per iteration: the steps and the order of two pair
            # iterations, with one loop barrier. Chunk c + j uses set
            # (par + j) % 2, so the next iteration starts on set par again.
            order.reset()
            nmq = _Namer(tid, "quad")
            _dma(tok_ap(o, (tq_sb, 0)), o_pend)
            nisa.tensor_scalar(
                dst=tf_sb,
                data=t_sb,
                op0=nl.add,
                operand0=4 * C,
                op1=nl.minimum,
                operand1=Tp - C,
                engine=nisa.engine.vector,
            )
            lds = [loader(nmq, (t_sb, j * C), sets[(par + j) % 2]["v"], pad_rel=j * C) for j in (1, 2, 3)]
            lds.append(loader(nmq, (tf_sb, 0), sets[par]["v"], pad_rel=4 * C))
            outs_q = [_Out(nmq, (t_sb, j * C)) for j in range(3)] + [_Out(nmq, None, tile=o_pend)]
            quad_items = []
            for h in range(H):

                def nmx(tag, h=h):
                    return _Namer(tid, f"q_h{h}_{tag}")

                items = []
                for j in range(4):
                    cur, nxt = sets[(par + j) % 2], sets[(par + j + 1) % 2]
                    items.append(
                        (
                            back(h, cur, cur["v"], outs_q[j], nmx(f"b{j + 1}"), lead_c="loop", loop=True),
                            front(h, lds[j], nxt, nmx(f"f{j + 2}"), loop=True),
                        )
                    )
                quad_items.append(items)
            _run_items(quad_items, head_groups)
            nisa.activation(dst=tq_sb, op=nl.copy, data=t_sb, scale=1.0, bias=float(3 * C), engine=nisa.engine.scalar)
            nisa.activation(dst=t_sb, op=nl.copy, data=t_sb, scale=1.0, bias=float(4 * C), engine=nisa.engine.scalar)
            if pad is not None:
                pad.loop_advance(4 * C)

        if chunks_per_iter == 4:
            nl.fori_loop(0, nquad_reg, quad_body)
        nl.fori_loop(0, npair_reg, pair_body)
        nl.fori_loop(0, ntail_reg, tail_body)
        _dma(tok_ap(o, (tq_sb, 0)), o_pend)

    _dma(s_fin.ap(pattern=[[V, K], [K * V, H], [1, V]]), S_all)
    if lead is not None:
        _dma(lead.out.ap(pattern=[[V, K], [K * V, H], [1, V]]), lead.tile)
        return o, s_fin, lead.out

    return o, s_fin


class _PadMask:
    """Zero the pad rows (the rows at or after ``valid_len``) of each loaded chunk.

    The scan contracts over the chunk's tokens: the gate prefix sums, the pair
    matrices with their causal masks, the inverse and the state update all
    multiply 0/1 matrices into token rows. A pad row that holds a NaN or Inf
    would reach every row of its chunk and the state through ``0 * NaN``. The
    pad rows of the loaded tiles therefore become exact zeros by predicated
    copies (a pad row with ``q = k = v = 0`` and ``g = -0.0`` leaves the state
    unchanged and has a zero output), so the scan's contract that pad rows are
    zero holds for any pad input.

    ``vtok[p] = valid_len - p``: row ``p`` of the token-major tiles of the chunk
    at token ``t`` is a pad row if ``vtok[p] <= t``. ``vchan = valid_len`` on
    every partition: column ``s`` of a channel-major tile is a pad row if
    ``s + t >= vchan``. In the loop, ``cur_tok`` and ``cur_chan`` hold them minus
    the first token of the loop's current chunk.
    """

    def __init__(self, tid, valid_len):
        nm = self.nm = _Namer(tid, "pad")
        vl_i = _sb((C, 1), nm("vl_i"), _I32)
        nisa.dma_copy(
            dst=vl_i,
            src=valid_len.ap(pattern=[[0, C], [1, 1]]),
            dge_mode=nisa.dge_mode.hwdge,
            engine=nisa.engine.scalar,
        )
        self.vchan = _sb((C, 1), nm("vchan"))
        nisa.tensor_copy(dst=self.vchan, src=vl_i, engine=nisa.engine.vector)
        negp = _sb((C, 1), nm("negp"))
        nisa.iota(dst=negp, pattern=[[0, 1]], offset=0, channel_multiplier=-1)
        self.vtok = _sb((C, 1), nm("vtok"))
        nisa.tensor_tensor(dst=self.vtok, data1=self.vchan, data2=negp, op=nl.add)
        self.ramp = _sb((C, C), nm("ramp"))
        nisa.iota(dst=self.ramp, pattern=[[1, C]], offset=0, channel_multiplier=0)
        self.cur_tok = _sb((C, 1), nm("cur_tok"))
        self.cur_chan = _sb((C, 1), nm("cur_chan"))

    def loop_init(self, t0):
        """Before the loop: the bases minus ``t0``, the first token of the first loop chunk."""
        nisa.tensor_scalar(dst=self.cur_tok, data=self.vtok, op0=nl.subtract, operand0=float(t0))
        nisa.tensor_scalar(dst=self.cur_chan, data=self.vchan, op0=nl.subtract, operand0=float(t0))

    def loop_advance(self, n):
        """At the end of a loop iteration that ran ``n`` tokens."""
        nisa.tensor_scalar(dst=self.cur_tok, data=self.cur_tok, op0=nl.subtract, operand0=float(n))
        nisa.tensor_scalar(dst=self.cur_chan, data=self.cur_chan, op0=nl.subtract, operand0=float(n))

    def predicates(self, nmx, t_static, t_rel):
        """The pad-row predicates of one chunk: ``[C, 1]`` for its token-major tiles and
        ``[C, C]`` for its channel-major ones. ``t_static`` is the chunk's first token,
        or None for a loop chunk at ``t_rel`` tokens after the reference chunk."""
        if t_static is not None:
            base_tok, base_chan, t = self.vtok, self.vchan, t_static
        else:
            base_tok, base_chan, t = self.cur_tok, self.cur_chan, t_rel
        pad_tok = _sb((C, 1), nmx("pad_tok"), nl.uint8)
        nisa.tensor_scalar(dst=pad_tok, data=base_tok, op0=nl.less_equal, operand0=float(t))
        pad_chan = _sb((C, C), nmx("pad_chan"), nl.uint8)
        nisa.tensor_scalar(
            dst=pad_chan, data=self.ramp, op0=nl.add, operand0=float(t), op1=nl.greater_equal, operand1=base_chan
        )
        return pad_tok, pad_chan

    def zero_v(self, pm, H, v3):
        """Zero the pad rows of ``v3 [C, H, V]`` (tokens on the partitions)."""
        pad_tok, _ = pm
        nisa.tensor_copy_predicated(dst=v3, src=0.0, predicate=pad_tok.ap(pattern=[[1, C], [0, H], [0, v3.shape[2]]]))

    def zero_head(self, isa, pm, h, g3, chan_tiles):
        """Zero the pad rows of head ``h``: ``g3[:, h]`` (-0.0) and the channel-major
        ``[K, H, C]`` tiles' ``[:, h]``, right before the head's step 1 reads them."""
        pad_tok, pad_chan = pm
        isa.tensor_copy_predicated(dst=g3[:, h, :], src=-0.0, predicate=pad_tok.ap(pattern=[[1, C], [0, g3.shape[2]]]))
        for x in chan_tiles:
            isa.tensor_copy_predicated(dst=x[:, h, :], src=0.0, predicate=pad_chan.ap(pattern=[[C, C], [1, C]]))


class _LeadState:
    """The state at the start of the last chunk that ran, for ``lead_state``.

    ``tile [K, H, V]`` starts as ``S0``. Before its state update each chunk
    copies the state into it (``stage``): a loop chunk always (every chunk
    that the loop runs holds a valid row), a static chunk ``c >= 1`` only if
    ``c * C < valid_len`` (``copy_pred``). After the scan, ``tile`` thus holds
    the state at the start of the last chunk with a valid row, or ``S0``.

    With ``valid_start``, a static chunk whose rows all lie before
    ``vs = clamp(valid_start, 0, valid_len)`` (``(c + 1) * C <= vs``,
    ``keep_pred``) puts the copy back into the state after its update
    (``keep``). Such a chunk has zero inputs, so its update is an identity
    step only if ``exp(0)`` is exactly 1 and ``S * 1 + 0`` keeps the bits;
    the copy back keeps the state bit for bit in any case (``-0.0`` too). The
    loop never runs a chunk whose rows all lie before ``vs``: it starts at the
    chunk of ``vs`` (or at chunk ``N_STATIC``).

    The predicates are ``[K, 1]`` uint8 tiles, built from loads of
    ``valid_len`` and ``valid_start`` on every partition (``setup``, a stage
    generator, so VectorE does not wait for the loads ahead of chunk 0).
    """

    def __init__(self, tid, valid_len, valid_start, n_static, K, H, V):
        nm = self.nm = _Namer(tid, "lead")
        self.tile = _sb((K, H, V), nm("S_lead"))
        self.out = nl.ndarray((H, K, V), dtype=_F32, buffer=nl.shared_hbm, name=nm("s_lead"))
        self.valid_len, self.valid_start, self.n_static, self.K, self.V = valid_len, valid_start, n_static, K, V
        self.copy_pred = {}
        self.keep_pred = {}
        self.ready = False

    def setup(self):
        """Build ``copy_pred`` and ``keep_pred``, a stage generator."""
        nm, K = self.nm, self.K
        vl_i = _sb((K, 1), nm("vl_i"), _I32)
        _dma(vl_i, self.valid_len.ap(pattern=[[0, K], [1, 1]]))
        vs_i = None
        if self.valid_start is not None:
            vs_i = _sb((K, 1), nm("vs_i"), _I32)
            _dma(vs_i, self.valid_start.ap(pattern=[[0, K], [1, 1]]))
        # The compares run once the loads have surely arrived.
        for _ in range(16):
            yield
        vl = _sb((K, 1), nm("vl"))
        nisa.tensor_copy(dst=vl, src=vl_i, engine=nisa.engine.vector)
        for c in range(1, self.n_static):
            pred = _sb((K, 1), nm(f"copy{c}"), nl.uint8)
            nisa.tensor_scalar(dst=pred, data=vl, op0=nl.greater, operand0=float(c * C), engine=nisa.engine.vector)
            self.copy_pred[c] = pred
        if vs_i is not None:
            # vs = min(max(valid_start, 0), valid_len). All values are small integers (exact in fp32).
            vs = _sb((K, 1), nm("vs"))
            nisa.tensor_copy(dst=vs, src=vs_i, engine=nisa.engine.vector)
            nisa.tensor_scalar(dst=vs, data=vs, op0=nl.maximum, operand0=0.0, engine=nisa.engine.vector)
            nisa.tensor_tensor(dst=vs, data1=vs, data2=vl, op=nl.minimum, engine=nisa.engine.vector)
            for c in range(self.n_static):
                pred = _sb((K, 1), nm(f"keep{c}"), nl.uint8)
                nisa.tensor_scalar(
                    dst=pred, data=vs, op0=nl.greater_equal, operand0=float((c + 1) * C), engine=nisa.engine.vector
                )
                self.keep_pred[c] = pred
        self.ready = True

    def _pred(self, pred):
        if not self.ready:
            raise RuntimeError("kda_prefill_v2: a lead_state predicate was used before setup() built it")
        return pred.ap(pattern=[[1, self.K], [0, self.V]])

    def stage(self, isa, h, S, lead_c):
        """Copy the state of head ``h`` into the staging tile, at the start of a chunk's back."""
        if lead_c == "loop":
            isa.tensor_copy(dst=self.tile[:, h, :], src=S, engine=nisa.engine.gpsimd)
        elif lead_c > 0:
            isa.tensor_copy_predicated(dst=self.tile[:, h, :], src=S, predicate=self._pred(self.copy_pred[lead_c]))

    def keep(self, isa, h, S, lead_c):
        """With ``valid_start``: undo the update of a static chunk whose rows all lie before it."""
        if lead_c != "loop" and self.valid_start is not None:
            isa.tensor_copy_predicated(dst=S, src=self.tile[:, h, :], predicate=self._pred(self.keep_pred[lead_c]))


class _FrontBackOrder:
    """Keep each head's steps 1-2 ahead of its steps 3-5, as scheduling hints.

    An instruction of steps 3-5 (the back) of a head issues after the
    latest instruction of steps 1-2 (the front) of the same head that the
    trace emitted before it. The front may still run early. The hint has
    no semaphore. Without it, the post-scheduler delays the sub-block
    matmuls of the front until the inverse of the back is done and then
    runs them in a burst of up to 35, which trips the TensorE activity
    throttle; its tie breaks follow the SBUF addresses, so the order, and
    the latency, change with the caller's live SBUF.
    """

    def __init__(self, n_heads):
        self.last = [None] * n_heads

    def reset(self):
        """Start a new scope (the static part, or a loop body)."""
        self.last = [None] * len(self.last)

    def isa(self, h, front):
        """``nisa`` for the front (``front=True``) or the back of head ``h``."""
        return _OrderedIsa(self, h, front)


class _OrderedIsa:
    """Forward ``nisa`` calls, recording the front or ordering the back."""

    def __init__(self, order, h, front):
        self.order, self.h, self.front = order, h, front

    def __getattr__(self, name):
        fn = getattr(nisa, name)

        def call(*args, **kwargs):
            ins = fn(*args, **kwargs)
            if not hasattr(ins, "depends_on"):
                return ins
            last = self.order.last
            if self.front:
                last[self.h] = ins
            elif last[self.h] is not None:
                ins.depends_on(last[self.h], kind=nisa.dependency.schedule)
            return ins

        return call


def _wide_wcat(dst, w8, nsub):
    """Build the 0/1 sum matrix of 16-token sub-blocks from that of 8-token ones.

    ``w8`` is ``wcat`` (``[C, C + NPAIR]``, 8-token sub-blocks); ``dst`` is
    ``[C, C + nsub * (nsub + 1) / 2]`` for 16-token sub-blocks:

    * The suffix of ``s`` in its 16-token sub-block is its 8-token suffix,
      plus all of the second 8-token half when ``s`` is in the first half.
    * The whole 16-token sub-blocks ``b'+1 .. b`` are the 8-token
      sub-blocks ``2b'+2 .. 2b+1``, the column of the 8-token pair
      ``(2b'+1, 2b+1)``.
    """

    def pair8(b, bp):
        return C + b * (b + 1) // 2 + bp

    # ScalarE builds it (it is idle at the start), in 2 * nsub + 1 ops.
    w8_free = w8.shape[1]
    d_free = dst.shape[1]
    for k in range(nsub):
        lo, hi = 16 * k, 16 * k + 8
        # First half: 8-token suffix plus the whole second half (a
        # per-partition bias).
        col = pair8(2 * k + 1, 2 * k)
        nisa.activation(dst=dst[:, lo:hi], op=nl.copy, data=w8[:, lo:hi], bias=w8[:, col : col + 1])
    # Second halves: the 8-token suffix, every 16 columns.
    nisa.tensor_copy(
        dst=dst.ap(pattern=[[d_free, C], [16, nsub], [1, 8]], offset=8),
        src=w8.ap(pattern=[[w8_free, C], [16, nsub], [1, 8]], offset=8),
        engine=nisa.engine.scalar,
    )
    for b in range(nsub):
        # The pairs (b', b) are the 8-token pairs (2b'+1, 2b+1), every
        # second column of that block.
        nisa.tensor_copy(
            dst=dst[:, C + b * (b + 1) // 2 : C + (b + 1) * (b + 2) // 2],
            src=w8.ap(pattern=[[w8_free, C], [2, b + 1]], offset=pair8(2 * b + 1, 1)),
            engine=nisa.engine.scalar,
        )


def _split_pairs(nmx, n_dyn, regs, quads=False):
    """Write ``npair = n_dyn >> 1`` and ``ntail = n_dyn & 1`` (``[1, 1]`` int32) into ``regs``.

    With ``quads`` (four chunks per loop iteration), also ``nquad = n_dyn >> 2``, and
    ``npair = (n_dyn >> 1) & 1``.
    """
    if quads:
        nquad = _sb((1, 1), nmx("nquad"), _I32)
        nisa.tensor_scalar(dst=nquad, data=n_dyn, op0=nl.right_shift, operand0=2)
        regs["nquad"] = nquad
    npair = _sb((1, 1), nmx("npair"), _I32)
    if quads:
        nisa.tensor_scalar(dst=npair, data=n_dyn, op0=nl.right_shift, operand0=1, op1=nl.bitwise_and, operand1=1)
    else:
        nisa.tensor_scalar(dst=npair, data=n_dyn, op0=nl.right_shift, operand0=1)
    ntail = _sb((1, 1), nmx("ntail"), _I32)
    nisa.tensor_scalar(dst=ntail, data=n_dyn, op0=nl.bitwise_and, operand0=1)
    regs["npair"] = npair
    regs["ntail"] = ntail


def _pairs_from_valid_len(tid, valid_len, n_static, NCH, regs, quads=False, own_tile=False, lead=None):
    """Write the loop count of a ``valid_len`` input into ``regs``, a stage generator.

    ``n_eff = min(max(ceil(valid_len / C), n_static), NCH)`` chunks run, so
    the pair loop runs ``(n_eff - n_static) >> 1`` pairs and the tail loop
    ``(n_eff - n_static) & 1`` chunks. The counts are ``[1, 1]`` int32 SBUF
    tensors in ``regs["npair"]`` and ``regs["ntail"]`` (``_split_pairs``). With ``own_tile``, the first
    operation writes a new tile and the loaded tile is only read.

    With ``lead = (valid_start, t0_sb, issue)``, the loop starts at chunk
    ``c0 = min(max(max(valid_start, 0) >> 7, n_static), n_eff)`` and runs
    ``n_eff - c0`` chunks: the generator writes ``min(c0, NCH - 1) * C`` (the
    first token of the first loop chunk, clamped to a chunk in range) into
    ``t0_sb``, then calls ``issue`` (the load of that chunk).
    """
    nmv = _Namer(tid, "vl")
    n = _sb((1, 1), nmv("n"), _I32)
    _dma(n, valid_len.ap(pattern=[[1, 1], [1, 1]]))
    if lead is not None:
        valid_start, t0_sb, issue = lead
        vs = _sb((1, 1), nmv("vs"), _I32)
        _dma(vs, valid_start.ap(pattern=[[1, 1], [1, 1]]))
    # The integer ops run once the load has surely arrived.
    for _ in range(16):
        yield
    src = n
    if own_tile:
        n = _sb((1, 1), nmv("cnt"), _I32)
    nisa.activation(dst=n, op=nl.copy, data=src, scale=1.0, bias=float(C - 1), engine=nisa.engine.scalar)
    nisa.tensor_scalar(dst=n, data=n, op0=nl.right_shift, operand0=C.bit_length() - 1)
    nisa.tensor_scalar(
        dst=n, data=n, op0=nl.maximum, operand0=n_static, op1=nl.minimum, operand1=NCH, engine=nisa.engine.vector
    )
    if lead is None:
        nisa.tensor_scalar(dst=n, data=n, op0=nl.subtract, operand0=n_static, engine=nisa.engine.vector)
    else:
        # c0 = min(max(max(vs, 0) >> 7, n_static), n_eff). The clamp at 0 comes first, so the shift only sees
        # values >= 0. All values are small integers (exact in the fp32 path of the arithmetic ops).
        c0 = _sb((1, 1), nmv("c0"), _I32)
        nisa.tensor_scalar(dst=c0, data=vs, op0=nl.maximum, operand0=0, engine=nisa.engine.vector)
        nisa.tensor_scalar(dst=c0, data=c0, op0=nl.right_shift, operand0=C.bit_length() - 1, engine=nisa.engine.vector)
        nisa.tensor_scalar(dst=c0, data=c0, op0=nl.maximum, operand0=n_static, engine=nisa.engine.vector)
        nisa.tensor_tensor(dst=c0, data1=c0, data2=n, op=nl.minimum, engine=nisa.engine.vector)
        nisa.tensor_tensor(dst=n, data1=n, data2=c0, op=nl.subtract, engine=nisa.engine.vector)
        nisa.tensor_scalar(
            dst=t0_sb, data=c0, op0=nl.minimum, operand0=NCH - 1, op1=nl.multiply, operand1=C, engine=nisa.engine.vector
        )
        issue()
    _split_pairs(nmv, n, regs, quads)


def _zero_test_gen(tid, qT, kT, gtok, n_static, regs, quads=False):
    """Find the chunks to run, as a stage generator for the static driver.

    A chunk whose ``q``, ``k`` and ``g`` are zero on every row and head
    leaves the state unchanged and has a zero output. The test runs on
    chunks ``n_static .. NCH - 1``. Each value enters an abs-max reduction,
    so ``-0.0`` and ``+0.0`` both count as zero and any other value (also a
    denormal) as nonzero. ``n_eff`` is one more than the last chunk with a
    nonzero value, at least ``n_static``. At its end the generator writes
    the loop counts of ``n_eff - n_static`` chunks into ``regs["npair"]``
    and ``regs["ntail"]`` (``_split_pairs``).

    The q and k loads use the Sync DMA queue and reduce on VectorE; the g
    loads use the ScalarE DMA queue and reduce on ScalarE. One queue's rate
    is limited, so two queues shorten the load time of the 17 MB. The
    reductions run in small pieces spread over the static chunks.
    """
    nmd = _Namer(tid, "zt")
    H, K, Tp = qT.shape
    NCH = Tp // C
    nchk = NCH - n_static
    c_lo = n_static
    # q and k: channel-major pieces of pq chunks, abs-max over the tokens
    # of each chunk: rqk[k, (tensor, head), chunk].
    pq = max(1, DET_ELEMS // (H * C))
    rqk = _sb((K, 2 * H, nchk), nmd("rqk"))
    # g: token-major, grows tokens of one head per partition, so partition
    # p of group gi holds chunk c_lo + cpg * gi + p // ppc:
    # rg[p, (group, head, piece)].
    grows = DET_ELEMS // K
    ppc = C // grows
    cpg = C // ppc
    ngrp = (nchk + cpg - 1) // cpg
    rg = _sb((C, ngrp * H * DET_G_SPLIT), nmd("rg"))
    if nchk % cpg:
        nisa.memset(dst=rg, value=0.0)
    qk_bufs = [_sb((C, DET_ELEMS), nmd("zbuf")) for _ in range(DET_BUFS)]
    g_bufs = [_sb((C, DET_ELEMS), nmd("gbuf")) for _ in range(DET_BUFS)]

    jobs = []
    for ti, x in enumerate((qT, kT)):
        for j0 in range(0, nchk, pq):
            pc = min(pq, nchk - j0)

            def dma(buf, x=x, j0=j0, pc=pc):
                nisa.dma_copy(
                    dst=buf.ap(pattern=[[DET_ELEMS, K], [pc * C, H], [1, pc * C]]),
                    src=x.ap(pattern=[[Tp, K], [K * Tp, H], [1, pc * C]], offset=(c_lo + j0) * C),
                    dge_mode=nisa.dge_mode.hwdge,
                    engine=nisa.engine.sync,
                )

            reds = []
            for i0 in range(0, pc, DET_QK_SPLIT):
                ni = min(DET_QK_SPLIT, pc - i0)

                def red(buf, ti=ti, j0=j0, pc=pc, i0=i0, ni=ni):
                    piece = buf.ap(pattern=[[DET_ELEMS, K], [pc * C, H], [C, ni], [1, C]], offset=i0 * C)
                    if is_trn2():
                        # Trn2 VectorE has no abs_max: take |x| in place on ScalarE, then the maximum.
                        nisa.activation(dst=piece, op=nl.abs, data=piece)
                        nisa.tensor_reduce(
                            dst=rqk[:, ti * H : (ti + 1) * H, j0 + i0 : j0 + i0 + ni],
                            op=nl.maximum,
                            data=piece,
                            axis=3,
                        )
                        return
                    nisa.tensor_reduce(
                        dst=rqk[:, ti * H : (ti + 1) * H, j0 + i0 : j0 + i0 + ni],
                        op=nl.abs_max,
                        data=buf.ap(pattern=[[DET_ELEMS, K], [pc * C, H], [C, ni], [1, C]], offset=i0 * C),
                        axis=3,
                    )

                reds.append(red)
            jobs.append((dma, reds))
    qk_jobs, jobs = jobs, []
    for gi in range(ngrp):
        nc = min(cpg, nchk - cpg * gi)
        npt = ppc * nc
        for h in range(H):

            def dma(buf, gi=gi, h=h, npt=npt):
                nisa.dma_copy(
                    dst=buf[0:npt, :],
                    src=gtok.ap(
                        pattern=[[grows * K, npt], [1, grows * K]],
                        offset=h * Tp * K + (c_lo + cpg * gi) * C * K,
                    ),
                    dge_mode=nisa.dge_mode.hwdge,
                    engine=nisa.engine.scalar,
                )

            reds = []
            w = DET_ELEMS // DET_G_SPLIT
            for i in range(DET_G_SPLIT):

                def red(buf, gi=gi, h=h, npt=npt, i=i):
                    col = (gi * H + h) * DET_G_SPLIT + i
                    if is_trn2():
                        # Trn2 has no activate2: |x| in place on ScalarE, then the maximum on VectorE.
                        piece = buf[0:npt, i * w : (i + 1) * w]
                        nisa.activation(dst=piece, op=nl.abs, data=piece)
                        nisa.tensor_reduce(dst=rg[0:npt, col : col + 1], op=nl.maximum, data=piece, axis=1)
                        return
                    nisa.activate2(
                        dst=buf[0:npt, i * w : (i + 1) * w],
                        op=nl.copy,
                        data=buf[0:npt, i * w : (i + 1) * w],
                        imm0=0.0,
                        imm1=0.0,
                        op0=nl.bypass,
                        op1=nl.bypass,
                        reduce_op=nl.abs_max,
                        reduce_res=rg[0:npt, col : col + 1],
                        reduce_cmd=nisa.reduce_cmd.reset_reduce,
                    )

                reds.append(red)
            jobs.append((dma, reds))

    def finish():
        # Weight each nonzero entry by one more than its chunk index, then
        # take the maximum over everything.
        wq_i = _sb((K, nchk), nmd("wq_i"), _I32)
        nisa.iota(dst=wq_i, pattern=[[1, nchk]], offset=c_lo + 1, channel_multiplier=0)
        wq = _sb((K, nchk), nmd("wq"))
        nisa.tensor_copy(dst=wq, src=wq_i)
        fq = _sb((K, 2 * H, nchk), nmd("fq"))
        nisa.scalar_tensor_tensor(
            dst=fq,
            data=rqk,
            op0=nl.not_equal,
            operand0=0.0,
            op1=nl.multiply,
            operand1=wq.expand_dim(1).broadcast(1, 2 * H),
        )
        wg_i = _sb((C, ngrp), nmd("wg_i"), _I32)
        nisa.iota(dst=wg_i, pattern=[[C, ngrp]], offset=0, channel_multiplier=1)
        nisa.tensor_scalar(dst=wg_i, data=wg_i, op0=nl.right_shift, operand0=ppc.bit_length() - 1)
        nisa.tensor_scalar(dst=wg_i, data=wg_i, op0=nl.add, operand0=c_lo + 1)
        wg = _sb((C, ngrp), nmd("wg"))
        nisa.tensor_copy(dst=wg, src=wg_i)
        fg = _sb((C, ngrp, H * DET_G_SPLIT), nmd("fg"))
        nisa.scalar_tensor_tensor(
            dst=fg,
            data=rg.reshape((C, ngrp, H * DET_G_SPLIT)),
            op0=nl.not_equal,
            operand0=0.0,
            op1=nl.multiply,
            operand1=wg.expand_dim(2).broadcast(2, H * DET_G_SPLIT),
        )
        w2 = _sb((C, 2), nmd("w2"))
        nisa.tensor_reduce(dst=w2[:, 0:1], op=nl.maximum, data=fq, axis=(1, 2))
        nisa.tensor_reduce(dst=w2[:, 1:2], op=nl.maximum, data=fg, axis=(1, 2))
        w1 = _sb((C, 1), nmd("w1"))
        nisa.tensor_reduce(dst=w1, op=nl.maximum, data=w2, axis=1)
        # The maximum over the partitions: an exact PE transpose, then a
        # free-axis reduction.
        ps_w = _ps((1, C), nmd("ps_w"))
        nisa.nc_transpose(dst=ps_w, data=w1)
        n_f = _sb((1, 1), nmd("n_f"))
        nisa.tensor_reduce(dst=n_f, op=nl.maximum, data=ps_w, axis=1)
        # The count of loop chunks, max(n_f, n_static) - n_static. The
        # values are small integers, so the conversion to int32 is exact.
        n_dyn = _sb((1, 1), nmd("n_dyn"), _I32)
        nisa.tensor_scalar(
            dst=n_dyn, data=n_f, op0=nl.maximum, operand0=float(n_static), op1=nl.subtract, operand1=float(n_static)
        )
        _split_pairs(nmd, n_dyn, regs, quads)

    g_jobs = jobs
    # The first DET_BUFS loads of each queue issue at once (the Sync ones
    # after the static chunk loads). The load into a buffer that a
    # reduction frees issues right after that reduction: on Sync it waits
    # only on the Sync queue (no chunk load follows it there before the
    # loop), and on ScalarE it follows the ScalarE reduction itself. The
    # reductions of load j start at round DET_R0 + DET_GAP * j, after the
    # data has arrived (each 2 MB load takes time on its queue).
    events = {}
    for jobs_q, pool in ((qk_jobs, qk_bufs), (g_jobs, g_bufs)):
        for j, (dma, reds) in enumerate(jobs_q):
            buf = pool[j % DET_BUFS]
            if j < DET_BUFS:
                events.setdefault(0, []).append(lambda dma=dma, buf=buf: dma(buf))
            r0 = DET_R0 + DET_GAP * j
            for i, red in enumerate(reds):
                events.setdefault(r0 + i, []).append(lambda red=red, buf=buf: red(buf))
            if j + DET_BUFS < len(jobs_q):
                nxt = jobs_q[j + DET_BUFS][0]
                events.setdefault(r0 + len(reds) - 1, []).append(lambda dma=nxt, buf=buf: dma(buf))
    last = max(events)
    events.setdefault(last + 8, []).append(finish)
    for r in range(last + 9):
        for act in events.get(r, ()):
            act()
        yield


ZF_CHUNKS = 4  # output chunks of zeros per DMA


def _zero_fill(o, n_static, zero):
    """Write zeros to the output rows of chunks ``n_static ..``.

    ``zero`` is a ``[C, H * ZF_CHUNKS * V]`` zero tile. The loop rewrites
    the rows of each chunk it runs after these writes.
    """
    H, Tp, V = o.shape
    NCH = Tp // C
    chunks_per_dma = ZF_CHUNKS
    for c0 in range(n_static, NCH, chunks_per_dma):
        nc = min(chunks_per_dma, NCH - c0)
        _dma(
            o.ap(pattern=[[V, C], [Tp * V, H], [C * V, nc], [1, V]], offset=c0 * C * V),
            zero.ap(pattern=[[H * chunks_per_dma * V, C], [nc * V, H], [V, nc], [1, V]]),
            priority=3,
        )
