# SPDX-License-Identifier: Apache-2.0
"""NKI flash kernel for Kimi-K3 MLA with shared latent keys and values.

The kernel bounds an FP8 pool at 4 GiB and owns its transpose identities.
Options: ``mx_values`` (MX value matmuls in the FP8 cache steps), ``w_v``
(the value projection before the DCP merge) and ``fp8_query="mx"`` (the
MXFP8 latent query of the MLA block flash). ``mx_values=False``, no ``w_v``
and ``fp8_query="split"`` (the defaults) give the base kernel. Make a later
kernel change in a new copy of the file.
Local change: the fp32 transpose identity shares the SBUF tile of the
two-byte identity (``_Fp32View``). neuronx-cc 2.28 can put an fp32 identity
tile of its own in the dynamic DMA scratch, and the rank then hangs. The
values and the results do not change.

This kernel is the absorbed (latent) variant of the Kimi-K3 MLA flash
attention kernel of the kernel library. The heads exist on the query
side only. All heads attend the same keys (the slab rows ``[c | k_rot]``)
and the same values (the latent part ``c``). ``prefill/mla.py`` ``flash_partial``
has the math and the input contract.

Changes against the library kernel:

* The cache rows are the row-major latent slab rows ``[c | k_rot]``, the
  storage of record. The kernel reads them directly (no copy, no second
  layout). TensorE transposes each key block on chip into the
  feature-major operand of the score matmul. The value operand is the
  key-major block itself.
* The cache is a page pool ``[NP, 128, R + DR]`` with a block table
  ``[1, NBT]``. The table gives the physical page of each logical page of
  128 slots. For each block, the kernel loads the 16 table entries of the
  block, then one DMA for each page reads the page through the entry. A
  contiguous slab is the case of the identity table.
* The cache block loop has a runtime trip count. The runtime input is
  ``prior_tokens``, the count of the rank-local slots before the segment.
  The kernel computes ``n_blocks = ceil(prior_tokens / 2048)`` on the
  device, and streams the first ``n_blocks`` blocks of 2048 slots only. An
  early segment thus reads a short prefix of the cache. The marker still
  hides the slots of the last block that are not in the prior. With the
  MX values (below), a last block whose second 1024-key step holds no
  prior slot runs its first step only (the half tail), in a tail loop of
  its own: a fully hidden step is the identity, so the result is the same.
  With the short-prior variants (``prior_variants``) and an odd block
  count the last block runs only its 512-key steps that hold prior slots,
  if they are 1 or 2 (``TAIL_STEPS``): the group loop has a variant for
  each count. A view of one block (a prior under 2048 slots)
  has variants with no block loop: the block stays in buffer set 0 for
  all groups, and only its steps that hold prior slots run.
* The loop order is head group outer, cache block inner. One block load
  and one block transpose serve ``G`` heads and every query tile.
* The own keys of the call are a separate key set with a position bias
  (``cbias``), not the chunk-local causal pattern. Under DCP the own keys
  of a rank are not the query positions.

Kept from the library kernel: the 2048-key blocks, the 512-key
softmax steps, the pre-scale marker, the online softmax, and the
``(acc, m, l)`` result.

Loops. The head-group loop is a rolled ``nki.experimental.fori_loop`` with
a static bound. The cache blocks run in a rolled loop over pairs of
blocks and a 0-or-1-trip tail loop, both with register bounds
(``unroll=False``). The loops keep their offsets in SBUF counters, and the
DMAs use ``scalar_offset``. The instruction count thus does not change
with ``H`` or with the cache size. The loops inside one block (key steps,
heads of a group, query tiles) are short static Python loops, and they
index SBUF with compile-time offsets.

Schedule. A key step (512 keys of one 128-row query tile) is TensorE
bound: 5 score passes (4 latent tiles and the rope tile, whose spare
partition carries the marker) and 4 value passes.

* The step scratch rotates over ``NBUF`` slots, so consecutive steps do
  not wait for each other through a shared buffer.
* The softmax reads the scores in PSUM; the exponential applies the
  scale. p^T is one SBUF-to-SBUF DMA transpose per step.
* The cache blocks alternate between two buffer sets: the next block
  loads, and its keys are transposed between the steps of the current
  block. With the half tail (below), a load past the last block of the
  prior loads block 0 instead, so a group's last block-loop body leaves
  the next group's first block packed in set 0.
* The own keys run first in a group and set the state of every column.
  With the MX values (below), a group's end runs the next group's own-key
  steps while it stores: the next group's queries (loaded during the
  group's block loops) convert column by column, and a column's own-key
  step waits only for the store of its own heads' columns. Otherwise a
  group starts with its own-key steps, and its end prepares the next
  group's queries and first block while it stores.

FP8 cache. With ``fp8_matmul`` the cache rows stay FP8 in SBUF. A matmul
with the bf16 identity transposes them exactly into the bf16 keys of the
score matmuls. The weights of a step are rounded to FP8 after their DMA
transpose (bf16), and double-row matmuls give the step's value sum.
With ``fp8_query="split"`` (the default) the latent query is the sum of
two e4m3 parts, ``hi = e4m3(q)`` and ``lo = e4m3(q - hi)``, and each part
contracts all 512 latent features with the FP8 keys in one MX matmul
(unit scales): a 32-bit TensorE transpose of a key tile gives partition c
the latent features 4c .. 4c + 3, the MX packing, and the queries take
the same packing (feature-major queries load four rows a partition,
token-major ones transpose as 32-bit words). With ``fp8_query="mx"`` the latent query is
the hi part only, in MXFP8 (an E8M0 scale a 32 features of a query, from
``nisa.quantize_mx``), one MX matmul with the keys. With ``fp8_query=True`` the
queries and the latent keys are FP8, and the latent part of a score takes
two double-row matmuls; with False the queries stay bf16.

MX values (``mx_values``, with the split query). The value sum of 512
keys is one MX matmul too. Its contraction packs 4 keys in an element, so both operands
hold 4 keys a partition: the score columns of a 512-key group g of a
block are ordered so that columns 512 g + 4 w .. 512 g + 4 w + 3 are the
keys at partition w of the row tiles 4 g .. 4 g + 3 (``mx_perm``: the
copies of the key transposes and of the marker read in that order). The
exponential writes the weights in e4m3 (rounding the fp32 values, as the
GPU), one DMA transpose of the weights viewed as fp32 gives p^T as 32-bit
words of 4 keys (the MX stationary), and a ScalarE copy per group packs
the value rows the same way (``vmx``). The MX matmul rounds each pair and
then each quad of products to 11 bits before its fp32 accumulation
(~2e-4 of the sum); with the 4-bit weights it is as accurate as the GPU's
FP8 attention on the checkpoint's 24 MLA layers (``mla_flash_latent``).

P prescale (``p_prescale``, with the FP8 value matmuls). The weights of a
cache step are relative to the running max of the column, which includes
the own keys (they run first) and every earlier key. At a long prior much
of the softmax mass then sits far below that max, where e4m3 has
subnormals or rounds to 0. The prescale rounds ``2^8 p`` instead (at most
256, so below 448), and undoes the 2^8 once: with the MX values in the
E8M0 scale of the value matmuls and in the row sum, so ``(acc, m, l)``
keep their meaning; with the double-row values in the casts of p^T, the
own keys' acc copy and the group end (``m - 8 ln 2`` and ``256 l``), so
the state stays an exact softmax state with the same ``acc / l`` and
``m + ln l``.

The slots at or after ``prior_tokens`` of an FP8 pool may hold any bytes:
the rows load with gathers that give a hidden row an index past the pool,
which the DMA skips. The gather's dynamic index (a row or page index scaled
by the row stride) is formed wider than 32 bits, so a gather reaches the
whole pool, at or over 4 GiB (measured byte-exact on Trn3 for the row gather,
the page gather and the skipped hidden-row index). A large *static* offset is
the limit instead: the DGE base-address immediate is 32-bit, so an FP8
``cache`` must be its own tensor (not a view into a larger allocation) at
offset 0, and the kernel adds no static offset over 4 GiB. The row count
stays at most 2^24 so the index is exact in the fp32 arithmetic that forms
it; a pool may be any size up to that bound.

The FP8 value matmuls halve the value passes, so VectorE and ScalarE
would bound a 512-key step. A cache step therefore has 1024 keys (two
PSUM banks of scores, one max, one acc update), and the steps of a pair
of blocks run as one software pipeline: the FP8 rounding (half on
ScalarE, half on VectorE) and the value matmuls of a step run
``MM8_LAG - 1`` and ``MM8_LAG`` steps after its softmax. Schedule
dependencies keep each engine's issue order as emitted, and each chain
has its own PSUM tiles; a step then waits only for its own inputs, and
TensorE runs at its bound (14 matmuls of 512 columns a step).
With MX values a step is 8 matmuls of 512 columns (4 MX scores, 2 rope,
2 MX values) and no rounding stage; the value matmuls run ``MXV_LAG``
steps after the softmax. The scores of a step reuse the PSUM banks of the
step two before, so the chain max (VectorE), exponential (ScalarE, one
per bank, alpha after them) bounds the pace: the MX score matmuls come
before the rope ones, the row sums update once a round of columns, and
the key transposes and value packing of the next block run in small
pieces between the steps.

No prior. A call with ``prior_tokens = 0`` streams no cache block (the
segment path of a prompt that fits one segment). Such a call skips all of
the above and runs ``_own_only``: one static pipeline over the heads
with the own keys only, no loop inside, and static DMA addresses. The
compiler has no runtime branch, so the two paths are loops of 0 or 1
trips; the results of the two paths are the same.

Query layout. The score matmuls take the queries feature-major. With
``q_layout="token"`` the queries are ``[Sq, H, R]`` and TensorE
transposes them on chip. With ``q_layout="feature"`` the caller gives
``[H, R, Sq]``: the queries load with DMAs straight into the score
operand, and no query transpose runs. Both layouts give the same result.

Value projection. With ``w_v`` (each head's ``W_V``, the value rows of
``kv_b_proj``) the partial is ``acc @ W_V^T``, stored feature-major
(``[H, VD, Sq]``), so the cross-rank merge moves ``VD`` values a query
instead of ``R``. A call with a prior projects at each group end: fp32
TensorE transposes of the group's acc, one bf16 rounding, and bf16
matmuls with the group's ``W_V`` as the stationary. A call with no prior
projects the own keys' value rows instead (bf16 per key and head), and
its value matmuls take them as the stationary.
"""

from __future__ import annotations

import math
import weakref

import nki
import nki.experimental as nex
import nki.isa as nisa
import nki.language as nl
from nki.compiler._builtins import get_backend

P = 128  # partition tile
BLOCK = 2048  # cache keys for each block
KT = 512  # keys for each softmax step (one fp32 PSUM bank)
KT_MM8 = 2 * KT  # keys for each cache step with the FP8 value matmuls (two PSUM banks)
SUBS = BLOCK // P
NSTEP = BLOCK // KT
TP_BANK = 2 * KT // P  # bf16 transposes that share one PSUM bank
NBUF = 4  # key steps in flight, each with its own scratch slot
OWN_AHEAD = 3  # own-key steps whose softmax runs ahead of the value matmuls
# The group-end pipeline of the MX values (``own_at_end``):
CONV_AHEAD = 6  # query columns whose split-query conversion runs ahead of their own-key steps
OWN_COPY_DVE = (3,)  # query tiles whose own-key acc copy runs on VectorE (the rest on ScalarE)
# Heads of one acc store DMA. Every dynamic-offset DMA of the group body takes descriptor
# registers, and a program that inlines many calls has few.
GRP_HEADS = 4
K_AFTER = 16  # first step of a block that the next block's key transposes follow
TAIL_STEPS = (1, 2)  # tail-block step counts with their own group-loop variant
MM8_LAG = 3  # cache steps that the FP8 value matmuls of a step run behind its softmax
MXV_LAG = 3  # the same with the MX values (no rounding stage)
N_P8 = 4  # slots of the FP8 weights of the MX-value steps (read by their DMA transpose only)
N_PTW = MXV_LAG + 1  # slots of their p^T words (read by the value matmuls, MXV_LAG steps later)
# The last steps of a block-loop pipeline whose p^T words come from TensorE transposes instead of the DMA
# transpose: the pipeline drain then waits for them, which is shorter than the DMA latency.
PE_DRAIN = 2
# The last own-key steps of a group end whose p^T comes from TensorE transposes (``own_drain_transpose``).
OWN_PE_DRAIN = 2
MX_PACK = 4  # FP8 values in one element of an MX matmul operand (float8_e4m3fn_x4)
MX_UNIT = 127  # the E8M0 code of the MX scale 2^0
PS_LOG2 = 8  # log2 of the P prescale (p_prescale): e4m3(2^8 p) <= 256, below the e4m3 maximum of 448
PS_SCALE = float(2**PS_LOG2)
PS_LN = PS_LOG2 * math.log(2.0)
OWN_QS = 4  # heads of transposed queries in flight (no-prior path)
OWN_ACC = 6  # heads of acc staging in flight (no-prior path)
OWN_QL = 4  # query loads in flight (no-prior path)
OWN_BUF = 7  # unit scratch slots (no-prior path)
OWN_LEAD = 4  # units whose softmax runs ahead of the value matmuls (no-prior path)
OWN_MID = 2  # units whose p^T runs ahead of the value matmuls (no-prior path)
OWN_WV = 4  # heads of W_V and of projected own values in flight (no-prior path with w_v)
# The static DMAs of the setup and of the no-prior path: hardware
# descriptor generation on the Sync engine, which has no other work.
OWN_DMA = dict(dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
# DMA QoS classes (NeuronCore-v4, lower is higher; the default is 2): the p^T transposes of the steps, which the value
# matmuls wait for, and the next group's staged loads, which only the group end reads.
PT_PRIORITY = 0
STAGE_PRIORITY = 3
# The pair body's staged loads issue one at a time, after block-0 steps STAGE_STEP0 + j * STAGE_STRIDE.
STAGE_STEP0 = 4
STAGE_STRIDE = 5


def _dma_mode(src, dst):
    """Give ``OWN_DMA`` for a DMA that keeps the dtype, else the default DGE.

    The hardware DGE cannot convert dtypes (NCC_IBIR098), and an FP8 input
    loads into a bf16 tile.
    """
    return OWN_DMA if src.dtype == dst.dtype else {}


OWN_PV = 4  # PSUM banks of the value matmuls (no-prior path)

_F32 = nl.float32
_FP8_DTYPES = (nl.float8_e4m3fn, nl.float8_e4m3, nl.float8_e5m2)

# The calls of this kernel in each trace, keyed by the trace context (see _new_trace_id).
_TRACE_CALLS = weakref.WeakKeyDictionary()


class _Namer:
    """Give each buffer a name that is unique in one kernel trace.

    The trace frontend needs a unique name for each op. Create a new
    ``_Namer`` at the start of each call of a loop body. The body then gives
    the same names on each call, and the number of names stays bounded.
    """

    def __init__(self, trace_id, scope):
        self.prefix = f"f{trace_id}_{scope}"
        self.n = 0

    def __call__(self, name):
        self.n += 1
        return f"{self.prefix}_{name}_{self.n}"


class _EngineOrder:
    """Keep the issue order of instructions on each engine, as emitted.

    Each instruction recorded under an engine key gets a schedule
    dependency on the previous one under that key, so the compiler issues
    them in program order on that engine (no hardware wait).
    """

    def __init__(self):
        self.last = {}

    def __call__(self, key, ins):
        if hasattr(ins, "depends_on"):
            prev = self.last.get(key)
            if prev is not None:
                ins.depends_on(prev, kind=nisa.dependency.schedule)
            self.last[key] = ins
        return ins


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


class _Fp32View:
    """Give the fp32 view of columns ``P .. 3 P`` of a two-byte ``[P, 3 P]`` SBUF tile.

    ``view[0:k, 0:n]`` is the fp32 access pattern of partitions ``0 .. k``
    and fp32 columns ``0 .. n`` of that byte range. The kernel keeps its fp32
    transpose identity in such a tile, after its two-byte identity (see
    ``mla_flash_latent``).

    Example:
        >>> # id32 = _Fp32View(id_all); _pe_transpose(dst, data, id32)
    """

    def __init__(self, tile):
        self.tile = tile

    def __getitem__(self, index):
        rows, cols = index
        assert not rows.start and not cols.start, "the fp32 view starts at partition 0 and column 0"  # noqa: S101
        return self.tile.ap(pattern=[[3 * P // 2, rows.stop], [1, cols.stop]], offset=P // 2, dtype=_F32)


def _pe_transpose(dst, data, ident):
    """Transpose ``data`` [K, n] into the PSUM tile ``dst`` [n, K] on TensorE, with the identity ``ident``.

    This is the instruction of ``nc_transpose`` on TensorE (transpose mode,
    bit exact), with a kernel-owned identity in ``data``'s dtype instead of
    NKI's shared one. Transpose mode reads the identity: each output element
    takes the input element where the identity is nonzero. The shared
    identity has a fixed SBUF address that the allocator does not reserve,
    so in a program with several calls it can give those bytes to a kernel
    tensor, and the transposes then read that tensor's values.
    """
    k = data.shape[0]
    return nisa.nc_matmul(dst=dst, stationary=data, moving=ident[0:k, 0:k], is_transpose=True, accumulate=False)


QUAD = 4  # consecutive key rows interleaved per feature in a key-quad cache page


def _quad_shuffle(dst, src, part_stride, offset, r, engine):
    """Turn a key-quad latent tile (a word = 4 keys of one feature) into the word-key order
    (a word = 4 features of one key), within each partition.

    Vendored 4x4 byte transpose of each 16-byte block. ``src`` holds ``r`` latent features: its byte
    ``4 f + i`` is feature ``f`` (0 .. r - 1) lane ``i`` (one of 4 consecutive keys). ``dst`` (``[P, 4 r]``
    uint8, contiguous) byte ``16 c + 4 i + d`` takes ``src`` byte ``16 c + 4 d + i``, so ``dst`` word
    ``(c, i)`` (bytes ``16 c + 4 i .. + 3``) holds features ``4 c .. 4 c + 3`` of lane ``i``. The word
    transposes then read ``dst`` as 32-bit words. This is the only call site, so a shared library helper can
    replace it in one place.
    """
    return nisa.tensor_copy(
        dst=dst,
        src=src.ap(pattern=[[part_stride, P], [QUAD * QUAD, r // QUAD], [1, QUAD], [QUAD, QUAD]], offset=offset),
        engine=engine,
    )


def _own_only(
    nm,
    q_c,
    q_r,
    kc_rows,
    kc_fm,
    b_sb,
    acc_out,
    m_out,
    l_out,
    scale,
    G,
    widths,
    id_dt,
    id32,
    q_layout="token",
    w_v=None,
):
    """Attend the own keys only, for every head: the path of a call with no prior.

    With ``prior_tokens = 0`` the call streams no cache block, and the own
    keys are the only keys. This path has no loop: the heads run in one
    static stream, so no loop barrier drains the engines, and the DMAs
    have static addresses.

    The work runs in units of ``QB`` query tiles of one head (a whole head
    at ``Kc = 128``): the scores of a unit fill one PSUM bank, so one
    VectorE pass adds the bias and one finds the row maxima of the unit.
    A unit has three stages, which run ``OWN_LEAD``, ``OWN_MID`` and 0
    units ahead of the stores:

    * the scores and the softmax (TensorE, VectorE, ScalarE);
    * p^T by TensorE transposes, with one copy for the unit;
    * the value matmuls and the copies of acc, two PSUM banks a copy.

    The query rows load two heads per DMA, ``OWN_QL`` loads ahead, and
    the queries of a head are transposed (bf16 TensorE transposes, up to
    eight in one PSUM bank) ``OWN_QS - 1`` heads before its scores. Each
    head stores its acc after its last unit, and each group of ``G``
    heads stores m and l. The math of a step is the own-key step of the
    prior path, so the result is the same.

    With ``q_layout="feature"``, each head of the queries loads with two
    contiguous DMAs straight into its ``q_fm`` slot, and no query
    transpose runs.

    With ``w_v``, a head's own value rows are first projected with its
    ``W_V`` (four matmuls, rounded to the operand dtype). Its value
    matmuls then take the projected values as the stationary and p^T as
    the moving operand, and give its acc transposed, ``[VD, Sq]``.

    Args:
        nm: The ``_Namer`` of the call.
        q_c: ``[Sq, H, R]`` latent queries (HBM), or ``[H, R, Sq]`` with
            ``q_layout="feature"``.
        q_r: ``[Sq, H, DR]`` rope queries (HBM), or ``[H, DR, Sq]`` with
            ``q_layout="feature"``.
        kc_rows: ``[P, n_kc, D]`` own key rows (SBUF).
        kc_fm: ``[P, NF, Kc]`` own keys, feature-major (SBUF), or a pair of
            ``[P, NF - 1, Kc]`` latent tiles and ``[P, 1, Kc]`` rope tile.
        b_sb: ``[P, NQT, Kc]`` fp32 own-key bias (SBUF).
        acc_out: ``[H, Sq, R]`` fp32 output (HBM), or ``[H, VD, Sq]`` with ``w_v``.
        m_out: ``[H, Sq, 1]`` fp32 output (HBM).
        l_out: ``[H, Sq, 1]`` fp32 output (HBM).
        scale: The softmax scale.
        G: Heads of one m and l store.
        widths: Rows of each feature tile of the contraction.
        id_dt: ``[P, P]`` identity in the operand dtype (SBUF).
        id32: ``[P, P]`` fp32 identity (SBUF).
        q_layout: ``"token"`` or ``"feature"``, the layout of the queries.
        w_v: ``[H, P, R // P, VD]`` value projection (HBM, the kernel's
            ``w_v``), or None.
    """
    feature = q_layout == "feature"
    if feature:
        H, R, Sq = q_c.shape
        DR = q_r.shape[1]
    else:
        Sq, H, R = q_c.shape
        DR = q_r.shape[2]
    NQT = Sq // P
    NF = len(widths)
    kc_parts = kc_fm if isinstance(kc_fm, tuple) else (kc_fm,)
    Kc = kc_parts[0].shape[2]
    n_kc = Kc // P
    dt = kc_rows.dtype
    wv = w_v is not None
    # A head's acc row: R latent values, or VD values with the value projection.
    VD = w_v.shape[3] if wv else R
    NRW = R // P

    def kc_tile(f, w, k0=0, k1=None):
        """Give rows 0 .. w of feature tile ``f`` of the own keys ``k0 .. k1``."""
        k1 = Kc if k1 is None else k1
        for part in kc_parts:
            if f < part.shape[1]:
                return part[0:w, f, k0:k1]
            f -= part.shape[1]
        raise AssertionError("feature tile out of range")

    ncol = G * NQT
    # Query tiles of one unit: their scores fill one fp32 PSUM bank.
    QB = max(qb for qb in range(1, NQT + 1) if NQT % qb == 0 and qb * Kc <= KT)
    units = [(j, qt0) for j in range(H) for qt0 in range(0, NQT, QB)]
    n_units = len(units)

    # -m and l of every (head, query tile) column, column h * NQT + qt.
    nm_all = nl.ndarray((P, H * NQT), dtype=_F32, buffer=nl.sbuf, name=nm("o_nm"))
    l_all = nl.ndarray((P, H * NQT), dtype=_F32, buffer=nl.sbuf, name=nm("o_l"))
    m_st = nl.ndarray((P, P), dtype=_F32, buffer=nl.sbuf, name=nm("o_m"))
    st_t = nl.ndarray((P, 2, P), dtype=_F32, buffer=nl.sbuf, name=nm("o_st"))
    # The query rows of OWN_QL loads of QH heads each, in a ring: a load
    # reads QH * R contiguous values of a row.
    QH = max(qh for qh in range(1, 3) if H % qh == 0)
    n_loads = H // QH
    if not feature:
        qc_rows = nl.ndarray((P, OWN_QL, NQT, QH * R), dtype=dt, buffer=nl.sbuf, name=nm("o_qc"))
        qr_rows = nl.ndarray((P, OWN_QL, NQT, QH * DR), dtype=dt, buffer=nl.sbuf, name=nm("o_qr"))
    # The feature-major queries of OWN_QS heads: tile (f, qt) at column
    # (f * NQT + qt) * P.
    q_fm = nl.ndarray((P, OWN_QS, NF * NQT * P), dtype=dt, buffer=nl.sbuf, name=nm("o_qfm"))
    # A head's acc: [P, NQT, R] (rows of the query tiles), or with w_v acc^T [VD, Sq].
    acc_shape = (P, OWN_ACC, Sq) if wv else (P, OWN_ACC, NQT, R)
    acc_st = nl.ndarray(acc_shape, dtype=_F32, buffer=nl.sbuf, name=nm("o_acc"))
    if wv:
        # W_V of OWN_WV heads (wv_sb[c, s, k * VD + d] = W_V[h, d, k * P + c]), and the projected own
        # values of OWN_WV heads (vo_sb[p, s, jk, d], own key jk * P + p).
        wv_sb = nl.ndarray((P, OWN_WV, NRW * VD), dtype=dt, buffer=nl.sbuf, name=nm("o_wv"))
        vo_sb = nl.ndarray((P, OWN_WV, n_kc, VD), dtype=dt, buffer=nl.sbuf, name=nm("o_vo"))
    sc = nl.ndarray((P, OWN_BUF, QB * Kc), dtype=_F32, buffer=nl.sbuf, name=nm("o_sc"))
    # p in fp32: its fp32 transposes share one PSUM tensor with the stats
    # transposes. The copy of p^T rounds to the operand dtype.
    pb = nl.ndarray((P, OWN_BUF, QB * Kc), dtype=_F32, buffer=nl.sbuf, name=nm("o_p"))
    ptb = nl.ndarray((P, OWN_BUF, QB * Kc), dtype=dt, buffer=nl.sbuf, name=nm("o_pt"))
    # PSUM: two banks for the query transposes, one for the scores, one for
    # p^T and the stats transposes of a group, and OWN_PV for the value
    # matmuls. This fills the 8 banks; one more would spill a bank.
    # Feature-major queries need no transpose banks.
    if not feature:
        ps_tq = nl.ndarray((P, 2, TP_BANK * P), dtype=dt, buffer=nl.psum, name=nm("o_ps_tq"))
    ps_sc = nl.ndarray((P, 1, KT), dtype=_F32, buffer=nl.psum, name=nm("o_ps_sc"))
    ps_pt = nl.ndarray((P, 1, KT), dtype=_F32, buffer=nl.psum, name=nm("o_ps_pt"))
    ps_pv = nl.ndarray((P, OWN_PV, KT), dtype=_F32, buffer=nl.psum, name=nm("o_ps_pv"))

    def load_queries(n):
        """Load the query rows of the heads n * QH .. n * QH + QH into ring slot n % OWN_QL."""
        b = n % OWN_QL
        nisa.dma_copy(
            dst=qc_rows[0:P, b, 0:NQT, 0 : QH * R],
            src=q_c.ap(pattern=[[H * R, P], [P * H * R, NQT], [1, QH * R]], offset=n * QH * R),
            **_dma_mode(q_c, qc_rows),
        )
        nisa.dma_copy(
            dst=qr_rows[0:P, b, 0:NQT, 0 : QH * DR],
            src=q_r.ap(pattern=[[H * DR, P], [P * H * DR, NQT], [1, QH * DR]], offset=n * QH * DR),
            **_dma_mode(q_r, qr_rows),
        )

    # The query tiles of a head in (f, qt) order, cut into PSUM banks of up
    # to TP_BANK tiles of one width.
    bank_tiles = []
    for t, (f, qt) in enumerate((f, qt) for f in range(NF) for qt in range(NQT)):
        if bank_tiles and len(bank_tiles[-1]) < TP_BANK and widths[bank_tiles[-1][-1][1]] == widths[f]:
            bank_tiles[-1].append((t, f, qt))
        else:
            bank_tiles.append([(t, f, qt)])
    n_bank = [0]
    NR = R // P

    def load_head_fm(j):
        """Load the feature-major queries of head ``j`` into its ``q_fm`` slot.

        ``q_fm[c, s, f * Sq + t] = q_c[j, f * P + c, t]`` for the latent
        tiles, and the rope tile follows them. Each DMA row is ``Sq``
        contiguous values.
        """
        s = j % OWN_QS
        nisa.dma_copy(
            dst=q_fm[0:P, s, 0 : NR * Sq],
            src=q_c.ap(pattern=[[Sq, P], [P * Sq, NR], [1, Sq]], offset=j * R * Sq),
            **_dma_mode(q_c, q_fm),
        )
        nisa.dma_copy(
            dst=q_fm[0:DR, s, NR * Sq : (NR + 1) * Sq],
            src=q_r.ap(pattern=[[Sq, DR], [1, Sq]], offset=j * DR * Sq),
            **_dma_mode(q_r, q_fm),
        )

    def transpose_head(j):
        """Transpose the queries of head ``j`` into its ``q_fm`` slot.

        After the last head of a load, the rows of the load OWN_QL ahead
        go into the freed ring slot. With feature-major queries, load the
        head instead.
        """
        if feature:
            load_head_fm(j)
            return
        n, hh = divmod(j, QH)
        b, s = n % OWN_QL, j % OWN_QS
        for bt in bank_tiles:
            k = n_bank[0] % 2
            n_bank[0] += 1
            w = widths[bt[0][1]]
            for c, (_, f, qt) in enumerate(bt):
                if f * P < R:
                    src = qc_rows[0:P, b, qt, hh * R + f * P : hh * R + f * P + w]
                else:
                    src = qr_rows[0:P, b, qt, hh * DR : hh * DR + w]
                _pe_transpose(ps_tq[0:w, k, c * P : (c + 1) * P], src, id_dt)
            t0 = bt[0][0]
            nisa.tensor_copy(
                dst=q_fm[0:w, s, t0 * P : (t0 + len(bt)) * P],
                src=ps_tq[0:w, k, 0 : len(bt) * P],
                engine=nisa.vector_engine,
            )
        if hh == QH - 1 and n + OWN_QL < n_loads:
            load_queries(n + OWN_QL)

    def scores(u):
        """Give the scores and the softmax of unit ``u``."""
        j, qt0 = units[u]
        s, o, k = j % OWN_QS, u % OWN_BUF, 0
        for q in range(QB):
            for f in range(NF):
                w = widths[f]
                c0 = (f * NQT + qt0 + q) * P
                nisa.nc_matmul(
                    dst=ps_sc[0:P, k, q * Kc : (q + 1) * Kc],
                    stationary=q_fm[0:w, s, c0 : c0 + P],
                    moving=kc_tile(f, w),
                    accumulate=(f > 0),
                )
        # s = scale * scores + bias, for the QB query tiles in one pass.
        nisa.scalar_tensor_tensor(
            dst=sc[0:P, o, 0 : QB * Kc],
            data=ps_sc[0:P, k, 0 : QB * Kc],
            op0=nl.multiply,
            operand0=scale,
            op1=nl.add,
            operand1=b_sb[0:P, qt0 : qt0 + QB, 0:Kc],
        )
        col = j * NQT + qt0
        sc3 = sc[0:P, o, 0 : QB * Kc].reshape((P, QB, Kc))
        nisa.tensor_reduce(dst=nm_all[0:P, col : col + QB], op=nl.maximum, data=sc3, axis=2, negate=True)
        for q in range(QB):
            nisa.activation(
                dst=pb[0:P, o, q * Kc : (q + 1) * Kc],
                op=nl.exp,
                data=sc[0:P, o, q * Kc : (q + 1) * Kc],
                bias=nm_all[0:P, col + q : col + q + 1],
                reduce_op=nl.add,
                reduce_res=l_all[0:P, col + q : col + q + 1],
                reduce_cmd=nisa.reduce_cmd.reset_reduce,
            )

    def transpose_p(u):
        """Give p^T of unit ``u`` by TensorE transposes (bit exact), one per key tile."""
        o = u % OWN_BUF
        for jk in range(QB * n_kc):
            _pe_transpose(ps_pt[0:P, 0, jk * P : (jk + 1) * P], pb[0:P, o, jk * P : (jk + 1) * P], id32)
        nisa.tensor_copy(dst=ptb[0:P, o, 0 : QB * Kc], src=ps_pt[0:P, 0, 0 : QB * Kc], engine=nisa.vector_engine)

    n_pv = [0]

    def load_wv(j):
        """Load W_V of head ``j`` into its ``wv_sb`` slot: NRW * VD contiguous values a partition."""
        nisa.dma_copy(
            dst=wv_sb[0:P, j % OWN_WV, 0 : NRW * VD],
            src=w_v.ap(pattern=[[NRW * VD, P], [1, NRW * VD]], offset=j * P * NRW * VD),
            **_dma_mode(w_v, wv_sb),
        )

    def project_values(j):
        """Project the own value rows with W_V of head ``j``: vo[key, d] = sum_r kc[key, r] W_V[j, d, r].

        The PSUM banks 2 and 3 of the value matmuls hold it (the value matmuls of a head use one bank
        in this mode), and the copy rounds it to the operand dtype.
        """
        s, k = j % OWN_WV, 2 + j % 2
        for jk in range(n_kc):
            for f in range(NRW):
                nisa.nc_matmul(
                    dst=ps_pv[0:P, k, jk * VD : (jk + 1) * VD],
                    stationary=kc_tile(f, P, jk * P, (jk + 1) * P),
                    moving=wv_sb[0:P, s, f * VD : (f + 1) * VD],
                    accumulate=(f > 0),
                )
        nisa.tensor_copy(
            dst=vo_sb[0:P, s, 0:n_kc, 0:VD],
            src=ps_pv[0:P, k, 0 : n_kc * VD].reshape((P, n_kc, VD)),
            engine=nisa.vector_engine,
        )

    def values_wv(u):
        """Give the value matmuls of unit ``u`` with the projected own values: the head's acc^T ``[VD, Sq]``.

        The projected values are the stationary and p^T the moving operand, so with one key tile a single
        matmul covers the QB query tiles of the unit. The query tiles of a head fill one PSUM bank, copied
        after its last unit.
        """
        j, qt0 = units[u]
        o, k, s = u % OWN_BUF, j % 2, j % OWN_WV
        for qi in range(1 if n_kc == 1 else QB):
            for jk in range(n_kc):
                if n_kc == 1:
                    dst, moving = ps_pv[0:VD, k, qt0 * P : (qt0 + QB) * P], ptb[0:P, o, 0 : QB * P]
                else:
                    c0 = (qi * n_kc + jk) * P
                    dst, moving = ps_pv[0:VD, k, (qt0 + qi) * P : (qt0 + qi + 1) * P], ptb[0:P, o, c0 : c0 + P]
                nisa.nc_matmul(dst=dst, stationary=vo_sb[0:P, s, jk, 0:VD], moving=moving, accumulate=(jk > 0))
        if qt0 + QB == NQT:
            nisa.tensor_copy(dst=acc_st[0:VD, j % OWN_ACC, 0:Sq], src=ps_pv[0:VD, k, 0:Sq], engine=nisa.scalar_engine)

    def values(u):
        """Give the value matmuls of unit ``u`` and the copies of its acc."""
        if wv:
            values_wv(u)
            return
        j, qt0 = units[u]
        o = u % OWN_BUF
        pend = []
        for q in range(QB):
            k = n_pv[0] % OWN_PV
            n_pv[0] += 1
            for jk in range(n_kc):
                c0 = q * Kc + jk * P
                nisa.nc_matmul(
                    dst=ps_pv[0:P, k, 0:R],
                    stationary=ptb[0:P, o, c0 : c0 + P],
                    moving=kc_rows[0:P, jk, 0:R],
                    accumulate=(jk > 0),
                )
            pend.append((k, qt0 + q))
            # Two consecutive banks share one copy.
            if len(pend) == 2 and pend[1][0] == pend[0][0] + 1:
                (k0, t0), _ = pend
                nisa.tensor_copy(
                    dst=acc_st[0:P, j % OWN_ACC, t0 : t0 + 2, 0:R],
                    src=ps_pv[0:P, k0 : k0 + 2, 0:R],
                    engine=nisa.scalar_engine,
                )
                pend = []
            elif len(pend) == 2:
                for k1, t1 in pend[:1]:
                    nisa.tensor_copy(
                        dst=acc_st[0:P, j % OWN_ACC, t1, 0:R], src=ps_pv[0:P, k1, 0:R], engine=nisa.scalar_engine
                    )
                pend = pend[1:]
        for k1, t1 in pend:
            nisa.tensor_copy(dst=acc_st[0:P, j % OWN_ACC, t1, 0:R], src=ps_pv[0:P, k1, 0:R], engine=nisa.scalar_engine)

    def store_head(j):
        if wv:
            # acc_out[j, d, :] = acc_st[d, slot, :].
            nisa.dma_copy(
                dst=acc_out.ap(pattern=[[Sq, VD], [1, Sq]], offset=j * VD * Sq),
                src=acc_st[0:VD, j % OWN_ACC, 0:Sq],
                **OWN_DMA,
            )
            return
        # acc_out[j, qt * P + p, :] = acc_st[p, slot, qt, :].
        nisa.dma_copy(
            dst=acc_out.ap(pattern=[[R, P], [P * R, NQT], [1, R]], offset=j * Sq * R),
            src=acc_st[0:P, j % OWN_ACC, 0:NQT, 0:R],
            **OWN_DMA,
        )

    def store_stats(g):
        """Store m and l of head group ``g``, transposed so that a row of a DMA is contiguous.

        m_out[h, qt * P + p] is column h * NQT + qt, partition p. A store of
        the columns as they are would write 4-byte pieces. TensorE
        transposes (bit exact) give each column a partition, and the store
        writes P contiguous values for each.
        """
        for a in range(0, ncol, P):
            n, c0 = min(P, ncol - a), g * ncol + a
            nisa.tensor_scalar(dst=m_st[0:P, 0:n], data=nm_all[0:P, c0 : c0 + n], op0=nl.multiply, operand0=-1.0)
            _pe_transpose(ps_pt[0:n, 0, 0:P], m_st[0:P, 0:n], id32)
            _pe_transpose(ps_pt[0:n, 0, P : 2 * P], l_all[0:P, c0 : c0 + n], id32)
            nisa.tensor_copy(dst=st_t[0:n, 0:2, 0:P], src=ps_pt[0:n, 0, 0 : 2 * P], engine=nisa.vector_engine)
            nisa.dma_copy(dst=m_out.ap(pattern=[[P, n], [1, P]], offset=c0 * P), src=st_t[0:n, 0, 0:P], **OWN_DMA)
            nisa.dma_copy(dst=l_out.ap(pattern=[[P, n], [1, P]], offset=c0 * P), src=st_t[0:n, 1, 0:P], **OWN_DMA)

    if not feature:
        for n in range(min(OWN_QL, n_loads)):
            load_queries(n)
    for j in range(min(OWN_QS - 1, H)):
        transpose_head(j)
    for t in range(-OWN_LEAD, n_units):
        u = t + OWN_LEAD
        if 0 <= u < n_units:
            j, qt0 = units[u]
            if wv and qt0 == 0:
                # W_V of the head, which its value projection reads OWN_LEAD - OWN_MID units later.
                load_wv(j)
            scores(u)
            if qt0 + QB == NQT and j + OWN_QS - 1 < H:
                # The queries of a later head, OWN_QS - 1 heads before its
                # scores.
                transpose_head(j + OWN_QS - 1)
        if 0 <= t + OWN_MID < n_units:
            transpose_p(t + OWN_MID)
            if wv and units[t + OWN_MID][1] == 0:
                project_values(units[t + OWN_MID][0])
        if t >= 0:
            values(t)
            j, qt0 = units[t]
            if qt0 + QB == NQT:
                store_head(j)
                if (j + 1) % G == 0:
                    store_stats(j // G)


@nki.jit
def mla_flash_latent(
    q_c,
    q_r,
    cache,
    block_table,
    marker,
    kc,
    cbias,
    ident,
    prior_tokens,
    scale: float,
    heads_per_group: int = 8,
    prior_variants: bool = False,
    fp8_matmul: bool = True,
    fp8_query: bool | str = "split",
    q_layout: str = "token",
    mx_values: bool = False,
    w_v=None,
    p_prescale: bool = False,
    short_prior_steps: tuple = (),
    pad_scalars: bool = False,
    qk_snake: bool = False,
    softmax_chain_order: bool = False,
    own_drain_transpose: bool = False,
    quad_pages: bool = False,
):
    """Compute the shared-key MLA flash partial of all heads.

    Args:
        q_c: ``[Sq, H, R]`` absorbed latent queries. ``Sq`` is a multiple of
            128 and at most 512. ``R`` is a multiple of 128 and at most 512.
            With ``q_layout="feature"``, ``[H, R, Sq]``.
        q_r: ``[Sq, H, DR]`` rope queries, ``DR <= 128``. With
            ``q_layout="feature"``, ``[H, DR, Sq]``.
        cache: ``[NP, 128, R + DR]`` page pool of the rank-local slab rows,
            bf16, or FP8 (``float8_e4m3fn`` values with a per-layer scale of
            1, as the GPU KV cache). The own keys and, unless
            ``fp8_matmul``, the cache rows load with converting DMAs into
            bf16 tiles, which is exact. In an FP8 pool the slots at or after
            ``prior_tokens`` may hold any bytes, NaN codes included: the
            kernel does not read them. A bf16 pool must keep them finite
            and small (``prefill/mla.py`` ``flash_partial``). An FP8 pool has at most
            2^24 rows. The cache can also be the contiguous cache base of the
            layer in another shape, for example the flat runner base
            ``[NP * 128 * (R + DR)]``. The kernel then views it as ``NP``
            pages of ``[128, R + DR]``. Give the base itself, not a torch view
            of it: neuronx-cc copies the whole base for a torch view that
            feeds the kernel.
        block_table: ``[1, NBT]`` int32 physical page of each logical
            page, each in ``[0, NP)``. ``NBT`` is a multiple of 16. The
            logical view has ``S = NBT * 128`` slots.
        marker: ``[1, S]`` marker of the logical slots, ``0`` visible or
            ``MARKER`` hidden. The kernel adds it to the score before the
            scale.
        kc: ``[Kc, R + DR]`` own keys of the call. ``Kc`` is a multiple of
            128 and at most 512. In the cache dtype; with an FP8 cache, the
            rows as the cache write quantized them.
        cbias: ``[Sq, Kc]`` fp32 own-key bias, added after the scale.
        ident: ``[128, 128]`` identity in the operand dtype (bf16 with an
            FP8 cache). The queries and the marker are in the operand
            dtype too; any float input is converted on load. Every
            TensorE transpose of the kernel uses it (or its fp32 copy),
            not NKI's shared identity.
        prior_tokens: ``[1, 1]`` int32 runtime count of the rank-local
            slots before the segment. The kernel streams
            ``ceil(min(prior_tokens, S) / 2048)`` blocks and ignores the
            slots after them.
        scale: The softmax scale.
        heads_per_group: The most heads that share one pass over the
            cache. The kernel uses the largest divisor of ``H`` that is at
            most this value. A group of 8 heads at ``Sq = 512`` fills most
            of SBUF.
        prior_variants: Add the group-loop variants for a short prior: a
            one-block view keeps its block resident and runs only its key
            steps that hold prior slots, and a tail block of 1 or 2 such
            steps runs only those. They make a prefix-cached call with a
            prior under 2048 slots faster, slow a call with a
            longer prior, and make the kernel ~2.4x larger (60K
            instead of 25K instructions; a longer compile). A call with
            no prior does not use them.
        fp8_matmul: With an FP8 cache, compute the value sums of the cache
            steps in FP8: the cache rows stay FP8 in SBUF, the attention
            weights are rounded to ``float8_e4m3fn``, and double-row
            matmuls take 256 keys a pass (2 passes for 512 keys instead
            of 4), or MX matmuls 512 keys (``mx_values``). The row sum adds
            the unrounded weights. The GPU's FP8 attention rounds the
            weights the same way, and also rounds its queries and per-head
            keys and values, so this is no less accurate than the GPU.
            False upcasts exactly. A bf16 cache (or another FP8 format) and
            a call with no prior do not use it, and the own keys keep bf16
            weights.
        fp8_query: With ``fp8_matmul``, the form of the latent part of
            the cache scores. ``"split"`` (the default) gives the latent
            query as two e4m3 parts, ``hi = e4m3(q)`` and
            ``lo = e4m3(q - hi)``, and contracts all 512 latent features
            of each part in one MX matmul with the FP8 cache keys (unit
            scales): 2 passes instead of 4, with the latent query as
            accurate as bf16 to about 2^-8 (the MX matmul rounds its pair
            sums to 11 bits). The rope query stays bf16, and the own keys
            take the same scores. The parts hold ``|q| < 896``. On the
            checkpoint's 24 MLA layers its error is that of bf16 queries
            within 1.5% (and at most 0.98x the GPU's FP8 attention), for
            rank-local priors of 384 to 32640 slots. False keeps the bf16
            queries: the scores are exact (upcast keys). ``True`` rounds
            the queries to ``float8_e4m3fn`` (unit scale, saturated at
            448) and uses double-row matmuls (2 passes instead of 4), the
            GPU's FP8 query. In the latent form a query rounding error is
            weighted by the latent values, and this is less accurate than
            the GPU on several layers (up to 1.4x its error). ``"mx"``
            (opt-in, with ``q_layout="feature"``) keeps the hi part only,
            in MXFP8: each 32 latent features of a query share one E8M0
            scale (``nisa.quantize_mx``: the block max rounded down to a
            power of two, over 2^7), so no value saturates, and the
            score drops the lo pass (1 latent MX pass a bank instead of 2,
            in the cache and the own-key steps). In the normal range the
            rounding is the GPU's e4m3 cast of the unscaled query, without
            its saturation above 448 and its subnormals below 2^-6; the
            rope query stays bf16. This changes the result (the precision
            of the GPU's FP8 query instead of about bf16). On captured
            serving calls of six MLA layers (priors 8K to 1M) most heads
            are within the error of the GPU's FP8 attention, but some are
            above it (up to 3.3x on one head), as the latent form weights
            the query rounding by the latent values.
        q_layout: ``"token"`` (default) or ``"feature"``, the layout of
            ``q_c`` and ``q_r``. The feature-major queries load with DMAs
            straight into the score operand, so the kernel runs no query
            transpose. The result is the same for both layouts.
        mx_values: With ``fp8_matmul`` and ``fp8_query="split"``, compute
            the value sums of the cache steps as MX matmuls (opt-in): one
            pass of 512 keys instead of two double-row passes, so a 1024-key
            step takes 8 TensorE passes instead of 10, and no ScalarE or
            VectorE pass rounds the weights (the exponential writes them in
            e4m3, rounded from fp32 as the GPU does). The MX matmul rounds
            each pair and each quad of products to 11 bits before its fp32
            accumulation (~2e-4 of the sum, against the 2^-4 of the e4m3
            weights). On the checkpoint's 24 MLA layers, for rank-local
            priors of 384 to 32640 slots, flat and peaked queries, its
            error is at most 0.98x the GPU's FP8 attention, within 0.01 of
            the double-row values. Faster calls with 2 to 16
            cache blocks (and in the 1M last segment on trn3). Other
            modes, a bf16 cache and a call with no prior ignore it.
        w_v: None (the default), or the value projection of every head,
            ``[H, 128, R // 128, VD]`` with ``w_v[h, c, k, d] =
            W_V[h, d, 128 k + c]`` for the value rows ``W_V [H, VD, R]`` of
            ``kv_b_proj`` (``prefill/mla.py`` ``wv_kernel_layout``), in the operand dtype.
            ``VD`` is at most 128. The kernel then returns the projected
            partial feature-major, ``acc [H, VD, Sq]`` with
            ``acc[h, d, t] = (acc_latent[h] @ W_V[h]^T)[t, d]``, so the
            cross-rank merge reads and reduces ``VD`` values a query
            instead of ``R``; ``m`` and ``l`` do not change. A call with a prior rounds the
            fp32 latent acc of each head to the operand dtype once, at its
            group end, and multiplies it with ``W_V`` (fp32 accumulation). A
            call with no prior rounds the own keys' projected value rows
            ``W_V c`` instead (one rounding a key and head, as the GPU's
            values). Either rounding is that of the bf16 output after the
            merge; on the checkpoint's 24 MLA layers the merged output is no
            less accurate than with the latent merge.
        p_prescale: With ``fp8_matmul``, round the attention weights of the
            cache steps to e4m3 as ``2^8 p`` instead of ``p`` (opt-in). The
            weights are relative to the running max of the column, which
            the own keys set and every earlier key raises, so at a long
            prior (or when the own keys score higher than the prior) much of
            the softmax mass falls to e4m3 subnormals or to 0: a bias, since
            the row sum keeps the unrounded weights. The GPU's FP8 attention
            rounds against a stale max per 64K-token chunk, so it loses less.
            With the prescale the weights stay at most 256 (below 448), and
            the 2^8 is undone exactly once. With ``mx_values`` the value
            matmuls take an E8M0 scale of 2^-8 and the row sum a factor
            2^-8, so ``acc``, ``m`` and ``l`` keep their meaning. With the
            double-row values the returned state is shifted: ``m - 8 ln 2``,
            ``256 l`` and ``256 acc``, so ``acc / l`` and ``m + ln l`` (all
            that the LSE merge reads) are unchanged. Other modes, a bf16
            cache and a call with no prior ignore it.
        short_prior_steps: Step counts (1024-key steps of the prior,
            ``ceil(min(prior_tokens, S) / 1024)``, at most 7) whose calls run
            a group-loop variant with that count's block bodies inline
            (opt-in; MX values without ``prior_variants`` only). A rolled
            loop boundary costs a drain and two all-engine barriers, and a
            group of the rolled path crosses ~11 of them, so a variant takes
            time off each head group. Counts above ``ceil(S / 1024)`` are
            dropped, and a view whose step counts are all listed has no
            rolled group loop: with a 16-page table (S 2048) and
            ``(1, 2, 3, 4)``, an 8-call program on trn3 runs faster
            at priors of 128 to 1920 slots and about as fast at prior 0, and
            it is smaller. In a larger view each listed count adds one loop
            that every other call skips at a cost, one more copy of the group
            body (about +20% instructions per count) and sequencer
            registers; ``(1, 2, 3, 4)`` fits a 24-call stage. The result is
            the same as without it. The extra code moves the SBUF layout and
            the loop placement. ``p8_buf`` (the FP8 weights) is the source of
            the p^T DMA transposes that feed the value MX matmuls; when the
            allocator puts it off a 256-B boundary, the transposes split into
            ~2.5x the DMA descriptors and those matmuls run slower. The
            loop back edges can also turn slow. With a 256-page table that
            cost time at long priors in the measured layouts. Check the
            layout of the program that uses it.
        pad_scalars: Allocate each 4-byte scalar SBUF tile (trip counts,
            flags, block counters) as 512 B per partition and use its first
            column, so that no scalar tile moves a later tile off a 512-B
            boundary. The allocator packs these tiles in front of others at
            8-B steps: in front of ``p8_buf``, the source of the p^T DMA
            transposes, that splits the transposes into ~2.5x the DMA
            descriptors and slows the value MX matmuls; in front of
            the own-key tiles it slows the prior-0 path. For programs
            whose code grows (``short_prior_steps``) and whose layout moves.
            Costs at most ~17 KiB of SBUF per partition while the tiles live
            (a one-call program still compiles with 200 KiB caller-live).
            The result is the same as without it.
        qk_snake: Run a step's split-query MX score matmuls in snake order
            (bank 0 hi, lo; bank 1 lo, hi) instead of hi, lo in each bank.
            The middle two matmuls then share their stationary, so TensorE
            loads it once: 128 fewer instructions in a pair-loop body
            (2069 -> 1941), and consecutive matmuls of one bank still share
            their moving keys. A body of 2050-2304 instructions spans 9 or
            10 of TensorE's 256-instruction fetch blocks depending on where
            the loop lands, and at 10 every back edge is slow (a
            refetch, then a throttled restart); at 1941 it spans at most 9.
            A bank's two partial scores add in either order to the same
            fp32 sum, so the result is the same as without it.
        softmax_chain_order: Keep a cache step's acc update out of the
            softmax chain of the MX-value steps (opt-in). The scores of a
            step reuse the PSUM banks of the step two before, so TensorE
            waits for that step's chain: the max (VectorE), -m (VectorE),
            then the exponential (ScalarE). The acc update of the step
            ``MXV_LAG`` before is not in the engine order (its ISA call
            returns a raw handle), and the compiler schedules it between the
            max and -m, which puts VectorE work into the chain;
            at long priors TensorE then waits for it in each step (in a 1M
            call). With the option VectorE issues the max, -m, then the acc
            update, and the step's last exponential waits for that acc
            update, so TensorE's existing wait on the exponential still
            orders the next value matmul into the acc update's PSUM bank (no
            new TensorE instruction). Only issue order and one ScalarE
            semaphore wait change, so the result is the same as without it.
        own_drain_transpose: Give the last ``OWN_PE_DRAIN`` own-key steps
            of a group end (MX values) their p^T by bf16 TensorE transposes
            (bit exact) and a VectorE copy in their value stage, instead of
            the DMA transpose issued with their softmax (opt-in). Those
            steps' value matmuls then wait for the copy instead of the DMA
            transpose's latency, which is the drain of the own-key pipeline
            at every group end. The p^T bytes are the same, so the result is
            the same as without it.
        quad_pages: Read the cache pages in the key-quad layout (opt-in, the
            key-quad cache format; needs ``mx_values``). A page then
            stores its 128 rows as 32 latent quad rows of 2,048 bytes (byte
            ``2048 j + 4 f + i`` is latent feature ``f`` of row ``4 j + i``)
            and the 64 rope bytes of each row row-major after them (byte
            ``65536 + 64 o``). The latent of a 512-key group loads whole quad
            rows (contiguous), and the loaded tile is the MX value operand
            with no on-chip value pack: partition ``p`` packs the four
            consecutive keys ``4 p .. 4 p + 3`` of the group, as the value
            matmul and the natural-order p^T need. The latent QK keys take one
            4x4 byte shuffle per group (``_quad_shuffle``) before the word
            transposes, in place of the value pack. The rope keys load
            row-major as without it. The writer zeroes every lane and row at
            or past the sequence end, so the whole-quad loads carry no NaN,
            and the marker still hides them. Off (the default) the kernel is
            byte-identical to the row-major reader.

    Returns:
        ``(acc [H, Sq, R], m [H, Sq, 1], l [H, Sq, 1])`` fp32 HBM tensors;
        ``acc [H, VD, Sq]`` (feature-major) with ``w_v``.

    Example:
        >>> # acc, m, l = nki.simulate(mla_flash_latent[1])(*numpy_inputs,
        >>> #                                               scale=0.07)
    """
    assert q_layout in ("token", "feature"), "q_layout must be token or feature"  # noqa: S101
    feature = q_layout == "feature"
    if feature:
        H, R, Sq = q_c.shape
        DR = q_r.shape[1]
        assert tuple(q_r.shape) == (H, DR, Sq), "q_r must be [H, DR, Sq]"  # noqa: S101
    else:
        Sq, H, R = q_c.shape
        DR = q_r.shape[2]
    if len(cache.shape) != 3:
        # The cache base itself: view it as pages of [128, R + DR] here, in the kernel.
        numel = 1
        for size in cache.shape:
            numel *= size
        assert numel % (P * (R + DR)) == 0, "the cache base must hold whole [128, R + DR] pages"  # noqa: S101
        cache = cache.reshape((numel // (P * (R + DR)), P, R + DR))
    NP, PG, D = cache.shape
    NBT = block_table.shape[1]
    S = NBT * P
    Kc = kc.shape[0]
    # The operand dtype of the matmuls: bf16 for an FP8 cache. Every load
    # converts its input to it (an FP8 value is exact in bf16).
    dt = nl.bfloat16 if cache.dtype in _FP8_DTYPES else cache.dtype
    # The FP8 matmuls of the cache steps, for the GPU cache format (OCP E4M3).
    mm8 = fp8_matmul and cache.dtype == nl.float8_e4m3fn
    fp8_rows = cache.dtype in _FP8_DTYPES
    assert fp8_query in (False, True, "split", "mx"), "fp8_query must be False, True, 'split' or 'mx'"  # noqa: S101
    # q8: the e4m3 queries of double-row scores; qmx: the latent query of MX scores, split (hi and lo parts,
    # unit scales) or, with qhi (fp8_query="mx"), the hi part only, in MXFP8 with a block scale a 32 features.
    q8 = mm8 and fp8_query is True
    qmx = mm8 and fp8_query in ("split", "mx")
    qhi = qmx and fp8_query == "mx"
    assert not qhi or feature, "fp8_query='mx' needs q_layout='feature'"  # noqa: S101
    # The MX parts of the latent query: (hi, lo), or hi only.
    Q_PARTS = (0,) if qhi else (0, 1)
    # mxv: the value sums of the cache steps as MX matmuls (with the split query; opt-in).
    mxv = qmx and mx_values
    # prescale: the P prescale of the FP8 cache steps (opt-in): e4m3(2^8 p), undone once.
    prescale = mm8 and p_prescale
    # softmax_chain_order: VectorE issues a cache step's acc update after the max and -m of the step MXV_LAG later.
    chain_order = mxv and softmax_chain_order
    # own_drain_transpose: the own-key pipeline of a group end (own_at_end) drains through TensorE transposes.
    own_pe_drain = mxv and own_drain_transpose
    # The group-boundary pipeline (MX values only): a group end runs the next group's own-key steps
    # (``next_group``), so they overlap its stores. The other paths keep the schedule with the own-key
    # steps at a group start: run at a group end, they keep the group state (acc and the statistics)
    # live across the group loop's back edge, and with the double-row values (more SBUF) the
    # allocator's coloring fallback then places the cache rows over its reserved top bytes under
    # caller pressure (NCC_INLA001).
    own_at_end = mxv
    # The half tail (MX values, no short-prior variants): a last block whose second 1024-key step holds
    # no prior slot runs its first step only (a fully hidden step is the identity, so the result is the
    # same), in a tail loop of its own.
    half_tail = mxv and not prior_variants
    # The block loads wrap (half tail): a load past the prior's last block loads block 0, the next group's
    # first block, so the last block-loop body of a group leaves it in set 0 and the group end loads none.
    wrap = half_tail
    # quad: the key-quad cache reader. It supports the served MX-value path only.
    quad = quad_pages and mxv
    assert not quad_pages or mxv, "quad_pages needs the MX value sums (mx_values)"  # noqa: S101
    assert not quad or not prior_variants, "quad_pages does not support prior_variants yet"  # noqa: S101
    assert not quad or cache.dtype in _FP8_DTYPES, "quad_pages needs an FP8 cache"  # noqa: S101
    # Bytes of a key-quad page: R latent quad rows of 4 R bytes (one feature of 4 rows per word), then the
    # 64 rope bytes of each row row-major. QUAD_BYTES is a latent quad row, also R values of the MX operand.
    QUAD_BYTES = R * QUAD
    QUAD_ROWS = PG // QUAD
    ROPE_BASE = QUAD_ROWS * QUAD_BYTES
    assert not qmx or R == MX_PACK * P, "fp8_query='split' or 'mx' needs R = 512"  # noqa: S101
    assert D == R + DR, "cache rows must be [R latent | DR rope]"  # noqa: S101
    assert Sq % P == 0 and Sq <= KT, "Sq must be a multiple of 128, <= 512"  # noqa: S101
    assert R % P == 0 and R <= KT, "R must be a multiple of 128, <= 512"  # noqa: S101
    assert 0 < DR <= P, "DR must be in (0, 128]"  # noqa: S101
    assert PG == P, "cache pages must have 128 rows"  # noqa: S101
    assert NBT % SUBS == 0 and NBT > 0, "block_table must have a positive multiple of 16 pages"  # noqa: S101
    assert marker.shape[1] == S, "marker must have one entry for each slot"  # noqa: S101
    assert Kc % P == 0 and 0 < Kc <= KT, "Kc must be a multiple of 128, <= 512"  # noqa: S101
    G = max(g for g in range(1, min(heads_per_group, H) + 1) if H % g == 0)
    NQT = Sq // P
    NF = (D + P - 1) // P  # feature tiles of the contraction
    NFL = R // P  # the latent tiles; the last tile holds the rope part
    widths = [min(P, D - f * P) for f in range(NF)]
    n_kc = Kc // P
    # The marker rides in the spare partition of the last feature tile: the
    # queries hold 1 there and the cache keys hold the marker, so the score
    # matmul adds the marker with no extra matmul. A last tile without a
    # spare partition keeps a rank-1 marker matmul.
    w_tail = widths[NF - 1]
    fold = w_tail < P
    rows_qk = [w + 1 if fold and f == NF - 1 else w for f, w in enumerate(widths)]

    wv = w_v is not None
    NRW = R // P
    VD = R
    if wv:
        VD = w_v.shape[3]
        assert tuple(w_v.shape) == (H, P, NRW, VD), "w_v must be [H, 128, R // 128, VD]"  # noqa: S101
        # A head's projected acc^T [VD, Sq] is one PSUM bank, and one acc column at a group end.
        assert 0 < VD <= P and Sq <= R, "w_v needs VD <= 128 and Sq <= R"  # noqa: S101
        assert Kc // P * VD <= KT, "w_v needs VD * Kc / 128 <= 512"  # noqa: S101

    tid = _new_trace_id()
    nm = _Namer(tid, "const")

    def scalar_tile(rows, dtype, name):
        """Give a [rows, 1] SBUF tile (``pad_scalars``: the first column of a [rows, 128] one)."""
        if pad_scalars:
            return nl.ndarray((rows, 128), dtype=dtype, buffer=nl.sbuf, name=name)[0:rows, 0:1]
        return nl.ndarray((rows, 1), dtype=dtype, buffer=nl.sbuf, name=name)

    acc_out = nl.ndarray((H, VD, Sq) if wv else (H, Sq, R), dtype=_F32, buffer=nl.shared_hbm, name=nm("acc_out"))
    m_out = nl.ndarray((H, Sq, 1), dtype=_F32, buffer=nl.shared_hbm, name=nm("m_out"))
    l_out = nl.ndarray((H, Sq, 1), dtype=_F32, buffer=nl.shared_hbm, name=nm("l_out"))

    # id32 is the identity of the fp32 TensorE transposes (_pe_transpose); id_sb serves those in dt. With a
    # two-byte dt, the two identities share one tile: id_sb in columns 0 .. P, and id32 in the bytes of columns
    # P .. 3 P (_Fp32View). Do not give id32 a tile of its own: neuronx-cc 2.28 can put such an fp32 tile at
    # SBUF byte 0, in the dynamic DMA scratch (DynamicDMAScratchLoc, SBUF [0, 16 KiB)) that holds the
    # descriptor rings of the software DGE. The copy into id32 then overwrites live descriptors, all DMA engines
    # of the core abort, and the rank hangs. The compiler places the shared tile as it places id_sb.
    if dt == _F32:
        id_sb = nl.ndarray((P, P), dtype=dt, buffer=nl.sbuf, name=nm("id_sb"))
        nisa.dma_copy(dst=id_sb, src=ident)
        id32 = id_sb
    else:
        id_all = nl.ndarray((P, 3 * P), dtype=dt, buffer=nl.sbuf, name=nm("id_sb"))
        id_sb = id_all[0:P, 0:P]
        nisa.dma_copy(dst=id_sb, src=ident)
        id32 = _Fp32View(id_all)
        nisa.tensor_copy(dst=id32[0:P, 0:P], src=id_sb, engine=nisa.vector_engine)
    if not fold:
        ones_row = nl.ndarray((1, P), dtype=dt, buffer=nl.sbuf, name=nm("ones"))
        nisa.memset(dst=ones_row, value=1.0)

    # The runtime trip count of the cache block loop:
    # n_blocks = (min(prior, S) + 2047) >> 11. The add and the minimum are
    # exact, because the values are much less than 2^24.
    nb_sb = scalar_tile(1, nl.int32, nm("nb_sb"))
    nisa.dma_copy(dst=nb_sb, src=prior_tokens[0:1, 0:1])
    nisa.tensor_scalar(dst=nb_sb, data=nb_sb, op0=nl.minimum, operand0=S, op1=nl.add, operand1=BLOCK - 1)
    nisa.tensor_scalar(dst=nb_sb, data=nb_sb, op0=nl.right_shift, operand0=BLOCK.bit_length() - 1)
    # The block loop runs over pairs of blocks with two buffer sets, and an
    # odd count leaves one tail block.
    npair_sb = scalar_tile(1, nl.int32, nm("npair_sb"))
    if half_tail:
        ntail_sb = scalar_tile(1, nl.int32, nm("ntail_sb"))
        # In 1024-key steps: ns = ceil(min(prior, S) / 1024) steps, whose last one holds prior slots.
        # A pair body runs 4 steps, a full tail block 2 and a half tail block 1, so the pair loop takes
        # ns >> 2 trips, the full tail (ns >> 1) & 1 and the half tail ns & 1.
        ns_sb = scalar_tile(1, nl.int32, nm("ns_sb"))
        nisa.dma_copy(dst=ns_sb, src=prior_tokens[0:1, 0:1])
        nisa.tensor_scalar(dst=ns_sb, data=ns_sb, op0=nl.minimum, operand0=S, op1=nl.add, operand1=KT_MM8 - 1)
        nisa.tensor_scalar(dst=ns_sb, data=ns_sb, op0=nl.right_shift, operand0=KT_MM8.bit_length() - 1)
        nisa.tensor_scalar(dst=npair_sb, data=ns_sb, op0=nl.right_shift, operand0=2)
        nisa.tensor_scalar(dst=ntail_sb, data=ns_sb, op0=nl.right_shift, operand0=1, op1=nl.bitwise_and, operand1=1)
        nhalf_sb = scalar_tile(1, nl.int32, nm("nhalf_sb"))
        nisa.tensor_scalar(dst=nhalf_sb, data=ns_sb, op0=nl.bitwise_and, operand0=1)
    if wrap:
        # The end slot of the streamed blocks, n_blocks * 2048.
        nbB_sb = scalar_tile(1, nl.int32, nm("nbB_sb"))
        nisa.tensor_scalar(dst=nbB_sb, data=nb_sb, op0=nl.left_shift, operand0=BLOCK.bit_length() - 1)
        # 1 while the next block to load is a prior block, else 0 (the load wraps to block 0).
        wr_sb = scalar_tile(1, nl.int32, nm("wr_sb"))
    else:
        nisa.tensor_scalar(dst=npair_sb, data=nb_sb, op0=nl.right_shift, operand0=1)
        ntail_sb = scalar_tile(1, nl.int32, nm("ntail_sb"))
        nisa.tensor_scalar(dst=ntail_sb, data=nb_sb, op0=nl.bitwise_and, operand0=1)
    # Short priors (half tail): a call whose step count is in short_ns runs a group-loop variant with that count's
    # block bodies inline, and the rolled group loop (``gen``) runs the other calls. A view of at most
    # max(short_ns) steps needs no rolled group loop.
    ns_max = -(-S // KT_MM8)
    assert all(isinstance(k, int) and 0 < k < 8 for k in short_prior_steps), "short_prior_steps must be in 1..7"  # noqa: S101
    short_ns = tuple(sorted({k for k in short_prior_steps if k <= ns_max})) if half_tail else ()
    gen = not short_ns or ns_max > max(short_ns) or short_ns != tuple(range(1, max(short_ns) + 1))
    if gen:
        npair_reg = nisa.register_alloc()
        nisa.register_load(npair_reg, npair_sb)
        ntail_reg = nisa.register_alloc()
        nisa.register_load(ntail_reg, ntail_sb)
        if half_tail:
            nhalf_reg = nisa.register_alloc()
            nisa.register_load(nhalf_reg, nhalf_sb)
    # The trip counts of the two paths, 0 or 1: a call with a prior runs
    # cache_path, and a call with no prior runs the own-key path. The
    # compiler has no branch, so each path is a loop of 0 or 1 trips.
    cache_sb = scalar_tile(1, nl.int32, nm("cache_sb"))
    nisa.tensor_scalar(dst=cache_sb, data=nb_sb, op0=nl.minimum, operand0=1)
    own_sb = scalar_tile(1, nl.int32, nm("own_sb"))
    nisa.tensor_scalar(dst=own_sb, data=cache_sb, op0=nl.multiply, operand0=-1, op1=nl.add, operand1=1)
    cache_reg = nisa.register_alloc()
    nisa.register_load(cache_reg, cache_sb)
    own_reg = nisa.register_alloc()
    nisa.register_load(own_reg, own_sb)
    if short_ns:
        # vn_sbs[k] = H / G if ns == k else 0, and gen_sb = H / G if ns > max(short_ns) else 0: the trip counts.
        vn_sbs = {}
        for k in short_ns:
            vn_sbs[k] = scalar_tile(1, nl.int32, nm("vn_sb"))
            nisa.tensor_scalar(dst=vn_sbs[k], data=ns_sb, op0=nl.equal, operand0=k, op1=nl.multiply, operand1=H // G)
        if gen:
            gen_sb = scalar_tile(1, nl.int32, nm("gen_sb"))
            if short_ns == tuple(range(1, max(short_ns) + 1)):
                nisa.tensor_scalar(
                    dst=gen_sb, data=ns_sb, op0=nl.greater, operand0=max(short_ns), op1=nl.multiply, operand1=H // G
                )
            else:
                # The step counts with no variant of their own: H / G minus the variants' trip counts.
                nisa.tensor_scalar(
                    dst=gen_sb, data=vn_sbs[short_ns[0]], op0=nl.multiply, operand0=-1, op1=nl.add, operand1=H // G
                )
                for k in short_ns[1:]:
                    nisa.tensor_tensor(dst=gen_sb, data1=gen_sb, data2=vn_sbs[k], op=nl.subtract)
        # The trip counts go to registers here, after the paths' own: flag tiles that stay live into cache_path
        # sit under the own-key path and can move its tiles off their 256-B alignment (slower at prior 0).
        vn_regs = {}
        for k in short_ns:
            vn_regs[k] = nisa.register_alloc()
            nisa.register_load(vn_regs[k], vn_sbs[k])
        if gen:
            gen_reg = nisa.register_alloc()
            nisa.register_load(gen_reg, gen_sb)

    def transpose_rows(nmx, dst_fm, rows, n_sub):
        """Transpose key-major rows into feature-major tiles.

        ``rows`` is ``[P, n_sub, D]`` (key ``sub * P + p`` at ``[p, sub]``).
        ``dst_fm`` is ``[P, NF, n_sub * P]``, or a pair of a latent and a rope
        tensor (``transpose_banks``); key ``sub * P + p`` goes to
        column ``sub * P + p``. The transposes are bf16 TensorE transposes
        (bit exact). Up to eight share one PSUM bank and one copy, and the
        copies alternate between VectorE and ScalarE.
        """
        banks = transpose_banks
        if isinstance(dst_fm, tuple) and dst_fm[0] == "mx":
            banks = transpose_banks_mx
        elif isinstance(dst_fm, tuple) and dst_fm[0] == "mxv":
            banks = transpose_banks_mxv
        for emit in banks(nmx, dst_fm, rows, n_sub):
            emit()

    def transpose_banks(nmx, dst_fm, rows, n_sub, copy_engine=None, ps_tp=None, od=None):
        """Give the transposes of ``transpose_rows``, one closure per PSUM bank.

        The banks of the first keys come first, so the first key step can
        start before the last keys are transposed. FP8 rows transpose by a
        matmul with the bf16 identity into bf16 PSUM, which is exact (a
        TensorE transpose keeps the FP8 dtype and needs a 2-element output
        step). A ``dst_fm`` pair ``(latent, rope)`` takes the latent tiles
        and the rope tile in two tensors. The copies out of PSUM alternate
        between VectorE and ScalarE, or all run on ``copy_engine``. With
        ``ps_tp``, every bank goes through that one PSUM tile. With ``od``
        (an ``_EngineOrder``), each engine issues them in program order.
        """
        banks = []
        for s0 in range(0, n_sub, TP_BANK):
            for f in range(NF):

                def emit(s0=s0, f=f, n_copy=len(banks)):
                    w = widths[f]
                    ns = min(TP_BANK, n_sub - s0)
                    ps = (
                        ps_tp
                        if ps_tp is not None
                        else nl.ndarray((P, TP_BANK * P), dtype=dt, buffer=nl.psum, name=nmx("tp_k"))
                    )
                    for j in range(ns):
                        src = rows[0:P, s0 + j, f * P : f * P + w]
                        if rows.dtype == dt:
                            ins = _pe_transpose(ps[0:w, j * P : (j + 1) * P], src, id_sb)
                        else:
                            ins = nisa.nc_matmul(
                                dst=ps[0:w, j * P : (j + 1) * P],
                                stationary=src,
                                moving=id_sb[0:P, 0:P],
                                accumulate=False,
                            )
                        if od is not None:
                            od("pe", ins)
                    if isinstance(dst_fm, tuple):
                        fm, ft = (dst_fm[0], f) if f < NFL else (dst_fm[1], 0)
                    else:
                        fm, ft = dst_fm, f
                    eng = (nisa.vector_engine, nisa.scalar_engine)[n_copy % 2] if copy_engine is None else copy_engine
                    ins = nisa.tensor_copy(dst=fm[0:w, ft, s0 * P : (s0 + ns) * P], src=ps[0:w, 0 : ns * P], engine=eng)
                    if od is not None:
                        od("dve" if eng == nisa.vector_engine else "act", ins)

                banks.append(emit)
        return banks

    def mx_perm(src, n_part, part_step, n_group):
        """Read ``src`` (``n_group`` groups of 4 key tiles of P columns, tile-major) in the MX value order.

        Element ``(g, 4 w + i)`` of the read is column ``(4 g + i) * P + w`` of ``src``, so 4
        consecutive columns hold the keys that one partition of the MX value operand packs.
        """
        return src.ap(pattern=[[part_step, n_part], [MX_PACK * P, n_group], [1, P], [P, MX_PACK]], offset=0)

    def mx_words(nmx, k_mx, rows, n_sub, l0, ns, ps=None, od=None, eng=None, perm=False):
        """Transpose the latent part of the FP8 row tiles ``l0 .. l0 + ns`` (``ns`` <= 4) as 32-bit words.

        Word ``c`` of a row holds latent features ``4c .. 4c + 3``, so ``k_mx[c, k]`` (fp32 view) is
        the MX packing of key ``k`` (``float8_e4m3fn_x4``). The TensorE transposes are bit exact; four
        share one fp32 PSUM bank and one copy. With ``perm`` the copy writes the keys in the MX value
        order of the group (``mx_perm``).
        """
        if ps is None:
            ps = nl.ndarray((P, KT), dtype=_F32, buffer=nl.psum, name=nmx("tp_mx"))
        for j in range(ns):
            ins = _pe_transpose(
                ps[0:P, j * P : (j + 1) * P],
                rows.ap(
                    pattern=[[n_sub * D // MX_PACK, P], [1, R // MX_PACK]],
                    offset=(l0 + j) * D // MX_PACK,
                    dtype=_F32,
                ),
                id32,
            )
            if od is not None:
                od("pe", ins)
        eng = nisa.vector_engine if eng is None else eng
        src = ps[0:P, 0 : ns * P]
        if perm:
            assert ns == MX_PACK, "a permuted copy takes whole groups of 4 key tiles"  # noqa: S101
            src = ps.ap(pattern=[[KT, P], [1, P], [P, MX_PACK]], offset=0)
        ins = nisa.tensor_copy(dst=k_mx[0:P, l0 * P : (l0 + ns) * P], src=src, engine=eng)
        if od is not None:
            od("dve" if eng == nisa.vector_engine else "act", ins)

    def transpose_banks_mx(nmx, dst_fm, rows, n_sub, copy_engine=None, ps_tp=None, ps_tpx=None, od=None):
        """Give the key transposes of the split-query MX scores (``fp8_query="split"``), one closure per PSUM bank.

        ``dst_fm`` is ``("mx", k_mx, k_fr)``: the latent keys go to ``k_mx`` as 32-bit words
        (``mx_words``, four key tiles a bank) and the rope part to ``k_fr`` as in
        ``transpose_banks`` (the matmul with the bf16 identity, eight key tiles a bank). For each
        eight key tiles the latent banks come first. ``ps_tpx`` and ``ps_tp`` are the PSUM tiles
        of the two kinds; the copies alternate between VectorE and ScalarE, or run on
        ``copy_engine``.
        """
        _, k_mx, k_fr = dst_fm
        w = widths[NF - 1]
        lat_bank = KT // P
        banks = []

        def engine(n_copy):
            return (nisa.vector_engine, nisa.scalar_engine)[n_copy % 2] if copy_engine is None else copy_engine

        for s0 in range(0, n_sub, TP_BANK):
            for l0 in range(s0, min(s0 + TP_BANK, n_sub), lat_bank):

                def emit_lat(l0=l0, n_copy=len(banks)):
                    mx_words(nmx, k_mx, rows, n_sub, l0, min(lat_bank, n_sub - l0), ps_tpx, od, engine(n_copy))

                banks.append(emit_lat)

            def emit_rope(s0=s0, n_copy=len(banks)):
                ns = min(TP_BANK, n_sub - s0)
                ps = (
                    ps_tp
                    if ps_tp is not None
                    else nl.ndarray((P, TP_BANK * P), dtype=dt, buffer=nl.psum, name=nmx("tp_k"))
                )
                for j in range(ns):
                    ins = nisa.nc_matmul(
                        dst=ps[0:w, j * P : (j + 1) * P],
                        stationary=rows[0:P, s0 + j, R:D],
                        moving=id_sb[0:P, 0:P],
                        accumulate=False,
                    )
                    if od is not None:
                        od("pe", ins)
                eng = engine(n_copy)
                ins = nisa.tensor_copy(dst=k_fr[0:w, 0, s0 * P : (s0 + ns) * P], src=ps[0:w, 0 : ns * P], engine=eng)
                if od is not None:
                    od("dve" if eng == nisa.vector_engine else "act", ins)

            banks.append(emit_rope)
        return banks

    def transpose_banks_mxv(
        nmx, dst_fm, rows, n_sub, copy_engine=None, ps_tp=None, ps_tpx=None, od=None, n_grp=None, g0=0
    ):
        """Give the key transposes and value packing of the MX-value steps (``mx_values``) as closures.

        ``dst_fm`` is ``("mxv", k_mx, k_fr, vmx, mk_in, mk_out)``. For each group of four key tiles:
        the latent keys go to ``k_mx`` as 32-bit words (``mx_words``), the rope part to ``k_fr`` (the
        matmul with the bf16 identity, in a half of the bank ``ps_tp``), ScalarE copies the marker
        ``mk_in`` into ``mk_out``, and ScalarE packs the value rows into ``vmx`` in four pieces
        (``vmx[w, g, 4 r + i]`` = feature r of the row at partition w of tile 4 g + i). The key and
        marker copies write the keys in the MX value order (``mx_perm``). ``ps_tpx`` and ``ps_tp`` are
        the PSUM tiles of the two transpose kinds; their copies alternate between VectorE and
        ScalarE, or run on ``copy_engine``. With ``n_grp``, only the first ``n_grp`` groups; with ``g0``, the
        groups from ``g0`` on.
        """
        _, k_mx, k_fr, vmx, mk_in, mk_out = dst_fm[:6]
        rope_q = dst_fm[6] if quad else None
        w = widths[NF - 1]
        lat_bank = KT // P
        banks = []

        def engine(n_copy):
            return (nisa.vector_engine, nisa.scalar_engine)[n_copy % 2] if copy_engine is None else copy_engine

        # Small closures (one PSUM bank's copy, or a quarter of a group's value packing), so that none
        # holds up the in-order engines for long between two key steps.
        for g in range(g0, n_sub // MX_PACK if n_grp is None else n_grp):
            l0 = MX_PACK * g

            def emit_lat(g=g, l0=l0, n_copy=len(banks)):
                if not quad:
                    mx_words(nmx, k_mx, rows, n_sub, l0, lat_bank, ps_tpx, od, engine(n_copy), True)
                    return
                # Key-quad latent: one 4x4 byte shuffle of the loaded value tile (a word = 4 keys of one
                # feature) into word-key order (a word = 4 features of one key), then four 32-bit transposes
                # (lane i gives the keys 4 p + i). The keys come out in natural order (no ``mx_perm``): a
                # score column is its key, so the p^T words (4 consecutive keys) match the value operand.
                shuf = nl.ndarray((P, QUAD_BYTES), dtype=nl.uint8, buffer=nl.sbuf, name=nmx("k_shuf"))
                seng = engine(n_copy)
                # The latent is in the staging ``rows`` (the vmx layout); the value operand ``vmx`` is written
                # by the deferred ``emit_vcopy`` bank below.
                rows_u8 = rows.view(nl.uint8).reshape((P, NSTEP * QUAD_BYTES))
                s_ins = _quad_shuffle(shuf, rows_u8, NSTEP * QUAD_BYTES, g * QUAD_BYTES, R, seng)
                if od is not None:
                    od("dve" if seng == nisa.vector_engine else "act", s_ins)
                ps = ps_tpx if ps_tpx is not None else nl.ndarray((P, KT), dtype=_F32, buffer=nl.psum, name=nmx("tp_mx"))
                for i in range(QUAD):
                    ins = _pe_transpose(
                        ps[0:P, i * P : (i + 1) * P],
                        shuf.ap(pattern=[[R, P], [QUAD, R // QUAD]], offset=i, dtype=_F32),
                        id32,
                    )
                    if od is not None:
                        od("pe", ins)
                ceng = engine(n_copy)
                ins = nisa.tensor_copy(
                    dst=k_mx.ap(pattern=[[BLOCK, P], [1, QUAD], [QUAD, P]], offset=KT * g),
                    src=ps.ap(pattern=[[KT, P], [P, QUAD], [1, P]], offset=0),
                    engine=ceng,
                )
                if od is not None:
                    od("dve" if ceng == nisa.vector_engine else "act", ins)

            def emit_rope(g=g, l0=l0, n_copy=len(banks)):
                ps = (
                    ps_tp
                    if ps_tp is not None
                    else nl.ndarray((P, TP_BANK * P), dtype=dt, buffer=nl.psum, name=nmx("tp_k"))
                )
                c0 = (g % 2) * MX_PACK * P  # the two halves of the bank alternate between groups
                for j in range(MX_PACK):
                    ins = nisa.nc_matmul(
                        dst=ps[0:w, c0 + j * P : c0 + (j + 1) * P],
                        stationary=rope_q[0:P, l0 + j, 0:DR] if quad else rows[0:P, l0 + j, R:D],
                        moving=id_sb[0:P, 0:P],
                        accumulate=False,
                    )
                    if od is not None:
                        od("pe", ins)
                eng = engine(n_copy)
                # Quad keys are natural order (a straight copy); else the MX value order (``mx_perm``).
                src = (
                    ps[0:w, c0 : c0 + MX_PACK * P]
                    if quad
                    else mx_perm(ps[0:w, c0 : c0 + MX_PACK * P], w, TP_BANK * P, 1)
                )
                ins = nisa.tensor_copy(dst=k_fr[0:w, 0, l0 * P : (l0 + MX_PACK) * P], src=src, engine=eng)
                if od is not None:
                    od("dve" if eng == nisa.vector_engine else "act", ins)

            def emit_mk(l0=l0):
                src = (
                    mk_in[0:1, l0 * P : (l0 + MX_PACK) * P]
                    if quad
                    else mx_perm(mk_in[0:1, l0 * P : (l0 + MX_PACK) * P], 1, BLOCK, 1)
                )
                ins = nisa.tensor_copy(
                    dst=mk_out[0:1, l0 * P : (l0 + MX_PACK) * P], src=src, engine=nisa.scalar_engine
                )
                if od is not None:
                    od("act", ins)

            banks += [emit_lat, emit_rope, emit_mk]
            if quad:
                # The value operand is a straight copy of the staging ``rows`` (already the vmx layout). It is
                # deferred (appended last, like ``emit_v``) so the block's lagged value matmuls read the live
                # ``vmx`` before a prefetch of the next block overwrites the staging ``rows``.
                def emit_vcopy(g=g):
                    ins = nisa.tensor_copy(
                        dst=vmx[0:P, g, 0:QUAD_BYTES],
                        src=rows[0:P, g, 0:QUAD_BYTES],
                        engine=nisa.scalar_engine,
                    )
                    if od is not None:
                        od("act", ins)

                banks.append(emit_vcopy)
                continue
            rq = R // MX_PACK
            for h in range(MX_PACK):

                def emit_v(g=g, h=h):
                    # vmx[w, g, 4 r + i] = rows[w, 4 g + i, r] for r in [rq h, rq (h + 1))
                    ins = nisa.tensor_copy(
                        dst=vmx[0:P, g, h * rq * MX_PACK : (h + 1) * rq * MX_PACK],
                        src=rows.ap(pattern=[[n_sub * D, P], [1, rq], [D, MX_PACK]], offset=MX_PACK * g * D + h * rq),
                        engine=nisa.scalar_engine,
                    )
                    if od is not None:
                        od("act", ins)

                banks.append(emit_v)
        return banks

    # The own keys and their bias are the same for every head group. Load
    # and transpose them one time.
    kc_rows = nl.ndarray((P, n_kc, D), dtype=dt, buffer=nl.sbuf, name=nm("kc_rows"))
    nisa.dma_copy(dst=kc_rows, src=kc.ap(pattern=[[D, P], [P * D, n_kc], [1, D]], offset=0), **_dma_mode(kc, kc_rows))
    if qmx:
        # The split query's cache path reads the rope tile only, so the latent tiles (the no-prior path's)
        # are a tensor of their own, which the cache path does not keep live.
        kc_fm = (
            nl.ndarray((P, NFL, Kc), dtype=dt, buffer=nl.sbuf, name=nm("kc_fl")),
            nl.ndarray((P, 1, Kc), dtype=dt, buffer=nl.sbuf, name=nm("kc_fr")),
        )
        kc_rope, rope_t = kc_fm[1], 0
    else:
        kc_fm = nl.ndarray((P, NF, Kc), dtype=dt, buffer=nl.sbuf, name=nm("kc_fm"))
        kc_rope, rope_t = kc_fm, NF - 1
    transpose_rows(nm, kc_fm, kc_rows, n_kc)
    if fold:
        # The own keys have no marker.
        nisa.memset(dst=kc_rope[w_tail : w_tail + 1, rope_t, 0:Kc], value=0.0)
    b_sb = nl.ndarray((P, NQT, Kc), dtype=_F32, buffer=nl.sbuf, name=nm("b_sb"))
    nisa.dma_copy(dst=b_sb, src=cbias.ap(pattern=[[Kc, P], [P * Kc, NQT], [1, Kc]], offset=0), **OWN_DMA)

    def cache_path(_):
        """Attend the cache prior and the own keys: the path of a call with a prior."""
        if qmx:
            # The split-query scores of the own keys: their FP8 rows (as the cache write quantized them,
            # so the bf16 rows round back exactly) in the MX packing. The rope part stays in kc_rope.
            kc8_rows = nl.ndarray((P, n_kc, D), dtype=cache.dtype, buffer=nl.sbuf, name=nm("kc8_rows"))
            nisa.tensor_copy(dst=kc8_rows, src=kc_rows, engine=nisa.vector_engine)
            kc_mx = nl.ndarray((P, Kc), dtype=_F32, buffer=nl.sbuf, name=nm("kc_mx"))
            for l0 in range(0, n_kc, KT // P):
                mx_words(nm, kc_mx, kc8_rows, n_kc, l0, min(KT // P, n_kc - l0))
            mx_scale = nl.ndarray((P, KT), dtype=nl.uint8, buffer=nl.sbuf, name=nm("mx_scale"))
            nisa.memset(dst=mx_scale, value=MX_UNIT)
            if prescale and mxv:
                # The E8M0 scale 2^-8 of the MX value matmuls' stationary: the prescaled weights.
                mx_scale_p = nl.ndarray((P, P), dtype=nl.uint8, buffer=nl.sbuf, name=nm("mx_scale_p"))
                nisa.memset(dst=mx_scale_p, value=MX_UNIT - PS_LOG2)
        if prior_variants:
            # The key steps of the last block that hold prior slots:
            # ceil((min(prior, S) - 2048 * (n_blocks - 1)) / 512), in [1, NSTEP] when
            # n_blocks > 0. With an odd block count the last block is the tail, and
            # a tail runs only these steps (the later steps see hidden slots only).
            # The step count is static in a loop body, so the group loop has a
            # variant for 1 and for 2 tail steps (TAIL_STEPS) and a full one; the
            # variant whose flag is 1 runs. An even block count uses the full one.
            ns_sb = scalar_tile(1, nl.int32, nm("ns_sb"))
            nisa.dma_copy(dst=ns_sb, src=prior_tokens[0:1, 0:1])
            nisa.tensor_scalar(dst=ns_sb, data=ns_sb, op0=nl.minimum, operand0=S)
            tmp_sb = scalar_tile(1, nl.int32, nm("tmp_sb"))
            nisa.tensor_scalar(dst=tmp_sb, data=nb_sb, op0=nl.add, operand0=-1, op1=nl.multiply, operand1=BLOCK)
            nisa.tensor_tensor(dst=ns_sb, data1=ns_sb, data2=tmp_sb, op=nl.subtract)
            nisa.tensor_scalar(dst=ns_sb, data=ns_sb, op0=nl.add, operand0=KT - 1)
            nisa.tensor_scalar(dst=ns_sb, data=ns_sb, op0=nl.right_shift, operand0=KT.bit_length() - 1)
            # s_eff = NSTEP - (NSTEP - steps) * (n_blocks odd).
            nisa.tensor_scalar(dst=tmp_sb, data=ns_sb, op0=nl.multiply, operand0=-1, op1=nl.add, operand1=NSTEP)
            nisa.tensor_tensor(dst=tmp_sb, data1=tmp_sb, data2=ntail_sb, op=nl.multiply)
            nisa.tensor_scalar(dst=ns_sb, data=tmp_sb, op0=nl.multiply, operand0=-1, op1=nl.add, operand1=NSTEP)

            # one_sb = (n_blocks == 1), many_sb = (n_blocks >= 2), as 0 or 1.
            one_sb = scalar_tile(1, nl.int32, nm("one_sb"))
            many_sb = scalar_tile(1, nl.int32, nm("many_sb"))
            nisa.tensor_scalar(dst=many_sb, data=nb_sb, op0=nl.add, operand0=-1, op1=nl.maximum, operand1=0)
            nisa.tensor_scalar(dst=many_sb, data=many_sb, op0=nl.minimum, operand0=1)
            nisa.tensor_scalar(dst=one_sb, data=many_sb, op0=nl.multiply, operand0=-1, op1=nl.add, operand1=1)

            def flag_reg(lo, hi, times=1, also=None, also2=None):
                """Give a register that is ``times`` if lo <= s_eff <= hi (and ``also``, ``also2``), else 0."""
                # (s_eff >= lo) * (s_eff <= hi), each as min(max(.., 0), 1).
                f_sb = scalar_tile(1, nl.int32, nm("f_sb"))
                g_sb = scalar_tile(1, nl.int32, nm("g_sb"))
                nisa.tensor_scalar(dst=f_sb, data=ns_sb, op0=nl.add, operand0=1 - lo, op1=nl.maximum, operand1=0)
                nisa.tensor_scalar(dst=f_sb, data=f_sb, op0=nl.minimum, operand0=1)
                nisa.tensor_scalar(dst=g_sb, data=ns_sb, op0=nl.multiply, operand0=-1, op1=nl.add, operand1=hi + 1)
                nisa.tensor_scalar(dst=g_sb, data=g_sb, op0=nl.maximum, operand0=0, op1=nl.minimum, operand1=1)
                nisa.tensor_tensor(dst=f_sb, data1=f_sb, data2=g_sb, op=nl.multiply)
                for extra in (also, also2):
                    if extra is not None:
                        # max(f, extra) when extra is ``("or", t)``, else f * extra.
                        if isinstance(extra, tuple):
                            nisa.tensor_tensor(dst=f_sb, data1=f_sb, data2=extra[1], op=nl.maximum)
                        else:
                            nisa.tensor_tensor(dst=f_sb, data1=f_sb, data2=extra, op=nl.multiply)
                if times != 1:
                    nisa.tensor_scalar(dst=f_sb, data=f_sb, op0=nl.multiply, operand0=times)
                reg = nisa.register_alloc()
                nisa.register_load(reg, f_sb)
                return reg

            # The trip counts of the group-loop variants: H / G or 0.
            # - full: n_blocks >= 2 with a full tail (or none); the group loop
            #   itself, so such a call pays one 0-trip loop (the wrapper).
            # - one block (n_blocks == 1, the APC case of a prior under 2048
            #   slots): the block stays in set 0 for every group, so these
            #   variants have no block loop and no reload at a group end; one
            #   variant for each step count.
            # - a short tail (TAIL_STEPS) after n_blocks >= 3.
            full_reg = flag_reg(TAIL_STEPS[-1] + 1, NSTEP, H // G, also=many_sb)
            part_reg = flag_reg(1, TAIL_STEPS[-1], also=("or", one_sb))
            one_reg = flag_reg(1, NSTEP, also=one_sb)
            single_regs = {st: flag_reg(st, st, H // G, also=one_sb) for st in range(1, NSTEP + 1)}
            tail_regs = {st: flag_reg(st, st, H // G, also=many_sb) for st in TAIL_STEPS}
        else:
            # One group-loop variant (full tails); a register trip count. With ``short_ns`` the rolled
            # group loop's trip count is ``gen_sb``.
            full_reg = None if short_ns else nisa.register_alloc(H // G)
        # Loop counters in SBUF: the first head of the group, and the first
        # logical slot and logical page of the next block to load. The bodies
        # advance them in place. kl_sb and pl_sb are the slot and page of a
        # load, clamped to the last block: the prefetch after the last block of
        # a group loads a block that no step reads.
        h_sb = scalar_tile(1, nl.int32, nm("h_sb"))
        nisa.memset(dst=h_sb, value=0)
        k_sb = scalar_tile(1, nl.int32, nm("k_sb"))
        pg_sb = scalar_tile(1, nl.int32, nm("pg_sb"))
        kl_sb = scalar_tile(1, nl.int32, nm("kl_sb"))
        pl_sb = scalar_tile(1, nl.int32, nm("pl_sb"))

        # The state of one head group, one column for each (head, query tile).
        # At G = 8 and Sq = 512 it is about 106 KB for each partition (acc
        # 64 KB, q_fm 40 KB). The running max is kept negated, in two slots
        # that alternate between key steps: a step reads slot ``par`` and
        # writes slot ``1 - par``, so the update needs no copy.
        if qmx:
            # The split latent query in the MX packing: column idx * P + t of q_mx[c, part] holds the
            # e4m3 bytes of latent features 4c .. 4c + 3 of query t of column idx (part 0 hi, 1 lo).
            # q_fm keeps the rope tile only (and its ones partition), at tile idx.
            # The token-major path writes it as 32-bit words (PSUM copies of transposes), the feature-major
            # path as FP8 bytes, so each writes it in its own dtype.
            # With qhi q_mx holds the hi part only, and q_sc its E8M0 scales: q_sc[32 b + j, idx * P + t]
            # scales latent features 32 (4 b + j) .. 32 (4 b + j) + 31 of query t of column idx (b < 4,
            # j < 4: the MX scale layout, four rows at the start of each 32-partition quadrant).
            if feature:
                q_mx = nl.ndarray(
                    (P, len(Q_PARTS), G * NQT * P * MX_PACK), dtype=cache.dtype, buffer=nl.sbuf, name=nm("q_mx")
                )
            else:
                q_mx = nl.ndarray((P, 2, G * NQT * P), dtype=_F32, buffer=nl.sbuf, name=nm("q_mx"))
            if qhi:
                q_sc = nl.ndarray((P, G * NQT * P), dtype=nl.uint8, buffer=nl.sbuf, name=nm("q_sc"))
                # quantize_mx's source: one head's queries with their 4-packs contiguous, two slots.
                q_il = nl.ndarray((P, 2, Sq * MX_PACK), dtype=dt, buffer=nl.sbuf, name=nm("q_il"))
                n_il = [0]
            q_fm = nl.ndarray((rows_qk[NF - 1] if mxv else P, G * NQT, P), dtype=dt, buffer=nl.sbuf, name=nm("q_fm"))
        else:
            q_fm = nl.ndarray((P, G * NQT * NF, P), dtype=cache.dtype if q8 else dt, buffer=nl.sbuf, name=nm("q_fm"))
        nm_run = nl.ndarray((P, 2, G * NQT), dtype=_F32, buffer=nl.sbuf, name=nm("nm_run"))
        m_fin = nl.ndarray((P, G * NQT), dtype=_F32, buffer=nl.sbuf, name=nm("m_fin"))
        st_t = nl.ndarray((P, 2, P), dtype=_F32, buffer=nl.sbuf, name=nm("st_t"))
        l_run = nl.ndarray((P, G * NQT), dtype=_F32, buffer=nl.sbuf, name=nm("l_run"))
        acc = nl.ndarray((P, G * NQT, R), dtype=_F32, buffer=nl.sbuf, name=nm("acc"))
        if wv:
            # The group's W_V (wv_g[c, hh, k * VD + d] = W_V[h0 + hh, d, k * P + c]).
            wv_g = nl.ndarray((P, G, NRW * VD), dtype=dt, buffer=nl.sbuf, name=nm("wv_g"))
            wv_dma = dict(dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.scalar) if w_v.dtype == dt else {}

        def load_wv_group(h_off, swdge=False, defer=False):
            """Load W_V of the heads ``h_off .. h_off + G``: G * NRW * VD contiguous values a partition.

            ``swdge`` and ``defer`` as in ``load_queries_fm``.
            """
            if defer:
                return [lambda after: load_wv_group(h_off, swdge).depends_on(after, kind=nisa.dependency.runtime)]
            return nisa.dma_copy(
                dst=wv_g,
                src=w_v.ap(
                    pattern=[[NRW * VD, P], [P * NRW * VD, G], [1, NRW * VD]],
                    offset=0,
                    scalar_offset=h_off,
                    indirect_dim=0,
                ),
                **(dict(oob_mode=nisa.oob_mode.skip, priority=STAGE_PRIORITY) if swdge else wv_dma),
            )

        def acc_t(s, k):
            """Give chunk ``k`` of the transposed acc of a head in slot ``s``: ``[P, Sq]`` in the operand dtype.

            ``acc_t(s, k)[c, qt * P + p] = acc[p, hh * NQT + qt, k * P + c]``. The two slots are the key
            steps' weight scratch ``p_buf`` and ``pt_buf``, which no step uses at a group end, or
            ``acc_t_buf``: with the MX values, whose scratch is too small and whose group end runs the next
            group's own-key steps.
            """
            if acc_t_own:
                return acc_t_buf[0:P, s, k * Sq : (k + 1) * Sq]
            buf = (p_buf, pt_buf)[s]
            return buf.ap(pattern=[[NBUF * pw, P], [1, Sq]], offset=k * Sq)

        def project_group(nmx):
            """Give (acc @ W_V^T)^T of each head of the group, ``[VD, Sq]`` in partitions 0 .. VD of acc column ``hh * NQT``.

            For each head: fp32 TensorE transposes (bit exact) of its NQT x NRW acc tiles, NQT of them in
            one PSUM bank; copies of the banks to ``acc_t`` round to the operand dtype; then NRW matmuls
            with the head's W_V chunks as the stationary and ``acc_t`` (all Sq queries) as the moving
            operand accumulate in one PSUM bank (fp32), whose copy overwrites the head's first acc column
            (its transposes have read it).
            """
            n_copy = 0
            for hh in range(G):
                s = hh % 2
                for k in range(NRW):
                    ps_t = nl.ndarray((P, NQT * P), dtype=_F32, buffer=nl.psum, name=nmx("wv_tp"))
                    for qt in range(NQT):
                        _pe_transpose(
                            ps_t[0:P, qt * P : (qt + 1) * P], acc[0:P, hh * NQT + qt, k * P : (k + 1) * P], id32
                        )
                    eng = (nisa.vector_engine, nisa.scalar_engine)[n_copy % 2]
                    n_copy += 1
                    nisa.tensor_copy(dst=acc_t(s, k), src=ps_t[0:P, 0:Sq], engine=eng)
                ps_o = nl.ndarray((P, Sq), dtype=_F32, buffer=nl.psum, name=nmx("wv_o"))
                for k in range(NRW):
                    nisa.nc_matmul(
                        dst=ps_o[0:VD, 0:Sq],
                        stationary=wv_g[0:P, hh, k * VD : (k + 1) * VD],
                        moving=acc_t(s, k),
                        accumulate=(k > 0),
                    )
                eng = (nisa.vector_engine, nisa.scalar_engine)[n_copy % 2]
                n_copy += 1
                nisa.tensor_copy(dst=acc[0:VD, hh * NQT, 0:Sq], src=ps_o[0:VD, 0:Sq], engine=eng)

        if qmx:

            def mx_query(part, idx):
                """Give the stationary of part ``part`` (0 hi, 1 lo) of the query column ``idx``."""
                return q_mx.ap(
                    pattern=[[len(Q_PARTS) * G * NQT * P, P], [1, P]],
                    offset=part * G * NQT * P + idx * P,
                    dtype=nl.float8_e4m3fn_x4,
                )

            def mx_query_scale(idx, unit):
                """Give the stationary scale of the query column ``idx``: its E8M0 block scales (qhi), else ``unit``."""
                return q_sc[0:P, idx * P : (idx + 1) * P] if qhi else unit

        if fold and qmx:
            nisa.memset(dst=q_fm[w_tail : w_tail + 1, 0 : G * NQT, 0:P], value=1.0)
        elif fold:
            # The query transposes write partitions 0 .. w_tail of the last
            # tile only, so the ones partition holds for every group.
            for idx in range(G * NQT):
                nisa.memset(dst=q_fm[w_tail : w_tail + 1, idx * NF + NF - 1, 0:P], value=1.0)

        # The keys of one cache step. With the FP8 value matmuls a step has
        # 1024 keys: its scores fill two PSUM banks, and the max, the
        # statistics and the acc update run once for both. The engines other
        # than TensorE then have the time to keep up with it.
        kt = KT_MM8 if mm8 else KT

        def block_steps(n_kstep):
            """Give the (first key, width) of the steps that cover the first ``n_kstep * KT`` keys of a block."""
            steps = [(k0, kt) for k0 in range(0, n_kstep * KT - kt + 1, kt)]
            done = len(steps) * kt
            if done < n_kstep * KT:
                steps.append((done, n_kstep * KT - done))
            return steps

        assert len(block_steps(NSTEP)) % 2 == 0, "a block must have an even number of key steps"  # noqa: S101

        # The scratch of a key step, in NBUF slots that consecutive steps
        # rotate through. With one scratch tensor, a step could not write it
        # before the previous step had read it, and the steps would run one at
        # a time.
        sc_buf = nl.ndarray((P, NBUF, Kc), dtype=_F32, buffer=nl.sbuf, name=nm("sc_buf"))
        # p and p^T in the operand dtype: of every step, or with the MX values of the own-key steps only.
        pw = Kc if mxv else kt
        p_buf = nl.ndarray((P, NBUF, pw), dtype=dt, buffer=nl.sbuf, name=nm("p_buf"))
        pt_buf = nl.ndarray((P, NBUF, pw), dtype=dt, buffer=nl.sbuf, name=nm("pt_buf"))
        # A slot of the transposed acc of the value projection (acc_t): p_buf or pt_buf, which no step uses at a
        # group end, or a tensor of its own: with the MX values p_buf only holds the own-key steps, which also
        # run at a group end.
        acc_t_own = wv and (own_at_end or NBUF * pw < NRW * Sq)
        if acc_t_own:
            acc_t_buf = nl.ndarray((P, 2, NRW * Sq), dtype=dt, buffer=nl.sbuf, name=nm("acc_t"))
        # Columns: t_max, alpha, p_sum.
        st_buf = nl.ndarray((P, NBUF, 3), dtype=_F32, buffer=nl.sbuf, name=nm("st_buf"))
        if mxv:
            # The weights of a cache step in FP8 (the exponential rounds them), and p^T as 32-bit words:
            # word w of query q in tile t holds the weights of the 4 keys of score columns
            # 512 t + 4 w .. 512 t + 4 w + 3, the MX stationary of the step's value matmuls.
            p8_buf = nl.ndarray((P, N_P8, kt), dtype=cache.dtype, buffer=nl.sbuf, name=nm("p8_buf"))
            ptw_buf = nl.ndarray((P, N_PTW, kt // MX_PACK), dtype=_F32, buffer=nl.sbuf, name=nm("ptw_buf"))
            # Defined before the loops: an MX stationary whose first write is inside the block loops gets its
            # scales misplaced by the SBUF allocator's coloring fallback under caller pressure (NCC_INLA001).
            nisa.memset(dst=ptw_buf, value=0.0, engine=nisa.vector_engine)
            # alpha and the row sum of the steps of a round (one step of every column), in N_RND slots: the
            # row sums update once a round, for all columns in two VectorE ops (``round_l``), and a slot
            # stays until the value stage of its last step (MXV_LAG steps later) has read alpha.
            N_RND = MXV_LAG // (G * NQT) + 2
            st_rnd = nl.ndarray((P, N_RND, 2, G * NQT), dtype=_F32, buffer=nl.sbuf, name=nm("st_rnd"))
        elif mm8:
            # p^T of a cache step rounded to FP8, the stationary of its double-row value matmuls.
            pt8_buf = nl.ndarray((P, NBUF, kt), dtype=cache.dtype, buffer=nl.sbuf, name=nm("pt8_buf"))

        def round_l(rnd, od):
            """l = l * alpha + p_sum for every column, from round slot ``rnd`` (mxv).

            With the prescale p_sum adds the weights 2^8 p, so l = l * alpha + 2^-8 p_sum.
            """
            ncol = G * NQT
            if prescale:
                od(
                    "dve",
                    nisa.tensor_tensor(
                        dst=l_run[0:P, 0:ncol],
                        data1=l_run[0:P, 0:ncol],
                        data2=st_rnd[0:P, rnd, 0, 0:ncol],
                        op=nl.multiply,
                        engine=nisa.vector_engine,
                    ),
                )
                od(
                    "dve",
                    nisa.scalar_tensor_tensor(
                        dst=l_run[0:P, 0:ncol],
                        data=st_rnd[0:P, rnd, 1, 0:ncol],
                        op0=nl.multiply,
                        operand0=1.0 / PS_SCALE,
                        op1=nl.add,
                        operand1=l_run[0:P, 0:ncol],
                    ),
                )
                return
            for j, op in ((0, nl.multiply), (1, nl.add)):
                od(
                    "dve",
                    nisa.tensor_tensor(
                        dst=l_run[0:P, 0:ncol],
                        data1=l_run[0:P, 0:ncol],
                        data2=st_rnd[0:P, rnd, j, 0:ncol],
                        op=op,
                        engine=nisa.vector_engine,
                    ),
                )

        def attend(
            nmx,
            idx,
            slot,
            par,
            k_fm,
            k0,
            width,
            mk_row,
            v_rows,
            v_sub0,
            ps=None,
            pv=None,
            od=None,
            rnd=None,
            step=0,
            pt_ps=None,
        ):
            """Attend one cache key tile for the query column ``idx`` in scratch slot ``slot``.

            ``k_fm[0:w, f, k0:k0 + width]`` holds the keys, feature-major.
            With ``fold``, partition ``w_tail`` of the last tile holds the
            pre-scale marker. Else ``mk_row [1, >= k0 + width]`` is the
            pre-scale marker. ``v_rows[0:P, v_sub0 + j, 0:R]`` holds the values
            of keys ``j * P + p`` of the tile. The step reads the negated
            running max from slot ``par`` and writes it to slot ``1 - par``.
            ``ps`` and ``pv`` are PSUM tiles for the scores and the value sum,
            else the step allocates its own. With ``od`` (an ``_EngineOrder``)
            each engine issues the step's instructions in program order.
            With ``pt_ps`` (a [P, kt / 4] fp32 PSUM tile; MX values) the value stage gives the step's p^T
            words by TensorE transposes into it and a copy, instead of the DMA transpose of ``soft``.
            """
            o = od if od is not None else (lambda _key, ins: ins)
            acc_last = [None]  # softmax_chain_order: the value stage's acc update, once emitted
            pt_eng = [nisa.scalar_engine if pt_ps is not None and pt_ps[1] else nisa.vector_engine]
            if pt_ps is not None:
                pt_ps = pt_ps[0]
            vec = {"engine": nisa.vector_engine} if od is not None else {}
            t_max, alpha, p_sum = (st_buf[0:P, slot, j : j + 1] for j in range(3))
            if rnd is not None:
                alpha, p_sum = (st_rnd[0:P, rnd, j, idx : idx + 1] for j in range(2))
            qbase = idx * NF
            if ps is None:
                ps = nl.ndarray((P, kt), dtype=_F32, buffer=nl.psum, name=nmx("ps_sc"))
            # The scores of each 512 keys go to one PSUM bank.
            if mxv:
                # The split latent query: one MX matmul of all 512 latent features for each bank and part
                # (hi, then lo), unit scales (with qhi the hi part only, with its block scales); then the
                # bf16 rope tile of each bank, with the marker partition. The rope matmuls come last: the
                # matmul after a rope one streams after it.
                kx, kr = k_fm[1], k_fm[2]
                banks = [(c0, min(KT, width - c0), k0 + c0) for c0 in range(0, width, KT)]
                # Hi then lo in each bank, or snake order (qk_snake: bank 1 runs lo first, so the middle
                # two matmuls share their stationary and TensorE loads it once).
                mms = []
                for j, b in enumerate(banks):
                    parts = Q_PARTS[::-1] if qk_snake and j % 2 else Q_PARTS
                    mms += [(b, part, k > 0) for k, part in enumerate(parts)]
                for (c0, cw, kk), part, accum in mms:
                    o(
                        "pe",
                        nisa.nc_matmul_mx(
                            dst=ps[0:P, c0 : c0 + cw],
                            stationary=mx_query(part, idx),
                            moving=kx.ap(pattern=[[BLOCK, P], [1, cw]], offset=kk, dtype=nl.float8_e4m3fn_x4),
                            stationary_scale=mx_query_scale(idx, mx_scale[0:P, 0:P]),
                            moving_scale=mx_scale[0:P, 0:cw],
                            accumulate=accum,
                        ),
                    )
                w = rows_qk[NF - 1]
                for c0, cw, kk in banks:
                    o(
                        "pe",
                        nisa.nc_matmul(
                            dst=ps[0:P, c0 : c0 + cw],
                            stationary=q_fm[0:w, idx, 0:P],
                            moving=kr[0:w, 0, kk : kk + cw],
                            accumulate=True,
                        ),
                    )
                    if mk_row is not None:
                        o(
                            "pe",
                            nisa.nc_matmul(
                                dst=ps[0:P, c0 : c0 + cw],
                                stationary=ones_row[0:1, 0:P],
                                moving=mk_row[0:1, kk : kk + cw],
                                accumulate=True,
                            ),
                        )
            for c0 in range(0, 0 if mxv else width, KT):
                cw, kk = min(KT, width - c0), k0 + c0
                dst = ps[0:P, c0 : c0 + cw]
                if q8:
                    # FP8 queries and latent keys: double-row matmuls of two latent tiles (256 rows) each.
                    # The rope tile is bf16, for the marker in its spare partition.
                    k8, kr = k_fm
                    for f in range(0, NFL, 2):
                        pair = f + 1 < NFL
                        o(
                            "pe",
                            nisa.nc_matmul(
                                dst=dst,
                                stationary=q_fm[0:P, qbase + f : qbase + f + 2, 0:P]
                                if pair
                                else q_fm[0:P, qbase + f, 0:P],
                                moving=k8[0:P, f : f + 2, kk : kk + cw] if pair else k8[0:P, f, kk : kk + cw],
                                accumulate=(f > 0),
                                perf_mode=nisa.matmul_perf_mode.double_row if pair else nisa.matmul_perf_mode.none,
                            ),
                        )
                    w = rows_qk[NF - 1]
                    o(
                        "pe",
                        nisa.nc_matmul(
                            dst=dst,
                            stationary=q_fm[0:w, qbase + NF - 1, 0:P],
                            moving=kr[0:w, 0, kk : kk + cw],
                            accumulate=True,
                        ),
                    )
                elif qmx:
                    # The split latent query: one MX matmul of all 512 latent features for each part
                    # (hi, then lo), unit scales; then the bf16 rope tile with the marker partition.
                    _, kx, kr = k_fm
                    for part in Q_PARTS:
                        o(
                            "pe",
                            nisa.nc_matmul_mx(
                                dst=dst,
                                stationary=mx_query(part, idx),
                                moving=kx.ap(pattern=[[BLOCK, P], [1, cw]], offset=kk, dtype=nl.float8_e4m3fn_x4),
                                stationary_scale=mx_query_scale(idx, mx_scale[0:P, 0:P]),
                                moving_scale=mx_scale[0:P, 0:cw],
                                accumulate=part > 0,
                            ),
                        )
                    w = rows_qk[NF - 1]
                    o(
                        "pe",
                        nisa.nc_matmul(
                            dst=dst, stationary=q_fm[0:w, idx, 0:P], moving=kr[0:w, 0, kk : kk + cw], accumulate=True
                        ),
                    )
                else:
                    for f in range(NF):
                        w = rows_qk[f]
                        o(
                            "pe",
                            nisa.nc_matmul(
                                dst=dst,
                                stationary=q_fm[0:w, qbase + f, 0:P],
                                moving=k_fm[0:w, f, kk : kk + cw],
                                accumulate=(f > 0),
                            ),
                        )
                if mk_row is not None:
                    # + marker[j] on every query row: a rank-1 matmul.
                    o(
                        "pe",
                        nisa.nc_matmul(
                            dst=dst, stationary=ones_row[0:1, 0:P], moving=mk_row[0:1, kk : kk + cw], accumulate=True
                        ),
                    )
            nm_old = nm_run[0:P, par, idx : idx + 1]
            nm_new = nm_run[0:P, 1 - par, idx : idx + 1]
            # The max and the exponential read the raw scores in PSUM. The scale
            # is positive, so max(scale * s) = scale * max(s), and the
            # exponential applies the scale itself.
            if prescale and mxv:
                # One VectorE pass gives -m_new: the scores times -scale, written back over them (the
                # exponentials read them next, with the scale -1), and their minimum with -m_old. Then
                # -m_new + 8 ln 2, the exponent bias of the prescaled weights, in the step's unused alpha
                # slot (mxv keeps alpha in st_rnd), on ScalarE: the exponentials that read it follow it in
                # ScalarE's queue, and VectorE carries the rest of the step (with it on VectorE the step was
                # slower at 16 blocks).
                o(
                    "dve",
                    nisa.tensor_scalar_reduce(
                        dst=ps[0:P, 0:width],
                        data=ps[0:P, 0:width],
                        op0=nl.multiply,
                        operand0=-scale,
                        reduce_op=nl.minimum,
                        reduce_res=nm_new,
                        reduce_cmd=nisa.reduce_cmd.load_reduce,
                        reduce_init=nm_old,
                    ),
                )
                p_bias, exp_scale = st_buf[0:P, slot, 1:2], -1.0
                o("act", nisa.activation(dst=p_bias, op=nl.copy, data=nm_new, bias=PS_LN))
            else:
                o("dve", nisa.tensor_reduce(dst=t_max, op=nl.maximum, data=ps[0:P, 0:width], axis=1))
                # -m_new = min(-scale * t_max, -m_old).
                o(
                    "dve",
                    nisa.tensor_scalar(
                        dst=nm_new, data=t_max, op0=nl.multiply, operand0=-scale, op1=nl.minimum, operand1=nm_old, **vec
                    ),
                )
                p_bias, exp_scale = nm_new, scale
            n_sub = width // P
            p_t = pt_buf[0:P, slot, :]

            n_mx = width // KT  # MX value passes (512 keys each)

            def soft(after=None):
                if mxv:
                    # The exponential of each PSUM bank (the row sum accumulates on ScalarE), so the scores
                    # of a later step can reuse the first bank while the second one is read; alpha last,
                    # since only the value stage reads it.
                    for c0 in range(0, width, KT):
                        cw = min(KT, width - c0)
                        e_ins = o(
                            "act",
                            nisa.activation(
                                dst=p8_buf[0:P, step % N_P8, c0 : c0 + cw],
                                op=nl.exp,
                                data=ps[0:P, c0 : c0 + cw],
                                scale=exp_scale,
                                bias=p_bias,
                                reduce_op=nl.add,
                                reduce_res=p_sum if c0 + cw == width else None,
                                reduce_cmd=nisa.reduce_cmd.reset_reduce if c0 == 0 else nisa.reduce_cmd.reduce,
                            ),
                        )
                        if after is not None and c0 > 0 and c0 + cw == width:
                            # softmax_chain_order: the last bank's exponential follows the acc update of the
                            # step MXV_LAG before, so the score matmuls that wait for this exponential also
                            # order the next value matmul into that update's PSUM bank.
                            e_ins.depends_on(after, kind=nisa.dependency.runtime)
                    o("act", nisa.activation(dst=alpha, op=nl.exp, data=nm_old, scale=-1.0, bias=nm_new))
                else:
                    # alpha = exp(m_old - m_new).
                    o("act", nisa.activation(dst=alpha, op=nl.exp, data=nm_old, scale=-1.0, bias=nm_new))
                    o(
                        "act",
                        nisa.activation(
                            dst=p_buf[0:P, slot, 0:width],
                            op=nl.exp,
                            data=ps[0:P, 0:width],
                            scale=scale,
                            bias=nm_new,
                            reduce_op=nl.add,
                            reduce_res=p_sum,
                            reduce_cmd=nisa.reduce_cmd.reset_reduce,
                        ),
                    )
                if mxv and pt_ps is not None:
                    return  # p^T by TensorE transposes in the value stage
                if mxv:
                    # p^T as 32-bit words by one DMA transpose of the FP8 weights viewed as fp32, SBUF to
                    # SBUF: word w of the 128 words of tile t of a query row goes to partition w, tile t.
                    nw = kt // MX_PACK  # words of one slot
                    nisa.dma_transpose(
                        priority=PT_PRIORITY,
                        dst=ptw_buf.ap([[N_PTW * nw, P], [1, 1], [P, n_mx], [1, P]], offset=(step % N_PTW) * nw),
                        src=p8_buf.ap(
                            [[N_P8 * nw, P], [1, 1], [P, n_mx], [1, P]], offset=(step % N_P8) * nw, dtype=_F32
                        ),
                    )
                    return
                # p^T by one DMA transpose, SBUF to SBUF: key tile j of p goes to
                # p_t[:, j * P : (j + 1) * P], keys on the partitions.
                nisa.dma_transpose(
                    dst=pt_buf.ap([[NBUF * kt, P], [1, 1], [P, n_sub], [1, P]], offset=slot * kt),
                    src=p_buf.ap([[NBUF * kt, P], [1, 1], [P, n_sub], [1, P]], offset=slot * kt),
                )

            def stat():
                if rnd is not None:
                    return  # once a round (round_l)
                # l = l * alpha + p_sum.
                o(
                    "dve",
                    nisa.scalar_tensor_tensor(
                        dst=l_run[0:P, idx : idx + 1],
                        data=l_run[0:P, idx : idx + 1],
                        op0=nl.multiply,
                        operand0=alpha,
                        op1=nl.add,
                        operand1=p_sum,
                    ),
                )

            def cast_act():
                # The weights rounded to FP8 (nearest even, as the GPU's FP8 attention): the first half
                # of the key tiles on ScalarE, the rest on VectorE (``cast_dve``). With the prescale
                # the casts round 2^8 p (exact in bf16).
                h = (n_sub // 2) * P
                if prescale:
                    o(
                        "act",
                        nisa.activation(dst=pt8_buf[0:P, slot, 0:h], op=nl.copy, data=p_t[0:P, 0:h], scale=PS_SCALE),
                    )
                    return
                o("act", nisa.tensor_copy(dst=pt8_buf[0:P, slot, 0:h], src=p_t[0:P, 0:h], engine=nisa.scalar_engine))

            def cast_dve():
                h = (n_sub // 2) * P
                if prescale:
                    o(
                        "dve",
                        nisa.tensor_scalar(
                            dst=pt8_buf[0:P, slot, h:width],
                            data=p_t[0:P, h:width],
                            op0=nl.multiply,
                            operand0=PS_SCALE,
                            engine=nisa.vector_engine,
                        ),
                    )
                    return
                o(
                    "dve",
                    nisa.tensor_copy(dst=pt8_buf[0:P, slot, h:width], src=p_t[0:P, h:width], engine=nisa.vector_engine),
                )

            def values(pv=pv):
                if pv is None:
                    pv = nl.ndarray((P, R), dtype=_F32, buffer=nl.psum, name=nmx("pv"))
                if mxv and pt_ps is not None:
                    # p^T as 32-bit words: fp32 TensorE transposes (bit exact) of the weights of each 512 keys,
                    # then one copy to the step's p^T slot (the DMA transpose's output layout). The copy runs on
                    # the engine that the value stage of this step frees first (``pt_eng``).
                    nw = kt // MX_PACK
                    for t in range(n_mx):
                        o(
                            "pe",
                            _pe_transpose(
                                pt_ps[0:P, t * P : (t + 1) * P],
                                p8_buf.ap(
                                    pattern=[[N_P8 * nw, P], [1, P]], offset=(step % N_P8) * nw + t * P, dtype=_F32
                                ),
                                id32,
                            ),
                        )
                    eng = pt_eng[0]
                    o(
                        "dve" if eng == nisa.vector_engine else "act",
                        nisa.tensor_copy(
                            dst=ptw_buf[0:P, step % N_PTW, 0 : n_mx * P], src=pt_ps[0:P, 0 : n_mx * P], engine=eng
                        ),
                    )
                if mxv:
                    # MX matmuls of 512 keys each: the p^T words of tile t against the value rows of
                    # the same keys in the MX packing (``v_rows`` is ``vmx``, ``v_sub0`` its group).
                    for t in range(n_mx):
                        o(
                            "pe",
                            nisa.nc_matmul_mx(
                                dst=pv,
                                stationary=ptw_buf.ap(
                                    pattern=[[N_PTW * kt // MX_PACK, P], [1, P]],
                                    offset=(step % N_PTW) * kt // MX_PACK + t * P,
                                    dtype=nl.float8_e4m3fn_x4,
                                ),
                                moving=v_rows.ap(
                                    pattern=[[NSTEP * R, P], [1, R]],
                                    offset=(v_sub0 + t) * R,
                                    dtype=nl.float8_e4m3fn_x4,
                                ),
                                stationary_scale=(mx_scale_p if prescale else mx_scale)[0:P, 0:P],
                                moving_scale=mx_scale[0:P, 0:R],
                                accumulate=t > 0,
                            ),
                        )
                elif mm8:
                    # Double-row matmuls of two key tiles each with the FP8 cache rows.
                    for j in range(0, n_sub, 2):
                        pair = j + 1 < n_sub
                        o(
                            "pe",
                            nisa.nc_matmul(
                                dst=pv,
                                stationary=(
                                    pt8_buf.ap(pattern=[[NBUF * kt, P], [P, 2], [1, P]], offset=slot * kt + j * P)
                                    if pair
                                    else pt8_buf[0:P, slot, j * P : (j + 1) * P]
                                ),
                                moving=(
                                    v_rows[0:P, v_sub0 + j : v_sub0 + j + 2, 0:R]
                                    if pair
                                    else v_rows[0:P, v_sub0 + j, 0:R]
                                ),
                                accumulate=(j > 0),
                                perf_mode=nisa.matmul_perf_mode.double_row if pair else nisa.matmul_perf_mode.none,
                            ),
                        )
                else:
                    for j in range(n_sub):
                        nisa.nc_matmul(
                            dst=pv,
                            stationary=p_t[0:P, j * P : (j + 1) * P],
                            moving=v_rows[0:P, v_sub0 + j, 0:R],
                            accumulate=(j > 0),
                        )
                # acc = acc * alpha + pv.
                acc_ins = nisa.scalar_tensor_tensor(
                    dst=acc[0:P, idx, 0:R],
                    data=acc[0:P, idx, 0:R],
                    op0=nl.multiply,
                    operand0=alpha,
                    op1=nl.add,
                    operand1=pv,
                )
                if chain_order and od is not None:
                    # The ISA call returns a raw handle, which the engine order would skip: wrap it, so that
                    # VectorE issues it after the max and -m of the step emitted before it.
                    if not hasattr(acc_ins, "depends_on"):
                        acc_ins = nisa.NkiInstruction(acc_ins)
                    acc_last[0] = acc_ins
                o("dve", acc_ins)

            if mxv:
                # The caller interleaves the stages of consecutive steps (``run_block``).
                return soft, stat, values, acc_last
            if mm8:
                return soft, stat, cast_act, cast_dve, values
            soft()
            stat()
            values()
            return None

        def own_init_stages(nmx, idx, slot, pe_pt=False):
            """Give the own-key step of column ``idx`` as two stages, like ``attend`` with the MX values.

            The own-key step is the first step of a column, so it sets the
            state: ``-m`` in slot 0 of ``nm_run``, ``l`` and ``acc``. No reset
            of the state and no rescale is needed. With ``pe_pt``
            (``own_drain_transpose``) the value stage makes p^T by TensorE
            transposes and a copy, instead of the softmax's DMA transpose.
            """
            qt = idx % NQT
            t_max = st_buf[0:P, slot, 0:1]
            n_sub = n_kc
            p_t = pt_buf[0:P, slot, :]
            nm0 = nm_run[0:P, 0, idx : idx + 1]

            def scores_softmax():
                ps = nl.ndarray((P, KT), dtype=_F32, buffer=nl.psum, name=nmx("ps_own"))
                if qmx:
                    for part in Q_PARTS:
                        nisa.nc_matmul_mx(
                            dst=ps[0:P, 0:Kc],
                            stationary=mx_query(part, idx),
                            moving=kc_mx.ap(pattern=[[Kc, P], [1, Kc]], offset=0, dtype=nl.float8_e4m3fn_x4),
                            stationary_scale=mx_query_scale(idx, mx_scale[0:P, 0:P]),
                            moving_scale=mx_scale[0:P, 0:Kc],
                            accumulate=part > 0,
                        )
                    w = rows_qk[NF - 1]
                    nisa.nc_matmul(
                        dst=ps[0:P, 0:Kc],
                        stationary=q_fm[0:w, idx, 0:P],
                        moving=kc_rope[0:w, rope_t, 0:Kc],
                        accumulate=True,
                    )
                for f in () if qmx else range(NF):
                    w = rows_qk[f]
                    nisa.nc_matmul(
                        dst=ps[0:P, 0:Kc],
                        stationary=q_fm[0:w, idx * NF + f, 0:P],
                        moving=kc_fm[0:w, f, 0:Kc],
                        accumulate=(f > 0),
                    )
                src = sc_buf[0:P, slot, :]
                nisa.scalar_tensor_tensor(
                    dst=src[0:P, 0:Kc],
                    data=ps[0:P, 0:Kc],
                    op0=nl.multiply,
                    operand0=scale,
                    op1=nl.add,
                    operand1=b_sb[0:P, qt, 0:Kc],
                )
                if own_at_end:
                    # -max in one VectorE op: at a group end ScalarE also runs the query split.
                    nisa.tensor_reduce(dst=nm0, op=nl.maximum, data=src[0:P, 0:Kc], axis=1, negate=True)
                else:
                    nisa.tensor_reduce(dst=t_max, op=nl.maximum, data=src[0:P, 0:Kc], axis=1)
                    nisa.tensor_scalar(dst=nm0, data=t_max, op0=nl.multiply, operand0=-1.0)
                # l = sum(p) straight from the exponential.
                nisa.activation(
                    dst=p_buf[0:P, slot, 0:Kc],
                    op=nl.exp,
                    data=src[0:P, 0:Kc],
                    bias=nm0,
                    reduce_op=nl.add,
                    reduce_res=l_run[0:P, idx : idx + 1],
                    reduce_cmd=nisa.reduce_cmd.reset_reduce,
                )
                if pe_pt:
                    return  # p^T by TensorE transposes in the value stage
                nisa.dma_transpose(
                    priority=PT_PRIORITY,
                    dst=pt_buf.ap([[NBUF * pw, P], [1, 1], [P, n_sub], [1, P]], offset=slot * pw),
                    src=p_buf.ap([[NBUF * pw, P], [1, 1], [P, n_sub], [1, P]], offset=slot * pw),
                )

            def values():
                if pe_pt:
                    # own_drain_transpose: key tile j of p (queries on the partitions) to p_t[:, j * P : (j + 1) * P]
                    # (keys on the partitions), the DMA transpose's layout: bf16 TensorE transposes (bit exact) into
                    # one PSUM tile and one VectorE copy.
                    ps_pt = nl.ndarray((P, n_sub * P), dtype=dt, buffer=nl.psum, name=nmx("pt_own"))
                    for j in range(n_sub):
                        _pe_transpose(ps_pt[0:P, j * P : (j + 1) * P], p_buf[0:P, slot, j * P : (j + 1) * P], id_sb)
                    nisa.tensor_copy(
                        dst=p_t[0:P, 0 : n_sub * P], src=ps_pt[0:P, 0 : n_sub * P], engine=nisa.vector_engine
                    )
                pv = nl.ndarray((P, R), dtype=_F32, buffer=nl.psum, name=nmx("pv_own"))
                for j in range(n_sub):
                    nisa.nc_matmul(
                        dst=pv,
                        stationary=p_t[0:P, j * P : (j + 1) * P],
                        moving=kc_rows[0:P, j, 0:R],
                        accumulate=(j > 0),
                    )
                # ScalarE has spare time in the own-key steps; at a group end (own_at_end) VectorE takes a
                # share. With the prescale of the double-row values acc holds 2^8 times the sums (the cache
                # steps' value matmuls take 2^8 p).
                if prescale and not mxv:
                    nisa.activation(dst=acc[0:P, idx, 0:R], op=nl.copy, data=pv, scale=PS_SCALE)
                else:
                    dve = own_at_end and qt in OWN_COPY_DVE
                    eng = nisa.vector_engine if dve else nisa.scalar_engine
                    nisa.tensor_copy(dst=acc[0:P, idx, 0:R], src=pv, engine=eng)

            return scores_softmax, values

        # Two buffer sets for the cache blocks, [set][...]. The block in one set
        # is attended while the next block loads into the other.
        pages_ab = [nl.ndarray((1, SUBS), dtype=nl.int32, buffer=nl.sbuf, name=nm("pages")) for _ in range(2)]
        # With mm8 the rows stay FP8 (the value operand). With q8 the transposed keys are an
        # FP8 latent tensor and a bf16 rope tile, which holds the marker.
        rows_ab = [
            nl.ndarray(
                (P, NSTEP, R * MX_PACK) if quad else (P, SUBS, D),
                dtype=cache.dtype if mm8 else dt,
                buffer=nl.sbuf,
                name=nm("rows"),
            )
            for _ in range(1 if mxv else 2)
        ]
        if mxv:
            # With the MX values the steps read the packed copies (k_mx, k_fmr, vmx) only, and a block's rows
            # are packed while the block before it is attended. So the next block can load into the same
            # rows as soon as the block starts: the two sets share one rows tensor (and one loaded marker).
            rows_ab.append(rows_ab[0])
        if q8:
            kfm_ab = [
                (
                    nl.ndarray((P, NFL, BLOCK), dtype=cache.dtype, buffer=nl.sbuf, name=nm("k_fm8")),
                    nl.ndarray((P, 1, BLOCK), dtype=dt, buffer=nl.sbuf, name=nm("k_fmr")),
                )
                for _ in range(2)
            ]
        elif mxv:
            # The latent keys in the MX packing (32-bit words, mx_words), the bf16 rope tile with the marker,
            # the value rows in the MX packing (``vmx``) and the marker as loaded (``mk_in``). The score
            # columns of a 512-key group g of a block are in the MX value order: column 512 g + 4 w + i
            # holds key (4 g + i) * P + w, which row tile 4 g + i holds at partition w (``mx_perm``).
            kfm_ab = [
                (
                    "mxv",
                    nl.ndarray((P, BLOCK), dtype=_F32, buffer=nl.sbuf, name=nm("k_mx")),
                    nl.ndarray((rows_qk[NF - 1], 1, BLOCK), dtype=dt, buffer=nl.sbuf, name=nm("k_fmr")),
                    nl.ndarray((P, NSTEP, R * MX_PACK), dtype=cache.dtype, buffer=nl.sbuf, name=nm("vmx")),
                )
                for _ in range(2)
            ]
            mk_in = nl.ndarray((1, BLOCK), dtype=dt, buffer=nl.sbuf, name=nm("mk_in"))
            kfm_ab = [kf + (mk_in,) for kf in kfm_ab]
        elif qmx:
            # The latent keys in the MX packing (32-bit words, mx_words) and the bf16 rope tile with the marker.
            kfm_ab = [
                (
                    "mx",
                    nl.ndarray((P, BLOCK), dtype=_F32, buffer=nl.sbuf, name=nm("k_mx")),
                    nl.ndarray((P, 1, BLOCK), dtype=dt, buffer=nl.sbuf, name=nm("k_fmr")),
                )
                for _ in range(2)
            ]
        else:
            kfm_ab = [nl.ndarray((P, NF, BLOCK), dtype=dt, buffer=nl.sbuf, name=nm("k_fm")) for _ in range(2)]
        mk_ab = [None, None]
        if not fold:
            mk_ab = [nl.ndarray((1, BLOCK), dtype=dt, buffer=nl.sbuf, name=nm("mk")) for _ in range(2)]
        if mxv:
            # The marker in the score columns' order: the spare partition of the rope tile, or the marker row.
            kfm_ab = [
                kf + (kf[2][w_tail : w_tail + 1, 0, 0:BLOCK] if fold else mk_ab[b][0:1, 0:BLOCK],)
                for b, kf in enumerate(kfm_ab)
            ]
        if quad:
            # The key-quad reader loads the value operand (``vmx``, index 3) and a row-major rope tile straight
            # from the page, so it never materializes the row-major ``rows`` tile. ``rope_q`` (index 6) is a
            # block's rope keys (``rows[:, :, R:D]`` without the quad layout); it rides with the set's kfm.
            kfm_ab = [
                kf + (nl.ndarray((P, SUBS, DR), dtype=cache.dtype, buffer=nl.sbuf, name=nm("rope_q")),)
                for kf in kfm_ab
            ]
        if fp8_rows:
            # An FP8 pool slot at or after prior_tokens may hold any byte, NaN codes included (a
            # shared pool's bytes that the MLA did not write). The marker gives such a key weight 0,
            # but a NaN key or value would still make NaN sums, and a transpose (a matmul) spreads a
            # NaN to all keys of its tile. So the rows load row by row (a gather): each row at or
            # after prior_tokens gets an index past the pool, and the gather skips it. Its SBUF row
            # keeps an earlier, finite row; the rows start as zeros.
            # The gather index is a row index (fp32-exact under 2^24 rows); the DGE scales it by the
            # row stride wider than 32 bits, so the gather reaches a pool at or over 4 GiB. The
            # hidden-row index NP * PG stays past the pool and the DMA skips it (measured on Trn3).
            assert NP * PG <= 2**24, "an FP8 pool must have at most 2^24 rows"  # noqa: S101
            rows_flat = cache.reshape((NP * PG, D))
            if not quad:
                for rows in rows_ab[: 1 if mxv else 2]:
                    nisa.memset(dst=rows, value=0.0)
            # The per-partition operands are fp32 (exact: the slots and pool rows are under 2^24).
            prior_i = scalar_tile(P, nl.int32, nm("prior_i"))
            nisa.dma_copy(dst=prior_i, src=prior_tokens.ap(pattern=[[0, P], [1, 1]], offset=0))
            prior_b = scalar_tile(P, _F32, nm("prior_b"))
            nisa.tensor_copy(dst=prior_b, src=prior_i, engine=nisa.gpsimd_engine)
            if wrap:
                # n_blocks * 2048 on every partition (fp32, exact), for the wrap of the row gathers.
                nbB_i = scalar_tile(P, nl.int32, nm("nbB_i"))
                nisa.tensor_scalar(dst=nbB_i, data=prior_i, op0=nl.minimum, operand0=S, op1=nl.add, operand1=BLOCK - 1)
                nisa.tensor_scalar(
                    dst=nbB_i,
                    data=nbB_i,
                    op0=nl.right_shift,
                    operand0=BLOCK.bit_length() - 1,
                    op1=nl.left_shift,
                    operand1=BLOCK.bit_length() - 1,
                )
                nbB_b = scalar_tile(P, _F32, nm("nbB_b"))
                nisa.tensor_copy(dst=nbB_b, src=nbB_i, engine=nisa.gpsimd_engine)
            part_i = scalar_tile(P, nl.int32, nm("part_i"))
            nisa.iota(dst=part_i, pattern=[[0, 1]], offset=0, channel_multiplier=1)
            part_b = scalar_tile(P, _F32, nm("part_b"))
            nisa.tensor_copy(dst=part_b, src=part_i, engine=nisa.gpsimd_engine)
            # slot0[p, sub] = sub * P + p, the slot of a row in its block.
            slot0 = nl.ndarray((P, SUBS), dtype=nl.int32, buffer=nl.sbuf, name=nm("slot0"))
            nisa.iota(dst=slot0, pattern=[[P, SUBS]], offset=0, channel_multiplier=1)
            # The first slot of the next block, and of the block being loaded, on every partition.
            kb_b = scalar_tile(P, _F32, nm("kb_b"))
            klb_b = scalar_tile(P, _F32, nm("klb_b"))
            # page_end[0, sub] = (sub + 1) * P, the end slot of each page of block 0.
            page_end = nl.ndarray((1, SUBS), dtype=nl.int32, buffer=nl.sbuf, name=nm("page_end"))
            nisa.iota(dst=page_end, pattern=[[P, SUBS]], offset=P, channel_multiplier=0)
            pages_b = [nl.ndarray((P, SUBS), dtype=nl.int32, buffer=nl.sbuf, name=nm("pages_b")) for _ in range(2)]
            ridx_ab = [nl.ndarray((P, SUBS), dtype=nl.int32, buffer=nl.sbuf, name=nm("ridx")) for _ in range(2)]

        def load_rows(bset, first, n_sub=SUBS):
            """Load the rows of a block of an FP8 pool into set ``bset``, skipping those at or after prior_tokens.

            With ``n_sub``, only the first ``n_sub`` pages of the block.
            """
            pb, ridx, rows = pages_b[bset], ridx_ab[bset], rows_ab[bset]
            gps = nisa.gpsimd_engine
            if first:
                nisa.dma_copy(dst=pb, src=block_table.ap(pattern=[[0, P], [1, SUBS]], offset=0))
                nisa.tensor_scalar(dst=ridx, data=slot0, op0=nl.greater_equal, operand0=prior_b, engine=gps)
            else:
                if wrap:
                    # The block's first slot, or 0 (block 0) past the last block.
                    nisa.tensor_scalar(
                        dst=klb_b, data=kb_b, op0=nl.less, operand0=nbB_b, op1=nl.multiply, operand1=kb_b, engine=gps
                    )
                else:
                    nisa.tensor_scalar(dst=klb_b, data=kb_b, op0=nl.minimum, operand0=S - BLOCK, engine=gps)
                nisa.dma_copy(
                    dst=pb,
                    src=block_table.ap(pattern=[[0, P], [1, SUBS]], offset=0, scalar_offset=pl_sb, indirect_dim=1),
                )
                nisa.tensor_scalar(
                    dst=ridx, data=slot0, op0=nl.add, operand0=klb_b, op1=nl.greater_equal, operand1=prior_b, engine=gps
                )
            # ridx = page * PG + p, or the pool size (past the pool) for a hidden row.
            nisa.tensor_scalar(dst=ridx, data=ridx, op0=nl.multiply, operand0=NP * PG, engine=gps)
            nisa.tensor_scalar(dst=pb, data=pb, op0=nl.multiply, operand0=PG, op1=nl.add, operand1=part_b, engine=gps)
            nisa.tensor_tensor(dst=ridx, data1=ridx, data2=pb, op=nl.maximum, engine=gps)
            for sub in range(n_sub):
                nisa.dma_copy(
                    dst=rows[0:P, sub, 0:D],
                    src=rows_flat.ap(
                        pattern=[[D, P], [1, D]],
                        offset=0,
                        vector_offset=ridx[0:P, sub : sub + 1].view(nl.uint32),
                        indirect_dim=0,
                    ),
                    oob_mode=nisa.oob_mode.skip,
                )
            if not first:
                nisa.tensor_scalar(dst=kb_b, data=kb_b, op0=nl.add, operand0=BLOCK, engine=gps)

        def load_block(bset, first=False, reload=False, n_sub=SUBS):
            """Load the next cache block (logical slot ``k_sb``) into buffer set ``bset``.

            With ``first``, load the first block of the view (static offsets)
            and keep the counters. With ``reload`` (and ``first``), set 0 already
            holds the first block from an earlier load, unless a later block
            overwrote it, which a view of one block never does. With ``n_sub``
            (an FP8 pool), only the rows of the first ``n_sub`` pages load.
            With ``wrap``, a load past the prior's last block loads block 0.
            """
            pages, rows, k_fm = pages_ab[bset], rows_ab[bset], kfm_ab[bset]
            if not first and wrap:
                # The next block, or block 0 past the prior's last block (the next group's first block).
                gps = nisa.gpsimd_engine
                nisa.tensor_tensor(dst=wr_sb, data1=k_sb, data2=nbB_sb, op=nl.less, engine=gps)
                nisa.tensor_tensor(dst=kl_sb, data1=wr_sb, data2=k_sb, op=nl.multiply, engine=gps)
                nisa.tensor_tensor(dst=pl_sb, data1=wr_sb, data2=pg_sb, op=nl.multiply, engine=gps)
            elif not first:
                nisa.tensor_scalar(dst=kl_sb, data=k_sb, op0=nl.minimum, operand0=S - BLOCK, engine=nisa.gpsimd_engine)
                nisa.tensor_scalar(
                    dst=pl_sb, data=pg_sb, op0=nl.minimum, operand0=NBT - SUBS, engine=nisa.gpsimd_engine
                )
            if quad:
                # Load the latent into the staging ``rows`` (the vmx layout) and the rope keys into
                # ``rope_q``, straight from the key-quad page (whole quad rows and whole rope rows: the writer
                # zeroes the lanes and rows past the sequence end, so no NaN loads; the marker still hides
                # them). The value operand ``vmx`` is written later by a deferred ``emit_vcopy`` bank, so a
                # prefetch of the next block (the half-tail ``load_block(0)``) does not clobber the live value
                # operand before this block's lagged value matmuls read it (as the row-major pack defers).
                rope_q = k_fm[6]
                cache_flat = cache.reshape((NP, PG * D))
                if first:
                    nisa.dma_copy(dst=pages, src=block_table[0:1, 0:SUBS])
                else:
                    nisa.dma_copy(
                        dst=pages,
                        src=block_table.ap(
                            pattern=[[NBT, 1], [1, SUBS]], offset=0, scalar_offset=pl_sb, indirect_dim=1
                        ),
                    )
                for sub in range(n_sub):
                    g, sp = sub // MX_PACK, sub % MX_PACK
                    # The 32 latent quad rows of the page onto partitions [32 sp : 32 sp + 32] of group g:
                    # partition p = 32 sp + j packs the 4 consecutive keys sp * 128 + 4 j + i of the group.
                    nisa.dma_copy(
                        dst=rows[QUAD_ROWS * sp : QUAD_ROWS * (sp + 1), g, 0:QUAD_BYTES],
                        src=cache_flat.ap(
                            pattern=[[QUAD_BYTES, QUAD_ROWS], [1, QUAD_BYTES]],
                            offset=0,
                            scalar_offset=pages[0:1, sub : sub + 1],
                            indirect_dim=0,
                        ),
                    )
                    # The rope rows, row-major after the latent block (key sub * 128 + p on partition p).
                    nisa.dma_copy(
                        dst=rope_q[0:P, sub, 0:DR],
                        src=cache_flat.ap(
                            pattern=[[DR, P], [1, DR]],
                            offset=ROPE_BASE,
                            scalar_offset=pages[0:1, sub : sub + 1],
                            indirect_dim=0,
                        ),
                    )
            elif fp8_rows and not reload:
                load_rows(bset, first, n_sub)
            elif fp8_rows:
                # A reload of block 0 by page: the pages with a slot at or after prior_tokens are
                # skipped (an index past the pool). They exist only in a view of one block, where set 0
                # still holds them from its first load (a gather); a longer view has none in block 0.
                gps = nisa.gpsimd_engine
                nisa.dma_copy(dst=pages, src=block_table[0:1, 0:SUBS])
                hid = nl.ndarray((1, SUBS), dtype=nl.int32, buffer=nl.sbuf, name=nm("hid"))
                nisa.tensor_scalar(
                    dst=hid,
                    data=page_end,
                    op0=nl.greater,
                    operand0=prior_b[0:1, 0:1],
                    op1=nl.multiply,
                    operand1=NP,
                    engine=gps,
                )
                nisa.tensor_tensor(dst=pages, data1=pages, data2=hid, op=nl.maximum, engine=gps)
                for sub in range(SUBS):
                    nisa.dma_copy(
                        dst=rows[0:P, sub, 0:D],
                        src=cache.ap(
                            pattern=[[D, P], [1, D]], offset=0, scalar_offset=pages[0:1, sub : sub + 1], indirect_dim=0
                        ),
                        oob_mode=nisa.oob_mode.skip,
                    )
            else:
                if first:
                    nisa.dma_copy(dst=pages, src=block_table[0:1, 0:SUBS])
                else:
                    # The physical pages of the logical pages pl .. pl + 16.
                    nisa.dma_copy(
                        dst=pages,
                        src=block_table.ap(
                            pattern=[[NBT, 1], [1, SUBS]], offset=0, scalar_offset=pl_sb, indirect_dim=1
                        ),
                    )
                for sub in range(SUBS):
                    # Logical slots kl + sub * P + p are the rows p of one page.
                    # The page index scales by the page stride, 128 * D.
                    nisa.dma_copy(
                        dst=rows[0:P, sub, 0:D],
                        src=cache.ap(
                            pattern=[[D, P], [1, D]], offset=0, scalar_offset=pages[0:1, sub : sub + 1], indirect_dim=0
                        ),
                    )
            if first:
                mk_src = marker[0:1, 0:BLOCK]
            else:
                mk_src = marker.ap(pattern=[[S, 1], [1, BLOCK]], offset=0, scalar_offset=kl_sb, indirect_dim=1)
            if fold and q8:
                nisa.dma_copy(dst=k_fm[1][w_tail : w_tail + 1, 0, 0:BLOCK], src=mk_src)
            elif mxv:
                # The marker in slot order; the block's transposes permute it into the score columns.
                nisa.dma_copy(dst=k_fm[4][0:1, 0:BLOCK], src=mk_src)
            elif fold and qmx:
                nisa.dma_copy(dst=k_fm[2][w_tail : w_tail + 1, 0, 0:BLOCK], src=mk_src)
            elif fold:
                nisa.dma_copy(dst=k_fm[w_tail : w_tail + 1, NF - 1, 0:BLOCK], src=mk_src)
            else:
                nisa.dma_copy(dst=mk_ab[bset], src=mk_src)
            if not first:
                nisa.tensor_scalar(dst=k_sb, data=k_sb, op0=nl.add, operand0=BLOCK, engine=nisa.gpsimd_engine)
                nisa.tensor_scalar(dst=pg_sb, data=pg_sb, op0=nl.add, operand0=SUBS, engine=nisa.gpsimd_engine)

        def new_pipe(nmx):
            """Give the state of a software pipeline of FP8 cache steps (mm8), for one or more blocks.

            The pipeline keeps its own PSUM tiles for each chain: two score
            tiles (two banks each), two value sums and the key transposes.
            With a PSUM tile for each step, the allocator rotates every tile
            over all banks, so a step waits for any other tile that last used
            its bank. ``od`` keeps the issue order of each engine.
            """
            pipe = {
                "od": _EngineOrder(),
                "ps": nl.ndarray((P, 2, kt), dtype=_F32, buffer=nl.psum, name=nmx("ps_sc2")),
                "pv": nl.ndarray((P, 2, R), dtype=_F32, buffer=nl.psum, name=nmx("pv2")),
                "tp": nl.ndarray((P, TP_BANK * P), dtype=dt, buffer=nl.psum, name=nmx("ps_tp")),
                "stages": [],
            }
            if qmx:
                # The 32-bit key transposes (transpose mode) cannot share a PSUM tensor with matmuls.
                pipe["tpx"] = nl.ndarray((P, KT), dtype=_F32, buffer=nl.psum, name=nmx("ps_tpx"))
            return pipe

        def pipe_step(pipe, attend_args):
            """Emit step ``g`` of the pipeline and the delayed stages of the earlier steps.

            Each engine issues the stages in a fixed order (``_EngineOrder``),
            so a stage waits only for its own inputs. Step g emits its scores
            and max, then the FP8 rounding of step g - MM8_LAG + 1 (its DMA
            transpose is done), its exponential, the value matmuls of step
            g - MM8_LAG, and the row sum of step g - 1. ScalarE rounds its half
            before the exponential, and VectorE sums the rows one step late, so
            neither waits for the other's chain of max, exponential and row sum.
            With MX values (``mxv``) a step emits its scores and max, its
            exponentials, the value stage of step g - MXV_LAG, and at the
            first step of a round the row sums of the round before.
            """
            stages = pipe["stages"]
            g = len(stages)
            nmx, idx, *rest = attend_args
            ncol = G * NQT
            pt_ps = None
            if mxv and g >= pipe.get("n_steps", 1 << 30) - PE_DRAIN:
                # A drain step (``PE_DRAIN``): its p^T by TensorE transposes, into a half of the pipeline's
                # 32-bit transpose bank (the next block's key transposes are done by then). The last step's
                # copy runs on ScalarE (free after its exponentials), the earlier ones' on VectorE.
                nw = kt // MX_PACK
                h = (g % 2) * nw
                pt_ps = (pipe["tpx"][0:P, h : h + nw], g == pipe["n_steps"] - 1)
            st = attend(
                nmx,
                idx,
                g % NBUF,
                *rest,
                ps=pipe["ps"][0:P, g % 2, 0:kt],
                pv=pipe["pv"][0:P, g % 2, 0:R],
                od=pipe["od"],
                rnd=(g // ncol) % N_RND if mxv else None,
                step=g,
                pt_ps=pt_ps,
            )
            stages.append(st)
            if mxv and chain_order:
                # softmax_chain_order: the value stage of step g - MXV_LAG first (on VectorE its acc update
                # follows step g's max and -m), then step g's exponentials, the last of which waits for it.
                prev_acc = None
                if g - MXV_LAG >= 0:
                    stages[g - MXV_LAG][2]()
                    prev_acc = stages[g - MXV_LAG][3][0]
                st[0](prev_acc)
                if g % ncol == 0 and g > 0:
                    round_l((g // ncol - 1) % N_RND, pipe["od"])
                return
            if mxv:
                # No rounding stage: the exponential writes the FP8 weights. The row sums of a round
                # update one step into the next round (the last alpha of the round is done by then).
                st[0]()
                if g - MXV_LAG >= 0:
                    stages[g - MXV_LAG][2]()
                if g % ncol == 0 and g > 0:
                    round_l((g // ncol - 1) % N_RND, pipe["od"])
                return
            c = g - MM8_LAG + 1
            if c >= 0:
                stages[c][2]()
                stages[c][3]()
            st[0]()
            if c - 1 >= 0:
                stages[c - 1][4]()
            if g >= 1:
                stages[g - 1][1]()

        def pipe_drain(pipe):
            """Emit the delayed stages of the last steps of the pipeline."""
            stages = pipe["stages"]
            n = len(stages)
            if mxv:
                for c in range(max(0, n - MXV_LAG), n):
                    stages[c][2]()
                if n:
                    round_l(((n - 1) // (G * NQT)) % N_RND, pipe["od"])
                return
            if n:
                stages[n - 1][1]()
            for c in range(max(0, n - MM8_LAG), n):
                if c > n - MM8_LAG:
                    stages[c][2]()
                    stages[c][3]()
                stages[c][4]()

        def run_block(nmx, bset, nset=None, n_kstep=NSTEP, pipe=None, hooks=None, hooks_first=False):
            """Attend the transposed cache block in buffer set ``bset`` for the group.

            With ``nset``, the key transposes of the block that loads into set
            ``nset`` meanwhile run between the steps, from step K_AFTER on (its
            pages have arrived by then), so they are ready when that block
            starts. The steps that cover the first ``n_kstep * KT`` keys of
            the block run (``block_steps``). With mm8 the steps run in the
            software pipeline ``pipe`` (``new_pipe``), which the caller drains,
            else in one of their own. ``hooks`` maps a step of the block to
            more code to emit after it.
            """
            rows, k_fm = rows_ab[bset], kfm_ab[bset]
            steps = block_steps(n_kstep)
            n_steps = len(steps) * G * NQT
            own = mm8 and pipe is None
            if own:
                pipe = new_pipe(nmx)
            at = {}
            bank_items = []
            if nset is not None:
                # With mm8 ScalarE also rounds each step's p^T to FP8, so the
                # copies of these transposes run on VectorE.
                if qmx:
                    banks = (transpose_banks_mxv if mxv else transpose_banks_mx)(
                        nmx,
                        kfm_ab[nset],
                        rows_ab[nset],
                        SUBS,
                        copy_engine=nisa.vector_engine,
                        ps_tp=pipe["tp"],
                        ps_tpx=pipe["tpx"],
                        od=pipe["od"],
                    )
                else:
                    banks = transpose_banks(
                        nmx,
                        kfm_ab[nset],
                        rows_ab[nset],
                        SUBS,
                        copy_engine=nisa.vector_engine if mm8 else None,
                        ps_tp=pipe["tp"] if mm8 else None,
                        od=pipe["od"] if mm8 else None,
                    )
                stride = max(1, (n_steps - K_AFTER - 8) // len(banks))
                bank_items = [(min(n_steps - 1, K_AFTER + j * stride), emit) for j, emit in enumerate(banks)]
            hook_items = [(min(n_steps - 1, i), fn) for i, fn in (hooks or {}).items()]
            # A hook that loads the next block into set ``nset`` (``hooks_first``) must run before that set's
            # key transposes, which read the just-loaded rows. A short block (G * NQT <= 2) clamps both the
            # hook and the banks onto the last step; add the hooks first there. With G * NQT >= 3 the hook
            # (step MM8_LAG) and the banks (step >= K_AFTER, clamped) land on different steps, so the per-step
            # order is the same either way and the schedule is byte-identical.
            ordered = (hook_items + bank_items) if hooks_first else (bank_items + hook_items)
            for step, fn in ordered:
                at.setdefault(step, []).append(fn)
            for t, (k0, width) in enumerate(steps):
                for idx in range(G * NQT):
                    i = t * G * NQT + idx
                    if mm8:
                        if mxv:
                            pipe_step(pipe, (nmx, idx, t % 2, k_fm, k0, width, mk_ab[bset], k_fm[3], k0 // KT))
                        else:
                            pipe_step(pipe, (nmx, idx, t % 2, k_fm, k0, width, mk_ab[bset], rows, k0 // P))
                    else:
                        attend(nmx, idx, i % NBUF, t % 2, k_fm, k0, width, mk_ab[bset], rows, k0 // P)
                    for emit in at.get(i, ()):
                        emit()
            if own:
                pipe_drain(pipe)

        def pair_body(_, var):
            """Attend the two cache blocks in sets 0 and 1, and prefetch the next.

            Set 0 holds the first block, transposed, on entry. The second block
            loads and transposes while the first is attended, and the first
            block of the next pair while the second is attended. With mm8 the
            two blocks share one software pipeline, so the loads into set 0
            follow the last value matmuls of the first block.
            """
            nmp = _Namer(tid, f"pair{var}")
            staged = []
            if own_at_end:
                # The next group's queries; see ``stage_next``. With MX values their DMAs issue during block 0's
                # steps (after the block loads), one at a time, instead of all at once here.
                staged = stage_next(defer=mxv)
            load_block(1)
            if mm8:
                pipe = new_pipe(nmp)
                pipe["n_steps"] = 2 * len(block_steps(NSTEP)) * G * NQT
                # Each staged DMA waits for the latest VectorE instruction of its step.
                hooks = {
                    STAGE_STEP0 + j * STAGE_STRIDE: (lambda fn=fn: fn(pipe["od"].last["dve"]))
                    for j, fn in enumerate(staged)
                }
                run_block(nmp, 0, 1, pipe=pipe, hooks=hooks)
                # hooks_first: the load of the next block into set 0 must precede set 0's key transposes. A
                # short block (G * NQT <= 2) clamps both onto the last step, so the load must be added first,
                # else the transposes read the stale rows (the previous block) and set 0 packs the wrong block.
                run_block(nmp, 1, 0, pipe=pipe, hooks={MM8_LAG: lambda: load_block(0)}, hooks_first=True)
                pipe_drain(pipe)
            else:
                run_block(nmp, 0, 1)
                load_block(0)
                run_block(nmp, 1, 0)

        def tail_body(_, var, half_next=False):
            """Attend the first ``var`` key steps of the block after the block pairs, already in set 0.

            Without the half tail that block is the last one of an odd block
            count. A step reads the negated max from slot ``t % 2`` and writes
            slot ``1 - t % 2``, so after an odd step count the max is copied to
            slot 0, where the group end reads it. With ``half_next`` (the full
            tail block of ``half_tail``) the first half of the next block loads
            meanwhile, and its keys go to the first half of set 0 once this
            block's first step has read it, for the half tail that may follow.
            With ``wrap`` the whole next block loads (block 0 if no half tail
            follows), and its second half goes to set 0 after this block's
            second step; the half tail, the last block of its group, packs
            block 0 into set 0: its second half during the step and its first
            half after it. So the group end loads no block.
            """
            nmt = _Namer(tid, f"tail{var}{'h' if half_next else ''}")
            if own_at_end:
                stage_next()
            half_grp = KT_MM8 // (MX_PACK * P)  # the 512-key groups of a block half
            if half_next or wrap:
                pipe = new_pipe(nmt)
                n_steps = len(block_steps(var)) * G * NQT
                pipe["n_steps"] = n_steps

                def next_banks(**kw):
                    return transpose_banks_mxv(
                        nmt,
                        kfm_ab[0],
                        rows_ab[0],
                        SUBS,
                        copy_engine=nisa.vector_engine,
                        ps_tp=pipe["tp"],
                        ps_tpx=pipe["tpx"],
                        od=pipe["od"],
                        **kw,
                    )

            if half_next:
                # The next block's rows (the rows tensor is free: this block's keys are transposed). With ``wrap``
                # the whole block: past the last block it is block 0, the next group's first block.
                load_block(0, n_sub=SUBS if wrap else KT_MM8 // P)
                banks = next_banks(n_grp=half_grp)
                # The first step (G * NQT columns) reads the first half, and its value matmuls MXV_LAG steps later.
                first = G * NQT + MXV_LAG + 1
                # A short block runs the first step's last value matmuls in the drain, so its banks follow the drain.
                late = G * NQT + MXV_LAG > n_steps
                stride = max(1, (n_steps - first - 2) // len(banks))
                at = {}
                for j, emit in enumerate(() if late else banks):
                    at.setdefault(min(n_steps - 1, first + j * stride), []).append(emit)
                hooks = {i: (lambda fns=fns: [fn() for fn in fns]) for i, fns in at.items()}
                run_block(nmt, 0, n_kstep=var, pipe=pipe, hooks=hooks)
                pipe_drain(pipe)
                for emit in banks if late else ():
                    emit()
                if wrap:
                    # Its second half, once this block's second step has read set 0's second half.
                    for emit in next_banks(g0=half_grp):
                        emit()
            elif wrap:
                # The half tail, the last block of the group: block 0 (the next group's first block) loads, its
                # second half goes to set 0 during the step (which reads the first half only), and its first half
                # after the step.
                load_block(0)
                banks = next_banks(g0=half_grp)
                stride = max(1, (n_steps - K_AFTER - 2) // len(banks))
                at = {}
                for j, emit in enumerate(banks):
                    at.setdefault(min(n_steps - 1, K_AFTER + j * stride), []).append(emit)
                hooks = {i: (lambda fns=fns: [fn() for fn in fns]) for i, fns in at.items()}
                run_block(nmt, 0, n_kstep=var, pipe=pipe, hooks=hooks)
                pipe_drain(pipe)
                for emit in next_banks(n_grp=half_grp):
                    emit()
            else:
                run_block(nmt, 0, n_kstep=var)
            if len(block_steps(var)) % 2:
                nisa.tensor_copy(dst=nm_run[0:P, 0, :], src=nm_run[0:P, 1, :])

        # The rows of a group's queries, one DMA for each part: the heads
        # h0 .. h0 + G of a row are contiguous. q_c[qt * P + p, h0 + hh, r] goes
        # to qc_rows[p, qt, hh * R + r]. A group loads the next group's rows
        # (h_nx) during its cache steps (``stage_next``), and its end
        # transposes them.
        if not feature:
            qc_rows = nl.ndarray((P, NQT, G * R), dtype=dt, buffer=nl.sbuf, name=nm("qc_rows"))
            qr_rows = nl.ndarray((P, NQT, G * DR), dtype=dt, buffer=nl.sbuf, name=nm("qr_rows"))
        h_nx = scalar_tile(1, nl.int32, nm("h_nx"))
        if own_at_end:
            # The head offsets of the loads in a block-loop body: past the heads (skipped) after a group's first body.
            h_skip = scalar_tile(1, nl.int32, nm("h_skip"))
            h_ld = scalar_tile(1, nl.int32, nm("h_ld"))
            # The skip term of the last group's query loads (it would reload its own heads).
            h_lst = scalar_tile(1, nl.int32, _Namer(tid, "lst")("h_lst"))
        if qmx:
            qmx_part = G * NQT * P * MX_PACK  # q_mx bytes of one part in one partition
        if qmx and not feature:
            # The split query rows of two query tiles in turn: [slot, part (hi, lo), feature].
            hl8 = nl.ndarray((P, 2, 2, R), dtype=cache.dtype, buffer=nl.sbuf, name=nm("hl8"))
        NR = R // P
        qfm_part = G * NQT * NF * P  # q_fm values of one partition
        if feature:
            # The next group's queries load into q_stage and r_stage while
            # the group runs, and SBUF copies move them into q_fm at the
            # group end (a load into q_fm itself would wait for the group's
            # last read of it). q_stage[c, hh, f, t] is latent feature
            # f * P + c of query t of head h0 + hh. A partition row of a
            # tile is Sq contiguous values in HBM and in SBUF, so the load
            # moves whole rows (2 * Sq bytes) instead of the P-value pieces
            # of the q_fm order.
            q_stage = nl.ndarray((P, G, NR, Sq), dtype=dt, buffer=nl.sbuf, name=nm("q_stage"))
            stage_part = G * NR * Sq  # q_stage values of one partition
            # The rope rows: query half rh of rope feature c on partition
            # rh * P / 2 + c, so the two stages take the SBUF of the
            # token-major row buffers (more would make the allocator spill
            # them with the short-prior variants).
            RH = 2 if 2 * DR <= P and NQT % 2 == 0 else 1
            r_stage = nl.ndarray((P if RH > 1 else DR, G, Sq // RH), dtype=dt, buffer=nl.sbuf, name=nm("r_stage"))
            r_part = G * (Sq // RH)  # r_stage values of one partition
            # Hardware descriptor generation on the ScalarE queue: the
            # loads have a runtime head offset, so the default DGE would
            # build their descriptors on GpSimd.
            q_dma = dict(dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.scalar)

        def load_queries_fm(h_off, swdge=False, defer=False):
            """Load the feature-major queries of the heads ``h_off .. h_off + G`` into the stages.

            One DMA for the latent tiles of the group, and one for each half
            of the rope rows. With ``swdge`` they issue on the software DGE
            (GpSimd) instead of ScalarE's queue, and skip a head offset past
            the heads. With ``defer`` they are returned as closures that issue one DMA (one head of
            the latent tiles, or a half of the rope rows) after an instruction ``after``.
            """
            dma = dict(oob_mode=nisa.oob_mode.skip, priority=STAGE_PRIORITY) if swdge else q_dma
            if defer:
                fns = []
                head_pattern = [[MX_PACK * Sq, P], [1, MX_PACK * Sq]] if qmx else [[Sq, P], [P * Sq, NR], [1, Sq]]
                for hh in range(G):

                    def head(after, hh=hh):
                        nisa.dma_copy(
                            dst=q_stage[0:P, hh, 0:NR, 0:Sq],
                            src=q_c.ap(pattern=head_pattern, offset=hh * R * Sq, scalar_offset=h_off, indirect_dim=0),
                            **(dma if q_c.dtype == dt else {}),
                        ).depends_on(after, kind=nisa.dependency.runtime)

                    fns.append(head)
                for rh in range(RH):

                    def rope(after, rh=rh):
                        p0 = rh * (P // 2)
                        nisa.dma_copy(
                            dst=r_stage[p0 : p0 + DR, 0:G, 0 : Sq // RH],
                            src=q_r.ap(
                                pattern=[[Sq, DR], [DR * Sq, G], [1, Sq // RH]],
                                offset=rh * (Sq // RH),
                                scalar_offset=h_off,
                                indirect_dim=0,
                            ),
                            **(dma if q_r.dtype == dt else {}),
                        ).depends_on(after, kind=nisa.dependency.runtime)

                    fns.append(rope)
                return fns
            # q_stage[c, hh, f, t] = q_c[h_off + hh, f * P + c, t], or with the split query (MX packing)
            # q_c[h_off + hh, 4 c + f, t]: four contiguous rows for each partition.
            pattern = (
                [[MX_PACK * Sq, P], [R * Sq, G], [1, MX_PACK * Sq]]
                if qmx
                else [[Sq, P], [R * Sq, G], [P * Sq, NR], [1, Sq]]
            )
            nisa.dma_copy(
                dst=q_stage,
                src=q_c.ap(pattern=pattern, offset=0, scalar_offset=h_off, indirect_dim=0),
                **(dma if q_c.dtype == dt else {}),
            )
            for rh in range(RH):
                p0 = rh * (P // 2)
                nisa.dma_copy(
                    dst=r_stage[p0 : p0 + DR, 0:G, 0 : Sq // RH],
                    src=q_r.ap(
                        pattern=[[Sq, DR], [DR * Sq, G], [1, Sq // RH]],
                        offset=rh * (Sq // RH),
                        scalar_offset=h_off,
                        indirect_dim=0,
                    ),
                    **(dma if q_r.dtype == dt else {}),
                )

        def stage_queries(after=None, cols=None, rope=True, per_head=False):
            """Copy the stages into ``q_fm``: q_fm[c, (hh * NQT + qt) * NF + f, p] = q_stage[c, hh, f, qt * P + p].

            The latent tiles of the two halves of the heads copy on VectorE
            and ScalarE. The rope tiles copy on VectorE, partitions 0 .. DR
            only, so the marker partition of q_fm keeps its ones. With
            ``after``, the copies are scheduled after that instruction (the
            end of the group's steps): the scheduler would otherwise move
            them between the last steps, where they delay the in-order
            ScalarE and VectorE queues. With the split query, ``cols``
            ``(c0, c1)`` converts those query columns of one head only, and
            ``rope`` gives the rope copies too. ``per_head`` splits the split
            query's conversion by head (for a group end with no key
            transposes).
            """
            copies = []
            half = G // 2 if G > 1 else G
            # The split query of each half of the heads (whole query rows), or of the columns c0 .. c1 of one head.
            if cols is None:
                # (VectorE's lo of a head runs while ScalarE rounds the next head's hi with ``per_head``.)
                heads = [(hh, hh + 1) for hh in range(G)] if per_head else [(0, half), (half, G)]
                chunks = [(h0 * NQT, h1 * NQT, h1 - h0, Sq) for h0, h1 in heads]
            else:
                chunks = [(cols[0], cols[1], 1, (cols[1] - cols[0]) * P)]
            for c0, c1, nh, nt in chunks if qmx else ():
                if c1 > c0:
                    # The split query of the columns c0 .. c1: hi = e4m3(q) on ScalarE, lo = e4m3(q - hi)
                    # on VectorE, into the MX packing q_mx[c, part, (hh * Sq + t) * 4 + f] from
                    # q_stage[c, hh, f, t]. Column c0 is query tile c0 % NQT of head c0 // NQT.
                    hh, qt = divmod(c0, NQT)
                    src = q_stage.ap(
                        [[stage_part, P], [NR * Sq, nh], [1, nt], [Sq, MX_PACK]], offset=hh * NR * Sq + qt * P
                    )
                    if qhi:
                        # fp8_query="mx", one head at a time: ScalarE interleaves the head's columns (exact
                        # bf16 copy) into q_il[c, slot, t * 4 + f], then VectorE quantizes each 32 latent
                        # features (8 partitions x 4) of a query to e4m3 with one E8M0 scale, the block max
                        # rounded down to a power of two over 2^7, so no value saturates. quantize_mx reads
                        # its 4-packs contiguous; the words and scales of a head's columns are contiguous.
                        for hp in range(nh):
                            p0 = c0 + hp * NQT if cols is None else c0
                            p1 = p0 + NQT if cols is None else c1
                            n_w = (p1 - p0) * P
                            slot = n_il[0] % 2
                            n_il[0] += 1
                            hq, qq = divmod(p0, NQT)
                            copies.append(
                                nisa.tensor_copy(
                                    dst=q_il.ap(
                                        [[2 * Sq * MX_PACK, P], [MX_PACK, n_w], [1, MX_PACK]],
                                        offset=slot * Sq * MX_PACK,
                                    ),
                                    src=q_stage.ap(
                                        [[stage_part, P], [1, n_w], [Sq, MX_PACK]], offset=hq * NR * Sq + qq * P
                                    ),
                                    engine=nisa.scalar_engine,
                                )
                            )
                            copies.append(
                                nisa.quantize_mx(
                                    dst=q_mx.ap([[G * NQT * P, P], [1, n_w]], offset=p0 * P, dtype=nl.float8_e4m3fn_x4),
                                    src=q_il[0:P, slot, 0 : n_w * MX_PACK],
                                    dst_scale=q_sc[0:P, p0 * P : p1 * P],
                                )
                            )
                        continue
                    part_dst = [
                        q_mx.ap(
                            [[2 * qmx_part, P], [Sq * MX_PACK, nh], [MX_PACK, nt], [1, MX_PACK]],
                            offset=part * qmx_part + c0 * P * MX_PACK,
                        )
                        for part in range(2)
                    ]
                    copies.append(nisa.tensor_copy(dst=part_dst[0], src=src, engine=nisa.scalar_engine))
                    copies.append(
                        nisa.tensor_tensor(
                            dst=part_dst[1], data1=src, data2=part_dst[0], op=nl.subtract, engine=nisa.vector_engine
                        )
                    )
            for h0, h1, eng in () if qmx else ((0, half, nisa.vector_engine), (half, G, nisa.scalar_engine)):
                if h1 > h0:
                    cp = nisa.tensor_copy(
                        dst=q_fm.ap(
                            [[qfm_part, P], [NQT * NF * P, h1 - h0], [NF * P, NQT], [P, NR], [1, P]],
                            offset=h0 * NQT * NF * P,
                        ),
                        src=q_stage.ap(
                            [[stage_part, P], [NR * Sq, h1 - h0], [P, NQT], [Sq, NR], [1, P]], offset=h0 * NR * Sq
                        ),
                        engine=eng,
                    )
                    copies.append(cp)
            nq = NQT // RH
            for rh in range(RH) if rope else ():
                p0 = rh * (P // 2)
                if qmx:
                    # q_fm holds the rope tiles only: q_fm[c, hh * NQT + qt, p].
                    rope_dst = q_fm.ap([[G * NQT * P, DR], [NQT * P, G], [P, nq], [1, P]], offset=rh * nq * P)
                else:
                    rope_dst = q_fm.ap(
                        [[qfm_part, DR], [NQT * NF * P, G], [NF * P, nq], [1, P]], offset=(rh * nq * NF + NR) * P
                    )
                cp = nisa.tensor_copy(
                    dst=rope_dst,
                    src=r_stage[p0 : p0 + DR].ap([[r_part, DR], [Sq // RH, G], [P, nq], [1, P]], offset=0),
                    engine=nisa.vector_engine,
                )
                copies.append(cp)
            if after is not None:
                for cp in copies:
                    cp.depends_on(after, kind=nisa.dependency.schedule)

        def load_queries(h_off, swdge=False):
            """Load the token-major query rows of the heads ``h_off .. h_off + G``; ``swdge`` as in ``load_queries_fm``."""
            dma = dict(oob_mode=nisa.oob_mode.skip) if swdge else {}
            nisa.dma_copy(
                dst=qc_rows,
                src=q_c.ap(
                    pattern=[[H * R, P], [P * H * R, NQT], [1, G * R]], offset=0, scalar_offset=h_off, indirect_dim=1
                ),
                **dma,
            )
            nisa.dma_copy(
                dst=qr_rows,
                src=q_r.ap(
                    pattern=[[H * DR, P], [P * H * DR, NQT], [1, G * DR]], offset=0, scalar_offset=h_off, indirect_dim=1
                ),
                **dma,
            )

        def transpose_queries_mx(nmx):
            """Split the query rows into the MX parts ``q_mx`` and transpose the rope rows into ``q_fm``.

            For each query tile of a head: hi = e4m3(q) (ScalarE) and lo = e4m3(q - hi) (VectorE) in
            token-major rows, then 32-bit TensorE transposes (bit exact) give partition c the features
            4c .. 4c + 3 of each query. The query tiles of a part share one fp32 PSUM bank and one copy.
            """
            words = 2 * 2 * R // MX_PACK  # words of one partition of hl8
            for hh in range(G):
                ps = [nl.ndarray((P, NQT * P), dtype=_F32, buffer=nl.psum, name=nmx("tp_qmx")) for _ in range(2)]
                for qt in range(NQT):
                    sl = (hh * NQT + qt) % 2
                    src = qc_rows[0:P, qt, hh * R : (hh + 1) * R]
                    nisa.tensor_copy(dst=hl8[0:P, sl, 0, 0:R], src=src, engine=nisa.scalar_engine)
                    nisa.tensor_tensor(
                        dst=hl8[0:P, sl, 1, 0:R],
                        data1=src,
                        data2=hl8[0:P, sl, 0, 0:R],
                        op=nl.subtract,
                        engine=nisa.vector_engine,
                    )
                    for part in range(2):
                        _pe_transpose(
                            ps[part][0:P, qt * P : (qt + 1) * P],
                            hl8.ap(
                                pattern=[[words, P], [1, R // MX_PACK]],
                                offset=(sl * 2 + part) * R // MX_PACK,
                                dtype=_F32,
                            ),
                            id32,
                        )
                for part in range(2):
                    nisa.tensor_copy(
                        dst=q_mx[0:P, part, hh * NQT * P : (hh + 1) * NQT * P],
                        src=ps[part][0:P, 0 : NQT * P],
                        engine=(nisa.vector_engine, nisa.scalar_engine)[part],
                    )
                for qt in range(NQT):
                    ps_r = nl.ndarray((P, P), dtype=_F32, buffer=nl.psum, name=nmx("tp_q"))
                    nisa.nc_matmul(
                        dst=ps_r[0:DR, 0:P],
                        stationary=qr_rows[0:P, qt, hh * DR : (hh + 1) * DR],
                        moving=id_sb[0:P, 0:P],
                        accumulate=False,
                    )
                    nisa.tensor_copy(dst=q_fm[0:DR, hh * NQT + qt, 0:P], src=ps_r[0:DR, 0:P])

        def transpose_queries(nmx):
            """Transpose the query rows into the feature-major queries ``q_fm``."""
            if qmx:
                transpose_queries_mx(nmx)
                return
            # q_fm[c, (hh * NQT + qt) * NF + f, p] = q[qt * P + p, h0 + hh, f * P + c].
            for hh in range(G):
                for qt in range(NQT):
                    base = (hh * NQT + qt) * NF
                    for f in range(NF):
                        w = widths[f]
                        if f * P < R:
                            q_src = qc_rows[0:P, qt, hh * R + f * P : hh * R + f * P + w]
                        else:
                            q_src = qr_rows[0:P, qt, hh * DR : hh * DR + w]
                        ps = nl.ndarray((P, P), dtype=_F32, buffer=nl.psum, name=nmx("tp_q"))
                        nisa.nc_matmul(dst=ps[0:w, 0:P], stationary=q_src, moving=id_sb[0:P, 0:P], accumulate=False)
                        nisa.tensor_copy(dst=q_fm[0:w, base + f, 0:P], src=ps[0:w, 0:P])

        def next_group(nmx, after=None, convert=True, own_steps=True):
            """Make the staged queries the group's queries, and run the group's own-key steps.

            With ``own_at_end`` a group end prepares the next group this way
            while its own stores run. Otherwise the prologue only converts
            (``own_steps``) and a group start runs only the own-key steps
            (``convert``). The own keys run first in a group and set the state
            of every column; the step chain (softmax, transposes, value
            matmuls) would set the pace of these short steps, so the first
            stage of a step runs OWN_AHEAD steps ahead of the second. At a
            group end the next group's heads are not the ones being stored,
            so a column's own-key step waits only for the store of its own
            heads. The split feature-major queries then convert by column,
            ``CONV_AHEAD`` columns before their own-key steps. ``after`` as
            in ``stage_queries``.
            """
            by_column = feature and qmx and own_steps
            if not convert:
                by_column = False
            elif feature:
                # The rope tiles, and the latent ones unless they convert by column.
                stage_queries(after=after, cols=(0, 0) if by_column else None)
            else:
                transpose_queries(nmx)
            if not own_steps:
                return
            n_own = G * NQT
            # own_drain_transpose: the last OWN_PE_DRAIN steps (the pipeline's drain) take p^T from TensorE.
            pe_from = n_own - OWN_PE_DRAIN if own_pe_drain else n_own
            own = [own_init_stages(nmx, i, i % NBUF, pe_pt=i >= pe_from) for i in range(n_own)]
            n_conv = [0]

            def convert_upto(n_cols):
                while by_column and n_conv[0] < min(n_cols, n_own):
                    stage_queries(after=after, cols=(n_conv[0], n_conv[0] + 1), rope=False)
                    n_conv[0] += 1

            for i in range(-OWN_AHEAD, n_own):
                c = i + OWN_AHEAD
                if c < n_own:
                    convert_upto(c + CONV_AHEAD)
                    own[c][0]()
                if i >= 0:
                    own[i][1]()

        # The first group's queries (with own_at_end, its own-key steps too) and first
        # cache block. A group prepares the next group's at its end, while its stores run.
        nmp0 = _Namer(tid, "pro")
        load_group_queries = load_queries_fm if feature else load_queries
        load_group_queries(h_sb)
        if own_at_end:
            if wv:
                # A define of W_V outside the loops (the first group's): the SBUF allocator cannot
                # spill a tensor that only skipping in-loop DMAs define (NCC_IGCA103).
                load_wv_group(h_sb)
            load_block(0, first=True)
            next_group(nmp0)
        else:
            next_group(nmp0, own_steps=False)
            load_block(0, first=True)
        transpose_rows(nmp0, kfm_ab[0], rows_ab[0], SUBS)

        def stage_next(in_loop=True, defer=False):
            """Stage the next group's queries (heads ``h_nx``) during the group's cache steps.

            With ``own_at_end`` a group with block loops loads them in its first
            block-loop body (so no loop barrier waits for them), and a one-block group
            at its start; otherwise every group loads them at its start, before its
            own-key steps. They load whole, inside the group that converts them: a
            stage that lives across the group loop's back edge, or whose rows partly
            load at the group end, makes the SBUF allocator's coloring fallback place
            the cache rows over its reserved top bytes (NCC_INLA001) in programs of
            several calls or under caller pressure. With ``defer`` (feature-major queries, in a loop body)
            the head offsets are set here and the DMAs are returned as closures (``load_queries_fm``).
            """
            h_q, h_w = h_nx, h_sb
            if in_loop:
                # Only a group's first block-loop body loads (the block counter k_sb is then BLOCK); the
                # later ones give an offset past the heads, which the DMAs skip. They issue on GpSimd:
                # ScalarE runs the steps in order, and DMA triggers there would stall them. The head
                # offset of a token-major query load ([Sq, H, *], the head dimension second) skips by
                # Sq * H heads: every element is then past the end of the tensor, not only past the
                # heads (the simulator skips an access by its flat address only). The W_V and
                # feature-major loads index the head dimension first, so H is past their end.
                gps = nisa.gpsimd_engine
                nisa.tensor_scalar(
                    dst=h_skip,
                    data=k_sb,
                    op0=nl.greater_equal,
                    operand0=2 * BLOCK,
                    op1=nl.multiply,
                    operand1=H if feature else Sq * H,
                    engine=gps,
                )
                nisa.tensor_tensor(dst=h_ld, data1=h_skip, data2=h_nx, op=nl.add, engine=gps)
                # The last group (h_sb >= H - G) has no next group: its query loads skip too. Its group end then
                # converts the q_stage that the group before it loaded for it (the same heads).
                nisa.tensor_scalar(
                    dst=h_lst,
                    data=h_sb,
                    op0=nl.greater_equal,
                    operand0=H - G,
                    op1=nl.multiply,
                    operand1=H if feature else Sq * H,
                    engine=gps,
                )
                nisa.tensor_tensor(dst=h_ld, data1=h_ld, data2=h_lst, op=nl.add, engine=gps)
                h_q = h_ld
                if wv:
                    if not feature:
                        nisa.tensor_scalar(
                            dst=h_skip,
                            data=k_sb,
                            op0=nl.greater_equal,
                            operand0=2 * BLOCK,
                            op1=nl.multiply,
                            operand1=H,
                            engine=gps,
                        )
                    nisa.tensor_tensor(dst=h_skip, data1=h_skip, data2=h_sb, op=nl.add, engine=gps)
                    h_w = h_skip
            if defer and feature and in_loop:
                return load_queries_fm(h_q, swdge=True, defer=True) + (
                    load_wv_group(h_w, swdge=True, defer=True) if wv else []
                )
            if feature:
                load_queries_fm(h_q, swdge=in_loop)
            else:
                load_queries(h_q, swdge=in_loop)
            if wv:
                # The group's W_V, which only its end reads.
                load_wv_group(h_w, swdge=in_loop)
            return []

        def group_body(_, var, single=False, static=None):
            """Attend all keys for the heads ``h_sb .. h_sb + G``.

            On entry, ``q_fm`` holds the group's queries (with ``own_at_end``
            every column also holds the state after the own keys), and buffer
            set 0 holds the first cache block, transposed. With ``single``, the
            view has one block: its first ``var`` steps run with no block loop,
            and the block stays in set 0 for the next group. With ``static``
            (``short_ns``), (pair bodies, full tails, half tails) run inline, in
            the block loops' order.
            """
            nmg = _Namer(tid, f"grp{var}{'s' if single else ''}")
            # The next group's queries (the last group loads its own again).
            nisa.tensor_scalar(dst=h_nx, data=h_sb, op0=nl.add, operand0=G, op1=nl.minimum, operand1=H - G)
            if not own_at_end:
                # The next group's queries load while the own-key steps run.
                stage_next(in_loop=False)
                next_group(nmg, convert=False)
            if single:
                if own_at_end:
                    # No loop barrier follows, so the load overlaps the steps.
                    stage_next(in_loop=False)
                run_block(nmg, 0, n_kstep=var)
                if len(block_steps(var)) % 2:
                    nisa.tensor_copy(dst=nm_run[0:P, 0, :], src=nm_run[0:P, 1, :])
            else:
                # The cache prior: a runtime number of blocks, rolled. Block 0
                # is in set 0, so the loads start at block 1.
                nisa.memset(dst=k_sb, value=BLOCK)
                nisa.memset(dst=pg_sb, value=SUBS)
                if fp8_rows:
                    nisa.memset(dst=kb_b, value=BLOCK)
                if static is not None:
                    # A short prior: the block-loop bodies inline, with no loop boundary (a drain and two
                    # all-engine barriers each) between them, and the group loop's back edge keeps its
                    # prefetch hint (an inner loop's hint replaces it).
                    n_pair, n_full, n_half = static
                    for _ in range(n_pair):
                        pair_body(0, var)
                    if n_full:
                        tail_body(0, var, half_tail)
                    if n_half:
                        tail_body(0, KT_MM8 // KT)
                else:
                    nex.fori_loop(0, npair_reg, lambda i: pair_body(i, var), unroll=False)
                    nex.fori_loop(0, ntail_reg, lambda i: tail_body(i, var, half_tail), unroll=False)
                    if half_tail:
                        # A last block whose second step holds no prior slot runs its first step only.
                        nex.fori_loop(0, nhalf_reg, lambda i: tail_body(i, KT_MM8 // KT), unroll=False)
                if not own_at_end:
                    # The next group's first block, issued before the stores so
                    # that its pages do not queue behind them.
                    load_block(0, first=True, reload=True)

            # The last step leaves the max in slot 0.
            if prescale and not mxv:
                # acc holds 2^8 times the sums: the state (m - 8 ln 2, 2^8 l, acc) is the same softmax.
                fin = nisa.tensor_scalar(
                    dst=m_fin, data=nm_run[0:P, 0, :], op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=-PS_LN
                )
                nisa.tensor_scalar(dst=l_run, data=l_run, op0=nl.multiply, operand0=PS_SCALE, engine=nisa.vector_engine)
            else:
                fin = nisa.tensor_scalar(dst=m_fin, data=nm_run[0:P, 0, :], op0=nl.multiply, operand0=-1.0)
            if wv:
                # out[h0 + hh, d, :] = acc[d, hh * NQT, 0:Sq] after the projection.
                project_group(nmg)
            # out[h0 + hh, qt * P + p, :] = column hh * NQT + qt (or the projection), with h0
            # from h_sb. The columns of a group are contiguous in the outputs. With own_at_end
            # one DMA stores each GRP_HEADS heads, so the next group's own-key steps of a head
            # wait only for the store of those heads' columns; else one DMA stores the group.
            n_st = GRP_HEADS if own_at_end else G
            for a0 in range(0, G, n_st):
                a1 = min(G, a0 + n_st)
                if wv:
                    nisa.dma_copy(
                        dst=acc_out.ap(
                            pattern=[[Sq, VD], [VD * Sq, a1 - a0], [1, Sq]],
                            offset=a0 * VD * Sq,
                            scalar_offset=h_sb,
                            indirect_dim=0,
                        ),
                        src=acc.ap(pattern=[[G * NQT * R, VD], [NQT * R, a1 - a0], [1, Sq]], offset=a0 * NQT * R),
                    )
                    continue
                nisa.dma_copy(
                    dst=acc_out.ap(
                        pattern=[[R, P], [P * R, (a1 - a0) * NQT], [1, R]],
                        offset=a0 * Sq * R,
                        scalar_offset=h_sb,
                        indirect_dim=0,
                    ),
                    src=acc[0:P, a0 * NQT : a1 * NQT, 0:R],
                )
            # m and l: TensorE transposes (bit exact) give each column a
            # partition, so a stored row is P contiguous values, not P
            # 4-byte pieces.
            ncol = G * NQT
            for a in range(0, ncol, P):
                n = min(P, ncol - a)
                ps_st = nl.ndarray((P, 2, P), dtype=_F32, buffer=nl.psum, name=nmg("ps_st"))
                _pe_transpose(ps_st[0:n, 0, 0:P], m_fin[0:P, a : a + n], id32)
                _pe_transpose(ps_st[0:n, 1, 0:P], l_run[0:P, a : a + n], id32)
                nisa.tensor_copy(dst=st_t[0:n, 0:2, 0:P], src=ps_st[0:n, 0:2, 0:P], engine=nisa.vector_engine)
                nisa.dma_copy(
                    dst=m_out.ap(pattern=[[P, n], [1, P]], offset=a * P, scalar_offset=h_sb, indirect_dim=0),
                    src=st_t[0:n, 0, 0:P],
                )
                nisa.dma_copy(
                    dst=l_out.ap(pattern=[[P, n], [1, P]], offset=a * P, scalar_offset=h_sb, indirect_dim=0),
                    src=st_t[0:n, 1, 0:P],
                )
            if own_at_end:
                if not single and not wrap:
                    # The next group's first block (set 0 held the last block). Its pages
                    # queue behind the stores, and its key transposes follow the own-key
                    # steps: the next group's cache steps are the first to read it.
                    load_block(0, first=True, reload=True)
                # The next group's queries (staged during this group's cache steps) and
                # own-key steps, while the stores run.
                next_group(nmg, after=fin)
                if not single and not wrap:
                    transpose_rows(nmg, kfm_ab[0], rows_ab[0], SUBS)
            else:
                # The next group's keys and queries, while the stores run.
                if not single:
                    transpose_rows(nmg, kfm_ab[0], rows_ab[0], SUBS)
                if feature:
                    # The next group's queries, staged at the group start. A one-block view transposes no
                    # keys here, so the split query's conversion runs by head (with the key transposes,
                    # by head is the slower order).
                    stage_queries(after=fin, per_head=single)
                else:
                    transpose_queries(nmg)
            nisa.tensor_scalar(dst=h_sb, data=h_sb, op0=nl.add, operand0=G)

        def one_loops(_):
            for var in range(1, NSTEP + 1):
                nex.fori_loop(0, single_regs[var], lambda i, var=var: group_body(i, var, True), unroll=False)

        def part_loops(_):
            nex.fori_loop(0, one_reg, one_loops, unroll=False)
            for var in TAIL_STEPS:
                nex.fori_loop(0, tail_regs[var], lambda i, var=var: group_body(i, var), unroll=False)

        if gen:
            nex.fori_loop(0, gen_reg if short_ns else full_reg, lambda i: group_body(i, NSTEP), unroll=False)

        def short_loops():
            """Emit the short priors' group-loop variants: H / G trips of the call's step count, 0 of the others.

            One loop each, next to the rolled one: in a 0/1-trip wrapper loop they would exhaust the sequencer
            registers of a program of 24 calls (the compiler cannot spill them).
            """
            nonlocal tid
            tid0 = tid
            for k in short_ns:
                # Each copy of the block bodies traces its own names.
                tid = f"{tid0}s{k}"
                shape = (k >> 2, (k >> 1) & 1, k & 1)
                nex.fori_loop(0, vn_regs[k], lambda i, s=shape: group_body(i, NSTEP, static=s), unroll=False)
            tid = tid0

        short_loops()
        if prior_variants:
            nex.fori_loop(0, part_reg, part_loops, unroll=False)

    def own_path(_):
        _own_only(
            _Namer(tid, "own"),
            q_c,
            q_r,
            kc_rows,
            kc_fm,
            b_sb,
            acc_out,
            m_out,
            l_out,
            scale,
            G,
            widths,
            id_sb,
            id32,
            q_layout,
            w_v,
        )

    nex.fori_loop(0, own_reg, own_path, unroll=False)
    nex.fori_loop(0, cache_reg, cache_path, unroll=False)
    return acc_out, m_out, l_out
