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

"""KDA decode attention body of the Kimi-K3 decode giga-kernel (TP32, LNC1, Trn3).

The body runs one KDA layer for B sequences (one token each) on one TP rank, and reads and writes the
paged KDA state of each sequence.
"""

import functools

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from .utils.kernel_assert import kernel_assert
from .utils.kernel_helpers import div_ceil
from .common import check_pair_group, dma_priority, is_trn3, pair_shared_buffer
from .tkg_weight_dma_gate import gate_weight_dma, release_weight_gate, wait_for_completion
from .tkg_mx_quant import MX_LANES, load_mx_scales, mx_hT_shapes, mx_scale_view, mx_weight_format, quantize_hT_mx

P_MAX = 128
"""Partitions of an SBUF tile."""

HEAD_DIM = 128
"""KDA head dimension (key and value)."""

_QUADRANT = 32
"""Partition quadrant of the engines. An operand of fewer than 32 partitions starts at a multiple of this."""

CONV_TAPS = 4
"""Short convolution kernel size. The conv state holds the last CONV_TAPS - 1 inputs."""

N_CONV_PROJ = 3
"""Projections with a short convolution: q, k and v."""

CONV_DTYPES = ("bf16", "fp32")
"""Forms of the short convolution. ``bf16``: BF16 weights in ``kda_w_small``, BF16 products and a BF16 output.
``fp32``: the FP32 checkpoint weights in ``kda_params``, FP32 products, sum and SiLU output (the GPU decode form)."""

GATE_LOWER_BOUND = -5.0
"""Lower bound of the per-channel log decay."""

L2_EPS = 1.0e-6
"""Epsilon of the q and k L2 norms."""

NORM_EPS = 1.0e-5
"""Epsilon of the gated output RMSNorm."""

PAGE_WIDTH = 73728
"""Elements in one page of the serving cache pool."""

MAX_LOCAL_HEADS = 4
"""Most KDA heads of one rank. The value rows of all heads fill one fp32 PSUM bank."""

MAX_SEQUENCES = 32
"""Most sequences of one call. The input projection keeps ``[128, 14, B]`` fp32 in one PSUM bank, and
the difference rows of a group of sequences fit on the 128 partitions."""

KDA_WEIGHT_NAMES = ("kda_w_in", "kda_w_out", "kda_w_small", "kda_params")
"""Ordered names of the per-layer KDA kernel weight tensors of one rank.

With ``Hl`` local heads and ``W = Hl * 128``:

* ``kda_w_in``: bf16 ``[H, N]`` as ``[in, out]``. Columns: q ``[0, W)``, k ``[W, 2W)``,
  v ``[2W, 3W)``, g ``[3W, 4W)``, f_a ``[4W, 4W+128)``, b ``[4W+128, 4W+128+Hl)``, then zeros up to
  ``N``. See ``kda_in_columns``.
* ``kda_w_out``: bf16 ``[W, H]``, ``o_proj`` as ``[in, out]``.
* ``kda_w_small``: bf16 ``[128, W + 12*Hl]``. Columns ``[0, W)`` hold ``f_b`` as ``[in, out]``.
  Column ``W + 4*(Hl*p + h) + t`` of partition ``d`` holds tap ``t`` of the conv weight of
  projection ``p`` (q, k, v) for channel ``h*128 + d``, cast to bf16.
* ``kda_params``: fp32 ``[128, 2*Hl + 1]``. Column ``h`` holds ``dt_bias[h*128 + d]``, column
  ``Hl + h`` holds ``A_log[h]``, and column ``2*Hl`` holds ``o_norm[d]``.

``kda_decode_torch.build_kda_rank_weights`` builds these tensors from the serving weights.
"""

KDA_WEIGHT_DMAS = ("pool", "act")
"""The queues of the KDA verify's output-projection, small-weight and parameter loads (``weight_dma``): ``pool`` (the
default) the software DGE on GpSimd, ``act`` the Activation hardware DGE queue. Either moves the same bytes."""


def kda_weight_dma_options(weight_dma: str = "pool") -> dict:
    """Return the ``nisa.dma_copy`` queue options of a KDA weight load for ``weight_dma`` (``KDA_WEIGHT_DMAS``)."""
    kernel_assert(weight_dma in KDA_WEIGHT_DMAS, f"weight_dma must be one of {KDA_WEIGHT_DMAS}, got {weight_dma!r}")
    if weight_dma == "act":
        return {"dge_mode": nisa.dge_mode.hwdge, "engine": nisa.engine.scalar}
    return {"dge_mode": nisa.dge_mode.swdge}


KDA_READOUTS = ("reference", "decoupled", "decoupled_bf16")
"""Forms of the readout and the state matmuls of the KDA body (the ``readout`` argument).

* ``reference``: ``out = q @ S'`` with the fp32 new state, the reference order. All the recurrence
  matmuls are fp32.
* ``decoupled``: ``out = (q * decay) @ S + (q . k) * diff`` from the old state, so the output does not
  wait for the state update. The matmuls are fp32.
* ``decoupled_bf16``: the decoupled form with the loaded bf16 state as the stationary operand (the
  state is exact in bf16). The fp32 moving vectors (``k * decay``, ``q * decay`` and the difference
  rows) are split into bf16 high and low parts, ``x = hi + lo`` with ``hi = bf16(x)`` and
  ``lo = bf16(x - hi)``, which keeps about 16 mantissa bits. The TensorE accumulates the bf16
  products in fp32.
"""

KDA_STATE_DTYPES = ("bf16", "fp32", "fp32_hilo")
"""Forms of the recurrent state in the cache page (the ``state_dtype`` argument).

* ``bf16``: a bf16 state ``[Hl, 128 k, 128 v]`` after the conv state. The update rounds the new state to bf16.
* ``fp32``: an fp32 state at the same byte offset (2 bf16 elements of the pool per value), as the GPU serving form
  keeps it. The update adds in fp32 and stores the fp32 sums. With ``decoupled_bf16`` the prediction matmul takes
  ``bf16(S)`` as its one stationary operand, a VectorE copy of each selected state made beside the input projection.
* ``fp32_hilo``: the fp32 state, and with ``decoupled_bf16`` two stationary operands per head, ``hi = bf16(S)`` and
  ``lo = bf16(S - hi)``, accumulated in PSUM (about 16 mantissa bits of S).

The ``reference`` and ``decoupled`` readouts use the fp32 state as their fp32 stationary operand in both fp32 forms.
"""


KDA_BF16_PARTS = ("gate", "decay", "sigmoid", "norm", "moving", "onorm", "update")
"""Parts of the KDA body that the option ``bf16_parts`` runs in BF16 in place of FP32 (an accuracy study; the values
change): ``gate`` the decay gate math up to its sigmoid, ``decay`` the decay itself, ``sigmoid`` the beta and output gate
sigmoids, ``norm`` the q / k L2-norm outputs, ``moving`` the prediction vectors of ``decoupled_bf16`` as one BF16 part
(no low part), ``onorm`` the gated RMSNorm, ``update`` the decayed state of the update."""


def kda_default_readout(n_seq: int) -> str:
    """Return the readout form of ``n_seq`` sequences, for a ``readout`` of None.

    One sequence uses ``reference``, the reference operation order, which is faster for two
    layers on Trn3. More sequences use ``decoupled_bf16``, which is faster at B=4 and
    at B=32, with the same errors on Trn3.

    Args:
        n_seq (int): Sequences of the call.

    Returns:
        str: A member of ``KDA_READOUTS``.

    Example:
        >>> kda_default_readout(1)
        'reference'
    """
    return "reference" if n_seq == 1 else "decoupled_bf16"


_W_IN_COLUMN_ALIGN = 8
"""Column alignment of ``kda_w_in``. It keeps each weight row 16-byte aligned for the DMA."""

_W_IN_TILES_PER_DMA = 4
"""Hidden K-tiles (of 128 rows) of ``kda_w_in`` in one DMA."""

_W_IN_BUFFERS = 4
"""SBUF buffers in the ``kda_w_in`` ring. A buffer holds one DMA."""

_W_IN_TAIL_TILES = (2, 1, 1)
"""K-tiles of the last ``kda_w_in`` DMAs. Small last chunks shorten the TensorE work after the last load."""

_PSUM_BANK_FP32 = 512
"""fp32 elements of one PSUM bank in one partition. Keep each matmul output in one bank."""

_MX_MOVING_MAX = 512
"""Most moving columns of one nc_matmul_mx: the column chunk of the token-stationary MXFP8 in-projection (``x_stationary``)."""


class XStationaryPsum:
    """The result of the token-stationary MXFP8 in-projection (``_gemv_channel_major_mx`` ``x_stationary``): the
    token-major fp32 PSUM sums of ``count`` rows, one tile ``[count, width]`` a column chunk of at most ``_MX_MOVING_MAX``
    columns (``chunks``: ``(first column, width)``), for ``in_proj_copies_x_stationary``."""

    def __init__(self, chunks: list, psums: list, count: int):
        self.chunks = chunks
        self.psums = psums
        self.count = count


_RECURRENCE_GROUP = 8
"""Sequences of one transpose group in the recurrence. Two groups of fp32 states are in SBUF at a time."""

_W_IN_DMA_PRIORITY = 0
"""DMA priority of the ``kda_w_in`` stream (0 is the highest). The input projection is on the critical path."""

_W_OUT_DMA_PRIORITY = 3
"""DMA priority of the ``kda_w_out`` load.

The load uses the GpSimd queue, so it does not wait in the in-order SP and ACT queues of the
``kda_w_in`` stream, and its low priority leaves the DMA bandwidth to that stream.
"""

_W_IN_HEAD_DMA_PRIORITY = 3
"""DMA priority of the ``kda_w_in`` head loads (``kda_w_in_head_loads``).

The head loads run in the output glue of the feed-forward layer before, well before the
input projection reads them, so their low priority leaves the DMA bandwidth to the glue's own loads.
"""

_ALL_REDUCE_IN_SBUF = False
"""Run the output all-reduce on SBUF tiles instead of private HBM tiles.

An SBUF all-reduce of the ``[128, 56]`` bf16 partial over TP32 gives wrong results on Trn3 with
this toolchain: partitions ``4k+1 .. 4k+3`` receive the sum of partition ``4k``. The HBM
all-reduce of the same partial is exact. Keep this off until the SBUF collective is fixed.
"""


def kda_decode_attention(
    x_hT: nl.NkiTensor,
    kda_w_in: nl.NkiTensor,
    kda_w_out: nl.NkiTensor,
    kda_w_small: nl.NkiTensor,
    kda_params: nl.NkiTensor,
    kv_pool: nl.NkiTensor,
    page_id: nl.NkiTensor,
    position: nl.NkiTensor,
    tp_group: ReplicaGroup,
    tag: str,
    readout: str = None,
    sources: dict = None,
    conv_dtype: str = "bf16",
    w_in_buffers: int = 0,
    state_loads_first: bool = False,
    state_loads_early: bool = False,
    batched_conv_stores: bool = False,
    conv_load_gather: bool = False,
    front_hook=None,
    scatter_keep: list = None,
    conv_after: tuple = (),
    conv_stores_early: bool = False,
    state_dtype: str = "bf16",
    bf16_update_rows: bool = False,
    bf16_rows_flat: bool = False,
    kda_w_in_scale: nl.NkiTensor = None,
    bf16_parts: tuple = (),
    conv_copy_wait: tuple = (),
    w_in_ring=None,
    w_in_head: tuple = (),
    front_ahead: dict = None,
    reduce_hook=None,
    w_out_after_w_in: bool = False,
    w_in_marks: dict = None,
    null_page_rows: int = 0,
    kda_w_out_scale: nl.NkiTensor = None,
    null_page: int = 0,
    paired_pages: bool = False,
) -> nl.NkiTensor:
    """Run the KDA decode attention of one layer for B sequences (one token each) on one TP rank.

    This is a body function, not a ``@nki.jit`` kernel. A kernel calls it positionally once per
    KDA layer. The body is for decode with 1 to 32 sequences and 1 to 4 local heads (3 at TP32 with
    96 heads). It names every DMA and the collective with ``tag``, so the names stay unique when a
    kernel inlines it once per layer. The attribute ``token_major`` is True: the body takes and
    returns token-major hT tiles.

    Dimensions:
        H: Hidden size (7168), a multiple of 128.
        Hl: Local KDA heads (3).
        W: Local KDA width, Hl * 128 (384).
        N: Columns of the fused input projection (1672 for Hl = 3).
        P: Pages in the cache pool.
        B: Sequences, one token each (1 to 32).

    Args:
        x_hT (nl.NkiTensor): [128, B, H/128] bf16 SBUF, normalized hidden in token-major hT layout.
            Element ``(p, t, j)`` is ``x[t, j*128 + p]``.
        kda_w_in (nl.NkiTensor): [H, N] bf16 HBM, fused input projection, see ``KDA_WEIGHT_NAMES``.
        kda_w_out (nl.NkiTensor): [W, H] bf16 HBM, output projection.
        kda_w_small (nl.NkiTensor): [128, W + 12*Hl] bf16 HBM, ``f_b`` and the conv weights.
        kda_params (nl.NkiTensor): [128, 2*Hl + 1] fp32 HBM, ``dt_bias``, ``A_log`` and ``o_norm`` (and the 12*Hl
            fp32 conv weights with ``conv_dtype`` fp32).
        kv_pool (nl.NkiTensor): [P, 73728] bf16 HBM, paged cache pool with one page per row. The
            body updates the page of each sequence in place.
        page_id (nl.NkiTensor): [B, 1] int32 HBM, page of each sequence in this layer.
        position (nl.NkiTensor): [B, 1] int32 HBM, position of the token of each sequence.
        tp_group (ReplicaGroup): TP replica group of the output all-reduce.
        tag (str): Unique per-layer name prefix.
        readout (str): Form of the readout and the state matmuls, one of ``KDA_READOUTS``. None (the
            default) takes the form from ``kda_default_readout(B)``.
        sources (dict): None, or the persistent collective tensors of the kernel: the all-reduce then runs in
            ``kda_ar_hbm`` and ``kda_ar_dst_hbm`` (``kda_collective_hbm``).
        conv_dtype (str): Form of the short convolution, one of ``CONV_DTYPES``. ``fp32`` reads the FP32 conv
            weights from ``kda_params`` (``kda_small_columns(Hl, "fp32")``) and keeps the conv output in FP32.
        w_in_buffers (int): SBUF buffers of the ``kda_w_in`` ring of the input projection. 0 (the default) takes
            ``_w_in_buffers(Hl, B)``. More buffers let a load start earlier: a load into a ring buffer waits for the
            matmuls that read the buffer before. Each buffer takes ``_W_IN_TILES_PER_DMA * N * 2`` bytes per
            partition. The output does not change.
        state_loads_first (bool): trace the recurrent-state loads and their selects before the input projection.
            The state tile then lives across the projection, so the allocator gives it its own bytes, not the
            bytes of a ``kda_w_in`` ring buffer, and the loads do not wait for the projection matmuls. Takes
            ``B * Hl * 256`` more bytes per partition during the projection. The output does not change.
        state_loads_early (bool): trace the recurrent-state loads and their selects inside the input projection, right
            after the first fills of the ``kda_w_in`` ring, before its matmuls. The state tile then gets its own bytes,
            as with ``state_loads_first``. On the SP queue the loads follow all the first fills, each other in trace
            order and the weight gate, and every SP refill of the ring follows the last load (schedule edges). So the
            loads land during the attention mix, and the scheduler does not mix them with the refills. The VectorE
            then runs the selects in the idle time of the projection. Takes ``B * Hl * 256`` more bytes per partition during the projection.
            Not with ``state_loads_first``. The output does not change.
        batched_conv_stores (bool): store the conv states of each group of ``128 // (3*Hl)`` sequences with one
            scatter DMA (the Pool queue), not one DMA per sequence. The per-sequence stores write one pool tensor
            with runtime page rows, so each one waits for the one before. The scatter needs a pool width that is a
            multiple of the conv row (``128 * (CONV_TAPS-1)``) and a different write page for each sequence with a
            write (the write order of a page that two sequences share is not defined), or the ``null_page_rows`` of the
            sequences that share one. The off writes all go to page 0, so they store zeros there: the same bytes in
            any order. The output does not change.
        conv_load_gather (bool): load the conv states of each group of ``128 // (3*Hl)`` sequences with one gather
            DMA (the Pool queue), not one DMA per sequence on the SP queue. The SP queue issues each per-sequence load
            with page register loads, and at 16 or more sequences these loads delay the input projection and the
            recurrent-state loads behind them. The gather reads the same rows into the same partitions, and an off
            read still reads page 0, which the window select replaces with zeros. The gathers are traced right after
            the page counts, so they come before the ``kda_w_out`` load in the Pool queue. Needs a pool width that is
            a multiple of the conv row. The output does not change.
        front_hook (callable): None (the default), or a function that the body calls with a list of instructions
            right after the conv-state index ops and the gathers at the start of the body: the last index op of each
            group and each gather DMA. The caller traces its own work there, after those instructions in the
            schedule, so the gathers and the conv transposes that read them do not wait behind it on VectorE.
            With ``batched_conv_stores``, ``sources`` may hold persistent SBUF tiles ``kda_conv_index_<g>`` (int32)
            and ``kda_conv_off_<g>`` (uint16), each ``[>= rows, 1]``, for the scatter rows of group ``g``
            (``kda_conv_index_tiles``). The output does not change.
        scatter_keep (list): None (the default), or a list that the body extends by the SBUF views that its conv-state
            stores read and the stores themselves: ``(source rows [rows, 384], pool-row index [rows, 1] or None, store
            DMAs)`` of each group of sequences (with ``batched_conv_stores`` the index and the one scatter of the group,
            else no index and the per-sequence stores of the group). The compiler ends the live range of a DMA source
            at the DMA trigger, and a later tile that takes the bytes may not wait for the DMA. A caller that reads
            every byte of the views later, after the stores (runtime dependencies, which wait for the stores'
            completion; ``kda_read_scatter_keep``), keeps the bytes until the stores have read them. The output does
            not change.
        conv_after (tuple): instructions that the TensorE transposes of the conv states follow in the schedule (a
            schedule edge on each), for example the ones-sum matmuls of the AttnRes mix before the body, so the
            in-order TensorE does not hold that matmul behind transposes that wait for the conv-state loads. Empty
            (default) adds no edge. The output does not change.
        conv_stores_early (bool): with ``batched_conv_stores``, form the conv-state rows (transposes and the ScalarE
            copy) as soon as the window holds the new conv state, and issue the scatters right after the L2 norms of
            q and k (VectorE and Pool work there waits on the last norm instruction), not after the recurrent-state
            stores. The scatters write the conv part of each page and the recurrent stores the recurrent part, so
            the order between them does not change the pool. The recurrent stores then follow the scatters, and
            the residual mix does not wait behind the recurrent-store chain for the scatter. The output and the
            pool do not change.
        state_dtype (str): Form of the recurrent state in the page, one of ``KDA_STATE_DTYPES``. ``bf16`` (default)
            keeps the program. ``fp32`` and ``fp32_hilo`` read and write an fp32 state at the byte offset of the bf16
            one; the page then ends at ``kda_page_offsets(Hl, state_dtype)["end"]`` bf16 elements (101760 at Hl = 3),
            so the pool must be at least that wide (512-row MLA pages give 147456). The state DMAs move twice the
            bytes, and the state tile takes ``B * Hl * 512`` bytes per partition, plus ``B * Hl * 256`` for each bf16
            stationary part of ``decoupled_bf16`` (one for ``fp32``, two for ``fp32_hilo``).
        paired_pages (bool): with an fp32 ``state_dtype``, each page id names the even page of an aligned pair of
            the pool (``kda_page_span``): the KDA page is the ``2 * width`` elements of the pair, so a 256-row pool
            ``[P, 73728]`` holds the fp32 state with the bytes of the 512-row page ``[P / 2, 147456]`` of id / 2. EVEN
            ids <= P - 2 only, so id + 1 < P: the page counts take even ids in ``[1, P - 1)`` (``kda_page_bound``, P - 1
            excluded) and treat an odd id or an id >= P - 1 as off (``_even_pages_only``: page 0, or the null page).
            The output and the pool bytes are those of the 512-row form. Not with a bf16 state; needs an even
            ``null_page``. False (default) keeps the program.
        bf16_update_rows (bool): with the ``decoupled_bf16`` readout, form the bf16 high and low rows of the state
            update before the transpose (``_bf16_diff_rows``): one bf16 TensorE transpose for each 4 sequences in
            place of an fp32 identity matmul for each sequence, and one broadcast matmul for each head that adds the
            high and the low row in place of two. The outputs and the pool do not change. No effect with the other
            readouts. False (default) keeps the program.
        bf16_rows_flat (bool): with ``bf16_update_rows``, each transpose of the rows reads its data (the stationary
            operand of the matmul) through a flat view, one free dimension of adjacent columns, in place of a
            ``[slots, stride]`` slice with two free dimensions. The compiler BIR verifier refuses a stationary operand
            with two free dimensions. The same bytes move, so the outputs and the pool do not change. Needs
            ``bf16_update_rows``. False (default) keeps the program.
        bf16_parts (tuple): names of ``KDA_BF16_PARTS`` to run in BF16 (an accuracy study; the values change). ()
            (the default) keeps the program.
        kda_w_in_scale (nl.NkiTensor): None (the default, a BF16 ``kda_w_in``), or the ``[128, S, N]`` uint8
            scales of an MXFP8 ``kda_w_in`` ``[H / 512, 128, 4 N]`` (``tkg_mx_quant``): the input projection then runs
            MXFP8 x MXFP8 with the hidden quantized on the VectorE. The values change (MXFP8 rounding of both operands).
        conv_copy_wait (tuple): conv-state store DMAs of earlier bodies that read this body's persistent conv-state
            tiles (``kda_conv_rows_<g>``, ``kda_conv_index_<g>`` and ``kda_conv_off_<g>`` in ``sources``). The first
            write of this body into each such tile gets a runtime dependency on each of them (a wait on its DMA
            completion), so a store finishes reading a tile before a later body rewrites it. Empty (default) adds no
            edge. The output's ``conv_stores`` gives the DMAs to pass to a later body.
        w_in_ring (WInRing): None (the default), or the ``kda_w_in`` ring of this layer whose first fills the caller
            issued earlier (``prefetch_w_in_ring``, for example inside the output collective of the feed-forward body
            before). The input projection then reads those buffers and issues only the refills. The ring sets the
            buffer count (``w_in_buffers`` must be 0 or the same count). Not with ``state_loads_early``. The output
            does not change.
        w_in_head (tuple): empty (the default), or the SBUF tiles of ``kda_w_in_head_loads`` for this layer: the first
            ``len(w_in_head)`` chunks of ``kda_w_in``, which the caller loaded earlier (in the output glue of the layer
            before). The input projection reads those chunks from them, and its ring of ``w_in_buffers`` covers the
            later chunks (``_gemv_channel_major`` ``preloaded``). The output does not change.
        front_ahead (dict): None (the default), or the result of ``kda_decode_front_ahead`` for this layer, which the
            caller traced one phase ahead with the same ``batched_conv_stores`` and ``conv_load_gather``: the page
            counts, the scatter rows, the conv-state gathers and (with ``conv_load_gather``) the staged conv states, and
            the loads of the recurrent states, traced with the ``state_dtype`` of the body. The body then uses them in
            place of its own; it still runs the selects of the recurrent states and, for an fp32 form, makes the bf16
            stationary parts of each selected state right after them (at the place of ``state_loads_early``, or before
            the conv states). The output and the pool do not change.
        reduce_hook (callable): None (the default), or a function that the body calls with the attention all-reduce,
            right after it issues the all-reduce and before it traces the recurrent-state stores. The caller can trace
            the next body's weight loads there (``decode_block_body`` ``moe_in_proj_at_kda_reduce``): their SBUF tiles
            are then live while the staged states are, so the compiler does not place them on those bytes, and the
            loads do not wait for the state stores. The output does not change.
        w_out_after_w_in (bool): False (the default) keeps the program. True: the ``kda_w_out`` load also follows the
            last two ``kda_w_in`` loads (one on each queue) in the schedule and waits for their completion, so it
            streams after the input projection's weights, in the recurrence (where the DMA rings are quiet), not beside
            the output collective of the body before. The output does not change.
        w_in_marks (dict): None (the default), or a dict that the body fills with the ``kda_w_in`` loads (``fills``, in
            chunk order) and the last instruction of the input projection (``end``) (``_gemv_channel_major`` ``marks``),
            for a caller that orders later loads after them. The program does not change.
        null_page_rows (int): 0 (the default) keeps the program. Else the page in ``[1, P)`` that the caller gives each
            invalid sequence (a pad sequence, or one with no KDA state), and that no valid sequence has, so several
            sequences can write it. With ``batched_conv_stores``, a sequence ``s`` at this page writes its conv rows
            to rows ``3*Hl*s .. 3*Hl*s + 3*Hl - 1`` of the page (``_conv_scatter_rows``), not all to its first rows, so
            no two rows of a scatter write one pool row. The rows stay in the page: it needs ``B * 3 * Hl`` conv rows
            of ``128 * (CONV_TAPS-1)`` elements in a page. The other sequences keep their pool rows, and the conv-state
            reads do not change, so the output and every other page keep their bits; only the null page holds other
            bytes. No effect without ``batched_conv_stores``.
        kda_w_out_scale (nl.NkiTensor): None (the default, a BF16 ``kda_w_out``), or the ``[128, 1, H]`` uint8
            scales of an MXFP8 ``kda_w_out`` ``[1, 128, 4 H]`` (``kda_w_out_form``): the local heads are the hT_quad
            lanes and the lanes past them are zero. The output projection then runs MXFP8 x MXFP8 with ``o``
            quantized on the VectorE. The values change (MXFP8 rounding of both operands).
        null_page (int): 0 (the default), or the KDA null page that the caller keeps for no request (a page
            other than 0 that no request holds, for example block 1). A sequence at that page then reads and writes like
            an off one (at page 0 or a page id of P or more): its read gives zero states, and its scatter rows
            (``batched_conv_stores``) store zeros. Every off access goes to the null page, not to page 0. The rows of
            the other pages do not change. It does not go with ``null_page_rows``.

    Returns:
        nl.NkiTensor: [128, B, H/128] bf16 SBUF, the all-reduced attention output in token-major
        hT layout. Its attribute ``side_effects`` holds the state store DMAs, which the output does
        not depend on, and ``conv_stores`` the conv-state store DMAs among them.

    Notes:
        - Run with LNC1 on Trn3.
        - Page 0 is the null page. A sequence at page 0, at a page id of P or more, or at position
          0 reads zero states. The body always reads a valid page (page 0 when the read is off),
          and a select puts zeros in place of an off read, so a NaN in a stale page cannot reach
          the output. A sequence at page 0 or at a page id of P or more writes its new states to
          page 0. Page 0 then holds finite values that no read uses. With ``null_page``, the KDA null page
          takes the place of page 0 here, and a sequence at it is off too. The body has no control
          flow: an ``nl.fori_loop`` is a barrier for all engines, and it stops the weight
          prefetch of the next layer.
        - Do not skip a page access with an out of bounds index and ``oob_mode.skip``. The DGE
          forms the byte offset in 32 bits and checks it against the parent tensor of a view, so
          the access can wrap into another row of a pool that is a view of a larger tensor.
        - Page layout: the conv state ``[3, W, 3]`` (q, k, v; channel; taps oldest first) at
          element 0, and the recurrent state ``[Hl, 128 k, 128 v]`` after it (bf16, or fp32 with ``state_dtype``).
        - The projection GEMV uses each 128x128 weight tile as the stationary operand and the B
          hidden columns of a K-tile as the moving operand, so it gives the channel-major layout
          directly and its TensorE time does not grow with B.
        - The recurrence keeps the key dimension on the partitions. The fp32 state is the
          stationary operand of the prediction matmul, so the prediction comes out with the value
          dimension on the partitions. One transpose per group of sequences puts the difference
          rows on the partitions, and a one-hot stationary column copies each row to all 128
          partitions. The rank-1 update is then one VectorE pass per head.
        - ``o_proj`` uses the weight tile as the stationary operand, so its output is in hT layout.
        - The all-reduce runs on private HBM tiles, see ``_ALL_REDUCE_IN_SBUF``.
        - The readout form comes from ``kda_default_readout(B)``: ``reference`` for one sequence and
          ``decoupled_bf16`` for more. Pass ``readout`` to choose another form.
        - With a decoupled readout the output does not need the new states. The body then runs the
          state update after the output projection and the all-reduce trigger, so the update runs
          while the all-reduce is in flight.

    Pseudocode:
        q, k, v, g, a, b = x @ [Wq | Wk | Wv | Wg | Wf_a | Wb]          # bf16 outputs
        q, k, v = silu(conv(state_conv, q, k, v))                        # bf16, per channel
        q = l2norm(q) / sqrt(128)
        k = l2norm(k)
        decay = exp(-5 * sigmoid(exp(A_log) * (bf16(a @ Wf_b) + dt_bias)))
        beta = sigmoid(b)
        diff = (v - (k * decay) @ S) * beta
        S' = S * decay + k (x) diff
        out = bf16(q @ S')                    # decoupled: bf16((q * decay) @ S + (q . k) * diff)
        o = bf16(out * rsqrt(mean(out^2) + 1e-5) * o_norm * sigmoid(g))
        return all_reduce(bf16(o @ Wo))
    """
    n_seq = x_hT.shape[1]
    hidden_tiles = x_hT.shape[2]
    width, hidden = kda_w_out_form(kda_w_out, kda_w_out_scale, kda_w_small, hidden_tiles)
    heads = width // HEAD_DIM
    cols = kda_in_columns(heads)
    small = kda_small_columns(heads, conv_dtype)
    page = kda_page_offsets(heads, state_dtype)
    n_in = cols["width"]
    n_blocks = cols["blocks"]

    kernel_assert(
        hidden == hidden_tiles * P_MAX, f"x_hT has {hidden_tiles} tiles of 128, kda_w_out has {hidden} columns"
    )
    kernel_assert(
        width == heads * HEAD_DIM and 1 <= heads <= MAX_LOCAL_HEADS,
        f"local KDA width {width} must be 128 * Hl with 1 <= Hl <= {MAX_LOCAL_HEADS}",
    )
    w_in_shape = (hidden, n_in) if kda_w_in_scale is None else mx_hT_shapes(hidden, n_in)[0]
    kernel_assert(tuple(kda_w_in.shape) == w_in_shape, f"kda_w_in shape {kda_w_in.shape} must be {w_in_shape}")
    kernel_assert(
        tuple(kda_w_small.shape) == (P_MAX, small["small_width"]),
        f"kda_w_small shape {kda_w_small.shape} must be ({P_MAX}, {small['small_width']})",
    )
    a_exp_ready = kda_params_a_exp(kda_params, small, heads)
    kernel_assert(not paired_pages or state_dtype != "bf16", "paired_pages is a form of the fp32 KDA state pages")
    kernel_assert(not paired_pages or null_page % 2 == 0, f"paired_pages needs an even null_page, got {null_page}")
    span = kda_page_span(paired_pages)
    kernel_assert(
        span * kv_pool.shape[1] >= page["end"],
        f"cache page width {span * kv_pool.shape[1]} must be >= {page['end']}",
    )
    kernel_assert(
        1 <= n_seq <= MAX_SEQUENCES,
        f"x_hT carries {n_seq} sequences, the KDA decode body supports 1 to {MAX_SEQUENCES}",
    )
    kernel_assert(w_in_buffers >= 0, f"w_in_buffers must be 0 (the default) or a buffer count, got {w_in_buffers}")
    kernel_assert(
        w_in_ring is None or (not state_loads_early and w_in_buffers in (0, w_in_ring.buffers)),
        "w_in_ring sets the ring buffers (w_in_buffers 0 or the same count) and takes no state_loads_early",
    )
    kernel_assert(not conv_stores_early or batched_conv_stores, "conv_stores_early needs batched_conv_stores")
    _check_null_page_rows(null_page_rows, kv_pool.shape[0])
    kernel_assert(
        not (null_page and null_page_rows),
        "null_page (an off page) and null_page_rows (own conv rows) are two forms of the KDA null page: give one",
    )
    kernel_assert(not bf16_rows_flat or bf16_update_rows, "bf16_rows_flat needs bf16_update_rows")
    kernel_assert(
        not (state_loads_first and state_loads_early),
        "state_loads_first and state_loads_early are two places of the same loads: set one of them",
    )
    kernel_assert(
        front_ahead is None
        or (
            front_ahead["batched_conv_stores"] == batched_conv_stores
            and front_ahead["conv_load_gather"] == conv_load_gather
            and front_ahead.get("null_page_rows", 0) == null_page_rows
            and front_ahead.get("null_page", 0) == null_page
            and front_ahead.get("state_dtype", "bf16") == state_dtype
            and front_ahead.get("paired_pages", False) == paired_pages
            and not state_loads_first
            and not conv_copy_wait
        ),
        "front_ahead must come from kda_decode_front_ahead with the batched_conv_stores, conv_load_gather, "
        "null_page_rows, null_page, state_dtype and paired_pages of the body, without state_loads_first (its states are loaded "
        "ahead) and without conv_copy_wait (its scatter rows are written ahead)",
    )
    kernel_assert(
        not w_in_head or kda_w_in_scale is None, "w_in_head takes the BF16 kda_w_in, not an MXFP8 one (kda_w_in_scale)"
    )
    kernel_assert(
        not w_in_head or w_in_ring is None,
        "w_in_head and w_in_ring are two forms of the early kda_w_in loads: use one",
    )
    gemv_marks = {} if w_out_after_w_in or w_in_marks is not None else None
    kernel_assert(
        gemv_marks is None or kda_w_in_scale is None,
        "w_out_after_w_in and w_in_marks take the BF16 kda_w_in, not an MXFP8 one (kda_w_in_scale)",
    )
    kernel_assert(
        not (w_out_after_w_in and kda_w_out_scale is not None),
        "w_out_after_w_in orders the BF16 kda_w_out load: not with an MXFP8 kda_w_out (kda_w_out_scale)",
    )
    readout = kda_default_readout(n_seq) if readout is None else readout
    kernel_assert(readout in KDA_READOUTS, f"readout {readout!r} must be one of {KDA_READOUTS}")
    kernel_assert(
        all(part in KDA_BF16_PARTS for part in bf16_parts), f"bf16_parts {bf16_parts} must be in {KDA_BF16_PARTS}"
    )
    kernel_assert(
        tuple(page_id.shape) == (n_seq, 1) and tuple(position.shape) == (n_seq, 1),
        f"page_id {page_id.shape} and position {position.shape} must be ({n_seq}, 1)",
    )
    kernel_assert(
        state_dtype == "bf16" or kv_pool.shape[1] % 2 == 0,
        f"an fp32 state needs an even page width (an fp32 view of the bf16 pool), got {kv_pool.shape[1]}",
    )
    # The bf16 stationary parts of the fp32 state for the decoupled_bf16 prediction (none for a bf16 state).
    n_parts = _stationary_parts(state_dtype) if readout == "decoupled_bf16" else 0

    bf16 = nl.bfloat16
    fp32 = nl.float32

    def part_dt(part):
        return bf16 if part in bf16_parts else fp32

    n_conv = N_CONV_PROJ * heads
    taps = CONV_TAPS - 1

    # Step 1: constants. The identities are shared tiles of the kernel.
    ones_rows = nl.ndarray((_W_IN_COLUMN_ALIGN, P_MAX), dtype=bf16, buffer=nl.sbuf)
    nisa.memset(dst=ones_rows, value=1.0)
    ones_sq = nl.ndarray((P_MAX, P_MAX), dtype=fp32, buffer=nl.sbuf)
    nisa.memset(dst=ones_sq, value=1.0)
    # nc_matmul needs both operands FP32 or neither: the BF16 squares of the "onorm" part sum with BF16 ones.
    ones_onorm = ones_sq
    if "onorm" in bf16_parts:
        ones_onorm = nl.ndarray((P_MAX, P_MAX), dtype=bf16, buffer=nl.sbuf)
        nisa.memset(dst=ones_onorm, value=1.0)
    ident = nl.shared_identity_matrix(n=P_MAX, dtype=bf16)
    ident32 = nl.shared_identity_matrix(n=P_MAX, dtype=fp32)

    # Step 2: the small weights use the GpSimd DMA queue. The page path uses the SP queue and the
    # VectorE, so the state DMAs do not wait for the GpSimd, which starts late in a collective kernel.
    small_sb = nl.ndarray((P_MAX, small["small_width"]), dtype=bf16, buffer=nl.sbuf)
    # after_result: with the block's consumer gates, the GpSimd queue issues these three loads after the result of the
    # previous body's output collective has landed, not during its flight.
    gate_weight_dma(
        nisa.dma_copy(
            dst=small_sb,
            src=kda_w_small,
            dge_mode=nisa.dge_mode.swdge,
            name=f"{tag}_kda_w_small_load",
        ),
        after_result=True,
    )
    params_sb = nl.ndarray((P_MAX, kda_params.shape[1]), dtype=fp32, buffer=nl.sbuf)
    gate_weight_dma(
        nisa.dma_copy(
            dst=params_sb,
            src=kda_params,
            dge_mode=nisa.dge_mode.swdge,
            name=f"{tag}_kda_params_load",
        ),
        after_result=True,
    )
    if front_ahead is None:
        read_page, write_page, read_mask, write_mask = _page_counts(
            page_id,
            position,
            kda_page_bound(kv_pool.shape[0], paired_pages),
            tag,
            null_page=null_page,
            even_only=paired_pages,
        )
        front_done = [] if front_hook is not None else None
        scatter_rows = (
            _conv_scatter_rows(
                kv_pool,
                write_page,
                n_conv,
                persistent=sources,
                done=front_done,
                wait=conv_copy_wait,
                null_page_rows=null_page_rows,
                null_page=null_page,
            )
            if batched_conv_stores
            else None
        )
        gathered_conv = (
            _gather_conv_rows(kv_pool, read_page, n_conv, tag, done=front_done) if conv_load_gather else None
        )
    else:
        read_page, write_page, read_mask, write_mask = front_ahead["pages"]
        scatter_rows, gathered_conv = front_ahead["scatter_rows"], front_ahead["gathered_conv"]
        front_done = list(front_ahead["done"]) if front_hook is not None else None
    if front_hook is not None:
        front_hook(front_done)

    staged_state = None
    stationaries = ()
    state_form = {"state_dtype": state_dtype, "parts": n_parts}
    if state_loads_first:
        staged_state, stationaries = _load_recurrent_states(
            kv_pool, read_page, read_mask, heads, page, tag, **state_form
        )
    early_states = []

    def early_state_loads(first_fills):
        """Trace the state loads and selects after the first fills of the ring; return the loads (state_loads_early)."""
        loads = []
        if front_ahead is not None:
            # The states were loaded ahead: only their selects run here, and no SP refill waits for a state load.
            early_states.append(_select_recurrent_states(front_ahead["staged_state"], read_mask, heads, n_parts))
            return loads
        early_states.append(
            _load_recurrent_states(
                kv_pool,
                read_page,
                read_mask,
                heads,
                page,
                tag,
                after=first_fills,
                gated=True,
                loads=loads,
                **state_form,
            )
        )
        return loads

    # Step 3: fused input projection in the channel-major layout, then the output projection weight load.
    proj_psum = _gemv_channel_major(
        x_hT,
        kda_w_in,
        tag,
        buffers=w_in_ring.buffers if w_in_ring is not None else w_in_buffers or _w_in_buffers(heads, n_seq),
        **({"after_first_fills": early_state_loads} if state_loads_early else {}),
        **({"w_scale": kda_w_in_scale} if kda_w_in_scale is not None else {}),
        **({"ring": w_in_ring} if w_in_ring is not None else {}),
        **({"preloaded": tuple(w_in_head)} if w_in_head else {}),
        **({"marks": gemv_marks} if gemv_marks is not None else {}),
    )
    if w_in_marks is not None:
        w_in_marks.update(gemv_marks)
    if state_loads_early:
        kernel_assert(len(early_states) == 1, "the input projection calls after_first_fills once")
        staged_state, stationaries = early_states[0]
    if kda_w_out_scale is not None:
        w_out_sb = load_kda_w_out(
            kda_w_out, kda_w_out_scale, heads, hidden, tag, _W_OUT_DMA_PRIORITY, after_result=True
        )
    else:
        w_out_sb = nl.ndarray((P_MAX, heads, hidden), dtype=bf16, buffer=nl.sbuf)
        w_out_load = gate_weight_dma(
            nisa.dma_copy(
                dst=w_out_sb,
                src=kda_w_out.reshape((heads, P_MAX, hidden)).permute((1, 0, 2)),
                priority=dma_priority(_W_OUT_DMA_PRIORITY),
                dge_mode=nisa.dge_mode.swdge,
                name=f"{tag}_kda_w_out_load",
            ),
            after_result=True,
        )
        if w_out_after_w_in:
            after_w_in_loads(w_out_load, gemv_marks)

    # Step 4: load the states of the read page of each sequence (page 0 when the read is off). A
    # select puts zeros in place of an off read, in the recurrent state and in the conv window. A
    # select, not a multiply, so a NaN in a stale page cannot pass. The selects need only the loads
    # and the page path, so the VectorE runs them during the input projection.
    if staged_state is None and front_ahead is not None:
        staged_state, stationaries = _select_recurrent_states(front_ahead["staged_state"], read_mask, heads, n_parts)
    if staged_state is None:
        staged_state, stationaries = _load_recurrent_states(
            kv_pool, read_page, read_mask, heads, page, tag, **state_form
        )
    if front_ahead is not None and front_ahead["staged_conv"] is not None:
        staged_conv = front_ahead["staged_conv"]
    else:
        staged_conv = _load_conv_states(kv_pool, read_page, n_conv, tag, gathered=gathered_conv, after=conv_after)
    window = nl.ndarray((P_MAX, n_seq, n_conv, CONV_TAPS), dtype=bf16, buffer=nl.sbuf)
    nisa.memset(dst=window, value=0.0)
    nisa.tensor_copy_predicated(
        dst=window[0:P_MAX, 0:n_seq, 0:n_conv, 0:taps],
        src=staged_conv,
        predicate=_per_sequence(read_mask, (n_conv, taps)),
    )

    # Step 5: work that needs only the parameters and the states, off the critical path.
    # exp(A_log) runs on the GpSimd, which loads the parameters. On the in-order ScalarE it would
    # hold back the ACT-queue weight loads until the parameters arrive. Trn2 has no GpSimd activation.
    a_exp = kda_a_exp(params_sb, small, heads, a_exp_ready)
    conv_w = _conv_weights(small_sb, params_sb, small, n_conv, conv_dtype)
    conv_dt = fp32 if conv_dtype == "fp32" else bf16
    conv_prod = nl.ndarray((P_MAX, n_seq, n_conv, CONV_TAPS), dtype=conv_dt, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=conv_prod[0:P_MAX, 0:n_seq, 0:n_conv, 0:taps],
        data1=window[0:P_MAX, 0:n_seq, 0:n_conv, 0:taps],
        data2=conv_w[0:P_MAX, 0:n_conv, 0:taps].expand_dim(1).broadcast(1, n_seq),
        op=nl.multiply,
        engine=nisa.engine.gpsimd,
    )

    # Step 6: round the projection to bf16 in the token-major channel layout. Column c of cm holds
    # 128-column block c (the q, k, v and g heads, then f_a), and the last Hl columns hold the b
    # logits on every partition. The b block comes out on partitions 0..Hl-1. An identity mask puts
    # logit h of token t in column (t, h), and a ones matmul adds over the partitions, so each
    # partition gets every logit.
    n_cm = n_blocks + heads
    cm = nl.ndarray((P_MAX, n_seq, n_cm), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_copy(
        dst=cm[0:P_MAX, 0:n_seq, 0:n_blocks],
        src=proj_psum[0:P_MAX, 0:n_blocks, 0:n_seq].permute((0, 2, 1)),
        engine=nisa.engine.vector,
    )
    b_rows = n_in - n_blocks * P_MAX
    b_logits = nl.ndarray((b_rows, n_seq), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=b_logits, src=proj_psum[0:b_rows, n_blocks, 0:n_seq], engine=nisa.engine.scalar)
    b_masked = nl.ndarray((b_rows, n_seq, heads), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=b_masked,
        data1=ident[0:b_rows, 0:heads].expand_dim(1).broadcast(1, n_seq),
        data2=b_logits.expand_dim(2).broadcast(2, heads),
        op=nl.multiply,
    )
    b_psum = nl.ndarray((P_MAX, n_seq, heads), dtype=fp32, buffer=nl.psum)
    nisa.nc_matmul(dst=b_psum, stationary=ones_rows[0:b_rows, 0:P_MAX], moving=b_masked)
    nisa.tensor_copy(dst=cm[0:P_MAX, 0:n_seq, n_blocks:n_cm], src=b_psum, engine=nisa.engine.scalar)

    # Step 7: short convolution: products of the window [state taps | new value], summed in fp32 and rounded
    # to conv_dt, then SiLU (bf16: bf16 products and output). ScalarE runs exp and ln only, which share one
    # function table.
    nisa.tensor_tensor(
        dst=conv_prod[0:P_MAX, 0:n_seq, 0:n_conv, taps],
        data1=cm[0:P_MAX, 0:n_seq, 0:n_conv],
        data2=conv_w[0:P_MAX, 0:n_conv, taps].expand_dim(1).broadcast(1, n_seq),
        op=nl.multiply,
    )
    conv_sum = nl.ndarray((P_MAX, n_seq, n_conv), dtype=conv_dt, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=conv_sum, op=nl.add, data=conv_prod, axis=3)
    conv_exp = nl.ndarray((P_MAX, n_seq, n_conv), dtype=fp32, buffer=nl.sbuf)
    nisa.activation(dst=conv_exp, op=nl.exp, data=conv_sum, scale=-1.0)
    conv_out = nl.ndarray((P_MAX, n_seq, n_conv), dtype=conv_dt, buffer=nl.sbuf)
    _divide_by_one_plus(conv_out, conv_exp, conv_sum)

    # Step 8: decay gate, decay = exp(-5 * sigmoid(exp(A_log) * (bf16(f_a @ Wf_b) + dt_bias))).
    gate_psum = nl.ndarray((P_MAX, heads, n_seq), dtype=fp32, buffer=nl.psum)
    for head_idx in range(heads):
        f_b_start = small["f_b"] + head_idx * HEAD_DIM
        nisa.nc_matmul(
            dst=gate_psum[0:P_MAX, head_idx, 0:n_seq],
            stationary=small_sb[0:P_MAX, f_b_start : f_b_start + HEAD_DIM],
            moving=cm[0:P_MAX, 0:n_seq, 4 * heads],
        )
    gate_bf = nl.ndarray((P_MAX, n_seq, heads), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=gate_bf, src=gate_psum.permute((0, 2, 1)), engine=nisa.engine.vector)
    gate_shift = nl.ndarray((P_MAX, n_seq, heads), dtype=part_dt("gate"), buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=gate_shift,
        data1=gate_bf,
        data2=params_sb[0:P_MAX, small["dt_bias"] : small["dt_bias"] + heads].expand_dim(1).broadcast(1, n_seq),
        op=nl.add,
    )
    gate_scaled = nl.ndarray((P_MAX, n_seq, heads), dtype=part_dt("gate"), buffer=nl.sbuf)
    nisa.tensor_tensor(dst=gate_scaled, data1=gate_shift, data2=a_exp.expand_dim(1).broadcast(1, n_seq), op=nl.multiply)
    gate_exp = nl.ndarray((P_MAX, n_seq, heads), dtype=part_dt("gate"), buffer=nl.sbuf)
    nisa.activation(dst=gate_exp, op=nl.exp, data=gate_scaled, scale=-1.0)
    gate_sig = nl.ndarray((P_MAX, n_seq, heads), dtype=part_dt("gate"), buffer=nl.sbuf)
    _divide_by_one_plus(gate_sig, gate_exp, 1.0)
    decay = nl.ndarray((P_MAX, n_seq, heads), dtype=fp32, buffer=nl.sbuf)
    if "decay" in bf16_parts:
        # The BF16 values in an FP32 tile: a tensor_scalar operand must be FP32.
        decay_bf = nl.ndarray((P_MAX, n_seq, heads), dtype=bf16, buffer=nl.sbuf)
        nisa.activation(dst=decay_bf, op=nl.exp, data=gate_sig, scale=GATE_LOWER_BOUND)
        nisa.tensor_copy(dst=decay, src=decay_bf)
    else:
        nisa.activation(dst=decay, op=nl.exp, data=gate_sig, scale=GATE_LOWER_BOUND)

    # Step 9: output gate and beta, sigmoid of the g and b columns of the channel-major tile.
    n_gb = n_cm - 3 * heads
    gb_exp = nl.ndarray((P_MAX, n_seq, n_gb), dtype=part_dt("sigmoid"), buffer=nl.sbuf)
    nisa.activation(dst=gb_exp, op=nl.exp, data=cm[0:P_MAX, 0:n_seq, 3 * heads : n_cm], scale=-1.0)
    gb_sig = nl.ndarray((P_MAX, n_seq, n_gb), dtype=part_dt("sigmoid"), buffer=nl.sbuf)
    _divide_by_one_plus(gb_sig, gb_exp, 1.0)
    out_gate = gb_sig[0:P_MAX, 0:n_seq, 0:heads]
    beta = gb_sig[0:P_MAX, 0:n_seq, heads + 1 : 2 * heads + 1]

    # Step 10: the new conv state is the last CONV_TAPS-1 taps of the window.
    nisa.tensor_copy(
        dst=window[0:P_MAX, 0:n_seq, 0:n_conv, taps], src=cm[0:P_MAX, 0:n_seq, 0:n_conv], engine=nisa.engine.gpsimd
    )
    conv_rows = _conv_state_rows(window, sources, conv_copy_wait) if conv_stores_early else None

    # Step 11: L2 norms of q and k. One ones matmul gives the sums over the partitions.
    qk_sq = nl.ndarray((P_MAX, n_seq, 2 * heads), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=qk_sq,
        data1=conv_out[0:P_MAX, 0:n_seq, 0 : 2 * heads],
        data2=conv_out[0:P_MAX, 0:n_seq, 0 : 2 * heads],
        op=nl.multiply,
    )
    qk_ss = nl.ndarray((P_MAX, n_seq, 2 * heads), dtype=fp32, buffer=nl.psum)
    nisa.nc_matmul(dst=qk_ss, stationary=ones_sq, moving=qk_sq, is_stationary_onezero=True)
    # rsqrt(x) = exp(-0.5 * ln(x)). ln and exp share one ScalarE function table, so the ScalarE
    # does not switch its table (a cost on the critical path for each switch, two per layer).
    qk_log = nl.ndarray((P_MAX, n_seq, 2 * heads), dtype=fp32, buffer=nl.sbuf)
    nisa.activation(dst=qk_log, op=nl.log, data=qk_ss, bias=L2_EPS)
    qk_rnorm = nl.ndarray((P_MAX, n_seq, 2 * heads), dtype=fp32, buffer=nl.sbuf)
    nisa.activation(dst=qk_rnorm, op=nl.exp, data=qk_log, scale=-0.5)
    # k_n and q_n share one tile [128, B, Hl, 2] (k, then q), so one instruction scales both by
    # the decay for the moving columns of the prediction matmul.
    qk_n = nl.ndarray((P_MAX, n_seq, heads, 2), dtype=part_dt("norm"), buffer=nl.sbuf)
    k_n = qk_n[0:P_MAX, 0:n_seq, 0:heads, 0]
    q_n = qk_n[0:P_MAX, 0:n_seq, 0:heads, 1]
    nisa.scalar_tensor_tensor(
        dst=q_n,
        data=conv_out[0:P_MAX, 0:n_seq, 0:heads],
        op0=nl.multiply,
        operand0=float(HEAD_DIM) ** -0.5,
        op1=nl.multiply,
        operand1=qk_rnorm[0:P_MAX, 0:n_seq, 0:heads],
    )
    nisa.tensor_tensor(
        dst=k_n,
        data1=conv_out[0:P_MAX, 0:n_seq, heads : 2 * heads],
        data2=qk_rnorm[0:P_MAX, 0:n_seq, heads : 2 * heads],
        op=nl.multiply,
    )
    # The moving columns of the prediction matmul: k * decay, then q * decay for a decoupled readout.
    # decoupled_bf16 splits each column into bf16 high and low parts.
    decoupled = readout != "reference"
    split = readout == "decoupled_bf16"
    lo = split and "moving" not in bf16_parts
    n_vec = 2 if decoupled else 1
    n_mv = n_vec * (2 if lo else 1)
    mv_in = nl.ndarray((P_MAX, n_seq, heads, n_mv), dtype=bf16 if split else fp32, buffer=nl.sbuf)
    decay_cols = decay.expand_dim(3).broadcast(3, n_vec)
    if split:
        # Columns (k_hi, k_lo, q_hi, q_lo) of the decayed vectors.
        scaled = nl.ndarray((P_MAX, n_seq, heads, n_vec), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=scaled, data1=qk_n[0:P_MAX, 0:n_seq, 0:heads, 0:n_vec], data2=decay_cols, op=nl.multiply)
        high = mv_in[0:P_MAX, 0:n_seq, 0:heads, 0 : n_mv : n_mv // n_vec]
        nisa.tensor_copy(dst=high, src=scaled, engine=nisa.engine.vector)
        if lo:
            nisa.tensor_tensor(dst=mv_in[0:P_MAX, 0:n_seq, 0:heads, 1:n_mv:2], data1=scaled, data2=high, op=nl.subtract)
    else:
        nisa.tensor_tensor(dst=mv_in, data1=qk_n[0:P_MAX, 0:n_seq, 0:heads, 0:n_vec], data2=decay_cols, op=nl.multiply)
    qk_psum = None
    if decoupled:
        qk_prod = nl.ndarray((P_MAX, n_seq, heads), dtype=fp32, buffer=nl.sbuf)
        norm_last = nisa.tensor_tensor(dst=qk_prod, data1=q_n, data2=k_n, op=nl.multiply)
        qk_psum = nl.ndarray((P_MAX, n_seq, heads), dtype=fp32, buffer=nl.psum)
        nisa.nc_matmul(dst=qk_psum, stationary=ones_sq, moving=qk_prod, is_stationary_onezero=True)
    else:
        norm_last = None
    early_conv_stores = ()
    if conv_rows is not None:
        early_conv_stores = _scatter_conv_rows(
            kv_pool, conv_rows, scatter_rows, tag, after=norm_last, keep=scatter_keep
        )

    # Step 12: delta rule and readout, see _recurrence.
    out_bf, state_update = _recurrence(
        staged_state,
        conv_out[0:P_MAX, 0:n_seq, 2 * heads : 3 * heads],
        k_n,
        q_n,
        decay,
        beta,
        mv_in,
        qk_psum,
        ident,
        ident32,
        readout,
        stationaries,
        bf16_update_rows=bf16_update_rows,
        bf16_update="update" in bf16_parts,
        bf16_rows_flat=bf16_rows_flat,
    )

    # Step 13: gated RMSNorm over v, o = bf16(out * rsqrt(mean(out^2) + eps) * o_norm * out_gate).
    out_sq = nl.ndarray((P_MAX, n_seq, heads), dtype=part_dt("onorm"), buffer=nl.sbuf)
    nisa.tensor_tensor(dst=out_sq, data1=out_bf, data2=out_bf, op=nl.multiply)
    out_ss = nl.ndarray((P_MAX, n_seq, heads), dtype=fp32, buffer=nl.psum)
    nisa.nc_matmul(dst=out_ss, stationary=ones_onorm, moving=out_sq, is_stationary_onezero=True)
    out_log = nl.ndarray((P_MAX, n_seq, heads), dtype=part_dt("onorm"), buffer=nl.sbuf)
    nisa.activation(dst=out_log, op=nl.log, data=out_ss, scale=1.0 / HEAD_DIM, bias=NORM_EPS)
    # out * gate runs while the TensorE and the ScalarE form the norm, so one VectorE pass follows
    # the norm: o = bf16((rstd * o_norm) * (out * gate)).
    out_gated = nl.ndarray((P_MAX, n_seq, heads), dtype=part_dt("onorm"), buffer=nl.sbuf)
    nisa.tensor_tensor(dst=out_gated, data1=out_bf, data2=out_gate, op=nl.multiply)
    out_rstd = nl.ndarray((P_MAX, n_seq, heads), dtype=part_dt("onorm"), buffer=nl.sbuf)
    nisa.activation(dst=out_rstd, op=nl.exp, data=out_log, scale=-0.5)
    o_bf = nl.ndarray((P_MAX, n_seq, heads), dtype=bf16, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=o_bf,
        data=out_rstd,
        op0=nl.multiply,
        operand0=params_sb[0:P_MAX, small["o_norm"] : small["o_norm"] + 1],
        op1=nl.multiply,
        operand1=out_gated,
    )

    # Step 14: output projection into the token-major hT layout, then the all-reduce over the TP group.
    part_bf, last_matmul, last_copy = _output_projection(o_bf, w_out_sb, hidden_tiles, tag)
    reduced = nl.ndarray((P_MAX, n_seq, hidden_tiles), dtype=bf16, buffer=nl.sbuf)
    update_last = None
    if _ALL_REDUCE_IN_SBUF:
        all_reduce = release_weight_gate(
            ncc.all_reduce(
                srcs=[part_bf], dsts=[reduced], replica_group=tp_group, op=nl.add, name=f"{tag}_kda_all_reduce"
            )
        )
        if state_update is not None:
            update_last = state_update(last_matmul, last_copy, last_copy)
    else:
        shape = (P_MAX, n_seq, hidden_tiles)
        if sources and "kda_ar_hbm" in sources:
            # The persistent pair may hold more rows than this call (``kda_collective_hbm`` ``reduce_rows``):
            # the all-reduce then runs on the whole buffer so the DP1 decode's reduction has the verify's
            # message size (the runtime picks the collective algorithm by message size; bitwise decode ==
            # verify). This call stores its rows in the first ``n_seq`` and reads them back.
            ar_src, ar_dst = sources["kda_ar_hbm"], sources["kda_ar_dst_hbm"]
            reduce_rows = ar_src.shape[1]
            kernel_assert(
                tuple(ar_src.shape) == (P_MAX, reduce_rows, hidden_tiles) and reduce_rows >= n_seq,
                f"kda_ar_hbm must be (128, >= {n_seq}, {hidden_tiles})",
            )
        else:
            ar_src = nl.ndarray(shape, dtype=bf16, buffer=nl.private_hbm, name=f"{tag}_kda_ar_src")
            ar_dst = nl.ndarray(shape, dtype=bf16, buffer=nl.private_hbm, name=f"{tag}_kda_ar_dst")
        # The pair may hold more rows than this call: the store and load keep the first n_seq rows, the
        # all-reduce runs on the whole pair. Without padding use the whole tensor so the IR is unchanged.
        store_dst = ar_src if ar_src.shape[1] == n_seq else ar_src[0:P_MAX, 0:n_seq, 0:hidden_tiles]
        load_src = ar_dst if ar_dst.shape[1] == n_seq else ar_dst[0:P_MAX, 0:n_seq, 0:hidden_tiles]
        # The AR DMAs use the ACT queue. The SP queue carries the state DMAs, which wait for the
        # recurrence, and an in-order queue holds every later DMA behind them.
        ar_store = nisa.dma_copy(
            dst=store_dst,
            src=part_bf,
            dge_mode=nisa.dge_mode.hwdge,
            engine=nisa.engine.scalar,
            name=f"{tag}_kda_ar_src_store",
        )
        check_pair_group(ar_dst, tp_group, f"{tag}_kda_all_reduce")
        all_reduce = release_weight_gate(
            ncc.all_reduce(
                srcs=[ar_src], dsts=[ar_dst], replica_group=tp_group, op=nl.add, name=f"{tag}_kda_all_reduce"
            )
        )
        if state_update is not None:
            # A decoupled readout updates the states while the all-reduce is in flight. The AR
            # destination load follows in program order, so the in-order ScalarE does not wait
            # for the all-reduce before the decays of the update.
            update_last = state_update(last_matmul, last_copy, ar_store)
        nisa.dma_copy(
            dst=reduced,
            src=load_src,
            dge_mode=nisa.dge_mode.hwdge,
            engine=nisa.engine.scalar,
            name=f"{tag}_kda_ar_dst_load",
        )

    if reduce_hook is not None:
        reduce_hook(all_reduce)

    # Step 15: write the states to the write page of each sequence (page 0 when the write is off).
    # The recurrence rounded the recurrent states into staged_state.
    stores = []
    for seq in range(n_seq):
        stores.append(
            nisa.dma_copy(
                dst=_recurrent_state_view(
                    kv_pool, write_page[0:1, seq : seq + 1], heads, page["recurrent"], state_dtype
                ),
                src=staged_state[0:P_MAX, seq, 0:heads, 0:HEAD_DIM],
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.sync,
                name=f"{tag}_kda_recurrent_state_store_{seq}",
            )
        )
    if conv_rows is not None:
        conv_stores = list(early_conv_stores)
    else:
        # With bf16_update_rows the update returns its last broadcast matmul, and the conv-state transposes follow it in
        # the schedule, as they do without the option: the scheduler otherwise moves them between the o_proj matmuls.
        conv_stores = _store_conv_states(
            kv_pool,
            write_page,
            window,
            tag,
            sources,
            scatter_rows=scatter_rows,
            keep=scatter_keep,
            transpose_after=() if update_last is None else (update_last,),
            conv_wait=conv_copy_wait,
        )
    stores.extend(conv_stores)
    object.__setattr__(reduced, "side_effects", tuple(stores))
    object.__setattr__(reduced, "conv_stores", tuple(conv_stores))
    return reduced


kda_decode_attention.token_major = True
# decode_block_body passes scatter_keep to a KDA body with conv_keep, and conv_copy_wait to one with conv_tiles (a body
# that writes the persistent conv-state tiles of its sources).
kda_decode_attention.conv_keep = True
kda_decode_attention.conv_tiles = True


def _recurrence(
    staged_state,
    v_cols,
    k_n,
    q_n,
    decay,
    beta,
    mv_in,
    qk_psum,
    ident,
    ident32,
    readout,
    stationaries=(),
    bf16_update_rows=False,
    bf16_update=False,
    bf16_rows_flat=False,
):
    """Run the delta rule of each sequence. Return the bf16 readout ``[128 v, B, Hl]`` and the state update.

    ``staged_state`` (bf16, or fp32 for an fp32 state) holds the selected states (zeros for an off read). The update
    writes the new recurrent states into it. ``stationaries`` (``decoupled_bf16`` with an fp32 state) are the bf16
    tiles ``[128 k, B, Hl, 128 v]`` of ``_load_recurrent_states``: each is a stationary operand of the prediction of
    each head, and their products add in PSUM. Empty (default): the stationary is the bf16 ``staged_state``.

    For each sequence the selected state ``S[k, h, v]`` is the stationary operand of one matmul per
    head, which gives the prediction ``S^T (k * decay)`` with v on the partitions. A group of
    sequences then forms ``diff = (v - prediction) * beta`` and transposes it, so each (sequence,
    head) row is on its own partition. A one-hot stationary column copies the row to all 128
    partitions, and one VectorE pass per head adds ``k * diff`` to the decayed state.

    The ``reference`` readout is ``S'^T q``, one more matmul per head on the fp32 new state, so this
    function runs the update and returns None for it. The decoupled readouts are
    ``S^T (q * decay) + (q . k) * diff``: they come from the prediction matmul. This function then
    returns the update as a function of three instructions of the output path (the last ``o_proj``
    matmul, the last ``o_proj`` copy and the AR source store). The caller runs it after the
    all-reduce trigger, and each update instruction gets a schedule dependency on the instruction of
    its engine. So the TensorE, the VectorE and the ScalarE run the output path first, and the
    update runs while the all-reduce is in flight. ``decoupled_bf16`` uses the bf16 state and bf16 high and
    low parts of the vectors, see ``KDA_READOUTS``. With ``bf16_update_rows`` (``decoupled_bf16`` only) the update
    takes its rows from ``_bf16_diff_rows`` and adds the high and the low row in one broadcast matmul per head; the update
    function then returns its last broadcast matmul (else None), which the conv-state transposes follow in the schedule.
    """
    n_seq, heads = k_n.shape[1], k_n.shape[2]
    n_mv = mv_in.shape[3]
    fp32 = nl.float32
    bf16 = nl.bfloat16
    decoupled = readout != "reference"
    split = readout == "decoupled_bf16"
    kernel_assert(
        not split or bool(stationaries) == (staged_state.dtype == fp32),
        "decoupled_bf16 needs the bf16 stationary parts of an fp32 state, and none for a bf16 state",
    )
    bf16_rows = bf16_update_rows and split
    kernel_assert(
        not (bf16_rows and bf16_update),
        "bf16_update_rows and the bf16 update part (bf16_parts 'update') are separate forms of the state update: use one",
    )
    lo = split and n_mv == 2 * (2 if decoupled else 1)
    group = min(n_seq, _RECURRENCE_GROUP)
    n_bufs = min(n_seq, 2 * group)
    state_dt = bf16 if bf16_update else fp32
    state_bufs = [nl.ndarray((P_MAX, heads, HEAD_DIM), dtype=state_dt, buffer=nl.sbuf) for _ in range(n_bufs)]
    pred_psum = nl.ndarray((P_MAX, n_seq, heads, n_mv), dtype=fp32, buffer=nl.psum)
    diff = nl.ndarray((P_MAX, n_seq, heads), dtype=fp32, buffer=nl.sbuf)
    out_bf = nl.ndarray((P_MAX, n_seq, heads), dtype=bf16, buffer=nl.sbuf)
    qkb = None
    if decoupled:
        qkb = nl.ndarray((P_MAX, n_seq, heads), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=qkb, data1=qk_psum, data2=beta, op=nl.multiply)
    groups = [(first, min(n_seq, first + group)) for first in range(0, n_seq, group)]

    def predict(first, last):
        """Predict, and for the reference readout also decay the fp32 state in place."""
        width = last - first
        for seq in range(first, last):
            state = state_bufs[seq % n_bufs]
            loaded = staged_state[0:P_MAX, seq, 0:heads, 0:HEAD_DIM]
            if split:
                parts = [part[0:P_MAX, seq, 0:heads, 0:HEAD_DIM] for part in stationaries] or [loaded]
            elif decoupled and staged_state.dtype == fp32:
                # An fp32 state is the fp32 stationary itself; the update decays it into the buffer.
                parts = [loaded]
            else:
                # The GpSimd widens the state to fp32 for the fp32 matmuls.
                nisa.tensor_copy(dst=state, src=loaded, engine=nisa.engine.gpsimd)
                parts = [state]
            stationary = parts[0]
            for head_idx in range(heads):
                for part_idx, part in enumerate(parts):
                    # The parts of a head accumulate back to back, so no other group opens in the PSUM zone between.
                    nisa.nc_matmul(
                        dst=pred_psum[0:P_MAX, seq, head_idx, 0:n_mv],
                        stationary=part[0:P_MAX, head_idx, 0:HEAD_DIM],
                        moving=mv_in[0:P_MAX, seq, head_idx, 0:n_mv],
                        accumulate=part_idx > 0,
                    )
            if not decoupled:
                _decay_state(state, stationary, decay, seq)
        # diff = (v - prediction) * beta for the group, with v on the partitions.
        v_minus = nl.ndarray((P_MAX, width, heads), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=v_minus,
            data1=v_cols[0:P_MAX, first:last, 0:heads],
            data2=pred_psum[0:P_MAX, first:last, 0:heads, 0],
            op=nl.subtract,
        )
        if lo:
            nisa.tensor_tensor(
                dst=v_minus, data1=v_minus, data2=pred_psum[0:P_MAX, first:last, 0:heads, 1], op=nl.subtract
            )
        if decoupled:
            # out = S^T (q * decay) + (q . k) * diff = S^T (q * decay) + (v - prediction) * qkb,
            # with qkb = (q . k) * beta from before the prediction. So the output does not wait
            # for diff, and diff (for the update) follows it.
            q_part = nl.ndarray((P_MAX, width, heads), dtype=fp32, buffer=nl.sbuf)
            q_col = 2 if lo else 1
            nisa.tensor_tensor(dst=q_part, data1=v_minus, data2=qkb[0:P_MAX, first:last, 0:heads], op=nl.multiply)
            for col in range(q_col, n_mv):
                last_col = col == n_mv - 1
                nisa.tensor_tensor(
                    dst=out_bf[0:P_MAX, first:last, 0:heads] if last_col else q_part,
                    data1=q_part,
                    data2=pred_psum[0:P_MAX, first:last, 0:heads, col],
                    op=nl.add,
                )
        nisa.tensor_tensor(
            dst=diff[0:P_MAX, first:last, 0:heads],
            data1=v_minus,
            data2=beta[0:P_MAX, first:last, 0:heads],
            op=nl.multiply,
        )

    def update(first, last, bcast_bufs, out_psum, after, bf16_parts=None):
        """Add k (x) diff to the decayed states of the group and write the new states (bf16, or fp32 for an fp32 state).

        ``after`` maps the engines "pe", "dve" and "act" to an instruction of the output path (or
        is None). An instruction of the update on that engine gets a schedule dependency on it, so
        the scheduler does not move the update in front of the output path. Only the instructions
        whose inputs are ready early need the edge: the transpose, the copies of its rows, and the
        decays. The broadcast matmuls read the row copies, so they follow the transpose anyway. An
        edge costs trace time that grows with the kernel, so the update adds no other edges.

        ``bf16_parts`` is None, or the PSUM row tiles and the row selector of ``_bf16_update_parts``: the rows then
        come from ``_bf16_diff_rows``, and one broadcast matmul for each head adds the high and the low row. It returns
        the last broadcast matmul of the group with ``bf16_parts``, else None.
        """

        def order(instr, engine):
            # scalar_tensor_tensor returns no instruction handle. Its inputs come from instructions
            # with an edge, so the scheduler places it after them.
            if after is not None and hasattr(instr, "depends_on"):
                instr.depends_on(after[engine], kind=nisa.dependency.schedule)
            return instr

        # Transpose each sequence of the group, so row (sequence, head) is on its own partition. The
        # rows of a sequence start at partition 32 (seq - first), and each sequence gets its own row
        # tiles, because the broadcast matmul of a sequence must read only its own rows: a matmul adds
        # a 0 x row term for every other row of its moving operand, and 0 x Inf is NaN. A shared tile
        # would let one sequence with a non-finite state break the new state of every sequence of the
        # group. The PSUM tile has the width of a shared group tile, so PSUM holds the same bytes.
        stride = -(-heads // _QUADRANT) * _QUADRANT
        per_tile = max(1, P_MAX // stride)
        width = last - first
        last_bcast = None
        if bf16_parts is not None:
            rows_of = _bf16_diff_rows(diff, first, last, bf16_parts[0], order, flat=bf16_rows_flat)
        else:
            rows_psum = [
                nl.ndarray((min(per_tile, width - start) * stride, HEAD_DIM), dtype=fp32, buffer=nl.psum)
                for start in range(0, width, per_tile)
            ]
            rows_of = {}
            for seq in range(first, last):
                index = seq - first
                base = (index % per_tile) * stride
                source = rows_psum[index // per_tile][base : base + heads, 0:HEAD_DIM]
                order(
                    nisa.nc_matmul(
                        dst=source,
                        stationary=diff[0:P_MAX, seq, 0:heads],
                        moving=ident32,
                        is_moving_onezero=True,
                        accumulate=False,
                    ),
                    "pe",
                )
                if split:
                    tiles = [nl.ndarray((heads, HEAD_DIM), dtype=bf16, buffer=nl.sbuf) for _ in range(2)]
                    order(nisa.tensor_copy(dst=tiles[0], src=source, engine=nisa.engine.scalar), "act")
                    order(nisa.tensor_tensor(dst=tiles[1], data1=source, data2=tiles[0], op=nl.subtract), "dve")
                else:
                    tiles = [nl.ndarray((heads, HEAD_DIM), dtype=fp32, buffer=nl.sbuf)]
                    order(nisa.tensor_copy(dst=tiles[0], src=source, engine=nisa.engine.scalar), "act")
                rows_of[seq] = tiles
        one_hot = ident if split else ident32
        for seq in range(first, last):
            state = state_bufs[seq % n_bufs]
            if decoupled:
                # The decoupled path decays the state here, from the selected bf16 state, because
                # the prediction of later groups reuses the fp32 buffers.
                for instr in _decay_state(state, staged_state[0:P_MAX, seq, 0:heads, 0:HEAD_DIM], decay, seq):
                    order(instr, "act")
            bcast = bcast_bufs[seq % len(bcast_bufs)]
            for head_idx in range(heads):
                if bf16_parts is not None:
                    # One matmul adds the high and the low row of the head: high + low is exact in fp32.
                    rows, base = rows_of[seq]
                    last_bcast = nisa.nc_matmul(
                        dst=bcast[0:P_MAX, head_idx, 0:HEAD_DIM],
                        stationary=bf16_parts[1][base : base + 2 * heads, head_idx : head_idx + 1].broadcast(1, P_MAX),
                        moving=rows,
                        accumulate=False,
                    )
                    continue
                for part_idx, part in enumerate(rows_of[seq]):
                    # No order edge: the moving rows come from the ordered transpose and copies.
                    nisa.nc_matmul(
                        dst=bcast[0:P_MAX, head_idx, 0:HEAD_DIM],
                        stationary=one_hot[0:heads, head_idx : head_idx + 1].broadcast(1, P_MAX),
                        moving=part,
                        is_stationary_onezero=not split,
                        accumulate=part_idx > 0,
                    )
            # S' = S * decay + k (x) diff. A decoupled readout needs only the stored S' (bf16 or fp32).
            for head_idx in range(heads):
                order(
                    nisa.scalar_tensor_tensor(
                        dst=staged_state[0:P_MAX, seq, head_idx, 0:HEAD_DIM]
                        if decoupled
                        else state[0:P_MAX, head_idx, 0:HEAD_DIM],
                        data=bcast[0:P_MAX, head_idx, 0:HEAD_DIM],
                        op0=nl.multiply,
                        operand0=k_n[0:P_MAX, seq, head_idx : head_idx + 1],
                        op1=nl.add,
                        operand1=state[0:P_MAX, head_idx, 0:HEAD_DIM],
                    ),
                    "dve",
                )
            if not decoupled:
                for head_idx in range(heads):
                    nisa.nc_matmul(
                        dst=out_psum[0:P_MAX, seq, head_idx : head_idx + 1],
                        stationary=state[0:P_MAX, head_idx, 0:HEAD_DIM],
                        moving=q_n[0:P_MAX, seq, head_idx : head_idx + 1],
                        accumulate=False,
                    )
                nisa.tensor_copy(
                    dst=staged_state[0:P_MAX, seq, 0:heads, 0:HEAD_DIM], src=state, engine=nisa.engine.gpsimd
                )
        return last_bcast if bf16_parts is not None else None

    def new_bcast_bufs():
        # Two PSUM tiles take turns for the broadcast rows. A reused PSUM tile needs
        # accumulate=False: with the default (None) the compiler accumulates every write after
        # the first one.
        return [nl.ndarray((P_MAX, heads, HEAD_DIM), dtype=fp32, buffer=nl.psum) for _ in range(min(n_seq, 2))]

    if decoupled:
        for first, last in groups:
            predict(first, last)

        def run_update(after_pe, after_dve, after_act):
            after = {"pe": after_pe, "dve": after_dve, "act": after_act}
            bcast_bufs = new_bcast_bufs()
            bf16_parts = _bf16_update_parts(ident, heads) if bf16_rows else None
            last_bcast = None
            for first, last in groups:
                last_bcast = update(first, last, bcast_bufs, None, after, bf16_parts)
            return last_bcast

        return out_bf, run_update

    out_psum = nl.ndarray((P_MAX, n_seq, heads), dtype=fp32, buffer=nl.psum)
    bcast_bufs = new_bcast_bufs()
    for first, last in groups:
        predict(first, last)
        update(first, last, bcast_bufs, out_psum, None)
    nisa.tensor_copy(dst=out_bf, src=out_psum, engine=nisa.engine.vector)
    return out_bf, None


def _bf16_update_parts(ident, heads, selector=None):
    """Return the two bf16 PSUM row tiles and the bf16 row selector of ``_bf16_diff_rows``.

    The row tiles ``[128, 128]`` take turns, and each PSUM tensor takes its own bank, so the transpose of one tile does
    not wait for the ScalarE copy out of the other (the compiler orders cross-engine accesses to one bank).
    ``selector``: None (default) builds the selector (``_bf16_row_selector``); else a selector that the caller built
    once for many calls (the same values).
    """
    if selector is None:
        selector = _bf16_row_selector(ident, heads)
    row_psum = [nl.ndarray((P_MAX, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.psum) for _ in range(2)]
    return row_psum, selector


def _bf16_row_selector(ident, heads):
    """Return the bf16 row selector ``[128, Hl]`` of ``_bf16_update_parts``.

    The selector holds, at partition ``32 j + r`` (``r < 2 Hl``) and column ``h``, 1 for ``r = h`` (the high row of head
    ``h``) and ``r = Hl + h`` (its low row), else 0: the GpSimd sums two columns of the bf16 identity. The other
    partitions are not written and no matmul reads them.
    """
    rows = 2 * heads
    stride = -(-rows // _QUADRANT) * _QUADRANT
    selector = nl.ndarray((P_MAX, heads), dtype=nl.bfloat16, buffer=nl.sbuf)
    for base in range(0, P_MAX, stride):
        nisa.tensor_tensor(
            dst=selector[base : base + rows, 0:heads],
            data1=ident[base : base + rows, base : base + heads],
            data2=ident[base : base + rows, base + heads : base + rows],
            op=nl.add,
            engine=nisa.engine.gpsimd,
        )
    return selector


def bf16_diff_cols(heads: int, width: int, engine=None):
    """Return a zeroed bf16 rows tile of ``_bf16_diff_rows`` for ``width`` sequences of ``heads`` heads (``cols``).

    ``_bf16_diff_rows`` writes only the high and low rows of each slot, so the zero columns of a tile that it reuses stay
    zero. ``engine`` (default GpSimd) runs the memset.
    """
    rows = 2 * heads
    stride = -(-rows // _QUADRANT) * _QUADRANT
    per_tile = max(1, P_MAX // stride)
    n_tiles = -(-width // per_tile)
    cols = nl.ndarray((P_MAX, n_tiles * per_tile, stride), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=cols, value=0.0, engine=engine or nisa.engine.gpsimd)
    return cols


def _bf16_diff_rows(diff, first, last, row_psum, order=None, flat=False, slot=None, cols=None) -> dict:
    """Return the bf16 high and low rows of the difference of the sequences ``first .. last - 1`` for the update.

    ``diff`` is the fp32 difference ``[128 v, B, Hl]``. The result maps each sequence to ``(rows, base)``: ``rows``
    ``[2 Hl, 128]`` (SBUF, partitions ``base .. base + 2 Hl - 1``) holds the high rows ``bf16(diff)`` of the Hl heads,
    then the low rows ``bf16(diff - high)``, with the value dimension on the free axis.

    The high and the low parts come from the same ScalarE copy and VectorE subtract as those of the fp32 identity
    matmul of ``_recurrence``, on the same fp32 values, so they hold the same bits. One bf16 TensorE transpose (a bit
    move) for each 4 sequences then puts them on the partitions, and one ScalarE copy moves each transposed tile to
    SBUF. Each sequence takes a quadrant of 32 partitions, so its broadcast matmuls read only its own rows (0 x Inf
    is NaN). The columns past the 2 Hl rows of a sequence are zeros that no matmul reads. ``row_psum`` are the PSUM
    tiles of ``_bf16_update_parts``. ``order(instr, engine)`` (None: no edges) adds the schedule edges of the decode
    update on the engines "act", "dve" and "pe". With ``flat`` each transpose reads its ``count * stride`` columns
    through a flat view of the slots (one free dimension; the same bytes), else as a ``[count, stride]`` slice.
    ``slot`` (None: the tile of ``first`` in its group of 4 sequences) picks the first ``row_psum`` tile, so that two
    sequence groups of one token take different PSUM tiles. ``cols`` (None: a new tile and its memset here) is a zeroed
    rows tile of ``bf16_diff_cols`` for these sequences: the call writes only its high and low rows, so its zero
    columns stay zero and the call issues no memset. The same bytes reach the transposes.
    """
    order = order or (lambda instr, engine: instr)
    heads = diff.shape[2]
    rows = 2 * heads
    stride = -(-rows // _QUADRANT) * _QUADRANT
    per_tile = max(1, P_MAX // stride)
    width = last - first
    n_tiles = -(-width // per_tile)
    if cols is None:
        cols = nl.ndarray((P_MAX, n_tiles * per_tile, stride), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(dst=cols, value=0.0, engine=nisa.engine.gpsimd)
    else:
        kernel_assert(
            tuple(cols.shape) == (P_MAX, n_tiles * per_tile, stride) and cols.dtype == nl.bfloat16,
            f"cols must be a bf16 [{P_MAX}, {n_tiles * per_tile}, {stride}] tile, got {cols.dtype} {tuple(cols.shape)}",
        )
    high = cols[0:P_MAX, 0:width, 0:heads]
    order(nisa.tensor_copy(dst=high, src=diff[0:P_MAX, first:last, 0:heads], engine=nisa.engine.scalar), "act")
    order(
        nisa.tensor_tensor(
            dst=cols[0:P_MAX, 0:width, heads:rows],
            data1=diff[0:P_MAX, first:last, 0:heads],
            data2=high,
            op=nl.subtract,
            engine=nisa.engine.vector,
        ),
        "dve",
    )
    # A TensorE transpose loads data as its stationary operand, which takes one free dimension (the BIR verifier refuses
    # two). The slots of a tile are adjacent in the free axis, so the flat view reads count * stride columns as one run.
    cols_flat = cols.reshape((P_MAX, n_tiles * per_tile * stride)) if flat else None
    rows_of = {}
    for tile in range(n_tiles):
        start = tile * per_tile
        count = min(per_tile, width - start)
        tile_psum = row_psum[((first // per_tile if slot is None else slot) + tile) % len(row_psum)]
        if flat:
            data = cols_flat[0:P_MAX, start * stride : (start + count) * stride]
        else:
            data = cols[0:P_MAX, start : start + count, 0:stride]
        order(
            nisa.nc_transpose(
                dst=tile_psum[0 : count * stride, 0:HEAD_DIM],
                data=data,
                engine=nisa.engine.tensor,
            ),
            "pe",
        )
        tile_sb = nl.ndarray((count * stride, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
        order(
            nisa.tensor_copy(dst=tile_sb, src=tile_psum[0 : count * stride, 0:HEAD_DIM], engine=nisa.engine.scalar),
            "act",
        )
        for index in range(count):
            base = index * stride
            rows_of[first + start + index] = (tile_sb[base : base + rows, 0:HEAD_DIM], base)
    return rows_of


def _decay_state(state, source, decay, seq):
    """Write ``source * decay`` (decay per key row and head) into the fp32 ``state`` on the ScalarE. Return the instructions."""
    heads = state.shape[1]
    return [
        nisa.tensor_scalar(
            dst=state[0:P_MAX, head_idx, 0:HEAD_DIM],
            data=source[0:P_MAX, head_idx, 0:HEAD_DIM],
            op0=nl.multiply,
            operand0=decay[0:P_MAX, seq, head_idx : head_idx + 1],
            engine=nisa.engine.scalar,
        )
        for head_idx in range(heads)
    ]


def kda_w_out_form(kda_w_out, kda_w_out_scale, kda_w_small, hidden_tiles: int) -> tuple[int, int]:
    """Return ``(W, H)`` of the output projection: the BF16 ``[W, H]`` shape, or the MXFP8 padded-lane form.

    The MXFP8 ``kda_w_out`` is ``[1, 128, 4 H]`` uint8 with ``[128, 1, H]`` scales (``tkg_mx_quant`` hT_quad): lane
    ``q`` holds the 128 rows of local head ``q``, and the lanes past the local heads are zero. Its head count comes
    from the ``kda_w_small`` width.
    """
    if kda_w_out_scale is None:
        return tuple(kda_w_out.shape)
    hidden = hidden_tiles * P_MAX
    data_shape, scale_shape = mx_hT_shapes(MX_LANES * P_MAX, hidden)
    kernel_assert(
        tuple(kda_w_out.shape) == data_shape and tuple(kda_w_out_scale.shape) == scale_shape,
        f"MXFP8 kda_w_out {kda_w_out.shape} / scales {kda_w_out_scale.shape} must be {data_shape} / {scale_shape}",
    )
    kernel_assert(is_trn3(), "the MXFP8 output projection needs Trn3")
    return kda_local_heads(kda_w_out, kda_w_small) * HEAD_DIM, hidden


def kda_local_heads(kda_w_out, kda_w_small) -> int:
    """Return the local KDA heads of one layer's weights: the BF16 ``kda_w_out`` rows, or the ``kda_w_small`` width.

    ``kda_w_small`` is ``[128, 140 Hl]`` in every conv form (``kda_small_columns``), so it also gives the head count
    of the MXFP8 padded-lane ``kda_w_out``, whose shape does not.
    """
    if kda_w_out.dtype == nl.bfloat16:
        return kda_w_out.shape[0] // HEAD_DIM
    heads = [h for h in range(1, MAX_LOCAL_HEADS + 1) if kda_small_columns(h)["small_width"] == kda_w_small.shape[1]]
    kernel_assert(len(heads) == 1, f"kda_w_small width {kda_w_small.shape[1]} gives no local head count")
    return heads[0]


def load_kda_w_out(
    kda_w_out,
    kda_w_out_scale,
    heads: int,
    hidden: int,
    tag: str,
    priority: int,
    after_result=False,
    weight_dma="pool",
    after: tuple = (),
):
    """Load the output projection: the BF16 ``[128, Hl, H]`` tile, or the MXFP8 ``(data [128, 4 H], scale copies)``, on
    the queue of ``weight_dma`` (``KDA_WEIGHT_DMAS``). Each load follows the instructions ``after`` in the schedule and
    waits for their completion (none by default)."""
    queue = kda_weight_dma_options(weight_dma)

    def _after(instr):
        if after:
            _schedule_after(instr, after)
            _runtime_after(instr, after)
        return instr

    if kda_w_out_scale is None:
        w_out_sb = nl.ndarray((P_MAX, heads, hidden), dtype=nl.bfloat16, buffer=nl.sbuf)
        _after(
            gate_weight_dma(
                nisa.dma_copy(
                    dst=w_out_sb,
                    src=kda_w_out.reshape((heads, P_MAX, hidden)).permute((1, 0, 2)),
                    priority=dma_priority(priority),
                    **queue,
                    name=f"{tag}_kda_w_out_load",
                ),
                **({"after_result": True} if after_result else {}),
            )
        )
        return w_out_sb
    data_sb = nl.ndarray((P_MAX, kda_w_out.shape[2]), dtype=kda_w_out.dtype, buffer=nl.sbuf)
    _after(
        gate_weight_dma(
            nisa.dma_copy(
                dst=data_sb,
                src=kda_w_out[0],
                priority=dma_priority(priority),
                **queue,
                name=f"{tag}_kda_w_out_load",
            ),
            **({"after_result": True} if after_result else {}),
        )
    )

    def load_scales(dst, src, copy):
        _after(
            gate_weight_dma(
                nisa.dma_copy(
                    dst=dst,
                    src=src,
                    priority=dma_priority(priority),
                    **queue,
                    name=f"{tag}_kda_w_out_scale_load_{copy}",
                )
            )
        )

    scales = load_mx_scales(
        kda_w_out_scale, lambda shape, copy: nl.ndarray(shape, dtype=nl.uint8, buffer=nl.sbuf), load_scales
    )
    return data_sb, scales


def _output_projection(o_bf, w_out_sb, hidden_tiles, tag: str = "kda", out=None, filler=None, filler_every=0):
    """Return ``bf16(o @ Wo)`` in the token-major hT layout ``[128, B, H/128]``, the last matmul and the last copy.

    The weight tile is the stationary operand and the B values of a head are the moving operand. A
    PSUM tile holds the hidden tiles of one bank, with the token stride padded to a power of two,
    so no matmul output crosses a bank. With the MXFP8 weight of ``load_kda_w_out`` (a tuple), ``o`` is quantized
    in the hT_quad lanes (zero lanes past the heads) and each hidden tile is one MXFP8 matmul over the 512 rows.
    ``out``: optional bf16 SBUF ``[128, B, H/128]`` tile for the result (None allocates one). ``filler`` (None: none) is
    called as ``filler(last matmul)`` after every ``filler_every`` hidden tiles of matmuls (not after the last): the
    caller traces work there that sits between them in the TensorE order (``kda_verify_attention`` ``late_keep_warm``).
    The instruction ``filler`` returns (None: none) goes ahead of the next tile's matmuls in the schedule, so the
    compiler keeps it between the two tile groups.
    """
    n_seq, heads = o_bf.shape[1], o_bf.shape[2]
    seq_stride = 1 << (n_seq - 1).bit_length()
    tiles_per_bank = _PSUM_BANK_FP32 // seq_stride
    if out is None:
        part_bf = nl.ndarray((P_MAX, n_seq, hidden_tiles), dtype=nl.bfloat16, buffer=nl.sbuf)
    else:
        kernel_assert(
            tuple(out.shape) == (P_MAX, n_seq, hidden_tiles) and out.dtype == nl.bfloat16 and out.buffer == nl.sbuf,
            f"the output projection tile must be a bf16 SBUF [{P_MAX}, {n_seq}, {hidden_tiles}] tile",
        )
        part_bf = out
    last_matmul = last_copy = pinned = None
    mx_form = isinstance(w_out_sb, tuple)
    if mx_form:
        w_data, w_scales = w_out_sb
        w_mx = w_data.view(nl.float8_e4m3fn_x4)
        # An even token stride keeps the moving columns on an 8-byte boundary (MatmulMX refuses odd x4 starts).
        o_mx, o_scale = quantize_hT_mx(o_bf, f"{tag}_kda_w_out", n_seq + n_seq % 2)
    for first in range(0, hidden_tiles, tiles_per_bank):
        last = min(hidden_tiles, first + tiles_per_bank)
        part_psum = nl.ndarray((P_MAX, last - first, seq_stride), dtype=nl.float32, buffer=nl.psum)
        for tile_idx in range(first, last):
            if filler is not None and filler_every and tile_idx and tile_idx % filler_every == 0:
                pinned = filler(last_matmul)
            if mx_form:
                last_matmul = nisa.nc_matmul_mx(
                    dst=part_psum[0:P_MAX, tile_idx - first, 0:n_seq],
                    stationary=w_mx[0:P_MAX, tile_idx * P_MAX : (tile_idx + 1) * P_MAX],
                    moving=o_mx[0:P_MAX, 0, 0:n_seq],
                    stationary_scale=mx_scale_view(w_scales, 0, tile_idx * P_MAX, P_MAX),
                    moving_scale=o_scale[0:P_MAX, 0, 0:n_seq],
                    accumulate=False,
                )
                _schedule_after(last_matmul, (pinned,))
                pinned = None
                continue
            for head_idx in range(heads):
                last_matmul = nisa.nc_matmul(
                    dst=part_psum[0:P_MAX, tile_idx - first, 0:n_seq],
                    stationary=w_out_sb[0:P_MAX, head_idx, tile_idx * P_MAX : (tile_idx + 1) * P_MAX],
                    moving=o_bf[0:P_MAX, 0:n_seq, head_idx],
                    accumulate=head_idx > 0,
                )
                _schedule_after(last_matmul, (pinned,))
                pinned = None
        last_copy = nisa.tensor_copy(
            dst=part_bf[0:P_MAX, 0:n_seq, first:last],
            src=part_psum[0:P_MAX, 0 : last - first, 0:n_seq].permute((0, 2, 1)),
            engine=nisa.engine.vector,
        )
    return part_bf, last_matmul, last_copy


def _per_sequence(mask, inner):
    """Return the ``[128, B]`` per-sequence ``mask`` broadcast to ``[128, B, *inner]``."""
    view = mask
    for dim, size in enumerate(inner):
        view = view.expand_dim(2 + dim).broadcast(2 + dim, size)
    return view


def kda_in_columns(local_heads: int) -> dict:
    """Return the column layout of the fused ``kda_w_in`` tensor.

    Args:
        local_heads (int): KDA heads of one rank.

    Returns:
        dict: The start columns ``q``, ``k``, ``v``, ``g``, ``f_a`` and ``b``, the number of
        128-column blocks ``blocks`` (the q, k, v and g heads, then f_a), and the padded width
        ``width``.

    Example:
        >>> kda_in_columns(3)["width"]
        1672
    """
    width = local_heads * HEAD_DIM
    b_col = 4 * width + HEAD_DIM
    return {
        "q": 0,
        "k": width,
        "v": 2 * width,
        "g": 3 * width,
        "f_a": 4 * width,
        "b": b_col,
        "blocks": 4 * local_heads + 1,
        "width": div_ceil(b_col + local_heads, _W_IN_COLUMN_ALIGN) * _W_IN_COLUMN_ALIGN,
    }


def kda_in_columns_head_major(local_heads: int) -> dict:
    """Return the head-major column layout of the fused ``kda_w_in`` (the ``kda_w_in_head_major`` weight order).

    The columns of ``kda_in_columns`` in the order f_a, b, then q / k / v of each head, then g of each head, so the
    columns that the first heads need come first. ``starts`` maps each output block of ``kda_in_columns`` (q heads
    0 .. Hl - 1, k heads, v heads, g heads, f_a, then b) to its start column here, and ``part_a`` is the end of the first
    DMA part of a weight group (f_a, b and the q / k / v of heads 0 .. Hl - 2); the second part holds the q / k / v of
    the last head and the g of all heads.

    Example:
        >>> kda_in_columns_head_major(3)["part_a"]
        904
    """
    cols = kda_in_columns(local_heads)
    heads = local_heads
    b_width = cols["width"] - cols["b"]
    order = [(4 * heads, P_MAX), (4 * heads + 1, b_width)]
    order += [(kind * heads + head, P_MAX) for head in range(heads) for kind in range(3)]
    order += [(3 * heads + head, P_MAX) for head in range(heads)]
    starts, widths, col = {}, {}, 0
    for block, width in order:
        starts[block], widths[block] = col, width
        col += width
    # Part A ends after the v block of head Hl - 2 (after b at one head).
    part_a = starts[3 * heads - 2] + P_MAX if heads > 1 else P_MAX + b_width
    return {"starts": starts, "widths": widths, "part_a": part_a, "width": col}


def kda_in_head_major_permutation(local_heads: int) -> list:
    """Return the source column of each column of the head-major ``kda_w_in`` (``kda_in_columns_head_major``)."""
    cols = kda_in_columns(local_heads)
    major = kda_in_columns_head_major(local_heads)
    old = {block: block * P_MAX for block in range(4 * local_heads + 1)}
    old[4 * local_heads + 1] = cols["b"]
    perm = [0] * major["width"]
    for block, start in major["starts"].items():
        for i in range(major["widths"][block]):
            perm[start + i] = old[block] + i
    return perm


def kda_conv_body(body, conv_dtype: str):
    """Return the KDA body ``body`` with ``conv_dtype`` bound, with the attributes of ``body`` (``token_major``)."""
    kernel_assert(conv_dtype in CONV_DTYPES, f"conv_dtype {conv_dtype!r} must be one of {CONV_DTYPES}")

    @functools.wraps(body)
    def bound(*args, **kwargs):
        return body(*args, conv_dtype=conv_dtype, **kwargs)

    return bound


def kda_options_body(body, **options):
    """Return the KDA body ``body`` with keyword ``options`` bound, with the attributes of ``body`` (``token_major``)."""

    @functools.wraps(body)
    def bound(*args, **kwargs):
        return body(*args, **options, **kwargs)

    return bound


def kda_w_in_head_bytes(kda_w_in: nl.NkiTensor) -> int:
    """Return the SBUF bytes per partition of one ``kda_w_in`` head chunk (``kda_w_in_head_loads``)."""
    return _W_IN_TILES_PER_DMA * kda_w_in.shape[-1] * 2


def kda_w_in_head_loads(
    kda_w_in: nl.NkiTensor, address: int, chunks: int, tag: str, after_load=None, after_collective=None
) -> tuple:
    """Load the first ``chunks`` chunks of a layer's ``kda_w_in`` into SBUF tiles at explicit addresses; return the tiles.

    The tiles are the ``w_in_head`` of ``kda_decode_attention``: its input projection reads its first chunks from
    them. Chunk ``c`` takes the explicit bytes ``[address + c * S, address + (c + 1) * S)`` of every partition, with
    ``S = kda_w_in_head_bytes(kda_w_in)``. Each load is one hardware DGE DMA on the ACT queue (the SP queue carries the
    state and glue DMAs of the layers). ``after_load`` is None, or the instruction that the loads wait for (a semaphore
    wait, for example the load of the result of the routed all-reduce of the MoE layer before), and
    ``after_collective`` the instruction that they follow in the schedule (that all-reduce), so the compiler does not
    move them into the flight of a collective. The caller gives addresses that no other explicit tile uses while the
    head is live, and where no compiler-placed tile is live at the time of the loads.

    Args:
        kda_w_in (nl.NkiTensor): [H, N] bf16 HBM, the layer's fused input projection.
        address (int): explicit SBUF byte of the first chunk (bytes per partition, a multiple of 32).
        chunks (int): head chunks, whole ``_W_IN_TILES_PER_DMA``-tile chunks before the last chunk.
        tag (str): Unique per-layer name prefix of the layer that reads the head.
        after_load, after_collective: see above.

    Returns:
        tuple: ``chunks`` SBUF tiles ``[128, 4, N]`` bf16.
    """
    hidden, n_out = kda_w_in.shape
    hidden_tiles = hidden // P_MAX
    sizes = _w_in_chunk_sizes(hidden_tiles)
    kernel_assert(
        1 <= chunks < len(sizes) and all(size == _W_IN_TILES_PER_DMA for size in sizes[:chunks]),
        f"a kda_w_in head takes 1 to {len(sizes) - 1} whole chunks of {_W_IN_TILES_PER_DMA} tiles, got {chunks}",
    )
    kernel_assert(kda_w_in.dtype == nl.bfloat16, f"kda_w_in must be bf16, got {kda_w_in.dtype}")
    kernel_assert(address >= 0 and address % 32 == 0, f"head address {address} must be a multiple of 32")
    chunk_bytes = kda_w_in_head_bytes(kda_w_in)
    w_tiles = kda_w_in.reshape((hidden_tiles, P_MAX, n_out))
    tiles = []
    for chunk_idx in range(chunks):
        tile = nl.ndarray(
            (P_MAX, _W_IN_TILES_PER_DMA, n_out),
            dtype=kda_w_in.dtype,
            buffer=nl.sbuf,
            name=f"{tag}_kda_w_in_head_{chunk_idx}",
            address=(0, address + chunk_idx * chunk_bytes),
        )
        first = chunk_idx * _W_IN_TILES_PER_DMA
        load = nisa.dma_copy(
            dst=tile,
            src=w_tiles[first : first + _W_IN_TILES_PER_DMA].permute((1, 0, 2)),
            priority=dma_priority(_W_IN_HEAD_DMA_PRIORITY),
            dge_mode=nisa.dge_mode.hwdge,
            engine=nisa.engine.scalar,
            name=f"{tag}_kda_w_in_head_load_{chunk_idx}",
        )
        handle = load if hasattr(load, "depends_on") else nisa.NkiInstruction(load)
        if after_load is not None:
            handle.depends_on(after_load, kind=nisa.dependency.runtime)
        if after_collective is not None:
            handle.depends_on(after_collective, kind=nisa.dependency.schedule)
        tiles.append(tile)
    return tuple(tiles)


def kda_decode_front_ahead(
    kv_pool: nl.NkiTensor,
    page_id: nl.NkiTensor,
    position: nl.NkiTensor,
    heads: int,
    tag: str,
    batched_conv_stores: bool = False,
    conv_load_gather: bool = False,
    sources: dict = None,
    after: tuple = (),
    null_page_rows: int = 0,
    null_page: int = 0,
    state_dtype: str = "bf16",
    paired_pages: bool = False,
) -> dict:
    """Trace the front of ``kda_decode_attention`` that needs only the page metadata and the cache pool, and return it
    as the ``front_ahead`` of the body.

    These are the page counts, the conv-state scatter rows (``batched_conv_stores``), the conv-state gathers with
    their rows, transposes and copy into the staged conv states (``conv_load_gather``), and the loads of the recurrent
    states (on the ACT queue, without their selects, which the body runs). They read neither the hidden nor the weights
    of the layer, and the pool rows that they read are written only by the same layer later. So a caller can trace them
    one phase ahead, in the flight of the output all-reduce of the feed-forward layer before. ``after`` holds the
    instructions that the metadata loads and the row iotas follow in the schedule (the ops that read no earlier tile
    of the front), so the front starts after them. ``sources`` holds the persistent index tiles of the scatter rows that
    the body would take (``kda_conv_index_tiles``), or None. ``null_page_rows`` is the ``null_page_rows`` of the body (0, the
    default, keeps the program). ``state_dtype`` is the ``state_dtype`` of the body (``KDA_STATE_DTYPES``): an fp32 form
    loads the fp32 states of the pages (the body makes their bf16 stationary parts after the selects); ``bf16`` (the
    default) keeps the program. ``paired_pages`` is the ``paired_pages`` of the body. The instructions are those of the
    body front, so the values do not change.

    Returns:
        dict: ``pages`` (read page, write page, read mask, write mask), ``scatter_rows``, ``gathered_conv``,
        ``staged_conv`` (None without ``conv_load_gather``), ``staged_state`` (loaded, not selected), ``done`` (the
        ``front_hook`` list of the body: the last index op of each group and each gather DMA), ``vector_done`` (the
        VectorE ops of the page counts and the rows, for the schedule edges of the caller), the two flags and
        ``null_page_rows``.
    """
    _check_null_page_rows(null_page_rows, kv_pool.shape[0])
    kernel_assert(
        not (null_page and null_page_rows),
        "null_page (an off page) and null_page_rows (own conv rows) are two forms of the KDA null page: give one",
    )
    kernel_assert(state_dtype in KDA_STATE_DTYPES, f"state_dtype {state_dtype!r} must be one of {KDA_STATE_DTYPES}")
    kernel_assert(
        state_dtype == "bf16" or kv_pool.shape[1] % 2 == 0,
        f"an fp32 state needs an even page width (an fp32 view of the bf16 pool), got {kv_pool.shape[1]}",
    )
    n_conv = N_CONV_PROJ * heads
    page = kda_page_offsets(heads, state_dtype)
    kernel_assert(not paired_pages or state_dtype != "bf16", "paired_pages is a form of the fp32 KDA state pages")
    kernel_assert(not paired_pages or null_page % 2 == 0, f"paired_pages needs an even null_page, got {null_page}")
    span = kda_page_span(paired_pages)
    kernel_assert(
        span * kv_pool.shape[1] >= page["end"],
        f"cache page width {span * kv_pool.shape[1]} must be >= {page['end']}",
    )
    page_done, index_done, done = [], [], []
    pages = _page_counts(
        page_id,
        position,
        kda_page_bound(kv_pool.shape[0], paired_pages),
        tag,
        after=after,
        done=page_done,
        null_page=null_page,
        even_only=paired_pages,
    )
    read_page, write_page, read_mask, _ = pages
    scatter_rows = (
        _conv_scatter_rows(
            kv_pool,
            write_page,
            n_conv,
            persistent=sources,
            done=done,
            after=after,
            index_done=index_done,
            null_page_rows=null_page_rows,
            null_page=null_page,
        )
        if batched_conv_stores
        else None
    )
    gathered_conv = (
        _gather_conv_rows(kv_pool, read_page, n_conv, tag, done=done, after=after, index_done=index_done)
        if conv_load_gather
        else None
    )
    # _gather_conv_rows adds its gather DMAs after all its index ops, one for each group. The pool-row ops of each group
    # (index_done) are not in done: the gathers read them, so the caller's edges follow them too.
    vector_done = page_done + index_done + (done[: len(done) - len(gathered_conv)] if gathered_conv else list(done))
    staged_conv = (
        _load_conv_states(kv_pool, read_page, n_conv, tag, gathered=gathered_conv) if conv_load_gather else None
    )
    staged_state, _ = _load_recurrent_states(
        kv_pool,
        read_page,
        read_mask,
        heads,
        page,
        tag,
        state_dtype=state_dtype,
        engine=nisa.engine.scalar,
        select=False,
    )
    return {
        "pages": pages,
        "scatter_rows": scatter_rows,
        "gathered_conv": gathered_conv,
        "staged_conv": staged_conv,
        "staged_state": staged_state,
        "done": done,
        "vector_done": vector_done,
        "batched_conv_stores": batched_conv_stores,
        "conv_load_gather": conv_load_gather,
        "null_page_rows": null_page_rows,
        "null_page": null_page,
        "state_dtype": state_dtype,
        "paired_pages": paired_pages,
    }


def kda_front_ahead_body(
    body,
    batched_conv_stores: bool = False,
    conv_load_gather: bool = False,
    null_page_rows: int = 0,
    null_page: int = 0,
):
    """Return the KDA decode body ``body`` with the attribute ``front_ahead``: ``kda_decode_front_ahead`` with the conv
    options of the body bound (``null_page_rows`` only when it is not 0), called as
    ``front_ahead(kv_pool, page_id, position, heads, tag, sources, after)``.
    ``decode_block_body`` (``kda_front_ahead``) then traces the front of each KDA layer that follows a 2-server MoE layer
    of the block in the flight of that layer's output all-reduce, and passes it to the body. The output does not
    change."""

    @functools.wraps(body)
    def bound(*args, **kwargs):
        return body(*args, **kwargs)

    bound.front_ahead = functools.partial(
        kda_decode_front_ahead,
        batched_conv_stores=batched_conv_stores,
        conv_load_gather=conv_load_gather,
        **({"null_page_rows": null_page_rows} if null_page_rows else {}),
        **({"null_page": null_page} if null_page else {}),
    )
    return bound


def kda_small_columns(local_heads: int, conv_dtype: str = "bf16") -> dict:
    """Return the column layouts of the ``kda_w_small`` and ``kda_params`` tensors.

    Args:
        local_heads (int): KDA heads of one rank.
        conv_dtype (str): Form of the short convolution, one of ``CONV_DTYPES``. ``fp32`` adds the FP32 conv
            weights to ``kda_params`` at ``conv32``.

    Returns:
        dict: The ``kda_w_small`` start columns ``f_b`` and ``conv`` and its width
        ``small_width``, and the ``kda_params`` start columns ``dt_bias``, ``a_log``, ``o_norm``
        (and ``conv32`` for ``fp32``) and its width ``params_width``. A ``kda_params`` may also be
        ``params_width + local_heads`` wide: its last ``local_heads`` columns (from ``a_exp``) then hold
        ``exp(A_log)`` (``kda_params_a_exp``), which the bodies read in place of computing it.

    Example:
        >>> kda_small_columns(3)["small_width"]
        420
    """
    kernel_assert(conv_dtype in CONV_DTYPES, f"conv_dtype {conv_dtype!r} must be one of {CONV_DTYPES}")
    width = local_heads * HEAD_DIM
    cols = {
        "f_b": 0,
        "conv": width,
        "small_width": width + N_CONV_PROJ * local_heads * CONV_TAPS,
        "dt_bias": 0,
        "a_log": local_heads,
        "o_norm": 2 * local_heads,
        "params_width": 2 * local_heads + 1,
    }
    if conv_dtype == "fp32":
        cols["conv32"] = cols["params_width"]
        cols["params_width"] += N_CONV_PROJ * local_heads * CONV_TAPS
    cols["a_exp"] = cols["params_width"]
    return cols


def kda_params_a_exp(kda_params: nl.NkiTensor, small: dict, local_heads: int) -> bool:
    """Return True when ``kda_params`` carries the precomputed ``exp(A_log)`` columns (``kda_small_columns`` ``a_exp``,
    ``params_width + local_heads`` wide), False for the plain width; any other width stops the trace."""
    width = kda_params.shape[1]
    kernel_assert(
        tuple(kda_params.shape) in ((P_MAX, small["params_width"]), (P_MAX, small["params_width"] + local_heads)),
        f"kda_params shape {kda_params.shape} must be ({P_MAX}, {small['params_width']}) or "
        f"({P_MAX}, {small['params_width'] + local_heads}) with the exp(A_log) columns",
    )
    return width == small["params_width"] + local_heads


def kda_a_exp(params_sb: nl.NkiTensor, small: dict, heads: int, precomputed: bool) -> nl.NkiTensor:
    """Return ``exp(A_log)`` ``[128, Hl]`` fp32 of the loaded ``kda_params``: the precomputed columns (no instruction),
    or one exp activation (on the GpSimd on trn3, which loads the parameters; on the ScalarE on Trn2)."""
    if precomputed:
        return params_sb[0:P_MAX, small["a_exp"] : small["a_exp"] + heads]
    a_exp = nl.ndarray((P_MAX, heads), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=a_exp,
        op=nl.exp,
        data=params_sb[0:P_MAX, small["a_log"] : small["a_log"] + heads],
        engine=nisa.engine.gpsimd if is_trn3() else nisa.engine.scalar,
    )
    return a_exp


def kda_a_exp_init(kda_params: nl.NkiTensor, out: nl.NkiTensor, local_heads: int, conv_dtype: str = "bf16") -> None:
    """Write the wide ``kda_params`` of each layer: the plain columns, then ``exp(A_log)`` at ``a_exp``.

    The weight-load step of the precomputed form (``kda_small_columns`` ``a_exp``): it runs the activation instruction
    of the bodies (``kda_a_exp``: the exp on the GpSimd on trn3, on the ScalarE on Trn2) on the loaded parameters, so the
    columns hold the bits of the per-step exp, and the per-step programs then carry no exp.

    Args:
        kda_params (nl.NkiTensor): [L, 128, W] fp32 HBM, the plain ``kda_params`` of L layers (W = ``params_width``).
        out (nl.NkiTensor): [L, 128, W + Hl] fp32 HBM, the wide ``kda_params``.
        local_heads (int): KDA heads of one rank (Hl).
        conv_dtype (str): The ``conv_dtype`` of the layout (``kda_small_columns``).
    """
    small = kda_small_columns(local_heads, conv_dtype)
    n_layers, width = kda_params.shape[0], small["params_width"]
    kernel_assert(
        tuple(kda_params.shape) == (n_layers, P_MAX, width),
        f"kda_params shape {kda_params.shape} must be ({n_layers}, {P_MAX}, {width})",
    )
    kernel_assert(
        tuple(out.shape) == (n_layers, P_MAX, width + local_heads),
        f"out shape {out.shape} must be ({n_layers}, {P_MAX}, {width + local_heads})",
    )
    for layer in range(n_layers):
        params_sb = nl.ndarray((P_MAX, width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=params_sb, src=kda_params[layer], dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
        a_exp = kda_a_exp(params_sb, small, local_heads, False)
        nisa.dma_copy(
            dst=out[layer, 0:P_MAX, 0:width], src=params_sb, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync
        )
        nisa.dma_copy(
            dst=out[layer, 0:P_MAX, width : width + local_heads],
            src=a_exp,
            dge_mode=nisa.dge_mode.hwdge,
            engine=nisa.engine.sync,
        )


@nki.jit
def kimi_k3_kda_a_exp_init_kernel(kda_params: nl.NkiTensor, conv_dtype: str = "bf16") -> nl.NkiTensor:
    """Return the wide ``kda_params`` of the precomputed ``exp(A_log)`` form: the weight-load kernel of the KDA layers.

    A kernel of its own (one NEFF), run once per rank at weight load and not fused into the verify or decode kernels:
    it runs ``kda_a_exp_init`` on the plain ``kda_params``, so the ``exp(A_log)`` columns hold the bits of the per-step
    exp of the bodies (the GpSimd on trn3, the ScalarE on Trn2). The verify and decode bodies then take the wide
    parameters and carry no exp.

    Dimensions:
        L: KDA layers.
        Hl: KDA heads of one rank.
        W: ``kda_small_columns(Hl, conv_dtype)["params_width"]``, 2*Hl + 1 (``bf16``) or 14*Hl + 1 (``fp32``).

    Args:
        kda_params (nl.NkiTensor): [L, 128, W] fp32 HBM, the plain ``kda_params`` of the L layers (``dt_bias``,
            ``A_log``, ``o_norm``, and the fp32 conv weights with ``conv_dtype`` ``fp32``).
        conv_dtype (str): The ``conv_dtype`` of the layout, one of ``CONV_DTYPES``.

    Returns:
        nl.NkiTensor: [L, 128, W + Hl] fp32 HBM: the plain columns, then ``exp(A_log)`` (``kda_small_columns`` ``a_exp``).
    """
    kernel_assert(conv_dtype in CONV_DTYPES, f"conv_dtype {conv_dtype!r} must be one of {CONV_DTYPES}")
    n_layers, width = kda_params.shape[0], kda_params.shape[2]
    per_head = 14 if conv_dtype == "fp32" else 2
    local_heads = (width - 1) // per_head
    kernel_assert(
        local_heads >= 1 and kda_small_columns(local_heads, conv_dtype)["params_width"] == width,
        f"kda_params width {width} is not a params_width of conv_dtype {conv_dtype!r}",
    )
    out = nl.ndarray(
        (n_layers, P_MAX, width + local_heads), dtype=nl.float32, buffer=nl.shared_hbm, name="kda_params_wide"
    )
    kda_a_exp_init(kda_params, out, local_heads, conv_dtype)
    return out


def kda_page_span(paired_pages: bool = False) -> int:
    """Return the cache pool pages that one KDA page id spans: 2 with ``paired_pages``, else 1.

    With ``paired_pages`` a KDA page id names the even page ``2 q`` of an aligned pair ``2 q, 2 q + 1`` of the pool, and the
    page is the ``2 * width`` elements of the pair (the pool rows are contiguous). So the KDA page of id ``2 q`` holds the
    bytes of page ``q`` of the same pool viewed as ``[P / 2, 2 * width]``: the fp32 state of a 256-row pool ``[P, 73728]``
    takes the layout of the 512-row pages ``[P / 2, 147456]``, and the MLA pages of the pool keep their 256 rows.

    Example:
        >>> kda_page_span(True)
        2
    """
    return 2 if paired_pages else 1


def kda_page_bound(n_pool_pages: int, paired_pages: bool = False) -> int:
    """Return the bound ``n_pages`` of the valid KDA page ids of a pool of ``n_pool_pages`` pages (the page counts take an
    id in ``[1, n_pages)``): the pool pages, or one fewer with ``paired_pages``, so the second page of every valid
    pair is in the pool (an id >= P - 1 is off, so id + 1 < P). With ``paired_pages`` the ids are EVEN ids only: the
    page counts treat an odd id as off (``_even_pages_only``), so it reads and writes page 0 (or the KDA null page),
    never the next page. So the valid pair bases are the even ids in ``[1 + null_page, P - 2]``.

    Example:
        >>> kda_page_bound(512, True)
        511
    """
    return n_pool_pages - 1 if paired_pages else n_pool_pages


def kda_page_offsets(local_heads: int, state_dtype: str = "bf16") -> dict:
    """Return the bf16-element offsets of the KDA states in one cache page.

    Args:
        local_heads (int): KDA heads of one rank.
        state_dtype (str): Form of the recurrent state, one of ``KDA_STATE_DTYPES``. An fp32 state starts at the
            same offset and takes two bf16 elements per value.

    Returns:
        dict: The start offsets ``conv`` and ``recurrent``, and the end ``end`` of the used part, in bf16 elements.

    Example:
        >>> kda_page_offsets(3)["recurrent"]
        3456
        >>> kda_page_offsets(3, "fp32")["end"]
        101760
    """
    kernel_assert(state_dtype in KDA_STATE_DTYPES, f"state_dtype {state_dtype!r} must be one of {KDA_STATE_DTYPES}")
    conv_size = N_CONV_PROJ * local_heads * HEAD_DIM * (CONV_TAPS - 1)
    elements = 1 if state_dtype == "bf16" else 2
    return {
        "conv": 0,
        "recurrent": conv_size,
        "end": conv_size + elements * local_heads * HEAD_DIM * HEAD_DIM,
    }


class WInRing:
    """The ``kda_w_in`` ring of one layer whose first fills of chunks ``first .. buffers - 1`` a caller issued before
    the KDA body.

    ``prefetch_w_in_ring`` builds it; the body's input projection (``_gemv_channel_major`` ``ring``) takes those buffers,
    issues the first fills of chunks ``0 .. first - 1`` itself (weight-gated, as without a ring) and then the refills.
    ``buffers`` is the ring size, ``bufs`` and ``fills`` the buffers and the first-fill DMAs of the prefetched chunks (chunk
    order), ``shape`` the shape of the ``kda_w_in`` view that the fills read. With an MXFP8 ``kda_w_in`` a chunk is one MX
    group, and ``scales`` holds the SBUF scale copies (``load_mx_scales``) that the ring loaded with its fills.
    """

    def __init__(self, shape: tuple, buffers: int, first: int, bufs: list, fills: list, scales: list = None):
        self.shape = shape
        self.buffers = buffers
        self.first = first
        self.bufs = bufs
        self.fills = fills
        self.scales = scales


def prefetch_w_in_ring(
    w: nl.NkiTensor,
    tag: str,
    buffers: int | None,
    after: tuple = (),
    wait: tuple = (),
    first: int = 0,
    w_scale: nl.NkiTensor = None,
    priority: int = _W_IN_DMA_PRIORITY,
) -> WInRing:
    """Issue the first fills of chunks ``first .. buffers - 1`` of a KDA layer's ``kda_w_in`` ring ahead of its body;
    return the ring.

    ``w`` is the layer's HBM ``kda_w_in`` [H, N]. The fills are the loads of those chunks of the input projection, in the
    buffers, queue (SP hardware DGE) and form of ``_gemv_channel_major``, so the projection reads the same bytes. Each
    fill follows the instructions ``after`` and the fill of the chunk before it in the schedule (schedule edges: the
    compiler can neither hoist a fill above them nor reorder the fills; the SP queue is in order) and waits for the
    instructions ``wait`` (a semaphore wait). They are not weight-gated
    (``tkg_weight_dma_gate``): the caller picks the place. ``first`` > 0 leaves the first chunks to the body, which issues them with
    its weight gate. Pass the ring to the body (``w_in_ring``) of the same layer. The output of the body does not change.

    ``w_scale`` is None (a BF16 ``w``), or the scales of the MXFP8 / MXFP4 ``w`` of ``_gemv_channel_major_mx``: a chunk is
    then one MX group, and the ring also loads the scale copies, before its fills. ``priority`` is the DMA QoS level of
    the fills (the projection's by default; on trn3, 3 streams them in the DMA time that the work they overlap leaves).
    ``buffers`` None is the ring of ``buffers=None``: one buffer per chunk, of its size, so the whole weight is
    prefetched (``first`` 0).
    """
    if w_scale is not None:
        return _prefetch_w_in_ring_mx(w, w_scale, tag, buffers, after, wait, first, priority)
    hidden, n_out = w.shape
    kernel_assert(hidden % P_MAX == 0, f"kda_w_in has {hidden} rows, not a multiple of {P_MAX}")
    chunk_sizes = _w_in_chunk_sizes(hidden // P_MAX)
    n_bufs = _prefetch_buffers(buffers, len(chunk_sizes), first)
    w_tiles = w.reshape((hidden // P_MAX, P_MAX, n_out))
    bufs, fills = [], []
    for chunk_idx in range(first, n_bufs):
        tiles = _W_IN_TILES_PER_DMA if buffers is not None else chunk_sizes[chunk_idx]
        buf = nl.ndarray((P_MAX, tiles, n_out), dtype=w.dtype, buffer=nl.sbuf)
        fill = _w_in_load(
            buf,
            w_tiles,
            sum(chunk_sizes[:chunk_idx]),
            chunk_sizes[chunk_idx],
            n_out,
            nisa.engine.sync,
            f"{tag}_kda_w_in_prefetch_{chunk_idx}",
            priority,
        )
        _order_prefetch(fill, fills, after, wait)
        bufs.append(buf)
        fills.append(fill)
    return WInRing(tuple(w.shape), n_bufs, first, bufs, fills)


def _prefetch_buffers(buffers: int | None, n_chunks: int, first: int) -> int:
    """Return the ring size of ``prefetch_w_in_ring`` for ``n_chunks`` chunks; check ``buffers`` and ``first``."""
    kernel_assert(
        buffers is None or (isinstance(buffers, int) and buffers >= 1),
        f"a prefetched ring needs None or >= 1 buffers, got {buffers}",
    )
    n_bufs = n_chunks if buffers is None else min(buffers, n_chunks)
    kernel_assert(
        0 <= first < n_bufs and (buffers is not None or first == 0),
        f"the first prefetched chunk {first} must be in 0 .. {n_bufs - 1} (0 for a whole-weight ring)",
    )
    return n_bufs


def _order_prefetch(load, fills: list, after: tuple, wait: tuple) -> None:
    # In chunk order on the in-order queue: after the caller's instructions and after the load before it.
    before = [f if hasattr(f, "depends_on") else nisa.NkiInstruction(f) for f in fills[-1:]]
    _schedule_after(load, list(after) + before)
    if wait:
        # The ring loads run on the Sync queue (each caller gives nisa.engine.sync).
        wait_for_completion(load, wait, engine=nisa.engine.sync)


def _prefetch_w_in_ring_mx(w, w_scale, tag: str, buffers, after: tuple, wait: tuple, first: int, priority: int):
    """``prefetch_w_in_ring`` of an MXFP8 / MXFP4 ``w``: the scale copies, then the first fills of MX groups
    ``first ..``, in the buffers, queue and form of ``_gemv_channel_major_mx``."""
    kernel_assert(is_trn3(), "the MXFP8 input projection needs Trn3")
    n_bufs = _prefetch_buffers(buffers, w.shape[0], first)
    loads = []

    def load_scales(dst, src, copy):
        instr = nisa.dma_copy(
            dst=dst,
            src=src,
            priority=dma_priority(priority),
            dge_mode=nisa.dge_mode.hwdge,
            engine=nisa.engine.sync,
            name=f"{tag}_kda_w_in_scale_prefetch_{copy}",
        )
        _order_prefetch(instr, loads, after, wait)
        loads.append(instr)

    scales = load_mx_scales(w_scale, lambda shape, copy: nl.ndarray(shape, dtype=nl.uint8, buffer=nl.sbuf), load_scales)
    bufs, fills = [], []
    for group in range(first, n_bufs):
        buf = nl.ndarray((P_MAX, w.shape[2]), dtype=w.dtype, buffer=nl.sbuf)
        fill = nisa.dma_copy(
            dst=buf,
            src=w[group],
            priority=dma_priority(priority),
            dge_mode=nisa.dge_mode.hwdge,
            engine=nisa.engine.sync,
            name=f"{tag}_kda_w_in_prefetch_{group}",
        )
        _order_prefetch(fill, loads, after, wait)
        loads.append(fill)
        bufs.append(buf)
        fills.append(fill)
    return WInRing(tuple(w.shape), n_bufs, first, bufs, fills, scales)


def _w_in_load(w_buf, w_tiles, first: int, count: int, n_out: int, engine, name: str, priority=_W_IN_DMA_PRIORITY):
    """Load K-tiles ``first .. first + count - 1`` of ``w_tiles`` [H/128, 128, N] into the ring buffer ``w_buf``."""
    return nisa.dma_copy(
        dst=w_buf[0:P_MAX, 0:count, 0:n_out],
        src=w_tiles[first : first + count].permute((1, 0, 2)),
        priority=dma_priority(priority),
        dge_mode=nisa.dge_mode.hwdge,
        engine=engine,
        name=name,
    )


W_IN_REFILL_QUEUES = ("alternate", "sync")
"""Forms of the queues of the ``kda_w_in`` ring refills (``_gemv_channel_major`` ``refill_queues``)."""


def _gemv_channel_major(
    x_hT: nl.NkiTensor,
    w: nl.NkiTensor,
    tag: str,
    buffers: int | None = _W_IN_BUFFERS,
    after_first_fills=None,
    w_scale: nl.NkiTensor = None,
    ring: WInRing = None,
    preloaded: tuple = (),
    marks: dict = None,
    fills_after: tuple = (),
    lanes_view: bool = True,
    quant_hook=None,
    psum_bufs: int = 1,
    refill_queues: str = "alternate",
    late_blocks: tuple = (),
    late: dict = None,
    x_stationary: bool = False,
) -> nl.NkiTensor:
    """Return ``x @ w`` for a token-major hT ``x`` ``[128, T, H/128]`` and an HBM ``w`` ``[H, N]`` in the channel-major layout.

    The result is an fp32 PSUM tile ``[128, ceil(N/128), T]`` (an fp32 SBUF tile on trn2, ``_gemv_chunk_groups``).
    Element ``(p, c, t)`` is output column ``c*128 + p`` of token ``t``. Each 128x128 weight tile is the stationary operand and the K-tile
    of ``x`` is the moving operand, so a matmul costs about one stationary load and the result needs
    no transpose. The moving operand has T columns, so the TensorE time does not grow with T.

    The weight streams through a ring of ``buffers`` SBUF buffers. A load into a buffer follows
    the matmuls of the previous chunk in that buffer in program order, so the ring bounds the SBUF
    use. The first fill of the ring uses the SP hardware DGE queue, and the later loads alternate
    between the ACT and the SP queues. The last chunks are smaller, so less TensorE work remains
    after the last load. With ``buffers=None`` each DMA gets its own buffer of its size: every load
    is a first fill, so the whole weight streams from the gate on and the matmuls do not wait for
    the refills of a ring.

    Keep each matmul output in one PSUM bank. On Trn3 a matmul can span several banks, but on
    hardware its overwrite clears only the first bank, and its accumulation is wrong after a
    per-bank overwrite.

    ``after_first_fills`` is None (the default), or a function that the GEMV calls once with the first-fill DMAs of
    the ring, after it traces them and before the first matmul. The function traces its own SP-queue DMAs there and
    returns them. Every refill on the SP queue then follows the last of them in the schedule, so the scheduler does
    not put a refill in front of them.

    With ``w_scale``, ``w`` is the MXFP8 (uint8) or MXFP4 (uint16) weight in the hT_quad layout of ``tkg_mx_quant``
    (``_gemv_channel_major_mx``).

    ``ring`` is None (the default), or the ring of ``prefetch_w_in_ring`` with the first fills of its chunks
    ``ring.first ..`` already issued: the GEMV reads those buffers, issues the first fills of the chunks before them (with
    the weight gate) and the refills (``buffers`` must be the ring's count). Not with ``after_first_fills``. With
    ``w_scale`` the ring is that of the MX weight (``prefetch_w_in_ring`` ``w_scale``).
    With ``fills_after``, only the first fills that the GEMV issues follow those instructions; the prefetched fills
    keep their place.

    ``preloaded`` is empty (the default), or the SBUF tiles ``[128, 4, N]`` that hold the first ``len(preloaded)``
    chunks of ``w`` (``kda_w_in_head_loads``, loaded earlier by the caller). The matmuls of those chunks read them, and
    the ring covers the later chunks only: its first fills are the chunks after the preloaded ones. The matmuls and
    their order do not change, so the output does not change. Not with ``ring`` or ``w_scale``.

    ``marks`` is None (the default), or a dict that the GEMV fills for its caller: ``fills``, the weight DMAs of the
    chunks in chunk order (the prefetched first fills of ``ring`` included, the ``preloaded`` chunks not), and ``end``,
    its last instruction (the last fp32 accumulate on trn2, the last matmul on trn3; with ``w_scale``, the MX group
    loads in group order and the last MX matmul). The program does not change.
    ``fills_after`` holds instructions that every first fill follows in the schedule (schedule edges), so a small DMA
    that the caller traces before the GEMV is not queued behind the ring. () (the default) adds no edge.

    ``lanes_view`` (MXFP8 only): the quantize of ``x`` reads its lanes in place (``quantize_hT_mx`` ``lanes_view``);
    False (not the default) makes the VectorE lane copies. The same bits.

    ``quant_hook`` is None (the default), or a function that the MX GEMV (``w_scale``) calls once with the quantize_mx
    instruction of ``x``, right after it traces it and before the first matmul. Not with the BF16 weight.
    ``psum_bufs`` (trn2, the BF16 weight): the PSUM tiles that the chunk sums of ``_gemv_chunk_groups`` take in turn. 1
    (the default) keeps the program: each chunk takes a new tile, which reuses the bytes of the chunk before it, so the
    next chunk's matmuls wait for the VectorE add of the chunk before. With 2 or more, a chunk's matmuls wait only for
    the add of the chunk ``psum_bufs`` before. The same matmuls and adds, so the bits do not change.
    ``refill_queues`` (the BF16 weight): the queues of the ring's refills (the loads after its first fills):
    ``alternate`` (the default) keeps the program, the ACT and the SP queues in turn; ``sync`` issues them all on the
    SP queue, so no refill waits behind the ScalarE's compute in the ACT queue. The same loads, so the bits do not
    change.
    ``late_blocks`` (MXFP8 only; () keeps the program) and ``late``: as in ``_gemv_channel_major_mx``.
    ``x_stationary`` (MXFP8 only; False keeps the program): as in ``_gemv_channel_major_mx`` (an ``XStationaryPsum``).
    """
    kernel_assert(quant_hook is None or w_scale is not None, "quant_hook is for the MX weight (w_scale)")
    kernel_assert(not late_blocks or w_scale is not None, "late_blocks is for the MX weight (w_scale)")
    kernel_assert(not x_stationary or w_scale is not None, "x_stationary is for the MX weight (w_scale)")
    kernel_assert(
        refill_queues in W_IN_REFILL_QUEUES and (refill_queues == "alternate" or w_scale is None),
        f"refill_queues {refill_queues!r} must be one of {W_IN_REFILL_QUEUES} ('alternate' with an MX weight)",
    )
    kernel_assert(
        psum_bufs == 1 or (psum_bufs >= 2 and not is_trn3() and w_scale is None),
        f"psum_bufs {psum_bufs!r}: 1, or 2 or more for the trn2 BF16 weight (the chunk sums of _gemv_chunk_groups)",
    )
    if w_scale is not None:
        kernel_assert(not preloaded, "preloaded chunks take the BF16 weight, not an MX one (w_scale)")
        return _gemv_channel_major_mx(
            x_hT,
            w,
            w_scale,
            tag,
            buffers,
            after_first_fills,
            ring=ring,
            **({"marks": marks} if marks is not None else {}),
            lanes_view=lanes_view,
            **({"quant_hook": quant_hook} if quant_hook is not None else {}),
            **({"late_blocks": tuple(late_blocks), "late": late} if late_blocks else {}),
            **({"x_stationary": True} if x_stationary else {}),
        )
    n_tokens = x_hT.shape[1]
    hidden_tiles = x_hT.shape[2]
    n_out = w.shape[1]
    n_blocks = div_ceil(n_out, P_MAX)
    w_tiles = w.reshape((hidden_tiles, P_MAX, n_out))
    chunk_sizes = _w_in_chunk_sizes(hidden_tiles)
    chunk_starts = [sum(chunk_sizes[:chunk_idx]) for chunk_idx in range(len(chunk_sizes))]
    n_chunks = len(chunk_sizes)
    n_head = len(preloaded)
    kernel_assert(
        n_head == 0 or ring is None,
        "preloaded chunks (a kda_w_in head) and a prefetched ring are two forms of the early kda_w_in loads: use one",
    )
    kernel_assert(
        n_head == 0
        or (is_trn3() and n_head < n_chunks and all(size == _W_IN_TILES_PER_DMA for size in chunk_sizes[:n_head])),
        f"preloaded takes whole {_W_IN_TILES_PER_DMA}-tile chunks before the last chunk, on trn3; got {n_head} of {chunk_sizes}",
    )
    for head_tile in preloaded:
        kernel_assert(
            tuple(head_tile.shape) == (P_MAX, _W_IN_TILES_PER_DMA, n_out) and head_tile.dtype == w.dtype,
            f"a preloaded chunk must be [{P_MAX}, {_W_IN_TILES_PER_DMA}, {n_out}] {w.dtype}, got {head_tile.shape}",
        )
    # The ring covers chunks n_head .. n_chunks - 1; ring index r = chunk_idx - n_head.
    n_ring = n_chunks - n_head
    n_bufs = n_ring if buffers is None else min(buffers, n_ring)
    if ring is not None:
        kernel_assert(
            after_first_fills is None
            and buffers is not None
            and n_bufs == ring.buffers
            and ring.shape == tuple(w.shape)
            and ring.scales is None,
            f"the prefetched ring ({ring.buffers} buffers of {ring.shape}) does not match this projection "
            f"({n_bufs} buffers of {tuple(w.shape)}), or after_first_fills is set",
        )
        # The prefetched chunks take the ring's buffers; the body allocates the buffers of the chunks before them.
        w_bufs = [
            nl.ndarray((P_MAX, _W_IN_TILES_PER_DMA, n_out), dtype=w.dtype, buffer=nl.sbuf) for _ in range(ring.first)
        ]
        w_bufs += list(ring.bufs)
    else:
        buf_tiles = chunk_sizes[n_head:] if buffers is None else [_W_IN_TILES_PER_DMA] * n_bufs
        w_bufs = [nl.ndarray((P_MAX, tiles, n_out), dtype=w.dtype, buffer=nl.sbuf) for tiles in buf_tiles]
    dma_engines = (nisa.engine.sync,) if refill_queues == "sync" else (nisa.engine.scalar, nisa.engine.sync)

    def load_engine(chunk_idx):
        # The first load of each ring buffer is a gated weight DMA (see tkg_weight_dma_gate). A gate holds the
        # in-order queue of its trigger engine, so these loads use the SP queue. The later loads
        # alternate between the ACT and the SP queues.
        ring_idx = chunk_idx - n_head
        return nisa.engine.sync if ring_idx < n_bufs else dma_engines[(ring_idx - n_bufs) % len(dma_engines)]

    chunk_fills = {}  # chunk index -> its weight DMA (marks)

    def load_chunk(chunk_idx):
        load = _w_in_load(
            w_bufs[(chunk_idx - n_head) % n_bufs],
            w_tiles,
            chunk_starts[chunk_idx],
            chunk_sizes[chunk_idx],
            n_out,
            load_engine(chunk_idx),
            f"{tag}_kda_w_in_load_{chunk_idx}",
        )
        chunk_fills[chunk_idx] = load
        return load

    def mark(end):
        if marks is not None:
            marks["fills"] = [chunk_fills[chunk_idx] for chunk_idx in sorted(chunk_fills)]
            marks["end"] = end

    # One buffer per DMA: the loads after the first _W_IN_BUFFERS wait for the last collective of the previous
    # body (the late gate), so only the ring's bytes stream beside that collective.
    first_fills = []
    for ring_idx in range(n_bufs):
        if ring is not None and ring_idx >= ring.first:
            first_fills.append(ring.fills[ring_idx - ring.first])
            chunk_fills[n_head + ring_idx] = first_fills[-1]
            continue
        fill = gate_weight_dma(load_chunk(n_head + ring_idx), late=buffers is None and ring_idx >= _W_IN_BUFFERS)
        if fills_after:
            fill.depends_on(*fills_after, kind=nisa.dependency.schedule)
        first_fills.append(fill)
    # The caller's DMAs after the first fills (after_first_fills): the first SP refill follows the last of them.
    refill_after = []
    if after_first_fills is not None:
        refill_after = [instr for instr in after_first_fills(first_fills)[-1:] if instr is not None]
    if not is_trn3():
        out_sb, end = _gemv_chunk_groups(
            x_hT, w_bufs, chunk_sizes, chunk_starts, n_bufs, n_out, load_chunk, psum_bufs=psum_bufs
        )
        mark(end)
        return out_sb
    out_psum = nl.ndarray((P_MAX, n_blocks, n_tokens), dtype=nl.float32, buffer=nl.psum)
    end = None
    for chunk_idx in range(n_chunks):
        first = chunk_starts[chunk_idx]
        w_buf = preloaded[chunk_idx] if chunk_idx < n_head else w_bufs[(chunk_idx - n_head) % n_bufs]
        for tile_idx in range(chunk_sizes[chunk_idx]):
            hidden_tile = first + tile_idx
            for block_idx in range(n_blocks):
                col_start = block_idx * P_MAX
                block_cols = min(P_MAX, n_out - col_start)
                end = nisa.nc_matmul(
                    dst=out_psum[0:block_cols, block_idx, 0:n_tokens],
                    stationary=w_buf[0:P_MAX, tile_idx, col_start : col_start + block_cols],
                    moving=x_hT[0:P_MAX, 0:n_tokens, hidden_tile],
                    accumulate=hidden_tile > 0,
                )
        if chunk_idx >= n_head and chunk_idx + n_bufs < n_chunks:
            refill = load_chunk(chunk_idx + n_bufs)
            # Every SP refill, not only the first: the scheduler can otherwise move a later SP refill ahead of the caller's
            # DMAs, and the SP queue then holds them behind the WAR wait of that refill (MEASURED in the compiled order).
            if refill_after and load_engine(chunk_idx + n_bufs) == nisa.engine.sync:
                _schedule_after(refill, refill_after)
    mark(end)
    return out_psum


def _gemv_channel_major_mx(
    x_hT,
    w,
    w_scale,
    tag: str,
    buffers: int | None,
    after_first_fills=None,
    ring: WInRing = None,
    marks: dict = None,
    lanes_view: bool = True,
    quant_hook=None,
    late_blocks: tuple = (),
    late: dict = None,
    x_stationary: bool = False,
) -> nl.NkiTensor:
    """Return ``x @ w`` of ``_gemv_channel_major`` from an MXFP8 weight and the MXFP8 form of ``x_hT``.

    ``w`` is ``[G, 128, 4 N]`` float8_e4m3fn or uint8 (MXFP8), or ``[G, 128, N]`` uint16 (MXFP4, half the bytes), and
    ``w_scale`` ``[128, S, N]`` uint8 (``tkg_mx_quant``). The activation is MXFP8 (``quantize_hT_mx``). One DMA
    loads one MX group (512 rows, the 4 hidden tiles of a BF16 chunk), and each 128-column weight tile of the group
    is the stationary operand of one nc_matmul_mx against the T token columns. The ring, the gates and the queues
    are those of the BF16 form. The result has the fp32 PSUM layout of ``_gemv_channel_major``. ``ring``: as in
    ``_gemv_channel_major``, the ring of ``prefetch_w_in_ring`` with ``w_scale``, which also loaded the scales.
    ``marks``: as in ``_gemv_channel_major``, the group loads (first fills, then refills) and the last matmul.
    ``quant_hook``: as in ``_gemv_channel_major``.
    ``late_blocks`` (() keeps the program): 128-column output blocks whose matmuls on the groups that stay in SBUF to the
    end of the call (the last ``buffers`` groups, which no refill takes) run later, when the caller calls
    ``late["run"]()``. Their matmuls on the refilled groups still run in the group loop, as without ``late_blocks``. The
    late blocks take their own fp32 PSUM tile ``[128, len(late_blocks), T]`` (``late["psum"]``, also the return value of
    ``late["run"]``), and ``late["end"]`` gets the last late matmul. Each block still adds its groups in group order, so
    the values do not change; the result tile holds no values at the late blocks. ``late`` is the dict that the GEMV
    fills.
    ``x_stationary`` (an MXFP8 weight, no ``late_blocks``; False keeps the program): the quantized tokens of each MX group
    are the stationary operand (at most 128 rows, padded to even) and the weight columns of the group move, in chunks of
    at most ``_MX_MOVING_MAX`` columns: one nc_matmul_mx a group and chunk in place of one a group and 128-column block,
    so TensorE loads one small stationary a group instead of a 128x128 tile a block (the token-stationary pattern of the
    MoE MXFP8 in-projection). Each column still adds its groups in group order. The result is an ``XStationaryPsum``:
    the token-major fp32 sums, one PSUM tile a chunk (``in_proj_copies_x_stationary`` gives the served bf16 tiles).
    """
    kernel_assert(is_trn3(), "the MXFP8 input projection needs Trn3")
    n_tokens, hidden_tiles = x_hT.shape[1], x_hT.shape[2]
    n_groups = hidden_tiles // MX_LANES
    fp4, n_out, w_x4 = mx_weight_format(w)
    data_shape, scale_shape = mx_hT_shapes(hidden_tiles * P_MAX, n_out, fp4)
    kernel_assert(
        tuple(w.shape) == data_shape and tuple(w_scale.shape) == scale_shape,
        f"MXFP8 weight {w.shape} / scales {w_scale.shape} must be {data_shape} / {scale_shape}",
    )
    n_blocks = div_ceil(n_out, P_MAX)
    n_bufs = n_groups if buffers is None else min(buffers, n_groups)
    if ring is not None:
        kernel_assert(
            after_first_fills is None
            and n_bufs == ring.buffers
            and ring.shape == tuple(w.shape)
            and ring.scales is not None,
            f"the prefetched MX ring ({ring.buffers} buffers of {ring.shape}) does not match this projection "
            f"({n_bufs} buffers of {tuple(w.shape)}), or after_first_fills is set",
        )
    n_own = n_bufs if ring is None else ring.first
    w_bufs = [nl.ndarray((P_MAX, w.shape[2]), dtype=w.dtype, buffer=nl.sbuf) for _ in range(n_own)]
    if ring is not None:
        w_bufs += list(ring.bufs)
    dma_engines = (nisa.engine.scalar, nisa.engine.sync)

    def load_engine(group):
        return nisa.engine.sync if group < n_bufs else dma_engines[(group - n_bufs) % len(dma_engines)]

    def load_group(group):
        return nisa.dma_copy(
            dst=w_bufs[group % n_bufs],
            src=w[group],
            priority=dma_priority(_W_IN_DMA_PRIORITY),
            dge_mode=nisa.dge_mode.hwdge,
            engine=load_engine(group),
            name=f"{tag}_kda_w_in_load_{group}",
        )

    def load_scales(dst, src, copy):
        gate_weight_dma(
            nisa.dma_copy(
                dst=dst,
                src=src,
                priority=dma_priority(_W_IN_DMA_PRIORITY),
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.sync,
                name=f"{tag}_kda_w_in_scale_load_{copy}",
            )
        )

    if ring is None:
        scale_sb = load_mx_scales(
            w_scale, lambda shape, copy: nl.ndarray(shape, dtype=nl.uint8, buffer=nl.sbuf), load_scales
        )
    else:
        scale_sb = ring.scales
    first_fills = []
    for group in range(n_bufs):
        if ring is not None and group >= ring.first:
            first_fills.append(ring.fills[group - ring.first])
            continue
        first_fills.append(gate_weight_dma(load_group(group), late=buffers is None and group >= _W_IN_BUFFERS))
    refill_after = []
    if after_first_fills is not None:
        refill_after = [instr for instr in after_first_fills(first_fills)[-1:] if instr is not None]
    elif ring is not None:
        # The scheduler can otherwise put a TensorE-gated SP refill ahead of the ring's later fills, and the SP queue
        # then holds the groups the next matmuls need behind the WAR wait of that refill (MEASURED in the compiled order).
        refill_after = [fill for fill in ring.fills[-1:] if fill is not None]
    # An even token stride keeps each group's moving columns on an 8-byte boundary (MatmulMX refuses odd x4 starts).
    quantized = []
    # quantize_mx reads the lanes of x in place (no VectorE lane copies ahead of the first matmul; the same bits).
    # lanes_view False (not the default) makes the lane copies.
    x_mx, x_scale = quantize_hT_mx(
        x_hT, f"{tag}_kda_w_in", n_tokens + n_tokens % 2, lanes_view=lanes_view, done=quantized
    )
    if quant_hook is not None:
        quant_hook(quantized[-1])
    padded = n_tokens + n_tokens % 2
    kernel_assert(
        not x_stationary or (not fp4 and not late_blocks and padded <= P_MAX),
        f"x_stationary needs an MXFP8 weight, no late_blocks and at most {P_MAX} rows, got fp4={fp4}, "
        f"late_blocks={tuple(late_blocks)}, {n_tokens} rows",
    )
    x_chunks = [(c0, min(_MX_MOVING_MAX, n_out - c0)) for c0 in range(0, n_out, _MX_MOVING_MAX)] if x_stationary else []
    x_psums = [nl.ndarray((padded, width), dtype=nl.float32, buffer=nl.psum) for _, width in x_chunks]
    out_psum = None if x_stationary else nl.ndarray((P_MAX, n_blocks, n_tokens), dtype=nl.float32, buffer=nl.psum)
    late_blocks = tuple(late_blocks)
    kernel_assert(
        not late_blocks
        or (
            late is not None
            and len(set(late_blocks)) == len(late_blocks)
            and all(0 <= block < n_blocks for block in late_blocks)
        ),
        f"late_blocks {late_blocks} must be distinct blocks of the {n_blocks}, with a late dict",
    )
    late_index = {block: index for index, block in enumerate(late_blocks)}
    late_psum = (
        nl.ndarray((P_MAX, len(late_blocks), n_tokens), dtype=nl.float32, buffer=nl.psum) if late_blocks else None
    )
    # The groups from kept_from on are not refilled in this call: the late blocks' matmuls on them can wait.
    kept_from = max(0, n_groups - n_bufs)

    def block_matmul(group, block_idx):
        w_mx = w_bufs[group % n_bufs].view(w_x4)
        col_start = block_idx * P_MAX
        block_cols = min(P_MAX, n_out - col_start)
        if block_idx in late_index:
            dst = late_psum[0:block_cols, late_index[block_idx], 0:n_tokens]
        else:
            dst = out_psum[0:block_cols, block_idx, 0:n_tokens]
        return nisa.nc_matmul_mx(
            dst=dst,
            stationary=w_mx[0:P_MAX, col_start : col_start + block_cols],
            moving=x_mx[0:P_MAX, group, 0:n_tokens],
            stationary_scale=mx_scale_view(scale_sb, group, col_start, block_cols),
            moving_scale=x_scale[0:P_MAX, group, 0:n_tokens],
            accumulate=group > 0,
        )

    refills, end = [], None
    for group in range(n_groups):
        for (c0, width), x_psum in zip(x_chunks, x_psums):
            w_mx = w_bufs[group % n_bufs].view(w_x4)
            end = nisa.nc_matmul_mx(
                dst=x_psum[0:padded, 0:width],
                stationary=x_mx[0:P_MAX, group, 0:padded],
                moving=w_mx[0:P_MAX, c0 : c0 + width],
                stationary_scale=x_scale[0:P_MAX, group, 0:padded],
                moving_scale=mx_scale_view(scale_sb, group, c0, width),
                accumulate=group > 0,
            )
        for block_idx in range(0 if x_stationary else n_blocks):
            if block_idx in late_index and group >= kept_from:
                continue
            end = block_matmul(group, block_idx)
        if group + n_bufs < n_groups:
            refill = load_group(group + n_bufs)
            refills.append(refill)
            if refill_after and load_engine(group + n_bufs) == nisa.engine.sync:
                _schedule_after(refill, refill_after)
    if marks is not None:
        marks["fills"] = [*first_fills, *refills]
        marks["end"] = end
    if late_blocks:

        def run_late():
            last = None
            for group in range(kept_from, n_groups):
                for block_idx in late_blocks:
                    last = block_matmul(group, block_idx)
            late["end"] = last
            return late_psum

        late.update(psum=late_psum, run=run_late)
    if x_stationary:
        return XStationaryPsum(x_chunks, x_psums, n_tokens)
    return out_psum


class HeadMajorGemv:
    """The MXFP8 input projection of a head-major ``kda_w_in`` in passes (``kda_verify_attention`` ``head_pipeline``
    ``heads``; the layout of ``kda_in_columns_head_major``).

    Each weight group loads as two DMA parts, ``A`` (f_a, b and the q / k / v of heads 0 .. Hl - 2) and ``BC`` (the
    q / k / v of the last head and the g of all heads), each one contiguous run of every partition (3,616 B and 3,072 B
    at Hl = 3). The parts take the slots of one ring in the order A_0 .. A_{G-1}, BC_0 .. BC_{G-1}: part i takes slot
    i % n_slots, so the A parts are all in SBUF and BC_g reuses the slot of A_g after the A matmuls of group g. The A
    parts load on the SP queue first, then the BC parts. ``run("A")`` traces the matmuls of the A blocks (all groups),
    ``run("B")`` those of the last head's q / k / v, ``run("C", groups)`` those of the g blocks on ``groups`` (in group
    order across the calls). Each block adds its groups in group order, with the weights and scales of the original
    layout's block, so its PSUM values are those of ``_gemv_channel_major_mx``.
    """

    def __init__(
        self,
        x_hT,
        w,
        w_scale,
        tag: str,
        local_heads: int,
        n_slots: int,
        marks: dict = None,
        quant_hook=None,
        fills_after: tuple = (),
    ):
        kernel_assert(is_trn3(), "the MXFP8 input projection needs Trn3")
        n_tokens, hidden_tiles = x_hT.shape[1], x_hT.shape[2]
        self.n_groups = hidden_tiles // MX_LANES
        fp4, n_out, w_x4 = mx_weight_format(w)
        kernel_assert(not fp4, "the head-major input projection takes the MXFP8 kda_w_in")
        layout = kda_in_columns_head_major(local_heads)
        kernel_assert(
            n_out == layout["width"], f"head-major kda_w_in has {n_out} columns, the layout {layout['width']}"
        )
        kernel_assert(
            n_slots >= self.n_groups,
            f"the head-major input projection needs {self.n_groups} part slots (all A parts in SBUF), got {n_slots}",
        )
        self.heads, self.n_tokens, self.w_x4, self.n_out, self.layout = local_heads, n_tokens, w_x4, n_out, layout
        part_a = layout["part_a"]
        self.parts = {"A": (0, part_a), "BC": (part_a, n_out)}
        slot_cols = max(part_a, n_out - part_a)
        self.slots = [nl.ndarray((P_MAX, MX_LANES * slot_cols), dtype=w.dtype, buffer=nl.sbuf) for _ in range(n_slots)]
        self.n_slots = n_slots

        def load_scales(dst, src, copy):
            gate_weight_dma(
                nisa.dma_copy(
                    dst=dst,
                    src=src,
                    priority=dma_priority(_W_IN_DMA_PRIORITY),
                    dge_mode=nisa.dge_mode.hwdge,
                    engine=nisa.engine.sync,
                    name=f"{tag}_kda_w_in_scale_load_{copy}",
                )
            )

        self.scale_sb = load_mx_scales(
            w_scale, lambda shape, copy: nl.ndarray(shape, dtype=nl.uint8, buffer=nl.sbuf), load_scales
        )
        self.w, self.tag = w, tag
        self.part_list = [("A", g) for g in range(self.n_groups)] + [("BC", g) for g in range(self.n_groups)]
        self.fills = []
        # A part takes its slot after the matmuls of the part before it in that slot: the first n_slots parts load
        # now, and part i >= n_slots loads after the matmuls of part i - n_slots (an A part, since n_slots >= G), in
        # pass A, so its DMA follows those reads in program order.
        self.refill_after = {index - n_slots: index for index in range(n_slots, len(self.part_list))}
        for index in range(min(n_slots, len(self.part_list))):
            fill = self._load(index)
            # The first fills follow fills_after in the schedule (as the first fills of _gemv_channel_major).
            _schedule_after(fill, fills_after)
            self.fills.append(gate_weight_dma(fill))
        fills = self.fills
        self.slot_of = {("A", g): g % n_slots for g in range(self.n_groups)}
        self.slot_of.update({("BC", g): (self.n_groups + g) % n_slots for g in range(self.n_groups)})
        quantized = []
        self.x_mx, self.x_scale = quantize_hT_mx(
            x_hT, f"{tag}_kda_w_in", n_tokens + n_tokens % 2, lanes_view=True, done=quantized
        )
        if quant_hook is not None:
            quant_hook(quantized[-1])
        heads = local_heads
        # Pass A blocks: q / k / v of heads 0 .. Hl - 2 (kind-major, as the conv tile of the stage), then f_a and b.
        self.blocks = {
            "A": [kind * heads + head for kind in range(3) for head in range(heads - 1)] + [4 * heads, 4 * heads + 1],
            "B": [kind * heads + heads - 1 for kind in range(3)],
            "C": [3 * heads + head for head in range(heads)],
        }
        self.psum_a = nl.ndarray((P_MAX, len(self.blocks["A"]), n_tokens), dtype=nl.float32, buffer=nl.psum)
        # B and C share one PSUM tensor (one bank): C's groups start after the copy of B.
        self.psum_bc = nl.ndarray((P_MAX, 3 + heads, n_tokens), dtype=nl.float32, buffer=nl.psum)
        self.end = None
        if marks is not None:
            marks["fills"] = fills

    def _load(self, index: int):
        """Issue the DMA of part ``index`` of ``part_list`` into its slot; return it."""
        name, group = self.part_list[index]
        c0, c1 = self.parts[name]
        return nisa.dma_copy(
            dst=self.slots[index % self.n_slots][0:P_MAX, 0 : MX_LANES * (c1 - c0)],
            src=self.w[group, 0:P_MAX, MX_LANES * c0 : MX_LANES * c1],
            priority=dma_priority(_W_IN_DMA_PRIORITY),
            dge_mode=nisa.dge_mode.hwdge,
            engine=nisa.engine.sync,
            name=f"{self.tag}_kda_w_in_{name}_load_{group}",
        )

    def _block(self, block: int):
        """Return the part, the column in the part, the start column and the width of output block ``block``."""
        start, width = self.layout["starts"][block], self.layout["widths"][block]
        name = "A" if start < self.parts["A"][1] else "BC"
        return name, start - self.parts[name][0], start, width

    def run(self, name: str, groups=None):
        """Trace the matmuls of pass ``name`` (``A``, ``B`` or ``C``) on ``groups`` (None: all); return its PSUM view."""
        blocks = self.blocks[name]
        if name == "A":
            dst_tile, base = self.psum_a, 0
        else:
            dst_tile, base = self.psum_bc, 0 if name == "B" else 3
        for group in range(self.n_groups) if groups is None else groups:
            for index, block in enumerate(blocks):
                part, col, start, width = self._block(block)
                w_mx = self.slots[self.slot_of[(part, group)]].view(self.w_x4)
                self.end = nisa.nc_matmul_mx(
                    dst=dst_tile[0:width, base + index, 0 : self.n_tokens],
                    stationary=w_mx[0:P_MAX, col : col + width],
                    moving=self.x_mx[0:P_MAX, group, 0 : self.n_tokens],
                    stationary_scale=mx_scale_view(self.scale_sb, group, start, width),
                    moving_scale=self.x_scale[0:P_MAX, group, 0 : self.n_tokens],
                    accumulate=group > 0,
                )
            if name == "A" and group in self.refill_after:
                # The A part of this group is read: the part that takes its slot loads now.
                self.fills.append(self._load(self.refill_after.pop(group)))
        return dst_tile[0:P_MAX, base : base + len(blocks), 0 : self.n_tokens]


def in_proj_copies_x_stationary(proj: XStationaryPsum, cm, b_logits, first: int, n_blocks: int, b_rows: int):
    """Copy the token-stationary in-projection sums ``proj`` (``XStationaryPsum``) of rows ``first .. first + count`` into
    the bf16 channel-major ``cm`` (its first ``n_blocks`` 128-column blocks) and the ``b_rows`` b-logit rows
    ``b_logits``, with the bits of the served copies.

    The fp32 -> bf16 roundings run on the engines of the served copies (VectorE for the block columns, ScalarE for the
    b-logit columns) into a token-major bf16 tile; TensorE transposes each 128-column block (and the b-logit columns)
    back to the channel-major layout, and UINT16 ScalarE copies move them into ``cm`` and ``b_logits`` (a float-typed
    ScalarE copy writes +0 for -0 on trn3; the UINT16 copy keeps every bit)."""
    bf16, u16 = nl.bfloat16, nl.uint16
    count = proj.count
    n_cm = n_blocks * P_MAX
    tok = nl.ndarray((count, n_cm + b_rows), dtype=bf16, buffer=nl.sbuf)
    for (c0, width), psum in zip(proj.chunks, proj.psums):
        c1 = c0 + width
        if c0 < n_cm:
            a1 = min(c1, n_cm)
            nisa.tensor_copy(dst=tok[0:count, c0:a1], src=psum[0:count, 0 : a1 - c0], engine=nisa.engine.vector)
        if c1 > n_cm:
            b0 = max(c0, n_cm)
            nisa.tensor_copy(dst=tok[0:count, b0:c1], src=psum[0:count, b0 - c0 : width], engine=nisa.engine.scalar)
    blocks = n_blocks + (1 if b_rows else 0)
    cols_psum = nl.ndarray((P_MAX, blocks, count), dtype=bf16, buffer=nl.psum)
    for block in range(blocks):
        cols = P_MAX if block < n_blocks else b_rows
        nisa.nc_transpose(
            dst=cols_psum[0:cols, block, 0:count],
            data=tok[0:count, block * P_MAX : block * P_MAX + cols],
            engine=nisa.engine.tensor,
        )
    nisa.tensor_copy(
        dst=cm.view(u16)[0:P_MAX, first : first + count, 0:n_blocks],
        src=cols_psum.view(u16)[0:P_MAX, 0:n_blocks, 0:count].permute((0, 2, 1)),
        engine=nisa.engine.scalar,
    )
    if b_rows:
        nisa.tensor_copy(
            dst=b_logits.view(u16)[0:b_rows, first : first + count],
            src=cols_psum.view(u16)[0:b_rows, n_blocks, 0:count],
            engine=nisa.engine.scalar,
        )


def _gemv_chunk_groups(
    x_hT, w_bufs, chunk_sizes, chunk_starts, n_bufs: int, n_out: int, load_chunk, psum_bufs: int = 1
):
    """Return ``x @ w`` of ``_gemv_channel_major`` as an fp32 SBUF tile ``[128, ceil(N/128), T]`` (trn2), and the last
    VectorE accumulate.

    On trn2 the start matmul of an accumulation group resets the accumulate state of its PSUM zone, so a group that
    is still open in the zone loses its partial sum. The column blocks therefore accumulate one chunk of K-tiles at
    a time, one block after the other, and VectorE adds the chunk sums in fp32 SBUF. ``psum_bufs``: 1 gives each
    chunk a new PSUM tile; more takes that many tiles in turn (``_gemv_channel_major``). Each tile is its own PSUM
    tensor, so its groups stay in its own bank.
    """
    n_tokens = x_hT.shape[1]
    n_blocks = div_ceil(n_out, P_MAX)
    full_blocks = n_out // P_MAX
    tail_rows = n_out - full_blocks * P_MAX
    n_chunks = len(chunk_sizes)
    out_sb = nl.ndarray((P_MAX, n_blocks, n_tokens), dtype=nl.float32, buffer=nl.sbuf)
    turns = [
        nl.ndarray((P_MAX, n_blocks, n_tokens), dtype=nl.float32, buffer=nl.psum)
        for _ in range(psum_bufs if psum_bufs > 1 else 0)
    ]
    last = None
    for chunk_idx in range(n_chunks):
        first = chunk_starts[chunk_idx]
        w_buf = w_bufs[chunk_idx % n_bufs]
        if turns:
            chunk_psum = turns[chunk_idx % len(turns)]
        else:
            chunk_psum = nl.ndarray((P_MAX, n_blocks, n_tokens), dtype=nl.float32, buffer=nl.psum)
        for block_idx in range(n_blocks):
            col_start = block_idx * P_MAX
            block_cols = min(P_MAX, n_out - col_start)
            for tile_idx in range(chunk_sizes[chunk_idx]):
                nisa.nc_matmul(
                    dst=chunk_psum[0:block_cols, block_idx, 0:n_tokens],
                    stationary=w_buf[0:P_MAX, tile_idx, col_start : col_start + block_cols],
                    moving=x_hT[0:P_MAX, 0:n_tokens, first + tile_idx],
                    accumulate=tile_idx > 0,
                )
        for rows, blocks in ((P_MAX, slice(0, full_blocks)), (tail_rows, slice(full_blocks, n_blocks))):
            if rows == 0 or blocks.start == blocks.stop:
                continue
            acc = out_sb[0:rows, blocks, 0:n_tokens]
            part = chunk_psum[0:rows, blocks, 0:n_tokens]
            if chunk_idx == 0:
                last = nisa.tensor_copy(dst=acc, src=part, engine=nisa.engine.vector)
            else:
                last = nisa.tensor_tensor(dst=acc, data1=acc, data2=part, op=nl.add, engine=nisa.engine.vector)
        if chunk_idx + n_bufs < n_chunks:
            load_chunk(chunk_idx + n_bufs)
    return out_sb, last


def _schedule_after(instr, producers) -> None:
    """Put ``instr`` (a raw or wrapped instruction) after the instructions ``producers`` in the schedule."""
    producers = [producer for producer in producers if producer is not None]
    if instr is None or not producers:
        return
    handle = instr if hasattr(instr, "depends_on") else nisa.NkiInstruction(instr)
    handle.depends_on(*producers, kind=nisa.dependency.schedule)


def _runtime_after(instr, producers, engine=None) -> None:
    """Make ``instr`` (a raw or wrapped instruction) wait at runtime for each instruction of ``producers``.

    ``engine``: None, or the trigger engine of ``instr``. With an engine, a gate collective whose result load runs on
    that engine gets a schedule edge on the load in place of a runtime edge (``tkg_weight_dma_gate.wait_for_completion``).
    """
    if engine is not None:
        if instr is not None and any(producer is not None for producer in producers):
            wait_for_completion(instr, producers, engine=engine)
        return
    producers = [
        producer if hasattr(producer, "depends_on") else nisa.NkiInstruction(producer) for producer in producers
    ]
    if instr is None or not producers:
        return
    handle = instr if hasattr(instr, "depends_on") else nisa.NkiInstruction(instr)
    handle.depends_on(*producers, kind=nisa.dependency.runtime)


def after_w_in_loads(instr, marks: dict) -> None:
    """Put the DMA ``instr`` after the last two ``kda_w_in`` loads of ``marks`` (``_gemv_channel_major``; one load on
    each of the two queues) in the schedule, and make it wait for their completion.

    When ``marks`` has the last instruction of the input projection (``end``), the wait is on it: it reads the last
    loads, so it completes after them, and a runtime edge on a DMA breaks the released neuronx-cc 2.28
    (``tkg_weight_dma_gate.wait_for_completion``).
    """
    loads = list(marks["fills"][-2:])
    _schedule_after(instr, loads)
    end = marks.get("end")
    _runtime_after(instr, [end] if end is not None else loads)


def _w_in_buffers(local_heads: int, n_seq: int) -> int | None:
    """Return the ``kda_w_in`` buffers of ``_gemv_channel_major`` at ``local_heads`` KDA heads of one rank.

    One head (TP128) and one sequence take one buffer per DMA (9.3 MB, 73 KB per partition), so the whole weight
    streams during the MoE tail and the attention head. More heads or sequences keep the ring of ``_W_IN_BUFFERS``:
    at two sequences the late stream collides with the HBM-bank DMAs of the longer attention head.
    """
    return None if local_heads == 1 and n_seq == 1 else _W_IN_BUFFERS


def _w_in_chunk_sizes(hidden_tiles: int) -> list:
    """Return the K-tiles of each ``kda_w_in`` DMA: full chunks, then the ``_W_IN_TAIL_TILES``."""
    tail = [size for size in _W_IN_TAIL_TILES]
    body = hidden_tiles - sum(tail)
    if body < 0:
        return [hidden_tiles]
    sizes = [_W_IN_TILES_PER_DMA] * (body // _W_IN_TILES_PER_DMA)
    if body % _W_IN_TILES_PER_DMA:
        sizes.append(body % _W_IN_TILES_PER_DMA)
    return sizes + tail


def _conv_weights(small_sb, params_sb, small: dict, n_conv: int, conv_dtype: str) -> nl.NkiTensor:
    """Return the ``[128, 3*Hl, CONV_TAPS]`` conv weight view: BF16 from ``kda_w_small``, or FP32 from ``kda_params``."""
    if conv_dtype == "fp32":
        return params_sb[0:P_MAX, small["conv32"] : small["conv32"] + n_conv * CONV_TAPS].reshape(
            (P_MAX, n_conv, CONV_TAPS)
        )
    return small_sb[0:P_MAX, small["conv"] : small["conv"] + n_conv * CONV_TAPS].reshape((P_MAX, n_conv, CONV_TAPS))


def _divide_by_one_plus(dst: nl.NkiTensor, exp_neg: nl.NkiTensor, numerator) -> None:
    """Write ``numerator / (1 + exp_neg)`` to ``dst`` on the VectorE.

    With ``exp_neg = exp(-x)`` this is ``sigmoid(x)`` for a numerator of 1.0 and ``silu(x)`` for a
    numerator of ``x``. The ScalarE then runs exp and ln only, which share one function table.
    """
    denominator = nl.ndarray(exp_neg.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=denominator, data=exp_neg, op0=nl.add, operand0=1.0)
    if isinstance(numerator, float):
        nisa.reciprocal(dst=dst, data=denominator)
    else:
        inverse = nl.ndarray(exp_neg.shape, dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=inverse, data=denominator)
        nisa.tensor_tensor(dst=dst, data1=numerator, data2=inverse, op=nl.multiply)


def _off_to_null_page(page_dma, valid, shape: tuple, null_page: int) -> None:
    """With a KDA null page (``null_page`` > 0), add ``null_page * (1 - valid)`` to ``page_dma`` (the page id times
    ``valid``, FP32 SBUF of ``shape``) in place, so each off access goes to the null page, not to page 0. Without one it
    adds no instruction."""
    if not null_page:
        return
    vector = nisa.engine.vector
    off = nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=off,
        data=valid,
        op0=nl.multiply,
        operand0=-float(null_page),
        op1=nl.add,
        operand1=float(null_page),
        engine=vector,
    )
    nisa.tensor_tensor(dst=page_dma, data1=page_dma, data2=off, op=nl.add, engine=vector)


def _even_pages_only(high: nl.NkiTensor, page_i: nl.NkiTensor, shape: tuple) -> None:
    """``paired_pages``: zero the 0/1 ``high`` count of every odd page id (2 VectorE ops), so an odd id is off, as an
    out-of-range id is: its accesses go to page 0 (or the KDA null page), never to a pair that reaches the next page.
    ``page_i`` is the int32 page-id tile behind ``high``. An even id keeps its count, so the bits do not change."""
    odd = nl.ndarray(shape, dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=odd, data=page_i, op0=nl.bitwise_and, operand0=1, engine=nisa.engine.vector)
    nisa.scalar_tensor_tensor(dst=high, data=odd, op0=nl.equal, operand0=0, op1=nl.multiply, operand1=high)


def _page_counts(
    page_id: nl.NkiTensor,
    position: nl.NkiTensor,
    n_pages: int,
    tag: str,
    after=(),
    done=None,
    null_page: int = 0,
    even_only: bool = False,
):
    """Return the DMA pages and the 0/1 masks of the read and the write of each sequence.

    The write count of a sequence is 1 for a page in ``[1, n_pages)`` and 0 otherwise. The read count
    is the write count, or 0 at position 0 (positions are not negative). Each DMA page is the page id
    when its count is 1, and page 0 otherwise, so every access stays in the pool. With a KDA null page
    (``null_page`` > 0), the write count is 1 only for a page in ``(null_page, n_pages)``, and each off
    access goes to the null page.

    A broadcast DMA on the SP queue puts the ``B`` page ids and positions on all 128 partitions, and
    the VectorE computes the counts on every partition. So the masks need no TensorE matmul, and
    the state DMAs on the SP queue do not wait for the GpSimd, which starts late in a collective
    kernel. The TensorE and the DMA queues run in program order, so one late instruction holds back
    every later one.

    Args:
        page_id (nl.NkiTensor): [B, 1] int32 HBM, page of each sequence.
        position (nl.NkiTensor): [B, 1] int32 HBM, position of each sequence.
        n_pages (int): Pages in the cache pool.
        tag (str): Unique per-layer name prefix.
        after (tuple): instructions that the two metadata loads follow in the schedule (a schedule edge on each).
            Empty (default) adds no edge.
        done (list): None (default), or a list that takes the last two VectorE ops (the page and the mask copies).
        null_page (int): 0 (default), or the KDA null page (``kda_decode_attention`` ``null_page``).
        even_only (bool): ``paired_pages``: an odd page id is off as well (``_even_pages_only``). False (default)
            keeps the program.

    Returns:
        tuple: ``read_page`` and ``write_page``, SBUF int32 ``[128, B]`` (use partition 0 as a DMA
        offset), and ``read_mask`` and ``write_mask``, SBUF uint16 ``[128, B]``.
    """
    n_seq = page_id.shape[0]
    meta_i = nl.ndarray((P_MAX, 2, n_seq), dtype=nl.int32, buffer=nl.sbuf)
    for meta_col, meta in ((0, page_id), (1, position)):
        load = nisa.dma_copy(
            dst=meta_i[0:P_MAX, meta_col, 0:n_seq],
            src=meta.ap(pattern=[[0, P_MAX], [1, n_seq]], offset=0),
            dge_mode=nisa.dge_mode.hwdge,
            engine=nisa.engine.sync,
            name=f"{tag}_kda_meta_load_{meta_col}",
        )
        _schedule_after(load, after)
    vector = nisa.engine.vector
    meta_f = nl.ndarray((P_MAX, 2, n_seq), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=meta_f, src=meta_i, engine=vector)
    page_f = meta_f[0:P_MAX, 0, 0:n_seq]
    # Rows: write count, read count, read page, write page.
    values_f = nl.ndarray((P_MAX, 4, n_seq), dtype=nl.float32, buffer=nl.sbuf)
    write_count = values_f[0:P_MAX, 0, 0:n_seq]
    read_count = values_f[0:P_MAX, 1, 0:n_seq]
    page_high = nl.ndarray((P_MAX, n_seq), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=page_high, data=page_f, op0=nl.less, operand0=float(n_pages), engine=vector)
    if even_only:
        _even_pages_only(page_high, meta_i[0:P_MAX, 0, 0:n_seq], (P_MAX, n_seq))
    nisa.scalar_tensor_tensor(
        dst=write_count,
        data=page_f,
        op0=nl.greater_equal,
        operand0=float(1 + null_page),
        op1=nl.multiply,
        operand1=page_high,
    )
    nisa.scalar_tensor_tensor(
        dst=read_count,
        data=meta_f[0:P_MAX, 1, 0:n_seq],
        op0=nl.greater_equal,
        operand0=1.0,
        op1=nl.multiply,
        operand1=write_count,
    )
    for page_row, count in ((2, read_count), (3, write_count)):
        nisa.tensor_tensor(
            dst=values_f[0:P_MAX, page_row, 0:n_seq], data1=page_f, data2=count, op=nl.multiply, engine=vector
        )
        _off_to_null_page(values_f[0:P_MAX, page_row, 0:n_seq], count, (P_MAX, n_seq), null_page)
    pages_i = nl.ndarray((P_MAX, 2, n_seq), dtype=nl.int32, buffer=nl.sbuf)
    pages_copy = nisa.tensor_copy(dst=pages_i, src=values_f[0:P_MAX, 2:4, 0:n_seq], engine=vector)
    masks = nl.ndarray((P_MAX, 2, n_seq), dtype=nl.uint16, buffer=nl.sbuf)
    masks_copy = nisa.tensor_copy(dst=masks, src=values_f[0:P_MAX, 0:2, 0:n_seq], engine=vector)
    if done is not None:
        done.extend((pages_copy, masks_copy))
    return (
        pages_i[0:P_MAX, 0, 0:n_seq],
        pages_i[0:P_MAX, 1, 0:n_seq],
        masks[0:P_MAX, 1, 0:n_seq],
        masks[0:P_MAX, 0, 0:n_seq],
    )


def _conv_scatter_rows(
    kv_pool: nl.NkiTensor,
    write_page: nl.NkiTensor,
    n_conv: int,
    persistent: dict = None,
    done: list = None,
    wait: tuple = (),
    after: tuple = (),
    index_done: list = None,
    null_page_rows: int = 0,
    null_page: int = 0,
) -> list:
    """Return the pool row of each conv row, for the scatter of each group of sequences (``_store_conv_states``).

    With the read page in place of ``write_page`` it gives the rows of the conv-state gather (``_gather_conv_rows``).

    The pool view has rows of ``row = 128 * (CONV_TAPS-1)`` elements, so page ``p`` starts at row ``p * W / row``.
    Partition ``r`` of group ``[first, last)`` holds conv row ``r % n_conv`` of sequence ``first + r // n_conv``.
    Its pool row is ``write_page[s] * W / row + r - n_conv * (s - first)`` at the one sequence ``s`` where
    ``0 <= r - n_conv * (s - first) < n_conv``, so a one-hot of that range picks the sequence without a division.

    With a ``null_page_rows`` (not 0), a sequence ``s`` at that page takes the pool rows ``null_page_rows * W / row + n_conv * s
    .. + n_conv - 1``: each sequence of the step has its own rows of the page, so no two partitions of a scatter (and
    no two scatters of the call) write one pool row. A gather (the read page) takes no ``null_page_rows``.

    Args:
        kv_pool (nl.NkiTensor): [P, W] bf16 HBM, the paged cache pool.
        write_page (nl.NkiTensor): [128, B] int32 SBUF, the write page of each sequence on every partition.
        n_conv (int): Conv rows of a page, 3 * Hl.
        persistent (dict): None, or the persistent tiles ``kda_conv_index_<g>`` / ``kda_conv_off_<g>`` to write.
        done (list): None, or a list that takes the last index op of each group.
        wait (tuple): DMAs that read the persistent tiles before (``kda_decode_attention`` ``conv_copy_wait``): each
            write into a persistent tile gets a runtime dependency on them. Empty (default) adds no edge.
        after (tuple): instructions that the row iota of each group follows in the schedule (the only op of a group
            that reads no page). Empty (default) adds no edge.
        index_done (list): None (default), or a list that takes the op that writes the pool rows of each group.
        null_page_rows (int): 0 (default) keeps the program, else the page of the invalid sequences
            (``kda_decode_attention`` ``null_page_rows``).

    Returns:
        list: For each group, an int32 SBUF ``[rows, 1]`` tile with the pool rows of its partitions, and a uint16
        ``[rows, 1]`` mask that is 1 on the partitions of an off write (page 0, or the null page with ``null_page``).
    """
    row = HEAD_DIM * (CONV_TAPS - 1)
    kernel_assert(kv_pool.shape[1] % row == 0, f"batched conv stores need a page width that is a multiple of {row}")
    rows_per_page = kv_pool.shape[1] // row
    kernel_assert(
        kv_pool.shape[0] * rows_per_page < 1 << 24, "batched conv stores need fewer than 2**24 pool rows (fp32 index)"
    )
    n_seq = write_page.shape[1]
    group = P_MAX // n_conv
    fp32 = nl.float32
    vector = nisa.engine.vector
    page_f = nl.ndarray((P_MAX, n_seq), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=page_f, src=write_page, engine=vector)
    page_rows = None
    if null_page_rows:
        kernel_assert(
            n_seq * n_conv <= rows_per_page,
            f"null_page_rows needs {n_seq} * {n_conv} conv rows of {row} elements in a page, the pool page has "
            f"{rows_per_page}",
        )
        # page_rows[s] = page[s] * rows_per_page + (page[s] == null_page_rows) * n_conv * s, the first pool row of s.
        seq_rows = nl.ndarray((P_MAX, n_seq), dtype=fp32, buffer=nl.sbuf)
        _schedule_after(nisa.iota(dst=seq_rows, pattern=[[n_conv, n_seq]], offset=0, channel_multiplier=0), after)
        null_rows = nl.ndarray((P_MAX, n_seq), dtype=fp32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(
            dst=null_rows, data=page_f, op0=nl.equal, operand0=float(null_page_rows), op1=nl.multiply, operand1=seq_rows
        )
        page_rows = nl.ndarray((P_MAX, n_seq), dtype=fp32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(
            dst=page_rows,
            data=page_f,
            op0=nl.multiply,
            operand0=float(rows_per_page),
            op1=nl.add,
            operand1=null_rows,
        )
    tiles = []
    for first in range(0, n_seq, group):
        n_rows = (min(n_seq, first + group) - first) * n_conv
        lag_i = nl.ndarray((n_rows, n_seq), dtype=nl.int32, buffer=nl.sbuf)
        _schedule_after(
            nisa.iota(dst=lag_i, pattern=[[-n_conv, n_seq]], offset=n_conv * first, channel_multiplier=1), after
        )
        lag = nl.ndarray((n_rows, n_seq), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=lag, src=lag_i, engine=vector)
        at_least = nl.ndarray((n_rows, n_seq), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=at_least, data=lag, op0=nl.greater_equal, operand0=0.0, engine=vector)
        # (lag >= n_conv) - (lag >= 0): -1 at the sequence of the row, 0 elsewhere.
        pick = nl.ndarray((n_rows, n_seq), dtype=fp32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(
            dst=pick, data=lag, op0=nl.greater_equal, operand0=float(n_conv), op1=nl.subtract, operand1=at_least
        )
        target = nl.ndarray((n_rows, n_seq), dtype=fp32, buffer=nl.sbuf)
        if page_rows is None:
            nisa.scalar_tensor_tensor(
                dst=target,
                data=page_f[0:n_rows, 0:n_seq],
                op0=nl.multiply,
                operand0=float(rows_per_page),
                op1=nl.add,
                operand1=lag,
            )
        else:
            nisa.tensor_tensor(dst=target, data1=page_rows[0:n_rows, 0:n_seq], data2=lag, op=nl.add, engine=vector)
        nisa.tensor_tensor(dst=target, data1=target, data2=pick, op=nl.multiply, engine=vector)
        neg_row = nl.ndarray((n_rows, 1), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=neg_row, data=target, op=nl.add, axis=(1,))
        # Persistent tiles (kda_conv_index_tiles) keep their bytes through the kernel: a later tile cannot take them
        # while the scatter still reads them, so it does not wait for the scatter.
        kept_i = (persistent or {}).get(f"kda_conv_index_{first // group}")
        kept_off = (persistent or {}).get(f"kda_conv_off_{first // group}")
        if kept_i is not None:
            rows_i = kept_i[0:n_rows, 0:1]
        else:
            rows_i = nl.ndarray((n_rows, 1), dtype=nl.int32, buffer=nl.sbuf)
        index_write = nisa.tensor_scalar(dst=rows_i, data=neg_row, op0=nl.multiply, operand0=-1.0, engine=vector)
        if kept_i is not None:
            _runtime_after(index_write, wait)
        if index_done is not None:
            index_done.append(index_write)
        # A row of an off write (page 0, or the null page and below with ``null_page``) has a pool row below
        # (null_page + 1) * rows_per_page.
        if kept_off is not None:
            off = kept_off[0:n_rows, 0:1]
        else:
            off = nl.ndarray((n_rows, 1), dtype=nl.uint16, buffer=nl.sbuf)
        last = nisa.tensor_scalar(
            dst=off, data=neg_row, op0=nl.greater, operand0=-float(rows_per_page * (null_page + 1)), engine=vector
        )
        if kept_off is not None:
            _runtime_after(last, wait)
        if done is not None:
            done.append(last)
        tiles.append((rows_i, off))
    return tiles


def _check_null_page_rows(null_page_rows: int, n_pages: int) -> None:
    """Stop the trace unless ``null_page_rows`` is 0 or a page of the pool other than page 0 (``kda_decode_attention``)."""
    kernel_assert(
        isinstance(null_page_rows, int)
        and not isinstance(null_page_rows, bool)
        and (null_page_rows == 0 or 0 < null_page_rows < n_pages),
        f"null_page_rows must be 0 (the default) or a page in [1, {n_pages}), got {null_page_rows!r}",
    )


def _conv_rows_view(kv_pool: nl.NkiTensor, page_index: nl.NkiTensor, n_conv: int) -> nl.NkiTensor:
    """Return the conv state of one page as an HBM view ``[3*Hl, 128*(CONV_TAPS-1)]``.

    Row ``Hl*p + h`` holds channels ``h*128 .. h*128 + 127`` of projection ``p``, three taps per
    channel, oldest first. Each row is contiguous, so the DMA moves 3*Hl rows of 768 bytes.
    """
    row = HEAD_DIM * (CONV_TAPS - 1)
    return kv_pool.ap(pattern=[[row, n_conv], [1, row]], offset=0, scalar_offset=page_index, indirect_dim=0)


def _stationary_parts(state_dtype: str) -> int:
    """Return the bf16 stationary parts of the ``decoupled_bf16`` prediction for a ``state_dtype`` state (0 for bf16)."""
    return {"fp32": 1, "fp32_hilo": 2}.get(state_dtype, 0)


def _load_recurrent_states(
    kv_pool,
    read_page,
    read_mask,
    heads,
    page,
    tag,
    after=(),
    gated=False,
    loads=None,
    state_dtype="bf16",
    parts=0,
    engine=None,
    select=True,
):
    """Load the recurrent state of the read page of each sequence as ``[128 k, B, Hl, 128 v]`` (bf16, or fp32).

    Return the state tile and a tuple of ``parts`` bf16 stationary tiles of the same shape (``KDA_STATE_DTYPES``).
    A select puts zeros in place of an off read (``read_mask`` 0), so a NaN in a stale page cannot pass.
    With ``after`` (none by default), the first load follows those instructions in the schedule and each later load
    follows the load before it, so all the loads keep the trace order behind ``after``. With ``gated``, each load also
    follows the weight gate (``gate_weight_dma``), as the first fills of the ``kda_w_in`` ring do. ``loads`` is None,
    or a list that the function extends with the load DMAs. ``engine`` is the trigger engine of the hardware DGE loads
    (None, the default: the SP queue). With ``select`` False the function leaves out the selects
    (``_select_recurrent_states`` runs them later); it then takes no stationary parts.

    ``state_dtype`` ``fp32`` or ``fp32_hilo`` loads the fp32 state of the page. Right after the select of a sequence
    the VectorE then writes its ``parts`` stationary tiles: ``bf16(S)``, and for 2 parts ``bf16(S - bf16(S))``. They
    need only the load, so they run in the idle VectorE time of the input projection, as the selects do. Each
    sequence has its own rows of each tile, so a NaN state reaches only the prediction of its own sequence.
    """
    kernel_assert(select or not parts, "the stationary parts of the recurrent states need their selects")
    n_seq = read_page.shape[1]
    dtype = nl.bfloat16 if state_dtype == "bf16" else nl.float32
    staged_state = nl.ndarray((P_MAX, n_seq, heads, HEAD_DIM), dtype=dtype, buffer=nl.sbuf)
    stationaries = tuple(
        nl.ndarray((P_MAX, n_seq, heads, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(parts)
    )
    previous = None
    for seq in range(n_seq):
        load = nisa.dma_copy(
            dst=staged_state[0:P_MAX, seq, 0:heads, 0:HEAD_DIM],
            src=_recurrent_state_view(kv_pool, read_page[0:1, seq : seq + 1], heads, page["recurrent"], state_dtype),
            dge_mode=nisa.dge_mode.hwdge,
            engine=nisa.engine.sync if engine is None else engine,
            name=f"{tag}_kda_recurrent_state_load_{seq}",
        )
        if after:
            # One edge per load: the scheduler can otherwise move a later load ahead of the others (MEASURED in the
            # compiled order: with an edge on each load to the last first fill only, it put the last load first).
            _schedule_after(load, after if previous is None else [previous])
            previous = load
        if gated and load is not None:
            gate_weight_dma(load if hasattr(load, "depends_on") else nisa.NkiInstruction(load))
        if loads is not None:
            loads.append(load)
        if not select:
            continue
        loaded = _select_recurrent_state(staged_state, read_mask, heads, seq)
        _state_parts(loaded, stationaries, heads, seq)
    return staged_state, stationaries


def _state_parts(loaded, stationaries, heads, seq):
    """Write the bf16 stationary parts of the selected state ``loaded`` of sequence ``seq`` into its rows of
    ``stationaries`` (none, ``bf16(S)``, or ``bf16(S)`` and ``bf16(S - bf16(S))``), on the VectorE."""
    if not stationaries:
        return
    high = stationaries[0][0:P_MAX, seq, 0:heads, 0:HEAD_DIM]
    nisa.tensor_copy(dst=high, src=loaded, engine=nisa.engine.vector)
    if len(stationaries) > 1:
        # S - bf16(S) is exact in fp32; its bf16 rounding is the low part.
        nisa.tensor_tensor(
            dst=stationaries[1][0:P_MAX, seq, 0:heads, 0:HEAD_DIM],
            data1=loaded,
            data2=high,
            op=nl.subtract,
            engine=nisa.engine.vector,
        )


def _select_recurrent_states(staged_state, read_mask, heads, parts=0):
    """Run the selects of ``_load_recurrent_states`` on ``staged_state`` (loaded with ``select`` False), each followed
    by the writes of its ``parts`` bf16 stationary parts, as the loads with ``select`` make them. Return the state tile and
    the tuple of the part tiles."""
    stationaries = tuple(nl.ndarray(staged_state.shape, dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(parts))
    for seq in range(staged_state.shape[1]):
        loaded = _select_recurrent_state(staged_state, read_mask, heads, seq)
        _state_parts(loaded, stationaries, heads, seq)
    return staged_state, stationaries


def _select_recurrent_state(staged_state, read_mask, heads, seq):
    """Put zeros in place of the loaded recurrent state of sequence ``seq`` for an off read (``read_mask`` 0); return
    the view of that sequence."""
    loaded = staged_state[0:P_MAX, seq, 0:heads, 0:HEAD_DIM]
    nisa.select_reduce(
        dst=loaded,
        predicate=_per_sequence(read_mask[0:P_MAX, seq : seq + 1], (heads, HEAD_DIM)).squeeze_dim(1),
        on_true=loaded,
        on_false=0.0,
    )
    return loaded


def _gather_conv_rows(kv_pool, read_page, n_conv, tag, done=None, after=(), index_done=None, gather_rows=None):
    """Load the conv rows of the read page of each sequence with one gather DMA per group of ``128 // (3*Hl)`` sequences.

    Return the ``[rows, 128*(CONV_TAPS-1)]`` bf16 row tile of each group, in the layout of the per-sequence loads of
    ``_load_conv_states``: partition ``r`` holds conv row ``r % (3*Hl)`` of sequence ``first + r // (3*Hl)``. The
    gather reads the row view of the pool at the rows of ``_conv_scatter_rows`` (the read page of the sequence times
    the rows of a page, plus the conv row), so it moves the same bytes as the per-sequence loads. An off read reads
    page 0, as the per-sequence load does. ``gather_rows`` (default None) holds the ``_conv_scatter_rows`` result of
    these read pages that the caller made, which then replaces the call here.
    """
    n_seq = read_page.shape[1]
    row = HEAD_DIM * (CONV_TAPS - 1)
    group = P_MAX // n_conv
    pool_rows = kv_pool.reshape((kv_pool.shape[0] * (kv_pool.shape[1] // row), row))
    if gather_rows is None:
        gather_rows = _conv_scatter_rows(kv_pool, read_page, n_conv, done=done, after=after, index_done=index_done)
    tiles = []
    for first, (rows_index, _) in zip(range(0, n_seq, group), gather_rows):
        n_rows = (min(n_seq, first + group) - first) * n_conv
        rows = nl.ndarray((n_rows, row), dtype=nl.bfloat16, buffer=nl.sbuf)
        gather = nisa.dma_copy(
            dst=rows,
            src=pool_rows.ap(
                pattern=[[row, n_rows], [1, row]], offset=0, vector_offset=rows_index[0:n_rows, 0:1], indirect_dim=0
            ),
            dge_mode=nisa.dge_mode.swdge,
            name=f"{tag}_kda_conv_state_gather_{first // group}",
        )
        if done is not None:
            done.append(gather)
        tiles.append(rows)
    return tiles


def _load_conv_states(kv_pool, read_page, n_conv, tag, gathered=None, after=(), load_after=(), load_follow=()):
    """Load the conv state of the read page of each sequence as ``[128 d, B, 3*Hl, CONV_TAPS-1]`` bf16.

    A DMA per sequence loads the contiguous rows (``_conv_rows_view``), so the rows of up to
    ``128 // (3*Hl)`` sequences fill the partitions of one tile. One TensorE transpose per tap then
    puts the channel on the partitions. A DMA with the channel on the partitions would move 6-byte
    elements, and at B=32 those DMAs take a long time. ``gathered`` is None, or the row tiles of
    ``_gather_conv_rows``, which then replace the per-sequence loads. Each transpose follows the instructions
    ``after`` in the schedule (none by default). Each per-sequence load follows the instructions ``load_after`` in the
    schedule and waits for them at runtime (none by default), and follows the instructions ``load_follow`` in the
    schedule only (none by default).
    """
    n_seq = read_page.shape[1]
    taps = CONV_TAPS - 1
    row = HEAD_DIM * taps
    group = P_MAX // n_conv
    staged_conv = nl.ndarray((P_MAX, n_seq, n_conv, taps), dtype=nl.bfloat16, buffer=nl.sbuf)
    for first in range(0, n_seq, group):
        last = min(n_seq, first + group)
        n_rows = (last - first) * n_conv
        if gathered is not None:
            rows = gathered[first // group]
        else:
            rows = nl.ndarray((n_rows, row), dtype=nl.bfloat16, buffer=nl.sbuf)
            for seq in range(first, last):
                load = nisa.dma_copy(
                    dst=rows[(seq - first) * n_conv : (seq - first + 1) * n_conv, 0:row],
                    src=_conv_rows_view(kv_pool, read_page[0:1, seq : seq + 1], n_conv),
                    dge_mode=nisa.dge_mode.hwdge,
                    engine=nisa.engine.sync,
                    name=f"{tag}_kda_conv_state_load_{seq}",
                )
                _schedule_after(load, list(load_after) + list(load_follow))
                _runtime_after(load, load_after, engine=nisa.engine.sync)
        # A bit-accurate transpose, so a NaN in a stale page stays in its own element. A matmul with
        # an identity would spread it to every element (NaN * 0 is NaN).
        # Pad the tap stride to an even count: a bf16 PSUM access must be 4-byte aligned.
        cols_psum = nl.ndarray((P_MAX, taps, n_rows + n_rows % 2), dtype=nl.bfloat16, buffer=nl.psum)
        for tap in range(taps):
            tr = nisa.nc_transpose(dst=cols_psum[0:P_MAX, tap, 0:n_rows], data=rows[0:n_rows, tap:row:taps])
            if after:
                handle = tr if hasattr(tr, "depends_on") else nisa.NkiInstruction(tr)
                handle.depends_on(*after, kind=nisa.dependency.schedule)
        nisa.tensor_copy(
            dst=staged_conv[0:P_MAX, first:last, 0:n_conv, 0:taps],
            src=cols_psum[0:P_MAX, 0:taps, 0:n_rows].reshape_dim(2, (last - first, n_conv)).permute((0, 2, 3, 1)),
            engine=nisa.engine.scalar,
        )
    return staged_conv


def _store_conv_states(
    kv_pool,
    write_page,
    window,
    tag,
    sources=None,
    copy_wait=None,
    scatter_rows=None,
    keep=None,
    transpose_after=(),
    conv_wait=(),
    zero_rows=None,
    window_seq=None,
):
    """Store the last CONV_TAPS-1 taps of ``window`` ``[128 d, B, 3*Hl, CONV_TAPS]`` to the write page of each sequence.

    One TensorE transpose per tap puts the rows of up to ``128 // (3*Hl)`` sequences on the
    partitions, and a DMA per sequence stores its contiguous rows (``_conv_rows_view``). Return the
    store DMAs. The rows of group ``g`` go to ``sources["kda_conv_rows_<g>"]`` when the kernel gives it
    (``kda_conv_rows_tiles``), else to a tile of this layer.

    ``copy_wait`` is an optional list of DMA instructions that read the ``rows`` tile of a previous
    call (its conv stores). The ``rows`` copy of this call waits for them, so a previous store finishes
    reading the tile before this call overwrites it (nki-dma-source-liveness). When it is empty the
    call adds no edge, so the decode and records paths do not change.

    ``scatter_rows`` is None, or the pool rows and the off-write masks of each group (``_conv_scatter_rows``): each
    group then takes one scatter DMA into the row view of the pool, in place of one DMA per sequence. The rows of an
    off write become zeros first, because several sequences can write page 0 and the scatter has no order. ``keep``
    (a list, or None) takes the ``(source rows, pool-row index or None, store DMAs)`` of each group
    (``kda_decode_attention`` ``scatter_keep``). ``transpose_after`` (default none) holds instructions that each
    transpose follows in the schedule. ``conv_wait`` (default none) holds DMAs that read the persistent rows tiles
    before: the rows copy into a persistent tile gets a runtime dependency on each of them. ``zero_rows`` (default
    None) is a bf16 zero tile ``[128, 128 * (CONV_TAPS-1)]`` that lives through the kernel
    (``kda_conv_zero_rows_tile``): the off writes of a scatter then take their zeros from it, and the call issues no
    memset. ``window_seq`` (default None) takes the place of ``window``: the conv inputs by position ``[128 d,
    CONV_TAPS - 1 + Q, B', 3*Hl]`` (``_conv_tokens``), whose rows ``j * B' + b`` read tap ``t`` at position
    ``j + t``. Each transpose reads the bytes of the window from it, so no copy builds the window. The values do not
    change.
    """
    conv_row = HEAD_DIM * (CONV_TAPS - 1)
    copy_wait = tuple(copy_wait or ())
    stores = []
    if window_seq is not None:
        kernel_assert(window is None, "window_seq takes the place of window")
        position = window_seq.shape[2] * window_seq.shape[3]
        n_seq, n_conv = (window_seq.shape[1] - (CONV_TAPS - 1)) * window_seq.shape[2], window_seq.shape[3]
    else:
        n_seq, n_conv = window.shape[1], window.shape[2]
    taps = CONV_TAPS - 1
    row = HEAD_DIM * taps
    group = P_MAX // n_conv
    seq_stride = n_conv * CONV_TAPS
    for first in range(0, n_seq, group):
        last = min(n_seq, first + group)
        n_rows = (last - first) * n_conv
        rows_psum = nl.ndarray((n_rows, taps, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.psum)
        for tap in range(taps):
            if window_seq is not None:
                # Tap 1 + tap of rows first.. is position 1 + tap of their tokens: one contiguous run of n_rows.
                columns = window_seq.ap(
                    pattern=[[window_seq.shape[1] * position, P_MAX], [1, n_rows]],
                    offset=(1 + tap) * position + first * n_conv,
                )
            else:
                # Rows (sequence, projection head) of tap 1 + tap: one free dimension of stride CONV_TAPS.
                columns = window.ap(
                    pattern=[[n_seq * seq_stride, P_MAX], [CONV_TAPS, n_rows]], offset=first * seq_stride + 1 + tap
                )
            transpose = nisa.nc_transpose(dst=rows_psum[0:n_rows, tap, 0:HEAD_DIM], data=columns)
            if transpose_after:
                _schedule_after(transpose, transpose_after)
        persistent = (sources or {}).get(f"kda_conv_rows_{first // group}")
        if persistent is None:
            rows = nl.ndarray((n_rows, HEAD_DIM, taps), dtype=nl.bfloat16, buffer=nl.sbuf)
        else:
            rows = persistent[0:n_rows, 0:HEAD_DIM, 0:taps]
        rows_copy = nisa.tensor_copy(dst=rows, src=rows_psum.permute((0, 2, 1)), engine=nisa.engine.scalar)
        if copy_wait and hasattr(rows_copy, "depends_on"):
            if persistent is not None:
                # One tensor: the compiler orders this write after each DMA that read the tile (a WAR edge).
                rows_copy.depends_on(*copy_wait, kind=nisa.dependency.schedule)
            else:
                rows_copy.depends_on(*copy_wait)
        if persistent is not None:
            _persistent_rows_after(rows_copy, conv_wait)
        if scatter_rows is not None:
            pool_rows = kv_pool.reshape((kv_pool.shape[0] * (kv_pool.shape[1] // row), row))
            rows_index, off = scatter_rows[first // group]
            src = rows[0:n_rows, 0:HEAD_DIM, 0:taps].flatten_dims(1, 2)
            if zero_rows is None:
                zeros = nl.ndarray((n_rows, row), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.memset(dst=zeros, value=0.0)
            else:
                kernel_assert(
                    n_rows <= zero_rows.shape[0], f"zero_rows has {zero_rows.shape[0]} rows, the scatter {n_rows}"
                )
                kernel_assert(
                    row <= zero_rows.shape[1], f"zero_rows has {zero_rows.shape[1]} columns, a conv row {row}"
                )
                zeros = zero_rows[0:n_rows, 0:row]
            nisa.tensor_copy_predicated(dst=src, src=zeros, predicate=off[0:n_rows, 0:1].broadcast(1, row))
            scatter = nisa.dma_copy(
                dst=pool_rows.ap(
                    pattern=[[row, n_rows], [1, row]],
                    offset=0,
                    vector_offset=rows_index[0:n_rows, 0:1],
                    indirect_dim=0,
                ),
                src=src,
                dge_mode=nisa.dge_mode.swdge,
                name=f"{tag}_kda_conv_state_scatter_{first // group}",
            )
            stores.append(scatter)
            if keep is not None:
                keep.append((src, rows_index[0:n_rows, 0:1], (scatter,)))
            continue
        group_stores = []
        for seq in range(first, last):
            group_stores.append(
                nisa.dma_copy(
                    dst=_conv_rows_view(kv_pool, write_page[0:1, seq : seq + 1], n_conv),
                    src=rows[(seq - first) * n_conv : (seq - first + 1) * n_conv, 0:HEAD_DIM, 0:taps].flatten_dims(
                        1, 2
                    ),
                    dge_mode=nisa.dge_mode.hwdge,
                    engine=nisa.engine.sync,
                    name=f"{tag}_kda_conv_state_store_{seq}",
                )
            )
        stores.extend(group_stores)
        if keep is not None:
            keep.append((rows[0:n_rows, 0:HEAD_DIM, 0:taps].flatten_dims(1, 2), None, tuple(group_stores)))
    return stores


def _conv_state_rows(window, sources=None, conv_wait=()):
    """Return the conv-state row tiles of ``_store_conv_states`` (its transposes and ScalarE copy), one per group.

    Each entry is ``(first, n_rows, rows)``: the ``[n_rows, 128, CONV_TAPS-1]`` bf16 rows of the sequences
    ``first ..`` of the group, in the layout of the scatter (``conv_stores_early`` of ``kda_decode_attention``).
    The copy into a persistent tile gets a runtime dependency on each DMA of ``conv_wait`` (as ``_store_conv_states``).
    """
    n_seq, n_conv = window.shape[1], window.shape[2]
    taps = CONV_TAPS - 1
    group = P_MAX // n_conv
    seq_stride = n_conv * CONV_TAPS
    out = []
    for first in range(0, n_seq, group):
        last = min(n_seq, first + group)
        n_rows = (last - first) * n_conv
        rows_psum = nl.ndarray((n_rows, taps, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.psum)
        for tap in range(taps):
            columns = window.ap(
                pattern=[[n_seq * seq_stride, P_MAX], [CONV_TAPS, n_rows]], offset=first * seq_stride + 1 + tap
            )
            nisa.nc_transpose(dst=rows_psum[0:n_rows, tap, 0:HEAD_DIM], data=columns)
        persistent = (sources or {}).get(f"kda_conv_rows_{first // group}")
        if persistent is None:
            rows = nl.ndarray((n_rows, HEAD_DIM, taps), dtype=nl.bfloat16, buffer=nl.sbuf)
        else:
            rows = persistent[0:n_rows, 0:HEAD_DIM, 0:taps]
        rows_copy = nisa.tensor_copy(dst=rows, src=rows_psum.permute((0, 2, 1)), engine=nisa.engine.scalar)
        if persistent is not None:
            _persistent_rows_after(rows_copy, conv_wait)
        out.append((first, n_rows, rows))
    return out


def _persistent_rows_after(rows_copy, dmas) -> None:
    """Put the write ``rows_copy`` into a persistent conv-state rows tile after the DMAs ``dmas`` that read the tile.

    The tile is one tensor through the kernel, so the compiler orders the write after each DMA that read it (a WAR
    edge, a semaphore wait on the DMA). This call adds a schedule edge only. A runtime edge on a DMA producer breaks
    the released neuronx-cc 2.28 (``tkg_weight_dma_gate.wait_for_completion``). The values do not change.
    """
    _schedule_after(rows_copy, dmas)


def _scatter_conv_rows(kv_pool, conv_rows, scatter_rows, tag, after=None, keep=None):
    """Store the row tiles of ``_conv_state_rows`` with the scatter of ``_store_conv_states``. Return the scatter DMAs.

    ``after`` is None or a VectorE instruction: the zero select of the off writes gets a schedule dependency on it,
    so it does not run in front of the work traced before ``after``. ``keep`` (a list, or None) takes the
    ``(source rows, pool-row index, (scatter DMA,))`` of each scatter, as in ``_store_conv_states``.
    """
    row = HEAD_DIM * (CONV_TAPS - 1)
    pool_rows = kv_pool.reshape((kv_pool.shape[0] * (kv_pool.shape[1] // row), row))
    stores = []
    for index, (first, n_rows, rows) in enumerate(conv_rows):
        rows_index, off = scatter_rows[index]
        src = rows[0:n_rows, 0:HEAD_DIM, 0 : CONV_TAPS - 1].flatten_dims(1, 2)
        zero_rows = nl.ndarray((n_rows, row), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(dst=zero_rows, value=0.0, engine=nisa.engine.gpsimd)
        select = nisa.tensor_copy_predicated(dst=src, src=zero_rows, predicate=off[0:n_rows, 0:1].broadcast(1, row))
        if after is not None and hasattr(select, "depends_on"):
            select.depends_on(after, kind=nisa.dependency.schedule)
        scatter = nisa.dma_copy(
            dst=pool_rows.ap(
                pattern=[[row, n_rows], [1, row]],
                offset=0,
                vector_offset=rows_index[0:n_rows, 0:1],
                indirect_dim=0,
            ),
            src=src,
            dge_mode=nisa.dge_mode.swdge,
            name=f"{tag}_kda_conv_state_scatter_{index}",
        )
        stores.append(scatter)
        if keep is not None:
            keep.append((src, rows_index[0:n_rows, 0:1], (scatter,)))
    return stores


def _recurrent_state_view(
    kv_pool: nl.NkiTensor, page_index: nl.NkiTensor, local_heads: int, offset: int, state_dtype: str = "bf16"
) -> nl.NkiTensor:
    """Return the recurrent state of one page as an HBM view ``[128 k, Hl, 128 v]``.

    ``offset`` counts bf16 elements of the page. An fp32 ``state_dtype`` views the bf16 pool as fp32 (half the
    elements per page) and reads the fp32 state at the same byte offset.
    """
    if state_dtype != "bf16":
        kernel_assert(offset % 2 == 0, f"the fp32 state offset {offset} must be even")
        kv_pool = kv_pool.view(nl.float32)
        offset //= 2
    return kv_pool.ap(
        pattern=[[HEAD_DIM, P_MAX], [HEAD_DIM * HEAD_DIM, local_heads], [1, HEAD_DIM]],
        offset=offset,
        scalar_offset=page_index,
        indirect_dim=0,
    )


def kda_read_scatter_keep(scatter_keep, sinks: dict, after=()) -> None:
    """Read every byte of the SBUF views of each conv-state scatter once more, after the scatter has read them.

    ``scatter_keep`` holds the ``(source rows, pool-row index or None, store DMAs)`` of each group of sequences
    (``kda_decode_attention`` ``scatter_keep``). ``sinks`` maps ``str(dtype)`` of bf16 and int32 to a ``[128, 1]``
    tile that the caller keeps live (for example by a store at the end of the kernel), so the compiler keeps the
    reads. ``after`` holds instructions that each read follows in the schedule. A VectorE reduce reads the whole source rows and an add reads
    the index, each with a runtime dependency on the group's stores (a wait on each DMA's completion), and the reduced
    rows are added into the bf16 sink.

    The compiler ends the live range of a DMA source at the DMA trigger, and a later tile that takes the bytes of a
    source may not wait for the DMA. After these reads the views stay live until the scatter has read them, and a
    later writer of any of their bytes follows a read that waits for the scatter. The output does not change.
    """

    def add(view):
        sink = sinks[str(view.dtype)][0 : view.shape[0], 0:1]
        return nisa.tensor_tensor(dst=sink, data1=sink, data2=view, op=nl.add, engine=nisa.engine.vector)

    for src_rows, index, dmas in scatter_keep:
        row_sum = nl.ndarray((src_rows.shape[0], 1), dtype=src_rows.dtype, buffer=nl.sbuf)
        reads = [(nisa.tensor_reduce(dst=row_sum, op=nl.add, data=src_rows, axis=(1,)), True)]
        reads.append((add(row_sum), False))
        if index is not None:
            reads.append((add(index), True))
        for read, after_stores in reads:
            _schedule_after(read, after)
            if after_stores:
                _runtime_after(read, dmas)


def kda_conv_rows_tiles(replica_tokens: int, plan) -> dict:
    """Return persistent SBUF sources of the conv-state store DMAs, one for each group of sequences.

    The compiler ends the live range of a DMA source at the DMA trigger, so a tile of a later body can take the
    bytes of a per-layer source while the store still reads them. A source that lives through the kernel keeps
    its bytes, and the next layer's write to it waits for the store. ``plan`` allocates the tiles (see
    ``decode_block.collective_sources``). The groups are sized for MAX_LOCAL_HEADS, so they cover any head count.
    """
    group = P_MAX // (N_CONV_PROJ * MAX_LOCAL_HEADS)
    return {
        f"kda_conv_rows_{g}": plan.tile((P_MAX, HEAD_DIM, CONV_TAPS - 1), nl.bfloat16, f"kda_conv_rows_{g}")
        for g in range(-(-replica_tokens // group))
    }


def kda_conv_index_tiles(replica_tokens: int, plan) -> dict:
    """Return persistent SBUF tiles of the conv-state scatter rows (``kda_decode_attention`` ``front_hook``).

    For each group of sequences an int32 ``[128, 1]`` tile of the pool rows and a uint16 ``[128, 1]`` tile of the
    off-write mask. A per-layer tile ends its live range at the scatter trigger, so a later tile (the MLP mix after
    the residual add) can take its bytes and wait for the scatter, which waits for the state stores. A tile that
    lives through the kernel keeps its bytes. ``plan`` allocates the tiles; the groups are sized for
    MAX_LOCAL_HEADS, so they cover any head count.
    """
    group = P_MAX // (N_CONV_PROJ * MAX_LOCAL_HEADS)
    tiles = {}
    for g in range(-(-replica_tokens // group)):
        tiles[f"kda_conv_index_{g}"] = plan.tile((P_MAX, 1), nl.int32, f"kda_conv_index_{g}")
        tiles[f"kda_conv_off_{g}"] = plan.tile((P_MAX, 1), nl.uint16, f"kda_conv_off_{g}")
    return tiles


def kda_conv_zero_rows_tile(plan) -> nl.NkiTensor:
    """Return the bf16 zero tile ``[128, 128 * (CONV_TAPS-1)]`` of the conv-state scatters (``_store_conv_states``
    ``zero_rows``), zeroed once. ``plan`` allocates the tile; a tile that lives through the kernel serves the scatters
    of every KDA layer, so no layer zeroes a tile of its own.
    """
    zero_rows = plan.tile((P_MAX, HEAD_DIM * (CONV_TAPS - 1)), nl.bfloat16, "kda_conv_zero_rows")
    nisa.memset(dst=zero_rows, value=0.0)
    return zero_rows


def kda_slot_state_tiles(n_seq: int, heads: int, state_dtype: str, plan, depth: int = 2) -> dict:
    """Return the persistent SBUF sources of the per-token slot-state store DMAs (``verify_kda``).

    In states mode the verify body stores each scored token's recurrent state to its slot page. The
    compiler ends a DMA source's live range at the trigger, so a per-call state tile can lose its bytes
    to a later instruction while the store still reads them (nki-dma-source-liveness). A ring of ``depth``
    tiles that lives through the kernel keeps its bytes; the next write into a ring tile waits at runtime
    for the slot stores that read it last (``verify_kda.kda_verify_attention`` ``slot_state_ring`` and
    ``slot_state_waits``, as the conv-state stores use ``kda_conv_rows_tiles``). ``depth`` 2 (the default)
    lets token ``j + 1`` read the tile of token ``j`` while token ``j + 2`` reuses it; 1 keeps one tile, so
    token ``j + 1`` waits for token ``j`` (the FIT fallback). ``plan`` allocates the tiles.
    """
    state_dt = nl.bfloat16 if state_dtype == "bf16" else nl.float32
    return {
        f"kda_slot_state_{r}": plan.tile((P_MAX, n_seq, heads, HEAD_DIM), state_dt, f"kda_slot_state_{r}")
        for r in range(depth)
    }


def kda_collective_hbm(
    replica_tokens: int, hidden_tiles: int, reduce_rows: int = None, name: str = "", pair_shared: bool = False
) -> dict:
    """Return the persistent HBM source and destination of the KDA output all-reduce, one pair for all layers.

    The pair is used from the first KDA layer on, so no other HBM tensor takes its bytes during the kernel.
    ``reduce_rows`` (default None = ``replica_tokens``) sizes the pair at more rows than the call: the DP1
    decode sizes its reduction as the verify's (Q*S rows), so the runtime picks the same collective
    algorithm by message size (bitwise decode == verify).
    ``name`` suffixes the HBM tensor names, so a kernel that builds one pair for each of several blocks (the
    verify two-step identity) does not clash on the op name. The default empty suffix keeps the names.
    ``pair_shared`` puts the destination in LNC1 shared HBM (``common.check_pair_shared_cc_dsts`` ``kda_ar``); the
    source stays private. False (default) keeps both private.
    """
    shape = (P_MAX, reduce_rows or replica_tokens, hidden_tiles)
    return {
        key: nl.ndarray(
            shape,
            dtype=nl.bfloat16,
            buffer=pair_shared_buffer(pair_shared and key == "kda_ar_dst_hbm"),
            name=f"{key}{name}",
        )
        for key in ("kda_ar_hbm", "kda_ar_dst_hbm")
    }
