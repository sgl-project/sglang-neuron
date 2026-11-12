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

"""Shared hT-layout helpers of the Kimi-K3 decode giga-kernel.

The hT layout holds ``T`` hidden vectors of width ``H`` as an SBUF tile ``[128, T, H/128]``
(token-major): element ``(p, t, j)`` is ``x[t, j*128 + p]``. All 128 partitions take part in every
vector instruction, and a K-tile ``x_hT[:, :, j]`` is the stationary operand of a GEMV against the
weight rows ``j*128 .. j*128+127``. For one token the bytes equal those of ``[128, H/128, 1]``.

A sum over the hidden dimension is a per-partition reduction followed by one TensorE matmul with an
all-ones stationary, which leaves the total on every partition.

The host stores the per-layer vector weights (norm weights, AttnRes score weights) in hT order, so
one contiguous DMA loads them. ``to_hT_order`` gives that order.
"""

from contextlib import contextmanager
from typing import NamedTuple

import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
import numpy as np

from .utils.kernel_assert import kernel_assert
from .tkg_sbuf_alloc import AUTO_SBUF

P_MAX = 128
"""Partitions of an hT tile."""
PSUM_BANK_FP32 = 512
"""fp32 elements of one PSUM bank on each partition. A matmul must not write more than one bank."""
COMPACT_MIX_TOKENS = 4
"""Above this many tokens, ``attn_res_hT`` works on one bank entry at a time. The ``[128, C, T, n]`` FP32
scratch tiles of a full bank (28 KB at C = 8, T = 8) do not fit beside a MoE body in an allocated kernel
(NCC_EGCA111). Up to this many tokens the mix keeps its one-pass form."""
_PASS_BYTES = 8192
"""FP32 scratch bytes per partition of one pass of the cached mix over the entries of an SBUF bank."""
# The instructions that the row sums of a ``row_sums_order`` scope follow in the schedule.
_SUMS_AFTER = []
# The instructions that only the next VectorE row-sum instruction follows (``order_after(instrs, once=True)``).
_SUMS_ONCE = {"instrs": ()}


def to_hT_order(vector: np.ndarray) -> np.ndarray:
    """Return the host order of a ``[..., H]`` vector weight for a contiguous hT load.

    Element ``p * (H/128) + j`` of the result is element ``j * 128 + p`` of the input.

    Args:
        vector: A host array whose last axis has width ``H``, a multiple of 128.

    Returns:
        An array of the same shape in hT order.

    Example:
        >>> to_hT_order(np.arange(256.0))[:3]
        array([  0., 128.,   1.])
    """
    width = vector.shape[-1]
    n_tiles = width // P_MAX
    lead = vector.shape[:-1]
    swapped = vector.reshape(*lead, n_tiles, P_MAX).swapaxes(-1, -2)
    return np.ascontiguousarray(swapped).reshape(*lead, width)


def is_trn3() -> bool:
    """Return whether the target is trn3 or newer (NeuronCore-v4)."""
    return nisa.get_nc_version() >= nisa.nc_version.gen4


def fp8_page_dtype():
    """Return the dtype of the FP8 cache pages: float8_e4m3fn on trn3, the legacy float8_e4m3 on trn2.

    Trn2 has only the legacy FP8 format (largest finite value 240). The two formats have the same bits for every value
    below 240 in magnitude, so a page that holds no such value reads the same on both targets.
    """
    return nl.float8_e4m3fn if is_trn3() else nl.float8_e4m3


def fp8_kernel_pages(fp8_pool: nl.NkiTensor, kernel_page_width: int) -> nl.NkiTensor:
    """Return an FP8 page pool ``[P, k * kernel_page_width]`` as the pool of its ``k P`` kernel pages, with no copy.

    A served page holds ``k`` MLA kernel pages of 128 rows one after the other: ``k = 2`` for 256-row pages (a BF16
    pool ``[P, 73728]``), ``k = 4`` for 512-row pages (``[P, 147456]``, the page of an fp32 KDA state). Kernel page
    ``k p + j`` is part ``j`` of page ``p``, so the block table lists ``k`` consecutive kernel pages for each page,
    and runs of 2 (and of 4 at ``k = 4``) hold. ``k = 1`` returns the pool as it is.
    """
    per_page, rest = divmod(fp8_pool.shape[1], kernel_page_width)
    kernel_assert(
        rest == 0 and per_page >= 1,
        f"an FP8 page of {fp8_pool.shape[1]} bytes must hold whole kernel pages of {kernel_page_width} bytes",
    )
    if per_page == 1:
        return fp8_pool
    return fp8_pool.reshape((per_page * fp8_pool.shape[0], kernel_page_width))


def fp8_page_max() -> float:
    """Return the largest finite value of ``fp8_page_dtype``: the clamp bound of an FP8 cache row."""
    return 448.0 if is_trn3() else 240.0


def dma_priority(level: int):
    """Return the DMA QoS priority ``level`` on trn3, or None on trn2, which has no DMA QoS."""
    return level if is_trn3() else None


BANK_DMAS = ("act", "sp")
"""The queues of the AttnRes bank-entry loads of a DRAM bank (``bank_dma`` of ``attn_res_hT`` and
``attn_res_plain_bank_part``): ``act`` the Activation hardware DGE queue, ``sp`` the SP hardware DGE queue, both at the
critical DMA priority. Without one, the compiler picks the queue, and in a verify step the loads can wait behind the
expert-weight stream on a shared queue. Either queue moves the same bytes."""


_STATE_O16_ROWS = 4
"""Rows of each BF16 numerator chunk of ``store_state_bf16_o``."""


class StateParts(NamedTuple):
    """The local MLA state of the BF16-o form in SBUF: ``o`` BF16 numerators, ``sums`` FP32 softmax sums."""

    o: object
    sums: object


def mla_state_bf16_o_views(pair, lora: int, sum_rows: int = 0, sums=None):
    """Return ``(o, sums, rows)``: the BF16-o views of one FP32 MLA state tensor ``[n, rows * (lora + 1)]`` in HBM.

    The BF16-o form of the MLA state reduce-scatter (``state_bf16_o`` of ``mla_block_attention`` and
    ``mla_decode_attention``) keeps the FP32 tensor and splits its bytes: ``o`` ``[n, rows * lora]`` BF16 (the
    numerators) in the first ``n * rows * lora * 2`` bytes, then ``sums`` ``[n, rows]`` FP32 (the softmax sums). Each
    view is contiguous, so each reduce-scatter moves one block, and the sums start on a 1 KiB boundary. ``sum_rows``
    (more than ``rows``) pads each row of ``sums`` to that many values: the sums reduce-scatter then moves at least
    ``sum_rows`` values a row (the padding is never read). 0 (default) keeps ``rows``. ``sums`` (an FP32 tensor
    ``[n, >= max(rows, sum_rows)]``) takes the sums in a tensor of their own, so the two reduce-scatters of the state
    never write views of one tensor; None (default) keeps them in the state tensor.
    """
    n, width = pair.shape
    rows = width // (lora + 1)
    kernel_assert(
        rows * (lora + 1) == width and lora % 2 == 0, f"an MLA state tensor is [n, rows * {lora + 1}], got {pair.shape}"
    )
    flat = pair.reshape((n * width,))
    o_words = n * rows * lora // 2
    sum_width = max(rows, sum_rows)
    if sums is not None:
        kernel_assert(
            sums.dtype == nl.float32 and tuple(sums.shape)[0] == n and tuple(sums.shape)[1] >= sum_width,
            f"the BF16-o sums tensor must be FP32 [{n}, >= {sum_width}], got {sums.dtype} {tuple(sums.shape)}",
        )
        return flat[0:o_words].view(nl.bfloat16).reshape((n, rows * lora)), sums, rows
    kernel_assert(
        o_words + n * sum_width <= n * width, f"{sum_width} padded sums a row do not fit the state tensor {pair.shape}"
    )
    o = flat[0:o_words].view(nl.bfloat16).reshape((n, rows * lora))
    sums = flat[o_words : o_words + n * sum_width].reshape((n, sum_width))
    return o, sums, rows


def store_state_bf16_o(state, o_rows, sum_rows, lora: int, sb, name: str, dma: dict = None):
    """Store the FP32 SBUF MLA state ``[n, R, lora + 1]`` in the BF16-o form.

    VectorE casts the numerators to BF16 in chunks of ``_STATE_O16_ROWS`` rows, which alternate between two tiles, so
    the store of a chunk runs beside the cast of the next. The sums keep their FP32 bits. ``o_rows(r0, r)`` and
    ``sum_rows(r0, r)`` return the destinations of rows ``r0 .. r0 + r - 1``: a list of ``(ap, first, count)``, with
    ``ap`` ``[n, count, lora]`` (BF16) or ``[n, count, 1]`` (FP32) for the rows ``first .. first + count - 1``.
    ``dma`` (None by default) adds ``nisa.dma_copy`` arguments (a priority and a queue) to every store.
    """
    dma = dma or {}
    n, rows, _ = state.shape
    chunk = min(_STATE_O16_ROWS, rows)
    bufs = [sb.tile((n, chunk, lora), nl.bfloat16) for _ in range(1 if rows <= chunk else 2)]
    for i, r0 in enumerate(range(0, rows, chunk)):
        r = min(chunk, rows - r0)
        buf = bufs[i % len(bufs)]
        nisa.tensor_copy(dst=buf[0:n, 0:r, 0:lora], src=state[0:n, r0 : r0 + r, 0:lora], engine=nisa.engine.vector)
        for ap, first, count in o_rows(r0, r):
            nisa.dma_copy(dst=ap, src=buf[0:n, first - r0 : first - r0 + count, 0:lora], name=f"{name}_o{first}", **dma)
    for ap, first, count in sum_rows(0, rows):
        nisa.dma_copy(dst=ap, src=state[0:n, first : first + count, lora : lora + 1], name=f"{name}_sum{first}", **dma)


def load_state_bf16_o(
    o_view,
    sums_view,
    view_rows: int,
    hbm_row,
    local_heads: int,
    rows: int,
    lora: int,
    dense,
    sb,
    name,
    sum_rows=None,
    dma: dict = None,
    sum_aps=None,
):
    """Load the local MLA state of the BF16-o form after its reduce-scatters; return its ``StateParts``.

    ``o_view`` / ``sums_view`` are the destination views of ``mla_state_bf16_o_views`` (``view_rows`` rows a
    tensor row), and ``hbm_row(h)`` is the tensor row of local head ``h``. ``dense`` keeps one row of a head on each
    partition (``o`` ``[local_heads * rows, lora]``, ``sums`` ``[local_heads * rows, 1]``), else the heads are the
    partitions (``[local_heads, rows, lora]`` and ``[local_heads, rows, 1]``). ``sum_rows``: the values a tensor row of
    ``sums_view`` when they are padded (``mla_state_bf16_o_views`` ``sum_rows``); None takes the row of ``sums_view``
    (``view_rows``, or more for a sums tensor of its own with padded columns). ``dma`` (None by default) adds
    ``nisa.dma_copy`` arguments (a priority and a queue) to every load. ``sum_aps`` (None by default) gives the sums
    sources of local head ``h``: ``sum_aps(h)`` returns ``(first row, count, AP)`` runs that load rows
    ``first .. first + count - 1`` of the head in place of its one row of ``sums_view`` (head-major: ``[1, count, 1]``
    APs, one DMA a run into the head's partition, after the numerator load).
    """
    dma = dma or {}
    sum_stride = sums_view.shape[1] if sum_rows is None else sum_rows
    if dense:
        o = sb.tile((local_heads * rows, lora), nl.bfloat16)
        sums = sb.tile((local_heads * rows, 1), nl.float32)
        for h in range(local_heads):
            row = hbm_row(h)
            nisa.dma_copy(
                dst=o[h * rows : (h + 1) * rows, 0:lora],
                src=o_view.ap(pattern=[[lora, rows], [1, lora]], offset=row * view_rows * lora),
                name=f"{name}_o_load_{h}",
                **dma,
            )
            if sum_aps is not None:
                for first, count, ap in sum_aps(h):
                    nisa.dma_copy(
                        dst=sums[h * rows + first : h * rows + first + count, 0:1],
                        src=ap,
                        name=f"{name}_sum_load_{h}_{first}",
                        **dma,
                    )
                continue
            nisa.dma_copy(
                dst=sums[h * rows : (h + 1) * rows, 0:1],
                src=sums_view.ap(pattern=[[1, rows], [1, 1]], offset=row * sum_stride),
                name=f"{name}_sum_load_{h}",
                **dma,
            )
        return StateParts(o, sums)
    kernel_assert(
        all(hbm_row(h) == hbm_row(0) + h for h in range(local_heads)), "the head-major BF16-o load needs adjacent rows"
    )
    o = sb.tile((local_heads, rows, lora), nl.bfloat16)
    sums = sb.tile((local_heads, rows, 1), nl.float32)
    first = hbm_row(0)
    nisa.dma_copy(
        dst=o,
        src=o_view.ap(
            pattern=[[view_rows * lora, local_heads], [lora, rows], [1, lora]], offset=first * view_rows * lora
        ),
        name=f"{name}_o_load",
        **dma,
    )
    if sum_aps is not None:
        for h in range(local_heads):
            for r0, count, ap in sum_aps(h):
                nisa.dma_copy(
                    dst=sums[h : h + 1, r0 : r0 + count, 0:1],
                    src=ap,
                    name=f"{name}_sum_load_{h}_{r0}",
                    **dma,
                )
        return StateParts(o, sums)
    nisa.dma_copy(
        dst=sums,
        src=sums_view.ap(pattern=[[sum_stride, local_heads], [1, rows], [1, 1]], offset=first * sum_stride),
        name=f"{name}_sum_load",
        **dma,
    )
    return StateParts(o, sums)


def state_rows_plain(view, view_rows: int, n: int, lora: int, row0: int = 0, stride: int = 1):
    """Return the ``o_rows`` / ``sum_rows`` of ``store_state_bf16_o`` for rows ``0 ..`` of tensor rows
    ``row0 + h * stride`` (head ``h``): ``lora`` 0 for the sums view, else the numerator view."""
    width = lora if lora else 1

    def rows_of(r0, r):
        return [
            (
                view.ap(
                    pattern=[[stride * view_rows * width, n], [width, r], [1, width]],
                    offset=(row0 * view_rows + r0) * width,
                ),
                r0,
                r,
            )
        ]

    return rows_of


def bank_dma_options(bank_dma: str = None) -> dict:
    """Return the ``nisa.dma_copy`` options of the bank-entry loads for ``bank_dma`` (``BANK_DMAS``), {} for None."""
    if bank_dma is None:
        return {}
    kernel_assert(bank_dma in BANK_DMAS, f"bank_dma must be one of {BANK_DMAS} or None, got {bank_dma!r}")
    engine = nisa.engine.scalar if bank_dma == "act" else nisa.engine.sync
    return {"dge_mode": nisa.dge_mode.hwdge, "engine": engine, "priority": dma_priority(0)}


PAIR_SHARED_CC_DST_SITES = ("moe_out", "moe_routed", "moe_gather", "kda_ar")
"""The collective destinations that ``pair_shared_cc_dsts`` can put in LNC1 shared HBM (``check_pair_shared_cc_dsts``):
``moe_out``, the HBM output all-reduce of the MoE and dense layers (``out_dst_hbm``; the TP32 output all-reduce of the
multi-server MoE body); ``moe_routed``, the HBM routed all-reduce of the MoE layers over the world group
(``routed_dst_hbm``, ``routed32_dst_hbm``); ``moe_gather``, the HBM latent all-gather of the MoE layers over the world
group (``gather_dst_hbm``); and ``kda_ar``, the output all-reduce of the KDA layers over the attention group
(``kda_ar_dst_hbm``). Each group holds both cores of every HBM stack, so the two cores get the same bytes. The Trn2
entries name the same sites on their tensors (those of the Trn2 MoE body, which is not in this package, and
``multinode.tkg_ffn_world_cc.ffn_collective_hbm``): ``moe_out`` the TP32 MoE output all-reduce and the world dense output
all-reduce, ``moe_routed`` the world routed all-reduce, ``moe_gather`` the TP32 shard gather of the router rows, and
``kda_ar`` the KDA all-reduce.

An all-reduce is ordered with the other core of the stack by its data: every output byte holds the input of each core of
the group. The runtime may skip the start handshake of the latent all-gather, whose destination does not overlap the one of
the previous all-gather of its group (the world x gather, or the second destination of the ``gather`` ping-pong, comes
between two latent gathers). Its start is then ordered at distance 2: every writer of a gather writes the destination only
after it completed the previous collective of the group, which needs the contribution of every rank, the other core of the
stack included, and the compiler's unique-tensors proof on each core orders that core's reads of the previous gather before
its own trigger of that collective. That needs both cores to run the same program on one collective stream, with a
non-empty collective of the group, other than a permute or a prepare, that does not touch the destination, before each
gather."""


PAIR_SHARED_SOURCES_KEY = "pair_shared_cc_dsts"
"""The key of the sites in the collective sources of a program with ``pair_shared_cc_dsts`` (absent without them).
The program is compiled with ``--lnc1-with-shared-mem``, which shares every ``nl.shared_hbm`` scratch tensor between the
two cores of an HBM stack, so the bodies that read the key keep their per-core scratch in private HBM
(``tkg_moe._routed_latent_moe_decode``: the routed partial of the TP128 kernels)."""


def pair_shared_program(sources) -> bool:
    """Return True when the collective sources ``sources`` are those of a program with ``pair_shared_cc_dsts``."""
    return bool(sources) and bool(sources.get(PAIR_SHARED_SOURCES_KEY))


def check_pair_shared_cc_dsts(sites, trn2_entry: bool = False) -> tuple:
    """Return the sites of ``pair_shared_cc_dsts`` as a tuple; fail on an unknown or repeated name, and off trn3 unless
    ``trn2_entry``.

    At LNC1 each core of an HBM stack runs its own process and keeps its own copy of each collective destination in
    private HBM. A named destination goes to ``nl.shared_hbm`` instead: one buffer for the two cores, which the
    collective writes once. It needs the neuronx-cc option --lnc1-with-shared-mem (it applies to the whole program, and
    without it the compiler keeps the tensor private) and a runtime with pair-shared HBM. Collective sources always stay
    private. () changes nothing.

    ``trn2_entry``: the caller is a Trn2 entry (``trn2.decode_model_trn2``, ``trn2.verify_model_trn2``), whose sites are
    the Trn2 MoE and feed-forward tensors; it takes the sites on trn2 too. Every other caller takes them on trn3 only.
    """
    sites = tuple(sites)
    kernel_assert(
        all(site in PAIR_SHARED_CC_DST_SITES for site in sites) and len(set(sites)) == len(sites),
        f"pair_shared_cc_dsts {sites} must be distinct names of {PAIR_SHARED_CC_DST_SITES}",
    )
    kernel_assert(
        not sites or trn2_entry or is_trn3(),
        f"pair_shared_cc_dsts {sites} is for trn3 only here; on trn2 only the Trn2 entries take it, else give ()",
    )
    return sites


def pair_shared_buffer(shared: bool):
    """Return the HBM buffer of a collective destination: ``nl.shared_hbm`` when it is pair-shared, else private."""
    return nl.shared_hbm if shared else nl.private_hbm


def check_pair_group(dst: nl.NkiTensor, group, name: str) -> None:
    """Fail when the collective ``name`` writes a shared HBM ``dst`` over a replica group in which some rank r is not in
    the rank list of r ^ 1, the other core of its HBM stack (a strided DP group). A private ``dst`` always passes."""
    if not nl.shared_hbm == dst.buffer:
        return
    rank_lists = getattr(group, "_value", None)
    kernel_assert(rank_lists is not None, f"{name}: a shared HBM destination needs a replica group with rank lists")
    for ranks in rank_lists:
        members = set(ranks)
        apart = [rank for rank in ranks if rank ^ 1 not in members]
        kernel_assert(
            not apart,
            f"{name}: a shared HBM destination needs ranks r and r ^ 1 in one rank list of the replica group (the "
            f"world or TP group); rank {apart[:1]} is apart from its pair core in {ranks[:8]}...",
        )


def new_ones(dtype=nl.float32):
    """Return an all-ones ``[128, 128]`` SBUF tile for cross-partition sums."""
    ones = nl.ndarray((P_MAX, P_MAX), dtype=dtype, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    return ones


def load_rows_hT(src, dst_hT):
    """Load natural rows ``src`` (HBM ``[T, H]``) into the hT tile ``dst_hT`` (SBUF ``[128, T, H/128]``).

    Each row loads as ``[H/128, 128]`` with one contiguous DMA, then one TensorE transpose moves it
    to the hT layout.
    """
    T, H = src.shape
    n_tiles = H // P_MAX
    kernel_assert(n_tiles <= P_MAX, f"H={H} needs at most {P_MAX} hT tiles")
    for t in range(T):
        rows = nl.ndarray((n_tiles, P_MAX), dtype=src.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=rows, src=src.ap(pattern=[[P_MAX, n_tiles], [1, P_MAX]], offset=t * H))
        hT_psum = nl.ndarray((P_MAX, n_tiles), dtype=src.dtype, buffer=nl.psum)
        nisa.nc_transpose(dst=hT_psum, data=rows)
        nisa.tensor_copy(dst=dst_hT[0:P_MAX, t, 0:n_tiles], src=hT_psum)


ROWS_HT_QUEUES = ("sync", "scalar")
"""The DMA queues of ``load_rows_hT_batched`` (the SP and ACT hardware DGE queues), taken in turn by the chunks."""


def load_rows_hT_batched(
    src, dst_hT, src_row0=0, queues=ROWS_HT_QUEUES, stage_bytes=4096, sb=AUTO_SBUF, name="rows_hT"
):
    """Load ``R`` natural rows of ``src`` (HBM ``[N, H]``) from row ``src_row0`` into the hT tile ``dst_hT``
    (SBUF ``[128, R, H/128]``, a tile or a view such as a column range of a wider tile), as ``load_rows_hT``.

    The rows load in column chunks: one DMA a chunk puts ``[R, c * 128]`` in a staging tile of at most ``stage_bytes``
    a partition (each partition reads one contiguous run of a row), then ``c`` TensorE transposes of ``[R, 128]`` blocks
    fill one PSUM tile ``[128, c, R]``, and one copy writes it to ``dst_hT[:, :, chunk]``. Above 128 rows the rows
    go in groups of 128. The chunks take ``queues`` (``"sync"`` / ``"scalar"``) in turn. The values only move, so
    the result is bit-equal to ``load_rows_hT``. ``sb`` (``tkg_sbuf_alloc.SbufTiles``) allocates the staging tiles.
    """
    N, H = src.shape
    _, R, n_tiles = dst_hT.shape
    kernel_assert(H == n_tiles * P_MAX, f"load_rows_hT_batched: H {H} must be {n_tiles} hT tiles of {P_MAX}")
    kernel_assert(0 <= src_row0 and src_row0 + R <= N, f"load_rows_hT_batched: rows {src_row0}..{src_row0 + R} of {N}")
    engines = {"sync": nisa.engine.sync, "scalar": nisa.engine.scalar}
    kernel_assert(
        len(queues) > 0 and all(queue in engines for queue in queues),
        f"load_rows_hT_batched: queues {queues!r} must be names of {tuple(engines)}",
    )
    size = 2 if src.dtype in (nl.bfloat16, nl.float16) else 4
    item = 0
    for g0 in range(0, R, P_MAX):
        rg = min(P_MAX, R - g0)
        # One PSUM bank (2 KB a partition) takes the c blocks of [128, rg].
        per_chunk = max(1, min(n_tiles, stage_bytes // (P_MAX * size), PSUM_BANK_FP32 * 4 // (rg * size)))
        for k0 in range(0, n_tiles, per_chunk):
            c = min(per_chunk, n_tiles - k0)
            stage = sb.tile((rg, c * P_MAX), src.dtype)
            nisa.dma_copy(
                dst=stage,
                src=src.ap(pattern=[[H, rg], [1, c * P_MAX]], offset=(src_row0 + g0) * H + k0 * P_MAX),
                dge_mode=nisa.dge_mode.hwdge,
                engine=engines[queues[item % len(queues)]],
                name=f"{name}_{g0}_{k0}",
            )
            item += 1
            hT_psum = nl.ndarray((P_MAX, c, rg), dtype=src.dtype, buffer=nl.psum)
            for j in range(c):
                nisa.nc_transpose(dst=hT_psum[0:P_MAX, j, 0:rg], data=stage[0:rg, j * P_MAX : (j + 1) * P_MAX])
            nisa.tensor_copy(
                dst=dst_hT[0:P_MAX, g0 : g0 + rg, k0 : k0 + c],
                src=hT_psum.ap(pattern=[[c * rg, P_MAX], [1, rg], [rg, c]], offset=0),
            )


def store_hT_rows(dst, src_hT, sb=AUTO_SBUF, ring=None, waits=None, queues=None):
    """Store the hT tile ``src_hT`` (SBUF ``[128, T, H/128]``) as natural rows ``dst`` (HBM ``[T, H]``).

    ``sb`` (``tkg_sbuf_alloc.SbufTiles``) allocates the SBUF staging tiles. The compiler ends a DMA source's live
    range at the trigger, so a staging tile can lose its bytes to a later instruction while its store still reads
    them. ``ring`` (a tuple of SBUF ``[H/128, 128]`` tiles that live through the kernel) keeps them: row ``t`` stages
    in ``ring[t % len(ring)]``, and its copy waits for the store that read that tile last (row ``t - len(ring)``,
    or the stores of an earlier call in ``waits``, a dict of ring index to DMA handles that the call updates).
    ``queues`` (None, the default: the default queue of every store): a tuple of DMA queues, each a dict of
    ``dma_copy`` keywords (``engine``, ``dge_mode``); row ``t`` stores on ``queues[t % len(queues)]``. The same bytes.
    """
    T, H = dst.shape
    n_tiles = H // P_MAX
    last = {} if waits is None else waits
    for t in range(T):
        rows_psum = nl.ndarray((n_tiles, P_MAX), dtype=src_hT.dtype, buffer=nl.psum)
        nisa.nc_transpose(dst=rows_psum, data=src_hT[0:P_MAX, t, 0:n_tiles])
        slot = None if ring is None else t % len(ring)
        rows = sb.tile((n_tiles, P_MAX), src_hT.dtype) if ring is None else ring[slot]
        copy = nisa.tensor_copy(dst=rows, src=rows_psum)
        if last.get(slot) and hasattr(copy, "depends_on"):
            copy.depends_on(*last[slot])
        queue = {} if queues is None else queues[t % len(queues)]
        store = nisa.dma_copy(dst=dst.ap(pattern=[[P_MAX, n_tiles], [1, P_MAX]], offset=t * H), src=rows, **queue)
        if ring is not None:
            last[slot] = [store]


def load_vector_hT(src, dtype=None):
    """Load an hT-ordered vector weight ``src`` (HBM ``[H]``) as an SBUF ``[128, H/128]`` tile."""
    (H,) = src.shape
    n_tiles = H // P_MAX
    dst = nl.ndarray((P_MAX, n_tiles), dtype=dtype or src.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=dst, src=src.ap(pattern=[[n_tiles, P_MAX], [1, n_tiles]], offset=0))
    return dst


def _broadcast(tile, pattern):
    """Return a view of an SBUF tile with an explicit ``[[stride, count], ...]`` pattern.

    A stride of 0 repeats a dimension, for example a per-token scale over the hidden tiles. ``ap()``
    addresses the storage of the whole tile, so ``tile`` must be a whole tile, not a slice: use it only
    for tiles that the helper allocates.
    """
    return tile.ap(pattern=pattern, offset=0)


class WeightRow(NamedTuple):
    """Row ``index`` of a whole SBUF ``[128, K, n]`` (or ``[128, n]``) weight tile, in a form that can repeat."""

    tile: nl.NkiTensor
    index: int = 0

    def repeat(self, rows: int):
        """Return the ``[128, rows, n]`` view of the row, repeated over ``rows``."""
        free = int(np.prod(self.tile.shape[1:]))
        n_tiles = self.tile.shape[-1]
        return self.tile.ap(pattern=[[free, P_MAX], [0, rows], [1, n_tiles]], offset=self.index * n_tiles)


@contextmanager
def row_sums_order():
    """A scope that orders the row sums (``_row_sums``: the sums of the AttnRes mixes and the norms) after given
    instructions in the schedule (a schedule edge, not a semaphore wait).

    The scope yields ``order_after(instrs, once=False)``: every row sum traced after the call follows ``instrs``, until
    the next call. A mix sums its bank entries before it needs the stream, so the scheduler can otherwise put the bank
    sums of a later mix into an idle gap of the engine, where they delay the next instruction of the critical chain.
    With ``once``, only the first VectorE row-sum instruction traced after the call takes the edges (one edge for
    each call: VectorE runs in program order, and the later sums follow the first one there). ``once_instrs`` are
    more instructions that only that first instruction follows, beside ``instrs`` for every row sum. Without a call, or
    with empty ``instrs`` and ``once_instrs``, no edge is added and the program does not change.
    """

    def order_after(instrs, once=False, once_instrs=()):
        _SUMS_AFTER[:] = () if once else tuple(instrs)
        _SUMS_ONCE["instrs"] = (*(tuple(instrs) if once else ()), *once_instrs)

    previous, previous_once = list(_SUMS_AFTER), _SUMS_ONCE["instrs"]
    _SUMS_AFTER.clear()
    _SUMS_ONCE["instrs"] = ()
    try:
        yield order_after
    finally:
        _SUMS_AFTER[:] = previous
        _SUMS_ONCE["instrs"] = previous_once


def _after_scope(instr):
    """Add the schedule edges of ``row_sums_order`` to ``instr`` (a raw or wrapped VectorE instruction); return
    ``instr``. The ``once`` edges are then spent."""
    edges = (*_SUMS_AFTER, *_SUMS_ONCE["instrs"])
    if edges:
        handle = instr if hasattr(instr, "depends_on") else nisa.NkiInstruction(instr)
        handle.depends_on(*edges, kind=nisa.dependency.schedule)
        _SUMS_ONCE["instrs"] = ()
    return instr


# The instructions that consume the routed output of the last feed-forward body (``ffn_output_order``).
_FFN_OUTPUT = {"active": False, "instrs": []}


@contextmanager
def ffn_output_order(first_block: bool = True):
    """A scope in which a feed-forward body names the instructions that consume its routed output
    (``mark_ffn_output``), and the decode block takes them (``take_ffn_output``) for the ``row_sums_order`` of the
    next layer.

    The MoE bodies name the DMA that stores the routed partial into the source of the routed collective (or that
    collective, when the source is SBUF). The bank sums of the next layer's mixes then follow it in the schedule (a
    schedule edge, not a semaphore wait), so VectorE runs the PSUM copies of the routed partial before those sums.
    ``first_block`` (the first block of a kernel) starts with no instruction; a later block keeps those of the last
    body of the block before, so its first layer follows them too. Outside a scope ``mark_ffn_output`` records
    nothing.
    """
    if first_block:
        _FFN_OUTPUT["instrs"] = []
    previous = _FFN_OUTPUT["active"]
    _FFN_OUTPUT["active"] = True
    try:
        yield
    finally:
        _FFN_OUTPUT["active"] = previous


def mark_ffn_output(*instrs):
    """Record ``instrs`` (instructions that consume a feed-forward body's routed output) inside ``ffn_output_order``;
    outside it do nothing. Return the last of ``instrs``."""
    if _FFN_OUTPUT["active"]:
        _FFN_OUTPUT["instrs"].extend(instr for instr in instrs if instr is not None)
    return instrs[-1] if instrs else None


def take_ffn_output() -> tuple:
    """Return the instructions that ``mark_ffn_output`` recorded since the last call, and forget them."""
    taken = tuple(_FFN_OUTPUT["instrs"])
    _FFN_OUTPUT["instrs"] = []
    return taken


@contextmanager
def row_sums_after_ffn_output(active: bool):
    """With ``active``, the row sums traced in the scope follow the instructions of ``take_ffn_output`` (a
    ``row_sums_order`` scope; for the final mix after the last block). Without it the scope does nothing."""
    if not active:
        yield
        return
    with row_sums_order() as order_after:
        order_after(take_ffn_output(), once=True)
        yield


@contextmanager
def _sums_after(instrs):
    """A scope in which the row sums also follow ``instrs`` in the schedule (beside the ``row_sums_order`` edges)."""
    previous = list(_SUMS_AFTER)
    _SUMS_AFTER[:] = previous + [instr for instr in instrs if instr is not None]
    try:
        yield
    finally:
        _SUMS_AFTER[:] = previous


class ScalarSquares(NamedTuple):
    """The rows of a sum of squares (``_row_sums``) that run on ScalarE: the last ``rows`` rows of each sum.

    ``dst`` is an FP32 ``[128, n]`` SBUF tile for the squares themselves (only their sums are used). The ScalarE rows
    follow the VectorE multiply of the same sum and the instructions ``after`` in the schedule (schedule edges, not
    semaphore waits), so the scheduler cannot move them ahead of the VectorE rows into another engine's chain.
    """

    rows: int
    dst: nl.NkiTensor
    after: tuple = ()


def scalar_squares(rows: int, n_tiles: int, sb=AUTO_SBUF, after=()):
    """Return the ``ScalarSquares`` of ``rows`` rows (a new ``dst`` tile from ``sb``), or None for 0 rows."""
    if not rows:
        return None
    return ScalarSquares(int(rows), sb.tile((P_MAX, n_tiles), nl.float32), tuple(x for x in after if x is not None))


def _split_square_sums(values, out, scratch, act, named):
    """The sums of squares of ``_row_sums`` (R > 1) with the last ``act.rows`` rows on ScalarE.

    VectorE takes the first rows with the multiply and sum of the R > 1 form. ScalarE squares each other row in one
    ``nisa.activation`` with a reduce into a reset accumulator. Each is a plain FP32 sum of the squares in element order,
    so every row gets the bits of the VectorE form (MEASURED on trn3: 0 of 16,384 adversarial bf16 rows of 56 elements
    differ, signed zeros, subnormals, Inf and NaN payloads included). ``square`` is in every activation table set, so
    ScalarE loads no table for these rows. Returns the last instruction.
    """
    _, R, n_tiles = values.shape
    v = R - min(act.rows, R)
    last = None
    # The first ScalarE row follows the VectorE multiply (which carries the row_sums_order edges) or, without one, those
    # edges; the later rows follow it through the write of the shared act.dst tile. One edge for each sum keeps the
    # trace time low (the tracer's cost for an edge grows with the size of the kernel).
    edges = list(act.after)
    if v:
        mul = _after_scope(
            nisa.tensor_tensor(
                dst=scratch[0:P_MAX, 0:v, 0:n_tiles],
                data1=values[0:P_MAX, 0:v, 0:n_tiles],
                data2=values[0:P_MAX, 0:v, 0:n_tiles],
                op=nl.multiply,
                **named("mul"),
            )
        )
        edges.append(mul)
        last = nisa.tensor_reduce(
            dst=out[0:P_MAX, 0:v], data=scratch[0:P_MAX, 0:v, 0:n_tiles], op=nl.add, axis=(2,), **named("sum")
        )
    else:
        edges += [*_SUMS_AFTER, *_SUMS_ONCE["instrs"]]
    for r in range(v, R):
        last = nisa.activation(
            dst=act.dst,
            op=nl.square,
            data=values[0:P_MAX, r, 0:n_tiles],
            reduce_op=nl.add,
            reduce_res=out[0:P_MAX, r : r + 1],
            reduce_cmd=nisa.reduce_cmd.reset_reduce,
            **named(f"act{r}"),
        )
        if edges and r == v:
            handle = last if hasattr(last, "depends_on") else nisa.NkiInstruction(last)
            handle.depends_on(*edges, kind=nisa.dependency.schedule)
    return last


def _row_sums(values, out, scratch, weights=None, w_row=None, uniform=False, name=None, squares_on_act=None):
    """Write the per-partition sums over the last axis of ``values`` (``[128, R, n]``) to ``out``.

    With ``weights`` (``[128, n]``, a whole tile or a slice), sum ``values * weights``; otherwise sum
    ``values**2``. With ``w_row`` (a ``WeightRow`` of the weights) or ``uniform``, every R takes one multiply and
    one sum, so the sum of a row does not depend on R. The work runs on VectorE, so ScalarE switches fewer
    activation tables between the glue and the bodies. With ``squares_on_act`` (a ``ScalarSquares``), the last rows of a
    sum of squares of R > 1 rows run on ScalarE, with the same bits (``_split_square_sums``). ``name`` names the
    instructions (None: unnamed). Returns the last instruction.
    """
    _, R, n_tiles = values.shape

    def named(suffix):
        return {"name": f"{name}_{suffix}"} if name else {}

    if w_row is not None or (weights is None and (R > 1 or uniform)):
        if w_row is None and squares_on_act is not None and squares_on_act.rows and R > 1:
            return _split_square_sums(values, out, scratch, squares_on_act, named)
        data2 = values if w_row is None else w_row.repeat(R)
        _after_scope(nisa.tensor_tensor(dst=scratch, data1=values, data2=data2, op=nl.multiply, **named("mul")))
        return nisa.tensor_reduce(dst=out, data=scratch, op=nl.add, axis=(2,), **named("sum"))
    # One fused multiply and sum for each row. A weight slice cannot take a broadcast pattern.
    last = None
    for r in range(R):
        last = _after_scope(
            nisa.scalar_tensor_tensor(
                dst=scratch[0:P_MAX, r, 0:n_tiles],
                data=values[0:P_MAX, r, 0:n_tiles],
                op0=nl.multiply,
                operand0=1.0,
                op1=nl.multiply,
                operand1=values[0:P_MAX, r, 0:n_tiles] if weights is None else weights,
                reduce_op=nl.add,
                reduce_res=out if R == 1 else out[0:P_MAX, r : r + 1],
                **named(r),
            )
        )
    return last


def _ones_sum(part, ones, done=None):
    """Return ``ones.T @ part`` (the sum over the partitions on every partition) in fp32 PSUM.

    Each matmul writes at most one PSUM bank (512 fp32 columns). ``done``: optional list; the matmuls are appended.
    """
    _, width = part.shape
    tot = nl.ndarray((P_MAX, width), dtype=nl.float32, buffer=nl.psum)
    for c0 in range(0, width, PSUM_BANK_FP32):
        c1 = min(width, c0 + PSUM_BANK_FP32)
        mm = nisa.nc_matmul(
            dst=tot[0:P_MAX, c0:c1], stationary=ones, moving=part[0:P_MAX, c0:c1], is_stationary_onezero=True
        )
        if done is not None:
            done.append(mm)
    return tot


def rmsnorm_hT(
    x_hT,
    w_hT,
    eps,
    ones,
    sb=AUTO_SBUF,
    out=None,
    out16=None,
    w_row=None,
    done=None,
    act_square_rows=0,
    w_mul_row=None,
    one_round=False,
    out_dtype=None,
    w16_row=None,
):
    """Return ``bf16(bf16(x * rsqrt(mean(x^2) + eps)) * w)`` for each token of an hT tile.

    Args:
        x_hT: SBUF ``[128, T, n]`` input.
        w_hT: SBUF ``[128, n]`` weight in hT order (fp32 or bf16), a whole tile or a slice.
        eps: Norm epsilon.
        ones: All-ones fp32 ``[128, 128]`` SBUF tile.
        sb: the allocator of the SBUF tiles (``tkg_sbuf_alloc.SbufTiles``).
        out: optional SBUF ``[128, T, n]`` tile for the result; None allocates it with ``sb``.
        out16: optional FP16 SBUF ``[128, T, n]`` tile that also gets ``fp16(x * rsqrt(mean(x^2) + eps) * w)``
            from the FP32 values, with no BF16 rounding on the way (the input of an FP16 router).
        w_row: optional ``WeightRow`` that holds ``w_hT``: the weight multiply of all tokens is then one instruction,
            and the sums take the form of every T (``_row_sums``), so a token's result does not depend on T.
        done: optional list. The instructions that write ``out`` are appended, so a caller can order other work after
            the norm. None adds nothing.
        act_square_rows: the sums of squares of the last this-many tokens run on ScalarE, beside the VectorE sums of
            the others (``_split_square_sums``): the same bits. 0 (default) keeps the program.
        w_mul_row: optional ``WeightRow`` that holds ``w_hT``, without ``w_row``: only the weight multiply of all
            tokens is one instruction (an elementwise multiply, so the same bits); the sums keep their form. None
            (default) keeps the program.
        one_round: write ``out_dtype(x * rstd * w)`` with one rounding, the FP32 product ``x * rstd`` kept: one fused
            multiply of each token (the GPU form of a norm on an FP32 mix accumulator, ``attn_res_tensor_hT``). Not
            with ``w_row`` or ``w_mul_row``. False (default) keeps the program.
        out_dtype: the dtype of a new ``out`` tile (None: the dtype of ``x_hT``).
        w16_row: optional ``WeightRow`` that holds ``w_hT``, with ``out16``: the weight multiply of the FP16 copy is one
            instruction for all tokens on the repeated weight row, in place of one for each token (an elementwise
            multiply, so the same bits). None (default) keeps the program.

    Returns:
        SBUF ``[128, T, n]`` in the dtype of ``x_hT`` (``out_dtype`` if given).
    """
    _, T, n_tiles = x_hT.shape
    H = n_tiles * P_MAX
    kernel_assert(
        not (one_round and (w_row is not None or w_mul_row is not None)), "one_round takes no w_row or w_mul_row"
    )
    kernel_assert(w16_row is None or out16 is not None, "w16_row is the weight row of the FP16 copy: it needs out16")
    part = sb.tile((P_MAX, T), nl.float32)
    sq = sb.tile((P_MAX, T, n_tiles), nl.float32)
    _row_sums(x_hT, part, sq, uniform=w_row is not None, squares_on_act=scalar_squares(act_square_rows, n_tiles, sb))
    tot = _ones_sum(part, ones)
    rstd = sb.tile((P_MAX, T), nl.float32)
    nisa.activation(dst=rstd, op=nl.rsqrt, data=tot, scale=1.0 / float(H), bias=eps)
    if not one_round:
        normed = sb.tile((P_MAX, T, n_tiles), x_hT.dtype)
    out = sb.tile((P_MAX, T, n_tiles), out_dtype or x_hT.dtype) if out is None else out
    if out16 is not None:
        _rmsnorm_fp16(x_hT, w_hT, rstd, out16, sb, w16_row)
    if one_round:
        # (x * rstd) * w in FP32 inside one instruction, then one rounding to the dtype of out.
        writes = [
            nisa.scalar_tensor_tensor(
                dst=out[0:P_MAX, t, 0:n_tiles],
                data=x_hT[0:P_MAX, t, 0:n_tiles],
                op0=nl.multiply,
                operand0=rstd[0:P_MAX, t : t + 1],
                op1=nl.multiply,
                operand1=w_hT,
            )
            for t in range(T)
        ]
        if done is not None:
            # A raw instruction handle cannot be a schedule producer: wrap it, as for the other done lists.
            done.extend(w if hasattr(w, "depends_on") else nisa.NkiInstruction(w) for w in writes)
        return out
    if T == 1 and w_row is None:
        nisa.tensor_scalar(
            dst=normed[0:P_MAX, 0, 0:n_tiles], data=x_hT[0:P_MAX, 0, 0:n_tiles], op0=nl.multiply, operand0=rstd
        )
        writes = [
            nisa.tensor_tensor(
                dst=out[0:P_MAX, 0, 0:n_tiles], data1=normed[0:P_MAX, 0, 0:n_tiles], data2=w_hT, op=nl.multiply
            )
        ]
    else:
        rstd_b = _broadcast(rstd, [[T, P_MAX], [1, T], [0, n_tiles]])
        nisa.tensor_tensor(dst=normed, data1=x_hT, data2=rstd_b, op=nl.multiply)
        if w_row is not None or w_mul_row is not None:
            row = w_row if w_row is not None else w_mul_row
            writes = [nisa.tensor_tensor(dst=out, data1=normed, data2=row.repeat(T), op=nl.multiply)]
        else:
            # One multiply for each token: a weight slice cannot take a broadcast pattern.
            writes = [
                nisa.tensor_tensor(
                    dst=out[0:P_MAX, t, 0:n_tiles], data1=normed[0:P_MAX, t, 0:n_tiles], data2=w_hT, op=nl.multiply
                )
                for t in range(T)
            ]
    if done is not None:
        done.extend(writes)
    return out


def _rmsnorm_fp16(x_hT, w_hT, rstd, out16, sb=AUTO_SBUF, w16_row=None):
    """Write ``fp16(x * rstd * w)`` with FP32 intermediates (the FP16 output of ``rmsnorm_hT``).

    With ``w16_row`` (a ``WeightRow`` that holds ``w_hT``), the weight multiply of all tokens is one instruction on the
    repeated weight row: each element is the same FP32 product, rounded to FP16 once, so the bits do not change.
    """
    _, T, n_tiles = x_hT.shape
    normed32 = sb.tile((P_MAX, T, n_tiles), nl.float32)
    if T == 1:
        nisa.tensor_scalar(
            dst=normed32[0:P_MAX, 0, 0:n_tiles], data=x_hT[0:P_MAX, 0, 0:n_tiles], op0=nl.multiply, operand0=rstd
        )
    else:
        nisa.tensor_tensor(
            dst=normed32, data1=x_hT, data2=_broadcast(rstd, [[T, P_MAX], [1, T], [0, n_tiles]]), op=nl.multiply
        )
    if w16_row is not None:
        nisa.tensor_tensor(dst=out16, data1=normed32, data2=w16_row.repeat(T), op=nl.multiply)
        return
    for t in range(T):
        nisa.tensor_tensor(
            dst=out16[0:P_MAX, t, 0:n_tiles], data1=normed32[0:P_MAX, t, 0:n_tiles], data2=w_hT, op=nl.multiply
        )


def row_rstd_hT(x_hT, eps, ones, out, sb=AUTO_SBUF):
    """Write ``rsqrt(mean(x^2) + eps)`` of each token of ``x_hT`` (``[128, T, n]``) to ``out`` (fp32 ``[128, T]``).

    A bank entry keeps them from its push for the mixes of ``attn_res_hT`` (``bank_rstd``). The sums take the form
    of every T, so a token's value does not depend on T. That is the form of the bank squares of the plain mix for any
    bank of more than one row, so the kept values are those that the plain mix computes.
    """
    _, T, n_tiles = x_hT.shape
    part = sb.tile((P_MAX, T), nl.float32)
    sq = sb.tile((P_MAX, T, n_tiles), nl.float32)
    _row_sums(x_hT, part, sq, uniform=True)
    tot = _ones_sum(part, ones)
    nisa.activation(dst=out, op=nl.rsqrt, data=tot[0:P_MAX, 0:T], scale=1.0 / float(n_tiles * P_MAX), bias=eps)


class SplitBank:
    """An AttnRes bank ``[128, C, T, n]`` whose first entries sit in SBUF and the others in HBM.

    Entry ``c`` is ``sbuf_part[:, c]`` for ``c < S`` (``S = sbuf_part.shape[1]``) and ``hbm_part[:, c - S]`` after it;
    ``hbm_part`` is None when all entries sit in SBUF. The bank takes the place of a DRAM bank: ``buffer`` is HBM, so
    the mixes of ``attn_res_hT`` take the per-entry forms of a DRAM bank, an HBM push takes the wait of a DRAM bank
    (``decode_block.decode_block_body``), and the options that need an SBUF bank refuse it. A mix reads an SBUF entry in
    place and loads only the HBM entries into its reused tile, so the instructions on the entry values, and the values,
    are those of a DRAM bank. ``bank[0:128, 0:k]`` is the bank of the first ``k`` entries, and ``bank[0:128, c]`` is
    entry ``c`` in its home.
    """

    buffer = nl.private_hbm

    def __init__(self, sbuf_part: nl.NkiTensor, hbm_part: nl.NkiTensor = None, count: int = None):
        _, sbuf_entries, T, n_tiles = sbuf_part.shape
        kernel_assert(sbuf_part.buffer == nl.sbuf, "The first part of a split bank must be an SBUF tile")
        kernel_assert(
            hbm_part is None
            or (
                hbm_part.buffer != nl.sbuf
                and tuple(hbm_part.shape[2:]) == (T, n_tiles)
                and hbm_part.dtype == sbuf_part.dtype
            ),
            f"The second part of a split bank must be a DRAM [{P_MAX}, C, {T}, {n_tiles}] tensor in the bank dtype",
        )
        total = sbuf_entries + (0 if hbm_part is None else hbm_part.shape[1])
        count = total if count is None else count
        kernel_assert(0 < count <= total, f"A split bank view holds 1 to {total} entries, got {count}")
        self.sbuf_part = sbuf_part
        self.hbm_part = hbm_part
        self.sbuf_entries = sbuf_entries
        self.shape = (P_MAX, count, T, n_tiles)
        self.dtype = sbuf_part.dtype

    @property
    def hbm_entries(self) -> int:
        """Return the number of entries of the bank that sit in HBM."""
        return max(0, self.shape[1] - self.sbuf_entries)

    def in_sbuf(self, c: int) -> bool:
        """Return whether entry ``c`` sits in SBUF."""
        return c < self.sbuf_entries

    def __getitem__(self, key):
        parts, entries = key
        kernel_assert(parts == slice(0, P_MAX), f"A split bank takes all {P_MAX} partitions")
        if isinstance(entries, slice):
            kernel_assert(
                entries.start in (None, 0) and entries.step in (None, 1) and entries.stop is not None,
                "A split bank view holds the first entries of the bank",
            )
            return SplitBank(self.sbuf_part, self.hbm_part, entries.stop)
        c = int(entries)
        kernel_assert(0 <= c < self.shape[1], f"Entry {c} is outside the {self.shape[1]} entries of the split bank")
        if self.in_sbuf(c):
            return self.sbuf_part[0:P_MAX, c]
        return self.hbm_part[0:P_MAX, c - self.sbuf_entries]

    def push(self, c: int, src_hT):
        """Write the SBUF ``[128, T, n]`` tile ``src_hT`` to entry ``c``, which sits in SBUF, and return the DMA.

        The push is a DMA copy, as the push of a DRAM bank: the same bytes. No mix reads the entry before its push, so
        the push waits for no mix.
        """
        kernel_assert(self.in_sbuf(c), f"Entry {c} of the split bank sits in HBM")
        return nisa.dma_copy(
            dst=self.sbuf_part[0:P_MAX, c],
            src=src_hT,
            dge_mode=nisa.dge_mode.hwdge,
            engine=nisa.engine.sync,
        )


def attn_res_hT(
    bank_hT,
    stream_hT,
    w_hT,
    eps,
    ones,
    sb=AUTO_SBUF,
    bank_rstd=None,
    w_row=None,
    vector_exp=False,
    read_out=None,
    squares_first=False,
    bank_part=None,
    plain_part=None,
    act_square_rows=0,
    act_bank_square_rows=0,
    act_square_after=(),
    sums_done=None,
    act_done=None,
    act_product_rows=0,
    act_product_stt=False,
    bank_dma: str = None,
    out_dtype=None,
    rstd_out=None,
    act_stream_squares=False,
):
    """Mix the AttnRes candidates of each token in hT layout.

    The candidates are the ``C`` bank entries and then the stream. The score of a candidate ``c`` is
    ``sum(c * rsqrt(mean(c^2) + eps) * w)``. The result is ``bf16(sum_c softmax(score)_c * c)``, with the
    softmax over the candidates of each token.

    Args:
        bank_hT: SBUF ``[128, C, T, n]`` bank in hT layout (entry ``c`` is ``bank_hT[:, c]``), a DRAM bank, a
            ``SplitBank`` (above ``COMPACT_MIX_TOKENS`` rows), or ``None`` for an empty bank.
        stream_hT: SBUF ``[128, T, n]`` stream.
        w_hT: SBUF fp32 ``[128, n]`` score weight (``norm_weight * proj_weight``) in hT order, a whole tile
            or a slice.
        eps: Norm epsilon.
        ones: All-ones fp32 ``[128, 128]`` SBUF tile.
        sb: the allocator of the SBUF tiles (``tkg_sbuf_alloc.SbufTiles``).
        bank_rstd: optional fp32 ``[128, C, T]`` tile view: ``rsqrt(mean(c^2) + eps)`` of each bank entry, kept from its
            push (``row_rstd_hT``). The mix then sums no squares of the bank. With ``w_row``, the ``WeightRow`` that
            holds ``w_hT``, the weighted sums of all tokens of a candidate take one pass, and a token's result does not
            depend on T. Without it, every weighted sum keeps the fused multiply and sum of each row of the plain mix,
            so for T > 1 the result has the bits of the plain mix: only the work changes.
        vector_exp: take the softmax exponentials on VectorE (``nisa.exponential``, NeuronCore-v4), so the mix loads
            no activation table between the rsqrt of its scores and the rsqrt of the next norm. Trn2 keeps ScalarE.
        read_out: optional list. With a DRAM bank, the final mix compute is appended. That compute reads the
            last bank entry (through ``acc``), so a caller can order a later write of the bank (the next entry
            push) after it, a forward edge that still waits for the bank reads to finish before the push can
            overwrite the bank. (A DMA source's live range ends at the trigger, so the push must not start
            until a compute has consumed the read.) None (every caller but the loop/static B >= 4 verify) is
            inert.
        squares_first: with ``bank_rstd``, sum the stream squares and take their rsqrt before the weighted sums, with
            a ones-sum matmul of their own. The rsqrt and the exponential table load of ScalarE then run beside the
            weighted sums of VectorE. Each column of a matmul is a sum of its own moving column, so the values do not
            change. A DRAM bank streams its entries for the bank sums too (``attn_res_bank_part``).
        bank_part: with ``squares_first``, the fp32 ``[128, (C + 1) * T]`` tile of ``attn_res_bank_part``: the weighted
            sums of the bank entries, already done. The mix adds the stream sums in the last T columns.
        plain_part: for the plain mix (no ``bank_rstd``) of a bank above ``COMPACT_MIX_TOKENS`` rows, the
            ``(part, entries)`` of ``attn_res_plain_bank_part``: the square and weighted sums of the first ``entries``
            bank entries, already done in the columns of the mix's part tile. The mix sums the other entries and the
            stream into the same tile, with the same instructions, so the values do not change. None (default) keeps
            the program.
        act_square_rows: for the plain mix of an SBUF bank above ``COMPACT_MIX_TOKENS`` rows, the sum of squares of
            the last this-many rows of the stream runs on ScalarE, beside the VectorE sums of the others
            (``_split_square_sums``): the same bits. 0 (default) keeps the program.
        act_bank_square_rows: the same for the sums of squares of each bank entry. 0 (default) keeps them on VectorE.
        act_square_after: instructions that the ScalarE rows of the bank entries follow in the schedule (a schedule
            edge), for example the collective that the stream waits for, so they do not move into an earlier chain.
        sums_done: optional list. The plain mix appends its ones-sum matmuls, so a caller can order other TensorE work
            after them. None adds nothing.
        act_done: optional list. With ``act_square_rows``, the plain mix appends its last ScalarE square (the last
            stream row, which runs last), so a caller can order other ScalarE work after the squares. None adds
            nothing.
        act_product_rows: for the plain mix of an SBUF bank above ``COMPACT_MIX_TOKENS`` rows with at least 2 entries,
            the products of the last this-many rows of each bank entry after the first run on ScalarE, one
            ``tensor_scalar`` multiply of a row by its softmax weight (a ``[128, 1]`` pointer) each, beside the VectorE
            products of the other rows (``_mix_entries``). VectorE takes the stream term and the first entry first, then
            adds the entries in candidate order, so the FP32 sum order and the bits do not change. The products of an
            entry take one of two tiles, the other being the scratch of the stream sums, which is free after the
            scores. It also acts on a DRAM bank (an entry tile of its own a load) and on the mix of a kept bank rstd
            (``bank_rstd``, also with ``squares_first``). 0 (default) keeps the program.
        act_product_stt: with ``act_product_rows``, each VectorE row of an entry after the first takes one VectorE
            multiply-add (``_mix_entries`` ``stt``) in place of its multiply and its part of the add (the same bits).
        bank_dma: for a DRAM bank, the queue of the bank-entry loads (``BANK_DMAS``); the entries then alternate
            between two SBUF tiles, so the load of the next entry runs beside the work on the current one. The same
            bytes, so the values do not change. None (default) keeps the program.
        out_dtype: the dtype of the mix result. ``nl.float32`` returns the FP32 accumulator ``sum_c p_c * c`` unrounded
            (the last add writes FP32), so a norm with ``rmsnorm_hT`` ``one_round`` rounds the mix and the norm once,
            as on the GPU. None (default) keeps the dtype of the stream and the program. An empty bank returns the
            stream as it is.
        rstd_out: with ``bank_rstd`` and more than one row, an optional fp32 ``[128, T]`` view that gets the rstd of
            the stream (``rsqrt(mean(s^2) + eps)``), for the bank push of the stream. The mix sums the stream squares
            with the instructions of ``row_rstd_hT``, and a second rsqrt of the same sums writes the view, so it holds
            the values of ``row_rstd_hT`` (simulated bitwise at T = 32 and 2; at T = 1 the cached mix with ``w_row``
            differs in the last bit, so one row is refused) and the push needs no sums of its own. None (default)
            keeps the program.
        act_stream_squares: with ``bank_rstd``, ``squares_first``, ``bank_part`` and ``w_row``, ScalarE writes the
            FP32 squares of the bf16 stream and VectorE sums them (``_attn_res_squares_first``): the same exact squares
            and sums, so the bits do not change; VectorE runs the weighted rows of the stream meanwhile. False
            (default) keeps the program.

    Returns:
        SBUF ``[128, T, n]`` in ``out_dtype`` (default the stream's); the stream itself when the bank is empty.
    """
    if bank_hT is None:
        return stream_hT
    _, T, n_tiles = stream_hT.shape
    C = bank_hT.shape[1]
    Cp1 = C + 1
    H = n_tiles * P_MAX
    CT = C * T

    compact = T > COMPACT_MIX_TOKENS
    # A bank in DRAM (the long verify step keeps it there) streams one entry at a time through a reused
    # SBUF tile, so the whole bank need not sit in SBUF beside the attention body. An SBUF bank reads its
    # entries in place, so the mix of a decode step is unchanged.
    hbm_bank = bank_hT.buffer != nl.sbuf
    # A split bank reads its SBUF entries in place too: only its HBM entries take the reused tiles.
    split = isinstance(bank_hT, SplitBank)
    kernel_assert(not split or compact, f"A split bank needs more than {COMPACT_MIX_TOKENS} rows")
    loaded = bank_hT.hbm_entries if split else hbm_bank
    bank_entry_sb = sb.tile((P_MAX, T, n_tiles), bank_hT.dtype) if loaded else None
    entry_dma = bank_dma_options(bank_dma) if loaded else {}
    entry_tiles = [bank_entry_sb]
    if entry_dma:
        entry_tiles.append(sb.tile((P_MAX, T, n_tiles), bank_hT.dtype))
    entry_loads = [0]

    def bank_entry(c):
        if not hbm_bank or (split and bank_hT.in_sbuf(c)):
            return bank_hT[0:P_MAX, c]
        # With bank_dma the loads alternate between two tiles, in load order.
        tile = entry_tiles[entry_loads[0] % len(entry_tiles)]
        entry_loads[0] += 1
        nisa.dma_copy(dst=tile, src=bank_hT[0:P_MAX, c], **entry_dma)
        return tile

    kernel_assert(
        bank_rstd is not None or not (squares_first or bank_part is not None),
        "squares_first and bank_part need the bank rstd (bank_rstd)",
    )
    kernel_assert(plain_part is None or bank_rstd is None, "plain_part is for the plain mix (no bank_rstd)")
    kernel_assert(
        all(
            not r or (bank_rstd is None and compact and not hbm_bank and 0 < r < T)
            for r in (act_square_rows, act_bank_square_rows)
        ),
        f"act_square_rows and act_bank_square_rows are 0 < rows < {T} for the plain mix of an SBUF bank above "
        f"{COMPACT_MIX_TOKENS} rows",
    )
    kernel_assert(
        (sums_done is None and act_done is None) or (bank_rstd is None and not squares_first),
        "sums_done and act_done are for the plain mix (no bank_rstd)",
    )
    kernel_assert(
        not act_product_rows or (compact and 0 < act_product_rows <= T),
        f"act_product_rows is 0 < rows <= {T} above {COMPACT_MIX_TOKENS} rows",
    )
    kernel_assert(
        not act_stream_squares or (bank_rstd is not None and squares_first),
        "act_stream_squares is for the cached mix with squares_first (bank_rstd)",
    )
    kernel_assert(
        rstd_out is None or (bank_rstd is not None and T > 1),
        "rstd_out needs bank_rstd and more than one row: the stream squares then take row_rstd_hT's form",
    )
    if squares_first:
        return _attn_res_squares_first(
            bank_entry,
            bank_hT,
            stream_hT,
            w_hT,
            eps,
            ones,
            sb,
            bank_rstd,
            w_row,
            compact,
            vector_exp,
            bank_part,
            read_out,
            act_product_rows,
            bank_dma,
            out_dtype=out_dtype,
            rstd_out=rstd_out,
            act_stream_squares=act_stream_squares,
            act_product_stt=act_product_stt,
        )
    kernel_assert(bank_part is None, "bank_part needs squares_first")
    if bank_rstd is not None:
        return _attn_res_cached(
            bank_entry,
            bank_hT,
            stream_hT,
            w_hT,
            eps,
            ones,
            sb,
            bank_rstd,
            w_row,
            compact,
            vector_exp,
            read_out,
            act_product_rows,
            out_dtype=out_dtype,
            rstd_out=rstd_out,
            act_product_stt=act_product_stt,
        )
    # Per-partition partial sums of v^2 (part[:, 0]) and of v * w (part[:, 1]), candidate-major.
    first = 0
    if plain_part is None:
        part = sb.tile((P_MAX, 2, Cp1 * T), nl.float32)
    else:
        part, first = plain_part
        kernel_assert(compact, "plain_part needs more than COMPACT_MIX_TOKENS rows")
        kernel_assert(
            tuple(part.shape) == (P_MAX, 2, Cp1 * T) and 0 <= first <= C,
            f"plain_part must be a [{P_MAX}, 2, {Cp1 * T}] tile with at most {C} entries",
        )
    # One dst tile for the ScalarE rows of the mix: they run in program order on ScalarE anyway.
    act_dst = sb.tile((P_MAX, n_tiles), nl.float32) if act_square_rows or act_bank_square_rows else None
    bank_act = ScalarSquares(act_bank_square_rows, act_dst, tuple(act_square_after)) if act_bank_square_rows else None
    stream_act = ScalarSquares(act_square_rows, act_dst) if act_square_rows else None
    # With w_row the weighted sums of all rows of a candidate take one multiply and one sum (``_row_sums``), the
    # same FP32 products added in the same order as the fused multiply and sum of each row.
    wsums = {"w_row": w_row} if w_row is not None else {"weights": w_hT}
    if compact:
        scratch_s = sb.tile((P_MAX, T, n_tiles), nl.float32)
        for c in range(first, C):
            entry_c = bank_entry(c)
            _row_sums(entry_c, part[0:P_MAX, 0, c * T : (c + 1) * T], scratch_s, squares_on_act=bank_act)
            _row_sums(entry_c, part[0:P_MAX, 1, c * T : (c + 1) * T], scratch_s, **wsums)
    else:
        scratch_b = sb.tile((P_MAX, CT, n_tiles), nl.float32)
        scratch_s = sb.tile((P_MAX, T, n_tiles), nl.float32)
        bank_rows = bank_hT.reshape((P_MAX, CT, n_tiles))
        _row_sums(bank_rows, part[0:P_MAX, 0, 0:CT], scratch_b)
        _row_sums(bank_rows, part[0:P_MAX, 1, 0:CT], scratch_b, **wsums)
    stream_sq = _row_sums(stream_hT, part[0:P_MAX, 0, CT : CT + T], scratch_s, squares_on_act=stream_act)
    if act_done is not None and stream_act is not None:
        act_done.append(stream_sq)
    _row_sums(stream_hT, part[0:P_MAX, 1, CT : CT + T], scratch_s, **wsums)
    tot = _ones_sum(part.reshape((P_MAX, 2 * Cp1 * T)), ones, done=sums_done)

    # Scores and the softmax over the candidates of each token, on every partition.
    width = Cp1 * T
    rstd = sb.tile((P_MAX, width), nl.float32)
    nisa.activation(dst=rstd, op=nl.rsqrt, data=tot[0:P_MAX, 0:width], scale=1.0 / float(H), bias=eps)
    scores = sb.tile((P_MAX, Cp1, T), nl.float32)
    nisa.tensor_tensor(
        dst=scores.reshape((P_MAX, width)), data1=tot[0:P_MAX, width : 2 * width], data2=rstd, op=nl.multiply
    )
    if act_product_rows and C > 1:
        return _mix_from_scores(
            scores,
            bank_entry,
            bank_hT,
            stream_hT,
            compact,
            sb,
            vector_exp,
            read_out,
            act_product_rows,
            scratch_s,
            out_dtype=out_dtype,
            act_stt=act_product_stt,
        )
    return _mix_from_scores(
        scores, bank_entry, bank_hT, stream_hT, compact, sb, vector_exp, read_out, out_dtype=out_dtype
    )


def _attn_res_cached(
    bank_entry,
    bank_hT,
    stream_hT,
    w_hT,
    eps,
    ones,
    sb,
    bank_rstd,
    w_row,
    compact,
    vector_exp,
    read_out=None,
    act_product_rows=0,
    act_product_stt=False,
    out_dtype=None,
    rstd_out=None,
):
    """The mix of ``attn_res_hT`` with the bank rstd kept from the pushes, in fewer instructions.

    With ``w_row`` the weighted sums of all rows of a candidate take one multiply and one sum. Without it each row
    takes the fused multiply and sum of the plain mix (a weight slice cannot take a broadcast pattern), and the stream
    squares take the form of the plain mix. ``read_out`` takes the final mix compute, and ``out_dtype`` the dtype of
    the result, as for the plain mix.
    ``act_product_rows`` splits the entry products as in the plain mix (``_mix_entries``); the stream scratch is free
    after the scores, so it is the second product tile.
    """
    _, T, n_tiles = stream_hT.shape
    C = bank_hT.shape[1]
    Cp1 = C + 1
    CT = C * T
    width = Cp1 * T
    # Per-partition partial sums of v * w of every candidate, then v^2 of the stream (the bank keeps its rstd).
    sums = {"w_row": w_row} if w_row is not None else {"weights": w_hT}
    part = sb.tile((P_MAX, width + T), nl.float32)
    scratch_s = sb.tile((P_MAX, T, n_tiles), nl.float32)
    if not compact:
        scratch_b = sb.tile((P_MAX, CT, n_tiles), nl.float32)
        _row_sums(bank_hT.reshape((P_MAX, CT, n_tiles)), part[0:P_MAX, 0:CT], scratch_b, **sums)
    elif bank_hT.buffer != nl.sbuf:
        for c in range(C):
            _row_sums(bank_entry(c), part[0:P_MAX, c * T : (c + 1) * T], scratch_s, **sums)
    else:
        # An SBUF bank sums several entries in each pass, with at most _PASS_BYTES of FP32 scratch per partition.
        step = max(1, _PASS_BYTES // (T * n_tiles * 4))
        scratch_b = sb.tile((P_MAX, min(C, step) * T, n_tiles), nl.float32)
        for c0 in range(0, C, step):
            rows = (min(C, c0 + step) - c0) * T
            _row_sums(
                bank_hT[0:P_MAX, c0 : c0 + rows // T].reshape((P_MAX, rows, n_tiles)),
                part[0:P_MAX, c0 * T : c0 * T + rows],
                scratch_b[0:P_MAX, 0:rows],
                **sums,
            )
    _row_sums(stream_hT, part[0:P_MAX, CT:width], scratch_s, **sums)
    _row_sums(stream_hT, part[0:P_MAX, width : width + T], scratch_s, uniform=w_row is not None)
    tot = _ones_sum(part, ones)
    rstd = sb.tile((P_MAX, width), nl.float32)
    nisa.tensor_copy(dst=rstd[0:P_MAX, 0:CT], src=bank_rstd.reshape((P_MAX, CT)))
    for dst in (rstd[0:P_MAX, CT:width],) + (() if rstd_out is None else (rstd_out,)):
        nisa.activation(
            dst=dst,
            op=nl.rsqrt,
            data=tot[0:P_MAX, width : width + T],
            scale=1.0 / float(n_tiles * P_MAX),
            bias=eps,
        )
    scores = sb.tile((P_MAX, Cp1, T), nl.float32)
    nisa.tensor_tensor(dst=scores.reshape((P_MAX, width)), data1=tot[0:P_MAX, 0:width], data2=rstd, op=nl.multiply)
    if act_product_rows and C > 1:
        return _mix_from_scores(
            scores,
            bank_entry,
            bank_hT,
            stream_hT,
            compact,
            sb,
            vector_exp,
            read_out,
            act_product_rows,
            scratch_s,
            out_dtype=out_dtype,
            act_stt=act_product_stt,
        )
    return _mix_from_scores(
        scores, bank_entry, bank_hT, stream_hT, compact, sb, vector_exp, read_out, out_dtype=out_dtype
    )


def attn_res_bank_part(
    bank_hT,
    w_hT,
    sb=AUTO_SBUF,
    w_row=None,
    name=None,
    after=(),
    bank_dma: str = None,
    out: nl.NkiTensor = None,
    entries: tuple = None,
    in_place: bool = False,
):
    """Return the weighted sums of the bank entries of an ``attn_res_hT`` mix, for its ``bank_part``.

    The sums do not need the stream, so a caller can trace them where VectorE is idle. They take the instructions of
    the cached mix (``bank_rstd``) with the same weights: a fused multiply and sum for each row, or with ``w_row`` one
    multiply and one sum for each pass.

    Args:
        bank_hT: SBUF ``[128, C, T, n]`` bank (a view of its first C entries), or a DRAM bank above
            ``COMPACT_MIX_TOKENS`` rows (the verify bank): its entries then stream through SBUF one at a time, with
            the fused multiply and sum of each row of the cached mix of a DRAM bank.
        w_hT: SBUF fp32 ``[128, n]`` score weight of the mix (a whole tile or a slice).
        sb: the allocator of the tiles (``tkg_sbuf_alloc.SbufTiles``).
        w_row: optional ``WeightRow`` that holds ``w_hT`` (``mix_cache``).
        name: optional name prefix of the sum instructions.
        after: instructions that the sums follow in the schedule (a schedule edge, not a semaphore wait), so the
            scheduler cannot move them into the VectorE chain before those instructions.
        bank_dma: for a DRAM bank, the queue of the entry loads (``BANK_DMAS``), into two alternating tiles; None
            (default) one tile on the queue that the compiler picks.
        out: optional fp32 ``[128, (C + 1) * T]`` tile for the sums, so they can live in a frame other than the one of
            ``sb`` (the scratch tiles); None (default) takes a tile of ``sb``.
        entries: optional ``(c0, c1)``, DRAM bank only: sum only the entries ``c0 .. c1 - 1`` (into their columns of
            the part), so two calls on one ``out`` can split the entries; None (default) sums all C.
        in_place: DRAM bank only: the entries of a ``SplitBank`` that sit in SBUF are read in place, not loaded into
            the entry tile (the same values); False (default) loads every entry.

    Returns:
        fp32 ``[128, (C + 1) * T]``: the per-partition sums of row ``c * T + t`` in column ``c * T + t``; the last T
        columns are for the stream sums of the mix.
    """
    _, C, T, n_tiles = bank_hT.shape
    part = sb.tile((P_MAX, (C + 1) * T), nl.float32) if out is None else out
    kernel_assert(
        tuple(part.shape) == (P_MAX, (C + 1) * T) and part.dtype == nl.float32,
        f"attn_res_bank_part out must be fp32 [{P_MAX}, {(C + 1) * T}]",
    )
    sums = {"w_row": w_row} if w_row is not None else {"weights": w_hT}
    if bank_hT.buffer != nl.sbuf:
        kernel_assert(
            T > COMPACT_MIX_TOKENS, f"attn_res_bank_part of a DRAM bank needs more than {COMPACT_MIX_TOKENS} rows"
        )
        c0, c1 = (0, C) if entries is None else entries
        kernel_assert(0 <= c0 < c1 <= C, f"attn_res_bank_part entries must be (c0, c1) with 0 <= c0 < c1 <= {C}")
        resident = in_place and isinstance(bank_hT, SplitBank)
        loads = [c for c in range(c0, c1) if not (resident and bank_hT.in_sbuf(c))]
        tiles = [sb.tile((P_MAX, T, n_tiles), bank_hT.dtype) for _ in range(2 if bank_dma else 1)] if loads else []
        scratch = sb.tile((P_MAX, T, n_tiles), nl.float32)
        entry_dma = bank_dma_options(bank_dma)
        with _sums_after(after):
            for c in range(c0, c1):
                if c in loads:
                    entry = tiles[loads.index(c) % len(tiles)]
                    nisa.dma_copy(dst=entry, src=bank_hT[0:P_MAX, c], **entry_dma)
                else:
                    entry = bank_hT[0:P_MAX, c]
                _row_sums(
                    entry, part[0:P_MAX, c * T : (c + 1) * T], scratch, **sums, name=f"{name}_c{c}" if name else None
                )
        return part
    kernel_assert(entries is None and not in_place, "attn_res_bank_part entries / in_place take a DRAM bank")
    with _sums_after(after):
        if T <= COMPACT_MIX_TOKENS:
            scratch_b = sb.tile((P_MAX, C * T, n_tiles), nl.float32)
            _row_sums(bank_hT.reshape((P_MAX, C * T, n_tiles)), part[0:P_MAX, 0 : C * T], scratch_b, **sums, name=name)
            return part
        # As the cached mix of an SBUF bank: several entries in each pass, at most _PASS_BYTES of FP32 scratch.
        step = max(1, _PASS_BYTES // (T * n_tiles * 4))
        scratch_b = sb.tile((P_MAX, min(C, step) * T, n_tiles), nl.float32)
        for c0 in range(0, C, step):
            rows = (min(C, c0 + step) - c0) * T
            _row_sums(
                bank_hT[0:P_MAX, c0 : c0 + rows // T].reshape((P_MAX, rows, n_tiles)),
                part[0:P_MAX, c0 * T : c0 * T + rows],
                scratch_b[0:P_MAX, 0:rows],
                **sums,
                name=f"{name}_c{c0}" if name else None,
            )
    return part


def attn_res_plain_bank_part(
    bank_hT,
    w_hT,
    entries,
    sb=AUTO_SBUF,
    name=None,
    after=(),
    bank_dma: str = None,
    w_row: WeightRow = None,
    out: nl.NkiTensor = None,
):
    """Return the ``plain_part`` of a plain ``attn_res_hT`` mix: the sums of its first ``entries`` bank entries.

    The sums do not need the stream, so a caller can trace them where VectorE is idle. Each entry takes the instructions
    of the plain mix (a multiply and a sum of its squares, a fused multiply and sum for each weighted row) with the same
    weights, into the columns of the mix's part tile, so the mix gives the same bits. An entry of a DRAM bank (the
    verify above 8 rows) first loads into one of two SBUF tiles, so the load of the next entry runs beside the sums.

    Args:
        bank_hT: SBUF, DRAM or split (``SplitBank``) ``[128, C, T, n]`` bank view of the mix (its C entries), T above
            ``COMPACT_MIX_TOKENS``.
        w_hT: SBUF fp32 ``[128, n]`` score weight of the mix (a whole tile or a slice).
        entries: the number of first entries to sum; at most C are summed.
        sb: the allocator of the tiles (``tkg_sbuf_alloc.SbufTiles``).
        name: optional name prefix of the sum instructions.
        after: instructions that the sums follow in the schedule (a schedule edge, not a semaphore wait).
        bank_dma: the queue of the entry loads of a DRAM bank (``BANK_DMAS``); None leaves it to the compiler.
        w_row: optional ``WeightRow`` that holds ``w_hT``: the weighted sums of each entry then take one multiply and
            one sum, as the plain mix with ``w_row`` (the same bits as a fused multiply and sum for each row).
        out: optional fp32 ``[128, 2, (C + 1) * T]`` tile for the part, so it can live in a frame other than the one of
            ``sb`` (the scratch and load tiles); None (default) takes a tile of ``sb``.

    Returns:
        ``(part, entries)``: the fp32 ``[128, 2, (C + 1) * T]`` part tile of the mix and the number of entries summed.
    """
    _, C, T, n_tiles = bank_hT.shape
    kernel_assert(T > COMPACT_MIX_TOKENS, f"attn_res_plain_bank_part needs more than {COMPACT_MIX_TOKENS} rows")
    entries = min(int(entries), C)
    part = sb.tile((P_MAX, 2, (C + 1) * T), nl.float32) if out is None else out
    kernel_assert(
        tuple(part.shape) == (P_MAX, 2, (C + 1) * T) and part.dtype == nl.float32,
        f"attn_res_plain_bank_part out must be fp32 [{P_MAX}, 2, {(C + 1) * T}]",
    )
    scratch = sb.tile((P_MAX, T, n_tiles), nl.float32)
    hbm_bank = bank_hT.buffer != nl.sbuf
    # A split bank reads its SBUF entries in place: only its HBM entries load.
    split = isinstance(bank_hT, SplitBank)
    loads = entries > bank_hT.sbuf_entries if split else hbm_bank
    loaded = [sb.tile((P_MAX, T, n_tiles), bank_hT.dtype) for _ in range(2)] if loads else None
    with _sums_after(after):
        for c in range(entries):
            entry_c = bank_hT[0:P_MAX, c]
            if hbm_bank and not (split and bank_hT.in_sbuf(c)):
                nisa.dma_copy(
                    dst=loaded[c % 2],
                    src=entry_c,
                    **({"name": f"{name}_load{c}"} if name else {}),
                    **bank_dma_options(bank_dma),
                )
                entry_c = loaded[c % 2]
            _row_sums(entry_c, part[0:P_MAX, 0, c * T : (c + 1) * T], scratch, name=f"{name}_sq{c}" if name else None)
            _row_sums(
                entry_c,
                part[0:P_MAX, 1, c * T : (c + 1) * T],
                scratch,
                **({"w_row": w_row} if w_row is not None else {"weights": w_hT}),
                name=f"{name}_w{c}" if name else None,
            )
    return part, entries


def _attn_res_squares_first(
    bank_entry,
    bank_hT,
    stream_hT,
    w_hT,
    eps,
    ones,
    sb,
    bank_rstd,
    w_row,
    compact,
    vector_exp,
    bank_part,
    read_out=None,
    act_product_rows=0,
    bank_dma=None,
    out_dtype=None,
    rstd_out=None,
    act_stream_squares=False,
    act_product_stt=False,
):
    """The cached mix of ``attn_res_hT`` (``bank_rstd``) with the stream squares, their sum and rsqrt first.

    The instructions are those of ``_attn_res_cached``. Only the order changes, and the ones-sum matmul splits into the
    squares and the weighted sums; each output column of a matmul sums its own moving column, so the values are equal.
    ScalarE then takes the rsqrt (and the load of the exponential table after it) while VectorE sums the weighted rows.
    ``out_dtype``: the dtype of the result, as for the plain mix.
    A DRAM bank streams its entries for the bank sums (``attn_res_bank_part``, on ``bank_dma``) and again for the mix.
    ``read_out`` and ``act_product_rows`` act as in the cached mix; the stream scratch is free after the scores, so it is
    the second product tile.
    ``act_stream_squares`` (with ``bank_part`` and ``w_row``): ScalarE writes the FP32 squares of the stream into the
    squares scratch while VectorE multiplies the stream by the weight row; VectorE then sums the squares and the weighted
    rows with the tensor_reduce of the VectorE form. A square of a bf16 value is exact in FP32, so the squares, their sums
    and every later value keep their bits (finite values); VectorE does not run the squares' multiply pass.
    """
    _, T, n_tiles = stream_hT.shape
    C = bank_hT.shape[1]
    Cp1 = C + 1
    CT = C * T
    width = Cp1 * T
    sums = {"w_row": w_row} if w_row is not None else {"weights": w_hT}
    squares = sb.tile((P_MAX, T), nl.float32)
    scratch_q = sb.tile((P_MAX, T, n_tiles), nl.float32)
    if act_stream_squares:
        kernel_assert(
            bank_part is not None and w_row is not None and T > 1 and stream_hT.dtype == nl.bfloat16,
            "act_stream_squares needs bank_part, w_row, more than one row and a bf16 stream",
        )
        kernel_assert(tuple(bank_part.shape) == (P_MAX, width), f"bank_part must be [{P_MAX}, {width}]")
        vector = nisa.engine.vector
        nisa.activation(dst=scratch_q, op=nl.square, data=stream_hT, engine=nisa.engine.scalar)
        scratch_s = sb.tile((P_MAX, T, n_tiles), nl.float32)
        _after_scope(
            nisa.tensor_tensor(dst=scratch_s, data1=stream_hT, data2=w_row.repeat(T), op=nl.multiply, engine=vector)
        )
        # tensor_reduce takes no engine argument; it runs on VectorE, as in _row_sums.
        squares_done = nisa.tensor_reduce(dst=squares, data=scratch_q, op=nl.add, axis=(2,))
        nisa.tensor_reduce(dst=bank_part[0:P_MAX, CT:width], data=scratch_s, op=nl.add, axis=(2,))
    else:
        squares_done = _row_sums(stream_hT, squares, scratch_q, uniform=w_row is not None)
    tot_q = _ones_sum(squares, ones)
    rstd = sb.tile((P_MAX, width), nl.float32)
    for dst in (rstd[0:P_MAX, CT:width],) + (() if rstd_out is None else (rstd_out,)):
        nisa.activation(dst=dst, op=nl.rsqrt, data=tot_q[0:P_MAX, 0:T], scale=1.0 / float(n_tiles * P_MAX), bias=eps)
    # The scheduler would otherwise put weighted rows ahead of the squares (and so ahead of the rsqrt).
    if bank_part is None:
        bank_part = attn_res_bank_part(bank_hT, w_hT, sb, w_row, after=(squares_done,), bank_dma=bank_dma)
    kernel_assert(tuple(bank_part.shape) == (P_MAX, width), f"bank_part must be [{P_MAX}, {width}]")
    if not act_stream_squares:
        scratch_s = sb.tile((P_MAX, T, n_tiles), nl.float32)
        with _sums_after((squares_done,)):
            _row_sums(stream_hT, bank_part[0:P_MAX, CT:width], scratch_s, **sums)
    tot = _ones_sum(bank_part, ones)
    nisa.tensor_copy(dst=rstd[0:P_MAX, 0:CT], src=bank_rstd.reshape((P_MAX, CT)))
    scores = sb.tile((P_MAX, Cp1, T), nl.float32)
    nisa.tensor_tensor(dst=scores.reshape((P_MAX, width)), data1=tot[0:P_MAX, 0:width], data2=rstd, op=nl.multiply)
    if act_product_rows and C > 1:
        return _mix_from_scores(
            scores,
            bank_entry,
            bank_hT,
            stream_hT,
            compact,
            sb,
            vector_exp,
            read_out,
            act_product_rows,
            scratch_s,
            out_dtype=out_dtype,
            act_stt=act_product_stt,
        )
    return _mix_from_scores(
        scores, bank_entry, bank_hT, stream_hT, compact, sb, vector_exp, read_out, out_dtype=out_dtype
    )


def attn_res_tensor_weights(w_hT, sb=AUTO_SBUF):
    """Return the TensorE form of an AttnRes score weight: the bf16 ``[128, n, 2]`` split ``(hi, lo)`` of ``w_hT``.

    ``hi`` is ``w`` rounded to bf16 and ``lo`` is the remainder ``w - hi`` rounded to bf16, so ``hi + lo`` holds 16
    significant bits of ``w``: all the bits of a product of two bf16 values (the host fold ``norm_weight * proj_weight``),
    and ``w`` to 2^-17 otherwise. Tile ``j`` of the split, ``[128, 2]``, is the stationary of the weighted sums of
    ``attn_res_tensor_hT``; the products of a bf16 row with each part are exact.

    Args:
        w_hT: SBUF fp32 ``[128, n]`` score weight in hT order (a whole tile or a slice).
        sb: the allocator of the tile (``tkg_sbuf_alloc.SbufTiles``).

    Returns:
        SBUF bf16 ``[128, n, 2]``.
    """
    _, n_tiles = w_hT.shape
    split = sb.tile((P_MAX, n_tiles, 2), nl.bfloat16)
    nisa.tensor_copy(dst=split[0:P_MAX, 0:n_tiles, 0], src=w_hT, engine=nisa.engine.vector)
    nisa.tensor_tensor(
        dst=split[0:P_MAX, 0:n_tiles, 1],
        data1=w_hT,
        data2=split[0:P_MAX, 0:n_tiles, 0],
        op=nl.subtract,
        engine=nisa.engine.vector,
    )
    return split


def _tensor_sums(dst, rows_of, w_split, after=()):
    """Accumulate ``sum_h v_h * w_h`` of the rows ``rows_of(j)`` (``[128, ...]`` bf16 views of hidden tile ``j``) into
    the fp32 PSUM ``dst`` (``[2, ...]``: the sums with ``hi`` in row 0 and with ``lo`` in row 1), one matmul for each
    hidden tile. The first matmul follows ``after`` in the schedule. Returns the last matmul."""
    n_tiles = w_split.shape[1]
    last = None
    for j in range(n_tiles):
        last = nisa.nc_matmul(dst=dst, stationary=w_split[0:P_MAX, j, 0:2], moving=rows_of(j), accumulate=j > 0)
        if j == 0 and after:
            handle = last if hasattr(last, "depends_on") else nisa.NkiInstruction(last)
            handle.depends_on(*after, kind=nisa.dependency.schedule)
    return last


def attn_res_tensor_hT(
    bank_hT,
    stream_hT,
    w_split,
    eps,
    ones,
    sb=AUTO_SBUF,
    bank_rstd=None,
    bank_after=(),
    act_square_rows=0,
    act_product_rows=0,
    act_product_stt=False,
    rstd_out=None,
    sums_done=None,
    act_done=None,
    read_out=None,
    bank_dma: str = None,
):
    """The AttnRes mix of ``attn_res_hT`` with its weighted sums on TensorE and FP32 compute, the GPU form.

    The scores and the softmax are those of ``attn_res_hT``, with three changes:

    - the weighted sums ``sum_h v_h w_h`` of every candidate run on TensorE: for each hidden tile ``j`` one matmul
      with the ``[128, 2]`` split of ``w`` (``attn_res_tensor_weights``) as the stationary and the hT rows of the
      candidates as the moving operand, so the products are exact and the sums are FP32 (over the 128 partitions in
      the array, over the tiles in PSUM). A ``K = 2`` matmul with an all-ones stationary then adds ``hi`` and ``lo``
      and puts the sums on every partition. The bank rows go first, in one group that needs no stream, so TensorE can
      run it in an idle window; the stream rows follow, beside the VectorE / ScalarE sums of the stream squares;
    - the bank entries keep their rstd from the push (``bank_rstd``), so the mix sums no squares of the bank;
    - the result is the FP32 accumulator ``sum_c p_c * c``, not rounded to bf16: the next norm takes it
      (``rmsnorm_hT`` ``one_round``), so the mix and its norm round once, as on the GPU.

    The products of the entry mix keep the instructions of ``attn_res_hT`` (``act_product_rows`` puts the last rows of
    the entries after the first on ScalarE).

    A DRAM bank (the verify above 8 or 16 rows) loads its entries one at a time into two SBUF tiles in turn, so the load
    of the next entry runs beside the matmuls (and later the products) of the current one, and runs one group of
    matmuls for each entry. Each output column of a matmul sums its own moving column, so an entry gets the same bits
    as in the one group of an SBUF bank. The entry mix reads the entries again in candidate order; an entry that is
    still in one of the two tiles is not loaded again (the whole bank when it has one or two entries). A split bank
    (the verify bank with its first entries in SBUF, ``_sbuf_part``) reads those entries in place, in one group of
    matmuls, and loads only the others.

    Args:
        bank_hT: SBUF ``[128, C, T, n]`` bank, a DRAM bank of that shape, or a split bank, C >= 1, T above
            ``COMPACT_MIX_TOKENS``.
        stream_hT: SBUF ``[128, T, n]`` stream.
        w_split: SBUF bf16 ``[128, n, 2]`` split of the score weight (``attn_res_tensor_weights``).
        eps: Norm epsilon.
        ones: All-ones fp32 ``[128, 128]`` SBUF tile.
        sb: the allocator of the SBUF tiles (``tkg_sbuf_alloc.SbufTiles``).
        bank_rstd: fp32 ``[128, C, T]`` view: ``rsqrt(mean(c^2) + eps)`` of each bank entry, kept from its push.
        bank_after: instructions that the bank-row matmuls follow in the schedule (a schedule edge), so they do not
            move into an earlier TensorE chain.
        act_square_rows: the sums of squares of the last this-many stream rows run on ScalarE (``_split_square_sums``).
        act_product_rows: as ``attn_res_hT``: the products of the last this-many rows of each bank entry after the
            first run on ScalarE (C >= 2).
        act_product_stt: with ``act_product_rows``, each VectorE row of an entry after the first takes one VectorE
            multiply-add (``_mix_entries`` ``stt``) in place of its multiply and its part of the add (the same bits).
        rstd_out: optional fp32 ``[128, T]`` view that gets the rstd of the stream (the values of ``row_rstd_hT``), for
            the bank push of the stream.
        sums_done: optional list. The matmuls of the stream (its weighted sums, the sum of hi and lo, the ones-sum of
            its squares) are appended.
        act_done: optional list. With ``act_square_rows``, the last ScalarE square is appended.
        read_out: optional list, as ``attn_res_hT``: the final mix compute is appended (it reads the last bank entry
            through the accumulator), so the push of a DRAM bank can wait for it.
        bank_dma: the queue of the entry loads of a DRAM or split bank (``BANK_DMAS``); None takes the SP hardware DGE
            queue.

    Returns:
        SBUF fp32 ``[128, T, n]``: ``sum_c softmax(score)_c * c``.
    """
    _, T, n_tiles = stream_hT.shape
    C = bank_hT.shape[1]
    Cp1 = C + 1
    CT = C * T
    width = Cp1 * T
    H = n_tiles * P_MAX
    kernel_assert(
        C >= 1 and T > COMPACT_MIX_TOKENS, f"attn_res_tensor_hT needs a bank and more than {COMPACT_MIX_TOKENS} rows"
    )
    # The first in_place entries sit in SBUF (all of an SBUF bank, none of a DRAM bank); the others load.
    sbuf_part, in_place = _sbuf_part(bank_hT)
    kernel_assert(bank_rstd is not None, "attn_res_tensor_hT needs the bank rstd (bank_rstd)")
    kernel_assert(
        not (in_place < C and act_product_rows), "act_product_rows is for an SBUF bank: a DRAM bank loads its entries"
    )
    kernel_assert(
        tuple(w_split.shape) == (P_MAX, n_tiles, 2) and w_split.dtype == nl.bfloat16,
        f"w_split must be a bf16 [{P_MAX}, {n_tiles}, 2] tile",
    )
    kernel_assert(not act_product_rows or 0 < act_product_rows <= T, f"act_product_rows is 0 < rows <= {T}")
    kernel_assert(not act_square_rows or 0 < act_square_rows < T, f"act_square_rows is 0 < rows < {T}")
    # The weighted sums of the bank rows, then of the stream rows, on TensorE ([2, C, T] and [2, T] fp32 PSUM).
    bank_sums = nl.ndarray((2, C, T), dtype=nl.float32, buffer=nl.psum)
    if in_place == C and sbuf_part is bank_hT:
        # An SBUF bank: one group of matmuls over all its entries.
        def bank_entry(c):
            return bank_hT[0:P_MAX, c]

        _tensor_sums(bank_sums, lambda j: bank_hT[0:P_MAX, 0:C, 0:T, j], w_split, after=bank_after)
    else:
        if in_place:
            # The SBUF entries of a split bank: one group of matmuls over them, in place.
            _tensor_sums(
                bank_sums[0:2, 0:in_place, 0:T],
                lambda j: sbuf_part[0:P_MAX, 0:in_place, 0:T, j],
                w_split,
                after=bank_after,
            )
        loads = _EntryLoads(bank_hT, sb, bank_dma) if in_place < C else None

        def bank_entry(c):
            return bank_hT[0:P_MAX, c] if c < in_place else loads.entry(c)

        for c in range(in_place, C):
            entry = bank_entry(c)
            _tensor_sums(
                bank_sums[0:2, c, 0:T],
                lambda j, entry=entry: entry[0:P_MAX, 0:T, j],
                w_split,
                after=bank_after if c == in_place == 0 else (),
            )
    # The stream squares on VectorE / ScalarE, beside the stream rows on TensorE.
    squares = sb.tile((P_MAX, T), nl.float32)
    scratch = sb.tile((P_MAX, T, n_tiles), nl.float32)
    stream_act = scalar_squares(act_square_rows, n_tiles, sb)
    squares_done = _row_sums(stream_hT, squares, scratch, squares_on_act=stream_act)
    if act_done is not None and stream_act is not None:
        act_done.append(squares_done)
    stream_sums = nl.ndarray((2, T), dtype=nl.float32, buffer=nl.psum)
    stream_mm = _tensor_sums(stream_sums, lambda j: stream_hT[0:P_MAX, 0:T, j], w_split)
    parts = sb.tile((2, width), nl.float32)
    nisa.tensor_copy(dst=parts[0:2, 0:CT], src=bank_sums.reshape((2, CT)), engine=nisa.engine.vector)
    nisa.tensor_copy(dst=parts[0:2, CT:width], src=stream_sums, engine=nisa.engine.vector)
    # hi + lo of every candidate on every partition (fp32 moving, 0/1 stationary: exact products, one FP32 sum).
    weighted = nl.ndarray((P_MAX, width), dtype=nl.float32, buffer=nl.psum)
    total_mm = nisa.nc_matmul(
        dst=weighted, stationary=ones[0:2, 0:P_MAX], moving=parts[0:2, 0:width], is_stationary_onezero=True
    )
    tot_q = _ones_sum(squares, ones, done=sums_done)
    if sums_done is not None:
        sums_done.extend([stream_mm, total_mm])
    rstd = sb.tile((P_MAX, width), nl.float32)
    nisa.tensor_copy(dst=rstd[0:P_MAX, 0:CT], src=bank_rstd.reshape((P_MAX, CT)), engine=nisa.engine.vector)
    nisa.activation(
        dst=rstd[0:P_MAX, CT:width],
        op=nl.rsqrt,
        data=tot_q[0:P_MAX, 0:T],
        scale=1.0 / float(H),
        bias=eps,
        engine=nisa.engine.scalar,
    )
    if rstd_out is not None:
        nisa.tensor_copy(dst=rstd_out, src=rstd[0:P_MAX, CT:width], engine=nisa.engine.vector)
    scores = sb.tile((P_MAX, Cp1, T), nl.float32)
    nisa.tensor_tensor(
        dst=scores.reshape((P_MAX, width)), data1=weighted, data2=rstd, op=nl.multiply, engine=nisa.engine.vector
    )

    act_rows = act_product_rows if act_product_rows and C > 1 else 0
    return _mix_from_scores(
        scores,
        bank_entry,
        bank_hT,
        stream_hT,
        True,
        sb,
        False,
        read_out,
        act_rows,
        scratch,
        out_dtype=nl.float32,
        act_stt=bool(act_product_stt and act_rows),
    )


def _sbuf_part(bank_hT):
    """Return ``(part, count)`` of an AttnRes bank: the SBUF tile that holds its first ``count`` entries in place.

    An SBUF bank is its own part (all its entries), a DRAM bank has none (``(None, 0)``), and a split bank (the verify
    bank whose first entries sit in SBUF and the others in DRAM: the DRAM bank interface plus ``sbuf_part``, an SBUF
    ``[128, S, T, n]`` tile of the first ``S`` entries) gives ``sbuf_part`` and the first ``min(S, C)`` entries.
    """
    C = bank_hT.shape[1]
    if bank_hT.buffer == nl.sbuf:
        return bank_hT, C
    part = getattr(bank_hT, "sbuf_part", None)
    if part is None:
        return None, 0
    kernel_assert(part.buffer == nl.sbuf, "The sbuf_part of a split AttnRes bank must be an SBUF tile")
    return part, min(C, part.shape[1])


class _EntryLoads:
    """The entry loads of a DRAM AttnRes bank for ``attn_res_tensor_hT``: two SBUF ``[128, T, n]`` tiles in turn.

    ``entry(c)`` returns an SBUF tile with entry ``c``: the tile that holds it when one of the two still does, else the
    tile used less recently, after a DMA of the entry into it. The tile framework orders each load after the readers
    of the entry before it in that tile.
    """

    def __init__(self, bank_hT, sb, bank_dma: str = None):
        _, _, T, n_tiles = bank_hT.shape
        self.bank_hT = bank_hT
        self.tiles = [sb.tile((P_MAX, T, n_tiles), bank_hT.dtype) for _ in range(2)]
        self.held = [None, None]
        self.recent = 1
        self.dma = (
            bank_dma_options(bank_dma) if bank_dma else {"dge_mode": nisa.dge_mode.hwdge, "engine": nisa.engine.sync}
        )

    def entry(self, c: int):
        """Return the SBUF tile of entry ``c`` (see the class)."""
        if c in self.held:
            self.recent = self.held.index(c)
            return self.tiles[self.recent]
        self.recent = 1 - self.recent
        nisa.dma_copy(dst=self.tiles[self.recent], src=self.bank_hT[0:P_MAX, c], **self.dma)
        self.held[self.recent] = c
        return self.tiles[self.recent]


ATTN_RES_AHEAD_CHUNKS = 4
"""Token chunks of the combine of ``attn_res_ahead_hT``. The GpSimd and VectorE products, the TensorE add and the
ScalarE copy of one chunk run beside those of the next chunk."""
ATTN_RES_AHEAD_STATS = ("products", "gram")
"""The forms of the stream sums of ``attn_res_ahead_hT``: "products" (ScalarE / GpSimd products, then a TensorE
ones-accumulate over the hidden tiles, and the TensorE hi / lo sums of ``s . w``) or "gram" (TensorE only: with each
hidden tile of the stream as the stationary, one matmul each for ``s s^T``, ``s B^T`` and ``s . w``; the diagonals are
the sums)."""
ATTN_RES_AHEAD_COMBINES = ("tensor", "vector")
"""The engines of the combine add of ``attn_res_ahead_hT``: "tensor" (a TensorE identity accumulate in PSUM, then a
ScalarE copy) or "vector" (a VectorE add)."""
ATTN_RES_NOMAX_LIMIT = 80.0
"""The largest stream exponent bound ``2 sqrt(H) ||w||`` of the no-max form of ``attn_res_ahead_hT``. Below it,
``exp(q_s - M)`` cannot overflow FP32 (whose exp overflows at 88.7)."""


class AheadPart(NamedTuple):
    """The ahead part of an AttnRes mix (``attn_res_ahead_part``), for ``attn_res_ahead_hT``.

    For the C bank entries ``v_c`` of the mix and their scores ``q_c`` (``attn_res_hT``), per token:
    ``bank`` (bf16 ``[128, T, n]``) is ``B = sum_c exp(q_c - M) v_c``, ``bank_g`` (bf16) is ``B * g``. The fp32
    ``[128, T]`` tiles, equal on every partition, are ``m = M = max_c q_c``, ``z = Z = sum_c exp(q_c - M)`` (at
    least 1) and ``n = N = ||B||^2`` (of the bf16 B).
    """

    bank: nl.NkiTensor
    bank_g: nl.NkiTensor
    m: nl.NkiTensor
    z: nl.NkiTensor
    n: nl.NkiTensor


def attn_res_nomax_bound(w: np.ndarray) -> float:
    """Return the bound ``2 sqrt(H) ||w||`` of the stream exponent ``q_s - M`` of an AttnRes mix with the score weight
    ``w`` (host array, ``H`` values).

    A score is ``q = (v . w) rsqrt(mean(v^2) + eps)`` and ``rsqrt(mean(v^2) + eps) <= sqrt(H) / ||v||``, so
    ``|q| <= sqrt(H) ||w||`` (Cauchy-Schwarz), and ``q_s - M`` is at most twice that. The no-max form of ``attn_res_ahead_hT`` needs a
    bound below ``ATTN_RES_NOMAX_LIMIT``.

    Example:
        >>> round(attn_res_nomax_bound(np.full(4, 0.5)), 3)
        4.0
    """
    w = np.asarray(w, dtype=np.float64)
    return float(2.0 * np.sqrt(w.size) * np.linalg.norm(w))


def _vector_exp(dst, src):
    """``dst = exp(src)`` on VectorE (NeuronCore-v4), so the mix loads no ScalarE activation table; ScalarE on Trn2."""
    if is_trn3():
        return nisa.exponential(dst=dst, src=src)
    return nisa.activation(dst=dst, op=nl.exp, data=src)


def _act_rsqrt(dst, src, scale: float, eps: float):
    """``dst = rsqrt(src * scale + eps)`` on ScalarE (the rsqrt of ``rmsnorm_hT``; VectorE tensor_scalar takes no
    power op on the device)."""
    return nisa.activation(dst=dst, op=nl.rsqrt, data=src, scale=scale, bias=eps)


def _ones_tiles(dst, prods, ones, n_tiles: int, width: int):
    """Accumulate ``ones.T @ prods[:, j]`` over the hidden tiles j into the fp32 PSUM ``dst`` (``[128, width]``): the sum
    over the partitions and the tiles of each column, on every partition, with one stationary. ``prods`` is a whole fp32
    ``[128, n, width]`` tile (tile j contiguous). Returns the last matmul."""
    last = None
    for j in range(n_tiles):
        last = nisa.nc_matmul(
            dst=dst,
            stationary=ones,
            moving=prods.ap(pattern=[[n_tiles * width, P_MAX], [1, width]], offset=j * width),
            accumulate=j > 0,
            is_stationary_onezero=True,
        )
    return last


def _by_tile(prods, k: int, T: int, n_tiles: int, width: int):
    """Return the ``[128, T, n]`` view of slot ``k`` (columns ``k T .. k T + T - 1``) of the ``[128, n, width]`` tile
    ``prods``: the write pattern of a product pass whose tiles ``_ones_tiles`` then sums."""
    return prods.ap(pattern=[[n_tiles * width, P_MAX], [1, T], [width, n_tiles]], offset=k * T)


def _tile_major(x, T: int, n_tiles: int):
    """Return the ``[128, n, T]`` view of the ``[128, T, n]`` tile ``x`` (tile ``j`` of token ``t`` at ``[j, t]``): the
    read pattern of a pass that writes a ``[128, n, T]`` tile of ``_ones_tiles`` in its own order. An SBUF write with a
    strided innermost dimension takes several times as long as a contiguous one on every engine, a strided read does
    not."""
    return x.ap(pattern=[[T * n_tiles, P_MAX], [1, n_tiles], [n_tiles, T]], offset=0)


def _broadcast_sums(dots, ones, sb, width: int):
    """Return the fp32 PSUM ``[128, width]`` of ``hi + lo`` of the ``[2, width]`` PSUM sums of ``_tensor_sums`` on every
    partition (a copy to SBUF, then a ``K = 2`` matmul with a 0 / 1 stationary: exact)."""
    parts = sb.tile((2, width), nl.float32)
    nisa.tensor_copy(dst=parts, src=dots, engine=nisa.engine.vector)
    out = nl.ndarray((P_MAX, width), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=out, stationary=ones[0:2, 0:P_MAX], moving=parts, is_stationary_onezero=True)
    return out


def attn_res_ahead_part(
    bank_hT,
    bank_rstd,
    w_split,
    g_row: WeightRow,
    ones,
    out,
    sb=AUTO_SBUF,
    bank_dma=None,
    after=(),
    pool=True,
    vector_share=False,
    squares_vector=False,
    load_tiles=2,
):
    """Build the ahead part of one AttnRes mix (``AheadPart``): ``attn_res_ahead_parts`` with one side.

    Returns:
        ``out``.
    """
    return attn_res_ahead_parts(
        bank_hT,
        bank_rstd,
        [(w_split, g_row, out)],
        ones,
        sb,
        bank_dma=bank_dma,
        after=after,
        pool=pool,
        vector_share=vector_share,
        squares_vector=squares_vector,
        load_tiles=load_tiles,
    )[0]


class _AheadLoads:
    """The entry loads of a DRAM AttnRes bank for ``attn_res_ahead_parts``: two SBUF ``[128, T, n]`` tiles in turn, an
    entry that one of them still holds is not loaded again, and the loads alternate between the queues ``queues``
    (``BANK_DMAS`` names; empty: the SP hardware DGE queue), so they do not serialize on one queue.

    With ``depth`` > 2 tiles, a load takes the least recently used tile, so the loads run ``depth - 1`` entries ahead of
    the reads of the entry before; loads 2 .. ``depth`` - 1 (the ones that do not wait for a tile) follow ``after`` in
    the schedule, so they do not move above the instructions that the part follows. The same bytes reach the reads in
    the same order."""

    def __init__(self, bank_hT, sb, queues=(), depth=2, after=()):
        _, _, T, n_tiles = bank_hT.shape
        kernel_assert(depth >= 2, f"_AheadLoads takes 2 or more tiles, got {depth}")
        self.bank_hT = bank_hT
        self.tiles = [sb.tile((P_MAX, T, n_tiles), bank_hT.dtype) for _ in range(depth)]
        self.held = [None] * depth
        self.recent = 1
        self.dmas = [bank_dma_options(q) for q in queues] or [
            {"dge_mode": nisa.dge_mode.hwdge, "engine": nisa.engine.sync}
        ]
        self.loads = 0
        self.depth = depth
        self.after = tuple(after)
        # depth > 2: the tile order, least recently used first.
        self.lru = list(range(depth))

    def entry(self, c: int):
        """Return the SBUF tile of entry ``c``."""
        if self.depth > 2:
            return self._entry_ring(c)
        if c in self.held:
            self.recent = self.held.index(c)
            return self.tiles[self.recent]
        self.recent = 1 - self.recent
        nisa.dma_copy(
            dst=self.tiles[self.recent], src=self.bank_hT[0:P_MAX, c], **self.dmas[self.loads % len(self.dmas)]
        )
        self.loads += 1
        self.held[self.recent] = c
        return self.tiles[self.recent]

    def _entry_ring(self, c: int):
        """``entry`` with ``depth`` > 2 tiles (least recently used tile first)."""
        if c in self.held:
            k = self.held.index(c)
        else:
            k = self.lru[0]
            load = nisa.dma_copy(
                dst=self.tiles[k], src=self.bank_hT[0:P_MAX, c], **self.dmas[self.loads % len(self.dmas)]
            )
            # Only the loads that the deeper ring frees from a tile wait (loads 2 .. depth - 1) take the edge.
            if 2 <= self.loads < self.depth and self.after:
                handle = load if hasattr(load, "depends_on") else nisa.NkiInstruction(load)
                handle.depends_on(*self.after, kind=nisa.dependency.schedule)
            self.loads += 1
            self.held[k] = c
        self.lru.remove(k)
        self.lru.append(k)
        return self.tiles[k]


def attn_res_ahead_parts(
    bank_hT,
    bank_rstd,
    sides,
    ones,
    sb=AUTO_SBUF,
    bank_dma=None,
    after=(),
    pool=True,
    vector_share=False,
    squares_vector=False,
    load_tiles=2,
    marks=None,
):
    """Build the ahead parts (``AheadPart``) of the AttnRes mixes that share the C bank entries ``bank_hT``: the bank
    side of ``attn_res_ahead_hT``.

    A part depends only on the bank entries and the score weight of its mix, not on the stream, so a caller traces it
    where the engines are idle (the collective-bound tail of the layer before). The mixes of one layer read the same
    entries and differ only in their weights, so one call builds their parts with one pass of entry loads for the scores
    and one for the weighted sums (an entry that a tile still holds is not loaded again). Each side ``(w_split, g_row,
    out)`` gives the bf16 ``[128, n, 2]`` hi / lo split of its score weight (``attn_res_tensor_weights``), the
    ``WeightRow`` of its norm weight and an ``AheadPart`` of whole tiles for its result.

    The weighted sums ``v_c . w`` run on TensorE (exact products of the hi / lo split, FP32 sums), the scores take the
    kept rstd of each entry (``bank_rstd``), the exponentials run on VectorE. ``B = sum_c exp(q_c - M) v_c`` runs one
    side on GpSimd and the next on VectorE (one side alone alternates its entries between them), and ``||B||^2`` takes
    ScalarE squares and a TensorE ones-accumulate over the hidden tiles.

    Args:
        bank_hT: SBUF or DRAM ``[128, C, T, n]`` bank view of the C >= 1 entries of the mixes.
        bank_rstd: fp32 ``[128, C, T]`` view: ``rsqrt(mean(v_c^2) + eps)`` of each entry, kept from its push.
        sides: the ``(w_split, g_row, out)`` of each mix.
        ones: all-ones fp32 ``[128, 128]`` SBUF tile.
        sb: the allocator of the scratch tiles.
        bank_dma: with a DRAM bank, None loads the entries on the SP hardware DGE queue; any ``BANK_DMAS`` name
            spreads them over all the ``BANK_DMAS`` queues in turn.
        after: instructions that the first TensorE sum follows in the schedule (a schedule edge).
        pool: the GpSimd share of the weighted sums and ``B * g``; False puts that work on VectorE (where GpSimd is
            busy, for example with the gated DMA issues of the MoE glue).
        vector_share: with ``pool``, the GpSimd share runs on VectorE with the same accumulators: one side keeps its
            two accumulators (its entries in turn, then one add), more sides keep one each, and every accumulator takes
            the same FP32 products and adds in the same order, so the values do not change where the two engines round
            alike (a hardware identity check). A GpSimd tensor_tensor of a ``[128, T, n]`` entry is much slower than the
            VectorE op and stalls the VectorE ops that share its SBUF port. False (default) keeps the program.
        squares_vector: the squares of ``||B||^2`` run on VectorE (a multiply of the bf16 ``B`` by itself) in place of
            the ScalarE square: the exact FP32 square of a bf16 value either way (as ``rmsnorm_hT`` ``act_square_rows``),
            in about a fifth of the ScalarE time. False (default) keeps the program.
        load_tiles: the SBUF tiles of the entry loads of a DRAM bank (``_AheadLoads``; at most C are taken): 2
            (default) keeps the program; more let the loads run ahead of the reads (with two tiles each load waits for
            the reads of the entry two before) and leave more entries held for the weighted sums, so fewer reloads.
            The same bytes in the same order.
        marks: None (default), or a list that gets the last VectorE instruction of each side's weighted sums (its
            ``B * g`` multiply), for a caller that orders later VectorE work after the part. The program does not change.

    Returns:
        The ``out`` of each side.
    """
    steps = _ahead_parts_steps(
        bank_hT, bank_rstd, sides, ones, sb, bank_dma, after, pool, vector_share, squares_vector, load_tiles, marks
    )
    while True:
        try:
            next(steps)
        except StopIteration as done:
            return done.value


def _ahead_parts_steps(
    bank_hT,
    bank_rstd,
    sides,
    ones,
    sb,
    bank_dma,
    after,
    pool,
    vector_share=False,
    squares_vector=False,
    load_tiles=2,
    marks=None,
    sums_after=None,
):
    """The instructions of ``attn_res_ahead_parts`` in its order, as a generator: it pauses after the score pass and
    after each entry of the weighted sums, and returns the ``out`` of each side. ``sums_after``: None, or a list that
    the first weighted-sum instruction follows in the schedule (read when that instruction is traced, so a caller may
    fill it while the generator pauses after the score pass)."""
    _, C, T, n_tiles = bank_hT.shape
    CT = C * T
    kernel_assert(C >= 1 and sides, "attn_res_ahead_parts needs a bank entry and a side")
    sbuf_bank = bank_hT.buffer == nl.sbuf
    kernel_assert(load_tiles >= 2, f"attn_res_ahead_parts load_tiles must be 2 or more, got {load_tiles}")
    # load_tiles: more than C tiles would hold nothing more; up to 2 keeps the program.
    depth = max(2, min(load_tiles, C))
    loads = (
        None
        if sbuf_bank
        else _AheadLoads(bank_hT, sb, BANK_DMAS if bank_dma else (), depth=depth, after=after if depth > 2 else ())
    )

    def entry(c):
        return bank_hT[0:P_MAX, c] if sbuf_bank else loads.entry(c)

    dots = [nl.ndarray((2, CT), dtype=nl.float32, buffer=nl.psum) for _ in sides]
    for c in range(C):
        v = entry(c)
        for k, (w_split, _, _) in enumerate(sides):
            _tensor_sums(
                dots[k][0:2, c * T : (c + 1) * T],
                lambda j, v=v: v[0:P_MAX, 0:T, j],
                w_split,
                after=after if c == k == 0 else (),
            )
    by_token = [[CT, P_MAX], [1, T], [T, C]]
    weights = []
    for k, (_, _, out) in enumerate(sides):
        weighted = _broadcast_sums(dots[k], ones, sb, CT)
        q = sb.tile((P_MAX, C, T), nl.float32)
        nisa.tensor_tensor(
            dst=q.reshape((P_MAX, CT)), data1=weighted, data2=bank_rstd.reshape((P_MAX, CT)), op=nl.multiply
        )
        nisa.tensor_reduce(dst=out.m, data=_broadcast(q, by_token), op=nl.maximum, axis=(2,))
        shifted = sb.tile((P_MAX, C, T), nl.float32)
        nisa.tensor_tensor(dst=shifted, data1=q, data2=_broadcast(out.m, [[T, P_MAX], [0, C], [1, T]]), op=nl.subtract)
        e = sb.tile((P_MAX, C, T), nl.float32)
        _vector_exp(e, shifted)
        nisa.tensor_reduce(dst=out.z, data=_broadcast(e, by_token), op=nl.add, axis=(2,))
        weights.append(e)
    yield
    # The FP32 accumulators: one engine for each side, or, for one side with pool, its entries in turn on GpSimd and
    # VectorE (two accumulators, one add).
    gp, ve = nisa.engine.gpsimd, nisa.engine.vector
    if len(sides) == 1 and pool:
        plans = [(gp, ve)]
    else:
        plans = [((gp if pool and k % 2 == 0 else ve),) for k in range(len(sides))]
    if vector_share:
        # vector_share: the same plans (accumulators, entry order, final add) with VectorE in place of GpSimd.
        plans = [tuple(ve for _ in eng) for eng in plans]
    accs = [[sb.tile((P_MAX, T, n_tiles), nl.float32) for _ in range(min(C, len(eng)))] for eng in plans]
    terms = [
        [sb.tile((P_MAX, T, n_tiles), nl.float32) for _ in range(min(max(C - len(eng), 0), len(eng)))] for eng in plans
    ]
    # The entries in reverse order: the last two loads of the score pass are still in the load tiles.
    for o, c in enumerate(reversed(range(C))):
        v = entry(c)
        for k, eng in enumerate(plans):
            i = o % len(eng)
            e_c = weights[k].ap(pattern=[[CT, P_MAX], [1, T], [0, n_tiles]], offset=c * T)
            if o < len(eng):
                first = nisa.tensor_tensor(dst=accs[k][i], data1=v, data2=e_c, op=nl.multiply, engine=eng[i])
                if o == k == 0 and sums_after:
                    first = first if hasattr(first, "depends_on") else nisa.NkiInstruction(first)
                    first.depends_on(*sums_after, kind=nisa.dependency.schedule)
                continue
            nisa.tensor_tensor(dst=terms[k][i], data1=v, data2=e_c, op=nl.multiply, engine=eng[i])
            nisa.tensor_tensor(dst=accs[k][i], data1=accs[k][i], data2=terms[k][i], op=nl.add, engine=eng[i])
        yield
    for k, (_, g_row, out) in enumerate(sides):
        eng = plans[k]
        if len(accs[k]) > 1:
            nisa.tensor_tensor(dst=out.bank, data1=accs[k][0], data2=accs[k][1], op=nl.add, engine=ve)
        else:
            nisa.tensor_copy(dst=out.bank, src=accs[k][0], engine=eng[0])
        bank_g = nisa.tensor_tensor(
            dst=out.bank_g, data1=out.bank, data2=g_row.repeat(T), op=nl.multiply, engine=eng[0]
        )
        if marks is not None:
            marks.append(bank_g)
        # N = ||B||^2 of the bf16 B: ScalarE squares, then a TensorE ones-accumulate over the hidden tiles. The squares
        # are written in their own (contiguous) order from a tile-major read of B: the same square at each address.
        squares = sb.tile((P_MAX, n_tiles, T), nl.float32)
        bank_by_tile = _tile_major(out.bank, T, n_tiles)
        if squares_vector:
            # squares_vector: the same exact FP32 squares on VectorE.
            nisa.tensor_tensor(dst=squares, data1=bank_by_tile, data2=bank_by_tile, op=nl.multiply, engine=ve)
        else:
            nisa.activation(dst=squares, op=nl.square, data=bank_by_tile, engine=nisa.engine.scalar)
        tot = nl.ndarray((P_MAX, T), dtype=nl.float32, buffer=nl.psum)
        _ones_tiles(tot, squares, ones, n_tiles, T)
        nisa.tensor_copy(dst=out.n, src=tot, engine=nisa.engine.vector)
    return [out for _, _, out in sides]


def attn_res_ahead_parts_staged(
    bank_hT,
    bank_rstd,
    sides,
    ones,
    sb=AUTO_SBUF,
    bank_dma=None,
    after=(),
    pool=True,
    vector_share=False,
    squares_vector=False,
    load_tiles=2,
):
    """``attn_res_ahead_parts`` in steps, for a caller that spreads the weighted sums over the gaps of another chain.

    The score pass and the exponentials are traced now. Return ``(outs, advance)``: ``advance(n)`` traces the weighted sums
    of the next ``n`` entries (the entries in the order of ``attn_res_ahead_parts``) and ``advance(None)`` traces the
    rest, the last entries and the bank norm; ``advance`` returns True once the parts are complete. The instructions and
    their order on each engine are those of ``attn_res_ahead_parts``, so the parts hold the same bits; only their place
    in the trace moves. ``vector_share``, ``squares_vector`` and ``load_tiles`` as in ``attn_res_ahead_parts``.
    ``advance(n, after)``: before the first weighted sum is traced, ``after`` (instructions) adds a schedule edge: the
    first weighted-sum instruction follows them (a call after that one ignores ``after``). The program of a caller that
    passes no ``after`` does not change.
    """
    sums_after = []
    steps = _ahead_parts_steps(
        bank_hT,
        bank_rstd,
        sides,
        ones,
        sb,
        bank_dma,
        after,
        pool,
        vector_share,
        squares_vector,
        load_tiles,
        sums_after=sums_after,
    )
    next(steps)
    state = {"done": False}

    def advance(n=None, after=()):
        sums_after.extend(after)
        count = 0
        while not state["done"] and (n is None or count < n):
            try:
                next(steps)
                count += 1
            except StopIteration:
                state["done"] = True
        return state["done"]

    return [out for _, _, out in sides], advance


def new_ahead_part(T: int, n_tiles: int, sb=AUTO_SBUF) -> AheadPart:
    """Return an ``AheadPart`` of new whole tiles (``attn_res_ahead_part`` ``out``)."""
    return AheadPart(
        bank=sb.tile((P_MAX, T, n_tiles), nl.bfloat16),
        bank_g=sb.tile((P_MAX, T, n_tiles), nl.bfloat16),
        m=sb.tile((P_MAX, T), nl.float32),
        z=sb.tile((P_MAX, T), nl.float32),
        n=sb.tile((P_MAX, T), nl.float32),
    )


def _gram_comb(stream_hT, bank, w_split, sb):
    """Return the bf16 ``[128, 2T + 2, n]`` comb of ``_gram_stats`` (``attn_res_ahead_hT`` ``gram_merge``): rows
    ``0 .. T - 1`` the stream, ``T .. 2T - 1`` the bank and rows ``2T`` / ``2T + 1`` the hi / lo weight split, so the
    ``2T + 2`` columns of hidden tile ``j`` are one moving operand. Each part is a VectorE copy (the same bits); the
    bank and weight rows come first, as they do not wait for the stream."""
    _, T, n_tiles = stream_hT.shape
    comb = sb.tile((P_MAX, 2 * T + 2, n_tiles), nl.bfloat16)
    nisa.tensor_copy(
        dst=comb[0:P_MAX, T : 2 * T, 0:n_tiles], src=bank[0:P_MAX, 0:T, 0:n_tiles], engine=nisa.engine.vector
    )
    for half in range(2):
        nisa.tensor_copy(
            dst=comb[0:P_MAX, 2 * T + half, 0:n_tiles], src=w_split[0:P_MAX, 0:n_tiles, half], engine=nisa.engine.vector
        )
    nisa.tensor_copy(dst=comb[0:P_MAX, 0:T, 0:n_tiles], src=stream_hT, engine=nisa.engine.vector)
    return comb


def _gram_stats(stream_hT, bank, w_split, ones, sb, comb=None):
    """Return the fp32 PSUM ``[128, k T]`` of the stream sums of ``attn_res_ahead_hT`` on every partition, on TensorE:
    ``S = s . s``, then with ``bank`` ``X = s . B``, then with ``w_split`` ``D = s . w`` (hi + lo), for each token.

    Each hidden tile ``j`` of the stream (bf16 ``[128, T]``) is the stationary of up to three matmuls: the moving
    operands are the same stream tile, the bank tile ``j`` and the hi / lo weight split of tile ``j``. The products of
    bf16 values are exact and the sums FP32, accumulated over the tiles in PSUM, so the diagonals of ``s s^T`` and
    ``s B^T`` are the sums. VectorE masks the diagonals (and puts ``D`` on them), and a ones matmul sums each column of
    the masks onto every partition.

    With ``comb`` (``_gram_comb``, which needs ``bank`` and ``w_split``), tile ``j`` is the stationary of ONE matmul
    whose moving operand is the tile's ``2T + 2`` comb columns: each PSUM element gets the same products in the same
    tile order, so the sums are the same, in a third of the matmuls and weight loads."""
    _, T, n_tiles = stream_hT.shape
    ident = nl.shared_identity_matrix(P_MAX, dtype=nl.float32)
    if comb is not None:
        kernel_assert(bank is not None and w_split is not None, "the gram comb holds the bank and the weight split")
        width = 2 * T + 2
        kernel_assert(
            tuple(comb.shape) == (P_MAX, width, n_tiles),
            f"the gram comb is [{P_MAX}, {width}, {n_tiles}]: {comb.shape}",
        )
        g_all = nl.ndarray((T, width), dtype=nl.float32, buffer=nl.psum)
        for j in range(n_tiles):
            nisa.nc_matmul(
                dst=g_all,
                stationary=stream_hT[0:P_MAX, 0:T, j],
                moving=comb[0:P_MAX, 0:width, j],
                accumulate=j > 0,
            )
        g_ss, g_sb = g_all[0:T, 0:T], g_all[0:T, T : 2 * T]
        sw_hi, sw_lo = g_all[0:T, 2 * T : 2 * T + 1], g_all[0:T, 2 * T + 1 : width]
    else:
        g_ss = nl.ndarray((T, T), dtype=nl.float32, buffer=nl.psum)
        g_sb = nl.ndarray((T, T), dtype=nl.float32, buffer=nl.psum) if bank is not None else None
        g_sw = nl.ndarray((T, 2), dtype=nl.float32, buffer=nl.psum) if w_split is not None else None
        for j in range(n_tiles):
            stat = stream_hT[0:P_MAX, 0:T, j]
            nisa.nc_matmul(dst=g_ss, stationary=stat, moving=stat, accumulate=j > 0)
            if g_sb is not None:
                nisa.nc_matmul(dst=g_sb, stationary=stat, moving=bank[0:P_MAX, 0:T, j], accumulate=j > 0)
            if g_sw is not None:
                nisa.nc_matmul(dst=g_sw, stationary=stat, moving=w_split[0:P_MAX, j, 0:2], accumulate=j > 0)
        sw_hi, sw_lo = (g_sw[0:T, 0:1], g_sw[0:T, 1:2]) if g_sw is not None else (None, None)
    k = 1 + (g_sb is not None) + (sw_hi is not None)
    masked = sb.tile((T, k * T), nl.float32)
    eye = ident[0:T, 0:T]
    nisa.tensor_tensor(dst=masked[0:T, 0:T], data1=g_ss, data2=eye, op=nl.multiply, engine=nisa.engine.vector)
    if g_sb is not None:
        nisa.tensor_tensor(dst=masked[0:T, T : 2 * T], data1=g_sb, data2=eye, op=nl.multiply, engine=nisa.engine.vector)
    if sw_hi is not None:
        # hi + lo of s . w (a VectorE operand pair cannot both sit in PSUM: the lo column goes to SBUF first).
        lo = sb.tile((T, 1), nl.float32)
        nisa.tensor_copy(dst=lo, src=sw_lo, engine=nisa.engine.vector)
        d = sb.tile((T, 1), nl.float32)
        nisa.tensor_tensor(dst=d, data1=sw_hi, data2=lo, op=nl.add, engine=nisa.engine.vector)
        nisa.tensor_tensor(
            dst=masked[0:T, (k - 1) * T : k * T],
            data1=eye,
            data2=d.ap(pattern=[[1, T], [0, T]], offset=0),
            op=nl.multiply,
            engine=nisa.engine.vector,
        )
    out = nl.ndarray((P_MAX, k * T), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=out, stationary=ones[0:T, 0:P_MAX], moving=masked, is_stationary_onezero=True)
    return out


def attn_res_ahead_hT(
    stream_hT,
    part,
    w_split,
    g_row: WeightRow,
    eps,
    ones,
    sb=AUTO_SBUF,
    out=None,
    nomax=False,
    done=None,
    pool=True,
    out16=None,
    combine="tensor",
    stop=None,
    stats_form="products",
    gram_merge=False,
):
    """The AttnRes mix and its RMSNorm in one pass over the stream, from the ahead part of the bank.

    It returns ``rmsnorm(sum_c softmax(q)_c * c) * g`` of ``attn_res_hT`` + ``rmsnorm_hT`` over the C bank entries and
    the stream s (the same math, other FP32 rounding: an accuracy-class form). With the ahead part ``(B, M, Z, N)``
    (``attn_res_ahead_part``) of the bank:

        q_s = (s . w) rsqrt(mean(s^2) + eps); e = exp(q_s - M); m = (e s + B) / (Z + e) = alpha s + beta B
        ||m||^2 = alpha^2 ||s||^2 + 2 alpha beta (s . B) + beta^2 N; r = rsqrt(||m||^2 / H + eps)
        x = (alpha r) (s * g) + (beta r) (B * g)

    With ``nomax`` False (default) the stream exponent takes the shift ``max(M, q_s)`` (and B, Z a factor), so no
    exponent overflows. ``nomax`` True takes ``exp(q_s - M)`` directly: only for a score weight whose
    ``attn_res_nomax_bound`` is below ``ATTN_RES_NOMAX_LIMIT``.

    The engines: TensorE sums ``s . w`` (hi / lo split, one stationary a tile) and, with one stationary, the squares and
    ``s . B`` over the partitions and the hidden tiles. ScalarE takes the squares, GpSimd ``s * B``, VectorE ``s * g`` and
    the per-token scalars (the exponentials on VectorE, the two rsqrt on ScalarE). The combine runs in ``ATTN_RES_AHEAD_CHUNKS`` token chunks: GpSimd
    multiplies ``s * g`` by ``alpha r``, VectorE ``B * g`` by ``beta r``, TensorE adds both into PSUM (an identity
    accumulate), and ScalarE writes the bf16 result.

    Args:
        stream_hT: SBUF bf16 ``[128, T, n]`` stream.
        part: the ``AheadPart`` of the bank, or None for an empty bank (the result is then ``rmsnorm(s) * g``).
        w_split: bf16 ``[128, n, 2]`` split of the score weight (``attn_res_tensor_weights``).
        g_row: the ``WeightRow`` of the norm weight.
        eps: Norm epsilon.
        ones: all-ones fp32 ``[128, 128]`` SBUF tile.
        sb: the allocator of the scratch tiles.
        out: optional SBUF ``[128, T, n]`` tile for the result (None: a new tile of the stream dtype).
        nomax: the no-max form (see above).
        done: optional list; the instructions that write ``out`` (and ``out16``) are appended.
        pool: the GpSimd share (``s * B`` and the ``s * g`` combine multiply); False puts it on VectorE, for a mix
            where GpSimd is busy (the MLP mix, beside the gated DMA issues of the MoE glue).
        out16: optional FP16 SBUF ``[128, T, n]`` tile that also gets the result in FP16 from the FP32 sum (the input
            of an FP16 router, as ``rmsnorm_hT`` ``out16``).
        combine: the engines of the combine add (``ATTN_RES_AHEAD_COMBINES``): "tensor" (default) adds the two products
            on TensorE in PSUM and ScalarE writes the result; "vector" adds them on VectorE, which writes the result.
            The same FP32 values.
        stats_form: the form of the stream sums (``ATTN_RES_AHEAD_STATS``); the same exact products and FP32 sums.
        gram_merge: with ``stats_form`` "gram" and a bank part, the stream sums take one TensorE matmul a hidden tile
            on a comb of the stream, the bank and the weight split (``_gram_comb``, three VectorE copies) in place of
            three: the same sums. False (default) keeps the program.
        stop: a timing probe for the tests: "stats" returns after the stream sums, "scalars" after the per-token
            scalars (``out`` then holds those values in its first token columns, not the mix). None (default) runs the
            whole mix.

    Returns:
        SBUF ``[128, T, n]``: the normed mix, in the dtype of ``out`` (default the stream's).
    """
    _, T, n_tiles = stream_hT.shape
    H = n_tiles * P_MAX
    kernel_assert(combine in ATTN_RES_AHEAD_COMBINES, f"combine must be one of {ATTN_RES_AHEAD_COMBINES}")
    kernel_assert(stats_form in ATTN_RES_AHEAD_STATS, f"stats_form must be one of {ATTN_RES_AHEAD_STATS}")
    kernel_assert(stop in (None, "stats", "scalars"), "stop must be None, 'stats' or 'scalars'")
    kernel_assert(not gram_merge or stats_form == "gram", f"gram_merge merges the gram stream sums, got {stats_form!r}")
    out = sb.tile((P_MAX, T, n_tiles), stream_hT.dtype) if out is None else out
    chunks = ATTN_RES_AHEAD_CHUNKS if T % ATTN_RES_AHEAD_CHUNKS == 0 else 1
    rows = T // chunks
    kernel_assert(rows * n_tiles <= PSUM_BANK_FP32, f"a combine chunk of {rows} x {n_tiles} exceeds one PSUM bank")
    by_tok = [[T, P_MAX], [1, T], [0, n_tiles]]
    # gram_merge: the comb copies go ahead of the s * g multiply, which TensorE does not wait for.
    comb = _gram_comb(stream_hT, part.bank, w_split, sb) if gram_merge and part is not None else None
    s_g = sb.tile((P_MAX, T, n_tiles), nl.float32)
    nisa.tensor_tensor(dst=s_g, data1=stream_hT, data2=g_row.repeat(T), op=nl.multiply, engine=nisa.engine.vector)
    side = nisa.engine.gpsimd if pool else nisa.engine.vector
    if stats_form == "gram":
        # S, X and D in one group of TensorE matmuls (columns 0 .. T - 1, T .. 2T - 1, 2T .. 3T - 1).
        width = 3 * T if part is not None else T
        sums = _gram_stats(
            stream_hT, None if part is None else part.bank, None if part is None else w_split, ones, sb, comb=comb
        )
    else:
        width = 2 * T if part is not None else T
        prods = sb.tile((P_MAX, n_tiles, width), nl.float32)
        nisa.activation(dst=_by_tile(prods, 0, T, n_tiles, width), op=nl.square, data=stream_hT)
        if part is not None:
            nisa.tensor_tensor(
                dst=_by_tile(prods, 1, T, n_tiles, width),
                data1=stream_hT,
                data2=part.bank,
                op=nl.multiply,
                engine=side,
            )
        sums = nl.ndarray((P_MAX, width), dtype=nl.float32, buffer=nl.psum)
        _ones_tiles(sums, prods, ones, n_tiles, width)
    stats = sb.tile((P_MAX, width), nl.float32)
    nisa.tensor_copy(dst=stats, src=sums, engine=nisa.engine.vector)
    S = stats[0:P_MAX, 0:T]
    if part is None:
        r = sb.tile((P_MAX, T), nl.float32)
        _act_rsqrt(r, S, 1.0 / float(H), eps)
        a_writes = []
        for k in range(chunks):
            t0, t1 = k * rows, (k + 1) * rows
            for dst in (out,) if out16 is None else (out, out16):
                a_writes.append(
                    nisa.tensor_tensor(
                        dst=dst[0:P_MAX, t0:t1, 0:n_tiles],
                        data1=s_g[0:P_MAX, t0:t1, 0:n_tiles],
                        data2=r.ap(pattern=[[T, P_MAX], [1, rows], [0, n_tiles]], offset=t0),
                        op=nl.multiply,
                        engine=nisa.engine.vector,
                    )
                )
        if done is not None:
            done.extend(a_writes)
        return out
    X = stats[0:P_MAX, T : 2 * T]
    if stats_form == "gram":
        D = stats[0:P_MAX, 2 * T : 3 * T]
    else:
        dots = nl.ndarray((2, T), dtype=nl.float32, buffer=nl.psum)
        _tensor_sums(dots, lambda j: stream_hT[0:P_MAX, 0:T, j], w_split)
        D = _broadcast_sums(dots, ones, sb, T)
    if stop == "stats":
        nisa.tensor_copy(dst=out[0:P_MAX, 0, 0:T], src=D, engine=nisa.engine.vector)
        nisa.tensor_copy(dst=out[0:P_MAX, 1, 0:T], src=S, engine=nisa.engine.vector)
        nisa.tensor_copy(dst=out[0:P_MAX, 2, 0:T], src=X, engine=nisa.engine.vector)
        return out
    # The per-token scalars ([128, T] fp32 on every partition): VectorE, the two rsqrt on ScalarE.
    v = [sb.tile((P_MAX, T), nl.float32) for _ in range(8)]
    rs, q, e, k_den, alpha, beta, n2, r = v
    _act_rsqrt(rs, S, 1.0 / float(H), eps)
    nisa.tensor_tensor(dst=q, data1=D, data2=rs, op=nl.multiply)
    if nomax:
        nisa.tensor_tensor(dst=e, data1=q, data2=part.m, op=nl.subtract)
        _vector_exp(e, e)
        nisa.tensor_tensor(dst=k_den, data1=part.z, data2=e, op=nl.add)
        nisa.reciprocal(dst=beta, data=k_den)
    else:
        top = sb.tile((P_MAX, T), nl.float32)
        nisa.tensor_tensor(dst=top, data1=q, data2=part.m, op=nl.maximum)
        nisa.tensor_tensor(dst=e, data1=q, data2=top, op=nl.subtract)
        _vector_exp(e, e)
        k = sb.tile((P_MAX, T), nl.float32)
        nisa.tensor_tensor(dst=k, data1=part.m, data2=top, op=nl.subtract)
        _vector_exp(k, k)
        # den = k Z + e; beta = k / den
        nisa.tensor_tensor(dst=k_den, data1=k, data2=part.z, op=nl.multiply)
        nisa.tensor_tensor(dst=k_den, data1=k_den, data2=e, op=nl.add)
        nisa.reciprocal(dst=k_den, data=k_den)
        nisa.tensor_tensor(dst=beta, data1=k, data2=k_den, op=nl.multiply)
    nisa.tensor_tensor(dst=alpha, data1=e, data2=k_den if not nomax else beta, op=nl.multiply)
    # ||m||^2 = alpha^2 S + 2 alpha beta X + beta^2 N, in the order (alpha S + 2 beta X) alpha + beta^2 N.
    tmp = sb.tile((P_MAX, T), nl.float32)
    nisa.tensor_tensor(dst=n2, data1=alpha, data2=S, op=nl.multiply)
    nisa.tensor_tensor(dst=tmp, data1=beta, data2=X, op=nl.multiply)
    nisa.scalar_tensor_tensor(dst=n2, data=tmp, op0=nl.multiply, operand0=2.0, op1=nl.add, operand1=n2)
    nisa.tensor_tensor(dst=n2, data1=n2, data2=alpha, op=nl.multiply)
    nisa.tensor_tensor(dst=tmp, data1=beta, data2=beta, op=nl.multiply)
    nisa.tensor_tensor(dst=tmp, data1=tmp, data2=part.n, op=nl.multiply)
    nisa.tensor_tensor(dst=n2, data1=n2, data2=tmp, op=nl.add)
    _act_rsqrt(r, n2, 1.0 / float(H), eps)
    a_s = sb.tile((P_MAX, T), nl.float32)
    b_s = sb.tile((P_MAX, T), nl.float32)
    nisa.tensor_tensor(dst=a_s, data1=alpha, data2=r, op=nl.multiply)
    nisa.tensor_tensor(dst=b_s, data1=beta, data2=r, op=nl.multiply)
    if stop == "scalars":
        nisa.tensor_copy(dst=out[0:P_MAX, 0, 0:T], src=a_s, engine=nisa.engine.vector)
        nisa.tensor_copy(dst=out[0:P_MAX, 1, 0:T], src=b_s, engine=nisa.engine.vector)
        return out
    # The combine, by token chunk: GpSimd (s g) a, VectorE (B g) b, the TensorE identity add, the ScalarE copy.
    # The kernel's shared FP32 identity (on the loop path, the heap tile of install_identities).
    ident = nl.shared_identity_matrix(P_MAX, dtype=nl.float32)
    t_a = sb.tile((P_MAX, T, n_tiles), nl.float32)
    t_b = sb.tile((P_MAX, T, n_tiles), nl.float32)
    writes = []
    for c in range(chunks):
        t0, t1 = c * rows, (c + 1) * rows
        pat = [[T, P_MAX], [1, rows], [0, n_tiles]]
        nisa.tensor_tensor(
            dst=t_a[0:P_MAX, t0:t1, 0:n_tiles],
            data1=s_g[0:P_MAX, t0:t1, 0:n_tiles],
            data2=a_s.ap(pattern=pat, offset=t0),
            op=nl.multiply,
            engine=side,
        )
        nisa.tensor_tensor(
            dst=t_b[0:P_MAX, t0:t1, 0:n_tiles],
            data1=part.bank_g[0:P_MAX, t0:t1, 0:n_tiles],
            data2=b_s.ap(pattern=pat, offset=t0),
            op=nl.multiply,
            engine=nisa.engine.vector,
        )
        if combine == "vector":
            for dst in (out,) if out16 is None else (out, out16):
                writes.append(
                    nisa.tensor_tensor(
                        dst=dst[0:P_MAX, t0:t1, 0:n_tiles],
                        data1=t_a[0:P_MAX, t0:t1, 0:n_tiles],
                        data2=t_b[0:P_MAX, t0:t1, 0:n_tiles],
                        op=nl.add,
                        engine=nisa.engine.vector,
                    )
                )
            continue
        acc = nl.ndarray((P_MAX, rows * n_tiles), dtype=nl.float32, buffer=nl.psum)
        for src, first in ((t_a, True), (t_b, False)):
            nisa.nc_matmul(
                dst=acc,
                stationary=ident,
                moving=src.ap(pattern=[[T * n_tiles, P_MAX], [1, rows * n_tiles]], offset=t0 * n_tiles),
                accumulate=not first,
                is_stationary_onezero=True,
            )
        acc_rows = acc.reshape((P_MAX, rows, n_tiles))
        writes.append(nisa.activation(dst=out[0:P_MAX, t0:t1, 0:n_tiles], op=nl.copy, data=acc_rows))
        if out16 is not None:
            writes.append(nisa.activation(dst=out16[0:P_MAX, t0:t1, 0:n_tiles], op=nl.copy, data=acc_rows))
    if done is not None:
        done.extend(writes)
    return out


def _mix_from_scores(
    scores,
    bank_entry,
    bank_hT,
    stream_hT,
    compact,
    sb,
    vector_exp=False,
    read_out=None,
    act_rows=0,
    spare=None,
    out_dtype=None,
    act_stt=False,
):
    """Return ``bf16(sum_c softmax(score)_c * c)`` from the ``[128, C + 1, T]`` fp32 scores of ``attn_res_hT``.

    With ``act_rows`` (``attn_res_hT`` ``act_product_rows``, compact form, C > 1), the stream term goes first, so VectorE
    has work while ScalarE takes the first products, and ``_mix_entries`` puts the products of the last ``act_rows``
    rows of entries 1 .. C - 1 on ScalarE, in ``term`` and ``spare``. The sums keep their order. ``out_dtype``: the
    dtype of the result (None: the dtype of the stream; FP32 keeps the accumulator unrounded). ``act_stt``: the VectorE
    rows of ``_mix_entries`` take its ``stt`` multiply-adds (the same bits).
    """
    _, T, n_tiles = stream_hT.shape
    C = bank_hT.shape[1]
    Cp1 = C + 1
    CT = C * T
    width = Cp1 * T
    by_token = [[width, P_MAX], [1, T], [T, Cp1]]
    row_max = sb.tile((P_MAX, T), nl.float32)
    nisa.tensor_reduce(dst=row_max, data=_broadcast(scores, by_token), op=nl.maximum, axis=(2,))
    shifted = sb.tile((P_MAX, Cp1, T), nl.float32)
    nisa.tensor_tensor(
        dst=shifted, data1=scores, data2=_broadcast(row_max, [[T, P_MAX], [0, Cp1], [1, T]]), op=nl.subtract
    )
    expv = sb.tile((P_MAX, Cp1, T), nl.float32)
    if vector_exp and is_trn3():
        nisa.exponential(dst=expv, src=shifted)
    else:
        nisa.activation(dst=expv, op=nl.exp, data=shifted)
    denom = sb.tile((P_MAX, T), nl.float32)
    nisa.tensor_reduce(dst=denom, data=_broadcast(expv, by_token), op=nl.add, axis=(2,))
    inv_denom = sb.tile((P_MAX, T), nl.float32)
    nisa.reciprocal(dst=inv_denom, data=denom)
    prob = sb.tile((P_MAX, Cp1, T), nl.float32)
    nisa.tensor_tensor(
        dst=prob, data1=expv, data2=_broadcast(inv_denom, [[T, P_MAX], [0, Cp1], [1, T]]), op=nl.multiply
    )

    # acc = sum_c p_c * v_c in fp32, then one cast. The bank terms reduce with the candidates
    # innermost; the stream term adds last.
    if act_rows:
        stream_term = sb.tile((P_MAX, T, n_tiles), nl.float32)
        nisa.tensor_tensor(
            dst=stream_term,
            data1=stream_hT,
            data2=prob.ap(pattern=[[width, P_MAX], [1, T], [0, n_tiles]], offset=CT),
            op=nl.multiply,
        )
        acc = _mix_entries(bank_entry, C, prob, width, T, n_tiles, sb, act_rows, spare, stt=act_stt)
    else:
        if compact:
            acc = _mix_entries(bank_entry, C, prob, width, T, n_tiles, sb)
        else:
            acc = _mix_bank(bank_hT, prob, width, sb)
        stream_term = sb.tile((P_MAX, T, n_tiles), nl.float32)
        nisa.tensor_tensor(
            dst=stream_term,
            data1=stream_hT,
            data2=prob.ap(pattern=[[width, P_MAX], [1, T], [0, n_tiles]], offset=CT),
            op=nl.multiply,
        )
    out = sb.tile((P_MAX, T, n_tiles), out_dtype or stream_hT.dtype)
    mix = nisa.tensor_tensor(dst=out, data1=acc, data2=stream_term, op=nl.add)
    if read_out is not None:
        read_out.append(mix)
    return out


def _mix_bank(bank_hT, prob, width, sb):
    """Return ``sum_c p_c * v_c`` (FP32 ``[128, T, n]``) of the bank entries in one pass over ``[128, C, T, n]``."""
    _, C, T, n_tiles = bank_hT.shape
    CT = C * T
    weighted = sb.tile((P_MAX, C, T, n_tiles), nl.float32)
    nisa.tensor_tensor(
        dst=weighted,
        data1=bank_hT,
        data2=_broadcast(prob, [[width, P_MAX], [T, C], [1, T], [0, n_tiles]]),
        op=nl.multiply,
    )
    acc = sb.tile((P_MAX, T, n_tiles), nl.float32)
    if C == 1:
        nisa.tensor_copy(dst=acc, src=weighted.reshape((P_MAX, T, n_tiles)))
    else:
        nisa.tensor_reduce(
            dst=acc,
            data=_broadcast(weighted, [[CT * n_tiles, P_MAX], [n_tiles, T], [1, n_tiles], [T * n_tiles, C]]),
            op=nl.add,
            axis=(3,),
        )
    return acc


def _mix_entries(bank_entry, C, prob, width, T, n_tiles, sb, act_rows=0, spare=None, stt=False):
    """Return ``sum_c p_c * v_c`` (FP32 ``[128, T, n]``), one bank entry at a time in candidate order.

    ``bank_entry(c)`` returns entry ``c`` as an SBUF tile (a view of an SBUF bank, or a reused tile that a
    DRAM bank streams through).

    With ``act_rows``, ScalarE multiplies the last ``act_rows`` rows of each entry after the first, one
    ``tensor_scalar`` by the row's weight each, and VectorE the other rows in one multiply. The products of entry c
    go to ``term`` or ``spare`` in turn (``spare``: an FP32 ``[128, T, n]`` tile that is free here), so ScalarE can
    take the products of entry c + 1 while VectorE adds entry c. A product is an FP32 multiply of a bf16 row by an
    FP32 weight on either engine, and the adds keep their order, so the result does not change.

    ``stt`` (with ``act_rows``): each VectorE row of an entry after the first takes one ``scalar_tensor_tensor``,
    ``acc = entry * weight + acc``, in place of the multiply into ``term`` and that row's part of the add; the add then
    covers the ScalarE rows only. MEASURED on trn3 hardware: the VectorE multiply-add rounds the
    product to FP32 before the add and keeps the IEEE sign of zeros, so every element takes the same FP32 product and
    add, in the same order: the same bits.
    """
    acc = sb.tile((P_MAX, T, n_tiles), nl.float32)
    term = sb.tile((P_MAX, T, n_tiles), nl.float32)
    if act_rows:
        bufs = (term, spare if spare is not None else sb.tile((P_MAX, T, n_tiles), nl.float32))
        vec_rows = T - act_rows
        for c in range(C):
            entry = bank_entry(c)
            if c == 0:
                p_0 = prob.ap(pattern=[[width, P_MAX], [1, T], [0, n_tiles]], offset=0)
                nisa.tensor_tensor(dst=acc, data1=entry, data2=p_0, op=nl.multiply)
                continue
            buf = bufs[(c - 1) % 2]
            if stt:
                for t in range(vec_rows):
                    nisa.scalar_tensor_tensor(
                        dst=acc[0:P_MAX, t, 0:n_tiles],
                        data=entry[0:P_MAX, t, 0:n_tiles],
                        op0=nl.multiply,
                        operand0=prob[0:P_MAX, c, t : t + 1],
                        op1=nl.add,
                        operand1=acc[0:P_MAX, t, 0:n_tiles],
                    )
            elif vec_rows:
                p_c = prob.ap(pattern=[[width, P_MAX], [1, vec_rows], [0, n_tiles]], offset=c * T)
                nisa.tensor_tensor(
                    dst=buf[0:P_MAX, 0:vec_rows, 0:n_tiles],
                    data1=entry[0:P_MAX, 0:vec_rows, 0:n_tiles],
                    data2=p_c,
                    op=nl.multiply,
                )
            for t in range(vec_rows, T):
                nisa.tensor_scalar(
                    dst=buf[0:P_MAX, t, 0:n_tiles],
                    data=entry[0:P_MAX, t, 0:n_tiles],
                    op0=nl.multiply,
                    operand0=prob[0:P_MAX, c, t : t + 1],
                    engine=nisa.engine.scalar,
                )
            if stt:
                nisa.tensor_tensor(
                    dst=acc[0:P_MAX, vec_rows:T, 0:n_tiles],
                    data1=acc[0:P_MAX, vec_rows:T, 0:n_tiles],
                    data2=buf[0:P_MAX, vec_rows:T, 0:n_tiles],
                    op=nl.add,
                    engine=nisa.engine.vector,
                )
            else:
                nisa.tensor_tensor(dst=acc, data1=acc, data2=buf, op=nl.add)
        return acc
    for c in range(C):
        p_c = prob.ap(pattern=[[width, P_MAX], [1, T], [0, n_tiles]], offset=c * T)
        nisa.tensor_tensor(dst=acc if c == 0 else term, data1=bank_entry(c), data2=p_c, op=nl.multiply)
        if c:
            nisa.tensor_tensor(dst=acc, data1=acc, data2=term, op=nl.add)
    return acc


def token_reduce_scatter(part_hT, tp_group, tag, tp_size, sb=AUTO_SBUF):
    """Reduce an attention-output partial over the TP group and scatter the tokens, one token per rank.

    ``reduce_scatter`` scatters the partition dimension only, so the R tokens go on the partition dimension for the
    collective: transpose the hT partial ``[128, R, n_tiles]`` to tokens-on-partition ``[R, 128 * n_tiles]``, reduce-scatter
    that over ``tp_group`` (``R // tp_size`` rows per rank, ``collective_dim=0``), then transpose the kept rows back to hT.
    With ``R == tp_size`` each rank keeps one token's full hidden, so the residual, the MLP mix and the norm run on that
    one row. The reduce is bit-equal to the all-reduce (a kernel test checks this).

    Args:
        part_hT: SBUF ``[128, R, n_tiles]`` local attention-output partial of this rank.
        tp_group: the TP replica group.
        tag: name prefix of the collective and the transposes.
        tp_size: the TP group size (R must be a multiple of it).
        sb: the SBUF tile allocator.

    Returns:
        SBUF ``[128, R // tp_size, n_tiles]`` hT tile: the reduced tokens this rank keeps (one token when R == tp_size).
    """
    p_max, rows, n_tiles = part_hT.shape
    kernel_assert(rows % tp_size == 0, f"token_reduce_scatter needs R ({rows}) a multiple of tp_size ({tp_size})")
    kept = rows // tp_size
    hidden = p_max * n_tiles
    # Forward: [128, R, n_tiles] -> [R, hidden] (token t on partition t, hidden index j * 128 + p on the free dim).
    tokens = sb.tile((rows, n_tiles, p_max), part_hT.dtype)
    for j in range(n_tiles):
        psum = nl.ndarray((rows, p_max), dtype=part_hT.dtype, buffer=nl.psum)
        nisa.nc_transpose(dst=psum, data=part_hT[0:p_max, 0:rows, j])
        nisa.tensor_copy(dst=tokens[0:rows, j, :], src=psum)
    rs_dst = sb.tile((kept, hidden), part_hT.dtype)
    ncc.reduce_scatter(
        srcs=[tokens.reshape((rows, hidden))],
        dsts=[rs_dst],
        replica_group=tp_group,
        collective_dim=0,
        op=nl.add,
        name=f"{tag}_token_rs",
    )
    # Inverse: [kept, hidden] -> [128, kept, n_tiles] (hT of the kept tokens).
    out_hT = sb.tile((p_max, kept, n_tiles), part_hT.dtype)
    rs_view = rs_dst.reshape((kept, n_tiles, p_max))
    for j in range(n_tiles):
        psum = nl.ndarray((p_max, kept), dtype=part_hT.dtype, buffer=nl.psum)
        nisa.nc_transpose(dst=psum, data=rs_view[0:kept, j, :])
        nisa.tensor_copy(dst=out_hT[0:p_max, :, j], src=psum)
    return out_hT


def one_row_moe_glue(
    s_hT,
    y_row,
    bank_hT,
    t_rank,
    mlp_w,
    post_w,
    eps,
    ones,
    sb=AUTO_SBUF,
    t_pad=8,
    residual=True,
    name="k3_rs",
    fp16=False,
    n_bank=None,
):
    """Form the MoE-input glue of attn_rs_moe_input on the one token this rank keeps (token ``t_rank``).

    After the attention reduce-scatter this rank holds its one reduced row ``y_row`` and the full replica stream
    ``s_hT``. Read the stream row of token ``t_rank`` (a runtime-offset read), add the attention row for the residual,
    then run the MLP AttnRes mix and the post-attention norm on that one row. The mix runs through a tile padded to
    ``t_pad`` (> COMPACT_MIX_TOKENS) so it takes the same compact path as the all-row glue, and so the one-row result is
    bit-exact to the all-row row (a kernel test checks this). The bank is sliced to token ``t_rank`` the same way.

    Args:
        s_hT: SBUF ``[128, R, n_tiles]`` full replica stream (the layer input stream, replicated across the TP ranks).
        y_row: SBUF ``[128, 1, n_tiles]`` reduced attention output of this rank's token.
        bank_hT: the whole SBUF or HBM ``[128, C, R, n_tiles]`` AttnRes bank tensor (not a slice), or None for an empty
            bank. The bank read addresses the storage of the whole tensor, so its partition stride is ``C * R * n_tiles``.
        t_rank: INT32 ``[1, 1]`` SBUF tile, this rank's token index in its replica (its TP rank).
        mlp_w: SBUF fp32 ``[128, n_tiles]`` MLP AttnRes score weight.
        post_w: SBUF ``[128, n_tiles]`` post-attention norm weight.
        eps: norm epsilon. ones: all-ones fp32 ``[128, 128]`` SBUF tile.
        sb: the SBUF tile allocator. t_pad: the pad length of the mix (> COMPACT_MIX_TOKENS, default 8).
        fp16: also return the FP16 copy of the normed row from the FP32 norm values (``rmsnorm_hT`` ``out16``), the
            input of an FP16 router.
        n_bank: the filled entries of the bank (default: all ``C``). The verify bank holds every snapshot layer of the
            call, so a layer reads its first ``n_bank`` entries.

    Returns:
        ``(resid_row, moe_input)``: the ``[128, 1, n_tiles]`` residual row (for the stream all-gather) and the
        ``[128, 1, n_tiles]`` normed MoE input of this token; with ``fp16``, its FP16 copy as a third value.
    """
    p_max, rows, n_tiles = s_hT.shape
    kernel_assert(
        not isinstance(bank_hT, SplitBank),
        "one_row_moe_glue reads the bank rows in one DMA: it takes no split AttnRes bank (attn_res_sbuf_entries)",
    )
    capacity = 0 if bank_hT is None else bank_hT.shape[1]
    n_bank = capacity if n_bank is None else int(n_bank)
    kernel_assert(n_bank <= capacity, f"n_bank {n_bank} is above the bank capacity {capacity}")
    resid = sb.tile((p_max, 1, n_tiles), s_hT.dtype)
    if residual:
        # Runtime-offset read of the stream row of token t_rank: partition stride rows*n_tiles, the token axis indirect.
        s_row = sb.tile((p_max, 1, n_tiles), s_hT.dtype)
        nisa.dma_copy(
            dst=s_row,
            src=s_hT.ap(
                pattern=[[rows * n_tiles, p_max], [n_tiles, 1], [1, n_tiles]],
                offset=0,
                scalar_offset=t_rank,
                indirect_dim=1,
            ),
            name=f"{name}_stream_row",
        )
        nisa.tensor_tensor(dst=resid, data1=s_row, data2=y_row, op=nl.add)
    else:
        # A banks_snapshot layer resets the stream to the attention output (no residual add).
        nisa.tensor_copy(dst=resid, src=y_row)
    # Pad the one row to t_pad so the mix takes the compact path (bit-exact to the all-row row).
    resid_pad = sb.tile((p_max, t_pad, n_tiles), s_hT.dtype)
    for r in range(t_pad):
        nisa.tensor_copy(dst=resid_pad[0:p_max, r : r + 1, :], src=resid)
    bank_pad = None
    if n_bank:
        # Read the n_bank filled entries at token t_rank in one DMA: the token axis (dim 2) is the indirect one. ap()
        # addresses the storage of the whole bank, so the partition stride is the capacity C * rows * n_tiles.
        bank_t = sb.tile((p_max, n_bank, 1, n_tiles), bank_hT.dtype)
        nisa.dma_copy(
            dst=bank_t,
            src=bank_hT.ap(
                pattern=[[capacity * rows * n_tiles, p_max], [rows * n_tiles, n_bank], [n_tiles, 1], [1, n_tiles]],
                offset=0,
                scalar_offset=t_rank,
                indirect_dim=2,
            ),
            name=f"{name}_bank_rows",
        )
        bank_pad = sb.tile((p_max, n_bank, t_pad, n_tiles), bank_hT.dtype)
        for c in range(n_bank):
            for r in range(t_pad):
                nisa.tensor_copy(dst=bank_pad[0:p_max, c, r : r + 1, :], src=bank_t[0:p_max, c, 0:1, :])
    mixed = attn_res_hT(bank_pad, resid_pad, mlp_w, eps, ones, sb)
    h16_pad = sb.tile((p_max, t_pad, n_tiles), nl.float16) if fp16 else None
    h_pad = rmsnorm_hT(mixed, post_w, eps, ones, sb, **({"out16": h16_pad} if fp16 else {}))
    moe_input = sb.tile((p_max, 1, n_tiles), s_hT.dtype)
    nisa.tensor_copy(dst=moe_input, src=h_pad[0:p_max, 0:1, :])
    if not fp16:
        return resid, moe_input
    moe_input16 = sb.tile((p_max, 1, n_tiles), nl.float16)
    nisa.tensor_copy(dst=moe_input16, src=h16_pad[0:p_max, 0:1, :])
    return resid, moe_input, moe_input16


def tp_stream_gather(resid_row, tp_group, src_hbm, dst_hbm, rows, sb=AUTO_SBUF, name="k3_sprime"):
    """All-gather the one residual row of each rank of a replica into the full replica stream (the s' all-gather).

    After the one-row glue of attn_rs_moe_input each rank holds its token's residual ``resid_row`` [128, 1, n_tiles].
    The replica's updated stream s' = s + attention is the R rows gathered over the TP group. The gather runs through
    HBM with the collective_dim=0 partition-stack + permute-load path, so no transpose is needed: world (TP) rank t's
    row lands at token t of s', which is the replica's token order.

    Args:
        resid_row: SBUF [128, 1, n_tiles] residual row of this rank's token.
        tp_group: the TP replica group (``rows`` ranks).
        src_hbm: HBM [128, n_tiles] gather source. dst_hbm: HBM [rows * 128, n_tiles] gather destination.
        rows: R, the TP group size (the replica's token count).
        sb: the SBUF tile allocator.

    Returns:
        SBUF [128, rows, n_tiles] the full replica stream s'.
    """
    p_max, _, n_tiles = resid_row.shape
    nisa.dma_copy(dst=src_hbm, src=resid_row.reshape((p_max, n_tiles)), name=f"{name}_store")
    ncc.all_gather(srcs=[src_hbm], dsts=[dst_hbm], replica_group=tp_group, collective_dim=0, name=f"{name}_gather")
    s_prime = sb.tile((p_max, rows, n_tiles), resid_row.dtype)
    nisa.dma_copy(
        dst=s_prime.reshape((p_max, rows, n_tiles)),
        src=dst_hbm.reshape((rows, p_max, n_tiles)).permute((1, 0, 2)),
        name=f"{name}_load",
    )
    return s_prime
