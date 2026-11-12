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

"""GLM DSA sparse-attention indexer, bf16.

Projects Q, K and the per-head score weights, writes the paged K cache, all-gathers the CP context, scores Q against it in
position blocks under a causal mask, and returns the top-k positions per query.

``S_local`` is the queries this rank owns and ``S_global = cp_degree * S_local`` the context it
scores against; see :class:`GlmIndexerConfig`. Target envelope is ``S_local`` 2k-16k.
"""

import math
from dataclasses import dataclass

import nki.collectives as ncc
import nki.experimental
import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import dge_mode

from .utils.sbuf_scoped_alloc import SbufManager, scoped_name
from .utils.kernel_assert import kernel_assert
from .utils.kernel_helpers import div_ceil, get_max_positive_value_for_dtype, get_verified_program_sharding_info
from .utils.logging import get_logger
from .attention.common import RopeLayout
from .indexer_topk import (
    SEG_MAX,
    alloc_chunk_candidates,
    alloc_chunk_stage1_scratch,
    topk_chunk_merge,
    topk_chunk_stage1,
    topk_geometry,
    topk_over_score_chunked,
)

# --------------------------------------------------------------------------------------------
# Hardware constants
# --------------------------------------------------------------------------------------------

P_MAX = 128
BF16_K_TILE = 128
PSUM_FMAX = 512
_NUM_PSUM_BANKS = 8
MATMUL_TILE = 512
# Positions scored per PSUM tile -- ``MATMUL_TILE * 8``.
POS_BLOCK = 4096
KPOS_COLS = 512
"""Column width the key-position build works in, bounding its int32 scratch."""

SCORE_RING_MIN = 2

_ROLL_MIN_TILES = 4
"""Query tiles per core above which the projection tile loop runs as a device loop."""

# --------------------------------------------------------------------------------------------
# FP8 paged cache constants
# --------------------------------------------------------------------------------------------

# Floor on the per-token amax so an all-zero (padding) token cannot divide by zero; the row still
# quantizes to exact zeros. Matches the MLA fp8 cache.
_AMAX_FLOOR = 1e-30
# Headroom on the per-token scale before it is rounded to the row's bf16 field: rounding may move it
# down by up to 2^-9, which would push the amax element past fp8_amax (the next e4m3 code is inf).
_SCALE_HEADROOM = 1.0 + 2.0**-8
# Row-stride alignment for the fused fp8 cache row: 8 bf16 == 16 bytes, the DMA tran_size.
_KV_ROW_ALIGN_BF16 = 8
# Keys per PE transpose in the fp8 K pass. An fp8 transpose must write PSUM with an output element
# step of 2 (HW), so each transposed fp8 takes 2 bytes and a 2 KB bank holds 8 transposes.
_KT_TP = 128
_FP8_TP_OUT_STEP = 2
_KT_TP_PER_BANK = 8

def kv_row_bf16_width(head_dim: int) -> int:
    """bf16 containers per fused fp8 cache row, padded to a 16-byte (8 bf16) stride.

    The row holds ``head_dim`` fp8 K values bitcast two-per-bf16 in natural channel order
    (``head_dim // 2`` containers), then one bf16 per-token dequant scale, then padding -- the MLA
    fp8 cache's layout without its separate ``k_pe`` field, since the indexer quantizes its whole
    ``head_dim`` (RoPE included). GLM's ``head_dim=128`` gives ``64 + 1 = 65 -> 72`` bf16 (144 B).
    """
    return div_ceil(head_dim // 2 + 1, _KV_ROW_ALIGN_BF16) * _KV_ROW_ALIGN_BF16


def score_ring_slots(tiles_per_core: int) -> int:
    """Score staging slots needed: ONE PER TILE THIS CORE OWNS, not a fixed 2.

    The score phase writes every tile of a group across the position-block loop and only then ranks
    them, so a slot must survive from its write until its top-k reads it. With fewer slots than
    tiles, a later tile overwrites an unranked one and both rank the wrong rows.

    So the count is ceil(num_q_tiles / num_cores). Not yet sufficient for cp64 at 1M, where 64 tiles
    per core would need 34 GB; that shape needs the top-k fused per group with a shrunken panel.
    """
    return max(SCORE_RING_MIN, tiles_per_core)


# Usable SBUF per partition on gen3 (trn3): 224 KB - 16 KB - 8. gen4 has 240 KB.
SBUF_BUDGET_BYTES = 212984
_NAME_SEQ = [0]


def _uid() -> str:
    """Unique suffix for ``alloc_stack`` names.

    Names must be unique across the whole trace -- ``close_scope`` frees a scope's addresses but
    not its names -- and every loop here is Python-unrolled, so a literal name collides on the
    second iteration.
    """
    _NAME_SEQ[0] += 1
    return f"_{_NAME_SEQ[0]}"


def _is_dyn(row) -> bool:
    """Whether ``row`` is a device value (a rolled loop's induction variable) rather than a trace-time int."""
    return not isinstance(row, int)


def _row_slice(t: nl.NkiTensor, row, rows: int, cols: int) -> nl.NkiTensor:
    """``t[row : row + rows, 0 : cols]``, with ``row`` a trace-time int or a device value."""
    return t[nl.ds(row, rows), 0:cols] if _is_dyn(row) else t[row : row + rows, 0:cols]


# --------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GlmIndexerConfig:
    """The one shape this kernel serves.

    Frozen on purpose: add a field only when a second production shape genuinely needs it.

    Attributes:
        s_local: queries this rank owns (the design envelope is 2048-16384).
        s_global: context length each query scores against: ``cp_degree * s_kv_local``, which is
            ``cp_degree * s_local`` without a cached prefix and larger with one (APC).
        n_heads: indexer heads. GLM uses 32; the score sums over them weighted by ``weights_proj(x)``.
        head_dim: per-head width of the indexer Q/K.
        dim: model hidden size, the K projection's contraction.
        q_lora_rank: latent width of the Q input, the Q projection's contraction.
        rope_head_dim: leading slice of ``head_dim`` that carries RoPE.
        index_topk: positions selected per query.
    """

    s_local: int
    s_global: int
    n_heads: int = 32
    head_dim: int = 128
    dim: int = 6144
    q_lora_rank: int = 2048
    rope_head_dim: int = 64
    index_topk: int = 2048
    # Force the top-k segment count instead of deriving it from the row width. 0 = derive.
    topk_nseg: int = 0
    # Chunk the score at this many positions, ranking each chunk as it is produced. 0 = off (the whole row is scored, then ranked)
    score_chunk: int = 0
    # Roll the position-block loop with ``nl.fori_loop``. 0 = AUTO, 1 = force on, -1 = force off.
    roll_score: int = 0
    # CP interleave granularity of the K cache, in tokens. 0 = contiguous (one block per rank).
    cp_interleave: int = 0
    # Store the paged K cache in fp8 (per-token quant over head_dim, packed 2-per-bf16 with an
    # in-row scale). False keeps the bf16 cache path unchanged.
    kv_fp8: bool = False
    # CP ranks the K cache is interleaved across. 0 = derive ``s_global // s_local``, which holds only
    # without a cached prefix -- the single-rank ``k_ctx`` simulation relies on it.
    cp_ranks: int = 0
    # Pack two heads per score tile when the query tile uses <= 64 partitions (small-S / GK
    # prefill shards): halves the matmul/relu/accumulate instruction time of the score stage.
    pack_head_pairs: bool = False
    # Pack G = 128 // rows heads per score tile (8 at 16 queries/rank) and do the weighted
    # head-sum on PE: relu tiles [128 = (head, query), cols] contract against a per-tile
    # selection-weight stationary W[(h, r), r'] = w[r, h] * (r == r'), accumulating in PSUM.
    pack_heads: bool = False
    # Fast paths: double-buffered top-k batches
    # with a k/16-column decode on HW DGE, and a per-output-tile wq_b load so the Q matmuls start
    # after the first slice instead of after the whole 16.8 MB.
    fast_paths: bool = False
    # wq_b load chunk under fast_paths, in 512-column PSUM tiles (1 = per output tile).
    wq_chunk_tiles: int = 1
    # Chunked score: ping-pong the per-block K / key-position / accumulator buffers so block b+1's
    # K load and first accumulate do not wait (WAR) on block b's matmuls and score store.
    score_pingpong: bool = False
    # Single top-level device loop in the chunked score (the runtime's loop synchronization takes only one):
    # the chunk loop stays rolled, the position-block and top-k batch loops inside it unroll.
    flat_loops: bool = False
    # Accumulate the score in bf16 instead of fp32. Experimental: a 2-byte accumulator may let the
    # per-head weighted accumulate run in a faster Vector mode, at the cost of bf16 rounding per head.
    score_acc_bf16: bool = False
    # pack_heads: run the per-tile head-sum matmuls in PE column-tiling mode, 4 column tiles at once (one
    # per 32-column quadrant), issued as one burst per 2048 positions after that step's logits; their
    # results are masked and evicted once per 4 column tiles. PE then spends 1 slot per 4 head-sums
    # instead of 1 per head-sum (1 per 2 at 33-64 rows: each head-sum is two 32-row quadrants). Needs the
    # pipelined, hoisted, fused-mask chunked score with one query tile of <= 64 rows (see
    # score_block_for_group).
    col_headsum: bool = False
    # col_headsum on the rolled chunk path too (> _PIPELINE_MAX_CHUNKS chunks): key positions load per block
    # in the quadrant layout instead of from the pipelined path's resident block-periodic table.
    col_rolled: bool = False
    # Project Q after issuing the K all-gather (only K is gathered), so the wq load + Q matmuls run under it.
    q_after_gather: bool = False
    @property
    def score_acc_dtype(self):
        """dtype of the SBUF score accumulator."""
        return nl.bfloat16 if self.score_acc_bf16 else nl.float32

    @property
    def cp_degree(self) -> int:
        """CP ranks the context is split across."""
        return self.cp_ranks or max(1, self.s_global // self.s_local)

    @property
    def kv_cache_head_width(self) -> int:
        """Minor-dim width of one paged K cache row, in the cache's container dtype (bf16).

        bf16 cache: ``head_dim`` values. fp8 cache: the fused packed row (see
        :func:`kv_row_bf16_width`), which carries the fp8 K and its per-token scale.
        """
        return kv_row_bf16_width(self.head_dim) if self.kv_fp8 else self.head_dim

    @property
    def kw_out_dim(self) -> int:
        """Width of the fused K + weights projection: K's ``head_dim`` then ``n_heads`` weights."""
        return self.head_dim + self.n_heads

    @property
    def weights_scale(self) -> float:
        """``n_heads ** -0.5 * head_dim ** -0.5``, folded into the projected weights.

        The reference scores ``relu(q . k * head_dim**-0.5) * weights_proj(x) * n_heads**-0.5``; both
        scalars are positive and relu is positively homogeneous, so applying them to the weights is exact.
        """
        return (1.0 / math.sqrt(self.n_heads)) * (1.0 / math.sqrt(self.head_dim))

    @property
    def q_out_dim(self) -> int:
        """Width of the projected Q: all heads concatenated."""
        return self.n_heads * self.head_dim

    @property
    def q_panel_tiles(self) -> int:
        """Query tiles held in SBUF at once while K streams past (the "G" of the 2-D blocking).

        The score reuses BOTH operands: a query tile's Q is needed by every position block, and
        every K position is needed by every query tile. Holding one and streaming the other re-reads
        the other once per step -- at 1M that is ~34 GB either way. Holding G tiles instead makes
        each K read serve G tiles, cutting K traffic by G.

        G is derived from what actually fits, not chosen: per tile the panel costs a transposed Q
        (``n_heads * head_dim * 2`` B) plus an fp32 accumulator (``POS_BLOCK * 4`` B), against the
        budget less the shared K block and relu scratch. NOTE: when the top-k is fused in it will add
        ~12 KB/partition of running state PER TILE, which will force G down -- that dependency is
        why this is computed rather than hardcoded.
        """
        return max(
            1, min(self.num_q_tiles, (SBUF_BUDGET_BYTES - self._score_shared_bytes) // self._score_per_tile_bytes)
        )

    @property
    def _score_per_tile_bytes(self) -> int:
        """SBUF per partition the score scope holds PER PANEL TILE."""
        acc = POS_BLOCK * 4 * (2 if (self.score_pingpong and not self.score_acc_bf16) else 1)
        return self.n_heads * self.head_dim * 2 + acc + self.n_heads * 4 + 4

    @property
    def _chunk_topk_sbuf_bytes(self) -> int:
        """SBUF per partition the CHUNKED top-k holds live beside the panel. 0 when not chunked.

        Chunking ranks each chunk inside the score loop, so unlike the unfused path this scratch must
        coexist with the resident Q panel -- which is why the panel has to shrink. That is the price of
        the memory win: the panel divides K traffic, so fewer tiles means more K reads, traded against
        the score ring going from 16.4 GB to tens of MB at cp64/1M.
        """
        if not self.score_chunk:
            return 0
        k = self.index_topk
        n_ch = div_ceil(self.s_global, self.score_chunk)
        src_x = self.score_chunk // TOPK_SNAKE
        cand_x = n_ch * (k // TOPK_SNAKE)
        n_blk = div_ceil(cand_x, P_MAX)
        return (
            src_x * 2  # s1_src bf16
            + k * (2 + 4 + 4 + 4 + 4)  # s1 val/idx/pos/hi/lo
            + cand_x * (2 + 4)  # cand_v bf16 + cand_p int32
            + k * (2 + 4)  # merge val/idx
            + 2 * n_blk * P_MAX * 4  # oh_srct + oh_srct_b, full-width transposed candidates
            + src_x * 2
            + k * (2 + 4 + 4)  # this kernel's own per-chunk stage-1 scope
            + 8192  # one-hot constants, index rows and slack
        )

    @property
    def _score_shared_bytes(self) -> int:
        """SBUF per partition the score scope holds REGARDLESS of panel size.

        Count every term here, not only the K block and
        the relu scratch. Undercounting by ~49 KB lets G come out 7, which does not actually fit: the
        fused 128k compile has only 3908 B free, and the unfused one then fails with
        ``[NCC_IBVF029] q_stage and q_panel overlap in memory`` -- the 8 KB Q staging buffer had
        nowhere to go but on top of the panel. q_stage is counted here even though it is transient,
        because it must coexist with a full panel while the panel is being filled.
        """
        return (
            # K block (bf16, or fp8 plus its bf16 per-key scale row)
            POS_BLOCK * (3 if self.kv_fp8 else 2) * (2 if self.score_pingpong else 1)
            + POS_BLOCK * 4
            + POS_BLOCK * 2
            + POS_BLOCK * 4
            + self.n_heads * self.head_dim * 2
            + self._chunk_topk_sbuf_bytes
        )

    @property
    def num_q_tiles(self) -> int:
        """Query tiles of ``P_MAX`` rows needed to cover ``s_local``."""
        return div_ceil(self.s_local, P_MAX)

    def validate(self) -> None:
        """Check the assumptions the kernel body relies on, with the reason for each."""
        kernel_assert(
            self.dim % BF16_K_TILE == 0,
            f"dim must be a multiple of {BF16_K_TILE} so the K-major weight load is one "
            f"rectangular DMA, got {self.dim}",
        )
        kernel_assert(
            self.q_lora_rank % BF16_K_TILE == 0,
            f"q_lora_rank must be a multiple of {BF16_K_TILE} for the same reason, got {self.q_lora_rank}",
        )
        kernel_assert(
            self.kw_out_dim <= PSUM_FMAX,
            f"head_dim + n_heads must fit one PSUM bank ({PSUM_FMAX}) so K and the weights project in "
            f"a single matmul output tile, got {self.head_dim} + {self.n_heads}",
        )
        kernel_assert(
            self.rope_head_dim <= self.head_dim and self.rope_head_dim % 2 == 0,
            f"rope_head_dim must be even and <= head_dim, got {self.rope_head_dim}",
        )
        kernel_assert(
            self.s_global >= self.s_local,
            f"s_global ({self.s_global}) cannot be smaller than s_local ({self.s_local})",
        )


# --------------------------------------------------------------------------------------------
# Shared building blocks
# --------------------------------------------------------------------------------------------


def load_weight_k_major(
    sbm: SbufManager, w_hbm: nl.NkiTensor, k_dim: int, out_dim: int, col_chunk: int = 0, rec: list = None
) -> nl.NkiTensor:
    """Stage a bf16 ``[k_dim, out_dim]`` weight into SBUF as ``[P_MAX, k_tiles, out_dim]``.

    K-major because ``nc_matmul`` contracts over the PARTITION axis: the moving operand must
    present contraction rows on partitions. ``k_dim`` is a multiple of ``BF16_K_TILE``, so the
    source is perfectly rectangular and this lowers to ONE 3-D DMA (partition = row within tile,
    middle = tile, inner = column), which the hardware DGE can coalesce.

    Hoist this per batch, never per query tile: it does not depend on the queries.

    NOTE ON SIZE. The result is ``k_tiles * out_dim * 2`` bytes per partition. For Q that is
    ``16 * 4096 * 2 = 128 KB`` at GLM's 32 heads -- comfortable against the ~245 KB budget, but it
    scales with ``n_heads``, and at 128 heads it would be 384 KB and not fit. That is why the first
    indexer needed to shard this across cores, and why pinning ``n_heads = 32`` removes the need.

    Args:
        sbm: SBUF stack allocator.
        w_hbm: ``[k_dim, out_dim]`` bf16 weight in HBM.
        k_dim: contraction size (multiple of ``BF16_K_TILE``).
        out_dim: output columns.
        rec: a list the handle of every load DMA is appended to (trace-time bookkeeping; no instruction).

    Returns:
        ``[P_MAX, ceil(k_dim / BF16_K_TILE), out_dim]`` bf16 SBUF tile.
    """
    k_tiles = div_ceil(k_dim, BF16_K_TILE)
    w_sb = sbm.alloc_stack((P_MAX, k_tiles, out_dim), nl.bfloat16, name=f"w_k_major{_uid()}")
    step = col_chunk if col_chunk else out_dim
    for c0 in range(0, out_dim, step):
        cw = min(step, out_dim - c0)
        ld = nisa.dma_copy(
            dst=w_sb[0:BF16_K_TILE, 0:k_tiles, c0 : c0 + cw],
            src=w_hbm.ap(
                pattern=[
                    [out_dim, BF16_K_TILE],  # partition: row within the K tile
                    [BF16_K_TILE * out_dim, k_tiles],  # middle: which K tile
                    [1, cw],  # inner: output column, contiguous
                ],
                offset=c0,
            ),
            dge_mode=dge_mode.hwdge,
        )
        if rec is not None:
            rec.append(ld)
    return w_sb


def load_wk_wp_k_major(
    sbm: SbufManager,
    wk_hbm: nl.NkiTensor,
    wp_hbm: nl.NkiTensor,
    cfg: "GlmIndexerConfig",
    wp_transposed: bool = False,
    dma_engine=None,
) -> nl.NkiTensor:
    """Stage ``wk`` and ``weights_proj^T`` side by side as one K-major ``[P_MAX, k_tiles, head_dim + n_heads]``.

    Both contract over ``dim`` against the same ``x``, so concatenating their output columns lets
    :func:`project_k` produce K and the per-head weights from ONE matmul chain over the ``x`` tile it
    already holds transposed -- 160 moving columns instead of a second 48-matmul chain.

    ``weights_proj`` is ``[n_heads, dim]`` (``nn.Linear`` layout), so it is transposed in on the way:
    one ``dma_transpose`` per 128-row contraction tile, ``n_heads`` wide.
    """
    k_tiles = div_ceil(cfg.dim, BF16_K_TILE)
    w_sb = sbm.alloc_stack((P_MAX, k_tiles, cfg.kw_out_dim), nl.bfloat16, name=f"wkw_k_major{_uid()}", align=32)
    nisa.dma_copy(
        dst=w_sb[0:BF16_K_TILE, 0:k_tiles, 0 : cfg.head_dim],
        src=wk_hbm.ap(
            pattern=[
                [cfg.head_dim, BF16_K_TILE],
                [BF16_K_TILE * cfg.head_dim, k_tiles],
                [1, cfg.head_dim],
            ],
        ),
        dge_mode=dge_mode.hwdge,
        # A separate HW DGE queue (Scalar sequencer): this load is 256 B runs (a slow load)
        # and on the default Sync queue it stalls the wq_b stream queued behind it.
        **({"engine": dma_engine} if dma_engine is not None else {}),
    )
    if wp_transposed:
        # weights_proj arrives [dim, n_heads] (already K-major): one strided copy, no
        # dma_transposes to entangle the in-order queues ahead of the weight loads.
        nisa.dma_copy(
            dst=w_sb[0:BF16_K_TILE, 0:k_tiles, cfg.head_dim : cfg.kw_out_dim],
            src=wp_hbm.ap(
                pattern=[
                    [cfg.n_heads, BF16_K_TILE],
                    [BF16_K_TILE * cfg.n_heads, k_tiles],
                    [1, cfg.n_heads],
                ],
            ),
            dge_mode=dge_mode.hwdge,
            **({"engine": dma_engine} if dma_engine is not None else {}),
        )
        return w_sb
    for k_tile in range(k_tiles):
        nisa.dma_transpose(
            dst=w_sb[:BF16_K_TILE, k_tile, cfg.head_dim : cfg.kw_out_dim],
            src=wp_hbm[: cfg.n_heads, k_tile * BF16_K_TILE : (k_tile + 1) * BF16_K_TILE],
        )
    return w_sb


def load_activation_k_major(
    dst_sb: nl.NkiTensor,
    src_hbm: nl.NkiTensor,
    row_start: int,
    rows: int,
    k_dim: int,
    sbm: SbufManager = None,
    after: list = None,
) -> None:
    """Transpose a whole activation from HBM into SBUF K-major, one ``dma_transpose`` per 128-column
    contraction tile, all destinations distinct so every transfer can be in flight at once.

    A single ``dma_transpose`` does the whole load -- rather than ``dma_copy`` then
    query-major tile into SBUF, then ``nc_transpose`` it through PSUM. That removes the staging
    buffer, leaves the tensor engine entirely to the matmuls, and frees the transpose's PSUM banks.

    ``nc_matmul`` contracts over the PARTITION axis, so the activation must present contraction
    rows on partitions as the stationary operand; arriving transposed is exactly what is wanted.
    Per :func:`nki.isa.dma_transpose`, ``src`` may live in HBM and needs ``shape[-1] <= 128`` with a
    2-byte dtype -- satisfied by taking one ``BF16_K_TILE``-wide column block per call.

    ``dst_sb`` must be allocated with ``align=32``: the XBar descriptor verifier requires 32-byte
    alignment on a DMA-transpose destination, and the CPU simulator does not model that, so an
    unaligned destination passes simulation and fails on hardware.

    Args:
        dst_sb: ``[P_MAX, k_tiles, P_MAX]`` bf16 destination, allocated with ``align=32``.
        src_hbm: ``[rows_total, k_dim]`` bf16 activation in HBM.
        row_start: first row of this query tile.
        rows: real rows in this query tile.
        k_dim: full contraction size (a multiple of ``BF16_K_TILE``).
        sbm: SBUF allocator, needed only when ``row_start`` is a device value.
        after: (static ``row_start`` only) instructions every ``dma_transpose`` is scheduled after, with
            ``dependency.schedule`` edges: issue order for the compiler's schedulers, no semaphore, no instruction.

    With a device-valued ``row_start`` (a rolled tile loop) the load is a natural ``dma_copy`` plus
    Tensor Engine transposes instead: inside a device loop a ``dma_transpose`` off a kernel input is
    rejected ([NCC_IBIR251]). The PE transpose is bit-exact from NeuronCore-v3 on.
    """
    if _is_dyn(row_start):
        _load_activation_k_major_pe(sbm, dst_sb, src_hbm, row_start, rows, k_dim)
        return
    for k_tile in range(div_ceil(k_dim, BF16_K_TILE)):
        k_cols = min(BF16_K_TILE, k_dim - k_tile * BF16_K_TILE)
        tp = nisa.dma_transpose(
            dst=dst_sb[:k_cols, k_tile, :rows],
            src=src_hbm.ap(
                pattern=[
                    [k_dim, rows],
                    [1, k_cols],
                ],
                offset=row_start * k_dim + k_tile * BF16_K_TILE,
            ),
            axes=(1, 0),
        )
        if after:
            tp.depends_on(*after, kind=nisa.dependency.schedule)


def _load_activation_k_major_pe(sbm, dst_sb, src_hbm, row_start, rows: int, k_dim: int) -> None:
    """:func:`load_activation_k_major` through PSUM: natural loads, 8 transposes per PSUM bank.

    The natural-layout rows are staged one bank's worth (8 contraction tiles, 2 KB/partition) at a
    time in a 2-deep ring rather than whole: the whole ``x`` row block is 12 KB/partition at
    ``dim=6144``, which on top of the hoisted Q weight overflows SBUF.
    """
    kernel_assert(rows == P_MAX and k_dim % BF16_K_TILE == 0, f"PE K-major load needs whole tiles; rows={rows}")
    k_tiles = k_dim // BF16_K_TILE
    per_bank = 2 * PSUM_FMAX // P_MAX  # a 2 KB bank holds 8 bf16 [128, 128] transposes
    slab = per_bank * BF16_K_TILE
    sbm.open_scope(name="act_pe_load")
    nat = sbm.alloc_stack((P_MAX, 2, slab), nl.bfloat16, name=f"act_nat{_uid()}")
    for t0 in range(0, k_tiles, per_bank):
        n = min(per_bank, k_tiles - t0)
        c0, slot = t0 * BF16_K_TILE, (t0 // per_bank) % 2
        cols = n * BF16_K_TILE
        src = (
            src_hbm[nl.ds(row_start, P_MAX), c0 : c0 + cols]
            if _is_dyn(row_start)
            else src_hbm[row_start : row_start + P_MAX, c0 : c0 + cols]
        )
        nisa.dma_copy(dst=nat[:P_MAX, slot, :cols], src=src)
        ps = nl.ndarray((P_MAX, per_bank * P_MAX), dtype=nl.bfloat16, buffer=nl.psum)
        for j in range(n):
            nisa.nc_transpose(
                dst=ps[:P_MAX, j * P_MAX : (j + 1) * P_MAX],
                data=nat[:P_MAX, slot, j * BF16_K_TILE : (j + 1) * BF16_K_TILE],
            )
        nisa.tensor_copy(
            dst=dst_sb[:P_MAX, t0 : t0 + n, :P_MAX],
            src=ps[:P_MAX, 0 : n * P_MAX].reshape_dim(1, [n, P_MAX]),
        )
    sbm.close_scope()


#: GLM_GK_SYNC_HOL: ``[loads, wq_first]`` while the prefill GK traces a full layer's
#: indexer score, else ``[None, False]``. ``loads``: the head-TP QKV's per-head-group wq_b / W_uk load DMAs; every qr
#: ``dma_transpose`` of the score's Q projection is scheduled after them (``dependency.schedule``: issue order only, no
#: semaphore). All of them issue from the in-order Sync queue, and the transposes are held there until the CP key all-gather
#: in flight completes: ahead of the QKV's last two head groups' loads, they stall the QKV's phase B for the
#: whole all-gather. ``wq_first`` (GLM_GK_SYNC_HOL=2): the transposes also follow the score's own wq loads (load_weight_k_major),
#: so the wq load issues as soon as the QKV frees its SBUF instead of after the all-gather.
QR_T_AFTER = [None, False]


# --------------------------------------------------------------------------------------------
# Stage 1: projections
# --------------------------------------------------------------------------------------------


def project_q(
    sbm: SbufManager,
    qr_hbm: nl.NkiTensor,
    wq_sb: nl.NkiTensor,
    cfg: GlmIndexerConfig,
    row_start: int,
    rows: int,
    q_sb: nl.NkiTensor,
    rope: tuple = None,
    qr_t_pre: nl.NkiTensor = None,
    qr_t_after: list = None,
) -> None:
    """Project one query tile's latent Q: ``qr[rows, q_lora_rank] @ wq -> q_sb[rows, q_out_dim]``.

    ``q_out_dim`` (4096 at GLM) exceeds one PSUM bank, so the output is tiled by ``PSUM_FMAX`` and
    each output tile accumulates over the ``q_lora_rank / 128`` contraction tiles. Banks rotate so
    adjacent output tiles can issue without waiting for the previous bank to drain, and the
    accumulate flag is explicit: with rotation, auto-detect would treat a reused bank as
    previously-written and accumulate into its prior owner's contents.

    Args:
        sbm: SBUF stack allocator.
        qr_hbm: ``[s_local, q_lora_rank]`` bf16 latent Q.
        wq_sb: hoisted K-major Q weight from :func:`load_weight_k_major`.
        cfg: kernel configuration.
        row_start: first query row of this tile.
        rows: real query rows in this tile.
        q_sb: ``[P_MAX, q_out_dim]`` bf16 destination.
        qr_t_after: (GLM_GK_SYNC_HOL) instructions the qr ``dma_transpose`` path is scheduled after
            (``dependency.schedule``); see :data:`QR_T_AFTER`.
    """
    k_tiles = div_ceil(cfg.q_lora_rank, BF16_K_TILE)
    out_tiles = div_ceil(cfg.q_out_dim, PSUM_FMAX)

    sbm.open_scope(name="project_q")
    if qr_t_pre is not None:
        # q_after_gather: loaded (dma_transpose) before the K gather, which would hold it on the HWDGE queue.
        qr_t_sb = qr_t_pre
    else:
        qr_t_sb = sbm.alloc_stack((P_MAX, k_tiles, P_MAX), nl.bfloat16, name=f"qr_t{_uid()}", align=32)
        load_activation_k_major(
            qr_t_sb, qr_hbm, row_start, rows, cfg.q_lora_rank, sbm=sbm,
            **({"after": qr_t_after} if qr_t_after and not _is_dyn(row_start) else {}),
        )

    n_banks = min(out_tiles, _NUM_PSUM_BANKS)
    banks = [nl.ndarray((P_MAX, PSUM_FMAX), dtype=nl.float32, buffer=nl.psum) for _ in range(n_banks)]
    for out_tile in range(out_tiles):
        col0 = out_tile * PSUM_FMAX
        cols = min(PSUM_FMAX, cfg.q_out_dim - col0)
        bank = banks[out_tile % n_banks]
        for k_tile in range(k_tiles):
            nisa.nc_matmul(
                dst=bank[:rows, :cols],
                stationary=qr_t_sb[:BF16_K_TILE, k_tile, :rows],
                moving=wq_sb[:BF16_K_TILE, k_tile, col0 : col0 + cols],
                accumulate=(k_tile != 0),
            )
        if rope is not None:
            """
            RoPE HERE, in the fp32 PSUM bank, before the bf16 read-out.
            """
            cos_sb, sin_sb = rope
            kernel_assert(
                PSUM_FMAX % cfg.head_dim == 0,
                f"roping Q in the PSUM bank needs PSUM_FMAX ({PSUM_FMAX}) to be a whole number of "
                f"head_dim ({cfg.head_dim}); otherwise a head straddles two banks",
            )
            for h_in_tile in range(cols // cfg.head_dim):
                _rope_interleaved(
                    sbm,
                    bank,
                    cos_sb,
                    sin_sb,
                    bank,
                    rows,
                    cfg.rope_head_dim,
                    col_off=h_in_tile * cfg.head_dim,
                )
        nisa.tensor_copy(dst=q_sb[:rows, col0 : col0 + cols], src=bank[:rows, :cols])
    sbm.close_scope()


def project_k(
    sbm: SbufManager,
    x_hbm: nl.NkiTensor,
    wkw_sb: nl.NkiTensor,
    cfg: GlmIndexerConfig,
    row_start: int,
    rows: int,
    k_sb: nl.NkiTensor,
    w_sb: nl.NkiTensor,
) -> None:
    """Project one query tile's K and score weights from the same ``x`` tile.

    ``x[rows, dim] @ [wk | weights_proj^T] -> [k | w]``: K lands in ``k_sb``, and the per-head
    weights, scaled by :attr:`GlmIndexerConfig.weights_scale`, in ``w_sb``. ``head_dim + n_heads``
    fits one PSUM bank, so there is no output tiling. Output stays query-major because the next
    stage (LayerNorm + RoPE) wants it that way, and so does the score's per-row weight operand.

    Args:
        sbm: SBUF stack allocator.
        x_hbm: ``[s_local, dim]`` bf16 hidden states.
        wkw_sb: hoisted K-major fused weight from :func:`load_wk_wp_k_major`.
        cfg: kernel configuration.
        row_start: first row of this tile.
        rows: real rows in this tile.
        k_sb: ``[P_MAX, head_dim]`` f32 destination.
        w_sb: ``[P_MAX, n_heads]`` f32 destination.
    """
    k_tiles = div_ceil(cfg.dim, BF16_K_TILE)

    sbm.open_scope(name="project_k")
    x_t_sb = sbm.alloc_stack((P_MAX, k_tiles, P_MAX), nl.bfloat16, name=f"x_t{_uid()}", align=32)
    load_activation_k_major(x_t_sb, x_hbm, row_start, rows, cfg.dim, sbm=sbm)

    psum = nl.ndarray((P_MAX, cfg.kw_out_dim), dtype=nl.float32, buffer=nl.psum)
    for k_tile in range(k_tiles):
        nisa.nc_matmul(
            dst=psum[:rows, : cfg.kw_out_dim],
            stationary=x_t_sb[:BF16_K_TILE, k_tile, :rows],
            moving=wkw_sb[:BF16_K_TILE, k_tile, : cfg.kw_out_dim],
            accumulate=(k_tile != 0),
        )
    nisa.tensor_copy(dst=k_sb[:rows, : cfg.head_dim], src=psum[:rows, : cfg.head_dim])
    nisa.tensor_scalar(
        dst=w_sb[:rows, : cfg.n_heads],
        data=psum[:rows, cfg.head_dim : cfg.kw_out_dim],
        op0=nl.multiply,
        operand0=cfg.weights_scale,
    )
    sbm.close_scope()


# --------------------------------------------------------------------------------------------
# Stage 2: K post-processing (LayerNorm + RoPE)
# --------------------------------------------------------------------------------------------


def _layernorm(
    sbm: SbufManager,
    x_sb: nl.NkiTensor,
    gamma_sb: nl.NkiTensor,
    beta_sb: nl.NkiTensor,
    out_sb: nl.NkiTensor,
    S: int,
    H: int,
    eps: float = 1e-6,
) -> None:
    """LayerNorm with fused gamma/beta application.

    Uses nisa.bn_stats + nisa.bn_aggr to compute mean/var in one Vector Engine pass,
    then a fused scalar_tensor_tensor for ((x - mean) * rvar * gamma) + beta.

    gamma_sb/beta_sb are [P_MAX, H] (pre-broadcast by caller).
    """
    BN_STATS_TILE_SIZE = 512
    BN_STATS_DST_SIZE = 6
    NUM_BN_TILES = div_ceil(H, BN_STATS_TILE_SIZE)

    sbm.open_scope(name="layernorm")

    bn_stats_sb = sbm.alloc_stack((P_MAX, BN_STATS_DST_SIZE * NUM_BN_TILES), nl.float32, name=f"bn_stats_sb{_uid()}")
    aggr_sb = sbm.alloc_stack((P_MAX, 2), nl.float32, name=f"aggr_sb{_uid()}")  # [mean, rvar]

    for bn_tile_idx in nl.affine_range(NUM_BN_TILES):
        bn_off = bn_tile_idx * BN_STATS_TILE_SIZE
        bn_sz = min(BN_STATS_TILE_SIZE, H - bn_off)
        nisa.bn_stats(
            dst=bn_stats_sb[:S, nl.ds(bn_tile_idx * BN_STATS_DST_SIZE, BN_STATS_DST_SIZE)],
            data=x_sb[:S, nl.ds(bn_off, bn_sz)],
        )

    nisa.bn_aggr(
        dst=aggr_sb[:S, 0:2],
        data=bn_stats_sb[:S, 0 : BN_STATS_DST_SIZE * NUM_BN_TILES],
    )

    # rvar = 1 / sqrt(var + eps) in place.
    nisa.activation(
        dst=aggr_sb[:S, 1:2],
        op=nl.rsqrt,
        data=aggr_sb[:S, 1:2],
        bias=eps,
    )

    # x_norm = (x - mean) * rvar  (fused subtract + multiply via tensor_scalar two-op)
    x_norm = sbm.alloc_stack((P_MAX, H), nl.float32, name=f"x_norm{_uid()}")
    nisa.tensor_scalar(
        dst=x_norm[:S, :H],
        data=x_sb[:S, :H],
        op0=nl.subtract,
        operand0=aggr_sb[:S, 0:1],
        op1=nl.multiply,
        operand1=aggr_sb[:S, 1:2],
    )

    # out = x_norm * gamma, then + beta below (op0 identity multiply by 1.0, op1 multiplies gamma)
    nisa.scalar_tensor_tensor(
        dst=out_sb[:S, :H],
        data=x_norm[:S, :H],
        op0=nl.multiply,
        operand0=1.0,
        op1=nl.multiply,
        operand1=gamma_sb[:S, :H],
    )
    nisa.tensor_tensor(
        dst=out_sb[:S, :H],
        data1=out_sb[:S, :H],
        data2=beta_sb[:S, :H],
        op=nl.add,
    )

    sbm.close_scope()


# RoPE — non-interleaved, applied once per S-tile to all Q heads or to K.
def _rope_non_interleaved(
    sbm: SbufManager,
    x_sb: nl.NkiTensor,
    cos_sb: nl.NkiTensor,
    sin_sb: nl.NkiTensor,
    out_sb: nl.NkiTensor,
    S: int,
    D: int,
) -> None:
    """In-place non-interleaved RoPE on a [S, D] slab using minimum ops.

    out = [X1*cos - X2*sin, X1*sin + X2*cos]
    """
    D_half = D // 2
    sbm.open_scope(name="rope")

    t_neg_x2_sin = sbm.alloc_stack((P_MAX, D_half), nl.float32, name=f"t_neg_x2_sin{_uid()}")
    t_x1_sin = sbm.alloc_stack((P_MAX, D_half), nl.float32, name=f"t_x1_sin{_uid()}")
    t_x_cos = sbm.alloc_stack((P_MAX, D), nl.float32, name=f"t_x_cos{_uid()}")

    # -X2 * sin
    nisa.scalar_tensor_tensor(
        dst=t_neg_x2_sin[:S, :D_half],
        data=x_sb[:S, D_half:D],
        op0=nl.multiply,
        operand0=-1.0,
        op1=nl.multiply,
        operand1=sin_sb[:S, :D_half],
    )
    # X1 * sin
    nisa.tensor_tensor(
        dst=t_x1_sin[:S, :D_half],
        data1=x_sb[:S, :D_half],
        data2=sin_sb[:S, :D_half],
        op=nl.multiply,
    )
    # X * cos (full slab)
    # cos_sb is [S, D_half]; broadcast across the two halves.
    nisa.tensor_tensor(
        dst=t_x_cos[:S, :D_half],
        data1=x_sb[:S, :D_half],
        data2=cos_sb[:S, :D_half],
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=t_x_cos[:S, D_half:D],
        data1=x_sb[:S, D_half:D],
        data2=cos_sb[:S, :D_half],
        op=nl.multiply,
    )
    # First half of output: X1*cos - X2*sin = t_x_cos[:D/2] + t_neg_x2_sin
    nisa.tensor_tensor(
        dst=out_sb[:S, :D_half],
        data1=t_x_cos[:S, :D_half],
        data2=t_neg_x2_sin[:S, :D_half],
        op=nl.add,
    )
    # Second half: X1*sin + X2*cos = t_x1_sin + t_x_cos[D/2:]
    nisa.tensor_tensor(
        dst=out_sb[:S, D_half:D],
        data1=t_x1_sin[:S, :D_half],
        data2=t_x_cos[:S, D_half:D],
        op=nl.add,
    )

    sbm.close_scope()


def _rope_interleaved(
    sbm: SbufManager,
    x_sb: nl.NkiTensor,
    cos_sb: nl.NkiTensor,
    sin_sb: nl.NkiTensor,
    out_sb: nl.NkiTensor,
    S: int,
    D: int,
    col_off: int = 0,
) -> None:
    """INTERLEAVED (adjacent-pair) RoPE on [S, D] columns at ``col_off`` of a slab.

    Pair j is columns (2j, 2j+1), rotated by the same angle the half-split variant uses, so only the
    element pairing changes -- reached with stride-2 access patterns, identical op count.

    ``x_sb`` / ``out_sb`` are the FULL buffers, not ``[:, :D]`` slices: an ``ap()`` partition stride
    must equal the tensor's own free width, so a sliced view would carry the wrong stride.
    """
    D_half = D // 2
    x_row = x_sb.shape[1]
    out_row = out_sb.shape[1]
    sbm.open_scope(name="rope_il")

    x_even = x_sb.ap(pattern=[[x_row, S], [2, D_half]], offset=col_off)
    x_odd = x_sb.ap(pattern=[[x_row, S], [2, D_half]], offset=col_off + 1)
    o_even = out_sb.ap(pattern=[[out_row, S], [2, D_half]], offset=col_off)
    o_odd = out_sb.ap(pattern=[[out_row, S], [2, D_half]], offset=col_off + 1)

    # Stage the two cross terms first: out may alias x, so read x before overwriting it.
    t_even = sbm.alloc_stack((P_MAX, D_half), nl.float32, name=f"t_even{_uid()}")
    t_odd = sbm.alloc_stack((P_MAX, D_half), nl.float32, name=f"t_odd{_uid()}")
    nisa.tensor_tensor(dst=t_even[:S, :D_half], data1=x_odd, data2=sin_sb[:S, :D_half], op=nl.multiply)
    nisa.tensor_tensor(dst=t_odd[:S, :D_half], data1=x_even, data2=sin_sb[:S, :D_half], op=nl.multiply)

    # cos terms, then combine: out_even = x_even*cos - x_odd*sin ; out_odd = x_odd*cos + x_even*sin.
    nisa.tensor_tensor(dst=o_even, data1=x_even, data2=cos_sb[:S, :D_half], op=nl.multiply)
    nisa.tensor_tensor(dst=o_odd, data1=x_odd, data2=cos_sb[:S, :D_half], op=nl.multiply)
    nisa.tensor_tensor(dst=o_even, data1=o_even, data2=t_even[:S, :D_half], op=nl.subtract)
    nisa.tensor_tensor(dst=o_odd, data1=o_odd, data2=t_odd[:S, :D_half], op=nl.add)

    sbm.close_scope()


# K post-processing: LayerNorm + RoPE. Biased variance with eps inside the rsqrt, and INTERLEAVED
# (adjacent-pair) RoPE, which is GLM-MoE-DSA's pairing -- hence the layout argument.


def fused_layernorm_rope_k(
    sbm: SbufManager,
    k_sb: nl.NkiTensor,
    gamma_sb: nl.NkiTensor,
    beta_sb: nl.NkiTensor,
    cos_sb: nl.NkiTensor,
    sin_sb: nl.NkiTensor,
    k_out: nl.NkiTensor,
    S: int,
    head_dim: int,
    rope_head_dim: int,
    rope_layout: RopeLayout = RopeLayout.HALF_SPLIT,
) -> None:
    """LayerNorm K over the full head_dim, then RoPE the leading ``rope_head_dim`` columns in place.

    Any trailing nope columns are copied through unchanged.

    Args:
        sbm: SBUF stack allocator used for scratch buffers.
        k_sb: ``[S, head_dim]`` input K in SBUF.
        gamma_sb: ``[P_MAX, head_dim]`` LayerNorm scale (pre-broadcast by caller).
        beta_sb: ``[P_MAX, head_dim]`` LayerNorm bias (pre-broadcast by caller).
        cos_sb: ``[S, rope_head_dim // 2]`` RoPE cosine frequencies.
        sin_sb: ``[S, rope_head_dim // 2]`` RoPE sine frequencies.
        k_out: ``[S, head_dim]`` output buffer, written in place.
        S: active sequence length (partition rows) for this tile.
        head_dim: total head dimension.
        rope_head_dim: leading number of columns to apply RoPE to.
        rope_layout: element pairing; ``INTERLEAVED`` is GLM-MoE-DSA's.
    """
    sbm.open_scope(name="ln_rope_k")

    k_normed = sbm.alloc_stack((P_MAX, head_dim), nl.float32, name=f"k_normed{_uid()}")
    _layernorm(sbm, k_sb, gamma_sb, beta_sb, k_normed, S, head_dim)

    # RoPE on the rope_head_dim slice; copy nope tail through. The interleaved variant takes the
    # FULL buffers (see its docstring: ap() partition stride must be the parent row width).
    if rope_layout.is_interleaved():
        _rope_interleaved(sbm, k_normed, cos_sb, sin_sb, k_out, S, rope_head_dim)
    else:
        _rope_non_interleaved(
            sbm, k_normed[:, :rope_head_dim], cos_sb, sin_sb, k_out[:, :rope_head_dim], S, rope_head_dim
        )
    if head_dim > rope_head_dim:
        nisa.tensor_copy(dst=k_out[:, rope_head_dim:head_dim], src=k_normed[:, rope_head_dim:head_dim])

    sbm.close_scope()


def broadcast_norm_weight(sbm: SbufManager, w_hbm: nl.NkiTensor, head_dim: int, tag: str) -> nl.NkiTensor:
    """Stage a ``[head_dim]`` LayerNorm weight broadcast across all partitions.

    ``fused_layernorm_rope_k`` wants gamma/beta as ``[P_MAX, head_dim]`` so the fused
    ``scalar_tensor_tensor`` can read one value per (row, column) without a per-row reload. The
    broadcast is a view on the HBM side, so this is one DMA and costs 0.5 KB/partition.

    Loop-invariant -- hoist once per kernel, never per query tile.
    """
    w_sb = sbm.alloc_stack((P_MAX, head_dim), nl.float32, name=f"{tag}{_uid()}")
    nisa.dma_copy(dst=w_sb[:P_MAX, :head_dim], src=w_hbm.reshape((1, head_dim)).broadcast(dim=0, size=P_MAX))
    return w_sb


def load_rope_tables(
    sbm: SbufManager,
    cos_hbm: nl.NkiTensor,
    sin_hbm: nl.NkiTensor,
    cfg: "GlmIndexerConfig",
    row_start: int,
    rows: int,
) -> tuple:
    """This tile's ``cos``/``sin`` rows, shared by the Q and K rotations.

    ONE table for every head: HF's indexer passes ``unsqueeze_dim=2`` to
    ``apply_rotary_pos_emb_interleave``, so the angle depends on the token position only, not on the
    head. Loaded once per tile and handed to :func:`project_q`, which rotates each head in place.
    """
    rope_half = cfg.rope_head_dim // 2
    cos_sb = sbm.alloc_stack((P_MAX, rope_half), nl.float32, name=f"qcos{_uid()}")
    sin_sb = sbm.alloc_stack((P_MAX, rope_half), nl.float32, name=f"qsin{_uid()}")
    nisa.dma_copy(dst=cos_sb[:rows, :rope_half], src=_row_slice(cos_hbm, row_start, rows, rope_half))
    nisa.dma_copy(dst=sin_sb[:rows, :rope_half], src=_row_slice(sin_hbm, row_start, rows, rope_half))
    return cos_sb, sin_sb


def postprocess_k(
    sbm: SbufManager,
    k_sb: nl.NkiTensor,
    gamma_sb: nl.NkiTensor,
    beta_sb: nl.NkiTensor,
    cos_hbm: nl.NkiTensor,
    sin_hbm: nl.NkiTensor,
    cfg: GlmIndexerConfig,
    row_start: int,
    rows: int,
    k_out_sb: nl.NkiTensor,
) -> None:
    """LayerNorm the projected K, then RoPE its leading ``rope_head_dim`` columns.

    Reuses ``fused_layernorm_rope_k``: it is public, already validated on
    hardware for both RoPE pairings, and re-deriving LayerNorm's ``bn_stats``/``bn_aggr`` sequence
    here would duplicate numerically sensitive code for no gain. GLM uses ``INTERLEAVED`` pairing
    (adjacent columns ``2j``/``2j+1``), where the half-split RoPE uses ``HALF_SPLIT``; the angle set is the
    same, only the pairing differs. Inline it here if that package is ever retired.

    ``cos``/``sin`` carry ``rope_head_dim // 2`` angles per row and are indexed by ABSOLUTE query
    row, so each tile loads its own slice -- they are not loop-invariant like gamma/beta.

    Args:
        sbm: SBUF stack allocator.
        k_sb: ``[P_MAX, head_dim]`` fp32 projected K for this tile.
        gamma_sb: ``[P_MAX, head_dim]`` fp32 LayerNorm scale, pre-broadcast.
        beta_sb: ``[P_MAX, head_dim]`` fp32 LayerNorm bias, pre-broadcast.
        cos_hbm: ``[s_local, rope_head_dim // 2]`` fp32 cosine table.
        sin_hbm: ``[s_local, rope_head_dim // 2]`` fp32 sine table.
        cfg: kernel configuration.
        row_start: first query row of this tile.
        rows: real rows in this tile.
        k_out_sb: ``[P_MAX, head_dim]`` fp32 destination.
    """
    rope_half = cfg.rope_head_dim // 2

    sbm.open_scope(name="postprocess_k")
    cos_sb = sbm.alloc_stack((P_MAX, rope_half), nl.float32, name=f"cos{_uid()}")
    sin_sb = sbm.alloc_stack((P_MAX, rope_half), nl.float32, name=f"sin{_uid()}")
    nisa.dma_copy(dst=cos_sb[:rows, :rope_half], src=_row_slice(cos_hbm, row_start, rows, rope_half))
    nisa.dma_copy(dst=sin_sb[:rows, :rope_half], src=_row_slice(sin_hbm, row_start, rows, rope_half))

    fused_layernorm_rope_k(
        sbm,
        k_sb,
        gamma_sb,
        beta_sb,
        cos_sb,
        sin_sb,
        k_out_sb,
        rows,
        cfg.head_dim,
        cfg.rope_head_dim,
        rope_layout=RopeLayout.INTERLEAVED,
    )
    sbm.close_scope()


# --------------------------------------------------------------------------------------------
# Stage 3: paged K cache write
# --------------------------------------------------------------------------------------------


def quantize_k_per_token(
    sbm: SbufManager,
    k_sb: nl.NkiTensor,
    dst_fp8_ap: nl.NkiTensor,
    scale_sb: nl.NkiTensor,
    rows: int,
    head_dim: int,
    fp8_amax: float,
) -> None:
    """Quantize this tile's post-processed K to fp8, one scale per token, as the MLA fp8 cache does.

    Tokens sit on partitions and ``head_dim`` on the free axis, so the per-token amax is a free-axis
    reduce and the rescale a per-partition scalar. Per TOKEN because a top-k mixes rows from every
    CP rank and chunk, so only a scale that travels in its own row stays valid; one scale over all
    of ``head_dim`` factors out of the whole dot product, which is what lets the score skip dequant.

    The scale is rounded to bf16 (the row's field) BEFORE taking its reciprocal, with headroom so
    rounding cannot push the amax element past ``fp8_amax``; dequant is then an exact round trip.
    Writes the dequant factor to ``scale_sb`` (bf16) and the quantized K to ``dst_fp8_ap``.
    """
    sbm.open_scope(name="quantize_k")
    amax_sb = sbm.alloc_stack((P_MAX, 1), nl.float32, name=f"k_amax{_uid()}")
    neg_min_sb = sbm.alloc_stack((P_MAX, 1), nl.float32, name=f"k_neg_min{_uid()}")
    scale_f32_sb = sbm.alloc_stack((P_MAX, 1), nl.float32, name=f"k_scale_f32{_uid()}")
    recip_sb = sbm.alloc_stack((P_MAX, 1), nl.float32, name=f"k_qtz_recip{_uid()}")

    # max|k| as max(max(x), -min(x)): two free-axis reduces plus a combine, negate giving -min free.
    nisa.tensor_reduce(amax_sb[:rows, :], op=nl.maximum, data=k_sb[:rows, :head_dim], axis=1)
    nisa.tensor_reduce(neg_min_sb[:rows, :], op=nl.minimum, data=k_sb[:rows, :head_dim], axis=1, negate=True)
    nisa.tensor_tensor(amax_sb[:rows, :], amax_sb[:rows, :], neg_min_sb[:rows, :], op=nl.maximum)
    nisa.tensor_scalar(amax_sb[:rows, :], amax_sb[:rows, :], op0=nl.maximum, operand0=_AMAX_FLOOR)
    nisa.tensor_scalar(scale_sb[:rows, :], amax_sb[:rows, :], op0=nl.multiply, operand0=_SCALE_HEADROOM / fp8_amax)
    nisa.tensor_copy(dst=scale_f32_sb[:rows, :], src=scale_sb[:rows, :])
    nisa.reciprocal(recip_sb[:rows, :], scale_f32_sb[:rows, :])
    nisa.tensor_scalar(dst_fp8_ap, k_sb[:rows, :head_dim], op0=nl.multiply, operand0=recip_sb[:rows, :])
    sbm.close_scope()


def store_k_paged(
    sbm: SbufManager,
    k_sb: nl.NkiTensor,
    k_cache_hbm: nl.NkiTensor,
    slot_mapping_hbm: nl.NkiTensor,
    cfg: GlmIndexerConfig,
    row_start: int,
    rows: int,
    fp8_dtype=None,
    local_out: nl.NkiTensor = None,
) -> None:
    """Scatter this tile's post-processed K into the paged cache at the slots ``slot_mapping`` names.

    Paged-KV convention: ``slot_mapping[token]`` is a flat ``block_id * block_size + offset``, so the
    cache is addressed as ``[n_slots, kv_cache_head_width]`` and only the read side needs the block
    table. Padded tokens carry ``-1``, hence ``oob_mode.skip``. Stored TOKEN-major so a token's row
    is one descriptor.

    Two row layouts, selected by ``cfg.kv_fp8``:

    * bf16: ``head_dim`` bf16 values, cast from the fp32 K.
    * fp8: the fused packed row -- ``head_dim`` fp8 values bitcast two-per-bf16 in natural channel
      order, then the per-token bf16 dequant scale, then padding (see :func:`kv_row_bf16_width`).
      One dtype per row, so the scatter and the CP gather each stay one descriptor per token.

    Args:
        sbm: SBUF stack allocator.
        k_sb: ``[P_MAX, head_dim]`` fp32 K after LayerNorm + RoPE.
        k_cache_hbm: paged cache whose flattened leading dims are slots and whose minor dim is
            ``cfg.kv_cache_head_width``.
        slot_mapping_hbm: ``[s_local]`` int32 flat slot per token, ``-1`` for padding.
        cfg: kernel configuration.
        row_start: first query row of this tile (a trace-time int or a device value).
        rows: real rows in this tile.
        fp8_dtype: fp8 dtype of the packed latent; required when ``cfg.kv_fp8``.
        local_out: ``[s_local, kv_cache_head_width]`` buffer the CP gather reads. When given, the same
            row is written there too, so the gather moves exactly the stored (packed) row.
    """
    head_width = cfg.kv_cache_head_width
    n_slots = k_cache_hbm.size // head_width

    sbm.open_scope(name="store_k_paged")
    slot_sb = sbm.alloc_stack((P_MAX, 1), nl.int32, name=f"k_slot{_uid()}")
    nisa.dma_copy(
        dst=slot_sb[:rows, :],
        src=(
            slot_mapping_hbm.reshape((cfg.s_local,)).ap(
                pattern=[[1, rows], [1, 1]], offset=0, scalar_offset=row_start, indirect_dim=0
            )
            if _is_dyn(row_start)
            else slot_mapping_hbm.reshape((cfg.s_local,)).ap(pattern=[[1, rows], [1, 1]], offset=row_start)
        ),
        dge_mode=dge_mode.hwdge,
    )

    row_sb = sbm.alloc_stack((P_MAX, head_width), nl.bfloat16, name=f"k_row{_uid()}")
    if cfg.kv_fp8:
        kernel_assert(fp8_dtype is not None, "the fp8 cache store needs fp8_dtype")
        lat_bf16 = cfg.head_dim // 2
        # The latent span as fp8: under a 1-byte view the partition stride counts fp8 elements.
        lat_fp8_ap = row_sb.ap(pattern=[[head_width * 2, rows], [1, cfg.head_dim]], offset=0, dtype=fp8_dtype)
        quantize_k_per_token(
            sbm,
            k_sb,
            lat_fp8_ap,
            row_sb[:, lat_bf16 : lat_bf16 + 1],
            rows,
            cfg.head_dim,
            get_max_positive_value_for_dtype(fp8_dtype),
        )
    else:
        nisa.tensor_copy(dst=row_sb[:rows, : cfg.head_dim], src=k_sb[:rows, : cfg.head_dim])

    nisa.dma_copy(
        dst=k_cache_hbm.reshape((n_slots, head_width)).ap(
            pattern=[[head_width, rows], [1, head_width]],
            offset=0,
            vector_offset=slot_sb[:rows, :],
            indirect_dim=0,
        ),
        src=row_sb[:rows, :head_width],
        oob_mode=nisa.oob_mode.skip,
        dge_mode=dge_mode.swdge,
    )
    if local_out is not None:
        nisa.dma_copy(dst=_row_slice(local_out, row_start, rows, head_width), src=row_sb[:rows, :head_width])
    sbm.close_scope()


def build_kt_fp8(
    sbm: SbufManager,
    k_packed_hbm: nl.NkiTensor,
    kt_hbm: nl.NkiTensor,
    kscale_hbm: nl.NkiTensor,
    cfg: GlmIndexerConfig,
    end_pos: int,
    fp8_dtype,
) -> None:
    """Split packed fp8 rows ``[end_pos, kv_cache_head_width]`` into the score's K operands, no dequant.

    Writes ``kt_hbm[head_dim, end_pos]`` fp8 (channels on partitions, the matmul's moving layout) and
    ``kscale_hbm[end_pos]`` bf16 (the per-key dequant scale). The score multiplies Q (bf16) by the fp8
    K directly and applies each key's scale once after the head sum -- exact, since the scale is
    positive and relu is positively homogeneous -- so K is never dequantized.

    ``dma_transpose`` needs a 2-byte dtype and the packed row pairs two channels per container, so
    the transpose runs on PE instead: bit-accurate for fp8, 8 per PSUM bank (element step 2). Batched a POS_BLOCK at
    a time, one load and one store each, double buffered.
    """
    head_width = cfg.kv_cache_head_width
    lat_bf16 = cfg.head_dim // 2
    kernel_assert(cfg.head_dim == _KT_TP, f"the fp8 K pass transposes head_dim == {_KT_TP}; got {cfg.head_dim}")
    packed = k_packed_hbm.reshape((k_packed_hbm.size // head_width, head_width))
    ks_flat = kscale_hbm.reshape((kscale_hbm.size,))
    sub_stride = (POS_BLOCK // _KT_TP) * head_width * 2  # one partition's staged bytes, in fp8 elements

    sbm.open_scope(interleave_degree=2, name="kt_fp8")
    for c0 in range(0, end_pos, POS_BLOCK):
        cw = min(POS_BLOCK, end_pos - c0)
        n_full, tail = cw // _KT_TP, cw % _KT_TP
        n_sub = n_full + (1 if tail else 0)
        sub_rows = [_KT_TP] * n_full + ([tail] if tail else [])
        # Key c0 + j * 128 + p lands on partition p, sub-tile j.
        stage = sbm.alloc_stack((P_MAX, POS_BLOCK // _KT_TP, head_width), nl.bfloat16, name=f"ktp{_uid()}")
        if n_full:
            nisa.dma_copy(
                dst=stage[:P_MAX, :n_full, :head_width],
                src=packed.ap(
                    pattern=[[head_width, P_MAX], [_KT_TP * head_width, n_full], [1, head_width]],
                    offset=c0 * head_width,
                ),
            )
        if tail:
            t_lo = c0 + n_full * _KT_TP
            nisa.dma_copy(dst=stage[:tail, n_full, :head_width], src=packed[t_lo : t_lo + tail, :head_width])
        kt_sb = sbm.alloc_stack((P_MAX, POS_BLOCK), fp8_dtype, name=f"kto{_uid()}")
        for t0 in range(0, n_sub, _KT_TP_PER_BANK):
            n = min(_KT_TP_PER_BANK, n_sub - t0)
            ps = nl.ndarray((P_MAX, _KT_TP_PER_BANK * _KT_TP, _FP8_TP_OUT_STEP), dtype=fp8_dtype, buffer=nl.psum)
            for j in range(n):
                r = sub_rows[t0 + j]
                nisa.nc_transpose(
                    dst=ps[:P_MAX, j * _KT_TP : j * _KT_TP + r, 0],
                    data=stage.ap(
                        pattern=[[sub_stride, r], [1, cfg.head_dim]], offset=(t0 + j) * head_width * 2, dtype=fp8_dtype
                    ),
                )
            cols = sum(sub_rows[t0 : t0 + n])
            nisa.tensor_copy(
                dst=kt_sb[:P_MAX, t0 * _KT_TP : t0 * _KT_TP + cols],
                src=ps[:P_MAX, :cols, 0],
                engine=nisa.engine.vector if (t0 // _KT_TP_PER_BANK) % 2 == 0 else nisa.engine.scalar,
            )
        nisa.dma_copy(dst=kt_hbm[0 : cfg.head_dim, c0 : c0 + cw], src=kt_sb[: cfg.head_dim, :cw])
        # The scale column [keys, sub-tile] -> [sub-tile, keys], so each row stores contiguously.
        sc_sb = sbm.alloc_stack((P_MAX, P_MAX), nl.bfloat16, name=f"kts{_uid()}")
        if n_full:
            sc_ps = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=sc_ps[:n_full, :P_MAX], data=stage[:P_MAX, :n_full, lat_bf16])
            nisa.tensor_copy(dst=sc_sb[:n_full, :P_MAX], src=sc_ps[:n_full, :P_MAX])
            nisa.dma_copy(dst=ks_flat.ap(pattern=[[_KT_TP, n_full], [1, P_MAX]], offset=c0), src=sc_sb[:n_full, :P_MAX])
        if tail:
            tl_ps = nl.ndarray((1, P_MAX), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tl_ps[0:1, :tail], data=stage[:tail, n_full, lat_bf16 : lat_bf16 + 1])
            tl_sb = sbm.alloc_stack((1, P_MAX), nl.bfloat16, name=f"ktt{_uid()}")
            nisa.tensor_copy(dst=tl_sb[0:1, :tail], src=tl_ps[0:1, :tail])
            nisa.dma_copy(
                dst=ks_flat.ap(pattern=[[tail, 1], [1, tail]], offset=c0 + n_full * _KT_TP), src=tl_sb[0:1, :tail]
            )
        sbm.increment_section()
    sbm.close_scope()


def dequant_k_packed(
    sbm: SbufManager,
    k_packed_hbm: nl.NkiTensor,
    k_dense_hbm: nl.NkiTensor,
    cfg: GlmIndexerConfig,
    ctx: int,
    fp8_dtype,
) -> None:
    """[Testing] Dense bf16 K from packed fp8 rows, for the ``emit_proj`` gathered-K output only.

    The score never uses this -- it reads the fp8 K directly (see :func:`build_kt_fp8`).
    """
    head_width = cfg.kv_cache_head_width
    lat_bf16 = cfg.head_dim // 2
    packed = k_packed_hbm.reshape((k_packed_hbm.size // head_width, head_width))
    dense = k_dense_hbm.reshape((k_dense_hbm.size // cfg.head_dim, cfg.head_dim))
    sbm.open_scope(interleave_degree=2, name="dequant_k")
    for base in range(0, ctx, P_MAX):
        rows = min(P_MAX, ctx - base)
        row_sb = sbm.alloc_stack((P_MAX, head_width), nl.bfloat16, name=f"deq_row{_uid()}")
        k_sb = sbm.alloc_stack((P_MAX, cfg.head_dim), nl.bfloat16, name=f"deq_k{_uid()}")
        sc_sb = sbm.alloc_stack((P_MAX, 1), nl.float32, name=f"deq_sc{_uid()}")
        nisa.dma_copy(dst=row_sb[:rows, :head_width], src=packed[base : base + rows, :head_width])
        nisa.tensor_copy(dst=sc_sb[:rows, 0:1], src=row_sb[:rows, lat_bf16 : lat_bf16 + 1])
        nisa.tensor_scalar(
            dst=k_sb[:rows, : cfg.head_dim],
            data=row_sb.ap(pattern=[[head_width * 2, rows], [1, cfg.head_dim]], offset=0, dtype=fp8_dtype),
            op0=nl.multiply,
            operand0=sc_sb[:rows, 0:1],
        )
        nisa.dma_copy(dst=dense[base : base + rows, : cfg.head_dim], src=k_sb[:rows, : cfg.head_dim])
        sbm.increment_section()
    sbm.close_scope()


# --------------------------------------------------------------------------------------------
# Stage 4: CP all-gather of this chunk's K
# --------------------------------------------------------------------------------------------


def cp_replica_group(cp_degree: int, replica_groups: tuple = None) -> "ncc.ReplicaGroup":
    """The CP rank-lists this kernel's collectives run over.

    ``replica_groups`` comes from the framework, which is the only thing that knows how CP interacts
    with TP/PP -- a rank's CP peers are contiguous global ranks only when CP is the outermost split.
    It must be a compile-time list of rank-lists, not a tensor: the participant set defines the NEFF's
    collective topology. Defaults to ``[[0, cp_degree)]``, which is correct only for pure CP.
    """
    if replica_groups is None:
        return ncc.ReplicaGroup([list(range(cp_degree))])
    groups = [list(g) for g in replica_groups]
    kernel_assert(
        all(len(g) == cp_degree for g in groups),
        f"every CP replica group must hold cp_degree={cp_degree} ranks; got {[len(g) for g in groups]}",
    )
    return ncc.ReplicaGroup(groups)


def gather_k_across_cp(
    k_local_hbm: nl.NkiTensor,
    k_gathered_hbm: nl.NkiTensor,
    cp_degree: int,
    replica_groups: tuple = None,
) -> None:
    """All-gather this rank's chunk of K across the CP ranks, inside the kernel.

    Each CP rank projects K only for the ``s_local`` tokens it owns, but every rank's queries score
    against the whole ``s_global`` context, so the chunk has to be shared. Doing it in-kernel avoids
    the write-out / gather / read-back round trip a framework-level collective forces, which at
    CP64 / 1M is the dominant traffic in this kernel.

    ``collective_dim=0`` concatenates along the token axis, so rank r's ``s_local`` rows land at
    ``[r * s_local, (r+1) * s_local)`` of the destination -- the natural global order, provided the
    replica group lists CP ranks in context order.

    The destination is ``shared_hbm`` when LNC > 1: an all-gather produces identical output on every
    rank, so both NeuronCores on a stack can read one allocation instead of each holding a private
    copy, halving both capacity and collective bandwidth. With LNC=1 the source must be
    ``private_hbm`` per the ``nki.collectives.all_gather`` contract.

    Args:
        k_local_hbm: ``[s_local, head_dim]`` bf16, this rank's post-processed K.
        k_gathered_hbm: ``[cp_degree * s_local, head_dim]`` bf16 destination.
        cp_degree: CP ranks participating.
    """
    return ncc.all_gather(
        dsts=[k_gathered_hbm],
        srcs=[k_local_hbm],
        replica_group=cp_replica_group(cp_degree, replica_groups),
        collective_dim=0,
    )


# --------------------------------------------------------------------------------------------
# Stage 5: blocked score
# --------------------------------------------------------------------------------------------


HEAD_CHUNK = 8
"""Heads transposed per ``dma_transpose``.

A 3-D transpose with ``axes=(2, 1, 0)`` flips ``[rows, HEAD_CHUNK, head_dim]`` to
``[head_dim, HEAD_CHUNK, rows]``, producing EVERY head in the chunk in one DMA. Per-head transposes
cost ``n_heads`` DMA ops instead of ``n_heads / HEAD_CHUNK``."""


def linearize_pages(
    sbm: SbufManager,
    k_cache_hbm: nl.NkiTensor,
    block_table_hbm: nl.NkiTensor,
    k_lin_hbm: nl.NkiTensor,
    cfg: "GlmIndexerConfig",
    end_pos: int,
    block_size: int,
) -> None:
    """Gather the paged cache into CONTIGUOUS logical order: ``k_lin[p] = cache[slot(p)]``.

    ``slot(p) = block_table[p // block_size] * block_size + p % block_size``, the paged-KV convention
    the store side already uses. Reading through the table is what lets a resident prefix be scored.

    Linearized in HBM rather than transposed straight from the pages: the score's ``dma_transpose``
    needs an HBM source, and staging pages token-major in SBUF first cannot work because 4096 tokens
    do not fit in 128 partitions. Costs one extra ``T * head_dim`` bf16 tensor and one pass over K.

    This also covers CP token interleaving: the layout is expressed by which ids the table lists and
    in what order, so neither the score nor the mask needs an interleave parameter.
    """
    kernel_assert(
        POS_BLOCK % block_size == 0,
        f"block_size ({block_size}) must divide POS_BLOCK ({POS_BLOCK}); a partial page at a block "
        f"boundary would need a masked tail read",
    )
    n_blocks = div_ceil(end_pos, block_size)
    head_width = cfg.kv_cache_head_width
    page = block_size * head_width
    n_pages_cache = k_cache_hbm.size // page
    rows_total = n_blocks * block_size

    sbm.open_scope(name="linearize_pages")
    # BATCHED GATHER: with a page as the unit row, the index tensor IS the block table -- one
    # descriptor moves 128*M pages instead of one DMA per page. The hardware reads a 2D
    # vector_offset column-major (indices.T.flatten()), so partition i / column j must hold
    # block_table[j * P_MAX + i] to land pages in linear order.
    # A partial last batch (n_blocks % 128 != 0, e.g. 64 pages at a 4096-position context) is
    # gathered the same way on fewer partitions; the per-page fallback below costs a
    # serialized SW-DGE issue PER PAGE.
    if k_lin_hbm.size // head_width >= rows_total and k_lin_hbm.size % page == 0:
        cache_pages = k_cache_hbm.reshape((n_pages_cache, page))
        lin_pages = k_lin_hbm.reshape((k_lin_hbm.size // page, page))
        idx = sbm.alloc_stack((P_MAX, 1), nl.uint32, name=f"bt{_uid()}")
        # An indirect DMA needs its non-indirect side in SBUF ([NCC_IBIR231]), so the gather stages
        # 128 whole pages in SBUF and one contiguous write pushes them out.
        stage = sbm.alloc_stack((P_MAX, page), nl.bfloat16, name=f"pg{_uid()}", align=32)
        for gb in range(0, n_blocks, P_MAX):
            n = min(P_MAX, n_blocks - gb)
            nisa.dma_copy(
                dst=idx[0:n, 0:1],
                src=block_table_hbm.reshape((block_table_hbm.size, 1))[gb : gb + n, 0:1],
            )
            nisa.dma_copy(
                dst=stage[0:n, 0:page],
                src=cache_pages.ap(
                    pattern=[[page, n], [1, page]],
                    offset=0,
                    vector_offset=idx[0:n, 0:1],
                    indirect_dim=0,
                ),
                dge_mode=dge_mode.swdge,
            )
            nisa.dma_copy(dst=lin_pages[gb : gb + n, 0:page], src=stage[0:n, 0:page])
        sbm.close_scope()
        return

    # Fallback: one descriptor per page. Page ids pre-scaled to a ROW offset, since scalar_offset
    # counts the access pattern's leading stride.
    flat_cache = k_cache_hbm.reshape((k_cache_hbm.size // head_width, head_width))
    bt = sbm.alloc_stack((1, n_blocks), nl.uint32, name=f"bt{_uid()}")
    nisa.dma_copy(
        dst=bt[0:1, 0:n_blocks],
        src=block_table_hbm.reshape((1, block_table_hbm.size))[0:1, 0:n_blocks],
    )
    nisa.tensor_scalar(
        dst=bt[0:1, 0:n_blocks],
        data=bt[0:1, 0:n_blocks],
        op0=nl.multiply,
        operand0=block_size,
        engine=nisa.engine.vector,
    )
    for b in range(n_blocks):
        rows = min(block_size, end_pos - b * block_size)
        nisa.dma_copy(
            dst=k_lin_hbm[b * block_size : b * block_size + rows, 0:head_width],
            src=flat_cache.ap(
                pattern=[[head_width, rows], [1, head_width]],
                offset=0,
                scalar_offset=bt.ap(pattern=[[n_blocks, 1], [1, 1]], offset=b),
            ),
        )
    sbm.close_scope()


def load_q_panel_transposed(
    sbm: SbufManager,
    q_hbm: nl.NkiTensor,
    qt_sb: nl.NkiTensor,
    cfg: GlmIndexerConfig,
    tile: int,
    rows: int,
    slot: int,
    row=None,
) -> None:
    """Load one query tile's Q as ``[head_dim, n_heads, queries]`` -- contraction on partitions.

    ``row``, when given, is the tile's first query row and may be a device value (a rolled group
    loop's induction variable); it replaces ``tile * P_MAX``.

    ``nc_matmul`` contracts over partitions, so the score's stationary operand must present
    ``head_dim`` there, while Q is stored query-major. The flip is done as ``n_heads / HEAD_CHUNK``
    three-dimensional ``dma_transpose``es rather than one per head: viewing the staged tile as
    ``[rows, HEAD_CHUNK, head_dim]`` and transposing with ``axes=(2, 1, 0)`` emits a whole chunk of
    heads per DMA. ``head_dim`` is 128, exactly ``dma_transpose``'s ``shape[-1]`` limit.

    Staging through SBUF (rather than transposing straight from HBM) is what makes the 3-D view
    possible, and the staging copy is one contiguous DMA of 8 KB/partition.
    """
    sbm.open_scope(name="q_panel_load")
    stage_sb = sbm.alloc_stack((P_MAX, cfg.q_out_dim), nl.bfloat16, name=f"q_stage{_uid()}", align=32)
    q_rows = (
        q_hbm[tile * P_MAX : tile * P_MAX + rows, : cfg.q_out_dim]
        if row is None
        else q_hbm[nl.ds(row, rows), 0 : cfg.q_out_dim]
    )
    nisa.dma_copy(dst=stage_sb[:rows, : cfg.q_out_dim], src=q_rows)
    transpose_q_tile_to_panel(stage_sb, qt_sb, cfg, rows, slot)
    sbm.close_scope()


def transpose_q_tile_to_panel(q_sb: nl.NkiTensor, qt_sb: nl.NkiTensor, cfg, rows: int, slot: int) -> None:
    """``qt_sb[:, slot, h, q] = q_sb[q, h * head_dim : (h + 1) * head_dim]``, one 3-D transpose per head chunk."""
    for h0 in range(0, cfg.n_heads, HEAD_CHUNK):
        n_h = min(HEAD_CHUNK, cfg.n_heads - h0)
        chunk_view = q_sb[:rows, h0 * cfg.head_dim : (h0 + n_h) * cfg.head_dim].reshape_dim(1, [n_h, cfg.head_dim])
        nisa.dma_transpose(
            dst=qt_sb[: cfg.head_dim, slot, h0 : h0 + n_h, :rows],
            src=chunk_view,
            axes=(2, 1, 0),
        )


def _ts(dst, src, op, imm, engine):
    """One ``tensor_scalar`` with an immediate, on a named engine."""
    nisa.tensor_scalar(dst, src, op0=op, operand0=imm, engine=engine)


def build_key_positions(sbm, dst_hbm, s_kv_local, cp_degree, interleave):
    """Fill ``dst_hbm[g]`` with the global position of gathered key ``g``, once per kernel.

    ``owned(rank, l) = (l // I) * (I*W) + rank * I + (l % I)`` for ``g = rank * s_kv_local + l``.
    Built rank by rank, so ``rank`` and ``l`` are loop indices rather than a division of ``g``: that
    is what lets ``s_kv_local`` be any multiple of ``I`` (a cached prefix of n blocks per rank), while
    ``I`` and ``W`` stay powers of two so the per-``l`` part is shifts and masks. The ``l`` part is
    rank-invariant, so it is computed once per column chunk and each rank only adds ``rank * I``.

    Each rank's ``s_kv_local`` slots are laid out ``[gcd(s_kv_local, 128), cols]`` so every op spans
    as many partitions as the slice allows. Only ``[0, cp_degree * s_kv_local)`` is written; the
    score never reads past ``end_pos``, which the caller pins to exactly that.
    """
    lg_i = int(math.log2(interleave))
    lg_p = int(math.log2(interleave * cp_degree))
    rows = math.gcd(s_kv_local, P_MAX)
    cols = s_kv_local // rows
    vec = nisa.engine.vector

    sbm.open_scope(name="key_positions")
    cw = min(cols, KPOS_COLS)
    l_sb = sbm.alloc_stack((P_MAX, cw), nl.int32, name=f"kpl{_uid()}")
    h = sbm.alloc_stack((P_MAX, cw), nl.int32, name=f"kph{_uid()}")
    base = sbm.alloc_stack((P_MAX, cw), nl.float32, name=f"kpb{_uid()}")
    # Ranks are batched along a free axis: out[p, r, j] = base[p, j] + (r0 + r) * I, ONE add and
    # ONE DMA per rank batch instead of a Vector->DMA round trip per rank (cp_degree serialized
    # hops: 64-128 of them at CP64/CP128). The rank-offset tile is loop-invariant.
    rb = max(1, min(cp_degree, 4096 // cw))
    ro_i = sbm.alloc_stack((P_MAX, rb, cw), nl.int32, name=f"kpri{_uid()}")
    ro = sbm.alloc_stack((P_MAX, rb, cw), nl.float32, name=f"kprf{_uid()}")
    nisa.iota(dst=ro_i[:rows, :, :], pattern=[[interleave, rb], [0, cw]], offset=0, channel_multiplier=0)
    nisa.tensor_copy(dst=ro[:rows, :, :], src=ro_i[:rows, :, :])
    out_bufs = [sbm.alloc_stack((P_MAX, rb, cw), nl.float32, name=f"kpo{_uid()}_{i}") for i in range(2)]
    n_batch = 0
    for c in range(0, cols, cw):
        w = min(cw, cols - c)
        # l = i * cols + c + j, so one DMA out preserves linear order within the rank's slice.
        nisa.iota(dst=l_sb[:rows, :w], pattern=[[1, w]], offset=c, channel_multiplier=cols)
        _ts(h[:rows, :w], l_sb[:rows, :w], nl.right_shift, lg_i, vec)
        _ts(h[:rows, :w], h[:rows, :w], nl.left_shift, lg_p, vec)  # (l // I) * (I*W)
        _ts(l_sb[:rows, :w], l_sb[:rows, :w], nl.bitwise_and, interleave - 1, vec)  # l % I
        nisa.tensor_tensor(h[:rows, :w], h[:rows, :w], l_sb[:rows, :w], op=nl.add, engine=vec)
        nisa.tensor_copy(dst=base[:rows, :w], src=h[:rows, :w])
        for r0 in range(0, cp_degree, rb):
            nr = min(rb, cp_degree - r0)
            out = out_bufs[n_batch % 2]
            n_batch += 1
            nisa.tensor_tensor(
                out[:rows, :nr, :w],
                ro[:rows, :nr, :w],
                base[:rows, :w].expand_dim(dim=1).broadcast(dim=1, size=nr),
                op=nl.add,
                engine=vec,
            )
            if r0:
                nisa.tensor_scalar(
                    dst=out[:rows, :nr, :w], data=out[:rows, :nr, :w], op0=nl.add, operand0=float(r0 * interleave)
                )
            nisa.dma_copy(
                dst=dst_hbm.ap(pattern=[[cols, rows], [s_kv_local, nr], [1, w]], offset=r0 * s_kv_local + c),
                src=out[:rows, :nr, :w],
            )
    sbm.close_scope()


def apply_causal_mask(
    sbm: SbufManager,
    acc_sb: nl.NkiTensor,
    kpos_sb: nl.NkiTensor,
    qpos_sb: nl.NkiTensor,
    slot: int,
    rows: int,
    blk: int,
    kscale_sb: nl.NkiTensor = None,
) -> None:
    """Add ``MASK_NEG`` wherever position p is not visible to query q: ``k_pos[p] > q_pos[q]``.

    With an fp8 cache, ``kscale_sb`` holds each key's dequant scale broadcast across partitions, and
    the head sum is scaled by it first -- the fp8 K's whole dequant, applied once per block.

    Applied once per (query tile, position block) AFTER the head sum, since visibility is positional
    and independent of h. ``kpos_sb`` holds the intra-block ramp j and ``qpos_sb`` the block-relative
    query position, so the comparison is the same one global positions would give.
    """
    sbm.open_scope(name="causal_mask")
    if kscale_sb is not None:
        nisa.tensor_tensor(
            dst=acc_sb[:rows, slot, :blk], data1=acc_sb[:rows, slot, :blk], data2=kscale_sb[:rows, :blk], op=nl.multiply
        )
    mask_sb = sbm.alloc_stack((P_MAX, POS_BLOCK), nl.float32, name=f"mask{_uid()}")
    # mask01[q, p] = (k_pos[p] - q_pos[q]) > 0
    nisa.tensor_scalar(
        dst=mask_sb[:rows, :blk],
        data=kpos_sb[:rows, :blk],
        op0=nl.subtract,
        operand0=qpos_sb[:rows, slot : slot + 1],
        op1=nl.greater,
        operand1=0.0,
    )
    nisa.scalar_tensor_tensor(
        dst=acc_sb[:rows, slot, :blk],
        data=mask_sb[:rows, :blk],
        op0=nl.multiply,
        operand0=MASK_NEG,
        op1=nl.add,
        operand1=acc_sb[:rows, slot, :blk],
    )
    sbm.close_scope()


MASK_SCALE = -(2.0**102)
"""The fused causal mask works on positions pre-scaled by -2^102. A power of
two keeps every product exact (positions < 2^24, so |products| < 2^126), and 2^102 * (q + 0.5 - k) is
>= 2^101 (far above any score) where key k is visible to query q and <= -2^101 ~ -2.5e30 (below the
-1e30 a masked score is taken to be) where it is not."""


def scale_key_positions(sbm, kpos_hbm, ctx_row: int):
    """``kposS[p] = MASK_SCALE * kpos[p]`` in HBM, once per kernel."""
    cols = ctx_row // P_MAX
    out = nl.ndarray((ctx_row, 1), dtype=nl.float32, buffer=nl.private_hbm, name=scoped_name(f"kpos_scaled{_uid()}"))
    sbm.open_scope(name="kpos_scale")
    t = sbm.alloc_stack((P_MAX, cols), nl.float32, name=f"kps{_uid()}")
    nisa.dma_copy(dst=t[:P_MAX, :cols], src=kpos_hbm.reshape((P_MAX, cols)))
    nisa.tensor_scalar(dst=t[:P_MAX, :cols], data=t[:P_MAX, :cols], op0=nl.multiply, operand0=MASK_SCALE)
    nisa.dma_copy(dst=out.reshape((P_MAX, cols)), src=t[:P_MAX, :cols])
    sbm.close_scope()
    return out


def apply_causal_mask_min(acc_dst, acc_src, kpos_s_sb, qpos_s_sb, slot: int, rows: int, blk: int, col: int = 0) -> None:
    """``acc_dst = min(acc_src, kposS - qposS)``: the whole causal mask in ONE Vector op.

    ``kpos_s_sb`` holds ``MASK_SCALE * kpos`` for the block and ``qpos_s_sb`` ``MASK_SCALE * (qpos + 0.5)``,
    so ``kposS - qposS = 2^102 * (qpos + 0.5 - kpos)``: huge positive where the key is visible (the min
    keeps the score), huge negative where it is not. ``acc_src`` may be the PSUM tile being evicted,
    which folds the eviction copy in too. ``col`` offsets the key-position columns.
    """
    nisa.scalar_tensor_tensor(
        dst=acc_dst,
        data=kpos_s_sb[:rows, col : col + blk],
        op0=nl.subtract,
        operand0=qpos_s_sb[:rows, slot : slot + 1],
        op1=nl.minimum,
        operand1=acc_src,
    )


def build_pack_heads_consts(sbm, qt_sb, w_sb, cfg, n_in_group: int, tile_rows: list):
    """The ``pack_heads`` operands of one query panel: packed Q (``q8``) and selection weights (``wsel``).

    Factored out of :func:`score_block_for_group` so a caller can build them
    ONCE per query panel instead of once per position block (they depend only on Q and the head weights).
    """
    rows = tile_rows[0]
    G = P_MAX // rows
    ng = cfg.n_heads // G
    hd = cfg.head_dim
    # eye[p, r'] = ((p % rows) == r'), shared by every tile.
    pi = sbm.alloc_stack((P_MAX, rows), nl.int32, name=f"pk_pi{_uid()}")
    nisa.iota(dst=pi, pattern=[[0, rows]], offset=0, channel_multiplier=1)
    nisa.tensor_scalar(dst=pi, data=pi, op0=nl.bitwise_and, operand0=rows - 1, engine=nisa.engine.vector)
    ji = sbm.alloc_stack((P_MAX, rows), nl.int32, name=f"pk_ji{_uid()}")
    nisa.iota(dst=ji, pattern=[[1, rows]], offset=0, channel_multiplier=0)
    eye = sbm.alloc_stack((P_MAX, rows), nl.float32, name=f"pk_eye{_uid()}")
    nisa.tensor_tensor(dst=eye, data1=pi, data2=ji, op=nl.equal, engine=nisa.engine.vector)
    q8s, wsels = [], []
    w_pstride = w_sb.shape[1] * w_sb.shape[2]
    q_pstride = qt_sb.shape[1] * qt_sb.shape[2] * qt_sb.shape[3]
    # Query columns per head in the panel: P_MAX, or fewer for a panel sized to its tile.
    qw = qt_sb.shape[3]
    for i in range(n_in_group):
        # Q packed: q8[:, g, h*rows + r] = qt[:, i, G*g + h, r]
        q8 = sbm.alloc_stack((P_MAX, ng, G * rows), nl.bfloat16, name=f"pk_q{_uid()}", align=32)
        for h in range(G):
            nisa.tensor_copy(
                dst=q8[:hd, :, h * rows : (h + 1) * rows],
                src=qt_sb.ap(
                    pattern=[[q_pstride, hd], [G * qw, ng], [1, rows]],
                    offset=(i * cfg.n_heads + h) * qw,
                ),
            )
        # T[h*rows + r, g] = w[r, i, G*g + h]  (partition move -> DMA)
        T = sbm.alloc_stack((P_MAX, ng), nl.float32, name=f"pk_t{_uid()}")
        for h in range(G):
            nisa.dma_copy(
                dst=T[h * rows : (h + 1) * rows, 0:ng],
                src=w_sb.ap(pattern=[[w_pstride, rows], [G, ng]], offset=i * cfg.n_heads + h),
            )
        wsel = sbm.alloc_stack((P_MAX, ng, rows), nl.bfloat16, name=f"pk_w{_uid()}")
        nisa.tensor_tensor(
            dst=wsel,
            data1=T.expand_dim(dim=2).broadcast(dim=2, size=rows),
            data2=eye.expand_dim(dim=1).broadcast(dim=1, size=ng),
            op=nl.multiply,
            engine=nisa.engine.vector,
        )
        q8s.append(q8)
        wsels.append(wsel)
    return q8s, wsels


def score_block_for_group(
    sbm: SbufManager,
    qt_sb: nl.NkiTensor,
    kt_sb: nl.NkiTensor,
    w_sb: nl.NkiTensor,
    acc_sb: nl.NkiTensor,
    cfg: GlmIndexerConfig,
    n_in_group: int,
    tile_rows: list,
    blk: int,
    fused_mask: tuple = None,
    pack_consts: tuple = None,
    col_mode: bool = False,
) -> bool:
    """``acc[i][q, p] = sum_h relu(K_p . q_{q,h}) * w[q, h]`` for every tile in the panel, one block.

    ``fused_mask``: ``(kposS_sb, qposS_sb)``; where a path can fold the causal
    mask into its PSUM eviction it does, and returns True so the caller skips the separate mask.

    ``col_mode`` (``cfg.col_headsum``): ``acc_sb`` and ``fused_mask`` are in the column-quadrant layout
    (see ``GlmIndexerConfig.col_headsum``).

    Loop order is HEAD-outer, TILE-inner, and that ordering is the point.

    Every head's weighted accumulate targets the same accumulator, so with tile-outer the
    ``n_heads`` accumulates for a tile form ONE dependency chain on Vector -- a true data dependency
    that no amount of double buffering can overlap, which is why successive overlap tricks kept
    returning almost nothing. Iterating tiles inside the head loop makes consecutive accumulates write DIFFERENT
    tiles' accumulators, so they are independent and pipeline. The alternative -- splitting each
    tile's accumulator into partials -- buys the same independence but costs
    ``POS_BLOCK * 4`` bytes/partition per extra partial, which at ACC_SPLIT=4 would cut the Q panel
    from 7 tiles to 2 and so triple K traffic. This costs nothing.

    Heads still cannot accumulate in PSUM: ReLU sits between the dot product and the weighted sum.
    ``interleave_degree=2`` rotates the PSUM tile and the ReLU buffer so head h+1's matmul (PE)
    overlaps head h's ReLU (Scalar); PSUM caps that at 2 in flight, since a ``POS_BLOCK``-wide bf16
    tile is 4 of the 8 banks.
    """
    # HEAD-PAIR PACKING (small-query tiles): a tile with rows <= 64 leaves half of every
    # engine's partitions idle for all n_heads iterations. Packing heads (2h, 2h+1) into one
    # [2 * rows, blk] tile (head 2h's queries on partitions [0, rows), head 2h+1's on
    # [rows, 2*rows)) halves the matmul, relu and accumulate INSTRUCTION time — engine time
    # scales with columns, not occupied partitions. The two half-accumulators are folded with
    # one tensor_tensor per tile at the end. The stationary pair qt_sb[:, i, 2h:2h+2, :rows]
    # is contiguous, so the matmul needs no relayout.
    G = P_MAX // max(tile_rows)
    if cfg.pack_heads and G >= 2 and cfg.n_heads % G == 0 and all(r == tile_rows[0] for r in tile_rows):
        rows = tile_rows[0]
        ng = cfg.n_heads // G
        hd = cfg.head_dim
        kernel_assert(rows & (rows - 1) == 0, f"pack_heads needs power-of-two rows, got {rows}")
        # Partitions of a logits / ReLU tile.
        GR = G * rows
        sbm.open_scope(name="score_pack_consts")
        if pack_consts is None:
            pack_consts = build_pack_heads_consts(sbm, qt_sb, w_sb, cfg, n_in_group, tile_rows)
        q8s, wsels = pack_consts
        # Software-pipelined by one head group, with a 3-deep ReLU ring.
        # PE runs in order, so issuing group g's head-sum right after its logits would park PE on the
        # ReLU (Scalar/Vector) every group; and a 2-deep ReLU buffer would make ReLU g+2 wait (WAR) on
        # head-sum g. So PE issues logits g+1 before head-sum g, and a ReLU never overwrites a tile
        # an in-flight head-sum still reads.
        sbm.open_scope(name="score_pack")
        # Logits run _PACK_LOOKAHEAD head groups ahead of their head-sum, carried across column tiles and
        # query tiles, so a ReLU that finishes late (Scalar / Vector) does not leave PE idle.
        look = _PACK_LOOKAHEAD
        # col_mode: two burst units' ReLUs live at once (unit k's while unit k-1's head-sums read theirs).
        if col_mode:
            n_pos, cw = col_quad_geometry(rows)
            # Head groups per burst unit: a whole column step when it holds at most _COL_UNIT_TILES ReLU tiles
            # (<= 16 rows); a larger step is split so the ReLU ring stays at 2 * _COL_UNIT_TILES tiles at 64
            # rows and 2 * _COL_UNIT_TILES_32 at 32 rows (a whole 32-row step needs 64 tiles; with the
            # 1M pipelined score's stage-1 scratch even 32 leave too little SBUF at 4k active tokens).
            unit = _COL_UNIT_TILES_32 if n_pos == COL_QUADS and rows > COL_QUAD // 2 else _COL_UNIT_TILES
            gc = ng if ng * n_pos <= unit else max(1, min(ng, unit // n_pos))
            kernel_assert(ng % gc == 0, f"col_headsum: {ng} head groups do not split into units of {gc}")
            n_rl = 2 * gc * n_pos
        else:
            n_rl = look + 2
        # col_mode: two column tiles of one head group share a PSUM bank, their stationary and one ReLU.
        n_grp = 2 if col_mode else 1
        rl_w = n_grp * MATMUL_TILE
        n_rl //= n_grp
        # 64 B aligned: a matmul moving operand off that alignment measured slower per matmul.
        rl_ring = [
            sbm.alloc_stack((P_MAX, rl_w), nl.bfloat16, name=f"pk_rl{_uid()}", align=64) for _ in range(n_rl)
        ]

        def _relu(dst, src):
            # On Vector: its PSUM-read ReLU is faster than Scalar's, and a Scalar ReLU holds its logits PSUM bank long
            # enough to stall a later logits matmul.
            nisa.tensor_scalar(dst=dst, data=src, op0=nl.maximum, operand0=0.0, engine=nisa.engine.vector)

        # One running ring index: a ReLU never overwrites a tile an in-flight head-sum still reads.
        rl_count = [0]
        # The PSUM logits tile of the most recent ReLU (col_mode's ordering gate reads it).
        last_lg = [None]

        def _mm_logits(dst_view, i, g, col, cols):
            """Logits of keys [col, col + cols) of head group g into ``dst_view``."""
            nisa.nc_matmul(
                dst=dst_view,
                stationary=q8s[i][:hd, g, :GR],
                moving=kt_sb[:hd, col : col + cols],
                accumulate=False,
            )

        def _logits_relu(i, g, col, cols):
            lg_ps = nl.ndarray((P_MAX, MATMUL_TILE), dtype=nl.bfloat16, buffer=nl.psum)
            _mm_logits(lg_ps[:GR, :cols], i, g, col, cols)
            rl = rl_ring[rl_count[0] % n_rl]
            rl_count[0] += 1
            last_lg[0] = lg_ps
            _relu(rl[:GR, :cols], lg_ps[:GR, :cols])
            return rl[:GR]

        def _logits_relu_pair(i, g, cols_):
            """Full column tiles of head group ``g`` side by side in PSUM (two per bank): matmuls on one stationary, one
            ReLU over all of them."""
            w_ = len(cols_) * MATMUL_TILE
            lg_ps = nl.ndarray((P_MAX, w_), dtype=nl.bfloat16, buffer=nl.psum)
            for j, col in enumerate(cols_):
                _mm_logits(lg_ps[:GR, j * MATMUL_TILE : (j + 1) * MATMUL_TILE], i, g, col, MATMUL_TILE)
            rl = rl_ring[rl_count[0] % n_rl]
            rl_count[0] += 1
            last_lg[0] = lg_ps
            _relu(rl[:GR, :w_], lg_ps[:GR, :w_])
            return [rl[:GR, j * MATMUL_TILE : (j + 1) * MATMUL_TILE] for j in range(len(cols_))]

        def _evict(acc_ps, i, col, cols, ev_rows=rows):
            if fused_mask is not None:
                # Evict + causal mask in one op.
                apply_causal_mask_min(
                    acc_sb[:ev_rows, i, col : col + cols],
                    acc_ps[:ev_rows, :cols],
                    fused_mask[0],
                    fused_mask[1],
                    i,
                    ev_rows,
                    cols,
                    col,
                )
            else:
                nisa.tensor_copy(
                    dst=acc_sb[:rows, i, col : col + cols],
                    src=acc_ps[:rows, :cols],
                    engine=nisa.engine.vector if (col // MATMUL_TILE) % 2 else nisa.engine.scalar,
                )

        if col_mode:
            """
            Column-tiled head-sums (cfg.col_headsum). The head-sum's stationary is only [128, rows] (rows = 16
            queries), but a matmul costs PE one slot per moving column whatever its stationary width, so in
            the plain path every head-sum cost as much as a logits matmul and PE spent much of its time on them.
            In PE column-tiling mode (tile_size=(128, 32)) four such matmuls run concurrently, one per 32-column
            quadrant, each with its own moving operand: quadrant qd takes position range
            [qd * blk / 4, (qd + 1) * blk / 4), so column tile u of quadrant qd is block column
            (qd * n_u + u) * MATMUL_TILE and its head-sum lands on PSUM partitions [qd * 32, qd * 32 + rows)
            of one shared accumulator per u. That accumulator is masked and evicted once, on 112 partitions,
            per 4 column tiles (not once per tile on 16 partitions).

            Quadrant overlap only happens when the 4 head-sums issue back to back, and every switch between
            full-array and column-tiled matmuls costs PE time. So each column step u (2048 positions)
            issues its 16 logits back to back, then step u-1's 16 head-sums as 4 rounds (head groups) of 4
            quadrants: one switch each way per step. Left alone, the scheduler hoists ready head-sums back
            in between the logits; the burst therefore reads a copy of the selection weights that
            data-depends on step u's last logits PSUM tile (0 * logits + wsel, exact). Reading the PSUM
            logits rather than the SBUF ReLU tile matters: an SBUF-only Vector op stalls whenever it
            overlaps a GpSimd top-k.

            33-64 rows (8k active at CP128): the block splits into 2 position ranges on partitions [0, 64) and
            [64, 128), and each head-sum is two 32-row quadrant matmuls reading the same ReLU tile, so a round
            of 4 quadrants still runs back to back. A column step (1024 positions) then has 32 logits, twice
            the <= 32-row step, so it is issued as 2 burst units of 8 head groups: the ReLU ring stays at 32
            tiles, and only each unit's first head group is gated (the rest accumulate onto it in PSUM).
            """
            kernel_assert(
                fused_mask is not None and rows <= COL_MAX_ROWS and blk % (n_pos * MATMUL_TILE) == 0,
                f"col_headsum needs rows <= {COL_MAX_ROWS} and blk a multiple of {n_pos * MATMUL_TILE} "
                f"(rows={rows}, blk={blk})",
            )
            kernel_assert(n_in_group == 1, "col_headsum supports one query tile per panel")
            n_u = blk // (n_pos * MATMUL_TILE)
            q_rows = (n_pos - 1) * cw + rows
            # Head-sum column tiles: (PSUM partition, wsel row range) of every 32-column quadrant. At <= 32
            # rows each position range is one quadrant; at 64 rows the two 32-row halves of one position
            # range's head-sum run as two quadrants, both reading the same ReLU tile.
            col_tiles = [
                (pq, pq * cw + r0, r0, min(rows, r0 + COL_QUAD))
                for pq in range(n_pos)
                for r0 in range(0, rows, COL_QUAD)
            ]
            # Gated head groups per unit: all of them at <= 32 rows (unchanged); at 64 rows only the first --
            # the rest accumulate onto it in PSUM, so they cannot issue before it anyway.
            gn = gc if n_pos == COL_QUADS else 1
            wb_cols = rows
            wsel_b = [
                sbm.alloc_stack((P_MAX, gn, wb_cols), nl.bfloat16, name=f"pk_wb{_uid()}", align=64) for _ in range(2)
            ]
            units = [(u, g0) for u in range(n_u) for g0 in range(0, ng, gc)]

            def _logits_unit(k):
                u, g0 = units[k]
                return [
                    [
                        v
                        for pq in range(0, n_pos, n_grp)
                        for v in _logits_relu_pair(0, g, [((pq + j) * n_u + u) * MATMUL_TILE for j in range(n_grp)])
                    ]
                    for g in range(g0, g0 + gc)
                ]

            def _gate(k, src):
                """``wb = 0 * src + wsel``: unit ``k``'s gated selection weights, ordered after ``src`` (PSUM)."""
                g0 = units[k][1]
                wb = wsel_b[k % 2]
                nisa.scalar_tensor_tensor(
                    dst=wb[:GR],
                    data=src.ap(pattern=[[src.shape[1], GR], [wb_cols, gn], [1, wb_cols]], offset=0),
                    op0=nl.multiply,
                    operand0=0.0,
                    op1=nl.add,
                    operand1=wsels[0][:GR, g0 : g0 + gn, :],
                )
                return wb

            acc_cur = [None]

            def _burst(k, unit_rls, wb):
                """Unit ``k``'s head-sums; returns the step's eviction once its last unit is issued."""
                u, g0 = units[k]
                if g0 == 0:
                    acc_cur[0] = nl.ndarray((P_MAX, MATMUL_TILE), dtype=nl.float32, buffer=nl.psum)
                acc = acc_cur[0]
                for g in range(g0, g0 + gc):
                    for pq, p0, r0, r1 in col_tiles:
                        st = wb[:GR, g - g0, r0:r1] if g - g0 < gn else wsels[0][:GR, g, r0:r1]
                        nisa.nc_matmul(
                            dst=acc[p0 : p0 + r1 - r0, :],
                            stationary=st,
                            moving=unit_rls[g - g0][pq],
                            accumulate=(g > 0),
                            tile_position=(0, p0),
                            tile_size=(P_MAX, COL_QUAD),
                        )
                if g0 + gc == ng:
                    return (acc, 0, u * MATMUL_TILE, MATMUL_TILE, q_rows)
                return None

            def _burst_evict(k, unit_rls, wb, pend_ev):
                ev = _burst(k, unit_rls, wb)
                if ev is None:
                    return pend_ev
                if pend_ev is not None:
                    _evict(*pend_ev)
                return ev

            # Unit k-1's burst is gated on unit k's last logits. With every group gated (<= 32 rows) the gate
            # is unit-agnostic and is issued every unit, as before; otherwise it carries unit k-1's weights.
            full = gn == ng
            prev = None
            pend_ev = None
            wb = None
            for k in range(len(units)):
                unit = _logits_unit(k)
                if full:
                    wb = _gate(k, last_lg[0])
                if prev is not None:
                    if not full:
                        wb = _gate(k - 1, last_lg[0])
                    pend_ev = _burst_evict(k - 1, prev, wb, pend_ev)
                prev = unit
            # The last unit has no later logits to gate on: gate it on its own last logits.
            if not full:
                wb = _gate(len(units) - 1, last_lg[0])
            pend_ev = _burst_evict(len(units) - 1, prev, wb, pend_ev)
            _evict(*pend_ev)
            sbm.close_scope()  # score_pack
            sbm.close_scope()  # score_pack_consts
            return True

        steps = [
            (col, min(MATMUL_TILE, blk - col), i, g)
            for col in range(0, blk, MATMUL_TILE)
            for i in range(n_in_group)
            for g in range(ng)
        ]
        rls = {}
        for t in range(min(look, len(steps))):
            c_, w_, i_, g_ = steps[t]
            rls[t] = _logits_relu(i_, g_, c_, w_)
        # A tile's eviction waits until the next tile's first head-sum: Vector runs in order, and an
        # eviction issued right after its head-sums would park it on PE ahead of the ReLUs PE waits for.
        pending = None
        acc_ps = None
        for t, (col, cols, i, g) in enumerate(steps):
            if t + look < len(steps):
                c_, w_, i_, g_ = steps[t + look]
                rls[t + look] = _logits_relu(i_, g_, c_, w_)
            if g == 0:
                if pending is not None:
                    _evict(*pending)
                    pending = None
                acc_ps = nl.ndarray((P_MAX, MATMUL_TILE), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(
                dst=acc_ps[:rows, :cols],
                stationary=wsels[i][:GR, g, :rows],
                moving=rls.pop(t)[:, :cols],
                accumulate=(g > 0),
            )
            if g == ng - 1:
                pending = (acc_ps, i, col, cols)
        _evict(*pending)
        sbm.close_scope()  # score_pack
        sbm.close_scope()  # score_pack_consts
        return fused_mask is not None

    pack_pairs = cfg.pack_head_pairs and cfg.n_heads % 2 == 0 and all(r * 2 <= P_MAX for r in tile_rows)
    if pack_pairs:
        # Paired per-partition weights: w2[r + g*rows, i, j] = w[r, i, 2j + g]. Loop-invariant,
        # so it lives in its OWN scope — an interleaved scope's section rotation overlays
        # buffers hoisted inside it.
        sbm.open_scope(name="score_wpair")
        w2_sb = sbm.alloc_stack((P_MAX, n_in_group, cfg.n_heads // 2), w_sb.dtype, name=f"wpair{_uid()}")
        # Q relayout: qt_sb's per-head row axis is padded to P_MAX, so heads (2h, 2h+1) are not
        # one contiguous stationary run. Pack them as q2[:, i, h2, g*rows + r] = qt[:, i, 2h2+g, r]
        # (one copy per tile; the matmul stationary then has a single free dim).
        rmax = max(tile_rows)
        q2_sb = sbm.alloc_stack(
            (P_MAX, n_in_group, cfg.n_heads // 2, 2 * rmax), nl.bfloat16, name=f"qpair{_uid()}", align=32
        )
        for i in range(n_in_group):
            rows = tile_rows[i]
            for g in range(2):
                nisa.tensor_copy(
                    dst=q2_sb[: cfg.head_dim, i, :, g * rows : (g + 1) * rows],
                    src=qt_sb.ap(
                        pattern=[
                            [qt_sb.shape[1] * cfg.n_heads * P_MAX, cfg.head_dim],
                            [2 * P_MAX, cfg.n_heads // 2],
                            [1, rows],
                        ],
                        offset=(i * cfg.n_heads + g) * P_MAX,
                    ),
                )
        for i in range(n_in_group):
            rows = tile_rows[i]
            nisa.tensor_copy(
                dst=w2_sb[0:rows, i, :],
                src=w_sb.ap(
                    pattern=[[w_sb.shape[1] * cfg.n_heads, rows], [2, cfg.n_heads // 2]], offset=i * cfg.n_heads
                ),
            )
            nisa.tensor_copy(
                dst=w2_sb[rows : 2 * rows, i, :],
                src=w_sb.ap(
                    pattern=[[w_sb.shape[1] * cfg.n_heads, rows], [2, cfg.n_heads // 2]], offset=i * cfg.n_heads + 1
                ),
            )
        sbm.open_scope(interleave_degree=2, name="score_heads")
        for h2 in range(cfg.n_heads // 2):
            for i in range(n_in_group):
                rows = tile_rows[i]
                relu_sb = sbm.alloc_stack((P_MAX, POS_BLOCK), nl.bfloat16, name=f"relu{_uid()}")
                logits_psum = nl.ndarray((P_MAX, POS_BLOCK), dtype=nl.bfloat16, buffer=nl.psum)
                for col in range(0, blk, MATMUL_TILE):
                    cols = min(MATMUL_TILE, blk - col)
                    nisa.nc_matmul(
                        dst=logits_psum[: 2 * rows, col : col + cols],
                        stationary=q2_sb[: cfg.head_dim, i, h2, : 2 * rows],
                        moving=kt_sb[: cfg.head_dim, col : col + cols],
                        accumulate=False,
                    )
                nisa.activation(dst=relu_sb[: 2 * rows, :blk], op=nl.relu, data=logits_psum[: 2 * rows, :blk])
                if h2 == 0:
                    nisa.tensor_scalar(
                        dst=acc_sb[: 2 * rows, i, :blk],
                        data=relu_sb[: 2 * rows, :blk],
                        op0=nl.multiply,
                        operand0=w2_sb[: 2 * rows, i, 0:1],
                    )
                else:
                    nisa.scalar_tensor_tensor(
                        dst=acc_sb[: 2 * rows, i, :blk],
                        data=relu_sb[: 2 * rows, :blk],
                        op0=nl.multiply,
                        operand0=w2_sb[: 2 * rows, i, h2 : h2 + 1],
                        op1=nl.add,
                        operand1=acc_sb[: 2 * rows, i, :blk],
                    )
                sbm.increment_section()
        sbm.close_scope()  # score_heads (interleaved)
        # Fold the upper half-accumulator (odd heads) onto the real query rows. tensor_tensor
        # requires lhs/rhs at the SAME partition start, so the upper half is first copied to a
        # base-0 scratch (copies across partition bases are allowed).
        fold_sb = sbm.alloc_stack((P_MAX, POS_BLOCK), cfg.score_acc_dtype, name=f"accfold{_uid()}")
        for i in range(n_in_group):
            rows = tile_rows[i]
            nisa.tensor_copy(dst=fold_sb[:rows, :blk], src=acc_sb[rows : 2 * rows, i, :blk])
            nisa.tensor_tensor(
                dst=acc_sb[:rows, i, :blk],
                data1=acc_sb[:rows, i, :blk],
                data2=fold_sb[:rows, :blk],
                op=nl.add,
            )
        sbm.close_scope()  # score_wpair
        return

    sbm.open_scope(interleave_degree=2, name="score_heads")
    for h in range(cfg.n_heads):
        for i in range(n_in_group):
            rows = tile_rows[i]
            relu_sb = sbm.alloc_stack((P_MAX, POS_BLOCK), nl.bfloat16, name=f"relu{_uid()}")
            logits_psum = nl.ndarray((P_MAX, POS_BLOCK), dtype=nl.bfloat16, buffer=nl.psum)

            for col in range(0, blk, MATMUL_TILE):
                cols = min(MATMUL_TILE, blk - col)
                nisa.nc_matmul(
                    dst=logits_psum[:rows, col : col + cols],
                    stationary=qt_sb[: cfg.head_dim, i, h, :rows],
                    moving=kt_sb[: cfg.head_dim, col : col + cols],
                    accumulate=False,
                )
            nisa.activation(dst=relu_sb[:rows, :blk], op=nl.relu, data=logits_psum[:rows, :blk])

            if h == 0:
                # First head initializes this tile's accumulator, so no zeroing pass is needed.
                nisa.tensor_scalar(
                    dst=acc_sb[:rows, i, :blk],
                    data=relu_sb[:rows, :blk],
                    op0=nl.multiply,
                    operand0=w_sb[:rows, i, 0:1],
                )
            else:
                nisa.scalar_tensor_tensor(
                    dst=acc_sb[:rows, i, :blk],
                    data=relu_sb[:rows, :blk],
                    op0=nl.multiply,
                    operand0=w_sb[:rows, i, h : h + 1],
                    op1=nl.add,
                    operand1=acc_sb[:rows, i, :blk],
                )
            sbm.increment_section()
    sbm.close_scope()


def score_over_context(
    sbm: SbufManager,
    q_hbm: nl.NkiTensor,
    k_hbm: nl.NkiTensor,
    weights_hbm: nl.NkiTensor,
    qpos_hbm: nl.NkiTensor,
    score_hbm: nl.NkiTensor,
    topk_hbm: nl.NkiTensor,
    cfg: GlmIndexerConfig,
    end_pos: int,
    tile_lo: int,
    tile_hi: int,
    fuse_topk: bool = True,
    score_out: nl.NkiTensor = None,
    k_cache_hbm: nl.NkiTensor = None,
    block_table_hbm: nl.NkiTensor = None,
    block_size: int = 0,
    kv_cache_fp8_dtype=None,
    q_resident: tuple = None,
    kt_pre: nl.NkiTensor = None,
) -> None:
    """Score this core's query tiles against ``[0, end_pos)``, K-outer over a resident Q panel.

    The top-k is issued here per group, as soon as that group's score rows are complete, so its
    GpSimd work overlaps the next group's score on PE/Scalar/Vector instead of adding to it.

    Loop order is (query-tile GROUP, position block, tile within group): each K block is read once
    and consumed by every tile in the panel, so K traffic falls by the panel size.

    K is S-agnostic -- the loop sees only a position range, so a position's origin (this rank, a peer,
    the paged cache) is invisible. Three independent raggednesses all have to be right: a partial
    group, a partial tile, and a partial position block.
    """
    panel = cfg.q_panel_tiles

    T_row = score_hbm.shape[-1]
    ctx_row = score_row_width(end_pos, cfg.topk_nseg)

    # Layout the mask converts from: rank-major, `interleave` tokens per rank per round.
    s_kv_local = max(1, end_pos // cfg.cp_degree)
    # At cp 1 the interleave cancels out of owned(), so any power of two will do.
    interleave = cfg.cp_interleave or (s_kv_local if cfg.cp_degree > 1 else 1)
    kernel_assert(
        end_pos % cfg.cp_degree == 0,
        f"end_pos ({end_pos}) must split evenly over the {cfg.cp_degree} CP ranks: each rank gathers the "
        f"same number of slots, and the mask maps slot -> position assuming exactly that",
    )
    kernel_assert(
        s_kv_local % interleave == 0
        and interleave & (interleave - 1) == 0
        and cfg.cp_degree & (cfg.cp_degree - 1) == 0,
        f"the mask's slot->global map needs power-of-two interleave ({interleave}) and cp_degree "
        f"({cfg.cp_degree}), and s_kv_local ({s_kv_local}) a multiple of the interleave",
    )
    mask_cp = cfg.cp_degree

    kpos_hbm = nl.ndarray((ctx_row, 1), dtype=nl.float32, buffer=nl.private_hbm, name=scoped_name("kpos"))
    build_key_positions(sbm, kpos_hbm, s_kv_local, mask_cp, interleave)

    # cp_degree == 1: the caller supplies a whole-context cache, so the table is resolved here over
    # the global extent. With a gather each rank linearizes its own shard instead (see the entry).
    if block_table_hbm is not None:
        k_lin = nl.ndarray(
            (ctx_row, cfg.kv_cache_head_width), dtype=nl.bfloat16, buffer=nl.private_hbm, name=scoped_name("k_lin")
        )
        linearize_pages(sbm, k_cache_hbm, block_table_hbm, k_lin, cfg, end_pos, block_size)
        k_hbm = k_lin

    kt_hbm = kt_pre
    kscale_hbm = None
    if kt_pre is not None:
        pass
    elif cfg.kv_fp8:
        # fp8: K stays fp8 into the score; its per-key scale is applied after the head sum.
        kt_hbm = nl.ndarray(
            (cfg.head_dim, ctx_row), dtype=kv_cache_fp8_dtype, buffer=nl.private_hbm, name=scoped_name("kt8_hbm")
        )
        kscale_hbm = nl.ndarray((ctx_row, 1), dtype=nl.bfloat16, buffer=nl.private_hbm, name=scoped_name("kscale"))
        build_kt_fp8(sbm, k_hbm, kt_hbm, kscale_hbm, cfg, end_pos, kv_cache_fp8_dtype)
    elif not cfg.score_chunk and should_roll_score(cfg, end_pos) and end_pos // POS_BLOCK > 1:
        # The chunked score always reads the pre-transposed K, even for a single position block
        # (a forced score_chunk at prefill sizes, e.g. end_pos == POS_BLOCK).
        kt_hbm = nl.ndarray(
            (cfg.head_dim, ctx_row), dtype=nl.bfloat16, buffer=nl.private_hbm, name=scoped_name("kt_hbm")
        )
        sbm.open_scope(name="k_pretranspose")
        kt_stage = sbm.alloc_stack((P_MAX, POS_BLOCK), nl.bfloat16, name=f"kt_stage{_uid()}", align=32)
        for c in range(0, end_pos, POS_BLOCK):
            cw = min(POS_BLOCK, end_pos - c)
            nisa.dma_transpose(
                dst=kt_stage[: cfg.head_dim, :cw],
                src=k_hbm.ap(pattern=[[cfg.head_dim, cw], [1, cfg.head_dim]], offset=c * cfg.head_dim),
                axes=(1, 0),
            )
            nisa.dma_copy(dst=kt_hbm[0 : cfg.head_dim, c : c + cw], src=kt_stage[: cfg.head_dim, :cw])
        sbm.close_scope()

    # Chunked (long context): every loop is a device loop, so the trace does not grow with context.
    if cfg.score_chunk:
        score_chunked_rolled(
            sbm,
            q_hbm,
            kt_hbm,
            kpos_hbm,
            weights_hbm,
            qpos_hbm,
            score_hbm,
            topk_hbm,
            cfg,
            end_pos,
            tile_lo,
            tile_hi,
            score_out=score_out,
            kscale_hbm=kscale_hbm,
            k_rows_hbm=k_hbm,
            q_resident=q_resident,
            kpos_layout=(s_kv_local, mask_cp, interleave),
        )
        return

    sbm.open_scope(name="score_invariants")
    qt_sb = sbm.alloc_stack((P_MAX, panel, cfg.n_heads, P_MAX), nl.bfloat16, name=f"q_panel{_uid()}", align=32)
    kt_sb = sbm.alloc_stack(
        (P_MAX, POS_BLOCK), kt_hbm.dtype if kt_hbm is not None else nl.bfloat16, name=f"k_block{_uid()}", align=32
    )
    w_sb = sbm.alloc_stack((P_MAX, panel, cfg.n_heads), nl.float32, name=f"weights{_uid()}")
    qpos_sb = sbm.alloc_stack((P_MAX, panel), nl.float32, name=f"qpos{_uid()}")
    # k_pos varies only with the position, so one precomputed row is replicated across partitions.
    kpos_sb = sbm.alloc_stack((P_MAX, POS_BLOCK), nl.float32, name=f"kpos{_uid()}")
    ks_sb = sbm.alloc_stack((P_MAX, POS_BLOCK), nl.bfloat16, name=f"kscale{_uid()}") if kscale_hbm is not None else None
    acc_sb = sbm.alloc_stack((P_MAX, panel, POS_BLOCK), cfg.score_acc_dtype, name=f"score_acc{_uid()}")

    sbm.open_scope(interleave_degree=1, name="score_groups")
    for group_lo in range(tile_lo, tile_hi, panel):
        group_hi = min(group_lo + panel, tile_hi)
        n_in_group = group_hi - group_lo

        # Panel load: Q transposed, plus this tile's per-head weights.
        for i in range(n_in_group):
            tile = group_lo + i
            rows = min(P_MAX, cfg.s_local - tile * P_MAX)
            load_q_panel_transposed(sbm, q_hbm, qt_sb, cfg, tile, rows, i)
            nisa.dma_copy(
                dst=w_sb[:rows, i, : cfg.n_heads],
                src=weights_hbm[tile * P_MAX : tile * P_MAX + rows, : cfg.n_heads],
            )
            nisa.dma_copy(
                dst=qpos_sb[:rows, i : i + 1],
                src=qpos_hbm.reshape((cfg.s_local, 1))[tile * P_MAX : tile * P_MAX + rows, 0:1],
            )

        tile_rows = [min(P_MAX, cfg.s_local - (group_lo + i) * P_MAX) for i in range(n_in_group)]

        def _pos_block(pos, blk):
            """Score ONE position block. ``pos`` may be a trace-time int or a fori_loop index."""
            if kt_hbm is not None:
                _load_kt_block(kt_sb, kt_hbm, pos, blk, cfg.head_dim)
            else:
                nisa.dma_transpose(
                    dst=kt_sb[: cfg.head_dim, :blk],
                    src=k_hbm.ap(pattern=[[cfg.head_dim, blk], [1, cfg.head_dim]], offset=pos * cfg.head_dim),
                    axes=(1, 0),
                )

            nisa.dma_copy(
                dst=kpos_sb[:P_MAX, :blk],
                src=kpos_hbm.reshape((1, ctx_row))[0:1, nl.ds(pos, blk)].broadcast(dim=0, size=P_MAX),
            )
            if ks_sb is not None:
                nisa.dma_copy(
                    dst=ks_sb[:P_MAX, :blk],
                    src=kscale_hbm.reshape((1, ctx_row))[0:1, nl.ds(pos, blk)].broadcast(dim=0, size=P_MAX),
                )
            score_block_for_group(sbm, qt_sb, kt_sb, w_sb, acc_sb, cfg, n_in_group, tile_rows, blk)
            for i in range(n_in_group):
                apply_causal_mask(sbm, acc_sb, kpos_sb, qpos_sb, i, tile_rows[i], blk, kscale_sb=ks_sb)
            for i in range(n_in_group):
                tile = group_lo + i
                nisa.dma_copy(
                    dst=score_hbm[tile - tile_lo, 0 : tile_rows[i], nl.ds(pos, blk)],
                    src=acc_sb[: tile_rows[i], i, :blk],
                )
                if score_out is not None:
                    # Debug copy to the full-size returned score; see emit_score.
                    nisa.dma_copy(
                        dst=score_out[tile * P_MAX : tile * P_MAX + tile_rows[i], nl.ds(pos, blk)],
                        src=acc_sb[: tile_rows[i], i, :blk],
                    )

        # Unrolled, this loop dominates the trace and pos_blocks is its largest multiplier, so
        # rolling it is what stops the instruction count growing with context.
        n_full = end_pos // POS_BLOCK
        tail = end_pos - n_full * POS_BLOCK
        if should_roll_score(cfg, end_pos) and n_full > 1:
            nl.fori_loop(0, n_full, lambda b: _pos_block(b * POS_BLOCK, POS_BLOCK))
        else:
            for b in range(n_full):
                _pos_block(b * POS_BLOCK, POS_BLOCK)
        if tail:
            _pos_block(n_full * POS_BLOCK, tail)

        # Pad tail to -inf so the snake-padded columns never win the top-k. One memset per tile.
        pad = T_row - end_pos
        if pad > 0:
            pad_sb = sbm.alloc_stack((P_MAX, pad), nl.float32, name=f"score_pad{_uid()}")
            nisa.memset(dst=pad_sb[:P_MAX, :pad], value=float("-inf"))
            for i in range(n_in_group):
                tile = group_lo + i
                nisa.dma_copy(
                    dst=score_hbm[tile - tile_lo, 0 : tile_rows[i], end_pos : end_pos + pad],
                    src=pad_sb[: tile_rows[i], :pad],
                )
                if score_out is not None:
                    nisa.dma_copy(
                        dst=score_out[tile * P_MAX : tile * P_MAX + tile_rows[i], end_pos : end_pos + pad],
                        src=pad_sb[: tile_rows[i], :pad],
                    )

        # This group's score rows are complete, so its top-k can run now -- on GpSimd, overlapping
        # the next group's score on PE/Scalar/Vector. Skipped when the top-k does not fit alongside
        # the resident panel; the caller then runs it as its own phase (see topk_fits_with_panel).
        if fuse_topk:
            for i in range(n_in_group):
                topk_for_tile(
                    sbm, score_hbm, topk_hbm, cfg, end_pos, group_lo + i, group_lo + i - tile_lo, T_row, cfg.index_topk
                )
        sbm.increment_section()
    sbm.close_scope()  # score_groups
    sbm.close_scope()  # score_invariants


_KT_PRE_SLOC = [0]


def _load_kt_block(dst_sb, kt_hbm, pos, blk: int, head_dim: int) -> None:
    """``dst_sb[d, j] = K[pos + j, d]`` from K-major K (``[head_dim, ctx]`` or rank-stacked ``[cp * head_dim, s_loc]``)."""
    s_loc = _KT_PRE_SLOC[0]
    if not s_loc:
        nisa.dma_copy(dst=dst_sb[:head_dim, :blk], src=kt_hbm[0:head_dim, nl.ds(pos, blk)])
        return
    if not isinstance(pos, int):
        # A device position: runtime // and % give the rank and shard offset (blocks never straddle).
        kernel_assert(s_loc % blk == 0, f"pre-transposed K: a {blk}-wide block can straddle a {s_loc}-row rank shard")
        nisa.dma_copy(
            dst=dst_sb[:head_dim, :blk],
            src=kt_hbm.ap(
                pattern=[[s_loc, head_dim], [1, blk]],
                offset=0,
                scalar_offset=(pos // s_loc) * (head_dim * s_loc) + pos % s_loc,
                indirect_dim=1,
            ),
        )
        return
    r0, p0 = pos // s_loc, pos % s_loc
    if p0 + blk <= s_loc:
        nisa.dma_copy(dst=dst_sb[:head_dim, :blk], src=kt_hbm.ap(pattern=[[s_loc, head_dim], [1, blk]], offset=r0 * head_dim * s_loc + p0))
        return
    kernel_assert(p0 == 0 and blk % s_loc == 0, f"pre-transposed K block {pos}+{blk} straddles a {s_loc}-row rank shard")
    n_r = blk // s_loc
    nisa.dma_copy(
        dst=dst_sb.ap(pattern=[[dst_sb.shape[1], head_dim], [s_loc, n_r], [1, s_loc]], offset=0),
        src=kt_hbm.ap(pattern=[[s_loc, head_dim], [head_dim * s_loc, n_r], [1, s_loc]], offset=r0 * head_dim * s_loc),
    )


def load_k_block_transposed(kt_sb, k_rows_hbm, pos, blk: int, head_dim: int) -> None:
    """``kt_sb[d, j] = k_rows[pos + j, d]`` for ``j < blk``, by one HWDGE DMA transpose.

    ``pos`` may be a device-loop value, which only the hardware DGE path
    can lower, and HWDGE needs a 3-D source with a leading dim of 16. So the block is read as
    ``[16, blk / 16, head_dim]`` with row ``pos + 16 * j + i`` at ``[i, j, :]``: the [2, 1, 0]
    transpose writes ``[head_dim, blk / 16, 16]``, whose flattened free index ``16 * j + i`` is the
    position order.
    """
    kernel_assert(blk % 16 == 0 and head_dim <= P_MAX, f"HWDGE K transpose needs blk % 16 == 0, got {blk}")
    n_j = blk // 16
    pattern = [[head_dim, 16], [16 * head_dim, n_j], [1, head_dim]]
    if isinstance(pos, int):
        src = k_rows_hbm.ap(pattern=pattern, offset=pos * head_dim)
    else:
        # A device value: a runtime row offset. scalar_offset counts rows of the tensor (one K row here), whatever the
        # pattern's dim-0 step.
        src = k_rows_hbm.ap(pattern=pattern, offset=0, scalar_offset=pos, indirect_dim=0)
    nisa.dma_transpose(
        dst=kt_sb.ap(pattern=[[kt_sb.shape[1], head_dim], [16, n_j], [1, 16]], offset=0),
        src=src,
        axes=(2, 1, 0),
        dge_mode=dge_mode.hwdge,
    )


# Most chunks for the written-out pipelined score loop (1M has 18); a device loop waits on collectives.
_PIPELINE_MAX_CHUNKS = 18
# Pipelined stage-1: a batch's finish (Vector decode + candidate stores) is placed this many position
# blocks after its issue (ring load + topk), so Vector never queues behind a topk still running.
_S1_FINISH_LAG = 2
# pack_heads: head groups the logits matmuls run ahead of their head-sum (see score_block_for_group).
_PACK_LOOKAHEAD = 2
# col_headsum: PE column quadrants and their width.
COL_QUADS = 4
COL_QUAD = P_MAX // COL_QUADS
# col_headsum: the most query rows a panel may hold (two 64-partition position halves).
COL_MAX_ROWS = 2 * COL_QUAD
# col_headsum: ReLU tiles per burst unit (one 2048-position step at <= 16 rows), and at 17-32 rows.
_COL_UNIT_TILES = 16
_COL_UNIT_TILES_32 = 8


def col_wide_enabled(cfg, end_pos: int) -> bool:
    """Whether a single resident query tile takes col_headsum's 64-row (2 position range) layout."""
    rows = cfg.s_local
    g = P_MAX // rows if rows else 0
    return bool(
        cfg.col_headsum
        and cfg.pack_heads
        and COL_QUAD < rows <= COL_MAX_ROWS
        and rows & (rows - 1) == 0
        and g >= 2
        and cfg.n_heads % g == 0
        and min(POS_BLOCK, end_pos) % (2 * MATMUL_TILE) == 0
    )


def col_quad_geometry(rows: int) -> tuple:
    """``(n_pos, width)`` of the col_headsum layout for a panel of ``rows`` query rows.

    The block's positions split into ``n_pos`` equal ranges; range ``pq``'s scores sit on partitions
    ``[pq * width, pq * width + rows)``. ``width`` is one 32-column PE quadrant up to 32 rows (4 ranges),
    else two (2 ranges, each head-sum split into two 32-row column tiles).
    """
    width = COL_QUAD if rows <= COL_QUAD else COL_MAX_ROWS
    return P_MAX // width, width


# Pipelined: the last chunk is this many position blocks, so the stage-1 top-k left exposed after the
# score (nothing follows it to overlap) ranks a short chunk; the rest split the remaining blocks evenly.
_PIPE_LAST_BLOCKS = 4


def pipeline_chunk_geometry(end_pos: int, k: int) -> list:
    """``[(start, width, rank_width)]`` of the pipelined chunks: ``rank_width`` is what stage-1 ranks
    (``width`` rounded up to the snake alignment; the ring pads the difference with -inf)."""
    n_blk = div_ceil(end_pos, POS_BLOCK)
    last = min(_PIPE_LAST_BLOCKS, n_blk)
    # Make the other chunks equal: stage-1 loads a chunk as one run only when it fills the ring row.
    for cand in range(last, 0, -1):
        r = n_blk - cand
        if r == 0 or r % div_ceil(r, SEG_MAX // POS_BLOCK) == 0:
            last = cand
            break
    rest = n_blk - last
    sizes = []
    if rest:
        m = div_ceil(rest, SEG_MAX // POS_BLOCK)
        sizes = [rest // m + (1 if j < rest % m else 0) for j in range(m)]
    geo, c0 = [], 0
    for nb in sizes + [last]:
        cw = min(nb * POS_BLOCK, end_pos - c0)
        rw = max(div_ceil(cw, SCORE_ROW_ALIGN) * SCORE_ROW_ALIGN, k)
        geo.append((c0, cw, rw))
        c0 += cw
    return geo


def chunk_pipeline_enabled(cfg, end_pos: int) -> bool:
    """Whether the chunked score is software-pipelined across chunks (see ``_PIPELINE_MAX_CHUNKS``)."""
    return bool(
        cfg.flat_loops
        and cfg.score_chunk
        and div_ceil(end_pos, cfg.score_chunk) <= _PIPELINE_MAX_CHUNKS
    )


def _owned_position(g: int, s_kv_local: int, cp_degree: int, interleave: int, rank_step: int = 1) -> int:
    """Host mirror of :func:`build_key_positions`: the global position of gathered key ``g``.

    ``rank_step``: gathered slot ``m`` holds CP rank ``m * rank_step`` (a strided gather group; its rank
    offset is common to every key, so the mirror takes it as 0)."""
    rank, l = divmod(g, s_kv_local)
    rank *= rank_step
    return (l // interleave) * (interleave * cp_degree) + rank * interleave + (l % interleave)


def block_periodic_key_offsets(end_pos: int, layout: tuple) -> list:
    """Each position block's key-position offset, if every block's positions are block 0's plus that
    offset (else None)."""
    if layout is None:
        return None
    s_loc, cp, il = layout[:3]
    st = layout[3] if len(layout) > 3 else 1
    base = [_owned_position(j, s_loc, cp, il, st) for j in range(min(POS_BLOCK, end_pos))]
    offs = []
    for p0 in range(0, end_pos, POS_BLOCK):
        off = _owned_position(p0, s_loc, cp, il, st)
        n = min(POS_BLOCK, end_pos - p0)
        if any(_owned_position(p0 + j, s_loc, cp, il, st) - off != base[j] for j in range(n)):
            return None
        offs.append(off)
    return offs


def alloc_pipelined_s1_sets(sbm, ch: int, k: int, tile_rows: list) -> list:
    """The pipelined stage-1's SBUF sets (see score_chunked_rolled's ``_group``)."""
    # Two sets of the topk OUTPUTS (val/idx): batch b's topk writes one while batch b-1's decode
    # still reads the other. The source tile and the decode's pos/hi/lo are shared -- GpSimd and
    # Vector each run the batches in order, so no two of their uses overlap.
    # Sets: one per batch in flight (issued, not yet finished); two source tiles, so batch b+1's
    # ring load lands while batch b's topk still reads its source.
    base = alloc_chunk_stage1_scratch(sbm, ch, k)
    base["srcs"] = [
        base["src"],
        sbm.alloc_stack((P_MAX, ch // TOPK_SNAKE), nl.bfloat16, name=f"cs_src_alt{_uid()}"),
    ]
    n_sets = min(_S1_FINISH_LAG + 1, div_ceil(max(tile_rows), TOPK_QUERIES_PER_CALL) * len(tile_rows))
    sets = [base]
    for j in range(1, n_sets):
        alt = dict(base)
        alt["val"] = sbm.alloc_stack((P_MAX, k), nl.bfloat16, name=f"cs_val_alt{j}{_uid()}")
        alt["idx"] = sbm.alloc_stack((P_MAX, k), nl.uint32, name=f"cs_idx_alt{j}{_uid()}")
        sets.append(alt)
    return sets


def score_chunked_rolled(
    sbm: SbufManager,
    q_hbm: nl.NkiTensor,
    kt_hbm: nl.NkiTensor,
    kpos_hbm: nl.NkiTensor,
    weights_hbm: nl.NkiTensor,
    qpos_hbm: nl.NkiTensor,
    score_hbm: nl.NkiTensor,
    topk_hbm: nl.NkiTensor,
    cfg: GlmIndexerConfig,
    end_pos: int,
    tile_lo: int,
    tile_hi: int,
    score_out: nl.NkiTensor = None,
    kscale_hbm: nl.NkiTensor = None,
    k_rows_hbm: nl.NkiTensor = None,
    q_resident: tuple = None,
    kpos_layout: tuple = None,
) -> None:
    """The chunked score and top-k with the group, chunk and position-block loops as device loops.

    ``kpos_layout``: ``(s_kv_local, cp_degree, interleave)`` of the gathered
    keys. When their positions repeat block to block up to an offset, the pipelined mask reads ONE
    resident key-position tile and folds each block's offset into the per-query scalar, instead of
    broadcasting the block's positions from HBM (a DMA Vector waited on every block).

    ``q_resident``: ``(qt_sb, w_sb)`` already filled by the projection for
    the single query tile, so the panel and the head weights are not reloaded from HBM.

    ``kt_hbm`` None (bf16 K): each position block is DMA-transposed straight from the token-major
    ``k_rows_hbm`` ``[ctx, head_dim]``, so no whole-context pre-transpose.

    Unrolled, the chunked path traces one score body per (group, chunk): 64 x 17 per core at 1M, and
    the neuronx-cc time grows superlinearly with that count. Here each loop body is traced once, so the
    instruction count does not depend on ``s_local`` or ``s_global``.

    The chunk geometry is the unrolled path's, so each chunk's top-k ranks the same scores. The chunk
    index is a device value in stage 1, so stage 1 stores chunk-LOCAL positions and the merge adds each
    chunk's base, a trace-time constant per candidate column block.

    Full panels run in the rolled group loop; the remainder panel, which holds any partial last
    tile, runs once unrolled with its real row counts. ``score_out`` is the optional debug copy of
    the full score (see ``emit_score``).
    """
    panel = cfg.q_panel_tiles
    ch = cfg.score_chunk
    k = cfg.index_topk
    n_ch = div_ceil(end_pos, ch)
    last_w = end_pos - (n_ch - 1) * ch
    n_roll_ch = n_ch if last_w == ch else n_ch - 1
    ctx_row = kpos_hbm.shape[0]
    # The one-op causal mask (bf16 K only) reads a -2^102-scaled position table.
    fused_mask = kscale_hbm is None and ctx_row % P_MAX == 0
    kpos_src_hbm = scale_key_positions(sbm, kpos_hbm, ctx_row) if fused_mask else kpos_hbm
    topk_flat = topk_hbm.reshape((cfg.s_local, k)) if topk_hbm is not None else None
    out_w = score_out.shape[-1] if score_out is not None else 0

    pipelined = chunk_pipeline_enabled(cfg, end_pos)
    if pipelined:
        geo = pipeline_chunk_geometry(end_pos, k)
        n_ch = len(geo)
        ch = max(rw for _, _, rw in geo)  # the ring width: the widest chunk ranked
    cand_v_h, cand_p_h = alloc_chunk_candidates(cfg, n_ch, k, panel * P_MAX)
    # One score ring per chunk parity, as SEPARATE HBM tensors: dependencies are tracked per tensor, so
    # two halves of one ring would still order chunk c's score writes after chunk c-1's top-k reads.
    rings = [score_hbm]
    if pipelined:
        ring_shape = (score_hbm.shape[0], score_hbm.shape[1], ch)
        if ch != score_hbm.shape[2]:
            rings = [
                nl.ndarray(
                    ring_shape, dtype=score_hbm.dtype, buffer=nl.private_hbm, name=scoped_name(f"score_ring_a{_uid()}")
                )
            ]
        rings.append(
            nl.ndarray(
                ring_shape, dtype=score_hbm.dtype, buffer=nl.private_hbm, name=scoped_name(f"score_ring_b{_uid()}")
            )
        )
        if geo[-1][2] != ch:
            # A short last chunk gets its own ring at its own width: stage-1's 8-query load is one
            # contiguous run only when the ring row IS the ranked chunk (else it does not lower).
            last_shape = (score_hbm.shape[0], score_hbm.shape[1], geo[-1][2])
            rings.append(
                nl.ndarray(
                    last_shape, dtype=score_hbm.dtype, buffer=nl.private_hbm, name=scoped_name(f"score_ring_l{_uid()}")
                )
            )

    def _rid(c):
        """Ring of chunk ``c``: the parity rings, or the short last chunk's own."""
        return 2 if pipelined and c == n_ch - 1 and len(rings) == 3 else c % 2

    # Ring parity of the chunk being scored (trace-time: the pipelined chunk loop is unrolled).
    ring_par = [0]
    # Static (pre-built) descriptors for the score loop's own DMAs when pipelined: their offsets are then
    # trace-time constants, and keeping them off GpSimd's software DGE is what lets the stage-1 top-k
    # there overlap the next chunk's score. (HW DGE is rejected for these: NCC_IBIR098.)
    sdma = {"dge_mode": dge_mode.none} if pipelined else {}

    sbm.open_scope(name="score_invariants")
    if q_resident is not None:
        kernel_assert(panel == 1 and tile_lo == 0 and tile_hi == 1, "a resident Q panel holds one query tile")
        qt_sb, w_sb = q_resident
    else:
        qt_sb = sbm.alloc_stack((P_MAX, panel, cfg.n_heads, P_MAX), nl.bfloat16, name=f"q_panel{_uid()}", align=32)
    kernel_assert(kt_hbm is not None or k_rows_hbm is not None, "the chunked score needs K")
    kt_dtype = kt_hbm.dtype if kt_hbm is not None else nl.bfloat16
    kt_sb = sbm.alloc_stack((P_MAX, POS_BLOCK), kt_dtype, name=f"k_block{_uid()}", align=32)
    if q_resident is None:
        w_sb = sbm.alloc_stack((P_MAX, panel, cfg.n_heads), nl.float32, name=f"weights{_uid()}")
    qpos_sb = sbm.alloc_stack((P_MAX, panel), nl.float32, name=f"qpos{_uid()}")
    qpos_s_sb = sbm.alloc_stack((P_MAX, panel), nl.float32, name=f"qpos_s{_uid()}") if fused_mask else None
    kpos_sb = sbm.alloc_stack((P_MAX, POS_BLOCK), nl.float32, name=f"kpos{_uid()}")
    ks_sb = sbm.alloc_stack((P_MAX, POS_BLOCK), nl.bfloat16, name=f"kscale{_uid()}") if kscale_hbm is not None else None
    # When pipelined, the masked score is written as bf16 (the ring's dtype), so
    # its ring store is a plain copy on a HW DGE queue. Cast in the DMA (fp32 -> bf16) it is SW DGE only,
    # i.e. GpSimd's queue, where every store waits behind the stage-1 topk running there.
    bf16_out = pipelined and fused_mask and score_hbm.dtype == nl.bfloat16
    # Pipelined + periodic key positions: (resident block-0 kposS tile, [P, n_blocks, panel] per-block
    # query scalars qposS - MASK_SCALE * offset); see kpos_layout.
    kpos_res = [None]
    kpos_offs = block_periodic_key_offsets(end_pos, kpos_layout) if (pipelined and fused_mask) else None
    # col_headsum at 33-64 resident query rows: every block evicts + masks from PSUM straight into the bf16
    # out ring, so the fp32 accumulators are dead; dropping them (32 KB with pingpong) is what makes the
    # 32-tile ReLU ring fit beside the 3-set stage-1 scratch of a 64-row panel.
    col_wide = col_wide_enabled(cfg, end_pos) and q_resident is not None and bf16_out and kpos_offs is not None
    acc_sb = (
        None if col_wide else sbm.alloc_stack((P_MAX, panel, POS_BLOCK), cfg.score_acc_dtype, name=f"score_acc{_uid()}")
    )
    bufs_a = (kt_sb, kpos_sb, ks_sb, acc_sb, None)
    bufs_b = bufs_a
    if cfg.score_pingpong:
        bufs_b = (
            sbm.alloc_stack((P_MAX, POS_BLOCK), kt_dtype, name=f"k_block_b{_uid()}", align=32),
            kpos_sb,  # shared: its reuse only orders block b+1's MASK, not its matmuls (pipelined: see pipe_bufs)
            sbm.alloc_stack((P_MAX, POS_BLOCK), nl.bfloat16, name=f"kscale_b{_uid()}") if ks_sb is not None else None,
            None
            if col_wide
            else sbm.alloc_stack((P_MAX, panel, POS_BLOCK), cfg.score_acc_dtype, name=f"score_acc_b{_uid()}"),
            None,
        )
    # Pipelined, allocated per panel in _group (scope "score_pipe", closed before the merge): the masked
    # bf16 scores, a 3-deep ring so a store may trail its block by two blocks, and set B's own key
    # positions, so block b+1's broadcast never waits on block b's mask.
    pipe_bufs = [None]
    # Flat path: running block count (selects the buffer set and out tile) and the deferred ring store.
    blk_ctr = [0]
    pend_store = [None]

    def _flat_bufs():
        j = blk_ctr[0]
        blk_ctr[0] += 1
        kt_b, kpos_b, ks_b, acc_b, _ = bufs_a if j % 2 == 0 else bufs_b
        if pipe_bufs[0] is None:
            return (kt_b, kpos_b, ks_b, acc_b, None)
        out_ring, kpos_alt = pipe_bufs[0]
        return (kt_b, kpos_b if j % 2 == 0 else kpos_alt, ks_b, acc_b, out_ring[j % 3])

    def _flush_store():
        if pend_store[0] is not None:
            pend_store[0]()
            pend_store[0] = None

    def _score_out_dst(row, rows, pos, blk):
        """``score_out[row : row + rows, pos : pos + blk]`` for trace-time or device ``row`` / ``pos``."""
        if isinstance(row, int):
            return score_out[row : row + rows, nl.ds(pos, blk)]
        return score_out.reshape((cfg.s_local * out_w,)).ap(
            pattern=[[out_w, rows], [1, blk]], offset=0, scalar_offset=row * out_w + pos, indirect_dim=0
        )

    # pack_heads operands of the current query panel, built once per panel in _group.
    pack_hoist = [None]
    # Whether the resident key positions / query scalars of the current panel use the col_headsum layout.
    col_on = [False]
    # Rolled col_headsum: the panel's quadrant-replicated qposS (see _group); key positions load per block.
    roll_q = [None]
    pad_tiles = {}
    # Pipelined: the stage-1 top-k's persistent SBUF (allocated in _group, outside the chunk loop).
    s1_scratch = [None]

    def _pos_block(pos, local, blk, row0, tile_rows, bufs=bufs_a):
        """Score one position block into the ring at chunk column ``local``. Either may be a device value.

        With an ``out`` tile in ``bufs`` (pipelined), the ring store is emitted after the NEXT block's loads
        (see ``_flush_store``): the Sync queue is in order, so a store issued right after its block would hold
        the next block's K transpose until this block's mask is done.
        """
        kt_sb, kpos_sb, ks_sb, acc_sb, out_sb = bufs
        if kt_hbm is not None:
            _load_kt_block(kt_sb, kt_hbm, pos, blk, cfg.head_dim)
        else:
            load_k_block_transposed(kt_sb, k_rows_hbm, pos, blk, cfg.head_dim)
        # The mask reads only the panel's query rows, so broadcast the key
        # positions to those rows, not to all 128 partitions.
        mask_rows = max(tile_rows)
        qpos_m = qpos_s_sb
        n_pos_r, cw_r = col_quad_geometry(mask_rows)
        # Rolled col_headsum: blocks narrower than n_pos column tiles (a chunk's tail) keep the plain head-sum.
        roll_col = roll_q[0] is not None and out_sb is None and blk % (n_pos_r * MATMUL_TILE) == 0
        if out_sb is not None and kpos_res[0] is not None:
            kpos_sb, s_tab = kpos_res[0]
            qpos_m = s_tab[:, blk_ctr[0] - 1, :]
        elif roll_col:
            # Quadrant layout: kpos_sb[pq * cw + r, n] = kposS[pos + pq * blk / n_pos + n].
            qw_r = blk // n_pos_r
            for pq in range(n_pos_r):
                nisa.dma_copy(
                    dst=kpos_sb[pq * cw_r : pq * cw_r + mask_rows, :qw_r],
                    src=kpos_src_hbm.reshape((1, ctx_row))[0:1, nl.ds(pos + pq * qw_r, qw_r)].broadcast(
                        dim=0, size=mask_rows
                    ),
                    **sdma,
                )
            qpos_m = roll_q[0]
        else:
            nisa.dma_copy(
                dst=kpos_sb[:mask_rows, :blk],
                src=kpos_src_hbm.reshape((1, ctx_row))[0:1, nl.ds(pos, blk)].broadcast(dim=0, size=mask_rows),
                **sdma,
            )
        if ks_sb is not None:
            nisa.dma_copy(
                dst=ks_sb[:P_MAX, :blk],
                src=kscale_hbm.reshape((1, ctx_row))[0:1, nl.ds(pos, blk)].broadcast(dim=0, size=P_MAX),
            )
        if out_sb is not None:
            _flush_store()
        n_in_group = len(tile_rows)
        # pack_heads (hoisted) evicts + masks straight from PSUM, so it can write the bf16 tile itself.
        evict_to_out = out_sb is not None and pack_hoist[0] is not None
        # col_headsum: the quadrant layout of the resident key positions and the score tile (built in _group).
        col_mode = col_on[0] and ((evict_to_out and kpos_res[0] is not None) or roll_col)
        masked = score_block_for_group(
            sbm,
            qt_sb,
            kt_sb,
            w_sb,
            out_sb if evict_to_out else acc_sb,
            cfg,
            n_in_group,
            tile_rows,
            blk,
            fused_mask=(kpos_sb, qpos_m) if fused_mask else None,
            pack_consts=pack_hoist[0],
            col_mode=col_mode,
        )
        res_sb = out_sb if out_sb is not None else acc_sb
        for i in range(n_in_group):
            if fused_mask and not masked:
                acc_i = acc_sb[: tile_rows[i], i, :blk]
                apply_causal_mask_min(res_sb[: tile_rows[i], i, :blk], acc_i, kpos_sb, qpos_m, i, tile_rows[i], blk)
            elif not fused_mask:
                apply_causal_mask(sbm, acc_sb, kpos_sb, qpos_sb, i, tile_rows[i], blk, kscale_sb=ks_sb)
        ring = rings[ring_par[0]]

        def _store():
            if col_mode:
                kernel_assert(score_out is None, "col_headsum does not support emit_score")
                n_pos, cw = col_quad_geometry(max(tile_rows))
                qw = blk // n_pos
                for i in range(n_in_group):
                    for qd in range(n_pos):
                        nisa.dma_copy(
                            dst=ring[i, 0 : tile_rows[i], nl.ds(local + qd * qw, qw)],
                            src=res_sb[qd * cw : qd * cw + tile_rows[i], i, :qw],
                            **({"dge_mode": dge_mode.hwdge, "engine": nisa.engine.sync} if out_sb is not None else sdma),
                        )
                return
            for i in range(n_in_group):
                nisa.dma_copy(
                    dst=ring[i, 0 : tile_rows[i], nl.ds(local, blk)],
                    src=res_sb[: tile_rows[i], i, :blk],
                    **({"dge_mode": dge_mode.hwdge, "engine": nisa.engine.sync} if out_sb is not None else sdma),
                )
                if score_out is not None and out_sb is None:
                    nisa.dma_copy(
                        dst=_score_out_dst(row0 + i * P_MAX, tile_rows[i], pos, blk),
                        src=res_sb[: tile_rows[i], i, :blk],
                    )

        if out_sb is not None:
            pend_store[0] = _store
        else:
            _store()

    def _chunk(ci, c0, cw, row0, tile_rows, hooks=None, rank_w=None):
        """Score chunk ``ci`` (``cw`` positions from ``c0``), then rank it into the candidates.

        ``hooks`` (pipelined): the previous chunk's split stage-1 steps ``(issue, finish)``. Batch b is
        issued right AFTER block b and finished after block b + 1. The score-ring store of each block is
        on GpSimd's queue (HW DGE is rejected for it), so a topk issued before the block would hold that
        store -- and the pingpong accumulator it reads -- for the whole topk. Issued after, the topk runs
        under block b + 1, and its decode (Vector) is queued only once it is done.
        """
        n_full = cw // POS_BLOCK
        tail = cw - n_full * POS_BLOCK
        tail0 = n_full * POS_BLOCK  # the tail's first column
        pend, issued = list(hooks or []), []

        # Stage-1 batches in flight (issued, not finished) share len(sets) scratch sets: finish the oldest before an
        # issue would reuse its set (program order decides what a finish reads).
        n_sets = len(s1_scratch[0]) if (s1_scratch[0] is not None and isinstance(s1_scratch[0], list)) else _S1_FINISH_LAG + 1

        def _after_block():
            if pend:
                while len(issued) >= n_sets:
                    issued.pop(0)[1]()
                step = pend.pop(0)
                step[0]()
                issued.append(step)
            if len(issued) > _S1_FINISH_LAG:
                issued.pop(0)[1]()

        if cfg.flat_loops:
            for b in range(n_full):
                _pos_block(c0 + b * POS_BLOCK, b * POS_BLOCK, POS_BLOCK, row0, tile_rows, _flat_bufs())
                _after_block()
        elif cfg.score_pingpong and n_full > 1:
            # Two blocks per iteration on alternating buffer sets: block 2b+1's loads and matmuls
            # only depend on set B, so they overlap block 2b's eviction and score store on set A.
            def _pair(b):
                _pos_block(c0 + (2 * b) * POS_BLOCK, (2 * b) * POS_BLOCK, POS_BLOCK, row0, tile_rows, bufs_a)
                _pos_block(c0 + (2 * b + 1) * POS_BLOCK, (2 * b + 1) * POS_BLOCK, POS_BLOCK, row0, tile_rows, bufs_b)

            n_pairs = n_full // 2
            if n_pairs > 1:
                nl.fori_loop(0, n_pairs, _pair)
            else:
                _pair(0)
            if n_full % 2:
                last = n_full - 1
                _pos_block(c0 + last * POS_BLOCK, last * POS_BLOCK, POS_BLOCK, row0, tile_rows, bufs_a)
        elif n_full > 1:
            nl.fori_loop(0, n_full, lambda b: _pos_block(c0 + b * POS_BLOCK, b * POS_BLOCK, POS_BLOCK, row0, tile_rows))
        elif n_full == 1:
            _pos_block(c0, 0, POS_BLOCK, row0, tile_rows)
        if tail:
            _pos_block(
                c0 + tail0,
                tail0,
                tail,
                row0,
                tile_rows,
                _flat_bufs() if cfg.flat_loops else bufs_a,
            )
            _after_block()
        for step in issued:
            step[1]()
        for step in pend:
            step[0]()
            step[1]()
        rw = ch if rank_w is None else rank_w
        if cw < rw or not pipelined:
            _flush_store()
        if cw < rw:
            # -inf the ring tail so the snake ranks a full chunk without the pad winning.
            # One -inf tile per pad width.
            pad_sb = pad_tiles.get(rw - cw)
            if pad_sb is None:
                pad_sb = sbm.alloc_stack((P_MAX, rw - cw), nl.float32, name=f"cpad{_uid()}")
                nisa.memset(dst=pad_sb[:P_MAX, : rw - cw], value=float("-inf"))
                pad_tiles[rw - cw] = pad_sb
            for i, rows in enumerate(tile_rows):
                nisa.dma_copy(dst=rings[ring_par[0]][i, 0:rows, cw:rw], src=pad_sb[:rows, : rw - cw])
        if not pipelined:
            _rank_chunk(ci, tile_rows, 0)

    def _score_out_chunk(ci, par, row0, tile_rows):
        """Pipelined: the debug score copy of chunk ``ci``, from its ring (HBM -> HBM), once it is ranked.

        Per block from the out tile it would be a cast DMA on GpSimd's queue, behind the stage-1 topk,
        holding an out tile (and so the score) for the topk's duration.
        """
        if score_out is None or pipe_bufs[0] is None:
            return
        c0, cw, _ = geo[ci]
        for i, rows in enumerate(tile_rows):
            nisa.dma_copy(dst=_score_out_dst(row0 + i * P_MAX, rows, c0, cw), src=rings[par][i, 0:rows, 0:cw])

    def _rank_steps(ci, tile_rows, par, ring_load_mode=dge_mode.hwdge, decode_engine=None, ring_load_engine=None):
        """The split stage-1 of chunk ``ci`` (ring parity ``par``): ``(issue, finish)`` per 8-query batch."""
        steps = []
        for i, rows in enumerate(tile_rows):
            steps += topk_chunk_stage1(
                sbm,
                rings[par],
                i,
                cand_v_h,
                cand_p_h,
                0,
                rows,
                ci,
                geo[ci][2] if pipelined else ch,
                k,
                i * P_MAX,
                local_positions=True,
                unroll_batches=True,
                scratch=s1_scratch[0],
                split=True,
                dma_mode=ring_load_mode,
                raw_positions=pipelined,
                ring_stride=rings[par].shape[2],
                decode_engine=decode_engine,
                ring_load_engine=ring_load_engine,
            )
        return steps

    def _rank_chunk(ci, tile_rows, par):
        """Stage-1 top-k of chunk ``ci`` from ring parity ``par`` into the candidate arrays."""
        for i, rows in enumerate(tile_rows):
            topk_chunk_stage1(
                sbm,
                rings[par],
                i,
                cand_v_h,
                cand_p_h,
                0,
                rows,
                ci,
                geo[ci][2] if pipelined else ch,
                k,
                i * P_MAX,
                local_positions=True,
                unroll_batches=cfg.flat_loops,
                scratch=s1_scratch[0][0] if pipelined else None,
                raw_positions=pipelined,
                ring_stride=rings[par].shape[2],
            )

    def _group(row0, tile_rows):
        """One panel of query tiles from query row ``row0`` (a device value when rolled)."""
        for i, rows in enumerate(tile_rows):
            row = row0 + i * P_MAX
            if q_resident is None:
                load_q_panel_transposed(sbm, q_hbm, qt_sb, cfg, 0, rows, i, row=row)
                nisa.dma_copy(dst=w_sb[:rows, i, : cfg.n_heads], src=weights_hbm[nl.ds(row, rows), 0 : cfg.n_heads])
            nisa.dma_copy(dst=qpos_sb[:rows, i : i + 1], src=qpos_hbm.reshape((cfg.s_local, 1))[nl.ds(row, rows), 0:1])
            if fused_mask:
                nisa.tensor_scalar(
                    dst=qpos_s_sb[:rows, i : i + 1],
                    data=qpos_sb[:rows, i : i + 1],
                    op0=nl.add,
                    operand0=0.5,
                    op1=nl.multiply,
                    operand1=MASK_SCALE,
                )
        g_pack = P_MAX // tile_rows[0]
        hoist = (
            cfg.pack_heads
            and g_pack >= 2
            and cfg.n_heads % g_pack == 0
            and all(r == tile_rows[0] for r in tile_rows)
            and tile_rows[0] & (tile_rows[0] - 1) == 0
        )
        if hoist:
            # Own scope: the chunk/block loops below rotate their sections, which must not hand these out.
            sbm.open_scope(name="pack_hoist")
            pack_hoist[0] = build_pack_heads_consts(sbm, qt_sb, w_sb, cfg, len(tile_rows), tile_rows)
        if pipelined:
            sbm.open_scope(name="s1_scratch")
            s1_scratch[0] = alloc_pipelined_s1_sets(sbm, ch, k, tile_rows)
        if pipelined and bf16_out:
            sbm.open_scope(name="score_pipe")
            pipe_bufs[0] = (
                [
                    sbm.alloc_stack((P_MAX, panel, POS_BLOCK), nl.bfloat16, name=f"score_out{j}{_uid()}")
                    for j in range(3)
                ],
                sbm.alloc_stack((P_MAX, POS_BLOCK), nl.float32, name=f"kpos_b{_uid()}")
                if cfg.score_pingpong
                else kpos_sb,
            )
            if kpos_offs is not None:
                mrows = max(tile_rows)
                kc = sbm.alloc_stack((P_MAX, POS_BLOCK), nl.float32, name=f"kpos_res{_uid()}", align=64)
                nkb = min(POS_BLOCK, end_pos)
                # The quadrant layout puts n_pos copies of the panel's query rows on 32- (<= 32 rows, 4
                # copies) or 64-partition (<= 64 rows, 2 copies) ranges; larger panels keep the full-width
                # head-sums.
                n_pos, cw = col_quad_geometry(mrows)
                col_on[0] = cfg.col_headsum and hoist and mrows <= COL_MAX_ROWS and nkb % (n_pos * MATMUL_TILE) == 0
                kernel_assert(col_on[0] or not col_wide, "col_headsum: predicted the wide layout but did not take it")
                if col_on[0]:
                    # Quadrant layout: kc[qd * cw + r, n] = kposS[qd * nkb / n_pos + n].
                    qw = nkb // n_pos
                    for qd in range(n_pos):
                        nisa.dma_copy(
                            dst=kc[qd * cw : qd * cw + mrows, :qw],
                            src=kpos_src_hbm.reshape((1, ctx_row))[0:1, qd * qw : (qd + 1) * qw].broadcast(
                                dim=0, size=mrows
                            ),
                            **sdma,
                        )
                else:
                    nisa.dma_copy(
                        dst=kc[:mrows, :nkb],
                        src=kpos_src_hbm.reshape((1, ctx_row))[0:1, 0:nkb].broadcast(dim=0, size=mrows),
                        **sdma,
                    )
                # Padded to 1 KB so the buffers allocated after it keep their alignment.
                n_pad = div_ceil(len(kpos_offs) * panel, 256) * 256 // panel
                s_tab = sbm.alloc_stack((P_MAX, n_pad, panel), nl.float32, name=f"qpos_blk{_uid()}", align=64)
                qsrc, qrows = qpos_s_sb, mrows
                if col_on[0]:
                    # Every position range masks the same queries: replicate qposS to rows qd * cw + r.
                    qsrc = sbm.alloc_stack((P_MAX, panel), nl.float32, name=f"qpos_s4{_uid()}")
                    nisa.memset(dst=qsrc, value=0.0)
                    for qd in range(n_pos):
                        nisa.dma_copy(dst=qsrc[qd * cw : qd * cw + mrows, :], src=qpos_s_sb[:mrows, :])
                    qrows = (n_pos - 1) * cw + mrows
                for j, off in enumerate(kpos_offs):
                    for i, rows in enumerate(tile_rows):
                        r_ = qrows if col_on[0] else rows
                        # kposS(pos_j + c) - qposS = kposS(c) - (qposS - MASK_SCALE * off_j); all exact in fp32.
                        nisa.tensor_scalar(
                            dst=s_tab[:r_, j, i : i + 1],
                            data=qsrc[:r_, i : i + 1],
                            op0=nl.subtract,
                            operand0=float(MASK_SCALE * off),
                        )
                kpos_res[0] = (kc, s_tab)
        mrows_r = max(tile_rows)
        n_pos_g, cw_g = col_quad_geometry(mrows_r)
        roll_on = (
            not pipelined
            and cfg.col_headsum
            and cfg.col_rolled
            and hoist
            and fused_mask
            and mrows_r <= COL_MAX_ROWS
            and ks_sb is None
            and score_out is None
        )
        if roll_on:
            # Rolled col_headsum: every position range masks the same queries, so replicate qposS to rows
            # pq * cw + r once per panel; the key positions are loaded per block in the matching layout.
            sbm.open_scope(name="col_roll")
            qs4 = sbm.alloc_stack((P_MAX, panel), nl.float32, name=f"qpos_s4r{_uid()}")
            nisa.memset(dst=qs4, value=0.0)
            for pq in range(n_pos_g):
                nisa.dma_copy(dst=qs4[pq * cw_g : pq * cw_g + mrows_r, :], src=qpos_s_sb[:mrows_r, :])
            roll_q[0] = qs4
            col_on[0] = True
        if pipelined:
            blk_ctr[0] = 0  # block index into kpos_offs / s_tab, from this panel's first block
            # Unrolled and software-pipelined: chunk c-1 is ranked (GpSimd) while chunk c is scored.
            for c in range(n_ch):
                ring_par[0] = _rid(c)
                # Mid-score ring loads on GpSimd's queue (SW DGE): each waits for the previous topk to free
                # its source tile, and on the in-order Sync queue that wait held the next K transpose.
                hooks = _rank_steps(c - 1, tile_rows, _rid(c - 1), dge_mode.swdge) if c > 0 else None
                _chunk(c, geo[c][0], geo[c][1], row0, tile_rows, hooks=hooks, rank_w=geo[c][2])
                if c > 0:
                    _score_out_chunk(c - 1, _rid(c - 1), row0, tile_rows)
            _flush_store()
            # The last chunk's stage-1 has nothing left to overlap: split it too, so batch b+1's ring
            # load lands (second source tile) while batch b's topk runs.
            last_steps = _rank_steps(n_ch - 1, tile_rows, _rid(n_ch - 1))
            for b, step in enumerate(last_steps):
                step[0]()
                if b >= _S1_FINISH_LAG:
                    last_steps[b - _S1_FINISH_LAG][1]()
            for step in last_steps[max(0, len(last_steps) - _S1_FINISH_LAG) :]:
                step[1]()
            _score_out_chunk(n_ch - 1, _rid(n_ch - 1), row0, tile_rows)
            ring_par[0] = 0
            # Free the score/stage-1 SBUF before the merge allocates its own.
            if pipe_bufs[0] is not None:
                sbm.close_scope()  # score_pipe
                pipe_bufs[0] = None
                kpos_res[0] = None
                col_on[0] = False
            sbm.close_scope()  # s1_scratch
            s1_scratch[0] = None
            if hoist:
                sbm.close_scope()  # pack_hoist
                pack_hoist[0] = None
        else:
            if n_roll_ch > 1:
                nl.fori_loop(0, n_roll_ch, lambda ci: _chunk(ci, ci * ch, ch, row0, tile_rows))
            elif n_roll_ch == 1:
                _chunk(0, 0, ch, row0, tile_rows)
            if last_w < ch:
                _chunk(n_ch - 1, (n_ch - 1) * ch, last_w, row0, tile_rows)
        if roll_on:
            sbm.close_scope()  # col_roll
            roll_q[0] = None
            col_on[0] = False

        for i, rows in enumerate(tile_rows):
            topk_chunk_merge(
                sbm,
                cand_v_h,
                cand_p_h,
                topk_flat,
                n_ch,
                ch,
                k,
                i * P_MAX,
                rows,
                i * P_MAX,
                local_positions=True,
                q_row_dyn=row0,
                unroll_batches=cfg.flat_loops,
                raw_src_x=[rw // TOPK_SNAKE for _, _, rw in geo] if pipelined else None,
                chunk_bases=[c0 for c0, _, _ in geo] if pipelined else None,
            )
        if hoist and not pipelined:
            sbm.close_scope()  # pack_hoist
            pack_hoist[0] = None

    sbm.open_scope(interleave_degree=1, name="score_groups")
    n_tiles = tile_hi - tile_lo
    last_rows = min(P_MAX, cfg.s_local - (tile_hi - 1) * P_MAX)
    n_groups, rem = n_tiles // panel, n_tiles % panel
    if last_rows < P_MAX and rem == 0:
        # The partial tile must sit in the unrolled remainder panel, where row counts are static.
        n_groups, rem = n_groups - 1, panel
    row_lo = tile_lo * P_MAX
    if n_groups > 1:
        nl.fori_loop(
            row_lo, row_lo + n_groups * panel * P_MAX, lambda r: _group(r, [P_MAX] * panel), step=panel * P_MAX
        )
    elif n_groups == 1:
        _group(row_lo, [P_MAX] * panel)
    if rem:
        _group(row_lo + n_groups * panel * P_MAX, [P_MAX] * (rem - 1) + [last_rows])
    sbm.close_scope()  # score_groups
    sbm.close_scope()  # score_invariants


# --------------------------------------------------------------------------------------------
# Stage 6: top-k
# --------------------------------------------------------------------------------------------

TOPK_SNAKE = 16
"""Partitions one ``nisa.topk`` problem occupies. 8 such snakes fit a 128-partition tile, so one
call ranks 8 queries independently."""

TOPK_QUERIES_PER_CALL = P_MAX // TOPK_SNAKE  # 8

MASK_NEG = -3.0e38
SCORE_ROW_ALIGN = 2 * TOPK_SNAKE  # 32
TOPK_N_MAX = 65536


def derive_score_chunk(end_pos: int, topk_nseg: int = 0) -> int:
    """The score chunk width to use for ``end_pos``, or 0 to leave the score unchunked.

    Only chunk where the top-k is segmented anyway: below ``TOPK_N_MAX`` chunking would create a
    merge tree and position gather that a single ``nisa.topk`` does not need.

    Take the fewest legal chunks and then DIVIDE the row by that count, rather than using ``SEG_MAX``
    directly -- the latter gives the same chunk count with a nearly empty last chunk, whose tail pad
    is allocated at full chunk width in fp32 and overflows SBUF.
    """
    T = score_row_width(end_pos, topk_nseg)
    if T < TOPK_N_MAX:
        return 0
    n_chunks = div_ceil(T, SEG_MAX)
    return div_ceil(div_ceil(T, n_chunks), SCORE_ROW_ALIGN) * SCORE_ROW_ALIGN


def chunked_topk_sbuf_bytes(end_pos: int, k: int, forced: int = 0) -> int:
    """SBUF bytes per partition the SEGMENTED top-k holds live, from its scratch set.

    Mirrors ``_alloc_chunked_scratch``: the stage-1 snake source and its five ``[P_MAX, k]`` decode
    buffers, the candidate arrays at ``nseg * k/16`` columns, the merge outputs, and the one-hot
    gather's transposed blocks. Rounded up generously -- this only decides whether to FUSE, so
    over-estimating costs a little overlap and under-estimating costs a compile failure.
    """
    T, nseg, seg = topk_segments(end_pos, forced)
    del T
    src_x_seg = seg // TOPK_SNAKE
    cand_x = nseg * (k // TOPK_SNAKE)
    n_blk = div_ceil(cand_x, P_MAX)
    return (
        src_x_seg * 2  # s1_src bf16
        + k * (2 + 4 + 4 + 4 + 4)  # s1_val bf16 + s1_idx/pos/hi/lo 4 B each
        + cand_x * (2 + 4)  # cand_v bf16 + cand_p f32
        + k * (2 + 4)  # m_val bf16 + m_idx u32
        + n_blk * TOPK_SNAKE * 4  # oh_srct f32
        + 4096  # small per-partition constants and slack
    )


def topk_fits_with_panel(cfg: "GlmIndexerConfig", end_pos: int) -> bool:
    """Whether the top-k can be FUSED into the score loop, i.e. coexist with the resident Q panel.

    The single-segment top-k is small (one ``[P_MAX, k]`` set per batch) and always fits. The
    SEGMENTED one holds 65-75 KB/partition, which does not fit beside a full panel: at 128k the
    compile failed by 188 bytes on the merge's ``m_val`` (``Stack OOM: requested=4096 B,
    available=3908 B``).

    Shrinking the panel to make room is the worse trade. The panel divides K traffic by G, so
    halving G to 4 doubles it, while the overlap being bought is GpSimd top-k against PE/Vector
    score -- and at this config the score is far longer than the top-k, so there is almost nothing to hide.
    Unfusing keeps the panel at full G and gives the top-k the whole budget.
    """
    if end_pos <= 0:
        return True
    _T, nseg, _seg = topk_segments(end_pos, cfg.topk_nseg)
    if nseg == 1:
        return True
    per_tile = cfg.n_heads * cfg.head_dim * 2 + POS_BLOCK * 4
    shared = POS_BLOCK * 2 + POS_BLOCK * 2
    panel_bytes = cfg.q_panel_tiles * per_tile + shared
    return panel_bytes + chunked_topk_sbuf_bytes(end_pos, cfg.index_topk, cfg.topk_nseg) <= SBUF_BUDGET_BYTES


_ROLL_MIN_BLOCKS = 4
"""Position blocks above which AUTO rolls the score loop. At or below this the trace is small
enough that unrolling costs nothing, and at 1 block fori_loop is a no-op."""


def should_roll_score(cfg: "GlmIndexerConfig", end_pos: int) -> bool:
    """Whether to roll the position-block loop for this context."""
    if cfg.roll_score == 1:
        return True
    if cfg.roll_score < 0:
        return False
    return end_pos // POS_BLOCK > _ROLL_MIN_BLOCKS


def score_row_width(end_pos: int, forced: int = 0) -> int:
    """Width the padded score row must be allocated at, for either top-k path.

    ``topk_geometry``'s ``T`` is only 16-aligned in the single-segment case (its
    ``round_up_to_chunks`` rounds to 16), and an odd ``T/16`` fails the backend ISA check -- see
    SCORE_ROW_ALIGN. So take the wider of the geometry's T and a 32-aligned round-up. In the
    SEGMENTED case T is ``nseg * seg`` with ``seg`` already 32-aligned, so the geometry wins and the
    row is exactly what the chunked path indexes.
    """
    if end_pos <= 0:
        return 1
    T_geo, _nseg, _seg = topk_segments(end_pos, forced)
    return max(T_geo, div_ceil(end_pos, SCORE_ROW_ALIGN) * SCORE_ROW_ALIGN)


def topk_segments(end_pos: int, forced: int = 0) -> tuple:
    """``(T, nseg, seg)`` for a top-k over ``end_pos`` positions.

    Delegates to ``topk_geometry``; see :func:`topk_for_tile` for the constraints its shape rules
    encode.
    """
    T, nseg, seg, _spc = topk_geometry(end_pos, forced)
    return T, nseg, seg


def topk_over_score(
    sbm: SbufManager,
    score_hbm: nl.NkiTensor,
    topk_hbm: nl.NkiTensor,
    cfg: GlmIndexerConfig,
    end_pos: int,
    tile_lo: int,
    tile_hi: int,
) -> None:
    """Select each query's top ``index_topk`` positions from its score row.

    ``nisa.topk`` ranks a 16-partition SNAKE: element ``i`` of a query's row lives at partition
    ``i % 16``, column ``i // 16``. The score, though, is query-MAJOR -- one query per partition,
    positions along the free axis -- so the two layouts disagree, and that disagreement is the whole
    difficulty. Reading the row back from HBM through an access pattern does the relayout for free,
    which is why this stage consumes ``score_hbm`` rather than the SBUF accumulator.

    The gather uses a CONTIGUOUS-CHUNK snake (partition ``16g + p`` of group ``g`` holds query
    ``8t+g``'s chunk ``p``) rather than a round-robin one: a stride-16 free axis fails DMA lowering,
    while a stride-1 axis lowers fine. topk finds the same winners by VALUE either way -- only the
    returned index needs decoding, ``pos = (i % 16) * src_x + (i // 16)``.

    This is deliberately the UNFUSED version: the score round-trips HBM. At 1M that round trip is
    ~512 MB per query tile, so the chunked score folds a running merge into the score loop instead. The
    unfused form exists so the top-k can be validated on its own -- fusing first would make a
    top-k bug and a score bug indistinguishable.
    """
    T = score_hbm.shape[-1]
    src_x = T // TOPK_SNAKE
    kernel_assert(
        T % SCORE_ROW_ALIGN == 0,
        f"the score row ({T}) must be a multiple of {SCORE_ROW_ALIGN}: a 16-multiple is needed for the "
        f"snake gather to collapse to a contiguous run (else the DMA will not lower), and a 32-multiple "
        f"so src_x = row/16 is EVEN (else nisa.topk fails the backend ISA check, NCC_IXCG164)",
    )
    k = cfg.index_topk
    kernel_assert(k <= end_pos, f"index_topk={k} exceeds end_pos={end_pos}")

    sbm.open_scope(name="topk_phase")
    for tile in range(tile_lo, tile_hi):
        topk_for_tile(sbm, score_hbm, topk_hbm, cfg, end_pos, tile, tile - tile_lo, T, k)
    sbm.close_scope()


def _decode_snake_index(sbm, idx_sb, dst_sb, stride: int, base: int, rows: int, k: int) -> None:
    """``dst = (idx % 16) * stride + idx // 16 + base``.

    ``nisa.topk`` reports a SNAKE index: element ``i`` of its input sits at partition ``i % 16``,
    column ``i // 16``. Every consumer of that index -- mapping it to a position, or to a flat offset
    in the candidate array -- is this same affine decode with a different ``stride``/``base``.
    """
    sbm.open_scope(name="snake_decode")
    hi = sbm.alloc_stack((P_MAX, k), nl.uint32, name=f"sd_hi{_uid()}")
    lo = sbm.alloc_stack((P_MAX, k), nl.uint32, name=f"sd_lo{_uid()}")
    nisa.tensor_scalar(dst=hi[:rows, :k], data=idx_sb[:rows, :k], op0=nl.right_shift, operand0=4)
    nisa.tensor_scalar(dst=lo[:rows, :k], data=idx_sb[:rows, :k], op0=nl.bitwise_and, operand0=15)
    if base:
        nisa.tensor_scalar(dst=hi[:rows, :k], data=hi[:rows, :k], op0=nl.add, operand0=base)
    nisa.scalar_tensor_tensor(
        dst=dst_sb[:rows, :k],
        data=lo[:rows, :k],
        op0=nl.multiply,
        operand0=stride,
        op1=nl.add,
        operand1=hi[:rows, :k],
    )
    sbm.close_scope()


def topk_for_tile(
    sbm: SbufManager,
    score_hbm: nl.NkiTensor,
    topk_hbm: nl.NkiTensor,
    cfg: GlmIndexerConfig,
    end_pos: int,
    tile: int,
    slot: int,
    T: int,
    k: int,
) -> None:
    """Top-k for ONE query tile, segmenting the row when it exceeds ``nisa.topk``'s reach.

    Callable from inside the score loop so a tile's top-k (GpSimd) overlaps the next tile's score
    (PE/Scalar/Vector); as separate phases the two costs add.

    Segment-and-merge (``end_pos`` past ``nisa.topk``'s ``n < 65536``) is delegated to
    :func:`topk_over_score_chunked`, whose contract already matches this kernel's padded score tile.
    A call's 8 snakes must be spent on 8 POSITION SEGMENTS of one query, not 8 queries, or the
    stage-1 gather does not collapse to one descriptor; and merge winners must be resolved with a
    one-hot matmul, since ``local_gather`` cannot move data across partitions.
    """
    rows = min(P_MAX, cfg.s_local - tile * P_MAX)
    _T_geo, nseg, seg = topk_segments(end_pos, cfg.topk_nseg)
    if nseg == 1:
        # Rank the WHOLE PADDED ROW, not the geometry's seg: the snake gather only collapses to one
        # descriptor when 16*seg_x equals the row stride. Pad columns are -inf, so they never win.
        seg = T
    seg_x = seg // TOPK_SNAKE
    k_cols = k // TOPK_SNAKE

    kernel_assert(seg < TOPK_N_MAX, f"segment width {seg} must stay under {TOPK_N_MAX}")
    kernel_assert(k % TOPK_SNAKE == 0, f"index_topk={k} must be a multiple of {TOPK_SNAKE}")

    if nseg > 1:
        kernel_assert(
            _T_geo == T,
            f"the score row must be padded to the chunked geometry's T; row is {T}, geometry wants "
            f"{_T_geo}. Allocate the score buffer from topk_segments(end_pos, topk_nseg)[0].",
        )
        topk_over_score_chunked(
            sbm,
            P_MAX,
            end_pos,
            k,
            score_hbm,
            tile,
            topk_hbm.reshape((cfg.s_local, k)),
            sc_slot=slot,
            s_tile_base_q=tile * P_MAX,
            topk_nseg=cfg.topk_nseg,
        )
        return

    fast = cfg.fast_paths
    if fast:
        # Double-buffered batches: batch b+1's score load and topk do not wait (WAR) on batch b's
        # decode + store through one shared buffer set.
        sbm.open_scope(interleave_degree=2, name="topk_batches")
    for batch in range(div_ceil(rows, TOPK_QUERIES_PER_CALL)):
        q_base = batch * TOPK_QUERIES_PER_CALL
        n_q = min(TOPK_QUERIES_PER_CALL, rows - q_base)

        if not fast:
            sbm.open_scope(name="topk_batch")
        src_sb = sbm.alloc_stack((P_MAX, seg_x), nl.bfloat16, name=f"tk_src{_uid()}")
        val_sb = sbm.alloc_stack((P_MAX, k), nl.bfloat16, name=f"tk_val{_uid()}")
        idx_sb = sbm.alloc_stack((P_MAX, k), nl.uint32, name=f"tk_idx{_uid()}")
        pos_sb = sbm.alloc_stack((P_MAX, k_cols if fast else k), nl.int32, name=f"tk_pos{_uid()}")

        # 8 queries as 8 snakes, fp32 -> bf16 on the way in; the AP collapses to one contiguous run
        # because the row is padded to a snake multiple.
        #
        # Addressed by SLOT, not global tile: the ring is per-core and the score phase writes it at
        # score_hbm[tile - tile_lo], so any core whose tile_lo is non-zero writes slot 0.
        nisa.dma_copy(
            dst=src_sb[:P_MAX, :seg_x],
            src=score_hbm.ap(
                pattern=[[T, TOPK_QUERIES_PER_CALL], [seg_x, TOPK_SNAKE], [1, seg_x]],
                offset=(slot * P_MAX + q_base) * T,
            ),
            dge_mode=dge_mode.hwdge if fast else dge_mode.swdge,
        )
        nisa.topk(val_dst=val_sb[:P_MAX, :k], idx_dst=idx_sb[:P_MAX, :k], src=src_sb[:P_MAX, :seg_x], n=seg)
        # Segment-local snake index -> absolute position. Only the first k/16 columns of each
        # snake partition are the answer (16 partitions x k/16 per query), so the fast path
        # decodes only those (k/16 instead of k columns: 16x less Vector work).
        _decode_snake_index(sbm, idx_sb, pos_sb, seg_x, 0, P_MAX, k_cols if fast else k)

        # These ARE the answer, so write them out directly rather than through candidate staging.
        rows_out = n_q * TOPK_SNAKE
        nisa.dma_copy(
            dst=topk_hbm.reshape((cfg.s_local * TOPK_SNAKE, k_cols))[
                (tile * P_MAX + q_base) * TOPK_SNAKE : (tile * P_MAX + q_base) * TOPK_SNAKE + rows_out,
                :k_cols,
            ],
            src=pos_sb[:rows_out, :k_cols],
            dge_mode=dge_mode.hwdge if fast else dge_mode.swdge,
        )
        if fast:
            sbm.increment_section()
        else:
            sbm.close_scope()
    if fast:
        sbm.close_scope()


# --------------------------------------------------------------------------------------------
# Kernel entry -- stages 1-6

# --------------------------------------------------------------------------------------------


@dataclass
class IndexerKeys(nl.NKIObject):
    """The ``glm_indexer_keys`` output that ``glm_indexer_score`` reads: the config and the gathered K."""

    cfg: GlmIndexerConfig
    num_cores: int
    tile_lo: int
    tile_hi: int
    end_pos: int
    cp_degree: int
    topk_nseg: int
    k_cache: nl.NkiTensor
    k_ctx: nl.NkiTensor
    block_table: nl.NkiTensor
    block_size: int
    kv_cache_fp8_dtype: object
    k_local: nl.NkiTensor
    weights: nl.NkiTensor
    k_all: nl.NkiTensor  # None with the pre-transposed K
    qpos_local: nl.NkiTensor
    kt_all: nl.NkiTensor  # None without it


def _indexer_config(
    x, wq, wk, cos, end_pos, cp_degree, k_ctx, topk_nseg, roll_score, score_chunk, cp_interleave, index_topk,
    kv_fp8, score_acc_bf16, pack_head_pairs, pack_heads, fast_paths, wq_chunk_tiles, score_pingpong, flat_loops,
    col_headsum=False, col_rolled=False,
):
    """Return the indexer config, this rank's share of the valid context, and the core sharding."""
    s_local, dim = x.shape
    head_dim = wk.shape[1]
    n_heads = wq.shape[1] // head_dim
    q_lora_rank = wq.shape[0]
    rope_head_dim = 2 * cos.shape[1]
    # This rank's share of the valid context: with a resident prefix (APC) the block table covers
    # more than this step's s_local tokens, so the shard and the gather must span all of it.
    s_kv_local = max(s_local, end_pos // cp_degree) if end_pos else s_local
    s_global = k_ctx.shape[0] if k_ctx is not None else cp_degree * s_kv_local
    cfg = GlmIndexerConfig(
        s_local=s_local,
        s_global=s_global,
        n_heads=n_heads,
        head_dim=head_dim,
        dim=dim,
        q_lora_rank=q_lora_rank,
        rope_head_dim=rope_head_dim,
        index_topk=index_topk,
        score_chunk=derive_score_chunk(end_pos or s_global, topk_nseg) if score_chunk == 0 else max(score_chunk, 0),
        pack_head_pairs=pack_head_pairs,
        pack_heads=pack_heads,
        fast_paths=fast_paths,
        wq_chunk_tiles=wq_chunk_tiles,
        score_pingpong=score_pingpong,
        flat_loops=flat_loops,
        col_headsum=col_headsum,
        col_rolled=col_rolled,
        topk_nseg=topk_nseg,
        cp_interleave=cp_interleave,
        roll_score=roll_score,
        cp_ranks=0 if k_ctx is not None else cp_degree,
        kv_fp8=kv_fp8,
        score_acc_bf16=score_acc_bf16,
    )
    cfg.validate()
    _, num_cores, core_id = get_verified_program_sharding_info("glm_indexer", max_sharding=2)
    # LNC sharding: split the QUERY TILES across cores, not the heads.
    tiles_per_core = div_ceil(cfg.num_q_tiles, num_cores)
    tile_lo = core_id * tiles_per_core
    tile_hi = min(tile_lo + tiles_per_core, cfg.num_q_tiles)
    return cfg, s_kv_local, num_cores, tile_lo, tile_hi


def _project_tiles(sbm, cfg, tile_lo: int, tile_hi: int, project_tile) -> None:
    """Run ``project_tile(row_start, rows)`` over this core's query tiles.

    Past _ROLL_MIN_TILES a device loop, so the trace does not grow with s_local; below it unrolled and
    double buffered across query tiles.
    """
    sbm.open_scope(interleave_degree=2, name="tile_loop")
    if tile_hi - tile_lo > _ROLL_MIN_TILES and cfg.s_local % P_MAX == 0:
        nl.fori_loop(tile_lo * P_MAX, tile_hi * P_MAX, lambda r: project_tile(r, P_MAX), step=P_MAX)
    else:
        for tile in range(tile_lo, tile_hi):
            project_tile(tile * P_MAX, min(P_MAX, cfg.s_local - tile * P_MAX))
            sbm.increment_section()
    sbm.close_scope()


def glm_indexer_keys(
    x: nl.NkiTensor,
    wq: nl.NkiTensor,
    wk: nl.NkiTensor,
    k_norm_gamma: nl.NkiTensor,
    k_norm_beta: nl.NkiTensor,
    cos: nl.NkiTensor,
    sin: nl.NkiTensor,
    slot_mapping: nl.NkiTensor,
    weights_proj: nl.NkiTensor,
    q_pos: nl.NkiTensor,
    end_pos: int = 0,
    cp_degree: int = 1,
    replica_groups: tuple = None,
    topk_nseg: int = 0,
    roll_score: int = 0,
    weights_proj_transposed: bool = False,
    k_ctx: nl.NkiTensor = None,
    k_cache: nl.NkiTensor = None,
    score_chunk: int = 0,
    block_table: nl.NkiTensor = None,
    block_size: int = 0,
    cp_interleave: int = 0,
    index_topk: int = 2048,
    kv_fp8: bool = False,
    kv_cache_fp8_dtype=nl.float8_e4m3fn,
    score_acc_bf16: bool = False,
    pack_head_pairs: bool = False,
    pack_heads: bool = False,
    fast_paths: bool = False,
    wq_chunk_tiles: int = 1,
    score_pingpong: bool = False,
    flat_loops: bool = False,
    col_headsum: bool = False,
    col_rolled: bool = False,
) -> IndexerKeys:
    """The key half of the GLM indexer: K and head-weight projections, paged K cache write, CP gather of K.

    Call it inside a kernel trace, before the attention QKV, so QKV runs under the K all-gather. Then
    ``glm_indexer_score`` scores against the returned keys.

    Args:
        x: ``[s_local, dim]`` bf16 hidden states.
        wq: ``[q_lora_rank, n_heads * head_dim]`` bf16 Q weight.
        wk: ``[dim, head_dim]`` bf16 K weight.
        k_norm_gamma: ``[head_dim]`` fp32 LayerNorm scale for K.
        k_norm_beta: ``[head_dim]`` fp32 LayerNorm bias for K.
        cos: ``[s_local, rope_head_dim // 2]`` fp32 RoPE cosine table.
        sin: ``[s_local, rope_head_dim // 2]`` fp32 RoPE sine table.
        slot_mapping: ``[s_local]`` int32 flat cache slot per token, ``-1`` for padding.
        weights_proj: ``[n_heads, dim]`` bf16 per-head score-weight projection. The kernel
            computes ``x @ weights_proj^T * n_heads**-0.5 * head_dim**-0.5`` itself.
        q_pos: ``[s_local]`` fp32 global position of each local query; all-gathered to give the
            gathered K's positions.
        end_pos: valid cache positions to score against, i.e. the filled prefix. 0 skips the
            score phase entirely ([Testing]).
        cp_degree: CP ranks, so ``s_global = cp_degree * s_local``. 1 skips the collective.
        replica_groups: CP rank-lists for the in-kernel collectives, from the framework. None assumes
            a single group ``[0, cp_degree)``, which only holds for pure CP.
        topk_nseg: [Testing] force the top-k segment count; production derives it (0).
        roll_score: [Testing] force the position-block loop rolled (1) or unrolled (-1); production
            derives it (0).
        k_ctx: [Testing] ``[s_global, head_dim]`` bf16 context supplied by the caller instead of
            gathered from peers, so one rank can stand in for a whole CP group.
        block_table: ``[n_blocks]`` int32 page ids, logical block i -> cache block block_table[i].
            The score then reads K through this table rather than from the gathered K, which is what
            lets a cached prefix participate. Requires block_size.
        block_size: tokens per cache page. Must divide POS_BLOCK.
        cp_interleave: CP interleave granularity of the cache in tokens; 0 means contiguous.
        score_chunk: [Testing] force the score chunk width, or -1 to disable chunking; production
            derives it (0). Chunking is what bounds the score ring.
        index_topk: positions selected per query.
        kv_fp8: store the paged K cache in fp8: per-token quant over ``head_dim``, packed two
            channels per bf16 container with the bf16 dequant scale in-row (``kv_row_bf16_width``
            containers per row). ``k_cache`` (and ``k_ctx``) then carry those packed rows. The CP
            gather moves the packed rows, and the score multiplies by the fp8 K directly, applying
            each key's scale after the head sum, so K is never dequantized.
        kv_cache_fp8_dtype: fp8 format of the packed K -- ``nl.float8_e4m3fn`` (OCP, max 448, the
            default: trn3 is NeuronCore-v4, and the per-token scale gains ~1.9x range below each
            token's max over legacy) or ``nl.float8_e4m3`` (legacy, max 240, pre-v4 hardware).
        score_acc_bf16: [Experimental] accumulate the score in bf16 rather than fp32.
        k_cache: ``[n_slots, head_dim]`` bf16 paged K cache, owned by the caller and written in
            place. Its shape sets the slot count; the block table addresses it on the read side.
        pack_head_pairs, pack_heads, fast_paths, wq_chunk_tiles, score_pingpong, flat_loops:
            score options, kept in the returned config for ``glm_indexer_score``.

    Returns:
        ``IndexerKeys``: the config, ``k_local`` (this rank's K rows), ``weights`` (``[s_local, n_heads]``
        fp32 head weights), ``qpos_local``, and the gathered K: ``k_all``, or with the pre-transposed
        gather ``kt_all`` (``[cp * head_dim, s_kv_local]`` K-major).
    """
    cfg, s_kv_local, num_cores, tile_lo, tile_hi = _indexer_config(
        x, wq, wk, cos, end_pos, cp_degree, k_ctx, topk_nseg, roll_score, score_chunk, cp_interleave, index_topk,
        kv_fp8, score_acc_bf16, pack_head_pairs, pack_heads, fast_paths, wq_chunk_tiles, score_pingpong,
        flat_loops, col_headsum, col_rolled,
    )
    kernel_assert(k_cache is not None, "k_cache is a caller-owned input; pass the paged K cache")
    collective_hbm = nl.private_hbm if num_cores == 1 else nl.shared_hbm
    # The rows the CP gather moves: dense bf16 K, or the packed fp8 rows (half the traffic).
    kv_width = cfg.kv_cache_head_width
    k_local = nl.ndarray((cfg.s_local, kv_width), dtype=nl.bfloat16, buffer=collective_hbm, name=scoped_name("k_local"))
    # Per-head score weights, written by the projection and read back per panel by the score. Each core
    # writes and reads only its own query tiles.
    weights = nl.ndarray(
        (cfg.s_local, cfg.n_heads), dtype=nl.float32, buffer=nl.private_hbm, name=scoped_name("weights")
    )
    ctx = cfg.s_global
    kernel_assert(
        k_ctx is not None or s_kv_local == cfg.s_local or block_table is not None,
        f"a cached prefix (end_pos={end_pos} > cp_degree * s_local={cp_degree * cfg.s_local}) lives only "
        f"in the paged cache, so it needs block_table to be read",
    )
    qpos_local = nl.ndarray((cfg.s_local, 1), dtype=nl.float32, buffer=collective_hbm, name=scoped_name("qpos_local"))
    # Pre-transposed K: each rank transposes its shard before the gather. A score block must stay inside a
    # shard (or cover whole shards at a static position), and a device-looped chunk must start on a block.
    pipelined = chunk_pipeline_enabled(cfg, end_pos)
    kt_pre_on = (
        not kv_fp8
        and k_ctx is None
        and cp_degree > 1
        and (s_kv_local % POS_BLOCK == 0 or (POS_BLOCK % s_kv_local == 0 and pipelined))
        and (not cfg.score_chunk or pipelined or cfg.score_chunk % POS_BLOCK == 0)
    )
    # The gathered K: K-major ``kt_all`` with the pre-transposed K, else token-major ``k_all``.
    kt_all = (
        nl.ndarray(
            (cp_degree * cfg.head_dim, ctx // cp_degree),
            dtype=nl.bfloat16,
            buffer=collective_hbm,
            name=scoped_name("kt_all"),
        )
        if kt_pre_on
        else None
    )
    k_all = (
        None
        if kt_pre_on
        else nl.ndarray((ctx, kv_width), dtype=nl.bfloat16, buffer=collective_hbm, name=scoped_name("k_all"))
    )

    sbm = SbufManager(
        sb_lower_bound=0,
        sb_upper_bound=nl.tile_size.total_available_sbuf_size,
        logger=get_logger("glm_indexer"),
    )
    sbm.open_scope(name="indexer_root")
    # Weights are query-independent: hoist ONCE, outside the tile loop, in a scope the tile loop does
    # not increment_section() over.
    sbm.open_scope(name="weight_hoist")
    wkw_sb = load_wk_wp_k_major(sbm, wk, weights_proj, cfg, wp_transposed=weights_proj_transposed)
    gamma_sb = broadcast_norm_weight(sbm, k_norm_gamma, cfg.head_dim, "k_gamma")
    beta_sb = broadcast_norm_weight(sbm, k_norm_beta, cfg.head_dim, "k_beta")

    def _project_tile(row_start, rows):
        """Project the K and head weights of one query tile and write K to the paged cache."""
        k_sb = sbm.alloc_stack((P_MAX, cfg.head_dim), nl.float32, name=f"k_tile{_uid()}")
        w_sb = sbm.alloc_stack((P_MAX, cfg.n_heads), nl.float32, name=f"w_tile{_uid()}")
        k_post_sb = sbm.alloc_stack((P_MAX, cfg.head_dim), nl.float32, name=f"k_post{_uid()}")
        project_k(sbm, x, wkw_sb, cfg, row_start, rows, k_sb, w_sb)
        postprocess_k(sbm, k_sb, gamma_sb, beta_sb, cos, sin, cfg, row_start, rows, k_post_sb)
        nisa.dma_copy(dst=_row_slice(weights, row_start, rows, cfg.n_heads), src=w_sb[:rows, : cfg.n_heads])
        store_k_paged(
            sbm,
            k_post_sb,
            k_cache,
            slot_mapping,
            cfg,
            row_start,
            rows,
            fp8_dtype=kv_cache_fp8_dtype,
            local_out=k_local,
        )

    _project_tiles(sbm, cfg, tile_lo, tile_hi, _project_tile)
    sbm.close_scope()

    # Each core wrote only its own query tiles' K, but everything below -- the page linearize, the
    # gather source, the cp=1 copy -- reads the whole cache and k_local, including the other core's rows.
    if num_cores >= 2:
        nisa.core_barrier(k_cache, (0, 1))
        nisa.core_barrier(k_local, (0, 1))

    nisa.dma_copy(dst=qpos_local[: cfg.s_local, 0:1], src=q_pos.reshape((cfg.s_local, 1)))
    if k_ctx is not None:
        nisa.dma_copy(dst=k_all[0:ctx, :kv_width], src=k_ctx[0:ctx, :kv_width])
    else:
        # This rank's own pages made contiguous, then gathered. The gathered cache stays interleaved
        # in global position; the causal mask is what resolves it. Skipping the gather at cp_degree=1
        # is the ONLY difference between the two topologies.
        src_local = k_local
        if block_table is not None and cp_degree > 1:
            shard = nl.ndarray(
                (s_kv_local, kv_width), dtype=nl.bfloat16, buffer=nl.private_hbm, name=scoped_name("k_shard")
            )
            linearize_pages(sbm, k_cache, block_table, shard, cfg, s_kv_local, block_size)
            src_local = shard
        if kt_pre_on:
            # Each rank transposes its own K shard, so the gather gives K-major K for the score.
            rows = src_local.shape[0]
            kt_local = nl.ndarray(
                (cfg.head_dim, rows), dtype=nl.bfloat16, buffer=nl.private_hbm, name=scoped_name("kt_local")
            )
            sbm.open_scope(name="kt_pre")
            kt_stage = sbm.alloc_stack((P_MAX, POS_BLOCK), nl.bfloat16, name=f"kt_pre_stage{_uid()}", align=32)
            for c in range(0, rows, POS_BLOCK):
                cw = min(POS_BLOCK, rows - c)
                nisa.dma_transpose(
                    dst=kt_stage[: cfg.head_dim, :cw],
                    src=src_local.ap(pattern=[[cfg.head_dim, cw], [1, cfg.head_dim]], offset=c * cfg.head_dim),
                    axes=(1, 0),
                )
                nisa.dma_copy(dst=kt_local[0 : cfg.head_dim, c : c + cw], src=kt_stage[: cfg.head_dim, :cw])
            sbm.close_scope()
            ncc.all_gather(
                dsts=[kt_all],
                srcs=[kt_local],
                replica_group=cp_replica_group(cp_degree, replica_groups),
                collective_dim=0,
            )
        elif cp_degree > 1:
            gather_k_across_cp(src_local, k_all, cp_degree, replica_groups)
        else:
            rows_local = min(s_kv_local, src_local.shape[0])
            nisa.dma_copy(dst=k_all[:rows_local, :kv_width], src=src_local[:rows_local, :kv_width])
    sbm.close_scope()
    return IndexerKeys(
        cfg=cfg,
        num_cores=num_cores,
        tile_lo=tile_lo,
        tile_hi=tile_hi,
        end_pos=end_pos,
        cp_degree=cp_degree,
        topk_nseg=topk_nseg,
        k_cache=k_cache,
        k_ctx=k_ctx,
        block_table=block_table,
        block_size=block_size,
        kv_cache_fp8_dtype=kv_cache_fp8_dtype,
        k_local=k_local,
        weights=weights,
        k_all=k_all,
        qpos_local=qpos_local,
        kt_all=kt_all,
    )


def glm_indexer_score(
    qr: nl.NkiTensor,
    wq: nl.NkiTensor,
    cos: nl.NkiTensor,
    sin: nl.NkiTensor,
    q_pos: nl.NkiTensor,
    keys: IndexerKeys,
    after_k_gather=None,
    emit_score: bool = False,
    emit_proj: bool = False,
    zero_topk: bool = False,
):
    """The query half of the GLM indexer: Q projection, score against ``keys``, top-k.

    Args:
        qr: ``[s_local, q_lora_rank]`` bf16 latent Q.
        wq: ``[q_lora_rank, n_heads * head_dim]`` bf16 Q weight.
        cos, sin: ``[s_local, rope_head_dim // 2]`` fp32 RoPE tables.
        q_pos: ``[s_local]`` fp32 global position of each local query.
        keys: the ``glm_indexer_keys`` output of the same layer.
        after_k_gather: called after the Q projection, before the score. The GK issues its KV and
            next-layer weight all-gathers there, so they run under the score.
        emit_score: [Testing] also return the full [s_local, T] score.
        emit_proj: [Testing] also return the projected Q and the gathered K (not with ``kt_all``).
        zero_topk: [Testing] return zeros for the top-k output instead of the picks.

    Returns:
        ``topk`` -- ``[s_local, 16, index_topk // 16]`` int32 positions in SNAKE layout.
        With ``emit_proj`` / ``emit_score`` the projected Q, the gathered K and the score
        follow it, in that order.
    """
    cfg, num_cores, tile_lo, tile_hi = keys.cfg, keys.num_cores, keys.tile_lo, keys.tile_hi
    end_pos, cp_degree, topk_nseg = keys.end_pos, keys.cp_degree, keys.topk_nseg
    k_cache, k_ctx, block_table, block_size = keys.k_cache, keys.k_ctx, keys.block_table, keys.block_size
    kv_cache_fp8_dtype = keys.kv_cache_fp8_dtype
    kernel_assert(not (emit_proj and keys.kt_all is not None), "emit_proj needs the plain K gather, not kt_all")
    q_out = nl.ndarray((cfg.s_local, cfg.q_out_dim), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    weights, k_all, kt_all = keys.weights, keys.k_all, keys.kt_all
    ctx = cfg.s_global
    k_gathered = nl.ndarray((ctx, cfg.head_dim), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    score_width = score_row_width(end_pos, topk_nseg)
    _ring_slots = cfg.q_panel_tiles if cfg.score_chunk else score_ring_slots(div_ceil(cfg.num_q_tiles, num_cores))
    _ring_width = cfg.score_chunk if cfg.score_chunk else score_width
    score = nl.ndarray(
        (_ring_slots, P_MAX, _ring_width),
        dtype=nl.bfloat16,
        buffer=nl.private_hbm,
        name=scoped_name("score_ring"),
    )

    score_out = (
        nl.ndarray((cfg.s_local, score_width), dtype=nl.float32, buffer=nl.shared_hbm)
        if emit_score
        else nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    )
    topk = nl.ndarray(
        (cfg.s_local, TOPK_SNAKE, max(cfg.index_topk // TOPK_SNAKE, 1)), dtype=nl.int32, buffer=nl.shared_hbm
    )
    """
    Top-k picks -- the kernel's actual product.

    ``zero_topk`` leaves it all-zero, so a multi-rank test can compare the score, the gather and
    the mask while the kernel still returns every output. A reference's top-k cannot check the
    picks: nisa.topk runs sorted=False, and near the cut a small score difference reorders the
    winners. Check the top-k against the kernel's OWN score instead, on one rank (a long-context
    k_ctx case reaches the same end_pos on one rank).
    """

    sbm = SbufManager(
        sb_lower_bound=0,
        sb_upper_bound=nl.tile_size.total_available_sbuf_size,
        logger=get_logger("glm_indexer"),
    )
    sbm.open_scope(name="indexer_root")

    # With one query tile per core (segmented prefill: <= 128 queries per
    # rank) the projection writes Q and the head weights straight into the score's resident panel,
    # so neither is stored to HBM and reloaded. Its own scope, outside every scope the projection
    # and the score rotate, so no increment_section() hands these addresses out again.
    q_res = None
    if cfg.score_chunk and not emit_proj and cfg.num_q_tiles == 1 and num_cores == 1:
        sbm.open_scope(name="q_resident")
        q_res = (
            sbm.alloc_stack((P_MAX, 1, cfg.n_heads, P_MAX), nl.bfloat16, name=f"q_res{_uid()}", align=32),
            sbm.alloc_stack((P_MAX, 1, cfg.n_heads), nl.float32, name=f"w_res{_uid()}"),
        )

    # Weights are query-independent: hoist ONCE, outside the tile loop. They must also live in a
    # scope the tile loop does not increment_section() over, or the loop will hand their addresses
    # out again to the second tile.
    sbm.open_scope(name="weight_hoist")
    # (GLM_GK_SYNC_HOL): the qr transposes' schedule-only predecessors (QR_T_AFTER); None: off.
    qr_t_after = list(QR_T_AFTER[0]) if QR_T_AFTER[0] else None
    wq_rec = [] if qr_t_after is not None and QR_T_AFTER[1] else None
    wq_sb = load_weight_k_major(
        sbm, wq, cfg.q_lora_rank, cfg.q_out_dim, col_chunk=PSUM_FMAX * cfg.wq_chunk_tiles if cfg.fast_paths else 0,
        **({"rec": wq_rec} if wq_rec is not None else {}),
    )
    if wq_rec:
        qr_t_after += wq_rec

    def _project_tile(row_start, rows):
        """Project the Q of one query tile (``row_start`` is a trace-time int, or a device value when rolled)."""
        q_sb = sbm.alloc_stack((P_MAX, cfg.q_out_dim), nl.bfloat16, name=f"q_tile{_uid()}")
        q_rope = load_rope_tables(sbm, cos, sin, cfg, row_start, rows)
        project_q(sbm, qr, wq_sb, cfg, row_start, rows, q_sb, rope=q_rope, **({"qr_t_after": qr_t_after} if qr_t_after else {}))
        if q_res is not None:
            transpose_q_tile_to_panel(q_sb, q_res[0], cfg, rows, 0)
        else:
            nisa.dma_copy(dst=_row_slice(q_out, row_start, rows, cfg.q_out_dim), src=q_sb[:rows, : cfg.q_out_dim])

    _project_tiles(sbm, cfg, tile_lo, tile_hi, _project_tile)
    if q_res is not None:
        nisa.dma_copy(dst=q_res[1][: cfg.s_local, 0, : cfg.n_heads], src=weights[: cfg.s_local, : cfg.n_heads])
    sbm.close_scope()

    if num_cores >= 2:
        nisa.core_barrier(k_cache, (0, 1))
        nisa.core_barrier(keys.k_local, (0, 1))

    if after_k_gather is not None:
        after_k_gather()
    if emit_proj:
        if cfg.kv_fp8:
            dequant_k_packed(sbm, k_all, k_gathered, cfg, ctx, kv_cache_fp8_dtype)
        else:
            nisa.dma_copy(dst=k_gathered, src=k_all)

    if end_pos > 0:
        """
        The score is q(s_local) @ K(s_global): every one of THIS rank's queries scores against the
        whole CP context, which is why K comes from the gathered buffer rather than k_local. With a
        cached prefix that context is read from the paged cache through the block table, so the
        prefix's K participates without being recomputed.
        """
        kernel_assert(
            end_pos <= ctx,
            f"end_pos ({end_pos}) exceeds the gathered K's {ctx} positions",
        )
        fuse = topk_fits_with_panel(cfg, end_pos)
        _KT_PRE_SLOC[0] = ctx // cp_degree if kt_all is not None else 0
        score_over_context(
            sbm,
            q_out,
            k_all,
            weights,
            q_pos,
            score,
            topk,
            cfg,
            end_pos,
            tile_lo,
            tile_hi,
            fuse_topk=fuse,
            score_out=score_out if emit_score else None,
            k_cache_hbm=k_ctx if k_ctx is not None else k_cache,
            block_table_hbm=block_table if cp_degree <= 1 else None,
            block_size=block_size,
            kv_cache_fp8_dtype=kv_cache_fp8_dtype,
            q_resident=q_res,
            kt_pre=kt_all,
        )
        _KT_PRE_SLOC[0] = 0
        if not fuse:
            topk_over_score(sbm, score, topk, cfg, end_pos, tile_lo, tile_hi)
    if q_res is not None:
        sbm.close_scope()  # q_resident
    sbm.close_scope()

    if zero_topk:
        sbm.open_scope(name="zero_topk")
        z = sbm.alloc_stack((P_MAX, max(cfg.index_topk // TOPK_SNAKE, 1)), nl.int32, name=f"ztk{_uid()}")
        nisa.memset(dst=z, value=0)
        flat = topk.reshape((cfg.s_local * TOPK_SNAKE, max(cfg.index_topk // TOPK_SNAKE, 1)))
        for base in range(0, cfg.s_local * TOPK_SNAKE, P_MAX):
            rows = min(P_MAX, cfg.s_local * TOPK_SNAKE - base)
            nisa.dma_copy(dst=flat[base : base + rows, :], src=z[:rows, :])
        sbm.close_scope()

    outs = [topk]
    if emit_proj:
        outs += [q_out, k_gathered]
    if emit_score:
        outs.append(score_out)
    return tuple(outs)
