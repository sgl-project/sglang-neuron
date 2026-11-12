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

"""
Shapes for the GLM-5.2 DSA indexer kernels.

The grid is ``128 = seq32 x key4``: one sequence per
four ranks, those four splitting its context. So inside a kernel there is NO batch axis --
``num_tokens`` is one sequence's speculative tokens, and ``num_keys`` is one quarter of its context.

As in the MLA projection kernels, **no shard degree is a parameter**. Each kernel reads its widths off
the operands it is handed, so the same kernel serves a sharded and an unsharded run.
"""

from dataclasses import dataclass

import nki.language as nl

from ....utils.kernel_assert import kernel_assert
from ....utils.kernel_helpers import div_ceil

P_MAX = 128
"""Partition-dim max."""

KEY_LOAD_MAX = 512
"""Keys the score loop fetches per DMA: one load per scored chunk. Wider loads measured neutral (the loads
already hide behind compute) and cost SBUF the giga kernel needs; narrower ones expose the per-DMA setup.
"""

HEAD_SUM_BANK_COLS = 512
"""fp32 columns of the PSUM bank the score loop accumulates its key-major head sums into. The head sum is
``[128 keys, T_q]`` per 128-key tile, so one bank holds ``512 // T_q`` tiles, and the loop evicts, masks and
transposes once per such block (``IndexerScoreShape.head_sum_block_tiles``).
"""

PAIR_CHUNKS = 1
"""Key chunks per residency-mask and eviction pass. Two looked cheaper (half the instructions) but measured
slower, because the paired head-sum tile spans two PSUM banks.
"""

PARTITION_ALIGN = 32
"""SBUF partition ranges must start on a multiple of 32 (the backend refuses anything else), which bounds how the
score's ``(token, head)`` partition axis can be split per token.
"""

F_MAX = 512
"""Moving-free / PSUM-free max per bank, in fp32 elements."""

SCORE_PSUM_DTYPE = nl.float32
"""PSUM dtype of the per-chunk score tile. An fp8 key cache scores in two accumulating half-depth matmuls, so the
destination holds fp32 between them.
"""

SCORE_KEY_CHUNK = F_MAX
"""Keys per score and head-sum matmul, and the width of every per-chunk PSUM tile (``IndexerScoreShape.key_chunk``)."""

PSUM_BANKS = 8
"""PSUM banks available to one kernel."""

SNAKE_PARTITIONS = 16
"""Partitions ``nisa.topk`` gives one logical row: element ``e`` sits at partition ``e % 16``, column ``e // 16``, so a
128-partition tile carries 8 rows.
"""

TOPK_MAX_N = 65536
"""Exclusive upper bound on ``nisa.topk``'s ``n``. It is why the key-shard degree is at least 4 at 128k."""

IDX_TOPK_MAX_CALL_N = 32768
"""The widest local key range one indexer select ranks; a wider shard is ranked in chunks of this width
(``local_topk_chunked``) and the local selects assert it (``assert_topk_call_n``). It is the local stage's chunk width,
not the device's limit: the merges above it rank wider sets in one select (36,864 in the 1M prefill, 40,960 in the 1M
MTP draft), which ``tests/e2e/kernels/test_wide_topk_collectives.py`` checks on hardware.
"""


def local_topk_chunked(num_keys: int) -> bool:
    """Whether a local key range is too wide for one select and must be chunked (``IDX_TOPK_MAX_CALL_N``).

    False at 128k (32768 keys a rank), True at 512k with cp 4 (131072). Every caller that picks a local
    select by key count asks this, so the score layout, the residency mask and the select cannot disagree.
    """
    return num_keys > IDX_TOPK_MAX_CALL_N


def assert_topk_call_n(n: int, site: str) -> None:
    """Refuse, at trace time, an indexer top-k select over more than ``IDX_TOPK_MAX_CALL_N`` candidates.

    Python only: it emits no instruction, so a select it lets through traces exactly as before.
    """
    kernel_assert(
        n <= IDX_TOPK_MAX_CALL_N,
        f"{site}: one top-k select over n={n} candidates is wider than IDX_TOPK_MAX_CALL_N="
        f"{IDX_TOPK_MAX_CALL_N}, the widest that leaves the core's later collectives intact; rank it in chunks",
    )


MASK_BOUND = 1.0e30
"""Clamp of the residency mask and magnitude of the dropped-slot sentinel. The mask computes
``min(score, keep * 2 * MASK_BOUND - MASK_BOUND)``, so ``2 * MASK_BOUND`` must stay finite in bf16. A clamp, not a
zero: ``weights_proj`` has either sign, so a real score can be negative and must still outrank a dropped slot.
"""

NEG_INF_BF16 = -MASK_BOUND
"""Value of a masked or padded score slot. Finite on purpose: ``nisa.topk`` folds NaN to the bf16 minimum, which would
rank a masked slot above a legitimately negative score.
"""

"""Where the shape admits it (``snake_score_layout``), the score loop writes its scores straight into ``nisa.topk``'s
snake tile in SBUF: the head sums and the mask are laid out in snake order, so one PE transpose per 128 keys replaces
the per-block transposing DMA, the HBM score round trip and the top-k's snake load. Bit-identical.
"""


def snake_score_layout(num_keys: int, num_tokens: int) -> bool:
    """True when the shape admits the snake score tile: a block is 16 chunks x ``num_tokens`` columns
    per 128-key tile within ONE PSUM bank, the 16 snake rows divide into whole chunks, and one select ranks
    the whole snake (``local_topk_chunked``; a chunked select reads the HBM scores)."""
    tiles_per_chunk = SCORE_KEY_CHUNK // P_MAX
    return (
        num_tokens * SNAKE_PARTITIONS <= P_MAX
        and SNAKE_PARTITIONS * tiles_per_chunk * num_tokens <= HEAD_SUM_BANK_COLS
        and num_keys % (SNAKE_PARTITIONS * SCORE_KEY_CHUNK) == 0
        and not local_topk_chunked(num_keys)
    )


@dataclass
class IndexerQProjShape(nl.NKIObject):
    """Extents for the column-sharded indexer query up-projection."""

    num_tokens: int
    """Query tokens this rank projects."""

    q_lora_rank: int
    """COMPLETE query latent width, shared with the MLA path's ``q_a`` output."""

    q_shard: int
    """This rank's slice of ``index_n_heads * index_head_dim``. Its width IS the shard."""

    dtype: nl.DType
    """Kernel compute dtype."""

    @property
    def num_q_lora_tiles(self) -> int:
        """Partition tiles the contraction spans."""
        return div_ceil(self.q_lora_rank, P_MAX)


IDX_KEY_FP8_MAX = 448.0
"""float8_e4m3fn range, the clamp of an fp8 indexer key cache (a uint8 ``k_cache``). Keys are stored as
``clamp(key, +-448)`` in e4m3 with no scale: they are LayerNorm outputs, and the top-k is invariant to a positive scale.
The host's prefill writer (``layout/decode/gk_kv_cache.pack_keys_torch``) writes the same bytes.
"""


@dataclass
class IndexerScoreShape(nl.NKIObject):
    """Extents for the indexer score kernel: k projection, RoPE, cache write, score, head reduce."""

    num_tokens: int
    """Query tokens. One sequence's speculative tokens under the seq32 grid."""

    num_new: int
    """Tokens whose key is written to the cache this step. Equal to num_tokens in plain decode."""

    num_heads: int
    """Indexer query heads. 32 in GLM-5.2, and NOT sharded -- the head axis is reduced."""

    head_dim: int
    """Indexer head width. 128 in GLM-5.2."""

    rope_dim: int
    """Rotated width, the LEADING slice of head_dim. 64 in GLM-5.2, exactly half."""

    hidden: int
    """Model hidden width, the contraction of wk and weights_proj."""

    num_keys: int
    """Context positions THIS rank holds. One quarter of the context under the key4 grid."""

    key_base: int
    """This rank's first global key position. Ownership is contiguous, so local i is key_base + i."""

    dtype: nl.DType
    """Kernel compute dtype."""

    key_fp8: bool = False
    """
    The key cache is uint8: each byte is one float8_e4m3fn feature (``IDX_KEY_FP8_MAX``), half the bytes of bf16.

    Read off the cache dtype, so the same kernel serves both formats. The score loop's transposing key DMA needs a
    2-byte dtype, so it moves the cache as bf16 WORDS, ``[keys, head_dim / 2]``: word ``j`` of a key holds features
    ``2j`` and ``2j + 1``. A stride-2 slice of the transposed tile's fp8 view is then the even or the odd feature
    half, and the score is two ``head_dim / 2``-deep matmuls against the query's matching halves.
    """

    @property
    def num_key_tiles(self) -> int:
        """Partition tiles the local key range spans."""
        return div_ceil(self.num_keys, P_MAX)

    @property
    def key_words(self) -> int:
        """Partitions of the transposed key tile: ``head_dim`` bf16 features, or ``head_dim / 2`` words of fp8 pairs."""
        return self.head_dim // 2 if self.key_fp8 else self.head_dim

    @property
    def q_packed_dim(self) -> int:
        """Width of the complete query this kernel consumes: every head's head_dim concatenated."""
        return self.num_heads * self.head_dim

    @property
    def key_chunk(self) -> int:
        """Keys scored per score-loop iteration: one PSUM bank of the score destination (512 fp32 columns). A chunk spanning
        two banks measured both slower and wrong on trn3, although the ISA allows a 4096-wide fp32 destination, so keep the
        chunk to one bank and re-check accuracy, not only latency, before changing it.
        """
        return SCORE_KEY_CHUNK

    @property
    def num_key_chunks(self) -> int:
        """Score-matmul iterations."""
        return div_ceil(self.num_keys, self.key_chunk)

    @property
    def key_pair(self) -> int:
        """Keys covered by one residency-mask + eviction pass. See ``PAIR_CHUNKS``."""
        return PAIR_CHUNKS * self.key_chunk

    @property
    def num_key_pairs(self) -> int:
        """Iterations of the outer score loop, i.e. eviction count."""
        return div_ceil(self.num_keys, self.key_pair)

    @property
    def head_sum_block_tiles(self) -> int:
        """
        128-key tiles whose head sums share one PSUM bank, i.e. the eviction/mask/transpose granularity.

        ``HEAD_SUM_BANK_COLS // num_tokens`` tiles fit a bank; rounded down to whole score chunks so a
        block boundary is also a chunk boundary, and capped at the key range.
        """
        tiles_per_chunk = self.key_chunk // P_MAX
        fit = (HEAD_SUM_BANK_COLS // self.num_tokens) // tiles_per_chunk * tiles_per_chunk
        return max(tiles_per_chunk, min(fit, self.num_key_tiles))

    @property
    def head_sum_block_keys(self) -> int:
        """Keys per head-sum block. See ``head_sum_block_tiles``."""
        return self.head_sum_block_tiles * P_MAX

    @property
    def mask_cols(self) -> int:
        """Columns of the key-major residency mask: one ``num_tokens``-wide group per 128-key tile."""
        return self.num_key_tiles * self.num_tokens

    @property
    def key_load(self) -> int:
        """Keys fetched per DMA, decoupled from ``key_chunk``: a key load's cost is mostly its fixed per-instruction term, so
        the loop loads ``KEY_LOAD_MAX`` keys at a time and scores them chunk by chunk.
        """
        return min(self.num_keys, max(self.key_chunk, KEY_LOAD_MAX))

    @property
    def num_key_loads(self) -> int:
        """Key-cache DMAs the score loop issues."""
        return div_ceil(self.num_keys, self.key_load)

    @property
    def tokens_per_group(self) -> int:
        """
        Query tokens whose heads together fill one score matmul's output partitions.

        The score matmul emits ``dst[(token, head), key]``, so a matmul could cover
        ``P_MAX / num_heads`` whole tokens. At GLM-5.2's 32 heads that is 4, i.e. two matmuls cover all
        8 speculative tokens and the partition dim is fully used.

        Grouping is only taken when ``num_heads`` is a multiple of ``PARTITION_ALIGN``. Token ``i`` of
        a group owns partitions ``[i * num_heads, (i + 1) * num_heads)``, and an SBUF partition range
        must START at a multiple of 32 -- so for a ``num_heads`` that is not, the per-token operands
        the head sum needs (its weight column and its 0/1 selector block) are inexpressible. Falling
        back to one token per group is always legal, and costs only the small unit-test shapes: every
        production ``num_heads`` this kernel is built for is a multiple of 32.
        """
        return P_MAX // self.num_heads if self.num_heads % PARTITION_ALIGN == 0 else 1

    @property
    def num_token_groups(self) -> int:
        """Score matmuls per key chunk: the (token, head) axis split to fit ``P_MAX`` partitions."""
        return div_ceil(self.num_tokens, self.tokens_per_group)

    def group_tokens(self, group: int) -> int:
        """Tokens in token group ``group``. Only the last group can be short."""
        return min(self.tokens_per_group, self.num_tokens - group * self.tokens_per_group)

    def group_partitions(self, group: int) -> int:
        """Partitions token group ``group`` occupies, i.e. its ``(token, head)`` pair count."""
        return self.group_tokens(group) * self.num_heads


@dataclass
class IndexerTopkShape(nl.NKIObject):
    """Extents for a ``nisa.topk`` stage, local or merge."""

    num_tokens: int
    """Query tokens, one logical top-k row each."""

    num_candidates: int
    """Row width fed to nisa.topk: num_keys for the local stage, degree * topk for the merge."""

    topk: int
    """Selected count. 2048 in GLM-5.2, an exact budget rather than a padded one."""

    dtype: nl.DType
    """Score dtype. nisa.topk requires bfloat16, which is asserted."""

    @property
    def snake_partitions(self) -> int:
        """Partitions the whole batch of rows occupies: 16 per row."""
        return self.num_tokens * SNAKE_PARTITIONS

    @property
    def snake_columns(self) -> int:
        """Free width one row's snake needs, i.e. ceil(num_candidates / 16)."""
        return div_ceil(self.num_candidates, SNAKE_PARTITIONS)


def get_indexer_q_proj_shape(q_resid: nl.NkiTensor, wq_b: nl.NkiTensor) -> IndexerQProjShape:
    """Derive the query up-projection extents. The column-shard degree is implied by wq_b's width."""

    shapes = IndexerQProjShape(
        num_tokens=q_resid.shape[0],
        q_lora_rank=q_resid.shape[1],
        q_shard=wq_b.shape[1],
        dtype=q_resid.dtype,
    )

    kernel_assert(
        shapes.num_tokens <= P_MAX,
        f"num_tokens={shapes.num_tokens} must be <= {P_MAX}, one token per partition",
    )
    kernel_assert(
        wq_b.shape[0] == shapes.q_lora_rank,
        f"wq_b.shape[0]={wq_b.shape[0]} must equal q_lora_rank={shapes.q_lora_rank}; the column "
        f"shard splits wq_b's OUTPUT, so its contraction stays whole",
    )
    kernel_assert(
        div_ceil(shapes.q_shard, F_MAX) <= PSUM_BANKS,
        f"q_shard={shapes.q_shard} needs {div_ceil(shapes.q_shard, F_MAX)} PSUM banks, only {PSUM_BANKS} exist",
    )
    return shapes


def indexer_cache_is_blocked(k_cache, stacked: bool = False) -> bool:
    """Whether an indexer cache is in the block form rather than the legacy flat form.

    Block form is ``[n_blocks, 1, block_size, head_dim]``, MLA's ``kv_cache`` shape. Flat is the legacy
    ``[num_keys, head_dim]``: it keeps the contiguous shard and cannot take an interleaved deal.
    ``stacked`` drops a leading layer axis first. Works on any object with ``.shape``.
    """
    shape = tuple(k_cache.shape)[1 if stacked else 0 :]
    kernel_assert(
        len(shape) in (2, 4),
        f"indexer cache must be [num_keys, head_dim] or [n_blocks, 1, block_size, head_dim], got {shape}",
    )
    return len(shape) == 4


def indexer_cache_keys(k_cache, stacked: bool = False) -> int:
    """Keys in an indexer cache of either form (see ``indexer_cache_is_blocked``)."""
    shape = tuple(k_cache.shape)[1 if stacked else 0 :]
    return shape[0] * shape[2] if indexer_cache_is_blocked(k_cache, stacked) else shape[0]


def get_indexer_score_shape(
    q: nl.NkiTensor,
    hidden_states: nl.NkiTensor,
    wk: nl.NkiTensor,
    weights_proj: nl.NkiTensor,
    k_cache: nl.NkiTensor,
    mask_add: nl.NkiTensor,
    num_heads: int,
    rope_dim: int,
    key_base: int,
) -> IndexerScoreShape:
    """Derive the score extents. num_heads cannot be recovered: q flattens head and head_dim."""

    head_dim = wk.shape[1]

    shapes = IndexerScoreShape(
        num_tokens=q.shape[0],
        num_new=hidden_states.shape[0],
        num_heads=num_heads,
        head_dim=head_dim,
        rope_dim=rope_dim,
        hidden=hidden_states.shape[1],
        num_keys=indexer_cache_keys(k_cache),
        key_base=key_base,
        dtype=q.dtype,
        key_fp8=k_cache.dtype == nl.uint8,
    )

    kernel_assert(
        shapes.num_tokens <= P_MAX,
        f"num_tokens={shapes.num_tokens} must be <= {P_MAX}, one token per partition",
    )
    kernel_assert(
        k_cache.dtype in (q.dtype, nl.uint8),
        f"k_cache dtype {k_cache.dtype} must be the query's {q.dtype} or uint8 (fp8 e4m3 keys, IDX_KEY_FP8_MAX)",
    )
    kernel_assert(
        not shapes.key_fp8 or shapes.head_dim % 2 == 0,
        f"an fp8 key cache needs an even head_dim, got {shapes.head_dim}: the score moves it as 2-byte words",
    )
    kernel_assert(
        shapes.num_keys % P_MAX == 0,
        f"num_keys={shapes.num_keys} must be a multiple of {P_MAX}: the head sum is tiled by 128 keys",
    )
    kernel_assert(
        tuple(mask_add.shape) == (P_MAX, shapes.mask_cols),
        f"mask_add is {tuple(mask_add.shape)}, expected ({P_MAX}, {shapes.mask_cols}) for "
        f"{shapes.num_keys} keys x {shapes.num_tokens} tokens",
    )
    kernel_assert(
        shapes.head_dim <= P_MAX,
        f"head_dim={shapes.head_dim} must be <= {P_MAX}, it is the score's contraction",
    )
    kernel_assert(
        shapes.num_heads <= P_MAX,
        f"num_heads={shapes.num_heads} must be <= {P_MAX}; the score matmul emits one partition per "
        f"(token, head) pair, so a single head's worth must fit the partition dim",
    )
    kernel_assert(
        q.shape[0] == shapes.num_tokens and q.shape[1] == shapes.q_packed_dim,
        f"q must be (num_tokens={shapes.num_tokens}, q_packed_dim={shapes.q_packed_dim}), got {q.shape}; "
        f"this kernel runs AFTER the all-to-all, so every head must be present",
    )
    kernel_assert(
        (k_cache.shape[1] == 1 and k_cache.shape[3] == shapes.head_dim)
        if indexer_cache_is_blocked(k_cache)
        else k_cache.shape[1] == shapes.head_dim,
        f"k_cache must be [n_blocks, 1, block_size, head_dim={shapes.head_dim}] or the legacy "
        f"[num_keys, head_dim={shapes.head_dim}], got {tuple(k_cache.shape)}",
    )
    kernel_assert(
        weights_proj.shape[0] == shapes.hidden and weights_proj.shape[1] == shapes.num_heads,
        f"weights_proj must be (hidden={shapes.hidden}, num_heads={shapes.num_heads}), got {weights_proj.shape}",
    )
    kernel_assert(
        wk.shape[0] == shapes.hidden,
        f"wk.shape[0]={wk.shape[0]} must equal hidden={shapes.hidden}; wk is REPLICATED, because a "
        f"column shard would split the axis its LayerNorm reduces over and the score contracts over",
    )
    # The rotated slice LEADS the head, which is the opposite of the MLA path's [nope | pe]. The
    # reference cats [q_rot, q_pass], so a kernel that assumed the MLA order would rotate the wrong 64.
    kernel_assert(
        0 < shapes.rope_dim <= shapes.head_dim,
        f"rope_dim={shapes.rope_dim} must be in (0, head_dim={shapes.head_dim}]",
    )
    kernel_assert(
        shapes.rope_dim % 2 == 0,
        f"rope_dim={shapes.rope_dim} must be even; interleaved RoPE pairs 2i with 2i+1",
    )
    kernel_assert(shapes.key_base >= 0, f"key_base={shapes.key_base} must be non-negative")
    return shapes


def get_indexer_topk_shape(scores: nl.NkiTensor, topk: int) -> IndexerTopkShape:
    """Derive a top-k stage's extents from its score row and the selected count."""

    shapes = IndexerTopkShape(
        num_tokens=scores.shape[0],
        num_candidates=scores.shape[1],
        topk=topk,
        dtype=scores.dtype,
    )

    # nisa.topk asserts all of these itself; restating them here turns a compiler-time assert deep in
    # the ISA into a kernel-entry message that names the operand.
    kernel_assert(
        shapes.dtype == nl.bfloat16,
        f"nisa.topk requires a bfloat16 source, got {shapes.dtype}. Note the reference ranks in fp32, "
        f"so the bf16 key is a real ~0.25% selection divergence, not a lossless choice",
    )
    kernel_assert(
        SNAKE_PARTITIONS <= shapes.num_candidates < TOPK_MAX_N,
        f"num_candidates={shapes.num_candidates} must be in [{SNAKE_PARTITIONS}, {TOPK_MAX_N}); the "
        f"upper bound is what forces a key-shard degree of at least 4 at a 131072 context",
    )
    kernel_assert(
        1 <= shapes.topk <= shapes.num_candidates,
        f"topk={shapes.topk} must be in [1, num_candidates={shapes.num_candidates}]",
    )
    kernel_assert(
        shapes.snake_partitions <= P_MAX,
        f"num_tokens={shapes.num_tokens} needs {shapes.snake_partitions} partitions at "
        f"{SNAKE_PARTITIONS} per row, over the {P_MAX} available; nisa.topk fits 8 rows per issue",
    )
    return shapes
