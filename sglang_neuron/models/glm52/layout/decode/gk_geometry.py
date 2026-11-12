# SPDX-License-Identifier: Apache-2.0
"""Sharding geometry of the GLM-5.2 decode giga-kernel (GK).

The GK maps one rank set two ways at the same time. The attention half is
``DP x (cp_degree x head_shard x token_shard)``. The MoE half is ``DP x TP`` with
expert parallelism over every rank. The two labellings agree on the token slots
that each rank owns, so the stack chains with no permutation.

This module holds the derived shapes, the rank decomposition and every structural
tensor that the kernel takes: the block table, the two slot mappings, the
positions and the offsets. The kernel package (nkilib) owns the layouts; this
module owns the host side of the same contract. One place holds the geometry, so
the weight loader, the cache loader and the model cannot disagree.

Terms:
    ``T_shard``: token slots that a rank holds on entry and on exit.
    ``T_local``: token slots that a rank scores in the MLA core.
    ``key shard``: the keys that one rank's indexer holds. The shard is DEALT
        with MLA's rule: global 32-key block ``g`` sits on key rank
        ``g % cp_degree`` at local block ``g // cp_degree``.
    ``spec``: padded speculative slots for each sequence. Slot 0 is the only
        real token. The other slots are zero padding.

Example:
    >>> geo = Glm52GkGeometry(ranks=128, num_layers=78)
    >>> geo.t_local, geo.num_blocks, geo.idx_keys
    (8, 1024, 32768)
    >>> geo.coords(5)["cp"]
    1
"""

from __future__ import annotations



from dataclasses import dataclass, field, replace

import numpy as np

# The packed latent cache row, in bytes: bf16 k_pe and the fp8 latent. The kernel owns the layout;
# this re-exports its width.
from sglang_neuron.models.glm52.kernels.decode.attention.mla.kv_row_layout import (  # noqa: E501
    ROW_BYTES,
)
from sglang_neuron.models.glm52.layout.glm_kv_row import STATIC_SCALE


# RoPE base of the GLM-5.2 checkpoint. The kernel gets the tables, not the base,
# so this value must match the checkpoint's `rope_theta`.
ROPE_THETA = 8000000.0

# The three normalization epsilons of the reference model. `input_layernorm` and
# `post_attention_layernorm` share one value. The KV latent norm and the
# indexer's k_norm use the smaller one.
INPUT_NORM_EPS = 1e-5
LATENT_NORM_EPS = 1e-6
IDX_NORM_EPS = 1e-6

# Latent dequantization scale of the paged MLA cache, the static row's (`glm_kv_row.STATIC_SCALE`). `pack_kv_rows`
# divides by it when it writes a row, and the kernel multiplies by it on read. A mismatch is a pure gain error, so the
# prefill's writers, the cache builder and the kernel argument all take this one value.
KV_SCALE = STATIC_SCALE

# Keys that ONE `nisa.topk` ranks in the indexer's local stage (the kernel's
# ``IDX_TOPK_MAX_CALL_N``). A wider key shard is ranked in equal chunks of at most
# this width, and one more select of at most this width merges the chunk winners
# on the rank (`idx_topk_chunking`).
IDX_TOPK_CHUNK = 32768

# Rows of the snake load of one chunk: a chunk width is a multiple of it (the
# kernel's ``SNAKE_PARTITIONS``).
IDX_SNAKE_PARTITIONS = 16

# Keys in one tile of the indexer's score and residency kernels (the kernel's
# ``P_MAX``): the decode kernel asserts a shard of a multiple of it, so a chunked shard
# must be one.
IDX_KEY_TILE = 128

# Exclusive bound on the row width of one `nisa.topk` (the ISA field).
TOPK_MAX_N = 65536


def idx_topk_chunking(idx_keys: int, index_topk: int) -> tuple[int, int] | None:
    """Return the chunk count and width ``(C, W)`` of the indexer's local stage, or None.

    The host twin of the decode kernel's ``_pick_local_chunks``
    (``kernels/decode/attention/indexer/indexer_topk_local_carry_index.py``). Up to `IDX_TOPK_CHUNK` keys,
    one select ranks the shard: ``(1, idx_keys)``. A wider shard is ranked in ``C``
    equal chunks of ``W = idx_keys / C`` keys. ``C`` is the smallest power of two
    for which ``W`` is at most `IDX_TOPK_CHUNK` and a multiple of
    `IDX_SNAKE_PARTITIONS`, and for which the ``C * index_topk`` chunk winners fit one
    more select of at most `IDX_TOPK_CHUNK`. At ``index_topk`` 2048, ``C`` is at most
    16, so a shard holds at most 524,288 keys.

    Args:
        idx_keys: Keys of one indexer shard (`Glm52GkGeometry.idx_keys`).
        index_topk: Keys that the indexer keeps for each query token.

    Returns:
        ``(C, W)``, or None when no chunking tiles the shard.

    Example:
        >>> idx_topk_chunking(262144, 2048), idx_topk_chunking(258048, 2048)
        ((8, 32768), (8, 32256))
        >>> idx_topk_chunking(98304, 2048), idx_topk_chunking(557056, 2048)
        ((4, 24576), None)
    """
    if idx_keys <= IDX_TOPK_CHUNK:
        return 1, idx_keys
    for k in range(idx_keys.bit_length()):
        chunks = 1 << k
        width = idx_keys // chunks
        if (
            idx_keys % chunks == 0
            and width <= IDX_TOPK_CHUNK
            and width % IDX_SNAKE_PARTITIONS == 0
            and chunks * index_topk <= IDX_TOPK_CHUNK
        ):
            return chunks, width
    return None



_MESH_CACHE: dict = {}


@dataclass(frozen=True)
class Glm52GkGeometry:
    """One sharding and shape point of the GLM-5.2 decode giga-kernel.

    The defaults are the production configuration that the nkilib reference runs:
    128 ranks, ``DP2 x (cp4 x token16)`` attention, o_proj heads over blocks of
    16 ranks, and MoE ``DP16 x TP8`` with 2 experts for each rank.

    Attributes:
        ranks: Devices in the model replica. Also the expert-parallel degree.
        num_layers: Decoder layers in the stack.
        dp_size: Attention data-parallel replicas.
        cp_degree: Context-parallel degree. Shards keys.
        head_shard: Head-shard degree. 1 is pure data parallelism.
        tp_size: Tensor-parallel degree of the MoE shared expert.
        t_shard: Token slots that each rank owns on entry and on exit.
        spec: Padded speculative slots for each sequence.
        oproj_tp: Ranks that share the output projection heads. 0 selects the CP
            group.
        qproj_tp: Ranks that share the query up-projection heads. 0 selects the
            token group.
        index_share_period: 4 gives the shipped ``F s s s`` indexer pattern. 1
            makes every layer full. 0 makes every layer shared.
        index_skip_topk_offset: The checkpoint's value. 3 reproduces the shipped
            pattern, where layers 0, 1 and 2 are full.
        layer_offset: Model layer index of stack layer 0.
        first_dense_layers: Leading layers that use the dense MLP.
        context: Key positions for each sequence, this step included.
        idx_fp8: fp8 indexer keys with `ROW_BYTES` latent rows (bf16 k_pe)
            (`idx_fp8_keys`).
        latent_tp: Ranks that share the target's down-projection columns
            (``wq_a``, ``wkv_a``). 0 selects the token group. A multiple of the
            token group shards them over the LATENT group of `latent_tp`
            consecutive ranks: each rank projects the group's tokens, and one
            all_to_all hands each rank its own token slice's latents. The MTP
            layer keeps the token-group shard (`q_lora_shard`).
        router_tp: Column-shard the MoE router over the TP group: TP position
            ``p`` holds experts ``[p * E / tp_size, ...)``. It saves 7/8 of the
            router on each rank and adds two small TP collectives to each MoE
            layer. The kernel selects the path from the router's shape.
        dense_tp: Ranks that split the dense MLP's columns. 0 selects the TP
            group. A multiple of `tp_size` holds ``1 / dense_tp`` of the dense
            weights on each rank, and the dense layers gather and scatter
            ``t_shard * dense_tp`` rows over consecutive blocks of `dense_tp`
            ranks.
        shared_tp: Ranks that split the target's shared expert. 0 selects the
            TP group. A multiple of `tp_size` holds ``1 / shared_tp`` of it on
            each rank over ``t_shard * shared_tp`` rows. The MTP layer keeps
            the TP group (`shared_intermediate`, `tokens_replica`).
        rope_split: Keep the RoPE tables as two small tables (`rope_split`):
            512 KiB instead of 256 MiB at a 1M context, at a max error of
            1.4e-7 instead of 3.0e-8. False keeps the two full tables.
        moe_sparse_slots: The MoE stacks hold the sparse layers only
            (`num_moe_layers`), with no slot for the dense layers. False keeps a
            slot for every layer.

        idx_pool: Pool the indexer over CP groups, dealt by whole top-k chunks (the chunk-dealt pool, CDP,
            `layout.idx_pool.IdxChunkPool`): chunk ``q`` of the sequence in pool slot ``s`` sits on pool member
            ``(s + q) % E`` at the rows the unpooled indexer uses, so every chunk select sees the unpooled bytes,
            and the step is bound by the pool's live chunks, not the longest sequence. Needs one sequence per CP
            group, head_shard 1, and at least two groups per pool (`idx_pool_degree`).
        idx_pool_split: Pools per key position. 1 pools every CP group into one pool. ``s`` deals the CP
            groups round-robin into ``s`` pools of ``groups / s`` ranks (group ``g`` is in pool ``g % s``);
            the pool collectives then span fewer ranks. The served pool is ``idx_pool_split`` 4 at
            ``1m``: 8 groups per pool, one chunk per (holder, owner).
        block_size: Rows in one block of the paged latent cache.
        hidden: Hidden dimension.
        total_heads: Query heads before the head shard.
        q_lora_rank: Query down-projection rank.
        kv_lora_rank: KV latent rank.
        qk_nope_head_dim: Per-head QK width that RoPE does not rotate.
        qk_rope_head_dim: Per-head QK width that RoPE rotates.
        v_head_dim: Per-head value width.
        idx_n_heads: Indexer query heads.
        idx_head_dim: Indexer head width.
        idx_rope_dim: Indexer rotated width.
        index_topk: Keys that the indexer keeps for each query token.
        num_experts: Routed experts in the model.
        top_k: Routed experts for each token.
        routed_scaling_factor: The checkpoint's value.
        moe_intermediate: Routed-expert intermediate width.
        shared_intermediate_full: Shared-expert intermediate width before the TP
            column shard.
        dense_intermediate_full: Dense-MLP intermediate width before the TP
            column shard.
        rope_theta: RoPE base. The GLM-5.2 config omits it, and the reference
            model's own default applies.
        vocab_size: Rows of `lm_head`.
    """

    ranks: int = 128
    num_layers: int = 78
    dp_size: int = 2
    cp_degree: int = 4
    head_shard: int = 1
    tp_size: int = 8
    t_shard: int = 2
    spec: int = 8
    oproj_tp: int = 16
    qproj_tp: int = 0
    latent_tp: int = 0
    index_share_period: int = 4
    index_skip_topk_offset: int = 3
    layer_offset: int = 0
    first_dense_layers: int = 3
    context: int = 131072
    idx_fp8: bool = False
    router_tp: bool = False
    dense_tp: int = 0
    shared_tp: int = 0
    rope_split: bool = False
    moe_sparse_slots: bool = False
    idx_pool: bool = False
    idx_pool_split: int = 1
    block_size: int = 32
    hidden: int = 6144
    total_heads: int = 64
    q_lora_rank: int = 2048
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256
    idx_n_heads: int = 32
    idx_head_dim: int = 128
    idx_rope_dim: int = 64
    index_topk: int = 2048
    rope_theta: float = ROPE_THETA
    num_experts: int = 256
    top_k: int = 8
    routed_scaling_factor: float = 2.5
    moe_intermediate: int = 2048
    shared_intermediate_full: int = 2048
    dense_intermediate_full: int = 12288
    vocab_size: int = 154880

    # Attention factors.
    @property
    def attn_size(self) -> int:
        """Ranks in one DP group."""
        return self.ranks // self.dp_size

    @property
    def token_group(self) -> int:
        """Ranks that share one token slice: ``cp_degree * head_shard``."""
        return self.cp_degree * self.head_shard

    @property
    def token_shard(self) -> int:
        """Token-shard degree, the third factor of `attn_size`."""
        return self.attn_size // self.token_group

    @property
    def tokens(self) -> int:
        """Token slots in one DP group."""
        return self.t_shard * self.attn_size

    @property
    def t_local(self) -> int:
        """Token slots that one rank scores in the MLA core."""
        return self.t_shard * self.token_group

    @property
    def num_seqs(self) -> int:
        """Sequences in one DP group."""
        return self.tokens // self.spec

    @property
    def num_seqs_local(self) -> int:
        """Sequences in one rank's token slice."""
        return self.t_local // self.spec

    @property
    def key_size(self) -> int:
        """Indexer key-shard degree."""
        return self.token_group // self.num_seqs_local

    @property
    def num_heads(self) -> int:
        """Query heads that one rank attends over."""
        return self.total_heads // self.head_shard

    @property
    def qk_head_dim(self) -> int:
        """One head's packed QK width."""
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def softmax_scale(self) -> float:
        """``qk_head_dim ** -0.5``. GLM-5.2 ships ``rope_type="default"``, so no
        mscale factor applies."""
        return self.qk_head_dim**-0.5

    @property
    def oproj_degree(self) -> int:
        """Ranks that share the output-projection heads."""
        return self.oproj_tp or self.cp_degree

    @property
    def oproj_slices(self) -> int:
        """Token slices in one o_proj block group."""
        return self.oproj_degree // self.cp_degree

    @property
    def qproj_degree(self) -> int:
        """Ranks that share the query up-projection heads."""
        return self.qproj_tp or self.token_group

    @property
    def qproj_slices(self) -> int:
        """Token slices in one query-path block group."""
        return self.qproj_degree // self.token_group

    @property
    def nh_out(self) -> int:
        """Query heads that one rank projects through `o_proj` and `W_UV`."""
        return self.num_heads // self.oproj_degree

    @property
    def wq_b_shard(self) -> int:
        """Columns of `wq_b` that one rank holds: whole heads of the query group (`validate`)."""
        return self.num_heads * self.qk_head_dim // self.qproj_degree

    @property
    def q_absorb_heads(self) -> int:
        """Heads of `q_absorb_w` that one rank holds."""
        return self.num_heads // self.qproj_degree

    @property
    def latent_degree(self) -> int:
        """Ranks that share the target's down-projection columns."""
        return self.latent_tp or self.token_group

    @property
    def latent_slices(self) -> int:
        """Token slices in one latent group."""
        return self.latent_degree // self.token_group

    @property
    def wq_a_shard(self) -> int:
        """Columns of the target's `wq_a` that one rank holds (over the latent group)."""
        return self.q_lora_rank // self.latent_degree

    @property
    def wkv_a_shard(self) -> int:
        """Columns of the target's `wkv_a` that one rank holds (over the latent group)."""
        return self.kv_row_dim // self.latent_degree

    @property
    def kv_row_dim(self) -> int:
        """One logical cache row: ``cat([k_pe, kv_latent])``."""
        return self.qk_rope_head_dim + self.kv_lora_rank

    @property
    def row_bytes(self) -> int:
        """Bytes of one packed latent cache row (`ROW_BYTES`, the decode kernel's static row)."""
        return ROW_BYTES

    @property
    def idx_fp8_keys(self) -> bool:
        """Whether the indexer keys are fp8 bytes (`idx_fp8`)."""
        return self.idx_fp8

    @property
    def idx_key_bytes(self) -> int:
        """Bytes of one indexer key: ``idx_head_dim`` bf16 values, or fp8 values (`idx_fp8_keys`)."""
        return self.idx_head_dim * (1 if self.idx_fp8_keys else 2)

    @property
    def q_lora_shard(self) -> int:
        """One rank's columns of the query down-projection."""
        return self.q_lora_rank // self.token_group

    @property
    def kv_row_shard(self) -> int:
        """One rank's columns of the KV down-projection."""
        return self.kv_row_dim // self.token_group

    @property
    def idx_q_width(self) -> int:
        """The complete indexer query width."""
        return self.idx_n_heads * self.idx_head_dim

    @property
    def idx_q_shard(self) -> int:
        """One rank's columns of the indexer query up-projection."""
        return self.idx_q_width // self.key_size

    @property
    def idx_keys(self) -> int:
        """Key positions that one indexer rank holds, dealt over the CP group (see `idx_global_positions`)."""
        return self.context // self.key_size

    @property
    def idx_topk_chunks(self) -> int:
        """Chunks of the indexer's local stage (`idx_topk_chunking`): 1 up to `IDX_TOPK_CHUNK` keys.

        Raises:
            ValueError: If no chunking tiles the shard.
        """
        chunking = idx_topk_chunking(self.idx_keys, self.index_topk)
        if chunking is None:
            raise ValueError(
                f"no chunking of the indexer's local stage tiles {self.idx_keys} keys "
                f"at index_topk {self.index_topk} (idx_topk_chunking)"
            )
        return chunking[0]

    @property
    def prior_len(self) -> int:
        """First position of this step in the kernel's coordinate system."""
        return self.context - self.spec

    @property
    def local_blocks_per_seq(self) -> int:
        """Blocks of one sequence that one CP rank owns."""
        return self.context // self.block_size // self.cp_degree

    @property
    def num_blocks(self) -> int:
        """Blocks in one rank's paged latent cache: every block of each of its sequences' context (the block
        table's columns past the guard)."""
        return self.num_seqs_local * self.local_blocks_per_seq

    @property
    def table_width(self) -> int:
        """Columns of `block_table`: ``local_blocks_per_seq + 1``.

        Column 0 is never a real block. The CP position shift maps an owned
        global block ``b = cp + k * cp_degree`` to local column ``k + 1``, and
        column 0 holds -1, which the kernel reads as the non-resident sentinel.
        """
        return self.local_blocks_per_seq + 1

    # Indexer paging.
    @property
    def idx_shard_degree(self) -> int:
        """Degree the indexer's keys are DEALT over: the CP degree.

        The indexer shares MLA's block table, so its keys take MLA's deal: global block ``g`` sits
        on key rank ``g % cp_degree`` at local block ``g // cp_degree``. One table then serves both
        caches, a pool block fans out identically for both, and a KV connector places both regions
        with one rule. This needs the key-shard degree to BE the CP degree; the kernel asserts the
        same geometry (``idx_block_size == block_size``).
        """
        if self.key_size != self.cp_degree:
            raise ValueError(
                f"the shared indexer block table deals keys over the CP group, so key_size "
                f"({self.key_size}) must equal cp_degree ({self.cp_degree})"
            )
        return self.cp_degree

    @property
    def idx_pool_degree(self) -> int:
        """E, the CP groups of one pool (`idx_pool`), or 0 when off.

        Raises:
            ValueError: If the geometry cannot pool: more than one sequence per CP group or a head shard, a split
                that does not divide the CP groups, a pool of one group (the kernel's pool collectives need two),
                or more top-k chunks per sequence than pool members (a holder sends each owner one chunk).
        """
        if not self.idx_pool:
            return 0
        if self.num_seqs_local != 1 or self.head_shard != 1:
            raise ValueError(
                f"idx_pool needs one sequence per CP group and head_shard 1, got num_seqs_local "
                f"{self.num_seqs_local} and head_shard {self.head_shard}"
            )
        groups = self.dp_size * self.token_shard
        if self.idx_pool_split < 1 or groups % self.idx_pool_split:
            raise ValueError(
                f"idx_pool_split {self.idx_pool_split} must divide the {groups} CP groups"
            )
        degree = groups // self.idx_pool_split
        if degree < 2:
            raise ValueError(
                f"idx_pool_split {self.idx_pool_split} leaves one CP group per pool; the pool's collectives need "
                "two or more (lower idx_pool_split)"
            )
        if self.idx_topk_chunks > degree:
            raise ValueError(
                f"the pooled indexer puts at most one top-k chunk of each sequence on a pool member: "
                f"{self.idx_topk_chunks} chunks over {degree} members; raise the pool (lower idx_pool_split)"
            )
        return degree

    @property
    def idx_pool_tokens(self) -> int:
        """Query tokens the pooled indexer scores per sequence: every spec row, which the kernel's chunk-dealt pool
        requires, so one geometry serves the decode (one real row) and the MTP draft (all eight)."""
        return self.spec

    @property
    def idx_pool_chunk(self) -> bool:
        """Whether the indexer is pooled (the pool deals whole top-k chunks)."""
        return bool(self.idx_pool_degree)

    @property
    def idx_pool_chunk_keys(self) -> int:
        """Keys of one CDP region: one top-k chunk of the local stage (`idx_topk_chunking`), 32,768 at 1M.

        Region ``r`` of a rank's indexer cache is rows ``[r * idx_pool_chunk_keys, (r + 1) * idx_pool_chunk_keys)``.
        """
        return self.idx_keys // self.idx_topk_chunks

    def idx_global_positions(self, key_rank: int) -> np.ndarray:
        """``[idx_keys]`` int64 global position of every LOCAL indexer key of one key rank.

        Host twin of the kernel's position arithmetic (nkilib ``indexer_global_positions_np``): local
        key ``i`` sits in global block ``cp_degree * (i // bs) + key_rank`` at offset ``i % bs``.
        """
        i = np.arange(self.idx_keys, dtype=np.int64)
        bs = self.idx_block_size
        return (self.idx_shard_degree * (i // bs) + key_rank) * bs + (i % bs)

    def idx_local_index(self, position: int, key_rank: int) -> int:
        """LOCAL indexer key index of ``position`` on key rank ``key_rank``, or -1 if it does not own it."""
        bs = self.idx_block_size
        g, intra = divmod(int(position), bs)
        if g % self.idx_shard_degree != key_rank:
            return -1
        local = (g // self.idx_shard_degree) * bs + intra
        return local if 0 <= local < self.idx_keys else -1

    @property
    def idx_block_size(self) -> int:
        """Rows in one block of the PAGED indexer key cache.

        This is `block_size`, the latent cache's block, and not the engine's
        `--block-size`. The value makes `idx_blocks` equal `num_blocks`, so both
        caches present one block count to a KV connector whose descriptor id
        math (``region * num_blocks + block``) carries a single count.
        """
        return self.block_size

    @property
    def idx_blocks(self) -> int:
        """Blocks in one rank's paged indexer key cache."""
        return self.idx_keys // self.idx_block_size

    def refuse_unpageable_indexer(self) -> None:
        """Refuse a geometry whose two caches cannot share one block count.

        A KV connector carries ONE block count for every region it registers.
        The latent cache holds `num_blocks` blocks and the indexer cache holds
        `idx_blocks`, and the two agree only when `head_shard` equals
        `num_seqs_local` (then ``key_size == cp_degree``). The kernel also needs
        a power-of-two block that divides `idx_keys`.

        Raises:
            ValueError: If the counts differ, or if the kernel cannot page the
                indexer cache at `idx_block_size`.
        """
        if self.idx_keys % self.idx_block_size != 0:
            raise ValueError(
                f"the indexer shard holds {self.idx_keys} keys and the block is "
                f"{self.idx_block_size}, which does not divide it. Choose a "
                "context length that is a multiple of block_size * key_size."
            )
        if self.idx_block_size & (self.idx_block_size - 1):
            raise ValueError(
                f"the indexer block is {self.idx_block_size} and the kernel needs "
                "a power of two. Set block_size to a power of two."
            )
        if self.idx_blocks != self.num_blocks:
            raise ValueError(
                f"the latent cache holds {self.num_blocks} blocks and the paged "
                f"indexer cache would hold {self.idx_blocks}. A KV connector "
                "carries one block count for every region, so the two must "
                f"agree. This needs head_shard ({self.head_shard}) to equal "
                f"num_seqs_local ({self.num_seqs_local})."
            )

    # MoE factors.
    @property
    def experts_per_rank(self) -> int:
        """Routed experts local to one rank."""
        return self.num_experts // self.ranks

    @property
    def router_columns(self) -> int:
        """Router columns (experts) that one rank holds: all, or one TP block under `router_tp`."""
        return self.num_experts // self.tp_size if self.router_tp else self.num_experts

    @property
    def shared_intermediate(self) -> int:
        """Shared-expert intermediate width after the TP column shard."""
        return self.shared_intermediate_full // self.tp_size

    @property
    def dense_degree(self) -> int:
        """Ranks that split the dense MLP: `dense_tp`, or the TP group."""
        return self.dense_tp or self.tp_size

    @property
    def dense_intermediate(self) -> int:
        """Dense-MLP intermediate width after the column shard over `dense_degree` ranks."""
        return self.dense_intermediate_full // self.dense_degree

    @property
    def tokens_replica(self) -> int:
        """Token rows of one TP replica's MoE block: ``t_shard * tp_size``."""
        return self.t_shard * self.tp_size

    @property
    def shared_degree(self) -> int:
        """Ranks that split the target's shared expert: `shared_tp`, or the TP group."""
        return self.shared_tp or self.tp_size

    @property
    def shared_width(self) -> int:
        """The target's shared-expert width on one rank."""
        return self.shared_intermediate_full // self.shared_degree

    @property
    def shared_rows(self) -> int:
        """Token rows of the target's shared expert: ``t_shard * shared_degree``."""
        return self.t_shard * self.shared_degree

    # Layer roles.
    def is_full(self, layer: int) -> bool:
        """Whether stack layer `layer` runs its own indexer.

        This is the reference model's rule,
        ``max(model_layer - index_skip_topk_offset + 1, 0) % index_topk_freq == 0``,
        which is NOT uniform at the head of the model: at the shipped
        ``freq = 4, offset = 3`` it makes layers 0, 1 and 2 full, layers 3, 4 and
        5 shared, and only from layer 6 does the ``F s s s`` group start.

        The rule is defined on the MODEL layer index, so `layer_offset` moves the
        phase. The expression repeats the kernel's `_is_full_layer`; the two must agree for every layer of the stack.
        """
        if self.index_share_period <= 0:
            return False
        model_layer = layer + self.layer_offset
        if self.index_skip_topk_offset == 0:
            return model_layer % self.index_share_period == 0
        return (
            max(model_layer - self.index_skip_topk_offset + 1, 0)
            % self.index_share_period
            == 0
        )

    @property
    def full_layers(self) -> tuple[int, ...]:
        """Stack indices that run their own indexer, in order."""
        return tuple(i for i in range(self.num_layers) if self.is_full(i))

    @property
    def num_full_layers(self) -> int:
        """Leading axis of every indexer stack. At least 1, because a stack with
        no full layer still passes one unread entry."""
        return max(len(self.full_layers), 1)

    @property
    def num_dense_layers(self) -> int:
        """Leading axis of every dense-MLP stack, by the same rule."""
        return max(self.first_dense_layers, 1)

    @property
    def num_moe_layers(self) -> int:
        """Leading axis of every MoE stack: one slot a layer, or the sparse layers only.

        A dense layer reads no router, routed-expert or shared-expert weight, so
        with `moe_sparse_slots` the store keeps no slot for it (at least 1). The
        kernel takes both forms and picks one from the stack's shape.
        """
        if not self.moe_sparse_slots:
            return self.num_layers
        return max(self.num_layers - min(self.first_dense_layers, self.num_layers), 1)

    def moe_slot(self, layer: int) -> int:
        """Index of sparse stack layer `layer` in the MoE stacks."""
        if not self.moe_sparse_slots:
            return layer
        return layer - min(self.first_dense_layers, self.num_layers)

    def is_dense(self, layer: int) -> bool:
        """Whether stack layer `layer` uses the dense MLP."""
        return layer < self.first_dense_layers

    # Rank decomposition.
    def coords(self, rank_id: int) -> dict[str, int]:
        """Decompose a global rank into its attention and indexer coordinates.

        The digit order ``i = token_group * kk + head_shard * cp + hh`` is what
        makes the exit's two scatters land on global slots
        ``[t_shard * i, t_shard * (i + 1))``.

        Args:
            rank_id: Global rank in ``[0, ranks)``.

        Returns:
            The keys ``dp``, ``i``, ``kk``, ``cp``, ``hh``, ``pos_in_group``,
            ``seq_local``, ``key_rank`` and ``seq_global``.
        """
        dp, i = divmod(rank_id, self.attn_size)
        kk, r = divmod(i, self.token_group)
        cp, hh = divmod(r, self.head_shard)
        pos = cp * self.head_shard + hh
        seq_local, key_rank = divmod(pos, self.key_size)
        return {
            "dp": dp,
            "i": i,
            "kk": kk,
            "cp": cp,
            "hh": hh,
            "pos_in_group": pos,
            "seq_local": seq_local,
            "key_rank": key_rank,
            "seq_global": self.num_seqs_local * kk + seq_local,
        }

    def slot_of(self, rank_id: int) -> int:
        """The decode slot (sequence) a global rank serves: its DP group's first slot plus its sequence in the group."""
        c = self.coords(rank_id)
        return c["dp"] * self.num_seqs + c["seq_global"]

    def q_group_position(self, rank_id: int) -> int:
        """Query-path group position of a rank.

        This selects the rank's `wq_b` column block and its `q_absorb_w` heads:
        the token-group position, or the block position under `qproj_tp`.
        """
        c = self.coords(rank_id)
        return self.token_group * (c["kk"] % self.qproj_slices) + c["pos_in_group"]

    def latent_group_position(self, rank_id: int) -> int:
        """Latent-group position of a rank: its `wq_a` / `wkv_a` column block.

        The token-group position, or ``token_group * (kk mod latent_slices) + pos`` under `latent_tp`.
        """
        c = self.coords(rank_id)
        return self.token_group * (c["kk"] % self.latent_slices) + c["pos_in_group"]

    def out_group_position(self, rank_id: int) -> int:
        """Output-path group position of a rank.

        This selects the rank's `o_proj_w` rows and `out_absorb_w` heads: the CP
        position, or the block position under `oproj_tp`.
        """
        c = self.coords(rank_id)
        return self.cp_degree * (c["kk"] % self.oproj_slices) + c["cp"]

    # Replica meshes.
    def replica_meshes(self) -> dict[str, tuple[tuple[int, ...], ...]]:
        """`_replica_meshes`, computed once for each geometry (the geometry is frozen). The KV intake calls it
        many times for each arriving request; each build costs host time in Python. Returns a new dict each time."""
        hit = _MESH_CACHE.get(self)
        if hit is None:
            hit = self._replica_meshes()
            _MESH_CACHE[self] = hit
        return dict(hit)

    def _replica_meshes(self) -> dict[str, tuple[tuple[int, ...], ...]]:
        """The nine collective meshes, as tuples of tuples of global ranks.

        A group's POSITION order is its semantics: for ``token``, ``cp``,
        ``head`` and ``block`` it decides which rows a gather or a scatter gives
        a rank. For ``key`` and ``seq`` it decides which columns or sequences a
        block holds. ``token``, ``head`` and ``block`` must hold consecutive
        ranks, because the runtime's ``all_to_all`` refuses a strided group.

        Returns:
            The keys ``token``, ``cp``, ``pool``, ``head``, ``block``, ``qblock``,
            ``seq``, ``key``, ``ep``, ``tp``, ``latent`` and ``dense``. ``pool`` is present only when the
            indexer is pooled (`idx_pool_degree` > 0): the pool of each key position, one rank per CP group,
            in group order.
            ``latent`` equals ``token`` unless `latent_tp` shards the
            down-projections over more than the token group. ``dense`` and
            ``shared`` equal ``tp`` unless `dense_tp` or `shared_tp` is wider.
        """
        cp, h, k = self.cp_degree, self.head_shard, self.token_shard
        sl, qsl, lsl = self.oproj_slices, self.qproj_slices, self.latent_slices
        tg, ns, ks = self.token_group, self.num_seqs_local, self.key_size
        base = [d * self.attn_size for d in range(self.dp_size)]

        def lin(b: int, kk: int, c: int, hh: int) -> int:
            return b + tg * kk + h * c + hh

        token = tuple(
            tuple(lin(b, kk, c, hh) for c in range(cp) for hh in range(h))
            for b in base
            for kk in range(k)
        )
        cp_mesh = tuple(
            tuple(lin(b, kk, c, hh) for c in range(cp))
            for b in base
            for kk in range(k)
            for hh in range(h)
        )
        head = tuple(
            tuple(lin(b, kk, c, hh) for hh in range(h))
            for b in base
            for kk in range(k)
            for c in range(cp)
        )
        block = tuple(
            tuple(
                lin(b, q * sl + p, c, hh)
                for p in range(sl)
                for c in range(cp)
                for hh in range(h)
            )
            for b in base
            for q in range(k // sl)
        )
        qblock = tuple(
            tuple(
                lin(b, q * qsl + p, c, hh)
                for p in range(qsl)
                for c in range(cp)
                for hh in range(h)
            )
            for b in base
            for q in range(k // qsl)
        )
        latent = tuple(
            tuple(
                lin(b, q * lsl + p, c, hh)
                for p in range(lsl)
                for c in range(cp)
                for hh in range(h)
            )
            for b in base
            for q in range(k // lsl)
        )
        seq = tuple(
            tuple(lin(b, kk, *divmod(s * ks + j, h)) for s in range(ns))
            for b in base
            for kk in range(k)
            for j in range(ks)
        )
        key = tuple(
            tuple(lin(b, kk, *divmod(s * ks + j, h)) for j in range(ks))
            for b in base
            for kk in range(k)
            for s in range(ns)
        )
        # MoE: expert parallelism over every rank, and TP over each contiguous
        # block of `tp_size` ranks.
        ep = (tuple(range(self.ranks)),)
        tp = tuple(
            tuple(range(t * self.tp_size, (t + 1) * self.tp_size))
            for t in range(self.ranks // self.tp_size)
        )
        dd, sd = self.dense_degree, self.shared_degree
        dense = tuple(
            tuple(range(t * dd, (t + 1) * dd)) for t in range(self.ranks // dd)
        )
        shared = tuple(
            tuple(range(t * sd, (t + 1) * sd)) for t in range(self.ranks // sd)
        )
        # The indexer POOL: for key position c and pool q, the rank at c of every CP group g with g % split == q,
        # in group (= slot) order. A strided group of 16 ranks, which the runtime's all_gather and all_to_all
        # accept. The mesh all_to_all refuses a 32-rank strided group
        # ("alltoall cannot be supported without Mesh algorithm ... rank_n(32)"), so the 32-group point splits.
        split = self.idx_pool_split
        pool = tuple(
            tuple(group[c] for g, group in enumerate(cp_mesh) if g % split == q)
            for c in range(cp)
            for q in range(split)
        )
        meshes = {
            "token": token,
            "cp": cp_mesh,
            "head": head,
            "block": block,
            "qblock": qblock,
            "seq": seq,
            "key": key,
            "ep": ep,
            "tp": tp,
            "latent": latent,
            "dense": dense,
            "shared": shared,
        }
        if self.idx_pool_degree:
            meshes["pool"] = pool
        return meshes

    # Structural tensors.
    def block_table(self) -> np.ndarray:
        """``[num_seqs_local, table_width]`` int32 identity table with the guard.

        Column 0 holds -1. Column ``k + 1`` holds the physical block of this
        rank's local logical block ``k``.
        """
        table = np.full((self.num_seqs_local, self.table_width), -1, dtype=np.int32)
        table[:, 1:] = np.arange(self.num_blocks, dtype=np.int32).reshape(
            self.num_seqs_local, self.local_blocks_per_seq
        )
        return table

    def cp_shift(self, rank_id: int) -> int:
        """Position shift that lets the kernel run at ``cp_rank = 0``."""
        return (self.cp_degree - self.coords(rank_id)["cp"]) * self.block_size

    def positions(self, rank_id: int) -> np.ndarray:
        """``[t_local]`` int32 CP-shifted positions of this step's token slots.

        This value is NOT advanced for each decode step. It feeds the causal keep
        test ``key_pos <= positions[tok]`` and the CP owner test of the sibling
        keys, and already sits at the compiled ``prior_len``, which orders the
        query after every key that a run can write.
        """
        seq_positions = np.arange(
            self.prior_len, self.prior_len + self.spec, dtype=np.int32
        )
        return (
            np.tile(seq_positions, self.num_seqs_local) + self.cp_shift(rank_id)
        ).astype(np.int32)

    def kv_slot_mapping(self, rank_id: int, position: int) -> np.ndarray:
        """``[num_seqs_local, spec]`` int32 cache row of this step's one real key.

        Only sequence 0 slot 0 is a real token, so every other entry is -1 and
        the kernel skips it. A mapped padding slot would write a zero key at
        ``position + 1 ...``, and the next step's residency threshold would then
        score it.

        Args:
            rank_id: Global rank.
            position: Global position of this step's token.
        """
        slots = np.full((self.num_seqs_local, self.spec), -1, dtype=np.int32)
        cp = self.coords(rank_id)["cp"]
        block_g = position // self.block_size
        if block_g % self.cp_degree == cp:
            block_l = block_g // self.cp_degree
            if block_l < self.local_blocks_per_seq:
                # sequence 0's local block k is physical block k of the identity table (`block_table`)
                slots[0, 0] = block_l * self.block_size + position % self.block_size
        return slots

    def idx_slot_mapping(self, rank_id: int, position: int) -> np.ndarray:
        """``[spec]`` int32 indexer-cache row of this step's one real key.

        The row is the dealt local index of `idx_local_index`, and -1 on a rank
        that does not own the position.
        """
        slots = np.full((self.spec,), -1, dtype=np.int32)
        local = self.idx_local_index(position, self.coords(rank_id)["key_rank"])
        if local >= 0:
            slots[0] = local
        return slots

    def idx_positions(self, threshold: int) -> np.ndarray:
        """``[spec]`` int32 indexer residency threshold, a TRUE position.

        `threshold` is the cut in GLOBAL coordinates. At step ``s`` of a decode
        run it is ``pos0 + s``, which makes every key that an earlier step wrote
        live. The kernel compares TRUE positions under the deal; the rank term
        rides `key_offset`.
        """
        return np.full((self.spec,), threshold, dtype=np.int32)

    def key_offset(self, rank_id: int) -> np.ndarray:
        """``[1, 1]`` int32 rank term of the local-to-global key position map: ``key_rank * idx_block_size``."""
        return np.array(
            [[self.coords(rank_id)["key_rank"] * self.idx_block_size]], dtype=np.int32
        )

    def cp_pos_offset(self, rank_id: int) -> np.ndarray:
        """``[1, 1]`` int32 CP position shift."""
        return np.array([[self.cp_shift(rank_id)]], dtype=np.int32)

    def rope_tables(self, position: int) -> tuple[np.ndarray, np.ndarray]:
        """Interleaved RoPE cosine and sine rows at one TRUE position.

        The prior keys in both caches carry their own rotation, so the query's
        table must sit at this step's own position. A stale table rotates the
        query against the keys by the step count, which lands on both the MLA
        logits and the indexer scores.

        The angle is computed in float64. One position on the fastest dimension
        is one radian, so float32 accumulation of the exponent is visible.

        Returns:
            Two ``[1, qk_rope_head_dim // 2]`` float32 rows.
        """
        half = self.qk_rope_head_dim // 2
        ang = position * self.rope_theta ** (-np.arange(half, dtype=np.float64) / half)
        cos = np.cos(ang).reshape(1, half).astype(np.float32)
        sin = np.sin(ang).reshape(1, half).astype(np.float32)
        return cos, sin

    def scalar_kwargs(self) -> dict[str, object]:
        """The kernel's rank-invariant scalar arguments.

        Every value here changes the traced graph, so the compile and the run
        must agree on all of them.
        """
        return {
            "num_layers": self.num_layers,
            "index_share_period": self.index_share_period,
            "cp_degree": self.cp_degree,
            "head_shard": self.head_shard,
            "token_shard": self.token_shard,
            "key_size": self.key_size,
            "num_heads": self.num_heads,
            "qk_nope_head_dim": self.qk_nope_head_dim,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "kv_lora_rank": self.kv_lora_rank,
            "idx_n_heads": self.idx_n_heads,
            "idx_rope_dim": self.idx_rope_dim,
            "index_topk": self.index_topk,
            "spec": self.spec,
            "softmax_scale": self.softmax_scale,
            "kv_scale": KV_SCALE,
            "rms_norm_eps": INPUT_NORM_EPS,
            "latent_norm_eps": LATENT_NORM_EPS,
            "idx_norm_eps": IDX_NORM_EPS,
            "ep_size": self.ranks,
            "tp_size": self.tp_size,
            "top_k": self.top_k,
            "routed_scaling_factor": self.routed_scaling_factor,
            "index_skip_topk_offset": self.index_skip_topk_offset,
            "idx_pool_degree": self.idx_pool_degree,
            "idx_pool_tokens": self.idx_pool_tokens if self.idx_pool else 0,
            "layer_offset": self.layer_offset,
            "first_dense_layers": self.first_dense_layers,
        }

    def validate(self) -> None:
        """Check the factors that the kernel asserts at trace time.

        Raises:
            ValueError: If a factor does not divide. The message names the
                factor and gives the value to correct.
        """
        if self.ranks % self.dp_size:
            raise ValueError(
                f"dp_size {self.dp_size} must divide ranks {self.ranks}. "
                "Set dp_size to a divisor of the rank count."
            )
        if self.attn_size % self.token_group:
            raise ValueError(
                f"token_group {self.token_group} must divide the DP group "
                f"{self.attn_size}. Lower cp_degree or head_shard."
            )
        if self.t_local % self.spec:
            raise ValueError(
                f"spec {self.spec} must divide t_local {self.t_local}. "
                "Raise t_shard, or lower cp_degree."
            )
        if self.oproj_degree % self.cp_degree:
            raise ValueError(
                f"oproj_tp {self.oproj_tp} must be a multiple of cp_degree "
                f"{self.cp_degree}. Use 0 for the CP group."
            )
        if self.token_shard % max(self.oproj_slices, 1):
            raise ValueError(
                f"oproj_slices {self.oproj_slices} must divide token_shard "
                f"{self.token_shard}. Lower oproj_tp."
            )
        if (
            self.latent_degree % self.token_group
            or self.token_shard % self.latent_slices
        ):
            raise ValueError(
                f"latent_tp {self.latent_tp} must be a multiple of token_group "
                f"{self.token_group} whose slice count divides token_shard "
                f"{self.token_shard}. Use 0 for the token group."
            )
        if self.latent_slices > 1 and (
            self.head_shard != 1
            or self.latent_slices * self.t_local > 128
            or (self.qproj_slices > 1 and self.qproj_degree % self.latent_degree)
        ):
            raise ValueError(
                f"latent_tp {self.latent_tp} needs head_shard 1, at most 128 tokens in "
                "the latent group, and a query block group made of whole latent "
                "groups. Lower latent_tp."
            )
        if self.qproj_degree % self.token_group:
            raise ValueError(
                f"qproj_tp {self.qproj_tp} must be a multiple of token_group "
                f"{self.token_group}. Use 0 for the token group."
            )
        if self.token_shard % self.qproj_slices or self.num_heads % self.qproj_degree:
            # the query path's block groups tile the token slices, and each rank's wq_b columns are whole heads
            # (RoPE and the W_UK absorb index per head)
            raise ValueError(
                f"qproj_tp {self.qproj_tp} must give a slice count ({self.qproj_slices}) that divides token_shard "
                f"{self.token_shard} and divide the {self.num_heads} heads. Lower qproj_tp."
            )
        if self.ranks % self.tp_size:
            raise ValueError(f"tp_size {self.tp_size} must divide ranks {self.ranks}.")
        if self.num_experts % self.ranks:
            # every rank holds the same whole number of experts, at least one (EP over all the ranks)
            raise ValueError(
                f"ranks {self.ranks} must divide num_experts {self.num_experts}."
            )
        if (
            self.dense_degree % self.tp_size
            or self.ranks % self.dense_degree
            or self.dense_intermediate_full % (128 * self.dense_degree)
            or self.t_shard * self.dense_degree > 128
        ):
            raise ValueError(
                f"dense_tp {self.dense_tp} must be a multiple of tp_size {self.tp_size} "
                f"that divides ranks {self.ranks}, leaves a multiple of 128 of the "
                f"{self.dense_intermediate_full} dense columns on each rank, and gathers "
                "at most 128 rows. Use 0 for the TP group."
            )
        if (
            self.shared_degree % self.tp_size
            or self.ranks % self.shared_degree
            or self.shared_intermediate_full % (128 * self.shared_degree)
            or self.t_shard * self.shared_degree > 128
        ):
            raise ValueError(
                f"shared_tp {self.shared_tp} must be a multiple of tp_size {self.tp_size} "
                f"that divides ranks {self.ranks}, leaves a multiple of 128 of the "
                f"{self.shared_intermediate_full} shared-expert columns on each rank, and "
                "gathers at most 128 rows. Use 0 for the TP group."
            )
        if self.router_tp and (self.num_experts // self.tp_size) % 8:
            raise ValueError(
                f"router_tp gives each rank {self.num_experts} / {self.tp_size} router "
                "columns, and the router reads them in groups of 8. Set router_tp to "
                "False, or use a tp_size that gives a multiple of 8."
            )


@dataclass
class Glm52StepInputs:
    """The kernel inputs that change for each decode step, with this rank's entry residual.

    `x_shard` carries the new token's embedding on rank 0 slot 0 alone. Every
    other rank holds zeros, because only global token slot 0 is a real token.

    Attributes:
        x_shard: ``[t_shard, hidden]`` entry residual of this rank.
        cos: ``[t_local, rope / 2]`` float32 RoPE cosine.
        sin: ``[t_local, rope / 2]`` float32 RoPE sine.
        idx_cos: ``[spec, rope / 2]`` float32 indexer RoPE cosine.
        idx_sin: ``[spec, rope / 2]`` float32 indexer RoPE sine.
        kv_new_slot_mapping: ``[num_seqs_local, spec]`` int32 latent cache row.
        idx_new_slot_mapping: ``[spec]`` int32 indexer cache row.
        idx_positions: ``[spec]`` int32 residency threshold.
    """

    x_shard: np.ndarray
    cos: np.ndarray
    sin: np.ndarray
    idx_cos: np.ndarray
    idx_sin: np.ndarray
    kv_new_slot_mapping: np.ndarray
    idx_new_slot_mapping: np.ndarray
    idx_positions: np.ndarray
    names: tuple[str, ...] = field(
        default=(
            "x_shard",
            "cos",
            "sin",
            "idx_cos",
            "idx_sin",
            "kv_new_slot_mapping",
            "idx_new_slot_mapping",
            "idx_positions",
        ),
        repr=False,
    )

    def as_dict(self) -> dict[str, np.ndarray]:
        """Return the step tensors keyed by the kernel's parameter names."""
        return {name: getattr(self, name) for name in self.names}


def build_step_inputs(
    geo: Glm52GkGeometry,
    rank_id: int,
    position: int,
    entry: np.ndarray | None,
    threshold: int | None = None,
) -> Glm52StepInputs:
    """Build one rank's step-dependent kernel inputs.

    Args:
        geo: The sharding point.
        rank_id: Global rank.
        position: Global position of this step's token.
        entry: ``[hidden]`` embedding of the token, or None for a zero entry.
            Used on rank 0 only.
        threshold: Residency cut in global coordinates. Defaults to `position`,
            which makes every key that an earlier step wrote live.

    Returns:
        The step tensors, in numpy.

    Example:
        >>> geo = Glm52GkGeometry()
        >>> step = build_step_inputs(geo, 0, geo.prior_len, None)
        >>> step.cos.shape
        (8, 32)
    """
    x = np.zeros((geo.t_shard, geo.hidden), dtype=np.float32)
    if rank_id == 0 and entry is not None:
        x[0] = np.asarray(entry, dtype=np.float32).ravel()
    return Glm52StepInputs(x_shard=x, **step_tensors(geo, rank_id, position, threshold))


def step_tensors(
    geo: Glm52GkGeometry,
    rank_id: int,
    position: int,
    threshold: int | None = None,
) -> dict[str, np.ndarray]:
    """Build one rank's position-dependent step tensors: `build_step_inputs` without the entry residual.

    A served decode embeds its tokens in the graph, so each step needs these alone; leaving the residual out saves
    allocating a ``[t_shard, hidden]`` buffer every step.

    Args:
        geo: The sharding point.
        rank_id: Global rank.
        position: Global position of this step's token.
        threshold: Residency cut in global coordinates; defaults to `position`.

    Returns:
        The kernel's step tensors by parameter name, every `Glm52StepInputs` field but ``x_shard``.
    """
    cos, sin = geo.rope_tables(position)
    return {
        "cos": np.repeat(cos, geo.t_local, axis=0),
        "sin": np.repeat(sin, geo.t_local, axis=0),
        "idx_cos": np.repeat(cos, geo.spec, axis=0),
        "idx_sin": np.repeat(sin, geo.spec, axis=0),
        "kv_new_slot_mapping": geo.kv_slot_mapping(rank_id, position),
        "idx_new_slot_mapping": geo.idx_slot_mapping(rank_id, position),
        "idx_positions": geo.idx_positions(position if threshold is None else threshold),
    }


def mtp_query_geometry(geo: Glm52GkGeometry) -> Glm52GkGeometry:
    """Return the geometry of the MTP layer's query path: the token group's.

    The MTP draft chain (the decode kernel's ``mtp/`` fork) runs its draft steps (``draft_rows``)
    on the token-group query path: the block-sharded query path of a target ``qproj_tp``
    does not take them. So a target whose ``qproj_tp`` is wider than the token group gives
    the MTP layer ``qproj_tp=0``: ``wq_b`` and ``q_absorb_w`` over the token group
    (``models/glm52/layout/decode/weight_loaders_mtp.py``, ``models/glm52/mtp/draft.py``). Only
    ``qproj_tp`` changes, so only those two slices and `Glm52GkGeometry.q_group_position`
    differ; a geometry whose query path is already the token group keeps its own.

    Args:
        geo: The target sharding point.

    Returns:
        ``geo`` itself, or ``geo`` with ``qproj_tp=0``.

    Example:
        >>> mtp_query_geometry(Glm52GkGeometry(oproj_tp=64, qproj_tp=64)).wq_b_shard
        4096
    """
    if geo.qproj_slices > 1:
        return replace(geo, qproj_tp=0)
    return geo
