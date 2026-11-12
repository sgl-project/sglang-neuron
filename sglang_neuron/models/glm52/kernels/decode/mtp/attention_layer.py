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

"""One whole GLM-5.2 DSA attention layer -- projections, indexer, MLA, output -- collectives included.

    AG[hidden] -> down_proj -> AG[latents] (+ AG[block latents] -> ... -> A2A[q]) -> q/kv_up -> RS[seq]
        -> {indexer: q_proj -> AG[q] -> kv_score -> topk_local -> AG[cand] -> merge -> AG[sel]}
        -> MLA attention -> AG[lse] -> cp_combine -> RS[ctx] -> out_proj -> RS[out] -> RS[head]

Every one of the nine kernels under ``attention/{mla_tkg,indexer_tkg}`` runs inside this single
``@nki.jit`` trace, so the ten collectives interleave with the matmuls instead of bracketing separate
kernel launches. Nothing here reimplements attention; this module is the wiring between the per-kernel
modules under ``attention/``.

Sharding (GLM-5.2, 128 devices per model replica, C32 x spec7 = 224 real tokens)
------------------------------------------------------------------------------
A DP group's 64 ranks factor THREE ways, and the split between the last two is a free parameter::

    128 = DP2 x 64        DP group g = ranks [64g, 64g + 64), 16 of the 32 sequences each

    64 = cp_degree x head_shard x token_shard        i = (64/k)*kk + h*cp + hh
         (8)        (h)          (k),  h * k = 8

    cp_degree   shards KEYS      -- every rank sees every token, a share of the context
    head_shard  shards HEADS     -- Nh = 64 / h
    token_shard shards TOKENS    -- T_local = 128 / k, and the rank's OWN sequences

    indexer, within one token slice:  (64/k) = num_seqs_local x key4

``head_shard = 1, token_shard = 8`` is pure data parallelism; ``head_shard = 8, token_shard = 1`` is
the head-only scheme, and reproduces it exactly.

**The token shard is the cheaper split.** Head sharding saves only weight bytes, and it costs on every other axis:

* It does **not** shrink the cache. MLA's cache holds one LATENT row per token and is head-independent, so
  the ranks of a head group hold BYTE-IDENTICAL copies. A token shard gives each rank only its own sequences,
  and cuts the cache by ``k``.
* It does **not** shrink the collectives. ``ctx`` is ``[T_local, Nh, C]``, and ``T_local * Nh`` is the same
  for every split.
* It **hurts both matmuls**: ``Nh`` is on the poor axis of each, the moving free dim of the score (against a
  fixed cost per matmul) and the stationary width of the context sum (of 128 array columns). Every per-token
  cost (the indirect gather, the transposes, their evictions, both matmuls) scales with ``T_local``, which the
  token shard cuts. At ``head_shard = 1`` the head-group collective is also gone.

The weight saving comes from the output path instead: **it is head-sharded over the CP group** for every
``head_shard``. The context ReduceScatter that the CP group runs anyway scatters the HEAD axis, so a rank
projects ``Nh / cp_degree`` heads for all ``T_local`` tokens. That is the same work as every head for
``T_local / cp_degree`` tokens, but ``o_proj_w`` is 25 MB per rank instead of 201 MB at ``Nh = 64``, and the
matmul's stationary free dim is ``cp_degree`` times wider. A second ReduceScatter after ``o_proj``, over the
same group, scatters the tokens. It composes onto global slot ``T_shard * i`` as a token scatter does, so the
digit order and the replica groups are the same.

Handed an ``out_absorb_w`` with FEWER than ``Nh / cp_degree`` heads, the layer shards the output path
``oproj_tp = Nh / nh_out`` ways over the BLOCK GROUP of ``oproj_tp`` consecutive ranks (the CP group
and its ``oproj_tp / cp_degree`` neighbouring token slices): one ``all_to_all`` of the key-partial
context replaces the context ReduceScatter, the receiver sums the CP partials on chip, ``o_proj`` runs
on ``nh_out`` heads for every token of the block, and one ReduceScatter over the block group lands the
MoE rows. Same collective count as the CP-sharded exit, ``o_proj_w`` at ``201 MB / oproj_tp`` per
rank. See :func:`_out_proj_block_sharded`.

**Entry and exit are token-sharded 128 ways**, rank ``r`` owning global slots
``[T_shard*r, T_shard*(r+1))`` with ``T_shard = 2``, for every ``(h, k)``. That is the layout
``moe_layer.py`` consumes and produces, so the two drop into each other
with only the layer's residual add in between. What pins the linearization is exactly that: the CP
token scatter hands a rank ``T_local / cp_degree`` rows and the head scatter ``T_shard`` of those, and
the two compose onto slot ``T_shard * i`` only for the digit order above.

ONE PROGRAM, 128 RANKS: every scalar here is rank-INVARIANT
-----------------------------------------------------------
This is the constraint that shapes the whole module, and it is not visible in the sub-kernels'
signatures. The kernel is traced and compiled **once, from rank 0's inputs**, and that single
NEFF runs on all 128 ranks.
Only ``np.ndarray`` inputs are dumped per rank; every Python ``int`` and ``float`` is constant-folded
at trace time. So a per-rank Python scalar silently becomes **rank 0's value everywhere** -- a bug
with no diagnostic, which shows up as a plausible-looking wrong answer.

The core kernels take exactly the per-rank scalars that break under that rule --
``mla_decode_attention(cp_rank=...)``, ``indexer_kv_score(key_base=...)``,
a local select's ``key_base`` -- plus this layer would want a per-rank ``seq_rank`` to slice
its own sequence out. All four are eliminated here rather than passed, and no core kernel is modified:

* **``seq_rank`` -> a ReduceScatter with ``op=maximum``.** After the latent gather every rank of a
  token group holds the SAME ``[T_local, *]`` tensors, and the indexer needs rows
  ``[spec*s, spec*(s+1))`` of two of them. A ReduceScatter over the sequence group hands group
  position ``s`` those rows reduced across the group -- and reducing identical operands with
  ``maximum`` returns them unchanged. So it is a pure SCATTER: bit-exact, no arithmetic, no per-rank
  offset anywhere. The selection gather on the way out uses the SAME group and is its exact inverse,
  which is why it lands precisely this rank's ``T_local`` query rows.

  ``op=add`` would also "work" and be wrong to use: summing ``seq_size`` identical bf16 values and
  dividing back is not exact (3x, 5x, 6x ... each round), which perturbs the indexer's scores at
  roughly the size of the selection's own tie margin. ``maximum`` has no such failure mode.

  An ``all_to_all`` expresses the same thing more directly, but the runtime's
  mesh algorithm aborts at EXECUTION on strided groups of more than two ranks (``encd_mesh_memcopy:
  Assertion tx_descs_n[j] == tx_descs_n[j - 1] ...``) while AllGather and ReduceScatter run on the very
  same group, and neither the Python validation nor the compiler catches it. The sequence group is
  strided, so the ReduceScatter stays: it is exact and unconditional, and it reads ``T`` rows to deliver
  ``spec`` of them either way -- the row count, not the primitive, is this scatter's cost (see Known
  limits). The one ``all_to_all`` in this layer (:func:`_out_proj_block_sharded`) runs on a
  CONTIGUOUS group, which the runtime accepts.
* **``key_base`` -> 0, and the offset rides two tensors.** ``indexer_kv_score`` reads ``key_base``
  only to place the residency threshold at ``positions[0] - key_base``, so the caller passes an
  ``idx_positions`` already reduced by ``key_rank * idx_keys`` and the kernel is given
  ``key_base = 0``. (The threshold reaches ``indexer_kv_score`` as the precomputed additive mask
  ``idx_mask_add``, built by ``indexer_residency_mask`` once per step -- by the giga kernel, or by this
  layer when run alone -- since it is identical for every indexer layer of a step.)
  the local select reads it only to label its output positions, so the labels come out local and the select
  folds ``key_offset`` -- a ``[1, 1]`` tensor -- into the live ones; the dead-slot sentinel is -1 and stays -1.

  ``idx_shard_degree > 1`` changes only WHICH positions a shard holds, never this scheme. The threshold
  can no longer be collapsed into one subtraction, so ``idx_positions`` carries TRUE positions;
  ``key_offset`` still carries the per-rank term. See ``indexer_global_positions_np``.
* **``cp_rank`` -> 0, exactly, via a position shift.** The two places the attention core uses it are
  round-robin block-ownership tests of the form
  ``((pos >> block_shift) & (cp_degree - 1)) == cp_rank`` (``prior_index.py`` for the gathered prior,
  ``mla_decode_attention`` for the speculative sibling). Shifting every position this rank sees by
  ``(cp_degree - cp_rank) * block_size`` turns the left side into ``(b - cp_rank) mod cp_degree``, so
  testing it against 0 is the SAME predicate. It is exact, not approximate:

    - ``block_l = block_g >> cp_shift`` becomes ``(b + cp_degree - cp_rank) / cp_degree``, which for
      an owned block ``b = cp_rank + k*cp_degree`` is ``k + 1``. So ``block_table``'s local logical
      index is shifted by one and column 0 is unused -- the caller fills it with -1.
    - ``intra = pos & (block_size - 1)`` is unchanged, because the shift is a whole number of blocks.
    - the causal mask compares a shifted key position against a shifted query position, so it is
      unchanged.
    - ``block_l`` for a block this rank does NOT own stays inside ``[0, local_blocks]``, so the table
      lookup never runs off the end; the ownership test rejects those slots anyway.

  The caller therefore supplies a shifted ``positions`` and a one-column-shifted ``block_table``, and
  this layer shifts the selector's output the same way (again live-only) before the attention core
  sees it. Note the RoPE tables are built from the TRUE positions -- they are consumed by
  ``mla_decode_q_kv_up``, which never sees the shift.
* **the CP combine is inlined** (:func:`_cp_softmax_combine`) rather than calling ``mla_cp_combine``,
  for the one thing no shift can fix: that kernel selects its own contribution with a STATIC slice
  ``exp[..., cp_rank]``. Inlining removes the need entirely, because this layer already holds its own
  ``lse`` as a tensor and can use it directly as the numerator. The arithmetic is
  ``mla_cp_combine``'s, and the golden still checks the result against
  ``mla_cp_combine_torch_ref`` called with the rank's TRUE index -- so the substitution is verified,
  not asserted.

Why ``spec`` is padded to 8
---------------------------
The exit is two ReduceScatters -- over CP8 then over KVDP8 -- so the token count the MLA scores must
be divisible by 64, and landing on ``T_shard = 2`` per rank pins it to exactly 128. But
``mla_decode_attention`` requires ``num_tokens == num_seqs * spec``, and 128 is not a multiple of the
model's speculation length 7. So each sequence is padded to ``spec_padded = 8`` slots: 7 real tokens
and one padding slot, 16 sequences, 128 slots, 112 real.

The padding slot is simply an **eighth speculative token whose output is discarded**, not a special
case. It carries a real consecutive position rather than a sentinel, because ``mla_decode_attention``
requires a sequence's ``spec`` slots to occupy consecutive positions -- that is what lets the sibling
causal mask be a position-free iota. So it cannot pollute a real token's attention (the band is causal
and the pad sits last), it cannot fault (its position is in range), and its output lands on a token
slot the MoE block also treats as padding.

Padding to 8 is also exactly what the indexer's top-k budget wants: ``nisa.topk`` spends 16
partitions per query row, so ``T_idx * 16 <= 128`` caps a rank at 8 query tokens.

Where the input norm runs, and why it is not left to the down-projection
-----------------------------------------------------------------------
``mla_decode_down_proj`` can fold the layer's ``input_layernorm`` into its hidden load, and the
standalone kernel tests use it that way. This layer does NOT: it norms ``x_shard`` itself, before the
entry AllGather, and passes ``input_norm_w=None`` down. The norm is per-token, so running it on this
rank's ``T_shard`` rows costs ``1/attn_size`` of running it on the gathered block -- but the real
reason is that **the DSA indexer needs the same normed tensor**. In the reference the decoder layer
applies ``input_layernorm`` and hands the result to the attention module, which passes it straight to
the indexer for both ``wk`` and ``weights_proj``. Feeding the indexer the RAW residual compiles, runs,
and is wrong.

The three norms do NOT share an epsilon. ``input_layernorm`` is constructed with
``config.rms_norm_eps`` (1e-5), while ``q_a_layernorm`` and ``kv_a_layernorm`` are constructed with no
eps argument and so run at ``GlmMoeDsaRMSNorm``'s class default of 1e-6, as does the indexer's
``nn.LayerNorm(head_dim, eps=1e-6)``. Hence three separate epsilon parameters below.

Collectives
-----------
At most twelve in the kernel, and fewer as ``token_shard`` rises: the head-group ReduceScatter
disappears at ``head_shard = 1``, and both the sequence scatter and the selection gather degenerate to
copies at ``num_seqs_local = 1``. The REFERENCE issues nine at ``head_shard = 1``, and only the
reference's count is bounded -- its multi-rank runner replays the collectives by a per-rank call
counter and accepts at most ten collective passes. The one the reference skips is the sequence
scatter, which it performs as a slice because a Python reference may read its own rank.

The two latent AllGathers are **fused** here into one gather of
a packed ``[T, q_lora_shard + kv_row_shard]`` row.

Four of the gathers concatenate on a FEATURE axis, which an HBM collective cannot do
(``_check_collective_dim`` requires ``collective_dim == 0``). Each therefore runs as a dim-0 gather
over the token axis followed by one strided DMA that re-lays the result -- see
:func:`_all_gather_columns`.

Known limits:

* **The sequence scatter reads ``T`` rows to deliver ``spec`` of them.** Its source is the whole
  replicated tensor, so it touches ``seq_size`` times the bytes the indexer consumes. An indirect row
  gather driven by a per-rank index tensor would avoid the collective entirely; the ReduceScatter is
  chosen because its semantics are exact and it is known to work on this runtime.
* **The indexer's 8 query rows underfill every matmul.** ``seq16`` gives a rank one sequence, so the
  query up-projection runs at a moving free dim of 8 against a fixed cost per matmul.
* **``wq_b`` is the biggest replicated weight**, 67.1 MB per rank, and it shards cleanly on its
  contraction ``q_lora_rank`` (degree ``cp_degree``, paid for with an AllReduce of ``[T_local, Nh *
  qk_head_dim]``). It is not done here: it reaches into ``mla_decode_q_kv_up`` and the attention core's
  ``q_absorbed`` layout. An FP8 checkpoint would halve every
  weight.
* **The context ReduceScatter moves the widest tensor**, ``Nh * T_local * C`` = 1 MB per rank, because
  the combine has to happen in the latent space before ``W_UV``.
"""

import contextlib

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from ..attention.mla.mla_decode_constants import P_MAX, MlaDecodeShape
from ..utils.kernel_assert import kernel_assert
from ..utils.row_layout import spread_rows_shape
from ..utils.scratch_hbm import scratch_hbm
from ..utils.trace_names import unique_name
from ..attention.indexer.indexer_block_rows import indexer_block_rows
from .attention.indexer.indexer_constants import (
    indexer_cache_is_blocked,
    indexer_cache_keys,
    local_topk_chunked,
    snake_score_layout,
)
from .attention.indexer.indexer_kv_score import indexer_key_prologue, indexer_kv_score
from ..attention.indexer.indexer_q_proj import indexer_q_proj
from .attention.indexer.indexer_residency_mask import indexer_residency_mask
from .attention.mla.mla_decode_attention import (
    mla_decode_attention,
    mla_decode_attention_prefetch,
    take_rank_major_heads,
)
from .attention.mla.mla_decode_down_proj import mla_decode_down_proj
from .attention.mla.mla_decode_out_proj import mla_decode_out_proj, mla_decode_out_proj_sbuf, o_proj_rows_offsets
from .attention.mla.mla_decode_q_kv_up import mla_decode_q_block_exchange, mla_decode_q_block_finish, mla_decode_q_kv_up
from .attention.mla.mla_proj_primitives import linear, load_gamma_broadcast, rmsnorm_inplace
from .attention.mla.prior_index import build_prior_index, export_prior_index
from .utils.cc_dst_ring import cc_dst
from .utils.cc_pin import pin_attn

_ERROR_PREFIX = "[GLM-5.2 DSA Attention Decode Layer DP/CP/KVDP Kernel]"

# Buffer class for every HBM tensor that is a collective operand. At LNC=1 the RUNTIME rejects a
# collective whose sources are shared_hbm, so both ends are private_hbm: reduce_scatter still enforces
# the strict rule (every src AND dst one buffer type), which rules out mixing. The sub-kernels whose
# outputs are gathered directly -- mla_decode_attention's lse, mla_decode_out_proj, indexer_q_proj,
# indexer_topk_local_packed -- return private_hbm for the same reason.
_COLLECTIVE_HBM = nl.private_hbm

_SNAKE_PARTITIONS = 16
"""Partitions ``nisa.topk`` devotes to one query row. Mirrors ``indexer_constants.SNAKE_PARTITIONS``."""

_A2A_CELL_ALIGN = 8
"""bf16 elements (16 bytes) each head cell of the fused combine's all_to_all payload is padded to."""

CP_COMBINE_IN_A2A = True
"""Block-sharded exit: carry the lse in the context ``all_to_all`` and combine on the receiver
(:func:`_out_proj_block_fused_combine`) instead of AllGathering the lse and scaling on the sender, as the rows layout
needs. One collective and the sender-side pass fewer per layer."""

def mtp_idx_chunk_loop(idx_keys: int, idx_block_rows) -> bool:
    """Whether the chain's full layer scores only the live top-k chunks with the giga kernel's indexer sub-kernels:
    a shard above the select cap and a position-ordered read. The giga kernel's dynamic chunk loop must be on then."""
    if not (local_topk_chunked(idx_keys) and idx_block_rows is None):
        return False
    from ..attention.indexer import indexer_constants as gk_idx

    kernel_assert(
        gk_idx.HEADSUM_PACKED and gk_idx.IDX_DYNAMIC_LOOP and gk_idx.IDX_CHUNK_LOOP,
        f"{_ERROR_PREFIX} the chain's chunk loop needs GLM52K_IDX_HEADSUM_PACKED, GLM52K_IDX_DYNAMIC_LOOP and "
        f"GLM52K_IDX_CHUNK_LOOP (the giga kernel's chunk loop, which the chain's step 0 then runs)",
    )
    return True


def mtp_idx_residency_mask(*args, idx_block_rows=None, **kwargs) -> nl.NkiTensor:
    """``indexer_residency_mask``: the giga kernel's (the packed head sum's layout) under ``mtp_idx_chunk_loop``, else
    this fork's. Same arguments."""
    if mtp_idx_chunk_loop(args[1], idx_block_rows):
        from ..attention.indexer.indexer_residency_mask import indexer_residency_mask as gk_mask

        return gk_mask(*args, **kwargs)
    return indexer_residency_mask(*args, **kwargs)


@contextlib.contextmanager
def _gk_key_hwdge(mode: str):
    """Trace the giga kernel's scorer with ``IDX_KEY_HWDGE = mode`` (restored after); empty is a no-op."""
    if not mode:
        yield
        return
    import importlib

    from ..attention.indexer import indexer_constants as gk_constants

    # The package re-exports the kernel function ``indexer_kv_score``, which shadows the submodule of that name.
    gk_score = importlib.import_module("..attention.indexer.indexer_kv_score", __package__)

    saved = gk_constants.IDX_KEY_HWDGE, gk_score.IDX_KEY_HWDGE
    gk_constants.IDX_KEY_HWDGE = gk_score.IDX_KEY_HWDGE = int(mode)
    try:
        yield
    finally:
        gk_constants.IDX_KEY_HWDGE, gk_score.IDX_KEY_HWDGE = saved


MTP_IDX_GROUP_TOKENS = 4
"""Query rows of the draft group: one 128-partition (token, head) group at 32 indexer heads."""


def mtp_idx_query_tokens(spec: int, idx_keys: int, idx_block_rows) -> int:
    """The step-0 indexer's query rows: in the chunk loop at ``spec == 8``, only the draft row's group of 4 (the only
    selection read after step 0), else ``spec``. The chain builds the residency mask for this many tokens."""
    if spec == 2 * MTP_IDX_GROUP_TOKENS and mtp_idx_chunk_loop(idx_keys, idx_block_rows):
        return MTP_IDX_GROUP_TOKENS
    return spec


def _draft_group_offset(draft_rows: nl.NkiTensor, scale: int) -> nl.NkiTensor:
    """[1, 1] int32 SBUF ``scale * (draft // 4)``, ``draft = draft_rows[0, 0]``: the runtime offset of the draft row's
    group of 4 in a layout with ``scale`` elements per group (``scalar_offset`` of a DMA)."""
    off = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=off[0:1, 0:1], src=draft_rows[0:1, 0:1])
    nisa.tensor_scalar(dst=off, data=off, op0=nl.right_shift, operand0=2)
    nisa.tensor_scalar(dst=off, data=off, op0=nl.multiply, operand0=scale)
    return off


def _pick_group_columns(tiles: list, draft_rows: nl.NkiTensor, width: int, cols: tuple, name: str) -> nl.NkiTensor:
    """One of two per-group ``[P_MAX, *]`` SBUF tiles' columns, chosen at run time by the draft row's group.

    ``tiles[g][:, cols[g] : cols[g] + width]`` goes to columns ``[g * width, (g + 1) * width)`` of a ``[P_MAX, 2 *
    width]`` HBM scratch, and one DMA at the runtime column ``width * (draft // 4)`` loads the chosen block back. The
    same bytes, by DMA only."""
    both = nl.ndarray((P_MAX, 2 * width), dtype=tiles[0].dtype, buffer=_COLLECTIVE_HBM, name=unique_name(name))
    for g, tile in enumerate(tiles):
        nisa.dma_copy(dst=both[0:P_MAX, g * width : (g + 1) * width], src=tile[0:P_MAX, cols[g] : cols[g] + width])
    out = nl.ndarray((P_MAX, width), dtype=tiles[0].dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=out,
        src=both.ap(
            pattern=[[2 * width, P_MAX], [1, width]],
            offset=0,
            scalar_offset=_draft_group_offset(draft_rows, width),
            indirect_dim=1,
        ),
    )
    return out


def _draft_group_query_side(prologue8: tuple, q_idx: nl.NkiTensor, draft_rows: nl.NkiTensor) -> tuple:
    """Draft group: the draft row's group out of the 8-row front: its 4 query rows, and the 8-row
    prologue's head-sum weights and RoPE rows of that group, which are exactly the 4-row prologue's (token ``4 g + c``
    is column ``4 g + c`` of group ``g``'s ``[(token, head), 8]`` weights, column ``c`` at 4 tokens; its RoPE rows sit at
    the same partitions). Returns ``(q_group [4, cols] HBM, (head_w, rope_rows, k_new) for 4 query rows)``."""
    G = MTP_IDX_GROUP_TOKENS
    head_w8, rope_rows8, k_new8 = prologue8
    kernel_assert(
        len(head_w8) == 2 and len(rope_rows8) == 2, f"{_ERROR_PREFIX} the draft group needs two 4-token groups"
    )
    head_w = [_pick_group_columns(head_w8, draft_rows, G, (0, G), "idx_group_head_w")]
    half = rope_rows8[0][0].shape[1]
    cos_g = _pick_group_columns([r[0] for r in rope_rows8], draft_rows, half, (0, 0), "idx_group_cos")
    sin_g = _pick_group_columns([r[1] for r in rope_rows8], draft_rows, half, (0, 0), "idx_group_sin")
    T, W = q_idx.shape
    q_group = nl.ndarray((G, W), dtype=q_idx.dtype, buffer=_COLLECTIVE_HBM, name=unique_name("idx_q_group"))
    nisa.dma_copy(
        dst=q_group.reshape((1, G * W)),
        src=q_idx.reshape((1, T * W)).ap(
            pattern=[[T * W, 1], [1, G * W]],
            offset=0,
            scalar_offset=_draft_group_offset(draft_rows, G * W),
            indirect_dim=1,
        ),
    )
    return q_group, (head_w, [(cos_g, sin_g)], k_new8)


def _write_all_new_keys(
    idx_k_cache: nl.NkiTensor,
    k_new_sbuf: nl.NkiTensor,
    new_slot_mapping: nl.NkiTensor,
    q_group: nl.NkiTensor,
) -> nl.NkiTensor:
    """Draft group: write all ``spec`` new keys with the giga kernel's ``_write_key_cache`` (the
    same clamp, cast and scatter it does for an 8-row layer), and return all-skip slots for the scorer's own write of
    its ``len(q_group)`` rows."""
    from ..attention.indexer.indexer_constants import IndexerScoreShape
    from ..attention.indexer.indexer_kv_score import _write_key_cache

    n_new = new_slot_mapping.shape[0]
    n_keys = indexer_cache_keys(idx_k_cache)
    HD = idx_k_cache.shape[-1]
    shapes = IndexerScoreShape(
        num_tokens=n_new,
        num_new=n_new,
        num_heads=1,
        head_dim=HD,
        rope_dim=0,
        hidden=0,
        num_keys=n_keys,
        key_base=0,
        dtype=q_group.dtype,
        key_fp8=idx_k_cache.dtype == nl.uint8,
    )
    cache_rows = idx_k_cache.reshape((n_keys, HD)) if indexer_cache_is_blocked(idx_k_cache) else idx_k_cache
    _write_key_cache(cache_rows, k_new_sbuf, new_slot_mapping, shapes)
    T = q_group.shape[0]
    skip_sb = nl.ndarray((T, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=skip_sb, value=-1)
    skip = nl.ndarray((T,), dtype=nl.int32, buffer=_COLLECTIVE_HBM, name=unique_name("idx_skip_slots"))
    nisa.dma_copy(dst=skip.reshape((T, 1)), src=skip_sb)
    return skip


def _pin_all(*tensors):
    """Pin every collective operand while the attention pins are on (long context; see ``utils/cc_pin.py``)."""
    pin_attn(*tensors)


def _all_gather_columns(
    src_hbm: nl.NkiTensor,
    replica_group: ReplicaGroup,
    group_size: int,
    tokens: int,
    width: int,
    name: str,
) -> nl.NkiTensor:
    """AllGather a ``[tokens, width]`` COLUMN shard and concatenate on the FEATURE axis.

    An HBM collective only concatenates on dim 0, so this gathers on the token axis -- landing rank
    ``r``'s block at rows ``[tokens*r, tokens*(r+1))`` -- and then re-lays it with a single strided
    DMA, because rank ``r``'s columns belong at ``[width*r, width*(r+1))``. The DMA reads ``(t, r, c)``
    with strides ``(width, tokens*width, 1)``, which is the transpose of the gather's two leading axes
    and nothing more.

    Args:
        src_hbm (nl.NkiTensor): [tokens, width] this rank's column slice in HBM.
        replica_group (ReplicaGroup): The group whose ranks hold the other slices, IN COLUMN ORDER.
        group_size (int): Ranks in that group.
        tokens (int): Rows, at most 128 (they ride the repack DMA's partition axis).
        width (int): This rank's column count.
        name (str): Debug name stem for the two allocations.

    Returns:
        nl.NkiTensor: [tokens, group_size * width] in HBM, the complete tensor.
    """
    gathered = cc_dst((tokens * group_size, width), src_hbm.dtype, _COLLECTIVE_HBM, f"{name}_rows")
    _pin_all(src_hbm, gathered)  # while pinning is on; see utils/cc_pin.py
    ncc.all_gather(srcs=[src_hbm], dsts=[gathered], replica_group=replica_group, collective_dim=0)

    # One HBM -> HBM DMA does the relayout; a bounce through SBUF (HBM -> SB -> HBM) would put two
    # DMA latencies on the critical path where one suffices.
    wide = nl.ndarray(
        (tokens, group_size * width), dtype=src_hbm.dtype, buffer=_COLLECTIVE_HBM, name=unique_name(f"{name}_wide")
    )
    nisa.dma_copy(
        dst=wide.reshape((tokens, group_size, width)),
        src=gathered.ap(pattern=[[width, tokens], [tokens * width, group_size], [1, width]], offset=0),
    )
    return wide


def _gather_latent_pair(
    packed: nl.NkiTensor,
    replica_group: ReplicaGroup,
    group_size: int,
    tokens: int,
    q_width: int,
    kv_width: int,
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """AllGather the (query latent | cache row) column pair and split it back into two complete tensors.

    Rank ``r``'s pair lands at rows ``[tokens*r, tokens*(r+1))`` of the gather; its query columns belong
    at ``[q_width*r, ...)`` of the latent and its cache-row columns at ``[kv_width*r, ...)`` of the row, so
    each output is one strided HBM -> HBM DMA reading ``(t, r, c)`` with strides ``(pair, tokens*pair, 1)``.
    Replaces ``_all_gather_columns`` + ``_split_latents`` on this path: the same result in 2 DMAs that ran
    in parallel instead of 6 in three dependent rounds (gather -> wide -> split).

    Args:
        packed (nl.NkiTensor): [tokens, q_width + kv_width] this rank's pair in HBM (a collective source).
        replica_group (ReplicaGroup): The group whose ranks hold the other column slices, IN COLUMN ORDER.
        group_size (int): Ranks in that group.
        tokens (int): Row count.
        q_width (int): One rank's query-latent columns.
        kv_width (int): One rank's cache-row columns.

    Returns:
        tuple[nl.NkiTensor, nl.NkiTensor]: (q_latent [tokens, group_size*q_width], kv_row
            [tokens, group_size*kv_width]), both complete and in HBM. ``kv_width`` 0 (the query latent alone)
            returns None for the row; at one token the gather's destination then IS the latent (rank ``r``'s
            columns at ``[q_width*r, ...)``), so no split DMA is needed.
    """
    pair = q_width + kv_width
    gathered = cc_dst((tokens * group_size, pair), packed.dtype, _COLLECTIVE_HBM, "latent_pair_rows")
    _pin_all(packed, gathered)  # while pinning is on; see utils/cc_pin.py
    ncc.all_gather(srcs=[packed], dsts=[gathered], replica_group=replica_group, collective_dim=0)
    if kv_width == 0 and tokens == 1:
        return gathered.reshape((1, group_size * q_width)), None

    q_latent = nl.ndarray(
        (tokens, group_size * q_width), dtype=packed.dtype, buffer=_COLLECTIVE_HBM, name=unique_name("q_latent_full")
    )
    if kv_width == 0:
        nisa.dma_copy(
            dst=q_latent.reshape((tokens, group_size, q_width)),
            src=gathered.ap(pattern=[[pair, tokens], [tokens * pair, group_size], [1, q_width]], offset=0),
        )
        return q_latent, None
    kv_row = nl.ndarray(
        (tokens, group_size * kv_width), dtype=packed.dtype, buffer=_COLLECTIVE_HBM, name=unique_name("kv_row_full")
    )
    nisa.dma_copy(
        dst=q_latent.reshape((tokens, group_size, q_width)),
        src=gathered.ap(pattern=[[pair, tokens], [tokens * pair, group_size], [1, q_width]], offset=0),
    )
    nisa.dma_copy(
        dst=kv_row.reshape((tokens, group_size, kv_width)),
        src=gathered.ap(pattern=[[pair, tokens], [tokens * pair, group_size], [1, kv_width]], offset=q_width),
    )
    return q_latent, kv_row


def _gather_block_latent(
    q_latent_shard: nl.NkiTensor,
    cos: nl.NkiTensor,
    sin: nl.NkiTensor,
    block_replica_group: ReplicaGroup,
    block_degree: int,
    token_group: int,
    tokens: int,
    width: int,
    name: str,
) -> tuple[nl.NkiTensor, nl.NkiTensor, nl.NkiTensor]:
    """AllGather the query latent's column shards -- and the RoPE tables -- over the BLOCK group and
    assemble the block's tokens.

    The block group is ``slices = block_degree / token_group`` token slices x the token group, in rank
    order ``(slice, c)``, so the dim-0 gather lands shard ``c`` of slice ``s`` at rows
    ``[(s * token_group + c) * tokens, ...)``. Token ``(s, t)`` of the block wants its ``token_group``
    shards side by side, which one DMA per slice re-lays (``(t, c, w)`` with strides ``(width,
    tokens * width, 1)`` inside the slice's row block), exactly as :func:`_all_gather_columns` does for
    one slice.

    The tables take a second AllGather on the same group, issued back to back with the first (a
    coalesced collective needs one dtype, and the tables are fp32): ``cos`` and ``sin`` side by side in
    one ``[tokens, rope]`` tensor. The ``token_group`` ranks of a slice hold identical tables, so the
    block's ``[slices * tokens, rope / 2]`` tables are shard 0's rows of every slice.

    Args:
        q_latent_shard (nl.NkiTensor): [tokens, width] this rank's column slice, HBM.
        cos (nl.NkiTensor): [tokens, rope / 2] fp32 this rank's cosine table, HBM.
        sin (nl.NkiTensor): [tokens, rope / 2] fp32 likewise.
        block_replica_group (ReplicaGroup): The ``block_degree`` consecutive ranks, in rank order.
        block_degree (int): Ranks in the block group.
        token_group (int): Column shards per token slice.
        tokens (int): Tokens per slice, at most 128 / slices (they ride the repack DMA's partitions).
        width (int): This rank's column count.
        name (str): Debug name stem.

    Returns:
        tuple[nl.NkiTensor, nl.NkiTensor, nl.NkiTensor]: the block's complete query latent
        ``[slices * tokens, token_group * width]`` (slice-major) and its ``cos`` / ``sin`` tables
        ``[slices * tokens, rope / 2]``, all in HBM.
    """
    slices = block_degree // token_group
    half = cos.shape[1]
    dtype = q_latent_shard.dtype
    # private_hbm for the collective sources: the down-projection's outputs and the tables are kernel IO.
    src = nl.ndarray((tokens, width), dtype=dtype, buffer=_COLLECTIVE_HBM, name=unique_name(f"{name}_src"))
    nisa.dma_copy(dst=src, src=q_latent_shard[0:tokens, 0:width])
    tab_src = nl.ndarray(
        (tokens, 2 * half), dtype=nl.float32, buffer=_COLLECTIVE_HBM, name=unique_name(f"{name}_tab_src")
    )
    nisa.dma_copy(dst=tab_src[0:tokens, 0:half], src=cos[0:tokens, 0:half])
    nisa.dma_copy(dst=tab_src[0:tokens, half : 2 * half], src=sin[0:tokens, 0:half])
    gathered = nl.ndarray(
        (tokens * block_degree, width), dtype=dtype, buffer=_COLLECTIVE_HBM, name=unique_name(f"{name}_rows")
    )
    tab_rows = nl.ndarray(
        (tokens * block_degree, 2 * half),
        dtype=nl.float32,
        buffer=_COLLECTIVE_HBM,
        name=unique_name(f"{name}_tab_rows"),
    )
    _pin_all(src, gathered, tab_src, tab_rows)  # while pinning is on; see utils/cc_pin.py
    ncc.all_gather(srcs=[src], dsts=[gathered], replica_group=block_replica_group, collective_dim=0)
    ncc.all_gather(srcs=[tab_src], dsts=[tab_rows], replica_group=block_replica_group, collective_dim=0)

    wide_sb = nl.ndarray((slices * tokens, token_group, width), dtype=dtype, buffer=nl.sbuf)
    tab_sb = nl.ndarray((slices * tokens, 2 * half), dtype=nl.float32, buffer=nl.sbuf)
    for s in range(slices):
        r0 = s * token_group * tokens  # first row of slice s (its shard 0)
        nisa.dma_copy(
            dst=wide_sb[s * tokens : (s + 1) * tokens, 0:token_group, 0:width],
            src=gathered.ap(pattern=[[width, tokens], [tokens * width, token_group], [1, width]], offset=r0 * width),
        )
        nisa.dma_copy(
            dst=tab_sb[s * tokens : (s + 1) * tokens, 0 : 2 * half], src=tab_rows[r0 : r0 + tokens, 0 : 2 * half]
        )
    block = nl.ndarray(
        (slices * tokens, token_group * width), dtype=dtype, buffer=_COLLECTIVE_HBM, name=unique_name(f"{name}_wide")
    )
    cos_block = nl.ndarray(
        (slices * tokens, half), dtype=nl.float32, buffer=_COLLECTIVE_HBM, name=unique_name(f"{name}_cos")
    )
    sin_block = nl.ndarray(
        (slices * tokens, half), dtype=nl.float32, buffer=_COLLECTIVE_HBM, name=unique_name(f"{name}_sin")
    )
    nisa.dma_copy(dst=block, src=wide_sb.reshape((slices * tokens, token_group * width)))
    nisa.dma_copy(dst=cos_block, src=tab_sb[0 : slices * tokens, 0:half])
    nisa.dma_copy(dst=sin_block, src=tab_sb[0 : slices * tokens, half : 2 * half])
    return block, cos_block, sin_block


def build_norm_reduce_ones(T_shard: int, H: int) -> nl.NkiTensor | None:
    """The block-diagonal ones matrix the partition-spread input norm reduces with, or None when the rows do not spread.

    ``[P_MAX, P_MAX]`` fp32, ones where both indices fall in the same token's partition block. Layer-invariant: the
    giga stack builds it once and hands it to every layer (``input_norm_reduce_ones``) so no layer spends
    memsets on it.

    Args:
        T_shard (int): Rows of the residual shard.
        H (int): Row width.

    Returns:
        nl.NkiTensor | None: The matrix in SBUF, or None.
    """
    spread = spread_rows_shape(T_shard, H)
    if spread is None:
        return None
    P, _ = spread
    ppt = P // T_shard
    ones_blk = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    if ppt == P:
        nisa.memset(ones_blk, 1.0)
    else:
        nisa.memset(ones_blk, 0.0)
        for t in range(T_shard):
            nisa.memset(ones_blk[t * ppt : (t + 1) * ppt, t * ppt : (t + 1) * ppt], 1.0)
    return ones_blk


def _rmsnorm_shard(
    x_shard: nl.NkiTensor,
    input_norm_w: nl.NkiTensor,
    eps: float,
    x_sb: nl.NkiTensor = None,
    reduce_ones: nl.NkiTensor = None,
) -> nl.NkiTensor:
    """Apply the layer's ``input_layernorm`` to this rank's token shard, before the entry AllGather.

    Runs here rather than inside ``mla_decode_down_proj`` because the DSA indexer consumes the SAME
    normed tensor -- see the module docstring.

    Args:
        x_shard (nl.NkiTensor): [T_shard, H] this rank's slots of the raw residual stream.
        input_norm_w (nl.NkiTensor): [1, H] fp32 RMSNorm gamma.
        eps (float): ``config.rms_norm_eps``, 1e-5 -- NOT the 1e-6 the latent norms use.
        x_sb (nl.NkiTensor): Optional SBUF copy of ``x_shard`` in the ``spread_rows_shape`` layout (the giga
            stack's residual add leaves one behind), which skips the load.
        reduce_ones (nl.NkiTensor): Optional ``build_norm_reduce_ones`` result to reuse; None builds one here.

    Returns:
        nl.NkiTensor: [T_shard, H] normed, in HBM, ready to be a collective source.
    """
    T_shard, H = x_shard.shape
    normed = nl.ndarray(
        (T_shard, H), dtype=x_shard.dtype, buffer=_COLLECTIVE_HBM, name=unique_name("hidden_normed_shard")
    )

    spread = spread_rows_shape(T_shard, H)
    if spread is None:
        x_tok = nl.ndarray((P_MAX, H), dtype=x_shard.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tok[0:T_shard, 0:H], src=x_shard[0:T_shard, 0:H])
        rmsnorm_inplace(x_tok[0:P_MAX, 0:H], load_gamma_broadcast(input_norm_w, T_shard), T_shard, eps)
        nisa.dma_copy(dst=normed[0:T_shard, 0:H], src=x_tok[0:T_shard, 0:H])
        return normed

    # Partition-spread: partition p holds elements [p*F, (p+1)*F) of the flattened rows, so token t is
    # partitions [t*ppt, (t+1)*ppt). Same arithmetic as rmsnorm_inplace -- fp32 sum of squares,
    # rsqrt(sum/H + eps), x * rstd * gamma -- with the sum formed as per-partition partials and one
    # block-diagonal ones matmul, so it can differ from the token-major sum in fp32 rounding order only.
    P, F = spread
    ppt = P // T_shard
    if x_sb is None:
        x_sb = nl.ndarray((P, F), dtype=x_shard.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_sb, src=x_shard.reshape((P, F)))
    kernel_assert(
        tuple(x_sb.shape) == (P, F) and x_sb.buffer == nl.sbuf,
        f"{_ERROR_PREFIX} x_sb must be a [{P}, {F}] SBUF tile, got {x_sb.shape} in {x_sb.buffer}",
    )

    inv_rms = _spread_inv_rms(x_sb, T_shard, H, eps, reduce_ones)

    # gamma in the same layout: every token's block of partitions sees the whole [1, H] row.
    gamma_sb = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
    for t in range(T_shard):
        nisa.dma_copy(dst=gamma_sb[t * ppt : (t + 1) * ppt, 0:F], src=input_norm_w[0:1, 0:H].reshape((ppt, F)))

    normed_sb = nl.ndarray((P, F), dtype=x_shard.dtype, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=normed_sb, data=x_sb, op0=nl.multiply, operand0=inv_rms[0:P, 0:1], op1=nl.multiply, operand1=gamma_sb
    )
    nisa.dma_copy(dst=normed.reshape((P, F)), src=normed_sb)
    return normed


def _spread_inv_rms(
    x_sb: nl.NkiTensor, T_shard: int, H: int, eps: float, reduce_ones: nl.NkiTensor = None
) -> nl.NkiTensor:
    """``rsqrt(mean(x^2) + eps)`` per partition of a partition-spread ``[P_MAX, F]`` row block (``_rmsnorm_shard``).

    Per-partition fp32 sums of squares, one block-diagonal ones matmul, one rsqrt: every partition of a token's
    block gets that token's value.

    Returns:
        nl.NkiTensor: [P_MAX, 1] fp32 SBUF.
    """
    P, F = x_sb.shape
    squares_discard = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
    partial = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation_reduce(dst=squares_discard, op=nl.square, data=x_sb, reduce_op=nl.add, reduce_res=partial)

    ones_blk = reduce_ones if reduce_ones is not None else build_norm_reduce_ones(T_shard, H)
    total_ps = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=total_ps, stationary=ones_blk, moving=partial)
    inv_rms = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=inv_rms, op=nl.rsqrt, data=total_ps, scale=1.0 / H, bias=eps)
    return inv_rms


def _row_offsets(rows_seq: nl.NkiTensor, rows_per_seq: int, n: int) -> nl.NkiTensor:
    """[P_MAX, 1] int32 SBUF, partition ``p < n`` = ``rows_seq[0] * rows_per_seq + p``: a row index into a
    ``[n_seq * rows_per_seq, W]`` view of the rows table, built on device (the row is the rank's)."""
    seq_i = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=seq_i[0:n, 0:1], src=rows_seq[0:1, 0:1].broadcast(0, n))
    seq_f = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=seq_f[0:n, 0:1], src=seq_i[0:n, 0:1])
    lane_i = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(lane_i[0:n, 0:1], pattern=[[0, 1]], offset=0, channel_multiplier=1)
    lane_f = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=lane_f[0:n, 0:1], src=lane_i[0:n, 0:1])
    # seq * rows_per_seq + lane, exact in fp32.
    nisa.scalar_tensor_tensor(
        dst=lane_f[0:n, 0:1],
        data=seq_f[0:n, 0:1],
        op0=nl.multiply,
        operand0=float(rows_per_seq),
        op1=nl.add,
        operand1=lane_f[0:n, 0:1],
    )
    rows_i = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=rows_i[0:n, 0:1], src=lane_f[0:n, 0:1])
    return rows_i


def _gathered_row_offsets(rows_seq: nl.NkiTensor, chunks_per_block: int, n_seq: int, n: int) -> nl.NkiTensor:
    """[P_MAX, 1] int32 SBUF, the :func:`_row_offsets` of a rank-major gathered rows table (``rows_in_groups``).

    The table ``[G * n_seq, W]`` holds column block ``r`` of every sequence at rows ``[r n_seq, (r + 1) n_seq)``. Viewed
    as rows of ``W / m`` elements (``m = chunks_per_block``), chunk ``l`` of sequence ``seq``'s row -- its elements
    ``[l W / m, (l + 1) W / m)`` -- is row ``(r n_seq + seq) m + l % m`` with ``r = l // m``, i.e.
    ``seq * m + r * m * (n_seq - 1) + l``: partition ``l < n`` gets that, built on device (the row is the rank's).
    """
    m = chunks_per_block
    # r = l // m as (l * mul) >> shift, exact for every l < n (checked here, at trace time).
    shift = next(k for k in range(24) if all((l * -(-(1 << k) // m)) >> k == l // m for l in range(n)))
    mul = -(-(1 << shift) // m)
    kernel_assert(
        (n - 1) * mul < 1 << 24 and m * n_seq * n < 1 << 24,
        f"{_ERROR_PREFIX} the gathered row offsets need exact fp32 integers, got m={m}, n_seq={n_seq}, n={n}",
    )
    seq_i = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=seq_i[0:n, 0:1], src=rows_seq[0:1, 0:1].broadcast(0, n))
    seq_f = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=seq_f[0:n, 0:1], src=seq_i[0:n, 0:1])
    lane_i = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(lane_i[0:n, 0:1], pattern=[[0, 1]], offset=0, channel_multiplier=1)
    blk_i = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
    if mul == 1:
        nisa.tensor_scalar(dst=blk_i[0:n, 0:1], data=lane_i[0:n, 0:1], op0=nl.right_shift, operand0=shift)
    else:
        nisa.tensor_scalar(dst=blk_i[0:n, 0:1], data=lane_i[0:n, 0:1], op0=nl.multiply, operand0=mul)
        nisa.tensor_scalar(dst=blk_i[0:n, 0:1], data=blk_i[0:n, 0:1], op0=nl.right_shift, operand0=shift)
    lane_f = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=lane_f[0:n, 0:1], src=lane_i[0:n, 0:1])
    blk_f = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=blk_f[0:n, 0:1], src=blk_i[0:n, 0:1])
    # seq * m + lane, then + r * m * (n_seq - 1), exact in fp32.
    nisa.scalar_tensor_tensor(
        dst=lane_f[0:n, 0:1],
        data=seq_f[0:n, 0:1],
        op0=nl.multiply,
        operand0=float(m),
        op1=nl.add,
        operand1=lane_f[0:n, 0:1],
    )
    nisa.scalar_tensor_tensor(
        dst=lane_f[0:n, 0:1],
        data=blk_f[0:n, 0:1],
        op0=nl.multiply,
        operand0=float(m * (n_seq - 1)),
        op1=nl.add,
        operand1=lane_f[0:n, 0:1],
    )
    rows_i = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=rows_i[0:n, 0:1], src=lane_f[0:n, 0:1])
    return rows_i


def _draft_row_k_tiles(
    rows_in: nl.NkiTensor,
    rows_seq: nl.NkiTensor,
    input_norm_w: nl.NkiTensor,
    eps: float,
    state: dict,
    groups: int = 1,
) -> nl.NkiTensor:
    """The input-normed draft row of this slice's ONE sequence, straight in ``linear``'s K-tile packing.

    Two indirect picks from the rows table, issued together: the row partition-spread (``[P_MAX, H / P_MAX]``,
    partition ``p`` = elements ``[p F, (p + 1) F)``), on which :func:`_spread_inv_rms` runs exactly as
    :func:`_rmsnorm_shard` does, and the row k-tile-major (``[H / P_MAX, P_MAX]``, row ``i`` = elements
    ``[128 i, 128 (i + 1))``), which ONE pass-through PE transpose turns into the column-major ``[P_MAX, H / P_MAX]``
    tile whose column ``i`` is k-tile ``i``. The normalize is ``_rmsnorm_shard``'s instruction, ``(x * inv_rms) *
    gamma`` per element in fp32 then bf16, on that tile with gamma laid out the same way; at one token every
    partition's ``inv_rms`` is the same value, so each element gets the same arithmetic. Replaces the pick, its
    store, the spread reload, the normed store and reload, and 48 [1, 128] transposes with their copies.

    ``state`` is trace-time memory the caller keeps across the chain's steps: the two index vectors, the
    column-major gamma and the norm's ones matrix are built on first use only.

    ``groups`` > 1: ``rows_in`` is a column AllGather's rank-major destination (``rows_in_groups`` of the layer) and the
    picks read the row's column blocks from it in place (:func:`_gathered_row_offsets`); the same bytes land.

    Returns:
        nl.NkiTensor: [P_MAX, H / P_MAX] bf16 SBUF.
    """
    R, W = rows_in.shape
    n_seq, H = R // groups, W * groups
    spread = spread_rows_shape(1, H)
    kernel_assert(
        spread is not None and H % P_MAX == 0,
        f"{_ERROR_PREFIX} the draft front pick needs a spreadable row, got H={H}",
    )
    P, F = spread
    nK = H // P_MAX
    kernel_assert(
        groups == 1 or (W % F == 0 and W % P_MAX == 0 and state.get("groups", groups) == groups),
        f"{_ERROR_PREFIX} a gathered rows_in needs whole picks per column block ({W} columns, picks of {F} and "
        f"{P_MAX}) and the same layout on every step",
    )
    if "gamma_cm" not in state:
        if groups == 1:
            state["spread_rows"] = _row_offsets(rows_seq, P, P)
            state["k_rows"] = _row_offsets(rows_seq, nK, nK)
        else:
            state["spread_rows"] = _gathered_row_offsets(rows_seq, W // F, n_seq, P)
            state["k_rows"] = _gathered_row_offsets(rows_seq, W // P_MAX, n_seq, nK)
            state["groups"] = groups
        # gamma column-major: partition p, column i = gamma[128 i + p] (one 4-byte element per descriptor, once).
        gamma_cm = nl.ndarray((P_MAX, nK), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=gamma_cm, src=input_norm_w.ap(pattern=[[1, P_MAX], [P_MAX, nK]], offset=0))
        state["gamma_cm"] = gamma_cm
        state["ones"] = build_norm_reduce_ones(1, H)
    x_sb = nl.ndarray((P, F), dtype=rows_in.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=x_sb,
        src=rows_in.reshape((R * W // F, F)).ap(
            pattern=[[F, P], [1, F]], offset=0, vector_offset=state["spread_rows"][0:P, 0:1], indirect_dim=0
        ),
    )
    kt_sb = nl.ndarray((P_MAX, P_MAX), dtype=rows_in.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=kt_sb[0:nK, 0:P_MAX],
        src=rows_in.reshape((R * W // P_MAX, P_MAX)).ap(
            pattern=[[P_MAX, nK], [1, P_MAX]], offset=0, vector_offset=state["k_rows"][0:nK, 0:1], indirect_dim=0
        ),
    )
    inv_rms = _spread_inv_rms(x_sb, 1, H, eps, state["ones"])
    cm_psum = nl.ndarray((P_MAX, P_MAX), dtype=rows_in.dtype, buffer=nl.psum)
    nisa.nc_transpose(cm_psum[0:P_MAX, 0:nK], kt_sb[0:nK, 0:P_MAX])
    x_cm = nl.ndarray((P_MAX, nK), dtype=rows_in.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=x_cm, src=cm_psum[0:P_MAX, 0:nK])
    normed_cm = nl.ndarray((P_MAX, nK), dtype=rows_in.dtype, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=normed_cm,
        data=x_cm,
        op0=nl.multiply,
        operand0=inv_rms[0:P_MAX, 0:1],
        op1=nl.multiply,
        operand1=state["gamma_cm"],
    )
    return normed_cm


def _pack_latents(q_latent_shard: nl.NkiTensor, kv_row_shard: nl.NkiTensor, tokens: int) -> nl.NkiTensor:
    """Lay the two down-projection shards side by side so ONE AllGather carries both.

    Both are column shards of the same ranks over the same token rows, so concatenating them on the
    feature axis and gathering once reconstructs both. That halves the entry's collective count, which
    is what leaves room for the sequence all-to-all inside the reference's budget of ten.

    Args:
        q_latent_shard (nl.NkiTensor): [tokens, q_lora_shard] from ``mla_decode_down_proj``.
        kv_row_shard (nl.NkiTensor): [tokens, kv_row_shard] from ``mla_decode_down_proj``.
        tokens (int): Row count.

    Returns:
        nl.NkiTensor: [tokens, q_lora_shard + kv_row_shard] in HBM.
    """
    q_width, kv_width = q_latent_shard.shape[1], kv_row_shard.shape[1]
    packed = nl.ndarray(
        (tokens, q_width + kv_width),
        dtype=q_latent_shard.dtype,
        buffer=_COLLECTIVE_HBM,
        name=unique_name("latent_pair"),
    )
    nisa.dma_copy(dst=packed[0:tokens, 0:q_width], src=q_latent_shard[0:tokens, 0:q_width])
    nisa.dma_copy(dst=packed[0:tokens, q_width : q_width + kv_width], src=kv_row_shard[0:tokens, 0:kv_width])
    return packed


def _split_latents(
    packed: nl.NkiTensor,
    tokens: int,
    group_size: int,
    q_width: int,
    kv_width: int,
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """Split the gathered latent pair back into the complete query latent and cache row.

    The gathered tensor is ``[tokens, group_size * (q_width + kv_width)]`` with rank ``r``'s pair at
    ``[r*(q_width+kv_width), ...)``. Each output is therefore a STRIDED selection: ``group_size``
    blocks spaced one pair apart, which one access pattern expresses.

    Args:
        packed (nl.NkiTensor): [tokens, group_size * (q_width + kv_width)] gathered pair in HBM.
        tokens (int): Row count.
        group_size (int): Column-shard degree.
        q_width (int): One rank's query-latent columns.
        kv_width (int): One rank's cache-row columns.

    Returns:
        tuple[nl.NkiTensor, nl.NkiTensor]: (q_latent, kv_row), both complete and in HBM.
    """
    pair = q_width + kv_width
    stride = group_size * pair

    q_sb = nl.ndarray((tokens, group_size, q_width), dtype=packed.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=q_sb, src=packed.ap(pattern=[[stride, tokens], [pair, group_size], [1, q_width]], offset=0))
    kv_sb = nl.ndarray((tokens, group_size, kv_width), dtype=packed.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=kv_sb, src=packed.ap(pattern=[[stride, tokens], [pair, group_size], [1, kv_width]], offset=q_width)
    )

    q_latent = nl.ndarray(
        (tokens, group_size * q_width), dtype=packed.dtype, buffer=_COLLECTIVE_HBM, name=unique_name("q_latent_full")
    )
    kv_row = nl.ndarray(
        (tokens, group_size * kv_width), dtype=packed.dtype, buffer=_COLLECTIVE_HBM, name=unique_name("kv_row_full")
    )
    nisa.dma_copy(dst=q_latent, src=q_sb.reshape((tokens, group_size * q_width)))
    nisa.dma_copy(dst=kv_row, src=kv_sb.reshape((tokens, group_size * kv_width)))
    return q_latent, kv_row


def build_shared_prior_index(
    gather_indices_true: nl.NkiTensor,
    cp_pos_offset: nl.NkiTensor,
    kv_cache: nl.NkiTensor,
    block_table: nl.NkiTensor,
    positions: nl.NkiTensor,
    T_local: int,
    num_seqs_local: int,
    spec: int,
    index_topk: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    cp_degree: int,
    io_dtype,
) -> tuple[nl.NkiTensor, nl.NkiTensor, nl.NkiTensor]:
    """Resolve one selection to this rank's prior-index tables ONCE, for every layer that shares it.

    ``mla_decode_attention`` spends significant time turning the 2048 selected positions per
    token into local cache rows, the snake-permuted gather index and the key-position mask operands.
    Those tables depend on the selection, ``block_table`` and the CP shift -- not on any layer's
    weights or cache contents -- so the three ``"shared"`` layers of an index-share group would each
    rebuild byte-identical copies. This performs the same CP shift the layer applies and the same
    ``build_prior_index`` the core would run, and stages the result in HBM for
    ``glm52_dsa_attention_decode_layer_kernel(prior_snake_rows=..., prior_key_pos=...,
    prior_natural_rows=...)``.

    Args:
        gather_indices_true (nl.NkiTensor): [T_local, index_topk] int32 TRUE global positions -- the
            full layer's ``gather_indices_out`` in its flat form.
        cp_pos_offset (nl.NkiTensor): [1, 1] int32, the per-rank CP position shift.
        kv_cache (nl.NkiTensor): Any layer's cache in the group; only its SHAPE is read.
        block_table (nl.NkiTensor): [num_seqs_local, table_width] int32, this rank's local table.
        positions (nl.NkiTensor): [T_local] int32 shifted query positions; only its shape is read.
        T_local, num_seqs_local, spec, index_topk, kv_lora_rank, qk_rope_head_dim, cp_degree: shapes.
        io_dtype: The attention io dtype (only recorded in the shape object).

    Returns:
        (snake_hbm, key_pos_hbm, natural_hbm): the three tables in ``shared_hbm``, each
        ``[P_MAX, T_local, index_topk // P_MAX]``.
    """
    gather_3d = gather_indices_true.reshape((num_seqs_local, spec, index_topk))
    # The same extents ``get_mla_decode_shape`` derives inside the core, from the same operands. Only
    # the fields ``build_prior_index`` reads matter (block geometry, CP, token counts); the head count
    # and dtype are recorded for completeness and never used on this path.
    block_size = kv_cache.shape[2]
    kernel_assert(positions.shape[0] == T_local, f"positions must be [{T_local}], got {positions.shape}")
    block_shift = 0
    while (1 << block_shift) < block_size:
        block_shift += 1
    mla = MlaDecodeShape(
        num_seqs=num_seqs_local,
        spec=spec,
        num_heads=1,
        kv_lora_dim=kv_lora_rank,
        qk_rope_dim=qk_rope_head_dim,
        num_prior=index_topk,
        num_blocks=kv_cache.shape[0],
        block_size=block_size,
        max_blocks=block_table.shape[1],
        block_shift=block_shift,
        cp_degree=cp_degree,
        cp_rank=0,
        dtype=io_dtype,
    )
    # cp_pos_offset is folded into the table build (live slots only), see build_prior_index.
    snake_rows, key_pos_f, natural_rows = build_prior_index(gather_3d, block_table, mla, pos_offset=cp_pos_offset)
    return export_prior_index(snake_rows, key_pos_f, natural_rows, name="shared_prior_index")


def _cp_softmax_combine(
    ctx_local: nl.NkiTensor,
    lse_local: nl.NkiTensor,
    lse_all: nl.NkiTensor,
    tokens: int,
    num_heads: int,
    latent_dim: int,
    cp_degree: int,
) -> nl.NkiTensor:
    """Scale this rank's partial context by its softmax weight over the CP ranks.

    The arithmetic of ``mla_cp_combine``, with one change: the numerator comes from ``lse_local``
    directly instead of from ``lse_all[cp_rank]``. That is the whole reason this is inlined -- a static
    slice by ``cp_rank`` cannot be made rank-invariant, and the layer already holds its own ``lse`` as
    a tensor. The golden still checks the result against ``mla_cp_combine_torch_ref`` called with the
    rank's TRUE index, so the substitution is verified rather than assumed.

        w_r = exp(lse_r - m) / sum_s exp(lse_s - m),   m = max_s lse_s

    is algebraically ``exp(lse_r - lse*)`` but never forms a ``log`` and never exponentiates a positive
    argument, so it cannot overflow. An empty rank's sentinel-magnitude ``lse`` underflows to exactly 0
    and contributes nothing; if every rank is empty the weights become a uniform ``1/cp_degree`` --
    meaningless but finite, never a NaN in the residual stream.

    PRECONDITION, not asserted here and not automatic: the CP ranks' key sets must be DISJOINT. A key
    scored on two ranks is counted twice by the ReduceScatter that follows, and a softmax weight
    cannot be un-mixed afterwards.

    Args:
        ctx_local (nl.NkiTensor): [T, Nh, C] this rank's partial context, already normalized by its own
            local denominator -- which is what makes the weight a plain softmax over ranks.
        lse_local (nl.NkiTensor): [T, Nh] fp32 this rank's own log-sum-exp.
        lse_all (nl.NkiTensor): [cp_degree * T, Nh] fp32, the AllGather of every rank's ``lse``.
        tokens (int): T.
        num_heads (int): Nh.
        latent_dim (int): C.
        cp_degree (int): Context-parallel degree.

    Returns:
        nl.NkiTensor: [Nh, T * C] this rank's context times its weight, in HBM, HEAD-MAJOR. A
        ReduceScatter(sum) over the CP group turns these into the combined context; the weights sum to
        1 by construction, so nothing further is needed.

        The head axis leads because that ReduceScatter both sums the CP partials AND scatters the HEAD
        axis, and an HBM collective only splits dim 0 (``_check_collective_dim``). Emitting head-major
        is therefore not a preference: it is the only layout in which one collective can do both. The
        weight is per ``(token, head)``, so nothing about the arithmetic above depends on which axis
        the following collective scatters -- only this final eviction changes.
    """
    T, Nh, C = tokens, num_heads, latent_dim

    """
    THE PARTITION AXIS IS THE LARGER OF Nh AND T.

    A Vector or Scalar instruction costs ``elements_per_partition`` cycles and is INDEPENDENT of how many
    partitions it uses, so the cheapest layout is the one that spreads the work over the most lanes. The
    widest tensor here is the context scale over ``Nh * T * C`` elements; putting the SHORTER axis on
    partitions leaves most lanes idle and multiplies the cost by the ratio. At ``head_shard = 1`` that is
    ``T = 16`` against ``Nh = 64``, so head-major runs 8192 elements deep on 64 lanes instead of 32768
    deep on 16 -- a quarter of the depth.

    It is also an SBUF constraint, not merely a cost one: the free width is ``other_axis * C``, and at
    ``head_shard = 8`` (``T = 128``, ``Nh = 8``) head-major would need 128 KB per partition for each of
    the two context tiles against a 240 KB budget -- it fails to compile with NCC_IBIR229. Choosing by
    size satisfies both at once.

    The arithmetic is identical either way -- every op is elementwise or a reduction over the CP axis,
    which stays innermost and contiguous -- so this is bit-exact whichever branch is taken.
    """
    head_major = Nh >= T
    rows, cols = (Nh, T) if head_major else (T, Nh)

    # lse_all is [cp, T, Nh] flattened; lse_local is [T, Nh]; ctx_local is [T, Nh, C]. Each pattern below
    # is just those strides read in the chosen partition order.
    lse_all_pat = [[1, Nh], [Nh, T], [T * Nh, cp_degree]] if head_major else [[Nh, T], [1, Nh], [T * Nh, cp_degree]]
    lse_own_pat = [[1, Nh], [Nh, T]] if head_major else [[Nh, T], [1, Nh]]
    ctx_pat = [[C, Nh], [Nh * C, T], [1, C]] if head_major else [[Nh * C, T], [C, Nh], [1, C]]

    lse_sb = nl.ndarray((P_MAX, cols, cp_degree), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=lse_sb[0:rows, 0:cols, 0:cp_degree], src=lse_all.ap(pattern=lse_all_pat, offset=0))
    own_sb = nl.ndarray((P_MAX, cols), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=own_sb[0:rows, 0:cols], src=lse_local.ap(pattern=lse_own_pat, offset=0))

    max_sb = nl.ndarray((P_MAX, cols, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(max_sb[0:rows, 0:cols, 0:1], op=nl.max, data=lse_sb[0:rows, 0:cols, 0:cp_degree], axis=2)

    shifted_sb = nl.ndarray((P_MAX, cols, cp_degree), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=shifted_sb[0:rows, 0:cols, 0:cp_degree],
        data1=lse_sb[0:rows, 0:cols, 0:cp_degree],
        data2=max_sb[0:rows, 0:cols, 0:1].broadcast(2, cp_degree),
        op=nl.subtract,
    )
    exp_sb = nl.ndarray((P_MAX, cols, cp_degree), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(exp_sb[0:rows, 0:cols, 0:cp_degree], op=nl.exp, data=shifted_sb[0:rows, 0:cols, 0:cp_degree])

    sum_sb = nl.ndarray((P_MAX, cols, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(sum_sb[0:rows, 0:cols, 0:1], op=nl.add, data=exp_sb[0:rows, 0:cols, 0:cp_degree], axis=2)
    recip_sb = nl.ndarray((P_MAX, cols, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(recip_sb[0:rows, 0:cols, 0:1], op=nl.reciprocal, data=sum_sb[0:rows, 0:cols, 0:1])

    # The numerator: exp(my own lse - m). Every argument is <= 0, so this can only underflow.
    own_shift_sb = nl.ndarray((P_MAX, cols), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=own_shift_sb[0:rows, 0:cols], data1=own_sb[0:rows, 0:cols], data2=max_sb[0:rows, 0:cols, 0], op=nl.subtract
    )
    weight_sb = nl.ndarray((P_MAX, cols, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(weight_sb[0:rows, 0:cols, 0], op=nl.exp, data=own_shift_sb[0:rows, 0:cols])
    nisa.tensor_tensor(
        dst=weight_sb[0:rows, 0:cols, 0:1],
        data1=weight_sb[0:rows, 0:cols, 0:1],
        data2=recip_sb[0:rows, 0:cols, 0:1],
        op=nl.multiply,
    )

    # The innermost run of both context patterns is C elements -- 1 KB descriptors in bfloat16, and the
    # runs walk HBM forwards, which is the good case rather than the strided one.
    ctx_sb = nl.ndarray((P_MAX, cols, C), dtype=ctx_local.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=ctx_sb[0:rows, 0:cols, 0:C], src=ctx_local.ap(pattern=ctx_pat, offset=0))
    scaled_sb = nl.ndarray((P_MAX, cols, C), dtype=ctx_local.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=scaled_sb[0:rows, 0:cols, 0:C],
        data1=ctx_sb[0:rows, 0:cols, 0:C],
        data2=weight_sb[0:rows, 0:cols, 0:1].broadcast(2, C),
        op=nl.multiply,
    )

    # The result must be HEAD-MAJOR whichever way it was computed: the ReduceScatter that follows scatters
    # the HEAD axis and an HBM collective splits dim 0 only. Head-major already is one contiguous copy;
    # token-major needs one strided DMA, writing element (t, h, c) at h*(T*C) + t*C + c.
    scaled = nl.ndarray(
        (Nh, T * C), dtype=ctx_local.dtype, buffer=_COLLECTIVE_HBM, name=unique_name("scaled_ctx_head_major")
    )
    if head_major:
        nisa.dma_copy(dst=scaled[0:Nh, 0 : T * C], src=scaled_sb[0:Nh, 0:T, 0:C])
    else:
        nisa.dma_copy(dst=scaled.ap(pattern=[[C, T], [T * C, Nh], [1, C]], offset=0), src=scaled_sb[0:T, 0:Nh, 0:C])
    return scaled


class _ShapeOnly:
    """A stand-in for a tensor whose shape and dtype a callee checks but whose data no one reads (``kv_free``)."""

    def __init__(self, shape: tuple, dtype):
        self.shape, self.dtype = tuple(shape), dtype


def _deliver_attn_out(attn_out: nl.NkiTensor, summed_hbm: nl.NkiTensor) -> None:
    """Move the exit ReduceScatter's result into the layer's shared_hbm output.

    A collective's destination cannot be a kernel IO tensor, so the output is reached through an SBUF bounce
    (two DMAs).

    Args:
        attn_out (nl.NkiTensor): [T_shard, H] the output, in shared HBM.
        summed_hbm (nl.NkiTensor): [T_shard, H] the collective's destination in private HBM.
    """
    out_sb = nl.ndarray(tuple(attn_out.shape), dtype=attn_out.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=out_sb, src=summed_hbm)
    nisa.dma_copy(dst=attn_out, src=out_sb)


def _scatter_rows(
    src_hbm: nl.NkiTensor,
    replica_group: ReplicaGroup,
    degree: int,
    rows_out: int,
    name: str,
) -> nl.NkiTensor:
    """Hand each rank of ``replica_group`` its own ``rows_out`` rows of an IDENTICAL tensor.

    A ReduceScatter with ``op=maximum`` over operands every group member already holds identically
    reduces to a pure scatter: group position ``s`` receives rows ``[rows_out*s, rows_out*(s+1))``
    bit-exactly, with no arithmetic and no per-rank offset. That last part is the whole point -- the
    kernel is compiled once from rank 0's inputs, so a compile-time row offset would be rank 0's on
    every rank (see the module docstring).

    ``op=add`` would be wrong, not merely slower: summing ``degree`` identical bf16 values and dividing
    back is inexact (3x, 5x, 6x ... each round), which perturbs the indexer's scores at the size of its
    own tie margin.

    Args:
        src_hbm (nl.NkiTensor): [rows_out * degree, W] in HBM, IDENTICAL on every group member.
        replica_group (ReplicaGroup): The group, in increasing row-block order.
        degree (int): Group size. 1 short-circuits to handing ``src_hbm`` through -- a one-rank
            collective would be a no-op with a launch cost, and the degenerate case is real (a rank
            whose token slice is exactly one sequence).
        rows_out (int): Rows this rank receives.
        name (str): Debug name.

    Returns:
        nl.NkiTensor: [rows_out, W] in HBM.
    """
    width = src_hbm.shape[1]
    if degree == 1:
        # Every rank of a one-rank group already holds exactly its rows: hand the source through. The
        # copy (HBM -> HBM, one DMA latency on the indexer's critical path) would buy
        # nothing but a fresh buffer.
        kernel_assert(src_hbm.shape[0] == rows_out, f"{_ERROR_PREFIX} {name}: degree 1 needs rows_out == rows")
        return src_hbm
    dst = cc_dst((rows_out, width), src_hbm.dtype, _COLLECTIVE_HBM, name, reduce_scatter=True)
    _pin_all(src_hbm, dst)  # while pinning is on; see utils/cc_pin.py
    ncc.reduce_scatter(srcs=[src_hbm], dsts=[dst], op=nl.maximum, replica_group=replica_group, collective_dim=0)
    return dst


def _gather_rows(
    src_hbm: nl.NkiTensor,
    replica_group: ReplicaGroup,
    degree: int,
    name: str,
    handle_out: list = None,
) -> nl.NkiTensor:
    """AllGather on dim 0, with the degenerate one-rank case short-circuited to the source itself.

    Args:
        src_hbm (nl.NkiTensor): [rows, W] this rank's block in HBM.
        replica_group (ReplicaGroup): The group, in increasing block order.
        degree (int): Group size. 1 returns ``src_hbm`` unchanged.
        name (str): Debug name.
        handle_out (list): Optional; receives the AllGather instruction (nothing for a one-rank group).

    Returns:
        nl.NkiTensor: [rows * degree, W] in HBM.
    """
    rows, width = src_hbm.shape
    if degree == 1:
        # A one-rank gather is the identity: hand the source through rather than copy it (see _scatter_rows).
        return src_hbm
    dst = cc_dst((rows * degree, width), src_hbm.dtype, _COLLECTIVE_HBM, name)
    _pin_all(src_hbm, dst)  # while pinning is on; see utils/cc_pin.py
    cc = ncc.all_gather(srcs=[src_hbm], dsts=[dst], replica_group=replica_group, collective_dim=0)
    if handle_out is not None:
        handle_out.append(cc)
    return dst


def select_rows(src_hbm: nl.NkiTensor, rows: nl.NkiTensor, n: int, name: str, whole: bool = False) -> nl.NkiTensor:
    """Rows ``rows[0:n]`` of a ``[R, W]`` HBM tensor, by one indirect DMA: a RUNTIME row choice.

    The row indices are a tensor because the kernel is compiled once for every rank (see the module
    docstring). ``src_hbm`` is first copied whole into a fresh tensor, so a sliced view (``cos_all[i]``)
    reaches the indirect DMA with offset 0. Only meant for small tensors (a token slice).

    Args:
        src_hbm (nl.NkiTensor): [R, W] in HBM, R <= 128.
        rows (nl.NkiTensor): [n, 1] int32 in HBM, each in ``[0, R)``.
        n (int): Rows to pick, at most 128.
        name (str): Debug name stem.
        whole (bool): ``src_hbm`` is a whole tensor (offset 0), so the copy is skipped.

    Returns:
        nl.NkiTensor: [n, W] private HBM, row ``k`` = ``src_hbm[rows[k]]``.
    """
    R, W = src_hbm.shape
    kernel_assert(R <= P_MAX and n <= P_MAX, f"{_ERROR_PREFIX} select_rows takes at most {P_MAX} rows, got {R} -> {n}")
    if whole:
        whole_hbm = src_hbm
    else:
        whole_sb = nl.ndarray((P_MAX, W), dtype=src_hbm.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=whole_sb[0:R, 0:W], src=src_hbm[0:R, 0:W])
        whole_hbm = nl.ndarray((R, W), dtype=src_hbm.dtype, buffer=_COLLECTIVE_HBM, name=unique_name(f"{name}_src"))
        nisa.dma_copy(dst=whole_hbm[0:R, 0:W], src=whole_sb[0:R, 0:W])
    rows_sb = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=rows_sb[0:n, 0:1], src=rows[0:n, 0:1])
    picked_sb = nl.ndarray((P_MAX, W), dtype=src_hbm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=picked_sb[0:n, 0:W],
        src=whole_hbm.ap(
            pattern=[[W, n], [1, W]],
            offset=0,
            vector_offset=rows_sb.ap(pattern=[[1, n], [1, 1]], offset=0),
            indirect_dim=0,
        ),
    )
    out = nl.ndarray((n, W), dtype=src_hbm.dtype, buffer=_COLLECTIVE_HBM, name=unique_name(name))
    nisa.dma_copy(dst=out[0:n, 0:W], src=picked_sb[0:n, 0:W])
    return out


def _exit_to_rows(
    partial: nl.NkiTensor, exit_rows: nl.NkiTensor, n_seq: int, rows_replica_group: ReplicaGroup
) -> nl.NkiTensor:
    """[n_seq, H] attention rows, the same on every rank, from this rank's head-partial exit rows.

    ``partial`` holds this rank's head share of ``o_proj`` for its exit rows (the block's, or the token
    slice's); they go into a zero ``[n_seq, H]`` table at ``exit_rows`` and one ``all_reduce(add)`` over every
    rank sums the head shares of each row -- and only its own group writes a row -- so no ReduceScatter
    and no later gather are needed.
    """
    n_exit, H = partial.shape
    table = nl.ndarray((n_seq, H), dtype=partial.dtype, buffer=_COLLECTIVE_HBM, name=unique_name("attn_rows_in"))
    zero_sb = nl.ndarray((P_MAX, H), dtype=partial.dtype, buffer=nl.sbuf)
    nisa.memset(dst=zero_sb, value=0.0)
    nisa.dma_copy(dst=table[0:n_seq, 0:H], src=zero_sb[0:n_seq, 0:H])
    rows_sb = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=rows_sb[0:n_exit, 0:1], src=exit_rows[0:n_exit, 0:1])
    part_sb = nl.ndarray((P_MAX, H), dtype=partial.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=part_sb[0:n_exit, 0:H], src=partial[0:n_exit, 0:H])
    nisa.dma_copy(
        dst=table.ap(
            pattern=[[H, n_exit], [1, H]],
            offset=0,
            vector_offset=rows_sb.ap(pattern=[[1, n_exit], [1, 1]], offset=0),
            indirect_dim=0,
        ),
        src=part_sb[0:n_exit, 0:H],
    )
    rows = nl.ndarray((n_seq, H), dtype=partial.dtype, buffer=_COLLECTIVE_HBM, name=unique_name("attn_rows"))
    _pin_all(table, rows)  # while pinning is on; see utils/cc_pin.py
    ncc.all_reduce(srcs=[table], dsts=[rows], replica_group=rows_replica_group, op=nl.add)
    return rows


def _exit_rows_table(state: dict, exit_rows: nl.NkiTensor, n_seq: int, H: int, dtype) -> tuple:
    """The ``rows_table`` of :func:`mla_decode_out_proj_sbuf` for a rows-layout exit: ONE ``[n_seq, H]`` table per chain.

    Every step's ``o_proj`` writes the same ``exit_rows`` (they depend on the rank only) and nothing else writes
    the table, so the zero fill of :func:`_exit_to_rows` is needed once: the other rows stay zero, and the
    ``all_reduce`` input is the same bytes as with a fresh table. ``state`` is trace-time memory shared by the
    steps (the caller passes the same dict to each); ``state["rezero"]`` zeroes it again on every step.

    Returns:
        tuple: ``(table [n_seq, H] private HBM, offsets [P_MAX, 1] uint32 SBUF)``.
    """
    first = "table" not in state
    if first:
        state["table"] = nl.ndarray((n_seq, H), dtype=dtype, buffer=_COLLECTIVE_HBM, name=unique_name("attn_rows_in"))
        state["offsets"] = o_proj_rows_offsets(exit_rows, n_seq, H)
    table = state["table"]
    if first or state.get("rezero", False):
        zero_sb = nl.ndarray((P_MAX, H), dtype=dtype, buffer=nl.sbuf)
        nisa.memset(dst=zero_sb, value=0.0)
        nisa.dma_copy(dst=table[0:n_seq, 0:H], src=zero_sb[0:n_seq, 0:H])
    return table, state["offsets"]


def _expand_draft_rows(src_hbm: nl.NkiTensor, draft_rows: nl.NkiTensor, rows_out: int, name: str) -> nl.NkiTensor:
    """``[T_q, W]`` draft-row results into a zero ``[rows_out, W]``: row ``draft_rows[k]`` gets row ``k``.

    The CP combine and the exit then run on the full token slice unchanged, and a row that had no query
    contributes an exact zero (ctx 0 and lse 0 give the combine finite weights times zero).
    """
    T_q, W = src_hbm.shape
    chunk = min(W, 8192)
    out = nl.ndarray((rows_out, W), dtype=src_hbm.dtype, buffer=_COLLECTIVE_HBM, name=unique_name(name))
    zero_sb = nl.ndarray((P_MAX, chunk), dtype=src_hbm.dtype, buffer=nl.sbuf)
    nisa.memset(dst=zero_sb, value=0.0)
    rows_sb = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=rows_sb[0:T_q, 0:1], src=draft_rows[0:T_q, 0:1])
    rows_ap = rows_sb.ap(pattern=[[1, T_q], [1, 1]], offset=0)
    for c0 in range(0, W, chunk):
        cw = min(chunk, W - c0)
        nisa.dma_copy(dst=out[0:rows_out, c0 : c0 + cw], src=zero_sb[0:rows_out, 0:cw])
    for c0 in range(0, W, chunk):
        cw = min(chunk, W - c0)
        val_sb = nl.ndarray((P_MAX, chunk), dtype=src_hbm.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=val_sb[0:T_q, 0:cw], src=src_hbm[0:T_q, c0 : c0 + cw])
        nisa.dma_copy(
            dst=out.ap(pattern=[[W, T_q], [1, cw]], offset=c0, vector_offset=rows_ap, indirect_dim=0),
            src=val_sb[0:T_q, 0:cw],
        )
    return out


def _expand_draft_cells(
    cells_q: nl.NkiTensor, draft_rows: nl.NkiTensor, num_heads: int, T_q: int, T_local: int, name: str
) -> nl.NkiTensor:
    """The fused combine's head-major payload, from ``T_q`` draft tokens to ``T_local`` tokens.

    Row ``n * T_q + k`` (head ``n`` of draft token ``k``) moves to row ``n * T_local + draft_rows[k]``; every
    other row is zero, so a non-draft token's context is zero and its lse bits read as 0.0. The cells are
    moved by DMA only, as the core writes them, so the lse bits stay raw.
    """
    cell = cells_q.shape[1]
    kernel_assert(num_heads <= P_MAX, f"{_ERROR_PREFIX} draft cells need num_heads <= {P_MAX}, got {num_heads}")
    rows_full = num_heads * T_local
    out = nl.ndarray((rows_full, cell), dtype=cells_q.dtype, buffer=_COLLECTIVE_HBM, name=unique_name(name))
    zero_sb = nl.ndarray((P_MAX, cell), dtype=cells_q.dtype, buffer=nl.sbuf)
    nisa.memset(dst=zero_sb, value=0.0)
    for r0 in range(0, rows_full, P_MAX):
        n = min(P_MAX, rows_full - r0)
        nisa.dma_copy(dst=out[r0 : r0 + n, 0:cell], src=zero_sb[0:n, 0:cell])
    for k in range(T_q):
        # Destination row of head n: n * T_local + draft_rows[k], one per partition.
        idx_sb = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(idx_sb[0:num_heads, 0:1], pattern=[[0, 1]], offset=0, channel_multiplier=T_local)
        row_sb = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=row_sb[0:num_heads, 0:1], src=draft_rows[k : k + 1, 0:1].broadcast(0, num_heads))
        nisa.tensor_tensor(
            dst=idx_sb[0:num_heads, 0:1], data1=idx_sb[0:num_heads, 0:1], data2=row_sb[0:num_heads, 0:1], op=nl.add
        )
        cells_sb = nl.ndarray((P_MAX, cell), dtype=cells_q.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=cells_sb[0:num_heads, 0:cell],
            src=cells_q.ap(pattern=[[T_q * cell, num_heads], [1, cell]], offset=k * cell),
        )
        nisa.dma_copy(
            dst=out.ap(
                pattern=[[cell, num_heads], [1, cell]],
                offset=0,
                vector_offset=idx_sb.ap(pattern=[[1, num_heads], [1, 1]], offset=0),
                indirect_dim=0,
            ),
            src=cells_sb[0:num_heads, 0:cell],
        )
    return out


@nki.jit
def glm52_dsa_attention_decode_layer_kernel(
    # ── input: this rank's 128-way token shard of the residual stream ──
    x_shard: nl.NkiTensor,  # [T_shard, H]
    input_norm_w: nl.NkiTensor,  # [1, H] input_layernorm weight
    # ── MLA projections ──
    wq_a: nl.NkiTensor,  # [H, q_lora_shard]   column shard over the TOKEN group
    wkv_a: nl.NkiTensor,  # [H, kv_row_shard]   column shard, columns ordered [k_pe | kv_latent]
    q_norm_w: nl.NkiTensor,  # [1, q_lora_rank]
    kv_norm_w: nl.NkiTensor,  # [1, C]
    wq_b: nl.NkiTensor,  # [q_lora_rank, Nh * (nope + rope) / q_degree]  COLUMNS over the token group, or the block group
    q_absorb_w: nl.NkiTensor,  # [Nh / q_degree, nope, C]         heads of this rank's wq_b columns
    out_absorb_w: nl.NkiTensor,  # [C, Nh / cp_degree, v_head]         HEAD shard x CP, latent-major
    o_proj_w: nl.NkiTensor,  # [(Nh / cp_degree) * v_head, H]      HEAD shard x CP
    cos: nl.NkiTensor,  # [T_local, rope // 2] fp32, TRUE positions
    sin: nl.NkiTensor,  # [T_local, rope // 2] fp32
    # ── DSA indexer ──
    idx_wq_b: nl.NkiTensor,  # [q_lora_rank, idx_q_shard]  column shard over the key group
    idx_wk: nl.NkiTensor,  # [H, idx_head_dim]           REPLICATED
    idx_k_norm_w: nl.NkiTensor,  # [1, idx_head_dim] fp32
    idx_k_norm_b: nl.NkiTensor,  # [1, idx_head_dim] fp32
    idx_weights_proj: nl.NkiTensor,  # [H, idx_n_heads]
    idx_cos: nl.NkiTensor,  # [spec, idx_rope // 2] fp32, this rank's sequence
    idx_sin: nl.NkiTensor,  # [spec, idx_rope // 2] fp32
    # ── caches and per-step runtime tensors ──
    kv_cache: nl.NkiTensor,  # [num_blocks, 1, block_size, 640] uint8, MUTATED
    kv_new_slot_mapping: nl.NkiTensor,  # [num_seqs_local, spec] int32
    block_table: nl.NkiTensor,  # [num_seqs_local, local_blocks + 1] int32, column 0 unused
    positions: nl.NkiTensor,  # [T_local] int32, CP-SHIFTED
    idx_k_cache: nl.NkiTensor,  # [n_blocks, 1, idx_block_size, idx_head_dim], MUTATED; unpaged is one block
    idx_new_slot_mapping: nl.NkiTensor,  # [spec] int32
    idx_positions: nl.NkiTensor,  # [spec] int32, KEY-SHIFTED
    # ── the two per-rank offsets, tensors because the kernel is compiled once for all ranks ──
    key_offset: nl.NkiTensor,  # [1, 1] int32, key_rank * idx_keys
    cp_pos_offset: nl.NkiTensor,  # [1, 1] int32, (cp_degree - cp_rank) * block_size
    # ── replica groups ──
    token_replica_group: ReplicaGroup,
    cp_replica_group: ReplicaGroup,
    head_replica_group: ReplicaGroup,
    seq_replica_group: ReplicaGroup,
    key_replica_group: ReplicaGroup,
    # ── degrees, all rank-invariant ──
    cp_degree: int,
    head_shard: int,
    token_shard: int,
    key_size: int,
    # ── model scalars ──
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
    idx_n_heads: int,
    idx_rope_dim: int,
    index_topk: int,
    spec: int,
    softmax_scale: float,
    kv_scale: float,
    input_norm_eps: float,
    latent_norm_eps: float,
    idx_norm_eps: float,
    block_replica_group: ReplicaGroup = None,  # the o_proj BLOCK group: cp_degree x slices consecutive ranks; block-sharded o_proj only
    qblock_replica_group: ReplicaGroup = None,  # the query path's block group: token_group x q_slices consecutive ranks; block-sharded query path only
    prev_gather_indices: nl.NkiTensor = None,  # [T_local, index_topk] int32 TRUE positions, or None
    prior_snake_rows: nl.NkiTensor = None,  # [P_MAX, T_local, index_topk / P_MAX] uint32, or None
    prior_key_pos: nl.NkiTensor = None,  # [P_MAX, T_local, index_topk / P_MAX] fp32, or None
    prior_natural_rows: nl.NkiTensor = None,  # [P_MAX, T_local, index_topk / P_MAX] uint32, or None
    idx_mask_add: nl.NkiTensor = None,  # [128, idx_keys / 128 * spec] fp32 key-major additive residency mask, or None to build it here
    idx_block_table: nl.NkiTensor = None,  # [1, idx_keys / idx_block_size + 1] int32 logical->physical, or None for a position-ordered cache
    idx_block_rows: nl.NkiTensor = None,  # [128, idx_keys / 128] uint32 from indexer_block_rows, or None to build it from idx_block_table
    attend_new_keys: bool = True,  # False: KVShare -- score only the gathered prior selection, never this step's own appended rows
    x_shard_sb: nl.NkiTensor = None,  # optional SBUF copy of x_shard in the spread_rows_shape layout, skips the input norm's load
    input_norm_reduce_ones: nl.NkiTensor = None,  # optional build_norm_reduce_ones(T_shard, H) result, built once by a stack
    prior_tables_out: list = None,  # optional: a FULL layer appends the (snake, key_pos, natural) tables it resolved, for the shared layers
    draft_rows: nl.NkiTensor = None,  # [num_seqs_local, 1] int32: a KVShare step queries only these rows of the token slice
    rows_in: nl.NkiTensor = None,  # [n_seq, H] one row per sequence, the same on every rank; replaces x_shard (needs draft_rows)
    rows_seq: nl.NkiTensor = None,  # [num_seqs_local, 1] int32 the rows_in row of each of this token slice's sequences
    rows_exit_seq: nl.NkiTensor = None,  # [n_exit, 1] int32 the rows of the exit's tokens (the block's, or the slice's)
    rows_replica_group: ReplicaGroup = None,  # every rank: the exit all_reduce that completes the attention rows
    exit_table: dict = None,  # optional, rows layout: trace-time state of a kernel-lifetime exit rows table, see _exit_rows_table
    front_pick: dict = None,  # optional, rows layout, one sequence per slice: trace-time state, see _draft_row_k_tiles
    rows_in_groups: int = 1,  # with front_pick: rows_in is a G-way column AllGather's [G * n_seq, H / G] rank-major output
    kv_free: int = 0,  # draft_rows only: 1 = no K/V path at all (see mla_decode_attention), 2 = also no sibling key tile
    early_keys: bool = False,  # with kv_free=2: gather the shared selection's keys right after the down-projection
    idx_draft_rows: nl.NkiTensor = None,  # [1, 1] int32, the draft row of the slice's sequence
    idx_shard_degree: int = 1,  # degree the indexer's keys are DEALT over; 1 = its own contiguous slice
    idx_block_size: int = 0,  # the indexer cache's block size; also the deal unit under a shared table
):
    """One rank's whole GLM-5.2 DSA attention layer, collectives included (see the module docstring).

    Dimensions:
        T_shard: Token slots this rank owns on entry and exit
        T_local: Token slots this rank's MLA scores (``T_shard * cp_degree * head_shard``)
        H: Hidden dimension
        C: KV latent rank (``kv_lora_rank``)
        Nh: Query heads this rank ATTENDS over (``64 / head_shard``)
        Nh_out: Query heads this rank PROJECTS through ``o_proj`` (``Nh / cp_degree``)

    Args:
        x_shard (nl.NkiTensor): [T_shard, H] this rank's slots of the pre-attention residual stream.
        input_norm_w (nl.NkiTensor): [1, H] fp32 ``input_layernorm`` weight, applied here.
        wq_a (nl.NkiTensor): [H, q_lora_shard] this rank's COLUMNS of the query down-projection. The
            column-shard degree is the TOKEN group's size, not the whole DP group's: a column shard
            requires its participants to hold the same tokens, and under token sharding only the token
            group does.
        wkv_a (nl.NkiTensor): [H, kv_row_shard] this rank's COLUMNS of the KV down-projection. Its
            columns must be permuted to ``[k_pe | kv_latent]``; the reference's
            ``kv_a_proj_with_mqa`` emits the inverse order.
        q_norm_w (nl.NkiTensor): [1, q_lora_rank] fp32 ``q_a_layernorm`` weight.
        kv_norm_w (nl.NkiTensor): [1, C] fp32 ``kv_a_layernorm`` weight, latent field only.
        wq_b (nl.NkiTensor): [q_lora_rank, Nh * (nope + rope) / q_degree] this rank's COLUMNS of the
            query up-projection -- a whole number of heads. ``q_degree = token_group`` shards it over the
            TOKEN group (whose members hold the same tokens; the absorbed query is AllGathered back inside
            ``mla_decode_q_kv_up``). ``q_degree = token_group * slices`` shards it over the BLOCK group
            (``qblock_replica_group``): the block's latents are gathered first, every rank projects and
            absorbs its heads for all the block's tokens, and one ``all_to_all`` hands each rank its own
            tokens. Either shard is on the output, not the contraction, so it is bit-exact.
        q_absorb_w (nl.NkiTensor): [Nh / q_degree, nope, C] the nope half of ``wkv_b`` for the heads whose
            ``wq_b`` columns this rank holds: ``mla_decode_q_kv_up`` absorbs those heads before the
            collective, instead of gathering the raw query and absorbing all ``Nh`` heads on every rank.
        out_absorb_w (nl.NkiTensor): [C, nh_out, v_head] (latent-major) the value half of ``wkv_b`` for
            this rank's OUTPUT heads. ``nh_out = Nh / cp_degree`` shards the output path over the CP
            group: the rank attends over all ``Nh`` heads but projects only the slice its CP group
            position receives from the context ReduceScatter. ``nh_out = Nh / oproj_tp`` with
            ``oproj_tp`` a multiple of ``cp_degree`` shards it over the block group instead
            (``block_replica_group``): position ``d`` projects heads ``[d * nh_out, (d + 1) * nh_out)``.
        o_proj_w (nl.NkiTensor): [nh_out * v_head, H] this rank's ROWS of the output projection, for
            the same head slice. Always a PARTIAL sum over heads -- the exit ReduceScatter completes
            it -- which is what cuts the weight from 201 MB to ``201 / (Nh / nh_out)`` MB per rank.
        cos (nl.NkiTensor): [T_local, rope // 2] fp32 interleaved RoPE cosine, TRUE positions.
        sin (nl.NkiTensor): [T_local, rope // 2] fp32 interleaved RoPE sine.
        idx_wq_b (nl.NkiTensor): [q_lora_rank, idx_q_shard] this rank's indexer query columns.
        idx_wk (nl.NkiTensor): [H, idx_head_dim] the indexer key projection, REPLICATED.
        idx_k_norm_w (nl.NkiTensor): [1, idx_head_dim] fp32 key LayerNorm gain.
        idx_k_norm_b (nl.NkiTensor): [1, idx_head_dim] fp32 key LayerNorm bias.
        idx_weights_proj (nl.NkiTensor): [H, idx_n_heads] per-head mixing weights.
        idx_cos (nl.NkiTensor): [spec, idx_rope // 2] fp32 cosine for THIS rank's sequence.
        idx_sin (nl.NkiTensor): [spec, idx_rope // 2] fp32 sine for this rank's sequence.
        kv_cache (nl.NkiTensor): [num_blocks, 1, block_size, 640] uint8 paged latent cache covering
            only this rank's OWN sequences -- which is the memory win of token sharding, and the
            reason head sharding was pure replication: the latent cache is head-independent, so the
            ranks of a head group held byte-identical copies.
        kv_new_slot_mapping (nl.NkiTensor): [num_seqs_local, spec] int32 destination rows, negative
            to skip.
        block_table (nl.NkiTensor): [num_seqs_local, local_blocks + 1] int32; column 0 is never a real
            block and must be negative (a consequence of the CP position shift).
        positions (nl.NkiTensor): [T_local] int32, absolute position PLUS
            ``(cp_degree - cp_rank) * block_size``.
        idx_k_cache (nl.NkiTensor): [n_blocks, 1, idx_block_size, idx_head_dim] this rank's key range,
            MUTATED. The legacy flat [idx_keys, idx_head_dim] is accepted with ``idx_shard_degree == 1``.
        idx_new_slot_mapping (nl.NkiTensor): [spec] int32 indexer-cache rows, negative to skip.
        idx_positions (nl.NkiTensor): [spec] int32. Positions MINUS ``key_rank * idx_keys`` under a
            CONTIGUOUS deal, TRUE positions under an interleaved one.
        key_offset (nl.NkiTensor): [1, 1] int32, the RANK TERM of the local-to-global position map:
            ``key_rank * idx_keys`` contiguous, ``key_rank * idx_block_size`` interleaved.
        idx_shard_degree (int): Degree the indexer's keys are DEALT over; 1 keeps its contiguous slice.
            ``cp_degree`` gives it MLA's rule, which is what lets ONE block table serve both caches.
            See ``indexer_global_positions_np``.
        idx_block_size (int): The indexer cache's block size, and the unit the keys are dealt in under a
            shared table, where it must equal MLA's ``block_size``.
        cp_pos_offset (nl.NkiTensor): [1, 1] int32 ``(cp_degree - cp_rank) * block_size``.
        token_replica_group (ReplicaGroup): The ranks sharing this rank's TOKEN slice, in increasing
            column order. Carries the entry hidden gather and the fused latent gather.
        cp_replica_group (ReplicaGroup): ``cp_degree`` ranks, spaced ``head_shard`` apart within a
            token group. Carries the log-sum-exp gather and BOTH exit ReduceScatters -- the first sums
            the key partials while scattering heads, the second sums the head partials while scattering
            tokens.
        head_replica_group (ReplicaGroup): ``head_shard`` consecutive ranks. Carries the final
            ReduceScatter, and is unused when ``head_shard == 1``.
        block_replica_group (ReplicaGroup): The o_proj BLOCK group: ``cp_degree * slices`` CONSECUTIVE ranks
            -- the CP group and its ``slices - 1`` neighbouring token slices, position ``cp_degree * (kk
            mod slices) + cp`` -- when the output path is block-sharded (``nh_out < Nh / cp_degree``);
            None otherwise. Carries the context ``all_to_all`` and the exit ReduceScatter. Contiguity is
            required: the runtime's ``all_to_all`` aborts on strided groups.
        qblock_replica_group (ReplicaGroup): The query path's block group: ``token_group * q_slices``
            CONSECUTIVE ranks, when ``wq_b`` is narrower than ``1 / token_group`` of the packed width; None
            otherwise. Carries the block latent AllGathers and the query ``all_to_all``. May differ in size
            from ``block_replica_group`` (e.g. TP8 query path with a TP16 o_proj).
        seq_replica_group (ReplicaGroup): The ranks of one token group sharing a key shard, in
            increasing sequence order. Size ``T_local / spec``. Carries BOTH the sequence scatter into
            the indexer and the selection gather back out -- the two are exact inverses, which is why
            the gather lands exactly ``T_local`` query rows.
        key_replica_group (ReplicaGroup): ``key_size`` ranks of one token group sharing a sequence, in
            increasing key-shard order. Carries the indexer query and candidate gathers.
        cp_degree (int): Context-parallel degree.
        head_shard (int): Head-shard degree. 1 is pure data parallelism.
        token_shard (int): Token-shard degree. ``cp_degree * head_shard * token_shard`` is the DP
            group size.
        key_size (int): Indexer key-shard degree.
        num_heads (int): Query heads on this rank, ``64 / head_shard``.
        qk_nope_head_dim (int): Per-head QK width RoPE does not rotate.
        qk_rope_head_dim (int): Per-head QK width RoPE rotates.
        kv_lora_rank (int): Latent width C.
        idx_n_heads (int): Indexer query heads. Not sharded; the head axis is reduced.
        idx_rope_dim (int): Indexer rotated width, the LEADING slice of each indexer head.
        index_topk (int): Keys the selector keeps per query token. A multiple of 128.
        spec (int): Token slots per sequence, PADDED (8 for GLM-5.2's speculation length 7).
        softmax_scale (float): ``qk_head_dim ** -0.5``; 0.0625 for GLM-5.2, no yarn mscale.
        kv_scale (float): Latent dequantization scale. A power of two.
        input_norm_eps (float): ``config.rms_norm_eps``, 1e-5. ``input_layernorm`` only.
        latent_norm_eps (float): Epsilon of ``q_a_layernorm`` and ``kv_a_layernorm``, 1e-6.
        idx_norm_eps (float): Epsilon of the indexer's ``k_norm``, 1e-6.
        prev_gather_indices (nl.NkiTensor): [T_local, index_topk] int32 in HBM, or None. This is the
            reference's ``prev_topk_indices``: on an ``indexer_types == "shared"`` layer the reference
            has no indexer submodule and reuses the nearest preceding ``"full"`` layer's selection
            unchanged. Passing it here selects that mode -- the four indexer sub-kernels and their four
            collectives are skipped, and ``idx_wq_b``, ``idx_wk``, ``idx_k_norm_w``, ``idx_k_norm_b``,
            ``idx_weights_proj``, ``idx_k_cache``, ``idx_new_slot_mapping``, ``idx_positions``,
            ``idx_cos``, ``idx_sin`` and ``key_offset`` are all unused and may be None, matching a
            checkpoint that ships no ``indexer.*`` tensors for such a layer. Positions must be TRUE
            global ones; the CP shift is applied here, so the value handed across layers is the same
            one ``gather_indices_out`` reports.
        attend_new_keys (bool): True (default) scores this step's own ``spec`` rows causally, as the
            decoder does. False is the GLM5.2 MTP ``KVShare`` rule for draft steps after the first: the
            queries see ONLY the gathered prior selection, never their own key. Meant for the shared
            mode, with a negative ``kv_new_slot_mapping`` so the cache is not written either.
        x_shard_sb (nl.NkiTensor): Optional SBUF copy of ``x_shard`` in the ``spread_rows_shape`` layout
            (what the giga stack's residual add leaves behind); the input norm then skips its load.
        input_norm_reduce_ones (nl.NkiTensor): Optional ``build_norm_reduce_ones(T_shard, H)`` result, so a
            stack of layers builds the input norm's reduction constant once instead of per layer.
        prior_tables_out (list): Optional. A FULL layer (one that runs its indexer) appends the prior-index
            tables its attention core resolved for the new selection, as the ``(prior_snake_rows,
            prior_key_pos, prior_natural_rows)`` triple in HBM the shared layers of its group take -- the
            tables ``build_shared_prior_index`` would otherwise rebuild from ``gather_indices_out``.

        draft_rows (nl.NkiTensor): Optional [num_seqs_local, 1] int32, the row of each sequence's draft
            token in this rank's ``T_local`` token slice. For an MTP KVShare step (a shared layer with
            ``attend_new_keys=False``), where only the draft row of a sequence feeds anything: the query
            projections and the attention core then run on those ``num_seqs_local`` rows (``spec`` 1)
            instead of all ``T_local``, and the core's output is expanded back to ``T_local`` rows with
            zeros before the CP combine and the exit, which are unchanged. Every other row's ``attn_out``
            is exactly zero. ``prev_gather_indices`` and the prior tables must then hold the draft rows
            only: ``[num_seqs_local, index_topk]`` and ``[P_MAX, num_seqs_local, index_topk / P_MAX]``
            (``build_shared_prior_index`` with ``T_local=num_seqs_local, spec=1``). ``cos``, ``sin``,
            ``positions`` and ``kv_new_slot_mapping`` stay full; the kernel picks the draft rows. Not for
            the block-sharded query path (``q_slices > 1``).
        rows_in (nl.NkiTensor): Optional [n_seq, H], with ``draft_rows``: the ROWS layout of an MTP draft
            step, one row per sequence (its draft row), the same on every rank, as a WHOLE HBM tensor (offset
            0; the row pick reads it by indirect DMA as is). ``x_shard`` is then None and
            unused. The layer norms rows ``rows_seq`` (this token slice's sequences) instead of gathering its
            token slice, runs the draft-row query path, and exits by writing its head-partial ``o_proj``
            rows at ``rows_exit_seq`` of a zero ``[n_seq, H]`` table and one ``all_reduce`` over
            ``rows_replica_group``: the return's ``attn_out`` is then ``[n_seq, H]``, every row's attention,
            on every rank. Needs ``head_shard == 1`` and, for a block-sharded o_proj, the fused combine.
        rows_seq, rows_exit_seq (nl.NkiTensor): [num_seqs_local, 1] and [n_exit, 1] int32, see ``rows_in``.
            ``n_exit`` is ``slices * num_seqs_local`` for a block-sharded o_proj (the block's sequences in
            slice order) and ``num_seqs_local`` otherwise.
        rows_replica_group (ReplicaGroup): Every rank, for the exit ``all_reduce``.
        exit_table (dict): Optional, with ``rows_in``, the fused combine and the on-chip context: the same dict on
            every step of a chain. ``o_proj`` then writes its rows straight from SBUF into ONE zero ``[n_seq, H]``
            table that the exit ``all_reduce`` reads, zeroed on first use (and on every step with
            ``exit_table["rezero"]``), instead of storing its output, zeroing a fresh table and moving the rows in
            through SBUF every step. See :func:`_exit_rows_table`.
        front_pick (dict): Optional, with ``rows_in`` and one sequence per token slice: the same dict on every step of a
            chain. The input norm then runs on two direct picks of the row from ``rows_in`` and hands the
            down-projection its K-tiles on chip (:func:`_draft_row_k_tiles`), instead of picking the row to HBM,
            norming it there and transposing it back in 48 pieces.
        rows_in_groups (int): With ``front_pick``: 1 (default) for an ``[n_seq, H]`` ``rows_in``; ``G`` > 1 for the
            rank-major destination of a ``G``-way column AllGather, ``[G * n_seq, H / G]`` with column block ``r`` of
            every row at rows ``[r n_seq, (r + 1) n_seq)``, which the picks read in place (no relayout to rows).
        kv_free (int): With ``draft_rows`` (KVShare, whose caller passes an all-negative ``kv_new_slot_mapping``):
            0 (default) runs the K/V path for nothing; 1 deletes it -- no ``wkv_a`` columns, no kv norm/RoPE, no
            ``kv_row_new`` or normed-latent store (``mla_decode_q_kv_up(q_only=True)``), and the core's quantize,
            cache write and sibling append (``mla_decode_attention(kv_free=)``) -- and gathers the query latent
            alone; 2 also drops the core's fully masked sibling key tile.
        early_keys (bool): With ``kv_free=2`` and the prior tables: trace the tables' loads and the prior-row
            gather (:func:`mla_decode_attention_prefetch`) right after the latent AllGather is issued instead of
            after the query AllGather, so the keys are on chip before the query is. (default: False)
        idx_draft_rows (nl.NkiTensor): Optional [1, 1] int32, a full layer in the chunk loop at spec 8: the row of
            the slice's one sequence's draft token. The indexer then scores the 4 query rows of that row's group only
            (see ``mtp_idx_query_tokens``), and ``idx_mask_add`` must be built for 4 tokens (``mtp_idx_query_tokens``). (default: None)

    Returns:
        Tuple[nl.NkiTensor, nl.NkiTensor, nl.NkiTensor, nl.NkiTensor]:
            attn_out: [T_shard, H] this rank's slots of the layer's attention output, complete.
            gather_indices_out: [num_seqs_local, spec, index_topk] int32, the selection for this
                rank's OWN sequences, in TRUE global positions. Bring-up observability. On a shared
                layer this is ``prev_gather_indices`` handed straight back, uncopied, shaped
                [T_local, index_topk] -- the same tensor the next shared layer consumes.
            kv_cache: the paged latent cache, returned so its mutation is part of the result.
            idx_k_cache: the indexer key cache, likewise. None on a shared layer, which has none.
    """
    rows = rows_in is not None
    hidden_k_tiles = None
    if rows:
        kernel_assert(
            rows_in_groups == 1 or (front_pick is not None and rows_in.shape[0] % rows_in_groups == 0),
            f"{_ERROR_PREFIX} rows_in_groups={rows_in_groups} needs front_pick and {rows_in.shape[0]} gathered rows "
            f"that divide by it",
        )
        T_shard, H = positions.shape[0] // (cp_degree * head_shard), rows_in.shape[1] * rows_in_groups
        rows_n_seq = rows_in.shape[0] // rows_in_groups
        io_dtype = rows_in.dtype
    else:
        T_shard, H = x_shard.shape
        io_dtype = x_shard.dtype
    token_group = cp_degree * head_shard  # ranks sharing this rank's token slice
    T_local = T_shard * token_group
    num_seqs_local = T_local // spec
    latent_dim = kv_lora_rank
    q_lora_shard, kv_row_shard = wq_a.shape[1], wkv_a.shape[1]
    q_lora_rank = q_lora_shard * token_group
    # The MTP layer keeps the TOKEN-GROUP down-projection shard: the latent-sharded down-projection (latent_tp)
    # is the target's alone (attention_layer.py), and this fork has no latent exchange.
    kernel_assert(
        q_norm_w.shape[1] == q_lora_rank,
        f"{_ERROR_PREFIX} wq_a holds {q_lora_shard} of the {q_norm_w.shape[1]} query-latent columns: the MTP layer's "
        f"wq_a / wkv_a are column-sharded over the token group ({token_group} ranks) only; a latent-sharded "
        f"down-projection (latent_tp) is the target's",
    )
    # A "shared" layer (``config.indexer_types[layer_idx] == "shared"``) has no indexer submodule at
    # all in the reference, so it carries no indexer weights and no indexer key cache -- every one of
    # those arguments is None and the selection arrives as ``prev_gather_indices`` instead. The branch
    # is on a Python value that is identical on all 128 ranks, so it constant-folds safely.
    run_indexer = prev_gather_indices is None
    idx_q_shard = idx_wq_b.shape[1] if run_indexer else 0
    k_cols = index_topk // _SNAKE_PARTITIONS
    # Heads this rank projects THROUGH o_proj, as opposed to the num_heads it attends over. The output
    # path is head-sharded a further cp_degree ways on top of head_shard, so it is read off the weight
    # rather than derived -- the weight is the authority on what the caller actually handed us.
    nh_out = out_absorb_w.shape[1]

    _validate_attention_layer(
        T_shard=T_shard,
        T_local=T_local,
        num_seqs_local=num_seqs_local,
        spec=spec,
        cp_degree=cp_degree,
        head_shard=head_shard,
        token_shard=token_shard,
        key_size=key_size,
        index_topk=index_topk,
        idx_q_shard=idx_q_shard,
        idx_n_heads=idx_n_heads,
        idx_head_dim=idx_wk.shape[1] if run_indexer else 0,
        positions_len=positions.shape[0],
        num_heads=num_heads,
        nh_out=nh_out,
        o_proj_rows=o_proj_w.shape[0],
        v_head_dim=out_absorb_w.shape[2],
        token_replica_group=token_replica_group,
        head_replica_group=head_replica_group,
        key_replica_group=key_replica_group,
        run_indexer=run_indexer,
    )

    # ═══════ C1: input norm on this rank's rows, then rebuild THIS RANK'S TOKEN SLICE ═══════
    # Over the token group, not the whole DP group: a rank only needs the tokens it will score, and
    # under token sharding that is T_local rather than every slot in the group.
    if rows:
        kernel_assert(
            draft_rows is not None
            and rows_seq is not None
            and rows_exit_seq is not None
            and rows_replica_group is not None
            and head_shard == 1,
            f"{_ERROR_PREFIX} rows_in needs draft_rows, rows_seq, rows_exit_seq, rows_replica_group and "
            f"head_shard == 1 (got {head_shard})",
        )
        # The draft rows of this slice's sequences are already on this rank: norm them, no gather.
        if front_pick is not None:
            kernel_assert(num_seqs_local == 1, f"{_ERROR_PREFIX} front_pick needs one sequence per token slice")
            hidden_k_tiles = _draft_row_k_tiles(
                rows_in, rows_seq, input_norm_w, input_norm_eps, front_pick, groups=rows_in_groups
            )
            hidden = _ShapeOnly((1, H), io_dtype)
        else:
            hidden = _rmsnorm_shard(
                select_rows(rows_in, rows_seq, num_seqs_local, "x_rows_slice", whole=True),
                input_norm_w,
                input_norm_eps,
            )
    else:
        hidden = _gather_rows(
            _rmsnorm_shard(x_shard, input_norm_w, input_norm_eps, x_sb=x_shard_sb, reduce_ones=input_norm_reduce_ones),
            token_replica_group,
            token_group,
            "hidden_all",
        )

    # ═══════ A KVShare draft step: only each sequence's draft row needs a query ═══════
    # T_q / spec_q are the query side's extents; the K/V side writes nothing and the exit keeps T_local.
    draft = draft_rows is not None
    T_q, spec_q = T_local, spec
    positions_q, cos_q, sin_q, slot_mapping_q = positions, cos, sin, kv_new_slot_mapping
    if draft:
        kernel_assert(
            not run_indexer and not attend_new_keys and tuple(draft_rows.shape) == (num_seqs_local, 1),
            f"{_ERROR_PREFIX} draft_rows is for a KVShare step: a shared layer (prev_gather_indices) with "
            f"attend_new_keys=False and draft_rows [{num_seqs_local}, 1], got run_indexer={run_indexer}, "
            f"attend_new_keys={attend_new_keys}, draft_rows {draft_rows.shape}",
        )
        T_q, spec_q = num_seqs_local, 1
        kernel_assert(
            kv_free in (0, 1, 2) and (not early_keys or (kv_free == 2 and prior_snake_rows is not None)),
            f"{_ERROR_PREFIX} kv_free={kv_free} must be 0, 1 or 2; early_keys needs kv_free=2 and the prior tables",
        )
        kernel_assert(
            tuple(prev_gather_indices.shape) == (T_q, index_topk),
            f"{_ERROR_PREFIX} with draft_rows, prev_gather_indices holds the draft rows only: "
            f"[{T_q}, {index_topk}], got {prev_gather_indices.shape}",
        )
        if not rows:
            hidden = select_rows(hidden, draft_rows, T_q, "hidden_draft")
        cos_q = select_rows(cos, draft_rows, T_q, "cos_draft")
        sin_q = select_rows(sin, draft_rows, T_q, "sin_draft")
        positions_q = select_rows(positions.reshape((T_local, 1)), draft_rows, T_q, "positions_draft").reshape((T_q,))
    kv_free = kv_free if draft else 0
    if draft and not kv_free:
        # One slot per sequence for the core's write contract; a KVShare caller's mapping is all negative.
        slot_sb = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=slot_sb[0:num_seqs_local, 0:1], src=kv_new_slot_mapping[0:num_seqs_local, 0:1])
        slot_mapping_q = nl.ndarray(
            (num_seqs_local, 1), dtype=nl.int32, buffer=_COLLECTIVE_HBM, name=unique_name("slot_mapping_draft")
        )
        nisa.dma_copy(dst=slot_mapping_q[0:num_seqs_local, 0:1], src=slot_sb[0:num_seqs_local, 0:1])

    # ═══════ Down-projections, then C2: one collective gathers both latents ═══════
    # The query path's shard degree, read off wq_b: the token group (its members hold the same tokens,
    # so the absorbed query is AllGathered inside the kernel), or the block group -- q_slices token
    # slices x the token group -- whose members hold the BLOCK's tokens. For the latter, one more
    # AllGather over the block group assembles the whole block's query latent (this rank's own slice is
    # among them), and the kernel's all_to_all hands every rank its own tokens back.
    q_degree = num_heads * (qk_nope_head_dim + qk_rope_head_dim) // wq_b.shape[1]
    idx_front, idx_prologue, idx_q_gathered = False, None, None
    idx_group_rows = None  # draft group: the draft row, with the front reorder's prologue
    q_slices = q_degree // token_group
    q_latent_block = cos_block = sin_block = None
    if q_slices > 1:
        kernel_assert(
            not draft,
            f"{_ERROR_PREFIX} draft_rows does not support the block-sharded query path: a KVShare draft step runs "
            f"the token-group query path, so hand the MTP layer wq_b / q_absorb_w over the token group "
            f"({token_group} ranks) even when the target's query path is block-sharded (qproj_tp)",
        )
        kernel_assert(
            qblock_replica_group is not None
            and q_degree % token_group == 0
            and head_shard == 1
            and token_shard % q_slices == 0,
            f"{_ERROR_PREFIX} wq_b holds 1/{q_degree} of the query columns, a block-sharded query path over "
            f"{q_slices} token slices: needs the qblock_replica_group ({q_degree} consecutive ranks), head_shard "
            f"== 1 (got {head_shard}) and q_slices | token_shard={token_shard}",
        )
        # The block path AllGathers q_latent_shard on its own, so it needs the shard as a whole tensor.
        q_latent_shard, kv_row_shard_t = mla_decode_down_proj(
            hidden_states=hidden, wq_a=wq_a, wkv_a=wkv_a, input_norm_w=None, norm_eps=input_norm_eps
        )
        latent_pair = _all_gather_columns(
            _pack_latents(q_latent_shard, kv_row_shard_t, T_local),
            token_replica_group,
            token_group,
            T_local,
            q_lora_shard + kv_row_shard,
            "latent_pair",
        )
        q_latent, kv_row = _split_latents(latent_pair, T_local, token_group, q_lora_shard, kv_row_shard)
        q_latent_block, cos_block, sin_block = _gather_block_latent(
            q_latent_shard, cos, sin, qblock_replica_group, q_degree, token_group, T_local, q_lora_shard, "q_block"
        )
    else:
        # The down-projection writes both shards straight into the collective source (1 DMA hop each),
        # and the gather's two outputs are split off with one strided DMA each. kv_free: the query latent alone.
        kv_pair_width = 0 if kv_free else kv_row_shard
        latent_pair_src = nl.ndarray(
            (T_q, q_lora_shard + kv_pair_width),
            dtype=hidden.dtype,
            buffer=_COLLECTIVE_HBM,
            name=unique_name("latent_pair"),
        )
        # The indexer front (see the target layer): its key side and query projection on the down-projection's and the
        # q-norm's K-tiles, and the scoring, the local top-k and the candidate gather right before wq_b.
        idx_front = run_indexer and num_seqs_local == 1 and spec == T_local
        # draft group: score the draft row's group of 4 query rows (needs the front reorder's prologue).
        idx_paged_rows = idx_block_rows if idx_block_rows is not None else idx_block_table
        if idx_draft_rows is not None and idx_front:
            kernel_assert(
                mtp_idx_query_tokens(spec, indexer_cache_keys(idx_k_cache), idx_paged_rows) == MTP_IDX_GROUP_TOKENS,
                f"{_ERROR_PREFIX} idx_draft_rows needs the chunk loop at spec 8",
            )
            idx_group_rows = idx_draft_rows
        hidden_tiles = [] if idx_front else None
        mla_decode_down_proj(
            hidden_states=hidden,
            wq_a=wq_a,
            wkv_a=None if kv_free else wkv_a,
            input_norm_w=None,
            norm_eps=input_norm_eps,
            output=latent_pair_src,
            hidden_tiles_out=hidden_tiles,
            hidden_tiles_sbuf=hidden_k_tiles,
        )
        q_latent, kv_row = _gather_latent_pair(
            latent_pair_src, token_replica_group, token_group, T_q, q_lora_shard, kv_pair_width
        )
        if kv_free:
            # Only the shape of the cache row is read downstream.
            kv_row = _ShapeOnly((T_q, qk_rope_head_dim + kv_lora_rank), hidden.dtype)
        if idx_front:
            # the indexer front: its key side, in the latent AllGather's PE gap
            key_prologue = indexer_key_prologue
            idx_paged = idx_block_rows if idx_block_rows is not None else idx_block_table
            if mtp_idx_chunk_loop(indexer_cache_keys(idx_k_cache), idx_paged):
                # The giga kernel's scorer takes it: the same instructions, on that module's shapes.
                from ..attention.indexer.indexer_kv_score import indexer_key_prologue as key_prologue
            idx_prologue = key_prologue(
                hidden_tiles[0][0],
                hidden_tiles[0][1],
                idx_wk,
                idx_k_norm_w,
                idx_k_norm_b,
                idx_weights_proj,
                idx_cos,
                idx_sin,
                idx_k_cache,
                idx_n_heads,
                idx_rope_dim,
                idx_norm_eps,
            )

            def idx_q_early(q_tiles, q_fn):
                # indexer_q_proj on the q-norm's K-tiles, each chain stored straight to the gather's source.
                nonlocal idx_q_gathered
                q_src = nl.ndarray(
                    (T_local, idx_q_shard), dtype=q_tiles.dtype, buffer=_COLLECTIVE_HBM, name=unique_name("idx_q_src")
                )
                linear(q_tiles, idx_wq_b, T_local, stationary_fn=q_fn, out_hbm=q_src)
                idx_q_gathered = _all_gather_columns(q_src, key_replica_group, key_size, spec, idx_q_shard, "idx_q")

    # kv_free: the shared selection's keys need nothing from this step, so they can be gathered ahead of the query.
    prefetched = None
    if early_keys and kv_free == 2:
        prefetched = mla_decode_attention_prefetch(kv_cache, prior_snake_rows, prior_key_pos)


    def run_indexer_now(q_latent_normed_arg, after_q_gather, split_merge=False):
        nonlocal idx_mask_add, idx_block_rows
        # The residency mask is a per-STEP quantity (it depends only on idx_positions[0]); a caller that
        # runs several indexer layers builds it once and passes it in, a standalone layer builds its own.
        # The legacy flat cache selects the pre-block path: contiguous shard, no deal.
        kernel_assert(
            idx_shard_degree == 1 or indexer_cache_is_blocked(idx_k_cache),
            f"idx_shard_degree={idx_shard_degree} needs the block-form indexer cache; the legacy flat "
            f"[idx_keys, idx_head_dim] cache keeps the contiguous shard only",
        )
        if idx_mask_add is None:
            idx_mask_add = mtp_idx_residency_mask(
                idx_positions,
                indexer_cache_keys(idx_k_cache),
                spec if idx_group_rows is None else MTP_IDX_GROUP_TOKENS,
                key_base=0,
                shard_degree=idx_shard_degree,
                block_size=idx_block_size,
                # The SAME tensor the select folds into its labels, so one runtime scalar serves both.
                rank_offset=key_offset if idx_shard_degree > 1 else None,
                idx_block_rows=idx_block_rows if idx_block_rows is not None else idx_block_table,
            )

        # Same per-STEP sharing as the residency mask: the row table depends on the block table only,
        # so a caller running several indexer layers builds it once. Both None leaves the cache
        # position-ordered, which is the pre-paging behaviour and every existing caller's path.
        if idx_block_rows is None and idx_block_table is not None:
            idx_keys = indexer_cache_keys(idx_k_cache)
            idx_block_rows = indexer_block_rows(idx_block_table, idx_keys, idx_keys // (idx_block_table.shape[1] - 1))

        return _run_indexer(
            q_latent_normed=q_latent_normed_arg,
            after_q_gather=after_q_gather,
            hidden=hidden,
            idx_wq_b=idx_wq_b,
            idx_wk=idx_wk,
            idx_k_norm_w=idx_k_norm_w,
            idx_k_norm_b=idx_k_norm_b,
            idx_weights_proj=idx_weights_proj,
            idx_cos=idx_cos,
            idx_sin=idx_sin,
            idx_k_cache=idx_k_cache,
            idx_new_slot_mapping=idx_new_slot_mapping,
            idx_mask_add=idx_mask_add,
            idx_block_rows=idx_block_rows,
            idx_shard_degree=idx_shard_degree,
            idx_block_size=idx_block_size,
            key_offset=key_offset,
            idx_positions=idx_positions,
            seq_replica_group=seq_replica_group,
            key_replica_group=key_replica_group,
            num_seqs_local=num_seqs_local,
            key_size=key_size,
            idx_n_heads=idx_n_heads,
            idx_rope_dim=idx_rope_dim,
            index_topk=index_topk,
            spec=spec,
            idx_norm_eps=idx_norm_eps,
            idx_q_shard=idx_q_shard,
            k_cols=k_cols,
            q_idx=idx_q_gathered,
            prologue=idx_prologue,
            split_merge=split_merge,
            idx_group_rows=idx_group_rows,
            cand_gather_out=cand_gate if split_merge else None,
        )

    idx_finish = None
    # the candidate gather's handle, which the query gather's source stores wait for.
    cand_gate = [] if idx_front else None

    def idx_scores_early():
        # the indexer front: scoring, local top-k and the candidate gather (AG0#4) ahead of wq_b; the merge
        # runs after mla_decode_q_kv_up, whose head gather (AG0#2) is then issued after AG0#4.
        nonlocal idx_finish
        idx_finish = run_indexer_now(None, lambda: None, split_merge=True)

    # ═══════ Up-projection, RoPE, W_UK absorb ═══════
    # Also the ONLY producer of the normalized query latent the indexer needs.
    q_absorbed, q_pe, kv_row_new_flat, q_latent_normed = mla_decode_q_kv_up(
        q_latent=q_latent,
        kv_row=kv_row,
        q_norm_w=q_norm_w,
        kv_norm_w=kv_norm_w,
        wq_b=wq_b,
        q_absorb_w=q_absorb_w,
        cos=cos_q,
        sin=sin_q,
        num_heads=num_heads,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        kv_lora_rank=kv_lora_rank,
        norm_eps=latent_norm_eps,
        # wq_b is COLUMN-sharded over the token group -- the same group that column-shards the two
        # down-projections, and the only one whose members hold the same tokens -- or over the block
        # group, whose members were just handed the block's tokens. The token-group shard cuts wq_b from
        # 67.1 MB to 16.8 MB per rank at CP4 for one 576 KB AllGather; the block shard divides that by
        # q_slices again (4.2 MB at 4 slices) for one 576 KB all_to_all.
        replica_group=token_replica_group if q_latent_block is None else qblock_replica_group,
        q_latent_block=q_latent_block,
        cos_block=cos_block,
        sin_block=sin_block,
        after_q_norm=idx_q_early if idx_front else None,
        before_up_proj=idx_scores_early if idx_front else None,
        q_stores_after=cand_gate,
        q_only=kv_free > 0,
        # a draft step (one query row): wq_b in the banded layout and the q-norm's column-major K-tiles
        wqb_banded_t1=draft,
        q_norm_colmajor_t1=draft,
    )

    # ═══════ The indexer, on the seq x key view of the token group ═══════
    # On an ``indexer_types == "shared"`` layer the reference has no indexer at all and reuses the
    # nearest preceding "full" layer's selection verbatim (modeling_glm_moe_dsa.py: prev_topk_indices).
    # Skipping the four sub-kernels also removes four collectives from this rank's layer.
    if run_indexer:
        # The block-sharded query path's all_to_all is issued INSIDE the indexer, after its query gather:
        # collectives complete in issue order, so issued after the query gather it overlaps the indexer's scoring
        # instead of stalling it.
        def issue_q_exchange():
            nonlocal q_absorbed, q_pe
            if q_latent_block is not None:
                q_absorbed, q_pe = mla_decode_q_block_exchange(q_absorbed, q_pe, qblock_replica_group)

        if idx_finish is not None:
            gather_indices_true, gather_indices_km = idx_finish()
        else:
            gather_indices_true, gather_indices_km = run_indexer_now(q_latent_normed, issue_q_exchange)
    else:
        gather_indices_true = prev_gather_indices
        gather_indices_km = None

    # The block-sharded query path's receive, placed AFTER the indexer on purpose: the engines run in
    # program order, so this is what hides the all_to_all's latency behind the indexer's work. (On a
    # shared layer the exchange is issued here and there is nothing to hide it behind.)
    if q_latent_block is not None:
        if not run_indexer:
            q_absorbed, q_pe = mla_decode_q_block_exchange(q_absorbed, q_pe, qblock_replica_group)
        q_absorbed, q_pe = mla_decode_q_block_finish(q_absorbed, q_pe)

    # ═══════ MLA attention over this rank's tokens and its share of their selected keys ═══════
    # The CP position shift (``cp_pos_offset``) is folded into build_prior_index: it lands on the
    # key-major tile there as two Vector ops instead of a token-major pass plus an HBM round trip here.
    # On the fused block-sharded exit the core writes its context and lse STRAIGHT into the all_to_all
    # payload (head-major cells, see _out_proj_block_fused_combine), so the 512 KB context never takes
    # the detour through its own HBM tensor and back.
    oproj_tp = num_heads // nh_out
    fused_combine = CP_COMBINE_IN_A2A and oproj_tp != cp_degree
    payload = None
    if fused_combine:
        payload = nl.ndarray(
            (num_heads * T_q, _a2a_cell(latent_dim)),
            dtype=io_dtype,
            buffer=_COLLECTIVE_HBM,
            name=unique_name("ctx_lse_cells"),
        )
    # The head gather left the queries rank-major in its own destination.
    heads_blocked = 1
    rank_major = take_rank_major_heads(q_absorbed)
    if rank_major is not None:
        q_absorbed, q_pe, heads_blocked = rank_major
    ctx_local, lse_local, kv_cache = mla_decode_attention(
        q_absorbed=q_absorbed,
        q_pe=q_pe,
        kv_cache=kv_cache,
        kv_row_new=(
            _ShapeOnly((num_seqs_local, spec_q, qk_rope_head_dim + kv_lora_rank), io_dtype)
            if kv_free
            else kv_row_new_flat.reshape((num_seqs_local, spec_q, qk_rope_head_dim + kv_lora_rank))
        ),
        new_slot_mapping=None if kv_free else slot_mapping_q,
        gather_indices=gather_indices_true.reshape((num_seqs_local, spec_q, index_topk)),
        block_table=block_table,
        positions=positions_q,
        softmax_scale=softmax_scale,
        kv_scale=kv_scale,
        spec=spec_q,
        kv_lora_rank=kv_lora_rank,
        cp_degree=cp_degree,
        cp_rank=0,
        # A SHARED layer may be handed the index tables its group's full layer already resolved (see
        # :func:`build_shared_prior_index`); they are a function of the selection, the block table and
        # the CP shift only, none of which change within the group.
        prior_snake_rows=prior_snake_rows,
        prior_key_pos=prior_key_pos,
        prior_natural_rows=prior_natural_rows,
        pos_offset=cp_pos_offset,
        positions_km=gather_indices_km,
        attend_new_keys=attend_new_keys,
        ctx_lse_out=payload,
        prior_tables_out=prior_tables_out if run_indexer else None,
        heads_blocked=heads_blocked,
        kv_free=kv_free,
        prefetched=prefetched,
    )

    # ═══════ A draft step's T_q results back into the T_local token slice, zeros elsewhere ═══════
    if draft and not rows:
        if fused_combine:
            payload = _expand_draft_cells(payload, draft_rows, num_heads, T_q, T_local, "ctx_lse_cells_full")
        else:
            ctx_local = _expand_draft_rows(
                ctx_local.reshape((T_q, num_heads * latent_dim)), draft_rows, T_local, "ctx_draft_full"
            ).reshape((T_local, num_heads, latent_dim))
            lse_local = _expand_draft_rows(lse_local, draft_rows, T_local, "lse_draft_full")

    # The exit's token count: the expanded slice, or a rows step's own draft tokens.
    T_exit = T_q if rows else T_local

    # ═══════ Every CP rank's log-sum-exp, so the combine can weight this rank's share ═══════
    # With the fused block-sharded exit the lse travels inside the context all_to_all instead and the
    # receiver forms the weights (see _out_proj_block_fused_combine); nothing to gather here.
    scaled_ctx = None
    if not fused_combine:
        lse_all = nl.ndarray(
            (cp_degree * T_exit, num_heads), dtype=nl.float32, buffer=_COLLECTIVE_HBM, name=unique_name("lse_all")
        )
        _pin_all(lse_local, lse_all)  # while pinning is on; see utils/cc_pin.py
        ncc.all_gather(srcs=[lse_local], dsts=[lse_all], replica_group=cp_replica_group, collective_dim=0)
        scaled_ctx = _cp_softmax_combine(ctx_local, lse_local, lse_all, T_exit, num_heads, latent_dim, cp_degree)

    # ═══════ The exit: which one depends on how many heads out_absorb_w carries ═══════
    # o_proj's output is a PARTIAL SUM over heads on every sharding, which is the contract
    # mla_decode_out_proj was written for (it asserts o_proj_w's row count against num_heads *
    # v_head_dim precisely to make that explicit). The head count is read off the operand, like every
    # other shard degree in this layer:
    #
    # * nh_out = Nh / cp_degree: heads over the CP group. The CP ReduceScatter scatters heads while it
    #   sums the key partials, o_proj runs on T_exit tokens, a second CP ReduceScatter sums the head
    #   partials while scattering tokens to T_shard.
    # * nh_out = Nh / oproj_tp, oproj_tp a multiple of cp_degree: heads over the BLOCK GROUP of oproj_tp
    #   consecutive ranks. See _out_proj_block_sharded.
    attn_out = None
    rows_exit = (rows_exit_seq, rows_n_seq, rows_replica_group) if rows else None
    if not rows:
        attn_out = nl.ndarray((T_shard, H), dtype=io_dtype, buffer=nl.shared_hbm, name=unique_name("attn_layer_out"))
    if oproj_tp != cp_degree:
        kernel_assert(
            block_replica_group is not None,
            f"{_ERROR_PREFIX} out_absorb_w carries {nh_out} of {num_heads} heads, an o_proj sharded "
            f"{oproj_tp} ways: that needs the block_replica_group ({oproj_tp} consecutive ranks)",
        )
        slices = oproj_tp // cp_degree
        kernel_assert(
            oproj_tp % cp_degree == 0 and token_shard % slices == 0 and head_shard == 1 and slices * T_exit <= P_MAX,
            f"{_ERROR_PREFIX} a block-sharded o_proj ({oproj_tp} ways) needs oproj_tp to be a multiple of "
            f"cp_degree={cp_degree}, its {slices} token slices to divide token_shard={token_shard}, head_shard "
            f"== 1 (got {head_shard}) and the block's {slices} x T_exit={T_exit} tokens to fit one partition "
            f"tile",
        )
        kernel_assert(
            not rows or fused_combine,
            f"{_ERROR_PREFIX} rows_in with a block-sharded o_proj needs the fused combine",
        )
        if fused_combine:
            fused_rows = _out_proj_block_fused_combine(
                attn_out,
                payload,
                out_absorb_w,
                o_proj_w,
                block_replica_group,
                T_exit,
                T_shard,
                oproj_tp,
                cp_degree,
                nh_out,
                latent_dim,
                H,
                rows_exit=rows_exit,
                exit_table=exit_table if rows else None,
            )
            if rows:
                attn_out = fused_rows
        else:
            _out_proj_block_sharded(
                attn_out,
                scaled_ctx,
                out_absorb_w,
                o_proj_w,
                block_replica_group,
                T_exit,
                T_shard,
                oproj_tp,
                cp_degree,
                nh_out,
                latent_dim,
                H,
            )
    else:
        # ═══════ The CP sum, which is also the re-shard onto the OUTPUT head slice ═══════
        # This collective scatters HEADS, not tokens, so that o_proj runs on
        # 1/cp_degree of the heads for ALL T_exit tokens rather than every head for T_exit/cp_degree
        # of them -- the same product of work, but with o_proj_w sharded cp_degree ways instead of
        # replicated, and with the matmul's stationary free dim cp_degree times wider.
        #
        # The sum it performs is unchanged: it still reduces the CP ranks' key-partials. Only which
        # slice of the result each rank keeps has moved from the token axis to the head axis.
        ctx_heads = nl.ndarray(
            (nh_out, T_exit * latent_dim),
            dtype=scaled_ctx.dtype,
            buffer=_COLLECTIVE_HBM,
            name=unique_name("ctx_head_shard"),
        )
        _pin_all(scaled_ctx, ctx_heads)  # while pinning is on; see utils/cc_pin.py
        ncc.reduce_scatter(
            srcs=[scaled_ctx], dsts=[ctx_heads], op=nl.add, replica_group=cp_replica_group, collective_dim=0
        )

        # Back to the token-major [T, Nh_out, C] that mla_decode_out_proj indexes. Two plain DMAs through
        # SBUF -- the head count here is at most 16 and the runs are C wide, so this is block movement
        # rather than a transpose.
        ctx_sb = nl.ndarray((P_MAX, nh_out, latent_dim), dtype=scaled_ctx.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=ctx_sb[0:T_exit, 0:nh_out, 0:latent_dim],
            src=ctx_heads.ap(pattern=[[latent_dim, T_exit], [T_exit * latent_dim, nh_out], [1, latent_dim]], offset=0),
        )
        ctx_tok_major = nl.ndarray(
            (T_exit, nh_out, latent_dim),
            dtype=scaled_ctx.dtype,
            buffer=_COLLECTIVE_HBM,
            name=unique_name("ctx_token_major"),
        )
        nisa.dma_copy(dst=ctx_tok_major[0:T_exit, 0:nh_out, 0:latent_dim], src=ctx_sb[0:T_exit, 0:nh_out, 0:latent_dim])

        # ═══════ W_UV absorb + o_proj, then the CP ReduceScatter that completes the head sum ═══════
        attn_partial = mla_decode_out_proj(ctx=ctx_tok_major, out_absorb_w=out_absorb_w, o_proj_w=o_proj_w)
    if rows and oproj_tp == cp_degree:
        # The CP group's head shares of its own sequences' rows; the all_reduce completes the head sum.
        attn_out = _exit_to_rows(attn_partial, rows_exit_seq, rows_n_seq, rows_replica_group)
    elif oproj_tp == cp_degree:
        tokens_cp = T_exit // cp_degree
        attn_cp = cc_dst((tokens_cp, H), attn_partial.dtype, _COLLECTIVE_HBM, "attn_cp_summed", reduce_scatter=True)
        _pin_all(attn_partial, attn_cp)  # while pinning is on; see utils/cc_pin.py
        ncc.reduce_scatter(
            srcs=[attn_partial], dsts=[attn_cp], op=nl.add, replica_group=cp_replica_group, collective_dim=0
        )

        # At head_shard == 1 the CP group covered every head, so attn_cp is COMPLETE and tokens_cp is
        # already T_shard. Otherwise a second reduction over the head group finishes the sum and
        # scatters tokens head_shard ways more, which composes onto global slot T_shard*i exactly as
        # before -- the token scatter moved from one collective to another, and the digit order is
        # untouched.
        if head_shard == 1:
            _deliver_attn_out(attn_out, attn_cp)
        else:
            attn_shard = nl.ndarray(
                (T_shard, H),
                dtype=attn_partial.dtype,
                buffer=_COLLECTIVE_HBM,
                name=unique_name("attn_shard"),
            )
            _pin_all(attn_cp, attn_shard)  # while pinning is on; see utils/cc_pin.py
            ncc.reduce_scatter(
                srcs=[attn_cp], dsts=[attn_shard], op=nl.add, replica_group=head_replica_group, collective_dim=0
            )
            _deliver_attn_out(attn_out, attn_shard)

    # A "shared" layer produced no selection of its own, so it hands the one it was given straight
    # back -- which is exactly how the reference threads ``topk_indices`` through consecutive shared
    # layers (the identical tensor object propagates). No copy, and no indexer key cache to return.
    if not run_indexer:
        return attn_out, prev_gather_indices, kv_cache, None

    gather_indices_out = nl.ndarray(
        (num_seqs_local, spec, index_topk), dtype=nl.int32, buffer=scratch_hbm(), name=unique_name("gather_indices_out")
    )
    nisa.dma_copy(dst=gather_indices_out.reshape((T_local, index_topk)), src=gather_indices_true)

    return attn_out, gather_indices_out, kv_cache, idx_k_cache


def _out_proj_block_sharded(
    attn_out: nl.NkiTensor,
    scaled_ctx: nl.NkiTensor,
    out_absorb_w: nl.NkiTensor,
    o_proj_w: nl.NkiTensor,
    block_replica_group: ReplicaGroup,
    T_local: int,
    T_shard: int,
    oproj_tp: int,
    cp_degree: int,
    nh_out: int,
    latent_dim: int,
    H: int,
) -> None:
    """
    The output projection sharded ``oproj_tp`` ways over HEADS, across the BLOCK GROUP: the CP group
    and its ``slices = oproj_tp / cp_degree`` neighbouring token slices, ``oproj_tp`` CONSECUTIVE ranks.
    Rank ``(kk, cp)`` sits at position ``d = cp_degree * (kk mod slices) + cp`` and projects heads
    ``[d * nh_out, (d + 1) * nh_out)`` for every one of the block's ``slices * T_local`` tokens.

    Why: with the output path head-sharded over the CP group alone, every token slice streams the
    SAME ``[Nh / cp_degree * v, H]`` o_proj rows -- 50 MB per rank per layer at cp4. Spreading the heads
    over ``oproj_tp`` ranks divides that by ``oproj_tp / cp_degree`` and widens the o_proj stationary
    from ``T_local`` to ``slices * T_local`` tokens, for the same number of collectives as the
    CP-sharded exit: the context ReduceScatter becomes an all_to_all, the output ReduceScatter runs on
    the block group instead of the CP group.

    Steps:
      1. ``all_to_all`` over the block group on the HEAD axis of the key-partial context: block ``d``
         of ``scaled_ctx`` (heads ``[d * nh_out, (d + 1) * nh_out)``, ``[nh_out, T_local * C]``) goes to
         position ``d``, and block ``s`` of the destination arrives from source ``s = cp_degree * slice
         + cp``: token slice ``slice``'s partial ``cp`` of THIS rank's heads. Every destination gets a
         distinct head slice, so the bytes equal the ReduceScatter's input it replaces.
      2. The CP combine, on chip: the ``cp_degree`` partials of each slice are summed in fp32 (the
         weights from ``_cp_softmax_combine`` sum to 1, so nothing further is needed) and the result laid
         out token-major ``[slices * T_local, nh_out, C]`` -- the block's tokens in ``(slice, t)`` order.
      3. ``mla_decode_out_proj`` -> ``[slices * T_local, H]``, partial over the ``oproj_tp`` head groups.
      4. ReduceScatter(add) over the block group on dim 0. Position ``p`` receives rows
         ``[p * T_shard, (p + 1) * T_shard)`` = slice ``p // cp_degree``, local tokens
         ``[cp' * T_shard, (cp' + 1) * T_shard)`` with ``cp' = p mod cp_degree`` -- exactly the MoE rows
         rank ``(kk', cp')`` owns (global slots ``T_shard * i``, ``i = token_group * kk' + cp'``). No
         permutation, no per-rank index: the destination IS the group position, which is what keeps
         the kernel rank-invariant (see the module docstring).

    Args:
        attn_out (nl.NkiTensor): [T_shard, H] kernel output, written here.
        scaled_ctx (nl.NkiTensor): [Nh, T_local * C] this rank's key-partial context times its CP
            weight, head-major, HBM (``_cp_softmax_combine``).
        out_absorb_w (nl.NkiTensor): [C, nh_out, v_head] latent-major W_UV for this rank's heads.
        o_proj_w (nl.NkiTensor): [nh_out * v_head, H] this rank's head rows, full hidden.
        block_replica_group (ReplicaGroup): The ``oproj_tp`` consecutive ranks, in position order.
        T_local, T_shard, oproj_tp, cp_degree, nh_out, latent_dim, H (int): Extents.
    """

    slices = oproj_tp // cp_degree
    T_all = slices * T_local
    C = latent_dim
    TC = T_local * C
    num_heads = oproj_tp * nh_out
    dtype = scaled_ctx.dtype
    rows = nh_out * T_local  # (head, token) pairs of this rank's heads, on partitions below

    # 1. The all_to_all. Dim 0 of [Nh, T_local * C] splits into oproj_tp blocks of nh_out heads: block d
    #    to position d, block s from source s. HBM operands, private_hbm like every collective here.
    ctx_a2a = cc_dst((num_heads, TC), dtype, _COLLECTIVE_HBM, "ctx_a2a")
    _pin_all(scaled_ctx, ctx_a2a)  # while pinning is on; see utils/cc_pin.py
    ncc.all_to_all(srcs=[scaled_ctx], dsts=[ctx_a2a], replica_group=block_replica_group, collective_dim=0)

    # 2. The CP sum. Received row (s, h) = (slice, cp, h) holds tokens x C. Loaded with (h, t) on
    #    partitions -- partition h * T_local + t reads at offset (h * T_local + t) * C, one stride -- and
    #    (slice, cp, C) on the free axis, the cp partials of a slice are free-axis neighbours: three
    #    fp32 adds over [rows, slices, C], one cast back. Summing in fp32 and rounding once is what the
    #    ReduceScatter it replaces does for the CP-sharded exit.
    recv_sb = nl.ndarray((P_MAX, slices, cp_degree, C), dtype=dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=recv_sb[0:rows, 0:slices, 0:cp_degree, 0:C],
        src=ctx_a2a.ap(
            pattern=[[C, rows], [cp_degree * nh_out * TC, slices], [nh_out * TC, cp_degree], [1, C]], offset=0
        ),
    )
    acc_f32 = nl.ndarray((P_MAX, slices, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=acc_f32[0:rows, 0:slices, 0:C],
        data1=recv_sb[0:rows, 0:slices, 0, 0:C],
        data2=recv_sb[0:rows, 0:slices, 1, 0:C],
        op=nl.add,
    )
    for cp in range(2, cp_degree):
        nisa.tensor_tensor(
            dst=acc_f32[0:rows, 0:slices, 0:C],
            data1=acc_f32[0:rows, 0:slices, 0:C],
            data2=recv_sb[0:rows, 0:slices, cp, 0:C],
            op=nl.add,
        )
    ctx_sum_sb = nl.ndarray((P_MAX, slices, C), dtype=dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ctx_sum_sb[0:rows, 0:slices, 0:C], src=acc_f32[0:rows, 0:slices, 0:C])

    #    Token-major [T_all, nh_out, C] for mla_decode_out_proj, token (slice, t) at row slice * T_local
    #    + t. Partition h * T_local + t writes at ((slice * T_local + t) * nh_out + h) * C: two partition
    #    strides, so one DMA per head, each a plain [T_local, slices, C] block copy of C-wide runs.
    ctx_tok_major = nl.ndarray(
        (T_all, nh_out, C), dtype=dtype, buffer=_COLLECTIVE_HBM, name=unique_name("ctx_block_token_major")
    )
    for h in range(nh_out):
        nisa.dma_copy(
            dst=ctx_tok_major.ap(pattern=[[nh_out * C, T_local], [T_local * nh_out * C, slices], [1, C]], offset=h * C),
            src=ctx_sum_sb[h * T_local : (h + 1) * T_local, 0:slices, 0:C],
        )

    # 3. + 4. The projection over the whole block, then the head sum as a ReduceScatter over the block
    #    group; T_all / oproj_tp = T_shard rows land on each rank, already in MoE order.
    _out_proj_block_tail(attn_out, ctx_tok_major, out_absorb_w, o_proj_w, block_replica_group, T_shard, H)


def _out_proj_block_tail(
    attn_out: nl.NkiTensor,
    ctx_tok_major: nl.NkiTensor,
    out_absorb_w: nl.NkiTensor,
    o_proj_w: nl.NkiTensor,
    block_replica_group: ReplicaGroup,
    T_shard: int,
    H: int,
    rows_exit: tuple = None,
    attn_partial: nl.NkiTensor = None,
) -> nl.NkiTensor:
    """``o_proj`` over the block's tokens, the head-group sum as a ReduceScatter, the output copy.

    Shared by :func:`_out_proj_block_sharded` and :func:`_out_proj_block_fused_combine`, which differ
    only in how they produce the combined token-major context.

    Args:
        attn_out (nl.NkiTensor): [T_shard, H] kernel output, written here.
        ctx_tok_major (nl.NkiTensor): [slices * T_local, nh_out, C] the CP-combined context of the
            block's tokens for this rank's heads, in HBM.
        out_absorb_w (nl.NkiTensor): [C, nh_out, v_head] latent-major W_UV for this rank's heads.
        o_proj_w (nl.NkiTensor): [nh_out * v_head, H] this rank's head rows, full hidden.
        block_replica_group (ReplicaGroup): The ``oproj_tp`` consecutive ranks, in position order.
        T_shard, H (int): Extents.
        rows_exit (tuple): Optional ``(exit_rows, n_seq, rows_replica_group)``: a rows-layout step's exit,
            the head-partial rows into a zero ``[n_seq, H]`` table and one ``all_reduce`` (see
            :func:`_exit_to_rows`), instead of the ReduceScatter. ``attn_out`` is then unused.
        attn_partial (nl.NkiTensor): Optional [slices * T_local, H] o_proj output already computed by the caller
            (the on-chip context path); ``ctx_tok_major`` is then unused. (default: None)

    Returns:
        nl.NkiTensor: The ``[n_seq, H]`` attention rows with ``rows_exit``, else None.
    """
    if attn_partial is None:
        attn_partial = mla_decode_out_proj(ctx=ctx_tok_major, out_absorb_w=out_absorb_w, o_proj_w=o_proj_w)
    if rows_exit is not None:
        exit_rows, n_seq, rows_replica_group = rows_exit
        return _exit_to_rows(attn_partial, exit_rows, n_seq, rows_replica_group)
    attn_rs = cc_dst((T_shard, H), attn_partial.dtype, _COLLECTIVE_HBM, "attn_block_summed", reduce_scatter=True)
    _pin_all(attn_partial, attn_rs)  # while pinning is on; see utils/cc_pin.py
    ncc.reduce_scatter(
        srcs=[attn_partial], dsts=[attn_rs], op=nl.add, replica_group=block_replica_group, collective_dim=0
    )
    _deliver_attn_out(attn_out, attn_rs)
    return None


def _a2a_cell(latent_dim: int) -> int:
    """Width of one head's cell in the fused-combine payload: C context values, 2 bf16 slots of fp32 lse, padding to 16 B."""
    return -(-(latent_dim + 2) // _A2A_CELL_ALIGN) * _A2A_CELL_ALIGN


def _out_proj_block_fused_combine(
    attn_out: nl.NkiTensor,
    payload: nl.NkiTensor,
    out_absorb_w: nl.NkiTensor,
    o_proj_w: nl.NkiTensor,
    block_replica_group: ReplicaGroup,
    T_local: int,
    T_shard: int,
    oproj_tp: int,
    cp_degree: int,
    nh_out: int,
    latent_dim: int,
    H: int,
    rows_exit: tuple = None,
    exit_table: dict = None,
) -> nl.NkiTensor:
    """
    :func:`_out_proj_block_sharded` with the CP softmax combine moved to the RECEIVER of the
    ``all_to_all``, so the ``lse`` AllGather over the CP group and the sender-side scaling disappear.

    Every rank's ``lse`` rides in the ``all_to_all`` next to the UNSCALED context it belongs to: the
    receiver holds all ``cp_degree`` partials of each ``(token, head)`` it projects, and with their
    ``lse`` it can form the CP softmax weights itself. The arithmetic is :func:`_cp_softmax_combine`'s,
    instruction for instruction, on the same values -- ``m = max_r lse_r``, ``exp(lse_r - m)``, the
    sum, its reciprocal, the fp32 weight, the bf16 ``ctx_r * w_r`` -- followed by the fp32 CP sum and
    single cast of :func:`_out_proj_block_sharded`. The weight of partial ``r`` no longer depends on
    which rank computes it, so nothing here needs a per-rank index (see the module docstring).

    Payload. ``[Nh * T_local, chunk]`` bf16 in HBM, HEAD-MAJOR: row ``n * T_local + t`` is head ``n`` of
    token ``t`` -- its ``C`` context values, then its fp32 ``lse`` in the next two bf16 slots (raw bits,
    moved by DMA only, never by a compute engine), then padding to a 16-byte multiple. Dim 0 splits into
    ``oproj_tp`` blocks of ``nh_out * T_local`` rows, block ``d`` (heads ``[d * nh_out, (d + 1) * nh_out)``)
    to position ``d``. The layout is chosen so ``mla_decode_attention`` can write it DIRECTLY from its
    head-on-partition context tile -- one DMA per token with a single partition stride of ``T_local``
    cells -- which is what removes the sender's 512 KB context round trip through HBM.

    Receiver. Block ``s = cp_degree * slice + cp`` arrives with token slice ``slice``'s partial ``cp`` of
    THIS rank's heads, rows ``(h, t)``. It is staged on partition ``p = (slice * nh_out + h) * T_local
    + t`` with the ``cp_degree`` partials on the free axis: one DMA per slice, and every combine op runs
    at ``slices * nh_out * T_local`` partitions. The combined context then goes out token-major
    (``[slices * T_local, nh_out, C]``, what ``mla_decode_out_proj`` consumes) with one DMA per
    ``(slice, head)``: ``T_local`` partitions to ``T_local`` rows ``nh_out * C`` apart.

    Args:
        attn_out (nl.NkiTensor): [T_shard, H] kernel output, written here.
        payload (nl.NkiTensor): [Nh * T_local, chunk] the cell table the attention core wrote, HBM.
        out_absorb_w (nl.NkiTensor): [C, nh_out, v_head] latent-major W_UV for this rank's heads.
        o_proj_w (nl.NkiTensor): [nh_out * v_head, H] this rank's head rows, full hidden.
        block_replica_group (ReplicaGroup): The ``oproj_tp`` consecutive ranks, in position order.
        T_local, T_shard, oproj_tp, cp_degree, nh_out, latent_dim, H (int): Extents.
        rows_exit (tuple): Optional, see :func:`_out_proj_block_tail`.
        exit_table (dict): Optional, with ``rows_exit`` and the on-chip context: see :func:`_exit_rows_table`.

    Returns:
        nl.NkiTensor: The ``[n_seq, H]`` attention rows with ``rows_exit``, else None.
    """
    slices = oproj_tp // cp_degree
    T = T_local
    T_all = slices * T
    C = latent_dim
    dtype = payload.dtype
    R = nh_out * T  # rows per all_to_all block = (head, token) pairs of one source
    P_recv = slices * R  # (slice, head, token) triples this rank projects
    kernel_assert(
        P_recv <= P_MAX and C % 2 == 0,
        f"{_ERROR_PREFIX} fused CP combine: slices * nh_out * T_local = {P_recv} must fit {P_MAX} partitions and "
        f"C = {C} must be even",
    )
    kernel_assert(
        dtype == nl.bfloat16 and tuple(payload.shape) == (oproj_tp * R, _a2a_cell(C)),
        f"{_ERROR_PREFIX} fused CP combine payload must be [{oproj_tp * R}, {_a2a_cell(C)}] bfloat16, got "
        f"{payload.shape} {dtype}",
    )
    chunk = _a2a_cell(C)

    # 1. The all_to_all: block d (rows [d * R, (d + 1) * R)) to position d; block s arrives from source
    #    s = cp_degree * slice + cp with token slice ``slice``'s partial ``cp`` of THIS rank's heads.
    recv_hbm = cc_dst((oproj_tp * R, chunk), dtype, _COLLECTIVE_HBM, "ctx_lse_a2a")
    _pin_all(payload, recv_hbm)  # while pinning is on; see utils/cc_pin.py
    ncc.all_to_all(srcs=[payload], dsts=[recv_hbm], replica_group=block_replica_group, collective_dim=0)

    # 2. Receiver: (slice, h, t) on partition slice * R + h * T + t, the cp partials on the free axis.
    #    Within a slice the partition stride is one row, the cp stride one block: one DMA per slice.
    recv_sb = nl.ndarray((P_MAX, cp_degree, chunk), dtype=dtype, buffer=nl.sbuf)
    for sl in range(slices):
        nisa.dma_copy(
            dst=recv_sb[sl * R : (sl + 1) * R, 0:cp_degree, 0:chunk],
            src=recv_hbm.ap(
                pattern=[[chunk, R], [R * chunk, cp_degree], [1, chunk]], offset=sl * cp_degree * R * chunk
            ),
        )
    # The CP softmax weights, exactly as _cp_softmax_combine forms them. Every op reads the lse bits in
    # place, through an fp32 view of the received cells: [P_recv, cp_degree], cp on the free axis.
    lse_v = recv_sb[0:P_recv, 0:cp_degree, C : C + 2].view(nl.float32)[0:P_recv, 0:cp_degree, 0]
    max_sb = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(max_sb[0:P_recv, 0:1], op=nl.max, data=lse_v, axis=1)
    shifted_sb = nl.ndarray((P_MAX, cp_degree), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=shifted_sb[0:P_recv, 0:cp_degree],
        data1=lse_v,
        data2=max_sb[0:P_recv, 0:1].broadcast(1, cp_degree),
        op=nl.subtract,
    )
    exp_sb = nl.ndarray((P_MAX, cp_degree), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(exp_sb[0:P_recv, 0:cp_degree], op=nl.exp, data=shifted_sb[0:P_recv, 0:cp_degree])
    sum_sb = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(sum_sb[0:P_recv, 0:1], op=nl.add, data=exp_sb[0:P_recv, 0:cp_degree], axis=1)
    recip_sb = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(recip_sb[0:P_recv, 0:1], op=nl.reciprocal, data=sum_sb[0:P_recv, 0:1])
    weight_sb = nl.ndarray((P_MAX, cp_degree, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=weight_sb[0:P_recv, 0:cp_degree, 0],
        data1=exp_sb[0:P_recv, 0:cp_degree],
        data2=recip_sb[0:P_recv, 0:1].broadcast(1, cp_degree),
        op=nl.multiply,
    )
    # ctx_r * w_r in bf16 (the sender's scaling, moved here), then the fp32 CP sum and one cast.
    scaled_sb = nl.ndarray((P_MAX, cp_degree, C), dtype=dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=scaled_sb[0:P_recv, 0:cp_degree, 0:C],
        data1=recv_sb[0:P_recv, 0:cp_degree, 0:C],
        data2=weight_sb[0:P_recv, 0:cp_degree, 0:1].broadcast(2, C),
        op=nl.multiply,
    )
    acc_f32 = nl.ndarray((P_MAX, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=acc_f32[0:P_recv, 0:C],
        data1=scaled_sb[0:P_recv, 0, 0:C],
        data2=scaled_sb[0:P_recv, 1, 0:C],
        op=nl.add,
    )
    for cp in range(2, cp_degree):
        nisa.tensor_tensor(
            dst=acc_f32[0:P_recv, 0:C],
            data1=acc_f32[0:P_recv, 0:C],
            data2=scaled_sb[0:P_recv, cp, 0:C],
            op=nl.add,
        )
    ctx_sum_sb = nl.ndarray((P_MAX, C), dtype=dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ctx_sum_sb[0:P_recv, 0:C], src=acc_f32[0:P_recv, 0:C])
    if C % P_MAX == 0:  # the combined context goes to the o_proj latent-major on chip, not through HBM
        ctx_t_sb = _ctx_latent_major_on_chip(ctx_sum_sb, slices, nh_out, T, C)
        if exit_table is not None and rows_exit is not None:
            exit_rows, n_seq, rows_replica_group = rows_exit
            kernel_assert(
                exit_rows.shape[0] == T_all,
                f"{_ERROR_PREFIX} exit_table needs one exit row per projected token, got {exit_rows.shape[0]} != {T_all}",
            )
            table = mla_decode_out_proj_sbuf(
                ctx_t_sb,
                T_all,
                out_absorb_w,
                o_proj_w,
                rows_table=_exit_rows_table(exit_table, exit_rows, n_seq, H, dtype),
            )
            rows = nl.ndarray((n_seq, H), dtype=dtype, buffer=_COLLECTIVE_HBM, name=unique_name("attn_rows"))
            _pin_all(table, rows)  # while pinning is on; see utils/cc_pin.py
            ncc.all_reduce(srcs=[table], dsts=[rows], replica_group=rows_replica_group, op=nl.add)
            return rows
        attn_partial = mla_decode_out_proj_sbuf(ctx_t_sb, T_all, out_absorb_w, o_proj_w)
        return _out_proj_block_tail(
            attn_out,
            None,
            out_absorb_w,
            o_proj_w,
            block_replica_group,
            T_shard,
            H,
            rows_exit=rows_exit,
            attn_partial=attn_partial,
        )
    # Token-major for mla_decode_out_proj: partition slice * R + h * T + t is row (slice * T + t) of head h,
    # so each (slice, head) run of T partitions is one DMA to T rows nh_out * C apart.
    ctx_tok_major = nl.ndarray(
        (T_all, nh_out, C), dtype=dtype, buffer=_COLLECTIVE_HBM, name=unique_name("ctx_block_token_major")
    )
    for sl in range(slices):
        for h in range(nh_out):
            p0 = sl * R + h * T
            nisa.dma_copy(
                dst=ctx_tok_major[sl * T : (sl + 1) * T, h, 0:C],
                src=ctx_sum_sb[p0 : p0 + T, 0:C],
            )

    return _out_proj_block_tail(
        attn_out, ctx_tok_major, out_absorb_w, o_proj_w, block_replica_group, T_shard, H, rows_exit=rows_exit
    )


def _ctx_latent_major_on_chip(ctx_sum_sb: nl.NkiTensor, slices: int, nh_out: int, T: int, C: int) -> nl.NkiTensor:
    """The fused combine's context, ``(slice, head, token)`` on partitions, -> ``mla_decode_out_proj_sbuf``'s layout.

    One PE transpose per 128-latent chunk (bf16 in and out, pass-through) puts the latent on partitions with the
    triples on the free axis; one Vector copy per chunk permutes them to head-major over the block's tokens,
    column ``(chunk * nh_out + h) * slices * T + slice * T + t``.

    Args:
        ctx_sum_sb (nl.NkiTensor): [slices * nh_out * T, C] bf16 SBUF, partition ``(slice * nh_out + h) * T + t``.
        slices, nh_out, T, C (int): Extents; C a multiple of 128.

    Returns:
        nl.NkiTensor: [P_MAX, C / 128 * nh_out * slices * T] bf16 SBUF.
    """
    R = nh_out * T
    P_recv = slices * R
    T_all = slices * T
    nC = C // P_MAX
    width = nh_out * T_all
    ctx_t_sb = nl.ndarray((P_MAX, nC * width), dtype=ctx_sum_sb.dtype, buffer=nl.sbuf)
    for ci in range(nC):
        ct_psum = nl.ndarray((P_MAX, P_recv), dtype=ctx_sum_sb.dtype, buffer=nl.psum)
        nisa.nc_transpose(ct_psum[0:P_MAX, 0:P_recv], ctx_sum_sb[0:P_recv, ci * P_MAX : (ci + 1) * P_MAX])
        # (slice, h, t) -> (h, slice, t)
        nisa.tensor_copy(
            dst=ctx_t_sb.ap(pattern=[[nC * width, P_MAX], [T_all, nh_out], [T, slices], [1, T]], offset=ci * width),
            src=ct_psum.ap(pattern=[[P_recv, P_MAX], [T, nh_out], [R, slices], [1, T]], offset=0),
            engine=nisa.vector_engine,
        )
    return ctx_t_sb


def _run_indexer(
    q_latent_normed: nl.NkiTensor,
    after_q_gather,
    hidden: nl.NkiTensor,
    idx_wq_b: nl.NkiTensor,
    idx_wk: nl.NkiTensor,
    idx_k_norm_w: nl.NkiTensor,
    idx_k_norm_b: nl.NkiTensor,
    idx_weights_proj: nl.NkiTensor,
    idx_cos: nl.NkiTensor,
    idx_sin: nl.NkiTensor,
    idx_k_cache: nl.NkiTensor,
    idx_new_slot_mapping: nl.NkiTensor,
    idx_mask_add: nl.NkiTensor,
    idx_block_rows: nl.NkiTensor,
    idx_shard_degree: int,
    idx_block_size: int,
    key_offset: nl.NkiTensor,
    seq_replica_group: ReplicaGroup,
    key_replica_group: ReplicaGroup,
    num_seqs_local: int,
    key_size: int,
    idx_n_heads: int,
    idx_rope_dim: int,
    index_topk: int,
    spec: int,
    idx_norm_eps: float,
    idx_q_shard: int,
    k_cols: int,
    q_idx: nl.NkiTensor | None = None,
    prologue: tuple | None = None,
    split_merge: bool = False,
    idx_positions: nl.NkiTensor | None = None,
    idx_group_rows: nl.NkiTensor | None = None,
    cand_gather_out: list = None,
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """Run the DSA indexer on the ``seq x key`` view and return the selection for this rank's tokens.

    The sequence scatter in and the selection gather out are the SAME group and are exact inverses, so
    the gather lands exactly ``T_local`` query rows -- this rank's own tokens, which is what the
    attention core wants. Under token sharding the scatter degree is ``num_seqs_local``, and at
    ``num_seqs_local == 1`` both degenerate to copies.

    Args:
        after_q_gather (callable): Trace-time hook run right after the query gather is issued; the layer
            issues the block-sharded query path's ``all_to_all`` there so it overlaps the scoring.
        q_latent_normed (nl.NkiTensor): [T_local, q_lora_rank] the NORMALIZED query latent.
        hidden (nl.NkiTensor): [T_local, H] the gathered normed residual stream.
        idx_wq_b (nl.NkiTensor): [q_lora_rank, idx_q_shard] this rank's query-projection columns.
        idx_wk (nl.NkiTensor): [H, idx_head_dim] replicated key projection.
        idx_k_norm_w (nl.NkiTensor): [1, idx_head_dim] fp32 LayerNorm gain.
        idx_k_norm_b (nl.NkiTensor): [1, idx_head_dim] fp32 LayerNorm bias.
        idx_weights_proj (nl.NkiTensor): [H, idx_n_heads] per-head mixing weights.
        idx_cos (nl.NkiTensor): [spec, idx_rope // 2] fp32 cosine table.
        idx_sin (nl.NkiTensor): [spec, idx_rope // 2] fp32 sine table.
        idx_k_cache (nl.NkiTensor): [idx_keys, idx_head_dim] this rank's key range, mutated.
        idx_new_slot_mapping (nl.NkiTensor): [spec] int32 destination rows, negative to skip.
        idx_mask_add (nl.NkiTensor): [128, idx_keys / 128 * spec] fp32 key-major additive residency mask for this rank's key
            range (0.0 keep / -MASK_BOUND drop), from ``indexer_residency_mask``.
        key_offset (nl.NkiTensor): [1, 1] int32, the rank term of the local-to-global position map.
        idx_shard_degree (int): Degree the indexer's keys are DEALT over; 1 = its contiguous slice.
        idx_block_size (int): The indexer cache's block size; the deal unit under a shared table.
        seq_replica_group (ReplicaGroup): The sequence group, in increasing sequence order.
        key_replica_group (ReplicaGroup): The key group, in increasing key-shard order.
        num_seqs_local (int): Sequences in this rank's token slice; the scatter/gather degree.
        key_size (int): Key-shard degree.
        idx_n_heads (int): Indexer query heads.
        idx_rope_dim (int): Indexer rotated width.
        index_topk (int): Global selection budget.
        spec (int): Padded token slots per sequence, i.e. the indexer's query row count.
        idx_norm_eps (float): Key LayerNorm epsilon.
        idx_q_shard (int): This rank's indexer query columns.
        k_cols (int): ``index_topk // 16``, the snake's column count.
        q_idx (nl.NkiTensor | None): The gathered query, when the caller already projected and gathered it
            (the indexer front); the projection and its gather are then skipped.
        prologue (tuple | None): :func:`indexer_key_prologue`'s result for ``indexer_kv_score(prologue=)``.
        split_merge (bool): Stop after issuing the candidate gather and return a callable that traces the merge and
            returns the selection (packed hand-off only).
        idx_positions (nl.NkiTensor | None): The residency mask's step positions, for the chunk loop
            (the giga kernel's dynamic chunk loop); None keeps the static scorer.
        idx_group_rows (nl.NkiTensor | None): [1, 1] int32, the draft row of the draft group. Only its
            group's 4 query rows are scored (:func:`_draft_group_query_side` picks them and their prologue tiles out of
            the 8-row ones), against ``idx_mask_add`` built for 4 tokens. Both halves of the 8 phase-1 rows take the
            group's candidates, so the merge and everything after it keep 8 rows.

    Returns:
        gathered (nl.NkiTensor): [T_local, index_topk] int32 -- every one of this rank's slots' selected
            TRUE GLOBAL key positions, or -1 where a slot has no live candidate.
        gathered_km (nl.NkiTensor): [num_seqs_local * P_MAX, spec * index_topk // P_MAX] int32, the same
            selection key-major, one block per sequence, for ``build_prior_index`` (see
            ``indexer_topk_merge_carry_index_packed``); None on the stock path, whose consumers read ``gathered``.
    """
    # This rank's ONE sequence, scattered out of the token slice it shares.
    if q_idx is None:
        q_resid = _scatter_rows(q_latent_normed, seq_replica_group, num_seqs_local, spec, "idx_q_resid")
    hidden_seq = _scatter_rows(hidden, seq_replica_group, num_seqs_local, spec, "idx_hidden")

    # ═══════ The query up-projection is column-sharded over the key group ═══════
    # RoPE rotates whole heads and the head axis is reduced by the weighted sum, so the score kernel
    # needs every head -- which is why this gather cannot be deferred past it.
    if q_idx is None:
        q_idx = _all_gather_columns(
            indexer_q_proj(q_resid=q_resid, wq_b=idx_wq_b), key_replica_group, key_size, spec, idx_q_shard, "idx_q"
        )
    after_q_gather()

    # The scores stay in SBUF as the local top-k's snake where the layout allows (only that stage reads them).
    snake_scores = snake_score_layout(indexer_cache_keys(idx_k_cache), spec)
    # chunk loop: the giga kernel's scorer, one loop over the live top-k chunks, each chunk scored and
    # selected in the same iteration into its slot of the phase-1 candidates (its _run_indexer's chunk_select); the
    # candidates of a chunk past the live length stay the sentinel.
    score_kernel = indexer_kv_score
    chunk_kw = {}
    phase1 = None
    if (
        idx_positions is not None
        and not snake_scores
        and mtp_idx_chunk_loop(indexer_cache_keys(idx_k_cache), idx_block_rows)
    ):
        from ..attention.indexer.indexer_kv_score import indexer_kv_score as score_kernel
        from ..attention.indexer.indexer_topk_local import indexer_topk_local_chunk
        from ..attention.indexer.indexer_topk_local_carry_index import _pick_local_chunks, chunk_phase1_alloc

        n_chunks, chunk_keys = _pick_local_chunks(indexer_cache_keys(idx_k_cache), index_topk)
        phase1 = chunk_phase1_alloc(spec, n_chunks, index_topk, sentinel=True)
        # draft group: score the group's rows; each chunk's candidates go to both halves of phase 1.
        T_sel, halves = spec, (0,)
        if idx_group_rows is not None:
            T_sel = MTP_IDX_GROUP_TOKENS
            kernel_assert(spec == 2 * T_sel, f"{_ERROR_PREFIX} the draft group needs spec {2 * T_sel}, got {spec}")
            halves = (0, T_sel)
            q_idx, prologue = _draft_group_query_side(prologue, q_idx, idx_group_rows)
        # Outside the loop like every body tile: the staging of one chunk's candidates and their slot offset.
        cand_stage = (
            nl.ndarray((T_sel, index_topk), dtype=nl.bfloat16, buffer=nl.sbuf),
            nl.ndarray((T_sel, index_topk), dtype=nl.int32, buffer=nl.sbuf),
        )
        cand_off = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)

        def chunk_select(chunk_scores, slot, label_p):
            # Traced once per call site (the peeled chunk, and the loop body), so each call names its own DMAs.
            cand_name = unique_name("mtp_idx_chunk_cand")
            cands = indexer_topk_local_chunk(
                chunk_scores, index_topk, 0, key_offset, idx_shard_degree, idx_block_size, extra_offset=label_p
            )
            nisa.tensor_scalar(dst=cand_off, data=slot, op0=nl.multiply, operand0=index_topk)
            for k, (src, stage, dst) in enumerate(zip(cands, cand_stage, phase1)):
                nisa.dma_copy(
                    dst=stage[0:T_sel, 0:index_topk],
                    src=src.ap(pattern=[[index_topk, T_sel], [1, index_topk]], offset=0),
                    name=f"{cand_name}_ld{k}",
                )
                for h0 in halves:
                    nisa.dma_copy(
                        dst=dst.ap(
                            pattern=[[n_chunks * index_topk, T_sel], [1, index_topk]],
                            offset=h0 * n_chunks * index_topk,
                            scalar_offset=cand_off,
                            indirect_dim=1,
                        ),
                        src=stage[0:T_sel, 0:index_topk],
                        name=f"{cand_name}_st{k}_{h0}",
                    )

        chunk_kw = dict(
            store_keys=chunk_keys,
            positions=idx_positions,
            rank_offset=key_offset if idx_shard_degree > 1 else None,
            shard_degree=idx_shard_degree,
            block_size=idx_block_size,
            chunk_select=chunk_select,
        )
    score_slot_mapping = idx_new_slot_mapping
    if idx_group_rows is not None:
        # The giga kernel's scorer writes as many keys as it scores queries (its _write_key_cache takes num_tokens),
        # so the 8 new keys are written here, by its own write with 8 tokens, and the scorer gets all-skip slots.
        score_slot_mapping = _write_all_new_keys(idx_k_cache, prologue[2], idx_new_slot_mapping, q_idx)
    with _gk_key_hwdge("1" if chunk_kw else ""):
        index_scores, idx_k_cache = score_kernel(
            q=q_idx,
            hidden_states=hidden_seq,
            wk=idx_wk,
            k_norm_weight=idx_k_norm_w,
            k_norm_bias=idx_k_norm_b,
            weights_proj=idx_weights_proj,
            cos=idx_cos,
            sin=idx_sin,
            k_cache=idx_k_cache,
            new_slot_mapping=score_slot_mapping,
            mask_add=idx_mask_add,
            num_heads=idx_n_heads,
            rope_dim=idx_rope_dim,
            # ZERO, not this rank's key range start: the residency threshold rides idx_mask_add and the
            # position labels are offset below. See the module docstring.
            key_base=0,
            norm_eps=idx_norm_eps,
            # None keeps the position-ordered read; a table makes the same cache tensor paged.
            block_rows=idx_block_rows,
            snake_out=snake_scores,
            prologue=prologue,
            **chunk_kw,
        )

    # ═══════ Stage 1, then the candidate gathers ═══════
    # Each shard offers the FULL budget, not a fraction: that is what makes the merge exact. The local select writes the
    # packed hand-off tile itself (indexer_topk_local_packed), and one all_gather of it feeds the packed merge.
    from .attention.indexer.indexer_topk_merge_carry_index import indexer_topk_merge_carry_index_packed

    from .attention.indexer.indexer_topk_local import indexer_topk_local_packed

    wide_merge = False
    if local_topk_chunked(indexer_cache_keys(idx_k_cache)):
        # A shard wider than one select may take (1M: 262144 keys a rank): the giga kernel's carry-index local (the
        # main copy's, as the chunk loop's phase 1 above) ranks it in chunks and publishes the same packed tile; the
        # shards trade their top-k scores first (the wide rounds), and the merge compacts each shard's share. Under
        # the chunk loop, phase 1 already ran in the score loop.
        from ..attention.indexer.indexer_topk_local_carry_index import indexer_topk_local_carry_index_packed

        kernel_assert(not snake_scores, f"{_ERROR_PREFIX} a chunked local select reads the HBM scores")
        wide_merge = True
        packed_local = indexer_topk_local_carry_index_packed(
            index_scores=index_scores,
            topk=index_topk,
            key_offset=key_offset,
            shard_degree=idx_shard_degree,
            block_size=idx_block_size,
            **({"phase1": phase1} if phase1 is not None else {}),
            wide_gather=lambda tile, name: _gather_rows(tile, key_replica_group, key_size, name),
        )
    elif snake_scores:
        packed_local = indexer_topk_local_packed(
            index_scores=None,
            topk=index_topk,
            key_offset=key_offset,
            src_snake=index_scores,
            shard_degree=idx_shard_degree,
            block_size=idx_block_size,
        )
    else:
        packed_local = indexer_topk_local_packed(
            index_scores=index_scores,
            topk=index_topk,
            key_offset=key_offset,
            shard_degree=idx_shard_degree,
            block_size=idx_block_size,
        )
    packed_all = _gather_rows(
        packed_local, key_replica_group, key_size, "cand_packed_all", handle_out=cand_gather_out
    )

    def merge():
        selected, selected_key_major = indexer_topk_merge_carry_index_packed(
            gathered=packed_all, topk=index_topk, degree=key_size,
            block_size=idx_block_size if idx_shard_degree > 1 else 0,
            wide=wide_merge,
        )
        gathered = _gather_rows(
            selected.reshape((spec, _SNAKE_PARTITIONS * k_cols)),
            seq_replica_group,
            num_seqs_local,
            "gather_indices",
        )
        gathered_km = _gather_rows(
            selected_key_major.reshape((P_MAX, spec * (index_topk // P_MAX))),
            seq_replica_group,
            num_seqs_local,
            "gather_indices_km",
        )
        return gathered, gathered_km

    return merge if split_merge else merge()


def _validate_attention_layer(
    T_shard: int,
    T_local: int,
    num_seqs_local: int,
    spec: int,
    cp_degree: int,
    head_shard: int,
    token_shard: int,
    key_size: int,
    index_topk: int,
    idx_q_shard: int,
    idx_n_heads: int,
    idx_head_dim: int,
    positions_len: int,
    num_heads: int,
    nh_out: int,
    o_proj_rows: int,
    v_head_dim: int,
    token_replica_group: ReplicaGroup,
    head_replica_group: ReplicaGroup,
    key_replica_group: ReplicaGroup,
    run_indexer: bool,
) -> None:
    """Reject the shardings and shapes this layer does not handle.

    The shard factorization and the exit's divisibility are checked here rather than left to the
    sub-kernels, because getting either wrong produces a running kernel whose tokens are silently
    attributed to the wrong ranks -- a sub-kernel only ever sees one rank's shapes and cannot tell.

    Args:
        T_shard (int): Token slots per rank on entry and exit.
        T_local (int): Token slots this rank's MLA scores.
        num_seqs_local (int): Sequences in this rank's token slice.
        spec (int): Padded token slots per sequence.
        cp_degree (int): Context-parallel degree.
        head_shard (int): Head-shard degree.
        token_shard (int): Token-shard degree.
        key_size (int): Key-shard degree of the indexer.
        index_topk (int): Selection budget.
        idx_q_shard (int): This rank's indexer query columns.
        idx_n_heads (int): Indexer query heads.
        idx_head_dim (int): Indexer head width.
        positions_len (int): Rows of ``positions``.
        num_heads (int): Query heads this rank ATTENDS over.
        nh_out (int): Query heads this rank projects through ``o_proj``, i.e. ``out_absorb_w``'s
            leading dim.
        o_proj_rows (int): Rows of ``o_proj_w``.
        v_head_dim (int): Per-head value width, ``out_absorb_w.shape[2]``.
        token_replica_group (ReplicaGroup): The token-slice collective group.
        head_replica_group (ReplicaGroup): The head-shard collective group.
        key_replica_group (ReplicaGroup): The indexer key-shard collective group.
        run_indexer (bool): False on an ``indexer_types == "shared"`` layer, which carries no indexer
            weights at all -- so the indexer-width and key-group checks have nothing to check.

    Returns:
        None
    """
    token_group = cp_degree * head_shard

    kernel_assert(
        nl.program_ndim() == 0 or nl.program_ndim() == 1,
        f"{_ERROR_PREFIX} only no specialization or specialization along one axis is supported, got "
        f"nl.program_ndim()={nl.program_ndim()}",
    )
    n_programs = nl.num_programs(axes=0) if nl.program_ndim() != 0 else 1
    kernel_assert(n_programs == 1, f"{_ERROR_PREFIX} runs on a single NeuronCore (LNC=1), got {n_programs} programs")

    kernel_assert(
        T_local <= nl.tile_size.pmax,
        f"{_ERROR_PREFIX} this rank's token slots must fit one partition tile: T_shard={T_shard} x "
        f"cp_degree={cp_degree} x head_shard={head_shard} = {T_local} > {nl.tile_size.pmax}",
    )
    kernel_assert(
        positions_len == T_local,
        f"{_ERROR_PREFIX} positions must cover this rank's slots: got {positions_len}, expected {T_local}",
    )
    kernel_assert(
        num_seqs_local * spec == T_local,
        f"{_ERROR_PREFIX} num_seqs_local={num_seqs_local} x spec={spec} must equal T_local={T_local}. "
        f"spec is the PADDED speculation length, so pick it to make this exact -- 8 for GLM-5.2's 7. A "
        f"token shard must also land on a whole number of sequences, or a rank would hold half of one "
        f"and the indexer could not run on it.",
    )

    # THE EXIT, which is what pins the linearization. Rank i must finish owning global slots
    # [T_shard*i, T_shard*(i+1)); the CP scatter hands it T_local/cp_degree rows and the head scatter
    # T_shard of those, so the two compose only when T_local/cp_degree == T_shard*head_shard.
    kernel_assert(
        T_local % cp_degree == 0 and (T_local // cp_degree) == T_shard * head_shard,
        f"{_ERROR_PREFIX} the exit does not line up: T_local={T_local} / cp_degree={cp_degree} must "
        f"equal T_shard={T_shard} x head_shard={head_shard}",
    )
    # THE OUTPUT HEAD SHARD. o_proj and W_UV are sharded over the CP group, or over a block group of a
    # multiple of cp_degree consecutive ranks -- which is legitimate only because that group's
    # collective (ReduceScatter or all_to_all) is what hands each rank its head slice, so a rank's slice
    # is chosen by its GROUP POSITION and never by a scalar it computes (see the module docstring).
    kernel_assert(
        nh_out > 0 and num_heads % nh_out == 0 and (num_heads // nh_out) % cp_degree == 0,
        f"{_ERROR_PREFIX} the output projection is head-sharded over the CP group or a block of token "
        f"slices, so out_absorb_w must carry num_heads={num_heads} / (k x cp_degree={cp_degree}) heads, "
        f"got {nh_out}",
    )
    kernel_assert(
        o_proj_rows == nh_out * v_head_dim,
        f"{_ERROR_PREFIX} o_proj_w must hold this rank's OUTPUT head rows: got {o_proj_rows}, expected "
        f"nh_out={nh_out} x v_head_dim={v_head_dim} = {nh_out * v_head_dim}",
    )
    kernel_assert(
        num_seqs_local * key_size == token_group,
        f"{_ERROR_PREFIX} a token group's {token_group} ranks must factor as num_seqs_local="
        f"{num_seqs_local} x key_size={key_size}, or the indexer cannot cover its slice's sequences",
    )

    # nisa.topk spends 16 partitions per query row, so the indexer's row count is hard-capped at 8.
    kernel_assert(
        spec * _SNAKE_PARTITIONS <= nl.tile_size.pmax,
        f"{_ERROR_PREFIX} spec={spec} query rows need {spec * _SNAKE_PARTITIONS} top-k partitions, "
        f"over the {nl.tile_size.pmax} available",
    )
    kernel_assert(
        index_topk % nl.tile_size.pmax == 0,
        f"{_ERROR_PREFIX} index_topk={index_topk} must be a multiple of {nl.tile_size.pmax}",
    )
    if run_indexer:
        kernel_assert(
            idx_q_shard * key_size == idx_n_heads * idx_head_dim,
            f"{_ERROR_PREFIX} idx_q_shard={idx_q_shard} x key_size={key_size} must equal the complete "
            f"indexer query width idx_n_heads={idx_n_heads} x idx_head_dim={idx_head_dim}",
        )
        kernel_assert(
            all(len(g) == key_size for g in key_replica_group._value),
            f"{_ERROR_PREFIX} key_replica_group must have {key_size} ranks per group, got {key_replica_group._value}",
        )

    # Group SHAPES, checked rather than trusted: token and head must be consecutive blocks because
    # their gather/scatter order is what maps a rank onto the global token slots it owns, and key must
    # be in increasing key-shard order because _all_gather_columns reads group position as column
    # position.
    kernel_assert(
        all(
            len(g) == token_group and list(g) == list(range(g[0], g[0] + token_group))
            for g in token_replica_group._value
        ),
        f"{_ERROR_PREFIX} token_replica_group must be consecutive blocks of {token_group} ranks, got "
        f"{token_replica_group._value}",
    )
    kernel_assert(
        all(
            len(g) == head_shard and list(g) == list(range(g[0], g[0] + head_shard)) for g in head_replica_group._value
        ),
        f"{_ERROR_PREFIX} head_replica_group must be consecutive blocks of {head_shard} ranks, got "
        f"{head_replica_group._value}",
    )
    kernel_assert(
        token_shard * token_group == token_shard * cp_degree * head_shard,
        f"{_ERROR_PREFIX} token_shard={token_shard} is inconsistent with cp_degree={cp_degree} and "
        f"head_shard={head_shard}",
    )
