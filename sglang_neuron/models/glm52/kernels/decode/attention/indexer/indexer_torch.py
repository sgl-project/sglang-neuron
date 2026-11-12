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
Torch references for the GLM-5.2 DSA indexer kernels.

Written against ``modeling_glm_moe_dsa.py``'s ``GlmMoeDsaIndexer``, with two documented departures the
kernels make on purpose:

* the scored key range stops at this step's base position, because the attention kernel appends this
  step's keys out of band (see ``indexer_kv_score``'s docstring);
* the score is ranked in bfloat16, because ``nisa.topk`` accepts nothing else.

Both are reproduced here rather than papered over, so the tests measure the kernels against what they
are specified to do.
"""

import ml_dtypes
import numpy as np
import torch

from .indexer_constants import P_MAX

SNAKE_PARTITIONS = 16
"""Partitions ``nisa.topk`` devotes to one logical row. Fixes the candidate slot order below."""

MASK_BOUND = 1.0e30
"""Magnitude of the residency clamp. Must match ``indexer_constants.MASK_BOUND``."""

IDX_KEY_FP8_MAX = 448.0
"""Clamp of an fp8 (uint8) key cache. Must match ``indexer_constants.IDX_KEY_FP8_MAX``."""


def pack_indexer_keys_fp8(keys) -> np.ndarray:
    """``[..., head_dim]`` float keys -> the uint8 rows of an fp8 indexer key cache: ``clamp(key)`` as float8_e4m3fn.

    The kernel's cache write and the host's prefill writer (``layout/decode/gk_kv_cache.pack_keys_torch``) store these bytes.
    Accepts numpy or torch; returns numpy.
    """
    keys = torch.as_tensor(np.asarray(keys, dtype=np.float32))
    return keys.clamp(-IDX_KEY_FP8_MAX, IDX_KEY_FP8_MAX).to(torch.float8_e4m3fn).view(torch.uint8).numpy()


def unpack_indexer_keys_fp8(rows) -> np.ndarray:
    """Inverse of :func:`pack_indexer_keys_fp8`: uint8 key rows -> float32 keys."""
    rows = torch.as_tensor(np.ascontiguousarray(np.asarray(rows, dtype=np.uint8)))
    return rows.view(torch.float8_e4m3fn).float().numpy()


def indexer_global_positions_np(
    num_keys: int, key_base: int = 0, shard_degree: int = 1, block_size: int = 0, shard_rank: int = 0
) -> np.ndarray:
    """Global position of every LOGICAL key of one shard, ``[num_keys]`` int64.

    Host twin of the position arithmetic in ``indexer_residency_mask`` and ``indexer_topk_local_packed``, and
    the single source of truth for it: the store builders place keys with this and the references mask
    with it, so a write and a read cannot drift. A drift is silent -- a pure key-axis permutation leaves
    scores finite and in range.

    CONTIGUOUS (``shard_degree == 1``): ``key_base + i``. INTERLEAVED: local key ``i`` sits in global
    block ``shard_degree * (i // block_size) + shard_rank``, which is how ``prior_index`` deals MLA.
    """
    i = np.arange(num_keys, dtype=np.int64)
    if shard_degree == 1:
        return key_base + i
    bs = block_size or num_keys
    return (shard_degree * (i // bs) + shard_rank) * bs + (i % bs)


def indexer_local_index_np(
    global_pos: int,
    num_keys: int,
    key_base: int = 0,
    shard_degree: int = 1,
    block_size: int = 0,
    shard_rank: int = 0,
) -> int:
    """LOGICAL key index of ``global_pos`` on this shard, or -1 if the shard does not own it.

    Exact inverse of :func:`indexer_global_positions_np`, and the only place ``slot_mapping`` should
    decide ownership from. ``-1`` means this shard does not hold that position.
    """
    if shard_degree == 1:
        local = global_pos - key_base
        return local if 0 <= local < num_keys else -1
    bs = block_size or num_keys
    g, intra = divmod(global_pos, bs)
    if g % shard_degree != shard_rank:
        return -1
    local = (g // shard_degree) * bs + intra
    return local if 0 <= local < num_keys else -1


def indexer_residency_mask_np(
    positions,
    key_base: int,
    num_keys: int,
    num_tokens: int,
    shard_degree: int = 1,
    block_size: int = 0,
    rank_offset: int = 0,
) -> np.ndarray:
    """
    Host-side twin of ``indexer_residency_mask``: the additive key-residency mask in the score loop's
    key-major layout, ``[128, (num_keys / 128) * num_tokens]`` fp32 with element
    ``[p, tile * num_tokens + t]`` being ``0.0`` where the key's GLOBAL position is below
    ``positions[0]`` and ``-MASK_BOUND`` otherwise, for every token ``t``.

    The global position comes from :func:`indexer_global_positions_np`.

    ``rank_offset`` is the kernel's own parameter in the kernel's own units -- ``shard_rank *
    block_size``, not the rank index. The harness diffs this signature against the kernel's.
    """
    n_tiles = num_keys // 128
    bs = block_size or num_keys
    shard_rank = (rank_offset or 0) // bs if shard_degree > 1 else 0
    key_pos = indexer_global_positions_np(num_keys, key_base, shard_degree, block_size, shard_rank)
    key_pos = key_pos.reshape(n_tiles, 128).T  # [p, tile]
    tile_mask = np.where(key_pos < int(positions[0]), 0.0, -MASK_BOUND).astype(np.float32)
    return np.repeat(tile_mask[:, :, None], num_tokens, axis=2).reshape(128, n_tiles * num_tokens)


def indexer_block_rows_np(block_table: np.ndarray, num_keys: int, block_size: int) -> np.ndarray:
    """Host twin of ``indexer_block_rows``: [P_MAX, num_keys // P_MAX] uint32 physical rows.

    Entry ``[p, tile]`` is the physical row of LOGICAL key ``tile * P_MAX + p``. ``P_MAX`` is the
    kernel's constant rather than a parameter: a different tile height here would silently produce a
    layout the kernel never emits. ``block_table`` is
    ``[1, num_keys // block_size + 1]`` with a GUARD at column 0, matching the MLA convention: logical
    block ``b`` is read at column ``b + 1``. An unallocated block (negative entry) resolves to row 0,
    which the residency mask discards for tail entries -- see the kernel's module docstring for the
    invariant this assumes and why a mid-range hole is the caller's problem, not the kernel's.
    """
    n_blocks = num_keys // block_size
    table = np.asarray(block_table, np.int64).reshape(n_blocks + 1)
    logical = np.arange(num_keys, dtype=np.int64)
    physical_block = np.maximum(table[logical // block_size + 1], 0)
    rows = np.minimum(physical_block * block_size + logical % block_size, num_keys - 1)
    n_tiles = num_keys // P_MAX
    # natural order: logical key tile * P_MAX + p lands at [p, tile]
    return rows.reshape(n_tiles, P_MAX).T.astype(np.uint32).copy()


def indexer_residency_keep_np(mask_add: np.ndarray, num_keys: int, num_tokens: int) -> np.ndarray:
    """``[num_tokens, num_keys]`` bool keep set decoded from the key-major additive mask."""
    n_tiles = num_keys // 128
    m = np.asarray(mask_add, dtype=np.float32).reshape(128, n_tiles, num_tokens)  # [p, tile, t]
    return np.transpose(m, (2, 1, 0)).reshape(num_tokens, num_keys) == 0.0


def indexer_q_proj_torch_ref(
    q_resid: torch.Tensor,  # [T_q, q_lora_rank]
    wq_b: torch.Tensor,  # [q_lora_rank, q_shard]
) -> dict[str, torch.Tensor]:
    """
    Reference for the column-sharded indexer query up-projection.

    No RoPE and no norm, matching the kernel: RoPE needs whole heads and so runs after the gather, and
    the reference applies no second norm to an already-normalized latent.
    """

    return {"q_shard": torch.matmul(q_resid, wq_b)}


def indexer_kv_score_torch_ref(
    q: torch.Tensor,  # [T_q, num_heads * head_dim]  COMPLETE, post all-to-all
    hidden_states: torch.Tensor,  # [T_q, hidden]
    wk: torch.Tensor,  # [hidden, head_dim]
    k_norm_weight: torch.Tensor,  # [1, head_dim] fp32
    k_norm_bias: torch.Tensor,  # [1, head_dim] fp32
    weights_proj: torch.Tensor,  # [hidden, num_heads]
    cos: torch.Tensor,  # [T_q, rope_dim // 2]
    sin: torch.Tensor,  # [T_q, rope_dim // 2]
    k_cache: torch.Tensor,  # [n_blocks, 1, block_size, head_dim]
    new_slot_mapping: torch.Tensor,  # [T_q] int32
    mask_add: torch.Tensor,  # [T_q, num_keys] fp32, 0.0 keep / -MASK_BOUND drop
    num_heads: int,
    rope_dim: int,
    key_base: int,
    norm_eps: float = 1e-6,
    block_rows: torch.Tensor | None = None,
    snake_out: bool = False,  # noqa: ARG001 — kernel signature parity; the reference returns the HBM layout
    prologue=None,  # noqa: ARG001 — kernel signature parity; precomputed key side, same values
) -> dict[str, torch.Tensor]:
    """
    Reference for the key projection, RoPE, cache write, and head-reduced score.

    Order matters and mirrors the kernel: the new keys are written to the cache FIRST, but the score
    then reads only positions strictly below ``positions[0]``, so the freshly written rows are never
    scored. That is what removes the write-then-read hazard entirely rather than ordering around it.

    ``block_rows`` mirrors the kernel's paged read: [P_MAX, num_keys // P_MAX] uint32 giving the
    PHYSICAL row of logical key ``tile * P_MAX + p``, as ``indexer_block_rows`` builds it. When given,
    the score gathers the cache in logical order instead of reading it as stored; ``new_slot_mapping``
    already carries a physical row in both modes, so the write is unchanged. None keeps the
    position-ordered read.

    A golden that could only express a position-ordered cache would agree with a paged kernel that
    ignored its block table, so this parameter is what lets the test fail on that defect.

    A uint8 ``k_cache`` is an fp8 key cache, as in the kernel: the new keys are written as
    :func:`pack_indexer_keys_fp8` bytes, and the score reads every key decoded from them.
    """

    tokens, hidden = hidden_states.shape
    head_dim = wk.shape[1]
    # The cache arrives BLOCKED, [n_blocks, 1, block_size, head_dim], like MLA's. Both the append and the
    # score address rows, so flatten once.
    k_cache = k_cache.reshape(-1, k_cache.shape[-1])
    num_keys = k_cache.shape[0]

    # --- this step's key: project, LayerNorm, rotate the LEADING rope_dim ---
    k_new = _layernorm(torch.matmul(hidden_states, wk), k_norm_weight, k_norm_bias, norm_eps)
    k_new = _rope_leading(k_new.unsqueeze(1), cos.unsqueeze(1), sin.unsqueeze(1), rope_dim).squeeze(1)

    cache = k_cache.clone()
    key_fp8 = cache.dtype == torch.uint8
    slots = new_slot_mapping.reshape(-1).long()
    for i in range(slots.numel()):
        if 0 <= slots[i] < num_keys:
            cache[slots[i]] = (
                torch.as_tensor(pack_indexer_keys_fp8(k_new[i].float())) if key_fp8 else k_new[i].to(cache.dtype)
            )
    stored = cache.view(torch.float8_e4m3fn).float() if key_fp8 else cache.float()

    # --- query: rotate the leading rope_dim of every head ---
    q_heads = q.reshape(tokens, num_heads, head_dim)
    q_rot = _rope_leading(q_heads, cos.unsqueeze(1), sin.unsqueeze(1), rope_dim).float()

    # --- the keys the score sees, in LOGICAL order ---
    # Unpaged: storage order is logical order. Paged: reorder through the row table first. The table
    # is [P_MAX, n_tiles] in natural order, so logical key i lives at [i % P_MAX, i // P_MAX].
    if block_rows is None:
        keys = stored
    else:
        p_max, n_tiles = block_rows.shape
        logical = torch.arange(num_keys)
        rows = block_rows.reshape(p_max, n_tiles).long()[logical % p_max, logical // p_max]
        keys = stored[rows]

    # --- score, in fp32 as the reference does, then narrowed once on the way out ---
    raw = torch.einsum("thd,ed->the", q_rot, keys) * (float(head_dim) ** -0.5)
    scores = torch.relu(raw)

    w = torch.matmul(hidden_states.float(), weights_proj.float()) * (float(num_heads) ** -0.5)
    index_scores = torch.einsum("th,the->te", w, scores)

    # Residency mask: keep a key only if its GLOBAL position precedes this step's own tokens. The
    # kernel receives that decision as the precomputed additive mask (indexer_residency_mask_np); the
    # reference reads the keep set off it and applies the clamp the model semantics call for.
    keep = torch.as_tensor(indexer_residency_keep_np(mask_add, num_keys, tokens))
    index_scores = torch.minimum(index_scores, torch.where(keep, MASK_BOUND, -MASK_BOUND))

    return {
        "index_scores": index_scores.to(torch.bfloat16),
        "k_cache": cache,
    }


def indexer_topk_local_torch_ref(
    index_scores: torch.Tensor,  # [T_q, num_candidates] bf16
    topk: int,
    key_base: int,
    shard_degree: int = 1,
    block_size: int = 0,
) -> dict[str, torch.Tensor]:
    """
    Reference for stage 1 of the context-parallel top-k.

    Reproduces the kernel's SLOT ORDER as well as its selection, because both are part of the contract
    the merge stage reads. ``nisa.topk`` returns its results snaked and in ASCENDING value order, so
    output ordinal ``o`` -- the ``o``-th smallest of the selected set -- lands at slot
    ``(o % 16) * (topk // 16) + (o // 16)``.

    Tie order is left to the hardware, so a test comparing slots elementwise will disagree on the
    threshold tie group. Compare the selected multiset instead; in bfloat16 that group is routinely
    tens of keys wide.

    The published label is the candidate's GLOBAL position, from
    :func:`indexer_global_positions_np`, so an interleaved deal is labelled here rather than corrected
    afterwards -- as the kernel does by folding ``key_offset`` in before the select.
    """

    tokens, num_candidates = index_scores.shape
    # Parameter-for-parameter with both kernels, so no test-side adapter is needed. Under an interleaved
    # deal ``key_base`` carries the rank's block offset, so the rank index is derived from it.
    shard_rank = key_base // block_size if shard_degree > 1 and block_size else 0
    base = 0 if shard_degree > 1 else key_base
    global_pos = indexer_global_positions_np(num_candidates, base, shard_degree, block_size, shard_rank)
    k_cols = topk // SNAKE_PARTITIONS

    scores_f32 = index_scores.float()
    values = torch.zeros((tokens, topk), dtype=torch.float32)
    positions = torch.full((tokens, topk), -1, dtype=torch.int32)

    for t in range(tokens):
        # Ascending, so ordinal 0 is the smallest of the selected set.
        top = torch.topk(scores_f32[t], topk, largest=True)
        order = torch.argsort(top.values)
        sel_val = top.values[order]
        sel_idx = top.indices[order]

        for ordinal in range(topk):
            slot = (ordinal % SNAKE_PARTITIONS) * k_cols + (ordinal // SNAKE_PARTITIONS)
            values[t, slot] = sel_val[ordinal]
            live = sel_val[ordinal] > -0.5 * MASK_BOUND
            positions[t, slot] = int(global_pos[int(sel_idx[ordinal])]) if live else -1

    # Reported with the snake's two trailing axes intact, matching the kernel's output contract.
    return {
        "cand_values": values.reshape(tokens, SNAKE_PARTITIONS, k_cols).to(torch.bfloat16),
        "cand_positions": positions.reshape(tokens, SNAKE_PARTITIONS, k_cols),
    }


def _layernorm(x: torch.Tensor, gain: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Applies a TRUE LayerNorm over the last dim, in fp32.

    Distinct from the RMSNorm used elsewhere in this fork on two counts: it subtracts the mean, and it
    adds a bias. Substituting the RMSNorm would change the selection, not merely the scale, because the
    bias breaks any uniform-positive-scaling argument.
    """

    x_f32 = x.float()
    mean = x_f32.mean(-1, keepdim=True)
    centred = x_f32 - mean
    var = centred.pow(2).mean(-1, keepdim=True)
    normed = centred * torch.rsqrt(var + eps)
    return (normed * gain.float() + bias.float()).to(x.dtype)


def _rope_leading(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_dim: int) -> torch.Tensor:
    """
    Applies interleaved RoPE to the LEADING ``rope_dim`` of the last axis, passing the tail through.

    Leading, not trailing: the reference cats ``[q_rot, q_pass]``, which is the opposite of the MLA
    path's ``[q_nope, q_pe]``. Interleaved means element ``2i`` pairs with ``2i + 1``.
    """

    rot, passthrough = x[..., :rope_dim], x[..., rope_dim:]
    even, odd = rot[..., ::2], rot[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1).flatten(-2)
    return torch.cat([rotated, passthrough], dim=-1)


def to_bf16(x: np.ndarray) -> np.ndarray:
    """Casts through ml_dtypes so a numpy-side value matches what the kernel's bfloat16 holds."""
    return x.astype(ml_dtypes.bfloat16).astype(np.float32)


def indexer_topk_merge_torch_ref(
    cand_values: torch.Tensor,  # [T_q, degree * topk] bf16
    cand_positions: torch.Tensor,  # [T_q, degree * topk] int32
    topk: int,
) -> dict[str, torch.Tensor]:
    """
    Reference for stage 2 of the context-parallel top-k.

    Same slot convention as stage 1: ascending value order, snaked, so ordinal ``o`` lands at slot
    ``(o % 16) * (topk // 16) + (o // 16)``. A dead candidate carries the mask sentinel and so ranks
    below every live one; when it is selected anyway its -1 passes straight through, which is the right
    answer for a shard with fewer than ``topk`` live keys.
    """

    tokens = cand_values.shape[0]
    k_cols = topk // SNAKE_PARTITIONS

    values_f32 = cand_values.float()
    out = torch.full((tokens, topk), -1, dtype=torch.int32)

    for t in range(tokens):
        top = torch.topk(values_f32[t], topk, largest=True)
        order = torch.argsort(top.values)
        sel_idx = top.indices[order]
        for ordinal in range(topk):
            slot = (ordinal % SNAKE_PARTITIONS) * k_cols + (ordinal // SNAKE_PARTITIONS)
            out[t, slot] = int(cand_positions[t, sel_idx[ordinal]])

    return {"topk_positions": out.reshape(tokens, SNAKE_PARTITIONS, k_cols)}
