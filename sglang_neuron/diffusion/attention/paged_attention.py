# SPDX-License-Identifier: Apache-2.0
"""Non-causal attention of a query chunk over a paged K/V history plus its own K/V.

This is the attention core of autoregressive ("causal") video DiTs such as LingBot-World on
vLLM-Omni's paged AR-Diffusion KV manager: every chunk's query attends, without a causal mask, to
the keys/values committed by earlier chunks (the "history": pinned sink frames plus a sliding
local window) and to the chunk's own keys and values. Static shapes throughout, so one graph
serves every chunk. The pass's inputs (:class:`PagedAttentionInputs`):

* the history is read from the per-layer pool views (K d-major ``[num_blocks, N, D, block]``,
  V ``[num_blocks, N, block, D]``) through the per-head block table ``[N, W]`` (folded ids
  ``block * N + head``), the first ``num_history_blocks`` entries;
* listed history position ``p`` is visible iff ``p < sink_end`` or ``window_start <= p < kv_len``
  (``bounds`` ``[1, 3]`` int32);
* the query attends to the visible history plus the current chunk's K/V (always visible);
* on the commit pass the chunk's K/V are written into the pool through the per-token indices
  ``kv_cache_update_idx`` (one pool slot per token and head), skipping out-of-range entries.

:func:`attend_over_paged_history` picks one of three implementations of the same attention:

* **Paged FA2 kernel** :func:`nki_fa2_paged_attend` (on device at LNC 1 or LNC 2, on Trainium 2
  and Trainium 3; the DiT's path; at LNC 2 the two cores of the logical rank split the heads, an
  odd head by query rows): ``kernels/lingbot/fa2/fa2_paged.py`` reads the history
  through the block table on chip (indirect DMA), masks it from ``bounds``, all-gathers the
  chunk's K/V over CP inside the history pass (CP 1: no collective; exact online softmax) and,
  on the commit pass, writes the chunk into the pool in place.
* **nkilib ``attention_cte``** :func:`nki_paged_attend` (on device where the paged FA2 kernel
  does not run: a platform without the instructions it uses, an LNC above 2) in its
  prefix-caching mode, with everything around it in torch: the chunk's K/V
  all-gathered over CP (one collective for K and V together); the visible history gathered from
  the pool as whole blocks into a dense ``[N, S_h, D]`` K/V prefix, visible tokens first, so the
  kernel's runtime ``prior_used_len`` (the visible count) masks the rest (sink end and window
  start on block boundaries, see :func:`visible_history_blocks`); ``attention_cte`` attends the
  queries to ``prefix[:prior_used_len] ++ chunk`` (exact online softmax, FP32 accumulation,
  d-major output); on the commit pass :func:`paged_commit_write_` writes the gathered chunk into
  the pool (in place).
* **Torch reference** :func:`_torch_paged_attend` (CPU, fake-tensor tracing, NKI disabled;
  masked keys get a large negative score bias), with the same commit write.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import nki
import nki.language as nl
import torch
from nkilib.core.attention.attention_cte import attention_cte
from sglang_neuron.diffusion.attention.cp_attention import (
    _ATTN_MAX_BS as MAX_BS,
    _ATTN_MAX_HEAD_DIM as MAX_HEAD_DIM,
    _ATTN_MAX_SEQLEN as MAX_SEQLEN,
)

from sglang_neuron.diffusion.vn.utils.neuron_utils import can_run_kernel
from sglang_neuron.diffusion.kernels.kernel_choice import device_reason, record_kernel_choice
from sglang_neuron.diffusion.kernels.lingbot.fa2.fa2_paged import (
    fa2_paged_cache_attention_cp_gather as _fa2_paged_attention_kernel,
)
from sglang_neuron.diffusion.neuron_api import wrap_nki_kernel
from sglang_neuron.utils.hardware_config import get_logical_nc_config

#: Score bias of a masked history key (bf16(-30000)).
MASK_BIAS = -29952.0

# The K pool is d-major, ``[num_blocks, N, D, block]``: the paged FA2 kernel streams the history
# keys with no transpose and its commit writes whole ``[D, block]`` planes. V is token-major,
# ``[num_blocks, N, block, D]``.


def key_pool_shape(num_blocks: int, heads: int, block: int, head_dim: int) -> tuple[int, ...]:
    """The d-major K pool's shape."""
    return (num_blocks, heads, head_dim, block)


def key_pool_token_major(key_pool: torch.Tensor) -> torch.Tensor:
    """A token-major ``[num_blocks, N, block, D]`` view of the K pool."""
    return key_pool.transpose(2, 3)


def commit_block_aligned(
    kv_cache_update_idx: torch.Tensor, block_size: int, num_slots: int | None = None
) -> bool:
    """Whether a commit covers whole blocks: tokens ``[f * B, (f + 1) * B)`` (index order) go to
    the consecutive slots of one block, so the paged FA2 kernel can write block by block. With
    ``num_slots`` (the pool's ``num_blocks * N * block``), every block must also lie inside the
    pool: the kernel's block writes have no range guard."""
    n, s = kv_cache_update_idx.shape
    if s % block_size:
        return False
    idx = kv_cache_update_idx.to(torch.int64).reshape(n, s // block_size, block_size)
    first = idx[..., :1]
    return bool(
        (first >= 0).all()
        and (num_slots is None or (first + block_size <= num_slots).all())
        and (first % block_size == 0).all()
        and torch.equal(idx, first + torch.arange(block_size))
    )


class PagedAttentionInputs(NamedTuple):
    """One DiT pass's paged-history inputs, as the graph takes them.

    Per-head block table ``[N, W]`` int32, ``bounds`` ``[1, 3]`` int32 and, on the commit pass,
    the per-token update index ``[N, S]`` int32 (``None`` on a probe); ``num_history_blocks``
    is static (the listed history width). Only the first ``num_history_blocks`` table columns
    are read; later columns (the read plan lists the current chunk's blocks there, ``W = 18``
    at 480p) are ignored by the attention and the commit write.

    Static hints (graph constants, default ``False``): ``all_visible`` promises that ``bounds``
    masks no listed history position (the steady state; set from the read plan's
    ``history_full``, and verified on the host against the plan's bounds by
    ``LingBotDiTRuntime.check_paged_plan`` before launch), so the attention skips the mask
    (bit-identical);
    ``write_only`` marks a commit whose attention output is unused (the commit pass's last
    layer), so only the chunk's K/V are gathered and written (the attention returns ``None``).
    """

    block_table: torch.Tensor
    bounds: torch.Tensor
    kv_cache_update_idx: torch.Tensor | None
    num_history_blocks: int
    all_visible: bool = False
    write_only: bool = False


def _torch_gather_paged_history(
    key_pool: torch.Tensor,
    value_pool: torch.Tensor,
    block_table: torch.Tensor,
    num_history_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The listed history as dense ``[N, D, Sh]`` keys and ``[N, Sh, D]`` values."""
    nb, n, b, d = value_pool.shape
    planes_k = key_pool_token_major(key_pool).reshape(nb * n, b, d)
    planes_v = value_pool.reshape(nb * n, b, d)
    ids = block_table[:, :num_history_blocks].to(torch.int64)  # [N, Hb]
    keys = planes_k.index_select(0, ids.reshape(-1)).reshape(n, num_history_blocks * b, d)
    values = planes_v.index_select(0, ids.reshape(-1)).reshape(n, num_history_blocks, b, d)
    keys = keys.transpose(1, 2)
    values = values.reshape(n, num_history_blocks * b, d)
    return keys, values


def _torch_history_visibility(bounds: torch.Tensor, history_tokens: int) -> torch.Tensor:
    """``[Sh]`` bool: the two-range mask of the listed history positions."""
    p = torch.arange(history_tokens, device=bounds.device, dtype=torch.int32)
    b = bounds.reshape(-1).to(torch.int32)
    return (p < b[0]) | ((p >= b[1]) & (p < b[2]))


def _torch_paged_attend(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_pool: torch.Tensor,
    value_pool: torch.Tensor,
    block_table: torch.Tensor,
    bounds: torch.Tensor,
    scale: float,
    num_history_blocks: int,
    all_visible: bool = False,
) -> torch.Tensor:
    """Exact attention over the paged visible history ++ the chunk, fp32 accumulation.

    ``all_visible``: ``bounds`` masks nothing, so no mask is added (adding it adds ``+0.0``).

    Args:
        query: ``[B, Sq, N, D]`` token-major queries; ``key`` / ``value``: ``[B, S, N, D]`` the
            whole current chunk (after QK-norm/RoPE).
    Returns:
        ``[B, N, D, Sq]`` (d-major, the o-projection layout), in ``query.dtype``.
    """
    k_hist, v_hist = _torch_gather_paged_history(
        key_pool, value_pool, block_table, num_history_blocks
    )
    q = query.float().permute(0, 2, 1, 3)  # [B, N, Sq, D]
    k_cur = key.float().permute(0, 2, 1, 3)
    v_cur = value.float().permute(0, 2, 1, 3)
    scores_h = torch.einsum("bnqd,nds->bnqs", q, k_hist.float()) * scale
    if not all_visible:
        visible = _torch_history_visibility(bounds, k_hist.shape[-1])
        scores_h = scores_h + torch.where(visible, 0.0, MASK_BIAS * scale).to(scores_h.dtype)
    scores_c = torch.einsum("bnqd,bnsd->bnqs", q, k_cur) * scale
    probs = torch.softmax(torch.cat((scores_h, scores_c), dim=-1), dim=-1)
    sh = k_hist.shape[-1]
    out = torch.einsum("bnqs,nsd->bnqd", probs[..., :sh], v_hist.float())
    out = out + torch.einsum("bnqs,bnsd->bnqd", probs[..., sh:], v_cur)
    return out.transpose(2, 3).to(query.dtype)


def paged_commit_write_(
    key_pool: torch.Tensor,
    value_pool: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache_update_idx: torch.Tensor,
) -> None:
    """In place: the chunk's ``[1, S, N, D]`` K/V into the pool by per-token indices.

    Static shapes (it also runs inside the device graphs): an out-of-range entry (a pad) is
    pointed at row 0, the null block's, and writes back the value it read there, so it changes
    nothing; valid entries never target the null block.
    """
    nb, n, b, d = value_pool.shape
    rows = nb * n * b
    v_rows = value_pool.view(rows, d)
    idx = kv_cache_update_idx.to(torch.int64).reshape(-1)  # [N * S], head-major
    ok = ((idx >= 0) & (idx < rows)).reshape(-1, 1)
    safe = torch.where(ok.reshape(-1), idx, torch.zeros_like(idx))
    new_k = key[0].transpose(0, 1).reshape(-1, d).to(key_pool.dtype)
    new_v = value[0].transpose(0, 1).reshape(-1, d).to(value_pool.dtype)
    new_v = torch.where(ok, new_v, v_rows.index_select(0, safe))
    v_rows.index_copy_(0, safe, new_v)
    # d-major K: slot ``plane * b + offset`` of dim ``j`` is element ``(plane * d + j) * b + offset``.
    dims = torch.arange(d, device=idx.device, dtype=torch.int64) * b
    elems = ((safe // b) * (d * b) + safe % b).reshape(-1, 1) + dims.reshape(1, -1)
    k_flat = key_pool.view(-1)
    old_k = k_flat.index_select(0, elems.reshape(-1)).reshape(-1, d)
    k_flat.index_copy_(0, elems.reshape(-1), torch.where(ok, new_k, old_k).reshape(-1))


# ------------------------------------------------------------------------------- device path


@nki.jit
def _paged_attention_kernel(q, k, v, k_prior, v_prior, prior_used_len, scale):
    """Non-causal flash attention over ``prior[:prior_used_len] ++ active`` K/V.

    q ``[BN, S_q, D]``, k/v ``[BN, S, D]``, k_prior/v_prior ``[BN, S_h, D]`` (all token-major),
    prior_used_len int32 ``[1]``. Output is d-major ``[BN, D, S_q]`` (``tp_out=True``).
    """
    return attention_cte(
        q=q,
        k=k,
        v=v,
        scale=scale,
        causal_mask=False,
        k_prior=k_prior,
        v_prior=v_prior,
        prior_used_len=prior_used_len,
        tp_q=True,
        tp_k=True,
        tp_out=True,
        cache_softmax=False,
        softmax_dtype=nl.float32,
        mm_out_dtype=nl.float32,
    )


# Wrapped once, at import (see the launcher note in cp_attention.py).
_paged_attention_launch = wrap_nki_kernel(_paged_attention_kernel, supported_lnc=(1, 2))


def _can_use_paged_attention_kernel(
    query: torch.Tensor, chunk_tokens: int, history_tokens: int
) -> bool:
    if not can_run_kernel(query):
        return record_kernel_choice("lingbot_cache_attention", False, device_reason(query))
    B, S_q, N, D = query.shape
    if B * N > MAX_BS:
        return record_kernel_choice(
            "lingbot_cache_attention", False, "B*N exceeds the kernel batch limit"
        )
    if S_q > MAX_SEQLEN or chunk_tokens + history_tokens > MAX_SEQLEN:
        return record_kernel_choice(
            "lingbot_cache_attention", False, "sequence exceeds the kernel limit"
        )
    if D > MAX_HEAD_DIM:
        return record_kernel_choice(
            "lingbot_cache_attention", False, "head dim exceeds the kernel limit"
        )
    if history_tokens == 0:
        return record_kernel_choice("lingbot_cache_attention", False, "empty history table")
    return record_kernel_choice("lingbot_cache_attention", True, "eligible")


def _gather_chunk_kv(key_shard, value_shard, cp_group):
    """The whole chunk's ``[B, S, N, D]`` K and V in shard order (one CP all-gather for both)."""
    cp_size = cp_group.world_size if cp_group is not None else 1
    if cp_size == 1:
        return key_shard, value_shard
    from sglang_neuron.diffusion.models.lingbot_world.lingbot_transformer import (
        cp_all_gather_seq,
    )

    B, S_local, N, D = key_shard.shape
    kv = torch.cat(
        (key_shard.reshape(B, S_local, N * D), value_shard.reshape(B, S_local, N * D)), -1
    )
    kv = cp_all_gather_seq(kv, cp_group, dim=1)
    key, value = kv.split(N * D, dim=-1)
    return key.reshape(B, -1, N, D), value.reshape(B, -1, N, D)


def visible_history_blocks(
    block_table: torch.Tensor,
    bounds: torch.Tensor,
    num_history_blocks: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The visible history's blocks, compacted to a prefix, and its length.

    The listed history position ``p`` (``0 <= p < num_history_blocks * block_size``) is visible
    iff ``p < sink_end`` or ``window_start <= p < kv_len`` (``bounds``). The compacted history
    lists the sink's blocks, then the window's from the block holding ``window_start`` on, so its
    first ``prior_used_len = sink_end + (kv_len - window_start)`` tokens are exactly the visible
    ones when ``sink_end`` and ``window_start`` fall on block boundaries (``aligned``). That holds
    for every read plan with one block per latent frame (the sink, the window and the chunk are
    whole frames); the history is then gathered as whole blocks (contiguous planes, no
    per-token gather). Entries past the window are clamped into range and masked by
    ``prior_used_len``.

    Returns the folded ``block * N + head`` ids ``[N, num_history_blocks]`` (int32),
    ``prior_used_len`` int32 ``[1]`` and ``aligned`` (bool scalar tensor).
    """
    history_tokens = num_history_blocks * block_size
    b = bounds.reshape(-1).to(torch.int32)
    sink_end = b[0].clamp(0, history_tokens)
    window_start = torch.maximum(b[1], sink_end)
    kv_len = torch.maximum(b[2].clamp(max=history_tokens), window_start)
    aligned = (sink_end % block_size == 0) & (window_start % block_size == 0)
    sink_blocks = sink_end // block_size
    i = torch.arange(num_history_blocks, device=bounds.device, dtype=torch.int32)
    src = torch.where(i < sink_blocks, i, i - sink_blocks + window_start // block_size)
    src = src.clamp(max=num_history_blocks - 1)
    ids = block_table[:, :num_history_blocks].index_select(1, src)
    prior_used_len = (sink_end + kv_len - window_start).reshape(1).to(torch.int32)
    return ids, prior_used_len, aligned


#: The LNCs the paged FA2 kernel runs at (LNC 2: the two cores split the heads).
FA2_PAGED_SUPPORTED_LNC = (1, 2)
#: Platform targets without instructions the paged FA2 kernel's Trainium 3 engine plan uses
#: (``nisa.exponential``, ``nisa.activate2``); they take ``attention_cte``.
FA2_PAGED_UNSUPPORTED_TARGETS = frozenset({"trn3pre"})


@functools.cache
def _resolve_platform_target() -> str | None:
    try:
        from sglang_neuron.backend.stack import get_platform_target

        return get_platform_target()
    except Exception:  # noqa: BLE001 - no Neuron platform (e.g. a CPU host)
        return None


@torch._dynamo.assume_constant_result
def _platform_target() -> str | None:
    """The device's platform target (``trn2``, ``trn3``, ...), ``None`` if unknown.

    The gate runs inside the DiT's compiled graphs, so Dynamo folds this to a constant at trace
    time (as ``record_kernel_choice`` does) instead of tracing the platform lookup's imports.
    """
    return _resolve_platform_target()


def _fa2_paged_layout() -> str | None:
    """Why the paged FA2 kernel does not run on this device (``None``: it does)."""
    lnc = get_logical_nc_config()
    if lnc not in FA2_PAGED_SUPPORTED_LNC:
        return f"LNC {lnc} is not supported by the paged FA2 kernel"
    if _platform_target() in FA2_PAGED_UNSUPPORTED_TARGETS:
        return "the platform lacks instructions the paged FA2 kernel uses"
    return None


def uses_block_aligned_history(like: torch.Tensor) -> bool:
    """Whether :func:`attend_over_paged_history` runs :func:`nki_paged_attend` for tensors on
    ``like``'s device, whose block-wise history compaction needs ``sink_end`` and
    ``window_start`` on block boundaries. The paged FA2 kernel (LNC 1 or 2, Trainium 2 or
    Trainium 3) and the torch reference take any bounds, at any CP degree.

    Host-side and conservative: the kernel's shape limits are not checked.
    """
    if not can_run_kernel(like):
        return False
    return _fa2_paged_layout() is not None


def nki_paged_attend(query, key, value, key_pool, value_pool, paged, scale) -> torch.Tensor:
    """nkilib ``attention_cte`` over the block-gathered visible history ++ the whole chunk.

    ``query`` ``[B, Sq, N, D]``, ``key`` / ``value`` the CP-gathered chunk ``[B, S, N, D]``.
    The visible history is gathered from the pool as whole blocks (:func:`visible_history_blocks`)
    into a dense prefix whose first ``prior_used_len`` tokens the kernel attends to. Returns
    ``[B, N, D, Sq]``.
    """
    B, S_q, N, D = query.shape
    block_size = value_pool.shape[2]
    history_tokens = int(paged.num_history_blocks) * block_size
    if getattr(paged, "all_visible", False):
        # Every listed position is visible: the compaction is the identity and the whole
        # listed history is the prefix (what visible_history_blocks computes for such bounds).
        ids = paged.block_table[:, : int(paged.num_history_blocks)]
        prior_used_len = torch.full(
            (1,), history_tokens, dtype=torch.int32, device=paged.bounds.device
        )
        aligned = None
    else:
        ids, prior_used_len, aligned = visible_history_blocks(
            paged.block_table, paged.bounds, int(paged.num_history_blocks), block_size
        )
    flat = ids.reshape(-1)
    planes = (-1, block_size, D)
    k_prior = key_pool.reshape(-1, D, block_size).index_select(0, flat).transpose(1, 2)
    k_prior = k_prior.reshape(N, history_tokens, D)
    v_prior = value_pool.reshape(planes).index_select(0, flat).reshape(N, history_tokens, D)
    q = query.permute(0, 2, 1, 3).reshape(B * N, S_q, D).contiguous()
    k = key.permute(0, 2, 1, 3).reshape(B * N, -1, D).contiguous()
    v = value.permute(0, 2, 1, 3).reshape(B * N, -1, D).contiguous()
    out = _paged_attention_launch(q, k, v, k_prior, v_prior, prior_used_len, scale=float(scale))
    # Bounds off block boundaries cannot be compacted block-wise: fail loudly (NaN) rather than
    # attend to the wrong keys. The torch reference and the paged FA2 kernel take any bounds.
    if aligned is not None:
        out = torch.where(aligned, out, torch.full_like(out, float("nan")))
    return out.reshape(B, N, D, S_q)


# The launcher is wrapped once, at import, at the process's LNC (see the launcher note in
# cp_attention.py).
_fa2_paged_attention_launch = wrap_nki_kernel(
    _fa2_paged_attention_kernel, supported_lnc=FA2_PAGED_SUPPORTED_LNC
)


def _can_use_fa2_paged_attention_kernel(query: torch.Tensor) -> bool:
    # Any CP degree: CP 1 runs the kernel without the collective.
    if not can_run_kernel(query):
        return record_kernel_choice("fa2_paged_cache_attention", False, device_reason(query))
    # Elsewhere on device attention_cte runs by design (its gate then reports for itself).
    reason = _fa2_paged_layout()
    if reason is not None:
        return record_kernel_choice("fa2_paged_cache_attention", False, reason, by_design=True)
    return record_kernel_choice("fa2_paged_cache_attention", True, "eligible")


def nki_fa2_paged_attend(
    query, key_shard, value_shard, key_pool, value_pool, paged, scale, cp_group, replica_groups
) -> torch.Tensor | None:
    """The paged FA2 kernel with the in-kernel CP gather (and, on a commit, the pool write).

    ``query`` ``[B, Sq, N, D]``, ``key_shard`` / ``value_shard`` this CP rank's ``[B, S_local, N,
    D]`` (``cp_group`` ``None``: CP 1, the whole chunk, no collective); ``replica_groups``: every
    CP group of the world. Returns ``[B, N, D, Sq]`` (``None`` on a ``paged.write_only`` commit:
    the kernel only gathers and writes the chunk). ``paged.all_visible``: the kernel attends the
    history without the mask.
    """
    B, S_q, N, D = query.shape
    commit = paged.kv_cache_update_idx is not None
    write_only = commit and bool(getattr(paged, "write_only", False))
    gather = cp_group is not None and cp_group.world_size > 1
    if gather:
        rank_id = cp_group.rank_in_group_tensor.reshape(1, 1).to(torch.float32)
        rank_id = rank_id.expand(1, 128).contiguous()
    else:
        rank_id = torch.zeros(1, 128, dtype=torch.float32, device=query.device)
    inputs = [
        query.contiguous(),
        key_shard.contiguous(),
        value_shard.contiguous(),
        key_pool,
        value_pool,
        paged.block_table,
        paged.bounds,
        rank_id,
    ]
    if commit:
        inputs.append(paged.kv_cache_update_idx)
    result = _fa2_paged_attention_launch(
        *inputs,
        replica_groups=tuple(tuple(g) for g in replica_groups) if gather else (),
        cp_size=cp_group.world_size if gather else 1,
        gather=gather,
        scale=float(scale),
        # The commit is written block by block: the runtime refuses a plan that does not cover
        # whole blocks (commit_block_aligned).
        num_history_blocks=int(paged.num_history_blocks),
        all_visible=bool(getattr(paged, "all_visible", False)),
        write_only=write_only,
    )
    if write_only:
        return None
    out = result[0] if commit else result
    return out.reshape(B, N, D, S_q)


def attend_over_paged_history(
    query: torch.Tensor,
    key_shard: torch.Tensor,
    value_shard: torch.Tensor,
    key_pool: torch.Tensor,
    value_pool: torch.Tensor,
    paged,
    scale: float,
    cp_group,
    replica_groups=(),
):
    """Exact attention over the paged history (block table + two-range mask) and the chunk.

    The history is read from this layer's pool views (K d-major ``[num_blocks, N, D, block]``, V
    ``[num_blocks, N, block, D]``)
    through ``paged.block_table`` (first ``paged.num_history_blocks`` entries) and masked by
    ``paged.bounds``; the chunk's K/V are all-gathered over CP. On the commit pass
    (``paged.kv_cache_update_idx`` set) the gathered chunk is also written into the pool in
    place. The paged FA2 kernel runs where it can (LNC 1 or 2; ``replica_groups``: every CP
    group of the world, for its in-kernel all-gather), else ``attention_cte`` or the torch
    reference (see the module docstring).

    ``query`` ``[B, Sq, N, D]`` (this rank's queries, not pre-scaled), ``key_shard`` /
    ``value_shard`` ``[B, S_local, N, D]``. Returns ``[B, N, D, Sq]`` (``None`` on a
    ``paged.write_only`` commit: only the pool write runs).
    """
    if _can_use_fa2_paged_attention_kernel(query):
        return nki_fa2_paged_attend(
            query,
            key_shard,
            value_shard,
            key_pool,
            value_pool,
            paged,
            scale,
            cp_group,
            replica_groups,
        )
    history_tokens = int(paged.num_history_blocks) * value_pool.shape[2]
    key, value = _gather_chunk_kv(key_shard, value_shard, cp_group)
    if getattr(paged, "write_only", False) and paged.kv_cache_update_idx is not None:
        paged_commit_write_(key_pool, value_pool, key, value, paged.kv_cache_update_idx)
        return None
    if _can_use_paged_attention_kernel(query, key.shape[1], history_tokens):
        out = nki_paged_attend(query, key, value, key_pool, value_pool, paged, scale)
    else:
        out = _torch_paged_attend(
            query,
            key,
            value,
            key_pool,
            value_pool,
            paged.block_table,
            paged.bounds,
            scale,
            paged.num_history_blocks,
            all_visible=bool(getattr(paged, "all_visible", False)),
        )
    if paged.kv_cache_update_idx is not None:
        paged_commit_write_(key_pool, value_pool, key, value, paged.kv_cache_update_idx)
    return out
