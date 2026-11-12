# SPDX-License-Identifier: Apache-2.0
"""Convert between Wan's interleaved RoPE and the QKV CTE kernel's rotate-half RoPE.

Wan rotates interleaved lane pairs ``(2j, 2j+1)``; the fused QKV kernel rotates the two halves
``(i, i + d/2)``. Reordering each Q/K head's channels to ``[even | odd]`` lanes at load time makes
rotate-half on the reordered channels equal Wan's interleaved RoPE on the originals.
"""

import torch


def deinterleave_perm(head_dim: int) -> torch.Tensor:
    """Per-head column permutation so the kernel's rotate-half RoPE reproduces Wan's
    interleaved RoPE.

    Wan rotates interleaved lane pairs ``(2j, 2j+1)``; the QKV CTE kernel rotates the two
    halves ``(i, i+d/2)``. Reordering the projection's output channels to ``[even | odd]``
    lanes makes rotate-half on the reordered channels equal interleaved RoPE on the originals:
    ``new[..., i] = old[..., perm[i]]`` with ``perm = [0,2,...,d-2, 1,3,...,d-1]``.
    """
    if head_dim % 2:
        raise ValueError(f"head_dim must be even for interleaved RoPE, got {head_dim}")
    half = head_dim // 2
    idx = torch.arange(half)
    return torch.cat([2 * idx, 2 * idx + 1])


def deinterleave_qk_columns(
    tensor: torch.Tensor, q_size: int, kv_size: int, head_dim: int
) -> torch.Tensor:
    """Apply :func:`deinterleave_perm` to the Q and K channel blocks of a fused-QKV tensor laid
    out ``[..., I]`` with ``I = q_size + 2*kv_size`` columns ordered ``[Q | K | V]``.

    Permutes each Q and K head's ``head_dim`` lanes; leaves V untouched. Operates on the last
    dimension so it applies uniformly to a ``[H, I]`` weight, a ``[1, I]`` bias, etc. Q and K get
    the identical permutation, so the attention scores ``Q·Kᵀ`` are invariant.
    """
    device = tensor.device
    perm = deinterleave_perm(head_dim).to(device)
    # Build one column-permutation index (Q/K heads permuted, V identity) and do a single
    # index_select, instead of separate reshape/index_select/cat allocations per block.
    q_idx = torch.arange(q_size, device=device).reshape(-1, head_dim)[:, perm].flatten()
    k_idx = q_size + torch.arange(kv_size, device=device).reshape(-1, head_dim)[:, perm].flatten()
    v_idx = torch.arange(q_size + kv_size, tensor.shape[-1], device=device)
    return tensor.index_select(-1, torch.cat((q_idx, k_idx, v_idx)))


def deinterleave_gamma_row(gamma: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Apply the per-head :func:`deinterleave_perm` to a qk-norm gamma vector.

    ``gamma`` is the ``DistributedRMSNorm`` weight over a full Q (or K) vector, shape
    ``[n_heads * head_dim]``. Returns ``[1, n_heads * head_dim]`` with each head's lanes
    permuted to match the de-interleaved weight/bias, so ``gamma * x`` stays aligned.
    """
    perm = deinterleave_perm(head_dim).to(gamma.device)
    n_heads = gamma.numel() // head_dim
    return gamma.reshape(n_heads, head_dim).index_select(-1, perm).reshape(1, n_heads * head_dim)


def rope_rotate_half_caches(rotary_emb, head_dim: int):
    """Build rotate-half cos/sin caches from Wan's interleaved RoPE frequencies.

    Wan's ``freqs_cos`` / ``freqs_sin`` carry each of the ``head_dim // 2`` base angles duplicated
    on adjacent (interleaved) lanes — ``[c0, c0, c1, c1, ...]``. The QKV CTE kernel's rotate-half
    RoPE instead wants each base angle duplicated across the two halves — ``cat([base, base])`` —
    where ``base`` is the distinct angles ``freqs[..., 0::2]``. Paired with the offline
    ``deinterleave_perm`` on the Q/K weight/bias/gamma, rotate-half over the de-interleaved
    channels reproduces Wan's interleaved RoPE.

    Shape-preserving on the trailing ``head_dim`` axis. Returns ``(cos_cache, sin_cache)``.
    """
    freqs_cos, freqs_sin = rotary_emb
    if freqs_cos.shape[-1] != head_dim or freqs_sin.shape[-1] != head_dim:
        raise ValueError(
            f"rotary_emb last dim must be head_dim={head_dim}, "
            f"got cos={tuple(freqs_cos.shape)} sin={tuple(freqs_sin.shape)}"
        )
    # Mirror apply_rotary_emb_wan's base-angle extraction exactly: cos from even lanes, sin from
    # odd lanes (they coincide for repeat_interleave'd freqs, but match Wan to be layout-robust).
    base_cos = freqs_cos[..., 0::2]
    base_sin = freqs_sin[..., 1::2]
    cos_cache = torch.cat([base_cos, base_cos], dim=-1)
    sin_cache = torch.cat([base_sin, base_sin], dim=-1)
    return cos_cache, sin_cache


def apply_rotate_half_rope(
    hidden: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
) -> torch.Tensor:
    """Torch fallback for the fused QKV kernel's rotate-half RoPE convention."""
    half = hidden.shape[-1] // 2
    first, second = hidden[..., :half], hidden[..., half:]
    rotated = torch.cat([-second, first], dim=-1)
    # Like apply_rotary_emb_wan, keep the activation dtype: the RoPE tables are FP32.
    return (hidden * cos_cache + rotated * sin_cache).to(hidden.dtype)
