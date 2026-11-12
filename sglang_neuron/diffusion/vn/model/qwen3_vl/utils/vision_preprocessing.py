# SPDX-License-Identifier: Apache-2.0
"""CPU preprocessing for the Qwen3-VL vision encoder, from ``grid_thw``: the 2-D rotary and
the bilinear taps of the position-embedding table, whose lookup runs on device.
"""

from __future__ import annotations

import torch


def compute_rotary_pos_emb(
    grid_thw: torch.Tensor,
    head_dim: int,
    spatial_merge_size: int,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """2-D vision RoPE ``(cos, sin)``, each ``[total_tokens, head_dim]``, as HF
    ``Qwen3VLVisionModel.rot_pos_emb`` and its forward compute them.

    ``grid_thw`` is ``[num_images, 3]``, (t, h, w) in patches. Tokens are in merge-block order
    (each ``spatial_merge_size`` square of patches contiguous); the row and column frequencies
    are concatenated.
    """
    half_dim = head_dim // 2

    max_hw = int(grid_thw[:, 1:].max().item())
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, half_dim, 2, dtype=torch.float32) / half_dim)
    )
    seq = torch.arange(max_hw, dtype=torch.float32)
    freq_table = torch.outer(seq, inv_freq)  # [max_hw, half_dim // 2]

    total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
    pos_ids = torch.empty((total_tokens, 2), dtype=torch.long)

    offset = 0
    for num_frames, height, width in grid_thw:
        num_frames = int(num_frames.item())
        height = int(height.item())
        width = int(width.item())
        merged_h = height // spatial_merge_size
        merged_w = width // spatial_merge_size

        block_rows = torch.arange(merged_h)
        block_cols = torch.arange(merged_w)
        intra_row = torch.arange(spatial_merge_size)
        intra_col = torch.arange(spatial_merge_size)

        # Patch row and column of each token, in merge-block order.
        row_idx = (
            block_rows[:, None, None, None] * spatial_merge_size
            + intra_row[None, None, :, None]
        )
        col_idx = (
            block_cols[None, :, None, None] * spatial_merge_size
            + intra_col[None, None, None, :]
        )

        row_idx = row_idx.expand(
            merged_h, merged_w, spatial_merge_size, spatial_merge_size
        ).reshape(-1)
        col_idx = col_idx.expand(
            merged_h, merged_w, spatial_merge_size, spatial_merge_size
        ).reshape(-1)

        coords = torch.stack((row_idx, col_idx), dim=-1)  # [h*w, 2]

        if num_frames > 1:
            coords = coords.repeat(num_frames, 1)

        num_tokens = coords.shape[0]
        pos_ids[offset : offset + num_tokens] = coords
        offset += num_tokens

    embeddings = freq_table[pos_ids]  # [total_tokens, 2, half_dim//2]
    rotary_pos_emb = embeddings.flatten(1)  # [total_tokens, half_dim]

    emb = torch.cat(
        (rotary_pos_emb, rotary_pos_emb), dim=-1
    )  # [total_tokens, head_dim]
    cos = emb.cos()
    sin = emb.sin()

    return cos, sin


def compute_position_indices_and_weights(
    grid_thw: torch.Tensor,
    num_grid_per_side: int,
    spatial_merge_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The 4 bilinear taps of the position-embedding table and their weights, per token.

    HF ``Qwen3VLVisionModel.fast_pos_embed_interpolate`` without the table lookup.
    ``num_grid_per_side`` is the side of the square table, ``sqrt(num_position_embeddings)``.

    Returns:
        ``(indices, weights)``, each ``[4, total_tokens]``: int32 table rows and FP32 weights,
        frame-repeated and in merge-block order.
    """
    idx_chunks: list[torch.Tensor] = []
    weight_chunks: list[torch.Tensor] = []

    for t, h, w in zip(grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]):
        t_val, h_val, w_val = int(t.item()), int(h.item()), int(w.item())

        indices, weights = _compute_bilinear_indices_and_weights(
            h_val, w_val, num_grid_per_side
        )
        indices = _apply_time_repeat_and_spatial_merge(
            indices, t_val, h_val, w_val, spatial_merge_size
        )
        weights = _apply_time_repeat_and_spatial_merge(
            weights, t_val, h_val, w_val, spatial_merge_size
        )

        idx_chunks.append(indices)
        weight_chunks.append(weights)

    idx_tensor = torch.cat(idx_chunks, dim=1).to(torch.int32)
    # FP32, so the 4-tap sum runs in FP32 and casts once.
    weight_tensor = torch.cat(weight_chunks, dim=1).float()

    return idx_tensor, weight_tensor


def _compute_bilinear_indices_and_weights(
    h: int, w: int, num_grid_per_side: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(indices, weights)``, each ``[4, h*w]``, of the 4 bilinear corners of an h x w grid."""
    h_idxs = torch.linspace(0, num_grid_per_side - 1, h)
    w_idxs = torch.linspace(0, num_grid_per_side - 1, w)

    h_floor = h_idxs.int()
    w_floor = w_idxs.int()
    h_ceil = (h_floor + 1).clip(max=num_grid_per_side - 1)
    w_ceil = (w_floor + 1).clip(max=num_grid_per_side - 1)

    dh = (h_idxs - h_floor).unsqueeze(1)  # (h, 1)
    dw = (w_idxs - w_floor).unsqueeze(0)  # (1, w)

    base_h = h_floor * num_grid_per_side
    base_h_ceil = h_ceil * num_grid_per_side

    indices = torch.stack(
        [
            (base_h[:, None] + w_floor[None, :]).flatten(),
            (base_h[:, None] + w_ceil[None, :]).flatten(),
            (base_h_ceil[:, None] + w_floor[None, :]).flatten(),
            (base_h_ceil[:, None] + w_ceil[None, :]).flatten(),
        ]
    )  # (4, h*w)

    weights = torch.stack(
        [
            ((1 - dh) * (1 - dw)).flatten(),
            ((1 - dh) * dw).flatten(),
            (dh * (1 - dw)).flatten(),
            (dh * dw).flatten(),
        ]
    )  # (4, h*w)

    return indices, weights


def _apply_time_repeat_and_spatial_merge(
    tensor: torch.Tensor,
    t: int,
    h: int,
    w: int,
    spatial_merge_size: int,
) -> torch.Tensor:
    """``[4, h*w]`` -> ``[4, t*h*w]``: repeated over the t frames, in merge-block order."""
    m = spatial_merge_size
    repeated = tensor.unsqueeze(1).expand(-1, t, -1).reshape(4, t * h * w)
    repeated = (
        repeated.view(4, t, h // m, m, w // m, m)
        .permute(0, 1, 2, 4, 3, 5)
        .contiguous()
        .reshape(4, t * h * w)
    )
    return repeated
