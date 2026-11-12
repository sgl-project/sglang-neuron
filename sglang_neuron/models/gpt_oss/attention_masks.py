# SPDX-License-Identifier: Apache-2.0
"""The two decode masks of a step: one for the sliding-window layers, one for the full-attention layers."""

from typing import Optional

import torch

import sglang_neuron.ops as NF


def decode_attention_masks(attn_metadata: dict, positions: torch.Tensor, *, sliding_window: int, q_heads: int,
                           swa_layer: Optional[str], full_layer: Optional[str]):
    """``(swa_mask, full_mask)`` for the layers ``swa_layer`` and ``full_layer`` stand for, None for a kind the model
    has no layer of.

    The runtime trims a sliding-window layer's block table to the pages its window can reach and gives each row the
    trimmed table's first position (``swa_kv_pos_offset``), so the window mask is built in the trimmed frame: both its
    end (the row's position) and its start are shifted by that offset.

    Under DCP (``q_heads`` is the group's gathered heads) the runtime gives each row's bounds over this rank's share of
    the prior in the layer's table frame (``dcp_kv_end``, plus ``dcp_kv_start`` for a sliding window), and
    ``dcp_active_owner`` marks the rows whose new token this rank keeps, so the token counts on its owner only.
    """
    first = swa_layer or full_layer
    page = attn_metadata[first]["block_size"]
    rows = attn_metadata[first]["block_table_tensor"].shape[0]
    s_active = positions.shape[0] // rows
    end = positions.reshape(1, rows * s_active)

    owner = attn_metadata[first].get("dcp_active_owner")

    def mask(layer, end, start=None):
        return NF.gen_attention_decode_mask(
            end.to(torch.float32), bs=rows, q_head=q_heads, s_active=s_active,
            s_prior=attn_metadata[layer]["max_blocks_per_seq"] * page, block_len=page,
            start_pos=None if start is None else start.to(torch.float32), active_gate=owner,
        )

    swa_mask = full_mask = None
    if owner is not None:
        if swa_layer is not None:
            md = attn_metadata[swa_layer]
            swa_mask = mask(swa_layer, md["dcp_kv_end"].reshape(1, rows), md["dcp_kv_start"].reshape(1, rows))
        if full_layer is not None:
            full_mask = mask(full_layer, attn_metadata[full_layer]["dcp_kv_end"].reshape(1, rows))
        return swa_mask, full_mask
    if swa_layer is not None:
        offset = attn_metadata[swa_layer]["swa_kv_pos_offset"].reshape(rows, 1).expand(rows, s_active)
        # end - offset is not negative for a real row, nor for a padding row (offset 0). A negative end would wrap the
        # window over the whole trimmed table, whose pages are all -1, and the kernel would read stale K there.
        swa_end = (end - offset.reshape(1, rows * s_active)).clamp(min=0)
        swa_mask = mask(swa_layer, swa_end, (swa_end - sliding_window + 1).clamp(min=0))
    if full_layer is not None:
        full_mask = mask(full_layer, end)
    return swa_mask, full_mask
