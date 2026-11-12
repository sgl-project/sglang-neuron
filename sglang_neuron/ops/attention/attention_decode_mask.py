# SPDX-License-Identifier: Apache-2.0
"""The decode attention mask, built once per step for all layers of one attention kind by nkilib's gen_mask_tkg_hbm
kernel, in the partition-major order the decode kernel reads. The torch path reproduces that order for CPU runs."""

import math
from typing import Optional

import nki
import torch
from nkilib.core.attention.gen_mask_tkg import gen_mask_tkg_hbm
from torch import Tensor

from sglang_neuron.backend.stack import wrap_nki
from sglang_neuron.utils.hardware_config import get_logical_nc_config
from sglang_neuron.utils.neuron_utils import can_run_kernel

P_MAX = 128

_gen_mask_tkg_hbm = nki.jit()(gen_mask_tkg_hbm)


def gen_attention_decode_mask(pos_ids: Tensor, *, bs: int, q_head: int, s_active: int, s_prior: int, block_len: int,
                              start_pos: Optional[Tensor] = None, active_gate: Optional[Tensor] = None) -> Tensor:
    """The [s_prior, bs, q_head, s_active] 0/1 mask of a decode step over a paged prior of ``s_prior`` slots.

    ``pos_ids`` [1, bs * s_active] ends each row's prior (exclusive) and ``start_pos``, for a sliding window, starts it.
    The step's own tokens sit in the last ``s_active`` slots and see each other causally. ``active_gate`` [bs] (DCP,
    one token a row) is 1 where this rank keeps the row's new token, so the group's merge counts it once.
    """
    if s_prior % P_MAX:
        raise ValueError(f"the decode mask's prior ({s_prior}) must be a multiple of {P_MAX}")
    active = torch.triu(torch.ones(s_active, s_active, dtype=torch.float32, device=pos_ids.device))
    active_mask = active[:, None, None, :].expand(s_active, bs, q_head, s_active)
    if active_gate is not None:
        if s_active != 1:
            raise ValueError("decode context parallelism decodes one token a row")
        active_mask = active_mask * active_gate.to(torch.float32).reshape(1, bs, 1, 1)
    active_mask = active_mask.contiguous()
    if can_run_kernel(pos_ids):
        return wrap_nki(_gen_mask_tkg_hbm)[get_logical_nc_config()](
            pos_ids.to(torch.float32), bs=bs, q_head=q_head, s_active=s_active, s_prior=s_prior,
            start_pos_hbm=start_pos, block_len=block_len, active_mask=active_mask,
        )
    return _torch_gen_attention_decode_mask(pos_ids, bs, q_head, s_active, s_prior, start_pos, block_len, active_mask)


def resize_block_len(block_len: int, bs: int, q_head: int, s_active: int, s_prior: int) -> int:
    """The page length the decode kernels work with: they shrink the page until a row's pages are a multiple of the
    partitions they tile (the LNC degree times P_MAX where they shard the prior, else P_MAX).

    This mirrors gen_mask_tkg_hbm's batch-sharding rule, including its ``s_prior <= lnc * P_MAX`` threshold; the two
    must agree or the mask is read in the wrong order.
    """
    lnc = get_logical_nc_config()
    batch_sharded = bs % lnc == 0 and (bs * q_head * s_active >= P_MAX or s_prior <= lnc * P_MAX)
    partitions = (lnc if not batch_sharded and s_prior >= lnc * P_MAX else 1) * P_MAX
    bucket_len = s_prior // block_len * block_len
    if bucket_len % partitions:
        return block_len
    # nkilib's resize, without its prints (not traceable)
    return math.gcd(block_len, bucket_len // partitions)


def _torch_gen_attention_decode_mask(pos_ids, bs, q_head, s_active, s_prior, start_pos, block_len, active_mask):
    """The kernel's mask: the prior part is the same for every token of a row (it ends at the row's first token, since
    the cache holds no later one) and the active part is ``active_mask``, placed at the slots of the last ``s_active``
    sequence positions."""
    block_len = resize_block_len(block_len, bs, q_head, s_active, s_prior)
    slot = _slot_positions(s_prior, block_len, pos_ids.device).view(s_prior, 1, 1, 1)
    end = pos_ids.view(bs, s_active).min(dim=1, keepdim=True).values.view(1, bs, 1, 1)
    if start_pos is None:
        mask = slot < end
    else:
        start = start_pos.view(bs, s_active).view(1, bs, 1, s_active)
        after_start, before_end = slot >= start, slot < end
        # a window whose start passed the end wraps around the ring of slots
        mask = torch.where(start > end, after_start | before_end, after_start & before_end)
    mask = mask.to(torch.float32).expand(s_prior, bs, q_head, s_active).contiguous()
    for k in range(s_active):
        mask[_slot_of(s_prior - s_active + k, block_len)] = active_mask[k]
    return mask


def _slot_positions(s_prior: int, block_len: int, device) -> Tensor:
    """The sequence position each slot of the kernel's [tiles, P_MAX] output holds."""
    i = torch.arange(s_prior, device=device, dtype=torch.float32)
    tile, partition = i // P_MAX, i % P_MAX
    return (tile // block_len) * P_MAX * block_len + partition * block_len + tile % block_len


def _slot_of(position: int, block_len: int) -> int:
    """The inverse of _slot_positions: the slot that holds sequence position ``position``."""
    fold, within = divmod(position, P_MAX * block_len)
    partition, offset = divmod(within, block_len)
    return (fold * block_len + offset) * P_MAX + partition
