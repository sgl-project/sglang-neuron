# SPDX-License-Identifier: Apache-2.0
"""Entering and leaving MiniMax-H3's TP regions under sequence parallelism, with the sequence
padded to a multiple of the TP size."""

import torch
import torch.distributed as dist
import torch.nn.functional as F

from sglang_neuron.diffusion.distributed.sequence_parallel import (
    sp_all_gather_seq,
    sp_reduce_scatter_seq,
)


def sp_padded_len(real_len: int, tp_size: int) -> int:
    """``real_len`` rounded up to a multiple of ``tp_size``."""
    return ((real_len + tp_size - 1) // tp_size) * tp_size


def sp_exit(
    output: torch.Tensor,
    tp_group,
    tp_size: int,
    sp_enabled: bool,
    padded_len: int | None = None,
) -> torch.Tensor:
    """Leave a TP region: reduce-scatter the partial sums over the sequence, padded to
    ``padded_len`` (SP), else all-reduce them."""
    if tp_size <= 1:
        return output
    if not sp_enabled:
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=tp_group)
        return output
    if padded_len is not None and output.shape[1] < padded_len:
        output = F.pad(output, (0, 0, 0, padded_len - output.shape[1]))
    return sp_reduce_scatter_seq(output, tp_group, tp_size)


def sp_entry(
    x: torch.Tensor,
    tp_group,
    tp_size: int,
    sp_enabled: bool,
    real_len: int | None = None,
) -> torch.Tensor:
    """Enter a TP region: all-gather the sequence shards and drop the padding past ``real_len``
    (SP); without SP every rank already holds the whole sequence."""
    if tp_size <= 1 or not sp_enabled:
        return x
    gathered = sp_all_gather_seq(x, tp_group, tp_size)
    if real_len is not None and gathered.shape[1] > real_len:
        gathered = gathered[:, :real_len, :]
    return gathered
