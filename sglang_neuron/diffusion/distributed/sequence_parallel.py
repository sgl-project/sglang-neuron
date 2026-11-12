# SPDX-License-Identifier: Apache-2.0
"""Sequence-parallel exchanges over the tensor-parallel group."""

import torch
import torch.distributed as dist
import torch.nn.functional as F


def sp_all_gather_seq(x: torch.Tensor, tp_group, tp_size: int) -> torch.Tensor:
    # B==1: [1, S, H] reshapes to [S, H] (a view of contiguous input), so the sequence is already
    # the gather (dim 0) axis — no transpose and no contiguous copy on either side.
    if x.shape[0] == 1:
        _, seq, hidden = x.shape
        out = torch.empty((seq * tp_size, hidden), dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(out, x.reshape(seq, hidden), group=tp_group)
        return out.reshape(1, seq * tp_size, hidden)
    seq_first = x.transpose(0, 1).contiguous()
    out = torch.empty(
        (seq_first.shape[0] * tp_size, *seq_first.shape[1:]),
        dtype=seq_first.dtype,
        device=seq_first.device,
    )
    dist.all_gather_into_tensor(out, seq_first, group=tp_group)
    return out.transpose(0, 1)


def sp_reduce_scatter_seq(
    x: torch.Tensor, tp_group, tp_size: int, reduce_dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Reduce-scatter ``x`` over the TP group along the sequence (dim 1).

    ``reduce_dtype`` (e.g. fp32): sum in that dtype and round once to ``x.dtype``, as an
    all-reduce that accumulates in fp32 does (a bf16 reduce-scatter may round every partial).
    """
    if reduce_dtype is not None and reduce_dtype != x.dtype:
        return sp_reduce_scatter_seq(x.to(reduce_dtype), tp_group, tp_size).to(x.dtype)
    # B==1: [1, S, H] reshapes to [S, H] (a view of contiguous input), so the sequence is already
    # the scatter (dim 0) axis — no transpose, and the contiguous RS output reshapes back to a view,
    # so neither the input nor the returned tensor needs a copy.
    if x.shape[0] == 1:
        _, seq, hidden = x.shape
        out = torch.empty((seq // tp_size, hidden), dtype=x.dtype, device=x.device)
        dist.reduce_scatter_tensor(
            out, x.reshape(seq, hidden), op=dist.ReduceOp.SUM, group=tp_group
        )
        return out.reshape(1, seq // tp_size, hidden)
    seq_first = x.transpose(0, 1).contiguous()
    out = torch.empty(
        (seq_first.shape[0] // tp_size, *seq_first.shape[1:]),
        dtype=seq_first.dtype,
        device=seq_first.device,
    )
    dist.reduce_scatter_tensor(out, seq_first, op=dist.ReduceOp.SUM, group=tp_group)
    return out.transpose(0, 1)


def sp_padded_len(real_len: int, tp_size: int) -> int:
    return ((real_len + tp_size - 1) // tp_size) * tp_size


def sp_exit(
    output: torch.Tensor,
    tp_group,
    tp_size: int,
    sp_enabled: bool,
    padded_len: int | None = None,
    reduce_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if tp_size <= 1:
        return output
    if not sp_enabled:
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=tp_group)
        return output
    if padded_len is not None and output.shape[1] < padded_len:
        output = F.pad(output, (0, 0, 0, padded_len - output.shape[1]))
    return sp_reduce_scatter_seq(output, tp_group, tp_size, reduce_dtype)


def sp_entry(
    x: torch.Tensor,
    tp_group,
    tp_size: int,
    sp_enabled: bool,
    real_len: int | None = None,
) -> torch.Tensor:
    if tp_size <= 1 or not sp_enabled:
        return x
    gathered = sp_all_gather_seq(x, tp_group, tp_size)
    if real_len is not None and gathered.shape[1] > real_len:
        gathered = gathered[:, :real_len, :]
    return gathered
