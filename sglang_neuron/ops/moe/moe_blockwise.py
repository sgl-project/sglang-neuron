# SPDX-License-Identifier: Apache-2.0
"""The token-to-block mapping of the prefill MoE: each local expert gets whole blocks of ``block_size`` token slots
for the tokens routed to it, in expert order. On a NeuronCore the find_nonzero_indices and indexed_flatten kernels
build it where the shapes suit them; elsewhere the same mapping is built in torch."""

import math

import nki
import torch
from nkilib.core.subkernels.find_nonzero_indices import find_nonzero_indices
from nkilib.core.subkernels.indexed_flatten import indexed_flatten
from torch import Tensor

from sglang_neuron.backend.stack import wrap_nki
from sglang_neuron.parallel import collectives as neuron_collectives
from sglang_neuron.utils.hardware_config import get_logical_nc_config
from sglang_neuron.utils.neuron_utils import can_run_kernel

_find_nonzero_indices = nki.jit()(find_nonzero_indices)
_indexed_flatten = nki.jit()(indexed_flatten)
_MAX_CHUNK = 16384


def build_blockwise_mapping(expert_affinities: Tensor, *, num_local_experts: int, num_experts_per_token: int,
                            block_size: int, moe_group, tp_degree: int, padding_mask: Tensor):
    """``(token_position_to_id, block_to_expert, conditions)`` for moe_cte from the [T, E_local] affinities (zero for
    the experts a token was not routed to) of the real tokens ``padding_mask`` [T] marks.

    token_position_to_id [N * block_size] holds a token per slot (-1 for an empty one), block_to_expert [N] the expert
    of each block and conditions [N] 1 for a block with a token. Across the ``tp_degree`` ranks of ``moe_group``, which
    hold every local expert with a slice of its intermediate dimension, each rank maps a share of the experts and the
    shares are combined.
    """
    tokens = expert_affinities.shape[0]
    selected = ((expert_affinities != 0) & padding_mask.to(torch.bool).unsqueeze(1)).to(torch.float32)
    chunk = min(tokens, _MAX_CHUNK)
    f_len = min(128, tokens // 16)
    sharded = tp_degree > 1 and num_local_experts % tp_degree == 0
    experts_per_rank = num_local_experts // tp_degree if sharded else num_local_experts
    num_blocks = _num_blocks(tokens, num_experts_per_token, num_local_experts, block_size)
    if _kernels_apply(selected, tokens, experts_per_rank, chunk, f_len, block_size):
        token_position_to_id, block_to_expert = _mapping_by_kernels(
            selected, num_local_experts, experts_per_rank, sharded, num_blocks, block_size, tokens, chunk, f_len,
            moe_group)
    else:
        token_position_to_id, block_to_expert = _mapping_in_torch(
            selected, num_local_experts, experts_per_rank, sharded, num_blocks, block_size, tokens, moe_group)
    conditions = torch.any(token_position_to_id.view(num_blocks, block_size) != -1, dim=1).to(torch.int32)
    return token_position_to_id.to(torch.int32), block_to_expert.to(torch.int32), conditions


def _num_blocks(tokens: int, top_k: int, experts: int, block_size: int) -> int:
    """The most blocks the routing can fill: every token's routes in full blocks, plus a partial block for every
    expert but one."""
    routes = tokens * min(top_k, experts)
    return min(math.ceil((routes - (experts - 1)) / block_size) + experts - 1, routes)


def _kernels_apply(selected: Tensor, tokens: int, experts: int, chunk: int, f_len: int, block_size: int) -> bool:
    """The kernels' shape limits: find_nonzero_indices splits the experts over the two cores and walks the tokens in
    chunks of a multiple of 128; indexed_flatten places rows of f_len slots, which must tile a block and split the
    tokens into a multiple of 16 rows."""
    return (can_run_kernel(selected) and experts % 2 == 0 and tokens % chunk == 0 and chunk % 128 == 0 and f_len > 0
            and block_size % f_len == 0 and tokens % f_len == 0 and (tokens // f_len) % 16 == 0)


def _mapping_by_kernels(selected, num_local_experts, experts_per_rank, sharded, num_blocks, block_size, tokens, chunk,
                        f_len, moe_group):
    device = selected.device
    lnc = get_logical_nc_config()
    # The first expert column of this rank's share, from the group's device rank tensor so that every rank traces one
    # graph. Computed without reshape or modulo, which dynamo would fold into a per-rank constant under chunked prefill.
    first = ((moe_group.rank_in_group_tensor.to(torch.int32) * experts_per_rank).to(torch.int32) if sharded
             else torch.zeros(1, dtype=torch.int32, device=device))
    indices, tokens_per_expert = wrap_nki(_find_nonzero_indices)[lnc](
        input_tensor=selected, col_start_id=first, n_cols=experts_per_rank, chunk_size=chunk)
    if sharded:
        tokens_per_expert = neuron_collectives.all_gather(moe_group, tokens_per_expert, dim=0)
    blocks_per_expert = ((tokens_per_expert + block_size - 1) // block_size).to(torch.long)
    first_block = torch.cat([torch.zeros(1, 1, dtype=torch.long, device=device),
                             _cumsum_matmul(blocks_per_expert.unsqueeze(1))[:-1]])
    flat = wrap_nki(_indexed_flatten)[lnc](
        input_tensor=indices,
        f_len=f_len,
        output_len=num_blocks * block_size + tokens,
        # in f_len-element rows
        row_offsets=(first_block * (block_size // f_len)).reshape(-1).to(torch.int32),
        row_offsets_start=first,
        padding_val=-1,
    )
    token_position_to_id = flat[:num_blocks * block_size].clone()
    if sharded:
        # the ranks' shares combine by max: an empty slot is -1 on every rank but the one that fills it
        token_position_to_id = neuron_collectives.all_reduce(moe_group, token_position_to_id, "max")
    block_to_expert = torch.sum(torch.arange(num_blocks, device=device, dtype=torch.int32).unsqueeze(0)
                                >= first_block[1:], dim=0)
    return token_position_to_id, block_to_expert


def _mapping_in_torch(selected, num_local_experts, experts_per_rank, sharded, num_blocks, block_size, tokens,
                      moe_group):
    device = selected.device

    def share(t):
        # this rank's expert columns, by a rank-tensor offset so that every rank traces one graph
        first = moe_group.rank_in_group_tensor.to(torch.int64) * experts_per_rank
        return t.index_select(1, first + torch.arange(experts_per_rank, device=device, dtype=torch.int64))

    if sharded:
        tokens_per_expert = neuron_collectives.all_gather(moe_group, share(selected).sum(dim=0), dim=0)
    else:
        tokens_per_expert = selected.sum(dim=0)
    blocks_per_expert = ((tokens_per_expert + block_size - 1) // block_size).to(torch.long)
    end_block = _cumsum_matmul(blocks_per_expert.unsqueeze(1)).squeeze(1)
    block_to_expert = torch.sum(torch.arange(num_blocks, device=device).unsqueeze(1) >= end_block[:-1], dim=1)

    # each route's slot: its rank among the expert's tokens, after the expert's earlier blocks; slot 0 of the output
    # is a sink for the unselected pairs, dropped at the end
    slot = _cumsum_matmul(selected)
    slot[:, 1:] += end_block[:-1] * block_size
    slot = slot.masked_fill(selected == 0, 0).to(torch.long)
    if sharded:
        slot = share(slot).masked_fill(share(selected) == 0, 0)
    out = torch.full((num_blocks * block_size + 1,), -1, device=device, dtype=torch.long)
    out[slot] = torch.arange(tokens, device=device, dtype=torch.long).unsqueeze(1)
    token_position_to_id = out[1:]
    if sharded:
        token_position_to_id = neuron_collectives.all_reduce(moe_group, token_position_to_id, "max")
    return token_position_to_id, block_to_expert


def _cumsum_matmul(t: Tensor, tile: int = 2048) -> Tensor:
    """The cumulative sum over the rows of a 2-D tensor, as lower-triangular matmuls in tiles of ``tile`` rows: a
    matmul runs on the tensor engine where a cumsum would not."""
    rows = t.shape[0]
    running = torch.zeros(1, t.shape[1], dtype=torch.float32, device=t.device)
    out = []
    for start in range(0, rows, tile):
        n = min(tile, rows - start)
        tril = torch.tril(torch.ones(n, n, device=t.device, dtype=torch.float32))
        part = running + tril @ t.narrow(0, start, n).to(torch.float32)
        out.append(part)
        running = part.narrow(0, -1, 1)
    return torch.cat(out).to(t.dtype)
