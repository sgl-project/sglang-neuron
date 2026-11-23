# SPDX-License-Identifier: Apache-2.0
"""The bf16 experts of a sparse MoE on this rank: nkilib's moe_tkg kernel runs every local expert's SiLU MLP over up to
128 tokens per call and sums the outputs scaled by the tokens' affinities, an expert a token was not routed to masked
out. A decode bucket of more than 128 rows runs it on 128-token tiles. On a CPU the torch path computes the same
sum.

All-expert mode reads every local expert's weights at every step, whatever the batch. For a few tokens the selective
mode (``SGLANG_NEURON_MOE_DECODE_SELECTIVE``: up to that many tokens, default SELECTIVE_MAX_TOKENS, 0 off) reads only
the experts the tokens chose: each token's k experts, a choice another rank holds replaced by a local
expert the token did not choose, with affinity zero (as ops/moe/experts.py does for DeepSeek-V2)."""

import nki
import torch
import torch.nn.functional as F
from nkilib.core.moe.moe_tkg.moe_tkg import moe_tkg
from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode
from torch import Tensor

from sglang_neuron import envs
from sglang_neuron.backend.stack import wrap_nki
from sglang_neuron.utils.hardware_config import get_logical_nc_config
from sglang_neuron.utils.neuron_utils import can_run_kernel

# the kernel's token limit per call
MAX_TOKENS = 128
# the default token count up to which the selective mode runs: k distinct experts a token, so at T tokens it reads
# at most T * k experts, fewer than the rank's E_local while T * k < E_local (measured faster up to 4 tokens on trn2
# and trn3 for Qwen3-30B-A3B at EP4)
SELECTIVE_MAX_TOKENS = 4


@nki.jit
def _moe_tkg_selected(hidden, gate_up, down, affinities, index):
    # POST_SCALE weighs each chosen expert's output by its affinity (zero for a filler)
    return moe_tkg(hidden, gate_up, down, affinities, index, is_all_expert=False, activation_fn=ActFnType.SiLU,
                   expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE)


@nki.jit
def _moe_tkg_all_experts(hidden, gate_up, down, affinities, index, rank_id):
    return moe_tkg(
        hidden, gate_up, down, affinities, index,
        is_all_expert=True,
        rank_id=rank_id,
        activation_fn=ActFnType.SiLU,
        mask_unselected_experts=True,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
    )


def moe_experts(hidden: Tensor, *, gate_up: Tensor, down: Tensor, affinities: Tensor, index: Tensor,
                rank_id: Tensor) -> Tensor:
    """This rank's experts on hidden [T, H]: sum over the local experts of affinity * down(silu(gate) * up), [T, H].

    gate_up is [E_local, H, 2, I] (gate at 0, up at 1), down [E_local, I, H]; affinities [T, E] fp32 hold each token's
    weight for the experts it was routed to (zero elsewhere) and index [T, K] int32 those experts, both over the global
    experts; rank_id [1, 1] int32 is this rank's EP rank, whose experts are [E_local * rank, E_local * (rank + 1))."""
    if not can_run_kernel(hidden):
        return _torch_moe_experts(hidden, gate_up, down, affinities, rank_id)
    if hidden.shape[0] <= _selective_tokens(gate_up.shape[0], index.shape[1]):
        return _selected(hidden, gate_up, down, affinities, index, rank_id)
    kernel = wrap_nki(_moe_tkg_all_experts)[get_logical_nc_config()]
    tiles = [kernel(hidden[s:s + MAX_TOKENS], gate_up, down, affinities[s:s + MAX_TOKENS], index[s:s + MAX_TOKENS],
                    rank_id)
             for s in range(0, hidden.shape[0], MAX_TOKENS)]
    return tiles[0] if len(tiles) == 1 else torch.cat(tiles, dim=0)


def _selective_tokens(local: int, top_k: int) -> int:
    """The most tokens a decode runs in the selective mode (0: never)."""
    if local < 2 * top_k:  # a token's choices held elsewhere need distinct local fillers
        return 0
    value = envs.moe_decode_selective_tokens()
    return SELECTIVE_MAX_TOKENS if value is None else value


def _selected(hidden, gate_up, down, affinities, index, rank_id):
    """moe_tkg over the experts the tokens chose. ``index`` [T, k] holds global expert ids; a choice this rank does not
    hold becomes the j-th local expert the token did not choose, with zero affinity, so every token names k distinct
    local experts."""
    local, top_k = gate_up.shape[0], index.shape[1]
    first = rank_id.reshape(1).to(torch.int64) * local
    local_ids = index.to(torch.int64) - first
    held = (local_ids >= 0) & (local_ids < local)
    columns = torch.arange(local, device=hidden.device)
    chosen = ((local_ids[:, :, None] == columns) & held[:, :, None]).any(dim=1)
    free = ~chosen
    order = torch.cumsum(free.to(torch.int32), dim=-1) - 1
    fillers = torch.stack([torch.where(free & (order == j), columns, 0).sum(-1) for j in range(top_k)], dim=-1)
    local_index = torch.where(held, local_ids, fillers).to(torch.int32)
    # [T, E_local] the tokens' weights for the local experts
    weights = affinities.index_select(1, first + columns).to(torch.float32)
    return wrap_nki(_moe_tkg_selected)[get_logical_nc_config()](hidden, gate_up, down, weights, local_index)


def _torch_moe_experts(hidden, gate_up, down, affinities, rank_id):
    local = gate_up.shape[0]
    experts = torch.arange(local, device=hidden.device) + rank_id.reshape(1).to(torch.int64) * local
    weights = affinities.index_select(1, experts)                                   # [T, E_local]
    x = hidden.float()
    gate = torch.einsum("th,ehi->tei", x, gate_up[:, :, 0].float())
    up = torch.einsum("th,ehi->tei", x, gate_up[:, :, 1].float())
    out = torch.einsum("tei,eih->teh", F.silu(gate) * up, down.float())
    return torch.einsum("te,teh->th", weights.float(), out).to(hidden.dtype)
