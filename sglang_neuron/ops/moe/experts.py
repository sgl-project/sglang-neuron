# SPDX-License-Identifier: Apache-2.0
"""The routed experts of a bf16 MoE layer whose experts are spread over the attention-TP ranks: each rank holds a
contiguous block of the experts, starting at ``first_expert`` (a tensor, so that every rank traces one graph), and
returns its partial sum of the routed outputs in fp32, which the caller combines and all-reduces.

A decode step on a NeuronCore runs nkilib's moe_tkg over the rank's experts: selectively, gathering only the experts its
tokens chose, for at most SELECTIVE_MAX_TOKENS tokens, and over every local expert with the unchosen ones masked above
that. A prefill, and every step on the host, runs each local expert over every token in torch, weighted by the token's
routing weight for it (zero for a token not routed to it), accumulating in fp32 as the reference model does."""

import torch
import torch.nn.functional as F
from nkilib.core.moe.moe_tkg.moe_tkg import moe_tkg
from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode
from torch import Tensor

from sglang_neuron import sampling
from sglang_neuron.backend.stack import wrap_nki
from sglang_neuron.utils.hardware_config import get_logical_nc_config
from sglang_neuron.utils.neuron_utils import can_run_kernel

# the token count up to which gathering the chosen experts beats running every local one
SELECTIVE_MAX_TOKENS = 2
# moe_tkg's token limit
MAX_FUSED_TOKENS = 128


def route(hidden: Tensor, router_weight: Tensor, top_k: int) -> tuple[Tensor, Tensor]:
    """(weights [T, k] fp32, global expert ids [T, k] int64) of each token's top-k experts by softmax score, from the
    replicated router ``router_weight`` [experts, H]. The selection is an exact reduction, never torch.topk on a device
    tensor (sampling._exact_topk)."""
    scores = torch.softmax(hidden.float() @ router_weight.float().t(), dim=-1)
    return sampling._exact_topk(scores, top_k)


def routed_experts(hidden: Tensor, gate_up: Tensor, down: Tensor, weights: Tensor, experts: Tensor,
                   first_expert: Tensor, *, fused: bool) -> Tensor:
    """[T, H] fp32 this rank's share of the routed outputs of ``hidden`` [T, H] over its experts ``gate_up`` [E, H, 2,
    I] (gate then up) and ``down`` [E, I, H]: the sum over each token's chosen experts that this rank holds of the
    expert's SiLU-gated MLP times the token's ``weights`` for it, ``experts`` the chosen global ids. ``fused`` runs
    moe_tkg on a NeuronCore."""
    local = gate_up.shape[0]
    ids = first_expert.to(torch.int64) + torch.arange(local, device=hidden.device)
    # [T, E] each token's weight for each local expert, zero where it did not choose it
    affinities = ((experts[:, :, None] == ids[None, None, :]) * weights[:, :, None]).sum(dim=1)
    if fused and can_run_kernel(hidden) and hidden.shape[0] <= MAX_FUSED_TOKENS:
        return _fused(hidden, gate_up, down, affinities, experts - first_expert.to(torch.int64))
    out = torch.zeros(hidden.shape, dtype=torch.float32, device=hidden.device)
    for e in range(local):
        gate = hidden @ gate_up[e, :, 0]
        up = hidden @ gate_up[e, :, 1]
        out = out + ((F.silu(gate) * up) @ down[e]).float() * affinities[:, e:e + 1]
    return out


def _fused(hidden, gate_up, down, affinities, local_ids):
    """moe_tkg over the rank's experts. Its index holds k distinct local experts per token: a chosen expert another
    rank holds is replaced by a local one the token did not choose, whose affinity is zero."""
    local, top_k = gate_up.shape[0], local_ids.shape[1]
    if local < 2 * top_k:
        raise ValueError(f"moe_tkg needs at least 2 x top-k local experts, so that every token has k distinct ones; "
                         f"this rank holds {local} for a top-{top_k}")
    held = (local_ids >= 0) & (local_ids < local)
    columns = torch.arange(local, device=hidden.device)
    chosen = ((local_ids[:, :, None] == columns) & held[:, :, None]).any(dim=1)
    # the j-th local expert the token did not choose, for its j-th replaced choice
    free = ~chosen
    order = torch.cumsum(free.to(torch.int32), dim=-1) - 1
    fillers = torch.stack([torch.where(free & (order == j), columns, 0).sum(-1) for j in range(top_k)], dim=-1)
    index = torch.where(held, local_ids, fillers).to(torch.int32)
    grid = get_logical_nc_config() if hidden.shape[1] % 256 == 0 else 1
    if hidden.shape[0] <= SELECTIVE_MAX_TOKENS:
        return wrap_nki(_selective)[grid](hidden, gate_up, down, affinities.to(torch.float32), index).float()
    rank = torch.zeros(1, 1, dtype=torch.int32, device=hidden.device)
    return wrap_nki(_all_experts)[grid](hidden, gate_up, down, affinities.to(torch.float32), index, rank).float()


def _selective(hidden, gate_up, down, affinities, index):
    # POST_SCALE weighs each chosen expert's output by its affinity
    return moe_tkg(hidden, gate_up, down, affinities, index, is_all_expert=False, activation_fn=ActFnType.SiLU,
                   expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE)


def _all_experts(hidden, gate_up, down, affinities, index, rank):
    # rank 0: the weights are this rank's experts alone, so every expert the kernel sees is local
    return moe_tkg(hidden, gate_up, down, affinities, index, is_all_expert=True, rank_id=rank,
                   activation_fn=ActFnType.SiLU, mask_unselected_experts=True,
                   expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE)
