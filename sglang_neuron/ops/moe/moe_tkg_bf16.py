# SPDX-License-Identifier: Apache-2.0
"""The bf16 experts of a sparse MoE on this rank: nkilib's moe_tkg kernel runs every local expert's SiLU MLP over up to
128 tokens per call and sums the outputs scaled by the tokens' affinities, an expert a token was not routed to masked
out. A decode bucket of more than 128 rows runs it on 128-token tiles. On a CPU the torch path computes the same
sum."""

import nki
import torch
import torch.nn.functional as F
from nkilib.core.moe.moe_tkg.moe_tkg import moe_tkg
from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode
from torch import Tensor

from sglang_neuron.backend.stack import wrap_nki
from sglang_neuron.utils.hardware_config import get_logical_nc_config
from sglang_neuron.utils.neuron_utils import can_run_kernel

# the kernel's token limit per call
MAX_TOKENS = 128


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
    kernel = wrap_nki(_moe_tkg_all_experts)[get_logical_nc_config()]
    tiles = [kernel(hidden[s:s + MAX_TOKENS], gate_up, down, affinities[s:s + MAX_TOKENS], index[s:s + MAX_TOKENS],
                    rank_id)
             for s in range(0, hidden.shape[0], MAX_TOKENS)]
    return tiles[0] if len(tiles) == 1 else torch.cat(tiles, dim=0)


def _torch_moe_experts(hidden, gate_up, down, affinities, rank_id):
    local = gate_up.shape[0]
    experts = torch.arange(local, device=hidden.device) + rank_id.reshape(1).to(torch.int64) * local
    weights = affinities.index_select(1, experts)                                   # [T, E_local]
    x = hidden.float()
    gate = torch.einsum("th,ehi->tei", x, gate_up[:, :, 0].float())
    up = torch.einsum("th,ehi->tei", x, gate_up[:, :, 1].float())
    out = torch.einsum("tei,eih->teh", F.silu(gate) * up, down.float())
    return torch.einsum("te,teh->th", weights.float(), out).to(hidden.dtype)
