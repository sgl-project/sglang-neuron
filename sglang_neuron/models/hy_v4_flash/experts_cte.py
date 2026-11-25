# SPDX-License-Identifier: Apache-2.0
"""Hy4-Flash's prefill routed experts: nkilib's moe_cte in its hybrid I-sharded form (shard_on_i_hybrid).

At EP32 a rank holds 16 of the 512 experts and a token picks 16, so a static block layout must make room for every
token on every local expert: as many blocks as running all of them densely. The routing fills about 1/32 of that, and
the hybrid kernel runs ``num_static_block`` blocks and walks the rest at run time, skipping each one ``conditions``
marks empty. Its blocks are 256 tokens (the I-sharded kernels take a multiple of 256); each core takes half of the
intermediate dimension.

The FP8 experts (moe.py) go to the kernel as they are, with their per-row scales spread per channel; their SwiGLU
clamps the gate above and the up both ways at ``swiglu_limit``."""

import nki
import nki.language as nl
import torch
import torch.nn.functional as F
from nkilib.core.moe.moe_cte.moe_cte import (ActFnType, ExpertAffinityScaleMode, MoECTEImplementation, MoECTESpec,
                                             ShardOnBlockConfig, ShardOnIConfig, SkipMode)
from nkilib.core.moe.moe_cte.moe_cte import moe_cte as _nki_moe_cte

from sglang_neuron.backend.stack import wrap_nki
from sglang_neuron.ops.moe.moe_blockwise import build_blockwise_mapping
from sglang_neuron.utils.hardware_config import get_logical_nc_config

BLOCK_SIZE = 256


@nki.jit(mode="trace")
def _moe_cte_hybrid(hidden_states, expert_affinities_masked, gate_up_proj_weight, down_proj_weight,
                    token_position_to_id, block_to_expert, conditions, gate_up_proj_scale, down_proj_scale,
                    num_static_block, limit):
    spec = MoECTESpec(implementation=MoECTEImplementation.shard_on_i_hybrid, shard_on_block=ShardOnBlockConfig(),
                      shard_on_I=ShardOnIConfig(num_static_block=num_static_block))
    return _nki_moe_cte(
        hidden_states=hidden_states, expert_affinities_masked=expert_affinities_masked,
        gate_up_proj_weight=gate_up_proj_weight, down_proj_weight=down_proj_weight,
        token_position_to_id=token_position_to_id, block_to_expert=block_to_expert, block_size=BLOCK_SIZE, spec=spec,
        conditions=conditions, gate_up_proj_scale=gate_up_proj_scale, down_proj_scale=down_proj_scale,
        activation_function=ActFnType.SiLU, skip_dma=SkipMode(skip_token=True, skip_weight=False),
        compute_dtype=nl.bfloat16, is_tensor_update_accumulating=True,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
        gate_clamp_upper_limit=limit, gate_clamp_lower_limit=None, up_clamp_upper_limit=limit,
        up_clamp_lower_limit=-limit)


def prefill_experts(x: torch.Tensor, gate_up: torch.Tensor, down: torch.Tensor, gate_up_scale: torch.Tensor,
                    down_scale: torch.Tensor, affinities: torch.Tensor, top_k: int, real: torch.Tensor,
                    limit: float) -> torch.Tensor:
    """[T, H] this rank's routed-expert output for x [T, H] over its experts (gate_up [E, H, 2, I] FP8 with scales
    [E, 2, I], down [E, I, H] FP8 with scales [E, H]); affinities [T, E] fp32, zero off a token's experts; real [T]
    the chunk's real tokens (padding rows stay out of the mapping)."""
    tokens = x.shape[0]
    experts, intermediate, hidden = down.shape
    pad = experts % 2
    token_position_to_id, block_to_expert, conditions = build_blockwise_mapping(
        F.pad(affinities, (0, pad)), num_local_experts=experts + pad, num_experts_per_token=top_k,
        block_size=BLOCK_SIZE, moe_group=None, tp_degree=1, padding_mask=real)
    block_to_expert = torch.clamp(block_to_expert, max=experts - 1)
    # the hybrid kernel reads two trailing conditions past the blocks
    conditions = F.pad(conditions, (0, 2))
    gu_scale = gate_up_scale.float().reshape(experts, 1, 2 * intermediate)
    dn_scale = down_scale.float().reshape(experts, 1, hidden)
    out = wrap_nki(_moe_cte_hybrid)[get_logical_nc_config()](
        x, affinities.to(x.dtype).reshape(-1, 1), gate_up, down, token_position_to_id, block_to_expert.reshape(-1, 1),
        conditions, gu_scale, dn_scale, num_static_block=experts, limit=float(limit))
    return (out[:, 0] if out.dim() == 3 else out)[:tokens]
