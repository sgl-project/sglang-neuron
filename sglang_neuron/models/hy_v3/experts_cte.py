# SPDX-License-Identifier: Apache-2.0
"""Hy3's prefill routed experts: nkilib's moe_cte in its hybrid I-sharded form (shard_on_i_hybrid).

At EP 64 a rank holds 3 of the 192 experts, so a chunk's mapping (ops/moe/moe_blockwise.py) has room for every token
on every local expert, a worst case of about 3 T / block blocks of which the routing fills a few. The block-sharded
bf16 kernel (ops/moe/moe_cte_bf16.py) runs all of them; the hybrid kernel runs ``num_static_block`` blocks and walks
the rest at run time, skipping each one ``conditions`` marks empty. Its blocks are 256 tokens (the I-sharded kernels
take a multiple of 256). Each core takes half of the intermediate dimension, so a hot expert's blocks never split
between the cores.

The mapping kernels split the expert columns over the two cores of the logical NeuronCore, so an odd count of local
experts would build the mapping in torch; one never-chosen expert column pads it to the kernel path, with the same
blocks (the pad gets none).

FP8 experts (experts_fp8.py) go to the kernel as they are, with their per-tensor scales spread per channel; the
kernel applies them after each matmul. They run on its copy that feeds the FP8 tiles to the matmul directly
(kernels/moe_cte_fp8w.py, kernels/PROVENANCE.md)."""

import nki
import nki.language as nl
import torch
import torch.nn.functional as F
from nkilib.core.moe.moe_cte.moe_cte import (ActFnType, ExpertAffinityScaleMode, MoECTEImplementation, MoECTESpec,
                                             ShardOnBlockConfig, ShardOnIConfig, SkipMode)
from nkilib.core.moe.moe_cte.moe_cte import moe_cte as _nki_moe_cte

from sglang_neuron.backend.stack import wrap_nki
from sglang_neuron.models.hy_v3.kernels.moe_cte_fp8w import moe_cte_fp8w
from sglang_neuron.ops.moe.moe_blockwise import build_blockwise_mapping
from sglang_neuron.utils.hardware_config import get_logical_nc_config

BLOCK_SIZE = 256


@nki.jit(mode="trace")
def _moe_cte_hybrid(hidden_states, expert_affinities_masked, gate_up_proj_weight, down_proj_weight,
                    token_position_to_id, block_to_expert, conditions, gate_up_proj_scale, down_proj_scale,
                    num_static_block):
    spec = MoECTESpec(implementation=MoECTEImplementation.shard_on_i_hybrid, shard_on_block=ShardOnBlockConfig(),
                      shard_on_I=ShardOnIConfig(num_static_block=num_static_block))
    return _nki_moe_cte(
        hidden_states=hidden_states, expert_affinities_masked=expert_affinities_masked,
        gate_up_proj_weight=gate_up_proj_weight, down_proj_weight=down_proj_weight,
        token_position_to_id=token_position_to_id, block_to_expert=block_to_expert, block_size=BLOCK_SIZE, spec=spec,
        conditions=conditions, gate_up_proj_scale=gate_up_proj_scale, down_proj_scale=down_proj_scale,
        activation_function=ActFnType.SiLU, skip_dma=SkipMode(skip_token=True, skip_weight=False),
        compute_dtype=nl.bfloat16, is_tensor_update_accumulating=True,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE)


def prefill_experts(x: torch.Tensor, gate_up: torch.Tensor, down: torch.Tensor, affinities: torch.Tensor, top_k: int,
                    real: torch.Tensor, gate_up_scale: torch.Tensor = None, down_scale: torch.Tensor = None):
    """[T, H] this rank's routed-expert output for x [T, H] over its experts (gate_up [E, H, 2, I], down [E, I, H], bf16
    or FP8 with per-tensor scales gate_up_scale [E, 2, 1] and down_scale [E, 1]); affinities [T, E] fp32, zero off a
    token's experts; real [T] marks the chunk's real tokens."""
    tokens = x.shape[0]
    experts, intermediate, hidden = down.shape
    pad = experts % 2
    token_position_to_id, block_to_expert, conditions = build_blockwise_mapping(
        F.pad(affinities, (0, pad)), num_local_experts=experts + pad, num_experts_per_token=top_k,
        block_size=BLOCK_SIZE, moe_group=None, tp_degree=1, padding_mask=real)
    # the pad expert has no blocks; its block ids past the real experts' map to an empty block
    block_to_expert = torch.clamp(block_to_expert, max=experts - 1)
    # the hybrid kernel reads two trailing conditions past the blocks
    conditions = F.pad(conditions, (0, 2))
    if gate_up_scale is not None:
        gate_up_scale = gate_up_scale.float().expand(experts, 2, intermediate).reshape(experts, 1, 2 * intermediate)
        down_scale = down_scale.float().expand(experts, hidden).reshape(experts, 1, hidden)
    masked = affinities.to(x.dtype).reshape(-1, 1)
    if gate_up_scale is not None:
        # FP8: the hybrid kernel's copy that feeds the FP8 weight tiles to the matmul as they are
        # (kernels/moe_cte_fp8w.py; nkilib's copies each tile to bf16 first, most of the experts' time)
        out = wrap_nki(moe_cte_fp8w)[get_logical_nc_config()](
            conditions, x, masked, gate_up, down, token_position_to_id, block_to_expert.reshape(-1, 1),
            gate_up_scale, down_scale, block_size=BLOCK_SIZE, num_static_block=experts, skip_token=True)
    else:
        out = wrap_nki(_moe_cte_hybrid)[get_logical_nc_config()](
            x, masked, gate_up, down, token_position_to_id, block_to_expert.reshape(-1, 1), conditions, None, None,
            num_static_block=experts)
    return (out[:, 0] if out.dim() == 3 else out)[:tokens]
