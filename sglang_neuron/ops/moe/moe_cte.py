# SPDX-License-Identifier: Apache-2.0
"""The prefill MoE experts: nkilib's moe_cte kernel in its MXFP4 block-sharded form (shard_on_block_mx), over the
token blocks build_blockwise_mapping assigns to this rank's experts. The kernel reads each token's routing
affinities from the packed rows rmsnorm_mx_prefill produces."""

import nki
import nki.language as nl
from nkilib.core.moe.moe_cte.moe_cte import (
    ActFnType,
    ExpertAffinityScaleMode,
    MoECTEImplementation,
    MoECTESpec,
    QuantizationConfig,
    ShardOnBlockConfig,
    ShardOnIConfig,
    SkipMode,
)
from nkilib.core.moe.moe_cte.moe_cte import moe_cte as _nki_moe_cte
from torch import Tensor

from sglang_neuron.backend.stack import wrap_nki
from sglang_neuron.utils.hardware_config import get_logical_nc_config

# the narrowest expert intermediate shard the kernel takes
MIN_INTERMEDIATE_SHARD = 128


@nki.jit(mode="trace")
def _moe_cte(hidden_states, gate_up_proj_weight, down_proj_weight, token_position_to_id, block_to_expert, block_size,
             top_k, ep_degree, ep_rank, conditions, gate_and_up_proj_bias, down_proj_bias, gate_up_proj_scale,
             down_proj_scale, gate_clamp_upper_limit, up_clamp_upper_limit, up_clamp_lower_limit):
    spec = MoECTESpec(
        implementation=MoECTEImplementation.shard_on_block_mx,
        # -1: the kernel splits the blocks into static and dynamic ones from top_k and ep_degree
        shard_on_block=ShardOnBlockConfig(n_dynamic_blocks=-1, top_k=top_k, ep_degree=ep_degree),
        shard_on_I=ShardOnIConfig(),
    )
    return _nki_moe_cte(
        hidden_states=hidden_states,
        # None: the affinities ride in the packed rows of hidden_states
        expert_affinities_masked=None,
        gate_up_proj_weight=gate_up_proj_weight,
        down_proj_weight=down_proj_weight,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        block_size=block_size,
        spec=spec,
        conditions=conditions,
        gate_and_up_proj_bias=gate_and_up_proj_bias,
        down_proj_bias=down_proj_bias,
        quantization_config=QuantizationConfig(gate_up_proj_scale=gate_up_proj_scale, down_proj_scale=down_proj_scale),
        activation_function=ActFnType.Swish,
        skip_dma=SkipMode(skip_token=True, skip_weight=True),
        compute_dtype=nl.bfloat16,
        is_tensor_update_accumulating=True,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=None,
        up_clamp_upper_limit=up_clamp_upper_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        ep_rank=ep_rank,
    )


def moe_cte(hidden_states: Tensor, *, gate_up_proj_weight: Tensor, down_proj_weight: Tensor,
            token_position_to_id: Tensor, block_to_expert: Tensor, block_size: int, top_k: int, ep_degree: int,
            ep_rank: Tensor, conditions: Tensor, gate_and_up_proj_bias: Tensor, down_proj_bias: Tensor,
            gate_up_proj_scale: Tensor, down_proj_scale: Tensor, gate_clamp_upper_limit: float,
            up_clamp_upper_limit: float, up_clamp_lower_limit: float) -> Tensor:
    """The experts' SwiGLU outputs for the packed rows ``hidden_states`` [T, row], summed over each token's experts
    and scaled by its affinities: [T, H].

    The weights are the MXFP4 tiles (gate/up [E, 128, 2, H // 512, I] uint16 with uint8 scales, down
    [E, I_tile, I_tiles, H]); ``ep_rank`` [1, 1] int32 locates this rank's experts among the global ones the packed
    affinities are indexed by. There is no CPU path: the kernel's MX arithmetic has no torch counterpart here.
    """
    intermediate = gate_up_proj_weight.shape[-1]
    if intermediate < MIN_INTERMEDIATE_SHARD:
        raise ValueError(f"an expert intermediate shard of {intermediate} has no moe_cte kernel (it needs at least "
                         f"{MIN_INTERMEDIATE_SHARD}): serve with more expert parallelism and fewer TP ranks per expert")
    if hidden_states.device.type == "cpu":
        raise RuntimeError("the MXFP4 prefill experts run on a NeuronCore only")
    out = wrap_nki(_moe_cte)[get_logical_nc_config()](
        hidden_states=hidden_states,
        gate_up_proj_weight=gate_up_proj_weight,
        down_proj_weight=down_proj_weight,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        block_size=block_size,
        top_k=top_k,
        ep_degree=ep_degree,
        ep_rank=ep_rank,
        conditions=conditions,
        gate_and_up_proj_bias=gate_and_up_proj_bias,
        down_proj_bias=down_proj_bias,
        gate_up_proj_scale=gate_up_proj_scale,
        down_proj_scale=down_proj_scale,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
    )
    # the accumulating kernel returns a [shards, T, H] accumulator whose first shard holds the sum
    return out[0] if out.dim() == 3 else out
