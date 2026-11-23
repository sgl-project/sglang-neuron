# SPDX-License-Identifier: Apache-2.0
"""The bf16 prefill experts of a sparse MoE on this rank: nkilib's moe_cte kernel (shard_on_block) runs each local
expert's MLP (SiLU, or GPT-OSS's clamped SwiGLU with biases) on blocks of the tokens routed to it, the token-to-block
mapping from build_blockwise_mapping, so an expert's weights are read once per block of its own tokens. The kernel has
no dynamic loop: it runs every block the routing could fill (every token's top-k on this rank), the ones no token
filled included, each at a full block's cost. On a CPU the torch path computes the same blocks.

With whole experts on the rank (EP equal to the attention TP) the routing fills only about ``tokens * top_k / EP``
of those slots, so most of the blocks are empty. There, ``hybrid`` (``SGLANG_NEURON_MOE_PREFILL``, the default on
Trainium2) runs nkilib's shard_on_i_hybrid on HYBRID_BLOCK_SIZE-token blocks: a static loop over the blocks the
routing is expected to fill, then a dynamic loop over the blocks the mapping marks filled (``conditions``), so an
empty block costs nothing."""

from typing import Optional

import nki
import nki.language as nl
import torch
import torch.nn.functional as F
from nkilib.core.moe.moe_cte.moe_cte import (
    ActFnType,
    ExpertAffinityScaleMode,
    MoECTEImplementation,
    MoECTESpec,
    ShardOnBlockConfig,
    ShardOnIConfig,
    SkipMode,
)
from nkilib.core.moe.moe_cte.moe_cte import moe_cte as _nki_moe_cte
from torch import Tensor

from sglang_neuron import envs
from sglang_neuron.backend.stack import target_family, wrap_nki
from sglang_neuron.ops.moe.moe_blockwise import build_blockwise_mapping
from sglang_neuron.ops.moe.swiglu import swiglu_mlp
from sglang_neuron.utils.hardware_config import get_logical_nc_config
from sglang_neuron.utils.neuron_utils import can_run_kernel

# tokens per block: the kernel's tile of 128 rows
BLOCK_SIZE = 128
# (largest expert intermediate shard, most tokens per call) at which the kernel's SiLU form (no biases) matched the
# torch path on a NeuronCore (trn3, LNC2): I 768 up to 8192 tokens; I 1408 and 1536 up to 2048, past which (4096
# tokens) it returns NaN rows from about token 2600 on and garbage in others
VALIDATED_SHAPES = ((768, 8192), (1536, 2048))
# the hybrid kernel's block: a multiple of 256 tokens
HYBRID_BLOCK_SIZE = 256
# (largest expert intermediate, most tokens per call) the hybrid form is validated at on a trn2 NeuronCore (LNC2):
# Qwen3-30B-A3B's whole experts (I 768) at EP4, by GSM8K served end to end with 2048-token chunks
HYBRID_VALIDATED_SHAPES = ((768, 2048),)


@nki.jit(mode="trace")
def _moe_cte_bf16_hybrid(hidden_states, expert_affinities_masked, gate_up_proj_weight, down_proj_weight,
                         token_position_to_id, block_to_expert, conditions, num_static_block: int):
    return _nki_moe_cte(
        hidden_states=hidden_states,
        expert_affinities_masked=expert_affinities_masked,
        gate_up_proj_weight=gate_up_proj_weight,
        down_proj_weight=down_proj_weight,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        conditions=conditions,
        block_size=HYBRID_BLOCK_SIZE,
        spec=MoECTESpec(implementation=MoECTEImplementation.shard_on_i_hybrid,
                        shard_on_I=ShardOnIConfig(num_static_block=num_static_block)),
        activation_function=ActFnType.SiLU,
        skip_dma=SkipMode(skip_token=True, skip_weight=False),
        compute_dtype=nl.bfloat16,
        is_tensor_update_accumulating=True,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
    )


@nki.jit(mode="trace")
def _moe_cte_bf16(hidden_states, expert_affinities_masked, gate_up_proj_weight, down_proj_weight,
                  token_position_to_id, block_to_expert):
    return _nki_moe_cte(
        hidden_states=hidden_states,
        expert_affinities_masked=expert_affinities_masked,
        gate_up_proj_weight=gate_up_proj_weight,
        down_proj_weight=down_proj_weight,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        block_size=BLOCK_SIZE,
        spec=MoECTESpec(implementation=MoECTEImplementation.shard_on_block, shard_on_block=ShardOnBlockConfig()),
        activation_function=ActFnType.SiLU,
        # an empty slot (token -1) loads nothing; a padding block names the last local expert and holds only empty slots
        skip_dma=SkipMode(skip_token=True, skip_weight=True),
        compute_dtype=nl.bfloat16,
        is_tensor_update_accumulating=True,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
    )


def moe_experts_blockwise(hidden: Tensor, *, gate_up: Tensor, down: Tensor, affinities: Tensor, top_k: int,
                          padding_mask: Optional[Tensor] = None, ep_degree: int = 1) -> Tensor:
    """This rank's experts on hidden [T, H]: sum over the local experts of affinity * down(silu(gate) * up), [T, H].

    gate_up is [E_local, H, 2, I] (gate at 0, up at 1), down [E_local, I, H]; affinities [T, E_local] fp32 hold each
    token's weight for the local experts it was routed to, zero elsewhere; top_k is the experts per token, which bounds
    the blocks the routing can fill. padding_mask [T] bool marks the real tokens, and the mapping leaves the other
    rows out. A prefill passes it: the hidden state of a chunk's padding row changes from run to run, so its routing
    would change the block fill, the split of the blocks over the two cores and the order of the kernel's
    accumulating writes, and so the bf16 sums of the real tokens. Without it, every row is routed."""
    tokens = hidden.shape[0]
    mask = (torch.ones(tokens, dtype=torch.bool, device=hidden.device) if padding_mask is None else padding_mask)
    if ep_degree > 1 and _use_hybrid(tokens, gate_up.shape[-1]):
        return _hybrid(hidden, gate_up, down, affinities, top_k, ep_degree, mask)
    token_position_to_id, block_to_expert, _ = build_blockwise_mapping(
        affinities, num_local_experts=gate_up.shape[0], num_experts_per_token=top_k, block_size=BLOCK_SIZE,
        moe_group=None, tp_degree=1, padding_mask=mask)
    if not can_run_kernel(hidden):
        return _torch_moe_experts_blockwise(hidden, gate_up, down, affinities, token_position_to_id, block_to_expert)
    _check_validated(tokens, gate_up.shape[-1])
    out = wrap_nki(_moe_cte_bf16)[get_logical_nc_config()](
        hidden, affinities.to(hidden.dtype).reshape(-1, 1), gate_up, down, token_position_to_id,
        block_to_expert.reshape(-1, 1))
    # [T, 2, H]: the two cores' partial sums, which the kernel reduces into the first
    return out[:, 0]


def _use_hybrid(tokens: int, intermediate: int) -> bool:
    """Whether a prefill of ``tokens`` over whole experts of intermediate ``intermediate`` runs the hybrid kernel."""
    mode = envs.moe_prefill_impl()
    if mode == "block":
        return False
    fits = (tokens % HYBRID_BLOCK_SIZE == 0
            and any(intermediate <= i and tokens <= t for i, t in HYBRID_VALIDATED_SHAPES))
    trn2 = target_family() == "trn2"
    if mode == "hybrid":
        if not trn2:
            raise ValueError("SGLANG_NEURON_MOE_PREFILL=hybrid: nkilib's shard_on_i_hybrid runs on Trainium2 only")
        if not fits:
            raise ValueError(f"SGLANG_NEURON_MOE_PREFILL=hybrid: the hybrid moe_cte kernel is not validated for "
                             f"{tokens} tokens of an expert intermediate of {intermediate} (validated: "
                             f"{HYBRID_VALIDATED_SHAPES}, tokens a multiple of {HYBRID_BLOCK_SIZE})")
        return True
    # auto: on Trainium2, which the hybrid kernel targets (two cores sharding I), where it fits
    return fits and trn2


def _hybrid(hidden, gate_up, down, affinities, top_k, ep_degree, mask):
    tokens = hidden.shape[0]
    token_position_to_id, block_to_expert, conditions = build_blockwise_mapping(
        affinities, num_local_experts=gate_up.shape[0], num_experts_per_token=top_k, block_size=HYBRID_BLOCK_SIZE,
        moe_group=None, tp_degree=1, padding_mask=mask)
    if not can_run_kernel(hidden):
        return _torch_moe_experts_blockwise(hidden, gate_up, down, affinities, token_position_to_id, block_to_expert,
                                            block_size=HYBRID_BLOCK_SIZE)
    # the blocks the routing is expected to fill, an even count (the two cores alternate blocks); the dynamic loop
    # runs the rest of the filled ones
    expected = -(-tokens * top_k // (ep_degree * HYBRID_BLOCK_SIZE))
    num_static = min(max(2, 2 * (expected // 2)), conditions.shape[0])
    # the dynamic loop stops at the first zero condition, so the table ends with one
    conditions = torch.cat([conditions, conditions.new_zeros(2)])
    out = wrap_nki(_moe_cte_bf16_hybrid)[get_logical_nc_config()](
        hidden, affinities.to(hidden.dtype).reshape(-1, 1), gate_up, down, token_position_to_id,
        block_to_expert.reshape(-1, 1), conditions, num_static)
    # [T, H]: the shard-on-I form sums the two cores' intermediate halves itself
    return out


def _check_validated(tokens: int, intermediate: int) -> None:
    if not any(intermediate <= i and tokens <= t for i, t in VALIDATED_SHAPES):
        raise ValueError(f"the bf16 moe_cte kernel is not validated for {tokens} tokens of an expert intermediate "
                         f"shard of {intermediate} (validated: {VALIDATED_SHAPES}); launch with a smaller "
                         "--chunked-prefill-size")


@nki.jit(mode="trace")
def _moe_cte_bf16_swiglu(hidden_states, expert_affinities_masked, gate_up_proj_weight, down_proj_weight,
                         token_position_to_id, block_to_expert, gate_and_up_proj_bias, down_proj_bias,
                         gate_clamp_upper_limit, up_clamp_upper_limit, up_clamp_lower_limit):
    return _nki_moe_cte(
        hidden_states=hidden_states,
        expert_affinities_masked=expert_affinities_masked,
        gate_up_proj_weight=gate_up_proj_weight,
        down_proj_weight=down_proj_weight,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        block_size=BLOCK_SIZE,
        spec=MoECTESpec(implementation=MoECTEImplementation.shard_on_block, shard_on_block=ShardOnBlockConfig()),
        gate_and_up_proj_bias=gate_and_up_proj_bias,
        down_proj_bias=down_proj_bias,
        activation_function=ActFnType.Swish,
        skip_dma=SkipMode(skip_token=True, skip_weight=True),
        compute_dtype=nl.bfloat16,
        is_tensor_update_accumulating=True,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=None,
        up_clamp_upper_limit=up_clamp_upper_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
    )


def moe_experts_blockwise_swiglu(hidden: Tensor, *, gate_up: Tensor, gate_up_bias: Tensor, down: Tensor,
                                 down_bias: Tensor, affinities: Tensor, token_position_to_id: Tensor,
                                 block_to_expert: Tensor, gate_clamp_upper_limit: float, up_clamp_upper_limit: float,
                                 up_clamp_lower_limit: float) -> Tensor:
    """moe_experts_blockwise with GPT-OSS's expert MLP (ops/moe/swiglu.py): gate/up biases [E_local, 2, I], down biases
    [E_local, H], the clamps, and a caller-built mapping of BLOCK_SIZE-token blocks (build_blockwise_mapping), so
    masked tokens stay out."""
    limits = dict(gate_clamp_upper_limit=gate_clamp_upper_limit, up_clamp_upper_limit=up_clamp_upper_limit,
                  up_clamp_lower_limit=up_clamp_lower_limit)
    if not can_run_kernel(hidden):
        def mlp(x, expert):
            return swiglu_mlp(x, gate_up=gate_up[expert], gate_up_bias=gate_up_bias[expert], down=down[expert],
                              down_bias=down_bias[expert], **limits)

        return _torch_moe_experts_blockwise(hidden, gate_up, down, affinities, token_position_to_id, block_to_expert,
                                            mlp)
    out = wrap_nki(_moe_cte_bf16_swiglu)[get_logical_nc_config()](
        hidden, affinities.to(hidden.dtype).reshape(-1, 1), gate_up, down, token_position_to_id,
        block_to_expert.reshape(-1, 1), gate_up_bias, down_bias.unsqueeze(1), **limits)
    return out[:, 0]


def _silu_mlp(gate_up, down):
    def mlp(x, expert):
        w = gate_up.float()[expert]                                                 # [N, H, 2, I]
        gate = torch.einsum("nbh,nhi->nbi", x, w[:, :, 0])
        up = torch.einsum("nbh,nhi->nbi", x, w[:, :, 1])
        return torch.einsum("nbi,nih->nbh", F.silu(gate) * up, down.float()[expert])

    return mlp


def _torch_moe_experts_blockwise(hidden, gate_up, down, affinities, token_position_to_id, block_to_expert, mlp=None,
                                 block_size=BLOCK_SIZE):
    """Each block's rows through ``mlp(x [N, B, H], expert [N])`` (SiLU when None), scaled by the affinities and summed
    per token."""
    tokens, size = hidden.shape
    local = gate_up.shape[0]
    slots = token_position_to_id.to(torch.int64).reshape(-1, block_size)            # [N, B], -1 for an empty slot
    expert = block_to_expert.to(torch.int64).reshape(-1)                              # [N]
    # an empty slot reads and writes row T, a zero row dropped at the end
    rows = torch.where(slots < 0, torch.full_like(slots, tokens), slots)
    x = torch.cat([hidden.float(), hidden.new_zeros(1, size, dtype=torch.float32)])[rows]       # [N, B, H]
    y = (mlp or _silu_mlp(gate_up, down))(x, expert)
    weight = torch.cat([affinities.float(), affinities.new_zeros(1, local, dtype=torch.float32)])
    scale = weight[rows, expert.unsqueeze(1)]                                       # [N, B]
    out = torch.zeros(tokens + 1, size, dtype=torch.float32, device=hidden.device)
    out = out.index_add(0, rows.reshape(-1), (y * scale.unsqueeze(-1)).reshape(-1, size))
    return out[:tokens].to(hidden.dtype)
