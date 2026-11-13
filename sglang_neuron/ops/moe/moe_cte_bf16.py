# SPDX-License-Identifier: Apache-2.0
"""The bf16 prefill experts of a sparse MoE on this rank: nkilib's moe_cte kernel (shard_on_block) runs each local
expert's MLP (SiLU, or GPT-OSS's clamped SwiGLU with biases) on blocks of the tokens routed to it, the token-to-block
mapping from build_blockwise_mapping, so an expert's weights are read once per block of its own tokens. The kernel has
no dynamic loop: it runs every block the routing could fill (every token's top-k on this rank), the ones no token
filled included, each at a full block's cost. On a CPU the torch path computes the same blocks."""

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
    SkipMode,
)
from nkilib.core.moe.moe_cte.moe_cte import moe_cte as _nki_moe_cte
from torch import Tensor

from sglang_neuron.backend.stack import wrap_nki
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
# the same on trn2 (LNC2): I 768 up to 4096 tokens (Qwen3-30B-A3B at TP8 EP8 and Qwen3-235B-A22B at TP32 EP16 serve
# right). I 1536 is wrong at every prefill tile: Qwen3-235B-A22B at TP32 EP32 answers garbage for one prompt of 800
# tokens in the 1024 tile and of 512 in the 512 tile, and every later request on that server is garbage too.
VALIDATED_SHAPES_TRN2 = ((768, 4096),)


def _moe_cte_bf16_kernel(skip_weight: bool):
    """The kernel with or without skipping a block's weight load when its expert's are loaded already.

    The skip (``SkipMode.skip_weight``) is wrong when one expert's blocks land on both cores: with a hot expert (512
    tokens of 512 routed to one of three local experts, on trn2 at LNC2) one core's blocks come out as garbage, and as
    NaN at 1024 tokens and more. Every block then loads its expert's weights instead."""

    @nki.jit(mode="trace")
    def kernel(hidden_states, expert_affinities_masked, gate_up_proj_weight, down_proj_weight, token_position_to_id,
               block_to_expert):
        return _moe_cte_bf16_body(hidden_states, expert_affinities_masked, gate_up_proj_weight, down_proj_weight,
                                  token_position_to_id, block_to_expert, skip_weight)

    return kernel


def _moe_cte_bf16_body(hidden_states, expert_affinities_masked, gate_up_proj_weight, down_proj_weight,
                       token_position_to_id, block_to_expert, skip_weight):
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
        skip_dma=SkipMode(skip_token=True, skip_weight=skip_weight),
        compute_dtype=nl.bfloat16,
        is_tensor_update_accumulating=True,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
    )


_moe_cte_bf16 = _moe_cte_bf16_kernel(skip_weight=True)
_moe_cte_bf16_reload = _moe_cte_bf16_kernel(skip_weight=False)


def moe_experts_blockwise(hidden: Tensor, *, gate_up: Tensor, down: Tensor, affinities: Tensor, top_k: int,
                          padding_mask: Optional[Tensor] = None, reload_weights: bool = False) -> Tensor:
    """This rank's experts on hidden [T, H]: sum over the local experts of affinity * down(silu(gate) * up), [T, H].

    gate_up is [E_local, H, 2, I] (gate at 0, up at 1), down [E_local, I, H]; affinities [T, E_local] fp32 hold each
    token's weight for the local experts it was routed to, zero elsewhere; top_k is the experts per token, which bounds
    the blocks the routing can fill. padding_mask [T] bool marks the real tokens, and the mapping leaves the other
    rows out. A prefill passes it: the hidden state of a chunk's padding row changes from run to run, so its routing
    would change the block fill, the split of the blocks over the two cores and the order of the kernel's
    accumulating writes, and so the bf16 sums of the real tokens. Without it, every row is routed. ``reload_weights``
    loads every block's expert weights (_moe_cte_bf16_kernel): a model whose routing can send many of a chunk's tokens to
    one of few local experts needs it."""
    tokens = hidden.shape[0]
    token_position_to_id, block_to_expert, _ = build_blockwise_mapping(
        affinities, num_local_experts=gate_up.shape[0], num_experts_per_token=top_k, block_size=BLOCK_SIZE,
        moe_group=None, tp_degree=1,
        padding_mask=(torch.ones(tokens, dtype=torch.bool, device=hidden.device) if padding_mask is None
                      else padding_mask))
    if not can_run_kernel(hidden):
        return _torch_moe_experts_blockwise(hidden, gate_up, down, affinities, token_position_to_id, block_to_expert)
    _check_validated(tokens, gate_up.shape[-1])
    out = wrap_nki(_moe_cte_bf16_reload if reload_weights else _moe_cte_bf16)[get_logical_nc_config()](
        hidden, affinities.to(hidden.dtype).reshape(-1, 1), gate_up, down, token_position_to_id,
        block_to_expert.reshape(-1, 1))
    # [T, 2, H]: the two cores' partial sums, which the kernel reduces into the first
    return out[:, 0]


def _resolve_validated_shapes():
    """The shapes validated on this launch's target: trn2's where it reports trn2, else trn3's (a CPU run included,
    which takes the torch path)."""
    from sglang_neuron.backend.stack import get_platform_target

    try:
        target = get_platform_target()
    except (RuntimeError, ImportError):  # no runtime to ask: a CPU run
        return VALIDATED_SHAPES
    return VALIDATED_SHAPES_TRN2 if target.startswith("trn2") else VALIDATED_SHAPES


# resolved once at import, as dtype_utils.FP8_CLAMP_MAX: the check runs while the prefill graph traces, where the
# runtime's target query cannot be called
_VALIDATED_ON_TARGET = _resolve_validated_shapes()


def _check_validated(tokens: int, intermediate: int) -> None:
    shapes = _VALIDATED_ON_TARGET
    if not any(intermediate <= i and tokens <= t for i, t in shapes):
        smaller_shard = ("launch with more ranks per expert (a smaller --ep-size or a larger --tp-size) for a smaller "
                         "shard")
        fix = (smaller_shard if not any(intermediate <= i for i, _ in shapes)
               else f"launch with a smaller --chunked-prefill-size, or {smaller_shard}")
        raise ValueError(f"the bf16 moe_cte kernel is not validated for {tokens} tokens of an expert intermediate "
                         f"shard of {intermediate} (validated on this target: {shapes}); {fix}")


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


def _torch_moe_experts_blockwise(hidden, gate_up, down, affinities, token_position_to_id, block_to_expert, mlp=None):
    """Each block's rows through ``mlp(x [N, B, H], expert [N])`` (SiLU when None), scaled by the affinities and summed
    per token."""
    tokens, size = hidden.shape
    local = gate_up.shape[0]
    slots = token_position_to_id.to(torch.int64).reshape(-1, BLOCK_SIZE)            # [N, B], -1 for an empty slot
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
