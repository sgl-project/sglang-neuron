# SPDX-License-Identifier: Apache-2.0
"""The decode MoE of one layer in one kernel: nkilib's moe_block_tkg runs the post-attention RMSNorm, the router's
softmax top-k and the experts, MXFP4 or BF16."""

import nki
import nki.language as nl
import torch
from nkilib.core.moe_block.moe_block_tkg import moe_block_tkg as _nki_moe_block_tkg
from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode, RouterActFnType
from torch import Tensor

from sglang_neuron.backend.stack import wrap_nki
from sglang_neuron.ops.moe.swiglu import swiglu_mlp
from sglang_neuron.utils.hardware_config import get_logical_nc_config
from sglang_neuron.utils.neuron_utils import can_run_kernel


@nki.jit
def _moe_block_tkg(inp, gamma, router_weights, router_bias, expert_gate_up_weights, expert_gate_up_weights_scale,
                   expert_gate_up_bias, expert_down_weights, expert_down_weights_scale, expert_down_bias, eps, top_k,
                   gate_clamp_upper_limit, up_clamp_upper_limit, up_clamp_lower_limit, hidden_actual, is_all_expert,
                   rank_id):
    # the uint16 MXFP4 tiles, as the kernel's float4 x4 type
    return _nki_moe_block_tkg(
        inp=inp,
        gamma=gamma,
        router_weights=router_weights,
        expert_gate_up_weights=expert_gate_up_weights.view(nl.float4_e2m1fn_x4),
        expert_down_weights=expert_down_weights.view(nl.float4_e2m1fn_x4),
        expert_gate_up_weights_scale=expert_gate_up_weights_scale,
        expert_down_weights_scale=expert_down_weights_scale,
        router_bias=router_bias,
        expert_gate_up_bias=expert_gate_up_bias,
        expert_down_bias=expert_down_bias,
        eps=eps,
        top_k=top_k,
        router_act_fn=RouterActFnType.SOFTMAX,
        router_pre_norm=False,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
        hidden_act_fn=ActFnType.Swish,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        router_mm_dtype=nl.bfloat16,
        hidden_actual=hidden_actual,
        is_all_expert=is_all_expert,
        rank_id=rank_id,
    )


@nki.jit
def _moe_block_tkg_bf16(inp, gamma, router_weights, router_bias, expert_gate_up_weights, expert_gate_up_bias,
                        expert_down_weights, expert_down_bias, eps, top_k, gate_clamp_upper_limit, up_clamp_upper_limit,
                        up_clamp_lower_limit, hidden_actual, is_all_expert, rank_id):
    # _moe_block_tkg for BF16 experts: no scales
    return _nki_moe_block_tkg(
        inp=inp,
        gamma=gamma,
        router_weights=router_weights,
        expert_gate_up_weights=expert_gate_up_weights,
        expert_down_weights=expert_down_weights,
        router_bias=router_bias,
        expert_gate_up_bias=expert_gate_up_bias,
        expert_down_bias=expert_down_bias,
        eps=eps,
        top_k=top_k,
        router_act_fn=RouterActFnType.SOFTMAX,
        router_pre_norm=False,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
        hidden_act_fn=ActFnType.Swish,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        router_mm_dtype=nl.bfloat16,
        hidden_actual=hidden_actual,
        is_all_expert=is_all_expert,
        rank_id=rank_id,
    )


def moe_block_tkg(inp: Tensor, *, gamma: Tensor, router_weights: Tensor, router_bias: Tensor,
                  expert_gate_up_weights: Tensor, expert_gate_up_bias: Tensor, expert_down_weights: Tensor,
                  expert_down_bias: Tensor, eps: float, top_k: int, gate_clamp_upper_limit: float,
                  up_clamp_upper_limit: float, up_clamp_lower_limit: float, hidden_actual: int, is_all_expert: bool,
                  rank_id, expert_gate_up_weights_scale: Tensor | None = None,
                  expert_down_weights_scale: Tensor | None = None) -> Tensor:
    """The MoE output of inp [1, T, H] over this rank's experts, before the reduction across the expert ranks:
    [T, H].

    gamma [1, H] fp32 and router_weights [H, E] are in the shuffled hidden order. With scales the experts are MXFP4
    tiles: gate/up [E_local, 128, 2, H // 512, I] and down [E_local, I_tile, I_tiles, H] uint16 with uint8 scales.
    Without, BF16: gate/up [E_local, H, 2, I] (gate at 0, up at 1) with biases [E_local, 2, I], down [E_local, I, H]
    with biases [E_local, H]. ``is_all_expert`` runs every local expert, which needs ``rank_id`` [1, 1] int32, the EP
    rank; otherwise the kernel loads only the routed experts. On a CPU the BF16 experts run in torch
    (ops/moe/swiglu.py); MXFP4 has no CPU path, since the kernel's MX arithmetic has no torch counterpart here.
    """
    limits = dict(gate_clamp_upper_limit=gate_clamp_upper_limit, up_clamp_upper_limit=up_clamp_upper_limit,
                  up_clamp_lower_limit=up_clamp_lower_limit)
    if not can_run_kernel(inp):
        if expert_gate_up_weights_scale is not None:
            raise RuntimeError("the MXFP4 decode MoE runs on a NeuronCore only")
        x = inp[0].float()
        variance = x.pow(2).sum(-1, keepdim=True) / hidden_actual
        normed = (x * torch.rsqrt(variance + eps) * gamma.float()).to(inp.dtype)
        logits = (normed @ router_weights).float() + router_bias.float()
        top, routed = torch.topk(logits, top_k, dim=-1)
        weights = torch.zeros_like(logits).scatter(1, routed, torch.softmax(top, dim=-1))
        local = expert_gate_up_weights.shape[0]
        local_experts = torch.arange(local, device=inp.device)
        if rank_id is not None:
            local_experts = local_experts + rank_id.reshape(1).to(torch.int64) * local
        out = swiglu_mlp(normed.unsqueeze(0).expand(local, -1, -1), gate_up=expert_gate_up_weights,
                         gate_up_bias=expert_gate_up_bias, down=expert_down_weights, down_bias=expert_down_bias,
                         **limits)                                                          # [E_local, T, H]
        return torch.einsum("te,eth->th", weights.index_select(1, local_experts), out).to(inp.dtype)
    kernel, scales = _moe_block_tkg_bf16, {}
    if expert_gate_up_weights_scale is not None:
        kernel, scales = _moe_block_tkg, dict(expert_gate_up_weights_scale=expert_gate_up_weights_scale,
                                              expert_down_weights_scale=expert_down_weights_scale)
    result = wrap_nki(kernel)[get_logical_nc_config()](
        inp=inp, gamma=gamma, router_weights=router_weights, router_bias=router_bias,
        expert_gate_up_weights=expert_gate_up_weights, expert_gate_up_bias=expert_gate_up_bias,
        expert_down_weights=expert_down_weights, expert_down_bias=expert_down_bias, eps=eps, top_k=top_k,
        hidden_actual=hidden_actual, is_all_expert=is_all_expert, rank_id=rank_id, **scales, **limits,
    )
    # the router logits come back after the output: the kernel's validation (NCC_INKI016) refuses to skip their store,
    # a compiler limitation (NCC_IGCA090)
    return result[0]
