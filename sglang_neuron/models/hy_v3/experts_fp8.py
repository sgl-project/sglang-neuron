# SPDX-License-Identifier: Apache-2.0
"""Hy3's routed experts in the checkpoint's per-tensor FP8.

Each local expert keeps its FP8 gate, up and down weights with their ``weight_scale`` and the static ``input_scale`` of
its activations. Trn2's FP8 kernels use the 240-capped e4m3, so every weight is halved as it loads (an exact exponent
shift for a normal value) and its weight scale doubled; the activation scales are doubled too, so calibrated
activations do not clip at 240 (as the vllm-neuron port does). The tensors stay ``float8_e4m3fn``, which the graphs
compile as Trn2's e4m3 (launch.py).

- decode: nkilib's moe_tkg in its static-FP8 mode, every local expert (``is_all_expert``) over up to 128 tokens a call;
- prefill: nkilib's hybrid I-sharded moe_cte takes them as they are (experts_cte.py)."""

import nki
import torch
from nkilib.core.moe.moe_tkg.moe_tkg import moe_tkg
from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode
from torch import nn

from sglang_neuron.backend.stack import wrap_nki
from sglang_neuron.models.qwen3_moe.moe import _down_loader, _gate_up_loader
from sglang_neuron.utils.hardware_config import get_logical_nc_config
from sglang_neuron.utils.weight_loader import SafetensorsWeightLoader, set_weight_loader

FP8 = torch.float8_e4m3fn
# SGLANG_NEURON_HY3_EXPERTS=bf16 dequantizes the routed experts to bf16 instead (an accuracy and speed reference)
EXPERTS_ENV = "SGLANG_NEURON_HY3_EXPERTS"


def enabled() -> bool:
    import os

    value = os.environ.get(EXPERTS_ENV, "fp8")
    if value not in ("fp8", "bf16"):
        raise ValueError(f"{EXPERTS_ENV}={value!r}: set fp8 (the checkpoint's) or bf16")
    return value == "fp8"
MAX_TOKENS = 128


def _halved(loader: SafetensorsWeightLoader) -> SafetensorsWeightLoader:
    return SafetensorsWeightLoader(transform=lambda slices, rank: (loader.load(slices, rank).float() * 0.5).to(FP8))


def _scalars(slices) -> torch.Tensor:
    return torch.stack([s[...].float().reshape(()) for s in slices])


def _expert_tp_rank(moe) -> int:
    """The rank of this rank among those an expert's intermediate dimension splits over (SparseMoE's layout)."""
    from sglang_neuron.parallel import groups
    from sglang_neuron.parallel.state import STATE

    tp, ep = moe.attn_tp_group.world_size, STATE.ep_degree
    if ep == 1:
        return moe.attn_tp_group.rank_in_group
    return groups.get_neuron_ep_tp_group().rank_in_group if tp // ep > 1 else 0


def make_fp8(moe) -> None:
    """Give ``moe`` (a HyV3MoE) FP8 expert weights and their scales in place of its bf16 ones."""
    e, (h, _, i) = moe.num_local_experts, moe.gate_up_weight.shape[1:]
    expert_tp_rank = _expert_tp_rank(moe)
    moe.gate_up_weight = nn.Parameter(torch.empty(e, h, 2, i, dtype=FP8), requires_grad=False)
    moe.down_weight = nn.Parameter(torch.empty(e, i, h, dtype=FP8), requires_grad=False)
    set_weight_loader(moe.gate_up_weight, _halved(_gate_up_loader(e, i, expert_tp_rank)))
    set_weight_loader(moe.down_weight, _halved(_down_loader(i, expert_tp_rank)))
    # [E_l, 2, 1] gate and up weight scales, [E_l, 1] down's, [E_l, 1] each input scale; all doubled
    moe.gate_up_scale = nn.Parameter(torch.empty(e, 2, 1, dtype=torch.float32), requires_grad=False)
    moe.down_scale = nn.Parameter(torch.empty(e, 1, dtype=torch.float32), requires_grad=False)
    moe.gate_up_in_scale = nn.Parameter(torch.empty(e, 1, dtype=torch.float32), requires_grad=False)
    moe.down_in_scale = nn.Parameter(torch.empty(e, 1, dtype=torch.float32), requires_grad=False)
    set_weight_loader(moe.gate_up_scale, SafetensorsWeightLoader(transform=lambda slices, rank: (
        torch.stack([_scalars(slices[:e]), _scalars(slices[e:])], dim=1).reshape(e, 2, 1) * 2)))
    set_weight_loader(moe.down_scale, SafetensorsWeightLoader(transform=lambda slices, rank: (
        _scalars(slices).reshape(e, 1) * 2)))
    # gate and up read the same input: one scale for both, the larger
    set_weight_loader(moe.gate_up_in_scale, SafetensorsWeightLoader(transform=lambda slices, rank: (
        torch.maximum(_scalars(slices[:e]), _scalars(slices[e:])).reshape(e, 1) * 2)))
    set_weight_loader(moe.down_in_scale, SafetensorsWeightLoader(transform=lambda slices, rank: (
        _scalars(slices).reshape(e, 1) * 2)))


def scale_mappings(experts: str, local, dst: str) -> dict:
    def keys(proj: str, kind: str) -> list:
        return [f"{experts}.{e}.{proj}.{kind}" for e in local]

    return {
        f"{dst}.gate_up_scale": keys("gate_proj", "weight_scale") + keys("up_proj", "weight_scale"),
        f"{dst}.down_scale": keys("down_proj", "weight_scale"),
        f"{dst}.gate_up_in_scale": keys("gate_proj", "input_scale") + keys("up_proj", "input_scale"),
        f"{dst}.down_in_scale": keys("down_proj", "input_scale"),
    }


@nki.jit
def _moe_tkg_fp8(hidden, gate_up, down, affinities, index, rank_id, gate_up_scale, down_scale, gate_up_in_scale,
                 down_in_scale):
    return moe_tkg(
        hidden, gate_up, down, affinities, index,
        is_all_expert=True,
        rank_id=rank_id,
        expert_gate_up_weights_scale=gate_up_scale,
        expert_down_weights_scale=down_scale,
        expert_gate_up_input_scale=gate_up_in_scale,
        expert_down_input_scale=down_in_scale,
        activation_fn=ActFnType.SiLU,
        mask_unselected_experts=True,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
    )


def decode_experts(moe, x: torch.Tensor, affinities: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """This rank's experts on the decode rows x [T, H], as ops/moe/moe_tkg_bf16.moe_experts over FP8 weights."""
    kernel = wrap_nki(_moe_tkg_fp8)[get_logical_nc_config()]
    rank_id = moe._ep_rank_id()
    tiles = [kernel(x[s:s + MAX_TOKENS], moe.gate_up_weight, moe.down_weight, affinities[s:s + MAX_TOKENS],
                    index[s:s + MAX_TOKENS], rank_id, moe.gate_up_scale, moe.down_scale, moe.gate_up_in_scale,
                    moe.down_in_scale)
             for s in range(0, x.shape[0], MAX_TOKENS)]
    return tiles[0] if len(tiles) == 1 else torch.cat(tiles, dim=0)
