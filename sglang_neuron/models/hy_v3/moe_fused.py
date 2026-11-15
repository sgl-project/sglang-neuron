# SPDX-License-Identifier: Apache-2.0
"""Hy3's decode MoE block in one NKI kernel (kernels/moe_tkg_fused.py, from the vllm-neuron Hy3 contrib): the
post-attention RMSNorm, the router, this rank's FP8 routed experts and a shard of the FP8 shared expert, with static
FP8 activations (W8A8) and double-row matmuls; the caller all-reduces its partial output. On by default for the decode
of FP8 experts; ``SGLANG_NEURON_HY3_DECODE_MOE_FUSED=0`` keeps the separate norm, router, moe_tkg and shared MLP.

The shared expert's decode shard: the ranks are in groups of ``SHARED_SHARDS``, each group holding the whole shared
expert, a rank its ``tp_rank % SHARED_SHARDS``-th slice of the intermediate dimension; its output is scaled by
``SHARED_SHARDS / tp`` so the all-reduce sums one shared expert. The kernel reads the slice in FP8 (halved, its scales
doubled, as experts_fp8.py), next to the prefill's bf16 shared expert (mlp.ShardedMLP)."""

import os

import torch
import torch.nn.functional as F
from torch import nn

from sglang_neuron.backend.stack import wrap_nki
from sglang_neuron.models.hy_v3.experts_fp8 import FP8, _halved
from sglang_neuron.models.hy_v3.kernels.moe_tkg_fused import moe_tkg_fused
from sglang_neuron.utils.hardware_config import get_logical_nc_config
from sglang_neuron.utils.weight_loader import SafetensorsWeightLoader, set_weight_loader

ENV = "SGLANG_NEURON_HY3_DECODE_MOE_FUSED"
SHARED_SHARDS = 8
MAX_TOKENS = 128
PAD = 128


def enabled() -> bool:
    value = os.environ.get(ENV, "1")
    if value not in ("0", "1"):
        raise ValueError(f"{ENV}={value!r}: set 0 or 1")
    return value == "1"


def applies(moe) -> bool:
    """Whether ``moe`` (a HyV3MoE) can decode through the kernel: FP8 experts whose intermediate shard is whole 512s, on
    a rank count the shared expert's shards divide."""
    tp = moe.attn_tp_group.world_size
    return (moe.fp8_experts and moe.down_weight.shape[1] % 512 == 0 and tp > SHARED_SHARDS
            and tp % SHARED_SHARDS == 0 and moe.shared_intermediate % SHARED_SHARDS == 0)


def make(moe) -> None:
    """Give ``moe`` the kernel's own tensors: the router transposed, the shared expert's FP8 decode shard and scales.
    An aggregated server's prefill graph holds them too, unused: its graphs share one device copy of the weights."""
    hidden, experts = moe.router_weight.shape[1], moe.router_weight.shape[0]
    shard = moe.shared_intermediate // SHARED_SHARDS
    # the kernel splits a shard's columns over the two cores in whole 128s: zero columns pad it (as the vllm-neuron
    # port's shared decode MLP), which add nothing (silu(0) * 0)
    padded = -(-shard // (2 * PAD)) * 2 * PAD
    moe.router_wT = nn.Parameter(torch.empty(hidden, experts, dtype=torch.bfloat16), requires_grad=False)
    set_weight_loader(moe.router_wT, SafetensorsWeightLoader(
        transform=lambda slices, rank: slices[0][...].to(torch.bfloat16).t().contiguous()))

    def rows(slices, rank):  # gate / up [I, H]: this rank's shard rows, as [H, padded]
        start = (rank % SHARED_SHARDS) * shard
        return F.pad(slices[0][start:start + shard].t(), (0, padded - shard)).contiguous()

    def columns(slices, rank):  # down [H, I]: this rank's shard columns, as [padded, H]
        start = (rank % SHARED_SHARDS) * shard
        return F.pad(slices[0][:, start:start + shard].t(), (0, 0, 0, padded - shard)).contiguous()

    for name, transform, shape in (("sh_gate", rows, (hidden, padded)), ("sh_up", rows, (hidden, padded)),
                                   ("sh_down", columns, (padded, hidden))):
        param = nn.Parameter(torch.empty(*shape, dtype=FP8), requires_grad=False)
        set_weight_loader(param, _halved(SafetensorsWeightLoader(transform=transform)))
        setattr(moe, name, param)
    # gate, up and down weight scales, then the gate / up input scale (the larger) and the down one; all doubled
    moe.sh_scales = nn.Parameter(torch.empty(5, dtype=torch.float32), requires_grad=False)
    set_weight_loader(moe.sh_scales, SafetensorsWeightLoader(transform=lambda slices, rank: torch.stack([
        slices[0][...].float().reshape(()), slices[1][...].float().reshape(()), slices[2][...].float().reshape(()),
        torch.maximum(slices[3][...].float().reshape(()), slices[4][...].float().reshape(())),
        slices[5][...].float().reshape(())]) * 2))


def mappings(moe, src: str, dst: str) -> dict:
    shared = f"{src}.mlp.shared_mlp"
    return {
        f"{dst}.router_wT": f"{src}.mlp.router.gate.weight",
        f"{dst}.sh_gate": [f"{shared}.gate_proj.weight"],
        f"{dst}.sh_up": [f"{shared}.up_proj.weight"],
        f"{dst}.sh_down": [f"{shared}.down_proj.weight"],
        f"{dst}.sh_scales": [f"{shared}.gate_proj.weight_scale", f"{shared}.up_proj.weight_scale",
                             f"{shared}.down_proj.weight_scale", f"{shared}.gate_proj.input_scale",
                             f"{shared}.up_proj.input_scale", f"{shared}.down_proj.input_scale"],
    }


def decode(moe, hidden: torch.Tensor, ln_w: torch.Tensor, eps: float) -> torch.Tensor:
    """This rank's partial MoE block output [T, H] for the residual stream ``hidden`` [T, H] (before the norm)."""
    experts = moe.num_local_experts
    s_in, s_din = moe.gate_up_in_scale[:, 0], moe.down_in_scale[:, 0]
    gu, dw = moe.gate_up_scale[:, :, 0], moe.down_scale[:, 0]
    routed = torch.stack([gu[:, 0] * s_in, gu[:, 1] * s_in / s_din, dw * s_din, 1 / s_in], dim=1)
    g, u, d, sh_in, sh_din = moe.sh_scales.unbind(0)
    weight = SHARED_SHARDS / moe.attn_tp_group.world_size
    shared = torch.stack([g * sh_in, u * sh_in / sh_din, d * sh_din * weight, 1 / sh_in]).reshape(1, 4)
    scales = torch.cat([routed, shared], dim=0).reshape(1, 4 * (experts + 1)).to(torch.float32)
    ebase = (moe._ep_rank_id().to(torch.float32) * experts).reshape(1, 1)
    kernel = wrap_nki(moe_tkg_fused)[get_logical_nc_config()]
    tiles = [kernel(hidden[s:s + MAX_TOKENS].to(torch.bfloat16), ln_w.view(1, -1), moe.router_wT,
                    moe.expert_bias.view(1, -1), ebase, moe.gate_up_weight, moe.down_weight, moe.sh_gate, moe.sh_up,
                    moe.sh_down, scales, eps=eps, route_scale=float(moe.scaling), top_k=moe.top_k)
             for s in range(0, hidden.shape[0], MAX_TOKENS)]
    return tiles[0] if len(tiles) == 1 else torch.cat(tiles, dim=0)
