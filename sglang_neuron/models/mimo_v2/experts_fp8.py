# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2's routed experts in Trn2's FP8 (e4m3, max 240), converted from the checkpoint without loss.

Trn2 has no MX matmul, so V2.6's MXFP4 experts cannot run as stored. Each expert matrix is dequantized
(weights.dequant_mxfp4) and divided by a **power-of-two scale per output row**, the smallest that brings the row's
largest value to 240 or below. A power of two only shifts exponents, and an e2m1 value has two significant bits, so
every quotient is an e4m3 number: the conversion is exact (tools/mimo_v2/check_fp8_experts.py checks the whole
checkpoint; ``rowmax / 240``, a scale that is not a power of two, leaves about 7% of the values inexact). V2.5's
128 x 128 FP8-block experts take the same path; for them the per-row requantization is a rounding.

The experts then hold half the bf16 bytes (4.7 instead of 9.5 GB a rank at EP64) and run weight-only FP8 (bf16
activations, FP8 weights, the per-row scale applied after each matmul):

- decode: nkilib's moe_tkg in its ROW mode (gate/up scales [E, 2, I], down scales [E, H]), every local expert;
- prefill: Hy3's hybrid I-sharded moe_cte copy that feeds FP8 tiles to the matmul (models/hy_v3/experts_cte.py,
  kernels/moe_cte_fp8w.py), which takes per-channel scales.

The tensors are ``float8_e4m3fn`` holding values of at most 240, which the graphs compile as Trn2's e4m3 (the same
bytes; launch.py's UNSAFE_FP8FNCAST and the host's compiler option). ``SGLANG_NEURON_MIMO_EXPERTS=bf16`` keeps the
bf16 experts, the accuracy reference."""

import os
import threading

import nki
import torch
from nkilib.core.moe.moe_tkg.moe_tkg import moe_tkg
from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode
from torch import nn

from sglang_neuron.backend.stack import wrap_nki
from sglang_neuron.models.mimo_v2.weights import DequantSlice
from sglang_neuron.utils.hardware_config import get_logical_nc_config
from sglang_neuron.utils.weight_loader import SafetensorsWeightLoader, set_weight_loader

FP8 = torch.float8_e4m3fn
FP8_MAX = 240.0
EXPERTS_ENV = "SGLANG_NEURON_MIMO_EXPERTS"
# moe_tkg's tokens a call
MAX_TOKENS = 128


def enabled(environ=os.environ) -> bool:
    value = environ.get(EXPERTS_ENV, "fp8").strip()
    if value not in ("fp8", "bf16"):
        raise ValueError(f"{EXPERTS_ENV}={value!r}: set fp8 (the experts in Trn2's FP8, converted without loss) or "
                         "bf16")
    return value == "fp8"


def requant_rows(w: torch.Tensor) -> tuple:
    """fp32 [out, in] -> (FP8 [out, in], fp32 [out]) with a power-of-two scale per output row (module docstring)."""
    amax = w.abs().amax(dim=1).clamp(min=1e-30)
    scale = torch.exp2(torch.ceil(torch.log2(amax / FP8_MAX)))
    q = (w / scale[:, None]).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    return q, scale


def _fp8_t(q: torch.Tensor) -> torch.Tensor:
    """An FP8 matrix transposed through its bytes (the CPU has no FP8 transpose kernels)."""
    return q.view(torch.uint8).t().contiguous().view(FP8)


class _Memo:
    """One requantization per (layer, rank, part) shared by a weight and its scale, whose loaders may run on the
    checkpoint loader's threads at once; each result is handed out once, so a reload recomputes."""

    def __init__(self):
        self._lock = threading.Lock()
        self._slots = {}

    def take(self, key, part: str, compute):
        with self._lock:
            slot = self._slots.setdefault(key, {"lock": threading.Lock(), "parts": None})
        with slot["lock"]:
            if slot["parts"] is None:
                slot["parts"] = compute()
            value = slot["parts"].pop(part)
            if not slot["parts"]:
                with self._lock:
                    self._slots.pop(key, None)
        return value


def _views(slices: list) -> list:
    """bf16 views of [weights..., scales...] (weights.with_scales), or the slices of a bf16 checkpoint."""
    if not slices or isinstance(slices[0], DequantSlice):
        return slices
    n = len(slices) // 2
    return [DequantSlice(w, s) for w, s in zip(slices[:n], slices[n:])]


def _gate_up(slices: list, local: int, rows: slice) -> dict:
    """[E, H, 2, I] FP8 and [E, 2, I] scales of the local experts' gate and up (``rows`` this rank's I rows)."""
    views = _views(slices)
    weights, scales = [], []
    for g, u in zip(views[:local], views[local:]):
        pair_w, pair_s = [], []
        for view in (g, u):
            q, s = requant_rows(view[rows].float())
            pair_w.append(_fp8_t(q))
            pair_s.append(s)
        weights.append(torch.stack([w.view(torch.uint8) for w in pair_w], dim=1).view(FP8))
        scales.append(torch.stack(pair_s))
    return {"weight": torch.stack([w.view(torch.uint8) for w in weights]).view(FP8), "scale": torch.stack(scales)}


def _down(slices: list, cols: slice) -> dict:
    """[E, I, H] FP8 and [E, H] scales of the local experts' down (``cols`` this rank's I columns)."""
    weights, scales = [], []
    for view in _views(slices):
        q, s = requant_rows(view[:, cols].float())
        weights.append(_fp8_t(q).view(torch.uint8))
        scales.append(s)
    return {"weight": torch.stack(weights).view(FP8), "scale": torch.stack(scales)}


def make_fp8(moe, layer_key: str) -> None:
    """Give ``moe`` (moe.MiMoV2MoE) FP8 expert weights and their per-row scales in place of its bf16 ones; each
    weight's mapping is its scale's too (moe.MiMoV2MoE.weight_mappings)."""
    e, (h, _, i) = moe.num_local_experts, moe.gate_up_weight.shape[1:]
    shard = slice(moe.expert_tp_rank * i, (moe.expert_tp_rank + 1) * i)
    memo = _Memo()
    moe.gate_up_weight = nn.Parameter(torch.empty(e, h, 2, i, dtype=FP8), requires_grad=False)
    moe.down_weight = nn.Parameter(torch.empty(e, i, h, dtype=FP8), requires_grad=False)
    moe.gate_up_scale = nn.Parameter(torch.empty(e, 2, i, dtype=torch.float32), requires_grad=False)
    moe.down_scale = nn.Parameter(torch.empty(e, h, dtype=torch.float32), requires_grad=False)

    def loader(kind: str, part: str):
        compute = (lambda slices: _gate_up(slices, e, shard)) if kind == "gate_up" else (
            lambda slices: _down(slices, shard))
        return SafetensorsWeightLoader(transform=lambda slices, rank: memo.take(
            (layer_key, kind, rank), part, lambda: compute(slices)))

    set_weight_loader(moe.gate_up_weight, loader("gate_up", "weight"))
    set_weight_loader(moe.gate_up_scale, loader("gate_up", "scale"))
    set_weight_loader(moe.down_weight, loader("down", "weight"))
    set_weight_loader(moe.down_scale, loader("down", "scale"))


@nki.jit
def _moe_tkg_row(hidden, gate_up, down, affinities, index, rank_id, gate_up_scale, down_scale):
    return moe_tkg(
        hidden, gate_up, down, affinities, index,
        is_all_expert=True,
        rank_id=rank_id,
        expert_gate_up_weights_scale=gate_up_scale,
        expert_down_weights_scale=down_scale,
        activation_fn=ActFnType.SiLU,
        mask_unselected_experts=True,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
    )


def decode_experts(moe, x: torch.Tensor, affinities: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """This rank's experts on the decode rows x [T, H] (ops/moe/moe_tkg_bf16.moe_experts over the FP8 weights)."""
    kernel = wrap_nki(_moe_tkg_row)[get_logical_nc_config()]
    rank_id = moe._ep_rank_id()
    tiles = [kernel(x[s:s + MAX_TOKENS], moe.gate_up_weight, moe.down_weight, affinities[s:s + MAX_TOKENS],
                    index[s:s + MAX_TOKENS], rank_id, moe.gate_up_scale, moe.down_scale)
             for s in range(0, x.shape[0], MAX_TOKENS)]
    return tiles[0] if len(tiles) == 1 else torch.cat(tiles, dim=0)
