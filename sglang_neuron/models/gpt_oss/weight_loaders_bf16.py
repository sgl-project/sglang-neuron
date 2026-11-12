# SPDX-License-Identifier: Apache-2.0
"""Loaders of the GPT-OSS BF16 experts: read this rank's shard of the MXFP4 blocks and scales, dequantize it on the host
with SGLang's MXFP4 dequantization, and lay it out for the BF16 MoE kernels: gate and up de-interleaved, the hidden
dimension padded and shuffled like every other GPT-OSS tensor (weight_loaders_mxfp4._shuffle_hidden_dim)."""

import torch
from sglang.srt.layers.quantization.mxfp4_tensor import MXFP4QuantizeUtil

from sglang_neuron.models.gpt_oss.weight_loaders_mxfp4 import _shuffle_hidden_dim
from sglang_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    ShardSpec,
    get_shard,
    get_shard_deinterleaved,
    pad_to_shape,
)

# the MXFP4 block: 32 values, two to a byte
MX_BLOCK = 32
# md5-pinned against SGLang 43881a7, checked on a BF16 launch (sglang_model._check_expert_format): dequantize_mxfp4's
# reshape assumes its layout (low nibble = even value, E8M0 bias 127, unfused [..., G, 32] result), and any other
# layout with as many values would pass it silently
_SOURCES_MD5 = {
    ("sglang.srt.layers.quantization.mxfp4_tensor", "MXFP4QuantizeUtil.dequantize"): "dd7b5ac7897c7f52564b4a6c2e7f8568",
}


def dequantize_mxfp4(blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """MXFP4 ``blocks`` [..., G, 16] (two E2M1 values a byte) with E8M0 ``scales`` [..., G] as bf16 [..., G * 32].
    Exact: an E2M1 value times a power of two is a bf16."""
    values = MXFP4QuantizeUtil.dequantize(quantized_data=blocks, dtype=torch.bfloat16, scale=scales,
                                          block_sizes=[MX_BLOCK])
    return values.reshape(*scales.shape[:-1], scales.shape[-1] * MX_BLOCK)


def bf16_gate_up_weight_loader(shard_spec: ShardSpec, expected_shape: tuple[int, ...]) -> SafetensorsWeightLoader:
    """gate_up_proj from blocks [E, 2I, H / 32, 16] and scales [E, 2I, H / 32] (rows interleave gate and up): this
    rank's share of the 2I rows, de-interleaved, as [E, H, 2, I_shard] (gate at 0, up at 1), H padded and shuffled."""
    experts, hidden, _, rows = expected_shape

    def transform(slices: list, rank: int):
        blocks, scales = (get_shard_deinterleaved(s, shard_spec.dim, shard_spec.size, shard_spec.num_shards, rank)
                          for s in slices)
        halves = [pad_to_shape(dequantize_mxfp4(b, s), (experts, rows, hidden)) for b, s in zip(blocks, scales)]
        weight = _shuffle_hidden_dim(torch.stack(halves, dim=1), dim=-1)       # [E, 2, I_shard, H]
        return weight.permute(0, 3, 1, 2).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def bf16_gate_up_bias_loader(shard_spec: ShardSpec, expected_shape: tuple[int, ...],
                             hidden_act_bias: float) -> SafetensorsWeightLoader:
    """gate_up_proj_bias from the interleaved [E, 2I]: this rank's share as [E, 2, I_shard], with ``hidden_act_bias``
    added to the up half before padding (the padding stays zero)."""
    experts, _, rows = expected_shape

    def transform(slices: list, rank: int):
        (checkpoint,) = slices
        gate, up = get_shard_deinterleaved(checkpoint, shard_spec.dim, shard_spec.size, shard_spec.num_shards, rank)
        # out of place: an unpadded shard can be a view of the tensor the loader reads
        halves = [pad_to_shape(half, (experts, rows)) for half in (gate, up + hidden_act_bias)]
        return torch.stack(halves, dim=1).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def bf16_down_weight_loader(shard_spec: ShardSpec, expected_shape: tuple[int, ...]) -> SafetensorsWeightLoader:
    """down_proj from blocks [E, H, I / 32, 16] and scales [E, H, I / 32]: this rank's share of the I / 32 blocks as
    [E, I_shard, H], H padded and shuffled."""
    experts, rows, hidden = expected_shape

    def transform(slices: list, rank: int):
        blocks, scales = (get_shard(s, shard_spec.dim, shard_spec.size, shard_spec.num_shards, rank) for s in slices)
        weight = pad_to_shape(dequantize_mxfp4(blocks, scales), (experts, hidden, rows))
        return _shuffle_hidden_dim(weight, dim=1).transpose(1, 2).contiguous()

    return SafetensorsWeightLoader(transform=transform)
