# SPDX-License-Identifier: Apache-2.0
"""The head of the prefill MoE: nkilib's rmsnorm_mx_prefill adds the residual, applies the post-attention RMSNorm,
quantizes the normed hidden to MX fp8 and routes it, in one pass.

Each output row packs ``[hidden fp8 (H) | MX scales | affinities (E bf16, as fp8 pairs) | pad]`` and is returned
viewed as bf16, so a gather of the rows carries the routing with them and moe_cte reads the affinities in place.
The kernel takes the hidden in the model's shuffled order (it de-interleaves on load), gamma in natural order and
the router weight in its swizzled order.
"""

import nki.language as nl
import torch
import torch.nn.functional as F
from nkilib.core.rmsnorm.rmsnorm_mx_prefill import rmsnorm_mx_prefill as _nki_rmsnorm_mx_prefill
from nkilib.core.utils.common_types import RouterActFnType
from torch import Tensor

from sglang_neuron.backend.stack import wrap_nki
from sglang_neuron.utils.hardware_config import get_logical_nc_config
from sglang_neuron.utils.neuron_utils import can_run_kernel

_MX_TILE = 512  # the kernel's hidden tile
_MX_BLOCK = 32  # hidden values per MX scale
_SCALE_BLOCK = 128  # scale columns per four hidden tiles
_QUADRANT = 32
_Q_WIDTH = 4
_FP8_MAX, _FP8_MAX_EXP = 448.0, 7


def scale_region(hidden_size: int) -> int:
    """The fp8 columns of a row's MX scales: 128 per four 512-wide hidden tiles."""
    return -(-(hidden_size // _MX_TILE) // 4) * _SCALE_BLOCK


def slice_affinities_from_packed(packed: Tensor, *, hidden_size: int, num_experts: int) -> Tensor:
    """The [T, E] bf16 affinities (zero outside each token's top-k) of bf16-viewed packed rows: a column slice."""
    start = (hidden_size + scale_region(hidden_size)) // 2
    return packed[:, start:start + num_experts]


def rmsnorm_mx_prefill(hidden_states: Tensor, *, residual: Tensor, gamma: Tensor, router_weights: Tensor,
                       router_bias: Tensor, eps: float, top_k: int, unpadded_hidden_size: int):
    """Norm and route ``hidden_states + residual`` ([T, H], shuffled hidden order; the mean is taken over
    ``unpadded_hidden_size``) with the softmax over each token's top-k router logits.

    gamma is [H] in natural order and router_weights [H, E] in the kernel's swizzled order. Returns ``(packed,
    expert_index, summed)``: the bf16-viewed packed rows, the [T, top_k] int32 experts and the pre-norm sum, the next
    residual.
    """
    if not can_run_kernel(hidden_states):
        return _torch_rmsnorm_mx_prefill(hidden_states, residual, gamma, router_weights, router_bias, eps, top_k,
                                         unpadded_hidden_size)
    return wrap_nki(_nki_rmsnorm_mx_prefill)[get_logical_nc_config()](
        hidden_states=hidden_states.unsqueeze(0),
        gamma=gamma.unsqueeze(0),
        router_weights=router_weights,
        router_bias=router_bias,
        eps=eps,
        top_k=top_k,
        router_act_fn=RouterActFnType.SOFTMAX,
        qmx_output_dtype=nl.float8_e4m3fn_x4,
        pack_scales=True,
        pack_affinities=True,
        unpadded_hidden_size=unpadded_hidden_size,
        hidden_interleaved=True,
        residual=residual.unsqueeze(0),
    )


def swizzle_h_index(hidden_size: int, device=None) -> Tensor:
    """For each slot of the kernel's swizzled hidden order, its natural index ``tile * 512 + 4 * p + q``."""
    tiles, partitions = hidden_size // _MX_TILE, _MX_TILE // _Q_WIDTH
    tile = torch.arange(tiles, device=device).reshape(tiles, 1, 1)
    q = torch.arange(_Q_WIDTH, device=device).reshape(1, _Q_WIDTH, 1)
    p = torch.arange(partitions, device=device).reshape(1, 1, partitions)
    return (tile * _MX_TILE + _Q_WIDTH * p + q).reshape(-1)


def _torch_rmsnorm_mx_prefill(hidden_states, residual, gamma, router_weights, router_bias, eps, top_k,
                              unpadded_hidden_size):
    """The kernel's outputs in torch. The packed hidden matches the kernel up to e4m3 rounding; the layout is
    exact."""
    summed = hidden_states + residual
    T, H = summed.shape
    # the shuffled order holds natural index 4 * p + q at q * H / 4 + p
    natural = summed.reshape(T, 4, H // 4).transpose(1, 2).reshape(T, H).float()
    variance = natural.pow(2).sum(-1, keepdim=True) / unpadded_hidden_size
    norm = (natural * torch.rsqrt(variance + eps) * gamma.float()).to(hidden_states.dtype)

    # the router projects the normed hidden in the swizzled order its weight is held in, in fp32
    logits = F.linear(norm.index_select(1, swizzle_h_index(H, norm.device)).float(), router_weights.T.float(),
                      router_bias.float())
    top, experts = torch.topk(logits, top_k, dim=-1)
    affinities = torch.zeros_like(logits).scatter_(1, experts, torch.softmax(top, dim=-1))
    return _pack(norm.float(), affinities, router_weights.dtype), experts.to(torch.int32), summed


def _pack(norm: Tensor, affinities: Tensor, compute_dtype: torch.dtype) -> Tensor:
    """The packed rows of the normed hidden [T, H] and the dense affinities [T, E], viewed as bf16."""
    T, H = norm.shape
    blocks = norm.to(compute_dtype).float().reshape(T, H // _MX_BLOCK, _MX_BLOCK)
    # a block's scale is its largest exponent less e4m3's, as a biased exponent byte (0 for an all-zero block)
    exponent = torch.floor(torch.log2(blocks.abs().amax(dim=2, keepdim=True).clamp(min=1e-30))) + 127.0
    scale = (exponent - _FP8_MAX_EXP).clamp(0, 255)
    factor = torch.pow(2.0, scale - 127.0)
    factor = torch.where(factor == 0, torch.ones_like(factor), factor)
    hidden = (blocks / factor).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn).reshape(T, H)

    # one scale byte per block at its column in the folded scale region, assembled into int32 words
    scales = torch.zeros(T, scale_region(H), dtype=torch.int32)
    scales = scales.index_copy(1, _scale_columns(H), scale.squeeze(-1).to(torch.int32)).reshape(T, -1, 4)
    words = scales[..., 0] | (scales[..., 1] << 8) | (scales[..., 2] << 16) | (scales[..., 3] << 24)

    parts = [hidden.view(torch.bfloat16), words.view(torch.bfloat16), affinities.to(torch.bfloat16)]
    # the fp8 row is padded to a multiple of 4 columns
    pad = -(H + scale_region(H) + 2 * affinities.shape[1]) % 4
    if pad:
        parts.append(torch.zeros(T, pad // 2, dtype=torch.bfloat16))
    return torch.cat(parts, dim=1)


def _scale_columns(hidden_size: int) -> Tensor:
    """The column of each 32-wide MX block's scale in the folded scale region."""
    b = torch.arange(hidden_size // _MX_BLOCK)
    tile, block = b // (_MX_TILE // _MX_BLOCK), b % (_MX_TILE // _MX_BLOCK)
    return (tile // 4) * _SCALE_BLOCK + (block // _Q_WIDTH) * _QUADRANT + (tile % 4) * _Q_WIDTH + block % _Q_WIDTH
