# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch references for Kimi-K3 :func:`moe_tkg_decode`."""

import math

import nki.language as nl
import numpy as np
import torch

from ...utils.mx_torch_common import (
    mx_matmul,
    quantize_to_mx,
    unpack_float4_x4,
    unpack_float8_e4m3fn_x4,
)

_PMAX = 128
_MX_TILE = 512
_Q_WIDTH = 4
_Q_HEIGHT = 8


def moe_tkg_decode_torch_ref(
    latent,
    local_affinities,
    expert_gate_up_weights,
    expert_down_weights,
    expert_gate_up_weights_scale,
    expert_down_weights_scale,
    beta: float = 4.0,
    linear_beta: float = 25.0,
    weight_buffers: int = 3,
    scale_buffers: int = 2,
    early_loads: bool = True,
    is_all_expert_dynamic: bool = False,
    n_static_experts: int = 0,
    active_expert_slots: int = 0,
    active_expert_count=None,
    half_weight_buffers: int = 0,
    gate_up_half_major: bool = False,
    output=None,
    column_groups_out=None,
    direct_latent: bool = False,
    transpose_lookahead: int = 2,  # the kernel's _TRANSPOSE_LOOKAHEAD_CHUNKS
    cross_expert_order: bool = False,
    quantized_latent: tuple = None,
    shared_scale_buffer: bool = False,
    extra_weight_buffers: tuple = (),
    extra_buffers_after=None,
    static_alt_tiles: bool | str = False,
    cross_expert_transpose_blocks: bool = False,
    stream=None,
    expert_queues: int = 1,
    preamble_buffers: int = 0,
    preamble_reuse: bool = False,
    transpose_block: bool = False,
    psum_consume: bool = False,  # ignored: the product reads the FP32 PSUM tile in place of its FP32 copy
    accum_memset: tuple = None,  # ignored: the engine and the order of the zero fill
    first_add_zero: bool = False,  # ignored: expert 0 adds +0.0 in place of the zero fill, the same FP32 add
    up_product_split: bool = False,  # ignored: the SiTU product runs in 128-column pieces
    resident_out: dict = None,  # ignored: the kernel records the buffers of its last expert
    last_expert_order: str = "default",  # ignored: the order of the last experts' matmuls
    affinity_sbuf_copy: bool = False,  # ignored: the engine of the SBUF affinity copy
) -> dict[str, torch.Tensor]:
    """Match the kernel's hardware-MX and rounding boundaries.

    The matmuls accumulate in FP32 across their 512-row tiles, as the
    kernel's PSUM does, and the SiTU output is rounded to FP16 before its
    MXFP8 quantization. The output is rounded to BF16 once, after the FP32
    cross-expert accumulation.

    ``is_all_expert_dynamic``, ``n_static_experts`` and
    ``active_expert_slots`` only select the kernel's expert path, and
    ``weight_buffers``, ``half_weight_buffers``, ``scale_buffers`` and
    ``early_loads`` only set its SBUF buffers and load order; every choice
    computes this result.
    ``active_expert_count`` is the kernel's debug count output; the
    reference does not compute it. With ``gate_up_half_major`` the gate/up
    weights come in the kernel's half-major layout, and ``output`` only names
    the kernel's destination. With ``column_groups_out`` the result comes in
    the kernel's column-group layout ``[128, T, D/128]``
    (``_column_group_layout``). ``direct_latent`` only selects where the
    kernel reads its latent, ``transpose_lookahead`` only sets the order of
    its activation transposes, ``cross_expert_order`` only sets the order
    of its expert matmuls, ``cross_expert_transpose_blocks`` only groups its
    activation transposes and ``static_alt_tiles`` only sets the SBUF buffers
    of its SiTU, PSUM copy and eviction tiles and the copy engines, so none of
    them changes the result. Neither do ``shared_scale_buffer``,
    ``extra_weight_buffers`` and ``extra_buffers_after``: they only set the
    kernel's SBUF buffers and the order of its loads. ``stream`` (a prefilled expert stream),
    ``expert_queues``, ``preamble_buffers`` and ``preamble_reuse`` only set the kernel's load schedule and
    buffers, and ``transpose_block`` only the order of its transposes, so they do not change the result either.
    ``psum_consume``, ``accum_memset``, ``first_add_zero``, ``up_product_split``, ``resident_out`` and
    ``last_expert_order`` only set the kernel's tiles, the order of its instructions or a record of its buffers; the
    kernel adds the same FP32 values in the same order, so they do not change the result either.
    ``affinity_sbuf_copy`` only sets the engine that copies an SBUF ``local_affinities`` into the FP32 affinity
    tile; the copied values do not change.
    ``quantized_latent`` is the kernel's MXFP8 expert input: a (codes, scales) pair in the kernel layout (codes
    ``[128, D/512, T]`` float8_e4m3fn_x4, E8M0 scales ``[128, D/512, T]`` UINT8 with block k of each 512 tile at
    partition 32 (k // 4) + k % 4, T a multiple of 4). Then ``latent`` is None, and the reference reads these bytes in
    place of quantizing ``latent``, as the kernel does.
    """
    affinities_t = _to_torch(local_affinities).float()
    gate_up_scale_t = _to_torch(expert_gate_up_weights_scale).to(torch.uint8)
    down_scale_t = _to_torch(expert_down_weights_scale).to(torch.uint8)
    gate_up_weights_np = np.asarray(expert_gate_up_weights)
    if gate_up_half_major:
        gate_up_weights_np = _from_half_major(gate_up_weights_np)
    down_weights_np = np.asarray(expert_down_weights)

    experts = gate_up_weights_np.shape[0]
    intermediate = gate_up_weights_np.shape[-1]
    if quantized_latent is None:
        hidden_quant_np, hidden_scale_t, tokens = _quantize_latent_torch(latent)
    else:
        hidden_quant_np, hidden_scale_t = _read_quantized_latent(latent, quantized_latent)
        tokens = hidden_quant_np.shape[2]
    n_hidden_tiles = hidden_quant_np.shape[1]
    hidden = n_hidden_tiles * _MX_TILE
    padded_tokens = hidden_quant_np.shape[2]

    output = torch.zeros((padded_tokens, hidden), dtype=torch.float32)
    for expert_idx in range(experts):
        gate = _gate_up_projection_torch(
            hidden_qtz=hidden_quant_np,
            hidden_scale=hidden_scale_t,
            weight_qtz=gate_up_weights_np[expert_idx, :, 0, :, :],
            weight_scale=gate_up_scale_t[expert_idx, :, 0, :, :],
            hidden=hidden,
            intermediate=intermediate,
            tokens=padded_tokens,
        )
        up = _gate_up_projection_torch(
            hidden_qtz=hidden_quant_np,
            hidden_scale=hidden_scale_t,
            weight_qtz=gate_up_weights_np[expert_idx, :, 1, :, :],
            weight_scale=gate_up_scale_t[expert_idx, :, 1, :, :],
            hidden=hidden,
            intermediate=intermediate,
            tokens=padded_tokens,
        )

        gate_f32 = gate.float()
        up_f32 = up.float()
        gate_term = beta * torch.tanh(gate_f32 / beta) * torch.sigmoid(gate_f32)
        up_term = linear_beta * torch.tanh(up_f32 / linear_beta)
        activation = (gate_term * up_term).to(torch.float16)

        expert_out = _down_projection_torch(
            activation=activation,
            weight_qtz=down_weights_np[expert_idx],
            weight_scale=down_scale_t[expert_idx],
            hidden=hidden,
            intermediate=intermediate,
            tokens=padded_tokens,
        )
        affinity = torch.zeros((padded_tokens, 1), dtype=torch.float32)
        affinity[:tokens] = affinities_t[:, expert_idx : expert_idx + 1]
        output += expert_out.float() * affinity

    out = output[:tokens].to(torch.bfloat16)
    if column_groups_out is not None:
        out = _column_group_layout(out)
    return {"out": out}


def _quantize_latent_torch(latent) -> tuple[np.ndarray, torch.Tensor, int]:
    """Swizzle the natural ``[T, D]`` latent as the kernel does and quantize it to MXFP8.

    Return the codes ``[128, D/512, T4]`` (T4 = T rounded up to 4), the dense scales ``[16, D/512, T4]`` and T.
    """
    latent_t = _to_torch(latent).to(torch.bfloat16)
    tokens, hidden = latent_t.shape
    n_hidden_tiles = hidden // _MX_TILE
    padded_tokens = math.ceil(tokens / _Q_WIDTH) * _Q_WIDTH
    hidden_swizzled = torch.zeros(
        (_PMAX, n_hidden_tiles, padded_tokens, _Q_WIDTH),
        dtype=torch.float32,
    )
    hidden_swizzled[:, :, :tokens, :] = (
        latent_t.reshape(tokens, n_hidden_tiles, _PMAX, _Q_WIDTH).permute(2, 1, 0, 3).float()
    )
    hidden_quant_np, hidden_scale_np = quantize_to_mx(
        hidden_swizzled.reshape(_PMAX, -1).numpy(),
        nl.float8_e4m3fn_x4,
    )
    hidden_quant_np = hidden_quant_np.reshape(_PMAX, n_hidden_tiles, padded_tokens)
    hidden_scale_t = torch.from_numpy(hidden_scale_np.reshape(_PMAX // _Q_HEIGHT, n_hidden_tiles, padded_tokens))
    return hidden_quant_np, hidden_scale_t, tokens


def _read_quantized_latent(latent, quantized_latent: tuple) -> tuple[np.ndarray, torch.Tensor]:
    """Return the codes and the dense scales ``[16, D/512, T]`` of a kernel-layout ``quantized_latent`` pair.

    Block k of each 512 tile has its scale at partition 32 (k // 4) + k % 4 (the layout of ``nisa.quantize_mx``).
    """
    if latent is not None:
        raise ValueError("moe_tkg_decode takes latent None with quantized_latent")
    if len(quantized_latent) != 2:
        raise ValueError("quantized_latent is a (codes, scales) pair")
    codes, scales = (np.asarray(part) for part in quantized_latent)
    if codes.ndim != 3 or codes.shape[0] != _PMAX or scales.shape != codes.shape or codes.shape[2] % _Q_WIDTH:
        raise ValueError(
            f"quantized_latent must be codes [128, D/512, T] and scales of the same shape with T a multiple of "
            f"{_Q_WIDTH}, got {codes.shape} and {scales.shape}"
        )
    rows = [32 * (k // 4) + k % 4 for k in range(_PMAX // _Q_HEIGHT)]
    scales_t = torch.from_numpy(np.ascontiguousarray(scales.view(np.uint8)[rows]))
    return codes, scales_t


def _column_group_layout(token_major: torch.Tensor) -> torch.Tensor:
    """Return the ``[128, T, D/128]`` column-group layout of a ``[T, D]`` output: element ``(g, t, c)`` is column
    ``g * D/128 + c`` of token ``t``."""
    tokens, hidden = token_major.shape
    return token_major.reshape(tokens, _PMAX, hidden // _PMAX).permute(1, 0, 2).contiguous()


def _from_half_major(gate_up: np.ndarray) -> np.ndarray:
    """Undo the half-major gate/up layout: ``[E, 128, 2, 2, D/512, I/2]`` back to ``[E, 128, 2, D/512, I]``."""
    e, p, g, d, i = gate_up.shape
    halves = gate_up.reshape(e, p, g, 2, d, i // 2).transpose(0, 1, 2, 4, 3, 5)
    return np.ascontiguousarray(halves).reshape(gate_up.shape)


def _gate_up_projection_torch(
    hidden_qtz: np.ndarray,
    hidden_scale: torch.Tensor,
    weight_qtz: np.ndarray,
    weight_scale: torch.Tensor,
    hidden: int,
    intermediate: int,
    tokens: int,
) -> torch.Tensor:
    """Mirror the FP32 PSUM accumulation over the hidden 512 tiles."""
    n_hidden_tiles = hidden // _MX_TILE
    n_intermediate_tiles = intermediate // _MX_TILE
    hidden_unpacked = unpack_float8_e4m3fn_x4(hidden_qtz.transpose(1, 0, 2).reshape(hidden // _Q_WIDTH, tokens))
    weight_unpacked = unpack_float4_x4(weight_qtz.transpose(1, 0, 2).reshape(hidden // _Q_WIDTH, intermediate))
    hidden_scales = hidden_scale.permute(1, 0, 2)
    weight_scales = weight_scale.permute(1, 0, 2)

    result = torch.zeros((intermediate, tokens), dtype=torch.float32)
    for tile_h in range(n_hidden_tiles):
        packed_rows = slice(tile_h * _PMAX, (tile_h + 1) * _PMAX)
        tile_result = mx_matmul(
            weight_unpacked[packed_rows],
            hidden_unpacked[packed_rows],
            weight_scales[tile_h],
            hidden_scales[tile_h],
        )
        result = result + tile_result

    return result.reshape(n_intermediate_tiles, _Q_WIDTH, _PMAX, tokens).permute(2, 0, 3, 1)


def _down_projection_torch(
    activation: torch.Tensor,
    weight_qtz: np.ndarray,
    weight_scale: torch.Tensor,
    hidden: int,
    intermediate: int,
    tokens: int,
) -> torch.Tensor:
    """Mirror the activation quantization and the FP32 PSUM accumulation over the I512 tiles."""
    n_intermediate_tiles = intermediate // _MX_TILE
    activation_flat = activation.float().permute(1, 0, 2, 3).reshape(-1, tokens * _Q_WIDTH).numpy()
    activation_qtz, activation_scale = quantize_to_mx(activation_flat, nl.float8_e4m3fn_x4)
    activation_unpacked = unpack_float8_e4m3fn_x4(activation_qtz)
    weight_unpacked = unpack_float4_x4(weight_qtz.transpose(1, 0, 2).reshape(-1, hidden))
    activation_scales = torch.from_numpy(activation_scale.astype(np.float32)).reshape(
        n_intermediate_tiles,
        _PMAX // _Q_HEIGHT,
        tokens,
    )
    weight_scales = weight_scale.permute(1, 0, 2)

    result = torch.zeros((tokens, hidden), dtype=torch.float32)
    for tile_i in range(n_intermediate_tiles):
        packed_rows = slice(tile_i * _PMAX, (tile_i + 1) * _PMAX)
        tile_result = mx_matmul(
            activation_unpacked[packed_rows],
            weight_unpacked[packed_rows],
            activation_scales[tile_i],
            weight_scales[tile_i],
        )
        result = result + tile_result
    return result


def _to_torch(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    array = np.asarray(value)
    if "bfloat16" in str(array.dtype):
        return torch.from_numpy(array.astype(np.float32)).to(torch.bfloat16)
    return torch.from_numpy(array)
