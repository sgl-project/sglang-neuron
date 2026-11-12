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

"""Host layouts of the Kimi-K3 MLA block attention body (``mla_block_attention``).

The functions build the per-rank kernel weights for ``local_heads`` heads (with or without the gate
region) and the GPT-J RoPE rows of the drafter. They parallel ``mla_decode_torch`` and reuse its
constants, so the production decode layouts stay in that module.
"""

import numpy as np
import torch

from .draft_mla_torch import DraftMLADims, rope_cos_sin
from .mla_decode_torch import (
    BLOCK,
    CACHE_WIDTH,
    HIDDEN,
    KV_LORA,
    NOPE,
    Q_HEAD,
    Q_LORA,
    ROPE,
    TP,
    V_HEAD,
    _to_bf16_numpy,
    to_hT_order,
)


def mla_block_rank_weights(weights: dict, tp_rank: int, local_heads: int, output_gate: bool) -> dict[str, np.ndarray]:
    """Build the ``MLA_WEIGHT_NAMES`` kernel weights of one TP32 rank for ``local_heads`` heads.

    This generalizes ``mla_decode_rank_weights`` to a head count of ``32 * local_heads`` and an
    optional gate region. Rank ``t`` owns heads ``local_heads t .. local_heads t + local_heads - 1``
    and the q_a columns ``48 t ..``; its kv_a column ``c`` is kv_a feature ``32 c + t``.

    Args:
        weights: global weights in ``weight[out, in]`` orientation, keyed ``q_a_proj``,
            ``q_a_layernorm``, ``q_b_proj``, ``kv_a_proj``, ``kv_a_layernorm``, ``kv_b_proj``,
            ``o_proj``, and ``g_proj`` when ``output_gate``.
        tp_rank: CP32 rank ``t``.
        local_heads: heads of this rank (2 for the drafter, 3 for the target).
        output_gate: include the gate region (the target), else omit it (the drafter).

    Returns:
        A dict with the keys of ``MLA_WEIGHT_NAMES``.
    """
    t = tp_rank
    w = {name: np.asarray(_to_bf16_numpy(value)) for name, value in weights.items()}
    q_lora_local = Q_LORA // TP
    cache_local = CACHE_WIDTH // TP
    v_local = local_heads * V_HEAD
    kv_a_rows = [c * TP + t for c in range(cache_local)]

    parts = [w["q_a_proj"][t * q_lora_local : (t + 1) * q_lora_local], w["kv_a_proj"][kv_a_rows]]
    if output_gate:
        parts.append(w["g_proj"][t * v_local : (t + 1) * v_local])
    in_proj = np.concatenate(parts, axis=0).T
    qkv_local = q_lora_local + cache_local
    in_tiles = in_proj.reshape(HIDDEN // BLOCK, BLOCK, in_proj.shape[1]).transpose(1, 0, 2)
    qkv_part = in_tiles[:, :, :qkv_local].reshape(BLOCK, -1)
    if output_gate:
        in_proj = np.concatenate((qkv_part, in_tiles[:, :, qkv_local:].reshape(BLOCK, -1)), axis=1)
    else:
        in_proj = qkv_part

    q_b = w["q_b_proj"][t * local_heads * Q_HEAD : (t + 1) * local_heads * Q_HEAD].T
    q_b = q_b.reshape(Q_LORA // BLOCK, BLOCK, q_b.shape[1]).transpose(1, 0, 2).reshape(BLOCK, -1)

    kv_b = w["kv_b_proj"][t * local_heads * (NOPE + V_HEAD) : (t + 1) * local_heads * (NOPE + V_HEAD)]
    kv_b = kv_b.reshape(local_heads, NOPE + V_HEAD, KV_LORA)
    absorb = kv_b[:, :NOPE, :].transpose(1, 0, 2).reshape(NOPE, local_heads * KV_LORA)
    value = kv_b[:, NOPE:, :].reshape(local_heads, V_HEAD, KV_LORA // BLOCK, BLOCK)
    value = value.transpose(3, 2, 0, 1).reshape(BLOCK, KV_LORA // BLOCK * local_heads * V_HEAD)

    o_proj = w["o_proj"][:, t * v_local : (t + 1) * v_local].T

    return {
        "mla_in_proj": np.ascontiguousarray(in_proj),
        "mla_q_a_norm": to_hT_order(w["q_a_layernorm"]),
        "mla_q_b_proj": np.ascontiguousarray(q_b),
        "mla_kv_a_norm": np.ascontiguousarray(w["kv_a_layernorm"]),
        "mla_kv_b_proj": np.ascontiguousarray(np.concatenate((absorb, value), axis=1)),
        "mla_o_proj": np.ascontiguousarray(o_proj),
    }


def mla_block_cp_rank_weights(
    weights: dict, rank: int, heads: int, output_gate: bool, cp: int = 128
) -> dict[str, np.ndarray]:
    """Build the ``MLA_WEIGHT_NAMES`` kernel weights of rank ``g`` of a CP``cp`` group (``mla_block_attention``
    with ``cp=128``, one local head).

    Rank ``g`` owns head ``g`` when ``g < heads`` (the ranks at and above ``heads`` get zero head weights), the
    q_a columns ``12 g .. 12 g + 11``, the kv latent features ``128 c + g`` (c = 0..3) and the k_pe feature ``g``
    when ``g < 64`` (a zero column above). The tensors have the CP32 layouts of ``mla_block_rank_weights`` for one
    local head, with 17 qkv columns.

    Args:
        weights: global weights in ``weight[out, in]`` orientation, as ``mla_block_rank_weights``.
        rank: CP rank ``g``.
        heads: heads of the layer (96 for the target, 64 for the drafter).
        output_gate: include the gate region (the target), else omit it (the drafter).
        cp: CP ranks.

    Returns:
        A dict with the keys of ``MLA_WEIGHT_NAMES``.
    """
    g = rank
    w = {name: np.asarray(_to_bf16_numpy(value)) for name, value in weights.items()}
    q_lora_local = Q_LORA // cp
    kv_latent_local = KV_LORA // cp
    zero_row = np.zeros((1, HIDDEN), dtype=w["kv_a_proj"].dtype)
    kv_rows = w["kv_a_proj"][[c * cp + g for c in range(kv_latent_local)]]
    kpe_row = w["kv_a_proj"][KV_LORA + g : KV_LORA + g + 1] if g < ROPE else zero_row
    own = g < heads

    def head_rows(name: str, width: int) -> np.ndarray:
        if own:
            return w[name][g * width : (g + 1) * width]
        return np.zeros((width, w[name].shape[1]), dtype=w[name].dtype)

    parts = [w["q_a_proj"][g * q_lora_local : (g + 1) * q_lora_local], kv_rows, kpe_row]
    if output_gate:
        parts.append(head_rows("g_proj", V_HEAD))
    in_proj = np.concatenate(parts, axis=0).T
    qkv_local = q_lora_local + kv_latent_local + 1
    in_tiles = in_proj.reshape(HIDDEN // BLOCK, BLOCK, in_proj.shape[1]).transpose(1, 0, 2)
    in_proj = in_tiles[:, :, :qkv_local].reshape(BLOCK, -1)
    if output_gate:
        in_proj = np.concatenate((in_proj, in_tiles[:, :, qkv_local:].reshape(BLOCK, -1)), axis=1)

    q_b = head_rows("q_b_proj", Q_HEAD).T
    q_b = q_b.reshape(Q_LORA // BLOCK, BLOCK, q_b.shape[1]).transpose(1, 0, 2).reshape(BLOCK, -1)

    kv_b = head_rows("kv_b_proj", NOPE + V_HEAD).reshape(1, NOPE + V_HEAD, KV_LORA)
    absorb = kv_b[:, :NOPE, :].transpose(1, 0, 2).reshape(NOPE, KV_LORA)
    value = kv_b[:, NOPE:, :].reshape(1, V_HEAD, KV_LORA // BLOCK, BLOCK)
    value = value.transpose(3, 2, 0, 1).reshape(BLOCK, KV_LORA // BLOCK * V_HEAD)

    if own:
        o_proj = w["o_proj"][:, g * V_HEAD : (g + 1) * V_HEAD].T
    else:
        o_proj = np.zeros((V_HEAD, w["o_proj"].shape[0]), dtype=w["o_proj"].dtype)

    return {
        "mla_in_proj": np.ascontiguousarray(in_proj),
        "mla_q_a_norm": to_hT_order(w["q_a_layernorm"]),
        "mla_q_b_proj": np.ascontiguousarray(q_b),
        "mla_kv_a_norm": np.ascontiguousarray(w["kv_a_layernorm"]),
        "mla_kv_b_proj": np.ascontiguousarray(np.concatenate((absorb, value), axis=1)),
        "mla_o_proj": np.ascontiguousarray(o_proj),
    }


def mla_block_rope_rows(positions: np.ndarray, dims: DraftMLADims) -> np.ndarray:
    """Return the ``[Q*B, 128]`` FP32 GPT-J RoPE rows of the query rows at ``positions``.

    Each row holds ``[cos (64) | sin (64)]`` pre-expanded for the pairs: element ``2i`` and ``2i+1``
    both hold the frequency ``i``, so the kernel rotation is ``x * cos + rot(x) * sin``.
    """
    cos, sin = rope_cos_sin(torch.as_tensor(np.asarray(positions)).reshape(-1), dims)
    cos = cos.repeat_interleave(2, dim=-1).numpy().astype(np.float32)
    sin = sin.repeat_interleave(2, dim=-1).numpy().astype(np.float32)
    return np.ascontiguousarray(np.concatenate((cos, sin), axis=-1))


def apply_rope_rows(x: np.ndarray, rope_rows: np.ndarray) -> np.ndarray:
    """Rotate ``x`` ``[rows, ..., 64]`` with ``rope_rows`` ``[rows, 128]`` (``[cos (64) | sin (64)]``).

    This is the host form of the kernel rotation, used to check the RoPE-row layout against
    ``draft_mla_torch.apply_rope``.
    """
    cos = rope_rows[:, :ROPE]
    sin = rope_rows[:, ROPE : 2 * ROPE]
    shape = (x.shape[0],) + (1,) * (x.ndim - 2) + (ROPE,)
    c = cos.reshape(shape)
    s = sin.reshape(shape)
    rot = np.empty_like(x, dtype=np.float32)
    xf = x.astype(np.float32)
    rot[..., 0::2] = -xf[..., 1::2]
    rot[..., 1::2] = xf[..., 0::2]
    return xf * c + rot * s
