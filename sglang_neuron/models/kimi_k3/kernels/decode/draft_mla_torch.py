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

"""Torch reference of the Kimi-K3 DSpark MLA drafter (one draft step).

The drafter has dense layers with MLA attention. Its cache rows have the target MLA format
``[rms(latent) * kv_a_norm | k_pe]`` (576 elements), but the drafter rotates ``q_pe`` and ``k_pe``
with YaRN RoPE (GPT-J interleaved pairs), and it has 64 heads. One draft step:

1. Context. For each verified target row, ``x = context_norm(context_proj(cat(aux)))``. For each
   layer ``l``, the cache row at the row position is ``[kv_a_norm_l(x Wkv_a_l [:512]) |
   rope(x Wkv_a_l [512:])]``. All layers project the same ``x``.
2. Block. Seven query rows at positions ``P .. P+6``: the anchor token, then mask tokens. Each layer
   writes the cache rows of its normalized input at the block positions, and every query attends to
   all keys at a position at or below ``P+6`` (non-causal).
3. Tail. Row ``i`` drafts the token at ``P+i+1``: ``logits_i = lm_head(final_norm(h_i)) +
   markov_w2 @ markov_w1[prev]``, with ``prev`` the anchor for row 0 and the previous draft after.

The functions round to BF16 at the module outputs and accumulate in FP32. ``absorbed=True`` runs the
attention in the latent space (the kernel form); ``absorbed=False`` expands the keys and values with
``kv_b`` (the training form). The two forms agree within the BF16 rounding.
"""

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class DraftMLADims:
    """Dimensions of the drafter. The defaults are the published Kimi-K3 DSpark drafter."""

    hidden: int = 7168
    layers: int = 5
    heads: int = 64
    q_lora: int = 1536
    kv_lora: int = 512
    nope: int = 128
    rope: int = 64
    v_head: int = 128
    intermediate: int = 14336
    vocab: int = 163840
    markov_rank: int = 256
    aux_layers: int = 5
    block: int = 7
    eps: float = 1e-5
    rope_theta: float = 50000.0
    yarn_factor: float = 32.0
    yarn_original: int = 32768
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    yarn_mscale: float = 1.0
    yarn_mscale_all_dim: float = 1.0
    mask_token: int = 163837

    @property
    def q_head(self) -> int:
        return self.nope + self.rope

    @property
    def cache_width(self) -> int:
        return self.kv_lora + self.rope

    @property
    def softmax_scale(self) -> float:
        """``q_head^-0.5`` times the square of the YaRN attention mscale."""
        return self.q_head**-0.5 * _yarn_mscale(self.yarn_factor, self.yarn_mscale_all_dim) ** 2


def _yarn_mscale(factor: float, mscale: float) -> float:
    return 1.0 if factor <= 1 else 0.1 * mscale * math.log(factor) + 1.0


def _bf16(x: torch.Tensor) -> torch.Tensor:
    """Round to BF16 and return FP32 values."""
    return x.to(torch.bfloat16).float()


def _rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Return ``bf16(bf16(x * rsqrt(mean(x^2) + eps)) * w)`` as FP32 values."""
    x32 = x.float()
    normed = _bf16(x32 * torch.rsqrt(x32.square().mean(dim=-1, keepdim=True) + eps))
    return _bf16(normed * weight.float())


def _linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Return the BF16-rounded ``x @ weight.T`` with an FP32 accumulation."""
    return _bf16(x.float() @ weight.float().T)


def yarn_inv_freq(dims: DraftMLADims) -> torch.Tensor:
    """Return the FP32 YaRN inverse frequencies ``[rope / 2]`` (DeepSeek YaRN scaling)."""
    dim = dims.rope
    pos_freqs = dims.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    extrapolation = 1.0 / pos_freqs
    interpolation = 1.0 / (dims.yarn_factor * pos_freqs)

    def correction_dim(rotations: float) -> float:
        return (dim * math.log(dims.yarn_original / (rotations * 2 * math.pi))) / (2 * math.log(dims.rope_theta))

    low = max(math.floor(correction_dim(dims.yarn_beta_fast)), 0)
    high = min(math.ceil(correction_dim(dims.yarn_beta_slow)), dim - 1)
    if low == high:
        high += 0.001
    ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
    mask = 1 - ramp
    return interpolation * (1 - mask) + extrapolation * mask


def rope_cos_sin(positions: torch.Tensor, dims: DraftMLADims) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the FP32 ``cos`` and ``sin`` ``[T, rope / 2]`` of integer positions.

    The YaRN table scales both by ``mscale(mscale) / mscale(mscale_all_dim)``, which is 1 for the
    published drafter.
    """
    scale = _yarn_mscale(dims.yarn_factor, dims.yarn_mscale) / _yarn_mscale(dims.yarn_factor, dims.yarn_mscale_all_dim)
    freqs = positions.to(torch.float32).reshape(-1, 1) * yarn_inv_freq(dims)
    return freqs.cos() * scale, freqs.sin() * scale


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the interleaved pairs ``(x[2i], x[2i+1])`` of ``x [T, ..., rope]`` in FP32.

    ``cos`` and ``sin`` are ``[T, rope / 2]``; the result is BF16-rounded.
    """
    shape = (cos.shape[0],) + (1,) * (x.dim() - 2) + (cos.shape[1],)
    c, s = cos.reshape(shape), sin.reshape(shape)
    x1, x2 = x.float()[..., 0::2], x.float()[..., 1::2]
    out = torch.stack((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1)
    return _bf16(out.flatten(-2))


@dataclass
class DraftMLAWeights:
    """Global drafter weights as FP32 tensors in the ``weight[out, in]`` orientation."""

    context_proj: torch.Tensor
    context_norm: torch.Tensor
    final_norm: torch.Tensor
    markov_w1: torch.Tensor
    markov_w2: torch.Tensor
    layers: list[dict[str, torch.Tensor]]


_LAYER_NAMES = {
    "input_layernorm": "input_layernorm.weight",
    "post_attention_layernorm": "post_attention_layernorm.weight",
    "q_a_proj": "self_attn.q_a_proj.weight",
    "q_a_layernorm": "self_attn.q_a_layernorm.weight",
    "q_b_proj": "self_attn.q_b_proj.weight",
    "kv_a_proj": "self_attn.kv_a_proj_with_mqa.weight",
    "kv_a_layernorm": "self_attn.kv_a_layernorm.weight",
    "kv_b_proj": "self_attn.kv_b_proj.weight",
    "o_proj": "self_attn.o_proj.weight",
    "gate_proj": "mlp.gate_proj.weight",
    "up_proj": "mlp.up_proj.weight",
    "down_proj": "mlp.down_proj.weight",
}


def draft_mla_weights(tensors: dict, layers: int) -> DraftMLAWeights:
    """Build the reference weights from the checkpoint tensors (names of the published drafter)."""

    def get(name: str) -> torch.Tensor:
        value = tensors[name]
        if isinstance(value, torch.Tensor):
            return value.float()
        if value.dtype == np.uint16:
            return bf16_bits_to_torch(value)
        return torch.from_numpy(np.asarray(value)).float()

    return DraftMLAWeights(
        context_proj=get("context_proj.weight"),
        context_norm=get("context_norm.weight"),
        final_norm=get("final_norm.weight"),
        markov_w1=get("markov_head.markov_w1.weight"),
        markov_w2=get("markov_head.markov_w2.weight"),
        layers=[{key: get(f"layers.{index}.{name}") for key, name in _LAYER_NAMES.items()} for index in range(layers)],
    )


def context_feature(aux: torch.Tensor, w: DraftMLAWeights, dims: DraftMLADims) -> torch.Tensor:
    """Return ``x = context_norm(context_proj(aux))`` ``[T, H]`` of the concatenated aux states ``[T, 5H]``."""
    return _rms(_linear(_bf16(aux.float()), w.context_proj), w.context_norm, dims.eps)


def cache_rows(
    x: torch.Tensor, positions: torch.Tensor, layer: dict[str, torch.Tensor], dims: DraftMLADims
) -> torch.Tensor:
    """Return the cache rows ``[T, 576]`` of one layer: ``[kv_a_norm(latent) | rope(k_pe)]``."""
    kv_a = _linear(x, layer["kv_a_proj"])
    latent = _rms(kv_a[:, : dims.kv_lora], layer["kv_a_layernorm"], dims.eps)
    cos, sin = rope_cos_sin(positions, dims)
    return torch.cat((latent, apply_rope(kv_a[:, dims.kv_lora :], cos, sin)), dim=-1)


def context_rows(x: torch.Tensor, positions: torch.Tensor, w: DraftMLAWeights, dims: DraftMLADims) -> torch.Tensor:
    """Return the context cache rows ``[L, T, 576]`` of all layers for the context feature ``x``."""
    return torch.stack([cache_rows(x, positions, layer, dims) for layer in w.layers])


def _attention(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    keys: torch.Tensor,
    kv_b: torch.Tensor,
    dims: DraftMLADims,
    absorbed: bool,
) -> torch.Tensor:
    """Return the per-head values ``[Q, heads, v_head]`` of queries over all ``keys`` ``[K, 576]``."""
    w_uk = kv_b[:, : dims.nope, :]
    w_uv = kv_b[:, dims.nope :, :]
    latent, k_pe = keys[:, : dims.kv_lora], keys[:, dims.kv_lora :]
    rope_scores = torch.einsum("qhr,kr->qhk", q_pe, k_pe)
    if absorbed:
        q_abs = _bf16(torch.einsum("qhd,hdc->qhc", q_nope, w_uk))
        scores = (torch.einsum("qhc,kc->qhk", q_abs, latent) + rope_scores) * dims.softmax_scale
        probs = torch.softmax(scores, dim=-1)
        out_latent = _bf16(torch.einsum("qhk,kc->qhc", probs, latent))
        return _bf16(torch.einsum("qhc,hvc->qhv", out_latent, w_uv))
    k_nope = _bf16(torch.einsum("kc,hdc->khd", latent, w_uk))
    values = _bf16(torch.einsum("kc,hvc->khv", latent, w_uv))
    scores = (torch.einsum("qhd,khd->qhk", q_nope, k_nope) + rope_scores) * dims.softmax_scale
    probs = torch.softmax(scores, dim=-1)
    return _bf16(torch.einsum("qhk,khv->qhv", probs, values))


def draft_layer_attention(
    x: torch.Tensor,
    positions: torch.Tensor,
    prior_rows: torch.Tensor,
    layer: dict[str, torch.Tensor],
    dims: DraftMLADims,
    absorbed: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the MLA attention of one drafter layer on the normalized block input.

    This is the body that the ``mla_block_attention`` kernel computes: the q path, the RoPE, the
    block cache rows, the non-causal attention over the prior and block keys, and ``o_proj``. It takes
    the already input-normalized stream and returns the attention output before the residual add.

    Args:
        x: ``[Q, H]`` input-normalized block stream (``input_layernorm`` applied).
        positions: ``[Q]`` block positions.
        prior_rows: ``[K, 576]`` cache rows of this layer at the positions before the block.
        layer: the layer weights.
        dims: drafter dimensions.
        absorbed: run the attention in the latent space.

    Returns:
        ``(attn_out [Q, H], block_rows [Q, 576])``: the ``o_proj`` output of the attention (before the
        residual add), and the cache rows that the layer writes at the block positions.
    """
    queries = x.shape[0]
    q_latent = _rms(_linear(x, layer["q_a_proj"]), layer["q_a_layernorm"], dims.eps)
    q = _linear(q_latent, layer["q_b_proj"]).reshape(queries, dims.heads, dims.q_head)
    cos, sin = rope_cos_sin(positions, dims)
    q_pe = apply_rope(q[..., dims.nope :], cos, sin)
    block_rows = cache_rows(x, positions, layer, dims)
    keys = torch.cat((prior_rows.float(), block_rows), dim=0)
    kv_b = layer["kv_b_proj"].reshape(dims.heads, dims.nope + dims.v_head, dims.kv_lora)
    values = _attention(q[..., : dims.nope], q_pe, keys, kv_b, dims, absorbed)
    return _linear(values.reshape(queries, dims.heads * dims.v_head), layer["o_proj"]), block_rows


def draft_layer(
    h: torch.Tensor,
    positions: torch.Tensor,
    prior_rows: torch.Tensor,
    layer: dict[str, torch.Tensor],
    dims: DraftMLADims,
    absorbed: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one drafter layer on the block rows.

    Args:
        h: ``[Q, H]`` block stream.
        positions: ``[Q]`` block positions.
        prior_rows: ``[K, 576]`` cache rows of this layer at the positions before the block.
        layer: the layer weights.
        dims: drafter dimensions.
        absorbed: run the attention in the latent space.

    Returns:
        ``(h_out [Q, H], block_rows [Q, 576])``: the block stream after the layer, and the cache rows
        that the layer writes at the block positions.
    """
    x = _rms(h, layer["input_layernorm"], dims.eps)
    attn_out, block_rows = draft_layer_attention(x, positions, prior_rows, layer, dims, absorbed)
    h = _bf16(h + attn_out)
    x = _rms(h, layer["post_attention_layernorm"], dims.eps)
    act = _bf16(torch.nn.functional.silu(_linear(x, layer["gate_proj"])))
    mlp = _linear(_bf16(act * _linear(x, layer["up_proj"])), layer["down_proj"])
    return _bf16(h + mlp), block_rows


def draft_block(
    anchor: int,
    position: int,
    prior_rows: torch.Tensor,
    embed: torch.Tensor,
    w: DraftMLAWeights,
    dims: DraftMLADims,
    absorbed: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the drafter layers on one block.

    Args:
        anchor: the anchor token (the bonus token of the last verify step) at ``position``.
        position: ``P``, the anchor position. The block rows are at ``P .. P + block - 1``.
        prior_rows: ``[L, P, 576]`` cache rows of each layer at the positions ``0 .. P-1``.
        embed: ``[V, H]`` target token embedding.
        w: drafter weights.
        dims: drafter dimensions.
        absorbed: run the attention in the latent space.

    Returns:
        ``(hidden [block, H], block_rows [L, block, 576])``: the final-normed block stream, and the
        cache rows of each layer at the block positions.
    """
    tokens = torch.tensor([anchor] + [dims.mask_token] * (dims.block - 1))
    positions = torch.arange(position, position + dims.block)
    h = _bf16(embed[tokens].float())
    written = []
    for index, layer in enumerate(w.layers):
        h, rows = draft_layer(h, positions, prior_rows[index], layer, dims, absorbed)
        written.append(rows)
    return _rms(h, w.final_norm, dims.eps), torch.stack(written)


def draft_tokens(hidden: torch.Tensor, lm_head: torch.Tensor, w: DraftMLAWeights, anchor: int) -> torch.Tensor:
    """Return the greedy drafts ``[block]`` int64 of the final-normed block stream.

    Row ``i`` adds the Markov bias of the previous token: the anchor for row 0, then the draft of
    row ``i - 1``.
    """
    base = _bf16(hidden.float() @ lm_head.float().T)
    drafts = []
    prev = anchor
    for row in range(hidden.shape[0]):
        bias = _bf16(w.markov_w2 @ w.markov_w1[prev])
        prev = int(torch.argmax(base[row] + bias))
        drafts.append(prev)
    return torch.tensor(drafts, dtype=torch.int64)


SAMPLE_CANDIDATES = 256
"""Candidates of a sampled draft slot at CP32 and CP128 (the sampler's ``max_top_k``)."""
SAMPLING_EPS = 1e-5
U_MIN = 1.1754944e-38


def sample_width(cp: int) -> int:
    """Return the candidates of a sampled slot: ``cp`` vocab shards times ``min(8, 256 // cp)``."""
    return cp * min(8, SAMPLE_CANDIDATES // cp)


def slot_candidates(logits: torch.Tensor, cp: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the candidates ``(values [K] fp32, ids [K] int64)`` of full-vocab logits ``[V]``.

    Each of the ``cp`` vocab shards gives its largest ``K / cp`` values (the smallest id first at a tie), in
    shard order (the kernel gather order).
    """
    per_shard = sample_width(cp) // cp
    shards = logits.float().reshape(cp, -1)
    order = torch.sort(-shards, dim=1, stable=True).indices[:, :per_shard]
    values = torch.gather(shards, 1, order).reshape(-1)
    ids = (order + torch.arange(cp).unsqueeze(1) * shards.shape[1]).reshape(-1)
    return values, ids


def draw_candidates(
    values: torch.Tensor, ids: torch.Tensor, top_k: float, top_p: float, temperature: float, u: float
) -> tuple[int, torch.Tensor, torch.Tensor]:
    """Return ``(draft, q_probs [K] fp32, q_ids [K] int64)`` of one slot (the rule of ``draft_mla._draw_candidates``).

    A greedy row (temperature below 1e-5) takes the largest value with the smallest id, with a one-hot q. A sampled
    row cuts by value thresholds: top_k at the ``top_k``-th descending value, top_p at the last descending value whose
    exclusive mass is below ``top_p``. Then it draws with ``u`` by the inverse CDF over the kept candidates in gather
    order; q is the distribution of that draw.
    """
    k = values.shape[0]
    values = values.float()
    best = values.max()
    if temperature < SAMPLING_EPS:
        token = int(ids[values == best].min())
        q = torch.zeros(k, dtype=torch.float32)
        q[0] = 1.0
        return token, q, torch.full((k,), token, dtype=torch.int64)
    inv_t = 1.0 / max(temperature, SAMPLING_EPS)
    desc = torch.sort(values, descending=True).values
    kth = k - 1 if top_k <= 0 else min(top_k, k) - 1
    t_k = desc[max(math.floor(kth), 0)]
    kept = torch.exp((desc - best) * inv_t) * (desc >= t_k)
    total = kept.sum()
    mass = (2.0 if top_p >= 1.0 else max(top_p, 1e-6)) * total
    inside = int(((torch.cumsum(kept, 0) - kept) < mass).sum())
    t_p = torch.maximum(desc[max(inside - 1, 0)], t_k)
    weight = torch.exp((values - best) * inv_t) * (values >= t_p)
    cdf = torch.cumsum(weight, 0)
    position = int((cdf < max(u, U_MIN) * cdf[-1]).sum())
    return int(ids[position]), weight / cdf[-1], ids.clone()


def draft_tokens_sampled(
    hidden: torch.Tensor,
    lm_head: torch.Tensor,
    w: DraftMLAWeights,
    anchor: int,
    params: tuple[float, float, float],
    u_draft: torch.Tensor,
    cp: int,
    prev_drafts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the drafts ``[block]`` int64 and the q lists ``[block, K]`` of the sampled drafter tail.

    ``params`` holds ``(top_k, top_p, temperature)`` and ``u_draft`` ``[block]`` the draws. Row ``i`` adds the
    Markov bias of the previous token to its BF16 block logits and draws with ``draw_candidates``. With
    ``prev_drafts`` the previous token of row ``i`` is ``prev_drafts[i - 1]`` (teacher forcing).
    """
    base = _bf16(hidden.float() @ lm_head.float().T)
    drafts, q_probs, q_ids = [], [], []
    prev = anchor
    for row in range(hidden.shape[0]):
        bias = _bf16(w.markov_w2 @ w.markov_w1[prev])
        values, ids = slot_candidates(_bf16(base[row] + bias), cp)
        token, q, q_id = draw_candidates(values, ids, *params, float(u_draft[row]))
        drafts.append(token)
        q_probs.append(q)
        q_ids.append(q_id)
        prev = token if prev_drafts is None else int(prev_drafts[row])
    return torch.tensor(drafts, dtype=torch.int64), torch.stack(q_probs), torch.stack(q_ids)


def open_safetensors(path: str | Path) -> dict[str, np.ndarray]:
    """Return zero-copy memory-mapped views of the BF16 tensors of a safetensors file.

    Each view holds the raw BF16 bits as ``uint16``; :func:`bf16_bits_to_torch` converts one view.
    """
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)
    data = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + header_len)
    tensors = {}
    for name, meta in header.items():
        if meta["dtype"] != "BF16":
            raise ValueError(f"Tensor {name} is {meta['dtype']}; this reader takes BF16 only.")
        start, end = meta["data_offsets"]
        tensors[name] = data[start:end].view(np.uint16).reshape(meta["shape"])
    return tensors


def bf16_bits_to_torch(bits: np.ndarray) -> torch.Tensor:
    """Return the FP32 tensor of a ``uint16`` array of BF16 bits (copies only this tensor)."""
    return torch.from_numpy(np.ascontiguousarray(bits).astype(np.int16)).view(torch.bfloat16).float()
