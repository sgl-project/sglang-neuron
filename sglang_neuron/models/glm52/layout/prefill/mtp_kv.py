# SPDX-License-Identifier: Apache-2.0
"""The prompt rows of the GLM-5.2 MTP layer, made by the prefill engine.

The decode engine runs the MTP draft layer (checkpoint layer 78) in the graph of
the target (`models/glm52/mtp/draft.py`). The draft attention reads
its own latent cache and its own indexer key cache. A disaggregated decode node
gets the caches of the 78 target layers from the prefill node. Without the MTP
rows of the prompt, the draft attends to rows that no step wrote, and fewer
drafts pass.

This module makes those rows on the prefill's last pipeline stage
(`StageLayout.mtp_rows`), in the formats of the target layers; a prefill launched
with ``SGLANG_NEURON_GLM52_PD_MTP=1`` sends them to the decode. The row of
position ``t`` needs no attention and no MoE::

    x_t = eh_proj([enorm(embed(token[t + 1])); hnorm(h_t)])
    n_t = input_layernorm(x_t)
    latent row:  kv_a(n_t), then RMSNorm of the 512 latent values and RoPE at t
    indexer key: LayerNorm(n_t @ wk), then RoPE at t on the leading 64 values

``h_t`` is the final hidden state of the target, after ``model.norm``. The step 0
of the draft writes the same rows at the decode positions. The row of the last
prompt position needs the first sampled token: for a decode that drafts, the prefill
stage that samples that token writes the row with it (`prefill.stage.Glm52PrefillStage._write_mtp_rows`).

The rows must be the ones the decode's draft would write, so the norm epsilons are
the decode geometry's (`gk_geometry`) and the fp8 range is the decode kernel's row
format's. The prefill loaders (``layout/prefill/weight_loaders``) take the MTP
layer's checkpoint keys from `mtp_hf_mappings`.

Example:
    >>> x = mtp_input(embeds, hidden, enorm_w, hnorm_w, eh_proj, 1e-5)  # doctest: +SKIP
    >>> k_pe, c_kv = latent_rows(rms_norm(x, norm_w, 1e-5), wkv_a, kv_norm_w,  # doctest: +SKIP
    ...                          cos, sin, 512, 1e-6)
"""

from __future__ import annotations

import torch
from torch import nn

from sglang_neuron.models.glm52.kernels.decode.attention.mla.kv_row_layout import (  # noqa: E501
    E4M3FN_MAX,
)

from sglang_neuron.models.glm52.layout import glm_kv_row

# INPUT_NORM_EPS is the epsilon of enorm, hnorm and input_layernorm (`rms_norm_eps`); LATENT_NORM_EPS the KV latent
# norm's and IDX_NORM_EPS the indexer LayerNorm's.
from sglang_neuron.models.glm52.layout.decode.gk_geometry import IDX_NORM_EPS, INPUT_NORM_EPS, LATENT_NORM_EPS


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Return RMSNorm of ``x`` in fp32 math, in the dtype of ``x``.

    Example:
        >>> rms_norm(torch.ones(2, 4), torch.ones(4), 1e-5).shape
        torch.Size([2, 4])
    """
    xf = x.float()
    scale = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * scale * weight.float()).to(x.dtype)


def rope_pairs_in_place(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Rotate the interleaved pairs of ``x``, and keep each pair at its own lanes.

    This is the ``k_pe`` layout of a latent cache row: the prefill QKV kernel of a
    target layer and the decode kernels write and read the pairs at lanes
    ``(2i, 2i + 1)``. HuggingFace's ``apply_rotary_pos_emb_interleave``
    gives the same values with the two halves one after the other.

    Args:
        x: ``[T, rope]`` with the pairs at lanes ``(2i, 2i + 1)``.
        cos: ``[T, rope / 2]`` cosine of each pair's angle.
        sin: ``[T, rope / 2]`` sine.

    Example:
        >>> rope_pairs_in_place(torch.tensor([[1.0, 2.0]]), torch.zeros(1, 1), torch.ones(1, 1)).tolist()
        [[-2.0, 1.0]]
    """
    x1, x2 = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    return torch.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).reshape(
        x.shape
    )


def mtp_input(
    embeds: torch.Tensor,
    hidden: torch.Tensor,
    enorm_w: torch.Tensor,
    hnorm_w: torch.Tensor,
    eh_proj: torch.Tensor,
    eps: float = INPUT_NORM_EPS,
) -> torch.Tensor:
    """Return the input of the MTP decoder layer, ``eh_proj([enorm(e); hnorm(h)])``.

    The caller sets the embedding of a position-0 row to zero, as the reference does.

    Args:
        embeds: ``[T, H]`` embedding of the token after each position.
        hidden: ``[T, H]`` final hidden state of the target at each position.
        enorm_w: ``[H]`` enorm weight.
        hnorm_w: ``[H]`` hnorm weight.
        eh_proj: ``[2H, H]`` ``eh_proj.weight`` transposed.
        eps: RMSNorm epsilon.

    Example:
        >>> mtp_input(torch.ones(2, 4), torch.ones(2, 4), torch.ones(4), torch.ones(4),
        ...           torch.ones(8, 4)).shape
        torch.Size([2, 4])
    """
    e = rms_norm(embeds, enorm_w, eps)
    h = rms_norm(hidden, hnorm_w, eps)
    return torch.cat([e, h], dim=-1) @ eh_proj


def latent_rows(
    normed: torch.Tensor,
    wkv_a: torch.Tensor,
    kv_norm_w: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    kv_lora_rank: int,
    eps: float = LATENT_NORM_EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(k_pe, c_kv)`` of each position: the two parts of a latent cache row.

    Args:
        normed: ``[T, H]`` input_layernorm output of the MTP layer.
        wkv_a: ``[H, kv_lora_rank + rope]`` ``kv_a_proj_with_mqa.weight`` transposed.
        kv_norm_w: ``[kv_lora_rank]`` ``kv_a_layernorm.weight``.
        cos: ``[T, rope / 2]`` cosine at each true position.
        sin: ``[T, rope / 2]`` sine.
        kv_lora_rank: Latent width.
        eps: Epsilon of the latent norm.

    Example:
        >>> k_pe, c_kv = latent_rows(torch.ones(2, 8), torch.ones(8, 6), torch.ones(4),
        ...                          torch.ones(2, 1), torch.zeros(2, 1), 4)
        >>> k_pe.shape, c_kv.shape
        (torch.Size([2, 2]), torch.Size([2, 4]))
    """
    kv = normed @ wkv_a
    c_kv = rms_norm(kv[:, :kv_lora_rank], kv_norm_w, eps)
    # fp32 math on the bf16 tables that the QKV kernel of a target layer reads.
    k_pe = rope_pairs_in_place(
        kv[:, kv_lora_rank:].float(),
        cos.to(kv.dtype).float(),
        sin.to(kv.dtype).float(),
    ).to(kv.dtype)
    return k_pe, c_kv


class GlmMoeDsaMtpKv(nn.Module):
    """The weights of the MTP layer that its prompt rows need.

    The model calls `rows` after its final norm, on the tokens of this rank, and writes the
    rows into the two MTP caches in the formats of the target layers.
    """

    def __init__(self, config) -> None:  # noqa: ANN001  GlmMoeDsaConfig
        super().__init__()
        h, dtype = config.hidden_size, config.torch_dtype
        self.kv_lora_rank = config.kv_lora_rank
        self.rope_head_dim = config.qk_rope_head_dim
        self.enorm_weight = nn.Parameter(torch.ones(h, dtype=dtype))
        self.hnorm_weight = nn.Parameter(torch.ones(h, dtype=dtype))
        self.eh_proj = nn.Parameter(torch.empty(2 * h, h, dtype=dtype))
        self.input_norm_weight = nn.Parameter(torch.ones(h, dtype=dtype))
        self.wkv_a = nn.Parameter(
            torch.empty(h, self.kv_lora_rank + self.rope_head_dim, dtype=dtype)
        )
        self.kv_norm_weight = nn.Parameter(torch.ones(self.kv_lora_rank, dtype=dtype))
        self.idx_wk = nn.Parameter(torch.empty(h, config.index_head_dim, dtype=dtype))
        self.idx_k_norm_weight = nn.Parameter(
            torch.ones(config.index_head_dim, dtype=torch.float32)
        )
        self.idx_k_norm_bias = nn.Parameter(
            torch.zeros(config.index_head_dim, dtype=torch.float32)
        )

    def rows(
        self,
        embeds: torch.Tensor,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(k_pe, c_kv, keys)`` of this rank's positions.

        Args:
            embeds: ``[T, H]`` embedding of the token after each position.
            hidden: ``[T, H]`` final hidden state of the target.
            positions: ``[T]`` true position of each row.
            cos: ``[T, rope / 2]`` fp32 cosine at each position.
            sin: ``[T, rope / 2]`` fp32 sine.
        """
        first = (positions.reshape(-1, 1) != 0).to(embeds.dtype)
        x = mtp_input(
            embeds * first, hidden, self.enorm_weight, self.hnorm_weight, self.eh_proj
        )
        normed = rms_norm(x, self.input_norm_weight, INPUT_NORM_EPS)
        k_pe, c_kv = latent_rows(
            normed, self.wkv_a, self.kv_norm_weight, cos, sin, self.kv_lora_rank
        )
        keys = indexer_key_local(
            normed,
            self.idx_wk,
            self.idx_k_norm_weight,
            self.idx_k_norm_bias,
            cos,
            sin,
            self.rope_head_dim,
        )
        return k_pe, c_kv, keys


def indexer_key_local(
    hidden_local, wk, gain, bias, cos, sin, rope: int, eps: float = IDX_NORM_EPS
):
    """Recompute the DSA indexer's cache key for a batch of tokens, in torch.

    The fused indexer kernel projects its key internally and scatters it into a cache it allocates
    itself, so the key is not exposed. This repeats that arithmetic: a true LayerNorm of ``x @ wk``,
    then an INTERLEAVED RoPE on the LEADING ``rope`` dims. That is the convention measured against
    a GPU capture of the indexer keys (rotated 0.9997, unrotated 0.87 to 0.96), not the kernel's.

    It is a pure function, so a CPU test can check it without the device.

    Args:
        hidden_local: ``[t, H]`` normed hidden of the tokens.
        wk: ``[H, D]`` key projection.
        gain: ``[D]`` LayerNorm gain.
        bias: ``[D]`` LayerNorm bias.
        cos: ``[t, rope // 2]`` cosine at each token's true position.
        sin: ``[t, rope // 2]`` sine.
        rope: Leading dims to rotate.
        eps: LayerNorm epsilon.

    Returns:
        ``[t, D]`` float32 keys.
    """
    key = hidden_local.float() @ wk.float()
    centred = key - key.mean(dim=-1, keepdim=True)
    variance = (centred * centred).mean(dim=-1, keepdim=True)
    key = centred * torch.rsqrt(variance + eps) * gain.float() + bias.float()
    even, odd = key[:, 0:rope:2], key[:, 1:rope:2]
    c, s = cos.float(), sin.float()
    rotated = torch.stack([even * c - odd * s, even * s + odd * c], dim=-1)
    return torch.cat([rotated.reshape(key.shape[0], rope), key[:, rope:]], dim=-1)


def pack_latent_rows(k_pe, c_kv, bitcast, row_words: int):
    """Pack `[T, R]` k_pe and `[T, L]` latent into `[T, row_words]` bf16 words of the target layers' static latent cache
    row (`glm_kv_row`): bf16 `k_pe / s`, then the e4m3fn `latent / s` in the channel order of the decode kernel (dim
    `256a + 128b + p` at latent byte `256a + 2p + b`), then zero pad to `row_words`.

    Args:
        k_pe: `[T, R]` rotated RoPE keys.
        c_kv: `[T, L]` normed latent, in natural (model) order.
        bitcast: `(tensor, dtype) -> tensor` for a size-changing reinterpret (the NKI bitcast in
            a Neuron graph; `Tensor.view` on the host).
        row_words: bf16 words of one row (`glm_kv_row.STATIC_ROW_BYTES // 2`).
    """
    n, L = c_kv.shape
    s = glm_kv_row.STATIC_SCALE
    rope = (k_pe.float() / s).to(torch.bfloat16)
    latent = (c_kv.float() / s).clamp(-E4M3FN_MAX, E4M3FN_MAX)
    # A transpose of the last two axes of [a, b, p] (an fp8 gather has no lowering).
    latent = latent.reshape(n, L // 256, 2, 128).permute(0, 1, 3, 2).reshape(n, L)
    parts = [
        bitcast(rope.contiguous(), torch.uint8),
        bitcast(latent.to(torch.float8_e4m3fn).contiguous(), torch.uint8),
    ]
    pad = 2 * row_words - sum(part.shape[1] for part in parts)
    if pad < 0:
        raise ValueError(
            f"A latent row of {row_words} bf16 words cannot hold the packed row."
        )
    if pad:
        parts.append(torch.zeros((n, pad), dtype=torch.uint8, device=c_kv.device))
    return bitcast(torch.cat(parts, dim=1).contiguous(), torch.bfloat16)


def pack_keys(keys, bitcast, fp8: bool):
    """Return `[T, D]` indexer keys in the format of the key cache: bf16, or fp8 e4m3fn bytes.

    The fp8 keys have no scale: the keys are LayerNorm outputs, and the top-k ignores a scale.
    The decode kernel writes the same bytes.
    """
    if not fp8:
        return keys.to(torch.bfloat16)
    f8 = keys.float().clamp(-E4M3FN_MAX, E4M3FN_MAX).to(torch.float8_e4m3fn)
    return bitcast(f8.contiguous(), torch.uint8)


def mtp_hf_mappings(layer: int, prefix: str = "model.mtp_kv.") -> dict[str, str]:
    """Return ``{parameter name: checkpoint key}`` of `GlmMoeDsaMtpKv`.

    Args:
        layer: The checkpoint index of the MTP layer: ``num_hidden_layers``.
        prefix: The module path of the producer in the model.

    Example:
        >>> mtp_hf_mappings(78)["model.mtp_kv.eh_proj"]
        'model.layers.78.eh_proj.weight'
    """
    p = f"model.layers.{layer}."
    a = p + "self_attn."
    return {
        prefix + "enorm_weight": p + "enorm.weight",
        prefix + "hnorm_weight": p + "hnorm.weight",
        prefix + "eh_proj": p + "eh_proj.weight",
        prefix + "input_norm_weight": p + "input_layernorm.weight",
        prefix + "wkv_a": a + "kv_a_proj_with_mqa.weight",
        prefix + "kv_norm_weight": a + "kv_a_layernorm.weight",
        prefix + "idx_wk": a + "indexer.wk.weight",
        prefix + "idx_k_norm_weight": a + "indexer.k_norm.weight",
        prefix + "idx_k_norm_bias": a + "indexer.k_norm.bias",
    }
