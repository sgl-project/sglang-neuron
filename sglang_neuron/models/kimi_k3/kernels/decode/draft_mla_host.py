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

"""Host weight layouts of the Kimi-K3 DSpark MLA drafter (the context stage, the layer, the tail, and the step).

The builders take the raw checkpoint tensors (``draft_mla_torch.open_safetensors`` views, torch
tensors, or numpy arrays) and return the per-rank kernel weights of one TP32 rank. They parallel
``mla_block_torch`` and reuse ``mla_block_rank_weights`` for the per-layer MLA weights. The load calls
``draft_mla_rank_weights`` once per rank to build every weight of the step.
"""

import numpy as np
import torch

from .draft_mla_torch import DraftMLADims, bf16_bits_to_torch
from .mla_block_torch import mla_block_cp_rank_weights, mla_block_rank_weights
from .mla_decode_torch import CACHE_WIDTH, HIDDEN, KV_LORA, ROPE, TP, _to_bf16_numpy, _to_torch, to_hT_order

_P_MAX = 128
_LOCAL_HEADS = 2
"""Drafter heads of one TP32 rank: 64 heads over 32 ranks."""
_DRAFT_HEADS = 64
"""Drafter attention heads. At CP32 one rank owns 2; at CP128 rank g owns head g (g < 64)."""
DRAFT_SHARD = 14336 // TP
"""Dense MLP intermediate columns of one rank: 14336 over 32 ranks is 448."""
V_SHARD = 163840 // TP
"""lm_head and Markov vocab rows of one rank: 5120."""
H_SHARD = HIDDEN // TP
"""Embedding hidden columns of one rank: 224."""
KV_A_SHARD = CACHE_WIDTH // TP
"""kv_a output columns of one rank: 576 over 32 ranks is 18."""

_LAYER_PREFIX = "layers.{}."
_ATTN_NAMES = {
    "q_a_proj": "self_attn.q_a_proj.weight",
    "q_a_layernorm": "self_attn.q_a_layernorm.weight",
    "q_b_proj": "self_attn.q_b_proj.weight",
    "kv_a_proj": "self_attn.kv_a_proj_with_mqa.weight",
    "kv_a_layernorm": "self_attn.kv_a_layernorm.weight",
    "kv_b_proj": "self_attn.kv_b_proj.weight",
    "o_proj": "self_attn.o_proj.weight",
}


def _convert(value) -> torch.Tensor:
    """Return a checkpoint value (BF16 bits, torch, or numpy) as an FP32 torch tensor."""
    if isinstance(value, np.ndarray) and value.dtype == np.uint16:
        return bf16_bits_to_torch(value)
    return _to_torch(value).float()


def _get(tensors: dict, name: str) -> torch.Tensor:
    """Return a whole checkpoint tensor as an FP32 torch tensor."""
    return _convert(tensors[name])


def _get_rows(tensors: dict, name: str, sl: slice) -> torch.Tensor:
    """Return rows ``sl`` of a checkpoint tensor, sliced on the memory map before the FP32 conversion.

    Slicing the memory-mapped view first keeps one rank's peak host memory to its shard, not the whole
    tensor (the full lm_head and embedding are 2.3 GiB each in FP32).
    """
    return _convert(tensors[name][sl])


def _get_cols(tensors: dict, name: str, sl: slice) -> torch.Tensor:
    """Return columns ``sl`` of a 2D checkpoint tensor, sliced before the FP32 conversion."""
    return _convert(tensors[name][:, sl])


def _layer_name(index: int, name: str) -> str:
    """Return the full checkpoint key of a layer tensor (attention, norm, or MLP short key)."""
    if name in _ATTN_NAMES:
        full = _ATTN_NAMES[name]
    elif name in ("input_layernorm", "post_attention_layernorm"):
        full = f"{name}.weight"
    else:
        full = f"mlp.{name}.weight"
    return _LAYER_PREFIX.format(index) + full


def _layer(tensors: dict, index: int, name: str) -> torch.Tensor:
    """Return the whole layer ``index`` checkpoint tensor ``name``."""
    return _get(tensors, _layer_name(index, name))


def _ht_tiled(weight_in_out: torch.Tensor) -> np.ndarray:
    """Tile an ``[in, out]`` weight as BF16 ``[128, in/128, out]``: element ``(p, j, n)`` is row ``j*128+p``."""
    rows, cols = weight_in_out.shape
    tiled = weight_in_out.reshape(rows // _P_MAX, _P_MAX, cols).permute(1, 0, 2).contiguous()
    return _to_bf16_numpy(tiled)


def _vector(weight: torch.Tensor) -> np.ndarray:
    """Return a 1D norm weight as a contiguous BF16 numpy vector (natural feature order)."""
    return _to_bf16_numpy(weight.reshape(-1))


def _vector_hT(weight: torch.Tensor) -> np.ndarray:
    """Return a 1D hidden norm weight in hT order for ``common.load_vector_hT`` (``rmsnorm_hT``)."""
    return _to_bf16_numpy(to_hT_order(weight.reshape(-1).numpy()))


def _context_kv_a_shard(tensors: dict, layer: int, rank: int, cp: int) -> np.ndarray:
    """Return the hT-tiled context kv_a projection shard of one rank: ``[128, 56, kv_a_shard]``.

    At CP32 the shard is the contiguous 18 output columns ``[18 t, 18 t + 18)``. At CP128 it is the
    target split: four kv latent features ``128 c + g`` (c = 0..3) and one k_pe feature ``g`` (g < 64, a
    zero column above), five columns. Both reconstruct the natural 576 cache row after the world gather.
    """
    name = _layer_name(layer, "kv_a_proj")
    if cp == TP:
        shard = CACHE_WIDTH // cp
        sel = _get_rows(tensors, name, slice(rank * shard, (rank + 1) * shard))
    else:
        latent = _get_rows(tensors, name, [c * cp + rank for c in range(KV_LORA // cp)])
        kpe = _get_rows(tensors, name, [KV_LORA + rank]) if rank < ROPE else torch.zeros((1, HIDDEN))
        sel = torch.cat([latent, kpe], dim=0)
    return _ht_tiled(sel.T)


def draft_context_rank_weights(tensors: dict, tp_rank: int, dims: DraftMLADims = DraftMLADims(), cp: int = TP) -> dict:
    """Build one rank's context-stage weights.

    Rank ``g`` owns ``context_proj`` output rows ``[H/cp g, H/cp g + H/cp)`` and its kv_a shard (see
    :func:`_context_kv_a_shard`), both all-gathered over the world. ``kv_a_norm`` (over the full 512
    latent) is replicated.

    Returns:
        ``context_proj`` [128, 280, H/cp] BF16 ``[in, out]`` hT-tiled, ``context_norm`` [7168] BF16,
        ``kv_a_proj`` [5, 128, 56, kv_a_shard] BF16 ``[in, out]`` hT-tiled, ``kv_a_norm`` [5, 512] BF16.
    """
    h_shard = HIDDEN // cp
    rows = slice(tp_rank * h_shard, (tp_rank + 1) * h_shard)
    context_proj = _get_rows(tensors, "context_proj.weight", rows).T
    kv_a_proj = np.stack([_context_kv_a_shard(tensors, l, tp_rank, cp) for l in range(dims.layers)])
    kv_a_norm = np.stack([_vector(_layer(tensors, l, "kv_a_layernorm")) for l in range(dims.layers)])
    return {
        "context_proj": _ht_tiled(context_proj),
        "context_norm": _vector_hT(_get(tensors, "context_norm.weight")),
        "kv_a_proj": kv_a_proj,
        "kv_a_norm": kv_a_norm,
    }


def draft_layer_rank_weights(
    tensors: dict, layer: int, tp_rank: int, dims: DraftMLADims = DraftMLADims(), cp: int = TP
) -> dict:
    """Build one rank's drafter layer weights: the MLA weights, the two norms, and the dense MLP shards.

    At CP32 rank ``t`` owns MLP intermediate columns ``[448 t, 448 t + 448)`` and MLA heads ``[2 t, 2 t + 2)``.
    At CP128 rank ``g`` owns intermediate columns ``[112 g, 112 g + 112)`` and head ``g`` (g < 64, zero pad above).

    Returns:
        the ``MLA_WEIGHT_NAMES`` tiles, ``input_layernorm`` and ``post_attention_layernorm`` [7168]
        BF16, ``dense_gate_up`` [128, 56, 2 * 14336/cp] BF16 ``[in, out]`` hT-tiled, ``dense_down``
        [14336/cp, 7168] BF16 ``[in, out]``.
    """
    draft_shard = 14336 // cp
    attn = {name: _layer(tensors, layer, name) for name in _ATTN_NAMES}
    if cp == TP:
        weights = mla_block_rank_weights(attn, tp_rank, _LOCAL_HEADS, output_gate=False)
    else:
        weights = mla_block_cp_rank_weights(attn, tp_rank, _DRAFT_HEADS, output_gate=False, cp=cp)
    shard = slice(tp_rank * draft_shard, (tp_rank + 1) * draft_shard)
    gate = _get_rows(tensors, _layer_name(layer, "gate_proj"), shard).T
    up = _get_rows(tensors, _layer_name(layer, "up_proj"), shard).T
    weights["input_layernorm"] = _vector_hT(_layer(tensors, layer, "input_layernorm"))
    weights["post_attention_layernorm"] = _vector_hT(_layer(tensors, layer, "post_attention_layernorm"))
    weights["dense_gate_up"] = _ht_tiled(torch.cat([gate, up], 1))
    weights["dense_down"] = _to_bf16_numpy(_get_cols(tensors, _layer_name(layer, "down_proj"), shard).T)
    return weights


def draft_tail_rank_weights(
    tensors: dict,
    tp_rank: int,
    dims: DraftMLADims = DraftMLADims(),
    vocab_size: int | None = None,
    cp: int = TP,
    markov_w1_full: bool = False,
) -> dict:
    """Build one rank's tail weights (``final_norm`` and the two Markov shards; NOT the lm_head).

    The drafter reuses the target lm_head, which the DSpark checkpoint does not hold, so the load shards it
    with :func:`draft_lm_head_rank_weights` and passes it to the entry directly. Rank ``g`` owns vocab
    rows ``[V_s g, V_s g + V_s)`` of both Markov tables (``V_s = vocab / cp``). ``markov_w1`` is sharded
    like ``markov_w2``: the owner of ``prev`` gathers its row and a tiny world all-reduce broadcasts it.

    Args:
        vocab_size: the vocabulary size (for the shard split). Default derives it from ``markov_w1``.
        markov_w1_full: return the full ``markov_w1`` [V, 256] (every rank holds the whole table) for the kernel's
            ``markov_w1_local``, in place of the vocab shard. Default False keeps the shard.

    Returns:
        ``final_norm`` [7168] BF16, ``markov_w1`` [V_s, 256] BF16 vocab shard, ``markov_w2``
        [128, 2, V_s] BF16 ``[in, out]`` hT-tiled.
    """
    vocab_size = tensors["markov_head.markov_w1.weight"].shape[0] if vocab_size is None else vocab_size
    v_shard = vocab_size // cp
    vocab = slice(tp_rank * v_shard, (tp_rank + 1) * v_shard)
    return {
        "final_norm": _vector_hT(_get(tensors, "final_norm.weight")),
        "markov_w1": _to_bf16_numpy(
            _get(tensors, "markov_head.markov_w1.weight")
            if markov_w1_full
            else _get_rows(tensors, "markov_head.markov_w1.weight", vocab)
        ),
        "markov_w2": _ht_tiled(_get_rows(tensors, "markov_head.markov_w2.weight", vocab).T),
    }


def draft_lm_head_rank_weights(lm_head, tp_rank: int, vocab_size: int | None = None, cp: int = TP) -> dict:
    """Shard the (target) lm_head for one rank: ``lm_head`` [7168, V_s] BF16 ``[in, out]`` (``V_s = vocab / cp``).

    ``lm_head`` is the target's ``lm_head.weight`` [V, 7168] tensor (the drafter reuses it). The load has it
    and calls this directly; the tests pass their synthetic lm_head.
    """
    vocab_size = (lm_head.shape[0] if hasattr(lm_head, "shape") else len(lm_head)) if vocab_size is None else vocab_size
    v_shard = vocab_size // cp
    vocab = slice(tp_rank * v_shard, (tp_rank + 1) * v_shard)
    return {"lm_head": _to_bf16_numpy(_convert(lm_head[vocab]).T)}


def draft_embed_rank_weights(
    tensors: dict, tp_rank: int, embed_name: str = "embed_tokens.weight", cp: int = TP
) -> dict:
    """Build one rank's embedding shard ``embed`` [V, H/cp] BF16: hidden columns ``[H/cp g, H/cp g + H/cp)``.

    The drafter reuses the target embedding; ``embed_name`` is the checkpoint tensor to shard.
    """
    h_shard = HIDDEN // cp
    cols = slice(tp_rank * h_shard, (tp_rank + 1) * h_shard)
    return {"embed": _to_bf16_numpy(_get_cols(tensors, embed_name, cols))}


def draft_mla_rank_weights(tensors: dict, tp_rank: int, dims: DraftMLADims = DraftMLADims(), cp: int = TP) -> dict:
    """Build every per-rank weight of ``kimi_k3_draft_mla_step`` for one rank of a CP``cp`` group.

    Args:
        tensors: the raw checkpoint tensors (``open_safetensors`` views, torch, or numpy).
        tp_rank: the rank ``g``.
        dims: drafter dimensions.
        cp: CP ranks (32 or 128).

    Returns:
        a dict with the context weights, the ``layers`` list of per-layer weights, the tail weights,
        and the embedding shard.
    """
    weights = draft_context_rank_weights(tensors, tp_rank, dims, cp)
    weights["layers"] = [draft_layer_rank_weights(tensors, l, tp_rank, dims, cp) for l in range(dims.layers)]
    weights.update(draft_tail_rank_weights(tensors, tp_rank, dims, cp=cp))
    if "lm_head.weight" in tensors:
        weights.update(draft_lm_head_rank_weights(tensors["lm_head.weight"], tp_rank, cp=cp))
    weights.update(draft_embed_rank_weights(tensors, tp_rank, cp=cp))
    return weights


_STEP_LAYER_KEYS = (
    "mla_in_proj",
    "mla_q_a_norm",
    "mla_q_b_proj",
    "mla_kv_a_norm",
    "mla_kv_b_proj",
    "mla_o_proj",
    "input_layernorm",
    "post_attention_layernorm",
    "dense_gate_up",
    "dense_down",
)
_STEP_LAYER_ARGS = (
    "mla_in_proj",
    "mla_q_a_norm",
    "mla_q_b_proj",
    "mla_kv_a_norm",
    "mla_kv_b_proj",
    "mla_o_proj",
    "input_norm",
    "post_norm",
    "dense_gate_up",
    "dense_down",
)


def draft_step_rank_weights(
    tensors: dict, tp_rank: int, dims: DraftMLADims = DraftMLADims(), cp: int = TP, markov_w1_full: bool = False
) -> dict:
    """Build the per-rank ``kimi_k3_draft_mla_step`` inputs, with the per-layer weights stacked ``[5, ...]``.

    Returns a flat dict keyed by the entry argument names: the context weights (``context_proj``,
    ``context_norm``, ``kv_a_proj`` [5,...], ``kv_a_norm`` [5, 512]), the stacked per-layer weights, the
    tail weights, and the embedding shard. ``cp`` selects the shard width (32 or 128). ``markov_w1_full`` gives the
    full ``markov_w1`` (the kernel's ``markov_w1_local``).
    """
    weights = dict(draft_context_rank_weights(tensors, tp_rank, dims, cp))
    per_layer = [draft_layer_rank_weights(tensors, layer, tp_rank, dims, cp) for layer in range(dims.layers)]
    for key, arg in zip(_STEP_LAYER_KEYS, _STEP_LAYER_ARGS, strict=True):
        weights[arg] = np.stack([layer[key] for layer in per_layer])
    weights.update(draft_tail_rank_weights(tensors, tp_rank, dims, cp=cp, markov_w1_full=markov_w1_full))
    if "lm_head.weight" in tensors:
        weights.update(draft_lm_head_rank_weights(tensors["lm_head.weight"], tp_rank, cp=cp))
    weights.update(draft_embed_rank_weights(tensors, tp_rank, cp=cp))
    return weights
