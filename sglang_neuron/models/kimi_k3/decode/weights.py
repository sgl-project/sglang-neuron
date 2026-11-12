# SPDX-License-Identifier: Apache-2.0
"""The per-rank weight stacks of the Kimi-K3 2-server decode (verify and drafter programs).

``load_decode_rank`` reads only the rows and columns that global rank ``g`` needs from the checkpoint and the drafter
checkpoint, and returns every weight of the two decode programs of that rank. Each layer stack is ``[L, ...]`` over
the layers of its kind in layer order: 69 KDA layers, 24 MLA layers, the dense layer 0, 92 MoE layers; ``norm_weights``
and ``attn_res_weights`` cover all 93 layers. The host builders of the vendored decode kernel tree give every layout
(``weights_attn``, ``weights_ffn``, ``weights_draft``).

Rank coordinates (``weights_source.DecodeRank``): TP rank ``t = g % 32``, replica ``g // 32``, expert group
``g % 128``, expert part ``g // 128``. Ranks ``g`` and ``g + 32 j`` hold the same tensors except ``expert_*`` and
``dense_*``.

Tensors of one rank (``G = 56`` hidden tiles of 128, ``V_t = 5120`` vocab ids of the TP rank):

=================================  ==============================  =======  ==============================
name                               shape                           dtype    content
=================================  ==============================  =======  ==============================
``norm_weights``                   ``[93, 2, 7168]``               bf16     input, post-attention norm (hT)
``attn_res_weights``               ``[93, 2, 7168]``               fp32     AttnRes score weights (hT)
``kda_w_in``                       ``[69, 14, 128, 6688]``         uint8    MXFP8 q|k|v|g|f_a|b, hT_quad
``kda_w_in_scale``                 ``[69, 128, 2, 1672]``          uint8    E8M0 scales
``kda_w_out``                      ``[69, 1, 128, 28672]``         uint8    MXFP8 o_proj, padded lanes
``kda_w_out_scale``                ``[69, 128, 1, 7168]``          uint8    E8M0 scales
``kda_w_small``                    ``[69, 128, 420]``              bf16     f_b, bf16 conv taps
``kda_params``                     ``[69, 128, 43]``               fp32     dt_bias, A_log, o_norm, fp32 taps
``mla_in_proj``                    ``[24, 14, 128, 1800]``         uint8    MXFP8 q_a|kv_a|g, hT_quad
``mla_in_proj_scale``              ``[24, 128, 2, 450]``           uint8    E8M0 scales
``mla_q_a_norm``                   ``[24, 1536]``                  bf16     hT order
``mla_q_b_proj``                   ``[24, 3, 128, 2304]``          uint8    MXFP8, hT_quad
``mla_q_b_proj_scale``             ``[24, 128, 1, 576]``           uint8    E8M0 scales
``mla_kv_a_norm``                  ``[24, 512]``                   bf16
``mla_kv_b_proj``                  ``[24, 128, 3584]``             uint8    MXFP8 absorb quadrants | value
``mla_kv_b_proj_scale``            ``[24, 16, 896]``               uint8    compact E8M0 scales
``mla_o_proj``                     ``[24, 1, 128, 28672]``         uint8    MXFP8, padded lanes
``mla_o_proj_scale``               ``[24, 16, 7168]``              uint8    compact E8M0 scales
``dense_gate_up``                  ``[1, 128, 56, 264]``           bf16     world shard, hT-tiled
``dense_down``                     ``[1, 132, 7168]``              bf16     world shard
``moe_in_proj``                    ``[92, 14, 128, 2048]``         uint8    MXFP8 latent block|shared g|u
``moe_in_proj_scale``              ``[92, 128, 2, 512]``           uint8    E8M0 scales
``moe_out_proj``                   ``[92, 304, 7168]``             bf16     column latent up|shared down
``moe_vectors``                    ``[92, 1, 4620]``               fp32     DP vectors, router biases, norm
``moe_router_weight``              ``[92, 128, 56, 28]``           fp16     router rows of the TP rank (hT)
``expert_gate_up_weights``         ``[92, 7, 128, 2, 7, 1536]``    uint16   MXFP4 static loop
``expert_gate_up_weights_scale``   ``[92, 7, 16, 2, 7, 1536]``     uint8
``expert_down_weights``            ``[92, 7, 128, 3, 3584]``       uint16
``expert_down_weights_scale``      ``[92, 7, 16, 3, 3584]``        uint8
``final_attn_res_weight``          ``[7168]``                      fp32     output AttnRes score (hT)
``final_norm_weight``              ``[7168]``                      bf16     final norm (hT)
``lm_head_weight``                 ``[7168, 5120]``                bf16     lm_head ``[H, V_t]``
``embed_tokens``                   ``[5120, 7168]``                bf16     embedding rows of the TP rank
``draft.<name>``                   ``weights_draft``               bf16     the drafter step weights
=================================  ==============================  =======  ==============================

A rank holds 8,219,445,056 bytes (7.66 GiB), of which the routed experts are 5,650,219,008.

``kda_params`` is the plain stack. The decode programs read it in the wide form ``[69, 128, 43 + 3]``: the trace
takes a zero buffer of the wide shape, and after the trace a device init program writes the plain columns and
``exp(A_log)`` of the 3 local heads into it (its exp is the device's).
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor

from sglang_neuron.models.kimi_k3.decode import weights_attn, weights_draft, weights_ffn
from sglang_neuron.models.kimi_k3.decode.weights_host import host_modules, put_layer, to_hT_order
from sglang_neuron.models.kimi_k3.decode.weights_source import DecodeRank, RankCheckpoint

FORMS = ("k3", "k7b2")
"""The served verify forms. Both take the same weights: the forms differ in the kernel arguments only."""

KINDS = ("vectors", "kda", "mla", "dense", "moe", "experts", "tail", "draft")
"""The stack kinds of ``load_decode_rank``, in output order."""

SHARED_KINDS = ("vectors", "kda", "mla", "moe", "tail", "draft")
"""The kinds whose tensors depend on the TP rank only: the DP replicas of one TP rank hold the same bytes."""

DRAFT_PREFIX = "draft."
TAIL_NAMES = ("final_attn_res_weight", "final_norm_weight", "lm_head_weight")
EMBED_NAME = "embed_tokens"


def _check_geometry(g: Any, rank: DecodeRank) -> None:
    problems = []
    if g.kda_num_heads % rank.tp or g.num_heads % rank.tp:
        problems.append(f"{g.kda_num_heads} KDA and {g.num_heads} MLA heads over TP {rank.tp}")
    if g.vocab_size % rank.tp:
        problems.append(f"the vocab {g.vocab_size} over TP {rank.tp}")
    if g.intermediate_size % rank.world:
        problems.append(f"the dense intermediate {g.intermediate_size} over the world {rank.world}")
    if g.first_k_dense_replace != 1:
        problems.append(f"first_k_dense_replace {g.first_k_dense_replace} (the decode has one dense layer, 0)")
    if problems:
        raise ValueError("The Kimi-K3 decode stacks cannot split " + "; ".join(problems) + ".")


def _tail(ckpt: RankCheckpoint, g: Any, rank: DecodeRank) -> dict[str, Tensor]:
    """The decode tail and the embedding of the TP rank: the output AttnRes score weight and the final norm in hT
    order, the lm_head shard ``[H, V_t]`` and the embedding shard ``[V_t, H]``."""
    bf16 = torch.bfloat16
    vocab = g.vocab_size // rank.tp
    v0, v1 = rank.tp_rank * vocab, (rank.tp_rank + 1) * vocab
    score = (ckpt.full("model.output_attn_res_norm.weight", bf16).float()
             * ckpt.full("model.output_attn_res_proj.weight", bf16).reshape(-1).float())
    return {
        "final_attn_res_weight": to_hT_order(score),
        "final_norm_weight": to_hT_order(ckpt.full("model.norm.weight", bf16)),
        "lm_head_weight": ckpt.rows("lm_head.weight", v0, v1, bf16).t().contiguous(),
        EMBED_NAME: ckpt.rows("model.embed_tokens.weight", v0, v1, bf16),
    }


def load_decode_rank(checkpoint_dir: str, drafter_dir: str | None, geometry: Any, rank: int, *, world: int = 256,
                     tp: int = 32, dp: int = 8, form: str = "k3", layers: Iterable[int] | None = None,
                     kinds: Iterable[str] | None = None, device: torch.device | str | None = None,
                     timings: dict | None = None) -> dict[str, Tensor]:
    """The weight stacks of global rank ``rank`` of the 2-server decode (module docstring).

    Args:
        checkpoint_dir: The checkpoint directory (safetensors shards and ``model.safetensors.index.json``).
        drafter_dir: The MLA DSpark drafter checkpoint directory; None loads no drafter weights.
        geometry: The model geometry (``prefill.geometry.read_geometry``).
        rank: The global rank, ``0 .. world - 1``.
        world, tp, dp: The decode world: ``world = tp * dp`` ranks, TP ``tp`` replicas.
        form: The served verify form (``FORMS``).
        layers: Only these layers go into the layer stacks (a test or a check of a few layers); None takes all.
        kinds: Only these stack kinds (``KINDS``); None takes all. A loader that converts the ``SHARED_KINDS`` once
            for each TP rank asks the other ranks for ``("dense", "experts")`` only.
        device: The device of the tensors. The stacks are allocated there and filled one layer at a time, so the host
            holds one converted layer at a time. None keeps them on the host.
        timings: When given, gets the seconds spent on each kind.

    Returns:
        ``{name: tensor}`` in ``KINDS`` order; the drafter weights as ``draft.<name>``.
    """
    if form not in FORMS:
        raise ValueError(f"The served verify form {form!r} is unknown. Use one of {FORMS}.")
    coords = DecodeRank(rank, world, tp, dp)
    g = geometry
    _check_geometry(g, coords)
    wanted = KINDS if kinds is None else tuple(k for k in KINDS if k in set(kinds))
    unknown = set(kinds or ()) - set(KINDS)
    if unknown:
        raise ValueError(f"The stack kinds {sorted(unknown)} are unknown. Use some of {KINDS}.")
    selected = list(range(g.num_layers)) if layers is None else sorted({int(i) for i in layers})
    by_kind = {
        "vectors": selected,
        "kda": [i for i in selected if g.is_kda(i)],
        "mla": [i for i in selected if g.is_mla(i)],
        "dense": [i for i in selected if not g.is_moe(i)],
        "moe": [i for i in selected if g.is_moe(i)],
        "experts": [i for i in selected if g.is_moe(i)],
    }
    host = host_modules()
    convert = {
        "vectors": lambda ckpt, i: weights_attn.vector_layer(ckpt, i),
        "kda": lambda ckpt, i: weights_attn.kda_layer(ckpt, i, coords, g.kda_num_heads, g.kda_head_dim, host.kda),
        "mla": lambda ckpt, i: weights_attn.mla_layer(ckpt, i, coords, host.mla),
        "dense": lambda ckpt, i: weights_ffn.dense_layer(ckpt, i, coords, host.dense),
        "moe": lambda ckpt, i: weights_ffn.moe_layer(ckpt, i, coords, host.moe_dp),
        "experts": lambda ckpt, i: weights_ffn.expert_layer(ckpt, i, coords, host.routed),
    }
    out: dict[str, Tensor] = {}
    ckpt = RankCheckpoint(checkpoint_dir)
    try:
        with torch.device("cpu"):
            for kind in wanted:
                start = time.perf_counter()
                if kind == "tail":
                    out.update({name: value.to(device) for name, value in _tail(ckpt, g, coords).items()})
                elif kind == "draft":
                    if drafter_dir is not None:
                        draft = weights_draft.load_draft_rank(drafter_dir, coords.tp_rank, host, tp)
                        out.update({f"{DRAFT_PREFIX}{name}": value.to(device) for name, value in draft.items()})
                else:
                    stacks: dict[str, Tensor] = {}
                    ids = by_kind[kind]
                    for index, layer in enumerate(ids):
                        put_layer(stacks, convert[kind](ckpt, layer), index, len(ids), device)
                    out.update(stacks)
                if timings is not None:
                    timings[kind] = time.perf_counter() - start
    finally:
        ckpt.close()
    return out
