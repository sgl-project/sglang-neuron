# SPDX-License-Identifier: Apache-2.0
"""Load one prefill stage's weights through the prefill causal-LM loaders (layout/prefill/weight_loaders.py), unedited.

The loaders are written for a whole causal LM with the embedding, final norm and lm_head. A stage holds only what its
layout needs, so this module wraps them instead of keeping per-stage copies, and every stage runs the same loader code:

* ``attach_stage_loaders`` gives the tree a placeholder for each piece the stage does not hold, runs
  ``attach_causal_lm_loaders``, and removes the placeholders again;
* ``stage_hf_mappings`` drops those pieces' keys from ``causal_lm_hf_mappings`` and checks that the mapping and the
  tree's parameters cover each other exactly, each parameter with its loader.

So every tensor a stage loads is the one the loader makes for that parameter.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

from sglang_neuron.models.glm52.layout import quark_compat
from sglang_neuron.models.glm52.layout.prefill.weight_loaders import attach_causal_lm_loaders, causal_lm_hf_mappings
from sglang_neuron.models.glm52.prefill.expert_placements import check_placement
from sglang_neuron.utils.checkpoints import CheckpointLoadResult, SafetensorsCheckpoint
from sglang_neuron.utils.weight_loader import get_weight_loader

# The keys causal_lm_hf_mappings gives the pieces besides the layers, by the layout fact that holds each.
EMBEDDING_KEY = "model.embed_tokens.weight"
NORM_KEY = "model.norm_weight"
LM_HEAD_KEY = "lm_head.weight"


@contextlib.contextmanager
def _placeholders(weights):
    """Stand-ins for the embedding, the lm_head and the final norm a stage does not hold, for the block's duration.

    attach_causal_lm_loaders sets a loader on each of the three unconditionally (it reads ``vocab_size_per_rank`` /
    ``out_features_per_rank`` and ``tp_size`` for the vocabulary shards). The stand-ins are plain objects, not
    parameters, so they never enter the tree's parameters, and the block puts ``None`` back.
    """
    model, absent = weights.model, []
    if model.embed_tokens is None:
        model.embed_tokens = SimpleNamespace(weight=SimpleNamespace(), vocab_size_per_rank=1, tp_size=1)
        absent.append((model, "embed_tokens"))
    if weights.lm_head is None:
        weights.lm_head = SimpleNamespace(weight=SimpleNamespace(), out_features_per_rank=1, tp_size=1)
        absent.append((weights, "lm_head"))
    if model.norm_weight is None:
        model.norm_weight = SimpleNamespace()
        absent.append((model, "norm_weight"))
    try:
        yield
    finally:
        for owner, name in absent:
            setattr(owner, name, None)


def attach_stage_loaders(weights) -> None:
    """Attach its loader to every parameter of the stage, through attach_causal_lm_loaders run as it is.

    The per-layer loaders read the checkpoint layer ``config.first_layer_index + i`` (the MoE placement, by checkpoint
    layer), so a stage's layer i loads checkpoint layer ``start_layer + i``.
    """
    with _placeholders(weights):
        attach_causal_lm_loaders(weights)


def _moe_ep_degree(weights) -> int | None:
    """The EP degree causal_lm_hf_mappings lists the expert slots for: that of the first MoE layer, which every MoE
    layer of the stage shares (the options' ep_degree); None for a stage without one."""
    moe0 = next((layer.mlp for layer in weights.model.layers if layer.is_moe), None)
    return moe0.ep_degree if moe0 is not None else None


def stage_hf_mappings(weights) -> dict:
    """``{parameter name: checkpoint key(s)}`` of the stage, without the keys of pieces it does not hold.

    Run after attach_stage_loaders. Raises unless the mapping and the parameters cover each other exactly and every
    parameter has a loader; otherwise load_sharded_pipelined would load an unmapped parameter from a key of its own
    name, or fail only at the end of a long load.
    """
    layout = weights.layout
    mappings = causal_lm_hf_mappings(weights.config, mtp=bool(layout.mtp_rows), ep_degree=_moe_ep_degree(weights))
    dropped = []
    if not layout.holds_embedding:
        dropped.append(EMBEDDING_KEY)
    if not layout.samples:
        dropped += [LM_HEAD_KEY, NORM_KEY]
    for key in dropped:
        del mappings[key]
    params = dict(weights.named_parameters())
    unmapped, unheld = sorted(set(params) - set(mappings)), sorted(set(mappings) - set(params))
    if unmapped or unheld:
        raise RuntimeError(f"The stage's checkpoint mapping and its parameters differ: parameters without a key "
                           f"{unmapped[:8]}, keys without a parameter {unheld[:8]}.")
    bare = [name for name, p in params.items() if get_weight_loader(p).transform is None]
    if bare:
        raise RuntimeError(f"Parameters without a loader (run attach_stage_loaders first): {bare[:8]}")
    return mappings


def _check_load(weights, path: str, rank: int, world: int | None) -> int:
    """Refuse a load this stack cannot make, before reading any byte; returns the load's world (the stage's CP)."""
    if weights.model._stack_names:
        raise RuntimeError("The stage's weights are already loaded (stacked): build a new tree to load again.")
    cp = int(weights.layout.cp)
    world = cp if world is None else int(world)
    if world != cp:
        raise ValueError(f"The stage's weights shard over its {cp} CP ranks, not {world}.")
    if not 0 <= rank < cp:
        raise ValueError(f"rank {rank} is not a rank of the stage's {cp}.")
    # An AttnFP8 checkpoint stores the attention and dense weights as FP8 / MXFP4, and the stage loaders have no bf16
    # upcast for them (no role has one): those bytes would load as bf16 as they are.
    if quark_compat.checkpoint_format(path) == quark_compat.QUARK_ATTN_FP8:
        raise NotImplementedError(f"{path} is a Quark AttnFP8 checkpoint; the GLM-5.2 prefill loads the MXFP4 export "
                                  "(bf16 attention and dense MLP, MXFP4 experts), amd/GLM-5.2-MXFP4.")
    if any(layer.is_moe for layer in weights.model.layers):
        check_placement(pp_size=weights.layout.pp_size)
    return world


def _stack_tree(weights) -> dict:
    """Attach the loaders, map the checkpoint and stack the per-layer weights: the tree a load fills."""
    attach_stage_loaders(weights)
    mappings = stage_hf_mappings(weights)
    weights.model.stack_weight_params(mappings)
    return mappings


def load_stage_weights(weights, path: str, device, *, rank: int, world: int | None = None) -> CheckpointLoadResult:
    """Load stage TP rank ``rank``'s tensors of the checkpoint directory ``path`` onto ``device``, straight into the
    kernel stacks, then make the stacks buffers.

    ``world`` is the stage's CP degree (``layout.cp``, the default): each loader shards over it, and the page-cache
    read splits the files over it, so the stage's ranks read the checkpoint once between them. Every parameter must
    get its tensor: a missing checkpoint key raises, and ``load_state_dict(strict=True)`` refuses a shape that differs.
    A tree loads once: the load turns its per-layer weights into the stacks.
    """
    world = _check_load(weights, path, rank, world)
    mappings = _stack_tree(weights)
    result = SafetensorsCheckpoint(path).load_sharded_pipelined(rank, world, weights, mappings, device)
    _assign(weights, result.state_dict)
    return result


def _assign(weights, state: dict) -> None:
    """Give every parameter its device tensor, freeze the tree and make the stacks buffers.

    The MTP module (layout/prefill/mtp_kv) builds trainable parameters, nn.Parameter's default; nothing here trains,
    and a trainable parameter in a compiled graph makes AOT autograd trace the backward too, so the tree is frozen.
    """
    weights.load_state_dict(state, strict=True, assign=True)
    weights.requires_grad_(False)
    weights.model.stacks_to_buffers()
