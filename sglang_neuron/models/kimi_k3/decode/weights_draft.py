# SPDX-License-Identifier: Apache-2.0
"""The per-rank weights of the Kimi-K3 MLA DSpark drafter step (5 dense MLA layers, context projection, Markov head).

The host builders of the vendored decode tree (``draft_mla_host``) build the CP32 layouts of attention TP
rank ``t`` from zero-copy views of the drafter checkpoint (``draft_mla_torch.open_safetensors``). The drafter uses
the target embedding and the target lm_head shard, so it holds no copy of them. Each rank holds the whole Markov
``markov_w1`` table ``[V, 256]``: a tail row gathers ``markov_w1[prev]`` locally.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sglang_neuron.models.kimi_k3.decode.weights_host import to_torch

DRAFT_CHECKPOINT_FILE = "model.safetensors"
DRAFT_TARGET_LAYERS = (2, 23, 47, 71, 89)
"""The target layers whose output feeds the drafter (0-indexed)."""

DRAFT_STACK_NAMES = ("context_proj", "context_norm", "kv_a_proj", "kv_a_norm", "mla_in_proj", "mla_q_a_norm",
                     "mla_q_b_proj", "mla_kv_a_norm", "mla_kv_b_proj", "mla_o_proj", "input_norm", "post_norm",
                     "dense_gate_up", "dense_down", "final_norm", "markov_w1", "markov_w2")
"""The drafter step weights, by drafter kernel argument."""

_LAYER_STACKS = {
    "mla_in_proj": "mla_in_proj",
    "mla_q_a_norm": "mla_q_a_norm",
    "mla_q_b_proj": "mla_q_b_proj",
    "mla_kv_a_norm": "mla_kv_a_norm",
    "mla_kv_b_proj": "mla_kv_b_proj",
    "mla_o_proj": "mla_o_proj",
    "input_layernorm": "input_norm",
    "post_attention_layernorm": "post_norm",
    "dense_gate_up": "dense_gate_up",
    "dense_down": "dense_down",
}
"""builder key of a drafter layer -> drafter step argument (stacked over the layers)."""

_EXPECTED = {
    "model_type": "k3_dspark",
    "hidden_size": 7168,
    "intermediate_size": 14336,
    "num_hidden_layers": 5,
    "num_attention_heads": 64,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "vocab_size": 163840,
    "draft_vocab_size": 163840,
    "num_target_layers": 5,
    "target_hidden_size": 7168,
    "mask_token_id": 163837,
    "markov_rank": 256,
    "markov_head_type": "vanilla",
}
_EXPECTED_ROPE = {
    "rope_type": "yarn",
    "factor": 32.0,
    "original_max_position_embeddings": 32768,
    "rope_theta": 50000.0,
    "beta_fast": 32,
    "beta_slow": 1,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
}
_MUST_BE_OFF = ("mla_use_nope", "mla_use_output_gate", "mla_use_qk_norm", "dspark_bonus_anchor")


def read_draft_config(draft_dir: str | Path) -> dict:
    """The drafter ``config.json``, checked against the MLA DSpark shape the drafter kernels implement.

    Raises:
        ValueError: The checkpoint differs; the message names every differing field.
    """
    config = json.loads((Path(draft_dir) / "config.json").read_text())
    wrong = {key: config.get(key) for key, value in _EXPECTED.items() if config.get(key) != value}
    rope = config.get("rope_parameters") or config.get("rope_scaling") or {}
    wrong.update({f"rope.{key}": rope.get(key) for key, value in _EXPECTED_ROPE.items() if rope.get(key) != value})
    wrong.update({key: True for key in _MUST_BE_OFF if config.get(key)})
    if tuple(config.get("target_layer_ids") or ()) != DRAFT_TARGET_LAYERS:
        wrong["target_layer_ids"] = config.get("target_layer_ids")
    if not config.get("sample_from_anchor", True):
        wrong["sample_from_anchor"] = False
    if config.get("is_causal") or config.get("dflash_config") or config.get("layer_types"):
        wrong["attention"] = "causal or per-layer attention config"
    if config.get("fc_norm"):
        wrong["fc_norm"] = True
    if wrong:
        raise ValueError(f"The Kimi-K3 MLA drafter kernels need the K3 DSpark shape; the checkpoint differs: {wrong}")
    return config


def draft_dims(config: dict, draft_torch: Any) -> Any:
    """The ``DraftMLADims`` of a checked drafter config."""
    rope = config.get("rope_parameters") or config.get("rope_scaling") or {}
    return draft_torch.DraftMLADims(
        hidden=config["hidden_size"],
        layers=config["num_hidden_layers"],
        heads=config["num_attention_heads"],
        q_lora=config["q_lora_rank"],
        kv_lora=config["kv_lora_rank"],
        nope=config["qk_nope_head_dim"],
        rope=config["qk_rope_head_dim"],
        v_head=config["v_head_dim"],
        intermediate=config["intermediate_size"],
        vocab=config["vocab_size"],
        markov_rank=config["markov_rank"],
        aux_layers=config["num_target_layers"],
        eps=config.get("rms_norm_eps", 1e-5),
        rope_theta=float(rope["rope_theta"]),
        yarn_factor=float(rope["factor"]),
        yarn_original=int(rope["original_max_position_embeddings"]),
        yarn_beta_fast=float(rope["beta_fast"]),
        yarn_beta_slow=float(rope["beta_slow"]),
        yarn_mscale=float(rope["mscale"]),
        yarn_mscale_all_dim=float(rope["mscale_all_dim"]),
        mask_token=config["mask_token_id"],
    )


def load_draft_rank(draft_dir: str | Path, tp_rank: int, host: Any, tp: int = 32) -> dict[str, torch.Tensor]:
    """The drafter step weights of attention TP rank ``tp_rank`` (``DRAFT_STACK_NAMES``), BF16.

    Args:
        host: The host builder modules (``weights_host.host_modules``).
    """
    if tp != 32:
        raise ValueError(f"The drafter shards of the 2-server decode are CP32; the replica is TP {tp}.")
    dims = draft_dims(read_draft_config(draft_dir), host.draft_torch)
    with torch.device("cpu"):
        tensors = host.draft_torch.open_safetensors(Path(draft_dir) / DRAFT_CHECKPOINT_FILE)
        weights = dict(host.draft_host.draft_context_rank_weights(tensors, tp_rank, dims))
        layers = [host.draft_host.draft_layer_rank_weights(tensors, layer, tp_rank, dims)
                  for layer in range(dims.layers)]
        for key, name in _LAYER_STACKS.items():
            weights[name] = np.stack([layer[key] for layer in layers])
        weights.update(host.draft_host.draft_tail_rank_weights(tensors, tp_rank, dims, markov_w1_full=True))
    out = {}
    for name in DRAFT_STACK_NAMES:
        value = weights[name]
        if np.asarray(value).dtype.name != "bfloat16":
            raise TypeError(f"The drafter host builder gave {name} as {np.asarray(value).dtype}; the drafter "
                            "weights are BF16.")
        out[name] = to_torch(value)
    return out
