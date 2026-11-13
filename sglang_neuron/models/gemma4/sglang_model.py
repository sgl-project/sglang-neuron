# SPDX-License-Identifier: Apache-2.0
"""Gemma4ForConditionalGeneration: the host SGLang builds for a Gemma 4 dense checkpoint (models/host.py). It serves
the checkpoint's language model, text only, on the graphs in graph.py."""

import copy
import json
import os

from torch import nn

from sglang_neuron.models.gemma4.config import gemma4_config, text_config
from sglang_neuron.models.gemma4.graph import Gemma4Graph
from sglang_neuron.models.host import ModelHost
from sglang_neuron.models.transformer.config import check_bf16_launch, check_no_expert_parallel


def _checkpoint_prefix(path: str) -> str:
    """The checkpoint's prefix of the language model's tensors: ``model.language_model.`` in the multimodal
    checkpoint, ``model.`` in a text-only one."""
    index = os.path.join(path or "", "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index) as f:
            names = json.load(f).get("weight_map", {})
        if any(n.startswith("model.language_model.") for n in names):
            return "model.language_model."
        return "model."
    try:
        from safetensors import safe_open
        single = os.path.join(path or "", "model.safetensors")
        with safe_open(single, "pt") as f:
            return "model.language_model." if any(k.startswith("model.language_model.") for k in f.keys()) \
                else "model."
    except (OSError, ImportError):
        return "model.language_model."


def _declare_kv_layout(text, cfg) -> None:
    """Make SGLang's uniform KV pool hold every layer's rows. SGLang sizes the pool from its own view of the config:
    head_dim 512 and 4 KV heads for the multimodal checkpoint (its Gemma 4 normalization takes the global layers'
    shape), or head_dim 256 and 16 KV heads for a config without per-layer shapes. A token needs, per rank, the larger
    of the sliding layers' 16 / TP x 256 and the global layers' 4 / TP x 512 elements; the layout states how many
    pool heads of SGLang's head_dim give that (the graph views each layer's tensors with its own row width)."""
    from sglang_neuron import sglang_tree
    from sglang_neuron.model_executor import kv_layout

    tp = int(sglang_tree.parallel_config().tp_size)
    need = max(max(1, cfg.num_key_value_heads // tp) * cfg.head_dim,
               max(1, cfg.num_global_key_value_heads // tp) * cfg.global_head_dim)
    pool_head_dim = int(text.head_dim)
    if need % pool_head_dim:
        raise ValueError(f"Gemma 4: a KV row of {need} elements is not a whole number of SGLang's {pool_head_dim}-wide "
                         "pool heads")
    kv_layout.declare(kv_layout.KVLayout(kv_heads_per_rank=need // pool_head_dim))


class Gemma4ForConditionalGeneration(ModelHost):
    name = "Gemma4"
    # the prefill attends through its own torch attention, not the segmented kernel's tiles
    segmented_prefill = False
    prefill_ladder = False
    # the decode reads only its graph's table width (graph.Attention._read), so shorter-context decode graphs serve
    # short rows
    decode_context_ladder = True

    def __init__(self, config, quant_config=None, prefix: str = "", **kwargs):
        text = copy.copy(text_config(config))
        text._name_or_path = config._name_or_path
        _declare_kv_layout(text, gemma4_config(text))
        # the runtime reads head_dim and num_key_value_heads as model-wide values (the contract digest, logs), which
        # transformers refuses on a config that still carries per-layer shapes; the graphs read the per-layer shape
        # from config.json (config.gemma4_config)
        text.allow_global_per_layer_attribute_access = True
        super().__init__(text, quant_config, prefix, **kwargs)

    def check_launch(self, config) -> None:
        check_bf16_launch(config, self.name)
        check_no_expert_parallel(self.name)
        gemma4_config(config)

    def build_graph(self, phase: str, config) -> nn.Module:
        return Gemma4Graph(gemma4_config(config), phase, _checkpoint_prefix(config._name_or_path))


EntryClass = [Gemma4ForConditionalGeneration]
