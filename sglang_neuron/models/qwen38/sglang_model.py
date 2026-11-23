# SPDX-License-Identifier: Apache-2.0
"""Qwen3_5ForConditionalGeneration: the host SGLang builds for a Qwen3.8 dense hybrid checkpoint (models/host.py).

It serves the checkpoint's language model, text only, on the graphs in graph.py, which run the Qwen3.5 per-layer
kernels. The host declares the GDN state (``state_spec``), the pool classes SGLang builds (``state_pool_class``), the
step runtime that moves the state bank (``runtime_class``) and binds both pools to the graphs.
"""

from __future__ import annotations

import logging
from typing import Iterable, Tuple

import torch
from torch import nn

from sglang_neuron.models.host import ModelHost
from sglang_neuron.models.qwen38.config import Qwen38Dims, Qwen38Layout, text_config
from sglang_neuron.models.transformer.config import check_bf16_launch, check_no_expert_parallel

logger = logging.getLogger(__name__)


class Qwen3_5ForConditionalGeneration(ModelHost):
    name = "Qwen3.8"
    # the prefill attention is the model's own kernel, not the platform's segmented one
    segmented_prefill = False
    # a prompt's last piece runs the smallest KV segment tile that holds it (512, 1024, ... below the chunk): both
    # prefill kernels take any multiple of 512 tokens (gdn_prefill's 512-token blocks, attn_prefill's L % 512), so a
    # short prompt does not pay for a whole padded chunk
    prefill_ladder = True
    # the decode reads only its graph's block-table width (paged pool), so shorter-context decode graphs serve
    # short rows: a quarter, a half and the whole context
    decode_context_ladder = True

    def __init__(self, config, quant_config=None, prefix: str = "", **kwargs):
        import dataclasses

        from sglang.srt.runtime_context import get_server_args

        from sglang_neuron import sglang_tree
        from sglang_neuron.models.qwen38.state import Qwen38StatePool, state_spec

        import copy

        text = copy.copy(text_config(config))
        # the text config SGLang nests does not carry the checkpoint path; the loader reads it from here
        text._name_or_path = config._name_or_path
        self.qwen38_dims = Qwen38Dims.from_hf(config)
        tp = int(sglang_tree.parallel_config().tp_size)
        self.qwen38_layout = Qwen38Layout(tp_size=tp, dims=self.qwen38_dims)
        # declared before the graphs are built: SGLang sizes the pools from it. The GDN kernels advance contiguous
        # rows, so the pool carries a working bank: row 0 is the framework's scratch, batch row b uses bank row
        # b + 1, and the runtime moves slots to rows around each step (runtime.py)
        rows = int(getattr(get_server_args(), "max_running_requests", None) or 8) + 1
        from sglang_neuron.models.qwen38.state import temporal_dtype

        self.state_spec = dataclasses.replace(
            state_spec(self.qwen38_dims, self.qwen38_layout, temporal_dtype()), working_rows=rows)
        # the working-bank operations: this model's row programs (bank_ops: the publish an in-place indirect-DMA
        # write, whose cost follows the rows, not the pool), and compiled canonical-slot copies whose buckets match
        # this server's slot count. The platform's default buckets reach 256 slots, and a 256-slot copy of 48 layers
        # of fp32 GDN state reserves 1.5 GB of scratchpad, which a trn2 core at LNC1 cannot spare beside the weights
        # (the runtime reserves the max over every loaded program)
        from sglang_neuron.mem_cache.state_ops import CompiledStateOps, WorkingBankOps
        from sglang_neuron.models.qwen38.bank_ops import KernelBankOps

        bucket = 1 << max(3, (rows - 1).bit_length())
        ops = WorkingBankOps(KernelBankOps(), CompiledStateOps(buckets=(bucket,)))
        self.state_pool_class = Qwen38StatePool.bind(self.state_spec, ops, name="Qwen38StatePool_gdn")
        super().__init__(text, quant_config, prefix, **kwargs)
        logger.info("%s host: TP %d (LNC1), %d layers (%d GDN, %d attention), %d GDN value heads and %d query heads "
                    "a rank", self.name, tp, self.qwen38_dims.num_layers, len(self.qwen38_dims.gdn_layers),
                    len(self.qwen38_dims.attention_layers), self.qwen38_layout.gdn_v_heads,
                    self.qwen38_layout.q_heads)

    @property
    def runtime_class(self):
        from sglang_neuron.models.qwen38.runtime import Qwen38Runtime

        return Qwen38Runtime

    def check_launch(self, config) -> None:
        check_bf16_launch(config, self.name)
        check_no_expert_parallel(self.name)
        from sglang_neuron import sglang_tree

        dims = Qwen38Dims.from_hf(config)
        problems = dims.problems(int(sglang_tree.parallel_config().tp_size))
        if problems:
            raise ValueError(f"{self.name}: " + "; ".join(problems))

    def build_graph(self, phase: str, config) -> nn.Module:
        from sglang_neuron.models.qwen38.graph import Qwen38Graph

        return Qwen38Graph(self.qwen38_dims, self.qwen38_layout, phase)

    # pools
    def bind_kv_cache(self, pool) -> None:
        """Bind the graphs to SGLang's hybrid pool: the paged KV stacks of the full-attention layers and the GDN
        state pool's working bank."""
        from sglang_neuron.backend import executor_cache
        from sglang_neuron.models.qwen38.kv_pool import Qwen38SlotKVPool
        from sglang_neuron.models.qwen38.state import Qwen38StatePool

        full = getattr(pool, "full_kv_pool", pool)
        state = getattr(pool, "mamba_pool", None)
        if not isinstance(full, Qwen38SlotKVPool):
            raise RuntimeError(f"the full-attention KV pool is {type(full).__name__}, not the Qwen3.8 paged pool")
        if not isinstance(state, Qwen38StatePool):  # the bound subclass of it
            raise RuntimeError(f"the request table holds {type(state).__name__}, not the Qwen3.8 GDN state pool")
        for graph in getattr(self, "model_graphs", {"_": self.model_graph}).values():
            graph.bind_state_bank(state.tensors["work.conv0"], state.tensors["work.temporal"])
            graph.bind_kv_cache(full.k_stack, full.v_stack, full.slots, full.block_size)
        from sglang_neuron.models.qwen38.runtime import warm_bank_moves

        warm_bank_moves(state)
        executor_cache.invalidate()

    # weights
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Read this rank's shard of every parameter from the checkpoint, in the kernels' layouts (weights.py). The
        lazy iterator SGLang hands over is left unread: the loader reads the checkpoint directly."""
        from sglang_neuron.models.qwen38.load import load_into

        if getattr(self, "_weights_loaded", False):
            raise NotImplementedError(f"{self.name} on Neuron does not update its weights in place")
        from sglang_neuron.model_executor.runtime_init import neuron_device
        from sglang_neuron.models.host import place_rest_on_device, share_weights

        path, device = self.hf_config._name_or_path, neuron_device()
        (first_phase, first), *others = self.model_graphs.items()
        load_into(first, path, device)
        place_rest_on_device(self.name, first_phase, first, device)
        for phase, graph in others:
            share_weights(first, graph, name=self.name, phase=phase, source_phase=first_phase)
            place_rest_on_device(self.name, phase, graph, device)
        self._weights_loaded = True
        logger.info("%s weights loaded onto %s from %s", self.name, device, path)


EntryClass = [Qwen3_5ForConditionalGeneration]
