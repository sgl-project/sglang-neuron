# SPDX-License-Identifier: Apache-2.0
"""LlamaForCausalLM: the host SGLang builds for a Llama dense checkpoint (models/host.py), on the shared dense
transformer (models/transformer). Llama 3 is the dense core as it stands; Llama 3.1 and 3.2 add the ``llama3`` RoPE
rescaling, which the graph applies to its rotary tables (model.py)."""

from torch import nn

from sglang_neuron.models.host import ModelHost
from sglang_neuron.models.llama.config import NAME, llama_config, rope_of
from sglang_neuron.models.llama.model import LlamaGraph
from sglang_neuron.models.transformer.config import check_bf16_launch, check_no_expert_parallel


class LlamaForCausalLM(ModelHost):
    name = NAME
    decode_context_ladder = True

    def check_launch(self, config) -> None:
        check_bf16_launch(config, self.name)
        check_no_expert_parallel(self.name)
        llama_config(config)

    def build_graph(self, phase: str, config) -> nn.Module:
        _, scaling = rope_of(config)
        return LlamaGraph(llama_config(config), phase, scaling)


EntryClass = [LlamaForCausalLM]
