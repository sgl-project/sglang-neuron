# SPDX-License-Identifier: Apache-2.0
"""HYV4FlashForConditionalGeneration: the host SGLang builds for a Hy4-Flash checkpoint (models/host.py), its graphs
model.Hy4FlashGraph.

Served on attention TP (no DP attention) with the 512 routed experts over EP = TP; the target layout is TP32 / EP32 on
half a trn2.48xlarge (32 logical cores): one sparse-layer head, two sliding-layer heads and 16 experts a rank."""

import os

from torch import nn

from sglang_neuron import sglang_tree
from sglang_neuron.models.host import ModelHost
from sglang_neuron.models.hy_v4_flash.config import NAME, hy4_flash_config
from sglang_neuron.models.hy_v4_flash.model import Hy4FlashGraph


def check_layout(config, tp: int, ep: int, attn_dp: int) -> None:
    problems = []
    if attn_dp > 1:
        problems.append("DP attention is not served")
    if tp != config.sparse.num_heads:
        # the pool holds one KV head a rank (sglang_patches/hy4_flash_launch.py: the config's 32 KV heads over the TP)
        problems.append(f"--tp-size {tp}: the KV pool holds one {config.sparse.num_heads}th of the config's KV heads a "
                        f"rank, --tp-size {config.sparse.num_heads}")
    for what, kind in (("sparse", config.sparse), ("sliding", config.swa)):
        if kind.num_heads % tp:
            problems.append(f"{kind.num_heads} {what}-attention heads do not shard over --tp-size {tp}")
    if ep != tp:
        problems.append(f"--ep-size {ep}: the experts run whole on their rank, --ep-size {tp}")
    elif config.num_experts % ep:
        problems.append(f"{config.num_experts} experts do not split over --ep-size {ep}")
    for what, size in (("dense", config.intermediate_size), ("shared-expert", config.shared_intermediate_size),
                       ("vocabulary", config.vocab_size)):
        if size % tp:
            problems.append(f"a {what} size of {size} does not shard over --tp-size {tp}")
    if problems:
        raise ValueError(f"{NAME}: " + "; ".join(problems))


class HYV4FlashForConditionalGeneration(ModelHost):
    name = NAME
    # the prefill attends through its own torch attention over the request's table, not the segmented kernel
    segmented_prefill = False
    # the context rungs of both families are the runtime's own (runtime.py)
    decode_context_ladder = False

    @property
    def runtime_class(self):
        from sglang_neuron.models.hy_v4_flash.runtime import Hy4FlashRuntime

        return Hy4FlashRuntime

    def check_launch(self, config) -> None:
        cfg = hy4_flash_config(config)
        pc = sglang_tree.parallel_config()
        check_layout(cfg, int(pc.tp_size), int(pc.ep_size or 1), sglang_tree.attention_dp_size(pc))
        if int(getattr(pc, "pp_size", 1) or 1) > 1:
            raise ValueError(f"{NAME}: --pp-size is not served; the graphs build every layer")
        if int(getattr(pc, "attn_dcp_size", 1) or 1) > 1:
            raise ValueError(f"{NAME}: --dcp-size is not served")

    def build_graph(self, phase: str, config) -> nn.Module:
        return Hy4FlashGraph(hy4_flash_config(config), phase)

    def bind_kv_cache(self, pool):
        """The pool's K and V buffers to every graph (each layer's MLA rows), then the graphs' per-slot state,
        allocated once and shared by an aggregated server's two graphs."""
        super().bind_kv_cache(pool)
        shared = None
        for graph in self.model_graphs.values():
            shared = graph.bind_state(graph.pool_slots, graph.pool_device, shared)

    def load_weights(self, weights):
        super().load_weights(weights)
        # the hash multipliers are buffers drawn from the loaded lookup table, which the second graph shares
        for graph in self.model_graphs.values():
            graph.bind_hashes()

    def neuron_compiler_args(self, phase: str, default_args: list[str]) -> list[str]:
        from sglang_neuron.models.hy_v4_flash.launch import FP8_CAST, FP8_FN_AS_E4M3

        args = super().neuron_compiler_args(phase, default_args)
        if os.environ.get(FP8_CAST) != "1":
            return args
        prefix = "--internal-hlo2tensorizer-options="
        for i, arg in enumerate(args):
            if arg.startswith(prefix):
                if FP8_FN_AS_E4M3 not in arg:
                    args = [*args[:i], f"{arg} {FP8_FN_AS_E4M3}", *args[i + 1:]]
                return args
        return [*args, prefix + FP8_FN_AS_E4M3]


EntryClass = [HYV4FlashForConditionalGeneration]
