# SPDX-License-Identifier: Apache-2.0
"""MiMoV2ForCausalLM: the host SGLang builds for a MiMo-V2 checkpoint (models/host.py), its graphs model.MiMoV2Graph.

Served on attention TP (no DP attention: SparseMoE and the aggregated server refuse it) with the experts over EP; the
target layout is TP64 / EP64 on the 64 logical cores of a trn2.48xlarge: one query head and one KV head of each
layer and 4 of the 256 experts a rank."""

import os

from torch import nn

from sglang_neuron.models.host import ModelHost
from sglang_neuron.models.mimo_v2.config import NAME, MiMoV2Config, mimo_v2_config
from sglang_neuron.models.mimo_v2.model import MiMoV2Graph, expert_scale_suffix
from sglang_neuron.models.transformer.config import check_bf16_launch


def check_layout(config: MiMoV2Config, tp: int, ep: int, attn_dp: int) -> None:
    """Refuse a parallel layout the graphs cannot shard, before any weight loads."""
    problems = []
    if attn_dp > 1:
        problems.append("DP attention is not served; the experts run on attention TP (launch without "
                        "--enable-dp-attention / --dp-size)")
    for kind_name, kind in (("full", config.full), ("sliding", config.swa)):
        if kind.num_heads % tp:
            problems.append(f"{kind.num_heads} {kind_name}-attention heads do not shard over --tp-size {tp}")
        if tp % kind.num_kv_heads:
            # the decode kernel attends one KV head per rank, so every rank replicates exactly one
            problems.append(f"{kind.num_kv_heads} {kind_name}-attention KV heads do not replicate one a rank across "
                            f"--tp-size {tp} (a multiple of {kind.num_kv_heads})")
    if config.num_experts % ep or tp % ep:
        problems.append(f"{config.num_experts} experts do not split over --ep-size {ep} within --tp-size {tp}")
    elif config.moe_intermediate_size % (tp // ep):
        problems.append(f"an expert intermediate size of {config.moe_intermediate_size} does not shard over the "
                        f"{tp // ep} ranks of each expert")
    if config.intermediate_size % tp:
        problems.append(f"a dense intermediate size of {config.intermediate_size} does not shard over --tp-size {tp}")
    if config.vocab_size % tp:
        problems.append(f"a vocabulary of {config.vocab_size} does not shard over --tp-size {tp}")
    if problems:
        raise ValueError(f"{NAME}: " + "; ".join(problems))


class MiMoV2ForCausalLM(ModelHost):
    name = NAME

    def _set_stage(self, config) -> None:
        super()._set_stage(config)
        from sglang_neuron.models.mimo_v2 import mtp

        if mtp.enabled():
            # the KV pool's layers after the target's are the MTP layers' (SGLang sizes the pool from end_layer)
            self.end_layer += mtp.num_layers()

    @property
    def runtime_class(self):
        from sglang_neuron.models.mimo_v2.runtime import MiMoV2Runtime

        return MiMoV2Runtime

    @staticmethod
    def decode_context_ladder(context: int) -> list:
        """The decode's contexts (graph_runner.BucketConfig): the decode attention reads its graph's whole table, so a
        row's step costs its graph's context, not its own. Rungs at 1/2, 1/4, 1/16 and 1/64 of the context, none
        below 8K: an AgentX step of rows of 10K-200K at a 1M server runs a 16K-256K graph instead of a 256K one."""
        rungs = {context // q for q in (1, 2, 4, 16, 64)}
        return sorted(r for r in rungs if r >= min(8192, context))

    def check_launch(self, config) -> None:
        from sglang_neuron import sglang_tree

        check_bf16_launch(config, self.name)
        mimo = mimo_v2_config(config)
        pc = sglang_tree.parallel_config()
        check_layout(mimo, int(pc.tp_size), int(pc.ep_size or 1), sglang_tree.attention_dp_size(pc))
        dcp = int(getattr(pc, "attn_dcp_size", 1) or 1)
        if dcp > 1:
            from sglang.srt.runtime_context import get_server_args

            from sglang_neuron.distributed.decode_context_parallel import validate_dcp_heads

            # each DCP group's ranks share one KV head of every layer kind (8 sliding KV heads: at most 8 at TP64)
            for kind in (mimo.full, mimo.swa):
                validate_dcp_heads(NAME, kind.num_heads, kind.num_kv_heads, int(pc.tp_size), dcp)
            if not getattr(get_server_args(), "disable_hybrid_swa_memory", False):
                raise ValueError(f"{NAME}: --dcp-size {dcp} is served on one KV pool; launch with "
                                 "--disable-hybrid-swa-memory (and SGLANG_NEURON_HYBRID_SWA_RADIX=0)")
        if int(getattr(pc, "pp_size", 1) or 1) > 1:
            raise ValueError(f"{NAME}: --pp-size is not served; the graphs build every layer")

    def build_graph(self, phase: str, config) -> nn.Module:
        from sglang_neuron.models.mimo_v2.attention import fp8_kv_cache

        if fp8_kv_cache():
            # token-major FP8 K pages, the layout nkilib's decode attention reads heads of 256 in (attention.py)
            from sglang_neuron.model_executor import kv_layout

            kv_layout.declare(kv_layout.KVLayout(fp8_k_packed=False))
        mimo = mimo_v2_config(config)
        suffix = expert_scale_suffix(config)
        return MiMoV2Graph(mimo, phase, quantized=bool(mimo.quant_method), expert_suffix=suffix)

    def neuron_compiler_args(self, phase: str, default_args: list[str]) -> list[str]:
        """FP8 experts on Trn2 compile their float8_e4m3fn tensors as Trn2's e4m3 (launch.py): a frontend option,
        passed through the hlo2tensorizer options."""
        from sglang_neuron.models.mimo_v2.launch import FP8_CAST, FP8_FN_AS_E4M3

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


EntryClass = [MiMoV2ForCausalLM]
