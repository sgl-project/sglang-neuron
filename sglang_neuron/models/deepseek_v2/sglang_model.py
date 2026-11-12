# SPDX-License-Identifier: Apache-2.0
"""DeepseekV2ForCausalLM: the host SGLang builds for a DeepSeek-V2 checkpoint (models/host.py)."""

from torch import nn

from sglang_neuron import sglang_tree
from sglang_neuron.models.deepseek_v2.config import deepseek_v2_config
from sglang_neuron.models.deepseek_v2.model import DeepseekV2Graph
from sglang_neuron.models.host import ModelHost
from sglang_neuron.models.transformer.config import check_bf16_launch, check_no_expert_parallel
from sglang_neuron.ops.moe.experts import MAX_FUSED_TOKENS

# the MLA kernels' query heads per rank, and the context tile of the decode kernel
KERNEL_HEADS = 4
DECODE_CONTEXT_TILE = 512
# the largest context compiled on hardware: the prefill kernel holds its whole window in SBUF, 9 bytes a token on each
# partition beside its other tiles, so a longer one may not fit
MAX_CONTEXT = 4096


class DeepseekV2ForCausalLM(ModelHost):
    name = "DeepSeek-V2"
    # the prefill attends through the MLA prefill kernel, over a window of the whole request: its cached prior and its
    # new tokens
    segmented_prefill = False
    prefill_holds_prior = True
    # the decode attention reads only its graph's context (ops/attention/mla pads a rung to whole supertiles)
    decode_context_ladder = True

    def check_launch(self, config) -> None:
        """Refuse a launch the MLA kernels can't serve: they need four query heads per rank, a decode context of whole
        512-token tiles (up to the largest compiled on hardware), and a prefill window holding the whole request, so the
        chunk must equal the context. The decode's experts run in moe_tkg, at most MAX_FUSED_TOKENS tokens a step, and
        the largest decode graph is --max-running-requests. The launch must set it, or SGLang would size the batch from
        the KV pool after this check."""
        from sglang.srt.runtime_context import get_model, get_schedule

        check_bf16_launch(config, self.name)
        check_no_expert_parallel(self.name)
        cfg = deepseek_v2_config(config)
        pc = sglang_tree.parallel_config()
        if sglang_tree.attention_dp_size(pc) != 1:
            raise ValueError("DeepSeek-V2 is served on attention TP without DP attention; launch without "
                             "--enable-dp-attention")
        heads = cfg.num_attention_heads // int(pc.tp_size)
        if heads * int(pc.tp_size) != cfg.num_attention_heads or heads != KERNEL_HEADS:
            raise ValueError(f"--tp-size {pc.tp_size}: DeepSeek-V2's MLA kernels take {KERNEL_HEADS} query heads per "
                             f"rank; launch with --tp-size {cfg.num_attention_heads // KERNEL_HEADS}")
        context = get_model().context_length
        if not context or context % DECODE_CONTEXT_TILE:
            raise ValueError(f"--context-length {context}: DeepSeek-V2's decode kernel reads the context in tiles of "
                             f"{DECODE_CONTEXT_TILE} tokens; launch with a --context-length that is a multiple of it")
        from sglang_neuron.envs import deepseek_v2_max_context

        max_context = deepseek_v2_max_context(MAX_CONTEXT)
        if context > max_context:
            raise ValueError(f"--context-length {context}: DeepSeek-V2's prefill kernel holds a whole window in SBUF "
                             f"and is compiled on hardware up to {max_context} tokens; launch with a --context-length "
                             f"of at most {max_context}")
        running = get_schedule().max_running_requests
        if not running or int(running) > MAX_FUSED_TOKENS:
            raise ValueError(f"--max-running-requests {running}: DeepSeek-V2's decode runs its experts in moe_tkg, "
                             f"which takes at most {MAX_FUSED_TOKENS} tokens a step; launch with "
                             f"--max-running-requests of at most {MAX_FUSED_TOKENS}")
        chunk = int(get_schedule().chunked_prefill_size or 0)
        if chunk != context:
            raise ValueError(f"--chunked-prefill-size {chunk}: DeepSeek-V2's prefill kernel holds a whole request in "
                             f"one window; launch with --chunked-prefill-size {context}, the context length")

    def bind_kv_cache(self, pool):
        """The paged latent pool (SGLANG_NEURON_MLA_POOL=paged) is bound only where its blocks are SGLang's pages
        whole: one block a page and no context parallelism, so a layer's [blocks, block, dim] viewed as rows is SGLang's
        slot layout (DeepseekV2Graph.bind_kv_cache)."""
        if getattr(pool, "is_neuron_latent_pool", False):
            cp = int(pool.cp.cp_size)
            if cp != 1 or int(pool.block_size) != int(pool.page_size) or tuple(pool.latent_shape()[1:3]) != (
                    pool.size // pool.page_size + 1, pool.page_size):
                raise ValueError(f"DeepSeek-V2 reads the paged latent pool as SGLang's token slots: it needs one block "
                                 f"a page and no context parallelism, not cp={cp}, block={pool.block_size}, "
                                 f"page={pool.page_size}, latent={tuple(pool.latent_shape())}")
        super().bind_kv_cache(pool)

    def build_graph(self, phase: str, config) -> nn.Module:
        return DeepseekV2Graph(deepseek_v2_config(config), phase)


EntryClass = [DeepseekV2ForCausalLM]
