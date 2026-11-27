# SPDX-License-Identifier: Apache-2.0
"""Inkling2LoopForCausalLM: the host SGLang builds for the text view of an Inkling-2-Mini checkpoint (models/host.py;
tools/inkling2/make_text_view.sh: the checkpoint's text_config as config.json, so SGLang needs no multimodal processor).

It serves the checkpoint's text decoder on the graphs in graph.py, TP over the attention-TP group, one KV head a rank.
The KV pool holds more layers than the config's 24 (graph.kv_layers: the 48 layer visits of the looped decoder and
the layers of the conv-state record), so the host builds it itself (``configure_kv_cache``): the platform's MHA pool
and paged allocator, sized from SGLang's budget at that layer count. The page is the state record's unit and must be
the chunk's divisor (``check_launch``).
"""

from __future__ import annotations

import logging

from torch import nn

from sglang_neuron.models.host import ModelHost
from sglang_neuron.models.inkling2.config import Inkling2Config

logger = logging.getLogger(__name__)

#: the KV page the graphs are built for: the conv-state record's unit (graph.StateLayout)
PAGE = 256


def tp_size() -> int:
    from sglang_neuron import sglang_tree

    return int(sglang_tree.parallel_config().tp_size)


class Inkling2LoopForCausalLM(ModelHost):
    name = "Inkling-2-Mini"
    # the prefill attention is the model's own (ops.attention_core over its window blocks), not the segmented kernel
    segmented_prefill = False
    # the prefill runs whole chunks: the conv-state record is written at the end of every page a chunk fills
    prefill_ladder = False
    # the decode reads its request's whole table (full layers) or its window blocks (sliding visits)
    decode_context_ladder = False

    def check_launch(self, config) -> None:
        from sglang.srt.runtime_context import get_schedule

        from sglang_neuron.models.transformer.config import check_no_expert_parallel

        check_no_expert_parallel(self.name)
        cfg = Inkling2Config.from_hf(config, tp_size=tp_size())
        problems = cfg.problems() if hasattr(cfg, "problems") else []
        schedule = get_schedule()
        if int(schedule.page_size or 1) != PAGE:
            problems.append(f"--page-size {schedule.page_size}: the graphs read and write {PAGE}-token pages (the "
                            f"conv-state record's unit); launch with --page-size {PAGE}")
        chunk = getattr(schedule, "chunked_prefill_size", None)
        if chunk is not None and int(chunk) > 0 and int(chunk) % PAGE:
            problems.append(f"--chunked-prefill-size {chunk}: a chunk must be whole {PAGE}-token pages")
        if problems:
            raise ValueError(f"{self.name}: " + "; ".join(problems))

    def build_graph(self, phase: str, config) -> nn.Module:
        from sglang_neuron.models.inkling2.graph import Inkling2Graph

        return Inkling2Graph(Inkling2Config.from_hf(config, tp_size=tp_size()), phase, PAGE)

    def neuron_compiler_args(self, phase: str, default_args: list) -> list:
        """Both graphs compile with the transformer model type and the hierarchy splitter, and the decode at -O3: on
        one core the decoder ran in 21.5 ms with these against 31.4 ms with the platform's defaults (generic model
        type, split finder), and the decode's 96 all-reduces a step need -O3 (the -O1 modular flow wraps each
        collective: 43.0 against 29.9 ms a token on the vLLM port). The prefill keeps the platform's -O1 (its TTFT was
        1.61 s at -O1 against 2.09 s at -O3 there). The device sampler's top-k (top logprobs, topk_topp) keeps the
        platform's decode arguments."""
        from sglang_neuron import sampling
        from sglang_neuron.backend.stack import with_hlo_options

        if phase == "decode" and (sampling.MAX_TOP_LOGPROBS > 0 or sampling.sampler_mode() == "topk_topp"):
            return super().neuron_compiler_args(phase, default_args)
        args = with_hlo_options([a for a in default_args if not a.startswith("--model-type")],
                                drop=("--hierarchy-splitter=",), add=("--hierarchy-splitter=true",))
        args.append("--model-type=transformer")
        if phase == "decode":
            args = [a for a in args if a not in ("-O0", "-O1", "-O2", "-O3")] + ["-O3"]
        return args

    # pools
    def configure_kv_cache(self, configurator, *, pre_model_load_memory):
        """The KV pool at ``graph.kv_layers`` layers (module docstring): SGLang's budget over this model's bytes a
        token, in whole pages, the platform's MHA pool and paged allocator, and SGLang's request table."""
        import torch
        from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigResult
        from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig
        from sglang.srt.platforms import current_platform
        from sglang.srt.runtime_context import get_disagg, get_schedule

        from sglang_neuron.model_executor.runtime_init import neuron_device
        from sglang_neuron.models.inkling2.graph import kv_layers

        schedule = get_schedule()
        cfg = self.model_graph.config
        layers = kv_layers(cfg, PAGE)
        if configurator.kv_cache_dtype != torch.bfloat16:
            raise ValueError(f"{self.name}: --kv-cache-dtype {configurator.kv_cache_dtype}; the graphs read a bfloat16 "
                             "KV cache (launch with --kv-cache-dtype auto)")
        per_token = layers * 2 * cfg.kv_heads_per_rank * cfg.head_dim * 2
        budget = int(configurator._profile_available_bytes(pre_model_load_memory))
        tokens = configurator._apply_token_constraints(max(budget // per_token - PAGE, 0))
        tokens = tokens // PAGE * PAGE
        if tokens < PAGE:
            raise ValueError(f"{self.name}: the KV budget ({budget / 2**30:.2f} GiB) holds no {PAGE}-token page at "
                             f"{layers} KV layers ({per_token} bytes a token)")
        pool = current_platform.get_kv_pool_cls(kind="mha")(
            tokens, PAGE, torch.bfloat16, cfg.kv_heads_per_rank, cfg.head_dim, layers, neuron_device(), False,
            start_layer=0, end_layer=layers - 1)
        allocator = current_platform.get_paged_allocator_cls()(
            tokens, page_size=PAGE, dtype=configurator.kv_cache_dtype, device=configurator.device, kvcache=pool,
            need_sort=get_disagg().disaggregation_mode in ("decode", "prefill"))
        max_running_requests = configurator.resolve_max_num_reqs(tokens)
        req_to_token_pool = configurator._build_req_to_token_pool(
            max_num_reqs=configurator._req_pool_rows(max_running_requests))
        config = MemoryPoolConfig(max_total_num_tokens=tokens, max_running_requests=max_running_requests,
                                  mem_fraction_static=schedule.mem_fraction_static)
        logger.info("%s KV pool: %d layers (%d layer visits + %d conv-state), %d tokens (%.2f GiB), %d requests",
                    self.name, layers, cfg.cache_layers, layers - cfg.cache_layers, tokens,
                    tokens * per_token / 2**30, max_running_requests)
        return KVCacheConfigResult(max_total_num_tokens=tokens, max_running_requests=max_running_requests,
                                   full_max_total_num_tokens=None, swa_max_total_num_tokens=None,
                                   req_to_token_pool=req_to_token_pool, token_to_kv_pool=pool,
                                   token_to_kv_pool_allocator=allocator, memory_pool_config=config)

    # weights
    def load_weights(self, weights):
        """Read this rank's shard of every parameter from the checkpoint (graph.Inkling2Graph.load_weights); the lazy
        iterator SGLang hands over is left unread."""
        from sglang_neuron.model_executor.runtime_init import neuron_device
        from sglang_neuron.models.host import place_rest_on_device, share_weights

        if getattr(self, "_weights_loaded", False):
            raise NotImplementedError(f"{self.name} on Neuron does not update its weights in place")
        path, device = self.hf_config._name_or_path, neuron_device()
        (first_phase, first), *others = self.model_graphs.items()
        first.load_weights(path, device)
        place_rest_on_device(self.name, first_phase, first, device)
        for phase, graph in others:
            share_weights(first, graph, name=self.name, phase=phase, source_phase=first_phase)
            place_rest_on_device(self.name, phase, graph, device)
            graph.finish_load(device)
        self._weights_loaded = True
        logger.info("%s weights loaded onto %s from %s", self.name, device, path)


EntryClass = [Inkling2LoopForCausalLM]
