# SPDX-License-Identifier: Apache-2.0
"""GptOssForCausalLM: the host SGLang builds for a GPT-OSS checkpoint (models/host.py), with GPT-OSS's launch checks
and its role graphs, with MXFP4 experts or, under SGLANG_NEURON_GPT_OSS_EXPERTS=bf16, BF16 ones."""

import logging
import math
import shlex

from torch import nn

from sglang_neuron.backend.stack import get_platform_target, normalize_target_family
from sglang_neuron.models.gpt_oss import bf16
from sglang_neuron.models.gpt_oss.config import DTYPE, PADDED_HIDDEN_SIZE, expert_format
from sglang_neuron.models.gpt_oss.mxfp4 import model_decode, model_prefill
from sglang_neuron.models.host import ModelHost
from sglang_neuron.models.phase import model_phase

logger = logging.getLogger(__name__)

_GRAPHS = {"prefill": model_prefill.GptOssForCausalLM, "decode": model_decode.GptOssForCausalLM}
_BF16_GRAPHS = {"prefill": bf16.PrefillGraph, "decode": bf16.DecodeGraph}
# the MXFP4 kernels' default padded expert intermediate size (the checkpoint's is 2880)
_STOCK_INTERMEDIATE = 3072


def padded_intermediate_size(world_size: int, ep_degree: int, intermediate: int) -> int:
    """The expert intermediate size padded so that each rank's shard (over world / ep_degree ranks) tiles the MXFP4
    kernels: a multiple of 32 up to 512, of 512 above. The stock 3072 is kept wherever its shard tiles."""

    def tiled(shard: int) -> bool:
        return shard % (32 if shard <= 512 else 512) == 0

    ep_tp = max(1, world_size // ep_degree)
    shard = math.ceil(intermediate / ep_tp)
    unit = 32 if shard <= 512 else 512
    padded = -(-shard // unit) * unit * ep_tp
    return max(_STOCK_INTERMEDIATE, padded) if tiled(_STOCK_INTERMEDIATE // ep_tp) else padded


def _check_target(giga_decode: bool) -> None:
    # nkilib fixes its QK path when it is first imported: the per-layer decode attention is validated on the
    # non-swapped path, the giga kernel's attention on the swapped one (catalog.py sets each launch's). Each reads its
    # own module's flag: the giga kernel ships its own copy of the attention kernels.
    if giga_decode:
        from sglang_neuron.ops import gpt_oss_giga_decode as giga

        # an image without the giga kernel is refused by name before its attention module is imported
        giga.require()
        from nkilib.models.gpt_oss.c128_giga_kernel.core.attention import attention_tkg_utils
    else:
        from nkilib.core.attention import attention_tkg_utils
    want = "0" if giga_decode else "1"
    if bool(attention_tkg_utils._ATTN_TKG_NO_SWAP) != (want == "1"):
        raise ValueError(f"nkilib's attention kernels were imported with the other QK path: the platform sets "
                         f"NKILIB_EXPERIMENTAL_ATTN_TKG_NO_SWAP={want} for this GPT-OSS launch; do not set it to "
                         "another value")
    target = get_platform_target()
    if normalize_target_family(target) != "trn3":
        raise ValueError(f"the GPT-OSS graphs run on Trainium3, not {target!r}")


# The BF16 expert kernels (prefill: moe_cte block-sharded with GPT-OSS's biases and clamped SwiGLU; decode:
# moe_block_tkg without scales) were checked against their torch paths on trn3 at LNC2, per (--tp-size, --ep-size)
# layout, on three checkpoint layers, prefill calls of 1024 to 8192 tokens and decode batches of 1 to 128
# (numbers in docs/model-recipes/gpt-oss.md). Served layouts:
_BF16_SERVED_LAYOUTS = {(16, 4)}
# measured and refused, with why
_BF16_REFUSED_LAYOUTS = {
    (8, 2): "its BF16 experts take 43.5 GB a rank, more than a logical core's 38.6 GB of HBM",
    (16, 16): "the prefill expert kernel returns NaN rows at its intermediate shard of 3072 from 4096 tokens a call",
}
# the largest prefill expert call and decode batch measured
_BF16_MAX_PREFILL_TOKENS, _BF16_MAX_DECODE_BATCH = 8192, 128


def _check_expert_format(experts: str, policy, config, world: int, ep: int, attn_dp: int) -> None:
    """Refuse BF16 experts on the giga-kernel decode (MXFP4 only) and wherever their kernels were not measured: other
    layouts, DP attention (its MoE calls take every DP group's tokens), and chunks or decode batches past the maxima."""
    if experts != "bf16":
        return
    if policy.giga_decode:
        raise ValueError("SGLANG_NEURON_GPT_OSS_EXPERTS=bf16: the giga-kernel decode runs the MXFP4 experts in its one "
                         "kernel; serve the per-layer decode, or unset SGLANG_NEURON_GPT_OSS_EXPERTS")
    from sglang.srt.runtime_context import get_schedule

    served = " or ".join(f"--tp-size {t} --ep-size {e}" for t, e in sorted(_BF16_SERVED_LAYOUTS))
    if (world, ep) not in _BF16_SERVED_LAYOUTS:
        ep_tp = max(1, world // ep)
        shard = padded_intermediate_size(world, ep, int(config.intermediate_size)) // ep_tp
        why = _BF16_REFUSED_LAYOUTS.get((world, ep), "its BF16 expert kernels are not measured on a NeuronCore at its "
                                                     f"expert shard ({shard} intermediate, "
                                                     f"{int(config.num_local_experts) // ep} experts a rank)")
        raise ValueError(f"SGLANG_NEURON_GPT_OSS_EXPERTS=bf16 at --tp-size {world} --ep-size {ep}: {why}; launch with "
                         f"{served}, or unset SGLANG_NEURON_GPT_OSS_EXPERTS")
    if attn_dp > 1:
        raise ValueError("SGLANG_NEURON_GPT_OSS_EXPERTS=bf16 with DP attention: its expert calls take every DP group's "
                         "tokens, which the BF16 expert kernels are not measured at; launch without "
                         "--enable-dp-attention")
    schedule = get_schedule()
    chunk = int(schedule.chunked_prefill_size or 0)
    if not 0 < chunk <= _BF16_MAX_PREFILL_TOKENS:
        raise ValueError(f"SGLANG_NEURON_GPT_OSS_EXPERTS=bf16 at --chunked-prefill-size {chunk}: the BF16 prefill "
                         f"expert kernel is measured at up to {_BF16_MAX_PREFILL_TOKENS} tokens a call; launch with a "
                         f"--chunked-prefill-size of at most {_BF16_MAX_PREFILL_TOKENS}")
    running = int(schedule.max_running_requests or 0)
    if not 0 < running <= _BF16_MAX_DECODE_BATCH:
        raise ValueError(f"SGLANG_NEURON_GPT_OSS_EXPERTS=bf16 at --max-running-requests {running}: the BF16 decode "
                         f"expert kernel is measured at batches of up to {_BF16_MAX_DECODE_BATCH}; launch with a "
                         f"--max-running-requests of at most {_BF16_MAX_DECODE_BATCH}")
    from sglang_neuron.models.gpt_oss import weight_loaders_bf16
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(weight_loaders_bf16._SOURCES_MD5, weight_loaders_bf16.__name__)


def _check_layout(config, world: int, ep: int, attn_dp: int) -> None:
    """Refuse a parallel layout or layer pattern the GPT-OSS graphs don't serve. Under DP attention the experts must
    split over EP: with --ep-size 1 they would shard over attention TP while their padding assumes the whole world.
    Each attention-TP rank must hold whole heads and either its own KV heads or one KV head replicated over an even
    group, and each EP rank whole experts. The layers must follow the released checkpoints' layer_types (even layers
    sliding, odd layers full), which the runtime reads for each layer's window."""
    from sglang_neuron.models.contract import derive_layer_kinds

    attn_tp = world // attn_dp
    if attn_dp > 1 and ep == 1:
        raise ValueError("GPT-OSS with DP attention splits its experts over EP: launch with --ep-size, as its decode "
                         "deployments do with --ep-size equal to --tp-size")
    heads, kv = int(config.num_attention_heads), int(config.num_key_value_heads)
    if heads % attn_tp:
        raise ValueError(f"GPT-OSS's {heads} attention heads do not split over an attention TP of {attn_tp}")
    if kv % attn_tp and attn_tp % kv:
        raise ValueError(f"GPT-OSS's {kv} KV heads neither split over nor replicate evenly across an attention TP of "
                         f"{attn_tp}")
    experts = int(config.num_local_experts)
    if experts % ep:
        raise ValueError(f"GPT-OSS's {experts} experts do not split over an EP size of {ep}")
    sliding = [kind.sliding_window is not None for kind in derive_layer_kinds(config, "GptOssForCausalLM")]
    if sliding != [i % 2 == 0 for i in range(len(sliding))]:
        raise ValueError("the GPT-OSS graphs slide the even layers and attend fully on the odd ones, as the released "
                         "checkpoints' layer_types do; this checkpoint's layer_types differ")


def _check_decode_context(context: int, dcp_size: int = 1) -> None:
    """Refuse a --context-length at which a decode bucket couldn't read the packed FP8 K cache directly. The runtime
    rounds the context up to LNC x P_MAX tokens (graph_runner.BucketConfig), and the kernel reads the packed cache only
    when its resized page still holds two tokens (packed_fp8_viable_for_bucket): true for every bucket when the full
    table is whole 2 x LNC x P_MAX-token tiles. Otherwise each step would re-swizzle every full layer's whole pool, and
    a disaggregated decode would write that stale copy over KV a transfer had just landed. Under DCP the runtime rounds
    each rank's table to whole tiles itself (decode_context_parallel.decode_context), so any context serves."""
    if dcp_size > 1:
        return
    from sglang_neuron.ops.attention.attention_decode_mask import P_MAX
    from sglang_neuron.utils.hardware_config import get_logical_nc_config

    rounding, tile = get_logical_nc_config() * P_MAX, 2 * get_logical_nc_config() * P_MAX
    if -(-context // rounding) * rounding % tile:
        raise ValueError(f"--context-length {context}: GPT-OSS's decode attention reads the packed FP8 K cache over a "
                         f"context of whole {tile}-token tiles (2 x LNC x {P_MAX}); launch with a --context-length that "
                         f"is a multiple of {tile}")


def _packed_fp8_decode(phase: str) -> bool:
    """Whether this process serves a decode graph, whose attention reads the packed FP8 K cache under
    _check_decode_context's tile constraint: an aggregated server or a disaggregated decode worker. A prefill-only
    worker builds no decode graph, so the constraint refuses none of its contexts."""
    return phase in ("decode", "all")


class GptOssForCausalLM(ModelHost):
    name = "GPT-OSS"
    # one prefill graph of the chunk: the MXFP4 prefill graph is the long one to compile, and the prefill role of its
    # deployments runs whole chunks of long prompts
    prefill_ladder = False
    # under PP each phase's per-layer graph builds one stage (distributed.pipeline; the giga-kernel decode builds none).
    # The boundary is at the kernels' padded width: hidden states and residual in the prefill, hidden states in decode
    pp_boundary_dtype = DTYPE
    pp_hidden_size = PADDED_HIDDEN_SIZE

    def check_launch(self, config) -> None:
        from sglang_neuron import serving_policy, sglang_tree

        policy = serving_policy.current()
        pc = sglang_tree.parallel_config()
        world, ep, attn_dp = int(pc.tp_size), int(pc.ep_size or 1), sglang_tree.attention_dp_size(pc)
        _check_expert_format(expert_format(), policy, config, world, ep, attn_dp)
        _check_target(policy.giga_decode)
        _check_layout(config, world, ep, attn_dp)
        dcp_size = int(getattr(pc, "attn_dcp_size", 1) or 1)
        if dcp_size > 1:
            from sglang_neuron.distributed.decode_context_parallel import validate_dcp_heads

            # the ranks of a DCP group must hold the same single KV head; the group gathers its query heads
            validate_dcp_heads("GPT-OSS", int(config.num_attention_heads), int(config.num_key_value_heads),
                               world // attn_dp, dcp_size)
        # SGLang's ModelConfig puts the resolved context (--context-length, else the checkpoint's) on the config; the
        # tile constraint is the packed-FP8 decode kernel's, so only a process that serves decode is checked (a
        # disaggregated prefill-only worker builds no decode graph and serves any context)
        if _packed_fp8_decode(model_phase()):
            _check_decode_context(int(config.context_len), dcp_size)

    def build_graph(self, phase: str, config) -> nn.Module:
        intermediate = padded_intermediate_size(self.tp_size, self.ep_degree, config.intermediate_size)
        experts = expert_format()
        logger.info("GPT-OSS %s graph: %s experts, expert intermediate %d", phase, experts, intermediate)
        if phase == "decode" and _giga_decode():
            if self.stage is not None:
                raise ValueError("the giga-kernel decode runs every layer in one kernel call: it builds no pipeline "
                                 "stage")
            from sglang_neuron.parallel import groups

            if groups.get_neuron_dcp_group() is not None:
                raise ValueError("the giga-kernel decode attends over each rank's whole KV: it serves no decode "
                                 "context parallelism; serve --dcp-size on the per-layer decode")
            from sglang_neuron.models.gpt_oss.giga.model_decode import GptOssForCausalLM as GigaDecode

            return GigaDecode.from_hf_config(config, intermediate)
        graphs = _BF16_GRAPHS if experts == "bf16" else _GRAPHS
        return graphs[phase].from_hf_config(config, intermediate, stage=self.stage)

    @property
    def runtime_class(self):
        """The giga-kernel decode's runtime, which builds its decode inputs in one numpy pass; the platform's for every
        other graph."""
        if _giga_decode():
            from sglang_neuron.models.gpt_oss.giga.runtime import GigaDecodeRuntime

            return GigaDecodeRuntime
        return None

    def neuron_compiler_args(self, phase: str, default_args: list[str]) -> list[str]:
        if phase == "decode" and _giga_decode():
            return _giga_decode_compiler_args(default_args)
        return super().neuron_compiler_args(phase, default_args)

    def neuron_after_warmup(self, family: str, warmed_keys: dict) -> None:
        """Refuse the giga-kernel decode when the warmed decode NEFF's kernel scope isn't its DMA-order schedule's, or
        fewer than schedule.MIN_BOUND_SHARE of the schedule's constraints name an instruction of it
        (giga/schedule.py)."""
        if family != "decode" or not _giga_decode():
            return
        import torch_neuronx

        from sglang_neuron.models.gpt_oss.giga import schedule

        neffs = []
        for graph, keys in warmed_keys.items():
            if not graph.startswith("decode:"):
                continue
            for key in keys:
                info = torch_neuronx._C.get_neff_info(key)
                if info and info[1]:
                    neffs.append(bytes(info[1]))
        schedule.check_compiled(neffs)

    def bind_kv_cache(self, pool):
        super().bind_kv_cache(pool)
        # the giga-kernel decode also takes the per-pair tensors the pool allocated its layers' halves in
        for graph in self.model_graphs.values():
            if hasattr(graph, "bind_kv_pairs"):
                graph.bind_kv_pairs(pool.layer_pairs)


def _giga_decode() -> bool:
    from sglang_neuron import serving_policy

    return serving_policy.current().giga_decode


def _giga_decode_compiler_args(default_args: list[str]) -> list[str]:
    """The giga-kernel decode graph's compiler arguments: -O2 without the modular-flow threshold (as the per-layer
    decode), the sampling all-gather not annotated as FSDP, and the kernel's DMA-order schedule with the BIR SBUF
    allocator it was derived with (the constraint pass reads the schedule's sbuf_alloc_limit). The schedule is named by
    its content (schedule.compiler_path), so a changed schedule compiles a NEFF of its own."""
    from sglang_neuron.models.gpt_oss.giga.schedule import compiler_path

    schedule = compiler_path()
    hlo_prefix, backend_prefix = "--internal-hlo2tensorizer-options=", "--internal-backend-options="
    args, have_backend = [], False
    for arg in default_args:
        if arg in ("-O0", "-O1", "-O2", "-O3"):
            continue
        if arg.startswith(hlo_prefix):
            options = [o for o in shlex.split(arg[len(hlo_prefix):])
                       if not o.startswith("--modular-flow-mac-threshold=")]
            arg = hlo_prefix + shlex.join(options + ["--enable-annotate-fsdp-cc=false"])
        elif arg.startswith(backend_prefix):
            arg = f"{arg} --dma-order-constraint-file={schedule}"
            have_backend = True
        args.append(arg)
    if not have_backend:
        args.append(f"{backend_prefix}--dma-order-constraint-file={schedule}")
    if not any(a.startswith(hlo_prefix) for a in args):
        args.append(hlo_prefix + "--enable-annotate-fsdp-cc=false")
    return args + ["--internal-bir-enable-sb-allocation", "-O2"]


EntryClass = [GptOssForCausalLM]
