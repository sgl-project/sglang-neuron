# SPDX-License-Identifier: Apache-2.0
"""The models the runtime serves, by the architecture a checkpoint's config names.

The platform reads a launch's entry before any scheduler starts. It refuses a KV cache the model's graphs do not serve,
sets the model's launch settings for every process of the launch, and points SGLang's model registry at the model's
package, so each process imports that model alone. This module imports nothing heavy: the launching process reads it
before it has decided anything else.
"""

from __future__ import annotations

import functools
import importlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from sglang_neuron.models.glm52.prefill import compile_args as _glm52_compile_args
from sglang_neuron.models.glm52 import points as _glm52_points
from sglang_neuron.models.glm52.prefill import recipe as _glm52_recipe
# imports only models.capability
from sglang_neuron.models.kimi_k3.decode import capability as _k3_decode

# Names the package SGLang builds the model from (SGLang's variable). The platform sets it in the launching process;
# every process the launch starts imports that package strictly (general_plugin._register_model_package) and its ranks
# run the package's rank hook (model_executor.runtime_init). Here, not in the platform: a process imports the platform,
# SGLang's platform plugin, only after SGLang, and a rank reads the name before it imports SGLang.
MODEL_PACKAGE_ENV = "SGLANG_EXTERNAL_MODEL_PACKAGE"


@dataclass(frozen=True)
class LaunchSetting:
    name: str
    value: str
    reason: str


@dataclass(frozen=True)
class ModelSpec:
    # the package SGLang builds the model from (SGLANG_EXTERNAL_MODEL_PACKAGE); one of its modules has an EntryClass
    # named after the architecture
    package: str
    # the --kv-cache-dtype values the model's graphs serve
    kv_cache_dtypes: tuple
    # environment settings the model's kernels read when they are imported; a model with launch rules (launch_hooks)
    # gets its launch's own from their launch_settings, and this field states its default layout's
    launch_settings: tuple = ()
    # the prefill window holds the request's cached prior with its new tokens (the model host's prefill_holds_prior)
    prefill_holds_prior: bool = False
    # the quantization methods the graphs serve, as a checkpoint's quantization_config or --quantization names them
    quantizations: tuple = ()
    # the model has a giga-kernel decode, which serves its decode over the kernel's layout
    # (serving_policy.giga_decode_layout) with these launch settings in place of launch_settings
    giga_decode: bool = False
    giga_decode_launch_settings: tuple = ()
    # the giga-kernel decode's attention DP over its 128 ranks (serving_policy.giga_decode_layout): SGLang's DP
    # attention groups, or 1 for a kernel that runs its own attention DP inside
    giga_decode_dp: int = 16
    # the sampler modes (SGLANG_NEURON_SAMPLER) the giga-kernel decode's tail serves: greedy alone for a fused tail that
    # returns greedy tokens (GPT-OSS), or topk_topp too for one that samples each request's top_k / top_p /
    # temperature on device (GLM-5.2: the GK's LM-head tail, models/glm52/tail_sampling.py)
    giga_decode_samplers: tuple = ("greedy",)
    # the sampler modes (SGLANG_NEURON_SAMPLER) the model's own runtime takes, empty for every mode the platform serves.
    # A runtime that takes only some of them (Qwen3.5's giga decode takes only greedy, models/qwen35/runtime.py) makes
    # the platform refuse a host-sampler launch at once and keeps the host sampler out of its request refusals
    # (platform.host_sampler_unserved)
    device_samplers: tuple = ()
    # a module of the host side of the model's in-kernel MTP step, which --speculative-algorithm NEURON_MTP serves
    # (speculative/neuron_mtp.py says what it defines), "" for a model whose decode runs none: platform.refuse_unserved
    # refuses the algorithm for it
    mtp_host: str = ""
    # the HBM scratchpad page, in MiB, the model's graphs compile with and the runtime bills in, by serving role
    # (serving_policy.ServingPolicy.role): ((role, MiB), ...); a role it does not name keeps the plugin's
    # (runtime_init.SCRATCHPAD_PAGE_MB)
    scratchpad_page_mb: tuple = ()
    # the --mem-fraction-static a launch of each serving role takes when it names none ((role, fraction), ...;
    # platform.default_mem_fraction); a role it does not name keeps SGLang's default
    mem_fraction_static: tuple = ()
    # the model builds pipeline stages (--pp-size above 1, distributed.pipeline): its host builds a stage's layers
    pipeline_stages: bool = False
    # a pipelined PD prefill of the model computes requests ahead of their decode seat
    # (serving_policy._prefill_compute_ahead, sglang_patches.pp_prefill_compute_ahead), HW-gated on this model's 1P1D;
    # a model without it never does
    prefill_compute_ahead: bool = False
    # a DSA model whose launch rules choose its attention backend and page: on Neuron, SGLang's DeepSeek-family
    # overrides declare neither for it (sglang_patches.dsa_platform_adjustments)
    owns_dsa_attention: bool = False
    # the req_to_token page sizes the model's graphs serve, the first one a launch without --page-size gets; empty: the
    # platform's PAGE_SIZE or the model's launch rules' page (platform.check_kv_cache)
    page_sizes: tuple = ()
    # the 0/1 environment switch that opts the model's PD decode into HiCache's host tier (its pools build their own
    # host tier; platform.refuse_unserved lifts the decode refusal when it reads 1); "" keeps the decode refused
    decode_hicache_env: str = ""
    # the per-layer decode attends over DCP KV shards (--dcp-size > 1, distributed.decode_context_parallel)
    decode_context_parallel: bool = False
    # the PD prefill stores its KV in DCP placement and attends over the prior's DCP shards (--dcp-size > 1 on the
    # prefill role, also with --pp-size above 1)
    prefill_context_parallel: bool = False
    # sliding-window layers read SGLang's window pool through the runtime's translated tables (mem_cache.hybrid_swa)
    hybrid_swa_memory: bool = False
    # the logical-core degree (NEURON_LOGICAL_NC_CONFIG, platform.set_logical_nc_config) the model's graphs and kernels
    # are built for; the giga-kernel decode runs 1 and a declared decode layout its own degree, whatever the model's
    logical_nc: int = 2
    # a module of the model's own launch rules (launch_hooks), "" for none. It defines kv_page_size(server_args) -> int
    # (the KV page of the launch's role and layout), launch_problems(server_args, checkpoint_config, environ) -> list
    # (every reason the launch is not served, run in the launching process before any scheduler starts),
    # launch_settings(server_args) -> tuple of LaunchSetting (the settings the platform applies for a launch that is not
    # a giga-kernel decode, in place of the launch_settings field, after launch_problems has passed) and
    # rank_runtime_env(role) -> dict (the Neuron runtime settings each rank of the serving role, serving_policy
    # .ServingPolicy.role, starts with over runtime_init.runtime_defaults, unless the launch sets them), and optionally
    # validate_request(obj, preferred_sampling_params, server_args, vocab_size) (the model's own admission rules, run on
    # every request before it is tokenized further), chat_sampling_params(request) -> dict (the sampling values that
    # replace a chat request's own, sglang_patches.model_chat_sampling), apply_defaults(server_args) (the launch values the model's layout
    # fixes where the launch leaves them unset, run first) and prepare_process() (the model's setup of each process:
    # the launching process, before SGLang resolves the arguments, and every scheduler). Its module-level imports are
    # stdlib only: the launching process imports it before it has decided anything else.
    launch_hooks: str = ""
    # a module of the model's PD KV hand-off, for a decode whose KV lives in its kernel's own caches rather than SGLang's
    # pool (sglang_patches.kernel_kv_handoff: DECODE_HOOKS and PREFILL_HOOKS), "" for upstream's path
    kv_handoff: str = ""
    # the --speculative-algorithm names (upper case) the model's own speculative worker serves
    # (model_executor.spec_worker); every other model refuses speculative decoding
    speculative_algorithms: tuple = ()
    # the PD decode layouts (capability.DecodeLaunchLayout) the model's decode is built for: a decode launched in one
    # is served over its nodes, at its logical NC degree, with its DCP groups inside the DP-attention groups
    # (serving_policy.declared_decode_layout); a launch over more than one node is refused for every other model
    decode_layouts: tuple = ()
    # the prefill graph writes a request's recurrent state at the end of each forward into its slot and into the
    # tracking slot SGLang's batch names (upstream's mamba extra buffer), so the chunk-keyed tree
    # (--radix-cache-backend neuron-chunk) holds an exact state checkpoint at every chunk boundary
    # (mem_cache.chunk_keyed_tree)
    chunk_state_checkpoints: bool = False
    # The model is served by its whole-model megakernel (--enable-nvidia-megakernel; megakernel/registry.SPECS), every
    # role a MegakernelServing names. None: the per-layer graphs (and, for GPT-OSS, its giga-kernel decode, which
    # giga_decode declares).
    megakernel: Optional["MegakernelServing"] = None
    # a PD prefill of the model posts a batch's failed NIXL writes again before it fails the room
    # (sglang_patches.nixl_repost_failed_writes): its decode can restart under a running prefill
    pd_prefill_reposts_failed_writes: bool = False
    # upstream SGLang's attention override providers of the architecture, as (module, function), that declare nothing
    # for it on Neuron: the model's graphs run its attention (sglang_patches.dsa_platform_adjustments.skip_providers)
    skipped_override_providers: tuple = ()


@dataclass(frozen=True)
class MegakernelServing:
    """How the platform serves a model's whole-model megakernel (serving_policy.resolve).

    Every program of such a model runs each layer's collectives inside its kernel calls over all the server's ranks,
    the regime the giga-kernel decode was measured in, so each role gets the giga runtime defaults
    (runtime_init.runtime_defaults)."""

    # Its DP ranks plan every step from SGLang's per-step DP-attention exchange (every replica's token and request
    # counts): each rank derives the same kernel calls, a prefill replica's and a decode replica's alike, and an idle
    # replica runs padding in the same calls. So an aggregated server serves --enable-dp-attention
    # (serving_policy.refuse_unserved_layouts), and a decode with DP attention keeps that exchange instead of the DP
    # identity, whose constant result would drop the counts the plan reads.
    dp_steps_from_exchange: bool = False
    # Its programs serve async execution in the megakernel host's host-ids mode (async_execution.configure(host_ids=)):
    # a step's kernel calls are submitted without a read in between and the step returns one pending result; the next
    # decode takes its ids on the host from that result's readback, so SGLang's result processing and the next batch's
    # scheduling run while the device runs the step. False: the launch needs --disable-overlap-schedule. A launch with a
    # speculative worker (NEXTN/MTP) runs synchronously either way (serving_policy.megakernel_async_blockers).
    async_steps: bool = False


# The build of nkilib's giga kernel that the giga-kernel decode's DMA-order schedule
# (models/gpt_oss/giga/dma_order.json) was derived on, pinned here for that decode. nkilib reads these flags itself,
# when it imports or traces the kernel. A flag that differs changes only a few percent of the DMAs the schedule names,
# which the schedule gate's share can't see.
_GIGA_BUILD_FLAGS = (
    ("VLLM_NEURON_MEGA_ROUTERW_DMA_ENGINE", "sync"),
    ("VLLM_NEURON_MEGA_H0_RESID1", "1"),
    ("VLLM_NEURON_MEGA_QMX_BATCH_XPOSE", "1"),
    ("VLLM_NEURON_MEGA_GU_RS_FLOWGATE", "1"),
    ("VLLM_NEURON_MEGA_MOE_OUT_FP8", "1"),
    ("VLLM_NEURON_MEGA_GU_IOUTER", "2"),
    ("VLLM_NEURON_MEGA_GU_PREFETCH_HSPLIT", "1"),
)
# The prefix of the giga kernel's own flags. Those beyond the pinned ones change the kernel's build and,
# for some, its numerics (several read any value, "0" too, as on), so the giga-kernel decode refuses them.
GIGA_KERNEL_FLAG_PREFIX = "VLLM_NEURON_MEGA_"
_GIGA_BUILD = ("the giga kernel's DMA-order schedule was derived on the kernel nkilib builds with this value (nkilib "
               "reads it when it imports or traces the kernel)")

MODELS = {
    "GptOssForCausalLM": ModelSpec(
        package="sglang_neuron.models.gpt_oss",
        kv_cache_dtypes=("fp8_e4m3",),
        quantizations=("mxfp4",),
        launch_settings=(LaunchSetting(
            "NKILIB_EXPERIMENTAL_ATTN_TKG_NO_SWAP", "1",
            "GPT-OSS's decode attention is validated on nkilib's non-swapped QK path, which nkilib fixes when it is "
            "imported"),),
        giga_decode=True,
        giga_decode_launch_settings=(LaunchSetting(
            "NKILIB_EXPERIMENTAL_ATTN_TKG_NO_SWAP", "0",
            "the giga kernel's attention is validated on nkilib's QK-swapped path, which nkilib fixes when it is "
            "imported"),) + tuple(LaunchSetting(name, value, _GIGA_BUILD) for name, value in _GIGA_BUILD_FLAGS),
        pipeline_stages=True,
        decode_context_parallel=True,
        hybrid_swa_memory=True,
    ),
    "Qwen3ForCausalLM": ModelSpec(package="sglang_neuron.models.qwen3", kv_cache_dtypes=("auto", "bf16", "bfloat16")),
    # the language model of the Gemma 4 dense checkpoint (text only)
    "Gemma4ForConditionalGeneration": ModelSpec(package="sglang_neuron.models.gemma4",
                                                kv_cache_dtypes=("auto", "bf16", "bfloat16"),
                                                launch_hooks="sglang_neuron.models.gemma4.launch"),
    "Glm4ForCausalLM": ModelSpec(package="sglang_neuron.models.glm4", kv_cache_dtypes=("auto", "bf16", "bfloat16")),
    "Qwen3MoeForCausalLM": ModelSpec(package="sglang_neuron.models.qwen3_moe",
                                     kv_cache_dtypes=("auto", "bf16", "bfloat16")),
    # the paged giga prefill writes each forward's final GDN state into SGLang's tracking slot, so the chunk-keyed tree
    # keeps an exact checkpoint at every chunk boundary (models/qwen35/prefix.py)
    "Qwen3_5MoeForCausalLM": ModelSpec(package="sglang_neuron.models.qwen35",
                                       kv_cache_dtypes=("auto", "bf16", "bfloat16"),
                                       # the paged pool's KV pages (powers of two the paged attention reads)
                                       page_sizes=(512, 256, 1024, 2048),
                                       # the decode's kv_cp pool host tier (models/qwen35/config.HICACHE_ENV)
                                       decode_hicache_env="SGLANG_NEURON_QWEN35_HICACHE",
                                       # the giga kernels run at LNC 1: 128 ranks on a node's physical cores
                                       logical_nc=1,
                                       # the giga decode's runtime takes only the greedy sampler
                                       # (models/qwen35/runtime.py)
                                       device_samplers=("greedy",),
                                       chunk_state_checkpoints=True,
                                       # a PD prefill posts a batch's failed writes again: its decode can restart
                                       # under it (sglang_patches.nixl_repost_failed_writes)
                                       pd_prefill_reposts_failed_writes=True,
                                       # every rank plans each step's programs from the DP-attention cohort snapshot
                                       # (models/qwen35/megakernel.py); its async decode reads its ids from device tokens
                                       megakernel=MegakernelServing(dp_steps_from_exchange=True, async_steps=True)),
    "DeepseekV2ForCausalLM": ModelSpec(package="sglang_neuron.models.deepseek_v2",
                                       kv_cache_dtypes=("auto", "bf16", "bfloat16"), prefill_holds_prior=True),
    # GLM-5.2 (models/glm52), one host whose PD role picks its giga kernel (sglang_model.py) and one set of
    # launch rules dispatching by the same role (launch.py), both at LNC1:
    # - the PD prefill on the prefill giga kernel, PP4 x TP32 or PP16 x TP8 (launch.launch_settings: its point's
    #   recipe; launch_settings here: the default PP4 point's). Its pool stores bf16 words of the kernel's 640-byte
    #   static KV row, whose FP8 latent is the kernel's own format;
    # - the PD decode on the decode giga kernel at TP128 without DP attention (the giga_decode fields: its static
    #   recipe), whose caches are the kernel's own, so no KV dtype flag reaches them.
    # The two roles pair over NIXL: the GLM PD KV hand-off (models/glm52/pd/handoff.py, pd/placement.py)
    # writes the prefill's KV into the decode kernel's own caches. The decode also runs on SGLang's fake KV transfer,
    # measured alone, and refuses every other backend (launch._decode_problems).
    # The MXFP4 checkpoint is a Quark export, which each role's graphs load themselves.
    "GlmMoeDsaForCausalLM": ModelSpec(
        package="sglang_neuron.models.glm52",
        kv_cache_dtypes=("auto", "bf16", "bfloat16"),
        quantizations=("quark",),
        launch_settings=tuple(LaunchSetting(*setting) for setting in _glm52_recipe.SETTINGS),
        giga_decode=True,
        # the static recipe plus the launch's decode point's kernel and runtime settings (models/glm52/points.py,
        # SGLANG_NEURON_GLM52_POINT, read when the catalog is imported: the launch's own environment)
        giga_decode_launch_settings=tuple(LaunchSetting(*setting)
                                          for setting in _glm52_points.current().launch_settings()),
        giga_decode_dp=1,
        giga_decode_samplers=("greedy", "topk_topp"),
        mtp_host="sglang_neuron.models.glm52.mtp.host",
        kv_handoff="sglang_neuron.models.glm52.pd.handoff",
        # each kernel's own page: the prefill's served page (one holds the 1M graph's scratchpad), and the launch's
        # decode point's for the GK (the runtime's default 512 at 128k, which it is measured with; 128 at 1M): at the
        # plugin's 3584 MiB the 128-rank decode NEFF's scratchpad, billed in whole pages, leaves its collectives no
        # room to load ("Failed to allocate resource")
        scratchpad_page_mb=(("prefill", _glm52_compile_args.SCRATCHPAD_PAGE_MB),
                            ("decode", _glm52_points.current().scratchpad_page_mb)),
        # the prefill's stage pools are stated in pages (--max-total-tokens), and their fit check leaves this fraction
        # less the weights for them: a quarter of a core's HBM stays for the graphs, the scratchpad and a device
        # profile's buffers, over stage 3's 8.1 GB pool at the served 16,384 pages (models/glm52/prefill/kv_pool.py)
        mem_fraction_static=(("prefill", 0.75),),
        pipeline_stages=True,
        # compute ahead on the PD prefill: a lower TTFT p50 on the GLM-5.2 1P1D at concurrency 128
        prefill_compute_ahead=True,
        owns_dsa_attention=True,
        logical_nc=1,
        launch_hooks="sglang_neuron.models.glm52.launch",
    ),
    # the text model (models/kimi_k3). PD prefill: PP stages of a TP group that stores its MLA rows in DCP placement,
    # FP8, one rank per physical NeuronCore; the checkpoint's MXFP4 experts are compressed-tensors. PD decode: the
    # 2-node DSPARK decode (models/kimi_k3/decode/capability), DCP 32 inside each DP-attention group
    "KimiK3ForConditionalGeneration": ModelSpec(
        package="sglang_neuron.models.kimi_k3",
        kv_cache_dtypes=("fp8_e4m3",),
        quantizations=("compressed-tensors",),
        pipeline_stages=True,
        prefill_context_parallel=True,
        decode_context_parallel=True,
        speculative_algorithms=_k3_decode.SPECULATIVE_ALGORITHMS,
        decode_layouts=_k3_decode.DECODE_LAYOUTS,
        logical_nc=1,
        # at LNC1 two ranks share one physical core's HBM, and the stage scratchpad needs well under one page; the
        # MLA pool of a layer a rank stays below the page (graph_runner.COMPILER_ARGS)
        launch_settings=(LaunchSetting(
            "NEURON_SCRATCHPAD_PAGE_SIZE", "512",
            "the Kimi-K3 stage graphs compile for a 512 MB HBM scratchpad page, which leaves room for the weights "
            "of a stage rank, its pools and the seam rings at LNC1"),),
        chunk_state_checkpoints=True,
        pd_prefill_reposts_failed_writes=True,
        # upstream's Kimi-K3 provider picks SGLang's CUDA attention kernels for a DCP launch and asserts on any other
        # decode backend
        skipped_override_providers=(("sglang.srt.arg_groups.model_overrides.kimi_k3", "_kimi_k3_overrides"),),
        launch_hooks="sglang_neuron.models.kimi_k3.launch",
    ),
}


def checkpoint_config(server_args) -> dict:
    """The checkpoint's config: its config.json for a local directory, else the hub config at the launch's revision.
    SGLang's own ModelConfig is not built yet here, and building it this early would memoize a configuration of
    arguments that are still resolving."""
    config = Path(server_args.model_path) / "config.json"
    if config.is_file():
        return json.loads(config.read_text())
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(server_args.model_path, trust_remote_code=bool(server_args.trust_remote_code),
                                      revision=server_args.revision).to_dict()


def architectures(server_args) -> list:
    """The ``architectures`` of the checkpoint's config."""
    return list(checkpoint_config(server_args).get("architectures") or [])


def checkpoint_quantization(server_args):
    """The quantization method of the checkpoint's quantization_config (a composite config's, else its text config's),
    None for an unquantized checkpoint."""
    config = checkpoint_config(server_args)
    quantization = config.get("quantization_config") or (config.get("text_config") or {}).get("quantization_config")
    return (quantization or {}).get("quant_method")


def served(names) -> list:
    """The architectures among ``names`` that the runtime serves."""
    return [name for name in names if name in MODELS]


def launch_hooks(spec: ModelSpec):
    """The module of a model's own launch rules (ModelSpec.launch_hooks), or None for a model without any."""
    return importlib.import_module(spec.launch_hooks) if spec.launch_hooks else None


def served_spec():
    """The ModelSpec of the model this process serves (the launch's SGLANG_EXTERNAL_MODEL_PACKAGE), or None."""
    return spec_of_package(os.environ.get(MODEL_PACKAGE_ENV, ""))


@functools.lru_cache(maxsize=None)
def _launch_hooks_of(package: str):
    spec = spec_of_package(package)
    return launch_hooks(spec) if spec is not None else None


def served_launch_hooks():
    """The launch rules of the model this process serves, or None (read on every request, so cached)."""
    return _launch_hooks_of(os.environ.get(MODEL_PACKAGE_ENV, ""))


def spec_of_package(package: str):
    """The ModelSpec of the model built from ``package`` (the launch's SGLANG_EXTERNAL_MODEL_PACKAGE), or None."""
    return next((spec for spec in MODELS.values() if spec.package == package), None)


def model_spec(server_args) -> tuple:
    """(architecture, ModelSpec) of the launch's checkpoint, refusing one the runtime does not serve."""
    names = architectures(server_args)
    found = served(names)
    if not found:
        raise ValueError(f"{server_args.model_path} is {names or 'a checkpoint without architectures'}; the Neuron "
                         f"runtime serves {', '.join(sorted(MODELS))}")
    return found[0], MODELS[found[0]]
