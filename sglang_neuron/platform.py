"""The Neuron platform for SGLang; sglang_neuron.activate returns its class path.

It sets the launch's server-argument defaults, refuses the arguments the runtime cannot serve, and gives SGLang the
Neuron classes for the graph runner, attention backend, KV pools and allocator."""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import torch
from sglang.srt.platforms.interface import PlatformCapabilities, SRTPlatform

from sglang_neuron.device import NeuronDeviceMixin
from sglang_neuron.models.catalog import MODEL_PACKAGE_ENV

# Tokens per KV page; the graphs are validated at this size only. The fused sliding-window prefill kernel needs one
# page to hold a sliding layer's whole window.
PAGE_SIZE = 128

logger = logging.getLogger(__name__)


def _int(server_args, name: str, default: int = 1) -> int:
    return int(getattr(server_args, name, default) or default)


def _decode_hicache_opted_in(spec) -> bool:
    """HiCache's host tier opted in on the model's PD decode: its catalog entry names the switch
    (catalog.ModelSpec.decode_hicache_env) and the launch sets it to 1."""
    name = getattr(spec, "decode_hicache_env", "") if spec is not None else ""
    if not name:
        return False
    from sglang_neuron import env_registry

    return env_registry.zero_or_one(name)


def resolve_checkpoint(server_args) -> None:
    """Turn --model-path into a local checkpoint directory before any scheduler starts.

    A Hugging Face hub id is downloaded once, here in the launching process, with SGLang's own download helper, so the
    Hugging Face cache, --download-dir and HF_HUB_OFFLINE all apply. The checkpoint's original/ files are skipped, as
    SGLang's loader skips them. --model-path then points at the local snapshot, which every scheduler reads. The
    tokenizer and the served model name keep the hub id, because SGLang resolved both before the platform defaults
    ran."""
    path = server_args.model_path
    if os.path.isdir(path):
        return
    from sglang.srt.model_loader.weight_utils import download_weights_from_hf

    local = download_weights_from_hf(path, getattr(server_args, "download_dir", None),
                                     ["*.safetensors", "*.json", "*.py"],
                                     revision=getattr(server_args, "revision", None), ignore_patterns=["original/**/*"])
    logger.info("[neuron] --model-path %s is served from its local snapshot %s", path, local)
    server_args.model_path = local


def _storage_extra_config(server_args) -> dict:
    raw = getattr(server_args, "hicache_storage_backend_extra_config", None)
    if not raw:
        return {}
    try:
        config = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except ValueError:
        return {}
    return config if isinstance(config, dict) else {}


def _plugin_storage_backend(server_args) -> Optional[str]:
    """The plugin's HiCache storage backend that --hicache-storage-backend dynamic names, or None: the remote DRAM
    tier (mem_cache/remote_dram/backend.py) or Mooncake Store (mem_cache/mooncake_store.py). Both read and write the
    Neuron host pool's layer-first pages themselves; SGLang's own backends read a page-first host buffer it lacks."""
    from sglang_neuron.mem_cache import storage_tiers
    from sglang_neuron.mem_cache.remote_dram import backend

    if getattr(server_args, "hicache_storage_backend", None) != "dynamic":
        return None
    config = _storage_extra_config(server_args)
    named = (config.get("backend_name"), config.get("module_path"), config.get("class_name"))
    if named == (backend.BACKEND_NAME, backend.MODULE_PATH, backend.CLASS_NAME):
        return backend.BACKEND_NAME
    tier = storage_tiers.MOONCAKE
    if named == (tier["backend_name"], tier["module_path"], tier["class_name"]):
        return tier["backend_name"]
    return None


def _is_remote_dram(server_args) -> bool:
    """--hicache-storage-backend dynamic naming the remote DRAM backend (mem_cache/remote_dram/backend.py)."""
    from sglang_neuron.mem_cache.remote_dram import backend

    return _plugin_storage_backend(server_args) == backend.BACKEND_NAME


def _unserved_speculative(server_args, spec=None) -> bool:
    """An algorithm other than NEURON_MTP and those the model's own speculative worker serves
    (catalog.ModelSpec.speculative_algorithms)."""
    from sglang_neuron.speculative.neuron_mtp import is_neuron_mtp

    algorithm = getattr(server_args, "speculative_algorithm", None)
    return (bool(algorithm) and not is_neuron_mtp(algorithm)
            and str(algorithm).upper() not in getattr(spec, "speculative_algorithms", ()))


def _speculative_flag(spec) -> str:
    """The refusal's name for an unserved algorithm: other than NEURON_MTP, or than the model's own algorithms (and
    NEURON_MTP when its decode runs an MTP step) for a model that declares some."""
    own = tuple(getattr(spec, "speculative_algorithms", ()))
    if not own:
        return "--speculative-algorithm other than NEURON_MTP"
    return "--speculative-algorithm other than " + " or ".join(own + (("NEURON_MTP",) if spec.mtp_host else ()))


def _mtp_without_a_step(server_args, spec) -> bool:
    """NEURON_MTP for a model whose decode runs no MTP step (catalog.ModelSpec.mtp_host), refused at launch rather
    than at its first request."""
    from sglang_neuron.speculative.neuron_mtp import is_neuron_mtp

    algorithm = getattr(server_args, "speculative_algorithm", None)
    return spec is not None and bool(algorithm) and is_neuron_mtp(algorithm) and not spec.mtp_host


def refuse_unserved(server_args, spec=None) -> None:
    """Refuse the arguments the runtime does not serve, naming each one; ``spec`` is the launch's catalog entry
    (catalog.ModelSpec), None for a launch whose model is not known yet. ``spec`` lifts --nnodes for a decode layout over
    that many nodes it declares (serving_policy.declared_decode_layout checks the layout's other flags) and
    --speculative-algorithm for an algorithm its own worker serves."""
    nnodes = _int(server_args, "nnodes")
    # a model that keeps chunk state checkpoints serves HiCache on the chunk-keyed tree, pipelined too: its hybrid MLA
    # and state pools have Neuron host pools (mem_cache.hybrid_host_pool), and SGLang's tree syncs its HiCache
    # decisions down the pipeline
    hybrid_hicache = spec is not None and getattr(spec, "chunk_state_checkpoints", False)
    unserved = {
        "--attn-cp-size above 1": _int(server_args, "attn_cp_size") > 1,
        # one server per node (xPyD runs several single-node servers), except a model's declared multi-node decode
        "--nnodes above 1": nnodes > 1 and not any(layout.nnodes == nnodes
                                                   for layout in getattr(spec, "decode_layouts", ())),
        # the served algorithms are a decode kernel's own MTP step (speculative/neuron_mtp.py), which checks its own
        # arguments, and a model's own speculative worker's; SGLang's workers run their own device ops and draft model
        _speculative_flag(spec): _unserved_speculative(server_args, spec),
        "--speculative-algorithm NEURON_MTP for a model whose decode runs no MTP step": _mtp_without_a_step(server_args,
                                                                                                         spec),
        # the compiled graphs carry no adapter weights: every request would run the base model
        "--enable-lora / --lora-paths": bool(getattr(server_args, "enable_lora", False)
                                             or getattr(server_args, "lora_paths", None)),
        # a forward runs one compiled graph: one request's prefill chunk, or a decode batch
        "--enable-mixed-chunk": bool(getattr(server_args, "enable_mixed_chunk", False)),
        # the graphs route experts inside the compiled step and return no expert ids
        "--enable-return-routed-experts": bool(getattr(server_args, "enable_return_routed_experts", False)),
        # torch-neuronx fixes the set of ranks when the runtime starts
        "--elastic-ep-backend / --elastic-ep-join-mode": bool(getattr(server_args, "elastic_ep_backend", None)
                                                              or getattr(server_args, "ep_join_mode", None)),
        # HiCache's storage tiers are the plugin's own backends, which read the layer-first host pool: the remote DRAM
        # tier (mem_cache/remote_dram) and Mooncake Store (mem_cache/mooncake_store); no host-pool retraction backup
        # on a decode
        "--hicache-storage-backend other than the remote DRAM tier or Mooncake Store (mem_cache/mooncake_store)": bool(
            getattr(server_args, "enable_hierarchical_cache", False)
            and getattr(server_args, "hicache_storage_backend", None)
            and _plugin_storage_backend(server_args) is None),
        # untested on a decode, whose only tree caches the KV its prefills sent; SGLang would still build the pinned
        # host pool and run its per-step TP-wide transfer check. A model whose pools build their own host tier opts its
        # decode in through its catalog entry's switch (ModelSpec.decode_hicache_env).
        "--enable-hierarchical-cache on a --disaggregation-mode decode server":
            bool(getattr(server_args, "enable_hierarchical_cache", False)
                 and getattr(server_args, "disaggregation_mode", None) == "decode")
            and not _decode_hicache_opted_in(spec),
        # untested with pipeline stages; the chunk-keyed tree keys a plain KV pool's pages only (both served for a model
        # that keeps chunk state checkpoints, hybrid_hicache)
        "--enable-hierarchical-cache with --pp-size above 1":
            bool(getattr(server_args, "enable_hierarchical_cache", False)) and _int(server_args, "pp_size") > 1
            and not hybrid_hicache,
        "--enable-hierarchical-cache with --radix-cache-backend neuron-chunk":
            bool(getattr(server_args, "enable_hierarchical_cache", False))
            and getattr(server_args, "radix_cache_backend", None) == "neuron-chunk" and not hybrid_hicache,
        # torch-memory-saver pauses CUDA allocations only; the KV pools and weights are torch-neuronx's
        "--enable-memory-saver": bool(getattr(server_args, "enable_memory_saver", False)),
        # the graphs return sampled tokens, but every forward of an embedding server needs pooled hidden states
        "--is-embedding": bool(getattr(server_args, "is_embedding", False)),
        # the graphs return the sampled tokens and logprobs, no hidden states
        "--enable-return-hidden-states / --return-hidden-states-mode":
            bool(getattr(server_args, "enable_return_hidden_states", False)
                 or getattr(server_args, "return_hidden_states_mode", None)),
        # every model loads its checkpoint's safetensors itself; any other format, or dummy weights, would compile and
        # serve graphs over parameters that were never loaded
        "--load-format other than auto or safetensors":
            (getattr(server_args, "load_format", None) or "auto") not in ("auto", "safetensors"),
        # the runtime always runs its compiled graphs; SGLang's flag would only mark its own decode graphs disabled
        "--disable-cuda-graph": bool(getattr(server_args, "disable_cuda_graph", False)),
        # every phase must run the Neuron backend. SGLang resolves split backends after this hook: two matching ones
        # replace --attention-backend, two different ones make a hybrid, and an MLA backend sets its own page size.
        "--prefill-attention-backend / --decode-attention-backend": any(
            getattr(server_args, name, None) not in (None, "neuron")
            for name in ("prefill_attention_backend", "decode_attention_backend")),
    }
    asked = [name for name, on in unserved.items() if on]
    if asked:
        raise ValueError(f"sglang-neuron does not serve {', '.join(asked)}; launch without it")
    # SGLang's GPU-id arithmetic would put a rank on another server's core, or two ranks on one
    placed = [name for name, on in (("--base-gpu-id", _int(server_args, "base_gpu_id", 0) != 0),
                                    ("--gpu-id-step", _int(server_args, "gpu_id_step") != 1)) if on]
    if placed:
        raise ValueError(f"{' and '.join(placed)}: a Neuron server's ranks run on the NeuronCores "
                         "NEURON_RT_VISIBLE_CORES names, rank i on the i-th; set it to place the server instead")


def check_pipeline_stages(server_args, architecture: str, spec) -> None:
    """Refuse --pp-size above 1 for a model without pipeline stages (ModelSpec.pipeline_stages), before weights load."""
    pp_size = _int(server_args, "pp_size")
    if pp_size > 1 and not spec.pipeline_stages:
        from sglang_neuron.models import catalog

        staged = sorted(name for name, other in catalog.MODELS.items() if other.pipeline_stages)
        raise ValueError(f"--pp-size {pp_size}: {architecture} builds no pipeline stages; the models that do are "
                         f"{', '.join(staged)}")


def resolve_hicache(server_args) -> None:
    """--enable-hierarchical-cache serves through the Neuron host pool (mem_cache/host_pool.py), which keeps a
    layer_first host copy and moves pages itself with host indices: direct I/O and layer_first are set here (SGLang
    defaults to its kernel backend and page_first)."""
    if not getattr(server_args, "enable_hierarchical_cache", False):
        return
    for name, value in (("hicache_io_backend", "direct"), ("hicache_mem_layout", "layer_first")):
        if getattr(server_args, name, None) != value:
            logger.info("[neuron] --%s %s (was %s): the Neuron host pool's", name.replace("_", "-"), value,
                        getattr(server_args, name, None))
            setattr(server_args, name, value)
    tier = _plugin_storage_backend(server_args)
    if tier is not None:
        # the backend moves host pages by host index itself (SGLang's batch_*_v1 interface)
        config = _storage_extra_config(server_args)
        if not config.get("interface_v1"):
            config["interface_v1"] = 1
            server_args.hicache_storage_backend_extra_config = json.dumps(config)
            logger.info("[neuron] the %s HiCache tier reads and writes host pages itself: interface_v1 set", tier)


def check_kv_cache(server_args, architecture: str, spec) -> None:
    """The graphs serve only their page and the model's own KV dtypes: PAGE_SIZE tokens, the pages a model declares
    (catalog.ModelSpec.page_sizes, the first one a launch without --page-size gets), or the page a model's own launch
    rules give for the launch's layout (catalog.ModelSpec.launch_hooks kv_page_size). A launch without --page-size gets
    that page."""
    from sglang_neuron.models import catalog

    hooks = catalog.launch_hooks(spec)
    page = PAGE_SIZE if hooks is None else int(hooks.kv_page_size(server_args))
    pages = spec.page_sizes or (page,)
    if server_args.page_size is None:
        server_args.page_size = pages[0]
    if server_args.page_size not in pages:
        if spec.page_sizes:
            raise ValueError(f"--page-size {server_args.page_size}: {architecture}'s graphs serve pages of "
                             f"{' or '.join(map(str, pages))} tokens; launch with one of them or without it")
        rule = "" if hooks is None else f" for {architecture} at this layout ({spec.launch_hooks}.kv_page_size)"
        raise ValueError(f"--page-size {server_args.page_size}: sglang-neuron serves pages of {page} tokens{rule}; "
                         f"launch with --page-size {page} or without it")
    if server_args.kv_cache_dtype not in spec.kv_cache_dtypes:
        raise ValueError(f"--kv-cache-dtype {server_args.kv_cache_dtype}: {architecture}'s graphs serve "
                         f"{' or '.join(spec.kv_cache_dtypes)}; launch with --kv-cache-dtype {spec.kv_cache_dtypes[0]}")


def default_mem_fraction(server_args, spec) -> None:
    """--mem-fraction-static: the model's default for the launch's role where the launch names none
    (catalog.ModelSpec.mem_fraction_static), and a refusal of a stated value outside (0, 1]. Runs before SGLang
    resolves its own default (arg_groups/platform_hook.py precedes memory_hook.handle_gpu_memory_settings), which
    reserves tp_size x pp_size / 8 GiB for a large parallel size: on a NeuronCore's 18 GiB that resolves below zero for
    a 128-rank server, and a pool sized or checked against it has no budget."""
    from sglang_neuron import serving_policy

    stated = getattr(server_args, "mem_fraction_static", None)
    if stated is None:
        default = dict(spec.mem_fraction_static).get(serving_policy.launch_role(server_args))
        if default is not None:
            server_args.mem_fraction_static = float(default)
        return
    if not 0 < float(stated) <= 1:
        raise ValueError(f"--mem-fraction-static {stated}: the share of each core's HBM that the weights and the KV "
                         "pools take, which leaves the rest to the graphs; launch with a value in (0, 1], or without "
                         "the flag for the model's default")


def apply_model_defaults(server_args, spec) -> None:
    """A model's own launch defaults and process setup, before anything else reads the arguments (its launch rules'
    optional apply_defaults(server_args) and prepare_process(), catalog.ModelSpec.launch_hooks); a model without
    either keeps the launch as it is."""
    from sglang_neuron.models import catalog

    hooks = catalog.launch_hooks(spec)
    if hooks is None:
        return
    if hasattr(hooks, "apply_defaults"):
        hooks.apply_defaults(server_args)
    if hasattr(hooks, "prepare_process"):
        hooks.prepare_process()


def check_model_launch(server_args, architecture: str, spec) -> None:
    """Refuse a launch a model's own launch rules do not serve (catalog.ModelSpec.launch_hooks launch_problems),
    naming every problem, before any scheduler starts; a model without rules passes."""
    from sglang_neuron.models import catalog

    hooks = catalog.launch_hooks(spec)
    if hooks is None:
        return
    problems = hooks.launch_problems(server_args, catalog.checkpoint_config(server_args), os.environ)
    if problems:
        raise ValueError(f"{architecture} cannot launch:\n  - " + "\n  - ".join(problems))


def check_quantization(server_args, architecture: str, spec) -> None:
    """The graphs run the checkpoint's weights as stored and never quantize them. Refuse a --quantization, or a
    checkpoint quantization method, that the model's graphs do not serve."""
    from sglang_neuron.models import catalog

    stated = getattr(server_args, "quantization", None)
    if stated is not None and stated not in spec.quantizations:
        served = " or ".join(spec.quantizations) or "none"
        raise ValueError(f"--quantization {stated}: {architecture}'s graphs run the checkpoint's own weights and "
                         f"quantize nothing (they serve {served}); launch without --quantization")
    method = catalog.checkpoint_quantization(server_args)
    if method is not None and method not in spec.quantizations:
        raise ValueError(f"{server_args.model_path} is quantized with {method}, which {architecture}'s graphs do not "
                         f"serve; serve its unquantized checkpoint")


def apply_model_settings(architecture: str, spec, giga_decode: bool = False, server_args=None) -> None:
    """Point SGLang's model registry at the model's package and set the environment its kernels read at import: the
    model's launch settings (its launch rules' launch_settings(server_args) for a model with launch rules,
    catalog.ModelSpec.launch_hooks, so ``server_args`` is the launch's arguments there), or its giga-kernel decode's on
    that decode. Both are environment settings, so every process the launch starts inherits them. The giga-kernel
    decode refuses a sampler its tail does not serve (spec.giga_decode_samplers: greedy alone where the fused tail
    returns greedy tokens), top logprobs (no tail returns them), and any other flag of the giga kernel in the launch
    environment (catalog.GIGA_KERNEL_FLAG_PREFIX)."""
    from sglang_neuron import sampling, serving_policy
    from sglang_neuron.models import catalog

    if giga_decode:
        samplers = tuple(spec.giga_decode_samplers)
        if sampling.sampler_mode() not in samplers or sampling.MAX_TOP_LOGPROBS:
            what = ("returns greedy tokens only" if samplers == ("greedy",)
                    else f"serves the samplers {', '.join(samplers)} and no logprobs")
            raise ValueError(f"SGLANG_NEURON_SAMPLER={sampling.sampler_mode()}, SGLANG_NEURON_MAX_TOP_LOGPROBS="
                             f"{sampling.MAX_TOP_LOGPROBS}: {architecture}'s giga-kernel decode's tail {what}; launch "
                             f"with SGLANG_NEURON_SAMPLER={' or '.join(samplers)} and SGLANG_NEURON_MAX_TOP_LOGPROBS=0")
        pinned = {setting.name for setting in spec.giga_decode_launch_settings}
        unpinned = sorted(name for name in os.environ
                          if name.startswith(catalog.GIGA_KERNEL_FLAG_PREFIX) and name not in pinned)
        if unpinned:
            raise ValueError(f"{', '.join(unpinned)}: the giga-kernel decode runs the kernel build its DMA-order "
                             f"schedule and its accuracy were validated on, which pins the giga kernel's flags "
                             f"(models/catalog.py); remove these from the launch environment")

    serving_policy.set_launch_setting(MODEL_PACKAGE_ENV, spec.package,
                                      f"sglang-neuron builds {architecture} from {spec.package}")
    hooks = catalog.launch_hooks(spec)
    if giga_decode:
        settings = spec.giga_decode_launch_settings
    elif hooks is not None:
        # a model with launch rules sets the settings of the launch's own layout
        settings = hooks.launch_settings(server_args)
    else:
        settings = spec.launch_settings
    for setting in settings:
        serving_policy.set_launch_setting(setting.name, setting.value, setting.reason)


def apply_speculative_settings(server_args, policy) -> None:
    """Set what SGLang's resolution of a model-served DSPARK launch requires before it runs (handle_speculative_decoding
    runs after the platform defaults; sglang_patches.dspark_server_args lets its device gate accept this platform).
    Under DP attention DSPARK needs the vocabulary-parallel LM head (--enable-dp-lm-head) and SGLang's own MoE dispatch
    (--moe-a2a-backend none: the model's programs route their experts themselves). Every launch verifies a fixed k + 1
    tokens per request (SGLANG_RAGGED_VERIFY_MODE=static)."""
    from sglang_neuron import serving_policy, sglang_tree

    if policy.speculative_algorithm != "DSPARK":
        return
    if sglang_tree.attention_dp_size(server_args) > 1:
        backend = getattr(server_args, "moe_a2a_backend", None) or "none"
        if backend != "none":
            raise ValueError(f"--moe-a2a-backend {backend}: the model's DSPARK programs route their experts inside the "
                             "compiled step; launch with --moe-a2a-backend none or without it")
        if not getattr(server_args, "enable_dp_lm_head", False):
            logger.info("[neuron] --enable-dp-lm-head: SGLang's DSPARK resolution requires it under DP attention")
            server_args.enable_dp_lm_head = True
    serving_policy.set_launch_setting("SGLANG_RAGGED_VERIFY_MODE", "static",
                                      "the model's DSPARK worker verifies a fixed k + 1 tokens per request")


def enable_streams(environ=os.environ) -> None:
    """Set TORCH_NEURONX_ENABLE_STREAMS=1 for the schedulers SGLang spawns from this environment.

    Each scheduler's torch-neuronx reads it at import. Without it every copy runs on the graphs' stream, queued between
    two graphs. The value is forced, not checked: this process has already imported torch-neuronx, whose import writes
    an unset variable back as "0" (torch_neuronx/__init__.py), so a launch's explicit 0 looks the same as no value.
    Forcing it is safe because the launching process runs no graph, and each rank refuses to run without the streams
    (device_io.check_streams)."""
    if environ.get("TORCH_NEURONX_ENABLE_STREAMS") == "0":
        logger.info("[neuron] TORCH_NEURONX_ENABLE_STREAMS=0 (torch-neuronx's value for an unset variable) is set to 1 "
                    "for the schedulers")
    environ["TORCH_NEURONX_ENABLE_STREAMS"] = "1"


def refuse_cpu_fallback(environ=os.environ) -> None:
    """Set TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=1 for the schedulers SGLang spawns from this environment.

    With it, an op torch-neuronx cannot lower to Neuron raises instead of running on the CPU. torch-neuronx's import
    writes its build's default into an unset variable (torch_neuronx/__init__.py, _build_config): 1 in a development
    build, which the plugin was validated on, and 0 in the published release build, which then falls back to the CPU
    silently, behind one warning, at a CPU copy of every input and output. The value is forced, not set by default, for
    the reason ``enable_streams`` gives: this process has already imported torch-neuronx, so the release build's 0 and
    a launch's explicit 0 look the same."""
    if environ.get("TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS") == "0":
        logger.info("[neuron] TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=0 (the torch-neuronx release build's "
                    "value for an unset variable) is set to 1 for the schedulers: an op that does not lower raises")
    environ["TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS"] = "1"


def _host_lnc_default():
    """The degree torch-neuronx's import writes into an unset NEURON_LOGICAL_NC_CONFIG on this host, or None where it
    cannot tell (torch_neuronx/__init__.py). torch-neuronx's answer is the variable itself whenever it is set
    (utils.get_lnc_setting), so it is asked with the variable hidden."""
    saved = os.environ.pop("NEURON_LOGICAL_NC_CONFIG", None)
    try:
        from torch_neuronx.utils import get_lnc_setting

        return str(get_lnc_setting())
    except Exception:  # noqa: BLE001 - no torch-neuronx, or no Neuron device to ask: no default was written either
        return None
    finally:
        if saved is not None:
            os.environ["NEURON_LOGICAL_NC_CONFIG"] = saved


def set_logical_nc_config(degree: int, environ=os.environ) -> None:
    """Set NEURON_LOGICAL_NC_CONFIG for the schedulers SGLang spawns from this environment: the serving policy's degree
    (serving_policy.ServingPolicy.logical_nc), 1 on the giga-kernel decode (serving_policy.giga_decode), whose 128 ranks
    are the node's physical cores, the degree of a model's declared decode layout, else the model's degree
    (catalog.ModelSpec.logical_nc: 2, or 1 for a model whose graphs and kernels are built for single cores).

    The value is set from the layout, not taken from the launch: this process has already imported torch-neuronx, whose
    import writes the host's default into an unset variable (torch_neuronx/__init__.py), so a value equal to that
    default may be the launch's or nobody's, and is replaced. Any other value contradicts the layout and is refused. The
    launching process runs no graph. neuron-ls reports the cores of the degree it runs under, so its cached answer from
    before is dropped."""
    from sglang_neuron.device import neuron_ls

    want = str(degree)
    have = environ.get("NEURON_LOGICAL_NC_CONFIG")
    if have is not None and have != want:
        if have != _host_lnc_default():
            raise ValueError(f"NEURON_LOGICAL_NC_CONFIG={have}: this launch's layout runs {want} (the giga-kernel "
                             "decode's 1, a model's declared decode layout's degree, else the model's degree, "
                             "catalog.ModelSpec.logical_nc); remove it from the launch environment")
        logger.info("[neuron] NEURON_LOGICAL_NC_CONFIG=%s (torch-neuronx's default for this host, or the launch's) is "
                    "set to %s for the schedulers", have, want)
    environ["NEURON_LOGICAL_NC_CONFIG"] = want
    neuron_ls.cache_clear()


def default_decode_batch_sizes(server_args, dp_identity: bool) -> None:
    """Default the decode batch sizes when the launch lists none (--cuda-graph-bs-decode, or decode bs in
    --cuda-graph-config).

    The runtime compiles one decode graph per size in the list (graph_runner.BucketConfig). The default is the powers
    of two below the batch, plus the batch itself, so a smaller batch runs the next graph up instead of the full-batch
    graph. Under the DP identity (serving_policy) a DP-attention decode gets the batch alone, since its ranks run in
    lockstep at that one size.

    The batch is what one scheduler runs: --max-running-requests divided over the attention-DP groups
    (sglang_tree.scheduler_batch; each --dp-size replica runs all of it), else --cuda-graph-max-bs-decode. With
    neither, SGLang sizes the batch from the KV pool after this hook, so SGLang's own decode sizes stand."""
    from sglang.srt.model_executor.cuda_graph_config import Phase, with_phase

    if (getattr(server_args, "cuda_graph_bs_decode", None) is not None
            or (Phase.DECODE, "bs") in getattr(server_args, "_cuda_graph_config_locked", ())):
        return
    from sglang_neuron import sglang_tree
    from sglang_neuron.utils.bucket_utils import batch_ladder

    config = server_args.cuda_graph_config
    batch = sglang_tree.scheduler_batch(server_args)
    if batch is None and config.decode.max_bs is not None:
        batch = int(config.decode.max_bs)
    if batch is None:
        return
    server_args.cuda_graph_config = with_phase(config, Phase.DECODE, bs=[batch] if dp_identity else batch_ladder(batch),
                                               max_bs=batch)


def _check_plugin_env(server_args) -> None:
    """Parse the plugin's own settings now, so a bad value stops the launch before any scheduler starts. The profile
    rank check reads the rank layout from ``server_args``."""
    from sglang_neuron import env_registry

    env_registry.refuse_unknown()
    # before sampling, whose import reads it with a bare int()
    env_registry.non_negative_int("SGLANG_NEURON_MAX_TOP_LOGPROBS")
    from sglang_neuron import envs, sampling
    from sglang_neuron.distributed import pp_boundary_dump
    from sglang_neuron.model_executor import host_trace, profiling

    sampling.return_original_logprob()
    if sampling.sampler_mode() == "topk_topp" and sampling.MAX_TOP_LOGPROBS > sampling.TOP_K_MAX:
        raise ValueError(f"SGLANG_NEURON_MAX_TOP_LOGPROBS={sampling.MAX_TOP_LOGPROBS}: the topk_topp sampler returns "
                         f"the top logprobs from its filter pool of {sampling.TOP_K_MAX} tokens")
    host_trace.traced_ranks()
    profiling.serve_window()
    envs.profile_ranks()
    profiling.refuse_overlap(_int(server_args, "tp_size"), _int(server_args, "pp_size"))
    envs.profile_device_mode()
    envs.mla_pool_layout()
    pp_boundary_dump.send_count()


# SGLang sources the host sampler relies on, md5-pinned against SGLang 43881a7 (source_guard), checked only when it runs
_HOST_SAMPLER_SOURCES_MD5 = {
    # seeds a batch only under --enable-deterministic-inference, so validate_device_sampling_request refuses a seed
    ("sglang.srt.sampling.sampling_batch_info", "SamplingBatchInfo.from_schedule_batch"):
        "1e3a5c2176a0bb1fa8a0a9c2c89980e5",
    # a grammar masks the host logits through xgrammar's CPU branch
    ("sglang.srt.constrained.xgrammar_backend", "XGrammarGrammar.apply_vocab_mask"): "7540419147fbc67aff1df4347aaffe93",
    # strict thinking's backend-level mask has no CPU branch, so check_host_sampler refuses it
    ("sglang.srt.constrained.xgrammar_backend", "XGrammarGrammarBackend.apply_vocab_mask"):
        "c2a189547a55c9550e55fff746284e5f",
}


def megakernel_serving(server_args, spec):
    """The model's catalog.MegakernelServing when the launch serves its whole-model megakernel (the model declares one
    and the launch passes --enable-nvidia-megakernel), else None."""
    megakernel = getattr(spec, "megakernel", None)
    if megakernel is None or not getattr(server_args, "enable_nvidia_megakernel", False):
        return None
    return megakernel


def host_sampler_unserved(server_args, megakernel=None, spec=None) -> list:
    """The options of this launch the host sampler (SGLANG_NEURON_SAMPLER=host) does not serve; [] if none.
    ``megakernel``: the launch's megakernel_serving. ``spec``: the launch's catalog entry (catalog.ModelSpec), None for
    a model not known here; a model whose runtime takes only some sampler modes (ModelSpec.device_samplers), the host
    sampler not among them, is not served."""
    from sglang_neuron import sampling, serving_policy

    role = str(getattr(server_args, "disaggregation_mode", "null") or "null")
    samplers = tuple(getattr(spec, "device_samplers", ()) or ())
    unserved = {
        # the model's own runtime refuses the host sampler at its graph build (models/qwen35/runtime.py)
        f"this model, whose runtime takes only the {' or '.join(samplers)} sampler":
            bool(samplers) and sampling.HOST_SAMPLER not in samplers,
        # PD stages run async and a decode's last stage relays its tokens on device; the aggregated case is untested
        "--pp-size above 1": _int(server_args, "pp_size") > 1,
        # its ranks run in lockstep without a per-step exchange, which needs async execution (serving_policy)
        "a decode with DP attention": serving_policy.runs_dp_identity(
            server_args, role, megakernel is not None and megakernel.dp_steps_from_exchange),
        # the width of the device sampler's top logprobs; SGLang's sampler returns as many as a request asks for
        "SGLANG_NEURON_MAX_TOP_LOGPROBS above 0": sampling.MAX_TOP_LOGPROBS > 0,
        # strict thinking masks with XGrammarGrammarBackend.apply_vocab_mask, which takes device logits only
        "--enable-strict-thinking": bool(getattr(server_args, "enable_strict_thinking", False)),
    }
    return [name for name, on in unserved.items() if on]


def check_host_sampler(server_args, megakernel=None, spec=None) -> None:
    """Refuse a host-sampler launch (SGLANG_NEURON_SAMPLER=host) that asks for what it does not serve, or whose model's
    runtime does not take the host sampler (``spec``, host_sampler_unserved). SGLang's sampler reads each step's logits
    after the step, so the launch runs without async execution."""
    from sglang_neuron import sampling
    from sglang_neuron.sglang_patches import source_guard

    if sampling.sampler_mode() != sampling.HOST_SAMPLER:
        return
    source_guard.check_targets(_HOST_SAMPLER_SOURCES_MD5, __name__)
    asked = host_sampler_unserved(server_args, megakernel, spec)
    if asked:
        raise ValueError(f"SGLANG_NEURON_SAMPLER={sampling.HOST_SAMPLER} is not served with {', '.join(asked)}; "
                         "launch with a device sampler, or without these")
    # refused rather than set: serving_policy and runtime_init.check_overlap_schedule both read the launch's own flag
    if not getattr(server_args, "disable_overlap_schedule", False):
        raise ValueError(f"SGLANG_NEURON_SAMPLER={sampling.HOST_SAMPLER}: SGLang's sampler reads each step's logits "
                         "after the step, so the launch runs without async execution; launch with "
                         "--disable-overlap-schedule")


class NeuronSRTPlatform(SRTPlatform, NeuronDeviceMixin):
    # Only graph capture, for the decode graph runner. No Triton kernels, piecewise prefill graphs, HiCache device
    # kernels or PyNccl communicators.
    capabilities = PlatformCapabilities(graph_capture=True)

    def support_mamba_cache_extra_buffer(self) -> bool:
        """Upstream's mamba extra buffer (the tracking slots its radix cache adopts as state checkpoints) is served for
        a model whose prefill graph writes the tracking slot at every forward's end, on the chunk-keyed tree, where
        every forward ends on a boundary the tree keys (ServingPolicy.chunk_state_checkpoints). SGLang asks after
        apply_server_args_defaults exported the launch's policy."""
        from sglang_neuron import serving_policy

        policy = serving_policy.current()
        # the kimi-prompt backend tracks the end state of a forward only (kimi_prompt_backend.track_entry)
        return policy.chunk_state_checkpoints or policy.kimi_prompt_cache

    def apply_server_args_defaults(self, server_args) -> None:
        from sglang_neuron import cpu_placement, disaggregation, sampling, serving_policy
        from sglang_neuron.backend import port_hold
        from sglang_neuron.distributed import pipeline
        from sglang_neuron.models import catalog
        from sglang_neuron import general_plugin
        from sglang_neuron.sglang_patches import chunk_keyed_tree, tokenizer_worker_plugins

        general_plugin.check_launch()
        resolve_checkpoint(server_args)
        architecture, spec = catalog.model_spec(server_args)
        apply_model_defaults(server_args, spec)
        # before anything reads it: a pipelined PD prefill's admission when the launch leaves it unset
        pipeline.default_pp_prefill_admission(server_args, spec.prefill_compute_ahead)
        chunk_keyed_tree.default_states_per_path(server_args, spec)
        default_mem_fraction(server_args, spec)
        refuse_unserved(server_args, spec)
        resolve_hicache(server_args)
        check_kv_cache(server_args, architecture, spec)
        chunk_keyed_tree.validate_launch(server_args, architecture, spec)
        check_quantization(server_args, architecture, spec)
        _check_plugin_env(server_args)
        megakernel = megakernel_serving(server_args, spec)
        check_host_sampler(server_args, megakernel, spec)
        check_pipeline_stages(server_args, architecture, spec)
        check_model_launch(server_args, architecture, spec)
        # DCP is served only on the PD decode of a model that declares it; refuse the rest before a scheduler admits
        # against SGLang's DCP-widened capacity
        from sglang_neuron.distributed import decode_context_parallel

        decode_context_parallel.validate_dcp_launch(server_args, architecture, spec)
        policy = serving_policy.resolve(server_args, prefill_holds_prior=spec.prefill_holds_prior,
                                        model_has_giga_decode=spec.giga_decode,
                                        giga_decode_dp=spec.giga_decode_dp,
                                        scratchpad_page_mb=spec.scratchpad_page_mb,
                                        model_computes_ahead=spec.prefill_compute_ahead,
                                        decode_layouts=spec.decode_layouts,
                                        speculative_algorithms=spec.speculative_algorithms,
                                        chunk_state_checkpoints=spec.chunk_state_checkpoints,
                                        logical_nc_config=spec.logical_nc, megakernel=megakernel)
        pipeline.validate_pp_launch(server_args, policy)
        from sglang_neuron.mem_cache import hybrid_swa

        hybrid_swa.configure(server_args, architecture, spec, policy)
        apply_model_settings(architecture, spec, policy.giga_decode, server_args)
        serving_policy.export(policy)
        apply_speculative_settings(server_args, policy)
        default_decode_batch_sizes(server_args, policy.dp_identity)
        enable_streams()
        refuse_cpu_fallback()
        set_logical_nc_config(policy.logical_nc)
        disaggregation.apply_launch_settings(server_args)
        # every rank this launch starts offsets its collectives' bootstrap port by this value. Only node 0 holds the
        # root port, so on a server over several nodes the other nodes take node 0's root at their runtime start
        # (runtime_init.ccom_rendezvous)
        if "SGLANG_NEURON_CCOM_PORT_OFFSET" not in os.environ and _int(server_args, "node_rank", 0) == 0:
            from sglang_neuron.model_executor import runtime_init

            os.environ["SGLANG_NEURON_CCOM_PORT_OFFSET"] = str(runtime_init.launch_ccom_port_offset(server_args))
        # this launch's processes share torch rendezvous files with each other and with no other server
        port_hold.export_launch_key()
        tokenizer_worker_plugins.configure(server_args)
        cpu_placement.leave_reserved_cpus_free()
        cpu_placement.bind_to_numa(server_args)

        # SGLang's tensors live on the host; the plugin owns the rank's NeuronCore
        if getattr(server_args, "device", None) in (None, "cuda", "neuron"):
            server_args.device = "cpu"
        server_args.attention_backend = "neuron"
        server_args.sampling_backend = "pytorch"
        # on SGLang's "cpu" device the decode graph runner is built only with torch.compile on, whatever decode graph
        # backend resolves, even a PD prefill server's disabled one (cuda_graph_setup.capture_decode_graph)
        server_args.enable_torch_compile = True
        # the device sampler refuses constrained requests at admission; the host sampler uses SGLang's grammar backend
        if sampling.sampler_mode() != sampling.HOST_SAMPLER:
            server_args.grammar_backend = "none"
        # SGLang's idle mem-leak check counts pool slots the NEFF still holds
        # between steps; make it a warning.
        os.environ.setdefault("SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE", "0")

    def supports_decode_context_parallel(self) -> bool:
        """Read only by the plugin's rebuild of initialize_model_parallel (sglang_patches.dcp_groups), which then builds
        the DCP groups; distributed.decode_context_parallel refuses the DCP launches the runtime does not serve."""
        return True

    def get_default_attention_backend(self) -> str:
        return "neuron"

    # speculative decoding: the served model's own worker (model_executor.spec_worker)
    def supports_speculative_algorithm(self, algorithm: str) -> bool:
        """Whether the launch's model serves ``algorithm`` with its own worker: the algorithm the serving policy
        recorded (catalog.ModelSpec.speculative_algorithms), which the platform defaults resolve before SGLang resolves
        the speculative arguments. Every other speculative launch was refused there."""
        from sglang_neuron import serving_policy

        served = serving_policy.current().speculative_algorithm
        return bool(served) and str(algorithm).upper() == served

    def get_speculative_worker_cls(self, algorithm: str):
        """SGLang's built-in speculative workers do not run on Neuron: every algorithm gets the platform's worker, which
        builds the loaded model's own worker or refuses."""
        from sglang_neuron.model_executor.spec_worker import NeuronSpeculativeWorker

        return NeuronSpeculativeWorker

    def get_graph_runner_cls(self) -> type:
        from sglang_neuron.model_executor.graph_runner import NeuronGraphRunner

        return NeuronGraphRunner

    def get_kv_pool_cls(self, *, kind):
        """MHA gets a pool in the kernels' block layout, MLA gets SGLang's MLA pool on the NeuronCore. SGLang asks for
        every kind each time it builds the pools; None keeps SGLang's own class for the rest."""
        if kind == "mha":
            from sglang_neuron.mem_cache.memory_pool import NeuronMHATokenToKVPoolKind

            return NeuronMHATokenToKVPoolKind
        if kind == "mla":
            from sglang_neuron.envs import mla_pool_layout

            if mla_pool_layout() == "paged":
                # opt-in: the paged latent pool (mem_cache/mla_pool.py), whose zero-on-free pages and HiCache host
                # pool (mem_cache/mla_host_pool.py) the MLA / DSA models share
                from sglang_neuron.mem_cache.mla_pool import NeuronMLATokenToKVPoolKind

                return NeuronMLATokenToKVPoolKind
            from sglang_neuron.mem_cache.memory_pool import NeuronMLATokenToKVPool

            return NeuronMLATokenToKVPool
        return None

    # megakernels (sglang_patches/megakernel_platform.py; docs/design/megakernel.md, Seams)
    def get_megakernel_target(self) -> str:
        """The device target megakernel specs are matched on (Requirements.isas, register(..., isa=...)): the
        platform target's family (trn2, trn3, ...), from NEURON_PLATFORM_TARGET_OVERRIDE or the runtime's instance
        family; static, so the tokenizer process answers it too."""
        from sglang_neuron.backend.stack import get_platform_target, normalize_target_family

        return normalize_target_family(get_platform_target())

    def install_megakernel_dispatch(self, *, model, built, binding):
        """The whole-step dispatch of a built megakernel: the megakernel host's (sglang_neuron.megakernel.host
        create_dispatch), a callable with model.forward's signature that EagerRunner runs for every covered step
        (sglang_patches/megakernel_eager_dispatch.py). The upstream model.forward wrapper (CUDA graphs) is never
        used on this platform, so a launch without the host is refused."""
        try:
            from sglang_neuron.megakernel.host import create_dispatch
        except ImportError as e:
            raise RuntimeError(f"--enable-nvidia-megakernel on Neuron needs the megakernel host "
                               f"(sglang_neuron.megakernel.host.create_dispatch): {e}") from None
        dispatch = create_dispatch(model=model, built=built, binding=binding)
        if not callable(dispatch):
            raise TypeError(f"sglang_neuron.megakernel.host.create_dispatch returned {type(dispatch).__name__}, "
                            "not a whole-step dispatch callable")
        return dispatch

    def activate_megakernel_pool_layout(self, pool_layout) -> None:
        """Accept a built megakernel's pool layout: an object with pool_class(kind) (the KV pool class of each kind
        its model builds, ValueError for another) and a state_pool_class. Validation only: each model's configurator
        reads its own built object's layout (sglang_patches/megakernel_pool_layout.py), never a process-global one."""
        if not callable(getattr(pool_layout, "pool_class", None)) or not hasattr(pool_layout, "state_pool_class"):
            raise NotImplementedError(f"the Neuron platform builds pools in a layout with pool_class(kind) and a "
                                      f"state_pool_class, not in {pool_layout!r}")

    def get_megakernel_kv_pool_cls(self, *, kind: str, pool_layout) -> type:
        self.activate_megakernel_pool_layout(pool_layout)
        return pool_layout.pool_class(kind)

    def get_megakernel_state_pool_cls(self, *, pool_layout) -> Optional[type]:
        self.activate_megakernel_pool_layout(pool_layout)
        return pool_layout.state_pool_class

    def get_paged_allocator_cls(self) -> type:
        """The platform's paged allocator, or the subclass the model declared while it was built
        (model_executor.kv_layout.KVLayout.paged_allocator_class)."""
        from sglang_neuron.mem_cache.allocator import NeuronPagedTokenToKVPoolAllocator
        from sglang_neuron.model_executor import kv_layout

        declared = kv_layout.current().paged_allocator_class
        if declared is None:
            return NeuronPagedTokenToKVPoolAllocator
        cls = declared()
        if not (isinstance(cls, type) and issubclass(cls, NeuronPagedTokenToKVPoolAllocator)):
            raise TypeError(f"the model's paged allocator class {cls!r} is not a NeuronPagedTokenToKVPoolAllocator "
                            "subclass, which holds a freed page until the launches that read it have run")
        return cls

    def get_quantization_config(self, quantization: str):
        # MXFP4 experts and a Quark export's weights run in the model's own kernels; SGLang gets an inert config for
        # the name (check_quantization refuses either name for a model that does not list it)
        if quantization in ("mxfp4", "quark"):
            from sglang_neuron.quant import passthrough_config

            return passthrough_config(quantization)
        return None

    def get_compile_backend(self, mode: str | None = None) -> str:
        # SGLang's own compiled helpers run on host tensors here; the serving graphs compile through the graph runner
        return "eager"

    def get_piecewise_backend_cls(self) -> type:
        raise NotImplementedError("piecewise graphs are not served by sglang-neuron")

    def get_group_coordinator_device(self, local_rank: int) -> torch.device:
        return torch.device("cpu")

    def get_dispatch_key_name(self) -> str:
        return "PrivateUse1"

    def init_backend(self) -> None:
        # the stack is set up when the plugin registers (process_init.init_stack)
        return None
