"""The plugin's SGLang hooks. SGLang's plugin loader calls register() in every process of a Neuron launch
(``SGLANG_PLATFORM=neuron``).

The launch hooks go into every process, the launching one included, because the launching process is where SGLang
resolves the server arguments and admits requests. The runtime hooks go only into the processes the launch starts,
because they follow the serving policy: the launching process resolves it in its platform defaults, after its plugins
have loaded, and exports it to every process it starts (serving_policy)."""

import functools
import logging
import os
import pkgutil

from sglang.srt.plugins.hook_registry import HookRegistry as _SGLangHookRegistry
from sglang.srt.plugins.hook_registry import HookType

logger = logging.getLogger(__name__)


class ResolvingHookRegistry:
    """SGLang's HookRegistry, but each target is resolved as it is registered.

    SGLang resolves targets only when it applies the hooks, after every plugin has loaded, and it logs and skips a
    target that does not resolve, so the launch would run without that hook. Resolving here makes a target that an
    SGLang change removed raise inside the plugin's registration, which fails closed (check_launch, register). A second
    REPLACE of one target by another function is refused the same way: SGLang only warns and keeps the last one, which
    would silently drop the other."""

    def __init__(self, registry):
        self._registry = registry

    def register(self, target: str, hook, hook_type) -> None:
        try:
            pkgutil.resolve_name(target)
        except (ImportError, AttributeError, ValueError) as error:
            raise RuntimeError(f"{target}: this SGLang has no such function to hook ({type(error).__name__}: {error}); "
                               "sglang-neuron is written against SGLang 43881a7") from error
        if hook_type == HookType.REPLACE:
            name = _hook_name(hook)
            other = next((_hook_name(h) for kind, h, _source in getattr(self._registry, "_hooks", {}).get(target, ())
                          if kind == HookType.REPLACE and _hook_name(h) != name), None)
            if other is not None:
                raise RuntimeError(f"{target}: {name} would replace {other}, which already replaces it; one REPLACE "
                                   "a target")
        self._registry.register(target, hook, hook_type)

    def __getattr__(self, name):
        return getattr(self._registry, name)


def _hook_name(hook) -> str:
    """A hook by its definition (a factory's closures share one), so registering the same hook again is not a second."""
    return f"{getattr(hook, '__module__', '?')}.{getattr(hook, '__qualname__', repr(hook))}"


# every hook the plugin and its modules register goes through this registry
HookRegistry = ResolvingHookRegistry(_SGLangHookRegistry)

# the attention backend the plugin registers (SGLang's --attention-backend choice)
ATTN_BACKEND_NAME = "neuron"

_TOKENIZER = "sglang.srt.managers.tokenizer_manager.TokenizerManager"
_CONTROL = "sglang.srt.managers.tokenizer_control_mixin.TokenizerControlMixin"
# the model's weights live in its compiled graphs, loaded once
_WEIGHT_UPDATES = (
    f"{_TOKENIZER}.update_weights_from_disk",
    f"{_CONTROL}.update_weights_from_tensor",
    f"{_CONTROL}.update_weights_from_distributed",
    f"{_CONTROL}.update_weights_from_ipc",
    f"{_CONTROL}.init_weights_update_group",
    f"{_CONTROL}.init_weights_send_group_for_remote_instance",
)
# Upstream functions whose logic the hooks below skip, replace or rewrite, pinned by md5 against SGLang 43881a7
# (source_guard). If one changes upstream, the launch stops. Hooks that call the original and only add behaviour
# around it are not pinned.
_LAUNCH_SOURCES_MD5 = {
    # an abort without logprob rows is formatted by add_logprob_to_meta_info instead
    ("sglang.srt.managers.tokenizer_manager", "TokenizerManager.convert_logprob_style"):
        "ec11343974cdefc7f5c077f9c03da245",
    # handles an abort itself and sends the scheduler only in_place and retract pauses, which a pipelined server refuses
    ("sglang.srt.managers.tokenizer_manager", "TokenizerManager.pause_generation"): "201c5381ee9e863303c8b562c79b16e2",
    # its first argument is the timeout a pipelined server refuses
    ("sglang.srt.managers.tokenizer_control_mixin", "TokenizerControlMixin.flush_cache"):
        "c01ade18185dfb7a995460013edecc1f",
}
_RUNTIME_SOURCES_MD5 = {
    # every use_* argument is a device communicator, turned off
    ("sglang.srt.distributed.parallel_state", "init_model_parallel_group"): "fe3cc1fab307a27a8e48c3ee6657bbec",
    # replaced by the NeuronCore's HBM, in the same units and with the same MIN over cpu_group
    ("sglang.srt.utils.common", "get_available_gpu_memory"): "7ddd40c3e8ba9f559e516b07771ed49e",
    # replaced by the torch copy it dispatches to off CUDA
    ("sglang.srt.layers.dp_attention", "memcpy_func"): "3dada4b81a6675de576fdc5fa4cdf287",
    ("sglang.srt.layers.dp_attention", "memcpy_cpu"): "6ab5f4cac56f480f6e82c4ffdb873a91",
    # skipped: they only bind and re-align for SGLang's CPU engine
    ("sglang.srt.distributed.bootstrap", "_bind_threads_if_cpu"): "1e52fe867f47edcd9ca7660377258941",
    ("sglang.srt.model_executor.model_runner", "adjust_config_with_unaligned_cpu_tp"):
        "0b961c0c5f169b47ef54864f93d86ea3",
    # skipped when the graph sampled on device (its tokens and logprobs are returned instead); runs on the host sampler
    ("sglang.srt.model_executor.model_runner", "ModelRunner.sample"): "0f4e42b94e8efa0ee418ac2ca1c9a297",
    # skipped after the forward's first request, along with preemption for a request the forward refuses
    ("sglang.srt.managers.schedule_policy", "PrefillAdder.add_one_req"): "7307599dd69fb761692018e8f1234ceb",
    ("sglang.srt.managers.schedule_policy", "PrefillAdder.preempt_to_schedule"): "4786f82c61c5ddc9ada287fc0fd0658e",
    # it would replace a request's pending abort with a successful finish, so the hook applies the abort first
    ("sglang.srt.disaggregation.prefill", "SchedulerDisaggregationPrefillMixin.process_disagg_prefill_inflight_queue"):
        "5b49bcb8e522b632073c48f2c075fd7f",
    # replaced: the requests go over the TP group's message queue instead of its broadcast
    ("sglang.srt.managers.scheduler_components.request_receiver",
     "SchedulerRequestReceiver._broadcast_reqs_across_ranks"): "2c89050881db2b90eb39e316e32cd8c6",
    # replaced for a model that builds its own KV pools: alloc_memory_pool takes its budgets, pools and allocator
    ("sglang.srt.mem_cache.kv_cache_configurator", "KVCacheConfigurator.configure"): "26002a8418c48534b2ae0f4dc60a703e",
    ("sglang.srt.model_executor.model_runner", "ModelRunner.alloc_memory_pool"): "b8acb227f87b6ce7940c4343d11d7c9b",
    # skipped for a model that runs no step through SGLang's attention backends: the hybrid linear-attention wrapper
    # and the decode graphs' gather alignment (_no_linear_attention_backend, _no_decode_gathers)
    ("sglang.srt.layers.attention.attention_registry", "attn_backend_wrapper"): "3c632b6bde44af715ca38957fd7a8e4d",
    ("sglang.srt.model_executor.model_runner", "ModelRunner.decode_graph_gather_requirements"):
        "2fbdd971ac0802feaa71f9435cd8348c",
    # the readers of the configurator fields _without_draft_kv_pool clears: the DFLASH draft cell size, the fused draft
    # pool's decision and the draft pool's budget
    ("sglang.srt.mem_cache.kv_cache_configurator", "KVCacheConfigurator.__post_init__"):
        "dc992a83c2deeef7bd47e390487058f8",
    ("sglang.srt.mem_cache.kv_cache_configurator", "KVCacheConfigurator._fused_draft_decision"):
        "f75de5ff155c59d64eb7ffb7dd794780",
    ("sglang.srt.model_executor.pool_configurator", "_dflash_draft_cell_size"): "f4f4333d3e77ae141d7dce96d7f07cdc",
    # wrapped for a model that allocates HBM beside SGLang's pools: the KV budget it returns
    # (_less_the_model_pools_beside_sglang)
    ("sglang.srt.mem_cache.kv_cache_configurator", "KVCacheConfigurator._profile_available_bytes"):
        "fb3fed1784bea1f39ccd4a5059b8d81b",
}


def _refuse_out_of_vocabulary(input_ids, vocab_size: int) -> None:
    """A token id outside the vocabulary would index past the embedding table. SGLang checks only a request's logprob
    token ids against the vocabulary, and a prompt may arrive as token ids (one list, or one list per prompt)."""
    if not input_ids:
        return
    for ids in input_ids if isinstance(input_ids[0], list) else (input_ids,):
        if ids and (min(ids) < 0 or max(ids) >= vocab_size):
            bad = next(t for t in ids if not 0 <= t < vocab_size)
            raise ValueError(f"input_ids holds token id {bad}, outside the model's vocabulary [0, {vocab_size})")


def _validate_request(original_fn, self, obj, input_ids, *args, **kwargs):
    """AROUND TokenizerManager._validate_one_request: a request the device sampler or the prefill graph cannot serve
    is answered with an error at admission, never served with an option silently dropped."""
    ret = original_fn(self, obj, input_ids, *args, **kwargs)
    from sglang_neuron import sampling

    if getattr(obj, "input_embeds", None) is not None or getattr(obj, "positional_embed_overrides", None) is not None:
        raise ValueError("sglang-neuron's graphs embed the prompt's token ids on device and take no input_embeds or "
                         "positional_embed_overrides; send text or input_ids")
    _refuse_out_of_vocabulary(input_ids, self.model_config.vocab_size)
    server_args = self.server_args
    sampling.validate_pd_request(obj, server_args)
    host_serves = self.__dict__.get("_neuron_host_serves")
    if host_serves is None:
        from sglang_neuron import platform
        from sglang_neuron.models import catalog

        # whether these arguments and the served model could run the host sampler, so a refusal can suggest it
        host_serves = self.__dict__["_neuron_host_serves"] = not platform.host_sampler_unserved(
            server_args, spec=catalog.served_spec())
    sampling.validate_device_sampling_request(obj, sampling.sampler_mode(),
                                              preferred=getattr(self, "preferred_sampling_params", None),
                                              deterministic=bool(server_args.enable_deterministic_inference),
                                              host_serves=host_serves)
    from sglang_neuron.speculative import neuron_mtp

    neuron_mtp.validate_request(obj, getattr(self, "preferred_sampling_params", None), server_args)
    from sglang_neuron.models import catalog

    hooks = catalog.served_launch_hooks()
    if hooks is not None and hasattr(hooks, "validate_request"):
        # the served model's own admission rules (catalog.ModelSpec.launch_hooks)
        hooks.validate_request(obj, getattr(self, "preferred_sampling_params", None), server_args,
                               self.model_config.vocab_size)
    if getattr(obj, "return_logprob", False):
        start = getattr(obj, "logprob_start_len", -1)
        if isinstance(start, list):
            raise ValueError("logprob_start_len must be a single integer for a single request")
        if start is not None and start != -1 and start < len(input_ids or ()):
            raise ValueError("prompt (input-token) logprobs are not supported on Neuron: the prefill graph returns "
                             "last-token logits only; use the default logprob_start_len=-1")
    return ret


def _refuse_weight_update(original_fn, self, *args, **kwargs):
    from fastapi import HTTPException

    raise HTTPException(status_code=400, detail=f"{original_fn.__name__} is not supported by sglang-neuron: the "
                                                "model's weights are loaded into its compiled graphs once")


def _refuse_pause_under_pp(original_fn, self, obj, *args, **kwargs):
    """AROUND TokenizerManager.pause_generation: a pipelined server refuses an in_place or retract pause with HTTP 400.

    SGLang's PP loops never read the pause flag, so in_place would leave the stages generating. A retract empties only
    the current micro-batch, and under the overlap schedule it pops a result_queue the overlap-order PP decode loop
    never fills, freeing KV of steps still in flight. An abort is safe: the tokenizer aborts every request, which the
    scheduler finds in every micro-batch, and waits for them to finish. Tokenizer workers have their own pause, but a
    pipelined launch runs none."""
    pp_size = self.server_args.pp_size
    if pp_size > 1 and obj.mode in ("in_place", "retract"):
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail=f"pause_generation mode={obj.mode} is not supported with "
                                                    f"--pp-size {pp_size}: SGLang's pipeline-parallel loops do not "
                                                    "pause; use mode=abort")
    return original_fn(self, obj, *args, **kwargs)


def _refuse_timed_flush_under_pp(original_fn, self, timeout_s=None, *args, **kwargs):
    """AROUND TokenizerControlMixin.flush_cache: a pipelined server refuses a flush with a timeout with HTTP 400.

    Each stage waits until it is idle itself, and its deadline markers (sglang_patches/flush_deadline_markers) follow
    its own leader's clock, so one stage could flush while another times out and their prefix caches would differ. An
    untimed flush is allowed; each stage flushes only when idle, so send it to an idle server."""
    pp_size = self.server_args.pp_size
    # SGLang serves a timeout of 0 or less as no timeout
    if pp_size > 1 and (timeout_s or 0) > 0:
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail=f"flush_cache with timeout={timeout_s} is not supported with "
                                                    f"--pp-size {pp_size}: each stage would time it out on its own; "
                                                    "flush without a timeout once the server is idle")
    return original_fn(self, timeout_s, *args, **kwargs)


# the error that kept the launch hooks from installing; check_launch refuses the launch with it
_launch_hooks_error = None


def register_launch_hooks() -> None:
    """A hook that fails to install (its SGLang target changed) is recorded, not raised, because SGLang's plugin loader
    would only log the exception. check_launch raises it later, from the platform defaults, where the user sees it."""
    global _launch_hooks_error
    try:
        _register_launch_hooks()
    except Exception as exc:  # noqa: BLE001 - kept for check_launch
        _launch_hooks_error = exc
        logger.error("[neuron] the launch hooks did not install (%s: %s); the launch is refused",
                     type(exc).__name__, exc)


def check_launch() -> None:
    """Refuse the launch if its launch hooks did not install, or if SGLANG_PLATFORM is not neuron.

    This runs from the platform's server-args defaults, where an exception reaches the user; SGLang only logs one
    raised while it loads its plugins. On a Neuron host SGLang picks the Neuron platform even without SGLANG_PLATFORM,
    since it is the only platform plugin that activates there. The plugin installs its hooks only under
    SGLANG_PLATFORM=neuron, so such a launch would otherwise fail later with an error that names nothing."""
    if _launch_hooks_error is not None:
        raise RuntimeError(f"sglang-neuron's SGLang patches did not install: {_launch_hooks_error}") \
            from _launch_hooks_error
    if os.environ.get("SGLANG_PLATFORM") != "neuron":
        raise RuntimeError("SGLang resolved the Neuron platform without SGLANG_PLATFORM=neuron, and sglang-neuron "
                           "installs its hooks only under it: launch with SGLANG_PLATFORM=neuron")


def _register_launch_hooks() -> None:
    from sglang_neuron.managers.logprob_errors import convert_logprob_style
    from sglang_neuron.sglang_patches import (bootstrap_listen_backlog, crash_diagnostics_platform,
                                              decode_radix_prefill_chunks, detokenizer_sender, gpt_oss_backend_check,
                                              http_fatal_sigquit, neuron_system_profile, pd_fake_prefill,
                                              quantization_registry, source_guard)

    source_guard.check_targets(_LAUNCH_SOURCES_MD5, __name__)

    quantization_registry.register()
    # --speculative-algorithm NEURON_MTP resolves in every process that parses the server arguments
    from sglang_neuron.speculative import neuron_mtp

    neuron_mtp.register()
    # a prefill's bootstrap server runs beside the tokenizer manager, in the launching process; it also relays each
    # prefill rank's chunk grid to the decodes
    bootstrap_listen_backlog.register(HookRegistry, HookType.REPLACE)
    decode_radix_prefill_chunks.register_bootstrap_server(HookRegistry, HookType.AROUND)
    # the launching process starts the detokenizer workers and registers its hooks before it resolves the policy
    detokenizer_sender.register_worker_launch(HookRegistry, HookType.REPLACE)
    # SGLang resolves the speculative arguments in the launching process, after the platform defaults: its DSPARK
    # device check accepts this platform for a model whose own worker serves DSPARK (the model's launch rules register
    # the draft checkpoint's config type, platform.apply_model_defaults)
    from sglang_neuron.sglang_patches import dspark_server_args

    dspark_server_args.register(HookRegistry, HookType.REPLACE)
    gpt_oss_backend_check.register(HookRegistry, HookType.AROUND)
    # a prefill without a decode peer (bring-up, timing): argument resolution and the tokenizer manager
    pd_fake_prefill.register(HookRegistry, HookType.AROUND)
    neuron_system_profile.register_tokenizer(HookRegistry, HookType.AROUND)
    # a crashing server neither triggers CUDA coredumps nor waits 60 s for them
    crash_diagnostics_platform.register(HookRegistry, HookType.REPLACE)
    # a multi-worker HTTP server keeps SGLang's fatal SIGQUIT handler under Uvicorn's supervisor
    http_fatal_sigquit.register(HookRegistry, HookType.AROUND, HookType.REPLACE)
    # SGLang's PP passes over the server arguments run here, before the policy is exported; the hooks keep the overlap
    # schedule on for a pipelined decode, reading the policy when they run
    from sglang_neuron.sglang_patches import pp_overlap_schedule

    pp_overlap_schedule.register(HookRegistry, HookType.AROUND)
    # SGLang's DSA server-arg adjustment (GLM-5.2, DeepSeek V3.2) runs here too, while the launching process resolves
    # the arguments: its CUDA capability read and GPU DSA passes must be hooked before then, not only in the schedulers
    from sglang_neuron.sglang_patches import dsa_platform_adjustments

    dsa_platform_adjustments.register(HookRegistry, HookType.AROUND)
    HookRegistry.register(f"{_TOKENIZER}._validate_one_request", _validate_request, HookType.AROUND)
    # the served model's own sampling values for a chat request (launch hooks chat_sampling_params; the Kimi-K3 API's)
    from sglang_neuron.sglang_patches import model_chat_sampling

    model_chat_sampling.register(HookRegistry, HookType.AROUND)
    HookRegistry.register(f"{_TOKENIZER}.convert_logprob_style", convert_logprob_style, HookType.AROUND)
    for target in _WEIGHT_UPDATES:
        HookRegistry.register(target, _refuse_weight_update, HookType.AROUND)
    HookRegistry.register(f"{_TOKENIZER}.pause_generation", _refuse_pause_under_pp, HookType.AROUND)
    HookRegistry.register(f"{_CONTROL}.flush_cache", _refuse_timed_flush_under_pp, HookType.AROUND)
    # the launching process checks its arguments against the backend choices
    _register_attention_backend()
    _register_megakernel_launch_seams()


def register() -> None:
    """SGLang's plugin entry point (pyproject.toml); it does nothing for a launch on another platform.

    Runtime registration fails closed. SGLang's loader logs a plugin's exception and carries on, which would serve a
    half-registered runtime, so a failure here raises SystemExit, which the loader does not catch."""
    if os.environ.get("SGLANG_PLATFORM") != "neuron":
        return
    from sglang_neuron import serving_policy

    register_launch_hooks()
    if serving_policy.ENV_KEY not in os.environ:
        return  # the launching process, before its platform defaults resolve the launch
    try:
        _register()
    except Exception as exc:
        logger.critical("sglang_neuron: runtime registration failed", exc_info=True)
        raise SystemExit(f"sglang_neuron: runtime registration failed ({type(exc).__name__}: {exc})") from exc


def _register() -> None:
    from sglang_neuron import process_init, serving_policy
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_RUNTIME_SOURCES_MD5, __name__)

    process_init.init_stack()
    _register_model_package()
    policy = serving_policy.current()
    from sglang_neuron.models import catalog

    hooks = catalog.served_launch_hooks()
    if hooks is not None and hasattr(hooks, "prepare_process"):
        # the served model's own setup of this process (catalog.ModelSpec.launch_hooks)
        hooks.prepare_process()

    # bring-up and SGLang's CUDA-only defaults
    from sglang_neuron.model_executor.runtime_init import init_torch_distributed_hook

    HookRegistry.register("sglang.srt.model_executor.model_runner.ModelRunner.init_torch_distributed",
                          init_torch_distributed_hook, HookType.AROUND)
    HookRegistry.register("sglang.srt.distributed.parallel_state.init_model_parallel_group",
                          _init_model_parallel_group_without_device_communicators, HookType.AROUND)
    HookRegistry.register("sglang.srt.layers.dp_attention.memcpy_func", _memcpy_torch, HookType.REPLACE)
    HookRegistry.register("sglang.srt.utils.common.get_available_gpu_memory", _available_device_memory_gb,
                          HookType.AROUND)
    HookRegistry.register("sglang.srt.distributed.bootstrap._bind_threads_if_cpu", _no_cpu_engine_binding,
                          HookType.AROUND)
    from sglang_neuron.sglang_patches import numa_node_published

    numa_node_published.register(HookRegistry, HookType.AROUND)
    HookRegistry.register("sglang.srt.model_executor.model_runner.adjust_config_with_unaligned_cpu_tp",
                          _keep_model_config, HookType.AROUND)

    # the model, its KV pool and its sampled outputs
    HookRegistry.register("sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator.configure", _model_kv_pools,
                          HookType.AROUND)
    HookRegistry.register("sglang.srt.layers.attention.attention_registry.attn_backend_wrapper",
                          _no_linear_attention_backend, HookType.AROUND)
    HookRegistry.register("sglang.srt.model_executor.model_runner.ModelRunner.decode_graph_gather_requirements",
                          _no_decode_gathers, HookType.AROUND)
    HookRegistry.register("sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator._apply_token_constraints",
                          _kv_pool_is_the_stated_capacity, HookType.AROUND)
    HookRegistry.register("sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator._profile_available_bytes",
                          _less_the_model_pools_beside_sglang, HookType.AROUND)
    from sglang_neuron.sglang_patches import state_pool_class

    # the recurrent-state pool class a model declares (a megakernel PoolLayout's, else NeuronModel.state_pool_class);
    # SGLang's own MambaPool for every other model
    state_pool_class.register(HookRegistry, HookType.AROUND)
    from sglang_neuron.sglang_patches import hybrid_linear_metadata

    # a hybrid linear-attention model's graph runs its linear attention itself: no host-side metadata for it
    hybrid_linear_metadata.register(HookRegistry, HookType.AROUND)
    HookRegistry.register("sglang.srt.model_executor.model_runner.ModelRunner.sample", _sample_on_device,
                          HookType.AROUND)
    # spec-v2's per-step KV allocation of a speculative decode, without its Triton kernels
    from sglang_neuron.sglang_patches import spec_v2_host_alloc

    spec_v2_host_alloc.register(HookRegistry, HookType.REPLACE)
    from sglang_neuron import sampling

    # device drawing modes only; SGLang's own sampler seeds a batch only under --enable-deterministic-inference
    if sampling.sampler_mode() in sampling.DRAWING_MODES:
        HookRegistry.register("sglang.srt.sampling.sampling_batch_info.SamplingBatchInfo.from_schedule_batch",
                              _sampling_info_with_seeds, HookType.AROUND)

    # the scheduler: one request per prefill forward
    HookRegistry.register("sglang.srt.managers.schedule_policy.PrefillAdder.add_chunked_req", _note_chunked_request,
                          HookType.AROUND)
    HookRegistry.register("sglang.srt.managers.schedule_policy.PrefillAdder.add_one_req", _one_request_per_prefill,
                          HookType.AROUND)
    HookRegistry.register("sglang.srt.managers.schedule_policy.PrefillAdder.preempt_to_schedule",
                          _preempt_only_for_an_admitted_request, HookType.AROUND)
    from sglang_neuron.sglang_patches import chunk_keyed_tree, decode_graph_buckets, prefill_chunk_grid

    prefill_chunk_grid.register(HookRegistry, HookType.AROUND, HookType.REPLACE)
    if policy.megakernel:
        # a megakernel decode's capture list is the runtime's own buckets (its warm captures), not upstream's
        # attention-TP-aligned subset — a per-layer model keeps upstream's
        decode_graph_buckets.register(HookRegistry, HookType.REPLACE)
    if policy.chunk_keyed_tree:
        chunk_keyed_tree.register()
    if policy.kimi_prompt_cache:
        from sglang_neuron.sglang_patches import kimi_prompt

        kimi_prompt.register(HookRegistry, HookType.AROUND)
    from sglang_neuron.mem_cache import hybrid_swa

    if hybrid_swa.radix_mode():
        # hybrid SWA under the radix tree: a finished request's window frees before its insert
        from sglang_neuron.sglang_patches import finish_swa_frontier

        finish_swa_frontier.register(HookRegistry, HookType.REPLACE)
    HookRegistry.register("sglang.srt.managers.scheduler.Scheduler.flush_cache", _flush_cache_drained,
                          HookType.AROUND)
    from sglang_neuron.managers.logprob_errors import finish_prefill_aborts

    HookRegistry.register(
        "sglang.srt.disaggregation.prefill.SchedulerDisaggregationPrefillMixin.process_disagg_prefill_inflight_queue",
        finish_prefill_aborts, HookType.AROUND)
    if policy.async_execution:
        from sglang_neuron.model_executor import async_execution

        async_execution.register_hooks(HookRegistry, HookType.AROUND)
    from sglang_neuron.managers import host_loop

    host_loop.register_hooks(HookRegistry, HookType.AROUND)
    if policy.request_broadcast_shm and not policy.dp_identity:
        # under the DP identity its control fence owns the broadcast
        from sglang_neuron.distributed.request_broadcast import broadcast_reqs_with_message_queue

        HookRegistry.register("sglang.srt.managers.scheduler_components.request_receiver.SchedulerRequestReceiver"
                              "._broadcast_reqs_across_ranks", broadcast_reqs_with_message_queue, HookType.AROUND)
    from sglang_neuron.sglang_patches import (abort_rid_match, batch_record_light, detokenizer_sender,
                                              dp_controller_blocking_recv, flush_deadline_markers, mlp_sync_no_pin,
                                              neuron_system_profile)

    flush_deadline_markers.register(HookRegistry, HookType.AROUND)
    neuron_system_profile.register(HookRegistry, HookType.AROUND)
    abort_rid_match.register(HookRegistry, HookType.REPLACE)
    batch_record_light.register(HookRegistry, HookType.REPLACE)
    dp_controller_blocking_recv.register(HookRegistry, HookType.REPLACE)
    mlp_sync_no_pin.register(HookRegistry, HookType.REPLACE)
    if policy.detokenizer_sender_rank1:
        detokenizer_sender.register(HookRegistry, HookType.AROUND)
    from sglang_neuron import disaggregation

    disaggregation.register_hooks(HookRegistry, HookType)
    _register_pipeline_parallel(policy)
    _register_decode_context_parallel(policy)
    _register_megakernel_runtime_seams(policy)
    # HiCache L2 (only reached with --enable-hierarchical-cache): the Neuron host pools, their transfer events and the
    # node's shared host budget
    from sglang_neuron.mem_cache import host_budget, host_pool, hybrid_host_pool

    host_pool.register_hooks(HookRegistry, HookType)
    hybrid_host_pool.register_hooks(HookRegistry, HookType)
    host_budget.register_hooks(HookRegistry, HookType)
    logger.info("[neuron] runtime registered (serving policy %s)", policy.to_json())


def _register_pipeline_parallel(policy) -> None:
    """The scheduler hooks of a pipelined launch (validate_pp_launch decides which run); none without PP.

    Every stage re-solves its KV pools from the token count the stages synced, so all hold the same full and window
    counts. An aggregated server adds only the loop state SGLang's own PP loop relies on; that loop processes each
    micro-batch's outputs on the host before its next step. A PD role runs its PP loop in the overlap order, plus the
    per-role fixes registered below."""
    if policy.pp_size <= 1:
        return
    from sglang_neuron.sglang_patches import pp_loop_state, pp_pool_resolve

    pp_pool_resolve.register(HookRegistry, HookType.AROUND)
    # SGLang's proxy transport slice-sends and all-gathers over attention TP, which reassembles a tensor only when
    # every rank of the stage holds the same one; the seam's proxies are one-element descriptors, so refuse any proxy
    # tensor the transport would slice (every proxy send funnels through _pp_send_dict_to_next_stage)
    HookRegistry.register("sglang.srt.managers.scheduler_pp_mixin.SchedulerPPMixin._pp_send_dict_to_next_stage",
                          _pp_proxy_send_checked, HookType.AROUND)
    if policy.role == "aggregated":
        pp_loop_state.register_aggregated(HookRegistry, HookType.AROUND)
        return
    pp_loop_state.register(HookRegistry, HookType.AROUND)
    if policy.role == "decode":
        from sglang_neuron.sglang_patches import (pp_decode_held_release, pp_decode_micro_batch_bound,
                                                  pp_decode_overlap_loop, pp_overlap_schedule)

        pp_overlap_schedule.register_scheduler(HookRegistry, HookType.AROUND, HookType.REPLACE)
        pp_decode_overlap_loop.register(HookRegistry, HookType.AROUND)
        pp_decode_micro_batch_bound.register(HookRegistry, HookType.REPLACE)
        pp_decode_held_release.register(HookRegistry, HookType.AROUND, HookType.REPLACE)
        return
    from sglang_neuron.sglang_patches import (pp_last_stage_async, pp_pending_chunked_abort,
                                              pp_prefill_inflight_wait, pp_prefill_kv_after_forward,
                                              pp_prefill_last_send, pp_prefill_overlap_loop)

    pp_pending_chunked_abort.register(HookRegistry, HookType.AROUND)
    pp_prefill_kv_after_forward.register(HookRegistry, HookType.AROUND)
    pp_last_stage_async.register(HookRegistry, HookType.AROUND)
    pp_prefill_overlap_loop.register(HookRegistry, HookType.REPLACE, HookType.AROUND)
    # the early last-chunk sends and the early answer (both read their switches when the loop calls them), and the hold
    # of a request that extends one in flight until that release, so it takes the release's prefix-cache hit
    pp_prefill_last_send.register(HookRegistry, HookType.AROUND)
    pp_prefill_inflight_wait.register(HookRegistry, HookType.AROUND)


def pp_proxy_slice_problem(tensors, attn_tp_size: int):
    """Why SGLang's proxy transport would corrupt ``tensors``, or None.

    ``GroupCoordinator.send_tensor_dict`` sends each rank 1/attn_tp of every tensor whose element count divides by the
    attention-TP size and the receiver all-gathers the slices over attention TP. That reassembles a tensor only when
    every rank of the stage holds the same one; a per-rank tensor comes back with other ranks' rows and no error. The
    seam's proxies carry one-element step and batch descriptors, which are never sliced at attention TP > 1, so any
    proxy tensor the transport would slice is refused."""
    if attn_tp_size <= 1:
        return None
    import torch

    for key, value in tensors.items():
        if not isinstance(value, torch.Tensor):
            continue
        n = value.numel()
        if n > 1 and n % attn_tp_size == 0:
            return (f"PP proxy tensor {key!r} ({n} elements) would be slice-sent and all-gathered over "
                    f"{attn_tp_size} attention-TP ranks, which mixes ranks' rows unless every rank holds the same "
                    f"tensor; Neuron proxies carry only one-element descriptors (stream real tensors over the PP seam)")
    return None


def _pp_proxy_send_checked(original_fn, self, tensor_dict, async_send=True, msg_type="default", ready_event=None):
    """AROUND SchedulerPPMixin._pp_send_dict_to_next_stage: refuse proxy content the slice/all-gather would corrupt.

    Every PP proxy send funnels here (upstream's _pp_send_proxy_to_next_stage and the overlap loops' direct sends,
    msg_type="proxy"); output and default dicts pass through."""
    if msg_type == "proxy":
        group = getattr(self, "attn_tp_group", None)
        problem = pp_proxy_slice_problem(tensor_dict, group.world_size if group is not None else 1)
        if problem:
            raise RuntimeError(problem)
    return original_fn(self, tensor_dict, async_send=async_send, msg_type=msg_type, ready_event=ready_event)


def _register_decode_context_parallel(policy) -> None:
    """Let SGLang build its DCP groups on Neuron (sglang_patches.dcp_groups); nothing without --dcp-size above 1."""
    if policy.dcp_size <= 1:
        return
    from sglang_neuron.sglang_patches import dcp_groups

    dcp_groups.register(HookRegistry, HookType.REPLACE)


def _register_megakernel_launch_seams() -> None:
    """The platform seams of upstream SGLang's megakernel binding (--enable-nvidia-megakernel) that every process
    reads: the platform's device target and the target registry the specs are found in (argument resolution, the
    tokenizer, the schedulers) and GPT-OSS's backend overrides left to a registered spec (argument resolution). Then the
    specs the plugin ships, on their targets (megakernel/registry.py). Without the flag every hooked function is
    upstream's (docs/design/megakernel.md, Seams; the seams a later spec needs register here with it)."""
    from sglang_neuron.megakernel.registry import register_specs
    from sglang_neuron.sglang_patches import megakernel_oot_overrides, megakernel_platform, megakernel_request_contract

    megakernel_platform.register(HookRegistry, HookType.AROUND, HookType.REPLACE)
    megakernel_request_contract.register(HookRegistry, HookType.AROUND)
    megakernel_oot_overrides.register()
    register_specs()


def _register_megakernel_runtime_seams(policy) -> None:
    """The megakernel seams of a scheduler process: a platform-owned whole-step dispatch run by EagerRunner, KV
    pools in a built megakernel's declared layout, and, on a launch of a model's whole-model megakernel
    (ServingPolicy.megakernel), the MLP sync cohort snapshot its spec can ask for (megakernel_platform.
    agree_mlp_sync_cohort turns it on; off, it changes nothing)."""
    from sglang_neuron.sglang_patches import megakernel_eager_dispatch, megakernel_pool_layout

    megakernel_eager_dispatch.register(HookRegistry, HookType.AROUND)
    megakernel_pool_layout.register(HookRegistry, HookType.AROUND)
    if policy.megakernel:
        from sglang_neuron.sglang_patches import mlp_sync_cohort_snapshot

        mlp_sync_cohort_snapshot.register(HookRegistry, HookType.AROUND, HookType.REPLACE)


def _register_model_package() -> None:
    """Strictly import the model package the platform chose (platform.MODEL_PACKAGE_ENV, set by the launching
    process before it starts this one).

    SGLang builds the model from that package, and imports its own models only for an architecture the package does
    not provide. But SGLang imports the package leniently: a module that raises is logged and skipped, and SGLang then
    builds its own class for that architecture. Importing strictly here makes that module's error stop the process."""
    from sglang_neuron.platform import MODEL_PACKAGE_ENV

    package = os.environ.get(MODEL_PACKAGE_ENV)
    if not package:
        raise RuntimeError(f"{MODEL_PACKAGE_ENV} is unset in a process of a Neuron launch: the platform sets it to the "
                           "model's package in the launching process, before it starts this one")
    from sglang.srt.models import registry

    registry.ModelRegistry.models.update(registry.import_model_classes(package, strict=True))


def _register_attention_backend() -> None:
    from sglang.srt import server_args
    from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS, register_attention_backend

    if ATTN_BACKEND_NAME not in server_args.ATTENTION_BACKEND_CHOICES:
        server_args.add_attention_backend_choices([ATTN_BACKEND_NAME])
    if ATTN_BACKEND_NAME not in ATTENTION_BACKENDS:
        @register_attention_backend(ATTN_BACKEND_NAME)
        def _neuron_attention_backend(model_runner):
            from sglang_neuron.attention.backend import NeuronAttnBackend

            return NeuronAttnBackend(model_runner)


def _init_model_parallel_group_without_device_communicators(original_fn, *args, **kwargs):
    """Turn off every device communicator SGLang offers. Every process group is gloo and the device collectives run
    inside the graphs; left on, the communicators would touch torch.cuda. The message-queue broadcaster stays as the
    caller asked."""
    for name in _device_communicator_params(original_fn):
        kwargs[name] = False
    return original_fn(*args, **kwargs)


@functools.lru_cache(maxsize=None)
def _device_communicator_params(fn) -> tuple:
    import inspect

    return tuple(name for name in inspect.signature(fn).parameters
                 if name.startswith("use_") and name != "use_message_queue_broadcaster")


def _memcpy_torch(dst, src, dim, offset, sz, offset_src):
    """dp_attention.memcpy_func: SGLang's torch copy, as its Triton copy is CUDA-only."""
    from sglang.srt.layers.dp_attention import memcpy_cpu

    memcpy_cpu(dst, src, dim, offset, sz, offset_src)


def _available_device_memory_gb(original_fn, device, gpu_id, distributed=False, empty_cache=True, cpu_group=None):
    """Free NeuronCore HBM in GB: the total less what the runtime holds, the minimum over ``cpu_group`` when
    ``distributed``. SGLang's device string is "cpu" here, so its own reading would size the KV pool from host RAM."""
    import torch
    from sglang.srt.platforms import current_platform

    total = float(current_platform.get_device_total_memory(gpu_id))
    used = float(current_platform.get_current_memory_usage())
    free_gb = max(total - used, 0.0) / (1 << 30)
    if distributed:
        t = torch.tensor([free_gb], dtype=torch.float32)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN, group=cpu_group)
        free_gb = float(t.item())
    return free_gb


def _no_cpu_engine_binding(original_fn, *args, **kwargs):
    # "cpu" here is where SGLang's batch tensors live, not SGLang's CPU engine; the engine's OpenMP binding would pin
    # rank r to NUMA node r
    return None


def _keep_model_config(original_fn, model_config, *args, **kwargs):
    # the CPU engine's TP head re-alignment does not apply
    return model_config


def _kv_pool_is_the_stated_capacity(original_fn, self, token_capacity, *args, **kwargs):
    """AROUND KVCacheConfigurator._apply_token_constraints: a launch that states --max-total-tokens gets a KV pool of
    exactly that size, or is refused. The pool's block count is a dimension of every compiled graph, so sizing the
    pool from each launch's memory reading would make every launch compile its own graphs."""
    from sglang.srt.runtime_context import get_schedule

    tokens = original_fn(self, token_capacity, *args, **kwargs)
    stated = get_schedule().max_total_tokens
    if stated is None:
        logger.warning("[neuron] the KV pool holds the %d tokens read from the device memory; its block count is a "
                       "dimension of every compiled graph, so state --max-total-tokens for a stable compile cache",
                       tokens)
    elif tokens < stated:
        raise ValueError(f"--max-total-tokens={stated} does not fit: the device memory holds a KV pool of {tokens} "
                         f"tokens; lower --max-total-tokens to at most {tokens}, or raise --mem-fraction-static")
    return tokens


def _less_the_model_pools_beside_sglang(original_fn, self, *args, **kwargs):
    """AROUND KVCacheConfigurator._profile_available_bytes: the KV budget less the bytes a model allocates beside
    SGLang's pools once they are built (``device_bytes_beside_pools(max_total_tokens)``, the model's own pools at the
    launch's --max-total-tokens), so --mem-fraction-static covers them too. A model without it keeps SGLang's budget.

    SGLang's budget starts from the smallest free memory over the world, so that every rank sizes the same pool, but
    the extra bytes of a rank are its own: Kimi-K3's drafter rows are on the drafter writer stage only, which also has
    the fewest layers and so the most free memory. Each rank thus charges its extra bytes against its own free memory,
    and the budget of every rank falls by how far the smallest of these headrooms is below the smallest free memory
    (``_beside_deficit``). On one rank that is the rank's extra bytes, as before."""
    from sglang.srt.runtime_context import get_parallel, get_schedule

    available = original_fn(self, *args, **kwargs)
    beside = getattr(getattr(self, "model", None), "device_bytes_beside_pools", None)
    if beside is None:
        return available
    extra = int(beside(get_schedule().max_total_tokens))
    free = int(_available_device_memory_gb(None, self.device, self.gpu_id) * (1 << 30))
    parallel = get_parallel()
    group = parallel.world_group.cpu_group if parallel.launch_world_size > 1 else None
    deficit = _beside_deficit(free, extra, group)
    if extra or deficit:
        logger.info("[neuron] %s allocates %.3f GiB beside SGLang's pools on this rank; the KV budget of every rank "
                    "leaves out %.3f GiB for them", type(self.model).__name__, extra / (1 << 30), deficit / (1 << 30))
    return available - deficit


def _beside_deficit(free: int, extra: int, group=None) -> int:
    """How far the smallest headroom of ``group`` (a rank's ``free`` bytes less its ``extra`` bytes beside SGLang's
    pools) is below the smallest ``free`` of ``group``, in bytes; ``group`` None: this rank alone (``extra``).

    Example:
        >>> _beside_deficit(8 << 30, 1 << 30)
        1073741824
    """
    import torch

    values = torch.tensor([free, free - extra], dtype=torch.int64)
    if group is not None:
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.MIN, group=group)
    return max(0, int(values[0]) - int(values[1]))


def _no_linear_attention_backend(original_fn, runner, full_attn_backend, *args, **kwargs):
    """AROUND attention_registry.attn_backend_wrapper: SGLang wraps the platform's attention backend in its hybrid
    linear-attention backend for a model with linear-attention layers (for Kimi-K3, a KDA backend that reads the
    state pool's conv and recurrent tensors). A model that runs no step through SGLang's attention backends
    (``runs_no_sglang_attention``: its speculative worker runs every step, and its forward is refused) keeps the
    platform's backend unwrapped; its state pool holds no such tensors."""
    if getattr(getattr(runner, "model", None), "runs_no_sglang_attention", False):
        return full_attn_backend
    return original_fn(runner, full_attn_backend, *args, **kwargs)


def _no_decode_gathers(original_fn, self, *args, **kwargs):
    """AROUND ModelRunner.decode_graph_gather_requirements: the MLP and attention-TP gathers SGLang's decode graphs need,
    which align every decode graph batch to whole attention-TP rows (startup sizing: the shared logits buffer, the
    reduction workspaces). A model that runs no step through SGLang's forward (``runs_no_sglang_attention``) has no
    such graphs; its own runtime does its gathers. Without this, the Kimi-K3 decode's one batch of 4 requests x 4 rows
    is not a multiple of its 32 attention-TP ranks and SGLang finds no batch size to size them for."""
    if getattr(getattr(self, "model", None), "runs_no_sglang_attention", False):
        return False, False
    return original_fn(self, *args, **kwargs)


def _without_draft_kv_pool(configurator) -> None:
    """Take the draft's KV pool out of a KV configurator's sizing and pools. For a DFLASH-family algorithm (DSPARK)
    SGLang budgets a separate draft KV pool beside the target's (the draft's per-token cost from the draft config,
    times the DCP size) and may fuse one into the target's entries (``draft_kv_num_layers``). A model whose own
    speculative worker holds the draft's rows in the model's pools has no such pool; budgeting it cuts the target's
    KV capacity several times over (the Kimi-K3 decode: 95,360 tokens a rank instead of its 435,072)."""
    import msgspec

    aux = configurator.spec_aux_config
    configurator.spec_aux_config = msgspec.structs.replace(aux, dflash_draft_num_layers=None,
                                                           draft_kv_num_layers=None, draft_model_config=None)
    configurator.dflash_draft_cell_size_per_token = None
    logger.info("[neuron] %s keeps the draft's KV in its own pools: SGLang's draft KV pool is not budgeted or built",
                type(configurator.model).__name__)


def _model_kv_pools(original_fn, self, *, pre_model_load_memory):
    """AROUND KVCacheConfigurator.configure: a model with configure_kv_cache(configurator, pre_model_load_memory=...)
    builds the KVCacheConfigResult itself; other models keep SGLang's.

    Its pool and allocator must be the platform's: the runtime binds the pool's buffers in the kernels' block layout,
    and only the platform's allocator holds a freed page until the queued graphs that read it have run. A stated
    --max-total-tokens, floored to whole pages, must be the pool's size, as in _kv_pool_is_the_stated_capacity.

    A model whose speculative worker keeps the draft's KV in the model's own pools (``draft_kv_in_model_pools``)
    sizes and builds without SGLang's draft KV pool: see _without_draft_kv_pool."""
    if getattr(self.model, "draft_kv_in_model_pools", False):
        _without_draft_kv_pool(self)
    configure_kv_cache = getattr(self.model, "configure_kv_cache", None)
    if configure_kv_cache is None:
        return original_fn(self, pre_model_load_memory=pre_model_load_memory)
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigResult
    from sglang.srt.platforms import current_platform
    from sglang.srt.runtime_context import get_schedule

    name = type(self.model).__name__
    result = configure_kv_cache(self, pre_model_load_memory=pre_model_load_memory)
    if not isinstance(result, KVCacheConfigResult):
        raise TypeError(f"{name}.configure_kv_cache returned {type(result).__name__}, not a KVCacheConfigResult")
    pools = tuple(current_platform.get_kv_pool_cls(kind=kind) for kind in ("mha", "mla"))
    if not isinstance(result.token_to_kv_pool, pools):
        raise ValueError(f"{name}.configure_kv_cache built a {type(result.token_to_kv_pool).__name__} KV pool; the "
                         f"runtime binds the platform's ({', '.join(cls.__name__ for cls in pools)}), whose buffers "
                         "are in the kernels' block layout")
    allocator = current_platform.get_paged_allocator_cls()
    if not isinstance(result.token_to_kv_pool_allocator, allocator):
        raise ValueError(f"{name}.configure_kv_cache built a {type(result.token_to_kv_pool_allocator).__name__}; the "
                         f"platform serves its {allocator.__name__}, which holds a freed page until the launches that "
                         "read it have run")
    schedule = get_schedule()
    stated = schedule.max_total_tokens
    if stated is not None and result.max_total_num_tokens != stated // schedule.page_size * schedule.page_size:
        raise ValueError(f"--max-total-tokens={stated}, but {name}.configure_kv_cache built a KV pool of "
                         f"{result.max_total_num_tokens} tokens; the pool's size is a dimension of every compiled "
                         f"graph, so build the stated size in whole {schedule.page_size}-token pages")
    return result


def _sample_on_device(original_fn, self, logits_output, forward_batch, *args, **kwargs):
    """AROUND ModelRunner.sample: return what the graph sampled on device; SGLang's sampler runs only on the host
    sampler's full logits (SGLANG_NEURON_SAMPLER=host).

    logits_output.neuron_sample holds the tokens, logprobs and top logprobs. For an async sample still in flight
    (neuron_pending_sample) the hook returns only its tokens and records the requested top-logprob counts on it."""
    import torch

    ns = getattr(logits_output, "neuron_sample", None)
    if ns is None:
        return original_fn(self, logits_output, forward_batch, *args, **kwargs)
    pending = getattr(logits_output, "neuron_pending_sample", None)
    if pending is not None:
        pending.top_logprobs_nums = tuple(int(k or 0) for k in (forward_batch.top_logprobs_nums or ()))
        return ns["tokens"]
    next_token_ids = ns["tokens"].to(torch.int64)
    if forward_batch.return_logprob:
        logits_output.next_token_logprobs = ns["logprobs"]
        nums = forward_batch.top_logprobs_nums or []
        if any(k and k > 0 for k in nums):
            # one tensor per row: the scheduler's move_logprobs_to_cpu calls .tolist() on each
            ks = [int(k or 0) for k in nums]
            logits_output.next_token_top_logprobs_val = [ns["top_vals"][i, :k] for i, k in enumerate(ks)]
            logits_output.next_token_top_logprobs_idx = [ns["top_ids"][i, :k] for i, k in enumerate(ks)]
    return next_token_ids


def _sampling_info_with_seeds(original_fn, cls, batch, vocab_size, *args, **kwargs):
    """Fill SamplingBatchInfo.sampling_seed with each row's request seed. SGLang builds it only under
    --enable-deterministic-inference, but every non-greedy device sampler keys its draws on it."""
    import torch

    info = original_fn(cls, batch, vocab_size, *args, **kwargs)
    if info.sampling_seed is None:
        from sglang_neuron.sampling import request_seed

        info.sampling_seed = torch.tensor([request_seed(r) for r in batch.reqs], dtype=torch.int64,
                                          device=info.temperatures.device)
    return info


def _note_chunked_request(original_fn, self, req, *args, **kwargs):
    """Mark the adder's pass as carrying a chunked request, whether the request joins the batch or waits for the next
    pass. _preempt_only_for_an_admitted_request reads the mark."""
    self.carries_chunked_req = True
    return original_fn(self, req, *args, **kwargs)


def _one_request_per_prefill(original_fn, self, req, has_chunked_req, *args, **kwargs):
    """A prefill forward runs one request: the adder stops after the first, and admits nothing beside a chunked
    request."""
    from sglang.srt.managers.schedule_policy import AddReqResult

    if self.can_run_list or has_chunked_req:
        return AddReqResult.OTHER
    return original_fn(self, req, has_chunked_req, *args, **kwargs)


def _preempt_only_for_an_admitted_request(original_fn, self, req, *args, **kwargs):
    """Preempt only for a request _one_request_per_prefill will admit: the pass's first, with no chunked request in
    the pass. Priority preemption frees running requests' KV, and the scheduler requeues them either way, so
    preempting for a request the pass then refuses would only make those requests recompute their prefill."""
    if self.can_run_list or getattr(self, "carries_chunked_req", False):
        return False
    return original_fn(self, req, *args, **kwargs)


def _flush_cache_drained(original_fn, self, *args, **kwargs):
    """AROUND Scheduler.flush_cache: drain the device queue before the pools reset.

    SGLang flushes only an idle scheduler, but under async execution an idle scheduler can still have its last
    launch's outputs and KV writes queued on the device, so every launch retires first. After a successful flush the
    token relay also forgets the rows of the requests the flush retired."""
    from sglang_neuron.model_executor.async_execution import drain_launches, reset_token_relay

    drain_launches()
    ok = original_fn(self, *args, **kwargs)
    if ok:
        reset_token_relay()
    return ok
