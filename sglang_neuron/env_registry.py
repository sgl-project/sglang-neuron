# SPDX-License-Identifier: Apache-2.0
"""Every SGLANG_NEURON_* environment variable the runtime reads, by class.

The platform refuses a launch that sets a SGLANG_NEURON_* name not listed here; otherwise a typo, or a name nothing
reads, would be silently ignored. docs/guides/reference-configuration.md documents each name.

Classes:
* config: a deployment setting with no SGLang flag;
* debug: diagnostics and profiling, off on a served launch;
* internal: set by the plugin for the processes it starts, never by a user;
* benchmark: pins a measurement and changes what the server emits, so it is off by default, never set on a server
  that answers real requests, and its reader logs a warning at launch when it is set.
"""

from __future__ import annotations

import os
import re

CONFIG, DEBUG, INTERNAL, BENCHMARK = "config", "debug", "internal", "benchmark"

KNOBS: dict = {
    # the device sampler
    "SGLANG_NEURON_SAMPLER": CONFIG,
    "SGLANG_NEURON_MAX_TOP_LOGPROBS": CONFIG,
    # GPT-OSS's expert numeric format (models/gpt_oss/config.expert_format)
    "SGLANG_NEURON_GPT_OSS_EXPERTS": CONFIG,
    # GLM-5.2 (models/glm52): the decode's weight store, the prefill's MTP rows for a drafting decode, the decode
    # point, the benchmark's pinned MTP acceptance and the decode's FP8 attention weights
    "SGLANG_NEURON_GLM52_WEIGHT_STORE": CONFIG,
    "SGLANG_NEURON_GLM52_PD_MTP": CONFIG,
    "SGLANG_NEURON_GLM52_POINT": CONFIG,
    "SGLANG_NEURON_GLM52_MTP_SYNTHETIC_AL": BENCHMARK,
    "SGLANG_NEURON_GLM52_ATTN_W8": CONFIG,
    # the Kimi-K3 prompt cache on the PD prefill (--radix-cache-backend kimi-prompt, mem_cache/kimi_prompt_backend):
    # its host tier, keep_previous, the chunk ends and the read-back of each page copy of a hit
    "SGLANG_NEURON_KIMI_PROMPT_HOST_SLOTS": CONFIG,
    "SGLANG_NEURON_KIMI_PROMPT_HOST_STATE_SLOTS": CONFIG,
    "SGLANG_NEURON_KIMI_PROMPT_KEEP_PREVIOUS": CONFIG,
    "SGLANG_NEURON_KIMI_PROMPT_CHUNK_ENDS": CONFIG,
    "SGLANG_NEURON_KIMI_PROMPT_VERIFY_COPY": DEBUG,
    # a forced mean acceptance length on the Kimi-K3 decode, for a step benchmark
    # (models/kimi_k3/decode/spec_worker.forced_acceptance)
    "SGLANG_NEURON_KIMI_K3_FORCED_AL": BENCHMARK,
    # a log line of the Kimi-K3 decode worker's host step times every N decode steps
    # (models/kimi_k3/decode/spec_worker.StepClock)
    "SGLANG_NEURON_KIMI_K3_STEP_TIMES": DEBUG,
    # the sampling of a Kimi-K3 chat request: the Kimi-K3 contract or the client's values
    # (models/kimi_k3/launch.chat_sampling_params)
    "SGLANG_NEURON_KIMI_K3_SAMPLING": CONFIG,
    # a pipelined PD prefill: every stage sends a request's last chunk at its own forward
    # (sglang_patches/pp_prefill_last_send), answers the request at those sends (pp_prefill_early_answer; default 1)
    # and holds a request that extends one in flight until its release (pp_prefill_inflight_wait; forward passes)
    "SGLANG_NEURON_PP_PREFILL_EARLY_LAST_SEND": CONFIG,
    "SGLANG_NEURON_PP_PREFILL_EARLY_ANSWER": CONFIG,
    "SGLANG_NEURON_PP_PREFILL_INFLIGHT_WAIT_PASSES": CONFIG,
    # one K3STEP log line for each harvested Kimi-K3 verify step (models/kimi_k3/decode/spec_worker.step_log_line)
    "SGLANG_NEURON_KIMI_K3_STEP_LOG": DEBUG,
    # opt-in hybrid SWA memory under the radix cache (mem_cache.hybrid_swa)
    "SGLANG_NEURON_HYBRID_SWA_RADIX": CONFIG,
    # Qwen3.5's packed checkpoint directory (models/qwen35/weights.py), the PD decode role's KV context-parallel
    # degree (models/qwen35/config.KV_CP_ENV) and the decode's opt-in HiCache host tier (config.HICACHE_ENV)
    "SGLANG_NEURON_QWEN35_PACK": CONFIG,
    "SGLANG_NEURON_QWEN35_KV_CP": CONFIG,
    # the kv_cp=1 PD prefill's token-major K pages (models/qwen35/config.KV_TM_ENV)
    "SGLANG_NEURON_QWEN35_KV_TM": CONFIG,
    "SGLANG_NEURON_QWEN35_HICACHE": CONFIG,
    # Qwen3.5's kv_cp scatter (sglang_patches.kv_cp_scatter): the posted-batch descriptor bound and drain, and its
    # prep/send dump; the model runtime's state dump (models/qwen35/runtime.py)
    "SGLANG_NEURON_QWEN35_SCATTER_MAX_DESCS": DEBUG,
    "SGLANG_NEURON_QWEN35_SCATTER_DRAIN": DEBUG,
    "SGLANG_NEURON_QWEN35_SCATTER_DUMP": DEBUG,
    "SGLANG_NEURON_QWEN35_STATE_DUMP": DEBUG,
    # opt-in paged latent pool for SGLang's MLA kind (envs.mla_pool_layout; mem_cache/mla_pool.py)
    "SGLANG_NEURON_MLA_POOL": CONFIG,
    # DeepSeek-V2's longest served context, raised for a long-context probe (envs.deepseek_v2_max_context)
    "SGLANG_NEURON_DSV2_MAX_CONTEXT": CONFIG,
    # opt-in HBM small-allocation arena guard, the NRT status 1200 workaround (backend.hbm_tail_guard)
    "SGLANG_NEURON_HBM_TAIL_GUARD": CONFIG,
    # HiCache L2's host<->device copies (mem_cache/host_pool.py)
    "SGLANG_NEURON_HICACHE_ASYNC_LOAD": CONFIG,
    "SGLANG_NEURON_HICACHE_STATS": DEBUG,
    "SGLANG_NEURON_HICACHE_VERIFY": DEBUG,
    # the HBM the recurrent-state radix checkpoint store leaves free (mem_cache/state_checkpoint_pool.RESERVE_ENV)
    "SGLANG_NEURON_STATE_CKPT_RESERVE_GB": CONFIG,
    # the in-flight bound (backend.device_io), which the giga-kernel decode's result lag takes as its lag
    "SGLANG_NEURON_MAX_INFLIGHT_LAUNCHES": CONFIG,
    # no Neuron device: the CPU tests run the torch paths
    "SGLANG_NEURON_CPU_MODE": DEBUG,
    # diagnostics and profiling
    "SGLANG_NEURON_DEBUG_MESH": DEBUG,
    # GLM-5.2's logit test: teacher forcing and raw logits (models/glm52/logit_probe)
    "SGLANG_NEURON_GLM52_LOGIT_TEST_DIR": DEBUG,
    "SGLANG_NEURON_HOST_TRACE": DEBUG,
    "SGLANG_NEURON_HOST_TRACE_RANKS": DEBUG,
    "SGLANG_NEURON_HOST_TRACE_SPANS": DEBUG,
    "SGLANG_NEURON_WEIGHT_HASH_DIR": DEBUG,
    "SGLANG_NEURON_PP_BOUNDARY_DUMP_DIR": DEBUG,
    "SGLANG_NEURON_PP_BOUNDARY_DUMP_STEPS": DEBUG,
    # the Kimi-K3 test dump, which the e2e tests turn on: the prefill's bytes of a request's pools
    # (models/kimi_k3/prefill_dump.py) and the logits both roles sample from (models/kimi_k3/logits_dump.py)
    "SGLANG_NEURON_KIMI_K3_TEST_DUMP_DIR": DEBUG,
    "SGLANG_NEURON_KIMI_K3_TEST_DUMP_RID": DEBUG,
    "SGLANG_NEURON_PROFILE_DIR": DEBUG,
    "SGLANG_NEURON_PROFILE_RANKS": DEBUG,
    "SGLANG_NEURON_PROFILE_DEVICE_MODE": DEBUG,
    **{f"SGLANG_NEURON_PROFILE_SERVE_{field}": DEBUG for field in ("RANKS", "START", "STEPS")},
    # the SGLang Diffusion platform (sglang_neuron/diffusion/envs.py; docs/model-recipes/flux1-dev.md)
    "SGLANG_NEURON_REQUIRE_NKI_KERNELS": DEBUG,
    "SGLANG_NEURON_DISABLE_NKI_KERNELS": DEBUG,
    # the diffusion platform's Wan2.2 pipelines (diffusion/envs.py, autoencoder_kl_wan.py)
    "SGLANG_NEURON_WAN_VAE_PATCH_PARALLEL_SIZE": CONFIG,
    "SGLANG_NEURON_VAE_MAX_DEVICE_GATHER_BYTES": CONFIG,
    "SGLANG_NEURON_WAN_SAVE_VIDEO_PT": DEBUG,
    # the diffusion platform's LingBot-World pipeline (diffusion/envs.py, lingbot_pipeline.py)
    "SGLANG_NEURON_LINGBOT_TIMING_FILE": DEBUG,
    "SGLANG_NEURON_WAN_SAVE_VIDEO_PT_INDICES": DEBUG,
    # set by the plugin
    "SGLANG_NEURON_SERVING_POLICY": INTERNAL,
    "SGLANG_NEURON_CCOM_PORT_OFFSET": INTERNAL,
    "SGLANG_NEURON_RENDEZVOUS_KEY": INTERNAL,
}


def refuse_unknown(environ=os.environ) -> None:
    unknown = sorted(k for k in environ if k.startswith("SGLANG_NEURON_") and k not in KNOBS)
    if unknown:
        raise ValueError(f"the runtime reads no such environment variable (a typo?): {', '.join(unknown)}; "
                         "remove it from the launch environment (docs/guides/reference-configuration.md lists the ones "
                         "it reads)")


def zero_or_one(name: str, environ=os.environ) -> bool:
    """A boolean knob read as == "1" (off when unset); any other value is refused by name, so a typo never silently
    turns an opt-in off or on."""
    raw = environ.get(name, "0")
    if raw not in ("0", "1"):
        raise ValueError(f"{name}={raw!r}: set it to 0 or 1")
    return raw == "1"


def non_negative_int(name: str, environ=os.environ) -> int:
    """A count read from ``name``: 0 when unset, and a value that is not a non-negative integer refused by name, where
    a bare int() would raise an error that names no variable, or take a negative count."""
    raw = environ.get(name)
    if raw is None:
        return 0
    if not re.fullmatch(r"[0-9]+", raw.strip()):
        raise ValueError(f"{name}={raw!r}: set a non-negative integer, or leave it unset")
    return int(raw)
