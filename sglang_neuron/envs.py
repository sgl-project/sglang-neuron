# SPDX-License-Identifier: Apache-2.0
"""The profile and trace switches more than one module reads, one accessor each."""

from __future__ import annotations

import os


def profile_dir(environ=os.environ) -> str:
    """SGLANG_NEURON_PROFILE_DIR: where the device profiles go."""
    return environ.get("SGLANG_NEURON_PROFILE_DIR") or "/tmp/neuron_profile"


def ranks(name: str, kind: str, environ=os.environ):
    """``name`` as a set of ``kind`` ranks (a comma list of ints >= 0), or None when it is unset or empty; a malformed
    value raises naming the variable."""
    raw = environ.get(name, "")
    if not raw.strip():
        return None
    try:
        parsed = {int(r) for r in raw.split(",") if r.strip()}
    except ValueError:
        parsed = set()
    if not parsed or min(parsed) < 0:
        raise ValueError(f"{name}={raw!r}: give comma-separated {kind} ranks >= 0, e.g. 0 or 0,64")
    return parsed


def profile_ranks(environ=os.environ) -> set:
    """SGLANG_NEURON_PROFILE_RANKS: the local ranks whose runtime device and system profile runs from process start
    (none when unset)."""
    return ranks("SGLANG_NEURON_PROFILE_RANKS", "local", environ) or set()


def profile_device_mode(environ=os.environ) -> str:
    """SGLANG_NEURON_PROFILE_DEVICE_MODE, the device mode of the SGLANG_NEURON_PROFILE_RANKS profile: model (the
    default) profiles one execution of each program, session every execution while the profile is on."""
    value = environ.get("SGLANG_NEURON_PROFILE_DEVICE_MODE") or "model"
    if value not in ("model", "session"):
        raise ValueError(f"SGLANG_NEURON_PROFILE_DEVICE_MODE={value!r}: set it to model or session, or leave it unset")
    return value


def host_trace_dir(environ=os.environ) -> str:
    """SGLANG_NEURON_HOST_TRACE: the directory of the scheduler loop's per-rank timelines; empty when the trace is
    off."""
    return environ.get("SGLANG_NEURON_HOST_TRACE", "")


def host_trace_spans(environ=os.environ) -> list:
    """SGLANG_NEURON_HOST_TRACE_SPANS: dotted names of extra functions the host trace times, each on its own row."""
    return [name.strip() for name in environ.get("SGLANG_NEURON_HOST_TRACE_SPANS", "").split(",") if name.strip()]


def weight_hash_dir(environ=os.environ) -> str:
    """SGLANG_NEURON_WEIGHT_HASH_DIR: where each rank writes the md5 of its loaded weights; empty when off."""
    return environ.get("SGLANG_NEURON_WEIGHT_HASH_DIR", "")


MLA_POOL_LAYOUTS = ("rows", "paged")


def mla_pool_layout(environ=os.environ) -> str:
    """SGLANG_NEURON_MLA_POOL: the pool the platform answers for SGLang's MLA kind. ``rows`` (the default) is
    SGLang's MLA pool on the NeuronCore (mem_cache/memory_pool.NeuronMLATokenToKVPool, one row per token slot);
    ``paged`` is the paged latent pool (mem_cache/mla_pool.NeuronMLATokenToKVPoolKind: zero-on-free pages, HiCache
    host copies). At one block a page and no context parallelism the two hold the same rows byte for byte."""
    value = (environ.get("SGLANG_NEURON_MLA_POOL") or "rows").strip()
    if value not in MLA_POOL_LAYOUTS:
        raise ValueError(f"SGLANG_NEURON_MLA_POOL={value!r}: set it to one of {', '.join(MLA_POOL_LAYOUTS)}, or leave "
                         "it unset")
    return value


def deepseek_v2_max_context(default: int, environ=os.environ) -> int:
    """SGLANG_NEURON_DSV2_MAX_CONTEXT: the longest --context-length a DeepSeek-V2 launch is let through (default
    ``default``, the longest compiled on hardware). A longer one is a probe: its prefill window must still fit in
    SBUF, which the compiler decides."""
    raw = environ.get("SGLANG_NEURON_DSV2_MAX_CONTEXT")
    if raw is None or not raw.strip():
        return default
    if not raw.strip().isdigit() or int(raw) <= 0:
        raise ValueError(f"SGLANG_NEURON_DSV2_MAX_CONTEXT={raw!r}: set a positive token count, or leave it unset")
    return int(raw)
