# SPDX-License-Identifier: Apache-2.0
"""``--radix-cache-backend kimi-prompt``: the Kimi-K3 prompt cache on the Kimi-K3 PD prefill server. The cache is
sglang_neuron/mem_cache/kimi_prompt_backend.py (END entries at any token, the KDA state slot adopted, chunk-end
entries, an optional host tier); this module has the launch side.

Against the chunk-keyed tree (``neuron-chunk``): a hit ends at the exact end of an earlier prompt or at a chunk end of
a running one, not at the last whole chunk, so a next turn of a session computes only its new tokens. The host tier
keeps whole entries (their pages, their drafter rows and their KDA state), so a host hit is exact too.

SGLang looks the backend up by name in its radix-cache registry, so every process of a launch that selects it registers
it (general_plugin, from serving_policy.kimi_prompt_cache). ``validate_launch`` refuses the launches it does not serve.

Upstream: needs a prefix cache that keeps a recurrent state at any token; until then this stays a Neuron backend.

Example:
    >>> from types import SimpleNamespace
    >>> selected(SimpleNamespace(radix_cache_backend="kimi-prompt"))
    True
"""

from __future__ import annotations

BACKEND = "kimi-prompt"


def selected(server_args) -> bool:
    """Whether the launch selects the Kimi-K3 prompt cache."""
    return getattr(server_args, "radix_cache_backend", None) == BACKEND


def validate_launch(server_args, architecture: str, spec) -> None:
    """Refuse ``kimi-prompt`` on a launch it does not serve: it keeps END entries over a PD prefill's KV pages and the
    state slots of a model whose prefill graph writes the state each forward ends with (``spec``, the catalog entry:
    ``chunk_state_checkpoints``).

    Raises:
        ValueError: The launch is not a PD prefill of such a model, or it asks for a cache feature that the backend
            replaces.
    """
    mode = str(getattr(server_args, "disaggregation_mode", "null") or "null")
    pp = int(getattr(server_args, "pp_size", 1) or 1)
    unserved = {
        f"{architecture}, whose prefill graph keeps no recurrent state at its forward ends":
            not spec.chunk_state_checkpoints,
        f"--disaggregation-mode {mode}: the backend serves the PD prefill only": mode != "prefill",
        "--disable-radix-cache, which asks for no prefix cache": bool(getattr(server_args, "disable_radix_cache",
                                                                              False)),
        "--enable-hierarchical-cache: the backend has its own host tier (SGLANG_NEURON_KIMI_PROMPT_HOST_SLOTS)":
            bool(getattr(server_args, "enable_hierarchical_cache", False)),
        "--enable-lmcache": bool(getattr(server_args, "enable_lmcache", False)),
        "--enable-flexkv": bool(getattr(server_args, "enable_flexkv", False)),
        "--enable-streaming-session": bool(getattr(server_args, "enable_streaming_session", False)),
        # SGLang sizes the state pool of each stage from its own free memory; stages with different slot counts would
        # evict different entries and then match different prefixes
        f"--pp-size {pp} without --max-mamba-cache-size":
            pp > 1 and getattr(server_args, "max_mamba_cache_size", None) is None,
    }
    asked = [name for name, on in unserved.items() if on]
    if asked:
        raise ValueError(f"--radix-cache-backend {BACKEND} is not served with {'; '.join(asked)}. Launch the Kimi-K3 "
                         "PD prefill with it, or launch with --radix-cache-backend neuron-chunk.")


def register(registry=None, around=None) -> None:
    """Register the backend in SGLang's radix-cache registry and, given the hook registry, the tracking rule of its
    chunk ends (kimi_prompt_backend.track_entry)."""
    from sglang_neuron.mem_cache import kimi_prompt_backend

    kimi_prompt_backend.register_backend()
    if registry is not None:
        kimi_prompt_backend.register_hooks(registry, around)
