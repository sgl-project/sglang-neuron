# SPDX-License-Identifier: Apache-2.0
"""``--radix-cache-backend neuron-chunk``: a prefix cache keyed on whole prefill chunks, so a hit is bit-identical to a
cold prefill. The tree is sglang_neuron/mem_cache/chunk_keyed_tree.py.

SGLang's default tree keys 128-token pages; this one keys whole ``--chunked-prefill-size`` chunks C. Both run the same
forwards (prefill_chunk_grid cuts every forward to whole chunks). Trade-offs:

* A hit is the chunk floor of the shared prefix: up to C - 128 tokens less reuse, and prompts of C tokens or fewer
  never hit. Inserts stop at the prompt's last whole chunk, so an aggregated server never reuses a previous turn's
  outputs. Eviction frees whole chunks.
* In return a hit runs a cold prefill's own forwards from a chunk boundary, over KV the same on-grid forwards wrote, so
  its prefill and first token are bit-identical to a cold run. Later tokens match only where the decode batches do.
* The prefill block table drops the default tree's extra tile (graph_runner's prefill_table_pad), so prefill compiles
  its own graph.

A PD decode with ``--disaggregation-decode-enable-radix-cache`` can use it too, keyed on its own per-rank chunk and
paired only with prefills whose chunk divides it (sglang_patches.decode_radix_prefill_chunks).

SGLang looks the backend up by name in its radix-cache registry, so every process of a launch that selects it registers
it (general_plugin._register, from serving_policy.chunk_keyed_tree). ``validate_launch`` refuses launches it does not
serve. The tree's pins against upstream SGLang 43881a7 are in mem_cache.chunk_keyed_tree.

Upstream: needs a chunk-keyed radix tree plus a whole-chunk prefill grid; until then this stays a Neuron backend."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# the --radix-cache-backend name the plugin registers
BACKEND = "neuron-chunk"
# the recurrent-state checkpoints a cached prompt keeps on its root-to-tail path on a prefill of a model that keeps one
# at every chunk end, where the launch leaves --mamba-max-states-per-path at SGLang's -1 (default_states_per_path)
DEFAULT_STATES_PER_PATH = 4


def selected(server_args) -> bool:
    """Whether the launch keys its prefix cache on whole prefill chunks (``--radix-cache-backend neuron-chunk``)."""
    return getattr(server_args, "radix_cache_backend", None) == BACKEND


def default_states_per_path(server_args, spec) -> int | None:
    """``--mamba-max-states-per-path`` for a prefill on this tree of a model that keeps a recurrent-state checkpoint at
    every chunk end (``spec.chunk_state_checkpoints``), where the launch leaves it at SGLang's -1 (unlimited):
    DEFAULT_STATES_PER_PATH. Returns the value set, or None (another launch, or a launch that sets it).

    Unlimited, every chunk a prompt computes keeps its state, and the pool's LRU frees only leaves, so the interior
    states of long prompts fill the slots and the states at the prompts' ends are freed first. A request that extends
    such a prompt then matches only up to the deepest state still kept and recomputes the rest. Capped, SGLang drops
    the shallowest interior states of the inserted path after each insert (the tail, forks and locked nodes stay, and
    so do the chunks' KV and any host copy HiCache wrote), so the slots keep the states at the ends of the cached
    prompts. A launch that sets -1 explicitly reads the same as one that leaves it unset."""
    from sglang_neuron import serving_policy

    if not (spec.chunk_state_checkpoints and selected(server_args)
            and serving_policy.launch_role(server_args) == "prefill"):
        return None
    if int(getattr(server_args, "mamba_max_states_per_path", -1)) != -1:
        return None
    server_args.mamba_max_states_per_path = DEFAULT_STATES_PER_PATH
    logger.info("[neuron] chunk state checkpoints on the %s tree: --mamba-max-states-per-path %d", BACKEND,
                DEFAULT_STATES_PER_PATH)
    return DEFAULT_STATES_PER_PATH


def validate_launch(server_args, architecture: str, spec) -> None:
    """Refuse any other ``--radix-cache-backend``, and neuron-chunk on a launch it does not serve. ``spec`` is the
    model's catalog entry."""
    from sglang_neuron.sglang_patches import kimi_prompt

    backend = getattr(server_args, "radix_cache_backend", None)
    if backend == kimi_prompt.BACKEND:
        kimi_prompt.validate_launch(server_args, architecture, spec)
        return
    if backend is None:
        # a model that keeps its recurrent state only at chunk ends (ModelSpec.chunk_state_checkpoints) has no state
        # where SGLang's default tree, keyed on KV pages, hits inside a chunk
        if spec.chunk_state_checkpoints and not getattr(server_args, "disable_radix_cache", False):
            raise ValueError(f"{architecture} keeps its recurrent-state checkpoints at prefill chunk ends, and "
                             "SGLang's default radix tree hits at KV pages inside a chunk: launch with "
                             f"--radix-cache-backend {BACKEND} or --disable-radix-cache")
        return
    if backend != BACKEND:
        raise ValueError(f"--radix-cache-backend {backend}: sglang-neuron serves SGLang's default radix tree, keyed on "
                         f"KV pages (launch without --radix-cache-backend), {BACKEND}, keyed on whole prefill "
                         f"chunks, or {kimi_prompt.BACKEND} on a Kimi-K3 PD prefill")
    # SGLang resolves a decode's --disable-radix-cache from --disaggregation-decode-enable-radix-cache before this check
    decode = str(getattr(server_args, "disaggregation_mode", "null") or "null") == "decode"
    unserved = {
        "--disable-radix-cache, which asks for no radix tree":
            not decode and bool(getattr(server_args, "disable_radix_cache", False)),
        # mem_cache.registry.create_tree_cache builds a named backend and never reaches the default factory's chain
        "--enable-lmcache: SGLang builds the named backend in place of the LMCache tree the flag selects, so the flag "
        "would be dropped": bool(getattr(server_args, "enable_lmcache", False)),
        "--enable-flexkv: SGLang builds the named backend in place of the FlexKV cache the flag selects, so the flag "
        "would be dropped": bool(getattr(server_args, "enable_flexkv", False)),
        "--disaggregation-mode decode without --disaggregation-decode-enable-radix-cache, on which SGLang runs no "
        "radix tree": decode and not getattr(server_args, "disaggregation_decode_enable_radix_cache", False),
        "--enable-streaming-session on a decode: a streaming session keeps a finished turn's KV, the outputs its "
        "decode steps wrote included, for its next turn to resume at, where a cold run prefills them":
            decode and bool(getattr(server_args, "enable_streaming_session", False)),
        f"{architecture}, whose prefill window holds the whole request, so its chunk is its context and no prompt "
        "shares a whole chunk with another": spec.prefill_holds_prior,
    }
    asked = [name for name, on in unserved.items() if on]
    if asked:
        raise ValueError(f"--radix-cache-backend {BACKEND} is not served with {'; '.join(asked)}; launch without "
                         "--radix-cache-backend, which keeps SGLang's default radix tree")
    # SGLang syncs the stages' KV token count but sizes a radix tree's state pool from each stage's own free memory;
    # stages with different slot counts evict checkpoints at different requests and then match different prefixes
    if (spec.chunk_state_checkpoints and int(getattr(server_args, "pp_size", 1) or 1) > 1
            and getattr(server_args, "max_mamba_cache_size", None) is None):
        raise ValueError(f"--radix-cache-backend {BACKEND} with --pp-size {server_args.pp_size}: {architecture}'s "
                         "stages keep state checkpoints in state pools SGLang sizes from each stage's free memory, "
                         "and stages with different slot counts would evict different checkpoints and match "
                         "different prefixes; launch with --max-mamba-cache-size")
    if decode:
        _check_decode_chunk(server_args)


def _check_decode_chunk(server_args) -> None:
    """Refuse a decode whose tree chunk (--chunked-prefill-size split over its attention-DP groups, as SGLang splits
    it) is not a positive multiple of the KV page."""
    from sglang_neuron import sglang_tree
    from sglang_neuron.platform import PAGE_SIZE

    flag = getattr(server_args, "chunked_prefill_size", None)
    groups = sglang_tree.attention_dp_size(server_args)
    chunk = None if flag is None else int(flag) // groups
    if chunk is not None and chunk > 0 and chunk % PAGE_SIZE == 0:
        return
    over = f" over its {groups} attention-DP groups" if groups > 1 else ""
    times = f" times {groups}" if groups > 1 else ""
    raise ValueError(f"--radix-cache-backend {BACKEND} on a decode keys its tree on whole chunks of its "
                     f"--chunked-prefill-size{over}, which must be a positive multiple of the {PAGE_SIZE}-token KV "
                     f"page (--chunked-prefill-size {'unset' if flag is None else flag}): launch the decode with "
                     f"--chunked-prefill-size set to its prefills' chunk{times}")


def register() -> None:
    """Register the backend in SGLang's radix-cache registry (mem_cache.chunk_keyed_tree checks its pins first)."""
    from sglang_neuron.mem_cache import chunk_keyed_tree

    chunk_keyed_tree.register_backend()
