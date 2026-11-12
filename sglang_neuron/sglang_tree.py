# SPDX-License-Identifier: Apache-2.0
"""How the plugin reads SGLang's parallel topology (SGLang at the revision docs/getting-started/setup-guide.md pins).

SGLang keeps a process's ranks and its resolved parallel configuration in the process's runtime context, published
when the process spawns. The plugin reads them only through the functions here, which are also the tests' seams:

* parallel_of(owner) returns a ModelRunner's or Scheduler's ranks. Tests patch it to read a test double's own ranks
  (owner.ps).
* parallel_config() returns the resolved configuration. Tests patch it with a configuration fake.

On upstream SGLang both are get_parallel(); they stay separate so a test that patches one does not patch the other.
runs_dp_attention and shares_prefixes keep in one place the upstream reads that the plugin's DP-attention and
prefix-sharing decisions follow."""
from __future__ import annotations


def parallel_of(owner):
    """The parallel ranks and widths (tp_rank, tp_size, attn_dp_rank, attn_dp_size, ...) of a ModelRunner or
    Scheduler. Upstream SGLang reads them from the process's runtime context and ignores ``owner``; the parameter is
    there so tests can patch this function to return each owner's own ranks."""
    from sglang.srt.runtime_context import get_parallel

    return get_parallel()


def rank_label(ps):
    """A rank's label in its trace and profile file names (``ps`` is what parallel_of returns). It is the TP rank,
    prefixed with the DP replica on a --dp-size server without DP attention, and with the pipeline stage under PP,
    because there every replica or stage counts its TP ranks from 0."""
    tp = getattr(ps, "tp_rank", "?")
    dp_rank = getattr(ps, "dp_rank", None)
    if (int(getattr(ps, "dp_size", 1) or 1) > 1 and int(getattr(ps, "attn_dp_size", 1) or 1) == 1
            and dp_rank is not None):
        return f"dp{dp_rank}_tp{tp}"
    if int(getattr(ps, "pp_size", 1) or 1) > 1:
        return f"pp{getattr(ps, 'pp_rank', '?')}_tp{tp}"
    return tp


def parallel_config():
    """The resolved parallel configuration (tp_size, ep_size, dp_size, attn_dp_size, ...), available in any process
    once the runtime context is published."""
    from sglang.srt.runtime_context import get_parallel

    return get_parallel()


def dp_widths(cfg) -> tuple:
    """(replicas, attention-DP groups) of a ServerArgs or a resolved parallel configuration.

    Under the deprecated --enable-dp-attention spelling, --dp-size is the attention-DP width. Upstream's resolution
    moves it to attn_dp_size and leaves dp_size as the replica count (parallel_hook's _handle_deprecated_dp_attention).
    This function reads a configuration that is not resolved yet the same way, and is the only reader of
    enable_dp_attention."""
    dp = int(getattr(cfg, "dp_size", 1) or 1)
    attn = int(getattr(cfg, "attn_dp_size", 1) or 1)
    if attn == 1 and bool(getattr(cfg, "enable_dp_attention", False)):
        return 1, dp
    return dp, attn


def attention_dp_size(cfg) -> int:
    """The number of attention-DP groups, as dp_widths reads it (1 means no DP attention)."""
    return dp_widths(cfg)[1]


def scheduler_batch(cfg):
    """The most requests one scheduler runs: --max-running-requests divided over the attention-DP groups, at least 1,
    the way upstream's KVCacheConfigurator.resolve_max_num_reqs divides it. Each --dp-size replica runs the full
    count. None without --max-running-requests."""
    max_running = getattr(cfg, "max_running_requests", None)
    if max_running is None:
        return None
    return max(1, int(max_running) // attention_dp_size(cfg))


def runs_dp_attention(parallel) -> bool:
    """Whether the process runs SGLang's DP-attention paths, read as upstream SGLang's scheduler reads it:
    attn_dp_enabled (an attention-DP width above one, or an elastic scale joiner). ``parallel`` is parallel_config()."""
    return bool(parallel.attn_dp_enabled)


def shares_prefixes(tree_cache) -> bool:
    """Whether the tree cache keeps a request's prefix for other requests (a radix tree, not SGLang's chunk cache):
    upstream SGLang's supports_prefix_sharing."""
    return bool(tree_cache.supports_prefix_sharing())
