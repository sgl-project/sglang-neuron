# SPDX-License-Identifier: Apache-2.0
"""HiCache over a DCP-sharded Neuron MHA pool (Hy3's prefill under --dcp-size): SGLang's
arg_groups.hicache_hook.resolve_hicache_dcp_compatibility refuses every non-MLA model, because SGLang's own MHA host
pool has no DCP index translation. The Neuron MHA pool keeps DCP pages as the MLA pool does (an allocator page is a
virtual page with this rank's block of it in block i, memory_pool.py), and mem_cache/host_pool.NeuronHostKVPool takes
the DCP size and rank as the MLA host pool does (hybrid_host_pool.NeuronHostMLAPool), so only that refusal is lifted:
every other check of the resolver (L3 buffer_only, speculative decoding, LMCache, HiSparse) still runs."""

TARGET = "sglang.srt.arg_groups.hicache_hook.resolve_hicache_dcp_compatibility"


def resolve_on_neuron(original_fn, server_args):
    from sglang.srt.arg_groups import hicache_hook

    dcp = int(getattr(server_args, "dcp_size", 1) or 1)
    if dcp <= 1 or not getattr(server_args, "enable_hierarchical_cache", False):
        return original_fn(server_args)
    saved = hicache_hook.use_mla_backend
    hicache_hook.use_mla_backend = lambda args: True  # the Neuron MHA host pool translates DCP pages itself
    try:
        return original_fn(server_args)
    finally:
        hicache_hook.use_mla_backend = saved


def register(registry, around) -> None:
    registry.register(TARGET, resolve_on_neuron, around)
