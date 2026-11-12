# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""The decode capture list is the runtime's own buckets, not upstream's alignment-filtered subset.

Upstream's ``get_batch_sizes_to_capture`` builds ``(capture_bs, compile_bs)`` for a NVIDIA decode runner: it reads
``cuda_graph_config.decode.bs`` and keeps only sizes aligned to the attention-TP gathered buffer (and CP), padding the
pool's size in. A Neuron decode's buckets are the launch's own ladder (platform's decode-bs hook writes them into
``cuda_graph_config.decode.bs``): the megakernel gathers inside one compiled graph, so every rung below attention-TP
that upstream's filter would drop is a real graph the runtime serves. On a DP-attention decode whose ladder tops below
attn-TP the filter drops every rung, and capture refuses to start (``assert len(capture_bs) > 0``).

The replacement keeps only the pool's bound: sizes of ``cuda_graph_config.decode.bs`` no larger than the request
pool. torch.compile has no meaning over the Neuron graphs, so ``compile_bs`` is empty.

Targets are md5-pinned against upstream SGLang 43881a7.

Upstream: SGLang would need a platform hook on its capture-list filter; until then this stays a Neuron module."""

from __future__ import annotations

_TARGET = "sglang.srt.model_executor.runner.base_cuda_graph_runner"
_TARGETS_MD5 = {
    (_TARGET, "get_batch_sizes_to_capture"): "a2622a98bf5299b8b1fd64b1540d2fcc",
}


def get_batch_sizes_to_capture(model_runner, captured_req_width: int = 1, *, gathered_buffer_required=None):
    """The decode runtime's buckets, minus only what the request pool cannot hold; compile_bs stays empty."""
    from sglang.srt.runtime_context import get_exec

    pool = model_runner.req_to_token_pool.size
    capture_bs = sorted({int(bs) for bs in get_exec().graph.cuda_graph_config.decode.bs if 0 < int(bs) <= pool})
    assert capture_bs, (f"the decode runtime's buckets {get_exec().graph.cuda_graph_config.decode.bs} hold no size "
                        f"the request pool ({pool} rows) admits")
    return capture_bs, []


def register(registry, replace) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
    registry.register(f"{_TARGET}.get_batch_sizes_to_capture", get_batch_sizes_to_capture, replace)
