# SPDX-License-Identifier: Apache-2.0
"""SGLang's speculative (spec-v2) decode KV allocation, on host tensors.

The overlap-scheduled speculative worker (spec-v2) allocates each step's KV with ``get_last_loc`` and
``assign_req_to_token_pool_func``, which launch Triton kernels on any engine but SGLang's CPU build. This platform keeps
SGLang's tensors on the host, where no Triton driver runs, so both are replaced by their torch forms, which compute the
same values: SGLang's own ``get_last_loc_torch``, and the per-request row writes the Triton kernel makes. The plain
decode never calls them."""

from __future__ import annotations

_TARGETS = (
    ("sglang.srt.mem_cache.allocation", "get_last_loc"),
    ("sglang.srt.mem_cache.allocation", "assign_req_to_token_pool_func"),
)


def get_last_loc(req_to_token, req_pool_indices_tensor, prefix_lens_tensor):
    """REPLACE: SGLang's torch form (allocation.get_last_loc_torch)."""
    from sglang.srt.mem_cache.allocation import get_last_loc_torch

    return get_last_loc_torch(req_to_token, req_pool_indices_tensor, prefix_lens_tensor)


def assign_req_to_token_pool_func(req_pool_indices, req_to_token, start_offset, end_offset, out_cache_loc,
                                  batch_size: int):
    """REPLACE: row ``req_pool_indices[i]`` of ``req_to_token`` takes the next ``end_offset[i] - start_offset[i]``
    slots of ``out_cache_loc`` at ``[start_offset[i], end_offset[i])``, request by request in batch order, in one
    indexed write (a write per request costs host time for each request, every speculative decode step)."""
    import torch

    starts = start_offset.reshape(-1)[:batch_size].long()
    rows = req_pool_indices.reshape(-1)[:batch_size].long()
    lens = (end_offset.reshape(-1)[:batch_size].long() - starts).clamp(min=0)
    taken = int(lens.sum())
    if taken != out_cache_loc.shape[0]:
        raise RuntimeError(f"spec-v2 allocation wrote {taken} of its {out_cache_loc.shape[0]} new KV slots")
    # each new slot's request row and its position in that row: the request's start plus its rank among the slots
    first = torch.cumsum(lens, 0) - lens
    cols = starts.repeat_interleave(lens) + torch.arange(taken) - first.repeat_interleave(lens)
    req_to_token[rows.repeat_interleave(lens), cols] = out_cache_loc.to(req_to_token.dtype)


def register(registry, replace) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_pinned(_TARGETS, __name__)
    registry.register("sglang.srt.mem_cache.allocation.get_last_loc", get_last_loc, replace)
    registry.register("sglang.srt.mem_cache.allocation.assign_req_to_token_pool_func", assign_req_to_token_pool_func,
                      replace)


__all__ = ["get_last_loc", "assign_req_to_token_pool_func", "register"]
