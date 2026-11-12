# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Keep every prefill forward on the chunk grid: whole ``--chunked-prefill-size`` chunks, except a prompt's last.

Upstream's PrefillAdder cuts a forward wherever its token budget runs out. With C the chunk size, a model whose prefill
keeps the chunk grid (models/base.segmented_prefill_cache_policy calls ``keep``) instead runs a cold prompt as [0, C),
[C, 2C), ..., and a prompt with a radix hit at page m as [m, m + C), [m + C, m + 2C), ...

* ``Scheduler.init_deterministic_inference_config`` (wrapped): on a scheduler that prefills, the truncation grid is the
  chunk. SGLang's deterministic inference sets the same grid, so a launch with both is refused.
* ``PrefillAdder.add_chunked_req`` (upstream's method plus the grid cut): a continuation that does not end the prompt is
  cut to whole chunks. If less than a chunk fits, it takes one chunk when the budget can allocate it, and otherwise
  waits for the next pass, as upstream's continuation does when no budget is left.
* A pass that carries a chunked request, placed or not, admits nothing beside it: general_plugin's rule of one request
  per prefill forward refuses every other request and preempts nothing for them. The scheduler counts every batch built
  while a request is chunked as one of that request's middle chunks. A batch built beside it would turn its real last
  chunk into a middle one, and the request would never finish.
* ``PrefillAdder.add_one_req_ignore_eos`` (wrapped; the ignore-EOS path without a radix tree): the cut is the budget's
  whole chunks, and a request whose cut would be empty waits for the next pass.

Targets are md5-pinned against upstream SGLang 43881a7.

Upstream: SGLang would need a prefill truncation grid of whole chunks as an option of its own, beside deterministic
inference's; until then this stays a Neuron module."""

from __future__ import annotations

from typing import Optional

_POLICY = "sglang.srt.managers.schedule_policy"
_SCHEDULER = "sglang.srt.managers.scheduler"
_TARGETS_MD5 = {
    (_POLICY, "PrefillAdder.add_chunked_req"): "a79936af415e0832070b61a7ab001fee",
    (_POLICY, "PrefillAdder.add_one_req_ignore_eos"): "8efd284226e3ded4c719781c67910e26",
    (_SCHEDULER, "Scheduler.init_deterministic_inference_config"): "ca9dac95c6f151f6396f34f6f1f80820",
}

_KEPT = False


def keep() -> None:
    """Put this scheduler process's prefill forwards on the chunk grid. Call it before the scheduler builds its
    truncation grid."""
    global _KEPT
    _KEPT = True


def kept() -> bool:
    return _KEPT


def grid_unit() -> Optional[int]:
    """The grid unit on this scheduler process: --chunked-prefill-size when the grid is kept and the scheduler prefills,
    else None. A PD decode server never truncates a prefill, so it gets None."""
    if not _KEPT:
        return None
    from sglang.srt.runtime_context import get_disagg, get_schedule

    if str(get_disagg().disaggregation_mode) == "decode":
        return None
    chunk = int(get_schedule().chunked_prefill_size or 0)
    if chunk <= 0:
        raise ValueError(f"the prefill chunk grid needs --chunked-prefill-size > 0, got {chunk}")
    return chunk


def init_deterministic_inference_config(original_fn, self, *args, **kwargs):
    ret = original_fn(self, *args, **kwargs)
    chunk = grid_unit()
    if chunk is None:
        return ret
    from sglang.srt.runtime_context import get_exec

    if get_exec().deterministic.enable_deterministic_inference:
        raise ValueError("the prefill chunk grid and --enable-deterministic-inference both set the truncation grid; "
                         "launch without --enable-deterministic-inference")
    self.truncation_align_size = chunk
    return ret


def add_one_req_ignore_eos(original_fn, self, req, *args, **kwargs):
    unit = grid_unit()
    budget = self.rem_chunk_tokens
    if unit is None or budget is None or len(req.full_untruncated_fill_ids) - req.prefix_len <= budget:
        return original_fn(self, req, *args, **kwargs)
    from sglang.srt.managers.schedule_policy import AddReqResult

    on_grid = max(budget, 0) // unit * unit
    if on_grid == 0:
        return AddReqResult.OTHER
    # Upstream's method cuts the request to rem_chunk_tokens, then subtracts the cut from rem_chunk_tokens. Give it
    # only the budget's whole chunks, and afterwards restore the full budget less the cut.
    self.rem_chunk_tokens = on_grid
    try:
        return original_fn(self, req, *args, **kwargs)
    finally:
        self.rem_chunk_tokens = budget - (on_grid - self.rem_chunk_tokens)


def add_chunked_req(self, req):
    """Upstream SGLang's PrefillAdder.add_chunked_req (43881a7, unchanged but for the marked grid cut)."""
    from sglang.srt.layers.cp.base import is_cp_sharded_padded
    from sglang.srt.managers.schedule_policy import CLIP_MAX_NEW_TOKENS

    unit = grid_unit()
    if self.dllm_config is not None:
        _rem_tokens = self._get_dllm_remain_tokens(req)
    else:
        _rem_tokens = self.memory_budget.available_chunk_tokens(
            self.rem_chunk_tokens
        )
        if _rem_tokens is None:
            return req
        if self.kv_shard_granule:
            # Align the absolute boundary to keep pages within one scratch region.
            # If no budget remains, use a full chunk to make progress.
            prefix_len = req.prefix_len
            floored = (
                prefix_len + _rem_tokens
            ) // self.kv_shard_granule * self.kv_shard_granule - prefix_len
            _rem_tokens = floored if floored > 0 else self.rem_chunk_tokens

    # A mid-chunk rank prefills this pass regardless of the delayer
    # verdict, so report prefillable=True and ignore the result.
    if self.prefill_delayer_single_pass is not None:
        self.prefill_delayer_single_pass.negotiate_should_allow_prefill(
            local_prefillable=True,
            running_batch=self.running_batch.batch_size(),
            max_prefill_bs=self.max_prefill_bs,
            max_running_requests=self.max_running_requests,
            waiting_queue_len=self.waiting_queue_len,
        )

    if self.chunked_req_limit is not None:
        assert self.chunked_req_limit > 0
        _rem_tokens = min(_rem_tokens, self.chunked_req_limit)

    cand_extend_input_len = len(req.full_untruncated_fill_ids) - req.prefix_len
    prefix_tokens = 0
    if is_cp_sharded_padded():
        # Existing prefix padding can supply part of this chunk's KV rows.
        prefix_tokens = req.prefix_len
    _rem_tokens = self.memory_budget.fit_chunk(
        extend_input_len=cand_extend_input_len,
        max_new_tokens=self._swa_new_tokens(req),
        chunk_limit=_rem_tokens,
    )
    if _rem_tokens is None:
        return req
    truncated = cand_extend_input_len > _rem_tokens
    new_len = self.floor_chunked_tokens(
        min(cand_extend_input_len, _rem_tokens), cand_extend_input_len
    )
    # --- the grid cut (prefill_chunk_grid): whole chunks unless the pass ends the prompt
    if unit and truncated:
        # Round the cut down to whole chunks; a cut below one chunk becomes one chunk. If the KV pool can allocate it,
        # the pass's chunk budget was the limit, and the overrun is bounded by one chunk, as upstream's own floor is.
        # If the pool cannot, the request waits unplaced for the next pass.
        new_len = min(cand_extend_input_len, max(new_len // unit, 1) * unit)
        if not self.memory_budget.can_allocate_prefill(
            paged_input=self.ceil_paged_tokens(new_len) + self.per_req_token_overhead,
            extend_input_len=new_len,
            max_new_tokens=self._swa_new_tokens(req),
            chunk_limit=self.rem_chunk_tokens,
        ):
            return req
        truncated = new_len < cand_extend_input_len
    # --- end of the grid cut
    request_slot_reserve = self._request_slot_reserve_for_req(
        req, extend_input_len=new_len
    )
    if request_slot_reserve and not self.memory_budget.can_allocate_prefill(
        paged_input=(
            self.ceil_paged_tokens(new_len)
            + self.per_req_token_overhead
            + request_slot_reserve
        ),
        extend_input_len=new_len,
        max_new_tokens=self._swa_new_tokens(req),
        chunk_limit=self.rem_chunk_tokens,
    ):
        return req
    # The continuing chunk must fit. Keep reservation outside assert for -O.
    reserved = self._kv_shard_reserve_scratch(
        prefix_len=req.prefix_len, extend_len=new_len
    )
    if not reserved:
        raise RuntimeError("chunked request exceeds the sharded assembly scratch")
    req.extend_end = req.prefix_len + new_len
    self.can_run_list.append(req)
    self._update_prefill_budget(
        prefix_tokens,
        req.extend_len,
        (
            min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS)
            if not truncated
            else 0
        ),
        req.retracted_stain,
        state_slot_reserve=self._state_slot_reserve_for_req(req),
        request_slot_reserve=request_slot_reserve,
        is_chunked_continuation=True,
    )

    # Return if chunked prefill not finished
    return req if truncated else None


def register(registry, around, replace) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
    registry.register(f"{_POLICY}.PrefillAdder.add_chunked_req", add_chunked_req, replace)
    registry.register(f"{_POLICY}.PrefillAdder.add_one_req_ignore_eos", add_one_req_ignore_eos, around)
    registry.register(f"{_SCHEDULER}.Scheduler.init_deterministic_inference_config",
                      init_deterministic_inference_config, around)
