# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Keep every prefill forward on the chunk grid: whole ``--chunked-prefill-size`` chunks, except a prompt's last.

Upstream's PrefillAdder cuts a forward wherever its token budget runs out. With C the chunk size, a model whose prefill
keeps the chunk grid (models/base.segmented_prefill_cache_policy calls ``keep``) instead runs a cold prompt as [0, C),
[C, 2C), ..., and a prompt with a radix hit at page m as [m, m + C), [m + C, m + 2C), ...

* ``Scheduler.init_deterministic_inference_config`` (wrapped): on a scheduler that prefills, the truncation grid is the
  chunk. SGLang's deterministic inference sets the same grid, so a launch with both is refused.
* ``PrefillAdder.__init__`` selects SGLang's strict continuation grid at the Neuron chunk size. SGLang's memory budget
  admits whole chunks and can wait for enough memory for the next chunk.
* A pass that carries a chunked request, placed or not, admits nothing beside it: general_plugin's rule of one request
  per prefill forward refuses every other request and preempts nothing for them. The scheduler counts every batch built
  while a request is chunked as one of that request's middle chunks. A batch built beside it would turn its real last
  chunk into a middle one, and the request would never finish.
* ``PrefillAdder.add_one_req_ignore_eos`` (wrapped; the ignore-EOS path without a radix tree): the cut is the budget's
  whole chunks, and a request whose cut would be empty waits for the next pass.

Source guards cover the scheduler and adder entry points this integration uses."""

from __future__ import annotations

from typing import Optional

_POLICY = "sglang.srt.managers.schedule_policy"
_SCHEDULER = "sglang.srt.managers.scheduler"
_TARGETS_MD5 = {
    (_POLICY, "PrefillAdder.__init__"): "c99b09b6e48dcb04e3fb22ec815d2c97",
    (_POLICY, "PrefillAdder._add_chunked_req"): "6f84570a06ad2fef0058b18287fedc73",
    (_POLICY, "PrefillAdder.grid_continuation_limit"): "f53cb6aca04bf555615b73fcd7aee823",
    (_POLICY, "PrefillAdder.add_one_req_ignore_eos"): "e7c573386398008e5a002efee1389a61",
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


def init_prefill_adder(original_fn, self, *args, **kwargs):
    result = original_fn(self, *args, **kwargs)
    unit = grid_unit()
    if unit is not None:
        self.mamba_chunk_grid = unit
        self.strict_chunk_grid = True
    return result


def register(registry, around) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
    registry.register(f"{_POLICY}.PrefillAdder.__init__", init_prefill_adder, around)
    registry.register(f"{_POLICY}.PrefillAdder.add_one_req_ignore_eos", add_one_req_ignore_eos, around)
    registry.register(f"{_SCHEDULER}.Scheduler.init_deterministic_inference_config",
                      init_deterministic_inference_config, around)
