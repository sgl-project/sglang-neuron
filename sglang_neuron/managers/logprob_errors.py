# SPDX-License-Identifier: Apache-2.0
"""Two hooks for aborted requests.

convert_logprob_style: an aborted request's last packet can carry no logprobs, and SGLang's converter then raises
TypeError. The hook renders the logprob history the request already has instead.

finish_prefill_aborts: when a PD prefill's KV transfer completes, SGLang replaces a pending abort with a successful
length=0 finish. The hook applies the abort first."""

_VALUES = (
    "input_token_logprobs_val", "output_token_logprobs_val",
    "input_top_logprobs_val", "output_top_logprobs_val",
    "input_token_ids_logprobs_val", "output_token_ids_logprobs_val",
    "input_top_logprobs_val_flat",
)


def convert_logprob_style(original, manager, meta_info, state, top_logprobs_num,
                          token_ids_logprob, return_text_in_logprobs, recv_obj,
                          recv_obj_index, *args, **kwargs):
    """AROUND TokenizerManager.convert_logprob_style. An abort whose packet has no logprob rows is rendered from the
    request's history. ``args`` and ``kwargs`` pass SGLang's ``finished`` (a keyword) on to the converter and its
    formatter."""
    reason = meta_info.get("finish_reason")
    if isinstance(reason, dict) and reason.get("type") == "abort":
        has_row = any(
            values is not None
            and len(values) > recv_obj_index
            and values[recv_obj_index] is not None
            and len(values[recv_obj_index]) > 0
            for values in (getattr(recv_obj, name, None) for name in _VALUES)
        )
        if not has_row:
            # Rejected requests have return_logprob disabled in the scheduler,
            # while TokenizerManager retains the original client preference.
            # Render the existing history through SGLang's normal formatter.
            return manager.add_logprob_to_meta_info(
                meta_info, state, top_logprobs_num, token_ids_logprob,
                return_text_in_logprobs, *args, **kwargs,
            )
    return original(manager, meta_info, state, top_logprobs_num,
                    token_ids_logprob, return_text_in_logprobs, recv_obj,
                    recv_obj_index, *args, **kwargs)


def finish_prefill_aborts(original, scheduler, *args, **kwargs):
    """AROUND SchedulerDisaggregationPrefillMixin.process_disagg_prefill_inflight_queue: keep a pending abort when the
    request's KV transfer completes."""
    from sglang.srt.managers.schedule_batch import FINISH_ABORT

    for req in scheduler.disagg_prefill_inflight_queue:
        if (
            not req.pending_bootstrap
            and not req.finished()
            and isinstance(req.to_finish, FINISH_ABORT)
        ):
            # Decode applies this state transition while consuming results.
            # Prefill otherwise replaces it with a successful length=0 finish.
            # Keep the existing transfer polling and resource-release order.
            req.update_finish_state()
    return original(scheduler, *args, **kwargs)
