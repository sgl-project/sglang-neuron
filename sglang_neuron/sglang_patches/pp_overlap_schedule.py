# SPDX-License-Identifier: Apache-2.0
"""Keep SGLang's overlap schedule on under PP where the serving policy runs the overlap-order PD decode PP loop, and
let run_batch's overlap branch run a pipeline stage.

Upstream turns the overlap schedule off for every pipelined launch (``_pipeline_parallel_overlap_disable``), asserts
it off (``check_pipeline_parallel_compat``), and its ``run_batch`` overlap branch calls the worker without the stage's
proxy tensors ("FIXME: pp is not compatible with overlap"). sglang_patches/pp_decode_overlap_loop needs that branch: it
launches a step before processing the previous one, with the first stage's input ids resolved through SGLang's
FutureMap from the tokens the last stage relays over the PP seam (model_executor/async_execution). Where
``ServingPolicy.overlap_schedule`` is on for a pipelined launch:

* the disable pass keeps the schedule, and the compatibility check still runs its other checks;
* ``Scheduler.run_batch`` is rebuilt to pass ``pp_proxy_tensors`` to the TP worker in the overlap branch
  (``RUN_BATCH_SUBSTITUTIONS``), under upstream's stage decorator. The overlap schedule under PP runs without
  speculative decoding, so the worker is always the TP worker, which takes the proxy;
* a non-last stage samples nothing, so ``_copy_result_to_cpu`` only records its copy event.

A pipelined PD prefill keeps the overlap schedule off (its last stage is async its own way,
sglang_patches/pp_last_stage_async), so these hooks change nothing there.

Upstream: passing the proxy in the overlap branch is a generic fix for upstream's FIXME; keeping the overlap schedule
on under PP is safe only with the overlap-order PP loops, so upstream would take both together.

Targets, md5-pinned against upstream SGLang 43881a7: ``_pipeline_parallel_overlap_disable`` and
``check_pipeline_parallel_compat`` (wrapped on every launch, ``_LAUNCH_MD5``); on a pipelined decode only
(``_SCHEDULER_MD5``), ``Scheduler.run_batch`` (rebuilt and replaced), ``Scheduler._copy_result_to_cpu`` (wrapped), and
``TpModelWorker.forward_batch_generation`` and ``Scheduler._relay_forward_payload`` (relied on)."""

from __future__ import annotations

_OVERRIDES = "sglang.srt.arg_groups.overrides"
_VALIDATION = "sglang.srt.arg_groups.validation_hook"
_SCHEDULER = "sglang.srt.managers.scheduler"
_WORKER = "sglang.srt.managers.tp_worker"
DISABLE_PASS = f"{_OVERRIDES}._pipeline_parallel_overlap_disable"
PP_COMPAT = f"{_VALIDATION}.check_pipeline_parallel_compat"
RUN_BATCH = f"{_SCHEDULER}.Scheduler.run_batch"
COPY_RESULT = f"{_SCHEDULER}.Scheduler._copy_result_to_cpu"
# (module, qualified name): md5 of the source; the server-argument passes every launch wraps (register), then the
# scheduler's targets, checked only where a pipelined decode registers its hooks (register_scheduler)
_LAUNCH_MD5 = {
    (_OVERRIDES, "_pipeline_parallel_overlap_disable"): "0a84a05a59cb3ccfb036db66eaf20308",
    (_VALIDATION, "check_pipeline_parallel_compat"): "44d64b766a8d3b06d8c1ec8d67338432",
}
_SCHEDULER_MD5 = {
    (_SCHEDULER, "Scheduler.run_batch"): "afa792c782bfe5e3914d892209b15982",
    # takes the proxy tensors and hands them to the model runner
    (_WORKER, "TpModelWorker.forward_batch_generation"): "e88b173c3d3e22490210f611363e5a77",
    (_SCHEDULER, "Scheduler._copy_result_to_cpu"): "eebfbe3c17511c91a2f8e8271f16245d",
    # stashes the step's sampled tokens for the next step's input ids
    (_SCHEDULER, "Scheduler._relay_forward_payload"): "b3c72d82ab6958412e8ce2db456e9d67",
}
# run_batch's decorator line, which stage_run_batch applies again itself
RUN_BATCH_DECORATOR = "@scheduler_stage_method(SCHEDULER_STAGE_RUN_BATCH)\n"
# run_batch's overlap branch calls the worker without the stage's proxy ("FIXME: pp is not compatible with overlap")
RUN_BATCH_SUBSTITUTIONS = [("batch, **fwd_kwargs", "batch, pp_proxy_tensors=pp_proxy_tensors, **fwd_kwargs")]


def _pp_overlap_schedule() -> bool:
    """Whether the launch runs SGLang's overlap schedule under PP (the serving policy decides)."""
    from sglang_neuron import serving_policy

    policy = serving_policy.current()
    return policy.pp_size > 1 and policy.overlap_schedule


def pipeline_parallel_overlap_disable(original_fn, view):
    """AROUND _pipeline_parallel_overlap_disable: keep the overlap schedule of a PP overlap-order launch (without PP
    upstream's pass changes nothing)."""
    return {} if _pp_overlap_schedule() else original_fn(view)


class _ScheduleOff:
    """A resolved view whose overlap schedule reads off, for the PP checks other than the schedule's own."""

    def __init__(self, cfg):
        self._cfg = cfg

    def __getattr__(self, name):
        return True if name == "disable_overlap_schedule" else getattr(self._cfg, name)


def check_pipeline_parallel_compat(original_fn, cfg):
    """AROUND check_pipeline_parallel_compat: a PP overlap-order launch passes with its overlap schedule on, through
    every other check."""
    return original_fn(_ScheduleOff(cfg) if _pp_overlap_schedule() else cfg)


def stage_run_batch():
    """Scheduler.run_batch rebuilt from source with the overlap branch's worker call taking the stage's proxy tensors
    (RUN_BATCH_SUBSTITUTIONS), under upstream's own decorator (scheduler_stage_method only wraps the call in its
    stage's metrics and trace range)."""
    import importlib
    import inspect

    from sglang_neuron.sglang_patches import source_guard

    scheduler = importlib.import_module(_SCHEDULER)
    upstream = inspect.unwrap(source_guard.resolve(_SCHEDULER, "Scheduler.run_batch"))
    rebuilt = source_guard.with_substitutions(upstream, RUN_BATCH_SUBSTITUTIONS, decorators=(RUN_BATCH_DECORATOR,))
    return scheduler.scheduler_stage_method(scheduler.SCHEDULER_STAGE_RUN_BATCH)(rebuilt)


def copy_result_to_cpu(original_fn, self, batch_result, batch):
    """AROUND Scheduler._copy_result_to_cpu: a non-last stage copies nothing and only records its copy event."""
    if not batch_result.has_sampled_token_ids and batch_result.pp_hidden_states_proxy_tensors is not None:
        batch_result.copy_done.record()
        return None
    return original_fn(self, batch_result, batch)


def register(registry, around) -> None:
    """The server-argument hooks, registered on every launch since the launching process registers hooks before it
    resolves the serving policy. They run after handle_platform_defaults has resolved and exported it."""
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_LAUNCH_MD5, __name__)
    registry.register(DISABLE_PASS, pipeline_parallel_overlap_disable, around)
    registry.register(PP_COMPAT, check_pipeline_parallel_compat, around)


def register_scheduler(registry, around, replace) -> None:
    """The scheduler's hooks, registered only where the serving policy runs the overlap schedule under PP, so no other
    launch pays for them in its step."""
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_SCHEDULER_MD5, __name__)
    registry.register(RUN_BATCH, stage_run_batch(), replace)
    registry.register(COPY_RESULT, copy_result_to_cpu, around)
