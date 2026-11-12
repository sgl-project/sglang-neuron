# SPDX-License-Identifier: Apache-2.0
"""``K3DecodeRuntime``: the two compiled decode programs of one rank of the Kimi-K3 2-node DSpark decode.

One runtime per rank (LNC1, one NeuronCore). It binds the rank's weights and pools into the programs
(``programs.DecodeRankTensors``), runs the KDA-params init program once after the weight load, compiles the verify
and the draft programs with ``torch.compile(backend="neuron", fullgraph=True, dynamic=False)`` under the served
compiler arguments (``SERVED_COMPILER_ARGS`` through ``backend.stack.compile_scope``), warms them up with the inputs
of a padding step (``synthetic_inputs``), and runs a step as the verify program then the draft program (``step``). A
step submits both programs and returns their device outputs without a host read, so a caller can submit the next step
before it reads this one (async carry: ``StepOutputs.carry_inputs``).

Ordering across ranks: the draft program's collectives start only once every rank's draft execution reached them
(the runtime's device execution barrier, ``NEURON_RT_ENABLE_HW_EXECUTION_BARRIER=1``), which each core starts after
its verify execution; the host fences nothing between the two programs. ``RUNTIME_ENV`` holds the runtime
settings of a rank, those of the served decode.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import os
import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch
from torch import Tensor

from sglang_neuron.models.kimi_k3.decode import host_state as hs
from sglang_neuron.models.kimi_k3.decode import kernel_calls as kc
from sglang_neuron.models.kimi_k3.decode import programs as pg
from sglang_neuron.models.kimi_k3.decode import spec_math as sm
from sglang_neuron.models.kimi_k3.decode.capability import RUNTIME_ENV, SCRATCHPAD_PAGE_MB
from sglang_neuron.models.kimi_k3.decode.layout import DP8_LAYOUT, DecodeLayout
from sglang_neuron.models.kimi_k3.decode.pools import DecodePools
from sglang_neuron.models.kimi_k3.decode.programs import DecodeProgramConfig, StepOutputs, VerifyOutputs

logger = logging.getLogger(__name__)

SERVED_COMPILER_ARGS = (
    "--auto-cast=none",
    "--verbose=35",
    "-O2",
    "--internal-hlo2tensorizer-options=--modular-flow-mac-threshold=10 --partitioner-max-disj-runs=64",
    "--internal-backend-options=--enable-verifier=false --enable-nested-dynamic-loop --max-load-color-rotation=false "
    "--skip-pass=dma_optimization_sb",
    "--model-type=generic",
)
"""The neuronx-cc arguments of every NEFF of the served decode (verify, drafter, KDA init), in order. No
``--hbm-scratchpad-page-size``: the programs compile at neuronx-cc's default page, ``SCRATCHPAD_PAGE_MB``, which the
runtime must run with (``RUNTIME_ENV``, ``check_runtime_env``)."""

PAGE_SIZE_ARG = "--hbm-scratchpad-page-size"

_NO_EAGER_FALLBACK = {"fallback_execution": False}


@dataclasses.dataclass(frozen=True)
class CompileOptions:
    """How the programs compile.

    Attributes:
        backend: the dynamo backend (``neuron`` on a device; ``aot_eager`` traces on the host).
        compiler_args: the neuronx-cc arguments (``backend.stack.compile_scope``).
        backend_options: the backend's options (the neuron backend raises instead of an eager fallback).
        scoped: hold the compiler environment around the compiling calls (off on the host).
        kernels: the programs call the NKI kernels (their wrappers are built, the runtime settings checked); off,
            the caller routes the launches elsewhere (host tests).
    """

    backend: Any = "neuron"
    compiler_args: tuple[str, ...] = SERVED_COMPILER_ARGS
    backend_options: Mapping[str, Any] = dataclasses.field(default_factory=lambda: dict(_NO_EAGER_FALLBACK))
    scoped: bool = True
    kernels: bool = True

    @classmethod
    def host(cls, backend: Any = "aot_eager") -> CompileOptions:
        """Trace on the host (CPU tests): ``aot_eager`` (or ``backend``), no compiler environment, no kernels."""
        return cls(backend=backend, backend_options={}, scoped=False, kernels=False)

    @classmethod
    def for_device(cls, device: Any) -> CompileOptions:
        """The options of the programs on ``device``: the neuron backend on a Neuron device, the host trace on the CPU
        (the plugin's CPU mode)."""
        return cls.host() if torch.device(device).type == "cpu" else cls()

    def scratchpad_page_mb(self) -> int:
        """The scratchpad page the programs compile for: the arguments' ``--hbm-scratchpad-page-size``, else
        neuronx-cc's default."""
        pages = [arg.split("=", 1)[1] for arg in self.compiler_args if arg.startswith(PAGE_SIZE_ARG + "=")]
        return int(pages[-1]) if pages else SCRATCHPAD_PAGE_MB


def runtime_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """The runtime settings of a decode rank (``RUNTIME_ENV``)."""
    return dict(RUNTIME_ENV if base is None else base)


def check_runtime_env(environ: Mapping[str, str] | None = None, page_mb: int = SCRATCHPAD_PAGE_MB) -> None:
    """Refuse a process whose runtime settings break the program pair: the device barrier off (nothing orders the draft
    execution of a rank after its peers' verify executions), LNC2, or a scratchpad page other than the one the
    programs compile for (``page_mb``; unset is the runtime's default, ``SCRATCHPAD_PAGE_MB``)."""
    environ = os.environ if environ is None else environ
    page = environ.get("NEURON_SCRATCHPAD_PAGE_SIZE") or str(SCRATCHPAD_PAGE_MB)
    problems = [f"{name}={environ.get(name)!r} (want {want})"
                for name, want in (("NEURON_RT_ENABLE_HW_EXECUTION_BARRIER", "1"), ("NEURON_LOGICAL_NC_CONFIG", "1"))
                if environ.get(name) != want]
    if page != str(page_mb):
        problems.append(f"NEURON_SCRATCHPAD_PAGE_SIZE={page!r} (want {page_mb}, the page the programs compile for)")
    if problems:
        raise ValueError(f"The Kimi-K3 decode programs need {', '.join(problems)}: the device barrier is the only "
                         "cross-rank fence between the verify and the draft executions, a rank runs on one logical "
                         "core, and the runtime's scratchpad page is the compiler's.")


def draft_softmax_scale(drafter_dir: str | None = None) -> float:
    """The drafter's softmax scale (``q_head^-0.5`` x the YaRN mscale squared) from its ``config.json``, or from the
    published drafter's dimensions."""
    draft_torch = kc.kernel_imports.host_module("draft_mla_torch")
    if drafter_dir is None:
        return float(draft_torch.DraftMLADims().softmax_scale)
    from sglang_neuron.models.kimi_k3.decode.weights_draft import draft_dims, read_draft_config

    return float(draft_dims(read_draft_config(drafter_dir), draft_torch).softmax_scale)


def program_config(form: str, *, geometry: Any = None, drafter_dir: str | None = None,
                   layout: DecodeLayout = DP8_LAYOUT, softmax_scale: float | None = None) -> DecodeProgramConfig:
    """The program settings of a served form from the checkpoint geometry (``prefill.geometry.read_geometry``) and
    the drafter config; without them the published Kimi-K3 values."""
    kwargs = {}
    if geometry is not None:
        mla = set(int(layer) for layer in geometry.mla_layers)
        kwargs = {"vocab": int(geometry.vocab_size), "hidden": int(geometry.hidden_size),
                  "rms_norm_eps": float(geometry.rms_norm_eps),
                  "is_mla": tuple(layer in mla for layer in range(int(geometry.num_layers)))}
    scale = draft_softmax_scale(drafter_dir) if softmax_scale is None else float(softmax_scale)
    return DecodeProgramConfig(form=form, layout=layout, draft_softmax_scale=scale, **kwargs)


def _host(tensor: Tensor) -> Tensor:
    return tensor.detach().to("cpu")


def _plain_kda_width(width: int, conv_dtype: str) -> bool:
    try:
        kc.kda_params_local_heads(width, conv_dtype)
    except ValueError:
        return False
    return True


class K3DecodeRuntime:
    """The verify and draft programs of one decode rank.

    Args:
        layout: the 256-rank layout (``DP8_LAYOUT``).
        form: ``k3`` or ``k7b2``.
        rank: the global rank (``32 dp + tp``; node ``rank // 128``).
        weights: ``weights.load_decode_rank(..., device=device)``; ``kda_params`` plain (the init program widens it)
            or wide.
        pools: the rank's ``DecodePools`` on ``device``.
        compile_options: ``CompileOptions`` (``CompileOptions.host()`` on the host).
        device: the rank's device (``neuron:<local rank>``).
        tp_group: the attention TP process group of the embedding all-reduce (traced as a functional collective);
            None runs the identity (one-rank compile or host tests).
        config: the program settings (``program_config``); None takes the published Kimi-K3 values.
        cumsum: the residual draw's cumsum (the NKI kernel on the device; ``spec_math.cumsum_matmul`` on the host).
    """

    def __init__(self, *, layout: DecodeLayout = DP8_LAYOUT, form: str = "k3", rank: int, weights: Mapping[str, Tensor],
                 pools: DecodePools, compile_options: CompileOptions = CompileOptions(), device: Any = None,
                 tp_group: Any = None, config: DecodeProgramConfig | None = None,
                 cumsum: Callable[[Tensor], Tensor] | None = None) -> None:
        self.layout = layout
        self.form = form
        self.rank = int(rank)
        self.device = torch.device("cpu") if device is None else torch.device(device)
        self.compile_options = compile_options
        self.config = config or program_config(form, layout=layout)
        if self.config.form != form or self.config.layout != layout:
            raise ValueError(f"The program config is for {self.config.form} on {self.config.layout.name}; the runtime "
                             f"serves {form} on {layout.name}.")
        self.pools = pools
        self.tensors = pg.DecodeRankTensors(weights, pools, self.rank, layout, device=self.device)
        self.tp_group = tp_group
        on_device = compile_options.kernels
        if on_device:
            check_runtime_env(page_mb=compile_options.scratchpad_page_mb())
            pg.prepare_kernels()
        self.cumsum = cumsum or (pg.nki_cumsum if on_device else sm.cumsum_matmul)
        self.verify_program = pg.VerifyProgram(self.tensors, self.config, self._tp_all_reduce, self.cumsum)
        self.draft_program = pg.DraftProgram(self.tensors, self.config)
        self._verify = self._compile(self.verify_program)
        self._draft = self._compile(self.draft_program)
        self._warmed = False
        self.timings: dict[str, float] = {}
        # On a Neuron device the step's copies go through the plugin's device_io: uploads on the side stream (the graph
        # submit waits for them by one event), a launch event after each graph, and the readback of a finished step
        # read straight from its runtime tensors. torch-neuronx's plain copies use the graphs' queue, and a blocking
        # one waits for every queued graph, so the host could not plan step s + 1 while step s runs.
        self._device_io = self.device.type == "neuron"
        if self._device_io:
            from sglang_neuron.backend import device_io

            device_io.check_streams()
            device_io.install()

    # compile

    def _tp_all_reduce(self, tensor: Tensor) -> Tensor:
        if self.tp_group is None:
            return tensor
        from sglang_neuron.parallel.collectives import all_reduce_pg

        return all_reduce_pg(self.tp_group, tensor, "sum", self.layout.attention_tp)

    def _compile(self, module: torch.nn.Module) -> Callable:
        import torch._dynamo

        torch._dynamo.config.cache_size_limit = 2**62
        torch._dynamo.config.accumulated_cache_size_limit = 2**62
        options = dict(self.compile_options.backend_options) or None
        return torch.compile(module, backend=self.compile_options.backend, fullgraph=True, dynamic=False,
                             options=options)

    def compile_scope(self):
        """The compiler environment of the compiling calls (the served arguments)."""
        if not self.compile_options.scoped:
            return contextlib.nullcontext()
        from sglang_neuron.backend import stack

        return stack.compile_scope(list(self.compile_options.compiler_args))

    # the KDA params init

    def init_kda_params(self) -> bool:
        """Widen ``kda_params`` with ``exp(A_log)`` on the device (one init program); return False when the stack is
        wide already. Run after the weight load, before the first verify trace."""
        plain = self.tensors.w_kda_params
        conv = kc.VERIFY_CALL[self.form]["kda_conv_dtype"]
        width = int(plain.shape[-1])
        if not _plain_kda_width(width, conv):
            if any(_plain_kda_width(width - h, conv) and kc.kda_params_local_heads(width - h, conv) == h
                   for h in range(1, 9)):
                return False
            raise ValueError(f"The kda_params stack has {width} columns: neither a plain nor a wide {conv} width.")
        program = self._compile(pg.KdaParamsInit(conv))
        start = time.perf_counter()
        with self.compile_scope():
            wide = program(plain)
            wide_host_shape = tuple(wide.shape)
        if wide_host_shape != kc.kda_params_wide_shape(plain.shape, conv) or wide.dtype != torch.float32:
            raise RuntimeError(f"The kda_params init program returned {wide.dtype} {wide_host_shape}.")
        self.tensors.set_kda_params(wide)
        self.timings["kda_init_s"] = time.perf_counter() - start
        return True

    # inputs

    def host_state(self, *, synthetic: hs.SyntheticAcceptance | None = None, max_model_len: int = 1 << 20,
                   forced: bool = True) -> hs.DecodeSpecHostState:
        """A host state of one replica for this form (forced counts input on, as served; live rows on k3)."""
        return hs.DecodeSpecHostState(self.config.k, max_model_len, self.config.vocab, synthetic=synthetic,
                                      forced=forced, live_rows=self.config.live_rows)

    def synthetic_inputs(self) -> dict[str, Tensor]:
        """The step inputs of a padding step (warm-up, an idle replica's step): every row a pad row, null pages and
        slots, zero sampler rows, no rider ring. The same builders as a real
        step, so warm-up and serving trace one graph each."""
        cfg = self.config
        batch, queries = cfg.batch, cfg.queries
        state = self.host_state()
        plan = state.plan_step([None] * batch, np.zeros(batch, dtype=np.int64), ())
        inputs = dict(state.step_tensors(plan))
        inputs.update(hs.null_step_outputs(batch, queries, probabilistic=True))
        width = inputs["spec_position_window"].shape[1]
        inputs.update(
            input_ids=torch.zeros(batch * queries, dtype=torch.int32),
            sampling_params=torch.zeros((batch * queries, pg.SAMPLER_PARAM_COLUMNS), dtype=torch.float32),
            mla_block_table=torch.full((batch, cfg.geometry.table_width), -1, dtype=torch.int32),
            mla_slot_window=torch.zeros((batch, width), dtype=torch.int64),
            spec_draft_rider_ring=torch.zeros((batch, 14), dtype=torch.int32),
            **{f"kda_state_table_{g}": torch.full((batch, 1 + queries), -1, dtype=torch.int32)
               for g in range(pg.KDA_STATE_GROUPS)},
        )
        return self.to_device(inputs)

    def to_device(self, inputs: Mapping[str, Tensor], step: bool = False) -> dict[str, Tensor]:
        """The host inputs on the device for the next step this thread launches (a device tensor stays). ``step``:
        the inputs of the one next ``step`` call, written straight into the input ring (``step_io.StepInputRing``)
        with no stream operation; reuse such a dict for one step only."""
        if step and self._device_io:
            ring = getattr(self, "_ring", None)
            if ring is None:
                from sglang_neuron.models.kimi_k3.decode.step_io import StepInputRing

                ring = self._ring = StepInputRing(self.device, self.to_device)
            return ring.write(inputs)
        if not self._device_io:
            return {name: value.to(self.device) for name, value in inputs.items()}
        from sglang_neuron.backend import device_io

        names = [name for name, value in inputs.items() if value.device.type == "cpu"]
        out = dict(inputs)
        out.update(zip(names, device_io.upload_many([inputs[name] for name in names], self.device)))
        return out

    def _launch(self, fn: Callable, *args, pre: list | None = None, times: dict | None = None,
                tag: str = "") -> tuple[Any, Any]:
        """Run one graph call; return its outputs and its launch event (None off the device). ``pre`` (a probe):
        append a timed event right before the call, after the in-flight bound's wait. ``times`` (a probe): set
        ``<tag>_wait`` / ``<tag>_call`` (the bound's wait, the graph call) and their thread CPU times, in ms.

        The graph's outputs come from rotating buffers (``backend.output_buffers``, as many sets as the input ring has,
        ``step_io.RING_DEPTH``), so with the ring's inputs and the carried outputs each graph repeats one IO set every
        few steps and the runtime reuses its IO rings instead of building them inside every submit."""
        if not self._device_io:
            return fn(*args), None
        from sglang_neuron.backend import device_io, output_buffers
        from sglang_neuron.models.kimi_k3.decode import step_io

        t0, c0 = time.perf_counter(), time.thread_time()
        device_io.before_launch()
        t1, c1 = time.perf_counter(), time.thread_time()
        if pre is not None:
            pre.append(_timed_event())
        with output_buffers.rotating(step_io.RING_DEPTH):
            out = fn(*args)
        event = device_io.after_launch()
        if times is not None:
            t2, c2 = time.perf_counter(), time.thread_time()
            times.update({f"{tag}_wait": (t1 - t0) * 1e3, f"{tag}_wait_cpu": (c1 - c0) * 1e3,
                          f"{tag}_call": (t2 - t1) * 1e3, f"{tag}_call_cpu": (c2 - c1) * 1e3,
                          f"{tag}_released_at": t1})
        return out, event

    def input_ring_times(self) -> dict[str, float]:
        """The input ring's host times of its last write in ms (``StepInputRing.last_times``), or {} without one."""
        ring = getattr(self, "_ring", None)
        return {} if ring is None else dict(ring.last_times)

    # steps

    def warmup(self, read: bool = True) -> StepOutputs:
        """Widen ``kda_params`` if needed, then compile and run the verify and the draft programs once on a padding
        step; read the outputs back (the first executions finish under the compiler environment). ``read`` False
        skips the read (compile-only mode, whose outputs are placeholders)."""
        self.init_kda_params()
        inputs = self.synthetic_inputs()
        start = time.perf_counter()
        with self.compile_scope():
            outputs = self.step(inputs, check=True)
            if read:
                _host(outputs.next_ids)
        self.timings["warmup_s"] = time.perf_counter() - start
        self._warmed = True
        return outputs

    def verify(self, inputs: Mapping[str, Tensor]) -> VerifyOutputs:
        values = self._verify(inputs)
        if getattr(self.verify_program, "dump_logits", False):
            # the logits dump's last output (programs.dumped_logits); step() hands it on with the step's outputs
            *values, self.dumped_logits = values
        return VerifyOutputs.from_tuple(values)

    def draft(self, verify: VerifyOutputs, inputs: Mapping[str, Tensor]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return self._draft(verify.aux, verify.accepted, verify.bonus, verify.row0, verify.offset, verify.mla_slot,
                           verify.position_rows, verify.rank_info, verify.sample_params, verify.u_draft,
                           verify.row_flags, inputs["spec_draft_lo"], inputs["spec_rope_window"],
                           inputs["spec_draft_rider_ring"])

    def step(self, inputs: Mapping[str, Tensor], check: bool = False, marks: list | None = None,
             launch_marks: list | None = None, times: dict | None = None) -> StepOutputs:
        """Submit one step (verify, then draft) and return its device outputs; nothing is read back. ``marks`` (a
        probe): append a timed event after each graph's launch, so the caller can time the graphs as they run.
        ``launch_marks`` (a probe): append a timed event right before the verify's launch (after the in-flight
        bound's wait); it completes when the graphs ahead of it end or when the host records it, whichever is later.
        ``times`` (a probe): each launch's host wait and call times (``_launch``), tags ``v`` and ``d``, and the
        monotonic time (s) each wait returned (``<tag>_released_at``: comparable across the processes of a host)."""
        if check:
            pg.check_step_inputs(inputs, self.config)
        stance = (torch.compiler.set_stance("fail_on_recompile")
                  if self._warmed and self.compile_options.kernels else contextlib.nullcontext())
        with stance:
            verify, _ = self._launch(self.verify, inputs, pre=launch_marks, times=times, tag="v")
            if marks is not None and self._device_io:
                marks.append(_timed_event())
            (next_ids, flags, next_q_probs, next_q_ids), event = self._launch(self.draft, verify, inputs, times=times,
                                                                              tag="d")
            if marks is not None and self._device_io:
                marks.append(_timed_event())
        ring = getattr(self, "_ring", None)
        if ring is not None:
            ring.used(event)
        outputs = StepOutputs(tokens=verify.tokens, accepted=verify.accepted, next_ids=next_ids, row0=verify.row0,
                              flags=flags, next_q_probs=next_q_probs, next_q_ids=next_q_ids, launch_event=event)
        if getattr(self.verify_program, "dump_logits", False):
            # the verify's logits under the logits dump (models/kimi_k3/logits_dump.py), read after launch_event
            outputs.dumped_logits = self.dumped_logits
        return outputs

    def enable_logits_dump(self) -> None:
        """Make the verify program also return the logits it samples from (models/kimi_k3/logits_dump.py): a program of
        its own, so call it before the warm-up's first trace."""
        if self._warmed:
            raise RuntimeError("The logits dump changes the verify program; enable it before the warm-up.")
        self.verify_program.dump_logits = True

    READ = ("tokens", "accepted", "next_ids", "row0", "flags")

    @classmethod
    def read(cls, outputs: StepOutputs) -> dict[str, np.ndarray]:
        """Read a step's outputs to the host; waits for that step only (its launch event), not for a step launched
        after it."""
        values = [getattr(outputs, name) for name in cls.READ]
        if outputs.launch_event is None:
            return {name: _host(value).numpy() for name, value in zip(cls.READ, values)}
        from sglang_neuron.backend import device_io

        host = device_io.readback_many(values, outputs.launch_event)
        return {name: value.numpy() for name, value in zip(cls.READ, host)}


def _timed_event():
    """A timed event recorded on the current stream (a probe of ``K3DecodeRuntime.step``). At the end of the file:
    a line moved above a traced method (``_tp_all_reduce``) changes the programs' HLO source metadata."""
    from torch_neuronx.streams import Event

    event = Event(enable_timing=True)
    event.record()
    return event
