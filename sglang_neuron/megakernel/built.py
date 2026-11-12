# SPDX-License-Identifier: Apache-2.0
"""The Neuron ``BuiltMegakernel``: SGLang's plan / forward contract over the platform's compiled-graph runtimes.

A spec's ``build`` returns a ``NeuronBuiltMegakernel`` declaring:

- its ``programs``: one ``PhaseProgram`` per step kind it covers (``decode``, ``prefill``, ``idle``), each naming the
  model graph family that runs it (a key of the model's ``model_graphs``), the ``NeuronModelRuntime`` subclass that
  builds its inputs and whether its steps go to the bound executor (``backend.direct_submit``);
- its ``pool_layout`` (``pools.PoolLayout``): the SGLang pool tensors the programs take and alias back;
- its ``widths``: the rows per step it serves (the decode seqs buckets it was built for).

Execution stays the platform's: each family is a ``torch.compile`` graph on the ``neuron`` backend whose body calls the
whole-model (or whole-stage) NKI kernel once through ``wrap_nki``, with the stacked ``[L, ...]`` weights and the pool
tensors as inputs and the kernel's in-place-written pools as aliased outputs. ``prepare`` binds the layout to the live
pools and builds, compiles and warms each family's runtime (``NeuronModel._build_runtimes``: the per-rank NEFF
agreement, the model's post-warm-up checks and the direct submit's bind all run there). ``plan`` reads host state only
(the attention metadata and the runtime's ``plan_step``, which agree on every DP rank) and returns a ``NeuronStep``, or
None for a step no program covers; ``forward`` runs the planned step on its family's runtime. There is no eager
path: ``original`` (the model's own forward) is never called, and the dispatch refuses a step ``plan`` does not cover.
"""
from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from . import upstream
from .pools import PoolLayout

logger = logging.getLogger(__name__)

MODES = ("decode", "prefill", "idle")


@dataclass(frozen=True)
class PhaseProgram:
    mode: str  # the step kind it covers: decode, prefill or idle
    family: str  # the model graph family that runs it: a key of the model's model_graphs ("decode", "prefill")
    runtime_class: Optional[type] = None  # a NeuronModelRuntime subclass; None: the platform runtime
    direct_submit: bool = False  # its steps go to the bound executor once two full calls proved the binding
    samples_in_forward: bool = False  # its graph returns sampled tokens (a fused sampling tail), not logits

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"phase program mode {self.mode!r} is not one of {list(MODES)}")
        if self.family not in ("decode", "prefill"):
            raise ValueError(f"phase program family {self.family!r}: the graph families are decode and prefill")
        if self.mode in ("decode", "prefill") and self.mode != self.family:
            raise ValueError(f"a {self.mode} program runs on the {self.mode} family, not {self.family}")
        if self.direct_submit and self.family != "decode":
            raise ValueError("direct submit serves a decode family only (its warmed shapes under tensor guards)")

    def digest(self) -> tuple:
        cls = self.runtime_class
        return (self.mode, self.family, None if cls is None else f"{cls.__module__}.{cls.__qualname__}",
                self.direct_submit, self.samples_in_forward)


@functools.lru_cache(maxsize=None)
def step_class() -> type:
    """``NeuronStep``: SGLang's ``Step`` (mode, rows, view) with what the runtime needs to run it: the graph family
    (``phase``), the runtime's step plan (bucket, context bucket) and the step's attention metadata. Built from the
    served tree's ``Step`` on first use."""
    import msgspec  # noqa: F401 -- the contract's struct library

    Step = upstream.require().Step

    class NeuronStep(Step, frozen=True):
        phase: str = ""
        plan: Any = None
        metadata: Any = None

    NeuronStep.__module__ = __name__
    return NeuronStep


@dataclass
class NeuronForwardView:
    """``Step.view`` of a Neuron step: what the step's forward may read besides the step itself."""

    metadata: Any
    pools: Any  # pools.BoundPools, or None without a layout
    forward_batch: Any


def _no_eager_path(*args, **kwargs):
    raise RuntimeError("a Neuron megakernel step has no eager path: the model's own forward is never called")


class NeuronBuiltMegakernel:
    """A spec's built object on Neuron (the module docstring). Subclass it only to add model-specific checks; the plan
    and forward are generic."""

    draft_sampler = None
    draft_loop = None
    draft_extend = None
    begin_draft_round = None
    graph_variants = None

    def __init__(self, *, name: str, model, programs: Sequence[PhaseProgram], pool_layout: Optional[PoolLayout],
                 widths: Sequence[int]):
        self.name = name
        self.model = model
        self.programs = {p.mode: p for p in programs}
        if len(self.programs) != len(programs) or not programs:
            raise ValueError(f"{name}: one program per step kind, at least one: {[p.mode for p in programs]}")
        graphs = getattr(model, "model_graphs", None) or {}
        missing = sorted({p.family for p in programs} - set(graphs))
        if missing:
            raise ValueError(f"{name}: programs run on the {missing} graph families, which the model did not build "
                             f"(it has {sorted(graphs)})")
        classes = {}
        for p in programs:
            if classes.setdefault(p.family, p.runtime_class) is not p.runtime_class:
                raise ValueError(f"{name}: the {p.family} family's programs name different runtime classes")
        if "idle" in self.programs and self.programs["idle"].family not in {p.family for p in programs
                                                                             if p.mode != "idle"}:
            raise ValueError(f"{name}: the idle program runs the {self.programs['idle'].family} family, which no "
                             "covered step kind runs")
        if pool_layout is not None and not isinstance(pool_layout, PoolLayout):
            raise TypeError(f"{name}: a Neuron megakernel's pool_layout is a sglang_neuron.megakernel.PoolLayout")
        self.pool_layout = pool_layout
        self.widths = tuple(sorted(int(w) for w in widths))
        self.covered_modes = frozenset(self.programs)
        self._runtimes: dict = {}
        self._pools = None
        self._runner = None
        self._attn = None

    # the contract
    def summary(self) -> str:
        layout = self.pool_layout.family if self.pool_layout is not None else "SGLang's pools"
        programs = ", ".join(f"{m}: {p.family}" + (" (direct submit)" if p.direct_submit else "")
                             for m, p in sorted(self.programs.items()))
        return f"{self.name}: {programs}; pools {layout}"

    def digest(self) -> tuple:
        """What every rank must hold the same (rank agreement): programs, widths and the pool layout."""
        return (self.name, tuple(p.digest() for _, p in sorted(self.programs.items())), self.widths,
                None if self.pool_layout is None else self.pool_layout.digest())

    def prepare(self, model_runner) -> dict:
        """After SGLang allocated the pools: bind the layout, then build, compile and warm every family's runtime.
        Returns the runtimes by family."""
        if self._runner is not None:
            raise RuntimeError(f"{self.name} is already prepared")
        if model_runner.model is not self.model:
            raise RuntimeError(f"{self.name} was built for another model than the runner's")
        if self.pool_layout is not None:
            self._pools = self.pool_layout.bind(model_runner.token_to_kv_pool, model_runner.req_to_token_pool)
            if self.pool_layout.zero_on_free and not getattr(model_runner.token_to_kv_pool_allocator,
                                                             "zeroes_freed_pages", False):
                raise ValueError(f"{self.name}: the kernel reads whole pages ({self.pool_layout.family} zero_on_free), "
                                 "and the KV allocator does not zero the pages it frees")
        families = {}
        for p in self.programs.values():
            families.setdefault(p.family, p)
        direct = {p.family for p in self.programs.values() if p.direct_submit}
        self._runtimes = self.model._build_runtimes(
            model_runner, runtime_classes={f: p.runtime_class for f, p in families.items()}, direct_submit=direct)
        decode = self._runtimes.get("decode")
        if decode is not None:
            served = tuple(decode.buckets.num_seqs_buckets)
            beyond = [b for b in served if b not in self.widths]
            if beyond:
                raise ValueError(f"{self.name}: the decode family serves the seqs buckets {list(served)}, and "
                                 f"{beyond} are not among the widths it was built for {list(self.widths)}; launch "
                                 "with seqs buckets among them")
        self._runner = model_runner
        # a hybrid model's backend is SGLang's HybridLinearAttnBackend over the platform's: the step metadata is the
        # full-attention backend's (the linear layers' state rides in the megakernel's own pools)
        attn = model_runner.attn_backend
        self._attn = getattr(attn, "full_attn_backend", attn)
        return self._runtimes

    def plan(self, input_ids, forward_batch, *args, **kwargs):
        if args or set(kwargs) - {"pp_proxy_tensors"}:
            raise ValueError(f"{self.name} takes no forward arguments beyond pp_proxy_tensors; got "
                             f"{len(args)} positional and {sorted(kwargs)}")
        if self._runner is None:
            raise RuntimeError(f"{self.name} plans a step before it was prepared")
        md = self._attn.build_metadata(forward_batch)
        planner = self._runtimes[next(iter(self.programs.values())).family]
        plan = planner.plan_step(forward_batch, md)
        mode = "idle" if plan.kind == "idle" or md.batch_size == 0 else plan.kind
        program = self.programs.get(mode)
        if program is None or (mode == "idle" and plan.phase != program.family):
            return None
        return step_class()(mode=mode, rows=int(plan.bucket), phase=plan.phase, plan=plan, metadata=md)

    def forward(self, step, original, input_ids, positions, forward_batch, *args, **kwargs):
        runtime = self._runtimes[step.phase]
        if self._pools is not None:
            self._pools.check(self._runner.token_to_kv_pool, self._runner.req_to_token_pool)
        if kwargs.get("pp_proxy_tensors") is not None:
            raise ValueError(f"{self.name}: pp_proxy_tensors arrived, and a whole-model kernel runs no pipeline stage "
                             "(distributed.pipeline refuses --enable-nvidia-megakernel under --pp-size)")
        if step.mode == "idle":
            # the pending decode output under async idle completion (SGLang's result queue owns its completion), else an
            # empty output
            return runtime.run_idle(step.plan)
        if step.mode == "decode":
            return runtime.run_decode(forward_batch, step.metadata, step.plan.bucket, step.plan.context)
        return runtime.run_prefill(forward_batch, step.metadata, step.plan.bucket)

    # the dispatch's view
    @property
    def pools(self):
        return self._pools

    def view(self, step, forward_batch) -> NeuronForwardView:
        return NeuronForwardView(metadata=step.metadata, pools=self._pools, forward_batch=forward_batch)

    @property
    def runtimes(self) -> dict:
        return dict(self._runtimes)

    no_eager_path = staticmethod(_no_eager_path)
