# SPDX-License-Identifier: Apache-2.0
"""Direct submit: a bound serving graph's steps handed straight to its torch-neuronx executor.

A megakernel decode program that declares it (``megakernel.PhaseProgram.direct_submit``; the GPT-OSS giga-kernel
decode is the first) would spend its steps' host time in layers that compute nothing for a one-shape graph; this module
hands them to the executor directly.

A compiled graph's call reaches its ``Executor`` through dynamo's frame (the guards, and the bytecode that fetches the
graph's inputs from the call's arguments and builds the output dict), AOTAutograd's runtime wrapper, torch-neuronx's
``fallback_wrapper`` and ``NeuronOutputCode.__call__``. None of these layers computes anything for such a decode
graph, which serves one warmed shape under tensor guards only: the executor gets the same list of inputs on every step
(the stacked weights, the KV pool tensors, the rank and scale tensors), of which only the step's own tensors
(token ids and their control, positions, block tables, KV slots) change. ``DirectSubmit`` builds that list
itself and calls the executor, whose own call (``Executor.__call__``: input mutations, preparation, output allocation,
``execute_compiled_graph``, writeback) runs as on the full path.

**Bind.** Two calls take the full path with an observation armed (the warm-up's call of the shape and the first serving
step). ``NeuronOutputCode.__call__``, wrapped by ``install``, records the list it hands the executor, the executor and
the tuple it returns. A call binds only where all of these hold, in both calls:

- the call ran exactly one NEFF execution, through the pinned ``fallback_wrapper`` without AutoCompile and with the
  eager fallback off (``fallback_execution`` False, so an execution error raises on both paths), or through
  AOTAutograd's own wrapper (a graph loaded from its cache);
- the output dict's values are the executor's returned tensors one to one, by identity: neither AOTAutograd nor dynamo
  adds an epilogue (a mutation's ``copy_``, a regenerated alias) or an output of its own;
- every executor input that is a tensor of the call's arguments is one by identity, at one position only, and is not
  model state (a parameter, buffer or tensor attribute of a module of the graph, or one of a group object's tensors);
  every other input is the same object in both calls;
- no module of the graph (its attributes, parameters, buffers or submodules) and no container of the call's arguments
  changed an entry during the call: a Python side effect there is one dynamo replays and a direct submit would skip.
  Writes deeper than that (an entry of a container a module holds, an attribute of an object it holds, a global) are
  not watched; the graph served this way makes none.

The executor positions whose input is an argument tensor are the step's slots, each read from the first argument path
that held it. Every other position keeps its object (the template). An argument container or tensor that several
paths held at the bind (one layer's metadata dict under every layer of its kind, one slot tensor in two dicts) must be
one object under all of them on every direct step, so a slot read from one path is the one dynamo would read from any.

**Shadow.** The first ``SHADOW_STEPS`` steps after the bind still take the full path: each builds the direct list too,
and the full path's list must be element-wise identical to it, and its output dict the executor's outputs at the bound
indices. A difference refuses the direct submit for good, with a warning that names the position. Only the designed
refusals turn the direct submit off; any other error in a bind or shadow proof is a bug in it and raises from that call.

**Direct steps.** A step reads each slot from the call's arguments, checks every alias and checks each slot tensor
against the bound one: type, dtype, shape, stride, device and requires_grad, the fields of dynamo's tensor guards that a
step's own tensors could change (their dispatch keys are not compared, since the scheduler makes every one in one grad
mode), and a tensor that is the object the last built step checked is not checked again. It then writes the slots into
the template, calls the executor and builds the output dict from its returned tuple. A step whose arguments differ from
the binding (a missing key, another shape, a broken alias) takes the full path, logged. The template is not checked per
step: the plugin refuses weight updates (``general_plugin``), a model binds its KV cache before its runtime exists, and
``executor_cache.invalidate`` (a KV cache bind) drops the binding, or a first bind call made before it, and the next two
full-path calls bind again.

**Several shapes.** A graph family of several warmed shapes (a ladder of rungs, decode batch buckets) is one compiled
graph per shape, each with its own executor. ``ShapeKeyedSubmit`` keeps one ``DirectSubmit`` a shape, keyed by the
shapes of the call's top-level tensor arguments, each bound, shadowed and refused on its own (the warm-up's call of the
shape is its first bind call, the shape's first serving step its second); at most one a warmed shape, and a call of
another key takes the full path. Two shapes that share a key would bind two executors, which the bind refuses.

**Refusals at construction**, each logged with its reason (the graph then keeps the full path): a torch build or
torch-neuronx source other than the pinned ones (``NeuronOutputCode.__call__``, ``neuron_backend_fx_compiler`` holding
``fallback_wrapper``, ``torch.version.git_version``), and a graph not compiled under tensor guards only
(``for_runtime``). The rank decides alone: both paths submit the same NEFF with the same inputs once per step.
"""

from __future__ import annotations

import functools
import logging
import operator
import sys
import threading
from typing import Callable, Optional

from sglang_neuron.backend import executor_cache, torch_pins

logger = logging.getLogger(__name__)

# the steps after a bind that still take the full path and must hand the executor the list a direct step builds
SHADOW_STEPS = 64
# md5 of the dedented source in torch_pins.TORCH_NEURONX_BUILD
_SOURCES = {
    # the layer the direct submit replaces: self._get_executor()(*inputs)
    "NeuronOutputCode.__call__": "648a3daec1332acc054d6e0a4bdaca9f",
    # defines fallback_wrapper, the boxed function AOTAutograd calls: its AutoCompile checks and the eager fallback
    "neuron_backend_fx_compiler": "1757e48221ab666f6919de4f1a622471",
}
_LOG_EVERY = 1000  # a step that takes the full path is logged at its first and every _LOG_EVERY-th occurrence

_BINDING, _SHADOW, _DIRECT, _REFUSED = "binding", "shadow", "direct", "refused"

# None: install() not run; "": installed; otherwise why it was not
_install_refusal: Optional[str] = None
# fallback_wrapper's code object (install), the caller an observed NeuronOutputCode call must have
_FALLBACK_CODE = None
# per thread: the observation an armed full-path call records into
_LOCAL = threading.local()


def wrap_output_code(cls) -> None:
    """Wrap ``cls.__call__`` (torch-neuronx's NeuronOutputCode) so a call on a thread with an armed observation records
    the inputs it hands the executor, the executor, the returned tuple and its caller's frame. Unarmed, it costs one
    thread-local read."""
    original = cls.__call__

    @functools.wraps(original)
    def __call__(self, inputs):
        observation = getattr(_LOCAL, "observation", None)
        if observation is None:
            return original(self, inputs)
        handed = tuple(inputs)
        outputs = original(self, inputs)
        observation.record(getattr(self, "_executor", None), handed, outputs, sys._getframe(1))
        return outputs

    __call__.__wrapped__ = original
    __call__._sgn_direct_submit = True
    cls.__call__ = __call__


def install() -> str:
    """Check the pins and wrap NeuronOutputCode.__call__ (once per process); "" when installed, otherwise the reason
    the direct submit cannot run on this build."""
    global _install_refusal, _FALLBACK_CODE
    if _install_refusal is not None:
        return _install_refusal
    import torch
    from torch_neuronx.neuron_dynamo_backend import backend, cache_artifact

    reasons = []
    # dynamo's frame and AOTAutograd's runtime wrapper, the layers the direct submit skips, are the torch build's
    if torch.version.git_version != torch_pins.TORCH_GIT_VERSION:
        reasons.append(f"torch {torch.__version__} ({torch.version.git_version}) is not the build it was proven on "
                       f"({torch_pins.TORCH_GIT_VERSION})")
    found = {"NeuronOutputCode.__call__": torch_pins.source_md5_or_empty(
                 cache_artifact.NeuronOutputCode.__dict__.get("__call__")),
             "neuron_backend_fx_compiler": torch_pins.source_md5_or_empty(backend.neuron_backend_fx_compiler)}
    changed = [name for name, want in _SOURCES.items() if found[name] != want]
    if changed:
        reasons.append(f"torch-neuronx's {', '.join(changed)} differ from {torch_pins.TORCH_NEURONX_BUILD}'s")
    codes = [c for c in backend.neuron_backend_fx_compiler.__code__.co_consts
             if getattr(c, "co_name", None) == "fallback_wrapper"]
    if len(codes) != 1:
        reasons.append(f"neuron_backend_fx_compiler defines {len(codes)} fallback_wrapper functions, not one")
    if reasons:
        _install_refusal = "; ".join(reasons)
        return _install_refusal
    _FALLBACK_CODE = codes[0]
    if not getattr(cache_artifact.NeuronOutputCode.__call__, "_sgn_direct_submit", False):
        wrap_output_code(cache_artifact.NeuronOutputCode)
    _install_refusal = ""
    return ""


def _caller_refusal(frame) -> str:
    """Why the caller of a NeuronOutputCode call may do work a direct submit skips, or "" for the two callers that do
    none: the pinned fallback_wrapper without AutoCompile and with the eager fallback off, and AOTAutograd's wrapper."""
    if frame.f_code is _FALLBACK_CODE:
        local = frame.f_locals
        if local.get("autocompile_attempt") is not None:
            return "the graph executes under torch-neuronx's AutoCompile"
        options = local.get("options")
        if not isinstance(options, dict) or options.get("fallback_execution") is not False:
            return "the graph runs eagerly where its NEFF execution fails (fallback_execution is not False)"
        return ""
    module = frame.f_globals.get("__name__", "")
    if module.startswith("torch._functorch."):
        return ""  # a graph AOTAutograd loaded from its cache: its wrapper calls the OutputCode itself
    return f"the NEFF's OutputCode is called from {module}.{frame.f_code.co_name}"


class _Observation:
    """What one full-path call handed the executor, armed on the calling thread."""

    def __init__(self, module, kwargs, side_effects: bool):
        self.calls = []
        self.kwargs = kwargs
        self.module = module
        # the bind checks that the call changed no module entry or argument container; a shadow step only compares lists
        self.before = _state_snapshot(module, kwargs) if side_effects else None

    def record(self, executor, inputs, outputs, caller) -> None:
        self.calls.append((executor, inputs, outputs, _caller_refusal(caller)))

    def __enter__(self):
        if getattr(_LOCAL, "observation", None) is not None:
            raise RuntimeError("direct submit: an observed graph call inside another")
        _LOCAL.observation = self
        return self

    def __exit__(self, *exc):
        _LOCAL.observation = None
        return False


def _containers(kwargs):
    """Every dict, list and tuple of the call's arguments in walk order, each once, with its first path."""
    seen, order = set(), []

    def visit(path, obj):
        if isinstance(obj, (dict, list, tuple)) and id(obj) not in seen:
            seen.add(id(obj))
            order.append((path, obj))
            for key, value in (obj.items() if isinstance(obj, dict) else enumerate(obj)):
                visit(path + (key,), value)

    visit((), kwargs)
    return order


def _state_snapshot(module, kwargs) -> list:
    """The identity of every entry of every module of ``module`` (its attributes, parameters, buffers and submodules)
    and of every container of the call's arguments: an entry dynamo's replayed side effects change differs after the
    call."""
    entries = []
    for m in module.modules():
        entries.append(tuple((k, id(v)) for k, v in vars(m).items()))
        for table in (m._parameters, m._buffers, m._modules):
            entries.append(tuple((k, id(v)) for k, v in table.items()))
    for _, obj in _containers(kwargs):
        items = obj.items() if isinstance(obj, dict) else enumerate(obj)
        entries.append(tuple((k, id(v)) for k, v in items))
    return entries


def _state_tensor_ids(module) -> set:
    """Ids of the model state a graph input may be: every parameter, buffer and tensor attribute of a module of
    ``module``, the tensors of its list, tuple and dict attributes (the KV pairs), and those of the plain objects it
    holds (a group's rank tensor), two levels deep."""
    import types

    import torch

    ids = set()

    def add(value, depth: int) -> None:
        if isinstance(value, torch.Tensor):
            ids.add(id(value))
        elif depth == 0 or isinstance(value, (torch.nn.Module, type, types.ModuleType)):
            return
        elif isinstance(value, (list, tuple)):
            for v in value:
                add(v, depth - 1)
        elif isinstance(value, dict):
            for v in value.values():
                add(v, depth - 1)
        elif hasattr(value, "__dict__"):
            for v in vars(value).values():
                add(v, depth - 1)

    for m in module.modules():
        for value in vars(m).values():
            add(value, 2)
    return ids


class _Refusal(Exception):
    pass


class _Facts:
    """One observed call's proofs: the executor, its inputs and outputs, the slots and aliases of the arguments."""

    def __init__(self, observation: _Observation, out, state_ids: set):
        import torch

        if observation.before is not None and _state_snapshot(observation.module, observation.kwargs) != \
                observation.before:
            raise _Refusal("the graph call changed a module attribute or an argument container (a Python side effect "
                           "dynamo replays)")
        if len(observation.calls) != 1:
            raise _Refusal(f"the graph call ran {len(observation.calls)} NEFF executions through NeuronOutputCode, "
                           "not one")
        self.executor, self.inputs, outputs, caller = observation.calls[0]
        if caller:
            raise _Refusal(caller)
        # the pinned NeuronOutputCode.__call__ always records its executor; this guards a change to it
        if self.executor is None:
            raise _Refusal("the NeuronOutputCode call left no executor")
        if not isinstance(out, dict) or not out:
            raise _Refusal(f"the graph returns a {type(out).__name__}, not a dict of tensors")
        outputs = tuple(outputs)
        index = []
        for key, value in out.items():
            hits = [i for i, t in enumerate(outputs) if t is value]
            if len(hits) != 1:
                raise _Refusal(f"output {key!r} is {'not' if not hits else 'more than once'} one of the executor's "
                               "returned tensors")
            index.append((key, hits[0]))
        if len(outputs) != len(out) or len({i for _, i in index}) != len(index):
            raise _Refusal(f"the executor returns {len(outputs)} tensors for the graph's {len(out)} outputs")
        self.outputs = tuple(index)
        # the arguments' tensors by identity, each with every path that holds it (the first through first-seen
        # containers only), and every container a later path holds again
        self.leaf_paths, self.container_aliases, canonical = {}, [], {}

        def visit(path, obj):
            if isinstance(obj, torch.Tensor):
                self.leaf_paths.setdefault(id(obj), []).append(path)
            elif isinstance(obj, (dict, list, tuple)):
                first = canonical.get(id(obj))
                if first is not None:
                    self.container_aliases.append((path, first))
                    return
                canonical[id(obj)] = path
                for key, value in (obj.items() if isinstance(obj, dict) else enumerate(obj)):
                    visit(path + (key,), value)

        visit((), observation.kwargs)
        self.slots = {}  # executor position -> the paths of its argument tensor
        for i, inp in enumerate(self.inputs):
            paths = self.leaf_paths.get(id(inp))
            if paths is None:
                continue
            if id(inp) in state_ids:
                raise _Refusal(f"argument {_name(paths[0])} is model state (executor input {i})")
            self.slots[i] = paths
        seen = {}
        for i, paths in self.slots.items():
            if paths[0] in seen:
                raise _Refusal(f"argument {_name(paths[0])} is executor inputs {seen[paths[0]]} and {i}")
            seen[paths[0]] = i


def _name(path) -> str:
    return "kwargs" + "".join(f"[{k!r}]" for k in path)


def _tuple_getter(indices: list) -> Callable:
    """``seq -> tuple(seq[i] for i in indices)``, through operator.itemgetter (which returns a bare item for one
    index)."""
    if len(indices) == 1:
        index = indices[0]
        return lambda seq: (seq[index],)
    return operator.itemgetter(*indices)


class DirectSubmit:
    """The direct submit of one runtime's graph family (see the module docstring). ``plan(kwargs)`` gives a direct
    step's executor inputs (None: the step takes the full path), ``submit(flat)`` launches them, and ``full(compiled,
    kwargs)`` runs the full path, observed while the binding is proven or shadowed."""

    def __init__(self, module, shadow: int, name: str = "graph"):
        self._module = module
        self._shadow_steps = shadow
        self._name = name
        self._generation = executor_cache._generation
        self._state = _BINDING
        self._first: Optional[_Facts] = None
        self._first_generation = None  # executor_cache's generation at the first bind call
        self._full_steps = 0
        self._state_ids: Optional[set] = None
        # set by _bind
        self._flat: Optional[list] = None
        self._bound_generation = None

    @property
    def state(self) -> str:
        return self._state

    # binding
    def _refuse(self, reason: str) -> None:
        self._state = _REFUSED
        self._first = None
        self._flat = None
        logger.warning("[neuron] direct submit of the %s refused: %s; its steps take the full call path", self._name,
                       reason)

    def _rebind_if_invalidated(self) -> bool:
        if self._state in (_SHADOW, _DIRECT) and self._bound_generation != self._generation[0]:
            logger.info("[neuron] direct submit of the %s: the executor caches were invalidated; binding again",
                        self._name)
            self._state, self._first, self._flat = _BINDING, None, None
            return True
        return False

    def full(self, compiled: Callable, kwargs: dict):
        """The full path (``compiled(**kwargs)``): observed while binding (two calls) and in the shadow steps."""
        self._rebind_if_invalidated()
        state = self._state
        if state is _DIRECT or state is _REFUSED:
            return compiled(**kwargs)
        expected = self._build(kwargs) if state is _SHADOW else None
        if state is _SHADOW and expected is None:
            return compiled(**kwargs)  # arguments the binding does not cover: nothing to compare
        observation = _Observation(self._module, kwargs, side_effects=state is _BINDING)
        with observation:
            out = compiled(**kwargs)
        # a designed refusal keeps the full path for good; any other error is a bug in the proof and raises from the
        # call that hit it: a bind call (the warm-up's, the first serving step's, or one of the two after a KV bind) or
        # a shadow step, whose output is then lost
        try:
            if state is _BINDING:
                self._observed(observation, out)
            else:
                self._shadowed(observation, expected, out)
        except _Refusal as refusal:
            self._refuse(str(refusal))
        return out

    def _observed(self, observation: _Observation, out) -> None:
        generation = self._generation[0]
        if self._first is not None and self._first_generation != generation:
            # an invalidate() since the first bind call (a KV cache bind) may have replaced the objects its template
            # holds: this call is the first of a new pair
            self._first = None
        if self._first is None:
            # the model state as of this pair (an invalidate() may have bound other KV tensors since the last one)
            self._state_ids = _state_tensor_ids(self._module)
            self._first, self._first_generation = _Facts(observation, out, self._state_ids), generation
            return
        first, self._first = self._first, None
        self._bind(first, _Facts(observation, out, self._state_ids))

    def _bind(self, a: _Facts, b: _Facts) -> None:
        if a.executor is not b.executor:
            raise _Refusal("the two bind calls ran different executors")
        if len(a.inputs) != len(b.inputs):
            raise _Refusal(f"the two bind calls handed {len(a.inputs)} and {len(b.inputs)} executor inputs")
        if a.outputs != b.outputs:
            raise _Refusal("the two bind calls returned different outputs")
        canon = {i: paths[0] for i, paths in b.slots.items()}
        if {i: paths[0] for i, paths in a.slots.items()} != canon:
            raise _Refusal("the two bind calls read the step's tensors from different arguments or positions")
        for i, (x, y) in enumerate(zip(a.inputs, b.inputs)):
            if i not in canon and x is not y:
                raise _Refusal(f"executor input {i} ({type(y).__name__} {tuple(getattr(y, 'shape', ()))}) is no "
                               "argument tensor and changed between the two bind calls")
        # the aliases a slot's object is read through: containers above a slot, and a slot tensor's other paths
        slot_paths = set(canon.values())
        aliases, a_aliases = (
            [(p, c) for p, c in facts.container_aliases if any(s[:len(c)] == c for s in slot_paths)]
            + [(p, paths[0]) for _, paths in sorted(facts.slots.items()) for p in paths[1:]] for facts in (b, a))
        if aliases != a_aliases:
            raise _Refusal("the two bind calls share their argument containers or tensors differently")
        # the fetch program: one register per path prefix, the children of one container in consecutive registers,
        # each container's children fetched by one getter (in C), parents before children
        children = {}
        for path in [*(path for _, path in sorted(canon.items())), *(path for pair in aliases for path in pair)]:
            for n in range(1, len(path) + 1):
                keys = children.setdefault(path[:n - 1], [])
                if path[n - 1] not in keys:
                    keys.append(path[n - 1])
        registers, fetch, queue = {(): 0}, [], [()]
        while queue:
            parent = queue.pop(0)
            keys = children.get(parent)
            if not keys:
                continue
            lo = len(registers)
            for key in keys:
                registers[parent + (key,)] = len(registers)
                queue.append(parent + (key,))
            fetch.append((lo, lo + len(keys), registers[parent], operator.itemgetter(*keys) if len(keys) > 1 else None,
                          keys[0]))
        positions = sorted(canon)
        tensors = [b.inputs[i] for i in positions]
        self._fetch = tuple(fetch)
        self._registers = len(registers)
        self._positions = tuple(positions)
        self._slot_paths = tuple((i, canon[i]) for i in positions)
        self._slot_get = _tuple_getter([registers[canon[i]] for i in positions])
        self._slot_meta = tuple((type(t), t.dtype, t.shape, t.stride(), t.device, t.requires_grad) for t in tensors)
        self._checked = tuple(tensors)
        self._alias_get = (_tuple_getter([registers[p] for p, _ in aliases]),
                           _tuple_getter([registers[c] for _, c in aliases])) if aliases else None
        # the executor's inputs: the template, with the slots of the last step built (the next one overwrites them)
        self._flat = list(b.inputs)
        self._executor = b.executor
        self._outputs = b.outputs
        self._bound_generation = self._generation[0]
        self._state = _SHADOW if self._shadow_steps > 0 else _DIRECT
        self._shadow_left = self._shadow_steps
        logger.info("[neuron] direct submit of the %s bound: %d executor inputs, %d read per step from %d argument "
                    "paths (%d aliases checked), %d outputs; %s", self._name, len(b.inputs), len(positions),
                    len(registers) - 1, len(aliases), len(b.outputs),
                    f"the next {self._shadow_steps} steps take the full path and compare" if self._shadow_steps
                    else "no shadow steps")

    def _shadowed(self, observation: _Observation, expected: list, out) -> None:
        if len(observation.calls) != 1:
            raise _Refusal(f"a shadow step ran {len(observation.calls)} NEFF executions, not one")
        executor, inputs, outputs, caller = observation.calls[0]
        if caller:
            raise _Refusal(caller)
        if executor is not self._executor:
            raise _Refusal("a shadow step ran another executor")
        if len(inputs) != len(expected):
            raise _Refusal(f"a shadow step handed {len(inputs)} executor inputs, the binding {len(expected)}")
        for i, (x, y) in enumerate(zip(inputs, expected)):
            if x is not y:
                where = next((_name(path) for pos, path in self._slot_paths if pos == i), "the template")
                raise _Refusal(f"a shadow step handed executor input {i} ({where}) another object than the binding")
        if not isinstance(out, dict) or len(out) != len(self._outputs) or len(outputs) != len(self._outputs) or any(
                key not in out or out[key] is not outputs[i] for key, i in self._outputs):
            raise _Refusal("a shadow step's outputs are not the executor's at the bound indices")
        self._shadow_left -= 1
        if self._shadow_left <= 0:
            self._state = _DIRECT
            logger.info("[neuron] direct submit of the %s on: %d steps matched the full call path", self._name,
                        self._shadow_steps)

    # direct steps
    def _build(self, kwargs: dict) -> Optional[list]:
        """The executor inputs of this step's arguments, or None (logged) where they differ from the binding."""
        regs = [None] * self._registers
        regs[0] = kwargs
        try:
            for lo, hi, src, getter, key in self._fetch:
                if getter is None:
                    regs[lo] = regs[src][key]
                else:
                    regs[lo:hi] = getter(regs[src])
        except (KeyError, IndexError, TypeError):
            return self._differs("a bound argument is missing")
        if self._alias_get is not None and any(map(operator.is_not, self._alias_get[0](regs),
                                                   self._alias_get[1](regs))):
            return self._differs("the arguments no longer share a container or tensor the binding shared")
        tensors = self._slot_get(regs)
        checked = self._checked
        for n, t in enumerate(tensors):
            # what dynamo's tensor guards check. A tensor the last built step checked is skipped: only an in-place
            # metadata op (set_, resize_, an in-place view such as unsqueeze_, requires_grad_) changes an object's
            # metadata, and the plugin applies none, as the executor binding cache relies on too
            if t is checked[n]:
                continue
            kind, dtype, shape, stride, device, requires_grad = self._slot_meta[n]
            if (type(t) is not kind or t.dtype is not dtype or t.shape != shape or t.stride() != stride
                    or t.device != device or t.requires_grad is not requires_grad):
                return self._differs(f"executor input {self._positions[n]} ({_name(self._slot_paths[n][1])}) is not "
                                     "a tensor of the bound type, dtype, shape, stride, device and requires_grad")
        self._checked = tensors
        flat = self._flat
        for pos, t in zip(self._positions, tensors):
            flat[pos] = t
        return flat

    def _differs(self, reason: str) -> None:
        self._full_steps += 1
        if self._full_steps == 1 or self._full_steps % _LOG_EVERY == 0:
            logger.warning("[neuron] direct submit of the %s: %s, the step takes the full call path (%d so far)",
                           self._name, reason, self._full_steps)
        return None

    def plan(self, kwargs: dict) -> Optional[list]:
        """A direct step's executor inputs, or None: the step takes the full path (binding, shadowing, refused, or
        arguments the binding does not cover)."""
        if self._state is not _DIRECT or self._rebind_if_invalidated():
            return None
        return self._build(kwargs)

    def submit(self, flat: list) -> dict:
        """Launch a direct step (``plan``'s inputs) and return its output dict."""
        outputs = self._executor(*flat)
        return {key: outputs[i] for key, i in self._outputs}


def shape_key(kwargs: dict) -> tuple:
    """A call's shape key: the shape of each top-level tensor argument, in argument order."""
    import torch

    return tuple(tuple(v.shape) if isinstance(v, torch.Tensor) else None for v in kwargs.values())


class ShapeKeyedSubmit:
    """The direct submit of a graph family of several warmed shapes (the module docstring): one ``DirectSubmit`` a
    shape key, at most ``shapes`` of them. ``plan`` gives (the shape's direct submit, its executor inputs) or None."""

    def __init__(self, module, shadow: int, shapes: int, name: str = "graph", key: Callable = shape_key):
        self._module, self._shadow, self._shapes, self._name, self._key = module, shadow, shapes, name, key
        self._by_key: dict = {}
        self._other = 0

    def _submit_of(self, kwargs: dict) -> Optional[DirectSubmit]:
        key = self._key(kwargs)
        submit = self._by_key.get(key)
        if submit is None:
            if len(self._by_key) >= self._shapes:
                self._other += 1
                if self._other == 1 or self._other % _LOG_EVERY == 0:
                    logger.warning("[neuron] direct submit of the %s: a call of shape %s beyond its %d warmed shapes "
                                   "takes the full call path (%d so far)", self._name, key, self._shapes, self._other)
                return None
            submit = self._by_key[key] = DirectSubmit(self._module, self._shadow, name=f"{self._name} {key}")
        return submit

    @property
    def state(self) -> str:
        """The shapes' states (one state where they agree): binding, shadow, direct or refused."""
        states = {s.state for s in self._by_key.values()}
        return states.pop() if len(states) == 1 else ",".join(sorted(states)) or _BINDING

    @property
    def submits(self) -> dict:
        return dict(self._by_key)

    def full(self, compiled: Callable, kwargs: dict):
        submit = self._submit_of(kwargs)
        return compiled(**kwargs) if submit is None else submit.full(compiled, kwargs)

    def plan(self, kwargs: dict):
        submit = self._by_key.get(self._key(kwargs))
        if submit is None:
            return None
        flat = submit.plan(kwargs)
        return None if flat is None else (submit, flat)

    def submit(self, planned) -> dict:
        submit, flat = planned
        return submit.submit(flat)


def for_runtime(module, *, guard_filter_fn: Optional[Callable], shapes: int, name: str):
    """The direct submit of a runtime's graph family (a ``DirectSubmit`` for one warmed shape, a ``ShapeKeyedSubmit``
    for several), or None (logged) where it is refused before any call: a graph compiled with more than tensor guards
    (``guard_filter_fn``, None for every guard), or an unpinned build."""
    import torch

    tensor_guards = guard_filter_fn is torch.compiler.keep_tensor_guards_unsafe
    if not tensor_guards:
        reason = (f"the graph serves {shapes} warmed shape{'' if shapes == 1 else 's'} under more than tensor guards, "
                  "and a direct submit takes shapes under tensor guards only, whose non-tensor arguments are constants")
    else:
        reason = install()
    if reason:
        logger.warning("[neuron] direct submit of the %s refused: %s; its steps take the full call path", name,
                       reason)
        return None
    if shapes == 1:
        return DirectSubmit(module, SHADOW_STEPS, name=name)
    return ShapeKeyedSubmit(module, SHADOW_STEPS, shapes, name=name)
