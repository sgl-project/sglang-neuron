# SPDX-License-Identifier: Apache-2.0
"""A per-input binding cache for torch-neuronx's compiled-graph executor.

torch-neuronx's ``Executor`` redoes all of its input handling in Python on every call: it prepares each input
(``_prepare_inputs``: copies scalars and CPU tensors to the device, makes inputs contiguous, views 64-bit integers as
32 bits) and validates every mutated input's copy-back target before and after the run (``_validate_copy_back_targets``,
``_writeback_shadowed_aliases``). A serving graph has hundreds of inputs (weights, KV pools, packed tables) and a step
changes only a few, so nearly all of that work gives the same answer every time.

What the cache does:

* Per executor and input position it keeps the last input object and its prepared tensor, when preparation returned
  the input itself or a view of it (no copy). The same object next time reuses the prepared tensor. Anything else goes
  through torch-neuronx's own per-input preparation (``_prepare_one``), and inputs that get copied are prepared on
  every call.
* An executor with an input that torch-neuronx's weight-layout optimization relayouts is left to torch-neuronx's
  ``_prepare_inputs`` on every call, since the relayout isn't cached here. The serving graphs don't enable it.
* The copy-back checks are skipped when every copy-back input and output is the same object the previous call checked,
  since an object's storage, shape and dtype don't change between calls. Any change runs torch-neuronx's checks. The
  aliased-output writeback is skipped for those objects only after a call that found every result already in place.
* After ``arm_writeback``, this thread's next call records its copy-back targets and whether the NEFF wrote each in
  place (``take_writeback``). A PP stage uses it to prove its forward wrote the boundary into its seam slot.
* If an executor's first call finds no tensor dump policy (TORCH_NEURONX_DYNAMO_TENSOR_DUMPS), it drops its tensor
  dumper, which would otherwise reread the environment twice per call to dump nothing. The same call reads
  TORCH_NEURONX_DYNAMO_DISABLE_CPU_AUTOCOPY once for the executor's per-input preparations, which torch-neuronx's own
  preparation rereads on every call.

The cache keys on object identity, and holds the previous call's inputs, so an input's id cannot be reused while it is
cached; it keeps the previous step's per-step inputs alive one extra step. A cached input whose storage were replaced
under the same object (``set_``, ``resize_``, ``param.data = other``) would be served its old prepared view: the plugin
never does that to a graph input, and a KV cache bind calls ``invalidate``, so the next call prepares every input
again.

The wrapped methods' sources, and those of the methods whose protocol the cache relies on (``Executor.__call__``: the
order of the three, the tensor dumper and the copy-back map; ``_allocate_output``: an aliased output is its input's
prepared tensor; ``_process_outputs``: an output is returned as allocated unless its dtype is restored), are checked
against the torch-neuronx build this was written for (torch_pins.TORCH_NEURONX_BUILD): on another build the cache
refuses to install, naming the method, rather than skip checks that build may have changed.

Once an executor has prepared a call's inputs, it hands its NEFF cache key to this thread's next submit
(``submitting``): ``Executor.__call__`` then allocates the outputs and calls ``execute_compiled_graph``, so a
pre-execute hook (backend.device_io.add_pre_execute) learns which NEFF it runs before. The program-switch fence
(model_executor.program_fence) keys a model execution by it.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from sglang_neuron.backend import spmd_check, torch_pins

logger = logging.getLogger(__name__)

# md5 of each wrapped method's dedented source in torch_pins.TORCH_NEURONX_BUILD
_SOURCES = {
    "_prepare_inputs": "854811810186d81c4a61e9e0b87f473b",
    "_validate_copy_back_targets": "75186816b9e63cf3bb82c293d031a692",
    "_writeback_shadowed_aliases": "5bc1bbe059d51113e776c49a5c071f07",
    # not wrapped: the call protocol the cache relies on
    "__call__": "5b2ca9388610897872693f4590e0a7ab",
    "_allocate_output": "74c8f1351a8bec153bbb27191a4cb30d",
    "_process_outputs": "d83d5d1cd4586794123effd230c90ad5",
}
_installed = False
# bumped by invalidate(): an executor whose caches were filled under an older generation drops them at its next call,
# and a direct submit bound under one binds again (backend.direct_submit reads this list)
_generation = [0]
# per thread: the NEFF cache key of the executor whose inputs were prepared last and that has not submitted yet
_SUBMITTING = threading.local()


def invalidate() -> None:
    """Drop every executor's cached preparations and checks at its next call, and every direct submit's binding
    (backend.direct_submit), after a KV cache bind."""
    _generation[0] += 1


def _current(executor) -> None:
    """Drop ``executor``'s caches if they predate the last invalidate()."""
    if executor.__dict__.get("_sgn_generation") != _generation[0]:
        for name in ("_sgn_prepared", "_sgn_validated", "_sgn_in_place"):
            executor.__dict__.pop(name, None)
        executor._sgn_generation = _generation[0]


def _md5(cls, name: str) -> str:
    """md5 of ``cls.name``'s dedented source, or "" where it has none (a missing or a built-in attribute)."""
    return torch_pins.source_md5_or_empty(cls.__dict__.get(name))


def _dump_policy_set() -> bool:
    import os

    return bool(os.environ.get("TORCH_NEURONX_DYNAMO_TENSOR_DUMPS", "").strip())


def _device_str(device_id) -> str:
    return f"neuron:{device_id}"


def _same_storage(inp, prepared) -> bool:
    """Preparation returned ``inp`` or a view of it (no copy), so the prepared tensor follows every write to ``inp``."""
    import torch

    if not isinstance(inp, torch.Tensor) or not isinstance(prepared, torch.Tensor):
        return False
    if prepared is inp:
        return True
    root = inp._base if inp._base is not None else inp
    return prepared._base is root


def _prepare_one(executor, inp, index: int, device_str: str, disable_cpu_autocopy: bool):
    """One input of torch-neuronx's Executor._prepare_inputs (its loop body without the weight-layout relayout, which
    the caller leaves to the original)."""
    import torch

    if not isinstance(inp, torch.Tensor):
        tensor = torch.tensor(inp, device=device_str)
    elif inp.device.type == "cpu":
        if disable_cpu_autocopy:
            raise RuntimeError(f"Input tensor at index {index} is on cpu device, expected neuron. "
                               "Move tensor to Neuron device first with tensor.to('neuron')")
        tensor = inp.to(device_str)
    else:
        tensor = inp
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    if tensor.dtype in executor.INT_64_TO_32:
        if tensor.dim() == 0:
            tensor = tensor.reshape(1).view(executor.INT_64_TO_32[tensor.dtype])
        else:
            tensor = tensor.view(executor.INT_64_TO_32[tensor.dtype])
    elif tensor.dtype == torch.float64:
        tensor = tensor.to(torch.float32)
    elif tensor.dtype == torch.complex128:
        tensor = tensor.to(torch.complex64)
    return tensor


def _prepare_inputs(original):
    def prepare(self, inputs, device_id):
        _current(self)
        cache = self.__dict__.get("_sgn_prepared")
        wlo = self.__dict__.get("_sgn_wlo_indices")
        if wlo is None:
            wlo = self._sgn_wlo_indices = frozenset(int(t["input_index"]) for t in self.wlo_transforms)
        if cache is None or len(cache) != len(inputs):
            if self._tensor_dumper is not None and not _dump_policy_set():
                # the dumper rereads its policy from the environment on every call, twice; with none set at the
                # executor's first call it dumps nothing, and the serving process's environment does not change
                self._tensor_dumper = None
            # torch-neuronx's own preparation rereads this on every call; the serving process's environment does not
            # change, so a miss (every step's own inputs miss) uses the executor's first reading
            from torch_neuronx.neuron_dynamo_backend.settings import _getenv_bool

            self._sgn_no_autocopy = _getenv_bool("TORCH_NEURONX_DYNAMO_DISABLE_CPU_AUTOCOPY", False)
            prepared = original(self, inputs, device_id)
            self._sgn_prepared = [(inp, p) if i not in wlo and _same_storage(inp, p) else None
                                  for i, (inp, p) in enumerate(zip(inputs, prepared))]
            spmd_check.note_first_call(self)
            return prepared
        misses = [i for i, inp in enumerate(inputs) if cache[i] is None or cache[i][0] is not inp]
        if not misses:
            return [entry[1] for entry in cache]
        if wlo.intersection(misses):
            # a relayouted input is never cached here (torch-neuronx owns the relayout and its cache), so every call of
            # an executor with one takes torch-neuronx's own path; the plugin's serving graphs enable no relayout
            self._sgn_prepared = None
            return prepare(self, inputs, device_id)
        device_str = _device_str(device_id)
        no_autocopy = self._sgn_no_autocopy
        prepared = [None if entry is None else entry[1] for entry in cache]
        for i in misses:
            p = _prepare_one(self, inputs[i], i, device_str, no_autocopy)
            prepared[i] = p
            cache[i] = (inputs[i], p) if _same_storage(inputs[i], p) else None
        return prepared

    return prepare


def _announcing(prepare):
    def announce(self, inputs, device_id):
        prepared = prepare(self, inputs, device_id)
        _SUBMITTING.key = self.cache_key
        return prepared

    return announce


def submitting() -> Optional[str]:
    """The NEFF cache key of the graph this thread submits next, taken once (None afterwards). A submit that finds
    None is not a compiled graph's: torch-neuronx's eager kernels also submit through ``execute_compiled_graph``, and
    an executor's writeback copies come after its own submit."""
    key = getattr(_SUBMITTING, "key", None)
    _SUBMITTING.key = None
    return key


def _validate_copy_back_targets(original):
    def validate(self, inputs):
        _current(self)
        targets = [inputs[i] if 0 <= i < len(inputs) else None for i in self.copy_back_map.values()]
        last = self.__dict__.get("_sgn_validated")
        if last is not None and len(last) == len(targets) and all(a is b for a, b in zip(last, targets)):
            return None
        original(self, inputs)
        self._sgn_validated = targets
        return None

    return validate


def _writeback_shadowed_aliases(original):
    def writeback(self, inputs, tensors):
        pairs = [(tensors[out_idx], inputs[in_idx]) for out_idx, in_idx in self.copy_back_map.items()]
        last = self.__dict__.get("_sgn_in_place")
        if last is not None and len(last) == len(pairs) and all(
                r is lr and o is lo for (r, o), (lr, lo) in zip(pairs, last)):
            for out_idx, in_idx in self.copy_back_map.items():
                tensors[out_idx] = inputs[in_idx]
            if getattr(_WRITEBACK, "armed", False):
                _WRITEBACK.armed, _WRITEBACK.last = False, [(o, True) for _, o in pairs]
            return tensors
        # remember the call only when the NEFF wrote every result in place, by torch-neuronx's own test on the results
        # before it runs: it returns the caller's tensor whether it copied back or not, so its result cannot tell
        each = [r.device == o.device and r.data_ptr() == o.data_ptr() and r.shape == o.shape and r.stride() == o.stride()
                for r, o in pairs]
        result = original(self, inputs, tensors)
        self._sgn_in_place = pairs if all(each) else None
        if getattr(_WRITEBACK, "armed", False):
            _WRITEBACK.armed, _WRITEBACK.last = False, [(o, w) for (_, o), w in zip(pairs, each)]
        return result

    return writeback


# this thread's armed writeback record (arm_writeback, take_writeback)
_WRITEBACK = threading.local()


def arm_writeback() -> None:
    """Record the copy-back targets of this thread's next executor call (``take_writeback``). An unarmed call pays only
    one attribute test."""
    _WRITEBACK.armed, _WRITEBACK.last = True, None


def take_writeback() -> Optional[list]:
    """The copy-back targets of this thread's first executor call after ``arm_writeback``, each with whether the NEFF
    wrote it in place (False: torch-neuronx copied it back). Taken once; None without such a call.

    A PP stage uses it to prove its forward wrote the boundary into its seam slot. The returned tensor can't tell, since
    AOTAutograd and torch-neuronx's writeback both return the caller's tensor either way."""
    last = getattr(_WRITEBACK, "last", None)
    _WRITEBACK.armed, _WRITEBACK.last = False, None
    return last


def check_call_protocol() -> None:
    """Refuse a torch-neuronx whose ``Executor.__call__`` is not the pinned source: it calls
    ``torch_neuronx._C.execute_compiled_graph`` by module attribute, which backend.device_io.add_pre_execute wraps."""
    from torch_neuronx.neuron_dynamo_backend.executor import Executor

    if _md5(Executor, "__call__") != _SOURCES["__call__"]:
        raise RuntimeError("torch-neuronx's Executor.__call__ differs from the build the plugin's pre-execute hook "
                           f"was written for (torch-neuronx {torch_pins.TORCH_NEURONX_BUILD}: it calls "
                           "torch_neuronx._C.execute_compiled_graph by module attribute); re-validate "
                           "sglang_neuron/backend/device_io.add_pre_execute against this torch-neuronx")


def install() -> None:
    """Wrap torch-neuronx's Executor methods (idempotent); refuse a torch-neuronx whose wrapped sources differ."""
    global _installed
    if _installed:
        return
    from torch_neuronx.neuron_dynamo_backend.executor import Executor

    changed = [name for name, want in _SOURCES.items() if _md5(Executor, name) != want]
    if changed:
        raise RuntimeError(f"torch-neuronx's Executor.{', Executor.'.join(changed)} differ from the build the "
                           f"plugin's executor binding cache was written for (torch-neuronx "
                           f"{torch_pins.TORCH_NEURONX_BUILD}); re-validate sglang_neuron/backend/executor_cache.py "
                           "against this torch-neuronx")
    Executor._prepare_inputs = _announcing(_prepare_inputs(Executor._prepare_inputs))
    Executor._validate_copy_back_targets = _validate_copy_back_targets(Executor._validate_copy_back_targets)
    Executor._writeback_shadowed_aliases = _writeback_shadowed_aliases(Executor._writeback_shadowed_aliases)
    _installed = True
    logger.info("[neuron] torch-neuronx executor binding cache installed")
