# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Weight DMA gates of the Kimi-K3 decode block.

A body frees its weight SBUF at its last matmul, which is the start of its output collective. The
weight DMAs of the next body then start at once and share the DMA rings with that collective, which
makes the collective slower. A gate puts those DMAs after the collective in the schedule.

The protocol has three calls:

- The block opens ``weight_gates()`` around the layers, and calls ``set_weight_gate`` with the
  output collective of the previous body before it calls the next body.
- A body passes the first DMA of each weight group to ``gate_weight_dma``. The DMA then follows the
  gate collectives in the schedule (a schedule edge, not a semaphore wait). Put a gated DMA on the SP
  or the GpSimd queue: the gate holds the in-order queue of its trigger engine, and the ACT queue also
  runs the activations.
- A body passes its output collectives to ``release_weight_gate``. The block takes the last one with
  ``take_released`` and makes it the gate of the next body. The output collectives of a body form a
  chain, so the last one completes last.

With ``weight_gates(consumer_gates=True)`` a body can also record the DMA that loads the result of its last output
collective (``release_gate_consumer``). A gated DMA with ``after_result`` then waits for that load (a semaphore wait),
so it starts after the collective has finished, not during its flight.

A prefetch (``PrefetchGate``) is the exception to the gate: its DMAs load the weights of a later body into fixed SBUF
tiles during the current body, so they wait for the last use of their destination bytes, not for a collective.

For a kernel that the block cannot change (``moe_tkg_decode``), ``gate_all_dmas()`` gates the
``nisa.dma_copy`` calls from HBM that it traces. A DMA into the same buffer view as an earlier gated DMA
gets no gate, because it waits for that DMA and its readers anyway. With ``gate_all_dmas(after=..., weights=...)``
the gated loads of the weights also wait for the given instructions (a semaphore wait), for example the first
reader of the result of the gate collective, so they start after that collective has finished.

Outside ``weight_gates()`` (a body that runs alone, or a block with the gates off) the calls do nothing.
The edges bind to instruction handles, not to names, so a change of a body cannot bind them to the
wrong instruction.
"""

from contextlib import contextmanager
from typing import NamedTuple

import nki.isa as nisa
import nki.language as nl

from .utils.kernel_assert import kernel_assert

_state = {"active": False, "wait": (), "late": (), "released": [], "consumer_gates": False, "result": ()}
# id(collective) -> (collective, the DMA that loads its result) (``release_gate_consumer``). Outside ``_state``, so a
# gate that the next block call takes (its ``gate`` argument) still finds the load. The entries keep both handles, so
# an id cannot be reused by another instruction while it is a key.
_RESULT_OF = {}
# id(collective) -> (collective, the DMA that loads its result, the trigger engine of that DMA).
# ``release_gate_consumer`` records it when the caller gives the engine, with or without consumer gates.
# ``wait_for_completion`` reads it. The entries keep the handles, so an id cannot be reused by another instruction while
# it is a key.
_RESULT_LOAD = {}
# id(DMA) -> the DMA: the owner writes of rows into an HBM pool (``record_pool_writes``). The compiler gives a later DMA
# that reads the pool at a runtime offset a RAW edge on each such write. ``wait_for_completion`` with ``pool_reads``
# reads it.
_POOL_WRITES = {}


@contextmanager
def weight_gates(enabled: bool = True, consumer_gates: bool = False):
    """Enable the gate calls for the bodies that run inside the scope.

    ``consumer_gates`` also enables ``release_gate_consumer`` and the ``after_result`` gates.

    Example:
        >>> with weight_gates():
        ...     set_weight_gate(())
    """
    previous = dict(_state, released=list(_state["released"]))
    _state.update(active=enabled, wait=(), late=(), released=[], consumer_gates=enabled and consumer_gates, result=())
    try:
        yield
    finally:
        _state.update(previous)


def set_weight_gate(producers, late=None) -> None:
    """Make the next ``gate_weight_dma`` calls wait for the instructions ``producers`` (``late``: the late calls).

    With consumer gates, the ``after_result`` calls also wait for the loads of the results of the ``late``
    collectives that ``release_gate_consumer`` recorded.
    """
    if _state["active"]:
        _state["wait"] = tuple(producers)
        _state["late"] = tuple(producers if late is None else late)
        results = (_RESULT_OF.get(id(p)) for p in _state["late"]) if _state["consumer_gates"] else ()
        _state["result"] = tuple(r[1] for p, r in zip(_state["late"], results) if r is not None and r[0] is p)


def gate_weight_dma(instr, *producers, late: bool = False, after_result: bool = False):
    """Make the weight DMA ``instr`` wait for the gate instructions and for ``producers``. Return ``instr``.

    ``producers`` are body-internal edges (for example a late weight after a gather of the same body). ``late`` waits
    for the late gate of ``set_weight_gate`` (the last collective of the previous body) in place of the gate.
    ``after_result`` also waits for the load of the result of that collective, when one is recorded (consumer gates),
    so the DMA does not share the DMA rings with the collective.
    Outside ``weight_gates()`` the call adds no edge, so a body that runs alone does not change.
    """
    wait = ((_state["late"] if late else _state["wait"]) + tuple(producers)) if _state["active"] else ()
    if wait:
        # A schedule edge orders the DMA after the gates as the form without ``kind`` does (neither is a semaphore
        # wait), and it stays in a NEFF that is built with a DMA-order file (that build drops the other form).
        instr.depends_on(*wait, kind=nisa.dependency.schedule)
    if after_result and _state["active"] and _state["result"]:
        # A semaphore wait: a schedule edge leaves no wait in the program, and the engine of the DMA would issue it as
        # soon as it reaches it, inside the flight of the collective.
        instr.depends_on(*_state["result"], kind=nisa.dependency.runtime)
    return instr


def order_after_gate_result(instr):
    """Put ``instr`` after the loads of the results of the gate collectives (``release_gate_consumer``) in the schedule
    (a schedule edge, not a semaphore wait). Return ``instr``.

    For a DMA on the queue of such a load that the body needs only later: without the edge the scheduler can put it
    ahead of the load on that in-order queue, where it holds the load behind its own waits. Without consumer gates, or
    outside ``weight_gates()``, the call adds no edge.
    """
    if _state["active"] and _state["result"]:
        instr.depends_on(*_state["result"], kind=nisa.dependency.schedule)
    return instr


def release_weight_gate(instr):
    """Record the output collective ``instr`` of the current body as a gate of the next body. Return ``instr``."""
    if _state["active"]:
        _state["released"].append(instr)
    return instr


def release_gate_consumer(instr, engine=None):
    """Record ``instr``, the DMA that loads the result of the last collective of ``release_weight_gate``, for the
    ``after_result`` gates of the next body (with consumer gates; otherwise nothing). Return ``instr``.

    ``engine`` is None (a DMA on the default queue), or the trigger engine of ``instr``, for example
    ``nisa.engine.sync``. With an engine, the call also records the load for ``wait_for_completion``, with or without
    consumer gates.
    """
    if _state["active"] and _state["released"]:
        collective = _state["released"][-1]
        if _state["consumer_gates"]:
            _RESULT_OF[id(collective)] = (collective, instr)
        if engine is not None:
            _RESULT_LOAD[id(collective)] = (collective, instr, engine)
    return instr


def record_pool_writes(writes) -> None:
    """Record the DMAs ``writes``, owner writes of rows into an HBM pool, for ``wait_for_completion`` ``pool_reads``.

    Example:
        >>> record_pool_writes(cache_writes)
    """
    for write in writes:
        if write is not None:
            _POOL_WRITES[id(write)] = write


def wait_for_completion(instr, producers, engine=None, pool_reads=False):
    """Make ``instr`` wait at runtime for the completion of each instruction of ``producers``. Return ``instr``.

    The released neuronx-cc 2.28 counts a runtime edge on a DMA or a collective producer twice in its dependency
    reduction pass. The pass then stops ("DepGraph: Num descendents already zero", or NCC_IXDR001). So this call gives
    no runtime edge to a gate collective whose result load (``release_gate_consumer``) has the trigger engine ``engine``
    of ``instr``. It gives ``instr`` a schedule edge on that load. The engine runs its instructions in order, and the
    load waits for the collective. So ``instr`` still starts after the collective has completed. ``pool_reads`` True:
    ``instr`` is a DMA that reads an HBM pool at a runtime offset. A pool write of ``record_pool_writes`` then gets a
    schedule edge: the compiler orders the read after the write (a RAW edge on the pool). Each other producer gets a
    runtime edge, as without ``pool_reads``. ``engine`` None and ``pool_reads`` False give a runtime edge to each
    producer.

    Example:
        >>> wait_for_completion(load, (out_all_reduce,), engine=nisa.engine.sync)
    """
    handle = instr if hasattr(instr, "depends_on") else nisa.NkiInstruction(instr)
    runtime = []
    for producer in producers:
        if producer is None:
            continue
        entry = _RESULT_LOAD.get(id(producer)) if engine is not None else None
        if pool_reads and _POOL_WRITES.get(id(producer)) is producer:
            handle.depends_on(
                producer if hasattr(producer, "depends_on") else nisa.NkiInstruction(producer),
                kind=nisa.dependency.schedule,
            )
        elif entry is not None and entry[0] is producer and entry[2] == engine:
            load = entry[1]
            handle.depends_on(
                load if hasattr(load, "depends_on") else nisa.NkiInstruction(load), kind=nisa.dependency.schedule
            )
        else:
            runtime.append(producer if hasattr(producer, "depends_on") else nisa.NkiInstruction(producer))
    if runtime:
        handle.depends_on(*runtime, kind=nisa.dependency.runtime)
    return instr


def released() -> tuple:
    """Return the collectives that the bodies recorded since the last ``take_released``, in order, and keep them."""
    return tuple(_state["released"]) if _state["active"] else ()


def take_released_pair(early: bool = False) -> tuple:
    """Return ``(take_released(early), the last collective)`` from the same records, and forget them all."""
    last = tuple(_state["released"][-1:])
    return take_released(early), last


def take_released(early: bool = False) -> tuple:
    """Return the last collective that the bodies recorded since the last call, and forget them all.

    The output collectives of a body form a chain (each one waits for the data of the one before), so
    the last one completes last, and a gate on it is enough. One producer for each gated DMA keeps the
    trace time low: the tracer's cost for an edge grows with the size of the kernel. With ``early``, return
    the collective before the last one (when there is one), so the next weights stream during the last one.
    """
    recorded = _state["released"]
    released = tuple(recorded[-2:-1] if early and len(recorded) > 1 else recorded[-1:])
    _state["released"].clear()
    return released


PREFETCH_KINDS = ("static", "gather")
"""Source kinds of a ``PrefetchLoad``: ``static`` (an HBM view known at trace time), ``gather`` (not supported yet)."""


class PrefetchLoad(NamedTuple):
    """One prefetch load; a ``gather`` stays single-pass: <= 512 data / index elements per partition per instruction.

    ``dst`` is the SBUF destination view, ``src`` the HBM source view, ``engine`` the trigger engine of the hardware
    DGE queue (None: the default queue), ``kind`` the source kind (``PREFETCH_KINDS``). A ``gather`` load would read
    its rows by an HBM index tensor on the Pool queue; ``PrefetchGate.issue`` refuses it.
    """

    dst: nl.NkiTensor
    src: nl.NkiTensor
    engine: object = None
    kind: str = "static"
    name: str = ""


class PrefetchGate:
    """Prefetch DMAs into fixed SBUF tiles that wait for the last use of their destination bytes, not for a gate.

    A body that uses the tiles of a slot records its last accesses (``release``). The next ``issue`` into the slot
    waits for them (a semaphore wait), and for nothing else: not for the gate of ``set_weight_gate``. So the loads of
    the next layer start when the bytes are free, during the attention of that layer, not after its attention output
    collective. The wait also orders a load after a write of another tensor at the same bytes (write after write),
    which the compiler orders only within one tensor. Issue the loads outside ``gate_all_dmas``, which would gate them.

    Example:
        >>> gate = PrefetchGate()
        >>> dmas = gate.issue([PrefetchLoad(tile, weights_hbm, nisa.engine.scalar)], slot="hot")
        >>> gate.release("hot", last_read)
    """

    def __init__(self):
        self.last_use = {}
        """slot -> the instructions of the last use of its bytes (``release``)."""

    def issue(self, loads, slot: str) -> list:
        """Issue the ``PrefetchLoad`` items ``loads`` into the tiles of ``slot`` and return their DMA instructions."""
        kernel_assert(
            getattr(nisa.dma_copy, "__name__", "") != "_gated_dma_copy",
            "Issue the prefetch loads outside gate_all_dmas: the scope would gate them",
        )
        after = self.last_use.pop(slot, ())
        dmas = []
        for load in loads:
            kernel_assert(load.kind in PREFETCH_KINDS, f"prefetch load kind must be one of {PREFETCH_KINDS}")
            kernel_assert(load.kind == "static", f"the {load.kind} prefetch loads are not supported yet")
            queue = {} if load.engine is None else {"engine": load.engine, "dge_mode": nisa.dge_mode.hwdge}
            instr = nisa.dma_copy(dst=load.dst, src=load.src, **queue, **({"name": load.name} if load.name else {}))
            instr = instr if hasattr(instr, "depends_on") else nisa.NkiInstruction(instr)
            if after:
                instr.depends_on(*after, kind=nisa.dependency.runtime)
            dmas.append(instr)
        return dmas

    def release(self, slot: str, *instrs) -> None:
        """Record ``instrs`` as the last accesses of the bytes of ``slot``: the next ``issue`` into the slot waits for
        them."""
        self.last_use[slot] = tuple(i if hasattr(i, "depends_on") else nisa.NkiInstruction(i) for i in instrs)


@contextmanager
def gate_all_dmas(
    per_queue: bool = False, after: tuple = (), weights: tuple = (), early: tuple = (), early_weights: tuple = ()
):
    """Pass every ``nisa.dma_copy`` from HBM that the scope traces to ``gate_weight_dma``.

    Use it around a body whose inputs come from the gate collectives, for example a feed-forward body
    after the attention output. Its activation DMAs wait for those collectives anyway, so the gate
    changes only when its weight DMAs start. This also reaches the DMAs of a kernel that the block
    cannot change, without the all-engine barrier of a loop fence (``nl.fori_loop``).

    A DMA into the same destination view as an earlier DMA of the scope is not gated. It waits for the
    earlier DMA (write after write) and for the readers of the earlier data (write after read), so it
    starts after the gate anyway. A weight ring that reuses its buffers then has one gated DMA for each
    buffer view, not one for each load (``moe_tkg_decode`` reuses one buffer for each projection).

    With ``per_queue``, only the first DMA from HBM of each queue (issuing engine and DGE mode) is gated. Its
    engine waits for the gate, so the later DMAs of the queue start after it. Use it for a body whose weight
    loads are in-order streams on a few queues (the TP128 selective-load kernel): one edge for each queue
    instead of one for each load, and the trace cost of an edge grows with the size of the kernel.

    With ``after``, each gated DMA that reads a static view (no runtime offset) of one of the HBM tensors
    ``weights`` also waits for the instructions ``after`` (a semaphore wait). The gate is a schedule edge, which
    leaves no wait in the program: the engine of a weight load issues it as soon as it reaches it, inside the
    flight of the gate collective, where it shares the DMA rings with that collective. A wait on the first reader
    of the collective's result starts the weight loads after the collective. The other gated DMAs (activations,
    and rows picked by a runtime index) wait for their producers anyway, so they get no wait. Empty ``after``
    (default) adds no wait.

    With ``early``, each DMA that reads a static view of one of the HBM tensors ``early_weights`` follows the
    instructions ``early`` in the schedule and waits for them (a semaphore wait) in place of the gate and of ``after``:
    it may start before the gate collective, once ``early`` has run (for example the end of the attention body's input
    projection, so that the next body's in-projection streams before that body's output collective, not in its
    flight). Empty ``early`` (default) changes nothing. Not with ``per_queue``.
    """
    if not (_state["active"] and _state["wait"]):
        yield
        return
    if per_queue and early:
        raise ValueError("gate_all_dmas: early takes the per-DMA gates, not per_queue")
    original = nisa.dma_copy
    # The destination views of the gated DMAs. The values keep each storage alive, so its id stays unique.
    written = {}
    queues = set()  # per_queue: the queues with a gated DMA
    # after: the storages of the weights, by id. The values keep each storage alive, so its id stays unique.
    storages = _gate_storages(weights) if after else ()
    weight_storages = {id(storage): storage for storage in storages if storage is not None}
    storages = _gate_storages(early_weights) if early else ()
    early_storages = {id(storage): storage for storage in storages if storage is not None}

    def _gate(instr, src):
        if early_storages and _static_view_of(src, early_storages):
            instr.depends_on(*early, kind=nisa.dependency.schedule)
            instr.depends_on(*early, kind=nisa.dependency.runtime)
            return instr
        gate_weight_dma(instr)
        if weight_storages and _static_view_of(src, weight_storages):
            instr.depends_on(*after, kind=nisa.dependency.runtime)
        return instr

    def _gated_dma_copy(*args, **kwargs):
        instr = original(*args, **kwargs)
        src = kwargs.get("src", args[1] if len(args) > 1 else None)
        # Gate only the DMAs that read HBM (weights). An SBUF source comes from the body's own work,
        # which waits for the gate collectives anyway.
        if instr is None or not nl.is_hbm(getattr(src, "buffer", None)):
            return instr
        if per_queue:
            queue = (str(kwargs.get("engine")), str(kwargs.get("dge_mode")))
            if queue in queues:
                return instr
            queues.add(queue)
            return _gate(instr, src)
        dst = kwargs.get("dst", args[0] if args else None)
        key = _view_key(dst)
        if key is not None:
            if key in written:
                return instr
            written[key] = dst._storage
        return _gate(instr, src)

    nisa.dma_copy = _gated_dma_copy
    try:
        yield
    finally:
        nisa.dma_copy = original


_POOL_AFTER = {"after": (), "edges": []}
"""The instructions that the next layer's GpSimd prework follows in the schedule (``set_pool_after``), and the
``(site, count)`` log of the edges ``pool_after`` added (a desk check reads it)."""


def set_pool_after(instrs) -> None:
    """Make the GpSimd prework that ``pool_after`` marks follow ``instrs`` in the schedule: the MoE tail's token-order
    gathers (``decode_block`` ``TAIL_POOL_PIN``), so that work of the next layer does not queue ahead of them on GpSimd.
    Schedule edges only: no wait, the values do not change."""
    _POOL_AFTER["after"] = tuple(instrs)


def clear_pool_after() -> None:
    """End ``set_pool_after`` (``decode_block_body`` calls it once its body is traced)."""
    _POOL_AFTER["after"] = ()


def pool_after(handle, site: str = ""):
    """Give the traced GpSimd prework instruction ``handle`` a schedule edge after the instructions of
    ``set_pool_after``; none set: no edge, the program does not change. Returns ``handle``."""
    after = _POOL_AFTER["after"]
    if not after or handle is None:
        return handle
    instr = handle if hasattr(handle, "depends_on") else nisa.NkiInstruction(handle)
    _POOL_AFTER["edges"].append((site, len(after)))
    for earlier in after:
        instr.depends_on(
            earlier if hasattr(earlier, "depends_on") else nisa.NkiInstruction(earlier), kind=nisa.dependency.schedule
        )
    return handle


def _gate_storages(tensors: tuple):
    """Yield the HBM storage of each tensor of ``tensors`` (None skipped): its ``_storage``, or each of its
    ``_gate_storages`` when it has them (a stack of several HBM tensors, ``moe_routed_plan.LayerStack``)."""
    for tensor in tensors:
        if tensor is None:
            continue
        several = getattr(tensor, "_gate_storages", None)
        yield from several if several is not None else (getattr(tensor, "_storage", None),)


def _static_view_of(src, storages: dict) -> bool:
    """Return True when ``src`` is a view of one of ``storages`` (by id) without a runtime offset."""
    if getattr(src, "_scalar_offset", None) is not None or getattr(src, "_vector_offset", None) is not None:
        return False
    return id(getattr(src, "_storage", None)) in storages


def _view_key(dst):
    """Return a key that is equal for two views of the same elements of one SBUF tensor, else None.

    The key holds the storage, the offset, the shape, and the strides, so equal keys mean the same
    elements. A view with a runtime offset, a dynamic shape, or a buffer other than SBUF gets None.
    """
    if dst is None or getattr(dst, "buffer", None) != nl.sbuf:
        return None
    storage = getattr(dst, "_storage", None)
    if storage is None or getattr(dst, "_scalar_offset", None) is not None:
        return None
    if getattr(dst, "_vector_offset", None) is not None:
        return None
    shape, strides, offset = tuple(dst.shape), tuple(dst.strides), dst.offset
    if not all(isinstance(v, int) for v in (*shape, *strides, offset)):
        return None
    return (id(storage), offset, shape, strides, str(dst.dtype))
