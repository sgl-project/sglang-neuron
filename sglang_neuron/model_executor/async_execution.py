# SPDX-License-Identifier: Apache-2.0
"""Device token relay and deferred output materialization for SGLang overlap.

The relay owns immutable per-forward tensors. Pool slot generations prevent a
finished request's future from being consumed after that slot is reallocated.
An unchanged batch passes its token tensor straight to the next compiled graph. Every
asynchronous decode has fixed-width token I/O (configure()), so a changed batch is reordered
inside the decode graph (TokenPlan); a decode batch resolved without a configured width is
refused. Without PP the one host read of a token is on an aggregated server, at a request's first decode
after its prefill: the prefill's token comes from the prefill's own readback, run by the scheduler
(waiting for that prefill) if the copy thread has not run it yet, so the step is read once. Every
launch's token vector is tagged with its phase (PendingSample records it as ``neuron_phase``), and
the relay never makes a prefill's vector the one a reordered decode indexes (TokenRelay.stash), even
when the prefill's vector happens to have the decode's width — only a decode's full-width vector may.

A whole-model megakernel that takes its decode ids on the host (graph_runner.NeuronModelRuntime.host_input_ids) runs
the host-ids mode instead (configure(host_ids=True), no token width): its steps return pending results like any other
(a step of several kernel calls through StepAssembly), and a decode batch resolves its ids on the host from them
(TokenRelay.host_ids), waiting for the previous step's readback only there, when the next step is built. SGLang's
result processing and the next batch's scheduling then run while the device runs the step.

Under PP the first decode stage gets its tokens from the last stage over the PP seam: each forward's rows become futures
of the next relayed step (stash_seam_tokens, SeamTokenInbox). A plan uses the landing slot of the forward most of its
rows come from and reads the other rows on the host; a step no forward reads is drained to the host within pp_size
forwards. The other stages embed nothing and take zero placeholder ids, as the first stage's plan does, so every stage
fingerprints the same batch.
"""

from __future__ import annotations

import collections
import functools
import logging
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)
_copy_executor = None
_pending_samples = weakref.WeakSet()
_pending_lock = threading.Lock()
# The newest launch's outputs and launch event (backend.device_io.after_launch), held strongly: a drain waits on it even
# if its sample was dropped.
_newest_launch = {"outputs": None, "event": None}


# The decode token I/O of this scheduler process's runtime (configure): the fixed decode token width, with which a
# changed batch is reordered in the graph (0: a runtime that serves no asynchronous decode).
_token_io = {"width": 0, "pp": None, "host_ids": False}
# whether a step's result is materialized inline at its CompletionEvent.synchronize (configure)
_inline_results = {"on": False}


def configure(width: int, inline_results: bool = False, pp=None, host_ids: bool = False) -> None:
    """Set the relay's decode token I/O from the runtime that serves this process's decode graphs, at its construction
    (NeuronModelRuntime.__init__, for a runtime of decode graphs; a prefill-only runtime leaves the width 0): the fixed
    width of the decode token vector, which reorders a changed batch in the graph. ``pp`` is a pipelined decode
    stage's role: "first" (its tokens come from the last stage over the PP seam), "other" (it embeds nothing), or None
    without PP. ``host_ids``: the runtime takes its decode ids on the host (the host-ids mode, module docstring), which
    has no token width and no PP relay."""
    if pp not in (None, "first", "other"):
        raise ValueError(f"a pipelined decode stage's role is 'first' or 'other', not {pp!r}")
    if host_ids and (int(width) or pp is not None):
        raise ValueError(f"the host-ids mode takes its decode ids on the host: no decode token width ({width}) and no "
                         f"PP token relay ({pp!r})")
    _token_io["width"], _token_io["pp"], _token_io["host_ids"] = int(width), pp, bool(host_ids)
    # A disaggregated decode processes each step's result only by synchronizing its CompletionEvent on the scheduler
    # thread, so the result is read there, inline, instead of on the copy thread the scheduler then waits for: one hop
    # and one cross-thread wake less per step on every rank. A prefill polls its results' query() and keeps the copy
    # thread.
    _inline_results["on"] = bool(inline_results)


class SeamTokenInbox:
    """A pipelined decode's first stage: the tokens the last stage relays over the PP seam (kind "tk"), one step per
    forward with rows, in forward order.

    ``device(step, width)`` first receives every earlier step to the host (handing its slot straight back, since no
    graph reads it) and returns ``step``'s landing slot, which the seam holds until the boundary send of the forward
    reading it. ``cpu(step)`` returns a step's host copy, for rows a changed batch takes from another step. A step whose
    slot went back with its forward has no copy, and asking for it again raises: the last stage may have written a later
    step there. A host copy lives as long as a token future of its step (``live``)."""

    def __init__(self, seam):
        self.seam = seam
        self.stashed = 0  # forwards registered so far (stash_seam_tokens), one relayed step each
        self.received = 0  # steps received so far
        self.host = {}
        self.live = weakref.WeakValueDictionary()  # step -> its _SeamTokens while a relay row holds it

    def _receive(self, step: int, rows=None, wait: bool = True):
        if step != self.received:
            raise RuntimeError(f"PP token relay: step {step} requested, the next one to receive is {self.received}")
        view = self.seam.recv_tokens(step, rows, wait=wait)[0]
        self.received = step + 1
        return view

    def drain_before(self, step: int) -> None:
        """Receive every step before ``step`` to the host, handing each slot back; keep host copies only of the steps a
        token future still holds."""
        from sglang_neuron.backend import device_io

        while self.received < step:
            early = self.received
            # a landing slot the network wrote: read beside the queued graphs
            tokens = device_io.read_landed(self._receive(early))
            self.seam.release("tk")
            if early in self.live:
                self.host[early] = tokens
        for old in [k for k in self.host if k not in self.live]:
            del self.host[old]

    def device(self, step: int, width: int):
        """``step``'s landing slot (int32 [width]) for the forward being built: the view at once where the seam's
        landing hook defers the WRITE wait to the forward's submit, else once it landed. A step already received to the
        host is uploaded from its copy."""
        if step < self.received:
            if step not in self.host:
                raise RuntimeError(f"PP token relay: step {step} went back to the last stage with the forward that "
                                   "read it")
            from sglang_neuron.backend import device_io
            from sglang_neuron.model_executor.runtime_init import neuron_device

            return device_io.upload(self.host[step], neuron_device())
        self.drain_before(step)
        return self._receive(step, width, wait=not self.seam.landing_hook)

    def cpu(self, step: int):
        if step in self.host:
            return self.host[step]
        if step < self.received:
            raise RuntimeError(f"PP token relay: step {step} went back to the last stage with the forward that read it")
        self.drain_before(step + 1)
        return self.host[step]


class _SeamTokens:
    """A first stage's token future: the tokens the last stage sampled for its forward ``step``, at the fixed decode
    token ``width``."""

    __slots__ = ("inbox", "step", "width", "__weakref__")

    def __init__(self, inbox, step, width):
        self.inbox, self.step, self.width = inbox, int(step), int(width)
        inbox.live[self.step] = self  # the step's host copy lives as long as this future

    def resolve(self):
        """The landing slot (device, int32 [width]); the forward that reads it waits for its WRITE at its submit."""
        return self.inbox.device(self.step, self.width)

    def to(self, device):
        if str(device) != "cpu":
            raise ValueError("a relayed token future resolves on the device through resolve()")
        return self.inbox.cpu(self.step)

    @property
    def shape(self):
        return (self.width,)


_seam_inbox = {"inbox": None}


def stash_seam_tokens(slots, generations, width: int, seam) -> None:
    """A pipelined decode's first stage, after launching a forward: register the forward's rows as futures of the next
    relayed "tk" step at the fixed decode token ``width``. Then receive to the host every step at least pp_size forwards
    old: an unread step (a micro-batch's last) would otherwise hold its slot until the last stage, its token ring full,
    stalls."""
    if _RELAY is None:
        raise RuntimeError("PP token relay: this scheduler has no FutureMap token relay")
    inbox = _seam_inbox["inbox"]
    if inbox is None:
        inbox = _seam_inbox["inbox"] = SeamTokenInbox(seam)
    step = inbox.stashed
    inbox.stashed = step + 1
    _RELAY.stash(tuple(int(s) for s in slots), tuple(int(g) for g in generations), _SeamTokens(inbox, step, width))
    # the just-launched forward's boundary send handed back the slot it read, so no token slot is held here
    inbox.drain_before(step - int(seam.pp_size) + 1)


_NO_SOURCE = -1  # token_control[1] value: take the host token id from token_control[0]


class TokenPlan:
    """How the next decode graph assembles its input ids on the device.

    prev: the latest decode forward's [W] int32 device tokens, or None when no row needs them.
    control: a fresh CPU int32 [2, W] tensor for a changed batch, or None for an unchanged batch
    (the runtime then passes a cached device identity control, so the steady state costs no H2D).
    """

    __slots__ = ("prev", "control")

    def __init__(self, prev, control):
        self.prev = prev
        self.control = control


def synchronize_pending():
    """Wait for this process's outstanding plugin graph outputs: each pending sample's readback waits on its launch
    event (backend.device_io.readback_many)."""
    with _pending_lock:
        pending = list(_pending_samples)
    for sample in pending:
        sample.materialize(return_logprob=False)


def drain_launches() -> None:
    """Wait for every submitted launch (scheduler thread).

    A launch whose sample was dropped unmaterialized is waited for too: launches run in
    submission order, so once the newest one's outputs are on the host every launch
    submitted so far (all from this scheduler thread) has finished."""
    synchronize_pending()
    from sglang_neuron.backend import device_io

    with _pending_lock:
        newest, event = _newest_launch["outputs"], _newest_launch["event"]
    if newest is not None:
        device_io.readback(newest["token_ids"], event)
    # and the launches that sample nothing: a pipeline stage's forward, whose boundary leaves on the seam
    device_io.wait_launches()


def _refuse_out_of_vocabulary(rows, vocab_size, what) -> None:
    """Raise when one of ``rows`` (a step's ids on the host, -1 where a speculative row is padded) is ``vocab_size`` or
    more, an id no token has; None checks nothing."""
    if vocab_size is None or rows.numel() == 0 or int(rows.numpy().max()) < vocab_size:
        return  # one reduction on the copy thread; the rows are found only for a refusal
    over = rows >= vocab_size
    if bool(over.any()):
        bad = over.reshape(over.shape[0], -1).any(dim=1).nonzero().reshape(-1)
        raise RuntimeError(f"{bad.numel()} {what} of a step hold a token id at or above the vocabulary size "
                           f"{vocab_size} (rows {bad[:8].tolist()}); the step is refused")


def refuse_invalid_token_ids(tokens, vocab_size=None) -> None:
    """Raise when a step's sampled ids (its real rows, on the host) hold an id no token has: a negative one, or one at
    or above the model's ``vocab_size`` (None: the sign alone). The device samplers always return a valid id
    (sampling._max_smallest_id; the sharded samplers mask their padding columns), so such a step's outputs are corrupt
    and are refused, never emitted. A valid step costs one host min/max pass on the copy thread."""
    if tokens.numel() == 0:
        return
    values = tokens.numpy()  # host ids: one torch call, then numpy's reductions, which take no further torch dispatch
    if int(values.min()) >= 0 and (vocab_size is None or int(values.max()) < vocab_size):
        return
    if bool((tokens < 0).any()):
        bad = (tokens < 0).nonzero().reshape(-1)
        raise RuntimeError(f"{bad.numel()} sampled row(s) of a step hold a negative token id "
                           f"(rows {bad[:8].tolist()}); the step is refused")
    _refuse_out_of_vocabulary(tokens, vocab_size, "sampled row(s)")


@functools.lru_cache(maxsize=None)
def _columns(width: int):
    """``[1, width]`` column indices, read-only, shared by every step of that width."""
    import numpy as np

    columns = np.arange(width).reshape(1, width)
    columns.flags.writeable = False
    return columns


def refuse_bad_spec_rows(tokens, vocab_size=None):
    """Raise when a speculative step's rows (``[rows, tokens per step]`` on the host: each request's accepted run, -1
    after it) are not well formed: every row emits at least one token, a -1 is only padding after the run, no other
    negative id appears, and no id is at or above the model's ``vocab_size`` (None: no upper bound). A row of another
    shape or ids is corrupt, and the step is refused, never emitted. Returns each row's accepted length as int32 (the
    step's ``accept_lens``). numpy's reductions, as refuse_invalid_token_ids: one torch call."""
    import torch

    values = tokens.numpy()
    accepted = values >= 0
    lens = accepted.sum(axis=1)
    prefix = _columns(values.shape[1]) < lens.reshape(-1, 1)
    bad = ((lens < 1) | (accepted != prefix).any(axis=1) | (values < -1).any(axis=1)).nonzero()[0]
    if bad.size:
        raise RuntimeError(f"{bad.size} speculative row(s) of a step are not an accepted run padded with -1 "
                           f"(rows {bad[:8].tolist()}); the step is refused")
    _refuse_out_of_vocabulary(tokens, vocab_size, "speculative row(s)")
    return torch.from_numpy(lens.astype("int32"))


class StepAssembly:
    """The tokens of a step that ran as several launches (a megakernel step of prefill chunk and lane calls): ``parts``,
    each launch's device token vector in launch order, read in one readback after the step's last launch event, and
    ``index`` (int64 [rows]), which picks each of the step's rows from the parts concatenated (each flattened).

    ``checks`` are more device outputs of the step's launches, read in the same readback, which ``check`` (given their
    host copies) inspects: a RuntimeError it raises refuses the step, as a negative token id does (a kernel's own
    per-row flags, read with the tokens instead of after each launch)."""

    __slots__ = ("parts", "index", "checks", "check")

    def __init__(self, parts, index, checks=(), check=None):
        import torch

        self.checks = tuple(checks)
        if self.checks and check is None:
            raise ValueError("a step's check outputs need the check that reads them")
        self.check = check
        self.parts = tuple(parts)
        if not self.parts:
            raise ValueError("a step's tokens come from at least one launch")
        self.index = torch.as_tensor(index, dtype=torch.int64).reshape(-1)
        total = sum(int(p.numel()) for p in self.parts)
        if self.index.numel() and (int(self.index.min()) < 0 or int(self.index.max()) >= total):
            raise ValueError(f"a step's row index {self.index.tolist()} reaches past its launches' {total} tokens")

    def tokens(self, host_parts):
        """The step's rows (int64) from the parts read to the host."""
        import torch

        flat = torch.cat([p.reshape(-1).to(torch.int64) for p in host_parts])
        return flat[self.index]

    def tensors(self) -> list:
        """What one readback reads: the parts, then the checks."""
        return [*self.parts, *self.checks]

    def read(self, hosts):
        """(the step's rows, the refusal or None) from the host copies of ``tensors()``."""
        tokens = self.tokens(hosts[:len(self.parts)])
        if self.check is None:
            return tokens, None
        try:
            self.check(hosts[len(self.parts):])
        except RuntimeError as refusal:
            return tokens, refusal
        return tokens, None


class PendingSample:
    def __init__(self, outputs, n_real, keep_alive=None, launch_event=None, phase=None, vocab_size=None,
                 assemble=None):
        """``launch_event``: the event recorded right after the launch that produced ``outputs``
        (backend.device_io.after_launch; None in a CPU-mode process), which their readback waits for. ``phase``: the
        launch's phase ("prefill", "decode"), recorded on its token vector (``neuron_phase``), so the token relay never
        makes a prefill's vector the one a reordered decode indexes (TokenRelay.stash). ``vocab_size``: the model's
        vocabulary, which every sampled id is below (refuse_invalid_token_ids, refuse_bad_spec_rows; None bounds the
        ids by their sign alone). ``assemble``: a StepAssembly when the step ran as several launches; its tokens are
        then read from the assembly's parts, and ``outputs["token_ids"]`` (the last launch's vector) is only the
        step's handle for the relay."""
        self.outputs = outputs
        self.n_real = n_real
        self.vocab_size = vocab_size
        self.keep_alive = keep_alive
        self.launch_event = launch_event
        self.assemble = assemble
        self._lock = threading.Lock()
        self._cpu = None
        self._refused = None
        self.top_logprobs_nums = ()
        # the token relay reaches the step's readback through its token vector (_TokenGroup.cpu); a weak reference, so
        # a dropped sample still leaves _pending_samples
        outputs["token_ids"].neuron_pending = weakref.ref(self)
        outputs["token_ids"].neuron_phase = phase
        with _pending_lock:
            _pending_samples.add(self)
            _newest_launch["outputs"], _newest_launch["event"] = outputs, launch_event

    def materialize(self, return_logprob=True):
        with self._lock:
            wanted = []
            if return_logprob:
                wanted = ["logprobs", "top_vals", "top_ids"] if any(self.top_logprobs_nums) else ["logprobs"]
            # the tokens on the first read, and every wanted output the host lacks, in one readback
            reads = [key for key in wanted if key in self.outputs and (self._cpu is None or key not in self._cpu)]
            if self._cpu is None:
                reads.insert(0, "token_ids")
            if reads:
                from sglang_neuron.backend import device_io

                refusal = None
                if self._cpu is None and self.assemble is not None:
                    # a step of several launches: its parts, its checks and the other outputs read in one readback
                    parts = self.assemble.tensors()
                    others = [key for key in reads if key != "token_ids" and key in self.outputs]
                    hosts = device_io.readback_many(parts + [self.outputs[key] for key in others], self.launch_event)
                    tokens, refusal = self.assemble.read(hosts[:len(parts)])
                    host = {"token_ids": tokens}
                    host.update(zip(others, hosts[len(parts):]))
                else:
                    host = dict(zip(reads, device_io.readback_many([self.outputs[key] for key in reads],
                                                                   self.launch_event)))
                if self._cpu is None:
                    import torch

                    self._cpu = {"tokens": host.pop("token_ids")[:self.n_real].to(torch.int64)}
                    self.keep_alive = None
                    with _pending_lock:
                        _pending_samples.discard(self)
                    try:
                        if refusal is not None:
                            raise refusal
                        if self._cpu["tokens"].dim() == 2:
                            # a speculative step's rows: each request's accepted tokens, -1 after them, and their
                            # lengths, which take_pending_sample hands SGLang
                            self._cpu["accept_lens"] = refuse_bad_spec_rows(self._cpu["tokens"], self.vocab_size)
                        else:
                            refuse_invalid_token_ids(self._cpu["tokens"], self.vocab_size)
                        # the token relay's host copy of the step's tokens (_TokenGroup.cpu), only of a step that passed
                        self.outputs["token_ids"].neuron_host_tokens = self._cpu["tokens"]
                    except RuntimeError as exc:
                        self._refused = exc
                self._cpu.update((key, value[:self.n_real]) for key, value in host.items())
            if self._refused is not None:  # on every read, whichever thread (copy or drain) read the step first
                raise self._refused
            for key in wanted:
                if key not in self._cpu:
                    raise RuntimeError(
                        f"this model's sampled output carries no {key!r}; admission "
                        "(sampling.validate_device_sampling_request) must refuse the request")
            return self._cpu


def mark_pending(res, pending: PendingSample):
    """Mark ``res`` (a step's LogitsProcessorOutput) as the result of an async launch, and return it. Every pending
    result is built here.

    The outputs reach the host when the result is processed (copy_to_cpu). Until then the scheduler only holds the
    step's device token vector (``neuron_sample``, which FutureMap.stash hands to the token relay). ``neuron_async``
    outlives ``neuron_pending_sample``: decisions every rank must make the same way read the flag, because the pending
    sample is cleared at a different time on each rank.

    A model's speculative step (a pending with ``fill_result``) keeps its device outputs with its worker, which carries
    them into the next step itself: it has no device token vector for the relay."""
    res.neuron_pending_sample = pending
    res.neuron_sample = None if hasattr(pending, "fill_result") else {"tokens": pending.outputs["token_ids"]}
    res.neuron_async = True
    return res


class CompletionEvent:
    """Host completion of a deferred copy, including propagation of failures. The scheduler synchronizes it when it
    processes the step's result."""

    def __init__(self, *args, **kwargs):
        self.future = None
        self.run = None  # an inline result: the copy synchronize() runs once on the calling thread
        self.recorded = False
        self.error = None  # the inline copy's failure, raised by every later synchronize() as a future's is

    def record(self, *args, **kwargs):
        self.recorded = True

    def synchronize(self):
        if self.run is not None:
            run, self.run = self.run, None
            try:
                run()
            except BaseException as error:
                self.error = error
                raise
            finally:
                self.recorded = True
        elif self.error is not None:
            raise self.error
        elif self.future is not None:
            self.future.result()

    def query(self):
        if self.run is not None:
            return False
        return self.future.done() if self.future is not None else self.recorded

    def wait(self, *args, **kwargs):
        self.synchronize()


class _CPUDeviceWithEvents:
    def __init__(self, base):
        self.base = base

    Event = CompletionEvent

    def __getattr__(self, name):
        return getattr(self.base, name)


class _TokenGroup:
    def __init__(self, keys, tokens):
        self.keys = tuple(keys)
        self.tokens = tokens
        self._cpu = None

    def cpu(self):
        """The group's tokens on the host.

        Normally this is the step's own readback (PendingSample.materialize), run here if the copy thread has not read
        it yet; either way the step is read once. Going through the live sample matters: a step with a negative
        token id is refused before the next decode uses its token, and the sample re-raises that refusal on every
        read. Only when the sample has been dropped does the group read its host copy (which a refused step never
        has) or the device tensor."""
        if self._cpu is None:
            ref = getattr(self.tokens, "neuron_pending", None)
            pending = ref() if ref is not None else None
            if pending is not None:
                host = pending.materialize(return_logprob=False)["tokens"]
            else:
                host = getattr(self.tokens, "neuron_host_tokens", None)
                if host is None:
                    host = self.tokens.to("cpu")
            self._cpu = host
        return self._cpu


class TokenRelay:
    """Consume-once, generation-qualified token futures; bounded by pool size."""

    def __init__(self):
        self.rows = {}
        # slot -> generation whose first token a CPU stash wrote into output_tokens_buf (prebuilt
        # or resumed requests). Consumed once, like device rows, so a stale host value is never used.
        self.host_fresh = {}
        # Group of the most recently launched non-empty decode forward: the only device token
        # vector the next graph can index. Idle (zero-row) stashes must not replace it.
        self.latest = None
        self.device_relays = 0
        self.reorders = 0
        self.host_rows = 0
        self.cross_group_rows = 0
        self.host_resolves = 0

    def discard(self, slots):
        for slot in slots:
            self.rows.pop(slot, None)

    def reset(self) -> None:
        """Forget every row (a successful flush_cache): the flush retires every request, so the relay drops their rows
        and the last decode's tokens with the pool; the slots' generations still reject a stale row on their own."""
        self.rows.clear()
        self.host_fresh.clear()
        self.latest = None

    def mark_host(self, slots, generations):
        self.discard(slots)
        self.host_fresh.update(zip(slots, generations))

    def stash(self, slots, generations, tokens):
        keys = tuple(zip(slots, generations))
        if not keys:
            return  # idle forward: keep `latest` pointing at the last real decode group
        group = _TokenGroup(keys, tokens)
        for row, (slot, generation) in enumerate(keys):
            self.rows[slot] = (generation, group, row)
            self.host_fresh.pop(slot, None)
        # With fixed-width decode token I/O only a decode's full-width vector is one the next decode can index: never
        # a prefill's (an aggregated server's), whatever its shape; a request's first decode reads that token on the
        # host, from the prefill's readback (the plan's cross-group rows, _TokenGroup.cpu)
        width = _token_io["width"]
        if not width or (tokens.shape == (width,) and getattr(tokens, "neuron_phase", None) != "prefill"):
            self.latest = group

    def host_ids(self, slots, generations, host_tokens) -> list:
        """The host-ids mode (module docstring): each row's next input id, read on the host. A row whose token a launched
        step holds takes it from that step's readback (_TokenGroup.cpu: the step's own read, run here if nothing read
        it yet, so the step is read once); a row a host stash wrote (a PD decode's newly landed or a resumed request)
        takes it from ``host_tokens`` (SGLang's FutureMap.output_tokens_buf). Consumes the rows like plan."""
        ids = []
        host_slots = []
        for row, (slot, generation) in enumerate(zip(slots, generations)):
            entry = self.rows.pop(slot, None)
            if entry is not None and entry[0] == generation:
                ids.append(int(entry[1].cpu()[entry[2]]))
                continue
            if self.host_fresh.pop(slot, None) != generation:
                raise RuntimeError(f"No decode token source for request slot {slot} generation {generation}")
            ids.append(None)
            host_slots.append((row, slot))
        if host_slots:
            values = host_tokens[[slot for _, slot in host_slots]].tolist()
            for (row, _), value in zip(host_slots, values):
                ids[row] = int(value)
            self.host_rows += len(host_slots)
        self.host_resolves += 1
        return ids

    def plan(self, slots, generations, host_tokens, width):
        """TokenPlan for one decode batch; never waits on a device future on the steady path."""
        import torch

        keys = tuple(zip(slots, generations))
        if len(keys) > width:
            raise RuntimeError("Decode batch exceeds the fixed decode token width")
        latest = self.latest
        if _token_io["pp"] == "first":
            # PP: micro-batches interleave, so the latest forward is another micro-batch's; take the group most of this
            # batch's rows come from
            counts = collections.Counter(self.rows[slot][1] for slot, generation in keys
                                         if self.rows.get(slot, (None,))[0] == generation)
            latest = counts.most_common(1)[0][0] if counts else None
        if latest is not None and keys and keys == latest.keys and all(
                self.rows.get(slot, (None, None))[1] is latest for slot in slots):
            for slot in slots:
                del self.rows[slot]
            self.device_relays += 1
            return TokenPlan(latest.tokens, None)
        host = [0] * width
        source = [_NO_SOURCE] * width
        host_slots = []
        for row, (slot, generation) in enumerate(keys):
            entry = self.rows.pop(slot, None)
            if entry is not None and entry[0] == generation:
                if entry[1] is latest:
                    source[row] = entry[2]
                else:
                    # A token from another forward than the latest decode: on an aggregated server, the request's
                    # prefill. It is read on the host through the prefill's own readback, which waits for that prefill
                    # if the copy thread has not read it yet (_TokenGroup.cpu). In PD decode a
                    # request that missed the last forward re-enters through prebuilt/resume (a CPU stash), so this
                    # is not expected there. Every such row is counted (cross_group_rows).
                    host[row] = int(entry[1].cpu()[entry[2]])
                    self.cross_group_rows += 1
                continue
            if self.host_fresh.pop(slot, None) != generation:
                raise RuntimeError(f"No decode token source for request slot {slot} generation {generation}")
            host_slots.append((row, slot))
        if host_slots:
            values = host_tokens[[slot for _, slot in host_slots]].tolist()
            for (row, _), value in zip(host_slots, values):
                host[row] = int(value)
            self.host_rows += len(host_slots)
        self.reorders += 1
        control = torch.tensor([host, source], dtype=torch.int32)  # fresh per step, never mutated
        prev = latest.tokens if any(s != _NO_SOURCE for s in source) else None
        return TokenPlan(prev, control)


# the token relay of this process's FutureMap, which a flush resets
_RELAY = None


def reset_token_relay() -> None:
    """Forget this process's relay rows (after a successful flush_cache)."""
    if _RELAY is not None:
        _RELAY.reset()


def _future_map_init(original_fn, self, device, spec_algo, req_to_token_pool, *args, **kwargs):
    global _RELAY
    original_fn(self, device, spec_algo, req_to_token_pool, *args, **kwargs)
    self.neuron_token_relay = _RELAY = TokenRelay()
    self.neuron_req_pool = req_to_token_pool


def _stash(original_fn, self, indices, payload, *args, **kwargs):
    from sglang_neuron.speculative.neuron_mtp import is_neuron_mtp

    if is_neuron_mtp(self.spec_algo):
        # a decode kernel's own MTP step carries every next input token on the device: no token of a step is relayed,
        # and a fresh request's first token reaches the device in the step's host inputs (speculative/neuron_mtp.py)
        return None
    if _token_io["pp"] is not None:
        # PP: the first stage's relay rows come from the seam (stash_seam_tokens), and a processed step's host stash
        # must not replace the next step's future; the other stages embed nothing
        if payload.bonus_tokens is not None and payload.bonus_tokens.device.type != "neuron":
            if _token_io["pp"] == "first":
                _mark_host_without_future(self, indices)
            return original_fn(self, indices, payload, *args, **kwargs)
        return None
    tokens = payload.bonus_tokens
    slots = tuple(indices.tolist())
    relay = self.neuron_token_relay
    generations = tuple(self.neuron_req_pool.req_generation[list(slots)].tolist())
    # in the host-ids mode a step's pending token handle is a relay row wherever it lives (a CPU-mode process's too)
    pending_handle = _token_io["host_ids"] and getattr(tokens, "neuron_pending", None) is not None
    if tokens is not None and (tokens.device.type == "neuron" or pending_handle):
        if not self.spec_algo.is_none():
            # a speculative step's next input is its draft input (bonus tokens and sequence lengths), which this relay
            # does not carry: a model's speculative worker under the overlap schedule relays its own on the device and
            # stashes host placeholders (models/kimi_k3/decode/spec_worker.py)
            raise RuntimeError("Neuron device token relay does not support speculative decoding: the speculative worker "
                               "must relay its own draft input on device under the overlap schedule")
        relay.stash(slots, generations, tokens)
        return
    relay.mark_host(slots, generations)
    return original_fn(self, indices, payload, *args, **kwargs)


def _mark_host_without_future(future_map, indices) -> None:
    """A pipelined decode's first stage: rows with no relayed future of their own generation (a PD decode's newly
    transferred or resumed requests, whose first input id the prefill sent) take the host token as their next input; a
    running request's rows keep their relayed future."""
    relay = future_map.neuron_token_relay
    slots = tuple(int(s) for s in indices.tolist())
    generations = tuple(int(g) for g in future_map.neuron_req_pool.req_generation[list(slots)].tolist())
    fresh = [(slot, generation) for slot, generation in zip(slots, generations)
             if relay.rows.get(slot, (None,))[0] != generation]
    if fresh:
        relay.mark_host(*map(tuple, zip(*fresh)))


def _resolve_inputs(original_fn, batch, future_map):
    import torch

    batch.neuron_token_plan = None
    if _token_io["pp"] == "other" and batch.prefill_input_ids_cpu is None and batch.input_ids is None:
        # a PP decode stage that embeds nothing: zero placeholder ids, like the first stage's, so every stage's batch
        # fingerprint hashes the same ids
        batch.input_ids = torch.zeros(len(batch.req_pool_indices), dtype=torch.int32)
        return None
    if batch.prefill_input_ids_cpu is None and batch.input_ids is None and future_map.spec_algo.is_none():
        slots = tuple(batch.req_pool_indices.tolist())
        generations = tuple(future_map.neuron_req_pool.req_generation[list(slots)].tolist())
        relay = future_map.neuron_token_relay
        if slots and _token_io["host_ids"]:
            # the host-ids mode: the ids on the host, from the previous step's readback (the module docstring)
            ids = relay.host_ids(slots, generations, future_map.output_tokens_buf)
            batch.input_ids = torch.tensor(ids, dtype=torch.int64)
            if relay.host_resolves == 1 or relay.host_resolves % 1000 == 0:
                logger.info("[neuron] decode ids on the host from the lagged readback: plans=%d (host rows=%d)",
                            relay.host_resolves, relay.host_rows)
            return
        if slots and not _token_io["width"]:
            # every asynchronous decode runtime configures its fixed decode token width (NeuronModelRuntime.__init__)
            raise RuntimeError("A decode batch resolved its tokens without a fixed decode token width")
        if slots:
            batch.neuron_token_plan = relay.plan(slots, generations, future_map.output_tokens_buf, _token_io["width"])
            # Shape-only placeholder for SGLang's CPU metadata; the graph assembles the real ids.
            batch.input_ids = torch.zeros(len(slots), dtype=torch.int32)
            count = relay.device_relays + relay.reorders
            if count == 1 or count % 1000 == 0:
                logger.info("[neuron] decode token plans=%d (identity=%d, reorder=%d, host rows=%d, cross-group "
                            "rows=%d)",
                            count, relay.device_relays, relay.reorders, relay.host_rows, relay.cross_group_rows)
            return
    # a mixed chunk (batch.mix_running_indices) never reaches here: the platform refuses --enable-mixed-chunk
    return original_fn(batch, future_map)


def _forward_batch_init(original_fn, cls, batch, *args, **kwargs):
    result = original_fn(cls, batch, *args, **kwargs)
    result.neuron_token_plan = getattr(batch, "neuron_token_plan", None)
    return result


def _scheduler_init_overlap(original_fn, self, *args, **kwargs):
    result = original_fn(self, *args, **kwargs)
    self.device_module = _CPUDeviceWithEvents(self.device_module)
    return result


def take_pending_sample(result, return_logprob: bool) -> None:
    """Write a result's pending sample (tokens, and logprobs where the batch returns them) into its host outputs once
    its forward has run. The token relay keeps the device token tensor separately until the next forward consumes it.
    A model's speculative step (a pending with ``fill_result``) writes the result's host fields itself: its tokens,
    accepted lengths and sequence lengths."""
    logits_output = result.logits_output
    pending = logits_output.neuron_pending_sample
    fill = getattr(pending, "fill_result", None)
    if fill is not None:
        fill(result)
        logits_output.neuron_pending_sample = None
        return
    sample = pending.materialize(return_logprob=return_logprob)
    tokens = sample["tokens"]
    if getattr(tokens, "ndim", 1) == 2:
        # a speculative step (speculative/neuron_mtp.py): SGLang's spec-v2 result processor reads row i as
        # next_token_ids[i * stride : i * stride + accept_lens[i]], the stride being the step's tokens per request
        result.next_token_ids = tokens.reshape(-1)
        result.accept_lens = sample["accept_lens"]
    else:
        result.next_token_ids = tokens
    logits_output.neuron_sample = sample
    if return_logprob:
        logits_output.next_token_logprobs = sample["logprobs"]
        nums = pending.top_logprobs_nums
        if any(nums):
            logits_output.next_token_top_logprobs_val = [sample["top_vals"][i, :k] for i, k in enumerate(nums)]
            logits_output.next_token_top_logprobs_idx = [sample["top_ids"][i, :k] for i, k in enumerate(nums)]
    logits_output.neuron_pending_sample = None


def _copy_result(original_fn, self, return_logprob, return_hidden_states=True):
    pending = getattr(self.logits_output, "neuron_pending_sample", None)
    if pending is None:
        return original_fn(self, return_logprob, return_hidden_states)
    global _copy_executor
    if _copy_executor is None and not _inline_results["on"]:
        _copy_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="neuron-output")
    event = CompletionEvent()
    self.copy_done = event

    def copy():
        take_pending_sample(self, return_logprob)
        original_fn(self, return_logprob, return_hidden_states)

    if _inline_results["on"]:
        event.run = copy
    else:
        event.future = _copy_executor.submit(copy)


# the upstream functions the hooks below skip for a device-resident batch, md5-pinned against upstream SGLang 43881a7
# (source_guard); the other hooks call the original and add the token relay around it
_SOURCES_MD5 = {
    ("sglang.srt.managers.overlap_utils", "FutureMap.stash"): "d2570dafb5b11a037f44f8b64f5f1d2f",
    ("sglang.srt.managers.overlap_utils", "resolve_forward_inputs"): "6bcb6d51708f84aee8ae3e87b257f472",
    ("sglang.srt.managers.utils", "GenerationBatchResult.copy_to_cpu"): "c582417d4ee3d2f60fc5a43878025301",
}


def register_hooks(registry, around):
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_SOURCES_MD5, __name__)
    for target, hook in (
        ("sglang.srt.managers.overlap_utils.FutureMap.__init__", _future_map_init),
        ("sglang.srt.managers.overlap_utils.FutureMap.stash", _stash),
        ("sglang.srt.managers.overlap_utils.resolve_forward_inputs", _resolve_inputs),
        ("sglang.srt.model_executor.forward_batch_info.ForwardBatch.init_new", _forward_batch_init),
        ("sglang.srt.managers.scheduler.Scheduler.init_overlap", _scheduler_init_overlap),
        ("sglang.srt.managers.utils.GenerationBatchResult.copy_to_cpu", _copy_result),
    ):
        registry.register(target, hook, around)
