# SPDX-License-Identifier: Apache-2.0
"""The pipeline-stage seam over NIXL: a pipelined server's stage boundary, and a pipelined PD decode's sampled tokens,
moved HBM to HBM with no host hop.

Each rank registers a send ring per kind it sends (one tensor per slot) and a landing ring per kind it receives. A step:

1. A non-last stage claims the next slot of each kind (``claim_send_slot``) before it launches its forward, which
   writes the boundary into the slots in place (the graph's ``*_out`` inputs). The claim waits only if the slot's
   previous step has not been acknowledged yet.
2. ``send_claimed`` queues the step with the forward's launch event and returns, so the scheduler moves on to the next
   micro-batch while the forward runs. The seam's worker thread waits for the event, pushes each slot into the peer's
   landing slot with one NIXL WRITE whose notification names (kind, step, rows), then hands this forward's own landing
   slots back to the previous stage.
3. The next stage takes the landing slot's view while it builds its inputs (``recv(..., wait=False)``) and waits for
   the WRITE right before its graph's submit (``await_landed``, a pre-execute hook), so the input build overlaps the
   previous stage's forward.
4. Once the consuming forward has run, the slot goes back to its sender with a two-integer gloo ack: from the worker
   after the stage's next send, or on the last stage from ``release_after`` (async forward) or ``release_all`` (sync
   forward, read back on the host). A sender reuses a slot only after its ack.

SGLang's PP loop still exchanges its proxy dict, carrying a one-element step id instead of the tensors, so its event
loop, micro-batching and request order stay SGLang's. A kind pairs (stage s, tp r) with (stage s + 1, tp r), as SGLang's
gloo PP transport does. Step ids are per-kind counters on both sides; SGLang drives every stage through the same batch
sequence, so the n-th send of a kind is its peer's n-th receive, and the receiver refuses any other step.

Kinds: "ph" and "pr", the prefill hidden states and residual, SP-sharded ``[bucket / attn_tp, hidden]``; "x0", "x1",
..., any further prefill boundary tensor a model declares, in its order, at its own row width and dtype
(``NeuronModel.pp_boundary_spec``); "dh", the decode hidden states, TP-replicated ``[batch bucket, hidden]``; and "tk",
a pipelined decode's sampled tokens, int32 ``[decode token width, 1]``, which the last stage samples into its slot in
place and relays to stage 0 for its next input ids, so no host waits for a sample. The first stage holds a token slot
until its next boundary send hands it back (or ``release`` for a step no forward reads). A transfer moves only the
step's rows. The model declares its boundary tensors and their widths and dtypes (GPT-OSS: ph, pr, dh). An aggregated
server's seam carries both phases' kinds (its prefill and decode runtimes share it) and no tokens, since SGLang's
aggregated PP loop passes them round the ring on the host.

A model may instead declare each prefill boundary tensor's kind itself (``BoundaryKind``, the model's
``pp_prefill_boundary_kinds``): its width and dtype, whether its rows are SP-sharded over the attention TP ranks, and a
number of row blocks. A slot holds ``blocks`` row blocks on every stage, so every boundary's slots have one size, and a
stage sends ``sent`` and receives ``received`` of them per step, a prefix of the slot (a boundary whose width in rows
grows along the pipeline, laid out block-major).

Failure: a failed seam operation leaves the stage pair out of step for good, so the first cause poisons the seam and
every later call raises it instead of a timeout it caused.

NIXL: the seam's agent has no progress thread; every seam call that waits polls NIXL itself, which progresses its rails.
Seam WRITEs carry a notification, which the target delivers once it has counted every write of the transfer. With NIXL
patch 0009 the target drops the notification of a partly failed transfer, so the WRITE's notification
("ppseam|<kind>|<step>|<rows>") proves the hop. Without 0009 (patch 0002 still notifies over a partly written slot) a
landing also needs the sender's commit ("ppseam|ok|<kind>|<step>"), posted once its WRITE ended DONE. Each rank checks
its loaded LIBFABRIC plugin for 0009 at seam build, and the commit is skipped only if every rank has it
(``notif_proves_landing``). A failed WRITE ends in ERR at the sender (patch 0007) and poisons its seam, and the receiver
never consumes the slot; the worker also posts "ppseam|err|<rank>|<cause>" so its receivers fail at once instead of
timing out.

Stale transfer ids: the target counts only notified transfers (patch 0008), each keyed by its sender and 16-bit transfer
id. An entry that never completes stays: a WRITE whose post failed partway, or whose write completion or notification
the target lost (EFA's unsolicited write receive, NIXL_LIBFABRIC_EFA_UNSOLICITED_WRITE_RECV=1, which every PD role sets,
can lose a remote write completion under load). A later WRITE of that sender drawing the same id would inherit its
counts and be notified before its own writes landed. None is posted while serving: a failed sender posts nothing more,
and a receiver that never sees a step's notification never acks its slot, so the sender posts at most ``capacity - 1``
more WRITEs to it (an id recurs only after 65,535 more) before both stages time out, poison their seams and stop the
server.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import threading
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from sglang_neuron.model_executor import host_trace

logger = logging.getLogger(__name__)

_SEAM: Optional["NeuronPPSeam"] = None


@dataclass(frozen=True)
class SeamSpec:
    """One kind's slot: ``rows`` x ``hidden`` elements of ``dtype``."""

    kind: str
    rows: int
    hidden: int
    dtype: torch.dtype

    @property
    def row_bytes(self) -> int:
        return self.hidden * self.dtype.itemsize

    @property
    def slot_bytes(self) -> int:
        return self.rows * self.row_bytes


@dataclass(frozen=True)
class BoundaryKind:
    """How one prefill boundary tensor crosses the seam: its kind (``kind``, the seam's name for it), row width and
    dtype, and its rows. A row block is the step's tokens over the attention TP ranks when ``sp_sharded``, else the
    step's tokens. The slot holds ``blocks`` row blocks of the largest prefill bucket; a step sends ``sent`` and
    receives ``received`` blocks of its own bucket."""

    kind: str
    hidden: int
    dtype: torch.dtype
    sp_sharded: bool = True
    blocks: int = 1
    sent: int = 1
    received: int = 1

    def block_rows(self, tokens: int, attn_tp: int) -> int:
        return int(tokens) // int(attn_tp) if self.sp_sharded else int(tokens)

    def slot_spec(self, largest_bucket: int, attn_tp: int) -> SeamSpec:
        return SeamSpec(self.kind, self.blocks * self.block_rows(largest_bucket, attn_tp), self.hidden, self.dtype)

    def sent_rows(self, tokens: int, attn_tp: int) -> int:
        return self.sent * self.block_rows(tokens, attn_tp)

    def received_rows(self, tokens: int, attn_tp: int) -> int:
        return self.received * self.block_rows(tokens, attn_tp)


# a string patches/nixl/0009 adds to the LIBFABRIC plugin (tools/build_nixl.sh checks it too)
NIXL_0009_MARKER = b"Dropping the notification for notif_xfer_id="


def loaded_plugin_drops_failed_notifications(maps: str = None, read=None) -> bool:
    """Whether the LIBFABRIC plugin this process loaded (per /proc/self/maps) carries patches/nixl/0009. ``maps`` and
    ``read`` (a path's bytes) stand in for /proc/self/maps and the file system in tests."""
    if maps is None:
        with open("/proc/self/maps") as f:
            maps = f.read()
    paths = sorted({line.split()[-1] for line in maps.splitlines()
                    if line.split() and line.split()[-1].endswith("/libplugin_LIBFABRIC.so")})
    if len(paths) != 1:
        return False  # no LIBFABRIC plugin loaded (another backend), or two: nothing to prove
    if read is None:
        def read(path):
            with open(path, "rb") as f:
                return f.read()
    return NIXL_0009_MARKER in read(paths[0])


def world_proves_landing(gathered: Sequence[Optional[dict]]) -> bool:
    """Whether every rank's NIXL drops a failed transfer's notification. Every rank takes the same answer, so a sender
    sends a commit exactly where its receiver waits for one."""
    return all(bool((g or {}).get("notif_proves")) for g in gathered)


def ring_capacity(pp_size: int) -> int:
    """Slots per kind: the PD prefill PP loop in overlap order holds up to pp_size + 2 micro-batches scheduled and not
    processed, within 2 * pp_size from PP2 up; at least four."""
    return 2 * max(2, int(pp_size))


def _poison_on_error(method):
    import functools

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except BaseException as error:
            self.poison(f"{method.__name__}: {type(error).__name__}: {error}")
            raise

    return wrapper


# the seam kind of each boundary tensor a stage's graph returns, by phase
BOUNDARY_KINDS = {
    "prefill": {"pipeline_hidden_states": "ph", "pipeline_residual": "pr"},
    "decode": {"pipeline_hidden_states": "dh"},
}
# a pipelined decode's sampled tokens, relayed from the last stage to the first
TOKEN_KIND = "tk"
_ACK_TAG_BASE = {"ph": 50_000, "pr": 60_000, "dh": 70_000, TOKEN_KIND: 80_000}
# the ack tags of the prefill kinds beyond _ACK_TAG_BASE, one band of _TAG_BAND each from _EXTRA_TAG_BASE. A model's
# further prefill boundary tensors by name are kind "x<i>" for the i-th, in band i; the kinds a model declares
# (BoundaryKind) take one band per kind, in the kinds' sorted order. A band holds every slot (ring_capacity stays far
# below it)
_EXTRA_KIND, _EXTRA_TAG_BASE, _TAG_BAND, _MAX_EXTRA = "x", 90_000, 10_000, 4
# kinds a model's prefill boundary may not take: the decode boundary's and the tokens' (an aggregated server's seam
# carries them too), and the words the seam's own notifications carry in the kind's field
_RESERVED_KINDS = frozenset(BOUNDARY_KINDS["decode"].values()) | {TOKEN_KIND, "ok", "err", "hello"}
_KIND_PATTERN = re.compile(r"[a-z][a-z0-9]{0,7}")


def boundary_kinds(names, phase: str = "prefill") -> Tuple[str, ...]:
    """The seam kinds of a stage's declared ``phase`` boundary names, in order. A prefill boundary may name tensors
    beyond the hidden states and residual: the i-th such name is kind "x<i>", so every stage of a model, which declares
    the same names, agrees on it. A decode boundary carries the hidden states alone."""
    fixed = BOUNDARY_KINDS[phase]
    extra = [name for name in names if name not in fixed]
    if extra and (phase != "prefill" or len(extra) > _MAX_EXTRA):
        raise ValueError(f"a {phase} pipeline boundary names {extra}: the seam carries {sorted(fixed)}"
                         + (f" and up to {_MAX_EXTRA} more prefill tensors" if phase == "prefill" else ""))
    return tuple(fixed[name] if name in fixed else f"{_EXTRA_KIND}{extra.index(name)}" for name in names)


def default_prefill_kinds(names, hidden: int, dtype: torch.dtype, widths=None) -> Dict[str, BoundaryKind]:
    """The prefill boundary of a model that declares no kinds: each name's fixed kind (``boundary_kinds``), SP-sharded
    rows of the hidden width and dtype, or of the name's own (``widths``, ``{name: (width, dtype)}``: the runtime's
    pp_boundary_specs), one row block each way."""
    widths = widths or {}
    return {name: BoundaryKind(kind, *_row(widths.get(name) or (hidden, dtype)))
            for name, kind in zip(names, boundary_kinds(names))}


def _row(spec) -> Tuple[int, torch.dtype]:
    width, dtype = spec
    return int(width), dtype


def check_prefill_kinds(declared, *, first: bool, last: bool,
                        architecture: str = "the model") -> Dict[str, BoundaryKind]:
    """A model's declared prefill boundary (``{name: BoundaryKind}``, in the order its graph takes them), refused with
    what to change if the seam cannot carry it on this stage."""
    if not isinstance(declared, dict) or not declared:
        raise ValueError(f"{architecture} declares pp_prefill_boundary_kinds {declared!r}: a non-empty dict of "
                         "boundary name -> pp_seam.BoundaryKind")
    problems = []
    kinds = [spec.kind if isinstance(spec, BoundaryKind) else None for spec in declared.values()]
    for name, spec in declared.items():
        if not isinstance(spec, BoundaryKind):
            problems.append(f"{name} is {type(spec).__name__}, not a pp_seam.BoundaryKind")
            continue
        if not _KIND_PATTERN.fullmatch(spec.kind) or spec.kind in _RESERVED_KINDS:
            problems.append(f"{name}'s kind {spec.kind!r} must be 1 to 8 lower-case letters or digits, starting with a "
                            f"letter, and none of {sorted(_RESERVED_KINDS)}")
        if kinds.count(spec.kind) > 1:
            problems.append(f"{name}'s kind {spec.kind!r} names another boundary tensor too")
        if int(spec.hidden) < 1 or not isinstance(spec.dtype, torch.dtype):
            problems.append(f"{name} needs a positive width and a torch dtype, not {spec.hidden} and {spec.dtype!r}")
        if not 0 <= spec.sent <= spec.blocks or not 0 <= spec.received <= spec.blocks or spec.blocks < 1:
            problems.append(f"{name} sends {spec.sent} and receives {spec.received} of a slot's {spec.blocks} row "
                            "blocks: a slot holds at least one, and a step moves at most the slot's")
        if not last and spec.sent < 1:
            problems.append(f"{name} sends no row block from a stage before the last")
        if not first and spec.received < 1:
            problems.append(f"{name} receives no row block on a stage after the first")
    if problems:
        raise ValueError(f"{architecture}'s prefill pipeline boundary: " + "; ".join(problems))
    return dict(declared)


def prefill_boundary(runtime) -> Dict[str, BoundaryKind]:
    """A pipeline stage runtime's prefill boundary by name, as its construction resolved it once
    (``NeuronModelRuntime._init_stage``): the model's declared kinds, or the fixed kinds of its ``pp_prefill_boundary``
    names at their widths (``default_prefill_kinds``)."""
    kinds = getattr(runtime, "pp_prefill_kinds", None)
    if kinds is None:
        raise ValueError("the runtime is no pipeline stage: its prefill boundary was not resolved "
                         "(NeuronModelRuntime._init_stage)")
    return kinds


def _ack_tag(kind: str, slot: int, kinds=()) -> int:
    """The ack's gloo tag of (kind, slot). A kind beyond _ACK_TAG_BASE takes the band of its index among the further
    kinds the seam carries (``kinds``), sorted, which both ends of a pairing hold alike (_connect). Without ``kinds``,
    a further tensor's kind "x<i>" takes band i: the same band, since a seam of names carries x0 ... x<n-1>."""
    base = _ACK_TAG_BASE.get(kind)
    if base is None:
        further = sorted(k for k in kinds if k not in _ACK_TAG_BASE)
        band = further.index(kind) if further else int(kind[len(_EXTRA_KIND):])
        base = _EXTRA_TAG_BASE + _TAG_BAND * band
    return base + slot


def seam_peers(kinds, *, pp_rank: int, pp_size: int, tp_rank: int,
               tp_size: int) -> Tuple[Dict[str, int], Dict[str, int]]:
    """This rank's send and receive peer for each kind, as PP-major world ranks: a boundary goes (stage s, tp r) to
    (stage s + 1, tp r), the sampled tokens wrap (last stage, tp r) to (stage 0, tp r). A kind missing from a side is
    not sent (received) here."""
    last = pp_size - 1
    world_rank = pp_rank * tp_size + tp_rank
    send: Dict[str, int] = {}
    recv: Dict[str, int] = {}
    for kind in kinds:
        if kind == TOKEN_KIND:
            if pp_size < 2:
                raise ValueError("the sampled-token relay needs at least two pipeline stages")
            if pp_rank == last:
                send[kind] = tp_rank
            if pp_rank == 0:
                recv[kind] = last * tp_size + tp_rank
            continue
        if pp_rank < last:
            send[kind] = world_rank + tp_size
        if pp_rank > 0:
            recv[kind] = world_rank - tp_size
    return send, recv


class NeuronPPSeam:
    """One seam per process; construction is collective over the gloo world."""

    SPIN_S = 0.02  # a receive polls back to back this long (a forward's wait), then every 0.2 ms
    # the boundary sends this rank writes to disk (distributed.pp_boundary_dump), or None
    _dump = None

    def __init__(self, *, specs: Dict[str, SeamSpec], pp_rank: int, pp_size: int, tp_rank: int, tp_size: int,
                 device: torch.device, gpu_id: int, timeout_s: float = 180.0):
        import torch.distributed as dist

        if not dist.is_initialized():
            raise RuntimeError("the PP seam needs torch.distributed initialized")
        self.specs = dict(specs)
        self.pp_rank, self.pp_size = int(pp_rank), int(pp_size)
        self.tp_rank, self.tp_size = int(tp_rank), int(tp_size)
        self.device = device
        self.gpu_id = int(gpu_id)
        self.timeout_s = float(timeout_s)
        self.capacity = ring_capacity(self.pp_size)
        self.world_rank = self.pp_rank * self.tp_size + self.tp_rank
        from sglang_neuron.distributed.pp_boundary_dump import BoundaryDump

        self._dump = BoundaryDump.from_env(self.pp_rank, self.tp_rank)
        if dist.get_rank() != self.world_rank:
            raise RuntimeError(f"PP seam rank layout: torch.distributed rank {dist.get_rank()}, PP-major rank "
                               f"{self.world_rank} (stage {self.pp_rank}, tp {self.tp_rank})")
        self._send_peer, self._recv_peer = seam_peers(self.specs, pp_rank=self.pp_rank, pp_size=self.pp_size,
                                                      tp_rank=self.tp_rank, tp_size=self.tp_size)
        self._agent = self._make_agent()
        # one tensor per send slot, so a forward writing one step never shares a buffer with another step's WRITE
        self._send_rings: Dict[str, List[torch.Tensor]] = {
            kind: [self._registered(self.specs[kind], ()) for _ in range(self.capacity)] for kind in self._send_peer}
        self._recv_rings: Dict[str, torch.Tensor] = {
            kind: self._registered(self.specs[kind], (self.capacity,)) for kind in self._recv_peer}
        regions = {kind: {"base": self._address(ring), "slot_bytes": self.specs[kind].slot_bytes,
                          "capacity": self.capacity, "dev": self.gpu_id} for kind, ring in self._recv_rings.items()}
        sends = {kind: [self.specs[kind].slot_bytes, self.capacity] for kind in self._send_peer}
        meta = {"agent": self._agent.get_agent_metadata(), "regions": regions, "sends": sends,
                "name": self._agent.name, "notif_proves": loaded_plugin_drops_failed_notifications()}
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, meta)
        # the WRITE notification alone proves a landing only if every rank's NIXL has patches/nixl/0009
        self.notif_proves_landing = world_proves_landing(gathered)
        if not self.notif_proves_landing:
            logger.warning("[neuron] PP seam: the loaded NIXL delivers a failed transfer's notification (no "
                           "patches/nixl/0009 on every rank), so each hop also waits for its sender's commit")
        self._peer_names: Dict[int, str] = {}
        self._remote_regions: Dict[str, Dict[str, int]] = {}
        self._connect(gathered)

        self._send_seq: Dict[str, int] = {k: 0 for k in self.specs}
        self._recv_seq: Dict[str, int] = {k: 0 for k in self.specs}
        # (kind, slot) -> (step, ack receive, its buffer): the release ack each WRITTEN slot waits for
        self._pending_acks: Dict[Tuple[str, int], Tuple[int, object, torch.Tensor]] = {}
        self._poisoned: Optional[str] = None
        # (kind, step) -> rows of the WRITE notifications not yet consumed, and the steps whose sender committed (a
        # landing needs both unless notif_proves_landing)
        self._notifs: Dict[Tuple[str, int], int] = {}
        self._committed: set = set()
        self._notif_lock = threading.Lock()
        # host trace only: (kind, step) -> time_ns its [notification, commit] was first drained, for the seam_seen row
        self._seen: Dict[Tuple[str, int], List[int]] = {}
        # kind -> (step, slot) of the landing slot this stage holds until its consuming forward has run
        self._held: Dict[str, Tuple[int, int]] = {}
        # kind -> (step, slot, rows) claimed by claim_send_slot and not sent yet
        self._claimed: Dict[str, Tuple[int, int, int]] = {}
        # deferred receives (kind, step, rows) whose WRITE the consuming forward waits for right before its submit
        self._landing: List[Tuple[str, int, int]] = []
        # one NIXL transfer handle per (kind, slot, rows), built once and posted again with each step's notification
        self._xfer_handles: Dict[Tuple[str, int, int], object] = {}
        # slot acks: point-to-point on a gloo group of their own (apart from SGLang's PP traffic), tagged by kind and
        # slot. The sender's worker arms each receive right after its WRITE; the receiver acks once the forward that
        # read the slot has run
        self._ack_group = dist.new_group(backend="gloo")
        # guards _transferred and _pending_acks, which the worker writes and the scheduler thread reads
        self._cv = threading.Condition()
        # per kind, the steps whose WRITE has completed
        self._transferred: Dict[str, int] = {k: 0 for k in self.specs}
        self._tasks: Optional["queue.Queue"] = None
        if self._send_peer or self._recv_peer:
            self._tasks = queue.Queue()
            threading.Thread(target=self._worker_loop, name="neuron-pp-seam", daemon=True).start()
        self._hello()
        dist.barrier()
        # on a device, deferred receives wait right before each graph submit; CPU mode receives with wait=True
        self.landing_hook = device.type != "cpu" and _install_landing_hook(self)
        logger.info("[neuron] PP seam up: stage %d/%d tp %d, %d slots of %s", self.pp_rank, self.pp_size,
                    self.tp_rank, self.capacity, sorted(self.specs))

    def _connect(self, gathered: Sequence[Optional[dict]]) -> None:
        """Check each kind's slots against what its peer published and add each send peer as a remote agent. A mismatch
        means the stages resolved different boundaries, and the seam refuses to come up."""
        for kind, peer in self._send_peer.items():
            spec = self.specs[kind]
            published = gathered[peer] or {}
            region = published.get("regions", {}).get(kind)
            if region is None or region["slot_bytes"] != spec.slot_bytes or region["capacity"] != self.capacity:
                raise RuntimeError(f"PP seam {kind}: the next stage's rank {peer} lands it in {region}, this stage "
                                   f"sends {spec.slot_bytes} B x {self.capacity} slots; the stages resolved different "
                                   "boundaries")
            self._remote_regions[kind] = region
            if peer not in self._peer_names:
                self._agent.add_remote_agent(published["agent"])
                self._peer_names[peer] = published["name"]
        for kind, peer in self._recv_peer.items():
            spec = self.specs[kind]
            sent = (gathered[peer] or {}).get("sends", {}).get(kind)
            if sent is None or list(sent) != [spec.slot_bytes, self.capacity]:
                raise RuntimeError(f"PP seam {kind}: the previous stage's rank {peer} sends {sent} (slot bytes, "
                                   f"slots), this stage lands {spec.slot_bytes} B x {self.capacity}; the stages "
                                   "resolved different boundaries")

    def _hello(self) -> None:
        """Open every NIXL connection up front: a hello to each send peer, then a wait for one from each receive peer.
        Without a progress thread a handshake only moves inside both ends' NIXL calls, so left to the first WRITE it
        could time the sender out while the receiver isn't polling yet."""
        for peer in sorted(set(self._send_peer.values())):
            self._agent.send_notif(self._peer_names[peer], f"ppseam|hello|{self.world_rank}".encode())
        waiting = set(self._recv_peer.values())
        deadline = time.monotonic() + self.timeout_s
        while waiting:
            for _agent_name, messages in (self._agent.get_new_notifs() or {}).items():
                for raw in messages:
                    text = raw.decode() if isinstance(raw, (bytes, bytearray)) else str(raw)
                    tag, _, rank = text.partition("|hello|")
                    if tag != "ppseam" or not rank.isdigit():
                        raise RuntimeError(f"PP seam: notification {text!r} before the seam is up")
                    waiting.discard(int(rank))
            if waiting:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"PP seam: no hello from ranks {sorted(waiting)} after {self.timeout_s:.0f} s")
                time.sleep(0)

    def _check_usable(self) -> None:
        if self._poisoned is not None:
            raise RuntimeError(f"PP seam unusable after an earlier failure: {self._poisoned}")

    def _check_landed(self) -> None:
        if self._landing:
            raise RuntimeError(f"PP seam: deferred receives {self._landing} were never awaited before their slots "
                               "went back (await_landed runs right before the consuming forward's submit)")

    def poison(self, cause: str) -> None:
        """Mark the seam broken; the first cause wins."""
        if self._poisoned is None:
            self._poisoned = cause

    def _registered(self, spec: SeamSpec, leading: tuple) -> torch.Tensor:
        tensor = torch.zeros((*leading, spec.rows, spec.hidden), dtype=spec.dtype, device=self.device).contiguous()
        ptr, nbytes = self._address(tensor), int(tensor.numel() * tensor.element_size())
        if not self._agent.register_memory([(ptr, nbytes, self.gpu_id, "")], "VRAM"):
            raise RuntimeError(f"PP seam {spec.kind}: NIXL refused to register {nbytes} B")
        return tensor

    def _address(self, tensor: torch.Tensor) -> int:
        """A registered tensor's device address, cached: the lookup synchronizes the device, and transfer handles are
        built while the stages serve."""
        cache = self.__dict__.setdefault("_addresses", {})
        address = cache.get(id(tensor))
        if address is None:
            from sglang_neuron.backend.device_io import device_address

            address = cache[id(tensor)] = device_address(tensor)
        return address

    @staticmethod
    def _make_agent():
        import uuid

        from nixl._api import nixl_agent, nixl_agent_config, nixl_thread_sync_t

        config = nixl_agent_config(enable_prog_thread=False, backends=[], num_threads=0,
                                   sync_mode=nixl_thread_sync_t.NIXL_THREAD_SYNC_STRICT)
        agent = nixl_agent(f"pp-seam-{uuid.uuid4().hex[:12]}", config)
        backend = os.environ.get("SGLANG_DISAGGREGATION_NIXL_BACKEND", "LIBFABRIC")
        params = json.loads(os.environ.get("SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS", "{}"))
        agent.create_backend(backend, params)
        return agent

    # the send side
    @_poison_on_error
    def claim_send_slot(self, kind: str, rows: int) -> Tuple[int, torch.Tensor]:
        """Claim the kind's next send slot for the forward to write its boundary into: (step, the slot's ``[rows,
        hidden]`` view). Waits until the slot's previous step was written and acked."""
        if kind not in self._send_rings:
            raise RuntimeError(f"PP seam {kind}: this rank sends no {kind} (the last stage has no successor)")
        spec = self.specs[kind]
        if not 0 < int(rows) <= spec.rows:
            raise ValueError(f"PP seam {kind}: {rows} boundary rows do not fit the {spec.rows}-row slot")
        self._check_usable()
        if kind in self._claimed:
            raise RuntimeError(f"PP seam {kind}: step {self._claimed[kind][0]} is claimed and not sent yet")
        step, slot = self._claim(kind)
        self._claimed[kind] = (step, slot, int(rows))
        return step, self._send_rings[kind][slot][:int(rows)]

    @_poison_on_error
    def send_claimed(self, kinds: Sequence[str], *, keep_alive=None, launch_event=None) -> int:
        """Send what the forward wrote into the claimed slots of ``kinds``; returns the step id the next stage presents.
        A device process must pass ``launch_event`` (backend.device_io.after_launch), which the worker waits on before
        its WRITE. ``keep_alive`` (the forward's inputs and outputs) is held until the worker ran the step, so no buffer
        of a queued forward is freed under it."""
        self._check_usable()
        self._check_landed()
        missing = [kind for kind in kinds if kind not in self._claimed]
        if missing or not kinds:
            raise RuntimeError(f"PP seam: no claimed slot of {missing or 'any kind'} to send")
        if launch_event is None and self.device.type != "cpu":
            raise RuntimeError("PP seam: a device stage's send needs its forward's launch event")
        staged = [(kind, *self._claimed.pop(kind)) for kind in kinds]
        steps = sorted({step for _kind, step, _slot, _rows in staged})
        if len(steps) != 1:
            raise RuntimeError(f"PP seam: the kinds of one boundary were claimed for different steps: {staged}")
        held, self._held = self._held, {}
        self._submit((staged, held, keep_alive, None, launch_event))
        return steps[0]

    def recv_tokens(self, step: int, rows: Optional[int] = None, wait: bool = True) -> Tuple[torch.Tensor, int]:
        """First stage: the tokens the last stage relayed as ``step`` (an int32 [rows] device view) and their row count.
        The slot is held like a boundary's, until the next boundary send or ``release(TOKEN_KIND)``."""
        view = self.recv(TOKEN_KIND, step, rows, wait=wait)[:, 0]
        return view, int(view.shape[0])

    @_poison_on_error
    def release_after(self, output: torch.Tensor, keep_alive=None, launch_event=None) -> None:
        """Hand this stage's held landing slots back once the forward that produced ``output`` has run (the last stage's
        async forward). The worker waits on ``launch_event`` (device) or else reads one element of ``output``, then
        acks."""
        self._check_usable()
        self._check_landed()
        if not self._held:
            return
        if self._tasks is None:
            raise RuntimeError("PP seam: an asynchronous forward's release needs the seam worker")
        held, self._held = self._held, {}
        self._submit(([], held, keep_alive, output, launch_event))

    def _submit(self, task) -> None:
        if self._tasks is None:
            self._run_task(task)  # a seam without a worker (CPU tests): the same order, inline
        else:
            self._tasks.put(task)

    def check_payload(self, kind: str, payload: torch.Tensor) -> None:
        """Refuse a boundary that does not fit the kind's slot (``[rows, hidden]`` of the kind's dtype, at most the
        slot's rows); a graph's write into the slot would otherwise cast or cut it silently."""
        spec = self.specs[kind]
        if (payload.dim() != 2 or int(payload.shape[0]) > spec.rows or int(payload.shape[1]) != spec.hidden
                or payload.dtype != spec.dtype):
            raise ValueError(f"PP seam {kind} payload {tuple(payload.shape)}/{payload.dtype} does not fit its slot "
                             f"[{spec.rows}, {spec.hidden}]/{spec.dtype}")

    def _claim(self, kind: str) -> Tuple[int, int]:
        """Wait until the kind's next slot is free, then assign its step (the kind's own counter): (step, slot)."""
        step = self._send_seq[kind]
        slot = step % self.capacity
        self._await_slot(kind, step)
        self._send_seq[kind] = step + 1
        return step, slot

    def _await_slot(self, kind: str, step: int) -> None:
        """The slot of ``step`` last carried step - capacity: wait until the worker WROTE that step, then for the
        receiver's ack."""
        prior = step - self.capacity
        if prior < 0:
            return
        deadline = time.monotonic() + self.timeout_s
        with self._cv:
            while self._transferred[kind] <= prior:
                self._check_usable()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"PP seam {kind}: step {prior} was not written within {self.timeout_s:.0f} s")
                self._cv.wait(remaining)
            pending = self._pending_acks.pop((kind, prior % self.capacity), None)
        if pending is None:
            return
        acked, work, buf = pending
        try:
            work.wait(timeout=timedelta(seconds=self.timeout_s))
        except RuntimeError as error:  # gloo reports its timeout as a RuntimeError
            raise TimeoutError(f"PP seam {kind}: the release ack for step {acked} (slot {prior % self.capacity}) did "
                               f"not arrive within {self.timeout_s:.0f} s") from error
        if int(buf[0]) != acked:
            raise RuntimeError(f"PP seam {kind} slot {prior % self.capacity}: ack for step {int(buf[0])}, expected "
                               f"{acked}")

    def _worker_loop(self) -> None:
        while True:
            task = self._tasks.get()
            try:
                self._run_task(task)
            except BaseException as error:  # noqa: BLE001 - recorded; the scheduler's next seam call raises it
                self.poison(f"seam worker: {type(error).__name__}: {error}")
                logger.error("[neuron] PP seam worker failed: %s: %s", type(error).__name__, error)
                self._notify_failure()
                with self._cv:
                    self._cv.notify_all()
            finally:
                task = None  # the step's keep_alive goes here, after it ran
                self._tasks.task_done()

    def _notify_failure(self) -> None:
        """Best effort: tell every rank this one sends to that its seam failed, so a receiver raises the cause at once
        instead of waiting out its timeout."""
        cause = (self._poisoned or "").replace("|", "/")[:400]
        for peer in sorted(set(self._send_peer.values())):
            try:
                self._agent.send_notif(self._peer_names[peer], f"ppseam|err|{self.world_rank}|{cause}".encode())
            except Exception:  # noqa: BLE001 - the receiver still times out; this rank already reports its cause
                logger.exception("[neuron] PP seam: telling rank %d of the failure failed", peer)

    def _run_task(self, task) -> None:
        # (staged, held, keep_alive, output, launch_event): wait for the forward (its event, or one element of
        # release_after's output), WRITE every staged slot, then hand the held landing slots back
        staged, held, _keep_alive, output, launch_event = task
        if self._poisoned is not None:
            return  # a broken pipeline moves nothing more
        tr = host_trace.trace()
        w0 = time.time_ns() if tr is not None else 0
        if launch_event is not None:
            launch_event.synchronize()
        elif output is not None:
            output.reshape(-1)[-1:].to("cpu")
        if tr is not None and staged:
            w1 = time.time_ns()
            tr.add("seam_fence", w0, w1, f"{staged[0][0]}{staged[0][1]}")
        if self._poisoned is not None:
            return  # poisoned while this step's forward ran
        if self._dump is not None and not self._dump.done and staged:
            # the forward has run and nothing writes the slots again before their WRITE and ack, so read_landed may copy
            # them on its side stream
            from sglang_neuron.backend.device_io import read_landed

            self._dump.write({kind: read_landed(self._send_rings[kind][slot][:rows]) for kind, _, slot, rows in staged},
                             {kind: step for kind, step, _, _ in staged})
        # WRITEs first: the next stage waits on them, while the acks only free slots of a ring with spares
        for kind, step, slot, rows in staged:
            self._write(kind, step, slot, rows)
        if staged:
            c0 = time.time_ns() if tr is not None else 0
            if not self.notif_proves_landing:
                # then the commits (only without 0009 on every rank): the receiver consumes a slot only once the sender
                # saw its WRITE succeed
                for kind, step, _slot, _rows in staged:
                    self._agent.send_notif(self._peer_names[self._send_peer[kind]],
                                           f"ppseam|ok|{kind}|{step}".encode())
            # with no progress thread, a queued notification only goes out on this rank's next NIXL call, which on a
            # send-only stage could be an idle stretch away. A finished transfer's state check does not progress the
            # rails; a drain does (and keeps what it receives for the scheduler)
            self._drain_notifs()
            if tr is not None:
                end, arg = time.time_ns(), f"{staged[0][0]}{staged[0][1]}"
                if not self.notif_proves_landing:
                    tr.add("seam_commit", c0, end, arg)
                tr.add("seam_write", w1, end, arg)
        self._acknowledge(held)

    def _write(self, kind: str, step: int, slot: int, rows: int) -> None:
        import torch.distributed as dist

        tr = host_trace.trace()
        spec = self.specs[kind]
        peer = self._send_peer[kind]
        remote = self._remote_regions[kind]
        notif = f"ppseam|{kind}|{step}|{rows}".encode()
        handle = self._xfer_handles.get((kind, slot, rows))
        if handle is None:
            nbytes = rows * spec.row_bytes
            src = self._agent.get_xfer_descs(np.array(
                [[self._address(self._send_rings[kind][slot]), nbytes, self.gpu_id]], dtype=np.uint64), "VRAM")
            dst = self._agent.get_xfer_descs(np.array(
                [[remote["base"] + slot * remote["slot_bytes"], nbytes, remote["dev"]]], dtype=np.uint64), "VRAM")
            handle = self._agent.initialize_xfer("WRITE", src, dst, self._peer_names[peer], notif)
            if not handle:
                raise RuntimeError(f"PP seam {kind} step {step}: NIXL built no transfer handle")
            self._xfer_handles[(kind, slot, rows)] = handle
        x0 = time.time_ns() if tr is not None else 0
        state = self._agent.transfer(handle, notif)
        deadline = time.monotonic() + self.timeout_s
        while state == "PROC":
            if time.monotonic() > deadline:
                raise TimeoutError(f"PP seam {kind} step {step}: the WRITE did not complete within "
                                   f"{self.timeout_s:.0f} s")
            state = self._agent.check_xfer_state(handle)
        if state != "DONE":
            raise RuntimeError(f"PP seam {kind} step {step}: WRITE state {state}")
        if tr is not None:
            tr.add("seam_xfer", x0, time.time_ns(), f"{kind}{step}")
        # arm this slot's release ack before the receiver can send it (it acks after its forward)
        buf = torch.empty(2, dtype=torch.int64)
        work = dist.irecv(buf, src=peer, tag=_ack_tag(kind, slot, self.specs), group=self._ack_group)
        with self._cv:
            self._pending_acks[(kind, slot)] = (step, work, buf)
            self._transferred[kind] = step + 1
            self._cv.notify_all()

    def _acknowledge(self, held: Dict[str, Tuple[int, int]]) -> None:
        import torch.distributed as dist

        for kind, (step, slot) in held.items():
            dist.send(torch.tensor([step, slot], dtype=torch.int64), dst=self._recv_peer[kind],
                      tag=_ack_tag(kind, slot, self.specs), group=self._ack_group)

    def flush(self) -> None:
        """Wait until the worker has run every queued step (tests; a drain before teardown)."""
        if self._tasks is not None:
            self._tasks.join()
        self._check_usable()

    # the receive side
    @_poison_on_error
    def recv(self, kind: str, step: int, rows: Optional[int] = None, wait: bool = True) -> torch.Tensor:
        """The landing slot of (kind, step) as a ``[rows, hidden]`` device view: exactly ``rows`` (another count from
        the sender is refused), or the sender's count when ``rows`` is None.

        ``wait=False`` (a forward's input build, ``rows`` given) returns the view at once and leaves the wait for its
        WRITE to ``await_landed``, right before the forward's submit."""
        if kind not in self._recv_rings:
            raise RuntimeError(f"PP seam {kind}: this rank receives no {kind} (the first stage has no predecessor)")
        self._check_usable()
        expected = self._recv_seq[kind]
        if step != expected:
            raise RuntimeError(f"PP seam {kind}: the scheduler presented step {step}, the seam expects {expected}: the "
                               "stages' batch order diverged")
        if kind in self._held:
            raise RuntimeError(f"PP seam {kind}: the previous slot was never released")
        slot = step % self.capacity
        if not wait:
            if rows is None:
                raise ValueError(f"PP seam {kind}: a deferred receive needs its row count")
            if not 0 < rows <= self.specs[kind].rows:
                raise RuntimeError(f"PP seam {kind} step {step}: {rows} rows, the slot holds 1..{self.specs[kind].rows}")
            self._recv_seq[kind] = expected + 1
            self._held[kind] = (step, slot)
            self._landing.append((kind, step, rows))
            return self._recv_rings[kind][slot, :rows]
        self._wait_notif(kind, step)
        sent_rows = self._take_landing(kind, step)
        self._recv_seq[kind] = expected + 1
        if not 0 < sent_rows <= self.specs[kind].rows:
            raise RuntimeError(f"PP seam {kind} step {step}: the sender shipped {sent_rows} rows, the slot holds "
                               f"1..{self.specs[kind].rows}")
        if rows is not None and sent_rows != rows:
            raise RuntimeError(f"PP seam {kind} step {step}: the sender shipped {sent_rows} rows, this stage "
                               f"compiled {rows}")
        self._held[kind] = (step, slot)
        return self._recv_rings[kind][slot, :sent_rows]

    def _wait_notif(self, kind: str, step: int) -> None:
        """Poll for (kind, step)'s landing: back to back for SPIN_S (the WRITE lands right after the previous stage's
        forward), then every 0.2 ms. A seam the worker poisoned meanwhile raises its cause, not a timeout."""
        deadline = time.monotonic() + self.timeout_s
        spin_until = time.perf_counter() + self.SPIN_S
        while not self._landed(kind, step):
            self._check_usable()
            self._drain_notifs()
            if self._landed(kind, step):
                break
            if time.monotonic() > deadline:
                missing = [what for what, have in (
                    ("the WRITE's notification", (kind, step) in self._notifs),
                    ("the sender's commit", self.notif_proves_landing or (kind, step) in self._committed)) if not have]
                raise TimeoutError(f"PP seam {kind} step {step}: {' and '.join(missing)} did not arrive within "
                                   f"{self.timeout_s:.0f} s")
            # sleep(0) yields the GIL (the seam worker shares it) without a timed sleep's slack
            time.sleep(0 if time.perf_counter() < spin_until else 0.0002)

    @_poison_on_error
    def await_landed(self) -> None:
        """Wait for the WRITE of every deferred receive and check its row count; runs right before the consuming
        forward's submit."""
        tr = host_trace.trace()
        while self._landing:
            kind, step, rows = self._landing[0]
            w0 = time.time_ns() if tr is not None else 0
            self._wait_notif(kind, step)
            if tr is not None:
                tr.add("seam_wait", w0, time.time_ns(), f"{kind}{step}")
                notif_t, commit_t = self._seen.get((kind, step), (0, 0))
                if notif_t:  # one instant where the commit came first, or where none is sent (0009)
                    tr.add("seam_seen", notif_t, max(notif_t, commit_t), f"{kind}{step}")
            sent_rows = self._take_landing(kind, step)
            self._landing.pop(0)
            if sent_rows != rows:
                raise RuntimeError(f"PP seam {kind} step {step}: the sender shipped {sent_rows} rows, this stage "
                                   f"compiled {rows}")

    def _landed(self, kind: str, step: int) -> bool:
        return (kind, step) in self._notifs and (self.notif_proves_landing or (kind, step) in self._committed)

    def _take_landing(self, kind: str, step: int) -> int:
        """Consume (kind, step)'s notification and commit (if one is sent); returns its rows."""
        self._committed.discard((kind, step))
        self._seen.pop((kind, step), None)
        return self._notifs.pop((kind, step))

    def _drain_notifs(self) -> None:
        """Move the agent's new notifications into ``_notifs`` and ``_committed``. The scheduler's waits and the worker
        both call it; only the scheduler consumes what it records."""
        with self._notif_lock:
            self._drain_notifs_locked()

    def _drain_notifs_locked(self) -> None:
        notifs = self._agent.get_new_notifs()
        if not notifs:
            return
        now = time.time_ns() if host_trace.trace() is not None else 0
        for _agent_name, messages in notifs.items():
            for raw in messages:
                text = raw.decode() if isinstance(raw, (bytes, bytearray)) else str(raw)
                if not text.startswith("ppseam|"):
                    continue
                fields = text.split("|", 3)
                if fields[1] == "err":
                    # the sending rank's seam failed: no commit will come
                    self.poison(f"PP seam: rank {fields[2]}'s seam failed: {fields[3] if len(fields) > 3 else ''}")
                    continue
                if fields[1] == "ok":
                    key = (fields[2], int(fields[3]))
                    self._committed.add(key)
                    if now:
                        self._seen.setdefault(key, [0, 0])[1] = now
                    continue
                _tag, kind, step, rows = fields
                key = (kind, int(step))
                self._notifs[key] = int(rows)
                if now:
                    self._seen.setdefault(key, [0, 0])[0] = now

    @_poison_on_error
    def release(self, kind: str) -> None:
        """Ack the held slot of one kind to its sender now, for a slot no forward reads any more."""
        self._check_usable()
        self._check_landed()
        if kind not in self._held:
            raise RuntimeError(f"PP seam {kind}: no slot of this kind is held")
        self._acknowledge({kind: self._held.pop(kind)})

    def release_all(self) -> None:
        """Ack every held slot once the consuming forward has run (the last stage of a synchronous step)."""
        if self._poisoned is not None:
            return  # a broken pipeline hands no slot back to a sender
        self._check_landed()
        held, self._held = self._held, {}
        self._acknowledge(held)


def _install_landing_hook(seam) -> bool:
    """Make every graph execution wait for the seam's deferred receives right before its submit
    (backend.device_io.add_pre_execute), so all of a stage's graph call but the submit overlaps the previous stage's
    WRITE."""
    from sglang_neuron.backend import device_io

    def await_landing():
        if seam._landing:
            seam.await_landed()

    device_io.add_pre_execute(await_landing)
    return True


def _specs(runtime) -> Dict[str, SeamSpec]:
    """The slots of a runtime's process. Prefill: each boundary kind's row blocks (``prefill_boundary``) of the largest
    prefill bucket; a model that declares no kinds has one block of the bucket over ``model_tp_size`` ranks
    (SP-sharded) for each ``pp_prefill_boundary`` name. Decode: the largest decode batch bucket (TP-replicated) for each
    ``pp_decode_boundary`` name, plus int32 [W, 1] sampled tokens with ``pp_token_relay`` (W = decode token width). An
    aggregated server's two runtimes (``phase_override`` set) share the process seam, so each builds both phases' slots
    from the same buckets."""
    phases = ("prefill", "decode") if getattr(runtime, "phase_override", None) is not None else (runtime.graphs,)
    specs = {}
    if "prefill" in phases:
        largest, attn_tp = int(runtime.buckets.prefill_token_buckets[-1]), int(runtime.model_tp_size)
        specs.update({spec.kind: spec.slot_spec(largest, attn_tp) for spec in prefill_boundary(runtime).values()})
    if "decode" not in phases:
        return specs
    hidden, dtype = int(runtime.pp_hidden_size), runtime.pp_boundary_dtype
    rows = int(runtime.buckets.num_seqs_buckets[-1])
    specs.update({kind: SeamSpec(kind, rows, hidden, dtype)
                  for kind in boundary_kinds(runtime.pp_decode_boundary, "decode")})
    if getattr(runtime, "pp_token_relay", False):
        specs[TOKEN_KIND] = SeamSpec(TOKEN_KIND, int(runtime.decode_token_width), 1, torch.int32)
    return specs


def get_seam(runtime, *, check: bool = False) -> NeuronPPSeam:
    """The process seam, built collectively at the first call (every rank calls it at warm-up).

    ``runtime`` gives the slots (``_specs``), ``tp_rank``, ``tp_size`` and its model runner's ``gpu_id``. ``check`` (at
    a runtime's construction) refuses a seam another runtime built for other slots."""
    global _SEAM
    if _SEAM is not None:
        if check:
            want = _specs(runtime)
            if want != _SEAM.specs:
                raise RuntimeError(f"this process's PP seam carries {sorted(_SEAM.specs)}, and this runtime's stage "
                                   f"{sorted(want)}")
        return _SEAM
    from sglang.srt.distributed import parallel_state as sgl_ps

    from sglang_neuron.model_executor.runtime_init import neuron_device

    pp = sgl_ps.get_pp_group()
    specs = _specs(runtime)
    _SEAM = NeuronPPSeam(specs=specs, pp_rank=pp.rank_in_group, pp_size=pp.world_size, tp_rank=runtime.tp_rank,
                         tp_size=runtime.tp_size, device=neuron_device(),
                         gpu_id=int(getattr(runtime.model_runner, "gpu_id", 0)))
    return _SEAM
