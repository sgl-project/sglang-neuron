# SPDX-License-Identifier: Apache-2.0
"""The role workers of LingBot-World: its ENCODE and DECODE role instances beside the DiT.

One Neuron world on its own cores (:data:`ROLE_CORES`), one process per core, hosts the DECODE role (the streaming VAE decode) and the ENCODE role
(UMT5 and the condition encoder, on the decoder's VAE). DiT rank 0 starts it at load
(:class:`RoleWorkerProcesses`) and drives it through its queues (:class:`LingBotRoleWorkers`). Chunk ``n``'s decode is submitted when the DiT finishes chunk
``n`` and collected after chunk ``n + 1`` is launched; a session's condition sequence is
encoded from session open, one chunk at a time between the decodes, up to the encoder's fixed
point.

Payloads (:mod:`.payload`) travel through the ``torch.multiprocessing`` queues themselves
(tensors in shared memory): each rank's inbox carries the requests with their inputs, and the
shared outbox carries readiness, the world's rank 0's results and errors. Every rank runs the
requests in order.

The top level is standard library only: the spawned children import it before they pin their
cores.
"""

from __future__ import annotations

import contextlib
import logging
import os
import queue
import threading
import time
import traceback
from dataclasses import dataclass, field

from sglang_neuron.diffusion.models.lingbot_world import payload as P

logger = logging.getLogger(__name__)

_SHUTDOWN = "__shutdown__"
_DONE = object()
_ENV_LOCK = threading.Lock()

#: trn2.48xlarge at LNC 1 (the DiT owns cores 0-63): one role-worker world on cores 64-79, UMT5
#: tensor-parallel over it and the VAE on a 2 x 8 grid.
ROLE_CORES = tuple(range(64, 80))
VAE_GRID = (2, 8)
#: The world's Neuron runtime root-communicator port.
ROOT_COMM_PORT = 63410


@dataclass(frozen=True)
class RoleWorld:
    """The role workers' Neuron world (one rank per core at LNC 1) and its roles' config."""

    cores: tuple[int, ...]
    master_port: int
    root_comm_port: int
    model_config: dict = field(default_factory=dict)
    #: The logical NeuronCore size the cores are numbered in (the DiT world's).
    lnc: int = 1

    @property
    def world_size(self) -> int:
        return len(self.cores)


@dataclass
class RoleChannels:
    """The role-worker world's queues: one inbox per rank, the shared outbox, and the highest
    request seq the parent abandoned."""

    inboxes: list
    outbox: object
    cancelled: object


def role_world(*, geometry: dict) -> RoleWorld:
    """The role-worker world: :data:`ROLE_CORES`, the VAE grid and the DiT stage's ``geometry``
    (``ar_diffusion_height`` / ``ar_diffusion_width``).

    The rendezvous port is a free one above the kernel's ephemeral range, picked here (DiT rank 0,
    before the spawn). Inside the range, a rank that retries its connection before rank 0 listens
    can be handed the port as its own source port (a TCP self-connect) and hold it, and rank 0's
    store then fails to listen (EADDRINUSE).
    """
    from sglang_neuron.utils.hardware_config import get_logical_nc_config

    model_config = {
        k: geometry[k] for k in ("ar_diffusion_height", "ar_diffusion_width") if k in geometry
    }
    model_config["lingbot_vae_grid"] = list(VAE_GRID)
    port = _free_local_port(avoid=(ROOT_COMM_PORT,))
    return RoleWorld(ROLE_CORES, port, ROOT_COMM_PORT, model_config, get_logical_nc_config())


def _ephemeral_port_range() -> tuple[int, int]:
    try:
        with open("/proc/sys/net/ipv4/ip_local_port_range") as f:
            lo, hi = (int(v) for v in f.read().split())
        return lo, hi
    except (OSError, ValueError):
        return 32768, 60999  # the Linux default


def _free_local_port(avoid: tuple[int, ...] = ()) -> int:
    """A TCP port free on 127.0.0.1 now and outside the ephemeral range (the role world's
    rendezvous; ``avoid``: ports taken by the world's other listeners)."""
    import socket

    lo, hi = _ephemeral_port_range()
    candidates = [*range(hi + 1, 65536), *range(1024, lo)]
    for port in candidates:
        if port in avoid:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))  # the store's server listens on every interface
            except OSError:
                continue
            return port
    raise RuntimeError("no free TCP port outside the ephemeral range for the role workers")


# --------------------------------------------------------------------------- payloads


def _ms(metadata: dict) -> float:
    """A payload's own ``seconds`` in ms."""
    return round(float(metadata.get("seconds", 0.0)) * 1e3, 2)


# ------------------------------------------------------------------------------ worker side


def _init_world(world: RoleWorld, rank: int):
    """Pin this child to its core and join the role-worker world (torch-neuronx)."""
    os.environ.pop("NEURON_VISIBLE_DEVICES", None)
    os.environ.update(
        {
            # The plugin's worker start-up takes rank r's core from this list
            # (cpu_placement.server_cores).
            "NEURON_RT_VISIBLE_CORES": ",".join(str(c) for c in world.cores),
            "NEURON_LOGICAL_NC_CONFIG": str(world.lnc),
            "NEURON_RT_ROOT_COMM_ID": f"127.0.0.1:{world.root_comm_port}",
            "MASTER_ADDR": "127.0.0.1",
            # Gloo's sockets on the loopback interface (the world is one host).
            "GLOO_SOCKET_IFNAME": "lo",
            "MASTER_PORT": str(world.master_port),
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world.world_size),
            "LOCAL_WORLD_SIZE": str(world.world_size),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    import torch
    from sglang.multimodal_gen.runtime.distributed import parallel_state as sglang_groups

    from sglang_neuron.diffusion import runtime

    neuron = runtime.configure_worker(rank)
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    # SGLang's world and groups, as its worker builds them; the models are bound to them below.
    backend = "cpu:gloo,neuron:neuron"
    sglang_groups.init_distributed_environment(
        world_size=world.world_size,
        rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{world.master_port}",
        local_rank=rank,
        backend=backend,
        device_id=neuron,
    )
    sglang_groups.initialize_model_parallel(
        tensor_parallel_degree=world.world_size, sequence_parallel_degree=1, backend=backend
    )
    device = runtime.set_worker_device()
    # One group: UMT5 is tensor-parallel over the world, the VAE grid spans it.
    runtime.initialize_model_groups(
        tensor_parallel_size=world.world_size, context_parallel_size=1, cfg_parallel_size=1
    )
    return device


def build_role(stage_role: str, world: RoleWorld, model_path: str, device, shared_vae=None):
    """A role instance (``"encode"`` / ``"decode"``) in the initialized world."""
    from sglang_neuron.diffusion.models.lingbot_world.roles import (
        LingBotDecodeRole,
        LingBotEncodeRole,
        RoleGeometry,
    )

    config = dict(world.model_config)
    geometry = RoleGeometry(
        model_path=model_path,
        height=int(config.get("ar_diffusion_height", 480)),
        width=int(config.get("ar_diffusion_width", 832)),
        frames_per_chunk=int(config.get("frames_per_chunk", 3)),
        device=device,
    )
    if stage_role == "decode":
        return LingBotDecodeRole(geometry, config, compile_graphs=True)
    return LingBotEncodeRole(geometry, config, compile_graphs=True, vae=shared_vae)


class RoleHost:
    """One rank's DECODE and ENCODE instances behind the role workers' requests.

    ``build(stage_role, shared_vae=None)`` makes a role instance (tests pass stand-ins).
    """

    def __init__(self, world: RoleWorld, rank: int, build) -> None:
        self.rank = rank
        self.decode = build("decode")
        self.encode = build("encode", shared_vae=self.decode.vae)

    def encode_steps(self, session_id, image, emit, stop=None, chunks=None):
        return self.encode.encode_condition_steps(session_id, image, emit, stop, chunks=chunks)

    def __call__(self, cmd, payload):
        if cmd == "decode":
            # Every rank's inbox holds the chunk's latents; rank 0 returns the frames.
            out = self.decode.decode_payload(payload)
            return out if self.rank == 0 else None
        if cmd == "release":
            self.decode.release_decode_session(payload)
            return None
        if cmd == "encode_prompt":
            request_id, input_ids, attention_mask = payload
            out = self.encode.encode_prompt_payload(request_id, input_ids, attention_mask)
            return out if self.rank == 0 else None
        raise ValueError(f"unknown role-worker command {cmd!r}")


def _rank0_choice(value: int) -> int:
    """Rank 0's ``value`` on every rank of the world."""
    import torch

    from sglang_neuron.diffusion.models.lingbot_world import dist as lingbot_dist

    t = torch.tensor([value], dtype=torch.int32)
    return int(lingbot_dist.broadcast_cpu_from_rank0(t, (1,), torch.int32)[0])


def _serve(role, rank: int, inbox, outbox, cancelled) -> bool:
    """Serve requests in order; run session encodes one chunk at a time between them.

    Returns ``True`` after a shutdown request, ``False`` after a failure: a request or an encode
    chunk that raised is reported and ends the loop, since the world's ranks may then disagree
    on their collectives (the caller exits, and DiT rank 0 fails its request on the error).

    With no encode active, every rank takes the next request (all inboxes get the same
    requests in the same order). While one is active, rank 0 chooses before each step -- the
    next queued request if one has arrived (a decode waits behind at most one encode chunk),
    else one more encode chunk -- and broadcasts the choice, so every rank issues the same
    collectives in the same order. An idle world never waits in a collective.
    """
    encodes: list = []  # [(seq, generator)], oldest first

    def handle(seq, cmd, payload) -> bool | None:
        """``True``: go on; ``False``: shut down; ``None``: failed."""
        if cmd == _SHUTDOWN:
            return False
        try:
            if cmd == "encode_session":
                session_id, image, chunks = payload

                def emit(chunk_handle, seq=seq):
                    outbox.put(("partial", seq, rank, chunk_handle))

                def stop(seq=seq):
                    return cancelled is not None and cancelled.value >= seq

                encodes.append((seq, role.encode_steps(session_id, image, emit, stop, chunks)))
                return True
            result = role(cmd, payload)
        except BaseException:  # noqa: BLE001 - surface the failure on the request
            outbox.put(("error", seq, rank, traceback.format_exc()))
            return None
        if rank == 0:
            outbox.put(("ok", seq, rank, result))
        return True

    while True:
        if not encodes or _rank0_choice(0 if rank != 0 or inbox.empty() else 1):
            go_on = handle(*inbox.get())
            if go_on is None:  # failed
                return False
            if go_on is False:  # shutdown
                return True
            continue
        seq, steps = encodes[0]
        try:
            done = next(steps, _DONE) is _DONE
        except BaseException:  # noqa: BLE001 - the encode ends with its error
            outbox.put(("error", seq, rank, traceback.format_exc()))
            return False
        if done:
            encodes.pop(0)
            if rank == 0:
                outbox.put(("ok", seq, rank, None))


def _worker_main(
    world: RoleWorld, rank: int, model_path: str, inbox, outbox, cancelled, parent_pid: int
) -> None:
    """Child entry point: build the role instances, report readiness, then serve requests."""
    _watch_parent(parent_pid)
    try:
        device = _init_world(world, rank)
        host = RoleHost(
            world,
            rank,
            lambda role, **kw: build_role(role, world, model_path, device, **kw),
        )
    except BaseException:  # noqa: BLE001 - report every start-up failure to the parent
        outbox.put(("error", -1, rank, traceback.format_exc()))
        return
    logger.info("LingBot role workers rank %d (core %d) ready", rank, world.cores[rank])
    outbox.put(("ready", -1, rank, None))
    if not _serve(host, rank, inbox, outbox, cancelled):
        # Fatal: deliver the error report, then exit (the other ranks' next collective fails,
        # and DiT rank 0 fails on the reported error).
        outbox.close()
        flush = threading.Thread(target=outbox.join_thread, daemon=True)
        flush.start()
        flush.join(timeout=30)
        os._exit(1)
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


# ------------------------------------------------------------------------------ parent side


class RoleWorkerProcesses:
    """The role-worker world's processes, started by DiT rank 0 at load.

    SGLang's diffusion workers are daemonic, and ``multiprocessing`` refuses children of a
    daemonic process, so the start lifts that flag for the duration of the spawns
    (:func:`_spawn_from_daemon`). The children are daemonic themselves (they end with DiT rank
    0) and exit when their parent is gone (:func:`_watch_parent`).
    """

    def __init__(self, world: RoleWorld, model_path: str) -> None:
        import torch.multiprocessing as tmp

        # torch's queues carry tensors through shared memory.
        ctx = tmp.get_context("spawn")
        self.world = world
        self.channels = RoleChannels(
            inboxes=[ctx.Queue() for _ in world.cores],
            outbox=ctx.Queue(),
            cancelled=ctx.Value("q", 0),
        )
        self.processes = []
        # The children read their environment at exec time: start them on the world's first
        # core (the rest is set in _init_world), not on the stage process's devices.
        with _ENV_LOCK, _spawn_from_daemon():
            saved = os.environ.get("NEURON_RT_VISIBLE_CORES")
            os.environ["NEURON_RT_VISIBLE_CORES"] = str(world.cores[0])
            try:
                for rank, inbox in enumerate(self.channels.inboxes):
                    proc = ctx.Process(
                        target=_worker_main,
                        args=(
                            world,
                            rank,
                            model_path,
                            inbox,
                            self.channels.outbox,
                            self.channels.cancelled,
                            os.getpid(),
                        ),
                        name=f"lingbot-role-{rank}",
                        daemon=True,
                    )
                    proc.start()
                    self.processes.append(proc)
            except BaseException:
                # The ranks already started would wait in their world's rendezvous: stop them.
                for proc in self.processes:
                    proc.kill()
                    proc.join(timeout=5)
                raise
            finally:
                if saved is None:
                    os.environ.pop("NEURON_RT_VISIBLE_CORES", None)
                else:
                    os.environ["NEURON_RT_VISIBLE_CORES"] = saved

def request_shutdown(channels: RoleChannels) -> None:
    """Ask every rank to exit after its current request. Best effort and idempotent: a queue
    that is already closed means its rank is gone."""
    for inbox in channels.inboxes:
        try:
            inbox.put((0, _SHUTDOWN, None))
        except Exception:  # noqa: BLE001 - a closed queue: the rank is gone
            pass


@contextlib.contextmanager
def _spawn_from_daemon():
    """Let a daemonic process (an SGLang diffusion worker) start the role workers."""
    import multiprocessing as mp

    config = mp.current_process()._config
    daemon = config.get("daemon")
    config["daemon"] = False
    try:
        yield
    finally:
        if daemon is not None:
            config["daemon"] = daemon


def _watch_parent(parent_pid: int) -> None:
    """Exit this role worker when its parent (DiT rank 0) is gone."""

    def watch() -> None:
        while True:
            time.sleep(2.0)
            if os.getppid() != parent_pid:
                os._exit(1)

    threading.Thread(target=watch, name="lingbot-role-parent-watch", daemon=True).start()


class RoleWorkersError(RuntimeError):
    """A role-worker request failed, timed out, or the world did not start."""


@dataclass
class _ConditionStream:
    seq: int
    latents: dict[int, object] = field(default_factory=dict)
    fixed_chunk: int | None = None
    #: Per encoded chunk: the encoder's own time (s).
    encode_seconds: list[float] = field(default_factory=list)

    def has(self, chunk: int) -> bool:
        """Whether ``chunk``'s condition is known (encoded, or at / after the fixed point)."""
        return chunk in self.latents or (self.fixed_chunk is not None and chunk >= self.fixed_chunk)

    def get(self, chunk: int):
        if self.fixed_chunk is not None and chunk >= self.fixed_chunk:
            return self.latents[self.fixed_chunk]
        return self.latents[chunk]


class LingBotRoleWorkers:
    """DiT rank 0's :class:`.aux_backend.LingBotAuxBackend`: the ENCODE / DECODE role instances
    in the role-worker world (:func:`role_world`), through the queues it was started
    with (:class:`RoleChannels`).

    Requests run in submission order on every rank; an abandoned request stops at its next
    chunk and whatever it still sends is dropped. One session at a time: the pipeline's KV
    spec admits one, so the runner releases a session before it opens the next.
    """

    def __init__(
        self,
        world: RoleWorld,
        channels: RoleChannels,
        *,
        start_timeout: float = 4 * 3600,
        request_timeout: float = 3600,
        processes=(),
    ) -> None:
        self.world = world
        # The ranks' processes (``RoleWorkerProcesses.processes``): a rank that exits without
        # reporting (killed, a runtime abort) fails the world instead of a request timing out.
        self._processes = list(processes)
        self.start_timeout = start_timeout
        self.request_timeout = request_timeout
        self._seq = 0
        self._results: dict[int, tuple[str, object]] = {}
        self._partials: dict[int, list] = {}
        self._abandoned: set[int] = set()
        self._conditions: dict[str, _ConditionStream] = {}
        self._prompts: dict[str, int] = {}
        self._decodes: dict[tuple[str, int], int] = {}
        self._vae_sessions: set[str] = set()  # sessions with decode state in the world
        # The roles' own times (ms) not yet reported (role_timings).
        self._prompt_ms: dict[str, float] = {}
        self._decode_ms: dict[tuple[str, int], float] = {}
        # Readiness (load + compile + warm-up, possibly long) is awaited on first use, so DiT
        # rank 0 does not hold up the DiT's own load and warm-up collectives meanwhile.
        self._started = time.monotonic()
        self._num_ready = 0
        self._ready = False
        self._inboxes = channels.inboxes
        self._outbox = channels.outbox
        # Highest request seq the parent abandoned (checked between encode chunks).
        self._cancelled = channels.cancelled

    # -- the world -----------------------------------------------------------------------

    def poll_ready(self, timeout: float) -> bool:
        """True once every rank has loaded, compiled and warmed up (waits up to ``timeout``)."""
        if self._ready:
            return True
        deadline = time.monotonic() + timeout
        while self._num_ready < self.world.world_size:
            if time.monotonic() - self._started > self.start_timeout:
                self._fail("LingBot role workers did not start in time")
            try:
                kind, _, rank, payload = self._outbox.get(
                    timeout=max(0.01, min(5.0, deadline - time.monotonic()))
                )
            except queue.Empty:
                self._check_alive()
                if time.monotonic() >= deadline:
                    return False
                continue
            if kind == "error":
                self.shutdown()
                raise RoleWorkersError(f"role workers rank {rank} failed to start:\n{payload}")
            self._num_ready += kind == "ready"
        self._ready = True
        logger.info("LingBot role workers ready on cores %s", list(self.world.cores))
        return True

    def _submit(self, cmd: str, payload) -> int:
        while not self.poll_ready(60.0):
            pass
        self._seq += 1
        for inbox in self._inboxes:
            inbox.put((self._seq, cmd, payload))
        return self._seq

    def _store(self, kind, seq, rank, payload) -> None:
        if kind == "error":
            # A failing rank exits after reporting, so the world is broken whichever request
            # failed (an abandoned one included): fail now, not at the next request's timeout.
            self._fail(f"LingBot role workers rank {rank}: request {seq} failed:\n{payload}")
        if seq in self._abandoned:
            if kind == "ok":
                self._abandoned.discard(seq)  # its last message
            return
        if kind == "partial":
            self._partials.setdefault(seq, []).append(payload)
        elif kind == "ok":
            self._results.setdefault(seq, ("ok", payload))

    def _fail(self, message: str) -> None:
        """Stop the role world and raise: a failed or stalled world cannot be resynced, and its
        rank 0 fails its request (``fatal_on_output_rank``)."""
        self.shutdown()
        raise RoleWorkersError(message)

    def _check_alive(self) -> None:
        for rank, proc in enumerate(self._processes):
            if proc.exitcode is not None:
                self._fail(f"LingBot role workers rank {rank} exited with {proc.exitcode}")

    def _pump(self, deadline: float) -> None:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._fail("LingBot role workers timed out")
            try:
                self._store(*self._outbox.get(timeout=min(remaining, 5.0)))
                return
            except queue.Empty:
                self._check_alive()

    def _result(self, seq: int):
        deadline = time.monotonic() + self.request_timeout
        while seq not in self._results:
            self._pump(deadline)
        return self._results.pop(seq)[1]

    def _abandon(self, seq: int) -> None:
        """Stop request ``seq`` at its next chunk boundary and drop whatever it still sends."""
        with self._cancelled.get_lock():
            self._cancelled.value = max(self._cancelled.value, seq)
        self._abandoned.add(seq)
        self._partials.pop(seq, None)
        self._results.pop(seq, None)

    # -- ENCODE --------------------------------------------------------------------------

    def submit_encode(
        self, request_id: str, *, image=None, prompt_tokens=None, chunks: int | None = None
    ) -> None:
        if image is not None:
            seq = self._submit("encode_session", (request_id, image.float().cpu(), chunks))
            self._conditions[request_id] = _ConditionStream(seq)
        if prompt_tokens is not None:
            input_ids, attention_mask = prompt_tokens
            self._prompts[request_id] = self._submit(
                "encode_prompt", (request_id, input_ids, attention_mask)
            )

    def prompt_embeds(self, request_id: str):
        data = self._result(self._prompts.pop(request_id))
        self._prompt_ms[request_id] = _ms(data[P.CONDITIONING_METADATA])
        return data[P.PROMPT_EMBEDS]

    def condition(self, request_id: str, chunk: int):
        """Chunk ``chunk``'s condition latent (CPU fp32), waiting for it if necessary."""
        stream = self._conditions[request_id]
        self._drain(stream)
        deadline = time.monotonic() + self.request_timeout
        while not stream.has(chunk):
            while not self._partials.get(stream.seq):
                self._pump(deadline)
            self._record_condition(stream, self._partials[stream.seq].pop(0))
        return stream.get(chunk)

    def _drain(self, stream) -> None:
        """Take the stream's arrived chunks without blocking."""
        while True:
            try:
                self._store(*self._outbox.get_nowait())
            except queue.Empty:
                break
        for data in self._partials.pop(stream.seq, []):
            self._record_condition(stream, data)

    def _record_condition(self, stream, data) -> None:
        meta = data[P.CONDITIONING_METADATA]
        chunk = int(meta["chunk_index"])
        stream.latents[chunk] = data[P.IMAGE_CONDITION]
        stream.encode_seconds.append(float(meta.get("seconds", 0.0)))
        if meta.get("fixed"):
            stream.fixed_chunk = chunk

    # -- DECODE --------------------------------------------------------------------------

    def submit_latents(self, request_id: str, chunk: int, latents, *, last: bool = False) -> None:
        payload = {
            P.LATENTS: latents.cpu(),
            P.LATENTS_METADATA: P.chunk_metadata(request_id, chunk, last=last),
        }
        self._decodes[(request_id, chunk)] = self._submit("decode", payload)
        self._vae_sessions.add(request_id)

    def collect_frames(self, request_id: str, chunk: int):
        """The decoded ``(F, H, W, 3)`` uint8 frames of a submitted chunk (waits if needed)."""
        out = self._result(self._decodes.pop((request_id, chunk)))
        self._decode_ms[(request_id, chunk)] = _ms(out[P.VIDEO_METADATA])
        return out[P.VIDEO]

    def role_timings(self, request_id: str, chunk: int) -> dict:
        """The roles' own times (ms) for a session's chunk, each reported once: the prompt encode (the
        UMT5 call of the session's prompt, if it ran), the chunk's condition encode (until the
        fixed point) and its VAE decode (once collected)."""
        out = {}
        if request_id in self._prompt_ms:
            out["prompt_encode_ms"] = self._prompt_ms.pop(request_id)
        stream = self._conditions.get(request_id)
        if stream is not None and chunk < len(stream.encode_seconds):
            out["condition_encode_ms"] = round(stream.encode_seconds[chunk] * 1e3, 2)
        if (request_id, chunk) in self._decode_ms:
            out["vae_decode_ms"] = self._decode_ms.pop((request_id, chunk))
        return out

    # -- lifecycle -----------------------------------------------------------------------

    def release(self, request_id: str) -> None:
        """Drop a session's state; a role-worker failure on the way is raised after the cleanup
        (the world has been stopped)."""
        failure: RoleWorkersError | None = None
        try:
            for key in [k for k in self._decodes if k[0] == request_id]:
                try:
                    self._result(self._decodes.pop(key))  # its frames are not wanted
                except RoleWorkersError as error:
                    failure = failure or error
            # Only a session that decoded has decode state to drop. Never submit otherwise: a
            # submit waits for the world to be ready, and the runner resets the warm-up session
            # on DiT rank 0 while the role workers may still be compiling -- blocking rank 0
            # there while the other DiT ranks run the first DiT graph outlasts Gloo's timeout.
            if request_id in self._vae_sessions:
                self._vae_sessions.discard(request_id)
                # Nothing waits for the release: its reply is dropped (a failure still fails).
                self._abandoned.add(self._submit("release", request_id))
        finally:
            # The session is gone whatever failed above (a later session must not see it live).
            prompt = self._prompts.pop(request_id, None)
            if prompt is not None:
                self._abandon(prompt)
            self._prompt_ms.pop(request_id, None)
            for key in [k for k in self._decode_ms if k[0] == request_id]:
                del self._decode_ms[key]
            stream = self._conditions.pop(request_id, None)
            if stream is not None:
                # A short session ends before the encoder's fixed point: stop the encode at its
                # next chunk instead of waiting for the rest of the sequence.
                self._abandon(stream.seq)
        if failure is not None:
            raise failure

    def shutdown(self) -> None:
        """Stop the world."""
        request_shutdown(RoleChannels(self._inboxes, self._outbox, self._cancelled))
