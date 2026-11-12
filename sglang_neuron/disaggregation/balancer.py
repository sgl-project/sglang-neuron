# SPDX-License-Identifier: Apache-2.0
"""The PD load balancer: one endpoint in front of a disaggregated deployment's prefill and decode servers, the one
users run and the one the disaggregated e2e tests put in front of their servers.

Routing. Each request goes to the prefill DP replica with the fewest outstanding prefills (a prefill server dispatches
to replica ``bootstrap_room % dp_size``, so the balancer picks the replica by choosing the room) and to the decode
server with the fewest requests in flight. A client's ``routed_dp_rank`` names a decode DP rank over every decode
server in order, which the determinism check uses to reach every rank.

Both halves are posted at once (SGLang's push protocol), and whichever fails first is the request's answer, so neither
half waits out the other's bootstrap timeout. A streamed request does not wait for the prefill's answer: SGLang's chat
and completions endpoints send a streamed response's 200 only with its first chunk, and a PD prefill gives that chunk
only when it answers the request, which a pipelined prefill can do several chunk times after the KV landed. So the
balancer streams the decode's answer as soon as the decode answers 200, and it watches the prefill (its status, then
its stream) until the decode's first event: a prefill failure before that event ends the request with the prefill's
error. Once the decode streams an event, the KV has landed or the decode reports its own failure. After a stream that
ends well, the prefill's connection stays open until the prefill answers (at most the request's timeout), because a
closed connection aborts the prefill's request, and the prefill then keeps nothing of it in its prefix cache.

A client that leaves. A client that leaves a stream ends it as ``client_left``: Starlette cancels the stream, and the
balancer closes both halves' connections, so each server aborts its request. A client that leaves a non-streamed
request ends it in the same way. uvicorn does not cancel such an endpoint; it only gives ``http.disconnect`` to the
next ``receive()``. So the balancer watches for that message while the request runs (_unless_client_leaves).

Retries. A request that fails before the client got a byte runs again once, as a new pair with a new bootstrap room:
a 5xx of either half, a broken connection, a streamed prefill's failure before the decode's first chunk, or a decode
stream whose first event is an error. A 4xx, the request's own timeout, a request naming its own ``rid`` and anything
after the first byte are not retried. ``--no-retry`` runs every request once, so a
client sees every server failure. A connection the server never accepted is tried up to three times
in all; one it dropped after accepting is not, since the server may hold the request and its room would collide.

The balancer must never be a request's bottleneck: it parses a request's body once, serializes its halves with orjson
where available, and freezes its startup heap with a high young-generation threshold, so a collection never stalls
the streams in flight. The client's session headers go on to both halves unchanged, and its Authorization header to
every server the balancer asks on its behalf, so servers started with SGLang's ``--api-key`` check the client's own
key; the balancer checks none. ``--trace-file PATH`` appends one JSON line per run of each request.

    python -m sglang_neuron.disaggregation.balancer --prefill http://P:30000 --decode http://D:30000
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import gc
import ipaddress
import json
import logging
import random
from dataclasses import dataclass, field
from time import perf_counter, time_ns
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
from uuid import uuid4

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from sglang_neuron.disaggregation import sse

try:
    import orjson
except ImportError:  # SGLang depends on it, so the image has it
    orjson = None

ROOM_LIMIT = 2**63
SESSION_HEADERS = ("x-session-id", "x-correlation-id", "x-dynamo-session-id")
# the pauses before a half's second and third connection attempts (_post)
CONNECT_PAUSES_S = (0.05, 0.15)
GC_THRESHOLDS = (100_000, 50, 1000)
TRACE_FLUSH_S = 1.0
# the milestones of a trace line, each as ``<name>_ms`` after the request's receipt (Trace)
MILESTONES = ("parsed", "p_sent", "d_sent", "p_headers", "d_headers", "first_chunk", "done")

log = logging.getLogger("sglang_neuron.disaggregation.balancer")

if orjson is not None:
    _loads, _dumps = orjson.loads, orjson.dumps
else:
    _loads = json.loads

    def _dumps(value) -> bytes:
        return json.dumps(value, separators=(",", ":")).encode()


@dataclass
class Backends:
    prefill: List[str]
    bootstrap_port: int
    decode: List[str]
    timeout_s: float
    prefill_dp: List[int] = field(default_factory=list)
    decode_dp: List[int] = field(default_factory=list)
    outstanding: Dict[Tuple[int, int], int] = field(default_factory=dict)
    in_flight: Dict[int, int] = field(default_factory=dict)
    retry: bool = True           # False under --no-retry
    retries: int = 0
    _sizes: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    async def dp_sizes(self, headers: Dict[str, str]) -> None:
        """Every server's DP size, read once from /server_info with the client's ``headers`` (_authorization). A
        server's refusal is the request's answer, such as its 401 to a client without the server's --api-key, and a
        server that does not answer (not up yet, a wrong URL) is a 503 naming it."""
        async with self._sizes:
            if self.prefill_dp:
                return
            async with aiohttp.ClientSession(headers=headers) as session:
                sizes = []
                for url in self.prefill + self.decode:
                    status, body, _ = await _ask(session, "GET", f"{url}/server_info", status=503)
                    if status != 200:
                        text = body.decode(errors="replace")
                        raise HTTPException(status, f"a server's /server_info answered {status}: {text}")
                    sizes.append(max(1, len(_json(f"{url}/server_info", body, 503).get("internal_states") or [None])))
            self.prefill_dp, self.decode_dp = sizes[:len(self.prefill)], sizes[len(self.prefill):]

    def pick_prefill(self) -> Tuple[int, int, Tuple[int, int]]:
        """(prefill server, bootstrap room, target) for the least-loaded (server, replica) target."""
        targets = [(p, r) for p, dp in enumerate(self.prefill_dp) for r in range(dp)]
        low = min(self.outstanding.get(t, 0) for t in targets)
        target = random.choice([t for t in targets if self.outstanding.get(t, 0) == low])
        self.outstanding[target] = self.outstanding.get(target, 0) + 1
        server, replica = target
        dp = self.prefill_dp[server]
        room = random.randrange(0, ROOM_LIMIT - dp) // dp * dp + replica
        return server, room, target

    def release(self, target: Tuple[int, int]) -> None:
        self.outstanding[target] = max(0, self.outstanding.get(target, 0) - 1)

    def pick_decode(self, routed_dp_rank) -> Tuple[int, Optional[int]]:
        """(decode server, its DP rank or None) for a request: the server and rank a global ``routed_dp_rank`` names,
        else the server with the fewest requests in flight."""
        if routed_dp_rank is not None:
            total = sum(self.decode_dp)
            if not (isinstance(routed_dp_rank, int) and 0 <= routed_dp_rank < total):
                raise HTTPException(400, f"routed_dp_rank={routed_dp_rank!r}: the decode servers have DP ranks "
                                         f"0..{total - 1}")
            for server, dp in enumerate(self.decode_dp):
                if routed_dp_rank < dp:
                    break
                routed_dp_rank -= dp
            decode, rank = server, routed_dp_rank
        else:
            low = min(self.in_flight.get(d, 0) for d in range(len(self.decode)))
            decode = random.choice([d for d in range(len(self.decode)) if self.in_flight.get(d, 0) == low])
            rank = None
        self.in_flight[decode] = self.in_flight.get(decode, 0) + 1
        return decode, rank

    def finish(self, decode: int) -> None:
        self.in_flight[decode] = max(0, self.in_flight.get(decode, 0) - 1)

    def fork(self, request: dict, server: int, room: int, decode_rank: Optional[int]) -> Tuple[dict, dict]:
        """The prefill and decode halves of one request; ``decode_rank`` is the DP rank of its decode server, or None
        for the server to pick."""
        prefill_rank = room % self.prefill_dp[server]
        host = urlparse(self.prefill[server]).hostname
        common = dict(request, bootstrap_host=f"[{host}]" if ":" in host else host,
                      bootstrap_port=self.bootstrap_port, bootstrap_room=room, disagg_prefill_dp_rank=prefill_rank)
        prefill_half = dict(common, routed_dp_rank=prefill_rank)
        decode_half = dict(common)
        decode_half.pop("routed_dp_rank", None)
        if decode_rank is not None:
            decode_half["routed_dp_rank"] = decode_rank
        return prefill_half, decode_half


@dataclass
class _Failure:
    """A failure before the client got a byte of the answer: its source (a half, or the balancer for a request it
    refused), the HTTP status to answer with (502 for a broken connection), the error, and whether a second run of the
    request can mend it."""
    source: str
    status: int
    error: str
    retryable: bool

    @classmethod
    def of_exception(cls, source: str, error: BaseException) -> "_Failure":
        # aiohttp's errors are the connection's and worth a retry; a timeout (asyncio's, which aiohttp's subclasses)
        # means the request's own budget is spent, and any other exception is not the server's
        retryable = isinstance(error, aiohttp.ClientError) and not isinstance(error, asyncio.TimeoutError)
        return cls(source, 502, f"{type(error).__name__}: {error}", retryable)

    @classmethod
    def of_event(cls, source: str, error: str, status: Optional[int]) -> "_Failure":
        """The failure a stream's event reports, with the status it names (sse.stream_failure)."""
        return cls(source, status or 502, error, status is None or not 400 <= status < 500)

    def answer(self) -> dict:
        return {"error": self.error, "source": self.source}


@dataclass
class _Origin:
    """What the trace lines of one request share: its id, its receipt time (wall clock ns, and the perf_counter origin
    of the lines' ms offsets), its endpoint, and what its parsed body showed."""
    rid: str
    recv_ns: int
    t0: float
    endpoint: str
    parsed: Optional[float] = None
    session: Optional[str] = None
    size: int = 0
    stream: bool = False


class _Request:
    """A client request: its parsed body, the headers both halves carry, and its trace origin (None without
    --trace-file). It runs as one _Attempt, and as a second one after a failure that may_retry allows."""

    def __init__(self, raw: bytes, headers, endpoint: str, origin: Optional[_Origin]):
        session = {name: value for name in SESSION_HEADERS if (value := headers.get(name))}
        try:
            body = _loads(raw)
        except ValueError as error:
            raise HTTPException(400, f"the request body is not JSON: {error}") from None
        if origin is not None:
            origin.parsed, origin.size, origin.session = perf_counter(), len(raw), next(iter(session.values()), None)
            origin.stream = isinstance(body, dict) and bool(body.get("stream"))
        if not isinstance(body, dict):
            raise HTTPException(400, "the request body is not a JSON object")
        _single_prompt(body)
        self.body, self.endpoint, self.streamed = body, endpoint, bool(body.get("stream"))
        self.auth = _authorization(headers)
        self.headers = {"Content-Type": "application/json", **session, **self.auth}
        self.retried = False
        self.origin = origin

    def may_retry(self, failure: _Failure) -> bool:
        # a request that names its own rid runs once (see the module docstring)
        return failure.retryable and not self.retried and backends.retry and self.body.get("rid") is None

    async def retry(self, attempt: "_Attempt", failure: _Failure) -> "_Attempt":
        """Drop ``attempt``, which ``failure``d before the client got a byte, and run the request again."""
        self.retried = True
        backends.retries += 1
        log.warning("%s failed before the client got a byte (%s: %s); running it again", self.endpoint,
                    failure.source, failure.error[:300])
        await attempt.end("retried", failure)
        return _Attempt(self, retry_reason=failure)


class _Attempt:
    """One run of a request's two halves: the prefill target and decode server it counts against, its own client
    session (closing it drops whatever half still runs), and its trace milestones."""

    def __init__(self, request: _Request, retry_reason: Optional[_Failure] = None):
        self.request, self.retry_reason = request, retry_reason
        self.number = int(retry_reason is not None)
        self.decode, self.decode_rank = backends.pick_decode(request.body.get("routed_dp_rank"))
        self.released = self.finished = self.ended = False
        try:
            self.server, self.room, self.target = backends.pick_prefill()
        except BaseException:
            self.finish()
            raise
        try:
            self.marks: Optional[Dict[str, float]] = {} if request.origin is not None else None
            prefill_half, decode_half = backends.fork(request.body, self.server, self.room, self.decode_rank)
            prefill_body, decode_body = _dumps(prefill_half), _dumps(decode_half)
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=backends.timeout_s))
        except BaseException:
            self.release()
            self.finish()
            raise
        self.chunks = None
        self.p_task = asyncio.ensure_future(self._half("p", backends.prefill[self.server], prefill_body))
        self.d_task = asyncio.ensure_future(self._half("d", backends.decode[self.decode], decode_body))

    def mark(self, milestone: str) -> None:
        if self.marks is not None:
            self.marks[milestone] = perf_counter()

    def release(self) -> None:
        """Stop counting against the prefill target: its prefill is done, or the attempt is over."""
        if not self.released:
            self.released = True
            backends.release(self.target)

    def finish(self) -> None:
        if not self.finished:
            self.finished = True
            backends.finish(self.decode)

    async def _half(self, half: str, server: str, body: bytes) -> aiohttp.ClientResponse:
        """Post one half; a non-streamed decode's answer is read in full, so that reading it is part of the run."""
        self.mark(f"{half}_sent")
        response = await _post(self.session, f"{server}/{self.request.endpoint}", body, self.request.headers)
        self.mark(f"{half}_headers")
        if half == "d" and not self.request.streamed and response.status == 200:
            await response.read()
        return response

    async def accepted(self) -> Optional[_Failure]:
        """Wait until both halves answer 200, or for the first failure, which cancels the other half. A non-streamed
        prefill answers once its prefill is done, which releases its target. A streamed request waits for the decode's
        200 only: the prefill half may still run, and first_event watches it."""
        pending = {self.p_task, self.d_task}
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                source = "prefill" if task is self.p_task else "decode"
                error = task.exception()
                if error is None and task.result().status == 200:
                    if task is self.p_task and not self.request.streamed:
                        self.release()
                    continue
                for other in pending:
                    other.cancel()
                if error is not None:
                    return _Failure.of_exception(source, error)
                response = task.result()
                text = (await response.read()).decode(errors="replace")
                return _Failure(source, response.status, text, response.status >= 500)
            if self.request.streamed and self.d_task in done:
                return None
        return None

    async def _prefill_failure(self) -> Optional[_Failure]:
        """The failure the prefill half reports (a broken connection, a status that is not 200, or an error event in
        its stream), or None at the end of its stream without one. The prefill's task is shielded, so a cancelled
        watch leaves the prefill's request running."""
        try:
            response = await asyncio.shield(self.p_task)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - a transport error of the prefill half
            return _Failure.of_exception("prefill", error)
        if response.status != 200:
            text = (await response.read()).decode(errors="replace")
            return _Failure("prefill", response.status, text, response.status >= 500)
        return await _stream_failure(response, "prefill")

    async def first_event(self) -> Tuple[bytes, Optional[_Failure]]:
        """Read the decode stream through its first whole data line while watching the prefill's own stream. Returns
        the bytes read and any failure before the client may have them: the prefill's stream reporting one or breaking
        first, the decode's stream breaking, or the decode's first event being an error (its bytes are kept, and pass
        on unchanged once no retry is left)."""
        self.chunks = self.d_task.result().content.iter_any()
        first = asyncio.ensure_future(_first_event(self.chunks))
        watcher = asyncio.ensure_future(self._prefill_failure())
        try:
            await asyncio.wait({first, watcher}, return_when=asyncio.FIRST_COMPLETED)
            if not first.done():
                error = watcher.exception()
                failure = _Failure.of_exception("prefill", error) if error is not None else watcher.result()
                if failure is not None:
                    return b"", failure
            try:
                held, line = await first
            except Exception as error:
                return b"", _Failure.of_exception("decode", error)
            if line is None or line == sse.DONE:
                return held, None
            error, status = sse.stream_failure(line)
            return held, None if error is None else _Failure.of_event("decode", error, status)
        finally:
            _drop(first)
            _drop(watcher)

    async def end(self, status: str, failure: Optional[_Failure] = None) -> None:
        """Stop counting the attempt against its servers, queue its trace line, and close the halves' responses, then
        the session, which drops whatever half still runs (a server aborts a request whose connection closed). Runs
        once. A response read only in part, such as a prefill stream that outlasts the decode's first event, holds its
        connection until closed; the session's close shuts the transport, though aiohttp then warns of an unclosed
        connection when it is collected.

        Nothing after the close's await is sure to run: when a stream's client leaves, Starlette (through anyio, on
        uvicorn's h11 server) cancels the stream again at every await until it ends. aiohttp's session close shuts
        every transport before its first await, so the halves drop even when that await is cut."""
        if self.ended:
            return
        self.ended = True
        # a stream that ended well while the prefill had not answered yet: keep the prefill's connection open until it
        # answers, in a task of its own (started before the first await here), then close the session
        keep_prefill = self.request.streamed and status == "ok" and not _answered(self.p_task)
        for task in (self.d_task,) if keep_prefill else (self.p_task, self.d_task):
            _drop(task)
            if task.done() and not task.cancelled() and task.exception() is None:
                task.result().close()
        self.release()
        self.finish()
        if self.marks is not None:
            self.mark("done")
            trace.put(_trace_item(self.request.origin, status, failure, self))
        if keep_prefill:
            asyncio.ensure_future(self._close_after_prefill())
            return
        await self.session.close()

    async def _close_after_prefill(self) -> None:
        """Read the prefill's answer to its end (at most the request's timeout), then close the session."""
        try:
            response = await asyncio.wait_for(asyncio.shield(self.p_task), backends.timeout_s)
            await asyncio.wait_for(response.read(), backends.timeout_s)
        except BaseException:  # noqa: BLE001 - the prefill half ends here in any case
            pass
        finally:
            _drop(self.p_task)
            await self.session.close()


def _answered(task: asyncio.Future) -> bool:
    """Whether a half's task is over: cancelled, failed, or answered with a response whose body was read to its end."""
    if not task.done() or task.cancelled() or task.exception() is not None:
        return task.done()
    at_eof = getattr(getattr(task.result(), "content", None), "at_eof", None)
    return at_eof is None or bool(at_eof())


def _drop(task: asyncio.Future) -> None:
    """Cancel a task, or take a finished one's exception, which asyncio would otherwise log as never retrieved."""
    if not task.done():
        task.cancel()
    elif not task.cancelled():
        task.exception()


async def _post(session, url: str, body: bytes, headers: Dict[str, str]) -> aiohttp.ClientResponse:
    """POST one half of a request. A connection the server refused or never accepted is tried up to three times,
    CONNECT_PAUSES_S apart; any other failure goes to the caller."""
    for pause in CONNECT_PAUSES_S:
        try:
            return await session.post(url, data=body, headers=headers)
        except aiohttp.ClientConnectorError:
            await asyncio.sleep(pause)
    return await session.post(url, data=body, headers=headers)


async def _first_event(chunks) -> Tuple[bytes, Optional[bytes]]:
    """A stream's bytes through its first whole ``data:`` line, and that line (None when the stream ended before one).
    A network chunk may end part-way through a line, or hold several."""
    held = bytearray()
    start = scan = 0
    async for chunk in chunks:
        held += chunk
        while (end := held.find(b"\n", scan)) >= 0:
            line = bytes(held[start:end]).rstrip()
            if line.startswith(b"data:"):
                return bytes(held), line
            start = scan = end + 1
        scan = len(held)
    return bytes(held), None


async def _stream_failure(response, source: str) -> Optional[_Failure]:
    """The failure a server's event stream reports (sse.stream_failure), or None at its end without one."""
    async for line in response.content:
        line = line.rstrip()
        if not line.startswith(b"data:") or line == sse.DONE:
            continue
        error, status = sse.stream_failure(line)
        if error is not None:
            return _Failure.of_event(source, error, status)
    return None


@contextlib.asynccontextmanager
async def lifespan(_app):
    """At startup, collect and freeze the heap, raise the collector's thresholds and start the trace writer; at
    shutdown, write what is left of the trace."""
    gc.collect()
    gc.freeze()
    gc.set_threshold(*GC_THRESHOLDS)
    writer = asyncio.ensure_future(trace.run()) if trace is not None else None
    try:
        yield
    finally:
        if writer is not None:
            trace.close()
            await writer


app = FastAPI(lifespan=lifespan)
backends: Optional[Backends] = None
trace: Optional["Trace"] = None


def _single_prompt(request: dict) -> None:
    for key in ("text", "input_ids", "prompt"):
        value = request.get(key)
        if isinstance(value, list) and value and not isinstance(value[0], int):
            raise HTTPException(400, "the balancer routes single-prompt requests only")


class _NoAnswer(HTTPException):
    """A server the balancer asked for the client gave no usable answer: no connection, a time-out, or a body that is
    not the JSON it must be. The client gets ``status`` (502, or 503 while the servers' sizes are unknown) and the
    documented ``{"error", "source"}``, not an opaque 500."""

    def __init__(self, url: str, error: BaseException, status: int):
        super().__init__(status, f"{url}: {type(error).__name__}: {error}"[:500])
        self.url = url


async def _ask(session, method: str, url: str, status: int = 502, **kwargs) -> Tuple[int, bytes, str]:
    """(status, body, content type) of one request the balancer sends a server for the client: the server's own
    answer, its status included, or _NoAnswer when there is none."""
    try:
        async with session.request(method, url, **kwargs) as resp:
            return resp.status, await resp.read(), resp.headers.get("Content-Type", "application/json")
    except (aiohttp.ClientError, asyncio.TimeoutError) as error:
        raise _NoAnswer(url, error, status) from None


def _json(url: str, body: bytes, status: int = 502) -> dict:
    try:
        return json.loads(body)
    except ValueError as error:
        raise _NoAnswer(url, error, status) from None


def _authorization(headers) -> Dict[str, str]:
    """The client's Authorization header, for each request the balancer sends a server on the client's behalf. A server
    started with SGLang's --api-key refuses every request without its key but /health, and the balancer checks no key
    itself: it passes on the client's, unchanged, to its own servers only, and never logs or traces it."""
    value = headers.get("authorization")
    return {"Authorization": value} if value else {}


async def _forward(client: Request, endpoint: str):
    origin = _Origin(uuid4().hex, time_ns(), perf_counter(), endpoint) if trace is not None else None
    try:
        request = _Request(await client.body(), client.headers, endpoint, origin)
        await backends.dp_sizes(request.auth)
        attempt = _Attempt(request)
    except HTTPException as error:
        if origin is not None:
            refused = _Failure("balancer", error.status_code, str(error.detail), False)
            trace.put(_trace_item(origin, "rejected", refused))
        raise
    if request.streamed:
        return await _answer(request, attempt)
    return await _unless_client_leaves(client, _answer(request, attempt))


async def _answer(request: _Request, attempt: _Attempt):
    """The answer to a request whose first attempt runs: a failure, the stream, or the decode's whole body. A
    cancellation (the client left) ends the attempt as ``client_left``."""
    try:
        failure = await attempt.accepted()
        if failure is not None and request.may_retry(failure):
            attempt = await request.retry(attempt, failure)
            failure = await attempt.accepted()
        if failure is not None:
            await attempt.end("failed", failure)
            return JSONResponse(failure.answer(), status_code=failure.status)
        if request.streamed:
            return StreamingResponse(_stream(request, attempt), media_type="text/event-stream")
        response = attempt.d_task.result()
        body = await response.read()
        attempt.mark("first_chunk")
        await attempt.end("ok")
        return Response(body, status_code=response.status, media_type="application/json")
    except BaseException as error:
        if isinstance(error, asyncio.CancelledError):
            await attempt.end("client_left")
        else:
            await attempt.end("failed", _Failure.of_exception("balancer", error))
        raise


# The status of the answer to a non-streamed request whose client left (the "client closed request" status of
# nginx). No client reads it: uvicorn does not send a response on a closed connection.
CLIENT_LEFT_STATUS = 499


async def _unless_client_leaves(client: Request, answer) -> Response:
    """Run ``answer``, the _answer of a non-streamed request, until it returns or its client leaves. When the client
    leaves, cancel ``answer``: its attempt then ends as ``client_left`` and closes both halves' connections, so each
    server aborts its request. A cancellation of this endpoint cancels ``answer`` in the same way."""
    answering = asyncio.ensure_future(answer)
    watch = asyncio.ensure_future(_client_leaves(client))
    try:
        await asyncio.wait({answering, watch}, return_when=asyncio.FIRST_COMPLETED)
        if not answering.done() and watch.exception() is not None:
            await asyncio.wait({answering})   # the watch failed, not the client: wait for the answer alone
    finally:
        if not answering.done():
            answering.cancel()
            await asyncio.wait({answering})
        _drop(watch)
    if answering.cancelled():
        return Response(status_code=CLIENT_LEFT_STATUS)
    return answering.result()


async def _client_leaves(client: Request) -> None:
    """Return when the client closes its connection. The request's body is read already, so the next ASGI message
    is the ``http.disconnect`` that uvicorn gives when the connection closes."""
    while (await client.receive()).get("type") != "http.disconnect":
        pass


async def _stream(request: _Request, attempt: _Attempt):
    """A streamed answer, whose 200 goes out once both halves answered. Its first chunk is the decode's first whole
    event (first_event); a failure before that runs the request again when may_retry allows."""
    status, failure = "client_left", None
    try:
        held, failure = await attempt.first_event()
        if failure is not None and request.may_retry(failure):
            attempt = await request.retry(attempt, failure)
            held, failure = b"", await attempt.accepted()
            if failure is None:
                held, failure = await attempt.first_event()
        if failure is not None and not held:
            # nothing of the decode's to pass on: the failure ends the stream as an error event
            status = "failed"
            yield b"data: " + _dumps(failure.answer()) + b"\n\n"
            yield sse.DONE + b"\n\n"
            return
        if held:
            attempt.release()  # the decode streams the first token, so the prefill is done
            attempt.mark("first_chunk")
            yield held
            async for chunk in attempt.chunks:
                yield chunk
        status = "ok" if failure is None else "failed"
    except Exception as error:
        status, failure = "failed", failure or _Failure.of_exception("decode", error)
        raise
    finally:
        await attempt.end(status, failure)


class Trace:
    """--trace-file: one JSON line for each run of each request (two for a request run again), and one for a request
    the balancer refused (status ``rejected``). The request path only queues an item (put_nowait); run(), started with
    the app, writes what was queued as JSON lines at least every TRACE_FLUSH_S, off the event loop, and the rest at
    shutdown. A line holds:

    - ``rid``, ``attempt`` (0, or 1 for the second run), ``session`` (the first session header the client sent),
      ``recv_ns`` (the wall clock at receipt), ``endpoint``, ``stream`` and ``bytes`` (the client body's size);
    - ``<milestone>_ms`` for each of MILESTONES, after the receipt on perf_counter (null for one the run did not reach):
      the body parsed, each half posted (``p_sent``, ``d_sent``) and its response headers in (``p_headers``,
      ``d_headers``), the first chunk handed to the client (a non-streamed answer's whole body), and the run's end;
    - ``status``: ``ok`` (the decode's answer reached the client; a stream's events past the first are not read),
      ``retried``, ``failed``, ``client_left`` or ``rejected``; for a failure, ``error_source`` (prefill, decode or
      balancer), ``error`` and ``error_status`` (the status the half answered or its error event named; 502 for a
      broken connection or an event that names none); on a second run, ``retry_reason``, the first run's error;
    - ``prefill``, ``replica``, ``room``, ``decode`` and ``decode_rank``: the run's picks (null for a refused
      request)."""

    def __init__(self, path: str):
        self.file = open(path, "ab")  # opened here, so a bad path fails the start; run() closes it
        self.queue: asyncio.Queue = asyncio.Queue()
        self.closing = asyncio.Event()

    def put(self, item: dict) -> None:
        self.queue.put_nowait(item)

    def close(self) -> None:
        """Have run() write the rest and return."""
        self.closing.set()

    async def run(self) -> None:
        try:
            while not self.closing.is_set():
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self.closing.wait(), TRACE_FLUSH_S)
                items = self._take()
                if items:
                    await asyncio.to_thread(self._write, items)
        finally:
            self._write(self._take())
            self.file.close()

    def _take(self) -> list:
        return [self.queue.get_nowait() for _ in range(self.queue.qsize())]

    def _write(self, items: list) -> None:
        if items:
            self.file.write(b"".join(_dumps(_trace_line(item)) + b"\n" for item in items))
            self.file.flush()


def _trace_item(origin: _Origin, status: str, failure: Optional[_Failure], attempt: Optional[_Attempt] = None) -> dict:
    """What a run of a request queues for its trace line: references only, which the writer turns into the line."""
    return {"origin": origin, "status": status, "failure": failure, "attempt": attempt and attempt.number,
            "marks": attempt.marks if attempt else {}, "retry_reason": attempt and attempt.retry_reason,
            "picks": attempt and (backends.prefill[attempt.server], attempt.target[1], attempt.room,
                                  backends.decode[attempt.decode], attempt.decode_rank)}


def _trace_line(item: dict) -> dict:
    origin, failure, reason = item["origin"], item["failure"], item["retry_reason"]
    marks = dict(item["marks"], parsed=origin.parsed)
    line = {"rid": origin.rid, "attempt": item["attempt"], "session": origin.session, "recv_ns": origin.recv_ns,
            "endpoint": origin.endpoint, "stream": origin.stream, "bytes": origin.size}
    for milestone in MILESTONES:
        at = marks.get(milestone)
        line[f"{milestone}_ms"] = None if at is None else round((at - origin.t0) * 1e3, 3)
    line.update(status=item["status"], error_source=failure and failure.source,
                error=failure and failure.error[:300], error_status=failure and failure.status,
                retry_reason=reason and reason.error[:300])
    line.update(zip(("prefill", "replica", "room", "decode", "decode_rank"), item["picks"] or (None,) * 5))
    return line


@app.post("/generate")
async def generate(request: Request):
    return await _forward(request, "generate")


@app.post("/v1/completions")
async def completions(request: Request):
    return await _forward(request, "v1/completions")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _forward(request, "v1/chat/completions")


@app.get("/health")
async def health():
    return Response(status_code=200)


@app.exception_handler(_NoAnswer)
async def _no_answer(request: Request, error: _NoAnswer):
    log.warning("%s %s: no answer from %s", request.method, request.url.path, error.detail)
    return JSONResponse({"error": error.detail, "source": error.url}, status_code=error.status_code)


@app.get("/server_info")
async def server_info(request: Request):
    """Every server's /server_info; a server's refusal (its 401 without the key) is the answer."""
    infos = []
    async with aiohttp.ClientSession(headers=_authorization(request.headers)) as session:
        for url in backends.prefill + backends.decode:
            status, body, media = await _ask(session, "GET", f"{url}/server_info")
            if status != 200:
                return Response(body, status_code=status, media_type=media)
            infos.append(_json(f"{url}/server_info", body))
    decode = infos[len(backends.prefill):]
    return {"internal_states": [s for info in decode for s in info.get("internal_states", [])],
            "prefill": infos[:len(backends.prefill)], "decode": decode, "balancer_retries": backends.retries}


@app.post("/flush_cache")
async def flush_cache(request: Request, timeout: float = 0.0):
    """Every server flushed with the client's own ``timeout``, if it gave one, except a pipelined server (pp_size above
    1 in its /server_info), which refuses a timed flush and is flushed untimed (workloads.flush)."""
    statuses = []
    async with aiohttp.ClientSession(headers=_authorization(request.headers)) as session:
        for url in backends.prefill + backends.decode:
            status, body, media = await _ask(session, "GET", f"{url}/server_info")
            if status != 200:
                return Response(body, status_code=status, media_type=media)
            pipelined = int(_json(f"{url}/server_info", body).get("pp_size") or 1) > 1
            params = {"timeout": str(timeout)} if timeout > 0 and not pipelined else {}
            statuses.append((await _ask(session, "POST", f"{url}/flush_cache", params=params))[0])
    return Response(status_code=200 if all(s == 200 for s in statuses) else 500)


@app.get("/v1/models")
async def models(request: Request):
    """The first prefill server's answer, its status included: a 503 or 404 there is no 200 here."""
    return await _first_prefill(request, "v1/models")


@app.get("/model_info")
async def model_info(request: Request):
    return await _first_prefill(request, "model_info")


@app.get("/get_model_info")
async def get_model_info(request: Request):
    """/model_info's older name, which SGLang's frontend-language client still asks first (RuntimeEndpoint: the GSM8K
    client sglang.test.few_shot_gsm8k, for one); the servers answer both."""
    return await _first_prefill(request, "get_model_info")


async def _first_prefill(request: Request, path: str) -> Response:
    async with aiohttp.ClientSession(headers=_authorization(request.headers)) as session:
        status, body, media = await _ask(session, "GET", f"{backends.prefill[0]}/{path}")
    return Response(body, status_code=status, media_type=media)


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:  # a host name
        return False


def _server_url(url: str) -> str:
    """A server URL as the balancer joins paths to it (``f"{url}/server_info"``): http or https, a host, no path but
    ``/``, which is dropped. ``host:30000`` or ``http://host:30000/v1`` would fail every request, not the start."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.path not in ("", "/") \
            or parsed.query or parsed.fragment:
        raise argparse.ArgumentTypeError(f"{url!r} is no server URL: give http://<host>:<port>, such as "
                                         "http://10.0.0.1:30000")
    return url.rstrip("/")


def main(argv=None) -> None:
    global backends, trace
    # each default named by %(default)s where it means something: ArgumentDefaultsHelpFormatter would also print
    # "(default: None)" after the required URLs and --trace-file, and "(default: False)" after --no-retry
    parser = argparse.ArgumentParser(description=(
        "The PD load balancer: one endpoint in front of a disaggregated deployment's prefill and decode servers, which "
        "sends each request to a prefill server and a decode server at once."))
    parser.add_argument("--prefill", action="append", required=True, type=_server_url,
                        help="prefill server URL (repeatable)")
    parser.add_argument("--bootstrap-port", type=int, default=8998,
                        help="the prefill servers' bootstrap port (default: %(default)s)")
    parser.add_argument("--decode", action="append", required=True, type=_server_url,
                        help="decode server URL (repeatable)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="the address to listen on (default: %(default)s, this host only; give an address for "
                             "clients on other hosts)")
    parser.add_argument("--port", type=int, default=8000, help="the port to listen on (default: %(default)s)")
    parser.add_argument("--timeout-s", type=float, default=3600.0,
                        help="the time limit of each half of a request, in seconds (default: %(default)s)")
    parser.add_argument("--trace-file", help="append one JSON line for each run of each request here (Trace)")
    parser.add_argument("--no-retry", action="store_true",
                        help="run each request once, so a server failure reaches the client (the e2e tests' setting)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # the decode halves bootstrap from the --prefill URL's host (fork): a loopback one is the decode's own host
    loopback = [url for url in args.prefill if _is_loopback(urlparse(url).hostname)]
    if loopback and not all(_is_loopback(urlparse(url).hostname) for url in args.decode):
        log.warning("--prefill %s is a loopback address, which the decode servers are told to bootstrap from: a decode "
                    "server on another host fails at its bootstrap timeout; give the prefill server's own address",
                    ", ".join(loopback))
    backends = Backends(args.prefill, args.bootstrap_port, args.decode, args.timeout_s, retry=not args.no_retry)
    trace = Trace(args.trace_file) if args.trace_file else None
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
