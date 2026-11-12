# SPDX-License-Identifier: Apache-2.0
"""Lease server of the remote DRAM tier: one process per host that lends DRAM (in PD, a decode host).

It checks the host's memory floors, maps a region of ``--gib`` GiB in chunks, faults it in on the EFA NUMA node in
huge pages, registers each chunk with its own NIXL agent, and gives each scheduler rank that asks a lease: a byte
range of one chunk and the agent metadata. The ranks then READ and WRITE their leases with NIXL; the server copies
nothing. It imports no SGLang.

Exposure and access (the server is a network listener handing out access to host memory):

- It binds one IP address, 127.0.0.1 by default, and refuses a wildcard address or a host name.
- It answers only the peers in ``--allow-ips``, checked before it reads a byte; an empty list refuses every peer.
  Each connection gets a 10 s read timeout.
- Off loopback it needs ``--auth sts``, ``--tls-dir`` and ``--nixl-backend LIBFABRIC`` (a UCX agent opens TCP
  listeners; this agent has no listen thread). With ``--auth sts`` a request carries the caller's presigned
  sts:GetCallerIdentity call; the server forwards it unchanged to the pinned regional STS endpoint and leases only to
  an ARN of ``--allow-arns`` (exact match). A token must sign the lease key and this server's ip:port, is accepted
  once, and only inside STS's 15-minute window. Any STS error refuses.
- With ``--tls-dir`` the lease channel is TLS 1.3 with a CA made for this run: the CA key never leaves memory, the
  server key file exists only until the server loads it, and the clients trust only ``ca.pem``.
- It exits after ``--max-lifetime-s`` (4 h by default), and frees its region under the memory floors: a guard every
  5 s releases tail chunks under ``--shrink-floor-gib`` (their leases are revoked) and stops the server under
  ``--stop-floor-gib``. The ranks see failed transfers, which become recomputes.
- Known limit: a lease is not isolated on the data plane. A leaseholder's agent can address the whole registered
  chunk its lease lies in, so the allow lists are the trust boundary: allow only the prefill hosts of one deployment,
  on a private network.

Example (loopback, a functional check):
    python -m sglang_neuron.mem_cache.remote_dram.server --gib 8 --host 127.0.0.1 --port 5590 --allow-ips 127.0.0.1
"""

from __future__ import annotations

import argparse
import base64
import calendar
import contextlib
import hashlib
import ipaddress
import json
import logging
import os
import re
import signal
import socket
import socketserver
import ssl
import stat
import sys
import threading
import time
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

from sglang_neuron.mem_cache.remote_dram import fill as fill_mod
from sglang_neuron.mem_cache.remote_dram.lease import (DEFAULT_STS_REGION, GIB, LEASE_HEADER, MAX_LINE, NONCE_HEADER,
                                                       SERVER_HEADER, SLOT_ALIGN, STS_BODY, sts_url)

logger = logging.getLogger("remote_dram_server")

ALLOW_IPS_ENV = "REMOTE_DRAM_ALLOW_IPS"
ALLOW_ARNS_ENV = "REMOTE_DRAM_ALLOW_ARNS"
#: Each read of a peer (the TLS handshake, the request line) waits at most this long.
READ_TIMEOUT_S = 10.0
GUARD_PERIOD_S = 5.0
#: Microseconds between two iterations of the NIXL progress thread (as the serving ranks' agents).
NIXL_PROGRESS_DELAY_US = 250


# -- address checks ---------------------------------------------------------------------------------------------------

def check_bind(host: str) -> str:
    """Why ``host`` is not one specific IP address to bind, or ""."""
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return f"--host {host!r} is not an IP address: give the one address to bind"
    if address.is_unspecified:
        return f"--host {host} binds every interface: give the one address to bind"
    return ""


def parse_allow(raw: Optional[str]) -> FrozenSet[str]:
    """The peer addresses of a comma list, normalized; empty means no peer.

    >>> sorted(parse_allow("10.0.0.2, 10.0.0.1"))
    ['10.0.0.1', '10.0.0.2']
    """
    return frozenset(str(ipaddress.ip_address(item.strip())) for item in (raw or "").split(",") if item.strip())


def parse_arns(raw: Optional[str]) -> FrozenSet[str]:
    return frozenset(item.strip() for item in (raw or "").split(",") if item.strip())


def lease_key(owner: str, key: str) -> str:
    """The region's key of a lease: its owner (the caller's ARN, else its address) and the rank's key."""
    return f"{owner}|{key}"


def check_policy(host: str, auth: str, tls_dir: str, nixl_backend: str) -> str:
    """Why this exposure is not allowed, or "": off loopback the server needs STS auth, TLS and LIBFABRIC."""
    error = check_bind(host)
    if error or ipaddress.ip_address(host).is_loopback:
        return error
    if auth != "sts":
        return f"--host {host} is off loopback: pass --auth sts and --allow-arns"
    if not tls_dir:
        return f"--host {host} is off loopback: pass --tls-dir"
    if nixl_backend != "LIBFABRIC":
        return f"--host {host} is off loopback: pass --nixl-backend LIBFABRIC (a UCX agent opens TCP listeners)"
    return ""


# -- the region -------------------------------------------------------------------------------------------------------

class NixlRegistrar:
    """Registers chunks with one NIXL agent that has no listen thread (the ranks get its metadata in the lease)."""

    def __init__(self, name: str, backend: str, progress_delay_us: int = NIXL_PROGRESS_DELAY_US):
        from nixl._api import nixl_agent, nixl_agent_config, nixlBind

        self.backend = backend
        agent_cls = nixlBind.nixlAgent

        def with_delay(agent_name, config):
            config.pthrDelay = progress_delay_us
            return agent_cls(agent_name, config)

        nixlBind.nixlAgent = with_delay
        try:
            self.agent = nixl_agent(name, nixl_agent_config(backends=[backend], enable_listen_thread=False))
        finally:
            nixlBind.nixlAgent = agent_cls

    def register(self, addr: int, nbytes: int):
        reg = self.agent.get_reg_descs([(addr, nbytes, 0, "")], "DRAM")
        self.agent.register_memory(reg, backends=[self.backend])
        return reg

    def deregister(self, handle) -> None:
        self.agent.deregister_memory(handle)

    def metadata(self) -> bytes:
        return self.agent.get_agent_metadata()


class Region:
    """The lent region: chunks registered one by one, and their leases (a bump allocator per chunk).

    The same key gets the same lease again; a known key asking another size is refused. A lease lies in one chunk,
    since a NIXL descriptor must lie in one registration. `release_tail` frees chunks from the end and revokes their
    leases: a revoked key is refused afterwards.
    """

    def __init__(self, chunks: Sequence[fill_mod.Chunk], registrar):
        self.chunks: List[Optional[fill_mod.Chunk]] = list(chunks)
        self.registrar = registrar
        self._regs = []
        try:
            for chunk in self.chunks:
                self._regs.append(registrar.register(chunk.addr, chunk.nbytes))
        except Exception:
            for reg in self._regs:
                with contextlib.suppress(Exception):
                    registrar.deregister(reg)
            raise
        self.metadata = base64.b64encode(registrar.metadata()).decode()
        self.nbytes = sum(c.nbytes for c in chunks)
        self._leases: Dict[str, Tuple[int, int, int]] = {}  # key -> (chunk, offset, nbytes)
        self._cursor = (0, 0)  # (chunk, next free offset in it)
        self._live = len(self.chunks)  # chunks [0, live) are held
        self._revoked: set = set()
        self._lock = threading.Lock()

    def lease(self, key: str, nbytes: int) -> dict:
        with self._lock:
            if key in self._revoked:
                return {"ok": False, "error": f"lease {key} was released: the host went under its memory floor"}
            known = self._leases.get(key)
            if known is not None:
                if known[2] != nbytes:
                    return {"ok": False, "error": f"key {key} holds {known[2]} B, not {nbytes} B"}
                index, offset = known[0], known[1]
            else:
                if nbytes <= 0:
                    return {"ok": False, "error": f"{nbytes} B is not a lease size"}
                index, offset = self._cursor
                while index < self._live and offset + nbytes > self.chunks[index].nbytes:
                    index, offset = index + 1, 0
                if index >= self._live:
                    return {"ok": False, "error": f"no room for {nbytes} B"}
                aligned = offset + (nbytes + SLOT_ALIGN - 1) // SLOT_ALIGN * SLOT_ALIGN
                self._cursor = (index, aligned)
                self._leases[key] = (index, offset, nbytes)
                logger.info("lease %r: %d B in chunk %d at offset %d", key, nbytes, index, offset)
            addr = self.chunks[index].addr + offset
        return {"ok": True, "addr": addr, "nbytes": nbytes, "dev": 0, "metadata": self.metadata}

    def stats(self) -> dict:
        with self._lock:
            return {"leases": len(self._leases), "chunks": self._live, "revoked": len(self._revoked),
                    "held_bytes": sum(c.nbytes for c in self.chunks[:self._live] if c is not None)}

    def release_tail(self, nbytes: int) -> int:
        """Deregister and unmap chunks from the end until ``nbytes`` are freed; revoke their leases. Returns the bytes
        freed. A chunk that fails to deregister stays mapped (its pages are pinned, and a peer can still write it)."""
        freed = 0
        with self._lock:
            while freed < nbytes and self._live > 0:
                index = self._live - 1
                chunk = self.chunks[index]
                try:
                    self.registrar.deregister(self._regs[index])
                except Exception:  # noqa: BLE001 - keep the chunk; it cannot be freed safely
                    logger.exception("deregister of chunk %d failed; it stays mapped", index)
                    break
                self._regs[index] = None
                chunk.close()
                self.chunks[index] = None
                self._live = index
                gone = [k for k, (i, _, _) in self._leases.items() if i >= index]
                for key in gone:
                    del self._leases[key]
                    self._revoked.add(key)
                if self._cursor[0] >= index:
                    self._cursor = (index, 0)
                freed += chunk.nbytes
                logger.warning("memory guard: released chunk %d (%.1f GiB), revoked %d leases", index,
                               chunk.nbytes / GIB, len(gone))
        return freed

    def close(self) -> None:
        with self._lock:
            for index in range(self._live):
                if self._regs[index] is not None:
                    with contextlib.suppress(Exception):
                        self.registrar.deregister(self._regs[index])
                    self._regs[index] = None
                if self.chunks[index] is not None:
                    self.chunks[index].close()
                    self.chunks[index] = None
            self._live = 0


class MemoryGuard:
    """Keeps the host above its floors: under ``stop_floor`` MemAvailable it stops the server; under ``shrink_floor``
    it releases tail chunks until MemAvailable would be back over the floor plus ``headroom``."""

    def __init__(self, region: Region, shrink_floor: int, stop_floor: int, on_stop: Callable[[str], None],
                 headroom: int = 0):
        self.region = region
        self.shrink_floor = shrink_floor
        self.stop_floor = stop_floor
        self.on_stop = on_stop
        self.headroom = headroom
        self.released = 0
        self.stopped = False

    def tick(self, info: Dict[str, int]) -> str:
        """One check: "ok", "shrink" or "stop"."""
        if self.stopped:
            return "stop"
        avail = info["MemAvailable"]
        if avail < self.stop_floor:
            self.stopped = True
            logger.error("memory guard: MemAvailable %.1f GiB under the stop floor %.1f GiB: stopping", avail / GIB,
                         self.stop_floor / GIB)
            self.on_stop("MemAvailable under the stop floor")
            return "stop"
        if avail < self.shrink_floor:
            self.released += self.region.release_tail(self.shrink_floor + self.headroom - avail)
            return "shrink"
        return "ok"


# -- the STS check ----------------------------------------------------------------------------------------------------

#: The one Authorization form botocore's SigV4 signer writes for an STS call. STS reads the last Signature of a looser
#: text, so anything else is refused: the signature the replay memory keeps is then the one STS checks.
SIGV4_AUTH = re.compile(
    r"AWS4-HMAC-SHA256 Credential=(?P<scope>[A-Z0-9]+/[0-9]{8}/[a-z0-9-]+/sts/aws4_request), "
    r"SignedHeaders=(?P<signed>[a-z0-9-]+(?:;[a-z0-9-]+)*), Signature=(?P<sig>[0-9a-f]{64})")


def sts_caller(url: str, headers: Dict[str, str], timeout: float) -> Optional[str]:
    """POST the presigned call to STS and return the caller ARN; None for any failure."""
    import urllib.request
    import xml.etree.ElementTree as ET

    request = urllib.request.Request(url, data=STS_BODY.encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as reply:  # noqa: S310 - the pinned https endpoint
            if reply.status != 200:
                return None
            body = reply.read(MAX_LINE)
        node = ET.fromstring(body).find(".//{https://sts.amazonaws.com/doc/2011-06-15/}Arn")
        return node.text.strip() if node is not None and node.text else None
    except Exception:  # noqa: BLE001 - any STS failure refuses the lease
        return None


class StsVerifier:
    """Checks a lease token (a presigned sts:GetCallerIdentity request) and returns the caller ARN.

    Refused before STS is called: a token that is not that call to the pinned endpoint, carries other headers or
    another Authorization form, is signed for another region, does not sign the lease key and this server, was seen
    before, or is outside STS's time window. Then STS must answer 200 with an ARN of ``allow_arns``.
    """

    HEADERS = frozenset({"authorization", "x-amz-date", "x-amz-security-token", "content-type",
                         LEASE_HEADER.lower(), NONCE_HEADER.lower(), SERVER_HEADER.lower()})
    #: STS takes a SigV4 request up to 15 min from its X-Amz-Date; one minute of margin.
    WINDOW_S = 16 * 60.0
    MAX_SEEN = 100_000

    def __init__(self, allow_arns: FrozenSet[str], region: str = DEFAULT_STS_REGION, server: str = "",
                 timeout: float = 5.0, caller_fn=sts_caller, clock=time.time):
        self.allow_arns = frozenset(allow_arns)
        self.region = region
        self.url = sts_url(region)
        self.server = server  # this server's ip:port; empty refuses every token
        self.timeout = timeout
        self.caller_fn = caller_fn
        self.clock = clock
        self._seen: Dict[bytes, float] = {}
        self._lock = threading.Lock()
        self._purged = 0.0

    def _first_use(self, sig: str, amz_date: str) -> str:
        try:
            signed = calendar.timegm(time.strptime(amz_date, "%Y%m%dT%H%M%SZ"))
        except ValueError:
            return "the token has no valid X-Amz-Date"
        now = self.clock()
        if abs(now - signed) > self.WINDOW_S:
            return "the token is outside the STS time window"
        digest = hashlib.sha256(sig.encode()).digest()
        with self._lock:
            if now - self._purged > 10.0:
                self._seen = {d: t for d, t in self._seen.items() if t > now}
                self._purged = now
            if digest in self._seen:
                return "the token was used before"
            if len(self._seen) >= self.MAX_SEEN:
                return "the replay memory is full"
            self._seen[digest] = max(now, signed) + self.WINDOW_S
        return ""

    def verify(self, token, key: str) -> Tuple[Optional[str], str]:
        """``(arn, "")`` for an accepted token, else ``(arn or None, reason)``."""
        if not isinstance(token, dict):
            return None, "no lease token"
        if token.get("url") != self.url or token.get("body") != STS_BODY:
            return None, "not a GetCallerIdentity call to the pinned STS endpoint"
        headers = token.get("headers")
        if not isinstance(headers, dict) or not all(isinstance(k, str) and isinstance(v, str)
                                                    for k, v in headers.items()):
            return None, "bad token headers"
        lower = {k.lower(): v for k, v in headers.items()}
        if len(lower) != len(headers):
            return None, "a token header is given twice"
        if set(lower) - self.HEADERS:
            return None, f"unexpected token headers {sorted(set(lower) - self.HEADERS)}"
        auth = SIGV4_AUTH.fullmatch(lower.get("authorization", ""))
        if auth is None:
            return None, "the Authorization header is not the SigV4 signer's form"
        signed = auth["signed"].split(";")
        if auth["scope"].split("/")[2] != self.region:
            return None, f"the token is not signed for STS in {self.region}"
        if LEASE_HEADER.lower() not in signed or lower.get(LEASE_HEADER.lower()) != key:
            return None, "the token does not sign this lease key"
        if not self.server or SERVER_HEADER.lower() not in signed or lower.get(SERVER_HEADER.lower()) != self.server:
            return None, "the token does not sign this server"
        if NONCE_HEADER.lower() not in signed:
            return None, "the token does not sign a nonce"
        if not self.allow_arns:
            return None, "the ARN allow list is empty"
        reason = self._first_use(auth["sig"], lower.get("x-amz-date", ""))
        if reason:
            return None, reason
        arn = self.caller_fn(self.url, headers, self.timeout)
        if arn is None:
            return None, "STS refused the token or did not answer"
        if arn not in self.allow_arns:
            return arn, "the caller is not in the ARN allow list"
        return arn, ""


# -- TLS --------------------------------------------------------------------------------------------------------------

CA_FILE, CERT_FILE, KEY_FILE = "ca.pem", "server.pem", "server.key"


def issue_run_certs(host: str, lifetime_s: float) -> Tuple[bytes, bytes, bytes]:
    """A new CA and a certificate for IP ``host`` (ECDSA P-256): PEMs of the CA cert, the server cert and the server
    key. The CA key is dropped here, so no other certificate can come from this CA."""
    import datetime

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    now = datetime.datetime.now(datetime.timezone.utc)
    start, end = now - datetime.timedelta(minutes=5), now + datetime.timedelta(seconds=lifetime_s + 3600)

    def usage(**on):
        names = ("digital_signature content_commitment key_encipherment data_encipherment key_agreement "
                 "key_cert_sign crl_sign encipher_only decipher_only").split()
        return x509.KeyUsage(**{n: on.get(n, False) for n in names})

    def base(subject, issuer, public_key):
        return (x509.CertificateBuilder().subject_name(subject).issuer_name(issuer).public_key(public_key)
                .serial_number(x509.random_serial_number()).not_valid_before(start).not_valid_after(end)
                .add_extension(x509.SubjectKeyIdentifier.from_public_key(public_key), critical=False))

    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"remote DRAM run CA {os.urandom(8).hex()}")])
    ca = (base(ca_name, ca_name, ca_key.public_key())
          .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
          .add_extension(usage(key_cert_sign=True, crl_sign=True), critical=True)
          .sign(ca_key, hashes.SHA256()))
    key = ec.generate_private_key(ec.SECP256R1())
    cert = (base(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, host)]), ca_name, key.public_key())
            .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address(host))]), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(usage(digital_signature=True), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
            .sign(ca_key, hashes.SHA256()))
    del ca_key
    pem = serialization.Encoding.PEM
    return (ca.public_bytes(pem), cert.public_bytes(pem),
            key.private_bytes(pem, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))


def check_tls_dir(path: str) -> str:
    """Why ``path`` is not a directory only this user can use (mode 0700, own uid, not a link), or ""."""
    try:
        info = os.lstat(path)
    except OSError as exc:
        return f"--tls-dir {path}: {exc.strerror}"
    if not stat.S_ISDIR(info.st_mode):
        return f"--tls-dir {path} is not a directory"
    if info.st_uid != os.geteuid() or info.st_mode & 0o077:
        return f"--tls-dir {path} has mode {stat.S_IMODE(info.st_mode):04o}, owner {info.st_uid}: give a 0700 dir"
    return ""


def _write_new(path: str, data: bytes, mode: int) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)


def make_run_tls(tls_dir: str, host: str, lifetime_s: float) -> Tuple[ssl.SSLContext, str, str]:
    """The run's TLS 1.3 server context. Writes ``ca.pem`` (for the clients) and ``server.pem``; the key file is
    written 0600 and deleted once loaded. Returns the context, the CA path and the CA's SHA-256 fingerprint."""
    error = check_tls_dir(tls_dir)
    if error:
        raise ValueError(error)
    ca, cert, key = issue_run_certs(host, lifetime_s)
    paths = {name: os.path.join(tls_dir, name) for name in (CA_FILE, CERT_FILE, KEY_FILE)}
    for path in paths.values():
        with contextlib.suppress(FileNotFoundError):
            os.unlink(path)
    _write_new(paths[CA_FILE], ca, 0o644)
    _write_new(paths[CERT_FILE], cert, 0o644)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    try:
        _write_new(paths[KEY_FILE], key, 0o600)
        context.load_cert_chain(paths[CERT_FILE], paths[KEY_FILE])
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(paths[KEY_FILE])
    return context, paths[CA_FILE], hashlib.sha256(ssl.PEM_cert_to_DER_cert(ca.decode())).hexdigest()


# -- the listener -----------------------------------------------------------------------------------------------------

def make_server(region: Region, host: str, port: int, allow: FrozenSet[str], verifier: Optional[StsVerifier] = None,
                tls: Optional[ssl.SSLContext] = None) -> socketserver.ThreadingTCPServer:
    """A TCP server answering one JSON-line lease request per allowed connection.

    A peer outside ``allow`` is closed before any byte is read (and before a TLS byte); a failed TLS handshake or a
    refused token closes the connection with no reply.
    """
    error = check_bind(host)
    if error:
        raise ValueError(error)

    class Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            peer = str(ipaddress.ip_address(self.client_address[0]))
            try:
                request = json.loads(self.rfile.readline(MAX_LINE) or b"{}")
            except (ValueError, OSError):
                request = {}
            if not isinstance(request, dict):
                request = {}
            key = str(request.get("key", ""))
            owner = peer
            if verifier is not None:
                arn, reason = verifier.verify(request.get("auth"), key)
                if reason:
                    Server.refused += 1
                    logger.warning("refused lease %r from %s (caller %s): %s", key, peer, arn, reason)
                    return
                owner = arn
            if request.get("op") != "lease":
                reply = {"ok": False, "error": f"unknown op {request.get('op')!r}"}
            else:
                try:
                    reply = region.lease(lease_key(owner, key), int(request.get("nbytes", 0)))
                except (TypeError, ValueError) as exc:
                    reply = {"ok": False, "error": str(exc)}
            self.wfile.write(json.dumps(reply).encode() + b"\n")

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True
        refused = 0

        def verify_request(self, request, client_address) -> bool:
            if str(ipaddress.ip_address(client_address[0])) in allow:
                return True
            Server.refused += 1
            logger.warning("refused peer %s: not in the allow list", client_address[0])
            return False

        def finish_request(self, request, client_address) -> None:
            request.settimeout(READ_TIMEOUT_S)
            if tls is None:
                return super().finish_request(request, client_address)
            try:
                conn = tls.wrap_socket(request, server_side=True)
            except OSError as exc:  # ssl.SSLError is an OSError
                Server.refused += 1
                logger.warning("refused peer %s: TLS handshake failed (%s)", client_address[0], exc)
                return
            with conn:
                self.RequestHandlerClass(conn, client_address, self)

    Server.address_family = socket.AF_INET6 if ipaddress.ip_address(host).version == 6 else socket.AF_INET
    server = Server((host, port), Handler)
    if verifier is not None:
        verifier.server = f"{host}:{server.server_address[1]}"
    return server


# -- the CLI ----------------------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m sglang_neuron.mem_cache.remote_dram.server",
                                description="Lease host DRAM to the remote DRAM tier of Neuron HiCache.")
    p.add_argument("--gib", type=float, required=True, help="region size")
    p.add_argument("--chunk-gib", type=float, default=32.0, help="registration chunk; no lease crosses one")
    p.add_argument("--host", default="127.0.0.1", help="the one IP address to bind")
    p.add_argument("--port", type=int, default=5590)
    p.add_argument("--allow-ips", default=os.environ.get(ALLOW_IPS_ENV, ""), help=f"comma list (or ${ALLOW_IPS_ENV})")
    p.add_argument("--auth", choices=("none", "sts"), default="none")
    p.add_argument("--allow-arns", default=os.environ.get(ALLOW_ARNS_ENV, ""), help=f"comma list (or ${ALLOW_ARNS_ENV})")
    p.add_argument("--sts-region", default=DEFAULT_STS_REGION)
    p.add_argument("--tls-dir", default="", help="a 0700 directory for this run's ca.pem")
    p.add_argument("--nixl-backend", default="LIBFABRIC", choices=("LIBFABRIC", "UCX"))
    p.add_argument("--max-lifetime-s", type=float, default=4 * 3600.0)
    p.add_argument("--floor-avail-gib", type=float, default=150.0, help="MemAvailable left after the region")
    p.add_argument("--floor-free-gib", type=float, default=0.0, help="MemFree left after the region (0: off)")
    p.add_argument("--shrink-floor-gib", type=float, default=100.0)
    p.add_argument("--stop-floor-gib", type=float, default=50.0)
    p.add_argument("--numa-node", default="auto", help="'auto' (the EFA node), a node number, or 'none'")
    p.add_argument("--fill-threads", type=int, default=16)
    p.add_argument("--allow-small-pages", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None, registrar_factory=NixlRegistrar, ready: Optional[Callable] = None) -> int:
    """The CLI. ``registrar_factory(name, backend)`` and ``ready(server)`` are for tests."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    error = check_policy(args.host, args.auth, args.tls_dir, args.nixl_backend)
    if error:
        logger.error("%s", error)
        return 2
    allow = parse_allow(args.allow_ips)
    if not allow:
        logger.error("--allow-ips is empty: the server would refuse every peer")
        return 2
    arns = parse_arns(args.allow_arns)
    if args.auth == "sts" and not arns:
        logger.error("--auth sts needs --allow-arns")
        return 2
    nbytes = int(args.gib * GIB) // fill_mod.HUGE * fill_mod.HUGE
    chunk = int(args.chunk_gib * GIB) // fill_mod.HUGE * fill_mod.HUGE
    floor = fill_mod.check_floors(nbytes, int(args.floor_avail_gib * GIB), int(args.floor_free_gib * GIB),
                                  fill_mod.meminfo())
    if floor:
        logger.error("%s", floor)
        return 2
    node = fill_mod.efa_node() if args.numa_node == "auto" else (None if args.numa_node == "none"
                                                                 else int(args.numa_node))
    cpus = fill_mod.node_cpus(node) if node is not None else None
    chunks = [fill_mod.Chunk(hi - lo) for lo, hi in fill_mod.plan_ranges(nbytes, chunk)]
    try:
        fill_mod.fill(chunks, threads=args.fill_threads, cpus=cpus, node=node)
        fill_mod.check_huge(chunks, allow_small=args.allow_small_pages)
        t0 = time.monotonic()
        region = Region(chunks, registrar_factory(f"remote-dram-{args.host}-{args.port}", args.nixl_backend))
        logger.info("registered %.1f GiB in %d chunks in %.1f s", nbytes / GIB, len(chunks), time.monotonic() - t0)
    except Exception:
        for c in chunks:
            c.close()
        raise
    tls = None
    if args.tls_dir:
        tls, ca_path, fingerprint = make_run_tls(args.tls_dir, args.host, args.max_lifetime_s)
        logger.info("TLS: give the ranks %s (sha256 %s)", ca_path, fingerprint)
    verifier = StsVerifier(arns, region=args.sts_region) if args.auth == "sts" else None
    server = make_server(region, args.host, args.port, allow, verifier, tls)
    stop = threading.Event()
    reason = ["signal"]

    def _stop(why: str) -> None:
        reason[0] = why
        stop.set()

    previous = {sig: signal.signal(sig, lambda *_: _stop("signal")) for sig in (signal.SIGINT, signal.SIGTERM)}
    guard = MemoryGuard(region, int(args.shrink_floor_gib * GIB), int(args.stop_floor_gib * GIB), _stop)
    threading.Thread(target=server.serve_forever, name="remote-dram-lease", daemon=True).start()
    logger.info("serving %.1f GiB on %s:%d to %s (auth %s, tls %s)", nbytes / GIB, args.host,
                server.server_address[1], ",".join(sorted(allow)), args.auth, bool(tls))
    if ready is not None:
        ready(server)
    deadline = time.monotonic() + args.max_lifetime_s
    while not stop.wait(max(0.0, min(GUARD_PERIOD_S, deadline - time.monotonic()))):
        if time.monotonic() >= deadline:
            _stop("max lifetime")
            break
        guard.tick(fill_mod.meminfo())
    logger.warning("stopping: %s; %s", reason[0], json.dumps(region.stats()))
    server.shutdown()
    server.server_close()
    region.close()
    for sig, handler in previous.items():
        signal.signal(sig, handler)
    return 0


if __name__ == "__main__":
    sys.exit(main())
