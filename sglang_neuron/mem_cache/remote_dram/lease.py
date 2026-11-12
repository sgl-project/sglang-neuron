# SPDX-License-Identifier: Apache-2.0
"""The lease channel of the remote DRAM tier: the request a scheduler rank sends a lease server, and the lease it gets.

One JSON line each way over TCP, TLS 1.3 off loopback. A request is ``{"op": "lease", "key": ..., "nbytes": ...}``,
with ``"auth"``: a presigned sts:GetCallerIdentity request (SigV4) when the server checks callers. A lease is a byte
range of the server's NIXL-registered region and the agent metadata to reach it: ``{"ok": true, "addr", "nbytes",
"dev", "metadata"}`` (metadata base64). A refused request gets no reply at all (the server closes the connection), or
``{"ok": false, "error": ...}`` for a request it read and could not serve.

The STS token: the caller signs a GetCallerIdentity call for one pinned regional endpoint, with the lease key, a
nonce and the server's ``ip:port`` in signed headers. The server forwards it unchanged to STS and reads back the
caller's ARN, so the caller's secret key never leaves its process and a token is good for one lease on one server.

Standard library only, with botocore for a token; the lease server imports this module without SGLang.
"""

from __future__ import annotations

import base64
import ipaddress
import json
import os
import socket
import ssl
from dataclasses import dataclass
from typing import Tuple

GIB = 1 << 30
#: Leases and slots start on a 4 KiB boundary.
SLOT_ALIGN = 4096
#: The longest request or reply line either side reads.
MAX_LINE = 1 << 16
DEFAULT_STS_REGION = "us-east-2"
STS_BODY = "Action=GetCallerIdentity&Version=2011-06-15"
#: Signed headers of a token: the lease key, a nonce (two tokens of one caller and key differ, so the server accepts
#: each once), and the server's ip:port (another server refuses the token).
LEASE_HEADER = "X-Remote-Dram-Lease"
NONCE_HEADER = "X-Remote-Dram-Nonce"
SERVER_HEADER = "X-Remote-Dram-Server"
AUTH_MODES = ("none", "sts")


def sts_url(region: str) -> str:
    return f"https://sts.{region}.amazonaws.com/"


class LeaseError(RuntimeError):
    """The server refused the lease, gave no reply, or could not be reached; the rank stores nothing remotely."""


def split_server(server: str) -> Tuple[str, int]:
    """``(ip, port)`` of ``ip:port`` (``[ip]:port`` for IPv6); the host must be an IP address.

    >>> split_server("10.0.0.2:5590")
    ('10.0.0.2', 5590)
    """
    host, sep, port = server.rpartition(":")
    if not sep:
        raise ValueError(f"server {server!r}: give ip:port")
    host = host.strip("[]")
    ipaddress.ip_address(host)
    return host, int(port)


def is_loopback(host: str) -> bool:
    return ipaddress.ip_address(host).is_loopback


def sts_token(key: str, server: str, region: str = DEFAULT_STS_REGION, credentials=None) -> dict:
    """A presigned sts:GetCallerIdentity request for ``region`` that names lease ``key`` and ``server`` (ip:port) in
    signed headers. ``credentials`` defaults to botocore's credential chain of this process."""
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest

    if credentials is None:
        import botocore.session

        found = botocore.session.get_session().get_credentials()
        if found is None:
            raise LeaseError("remote DRAM: no AWS credentials for the lease token")
        credentials = found.get_frozen_credentials()
    url = sts_url(region)
    request = AWSRequest(method="POST", url=url, data=STS_BODY, headers={
        "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
        LEASE_HEADER: key,
        NONCE_HEADER: os.urandom(16).hex(),
        SERVER_HEADER: server,
    })
    SigV4Auth(credentials, "sts", region).add_auth(request)
    return {"url": url, "body": STS_BODY, "headers": dict(request.headers.items())}


def tls_client_context(ca_file: str) -> ssl.SSLContext:
    """TLS 1.3, trusting only the CA certificate in ``ca_file`` (no system CAs), checking the server's IP."""
    context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=ca_file)
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    return context


@dataclass(frozen=True)
class Lease:
    server: str
    key: str
    addr: int
    nbytes: int
    dev: int
    metadata: bytes


def check_client_policy(server: str, auth: str, tls_ca: str) -> str:
    """Why a client would send a lease request it must not send, or "": off loopback a request needs TLS and a token,
    so no token or lease travels in plaintext and no server hands out a lease without checking the caller."""
    if auth not in AUTH_MODES:
        return f"auth {auth!r}: one of {', '.join(AUTH_MODES)}"
    host, _ = split_server(server)
    if is_loopback(host):
        return ""
    if not tls_ca:
        return f"server {server} is off loopback: give tls_ca (the CA certificate the lease server wrote)"
    if auth != "sts":
        return f"server {server} is off loopback: use auth sts"
    return ""


def request_lease(server: str, key: str, nbytes: int, timeout: float = 10.0, auth: str = "none", tls_ca: str = "",
                  sts_region: str = DEFAULT_STS_REGION, token_fn=sts_token) -> Lease:
    """Ask the lease server at ``server`` (ip:port) for ``nbytes`` under ``key``. The same key gets the same lease
    again (a restarted rank). Raises LeaseError on any refusal. With ``tls_ca`` a failed TLS check ends the request
    before any token is sent; there is no plaintext retry."""
    error = check_client_policy(server, auth, tls_ca)
    if error:
        raise LeaseError(f"remote DRAM: {error}")
    host, port = split_server(server)
    message = {"op": "lease", "key": key, "nbytes": int(nbytes)}
    try:
        with socket.create_connection((host, port), timeout=timeout) as raw:
            sock = raw
            if tls_ca:
                try:
                    sock = tls_client_context(tls_ca).wrap_socket(raw, server_hostname=host)
                except OSError as exc:  # ssl.SSLError is an OSError
                    raise LeaseError(f"remote DRAM: server {server} failed the TLS check ({exc})") from exc
            if auth == "sts":
                message["auth"] = token_fn(key, f"{host}:{port}", sts_region)
            with sock:
                sock.sendall(json.dumps(message).encode() + b"\n")
                reply = b""
                while not reply.endswith(b"\n") and len(reply) < MAX_LINE:
                    part = sock.recv(MAX_LINE)
                    if not part:
                        break
                    reply += part
    except OSError as exc:
        raise LeaseError(f"remote DRAM: lease server {server}: {exc}") from exc
    if not reply:
        raise LeaseError(f"remote DRAM: server {server} closed the connection with no reply: check its allow lists "
                         "for this host's address and IAM identity")
    try:
        body = json.loads(reply)
    except ValueError:
        raise LeaseError(f"remote DRAM: server {server} sent {len(reply)} B that are not a lease") from None
    if not isinstance(body, dict) or not body.get("ok"):
        raise LeaseError(f"remote DRAM: server {server} refused lease {key}: "
                         f"{body.get('error') if isinstance(body, dict) else body!r}")
    lease = Lease(server=server, key=key, addr=int(body["addr"]), nbytes=int(body["nbytes"]),
                  dev=int(body.get("dev", 0)), metadata=base64.b64decode(body["metadata"]))
    if lease.nbytes != int(nbytes):
        raise LeaseError(f"remote DRAM: server {server} leased {lease.nbytes} B for {key}, not {nbytes} B")
    return lease
