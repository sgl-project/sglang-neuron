# SPDX-License-Identifier: Apache-2.0
"""Rank 0 binds the torch.distributed rendezvous port at once and hands the socket to its TCPStore.

SGLang's PortArgs picks this port with get_free_port_below_ephemeral(), which binds it once to check that it is free and
then releases it. That keeps it clear of bind(0) and connect() ports, but until init_process_group() binds it, any
process that binds an explicit port in that range can take it (another PD server's Neuron runtime on the node, or
another server's check landing on the same port). So for a tcp:// init method of this host, rank 0 binds the port
itself (or a fresh one), passes the bound socket to TCPStore(master_listen_fd=) and writes the port it holds to a
rendezvous file, which the other ranks read instead of probing the original port: a runtime holding that port accepts
the connection but never answers as a TCPStore.

The file name also holds the launch's key (SGLANG_NEURON_RENDEZVOUS_KEY, unique per launch): two servers can request
the same port, also two started from one script or parent process, and one shared file would cross-wire their worlds.

Only activates for a tcp:// address literal of this machine: a loopback
address, or the wildcard address SGLang builds the init method from when the
server binds every interface (--host 0.0.0.0). A real address from --dist-init-addr, hostnames, env://, and
store= calls pass through unmodified. The stores carry the process group
timeout, as the stores of torch's own tcp:// rendezvous do.
"""

import errno
import ipaddress
import os
import socket
import time
import uuid
from datetime import timedelta
from urllib.parse import urlparse

from logging import getLogger as init_logger

logger = init_logger(__name__)

_applied = False
_original_init_process_group = None

_RENDEZVOUS_DIR = "/tmp"
_RENDEZVOUS_PREFIX = "sglang_neuron_dist_port_"
_POLL_INTERVAL_S = 0.01
_CLIENT_CONNECT_TIMEOUT_S = 5
LAUNCH_KEY = "SGLANG_NEURON_RENDEZVOUS_KEY"


def export_launch_key(environ=os.environ) -> None:
    """Give this launch a fresh rendezvous key, which every process it starts inherits; set even where the environment
    holds one, since a key inherited from an enclosing launch would share that launch's rendezvous files."""
    environ[LAUNCH_KEY] = uuid.uuid4().hex


def _group_key() -> str:
    """The id every rank of one server shares and no other server holds: the launch's key (export_launch_key, which the
    platform's launch hook runs before it starts any scheduler)."""
    key = os.environ.get(LAUNCH_KEY)
    if not key:
        raise RuntimeError(f"{LAUNCH_KEY} is unset: sglang-neuron's launch hook (NeuronPlatform."
                           "apply_server_args_defaults) sets it for every process of a server, and the ranks of a "
                           "loopback torch rendezvous key their rendezvous file on it")
    return key


def _names_this_host(host: str | None) -> bool:
    """Whether host is an address literal of this machine (loopback or wildcard).

    Example:
        >>> [_names_this_host(h) for h in ("0.0.0.0", "127.0.0.1", "::1", "10.1.2.3", "node-0")]
        [True, True, True, False, False]
    """
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or address.is_unspecified


def _family(host: str) -> socket.AddressFamily:
    return socket.AF_INET6 if ipaddress.ip_address(host).version == 6 else socket.AF_INET


def _rendezvous_path(port: int) -> str:
    """Return the rendezvous file of this server."""
    return os.path.join(_RENDEZVOUS_DIR, f"{_RENDEZVOUS_PREFIX}{port}_{_group_key()}")


def _init_process_group_patched(
    backend=None, init_method=None, timeout=None, world_size=-1, rank=-1, store=None, **kwargs
):
    """Intercept loopback TCP init to handle EADDRINUSE via file rendezvous. The parameters are torch's, in its order, so
    a positional call binds as it does on torch's own function."""
    if store is not None or not init_method or not init_method.startswith("tcp://"):
        return _original_init_process_group(
            backend=backend,
            init_method=init_method,
            timeout=timeout,
            store=store,
            rank=rank,
            world_size=world_size,
            **kwargs,
        )

    parsed = urlparse(init_method)
    host = parsed.hostname
    port = parsed.port

    # only this host's own addresses
    if not _names_this_host(host) or port is None:
        return _original_init_process_group(
            backend=backend,
            init_method=init_method,
            timeout=timeout,
            rank=rank,
            world_size=world_size,
            **kwargs,
        )

    store_timeout = timeout
    if store_timeout is None:
        from torch.distributed.constants import default_pg_timeout

        # torch's default for a non-NCCL world, which its own rendezvous uses
        store_timeout = default_pg_timeout

    if rank != 0:
        tcp_store = _rankN_connect_store(host, port, rank, world_size, store_timeout)
        return _original_init_process_group(backend=backend, timeout=timeout, store=tcp_store, rank=rank,
                                            world_size=world_size, **kwargs)
    file_path = _rendezvous_path(port)
    try:
        tcp_store = _rank0_create_store(host, port, world_size, store_timeout)
        return _original_init_process_group(backend=backend, timeout=timeout, store=tcp_store, rank=rank,
                                            world_size=world_size, **kwargs)
    finally:
        # init_process_group is a barrier: all ranks have connected, or the group failed to form.
        # The rendezvous file is no longer needed.
        _remove(file_path)


def _remove(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def _rank0_create_store(host: str, port: int, world_size: int, timeout: timedelta):
    """Bind port (original or fresh), write rendezvous file, return TCPStore."""
    from torch.distributed import TCPStore

    file_path = _rendezvous_path(port)

    # Remove stale file from a previous server that used the same port number
    _remove(file_path)

    sock = socket.socket(_family(host), socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
        sock.listen(128)
        actual_port = port
    except OSError as e:
        if e.errno != errno.EADDRINUSE:
            sock.close()
            raise
        sock.close()
        sock = socket.socket(_family(host), socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, 0))
        sock.listen(128)
        actual_port = sock.getsockname()[1]
        logger.warning(
            "EADDRINUSE on port %d — retrying with port %d", port, actual_port
        )

    fd = sock.detach()

    # Write file BEFORE TCPStore — the socket is already in LISTEN state (kernel
    # accepts connections into backlog). TCPStore will process them on construction.
    # TCPStore owns the fd once constructed; a failure before that closes it here.
    tmp_path = f"{file_path}.{os.getpid()}"
    try:
        with open(tmp_path, "w") as f:
            f.write(str(actual_port))
        os.rename(tmp_path, file_path)
    except BaseException:
        os.close(fd)
        _remove(tmp_path)
        raise

    store = TCPStore(
        host_name=host,
        port=actual_port,
        world_size=world_size,
        is_master=True,
        timeout=timeout,
        master_listen_fd=fd,
        use_libuv=False,
    )

    logger.info(
        "Rank 0 TCPStore on port %d (fd=%d, file=%s)", actual_port, fd, file_path
    )
    return store


def _rankN_connect_store(host: str, port: int, rank: int, world_size: int, timeout: timedelta):
    """Poll rendezvous file for rank 0's actual port, then connect."""
    from torch.distributed import TCPStore

    file_path = _rendezvous_path(port)
    deadline = time.monotonic() + timeout.total_seconds()
    last_port = last_error = None

    while True:
        try:
            with open(file_path, "r") as f:
                content = f.read()
        except FileNotFoundError:
            content = None
        if content is not None:
            actual_port = _read_port(file_path, content)
            try:
                store = TCPStore(
                    host_name=host,
                    port=actual_port,
                    world_size=world_size,
                    is_master=False,
                    timeout=timedelta(seconds=_CLIENT_CONNECT_TIMEOUT_S),
                    use_libuv=False,
                )
                # The short timeout is for the connect only (a stale file
                # names a dead port); later waits get the group's timeout.
                store.set_timeout(timeout)
                return store
            except Exception as e:
                # TCPStore connect failed (a stale file naming a dead port, which rank 0 overwrites shortly):
                # keep polling
                (logger.warning if last_error is None else logger.debug)(
                    "Rank %d: TCPStore connect to %s:%s (from %s) failed, retrying until %s: %s", rank, host,
                    actual_port, file_path, timeout, e)
                last_port, last_error = actual_port, e

        if time.monotonic() >= deadline:
            if last_error is not None:
                raise RuntimeError(
                    f"Rank {rank}: rendezvous file {file_path} names port {last_port}, but connecting a TCPStore client "
                    f"to {host}:{last_port} kept failing for {timeout}; last error: {last_error}. Check that rank 0 of "
                    f"this server is alive and that no other process holds port {last_port}.") from last_error
            raise RuntimeError(f"Rank {rank}: rendezvous file {file_path} did not appear within {timeout}; rank 0 of "
                               "this server never bound its TCPStore port.")
        time.sleep(_POLL_INTERVAL_S)


def _read_port(file_path: str, content: str) -> int:
    """The port in a rendezvous file. Rank 0 renames a whole file into place, so anything else is refused at once
    rather than polled for the group timeout."""
    try:
        port = int(content)
    except ValueError:
        port = 0
    if not 1 <= port <= 65535:
        raise RuntimeError(f"rendezvous file {file_path} holds {content!r}, not a port: rank 0 of this server writes "
                           "it whole, so another process wrote it; remove it and relaunch")
    return port


def apply_port_hold_patch():
    """Patch init_process_group with EADDRINUSE-resilient file rendezvous."""
    global _applied, _original_init_process_group

    if _applied:
        return

    import torch.distributed as dist

    from sglang_neuron.backend import torch_pins

    torch_pins.require("torch.distributed.init_process_group", dist.init_process_group, "backend/port_hold.py")
    _original_init_process_group = dist.init_process_group
    dist.init_process_group = _init_process_group_patched

    _applied = True
    logger.info("Port-hold patch applied (EADDRINUSE file-rendezvous)")
