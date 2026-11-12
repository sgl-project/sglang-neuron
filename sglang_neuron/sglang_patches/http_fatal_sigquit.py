# SPDX-License-Identifier: Apache-2.0
"""A multi-worker HTTP server keeps SGLang's fatal SIGQUIT handler while Uvicorn supervises its workers.

SGLang's subprocess watchdog reports a dead scheduler or detokenizer by sending SIGQUIT to the server process, whose
handler kills the process tree. With ``--tokenizer-worker-num`` > 1 that process runs Uvicorn's ``Multiprocess``
supervisor (``uvicorn.run(..., workers=N)`` or SGLang's ``_ReusePortMultiprocess``), whose constructor installs its own
handler for SIGQUIT, among its other signals, and queues the signal; recent Uvicorn dispatches no handler for it. The
watchdog's SIGQUIT is then swallowed: the HTTP workers keep accepting requests that no scheduler will ever answer.

Here the supervisor restores the handler the process had before its construction (on the main thread, the only one
that may set handlers): ``_ReusePortMultiprocess.__init__`` is wrapped to do so, and ``_setup_and_run_http_server`` is
upstream's, rebuilt so that its plain multi-worker branch runs the same supervisor ``uvicorn.run`` builds (a Config, its
bound socket, ``Multiprocess(config, sockets=[sock]).run()``) through a ``Multiprocess`` that restores the handler. The
single-worker server, the SSL-refresh server and Granian keep upstream's paths.

The HTTP server runs in the launching process, so these are launch hooks.

Upstream: a fix SGLang can take as it is.

Targets, md5-pinned against upstream SGLang 43881a7: ``_ReusePortMultiprocess.__init__`` (wrapped) and
``_setup_and_run_http_server`` (rebuilt; sglang_patches/tokenizer_worker_plugins pins it too)."""

from __future__ import annotations

import functools

_HTTP = "sglang.srt.entrypoints.http_server"
REUSEPORT_INIT = f"{_HTTP}._ReusePortMultiprocess.__init__"
SETUP_AND_RUN = f"{_HTTP}._setup_and_run_http_server"
_TARGETS_MD5 = {
    # binds its reserved SO_REUSEPORT listener, then Multiprocess.__init__(config, sockets=[])
    (_HTTP, "_ReusePortMultiprocess.__init__"): "1c792a70dd135857608e35f2c1bc835a",
    (_HTTP, "_setup_and_run_http_server"): "ed5029a85515bde9bd024e3f6cc5ed1b",
}
# upstream's plain multi-worker run and its replacement: the same supervisor, keeping the SIGQUIT handler
SUBSTITUTIONS = (
    ("                    uvicorn.run(app_path, **uvicorn_options)\n",
     "                    from sglang_neuron.sglang_patches.http_fatal_sigquit import run_workers\n"
     "\n"
     "                    run_workers(app_path, uvicorn_options)\n"),
)


def keep_sigquit(construct):
    """Run ``construct()`` and restore the process's SIGQUIT handler afterwards, on the main thread."""
    import signal
    import threading

    handler = signal.getsignal(signal.SIGQUIT)
    try:
        return construct()
    finally:
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGQUIT, handler)


@functools.lru_cache(maxsize=None)
def supervisor_class():
    """Uvicorn's Multiprocess, constructed with SGLang's SIGQUIT handler kept."""
    from uvicorn.supervisors.multiprocess import Multiprocess

    class SigquitKeepingMultiprocess(Multiprocess):
        def __init__(self, *args, **kwargs):
            keep_sigquit(lambda: super(SigquitKeepingMultiprocess, self).__init__(*args, **kwargs))

    return SigquitKeepingMultiprocess


def run_workers(app_path, uvicorn_options) -> None:
    """``uvicorn.run(app_path, **uvicorn_options)`` for workers > 1: Uvicorn's supervisor over one bound socket, through
    supervisor_class."""
    import uvicorn

    config = uvicorn.Config(app_path, **uvicorn_options)
    sock = config.bind_socket()
    try:
        supervisor_class()(config, sockets=[sock]).run()
    finally:
        sock.close()


def reuseport_init(original_fn, self, *args, **kwargs):
    """AROUND _ReusePortMultiprocess.__init__: upstream's construction, with the SIGQUIT handler kept."""
    keep_sigquit(lambda: original_fn(self, *args, **kwargs))


def setup_and_run_http_server():
    """_setup_and_run_http_server, upstream's with its plain multi-worker branch through run_workers."""
    from sglang_neuron.sglang_patches import source_guard

    return source_guard.with_substitutions(source_guard.resolve(_HTTP, "_setup_and_run_http_server"), SUBSTITUTIONS)


def register(registry, around, replace) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
    registry.register(REUSEPORT_INIT, reuseport_init, around)
    registry.register(SETUP_AND_RUN, setup_and_run_http_server(), replace)
