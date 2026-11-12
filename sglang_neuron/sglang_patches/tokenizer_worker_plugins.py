# SPDX-License-Identifier: Apache-2.0
"""Load the plugins in every tokenizer worker of a ``--tokenizer-worker-num`` > 1 launch.

Upstream starts each tokenizer worker as a fresh interpreter. The worker imports the app that
``SGLANG_HTTP_SERVER_APP_PATH`` names and builds its TokenizerManager in the app's lifespan (``init_multi_tokenizer``),
without calling ``load_plugins()``. The plugin's tokenizer-side hooks would then be missing from the very workers that
serve the requests. For such a launch the platform sets the app path to ``sglang_neuron.http_worker_app:app``, which is
SGLang's app imported after ``load_plugins()``. A custom app path is refused.

Relied-on targets, md5-pinned against upstream SGLang 43881a7: ``http_server.init_multi_tokenizer``,
``_run_granian_server`` and ``_setup_and_run_http_server``.

Upstream: SGLang can fix this by having its multi-tokenizer workers call load_plugins(), as every other process does;
the module then goes away."""

from __future__ import annotations

import os

APP_PATH = "sglang_neuron.http_worker_app:app"
SGLANG_APP_PATH = "sglang.srt.entrypoints.http_server:app"  # upstream SGLang's default
_ENV = "SGLANG_HTTP_SERVER_APP_PATH"
_HTTP_SERVER = "sglang.srt.entrypoints.http_server"
_TARGETS_MD5 = {
    "init_multi_tokenizer": "4c1585615632e5e9643e74f2dd35edaa",
    "_run_granian_server": "7e6bf456d795dfd5d31102adfa46df3e",
    "_setup_and_run_http_server": "ed5029a85515bde9bd024e3f6cc5ed1b",
}


def check_upstream() -> None:
    """Refuse an upstream http_server whose worker start-up or app-path handling differs from the source checked."""
    import importlib

    from sglang_neuron.sglang_patches import source_guard

    http_server = importlib.import_module(_HTTP_SERVER)
    for name, md5 in _TARGETS_MD5.items():
        source_guard.check(getattr(http_server, name), md5, f"{_HTTP_SERVER}.{name}", __name__)


def configure(server_args, environ=os.environ) -> None:
    """Name the plugin-loading app for a launch with more than one tokenizer worker."""
    if int(getattr(server_args, "tokenizer_worker_num", 1) or 1) <= 1:
        return
    check_upstream()
    current = environ.get(_ENV)
    if current not in (None, "", SGLANG_APP_PATH, APP_PATH):
        raise ValueError(f"{_ENV}={current!r} with --tokenizer-worker-num {server_args.tokenizer_worker_num}: its "
                         f"tokenizer workers would serve without the Neuron plugin's hooks; leave it unset (the "
                         f"platform serves {APP_PATH}, SGLang's app after load_plugins())")
    environ[_ENV] = APP_PATH
