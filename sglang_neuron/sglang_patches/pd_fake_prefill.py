# SPDX-License-Identifier: Apache-2.0
"""A PD prefill on SGLang's fake transfer backend: a prefill server without a decode peer, which moves no KV.

Upstream's fake backend serves a decode without a prefill, and the warm-up of every PD server (through the fake
bootstrap host). A prefill launch with it is refused by an assertion in ``handle_pd_disaggregation``, and the
prefill's tokenizer manager would start the backend's bootstrap server, which the fake backend does not have
(``get_kv_class`` gives None). For a prefill launched with ``--disaggregation-transfer-backend fake``:

* ``handle_pd_disaggregation`` runs as for a NIXL prefill: for that call its resolving view reads the backend as
  ``nixl`` (the prefill branch's one fake-specific statement is the assertion; the staging-buffer and role-switch
  checks that follow see the launch's own flags), and the server arguments are not written;
* ``start_disagg_service`` starts no bootstrap server.

Each request then gets upstream's fake sender (``PrefillBootstrapQueue.create_sender`` through ``get_kv_class``),
which reports a finished transfer as soon as the last chunk is sent. Every other launch runs both functions unchanged.

Targets are md5-pinned against upstream SGLang 43881a7.

Upstream: the fake backend could serve a prefill as it serves a decode; this module then goes away."""

from __future__ import annotations

import contextlib
import importlib

HANDLE = "sglang.srt.arg_groups.pd_disaggregation_hook.handle_pd_disaggregation"
# the tokenizer manager's own binding, which it calls
START = "sglang.srt.managers.tokenizer_manager.start_disagg_service"
_TARGETS_MD5 = {
    HANDLE: "add6282d525c6422e434aa0c0bde9e95",
    START: "56e8976a902af879bab75ed2e87c7bcc",
}
FAKE = "fake"


def fake_prefill(server_args) -> bool:
    return (str(getattr(server_args, "disaggregation_mode", "null")) == "prefill"
            and str(getattr(server_args, "disaggregation_transfer_backend", "")) == FAKE)


class _NixlBackendView:
    """A resolving view whose transfer backend reads ``nixl``; every other field is the view's."""

    def __init__(self, view):
        object.__setattr__(self, "_view", view)

    def __getattr__(self, name):
        if name == "disaggregation_transfer_backend":
            return "nixl"
        return getattr(self._view, name)


@contextlib.contextmanager
def _backend_read_as_nixl(hook):
    original = hook.resolving_view
    hook.resolving_view = lambda server_args: _NixlBackendView(original(server_args))
    try:
        yield
    finally:
        hook.resolving_view = original


def handle_pd_disaggregation(original_fn, server_args, *args, **kwargs):
    if not fake_prefill(server_args):
        return original_fn(server_args, *args, **kwargs)
    from sglang.srt.arg_groups import pd_disaggregation_hook

    # the server arguments are never written: the function reads the backend through its resolving view
    with _backend_read_as_nixl(pd_disaggregation_hook):
        return original_fn(server_args, *args, **kwargs)


def start_disagg_service(original_fn, *args, **kwargs):
    from sglang.srt.runtime_context import get_disagg

    disagg = get_disagg()
    if disagg.disaggregation_mode == "prefill" and disagg.disaggregation_transfer_backend == FAKE:
        return None
    return original_fn(*args, **kwargs)


def register(registry, around) -> None:
    from sglang_neuron.sglang_patches import source_guard

    for target, md5 in _TARGETS_MD5.items():
        module_name, _, name = target.rpartition(".")
        source_guard.check(getattr(importlib.import_module(module_name), name), md5, target, __name__)
    registry.register(HANDLE, handle_pd_disaggregation, around)
    registry.register(START, start_disagg_service, around)
