# SPDX-License-Identifier: Apache-2.0
"""Let the PD bootstrap server's listen queue hold every decode rank's first contact at once.

On its first request from a prefill it has not met, every rank of a decode fetches ``/route`` from that prefill's
bootstrap server at the same moment. Upstream's ``CommonKVBootstrapServer._run_server`` listens with aiohttp's default
backlog of 128, and Linux's accept queue holds one connection more, so more than 129 waiting connects overflow it: the
giga-kernel decode's 128 ranks fill it to the edge alone, and a relaunched rank or a second decode meeting the same
prefill at once goes past it. The kernel drops the excess SYNs, those ranks retry after the SYN retransmit timeout
(1 s), and the ranks that got through wait for them in the transfer poll's reduce.

``_run_server`` is rebuilt from upstream's source with ``backlog=BACKLOG`` (4096) added to its one ``web.TCPSite`` call
(``UPSTREAM_SITE`` -> ``NEURON_SITE``); the kernel caps it at ``net.core.somaxconn``. The server is otherwise
upstream's, its unauthenticated routes included.
``NixlKVBootstrapServer``, which ``get_kv_class`` maps the NIXL backend to, inherits the method. The server runs in the
process that runs SGLang's tokenizer manager, so the replacement is one of the launch hooks.

Target md5-pinned against upstream SGLang 43881a7: ``CommonKVBootstrapServer._run_server``; relied on: the NIXL
bootstrap server class (no ``_run_server`` of its own) and ``disaggregation.utils.get_kv_class``.

Upstream: a one-line change to ``CommonKVBootstrapServer._run_server`` (a backlog for its ``web.TCPSite``); nothing in
it is Neuron-specific, and any decode of more than 129 ranks needs it."""

from __future__ import annotations

TARGET = ("sglang.srt.disaggregation.common.conn", "CommonKVBootstrapServer._run_server")
_NIXL_SERVER = ("sglang.srt.disaggregation.nixl.conn", "NixlKVBootstrapServer")
_TARGETS_MD5 = {
    TARGET: "400d1be64a0af8b3378a17f4421d85a1",
    # inherits _run_server
    _NIXL_SERVER: "71d4dff3dcca56af4058b9b2eef0cda2",
    # maps the NIXL backend to that class
    ("sglang.srt.disaggregation.utils", "get_kv_class"): "611e883df7a0e779693b07c892ea71c9",
}
BACKLOG = 4096
UPSTREAM_SITE = "        site = web.TCPSite(self._runner, host=self.host, port=self.port)\n"
NEURON_SITE = f"        site = web.TCPSite(self._runner, host=self.host, port=self.port, backlog={BACKLOG})\n"


def run_server():
    """CommonKVBootstrapServer._run_server, upstream's with the listen backlog BACKLOG."""
    from sglang_neuron.sglang_patches import source_guard

    return source_guard.with_substitutions(source_guard.resolve(*TARGET), [(UPSTREAM_SITE, NEURON_SITE)])


def register(registry, replace) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
    # the class's md5 covers its body; this also catches a _run_server set on the class from outside it
    if "_run_server" in vars(source_guard.resolve(*_NIXL_SERVER)):
        raise RuntimeError(f"NixlKVBootstrapServer defines its own _run_server, which {__name__} does not replace")
    registry.register(".".join(TARGET), run_server(), replace)
