# SPDX-License-Identifier: Apache-2.0
"""Let the DP controller block on its tokenizer socket instead of spinning a core.

Upstream's ``DataParallelController.event_loop`` drains the socket with non-blocking receives and starts over as soon as
it is empty, so every DP server's controller keeps a core fully busy for its lifetime. On a decode node whose ranks need
every core, that core comes out of their budget. The loop is rebuilt from upstream's source with a ``zmq.Poller`` wait
of at most ``POLL_MS`` before each drain. A message wakes it at once, the drain and dispatch stay upstream's, and the
soft watchdog is still fed on every pass.

Target md5-pinned against upstream SGLang 43881a7.

Upstream: SGLang can take this fix as it is; its controller would then wait on its socket instead of spinning a core."""

from __future__ import annotations

_MODULE = "sglang.srt.managers.data_parallel_controller"
TARGET = f"{_MODULE}.DataParallelController.event_loop"
_TARGET_MD5 = "590d3264a7c8c343e1ca21b70e171974"
POLL_MS = 100
UPSTREAM_LOOP = "    while True:\n        while True:\n            self.soft_watchdog.feed()\n"
NEURON_LOOP = (
    "    poller = zmq.Poller()\n"
    "    poller.register(self.recv_from_tokenizer, zmq.POLLIN)\n"
    "    while True:\n"
    f"        poller.poll({POLL_MS})\n"
    "        while True:\n"
    "            self.soft_watchdog.feed()\n"
)


def event_loop():
    """Upstream's DataParallelController.event_loop with a bounded wait on the tokenizer socket before each drain."""
    from sglang_neuron.sglang_patches import source_guard

    original = source_guard.resolve(_MODULE, "DataParallelController.event_loop")
    source_guard.check(original, _TARGET_MD5, TARGET, __name__)
    return source_guard.with_substitutions(original, [(UPSTREAM_LOOP, NEURON_LOOP)])


def register(registry, replace) -> None:
    registry.register(TARGET, event_loop(), replace)
