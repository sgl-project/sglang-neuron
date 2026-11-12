# SPDX-License-Identifier: Apache-2.0
"""Let a NIXL transfer worker sleep ``AWAIT_POLL_S`` between polls of its WRITE handles instead of spinning.

Upstream's ``NixlKVManager._await_handles`` polls with ``time.sleep(0)``, so a prefill rank's transfer thread holds the
GIL for as long as its WRITEs are on the wire, and the rank's scheduler launches its next forwards late while the
stage's other ranks wait for it in their first collective. A 1 ms sleep sees a completion at most 1 ms later, against
transfers of milliseconds to seconds. A failed handle keeps upstream's path. SGLang has no setting for this poll."""

from __future__ import annotations

_MODULE = "sglang.srt.disaggregation.nixl.conn"
TARGET = f"{_MODULE}.NixlKVManager._await_handles"
# the upstream source, pinned in upstream_pins.json
PINNED = ((_MODULE, "NixlKVManager._await_handles"),)
AWAIT_POLL_S = 1e-3
# the method's source dedented, as the rebuild reads it
UPSTREAM_HEALTHY = "        if not any_failed:\n            time.sleep(0)\n            continue\n"
NEURON_HEALTHY = f"        if not any_failed:\n            time.sleep({AWAIT_POLL_S!r})\n            continue\n"


def await_handles():
    """Upstream's NixlKVManager._await_handles with AWAIT_POLL_S between its reads of healthy handles."""
    from sglang_neuron.sglang_patches import source_guard

    original = source_guard.resolve(_MODULE, "NixlKVManager._await_handles")
    source_guard.check(original, source_guard.pinned(_MODULE, "NixlKVManager._await_handles"), TARGET, __name__)
    return source_guard.with_substitutions(original, [(UPSTREAM_HEALTHY, NEURON_HEALTHY)])


def register(registry, replace) -> None:
    registry.register(TARGET, await_handles(), replace)
