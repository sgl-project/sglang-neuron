# SPDX-License-Identifier: Apache-2.0
"""The process-level switch of the plugin's Neuron backend, and the NKI trace-cache setting.

torch-neuronx reads its own compiler and runtime variables; this module forwards no names to it.
"""
from __future__ import annotations

import os

# NKI's trace cache keys a kernel trace without the compiler arguments, and the runtime compiles each graph family with
# its own, so a trace cached for one family must not be reused for another: off from the first import of this module,
# and the backend's init refuses a launch environment that set another value (stack.disable_nki_trace_cache).
os.environ.setdefault("NKI_DISABLE_TRACE_CACHE", "1")

_BOOLEAN_DEFAULTS = {
    # no Neuron device: the CPU test suites run the torch paths
    "SGLANG_NEURON_CPU_MODE": False,
}


def __getattr__(name):
    """The boolean switch, read from the environment on every access: unset is its default, "0" or "1" is off or on, and
    any other value is refused by name."""
    if name in _BOOLEAN_DEFAULTS:
        value = os.environ.get(name)
        if value is None:
            return _BOOLEAN_DEFAULTS[name]
        if value not in ("0", "1"):
            raise ValueError(f"{name}={value!r}: set it to 0 or 1, or leave it unset")
        return value == "1"
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
