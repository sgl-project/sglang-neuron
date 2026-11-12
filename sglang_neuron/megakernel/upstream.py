# SPDX-License-Identifier: Apache-2.0
"""The one module of the megakernel host that imports SGLang's megakernel contract
(``sglang.kernels.plugin_kernels.nvidia.spec``, upstream RFC sgl-project/sglang#46355).

The contract is SGLang's: ``MegakernelSpec`` (requirements, accepted features, numerics, ``max_rows``, ``build``),
``BuildContext``, ``Step`` and the ``BuiltMegakernel`` protocol (``widths``, ``summary``, ``plan``, ``forward`` and the
optional draft entries). The host adds nothing to the structs; a Neuron step is a ``Step`` subclass
(``built.NeuronStep``). A tree without the contract, or with another API version, refuses here with the reason.
"""
from __future__ import annotations

from typing import Any, Optional

try:
    from sglang.kernels.plugin_kernels.nvidia import spec as _spec
except Exception as _err:  # noqa: BLE001 -- a tree older than the contract; refused at use
    _spec = None
    _IMPORT_ERROR: Optional[BaseException] = _err
else:
    _IMPORT_ERROR = None

# The contract version the host and the shipped specs are written against.
API_VERSION = 2
# The SGLang flag under which SGLang's binding installs a megakernel (the platform seams route it to this host).
FLAG = "--enable-nvidia-megakernel"


def available() -> bool:
    """Whether the served SGLang tree carries the megakernel contract at the version the host is written against."""
    return _spec is not None and getattr(_spec, "MEGAKERNEL_API_VERSION", None) == API_VERSION


def unavailable_reason() -> str:
    if _spec is None:
        return (f"the served SGLang tree has no megakernel contract (sglang.kernels.plugin_kernels.nvidia.spec does not "
                f"import: {_IMPORT_ERROR!r})")
    return (f"the served SGLang tree's megakernel contract is API version {_spec.MEGAKERNEL_API_VERSION}; the host is "
            f"written against version {API_VERSION}")


def require() -> Any:
    """The contract module, or a refusal naming why this tree cannot serve a megakernel."""
    if not available():
        raise RuntimeError(unavailable_reason())
    return _spec


def contract():
    """(MegakernelSpec, Requirements, Numerics, NumericsKind, Feature, BuildContext, Step) of the served tree."""
    s = require()
    return s.MegakernelSpec, s.Requirements, s.Numerics, s.NumericsKind, s.Feature, s.BuildContext, s.Step
