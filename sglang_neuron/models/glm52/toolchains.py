# SPDX-License-Identifier: Apache-2.0
"""Whether a GLM-5.2 launch compiles on the released toolchain.

The released toolchain is the neuronx-cc and nki of the pinned Neuron SDK release (release/neuron_sdk.toml),
matched on both versions exactly, with the compiler on PATH from the same install this process imports
(Toolchain.same_install): torch-neuronx keys cached NEFFs on that package's version. The graphs, compiler arguments and
recipes are validated on that toolchain only, so a launch on any other is refused.
"""

from __future__ import annotations

from sglang_neuron.release import neuron_sdk

_SDK = neuron_sdk.pin()
#: (neuronx-cc, nki) of the released toolchain the recipe is validated on, with where it comes from
RELEASED_TOOLCHAINS = {
    (_SDK.wheels["neuronx-cc"].version, _SDK.wheels["nki"].version): f"{_SDK.release} (pip.repos.neuron.amazonaws.com)",
}


def is_released(tc) -> bool:
    """Whether a ``release.toolchain.Toolchain`` is the released toolchain: the compiler on PATH runs the imported
    neuronxcc package, at a (neuronx-cc, nki) pair of RELEASED_TOOLCHAINS."""
    return tc.same_install and (tc.neuronxcc_version, tc.nki) in RELEASED_TOOLCHAINS


def describe(tc) -> str:
    """``neuronx-cc <v> + nki <v>`` of a ``release.toolchain.Toolchain``, with the source of the released toolchain it
    is (``, <source>``), for the launch's log."""
    src = RELEASED_TOOLCHAINS[(tc.neuronxcc_version, tc.nki)] if is_released(tc) else None
    return f"neuronx-cc {tc.neuronxcc_version} + nki {tc.nki}" + (f", {src}" if src else "")


def released_names() -> str:
    """The released toolchains, for a refusal: ``neuronx-cc <v> + nki <v> (<source>)``, one after another."""
    return "; ".join(f"neuronx-cc {cc} + nki {nki} ({src})" for (cc, nki), src in RELEASED_TOOLCHAINS.items())
