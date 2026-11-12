# SPDX-License-Identifier: Apache-2.0
"""Neuron implementation of Wan2.2 A14B (T2V and I2V).

The exports resolve on first access: importing the DiT imports nki and torch-neuronx, which must
stay out of the launcher, and the launcher imports this package when it registers the pipelines.
"""

from typing import Any

__all__ = ["WanTransformer3DModel"]


def __getattr__(name: str) -> Any:
    if name == "WanTransformer3DModel":
        from .wan2_2_transformer import WanTransformer3DModel

        return WanTransformer3DModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
