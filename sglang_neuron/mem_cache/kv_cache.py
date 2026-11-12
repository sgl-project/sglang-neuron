# SPDX-License-Identifier: Apache-2.0
"""The KV cache a model declares: one LayerSpec per attention layer, in the order of SGLang's pool layers. The pool's
shape and dtype come from SGLang's model config and the launch's KV dtype; the model binds each layer's buffers by
name."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LayerSpec:
    name: str


@dataclass
class KVSpec:
    layers: list[LayerSpec]
