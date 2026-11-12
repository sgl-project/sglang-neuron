# SPDX-License-Identifier: Apache-2.0
"""The rank coordinates of the Kimi-K3 2-server decode and the checkpoint reads of one rank.

``DecodeRank`` gives the coordinates of global rank ``g`` that decide its weights: the attention TP rank ``g % tp``
(heads, vocab shard, MoE projection shards), the DP replica ``g // tp``, the world rank ``g`` (dense intermediate
shard) and the routed expert group and intermediate part (EP x TP experts: ``G = world / routed_tp`` groups, group
``g % G``, part ``g // G``).

``RankCheckpoint`` reads one slice of a checkpoint tensor through ``safe_open(...).get_slice``, so a rank reads only
its rows and columns, and casts it to the dtype of the stack input; a cast that can change a value stops the load.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

TEXT_PREFIX = "language_model."
EXPERT_GROUP_RANKS = 128
"""The routed expert groups of the served split: 7 experts on each group, ``world / 128`` intermediate parts."""


@dataclass(frozen=True)
class DecodeRank:
    """The coordinates of one global rank of the decode world (``world = tp * dp``)."""

    rank: int
    world: int = 256
    tp: int = 32
    dp: int = 8

    def __post_init__(self) -> None:
        if self.tp * self.dp != self.world:
            raise ValueError(f"The decode world {self.world} is not TP {self.tp} x DP {self.dp}.")
        if not 0 <= self.rank < self.world:
            raise ValueError(f"The rank {self.rank} is outside the decode world of {self.world} ranks.")
        if self.world % EXPERT_GROUP_RANKS:
            raise ValueError(f"The decode world {self.world} is not a whole number of {EXPERT_GROUP_RANKS}-rank "
                             "expert splits.")

    @property
    def tp_rank(self) -> int:
        return self.rank % self.tp

    @property
    def dp_rank(self) -> int:
        return self.rank // self.tp

    @property
    def routed_tp(self) -> int:
        """The intermediate parts of each routed expert."""
        return self.world // EXPERT_GROUP_RANKS


def _lossless(source: torch.dtype, target: torch.dtype) -> bool:
    if source == target:
        return True
    if not (source.is_floating_point and target.is_floating_point):
        return False
    narrow, wide = torch.finfo(source), torch.finfo(target)
    return wide.eps <= narrow.eps and wide.max >= narrow.max and wide.tiny <= narrow.tiny


class RankCheckpoint:
    """Slices of a sharded safetensors checkpoint (``model.safetensors.index.json``).

    Keys are given without the ``language_model.`` prefix; the reader adds it when the checkpoint has it.
    """

    def __init__(self, path: str):
        self.path = path
        with open(os.path.join(path, "model.safetensors.index.json")) as f:
            self.weight_map = json.load(f)["weight_map"]
        self._prefix = TEXT_PREFIX if any(k.startswith(TEXT_PREFIX) for k in self.weight_map) else ""
        self._handles: dict[str, Any] = {}

    def _slice(self, key: str):
        from safetensors import safe_open

        full = self._prefix + key
        name = self.weight_map.get(full)
        if name is None:
            raise KeyError(f"The checkpoint {self.path} has no tensor {full}.")
        if name not in self._handles:
            self._handles[name] = safe_open(os.path.join(self.path, name), framework="pt", device="cpu")
        return self._handles[name].get_slice(full)

    def shape(self, key: str) -> tuple[int, ...]:
        return tuple(self._slice(key).get_shape())

    def read(self, key: str, index: tuple = (), dtype: torch.dtype | None = None) -> Tensor:
        """The slice ``index`` (a tuple of slices, leading dims) of ``key``, in ``dtype`` (None: the checkpoint's).

        Raises:
            TypeError: The cast to ``dtype`` can change a value.
        """
        source = self._slice(key)
        index = tuple(index) + (slice(None),) * (len(source.get_shape()) - len(index))
        tensor = source[index]
        if dtype is None or tensor.dtype == dtype:
            return tensor.contiguous()
        if not _lossless(tensor.dtype, dtype):
            raise TypeError(f"The checkpoint tensor {key} is {tensor.dtype}; a cast to {dtype} would change its "
                            "values.")
        return tensor.to(dtype).contiguous()

    def rows(self, key: str, start: int, stop: int, dtype: torch.dtype | None = None) -> Tensor:
        return self.read(key, (slice(start, stop),), dtype)

    def cols(self, key: str, start: int, stop: int, dtype: torch.dtype | None = None) -> Tensor:
        return self.read(key, (slice(None), slice(start, stop)), dtype)

    def full(self, key: str, dtype: torch.dtype | None = None) -> Tensor:
        return self.read(key, (), dtype)

    def close(self) -> None:
        self._handles.clear()
