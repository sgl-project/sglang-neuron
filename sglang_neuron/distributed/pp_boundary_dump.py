# SPDX-License-Identifier: Apache-2.0
"""A pipelined stage's boundary sends on disk (SGLANG_NEURON_PP_BOUNDARY_DUMP_DIR), to compare with a reference run's
tensors.

Every rank that sends over the PP seam writes its first SGLANG_NEURON_PP_BOUNDARY_DUMP_STEPS sends (default 4) to
``<dir>/s<pp rank>_r<tp rank>_send<n>.pt``: each kind's rows on the host in their own dtype, and under ``"__meta__"``
the ranks, the send's index and each kind's seam step. The seam's worker reads the slots between the forward and the
WRITE, so dumped sends are not timing-representative. Warm-up forwards send nothing."""

from __future__ import annotations

import os
from typing import Dict, Optional

import torch

DIR = "SGLANG_NEURON_PP_BOUNDARY_DUMP_DIR"
STEPS = "SGLANG_NEURON_PP_BOUNDARY_DUMP_STEPS"
META = "__meta__"


def send_count(environ=os.environ) -> int:
    """How many sends a rank writes; a value that is not a non-negative integer is refused by name."""
    from sglang_neuron import env_registry

    return env_registry.non_negative_int(STEPS, environ) if STEPS in environ else 4


class BoundaryDump:
    def __init__(self, directory: str, sends: int, pp_rank: int, tp_rank: int):
        self.directory, self.sends, self.pp_rank, self.tp_rank = directory, sends, pp_rank, tp_rank
        self.written = 0

    @classmethod
    def from_env(cls, pp_rank: int, tp_rank: int, environ=os.environ) -> Optional["BoundaryDump"]:
        """The rank's dump, or None when SGLANG_NEURON_PP_BOUNDARY_DUMP_DIR is unset."""
        directory = environ.get(DIR, "")
        return cls(directory, send_count(environ), pp_rank, tp_rank) if directory else None

    @property
    def done(self) -> bool:
        """Whether the rank has written its sends; the seam reads no more slots once it has."""
        return self.written >= self.sends

    def write(self, tensors: Dict[str, torch.Tensor], steps: Dict[str, int]) -> str:
        """Write one send's host tensors by kind; the file's path."""
        os.makedirs(self.directory, exist_ok=True)
        path = os.path.join(self.directory, f"s{self.pp_rank}_r{self.tp_rank}_send{self.written}.pt")
        torch.save({**tensors, META: {"pp_rank": self.pp_rank, "tp_rank": self.tp_rank, "send": self.written,
                                      "steps": dict(steps)}}, path)
        self.written += 1
        return path
