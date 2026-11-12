# SPDX-License-Identifier: Apache-2.0
"""Rank helpers over SGLang Diffusion's world group: the output rank, host broadcasts and rank 0's
fatal-error exit."""

from __future__ import annotations

import contextlib
import os
import sys
import traceback
from collections.abc import Callable

import torch
import torch.distributed as dist


#: Called by :func:`fatal_on_output_rank` before rank 0 exits (see :func:`on_fatal`).
_ON_FATAL: Callable[[], None] | None = None


def on_fatal(stop: Callable[[], None] | None) -> None:
    """Have :func:`fatal_on_output_rank` call ``stop()`` before rank 0 exits. The serving pipeline
    passes one that stops the SGLang server, whose workers are watched only while they start:
    its clients then see their connections end instead of waiting on a dead rank 0."""
    global _ON_FATAL
    _ON_FATAL = stop


def is_output_rank() -> bool:
    """Whether this is the world's rank 0 (or there is no world): the rank that returns outputs."""
    return not dist.is_initialized() or dist.get_rank() == 0


def cpu_group():
    """The world group's CPU (Gloo) group; ``None`` without a world of more than one rank."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return None
    from sglang_neuron.diffusion.distributed.groups import get_world_group

    return get_world_group().cpu_group


def broadcast_cpu_from_rank0(tensor: torch.Tensor | None, shape, dtype) -> torch.Tensor:
    """Rank 0's host ``tensor`` (``shape``, ``dtype``) on every rank over the CPU group.

    Other ranks pass ``None`` (or anything: only rank 0's value is sent). Gloo has no bf16, so
    bf16 travels as fp32, which is exact.
    """
    group = cpu_group()
    if group is None:
        assert tensor is not None
        return tensor
    wire = torch.float32 if dtype == torch.bfloat16 else dtype
    if is_output_rank():
        buf = tensor.detach().to("cpu").to(wire).contiguous()
    else:
        buf = torch.empty(tuple(shape), dtype=wire)
    dist.broadcast(buf, src=0, group=group)
    return buf.to(dtype) if wire != dtype else buf


@contextlib.contextmanager
def fatal_on_output_rank(what: str):
    """Run rank-0-only work; in a world of more than one rank, a failure exits the process.

    A raised error would fail only rank 0's request while the other ranks wait in their next
    collective for it; exiting ends the worker instead of leaving the world hung. Without a world
    the error is raised as is, and so is a non-``Exception`` (``SystemExit``,
    ``KeyboardInterrupt``).
    """
    try:
        yield
    except Exception:
        if not dist.is_initialized() or dist.get_world_size() == 1:
            raise
        print(
            f"[lingbot] fatal: {what} failed on rank 0; exiting so the engine fails\n"
            f"{traceback.format_exc()}",
            file=sys.stderr,
            flush=True,
        )
        if _ON_FATAL is not None:
            with contextlib.suppress(Exception):
                _ON_FATAL()
        os._exit(1)
