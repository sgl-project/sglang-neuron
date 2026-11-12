# SPDX-License-Identifier: Apache-2.0
"""A recorded-camera session's per-chunk DiT inputs, built and uploaded once at session open.

On torch-neuronx a host-to-device copy waits for the device's queued work, so an upload between
two chunks leaves the device idle while it completes. A session whose camera is known up front
(``camera_trajectory``, so its chunk count too) has every chunk's inputs but the condition known
at session open: the re-noise draws (the session generator's sequence, in the per-chunk draw
order), the camera ray parameters, the temporal RoPE factors and every pass's paged read plan (the
paging is a deterministic host simulation). They are built here and uploaded as one packed tensor
per graph-input slot; a chunk takes views. The slots are separate packs because one graph call
must not get two views of one buffer (Dynamo guards that its inputs do not alias). The condition
is uploaded per chunk until it repeats (the encoder history's fixed point), then reused.
"""

from __future__ import annotations

import copy

import torch

from sglang_neuron.diffusion.models.lingbot_world.chunk_prep import (
    camera_ray_params,
    lingbot_rope_temporal_factors,
)
from sglang_neuron.diffusion.models.lingbot_world.kv_pool import PagedPass, PagedSession


def _pack(rows: list[torch.Tensor], device) -> torch.Tensor:
    return torch.stack(rows).contiguous().to(device)


def _device_plan(host: PagedPass, block_table, bounds, update_idx) -> PagedPass:
    return PagedPass(
        block_table=block_table,
        bounds=bounds,
        kv_cache_update_idx=update_idx,
        num_history_blocks=host.num_history_blocks,
        history_full=host.history_full,
        host_bounds=host.host_bounds,
        commit_block_aligned=host.commit_block_aligned,
    )


class SessionPrefetch:
    """Every chunk's device inputs but the condition (see the module docstring)."""

    def __init__(
        self,
        *,
        chunks: int,
        frames_per_chunk: int,
        steps: int,
        latent_shape: tuple[int, ...],
        poses: torch.Tensor,
        intrinsics: torch.Tensor,
        generator: torch.Generator,
        paging: PagedSession,
        device,
    ):
        self.chunks = chunks
        # The noise: chunk by chunk, the initial latent then one re-noise per step (the order
        # the per-chunk path draws them in).
        draws = [
            [torch.randn(latent_shape, generator=generator, dtype=torch.float32) for _ in range(steps)]
            for _ in range(chunks)
        ]
        self.noise_sums = [float(torch.stack(d).double().sum()) for d in draws]
        self._noise = [_pack([d[k] for d in draws], device) for k in range(steps)]
        f = frames_per_chunk
        self._params = _pack(
            [camera_ray_params(poses[c * f : (c + 1) * f], intrinsics[c * f : (c + 1) * f]) for c in range(chunks)],
            device,
        )  # fmt: skip
        self._rope = _pack(
            [lingbot_rope_temporal_factors(f, start_frame=c * f) for c in range(chunks)], device
        )
        # The read plans: the session's paging simulated on the host (it is deterministic).
        sim = copy.deepcopy(paging)
        sim.device = torch.device("cpu")
        denoise, commit, self._blocks = [], [], []
        for _ in range(chunks):
            denoise.append(sim.paged_for(False))
            commit.append(sim.paged_for(True))
            self._blocks.append(list(sim.current))
            sim.commit()
        d_bt = _pack([p.block_table for p in denoise], device)
        d_bounds = _pack([p.bounds for p in denoise], device)
        c_bt = _pack([p.block_table for p in commit], device)
        c_bounds = _pack([p.bounds for p in commit], device)
        c_idx = _pack([p.kv_cache_update_idx for p in commit], device)
        self._denoise = [
            _device_plan(p, d_bt[c], d_bounds[c], None) for c, p in enumerate(denoise)
        ]
        self._commit = [
            _device_plan(p, c_bt[c], c_bounds[c], c_idx[c]) for c, p in enumerate(commit)
        ]
        self._device = device
        self._condition: tuple[torch.Tensor, torch.Tensor] | None = None  # (host, device)

    def covers(self, chunk: int) -> bool:
        return 0 <= chunk < self.chunks

    def current_blocks(self, chunk: int) -> list[int]:
        """The pool blocks the simulated paging gave ``chunk`` (the session's must match)."""
        return self._blocks[chunk]

    def plan(self, chunk: int, commit: bool) -> PagedPass:
        return (self._commit if commit else self._denoise)[chunk]

    def params(self, chunk: int) -> torch.Tensor:
        return self._params[chunk]

    def rope(self, chunk: int) -> torch.Tensor:
        return self._rope[chunk]

    def noise(self, chunk: int) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """``(initial latent, re-noise draws)`` of ``chunk`` on the device."""
        return self._noise[0][chunk], tuple(pack[chunk] for pack in self._noise[1:])

    def condition(self, host: torch.Tensor) -> torch.Tensor:
        """``host`` on the device: the previous upload when it is the same condition."""
        if self._condition is not None and torch.equal(self._condition[0], host):
            return self._condition[1]
        dev = host.float().contiguous().to(self._device)
        self._condition = (host, dev)
        return dev
