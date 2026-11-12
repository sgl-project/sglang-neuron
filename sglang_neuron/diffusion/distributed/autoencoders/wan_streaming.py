# SPDX-License-Identifier: Apache-2.0
"""Session-owned causal-cache state for streaming Wan VAE decode and encode.

The Wan VAE is causal in time: each causal convolution reads the last
``CACHE_T`` frames of its own input from a per-layer cache. A whole-clip
``decode()`` builds that cache, threads it through every latent frame and drops
it. Real-time generation decodes one chunk of latent frames per autoregressive
block, so the cache has to outlive the call: chunk ``N + 1`` must continue chunk
``N`` exactly, or the stream shows a seam at every block.

These classes hold that cache for one session, with the lifecycle new state ->
decode (or encode) chunks -> ``release``. The cache is the fixed-shape, zero-initialised,
device-resident tensor list the compiled decoder graphs take and return (see
``NeuronAutoencoderKLWan._init_feat_cache``), not diffusers' ``None``/``"Rep"``
sentinel list. That keeps every chunk on the same two compiled graphs (opening
frame and steady state).

Resident size is bounded by construction: every slot holds at most ``CACHE_T``
frames at a fixed resolution, so the state is a function of the resolution and
never of the session length.

The cache has no recompute source -- it is the compressed tail of every frame
decoded so far -- so releasing it ends the stream.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class _WanCausalCacheState:
    session_id: str
    # One fixed-shape tensor per causal-cache slot, or None before the first chunk
    # (the shapes are only known once the first chunk's resolution is seen).
    feat_map: list[torch.Tensor] | None = None
    # Spatial size the cache was allocated for; later chunks must match it.
    spatial_shape: tuple[int, int] | None = None
    chunks_done: int = 0
    # Per-chunk wall time in seconds, appended by the VAE. Host-side bookkeeping only.
    chunk_seconds: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise ValueError("session_id must be a non-empty string.")

    def check_spatial_shape(self, shape: tuple[int, int]) -> None:
        """Reject a chunk whose ``(height, width)`` differs from the session's."""
        if self.spatial_shape is not None and tuple(shape) != self.spatial_shape:
            raise ValueError(
                f"session {self.session_id!r} was started at spatial size {self.spatial_shape}; "
                f"got a chunk at {tuple(shape)}. Start a new session to change resolution."
            )


@dataclass
class WanStreamingDecodeState(_WanCausalCacheState):
    """One session's decoder cache.

    ``frames_decoded`` counts *latent* frames. The session's first latent frame
    decodes to one pixel frame and every later one to four (Wan's 4x temporal
    factor), so a session's first 3-frame chunk yields 9 pixel frames and every
    later chunk 12 -- the same layout as a whole-clip decode, ``1 + 4 * (T - 1)``.
    """

    frames_decoded: int = 0

    def release(self) -> None:
        """Drop the cache and return the session to its pre-stream state."""
        self.feat_map = None
        self.spatial_shape = None
        self.frames_decoded = 0
        self.chunks_done = 0
        self.chunk_seconds = []


@dataclass
class WanStreamingEncodeState(_WanCausalCacheState):
    """One session's encoder cache.

    ``frames_encoded`` counts *pixel* frames. The session's first pixel frame
    encodes alone to one latent frame; after it every four pixel frames encode
    to one latent frame, so a chunk's pixel count must be ``1 + 4k`` on the
    session's first chunk and ``4k`` afterwards.
    """

    frames_encoded: int = 0

    def release(self) -> None:
        """Drop the cache and return the session to its pre-stream state."""
        self.feat_map = None
        self.spatial_shape = None
        self.frames_encoded = 0
        self.chunks_done = 0
        self.chunk_seconds = []
