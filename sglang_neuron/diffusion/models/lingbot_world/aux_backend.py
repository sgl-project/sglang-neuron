# SPDX-License-Identifier: Apache-2.0
"""The DiT's boundary to LingBot's ENCODE and DECODE roles.

The DiT calls only this protocol, on DiT rank 0; it broadcasts what the other DiT ranks need
itself. Where the roles run is the backend's business: :class:`.role_workers.LingBotRoleWorkers`
hosts them in a Neuron world beside the DiT (started by DiT rank 0); the device smoke test
substitutes a stand-in.

Standard library only.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LingBotAuxBackend(Protocol):
    def poll_ready(self, timeout: float) -> bool:
        """True once every role instance has loaded, compiled and warmed up."""

    def submit_encode(
        self, request_id: str, *, image=None, prompt_tokens=None, chunks: int | None = None
    ) -> None:
        """Start ENCODE work for a session: the condition sequence of ``image``
        (``[1, 3, 1, H, W]`` in [-1, 1]; at most ``chunks`` chunks) and / or the prompt
        ``(input_ids, attention_mask)``."""

    def prompt_embeds(self, request_id: str) -> Any:
        """The session's ``prompt_embeds`` (waits for the prompt payload)."""

    def condition(self, request_id: str, chunk: int) -> Any:
        """Chunk ``chunk``'s image condition (CPU fp32; waits). A chunk past the fixed point gets the fixed point's."""

    def submit_latents(self, request_id: str, chunk: int, latents, *, last: bool) -> None:
        """Hand a committed chunk's de-normalized latents to DECODE (chunks in order)."""

    def collect_frames(self, request_id: str, chunk: int) -> Any:
        """A chunk's ``(F, H, W, 3)`` uint8 frames (waits for its decode)."""

    def role_timings(self, request_id: str, chunk: int) -> dict:
        """The roles' own times (ms) for the chunk, each reported once (``prompt_encode_ms``,
        ``condition_encode_ms``, ``vae_decode_ms``; what is known)."""

    def release(self, request_id: str) -> None:
        """Drop the session's encode and decode state (an unfinished encode is abandoned)."""

    def shutdown(self) -> None:
        """Stop every role instance."""
