# SPDX-License-Identifier: Apache-2.0
"""The ENCODE and DECODE roles of LingBot-World, run by the role workers (:mod:`.role_workers`).

A role runs in the ``torch.distributed`` world its host has set up and shards over
``dist.group.WORLD``. ENCODE is UMT5 (tensor-parallel over the world) and the streaming Wan
condition encoder; DECODE is the exact streaming VAE decode, which keeps each session's causal
decoder state across chunks. Both VAE directions are spatially sharded on the
``lingbot_vae_grid`` ``n_h x n_w`` grid and share one VAE. Inputs and outputs are
:mod:`.payload` dicts.
"""

from __future__ import annotations

import itertools
import os
import time
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from sglang_neuron.diffusion.distributed.autoencoders.wan_spatial_shard import (
    DECODER_COMPILER_ARGS,
    ENCODER_COMPILER_ARGS,
)
from sglang_neuron.diffusion.models.lingbot_world import dist as lingbot_dist
from sglang_neuron.diffusion.models.lingbot_world import payload as P

#: neuronx-cc flags of the UMT5 graph (the Wan pipeline's text-encoder set).
TEXT_COMPILER_ARGS = (
    "--model-type=transformer",
    "--auto-cast=none",
    "-O1",
    "--hbm-scratchpad-page-size=2048",
    "--internal-hlo2tensorizer-options=--hierarchy-splitter=false",
)


def _compile_backend():
    from sglang_neuron.diffusion.compile_backend import register as get_compile_backend_name

    return get_compile_backend_name()


def _world():
    """The role's group and world size (``None``, 1 without a world)."""
    if dist.is_available() and dist.is_initialized():
        return dist.group.WORLD, dist.get_world_size()
    return None, 1


def _grid(model_config: dict, world: int) -> tuple[int, int]:
    n_h, n_w = tuple(model_config["lingbot_vae_grid"])
    if n_h * n_w != world:
        raise ValueError(f"VAE grid {n_h}x{n_w} needs {n_h * n_w} ranks, have {world}")
    return n_h, n_w


def _agreed(flag: bool, *, rank0: bool) -> bool:
    """Rank 0's ``flag`` on every rank (``rank0``), else whether it holds on every rank."""
    value = torch.tensor([int(flag)], dtype=torch.int32)
    if rank0:
        return bool(lingbot_dist.broadcast_cpu_from_rank0(value, (1,), torch.int32)[0])
    group = lingbot_dist.cpu_group()
    if group is not None:
        dist.all_reduce(value, op=dist.ReduceOp.MIN, group=group)
    return bool(value[0])


def load_wan_vae(model_path: str, device):
    """The checkpoint's Wan VAE on ``device`` (bf16, no tiling)."""
    from sglang_neuron.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
        NeuronAutoencoderKLWan,
    )

    vae = NeuronAutoencoderKLWan.from_pretrained(
        model_path, subfolder="vae", torch_dtype=torch.bfloat16
    )
    vae = vae.eval().to(device)
    vae.use_tiling = False
    return vae


def uint8_frames(video: torch.Tensor) -> np.ndarray:
    """One decoded block ``(1, 3, F, H, W)`` in ``[-1, 1]`` as ``(F, H, W, 3)`` uint8 on the host
    (``VideoProcessor.postprocess_video``'s arithmetic: denormalise and clamp in the decoder
    dtype, then widen, scale by 255 and round)."""
    frames = (video[0] / 2 + 0.5).clamp(0, 1).permute(1, 2, 3, 0)
    frames = frames.float().mul_(255).round_().to(torch.uint8)
    return np.ascontiguousarray(frames.cpu().numpy())


class RoleGeometry:
    """What both roles need of the checkpoint and the session geometry."""

    def __init__(
        self, *, model_path: str, height: int, width: int, frames_per_chunk: int, device
    ) -> None:
        self.model_path = model_path
        self.height, self.width = int(height), int(width)
        self.frames_per_chunk = int(frames_per_chunk)
        self.device = device

    @property
    def latent_hw(self) -> tuple[int, int]:
        return self.height // 8, self.width // 8


class LingBotEncodeRole:
    """ENCODE: UMT5 and the streaming condition encoder (on ``vae``: the decoder's, if given)."""

    def __init__(
        self, geometry: RoleGeometry, model_config: dict, *, compile_graphs: bool, vae=None
    ) -> None:
        self.geometry = geometry
        self.device = geometry.device
        self._model_path = geometry.model_path
        self._frames_per_chunk = geometry.frames_per_chunk
        group, world = _world()
        self.sharded_encoder = None
        self._build_text_encoder(group, world, compile_graphs=compile_graphs)
        self._build_condition_encoder(
            group, world, _grid(model_config, world), vae=vae, compile_graphs=compile_graphs
        )
        self._zeros: dict[tuple, Any] = {}
        if compile_graphs:
            # UMT5, and the opening (1 frame) and continuation (4 frames) encoder graphs.
            ids = torch.zeros(1, P.MAX_SEQUENCE_LENGTH, dtype=torch.long)
            self.encode_prompt_payload("__warmup__", ids, torch.ones_like(ids))
            image = torch.zeros(1, 3, 1, geometry.height, geometry.width)
            for _ in self.encode_condition_steps("__warmup__", image, lambda p: None, chunks=2):
                pass

    # -- construction (overridable: the CPU tests build stand-ins) -------------------------

    def _build_text_encoder(self, group, world, *, compile_graphs: bool) -> None:
        from sglang_neuron.diffusion.models.umt5_encoder.umt5_encoder import (
            NeuronTextEncoderWrapper,
        )

        encoder = NeuronTextEncoderWrapper(
            model_path=self._model_path, dtype=torch.bfloat16, tp_group=group if world > 1 else None
        )
        encoder.load_weights(os.path.join(self._model_path, "text_encoder"))
        if compile_graphs:
            encoder.compile(
                backend=_compile_backend(),
                fullgraph=True,
                dynamic=False,
                options={
                    "model_name": "lingbot_text_encoder",
                    "compiler_args": list(TEXT_COMPILER_ARGS),
                },
            )
        self.text_encoder = encoder.to(self.device)

    def _build_condition_encoder(self, group, world, grid, *, vae, compile_graphs) -> None:
        # The decoder's VAE when given: only the encoder graphs are added.
        self.condition_vae = vae if vae is not None else load_wan_vae(self._model_path, self.device)
        compile_kwargs = {
            "backend": _compile_backend() if compile_graphs else None,
            "fullgraph": True,
            "dynamic": False,
            "options": {"compiler_args": list(ENCODER_COMPILER_ARGS)},
        }
        from sglang_neuron.diffusion.distributed.autoencoders.wan_spatial_shard import (
            SpatialShardedWanEncoder,
        )

        self.sharded_encoder = SpatialShardedWanEncoder(self.condition_vae, group, *grid)
        if compile_graphs:
            self.sharded_encoder.compile(**compile_kwargs)

    # -- E: the prompt -------------------------------------------------------------------

    def encode_prompt_payload(self, request_id: str, input_ids, attention_mask):
        """UMT5 of one prompt: the E -> G prompt payload on rank 0 (``None`` elsewhere)."""
        t0 = time.perf_counter()
        out = self.text_encoder(input_ids.to(self.device), attention_mask.to(self.device))
        if not lingbot_dist.is_output_rank():
            return None
        embeds = out.last_hidden_state.cpu()
        meta = P.chunk_metadata(request_id, P.PROMPT_CHUNK, seconds=time.perf_counter() - t0)
        return {P.PROMPT_EMBEDS: embeds, P.CONDITIONING_METADATA: meta}

    # -- E: the image condition ----------------------------------------------------------

    def _encode(self, chunk: int, image, state):
        """Chunk ``chunk``'s posterior moments: ``[image, zeros]`` first, then zero frames
        (each rank's haloed block of them, kept on the device)."""
        vae, sharded = self.condition_vae, self.sharded_encoder
        t = int(getattr(vae.config, "scale_factor_temporal", 4))
        height, width = image.shape[-2:]
        if chunk == 0:
            zeros = torch.zeros(1, 3, t * (self._frames_per_chunk - 1), height, width)
            x = torch.cat((image.float(), zeros), dim=2).to(vae.dtype)
        else:
            key = (t * self._frames_per_chunk, *sharded.input_extent(height, width))
            if key not in self._zeros:
                self._zeros[key] = torch.zeros(1, 3, *key, dtype=vae.dtype).to(self.device)
            x = self._zeros[key]
        return sharded.encode_chunk(x, state, block=chunk > 0, full_size=(height, width))

    def encode_condition_steps(
        self,
        request_id: str,
        image,
        emit: Callable[[dict], None],
        stop: Callable[[], bool] | None = None,
        *,
        chunks: int | None = None,
    ) -> Iterator[int]:
        """Encode ``[image, zeros, ...]`` chunk by chunk until the encoder history is a fixed point.

        A generator that yields after every chunk, so a host can interleave decodes (every
        rank resumes it in lockstep). ``emit(payload)`` gets each chunk's E -> G payload on
        rank 0: the posterior-mode latent ``[1, z_dim, F, h, w]`` (CPU fp32) and metadata whose
        ``fixed`` marks the chunk whose returned history equals the one it was given (every
        later chunk's condition equals it). ``chunks`` (the session's chunk count) bounds the
        sequence: past it no chunk is read, so a history that never settles costs no more
        than the session. ``stop()`` (rank 0's answer) ends it before the next chunk.
        """
        state = self.sharded_encoder.new_encode_state(request_id)
        previous = previous_latent = None
        for chunk in range(chunks) if chunks is not None else itertools.count():
            if chunk > 0 and stop is not None and _agreed(stop(), rank0=True):
                break
            t0 = time.perf_counter()
            moments = self._encode(chunk, image, state)
            latent = moments.chunk(2, dim=1)[0].cpu().float()
            seconds = time.perf_counter() - t0
            # The fixed point is a returned history bit-identical to the one given. Copying the
            # history to the host is costly, so it is only compared once two consecutive
            # chunks' conditions agree (a necessary condition).
            fixed = False
            if previous_latent is not None and torch.equal(latent, previous_latent):
                history = [f.cpu() for f in state.feat_map]
                same = previous is not None and all(
                    torch.equal(a, b) for a, b in zip(previous, history, strict=True)
                )
                fixed = _agreed(same, rank0=False)
                previous = history
            else:
                previous = None
            if lingbot_dist.is_output_rank():
                meta = P.chunk_metadata(request_id, chunk, last=fixed, fixed=fixed, seconds=seconds)
                emit({P.IMAGE_CONDITION: latent, P.CONDITIONING_METADATA: meta})
            if fixed:
                break
            previous_latent = latent
            yield chunk
        state.release()


class LingBotDecodeRole:
    """DECODE: the exact spatially-sharded streaming VAE decode."""

    def __init__(self, geometry: RoleGeometry, model_config: dict, *, compile_graphs: bool) -> None:
        self.geometry = geometry
        self.device = geometry.device
        self._model_path = geometry.model_path
        self._frames_per_chunk = geometry.frames_per_chunk
        group, world = _world()
        self.vae = self._build_vae()
        self.decoder = self._build_decoder(
            self.vae, group, _grid(model_config, world), compile_graphs=compile_graphs
        )
        self.decode_states: dict[str, Any] = {}
        if compile_graphs:
            # The opening-frame and steady-state graphs (two chunks).
            state = self.decoder.new_decode_state("__warmup__")
            latents = torch.zeros(1, 16, self._frames_per_chunk, *geometry.latent_hw)
            for _ in range(2):
                self.decoder.decode_chunk(latents.bfloat16(), state)
            state.release()

    def _build_vae(self):
        return load_wan_vae(self._model_path, self.device)

    def _build_decoder(self, vae, group, grid, *, compile_graphs: bool):
        from sglang_neuron.diffusion.distributed.autoencoders.wan_spatial_shard import (
            SpatialShardedWanDecoder,
        )

        decoder = SpatialShardedWanDecoder(vae, group, *grid)
        if compile_graphs:
            decoder.compile(
                backend=_compile_backend(),
                fullgraph=True,
                dynamic=False,
                options={"compiler_args": list(DECODER_COMPILER_ARGS)},
            )
        return decoder

    def decode_payload(self, payload: dict) -> dict | None:
        """One G -> D chunk payload -> its D output (rank 0; ``None`` elsewhere).

        A session's chunks arrive in order: its decoder state is kept here across chunks.
        """
        meta = payload[P.LATENTS_METADATA]
        session_id, chunk = str(meta["session_id"]), meta["chunk_index"]
        state = self.decode_states.get(session_id)
        if state is None:
            state = self.decode_states[session_id] = self.decoder.new_decode_state(session_id)
        out = self.decoder.decode_chunk(payload[P.LATENTS].to(torch.bfloat16), state)
        if not lingbot_dist.is_output_rank():
            return None
        seconds = float(state.chunk_seconds[-1])
        return {
            P.VIDEO: uint8_frames(out),
            P.VIDEO_METADATA: P.chunk_metadata(
                session_id, chunk, last=bool(meta.get("last", False)), seconds=seconds
            ),
        }

    def release_decode_session(self, session_id: str) -> None:
        state = self.decode_states.pop(session_id, None)
        if state is not None:
            state.release()
