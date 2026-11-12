# SPDX-License-Identifier: Apache-2.0
"""LingBot-World-V2 real-time generation on Trainium: the model behind the SGLang stage.

One :class:`NeuronLingBotWorldModel` per DiT rank (TP8 x CP8 at LNC 1, 64 ranks): the DiT
(:class:`.lingbot_transformer.LingBotTransformer3DModel`) runs through
:class:`.dit_runtime.LingBotDiTRuntime`'s static graphs over the paged K/V pool
(:mod:`.kv_pool`); DiT rank 0 starts and drives the role workers (:mod:`.role_workers`: UMT5,
the streaming condition encoder and the streaming VAE decode on cores 64-79).

Each SGLang request is one chunk of a real-time session (``block_idx``). The chunk's four
denoise passes and its commit pass are launched back to back (each pass graph starts with the
previous step's DMD update), the chunk is synced once, its latents go to the decoder, and the
request returns decoded frames:

* the session's camera is known ahead (a recorded ``camera_trajectory``): the next chunk's prep
  and first denoise passes are launched before this chunk's decode is awaited, and the request
  returns this chunk's frames (early emission);
* the camera arrives with each request (``camera_actions``): the next chunk cannot start before
  its request, so the request waits for this chunk's decode and returns its frames.

As in SGLang's GPU pipeline, a chunk request without ``camera_actions`` has no camera condition
(the DiT applies no camera modulation), and ``lingbot_prompt_updated`` switches the session's
text conditioning to the request's prompt from that chunk on.

Every DiT rank runs the same launches in the same order: the decisions depend only on the
request, the session's host state and the in-flight budget, which are the same on every rank.
Rank 0's role-worker outputs (prompt embedding, condition latents) are broadcast over Gloo.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import OrderedDict

import msgspec
import numpy as np
import torch
import torch.nn.functional as F

from sglang_neuron.diffusion.models.lingbot_world import camera as C
from sglang_neuron.diffusion.models.lingbot_world.aux_backend import LingBotAuxBackend
from sglang_neuron.diffusion.models.lingbot_world.chunk_prep import assert_static_resolution
from sglang_neuron.diffusion.models.lingbot_world.dist import (
    broadcast_cpu_from_rank0,
    fatal_on_output_rank,
    is_output_rank,
)
from sglang_neuron.diffusion.models.lingbot_world.dit_runtime import (
    DEFAULT_MAX_INFLIGHT,
    LingBotDiTRuntime,
)
from sglang_neuron.diffusion.models.lingbot_world.kv_pool import (
    PagedPool,
    PagedSession,
    PagingGeometry,
)
from sglang_neuron.diffusion.models.lingbot_world.payload import (
    CAMERA_ACTIONS,
    CAMERA_TRAJECTORY,
    MAX_SEQUENCE_LENGTH,
    PROMPT_UPDATED,
    TOTAL_CHUNKS,
    tokenize,
)

logger = logging.getLogger(__name__)

#: LingBot-World-V2's DMD labels (the checkpoint's 4-step schedule before the flow shift).
LINGBOT_DMD_TIMESTEPS = (1000, 750, 500, 250)
#: The checkpoint's flow shift (``LingBotWorldV2CausalDMDConfig``).
LINGBOT_FLOW_SHIFT = 5.0
#: Chunks of a recorded-camera session whose inputs are prefetched at session open (device
#: memory per chunk and rank, mostly the noise draws); later chunks upload their own.
PREFETCH_MAX_CHUNKS = 64
#: Warm-up chunks at load: chunk 0 compiles the graphs of the filling history (chunks 1-4 reuse
#: them), chunk 5 is the first with every listed history frame visible (the steady-state twins).
WARMUP_CHUNKS = 6
#: Prompts whose embedding and text K/V are kept.
_PROMPT_CACHE_SIZE = 4



def build_shifted_flow_schedule(flow_shift: float) -> tuple[tuple[float, float], ...]:
    """``(timestep, sigma)`` per step: the DMD labels warped by the flow shift (vLLM-Omni's
    ``_build_shifted_flow_schedule``)."""
    if not (math.isfinite(flow_shift) and flow_shift > 0):
        raise ValueError(f"flow_shift must be a positive finite number, got {flow_shift}")
    base = torch.tensor(LINGBOT_DMD_TIMESTEPS, dtype=torch.float64) / 1000
    shifted = flow_shift * base / ((1.0 - base) + flow_shift * base)
    return tuple(
        (float(t), float(s)) for t, s in zip((shifted * 1000).tolist(), shifted.tolist(), strict=True)
    )


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1e3, 2)


def _lru(cache: OrderedDict, key, make, size: int = _PROMPT_CACHE_SIZE):
    """``cache[key]``, made on a miss; the cache keeps the last ``size`` keys."""
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    if size == 1:
        cache.clear()  # release the old entry before making the new one
    value = cache[key] = make()
    while len(cache) > size:
        cache.popitem(last=False)
    return value


def log_hbm(stage: str) -> None:
    """This rank's device memory (allocator counters), on DiT rank 0."""
    if not is_output_rank():
        return
    gib = 2**30
    logger.info(
        "[neuron-hbm] %s: allocated %.2f GiB (peak %.2f), reserved %.2f GiB",
        stage,
        torch.neuron.memory_allocated() / gib,
        torch.neuron.max_memory_allocated() / gib,
        torch.neuron.memory_reserved() / gib,
    )


def _load_json(model_path: str, name: str) -> dict:
    with open(os.path.join(model_path, name)) as f:
        return json.load(f)


def source_image(path: str, *, height: int, width: int) -> torch.Tensor:
    """``[1, 3, 1, H, W]`` in [-1, 1] on the host: the RGB image, bicubic-resized in float32
    (vLLM-Omni's ``_prepare_image_tensor``)."""
    import PIL.Image

    with PIL.Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0) / 255.0
    tensor = tensor * 2.0 - 1.0
    if tensor.shape[-2:] != (height, width):
        tensor = F.interpolate(tensor, size=(height, width), mode="bicubic", align_corners=False)
    if tuple(tensor.shape) != (1, 3, height, width):
        raise ValueError(f"LingBot source image must be RGB {height}x{width}, got {tuple(tensor.shape)}")
    return tensor.unsqueeze(2)


class ChunkCamera(msgspec.Struct, frozen=True):
    """A chunk's prepared camera (float64): ``poses [F, 4, 4]``, ``intrinsics [F, 4]``.

    ``conditioned=False``: the chunk has no camera condition (no ``camera_actions``); the DiT
    then applies no camera modulation, as SGLang's pipeline skips the camera injection.
    """

    poses: torch.Tensor
    intrinsics: torch.Tensor
    conditioned: bool = True


class _Started(msgspec.Struct):
    """A chunk whose prep and first ``steps`` denoise passes are already launched (and, when
    ``done`` is set, its commit pass: ``done`` / ``x0`` are its outputs)."""

    prep: object
    x_t: torch.Tensor | None
    flow: torch.Tensor | None
    steps: int
    commit: object = None
    done: torch.Tensor | None = None
    x0: torch.Tensor | None = None


class SessionState:
    """One real-time session on one DiT rank (host state; the history is in the K/V pool)."""

    def __init__(self, *, session_id: str, paging: PagedSession, generator: torch.Generator):
        self.session_id = session_id
        self.paging = paging
        self.generator = generator
        self.checksums: dict[int, dict] = {}
        self.next_chunk = 0
        self.total_chunks: int | None = None
        self.text_kv: tuple | None = None
        self.constants = None
        self.trajectory: ChunkCamera | None = None  # the whole session's, when recorded
        self.action_history: list[list[str]] = []
        self.started: dict[int, _Started] = {}
        self.prefetch = None  # SessionPrefetch: a recorded-camera session's uploaded inputs


class NeuronLingBotWorldModel:
    """LingBot-World-V2 on one DiT rank (see the module docstring)."""

    def __init__(
        self,
        model_path: str,
        *,
        height: int = 480,
        width: int = 832,
        num_layers: int | None = None,
        max_inflight: int = DEFAULT_MAX_INFLIGHT,
        timing_file: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        from transformers import AutoTokenizer

        from sglang_neuron.diffusion.models.lingbot_world.lingbot_transformer import (
            LingBotTransformer3DModel,
        )
        self.model_path = model_path
        self.height, self.width = int(height), int(width)
        vae_config = _load_json(model_path, "vae/config.json")
        self.vae_temporal = int(vae_config.get("scale_factor_temporal", 4))
        self.vae_spatial = int(vae_config.get("scale_factor_spatial", 8))
        self.latents_mean = torch.tensor(vae_config["latents_mean"]).view(1, -1, 1, 1, 1)
        self.latents_std = torch.tensor(vae_config["latents_std"]).view(1, -1, 1, 1, 1)
        transformer_config = _load_json(model_path, "transformer/config.json")
        self.frames_per_chunk = int(transformer_config.get("num_frames_per_block", 3))
        # The in-graph ray builder needs whole patches of whole 8x8 pixel cells.
        assert_static_resolution(
            self.height, self.width, tuple(transformer_config.get("patch_size", (1, 2, 2)))
        )
        self.latent_hw = (self.height // self.vae_spatial, self.width // self.vae_spatial)
        self.transformer = LingBotTransformer3DModel.from_config(
            transformer_config, num_layers=num_layers, latent_size=self.latent_hw
        ).to(dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        scheduler_config = _load_json(model_path, "scheduler/scheduler_config.json")
        shift = float(scheduler_config.get("flow_shift", scheduler_config.get("shift", LINGBOT_FLOW_SHIFT)))
        self.schedule = build_shifted_flow_schedule(shift)
        self.max_inflight = int(max_inflight)
        self.timing_file = timing_file
        self.role_world = None  # set by start_role_workers
        _, p_h, p_w = self.transformer.config.patch_size
        self.paging = PagingGeometry(
            tokens_per_frame=(self.latent_hw[0] // p_h) * (self.latent_hw[1] // p_w),
            frames_per_chunk=self.frames_per_chunk,
        )
        self.device = None
        self.runtime: LingBotDiTRuntime | None = None
        self.pool: PagedPool | None = None
        self.aux: LingBotAuxBackend | None = None
        self._role_processes = None
        self._prompt_cache: OrderedDict = OrderedDict()
        self._text_kv_cache: OrderedDict = OrderedDict()
        self.sessions: dict[str, SessionState] = {}

    # -- load ---------------------------------------------------------------------------

    @property
    def is_output_rank(self) -> bool:
        return is_output_rank()

    def load_weights(self) -> None:
        self.transformer.load_weights(os.path.join(self.model_path, "transformer"))
        # Inference only: a parameter that requires grad makes every compiled graph an autograd
        # graph, which keeps its activations on the device for a backward pass.
        self.transformer.requires_grad_(False)

    def to(self, device) -> NeuronLingBotWorldModel:
        self.device = torch.device(device)
        self.transformer.to(self.device)
        return self

    def start_role_workers(self) -> None:
        """DiT rank 0: start the role-worker world (it loads and compiles beside the DiT)."""
        if not self.is_output_rank:
            return
        from sglang_neuron.diffusion.models.lingbot_world.role_workers import (
            LingBotRoleWorkers,
            RoleWorkerProcesses,
            role_world,
        )

        geometry = {"ar_diffusion_height": self.height, "ar_diffusion_width": self.width}
        self.role_world = role_world(geometry=geometry)
        self._role_processes = RoleWorkerProcesses(self.role_world, self.model_path)
        self.aux = LingBotRoleWorkers(
            self.role_world,
            self._role_processes.channels,
            processes=self._role_processes.processes,
        )

    @torch.inference_mode()
    def compile(self) -> None:
        """Build the DiT runtime and the K/V pool, then run the warm-up rollout (every graph
        compiles at load) and wait for the role workers.

        Under ``torch.inference_mode()``, the mode SGLang's executor runs the pipeline's stages
        in (``NeuronDiffusionPlatform.inference_mode``): Dynamo guards on it, so a warm-up in any other
        grad mode would leave every graph to retrace on the first served chunk."""
        model = self.transformer
        log_hbm("DiT weights")
        self.runtime = LingBotDiTRuntime(
            model,
            device=self.device,
            latent_height=self.latent_hw[0],
            latent_width=self.latent_hw[1],
            frames_per_chunk=self.frames_per_chunk,
            compile_graphs=True,
            max_inflight=self.max_inflight,
        )
        self.pool = PagedPool(
            num_blocks=self.paging.blocks_per_session,
            num_layers=model.num_layers,
            heads=int(model.num_local_heads),
            head_dim=int(model.head_dim),
            block=self.paging.block,
            device=self.device,
            dtype=model.dtype,
        )
        self.runtime.bind_kv_cache(self.pool.flat())
        log_hbm("K/V pool")
        if self.is_output_rank:
            logger.info(
                "LingBot K/V pool: %d blocks of %d tokens, %.2f GiB per rank",
                self.pool.num_blocks,
                self.paging.block,
                self.pool.nbytes / 2**30,
            )
        t0 = time.perf_counter()
        self._warmup_rollout()
        log_hbm("DiT warm-up")
        if self.is_output_rank:
            logger.info(
                "LingBot DiT warm-up (%d chunks, %d graphs) in %.1f s",
                WARMUP_CHUNKS,
                len(self.runtime.graph_keys()),
                time.perf_counter() - t0,
            )
        self._await_role_workers()

    def _new_paging(self) -> PagedSession:
        return PagedSession(
            self.paging,
            num_blocks=self.pool.num_blocks,
            heads=int(self.transformer.num_local_heads),
            device=self.device,
        )

    def _warmup_rollout(self) -> None:
        """A synthetic rollout into the steady state through the serving graphs and calls."""
        rt = self.runtime
        frames = self.frames_per_chunk
        h, w = self.latent_hw
        paging = self._new_paging()
        hidden = int(self.transformer.text_embedding[0].in_features)
        embeds = torch.zeros(1, MAX_SEQUENCE_LENGTH, hidden, dtype=self.transformer.dtype)
        text_kv = rt.text_kv(embeds)
        constants = rt.session_constants([t for t, _ in self.schedule])
        log_hbm("text K/V")
        out_channels = int(self.transformer.config.out_channels)
        poses = torch.eye(4, dtype=torch.float64).repeat(frames, 1, 1)
        intrinsics = torch.tensor(
            [[0.6 * self.width, 0.6 * self.width, self.width / 2, self.height / 2]],
            dtype=torch.float64,
        ).repeat(frames, 1)
        for chunk in range(WARMUP_CHUNKS):
            g = torch.Generator().manual_seed(1234 + chunk)
            noise = torch.randn(len(self.schedule), 1, out_channels, frames, h, w, generator=g)
            prep = rt.launch_prep(
                start_frame=chunk * frames,
                poses=poses,
                intrinsics=intrinsics,
                noise=noise,
                condition=torch.zeros(1, 20, frames, h, w),
            )
            _, done = rt.run_chunk(prep, paging.paged_for, text_kv, constants)
            rt.synchronize(done)
            paging.commit()
            if chunk == 0:
                log_hbm("warm-up chunk 0")

    def _await_role_workers(self, poll_seconds: float = 10.0) -> None:
        """Hold every DiT rank until rank 0's role workers are ready, meeting in a short
        broadcast every ``poll_seconds`` so no collective waits long on rank 0."""
        t0 = time.perf_counter()
        while True:
            ready = True
            if self.is_output_rank and self.aux is not None:
                with fatal_on_output_rank("the role workers' start"):
                    ready = bool(self.aux.poll_ready(poll_seconds))
            flag = broadcast_cpu_from_rank0(torch.tensor([int(ready)]), (1,), torch.int64)
            if int(flag[0]):
                break
        if self.is_output_rank:
            logger.info("LingBot role workers ready (waited %.1f s)", time.perf_counter() - t0)

    # -- sessions -----------------------------------------------------------------------

    def release_session(self, session_id: str) -> None:
        state = self.sessions.pop(session_id, None)
        if state is not None and self.is_output_rank and self.aux is not None:
            with fatal_on_output_rank("the role workers' session release"):
                self.aux.release(session_id)

    def _open_session(self, session_id: str, *, prompt: str, seed: int, image_path: str, inputs):
        for other in list(self.sessions):
            self.release_session(other)  # one session at a time (the pool and the roles hold one)
        state = SessionState(
            session_id=session_id,
            paging=self._new_paging(),
            generator=torch.Generator(device="cpu").manual_seed(int(seed)),
        )
        total = inputs.get(TOTAL_CHUNKS)
        state.total_chunks = int(total) if total is not None else None
        trajectory = inputs.get(CAMERA_TRAJECTORY)
        if trajectory is not None:
            state.trajectory = self._prepare_trajectory(trajectory, state.total_chunks)
            state.prefetch = self._prefetch(state)
        # Read on every rank (a bad image fails the request on all of them); only the role-worker
        # submit is rank 0's.
        image = source_image(image_path, height=self.height, width=self.width)
        if self.is_output_rank:
            # The condition encoder starts at session open and works through the sequence (up to
            # the session's chunks) while the DiT runs.
            with fatal_on_output_rank("the session's condition encode"):
                self.aux.submit_encode(session_id, image=image, chunks=state.total_chunks)
        prompt_t0 = time.perf_counter()
        state.text_kv = self._text_kv(prompt, session_id)
        state.constants = self.runtime.session_constants([t for t, _ in self.schedule])
        self.sessions[session_id] = state
        if self.is_output_rank:
            logger.info("LingBot session %s opened (prompt %.1f ms)", session_id, _ms(prompt_t0))
        return state

    def _prefetch(self, state: SessionState):
        """The recorded-camera session's per-chunk inputs, uploaded once (see ``prefetch.py``);
        draws the session's noise from its generator."""
        from sglang_neuron.diffusion.models.lingbot_world.prefetch import SessionPrefetch

        out_channels = int(self.transformer.config.out_channels)
        return SessionPrefetch(
            chunks=min(int(state.total_chunks), PREFETCH_MAX_CHUNKS),
            frames_per_chunk=self.frames_per_chunk,
            steps=len(self.schedule),
            latent_shape=(1, out_channels, self.frames_per_chunk, *self.latent_hw),
            poses=state.trajectory.poses,
            intrinsics=state.trajectory.intrinsics,
            generator=state.generator,
            paging=state.paging,
            device=self.device,
        )

    def _plan(self, state: SessionState, chunk: int, commit: bool):
        """The next pass's read plan: the prefetched one when the session has it (the session's
        own paging still advances: it allocates the chunk's blocks), else built and uploaded."""
        if state.prefetch is not None and state.prefetch.covers(chunk):
            host = state.paging.plan_host(commit)
            plan = state.prefetch.plan(chunk, commit)
            blocks = state.prefetch.current_blocks(chunk)
            if host != plan.host_bounds or state.paging.current != blocks:
                raise RuntimeError(
                    f"the prefetched plan of chunk {chunk} diverged from the session's paging: "
                    f"bounds {plan.host_bounds} vs {host}, blocks {blocks} vs {state.paging.current}"
                )
            return plan
        return state.paging.paged_for(commit)

    def _prepare_trajectory(self, value: dict, total_chunks: int | None) -> ChunkCamera:
        trajectory = C.trajectory_from_lists(
            value["poses"], value["intrinsics"], value.get("dtype")
        )
        if total_chunks is None:
            raise ValueError(f"a {CAMERA_TRAJECTORY} session needs its chunk count ({TOTAL_CHUNKS})")
        latent_frames = total_chunks * self.frames_per_chunk
        num_frames = (latent_frames - 1) * self.vae_temporal + 1
        poses, intrinsics = C.prepare_recorded_trajectory(
            trajectory,
            num_frames=num_frames,
            num_latent_frames=latent_frames,
            height=self.height,
            width=self.width,
        )
        return ChunkCamera(poses=poses, intrinsics=intrinsics)

    def _text_kv(self, prompt: str, request_id: str):
        text = " ".join(prompt.strip().split())
        # One prompt's text K/V on the device (the prompt embeddings stay cached on the host), so
        # nothing is allocated past the warm-up.
        return _lru(
            self._text_kv_cache,
            text,
            lambda: self.runtime.text_kv(self._embed(text, request_id)),
            size=1,
        )

    def _embed(self, text: str, request_id: str) -> torch.Tensor:
        """UMT5 on the role workers (rank 0), broadcast to every DiT rank; cached per prompt."""

        def encode():
            embeds = None
            if self.is_output_rank:
                with fatal_on_output_rank("the prompt encode"):
                    tokens = tokenize(self.tokenizer, text, MAX_SEQUENCE_LENGTH)
                    self.aux.submit_encode(
                        request_id, prompt_tokens=(tokens.input_ids, tokens.attention_mask)
                    )
                    embeds = self.aux.prompt_embeds(request_id)
            hidden = int(self.transformer.text_embedding[0].in_features)
            return broadcast_cpu_from_rank0(
                embeds, (1, MAX_SEQUENCE_LENGTH, hidden), self.transformer.dtype
            )

        return _lru(self._prompt_cache, text, encode)

    # -- one chunk ----------------------------------------------------------------------

    def _chunk_camera(self, state: SessionState, chunk: int, inputs) -> ChunkCamera:
        frames = self.frames_per_chunk
        if state.trajectory is not None:
            start, stop = chunk * frames, (chunk + 1) * frames
            return ChunkCamera(
                poses=state.trajectory.poses[start:stop],
                intrinsics=state.trajectory.intrinsics[start:stop],
            )
        actions = inputs.get(CAMERA_ACTIONS)
        if not actions:
            # SGLang's semantics (_prepare_lingbot_world_condition): a chunk without actions gets
            # no camera condition and leaves the history as it is. The poses only feed the prep
            # graph's rays, whose camera modulation is then replaced by zeros.
            poses = torch.eye(4, dtype=torch.float64).repeat(frames, 1, 1)
            intrinsics = torch.tensor(
                [[500.0, 500.0, self.width / 2, self.height / 2]], dtype=torch.float64
            ).repeat(frames, 1)
            return ChunkCamera(poses=poses, intrinsics=intrinsics, conditioned=False)
        state.action_history.extend(list(a) for a in actions)
        poses, intrinsics = C.prepare_action_history(
            state.action_history, frames=frames, height=self.height, width=self.width
        )
        return ChunkCamera(poses=poses, intrinsics=intrinsics)

    def _condition(self, state: SessionState, chunk: int) -> torch.Tensor:
        """``[1, 20, F, h, w]``: the temporal mask (first frame of chunk 0) and the normalized
        image-condition latent, on every DiT rank."""
        shape = (1, int(self.transformer.config.out_channels), self.frames_per_chunk, *self.latent_hw)
        latent = None
        if self.is_output_rank:
            with fatal_on_output_rank("the chunk condition"):
                latent = self.aux.condition(state.session_id, chunk).float()
                if tuple(latent.shape) != shape:
                    raise RuntimeError(
                        f"condition encoder returned {tuple(latent.shape)}, want {shape}"
                    )
        latent = broadcast_cpu_from_rank0(latent, shape, torch.float32)
        latent = (latent - self.latents_mean) / self.latents_std
        mask = latent.new_zeros(1, self.vae_temporal, *latent.shape[2:])
        if chunk == 0:
            mask[:, :, 0] = 1
        return torch.cat((mask, latent), dim=1)

    def _launch_prep(self, state: SessionState, chunk: int, camera: ChunkCamera):
        condition = self._condition(state, chunk)
        pre = state.prefetch
        if pre is not None and pre.covers(chunk):
            if self.is_output_rank:
                state.checksums[chunk] = {
                    "condition_sum": float(condition.double().sum()),
                    "noise_sum": pre.noise_sums[chunk],
                }
            x_t, noises = pre.noise(chunk)
            return self.runtime.launch_prep_uploaded(
                start_frame=chunk * self.frames_per_chunk,
                params=pre.params(chunk),
                rope_t=pre.rope(chunk),
                x_t=x_t,
                noises=noises,
                condition=pre.condition(condition),
            )
        latent_shape = (1, int(self.transformer.config.out_channels), *condition.shape[2:])
        # vLLM-Omni's draw order: the chunk's initial latent, then one re-noise per step.
        noise = torch.stack(
            [
                torch.randn(latent_shape, generator=state.generator, dtype=torch.float32)
                for _ in range(len(self.schedule))
            ]
        )
        if self.is_output_rank:
            # Host checksums of the chunk's inputs (the timing row's determinism trace).
            state.checksums[chunk] = {
                "condition_sum": float(condition.double().sum()),
                "noise_sum": float(noise.double().sum()),
            }
        prep = self.runtime.launch_prep(
            start_frame=chunk * self.frames_per_chunk,
            poses=camera.poses,
            intrinsics=camera.intrinsics,
            noise=noise,
            condition=condition,
        )
        return prep if camera.conditioned else self.runtime.without_camera(prep)

    def _launch_steps(self, state: SessionState, prep, *, chunk: int, first: int, x_t, flow, stop: int):
        """Launch denoise steps ``[first, stop)``; returns ``(x_t, flow)`` of the last.

        One read plan serves every denoise step of a chunk: it is built (and uploaded) once,
        before the first launch. A host-to-device copy waits for the device's queued work, so
        an upload between two launches would serialize them."""
        paged = self._plan(state, chunk, False) if stop > first else None
        for k in range(first, stop):
            flow, x_t = self.runtime.launch_denoise(
                prep, paged, state.text_kv, state.constants, k, x_t, flow
            )
        return x_t, flow

    def _start_next(self, state: SessionState, chunk: int, prep) -> int:
        """Launch the first denoise steps of ``chunk`` (its prep already launched) that fit in
        the in-flight budget, the same on every rank; after the previous chunk's commit (the
        passes read the committed history). Returns the steps launched."""
        steps = min(len(self.schedule), max(self.runtime.free_launch_slots(), 0))
        whole = steps == len(self.schedule) and self.runtime.free_launch_slots() > steps
        # Every upload of the chunk (its read plans) before its first launch: the host then
        # queues the whole chunk -- its denoise steps and, room permitting, its commit -- and
        # returns while the device runs it.
        commit = self._plan(state, chunk, True) if whole else None
        x_t, flow = self._launch_steps(
            state, prep, chunk=chunk, first=0, x_t=None, flow=None, stop=steps
        )
        started = _Started(prep=prep, x_t=x_t, flow=flow, steps=steps)
        if whole:
            started.done, started.x0 = self.runtime.launch_commit(
                prep, commit, state.text_kv, state.constants, x_t, flow
            )
            started.commit = commit
        state.started[chunk] = started
        return steps

    def generate_chunk(self, req) -> tuple[list[np.ndarray], dict]:
        """Run one chunk request on this rank; returns ``(frames, timing)``: rank 0's decoded
        ``(F, H, W, 3)`` uint8 frame arrays (in order; possibly none), else ``[]``."""
        t0 = time.perf_counter()
        session_id = str(req.realtime_session_id or req.request_id)
        chunk = int(req.block_idx)
        inputs = dict(req.condition_inputs or {})
        state = self.sessions.get(session_id)
        if chunk == 0 or state is None:
            if chunk != 0:
                raise ValueError(f"LingBot session {session_id} has no state for chunk {chunk}")
            state = self._open_session(
                session_id,
                prompt=_prompt_text(req),
                seed=int(req.seed if req.seed is not None else 42),
                image_path=_image_path(req),
                inputs=inputs,
            )
        if chunk != state.next_chunk:
            raise ValueError(f"LingBot chunks must be contiguous: got {chunk}, want {state.next_chunk}")
        if state.total_chunks is not None and chunk >= state.total_chunks:
            raise ValueError(
                f"LingBot session {session_id} has {state.total_chunks} chunks; got chunk {chunk}"
            )
        last = state.total_chunks is not None and chunk + 1 >= state.total_chunks
        timing = {"chunk": chunk}
        started = state.started.pop(chunk, None)
        if chunk > 0 and inputs.get(PROMPT_UPDATED):
            # A new prompt from this chunk on (SGLang resets its cross-attention cache): new text
            # K/V on every rank. A chunk started early ran its steps with the old prompt: keep
            # its prep (camera, condition, noise) and run its steps and commit again; the commit
            # rewrites the same pool slots.
            state.text_kv = self._text_kv(_prompt_text(req), session_id)
            if started is not None:
                started = _Started(prep=started.prep, x_t=None, flow=None, steps=0)
            timing["prompt_updated"] = True
        if started is None:
            prep = self._launch_prep(state, chunk, self._chunk_camera(state, chunk, inputs))
            started = _Started(prep=prep, x_t=None, flow=None, steps=0)
        timing["early_steps"] = started.steps
        if started.done is not None:  # the whole chunk is already queued
            commit, done, x0 = started.commit, started.done, started.x0
        else:
            # The commit plan is uploaded before the remaining steps are queued (see
            # _launch_steps on uploads between launches).
            commit = self._plan(state, chunk, True)
            x_t, flow = self._launch_steps(
                state, started.prep, chunk=chunk, first=started.steps, x_t=started.x_t,
                flow=started.flow, stop=len(self.schedule),
            )
            done, x0 = self.runtime.launch_commit(
                started.prep, commit, state.text_kv, state.constants, x_t, flow
            )
        timing["launched_ms"] = _ms(t0)
        sync_t0 = time.perf_counter()
        # The chunk's one host sync. A device-to-host read waits for all queued work, so it runs
        # before the next chunk is queued, and rank 0 reads the clean latent right after it.
        self.runtime.synchronize(done)
        timing["dit_wait_ms"] = _ms(sync_t0)
        x0_host = None
        if self.is_output_rank:
            with fatal_on_output_rank("the latent readback"):
                t_x0 = time.perf_counter()
                x0_host = x0.cpu()
                timing["x0_readback_ms"] = _ms(t_x0)
        state.paging.commit()
        state.next_chunk += 1
        if state.trajectory is not None and not last:
            # The camera is known ahead: the next chunk's prep and its denoise steps and commit
            # (in-flight budget permitting) are queued back to back, before rank 0 hands this
            # chunk to the decoder, so the device runs the next chunk meanwhile.
            t_np = time.perf_counter()
            camera = self._chunk_camera(state, chunk + 1, {})
            next_prep = self._launch_prep(state, chunk + 1, camera)
            timing["next_prep_ms"] = _ms(t_np)
            t_sn = time.perf_counter()
            timing["next_early_steps"] = self._start_next(state, chunk + 1, next_prep)
            timing["start_next_ms"] = _ms(t_sn)
        frames = self._decode(state, chunk, x0_host, last=last, timing=timing)
        timing["chunk_ms"] = _ms(t0)
        timing["steady"] = bool(commit.history_full)
        timing.update(self.runtime.take_launch_stats())
        self._record_timing(session_id, timing)
        return frames, timing

    def _decode(self, state: SessionState, chunk: int, x0_host, *, last: bool, timing):
        """Rank 0: submit the chunk's latents (``x0_host``, read back) to the decoder and collect
        this chunk's frames (one ``(F, H, W, 3)`` array); ``[]`` on the other ranks."""
        if not self.is_output_rank:
            return []
        with fatal_on_output_rank("the decode submit"):
            vae_latents = (x0_host * self.latents_std + self.latents_mean).to(torch.bfloat16)
            timing.update(state.checksums.pop(chunk, {}))
            timing["latents_sum"] = float(vae_latents.double().sum())
            self.aux.submit_latents(state.session_id, chunk, vae_latents, last=last)
        # Every aux call here is rank 0's alone: a failure must end the worker, not leave the
        # other ranks waiting in the next request's broadcast.
        with fatal_on_output_rank("the decode"):
            t0 = time.perf_counter()
            frames = [self.aux.collect_frames(state.session_id, chunk)]
            timing["decode_wait_ms"] = _ms(t0)
            timing.update(self.aux.role_timings(state.session_id, chunk))
        return frames

    def _record_timing(self, session_id: str, row: dict) -> None:
        if not self.is_output_rank:
            return
        row = {"session": session_id, "emitted_at": round(time.time(), 4), **row}
        logger.debug("LingBot chunk %s", json.dumps(row))
        if self.timing_file:
            with open(self.timing_file, "a") as f:
                f.write(json.dumps(row) + "\n")

    def shutdown(self) -> None:
        if self.aux is not None:
            self.aux.shutdown()


def _prompt_text(req) -> str:
    prompt = req.prompt
    if isinstance(prompt, list):
        prompt = prompt[0]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("LingBot-World needs a non-empty text prompt")
    return prompt


def _image_path(req) -> str:
    path = req.image_path
    if isinstance(path, list):
        path = path[0]
    if not path:
        raise ValueError("LingBot-World needs the session's first frame (first_frame / image_path)")
    return str(path)
