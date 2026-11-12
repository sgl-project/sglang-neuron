# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 t2va and fl2va on Neuron, independent of the serving framework.

``NeuronMiniMaxH3Model`` follows diffusers' ``MiniMaxH3ModularPipeline``. The checkpoint is
guidance-distilled, so every step is one DiT forward.
"""

import itertools
import logging
import math
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.distributed as dist
from diffusers.models.transformers.transformer_minimax_h3 import (
    MiniMaxH3Transformer3DModel as DiffusersMiniMaxH3Transformer3DModel,
)
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3PrepareLayoutStep,
    patchify_video_latents,
)
from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3ResizeStep
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_AUDIO_TAG,
    MINIMAX_H3_FPS,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
    MiniMaxH3ModularPipeline,
    align_num_frames,
    audio_latent_num_frames,
    resolve_canvas_size,
    video_latent_num_frames,
)
from diffusers.schedulers import MiniMaxH3Scheduler
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from torch import nn
from transformers import AutoTokenizer, Qwen3VLProcessor

from sglang_neuron.diffusion.distributed.autoencoders.autoencoder_kl_minimax_h3 import (
    DistributedAutoencoderKLMiniMaxH3,
    NeuronAutoencoderKLMiniMaxH3,
)
from sglang_neuron.diffusion.distributed.autoencoders.autoencoder_kl_minimax_h3_audio import (
    AUDIO_WINDOW,
    NeuronAutoencoderKLMiniMaxH3Audio,
    audio_windows,
)
from sglang_neuron.diffusion.models.minimax_h3.encoders import (
    MINIMAX_H3_TEXT_SEQ_BUCKET,
    NeuronQwen3VLTextEncoder,
)
from sglang_neuron.diffusion.distributed.autoencoders.vae_executor import new_vae_parallel_groups
from sglang_neuron.diffusion.distributed.groups import get_tp_group, get_world_group
from sglang_neuron.diffusion.models.minimax_h3.minimax_h3_transformer import (
    MiniMaxH3Transformer3DModel,
)
from sglang_neuron.diffusion.runtime import neuron_device
from sglang_neuron.diffusion.stage_timing import StageMarks

logger = logging.getLogger(__name__)

# The reference's canvas (16:9), frame count and step count when a request leaves them unset.
DEFAULT_HEIGHT = 768
DEFAULT_WIDTH = 1344
DEFAULT_NUM_FRAMES = 124
DEFAULT_NUM_STEPS = 50
# MiniMaxH3Scheduler.set_timesteps needs two sigma grid points.
MIN_NUM_STEPS = 2
# _to_uint8_frames: frames per thread-pool task, and the pool's size.
_FRAMES_PER_TASK = 4
_HOST_THREADS = 32
# The durations the checkpoint was released for.
MIN_DURATION_S = 5.0
MAX_DURATION_S = 15.0
MAX_AUDIO_WINDOWS = len(
    audio_windows(audio_latent_num_frames(int(MAX_DURATION_S * MINIMAX_H3_FPS)))
)
# The video VAE's chunking (``clip_length``, ``tokens_chunk_size``) and spatial compression.
VAE_FRAMES_PER_CHUNK = 17
VAE_LATENTS_PER_CHUNK = 5
VAE_SPATIAL_COMPRESSION = 16
# Every audio VAE window length a request reaches: each short clip's own, then AUDIO_WINDOW.
AUDIO_WINDOW_LENGTHS = (
    *itertools.takewhile(
        lambda n: n < AUDIO_WINDOW,
        map(audio_latent_num_frames, itertools.count(VAE_LATENTS_PER_CHUNK, VAE_FRAMES_PER_CHUNK)),
    ),
    AUDIO_WINDOW,
)
# The video VAE's pixel convention: ImageNet-normalized RGB over [0, 1].
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)

TEXT_ENCODER_COMPILER_ARGS = [
    "--model-type=transformer",
    "--auto-cast=none",
    "-O1",
    "--hbm-scratchpad-page-size=2048",
]
TRANSFORMER_COMPILER_ARGS = [
    "--model-type=transformer",
    "--auto-cast=none",
    "-O2",
    "--hbm-scratchpad-page-size=2048",
]
VAE_COMPILER_ARGS = [
    "--model-type=unet-inference",
    "--auto-cast=none",
    "--internal-max-instruction-limit=15000000",
    "-O1",
    "--hbm-scratchpad-page-size=2048",
]
AUDIO_VAE_COMPILER_ARGS = [
    "--model-type=unet-inference",
    "--auto-cast=none",
    "-O1",
    "--hbm-scratchpad-page-size=2048",
]


def encode_keyframe_condition(vae, keyframe, device, encode_seed: int) -> torch.Tensor:
    """The reference's ``encode_vae_condition`` for one keyframe: normalized ``(1, C, 1, H, W)``
    latents on the host. Pixels are normalized on the host: a host-to-device copy cannot cast."""
    pixel_mean = torch.tensor(PIXEL_MEAN).view(1, -1, 1, 1, 1)
    pixel_std = torch.tensor(PIXEL_STD).view(1, -1, 1, 1, 1)
    pixels = torch.from_numpy(np.array(keyframe)).permute(2, 0, 1)[None, :, None]
    pixels = (pixels.float().div(255.0) - pixel_mean) / pixel_std
    posterior = vae.encode(pixels.to(device), return_dict=False)[0]
    latents = posterior.sample(generator=torch.Generator().manual_seed(encode_seed))
    # On the host first: an untiled single-rank encode leaves the moments on the device.
    latents = latents.cpu().to(torch.float16).float()
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
    return (latents - latents_mean) / latents_std


def _quantize_frames(clip: torch.Tensor) -> torch.Tensor:
    """An ImageNet-normalized ``(B, 3, t, H, W)`` clip as rounded ``(B, t, H, W, 3)`` uint8
    frames, on the clip's device."""
    mean = torch.tensor(PIXEL_MEAN, device=clip.device).view(1, -1, 1, 1, 1)
    std = torch.tensor(PIXEL_STD, device=clip.device).view(1, -1, 1, 1, 1)
    clip = clip.float() * std + mean
    return clip.clamp(0, 1).mul(255).round().permute(0, 2, 3, 4, 1).to(torch.uint8)


def _to_uint8_frames(video: torch.Tensor) -> torch.Tensor:
    """``_quantize_frames`` for a whole ``(B, 3, T, H, W)`` video on the host.

    Done on the output rank, so the worker returns 1 byte per value instead of 2. A few frames
    per task run on a thread pool, in the caller's inference mode: the workers run one intra-op
    thread. An empty tensor (a VAE rank that does not hold the video) passes through.
    """
    if video.numel() == 0:
        return video
    batch, channels, num_frames, height, width = video.shape
    frames = torch.empty((batch, num_frames, height, width, channels), dtype=torch.uint8)
    inference, grad = torch.is_inference_mode_enabled(), torch.is_grad_enabled()

    def convert(start: int) -> None:
        with torch.inference_mode(inference), torch.set_grad_enabled(grad):
            frames[:, start : start + _FRAMES_PER_TASK] = _quantize_frames(
                video[:, :, start : start + _FRAMES_PER_TASK]
            )

    starts = range(0, num_frames, _FRAMES_PER_TASK)
    workers = min(_HOST_THREADS, len(os.sched_getaffinity(0)), len(starts))
    with ThreadPoolExecutor(max(workers, 1)) as pool:
        for future in [pool.submit(convert, start) for start in starts]:
            future.result()
    return frames


def step_coefficients(scheduler) -> torch.Tensor:
    """Each step's ``(1 - t, r, 1 - r)``, ``r = sigma_next / sigma``, in float32 as
    ``MiniMaxH3Scheduler.step`` computes them: ``[num_steps, 3]``."""
    ratio = scheduler.sigmas[1:] / scheduler.sigmas[:-1]
    return torch.stack([1 - scheduler.timesteps, ratio, 1.0 - ratio], dim=-1)


def denoise_update(
    noise_pred, audio_noise_pred, video_rows, audio_rows, coefficients, condition_rows, dtype
):
    """``MiniMaxH3Scheduler.step`` (``eta = 0``, float32) for both schedules, and the DiT's next
    ``hidden_states`` and ``audio_hidden_states`` in ``dtype``.

    ``coefficients`` is ``[2, 3]``: the video and audio ``step_coefficients`` rows. The
    conditioning rows (None without keyframes) lead ``noise_pred`` and are never stepped.
    """
    num_condition = 0 if condition_rows is None else condition_rows.shape[0]
    stepped = []
    for index, (sample, velocity) in enumerate(
        ((video_rows, noise_pred[0, num_condition:]), (audio_rows, audio_noise_pred[0]))
    ):
        sigma_t, ratio, one_minus_ratio = coefficients[index].unbind()
        denoised = sample + sigma_t * velocity.float()
        stepped.append(ratio * sample + one_minus_ratio * denoised)
    video_rows, audio_rows = stepped
    hidden_states = video_rows.to(dtype)
    if condition_rows is not None:
        hidden_states = torch.cat([condition_rows, hidden_states])
    return video_rows, audio_rows, hidden_states[None], audio_rows[None].to(dtype)


class NeuronMiniMaxH3Model(nn.Module):
    """MiniMax-H3 t2va and fl2va on Neuron: TP conditioner, TP/CP DiT, video VAE on rank 0 or
    patch-parallel over ``vae_patch_parallel_size`` ranks, audio VAE windows on ranks
    1..``MAX_AUDIO_WINDOWS``. Every rank calls ``generate`` in lockstep.
    """

    def __init__(
        self,
        model_path: str,
        *,
        dtype: torch.dtype = torch.bfloat16,
        vae_patch_parallel_size: int = 64,
        model_config: dict | None = None,
    ):
        super().__init__()
        self.model_path = model_path
        self.device = neuron_device()
        self.dtype = dtype
        model = model_path
        local_files_only = os.path.isdir(model)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model, subfolder="tokenizer", local_files_only=local_files_only
        )
        # Only keyframe requests read it, so a t2va checkpoint may leave ``processor`` out. Loaded
        # here, not per request: a failed read on some ranks would leave the others in the
        # conditioner's collectives.
        self._processor = None
        if not local_files_only or os.path.isdir(os.path.join(model, "processor")):
            self._processor = Qwen3VLProcessor.from_pretrained(
                model, subfolder="processor", local_files_only=local_files_only
            )
        self._resize_keyframes = MiniMaxH3ModularPipeline(blocks=MiniMaxH3ResizeStep())
        # The TP group, not WORLD: the vision tower's 16 heads cannot split over tp * cp ranks.
        self.text_encoder = NeuronQwen3VLTextEncoder(
            model_path=model, dtype=self.dtype, tp_group=get_tp_group().device_group
        )

        cfg = DiffusersMiniMaxH3Transformer3DModel.load_config(
            model, subfolder="transformer", local_files_only=local_files_only
        )
        cfg.update(model_config or {})
        # Not cast: a cast would round the FP32 rope.inv_freq buffer to BF16.
        self.transformer = MiniMaxH3Transformer3DModel(**cfg)
        # ((conditioned, video timesteps, audio timesteps), each step's block_modulations).
        self._adaln_cache: tuple[tuple, list[tuple[torch.Tensor, ...]]] | None = None
        # Eager until compile_denoise_update.
        self._denoise_update = denoise_update

        self.scheduler = MiniMaxH3Scheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only
        )
        self.audio_scheduler = MiniMaxH3Scheduler.from_pretrained(
            model, subfolder="audio_scheduler", local_files_only=local_files_only
        )

        self.is_output_rank = dist.get_rank() == 0
        vae_ranks = vae_patch_parallel_size
        # new_group is collective: the ranks outside the group call it too.
        vae_group, vae_error_group = (
            new_vae_parallel_groups(list(range(vae_ranks))) if vae_ranks > 1 else (None, None)
        )
        self.vae = None
        if dist.get_rank() < vae_ranks:
            vae_cls = (
                NeuronAutoencoderKLMiniMaxH3
                if vae_group is None
                else DistributedAutoencoderKLMiniMaxH3
            )
            vae_config = vae_cls.load_config(
                model, subfolder="vae", local_files_only=local_files_only
            )
            # Cast before load_weights, which promotes the encoder half back to FP32 first.
            self.vae = vae_cls.from_config(vae_config).to(self.dtype)
            if vae_group is not None:
                self.vae.init_distributed(group=vae_group, error_group=vae_error_group)
                self.vae.set_parallel_size(vae_ranks)
                # The device merge returns the frames already quantized.
                self.vae.frame_fn = _quantize_frames
        self._vae_ranks = vae_ranks
        # Off rank 0 when there are other ranks: rank 0's HBM goes to the video VAE merge.
        num_audio_ranks = min(dist.get_world_size() - 1, MAX_AUDIO_WINDOWS)
        self._audio_ranks = list(range(1, 1 + num_audio_ranks)) or [0]
        if isinstance(self.vae, DistributedAutoencoderKLMiniMaxH3):
            # The video tiles prefer the other ranks, so the windows decode beside them.
            self.vae.distributed_executor.tile_ranks = [
                rank for rank in range(vae_ranks) if rank not in self._audio_ranks
            ]
        audio_vae_config = NeuronAutoencoderKLMiniMaxH3Audio.load_config(
            model, subfolder="audio_vae", local_files_only=local_files_only
        )
        self._audio_hop = math.prod(audio_vae_config["decoder_rates"])
        self.audio_sample_rate = audio_vae_config["sampling_rate"]
        self.fps = MINIMAX_H3_FPS
        self.audio_vae = None
        if dist.get_rank() in self._audio_ranks:
            self.audio_vae = NeuronAutoencoderKLMiniMaxH3Audio.from_config(audio_vae_config)

    def to(self, *args, **kwargs):
        """Move the text encoder, transformer and, where this rank holds them, the VAEs."""
        for module in (self.text_encoder, self.transformer, self.vae, self.audio_vae):
            if module is not None:
                module.to(*args, **kwargs)
        return self

    def load_weights(self):
        """Load the conditioner, DiT and VAE weights."""
        model_path = self.model_path
        self.text_encoder.load_weights(model_path)
        self.transformer.load_weights(os.path.join(model_path, "transformer"))
        if self.vae is not None:
            self.vae.load_weights(os.path.join(model_path, "vae"))
        if self.audio_vae is not None:
            self.audio_vae.load_weights(os.path.join(model_path, "audio_vae"))

    @property
    def processor(self) -> Qwen3VLProcessor:
        """The conditioner's processor, for the keyframes' vision inputs."""
        if self._processor is None:
            raise ValueError(
                f"MiniMax-H3 keyframes need the checkpoint's processor/, which {self.model_path} "
                "does not have"
            )
        return self._processor

    def compile_text_encoder(self, *args, **kwargs):
        options = {**kwargs.pop("options", {}), "model_name": "minimax_h3_text_encoder"}
        options["compiler_args"] = TEXT_ENCODER_COMPILER_ARGS
        kwargs.setdefault("fullgraph", True)
        self.text_encoder.compile(*args, options=options, **kwargs)

    def compile_vae(self, *args, **kwargs):
        if self.vae is None:
            return
        options = {**kwargs.pop("options", {}), "model_name": "minimax_h3_vae"}
        options["compiler_args"] = VAE_COMPILER_ARGS
        # Decoder tiles use one fixed spatial shape.
        kwargs.setdefault("fullgraph", True)
        # Lazy: a t2va request never builds the keyframe encoder.
        self.vae.compile(*args, options=options, compile_encoder=True, **kwargs)

    def compile_audio_vae(self, *args, **kwargs):
        if self.audio_vae is None:
            return
        options = {**kwargs.pop("options", {}), "model_name": "minimax_h3_audio_vae"}
        options["compiler_args"] = AUDIO_VAE_COMPILER_ARGS
        kwargs.setdefault("fullgraph", True)
        # Its graph instances share a Dynamo cache, where a second shape would turn symbolic.
        kwargs.setdefault("dynamic", False)
        self.audio_vae.compile(
            *args, window_lengths=AUDIO_WINDOW_LENGTHS, options=options, **kwargs
        )

    def compile_transformer(self, *args, **kwargs):
        """Compile the DiT at ``-O2``.

        ``-O1`` deadlocks the DiT NEFF between collectives inside a block: POOL waits on a DMA
        semaphore that never fires while PE runs the attention / FFN matmuls. ``-O2`` schedules
        the DMA producer differently, with the same graph.
        """
        options = {**kwargs.pop("options", {}), "model_name": "minimax_h3_transformer"}
        options["compiler_args"] = TRANSFORMER_COMPILER_ARGS
        kwargs.setdefault("fullgraph", True)
        # The packed sequence has the prompt's real token count: each count gets its own static
        # graph rather than a symbolic-shape one.
        kwargs.setdefault("dynamic", False)
        self.transformer.compile(*args, options=options, **kwargs)

    def compile_denoise_update(self, *args, **kwargs):
        """Compile ``denoise_update``: eager device ops each launch a graph of their own."""
        options = {**kwargs.pop("options", {}), "model_name": "minimax_h3_denoise_update"}
        kwargs.setdefault("fullgraph", True)
        kwargs.setdefault("dynamic", False)
        self._denoise_update = torch.compile(denoise_update, *args, options=options, **kwargs)

    def compile(self, *args, **kwargs):
        """Compile the text encoder, VAEs, DiT and the denoise update for Neuron."""
        self.compile_text_encoder(*args, **kwargs)
        self.compile_vae(*args, **kwargs)
        self.compile_audio_vae(*args, **kwargs)
        self.compile_transformer(*args, **kwargs)
        self.compile_denoise_update(*args, **kwargs)
        return self

    def resolve_geometry(
        self,
        height: int | None,
        width: int | None,
        num_frames: int | None,
        first_keyframe: Image.Image | None = None,
    ) -> tuple[int, int, int]:
        """``(height, width, num_frames)`` as the reference resolves them; ``None`` is unset. An
        unset canvas follows the first keyframe's aspect ratio, if there is one.

        A duration outside the released 5-15 s warns rather than raising, so short test clips run.
        """
        if (height is None) != (width is None):
            raise ValueError("`height` and `width` have to be passed together, or neither of them.")
        multiple = VAE_SPATIAL_COMPRESSION * self.transformer.config.patch_size[2]
        if height is None and first_keyframe is not None:
            height, width = resolve_canvas_size(
                *first_keyframe.size,
                multiple,
                self._resize_keyframes.config.canvas_short_edge,
                self._resize_keyframes.config.canvas_max_pixels,
            )
        else:
            height = height or DEFAULT_HEIGHT
            width = width or DEFAULT_WIDTH
        if height % multiple or width % multiple:
            raise ValueError(
                f"`height` and `width` must be multiples of {multiple}, got {height}x{width}."
            )
        num_frames = num_frames or DEFAULT_NUM_FRAMES
        num_frames = align_num_frames(num_frames, VAE_FRAMES_PER_CHUNK, VAE_LATENTS_PER_CHUNK)
        duration = num_frames / MINIMAX_H3_FPS
        if not MIN_DURATION_S <= duration <= MAX_DURATION_S:
            logger.warning(
                "MiniMax-H3 was released for %.0f-%.0f s; generating %d frames (%.2f s) anyway.",
                MIN_DURATION_S,
                MAX_DURATION_S,
                num_frames,
                duration,
            )
        return height, width, num_frames

    def encode_prompt(
        self, prompt: str, keyframes: list[Image.Image] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The conditioner's ``[1, T, text_dim]`` embedding of the presentation, on the host,
        and each row's modality tag.

        The prompt is presented verbatim, with no special tokens or chat template, after a
        ``"<Picture i>: "`` label and a vision block (tagged video) per keyframe.
        """
        token_ids, token_tags, vision = [], [], {}
        if keyframes:
            vision = self.processor.image_processor(images=keyframes, return_tensors="pt")
            merge = self.processor.image_processor.merge_size**2
            for index, grid in enumerate(vision["image_grid_thw"]):
                num_image_tokens = int(grid.prod()) // merge
                label = f"<Picture {index + 1}>: "
                label_ids = self.tokenizer(label, add_special_tokens=False)["input_ids"]
                vision_ids = (
                    [self.tokenizer.convert_tokens_to_ids("<|vision_start|>")]
                    + [self.tokenizer.convert_tokens_to_ids("<|image_pad|>")] * num_image_tokens
                    + [self.tokenizer.convert_tokens_to_ids("<|vision_end|>")]
                )
                token_ids += label_ids + vision_ids
                token_tags += [MINIMAX_H3_TEXT_TAG] * len(label_ids)
                token_tags += [MINIMAX_H3_VIDEO_TAG] * len(vision_ids)
        token_ids += self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        token_tags += [MINIMAX_H3_TEXT_TAG] * (len(token_ids) - len(token_tags))
        text_inputs = self.tokenizer.pad(
            {"input_ids": [token_ids]},
            pad_to_multiple_of=MINIMAX_H3_TEXT_SEQ_BUCKET,
            padding_side="right",
            return_attention_mask=True,
            return_tensors="pt",
        )
        prompt_embeds = self.text_encoder(
            text_inputs["input_ids"],
            text_inputs["attention_mask"],
            vision.get("pixel_values"),
            vision.get("image_grid_thw"),
        ).last_hidden_state.cpu()
        return prompt_embeds, torch.tensor(token_tags, dtype=torch.long)

    def encode_keyframe(self, keyframe: Image.Image, latent_height, latent_width) -> torch.Tensor:
        """``encode_keyframe_condition`` for the fitted keyframe, on every rank; ranks outside
        the VAE group get it from rank 0."""
        condition = torch.empty(
            1,
            self.transformer.config.in_channels,
            1,
            latent_height,
            latent_width,
            dtype=torch.float32,
        )
        if self.vae is not None:
            condition = encode_keyframe_condition(
                self.vae, keyframe, self.device, self._resize_keyframes.keyframe_encode_seed
            )
        if self._vae_ranks < dist.get_world_size():
            # The condition is on the host, so the CPU group carries it.
            dist.broadcast(condition, src=0, group=get_world_group().cpu_group)
        return condition

    def block_modulations(self, timestep: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """The DiT's ``block_modulations`` for ``timestep``."""
        return self.transformer.block_modulations(timestep)

    def _step_timestep(self, t, audio_t, conditioned: bool) -> torch.Tensor:
        """A step's DiT timesteps on the device: video, audio and, with keyframes, the
        conditioning rows' ``max(t, keyframe_noise_aug)``."""
        timestep = [t, audio_t]
        if conditioned:
            timestep.append(t.clamp(min=self._resize_keyframes.keyframe_noise_aug))
        return torch.stack(timestep).to(self.device)

    def _step_timesteps(self, conditioned: bool) -> list[torch.Tensor]:
        """``_step_timestep`` for every step of the schedulers' current timesteps."""
        return [
            self._step_timestep(t, audio_t, conditioned)
            for t, audio_t in zip(self.scheduler.timesteps, self.audio_scheduler.timesteps)
        ]

    def _cached_block_modulations(
        self, conditioned: bool, timesteps: list[torch.Tensor] | None = None
    ) -> list[tuple[torch.Tensor, ...]]:
        """Every step's ``block_modulations`` for the schedulers' current timesteps, on the
        device, cached per schedule; ``timesteps`` are ``_step_timesteps(conditioned)``, if the
        caller has them.

        The AdaLN projections depend on the timestep alone, so requests that share a schedule
        reuse them and the DiT runs no AdaLN projection or gather per step. Only the last
        schedule is kept: 0.87 GB per rank at 50 blocks and 49 steps, half again with keyframes'
        third timestep. Anything that changes the weights in place must reset ``_adaln_cache``.
        """
        key = (
            conditioned,
            tuple(self.scheduler.timesteps.tolist()),
            tuple(self.audio_scheduler.timesteps.tolist()),
        )
        if self._adaln_cache is not None and self._adaln_cache[0] == key:
            return self._adaln_cache[1]
        # Free the old schedule's tables before computing the new one's.
        self._adaln_cache = None
        if timesteps is None:
            timesteps = self._step_timesteps(conditioned)
        tables = [self.block_modulations(timestep) for timestep in timesteps]
        self._adaln_cache = (key, tables)
        return tables

    def denoise(self, prompt_embeds, layout, video_rows, audio_rows, num_steps, marks=None):
        """The reference's dual-schedule loop, one compiled DiT forward per step.

        The reference embeds the sequence's distinct timesteps, a set that shrinks at the first
        step (both schedules start at ``t = 0``) and would compile a second DiT graph. One entry
        per schedule keeps the shape fixed and every row's embedding unchanged: video (and text),
        audio and, with keyframes, the conditioning rows (``_step_timestep``).

        The latents stay on the device and each step's update is its own graph
        (``denoise_update``), so the host queues the steps without waiting on them.

        Returns the denoised generated video and audio rows on the host, and the step marks
        (``marks``, host clock by default): the start, then the end of every step.
        """
        position_ids, token_tags, video_indices, audio_indices, text_indices, num_condition, _ = (
            layout
        )
        self.scheduler.set_timesteps(num_steps)
        self.audio_scheduler.set_timesteps(num_steps)
        # set_timesteps collapses float32 sigma collisions, which can differ between the shifts.
        if len(self.scheduler.timesteps) != len(self.audio_scheduler.timesteps):
            raise ValueError(
                f"{num_steps} steps give {len(self.scheduler.timesteps)} video and "
                f"{len(self.audio_scheduler.timesteps)} audio timesteps"
            )
        # Constant across steps, so copied once, each cast on the host first: a host-to-device
        # copy cannot cast.
        timestep_indices = (token_tags == MINIMAX_H3_AUDIO_TAG).long()
        # The conditioning rows read the third timestep.
        timestep_indices[video_indices[:num_condition]] = 2
        dit_inputs = {
            "encoder_hidden_states": prompt_embeds.to(self.dtype).to(self.device),
            "timestep_indices": timestep_indices.to(self.device),
            "token_tags": token_tags.to(self.device),
            "position_ids": position_ids.float().to(self.device),
            "video_indices": video_indices.to(self.device),
            "audio_indices": audio_indices.to(self.device),
            "text_indices": text_indices.to(self.device),
        }
        timesteps = self._step_timesteps(num_condition > 0)
        cached = None
        if self.transformer.config.adaln_cache:
            cached = self._cached_block_modulations(num_condition > 0, timesteps)
        condition_rows = None
        if num_condition:
            condition_rows = video_rows[:num_condition].to(self.dtype).to(self.device)
        hidden_states = video_rows.to(self.dtype)[None].to(self.device)
        audio_hidden_states = audio_rows[None].to(self.dtype).to(self.device)
        video_rows = video_rows[num_condition:].to(self.device)
        audio_rows = audio_rows.to(self.device)
        coefficients = torch.stack(
            [step_coefficients(self.scheduler), step_coefficients(self.audio_scheduler)], dim=1
        )
        coefficients = [step.to(self.device) for step in coefficients]
        marks = marks or StageMarks(device_events=False)
        step_marks = [marks.mark()]
        for index, timestep in enumerate(timesteps):
            block_modulations = (
                cached[index] if cached is not None else self.block_modulations(timestep)
            )
            noise_pred, audio_noise_pred = self.transformer(
                hidden_states=hidden_states,
                audio_hidden_states=audio_hidden_states,
                timestep=timestep,
                block_modulations=block_modulations,
                return_dict=False,
                **dit_inputs,
            )
            video_rows, audio_rows, hidden_states, audio_hidden_states = self._denoise_update(
                noise_pred,
                audio_noise_pred,
                video_rows,
                audio_rows,
                coefficients[index],
                condition_rows,
                self.dtype,
            )
            step_marks.append(marks.mark())
        video_rows, audio_rows = video_rows.cpu(), audio_rows.cpu()
        return video_rows, audio_rows, step_marks

    def unpack_video(self, video_rows, num_latent_frames, latent_height, latent_width):
        """The reference's after-denoise unpack: ``(1, C, T, H, W)`` video latents."""
        patch_t, patch_h, patch_w = self.transformer.config.patch_size
        channels = self.transformer.config.in_channels
        rows = video_rows.reshape(
            -1,
            num_latent_frames // patch_t,
            latent_height // patch_h,
            latent_width // patch_w,
            channels,
            patch_t,
            patch_h,
            patch_w,
        ).permute(0, 4, 1, 5, 2, 6, 3, 7)
        return rows.reshape(
            -1, channels, num_latent_frames, latent_height, latent_width
        ).contiguous()

    def decode(self, latents: torch.Tensor) -> torch.Tensor | None:
        """Denormalize the latents and decode them on the VAE ranks: the ImageNet-normalized
        ``(1, 3, T, H, W)`` video on rank 0's host, or the VAE's ``frame_fn`` frames."""
        if self.vae is None:
            return None
        latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1)
        z = (latents * latents_std + latents_mean).to(self.vae.dtype).to(self.device)
        return self.vae.decode(z, return_dict=False)[0].cpu()

    def _decode_audio_first(self, latent_shape: tuple[int, ...]) -> bool:
        """Whether this rank decodes its audio windows before the video: it has no video tile."""
        if self.audio_vae is None:
            return False
        if not isinstance(self.vae, DistributedAutoencoderKLMiniMaxH3):
            return self.vae is None
        return self.vae.distributed_executor.rank not in self.vae.decode_tile_ranks(latent_shape)

    def decode_audio_windows(
        self, audio_rows: torch.Tensor, num_audio_latents: int
    ) -> dict[int, torch.Tensor]:
        """This rank's decoded audio windows, ``{window index: (2, 1, samples)}`` on its host.

        The rows are channel-major; the mono audio VAE takes the two channels as two batch items.
        Window ``i`` decodes on audio rank ``i % len(audio ranks)``.
        """
        windows = audio_windows(num_audio_latents)
        owners = [self._audio_ranks[i % len(self._audio_ranks)] for i in range(len(windows))]
        rank = dist.get_rank()
        pieces = {}
        if self.audio_vae is not None:
            latents = audio_rows.reshape(MINIMAX_H3_AUDIO_CHANNELS, num_audio_latents, -1)
            latents = latents.permute(0, 2, 1).float()
            latents_mean = torch.tensor(self.audio_vae.config.latents_mean).view(1, -1, 1)
            latents_std = torch.tensor(self.audio_vae.config.latents_std).view(1, -1, 1)
            latents = latents * latents_std + latents_mean
            for i, window in enumerate(windows):
                if owners[i] == rank:
                    pieces[i] = self.audio_vae.decode_window(latents, window).float()
        return pieces

    def _try_decode_audio_windows(
        self, audio_rows: torch.Tensor, num_audio_latents: int
    ) -> tuple[dict[int, torch.Tensor], Exception | None]:
        """``decode_audio_windows``, returning its error for ``_raise_on_any_rank``."""
        try:
            return self.decode_audio_windows(audio_rows, num_audio_latents), None
        except Exception as error:
            logger.exception("MiniMax-H3 audio decode failed on rank %d", dist.get_rank())
            return {}, error

    def _raise_on_any_rank(self, error: Exception | None, stage: str) -> None:
        """Raise on every rank when ``error`` is set on any (a flag all-reduced over the world's
        CPU group), so a rank-local failure does not leave the others in the next collective."""
        failed = torch.tensor([error is not None], dtype=torch.int32)
        dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=get_world_group().cpu_group)
        if error is not None:
            raise error
        if failed.item():
            raise RuntimeError(f"MiniMax-H3 {stage} failed on another rank")

    def decode_audio(
        self,
        audio_rows: torch.Tensor,
        num_audio_latents: int,
        pieces: dict[int, torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        """The reference's audio unpack, denormalization and decode: the stereo soundtrack as
        ``(1, 2, samples)`` on rank 0's host, ``None`` elsewhere.

        ``pieces`` are this rank's windows if it already decoded them. Each window's core goes
        from its owner's host to rank 0 over the CPU group.
        """
        if pieces is None:
            pieces = self.decode_audio_windows(audio_rows, num_audio_latents)
        windows = audio_windows(num_audio_latents)
        owners = [self._audio_ranks[i % len(self._audio_ranks)] for i in range(len(windows))]
        cpu_group = get_world_group().cpu_group
        if not self.is_output_rank:
            for piece in pieces.values():
                dist.send(piece.contiguous(), dst=0, group=cpu_group)
            return None
        for i, (_, core_start, core_end) in enumerate(windows):
            if i not in pieces:
                samples = (core_end - core_start) * self._audio_hop
                pieces[i] = torch.empty((MINIMAX_H3_AUDIO_CHANNELS, 1, samples))
                dist.recv(pieces[i], src=owners[i], group=cpu_group)
        audio = torch.cat([pieces[i] for i in range(len(windows))], dim=-1)
        return audio.permute(1, 0, 2)

    def decode_frames(self, latents: torch.Tensor) -> torch.Tensor | None:
        """``decode``, as ``(1, T, H, W, 3)`` uint8 frames."""
        video = self.decode(latents)
        if video is None or video.dtype == torch.uint8:
            return video
        return _to_uint8_frames(video)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        *,
        image: Image.Image | None = None,
        last_image: Image.Image | None = None,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
        num_steps: int | None = None,
        generator: torch.Generator | None = None,
        flow_shift: float | None = None,
        audio_flow_shift: float | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, dict]:
        """Generate one video and its soundtrack.

        ``image`` / ``last_image`` are the first / last keyframes (fl2va); ``flow_shift`` /
        ``audio_flow_shift`` the video / audio sigma shifts. ``None`` arguments take the
        reference's defaults. Returns ``(video, audio, metrics)``: on the output rank, the
        ``(1, T, H, W, 3)`` uint8 frames and the ``(1, 2, samples)`` float32 soundtrack on the
        host; ``None`` on the other ranks.
        """
        t_start = time.perf_counter()
        # Each keyframe by input name, in packed order, and the end of the video it anchors.
        keyframe_inputs, keyframe_anchors = {}, []
        inputs = (("first", "image", image), ("last", "last_image", last_image))
        for anchor, key, keyframe in inputs:
            if keyframe is not None:
                if not isinstance(keyframe, Image.Image):
                    raise ValueError(
                        f"MiniMax-H3 keyframe `{key}` must be a PIL image, got "
                        f"{type(keyframe).__name__}"
                    )
                # The VAE takes three channels.
                keyframe_inputs[key] = keyframe.convert("RGB")
                keyframe_anchors.append(anchor)
        if not prompt:
            raise ValueError("MiniMax-H3 requires a text prompt")

        height, width, num_frames = self.resolve_geometry(
            height, width, num_frames, next(iter(keyframe_inputs.values()), None)
        )
        num_steps = num_steps or DEFAULT_NUM_STEPS
        if num_steps < MIN_NUM_STEPS:
            # Checked before the encodes run on every rank.
            raise ValueError(f"MiniMax-H3 needs at least {MIN_NUM_STEPS} steps, got {num_steps}")
        num_latent_frames = video_latent_num_frames(
            num_frames, VAE_FRAMES_PER_CHUNK, VAE_LATENTS_PER_CHUNK
        )
        latent_height = height // VAE_SPATIAL_COMPRESSION
        latent_width = width // VAE_SPATIAL_COMPRESSION
        num_audio_latents = audio_latent_num_frames(num_frames)
        patch_size = tuple(self.transformer.config.patch_size)
        keyframes = []
        if keyframe_inputs:
            keyframes = self._resize_keyframes(
                **keyframe_inputs, height=height, width=width, output="keyframes"
            )

        marks = StageMarks(device_events=self.is_output_rank)
        text_start = marks.mark()
        prompt_embeds, text_token_tags = self.encode_prompt(prompt, keyframes)
        vae_encode_start = marks.mark()
        conditions = [
            self.encode_keyframe(keyframe, latent_height, latent_width) for keyframe in keyframes
        ]
        vae_encode_end = marks.mark()

        layout = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            text_token_tags,
            num_latent_frames,
            latent_height,
            latent_width,
            num_audio_latents,
            patch_size,
            MINIMAX_H3_AUDIO_CHANNELS,
            MINIMAX_H3_AUDIO_TAG,
            MINIMAX_H3_VIDEO_TAG,
            keyframe_anchors=tuple(keyframe_anchors),
        )

        # The reference's draw order on the request's generator: each keyframe's noise
        # augmentation in packed order, the video noise as a latent tensor, then the audio noise
        # in row layout.
        condition_rows = []
        for condition in conditions:
            noise = randn_tensor(condition.shape, generator=generator, dtype=torch.float32)
            condition_rows.append(
                patchify_video_latents(
                    self.scheduler.scale_noise(
                        condition, self._resize_keyframes.keyframe_noise_aug, noise
                    ),
                    patch_size,
                )
            )
        latents = randn_tensor(
            (
                1,
                self.transformer.config.in_channels,
                num_latent_frames,
                latent_height,
                latent_width,
            ),
            generator=generator,
            dtype=torch.float32,
        )
        audio_rows = randn_tensor(
            (
                num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS,
                self.transformer.config.audio_in_channels,
            ),
            generator=generator,
            dtype=torch.float32,
        )

        video_rows = torch.cat([*condition_rows, patchify_video_latents(latents, patch_size)])

        shifts = ((self.scheduler, flow_shift), (self.audio_scheduler, audio_flow_shift))
        for scheduler, shift in shifts:
            scheduler.set_shift(scheduler.config.shift if shift is None else shift)
        denoise_start = marks.mark()
        video_rows, audio_rows, step_marks = self.denoise(
            prompt_embeds, layout, video_rows, audio_rows, num_steps, marks=marks
        )
        audio_first_start = marks.mark()
        # Every rank holds the same rows; checked before the decode collectives.
        finite = bool(torch.isfinite(video_rows).all() and torch.isfinite(audio_rows).all())
        self._raise_on_any_rank(
            None if finite else ValueError("Neuron pipeline produced non-finite values"), "denoise"
        )
        latent_shape = (
            1,
            self.transformer.config.in_channels,
            num_latent_frames,
            latent_height,
            latent_width,
        )
        # A rank without a video tile decodes its audio windows during the video decode.
        audio_pieces, audio_error = None, None
        if self._decode_audio_first(latent_shape):
            audio_pieces, audio_error = self._try_decode_audio_windows(audio_rows, num_audio_latents)
        vae_decode_start = marks.mark()
        video = self.decode_frames(
            self.unpack_video(video_rows, num_latent_frames, latent_height, latent_width)
        )
        audio_decode_start = marks.mark()
        if audio_pieces is None and audio_error is None:
            audio_pieces, audio_error = self._try_decode_audio_windows(audio_rows, num_audio_latents)
        # Before the windows go to rank 0: a window that failed on its owner would leave rank 0
        # waiting in its receive.
        self._raise_on_any_rank(audio_error, "audio decode")
        audio = self.decode_audio(audio_rows, num_audio_latents, audio_pieces)
        audio_decode_end = marks.mark()

        e2e_forward_seconds = time.perf_counter() - t_start
        # Read after the output rank's host copies, when its device events are complete.
        text_encode_seconds = marks.seconds(text_start, vae_encode_start)
        vae_encode_seconds = marks.seconds(vae_encode_start, vae_encode_end)
        denoise_seconds = marks.seconds(denoise_start, audio_first_start)
        vae_decode_seconds = marks.seconds(vae_decode_start, audio_decode_start)
        audio_decode_seconds = marks.seconds(audio_first_start, vae_decode_start) + marks.seconds(
            audio_decode_start, audio_decode_end
        )
        step_seconds = [marks.seconds(a, b) for a, b in zip(step_marks, step_marks[1:])]
        metrics = {
            "text_encode_seconds": text_encode_seconds,
            "vae_encode_seconds": vae_encode_seconds,
            "denoise_seconds": denoise_seconds,
            "denoise_step_seconds": step_seconds,
            "denoise_step_p50_seconds": statistics.median(step_seconds) if step_seconds else 0.0,
            "vae_decode_seconds": vae_decode_seconds,
            "audio_decode_seconds": audio_decode_seconds,
            "e2e_forward_seconds": e2e_forward_seconds,
            "frames_per_second": num_frames / e2e_forward_seconds,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "num_steps": len(step_seconds),
            "task": "fl2va " + "+".join(keyframe_anchors) if keyframes else "t2va",
        }
        if not self.is_output_rank:
            return None, None, metrics
        logger.info(
            "MiniMax-H3 %s %dx%dx%d, %d steps: text encode %.2fs, VAE encode %.2fs, denoise "
            "%.2fs, VAE decode %.2fs, audio decode %.2fs, total %.2fs",
            metrics["task"],
            height,
            width,
            num_frames,
            len(step_seconds),
            text_encode_seconds,
            vae_encode_seconds,
            denoise_seconds,
            vae_decode_seconds,
            audio_decode_seconds,
            e2e_forward_seconds,
        )
        return video, audio, metrics
