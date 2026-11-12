# SPDX-License-Identifier: Apache-2.0
"""FLUX.1-dev on Neuron; SGLang runs it through ``NeuronFluxStage`` (``flux_pipeline.py``).

``NeuronFluxModel`` holds compiled CLIP-L and T5-XXL encoders, the TP x CP DiT with the
Euler step inside its graph, and the patch-parallel VAE decoder. ``generate`` runs one
prompt.
"""

import statistics
import time
from typing import NamedTuple

import numpy as np
import torch
import torch.distributed as dist
from diffusers.models.transformers.transformer_flux import (
    FluxTransformer2DModel as DiffusersFluxTransformer2DModel,
)
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.configs.pipeline_configs.base import pack_latents_2x2
from sglang.multimodal_gen.runtime.pipelines_core.diffusion_scheduler_utils import (
    calculate_linear_shift,
    retrieve_timesteps,
)
from torch import nn
from transformers import CLIPTokenizer, T5TokenizerFast

from sglang_neuron.diffusion.distributed.autoencoders.autoencoder_kl_flux import build_flux_vae
from sglang_neuron.diffusion.distributed.groups import get_world_group
from sglang_neuron.diffusion.models.flux.flux_transformer import (
    T5_SEQ_LEN,
    TRANSFORMER_COMPILER_ARGS,
    NeuronFluxTransformer2DModel,
    compile_unit,
)
from sglang_neuron.diffusion.models.flux.text_encoders import (
    NeuronCLIPTextEncoder,
    NeuronT5Encoder,
)
from sglang_neuron.diffusion.runtime import neuron_device
from sglang_neuron.diffusion.stage_timing import StageMarks


class FluxPromptEmbeds(NamedTuple):
    """Device-resident text conditioning for one prompt."""

    prompt_embeds: torch.Tensor  # T5 hidden states [1, 512, 4096]
    pooled: torch.Tensor  # CLIP pooled output [1, 768]



def unpack_latents_2x2(latents: torch.Tensor, h_lat: int, w_lat: int) -> torch.Tensor:
    """Inverse of SGLang's ``pack_latents_2x2``: ``[B, (h/2)(w/2), 4C]`` -> ``[B, C, h, w]``."""
    batch_size, _, packed_channels = latents.shape
    channels = packed_channels // 4
    x = latents.view(batch_size, h_lat // 2, w_lat // 2, channels, 2, 2)
    x = x.permute(0, 3, 1, 4, 2, 5)
    return x.reshape(batch_size, channels, h_lat, w_lat)

class NeuronFluxModel(nn.Module):
    """FLUX.1-dev on Neuron: compiled text encoders, TP/CP DiT, distributed VAE, per-image loop."""

    def __init__(
        self,
        model_path: str,
        *,
        dtype: torch.dtype = torch.bfloat16,
        vae_patch_parallel_size: int = 64,
        model_config: dict | None = None,
    ):
        nn.Module.__init__(self)
        self.model_path = model_path
        self.device = neuron_device()
        model = model_path
        local_files_only = True

        model_config = dict(model_config or {})
        t5_precision = model_config.pop("t5_precision", "bf16")

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model, subfolder="tokenizer", local_files_only=local_files_only
        )
        self.tokenizer_2 = T5TokenizerFast.from_pretrained(
            model, subfolder="tokenizer_2", local_files_only=local_files_only
        )
        self.text_encoder = NeuronCLIPTextEncoder(model, dtype)
        self.text_encoder_2 = NeuronT5Encoder(model, t5_precision, output_dtype=dtype)
        self.vae, _ = build_flux_vae(model, dtype, vae_patch_parallel_size)
        self.is_vae_rank = self.vae is not None

        cfg = DiffusersFluxTransformer2DModel.load_config(
            model, subfolder="transformer", local_files_only=local_files_only
        )
        cfg.update(model_config)
        if not cfg.get("guidance_embeds", False):
            raise ValueError(
                "NeuronFluxModel supports guidance-distilled FLUX.1-dev checkpoints only"
            )
        self.transformer = NeuronFluxTransformer2DModel(**cfg)

        self.is_output_rank = dist.get_rank() == 0
        self._schedules: dict[tuple, tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]] = {}

    def generate(
        self,
        prompt: str,
        *,
        height: int = 1024,
        width: int = 1024,
        num_steps: int = 28,
        guidance_scale: float = 3.5,
        generators: list[torch.Generator | None] | None = None,
        num_images: int = 1,
    ) -> tuple[torch.Tensor | None, dict]:
        """Generate ``num_images`` images for one prompt, one compiled image at a time.

        Returns ``(images, metrics)``: on the output rank, a ``[n, 3, H, W]`` bf16 CPU
        tensor in [-1, 1]; ``None`` on the other ranks.
        """
        t_start = time.perf_counter()
        n = max(1, num_images)
        if generators is None:
            generators = [None] * n
        self.transformer.sequence_layout(height, width)
        marks = StageMarks(device_events=self.is_output_rank)

        text_start = marks.mark()
        embeds = self.encode_prompt(prompt)
        condition_start = marks.mark()
        # Copy the noise to the device before the condition graph starts. A host-to-device copy
        # made while that graph runs can corrupt one of its modulation outputs on one rank, and
        # that changes every image of the request. The outputs are validated with this order;
        # it is not known whether torch-neuronx needs it.
        noise = [self.prepare_latents(height, width, generator) for generator in generators]
        cond = self.condition(embeds, height, width, num_steps, guidance_scale)
        condition_end = marks.mark()
        spans = []  # (denoise start, decode start, decode end) per image
        step_marks: list = []
        images = []
        for i in range(n):
            denoise_start = marks.mark()
            latents, step_marks = self.denoise(
                noise[i], cond, height, width, num_steps, guidance_scale, marks=marks
            )
            decode_start = marks.mark()
            image = self.decode(latents, height, width)
            spans.append((denoise_start, decode_start, marks.mark()))
            if image is not None:
                images.append(image)
            if i < n - 1:
                dist.all_reduce(torch.zeros(1), group=get_world_group().cpu_group)

        e2e_forward_seconds = time.perf_counter() - t_start
        # After the output rank's image copy: its device events are complete.
        text_encode_seconds = marks.seconds(text_start, condition_start)
        condition_seconds = marks.seconds(condition_start, condition_end)
        denoise_seconds = sum(marks.seconds(start, end) for start, end, _ in spans)
        vae_decode_seconds = sum(marks.seconds(start, end) for _, start, end in spans)
        step_seconds = [marks.seconds(a, b) for a, b in zip(step_marks, step_marks[1:])]
        metrics = {
            "text_encode_seconds": text_encode_seconds,
            "condition_seconds": condition_seconds,
            "denoise_seconds": denoise_seconds,
            "denoise_step_seconds": step_seconds,
            "denoise_step_p50_seconds": statistics.median(step_seconds) if step_seconds else 0.0,
            "vae_decode_seconds": vae_decode_seconds,
            "e2e_forward_seconds": e2e_forward_seconds,
            "num_images": n,
            "height": height,
            "width": width,
            "num_steps": len(step_seconds),
            "guidance_scale": guidance_scale,
        }
        if not self.is_output_rank:
            return None, metrics
        return torch.cat(images), metrics

    def to(self, *args, **kwargs):
        """Move the text encoders, transformer and (on VAE ranks) the VAE."""
        for module in (self.text_encoder, self.text_encoder_2, self.transformer, self.vae):
            if module is not None:
                module.to(*args, **kwargs)
        return self

    def load_weights(self):
        """Load T5 and DiT weights from the model directory; CLIP and VAE load in __init__."""
        import os

        model_path = self.model_path
        self.text_encoder_2.load_weights(os.path.join(model_path, "text_encoder_2"))
        self.transformer.load_weights(os.path.join(model_path, "transformer"))

    def compile(self, *args, **kwargs):
        """Wrap every device component and pipeline helper with lazy torch.compile."""
        self.text_encoder.compile(*args, **kwargs)
        self.text_encoder_2.compile(*args, **kwargs)
        self.transformer.compile(*args, **kwargs)
        if self.is_vae_rank:
            self.vae.compile(*args, **kwargs)
        self._vae_input_graph = compile_unit(
            self.vae_input, "flux_vae_input", TRANSFORMER_COMPILER_ARGS, *args, **kwargs
        )
        return self

    def encode_prompt(self, prompt: str) -> FluxPromptEmbeds:
        """Submit CLIP-L (pooled) and T5-XXL (512 tokens, unmasked) for one prompt; no wait."""
        clip_ids = self.tokenizer(
            prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
        ).input_ids
        pooled = self.text_encoder(
            clip_ids.to(self.device), self.text_encoder.eos_index(clip_ids).to(self.device)
        )
        t5_ids = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=T5_SEQ_LEN,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        prompt_embeds = self.text_encoder_2(t5_ids.to(self.device))
        return FluxPromptEmbeds(prompt_embeds, pooled)

    def prepare_latents(
        self, height: int, width: int, generator: torch.Generator | None
    ) -> torch.Tensor:
        """Draw one (1, 16, h, w) bf16 CPU noise sample, pack it on the host and upload it."""
        h, w = 2 * (height // 16), 2 * (width // 16)
        noise = torch.randn((1, 16, h, w), generator=generator, device="cpu", dtype=torch.bfloat16)
        return pack_latents_2x2(noise, 1, 16, h, w).contiguous().to(self.device)

    def prepare_timesteps(self, num_inference_steps, sigmas, image_seq_len):
        """Flow-match timesteps with the image-size shift ``mu``: SGLang's FLUX helpers
        (``calculate_linear_shift``, ``retrieve_timesteps``) on the checkpoint's diffusers scheduler."""
        sigmas = (
            np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            if sigmas is None
            else sigmas
        )
        mu = calculate_linear_shift(
            image_seq_len,
            base_seq_len=self.scheduler.config.get("base_image_seq_len", 256),
            max_seq_len=self.scheduler.config.get("max_image_seq_len", 4096),
            base_shift=self.scheduler.config.get("base_shift", 0.5),
            max_shift=self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            sigmas=sigmas,
            mu=mu,
        )
        return timesteps, num_inference_steps

    def _schedule(self, height, width, num_steps, guidance_scale):
        """Stacked timestep projections, guidance projection and per-step Euler dt on device."""
        key = (height, width, num_steps, guidance_scale)
        if key not in self._schedules:
            timesteps, _ = self.prepare_timesteps(num_steps, None, (height // 16) * (width // 16))
            sigmas = self.scheduler.sigmas
            t_proj = self.transformer.time_projection(
                timesteps.to(torch.bfloat16) / 1000 * 1000
            ).to(self.device)
            g_model = torch.full([1], guidance_scale, dtype=torch.float32).to(torch.bfloat16) * 1000
            g_proj = self.transformer.time_projection(g_model).to(self.device)
            dt = [
                (sigmas[i + 1] - sigmas[i]).reshape(1).to(self.device)
                for i in range(len(timesteps))
            ]
            self._schedules[key] = (t_proj, g_proj, dt)
        return self._schedules[key]

    def condition(self, embeds, height, width, num_steps, guidance_scale):
        """Per-request DiT conditioning: text tokens and all steps' modulation rows."""
        layout = self.transformer.sequence_layout(height, width)
        t_proj, g_proj, _ = self._schedule(height, width, num_steps, guidance_scale)
        return self.transformer.condition(
            embeds.prompt_embeds, embeds.pooled, t_proj, g_proj, layout
        )

    def denoise(self, latents, cond, height, width, num_steps, guidance_scale, marks=None):
        """Run the flow-matching Euler loop; returns final latents and the step marks.

        ``marks`` (``StageMarks``, host clock by default) marks the start and the end of every
        step: ``num_steps + 1`` marks; read them with ``marks.seconds`` after the request.
        """
        layout = self.transformer.sequence_layout(height, width)
        _, _, dt = self._schedule(height, width, num_steps, guidance_scale)
        marks = marks or StageMarks(device_events=False)
        step_marks = [marks.mark()]
        for i in range(len(dt)):
            latents = self.transformer(latents, cond, i, layout, dt[i])
            step_marks.append(marks.mark())
        return latents, step_marks

    def decode(self, latents, height, width):
        """Decode packed latents on the VAE group; rank 0 returns a [1, 3, H, W] bf16 CPU image."""
        if not self.is_vae_rank:
            return None
        z = self._vae_input_graph(
            latents, height, width, self.vae.config.scaling_factor, self.vae.config.shift_factor
        )
        image = self.vae.decode(z, return_dict=False)[0]
        if not self.is_output_rank:
            return None
        return image.to("cpu")

    @staticmethod
    def vae_input(latents, height, width, scaling_factor, shift_factor):
        """Unpack [1, N, 64] latents to [1, 16, H/8, W/8] and undo the VAE scaling and shift."""
        z = unpack_latents_2x2(latents, height // 8, width // 8)
        return z / scaling_factor + shift_factor

