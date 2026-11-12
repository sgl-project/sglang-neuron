# SPDX-License-Identifier: Apache-2.0
"""SGLang Diffusion pipeline that serves MiniMax-H3 (t2va, fl2va) with the Neuron model.

Requests are SGLang's MiniMax-H3 requests (``task``, ``conditions``, ``target``), resolved by its
admission before the queue. One stage runs ``NeuronMiniMaxH3Model.generate`` on every rank in
lockstep. Launch with ``--num-gpus 64 --tp-size 8`` on trn2.48xlarge; the model builds its own
TP x CP groups.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch
import torch.distributed as dist
from PIL import Image
from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
from sglang.multimodal_gen.configs.sample.minimax_h3 import MiniMaxH3SamplingParams
from sglang.multimodal_gen.configs.task_type import ModelTaskType
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
    material_io,
    resolved_plan,
)
from sglang.multimodal_gen.runtime.utils.vision import load_image

from sglang_neuron.diffusion.distributed.groups import get_world_group
from sglang_neuron.diffusion.models.minimax_h3 import MINIMAX_H3_CLASS_NAME

logger = logging.getLogger(__name__)

# What t2va and fl2va read from the checkpoint (about 145 GB of its 498 GB).
MINIMAX_H3_ALLOW_PATTERNS = (
    "model_index.json",
    "text_encoder/*",
    "tokenizer/*",
    "processor/*",
    "transformer/*",
    "vae/*",
    "audio_vae/*",
    "scheduler/*",
    "audio_scheduler/*",
)

@dataclass
class NeuronMiniMaxH3PipelineConfig(PipelineConfig):
    """Text (+ keyframes) to video; the Neuron model owns its components.

    ``transformer_overrides`` (``--pipeline-config``) update the DiT's checkpoint config.
    """

    task_type: ModelTaskType = ModelTaskType.TI2V
    transformer_overrides: dict = field(default_factory=dict)


MINIMAX_H3_NEURON_TASKS = ("t2va", "fl2va")


@dataclass
class NeuronMiniMaxH3SamplingParams(MiniMaxH3SamplingParams):
    """SGLang's MiniMax-H3 sampling parameters, limited to what the Neuron model runs."""

    # The floor SGLang's warmup request takes: the model's scheduler needs two steps.
    min_num_inference_steps: ClassVar[int] = 2

    # The admission sets the frame count.
    adjust_frames: bool = False

    def _validate(self) -> None:
        super()._validate()
        if self.task is not None and self.task not in MINIMAX_H3_NEURON_TASKS:
            raise ValueError(
                f"MiniMax-H3 on Neuron serves the tasks {', '.join(MINIMAX_H3_NEURON_TASKS)}, "
                f"got {self.task!r}"
            )
        if int(self.num_outputs_per_prompt or 1) > 1:
            raise ValueError("MiniMax-H3 on Neuron generates one video per request")
        unsupported = [
            name
            for name in (
                "imgvid_cond_noise_aug_for_inference",
                "audio_cond_noise_aug_for_inference",
            )
            if getattr(self, name) is not None
        ] + [
            name
            for name in ("enable_cache_dit", "enable_spectrum", "rollout")
            if getattr(self, name)
        ]
        if unsupported:
            raise ValueError(f"MiniMax-H3 on Neuron does not support {', '.join(unsupported)}")


def _local_model_path(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    from huggingface_hub import snapshot_download

    return snapshot_download(model_path, allow_patterns=list(MINIMAX_H3_ALLOW_PATTERNS))


def _prompt_of(batch: Req) -> str:
    prompt = batch.prompt
    if isinstance(prompt, (list, tuple)):
        if len(prompt) != 1:
            raise NotImplementedError("the Neuron MiniMax-H3 pipeline runs one prompt per request")
        prompt = prompt[0]
    if not prompt:
        raise ValueError("MiniMax-H3 needs a text prompt")
    return prompt


def _keyframes(batch: Req, plan) -> tuple[Image.Image | None, Image.Image | None]:
    """The plan's first (``frame_index`` 0) and last (-1) keyframes, if any."""
    keyframes: dict[int, Any] = {}
    for material in plan.materials:
        if material.material_chain != "image.target_canvas" or material.frame_index not in (0, -1):
            raise NotImplementedError(
                f"MiniMax-H3 on Neuron takes keyframes at frame_index 0 and -1, got the "
                f"{material.role} {material.condition_type} conditions[{material.condition_index}]"
            )
        keyframes[material.frame_index] = material

    def localize(material) -> str:
        return material_io.minimax_h3_localize_material_uri(
            batch,
            material.uri,
            condition_type=material.condition_type,
            condition_index=int(material.condition_index),
        )

    return tuple(_load_keyframes((keyframes.get(0), keyframes.get(-1)), localize))


def _load_keyframes(sources, localize=lambda source: source) -> list[Image.Image | None]:
    """Each source (its local file, ``localize``) loaded with SGLang's ``load_image`` on world
    rank 0, then broadcast to every rank over the world's CPU group.

    A URL is fetched once, and a failed load raises on every rank rather than leaving the others
    waiting in the text encoder's first collective.
    """

    def load():
        return [None if source is None else load_image(localize(source)) for source in sources]

    if not dist.is_initialized() or dist.get_world_size() == 1:
        return load()
    payload = [None]
    if dist.get_rank() == 0:
        try:
            payload = [("ok", [None if im is None else (im.size, im.tobytes()) for im in load()])]
        except Exception as e:  # noqa: BLE001 - every rank raises it below
            payload = [("error", f"{type(e).__name__}: {e}")]
    dist.broadcast_object_list(payload, src=0, group=get_world_group().cpu_group)
    status, value = payload[0]
    if status == "error":
        raise ValueError(f"MiniMax-H3 could not load a keyframe: {value}")
    return [None if v is None else Image.frombytes("RGB", *v) for v in value]


def _generator(batch: Req) -> torch.Generator:
    """A CPU generator seeded with the request's seed: the noise is the same on every host."""
    generator = batch.generator
    if isinstance(generator, list):
        generator = generator[0] if generator else None
    if generator is None or generator.device.type != "cpu":
        seed = batch.seed if batch.seed is not None else 0
        if isinstance(seed, list):
            seed = seed[0]
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return generator


def _freeze_gc_heap() -> None:
    """Move the loaded model's objects out of the collector's reach, so a request never pays
    for a full collection over them."""
    import gc

    gc.collect()
    gc.freeze()


def _record_request_metrics(request_metrics, metrics: dict) -> None:
    """Report the model's sub-stages in SGLang's per-request metrics, under SGLang's stage names."""
    if request_metrics is None:
        return
    request_metrics.record_stage("TextEncodingStage", metrics["text_encode_seconds"])
    request_metrics.record_stage("ImageVAEEncodingStage", metrics["vae_encode_seconds"])
    request_metrics.record_stage("DenoisingStage", metrics["denoise_seconds"])
    request_metrics.denoising_stages.add("DenoisingStage")
    for seconds in metrics["denoise_step_seconds"]:
        request_metrics.record_step(seconds)
    request_metrics.record_stage("DecodingStage", metrics["vae_decode_seconds"])
    request_metrics.record_stage("AudioDecodingStage", metrics["audio_decode_seconds"])


class NeuronMiniMaxH3Stage(PipelineStage):
    """Encode, denoise and decode one request with the Neuron MiniMax-H3 model."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch: Req, server_args) -> OutputBatch:
        if int(batch.num_outputs_per_prompt or 1) > 1:
            raise NotImplementedError("MiniMax-H3 on Neuron generates one video per request")
        plan = resolved_plan.minimax_h3_plan_from_batch(batch)
        if plan is None:
            raise ValueError("MiniMax-H3 needs a request with task, conditions and target")
        shape = plan.shape
        image, last_image = _keyframes(batch, plan)
        video, audio, metrics = self.model.generate(
            _prompt_of(batch),
            image=image,
            last_image=last_image,
            height=int(shape["height"]),
            width=int(shape["width"]),
            num_frames=int(shape["frame_count"]),
            num_steps=batch.num_inference_steps,
            generator=_generator(batch),
            flow_shift=plan.flow_shift,
            audio_flow_shift=plan.audio_flow_shift,
        )
        if not self.model.is_output_rank:
            return OutputBatch(output=None, metrics=batch.metrics)
        _record_request_metrics(batch.metrics, metrics)
        return OutputBatch(
            # One (T, H, W, 3) uint8 array per output: SGLang's ready-to-encode frame layout.
            output=video.numpy(),
            audio=audio,
            audio_sample_rate=self.model.audio_sample_rate,
            fps=self.model.fps,
            metrics=batch.metrics,
        )


class NeuronMiniMaxH3Pipeline(ComposedPipelineBase):
    """MiniMax-H3 (``MiniMaxH3ModularPipeline`` checkpoints) on Neuron."""

    pipeline_name = MINIMAX_H3_CLASS_NAME
    is_video_pipeline = True
    pipeline_config_cls = NeuronMiniMaxH3PipelineConfig
    sampling_params_cls = NeuronMiniMaxH3SamplingParams
    # The Neuron model loads its own components from the checkpoint directory.
    _required_config_modules: list[str] = []

    def load_modules(self, server_args, loaded_modules=None) -> dict:
        import torch.distributed as dist

        from sglang_neuron.diffusion import runtime
        from sglang_neuron.diffusion.envs import compile_kwargs
        from sglang_neuron.diffusion.models.minimax_h3.neuron_minimax_h3 import (
            NeuronMiniMaxH3Model,
        )

        world = dist.get_world_size()
        tp = int(server_args.tp_size or 1)
        if int(server_args.cfg_parallel_degree or 1) != 1:
            raise NotImplementedError("MiniMax-H3 is guidance-distilled; run with CFG parallel off")
        if world % tp:
            raise ValueError(f"world size {world} is not divisible by tp {tp}")
        cp = world // tp
        # The process group has set up the device runtime; make this rank's device current.
        runtime.set_worker_device()
        runtime.initialize_model_groups(
            tensor_parallel_size=tp, context_parallel_size=cp, cfg_parallel_size=1
        )
        model_path = _local_model_path(self.model_path)
        t0 = time.perf_counter()
        # The Neuron modules create their parameters in the model dtype (torch's default here).
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            model = NeuronMiniMaxH3Model(
                model_path,
                dtype=torch.bfloat16,
                vae_patch_parallel_size=world,
                model_config=server_args.pipeline_config.transformer_overrides,
            )
            model.load_weights()
        finally:
            torch.set_default_dtype(previous_dtype)
        model.to(runtime.neuron_device())
        # Lazy: each graph compiles to a NEFF on its first call.
        model.compile(**compile_kwargs())
        logger.info(
            "Neuron MiniMax-H3 loaded in %.1fs (tp=%d cp=%d world=%d)",
            time.perf_counter() - t0,
            tp,
            cp,
            world,
        )
        _freeze_gc_heap()
        return {"neuron_minimax_h3": model}

    def initialize_pipeline(self, server_args) -> None:
        return None

    def create_pipeline_stages(self, server_args) -> None:
        self.add_stage(
            NeuronMiniMaxH3Stage(self.modules["neuron_minimax_h3"]), "neuron_minimax_h3_stage"
        )
