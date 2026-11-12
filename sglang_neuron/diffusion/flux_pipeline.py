# SPDX-License-Identifier: Apache-2.0
"""SGLang Diffusion pipeline that serves FLUX.1-dev with the Neuron model.

SGLang's own FLUX pipeline runs a text-encoding, latent, timestep, denoising and
decoding stage over SGLang modules, with eager tensor ops between them. The Neuron
FLUX model keeps every per-step operation inside its compiled graphs (conditioning
precomputed once per request, the Euler step inside the DiT graph). So this pipeline
has one stage that calls ``NeuronFluxModel.generate``, on every rank in lockstep.

Launch: ``--num-gpus 64 --tp-size 8`` on trn2.48xlarge. SGLang then puts all 64
ranks in one replica (``dp_size=1``) and every rank receives each request; the
Neuron model uses SGLang's TP x SP groups as its TP x CP groups (8 x 8, on the Trn2 physical
mesh; ``distributed.groups``).
"""

from __future__ import annotations

import logging
import time

import torch
from sglang.multimodal_gen.configs.pipeline_configs.flux import FluxPipelineConfig
from sglang.multimodal_gen.configs.sample.flux import FluxSamplingParams
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage

from sglang_neuron.diffusion.output import to_output_images

logger = logging.getLogger(__name__)


def _prompt_of(batch: Req) -> str:
    prompt = batch.prompt
    if isinstance(prompt, (list, tuple)):
        if len(prompt) != 1:
            raise NotImplementedError("the Neuron FLUX pipeline runs one prompt per request")
        prompt = prompt[0]
    if prompt is None:
        raise ValueError("FLUX.1-dev needs a text prompt")
    return prompt


def _generators(batch: Req, n: int) -> list[torch.Generator | None]:
    """CPU generators: image ``i`` uses the ``(i+1)``-th draw of one seeded generator.

    This is the noise convention of the CPU references (``cpu_bf16_randn_sequential_v1``),
    so the same seed gives the same image.
    """
    generator = batch.generator
    if isinstance(generator, list):
        generator = generator[0] if generator else None
    if generator is None or generator.device.type != "cpu":
        seed = batch.seed if batch.seed is not None else 0
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return [generator] * n


def _record_request_metrics(request_metrics, metrics: dict) -> None:
    """Report the model's sub-stages in SGLang's per-request metrics (milliseconds).

    SGLang profiles this whole stage as ``NeuronFluxStage``; inside it the Neuron model runs
    text encoding, conditioning, the denoising loop and VAE decoding back to back. The
    names follow SGLang's own stage names so per-stage reports read the same as for its
    GPU pipelines; ``ConditioningStage`` is FLUX's per-request DiT conditioning.
    """
    if request_metrics is None:
        return
    request_metrics.record_stage("TextEncodingStage", metrics["text_encode_seconds"])
    request_metrics.record_stage("ConditioningStage", metrics["condition_seconds"])
    request_metrics.record_stage("DenoisingStage", metrics["denoise_seconds"])
    request_metrics.denoising_stages.add("DenoisingStage")
    for seconds in metrics["denoise_step_seconds"]:
        request_metrics.record_step(seconds)
    request_metrics.record_stage("DecodingStage", metrics["vae_decode_seconds"])


class NeuronFluxStage(PipelineStage):
    """Encode, condition, denoise and decode one request with the Neuron FLUX model."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch: Req, server_args) -> OutputBatch:
        if batch.negative_prompt and (getattr(batch, "true_cfg_scale", None) or 1.0) > 1.0:
            raise NotImplementedError("true CFG with a negative prompt is not supported on Neuron")
        height = int(batch.height or 1024)
        width = int(batch.width or 1024)
        steps = int(batch.num_inference_steps or 28)
        guidance = float(batch.guidance_scale if batch.guidance_scale is not None else 3.5)
        n = int(batch.num_outputs_per_prompt or 1)
        images, metrics = self.model.generate(
            _prompt_of(batch),
            height=height,
            width=width,
            num_steps=steps,
            guidance_scale=guidance,
            generators=_generators(batch, n),
            num_images=n,
        )
        if self.model.is_output_rank:
            _record_request_metrics(batch.metrics, metrics)
            self._log_metrics(metrics)
        output = None if images is None else to_output_images(images)
        return OutputBatch(output=output, metrics=batch.metrics)

    @staticmethod
    def _log_metrics(metrics: dict) -> None:
        logger.info(
            "[NeuronFlux] %dx%d steps=%d e2e_forward=%.3fs text=%.3fs denoise=%.3fs "
            "step_p50=%.4fs vae=%.3fs",
            metrics["width"],
            metrics["height"],
            metrics["num_steps"],
            metrics["e2e_forward_seconds"],
            metrics["text_encode_seconds"],
            metrics["denoise_seconds"],
            metrics["denoise_step_p50_seconds"],
            metrics["vae_decode_seconds"],
        )


class NeuronFluxPipeline(ComposedPipelineBase):
    """FLUX.1-dev (``FluxPipeline`` checkpoints) on Neuron."""

    pipeline_name = "FluxPipeline"
    pipeline_config_cls = FluxPipelineConfig
    sampling_params_cls = FluxSamplingParams
    # The Neuron model loads its own components from the checkpoint directory.
    _required_config_modules: list[str] = []

    def load_modules(self, server_args, loaded_modules=None) -> dict:
        import torch.distributed as dist

        from sglang_neuron.diffusion import runtime
        from sglang_neuron.diffusion.envs import compile_kwargs
        from sglang_neuron.diffusion.models.flux.neuron_flux import NeuronFluxModel

        world = dist.get_world_size()
        tp = int(server_args.tp_size or 1)
        cfg = int(server_args.cfg_parallel_degree or 1)
        if cfg != 1:
            raise NotImplementedError(
                "FLUX.1-dev uses guidance embedding; run with CFG parallel off"
            )
        if world % tp:
            raise ValueError(f"world size {world} is not divisible by tp {tp}")
        cp = world // tp
        # The process group has set up the device runtime; make this rank's device current.
        runtime.set_worker_device()
        runtime.initialize_model_groups(
            tensor_parallel_size=tp, context_parallel_size=cp, cfg_parallel_size=1
        )
        model_path = runtime.local_model_path(self.model_path)
        t0 = time.perf_counter()
        # Build and load under the model dtype as torch's default dtype: the Neuron
        # modules rely on it for the parameters they create.
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            model = NeuronFluxModel(
                model_path,
                dtype=torch.bfloat16,
                vae_patch_parallel_size=world,
            )
            model.load_weights()
        finally:
            torch.set_default_dtype(previous_dtype)
        vae = getattr(model, "vae", None)
        prepare = getattr(vae, "prepare_nki_conv_dispatch", None)
        if callable(prepare):
            prepare(runtime.neuron_device())
        model.to(runtime.neuron_device())
        # Lazy: each graph compiles to a NEFF on its first call.
        model.compile(**compile_kwargs())
        logger.info(
            "Neuron FLUX.1-dev loaded in %.1fs (tp=%d cp=%d world=%d)",
            time.perf_counter() - t0,
            tp,
            cp,
            world,
        )
        runtime.freeze_gc_heap()
        return {"neuron_flux": model}

    def initialize_pipeline(self, server_args) -> None:
        return None

    def create_pipeline_stages(self, server_args) -> None:
        self.add_stage(NeuronFluxStage(self.modules["neuron_flux"]), "neuron_flux_stage")
