# SPDX-License-Identifier: Apache-2.0
"""SGLang Diffusion pipelines that serve Wan2.2 A14B (T2V and I2V) with the Neuron model.

SGLang's own Wan pipelines run text encoding, latent preparation, timestep preparation,
denoising and decoding as separate stages over SGLang modules. The Neuron Wan model keeps every
per-step operation inside its compiled graphs and lays out its own TP x CP x CFG groups, so these
pipelines have one stage that calls ``NeuronWanModel.generate``, on every rank in lockstep (as
``flux_pipeline.NeuronFluxPipeline`` does for FLUX.1-dev).

Launch: ``--num-gpus 64 --tp-size 4 --cfg-parallel-size 2`` on trn2.48xlarge (the TP4 x CP8 x
CFG2 layout). CP is SGLang's SP degree, which SGLang resolves to world / (TP * CFG), and the CP
groups are SGLang's SP groups. The VAE patch-parallel degree is
``SGLANG_NEURON_WAN_VAE_PATCH_PARALLEL_SIZE`` (``envs.py``).

The Neuron pipelines run Wan2.2 A14B BF16 checkpoints only (two experts and a ``boundary_ratio``
in ``model_index.json``, no ``quantization_config``). SGLang picks a pipeline by
``model_index.json``'s ``_class_name``, which every Wan checkpoint shares, so every ``WanPipeline``
/ ``WanImageToVideoPipeline`` checkpoint reaches them; any other one (Wan2.1, Wan2.2 TI2V-5B, a
quantized export) is refused when the pipeline is constructed, before anything loads.

Defaults follow the reference pipeline the goldens were made with, not SGLang's (``NeuronWan22*``
configs below): flow shift 5.0 (SGLang's T2V default is 12.0), guidance 5.0 for both experts, and
an empty negative prompt ("" is encoded when a request has none). The configs are registered for the
``Wan-AI/Wan2.2-{T2V,I2V}-A14B-Diffusers`` paths, so SGLang resolves them for the Hub IDs and for
local directories of that basename (or ``--model-id Wan2.2-T2V-A14B-Diffusers``).
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import os
import time
from dataclasses import dataclass, field

import torch
from sglang.multimodal_gen.configs.pipeline_configs.wan import (
    Wan2_2_I2V_A14B_Config,
    Wan2_2_T2V_A14B_Config,
)
from sglang.multimodal_gen.configs.sample.wan import (
    Wan2_2_I2V_A14B_SamplingParam,
    Wan2_2_T2V_A14B_SamplingParam,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage

from sglang_neuron.diffusion.models.wan2_2.checkpoint import is_wan22_a14b
from sglang_neuron.diffusion.output import to_golden_frames, to_output_video

logger = logging.getLogger(__name__)

# The Wan2.2 I2V and T2V configs (TP4 x CP8 x CFG2) run one or two CFG replicas.
SUPPORTED_CFG_PARALLEL_SIZES = (1, 2)

# The flow shift and guidance scale the goldens were made with.
REFERENCE_FLOW_SHIFT = 5.0
REFERENCE_GUIDANCE_SCALE = 5.0
# (width, height) pairs. The first is SGLang's default when a request sets neither: 480 x 832.
# T2V also takes 1080 (rounded down to 1072 in the model); I2V does not round,
# so it lists only multiples of 16.
_T2V_RESOLUTIONS = (
    (832, 480),
    (480, 832),
    (1280, 720),
    (720, 1280),
    (1920, 1072),
    (1072, 1920),
    (1920, 1080),
    (1080, 1920),
)
_I2V_RESOLUTIONS = (
    (832, 480),
    (480, 832),
    (1280, 720),
    (720, 1280),
    (1920, 1072),
    (1072, 1920),
)


@dataclass
class NeuronWan22T2VA14BConfig(Wan2_2_T2V_A14B_Config):
    """SGLang's Wan2.2 T2V A14B config with the reference flow shift (5.0, not 12.0)."""

    flow_shift: float | None = REFERENCE_FLOW_SHIFT


@dataclass
class NeuronWan22I2VA14BConfig(Wan2_2_I2V_A14B_Config):
    """SGLang's Wan2.2 I2V A14B config (its flow shift, 5.0, is the reference one)."""

    flow_shift: float | None = REFERENCE_FLOW_SHIFT


@dataclass
class NeuronWan22T2VA14BSamplingParam(Wan2_2_T2V_A14B_SamplingParam):
    """The reference request defaults: guidance 5.0 for both experts, no negative prompt,
    480 x 832, 81 frames whatever the GPU count.

    Two differences from the reference pipeline a request can hit (behaviour kept as is):

    - A request ``flow_shift`` is applied on top of a scheduler built with that same shift (the
      double shift the configured shift also gets); the reference pipeline instead rebuilds an
      upstream scheduler with ``shift=1.0`` for a request override.
    - ``guidance_scale_2`` defaults to 5.0 here, so a request that sets only ``guidance_scale``
      keeps 5.0 for the low-noise expert; the reference pipeline copies ``guidance_scale`` into
      it.
    """

    negative_prompt: str | None = ""
    guidance_scale: float = REFERENCE_GUIDANCE_SCALE
    guidance_scale_2: float = REFERENCE_GUIDANCE_SCALE
    supported_resolutions: list[tuple[int, int]] | None = field(
        default_factory=lambda: list(_T2V_RESOLUTIONS)
    )

    def __post_init__(self) -> None:
        # The Neuron model shards the sequence itself; SGLang's frame alignment for
        # --num-gpus (81 -> 253 frames at 64) must not touch num_frames.
        self.adjust_frames = False
        super().__post_init__()


@dataclass
class NeuronWan22I2VA14BSamplingParam(Wan2_2_I2V_A14B_SamplingParam):
    """The reference request defaults: guidance 5.0 for both experts, no negative prompt,
    480 x 832, 81 frames whatever the GPU count (see ``NeuronWan22T2VA14BSamplingParam`` for the
    request ``flow_shift`` and ``guidance_scale_2`` differences from the reference pipeline)."""

    negative_prompt: str | None = ""
    guidance_scale: float = REFERENCE_GUIDANCE_SCALE
    guidance_scale_2: float = REFERENCE_GUIDANCE_SCALE
    supported_resolutions: list[tuple[int, int]] | None = field(
        default_factory=lambda: list(_I2V_RESOLUTIONS)
    )

    def __post_init__(self) -> None:
        self.adjust_frames = False
        super().__post_init__()


def reference_vae_patch_parallel_size(i2v: bool, height: int, width: int) -> int:
    """The reference VAE patch-parallel degree for one task and size.

    T2V 16 / 32 / 64 and I2V 16 / 32 / 32 at 480p / 720p / 1080p. Other sizes take the degree of
    the smallest of these that holds them.
    """
    pixels = height * width
    if pixels <= 480 * 832:
        return 16
    if pixels <= 720 * 1280:
        return 32
    return 32 if i2v else 64


def _load_resolution(server_args, sampling_params_cls) -> tuple[int, int, str]:
    """``(height, width, source)`` the VAE group is sized for at load time: the first
    ``--warmup-resolutions`` entry (``WxH``), else the sampling default (480 x 832)."""
    resolutions = getattr(server_args, "warmup_resolutions", None) or []
    for resolution in resolutions:
        try:
            width, height = (int(v) for v in str(resolution).lower().split("x"))
            return height, width, "--warmup-resolutions"
        except ValueError:
            continue
    width, height = sampling_params_cls().supported_resolutions[0]
    return height, width, "the default resolution"


@functools.lru_cache(maxsize=64)
def _cached_is_wan22_a14b(model_path: str, class_name: str) -> bool:
    return is_wan22_a14b(model_path, class_name)


def neuron_model_info(model_info, model_path: str):
    """SGLang's ``ModelInfo`` for ``model_path`` with the Neuron A14B configs when it is one.

    SGLang resolves the configs by the checkpoint's directory basename, so an A14B checkpoint in
    a directory of another name (``Wan2.2-T2V-A14B``, a symlink, a snapshot hash) would get the
    Wan2.1 configs (flow shift 3.0, guidance 3.0, 50 steps). The pipeline class is chosen from
    ``model_index.json`` already; this makes the configs follow it.
    """
    if model_info is None:
        return model_info
    for pipeline in (NeuronWanPipeline, NeuronWanI2VPipeline):
        if (
            model_info.pipeline_cls is pipeline
            and (
                model_info.pipeline_config_cls is not pipeline.pipeline_config_cls
                or model_info.sampling_param_cls is not pipeline.sampling_params_cls
            )
            and _cached_is_wan22_a14b(model_path, pipeline.pipeline_name)
        ):
            logger.info(
                "%s is a Wan2.2 A14B checkpoint: using %s / %s instead of %s / %s",
                model_path,
                pipeline.pipeline_config_cls.__name__,
                pipeline.sampling_params_cls.__name__,
                model_info.pipeline_config_cls.__name__,
                model_info.sampling_param_cls.__name__,
            )
            return dataclasses.replace(
                model_info,
                pipeline_config_cls=pipeline.pipeline_config_cls,
                sampling_param_cls=pipeline.sampling_params_cls,
            )
    return model_info


def install_model_info_resolution() -> None:
    """Wrap SGLang's ``registry.get_model_info`` with ``neuron_model_info``; idempotent.

    SGLang's callers import ``get_model_info`` from the registry module when they run, so they
    see the wrapper; the wrapper keeps ``cache_clear`` (the registry calls it on registration).
    """
    from sglang.multimodal_gen import registry

    original = registry.get_model_info
    if getattr(original, "_sglang_neuron_wan", False):
        return

    def get_model_info(model_path, backend=None, model_id=None):
        return neuron_model_info(original(model_path, backend, model_id), model_path)

    get_model_info.cache_clear = original.cache_clear
    get_model_info.__wrapped__ = original
    get_model_info.__doc__ = original.__doc__
    get_model_info._sglang_neuron_wan = True
    registry.get_model_info = get_model_info


def rebuild_pipeline_config(config, config_cls, server_args=None):
    """A ``config_cls`` instance carrying the values set on ``config``.

    The scalar fields of ``config`` that the user set explicitly (``ServerArgs`` records them) or
    that differ from its own class's defaults are kept; every other field takes ``config_cls``'s
    default.
    """
    rebuilt = config_cls()
    defaults = type(config)()
    names = {f.name for f in dataclasses.fields(rebuilt)}
    explicit = getattr(server_args, "is_arg_explicitly_set", lambda name: False)
    for spec in dataclasses.fields(config):
        value = getattr(config, spec.name)
        if spec.name not in names or not isinstance(value, (int, float, str, bool, type(None))):
            continue
        if explicit(spec.name) or value != getattr(defaults, spec.name, value):
            setattr(rebuilt, spec.name, value)
    return rebuilt


def parallel_sizes(server_args, world_size: int) -> tuple[int, int, int]:
    """``(tp, cp, cfg)`` of a Wan launch: CFG 1 or 2, CP the rest of the world.

    FLUX.1-dev's pipeline rejects CFG parallel (it uses guidance embedding); Wan2.2 runs its
    two CFG branches on two replicas.
    """
    tp = int(server_args.tp_size or 1)
    cfg = int(getattr(server_args, "cfg_parallel_degree", None) or 1)
    if cfg not in SUPPORTED_CFG_PARALLEL_SIZES:
        raise NotImplementedError(
            f"Wan2.2 on Neuron runs CFG parallel {SUPPORTED_CFG_PARALLEL_SIZES}, got {cfg}"
        )
    if world_size % (tp * cfg):
        raise ValueError(f"world size {world_size} is not divisible by tp {tp} x cfg {cfg}")
    return tp, world_size // (tp * cfg), cfg


def _prompt_of(batch: Req) -> str:
    prompt = batch.prompt
    if isinstance(prompt, (list, tuple)):
        if len(prompt) != 1:
            raise NotImplementedError("the Neuron Wan pipeline runs one prompt per request")
        prompt = prompt[0]
    if not prompt:
        raise ValueError("Wan2.2 needs a text prompt")
    return prompt


def _negative_prompt_of(batch: Req) -> str:
    negative = batch.negative_prompt
    if isinstance(negative, (list, tuple)):
        negative = negative[0] if negative else None
    return negative or ""


def generators_for(batch: Req, n: int) -> list[torch.Generator]:
    """One CPU generator per request seeded with the request's seed, shared by its ``n`` videos.

    The goldens were drawn the same way: a fresh CPU generator per request and, for
    ``num_outputs_per_prompt`` videos, each one drawn from it in turn. A CPU generator the caller
    set is used as given.
    """
    generator = batch.generator
    if isinstance(generator, list):
        generator = generator[0] if generator else None
    if generator is None or generator.device.type != "cpu":
        seed = batch.seed
        if isinstance(seed, (list, tuple)):
            seed = seed[0] if seed else None
        generator = torch.Generator(device="cpu").manual_seed(int(seed if seed is not None else 0))
    return [generator] * n


def _image_of(batch: Req):
    """The first-frame image: SGLang's loaded ``condition_image``, else ``image_path``."""
    image = getattr(batch, "condition_image", None)
    if image is None:
        image = batch.image_path
    if isinstance(image, (list, tuple)):
        if len(image) > 1:
            logger.warning("Received multiple images for one I2V request; using the first.")
        image = image[0] if image else None
    return image


def request_parameters(batch: Req, server_args, *, i2v: bool) -> dict:
    """``NeuronWanModel.generate`` keyword arguments for one request.

    Flow shift: the request's, else the server's (``--flow-shift``, default the pipeline
    config's), else 5.0. Boundary: the request's, else the server's, else the pipeline config's
    (``model_index.json``'s when ``None``). ``guidance_scale`` drives the high-noise expert,
    ``guidance_scale_2`` the low-noise one (SGLang's convention, and the reference pipeline's).
    """
    pipeline_config = server_args.pipeline_config
    flow_shift = batch.flow_shift
    if flow_shift is None:
        flow_shift = getattr(pipeline_config, "flow_shift", None)
    boundary_ratio = batch.boundary_ratio
    if boundary_ratio is None:
        boundary_ratio = getattr(server_args, "boundary_ratio", None)
    if boundary_ratio is None:
        boundary_ratio = getattr(pipeline_config, "boundary_ratio", None)
    n = int(batch.num_outputs_per_prompt or 1)
    params = {
        "negative_prompt": _negative_prompt_of(batch),
        "height": int(batch.height or 480),
        "width": int(batch.width or 832),
        "num_frames": int(batch.num_frames or 81),
        "num_steps": int(batch.num_inference_steps or 40),
        "guidance_scale": batch.guidance_scale,
        "guidance_scale_2": batch.guidance_scale_2,
        "flow_shift": float(5.0 if flow_shift is None else flow_shift),
        "boundary_ratio": None if boundary_ratio is None else float(boundary_ratio),
        "generators": generators_for(batch, n),
        "max_sequence_length": int(batch.max_sequence_length or 512),
    }
    if i2v:
        if params["height"] % 16 or params["width"] % 16:
            # I2V does not round the size.
            raise ValueError(
                f"Wan2.2 I2V needs height and width divisible by 16, got "
                f"{params['height']} x {params['width']}"
            )
        params["image"] = _image_of(batch)
    return params


def _record_request_metrics(request_metrics, metrics: dict) -> None:
    """Report the model's sub-stages in SGLang's per-request metrics (SGLang's stage names)."""
    if request_metrics is None:
        return
    request_metrics.record_stage("TextEncodingStage", metrics["text_encode_seconds"])
    if metrics["task"] == "i2v":
        request_metrics.record_stage("ImageVAEEncodingStage", metrics["vae_encode_seconds"])
    request_metrics.record_stage("DenoisingStage", metrics["denoise_seconds"])
    request_metrics.denoising_stages.add("DenoisingStage")
    for seconds in metrics["denoise_step_seconds"]:
        request_metrics.record_step(seconds)
    request_metrics.record_stage("DecodingStage", metrics["vae_decode_seconds"])


class NeuronWanStage(PipelineStage):
    """Encode, (I2V) condition, denoise and decode one request with the Neuron Wan model."""

    def __init__(self, model, *, i2v: bool):
        super().__init__()
        self.model = model
        self.i2v = i2v
        self._warned_sizes: set[tuple[int, int]] = set()

    def _check_vae_patch_parallel(self, height: int, width: int) -> None:
        """WARN once per size when the load-time VAE degree is not the reference one for it."""
        loaded = getattr(self.model, "vae_patch_parallel_size", None)
        if loaded is None or getattr(self.model, "vae_patch_parallel_size_fixed", False):
            return
        wanted = reference_vae_patch_parallel_size(self.i2v, height, width)
        if wanted != loaded and (height, width) not in self._warned_sizes:
            self._warned_sizes.add((height, width))
            logger.warning(
                "%dx%d runs with the VAE patch parallel degree %d chosen at load; the reference "
                "pipeline uses %d for this size. Set SGLANG_NEURON_WAN_VAE_PATCH_PARALLEL_SIZE=%d "
                "(or --warmup-resolutions %dx%d) at launch.",
                height,
                width,
                loaded,
                wanted,
                wanted,
                width,
                height,
            )

    def forward(self, batch: Req, server_args) -> OutputBatch:
        from sglang_neuron.diffusion.models.wan2_2.neuron_wan import describe

        params = request_parameters(batch, server_args, i2v=self.i2v)
        self._check_vae_patch_parallel(params["height"], params["width"])
        videos, metrics = self.model.generate(_prompt_of(batch), **params)
        if self.model.is_output_rank:
            _record_request_metrics(batch.metrics, metrics)
            logger.info(describe(metrics))
            save_golden_frames(videos)
        output = None if videos is None else to_output_video(videos)
        return OutputBatch(output=output, metrics=batch.metrics)


_request_index = 0  # requests this process has finished on the output rank


def save_golden_frames(
    videos: torch.Tensor | None, path: str | None = None, indices=None
) -> str | None:
    """Save the first video's frames for the accuracy gate, when asked; returns the path.

    ``SGLANG_NEURON_WAN_SAVE_VIDEO_PT`` (or ``path``) names a ``.pt`` file: ``torch.save`` of the
    ``[T, 3, H, W]`` frames in [0, 1] in the goldens' layout (``to_golden_frames``), before
    any uint8 / mp4 conversion, in the video's own dtype (bf16 from the VAE, which the goldens
    are exactly; ``.float()`` on load gives the float32 golden layout). Each saved request
    overwrites it; a ``{index}`` in the name (``str.format``) keeps one file per request.

    ``SGLANG_NEURON_WAN_SAVE_VIDEO_PT_INDICES`` (or ``indices``) limits the saves to those 0-based
    request indices of this process (unset: every request), so timed runs need not pay for the
    write (about 0.2 / 0.45 / 1 GB at 480p / 720p / 1080p).
    """
    global _request_index
    if videos is None:
        return None
    index = _request_index
    _request_index += 1
    from sglang_neuron.diffusion import envs

    if path is None:
        path = envs.SGLANG_NEURON_WAN_SAVE_VIDEO_PT
    if not path:
        return None
    if indices is None:
        indices = envs.SGLANG_NEURON_WAN_SAVE_VIDEO_PT_INDICES
    if indices is not None and index not in indices:
        return None
    path = path.format(index=index)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(to_golden_frames(videos, dtype=None), path)
    logger.info("[NeuronWan] saved the decoded frames of request %d to %s", index, path)
    return path


class _NeuronWanPipelineBase(ComposedPipelineBase):
    """Shared load and stage set-up of the Neuron Wan2.2 A14B pipelines."""

    # The Neuron model loads its own components from the checkpoint directory.
    _required_config_modules: list[str] = []
    i2v = False
    hf_model_paths: tuple[str, ...] = ()

    def __new__(cls, model_path: str, server_args, *args, **kwargs):
        if not is_wan22_a14b(model_path, cls.pipeline_name):
            raise ValueError(
                f"{model_path} is not a Wan2.2 A14B {cls.pipeline_name} checkpoint with BF16 "
                "weights. On Neuron, Wan runs Wan2.2 A14B BF16 only; Wan2.1, Wan2.2 TI2V-5B and "
                "quantized exports are not supported"
            )
        return super().__new__(cls)

    def load_modules(self, server_args, loaded_modules=None) -> dict:
        import torch.distributed as dist

        from sglang_neuron.diffusion import envs, runtime
        from sglang_neuron.diffusion.models.wan2_2.neuron_wan import NeuronWanModel

        world = dist.get_world_size()
        tp, cp, cfg = parallel_sizes(server_args, world)
        # The process group has set up the device runtime; make this rank's device current.
        runtime.set_worker_device()
        runtime.initialize_model_groups(
            tensor_parallel_size=tp, context_parallel_size=cp, cfg_parallel_size=cfg
        )
        model_path = runtime.local_model_path(self.model_path)
        if not isinstance(server_args.pipeline_config, self.pipeline_config_cls):
            # A checkpoint SGLang did not resolve to the Neuron config (another directory
            # name and a launcher without the plugin's resolution): the Wan2.2 A14B defaults.
            previous = type(server_args.pipeline_config).__name__
            server_args.pipeline_config = rebuild_pipeline_config(
                server_args.pipeline_config, self.pipeline_config_cls, server_args
            )
            logger.info(
                "%s: pipeline config %s rebuilt as %s (flow_shift=%s, boundary_ratio=%s)",
                self.model_path,
                previous,
                self.pipeline_config_cls.__name__,
                server_args.pipeline_config.flow_shift,
                server_args.pipeline_config.boundary_ratio,
            )
        vae_patch_parallel_size, source = self.vae_patch_parallel_size(server_args, world)
        boundary_ratio = getattr(server_args, "boundary_ratio", None)
        if boundary_ratio is None:
            boundary_ratio = getattr(server_args.pipeline_config, "boundary_ratio", None)
        t0 = time.perf_counter()
        # Build and load under the model dtype as torch's default dtype: the Neuron modules rely
        # on it for the parameters they create.
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            model = NeuronWanModel(
                model_path,
                dtype=torch.bfloat16,
                vae_patch_parallel_size=vae_patch_parallel_size,
                boundary_ratio=boundary_ratio,
            )
            model.load_weights()
        finally:
            torch.set_default_dtype(previous_dtype)
        model.vae_patch_parallel_size = vae_patch_parallel_size
        model.vae_patch_parallel_size_fixed = (
            envs.SGLANG_NEURON_WAN_VAE_PATCH_PARALLEL_SIZE is not None
        )
        logger.info("Neuron Wan2.2 VAE patch parallel %d (%s)", vae_patch_parallel_size, source)
        model.to(runtime.neuron_device())
        # Lazy: each graph compiles to a NEFF on its first call.
        model.compile(**envs.compile_kwargs())
        logger.info(
            "Neuron Wan2.2 A14B %s loaded in %.1fs (tp=%d cp=%d cfg=%d vae_pp=%d world=%d)",
            "I2V" if self.i2v else "T2V",
            time.perf_counter() - t0,
            tp,
            cp,
            cfg,
            vae_patch_parallel_size,
            world,
        )
        runtime.freeze_gc_heap()
        return {"neuron_wan": model}

    @classmethod
    def vae_patch_parallel_size(cls, server_args, world: int) -> tuple[int, str]:
        """``(degree, source)``: ``SGLANG_NEURON_WAN_VAE_PATCH_PARALLEL_SIZE``, else the reference
        degree for this task at the load-time resolution; clamped to the world."""
        from sglang_neuron.diffusion import envs

        requested = envs.SGLANG_NEURON_WAN_VAE_PATCH_PARALLEL_SIZE
        if requested is not None:
            return min(requested, world), "SGLANG_NEURON_WAN_VAE_PATCH_PARALLEL_SIZE"
        height, width, source = _load_resolution(server_args, cls.sampling_params_cls)
        degree = reference_vae_patch_parallel_size(cls.i2v, height, width)
        return min(degree, world), f"the reference degree for {height}x{width}, {source}"

    def initialize_pipeline(self, server_args) -> None:
        return None

    def create_pipeline_stages(self, server_args) -> None:
        self.add_stage(NeuronWanStage(self.modules["neuron_wan"], i2v=self.i2v), "neuron_wan_stage")


class NeuronWanPipeline(_NeuronWanPipelineBase):
    """Wan2.2 T2V A14B (``WanPipeline`` checkpoints with two experts) on Neuron."""

    pipeline_name = "WanPipeline"
    pipeline_config_cls = NeuronWan22T2VA14BConfig
    sampling_params_cls = NeuronWan22T2VA14BSamplingParam
    hf_model_paths = ("Wan-AI/Wan2.2-T2V-A14B-Diffusers",)


class NeuronWanI2VPipeline(_NeuronWanPipelineBase):
    """Wan2.2 I2V A14B (``WanImageToVideoPipeline`` checkpoints with two experts) on Neuron."""

    pipeline_name = "WanImageToVideoPipeline"
    pipeline_config_cls = NeuronWan22I2VA14BConfig
    sampling_params_cls = NeuronWan22I2VA14BSamplingParam
    hf_model_paths = ("Wan-AI/Wan2.2-I2V-A14B-Diffusers",)
    i2v = True
