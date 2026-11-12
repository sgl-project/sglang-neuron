# SPDX-License-Identifier: Apache-2.0
"""The SGLang Diffusion pipeline that serves LingBot-World-V2 real-time generation on Neuron.

SGLang's ``LingBotWorldCausalDMDPipeline`` runs input validation, text and image encoding,
latent preparation, the causal DMD denoise and the causal VAE decode as stages over SGLang
modules, one real-time chunk per request. The Neuron model keeps the DiT's per-step work in
static graphs over its paged K/V history and runs the encoders and the VAE on their own cores,
so this pipeline has one stage that runs a chunk with ``NeuronLingBotWorldModel.generate_chunk``
on every rank in lockstep (as the FLUX.1-dev and Wan2.2 pipelines run ``generate``).

Launch: ``--num-gpus 64 --tp-size 8`` on trn2.48xlarge at LNC 1 (the DiT's TP8 x CP8 on cores
0-63; CP = world / TP); the role workers take cores 64-79. Sessions are served through SGLang's
``/v1/realtime_video/generate`` WebSocket (:class:`NeuronLingBotWorldRealtimeAdapter`).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

from sglang.multimodal_gen.configs.pipeline_configs.lingbot_world import (
    LingBotWorldV2CausalDMDConfig,
)
from sglang.multimodal_gen.configs.sample.lingbot_world import LingBotWorldSamplingParams
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.realtime.session import BaseRealtimeState
from sglang.multimodal_gen.runtime.realtime.video import RAW_RGB_CONTENT_TYPE

from sglang_neuron.diffusion import envs
from sglang_neuron.diffusion.models.lingbot_world.payload import CAMERA_TRAJECTORY, TOTAL_CHUNKS

logger = logging.getLogger(__name__)

LINGBOT_WORLD_V2_HUB_ID = "robbyant/lingbot-world-v2-14b-causal-fast-diffusers"
#: The validated layout: TP8 x CP8 over 64 DiT ranks.
LINGBOT_TP_SIZE = 8
LINGBOT_CP_SIZE = 8
#: The one served resolution, as SGLang's realtime ``size`` (``WxH``).
LINGBOT_SIZE = "832x480"


@dataclass
class NeuronLingBotWorldV2Config(LingBotWorldV2CausalDMDConfig):
    """SGLang's LingBot-World-V2 config (flow shift 5, DMD steps 1000/750/500/250) on Neuron."""


class _NeuronLingBotSession(BaseRealtimeState):
    """Ties the Neuron model's session state to SGLang's worker-side real-time session."""

    def __init__(self) -> None:
        self.model = None
        self.session_id: str | None = None

    def dispose(self) -> None:
        if self.model is not None and self.session_id is not None:
            self.model.release_session(self.session_id)


def raw_rgb_output(frames, batch: Req) -> OutputBatch:
    """An ``OutputBatch`` of SGLang's raw RGB transport (``realtime/video.py``) from uint8
    ``(F, H, W, 3)`` frame arrays.

    No frames (a non-output DiT rank) is an empty batch list, not ``None``: SGLang fails a request
    whose output, files and frame batches are all ``None``, and sends no frames for an empty list.
    """
    output = OutputBatch(output=None, metrics=batch.metrics)
    output.raw_frame_content_type = RAW_RGB_CONTENT_TYPE
    if not frames:
        output.raw_frame_batches = []
        return output
    raw = [frame.tobytes() for block in frames for frame in block]
    height, width, channels = (int(d) for d in frames[0].shape[1:])
    output.raw_frame_batches = [raw]
    output.raw_frame_metadata = {
        "format": "rgb24",
        "width": width,
        "height": height,
        "channels": channels,
        "bytes_per_frame": width * height * channels,
    }
    return output


class NeuronLingBotStage(PipelineStage):
    """One real-time chunk with the Neuron LingBot-World model."""

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch: Req, server_args) -> OutputBatch:
        if batch.session is not None:
            state = batch.session.get_or_create_state(_NeuronLingBotSession)
            state.model = self.model
            state.session_id = str(batch.realtime_session_id)
        frames, timing = self.model.generate_chunk(batch)
        if self.model.is_output_rank:
            logger.debug("LingBot chunk %s", timing)
            return raw_rgb_output(frames, batch)
        return OutputBatch(output=None, metrics=batch.metrics)


class NeuronLingBotWorldPipeline(ComposedPipelineBase):
    """LingBot-World-V2 causal-fast (``LingBotWorldCausalDMDPipeline`` checkpoints) on Neuron."""

    pipeline_name = "LingBotWorldCausalDMDPipeline"
    pipeline_config_cls = NeuronLingBotWorldV2Config
    sampling_params_cls = LingBotWorldSamplingParams
    hf_model_paths = (LINGBOT_WORLD_V2_HUB_ID,)
    # The Neuron model loads its own components from the checkpoint directory.
    _required_config_modules: list[str] = []

    def __new__(cls, model_path: str, server_args, *args, **kwargs):
        if not is_lingbot_world_v2(model_path):
            raise NotImplementedError(
                f"{model_path} is not a LingBot-World-V2 causal checkpoint: the Neuron platform "
                "serves LingBot-World-V2 only"
            )
        return super().__new__(cls)

    def load_modules(self, server_args, loaded_modules=None) -> dict:
        import torch
        import torch.distributed as dist

        from sglang_neuron.diffusion import runtime
        from sglang_neuron.diffusion.models.lingbot_world import dist as lingbot_dist
        from sglang_neuron.diffusion.models.lingbot_world.neuron_lingbot import (
            NeuronLingBotWorldModel,
        )

        world = dist.get_world_size()
        tp = int(server_args.tp_size or 1)
        if (tp, world) != (LINGBOT_TP_SIZE, LINGBOT_TP_SIZE * LINGBOT_CP_SIZE):
            raise NotImplementedError(
                f"LingBot-World on Neuron runs TP{LINGBOT_TP_SIZE} x CP{LINGBOT_CP_SIZE} "
                f"(--num-gpus {LINGBOT_TP_SIZE * LINGBOT_CP_SIZE} --tp-size {LINGBOT_TP_SIZE}); "
                f"got tp {tp} over {world} ranks"
            )
        runtime.set_worker_device()
        runtime.initialize_model_groups(
            tensor_parallel_size=tp, context_parallel_size=world // tp, cfg_parallel_size=1
        )
        # A fatal error on rank 0 (a role worker's included) stops the server: SGLang watches its
        # workers only while they start, so a dead rank 0 would leave its clients waiting.
        lingbot_dist.on_fatal(_stop_server)
        model_path = runtime.local_model_path(self.model_path)
        t0 = time.perf_counter()
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            model = NeuronLingBotWorldModel(
                model_path, timing_file=envs.SGLANG_NEURON_LINGBOT_TIMING_FILE
            )
            model.load_weights()
        finally:
            torch.set_default_dtype(previous_dtype)
        model.to(runtime.neuron_device())
        # The role workers load and compile beside the DiT's warm-up.
        model.start_role_workers()
        model.compile()
        logger.info(
            "Neuron LingBot-World-V2 loaded in %.1fs (tp=%d cp=%d world=%d)",
            time.perf_counter() - t0,
            tp,
            world // tp,
            world,
        )
        runtime.freeze_gc_heap()
        return {"neuron_lingbot": model}

    def initialize_pipeline(self, server_args) -> None:
        return None

    def create_pipeline_stages(self, server_args) -> None:
        self.add_stage(NeuronLingBotStage(self.modules["neuron_lingbot"]), "neuron_lingbot_stage")


def is_lingbot_world_v2(model_path: str) -> bool:
    """The V2 causal-fast checkpoint: its Hub ID, or a directory of that basename."""
    name = os.path.basename(os.path.normpath(str(model_path)))
    return str(model_path) == LINGBOT_WORLD_V2_HUB_ID or name == LINGBOT_WORLD_V2_HUB_ID.split("/")[1]


def _stop_server() -> None:
    """SIGTERM to this worker's parent, the SGLang launcher, which then stops the server."""
    import signal

    os.kill(os.getppid(), signal.SIGTERM)


def realtime_adapter_cls():
    """The Neuron real-time adapter class (imports SGLang's LingBot adapter lazily)."""
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters.lingbot_world_realtime_adapter import (
        LingBotWorldRealtimeAdapter,
    )

    class NeuronLingBotWorldRealtimeAdapter(LingBotWorldRealtimeAdapter):
        """SGLang's LingBot adapter, plus what the Neuron session needs on its first chunk:

        * ``neuron_total_chunks``: the session's ``max_chunks`` (the condition encoder's bound
          and the last chunk's decode flush);
        * ``camera_trajectory``: a recorded camera (``{"poses": [N][4][4], "intrinsics":
          [N][4], "dtype": "float32"}``) from the init request's ``condition_inputs``, which
          replaces ``camera_actions`` for the session.

        An init request that asks for what this model does not do (another ``size``, a
        ``realtime_causal_*`` K/V window, frame interpolation or upscaling) is refused.
        """

        async def on_init(self, session, request):
            refused = []
            if request.size not in (None, LINGBOT_SIZE):
                refused.append(f"size {request.size!r} (served: {LINGBOT_SIZE})")
            for name in ("realtime_causal_sink_size", "realtime_causal_kv_cache_num_frames"):
                if getattr(request, name, None) is not None:
                    refused.append(f"{name} (the K/V window is the checkpoint's)")
            for name in ("enable_frame_interpolation", "enable_upscaling"):
                if getattr(request, name, None):
                    refused.append(name)
            if refused:
                raise ValueError(
                    "LingBot-World on Neuron does not support " + ", ".join(refused)
                )
            await super().on_init(session, request)

        def sample_chunk_inputs(self, session, server_args, chunk, chunk_size):
            inputs = super().sample_chunk_inputs(session, server_args, chunk, chunk_size)
            if chunk.index == 0 and session.request is not None:
                request = session.request
                if request.max_chunks is not None:
                    inputs.condition_inputs[TOTAL_CHUNKS] = int(request.max_chunks)
                trajectory = (request.condition_inputs or {}).get(CAMERA_TRAJECTORY)
                if trajectory is not None:
                    inputs.condition_inputs[CAMERA_TRAJECTORY] = trajectory
            return inputs

    return NeuronLingBotWorldRealtimeAdapter
