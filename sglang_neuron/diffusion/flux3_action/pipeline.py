# SPDX-License-Identifier: Apache-2.0
"""SGLang's FLUX.3-Action pipeline on Neuron: SGLang's stages and modules, with the Neuron DiT, VAE and sampler.

The preprocess, observation and denoising stages are SGLang's, subclassed so that no tensor op runs eagerly on the
device: three uint8 DROID views stay pixels (the VAE graph casts them and builds the canvas), the observation tokens
come from one VAE graph, the DiT's segments, blocks and output heads run as compiled graphs, each denoising step's CFG
combine and UniPC update is one device graph, and the actions are computed on the host. The text-encoding and
postprocess stages are SGLang's, with the text encoder compiled per pad bucket.
"""

from __future__ import annotations

import functools
import time

import numpy as np
import torch
from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    get_sp_group,
    get_tp_rank,
    get_tp_world_size,
)
from sglang.multimodal_gen.runtime.distributed.cfg_parallel_utils import run_cfg_parallel
from sglang.multimodal_gen.runtime.distributed.parallel_state import get_cfg_group
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines import flux3_action as upstream
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import flux3_action as stages
from sglang.multimodal_gen.runtime.pipelines_core.stages.vla import (
    VLAActionPostprocessStage,
    vla_state,
    vla_timings,
)

from sglang_neuron.diffusion import runtime
from sglang_neuron.diffusion.compile_backend import compile_graph
from sglang_neuron.diffusion.distributed.parallel_state import register_replica_groups
from sglang_neuron.diffusion.flux3_action import vae as neuron_vae
from sglang_neuron.diffusion.flux3_action.dit import NeuronFlux3Transformer
from sglang_neuron.diffusion.flux3_action.sampler import device_unipc
from sglang_neuron.diffusion.flux3_action.text_encoder import NeuronFlux3TextEncoder

WORKER_CPU_THREADS = 4  # torch / OpenMP threads per worker (runtime.configure_worker starts each at 1)


def droid_views(observation: dict, config) -> torch.Tensor | None:
    """The three uint8 360x640 DROID views of an observation, stacked ``[3, 360, 640, 3]`` (wrist, left, right), or
    None when the request is anything else (a composite, another layout or dtype), which SGLang composes."""
    if config.camera_layout != "droid":
        return None
    named = {k: v for k, v in observation.items() if k.startswith("observation.images.")}
    named.update(observation.get("images") or {})
    images = {stages._canonical_camera(name, config.camera_aliases): value for name, value in named.items()}
    cams = [np.asarray(images[key]) for key in config.image_keys if key in images]
    if "composite" in images or len(cams) != 3 or any(
            c.dtype != np.uint8 or c.shape != (*neuron_vae.DROID_CAMERA_HW, 3) for c in cams):
        return None
    return torch.from_numpy(np.stack(cams))


class NeuronFlux3ActionPreprocessStage(stages.Flux3ActionPreprocessStage):
    """SGLang's preprocess, with three uint8 DROID views kept as pixels in the observation's ``canvas``."""

    def forward(self, batch, server_args):
        start = time.perf_counter()
        state = vla_state(batch)
        if batch.is_warmup:
            state["flux3_observation"] = stages._warmup_observation(self.config)
        else:
            observation = dict(state.get("observation") or {})
            observation.setdefault("prompt", batch.prompt)
            views = droid_views(observation, self.config)
            if views is None:
                state["flux3_observation"] = stages.parse_observation(observation, self.config)
            else:
                prompt = observation.get("prompt") or observation.get("task") or ""
                if isinstance(prompt, (list, tuple)):
                    if len(prompt) != 1:
                        raise ValueError("FLUX 3 Action serves one observation per request")
                    prompt = prompt[0]
                state["flux3_observation"] = stages.Flux3ActionObservation(
                    canvas=views, state=stages._observation_state(observation, state_dim=self.config.state_dim),
                    prompt=str(prompt))
        vla_timings(batch)["preprocess_ms"] = (time.perf_counter() - start) * 1000
        return batch


class NeuronFlux3ActionTextEncodingStage(stages.Flux3ActionTextEncodingStage):
    """SGLang's text-encoding stage, the text ids passed from the host (the RoPE tables are cached by them)."""

    def _encode(self, captions, device):
        with self.use_declared_component(component_name="text_encoder", module=self.text_encoder):
            encoded = [self.text_encoder.encode([c])[0] for c in captions]
        with (self.use_declared_component(component_name="transformer", module=self.transformer),
              set_forward_context(current_timestep=0, attn_metadata=None)):
            return {caption: self.transformer.encode_context(ctx=ctx.to(device), ctx_ids=stages.text_ids(ctx.shape[1]))
                    for caption, ctx in zip(captions, encoded)}


class NeuronFlux3ActionObservationEncodingStage(stages.Flux3ActionObservationEncodingStage):
    """SGLang's observation stage: the frame goes to the device as it is (uint8 DROID pixels or a canvas) and its tokens
    come from the Neuron VAE graph (``vae.tokens``), which casts and builds the canvas; the stream ids stay on the host
    (the RoPE tables are cached by them) and the zero timestep stays on the device."""

    def forward(self, batch, server_args):
        start = time.perf_counter()
        cfg = self.config
        device = get_local_torch_device()
        state = vla_state(batch)
        observation = state["flux3_observation"]
        h, w = cfg.latent_hw
        frame_ids = stages.video_ids(frames=1, h=h, w=w, first_frame=0, fps=cfg.fps)
        action, action_ids = self._state_tokens(observation.state, device)
        if getattr(self, "_zero", None) is None:
            self._zero = torch.zeros(1).to(device)
        stream_inputs = (frame_ids, action, action_ids, self._zero)
        with self._vae_use():
            video = self.vae.neuron_tokens(observation.canvas.to(device)[None])
        with self._transformer_use(), set_forward_context(current_timestep=0, attn_metadata=None):
            tensors = self._encode_streams(video, *stream_inputs)
        state["flux3_conditioning"] = stages._states_from_tensors(["video_cond", f"{cfg.action_modality}_cond"],
                                                                  tensors)
        vla_timings(batch)["observation_ms"] = (time.perf_counter() - start) * 1000
        return batch


class NeuronFlux3ActionDenoisingStage(stages.Flux3ActionDenoisingStage):
    """SGLang's denoising stage: each step's CFG combine and UniPC update as one device graph
    (``sampler.device_unipc``), the target streams cast inside their graphs, and the actions computed on the host from
    the solved action stream. With CFG parallel over two ranks, each rank runs its own branch and the step graph
    gathers both (in place of SGLang's eager ``run_cfg_parallel`` gather)."""

    def _target_streams(self, samples, ropes, t, *, order):
        return [self.transformer.encode_stream(name=name, x=x, ids=None, timesteps=t, rope=rope)
                for name, x, rope in zip(order, samples, ropes, strict=True)]

    def forward(self, batch, server_args):
        cfg = self.config
        device = get_local_torch_device()
        state = vla_state(batch)
        observation = state["flux3_observation"]
        contexts = state["flux3_contexts"]
        conditioning = state["flux3_conditioning"]
        guidance = state["flux3_guidance"]
        steps = batch.num_inference_steps
        seed = batch.seed[0] if isinstance(batch.seed, list) else batch.seed
        horizon = batch.action_horizon or cfg.action_horizon
        if horizon > cfg.action_horizon:
            raise ValueError(f"action_horizon {horizon} exceeds the policy chunk {cfg.action_horizon}")

        start = time.perf_counter()
        samples, ids = self._noised_streams(seed)
        samples = {k: v.to(device) for k, v in samples.items()}
        order = list(samples)
        ropes = tuple(self.transformer.rope(ids[k]) for k in order)
        cfg_policy = stages._cfg_parallel_policy(contexts, server_args)
        state_names = [s.name for s in (*conditioning, *contexts)]
        states = tuple(t for s in (*conditioning, *contexts) for t in stages._state_tensors(s))
        denoise_step = functools.partial(self._denoise_step, order=order, state_names=state_names)
        cfg_gather = (cfg_policy is not None and len(cfg_policy.branches) == 2
                      and get_cfg_group().world_size == 2)
        step = 0

        def predict(current, timestep):
            nonlocal step
            current_streams = tuple(current[k] for k in order)
            with set_forward_context(current_timestep=step, attn_metadata=None, forward_batch=batch):
                if cfg_gather:
                    targets = self._target_streams(current_streams, ropes, timestep, order=order)
                    branch = cfg_policy.branches[get_cfg_group().rank_in_group]
                    raw = [self._denoise_pass(branch.kwargs["context"], targets, conditioning, order=order)]
                elif cfg_policy is not None:
                    targets = self._target_streams(current_streams, ropes, timestep, order=order)
                    raw = run_cfg_parallel(
                        cfg_policy,
                        lambda branch: self._denoise_pass(branch.kwargs["context"], targets, conditioning,
                                                          order=order))
                else:
                    flat = self._graph.run(denoise_step, (*current_streams, timestep, *ropes, *states),
                                           key=(tuple(order), tuple(state_names)))
                    raw = [flat[i:i + len(order)] for i in range(0, len(flat), len(order))]
            step += 1
            return [dict(zip(order, p)) for p in raw]

        with self.use_declared_component(component_name="transformer", module=self.transformer):
            result = device_unipc(samples, predict, scheduler=self.scheduler, n_steps=steps,
                                  shift=cfg.sampler_shift, guidance=guidance, device=device, cfg_gather=cfg_gather)
        targets = result[cfg.action_modality][0].cpu() / cfg.action_scale
        actions = stages.targets_to_actions(targets, state=observation.state, config=cfg)
        state["actions"] = actions[None, :horizon]
        vla_timings(batch)["denoise_ms"] = (time.perf_counter() - start) * 1000
        return batch


class NeuronFlux3ActionPipeline(upstream.Flux3ActionPipeline):
    pipeline_name = upstream.Flux3ActionPipeline.pipeline_name

    def create_pipeline_stages(self, server_args) -> None:
        """SGLang's stages, with the Neuron preprocess, text-encoding, observation and denoising stages, warmed up by
        ``warm_up`` before the server reports ready."""
        config = server_args.pipeline_config
        transformer = self.get_module("transformer")
        self.add_stage(NeuronFlux3ActionPreprocessStage(config), "flux3_action_preprocess")
        self.add_stage(NeuronFlux3ActionTextEncodingStage(config, transformer=transformer,
                                                          text_encoder=self.get_module("text_encoder")),
                       "flux3_action_text_encoding")
        self.add_stage(NeuronFlux3ActionObservationEncodingStage(config, transformer=transformer,
                                                                 vae=self.get_module("vae")),
                       "flux3_action_observation_encoding")
        self.add_stage(NeuronFlux3ActionDenoisingStage(config, transformer, scheduler=self.get_module("scheduler")),
                       "flux3_action_denoise")
        self.add_stage(VLAActionPostprocessStage(), "flux3_action_postprocess")
        self.warm_up(server_args)

    def warm_up(self, server_args) -> None:
        """Run one DROID request through the stages on every rank: three black uint8 views, a zero state, and the
        package's default steps and guidance, so that the first served request compiles no graph."""
        config = server_args.pipeline_config
        if config.camera_layout != "droid":
            return
        view = np.zeros((*neuron_vae.DROID_CAMERA_HW, 3), dtype=np.uint8)
        params = self.sampling_params_cls(prompt="", images={key: view for key in config.image_keys},
                                          state=np.zeros(config.state_dim, dtype=np.float32))
        params._adjust(server_args)
        self.forward(prepare_request(server_args, params), server_args)

    def load_modules(self, server_args, loaded_modules: dict[str, torch.nn.Module] | None = None):
        """SGLang's load, on the worker's device made current, with ``WORKER_CPU_THREADS`` host threads, and the models
        bound to SGLang's TP / SP / CFG groups on the plugin's layout, registered with the compiler as FLUX.1's are."""
        runtime.set_worker_device()
        runtime.limit_worker_threads(WORKER_CPU_THREADS)
        tp, sp = get_tp_world_size(), get_sp_group().world_size
        runtime.initialize_model_groups(tensor_parallel_size=tp, context_parallel_size=sp,
                                        cfg_parallel_size=get_cfg_group().world_size)
        register_replica_groups(tp_size=tp, cp_size=sp)
        return super().load_modules(server_args, loaded_modules)

    @staticmethod
    def _load_transformer(server_args, config, package):
        """SGLang's DiT on the Neuron block: SGLang reads and TP-partitions the checkpoint on the host, the blocks
        convert to the kernels' layout, and the block chains run compiled (``NeuronFlux3Transformer``). The Neuron
        block runs the bf16 package only; a quantized (FP8r) package raises ``NotImplementedError``."""
        if config.quantization is not None:
            raise NotImplementedError(
                f"FLUX.3-Action {config.quantization} packages are not supported on Neuron; use the bf16 package"
            )
        transformer = upstream.maybe_load_fsdp_model(
            model_cls=NeuronFlux3Transformer,
            init_params={"config": config.dit_config, "hf_config": {}},
            weight_dir_list=[str(package / "model.safetensors")],
            device=torch.device("cpu"),
            hsdp_replicate_dim=server_args.hsdp_replicate_dim,
            hsdp_shard_dim=server_args.hsdp_shard_dim,
            param_dtype=upstream.resolve_component_precision(server_args, "transformer"),
            reduce_dtype=torch.float32,
            component_starts_on_cpu=True,
            fsdp_inference=False,
            pin_cpu_memory=False,
            strict=True,
        )
        transformer.prepare()
        transformer = transformer.to(runtime.neuron_device())
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
        return transformer

    @staticmethod
    def _load_text_encoder(server_args, config):
        """SGLang's text encoder, its language model compiled per pad bucket (``NeuronFlux3TextEncoder``)."""
        spec = server_args.component_paths.get("text_encoder") or config.text_encoder_id
        encoder = NeuronFlux3TextEncoder(spec, output_layers=config.text_output_layers,
                                         pad_multiple=config.text_pad_multiple, max_length=config.text_max_length)
        return encoder.to(runtime.neuron_device())

    @staticmethod
    def _load_vae(server_args, config):
        """SGLang's VAE encoder, read on the host (safetensors cannot open a file on the neuron device), with the
        Neuron neighborhood attention, split over the TP group (and its first stages' rows over a CP pair), compiled
        with the DROID canvas into one graph; a local ``video_vae.safetensors`` comes as a component weights path."""
        spec = (server_args.component_weights_paths.get("vae") or server_args.component_paths.get("vae")
                or config.video_vae_id)
        with torch.device("meta"):
            vae = upstream.Flux3VideoVAE(config.vae_config)
        path = upstream._resolve_file(spec, upstream.VIDEO_VAE_FILENAME)
        vae.load_checkpoint(upstream.load_file(path, device="cpu"))
        neuron_vae.tensor_parallel(vae, get_tp_rank(), get_tp_world_size())
        sp = get_sp_group()
        split = neuron_vae.row_split(vae, sp.rank_in_group, config.canvas_hw[0]) if sp.world_size == 2 else None
        vae = vae.to(device=runtime.neuron_device(), dtype=upstream.resolve_component_precision(server_args, "vae"))
        neuron_vae.prepare(runtime.neuron_device())
        vae.neuron_tokens = compile_graph(
            functools.partial(neuron_vae.tokens, vae, canvas_hw=tuple(config.canvas_hw),
                              latent_hw=tuple(config.latent_hw), split=split), neuron_vae.COMPILER_ARGS)
        return vae
