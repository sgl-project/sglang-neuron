# SPDX-License-Identifier: Apache-2.0
"""``sglang.multimodal_gen.plugins`` entry point: register the Neuron diffusion pipelines and SGLang patches.

SGLang loads this only when the Neuron diffusion platform is the selected platform
(plugins from other platforms' packages are skipped), so the Neuron pipelines replace
SGLang's own only on Neuron. The Wan ones take every ``WanPipeline`` / ``WanImageToVideoPipeline``
checkpoint and refuse any but Wan2.2 A14B BF16 (``models/wan2_2/wan_pipeline.py``); LingBot-World-V2
(``LingBotWorldCausalDMDPipeline``) comes with its real-time adapter
(``models/lingbot_world/lingbot_pipeline.py``). It runs in every process that loads SGLang
Diffusion's plugins, the workers included, before SGLang applies the registered hooks.
"""


def register() -> None:
    from sglang.multimodal_gen.registry import register_pipeline
    from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry
    from sglang.srt.plugins.hook_registry import HookType

    from sglang_neuron.diffusion.flux3_action.pipeline import NeuronFlux3ActionPipeline
    from sglang_neuron.diffusion.flux_pipeline import NeuronFluxPipeline
    from sglang_neuron.diffusion.minimax_h3_pipeline import NeuronMiniMaxH3Pipeline
    from sglang_neuron.diffusion.models.minimax_h3 import MINIMAX_H3_HF_PATHS
    from sglang_neuron.diffusion.models.wan2_2.wan_pipeline import (
        NeuronWanI2VPipeline,
        NeuronWanPipeline,
        install_model_info_resolution,
    )
    from sglang_neuron.sglang_patches import diffusion_mesh_rank_order

    for pipeline in (NeuronFluxPipeline, NeuronFlux3ActionPipeline):
        register_pipeline(
            pipeline,
            sampling_param_cls=pipeline.sampling_params_cls,
            pipeline_config_cls=pipeline.pipeline_config_cls,
            overwrite=True,
        )
    # The Hub IDs and the checkpoint's model_index class select the Neuron pipeline over
    # SGLang's own.
    h3_class = NeuronMiniMaxH3Pipeline.pipeline_name.lower()
    register_pipeline(
        NeuronMiniMaxH3Pipeline,
        sampling_param_cls=NeuronMiniMaxH3Pipeline.sampling_params_cls,
        pipeline_config_cls=NeuronMiniMaxH3Pipeline.pipeline_config_cls,
        hf_model_paths=list(MINIMAX_H3_HF_PATHS),
        model_detectors=[lambda probe: probe == h3_class],
        overwrite=True,
    )
    # SGLang's own TP / SP / CFG groups on the Neuron layout; the models use them (distributed.groups).
    diffusion_mesh_rank_order.register(HookRegistry, HookType.AROUND)
    # The Wan2.2 A14B Hub IDs (and local directories of that basename) resolve to the Neuron
    # configs, which carry the Neuron Wan2.2 defaults.
    for pipeline in (NeuronWanPipeline, NeuronWanI2VPipeline):
        register_pipeline(
            pipeline,
            sampling_param_cls=pipeline.sampling_params_cls,
            pipeline_config_cls=pipeline.pipeline_config_cls,
            hf_model_paths=list(pipeline.hf_model_paths),
            overwrite=True,
        )
    # ...and so does an A14B checkpoint in a directory of any other name.
    install_model_info_resolution()
    _register_lingbot_world(register_pipeline)


def _register_lingbot_world(register_pipeline) -> None:
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.registry import (
        register_realtime_model_adapter,
    )

    from sglang_neuron.diffusion.models.lingbot_world.lingbot_pipeline import (
        NeuronLingBotWorldPipeline,
        realtime_adapter_cls,
    )

    register_pipeline(
        NeuronLingBotWorldPipeline,
        sampling_param_cls=NeuronLingBotWorldPipeline.sampling_params_cls,
        pipeline_config_cls=NeuronLingBotWorldPipeline.pipeline_config_cls,
        hf_model_paths=list(NeuronLingBotWorldPipeline.hf_model_paths),
        overwrite=True,
    )
    register_realtime_model_adapter(
        NeuronLingBotWorldPipeline.pipeline_config_cls, realtime_adapter_cls()
    )
    # SGLang registers its built-in real-time adapters on first lookup, importing every one;
    # the Cosmos and SANA-WM adapters import CUDA-only kernels and fail on Neuron. Only LingBot
    # serves real time here, so its adapter is registered above and the built-ins are skipped.
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime import registry

    registry._BUILTIN_ADAPTERS_REGISTERED = True
