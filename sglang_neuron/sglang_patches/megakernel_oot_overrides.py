# SPDX-License-Identifier: Apache-2.0
"""A GPT-OSS server that an out-of-tree platform's megakernel spec serves gets no CUDA-side GPT-OSS backend overrides.

Upstream's GPT-OSS override provider (``model_overrides.gpt_oss._gpt_oss_overrides``) picks an attention backend
(``triton`` on a platform it does not know) and an MoE runner backend for SGLang's own GPT-OSS kernels, and the model
dtype (bfloat16 for an MXFP4 checkpoint). A whole-model megakernel spec owns the backend choices itself: when the
platform is out of tree, ``--enable-nvidia-megakernel`` is set and a spec is registered for ``GptOssForCausalLM``, the
provider declares only its dtype (``KEPT``). The dtype stays because it reaches the model: SGLang's loader constructs
the model under ``set_default_torch_dtype(model_config.dtype)``, so any tensor the model builds without an explicit
dtype takes it (the YaRN ``concentration`` of ``GptOssRotaryEmbedding``, a giga-kernel decode graph input), and without
the override SGLang resolves float16 for the GPT-OSS checkpoint, whose config names no dtype. Every Neuron GPT-OSS
server therefore resolves the same bfloat16, megakernel or not, and the spec requires it. Registration metadata only is
read here; the spec's full check (target, features, dtype, configuration) still runs at ``handle_nvidia_megakernel``.
Every other server gets upstream's declarations unchanged.

The provider is stored in ``model_override_base._MODEL_OVERRIDE_FNS`` when its module is imported, and
``overrides.collect_model_override_declarations`` calls it from that list, so the module replaces the list entry
(patching the module attribute would reach nothing). The wrapper keeps the provider's name, which the declarations are
recorded under.

The rest of upstream's GPT-OSS handling, the attention-backend assertion of
``model_hook.handle_model_specific_adjustments``, is gpt_oss_backend_check's: it accepts the platform's ``neuron``
backend, which a Neuron megakernel server also resolves to, so no second change is needed there.

Targets are md5-pinned against upstream SGLang 43881a7.

Upstream: belongs with the platform megakernel seams (sgl-project/sglang#46355): a registered out-of-tree spec owns its
model's backend requirements."""

from __future__ import annotations

import functools

ARCHITECTURE = "GptOssForCausalLM"
_PROVIDER_MODULE = "sglang.srt.arg_groups.model_overrides.gpt_oss"
_BASE = "sglang.srt.arg_groups.model_override_base"
# (module, qualified name): md5 of the source
_TARGETS_MD5 = {
    (_PROVIDER_MODULE, "_gpt_oss_overrides"): "11bd42f666ae4691eb939ad12e95f0e7",
    # the providers' registry, appended to at import
    (_BASE, "register_model_override"): "bf7aa0e50717341ab9366df48f389e88",
    # calls each registered provider from the registry, recording it under its __qualname__
    ("sglang.srt.arg_groups.overrides", "collect_model_override_declarations"): "4cb4b3df30443e611497a5796d513dbf",
}
_MARK = "_sglang_neuron_oot_spec_skip"
# upstream's declarations a server the out-of-tree spec serves keeps: the model dtype (module docstring)
KEPT = ("dtype",)


def uses_oot_model_spec(server_args, architecture: str) -> bool:
    """Whether a registered out-of-tree megakernel spec serves ``architecture``: an out-of-tree platform, the
    megakernel flag set, and a spec registered for it (registration metadata only)."""
    from sglang.kernels.plugin_kernels.nvidia import spec as registry

    if not _platform().is_out_of_tree() or not bool(getattr(server_args, "enable_nvidia_megakernel", False)):
        return False
    return registry.spec_registered([architecture])


def _platform():
    """SGLang's current platform (the tests' seam)."""
    from sglang.srt.platforms import current_platform

    return current_platform


def skipping(provider):
    """``provider``, declaring only its ``KEPT`` entries for a server a registered out-of-tree spec serves."""

    @functools.wraps(provider)
    def wrapper(server_args, hf_config):
        declared = provider(server_args, hf_config)
        if uses_oot_model_spec(server_args, ARCHITECTURE):
            return {key: value for key, value in declared.items() if key in KEPT}
        return declared

    setattr(wrapper, _MARK, True)
    return wrapper


def register() -> None:
    """Replace upstream's GPT-OSS provider in the override registry with skipping(provider); idempotent."""
    import importlib

    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
    provider = importlib.import_module(_PROVIDER_MODULE)._gpt_oss_overrides
    providers = importlib.import_module(_BASE)._MODEL_OVERRIDE_FNS.get(ARCHITECTURE, [])
    if any(getattr(fn, _MARK, False) for fn in providers):
        return
    at = [i for i, fn in enumerate(providers) if fn is provider]
    if len(at) != 1:
        raise RuntimeError(f"{__name__}: SGLang's override registry holds upstream's {ARCHITECTURE} provider "
                           f"{len(at)} times, not once; re-validate {__name__} against this SGLang")
    providers[at[0]] = skipping(provider)
