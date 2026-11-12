# SPDX-License-Identifier: Apache-2.0
"""A megakernel spec's request checks reject a request it cannot serve in the tokenizer, before the request reaches any
scheduler.

A spec may offer ``request_disable_reasons(tokenized_request) -> list[str]`` (a host-only check of a normalized
TokenizedGenerateReqInput / TokenizedEmbeddingReqInput; ``PlatformMegakernelSpec`` of the megakernel_platform seams
declares the field, a plain MegakernelSpec has none). When ``--enable-nvidia-megakernel`` is set and the MSA plugin
does not serve the model, ``TokenizerManager.init_model_config`` resolves the spec once, on this platform's megakernel
target, and keeps a validator; ``TokenizerManager._create_tokenized_object`` runs it on each tokenized request and
raises ``ValueError`` with every reason, which the tokenizer returns as the request's error (HTTP 400). The spec is never
looked up per request.

Refuse, never corrupt: with the flag set, a frontend that finds no spec refuses to start (argument resolution's
``handle_nvidia_megakernel`` has already accepted one, so the frontend's plugin failed to load). The frontend check is a
rejection of requests, never the only guard: a spec's workers still check what they run.

An upstream SGLang change could record the resolved spec in new ExecKernel fields at argument resolution; those
msgspec fields cannot be added at run time, so the frontend resolves the spec itself. Its target answer is a
static platform name (no device probe), so the frontend and the workers resolve the same spec. Targets are md5-pinned
against upstream SGLang 43881a7.

Upstream: belongs with the platform megakernel seams (sgl-project/sglang#46355)."""

from __future__ import annotations

import importlib

_TOKENIZER = "sglang.srt.managers.tokenizer_manager"
_BINDING = "sglang.srt.kernel_plugins.nvidia.binding"
# (module, qualified name): md5 of the source
_TARGETS_MD5 = {
    # sets self.model_config, which the validator resolves from
    (_TOKENIZER, "TokenizerManager.init_model_config"): "5bfa039122d7009236d7d9866c12d175",
    # returns the normalized tokenized request the scheduler receives
    (_TOKENIZER, "TokenizerManager._create_tokenized_object"): "c628809d74a6d31661bcbccf147f7982",
    # the flag as the served process reads it
    (_BINDING, "megakernel_enabled"): "517b7a1093a7f3f70029c112ca777fb1",
    ("sglang.srt.kernel_plugins.amd.binding", "msa_serves"): "3db13e66e9ea56de5c860997920fd09f",
}
VALIDATOR = "_megakernel_request_validator"


def megakernel_flag(cfg) -> bool:
    """Whether a resolved view or server arguments set ``--enable-nvidia-megakernel``, the switch of the megakernel
    path on every platform."""
    return bool(getattr(cfg, "enable_nvidia_megakernel", False))


def find_platform_spec(architectures):
    """(spec, reason) for ``architectures`` on this platform's megakernel target, as the binding looks it up
    (``binding.device_isa``, which the megakernel_platform seams answer from the platform; ``spec.find_spec``)."""
    from sglang.kernels.plugin_kernels.nvidia import spec as registry

    binding = importlib.import_module(_BINDING)
    return registry.find_spec(architectures=architectures, isa=binding.device_isa())


def create_request_validator(server_args, model_config):
    """The validator of the megakernel spec this server selects (None when the flag is off, the MSA plugin serves the
    model, or the spec has no request checks); raises when the flag is set and no spec is found."""
    from sglang.srt.kernel_plugins.amd.binding import msa_serves

    binding = importlib.import_module(_BINDING)
    if not (megakernel_flag(server_args) or binding.megakernel_enabled()):
        return None
    architectures = model_config.hf_config.architectures
    if msa_serves(architectures):
        return None
    found, why = find_platform_spec(architectures)
    if found is None:
        raise ValueError(f"--enable-nvidia-megakernel: the tokenizer finds no megakernel spec: {why}")
    check = getattr(found, "request_disable_reasons", None)
    if check is None:
        return None
    name = found.name

    def validate(request) -> None:
        reasons = check(request)
        if reasons:
            raise ValueError(f"{name} cannot serve this request:\n  - " + "\n  - ".join(reasons))

    return validate


def init_model_config(original_fn, self, *args, **kwargs):
    """AROUND TokenizerManager.init_model_config: upstream's, then the request validator of the selected spec."""
    result = original_fn(self, *args, **kwargs)
    setattr(self, VALIDATOR, create_request_validator(self.server_args, self.model_config))
    return result


def create_tokenized_object(original_fn, self, *args, **kwargs):
    """AROUND TokenizerManager._create_tokenized_object: the tokenized request, checked by the selected spec."""
    tokenized = original_fn(self, *args, **kwargs)
    validate = getattr(self, VALIDATOR, None)
    if validate is not None:
        validate(tokenized)
    return tokenized


def register(registry, around) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
    registry.register(f"{_TOKENIZER}.TokenizerManager.init_model_config", init_model_config, around)
    registry.register(f"{_TOKENIZER}.TokenizerManager._create_tokenized_object", create_tokenized_object, around)
