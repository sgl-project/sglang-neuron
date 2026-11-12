# SPDX-License-Identifier: Apache-2.0
"""The launch rules of the Kimi-K3 PD roles (catalog.ModelSpec.launch_hooks), each rule its role's.

The platform runs ``apply_defaults``, ``kv_page_size`` and ``launch_problems`` in the launching process, before any
scheduler starts, and applies ``launch_settings`` there; ``prepare_process`` runs in the launching process and in every
scheduler process (the drafter checkpoint's config type, which SGLang reads through AutoConfig while it resolves the
speculative arguments and again for the draft model config); ``validate_request`` runs on every request at the
tokenizer of both PD servers, so a request the decode programs do not serve is refused before the prefill computes its
prompt.

- Prefill: one node, PP8 x TP16 at LNC1, DCP over the stage's TP group, the served layer split ``SERVED_PARTITION``,
  chunks of whole stripe cycles up to the MX linear kernel's row cap. The drafter checkpoint is the launch's
  ``--speculative-draft-model-path`` (as an upstream PD EAGLE prefill takes it); without it the prefill writes no
  drafter rows.
- Decode: the two-node layout the catalog entry declares (decode/capability.TWO_NODE_DP8), DSPARK with k = 3 or 7; the
  pools and slots follow the programs' page layout (``apply_defaults`` sets them where the launch leaves them unset).

Both roles take the patched NIXL (``tools/build_nixl.sh``), the ``topk_topp`` device sampler and the 512 MB
scratchpad page. ``chat_sampling_params`` gives each chat request the Kimi-K3 sampling contract (temperature 1.0 with
thinking, 0.6 without, top-p 0.95), as the Kimi-K3 API does, unless ``SGLANG_NEURON_KIMI_K3_SAMPLING=client``.
Imports at module level are stdlib only: the catalog names this module and the launching process
imports it before anything else is decided; the rest is imported where it is used."""

from __future__ import annotations

import os

ROLES = ("prefill", "decode")
STRIPE = 128
"""The KV page of both roles: one 128-token MLA stripe."""
PARTITION_ENV = "SGLANG_PP_LAYER_PARTITION"
SAMPLER = "topk_topp"
SCRATCHPAD_PAGE_MB = "512"
NEFF_CACHE_LOCK_TIMEOUT_S = "14400"
"""torch-neuronx's NEFF cache lock wait, which the stage graphs' and the decode programs' compiles outlast at its
default."""
DECODE_EFA_ENV = (("FI_PROVIDER", "efa"), ("FI_EFA_USE_DEVICE_RDMA", "1"), ("FI_EFA_FORK_SAFE", "1"),
                  ("FI_EFA_ENABLE_SHM_TRANSFER", "0"))
"""The libfabric settings the two-node decode runs with, unless the launch sets them."""
DECODE_HASH_SEED = "42"
DECODE_REFUSED_ENV = ("NEURON_RT_A2AV_ONE_RANK_PER_CHIP", "NEURON_RT_DBG_EARLY_POSTING", "NKI_SIMULATOR")
"""Runtime and kernel switches the decode programs are not built or validated with."""
SAMPLING_ENV = "SGLANG_NEURON_KIMI_K3_SAMPLING"
"""``contract`` (the default): each chat request samples with the Kimi-K3 values (``chat_sampling_params``);
``client``: each chat request samples with its own values and SGLang's defaults."""
SAMPLING_MODES = ("contract", "client")
CHAT_TOOL_CALL_GRAMMAR = False
"""Whether a chat request with tools takes the grammar of SGLang's tool-call parser (sglang_patches.model_chat_sampling).
The decode samples on the device and applies no grammar, and both servers refuse every request that asks for one
(``request_problems``). So a chat request with tools and ``tool_choice`` ``auto`` runs without the parser's grammar: the
model writes its calls in its own markup, and ``--tool-call-parser kimi_k3`` reads them into ``tool_calls``. A request
whose ``tool_choice`` forces a call (``required`` or a named tool) is refused, because no grammar can force it."""
THINKING_TEMPERATURE = 1.0
"""The Kimi-K3 temperature of a request that renders the think channel."""
INSTANT_TEMPERATURE = 0.6
"""The Kimi-K3 temperature of a request without the think channel."""
TOP_P = 0.95
"""The Kimi-K3 top-p of every chat request."""
EARLY_LAST_SEND_ENV = "SGLANG_NEURON_PP_PREFILL_EARLY_LAST_SEND"
PREFILL_ENV_DEFAULTS = ((EARLY_LAST_SEND_ENV, "1"),)
"""The environment the prefill defaults to: every PP stage sends a request's last chunk at its own forward
(sglang_patches/pp_prefill_last_send). A launch value is kept."""


def _role(server_args) -> str:
    return str(getattr(server_args, "disaggregation_mode", "null") or "null")


def _int(server_args, name: str, default: int = 1) -> int:
    value = getattr(server_args, name, None)
    return default if value is None else int(value)


def _dp_ranks(server_args) -> int:
    """The decode's DP replicas, one scheduler each: replicas times attention-DP groups (SGLang's ``num_dp_ranks``).
    SGLang resolves ``--dp-size 8 --enable-dp-attention`` to ``attn_dp_size`` 8 and ``dp_size`` 1 before the platform
    defaults run."""
    return _int(server_args, "dp_size") * _int(server_args, "attn_dp_size")


def _stated(server_args, name: str):
    """The value the launch itself gave ``name``, before SGLang's resolution derived one (its raw input, which SGLang
    keeps while it resolves); a value SGLang already derived for an unset field reads as unset. SGLang's PD hook runs
    before the platform defaults and derives ``--disaggregation-decode-extra-slots`` from a ``--max-running-requests``
    that is still unset then (0)."""
    raw = getattr(server_args, "_raw_input", None)
    if isinstance(raw, dict) and name in raw:
        return raw[name]
    return getattr(server_args, name, None)


def kv_page_size(server_args) -> int:
    """Both roles store the MLA rows in 128-token stripes."""
    return STRIPE


def _decode_shape(server_args):
    """(queries Q, sequences a replica, rings a replica, the pools' tokens a rank) of a decode launch, or None for a
    --speculative-num-draft-tokens the programs do not serve."""
    from sglang_neuron.models.kimi_k3.decode import pools, sgl_pools
    from sglang_neuron.models.kimi_k3.decode.host import FORM_BATCH, FORMS

    queries = _int(server_args, "speculative_num_draft_tokens", 0)
    if queries not in FORMS:
        return None
    return (queries, FORM_BATCH[FORMS[queries]], pools.rider_slots(queries),
            sgl_pools.DecodePageLayout(queries, 1).max_total_tokens)


def apply_defaults(server_args, environ=os.environ) -> None:
    """The launch values the decode programs' page layout fixes, where the launch leaves them unset: the pools' tokens
    a rank (--max-total-tokens), the running requests of the 8 replicas (--max-running-requests, the sequences of each
    replica's batch) and the request slots beside them (--disaggregation-decode-extra-slots: a replica holds as many
    requests as its rider pool has rings, running and in transfer); and the environment both roles default to and
    the prefill defaults to (``PREFILL_ENV_DEFAULTS``: the early last-chunk sends). ``launch_problems`` refuses another
    value of the three launch values; the environment keeps a value the launch sets."""
    environ.setdefault("TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT", NEFF_CACHE_LOCK_TIMEOUT_S)
    if _role(server_args) == "prefill":
        for name, value in PREFILL_ENV_DEFAULTS:
            environ.setdefault(name, value)
    if _role(server_args) != "decode":
        return
    # the schedulers both nodes start take the hash seed the two-node decode was validated with
    environ.setdefault("PYTHONHASHSEED", DECODE_HASH_SEED)
    for name, value in DECODE_EFA_ENV:
        environ.setdefault(name, value)
    shape = _decode_shape(server_args)
    if shape is None:
        return
    _, seqs, rings, tokens = shape
    dp = _dp_ranks(server_args)
    for name, value in (("max_total_tokens", tokens), ("max_running_requests", dp * seqs),
                        ("disaggregation_decode_extra_slots", rings - seqs)):
        if _stated(server_args, name) is None:
            setattr(server_args, name, value)


def launch_settings(server_args) -> tuple:
    """The settings both roles run with: the 512 MB scratchpad page the stage graphs and the decode programs compile
    for, and the ``topk_topp`` device sampler, the one mode that matches what the graphs sample with (each request's
    temperature, top-k, top-p and seed)."""
    from sglang_neuron.models import catalog

    return (
        catalog.LaunchSetting("NEURON_SCRATCHPAD_PAGE_SIZE", SCRATCHPAD_PAGE_MB,
                              "the Kimi-K3 stage graphs and decode programs compile for a 512 MB HBM scratchpad page, "
                              "which leaves room for a rank's weights and pools at LNC1"),
        catalog.LaunchSetting("SGLANG_NEURON_SAMPLER", SAMPLER,
                              "the Kimi-K3 prefill's last stage and the decode programs sample each request with its "
                              "temperature, top-k, top-p and seed on the device"),
    )


def rank_runtime_env(role: str) -> dict:
    """No settings over the role's defaults: a decode rank starts with its declared layout's (capability.RUNTIME_ENV,
    runtime_init.runtime_defaults)."""
    return {}


def prepare_process() -> None:
    """Register the drafter checkpoint's config type with AutoConfig (drafter_config) in this process."""
    from sglang_neuron.models.kimi_k3 import drafter_config

    drafter_config.register()


def nixl_problems() -> list:
    """The patched NIXL (tools/build_nixl.sh, patches/nixl) must be the one this launch imports: both roles move the KV
    over its LIBFABRIC plugin, and the prefill's PP seam its stage boundaries; a container's own NIXL wheel hangs or
    crashes there."""
    import importlib.util
    from pathlib import Path

    from sglang_neuron.distributed.pp_seam import NIXL_0009_MARKER

    try:
        import torch

        variant = f"nixl_cu{torch.version.cuda.split('.')[0]}"
        spec = importlib.util.find_spec(variant)
    except (ImportError, AttributeError, ValueError):
        spec = None
    if spec is None or spec.origin is None:
        return ["no NIXL on the path: put the NIXL overlay tools/build_nixl.sh builds ahead of SGLang on PYTHONPATH"]
    plugin = Path(spec.origin).resolve().parent.parent / f".{variant}.mesonpy.libs" / "plugins" / \
        "libplugin_LIBFABRIC.so"
    try:
        patched = NIXL_0009_MARKER in plugin.read_bytes()
    except OSError:
        patched = False
    if not patched:
        return [f"the NIXL at {Path(spec.origin).parent} is not the patched build (its LIBFABRIC plugin {plugin} lacks "
                "patches/nixl): put the NIXL overlay tools/build_nixl.sh builds ahead of SGLang on PYTHONPATH and "
                "leave NIXL_PLUGIN_DIR unset"]
    if os.environ.get("NIXL_PLUGIN_DIR"):
        return [f"NIXL_PLUGIN_DIR={os.environ['NIXL_PLUGIN_DIR']}: the overlay's own plugins must load; unset it"]
    return []


def _prefill_problems(server_args, environ) -> list:
    from sglang_neuron.models.kimi_k3.prefill.mx_linear import MX_LINEAR_MAX_ROWS
    from sglang_neuron.models.kimi_k3.prefill.stage_plan import SERVED_PARTITION

    problems = []
    pp = _int(server_args, "pp_size")
    if pp != len(SERVED_PARTITION):
        problems.append(f"--pp-size {pp}: the prefill serves {len(SERVED_PARTITION)} stages; launch with --pp-size "
                        f"{len(SERVED_PARTITION)}")
    want = ",".join(map(str, SERVED_PARTITION))
    partition = environ.get(PARTITION_ENV)
    if partition != want:
        problems.append(f"{PARTITION_ENV}={partition if partition is not None else '(unset)'}: the stages serve the "
                        f"split {want}, which ends every stage after an MLA layer and gives each its AttnRes table; "
                        f"launch with {PARTITION_ENV}={want}")
    chunk = _int(server_args, "chunked_prefill_size", 0)
    if chunk > MX_LINEAR_MAX_ROWS:
        problems.append(f"--chunked-prefill-size {chunk}: the MX linear kernel takes at most {MX_LINEAR_MAX_ROWS} rows "
                        "a call; launch with a chunk of at most that")
    draft = getattr(server_args, "speculative_draft_model_path", None)
    if draft is not None and not os.path.isdir(draft):
        problems.append(f"--speculative-draft-model-path {draft}: no such drafter checkpoint directory")
    if draft is not None and getattr(server_args, "max_total_tokens", None) is None:
        problems.append(f"--speculative-draft-model-path {draft} without --max-total-tokens: the drafter row pools "
                        "take HBM at the pool's tokens, which the KV budget leaves out only for a stated pool size; "
                        "launch with --max-total-tokens")
    if getattr(server_args, "speculative_algorithm", None) is not None:
        problems.append(f"--speculative-algorithm {server_args.speculative_algorithm} on the prefill: the prefill "
                        "writes the drafter's context rows itself (--speculative-draft-model-path) and runs no "
                        "speculative step")
    return problems


def _decode_problems(server_args, environ) -> list:
    problems = []
    shape = _decode_shape(server_args)
    if shape is None:
        problems.append(f"--speculative-num-draft-tokens {getattr(server_args, 'speculative_num_draft_tokens', None)}:"
                        " the decode programs verify 4 (k 3) or 8 (k 7) rows a request")
    else:
        _, seqs, rings, tokens = shape
        dp = _dp_ranks(server_args)
        for name, flag, want in (("max_total_tokens", "--max-total-tokens", tokens),
                                 ("max_running_requests", "--max-running-requests", dp * seqs),
                                 ("disaggregation_decode_extra_slots", "--disaggregation-decode-extra-slots",
                                  rings - seqs)):
            have = getattr(server_args, name, None)
            if have is not None and int(have) != want:
                problems.append(f"{flag} {have}: the decode programs' page layout serves {want}; launch without it")
    if not getattr(server_args, "speculative_draft_model_path", None):
        problems.append("no --speculative-draft-model-path: the draft program runs the drafter checkpoint")
    for name in DECODE_REFUSED_ENV:
        if environ.get(name) is not None:
            problems.append(f"{name}={environ[name]}: the decode programs are not built with it; unset it")
    return problems


def launch_problems(server_args, checkpoint_config: dict, environ=os.environ) -> list:
    """Every reason the launch is not served (the module docstring), naming the flag or variable to change."""
    role = _role(server_args)
    if role not in ROLES:
        return [f"--disaggregation-mode {role}: Kimi-K3 serves the PD prefill and the PD decode; launch one with "
                "--disaggregation-mode prefill or decode"]
    problems = _prefill_problems(server_args, environ) if role == "prefill" else _decode_problems(server_args, environ)
    mode = environ.get(SAMPLING_ENV, SAMPLING_MODES[0])
    if mode not in SAMPLING_MODES:
        problems.append(f"{SAMPLING_ENV}={mode}: the chat sampling is {' or '.join(SAMPLING_MODES)}; set one of them "
                        "or unset it (contract)")
    if getattr(server_args, "disaggregation_transfer_backend", None) == "nixl" or role == "prefill":
        problems += nixl_problems()
    return problems


UNSERVED_SAMPLING = (("frequency_penalty", 0.0), ("presence_penalty", 0.0), ("repetition_penalty", 1.0),
                     ("min_p", 0.0), ("min_new_tokens", 0))
"""The sampling options the graphs do not apply, with the value that leaves them off."""
STRUCTURED_OUTPUT = ("json_schema", "regex", "ebnf", "structural_tag")


def request_problems(obj, preferred=None) -> list:
    """What a request asks that the Kimi-K3 graphs do not serve: they sample with temperature, top-k, top-p and a
    seed only, and return no logprobs or hidden states."""
    from sglang_neuron import sampling

    get = sampling._request_params(obj, preferred)
    problems = [f"structured output ({name})" for name in STRUCTURED_OUTPUT if get(name) is not None]
    for name, off in UNSERVED_SAMPLING:
        value = get(name, off)
        if value != off:
            problems.append(f"{name}={value}")
    if get("logit_bias"):
        problems.append("logit_bias")
    if get("custom_params") or getattr(obj, "custom_logit_processor", None):
        problems.append("a custom logit processor")
    if getattr(obj, "return_logprob", False):
        problems.append("logprobs")
    if getattr(obj, "return_hidden_states", False):
        problems.append("hidden states")
    return problems


def validate_request(obj, preferred, server_args, vocab_size: int) -> None:
    """Refuse a request the decode programs do not serve (``request_problems``), at the tokenizer of either PD server,
    with HTTP 400."""
    problems = request_problems(obj, preferred)
    if problems:
        raise ValueError("Kimi-K3 on Neuron samples with temperature, top-p, top-k and a seed only, not "
                         + ", ".join(problems))


def chat_sampling_params(request, environ=os.environ) -> dict:
    """Return the sampling values that replace the values of a chat request: the Kimi-K3 contract of the Kimi-K3 API.

    The contract gives temperature 1.0 when the request renders the think channel, else 0.6, and top-p 0.95. The
    request renders the think channel unless its ``thinking`` template kwarg is False. With
    ``SGLANG_NEURON_KIMI_K3_SAMPLING=client`` the function returns no values. The request then keeps its own values
    and SGLang's defaults (temperature 1.0 and top-p 1.0 for a request that sets none).

    The values change the acceptance of the speculative decode. An AgentX request sets no value. On real AgentX
    turns, top-p 0.95 gave a longer mean acceptance length than top-p 1.0 (k = 3, T 1.0, 512 tokens).

    Example:
        >>> from types import SimpleNamespace
        >>> chat_sampling_params(SimpleNamespace(chat_template_kwargs=None), environ={})
        {'temperature': 1.0, 'top_p': 0.95}
        >>> chat_sampling_params(SimpleNamespace(chat_template_kwargs={"thinking": False}), environ={})
        {'temperature': 0.6, 'top_p': 0.95}
    """
    if environ.get(SAMPLING_ENV, SAMPLING_MODES[0]) == "client":
        return {}
    kwargs = getattr(request, "chat_template_kwargs", None) or {}
    thinking = bool(kwargs.get("thinking", True)) if isinstance(kwargs, dict) else True
    return {"temperature": THINKING_TEMPERATURE if thinking else INSTANT_TEMPERATURE, "top_p": TOP_P}
