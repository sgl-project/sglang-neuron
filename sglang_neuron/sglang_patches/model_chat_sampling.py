# SPDX-License-Identifier: Apache-2.0
"""The sampling values of the served model for a chat request, over the values of SGLang's chat API.

SGLang turns a ``/v1/chat/completions`` request into its sampling parameters in
``ChatCompletionRequest.to_sampling_params``. It takes the values of the request, else the generation config of the
model, else the OpenAI defaults (temperature 1.0, top-p 1.0). A model can fix the sampling of its chat API with the
launch hook ``chat_sampling_params(request) -> dict`` (catalog.ModelSpec.launch_hooks). The Kimi-K3 launch rules
give the Kimi-K3 values (models/kimi_k3/launch.chat_sampling_params). This AROUND hook writes the values of the hook
over the result of SGLang. A model without the hook keeps the behaviour of SGLang. A ``/generate`` request and each
field that the hook does not return also keep it.

SGLang also adds the grammar of its tool-call parser here (the ``tool_call_constraint`` argument: a structural tag, a
JSON schema or an EBNF for a request with tools). A model whose launch hooks set ``CHAT_TOOL_CALL_GRAMMAR = False``
samples without a grammar, so the hook drops that argument for ``tool_choice`` ``auto``: the request is served without
the grammar, and the parser still reads the calls from the output. ``tool_choice`` ``required`` or a named tool needs
the grammar to force the call, so the hook refuses it (HTTP 400) and does not serve it with the choice dropped. A grammar
that the request asks for itself (``response_format``, ``regex``, ``ebnf``) stays, and the model's admission rules refuse
it as before.

Example::

    model_chat_sampling.register(HookRegistry, HookType.AROUND)  # general_plugin, every process
"""

from __future__ import annotations

_PROTOCOL = "sglang.srt.entrypoints.openai.protocol"
TARGETS_MD5 = {
    # the request's sampling dict: its own values, the generation config, then _DEFAULT_SAMPLING_PARAMS
    (_PROTOCOL, "ChatCompletionRequest.to_sampling_params"): "682600a77b91c5d148dc61503a8c257c",
}


# the position of ``tool_call_constraint`` in to_sampling_params(stop, model_generation_config, tool_call_constraint, ...)
_TOOL_CALL_CONSTRAINT_ARG = 2
FORCED_CALL_REFUSAL = ("tool_choice 'required' and a named tool are not served: this server applies no grammar of the "
                       "tool-call parser, so it cannot force a call. Send tool_choice 'auto' or 'none'")


def forces_a_call(request) -> bool:
    """Whether the request's ``tool_choice`` forces a tool call: ``required`` or a named tool.

    Example:
        >>> from types import SimpleNamespace
        >>> [forces_a_call(SimpleNamespace(tool_choice=c)) for c in ("auto", "none", "required", None)]
        [False, False, True, False]
    """
    choice = getattr(request, "tool_choice", None)
    return choice == "required" or (choice is not None and not isinstance(choice, str))


def without_tool_call_grammar(args: tuple, kwargs: dict) -> tuple:
    """``(args, kwargs)`` of a to_sampling_params call with no ``tool_call_constraint``.

    Example:
        >>> without_tool_call_grammar(([], {}, ("structural_tag", "tag")), {})
        (([], {}, None), {})
        >>> without_tool_call_grammar(([], {}), {"tool_call_constraint": ("json_schema", "{}")})
        (([], {}), {'tool_call_constraint': None})
    """
    if len(args) > _TOOL_CALL_CONSTRAINT_ARG:
        args = (*args[:_TOOL_CALL_CONSTRAINT_ARG], None, *args[_TOOL_CALL_CONSTRAINT_ARG + 1:])
    if "tool_call_constraint" in kwargs:
        kwargs = {**kwargs, "tool_call_constraint": None}
    return args, kwargs


def to_sampling_params(original_fn, self, *args, **kwargs):
    """AROUND ChatCompletionRequest.to_sampling_params: SGLang's sampling dict, then the served model's chat sampling
    values (launch hooks ``chat_sampling_params``) over it. For a model whose launch hooks set
    ``CHAT_TOOL_CALL_GRAMMAR = False``, SGLang builds the dict without the tool-call parser's grammar, and a request
    whose ``tool_choice`` forces a call is refused (``FORCED_CALL_REFUSAL``)."""
    from sglang_neuron.models import catalog

    hooks = catalog.served_launch_hooks()
    if hooks is not None and getattr(hooks, "CHAT_TOOL_CALL_GRAMMAR", True) is False:
        constraint = (args[_TOOL_CALL_CONSTRAINT_ARG] if len(args) > _TOOL_CALL_CONSTRAINT_ARG
                      else kwargs.get("tool_call_constraint"))
        if constraint is not None and forces_a_call(self):
            raise ValueError(FORCED_CALL_REFUSAL)
        args, kwargs = without_tool_call_grammar(args, kwargs)
    params = original_fn(self, *args, **kwargs)
    if hooks is not None and hasattr(hooks, "chat_sampling_params"):
        params.update(hooks.chat_sampling_params(self))
    return params


def register(registry, around) -> None:
    """Register the hook (general_plugin, every process: the chat API runs in the HTTP server's process)."""
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(TARGETS_MD5, __name__)
    registry.register(f"{_PROTOCOL}.ChatCompletionRequest.to_sampling_params", to_sampling_params, around)
