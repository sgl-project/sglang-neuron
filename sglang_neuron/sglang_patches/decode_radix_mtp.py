# SPDX-License-Identifier: Apache-2.0
"""A NEURON_MTP decode with --disaggregation-decode-enable-radix-cache: SGLang's
arg_groups.speculative_hook.handle_speculative_decoding refuses the decode radix cache for every speculative algorithm
but EAGLE / EAGLE3 / DSPARK, whose draft KV lives in the target's pool. NEURON_MTP keeps the MTP layer's KV the same
way: one more layer of the same pool, written over the prompt by the prefill (SGLANG_NEURON_HY3_PD_MTP_KV) and over
every accepted span by the decode step, so a cached prefix holds both. A cached position whose MTP KV the decode did
not write (a span's last draft) changes only what the drafts propose, never an output: the target verifies every token.
Only that refusal is lifted; every other check of the hook still runs."""

TARGET = "sglang.srt.arg_groups.speculative_hook.handle_speculative_decoding"


def handle_on_neuron(original_fn, server_args):
    from sglang_neuron.speculative.neuron_mtp import is_neuron_mtp

    lift = (str(getattr(server_args, "disaggregation_mode", "null")) == "decode"
            and getattr(server_args, "disaggregation_decode_enable_radix_cache", False)
            and is_neuron_mtp(getattr(server_args, "speculative_algorithm", None)))
    if not lift:
        return original_fn(server_args)
    object.__setattr__(server_args, "disaggregation_decode_enable_radix_cache", False)
    try:
        return original_fn(server_args)
    finally:
        object.__setattr__(server_args, "disaggregation_decode_enable_radix_cache", True)


def register(registry, around) -> None:
    registry.register(TARGET, handle_on_neuron, around)
