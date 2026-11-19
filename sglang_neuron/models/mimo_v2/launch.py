# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2's launch rules (catalog.ModelSpec.launch_hooks). Imports stay inside the functions: the launching process
reads this module first.

- The text decoder only: MiMo-V2.6's checkpoint carries a vision and an audio tower (``vision_config``,
  ``audio_config``), so SGLang would serve it multimodal; a launch that names no ``--enable-multimodal`` is served
  text-only, and one that asks for the towers is refused.
- The KV page is the platform's 128 tokens, which the sliding window (128) is a whole number of: nkilib's segmented
  prefill loads a window's pages only when the window divides by the page.
- The routed experts run in Trn2's FP8 (experts_fp8.py): their kernels
  compile ``float8_e4m3fn`` only as Trn2's own e4m3 (``UNSAFE_FP8FNCAST=1``), whose 240 bounds the converted values."""

import os

FP8_CAST = "UNSAFE_FP8FNCAST"
# neuronx-cc compiles a graph's float8_e4m3fn tensors on Trn2 only as its own e4m3 (models/hy_v3/sglang_model.py)
FP8_FN_AS_E4M3 = "--experimental-unsafe-fp8e4m3fn-as-fp8e4m3"


def fp8_experts(server_args, environ=os.environ) -> bool:
    """Whether the launch's experts are FP8: a quantized checkpoint (always served in FP8)."""
    from sglang_neuron.models import catalog

    return catalog.checkpoint_quantization(server_args) == "fp8"


def kv_page_size(server_args) -> int:
    from sglang_neuron.platform import PAGE_SIZE

    return PAGE_SIZE


def apply_defaults(server_args) -> None:
    """Serve the text decoder: a launch without ``--enable-multimodal`` runs without the towers."""
    if getattr(server_args, "enable_multimodal", None) is None and hasattr(server_args, "enable_multimodal"):
        server_args.enable_multimodal = False


def launch_problems(server_args, checkpoint_config, environ=os.environ) -> list:
    from sglang_neuron.models.mimo_v2.config import config_problems

    problems = list(config_problems(checkpoint_config))
    if getattr(server_args, "enable_multimodal", None):
        problems.append("--enable-multimodal: the Neuron graphs serve MiMo-V2's text decoder only (no vision or audio "
                        "tower)")
    if getattr(server_args, "mm_enable_dp_encoder", False):
        problems.append("--mm-enable-dp-encoder: no multimodal encoder is served")
    page = getattr(server_args, "page_size", None) or kv_page_size(server_args)
    window = int(checkpoint_config.get("sliding_window") or 0)
    if window and window % int(page):
        problems.append(f"--page-size {page}: the sliding window ({window}) must be a whole number of pages")
    return problems


def launch_settings(server_args) -> tuple:
    from sglang_neuron.backend.stack import get_platform_target, normalize_target_family
    from sglang_neuron.models import catalog

    if normalize_target_family(get_platform_target()) != "trn2":
        return ()
    if fp8_experts(server_args):
        return (catalog.LaunchSetting(FP8_CAST, "1", "the FP8 expert kernels compile float8_e4m3fn as Trn2's e4m3 "
                                      "(the converted experts are at most 240)"),)
    if str(getattr(server_args, "kv_cache_dtype", "auto") or "auto").startswith("fp8"):
        return (catalog.LaunchSetting(FP8_CAST, "1", "the FP8 KV cache compiles float8_e4m3fn as Trn2's e4m3 (the "
                                      "graphs clip its values to 240)"),)
    return ()


def rank_runtime_env(role) -> dict:
    return {}

