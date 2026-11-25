# SPDX-License-Identifier: Apache-2.0
"""Hy4-Flash's launch rules (catalog.ModelSpec.launch_hooks). Imports stay inside the functions: the launching process
reads this module first.

- The text decoder only: the checkpoint carries a vision tower (``vision_config``), which the graphs do not serve; a
  launch that names no ``--enable-multimodal`` is served text-only, one that asks for the tower is refused.
- The routed experts run in Trn2's FP8 (moe.py): their tensors are ``float8_e4m3fn`` holding at most 240, compiled as
  Trn2's own e4m3 (``UNSAFE_FP8FNCAST=1``, the host's compiler option).
- The config type is registered in every process (prepare_process; sglang_patches/hy4_flash_launch registers it too,
  when the plugin loads)."""

import os

FP8_CAST = "UNSAFE_FP8FNCAST"
FP8_FN_AS_E4M3 = "--experimental-unsafe-fp8e4m3fn-as-fp8e4m3"


def prepare_process() -> None:
    from sglang_neuron.sglang_patches.hy4_flash_launch import register_config

    register_config()


def kv_page_size(server_args) -> int:
    from sglang_neuron.platform import PAGE_SIZE

    return PAGE_SIZE


def apply_defaults(server_args) -> None:
    if getattr(server_args, "enable_multimodal", None) is None and hasattr(server_args, "enable_multimodal"):
        server_args.enable_multimodal = False


def launch_problems(server_args, checkpoint_config, environ=os.environ) -> list:
    from sglang_neuron.models.hy_v4_flash.config import config_problems, text_config

    problems = list(config_problems(text_config(checkpoint_config)))
    if getattr(server_args, "enable_multimodal", None):
        problems.append("--enable-multimodal: the Neuron graphs serve Hy4-Flash's text decoder only (no vision tower)")
    if getattr(server_args, "speculative_algorithm", None):
        problems.append(f"--speculative-algorithm {server_args.speculative_algorithm}: the DFly drafter is not served")
    return problems


def launch_settings(server_args) -> tuple:
    from sglang_neuron.backend.stack import get_platform_target, normalize_target_family
    from sglang_neuron.models import catalog

    if normalize_target_family(get_platform_target()) != "trn2":
        return ()
    if catalog.checkpoint_quantization(server_args) == "fp8":
        return (catalog.LaunchSetting(FP8_CAST, "1", "the FP8 routed experts compile float8_e4m3fn as Trn2's e4m3 "
                                      "(the converted experts are at most 240)"),)
    return ()


def rank_runtime_env(role) -> dict:
    return {}
