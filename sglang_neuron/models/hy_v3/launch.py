# SPDX-License-Identifier: Apache-2.0
"""Hy3's launch rules (catalog.ModelSpec.launch_hooks).

An FP8 KV cache on a Trn2 target: its kernels compile ``float8_e4m3fn`` only as the target's own FP8 format
(``UNSAFE_FP8FNCAST=1``, NKI's legacy cast to Trn2's e4m3, whose clamp of 240 the KV scales and the platform's
FP8_CLAMP_MAX are for); a Trn3 target has the format natively. A launch with an FP8 cache also needs a KV scale profile
(kv_scales.py). Imports stay inside the functions: the launching process reads this module first."""

import os

PROFILE_ENV = "SGLANG_NEURON_HY3_KV_SCALES"
FP8_CAST = "UNSAFE_FP8FNCAST"


def _fp8_kv(server_args) -> bool:
    return str(getattr(server_args, "kv_cache_dtype", "auto")).startswith("fp8")


def _trn2() -> bool:
    from sglang_neuron.backend.stack import get_platform_target, normalize_target_family

    return normalize_target_family(get_platform_target()) == "trn2"


def kv_page_size(server_args) -> int:
    from sglang_neuron.platform import PAGE_SIZE

    return PAGE_SIZE


def launch_problems(server_args, checkpoint_config, environ=os.environ) -> list:
    if not _fp8_kv(server_args):
        return []
    path = environ.get(PROFILE_ENV) or os.path.join(str(server_args.model_path), "kv_scales.safetensors")
    if not os.path.isfile(path):
        return [f"--kv-cache-dtype {server_args.kv_cache_dtype} needs Hy3's KV scale profile: set {PROFILE_ENV} or put "
                f"kv_scales.safetensors in the checkpoint directory ({path} does not exist)"]
    return []


def launch_settings(server_args) -> tuple:
    from sglang_neuron.models import catalog

    if _fp8_kv(server_args) and _trn2():
        return (catalog.LaunchSetting(FP8_CAST, "1", "the FP8 KV kernels compile float8_e4m3fn as Trn2's e4m3 "
                                      "(clamp 240, the range of the KV scales)"),)
    return ()


def rank_runtime_env(role) -> dict:
    return {}
