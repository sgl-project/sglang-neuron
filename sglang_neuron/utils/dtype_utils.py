# SPDX-License-Identifier: Apache-2.0
import torch

# TRN2 uses e4m3 (with inf), max finite = 240.
# TRN3 uses e4m3fn (no inf), max finite = 448.
_FP8_E4M3_MAX = 240.0
_FP8_E4M3FN_MAX = 448.0


def _resolve_fp8_clamp_max() -> float:
    from sglang_neuron.backend import envs
    from sglang_neuron.backend.stack import get_platform_target

    try:
        target = get_platform_target()
    except RuntimeError:
        # No platform override and no NRT (bare CPU run).
        # CPU supports finite FP8 dtypes, so use the e4m3fn max.
        if envs.SGLANG_NEURON_CPU_MODE:
            return _FP8_E4M3FN_MAX
        raise

    if target.startswith("trn3"):
        return _FP8_E4M3FN_MAX
    else:
        return _FP8_E4M3_MAX


# Resolved once at import time so it's a constant during torch.compile tracing.
FP8_CLAMP_MAX: float = _resolve_fp8_clamp_max()


def is_fp8(t: torch.Tensor) -> bool:
    return t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)


def fp8_round(t: torch.Tensor, fp8_dtype: torch.dtype, dtype: torch.dtype) -> torch.Tensor:
    """``t`` clamped to the FP8 range, cast to ``fp8_dtype`` and back to ``dtype``: the values an FP8 store keeps."""
    return t.float().clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX).to(fp8_dtype).to(dtype)
