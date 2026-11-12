# SPDX-License-Identifier: Apache-2.0
"""SwiGLU MLP ``silu(x @ gate) * (x @ up) @ down`` on the nkilib MLP kernel, with a torch fallback.

One path only: BF16 weights, no bias, no fused norm or residual, no quantization.
"""

import functools
import math

import torch
from torch import Tensor

from nkilib.core.mlp.mlp import mlp as nkilib_mlp
from nkilib.core.mlp.mlp_parameters import BS_TILE_SIZE
from nkilib.core.utils.common_types import (
    ActFnType,
    ComputationMode,
    DtypeMode,
    NormType,
    QuantizationType,
)
from sglang_neuron.diffusion.neuron_api import nki_wrap_function
from sglang_neuron.diffusion.vn.utils.hardware_config import get_logical_nc_config
from sglang_neuron.diffusion.vn.utils.neuron_utils import can_run_kernel


@functools.cache
def wrapped_mlp():
    """The runtime's ``wrap_nki(nkilib_mlp)``, built on first use: importing a runtime at module
    import would start it before the worker claims its NeuronCore."""
    return nki_wrap_function()(nkilib_mlp)


# nkilib tiling limits (from its source; the NKI API does not export them).
_PMAX = 128
_NUM_HW_PSUM_BANKS = 8
_SRC_PROJ_INT_DIM_TILE_SIZE = 512
# The T bound at or below which this gate applies nkilib's TKG sharding constraint. nkilib's own
# TKG / CTE split is at TKG_BS_SEQLEN_THRESHOLD (96); 128 keeps T in (96, 128] off the CTE
# I-tiling check, which rejects large-I shapes that run.
_TKG_H_SHARDING_T_LIMIT = 128
_MIN_H_FOR_I_SHARDING = 7168
_MIN_I_FOR_I_SHARDING = 1024
_MAX_T_FOR_I_SHARDING = 256


def _can_use_kernel(hidden: Tensor, gate_w: Tensor) -> bool:
    """Whether the nkilib kernel can run ``hidden`` ``[T, H]`` with ``gate_w`` ``[H, I]``.

    ``H`` must be a multiple of 128. At small ``T`` (TKG) the kernel shards on T when
    ``T >= BS_TILE_SIZE * lnc`` and ``T % lnc == 0``, else on H, which needs ``H // 128`` to
    divide over the ``lnc`` cores. Otherwise (CTE) the intermediate dim's 512-wide tiles must fit
    the 8 PSUM banks, ``I`` halved over the cores when the kernel shards it.

    Example:
        >>> _can_use_kernel(torch.randn(32, 256), torch.randn(256, 512))  # CPU tensor
        False
    """
    if not can_run_kernel(hidden):
        return False
    T, H = hidden.shape
    if H % _PMAX != 0:
        return False
    lnc = get_logical_nc_config()
    if T <= _TKG_H_SHARDING_T_LIMIT:
        if T >= BS_TILE_SIZE * lnc and T % lnc == 0:
            return True
        return (H // _PMAX) % lnc == 0
    I = gate_w.shape[1]
    can_shard_on_i = (
        H >= _MIN_H_FOR_I_SHARDING and I >= _MIN_I_FOR_I_SHARDING and T <= _MAX_T_FOR_I_SHARDING
    )
    effective_i = I // lnc if can_shard_on_i else I
    return math.ceil(effective_i / _SRC_PROJ_INT_DIM_TILE_SIZE) <= _NUM_HW_PSUM_BANKS


def mlp(hidden: Tensor, gate_w: Tensor, up_w: Tensor, down_w: Tensor) -> Tensor:
    """``silu(hidden @ gate_w) * (hidden @ up_w) @ down_w`` on the kernel where it can run.

    ``hidden`` is ``[T, H]``, ``gate_w`` and ``up_w`` ``[H, I]``, ``down_w`` ``[I, H]``; the
    output is ``[T, H]``.

    Example:
        >>> T, H, I = 32, 256, 512
        >>> mlp(torch.randn(T, H), torch.randn(H, I), torch.randn(H, I), torch.randn(I, H)).shape
        torch.Size([32, 256])
    """
    if not _can_use_kernel(hidden, gate_w):
        return (torch.nn.functional.silu(hidden @ gate_w) * (hidden @ up_w)) @ down_w

    result = wrapped_mlp()[get_logical_nc_config()](
        hidden_tensor=hidden.unsqueeze(0),  # the kernel takes [B, S, H]
        gate_proj_weights_tensor=gate_w,
        up_proj_weights_tensor=up_w,
        down_proj_weights_tensor=down_w,
        normalization_weights_tensor=None,
        gate_proj_bias_tensor=None,
        up_proj_bias_tensor=None,
        down_proj_bias_tensor=None,
        normalization_bias_tensor=None,
        fused_add_tensor=None,
        store_fused_add_result=False,
        activation_fn=ActFnType.SiLU,
        normalization_type=NormType.NO_NORM,
        quantization_type=QuantizationType.NONE,
        gate_w_scale=None,
        up_w_scale=None,
        down_w_scale=None,
        gate_up_in_scale=None,
        down_in_scale=None,
        quant_clipping_bound=0.0,
        output_dtype=None,
        store_output_in_sbuf=False,
        eps=1e-6,
        skip_gate_proj=False,
        use_tkg_gate_up_proj_column_tiling=True,
        use_tkg_down_proj_column_tiling=True,
        use_tkg_down_proj_optimized_layout=False,
        gate_clamp_upper_limit=None,
        gate_clamp_lower_limit=None,
        up_clamp_upper_limit=None,
        up_clamp_lower_limit=None,
        mode=ComputationMode.AUTO,
        force_cte_mode=False,
        dtype_mode=DtypeMode.AUTO,
    )
    if isinstance(result, (tuple, list)):
        result = result[0]
    return result.squeeze(0)
