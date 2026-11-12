# SPDX-License-Identifier: Apache-2.0
"""Tensor-parallel GELU feed-forward on the NKI MLP kernel."""

import math

import nki
import torch
import torch.nn as nn
import torch.nn.functional as F
from nkilib.core.mlp.mlp import mlp as nkilib_mlp
from nkilib.core.utils.common_types import (
    ActFnType,
    DtypeMode,
    NormType,
    QuantizationType,
)

from sglang_neuron.diffusion.distributed.groups import (
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from sglang_neuron.diffusion.distributed.sequence_parallel import (
    sp_all_gather_seq,
    sp_entry,
    sp_exit,
    sp_reduce_scatter_seq,
)
from sglang_neuron.diffusion.distributed.tp_loaders import (
    col_bias_loader,
    col_weight_loader,
    row_bias_loader,
    row_weight_loader,
)
from sglang_neuron.diffusion.kernels.kernel_choice import device_reason, record_kernel_choice
from sglang_neuron.diffusion.neuron_api import wrap_nki_kernel
from sglang_neuron.diffusion.vn.utils.neuron_utils import can_run_kernel
from sglang_neuron.utils.weight_loader import set_weight_loader


@nki.jit
def _wan_mlp_kernel(
    hidden,
    up_weight,
    down_weight,
    up_bias,
    down_bias,
):
    """
    Wan FFN:
        output = GELU_tanh(hidden @ up_weight + up_bias)
                 @ down_weight + down_bias

    Input:
        hidden:      [B, T, H]
        up_weight:   [H, I]
        down_weight: [I, H_out]
        up_bias:     [1, I]
        down_bias:   [1, H_out]

    Output:
        [B, T, H_out]
    """
    return nkilib_mlp(
        hidden_tensor=hidden,
        # The kernel requires a gate weight argument even when skip_gate_proj=True.
        # It is unused, so reuse up_weight.
        gate_proj_weights_tensor=up_weight,
        up_proj_weights_tensor=up_weight,
        down_proj_weights_tensor=down_weight,
        normalization_weights_tensor=None,
        # gate bias is also unused when skip_gate_proj=True.
        gate_proj_bias_tensor=up_bias,
        up_proj_bias_tensor=up_bias,
        down_proj_bias_tensor=down_bias,
        normalization_bias_tensor=None,
        fused_add_tensor=None,
        store_fused_add_result=False,
        activation_fn=ActFnType.GELU_Tanh_Approx,
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
        skip_gate_proj=True,
        # Match the configuration of vn.functional.mlp.
        use_tkg_gate_up_proj_column_tiling=True,
        use_tkg_down_proj_column_tiling=True,
        use_tkg_down_proj_optimized_layout=False,
        gate_clamp_upper_limit=None,
        gate_clamp_lower_limit=None,
        up_clamp_upper_limit=None,
        up_clamp_lower_limit=None,
        force_cte_mode=False,
        dtype_mode=DtypeMode.AUTO,
    )


# Wrapped once, at import, as a module-level launcher. A wrap_nki call inside compiled code makes
# Dynamo guard on the @nki.jit object's __wrapped__; on a warm start from a Dynamo precompile
# package that attribute comes back as _Missing and the guard fails. Launches at the configured LNC.
_wan_mlp_launch = wrap_nki_kernel(_wan_mlp_kernel, supported_lnc=(1, 2))


def _wan_nki_mlp(
    hidden: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_bias: torch.Tensor,
    down_bias: torch.Tensor,
) -> torch.Tensor:
    return _wan_mlp_launch(
        hidden,
        up_weight,
        down_weight,
        up_bias,
        down_bias,
    )


# nkilib MLP kernel constraints for the grid=2 (lnc=2) launch. The
# Wan kernel runs skip_gate_proj=True (up -> GELU_tanh -> down), so only the
# up/down projection dims are constrained.
_MLP_TKG_BS_SEQLEN_THRESHOLD = 128


_MLP_SRC_PROJ_INT_DIM_TILE_SIZE = 512


_MLP_NUM_HW_PSUM_BANKS = 8


def _can_use_wan_mlp_kernel(
    hidden: torch.Tensor,
    up_weight: torch.Tensor,
    *operands: torch.Tensor,
) -> bool:
    """Whether the nkilib MLP kernel can run for these Wan FFN inputs.

    Same contract as ``cp_attention._can_use_o_proj_kernel``: returns False (so the
    caller takes the torch fallback) when NKI kernels can't run on this device
    (CPU mode, native fake-tensor tracing, or kernels disabled — via
    ``can_run_kernel``) or when the dimensions violate the grid=2 tiling rules:

    * Hidden dim ``H`` must be 128-aligned.
    * **TKG mode** (``B * T <= 128``): ``H`` is sharded across 2 cores, so
      ``H // 128`` must be even, i.e. ``H % 256 == 0``.
    * **CTE mode** (``B * T > 128``): intermediate-dim tiles must fit the PSUM banks —
      ``ceil(I / 512) <= 8``.
    * Every operand must share ``hidden``'s dtype, and that dtype must not be fp32.
      nkilib advertises fp32 input but silently downgrades compute to bf16 for it
      (``mlp_cte_constants._get_compute_data_type``: ``io_type == float32 -> bfloat16``),
      and then loads the tile with ``nisa.dma_transpose``, which cannot convert dtype —
      so an fp32 operand aborts the kernel with "dma_transpose src and dst types should
      match, got src.dtype=float32, dst.dtype=bfloat16". Reproduced on trn2.48xlarge
      at ``[1, 4096, 5120] x [5120, 3456]``.

    ``hidden`` is ``[B, T, H]`` and ``up_weight`` is ``[H, I]`` (see
    :func:`_wan_mlp_kernel`). The Wan FFN always passes biases, so the nkilib
    I-sharding fast path (bias-free only) never applies and is not modeled here.
    """
    if not can_run_kernel(hidden):
        return record_kernel_choice("wan_mlp", False, device_reason(hidden))

    # fp32 has no working path through the kernel (see the dtype rule above), and a mixed
    # set of operands hits the same dma_transpose dtype check on whichever one differs.
    dtypes = tuple(t.dtype for t in (hidden, up_weight, *operands) if t is not None)
    if hidden.dtype == torch.float32 or any(dt != hidden.dtype for dt in dtypes):
        return record_kernel_choice("wan_mlp", False, "fp32 or mixed operand dtypes")

    if hidden.dim() != 3 or up_weight.dim() != 2:
        return record_kernel_choice("wan_mlp", False, "unexpected operand rank")

    B, T, H = hidden.shape
    tokens = B * T
    inner_dim = up_weight.shape[1]

    if H % 128 != 0:
        return record_kernel_choice("wan_mlp", False, "H is not 128-aligned")

    if tokens <= _MLP_TKG_BS_SEQLEN_THRESHOLD:
        # TKG: hidden dim sharded across 2 cores, H // 128 must be even.
        if H % 256 != 0:
            return record_kernel_choice("wan_mlp", False, "TKG: H is not 256-aligned")
        return record_kernel_choice("wan_mlp", True, "eligible (TKG)")

    # CTE: PSUM-bank constraint on the intermediate dimension.
    if math.ceil(inner_dim / _MLP_SRC_PROJ_INT_DIM_TILE_SIZE) > _MLP_NUM_HW_PSUM_BANKS:
        return record_kernel_choice(
            "wan_mlp", False, "CTE: intermediate dim exceeds the PSUM banks"
        )
    return record_kernel_choice("wan_mlp", True, "eligible (CTE)")


def _wan_mlp(
    hidden: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_bias: torch.Tensor,
    down_bias: torch.Tensor,
) -> torch.Tensor:
    """Wan FFN via the NKI MLP kernel with a torch fallback.

    Mirrors :func:`o_proj`: use the kernel when
    :func:`_can_use_wan_mlp_kernel` allows, otherwise compute
    ``GELU_tanh(hidden @ up_weight + up_bias) @ down_weight + down_bias`` in
    torch (matching the kernel's ``ActFnType.GELU_Tanh_Approx``).
    """
    if _can_use_wan_mlp_kernel(hidden, up_weight, down_weight, up_bias, down_bias):
        return _wan_nki_mlp(hidden, up_weight, down_weight, up_bias, down_bias)

    up = torch.matmul(hidden, up_weight) + up_bias
    up = F.gelu(up, approximate="tanh")
    return torch.matmul(up, down_weight) + down_bias


class TPFeedForward(nn.Module):
    """TP-sharded GELU feed-forward with raw nn.Parameter (LLaMA pattern).

    The up projection is column-parallel and the down projection row-parallel, both stored
    transposed; the down bias is pre-scaled by ``1 / tp`` (``row_bias_loader``). ``forward`` (the
    Wan2.2 blocks) uses the NKI MLP kernel (skip_gate=True, GELU_Tanh_Approx) for
    up_proj -> GELU(tanh) -> down_proj, then reduces the TP partial sums (all-reduce, or
    reduce-scatter under sequence parallelism). The FLUX blocks use only the parameters: they
    compute ``GELU_tanh(x @ up + up_bias) @ down + down_bias`` themselves
    (``FluxDoubleBlock._feed_forward``) and reduce the TP partial sums themselves.
    """

    def __init__(
        self,
        dim: int,
        inner_dim: int,
        dim_out: int | None = None,
        bias: bool = True,
        tp_sequence_parallel: bool = False,
    ):
        super().__init__()
        dim_out = dim_out or dim
        tp_size = get_tensor_model_parallel_world_size()
        self.tp_size = tp_size
        self.tp_group = get_tp_group().device_group
        self.sp_enabled = tp_sequence_parallel
        self.sp_real_len: int | None = None
        self.sp_padded_len: int | None = None
        # The SP reduce-scatter's summation dtype (None: the output dtype).
        self.sp_reduce_dtype: torch.dtype | None = None
        inner_per_rank = inner_dim // tp_size

        # Up projection: [dim, inner_per_rank] (transposed layout)
        self.up_proj_weight = nn.Parameter(torch.empty(dim, inner_per_rank))
        set_weight_loader(self.up_proj_weight, col_weight_loader(inner_per_rank, tp_size))
        if bias:
            self.up_proj_bias = nn.Parameter(torch.empty(inner_per_rank))
            set_weight_loader(self.up_proj_bias, col_bias_loader(inner_per_rank, tp_size))
        else:
            self.up_proj_bias = None

        # Down projection: [inner_per_rank, dim_out] (transposed layout)
        self.down_proj_weight = nn.Parameter(torch.empty(inner_per_rank, dim_out))
        set_weight_loader(self.down_proj_weight, row_weight_loader(inner_per_rank, tp_size))
        if bias:
            self.down_proj_bias = nn.Parameter(torch.empty(dim_out))
            if tp_size > 1:
                set_weight_loader(self.down_proj_bias, row_bias_loader(tp_size, dim_out))
        else:
            self.down_proj_bias = None

    def _use_padded_sp_path(self) -> bool:
        """Whether SP already provides the even sequence length required by the MLP."""
        return (
            self.sp_enabled
            and self.tp_size > 1
            and self.sp_padded_len is not None
            and self.sp_padded_len % 2 == 0
        )

    def _sp_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the token-wise FFN without slicing and restoring SP padding."""
        gathered = sp_all_gather_seq(hidden_states, self.tp_group, self.tp_size)
        output = _wan_mlp(
            gathered,
            self.up_proj_weight,
            self.down_proj_weight,
            self.up_proj_bias.unsqueeze(0),
            self.down_proj_bias.unsqueeze(0),
        )
        return sp_reduce_scatter_seq(output, self.tp_group, self.tp_size, self.sp_reduce_dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Unlike attention, the FFN is token-wise, so the padded row cannot affect live rows.
        if self._use_padded_sp_path():
            return self._sp_forward(hidden_states)

        hidden_states = sp_entry(
            hidden_states, self.tp_group, self.tp_size, self.sp_enabled, self.sp_real_len
        )
        input_shape = hidden_states.shape

        hidden_2d = hidden_states.reshape(-1, input_shape[-1])
        original_tokens = hidden_2d.shape[0]

        # Pad an odd token count by one row for the grid=(2,) launch.
        pad_tokens = original_tokens % 2
        if pad_tokens:
            hidden_2d = F.pad(hidden_2d, (0, 0, 0, 1))

        # NKI-Lib MLP expects [B, S, H].
        # Use B=1 and S=total token count.
        hidden_3d = hidden_2d.unsqueeze(0).contiguous()

        output_3d = _wan_mlp(
            hidden_3d,
            self.up_proj_weight,
            self.down_proj_weight,
            self.up_proj_bias.unsqueeze(0),
            self.down_proj_bias.unsqueeze(0),
        )

        # [1, T, H_out] -> [T, H_out]
        output = output_3d.squeeze(0)

        if pad_tokens:
            output = output[:original_tokens]

        output = output.reshape(*input_shape[:-1], output.shape[-1])
        return sp_exit(
            output,
            self.tp_group,
            self.tp_size,
            self.sp_enabled,
            self.sp_padded_len,
            self.sp_reduce_dtype,
        )
