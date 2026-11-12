# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RMSNorm + MoE-input MX quantize, and the sigmoid top-K router on its output, for token generation.

The two stages are separate calls so a caller can dispatch the MX pair (pack it, issue its AllGathers) before
the router runs: nothing the gathers carry depends on the routing.
"""

import nki.isa as nisa
import nki.language as nl

from ..utils.kernel_assert import kernel_assert
from ..utils.scratch_hbm import scratch_hbm
from .rmsnorm_mx_quantize import rmsnorm_mx_quantize_tkg
from .router_topk import router_topk

# The router matmul contracts over H in bfloat16, so the normalized hidden states are cast to it
# when the model hands us float16 inputs.
_ROUTER_MM_DTYPE = nl.bfloat16

# SBUF partition count. The fused norm emits the unquantized hidden states as [128, T, H/128].
_P_MAX = 128

# The MX swizzle and the matmuls that consume the quantized hidden states tile H by 512
# (128 partitions x 4 x4-packed elements per tile).
_MX_H_TILE = 512

# The top-K selection is a max8 / nc_find_index8 reduction, which consumes whole groups of 8
# expert columns.
_TOPK_GROUP = 8

_ROUTER_ERROR_PREFIX = "[RMSNorm Router Quantize TKG Kernel]"


def rmsnorm_mx_quantize_for_router(
    hidden_states: nl.NkiTensor,
    gamma: nl.NkiTensor,
    eps: float,
    output_quant: nl.NkiTensor = None,
    output_scale: nl.NkiTensor = None,
    input_parts: tuple = None,
) -> tuple[nl.NkiTensor, nl.NkiTensor, nl.NkiTensor]:
    """RMSNorm, the MX pair, and the router's input: the first stage, before :func:`router_topk_from_normed`.

    The pair is ready well before the router's top-k, so a caller dispatches it first.

    Args:
        hidden_states (nl.NkiTensor): [B, S, H] hidden states in HBM.
        gamma (nl.NkiTensor): [1, H] RMSNorm weights in HBM.
        eps (float): RMSNorm epsilon.
        output_quant (nl.NkiTensor): Optional [128, H/512, B*S] x4-packed destination, in HBM or SBUF
            (both outputs in the same class). None allocates the pair in shared HBM.
        output_scale (nl.NkiTensor): Optional [128, H/512, B*S] uint8 micro-scale destination.
        input_parts (tuple): Optional two [B, S, H] HBM tensors whose sum is ``hidden_states``
            (:func:`rmsnorm_mx_quantize_tkg`).

    Returns:
        tuple[nl.NkiTensor, nl.NkiTensor, nl.NkiTensor]: (hidden_quant, hidden_scale, router_in), the MX
            pair as handed in or allocated, and the normalized hidden states in SBUF as [128, T, H/128]
            in the router matmul's dtype.
    """
    B, S, H = hidden_states.shape
    T = B * S
    H_free = H // _P_MAX
    num_H_tiles = H // _MX_H_TILE
    _validate_rmsnorm_router_quantize(T=T, H=H)

    # The MoE-consumable MX pair: x4-packed quantized hidden states and their uint8 micro-scales,
    # both in the [128, H/512, T] layout the expert projections read without a reshape.
    kernel_assert(
        (output_quant is None) == (output_scale is None),
        f"{_ROUTER_ERROR_PREFIX} output_quant and output_scale must be given together",
    )
    hidden_quant = output_quant
    hidden_scale = output_scale
    if hidden_quant is None:
        hidden_quant = nl.ndarray((_P_MAX, num_H_tiles, T), dtype=nl.float8_e4m3fn_x4, buffer=nl.shared_hbm)
        hidden_scale = nl.ndarray((_P_MAX, num_H_tiles, T), dtype=nl.uint8, buffer=nl.shared_hbm)

    # RMSNorm writes the unquantized result to SBUF for the router and the MX pair for the MoE. The
    # SBUF layout is [128, T, H/128], which is the layout the router's x loader expects.
    rmsnorm_out_sb = nl.ndarray((_P_MAX, T, H_free), dtype=hidden_states.dtype, buffer=nl.sbuf)
    rmsnorm_mx_quantize_tkg(
        input=hidden_states,
        gamma=gamma,
        output=rmsnorm_out_sb,
        output_quant=hidden_quant,
        output_scale=hidden_scale,
        eps=eps,
        input_parts=input_parts,
    )

    router_in = rmsnorm_out_sb
    if rmsnorm_out_sb.dtype != _ROUTER_MM_DTYPE:
        router_in = nl.ndarray((_P_MAX, T, H_free), dtype=_ROUTER_MM_DTYPE, buffer=nl.sbuf)
        nisa.tensor_copy(dst=router_in, src=rmsnorm_out_sb)
    return hidden_quant, hidden_scale, router_in


def router_topk_from_normed(
    router_in: nl.NkiTensor,
    router_weights: nl.NkiTensor,
    router_bias: nl.NkiTensor,
    top_k: int,
    routed_scaling_factor: float,
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """The sigmoid top-K router on the normalized states: the second stage, after
    :func:`rmsnorm_mx_quantize_for_router`.

    Args:
        router_in (nl.NkiTensor): [128, T, H/128] normalized hidden states in SBUF, in the router
            matmul's dtype (what :func:`rmsnorm_mx_quantize_for_router` returns).
        router_weights (nl.NkiTensor): [H, E] bfloat16 router weights in HBM.
        router_bias (nl.NkiTensor): [1, E] selection bias in HBM.
        top_k (int): Experts selected per token.
        routed_scaling_factor (float): Final multiplier on the L1-normalized affinities.

    Returns:
        tuple[nl.NkiTensor, nl.NkiTensor, nl.NkiTensor]: (expert_index [T, top_k] int32 in shared HBM,
            expert_affinities [T, E] bfloat16 in private HBM -- a collective source, expert_affinities_sb
            the [T, E] SBUF tile the affinities were stored from -- an on-chip handle on the router's end).
    """
    _, T, H_free = router_in.shape
    H = H_free * _P_MAX
    _, E = router_weights.shape
    _validate_rmsnorm_router_quantize(T=T, H=H, E=E, top_k=top_k)

    expert_index = nl.ndarray((T, top_k), dtype=nl.int32, buffer=scratch_hbm())
    # private_hbm: the MoE layer AllGathers the affinities over the EP group directly, and at LNC=1 the
    # runtime rejects a collective whose sources are shared_hbm.
    expert_affinities = nl.ndarray((T, E), dtype=nl.bfloat16, buffer=nl.private_hbm)

    _, _, expert_affinities_sb = router_topk(
        x=router_in,
        w=router_weights,
        w_bias=router_bias,
        expert_affinities=expert_affinities,
        expert_index=expert_index,
        k=top_k,
        routed_scaling_factor=routed_scaling_factor,
    )
    return expert_index, expert_affinities, expert_affinities_sb


def _validate_rmsnorm_router_quantize(T: int, H: int, E: int = None, top_k: int = None) -> None:
    """
    Reject the launches and shapes the two stages of this module do not handle.

    Args:
        T (int): Number of tokens (B * S).
        H (int): Hidden dimension.
        E (int): Number of routed experts, or None to skip the router checks (quantize stage alone).
        top_k (int): Number of experts selected per token, or None likewise.

    Returns:
        None
    """
    # Single NeuronCore only: there is no token split, so a second program would recompute every
    # token and race with the first one on the shared-HBM outputs.
    kernel_assert(
        nl.program_ndim() == 0 or nl.program_ndim() == 1,
        f"{_ROUTER_ERROR_PREFIX} only no specialization or specialization along one axis is supported, got "
        f"nl.program_ndim()={nl.program_ndim()}",
    )
    n_programs = nl.num_programs(axes=0) if nl.program_ndim() != 0 else 1
    kernel_assert(
        n_programs == 1,
        f"{_ROUTER_ERROR_PREFIX} runs on a single NeuronCore (LNC=1), got {n_programs} programs",
    )

    # One token per SBUF partition: the router emits a single token tile and does not tile T.
    kernel_assert(
        T <= _P_MAX,
        f"{_ROUTER_ERROR_PREFIX} T = B * S must be at most {_P_MAX} tokens, got T={T}",
    )

    # The MX swizzle tiles H by 512 and does not pad a partial tile.
    kernel_assert(
        H % _MX_H_TILE == 0,
        f"{_ROUTER_ERROR_PREFIX} H must be a multiple of {_MX_H_TILE}, got H={H}",
    )

    if E is None:
        return
    # The top-K reduction walks the E expert columns in groups of 8, and cannot select more
    # experts than exist.
    kernel_assert(
        E % _TOPK_GROUP == 0 and top_k <= E,
        f"{_ROUTER_ERROR_PREFIX} E must be a multiple of {_TOPK_GROUP} and top_k must be at most E, got "
        f"E={E}, top_k={top_k}",
    )
