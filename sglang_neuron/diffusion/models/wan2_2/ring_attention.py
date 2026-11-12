# SPDX-License-Identifier: Apache-2.0
"""Wan2.2's context-parallel self-attention: the const-max ring kernel on local K/V.

The kernel is Wan's copy (``kernels/wan/ring_attention_const_max_fwd.py``). It needs no online-max
pass: it bounds the softmax maximum at one value per query row from the Cauchy-Schwarz L2 bound on
the scaled scores, c_i = scale * |q_i| * max_j |k_j|. Query rows do not rotate through the ring, so
every c_i is static across ring steps and the cross-step reduction is pure addition. K and V stay
local: the kernel drives its own ring of collective_permute_implicit exchanges across the CP group.

Where the kernel cannot run (CPU tensors, fake-tensor tracing, NKI kernels disabled, LNC 1)
attention falls back to the shared all-gather K/V + flash (``cp_attention.cp_self_attention``). No
setting selects the path.
"""

import logging

import torch

from sglang_neuron.diffusion.attention import cp_attention
from sglang_neuron.diffusion.kernels.kernel_choice import device_reason, record_kernel_choice
from sglang_neuron.diffusion.kernels.wan.ring_attention_const_max_fwd import (
    ring_attention_const_max_fwd,
)
from sglang_neuron.diffusion.neuron_api import wrap_nki_kernel_if_supported
from sglang_neuron.diffusion.vn.utils.neuron_utils import can_run_kernel

logger = logging.getLogger(__name__)

# The ring-path decisions already logged, so that each is logged once per process.
_RING_LOGGED: set[tuple[bool, str]] = set()


@torch._dynamo.assume_constant_result
def _log_ring_choice(use_ring: bool, reason: str) -> bool:
    """Log the ring-attention decision once, at trace time and off the graph; return ``use_ring``.

    The two lines are the on-device proof of which CP self-attention path ran:
    ``WAN_RING_KERNEL=const_max`` for the ring kernel and ``WAN_RING_FALLBACK=allgather`` (a
    WARNING) for the all-gather + flash fallback. Dynamo folds the call to a constant, like
    :func:`record_kernel_choice`, so nothing is logged from inside the compiled graph.
    """
    decision = (use_ring, reason)
    if decision not in _RING_LOGGED:
        _RING_LOGGED.add(decision)
        if use_ring:
            logger.info("WAN_RING_KERNEL=const_max ring_attention_const_max_fwd reason=%s", reason)
        else:
            logger.warning("WAN_RING_FALLBACK=allgather reason=%s", reason)
    return use_ring


def _can_use_ring_attention_kernel(value: torch.Tensor) -> bool:
    """Whether the ring-attention kernel can run: no shape limits, only device and LNC checks."""
    if not can_run_kernel(value):
        reason = device_reason(value)
        record_kernel_choice("ring_attention", False, reason)
        return _log_ring_choice(False, reason)
    if _ring_attention_launch is None:
        reason = "the ring kernel runs at LNC 2 only"
        record_kernel_choice("ring_attention", False, reason, by_design=True)
        return _log_ring_choice(False, reason)
    record_kernel_choice("ring_attention", True, "eligible")
    return _log_ring_choice(True, "eligible")


# Built once, at import, like cp_attention's _attention_launch: a wrap_nki call inside the traced
# forward breaks a warm start from a Dynamo precompile package. LNC 2 only: None at LNC 1, where
# the gate takes the all-gather path.
_ring_attention_launch = wrap_nki_kernel_if_supported(
    ring_attention_const_max_fwd, supported_lnc=(2,)
)


def ring_cp_self_attention(query, key, value, scale, cp_size, cp_group, cp_replica_groups=None):
    """Context-parallel self-attention of the Wan2.2 DiT blocks.

    Takes already-projected, RoPE'd, head-split q/k/v **token-major** in ``[B, local_S, N, D]``
    (what the qkv projection + qk-norm + RoPE produce, so nothing transposes on the way in) and
    returns the attention output **d-major** as ``[B, N, D, local_S]``, the layout the
    o-projection kernels consume.

    With ``cp_size > 1`` the ring kernel is the path: it transposes Q/K to d-major itself and
    returns head-major ``[B, N, local_S, D]``. Self-attention is non-causal, so ``softmax_scale``
    is passed directly. ``cp_replica_groups`` is a tuple-of-tuples of the global ranks of every CP
    ring (each SPMD rank resolves its own ring; ``parallel_state.get_cp_replica_groups``) and the
    CP degree is the kernel's ``num_workers``. Without replica groups, or where the kernel cannot
    run, this is the all-gather + flash path.
    """
    if cp_size > 1 and cp_replica_groups is not None and _can_use_ring_attention_kernel(value):
        ring_out = _ring_attention_launch(
            q=query,
            k=key,
            v=value,
            replica_groups=cp_replica_groups,
            num_workers=cp_size,
            softmax_scale=scale,
            training=False,  # no LSE out for inference
        )
        # The ring kernel emits seq-major; transpose to the d-major contract here.
        return ring_out.transpose(2, 3).contiguous()
    return cp_attention.cp_self_attention(query, key, value, scale, cp_size, cp_group)
