# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3's context-parallel self-attention: the NKI const-max ring kernel on local K/V, or an
all-gather and local flash attention where the kernel cannot run."""

import torch

from sglang_neuron.diffusion.attention.cp_attention import nki_attention
from sglang_neuron.diffusion.kernels.kernel_choice import device_reason, record_kernel_choice
from sglang_neuron.diffusion.neuron_api import wrap_nki_kernel
from sglang_neuron.diffusion.vn.utils.neuron_utils import can_run_kernel


def _can_use_ring_attention_kernel(value: torch.Tensor) -> bool:
    """Whether the ring kernel can run; it has no shape limits."""
    if not can_run_kernel(value):
        return record_kernel_choice("minimax_h3_ring_attention", False, device_reason(value))
    return record_kernel_choice("minimax_h3_ring_attention", True, "eligible")


def _ring_attend(query, key, value, replica_groups, num_workers, scale, num_pad_keys=0):
    """Ring attention with the NKI ``ring_attention_const_max_fwd`` kernel.

    No online max: the softmax max of query row ``i`` is bounded by Cauchy–Schwarz,
    ``c_i = scale·‖q_i‖·max_j‖k_j‖``. Query rows stay on their rank, so ``c_i`` is fixed across
    ring steps and the steps' partial sums just add.

    Input token-major ``[B, local_S, N, D]``; output d-major ``[B, N, D, local_S]`` (``tp_out``),
    which is what the o-projection kernel consumes — the kernel transposes each output tile on the
    PE, so the attention output is never transposed in HBM. The kernel rotates K/V over the CP
    group itself; ``replica_groups`` holds every CP ring's global ranks. ``num_pad_keys`` counts the
    zero pad keys that end the sequence, whose share of each row's sum the kernel subtracts.
    """
    from sglang_neuron.diffusion.kernels.flux.ring_attention_const_max_fwd import (
        ring_attention_const_max_fwd,
    )

    return wrap_nki_kernel(ring_attention_const_max_fwd, supported_lnc=(2,))(
        q=query,
        k=key,
        v=value,
        replica_groups=replica_groups,
        num_workers=num_workers,
        softmax_scale=scale,
        training=False,  # no LSE output for inference
        num_pad_keys=num_pad_keys,
        tp_out=True,
    )


def cp_ring_self_attention(
    query,
    key,
    value,
    scale,
    cp_size,
    cp_group,
    cp_replica_groups,
    kv_valid=None,
    num_pad=0,
):
    """Context-parallel self-attention on local K/V.

    Takes RoPE'd, head-split q/k/v token-major ``[B, local_S, N, D]`` and returns d-major
    ``[B, N, D, local_S]`` for the o-projection. ``kv_valid`` ``[1, local_S]`` (1 real, 0 pad)
    marks this rank's pad keys; ``num_pad`` counts them over the CP group, where they end the
    sequence. Both paths exclude them exactly.

    Where the ring kernel cannot run (CP size 1, CPU, fake-tensor tracing, NKI kernels disabled),
    K/V are all-gathered over the CP group for local flash attention.
    """
    if kv_valid is not None:
        # Zero K keeps pad keys out of the bound's max‖k‖; zero K and V let the kernel subtract
        # their exact share of each row's sum.
        valid = kv_valid.to(key.dtype)[:, :, None, None]
        key = key * valid
        value = value * valid

    if cp_size > 1 and _can_use_ring_attention_kernel(value):
        # The kernel writes the o-projection's layout itself (see _ring_attend).
        return _ring_attend(
            query,
            key,
            value,
            replica_groups=cp_replica_groups,
            num_workers=cp_size,
            scale=scale,
            num_pad_keys=num_pad,
        )

    query = query.transpose(1, 2)  # [B, local_S, N, D] -> [B, N, local_S, D]
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    if cp_size > 1:
        key = cp_group.all_gather(key.contiguous(), dim=2)
        value = cp_group.all_gather(value.contiguous(), dim=2)
        if num_pad:
            key, value = key[:, :, :-num_pad], value[:, :, :-num_pad]
    return nki_attention(query, key, value, scale)
