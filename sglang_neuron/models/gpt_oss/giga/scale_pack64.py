# SPDX-License-Identifier: Apache-2.0
"""The PACK64 layout of the gate/up MX scales, which the giga-kernel's MoE reads by default.

The kernel reads the layout its VLLM_NEURON_MEGA_SCALE_PACK64 flag selects (on by default, and the kernel asserts the
packed shape when on). The launch refuses a giga flag it does not pin, so the host remaps the loaded scales once, at
weight load, and refuses a kernel that froze the flag off at its import (ops.gpt_oss_giga_decode.scale_pack64).
"""
import torch

SCALE_P16 = 16  # scale partitions of the unpacked gate/up layout
SCALE_GATE_UP = 2
SCALE_ROWS_PER_QUADRANT = 4
SCALE_FREE_PER_ROW = 128


def pack64_gate_up_scale(scale: torch.Tensor) -> torch.Tensor:
    """Remap a gate/up MX scale ``[..., E, 16, 2, H/512, I]`` to the kernel's PACK64 layout
    ``[..., E, 2, 64, H/512, I/4]``.

    The packed load reads 16 scale rows per SBUF quadrant instead of 4: the 4-wide I factor moves out of the free
    dimension into the partitions. For data free index ``i = tile_i*512 + k*128 + j``, the scale of partition row
    ``4*q + r`` goes to partition ``16*q + 4*k + r`` and free index ``(h, tile_i*128 + j)``. The remap is exact and
    invertible (the kernel library's own reference is ``pack64_gate_up_scale`` in its test utilities)."""
    if scale.ndim < 5:
        raise ValueError(f"the unpacked gate/up MX scale has dimensions [..., E, 16, 2, H/512, I]; got "
                         f"{tuple(scale.shape)}")
    *lead, p16, g, n_h512, i_dim = scale.shape
    if p16 != SCALE_P16 or g != SCALE_GATE_UP:
        raise ValueError(f"the unpacked gate/up MX scale is [..., E, {SCALE_P16}, {SCALE_GATE_UP}, H/512, I]; got "
                         f"{tuple(scale.shape)}")
    q_n = p16 // SCALE_ROWS_PER_QUADRANT
    r_n = k_n = SCALE_ROWS_PER_QUADRANT
    j_n = SCALE_FREE_PER_ROW
    tile_i = k_n * j_n
    if i_dim % tile_i:
        raise ValueError(f"I={i_dim} is not a multiple of {tile_i}: the PACK64 layout needs whole {tile_i}-wide I "
                         "tiles")
    n_ti = i_dim // tile_i
    n_lead = len(lead)
    x = scale.reshape(*lead, q_n, r_n, g, n_h512, n_ti, k_n, j_n)
    # relative axes 0=q 1=r 2=g 3=n_h512 4=n_ti 5=k 6=j -> g, q, k, r, n_h512, n_ti, j
    perm = tuple(range(n_lead)) + tuple(n_lead + a for a in (2, 0, 5, 1, 3, 4, 6))
    x = x.permute(*perm).contiguous()
    return x.reshape(*lead, g, q_n * k_n * r_n, n_h512, n_ti * j_n)
