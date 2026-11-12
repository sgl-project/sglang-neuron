# SPDX-License-Identifier: Apache-2.0
"""MXFP8 (E4M3, x4) weight packing for the giga MoE expert kernels, without nki.

The layout is the one that the expert kernels read (models/qwen35/graph.py operand shapes). The numerics:
one power-of-two scale per 32-element hardware group, exponent
``floor(log2(amax)) - 8``, values clipped to +-448 and rounded to E4M3.

  w_q  [128, K/512, N]  float8_e4m3fn_x4: x4 slot j of (p, tile) holds contraction index
                        k = j*(K/4) + tile*128 + p   (kind "h": gate / up, K = H)
                        k = tile*512 + j*128 + p      (kind "i": down, K = I)
  w_s  [16, K/512, N]   uint8 E8M0, one scale per 8 partitions x 4 slots (row = p // 8)

Returned as raw bytes: ``w_q`` is uint8 ``[128, K/512, N, 4]`` (the 4 bytes of one
x4 element, slot j at byte j; nki's ``float8_e4m3fn_x4`` itemsize is 4), so the
packer needs neither nki nor neuron_dtypes. Expert tensors (per rank, E_L experts):

  gate_up_w     [E_L, 128, 2, H/512, I] x4   (gate, up stacked on axis 2)
  gate_up_scale [E_L, 16, 2, H/512, I] u8
  down_w        [E_L, 128, I/512, H] x4
  down_scale    [E_L, 16, I/512, H] u8
"""

from __future__ import annotations

import numpy as np

P = 128
Q = 4


def pack_mx_fp8(w, kind: str, act: bool = False):
    """W [K, N] (contraction first) -> (w_q uint8 [128, K/512, N, 4], w_s uint8 [16, K/512, N])."""
    import torch

    w = torch.as_tensor(np.asarray(w, dtype=np.float32)) if not isinstance(w, torch.Tensor) else w.float()
    K, N = w.shape
    if K % (Q * P):
        raise ValueError(f"contraction dim {K} must be a multiple of 512")
    n_t = K // (Q * P)
    if kind == "h":
        g = w.reshape(Q, n_t, P, N).permute(2, 1, 0, 3)  # [p, tile, j, N]
    elif kind == "i":
        g = w.reshape(n_t, Q, P, N).permute(2, 0, 1, 3)
    else:
        raise ValueError(kind)
    grp = g.reshape(P // 8, 8, n_t, Q, N)
    amax = grp.abs().amax(dim=(1, 3), keepdim=True).clamp(min=2.0**-120)
    # E8M0 encodes exponents -127..127 (byte 255 is NaN). All-zero groups are common in the FP8
    # checkpoint (~65% of expert bytes are zero), and their clamped amax would give -128 -> 0xFF.
    e = (torch.floor(torch.log2(amax)) - 8 + (1 if act else 0)).clamp(-127, 127)
    y = (grp / torch.pow(2.0, e)).clamp(-448, 448).to(torch.float8_e4m3fn)
    data = y.reshape(P, n_t, Q, N).permute(0, 1, 3, 2).contiguous().view(torch.uint8).numpy()
    w_s = (e.reshape(P // 8, n_t, N) + 127).to(torch.uint8).numpy()
    return np.ascontiguousarray(data), np.ascontiguousarray(w_s)


def unpack_mx_fp8(w_q, w_s, kind: str):
    """Inverse of pack_mx_fp8 (dequantised fp32 W [K, N]); verification only."""
    import torch

    w_q = np.asarray(w_q, dtype=np.uint8)
    P_, n_t, N, _ = w_q.shape
    y = torch.from_numpy(np.ascontiguousarray(w_q)).view(torch.float8_e4m3fn).float()  # [P, n_t, N, 4]
    y = y.permute(0, 1, 3, 2)  # [P, n_t, j, N]
    s = torch.from_numpy(np.asarray(w_s, dtype=np.uint8).astype(np.float32)) - 127.0  # [16, n_t, N]
    scale = torch.pow(2.0, s).repeat_interleave(8, dim=0)[:, :, None, :]  # [P, n_t, 1, N]
    g = y * scale
    if kind == "h":
        return g.permute(2, 1, 0, 3).reshape(Q * n_t * P_, N).numpy()
    return g.permute(1, 2, 0, 3).reshape(n_t * Q * P_, N).numpy()


def pack_expert(gate, up, down):
    """One expert from HF [I, H] gate_proj / up_proj and [H, I] down_proj weights
    (any float) -> (gate_up_w [128, 2, H/512, I, 4], gate_up_scale [16, 2, H/512, I],
    down_w [128, I/512, H, 4], down_scale [16, I/512, H])."""
    import torch

    g = torch.as_tensor(gate).float().T.contiguous()  # [H, I]
    u = torch.as_tensor(up).float().T.contiguous()
    d = torch.as_tensor(down).float().T.contiguous()  # [I, H]
    gq, gs = pack_mx_fp8(g, "h")
    uq, us = pack_mx_fp8(u, "h")
    dq, ds = pack_mx_fp8(d, "i")
    return (
        np.stack([gq, uq], axis=1),
        np.stack([gs, us], axis=1),
        dq,
        ds,
    )


def expert_shapes(H: int, I: int, e_l: int, layers: int = 0) -> dict:
    """Raw-byte shapes (uint8) of the four expert operands for ``e_l`` experts per
    rank, optionally stacked over ``layers``."""
    lead = (layers,) if layers else ()
    return {
        "gate_up_w": lead + (e_l, P, 2, H // 512, I, 4),
        "gate_up_scale": lead + (e_l, 16, 2, H // 512, I),
        "down_w": lead + (e_l, P, I // 512, H, 4),
        "down_scale": lead + (e_l, 16, I // 512, H),
    }
