# SPDX-License-Identifier: Apache-2.0
"""GroupNorm (+ optional SiLU) for the FLUX.1 VAE decoder: one SBUF-resident pass per group.

Each group's ``channels_per_group`` channels are loaded once as a ``[128, channels_per_group, HW/128]``
tile (spatial positions across partitions), reduced to fp32 sum and sum of squares, combined across
partitions with a ones matmul, and written back with one Scalar-engine activation per channel that
applies ``func(x * scale_c + shift_c)``. The groups are split between the two cores of an LNC2 launch.
"""

import nki
import nki.isa as nisa
import nki.language as nl

from nkilib.core.utils.allocator import sizeinbytes
from nkilib.core.utils.kernel_assert import kernel_assert

_P = nl.tile_size.pmax


@nki.jit
def flux_group_norm_fwd(
    x: nl.NkiTensor,
    weight: nl.NkiTensor,
    bias: nl.NkiTensor,
    num_groups: int = 32,
    eps: float = 1e-6,
    silu: bool = False,
):
    """``GroupNorm(x)`` with affine, optionally followed by SiLU.

    Args:
        x (nl.NkiTensor): [1, C, H, W] bf16; H * W must be a multiple of 128.
        weight (nl.NkiTensor): [C] fp32 affine scale.
        bias (nl.NkiTensor): [C] fp32 affine shift.
        num_groups (int): Number of channel groups; C must be a multiple of it.
        eps (float): Variance epsilon.
        silu (bool): Apply SiLU to the normalized output.

    Returns:
        out (nl.NkiTensor): [1, C, H, W], dtype of x.
    """
    batch, channels, height, width = x.shape
    kernel_assert(batch == 1, "flux_group_norm_fwd supports batch 1")
    hw = height * width
    kernel_assert(hw % _P == 0, f"H * W must be a multiple of {_P}, got {hw}")
    kernel_assert(channels % num_groups == 0, "channels must be a multiple of num_groups")
    cpg = channels // num_groups
    free = hw // _P
    count = float(cpg * hw)
    group_bytes = cpg * free * (2 * sizeinbytes(x.dtype) + sizeinbytes(nl.float32))
    kernel_assert(
        group_bytes <= nl.tile_size.total_available_sbuf_size,
        f"one group's tiles need {group_bytes} B per partition, more than "
        f"{nl.tile_size.total_available_sbuf_size} B of SBUF",
    )

    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    xv = x.reshape((channels, _P, free))
    ov = out.reshape((channels, _P, free))

    num_shards = nl.num_programs(0) if nl.program_ndim() == 1 else 1
    shard_id = nl.program_id(0) if nl.program_ndim() == 1 else 0
    per_shard = (num_groups + num_shards - 1) // num_shards
    first = shard_id * per_shard
    last = min(num_groups, first + per_shard)

    ones = nl.ndarray((_P, _P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(ones, value=1.0)
    act = nl.silu if silu else nl.copy

    for group in range(first, last):
        c0 = group * cpg
        tile = nl.ndarray((_P, cpg * free), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=tile,
            src=xv.ap(pattern=[[free, _P], [_P * free, cpg], [1, free]], offset=c0 * _P * free),
        )
        stats = nl.ndarray((_P, 2), dtype=nl.float32, buffer=nl.sbuf)
        squares = nl.ndarray((_P, cpg * free), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=stats[:, 0:1], op=nl.add, data=tile, axis=[1], keepdims=True)
        nisa.activation(
            dst=squares,
            op=nl.square,
            data=tile,
            reduce_op=nl.add,
            reduce_res=stats[:, 1:2],
            reduce_cmd=nisa.reduce_cmd.reset_reduce,
        )
        totals_psum = nl.ndarray((_P, 2), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=totals_psum, stationary=ones, moving=stats)
        totals = nl.ndarray((_P, 2), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=totals, data=totals_psum, op0=nl.multiply, operand0=1.0 / count)
        mean = totals[:, 0:1]
        var = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=var, data1=mean, data2=mean, op=nl.multiply)
        nisa.tensor_tensor(dst=var, data1=totals[:, 1:2], data2=var, op=nl.subtract)
        inv = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=inv, data=var, op0=nl.maximum, operand0=0.0, op1=nl.add, operand1=eps
        )
        nisa.activation(dst=inv, op=nl.rsqrt, data=inv)

        w = nl.ndarray((_P, cpg), dtype=nl.float32, buffer=nl.sbuf)
        b = nl.ndarray((_P, cpg), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=w, src=weight.ap(pattern=[[0, _P], [1, cpg]], offset=c0))
        nisa.dma_copy(dst=b, src=bias.ap(pattern=[[0, _P], [1, cpg]], offset=c0))
        scale = nl.ndarray((_P, cpg), dtype=nl.float32, buffer=nl.sbuf)
        shift = nl.ndarray((_P, cpg), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=scale, data=w, op0=nl.multiply, operand0=inv)
        nisa.tensor_scalar(dst=shift, data=scale, op0=nl.multiply, operand0=mean)
        nisa.tensor_tensor(dst=shift, data1=b, data2=shift, op=nl.subtract)

        result = nl.ndarray((_P, cpg * free), dtype=x.dtype, buffer=nl.sbuf)
        for j in range(cpg):
            cols = slice(j * free, (j + 1) * free)
            nisa.activation(
                dst=result[:, cols],
                op=act,
                data=tile[:, cols],
                scale=scale[:, j : j + 1],
                bias=shift[:, j : j + 1],
            )
        nisa.dma_copy(
            dst=ov.ap(pattern=[[free, _P], [_P * free, cpg], [1, free]], offset=c0 * _P * free),
            src=result,
        )
    return out
