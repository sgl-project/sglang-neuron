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

"""Collective helpers of the Kimi-K3 prefill block kernels (KDA block, MLA block).

The helpers keep every collective payload in ``private_hbm`` (LNC1), and they put no GpSimd work
and no DMA transpose beside a collective.

Feature-major image (the layout of the MoE block kernel). For a row block ``x [Tl, H]`` of rank ``r``
with ``nT = Tl / 128`` tiles and ``HK = H / 128`` hidden chunks, the image of the rank is
``[128, nT * HK * 128 + extra]``: column ``(i * HK + kc) * 128 + t`` of partition ``p`` holds
``x[i * 128 + t, kc * 128 + p]``. The ``extra`` columns hold caller data (for example ``f_a^T``). The
all-gather stacks the images of the ranks on dim 0: rank ``r`` at rows ``r * 128 .. r * 128 + 127``.
A consumer loads one 128-row tile with one DMA of contiguous runs and needs no transpose.
"""

import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl

_P = 128
_VEC = nisa.engine.vector
_ACT = nisa.engine.scalar


def pe_transpose(dst, data, ident):
    """Transpose ``data [K, n]`` into ``dst [n, K]`` on TensorE in transpose mode (bit exact).

    Transpose mode moves each element; it does not multiply. A NaN in one row thus stays in its own
    column of ``dst`` (a matmul with an identity gives 0 x NaN = NaN in every column).

    Example:
        >>> pe_transpose(ps[:, 0:128], rows[:, 0:128], ident_bf16)
    """
    k = data.shape[0]
    nisa.nc_matmul(
        dst=dst, stationary=data, moving=ident[0:k, 0:k], is_transpose=True, is_moving_onezero=True, accumulate=False
    )


def transpose_rows_bf16(rows, dst_T, ident_bf16, name):
    """Transpose ``rows [128 t, n_cols]`` bf16 into ``dst_T [128 c, n_cols / 128, 128 t]`` (8 per PSUM bank).

    Example:
        >>> transpose_rows_bf16(rows, xt, id_bf, "kb_xT0")
    """
    n_tiles = rows.shape[1] // _P
    per_bank = 8
    for first in range(0, n_tiles, per_bank):
        count = min(per_bank, n_tiles - first)
        ps = nl.ndarray((_P, per_bank, _P), dtype=nl.bfloat16, buffer=nl.psum, name=f"{name}_ps{first}")
        for j in range(count):
            k = first + j
            pe_transpose(ps[:, j, :], rows[:, k * _P : (k + 1) * _P], ident_bf16)
        nisa.tensor_copy(dst=dst_T[:, first : first + count, :], src=ps[:, 0:count, :], engine=_VEC)


def gather_rows_feature_major(image_src, image_dst, group, name):
    """All-gather the feature-major images of the ranks (one HBM collective, dim 0).

    Args:
        image_src: ``[128, cols]`` bf16 ``private_hbm``, the image of this rank.
        image_dst: ``[W * 128, cols]`` bf16 ``private_hbm``.
        group: The ``ReplicaGroup``.
        name: The collective name.

    Example:
        >>> gather_rows_feature_major(src, dst, group, "kb_agx")
    """
    ncc.all_gather(srcs=[image_src], dsts=[image_dst], replica_group=group, collective_dim=0, name=name)


def reduce_scatter_rows(src, dst, group, name):
    """Reduce-scatter ``src [T, n]`` by row blocks into ``dst [T / W, n]`` (sum, one HBM collective).

    Example:
        >>> reduce_scatter_rows(piece_src, piece_dst, group, "kb_rs0")
    """
    ncc.reduce_scatter(srcs=[src], dsts=[dst], replica_group=group, collective_dim=0, op=nl.add, name=name)
