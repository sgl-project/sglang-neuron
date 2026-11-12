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

"""Collectives with an SBUF interface for operands that are produced and consumed on chip.

Every call takes and returns SBUF tiles and bounces through private HBM (store, HBM collective, load), because the
SBUF-destination form of ``nki.collectives`` is not deterministic in the 128-rank draft chain on trn3.
"""

import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from ...utils.kernel_assert import kernel_assert
from ...utils.trace_names import unique_name
from .cc_pin import pin

_P_MAX = 128


def all_reduce_sbuf(
    src_sb: nl.NkiTensor, replica_group: ReplicaGroup, op, name: str, handles_out: dict = None
) -> nl.NkiTensor:
    """All-reduce a ``[P, F]`` SBUF tile over ``replica_group`` into a fresh ``[P, F]`` SBUF tile, through private HBM.

    Args:
        src_sb (nl.NkiTensor): [P, F] SBUF, free axis contiguous (a whole tile or a partition slice of one).
        replica_group (ReplicaGroup): The group.
        op: ``nl.add``, ``nl.maximum`` or ``nl.minimum``.
        name (str): Debug name stem of the HBM bounce buffers.
        handles_out (dict): Optional; receives ``"all_reduce"``, the collective's instruction.

    Returns:
        nl.NkiTensor: [P, F] SBUF, the same on every rank of the group.
    """
    P, F = src_sb.shape
    dst_sb = nl.ndarray((P, F), dtype=src_sb.dtype, buffer=nl.sbuf)
    dst_hbm = all_reduce_hbm(src_sb, replica_group, op, name, handles_out=handles_out)
    nisa.dma_copy(dst=dst_sb, src=dst_hbm)
    return dst_sb


def all_reduce_hbm(
    src_sb: nl.NkiTensor, replica_group: ReplicaGroup, op, name: str, handles_out: dict = None
) -> nl.NkiTensor:
    """All-reduce a ``[P, F]`` SBUF tile through private HBM, the result left in HBM.

    The store and the HBM AllReduce of :func:`all_reduce_sbuf` without its load: the caller
    loads the result where and how it consumes it (in column blocks, say).

    Args:
        src_sb (nl.NkiTensor): [P, F] SBUF, free axis contiguous.
        replica_group (ReplicaGroup): The group.
        op: ``nl.add``, ``nl.maximum`` or ``nl.minimum``.
        name (str): Debug name stem of the HBM buffers.
        handles_out (dict): Optional; receives ``"all_reduce"``, the collective's instruction.

    Returns:
        nl.NkiTensor: [P, F] private HBM, the same on every rank of the group.
    """
    P, F = src_sb.shape
    src_hbm = nl.ndarray((P, F), dtype=src_sb.dtype, buffer=nl.private_hbm, name=unique_name(f"{name}_src"))
    dst_hbm = nl.ndarray((P, F), dtype=src_sb.dtype, buffer=nl.private_hbm, name=unique_name(f"{name}_dst"))
    nisa.dma_copy(dst=src_hbm, src=src_sb)
    pin(src_hbm), pin(dst_hbm)  # long context only; see utils/cc_pin.py
    cc = ncc.all_reduce(srcs=[src_hbm], dsts=[dst_hbm], replica_group=replica_group, op=op)
    if handles_out is not None:
        handles_out["all_reduce"] = cc
    return dst_hbm


def all_gather_columns_sbuf(
    src_sb: nl.NkiTensor, replica_group: ReplicaGroup, group_size: int, name: str, handles_out: dict = None
) -> nl.NkiTensor:
    """All-gather a ``[P, F]`` SBUF column shard into ``[P, group_size * F]``, through private HBM: position ``r``'s at
    columns ``[F r, F (r + 1))``.

    Args:
        src_sb (nl.NkiTensor): [P, F] SBUF, free axis contiguous.
        replica_group (ReplicaGroup): The group, IN COLUMN ORDER.
        group_size (int): Ranks in the group.
        name (str): Debug name stem of the HBM bounce buffers.
        handles_out (dict): Optional; receives ``"all_gather"``, the collective's instruction.

    Returns:
        nl.NkiTensor: [P, group_size * F] SBUF, the same on every rank of the group.
    """
    P, F = src_sb.shape
    kernel_assert(P <= _P_MAX, f"[SBUF collectives] {name}: {P} rows exceed {_P_MAX} partitions")
    dst_sb = nl.ndarray((P, group_size * F), dtype=src_sb.dtype, buffer=nl.sbuf)
    rows_hbm = all_gather_columns_hbm(src_sb, replica_group, group_size, name, handles_out=handles_out)
    load_gathered_columns(dst_sb, rows_hbm, group_size)
    return dst_sb


def all_gather_columns_hbm(
    src_sb: nl.NkiTensor, replica_group: ReplicaGroup, group_size: int, name: str, handles_out: dict = None
) -> nl.NkiTensor:
    """All-gather a ``[P, F]`` SBUF column shard through private HBM, left in HBM in the collective's rank-major order.

    The store and the HBM AllGather of :func:`all_gather_columns_sbuf` without its relayout
    load: a consumer that reads by DMA (an indirect row pick, a later load) takes the gathered rows from HBM directly.

    Args:
        src_sb (nl.NkiTensor): [P, F] SBUF, free axis contiguous.
        replica_group (ReplicaGroup): The group, IN COLUMN ORDER.
        group_size (int): Ranks in the group.
        name (str): Debug name stem of the HBM buffers.
        handles_out (dict): Optional; receives ``"all_gather"``, the collective's instruction.

    Returns:
        nl.NkiTensor: [group_size * P, F] private HBM, row ``r P + p`` = position ``r``'s columns of row ``p``.
    """
    P, F = src_sb.shape
    src_hbm = nl.ndarray((P, F), dtype=src_sb.dtype, buffer=nl.private_hbm, name=unique_name(f"{name}_src"))
    rows_hbm = nl.ndarray(
        (group_size * P, F), dtype=src_sb.dtype, buffer=nl.private_hbm, name=unique_name(f"{name}_rows")
    )
    nisa.dma_copy(dst=src_hbm, src=src_sb)
    pin(src_hbm), pin(rows_hbm)  # long context only; see utils/cc_pin.py
    cc = ncc.all_gather(srcs=[src_hbm], dsts=[rows_hbm], replica_group=replica_group, collective_dim=0)
    if handles_out is not None:
        handles_out["all_gather"] = cc
    return rows_hbm


def load_gathered_columns(dst_sb: nl.NkiTensor, rows_hbm: nl.NkiTensor, group_size: int) -> nl.NkiTensor:
    """Load :func:`all_gather_columns_hbm`'s ``[group_size * P, F]`` into the complete ``[P, group_size * F]`` SBUF rows.

    Returns:
        nl.NkiTensor: ``dst_sb``.
    """
    P, W = dst_sb.shape
    F = W // group_size
    # Gathered row (r, p) is partition p's columns [F r, F (r + 1)): one DMA reading (p, r, f).
    nisa.dma_copy(
        dst=dst_sb.reshape((P, group_size, F)),
        src=rows_hbm.ap(pattern=[[F, P], [P * F, group_size], [1, F]], offset=0),
    )
    return dst_sb


__all__ = [
    "all_reduce_sbuf",
    "all_reduce_hbm",
    "all_gather_columns_sbuf",
    "all_gather_columns_hbm",
    "load_gathered_columns",
]
