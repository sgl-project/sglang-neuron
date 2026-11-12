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

"""GLM-5.2 DSA indexer: resolve the logical key positions of a paged indexer cache to physical rows, once per step.

The cache tensor keeps its shape and dtype; a block table permutes blocks within it (as ``prior_index`` does for the
MLA cache), which lets two sequences share rows. The rows come back in natural logical order (entry ``[p, tile]`` is
logical key ``tile * P_MAX + p``), so the residency mask's single threshold, the key-major ``mask_add`` layout and
the top-k's logical positions all stay as they are. The table has a guard column 0, like MLA's, so logical block
``b`` is column ``b + 1``.

Unallocated entries (negative, and the guard) resolve to block 0. That is only correct for blocks at or past the
step's base position, which the mask drops. A missing block inside the live range would be scored with block 0's
keys, and no other rank holds those keys, so the host must guarantee that every live block is allocated.
"""

import nki
import nki.isa as nisa
import nki.language as nl

from ...utils.kernel_assert import kernel_assert
from .indexer_constants import P_MAX


def _log2_int(value: int) -> int:
    """Exact log2 of a power of two."""
    bits = value.bit_length() - 1
    kernel_assert(value == 1 << bits, f"{value} must be a power of two")
    return bits


@nki.jit
def indexer_block_rows(
    block_table: nl.NkiTensor,
    num_keys: int,
    block_size: int,
) -> nl.NkiTensor:
    """
    Resolve every logical key of this rank's shard to a physical cache row.

    Usage:
        Tuned for: GLM-5.2 DSA indexer decode, once per step per rank, LNC=1
        Applicable to: any position-ordered key shard stored in a paged cache of equal row count

    Args:
        block_table (nl.NkiTensor): [1, num_keys // block_size + 1] int32 @ HBM. Entry ``[0, b + 1]``
            is the physical block holding this shard's ``b``-th LOGICAL block, or negative if
            unallocated. **Column 0 is a guard and is never a real block** -- it matches the MLA
            table's width convention (see the module docstring) and should be negative.
            One row because the indexer carries no batch axis -- its grid gives one sequence per
            ``key_size`` ranks, so a kernel instance serves exactly one sequence.
        num_keys (int): Logical key positions this rank holds. A multiple of ``P_MAX``.
        block_size (int): Rows per block. A power of two, and a divisor of ``num_keys``.

    Returns:
        nl.NkiTensor: [P_MAX, num_keys // P_MAX] uint32 @ HBM. Entry ``[p, tile]`` is the physical row
        of logical key ``tile * P_MAX + p``. shared_hbm so the kernel can run standalone; it is only
        ever read by DMA.
    """
    kernel_assert(num_keys % P_MAX == 0, f"num_keys={num_keys} must be a multiple of {P_MAX}")
    kernel_assert(num_keys % block_size == 0, f"num_keys={num_keys} must be a multiple of block_size={block_size}")
    n_tiles = num_keys // P_MAX
    n_blocks = num_keys // block_size
    block_shift = _log2_int(block_size)
    table_width = n_blocks + 1  # column 0 is the guard; logical block b sits at column b + 1
    kernel_assert(
        tuple(block_table.shape) == (1, table_width),
        f"block_table is {tuple(block_table.shape)}, expected (1, {table_width}) for "
        f"{num_keys} keys at block_size={block_size} (n_blocks + 1: column 0 is the guard)",
    )

    # logical_key[p, tile] = tile * P_MAX + p -- the same natural order the residency mask uses, so
    # the two tables are consistent by construction rather than by convention.
    logical = nl.ndarray((P_MAX, n_tiles), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(logical, pattern=[[P_MAX, n_tiles]], offset=0, channel_multiplier=1)

    # + 1 for the guard column: logical block b is read at table column b + 1.
    logical_block = nl.ndarray((P_MAX, n_tiles), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(logical_block, logical, op0=nl.right_shift, operand0=block_shift)
    nisa.tensor_scalar(logical_block, logical_block, op0=nl.add, operand0=1)
    intra = nl.ndarray((P_MAX, n_tiles), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(intra, logical, op0=nl.bitwise_and, operand0=block_size - 1)

    # The table is small and every partition may index any entry, so broadcast it once (stride 0 on
    # the partition axis) and look it up on chip. An indirect DMA per lookup would move 4 bytes per
    # descriptor, which is the traffic profile that dominates a gather -- the same reason
    # prior_index._resolve_local_rows does its block-table lookup with nc_n_gather.
    table_sb = nl.ndarray((P_MAX, table_width), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=table_sb[0:P_MAX, 0:table_width],
        src=block_table.reshape((table_width, 1)).ap(pattern=[[0, P_MAX], [1, table_width]], offset=0),
    )
    logical_block_u32 = nl.ndarray((P_MAX, n_tiles), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=logical_block_u32, src=logical_block)
    physical_block = nl.ndarray((P_MAX, n_tiles), dtype=nl.int32, buffer=nl.sbuf)
    nisa.nc_n_gather(dst=physical_block, data=table_sb[0:P_MAX, 0:table_width], indices=logical_block_u32)

    # An UNALLOCATED block (negative entry, including the column-0 guard) still has to yield an
    # in-range row, because the gather that consumes this table moves real bytes for every slot --
    # skipping would leave the destination untouched and the score matmul reads it directly.
    #
    # This clamp covers the TAIL of the window ONLY: positions at or beyond this step's base, which
    # indexer_residency_mask has already driven to -MASK_BOUND, so the bytes are discarded by the mask
    # rather than by the row choice. An unallocated block INSIDE the live range is neither handled nor
    # detected -- see "WHAT THE INTEGRATION MUST GUARANTEE" in the module docstring for why masking it
    # would not be a fix either, and where the check belongs.
    nisa.tensor_scalar(physical_block, physical_block, op0=nl.maximum, operand0=0)

    row = nl.ndarray((P_MAX, n_tiles), dtype=nl.int32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=row,
        data=physical_block,
        op0=nl.multiply,
        operand0=float(block_size),
        op1=nl.add,
        operand1=intra,
    )
    # Never let a resolved row leave the tensor. A dtype-changing copy goes through an fp32
    # intermediate whose integer cast SATURATES rather than wraps, so an out-of-range value would
    # land on a real row rather than fault -- clamp explicitly instead of relying on that.
    nisa.tensor_scalar(row, row, op0=nl.minimum, operand0=num_keys - 1)

    row_u32 = nl.ndarray((P_MAX, n_tiles), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=row_u32, src=row)
    out = nl.ndarray((P_MAX, n_tiles), dtype=nl.uint32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=row_u32)
    return out
