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

"""The partition-spread SBUF layout of a few token-major rows.

A decode rank's slice of the residual stream is ``[T_shard, H]`` with ``T_shard`` as small as 2. Loaded
as-is, every elementwise op on it runs on two partitions and walks ``H`` elements serially (slow at
H = 6144); the same contiguous bytes viewed as ``[128, T_shard * H / 128]`` put ``H / 64`` elements on
each partition and run much faster. The view is a plain reshape of a row-major tensor,
so loads and stores need no access pattern.
"""

_P_MAX = 128


def spread_rows_shape(T_shard: int, H: int) -> tuple[int, int] | None:
    """``[P_MAX, T_shard * H / P_MAX]`` for a ``[T_shard, H]`` row block whose tokens cover whole partition blocks.

    Partition ``p`` holds elements ``[p*F, (p+1)*F)`` of the flattened rows, so token ``t`` is partitions
    ``[t*ppt, (t+1)*ppt)`` with ``ppt = P_MAX / T_shard``. The per-token block must be 32-aligned (or the
    whole tile) so a block-diagonal constant can be built with partition-aligned memsets.

    Args:
        T_shard (int): Rows.
        H (int): Row width.

    Returns:
        tuple[int, int] | None: ``(P_MAX, F)`` with ``T_shard * H == P_MAX * F``, or None when the rows
            do not tile that way and the caller should stay token-major.
    """
    if (T_shard * H) % _P_MAX != 0 or _P_MAX % T_shard != 0:
        return None
    F = (T_shard * H) // _P_MAX
    ppt = _P_MAX // T_shard
    if H != ppt * F or (ppt % 32 != 0 and ppt != _P_MAX):
        return None
    return (_P_MAX, F)
