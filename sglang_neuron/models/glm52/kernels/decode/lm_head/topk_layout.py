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

"""Shapes, sentinels and trace-time helpers for a ``nisa.topk`` call over a ``[BxS, vocab]`` row set.

``nisa.topk`` reads its input in a 16-partition "snake" layout (one top-k per group of 16 partitions, 8 rows per
128-partition tile). ``chunked_topk`` (``decode/lm_head/chunked_topk.py``) runs its inlined top-k with the config,
validation and de-snake column runs here.
"""

from dataclasses import dataclass
from typing import Tuple

import nki.language as nl
import numpy as np

from ..utils.kernel_assert import kernel_assert
from .tail_helpers import div_ceil

# Most-negative finite bfloat16: the sentinel of whole pad rows, whose results are never read, and the NaN fold
# target (bfloat16 has no unrepresentable band, so a NaN slot comes back as this finite value). Snake slots inside a
# real row are padded with -inf instead: a finite pad outranks a real -inf logit and would enter the top-k.
BFLOAT16_MIN = -3.3895314e38

# nisa.topk runs one independent top-k per group of 16 partitions.
PARTS_PER_GROUP = 16
# Hardware max partition dimension (8 groups of 16).
PMAX = 128
GROUPS_PER_TILE = PMAX // PARTS_PER_GROUP  # 8


@dataclass(frozen=True, eq=True)
class TopkConfig(nl.NKIObject):
    """Configuration of one nisa.topk top-k over ``[BxS, vocab]``.

    Attributes:
        BxS: Combined batch*sequence dimension (number of rows).
        vocab_size: Length of the dimension reduced over (n for nisa.topk).
        k: Number of largest elements to return.
        sorted: Whether output must be in descending order.
        inp_dtype: Input/value data type (must be bfloat16).
        index_dtype: Output index data type (uint32).
        n_prgs: Number of logical cores (SPMD programs) over BxS.
        prg_id: Program id (set at config-build time for BxS == 1 path).
        per_lnc_BxS: Rows handled per logical core.
        out_shape: Logical output shape (mirrors input leading dims + k).
    """

    BxS: int
    vocab_size: int
    k: int
    sorted: bool
    inp_dtype: np.dtype
    index_dtype: np.dtype
    n_prgs: int
    prg_id: int
    per_lnc_BxS: int
    out_shape: tuple


def create_topk_config(
    inp_shape: Tuple,
    inp_dtype: np.dtype,
    k: int,
    sorted: bool = True,
    num_programs: int = 2,
) -> TopkConfig:
    """Build a TopkConfig from an input shape (2D or 3D) and parameters."""
    BxS = 1
    for d in inp_shape[:-1]:
        BxS *= d
    vocab_size = inp_shape[-1]
    out_shape = tuple(list(inp_shape[:-1]) + [k])

    n_prgs = num_programs
    prg_id = 0
    if BxS == 1:
        n_prgs = 1
        prg_id = 0

    per_lnc_BxS = div_ceil(BxS, n_prgs)

    return TopkConfig(
        BxS=BxS,
        vocab_size=vocab_size,
        k=k,
        sorted=sorted,
        inp_dtype=inp_dtype,
        index_dtype=nl.uint32,
        n_prgs=n_prgs,
        prg_id=prg_id,
        per_lnc_BxS=per_lnc_BxS,
        out_shape=out_shape,
    )


def _valid_desnake_col_runs(k: int, k_cols: int, k_pad: int) -> Tuple[Tuple[int, int, int], ...]:
    """Contiguous runs mapping VALID de-snaked cv columns to compacted output columns.

    nisa.topk writes only snake positions [0, k), and cv column j holds snake-output
    position pos = (j // k_cols) + 16 * (j % k_cols), which is a real topk output iff
    pos < k. When k_pad > k (k % 16 != 0) the valid columns are NOT a contiguous prefix
    -- they are scattered among the k_pad columns, interleaved with the unwritten
    garbage columns. The unsorted path must COMPACT the valid columns to the
    contiguous output prefix [0, k). Each returned run is (src_col_start,
    dst_col_start, length): copy cv[:, src:src+length] -> out[:, dst:dst+length].

    For k % 16 == 0 (k_pad == k) the de-snake maps positions [0, k) onto cv columns
    [0, k), so the valid columns are already a contiguous prefix and this collapses to
    a SINGLE run (0, 0, k) (a full-width copy). Likewise for k_cols == 1 (k <= 16),
    where cv column j holds position j, so positions [0, k) occupy cv columns [0, k).

    All arguments are compile-time constants, so this resolves fully at trace time.
    Computed in a module-level helper (NOT inline in the @nki.jit body, where the
    parser frontend rejects the list-building expression as an "unsupported expression").
    """
    runs = []
    dst_col = 0
    in_run = False
    run_src_start = 0
    run_dst_start = 0
    run_len = 0
    for col_idx in range(k_pad):
        col_quotient, col_remainder = divmod(col_idx, k_cols)
        pos = col_quotient + PARTS_PER_GROUP * col_remainder
        if pos < k:
            if in_run:
                run_len += 1
            else:
                in_run = True
                run_src_start = col_idx
                run_dst_start = dst_col
                run_len = 1
            dst_col += 1
        elif in_run:
            runs.append((run_src_start, run_dst_start, run_len))
            in_run = False
            run_len = 0
    if in_run:
        runs.append((run_src_start, run_dst_start, run_len))
    return tuple(runs)


def _validate_topk(config: TopkConfig) -> None:
    """Validate that the requested shape satisfies every nisa.topk constraint."""
    n = config.vocab_size
    k = config.k
    kernel_assert(config.inp_dtype == nl.bfloat16, "topk requires bfloat16 input")
    kernel_assert(8 <= n < 65536, f"topk requires 8 <= vocab ({n}) < 65536")
    kernel_assert(1 <= k < 32768, f"topk requires 1 <= k ({k}) < 32768")
    kernel_assert(k <= n, f"topk requires k ({k}) <= vocab ({n})")
