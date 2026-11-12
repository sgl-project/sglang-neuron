# SPDX-License-Identifier: Apache-2.0
"""Two-level RoPE tables: the cosine and sine at any position from two small tables.

WHY THIS EXISTS. The kernel reads the RoPE rows of the step's positions only, but
a full ``[max_position, rope / 2]`` float32 table pair costs 256 MiB on every rank
at a 1M context. Position ``p = hi * LO_ROWS + lo`` gives ``angle(p) = angle(hi *
LO_ROWS) + angle(lo)``, so two tables of about ``sqrt(max_position)`` rows each and
the angle-addition identities give the same rows from 512 KiB.

Both tables come from float64 angles, as the full table does. The addition runs in
float32 and adds about 1.4e-7 of error against float64 (the full float32 table has
3e-8). The kernel casts the rotated keys to bf16 or fp8, which is far coarser.

Example:
    >>> import torch
    >>> tables = split_rope_tables(rope_theta=8000000.0, half=32, max_position=4096)
    >>> cos, sin = rope_rows(tables, torch.tensor([0, 1025], dtype=torch.int64))
    >>> cos.shape, float(cos[0, 0])
    (torch.Size([2, 32]), 1.0)
"""

from __future__ import annotations

import numpy as np
import torch

#: Rows of the low table. 1024 makes both tables 1024 rows at a 1M context.
LO_ROWS = 1024

#: Buffer names, in the order `rope_rows` takes them.
TABLE_NAMES = ("rope_cos_hi", "rope_sin_hi", "rope_cos_lo", "rope_sin_lo")


def split_rope_tables(
    rope_theta: float, half: int, max_position: int
) -> dict[str, torch.Tensor]:
    """Return the four float32 tables, keyed by `TABLE_NAMES`, on CPU.

    Args:
        rope_theta: The RoPE base.
        half: Half the rotated width (the table width).
        max_position: Positions the tables must cover.

    Returns:
        ``rope_{cos,sin}_hi`` of ``ceil(max_position / LO_ROWS)`` rows at angles
        ``hi * LO_ROWS``, and ``rope_{cos,sin}_lo`` of `LO_ROWS` rows at angles ``lo``.
    """
    inv = rope_theta ** (-np.arange(half, dtype=np.float64) / half)
    hi_rows = -(-int(max_position) // LO_ROWS)
    hi = (np.arange(hi_rows, dtype=np.float64) * LO_ROWS).reshape(-1, 1) * inv
    lo = np.arange(LO_ROWS, dtype=np.float64).reshape(-1, 1) * inv
    angles = {"hi": hi, "lo": lo}
    return {
        f"rope_{fn.__name__}_{part}": torch.from_numpy(fn(angle).astype(np.float32))
        for part, angle in angles.items()
        for fn in (np.cos, np.sin)
    }


def rope_rows(
    tables: dict[str, torch.Tensor], index: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the ``[len(index), half]`` float32 cosine and sine rows at `index`.

    Torch operations only, so the rows are built in the traced graph on device.

    Args:
        tables: The `split_rope_tables` tensors (or the model buffers of the same names).
        index: int64 positions, already clamped to the covered range.
    """
    cos_hi, sin_hi, cos_lo, sin_lo = (tables[name] for name in TABLE_NAMES)
    hi = torch.div(index, LO_ROWS, rounding_mode="floor")
    lo = index - hi * LO_ROWS
    ch, sh = cos_hi.index_select(0, hi), sin_hi.index_select(0, hi)
    cl, sl = cos_lo.index_select(0, lo), sin_lo.index_select(0, lo)
    return ch * cl - sh * sl, sh * cl + ch * sl
