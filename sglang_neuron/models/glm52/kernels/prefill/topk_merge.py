# SPDX-License-Identifier: Apache-2.0
"""The carry-index top-k merge of the chunked indexer top-k.

The chunked top-k leaves ``n_chunks`` sets of ``k`` candidates per query, as a 16-partition snake per
query: values ``cand_v`` bf16 and global positions ``cand_p`` fp32 in the SAME blocked layout.
``merge_batch`` ranks them in one select that carries each candidate's position through, so the winners
come out as positions directly (no one-hot position lookup on the PE array).

The select runs in released NKI instructions (``topk_select.emit_select_nki``), in one level over every candidate.
"""

import nki.language as nl

from .utils.kernel_assert import kernel_assert

_SNAKE = 16
_P_MAX = 128
_CALLS = [0]


def merge_batch(sbm, cand_v, cand_p, n_cand_cols: int, k: int, write_rows, block_bases, block_widths) -> None:
    """Merge one FULL batch of 8 queries and hand its ``[128, k/16]`` int32 positions to ``write_rows``.

    The select runs in released NKI instructions, one level, on the candidate tiles themselves
    (``topk_select.emit_select_nki``: up to 65,535 candidates in one ``nisa.topk``).

    Args:
        cand_v: ``[128, n_cand_cols]`` bf16 snake values, query ``j`` on partitions ``16j..16j+15``.
        cand_p: ``[128, n_cand_cols]`` fp32 global positions, same layout.
        n_cand_cols: candidate columns per partition; one query has ``16 * n_cand_cols`` candidates.
        k: the top-k.
        write_rows: called with the ``[128, k/16]`` int32 SBUF tile; partition ``16j + q`` holds
            ``k/16`` of query ``j``'s positions, so a flat ``[8, k]`` HBM block reshaped to
            ``[128, k/16]`` is the matching destination.
        block_bases, block_widths: each chunk's first position and width (one chunk per ``k/16`` columns), which the
            NKI select needs to decode a candidate's chunk-relative payload.
    """
    k_cols = k // _SNAKE
    kernel_assert(k_cols % 2 == 0, f"the top-k merge needs k a multiple of 32, got {k}")
    n_cand = _SNAKE * n_cand_cols
    kernel_assert(k <= n_cand, f"the top-k merge ranks {n_cand} candidates, fewer than k={k}")

    # alloc_stack names must be unique across the trace; an unrolled caller traces this per batch.
    t = _CALLS[0]
    _CALLS[0] += 1
    from sglang_neuron.models.glm52.kernels import topk_select

    n_blk = n_cand_cols // k_cols
    sbm.open_scope(name="nki_merge")
    out_i = sbm.alloc_stack((_P_MAX, k_cols), nl.int32, name=f"nm_out_i{t}")
    names = [0]

    def _alloc(shape, dtype, name):
        names[0] += 1
        return sbm.alloc_stack(shape, dtype, name=f"{name}_{t}_{names[0]}", align=64)

    topk_select.emit_select_nki(cand_v, cand_p, out_i, n_cand_cols, k, list(block_bases[:n_blk]),
                                list(block_widths[:n_blk]), alloc=_alloc)
    write_rows(out_i)
    sbm.close_scope()
