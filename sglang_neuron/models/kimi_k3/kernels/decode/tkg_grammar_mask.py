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

"""Kimi-K3 verify tail: the static grammar mask of the verify rows (off unless the call passes its tensors).

A grammar is a token-level automaton in tables of fixed shapes, so a new grammar is new table data and not a new
program. Each rank holds the tables of P slots, each of at most S states, K token classes and M distinct masks:

- ``class_of`` ``[P * V, 1]`` INT32: the class ``c`` (``0 <= c < K``) of token ``t`` in slot ``p``, at ``p * V + t``.
- ``next`` ``[P * S * K, 2]`` INT32: the state value of state ``s`` of slot ``p`` is ``u = (p * S + s) * K``. At row
  ``u + c`` the table holds ``(u', m')``: the state value of the successor and the global mask row of the successor
  (``p * M + local mask``). A banned class holds ``(0, 0)``.
- ``mask_bias`` ``[P * M, V_s]`` BF16: the columns of this rank's vocab shard of each mask row, -0.0 for an allowed
  token and -inf for a banned token.

State value 0 is the INERT state (slot 0, state 0): mask row 0 allows every token (all -0.0), and every class of it
leads back to it. A row without a grammar stays INERT, so its logits keep their bits: ``x + (-0.0) == x`` in the fp32
add of the PSUM epilogue for every x, also -0.0.

The rows of one verify call (``rows_in`` ``[B, 4]`` INT32, one row for each sequence ``b``):
``(start_u, start_m, fold_first, class_offset)``, and ``ids`` ``[R, 1]`` INT32, the input token of each verify row
(row ``j * B + b`` is token ``j`` of sequence ``b``, the order of the stream):

- row 0: with ``fold_first`` 1, ``(u_0, m_0) = next[start_u + class(ids[b])]``. ``start_u`` is then the state before
  the row-0 token (the bonus token of the last step). With ``fold_first`` 0, ``(u_0, m_0) = (start_u, start_m)``: the
  state after the row-0 token, given by the host (a new sequence).
- row j >= 1: ``(u_j, m_j) = next[u_{j-1} + class(ids[j * B + b])]``, with ``class(t) = class_of[class_offset + t]``
  and ``class_offset = p * V`` for a sequence in slot p.

Verify row ``j`` of sequence ``b`` takes mask row ``m_j``. A draft that the grammar bans after row ``j - 1`` has -inf
there, so it has no target probability, and both the greedy prefix rule and the probabilistic rules reject it. Its
successor is INERT, so the rows after it allow every token and give no all -inf row. ``rows_out`` ``[B, Q]`` INT32
receives ``u_j``: the host checks the rows that it commits, and the next call starts each sequence from
``u_{a - 1}`` (``a`` accepted tokens) with ``fold_first`` 1.

The fold runs inside the ``lm_head`` tail, on the software DGE queue and a few small Vector Engine ops, while the
``lm_head`` weight stream fills the first vocab chunk. ``grammar_mask_rows_reference`` is the NumPy model of the fold.
"""

from dataclasses import dataclass

import nki.isa as nisa
import nki.language as nl
import numpy as np

from .utils.kernel_assert import kernel_assert

GRAMMAR_ROW_FIELDS = 4
"""Columns of ``rows_in``: ``(start_u, start_m, fold_first, class_offset)``."""
GRAMMAR_INDEX_LIMIT = 1 << 24
"""The table indices stay below 2**24, so the Vector Engine adds them exactly in fp32."""
INERT_STATE = 0
"""The state value of the INERT state (slot 0, state 0)."""
INERT_MASK = 0
"""The mask row of the INERT state (all tokens allowed)."""


@dataclass(frozen=True)
class GrammarMask:
    """The HBM tensors of the static grammar mask of one verify call (see the module docstring)."""

    ids: nl.NkiTensor
    rows_in: nl.NkiTensor
    class_of: nl.NkiTensor
    next: nl.NkiTensor
    mask_bias: nl.NkiTensor
    rows_out: nl.NkiTensor


def check_grammar_mask(grammar: GrammarMask, rows: int, queries: int, vocab_shard: int, dtype) -> None:
    """Assert the shapes and dtypes of the grammar tensors of a call with ``rows`` verify rows of ``queries`` tokens."""
    kernel_assert(queries >= 1 and rows % queries == 0, f"{rows} verify rows do not split into {queries} queries")
    seqs = rows // queries
    kernel_assert(
        tuple(grammar.ids.shape) == (rows, 1) and grammar.ids.dtype == nl.int32,
        f"grammar ids must be [{rows}, 1] INT32, got {tuple(grammar.ids.shape)} {grammar.ids.dtype}",
    )
    kernel_assert(
        tuple(grammar.rows_in.shape) == (seqs, GRAMMAR_ROW_FIELDS) and grammar.rows_in.dtype == nl.int32,
        f"grammar rows_in must be [{seqs}, {GRAMMAR_ROW_FIELDS}] INT32, got {tuple(grammar.rows_in.shape)}",
    )
    kernel_assert(
        tuple(grammar.rows_out.shape) == (seqs, queries) and grammar.rows_out.dtype == nl.int32,
        f"grammar rows_out must be [{seqs}, {queries}] INT32, got {tuple(grammar.rows_out.shape)}",
    )
    kernel_assert(
        len(grammar.class_of.shape) == 2 and grammar.class_of.shape[1] == 1 and grammar.class_of.dtype == nl.int32,
        f"grammar class_of must be [P * V, 1] INT32, got {tuple(grammar.class_of.shape)}",
    )
    kernel_assert(
        len(grammar.next.shape) == 2 and grammar.next.shape[1] == 2 and grammar.next.dtype == nl.int32,
        f"grammar next must be [P * S * K, 2] INT32, got {tuple(grammar.next.shape)}",
    )
    kernel_assert(
        grammar.class_of.shape[0] < GRAMMAR_INDEX_LIMIT and grammar.next.shape[0] < GRAMMAR_INDEX_LIMIT,
        f"the grammar tables must have fewer than {GRAMMAR_INDEX_LIMIT} rows",
    )
    kernel_assert(
        len(grammar.mask_bias.shape) == 2
        and grammar.mask_bias.shape[1] == vocab_shard
        and grammar.mask_bias.dtype == dtype,
        f"grammar mask_bias must be [P * M, {vocab_shard}] in the logits dtype, got {tuple(grammar.mask_bias.shape)}",
    )


def _gather_rows(table, index, rows: int, cols: int, dst, name: str, offset: int = 0):
    """Gather ``cols`` columns from ``offset`` of the rows ``index`` ``[rows, 1]`` INT32 of the HBM ``table``."""
    width = table.shape[1]
    return nisa.dma_copy(
        dst=dst,
        src=table.ap(pattern=[[width, rows], [1, cols]], offset=offset, vector_offset=index, indirect_dim=0),
        dge_mode=nisa.dge_mode.swdge,
        name=name,
    )


def grammar_mask_rows(grammar: GrammarMask, queries: int, tag: str):
    """Fold the grammar state along the verify rows; return the SBUF ``[R, 1]`` INT32 mask row of each row.

    The state values go to ``grammar.rows_out``. Every DMA runs on the software DGE queue, so none waits behind the
    ``lm_head`` weight stream of the hardware DGE queues.
    """
    rows = grammar.ids.shape[0]
    seqs = rows // queries
    i32 = nl.int32
    vector = nisa.engine.vector
    swdge = nisa.dge_mode.swdge

    rows_in = nl.ndarray((seqs, GRAMMAR_ROW_FIELDS), dtype=i32, buffer=nl.sbuf)
    nisa.dma_copy(dst=rows_in, src=grammar.rows_in, dge_mode=swdge, name=f"{tag}_grammar_rows_in")
    # The scalar operands of the Vector Engine are fp32; the values are below 2**24, so the copy is exact.
    rows_f = nl.ndarray((seqs, GRAMMAR_ROW_FIELDS), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=rows_f, src=rows_in, engine=vector)
    # Sequence b on partition b, token j on column j: row j * B + b of the stream-order ids.
    ids = nl.ndarray((seqs, queries), dtype=i32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=ids,
        src=grammar.ids.ap(pattern=[[1, seqs], [seqs, queries]], offset=0),
        dge_mode=swdge,
        name=f"{tag}_grammar_ids",
    )
    class_index = nl.ndarray((seqs, queries), dtype=i32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=class_index, data=ids, op0=nl.add, operand0=rows_f[0:seqs, 3:4], engine=vector)
    classes = nl.ndarray((seqs, queries), dtype=i32, buffer=nl.sbuf)
    for j in range(queries):
        _gather_rows(
            grammar.class_of,
            class_index[0:seqs, j : j + 1],
            seqs,
            1,
            classes[0:seqs, j : j + 1],
            f"{tag}_grammar_class_{j}",
        )

    # (u_j, m_j) of each row j: column 2 j holds u_j, column 2 j + 1 holds m_j.
    states = nl.ndarray((seqs, 2 * queries), dtype=i32, buffer=nl.sbuf)
    index = nl.ndarray((seqs, 1), dtype=i32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=index, data1=rows_in[0:seqs, 0:1], data2=classes[0:seqs, 0:1], op=nl.add, engine=vector)
    folded = nl.ndarray((seqs, 2), dtype=i32, buffer=nl.sbuf)
    _gather_rows(grammar.next, index, seqs, 2, folded, f"{tag}_grammar_next_0")
    # Row 0: start + fold_first * (folded - start), exact in fp32 for indices below 2**24.
    delta = nl.ndarray((seqs, 2), dtype=i32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=delta, data1=folded, data2=rows_in[0:seqs, 0:2], op=nl.subtract, engine=vector)
    nisa.scalar_tensor_tensor(
        dst=states[0:seqs, 0:2],
        data=delta,
        op0=nl.multiply,
        operand0=rows_f[0:seqs, 2:3],
        op1=nl.add,
        operand1=rows_in[0:seqs, 0:2],
    )
    for j in range(1, queries):
        step = nl.ndarray((seqs, 1), dtype=i32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=step,
            data1=states[0:seqs, 2 * (j - 1) : 2 * j - 1],
            data2=classes[0:seqs, j : j + 1],
            op=nl.add,
            engine=vector,
        )
        _gather_rows(grammar.next, step, seqs, 2, states[0:seqs, 2 * j : 2 * j + 2], f"{tag}_grammar_next_{j}")

    pairs = states.reshape((seqs, queries, 2))
    nisa.dma_copy(dst=grammar.rows_out, src=pairs[0:seqs, 0:queries, 0:1], dge_mode=swdge, name=f"{tag}_grammar_rows")
    # The mask rows in stream order (row j * B + b) go through HBM: an SBUF partition shuffle does not lower.
    mask_hbm = nl.ndarray((rows, 1), dtype=i32, buffer=nl.private_hbm, name=f"{tag}_grammar_mask_rows")
    nisa.dma_copy(
        dst=mask_hbm.ap(pattern=[[1, seqs], [seqs, queries]], offset=0),
        src=pairs[0:seqs, 0:queries, 1:2],
        dge_mode=swdge,
        name=f"{tag}_grammar_mask_rows_store",
    )
    mask_rows = nl.ndarray((rows, 1), dtype=i32, buffer=nl.sbuf)
    nisa.dma_copy(dst=mask_rows, src=mask_hbm, dge_mode=swdge, name=f"{tag}_grammar_mask_rows_load")
    return mask_rows


def grammar_mask_bias(grammar: GrammarMask, mask_rows, chunks, dtype, tag: str):
    """Return one SBUF ``[R, width]`` bias tile for each vocab chunk ``(c0, width)``: the mask row of each row."""
    rows = mask_rows.shape[0]
    tiles = []
    for c0, width in chunks:
        bias = nl.ndarray((rows, width), dtype=dtype, buffer=nl.sbuf)
        _gather_rows(
            grammar.mask_bias, mask_rows[0:rows, 0:1], rows, width, bias, f"{tag}_grammar_bias_{c0}", offset=c0
        )
        tiles.append(bias)
    return tiles


# ---------------------------------------------------------------- NumPy model


def grammar_mask_rows_reference(
    ids: np.ndarray, rows_in: np.ndarray, class_of: np.ndarray, next_table: np.ndarray, queries: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(rows_out [B, Q], mask_rows [R])`` INT32 of the fold, as the kernel computes them.

    ``ids`` ``[R]`` or ``[R, 1]`` is in stream order (row ``j * B + b``); ``rows_in``, ``class_of`` and
    ``next_table`` are the kernel inputs.

    Example:
        >>> class_of = np.zeros((4, 1), np.int32)
        >>> nxt = np.zeros((1, 2), np.int32)
        >>> grammar_mask_rows_reference(np.zeros(2, np.int32), np.zeros((1, 4), np.int32), class_of, nxt, 2)[0].tolist()
        [[0, 0]]
    """
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    rows_in = np.asarray(rows_in, dtype=np.int64)
    class_of = np.asarray(class_of, dtype=np.int64).reshape(-1)
    next_table = np.asarray(next_table, dtype=np.int64).reshape(-1, 2)
    seqs = rows_in.shape[0]
    assert ids.shape[0] == seqs * queries, (ids.shape, seqs, queries)  # noqa: S101
    tokens = ids.reshape(queries, seqs).T
    classes = class_of[tokens + rows_in[:, 3:4]]
    rows_out = np.zeros((seqs, queries), dtype=np.int64)
    masks = np.zeros((seqs, queries), dtype=np.int64)
    folded = next_table[rows_in[:, 0] + classes[:, 0]]
    fold = rows_in[:, 2]
    rows_out[:, 0] = rows_in[:, 0] + fold * (folded[:, 0] - rows_in[:, 0])
    masks[:, 0] = rows_in[:, 1] + fold * (folded[:, 1] - rows_in[:, 1])
    for j in range(1, queries):
        step = next_table[rows_out[:, j - 1] + classes[:, j]]
        rows_out[:, j], masks[:, j] = step[:, 0], step[:, 1]
    return rows_out.astype(np.int32), masks.T.reshape(-1).astype(np.int32)
