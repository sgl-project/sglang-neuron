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

"""The GLM-5.2 MTP draft chain in one kernel: ``num_steps`` draft steps of layer 78, with IndexShare and KVShare.

The MTP layer is shared across draft steps, and two rules make the later steps cheaper than the first:

* IndexShare: only step 0 runs the indexer and writes the indexer key cache; later steps reuse its selection.
* KVShare: a later step attends only the keys step 0 attended (the selected prior keys plus step 0's own rows), not
  its own, so it writes no K/V (``attend_new_keys=False``).

Nothing between steps goes through the host. The draft token's embedding is looked up in the vocabulary-sharded table
on the device, the next step's ``previous_hidden`` is ``shared_head.norm`` of this step's residual, and the per-step
query positions arrive as stacked RoPE tables.

Step 0 is layer 78 as a full decoder layer (prologue, attention with its indexer and caches, the FP8 MoE and the
head), sharded like the giga kernel except ``eh_proj``, which is column-sharded over ``eh_proj_replica_group``. For
later steps the host passes all-negative cache slot mappings, the step's query positions, and ``shared_extra_keys``,
the positions of step 0's own rows, which the kernel splices into step 0's selection.

After step 0's attention the chain runs on the draft row of each sequence only (``live_rows``), held by every rank:
the MoE, the greedy head, the next embedding, and from step 1 on the prologue and attention exit too.
"""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from ..attention.mla.mla_decode_constants import F_MAX, P_MAX
from ..utils.kernel_assert import kernel_assert
from ..utils.row_layout import spread_rows_shape
from ..utils.trace_names import unique_name
from ..attention.indexer.indexer_block_rows import indexer_block_rows
from .attention.indexer.indexer_constants import TOPK_MAX_N, indexer_cache_is_blocked, indexer_cache_keys
from .attention.mla.mla_proj_primitives import load_gamma_broadcast
from .prologue import eh_proj_columns, eh_proj_row_tiles, mtp_prologue_norms, mtp_prologue_row_norm
from .greedy_tail import glm52_mtp_greedy_tail
from .utils.cc_dst_ring import close_cc_dst_ring, open_cc_dst_ring
from .utils.cc_pin import close_pins, open_pins, pin
from .utils.sbuf_collectives import (
    all_gather_columns_hbm,
    all_gather_columns_sbuf,
    all_reduce_sbuf,
    load_gathered_columns,
)
from .attention_layer import (
    build_shared_prior_index,
    glm52_dsa_attention_decode_layer_kernel,
    mtp_idx_query_tokens,
    mtp_idx_residency_mask,
    select_rows,
)
from .moe_layer_bf16 import glm52_moe_rows_bf16, group_rows_index

_ERROR_PREFIX = "[GLM-5.2 MTP Draft Chain]"

GK_OPTIONS = ("idx_fp8",)
"""The 1M HBM cuts of the target that reach this chain and that it serves: ``idx_fp8`` (the draft's indexer cache
takes the target's key dtype, so a uint8 cache selects the fp8 score path of the forked indexer). The chain does
not read this value: the plugin's ``models/glm52/decode/kernel_limits.py`` parses it from this source (a tuple of str
literals) and refuses fp8 keys for the rows chain unless it is declared."""

# Collective operands at LNC=1 must be private_hbm (the runtime rejects shared_hbm sources).
_COLLECTIVE_HBM = nl.private_hbm

EMBED_PAD_ROWS = 2
"""The vocab-sharded embedding table carries one all-zero row BEFORE and one AFTER this rank's
``V_s`` rows, so a token outside the shard clamps onto a zero row and the AllReduce sums exactly one
non-zero contribution per token."""

_CC_DST_RING_DEPTH = 2
"""The draft steps' attention collective destinations (latent and query AllGathers, context all_to_all) rotate
through this many kernel-lifetime buffers per site from step 1 on (``utils/cc_dst_ring.py``), so no other
tensor shares their addresses."""

_KV_FREE = 2
"""The draft steps (KVShare, all-negative slot mappings) run the attention without its dead K/V path and its masked
sibling key tile (``kv_free`` of the attention layer); the shared selection's keys are gathered right after the draft
down-projection (``early_keys``)."""

_ENORM_BLOCKS = 3
"""Column blocks of the rows prologue's ``enorm`` scaling, each followed by its k-tiles' transposes and their
eviction, so the matmuls start after the first block instead of the whole scaling (elementwise: bit-identical)."""

_EXIT_ADD_BLOCKS = 8
"""The rows steps' residual add after the attention loads the exit AllReduce's HBM rows and adds them in this many
column blocks, so the adds overlap the load (elementwise: the same bytes)."""

_MOE_AR_BLOCKS = 8
"""The rows MoE leaves its AllReduce result in HBM, and the residual add after it loads and adds it in this many
column blocks, so the head's blocked norm starts behind the first block (elementwise: the same bytes)."""


def _residual_add(h: nl.NkiTensor, delta: nl.NkiTensor, tag: str) -> nl.NkiTensor:
    """``h + delta``, rounded once to ``h``'s dtype: the decode giga kernel's residual add, in its row-spread form.

    The add runs on the partition-spread view of the rows (``spread_rows_shape``, ``[128, 96]`` at the
    production shape): same bytes, same per-element result, 96 elements per partition instead of 6144.

    Args:
        h (nl.NkiTensor): [T_shard, H] the residual stream, in HBM.
        delta (nl.NkiTensor): [T_shard, H] the block's contribution in HBM, or an SBUF tile of the spread
            shape holding the same elements.
        tag (str): Name stem of the output buffer.

    Returns:
        nl.NkiTensor: the sum, [T_shard, H] in HBM in ``h``'s dtype.
    """
    T_shard, H = h.shape
    sb_shape = spread_rows_shape(T_shard, H) or (T_shard, H)
    h_sb = nl.ndarray(sb_shape, dtype=h.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=h_sb, src=h.reshape(sb_shape))
    d_sb = delta
    if delta.buffer != nl.sbuf:
        d_sb = nl.ndarray(sb_shape, dtype=h.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=d_sb, src=delta.reshape(sb_shape))

    sum_sb = nl.ndarray(sb_shape, dtype=h.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sum_sb, data1=h_sb, data2=d_sb, op=nl.add)

    out = nl.ndarray((T_shard, H), dtype=h.dtype, buffer=_COLLECTIVE_HBM, name=unique_name(f"residual_{tag}"))
    nisa.dma_copy(dst=out.reshape(sb_shape), src=sum_sb)
    return out


def _hnorm_early_fits(n_seq: int, H: int) -> bool:
    """Whether the next prologue's ``hnorm`` can run early (:func:`_prologue_hidden_tiles`) at ``n_seq`` rows: every
    half and every ``enorm`` block must start on a whole group of the ``F_MAX / n_seq`` k-tiles one PSUM eviction
    holds. Otherwise the prologue norms both halves itself."""
    per_psum = F_MAX // n_seq
    width = H // _ENORM_BLOCKS
    return H % (_ENORM_BLOCKS * P_MAX) == 0 and (H // P_MAX) % per_psum == 0 and (width // P_MAX) % per_psum == 0


def _shared_head_norm(hidden: nl.NkiTensor, gamma: nl.NkiTensor, eps: float, head: dict) -> nl.NkiTensor:
    """``shared_head.norm(hidden)`` on the rows: the next draft step's ``previous_hidden``, on chip.

    One bf16 rounding (fp32 reduce and fp32 gamma, written back into the bf16 tile), the same realization
    as the layer norms in the MoE block. The scaling is ordered after the head's AllGather source is written
    (``head``, the greedy tail's ``handles_out``), so it runs in the AllGather's shadow.

    Args:
        hidden (nl.NkiTensor): [n_seq, H] bf16 the step's pre-norm residual rows, in SBUF.
        gamma (nl.NkiTensor): [1, H] ``shared_head.norm.weight``.
        eps (float): RMSNorm epsilon.
        head (dict): The greedy tail's ``handles_out``.

    Returns:
        nl.NkiTensor: [n_seq, H] SBUF in ``hidden``'s dtype, whose next consumer is the prologue norm.
    """
    T, H = hidden.shape
    out_sb = nl.ndarray((T, H), dtype=hidden.dtype, buffer=nl.sbuf)
    _rmsnorm_into_after(out_sb, hidden, load_gamma_broadcast(gamma, T), T, eps, scale_after=head["pair_copy"])
    return out_sb


def _rmsnorm_into_after(
    dst_sb: nl.NkiTensor, x_sb: nl.NkiTensor, gamma_sb: nl.NkiTensor, P: int, eps: float, scale_after=None
) -> None:
    """``mla_proj_primitives.rmsnorm_into``'s three instructions, the scaling ordered after ``scale_after``."""
    F = x_sb.shape[1]
    squares_discard = nl.ndarray((P_MAX, F), dtype=nl.float32, buffer=nl.sbuf)
    inv_rms = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation_reduce(
        dst=squares_discard[0:P, 0:F],
        op=nl.square,
        data=x_sb[0:P, 0:F],
        reduce_op=nl.add,
        reduce_res=inv_rms[0:P, 0:1],
    )
    nisa.activation(inv_rms[0:P, 0:1], op=nl.rsqrt, data=inv_rms[0:P, 0:1], scale=1.0 / F, bias=eps)
    scale = nisa.scalar_tensor_tensor(
        dst=dst_sb[0:P, 0:F],
        data=x_sb[0:P, 0:F],
        op0=nl.multiply,
        operand0=inv_rms[0:P, 0:1],
        op1=nl.multiply,
        operand1=gamma_sb[0:P, 0:F],
    )
    if scale_after is not None:
        # scalar_tensor_tensor returns the bare traced op, not an NkiInstruction handle.
        nisa.NkiInstruction(scale).depends_on(scale_after)


def _vocab_parallel_embedding(
    embed_shard_pad: nl.NkiTensor,
    ep_replica_group: ReplicaGroup,
    out_dtype,
    idx_sb: nl.NkiTensor,
    n: int,
    handles_out: dict,
) -> nl.NkiTensor:
    """``embed_tokens[tokens]`` of the ``n`` draft tokens, from a vocab-sharded table: the next step's embedding rows.

    The table is sharded by vocabulary like the lm_head, so the row of a token lives on ONE rank. Every rank reads the
    rows it holds into an ``[n, H]`` tile whose other rows are zero, and one ``all_reduce(add)`` over the group hands
    every rank all of them -- the Megatron ``VocabParallelEmbedding`` masked-lookup-plus-reduce, with the mask folded
    into the index: ``idx = clamp(token - vocab_base + 1, 0, V_s + 1)`` lands outside this shard on a padding row of
    zeros. The greedy tail computes that index from its winning key (``row_index_out``).

    Args:
        embed_shard_pad (nl.NkiTensor): [V_s + 2, H] bf16, rows ``1..V_s`` this rank's vocabulary rows
            ``[rank * V_s, (rank + 1) * V_s)`` of ``embed_tokens.weight``, rows 0 and ``V_s + 1`` zero.
        ep_replica_group (ReplicaGroup): all ranks in rank order.
        out_dtype: The embeddings' dtype (bf16).
        idx_sb (nl.NkiTensor): [P_MAX, 1] int32 SBUF, the table row of each token (rows ``[0, n)``).
        n (int): Tokens, at most ``P_MAX``.
        handles_out (dict): Receives ``"emb_gather"`` and ``"all_reduce"``, the lookup's and the AllReduce's
            instructions.

    Returns:
        nl.NkiTensor: [n, H] SBUF, the same on every rank (``lm_head/sbuf_collectives.py``).
    """
    H = embed_shard_pad.shape[1]
    kernel_assert(n <= P_MAX, f"{_ERROR_PREFIX} the embedding lookup takes at most {P_MAX} rows, got {n}")
    emb_sb = nl.ndarray((n, H), dtype=out_dtype, buffer=nl.sbuf)
    handles_out["emb_gather"] = nisa.dma_copy(
        dst=emb_sb[0:n, 0:H],
        src=embed_shard_pad.ap(
            pattern=[[H, n], [1, H]],
            offset=0,
            vector_offset=idx_sb.ap(pattern=[[1, n], [1, 1]], offset=0),
            indirect_dim=0,
        ),
    )
    # Exactly one non-zero contribution per row, so the sum is exact.
    return all_reduce_sbuf(emb_sb, ep_replica_group, nl.add, "mtp_emb_rows_sum", handles_out=handles_out)


def _eh_proj_column_sharded(
    embeds: nl.NkiTensor,
    prev_hidden: nl.NkiTensor,
    positions: nl.NkiTensor,
    enorm_w: nl.NkiTensor,
    hnorm_w: nl.NkiTensor,
    eh_proj_cols: nl.NkiTensor,
    replica_group: ReplicaGroup,
    group_size: int,
    eps: float,
    stores_out: list = None,
) -> nl.NkiTensor:
    """The prologue with ``eh_proj`` sharded by output column over ``group_size`` consecutive ranks.

    Replicated, the projection reads all 151 MB of ``eh_proj`` on every rank in every step, for 2 tokens.
    Here group position ``p`` holds columns ``[N * p, N * (p + 1))``, ``N = H / group_size``, and computes
    them for all the group's tokens:

      1. ``mtp_prologue_norms``: this rank's normed ``concat(e, h)`` rows, ``[T_shard, 2H]``.
      2. AllGather over the group: ``[T_shard * group_size, 2H]``, rows in group-position order.
      3. ``eh_proj_columns``: ``[T_shard * group_size, N]``, this rank's columns of every row.
      4. ``all_to_all`` over the group: block ``d`` (the rows of position ``d``) goes to position ``d``, and
         block ``s`` arrives from position ``s`` with columns ``[N * s, N * (s + 1))`` of THIS rank's rows.
      5. One HBM -> HBM DMA puts block ``s`` at those columns: ``[T_shard, H]``.

    No step uses the rank's index, so the trace stays rank-invariant: the column choice is in the weight
    data and the row choice is in the group positions. Each output element is accumulated over the whole
    ``2H`` contraction on one rank (see ``eh_proj_columns``). The group must be
    contiguous, because the runtime's ``all_to_all`` aborts on strided groups.

    Args:
        embeds (nl.NkiTensor): [T_shard, H] this rank's draft-token embeddings.
        prev_hidden (nl.NkiTensor): [T_shard, H] this rank's ``previous_hidden``.
        positions (nl.NkiTensor): [T_shard, 1] int32 absolute positions, for the position-0 mask.
        enorm_w, hnorm_w (nl.NkiTensor): [1, H] the two norm gammas.
        eh_proj_cols (nl.NkiTensor): [2H, N] this rank's columns of ``eh_proj.weight`` TRANSPOSED.
        replica_group (ReplicaGroup): ``group_size`` consecutive ranks, in rank order.
        group_size (int): Ranks in the group, ``H / N``.
        eps (float): RMSNorm epsilon.
        stores_out (list): Optional; receives the stores of the normed rows (the AllGather's source).

    Returns:
        nl.NkiTensor: [T_shard, H] private HBM, the MTP block's ``hidden_states`` for this rank's slots.
    """
    T_shard, H = embeds.shape
    N = eh_proj_cols.shape[1]
    T_all = T_shard * group_size
    dtype = embeds.dtype

    normed = nl.ndarray((T_shard, 2 * H), dtype=dtype, buffer=_COLLECTIVE_HBM, name=unique_name("mtp_eh_normed"))
    mtp_prologue_norms(embeds, prev_hidden, positions, enorm_w, hnorm_w, normed, eps, stores_out=stores_out)
    normed_all = nl.ndarray((T_all, 2 * H), dtype=dtype, buffer=_COLLECTIVE_HBM, name=unique_name("mtp_eh_normed_all"))
    pin(normed), pin(normed_all)  # long context only; see utils/cc_pin.py
    ncc.all_gather(srcs=[normed], dsts=[normed_all], replica_group=replica_group, collective_dim=0)

    cols = nl.ndarray((T_all, N), dtype=dtype, buffer=_COLLECTIVE_HBM, name=unique_name("mtp_eh_cols"))
    eh_proj_columns(normed_all, eh_proj_cols, cols)
    recv = nl.ndarray((T_all, N), dtype=dtype, buffer=_COLLECTIVE_HBM, name=unique_name("mtp_eh_recv"))
    pin(cols), pin(recv)  # long context only; see utils/cc_pin.py
    ncc.all_to_all(srcs=[cols], dsts=[recv], replica_group=replica_group, collective_dim=0)

    # Received row (s, t) is token t's columns [N * s, N * (s + 1)): the layout _all_gather_columns relays.
    x_shard = nl.ndarray((T_shard, H), dtype=dtype, buffer=_COLLECTIVE_HBM, name=unique_name("mtp_eh_x"))
    nisa.dma_copy(
        dst=x_shard.reshape((T_shard, group_size, N)),
        src=recv.ap(pattern=[[N, T_shard], [T_shard * N, group_size], [1, N]], offset=0),
    )
    return x_shard


def _gated_copy(src: nl.NkiTensor, after: list, name: str) -> nl.NkiTensor:
    """A private-HBM copy of ``src`` that waits for every instruction in ``after``: whatever reads the copy instead of
    ``src`` runs after them. The same bytes."""
    dst = nl.ndarray(src.shape, dtype=src.dtype, buffer=_COLLECTIVE_HBM, name=unique_name(name))
    copy = nisa.dma_copy(dst=dst, src=src)
    if after:
        copy.depends_on(*after)
    return dst


def _slot_row_tables(
    rank_id: nl.NkiTensor, live_rows: nl.NkiTensor, T_shard: int, spec: int
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """The map between this rank's token slots and the one-row-per-sequence ("rows") layout, on device.

    Slot ``j`` of rank ``r`` is global slot ``g = T_shard * r + j`` of sequence ``g // spec``, and it is that
    sequence's draft row iff ``live_rows[g // spec] == g``. Built from ``rank_id`` because one NEFF serves
    every rank: a Python rank would fold to rank 0's value.

    Returns:
        tuple: ``(seq_i, draft_dst)``, each [P_MAX, 1] SBUF with rows ``[0, T_shard)``: ``seq_i`` int32 the
        sequence of each slot; ``draft_dst`` uint32 that sequence for a draft-row slot and an out-of-range
        row (``n_seq``) for every other slot, so an indirect write skips it.
    """
    n_seq = live_rows.shape[0]
    kernel_assert(spec & (spec - 1) == 0, f"{_ERROR_PREFIX} the rows layout needs spec a power of two, got {spec}")
    rank_i = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
    # uint32 -> int32 has the same bits for any rank id, so the DMA's reinterpretation is exact.
    nisa.dma_copy(dst=rank_i[0:T_shard, 0:1], src=rank_id[0:1, 0:1].view(nl.int32).broadcast(0, T_shard))
    rank_f = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=rank_f[0:T_shard, 0:1], src=rank_i[0:T_shard, 0:1])
    lane_i = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(lane_i[0:T_shard, 0:1], pattern=[[0, 1]], offset=0, channel_multiplier=1)
    g_f = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=g_f[0:T_shard, 0:1], src=lane_i[0:T_shard, 0:1])
    nisa.scalar_tensor_tensor(
        dst=g_f[0:T_shard, 0:1],
        data=rank_f[0:T_shard, 0:1],
        op0=nl.multiply,
        operand0=float(T_shard),
        op1=nl.add,
        operand1=g_f[0:T_shard, 0:1],
    )
    g_i = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=g_i[0:T_shard, 0:1], src=g_f[0:T_shard, 0:1])
    seq_i = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=seq_i[0:T_shard, 0:1], data=g_i[0:T_shard, 0:1], op0=nl.right_shift, operand0=spec.bit_length() - 1
    )

    # live_rows[seq] -- the global slot of each slot's sequence's draft row -- and the draft test.
    live_i = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=live_i[0:T_shard, 0:1],
        src=live_rows.ap(
            pattern=[[1, T_shard], [1, 1]],
            offset=0,
            vector_offset=seq_i.ap(pattern=[[1, T_shard], [1, 1]], offset=0),
            indirect_dim=0,
        ),
    )
    live_f = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=live_f[0:T_shard, 0:1], src=live_i[0:T_shard, 0:1])
    is_draft = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=is_draft[0:T_shard, 0:1], data1=live_f[0:T_shard, 0:1], data2=g_f[0:T_shard, 0:1], op=nl.equal
    )

    # draft ? seq : n_seq, as n_seq + is_draft * (seq - n_seq), exact in fp32.
    dst_f = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=dst_f[0:T_shard, 0:1], src=seq_i[0:T_shard, 0:1])
    nisa.tensor_scalar(dst=dst_f[0:T_shard, 0:1], data=dst_f[0:T_shard, 0:1], op0=nl.subtract, operand0=float(n_seq))
    nisa.tensor_scalar(
        dst=dst_f[0:T_shard, 0:1],
        data=dst_f[0:T_shard, 0:1],
        op0=nl.multiply,
        operand0=is_draft[0:T_shard, 0:1],
        op1=nl.add,
        operand1=float(n_seq),
    )
    draft_dst = nl.ndarray((P_MAX, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=draft_dst[0:T_shard, 0:1], src=dst_f[0:T_shard, 0:1])
    return seq_i, draft_dst


def _rows_from_slots(
    h_shard: nl.NkiTensor, draft_dst: nl.NkiTensor, n_seq: int, ep_replica_group: ReplicaGroup
) -> nl.NkiTensor:
    """[n_seq, H] the draft row of every sequence, the same on every rank, from the token-sharded rows.

    Each rank writes its draft-row slot (at most ``T_shard`` of them) into a zero table at the row of its
    sequence -- the other slots are skipped as out of range -- and one ``all_reduce(add)`` completes it.
    Every row has exactly one non-zero contribution, so the sum is exact.
    """
    T_shard, H = h_shard.shape
    table = nl.ndarray((n_seq, H), dtype=h_shard.dtype, buffer=_COLLECTIVE_HBM, name=unique_name("mtp_rows_in"))
    zero_sb = nl.ndarray((P_MAX, H), dtype=h_shard.dtype, buffer=nl.sbuf)
    nisa.memset(dst=zero_sb, value=0.0)
    nisa.dma_copy(dst=table[0:n_seq, 0:H], src=zero_sb[0:n_seq, 0:H])
    h_sb = nl.ndarray((P_MAX, H), dtype=h_shard.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=h_sb[0:T_shard, 0:H], src=h_shard[0:T_shard, 0:H])
    nisa.dma_copy(
        dst=table.ap(pattern=[[H, T_shard], [1, H]], offset=0, vector_offset=draft_dst[0:T_shard, 0:1], indirect_dim=0),
        src=h_sb[0:T_shard, 0:H],
        oob_mode=nisa.oob_mode.skip,
    )
    rows = nl.ndarray((n_seq, H), dtype=h_shard.dtype, buffer=_COLLECTIVE_HBM, name=unique_name("mtp_rows"))
    pin(table), pin(rows)  # long context only; see utils/cc_pin.py
    ncc.all_reduce(srcs=[table], dsts=[rows], replica_group=ep_replica_group, op=nl.add)
    return rows


def _slots_from_rows(rows: nl.NkiTensor, seq_i: nl.NkiTensor, T_shard: int, name: str) -> nl.NkiTensor:
    """[T_shard, H] private HBM: each of this rank's slots takes its sequence's row. No collective."""
    n_seq, H = rows.shape
    out_sb = nl.ndarray((P_MAX, H), dtype=rows.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=out_sb[0:T_shard, 0:H],
        src=rows.ap(
            pattern=[[H, T_shard], [1, H]],
            offset=0,
            vector_offset=seq_i.ap(pattern=[[1, T_shard], [1, 1]], offset=0),
            indirect_dim=0,
        ),
    )
    out = nl.ndarray((T_shard, H), dtype=rows.dtype, buffer=_COLLECTIVE_HBM, name=unique_name(name))
    nisa.dma_copy(dst=out[0:T_shard, 0:H], src=out_sb[0:T_shard, 0:H])
    return out


def _rows_index_hbm(rank_id: nl.NkiTensor, group_size: int, rows_per_group: int, name: str) -> nl.NkiTensor:
    """[rows_per_group, 1] int32 HBM: this rank's group's rows of the rows layout (``group_rows_index``)."""
    rows_sb = group_rows_index(rank_id, group_size, rows_per_group)
    rows = nl.ndarray((rows_per_group, 1), dtype=nl.int32, buffer=_COLLECTIVE_HBM, name=unique_name(name))
    nisa.dma_copy(dst=rows[0:rows_per_group, 0:1], src=rows_sb[0:rows_per_group, 0:1])
    return rows


def _prologue_rows(
    emb_rows: nl.NkiTensor,
    prev_rows: nl.NkiTensor,
    enorm_w: nl.NkiTensor,
    hnorm_w: nl.NkiTensor,
    eh_proj_cols: nl.NkiTensor,
    replica_group: ReplicaGroup,
    group_size: int,
    eps: float,
    head: dict = None,
    hidden_tiles: tuple = None,
    gathered: bool = False,
) -> nl.NkiTensor:
    """The prologue in the rows layout: [n_seq, H] ``hidden_states`` of every sequence, on every rank, in SBUF.

    Every rank holds all ``n_seq`` rows, so it norms them and multiplies them by its ``eh_proj`` column shard
    with no input gather, and one AllGather of the column slices over the shard group assembles the rows --
    against :func:`_eh_proj_column_sharded`'s AllGather of the normed rows and ``all_to_all`` back. No
    position-0 mask: every draft step after the first is at least 2 positions past the target's.

    The inputs arrive in SBUF (the embedding AllReduce, the shared-head norm), the normed halves feed the
    projection's transposes in place, and the projection evicts into the gather's source. The gather
    (``lm_head/sbuf_collectives.py``) lands position ``p``'s columns at ``[N p, N (p + 1))``, through private HBM.

    ``head``: optionally the previous step's head handles: the gamma loads are then issued after its embedding
    AllReduce trigger, so their broadcast packets do not hold the DMA engines while the head's band moves.
    ``hidden_tiles``: optionally the ``(row_tiles, k0)`` of :func:`_prologue_hidden_tiles` (``prev_rows`` is then
    unused): only ``enorm`` and its transposes remain here, in ``_ENORM_BLOCKS`` column blocks.
    ``gathered``: return the HBM AllGather's rank-major ``[group_size * n_seq, N]`` rows instead, which the attention
    picks its row from in place (:func:`all_gather_columns_hbm`).
    """
    gather = all_gather_columns_hbm if gathered else all_gather_columns_sbuf
    n_seq, H = emb_rows.shape
    N = eh_proj_cols.shape[1]
    gamma_after = None if head is None else [head["all_reduce"]]
    if hidden_tiles is not None:
        row_tiles, k0 = hidden_tiles
        width = H // _ENORM_BLOCKS
        kernel_assert(
            H % _ENORM_BLOCKS == 0 and (width // P_MAX) % (F_MAX // n_seq) == 0,
            f"{_ERROR_PREFIX} {_ENORM_BLOCKS} enorm blocks must cut H={H} into whole transpose groups",
        )

        def transpose_block(b, embeds_sb):
            evictions = []
            eh_proj_row_tiles(
                [embeds_sb[0:P_MAX, b * width : (b + 1) * width]],
                n_seq,
                2 * k0,
                k0=b * width // P_MAX,
                row_tiles=row_tiles,
                evict_handles_out=evictions,
            )
            return evictions[-1]

        mtp_prologue_row_norm(
            emb_rows,
            enorm_w,
            eps,
            gamma_after=gamma_after,
            scale_blocks=_ENORM_BLOCKS,
            between_blocks=transpose_block,
        )
        cols_sb = nl.ndarray((n_seq, N), dtype=emb_rows.dtype, buffer=nl.sbuf)
        eh_proj_columns([], eh_proj_cols, cols_sb, pretransposed=(row_tiles, 0))
        return gather(cols_sb, replica_group, group_size, "mtp_x_rows")
    halves = mtp_prologue_norms(emb_rows, prev_rows, None, enorm_w, hnorm_w, None, eps, gamma_after=gamma_after)
    cols_sb = nl.ndarray((n_seq, N), dtype=emb_rows.dtype, buffer=nl.sbuf)
    eh_proj_columns(halves, eh_proj_cols, cols_sb)
    return gather(cols_sb, replica_group, group_size, "mtp_x_rows")


def _prologue_hidden_tiles(prev_rows: nl.NkiTensor, hnorm_w: nl.NkiTensor, eps: float, head: dict = None) -> tuple:
    """The next rows prologue's ``hnorm(prev_rows)``, transposed into the second half of its ``eh_proj`` k-tiles.

    Traced before the step's embedding lookup, so none of its tiles takes the lookup's or the AllReduce's SBUF
    (a tile first used after the AllReduce trigger may, and then waits for the AllReduce to finish). With ``head``
    its gamma load is issued after the head AllGather trigger.

    Returns:
        tuple: ``(row_tiles, k0)`` for :func:`_prologue_rows`' ``hidden_tiles``: k-tiles ``[k0, 2 k0)`` are filled.
    """
    n_seq, H = prev_rows.shape
    gamma_after = None if head is None else [head["all_gather"]]
    hidden_sb = mtp_prologue_row_norm(prev_rows, hnorm_w, eps, gamma_after=gamma_after)
    k0 = H // P_MAX
    return eh_proj_row_tiles([hidden_sb], n_seq, 2 * k0, k0=k0), k0


def _rows_add(a_sb: nl.NkiTensor, b: nl.NkiTensor, blocks: int) -> nl.NkiTensor:
    """``a + b`` on ``[n_seq, H]`` rows in the residual dtype, SBUF out; ``b`` may be in HBM (it is then loaded).

    The giga stack's ``_residual_add`` arithmetic (one bf16 add), without its HBM copy of the sum: in the rows
    layout every consumer of the sum reads it on chip. ``b`` is loaded and added in ``blocks`` column blocks (the
    same Vector add per element), so each block's add starts behind its own load.
    """
    n, H = a_sb.shape
    kernel_assert(H % blocks == 0, f"{_ERROR_PREFIX} {blocks} residual-add blocks must divide H={H}")
    w = H // blocks
    b_sb = b if b.buffer == nl.sbuf else nl.ndarray((n, H), dtype=a_sb.dtype, buffer=nl.sbuf)
    sum_sb = nl.ndarray((n, H), dtype=a_sb.dtype, buffer=nl.sbuf)
    for k in range(blocks):
        cols = slice(k * w, (k + 1) * w)
        if b.buffer != nl.sbuf:
            nisa.dma_copy(dst=b_sb[0:n, cols], src=b[0:n, cols])
        nisa.tensor_tensor(
            dst=sum_sb[0:n, cols],
            data1=a_sb[0:n, cols],
            data2=b_sb[0:n, cols],
            op=nl.add,
            engine=nisa.engine.vector,
        )
    return sum_sb


def _rows_to_hbm(rows_sb: nl.NkiTensor, name: str) -> nl.NkiTensor:
    """[n, W] private HBM copy of an SBUF row tile, for a consumer that reads by indirect DMA."""
    n, W = rows_sb.shape
    out = nl.ndarray((n, W), dtype=rows_sb.dtype, buffer=_COLLECTIVE_HBM, name=unique_name(name))
    nisa.dma_copy(dst=out, src=rows_sb)
    return out


@nki.jit
def glm52_mtp_draft_chain_kernel(
    # ── step-0 prologue inputs: this rank's token slots ──
    inputs_embeds: nl.NkiTensor,  # [T_shard, H] bf16, embed_tokens[token the target sampled]
    previous_hidden: nl.NkiTensor,  # [T_shard, H] bf16, target's hidden POST final norm (raw; hnorm here)
    mtp_positions_all: nl.NkiTensor,  # [S, T_shard, 1] int32 absolute positions per step, for the position-0 mask
    enorm_w: nl.NkiTensor,  # [1, H]
    hnorm_w: nl.NkiTensor,  # [1, H]
    eh_proj_w_t: nl.NkiTensor,  # [2H, H / G] bf16, this rank's columns of eh_proj.weight TRANSPOSED
    # ── embedding feedback, vocab-sharded ──
    embed_shard_pad: nl.NkiTensor,  # [V_s + 2, H] bf16, this rank's embed_tokens rows with a zero row before and after
    vocab_base: nl.NkiTensor,  # [1, 1] int32 rank * V_s
    # ── attention, layer 78 ──
    input_norm_w: nl.NkiTensor,
    wq_a: nl.NkiTensor,
    wkv_a: nl.NkiTensor,
    q_norm_w: nl.NkiTensor,
    kv_norm_w: nl.NkiTensor,
    wq_b: nl.NkiTensor,
    q_absorb_w: nl.NkiTensor,
    out_absorb_w: nl.NkiTensor,
    o_proj_w: nl.NkiTensor,
    cos_all: nl.NkiTensor,  # [S, T_local, rope // 2] fp32, the query positions of each step
    sin_all: nl.NkiTensor,  # [S, T_local, rope // 2] fp32
    # ── DSA indexer, layer 78 (step 0 only: IndexShare) ──
    idx_wq_b: nl.NkiTensor,
    idx_wk: nl.NkiTensor,
    idx_k_norm_w: nl.NkiTensor,
    idx_k_norm_b: nl.NkiTensor,
    idx_weights_proj: nl.NkiTensor,
    idx_cos: nl.NkiTensor,  # [spec, idx_rope // 2] fp32, step 0's positions
    idx_sin: nl.NkiTensor,
    # ── caches, the MTP layer's OWN slabs, MUTATED (step 0 writes them) ──
    kv_cache: nl.NkiTensor,
    idx_k_cache: nl.NkiTensor,
    # ── per-step runtime tensors ──
    kv_new_slot_mapping_all: nl.NkiTensor,  # [S, num_seqs_local, spec] int32; steps 1.. all NEGATIVE (KVShare: no write)
    block_table: nl.NkiTensor,
    positions_all: nl.NkiTensor,  # [S, T_local] int32, CP-SHIFTED query positions per step
    idx_new_slot_mapping: nl.NkiTensor,  # [spec] int32, step 0
    idx_positions: nl.NkiTensor,  # [spec] int32, KEY-SHIFTED, step 0
    key_offset: nl.NkiTensor,
    cp_pos_offset: nl.NkiTensor,
    shared_extra_keys: nl.NkiTensor,  # [T_local, spec] int32 TRUE positions of step 0's own rows, per query row
    # ── MoE, layer 78, FP8 experts ──
    gamma: nl.NkiTensor,
    router_weights: nl.NkiTensor,
    router_bias: nl.NkiTensor,
    routed_gate_up_weights: nl.NkiTensor,
    routed_down_weights: nl.NkiTensor,
    shared_gate_up_weights: nl.NkiTensor,
    shared_down_weights: nl.NkiTensor,
    rank_id: nl.NkiTensor,
    # ── the draft head ──
    shared_head_norm_w: nl.NkiTensor,  # [1, H] fp32 shared_head.norm.weight
    lm_head_weight_t: nl.NkiTensor,  # [H, V_s] bf16
    sampling_params_all: nl.NkiTensor,  # [S, T_shard, 3|4] fp32 per step, rows this rank OWNS
    # ── replica groups ──
    token_replica_group: ReplicaGroup,
    cp_replica_group: ReplicaGroup,
    head_replica_group: ReplicaGroup,
    seq_replica_group: ReplicaGroup,
    key_replica_group: ReplicaGroup,
    ep_replica_group: ReplicaGroup,
    # ── degrees and model scalars, as in the giga kernel ──
    cp_degree: int,
    head_shard: int,
    token_shard: int,
    key_size: int,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
    idx_n_heads: int,
    idx_rope_dim: int,
    index_topk: int,
    spec: int,
    softmax_scale: float,
    kv_scale: float,
    rms_norm_eps: float,
    latent_norm_eps: float,
    idx_norm_eps: float,
    ep_size: int,
    tp_size: int,
    top_k: int,
    routed_scaling_factor: float,
    num_steps: int,
    live_rows: nl.NkiTensor,  # [n_seq, 1] int32 global slot of each sequence's draft row
    draft_rows: nl.NkiTensor,  # [num_seqs_local, 1] int32 draft row of each sequence in this rank's token slice
    eh_proj_replica_group: ReplicaGroup,  # G consecutive ranks, the eh_proj column-shard group
    routed_fp8_scales: nl.NkiTensor,  # [E_L, 3] fp32: one factor per routed expert weight (gate, up, down)
    shared_fp8_scales: nl.NkiTensor,  # [1, 3] fp32
    block_replica_group: ReplicaGroup = None,
    qblock_replica_group: ReplicaGroup = None,
    idx_block_table: nl.NkiTensor = None,
    idx_shard_degree: int = 1,  # degree the indexer's keys are DEALT over; 1 = its own contiguous slice
    idx_block_size: int = 0,  # the indexer cache's block size; the deal unit under a shared table
):
    """``num_steps`` GLM-5.2 MTP draft steps on this rank's token slots, in one trace.

    Step 0 is the full layer: its own indexer, its K/V and indexer key written. Steps 1.. reuse step 0's selection
    spliced with step 0's own rows (IndexShare), score no appended row and write nothing (KVShare), take
    ``shared_head.norm`` of the previous residual as ``previous_hidden`` and the previous draft's embedding, looked
    up on device, as ``inputs_embeds``.

    After each step's attention only the draft row of a sequence feeds anything, so the rest of the step runs on
    those ``n_seq`` rows, held by EVERY rank: one ``all_reduce`` assembles them from the token-sharded residual at
    step 0 (``_rows_from_slots``), the MoE runs on them with one more (``glm52_moe_rows_bf16``), the head reads them
    directly, and the next step's embedding is looked up for the ``n_seq`` tokens with a third. Steps 1.. run their
    prologue and attention exit on the rows too (the ROWS layout), so no step after the first touches the
    token-sharded slots. The rows stay in SBUF between the attention and the next step's attention, and the
    collectives among them take and return SBUF tiles (``lm_head/sbuf_collectives.py``, through private HBM).
    At the end each slot takes its sequence's row (``_slots_from_rows``), so ``out`` holds the draft row's residual
    in every slot of a sequence.

    Args:
        inputs_embeds (nl.NkiTensor): [T_shard, H] bf16 ``embed_tokens`` of each slot's step-0 input token, the
            token the target just sampled there.
        previous_hidden (nl.NkiTensor): [T_shard, H] bf16 the target's hidden state at those slots, after
            ``model.norm`` (``hnorm`` is applied here).
        enorm_w, hnorm_w (nl.NkiTensor): [1, H] ``enorm.weight`` and ``hnorm.weight``.
        mtp_positions_all (nl.NkiTensor): [S, T_shard, 1] int32 absolute position of each slot at each
            step (the target's position plus ``1 + step``). Only tested against zero.
        eh_proj_w_t (nl.NkiTensor): [2H, H / G] this rank's column shard of ``eh_proj.weight`` transposed: group
            position ``p`` holds columns ``[H / G * p, H / G * (p + 1))``; see :func:`_eh_proj_column_sharded`.
        embed_shard_pad (nl.NkiTensor): [V_s + 2, H] bf16 this rank's ``embed_tokens`` rows, zero-padded
            front and back; see ``_vocab_parallel_embedding``. V_s must equal the lm_head shard's.
        vocab_base (nl.NkiTensor): [1, 1] int32 ``rank * V_s``.
        cos_all, sin_all (nl.NkiTensor): [S, T_local, rope // 2] fp32 RoPE tables at each step's TRUE
            query positions.
        idx_*: Layer 78's indexer weights and tables (step 0 only).
        kv_cache, idx_k_cache: The MTP layer's own slabs, MUTATED by step 0.
        kv_new_slot_mapping_all (nl.NkiTensor): [S, num_seqs_local, spec] int32. Step 0's real mapping;
            under KVShare every later step must be all negative, which the cache write skips.
        positions_all (nl.NkiTensor): [S, T_local] int32 CP-shifted query positions per step.
        shared_extra_keys (nl.NkiTensor): [T_local, spec] int32 TRUE positions of step 0's own ``spec``
            rows for each query row's sequence. Steps 1.. attend them as prior keys (they are K/V from
            target-model hidden states, written by step 0), in place of the ``spec`` lowest-ranked
            entries of the selection.
        sampling_params_all (nl.NkiTensor): [S, T_shard, 3|4] fp32 per-step sampler parameters, which the greedy
            draft does not read: only their leading axis is checked against ``num_steps``.
        num_steps (int): Drafts per launch, S. Rank-invariant.
        live_rows (nl.NkiTensor): [n_seq, 1] int32 global slot indices (into the ``T_shard * ep_size``
            gathered rows) of the ONE draft row per sequence (the last accepted slot), identical on every
            rank. Every step runs the routed experts and the head over those rows only. This holds at step 0
            too: the K/V and indexer-key writes for all accepted rows depend on the layer input alone and are
            done in the attention block, and with a single MTP layer no later computation reads a non-draft
            row's output.
        draft_rows (nl.NkiTensor): [num_seqs_local, 1] int32, the row of each sequence's draft token in this rank's
            ``T_local`` token slice: ``live_rows[s] - T_local * (rank // token_group)`` for this slice's
            sequences. Steps 1.. query only those rows, from the draft rows of step 0's spliced selection
            (``draft_rows`` in ``glm52_dsa_attention_decode_layer_kernel``). Needs the token-group query path:
            ``wq_b`` / ``q_absorb_w`` over the token group, whatever the target's ``qproj_tp``.
        eh_proj_replica_group (ReplicaGroup): ``G`` consecutive ranks, in rank order, ``G`` more than one. Pass the
            block group when ``G`` equals its size, so the step uses one ``all_to_all`` group, not two.
        routed_fp8_scales, shared_fp8_scales (nl.NkiTensor): [E_L, 3] / [1, 3] fp32, the factors of the FP8
            experts, one per weight (gate, up, down), which ``mlp_bf16`` folds into instructions it issues anyway.
        input_norm_w ... o_proj_w, gamma: Layer 78's attention weights and ``post_attention_layernorm.weight``, in
            the per-layer shapes of ``glm52_dsa_attention_decode_layer_kernel``.
        router_weights (nl.NkiTensor): [H, E] bf16 ``mlp.gate.weight`` transposed; ``router_bias`` [1, E] fp32
            ``e_score_correction_bias``, which must stay fp32.
        routed_* / shared_* weights: Layer 78's float8_e4m3fn experts, this rank's routed experts (gate and up
            interleaved on axis 2) and the shared expert's TP shards.
        shared_head_norm_w (nl.NkiTensor): [1, H] fp32 ``shared_head.norm.weight``, the draft's own final norm.
        lm_head_weight_t (nl.NkiTensor): [H, V_s] bf16 this rank's columns of the main model's ``lm_head.weight``
            transposed (the checkpoint has no ``shared_head.head``).
        Replica groups, degrees, model scalars and the per-step runtime tensors: as the giga kernel takes them,
            at each step's positions.

    Returns:
        tuple, in this order:
            kv_cache, idx_k_cache: the MTP layer's caches after step 0's writes (no later step writes).
            selection: [T_local, index_topk] int32 step 0's selection (unspliced), the IndexShare carry.
            tokens_all: [S, n_seq, 1] uint32 the draft token of every step for every sequence, the same on every
                rank.
            out: [T_shard, H] the LAST step's pre-norm residual.
    """
    T_shard, H = inputs_embeds.shape
    T_local = T_shard * cp_degree * head_shard
    V_pad = embed_shard_pad.shape[0]
    V_s = lm_head_weight_t.shape[1]
    # The draft steps run the TOKEN-GROUP query path: the attention layer's block-sharded query path does not
    # take draft_rows. So the MTP layer keeps wq_b / q_absorb_w over the token group even when the target's query
    # path is block-sharded (qproj_tp) -- about 18 MB per rank at 64 ways -- just as it keeps its down-projections
    # over the token group under the target's latent_tp.
    q_degree = num_heads * (qk_nope_head_dim + qk_rope_head_dim) // wq_b.shape[1]
    kernel_assert(
        q_degree <= cp_degree * head_shard,
        f"{_ERROR_PREFIX} wq_b holds 1/{q_degree} of the query columns, a block-sharded query path: the draft "
        f"steps run the token-group query path, so the MTP layer's wq_b / q_absorb_w must be sharded over the "
        f"token group ({cp_degree * head_shard} ranks) or not at all; shard the MTP store's query path over the "
        f"token group",
    )
    n_seq = live_rows.shape[0]
    seq_i = draft_dst = None  # the slot <-> rows tables, built after step 0's prologue
    S = num_steps

    kernel_assert(S >= 1, f"{_ERROR_PREFIX} num_steps must be at least 1, got {S}")
    kernel_assert(
        mtp_positions_all.shape[0] == S and cos_all.shape[0] == S and sin_all.shape[0] == S,
        f"{_ERROR_PREFIX} per-step tensors need leading axis num_steps={S}",
    )
    kernel_assert(
        kv_new_slot_mapping_all.shape[0] == S and positions_all.shape[0] == S and sampling_params_all.shape[0] == S,
        f"{_ERROR_PREFIX} per-step tensors need leading axis num_steps={S}",
    )
    kernel_assert(
        V_pad == V_s + EMBED_PAD_ROWS and embed_shard_pad.shape[1] == H,
        f"{_ERROR_PREFIX} embed_shard_pad must be [V_s + {EMBED_PAD_ROWS}, H] = {(V_s + EMBED_PAD_ROWS, H)}, "
        f"got {embed_shard_pad.shape}",
    )
    kernel_assert(
        shared_extra_keys.shape == (T_local, spec),
        f"{_ERROR_PREFIX} shared_extra_keys must be [T_local, spec] = {(T_local, spec)}, got {shared_extra_keys.shape}",
    )
    kernel_assert(
        n_seq * spec == T_shard * ep_size,
        f"{_ERROR_PREFIX} the head needs one live row per sequence: {n_seq} x spec={spec} != {T_shard * ep_size} slots",
    )
    eh_proj_cols = eh_proj_w_t.shape[1]
    kernel_assert(
        eh_proj_w_t.shape[0] == 2 * H and H % eh_proj_cols == 0,
        f"{_ERROR_PREFIX} eh_proj_w_t must be [2H, H / G] for a group of G ranks, got {eh_proj_w_t.shape} at H={H}",
    )
    eh_proj_tp = H // eh_proj_cols
    kernel_assert(
        eh_proj_replica_group is not None
        and eh_proj_tp > 1
        and eh_proj_tp * T_shard <= P_MAX
        and ep_size % eh_proj_tp == 0,
        f"{_ERROR_PREFIX} the rows layout needs eh_proj column-sharded over more than one rank ({eh_proj_tp} ways) "
        f"with its eh_proj_replica_group, {eh_proj_tp} x T_shard={T_shard} <= {P_MAX} gathered rows, and "
        f"{eh_proj_tp} | ep_size={ep_size}",
    )
    token_group = cp_degree * head_shard
    num_seqs_local = T_local // spec
    rows_seq = _rows_index_hbm(rank_id, token_group, num_seqs_local, "mtp_rows_seq")
    oproj_tp = num_heads // out_absorb_w.shape[1]
    if oproj_tp != cp_degree:
        rows_exit_seq = _rows_index_hbm(rank_id, oproj_tp, (oproj_tp // cp_degree) * num_seqs_local, "mtp_rows_exit")
    else:
        rows_exit_seq = rows_seq

    # Step-0-only indexer helpers (per step, not per layer; one layer here). Under a shared block table
    # (idx_shard_degree > 1) the keys are DEALT over the CP group and the mask compares TRUE positions,
    # with the rank term in key_offset, exactly as the decode giga-kernel does.
    kernel_assert(
        idx_shard_degree == 1 or indexer_cache_is_blocked(idx_k_cache),
        f"{_ERROR_PREFIX} idx_shard_degree={idx_shard_degree} needs the block-form indexer cache",
    )
    idx_keys = indexer_cache_keys(idx_k_cache)
    # the chunk loop takes the giga kernel's mask (the packed head sum's layout); see mtp_idx_chunk_loop.
    # draft group: step 0's indexer queries the draft row's group of 4 rows (mtp_idx_query_tokens).
    idx_query_tokens = mtp_idx_query_tokens(spec, idx_keys, idx_block_table)

    def step0_mask(positions_in, key_offset_in):
        return mtp_idx_residency_mask(
            positions_in,
            idx_keys,
            idx_query_tokens,
            key_base=0,
            shard_degree=idx_shard_degree,
            block_size=idx_block_size,
            rank_offset=key_offset_in if idx_shard_degree > 1 else None,
            idx_block_rows=idx_block_table,
        )

    idx_mask_add = None  # built after step 0's prologue
    idx_block_rows = None
    if idx_block_table is not None:
        idx_block_rows = indexer_block_rows(idx_block_table, idx_keys, idx_keys // (idx_block_table.shape[1] - 1))

    tokens_all = nl.ndarray((S, n_seq, 1), dtype=nl.uint32, buffer=nl.shared_hbm, name=unique_name("mtp_tokens_all"))
    selection = nl.ndarray(
        (T_local, index_topk), dtype=nl.int32, buffer=nl.shared_hbm, name=unique_name("mtp_selection")
    )

    # Long context (idx_keys >= TOPK_MAX_N a rank): pin every collective operand of the steps to the kernel end, as
    # the giga kernel pins its layers' (utils/cc_pin.py). The attention ring opens at step 1, so step 0's
    # destinations are per-call buffers and are pinned too. Shorter contexts keep the unpinned graph: with pinning
    # off nothing is emitted.
    open_pins(default=idx_keys >= TOPK_MAX_N)
    embeds = inputs_embeds
    prev_hidden = previous_hidden
    emb_rows = prev_rows = None  # the rows-layout step inputs, set by each step for the next
    head = None  # the previous step's head handles
    hidden_tiles = None  # the next prologue's hnorm k-tiles (_prologue_hidden_tiles)
    # The head's (max, index) AllGather destinations, two chain-lifetime HBM buffers the steps rotate through, so no
    # other tensor takes their addresses and every head AllGather is handshake-free.
    head_pairs = [
        nl.ndarray(
            (ep_size * n_seq, 2),
            dtype=nl.float32,
            buffer=_COLLECTIVE_HBM,
            name=unique_name(f"mtp_head_pairs_ring{slot}"),
        )
        for slot in range(2)
    ]
    shared_sel = None
    shared_tables = (None, None, None)
    hidden = None
    # Trace-time state the rows-layout attention keeps across the draft steps: the exit rows go straight from SBUF
    # into one kernel-lifetime zero table that the exit AllReduce reads, zeroed again every step (exit_table), and
    # the attention picks its sequence's row straight from the rows (front_pick).
    exit_table = {"rezero": True}
    front_pick = {}

    for i in range(S):
        full = i == 0
        if i == 1:
            open_cc_dst_ring(_CC_DST_RING_DEPTH)

        # ═══════ Prologue ═══════
        rows_step = i > 0
        # The attention's pick reads whole 128-column k-tiles out of each rank's eh_proj column block of the
        # gathered rows, so a block narrower than a multiple of 128 (96 columns under a 64-way eh_proj) takes the
        # SBUF rows instead.
        rows_gathered = rows_step and (H // eh_proj_tp) % P_MAX == 0
        if rows_step:
            x_rows_out = _prologue_rows(
                emb_rows,
                prev_rows,
                enorm_w,
                hnorm_w,
                eh_proj_w_t,
                eh_proj_replica_group,
                eh_proj_tp,
                rms_norm_eps,
                head=head,
                hidden_tiles=hidden_tiles,
                gathered=rows_gathered,
            )
            if rows_gathered:
                # The gathered HBM rows feed the attention's pick as they are; the SBUF rows load after the attention.
                x_rows, x_rows_sb = x_rows_out, None
            else:
                # The attention picks its slice's rows by indirect DMA, which reads HBM; the residual add reads SBUF.
                x_rows_sb = x_rows_out
                x_rows = _rows_to_hbm(x_rows_sb, "mtp_x_rows")
            x_shard = None
        else:
            prologue_stores = []
            x_shard = _eh_proj_column_sharded(
                embeds,
                prev_hidden,
                mtp_positions_all[i],
                enorm_w,
                hnorm_w,
                eh_proj_w_t,
                eh_proj_replica_group,
                eh_proj_tp,
                rms_norm_eps,
                stores_out=prologue_stores,
            )
            # Step 0's helper tables -- the indexer's residency mask and the slot <-> rows tables -- from copies of
            # their inputs made once the prologue's normed rows are stored (the eh_proj AllGather's source), so they
            # run in that AllGather's shadow, where every engine waits, instead of ahead of the prologue norms.
            idx_mask_add = step0_mask(
                _gated_copy(idx_positions, prologue_stores, "mtp_idx_positions_late"),
                _gated_copy(key_offset, prologue_stores, "mtp_key_offset_late") if idx_shard_degree > 1 else key_offset,
            )
            seq_i, draft_dst = _slot_row_tables(
                _gated_copy(rank_id, prologue_stores, "mtp_rank_id_late"), live_rows, T_shard, spec
            )

        # ═══════ Attention: full on step 0, shared selection + KVShare after ═══════
        _layer_out = glm52_dsa_attention_decode_layer_kernel(
            x_shard=x_shard,
            input_norm_w=input_norm_w,
            wq_a=wq_a,
            wkv_a=wkv_a,
            q_norm_w=q_norm_w,
            kv_norm_w=kv_norm_w,
            wq_b=wq_b,
            q_absorb_w=q_absorb_w,
            out_absorb_w=out_absorb_w,
            o_proj_w=o_proj_w,
            cos=cos_all[i],
            sin=sin_all[i],
            idx_wq_b=idx_wq_b if full else None,
            idx_wk=idx_wk if full else None,
            idx_k_norm_w=idx_k_norm_w if full else None,
            idx_k_norm_b=idx_k_norm_b if full else None,
            idx_weights_proj=idx_weights_proj if full else None,
            idx_cos=idx_cos if full else None,
            idx_sin=idx_sin if full else None,
            kv_cache=kv_cache,
            kv_new_slot_mapping=kv_new_slot_mapping_all[i],
            block_table=block_table,
            positions=positions_all[i],
            idx_k_cache=idx_k_cache if full else None,
            idx_new_slot_mapping=idx_new_slot_mapping if full else None,
            idx_positions=idx_positions if full else None,
            key_offset=key_offset if full else None,
            cp_pos_offset=cp_pos_offset,
            token_replica_group=token_replica_group,
            cp_replica_group=cp_replica_group,
            head_replica_group=head_replica_group,
            seq_replica_group=seq_replica_group,
            key_replica_group=key_replica_group,
            block_replica_group=block_replica_group,
            qblock_replica_group=qblock_replica_group,
            cp_degree=cp_degree,
            head_shard=head_shard,
            token_shard=token_shard,
            key_size=key_size,
            num_heads=num_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            kv_lora_rank=kv_lora_rank,
            idx_n_heads=idx_n_heads,
            idx_rope_dim=idx_rope_dim,
            index_topk=index_topk,
            spec=spec,
            softmax_scale=softmax_scale,
            kv_scale=kv_scale,
            input_norm_eps=rms_norm_eps,
            latent_norm_eps=latent_norm_eps,
            idx_norm_eps=idx_norm_eps,
            prev_gather_indices=None if full else shared_sel,
            prior_snake_rows=None if full else shared_tables[0],
            prior_key_pos=None if full else shared_tables[1],
            prior_natural_rows=None if full else shared_tables[2],
            idx_mask_add=idx_mask_add if full else None,
            idx_block_rows=idx_block_rows if full else None,
            idx_shard_degree=idx_shard_degree,
            idx_block_size=idx_block_size,
            # KVShare: a later step scores only the gathered set, never its own appended row.
            attend_new_keys=full,
            draft_rows=None if full else draft_rows,
            rows_in=x_rows if rows_step else None,
            rows_seq=rows_seq if rows_step else None,
            rows_exit_seq=rows_exit_seq if rows_step else None,
            rows_replica_group=ep_replica_group if rows_step else None,
            exit_table=exit_table if rows_step else None,
            front_pick=front_pick if rows_step else None,
            rows_in_groups=eh_proj_tp if rows_gathered else 1,
            kv_free=0 if full else _KV_FREE,
            early_keys=not full,
            idx_draft_rows=draft_rows if (i == 0 and idx_query_tokens != spec) else None,
        )
        attn_out, selection_raw = _layer_out[0], _layer_out[1]
        if rows_gathered:
            # Only the residual add reads it, after the attention's exit AllReduce.
            x_rows_sb = load_gathered_columns(
                nl.ndarray((n_seq, H), dtype=x_rows.dtype, buffer=nl.sbuf), x_rows, eh_proj_tp
            )

        if i == 0:
            selection_flat = selection_raw.reshape((T_local, index_topk))
            sel_sb = nl.ndarray((T_local, index_topk), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(dst=sel_sb, src=selection_flat)
            nisa.dma_copy(dst=selection, src=sel_sb)
            if S > 1:
                # IndexShare + KVShare carry: step 0's selection with its last ``spec`` entries replaced
                # by step 0's own rows, so those keys (K/V from target hidden states, written by step 0)
                # stay visible to every later step as prior keys. The selection never holds a position
                # at or past this step's own, so the splice introduces no duplicate.
                nisa.dma_copy(dst=sel_sb[0:T_local, index_topk - spec : index_topk], src=shared_extra_keys)
                shared_sel = nl.ndarray(
                    (T_local, index_topk), dtype=nl.int32, buffer=_COLLECTIVE_HBM, name=unique_name("mtp_shared_sel")
                )
                nisa.dma_copy(dst=shared_sel, src=sel_sb)
                # Resolved once: the tables depend on the selection, the block table and the CP shift, none of which
                # change across the chain. Steps 1.. query the draft rows only, so only their selection rows and
                # tables are built.
                shared_sel = select_rows(shared_sel, draft_rows, num_seqs_local, "mtp_shared_sel_draft")
                # Only the SHAPE of positions is read here.
                positions_draft = nl.ndarray(
                    (num_seqs_local,), dtype=nl.int32, buffer=_COLLECTIVE_HBM, name=unique_name("mtp_pos_draft")
                )
                shared_tables = build_shared_prior_index(
                    gather_indices_true=shared_sel,
                    cp_pos_offset=cp_pos_offset,
                    kv_cache=kv_cache,
                    block_table=block_table,
                    positions=positions_draft,
                    T_local=num_seqs_local,
                    num_seqs_local=num_seqs_local,
                    spec=1,
                    index_topk=index_topk,
                    kv_lora_rank=kv_lora_rank,
                    qk_rope_head_dim=qk_rope_head_dim,
                    cp_degree=cp_degree,
                    io_dtype=inputs_embeds.dtype,
                )

        # ═══════ The rest of the step on one row per sequence, on every rank ═══════
        # Every [n_seq, H] tensor from here to the next step's attention stays in SBUF.
        if rows_step:
            h_rows = _rows_add(x_rows_sb, attn_out, blocks=_EXIT_ADD_BLOCKS)
        else:
            hidden = _residual_add(x_shard, attn_out, f"mtp_attn_s{i}")
            h_rows = nl.ndarray((n_seq, H), dtype=inputs_embeds.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=h_rows, src=_rows_from_slots(hidden, draft_dst, n_seq, ep_replica_group))
        moe_weight_dmas = []
        moe_rows = glm52_moe_rows_bf16(
            h_rows=h_rows,
            gamma=gamma,
            router_weights=router_weights,
            router_bias=router_bias,
            routed_gate_up_weights=routed_gate_up_weights,
            routed_down_weights=routed_down_weights,
            shared_gate_up_weights=shared_gate_up_weights,
            shared_down_weights=shared_down_weights,
            rank_id=rank_id,
            ep_replica_group=ep_replica_group,
            ep_size=ep_size,
            tp_size=tp_size,
            top_k=top_k,
            eps=rms_norm_eps,
            routed_scaling_factor=routed_scaling_factor,
            routed_fp8_scales=routed_fp8_scales,
            shared_fp8_scales=shared_fp8_scales,
            weight_dmas_out=moe_weight_dmas,
        )
        h_rows = _rows_add(h_rows, moe_rows, blocks=_MOE_AR_BLOCKS)
        # The tokens go straight to this step's output row and, on chip, to the next step's lookup.
        tok_sb = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
        last = i + 1 == S
        # Before the next step: the head's handles, so the next step's work the scheduler hoists into the head (the
        # shared-head norm's scaling, the next prologue's gamma loads) is ordered behind it, off its critical path;
        # and the next step's embedding-table rows, written by the tail straight from its winning keys, with the
        # step's token store issued after the lookup.
        head = None if last else {}
        emb_idx = None if last else nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
        glm52_mtp_greedy_tail(
            hidden_states=h_rows,
            gamma=shared_head_norm_w,
            lm_head_weight_t=lm_head_weight_t,
            vocab_base=vocab_base,
            vocab_replica_group=ep_replica_group,
            vocab_size=ep_size * V_s,
            eps=rms_norm_eps,
            tokens_out=tokens_all[i],
            tokens_sb_out=tok_sb,
            lm_head_weights_after=moe_weight_dmas[-1],
            handles_out=head,
            row_index_out=None if emb_idx is None else (emb_idx, V_pad),
            store_tokens=emb_idx is None,
            pairs_dst=head_pairs[i % 2],
        )
        if last:
            hidden = _slots_from_rows(_rows_to_hbm(h_rows, "mtp_out_rows_hbm"), seq_i, T_shard, "mtp_out_rows")
            continue
        prev_rows = _shared_head_norm(h_rows, shared_head_norm_w, rms_norm_eps, head)
        # The next prologue's hnorm and its half of the eh_proj transposes run here, in the head AllGather and
        # embedding AllReduce shadow, instead of after the AllReduce behind enorm (the matmuls keep their k order).
        if _hnorm_early_fits(n_seq, H):
            hidden_tiles = _prologue_hidden_tiles(prev_rows, hnorm_w, rms_norm_eps, head)
        emb_rows = _vocab_parallel_embedding(
            embed_shard_pad, ep_replica_group, inputs_embeds.dtype, emb_idx, n_seq, head
        )
        # The tail's token store, issued after the lookup on the same (GpSimd) queue.
        store = nisa.dma_copy(dst=tokens_all[i][0:n_seq, 0:1], src=tok_sb[0:n_seq, 0:1])
        store.depends_on(head["emb_gather"])

    close_cc_dst_ring()
    out = nl.ndarray((T_shard, H), dtype=inputs_embeds.dtype, buffer=nl.shared_hbm, name=unique_name("mtp_chain_out"))
    out_sb = nl.ndarray((T_shard, H), dtype=out.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=out_sb, src=hidden)
    close_pins(out_sb)  # no-op when pinning is off
    nisa.dma_copy(dst=out, src=out_sb)

    # ``out`` LAST, after the per-step token outputs: a validator for the final residual can then know which
    # rows followed the reference's token path (a flipped argmax on a row makes its later steps a different,
    # equally valid, computation).
    return kv_cache, idx_k_cache, selection, tokens_all, out


__all__ = ["glm52_mtp_draft_chain_kernel", "EMBED_PAD_ROWS"]
