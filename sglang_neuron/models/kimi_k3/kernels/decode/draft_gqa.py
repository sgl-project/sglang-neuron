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

"""Kimi-K3 DSpark GQA drafter kernels (one DP1 replica on 128 ranks, LNC1, Trn3).

The drafter has 5 Qwen3 layers (64 query heads, 16 key/value heads, head dim 64), the context stage
``hidden_norm(fc(cat taps))``, the target embedding and lm_head, and a rank-256 Markov head. One draft step drafts
7 tokens for each of B sequences (1 <= B <= 8) in one kernel call.

Sharding: the attention is TP16 x CP8, and the context stage, the MLP and the tail are TP128. Rank ``8 h + c`` holds
kv head ``h`` (query heads ``4 h .. 4 h + 3``) and the CP8 slice ``c`` of the positions. The 8 ranks of one kv head are
one chip; the attention merge is a CP8 all-reduce inside it.

Rider rows: rank ``(h, c)`` holds one row of 128 values ``[K_h | V_h]`` per position of its slice and per layer. K is
the neox RoPE of ``k_norm(k_proj f)`` at the true position, and V is ``v_proj f``, with ``f`` the context feature.
Position ``p`` of sequence ``b`` is in slice ``(p >> 7) & 7``, in page ``block_table[b, p >> 15]``, at row
``((p >> 10) & 31) * 128 + (p & 127)`` of the 4096-row page.

Batch: the block rows are token-major (row ``t B + b``), and so are the verify rows (row ``j B + b``). Every
collective carries all B sequences, so the collective count does not depend on B. Each sequence reads only its own
pages; two sequences can read one shared page, but a sequence writes only its own rows.

The pieces:

- ``gqa_step_constants``: the tiles that the 5 layers share (key masks, RoPE columns, constant stationaries).
- ``gqa_block_attention``: one layer's attention in draft mode. Each rank projects its own head's q, k and v for the
  7 block rows and the 8 verify rows (the 8 ranks of a head do this alike, so no collective is needed). It attends the
  4 query heads over the prior keys of its slice; the rank of slice 0 also attends the current keys (the 7 block keys
  and the accepted verify keys). A CP8 all-reduce adds the numerators and the sums. Each rank runs head ``h``'s 256
  inputs of ``o_proj`` into its 896 output columns; a TP16 all-reduce adds the 16 heads, and a CP8 all-gather
  rebuilds the 7168 row. The owner rank of each verify position writes its rider row.
- ``draft_gqa_layer``: ``h += attention(input_norm(h))``, then ``h += silu_mlp(post_attention_norm(h))`` (TP128).
- ``kimi_k3_draft_gqa_step``: the context feature, the embed gather, 5 layers, and the tail.

The softmax reference of a query row is its own block-key score plus ``64.46 / scale``, as in ``tkg_mla_block``: the
weights of all ranks share it, so the partial sums add, and no running maximum is needed. A score more than
``60 / scale`` above the reference is capped.
"""

import math

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from .utils.kernel_assert import kernel_assert
from .common import P_MAX, load_vector_hT, new_ones, rmsnorm_hT, store_hT_rows
from .draft_gqa_tail import gqa_draft_tail
from .draft_mla import (
    _MARKOV_RANK,
    _ArenaPlan,
    _build_aux_hT,
    _compute_key_start,
    _embed_block,
    _load_rows_hT,
    _matmul_tok,
    _own_positions,
    _rows_to_hT,
    _silu_mlp,
)
from .tkg_sbuf_alloc import AUTO_SBUF, ExplicitSbuf, SbufTiles, install_identities

HIDDEN = 7168
_H_TILES = HIDDEN // P_MAX
CP = 128
"""Ranks of the drafter: TP128 for the context stage, the MLP and the tail."""
_CP8 = 8
_TP16 = 16
_LAYERS = 5
_BLOCK = 7
_VERIFY = 8
_MASK_TOKEN = 163824
_EPS = 1e-5

_HEAD_DIM = 64
_Q_HEADS = 4
"""Query heads of one rank (one kv head)."""
_Q_COLS = _Q_HEADS * _BLOCK
"""Query columns of one rank: 4 heads x 7 rows; column ``7 a + t`` is head ``4 h + a``, row ``t``."""
_QKV_COLS = (_Q_HEADS + 2) * _HEAD_DIM
_ROW = 2 * _HEAD_DIM
"""Values of one rider row, ``[K_h | V_h]``."""
_PAGE_ROWS = 4096
"""Rows of one page: the 4096 positions of one rank's slice in a 32768-token block."""
_BLOCK_TOKENS = 32768
_STRIPE = 128
_PART_ROWS = _PAGE_ROWS // P_MAX
"""Page rows on one partition: partition ``p`` takes rows ``32 p .. 32 p + 31``, so key block ``k`` (column ``k``)
holds row ``32 p + k`` on partition ``p``."""
_KEY_GROUP = 16
"""Key blocks of one score tile: 16 x 28 columns fill one FP32 PSUM bank."""
_GROUP_COLS = _KEY_GROUP * _Q_COLS
_O_COLS = HIDDEN // _CP8
"""Output columns of ``o_proj`` on one rank (896, 7 hT tiles)."""
_O_TILES = _O_COLS // P_MAX
_MAX_SEQS = 8
"""Largest batch: the block rows and the verify rows of 8 sequences fill the 128 current rows."""
_BATCH_MAX_PAGES = 8
"""Pages per sequence and layer of a batched call (B > 1), at most: the checked fit is 256K full context at B = 8."""
_MERGE_OFFSET = 93.0 * math.log(2.0)
"""Offset of the softmax reference above the scaled own-key score, so the own key weighs ``2**-93``."""
_WEIGHT_CAP = 60.0
_MASK_BIAS = -1.0e30
_FP8_MAX = 448.0
_BF16_KEY_MAX = 65536.0


def gqa_current_rows(tokens):
    """Return ``(verify_row0, cur_rows)`` of B sequences.

    Rows ``0 .. 7B - 1`` of the current rows are the block rows, and the verify rows start at the next quadrant (a
    partition start of 32 or 64), so each set starts at a quadrant. B = 1 gives ``(32, 40)``; B = 8 gives ``(64, 128)``.
    """
    verify_row0 = 32 * (-(-(_BLOCK * tokens) // 32))
    return verify_row0, verify_row0 + _VERIFY * tokens


def _seq_suffix(b):
    """Return the instruction name suffix of sequence ``b`` (empty for sequence 0)."""
    return f"_s{b}" if b else ""


def gqa_replica_groups():
    """Return the ``(cp8, tp16)`` replica groups of rank ``8 h + c``: the 8 slices of one kv head, and the 16 heads
    of one slice."""
    cp8 = ReplicaGroup([[_CP8 * h + c for c in range(_CP8)] for h in range(_TP16)])
    tp16 = ReplicaGroup([[_CP8 * h + c for h in range(_TP16)] for c in range(_CP8)])
    return cp8, tp16


def gqa_rider_pool(rider_pool):
    """Return the rider pool as ``[pages, 4096 * 128]`` BF16 or FP8 (a BF16 ``[pages, 4096 * 64]`` holds FP8 pairs)."""
    if rider_pool.dtype == nl.bfloat16 and rider_pool.shape[1] == _PAGE_ROWS * _ROW // 2:
        return rider_pool.view(nl.float8_e4m3fn).reshape((rider_pool.shape[0], 2 * rider_pool.shape[1]))
    return rider_pool


def _rope_column_rows(window, flat_idx, part0, rows, dst, tag):
    """Gather ``window_flat[flat_idx[r]]`` into the rows ``part0 + r`` of ``dst`` (``[cur_rows, 128]`` FP32 SBUF)."""
    width = window.shape[2]
    flat = window.reshape((window.shape[0] * window.shape[1], width))
    for r in range(rows):
        nisa.dma_copy(
            dst=dst[part0 + r : part0 + r + 1, 0:width],
            src=flat.ap(
                pattern=[[width, 1], [1, width]], offset=0, scalar_offset=flat_idx[r : r + 1, 0:1], indirect_dim=0
            ),
            dge_mode=nisa.dge_mode.hwdge,
            engine=nisa.engine.sync,
            name=f"{tag}_rope_{r}",
        )


def _rotation_matrix(sb):
    """Return the BF16 ``[64, 64]`` stationary ``R`` with ``R^T x = rotate_half(x) = cat(-x[32:], x[:32])``.

    Column ``m < 32`` takes ``-x[m + 32]`` and column ``m >= 32`` takes ``x[m - 32]``. The two 32 x 32 blocks come
    from the diagonal blocks of the identity, on their own partitions.
    """
    bf16 = nl.bfloat16
    identity = nl.shared_identity_matrix(P_MAX, dtype=bf16)
    rot = sb.tile((_HEAD_DIM, _HEAD_DIM), bf16)
    nisa.memset(dst=rot, value=0.0)
    nisa.tensor_scalar(dst=rot[32:64, 0:32], data=identity[32:64, 32:64], op0=nl.multiply, operand0=-1.0)
    nisa.tensor_copy(dst=rot[0:32, 32:64], src=identity[0:32, 0:32])
    return rot


def _own_select(tokens, sb):
    """Return the FP32 ``[cur_rows, 28 B]`` mask of the own block key of each query column.

    Column ``28 b + 7 a + t`` (sequence ``b``, head ``a``, block row ``t``) selects current row ``t B + b``.
    """
    fp32 = nl.float32
    int32 = nl.int32
    _, cur_rows = gqa_current_rows(tokens)
    cols = _Q_COLS * tokens
    col_i = sb.tile((cur_rows, cols), int32)
    pattern = [[0, _Q_HEADS], [1, _BLOCK]] if tokens == 1 else [[1, tokens], [0, _Q_HEADS], [tokens, _BLOCK]]
    nisa.iota(dst=col_i, pattern=pattern, offset=0, channel_multiplier=0)
    part_i = sb.tile((cur_rows, 1), int32)
    nisa.iota(dst=part_i, pattern=[[0, 1]], offset=0, channel_multiplier=1)
    col_f = sb.tile((cur_rows, cols), fp32)
    nisa.tensor_copy(dst=col_f, src=col_i)
    part_f = sb.tile((cur_rows, 1), fp32)
    nisa.tensor_copy(dst=part_f, src=part_i)
    select = sb.tile((cur_rows, cols), fp32)
    nisa.tensor_scalar(dst=select, data=col_f, op0=nl.equal, operand0=part_f)
    return select


def _bias_moving(sb):
    """Return the ``[17, 16, 28]`` BF16 bias moving operand of a score tile: row ``1 + b`` is 1 on the columns of key
    block ``b`` (``28 b .. 28 b + 27``), row 0 takes the negative reference of each layer."""
    fp32 = nl.float32
    int32 = nl.int32
    rows = 1 + _KEY_GROUP
    ind_i = sb.tile((rows, _KEY_GROUP, _Q_COLS), int32)
    nisa.iota(dst=ind_i, pattern=[[1, _KEY_GROUP], [0, _Q_COLS]], offset=1, channel_multiplier=-1)
    ind_f = sb.tile((rows, _KEY_GROUP, _Q_COLS), fp32)
    nisa.tensor_copy(dst=ind_f, src=ind_i)
    moving = sb.tile((rows, _KEY_GROUP, _Q_COLS), nl.bfloat16)
    nisa.tensor_scalar(dst=moving, data=ind_f, op0=nl.equal, operand0=0.0)
    return moving


def _mask_bias(valid, out):
    """Write ``(valid - 1) * 1e30`` into the BF16 bias stationary ``out``, then 1 into its row 0."""
    nisa.tensor_scalar(dst=out, data=valid, op0=nl.subtract, operand0=1.0, op1=nl.multiply, operand1=-_MASK_BIAS)
    nisa.memset(dst=out[0:1, 0 : out.shape[1]], value=1.0)


def _rank_scalars(rank_info, rows, sb):
    """Return ``(c, c == 0)`` FP32 ``[rows, 1]`` of this rank (``c = rank & 7``) and the INT32 ``[rows, 1]`` c."""
    fp32 = nl.float32
    int32 = nl.int32
    rank_i = sb.tile((rows, 1), int32)
    nisa.dma_copy(
        dst=rank_i,
        src=rank_info[0:1, 2:3].broadcast(0, rows),
        dge_mode=nisa.dge_mode.hwdge,
        engine=nisa.engine.sync,
        name=f"gqa_rank_{rows}",
    )
    c_i = sb.tile((rows, 1), int32)
    nisa.tensor_scalar(dst=c_i, data=rank_i, op0=nl.bitwise_and, operand0=_CP8 - 1)
    c_f = sb.tile((rows, 1), fp32)
    nisa.tensor_copy(dst=c_f, src=c_i)
    is_c0 = sb.tile((rows, 1), fp32)
    nisa.tensor_scalar(dst=is_c0, data=c_f, op0=nl.equal, operand0=0.0)
    return c_f, is_c0, c_i


def gqa_page_count(nb_table, window, tokens):
    """Return the blocks that each layer reads per sequence: all ``nb_table``, or ``ceil(window / 32768) + 1`` with a window.

    A batched call (``tokens`` > 1) holds the page masks of every sequence, so it fails closed above
    ``_BATCH_MAX_PAGES`` pages per sequence, the largest table whose fit is checked.
    """
    page_count = nb_table if window <= 0 else min(nb_table, -(-window // _BLOCK_TOKENS) + 1)
    kernel_assert(
        tokens == 1 or page_count <= _BATCH_MAX_PAGES,
        f"a batched GQA drafter step (B = {tokens}) reads {page_count} pages per sequence, above the checked "
        f"{_BATCH_MAX_PAGES}; use a window or B = 1",
    )
    return page_count


def gqa_step_constants(row0, accepted, key_start, block_table, rank_info, cur_rope, num_pages, sb, page_count=None):
    """Build the tiles that the 5 layers of one step share.

    Args:
        row0, accepted, key_start: ``[B, 1]`` INT32 HBM verify row-0 position, accepted count (``P = row0 + accepted``),
            and the lower bound of the visible prior positions of each sequence.
        block_table: ``[B, NB]`` INT32 HBM page of each 32768-token block of each sequence.
        rank_info: ``[1, 4]`` INT32 HBM, the rank in column 2.
        cur_rope: ``[cur_rows, 128]`` FP32 SBUF neox ``[cos | sin]`` rows of the current rows (``gqa_current_rope``).
        num_pages: pages of one layer's pool; the page ids are clamped into it.
        page_count: the blocks that each layer reads per sequence (default: all ``NB``). With fewer, sequence ``b``
            reads the blocks ``n0 .. n0 + page_count - 1`` with ``n0 = min(key_start[b] >> 15, NB - page_count)``.

    Returns:
        A dict of SBUF tiles; ``page_bias``, ``block_ids`` and ``page_moving`` are lists with one tile per sequence. The
        bias stationaries are BF16 with row 0 = 1 (it adds the per-column negative reference of the moving row 0) and
        the key masks (0 or ``-1e30``) below: one row for each key block of a page tile, or one row for each sequence
        of the current keys.
    """
    fp32 = nl.float32
    bf16 = nl.bfloat16
    int32 = nl.int32
    hwdge = nisa.dge_mode.hwdge
    sync = nisa.engine.sync
    tokens = row0.shape[0]
    verify_row0, cur_rows = gqa_current_rows(tokens)
    nb_table = block_table.shape[1]
    nb = nb_table if page_count is None else min(page_count, nb_table)
    rows = 1 + _KEY_GROUP
    c_f, is_c0, _c_i = _rank_scalars(rank_info, rows, sb)
    slice_base = sb.tile((rows, 1), fp32)
    nisa.tensor_scalar(dst=slice_base, data=c_f, op0=nl.multiply, operand0=float(_STRIPE))
    table_col = block_table.reshape((tokens * nb_table, 1))
    width = nb * 2 * P_MAX

    page_biases, block_ids_all = [], []
    for b in range(tokens):
        sfx = _seq_suffix(b)
        # The scalars of sequence b on the 17 mask rows: row0, accepted, key_start.
        scal_i = sb.tile((rows, 3), int32)
        for col, src in enumerate((row0[b : b + 1, 0:1], accepted[b : b + 1, 0:1], key_start[b : b + 1, 0:1])):
            nisa.dma_copy(
                dst=scal_i[0:rows, col : col + 1],
                src=src.broadcast(0, rows),
                dge_mode=hwdge,
                engine=sync,
                name=f"gqa_scal_{col}{sfx}",
            )
        scal = sb.tile((rows, 3), fp32)
        nisa.tensor_copy(dst=scal, src=scal_i)
        row0_f = scal[0:rows, 0:1]
        key_f = scal[0:rows, 2:3]
        # The first block that the layers read: n0 = min(key_start >> 15, NB - nb), on every mask row.
        shift_i = sb.tile((rows, 1), int32)
        nisa.tensor_scalar(dst=shift_i, data=scal_i[0:rows, 2:3], op0=nl.right_shift, operand0=15)
        first_f = sb.tile((rows, 1), fp32)
        nisa.tensor_copy(dst=first_f, src=shift_i)
        nisa.tensor_scalar(dst=first_f, data=first_f, op0=nl.minimum, operand0=float(nb_table - nb))
        first_i = sb.tile((rows, 1), int32)
        nisa.tensor_copy(dst=first_i, src=first_f)
        block_base = sb.tile((rows, 1), fp32)
        nisa.tensor_scalar(dst=block_base, data=first_f, op0=nl.multiply, operand0=float(_BLOCK_TOKENS))

        # Prior keys. Mask row 1 + k of score tile (page n, group g) is key block 16 g + k; its partition p holds page
        # row 32 p + 16 g + k, which is position n * 32768 + (p >> 2) * 1024 + 128 c + (p & 3) * 32 + 16 g + k.
        # Visible: [key_start, row0). The block offset n * 32768 is a separate iota: an iota step must fit in int16.
        inner_i = sb.tile((rows, 2, P_MAX // 4, 4), int32)
        nisa.iota(dst=inner_i, pattern=[[_KEY_GROUP, 2], [1024, P_MAX // 4], [32, 4]], offset=-1, channel_multiplier=1)
        inner = sb.tile((rows, 2 * P_MAX), fp32)
        nisa.tensor_scalar(dst=inner, data=inner_i.reshape((rows, 2 * P_MAX)), op0=nl.add, operand0=slice_base)
        blk_i = sb.tile((rows, nb, 2 * P_MAX), int32)
        nisa.iota(dst=blk_i, pattern=[[1, nb], [0, 2 * P_MAX]], offset=0, channel_multiplier=0)
        pos = sb.tile((rows, nb, 2 * P_MAX), fp32)
        nisa.tensor_scalar(
            dst=pos, data=blk_i, op0=nl.multiply, operand0=float(_BLOCK_TOKENS), op1=nl.add, operand1=block_base
        )
        nisa.tensor_tensor(
            dst=pos, data1=pos, data2=inner[0:rows, 0 : 2 * P_MAX].expand_dim(1).broadcast(1, nb), op=nl.add
        )
        pos = pos.reshape((rows, width))
        lo_ok = sb.tile((rows, width), fp32)
        nisa.tensor_scalar(dst=lo_ok, data=pos, op0=nl.greater_equal, operand0=key_f)
        valid = sb.tile((rows, width), fp32)
        nisa.tensor_scalar(dst=valid, data=pos, op0=nl.less, operand0=row0_f)
        nisa.tensor_tensor(dst=valid, data1=valid, data2=lo_ok, op=nl.multiply)
        page_bias = sb.tile((rows, nb, 2, P_MAX), bf16)
        _mask_bias(valid, page_bias.reshape((rows, width)))
        page_biases.append(page_bias)

        block_ids = sb.tile((1, nb), int32)
        nisa.dma_copy(
            dst=block_ids,
            src=table_col.ap(
                pattern=[[1, 1], [1, nb]], offset=b * nb_table, scalar_offset=first_i[0:1, 0:1], indirect_dim=0
            ),
            dge_mode=hwdge,
            engine=sync,
            name=f"gqa_table{sfx}",
        )
        nisa.tensor_scalar(
            dst=block_ids, data=block_ids, op0=nl.minimum, operand0=float(num_pages - 1), op1=nl.maximum, operand1=0.0
        )
        block_ids_all.append(block_ids)

    # Current keys (slice 0 only). Stationary row 1 + b is the mask of sequence b: its block rows (current row t B + b)
    # always, its verify row j (current row verify_row0 + j B + b) when j < accepted and row0 + j >= key_start.
    srows = 1 + tokens
    cur_i = sb.tile((srows, 3), int32)
    nisa.memset(dst=cur_i, value=0)
    for col, src in enumerate((row0, accepted, key_start)):
        nisa.dma_copy(
            dst=cur_i[1:srows, col : col + 1],
            src=src[0:tokens, 0:1],
            dge_mode=hwdge,
            engine=sync,
            name=f"gqa_cur_{col}",
        )
    cur_s = sb.tile((srows, 3), fp32)
    nisa.tensor_copy(dst=cur_s, src=cur_i)
    key_minus_row0 = sb.tile((srows, 1), fp32)
    nisa.tensor_tensor(dst=key_minus_row0, data1=cur_s[0:srows, 2:3], data2=cur_s[0:srows, 0:1], op=nl.subtract)
    seq_i = sb.tile((srows, 1), int32)
    nisa.iota(dst=seq_i, pattern=[[0, 1]], offset=-1, channel_multiplier=1)
    seq_f = sb.tile((srows, 1), fp32)
    nisa.tensor_copy(dst=seq_f, src=seq_i)
    nq = _BLOCK * tokens
    # Sequence and verify index of each current row; the gap rows between the two sets match no sequence.
    col_seq_i = sb.tile((srows, cur_rows), int32)
    nisa.memset(dst=col_seq_i, value=-2)
    nisa.iota(dst=col_seq_i[0:srows, 0:nq], pattern=[[0, _BLOCK], [1, tokens]], offset=0, channel_multiplier=0)
    nisa.iota(
        dst=col_seq_i[0:srows, verify_row0:cur_rows],
        pattern=[[0, _VERIFY], [1, tokens]],
        offset=0,
        channel_multiplier=0,
    )
    col_j_i = sb.tile((srows, cur_rows), int32)
    nisa.memset(dst=col_j_i, value=-1)
    nisa.iota(
        dst=col_j_i[0:srows, verify_row0:cur_rows], pattern=[[1, _VERIFY], [0, tokens]], offset=0, channel_multiplier=0
    )
    col_seq = sb.tile((srows, cur_rows), fp32)
    nisa.tensor_copy(dst=col_seq, src=col_seq_i)
    j = sb.tile((srows, cur_rows), fp32)
    nisa.tensor_copy(dst=j, src=col_j_i)
    ver = sb.tile((srows, cur_rows), fp32)
    nisa.tensor_scalar(dst=ver, data=j, op0=nl.less, operand0=cur_s[0:srows, 1:2])
    ver_lo = sb.tile((srows, cur_rows), fp32)
    nisa.tensor_scalar(dst=ver_lo, data=j, op0=nl.greater_equal, operand0=key_minus_row0)
    nisa.tensor_tensor(dst=ver, data1=ver, data2=ver_lo, op=nl.multiply)
    nisa.tensor_scalar(dst=ver_lo, data=j, op0=nl.greater_equal, operand0=0.0)
    nisa.tensor_tensor(dst=ver, data1=ver, data2=ver_lo, op=nl.multiply)
    blk = sb.tile((srows, cur_rows), fp32)
    nisa.memset(dst=blk, value=0.0)
    nisa.memset(dst=blk[0:srows, 0:nq], value=1.0)
    cur_valid = sb.tile((srows, cur_rows), fp32)
    nisa.tensor_tensor(dst=cur_valid, data1=ver, data2=blk, op=nl.add)
    match = sb.tile((srows, cur_rows), fp32)
    nisa.tensor_scalar(dst=match, data=col_seq, op0=nl.equal, operand0=seq_f)
    nisa.tensor_tensor(dst=cur_valid, data1=cur_valid, data2=match, op=nl.multiply)
    nisa.tensor_scalar(dst=cur_valid, data=cur_valid, op0=nl.multiply, operand0=is_c0[0:srows, 0:1])
    cur_bias = sb.tile((srows, cur_rows), bf16)
    _mask_bias(cur_valid, cur_bias)

    # The current-key bias moving operand: row 1 + b is 1 on the 28 query columns of sequence b; row 0 takes the
    # negative references of each layer.
    cur_ind_i = sb.tile((srows, tokens, _Q_COLS), int32)
    nisa.iota(dst=cur_ind_i, pattern=[[1, tokens], [0, _Q_COLS]], offset=0, channel_multiplier=0)
    cur_ind = sb.tile((srows, tokens * _Q_COLS), fp32)
    nisa.tensor_copy(dst=cur_ind, src=cur_ind_i.reshape((srows, tokens * _Q_COLS)))
    cur_moving = sb.tile((srows, tokens * _Q_COLS), bf16)
    nisa.tensor_scalar(dst=cur_moving, data=cur_ind, op0=nl.equal, operand0=seq_f)

    # RoPE columns [64, cur_rows]: cos and sin of the current rows, the head dim on partitions.
    rope_t = []
    for half in range(2):
        col_psum = nl.ndarray((_HEAD_DIM, cur_rows), dtype=fp32, buffer=nl.psum)
        nisa.nc_transpose(dst=col_psum, data=cur_rope[0:cur_rows, half * _HEAD_DIM : (half + 1) * _HEAD_DIM])
        col = sb.tile((_HEAD_DIM, cur_rows), fp32)
        nisa.tensor_copy(dst=col, src=col_psum)
        rope_t.append(col)

    half_ones = sb.tile((_HEAD_DIM, _HEAD_DIM), fp32)
    nisa.memset(dst=half_ones, value=1.0)
    xcat = sb.tile((P_MAX, cur_rows, _H_TILES), bf16)
    nisa.memset(dst=xcat, value=0.0)
    ones_bf = sb.tile((P_MAX, 1), bf16)
    nisa.memset(dst=ones_bf, value=1.0)
    ones_cur = sb.tile((cur_rows, 1), fp32)
    nisa.memset(dst=ones_cur, value=1.0)
    return {
        "page_bias": page_biases,
        "cur_bias": cur_bias,
        "block_ids": block_ids_all,
        "cos_t": rope_t[0],
        "sin_t": rope_t[1],
        "rotation": _rotation_matrix(sb),
        "head_ones": half_ones,
        "own_select": _own_select(tokens, sb),
        "page_moving": [_bias_moving(sb) for _ in range(tokens)],
        "cur_moving": cur_moving,
        "xcat": xcat,
        "ones_bf": ones_bf,
        "ones_cur": ones_cur,
    }


def _norm_rope(xt_psum, groups, rows, w_col, consts, eps, sb):
    """Return ``rope(bf16(bf16(x * rsqrt(mean_d(x^2) + eps)) * w))`` BF16 ``[64, groups, rows]`` of a transposed tile.

    ``xt_psum`` is a ``[64, groups, rows]`` BF16 PSUM view: the head dim on partitions, one column per (group, row). The
    head-dim sums and ``rotate_half`` are matmuls with constant stationaries. Row ``r`` takes RoPE column ``r``.
    """
    fp32 = nl.float32
    bf16 = nl.bfloat16
    parts = _HEAD_DIM
    width = groups * rows
    x32 = sb.tile((parts, width), fp32)
    nisa.tensor_copy(dst=x32.reshape((parts, groups, rows)), src=xt_psum)
    sq = sb.tile((parts, width), fp32)
    nisa.tensor_tensor(dst=sq, data1=x32, data2=x32, op=nl.multiply)
    ss_psum = nl.ndarray((parts, width), dtype=fp32, buffer=nl.psum)
    nisa.nc_matmul(dst=ss_psum, stationary=consts["head_ones"], moving=sq)
    rstd = sb.tile((parts, width), fp32)
    nisa.activation(dst=rstd, op=nl.rsqrt, data=ss_psum, scale=1.0 / float(_HEAD_DIM), bias=eps)
    xr = sb.tile((parts, width), bf16)
    nisa.tensor_tensor(dst=xr, data1=x32, data2=rstd, op=nl.multiply)
    xn = sb.tile((parts, width), bf16)
    nisa.tensor_scalar(dst=xn, data=xr, op0=nl.multiply, operand0=w_col)
    rot_psum = nl.ndarray((parts, width), dtype=fp32, buffer=nl.psum)
    nisa.nc_matmul(dst=rot_psum, stationary=consts["rotation"], moving=xn)
    cos_b = consts["cos_t"][0:parts, 0:rows].expand_dim(1).broadcast(1, groups)
    sin_b = consts["sin_t"][0:parts, 0:rows].expand_dim(1).broadcast(1, groups)
    a = sb.tile((parts, groups, rows), fp32)
    nisa.tensor_tensor(dst=a, data1=xn.reshape((parts, groups, rows)), data2=cos_b, op=nl.multiply)
    b = sb.tile((parts, groups, rows), fp32)
    nisa.tensor_tensor(dst=b, data1=rot_psum.reshape((parts, groups, rows)), data2=sin_b, op=nl.multiply)
    out = sb.tile((parts, groups, rows), bf16)
    nisa.tensor_tensor(dst=out, data1=a, data2=b, op=nl.add)
    return out


def _load_column(src, sb, tag):
    """Load the first 64 values of an HBM ``[128]`` FP32 vector as a per-partition ``[64, 1]`` column."""
    col = sb.tile((_HEAD_DIM, 1), nl.float32)
    nisa.dma_copy(
        dst=col,
        src=src.ap(pattern=[[1, _HEAD_DIM], [1, 1]], offset=0),
        dge_mode=nisa.dge_mode.hwdge,
        engine=nisa.engine.sync,
        name=f"{tag}_col",
    )
    return col


def _seq_row(src, tokens, tag, sb):
    """Load the HBM ``[B, 1]`` INT32 ``src`` as one SBUF row ``[1, B]``."""
    row = sb.tile((1, tokens), nl.int32)
    view = src[0:1, 0:1] if tokens == 1 else src.ap(pattern=[[tokens, 1], [1, tokens]], offset=0)
    nisa.dma_copy(dst=row, src=view, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync, name=tag)
    return row


def gqa_write_rows(verify_slots, row0, rank_info, sb):
    """Return the ``[8B, 1]`` INT32 pool row of each verify row: its slot on the owner slice, else 0 (the null page).

    Verify row ``j B + b`` is position ``row0[b] + j``, and its owner is slice ``((row0[b] + j) >> 7) & 7``; a slot below
    4096 (page 0) is never written.
    """
    fp32 = nl.float32
    int32 = nl.int32
    hwdge = nisa.dge_mode.hwdge
    sync = nisa.engine.sync
    tokens = row0.shape[0]
    rows = _VERIFY * tokens
    pos_row = _seq_row(row0, tokens, "gqa_w_row0", sb)
    own_pos = _own_positions(pos_row, _VERIFY, tokens, sb)
    owner_i = sb.tile((rows, 1), int32)
    nisa.tensor_scalar(dst=owner_i, data=own_pos, op0=nl.right_shift, operand0=7, op1=nl.bitwise_and, operand1=_CP8 - 1)
    slot_i = sb.tile((rows, 1), int32)
    nisa.dma_copy(dst=slot_i, src=verify_slots[0:rows, 0:1], dge_mode=hwdge, engine=sync, name="gqa_w_slot")
    c_f, _is_c0, _c_i = _rank_scalars(rank_info, rows, sb)
    token_f = sb.tile((rows, 2), fp32)
    nisa.tensor_copy(dst=token_f[0:rows, 0:1], src=owner_i)
    nisa.tensor_copy(dst=token_f[0:rows, 1:2], src=slot_i)
    owner = sb.tile((rows, 1), fp32)
    nisa.tensor_scalar(dst=owner, data=token_f[0:rows, 0:1], op0=nl.equal, operand0=c_f)
    write_ok = sb.tile((rows, 1), fp32)
    nisa.tensor_scalar(
        dst=write_ok,
        data=token_f[0:rows, 1:2],
        op0=nl.greater_equal,
        operand0=float(_PAGE_ROWS),
        op1=nl.multiply,
        operand1=owner,
    )
    write_mask = sb.tile((rows, 1), int32)
    nisa.tensor_scalar(dst=write_mask, data=write_ok, op0=nl.multiply, operand0=-1.0)
    write_row = sb.tile((rows, 1), int32)
    nisa.tensor_tensor(dst=write_row, data1=slot_i, data2=write_mask, op=nl.bitwise_and)
    return write_row


def gqa_block_attention(
    x_hT,
    qkv_proj,
    q_norm,
    k_norm,
    o_proj,
    pool,
    write_row,
    consts,
    groups,
    tag,
    sb=None,
    sources=None,
    softmax_scale=0.125,
    eps=_EPS,
    debug=None,
):
    """One drafter layer's attention in draft mode (TP16 x CP8); return the ``[128, 7B, 56]`` BF16 hT output.

    Args:
        x_hT: ``[128, 7B, 56]`` BF16 SBUF ``input_norm(h)`` of the block rows (row ``t B + b``).
        qkv_proj: ``[128, 56, 384]`` BF16 HBM ``[q heads 4h..4h+3 | k_h | v_h]`` of this rank's head, hT-tiled.
        q_norm, k_norm: ``[128]`` FP32 HBM ``cat(w, w)``; the kernel reads the first 64 values.
        o_proj: ``[64, 4, 896]`` BF16 HBM ``[d, a, n] = o_proj[896 c + n, (4 h + a) * 64 + d]``.
        pool: ``[pages, 4096 * 128]`` BF16 or FLOAT8_E4M3FN HBM rider pages of this layer and rank.
        write_row: ``[8B, 1]`` INT32 SBUF pool row of each verify row on its owner slice, else 0 (``gqa_write_rows``).
        consts: ``gqa_step_constants``; ``consts["xcat"]`` holds the context feature of the verify rows from row
            ``verify_row0`` (``gqa_current_rows``).
        groups: ``(cp8, tp16)`` replica groups (``gqa_replica_groups``).
        tag, sb: a name prefix and the SBUF allocator.
        sources: the persistent per-rank HBM collective tensors (``gqa_collective_sources``).
        debug: optional dict of HBM tensors (test only, B = 1): ``state`` ``[65, 28]`` FP32 (the numerator and the sums
            of this slice before the merge), ``negref`` ``[1, 28]`` BF16, and ``attn`` ``[64, 28]`` BF16 after the merge.
    """
    sb = AUTO_SBUF if sb is None else sb
    fp32 = nl.float32
    bf16 = nl.bfloat16
    hwdge = nisa.dge_mode.hwdge
    sync = nisa.engine.sync
    scale = float(softmax_scale)
    cap = _WEIGHT_CAP / scale
    cp8_group, tp16_group = groups
    tokens = x_hT.shape[1] // _BLOCK
    verify_row0, cur_rows = gqa_current_rows(tokens)
    nq = _BLOCK * tokens
    q_cols_all = _Q_COLS * tokens

    # Projection of this rank's head for the block rows (0 .. 7B - 1) and the verify rows (from verify_row0).
    xcat = consts["xcat"]
    nisa.tensor_copy(dst=xcat[0:P_MAX, 0:nq, 0:_H_TILES], src=x_hT)
    qkv = _matmul_tok(xcat, qkv_proj, _QKV_COLS, f"{tag}_qkv", sb)
    qkv3 = qkv.reshape((cur_rows, _Q_HEADS + 2, _HEAD_DIM))

    # Head dim on partitions: q [64, 4 heads x 7B rows] and k [64, cur_rows], then q_norm / k_norm and the RoPE.
    # Each head's rows take an even number of BF16 PSUM columns, so every transpose output starts 4-byte aligned.
    q_psum = nl.ndarray((_HEAD_DIM, _Q_HEADS, nq + nq % 2), dtype=bf16, buffer=nl.psum)
    for a in range(_Q_HEADS):
        nisa.nc_transpose(dst=q_psum[0:_HEAD_DIM, a, 0:nq], data=qkv3[0:nq, a, 0:_HEAD_DIM])
    k_psum = nl.ndarray((_HEAD_DIM, 1, cur_rows), dtype=bf16, buffer=nl.psum)
    nisa.nc_transpose(dst=k_psum[0:_HEAD_DIM, 0, 0:cur_rows], data=qkv3[0:cur_rows, _Q_HEADS, 0:_HEAD_DIM])
    q_view = q_psum[0:_HEAD_DIM, 0:_Q_HEADS, 0:nq]
    q_t = _norm_rope(q_view, _Q_HEADS, nq, _load_column(q_norm, sb, f"{tag}_qn"), consts, eps, sb)
    k_t = _norm_rope(k_psum, 1, cur_rows, _load_column(k_norm, sb, f"{tag}_kn"), consts, eps, sb)
    if tokens == 1:
        q_cols = q_t.reshape((_HEAD_DIM, _Q_COLS))
    else:
        # Sequence-major query columns: column 28 b + 7 a + t is head a, block row t of sequence b.
        q_cols = sb.tile((_HEAD_DIM, q_cols_all), bf16)
        nisa.tensor_copy(
            dst=q_cols.reshape((_HEAD_DIM, tokens, _Q_HEADS, _BLOCK)),
            src=q_t.reshape((_HEAD_DIM, _Q_HEADS, _BLOCK, tokens)).permute((0, 3, 1, 2)),
        )
    k_cols = k_t.reshape((_HEAD_DIM, cur_rows))
    v_cur = qkv3[0:cur_rows, _Q_HEADS + 1, 0:_HEAD_DIM]

    # Current keys: the scores give the reference of each query column (its own block key), then the masked weights.
    # One all-pairs tile for all sequences; the bias (row 1 + b of the stationary) masks the keys of other sequences.
    cur_psum = nl.ndarray((cur_rows, q_cols_all), dtype=fp32, buffer=nl.psum)
    nisa.nc_matmul(dst=cur_psum, stationary=k_cols, moving=q_cols, accumulate=False)
    own_part = sb.tile((cur_rows, q_cols_all), fp32)
    nisa.tensor_tensor(dst=own_part, data1=cur_psum, data2=consts["own_select"], op=nl.multiply)
    own_psum = nl.ndarray((1, q_cols_all), dtype=fp32, buffer=nl.psum)
    nisa.nc_matmul(dst=own_psum, stationary=consts["ones_cur"], moving=own_part)
    cur_moving = consts["cur_moving"]
    nisa.tensor_scalar(
        dst=cur_moving[0:1, 0:q_cols_all],
        data=own_psum,
        op0=nl.multiply,
        operand0=-1.0,
        op1=nl.subtract,
        operand1=_MERGE_OFFSET / scale,
    )
    page_moving2 = []
    for b in range(tokens):
        page_moving = consts["page_moving"][b]
        nisa.tensor_copy(
            dst=page_moving[0:1, 0:_KEY_GROUP, 0:_Q_COLS],
            src=cur_moving[0:1, b * _Q_COLS : (b + 1) * _Q_COLS].expand_dim(1).broadcast(1, _KEY_GROUP),
        )
        page_moving2.append(page_moving.reshape((1 + _KEY_GROUP, _GROUP_COLS)))
    nisa.nc_matmul(dst=cur_psum, stationary=consts["cur_bias"], moving=cur_moving, accumulate=True)
    capped = sb.tile((cur_rows, q_cols_all), fp32)
    nisa.tensor_scalar(dst=capped, data=cur_psum, op0=nl.minimum, operand0=cap)
    weights = sb.tile((cur_rows, q_cols_all), bf16)
    nisa.activation(dst=weights, op=nl.exp, data=capped, scale=scale)
    out_psum = nl.ndarray((_HEAD_DIM, q_cols_all), dtype=fp32, buffer=nl.psum)
    nisa.nc_matmul(dst=out_psum, stationary=v_cur, moving=weights, accumulate=False)
    cur_sums = nl.ndarray((1, q_cols_all), dtype=fp32, buffer=nl.psum)
    nisa.nc_matmul(dst=cur_sums, stationary=consts["ones_bf"][0:cur_rows, 0:1], moving=weights)
    sums = sb.tile((1, q_cols_all), fp32)

    # Prior keys of this rank's slice, one sequence at a time over its own pages. A loaded row is clamped finite (a
    # stale or unwritten masked row then weighs exactly 0) and converted to BF16. Key block k is column k of the page
    # tile: 128 keys, one on each partition.
    bound = _FP8_MAX if pool.dtype == nl.float8_e4m3fn else _BF16_KEY_MAX
    page_width = _PART_ROWS * _ROW
    for b in range(tokens):
        sfx = _seq_suffix(b)
        block_ids = consts["block_ids"][b]
        page_bias = consts["page_bias"][b]
        q_seq = q_cols[0:_HEAD_DIM, b * _Q_COLS : (b + 1) * _Q_COLS]
        out_seq = out_psum[0:_HEAD_DIM, b * _Q_COLS : (b + 1) * _Q_COLS]
        nb = block_ids.shape[1]
        page_sums = nl.ndarray((1, _GROUP_COLS), dtype=fp32, buffer=nl.psum)
        for n in range(nb):
            raw = sb.tile((P_MAX, page_width), pool.dtype)
            nisa.dma_copy(
                dst=raw,
                src=pool.ap(
                    pattern=[[page_width, P_MAX], [1, page_width]],
                    offset=0,
                    scalar_offset=block_ids[0:1, n : n + 1],
                    indirect_dim=0,
                ),
                dge_mode=hwdge,
                engine=sync,
                name=f"{tag}_page_{n}{sfx}",
            )
            rows = sb.tile((P_MAX, _PART_ROWS, _ROW), bf16)
            nisa.tensor_scalar(
                dst=rows.reshape((P_MAX, page_width)),
                data=raw,
                op0=nl.maximum,
                operand0=-bound,
                op1=nl.minimum,
                operand1=bound,
                engine=nisa.engine.gpsimd if (b * nb + n) % 2 else nisa.engine.vector,
            )
            for g in range(_PART_ROWS // _KEY_GROUP):
                keys_t = sb.tile((_HEAD_DIM, _KEY_GROUP, P_MAX), bf16)
                for half in range(2):
                    kt_psum = nl.ndarray((_HEAD_DIM, _KEY_GROUP // 2, P_MAX), dtype=bf16, buffer=nl.psum)
                    for kb in range(_KEY_GROUP // 2):
                        k = g * _KEY_GROUP + half * (_KEY_GROUP // 2) + kb
                        nisa.nc_transpose(dst=kt_psum[0:_HEAD_DIM, kb, 0:P_MAX], data=rows[0:P_MAX, k, 0:_HEAD_DIM])
                    first = half * (_KEY_GROUP // 2)
                    nisa.tensor_copy(
                        dst=keys_t[0:_HEAD_DIM, first : first + _KEY_GROUP // 2, 0:P_MAX],
                        src=kt_psum,
                        engine=nisa.engine.scalar if half else nisa.engine.vector,
                    )
                # Each key block's scores, then its bias (the negative reference of each column and the block's key
                # mask: moving row 1 + kb is 1 on block kb's columns) on the same PSUM columns, so the bias accumulates.
                score_psum = nl.ndarray((P_MAX, _GROUP_COLS), dtype=fp32, buffer=nl.psum)
                bias_stationary = page_bias[0 : 1 + _KEY_GROUP, n, g, 0:P_MAX]
                for kb in range(_KEY_GROUP):
                    cols = slice(kb * _Q_COLS, (kb + 1) * _Q_COLS)
                    nisa.nc_matmul(
                        dst=score_psum[0:P_MAX, cols],
                        stationary=keys_t[0:_HEAD_DIM, kb, 0:P_MAX],
                        moving=q_seq,
                        accumulate=False,
                    )
                    nisa.nc_matmul(
                        dst=score_psum[0:P_MAX, cols],
                        stationary=bias_stationary,
                        moving=page_moving2[b][0 : 1 + _KEY_GROUP, cols],
                        accumulate=True,
                    )
                page_capped = sb.tile((P_MAX, _GROUP_COLS), fp32)
                nisa.tensor_scalar(dst=page_capped, data=score_psum, op0=nl.minimum, operand0=cap)
                page_weights = sb.tile((P_MAX, _GROUP_COLS), bf16)
                nisa.activation(dst=page_weights, op=nl.exp, data=page_capped, scale=scale)
                for kb in range(_KEY_GROUP):
                    k = g * _KEY_GROUP + kb
                    cols = slice(kb * _Q_COLS, (kb + 1) * _Q_COLS)
                    nisa.nc_matmul(
                        dst=out_seq,
                        stationary=rows[0:P_MAX, k, _HEAD_DIM:_ROW],
                        moving=page_weights[0:P_MAX, cols],
                        accumulate=True,
                    )
                first_group = n == 0 and g == 0
                nisa.nc_matmul(
                    dst=page_sums, stationary=consts["ones_bf"], moving=page_weights, accumulate=not first_group
                )

        # The sums of this sequence: fold the 16 key-block sums of each query column (a halving tree of adds), then
        # add the current-key sums.
        sums_all = sb.tile((1, _KEY_GROUP, _Q_COLS), fp32)
        if nb:
            nisa.tensor_copy(dst=sums_all.reshape((1, _GROUP_COLS)), src=page_sums)
        else:
            nisa.memset(dst=sums_all, value=0.0)
        fold = sums_all
        width = _KEY_GROUP
        while width > 1:
            width //= 2
            half_sums = sb.tile((1, width, _Q_COLS), fp32)
            nisa.tensor_tensor(
                dst=half_sums,
                data1=fold[0:1, 0:width, 0:_Q_COLS],
                data2=fold[0:1, width : 2 * width, 0:_Q_COLS],
                op=nl.add,
            )
            fold = half_sums
        cols = slice(b * _Q_COLS, (b + 1) * _Q_COLS)
        nisa.tensor_tensor(dst=sums[0:1, cols], data1=fold.reshape((1, _Q_COLS)), data2=cur_sums[0:1, cols], op=nl.add)

    # Merge: the numerator [64, 28B] and the sums [28B] of this slice; a CP8 all-reduce adds the 8 slices.
    state_src = sources["state_src_hbm"]
    state_dst = sources["state_dst_hbm"]
    num = sb.tile((_HEAD_DIM, q_cols_all), fp32)
    nisa.tensor_copy(dst=num, src=out_psum)
    if debug is not None:
        nisa.dma_copy(dst=debug["state"][0:_HEAD_DIM, 0:_Q_COLS], src=num, name=f"{tag}_dbg_num")
        nisa.dma_copy(dst=debug["state"][_HEAD_DIM : _HEAD_DIM + 1, 0:_Q_COLS], src=sums, name=f"{tag}_dbg_sums")
        nisa.dma_copy(dst=debug["negref"], src=cur_moving[0:1, 0:_Q_COLS], name=f"{tag}_dbg_negref")
    nisa.dma_copy(dst=state_src[0:_HEAD_DIM, 0:q_cols_all], src=num, name=f"{tag}_state_store")
    nisa.dma_copy(dst=state_src[_HEAD_DIM : _HEAD_DIM + 1, 0:q_cols_all], src=sums, name=f"{tag}_sums_store")
    ncc.all_reduce(srcs=[state_src], dsts=[state_dst], replica_group=cp8_group, op=nl.add, name=f"{tag}_state_ar")
    merged = sb.tile((_HEAD_DIM, q_cols_all), fp32)
    nisa.dma_copy(dst=merged, src=state_dst[0:_HEAD_DIM, 0:q_cols_all], name=f"{tag}_state_load")
    den = sb.tile((_HEAD_DIM, q_cols_all), fp32)
    nisa.dma_copy(
        dst=den,
        src=state_dst.ap(pattern=[[0, _HEAD_DIM], [1, q_cols_all]], offset=_HEAD_DIM * q_cols_all),
        name=f"{tag}_den_load",
    )
    recip = sb.tile((_HEAD_DIM, q_cols_all), fp32)
    nisa.reciprocal(dst=recip, data=den)
    attn = sb.tile((_HEAD_DIM, q_cols_all), bf16)
    nisa.tensor_tensor(dst=attn, data1=merged, data2=recip, op=nl.multiply)
    if debug is not None:
        nisa.dma_copy(dst=debug["attn"], src=attn, name=f"{tag}_dbg_attn")
    if tokens == 1:
        attn_tok = attn.reshape((_HEAD_DIM, _Q_HEADS, _BLOCK))
    else:
        # Head-major, token-major columns for o_proj: column (a, t B + b).
        attn_tok = sb.tile((_HEAD_DIM, _Q_HEADS, nq), bf16)
        nisa.tensor_copy(
            dst=attn_tok.reshape((_HEAD_DIM, _Q_HEADS, _BLOCK, tokens)),
            src=attn.reshape((_HEAD_DIM, tokens, _Q_HEADS, _BLOCK)).permute((0, 2, 3, 1)),
        )

    # o_proj: head h's 256 inputs into this slice's 896 output columns (7 hT tiles), a TP16 all-reduce over the 16
    # heads, and a CP8 all-gather of the 8 column blocks.
    o_w = sb.tile((_HEAD_DIM, _Q_HEADS, _O_COLS), bf16)
    nisa.dma_copy(
        dst=o_w, src=o_proj[0:_HEAD_DIM, 0:_Q_HEADS, 0:_O_COLS], dge_mode=hwdge, engine=sync, name=f"{tag}_o_w"
    )
    o_psum = nl.ndarray((P_MAX, _O_TILES, nq), dtype=fp32, buffer=nl.psum)
    for tile in range(_O_TILES):
        for a in range(_Q_HEADS):
            nisa.nc_matmul(
                dst=o_psum[0:P_MAX, tile, 0:nq],
                stationary=o_w[0:_HEAD_DIM, a, tile * P_MAX : (tile + 1) * P_MAX],
                moving=attn_tok[0:_HEAD_DIM, a, 0:nq],
                accumulate=a > 0,
            )
    o_part = sb.tile((P_MAX, _O_TILES * nq), fp32)
    nisa.tensor_copy(dst=o_part, src=o_psum.reshape((P_MAX, _O_TILES * nq)))
    o_src, o_sum, o_all = sources["o_src_hbm"], sources["o_sum_hbm"], sources["o_all_hbm"]
    nisa.dma_copy(dst=o_src, src=o_part, name=f"{tag}_o_store")
    ncc.all_reduce(srcs=[o_src], dsts=[o_sum], replica_group=tp16_group, op=nl.add, name=f"{tag}_o_ar")
    ncc.all_gather(srcs=[o_sum], dsts=[o_all], replica_group=cp8_group, collective_dim=0, name=f"{tag}_o_ag")
    o_rows = sb.tile((P_MAX, _CP8, _O_TILES * nq), fp32)
    nisa.dma_copy(
        dst=o_rows,
        src=o_all.ap(pattern=[[_O_TILES * nq, P_MAX], [P_MAX * _O_TILES * nq, _CP8], [1, _O_TILES * nq]], offset=0),
        name=f"{tag}_o_load",
    )
    attn_hT = sb.tile((P_MAX, nq, _H_TILES), bf16)
    nisa.tensor_copy(dst=attn_hT, src=o_rows.reshape((P_MAX, _H_TILES, nq)).permute((0, 2, 1)))

    # Rider rows of the verify positions: [K_h | V_h] of the verify rows, written by the owner slice (others write row
    # 0 of the null page).
    kw_psum = nl.ndarray((cur_rows, _HEAD_DIM), dtype=bf16, buffer=nl.psum)
    nisa.nc_transpose(dst=kw_psum, data=k_cols)
    v0, v1 = verify_row0, cur_rows
    rider = sb.tile((cur_rows, _ROW), pool.dtype)
    nisa.tensor_scalar(
        dst=rider[v0:v1, 0:_HEAD_DIM],
        data=kw_psum[v0:v1, 0:_HEAD_DIM],
        op0=nl.minimum,
        operand0=bound,
        op1=nl.maximum,
        operand1=-bound,
    )
    nisa.tensor_scalar(
        dst=rider[v0:v1, _HEAD_DIM:_ROW],
        data=qkv3[v0:v1, _Q_HEADS + 1, 0:_HEAD_DIM],
        op0=nl.minimum,
        operand0=bound,
        op1=nl.maximum,
        operand1=-bound,
    )
    pool_rows = pool.reshape((pool.shape[0] * _PAGE_ROWS, _ROW))
    for j in range(_VERIFY * tokens):
        nisa.dma_copy(
            dst=pool_rows.ap(
                pattern=[[_ROW, 1], [1, _ROW]], offset=0, scalar_offset=write_row[j : j + 1, 0:1], indirect_dim=0
            ),
            src=rider[v0 + j : v0 + j + 1, 0:_ROW],
            dge_mode=hwdge,
            engine=sync,
            name=f"{tag}_rider_write_{j}",
        )
    return attn_hT


def gqa_collective_sources(tokens=1):
    """Return the persistent per-rank private HBM tensors of the attention collectives, one set for all layers.

    Each collective has its own source and destination tensor, so no later write reuses a buffer that a peer still
    reads; the layers serialize on the MLP all-reduce between them. Every tensor carries all B sequences.
    """
    fp32 = nl.float32
    q_cols = _Q_COLS * tokens
    o_cols = _O_TILES * _BLOCK * tokens

    def hbm(shape, name):
        return nl.ndarray(shape, dtype=fp32, buffer=nl.private_hbm, name=name)

    return {
        "state_src_hbm": hbm((_HEAD_DIM + 1, q_cols), "gqa_state_src"),
        "state_dst_hbm": hbm((_HEAD_DIM + 1, q_cols), "gqa_state_dst"),
        "o_src_hbm": hbm((P_MAX, o_cols), "gqa_o_src"),
        "o_sum_hbm": hbm((P_MAX, o_cols), "gqa_o_sum"),
        "o_all_hbm": hbm((_CP8 * P_MAX, o_cols), "gqa_o_all"),
    }


def gqa_current_rope(rope_window, rope_base, row0, accepted, sb):
    """Return the ``[cur_rows, 128]`` FP32 neox RoPE rows of the current rows.

    Block row ``t B + b`` is position ``P[b] + t`` and verify row ``verify_row0 + j B + b`` is position ``row0[b] + j``
    (``gqa_current_rows``); the gap rows are zero. ``rope_window`` is ``[B, W, 128]`` FP32 HBM ``[cos | sin]`` rows of
    sequence ``b`` from position ``rope_base[b]``.
    """
    int32 = nl.int32
    hwdge = nisa.dge_mode.hwdge
    sync = nisa.engine.sync
    tokens = row0.shape[0]
    verify_row0, cur_rows = gqa_current_rows(tokens)
    if tokens == 1:
        rows_i = sb.tile((1, 3), int32)
        for c, src in enumerate((row0, accepted, rope_base)):
            nisa.dma_copy(
                dst=rows_i[0:1, c : c + 1], src=src[0:1, 0:1], dge_mode=hwdge, engine=sync, name=f"gqa_rope_scal_{c}"
            )
        ctx_pos = sb.tile((1, 1), int32)
        nisa.tensor_tensor(dst=ctx_pos, data1=rows_i[0:1, 0:1], data2=rows_i[0:1, 2:3], op=nl.subtract)
        acc_row = rows_i[0:1, 1:2]
    else:
        # Flat window row of sequence b: b W + position - rope_base[b].
        row0_row = _seq_row(row0, tokens, "gqa_rope_scal_0", sb)
        acc_row = _seq_row(accepted, tokens, "gqa_rope_scal_1", sb)
        base_row = _seq_row(rope_base, tokens, "gqa_rope_scal_2", sb)
        bw = sb.tile((1, tokens), int32)
        nisa.iota(dst=bw, pattern=[[rope_window.shape[1], tokens]], offset=0)
        bw_minus_base = sb.tile((1, tokens), int32)
        nisa.tensor_tensor(dst=bw_minus_base, data1=bw, data2=base_row, op=nl.subtract)
        ctx_pos = sb.tile((1, tokens), int32)
        nisa.tensor_tensor(dst=ctx_pos, data1=row0_row, data2=bw_minus_base, op=nl.add)
    blk_pos = sb.tile((1, tokens), int32)
    nisa.tensor_tensor(dst=blk_pos, data1=ctx_pos, data2=acc_row, op=nl.add)
    cur_rope = sb.tile((cur_rows, 2 * _HEAD_DIM), nl.float32)
    nisa.memset(dst=cur_rope, value=0.0)
    nq, nv = _BLOCK * tokens, _VERIFY * tokens
    _rope_column_rows(rope_window, _own_positions(blk_pos, _BLOCK, tokens, sb), 0, nq, cur_rope, "gqa_blk")
    _rope_column_rows(rope_window, _own_positions(ctx_pos, _VERIFY, tokens, sb), verify_row0, nv, cur_rope, "gqa_ctx")
    return cur_rope


def gqa_context_feature(aux_hT, context_proj, context_norm, tp_group, tag, sb, sources):
    """Return the context feature ``hidden_norm(fc(cat taps))`` ``[128, 8, 56]`` BF16 hT of the verify rows.

    ``aux_hT`` is ``[128, 8, 280]`` BF16 SBUF; ``context_proj`` is this rank's ``[128, 280, 56]`` output shard. The
    world all-gather of the shards runs in the persistent ``ctx_src`` / ``ctx_dst`` tiles.
    """
    fc_local = _matmul_tok(aux_hT, context_proj, HIDDEN // CP, f"{tag}_fc", sb, out=sources["ctx_src"])
    fc_full = sources["ctx_dst"]
    ncc.all_gather(srcs=[fc_local], dsts=[fc_full], replica_group=tp_group, collective_dim=1, name=f"{tag}_fc_ag")
    norm_hT = load_vector_hT(context_norm, dtype=nl.float32)
    return rmsnorm_hT(_rows_to_hT(fc_full, sb), norm_hT, _EPS, new_ones(), sb=sb)


def draft_gqa_layer(
    h_hT,
    qkv_proj,
    q_norm,
    k_norm,
    o_proj,
    input_norm,
    post_norm,
    dense_gate_up,
    dense_down,
    pool,
    write_row,
    consts,
    tp_group,
    groups,
    tag,
    sb,
    sources,
    softmax_scale,
    ones,
):
    """Drafter layer: ``h += attention(input_norm(h))``, then ``h += silu_mlp(post_attention_norm(h))``.

    The MLP is ``draft_mla._silu_mlp`` with the TP128 intermediate shard (112 per rank) and its HBM all-reduce.
    """
    fp32 = nl.float32
    bf16 = nl.bfloat16
    x_hT = rmsnorm_hT(h_hT, load_vector_hT(input_norm, dtype=fp32), _EPS, ones, sb=sb)
    attn = gqa_block_attention(
        x_hT,
        qkv_proj,
        q_norm,
        k_norm,
        o_proj,
        pool,
        write_row,
        consts,
        groups,
        tag,
        sb=sb,
        sources=sources,
        softmax_scale=softmax_scale,
    )
    h_attn = sb.tile(tuple(h_hT.shape), bf16)
    nisa.tensor_tensor(dst=h_attn, data1=h_hT, data2=attn, op=nl.add)
    x_hT = rmsnorm_hT(h_attn, load_vector_hT(post_norm, dtype=fp32), _EPS, ones, sb=sb)
    mlp = _silu_mlp(
        x_hT,
        dense_gate_up,
        dense_down,
        tp_group,
        f"{tag}_mlp",
        sb,
        ar_src_hbm=sources["mlp_src_hbm"],
        ar_dst_hbm=sources["mlp_dst_hbm"],
        cp=CP,
    )
    h_out = sb.tile(tuple(h_hT.shape), bf16)
    nisa.tensor_tensor(dst=h_out, data1=h_attn, data2=mlp, op=nl.add)
    return h_out


@nki.jit
def kimi_k3_draft_gqa_step(
    aux: nl.NkiTensor,
    bonus: nl.NkiTensor,
    accepted: nl.NkiTensor,
    row0: nl.NkiTensor,
    draft_lo: nl.NkiTensor,
    rider_pool: nl.NkiTensor,
    block_table: nl.NkiTensor,
    verify_slots: nl.NkiTensor,
    rope_window: nl.NkiTensor,
    rope_base: nl.NkiTensor,
    context_proj: nl.NkiTensor,
    context_norm: nl.NkiTensor,
    qkv_proj: nl.NkiTensor,
    q_norm: nl.NkiTensor,
    k_norm: nl.NkiTensor,
    o_proj: nl.NkiTensor,
    input_norm: nl.NkiTensor,
    post_norm: nl.NkiTensor,
    dense_gate_up: nl.NkiTensor,
    dense_down: nl.NkiTensor,
    final_norm: nl.NkiTensor,
    lm_head: nl.NkiTensor,
    markov_w1: nl.NkiTensor,
    markov_w2: nl.NkiTensor,
    embed: nl.NkiTensor = None,
    block_embed: nl.NkiTensor = None,
    rank_info: nl.NkiTensor = None,
    tp_group: ReplicaGroup = None,
    softmax_scale: float = 0.125,
    layers: int = _LAYERS,
    window: int = 0,
    mask_token: int = _MASK_TOKEN,
    return_hidden: bool = False,
    return_layer_hidden: bool = False,
    confidence_h: nl.NkiTensor = None,
    confidence_m: nl.NkiTensor = None,
    confidence_bias: nl.NkiTensor = None,
):
    """Run one Kimi-K3 GQA draft step for B sequences (1 <= B <= 8) on one of 128 ranks; return ``[B, 7]`` INT32 drafts.

    The per-layer weights are stacked ``[5, ...]`` (by the host weight builder of the GQA drafter). The rows of the B
    sequences are token-major: aux row ``j B + b``, block row ``t B + b``.

    Args:
        aux: ``[5, 8B, 7168]`` BF16 HBM taps of the 8 verify rows of each sequence (positions ``row0 .. row0 + 7``).
        bonus, accepted, row0, draft_lo: ``[B, 1]`` INT32 HBM anchor token, accepted count (``P = row0 + accepted``),
            verify row-0 position, and the lower bound of the prior keys.
        rider_pool: ``[5 KP, 4096 * 128]`` FP8 (or BF16) HBM, the 5 layers' pages of this rank stacked; a BF16
            ``[5 KP, 4096 * 64]`` holds FP8 pairs (``gqa_rider_pool``).
        block_table: ``[B, NB]`` INT32 HBM page of each 32768-token block. Two sequences can name one page that both
            only read; the pages that a sequence writes are its own.
        verify_slots: ``[8B, 1]`` INT32 HBM pool row ``page * 4096 + i`` of each verify row on its owner slice
            (``< 4096``: no write). The written rows of different sequences are different.
        rope_window: ``[B, W, 128]`` FP32 HBM neox ``[cos | sin]`` rows from position ``rope_base[b]``; each window
            covers ``row0 .. P + 6`` of its sequence.
        context_proj .. markov_w2, embed: the rank weights. ``block_embed`` ``[7B, 7168]`` replaces the embed gather.
        rank_info, tp_group, softmax_scale: the rank id (column 2), the world group, and the softmax scale.
        layers, window, mask_token: the layer count, the context window (0 = full), and the mask token id.
        confidence_h, confidence_m, confidence_bias: the optional confidence head; with them the call also returns
            the ``[7B, 1]`` FP32 confidence logits.
    """
    sb = AUTO_SBUF
    bf16 = nl.bfloat16
    fp32 = nl.float32
    int32 = nl.int32
    hwdge = nisa.dge_mode.hwdge
    sync = nisa.engine.sync
    tokens = row0.shape[0]
    kernel_assert(1 <= tokens <= _MAX_SEQS, f"the GQA drafter step runs 1 <= B <= {_MAX_SEQS}, got {tokens}")
    total_v, total_q = _VERIFY * tokens, _BLOCK * tokens
    verify_row0, cur_rows = gqa_current_rows(tokens)
    pool_all = gqa_rider_pool(rider_pool)
    layer_pages = pool_all.shape[0] // layers
    pools = [
        pool_all[layer * layer_pages : (layer + 1) * layer_pages, 0 : _PAGE_ROWS * _ROW] for layer in range(layers)
    ]
    groups = gqa_replica_groups()
    v_s = lm_head.shape[1]

    # Persistent collective tiles: region heap tiles to the kernel end (own tensors, fixed addresses), and per-rank
    # private HBM for the all-reduces.
    region = ExplicitSbuf()
    install_identities(region)
    heap = SbufTiles(region, heap=True)
    sources = gqa_collective_sources(tokens)
    # The SBUF gather sources and destinations are views of two arenas (sources, destinations), as the MLA drafter
    # step: the tail argmax source is the last collective's source, so the source arena stays reserved to the kernel
    # end and no later tile takes the bytes of a gather source that a peer still reads.
    plan = _ArenaPlan()
    slots = {
        "ctx_src": plan.tile((total_v, HIDDEN // CP), bf16, "ctx_ag_src"),
        "emb_src": plan.tile((total_q, HIDDEN // CP), bf16, "emb_ag_src"),
        "tail_src": plan.tile((tokens, 2), fp32, "tail_ag_src"),
        "ctx_dst": plan.tile((total_v, HIDDEN), bf16, "ctx_ag_dst"),
        "emb_dst": plan.tile((total_q, HIDDEN), bf16, "emb_ag_dst"),
    }
    arenas = tuple(region.heap((P_MAX, w), bf16, f"gqa_arena_{i}") for i, w in enumerate(plan.widths))
    sources.update({key: plan.view(arenas, slot) for key, slot in slots.items()})
    # The argmax destination is reshaped by the argmax (a standalone tile keeps a valid partition step at B = 1).
    sources["tail_dst"] = heap.tile((tokens, 2 * CP), fp32, "tail_ag_dst")
    sources["markov_bias"] = heap.tile((tokens, v_s), bf16, "markov_bias")
    for base in (*arenas, sources["tail_dst"]):
        nisa.memset(dst=base, value=0.0)
    mlp_cols = total_q * _H_TILES
    sources["mlp_src_hbm"] = nl.ndarray((P_MAX, mlp_cols), dtype=fp32, buffer=nl.private_hbm, name="gqa_mlp_ar_src")
    sources["mlp_dst_hbm"] = nl.ndarray((P_MAX, mlp_cols), dtype=fp32, buffer=nl.private_hbm, name="gqa_mlp_ar_dst")
    sources["markov_src_hbm"] = nl.ndarray((tokens, _MARKOV_RANK), dtype=bf16, buffer=nl.private_hbm, name="gqa_mk_src")
    sources["markov_dst_hbm"] = nl.ndarray((tokens, _MARKOV_RANK), dtype=bf16, buffer=nl.private_hbm, name="gqa_mk_dst")

    # P = row0 + accepted; key_start = max(draft_lo, P - window).
    p_tile = sb.tile((tokens, 1), int32)
    acc_tile = sb.tile((tokens, 1), int32)
    nisa.dma_copy(dst=p_tile, src=row0[0:tokens, 0:1], dge_mode=hwdge, engine=sync, name="gqa_row0")
    nisa.dma_copy(dst=acc_tile, src=accepted[0:tokens, 0:1], dge_mode=hwdge, engine=sync, name="gqa_acc")
    nisa.tensor_tensor(dst=p_tile, data1=p_tile, data2=acc_tile, op=nl.add)
    key_start = _compute_key_start(draft_lo, p_tile, window, tokens, "gqa_key_start", sb)
    cur_rope = gqa_current_rope(rope_window, rope_base, row0, accepted, sb)
    nb_table = block_table.shape[1]
    page_count = gqa_page_count(nb_table, window, tokens)
    consts = gqa_step_constants(
        row0, accepted, key_start, block_table, rank_info, cur_rope, layer_pages, sb, page_count=page_count
    )

    # Context feature of the verify rows (rows 32..39 of the projection input) and their write rows.
    aux_hT = _build_aux_hT(aux, total_v, sb)
    f_hT = gqa_context_feature(aux_hT, context_proj, context_norm, tp_group, "ctx", sb, sources)
    nisa.tensor_copy(dst=consts["xcat"][0:P_MAX, verify_row0:cur_rows, 0:_H_TILES], src=f_hT)
    write_row = gqa_write_rows(verify_slots, row0, rank_info, sb)

    if block_embed is not None:
        h_hT = _load_rows_hT(block_embed, total_q, sb)
    else:
        h_hT = _embed_block(embed, bonus, mask_token, tokens, tp_group, "emb", sb, sources=sources)

    layer_hidden = None
    if return_layer_hidden:
        layer_hidden = nl.ndarray((layers, total_q, HIDDEN), dtype=bf16, buffer=nl.shared_hbm, name="layer_hidden")
    ones = new_ones()
    for layer in range(layers):
        h_hT = draft_gqa_layer(
            h_hT,
            qkv_proj[layer],
            q_norm[layer],
            k_norm[layer],
            o_proj[layer],
            input_norm[layer],
            post_norm[layer],
            dense_gate_up[layer],
            dense_down[layer],
            pools[layer],
            write_row,
            consts,
            tp_group,
            groups,
            f"L{layer}",
            sb,
            sources,
            softmax_scale,
            ones,
        )
        if return_layer_hidden:
            store_hT_rows(layer_hidden[layer], h_hT, sb=sb)

    hidden_out = None
    if return_hidden:
        hidden_out = nl.ndarray((total_q, HIDDEN), dtype=bf16, buffer=nl.shared_hbm, name="hidden")
    confidence = None
    confidence_out = None
    if confidence_h is not None:
        confidence = (confidence_h, confidence_m, confidence_bias)
        confidence_out = nl.ndarray((total_q, 1), dtype=fp32, buffer=nl.shared_hbm, name="confidence")
    drafts = gqa_draft_tail(
        h_hT,
        final_norm,
        lm_head,
        markov_w1,
        markov_w2,
        bonus,
        rank_info,
        tp_group,
        "k4",
        sb=sb,
        hidden_out=hidden_out,
        sources=sources,
        cp=CP,
        confidence=confidence,
        confidence_out=confidence_out,
    )
    out = nl.ndarray((tokens, _BLOCK), dtype=int32, buffer=nl.shared_hbm, name="drafts")
    nisa.dma_copy(dst=out, src=drafts, dge_mode=hwdge, engine=sync, name="drafts_store")
    outputs = [out, rider_pool]
    if return_hidden:
        outputs.append(hidden_out)
    if return_layer_hidden:
        outputs.append(layer_hidden)
    if confidence_out is not None:
        outputs.append(confidence_out)
    return tuple(outputs)
