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

"""Tail of the Kimi-K3 DSpark GQA drafter: the MLA drafter's ``draft_tail`` plus the optional confidence head.

``draft_tail`` runs the final norm, the lm_head vocab shard and the sequential Markov greedy. The confidence head of
the GQA checkpoint is ``logit = proj(cat(final-normed hidden, markov_w1[prev]))`` for each block row, with ``prev``
the anchor for row 0, then the previous draft. It runs after the greedy, on the drafts: the final-normed hidden
comes back from ``hidden_out``, and one owner-masked gather plus one all-reduce give ``markov_w1[prev]`` of every
row. The logit is FP32; the caller applies the sigmoid. Without ``confidence`` the tail is ``draft_tail``.
"""

import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl

from .common import P_MAX
from .draft_mla import _BLOCK, _H_TILES, _MARKOV_RANK, _MARKOV_TILES, _rows_to_hT, draft_tail
from .tkg_sbuf_alloc import AUTO_SBUF
from .mla_decode_torch import HIDDEN, TP


def _load_rows_hT(src, total_rows, tag, sb):
    """Load an HBM ``[T, 7168]`` tensor into the token-major hT tile ``[128, T, 56]`` BF16 (named ops)."""
    out = sb.tile((P_MAX, total_rows, _H_TILES), nl.bfloat16)
    for row in range(total_rows):
        rows = sb.tile((_H_TILES, P_MAX), src.dtype)
        nisa.dma_copy(
            dst=rows, src=src.ap(pattern=[[P_MAX, _H_TILES], [1, P_MAX]], offset=row * HIDDEN), name=f"{tag}_{row}"
        )
        psum = nl.ndarray((P_MAX, _H_TILES), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=psum, data=rows)
        nisa.tensor_copy(dst=out[0:P_MAX, row, 0:_H_TILES], src=psum)
    return out


def _dot_fp32(x_hT, w_sb, sb):
    """Return ``x @ w`` FP32 ``[T, 1]`` for hT ``x`` ``[128, T, K]`` and SBUF ``w`` ``[128, K, 1]`` (no BF16 round)."""
    _, rows, k_tiles = x_hT.shape
    psum = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.psum)
    for k in range(k_tiles):
        nisa.nc_matmul(dst=psum, stationary=x_hT[0:P_MAX, 0:rows, k], moving=w_sb[0:P_MAX, k, 0:1], accumulate=k > 0)
    out = sb.tile((rows, 1), nl.float32)
    nisa.tensor_copy(dst=out, src=psum)
    return out


def _previous_tokens(drafts, bonus, tokens, tag, sb):
    """Return ``prev`` ``[7B, 1]`` INT32 of every block row ``j B + b``: the anchor for ``j = 0``, else draft ``j - 1``."""
    rows = _BLOCK * tokens
    flat = nl.ndarray((rows,), dtype=nl.int32, buffer=nl.private_hbm, name=f"{tag}_prev_rows")
    anchor = sb.tile((tokens, 1), nl.int32)
    nisa.dma_copy(dst=anchor, src=bonus[0:tokens, 0:1], name=f"{tag}_anchor")
    nisa.dma_copy(dst=flat.ap(pattern=[[1, tokens], [1, 1]], offset=0), src=anchor, name=f"{tag}_prev_anchor")
    nisa.dma_copy(
        dst=flat.ap(pattern=[[1, tokens], [tokens, _BLOCK - 1]], offset=tokens),
        src=drafts[0:tokens, 0 : _BLOCK - 1],
        name=f"{tag}_prev_drafts",
    )
    prev = sb.tile((rows, 1), nl.int32)
    nisa.dma_copy(dst=prev, src=flat.ap(pattern=[[1, rows], [1, 1]], offset=0), name=f"{tag}_prev_load")
    return prev


def _markov_rows(prev, markov_w1, rank_info, tp_group, tag, sb):
    """Return ``markov_w1[prev]`` ``[R, 256]`` BF16 of every row: the owner of each id reads it, an HBM all-reduce sums."""
    fp32 = nl.float32
    rows = prev.shape[0]
    v_s = markov_w1.shape[0]
    rank_i = sb.tile((rows, 1), nl.int32)
    nisa.dma_copy(dst=rank_i, src=rank_info[0:1, 2:3].broadcast(0, rows), name=f"{tag}_rank")
    base_f = sb.tile((rows, 1), fp32)
    nisa.tensor_copy(dst=base_f, src=rank_i)
    nisa.tensor_scalar(dst=base_f, data=base_f, op0=nl.multiply, operand0=float(v_s))
    shard_base = sb.tile((rows, 1), nl.int32)
    nisa.tensor_copy(dst=shard_base, src=base_f)
    local_idx = sb.tile((rows, 1), nl.int32)
    nisa.tensor_tensor(dst=local_idx, data1=prev, data2=shard_base, op=nl.subtract)
    clamp = sb.tile((rows, 1), nl.int32)
    nisa.tensor_scalar(dst=clamp, data=local_idx, op0=nl.minimum, operand0=float(v_s - 1), op1=nl.maximum, operand1=0.0)
    li_f = sb.tile((rows, 1), fp32)
    cl_f = sb.tile((rows, 1), fp32)
    nisa.tensor_copy(dst=li_f, src=local_idx)
    nisa.tensor_copy(dst=cl_f, src=clamp)
    is_owner = sb.tile((rows, 1), fp32)
    nisa.tensor_scalar(dst=is_owner, data=li_f, op0=nl.equal, operand0=cl_f)
    local = sb.tile((rows, _MARKOV_RANK), nl.bfloat16)
    for r in range(rows):
        nisa.dma_copy(
            dst=local[r : r + 1, 0:_MARKOV_RANK],
            src=markov_w1.ap(
                pattern=[[_MARKOV_RANK, 1], [1, _MARKOV_RANK]],
                offset=0,
                scalar_offset=clamp[r : r + 1, 0:1],
                indirect_dim=0,
            ),
            name=f"{tag}_w1g_{r}",
        )
    nisa.tensor_scalar(dst=local, data=local, op0=nl.multiply, operand0=is_owner)
    # The all-reduce runs HBM to HBM (store, reduce, load), as the Markov all-reduce of the step.
    src_hbm = nl.ndarray((rows, _MARKOV_RANK), dtype=nl.bfloat16, buffer=nl.private_hbm, name=f"{tag}_w1_src_hbm")
    dst_hbm = nl.ndarray((rows, _MARKOV_RANK), dtype=nl.bfloat16, buffer=nl.private_hbm, name=f"{tag}_w1_dst_hbm")
    nisa.dma_copy(dst=src_hbm, src=local, name=f"{tag}_w1_ar_store")
    ncc.all_reduce(srcs=[src_hbm], dsts=[dst_hbm], replica_group=tp_group, op=nl.add, name=f"{tag}_w1_all_reduce")
    summed = sb.tile((rows, _MARKOV_RANK), nl.bfloat16)
    nisa.dma_copy(dst=summed, src=dst_hbm, name=f"{tag}_w1_ar_load")
    return summed


def gqa_draft_tail(
    h_hT,
    final_norm,
    lm_head_weight,
    markov_w1,
    markov_w2,
    bonus,
    rank_info,
    tp_group,
    tag,
    sb=None,
    hidden_out=None,
    sources=None,
    cp=TP,
    confidence=None,
    confidence_out=None,
):
    """``draft_tail``, then the optional confidence head. Return the drafts ``[B, 7]`` INT32 SBUF.

    Args:
        h_hT ... cp: as ``draft_mla.draft_tail`` (``markov_w1`` is this rank's vocab shard ``[V/cp, 256]``).
        confidence: optional ``(w_h, w_m, bias)``: ``w_h`` ``[128, 56, 1]`` and ``w_m`` ``[128, 2, 1]`` BF16 HBM
            hT-tiled, ``bias`` ``[1, 1]`` FP32 HBM (the host weight builder of the GQA drafter). With it,
            ``confidence_out`` ``[7B, 1]`` FP32 HBM receives the logit of row ``j B + b``.
    """
    sb = AUTO_SBUF if sb is None else sb
    total_rows = h_hT.shape[1]
    tokens = total_rows // _BLOCK
    if confidence is not None and hidden_out is None:
        hidden_out = nl.ndarray((total_rows, HIDDEN), dtype=nl.bfloat16, buffer=nl.private_hbm, name=f"{tag}_hidden")
    drafts = draft_tail(
        h_hT,
        final_norm,
        lm_head_weight,
        markov_w1,
        markov_w2,
        bonus,
        rank_info,
        tp_group,
        tag,
        sb=sb,
        hidden_out=hidden_out,
        sources=sources,
        cp=cp,
    )
    if confidence is None:
        return drafts
    w_h_hbm, w_m_hbm, bias_hbm = confidence
    x_hT = _load_rows_hT(hidden_out, total_rows, f"{tag}_conf_hidden", sb)
    prev = _previous_tokens(drafts, bonus, tokens, f"{tag}_conf", sb)
    m_hT = _rows_to_hT(_markov_rows(prev, markov_w1, rank_info, tp_group, f"{tag}_conf", sb), sb)
    w_h = sb.tile((P_MAX, x_hT.shape[2], 1), nl.bfloat16)
    nisa.dma_copy(dst=w_h, src=w_h_hbm[0:P_MAX, 0 : x_hT.shape[2], 0:1], name=f"{tag}_conf_w_h")
    w_m = sb.tile((P_MAX, _MARKOV_TILES, 1), nl.bfloat16)
    nisa.dma_copy(dst=w_m, src=w_m_hbm[0:P_MAX, 0:_MARKOV_TILES, 0:1], name=f"{tag}_conf_w_m")
    bias = sb.tile((total_rows, 1), nl.float32)
    nisa.dma_copy(dst=bias, src=bias_hbm[0:1, 0:1].broadcast(0, total_rows), name=f"{tag}_conf_bias")
    logit = sb.tile((total_rows, 1), nl.float32)
    nisa.tensor_tensor(dst=logit, data1=_dot_fp32(x_hT, w_h, sb), data2=_dot_fp32(m_hT, w_m, sb), op=nl.add)
    nisa.tensor_tensor(dst=logit, data1=logit, data2=bias, op=nl.add)
    nisa.dma_copy(dst=confidence_out[0:total_rows, 0:1], src=logit, name=f"{tag}_conf_store")
    return drafts
