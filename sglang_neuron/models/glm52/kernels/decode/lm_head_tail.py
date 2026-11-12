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

"""GLM-5.2 decode tail on device: final RMSNorm, lm_head, top-k and sampler, from the token shard.

    hidden shard [T_shard, H] -> AllGather (all ranks) -> [T_shard * ranks, H]
        -> tail_chunked (norm, vocab-sharded lm_head GEMM, all-to-all, chunked top-k, sampler)
        -> tokens [T_shard, 1]

The tail is the GLM fork of ``lm_head``; for GLM only the data differs:

* no hidden padding (``hidden_actual = H = 6144``);
* the norm rounds to bf16 as ``GlmMoeDsaRMSNorm`` does (``norm_rounding="bf16"``);
* vocab 154,880 = 128 x 1,210, one shard per rank, and ``chunked_topk`` takes the assembled row in 16 chunks of
  9,680 (``glm52_topk_chunks``);
* the lm_head is bf16 and untied.

The giga kernel leaves the residual token-sharded, and every rank computes its vocab shard of the logits for the
whole batch, so one AllGather over all ranks in rank order rebuilds ``[gbs, H]``. The rows a rank samples are
``owned_row_block``: its own slots with one all-to-all hop, a mixed-radix shuffle with several.
"""

from typing import Tuple

import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from .attention.mla.mla_decode_constants import P_MAX
from .lm_head.tail_chunked import compose_hops, owned_row_block, tail_chunked
from .utils.kernel_assert import kernel_assert
from .utils.trace_names import unique_name

_ERROR_PREFIX = "[GLM-5.2 LM Head Tail]"

# Collective operands at LNC=1 must be private_hbm (the runtime rejects shared_hbm sources).
_COLLECTIVE_HBM = nl.private_hbm

# Widest vocab chunk one stage-1 ``nisa.topk`` call of the tail reads, and the widest stage-2 reduce row
# (``P * k``), which ``glm52_topk_chunks`` caps too (``k <= 1024`` at ``P = 16``). In a serving engine on
# trn3 at LNC=1, one launch of a draft chain whose tail read 38,720-wide chunks (P = 4) made every later
# program on the core return zero or NaN from its collectives; 9,680-wide chunks (P = 16) did not. A
# lone select of that width, and the 40,960-wide one the 1M MTP merge runs, leave the collectives exact
# on the released toolchain (``tests/e2e/kernels/test_wide_topk_collectives.py``), so the trigger was
# not the select's width alone; the cap keeps the tail's chunking as served.
_TOPK_CHUNK_MAX_N = 16384


def glm52_topk_chunks(rows_per_rank: int, vocab: int, k: int) -> int:
    """The stage-1 chunk count ``P`` for the tail's top-k.

    ``pick_chunks``' own rule -- a legal power-of-two ``P``, preferring whole 8-row tiles, then the
    coalesced load (``16 | vocab / P``), then the smallest ``P`` -- restricted so that each stage-1
    chunk (``vocab / P``) AND the stage-2 reduce row (``P * k``) are at most ``_TOPK_CHUNK_MAX_N``
    wide. The two preferences stay preferences: 16 ranks (vocab 19,360, 2 rows per rank) has no
    ``P`` that meets both, and gets ``P = 4``. 128 ranks (154,880) gets ``P = 16`` for every
    ``k <= 1024``; a larger ``k`` has no legal ``P`` there.
    """
    best_key, best_p = None, None
    p = 1
    while p <= P_MAX:
        if vocab % p == 0:
            width = vocab // p
            if 8 <= width <= _TOPK_CHUNK_MAX_N and k <= width and 8 <= p * k <= _TOPK_CHUNK_MAX_N:
                key = (0 if (rows_per_rank * p) % 8 == 0 else 1, 0 if width % 16 == 0 else 1, p)
                if best_key is None or key < best_key:
                    best_key, best_p = key, p
        p *= 2
    kernel_assert(
        best_p is not None,
        f"{_ERROR_PREFIX} no top-k chunk count keeps vocab={vocab} chunks and the k={k} reduce at most "
        f"{_TOPK_CHUNK_MAX_N} wide",
    )
    return best_p


def lm_head_row_owner(rank: int, ranks: int, t_shard: int, hops: Tuple[int, ...]) -> Tuple[int, int]:
    """``(row_start, n_rows)`` of the global token slots whose tokens ``rank`` returns.

    The caller-facing half of the tail's row-ownership contract (``owned_row_block``), restated
    with this kernel's own arguments so the decode harness and the kernel cannot drift apart.
    """
    return owned_row_block(rank, ranks, t_shard * ranks, tuple(hops))


def glm52_lm_head_tail(
    hidden_shard: nl.NkiTensor,  # [T_shard, H] bf16/fp32, this rank's token slots
    final_norm_w: nl.NkiTensor,  # [1, H] fp32 ``model.norm.weight``
    lm_head_weight_t: nl.NkiTensor,  # [H, V_s] bf16, this rank's vocab shard, TRANSPOSED
    sampling_params: nl.NkiTensor,  # [T_shard, 3|4] fp32 [top_k, top_p, temperature(, draw)] for the rows this rank OWNS
    ep_replica_group: ReplicaGroup,  # all ``ranks`` ranks in rank order
    ranks: int,
    lm_head_hops: Tuple[int, ...],
    eps: float,
    k: int,
    norm_rounding: str = "bf16",
    top_p_rule: str = "drop_crossing",
):
    """One rank's share of the GLM-5.2 sampling tail. Returns ``(tokens, topk_values, topk_indices)``.

    Args:
        hidden_shard (nl.NkiTensor): [T_shard, H] post-last-layer residual stream, this rank's slots.
        final_norm_w (nl.NkiTensor): [1, H] fp32 final RMSNorm weight (bf16 values widened).
        lm_head_weight_t (nl.NkiTensor): [H, V_s] bf16 lm_head shard, ``V_s = V / ranks``, columns
            ``[rank * V_s, (rank + 1) * V_s)`` of ``lm_head.weight.T``, C-contiguous.
        sampling_params (nl.NkiTensor): [T_shard, 3] or [T_shard, 4] fp32 sampler parameters for the
            rows this rank owns (``lm_head_row_owner``); temperature below 1e-5 means greedy.
        ep_replica_group (ReplicaGroup): One group listing all ranks in rank order.
        ranks (int): World size (= vocab-parallel degree).
        lm_head_hops (tuple): all_to_all hop sizes, multiplying to ``ranks``: ``(ranks,)`` for one
            mesh, ``(8, 16)`` when the host rejects a 128-rank mesh.
        eps (float): RMSNorm epsilon (1e-5 for GLM).
        k (int): Candidates kept by the top-k (the sampler's top_k is clipped to it).
        norm_rounding (str): ``"bf16"`` (HF realization, default) or ``"fp32"``.
        top_p_rule (str): ``"drop_crossing"`` (default) drops the token whose cumulative probability
            crosses ``top_p``; ``"keep_crossing"`` keeps it, as HF does.

    Returns:
        Tuple[nl.NkiTensor, nl.NkiTensor, nl.NkiTensor]: tokens [T_shard, 1] uint32,
        topk_values [T_shard, k] bf16, topk_indices [T_shard, k] uint32 (global vocab ids).
    """
    T_shard, H = hidden_shard.shape
    kernel_assert(
        len(final_norm_w.shape) == 2 and final_norm_w.shape[1] == H, f"{_ERROR_PREFIX} final_norm_w must be [1, H]"
    )
    kernel_assert(lm_head_weight_t.shape[0] == H, f"{_ERROR_PREFIX} lm_head_weight_t must be [H, V_s]")
    kernel_assert(
        sampling_params.shape[0] == T_shard, f"{_ERROR_PREFIX} sampling_params must hold this rank's {T_shard} rows"
    )
    prod = 1
    for h in lm_head_hops:
        prod *= h
    kernel_assert(prod == ranks, f"{_ERROR_PREFIX} lm_head_hops {lm_head_hops} must multiply to ranks={ranks}")

    # ═══════ Every rank needs every row: AllGather the token shards in rank order ═══════
    src = nl.ndarray((T_shard, H), dtype=hidden_shard.dtype, buffer=_COLLECTIVE_HBM, name=unique_name("lm_head_src"))
    nisa.dma_copy(dst=src, src=hidden_shard)
    gathered = nl.ndarray(
        (T_shard * ranks, H), dtype=hidden_shard.dtype, buffer=_COLLECTIVE_HBM, name=unique_name("lm_head_hidden")
    )
    ncc.all_gather(srcs=[src], dsts=[gathered], replica_group=ep_replica_group, collective_dim=0)

    # ═══════ norm + lm_head + all_to_all + top-k + sampler ═══════
    topk_values, topk_indices, tokens = tail_chunked(
        hidden_states=gathered,
        gamma=final_norm_w,
        lm_head_weight_t=lm_head_weight_t,
        eps=eps,
        hidden_actual=None,
        zero_interleaved_pad=False,
        k=k,
        sampling_params=sampling_params,
        hop_meshes=compose_hops(ranks, tuple(lm_head_hops)),
        emit_logits=False,
        norm_rounding=norm_rounding,
        # Explicit de-interleave: the folded receive layout mis-assembled the row at 128 peers x 2 rows
        # (see fold_deinterleave in the tail), and the round trip it saves is 620 KB here.
        fold_deinterleave=False,
        topk_chunks=glm52_topk_chunks(T_shard, ranks * lm_head_weight_t.shape[1], k),
        top_p_rule=top_p_rule,
    )
    return tokens, topk_values, topk_indices
