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

"""GLM-MoE-DSA prefill giga-kernel (GK), LNC1: every decoder layer and the final norm in ONE trace.

One layer is the reference ``GlmMoeDsaDecoderLayer.forward``, with every collective inside the
trace::

    normed, r_in   = rmsnorm(hidden + residual)                       input norm
    keys           = indexer_keys(normed)                             full layer: K, K all-gather
    q_lift, q_pe, qr, kv_cache[slot] = mla_qkv(normed)                 absorbed QKV, paged KV write
    topk           = indexer_score(qr, keys)  |  prev_topk            full | shared layer
    kv_full        = all_gather_cp(kv_cache[pages])                   CP KV gather (full: inside the score)
    out_attn       = sparse_mla(q_lift, q_pe, kv_full, topk)          latent attention
    attn_o         = row_parallel_vup_oproj(out_attn, wuv, wo_shard)  V-up + o_proj
    MoE layer:   packed, r_out = rmsnorm_mx_router(attn_o + r_in)
                 out = reduce_scatter(bwmm(all_gather(packed))) + shared_expert(packed)
    dense layer: out, r_out = tp_mlp(rmsnorm(attn_o + r_in))
    return out, r_out                                                 layer output = r_out + out

Sharding (the 128-rank LNC1 topology), fixed for this kernel:

* attention: context parallel over ``cp_degree`` ranks, each owning ``T`` tokens. The BF16
  attention weights ``wqkv_a / wq_b / wuk / wuv`` arrive as this rank's 1/cp chunk and are
  all-gathered over the CP group in ONE coalesced collective. The stack issues layer i+1's gather
  right after layer i's KV all-gather, at the lowest DMA priority, so it drains under layer i.
* o_proj: row-parallel inside TP x CP wo groups: V-up own tokens, all-to-all the activations so
  each rank gets its Hdv K-slice, multiply against the LOCAL wo row shard, reduce-scatter.
* dense MLP: sequence-parallel tensor parallel over the same wo groups.
* MoE (MXFP4): EP x TP over ``moe_replica_groups``; the packed activation is all-gathered, each
  rank runs its local experts (host-packed MX scales), and a reduce-scatter returns its rows.

Every sub-kernel is an existing ``@nki.jit`` kernel called inline: inside an active trace a
``@nki.jit`` call runs its body, so the whole stack is one NEFF. The collectives stage through
named ``private_hbm`` buffers (LNC1 rejects ``shared_hbm`` collective sources and I/O tensors).
"""

import dataclasses
import os

import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import dge_mode

from .mlp.mlp import mlp
from .norm.rmsnorm_mx import rmsnorm_mx_prefill
from .norm.rmsnorm_residual import rmsnorm_residual_prefill
from .subkernels.find_nonzero_indices import find_nonzero_indices
from .utils.sbuf_scoped_alloc import SbufManager, get_logger, sbm_name_scope, scoped_name, sizeinbytes
from .utils.common_types import (
    ActFnType,
    ComputationMode,
    ExpertAffinityScaleMode,
    NormType,
    QuantizationType,
    RouterActFnType,
)
from .utils.kernel_assert import kernel_assert
from .utils.kernel_helpers import div_ceil, get_verified_program_sharding_info
from .moe.block_moe_mx import block_moe_mx
from .moe.block_moe_utils import SkipMode
from .moe import block_moe_utils as _moe_utils
from .attention.mla_bf16 import (
    WEIGHT_LOADS_REC as _QKV_WLOADS_REC,
    _bf16_matmul_split,
    _load_bf16_weights,
    _load_x_bf16_tile,
)
from .attention.common import MlaPrecision
from .attention.qkv import mla_qkv_cte_kernel
from .attention.sparse_attention import mla_sparse_attention_cte_kernel
from .attention.vup_oproj import mla_oproj_partial_cte_kernel, mla_vup_cte_kernel
from .oproj_mx import glm_vup_oproj_mx_kernel
from .indexer import glm_indexer_keys, glm_indexer_score
from .indexer import QR_T_AFTER as _QR_T_AFTER

P_MAX = 128
_MX_TILE = 512
# rmsnorm_mx_prefill packs 4 H512 scale tiles into one 128-wide block.
_SCALE_TILES_PER_BLOCK = 4
# block_moe_mx's dynamic loop reads 2 conditions past the last block.
_COND_PAD = 2
# find_nonzero_indices token chunk cap (the host mapping uses the same).
_MAPPING_MAX_CHUNK = 16384
# DMA QoS priority of the one-layer-ahead attention-weight prefetch (0 highest .. 3 lowest).
_ATTN_PREFETCH_PRIORITY = 3
# The query position of a bucket padding row (``models/glm52/prefill/inputs.PAD_POSITION``): above every real position.
_PAD_Q_POS = float(1 << 23)
_ERR = "[glm52 prefill]"


def _replica_group(groups, world: int) -> ncc.ReplicaGroup:
    """``ReplicaGroup`` from a tuple of rank tuples; None means one group ``[0, world)``."""
    if groups is None:
        return ncc.ReplicaGroup([list(range(world))])
    return ncc.ReplicaGroup([list(g) for g in groups])


def _group_size(groups, default: int) -> int:
    return default if groups is None else len(groups[0])


def _hbm_copy_rows(dst, src, rows: int, cols: int, chunk_rows: int = 1024):
    """HBM -> HBM copy of ``src[:rows, :cols]``, split into row chunks so it spreads over queues."""
    for r0 in range(0, rows, chunk_rows):
        n = min(chunk_rows, rows - r0)
        nisa.dma_copy(dst=dst[r0 : r0 + n, 0:cols], src=src[r0 : r0 + n, 0:cols])


def _all_gather(dst, src, replica_groups, w: int, priority=None):
    return ncc.all_gather(
        dsts=[dst], srcs=[src], replica_group=_replica_group(replica_groups, w), collective_dim=0, priority=priority
    )


def _reduce_scatter(dst, src, replica_groups, w: int):
    ncc.reduce_scatter(
        dsts=[dst], srcs=[src], replica_group=_replica_group(replica_groups, w), collective_dim=0, op=nl.add
    )


def _up_flag(name: str) -> bool:
    """``GLM_GK_UP_<name>=1``: one optional prefill schedule change. Unset or 0 (default): off."""
    raw = os.environ.get(f"GLM_GK_UP_{name}", "").strip()
    kernel_assert(raw in ("", "0", "1"), f"{_ERR} GLM_GK_UP_{name}={raw!r}: use 1, 0 or leave it unset")
    return raw == "1"


#: rows per rank (T) of the stack trace in progress, set by glm52_prefill_giga_kernel; read by GLM_GK_PRJ_QKV_T.
_PRJ_TRACE_T = [0]


def _prj_int(name: str, allowed=None) -> int:
    """``GLM_GK_PRJ_<name>=<int>``: a projection-section schedule knob. Unset or 0 (default): off."""
    raw = os.environ.get(f"GLM_GK_PRJ_{name}", "").strip()
    kernel_assert(
        raw == "" or raw.isdigit(), f"{_ERR} GLM_GK_PRJ_{name}={raw!r}: use a non-negative integer or leave it unset"
    )
    v = int(raw) if raw else 0
    if allowed is not None:
        kernel_assert(v in allowed, f"{_ERR} GLM_GK_PRJ_{name}={v}: use one of {sorted(allowed)}")
    return v


def _prj_qkv_opt() -> int:
    """``GLM_GK_PRJ_QKV=<bits>``: the hybrid QKV kernel's ``hy_opt`` bits (``mla_qkv_cte_kernel``): schedule-only
    changes, so the outputs are the same bits. ``GLM_GK_PRJ_QKV_T=<rows>[,<rows>...]`` limits them to the traces of
    those rows per rank (every rank of a graph runs the same NEFF); unset: every trace."""
    v = _prj_int("QKV")
    kernel_assert(v < 4096, f"{_ERR} GLM_GK_PRJ_QKV={v}: bits above 2048 are not defined")
    # bit 4 without 1024 (the corrected head-group budget) is wrong in every graph, layer 0 included.
    kernel_assert(not (v & 4) or (v & 1024), f"{_ERR} GLM_GK_PRJ_QKV={v}: bit 4 needs bit 1024 (wrong without it)")
    raw = os.environ.get("GLM_GK_PRJ_QKV_T", "").strip()
    if not v or not raw:
        return v
    rows = [r.strip() for r in raw.split(",")]
    kernel_assert(
        all(r.isdigit() for r in rows), f"{_ERR} GLM_GK_PRJ_QKV_T={raw!r}: use comma-separated rows per rank"
    )
    return v if _PRJ_TRACE_T[0] in {int(r) for r in rows} else 0


def _prj_xag() -> int:
    """``GLM_GK_PRJ_XAG=2`` (with GLM_GK_UP_S1DMAT): the head-TP x all-gather as 2 row-chunk all-gathers (each rank's
    rows ``[0, T / 2)`` and ``[T / 2, T)``); stage 1 multiplies chunk 0's gathered rows while chunk 1 is in flight, and
    writes every output row to its usual place in ``s1_send``. A stage-1 output row does not depend on the other rows of
    its 128-row tile, so ``s1_send`` holds the same bits. Unset / 0 / 1: one all-gather."""
    return 2 if _prj_int("XAG", allowed={0, 1, 2}) == 2 else 0


def _prj_normst() -> bool:
    """``GLM_GK_PRJ_NORMST=1``: the hybrid input norm (GLM_GK_UP_NORMRES's rmsnorm_residual_prefill) stores its bf16
    output per 1,536-column H tile on HW DGE right after that tile's scale, instead of one SW DGE store of the whole row
    after the last tile (the pre-norm sum's SW DGE store keeps its place, before the scale). The x all-gather's source
    then lands earlier and is no SW DGE write. The same values to the same bytes. Unset / 0: off."""
    return _prj_int("NORMST", allowed={0, 1}) == 1


def _sync_hol() -> int:
    """``GLM_GK_SYNC_HOL=1|2``: the Sync-queue head-of-line fix of the head-TP full layers. Their QKV issues its
    per-head-group wq_b / W_uk loads (``_load_bf16_weights_k_slab``) from the in-order Sync queue, and the scheduler
    places the indexer score's qr ``dma_transpose`` loads (glm_indexer ``load_activation_k_major``) ahead of the last
    head groups' loads there. A ``dma_transpose`` is held while the CP key all-gather is in flight, so those weights
    issue only at the all-gather's end and the QKV's phase B runs after it. With 1, every one of those transposes
    gets a ``dependency.schedule`` edge on every QKV group load: issue order for the schedulers, no semaphore, no
    instruction, no queue change, so the loads leave Sync as soon as their own buffer waits release and phase B
    finishes under the all-gather. 2: the score's own wq load (glm_indexer ``load_weight_k_major``) also follows
    them, so it issues when the QKV frees its SBUF instead of behind the held transposes. The same instructions on
    the same data: bit-identical. Unset or 0 (default): off."""
    raw = os.environ.get("GLM_GK_SYNC_HOL", "").strip()
    kernel_assert(raw in ("", "0", "1", "2"), f"{_ERR} GLM_GK_SYNC_HOL={raw!r}: use 1, 2, 0 or leave it unset")
    return int(raw or 0)


def _cc_sync(src_row, groups, w: int, tag: str, after=()):
    """A 32 B all-gather over ``groups`` ahead of the hybrid o_proj all-to-all and the MoE all-gathers, so no peer
    writes into a late rank's destination while that rank still runs its indirect DMAs (a destination with no other
    tensor at its address skips the runtime handshake). The guarded collective ``depends_on`` it, which orders the two
    triggers. ``src_row`` is a ``[1, n]`` slice of 32 B of the buffer the guarded collective sends; ``after`` are the
    instructions that must finish before the copy reads it."""
    n = src_row.shape[-1]
    s = nl.ndarray((1, n), dtype=src_row.dtype, buffer=nl.private_hbm, name=scoped_name(f"gk_sync_{tag}_s"))
    d = nl.ndarray((w, n), dtype=src_row.dtype, buffer=nl.private_hbm, name=scoped_name(f"gk_sync_{tag}_d"))
    cp = nisa.dma_copy(dst=s, src=src_row)
    if after:
        cp.depends_on(*after)
    return _all_gather(d, s, groups, w)


def send_life() -> bool:
    """``GLM_GK_SEND_LIFE=1``: keep each attention-weight prefetch's send buffers live until the layer that uses the
    weights has its attention output. That layer's KV all-gather completed first, and each peer issues it only after its
    own prefetch completed (its q / kv projection reads the gathered weights), so every peer has this rank's sends by
    then. Unset (default): off."""
    return os.environ.get("GLM_GK_SEND_LIFE", "").strip() == "1"


# GLM_GK_SEND_LIFE: id(first gathered weight) -> the send buffers of the prefetch that gathered it (trace time).
_SEND_LIVE: dict = {}


def moe_zinit_gate() -> bool:
    """``GLM_GK_MOE_ZINIT_GATE=1``: the expert outputs' zero-init DMAs wait (kind=runtime) on the MoE 32 B sync. That
    collective reads this layer's MoE input, which every peer builds only after its o_proj reduce-scatter (and so its
    o_proj all-to-all) completed; so once it completes here, every peer holds this rank's o_proj data and the zero-init
    may reuse those bytes. Unset (default): off."""
    return os.environ.get("GLM_GK_MOE_ZINIT_GATE", "").strip() == "1"


_ZINIT_SITES = ("kv", "moe", "rs", "oproj")


def _moe_zinit_at() -> str:
    """``GLM_GK_MOE_ZINIT_AT=<site>``: where the routed experts' output (the MoE reduce-scatter source) is zeroed.
    Unset (default): inside the routed bwmm, whose zero-init then runs at the start of the attention and competes
    with its KV row gathers. With a site, every MoE layer accumulates into ONE stack-lifetime buffer, so the
    zero-init aliases no per-layer tensor; the GK zeroes it with the DMAs of the bwmm's zero-init, and the bwmm skips
    its own (``init_output=False``). The site sets where the GK issues it and its earliest start, through a data
    fence (``_zinit_fence``) and no depends_on:

    * ``kv``: after the layer's local KV page gather, under its KV all-gather (PE and HWDGE queues idle there).
    * ``moe``: after the layer's router output, under the MoE token all-gather.
    * ``rs``: after the layer before (its reduce-scatter and its combine), under the next layer's norm and QKV.
    * ``oproj``: after the layer's attention output, under its V-up / o_proj (PE-bound, few DMAs), so the
      zero-init does not compete with the attention's KV row gathers.

    The buffer holds zeros when the first block accumulates (the zero-init writes it after the previous MoE layer's
    reduce-scatter read it, and the first accumulate writes it after the zero-init), and the blocks add in the same
    order, so the result is bit-identical."""
    raw = os.environ.get("GLM_GK_MOE_ZINIT_AT", "").strip()
    kernel_assert(
        raw in ("",) + _ZINIT_SITES, f"{_ERR} GLM_GK_MOE_ZINIT_AT={raw!r}: use one of {_ZINIT_SITES}, or leave it unset"
    )
    kernel_assert(
        not (raw and moe_zinit_gate()), f"{_ERR} GLM_GK_MOE_ZINIT_AT and GLM_GK_MOE_ZINIT_GATE: set one, not both"
    )
    return raw


def _moe_zinit_min_t() -> int:
    """``GLM_GK_MOE_ZINIT_MIN_T=<rows>``: GLM_GK_MOE_ZINIT_AT acts only in graphs of at least ``rows`` rows per rank.
    Smaller graphs keep the bwmm's own zero-init, which ends under the token all-gather there; in the larger ones the
    bwmm issues it after the mapping and its static block loop waits for all of it. Unset or 0 (default): every
    graph."""
    raw = os.environ.get("GLM_GK_MOE_ZINIT_MIN_T", "").strip()
    kernel_assert(
        raw == "" or raw.isdigit(), f"{_ERR} GLM_GK_MOE_ZINIT_MIN_T={raw!r}: use a row count, or leave it unset"
    )
    return int(raw) if raw else 0


def _zinit_fence(zeros, fences, tag: str):
    """Order every later read of the SBUF tile ``zeros`` after the writes of the first and last word of each tensor of
    ``fences``: the words are loaded, ANDed to 0 and ORed into ``zeros[0, 0]``, which every zero-init DMA reads. Plain
    data dependencies only; the bits of ``zeros`` do not change.

    No view of ``zeros``: a view of an ``nl.ndarray`` SBUF tile keeps its 2-byte element type (the MLIR verifier
    rejected the fp32-view memset). The int32 0 is converted to a bf16 +0.0 and added to ``zeros[0, 0]`` (+0.0):
    +0.0 + +0.0 = +0.0, the same bits."""
    for i, t in enumerate(fences):
        flat = t.reshape((t.size,)).view(nl.int32)
        for j, off in enumerate((0, flat.shape[0] - 1)):
            f = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf, name=scoped_name(f"gk_zif_{tag}{i}_{j}"))
            nisa.dma_copy(dst=f, src=flat.ap(pattern=[[1, 1], [1, 1]], offset=off))
            nisa.tensor_scalar(dst=f, data=f, op0=nl.bitwise_and, operand0=0, engine=nisa.engine.vector)
            fb = nl.ndarray((1, 1), dtype=zeros.dtype, buffer=nl.sbuf, name=scoped_name(f"gk_zifb_{tag}{i}_{j}"))
            nisa.tensor_copy(dst=fb, src=f, engine=nisa.engine.vector)
            nisa.tensor_tensor(
                dst=zeros[0:1, 0:1], data1=zeros[0:1, 0:1], data2=fb, op=nl.add, engine=nisa.engine.vector
            )


def _zero_moe_out(buf, fences, tag: str):
    """GLM_GK_MOE_ZINIT_AT: zero the stack-lifetime routed output ``buf`` ``[T_moe, H]`` with the bwmm's own zero-init
    (``output_initialization``: one 128-row HWDGE DMA per tile from a ``[128, H]`` SBUF zero tile), not before the
    writes of ``fences``."""
    rows, H = buf.shape
    zeros = nl.ndarray((P_MAX, H), dtype=nl.bfloat16, buffer=nl.sbuf, name=scoped_name(f"gk_zi_{tag}"))
    # A bf16 memset: the fp32 view of an nl.ndarray tile keeps the 2-byte element type, and the MLIR verifier rejects a
    # 4-byte value on it. +0.0 has the same bits either way.
    nisa.memset(dst=zeros, value=0.0)
    if fences:
        _zinit_fence(zeros, fences, tag)
    kernel_assert(_moe_utils.ZINIT_AFTER is None, f"{_ERR} GLM_GK_MOE_ZINIT_AT: a ZINIT_AFTER gate is set")
    _moe_utils.output_initialization(buf, _ZinitDims(rows, H), zeros=zeros)


class _ZinitDims:
    """The ``dims`` that ``output_initialization`` reads: ``T`` rows of ``H``."""

    def __init__(self, T: int, H: int):
        self.T = T
        self.H = H


def kvl_life() -> bool:
    """``GLM_GK_KVL_LIFE=1``: keep each layer's gathered kv_local tensors live until the layer's o_proj output exists.
    A peer finishes its o_proj only after its attention, which needs this rank's kv_local, so no later buffer that peers
    write (the next prefetch destination, the o_proj destination) can take its HBM first. Unset (default): off."""
    return os.environ.get("GLM_GK_KVL_LIFE", "").strip() == "1"


# GLM_GK_KVL_LIFE: the kv_local tensors gathered since the last layer's o_proj (trace time).
_KVL_LIVE: list = []


def _anchor_after(after_2d, tensors, tag: str):
    """Read ``after_2d``, then each of ``tensors``, through one SBUF tile into a shared_hbm sink. Each read of the
    tile is copied out before the next DMA overwrites it (write-after-read), so every read waits on the ``after_2d``
    read. An unread anchor is dropped by the compiler, and a depends_on on such a read fails to compile
    (NCC_IXDR001)."""
    n = 8 * (len(tensors) + 1)
    t = nl.ndarray((1, 8), dtype=after_2d.dtype, buffer=nl.sbuf, name=scoped_name(f"gk_{tag}_anc"))
    both = nl.ndarray((1, n), dtype=after_2d.dtype, buffer=nl.sbuf, name=scoped_name(f"gk_{tag}_anc_all"))
    nisa.dma_copy(dst=t[0:1, 0:8], src=after_2d[0:1, 0:8])
    nisa.tensor_copy(dst=both[0:1, 0:8], src=t[0:1, 0:8])
    for i, x in enumerate(tensors):
        kernel_assert(x.dtype == after_2d.dtype, f"{_ERR} {tag} anchor: {x.dtype} != {after_2d.dtype}")
        nisa.dma_copy(dst=t[0:1, 0:8], src=x[0:1, 0:8])
        nisa.tensor_copy(dst=both[0:1, 8 * (i + 1) : 8 * (i + 2)], src=t[0:1, 0:8])
    sink = nl.ndarray((1, n), dtype=after_2d.dtype, buffer=nl.shared_hbm, name=scoped_name(f"gk_{tag}_sink"))
    nisa.dma_copy(dst=sink[0:1, 0:n], src=both[0:1, 0:n])


def _anchor_sends(attn_w, after_2d):
    """GLM_GK_SEND_LIFE: anchor the send buffers of the prefetch that gathered ``attn_w`` on ``after_2d`` (this layer's
    attention output)."""
    sends = _SEND_LIVE.pop(id(attn_w[0]), None) if attn_w is not None else None
    if sends:
        _anchor_after(after_2d, sends, "send")


# A dma_copy whose size is not a multiple of 64 KiB is lowered as 64 KiB chunks plus a "peel_split" tail DMA, and a
# collective that reads the copy waits only on the chunks (neuronx-cc), so it can send stale tail bytes.
_DMA_CHUNK_BYTES = 65536


def _stage_copy(dst, src, rows: int, cols: int):
    """``dst[:rows, :cols] = src[:rows, :cols]`` into a collective's send buffer. ``GLM_GK_STAGE_SPLIT=1``
    issues a 64 KiB-multiple head and the tail as two DMAs on flat views, so the collective waits on both."""
    esize = sizeinbytes(dst.dtype)
    nbytes = rows * cols * esize
    if os.environ.get("GLM_GK_STAGE_SPLIT", "").strip() != "1" or nbytes % _DMA_CHUNK_BYTES == 0:
        nisa.dma_copy(dst=dst[0:rows, 0:cols], src=src[0:rows, 0:cols])
        return
    n = rows * cols
    m = (nbytes // _DMA_CHUNK_BYTES) * _DMA_CHUNK_BYTES // esize
    df, sf = dst.reshape((1, n)), src.reshape((1, n))
    nisa.dma_copy(dst=df[0:1, 0:m], src=sf[0:1, 0:m])
    nisa.dma_copy(dst=df[0:1, m:n], src=sf[0:1, m:n])


def _moe_pad_mask() -> bool:
    """``GLM_GK_MOE_PAD_MASK``: zero the router affinities of the bucket padding rows before the MoE all-gather.

    All padding rows of a step share one hidden state, so without the mask they route to one top-8 expert set and load
    its ranks with up to one extra block per 128 padding rows. A row with no nonzero affinity takes no block, and its
    routed output stays 0. Real rows keep their bits and their per-expert order. Unset or 1 (default): on. 0: off."""
    raw = os.environ.get("GLM_GK_MOE_PAD_MASK", "").strip()
    kernel_assert(raw in ("", "0", "1"), f"{_ERR} GLM_GK_MOE_PAD_MASK={raw!r}: use 0 or 1, or leave it unset")
    return raw != "0"


def tc_moe_ag() -> str:
    """``GLM_GK_TC_MOE_AG=aff``: the MoE dispatch all-gathers the fp32 router affinities of every row first
    (``[T, E]``), then the packed rows as before. The blockwise mapping (``find_nonzero_indices`` and its glue) reads
    the gathered affinities, so it, the routed experts' routing-table / weight loads and the output zero-init run under
    the packed-row all-gather instead of after it. The mapping sees the same fp32 values, and the experts read the
    same packed rows. ``split``: as ``aff``, and the second all-gather moves only ``[hidden | scale]``; the routed
    experts fill each gathered row's affinity tail from the gathered affinities (bwmm ``affinity_rows``), so their
    concat rows are byte for byte the fused ones. Unset or 0 (default): off."""
    raw = os.environ.get("GLM_GK_TC_MOE_AG", "").strip()
    kernel_assert(
        raw in ("", "0", "aff", "split"), f"{_ERR} GLM_GK_TC_MOE_AG={raw!r}: use aff, split, 0 or leave it unset"
    )
    return "" if raw in ("", "0") else raw


def _moe_dyn_inner() -> int:
    """``GLM_GK_MOE_DYN_INNER=<n>``: the routed bwmm's dynamic block loop runs chunks of ``n`` blocks
    (``block_moe_mx._DYN_INNER``, a power of two; the kernel's default is 8) and the remainder loop the rest. With
    ``1`` every dynamic block is one iteration of the chunked loop and the remainder loop never runs, so the chunk ->
    remainder hand-over goes. The same blocks in the same order: bit-identical. Unset or 0 (default): the kernel's 8."""
    raw = os.environ.get("GLM_GK_MOE_DYN_INNER", "").strip()
    kernel_assert(
        raw in ("", "0", "1", "2", "4", "8", "16"), f"{_ERR} GLM_GK_MOE_DYN_INNER={raw!r}: use 1, 2, 4, 8 or 16"
    )
    return int(raw) if raw else 0


def _moe_pf_tiles(T: int) -> int:
    """``GLM_GK_MOE_PF_TILES=<k>``: the routed bwmm prefetches the NEXT block's gate/up I tiles 0 .. k-1 across the
    block boundary into k persistent tile buffers (``prefetch_tiles``), so the next block's tile streams run under this
    block's down projection instead of queuing behind the next down-weight load, which waits (WAR) for that projection
    at the head of the in-order Sync queue. +(k - 1) x 24 KiB of persistent SBUF. The prefetches of a dynamic block that
    is not live are skipped. Same bytes into the same matmuls: bit-identical. Unset, 0 or 1 (default): tile 0 only.

    ``GLM_GK_MOE_PF_TILES_T128=<k>`` (same values) overrides it in the graph of 128 rows per rank, where the expert
    saving turns into idle at the next collective; every rank of a graph runs the same NEFF. Both values are checked in
    every trace, so a bad one is refused at the first compile."""
    base = os.environ.get("GLM_GK_MOE_PF_TILES", "").strip()
    scoped = os.environ.get("GLM_GK_MOE_PF_TILES_T128", "").strip()
    for name, raw in (("GLM_GK_MOE_PF_TILES", base), ("GLM_GK_MOE_PF_TILES_T128", scoped)):
        kernel_assert(raw in ("", "0", "1", "2", "3"), f"{_ERR} {name}={raw!r}: use 2, 3, 1, 0 or leave it unset")
    raw = scoped if T == 128 and scoped else base
    return int(raw) if raw not in ("", "0") else 1


class _bwmm_dyn_inner:
    """Trace-time override of the bwmm module's dynamic-loop chunk constants (restored on exit)."""

    def __init__(self, n: int):
        self.n = n

    def __enter__(self):
        from .moe import block_moe_mx as _bw

        self.mod = _bw
        self.old = (_bw._DYN_INNER, _bw._DYN_STEP, _bw._DYN_STEP_LOG2)
        if self.n:
            _bw._DYN_INNER = _bw._DYN_STEP = self.n
            _bw._DYN_STEP_LOG2 = self.n.bit_length() - 1
        return self

    def __exit__(self, *exc):
        self.mod._DYN_INNER, self.mod._DYN_STEP, self.mod._DYN_STEP_LOG2 = self.old
        return False


def _moe_wstream():
    """``GLM_GK_MOE_WRELAY=1|2``: the routed experts' bwmm ``weight_stream`` ``(tile_major, priority, second_queue)``;
    None when unset (default).

    ``1``: the host stores the routed gate/up weights I-tile-major (the prefill weight loaders'
    ``_gate_up_tile_major``, same shape, permuted bytes) and the bwmm reads them through the matching view
    (``_gup_tile_src``): a partition's part of an I tile is one 24 KiB run, of a stream chunk one 3 KiB run. ``2``: the
    same layout, and each 512-wide I tile is ONE DMA into the ping-pong tile buffer instead of 8 H-chunked streams. The
    streams keep their DMA class and queue. The same bytes reach the same SBUF buffers, and the matmuls and their PSUM
    groups are unchanged: bit-identical. Every graph's routed call reads them so (the stored layout is per process);
    the shared expert keeps its streams."""
    raw = os.environ.get("GLM_GK_MOE_WRELAY", "").strip()
    kernel_assert(raw in ("", "0", "1", "2"), f"{_ERR} GLM_GK_MOE_WRELAY={raw!r}: use 1, 2, 0 or leave it unset")
    relay = int(raw) if raw else 0
    return (relay, None, False) if relay else None


def _moe_aff8() -> bool:
    """``GLM_GK_MOE_AFF8=1``: the router packs the compact top-8 ``[ids (8) | affinities (8)]`` fp32 tail into each
    MoE row instead of the dense ``[E]`` fp32 affinities (``rmsnorm_mx_prefill(pack_topk8)``), so the token
    all-gather moves 64 B instead of 1 KB of affinities per row. Each rank rebuilds its local experts' dense columns
    ``[T_moe, E_local]`` from the gathered tails (``_aff8_local``) for the mapping, and the routed bwmm reads each
    block token's affinity from the compact tail (``compact_affinities``). Every rebuilt value equals the dense one
    bit for bit, so the routing and the result are unchanged. With GLM_GK_TC_MOE_AG the affinity all-gather moves the
    16 compact columns (and with ``split`` the bwmm gathers them as its affinity rows). Unset or 0 (default): off."""
    raw = os.environ.get("GLM_GK_MOE_AFF8", "").strip()
    kernel_assert(raw in ("", "0", "1"), f"{_ERR} GLM_GK_MOE_AFF8={raw!r}: use 1, 0 or leave it unset")
    return raw == "1"


def _shared_under_rs() -> str:
    """``GLM_GK_SHARED_UNDER_RS=late``: take the shared expert off the MoE combine reduce-scatter's critical path.
    Needs the replicated shared expert (``GLM_GK_SHARED_TP=1``). The shared expert is traced after the routed
    experts' reduce-scatter is issued, so it runs under it, and its rows are added after it (``RS(routed) + shared``,
    the replicated path's add). Same instructions and values as the replicated graph without the knob, in another
    program order: bit-identical to it. Against the tensor-parallel shared expert the shared output is one fp32 PSUM
    sum over the whole intermediate dim instead of m bf16 partials summed inside the reduce-scatter (reassociation).
    Unset or 0 (default): off."""
    raw = os.environ.get("GLM_GK_SHARED_UNDER_RS", "").strip()
    kernel_assert(raw in ("", "0", "late"), f"{_ERR} GLM_GK_SHARED_UNDER_RS={raw!r}: use late, 0 or leave it unset")
    return "" if raw in ("", "0") else raw


# GLM_GK_MOE_AFF8: (id, affinity) pairs of the compact tail (the router's top_k).
_AFF8_K = 8


def _aff8_local(moe_all, ep_rank, aff_col: int, T_moe: int, E_local: int):
    """GLM_GK_MOE_AFF8: ``[T_moe, W]`` fp32 private_hbm, ``W = max(8, E_local)``, column ``e < E_local`` = the dense
    affinity of this rank's local expert ``e`` (global ``ep_rank * E_local + e``) of every gathered row, the rest 0:
    ``max_k (ids[t, k] == g_e) * vals[t, k]`` from the compact tails at fp32 column ``aff_col`` of ``moe_all``
    (``[T_moe, r4]``). Row ``t = p * NJ + j`` sits on partition ``p``.
    ``E_local > 8`` (a GLM_GK_MOE_PLACEMENT with more than 8 experts a rank) widens the table to ``E_local`` columns;
    every column is the same per-column expression, and ``E_local <= 8`` keeps ``W = 8`` (the same instructions)."""
    K = _AFF8_K
    W = max(8, E_local)
    kernel_assert(
        E_local <= 64 and T_moe % P_MAX == 0, f"{_ERR} GLM_GK_MOE_AFF8: E_local {E_local} > 64 or T_moe {T_moe}"
    )
    r4 = moe_all.shape[1]
    kernel_assert(aff_col + 2 * K <= r4, f"{_ERR} GLM_GK_MOE_AFF8: the compact tail runs past the {r4}-word rows")
    NJ = T_moe // P_MAX
    src = moe_all.view(nl.float32)
    ids = nl.ndarray((P_MAX, NJ, K), dtype=nl.float32, buffer=nl.sbuf, name=scoped_name("gk_a8_ids"))
    vals = nl.ndarray((P_MAX, NJ, K), dtype=nl.float32, buffer=nl.sbuf, name=scoped_name("gk_a8_vals"))
    nisa.dma_copy(dst=ids, src=src.ap(pattern=[[NJ * r4, P_MAX], [r4, NJ], [1, K]], offset=aff_col))
    nisa.dma_copy(dst=vals, src=src.ap(pattern=[[NJ * r4, P_MAX], [r4, NJ], [1, K]], offset=aff_col + K))
    v = dict(engine=nisa.engine.vector)
    # g0 = ep_rank * E_local on every partition (a broadcast load of the [1, 1] HBM rank).
    g_i = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf, name=scoped_name("gk_a8_gi"))
    nisa.dma_copy(dst=g_i, src=ep_rank.reshape((1, 1)).ap(pattern=[[0, P_MAX], [1, 1]], offset=0))
    nisa.tensor_scalar(dst=g_i, data=g_i, op0=nl.multiply, operand0=E_local, **v)
    g0 = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf, name=scoped_name("gk_a8_g0"))
    nisa.tensor_copy(dst=g0, src=g_i)
    loc = nl.ndarray((P_MAX, NJ, W), dtype=nl.float32, buffer=nl.sbuf, name=scoped_name("gk_a8_loc"))
    nisa.memset(dst=loc, value=0.0)
    eq = nl.ndarray((P_MAX, NJ, K), dtype=nl.float32, buffer=nl.sbuf, name=scoped_name("gk_a8_eq"))
    ge = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf, name=scoped_name("gk_a8_ge"))
    for e in range(E_local):
        nisa.tensor_scalar(dst=ge, data=g0, op0=nl.add, operand0=float(e), **v)
        nisa.tensor_scalar(dst=eq, data=ids, op0=nl.equal, operand0=ge[0:P_MAX, 0:1], **v)
        nisa.tensor_tensor(dst=eq, data1=eq, data2=vals, op=nl.multiply, **v)
        nisa.tensor_reduce(dst=loc[0:P_MAX, 0:NJ, e : e + 1], op=nl.maximum, data=eq, axis=(2,), keepdims=True)
    out = nl.ndarray((T_moe, W), dtype=nl.float32, buffer=nl.private_hbm, name=scoped_name("gk_a8_loc_hbm"))
    nisa.dma_copy(dst=out.reshape((P_MAX, NJ * W)), src=loc.reshape((P_MAX, NJ * W)))
    return out


def _mask_pad_affinities(packed, q_pos, aff_col: int, n_exp: int, T: int):
    """Multiply the fp32 affinities ``packed[:, aff_col : aff_col + n_exp]`` of each row by ``q_pos < _PAD_Q_POS``, in
    place. A padding row gets +0.0 (affinities are >= 0). A real row is multiplied by 1.0, so its bits do not change."""
    q_rows = q_pos.reshape((T, 1))
    for r0 in range(0, T, P_MAX):
        n = min(P_MAX, T - r0)
        keep = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf, name=scoped_name(f"gk_padm_keep{r0}"))
        aff = nl.ndarray((P_MAX, n_exp), dtype=nl.float32, buffer=nl.sbuf, name=scoped_name(f"gk_padm_aff{r0}"))
        nisa.dma_copy(dst=keep[:n, :], src=q_rows[r0 : r0 + n, 0:1])
        nisa.dma_copy(dst=aff[:n, :], src=packed[r0 : r0 + n, aff_col : aff_col + n_exp])
        nisa.tensor_scalar(dst=keep[:n, :], data=keep[:n, :], op0=nl.less, operand0=_PAD_Q_POS)
        nisa.tensor_scalar(dst=aff[:n, :], data=aff[:n, :], op0=nl.multiply, operand0=keep[:n, 0:1])
        nisa.dma_copy(dst=packed[r0 : r0 + n, aff_col : aff_col + n_exp], src=aff[:n, :])


def _gather_attn_weights(wqkv_a, wq_b, wuk, wuv, cp_degree, cp_replica_groups, priority, tag, dims):
    """All-gather the four attention weights over the CP group in ONE coalesced collective.

    Each weight arrives as this rank's rank-major chunk of its row-major-flattened tensor, so
    concatenating the chunks along dim 0 reproduces the flat weight. Small all-gathers run at
    a much lower bandwidth than large ones, and collectives run one at a time, so one op of the
    summed bytes is much shorter than four. A collective source cannot be an I/O tensor on LNC1, so
    each chunk is staged through a private_hbm send buffer.
    """
    H, q_lora_rank, L, R, n_heads, qk_head, qk_nope = dims
    full_shapes = ((H, q_lora_rank + L + R), (q_lora_rank, n_heads * qk_head), (qk_nope, n_heads * L), None)
    sends, fulls = [], []
    for i, shard in enumerate((wqkv_a, wq_b, wuk, wuv)):
        rows, cols = shard.shape
        send = nl.ndarray(shard.shape, dtype=shard.dtype, buffer=nl.private_hbm, name=scoped_name(f"gk_{tag}{i}_send"))
        # One DMA, not row chunks: a chunked staging copy leaves later chunks queued behind unrelated
        # DMAs on the same in-order queue, delaying the all-gather start.
        _stage_copy(send, shard, rows, cols)
        sends.append(send)
        fulls.append(
            nl.ndarray(
                (rows * cp_degree, cols),
                dtype=shard.dtype,
                buffer=nl.private_hbm,
                name=scoped_name(f"gk_{tag}{i}_full"),
            )
        )
    ncc.all_gather(
        dsts=fulls,
        srcs=sends,
        replica_group=_replica_group(cp_replica_groups, cp_degree),
        collective_dim=0,
        priority=priority,
    )
    ret = tuple(f if shp is None else f.reshape(shp) for f, shp in zip(fulls, full_shapes))
    if send_life():
        _SEND_LIVE[id(ret[0])] = sends
    return ret


def _stage_send(src, tag: str):
    """A private_hbm copy of ``src``: a collective source cannot be an I/O tensor on LNC1."""
    rows, cols = src.shape
    send = nl.ndarray((rows, cols), dtype=src.dtype, buffer=nl.private_hbm, name=scoped_name(f"{tag}_send"))
    _stage_copy(send, src, rows, cols)
    return send


# Transport dtype of a collective that carries raw (non-bf16) bytes. fp8 / packed fp4 data moved as
# bf16 comes out of the all-gather altered (a plain DMA copy as bf16 is exact, so it is the collective's
# bf16 path): only an integer dtype is bit-transparent.
_RAW_DTYPE = nl.uint16


def _as_dtype(t, dtype):
    return t if t.dtype == dtype else t.view(dtype)


def _gather_rows(chunks, groups, priority, tag: str):
    """All-gather ``chunks`` (each this rank's row chunk) over ``groups`` in ONE coalesced collective.

    Returns the ``[rows * n, cols]`` gathers in the chunks' dtypes.
    """
    n = len(groups[0])
    sends, fulls = [], []
    for i, c in enumerate(chunks):
        rows, cols = c.shape
        sends.append(_stage_send(c, f"gk_{tag}{i}"))
        fulls.append(
            nl.ndarray((rows * n, cols), dtype=c.dtype, buffer=nl.private_hbm, name=scoped_name(f"gk_{tag}{i}_full"))
        )
    ncc.all_gather(dsts=fulls, srcs=sends, replica_group=_replica_group(groups, n), collective_dim=0, priority=priority)
    return fulls


def _stage1_cols(x_g, w, out, S: int, H: int, dmat: bool = False):
    """``out[S, n] = x_g[S, H] @ w[H, n]`` bf16, with ``w`` resident (this rank's wqkv_a column block)."""
    n = w.shape[1]
    sbm = SbufManager(
        sb_lower_bound=0,
        sb_upper_bound=nl.tile_size.total_available_sbuf_size,
        use_auto_alloc=False,
        logger=get_logger("gk_stage1"),
    )
    sbm.open_scope()
    w_sb = _load_bf16_weights(w, H, n, sbm, name="s1_w")
    tiles = [(s0, min(P_MAX, S - s0)) for s0 in range(0, S, P_MAX)]
    if dmat and H % P_MAX == 0 and all(sz % 16 == 0 for _, sz in tiles):
        # GLM_GK_UP_S1DMAT=1 (the caller's ``dmat``): double-buffered token tiles loaded K-major by one HW DGE DMA
        # transpose each (no staging, no PE transposes); tile i + 1's load is issued before tile i's matmuls, so it
        # lands under them. The same bf16 values reach the same matmuls.
        n_k = H // P_MAX
        x_ts = [
            sbm.alloc_stack((P_MAX, n_k, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"s1_xt{i}", align=64)
            for i in range(2)
        ]

        def _issue_load(i):
            s0, sz = tiles[i]
            n_c = sz // 16
            # src[a, k, c, d] = x[s0 + 16c + a, 128k + d], permuted (3, 1, 2, 0) to dst[d, k, c, a]:
            # (c, a) is the token.
            nisa.dma_transpose(
                dst=x_ts[i % 2].ap(pattern=[[n_k * P_MAX, P_MAX], [P_MAX, n_k], [16, n_c], [1, 16]], offset=0),
                src=x_g.ap(pattern=[[H, 16], [P_MAX, n_k], [16 * H, n_c], [1, P_MAX]], offset=s0 * H),
                axes=(3, 1, 2, 0),
                dge_mode=dge_mode.hwdge,
            )

        _issue_load(0)
        for i, (s0, sz) in enumerate(tiles):
            if i + 1 < len(tiles):
                _issue_load(i + 1)
            sbm.open_scope()
            (o_sb,) = _bf16_matmul_split(x_ts[i % 2], w_sb, H, P_MAX, n, [], sbm, name=f"s1_mm{s0}")
            nisa.dma_copy(dst=out[s0 : s0 + sz, 0:n], src=o_sb[0:sz, 0:n])
            sbm.close_scope()
        sbm.close_scope()
        return
    for s0 in range(0, S, P_MAX):
        sbm.open_scope()
        sz = min(P_MAX, S - s0)
        x_t = sbm.alloc_stack((P_MAX, H // P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"s1_x{s0}", align=64)
        _load_x_bf16_tile(x_g, x_t, s0, sz, P_MAX, H, sbm)
        (o_sb,) = _bf16_matmul_split(x_t, w_sb, H, P_MAX, n, [], sbm, name=f"s1_mm{s0}")
        nisa.dma_copy(dst=out[s0 : s0 + sz, 0:n], src=o_sb[0:sz, 0:n])
        sbm.close_scope()
    sbm.close_scope()


def _stage1_cols_xag2(x_g, w, out, g: int, T: int, H: int):
    """GLM_GK_PRJ_XAG=2: ``_stage1_cols(dmat=True)`` over the two row-chunk gathers (``x_g`` ``[2, g * T / 2, H]``):
    chunk 0's 128-row tiles first (they land while chunk 1's all-gather runs), then chunk 1's, one double-buffered
    stream of the same HW DGE DMA transposes. Tile row ``r`` of chunk ``c`` is member ``j = r // (T / 2)``'s row
    ``c T / 2 + r % (T / 2)``, so its output goes to ``out`` row ``j T + c T / 2 + r % (T / 2)`` (one store per member
    block): every row is the same matmuls on the same row as in ``_stage1_cols``, so ``out`` holds the same bits."""
    n = w.shape[1]
    th = T // 2
    sbm = SbufManager(
        sb_lower_bound=0,
        sb_upper_bound=nl.tile_size.total_available_sbuf_size,
        use_auto_alloc=False,
        logger=get_logger("gk_stage1"),
    )
    sbm.open_scope()
    w_sb = _load_bf16_weights(w, H, n, sbm, name="s1_w")
    n_k = H // P_MAX
    rows_c = g * th
    tiles = [(c, r0, min(P_MAX, rows_c - r0)) for c in range(2) for r0 in range(0, rows_c, P_MAX)]
    x_ts = [
        sbm.alloc_stack((P_MAX, n_k, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"s1_xt{i}", align=64)
        for i in range(2)
    ]

    def _issue_load(i):
        c, r0, sz = tiles[i]
        nisa.dma_transpose(
            dst=x_ts[i % 2].ap(pattern=[[n_k * P_MAX, P_MAX], [P_MAX, n_k], [16, sz // 16], [1, 16]], offset=0),
            src=x_g[c].ap(pattern=[[H, 16], [P_MAX, n_k], [16 * H, sz // 16], [1, P_MAX]], offset=r0 * H),
            axes=(3, 1, 2, 0),
            dge_mode=dge_mode.hwdge,
        )

    _issue_load(0)
    for i, (c, r0, sz) in enumerate(tiles):
        if i + 1 < len(tiles):
            _issue_load(i + 1)
        sbm.open_scope()
        (o_sb,) = _bf16_matmul_split(x_ts[i % 2], w_sb, H, P_MAX, n, [], sbm, name=f"s1_mm{c}_{r0}")
        for b0 in range(0, sz, th):
            j = (r0 + b0) // th
            dst0 = j * T + c * th
            nisa.dma_copy(dst=out[dst0 : dst0 + th, 0:n], src=o_sb[b0 : b0 + th, 0:n])
        sbm.close_scope()
    sbm.close_scope()


def _rows_of_position(src_2d, rank_hbm, T: int, C: int, name: str):
    """``dst[T, C] = src_2d[rank * T + t]``: this rank's own rows of a group-gathered ``[g*T, C]`` tensor.

    The rank is loaded here, not held in SBUF across the stack: at long contexts the indexer uses
    the whole SBUF, so a stack-lifetime SBUF scalar does not fit.
    """
    dst = nl.ndarray((T, C), dtype=src_2d.dtype, buffer=nl.private_hbm, name=scoped_name(f"{name}_own"))
    rank_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf, name=scoped_name(f"{name}_rank"))
    nisa.dma_copy(dst=rank_sb, src=rank_hbm.reshape((1, 1)))
    g = src_2d.shape[0] // T
    per_rank = src_2d.reshape((g, T * C))
    # 128-row chunks: T > 128 (a 32-rank stage at 8k) does not fit one SBUF tile.
    for t0 in range(0, T, P_MAX):
        n = min(P_MAX, T - t0)
        stage = nl.ndarray((P_MAX, C), dtype=src_2d.dtype, buffer=nl.sbuf, name=scoped_name(f"{name}_stage{t0}"))
        nisa.dma_copy(
            dst=stage[0:n, 0:C],
            src=per_rank.ap(pattern=[[C, n], [1, C]], offset=t0 * C, scalar_offset=rank_sb, indirect_dim=0),
        )
        nisa.dma_copy(dst=dst[t0 : t0 + n, 0:C], src=stage[0:n, 0:C])
    return dst


def _hybrid_qkv_gathers(normed, wqkv_a_p, groups, g: int, T: int, H: int, x_send=None, s1_dmat=False):
    """The collectives of the head-TP QKV's first stage: the group's tokens and their wqkv_a column blocks.

    Returns ``(x_g, s1_all)``. A full layer issues them before the indexer key all-gather, so the QKV
    that follows runs under it instead of queuing its gathers behind it.
    ``s1_dmat`` (GLM_GK_UP_S1DMAT): stage 1 DMA-transposes the gathered row-major tokens (``_stage1_cols(dmat=True)``).
    """
    gT = g * T
    if x_send is None:
        x_send = nl.ndarray((T, H), dtype=nl.bfloat16, buffer=nl.private_hbm, name=scoped_name("gk_hy_x_send"))
        nisa.dma_copy(dst=x_send[0:T, 0:H], src=normed.reshape((T, H))[0:T, 0:H])
    xag = _prj_xag() if s1_dmat else 0
    if xag:
        # GLM_GK_PRJ_XAG=2: x_g holds the two row-chunk gathers back to back ([2, g * T / 2, H], chunk c = every
        # member's rows [c T / 2, (c + 1) T / 2)); only its shape reaches the QKV kernel (it reads the stage-1 output).
        th = T // 2
        kernel_assert(
            T % 2 == 0 and P_MAX % th == 0 and th % 16 == 0 and H % P_MAX == 0,
            f"{_ERR} GLM_GK_PRJ_XAG=2: T / 2 = {th} rows must divide {P_MAX} and be a multiple of 16",
        )
        x_g = nl.ndarray((2, g * th, H), dtype=nl.bfloat16, buffer=nl.private_hbm, name=scoped_name("gk_hy_x_g"))
        for c in range(2):
            _all_gather(x_g[c], x_send[c * th : (c + 1) * th], groups, g)
        bw = wqkv_a_p.shape[1]
        s1_send = nl.ndarray((gT, bw), dtype=nl.bfloat16, buffer=nl.private_hbm, name=scoped_name("gk_hy_s1_send"))
        with sbm_name_scope("gk_hy_s1_"):
            _stage1_cols_xag2(x_g, wqkv_a_p, s1_send, g, T, H)
        s1_all = nl.ndarray((g * gT, bw), dtype=nl.bfloat16, buffer=nl.private_hbm, name=scoped_name("gk_hy_s1_all"))
        _all_gather(s1_all, s1_send, groups, g)
        return x_g.reshape((gT, H)), s1_all
    x_g = nl.ndarray((gT, H), dtype=nl.bfloat16, buffer=nl.private_hbm, name=scoped_name("gk_hy_x_g"))
    _all_gather(x_g, x_send, groups, g)

    bw = wqkv_a_p.shape[1]
    s1_send = nl.ndarray((gT, bw), dtype=nl.bfloat16, buffer=nl.private_hbm, name=scoped_name("gk_hy_s1_send"))
    with sbm_name_scope("gk_hy_s1_"):
        _stage1_cols(x_g, wqkv_a_p, s1_send, gT, H, dmat=s1_dmat)
    s1_all = nl.ndarray((g * gT, bw), dtype=nl.bfloat16, buffer=nl.private_hbm, name=scoped_name("gk_hy_s1_all"))
    _all_gather(s1_all, s1_send, groups, g)
    return x_g, s1_all


def _hybrid_qkv(
    normed, attn_w, q_norm_w, kv_norm_w, hy, kv_cache, dims, groups, g: int, T: int, H: int, x_send=None, pre=None,
    kv_slot_off=0, s1_dmat=False,
):
    """Head-TP QKV over the TP group's ``g*T`` tokens; returns this rank's ``(q_lift, q_pe)`` for its own
    tokens and ALL heads, the group's ``qr``, and the cache (own rows written; the others are -1 slots).

    ``pre``: ``_hybrid_qkv_gathers``'s result when the caller issued it already.
    """
    wqkv_a_p, wq_b_p, wuk_p, _ = attn_w
    n_heads, nope, R, L, q_lora, eps = dims
    hpr = n_heads // g
    gT = g * T
    # s1_dmat: GLM_GK_UP_S1DMAT (the caller's layer-level verdict).
    if pre is None:
        pre = _hybrid_qkv_gathers(normed, wqkv_a_p, groups, g, T, H, x_send, s1_dmat=s1_dmat)
    x_g, s1_all = pre
    bw = wqkv_a_p.shape[1]

    if hy.get("rope_full") is not None:
        # GLM_GK_ROPE_ONCE: the launch's tables, built once by the stack from the same cos_g / sin_g.
        cos_full, sin_full = hy["rope_full"]
    else:
        cos_full, sin_full = _build_rope_tables(hy["cos_g"], hy["sin_g"], gT, R)
    qkv_hy_opt = _prj_qkv_opt()
    # The q outputs are the all-to-all sources: chunk j (rank j's T tokens) is contiguous.
    q_send = nl.ndarray((g, T, hpr, L), dtype=nl.bfloat16, buffer=nl.private_hbm, name=scoped_name("gk_hy_q_send"))
    qpe_send = nl.ndarray((g, T, hpr, R), dtype=nl.bfloat16, buffer=nl.private_hbm, name=scoped_name("gk_hy_qpe_send"))
    with sbm_name_scope("gk_qkv_"):
        _, _, qr_g, kv_cache = mla_qkv_cte_kernel(
            x_hbm_mx=x_g.reshape((1, gT, H)),
            wqkv_a_hbm=wqkv_a_p,
            wqkv_a_scale_hbm=None,
            wq_b_hbm=wq_b_p,
            wq_b_scale_hbm=None,
            q_norm_gamma_hbm=q_norm_w,
            kv_norm_gamma_hbm=kv_norm_w,
            wuk_hbm=wuk_p,
            cos_cache_hbm=cos_full,
            sin_cache_hbm=sin_full,
            n_heads=hpr,
            qk_nope_head_dim=nope,
            qk_rope_head_dim=R,
            kv_lora_rank=L,
            qk_lora_rank=q_lora,
            norm_eps=eps,
            precision=MlaPrecision.BF16,
            kv_cache_hbm=kv_cache,
            slot_mapping_hbm=hy["slot_mapping_g"].reshape((1, gT)),
            kv_slot_offset=kv_slot_off,
            stage1_hbm=s1_all.reshape((g, gT, bw)),
            q_lift_out=q_send.reshape((1, gT, hpr, L)),
            q_pe_out=qpe_send.reshape((1, gT, hpr, R)),
            # GLM_GK_PRJ_QKV: schedule-only hy_opt bits.
            **({"hy_opt": qkv_hy_opt} if qkv_hy_opt else {}),
        )

    q_recv = nl.ndarray((g, T, hpr, L), dtype=nl.bfloat16, buffer=nl.private_hbm, name=scoped_name("gk_hy_q_recv"))
    qpe_recv = nl.ndarray((g, T, hpr, R), dtype=nl.bfloat16, buffer=nl.private_hbm, name=scoped_name("gk_hy_qpe_recv"))
    ncc.all_to_all(
        srcs=[q_send, qpe_send], dsts=[q_recv, qpe_recv], replica_group=_replica_group(groups, g), collective_dim=0
    )
    # Source j sent heads [j*hpr, (j+1)*hpr) of this rank's tokens: interleave them into [T, n_heads, *].
    q_lift = nl.ndarray((1, T, n_heads, L), dtype=nl.bfloat16, buffer=nl.private_hbm, name=scoped_name("gk_hy_q_lift"))
    q_pe = nl.ndarray((1, T, n_heads, R), dtype=nl.bfloat16, buffer=nl.private_hbm, name=scoped_name("gk_hy_q_pe"))
    for dst, src, d in ((q_lift, q_recv, L), (q_pe, qpe_recv, R)):
        dst2 = dst.reshape((T, n_heads * d))
        src2 = src.reshape((g * T, hpr * d))
        for j in range(g):
            nisa.dma_copy(dst=dst2[0:T, j * hpr * d : (j + 1) * hpr * d], src=src2[j * T : (j + 1) * T, 0 : hpr * d])
    hy["x_g"] = x_g
    return q_lift, q_pe, qr_g, kv_cache


def _hybrid_oproj(out_attn, wuv_p, wo_shard, wo_group_rank, groups, g: int, n_heads: int, T, H, L):
    """All-to-all the latent output to head slices, V-up + o_proj this rank's heads for the group's tokens,
    reduce-scatter the partials back to each rank's own tokens."""
    hpr = n_heads // g
    d_v = wuv_p.shape[1]
    kt = hpr * d_v // P_MAX
    send = nl.ndarray((g, T, hpr * L), dtype=nl.bfloat16, buffer=nl.private_hbm, name=scoped_name("gk_hy_o_send"))
    src2 = out_attn.reshape((T, n_heads * L))
    copies = [
        nisa.dma_copy(dst=send[j, 0:T, 0 : hpr * L], src=src2[0:T, j * hpr * L : (j + 1) * hpr * L]) for j in range(g)
    ]
    recv = nl.ndarray((g, T, hpr * L), dtype=nl.bfloat16, buffer=nl.private_hbm, name=scoped_name("gk_hy_o_recv"))
    # The copies read every query's output, so the sync follows all of the attention's row gathers.
    sync = _cc_sync(send[g - 1, T - 1 : T, 0:16], groups, g, "o", after=copies)
    a2a = ncc.all_to_all(srcs=[send], dsts=[recv], replica_group=_replica_group(groups, g), collective_dim=0)
    a2a.depends_on(sync)
    partial = nl.ndarray((g * T, H), dtype=nl.bfloat16, buffer=nl.private_hbm, name=scoped_name("gk_hy_partial"))
    if isinstance(wo_shard, tuple):
        # GLM_GK_PRJ_OMX=1: wo arrives as (MXFP8 fp8x4 rows, e8m0 scales); one fused V-up + MX o_proj call over the
        # group's g * T tokens (the activation is quantized to MXFP8 in SBUF; no vup_kt round trip).
        wo_q, wo_s = wo_shard
        with sbm_name_scope("gk_hy_vomx_"):
            glm_vup_oproj_mx_kernel(recv.reshape((g * T, hpr * L)), wuv_p, wo_q, wo_s, partial, kv_lora_rank=L)
    elif P_MAX % T == 0 or T % P_MAX == 0:
        # recv is [g*T, hpr*L] in source-rank-major token order: V-up whole 128-token chunks of it
        # (wuv loaded once per chunk, not per source) into the K-major [kt, 128, tokens] stationary
        # the partial kernel wants, the token columns already in the partial's row order. T > 128
        # (a 32-rank stage at 8k) splits each source's tokens into 128-token chunks.
        cs = min(g * T, P_MAX)
        nc = g * T // cs
        recv2 = recv.reshape((g * T, hpr * L))
        vup_kt = nl.ndarray(
            (nc, kt, P_MAX, cs), dtype=nl.bfloat16, buffer=nl.private_hbm, name=scoped_name("gk_hy_vup")
        )
        for c in range(nc):
            with sbm_name_scope(f"gk_hy_vup{c}_"):
                mla_vup_cte_kernel(
                    recv2[c * cs : (c + 1) * cs, 0 : hpr * L].reshape((1, cs, hpr * L)),
                    wuv_p,
                    kv_lora_rank=L,
                    output_tensor=vup_kt[c],
                )
        # The chunks play the partial kernel's "source ranks": wo is streamed once for all of them. More than
        # 8 chunks (T > 128) overflow its SBUF, so each call takes at most 8 and streams wo again.
        ncb = min(nc, 8)
        for b0 in range(0, nc, ncb):
            with sbm_name_scope("gk_oprp_" if b0 == 0 else f"gk_oprp{b0}_"):
                mla_oproj_partial_cte_kernel(
                    (vup_kt if ncb == nc else vup_kt[b0 : b0 + ncb]).reshape((ncb * kt, P_MAX, cs)),
                    wo_shard,
                    wo_group_rank,
                    n_ranks=ncb,
                    hdv_total=ncb * hpr * d_v,
                    output_tensor=partial if ncb == nc else partial[b0 * cs : (b0 + ncb) * cs],
                    input_pre_sliced=True,
                )
    else:
        # Block j = source j's tokens, K-major over this rank's Hdv rows: the partial kernel's pre-sliced input.
        vup_kt = nl.ndarray((g, kt, P_MAX, T), dtype=nl.bfloat16, buffer=nl.private_hbm, name=scoped_name("gk_hy_vup"))
        for j in range(g):
            with sbm_name_scope(f"gk_hy_vup{j}_"):
                mla_vup_cte_kernel(recv[j].reshape((1, T, hpr * L)), wuv_p, kv_lora_rank=L, output_tensor=vup_kt[j])
        with sbm_name_scope("gk_oprp_"):
            mla_oproj_partial_cte_kernel(
                vup_kt.reshape((g * kt, P_MAX, T)),
                wo_shard,
                wo_group_rank,
                n_ranks=g,
                hdv_total=n_heads * d_v,
                output_tensor=partial,
                input_pre_sliced=True,
            )
    out = nl.ndarray((T, H), dtype=nl.bfloat16, buffer=nl.private_hbm, name=scoped_name("gk_hy_out"))
    _reduce_scatter(out, partial, groups, g)
    return out.reshape((1, T, H))


def _build_rope_tables(cos, sin, T: int, R: int):
    """``[T, R/2]`` fp32 half tables -> ``[1, T, R]`` bf16 ``[c, c]`` tables for the QKV kernel."""
    half = R // 2
    cos_full = nl.ndarray((1, T, R), dtype=nl.bfloat16, buffer=nl.private_hbm, name=scoped_name("gk_cos_full"))
    sin_full = nl.ndarray((1, T, R), dtype=nl.bfloat16, buffer=nl.private_hbm, name=scoped_name("gk_sin_full"))
    for src, dst, tag in ((cos, cos_full, "cos"), (sin, sin_full, "sin")):
        dst2d = dst.reshape((T, R))
        for r0 in range(0, T, P_MAX):
            n = min(P_MAX, T - r0)
            h = nl.ndarray((P_MAX, half), dtype=nl.float32, buffer=nl.sbuf, name=scoped_name(f"gk_{tag}_h{r0}"))
            f = nl.ndarray((P_MAX, R), dtype=nl.bfloat16, buffer=nl.sbuf, name=scoped_name(f"gk_{tag}_f{r0}"))
            nisa.dma_copy(dst=h[:n, :], src=src[r0 : r0 + n, 0:half])
            nisa.tensor_copy(dst=f[:n, 0:half], src=h[:n, :])
            nisa.tensor_copy(dst=f[:n, half:R], src=h[:n, :])
            nisa.dma_copy(dst=dst2d[r0 : r0 + n, 0:R], src=f[:n, :])
    return cos_full, sin_full


def rope_once() -> bool:
    """``GLM_GK_ROPE_ONCE=1``: the hybrid QKV's ``[1, g*T, R]`` bf16 RoPE tables are built ONCE per launch by the
    stack (``_build_rope_tables`` of the launch inputs ``cos_g`` / ``sin_g``, at the launch start) instead of in every
    layer's ``_hybrid_qkv``: the inputs are the same tensors in every layer, so every layer's QKV reads the same bytes;
    only the per-layer rebuild (its DMAs and copies on the Sync queue ahead of the QKV's gamma loads) goes. Unset or 0
    (default): off."""
    raw = os.environ.get("GLM_GK_ROPE_ONCE", "").strip()
    kernel_assert(raw in ("", "0", "1"), f"{_ERR} GLM_GK_ROPE_ONCE={raw!r}: use 1, 0 or leave it unset")
    return raw == "1"


def _add_index_offset(idx, n: int, offset: int, name: str):
    """``idx += offset`` where ``idx >= 0`` (int32 ``[P, 1]`` SBUF tile, in place): a negative padding index stays out
    of range for ``oob_mode.skip``. The engine computes in fp32, so offset + index must stay below 2^24."""
    tmp = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf, name=scoped_name(name))
    nisa.tensor_scalar(
        dst=tmp[:n, :], data=idx[:n, :], op0=nl.greater_equal, operand0=0, op1=nl.multiply, operand1=offset
    )
    nisa.tensor_tensor(dst=idx[:n, :], data1=idx[:n, :], data2=tmp[:n, :], op=nl.add)


def _gather_local_kv_rows(
    kv_cache, slot_mapping, T: int, row: int, on_gpsimd: bool = False, name: str = "gk_kv_local", slot_off: int = 0,
):
    """This rank's KV rows in token order: ``kv_local[t] = kv_cache[slot_mapping[t]]``.

    The gathered cache is rank-major: rank r's ``T`` rows land at ``[r*T, (r+1)*T)``, which is
    what the attention kernel's CP position conversion expects (``S_kv_local == T``).
    """
    n_slots = kv_cache.size // row
    flat = kv_cache.reshape((n_slots, row))
    kv_local = nl.ndarray((T, row), dtype=nl.bfloat16, buffer=nl.private_hbm, name=scoped_name(name))
    dma_kw = dict(dge_mode=dge_mode.swdge) if on_gpsimd else {}
    for r0 in range(0, T, P_MAX):
        n = min(P_MAX, T - r0)
        slot = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf, name=scoped_name(f"gk_kv_slot{r0}"))
        nisa.dma_copy(dst=slot[:n, :], src=slot_mapping.reshape((T, 1))[r0 : r0 + n, 0:1], **dma_kw)
        if slot_off:
            _add_index_offset(slot, n, slot_off, f"{name}_off{r0}")
        stage = nl.ndarray((P_MAX, row), dtype=nl.bfloat16, buffer=nl.sbuf, name=scoped_name(f"gk_kv_stage{r0}"))
        nisa.dma_copy(
            dst=stage[:n, :row],
            src=flat.ap(pattern=[[row, n], [1, row]], offset=0, vector_offset=slot[:n, :], indirect_dim=0),
            dge_mode=dge_mode.swdge,
        )
        nisa.dma_copy(dst=kv_local[r0 : r0 + n, 0:row], src=stage[:n, :row], **dma_kw)
    if kvl_life():
        _KVL_LIVE.append(kv_local)
    return kv_local


def _gather_local_kv_pages(
    kv_cache, block_table, ctx_local: int, block_size: int, row: int, on_gpsimd: bool = False, tag: str = "",
    page_off: int = 0,
):
    """This rank's cached context in logical order: ``kv_local[l] = cache[bt[l // bs] * bs + l % bs]``.

    One indirect DMA per 128 pages, with a whole page as the unit row, so the block table itself is
    the index vector. Rows past the live context come along too; the attention and indexer masks
    give them positions later than every query.
    """
    kernel_assert(
        ctx_local % block_size == 0, f"{_ERR} kv_ctx_local {ctx_local} must be whole {block_size}-token pages"
    )
    n_pages = ctx_local // block_size
    page = block_size * row
    n_cache_pages = kv_cache.size // page
    cache_pages = kv_cache.reshape((n_cache_pages, page))
    kv_local = nl.ndarray(
        (ctx_local, row), dtype=nl.bfloat16, buffer=nl.private_hbm, name=scoped_name(f"gk_kv_local{tag}")
    )
    lin_pages = kv_local.reshape((n_pages, page))
    dma_kw = dict(dge_mode=dge_mode.swdge) if on_gpsimd else {}
    for p0 in range(0, n_pages, P_MAX):
        n = min(P_MAX, n_pages - p0)
        idx = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf, name=scoped_name(f"gk_kv_bt{tag}{p0}"))
        nisa.dma_copy(dst=idx[:n, :], src=block_table.reshape((block_table.size, 1))[p0 : p0 + n, 0:1], **dma_kw)
        if page_off:
            _add_index_offset(idx, n, page_off, f"gk_kv_bt_off{tag}{p0}")
        stage = nl.ndarray((P_MAX, page), dtype=nl.bfloat16, buffer=nl.sbuf, name=scoped_name(f"gk_kv_pg{tag}{p0}"))
        nisa.dma_copy(
            dst=stage[:n, :page],
            src=cache_pages.ap(pattern=[[page, n], [1, page]], offset=0, vector_offset=idx[:n, :], indirect_dim=0),
            dge_mode=dge_mode.swdge,
        )
        nisa.dma_copy(dst=lin_pages[p0 : p0 + n, 0:page], src=stage[:n, :page], **dma_kw)
    if kvl_life():
        _KVL_LIVE.append(kv_local)
    return kv_local


def _kv_spare_row(ctx_local: int) -> bool:
    """``GLM_GK_KV_PLANS=C``: whether the stack's KV buffer keeps one spare row past its ``cp * ctx_local`` gathered
    rows. No layer reads or writes the row. The PP4 stage programs were built and validated with it in their KV buffer
    (C = 64), so the buffer keeps it where C is a power of two below ``ctx_local`` that divides it; unset (PP16), or a C
    of the whole bucket: no spare row."""
    raw = os.environ.get("GLM_GK_KV_PLANS", "").strip()
    c = int(raw) if raw else 0
    if not c or c >= ctx_local:
        return False
    kernel_assert(
        c > 0 and c & (c - 1) == 0 and ctx_local % c == 0,
        f"{_ERR} GLM_GK_KV_PLANS={raw!r} must be a power of two that divides kv_ctx_local {ctx_local}",
    )
    return True


def _build_blockwise_mapping(
    gathered_f32, ep_rank, aff_col: int, T_moe: int, E_local: int, top_k: int, B: int, ep_cols: bool = True,
):
    """Blockwise token -> expert mapping of this rank's local experts, on device.

    The same algorithm as the framework's ``build_blockwise_mapping`` (non-TP-sharded path): the
    nonzero rows of each local expert's affinity column, padded to whole blocks, laid end to end. The routed experts
    read their token ids straight from those rows (``find_nonzero_indices``) through a per-block row-block table, so
    the rows are never flattened into a token-position table.

    Args:
        gathered_f32: ``[T_moe, R4]`` fp32 view of the gathered packed rows; the dense global
            affinity tail starts at fp32 column ``aff_col``.
        ep_rank: ``[1, 1]`` int32 EP rank; local expert ``e`` is global ``ep_rank * E_local + e``.
        ep_cols: False (``GLM_GK_MOE_AFF8``): the local experts are columns ``[aff_col, aff_col + E_local)`` of
            ``gathered_f32`` (no ``ep_rank * E_local`` offset).

    Returns:
        ``((indices [E_local, T_moe] int32, block_rows [N] int32), block_to_expert [N] int32, conditions [N+2] int32)``.
    """
    k_local = min(top_k, E_local)
    pairs = T_moe * k_local
    N = min(div_ceil(pairs - (E_local - 1), B) + E_local - 1, pairs)
    f_len = min(P_MAX, T_moe // 16)
    kernel_assert(B % f_len == 0, f"{_ERR} moe block size {B} must be a multiple of f_len {f_len}")
    kernel_assert(B & (B - 1) == 0, f"{_ERR} moe block size {B} must be a power of two")
    lg_b = B.bit_length() - 1

    # Column of this rank's first local expert inside the gathered rows.
    col = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf, name=scoped_name("gk_map_col"))
    if ep_cols:
        nisa.dma_copy(dst=col, src=ep_rank.reshape((1, 1)))
        nisa.tensor_scalar(dst=col, data=col, op0=nl.multiply, operand0=E_local, op1=nl.add, operand1=aff_col)
    else:
        nisa.memset(dst=col, value=aff_col)
    col_hbm = nl.ndarray((1,), dtype=nl.int32, buffer=nl.private_hbm, name=scoped_name("gk_map_col_hbm"))
    nisa.dma_copy(dst=col_hbm.reshape((1, 1)), src=col)

    indices, counts = find_nonzero_indices(
        gathered_f32,
        col_start_id=col_hbm,
        n_cols=E_local,
        chunk_size=min(T_moe, _MAPPING_MAX_CHUNK),
    )

    # blocks per expert, inclusive / exclusive cumsum over the E_local experts (tiny, on one row).
    cnt = nl.ndarray((1, E_local), dtype=nl.int32, buffer=nl.sbuf, name=scoped_name("gk_map_cnt"))
    nisa.dma_copy(dst=cnt, src=counts.reshape((1, E_local)))
    nisa.tensor_scalar(dst=cnt, data=cnt, op0=nl.add, operand0=B - 1)
    nisa.tensor_scalar(dst=cnt, data=cnt, op0=nl.right_shift, operand0=lg_b, engine=nisa.engine.vector)
    bpe = nl.ndarray((1, E_local), dtype=nl.float32, buffer=nl.sbuf, name=scoped_name("gk_map_bpe"))
    nisa.tensor_copy(dst=bpe, src=cnt)
    incl = nl.ndarray((1, E_local), dtype=nl.float32, buffer=nl.sbuf, name=scoped_name("gk_map_incl"))
    nisa.tensor_copy(dst=incl[0:1, 0:1], src=bpe[0:1, 0:1])
    for e in range(1, E_local):
        nisa.tensor_tensor(dst=incl[0:1, e : e + 1], data1=incl[0:1, e - 1 : e], data2=bpe[0:1, e : e + 1], op=nl.add)
    excl = nl.ndarray((1, E_local), dtype=nl.float32, buffer=nl.sbuf, name=scoped_name("gk_map_excl"))
    nisa.tensor_tensor(dst=excl, data1=incl, data2=bpe, op=nl.subtract)

    # Each expert's block-run row offset in f_len units. Nothing reads it (the routed experts take the nonzero rows);
    # dropping these two instructions would change every served program.
    roff_f = nl.ndarray((1, E_local), dtype=nl.float32, buffer=nl.sbuf, name=scoped_name("gk_map_roff_f"))
    nisa.tensor_scalar(dst=roff_f, data=excl, op0=nl.multiply, operand0=float(B // f_len))
    roff = nl.ndarray((1, E_local), dtype=nl.int32, buffer=nl.sbuf, name=scoped_name("gk_map_roff"))
    nisa.tensor_copy(dst=roff, src=roff_f)

    # block_to_expert[n] = #{e >= 1 : n >= excl[e]};  conditions[n] = n < incl[E-1].
    iota_n = nl.ndarray((1, N + _COND_PAD), dtype=nl.float32, buffer=nl.sbuf, name=scoped_name("gk_map_iota"))
    nisa.iota(iota_n, [[1, N + _COND_PAD]], offset=0)
    acc = nl.ndarray((1, N), dtype=nl.float32, buffer=nl.sbuf, name=scoped_name("gk_map_bte_f"))
    nisa.memset(dst=acc, value=0.0)
    ge = nl.ndarray((1, N), dtype=nl.float32, buffer=nl.sbuf, name=scoped_name("gk_map_ge"))
    # Row block of block n: n + sum_{e >= 1, n >= excl[e]} (T_moe / B - bpe[e - 1]), at most the padding row block.
    kernel_assert(T_moe % B == 0, f"{_ERR} the nonzero-row routing needs T_moe {T_moe} to be whole {B}-row blocks")
    rb_f = nl.ndarray((1, N), dtype=nl.float32, buffer=nl.sbuf, name=scoped_name("gk_map_rb_f"))
    nisa.tensor_copy(dst=rb_f, src=iota_n[0:1, 0:N])
    step = nl.ndarray((1, E_local), dtype=nl.float32, buffer=nl.sbuf, name=scoped_name("gk_map_rb_step"))
    nisa.tensor_scalar(dst=step, data=bpe, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=float(T_moe // B))
    for e in range(1, E_local):
        nisa.tensor_scalar(dst=ge, data=iota_n[0:1, 0:N], op0=nl.greater_equal, operand0=excl[0:1, e : e + 1])
        nisa.tensor_tensor(dst=acc, data1=acc, data2=ge, op=nl.add)
        nisa.tensor_scalar(dst=ge, data=ge, op0=nl.multiply, operand0=step[0:1, e - 1 : e])
        nisa.tensor_tensor(dst=rb_f, data1=rb_f, data2=ge, op=nl.add)
    bte = nl.ndarray((1, N), dtype=nl.int32, buffer=nl.sbuf, name=scoped_name("gk_map_bte"))
    nisa.tensor_copy(dst=bte, src=acc)
    block_to_expert = nl.ndarray((N,), dtype=nl.int32, buffer=nl.private_hbm, name=scoped_name("gk_block_to_expert"))
    nisa.dma_copy(dst=block_to_expert.reshape((1, N)), src=bte)

    cond_f = nl.ndarray((1, N + _COND_PAD), dtype=nl.float32, buffer=nl.sbuf, name=scoped_name("gk_map_cond_f"))
    nisa.tensor_scalar(dst=cond_f, data=iota_n, op0=nl.less, operand0=incl[0:1, E_local - 1 : E_local])
    # The two trailing pads must read 0 even when every block is live.
    nisa.memset(dst=cond_f[0:1, N : N + _COND_PAD], value=0.0)
    cond = nl.ndarray((1, N + _COND_PAD), dtype=nl.int32, buffer=nl.sbuf, name=scoped_name("gk_map_cond"))
    nisa.tensor_copy(dst=cond, src=cond_f)
    conditions = nl.ndarray((N + _COND_PAD,), dtype=nl.int32, buffer=nl.private_hbm, name=scoped_name("gk_conditions"))
    nisa.dma_copy(dst=conditions.reshape((1, N + _COND_PAD)), src=cond)
    nisa.tensor_scalar(dst=rb_f, data=rb_f, op0=nl.minimum, operand0=float(E_local * T_moe // B))
    rb = nl.ndarray((1, N), dtype=nl.int32, buffer=nl.sbuf, name=scoped_name("gk_map_rb"))
    nisa.tensor_copy(dst=rb, src=rb_f)
    block_rows = nl.ndarray((N,), dtype=nl.int32, buffer=nl.private_hbm, name=scoped_name("gk_block_rows"))
    nisa.dma_copy(dst=block_rows.reshape((1, N)), src=rb)
    return (indices, block_rows), block_to_expert, conditions


# GLM_GK_SHMAP_ONCE: the shared expert's routing tables of the current trace, by (rows, block) (reset at the stack
# entry).
_SHMAP_CACHE: dict = {}


def _shmap_once() -> bool:
    """``GLM_GK_SHMAP_ONCE=1``: build the shared expert's constant routing tables (``_shared_expert_mapping``) once per
    stack trace and reuse them in every MoE layer, instead of a memset / iota / four small DMAs per layer (same
    values). Unset or 0 (default): per layer."""
    raw = os.environ.get("GLM_GK_SHMAP_ONCE", "").strip()
    kernel_assert(raw in ("", "0", "1"), f"{_ERR} GLM_GK_SHMAP_ONCE={raw!r}: use 1, 0 or leave it unset")
    return raw == "1"


def _shared_expert_mapping_cached(T: int, B: int):
    """``_shared_expert_mapping``, once per (T, B) and stack trace with GLM_GK_SHMAP_ONCE=1."""
    if not _shmap_once():
        return _shared_expert_mapping(T, B)
    key = (T, B)
    if key not in _SHMAP_CACHE:
        with sbm_name_scope("gk_shmap_once_"):
            _SHMAP_CACHE[key] = _shared_expert_mapping(T, B)
    return _SHMAP_CACHE[key]


def _shared_expert_mapping(T: int, B: int):
    """Routing of a one-expert MoE that takes every row once: token ``t`` at position ``t``, affinity 1."""
    n_blocks = div_ceil(T, B)
    n_pos = n_blocks * B
    pos = nl.ndarray((1, n_pos), dtype=nl.int32, buffer=nl.sbuf, name=scoped_name("gk_sh_pos"))
    nisa.memset(dst=pos, value=-1)
    nisa.iota(pos[0:1, 0:T], [[1, T]], offset=0)
    tpid = nl.ndarray((n_pos,), dtype=nl.int32, buffer=nl.private_hbm, name=scoped_name("gk_sh_tpid"))
    nisa.dma_copy(dst=tpid.reshape((1, n_pos)), src=pos)
    bte_sb = nl.ndarray((1, n_blocks), dtype=nl.int32, buffer=nl.sbuf, name=scoped_name("gk_sh_bte_sb"))
    nisa.memset(dst=bte_sb, value=0)
    bte = nl.ndarray((n_blocks,), dtype=nl.int32, buffer=nl.private_hbm, name=scoped_name("gk_sh_bte"))
    nisa.dma_copy(dst=bte.reshape((1, n_blocks)), src=bte_sb)
    ones = nl.ndarray((P_MAX, div_ceil(T, P_MAX)), dtype=nl.bfloat16, buffer=nl.sbuf, name=scoped_name("gk_sh_ones"))
    nisa.memset(dst=ones, value=1.0)
    aff = nl.ndarray((T, 1), dtype=nl.bfloat16, buffer=nl.private_hbm, name=scoped_name("gk_sh_aff"))
    for r0 in range(0, T, P_MAX):
        n = min(P_MAX, T - r0)
        nisa.dma_copy(dst=aff[r0 : r0 + n, 0:1], src=ones[:n, 0:1])
    return tpid, bte, aff, n_blocks


def _row_parallel_oproj(out_attn, wuv, wo_shard, wo_group_rank, wo_replica_groups, w_wo: int, n_heads: int, T, H, L):
    """V-up own tokens, all-to-all the activations over the wo group, local o_proj, reduce-scatter.

    The V-up emits K-tiles rank-slice-major and each rank only consumes its own Hdv K-slice from
    every source, so the all-to-all moves 1/w of an all-gather's bytes and the partial kernel gets
    its slices at static offsets. The wo weight never moves.
    """
    kernel_assert(not isinstance(wo_shard, tuple), f"{_ERR} GLM_GK_PRJ_OMX needs the head-TP (hybrid) o_proj")
    hdv = n_heads * wuv.shape[-1]
    kt_total = hdv // P_MAX
    # The V-up writes straight into the private send buffer (a collective source cannot be an I/O
    # or shared_hbm tensor), so there is no staging copy on the critical path.
    send = nl.ndarray((kt_total, P_MAX, T), dtype=nl.bfloat16, buffer=nl.private_hbm, name=scoped_name("gk_oprp_send"))
    with sbm_name_scope("gk_oproj_"):
        mla_vup_cte_kernel(out_attn, wuv, kv_lora_rank=L, output_tensor=send)
    recv = nl.ndarray((kt_total, P_MAX, T), dtype=nl.bfloat16, buffer=nl.private_hbm, name=scoped_name("gk_oprp_x"))
    ncc.all_to_all(srcs=[send], dsts=[recv], replica_group=_replica_group(wo_replica_groups, w_wo), collective_dim=0)
    partial = nl.ndarray((w_wo * T, H), dtype=nl.bfloat16, buffer=nl.private_hbm, name=scoped_name("gk_oprp_partial"))
    with sbm_name_scope("gk_oprp_"):
        partial = mla_oproj_partial_cte_kernel(
            recv,
            wo_shard,
            wo_group_rank,
            n_ranks=w_wo,
            hdv_total=hdv,
            output_tensor=partial,
            input_pre_sliced=True,
        )
    out = nl.ndarray((T, H), dtype=nl.bfloat16, buffer=nl.private_hbm, name=scoped_name("gk_oprp_out"))
    _reduce_scatter(out, partial, wo_replica_groups, w_wo)
    return out.reshape((1, T, H))


def _tp_dense_mlp(
    attn_o, r_in, post_norm_w, mlp_gate_w, mlp_up_w, mlp_down_w, wo_replica_groups, w_wo, T, H, eps, after_gather=None,
):
    """Dense layer: residual add + post-norm, then a sequence-parallel tensor-parallel SwiGLU MLP.

    Tokens are CP-sharded, so per-rank partials of a plain TP MLP would mix different rows. Instead:
    norm OWN tokens, all-gather the normed activations over the wo group, run the SwiGLU for ALL
    gathered tokens against this rank's ``I/w`` shard (gate/up ``[H, I/w]``, down ``[I/w, H]``), and
    reduce-scatter the partial sums back to this rank's rows. The weights never move.
    """
    mlp_out = nl.ndarray((T, H), dtype=nl.bfloat16, buffer=nl.shared_hbm, name=scoped_name("gk_mlp_out"))
    r_out = nl.ndarray((T, H), dtype=nl.bfloat16, buffer=nl.shared_hbm, name=scoped_name("gk_mlp_r_out"))
    x_send = nl.ndarray((T, H), dtype=nl.bfloat16, buffer=nl.private_hbm, name=scoped_name("gk_mlp_x_send"))
    with sbm_name_scope("gk_mlpnorm_"):
        nouts = rmsnorm_mx_prefill(
            hidden_states=attn_o,
            gamma=post_norm_w,
            eps=eps,
            residual=r_in.reshape((1, T, H)),
            emit_norm_bf16=True,
            norm_bf16_out=x_send,
        )
    _hbm_copy_rows(r_out, nouts[1].reshape((T, H)), T, H)
    x_all = nl.ndarray((w_wo * T, H), dtype=nl.bfloat16, buffer=nl.private_hbm, name=scoped_name("gk_mlp_x_all"))
    _all_gather(x_all, x_send, wo_replica_groups, w_wo)
    if after_gather is not None:
        after_gather()
    with sbm_name_scope("gk_mlp_"):
        mlp_outs = mlp(
            hidden_tensor=x_all.reshape((1, w_wo * T, H)),
            gate_proj_weights_tensor=mlp_gate_w,
            up_proj_weights_tensor=mlp_up_w,
            down_proj_weights_tensor=mlp_down_w,
            activation_fn=ActFnType.SiLU,
            normalization_type=NormType.NO_NORM,
            quantization_type=QuantizationType.NONE,
            output_dtype=nl.bfloat16,
            eps=eps,
            mode=ComputationMode.PREFILL,
        )
    partial = mlp_outs[0] if isinstance(mlp_outs, (list, tuple)) else mlp_outs
    rs_src = nl.ndarray((w_wo * T, H), dtype=nl.bfloat16, buffer=nl.private_hbm, name=scoped_name("gk_mlp_rs_src"))
    nisa.dma_copy(dst=rs_src[0 : w_wo * T, 0:H], src=partial.reshape((w_wo * T, H))[0 : w_wo * T, 0:H])
    rs_out = nl.ndarray((T, H), dtype=nl.bfloat16, buffer=nl.private_hbm, name=scoped_name("gk_mlp_rs_out"))
    _reduce_scatter(rs_out, rs_src, wo_replica_groups, w_wo)
    _hbm_copy_rows(mlp_out, rs_out, T, H)
    return mlp_out, r_out


def _shared_tp_add(rs_src, shared_out, rank_hbm, m: int, sh_rows: int, H: int):
    """The shared-TP add of ``_moe``: add the shared expert's ``[sh_rows, H]`` partial ``shared_out`` into this rank's
    block of ``sh_rows`` rows (block ``rank // m``) of the routed reduce-scatter source ``rs_src`` ``[T_rs, H]``, after
    every routed add, on the vector engine. The block index (rank // m) is loaded here, not held across the stack."""
    T_rs = rs_src.shape[0]
    blk_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf, name=scoped_name("gk_shtp_blk"))
    nisa.dma_copy(dst=blk_sb, src=rank_hbm.reshape((1, 1)))
    nisa.tensor_scalar(
        dst=blk_sb, data=blk_sb, op0=nl.right_shift, operand0=m.bit_length() - 1, engine=nisa.engine.vector
    )
    rs_blk = rs_src.reshape((T_rs // sh_rows, sh_rows * H))
    for r0 in range(0, sh_rows, P_MAX):
        n = min(P_MAX, sh_rows - r0)
        rows = rs_blk.ap(pattern=[[H, n], [1, H]], offset=r0 * H, scalar_offset=blk_sb, indirect_dim=0)
        a = nl.ndarray((P_MAX, H), dtype=nl.bfloat16, buffer=nl.sbuf, name=scoped_name(f"gk_shtp_a{r0}"))
        b = nl.ndarray((P_MAX, H), dtype=nl.bfloat16, buffer=nl.sbuf, name=scoped_name(f"gk_shtp_b{r0}"))
        nisa.dma_copy(dst=a[:n, :], src=rows, dge_mode=dge_mode.hwdge)
        nisa.dma_copy(dst=b[:n, :], src=shared_out[r0 : r0 + n, 0:H])
        nisa.tensor_tensor(dst=a[:n, :], data1=a[:n, :], data2=b[:n, :], op=nl.add)
        nisa.dma_copy(dst=rows, src=a[:n, :], dge_mode=dge_mode.hwdge)


def _moe(
    attn_o,
    r_in,
    ep_rank,
    post_norm_w,
    router_w,
    router_bias,
    gate_up_w,
    gate_up_scale,
    down_w,
    down_scale,
    shared_gate_up_w,
    shared_gate_up_scale,
    shared_down_w,
    shared_down_scale,
    moe_replica_groups,
    w_moe: int,
    T: int,
    H: int,
    eps: float,
    top_k: int,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    moe_block_size: int,
    ep_degree: int,
    moe_stack_layers: int,
    after_gather=None,
    shared_tp: dict = None,
    pad_q_pos=None,
    rs_buf=None,
    zinit=None,
):
    """MoE layer: post-attention norm + MX quantize + NOAUX_TC router, routed experts, shared expert.

    The expert weights are views into ``[moe_stack_layers, E, ...]`` stacks: bwmm's same-expert
    weight reuse loads expert ``weight_skip_expert`` to OOB-skip the reload, and the bounds check is
    against the WHOLE stacked tensor, so that index must lie past the whole stack.
    ``pad_q_pos`` (GLM_GK_MOE_PAD_MASK): ``[T]`` query positions; padding rows get no routed expert.
    ``rs_buf`` (GLM_GK_MOE_ZINIT_AT): the stack-lifetime ``[T_moe, H]`` routed output, zeroed by the caller (or by
    ``zinit(fence)`` here, the ``moe`` site, after the router output), so the routed bwmm does not zero it.
    """
    T_moe = w_moe * T
    E_local = gate_up_w.shape[0]
    # GLM_GK_MOE_AFF8: the compact top-8 tail.
    aff8 = _moe_aff8()
    with sbm_name_scope("gk_postnorm_"):
        moe_outs = rmsnorm_mx_prefill(
            hidden_states=attn_o,
            gamma=post_norm_w,
            router_weights=router_w,
            router_bias=router_bias,
            eps=eps,
            top_k=top_k,
            router_act_fn=RouterActFnType.NOAUX_TC,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
            pack_affinities=True,
            residual=r_in.reshape((1, T, H)),
            # The packed rows land in private HBM, the all-gather's source as is.
            packed_buffer=nl.private_hbm,
            **({"pack_topk8": True} if aff8 else {}),
        )
    packed = moe_outs[0]  # [T, R4] fp32 view of [hidden fp8 | MX scale | fp32 affinities | pad]
    r_out = moe_outs[1]  # [T, H]
    r4 = packed.shape[1]
    scale_region = div_ceil(H // _MX_TILE, _SCALE_TILES_PER_BLOCK) * P_MAX
    aff_col = (H + scale_region) // 4
    if pad_q_pos is not None:
        n_exp = E_local * ep_degree
        if aff8:
            # GLM_GK_MOE_AFF8: the 8 compact affinities (the ids stay; a zero affinity routes nowhere).
            kernel_assert(aff_col + 2 * _AFF8_K <= r4, f"{_ERR} the compact tail runs past the packed row ({r4} words)")
            _mask_pad_affinities(packed, pad_q_pos, aff_col + _AFF8_K, _AFF8_K, T)
        else:
            kernel_assert(aff_col + n_exp <= r4, f"{_ERR} the {n_exp} affinities run past the packed row ({r4} words)")
            _mask_pad_affinities(packed, pad_q_pos, aff_col, n_exp, T)

    mx_w = H + scale_region
    sh_rows = T
    shared_in = None
    if shared_tp is not None:
        # Tensor-parallel shared expert: this rank holds 1/m of its intermediate dim and runs it for the
        # m*T tokens of its block of m consecutive ranks; the MoE reduce-scatter sums the m partials.
        # The block's rows come from a small all-gather issued ahead of the MoE one.
        m = shared_tp["m"]
        sh_rows = m * T
        own_in = nl.ndarray((T, mx_w), dtype=nl.float8_e4m3fn, buffer=nl.private_hbm, name=scoped_name("gk_shtp_own"))
        _hbm_copy_rows(own_in, packed.view(nl.float8_e4m3fn), T, mx_w)
        sh_src = own_in.view(_RAW_DTYPE)
        sh_all = nl.ndarray(
            (sh_rows, mx_w // 2), dtype=_RAW_DTYPE, buffer=nl.private_hbm, name=scoped_name("gk_shtp_all")
        )
        _all_gather(sh_all, sh_src, shared_tp["groups"], m)
        shared_in = sh_all.view(nl.float8_e4m3fn)

    # Gather the packed rows over the MoE group. Issued BEFORE the shared expert so the all-gather
    # is not queued behind the shared expert's weight DMAs (in-order queue).
    moe_all = (
        nl.ndarray((T_moe, r4), dtype=packed.dtype, buffer=nl.private_hbm, name=scoped_name("gk_moe_all"))
        if tc_moe_ag() != "split"
        else None
    )
    sync = _cc_sync(packed[T - 1 : T, 0:8], moe_replica_groups, w_moe, "m")
    map_src, map_col = moe_all, aff_col
    if tc_moe_ag():
        # The affinity columns of the (pad-masked) packed rows, gathered ahead of the rows themselves.
        # (GLM_GK_MOE_AFF8: the 16 compact columns.)
        n_aff = 2 * _AFF8_K if aff8 else E_local * ep_degree
        kernel_assert(aff_col + n_aff <= r4, f"{_ERR} GLM_GK_TC_MOE_AG: the {n_aff} affinities run past the packed row")
        aff_send = nl.ndarray((T, n_aff), dtype=nl.float32, buffer=nl.private_hbm, name=scoped_name("gk_tcaff_send"))
        nisa.dma_copy(dst=aff_send[0:T, 0:n_aff], src=packed.view(nl.float32)[0:T, aff_col : aff_col + n_aff])
        aff_all = nl.ndarray((T_moe, n_aff), dtype=nl.float32, buffer=nl.private_hbm, name=scoped_name("gk_tcaff_all"))
        ag_aff = _all_gather(aff_all, aff_send, moe_replica_groups, w_moe)
        ag_aff.depends_on(sync)
        map_src, map_col = aff_all, 0
    routed_rows, routed_kw = moe_all, {}
    if tc_moe_ag() == "split":
        # Only [hidden | scale] of each row (the affinity tail is in aff_all), staged contiguous: a collective source is
        # one dense buffer.
        kernel_assert(
            mx_w % 4 == 0 and aff_col == mx_w // 4,
            f"{_ERR} GLM_GK_TC_MOE_AG=split: the affinities must follow [hidden | scale]",
        )
        hs_send = nl.ndarray(
            (T, mx_w // 4), dtype=packed.dtype, buffer=nl.private_hbm, name=scoped_name("gk_tchs_send")
        )
        nisa.dma_copy(dst=hs_send[0:T, 0 : mx_w // 4], src=packed[0:T, 0 : mx_w // 4])
        moe_hs = nl.ndarray(
            (T_moe, mx_w // 4), dtype=packed.dtype, buffer=nl.private_hbm, name=scoped_name("gk_tchs_all")
        )
        ag = _all_gather(moe_hs, hs_send, moe_replica_groups, w_moe)
        routed_rows, routed_kw = moe_hs, dict(affinity_rows=aff_all)
    else:
        ag = _all_gather(moe_all, packed, moe_replica_groups, w_moe)
    ag.depends_on(sync)
    if moe_zinit_gate():
        _moe_utils.ZINIT_AFTER = sync
    if zinit is not None:
        # GLM_GK_MOE_ZINIT_AT=moe: the zero-init runs under the token all-gather.
        zinit(packed)
    if after_gather is not None:
        # The stack issues the next layer's o_proj gather here, under the expert compute.
        after_gather()

    expert_kw = dict(
        block_size=moe_block_size,
        activation_function=ActFnType.SiLU,
        compute_dtype=nl.bfloat16,
        # The kernel only supports the accumulating output; the output is zero-initialised.
        is_tensor_update_accumulating=True,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
        quantization_type=QuantizationType.MX,
        skip_dma=SkipMode(True, True),
        use_packed_scales=True,
    )

    # Shared expert: a one-expert blockwise MoE over this rank's own rows, affinity 1. Depends only
    # on this rank's packed rows, so it runs under the gather + mapping + routed stages.
    def _shared_expert():
        if shared_tp is None:
            shared_in_ = nl.ndarray(
                (T, mx_w), dtype=nl.float8_e4m3fn, buffer=nl.private_hbm, name=scoped_name("gk_shared_in")
            )
            _hbm_copy_rows(shared_in_, packed.view(nl.float8_e4m3fn), T, mx_w)
        else:
            shared_in_ = shared_in
        sh_tpid, sh_bte, sh_aff, sh_blocks = _shared_expert_mapping_cached(sh_rows, moe_block_size)
        with sbm_name_scope("gk_shared_"):
            return block_moe_mx(
                hidden_states=shared_in_,
                expert_affinities_masked=sh_aff,
                gate_up_proj_weight=shared_gate_up_w,
                down_proj_weight=shared_down_w,
                token_position_to_id=sh_tpid,
                block_to_expert=sh_bte,
                gate_up_proj_scale=shared_gate_up_scale,
                down_proj_scale=shared_down_scale,
                n_static_blocks=sh_blocks,
                top_k=1,
                weight_skip_expert=moe_stack_layers * shared_gate_up_w.shape[0],
                # Its routing table is tiny: a plain copy is not held behind the in-flight MoE
                # all-gather (a dma_transpose is), so the shared expert runs under the gather.
                routing_plain_dma=True,
                **expert_kw,
            )

    # GLM_GK_SHARED_UNDER_RS=late: the replicated shared expert after the MoE reduce-scatter is issued.
    sh_rs = _shared_under_rs()
    if sh_rs:
        kernel_assert(
            shared_tp is None,
            f"{_ERR} GLM_GK_SHARED_UNDER_RS=late needs the replicated shared expert (GLM_GK_SHARED_TP=1)",
        )
    if sh_rs != "late":
        shared_out = _shared_expert()

    if aff8:
        # GLM_GK_MOE_AFF8: this rank's local experts' dense columns, rebuilt from the gathered compact tails (of the
        # fused rows, or of the GLM_GK_TC_MOE_AG affinity all-gather).
        with sbm_name_scope("gk_aff8_"):
            map_src, map_col = _aff8_local(map_src, ep_rank, map_col, T_moe, E_local), 0
        routed_kw = dict(routed_kw, compact_affinities=True)
    with sbm_name_scope("gk_map_"):
        (token_rows, block_rows), block_to_expert, conditions = _build_blockwise_mapping(
            map_src, ep_rank, map_col, T_moe, E_local, top_k, moe_block_size, **({"ep_cols": False} if aff8 else {}),
        )
    # Routed experts, accumulated straight into the private_hbm reduce-scatter source.
    init_kw = {}
    if rs_buf is not None:
        kernel_assert(tuple(rs_buf.shape) == (T_moe, H), f"{_ERR} the routed output must be [{T_moe}, {H}]")
        rs_src = rs_buf
        # GLM_GK_MOE_ZINIT_AT: the GK zeroed rs_buf, so the bwmm only accumulates.
        init_kw = dict(init_output=False)
    else:
        rs_src = nl.ndarray((T_moe, H), dtype=nl.bfloat16, buffer=nl.private_hbm, name=scoped_name("gk_rs_src"))
    dyn_inner = _moe_dyn_inner()
    # GLM_GK_MOE_WRELAY: the host stored the routed gate/up weights tile-major, so every routed call reads them so.
    wstream = _moe_wstream()
    if wstream is not None:
        routed_kw = dict(routed_kw, weight_stream=wstream)
    # GLM_GK_MOE_PF_TILES: the routed call's cross-block gate/up prefetch depth.
    pf_tiles = _moe_pf_tiles(T)
    if pf_tiles > 1:
        routed_kw = dict(routed_kw, prefetch_tiles=pf_tiles)
    with sbm_name_scope("gk_routed_"), _bwmm_dyn_inner(dyn_inner):
        routed_all = block_moe_mx(
            hidden_states=routed_rows,
            expert_affinities_masked=None,
            gate_up_proj_weight=gate_up_w,
            down_proj_weight=down_w,
            # The routing comes as the mapping's nonzero rows and the row block of each block.
            token_position_to_id=None,
            block_to_expert=block_to_expert,
            ep_rank=ep_rank,
            conditions=conditions,
            gate_up_proj_scale=gate_up_scale,
            down_proj_scale=down_scale,
            n_static_blocks=div_ceil(div_ceil(T_moe * top_k, ep_degree), moe_block_size),
            top_k=top_k,
            ep_degree=ep_degree,
            output_tensor=rs_src,
            weight_skip_expert=moe_stack_layers * E_local,
            # A strided copy of the routing table, not a 4-byte int32 dma_transpose (the same table).
            routing_plain_dma=True,
            token_rows=token_rows,
            block_rows=block_rows,
            **init_kw,
            **routed_kw,
            **expert_kw,
        )
    _moe_utils.ZINIT_AFTER = None
    if shared_tp is not None:
        # Add the shared partial into this block's rows of the reduce-scatter source.
        _shared_tp_add(rs_src, shared_out, shared_tp["rank_hbm"], m, sh_rows, H)
        routed_all = rs_src
    routed = nl.ndarray((T, H), dtype=nl.bfloat16, buffer=nl.private_hbm, name=scoped_name("gk_routed"))
    _reduce_scatter(routed, routed_all, moe_replica_groups, w_moe)
    if sh_rs == "late":
        # GLM_GK_SHARED_UNDER_RS=late: under the reduce-scatter (its instructions follow the trigger in program order).
        shared_out = _shared_expert()

    moe_out = nl.ndarray((T, H), dtype=nl.bfloat16, buffer=nl.shared_hbm, name=scoped_name("gk_moe_out"))
    if shared_tp is not None:
        _hbm_copy_rows(moe_out, routed, T, H)
        return moe_out, r_out
    for r0 in range(0, T, P_MAX):
        n = min(P_MAX, T - r0)
        a = nl.ndarray((P_MAX, H), dtype=nl.bfloat16, buffer=nl.sbuf, name=scoped_name(f"gk_sum_a{r0}"))
        b = nl.ndarray((P_MAX, H), dtype=nl.bfloat16, buffer=nl.sbuf, name=scoped_name(f"gk_sum_b{r0}"))
        nisa.dma_copy(dst=a[:n, :], src=routed[r0 : r0 + n, 0:H])
        nisa.dma_copy(dst=b[:n, :], src=shared_out[r0 : r0 + n, 0:H])
        nisa.tensor_tensor(dst=a[:n, :], data1=a[:n, :], data2=b[:n, :], op=nl.add)
        nisa.dma_copy(dst=moe_out[r0 : r0 + n, 0:H], src=a[:n, :])
    return moe_out, r_out


def _layer(
    hidden,
    residual,
    cos,
    sin,
    q_pos,
    slot_mapping,
    cp_rank,
    ep_rank,
    wo_group_rank,
    kv_cache,
    input_norm_w,
    attn_w,
    q_norm_w,
    kv_norm_w,
    wo_shard,
    post_norm_w,
    block_table,
    prior_tokens,
    prev_topk,
    idx_w: dict,
    mlp_w: dict,
    moe_w: dict,
    after_kv_gather,
    after_mlp_gather,
    n_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
    q_lora_rank: int,
    index_topk: int,
    rms_norm_eps: float,
    mla_norm_eps: float,
    softmax_scale: float,
    cp_degree: int,
    cp_interleave: int,
    cp_replica_groups,
    wo_replica_groups,
    moe_replica_groups,
    kv_ctx_local: int,
    kv_block_size: int,
    moe_static: dict,
    kv_full,
    hybrid=None,
    key_rows_out=None,
    kv_slot_off=0,
    moe_rs_buf=None,
    zinit_kv=None,
    zinit_moe=None,
    zinit_oproj=None,
):
    """One decoder layer. Returns ``(out, r_out, kv_cache[, topk_out, idx_k_cache, raw_picks])``.

    ``kv_full`` is the destination of the layer's CP KV gather (``[cp * s_kv_local, kv_row]``, or one spare row more,
    which the layer does not read).

    ``key_rows_out``: a full layer's ``[T, d]`` copy of the key rows it writes to its cache.
    ``moe_rs_buf``, ``zinit_kv``, ``zinit_moe``: (GLM_GK_MOE_ZINIT_AT) the stack-lifetime routed output of a MoE layer
    and the zero-init of the ``kv`` site (called with its fence once the KV gather is issued) or the ``moe`` site.
    ``zinit_oproj``: (GLM_GK_MOE_ZINIT_AT=oproj) the zero-init, called with the attention output as its fence.
    """
    H = hidden.shape[1]
    T = slot_mapping.shape[0]
    R = qk_rope_head_dim
    L = kv_lora_rank
    wqkv_a, wq_b, wuk, wuv = attn_w
    has_prefix = block_table is not None
    s_kv_local = kv_ctx_local if has_prefix else T
    # Dense attention attends every key, so it needs no top-k: the sparse path runs only when the
    # global sequence exceeds the top-k budget, same as the model.
    dense = cp_degree * s_kv_local <= index_topk
    full_layer = not dense and prev_topk is None
    w_wo = _group_size(wo_replica_groups, cp_degree)
    kv_row = kv_cache.shape[-1]
    if kv_full.shape[0] == cp_degree * s_kv_local + 1:
        # GLM_GK_KV_PLANS: the stack's spare row past the gathered rows, which the layer does not read.
        kv_full = kv_full[0 : cp_degree * s_kv_local]

    # ---- input norm: r_in = hidden + residual, normed = rmsnorm(r_in) ----
    # Hybrid: the norm writes straight into the x all-gather's private source buffer.
    x_send = None
    if hybrid is not None:
        x_send = nl.ndarray((T, H), dtype=nl.bfloat16, buffer=nl.private_hbm, name=scoped_name("gk_hy_x_send"))
    if _up_flag("NORMRES") and hybrid is not None:
        # GLM_GK_UP_NORMRES=1: the hybrid path consumes only the bf16 norm (and the pre-norm sum), so a plain
        # residual RMSNorm (residual add in the DMA, no MX row). A different kernel: numerics may differ.
        with sbm_name_scope("gk_innorm_"):
            norm_outs = rmsnorm_residual_prefill(
                hidden=hidden.reshape((T, H)),
                gamma=input_norm_w.reshape((1, H)),
                residual=None if residual is None else residual.reshape((T, H)),
                eps=rms_norm_eps,
                out_tensor=x_send,
                # GLM_GK_PRJ_NORMST=1: the bf16 output stored per H tile on HW DGE.
                **({"store_per_tile": True} if _prj_normst() else {}),
            )
        # Outputs: the bf16 norm, [the pre-norm sum when a residual came in].
        r_in = hidden if residual is None else norm_outs[1]  # [T, H]
        normed = norm_outs[0]  # [T, H] bf16 natural H
    else:
        with sbm_name_scope("gk_innorm_"):
            norm_outs = rmsnorm_mx_prefill(
                hidden_states=hidden.reshape((1, T, H)),
                gamma=input_norm_w,
                eps=rms_norm_eps,
                residual=None if residual is None else residual.reshape((1, T, H)),
                emit_norm_bf16=True,
                norm_bf16_out=x_send,
            )
        # Outputs: packed MX row, [the pre-norm sum when a residual came in], the bf16 norm.
        r_in = hidden if residual is None else norm_outs[1]  # [T, H]
        normed = norm_outs[-1]  # [T, H] bf16 natural H

    kernel_assert(key_rows_out is None or full_layer, f"{_ERR} key_rows_out needs a sparse full layer")
    hy_pre = None
    if full_layer and hybrid is not None:
        # The head-TP QKV's gathers go first: behind the indexer key all-gather they would hold the QKV.
        hy_pre = _hybrid_qkv_gathers(
            normed, attn_w[0], wo_replica_groups, w_wo, T, H, x_send, s1_dmat=_up_flag("S1DMAT")
        )
    if full_layer:
        # The indexer keys and their all-gather run before QKV, so QKV runs under the all-gather.
        with sbm_name_scope("gk_idxk_"):
            idx_keys = glm_indexer_keys(
                normed,
                idx_w["wq_b"],
                idx_w["wk"],
                idx_w["k_norm_w"],
                idx_w["k_norm_b"],
                cos,
                sin,
                slot_mapping,
                idx_w["weights_proj"],
                q_pos,
                end_pos=cp_degree * s_kv_local,
                cp_degree=cp_degree,
                replica_groups=cp_replica_groups,
                weights_proj_transposed=True,
                k_cache=idx_w["k_cache"],
                block_table=block_table,
                block_size=kv_block_size if has_prefix else 0,
                cp_interleave=cp_interleave,
                index_topk=index_topk,
                pack_heads=True,
                flat_loops=True,
                score_pingpong=True,
                # Column-tiled head-sums issued in bursts of 16 (4-way concurrent on PE).
                col_headsum=True,
                # The column-tiled head-sums on the rolled chunk path too (> 4 score chunks: 256k+ NEFFs).
                col_rolled=True,
            )
        if key_rows_out is not None:
            # k_local holds the rows that store_k_paged wrote at the slots, in token order.
            k_rows = idx_keys.k_local
            w = k_rows.shape[1]
            nisa.dma_copy(dst=key_rows_out[0:T, 0:w], src=k_rows[0:T, 0:w])

    # ---- absorbed QKV + paged KV write ----
    # GLM_GK_SYNC_HOL: this full layer's QKV group-load handles (trace-time bookkeeping), for the score's
    # qr transposes below. Head-TP full layers only.
    hol = _sync_hol() if (full_layer and hybrid is not None) else 0
    hol_loads = [] if hol else None
    if hybrid is not None:
        # Head TP over the wo group: QKV for the group's tokens and this rank's heads, then an
        # all-to-all back to this rank's tokens and every head.
        _QKV_WLOADS_REC[0] = hol_loads
        q_lift, q_pe, qr_g, kv_cache = _hybrid_qkv(
            normed,
            attn_w,
            q_norm_w,
            kv_norm_w,
            hybrid,
            kv_cache,
            (n_heads, qk_nope_head_dim, R, L, q_lora_rank, mla_norm_eps),
            wo_replica_groups,
            w_wo,
            T,
            H,
            x_send=x_send,
            pre=hy_pre,
            kv_slot_off=kv_slot_off,
            s1_dmat=_up_flag("S1DMAT"),
        )
        _QKV_WLOADS_REC[0] = None
        qr = _rows_of_position(qr_g.reshape((w_wo * T, q_lora_rank)), hybrid["rank_hbm"], T, q_lora_rank, "gk_hy_qr")
    else:
        cos_full, sin_full = _build_rope_tables(cos, sin, T, R)
        with sbm_name_scope("gk_qkv_"):
            q_lift, q_pe, qr, kv_cache = mla_qkv_cte_kernel(
                x_hbm_mx=normed.reshape((1, T, H)),
                wqkv_a_hbm=wqkv_a,
                wqkv_a_scale_hbm=None,
                wq_b_hbm=wq_b,
                wq_b_scale_hbm=None,
                q_norm_gamma_hbm=q_norm_w,
                kv_norm_gamma_hbm=kv_norm_w,
                wuk_hbm=wuk,
                cos_cache_hbm=cos_full,
                sin_cache_hbm=sin_full,
                n_heads=n_heads,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=R,
                kv_lora_rank=L,
                qk_lora_rank=q_lora_rank,
                norm_eps=mla_norm_eps,
                precision=MlaPrecision.BF16,
                kv_cache_hbm=kv_cache,
                slot_mapping_hbm=slot_mapping.reshape((1, T)),
                kv_slot_offset=kv_slot_off,
            )

    # ---- DSA top-k: run the indexer (full layer) or reuse the previous one (shared layer). Traced
    # BEFORE the KV gather: only the attention needs kv_full, and the KV all-gather's peer-skew wait
    # otherwise blocks the QKV/indexer loads queued behind it on the in-order Sync queue. ----
    def _kv_gather():
        # CP gather of the packed KV rows: this call's rows, or the whole cached context.
        if has_prefix:
            kv_local = _gather_local_kv_pages(
                kv_cache, block_table, kv_ctx_local, kv_block_size, kv_row, page_off=kv_slot_off // kv_block_size
            )
        else:
            kv_local = _gather_local_kv_rows(kv_cache, slot_mapping, T, kv_row, slot_off=kv_slot_off)
        _all_gather(kv_full, kv_local, cp_replica_groups, cp_degree)
        return kv_local

    topk = prev_topk
    if full_layer:
        # The indexer issues the KV and next-layer weight all-gathers inside its score.
        if hol_loads:
            # GLM_GK_SYNC_HOL: the score's qr transposes are scheduled after the QKV group loads (and, at 2, its wq
            # loads).
            _QR_T_AFTER[0], _QR_T_AFTER[1] = list(hol_loads), hol == 2
        if has_prefix:
            kv_local = _gather_local_kv_pages(
                kv_cache, block_table, kv_ctx_local, kv_block_size, kv_row, on_gpsimd=True,
                page_off=kv_slot_off // kv_block_size,
            )
        else:
            kv_local = _gather_local_kv_rows(kv_cache, slot_mapping, T, kv_row, on_gpsimd=True, slot_off=kv_slot_off)

        def _after_k_gather():
            _all_gather(kv_full, kv_local, cp_replica_groups, cp_degree)
            if after_kv_gather is not None:
                after_kv_gather()

        if T in (128, 256, 512):
            # T / 64 indexer calls of 64 rows over the same keys, so the score keeps the packed head-sum.
            hr = 64
            topk_out = nl.ndarray(
                (1, T, index_topk), dtype=nl.int32, buffer=nl.shared_hbm, name=scoped_name("gk_topk_out")
            )

            def _half(src2d, r0, tag):
                cols = src2d.shape[1]
                dst = nl.ndarray(
                    (hr, cols), dtype=src2d.dtype, buffer=nl.private_hbm, name=scoped_name(f"gk_idxh_{tag}")
                )
                nisa.dma_copy(dst=dst[0:hr, 0:cols], src=src2d[r0 : r0 + hr, 0:cols])
                return dst

            n_h = idx_keys.weights.shape[1]
            for hh in range(T // hr):
                r0 = hh * hr
                # One 64-row query tile per call: the 128 / 256 / 512-row keys' tile range would run past it.
                kernel_assert(idx_keys.num_cores == 1, f"{_ERR} the 64-row indexer calls need one core (LNC1)")
                keys_h = dataclasses.replace(
                    idx_keys,
                    cfg=dataclasses.replace(idx_keys.cfg, s_local=hr),
                    tile_lo=0,
                    tile_hi=1,
                    weights=_half(idx_keys.weights.reshape((T, n_h)), r0, f"w{hh}"),
                    qpos_local=_half(idx_keys.qpos_local.reshape((T, 1)), r0, f"ql{hh}"),
                )
                qr_h = _half(qr.reshape((T, q_lora_rank)), r0, f"qr{hh}")
                cos_h = _half(cos.reshape((T, cos.shape[-1])), r0, f"cos{hh}")
                sin_h = _half(sin.reshape((T, sin.shape[-1])), r0, f"sin{hh}")
                qpos_h = _half(q_pos.reshape((T, 1)), r0, f"qp{hh}").reshape((hr,))
                with sbm_name_scope(f"gk_idx_h{hh}_"):
                    outs_h = glm_indexer_score(
                        qr_h, idx_w["wq_b"], cos_h, sin_h, qpos_h, keys_h,
                        after_k_gather=_after_k_gather if hh == 0 else None,
                    )
                nisa.dma_copy(
                    dst=topk_out.reshape((T, index_topk))[r0 : r0 + hr, 0:index_topk],
                    src=outs_h[0].reshape((hr, index_topk))[0:hr, 0:index_topk],
                )
            # The layer attends its picks.
            topk = topk_out
        else:
            with sbm_name_scope("gk_idx_"):
                idx_outs = glm_indexer_score(
                    qr.reshape((T, q_lora_rank)),
                    idx_w["wq_b"],
                    cos,
                    sin,
                    q_pos,
                    idx_keys,
                    after_k_gather=_after_k_gather,
                )
            # The kernel frames each query's K positions as [16, K/16]; the flat [1, T, K] view is the
            # topk_tiled=False layout the attention reads.
            topk = idx_outs[0].reshape((1, T, index_topk))
            # Return a fresh tensor, not the reshape view: a view as an output reports its base shape
            # ([T, 16, K/16]) to the caller and shifts every later output.
            topk_out = nl.ndarray(
                (1, T, index_topk), dtype=nl.int32, buffer=nl.shared_hbm, name=scoped_name("gk_topk_out")
            )
            _hbm_copy_rows(topk_out.reshape((T, index_topk)), topk.reshape((T, index_topk)), T, index_topk)

    _QR_T_AFTER[0], _QR_T_AFTER[1] = None, False  # GLM_GK_SYNC_HOL: off outside this full layer's score
    # The raw rank-major picks (a PP stage hands them to the next one).
    raw_picks = topk_out if full_layer else None

    # ---- CP gather of the packed KV rows (a full layer's ran inside its indexer score) ----
    kv_local_l = None  # GLM_GK_MOE_ZINIT_AT=kv: the fence of the zero-init
    if not full_layer:
        kv_local_l = _kv_gather()
        if after_kv_gather is not None:
            # The stack issues the NEXT layer's weight gather here: collectives run one at a time, so it
            # drains under this layer's attention compute instead of heading the next layer.
            after_kv_gather()
    if zinit_kv is not None:
        # GLM_GK_MOE_ZINIT_AT=kv: after the local page gather that feeds the KV all-gather (and after the next layer's
        # weight gather is issued), so neither queues behind it. q_lift when the KV gather ran inside an indexer.
        zinit_kv(kv_local_l if kv_local_l is not None else q_lift)

    # ---- latent attention over the gathered packed cache ----
    with sbm_name_scope("gk_attn_"):
        out_attn = mla_sparse_attention_cte_kernel(
            q_lift,
            q_pe,
            kv_full,
            None,
            softmax_scale,
            topk_indices_hbm=topk,
            topk_tiled=False,
            dense=dense,
            prior_tokens_hbm=prior_tokens,
            cp_rank_hbm=cp_rank,
            cp_world_size=cp_degree,
            cp_kv_cache_interleave_size=cp_interleave,
            precision=MlaPrecision.BF16,
            kv_cache_packed=True,
        )

    _anchor_sends(attn_w, out_attn.reshape((T, n_heads * L)))
    if zinit_oproj is not None:
        # GLM_GK_MOE_ZINIT_AT=oproj: the routed output's zero-init after the attention wrote its output, under the
        # o_proj.
        zinit_oproj(out_attn.reshape((T, n_heads * L)))
    if hybrid is not None:
        attn_o = _hybrid_oproj(out_attn, wuv, wo_shard, wo_group_rank, wo_replica_groups, w_wo, n_heads, T, H, L)
    else:
        attn_o = _row_parallel_oproj(out_attn, wuv, wo_shard, wo_group_rank, wo_replica_groups, w_wo, n_heads, T, H, L)
    if _KVL_LIVE:
        _anchor_after(attn_o.reshape((T, H)), list(_KVL_LIVE), "kvl")
        _KVL_LIVE.clear()

    if mlp_w is not None:
        out, r_out = _tp_dense_mlp(
            attn_o,
            r_in,
            post_norm_w,
            mlp_w["gate"],
            mlp_w["up"],
            mlp_w["down"],
            wo_replica_groups,
            w_wo,
            T,
            H,
            rms_norm_eps,
            after_gather=after_mlp_gather,
        )
    else:
        out, r_out = _moe(
            attn_o,
            r_in,
            ep_rank,
            post_norm_w,
            moe_w["router_w"],
            moe_w["router_bias"],
            moe_w["gate_up_w"],
            moe_w["gate_up_scale"],
            moe_w["down_w"],
            moe_w["down_scale"],
            moe_w["shared_gate_up_w"],
            moe_w["shared_gate_up_scale"],
            moe_w["shared_down_w"],
            moe_w["shared_down_scale"],
            moe_replica_groups,
            _group_size(moe_replica_groups, cp_degree),
            T,
            H,
            rms_norm_eps,
            after_gather=after_mlp_gather,
            shared_tp=moe_w.get("shared_tp"),
            pad_q_pos=q_pos if _moe_pad_mask() else None,
            **({"rs_buf": moe_rs_buf, "zinit": zinit_moe} if moe_rs_buf is not None else {}),
            **moe_static,
        )
    outs = [out, r_out, kv_cache]
    if full_layer:
        outs += [topk_out, idx_w["k_cache"], raw_picks]
    return tuple(outs)


def _layer_prefetch_hooks(li: int, n_layers: int, prefetch_attn, prefetch_wo) -> tuple:
    """``(after_kv_gather, after_mlp_gather)`` of layer ``li``: the hooks that issue layer ``li + 1``'s attention-weight
    gather after this layer's KV gather and its wo gather after this layer's MLP / MoE all-gather."""
    nxt = li + 1
    if nxt >= n_layers:
        return None, None
    return (lambda: prefetch_attn(nxt)), (lambda: prefetch_wo(nxt))


# NOTE: no @nki.jit here; the caller wraps with nki.jit() at the call site so the sub-kernels
# (themselves @nki.jit) inline into one trace instead of double-jitting.
def glm52_prefill_giga_kernel(
    # ---- activations / per-step control ----
    hidden: nl.NkiTensor,
    cos: nl.NkiTensor,
    sin: nl.NkiTensor,
    q_pos: nl.NkiTensor,
    slot_mapping: nl.NkiTensor,
    cp_rank: nl.NkiTensor,
    ep_rank: nl.NkiTensor,
    wo_group_rank: nl.NkiTensor,
    kv_caches: nl.NkiTensor,
    idx_k_caches: nl.NkiTensor,
    # ---- attention weights [L, ...] ----
    input_norm_w: nl.NkiTensor,
    wqkv_a: nl.NkiTensor,
    wq_b: nl.NkiTensor,
    q_norm_w: nl.NkiTensor,
    kv_norm_w: nl.NkiTensor,
    wuk: nl.NkiTensor,
    wuv: nl.NkiTensor,
    wo_shard: nl.NkiTensor,
    post_norm_w: nl.NkiTensor,
    final_norm_w: nl.NkiTensor,
    # ---- indexer weights [L_full, ...]; None for a launch of shared layers only (topk_in) ----
    idx_wq_b: nl.NkiTensor = None,
    idx_wk: nl.NkiTensor = None,
    idx_k_norm_w: nl.NkiTensor = None,
    idx_k_norm_b: nl.NkiTensor = None,
    idx_weights_proj: nl.NkiTensor = None,
    # ---- cached context (segmented prefill); None for a fresh prompt ----
    block_table: nl.NkiTensor = None,
    prior_tokens: nl.NkiTensor = None,
    # ---- the residual stream entering the first layer of a group; None for layer 0 ----
    residual: nl.NkiTensor = None,
    # ---- MoE weights [L_moe, ...] ----
    router_w: nl.NkiTensor = None,
    router_bias: nl.NkiTensor = None,
    gate_up_w: nl.NkiTensor = None,
    gate_up_scale: nl.NkiTensor = None,
    down_w: nl.NkiTensor = None,
    down_scale: nl.NkiTensor = None,
    shared_gate_up_w: nl.NkiTensor = None,
    shared_gate_up_scale: nl.NkiTensor = None,
    shared_down_w: nl.NkiTensor = None,
    shared_down_scale: nl.NkiTensor = None,
    # ---- dense-MLP weights [L_dense, ...] ----
    mlp_gate_w: nl.NkiTensor = None,
    mlp_up_w: nl.NkiTensor = None,
    mlp_down_w: nl.NkiTensor = None,
    # ---- GLM_GK_PRJ_OMX: [L, n_heads * v_head / (32 w), H] uint8 e8m0 scales; wo_shard is then the MXFP8
    # fp8x4 rows [L, n_heads * v_head / (4 w), H] int32 ----
    wo_scale: nl.NkiTensor = None,
    # ---- static config ----
    layer_mlp_types: tuple = (),
    layer_indexer_types: tuple = (),
    n_heads: int = 64,
    qk_nope_head_dim: int = 192,
    qk_rope_head_dim: int = 64,
    kv_lora_rank: int = 512,
    q_lora_rank: int = 2048,
    index_topk: int = 2048,
    rms_norm_eps: float = 1e-5,
    mla_norm_eps: float = 1e-6,
    softmax_scale: float = 0.0625,
    top_k: int = 8,
    n_group: int = 1,
    topk_group: int = 1,
    routed_scaling_factor: float = 2.5,
    moe_block_size: int = 256,
    ep_degree: int = 1,
    cp_degree: int = 1,
    cp_interleave: int = 1,
    cp_replica_groups: tuple = None,
    wo_replica_groups: tuple = None,
    wo_gather_groups: tuple = None,
    attn_tp_gather_groups: tuple = None,
    cos_g: nl.NkiTensor = None,
    sin_g: nl.NkiTensor = None,
    slot_mapping_g: nl.NkiTensor = None,
    shared_tp: int = 1,
    moe_replica_groups: tuple = None,
    kv_ctx_local: int = 0,
    kv_block_size: int = 128,
    emit_key_rows: bool = False,
    topk_in: nl.NkiTensor = None,
    emit_picks: bool = False,
):
    """Every GLM-MoE-DSA decoder layer and the final norm in one trace, LNC1.

    Dimensions:
        T: tokens this rank owns (its CP shard); H: hidden size; R: qk_rope_head_dim;
        L: kv_lora_rank; K: index_topk; cp: cp_degree; w: ranks of one wo group.

    Every per-layer weight and cache carries a leading layer axis, per kind: attention weights,
    norms and ``kv_caches`` ``[L, ...]``; indexer weights and ``idx_k_caches`` ``[L_full, ...]``, one
    per ``"full"`` layer; MoE weights ``[L_moe, ...]``; dense-MLP weights ``[L_dense, ...]``.

    Args:
        hidden: [T, H] bf16 embeddings of this rank's tokens. The stack starts at layer 0, which
            must be a full indexer layer unless ``topk_in`` is given.
        cos, sin: [T, R/2] fp32 RoPE tables at each owned token's global position.
        q_pos: [T] fp32 global position of each owned token (indexer mask).
        slot_mapping: [T] int32 paged-cache slot of each owned token (MLA and indexer caches).
        cp_rank, ep_rank: [1, 1] int32 this rank's CP / EP index.
        wo_group_rank: [1, 1] int32 this rank's position inside its wo replica group.
        kv_caches: [L, n_slots, kv_row] bf16 paged MLA caches, written in place.
        idx_k_caches: [L_full, n_slots, index_head_dim] bf16 paged indexer caches, written in place; None (with the
            ``idx_*`` weights) when the launch has no full layer.
        input_norm_w, post_norm_w, q_norm_w, kv_norm_w: [L, 1, *] bf16 norm gammas.
        wqkv_a, wq_b, wuk, wuv: [L, rows/cp, cols] bf16 rank-major 1/cp chunks of the absorbed QKV
            and V-up weights (one flat row when the rows do not divide); gathered in-trace.
        wo_shard: [L, n_heads * v_head / w, H] bf16 o_proj row shard of this rank's wo position.
        final_norm_w: [1, H] bf16 gamma of the model's final RMSNorm. None for a group of layers
            that is not the model's last: the stack then returns the residual stream instead.
        idx_*: indexer weights (``glm_indexer_keys`` / ``glm_indexer_score``); ``idx_weights_proj`` is stored
            ``[H, heads]``.
        block_table: [n_blocks] int32 this rank's cache pages in logical order, covering
            ``kv_ctx_local`` positions. None for a fresh prompt.
        prior_tokens: [1, 1] int32 this rank's cached-token count before this segment.
        residual: [T, H] bf16 residual stream of a group that starts mid-model: the ``r`` the
            previous group returned, with ``hidden`` its ``h``. None when the stack starts at
            layer 0.
        router_w: [L_moe, H, E] bf16 router weight in rmsnorm_mx_prefill's NOAUX layout.
        router_bias: [L_moe, 1, E] fp32 e_score_correction_bias.
        gate_up_w, gate_up_scale, down_w, down_scale: this rank's ``E_local`` MXFP4 routed experts
            in ``block_moe_mx``'s layout with host-packed scales.
        shared_*: the replicated MXFP4 shared expert in the same layout with ``E = 1``.
        mlp_gate_w, mlp_up_w, mlp_down_w: dense layers' bf16 SwiGLU MLP, TP-sharded over the wo
            group: ``[H, I/w]``, ``[H, I/w]``, ``[I/w, H]``.
        layer_mlp_types: per layer, ``"dense"`` or ``"sparse"``.
        layer_indexer_types: per layer, ``"full"`` or ``"shared"``; a shared layer reuses the
            top-k of the nearest preceding full layer.
        cp_replica_groups: CP rank lists (KV, indexer and attention-weight gathers).
        wo_replica_groups: TP x CP wo groups (o_proj and dense MLP), position order = shard index.
        wo_gather_groups: the groups of ranks that share a wo position, or None. When set,
            ``wo_shard`` is ``[L, n_heads * v_head / (w * n), H]``: this rank's 1/n row chunk of its
            wo shard, in group order. Layer i+1's shard is all-gathered over its group one layer
            ahead, under layer i's MLP / MoE compute (after its MLP or MoE all-gather).
        attn_tp_gather_groups: head TP over the wo groups (the hybrid layout), or None. The ranks of
            one wo position (ranks p, p + w, ...): ``wqkv_a / wq_b / wuk / wuv`` are this rank's 1/n
            chunks of its position's slice (wqkv_a ``[H, (q_lora + L + R) / w]``, the q block then the
            kv block; wq_b / wuk / wuv heads ``[p * n_heads / w, (p + 1) * n_heads / w)``), gathered over
            these groups one layer ahead. The QKV projections and V-up / o_proj run for the wo group's
            ``w * T`` tokens and this rank's heads; the attention stays context parallel.
        cos_g, sin_g, slot_mapping_g: with ``attn_tp_gather_groups``, the RoPE tables and cache slots
            of the wo group's ``w * T`` tokens in rank order; other ranks' slots are -1.
        shared_tp: m > 1 runs the shared expert tensor parallel over blocks of m consecutive ranks:
            the shared tensors hold this rank's 1/m intermediate slice (slice ``rank % m``), each block
            all-gathers its m*T rows ahead of the MoE gather, and the MoE reduce-scatter sums the m
            partials. Needs one MoE group in CP rank order.
        moe_replica_groups: MoE all-gather / reduce-scatter rank lists.
        kv_ctx_local: static per-rank context bucket with a cached prefix (pages of
            ``kv_block_size``). 0 = no prefix.
        emit_key_rows: also return ``key_rows`` ``[F, T, d]`` bf16, the key rows each of the F full layers wrote at
            ``slot_mapping``, in token order. A caller that copies the new rows elsewhere then reads them without a
            gather over the returned key stack.
        topk_in: [T, index_topk] int32 raw rank-major picks of the full layer before this launch (a PP stage that
            starts on a shared layer). The leading shared layers attend them as that full layer's shared layers do.
        emit_picks: also return ``picks`` [1, T, index_topk] int32, the raw picks of the launch's last full layer
            (``topk_in`` when it has none), right after ``h, r``. Not with ``final_norm_w``.

    Returns:
        ``(final_normed [T, H] bf16, kv_caches, idx_k_caches)``: the model's final RMSNorm of the
        last layer's output, and the aliased caches. Without ``final_norm_w``,
        ``(h [T, H], r [T, H], kv_caches, idx_k_caches)``: the last layer's output and residual,
        which the next group takes as ``hidden`` and ``residual``. With ``emit_key_rows``, ``key_rows``
        follows. ``emit_picks`` puts ``picks`` after ``r``; a launch without a full layer omits ``idx_k_caches``.
    """
    n_layers = len(layer_mlp_types)
    # T: rows of the per-rank token view (slot_mapping) and of the residual stream.
    T, H = slot_mapping.shape[0], hidden.shape[1]
    _SHMAP_CACHE.clear()
    _PRJ_TRACE_T[0] = T  # GLM_GK_PRJ_QKV_T
    _, n_prgs, _ = get_verified_program_sharding_info("glm52_prefill_giga_kernel", (0, 1), 1)
    kernel_assert(n_prgs == 1, f"{_ERR} is an LNC1 kernel, got {n_prgs} programs")
    kernel_assert(
        n_layers >= 1
        and len(layer_indexer_types) == n_layers
        and (layer_indexer_types[0] == "full" or topk_in is not None),
        f"{_ERR} layer_mlp_types and layer_indexer_types must list the same {n_layers} >= 1 layers, "
        "starting on a full indexer layer (or a shared one with topk_in)",
    )
    n_full = sum(kind == "full" for kind in layer_indexer_types)
    kernel_assert(
        (idx_k_caches is None) == (n_full == 0) and not (emit_picks and final_norm_w is not None),
        f"{_ERR} idx_k_caches comes with a full layer only, and emit_picks with no final norm",
    )
    w_wo = _group_size(wo_replica_groups, cp_degree)
    kernel_assert(cp_degree > 1 and w_wo > 1, f"{_ERR} needs cp_degree > 1 and wo groups > 1")
    kernel_assert(
        block_table is None or (prior_tokens is not None and kv_ctx_local >= T),
        f"{_ERR} a cached prefix needs prior_tokens and kv_ctx_local >= T ({T})",
    )
    layer_config = dict(
        n_heads=n_heads,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        kv_lora_rank=kv_lora_rank,
        q_lora_rank=q_lora_rank,
        index_topk=index_topk,
        rms_norm_eps=rms_norm_eps,
        mla_norm_eps=mla_norm_eps,
        softmax_scale=softmax_scale,
        cp_degree=cp_degree,
        cp_interleave=cp_interleave,
        cp_replica_groups=cp_replica_groups,
        wo_replica_groups=wo_replica_groups,
        moe_replica_groups=moe_replica_groups,
        kv_ctx_local=kv_ctx_local,
        kv_block_size=kv_block_size,
        moe_static=dict(
            top_k=top_k,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
            moe_block_size=moe_block_size,
            ep_degree=ep_degree,
            moe_stack_layers=gate_up_w.shape[0] if gate_up_w is not None else 1,
        ),
    )

    # Attention weights one layer ahead: layer 0's gather is issued up front, layer i+1's right
    # after layer i's KV all-gather at the lowest DMA priority, so it drains under layer i.
    qk_head = qk_nope_head_dim + qk_rope_head_dim
    attn_dims = (H, q_lora_rank, kv_lora_rank, qk_rope_head_dim, n_heads, qk_head, qk_nope_head_dim)
    attn_pre = [None] * n_layers

    hybrid = None
    attn_res = (False,) * 4
    if attn_tp_gather_groups is not None:
        hpr = n_heads // w_wo
        hy_shapes = (
            (H, (q_lora_rank + kv_lora_rank + qk_rope_head_dim) // w_wo),
            (q_lora_rank, hpr * qk_head),
            (qk_nope_head_dim, hpr * kv_lora_rank),
            (hpr * kv_lora_rank, wuv.shape[-1]),
        )
        # GLM_GK_ATTN_RESIDENT: the model stores a weight as its rank's WHOLE wo position slice
        # ([L, *slice], the gathered shape) instead of a 1/n chunk; such a weight is read in place, not gathered.
        attn_res = tuple(
            tuple(w.shape[1:]) == tuple(shp) for w, shp in zip((wqkv_a, wq_b, wuk, wuv), hy_shapes)
        )
        hybrid = dict(cos_g=cos_g, sin_g=sin_g, slot_mapping_g=slot_mapping_g, rank_hbm=wo_group_rank)

    def _same_groups(x, y):
        return x is not None and y is not None and tuple(tuple(g) for g in x) == tuple(tuple(g) for g in y)

    if hybrid is not None and rope_once():
        # GLM_GK_ROPE_ONCE: the hybrid QKV's tables of the wo group's w * T rows, once for every layer.
        with sbm_name_scope("gk_rope1_"):
            hybrid["rope_full"] = _build_rope_tables(cos_g, sin_g, w_wo * T, qk_rope_head_dim)
    kernel_assert(
        hidden.shape[0] == T and (residual is None or residual.shape[0] == hidden.shape[0]),
        f"{_ERR} hidden has {hidden.shape[0]} rows, slot_mapping {T}",
    )

    # Hybrid + sharded wo over the same groups: the wo shard rides in the attention-weight gather (one
    # collective, one layer ahead), so no second gather is in flight on those groups.
    wo_merged = hybrid is not None and _same_groups(wo_gather_groups, attn_tp_gather_groups)
    if wo_scale is not None:
        # GLM_GK_PRJ_OMX: the MXFP8 wo is the resident row shard of the head-TP o_proj (never gathered).
        kernel_assert(
            hybrid is not None and wo_gather_groups is None,
            f"{_ERR} GLM_GK_PRJ_OMX needs the hybrid o_proj, unsharded wo",
        )
        kernel_assert(
            wo_scale.shape[0] == wo_shard.shape[0] and wo_scale.shape[1] * 8 == wo_shard.shape[1],
            f"{_ERR} GLM_GK_PRJ_OMX: wo_scale {wo_scale.shape} vs MX wo {wo_shard.shape}",
        )
    sh_tp = None
    if shared_tp > 1 and gate_up_w is not None:
        kernel_assert(
            shared_tp & (shared_tp - 1) == 0,
            f"{_ERR} shared_tp={shared_tp} must be a power of two",
        )
        # A partition of several groups (one per PP stage) splits each group the same way.
        moe_groups = [list(range(cp_degree))] if moe_replica_groups is None else [list(g) for g in moe_replica_groups]
        cp_groups = [list(range(cp_degree))] if cp_replica_groups is None else [list(g) for g in cp_replica_groups]
        kernel_assert(moe_groups == cp_groups, f"{_ERR} shared_tp needs the MoE groups in CP rank order")
        sh_tp = dict(
            m=shared_tp,
            groups=tuple(tuple(g[i : i + shared_tp]) for g in moe_groups for i in range(0, len(g), shared_tp)),
            rank_hbm=cp_rank,
        )

    # o_proj shards one layer ahead.
    wo_pre = [None] * n_layers

    def _prio(li: int):
        return None if li == 0 else _ATTN_PREFETCH_PRIORITY

    def _prefetch_attn(li: int):
        if hybrid is not None:
            stacks = (wqkv_a, wq_b, wuk, wuv)
            # GLM_GK_ATTN_RESIDENT: a resident weight is the whole slice (no gather); the others are gathered.
            gathered = [i for i in range(4) if not attn_res[i]]
            shards = [stacks[i][li] for i in gathered]
            if wo_merged:
                shards.append(wo_shard[li])
            fulls = _gather_rows(shards, attn_tp_gather_groups, _prio(li), f"hyattnL{li}_") if shards else []
            got = dict(zip(gathered, fulls))
            attn_pre[li] = tuple(
                _as_dtype(stacks[i][li], nl.bfloat16)
                if attn_res[i]
                else _as_dtype(got[i], nl.bfloat16).reshape(hy_shapes[i])
                for i in range(4)
            )
            if wo_merged:
                wo_pre[li] = _as_dtype(fulls[-1], nl.bfloat16)
        else:
            attn_pre[li] = _gather_attn_weights(
                wqkv_a[li],
                wq_b[li],
                wuk[li],
                wuv[li],
                cp_degree,
                cp_replica_groups,
                _prio(li),
                f"attnL{li}",
                attn_dims,
            )

    def _prefetch_wo(li: int):
        if wo_gather_groups is not None and not wo_merged:
            fulls = _gather_rows([wo_shard[li]], wo_gather_groups, _prio(li), f"woL{li}_")
            wo_pre[li] = _as_dtype(fulls[0], nl.bfloat16)

    _prefetch_attn(0)
    _prefetch_wo(0)
    # The gathered KV of a layer is read only by its own attention, which finishes before the next
    # layer's gather (that waits on the next layer's QKV), so one buffer serves every layer.
    s_kv = kv_ctx_local if block_table is not None else hidden.shape[0]
    # GLM_GK_KV_PLANS: one spare row past the gathered rows (_kv_spare_row); the layers read only the first cp * s_kv.
    kv_spare = 1 if block_table is not None and _kv_spare_row(s_kv) else 0
    kv_full = nl.ndarray(
        (cp_degree * s_kv + kv_spare, kv_caches.shape[-1]),
        dtype=nl.bfloat16,
        buffer=nl.private_hbm,
        name=scoped_name("gk_kv_full0"),
    )

    key_rows = None
    if emit_key_rows:
        kernel_assert(n_full > 0, f"{_ERR} emit_key_rows needs a full layer")
        key_rows = nl.ndarray(
            (n_full, T, idx_k_caches.shape[-1]),
            dtype=nl.bfloat16,
            buffer=nl.shared_hbm,
            name=scoped_name("gk_key_rows"),
        )
    i_full = i_moe = i_dense = 0
    h, r, topk = hidden, residual, None
    picks = None
    if topk_in is not None:
        # The stage before's raw picks, which the leading shared layers attend as its full layer's shared layers do.
        kernel_assert(
            tuple(topk_in.shape) == (T, index_topk), f"{_ERR} topk_in must be [{T}, {index_topk}] int32"
        )
        picks = topk_in.reshape((1, T, index_topk))
        topk = picks
    kv_n_slots = kv_caches.shape[-2]
    # Every indirect DMA into the KV stack addresses the whole stack at offset 0 and adds the layer's slot (or page)
    # base to its index vector on device: a per-layer static base past 2^32 B does not compile. The add runs in fp32,
    # so it is exact below 2^24 slots.
    kernel_assert(
        kv_caches.shape[0] * kv_n_slots < (1 << 24),
        f"{_ERR} {kv_caches.shape[0]} x {kv_n_slots} KV slots >= 2^24; split the stage",
    )
    kv_flat = kv_caches.reshape((kv_caches.shape[0] * kv_n_slots, kv_caches.shape[-1]))

    # GLM_GK_MOE_ZINIT_AT: ONE routed output for every MoE layer, zeroed by the GK at the site before each MoE layer.
    zi_site = _moe_zinit_at() if gate_up_w is not None else ""
    if zi_site and _moe_zinit_min_t() and T < _moe_zinit_min_t():
        # GLM_GK_MOE_ZINIT_MIN_T: this smaller graph keeps the bwmm's own zero-init.
        zi_site = ""
    rs_buf = None
    if zi_site:
        T_moe = _group_size(moe_replica_groups, cp_degree) * T
        rs_buf = nl.ndarray((T_moe, H), dtype=nl.bfloat16, buffer=nl.private_hbm, name=scoped_name("gk_rs_buf"))

    def _zinit_hook(li: int, site: str):
        """The zero-init of MoE layer ``li`` at ``site``, called with its fence tensors (None: not this site)."""
        if zi_site != site or layer_mlp_types[li] == "dense":
            return None
        return lambda *fences: _zero_moe_out(rs_buf, fences, f"L{li}_{site}")

    for li in range(n_layers):
        is_full = layer_indexer_types[li] == "full"
        is_dense = layer_mlp_types[li] == "dense"
        idx_w = None
        if is_full:
            idx_w = dict(
                k_cache=idx_k_caches[i_full],
                wq_b=idx_wq_b[i_full],
                wk=idx_wk[i_full],
                k_norm_w=idx_k_norm_w[i_full],
                k_norm_b=idx_k_norm_b[i_full],
                weights_proj=idx_weights_proj[i_full],
            )
        zinit_rs = _zinit_hook(li, "rs")
        if zinit_rs is not None:
            # GLM_GK_MOE_ZINIT_AT=rs: after the layer before (its reduce-scatter and combine wrote h). The previous MoE
            # layer's reduce-scatter read rs_buf, so the zero-init follows it in any case.
            zinit_rs(h)
        mlp_w = moe_w = None
        if is_dense:
            mlp_w = dict(gate=mlp_gate_w[i_dense], up=mlp_up_w[i_dense], down=mlp_down_w[i_dense])
        else:
            moe_w = dict(
                router_w=router_w[i_moe],
                router_bias=router_bias[i_moe],
                gate_up_w=gate_up_w[i_moe],
                gate_up_scale=gate_up_scale[i_moe],
                down_w=down_w[i_moe],
                down_scale=down_scale[i_moe],
                shared_gate_up_w=shared_gate_up_w[i_moe],
                shared_gate_up_scale=shared_gate_up_scale[i_moe],
                shared_down_w=shared_down_w[i_moe],
                shared_down_scale=shared_down_scale[i_moe],
            )
            moe_w["shared_tp"] = sh_tp
        with sbm_name_scope(f"L{li}_"):
            outs = _layer(
                h,
                r,
                cos,
                sin,
                q_pos,
                slot_mapping,
                cp_rank,
                ep_rank,
                wo_group_rank,
                kv_flat,
                input_norm_w[li],
                attn_pre[li],
                q_norm_w[li],
                kv_norm_w[li],
                (
                    (wo_shard[li], wo_scale[li]) if wo_scale is not None
                    else (wo_shard[li] if wo_pre[li] is None else wo_pre[li])
                ),
                post_norm_w[li],
                block_table,
                prior_tokens,
                None if is_full else topk,
                idx_w,
                mlp_w,
                moe_w,
                *_layer_prefetch_hooks(li, n_layers, _prefetch_attn, _prefetch_wo),
                kv_full=kv_full,
                hybrid=hybrid,
                key_rows_out=key_rows[i_full] if key_rows is not None and is_full else None,
                kv_slot_off=li * kv_n_slots,
                moe_rs_buf=None if is_dense else rs_buf,
                zinit_kv=_zinit_hook(li, "kv"),
                zinit_moe=_zinit_hook(li, "moe"),
                zinit_oproj=_zinit_hook(li, "oproj"),
                **layer_config,
            )
        h, r = outs[0], outs[1]
        if is_full:
            topk, picks = outs[3], outs[5]
            i_full += 1
        if is_dense:
            i_dense += 1
        else:
            i_moe += 1

    caches = (kv_caches,) + ((idx_k_caches,) if n_full else ())
    if final_norm_w is None:
        h_out = nl.ndarray((T, H), dtype=nl.bfloat16, buffer=nl.shared_hbm, name=scoped_name("gk_h_out"))
        r_out = nl.ndarray((T, H), dtype=nl.bfloat16, buffer=nl.shared_hbm, name=scoped_name("gk_r_out"))
        _hbm_copy_rows(h_out, h.reshape((T, H)), T, H)
        _hbm_copy_rows(r_out, r.reshape((T, H)), T, H)
        picks_out = ()
        if emit_picks:
            if n_full:
                # The last full layer's own [1, T, K] shared_hbm picks.
                picks_out = (picks,)
            else:
                relay = nl.ndarray(
                    (1, T, index_topk), dtype=nl.int32, buffer=nl.shared_hbm, name=scoped_name("gk_picks_relay")
                )
                _hbm_copy_rows(relay.reshape((T, index_topk)), topk_in, T, index_topk)
                picks_out = (relay,)
        return (h_out, r_out) + picks_out + caches + ((key_rows,) if emit_key_rows else ())

    # The model's final RMSNorm of h + r, in the same trace.
    with sbm_name_scope("gk_finnorm_"):
        fin_outs = rmsnorm_mx_prefill(
            hidden_states=h.reshape((1, T, H)),
            gamma=final_norm_w,
            eps=rms_norm_eps,
            residual=r.reshape((1, T, H)),
            emit_norm_bf16=True,
        )
    final_normed = nl.ndarray((T, H), dtype=nl.bfloat16, buffer=nl.shared_hbm, name=scoped_name("gk_final_normed"))
    fin = fin_outs[-1].reshape((T, H))
    _hbm_copy_rows(final_normed, fin, T, H)
    return (final_normed,) + caches + ((key_rows,) if emit_key_rows else ())
