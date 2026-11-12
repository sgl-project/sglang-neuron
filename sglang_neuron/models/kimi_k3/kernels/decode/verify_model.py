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

"""Kimi-K3 verify giga-kernel: the decode layers on Q tokens of each sequence (speculative decoding).

A speculative step scores Q = 1 + k tokens of each sequence: the bonus token of the last step and k
draft tokens. The entry runs ``decode_block.decode_block_body`` on the Q * B rows of the replica with
verify bodies, so the step has the semantics of Q decode steps in sequence:

- Row ``j * B + b`` is token ``j`` of sequence ``b``, at position ``position_rows[j, b]``.
- KDA: ``kda_verify_attention`` first replays the accepted tokens of the previous step from their
  records onto the committed state (replay SSM), writes that state back to its page, then scores the Q
  tokens and writes their records.
- MLA: one call of ``mla_decode_attention`` for each token ``j``, on the B rows of token ``j``. Call ``j``
  writes the cache rows of token ``j`` and attends the rows of the earlier calls.
- Feed-forward: the latent MoE and the dense MLP bodies gather the rows of the 4 replicas onto at most
  128 stationary columns. They run on chunks of ``ffn_rows`` rows of each replica (32 by default, the
  gather cap of ``MAX_REPLICA_TOKENS``). One chunk takes all rows of a step up to B = 4, Q = 8 (T = 128);
  a B = 8, Q = 8 step runs two chunks of 32. ``ffn_rows = 32`` gives a shorter 93-layer B = 4
  step. The current runtime orders the collective completion correctly, so 32 rows stay bitwise and
  deterministic; the ``drain_writes`` read-back of the stock runtime is off by default.
- Tail: the greedy tokens (or the logits) of all Q * B rows.
- Aux states: the drafter input of each layer of ``AUX_LAYERS``, for all rows. The default tap is the
  pre-norm AttnRes mixture that layer ``i + 1`` reads (``aux_tap="mixture"``); ``"prefix"`` stores the
  stream after layer ``i``.
- Split: the step can run as two calls on two layer ranges, for example ``[0, 48)`` and ``[48, 93)``. The first
  call returns its AttnRes bank and its boundary fence (``return_attn_res_bank``), and the second call takes both
  (``attn_res_bank_in``, ``attn_res_fence_in``) with the stream of the first call, so the rows are bitwise equal to
  one call. The fence completes only after every rank has written all its outputs (``boundary_fence``), so no peer
  still reads a collective source of the first call when the second call starts.

Every part keeps the per-token numerics of a decode step of B rows, so the tokens of a verify step equal
the tokens of Q decode steps.
"""

import itertools
from contextlib import nullcontext

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from .utils.kernel_assert import kernel_assert
from .moe.routed import expert_loop_layer
from .common import (
    COMPACT_MIX_TOKENS,
    P_MAX,
    SplitBank,
    WeightRow,
    attn_res_hT,
    is_trn3,
    load_rows_hT,
    load_vector_hT,
    new_ones,
    pair_shared_buffer,
    rmsnorm_hT,
    store_hT_rows,
)
from .decode_block import (
    _BANK_HBM_ROWS,
    NUM_LAYERS,
    banks_snapshot,
    block_layer_counts,
    check_moe_sbuf_reduces,
    check_pair_shared_forms,
    collective_sources,
    decode_block_body,
    layer_sources,
    moe_option_sites,
    moe_options_in_use,
    new_bank_rstd,
    pair_shared_expert_path,
    region_streams,
    trn3_kda_w_in_prefetch,
)
from .decode_model import (
    MAX_POOLS,
    TAIL_TP_RANKS,
    WORLD_TP_RANKS,
    _group_stack,
    _moe_group,
    _rows,
    model_layer_groups,
)
from .tkg_lm_head_greedy import lm_head_greedy_hT
from .tkg_weight_dma_gate import set_weight_gate, weight_gates
from .tkg_sbuf_alloc import AUTO_SBUF, ExplicitSbuf, SbufTiles, check_identities, install_identities
from .tkg_kda import HEAD_DIM, KDA_READOUTS, _w_in_buffers, kda_local_heads, kda_slot_state_tiles
from .verify_kda import kda_verify_attention, kda_verify_prework
from .tkg_mla_block import (
    CP_ROWS_OPTIONS,
    CP_TARGET_HEADS,
    LOOP_TAILS,
    MLA_PASS_ROWS,
    ROWS_LOOP_CASTS,
    ROWS_OPTIONS,
    ROWS_PASS_ROWS,
    ROWS_SEQUENCES,
    mla_block_attention,
    mla_block_collective_hbm,
    mla_block_collective_sources,
    mla_pass_plans,
    packed_prior_form,
    q_call_fp8,
)
from .tkg_mla_paged import LORA_NORM_EPS, QUERY_SET_TOKENS, mla_decode_attention, mla_runs_loops
from .tkg_moe import (
    COLUMN_FORMS,
    DP_SIZE,
    SCORE_GATHER_COLS,
    TP_SIZE,
    X_GATHER_PARITY_MAX_TOKENS,
    dense_mlp_decode,
    latent_moe_decode,
)
from .tkg_moe_dp import select_moe_body
from .multinode.decode_model_2srv import hold_to_end

AUX_LAYERS = (2, 23, 47, 71, 89)
"""Target layers whose output feeds the drafter (0-indexed)."""
AUX_TAPS = ("mixture", "prefix")
"""Aux taps: the pre-norm attention mix of layer ``i + 1``, or the stream after layer ``i``."""
MAX_QUERIES = 9
"""Most tokens of each sequence in one verify step."""
AUX_RING_MAX_ROWS = MAX_QUERIES
"""Most rows of a verify call whose aux stores take the persistent staging ring (the B = 1 forms)."""
DEFAULT_QUERIES = 8
"""Tokens of each sequence in the serving verify step: the bonus token and 7 drafts."""
FFN_CHUNK_ROWS = 32
"""Rows of each replica in one feed-forward call (T = 4 * ``ffn_rows`` gathered). 32 is ``MAX_REPLICA_TOKENS``:
one call takes a step up to B = 4, Q = 8 (T = 128), and a B = 8, Q = 8 step runs two chunks of 32. The current
runtime orders the collective completion correctly, so 32 rows stay bitwise and deterministic without the
``drain_writes`` read-back."""

KDA_TOKEN_TILES_FREE_QUERIES = 4
"""The largest Q at which the KDA verify takes ``token_tiles`` without the slot DMA set. At Q = 8 a trn3 verify with
``token_tiles`` and the slot DMA set off hangs (a collective never completes on some cores, then a DMA abort); with the
whole slot DMA set and the persistent slot conv sources it runs clean. ``check_kda_token_tiles`` refuses the other
combinations above this Q."""
KDA_TOKEN_TILES_SLOT_DMAS = (
    "kda_conv_load_gather",
    "kda_batched_conv_stores",
    "kda_slot_pages_contiguous",
    "kda_batched_slot_conv_stores",
    "kda_bf16_update_rows",
)
"""The slot DMA set of the KDA verify that ``token_tiles`` needs above ``KDA_TOKEN_TILES_FREE_QUERIES``."""

VERIFY_TAKES_KDA_STATE_DTYPE = True
"""``kimi_k3_verify_model`` takes the fp32 KDA state pages (``kda_state_dtype`` of the decode), so a caller can check for
it before it passes any value but ``bf16``: an older kernel has the argument but refuses the fp32 forms at trace time."""


def check_kda_token_tiles(queries: int, token_tiles: bool, slot_dmas: dict, slot_conv_persistent: bool) -> None:
    """Refuse ``kda_token_tiles`` above ``KDA_TOKEN_TILES_FREE_QUERIES`` tokens a sequence unless every key of
    ``KDA_TOKEN_TILES_SLOT_DMAS`` (``slot_dmas``: name -> value) and the persistent slot conv sources are on: the
    combination that ran clean on trn3 hardware at Q = 8. Other combinations at that Q hung. Fail closed.

    Example:
        >>> check_kda_token_tiles(4, True, {}, False), check_kda_token_tiles(8, False, {}, False)
        (None, None)
    """
    missing = [name for name in KDA_TOKEN_TILES_SLOT_DMAS if not slot_dmas.get(name)]
    kernel_assert(
        not token_tiles or queries <= KDA_TOKEN_TILES_FREE_QUERIES or (not missing and slot_conv_persistent),
        f"kda_token_tiles at Q = {queries} (> {KDA_TOKEN_TILES_FREE_QUERIES}) needs the whole KDA slot DMA set and the "
        f"persistent slot conv sources (kda_slot_conv_persistent); missing: "
        f"{missing + ([] if slot_conv_persistent else ['kda_slot_conv_persistent'])}. Without them the verify hung on "
        f"trn3 hardware. Turn kda_token_tiles off, or turn on the slot DMA set and the persistent sources",
    )


def loop_weight_buffers(rows: int, k1_two_buffers: bool) -> int | None:
    """Return the expert-loop weight buffers of the verify MoE on the loop path (None: the kernel default).

    The AttnRes bank of a loop-path step sits in the explicit SBUF region, so a 64-row step trades the second
    buffer for the bank bytes: more than 32 rows keep one buffer, 9 to 32 rows keep two, smaller steps keep the
    default. A k1 step above ``_BANK_HBM_ROWS`` rows keeps its bank in HBM, so with ``k1_two_buffers`` (the
    ``moe_k1_two_weight_buffers`` option of such a step) it keeps two buffers at any row count. The buffer count
    changes only the pipelining of the weight DMAs, not the values.

    Example:
        >>> loop_weight_buffers(64, False), loop_weight_buffers(64, True), loop_weight_buffers(32, False)
        (1, 2, 2)
    """
    if k1_two_buffers:
        return 2
    return 1 if rows > 32 else (2 if rows > 8 else None)


def attn_res_split_entries(sbuf_entries: int, rows: int, mla_mode: str, region: bool, entries: int, split: bool) -> int:
    """Return the AttnRes bank entries of a verify call in the explicit SBUF region (``attn_res_sbuf_entries``).

    Up to ``_BANK_HBM_ROWS`` rows the whole bank sits in SBUF, so the count does nothing there. 0 (the default)
    keeps the program. Above, a k1 call on the explicit region (``region``) keeps the first
    ``min(sbuf_entries, entries)`` entries in the region heap and the others in HBM (``common.SplitBank``). The count is
    for the k1 verify only, and not for the split verify, whose bank input and output are whole HBM banks.

    Example:
        >>> attn_res_split_entries(4, 16, "k1", True, 8, False), attn_res_split_entries(4, 8, "k1", True, 8, False)
        (4, 0)
    """
    kernel_assert(sbuf_entries >= 0, f"attn_res_sbuf_entries is a count >= 0, got {sbuf_entries}")
    if not sbuf_entries:
        return 0
    kernel_assert(mla_mode == "k1", "attn_res_sbuf_entries is for the k1 verify (mla_mode k1)")
    kernel_assert(
        not split, "attn_res_sbuf_entries is not for the split verify: its bank input and output are HBM banks"
    )
    if rows <= _BANK_HBM_ROWS:
        return 0
    kernel_assert(
        region,
        f"attn_res_sbuf_entries above {_BANK_HBM_ROWS} rows needs the explicit SBUF region (the MLA loop path) for "
        "the SBUF entries",
    )
    return min(int(sbuf_entries), entries)


def aux_tap_source(layer: int, aux_tap: str) -> int:
    """Return the layer whose call writes aux layer ``layer``: the layer itself with ``prefix`` (the stream after it),
    the next layer with ``mixture`` (the mix that the next layer reads). A split caller takes aux layer ``layer`` from
    the call that holds the returned layer.

    Example:
        >>> aux_tap_source(47, "prefix"), aux_tap_source(47, "mixture")
        (47, 48)
    """
    kernel_assert(aux_tap in AUX_TAPS, f"aux_tap must be one of {AUX_TAPS}")
    return layer if aux_tap == "prefix" else layer + 1


FENCE_SHAPE = (P_MAX, 2)
"""Shape of the split boundary fence (``return_attn_res_bank``): two stream-dtype words on each partition."""


def boundary_sinks(outputs) -> list:
    """Return ``(tensor, offset)`` of one element of each HBM write site of the outputs of a call.

    A ``[K, R, H]`` output (the aux states, the layer streams) has a site for each leading index, a
    ``[128, C, R, H/128]`` AttnRes bank a site for each entry, and every other output (the stream, a page pool) one
    site. The offsets count elements of the output.
    """
    sinks = []
    for tensor in outputs:
        shape = tuple(tensor.shape)
        if len(shape) == 4:
            sinks += [(tensor, c * shape[2] * shape[3]) for c in range(shape[1])]
        elif len(shape) == 3:
            sinks += [(tensor, k * shape[1] * shape[2]) for k in range(shape[0])]
        else:
            sinks.append((tensor, 0))
    return sinks


def fence_tiles(outputs, dtype) -> tuple:
    """Return the SBUF tiles of ``boundary_fence`` for ``outputs``, each written once here.

    Call it before the output stores: a tile that is live across a store cannot take the bytes of its source, which
    the compiler frees at the start of the transfer.
    """
    n = len(boundary_sinks(outputs))
    shapes = (
        ((P_MAX, 2 * n), dtype),
        ((P_MAX, n), nl.int32),
        ((P_MAX, n), nl.int32),
        ((P_MAX, n), nl.float32),
        ((P_MAX, 1), nl.float32),
        (FENCE_SHAPE, dtype),
    )
    tiles = tuple(nl.ndarray(shape, dtype=dt, buffer=nl.sbuf) for shape, dt in shapes)
    # The memset of the mask (the second tile) is the zero of the fence: a runtime value, not an immediate.
    for tile in tiles:
        nisa.memset(dst=tile, value=0, engine=nisa.engine.vector)
    return tiles


def boundary_fence(outputs, group: ReplicaGroup, tiles: tuple) -> nl.NkiTensor:
    """Return the split boundary fence: an all-reduce over ``group`` of zeros that read every HBM output of the call.

    The source of a rank reads one element of each write site (``boundary_sinks``), so the rank contributes only
    after its outputs are written, and so after every collective that feeds an output has landed on it. The fence of
    a rank completes only after every peer has contributed, so no peer still reads a collective source of the call.
    The collective source and destination are HBM tensors of this call; the fence output is a copy of the destination.
    ``tiles`` are the ``fence_tiles`` of ``outputs``.
    """
    sinks = boundary_sinks(outputs)
    n = len(sinks)
    bits, mask, zero_i, zero_f, total, word = tiles
    kernel_assert(tuple(bits.shape) == (P_MAX, 2 * n), "Make the fence tiles for the outputs that the fence reads")
    # One partition of each site: the other partitions keep the zeros of ``fence_tiles``.
    for i, (sink, offset) in enumerate(sinks):
        nisa.dma_copy(
            dst=bits[0:1, 2 * i : 2 * i + 2],
            src=sink.ap(pattern=[[0, 1], [1, 2]], offset=offset),
            name=f"attn_res_fence_read_{i}",
        )
    # bits AND the zero mask is 0 for every value, NaN included: the source is zero and depends on every read.
    nisa.tensor_tensor(dst=zero_i, data1=bits.view(nl.int32), data2=mask, op=nl.bitwise_and, engine=nisa.engine.vector)
    nisa.tensor_copy(dst=zero_f, src=zero_i, engine=nisa.engine.scalar)
    # This NKI has no engine argument for tensor_reduce; it lands on the Vector Engine.
    nisa.tensor_reduce(dst=total, data=zero_f, op=nl.add, axis=(1,))
    for col in range(FENCE_SHAPE[1]):
        nisa.tensor_copy(dst=word[0:P_MAX, col : col + 1], src=total, engine=nisa.engine.scalar)
    src = nl.ndarray(FENCE_SHAPE, dtype=word.dtype, buffer=nl.private_hbm, name="attn_res_fence_src")
    dst = nl.ndarray(FENCE_SHAPE, dtype=word.dtype, buffer=nl.private_hbm, name="attn_res_fence_dst")
    nisa.dma_copy(dst=src, src=word, name="attn_res_fence_store")
    ncc.all_reduce(srcs=[src], dsts=[dst], replica_group=group, op=nl.add, name="attn_res_fence_all_reduce")
    fence = nl.ndarray(FENCE_SHAPE, dtype=word.dtype, buffer=nl.shared_hbm, name="attn_res_fence")
    nisa.dma_copy(dst=fence, src=dst, name="attn_res_fence_out")
    return fence


_MLA_LOCAL_HEADS = 3
"""Heads of one TP32 rank for the target MLA (96 heads / 32), for the bitwise "k1" call on the Q*B rows."""
_K1_LOOP_GROUPS = 4
"""Prior-key groups per loop iteration of the "k1" call above 16 rows, so its MLA frame fits the region stack."""
_K1_FIT_BLOCKS = 64
"""Local blocks (the 256K table) from which the "k1" call above 16 rows takes ``fit_passes``."""


class _LayerIndex:
    """Stands in for the KDA page table of ``decode_block_body``: indexing returns the block-local KDA index."""

    def __getitem__(self, index):
        return index


def _row_tile(x_hT, start: int, count: int, sb=AUTO_SBUF):
    """Return rows ``start .. start + count - 1`` of an hT tile as a whole tile."""
    if start == 0 and count == x_hT.shape[1]:
        return x_hT
    tile = sb.tile((P_MAX, count, x_hT.shape[2]), x_hT.dtype)
    nisa.tensor_copy(dst=tile, src=x_hT[0:P_MAX, start : start + count, 0 : x_hT.shape[2]])
    return tile


def _after(x_hT, prev_hT, sb=AUTO_SBUF):
    """Return ``x_hT`` (bitwise) as a tile that depends on ``prev_hT``, so the work on it starts after ``prev_hT``.

    ``bits(prev) & 0`` is 0 for every value, NaN included, so the multiply by ``1 + 0`` keeps every bit of x.
    """
    bits = prev_hT[0:P_MAX, 0, 0:2].view(nl.int32)
    zero_i = sb.tile((P_MAX, 1), nl.int32)
    nisa.tensor_scalar(dst=zero_i, data=bits, op0=nl.bitwise_and, operand0=0, engine=nisa.engine.vector)
    one = sb.tile((P_MAX, 1), nl.float32)
    nisa.tensor_scalar(dst=one, data=zero_i, op0=nl.add, operand0=1.0, engine=nisa.engine.vector)
    out = sb.tile(x_hT.shape, x_hT.dtype)
    nisa.tensor_scalar(dst=out, data=x_hT, op0=nl.multiply, operand0=one, engine=nisa.engine.vector)
    return out


def _in_chunks(body, h_hT, chunks, serial: bool = False, sb=AUTO_SBUF):
    """Run ``body(rows_view, chunk_index)`` on row chunks ``(start, count)`` and assemble the output rows.

    Each chunk takes a view of its rows of ``h_hT``, so a body that gathers its input gathers from ``h_hT``.
    With ``serial``, each chunk starts after the output of the chunk before it, so the collectives of two
    chunks are never in flight together. ``sb`` allocates the output tile: a region puts it in the layer
    frame, where it reuses the bytes the attention body freed instead of competing with it.
    """
    if len(chunks) == 1:
        return body(h_hT, 0)
    out = sb.tile(h_hT.shape, h_hT.dtype)
    y = None
    for idx, (start, count) in enumerate(chunks):
        x = h_hT[0:P_MAX, start : start + count, 0 : h_hT.shape[2]]
        if serial and y is not None:
            x = _after(x, y, sb)
        y = body(x, idx)
        nisa.tensor_copy(dst=out[0:P_MAX, start : start + count, 0 : h_hT.shape[2]], src=y)
    return out


def _with_sources(options: dict, sources) -> dict:
    """Return ``options`` with ``sources`` in place of the block sources, or unchanged for None."""
    return options if sources is None else {**options, "sources": sources}


def _chunk_sources(sources, idx: int):
    """Return ``sources`` with the output all-reduce source of feed-forward chunk ``idx`` (``out_hbm_sets``) and, with
    the column out-projection forms, the world gather pairs of the chunk (``gather_hbm_sets``) and, with
    ``column_shared_ag``, its shared gather tiles (``shared_ag_sets``)."""
    if sources is None or "out_hbm_sets" not in sources:
        return sources
    chunk = {"out_hbm": sources["out_hbm_sets"][idx]}
    for key in ("gather_hbm_sets", "shared_ag_sets"):
        if key in sources:
            chunk.update(sources[key][idx])
    return {**sources, **chunk}


def verify_bodies(
    queries: int,
    kda_state_page,
    kda_record_page,
    kda_replay_len,
    kda_state_position,
    ffn_rows: int,
    serial: bool = False,
    drain_writes: bool = False,
    sbuf=AUTO_SBUF,
    weight_buffers: int = None,
    mla_mode: str = "per_token",
    mla_sources=None,
    mla_state=None,
    kda_ar=None,
    ffn_sources=None,
    lora_norm_eps: float = LORA_NORM_EPS,
    ffn_calls=None,
    mla_calls=None,
    mla_cp: int = 0,
    kda_replay: str = "records",
    kda_state_slot_page=None,
    kda_write_state_slots=None,
    kda_prework_ahead: bool = False,
    kda_prework_deferred: bool = False,
    kda_conv_dtype: str = "bf16",
    kda_chain: str = "sequential",
    kda_token_tiles: bool = False,
    kda_slot_store_queues: int = 1,
    kda_slot_store_engine: str = "sync",
    kda_w_in_buffers: int = 0,
    kda_readout: str = None,
    kda_state_loads_first: bool = False,
    kda_slot_pages_first: bool = False,
    kda_conv_load_gather: bool = False,
    kda_batched_conv_stores: bool = False,
    kda_null_page: int = 0,
    kda_slot_state_ring: tuple = None,
    kda_slot_state_waits: dict = None,
    kda_slot_state_depth: int = 2,
    kda_shared_state_ring: bool = False,
    half_weight_buffers: bool = False,
    gate_up_half_major: bool = False,
    kda_ar_sbuf=None,
    mla_q_hbm: bool = False,
    mla_out_ag: bool = False,
    mla_k1_fit: bool = False,
    mla_zeroed_pages: bool = False,
    mla_loop_cast_engines: str = "off",
    mla_loop_pad_stationary: bool = False,
    moe_body=latent_moe_decode,
    mla_late_tail_weights: bool = False,
    mla_pass_q_prefetch: int = 0,
    mla_q_gather_buffers: int = 1,
    mla_tail_weights_after_rs: bool = False,
    mla_sums_rs_first: bool = None,
    mla_pass_rows_ahead: bool = False,
    mla_late_o_proj_weights: bool = False,
    mla_late_tail_frames: bool = False,
    dense_body=dense_mlp_decode,
    mla_o_proj_stream: bool = None,
    attn_rs_moe_input: bool = False,
    mla_k1_rows: bool = False,
    mla_k1_rows_options: dict | None = None,
    live_rows: nl.NkiTensor = None,
    mla_k1_packed: bool = False,
    mla_k1_pack_options: dict | None = None,
    mla_cp_own_key_select: bool = False,
    mla_loop_first_static: bool = False,
    mla_scores_ahead: bool = False,
    mla_folded_reference: bool = False,
    mla_cp_loop_p2: bool | None = None,
    mla_cp_rows_options: dict | None = None,
    mla_cp_loop_preload: bool = False,
    mla_critical_dma_priority: bool | str = False,
    mla_cache_write_off_sp: bool = False,
    mla_gemv_bands: bool = False,
    mla_batched_cache_write: bool = False,
    mla_own_rows_ahead: bool = False,
    kda_state_dtype: str = "bf16",
    kda_paired_pages: bool = False,
    mla_k1_fused_masks: bool = False,
    mla_k1_weights_on_act: bool = False,
    mla_k1_row_pipeline: bool = False,
    mla_k1_o_proj_ahead: bool = False,
    mla_loop_restart: str = "off",
    mla_loop_groups: int = 0,
    mla_o_proj_ring: int = 0,
    mla_loop_page_gather: str = "off",
    mla_o_proj_early: bool = False,
    mla_loop_inputs_ahead: bool = False,
    mla_wgate_early: str = "off",
    mla_loop_dr_keys: bool = False,
    mla_out_ar: bool = False,
    mla_state_bf16_o: bool = False,
    kda_slot_pages_contiguous: bool = False,
    kda_batched_slot_conv_stores: bool = False,
    kda_bf16_update_rows: bool = False,
    kda_bf16_cols_once: bool = False,
    kda_commit_after_w_in: bool = False,
    kda_slot_store_regs: bool = False,
    kda_slot_conv_after_ar: bool = False,
    kda_weight_dma: str = "pool",
    kda_late_keep_warm: int = 0,
    kda_slot_ring_per_call: bool = False,
    kda_slot_ring_keep_sink: nl.NkiTensor = None,
    kda_consts: dict = None,
    kda_slot_ring_keep_defer: list = None,
    kda_slot_conv_rows: dict = None,
    kda_slot_conv_waits: list = None,
    kda_update_bcast_depth: int = 2,
    kda_head_pipeline: str = "off",
    kda_out_ar_rows: bool = False,
    kda_in_proj_x_stationary: bool = False,
    kda_state_loads_after_ring: bool = False,
    kda_early_out_tokens: int = None,
    kda_in_proj_lanes_view: bool = True,
    kda_states_after_gate_result: bool = True,
    kda_token_hook: bool = True,
    kda_x_quant_hook: bool = False,
    kda_merged_state_loads: bool = False,
    kda_w_out_after_w_in_fills: bool = False,
    kda_slot_conv_from_inputs: bool = False,
    kda_conv_products_engine: str = "gpsimd",
    kda_recurrence_group: int = None,
    mla_split: dict = None,
    kda_conv_zero_rows: nl.NkiTensor = None,
    kda_page_groups: tuple = None,
    kda_page_sets: list = None,
    kda_replay_ahead: bool = False,
    kda_w_in_psum_bufs: int = 1,
    kda_w_in_refill_queues: str = "alternate",
):
    """Return the ``(kda, mla, moe, dense)`` bodies of one AttnRes block of the verify step.

    The bodies take the arguments that ``decode_block_body`` gives a decode body. The block gets
    ``_LayerIndex()`` as its KDA page table and ``position_rows`` ``[Q, B, 1]`` as its position, and the
    MLA slot table ``[n_mla, Q, B, 1]``.

    Args:
        queries: Q, the tokens of each sequence.
        kda_state_page, kda_record_page: HBM int32 ``[n_kda, B, 1]`` of the block's KDA layers.
        kda_replay_len, kda_state_position: HBM int32 ``[B, 1]`` (see ``kda_verify_attention``).
        ffn_rows: rows of each replica in one feed-forward call.
        serial: start each MLA call and each feed-forward chunk after the output of the one before it.
        drain_writes: the attention output of each layer waits for a static read-back of the layer's pool,
            which orders the next layer after the page writes have landed. It works around a runtime that
            can read a page before its write lands (NaN rows at 32 rows of each replica).
            The served runtime orders the reduce-scatter completion correctly, so the drain is off by default;
            at 32 rows the read-back is itself nondeterministic, so leave it off on that runtime.
        sbuf: the tile allocator of the feed-forward row chunks. On the loop path it is the layer region,
            so the chunk tiles reuse the bytes that the attention body freed; off the region it is auto.
        weight_buffers: SBUF weight buffers of the latent MoE expert loop (1 to 3), or None for its default
            of 3. The verify loop path uses fewer, because its AttnRes bank holds every row of the step and
            grows with the layers, so it needs the bytes the extra weight buffers would take.
        half_weight_buffers: with ``weight_buffers``, stream half projections through twice as many half-size
            buffers in place of them (``half_weight_buffers`` of ``moe_tkg_decode``; the same SBUF). The values
            do not change; False (default) keeps the program.
        gate_up_half_major: every static expert loop reads the half-major gate/up weights
            (``gate_up_half_major`` of ``moe_tkg_decode``). The values do not change; False (default) keeps the
            program.
        mla_sources: two parity sets of per-token MLA collective tiles (static path), used round-robin by the
            per-token calls so one token's all-gather source bytes are not taken by the next token after only
            the local completion. None on the loop path, where each call self-allocates in its freed frame.
            With ``mla_cp``, the persistent sites of the CP MLA body (``collective_sources(mla_cp=...)["mla_cp"]``).
            With ``mla_mode="k1"``, the sites of the multi-query call (``collective_sources(...)["verify_k1"]``).
        lora_norm_eps: the epsilon of the q_a and kv_a RMSNorms of both MLA modes (``lora_norm_eps`` of
            ``mla_block_attention`` and ``mla_decode_attention``).
        kda_ar: the persistent all-reduce HBM pair (source, destination) of each KDA layer of the block, by
            block-local KDA index, or None.
        kda_ar_sbuf: None (default), or the persistent SBUF all-reduce source tile of each KDA layer of the block, by
            block-local KDA index (an entry None keeps that layer on ``kda_ar``): the layer's output all-reduce runs
            SBUF -> SBUF (``verify_kda.kda_verify_attention`` ``ar_sbuf``). The values do not change.
        ffn_sources: the chunk-sized collective sources of the feed-forward bodies, or None for the block sources.
            With their ``pingpong`` set, each chunk copies its rows into the gather source set of its turn.
        ffn_calls: the counter of the chunk turns, shared by the blocks of one kernel (None: a new counter).
        mla_calls: the counter of the k1 MLA calls, shared by the blocks of one kernel (None: a new counter). With the
            ``qkv_parity`` key of ``mla_sources`` (two ``{"qkv", "qkv_dst"}`` sets), call n takes set n mod 2 for its
            qkv all-gather. Without the key, the counter does nothing.
        mla_cp: 0, or 128: the MLA layers run one bitwise ``mla_block_attention`` call with the CP128 layout on the
            Q*B rows, so each row is bitwise equal to the CP128 decode (``tkg_mla_block.mla_cp_decode_attention``).
        kda_prework_ahead: give the KDA body a ``prework`` attribute (``verify_kda.kda_verify_prework``), so the block
            traces the page metadata and the state loads of each KDA layer beside the output all-reduce of the
            feed-forward body before it (``decode_block_body``). The values do not change.
        kda_prework_deferred: with ``kda_prework_ahead`` and ``kda_replay`` "states": the prework of each KDA layer
            keeps the VectorE selects of its loads for the KDA body, which runs them after its input projection's
            quantize (``verify_kda.kda_verify_prework`` ``defer_selects``), so VectorE does not wait for the loads in
            the feed-forward tail. The prework also takes ``kda_merged_state_loads``. The same loads and selects, so
            the values do not change; False (default) keeps the program.
        kda_conv_dtype: form of the KDA short convolution (``verify_kda.kda_verify_attention`` ``conv_dtype``).
        kda_chain: the ``chain`` of each KDA verify call (``verify_kda.KDA_CHAINS``). The ``states`` slot-state ring
            serves the ``sequential`` chain (2 tiles) and the ``batched`` chain (Q tiles, from a kernel that allocates
            them). ``chunked``, and ``batched`` without a kernel ring, write the Q slot states from per-call tiles that
            a later instruction can reuse before the slot stores finish, so they are not for serving (default-off).
        kda_token_tiles, kda_slot_store_queues: ``states``: the ``token_tiles`` and ``slot_store_queues`` of each KDA
            verify call (``verify_kda.kda_verify_attention``). The defaults keep the program. Above
            ``KDA_TOKEN_TILES_FREE_QUERIES`` (4) tokens a sequence, ``kda_token_tiles`` needs the slot DMA set and the
            persistent slot conv sources (``check_kda_token_tiles``).
        kda_slot_store_engine: ``states``: the queue of the per-token slot state stores of each KDA verify call
            (``verify_kda.kda_verify_attention`` ``slot_store_engine``): "sync" (default) keeps the program, "gpsimd" or
            "split" moves them (or half of them) to the GpSimd software DGE queue. The bits do not change.
        kda_slot_state_ring, kda_slot_state_waits, kda_slot_state_depth: ``states`` with the ``sequential`` (or,
            ring and record only, the ``batched``) ``kda_chain``: the persistent slot-state ring tiles of the kernel,
            the shared waits record, and the ring depth
            (``verify_kda.kda_verify_attention`` ``slot_state_ring`` / ``slot_state_waits`` / ``slot_state_depth``). The
            ring and the record are None off a serving kernel, so the body keeps only the within-call edges. The bits
            do not change.
        kda_shared_state_ring: ``states`` with the ``sequential`` ``kda_chain``: the ``shared_state_ring`` of each KDA
            verify call. The shared-tile path (more than one head, no ``kda_token_tiles``) cycles the 2-tile slot-state
            ring, so the update of a token does not wait for the slot stores of the token before. The bits do not
            change; False (default) keeps the program.
        kda_w_in_buffers: the ``w_in_buffers`` of each KDA verify call (the ``kda_w_in`` ring depth). 0 (default)
            keeps the program. The KDA body also takes the ring of a layer whose first fills the block issued early
            (``w_in_ring``, ``decode_block_body`` ``kda_w_in_prefetch``).
        kda_readout: the ``readout`` of each KDA verify call (``tkg_kda.KDA_READOUTS``). None (default) keeps
            ``kda_default_readout`` of the call's sequences and the program.
        kda_state_loads_first, kda_slot_pages_first: ``states``: the ``state_loads_first`` and ``slot_pages_first`` of
            each KDA verify call. Not with ``kda_prework_ahead``. The defaults keep the program.
        kda_conv_load_gather: the ``conv_load_gather`` of each KDA verify call (one gather DMA per group of sequences
            for the committed conv-state load, ``verify_kda.kda_verify_attention``). Bitwise the same; False (default).
        kda_batched_conv_stores: the ``batched_conv_stores`` of each KDA verify call (one scatter DMA per group for
            the committed conv-state write, ``verify_kda.kda_verify_attention``). Bitwise the same; False (default).
        kda_null_page: 0 (default), or the KDA null page of each KDA verify call (``verify_kda.kda_verify_attention``
            ``null_page``, as ``decode_model`` ``kda_null_page``). The rows of the other pages do not change.
        kda_slot_pages_contiguous: ``states``: the ``slot_pages_contiguous`` of each KDA verify call and prework (the
            slot ids loaded in their HBM order, one contiguous run per partition). The bits do not change; False
            (default) keeps the program.
        kda_batched_slot_conv_stores: ``states``: the ``batched_slot_conv_stores`` of each KDA verify call (one scatter
            DMA per group of rows for the slot conv windows). The rows of every slot page do not change (an off slot's
            page 0 / null page gets zeros); False (default) keeps the program.
        kda_bf16_update_rows: the ``bf16_update_rows`` of each KDA verify call (with the ``decoupled_bf16`` readout,
            the bf16 high and low rows of each token's update from one bf16 TensorE transpose per 4 sequences, added in
            one broadcast matmul per head). The bits do not change; False (default) keeps the program.
        kda_commit_after_w_in: the ``commit_after_w_in`` of each KDA verify call (the committed-state stores after the
            input projection's prefetch fills on the Sync queue). The bits do not change; False (default) keeps the
            program.
        kda_slot_store_regs: ``states`` with one Sync slot-store queue: the ``slot_store_regs`` of each KDA verify call
            (each scored token's slot page ids in registers ahead of its slot state stores, so the Sync queue loads them
            while the store before is in flight). The bits do not change; False (default) keeps the program.
        kda_slot_conv_after_ar: ``states`` with the per-token state tiles: the ``slot_conv_after_ar`` of each KDA verify
            call (the slot conv-state stores after the call's output all-reduce, so the first slot state store waits for
            its own state only). The bits do not change; False (default) keeps the program.
        kda_weight_dma: the ``weight_dma`` of each KDA verify call (``KDA_WEIGHT_DMAS``: the queue of the output-projection,
            small-weight and parameter loads). The bits do not change; ``pool`` (default) keeps the program.
        kda_bf16_cols_once: with ``kda_bf16_update_rows`` and the per-token states path: the ``bf16_cols_once`` of each
            KDA verify call (the update's bf16 rows tiles zeroed once before the chain, no GpSimd memset in each token).
            The bits do not change; False (default) keeps the program.
        kda_slot_ring_per_call: with ``kda_token_tiles`` on the sequential ``states`` chain and no ``kda_slot_state_ring``:
            the ``slot_ring_per_call`` of each KDA verify call (a slot-state ring of Q tiles of the call, kept live until
            its stores have read it, so no token's update waits for an earlier token's slot stores). The bits do not
            change; False (default) keeps the program.
        kda_consts: the ``consts`` of each KDA verify call (``verify_kda.kda_verify_constants``: the kernel's all-ones tiles,
            set once, in place of the memsets of each call; the same values). None (default) keeps the program.
        kda_slot_ring_keep_sink: with ``kda_slot_ring_per_call``: the ``slot_ring_keep_sink`` of each KDA verify call,
            a tile that the caller keeps live to the kernel end (None: a sink of each call, dead code for the compiler).
        kda_slot_ring_keep_defer: with ``kda_slot_ring_keep_sink``: the ``slot_ring_keep_defer`` record of each KDA verify
            call (each call's keep-reads trace at a later call's start; the caller flushes the last).
        kda_late_keep_warm: the ``late_keep_warm`` of each KDA verify call (zero-data TensorE matmuls before and inside the
            late output projection, so it runs at the trn3 di/dt 8/8 rate). The bits do not change; 0 (default) keeps
            the program.
        kda_update_bcast_depth: the ``update_bcast_depth`` of each KDA verify call (the PSUM tiles of the state-update
            broadcast ring, 2 to 4). The bits do not change; 2 (default) keeps the program.
        kda_in_proj_x_stationary: the ``in_proj_x_stationary`` of each KDA verify call (the token-stationary MXFP8 input
            projection, bitwise); False (default) keeps the program.
        kda_state_loads_after_ring: the ``state_loads_after_ring`` of each KDA verify call (its state and conv-state
            loads after the last prefetch fill of its ``kda_w_in`` ring in the schedule, bitwise); with
            ``kda_state_loads_first``. False (default) keeps the program.
        kda_head_pipeline: the ``head_pipeline`` of each KDA verify call (``off`` (default) keeps the program; ``g_last``
            runs the output-gate blocks of the MXFP8 input projection after the decay gate; ``heads`` the head-major
            stages). The bits do not change.
        kda_out_ar_rows: ACCURACY CLASS (<= 1 BF16 ulp). The ``out_ar_rows`` of each KDA verify call (the output
            all-reduce as two row chunks). False (default) keeps the program.
        kda_early_out_tokens, kda_in_proj_lanes_view, kda_states_after_gate_result: the ``early_out_tokens`` (None:
            ``EARLY_OUT_TOKENS``; 0: no early output tail), ``in_proj_lanes_view`` (False: the lane copies)
            and ``states_after_gate_result`` (False: the state DMA order without the gate-result edges) of each KDA
            verify call.
        kda_token_hook: the KDA body takes ``token_hook``, so ``decode_block_body`` ``AHEAD_MLP_TOKEN_STEPS`` runs the
            MLP AttnRes ahead part in the token chain. True (default); False keeps the ahead part out of the token
            chain.
        kda_merged_state_loads: the KDA bodies load each sequence's state once, from its selected page
            (``verify_kda.kda_verify_attention`` ``merged_state_loads``). The values do not change. False (default).
        kda_w_out_after_w_in_fills: with the MXFP8 ``kda_w_in``, each KDA body loads its ``kda_w_out`` (and its MX
            scales) after the last two ``kda_w_in`` fills of its input projection (``verify_kda.kda_verify_attention``
            ``w_out_after_w_in_fills``). The values do not change. False (default) keeps the program.
        kda_x_quant_hook: the KDA body takes ``x_quant_hook``, so ``decode_block_body`` traces the bank sums of a KDA
            layer's MLP mix right after the quantize of its MXFP8 input projection. The bits do not change; False
            (default) keeps the program.
        kda_slot_conv_from_inputs: ``states`` with the sequential chain: the ``slot_conv_from_inputs`` of each KDA verify
            call (the slot conv stores read their windows from the conv inputs, with no GpSimd window copies). The bits
            do not change; False (default) keeps the program.
        kda_conv_products_engine: the ``conv_products_engine`` of each KDA verify call (the engine of the one-pass conv
            tap products, ``gpsimd`` or ``vector``). The bits of finite values do not change; ``gpsimd`` (default)
            keeps the program.
        kda_recurrence_group: the ``recurrence_group`` of each KDA verify call (the sequences of each delta-rule group
            of a token, each group with its own PSUM tiles). The bits do not change; None (default) keeps the program.
        kda_slot_conv_rows, kda_slot_conv_waits: ``states``: the ``slot_conv_rows`` (persistent SBUF tiles of the slot
            conv-state stores, ``verify_kda.kda_slot_conv_rows_tiles``) and the shared ``slot_conv_waits`` record of each
            KDA verify call. None (default) keeps per-call tiles.
        kda_conv_zero_rows: the bf16 zero tile of the conv-state scatters that lives through the kernel
            (``tkg_kda.kda_conv_zero_rows_tile``), for every KDA verify call (``kda_verify_attention``
            ``conv_zero_rows``). The values do not change. None (default) keeps a zero tile per scatter.
        kda_page_groups, kda_page_sets: the KV cache group of each KDA layer of the block (a tuple of ints) and the
            page set of each group (``verify_kda.kda_page_group_tiles``), shared by the blocks of the call: each KDA
            verify call takes the set of its group (``kda_verify_attention`` ``page_group``). None (default) keeps the
            per-layer page metadata.
        kda_replay_ahead: with ``kda_prework_ahead``, ``records`` and the ``sequential`` chain, the prework of each KDA
            layer also replays its records (``verify_kda.kda_verify_prework`` ``replay_ahead``): ``decode_block_body``
            passes it the layer's small weights, and the replay chain runs where the block traces the prework (beside
            the output all-reduce of the feed-forward body before, or in its window with ``kda_prework_window``), not
            in the KDA body. The bits do not change; False (default) keeps the program.
        kda_w_in_refill_queues: the ``w_in_refill_queues`` of each KDA verify call (``alternate``, the default, keeps
            the program; ``sync`` puts every BF16 ``kda_w_in`` ring refill on the SP queue). The bits do not change.
        kda_w_in_psum_bufs: the ``w_in_psum_bufs`` of each KDA verify call (Trn2: the PSUM tiles of the input
            projection's chunk sums, for calls of at most ``verify_kda.W_IN_PSUM_MAX_ROWS`` rows). The bits do not change;
            1 (default) keeps the program.
        mla_q_hbm: ``mla_mode="k1"``: the call gathers the queries of all rows once, in the HBM pair of
            ``mla_sources`` (``mla_block_attention`` ``q_hbm``).
        mla_out_ag: ``mla_mode="k1"``: one output all-gather a call (``mla_block_attention`` ``out_ag_call``).
        mla_k1_fit: ``mla_mode="k1"``: ``mla_block_attention`` ``fit_passes``.
        mla_zeroed_pages: the MLA loop paths read their cache pages with no clamp (``zeroed_pages`` of
            ``mla_block_attention`` and ``mla_decode_attention``).
        mla_loop_cast_engines: ``mla_mode="per_token"``: the per-token MLA calls take ``mla_decode_attention``
            ``loop_cast_engines`` (the engines of the FP8 -> BF16 page-tile casts of the per-sequence key loops).
            ``mla_mode="k1"`` (not ``mla_cp``): a page form ("dma", "dma6", "dma4") goes to the k1 call's
            ``mla_block_attention`` ``loop_cast_engines`` (BF16 group tiles from DMA-cast page loads, at most 8 / 6 / 4
            groups an iteration); the engine forms have no k1 form. Bitwise; "off" (default) keeps the calls.
        mla_loop_pad_stationary: ``mla_mode="per_token"``: the per-token MLA calls take ``mla_decode_attention``
            ``loop_pad_stationary`` (128-column stationaries in the per-sequence key loops). Bitwise; False (default)
            keeps the calls.
        moe_body: the MoE body of the decode (``tkg_moe_dp.select_moe_body``): ``latent_moe_decode`` (TP128
            projections, the default) or the DP4 x TP32 body. The feed-forward chunks call it as the decode does.
        mla_late_tail_weights: ``mla_mode="k1"``: ``mla_block_attention`` ``late_tail_weights``.
        mla_late_o_proj_weights: with ``mla_late_tail_weights``: ``mla_block_attention`` ``late_o_proj_weights``.
        mla_late_tail_frames: with ``mla_late_o_proj_weights``: ``mla_block_attention`` ``late_tail_frames``.
        mla_pass_q_prefetch: ``mla_mode="k1"``: ``mla_block_attention`` ``pass_q_prefetch``.
        mla_q_gather_buffers: ``mla_mode="k1"``: ``mla_block_attention`` ``q_gather_buffers``.
        mla_tail_weights_after_rs: ``mla_mode="k1"``: ``mla_block_attention`` ``tail_weights_after_rs``.
        mla_sums_rs_first: ``mla_block_attention`` ``sums_rs_first`` (None: ``SUM_RS_FIRST``).
        mla_pass_rows_ahead: ``mla_mode="k1"``: ``mla_block_attention`` ``pass_rows_ahead``.
        dense_body: the dense MLP body of the decode (layer 0): ``dense_mlp_decode`` (default) or, for a multi-host world,
            ``multinode.tkg_ffn_world_cc.dense_ffn_world``. The feed-forward chunks call it as the decode does.
        mla_o_proj_stream: the per-token MLA calls load one o_proj weight chunk at a time (``o_proj_stream`` of
            ``tkg_mla_paged.mla_decode_attention``); bitwise the same output. None (default) turns it on for the loop path
            above 16 rows; an entry gives the rule of its decode (the Trn2 entry: the loop path at 8 rows a call and
            fewer).
        mla_loop_restart: the ``loop_restart`` of each per-token MLA call (``tkg_mla_paged.LOOP_RESTARTS``): the
            iteration restart of its key loops. With ``mla_mode="k1"`` (not ``mla_cp``) a tail form of the k1 call
            ("tail4", "tail2", "tail1", "tail421", "tail21", "tail8", "tail8421": ``mla_block_attention``
            ``loop_tail_groups`` of ``tkg_mla_block.LOOP_TAILS``): the last iteration of each key loop with only its first
            4, 2 or 1 key groups when the others hold no prior key; the cascades "tail421" and "tail21" take the
            smallest of their tails that holds the iteration's groups; "tail8" and "tail8421" also run every other last
            iteration in a full-iteration tail, so the main loop reads only prior keys and skips the page clamp
            (bitwise the same on finite cache rows). "off" (default) keeps the program; bitwise the same output.
        mla_loop_groups: the ``loop_groups`` of each per-token MLA call (``tkg_mla_paged.LOOP_GROUP_COUNTS``): the key groups
            of one iteration of its key loops. 0 (default) keeps the program; bitwise the same output.
        mla_o_proj_ring: the ``o_proj_ring`` of each per-token MLA call that streams its o_proj
            (``tkg_mla_paged.O_PROJ_RINGS``): the o_proj weight chunks in flight. 0 (default) keeps the program; bitwise
            the same output.
        mla_loop_page_gather: the ``loop_page_gather`` of each per-token MLA call (``tkg_mla_paged.LOOP_PAGE_GATHERS``), or
            with ``mla_mode="k1"`` (not ``mla_cp``) of the k1 call (``mla_block_attention`` ``loop_page_gather``): the
            page loads of its key loops. "off" (default) keeps the program; bitwise the same output.
        mla_o_proj_early: with ``mla_o_proj_ring``, the ``o_proj_early`` of those calls: the ring's first chunks load
            with the head weights, before the key loops. False (default) keeps the program; bitwise the same output.
        mla_loop_inputs_ahead: the ``loop_inputs_ahead`` of each per-token MLA call: its sequences' loop inputs before
            its first key loop. False (default) keeps the program; bitwise the same output.
        mla_wgate_early: the ``wgate_early`` of each per-token MLA call (``tkg_mla_paged.WGATE_EARLY_FORMS``): where its
            gate projection weight loads issue. "off" (default) keeps the program; bitwise the same output.
        mla_k1_rows: ``mla_mode="k1"``: the key-stationary rows form of the CP32 call (``mla_block_attention``
            ``fused_masks`` and ``transposed_scores``). It changes the per-row sums, so the decode twin must run
            ``tkg_mla_block.mla_rows_decode_attention``. Default off.
        mla_k1_rows_options: with ``mla_k1_rows``: more ``mla_block_attention`` keys of the rows form, from
            ``tkg_mla_block.ROWS_OPTIONS``. The decode twin takes the values of the keys in
            ``tkg_mla_block.ROWS_TWIN_OPTIONS``.
        live_rows: None, or the runner's INT32 ``[B, 1]`` live-row mask (1: a live request, 0: an empty slot) of the
            k1 call (``mla_block_attention`` ``live_rows``): the padding sequences of ``rows_pad_skip`` and of the
            packed loop's ``pad_skip``.
        mla_k1_packed: ``mla_mode="k1"`` on the loop path: the call attends its prior keys with one packed loop of
            items of all Q rows of a sequence (``mla_block_attention`` ``packed_prior``), with
            ``mla_k1_pack_options`` as its ``packed_prior_options``. Bitwise the same; default off.
        mla_cp_own_key_select: the CP128 call takes ``mla_block_attention`` ``own_key_select``. Default off.
        mla_loop_first_static: the CP128 call takes ``mla_block_attention`` ``loop_first_static``. Default off.
        mla_scores_ahead: the CP128 call takes ``mla_block_attention`` ``scores_ahead``. Default off.
        mla_folded_reference: the CP128 call takes ``mla_block_attention`` ``folded_reference``. Default off.
        mla_cp_loop_p2: the CP128 call on the loop path takes ``mla_block_attention`` ``loop_p2`` when Q x 96 is a
            multiple of 128 (Q = 4, 8), the call is one pass of at most 16 rows and its masks are fused (B divides 16);
            another form keeps the call. None (default): at B <= 2; True: at each such B; False: off. Bitwise.
        mla_cp_rows_options: with ``mla_cp``, keys of the rows form for the CP128 call (``tkg_mla_block.CP_ROWS_OPTIONS``:
            ``rows_dr``, ``rows_dr_act_copies``, ``rows_dr_inplace``), as ``mla_k1_rows_options`` gives them to the k1 rows
            call. ``rows_dr`` (the GPU decode's e4m3 query with FP8 double_row QK keys) changes the bits, so the CP128
            decode of the deployment must take the same keys. None (default) keeps the call.
        mla_cp_loop_preload: with ``mla_loop_first_static``, the CP128 call on the loop path takes ``mla_block_attention``
            ``loop_preload`` (the static first iteration loads the pages of the second). Bitwise; default off.
        mla_critical_dma_priority: ``mla_block_attention`` ``critical_dma_priority`` (False, True or "query"). Bitwise;
            default off.
        mla_cache_write_off_sp: ``mla_block_attention`` ``cache_write_off_sp``. Bitwise; default off.
        mla_gemv_bands: ``mla_block_attention`` ``gemv_bands``. Bitwise; default off.
        mla_batched_cache_write: ``mla_block_attention`` ``batched_cache_write``: the call writes its current rows
            with one vector-indirect DMA, and each row that the rank does not own (or a pad row) goes to its own row
            of the null page. The values read do not change; only the null page holds other bytes. At most 128 rows
            a call. Default off.
        mla_own_rows_ahead: ``mla_block_attention`` ``own_rows_ahead`` (the bitwise own-key rows of each pass load
            at its start on the Activation queue, into quadrant partitions, in place of one DMA a row). Bitwise; default
            off.
        kda_state_dtype: the ``state_dtype`` of each KDA verify call and its prework (``tkg_kda.KDA_STATE_DTYPES``);
            ``bf16`` (default) keeps the calls as they were.
        kda_paired_pages: with an fp32 ``kda_state_dtype``, the ``paired_pages`` of each KDA verify call and its
            prework (``tkg_kda.kda_page_span``): the KDA page ids name the even page of an aligned pair of the
            256-row pool (EVEN ids in [1, P - 1) only, ``tkg_kda.kda_page_bound``: an odd id or an id >= P - 1 is
            off). False (default) keeps the calls as they were.
        mla_k1_fused_masks: ``mla_mode="k1"``: one mask row for each query row of a key group (``mla_block_attention``
            ``fused_masks``), also at a B that does not divide 16 when the call is one pass (``fused_single_pass``). The
            loop then needs no per-row register store of the mask threshold, which runs on every engine. Bitwise the
            same; default off.
        mla_k1_weights_on_act: ``mla_mode="k1"`` on the loop path, not ``mla_cp``: the transposed weights of the prior
            loop copy out of PSUM on ScalarE (``mla_block_attention`` ``weights_on_act``). Bitwise the same; default off.
        mla_k1_row_pipeline: ``mla_mode="k1"``, not ``mla_cp``: the k1 call's prior-key loop runs its rows as a
            two-stage software pipeline (``mla_block_attention`` ``loop_row_pipeline``). Bitwise; default off.
        mla_k1_o_proj_ahead: ``mla_mode="k1"``, not ``mla_cp``: the k1 call loads its o_proj weights with the head
            weights and keeps them through the prior-key loop (``mla_block_attention`` ``o_proj_ahead``). Bitwise;
            default off.
        mla_loop_dr_keys: ``mla_mode="per_token"``: the per-token MLA calls take ``mla_decode_attention``
            ``loop_dr_keys`` (the Trn2 per-sequence loop path). Not bitwise; default off.
        mla_out_ar: ``mla_mode="k1"``, not with ``mla_out_ag``: the output sum of the call is one all-reduce in the
            HBM pair of ``mla_sources`` (``mla_block_attention`` ``out_ar_call``). Default off.
        mla_state_bf16_o: ``mla_mode="k1"``: the state reduce-scatter carries the numerators in BF16 and the sums in
            FP32 (``mla_block_attention`` ``state_bf16_o``); the decode twin takes ``mla_decode_attention``
            ``state_bf16_o``. It rounds the numerators. Default off.
        mla_split: None, or the long-sequence split of the k1 MLA call (``mla_block_attention`` ``split``): ``{"group":
            the pair ReplicaGroup, "info": the step's effective [1, 4] INT32 split info, "position": [B + 2, 1] INT32 the
            row-0 positions of the extended batch, "live": [B + 2, 1] INT32 or None}``. The block gives the MLA layers
            the extended table ``[n_mla, B + 2, NB]`` and slots ``[n_mla, Q, B + 2, 1]``; the MLA body runs the call on
            the extended batch (zero rows in the foreign and pad slots) and keeps the B request slots of its output.
    """
    check_kda_token_tiles(
        queries,
        kda_token_tiles,
        {
            "kda_conv_load_gather": kda_conv_load_gather,
            "kda_batched_conv_stores": kda_batched_conv_stores,
            "kda_slot_pages_contiguous": kda_slot_pages_contiguous,
            "kda_batched_slot_conv_stores": kda_batched_slot_conv_stores,
            "kda_bf16_update_rows": kda_bf16_update_rows,
        },
        kda_slot_conv_rows is not None,
    )
    kernel_assert((kda_page_groups is None) == (kda_page_sets is None), "kda_page_groups and kda_page_sets go together")
    # The verify KDA page tables (the block gets a _LayerIndex stand-in in their place): DMAs gather by them.
    for table_name, table in (("kda_state_page", kda_state_page), ("kda_record_page", kda_record_page)):
        kernel_assert(
            not hasattr(table, "dtype") or table.dtype == nl.int32,
            f"{table_name} must be INT32 page ids (DMAs gather by them and do not take a 64-bit dtype), "
            f"got {getattr(table, 'dtype', None)}",
        )
    kernel_assert(
        mla_loop_restart == "off"
        or (mla_mode == "per_token" and not mla_cp)
        or (mla_mode == "k1" and not mla_cp and mla_loop_restart in LOOP_TAILS),
        f"mla_loop_restart is a key of the per-token MLA calls, or with mla_mode k1 a tail form {tuple(LOOP_TAILS)[1:]} "
        f"of the k1 call, got {mla_loop_restart!r} with mla_mode {mla_mode!r}, mla_cp {mla_cp}",
    )
    kernel_assert(
        (not mla_loop_groups and not mla_o_proj_ring and not mla_loop_inputs_ahead and mla_wgate_early == "off")
        or (mla_mode == "per_token" and not mla_cp),
        f"mla_loop_groups, mla_o_proj_ring, mla_loop_inputs_ahead and mla_wgate_early are keys of the per-token MLA "
        f"calls, got mla_mode {mla_mode!r}, mla_cp {mla_cp}",
    )
    kernel_assert(
        not mla_k1_weights_on_act or (mla_mode == "k1" and not mla_cp),
        f"mla_k1_weights_on_act is a key of the k1 call, got mla_mode {mla_mode!r}, mla_cp {mla_cp}",
    )
    kernel_assert(
        not mla_k1_row_pipeline or (mla_mode == "k1" and not mla_cp),
        f"mla_k1_row_pipeline is a key of the k1 call, got mla_mode {mla_mode!r}, mla_cp {mla_cp}",
    )
    kernel_assert(
        not mla_k1_o_proj_ahead or (mla_mode == "k1" and not mla_cp),
        f"mla_k1_o_proj_ahead is a key of the k1 call, got mla_mode {mla_mode!r}, mla_cp {mla_cp}",
    )
    kernel_assert(
        mla_loop_page_gather == "off" or not mla_cp,
        f"mla_loop_page_gather is a key of the per-token MLA calls and of the k1 call, got mla_cp {mla_cp}",
    )
    kernel_assert(
        not mla_loop_dr_keys or (mla_mode == "per_token" and not mla_cp),
        f"mla_loop_dr_keys is a key of the per-token MLA calls, got mla_mode {mla_mode!r}, mla_cp {mla_cp}",
    )
    # The default eps and pages keep the calls of the MLA bodies as they were.
    eps_kw = {"lora_norm_eps": lora_norm_eps} if lora_norm_eps != LORA_NORM_EPS else {}
    if mla_zeroed_pages:
        eps_kw = {**eps_kw, "zeroed_pages": True}
    token_kw = {"loop_dr_keys": True} if mla_loop_dr_keys else {}

    def o_proj_stream(loop_path, rows):
        return (loop_path and rows > 16) if mla_o_proj_stream is None else mla_o_proj_stream

    k1_cast = mla_mode == "k1" and not mla_cp and mla_loop_cast_engines in ROWS_LOOP_CASTS
    kernel_assert(
        mla_loop_cast_engines == "off" or (mla_mode == "per_token" and not mla_cp) or k1_cast,
        "mla_loop_cast_engines is a key of the per-token MLA calls, or a page form (dma, dma6, dma4) of the k1 call, "
        f"got {mla_loop_cast_engines!r} with mla_mode {mla_mode!r}, mla_cp {mla_cp}",
    )
    k1_cast_kw = {"loop_cast_engines": mla_loop_cast_engines} if k1_cast and mla_loop_cast_engines != "off" else {}
    cast_kw = {"loop_cast_engines": mla_loop_cast_engines} if mla_loop_cast_engines != "off" and not k1_cast else {}
    kernel_assert(
        not mla_loop_pad_stationary or (mla_mode == "per_token" and not mla_cp),
        f"mla_loop_pad_stationary is a key of the per-token MLA calls, got mla_mode {mla_mode!r}, mla_cp {mla_cp}",
    )
    if mla_loop_pad_stationary:
        cast_kw["loop_pad_stationary"] = True

    def drained(y, kv_pool, tag):
        if not drain_writes:
            return y
        rows = kv_pool.reshape((kv_pool.shape[0] * P_MAX, kv_pool.shape[1] // P_MAX))
        bits = nl.ndarray((P_MAX, 2), dtype=rows.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=bits, src=rows[0:P_MAX, 0:2], name=f"{tag}_drain_read")
        # (bits & 0) + 1 is exactly 1 for any pool bytes, NaN included, so the multiply keeps every bit of y.
        zero_i = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=zero_i, data=bits.view(nl.int32), op0=nl.bitwise_and, operand0=0, engine=nisa.engine.vector
        )
        one = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=one, data=zero_i, op0=nl.add, operand0=1.0, engine=nisa.engine.vector)
        out = nl.ndarray(y.shape, dtype=y.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=out, data=y, op0=nl.multiply, operand0=one, engine=nisa.engine.vector)
        for name in ("side_effects", "side_effect_sources"):
            if hasattr(y, name):
                object.__setattr__(out, name, getattr(y, name))
        return out

    def kda_slot_options(kda_k):
        if kda_replay != "states":
            return {}
        return {
            "replay": "states",
            "state_slot_page": kda_state_slot_page[kda_k],
            "write_state_slots": kda_write_state_slots[kda_k],
        }

    # The default state dtype keeps the KDA calls as they were.
    state_kw = {} if kda_state_dtype == "bf16" else {"state_dtype": kda_state_dtype}
    if kda_paired_pages:
        state_kw["paired_pages"] = True

    # The persistent slot-state ring of the sequential states chain (nki-dma-source-liveness): the kernel allocates
    # it and shares the waits record across the KDA calls and the blocks, so the next call's write of a ring tile
    # waits for this call's stores. The batched chain takes a kernel ring of Q tiles; chunked keeps per-call tiles
    # (see kda_chain). Off a serving kernel (no ring) the body keeps the within-call edges.
    slot_state_kw = {}
    if kda_replay == "states" and kda_chain == "sequential":
        slot_state_kw["slot_state_depth"] = kda_slot_state_depth
    kernel_assert(
        not kda_shared_state_ring or (kda_replay == "states" and kda_chain == "sequential"),
        "kda_shared_state_ring needs kda_replay 'states' and kda_chain 'sequential'",
    )
    if kda_shared_state_ring:
        slot_state_kw["shared_state_ring"] = True
    if kda_slot_ring_per_call:
        kernel_assert(
            kda_replay == "states" and kda_chain == "sequential" and kda_token_tiles and kda_slot_state_ring is None,
            "kda_slot_ring_per_call needs kda_token_tiles on the sequential states chain and no kda_slot_state_ring",
        )
        slot_state_kw["slot_ring_per_call"] = True
        if kda_slot_ring_keep_sink is not None:
            slot_state_kw["slot_ring_keep_sink"] = kda_slot_ring_keep_sink
        if kda_slot_ring_keep_defer is not None:
            slot_state_kw["slot_ring_keep_defer"] = kda_slot_ring_keep_defer
    if kda_replay == "states" and kda_chain in ("sequential", "batched") and kda_slot_state_ring is not None:
        slot_state_kw["slot_state_ring"] = kda_slot_state_ring
        slot_state_kw["slot_state_waits"] = kda_slot_state_waits

    kernel_assert(
        not kda_prework_deferred or (kda_prework_ahead and kda_replay == "states"),
        f"kda_prework_deferred keeps the selects of the states prework: it needs kda_prework_ahead and kda_replay "
        f"'states', got {kda_prework_ahead}, {kda_replay!r}",
    )
    kernel_assert(
        not kda_replay_ahead or (kda_prework_ahead and kda_replay == "records" and kda_chain == "sequential"),
        "kda_replay_ahead replays the records in the KDA prework: it needs kda_prework_ahead, kda_replay='records' "
        f"and the sequential chain, got {kda_prework_ahead}, {kda_replay!r}, {kda_chain!r}",
    )

    def kda_page_group_kw(kda_k):
        # kda_page_groups: the page set of the layer's KV cache group, which the composed states prework fills or
        # reads (kda_verify_prework page_group), as the body does with the inline prework.
        if kda_page_groups is None or not kda_prework_deferred:
            return {}
        return {"page_group": kda_page_sets[kda_page_groups[kda_k]]}

    def kda_prework(kv_pool, kda_k, heads, tag, weights=None):
        replay_kw = {}
        if kda_replay_ahead:
            kernel_assert(weights is not None, "kda_replay_ahead needs the layer's (w_small, params) for its prework")
            replay_kw = {
                "replay_ahead": {
                    "kda_w_small": weights[0],
                    "kda_params": weights[1],
                    **({} if kda_conv_dtype == "bf16" else {"conv_dtype": kda_conv_dtype}),
                    **({"readout": kda_readout} if kda_readout is not None else {}),
                    **({"recurrence_group": kda_recurrence_group} if kda_recurrence_group is not None else {}),
                    **(
                        {"conv_products_engine": kda_conv_products_engine}
                        if kda_conv_products_engine != "gpsimd"
                        else {}
                    ),
                    **({"bf16_update_rows": True} if kda_bf16_update_rows else {}),
                    **({"update_bcast_depth": kda_update_bcast_depth} if kda_update_bcast_depth != 2 else {}),
                    **({"weight_dma": kda_weight_dma} if kda_weight_dma != "pool" else {}),
                }
            }
        return kda_verify_prework(
            kv_pool,
            kda_state_page[kda_k],
            kda_record_page[kda_k],
            kda_replay_len,
            kda_state_position,
            heads,
            queries,
            tag,
            **kda_slot_options(kda_k),
            **state_kw,
            **({"conv_load_gather": True} if kda_conv_load_gather else {}),
            **({"null_page": kda_null_page} if kda_null_page else {}),
            **({"slot_pages_contiguous": True} if kda_slot_pages_contiguous else {}),
            **replay_kw,
            **({"defer_selects": True} if kda_prework_deferred else {}),
            **({"merged_state_loads": True} if kda_prework_deferred and kda_merged_state_loads else {}),
            **kda_page_group_kw(kda_k),
        )

    # decode_block_body passes the layer's (w_small, params) to a prework that replays the records.
    kda_prework.takes_weights = kda_replay_ahead

    def kda_body(
        h,
        w_in,
        w_out,
        w_small,
        params,
        kv_pool,
        kda_k,
        position_rows,
        tp_group,
        tag,
        sources=None,
        prework=None,
        kda_w_in_scale=None,
        scatter_keep=None,
        w_in_ring=None,
        w_out_after_w_in=False,
        w_in_marks=None,
        kda_w_out_scale=None,
        small_after_result=False,
        in_proj_hook=None,
        token_hook=None,
        tail_hook=None,
        x_quant_hook=None,
        state_loads_after=(),
    ):
        # The explicit kda_ar pair wins; else take the persistent all-reduce pair that decode_block_body forwards.
        ar_hbm = None if kda_ar is None else kda_ar[kda_k]
        if ar_hbm is None and sources is not None and "kda_ar_hbm" in sources:
            ar_hbm = (sources["kda_ar_hbm"], sources["kda_ar_dst_hbm"])
        ar_sbuf = None if kda_ar_sbuf is None else kda_ar_sbuf[kda_k]
        y = kda_verify_attention(
            h,
            w_in,
            w_out,
            w_small,
            params,
            kv_pool,
            kda_state_page[kda_k],
            kda_record_page[kda_k],
            kda_replay_len,
            kda_state_position,
            tp_group,
            tag,
            ar_hbm=ar_hbm,
            **({"ar_sbuf": ar_sbuf} if ar_sbuf is not None else {}),
            prework=prework,
            chain=kda_chain,
            **({"out_rs_tp": TP_SIZE} if attn_rs_moe_input else {}),
            **kda_slot_options(kda_k),
            **({} if kda_conv_dtype == "bf16" else {"conv_dtype": kda_conv_dtype}),
            **({"token_tiles": True} if kda_token_tiles else {}),
            **({} if kda_slot_store_queues == 1 else {"slot_store_queues": kda_slot_store_queues}),
            **({} if kda_slot_store_engine == "sync" else {"slot_store_engine": kda_slot_store_engine}),
            **({"w_in_buffers": kda_w_in_buffers} if kda_w_in_buffers else {}),
            **({"readout": kda_readout} if kda_readout is not None else {}),
            **({"kda_w_in_scale": kda_w_in_scale} if kda_w_in_scale is not None else {}),
            **({"kda_w_out_scale": kda_w_out_scale} if kda_w_out_scale is not None else {}),
            **state_kw,
            **({} if scatter_keep is None else {"scatter_keep": scatter_keep}),
            **({"w_in_ring": w_in_ring} if w_in_ring is not None else {}),
            **({"w_out_after_w_in": True} if w_out_after_w_in else {}),
            **({"w_in_marks": w_in_marks} if w_in_marks is not None else {}),
            **({"state_loads_first": True} if kda_state_loads_first else {}),
            **({"slot_pages_first": True} if kda_slot_pages_first else {}),
            **({"conv_load_gather": True} if kda_conv_load_gather else {}),
            **({"batched_conv_stores": True} if kda_batched_conv_stores else {}),
            **({"null_page": kda_null_page} if kda_null_page else {}),
            **slot_state_kw,
            **({"slot_pages_contiguous": True} if kda_slot_pages_contiguous else {}),
            **({"update_bcast_depth": kda_update_bcast_depth} if kda_update_bcast_depth != 2 else {}),
            **({"head_pipeline": kda_head_pipeline} if kda_head_pipeline != "off" else {}),
            **({"out_ar_rows": True} if kda_out_ar_rows else {}),
            **({"in_proj_x_stationary": True} if kda_in_proj_x_stationary else {}),
            **({"state_loads_after_ring": True} if kda_state_loads_after_ring else {}),
            **({"early_out_tokens": kda_early_out_tokens} if kda_early_out_tokens is not None else {}),
            **({} if kda_in_proj_lanes_view else {"in_proj_lanes_view": False}),
            **({} if kda_states_after_gate_result else {"states_after_gate_result": False}),
            **({"batched_slot_conv_stores": True} if kda_batched_slot_conv_stores else {}),
            **({"bf16_update_rows": True} if kda_bf16_update_rows else {}),
            **({"bf16_cols_once": True} if kda_bf16_cols_once else {}),
            **({"commit_after_w_in": True} if kda_commit_after_w_in else {}),
            **({"slot_store_regs": True} if kda_slot_store_regs else {}),
            **({"slot_conv_after_ar": True} if kda_slot_conv_after_ar else {}),
            **({"weight_dma": kda_weight_dma} if kda_weight_dma != "pool" else {}),
            **({"consts": kda_consts} if kda_consts is not None else {}),
            **({"late_keep_warm": kda_late_keep_warm} if kda_late_keep_warm else {}),
            **({"slot_conv_rows": kda_slot_conv_rows} if kda_slot_conv_rows is not None else {}),
            **({"slot_conv_waits": kda_slot_conv_waits} if kda_slot_conv_waits is not None else {}),
            **({"slot_conv_from_inputs": True} if kda_slot_conv_from_inputs else {}),
            **({"conv_products_engine": kda_conv_products_engine} if kda_conv_products_engine != "gpsimd" else {}),
            **({"recurrence_group": kda_recurrence_group} if kda_recurrence_group is not None else {}),
            **({"w_in_psum_bufs": kda_w_in_psum_bufs} if kda_w_in_psum_bufs != 1 else {}),
            **({"w_in_refill_queues": kda_w_in_refill_queues} if kda_w_in_refill_queues != "alternate" else {}),
            **({"conv_zero_rows": kda_conv_zero_rows} if kda_conv_zero_rows is not None else {}),
            **({"page_group": kda_page_sets[kda_page_groups[kda_k]]} if kda_page_groups is not None else {}),
            **({"small_after_result": True} if small_after_result else {}),
            **({"in_proj_hook": in_proj_hook} if in_proj_hook is not None else {}),
            **({"token_hook": token_hook} if token_hook is not None else {}),
            **({"tail_hook": tail_hook} if tail_hook is not None else {}),
            **({"x_quant_hook": x_quant_hook} if x_quant_hook is not None else {}),
            # The instructions of the block before that the Sync-queue state loads follow (kda_verify_attention
            # state_loads_after); () keeps the program.
            **({"state_loads_after": tuple(state_loads_after)} if state_loads_after else {}),
            **({"merged_state_loads": True} if kda_merged_state_loads else {}),
            **({"w_out_after_w_in_fills": True} if kda_w_out_after_w_in_fills else {}),
        )
        return drained(y, kv_pool, tag)

    # decode_block_body kda_small_after_out_ar asks this body for small_after_result.
    kda_body.small_after_result = True
    # decode_block_body asks this body for state_loads_after (the Sync-queue state loads after the block before).
    kda_body.state_loads_after = True
    # decode_block_body moe_prefetch_kda_after_in_proj asks this body for in_proj_hook.
    kda_body.in_proj_hook = True
    # decode_block_body AHEAD_MLP_TOKEN_STEPS asks this body for token_hook (kda_token_hook False: the ahead part not
    # in the token chain).
    kda_body.token_hook = kda_token_hook
    # decode_block_body attn_res_ahead_mlp_kda_tail asks this body for tail_hook.
    kda_body.tail_hook = True
    # decode_block_body places the bank sums of a KDA layer's MLP mix after its input quantize (x_quant_hook;
    # kda_x_quant_hook False: the place without the hook).
    kda_body.x_quant_hook = kda_x_quant_hook
    mla_call_turns = itertools.count() if mla_calls is None else mla_calls

    def mla_body(
        h,
        *args,
        loop_path=False,
        sbuf=None,
        fp8_pages=False,
        sources=None,
        fused_masks=True,
        transposed_scores=True,
        mla_in_proj_scale=None,
        mla_q_b_proj_scale=None,
        mla_kv_b_proj_scale=None,
        mla_o_proj_scale=None,
        head_prefetch=None,
        state_rs_hook=None,
        out_ar_hook=None,
    ):
        weights, (kv_pool, table, slots, position_rows, rank_info, tp_group, tag) = args[:6], args[6:]
        mx_in = {"mla_in_proj_scale": mla_in_proj_scale} if mla_in_proj_scale is not None else {}
        if mla_q_b_proj_scale is not None:
            mx_in["mla_q_b_proj_scale"] = mla_q_b_proj_scale
        if mla_kv_b_proj_scale is not None:
            mx_in["mla_kv_b_proj_scale"] = mla_kv_b_proj_scale
        if mla_o_proj_scale is not None:
            mx_in["mla_o_proj_scale"] = mla_o_proj_scale
        batch = h.shape[1] // queries
        # head_prefetch: the block's head loads of this layer (decode_block_body mla_head_prefetch), for the first call.
        kernel_assert(
            head_prefetch is None or not (mla_cp or mla_mode == "k1"),
            "mla_head_prefetch feeds the first call of the per-token MLA verify form only",
        )
        # state_rs_hook (decode_block_body attn_res_mla_mlp_early / attn_res_mla_mlp_plain_early): the k1 call, not CP:
        # the rows form or the BF16-o state reduce-scatter (attn_res_mla_mlp_early), or the one-pass plain form of the
        # Trn2 verify (attn_res_mla_mlp_plain_early).
        kernel_assert(
            state_rs_hook is None or (mla_mode == "k1" and not mla_cp),
            "state_rs_hook takes the k1 MLA call without mla_cp (the rows form, the BF16-o state reduce-scatter or the "
            "one-pass plain form)",
        )
        # out_ar_hook (decode_block_body attn_res_mla_mlp_tail_entries): the k1 call with the output all-reduce, not CP.
        kernel_assert(
            out_ar_hook is None or (mla_mode == "k1" and not mla_cp and mla_out_ar),
            "out_ar_hook takes the k1 MLA call with the call-level output all-reduce (mla_out_ar), without mla_cp",
        )
        # The MLA DMA and GEMV levers (bitwise, default off); passed to every block form.
        glm_levers = {
            **({"critical_dma_priority": mla_critical_dma_priority} if mla_critical_dma_priority else {}),
            **({"cache_write_off_sp": True} if mla_cache_write_off_sp else {}),
            **({"gemv_bands": True} if mla_gemv_bands else {}),
            **({"batched_cache_write": True} if mla_batched_cache_write else {}),
            **({"own_rows_ahead": True} if mla_own_rows_ahead else {}),
        }
        if mla_cp:
            y = mla_block_attention(
                h,
                *weights,
                kv_pool,
                table,
                slots.reshape((queries * batch, 1)),
                position_rows[0],
                rank_info,
                tp_group,
                tag,
                local_heads=1,
                q_rows=queries,
                causal=True,
                output_gate=True,
                write_cache=True,
                loop_path=loop_path,
                fp8_pages=fp8_pages,
                sbuf=sbuf,
                bitwise=True,
                sources=mla_sources,
                cp=mla_cp,
                heads=CP_TARGET_HEADS,
                fused_masks=fused_masks,
                transposed_scores=transposed_scores,
                **({"own_key_select": True} if mla_cp_own_key_select else {}),
                **({"loop_first_static": True} if mla_loop_first_static else {}),
                **({"scores_ahead": True} if mla_scores_ahead else {}),
                **({"folded_reference": True} if mla_folded_reference else {}),
                **(
                    {"loop_p2": True}
                    # None (default): one or two sequences (B <= 2).
                    if (batch <= 2 if mla_cp_loop_p2 is None else mla_cp_loop_p2)
                    and loop_path
                    and mla_runs_loops(table.shape[1])
                    and queries * CP_TARGET_HEADS % P_MAX == 0
                    and queries * batch <= MLA_PASS_ROWS
                    # The fused masks of the transposed-scores form (a B that divides the 16 rows of a pass).
                    and fused_masks
                    and transposed_scores
                    and MLA_PASS_ROWS % batch == 0
                    else {}
                ),
                **dict(mla_cp_rows_options or {}),
                **(
                    {"loop_preload": True}
                    if mla_cp_loop_preload and mla_loop_first_static and loop_path and mla_runs_loops(table.shape[1])
                    else {}
                ),
                **glm_levers,
                **eps_kw,
                **mx_in,
            )
            return drained(y, kv_pool, tag)
        if mla_mode == "k1":
            split_kw = {}
            mla_position = position_rows[0]
            mla_live = live_rows
            h_call = h
            split_v5 = mla_split is not None and mla_split.get("form") == "v5"
            # The v5 split program's extended batch (B request slots, then the 2 foreign slots), as v1's B + 2.
            split_ext = split_v5 and mla_split.get("ext") is not None
            if split_v5:
                # The v5 split: the MLA call runs its DP collectives.
                split_kw = {"split": {k: v for k, v in mla_split.items() if k not in ("position", "live")}}
            if mla_split is not None and (not split_v5 or split_ext):
                # The split's extended batch: the B request slots, then the foreign and the pad slot (zero rows).
                sb_split = AUTO_SBUF if sbuf is None else sbuf
                ext = batch + (mla_split["foreign_slots"] if split_ext else 2)
                h_call = sb_split.tile((P_MAX, queries * ext, h.shape[2]), h.dtype)
                for j in range(queries):
                    nisa.tensor_copy(
                        dst=h_call[0:P_MAX, j * ext : j * ext + batch, 0 : h.shape[2]],
                        src=h[0:P_MAX, j * batch : (j + 1) * batch, 0 : h.shape[2]],
                    )
                    nisa.memset(dst=h_call[0:P_MAX, j * ext + batch : (j + 1) * ext, 0 : h.shape[2]], value=0.0)
                mla_position = mla_split["position"]
                mla_live = mla_split.get("live")
                if not split_v5:
                    split_kw = {"split": {"group": mla_split["group"], "info": mla_split["info"]}}
            # One bitwise multi-query MLA call on the Q*B rows, with one persistent Q*B-sized source set (mla_sources).
            call_sources = mla_sources
            if mla_sources is not None and "qkv_parity" in mla_sources:
                # The qkv all-gather pair of this call: the two sets alternate by MLA call. Thus the next MLA layer
                # writes the other source tile, and a peer that still reads this source reads the bytes of this layer.
                call_sources = {k: v for k, v in mla_sources.items() if k != "qkv_parity"}
                call_sources.update(mla_sources["qkv_parity"][next(mla_call_turns) % 2])
            y = mla_block_attention(
                h_call,
                *weights,
                kv_pool,
                table,
                slots.reshape((queries * (ext if mla_split is not None and (not split_v5 or split_ext) else batch), 1)),
                mla_position,
                rank_info,
                tp_group,
                tag,
                local_heads=_MLA_LOCAL_HEADS,
                q_rows=queries,
                causal=True,
                output_gate=True,
                write_cache=True,
                loop_path=loop_path,
                fp8_pages=fp8_pages,
                sbuf=sbuf,
                bitwise=True,
                sources=call_sources,
                persistent_rs=call_sources is not None and "rs" in call_sources,
                # The rows form of several sequences caps the loop groups too, so its group tiles fit (pairs stay aligned).
                loop_group_cap=_K1_LOOP_GROUPS
                if (loop_path and (queries * batch > 16 or (mla_k1_rows and batch > 1)))
                else 0,
                **({"q_hbm": True} if mla_q_hbm else {}),
                **({"out_ag_call": True} if mla_out_ag else {}),
                **({"out_ar_call": True} if mla_out_ar else {}),
                **({"state_bf16_o": True} if mla_state_bf16_o else {}),
                **({"fit_passes": True} if mla_k1_fit else {}),
                **({"fused_masks": True} if (mla_k1_fit or mla_k1_rows or mla_k1_fused_masks) else {}),
                **({"fused_single_pass": True} if mla_k1_fused_masks else {}),
                **({"transposed_scores": True} if mla_k1_rows else {}),
                **(dict(mla_k1_rows_options) if mla_k1_rows and mla_k1_rows_options else {}),
                **({"live_rows": mla_live} if mla_live is not None else {}),
                **({"late_tail_weights": True} if mla_late_tail_weights else {}),
                **({"pass_q_prefetch": mla_pass_q_prefetch} if mla_pass_q_prefetch else {}),
                **({"q_gather_buffers": mla_q_gather_buffers} if mla_q_gather_buffers != 1 else {}),
                **({"tail_weights_after_rs": True} if mla_tail_weights_after_rs else {}),
                **({"sums_rs_first": mla_sums_rs_first} if mla_sums_rs_first is not None else {}),
                **({"pass_rows_ahead": True} if mla_pass_rows_ahead else {}),
                **({"packed_prior": True} if mla_k1_packed else {}),
                **({"packed_prior_options": dict(mla_k1_pack_options)} if mla_k1_pack_options else {}),
                **({"loop_page_gather": mla_loop_page_gather} if mla_loop_page_gather != "off" else {}),
                **({"weights_on_act": True} if mla_k1_weights_on_act else {}),
                **({"loop_row_pipeline": True} if mla_k1_row_pipeline else {}),
                **({"o_proj_ahead": True} if mla_k1_o_proj_ahead else {}),
                **({"loop_tail_groups": LOOP_TAILS[mla_loop_restart]} if mla_loop_restart != "off" else {}),
                **(
                    {"pass_plans": mla_sources["pass_plans"]}
                    if mla_sources is not None and "pass_plans" in mla_sources
                    else {}
                ),
                **({"out_rs_tp": TP_SIZE} if attn_rs_moe_input else {}),
                **glm_levers,
                **({"late_o_proj_weights": True} if mla_late_o_proj_weights else {}),
                **({"late_tail_frames": True} if mla_late_tail_frames else {}),
                **({"state_rs_hook": state_rs_hook} if state_rs_hook is not None else {}),
                **({"out_ar_hook": out_ar_hook} if out_ar_hook is not None else {}),
                **k1_cast_kw,
                **eps_kw,
                **mx_in,
                **split_kw,
            )
            if mla_split is not None and not split_v5:
                # Keep the B request slots of the extended output (v1; the v5 extended batch returns them only).
                y_req = sb_split.tile(h.shape, h.dtype)
                for j in range(queries):
                    nisa.tensor_copy(
                        dst=y_req[0:P_MAX, j * batch : (j + 1) * batch, 0 : h.shape[2]],
                        src=y[0:P_MAX, j * ext : j * ext + batch, 0 : h.shape[2]],
                    )
                y = y_req
            return drained(y, kv_pool, tag)
        sb = AUTO_SBUF if sbuf is None else sbuf
        out = sb.tile(h.shape, h.dtype)
        effects = []
        prev = None
        # Each query's rows run in sub-batches of at most QUERY_SET_TOKENS, so one MLA call never exceeds the
        # B=4 frame; at B=8 this stops the per-token MLA from spilling the collective destination arena at 93
        # layers (a reload DMA of a spilled arena shares bytes with a peer's gather output). B <= 4 keeps one
        # sub-batch per query, so the frame and the call count match the validated path.
        set_rows = min(batch, QUERY_SET_TOKENS)
        n_sub = -(-batch // set_rows)
        call = 0
        for j in range(queries):
            for sub in range(n_sub):
                r0 = sub * set_rows
                cnt = min(set_rows, batch - r0)
                base = j * batch + r0
                with nullcontext() if sbuf is None else sbuf.scope(f"{tag}_q{j}_{sub}"):
                    x = _row_tile(h, base, cnt, sb)
                    if serial and prev is not None:
                        x = _after(x, out[0:P_MAX, prev[0] : prev[0] + prev[1], 0 : h.shape[2]], sb)
                    # A parity set (allocated once before the layer loop) gives this sub-batch its own
                    # all-gather source bytes, round-robin over the sub-calls so the next sub-batch does not
                    # take them after only the local completion. On the loop path mla_sources is None: each
                    # call self-allocates in its scope, which the scope frees between sub-batches.
                    token_sources = mla_sources[call % len(mla_sources)] if mla_sources is not None else None
                    # De-parity: the call takes its own persistent destinations, so no later query reuses them early.
                    if mla_state:
                        token_sources = {**(token_sources or {}), **mla_state[call]}
                    y = mla_decode_attention(
                        x,
                        *weights,
                        kv_pool,
                        table[r0 : r0 + cnt],
                        slots[j][r0 : r0 + cnt],
                        position_rows[j][r0 : r0 + cnt],
                        rank_info,
                        tp_group,
                        f"{tag}_q{j}_{sub}",
                        loop_path=loop_path,
                        fp8_pages=fp8_pages,
                        **cast_kw,
                        sbuf=sbuf,
                        sources=token_sources,
                        **eps_kw,
                        **token_kw,
                        **mx_in,
                        o_proj_stream=o_proj_stream(loop_path, h.shape[1]),
                        **({"head_prefetch": head_prefetch} if call == 0 and head_prefetch is not None else {}),
                        **({"loop_restart": mla_loop_restart} if mla_loop_restart != "off" else {}),
                        **({"loop_groups": mla_loop_groups} if mla_loop_groups else {}),
                        **(
                            {"o_proj_ring": mla_o_proj_ring, **({"o_proj_early": True} if mla_o_proj_early else {})}
                            if mla_o_proj_ring and o_proj_stream(loop_path, h.shape[1])
                            else {}
                        ),
                        **({"loop_page_gather": mla_loop_page_gather} if mla_loop_page_gather != "off" else {}),
                        **({"loop_inputs_ahead": True} if mla_loop_inputs_ahead else {}),
                        **({"wgate_early": mla_wgate_early} if mla_wgate_early != "off" else {}),
                    )
                    nisa.tensor_copy(dst=out[0:P_MAX, base : base + cnt, 0 : h.shape[2]], src=y)
                    effects.extend(getattr(y, "side_effects", ()))
                    prev = (base, cnt)
                    call += 1
        object.__setattr__(out, "side_effects", tuple(effects))
        return drained(out, kv_pool, tag)

    calls = itertools.count() if ffn_calls is None else ffn_calls

    def ffn(body, extra_options=None):
        def run(h, *args, prefetch=None, **options):
            weights, (rank_info, dp_group, world_group, tag) = args[:-4], args[-4:]
            rows = h.shape[1]
            chunk = max(1, min(rows, ffn_rows))
            chunks = [(start, min(chunk, rows - start)) for start in range(0, rows, chunk)]
            opts = {**options, **(extra_options or {})}
            # The padding-row input (decode_block_body moe_pad_rows) goes to each chunk as the rows of that chunk.
            pad_rows = opts.pop("pad_rows", None)
            pads = [
                {} if pad_rows is None else {"pad_rows": pad_rows[start : start + count]} for start, count in chunks
            ]
            # The FP16 router copy of the layer input: each chunk takes the rows of its own input view.
            x16 = opts.pop("x16_hT", None)

            def chunk_x16(idx):
                if x16 is None:
                    return {}
                if len(chunks) == 1:
                    return {"x16_hT": x16}
                start, count = chunks[idx]
                return {"x16_hT": x16[0:P_MAX, start : start + count, 0 : x16.shape[2]]}

            # A prefetched expert ring (decode_block_body moe_prefetch) feeds the one chunk of the layer.
            kernel_assert(prefetch is None or len(chunks) == 1, "A prefetched expert ring feeds one feed-forward chunk")
            if prefetch is not None:
                opts["prefetch"] = prefetch
            # The feed-forward chunks read the chunk-sized ffn_sources: Q*B block rows overflow 128 partitions at B=8.
            sets = [ffn_sources] * len(chunks)
            if ffn_sources is not None and "pingpong" in ffn_sources:
                sets = [layer_sources(ffn_sources, next(calls)) for _ in chunks]
            sets = [_chunk_sources(opts.get("sources") if s is None else s, i) for i, s in enumerate(sets)]
            # The chunks gather from h, the layer-parity FFN input: its next writer waits for the next layer's gathers.
            # The chunks of the layer share one expert-loop scope: it counts the two-buffer static calls.
            with expert_loop_layer():
                return _in_chunks(
                    lambda x, idx: body(
                        x,
                        *weights,
                        rank_info,
                        dp_group,
                        world_group,
                        f"{tag}_c{idx}",
                        **_with_sources(opts, sets[idx]),
                        **pads[idx],
                        **chunk_x16(idx),
                    ),
                    h,
                    chunks,
                    serial,
                    sbuf,
                )

        run.token_major = True
        # decode_block_body takes the DP4 x TP32 form (its sources and SBUF plan) from this attribute.
        run.moe_sharding = getattr(body, "moe_sharding", "tp128")
        # The body traces a caller's work inside its output collective (``cc_window``), as the decode body does.
        run.cc_window = getattr(body, "cc_window", False)
        run.takes_pad_rows = getattr(body, "takes_pad_rows", False)
        # The routers that a DP body takes itself (the 2-server body: router_dtypes).
        if hasattr(body, "router_dtypes"):
            run.router_dtypes = body.router_dtypes
        # A body that takes the MXFP8 in- or out-projection (the 2-server body: in_proj_mx, out_proj_mx) gets its
        # scales through the options.
        run.in_proj_mx = getattr(body, "in_proj_mx", False)
        run.out_proj_mx = getattr(body, "out_proj_mx", False)
        # A body that reads the hot-expert tiles of moe_hot_prefetch gets them through the options.
        run.hot_experts = getattr(body, "hot_experts", False)
        # The 2-server body's glue key router_norm_one_mul acts in the norm before the body (decode_block_body).
        run.router_norm_one_mul = getattr(body, "router_norm_one_mul", False)
        # A per-kind pe_keep_warm of the 2-server body: decode_block_body passes keep_warm_lead through the options.
        run.keep_warm_by_lead = getattr(body, "keep_warm_by_lead", False)
        if getattr(body, "prefetch", None) is not None:
            run.prefetch = body.prefetch
        return run

    kda_body.token_major = True
    kda_body.prework = kda_prework if kda_prework_ahead else None
    # decode_block_body passes scatter_keep to this body (kda_read_scatter_keep); its conv-state sources are per-layer.
    kda_body.conv_keep = True
    mla_body.token_major = True
    # decode_block_body attn_res_mla_mlp_early (the k1 rows form, or the BF16-o state reduce-scatter) and
    # attn_res_mla_mlp_plain_early (the one-pass k1 call) pass state_rs_hook to this body.
    mla_body.state_rs_hook = mla_mode == "k1" and not mla_cp
    # With the BF16-o state reduce-scatter and without the rows form the hook's bank-entry loads take the SP queue:
    # after that reduce-scatter the ScalarE queue issues the MLA tail weights (tail_weights_after_rs), which pace the
    # output projection.
    mla_body.state_rs_hook_dma = "sp" if mla_body.state_rs_hook and mla_state_bf16_o and not mla_k1_rows else None
    # decode_block_body attn_res_mla_mlp_tail_entries passes out_ar_hook to this body: the k1 call with the call-level
    # output all-reduce.
    mla_body.out_ar_hook = mla_body.state_rs_hook and bool(mla_out_ar)
    moe_options = None
    if weight_buffers is not None:
        moe_options = {"weight_buffers": weight_buffers}
        if half_weight_buffers:
            moe_options["half_weight_buffers"] = 2 * weight_buffers
    if gate_up_half_major:
        moe_options = {**(moe_options or {}), "gate_up_half_major": True}
    moe = ffn(moe_body, moe_options)
    if getattr(moe_body, "a2av_expert_form", None):
        # The all-to-all-v body runs one call on all the rows of the step (its gain grows with the rows), not the
        # feed-forward chunks.
        def moe(h, *args, **options):
            with expert_loop_layer():
                return moe_body(h, *args, **{**options, **(moe_options or {})})

        moe.token_major = True
        moe.moe_sharding = moe_body.moe_sharding
        moe.a2av_expert_form = moe_body.a2av_expert_form
    return kda_body, mla_body, moe, ffn(dense_body)


@nki.jit
def kimi_k3_verify_model(
    stream: nl.NkiTensor,
    norm_weights: nl.NkiTensor,
    attn_res_weights: nl.NkiTensor,
    kda_w_in: nl.NkiTensor,
    kda_w_out: nl.NkiTensor,
    kda_w_small: nl.NkiTensor,
    kda_params: nl.NkiTensor,
    mla_in_proj: nl.NkiTensor,
    mla_q_a_norm: nl.NkiTensor,
    mla_q_b_proj: nl.NkiTensor,
    mla_kv_a_norm: nl.NkiTensor,
    mla_kv_b_proj: nl.NkiTensor,
    mla_o_proj: nl.NkiTensor,
    moe_in_proj: nl.NkiTensor,
    moe_out_proj: nl.NkiTensor,
    moe_vectors: nl.NkiTensor,
    expert_gate_up_weights: nl.NkiTensor,
    expert_gate_up_weights_scale: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_down_weights_scale: nl.NkiTensor,
    dense_gate_up: nl.NkiTensor,
    dense_down: nl.NkiTensor,
    kda_state_page: nl.NkiTensor,
    kda_record_page: nl.NkiTensor,
    kda_replay_len: nl.NkiTensor,
    kda_state_position: nl.NkiTensor,
    mla_block_table: nl.NkiTensor,
    mla_slot: nl.NkiTensor,
    position_rows: nl.NkiTensor,
    rank_info: nl.NkiTensor,
    tp_group: ReplicaGroup,
    dp_group: ReplicaGroup,
    world_group: ReplicaGroup,
    first_layer: int,
    eps: float,
    queries: int = DEFAULT_QUERIES,
    debug: bool = False,
    dma_gates: bool = True,
    pool_of_layer: tuple = None,
    mla_loop_path: bool = False,
    kv_pool_0: nl.NkiTensor = None,
    kv_pool_1: nl.NkiTensor = None,
    kv_pool_2: nl.NkiTensor = None,
    kv_pool_3: nl.NkiTensor = None,
    kv_pool_4: nl.NkiTensor = None,
    kv_pool_5: nl.NkiTensor = None,
    kv_pool_6: nl.NkiTensor = None,
    kv_pool_7: nl.NkiTensor = None,
    kv_pool_8: nl.NkiTensor = None,
    kv_pool_9: nl.NkiTensor = None,
    kv_pool_10: nl.NkiTensor = None,
    kv_pool_11: nl.NkiTensor = None,
    kv_pool_12: nl.NkiTensor = None,
    kv_pool_13: nl.NkiTensor = None,
    kv_pool_14: nl.NkiTensor = None,
    kv_pool_15: nl.NkiTensor = None,
    kv_pool_16: nl.NkiTensor = None,
    kv_pool_17: nl.NkiTensor = None,
    kv_pool_18: nl.NkiTensor = None,
    kv_pool_19: nl.NkiTensor = None,
    kv_pool_20: nl.NkiTensor = None,
    kv_pool_21: nl.NkiTensor = None,
    kv_pool_22: nl.NkiTensor = None,
    kv_pool_23: nl.NkiTensor = None,
    final_attn_res_weight: nl.NkiTensor = None,
    final_norm_weight: nl.NkiTensor = None,
    lm_head_weight: nl.NkiTensor = None,
    greedy: bool = False,
    return_logits: bool = False,
    return_aux: bool = False,
    mla_fp8_pages: bool = False,
    aux_layers: tuple = AUX_LAYERS,
    aux_tap: str = "mixture",
    ffn_rows: int = FFN_CHUNK_ROWS,
    mla_mode: str = "per_token",
    lora_norm_eps: float = LORA_NORM_EPS,
    cc_source_pingpong: bool = True,
    moe_active_expert_slots: int = 0,
    mla_cp: int = 0,
    dp_size: int = DP_SIZE,
    mla_options: dict = None,
    routed_kernel: str = "static",
    routed_reduce_dtype: str = "bf16",
    router_mode: str = None,
    router_dtype: str = "bf16",
    moe_router_weight: nl.NkiTensor = None,
    affinity_scaling: str = "pre_quant",
    affinity_hi_lo: bool = True,
    moe_sel_load_options: tuple = (),
    mix_cache: bool = False,
    early_ffn_gate: bool = None,
    kda_replay: str = "records",
    kda_state_slot_page: nl.NkiTensor = None,
    kda_write_state_slots: nl.NkiTensor = None,
    kda_prework_ahead: bool = False,
    moe_sbuf_reduces: tuple = (),
    kda_conv_dtype: str = "bf16",
    kda_chain: str = "sequential",
    kda_token_tiles: bool = False,
    kda_slot_store_queues: int = 1,
    kda_slot_store_engine: str = "sync",
    kda_slot_state_depth: int = 2,
    kda_w_in_buffers: int = 0,
    kda_conv_load_gather: bool = False,
    kda_batched_conv_stores: bool = False,
    kda_null_page: int = 0,
    kda_w_in_prefetch: str = "off",
    kda_w_in_prefetch_chunks: int = 0,
    moe_half_weight_buffers: bool = False,
    moe_gate_up_half_major: bool = False,
    mla_zeroed_pages: bool = False,
    moe_k1_two_weight_buffers: bool = False,
    shared_sharding: str = "tp128",
    latent_sharding: str = "tp128",
    moe_routed_exchange: str = "all_reduce",
    ffn_weights_after_attn: bool = False,
    mla_late_tail_weights: bool = False,
    mla_pass_q_prefetch: int = 0,
    mla_q_gather_buffers: int = 1,
    mla_tail_weights_after_rs: bool = False,
    mla_pass_rows_ahead: bool = False,
    mla_k1_rows: bool = False,
    mla_k1_rows_options: dict = None,
    routing_form: str = None,
    moe_dst_pingpong: tuple = (),
    attn_rs_moe_input: bool = False,
    moe_out_proj_form: str = "reduced",
    moe_latent_load: str = "strided",
    moe_routed_store_form: str = "relayout",
    moe_glue_dmas: tuple = (),
    moe_direct_latent: bool = False,
    moe_router_rows_drop_selection: bool = False,
    moe_latent_gather_mx: bool = False,
    attn_res_bank_in: nl.NkiTensor = None,
    return_attn_res_bank: bool = False,
    attn_res_fence_in: nl.NkiTensor = None,
    attn_res_cached_bank_sums: bool = False,
    mla_k1_packed: bool = False,
    mla_k1_pack_options: dict = None,
    mla_cp_own_key_select: bool = False,
    mla_loop_first_static: bool = False,
    mla_scores_ahead: bool = False,
    mla_folded_reference: bool = False,
    mla_cp_loop_p2: bool | None = None,
    mla_cp_rows_options: dict | None = None,
    mla_cp_loop_preload: bool = False,
    mla_critical_dma_priority: bool | str = False,
    mla_cache_write_off_sp: bool = False,
    mla_gemv_bands: bool = False,
    attn_res_bank_dma: str = None,
    norm_one_mul: bool = False,
    attn_res_mlp_early_entries: int = 0,
    kda_state_dtype: str = "bf16",
    kda_w_in_scale: nl.NkiTensor = None,
    mla_in_proj_scale: nl.NkiTensor = None,
    mla_q_b_proj_scale: nl.NkiTensor = None,
    mla_kv_b_proj_scale: nl.NkiTensor = None,
    mla_o_proj_scale: nl.NkiTensor = None,
    moe_in_proj_scale: nl.NkiTensor = None,
    kda_w_out_scale: nl.NkiTensor = None,
    moe_x_gather_hbm: bool = False,
    attn_res_sbuf_entries: int = 0,
    moe_x_world_gather: bool = False,
    kda_readout: str = None,
    mla_state_rows: int = 0,
    pair_shared_cc_dsts: tuple = (),
    kda_conv_keep: bool = False,
    moe_static_alt_tiles: bool | str = False,
    moe_pad_rows: nl.NkiTensor = None,
    attn_res_tensor_sums: bool = False,
    attn_res_one_round_norm: bool = False,
    norm_scalar_squares: int = 0,
    moe_preamble_buffers: int = 0,
    moe_hot_prefetch: dict = None,
    moe_preamble_reuse: bool = False,
    moe_x_dst_pair: bool | None = None,
    moe_weight_buffers: int | None = None,
    attn_res_early_sums: bool | None = None,
    kda_state_loads_first: bool = False,
    kda_slot_pages_first: bool = False,
    kda_out_ar_sbuf: bool | None = None,
    kda_slot_pages_contiguous: bool = False,
    kda_batched_slot_conv_stores: bool = False,
    kda_bf16_update_rows: bool = False,
):
    """Run the Kimi-K3 decoder layers of a range on Q tokens of each sequence of one DP replica.

    The arguments are those of ``decode_model.kimi_k3_decode_model`` with these changes (see the module
    docstring for the semantics).

    Dimensions:
        Q: tokens of each sequence (``queries``), 1 to 9. B: sequences, 1 to 4. R = Q * B rows.

    Args:
        stream: ``[R, H]`` bf16 embeddings; row ``j * B + b`` is token ``j`` of sequence ``b``.
        kda_state_page: ``[n_kda, B, 1]`` int32 page of the committed KDA state of each layer.
        kda_record_page: ``[n_kda, B, 1]`` int32 page of the replay records of each layer.
        kda_replay_len: ``[B, 1]`` int32 tokens of the previous step to replay (its accepted rows; 0 on
            the first step of a request).
        kda_state_position: ``[B, 1]`` int32 position of the committed state (0: no state).
        mla_block_table: ``[n_mla, B, NB]`` int32 local block table.
        mla_slot: ``[n_mla, Q, B, 1]`` int32 slot of the row of each token on its owner rank, else 0.
        position_rows: ``[Q, B, 1]`` int32 position of each token.
        queries: Q.
        mla_loop_path: let the MLA layers take the loop path (more than 16 local blocks), with explicit
            SBUF addresses as ``kimi_k3_decode_model`` does.
        return_aux: also return the aux state of each layer of ``aux_layers`` in the range,
            ``[len(aux_layers), R, H]`` (rows of the layers outside the range stay unwritten).
        aux_layers: static target layers of the aux states (``AUX_LAYERS``, the drafter input).
        aux_tap: ``"mixture"`` stores the pre-norm attention mix of layer ``i + 1`` for aux layer ``i``;
            ``"prefix"`` stores the stream after layer ``i``.
        ffn_rows: rows of each replica in one feed-forward call.
        lora_norm_eps: the epsilon of the MLA q_a and kv_a RMSNorms, as ``kimi_k3_decode_model``.
        cc_source_pingpong: two sets of the DP4 all-gather sources (see ``decode_block.kimi_k3_decode_block``).
        moe_active_expert_slots: K, the active-expert slots of the MoE layers, for the rows of one feed-forward
            call (see ``decode_block.decode_block_body``: the dynamic loop takes precedence where it applies, and an
            unsupported combination fails). 0 (default) runs every expert.
        mla_cp: 0, or 128 for the CP128 MLA layers (one bitwise ``mla_block_attention`` call on the Q*B rows; see
            ``verify_bodies``). ``tp_group`` is then the world group.
        mix_cache: keep the rstd of each AttnRes bank entry from its push (``decode_block.decode_block_body``). Always on
            with ``dp_size`` 1.
        early_ffn_gate: start the weights of each attention during the MoE tail (``decode_block.decode_block_body``).
            None turns it on with ``dp_size`` 1.
        moe_sel_load_options: ``sel_load``: (name, value) pairs of extra ``moe_sel_load_tp_start`` options
            (``decode_block.decode_block_body``). The values do not change; () (default) keeps the program.
        kda_replay: ``records`` (default) or ``states`` (``verify_kda.kda_verify_attention`` ``replay``).
        kda_state_slot_page, kda_write_state_slots: ``states``: ``[n_kda, B, 1]`` and ``[n_kda, B, Q]`` int32,
            the ``state_slot_page`` and ``write_state_slots`` of each KDA layer.
        kda_prework_ahead: trace the page metadata and the state loads of each KDA layer beside the output
            all-reduce of the feed-forward body before it (``verify_bodies``). The values do not change.
        kda_chain: form of the KDA scoring chain (``verify_kda.KDA_CHAINS``): ``sequential`` (default), ``chunked``
            (one head per rank) or ``batched`` (any head count and batch; it also solves the records replay); the last
            two are not bitwise equal to the decode steps. The ``states`` slot-state ring is for the ``sequential``
            chain only: ``batched`` and ``chunked`` write the Q slot states from per-call tiles that a later
            instruction can reuse before the slot stores finish (no ring), so they are not for serving until they take
            the ring and its edges (both are default-off).
        kda_token_tiles: ``states``: each scored token of every KDA layer takes its own state tiles at any local head
            count (``verify_kda.kda_verify_attention`` ``token_tiles``). The values do not change; False (default)
            keeps the program.
        kda_slot_state_depth: tiles of the ``states`` ``sequential`` slot-state ring (``verify_kda.kda_verify_attention``
            ``slot_state_depth``): 2 (default) alternates them so the chain pipelines, 1 is the in-place FIT fallback.
            The values do not change.
        kda_w_in_scale: None (default, a BF16 ``kda_w_in``), or the scales of an MXFP8 ``kda_w_in`` stack (the
            ``kda_w_in_scale`` of ``decode_model.kimi_k3_decode_model``, the same stacking): the KDA verify input
            projection runs MXFP8 x MXFP8 with the hidden quantized on VectorE. The values change.
        mla_in_proj_scale: None (default, a BF16 ``mla_in_proj``), or the scales of an MXFP8 ``mla_in_proj`` stack
            (the ``mla_in_proj_scale`` of ``decode_model.kimi_k3_decode_model``, the same stacking): the MLA q_a / kv_a
            and gate projections of every MLA form run MXFP8 x MXFP8 (``tkg_mla_block.mla_block_attention`` for the k1
            and CP forms, ``tkg_mla_paged.mla_decode_attention`` per token), as in the decode. The values change.
        moe_in_proj_scale: None (default, a BF16 ``moe_in_proj``), or the scales of an MXFP8 MoE in-projection (the
            ``moe_in_proj_scale`` of ``decode_model.kimi_k3_decode_model``, the same stacking): the MoE in-projection of
            every MoE layer runs MXFP8 x MXFP8, as in the decode. It needs an FP16 router. The values change.
        kda_w_out_scale: None (default, a BF16 ``kda_w_out``), or the scales of an MXFP8 padded-lane ``kda_w_out`` stack
            (``tkg_kda.kda_w_out_form``: ``kda_w_out`` then stacks ``[n, 1, 128, 4 H]`` uint8 and the scales
            ``[n, 128, 1, H]``): the KDA output projection of every KDA layer runs MXFP8 x MXFP8. The values change.
        kda_slot_store_queues: ``states``: 2 alternates the per-token slot state stores between the Sync and the
            ScalarE queues (``verify_kda.kda_verify_attention`` ``slot_store_queues``). The values do not change; 1
            (default) keeps the program.
        kda_slot_store_engine: ``states``: the queue of the per-token slot state stores
            (``verify_bodies`` ``kda_slot_store_engine``). "sync" (default) keeps the program; the bits do not change.
        kda_w_in_buffers: SBUF buffers of the ``kda_w_in`` ring of every KDA verify layer (``verify_kda``
            ``w_in_buffers``, as ``decode_model`` ``kda_w_in_buffers``). The values do not change; 0 (default) keeps the
            program.
        kda_conv_load_gather: load the committed conv state of each KDA verify layer with one gather DMA per group of
            sequences (``verify_kda.kda_verify_attention`` ``conv_load_gather``, as ``decode_model``
            ``kda_conv_load_gather``). The values do not change; False (default) keeps the program.
        kda_batched_conv_stores: write the committed conv state of each KDA verify layer with one scatter DMA per
            group of sequences (``verify_kda.kda_verify_attention`` ``batched_conv_stores``, as ``decode_model``
            ``kda_batched_conv_stores``). The values do not change; False (default) keeps the program.
        kda_null_page: 0 (default), or the KDA null page of each KDA verify call (``verify_kda.kda_verify_attention``
            ``null_page``, as ``decode_model`` ``kda_null_page``). The rows of the other pages do not change.
        kda_slot_pages_contiguous: ``kda_replay`` ``states``: each KDA verify layer loads the slot ids of
            ``kda_write_state_slots`` in their HBM order, one contiguous run per partition, and transposes them in the
            VectorE cast (``verify_bodies`` ``kda_slot_pages_contiguous``), in place of a transposed broadcast of 4-byte
            elements. The bits do not change; False (default) keeps the program.
        kda_batched_slot_conv_stores: ``kda_replay`` ``states``: each KDA verify layer writes the slot conv windows of
            its scored tokens with one scatter DMA per group of rows, not one serial Sync DMA per row
            (``verify_bodies`` ``kda_batched_slot_conv_stores``). The rows of every slot page do not change; an off
            slot's page 0 (or the null page) gets zeros. False (default) keeps the program.
        kda_bf16_update_rows: with the ``decoupled_bf16`` KDA readout, the state update of each scored token of every
            KDA verify layer takes its bf16 high and low rows from one bf16 TensorE transpose per 4 sequences and adds
            them in one broadcast matmul per head, in place of one fp32 identity matmul per sequence, its ScalarE and
            VectorE split, and two broadcast matmuls per head (``verify_bodies`` ``kda_bf16_update_rows``, as
            ``decode_model`` ``kda_bf16_update_rows``). The bits do not change; False (default) keeps the program.
        kda_w_in_prefetch, kda_w_in_prefetch_chunks: where each feed-forward layer before a KDA layer issues the first
            fills of that layer's ``kda_w_in`` ring (and the MXFP8 scales), and how many chunks from the ring's end
            (``decode_block.trn3_kda_w_in_prefetch``, 0: all). ``off`` (default) keeps the program; ``after_ffn_cc``
            issues them after the body's first output collective at DMA QoS level 3, and the KDA layer takes the ring
            (``w_in_ring``). The bytes, the buffers and the matmuls do not change, so the values do not change.
        moe_sbuf_reduces: the MoE all-reduces that run in SBUF on the MLA loop path (``decode_block.MOE_SBUF_REDUCES``):
            ``("routed_tp128",)`` runs the routed all-reduce of the MoE option bodies (sel_load) in SBUF, from a
            persistent source into a persistent destination (``decode_block.collective_sources``
            ``sbuf_routed_options``). ``"routed"`` does the same for the base MoE body (the static expert loop,
            ``sbuf_routed_reduce``), as in the decode model, with one feed-forward chunk: the routed store and load
            DMAs go away, so the all-reduce does not wait behind the MLA page gathers on the DMA queues. The output
            does not change.
            () (default) keeps the HBM all-reduces.
        kda_conv_dtype: form of the KDA short convolution, ``bf16`` (default) or ``fp32`` (``tkg_kda.CONV_DTYPES``).
            ``fp32`` needs the ``kda_params`` of ``build_kda_rank_weights(..., conv_dtype="fp32")``.
        moe_half_weight_buffers: where the verify trims the static expert loop to 1 or 2 weight buffers (more than 8
            rows), stream half projections through twice as many half-size buffers (``verify_bodies``; the same
            SBUF). The values do not change; False (default) keeps the program.
        moe_gate_up_half_major: the EP128 static gate/up weights are stored half-major (``verify_bodies``). The
            values do not change; False (default) reads the standard layout.
        mla_zeroed_pages: the MLA loop paths read their cache pages with no clamp. Set it only when the caller zeroes each
            MLA page at allocation, as for the decode (``mla_zeroed_pages`` of ``decode_model.kimi_k3_decode_model``).
        moe_k1_two_weight_buffers: a k1 step with its AttnRes bank in HBM keeps two expert-loop weight buffers on the
            loop path (``loop_weight_buffers``), in place of one above 32 rows. The values do not change; False
            (default) keeps the program.
        shared_sharding, latent_sharding: the sharding of the MoE projections, as ``decode_model.kimi_k3_decode_model``:
            both ``"tp128"`` (``latent_moe_decode``, the default) or both ``"dp4xtp32"`` (``tkg_moe_dp``; the ``moe_*``
            weights then have its layout). The verify must run the MoE body of the decode, so that each row is bitwise
            equal to the decode step of its token. The DP4 x TP32 body runs each feed-forward chunk of ``ffn_rows``
            rows (at most ``tkg_moe_dp.MAX_DP_ROWS``) with the static expert loop and the default MoE options.
        moe_routed_exchange: the routed exchange of the MoE layers, as ``decode_model.kimi_k3_decode_model``. With
            ``a2av`` the MoE body runs one call on all the rows of the step (not the feed-forward chunks). The decode
            NEFF of a config must take the same exchange.
        ffn_weights_after_attn: the weight loads of each feed-forward body start after the residual add that reads the
            attention output, not during the flight of the attention output collective (``decode_block.decode_block_body``;
            with ``dma_gates`` on). The values do not change. False (default) keeps the program.
        mla_late_tail_weights: ``mla_mode="k1"`` on the loop path: the MLA output gate and o_proj weights load
            after the prior-key loop (``late_tail_weights`` of ``mla_block_attention``), so the loop has room for its
            key buffers next to a routed MoE body. The outputs do not change. Default off.
        mla_pass_q_prefetch: ``mla_mode="k1"`` with the call query gather: the first gather tiles of each later MLA
            pass load before the loop of the pass before (``pass_q_prefetch`` of ``mla_block_attention``). The outputs
            do not change. 0 (default) keeps the program.
        mla_q_gather_buffers: ``mla_mode="k1"`` with the call query gather: 2 gives the query gather tiles of a pass two
            stack tiles in turn (``q_gather_buffers`` of ``mla_block_attention``). The outputs do not change. 1
            (default) keeps the program.
        mla_tail_weights_after_rs: ``mla_mode="k1"`` with the call-level BF16-o state reduce-scatter: the tail
            weights wait for it and load on the ScalarE queue (``tail_weights_after_rs`` of ``mla_block_attention``).
            The outputs do not change. False (default) keeps the program.
        mla_pass_rows_ahead: ``mla_mode="k1"`` with the packed p2 loop and the call query gather: the next MLA pass's
            own-key current rows and query gather tiles load before the state stores of the pass before
            (``pass_rows_ahead`` of ``mla_block_attention``). The outputs do not change. Default off.
        mla_k1_rows: ``mla_mode="k1"`` at CP32: the key-stationary rows form of the MLA call (``fused_masks`` and
            ``transposed_scores`` of ``mla_block_attention``). It changes the per-row sums, so its identity twin is the
            decode with ``mla_rows_twin`` (``tkg_mla_block.mla_rows_decode_attention``). Only on the loop path with
            ``tkg_mla_block.ROWS_SEQUENCES`` sequences for each replica, and above 16 rows only in the fit form (a table of at least
            64 local blocks), whose passes are sequence-major: each rows-form call then holds all Q rows of one sequence.
            Default off.
        mla_k1_rows_options: with ``mla_k1_rows``, more keys of the rows form of the MLA call (``tkg_mla_block``
            ``ROWS_OPTIONS``: ``folded_reference``, ``rows_v_stationary``, ``rows_dr``, ``rows_dr_act_copies``,
            ``rows_dr_inplace``, ``rows_prefetch``, ``rows_pair_lag``, ``rows_pad_skip``, ``rows_seq_hoist``,
            ``rows_setup_ahead``). The identity twin takes the values of the keys in ``tkg_mla_block.ROWS_TWIN_OPTIONS``.
            None (default) keeps the plain rows form.
        routing_form: TP128 kernels: the form of the top-16 routing of the MoE layers, as ``kimi_k3_decode_model``
            (``tkg_moe.ROUTING_FORMS``). None (default) keeps the program. The ``index_first_pe32`` forms read the
            router values as rank rows of the HBM world gather, which only the MLA loop path has: elsewhere the trace
            fails. They give the bits of ``index_first`` (the default), so the values do not change.
        moe_dst_pingpong: the gather sites of the MoE layers whose persistent HBM destination alternates by layer
            parity, as ``kimi_k3_decode_model`` (``decode_block.collective_sources`` ``dst_pingpong``): ``("gather",)``,
            the latent world gather. A gather that does not run in HBM (off the MLA loop path) keeps one destination.
            With ``dp_size`` 1 there is no DP x gather, so ``"x"`` fails. The values do not change; () (default) keeps
            the program.
        attn_rs_moe_input: the attn_rs_moe_input MoE-input scheme (``decode_block.decode_block_body``): reduce-scatter the attention
            output over the TP group, run the residual / mix / norm on the one row each rank keeps, world token all-gather
            the normed rows into the base body's x gather destination, and all-gather the updated stream beside the expert
            loop. Needs the DP4 x TP32 verify at R = TP_SIZE rows per replica (k7 at batch 16 and k3 at batch 32, 1M context) and ``moe_dst_pingpong``
            ``"x"``; it fails closed elsewhere (``decode_block.check_attn_rs_moe_input``). False (default) keeps the program.
        moe_out_proj_form: the out-projection form of the MoE layers, as ``decode_model.kimi_k3_decode_model``; the
            decode NEFF of a config must take the same form. A form other than ``reduced`` needs DP4 and the MLA loop
            path. With ``column`` the world gathers skip their start handshake, so each feed-forward chunk of a layer
            takes its own gather pairs (``collective_sources`` ``gather_hbm_sets``); with ``column_shared_ag`` also its
            own shared gather tiles (``shared_ag_sets``).
        moe_latent_load, moe_routed_store_form, moe_glue_dmas, moe_direct_latent: the static expert loop forms of the
            decode, as ``decode_model.kimi_k3_decode_model``; each feed-forward chunk of the verify runs them. The values
            do not change; the defaults keep the program. The verify routed all-reduce runs from a per-call tile, so
            ``pe`` also needs half-size weight buffers (``tkg_moe.check_routed_store_buffers``). The split verify
            (``attn_res_bank_in``, ``return_attn_res_bank``) takes the defaults.
        moe_router_rows_drop_selection, moe_latent_gather_mx: the latent world gather forms of the decode, as
            ``decode_model.kimi_k3_decode_model`` (the 21-word rows without the selection values, and with
            ``moe_latent_load`` ``pe32`` the UINT32 MXFP8 blocks of the expert input), with the weights of that form
            (``moe_vectors`` ``[n_moe, 1, 931]``; ``moe_latent_gather_mx``: ``moe_in_proj`` ``[n_moe, 128, 56, 135]``).
            Each feed-forward chunk gathers through the HBM latent gather tensors of the block sources
            (``collective_sources`` ``moe_gather_cols`` / ``moe_gather_mx``), sized by the chunk rows. They need DP4, the
            MLA loop path and the base MoE body of the TP128 projections (the static expert loop), and not the split
            verify, ``attn_rs_moe_input``, ``mla_cp`` or ``pair_shared_cc_dsts`` ``moe_gather``. The values do not
            change; False (default) keeps the program.
        moe_static_alt_tiles: the static expert loop of each feed-forward chunk takes its gate sigmoid, up tanh, down
            PSUM copy and activation eviction tiles from two buffers each, as ``decode_model.kimi_k3_decode_model``
            does (True, "act" or "act_copy"). The values do not change; False (default) keeps the program.
        attn_res_bank_in: the split verify: the AttnRes bank of the layers before ``first_layer``, HBM
            ``[128, n_in, R, H/128]`` (``n_in`` = the snapshot layers before ``first_layer``), the bank output of the
            call before (``return_attn_res_bank``). Give it exactly when ``first_layer`` > 0. The call copies it into
            its own bank before the first layer, so each row is bitwise equal to one call on all the layers. Not with
            ``mix_cache`` (the bank keeps no rstd). The caller gives the ``stream_out`` of the call before as
            ``stream``, and the weight stacks and the per-layer pages, slots and tables of the range.
        return_attn_res_bank: also return the bank of all snapshot layers up to the end of the range, a new HBM
            output ``[128, n, R, H/128]``, for the ``attn_res_bank_in`` of the next call, and then the boundary fence
            (``boundary_fence``, ``FENCE_SHAPE``), for its ``attn_res_fence_in``. The layer after the range must be
            a snapshot layer, and the call takes no final weights and no ``lm_head``. Each call returns the aux rows
            of its own layers: the call of layer ``aux_tap_source(i, aux_tap)`` writes aux layer ``i``.
        attn_res_fence_in: the split verify: the fence output of the call before. Give it exactly when
            ``first_layer`` > 0. The call loads its stream and ORs the zero fence word into its uint16 bits (an
            integer op, so no bit changes), so the stream of the first layer has a data edge on the fence.
        attn_res_bank_dma: None (default) keeps the program. ``act`` or ``sp`` (``common.BANK_DMAS``): the bank-entry
            loads of a DRAM AttnRes bank go on that hardware DGE queue at DMA priority 0, beside the expert stream
            (``decode_block.decode_block_body`` ``attn_res_bank_dma``). The values do not change.
        norm_one_mul: without ``mix_cache``, the weight multiply of each RMSNorm after an AttnRes mix is one
            instruction (``decode_block.decode_block_body`` ``norm_one_mul``). The values do not change. False
            (default) keeps the program.
        attn_res_mlp_early_entries: 0 (default) keeps the program. N > 0: the MLP AttnRes mix of each KDA layer sums
            the squares and weighted rows of its first N bank entries during the KDA body, not after the attention
            output (``decode_block.decode_block_body`` ``attn_res_kda_mlp_early_entries`` with
            ``attn_res_kda_mlp_early_hbm``). It needs the DRAM bank of the verify (more than 8 rows with k1, else more
            than 16) and the plain mix (not with ``mix_cache`` or ``attn_res_cached_bank_sums``). The values do not
            change.
        attn_res_cached_bank_sums: keep the rstd of each AttnRes bank entry from its push, so the mixes, the final mix
            included, sum no squares of the bank, as ``decode_model.kimi_k3_decode_model``. The values do not change.
            With the HBM bank above 16 rows, the bank push waits for the mix, as for the plain mix. It does nothing
            with ``mix_cache`` (always on with ``dp_size`` 1) or one row. False (default) keeps the program. Not with
            the split verify (``attn_res_bank_in``, ``return_attn_res_bank``): the bank input carries no rstd.
        mla_k1_packed: ``mla_mode="k1"`` at CP32 on the MLA loop path (Trn3, FP8 pages): each MLA call attends its
            prior keys with one packed loop (``tkg_mla_block.mla_block_attention`` ``packed_prior``) in place of one loop
            for each sequence. An item is a key group of a sequence with all its Q rows, and each row of an item runs
            the slot body of the packed decode loop on its own query and state, so each row stays bitwise equal to
            its decode step. The call runs whole sequences in each pass: one pass of up to 16 rows, or the
            sequence-major passes of the 256K-and-up form above 16 rows. Not with ``mla_k1_rows``. Default off.
        mla_k1_pack_options: with ``mla_k1_packed``, overrides of its packed-loop form (``tkg_mla_block``
            ``_PACKED_PRIOR_OPTIONS``: slots, tail_slots, prefetch, page_runs, latent_act, carry_main_only,
            sum_after_copy). None (default) takes the slot form of the served 1M-context batch-16 decode.
        mla_cp_own_key_select: with ``mla_cp``, the CP128 MLA call takes its own-key numerators with one one-hot matmul
            over the current rows (``mla_block_attention`` ``own_key_select``). The outputs are bitwise the same. False
            (default) keeps the program.
        mla_loop_first_static: with ``mla_cp`` on the loop path, the CP128 MLA call loads the page-table entries of a
            sequence once and runs its first loop iteration as static code (``mla_block_attention``
            ``loop_first_static``), so the page loads and key transposes of that iteration can start under the query
            chain. The bits do not change. The decode of the deployment takes the same key. False (default) keeps
            the program.
        mla_scores_ahead: with ``mla_cp`` on the loop path, the CP128 MLA call computes the weights of each pair of
            key groups before the sums of the pair before it (``mla_block_attention`` ``scores_ahead``). The bits
            do not change. The decode of the deployment takes the same key. False (default) keeps the program.
        mla_folded_reference: with ``mla_cp`` on the loop path, the CP128 MLA call adds the softmax reference in its
            rope matmul (``mla_block_attention`` ``folded_reference``). Other rounding, so the decode of the
            deployment must take the same key. False (default).
        mla_cp_loop_p2: with ``mla_cp`` on the loop path, the CP128 MLA call keeps its state in the (row, head) layout
            (``mla_block_attention`` ``loop_p2``) at Q = 4 and 8 (Q x 96 a multiple of 128) and up to 16 rows, so the
            numerators of each key block take Q x 96 / 128 matmuls in place of Q. The bits do not change, so the decode
            keeps its form. None (default) turns it on at B = 1 and 2; True also at B = 4 to 16 (B divides 16, Q x B at
            most 16); False keeps the head-major state.
        mla_cp_rows_options: with ``mla_cp``, keys of the rows form for the CP128 call (``tkg_mla_block.CP_ROWS_OPTIONS``:
            ``rows_dr``, ``rows_dr_act_copies``, ``rows_dr_inplace``), as ``mla_k1_rows_options`` gives them to the k1 rows
            call. ``rows_dr`` (the GPU decode's e4m3 query with FP8 double_row QK keys) changes the bits, so the CP128
            decode of the deployment must take the same keys. None (default) keeps the call.
        mla_cp_loop_preload: with ``mla_loop_first_static``, the CP128 call on the loop path takes ``mla_block_attention``
            ``loop_preload`` (the static first iteration loads the pages of the second). Bitwise; default off.
        moe_x_gather_hbm: the DP x gather of each feed-forward layer lands in the persistent HBM pair of
            ``collective_sources`` ``x_hbm`` and a load copies it to SBUF (``decode_block_body`` ``moe_x_gather_hbm``),
            not in the SBUF destination. It needs DP4, the base MoE body and the block MoE sources (no feed-forward
            chunk sources). The values do not change; False (default) keeps the program.
        moe_x_world_gather: the x gather of each feed-forward layer runs over the world group through the HBM pair of
            ``moe_x_gather_hbm`` (with or without that option): each rank sends its 4 partition rows of its replica
            (``decode_block_body`` ``moe_x_world_gather``, the form of ``kimi_k3_decode_model``), so each byte crosses the
            switch port of a chip once, not 3 times. The destination gets the bytes of the DP gather. It needs DP4, the
            MLA loop path, the base MoE body with the block MoE sources and one feed-forward chunk of more than
            ``tkg_moe.X_GATHER_PARITY_MAX_TOKENS`` rows, and ``"x"`` in ``moe_dst_pingpong``: with one destination each
            world gather keeps its start handshake (``decode_block.check_x_world_gather``). Not with the split verify,
            ``attn_rs_moe_input`` or ``mla_cp``. Precondition (the kernel cannot check it), as for the decode: the 32 TP
            ranks of a replica hold the same hidden. The values do not change; False (default) keeps the program.
        norm_scalar_squares: above ``COMPACT_MIX_TOKENS`` sequences, ScalarE takes the sums of squares of the last
            this-many sequences of each RMSNorm after an AttnRes mix (``decode_block.decode_block_body``). It applies
            on the verify bank too (unlike the AttnRes ``attn_res_scalar_*`` keys, which need the plain SBUF decode
            bank and are refused with ``verify_bank``). An order/engine change only, so the values do not change; 0
            (default) keeps the program.
        moe_preamble_buffers: K in 1 to 3 extra whole weight buffers of the static expert loop for the first K
            projections of each feed-forward call (``decode_block.decode_block_body`` ``moe_preamble_buffers``). At
            B 4 x Q 4 on the 1M loop path (T = 64, 2 whole buffers) K = 1 fits SBUF and K = 2 does not (NCC_EGCA111).
            The values do not change; 0 (default) keeps the program.
        moe_hot_prefetch: ``{"kda": N, "mla": N}`` loads the hot experts 0 .. N - 1 of each sel_load MoE layer into
            SBUF during the layer's attention (``decode_block.decode_block_body`` ``moe_hot_prefetch``). It needs
            ``routed_kernel="sel_load"``. The values do not change; None (default) keeps the program.
        moe_preamble_reuse: with ``moe_preamble_buffers``, each extra buffer also takes one more projection of the next
            expert (``decode_block.decode_block_body`` ``moe_preamble_reuse``). The values do not change; False
            (default) keeps the program.
        moe_x_dst_pair: on the loop path, the DP x gathers of the feed-forward layers alternate between the persistent
            SBUF destination ``x_dst`` and a second persistent one, ``x_dst2`` (``collective_sources`` ``x_pair``,
            ``decode_block_body`` ``x_parity``), from the second layer of the call. No gather then writes the
            destination of the gather before it, so the compiler and the runtime can skip their start handshakes.
            It needs DP4, the base MoE body and the SBUF x gather (not ``moe_x_gather_hbm``), and one more x gather
            tile of SBUF (``DP4 * R * 56`` BF16 a partition). None (default): on in the k1 fit form (above 16 rows at a
            table of ``_K1_FIT_BLOCKS`` local blocks or more) where the call allows it and takes no
            ``moe_preamble_buffers`` (both take SBUF; their pair has no measured fit), off elsewhere: the second tile
            fits the SBUF of the fit form, and at 16 rows the loop-path stack has no room for it. The values do not
            change; False keeps the program. With ``moe_weight_buffers`` the default is off too.
        moe_weight_buffers: the whole weight buffers of the static expert loop of the base MoE body, 1 to 3, in place of
            the rule of ``loop_weight_buffers`` (and of one buffer above 8 rows off the loop path). At B 4 x Q 4 on the
            1M loop path (T = 64) 3 fit SBUF, where the rule gives 2. Not with ``moe_preamble_buffers`` (another way to
            the same SBUF), ``moe_half_weight_buffers``, ``moe_k1_two_weight_buffers`` or a routed kernel option. The
            values do not change; None (default) keeps the rule.
        mla_critical_dma_priority: priority 0 on the small critical MLA DMAs (``mla_block_attention``
            ``critical_dma_priority``; "query" also the query path and the call-level state store / load on the SP
            hardware DGE queue). Bitwise the same; default off.
        mla_cache_write_off_sp: issue the MLA owner cache write off the SP queue (``mla_block_attention``
            ``cache_write_off_sp``). Bitwise the same; default off.
        mla_gemv_bands: run the narrow MLA x-stationary GEMVs in two PE column bands (``mla_block_attention``
            ``gemv_bands``). Bitwise the same; default off.
        attn_res_sbuf_entries: the k1 verify above ``_BANK_HBM_ROWS`` rows on the explicit SBUF region keeps its
            AttnRes bank in HBM. With this count, the first that-many bank entries (all of them if the count is larger)
            sit in the region heap and the others stay in HBM (``common.SplitBank``, ``attn_res_split_entries``). A mix
            reads an SBUF entry in place and loads only the HBM entries, with the instructions of the HBM bank, so the
            values do not change; the HBM entry loads fall from 1,632 to 432 per 93-layer step with 4 entries. Up to
            ``_BANK_HBM_ROWS`` rows the whole bank is in SBUF and the count does nothing. Only with ``mla_mode="k1"``,
            not with the split verify. 0 (default) keeps the program.
        kda_state_dtype: the form of the KDA recurrent state in the pages (``tkg_kda.KDA_STATE_DTYPES``), as the
            decode's ``kda_state_dtype`` (``kimi_k3_decode_model``): ``bf16`` (default) keeps the program; ``fp32`` and
            ``fp32_hilo`` read, chain and write the fp32 state, so the verify stays bitwise equal to the decode steps of
            the same form (``verify_kda.kda_verify_attention`` ``state_dtype``). The pools must be at least 101760 bf16
            elements wide at 3 local heads (the 512-row form ``[P, 147456]``).
        kda_readout: the form of the readout and the state matmuls of every KDA verify layer
            (``tkg_kda.KDA_READOUTS``, the ``readout`` of ``verify_kda.kda_verify_attention``), as the decode model's
            ``kda_readout``. None (default) keeps ``kda_default_readout(B)``: "reference" at one sequence and
            "decoupled_bf16" above, so a one-sequence step computes the KDA outputs and states in another operation
            order than the larger buckets. "decoupled_bf16" at one sequence runs the order of the larger programs, so a
            sequence keeps its KDA bits when serving changes its bucket; at two sequences and more it is the default form
            and keeps the program.
        mla_state_rows: ``mla_mode="k1"`` on the loop path: the FP32 MLA state reduce-scatter of each layer runs on
            the HBM pair of this many rows (``mla_block_collective_hbm`` ``reduce_rows``), at most ``MLA_PASS_ROWS``.
            The runtime picks the reduce-scatter algorithm, and so the summation order, by the message size: at 1M
            the pair of 4 or 8 rows (0.8 / 1.6 MB) takes the single-hop mesh and the pair of 12 or 16 rows (2.4 /
            3.2 MB) recursive halving, so the state sums of a sequence differ between the buckets. 16 gives every
            bucket the 16-row message of B = 4. The rows past the call's rows are not stored, read or used. A call
            of at least this many rows keeps its pair (a call of more than 16 rows already runs 16-row passes), so a
            deployment can pass 16 at every bucket. 0 (default) sizes the pair by the call's rows and keeps the
            program.
        pair_shared_cc_dsts: the collective destinations in LNC1 shared HBM, as
            ``decode_model.kimi_k3_decode_model``: ``moe_out``, ``moe_routed`` and ``moe_gather`` name the destinations of
            the feed-forward sources (``decode_block.collective_sources``), and ``kda_ar`` the destinations of the two
            persistent KDA all-reduce pairs, which alternate by KDA layer. With ``kda_ar`` the MLA loop path takes these
            pairs too, in place of a pair for each KDA call. The sources stay private and the values do not change. ()
            (default) keeps the program.
        kda_conv_keep: keep the KDA conv-state store sources of the verify KDA body until its stores have read
            them (``decode_block.decode_block_body`` ``kda_conv_keep``). Placement and sync only: the values do not
            change. False (default) keeps the program.
        moe_pad_rows: None (default) keeps the program. The HBM int32 ``[R, 1]`` padding-row input of the step, row
            ``j * B + b`` = token j of sequence b (1 for every row of a padding sequence, 0 for a real one): the MoE
            layers route those rows to no expert (``decode_block.decode_block_body`` ``moe_pad_rows``; a MoE body with
            ``takes_pad_rows``). The other rows keep their bits.
        attn_res_tensor_sums: on trn3 above 4 rows, the AttnRes mixes of the layers take the GPU form of the decode
            (``decode_block.decode_block_body`` ``attn_res_tensor_sums``): the weighted sums of every candidate on
            TensorE, the bank rstd kept from the pushes, and one rounding of each mix and its norm. A DRAM bank (above
            16 rows, or above 8 with ``mla_mode="k1"``) loads each entry into two SBUF tiles in turn, on the queue of
            ``attn_res_bank_dma`` (None: the SP hardware DGE queue). The final mix keeps its form, and the
            ``"mixture"`` aux tap stores the mix rounded to the stream dtype. Not bitwise the same (GPU parity). Not
            with ``mix_cache`` (or ``dp_size`` 1), ``attn_res_cached_bank_sums``, the split verify
            (``attn_res_bank_in``, ``return_attn_res_bank``), ``attn_res_mlp_early_entries`` or ``attn_rs_moe_input``
            (its one-row glue keeps its mix); ``norm_one_mul`` is inert with it. False (default) keeps the program.
        attn_res_one_round_norm: the GPU rounding of the AttnRes mixes without the TensorE sums
            (``decode_block.decode_block_body`` ``attn_res_one_round_norm``): each mix keeps its sums and returns its FP32
            accumulator, and its norm rounds once. The final mix keeps its form, and the ``"mixture"`` aux tap stores the
            mix rounded to the stream dtype. Not bitwise the same (GPU parity). Not with ``mix_cache`` (or ``dp_size`` 1)
            or ``attn_rs_moe_input``. ``attn_res_tensor_sums`` already gives this rounding. False (default) keeps the
            program.
        attn_res_early_sums: with the kept bank rstd (``mix_cache``, always on with ``dp_size`` 1, or
            ``attn_res_cached_bank_sums``), each mix sums the stream squares and takes their rsqrt first, so ScalarE
            loads the exponential table beside the VectorE sums, and the bank sums of the attention mix (and of the MLP
            mix of a KDA layer) run before the stream is ready; a DRAM bank streams its entries early
            (``decode_block.decode_block_body`` ``attn_res_early_sums``, as ``verify_model_2srv``). The same
            instructions in another order, so the values do not change; the final mix keeps its form. Needs the kept
            bank rstd. None (default): on with the kept bank rstd in a call that is not a split verify call
            (``attn_res_bank_in`` / ``return_attn_res_bank``), the form measured on the 1M-context batch-1 deployment, else off.
            False turns it off in every call.
        kda_state_loads_first, kda_slot_pages_first: ``kda_replay`` ``states`` only: trace the state loads, or the load
            of the slot pages, of every KDA verify layer before its input projection (``verify_bodies``, as
            ``verify_model_2srv``). Not with ``kda_prework_ahead``. The values do not change; False (default) keeps
            the program.
        kda_out_ar_sbuf: with a region (the MLA loop path or the MoE options), the output all-reduce of every KDA
            layer but the call's first runs SBUF -> SBUF from one persistent region-heap source tile, held to the call
            end, into the result tile (``verify_kda.kda_verify_attention`` ``ar_sbuf``), with no HBM source store and no
            destination load. The Neuron runtime takes the same all-reduce form and reduce path as for the HBM
            pair for the (128, rows, H/128) tile, so the values do not change. The call's first KDA all-reduce keeps its
            HBM pair, as the first collective of a group can skip its handshake. None (default): on in the DP1 verify
            (``dp_size`` 1, the 1M-context batch-1 deployment, where it was measured) with a region, outside split verify calls and
            without ``attn_rs_moe_input``; off elsewhere. False turns it off in every call.

    Returns:
        The stream ``[R, H]`` (the ``lm_head`` input with the final weights), then the tokens ``[R, 1]``
        int32 (``greedy``), the logits ``[R, V_s]`` (``return_logits``), the aux states (``return_aux``),
        the per-layer streams ``[L, R, H]`` (``debug``), the AttnRes bank and the boundary fence
        (``return_attn_res_bank``), and the page pools.
    """
    rows, H = stream.shape
    mix_cache = mix_cache or dp_size == 1
    early_ffn_gate = dp_size == 1 if early_ffn_gate is None else early_ffn_gate
    if attn_res_early_sums is None:
        # On where the kept bank rstd is (the 1M-context batch-1 DP1 verify: mix_cache), off in the split verify calls.
        split_call = attn_res_bank_in is not None or return_attn_res_bank
        attn_res_early_sums = bool(mix_cache or attn_res_cached_bank_sums) and not split_call
    kernel_assert(
        not attn_res_early_sums or mix_cache or attn_res_cached_bank_sums,
        "attn_res_early_sums needs the kept bank rstd: mix_cache (always on with dp_size 1) or "
        "attn_res_cached_bank_sums",
    )
    kernel_assert(
        not (kda_state_loads_first or kda_slot_pages_first) or (kda_replay == "states" and not kda_prework_ahead),
        "kda_state_loads_first and kda_slot_pages_first move the state loads of kda_replay 'states': not with "
        "kda_replay 'records' or kda_prework_ahead",
    )
    n_tiles = H // P_MAX
    num_layers = norm_weights.shape[0]
    kernel_assert(1 <= queries <= MAX_QUERIES, f"A verify step takes 1 to {MAX_QUERIES} tokens of each sequence")
    kernel_assert(rows % queries == 0, "The stream holds Q rows for each sequence")
    kernel_assert(first_layer + num_layers <= NUM_LAYERS, f"The model has {NUM_LAYERS} layers")
    kernel_assert(banks_snapshot(first_layer), "Start the layer range at a bank snapshot layer")
    n_in = sum(1 for i in range(first_layer) if banks_snapshot(i))
    kernel_assert(
        (attn_res_bank_in is not None) == (first_layer > 0),
        "Give attn_res_bank_in exactly when the layer range starts after layer 0",
    )
    kernel_assert(
        attn_res_bank_in is None
        or (tuple(attn_res_bank_in.shape) == (P_MAX, n_in, rows, n_tiles) and attn_res_bank_in.dtype == stream.dtype),
        f"attn_res_bank_in must be [{P_MAX}, {n_in}, {rows}, {n_tiles}] in the stream dtype: the bank entries of the "
        f"{n_in} snapshot layers before layer {first_layer}",
    )
    split = attn_res_bank_in is not None or return_attn_res_bank
    kernel_assert(
        (attn_res_fence_in is not None) == (first_layer > 0),
        "Give attn_res_fence_in (the fence output of the call before) exactly when the range starts after layer 0",
    )
    kernel_assert(
        attn_res_fence_in is None
        or (tuple(attn_res_fence_in.shape) == FENCE_SHAPE and attn_res_fence_in.dtype == stream.dtype),
        f"attn_res_fence_in must be {list(FENCE_SHAPE)} in the stream dtype",
    )
    kernel_assert(
        not split or rows > _BANK_HBM_ROWS,
        f"The split verify needs more than {_BANK_HBM_ROWS} rows: no test and no served config split the heap bank",
    )
    kernel_assert(
        not (mix_cache and split),
        "The split verify keeps no bank rstd: do not use mix_cache (or dp_size 1) with the bank input or output",
    )
    kernel_assert(
        not (split and attn_res_cached_bank_sums),
        "The split verify copies the bank entries of the call before without their rstd: do not use "
        "attn_res_cached_bank_sums with the bank input or output",
    )
    # The rule and the message of decode_block_body (mix_cache is also on with dp_size 1).
    kernel_assert(
        not attn_res_one_round_norm or not (mix_cache or attn_rs_moe_input),
        "attn_res_one_round_norm is the norm of an FP32 mix with one rounding: not with mix_cache (weight-row norms) "
        "or attn_rs_moe_input (its one-row glue keeps its mix)",
    )
    kernel_assert(
        not attn_res_tensor_sums
        or (
            rows > COMPACT_MIX_TOKENS
            and not (mix_cache or attn_res_cached_bank_sums or split or attn_res_mlp_early_entries or attn_rs_moe_input)
            and is_trn3()
        ),
        f"attn_res_tensor_sums is for the one-call verify above {COMPACT_MIX_TOKENS} rows on trn3: not with mix_cache "
        "(or dp_size 1), attn_res_cached_bank_sums, the split verify, attn_res_mlp_early_entries or attn_rs_moe_input",
    )
    kernel_assert(
        not split
        or (
            moe_out_proj_form == "reduced"
            and routing_form is None
            and not tuple(moe_dst_pingpong)
            and moe_latent_load == "strided"
            and moe_routed_store_form == "relayout"
            and not tuple(moe_glue_dmas)
            and not moe_direct_latent
            and not moe_router_rows_drop_selection
            and not moe_latent_gather_mx
        ),
        "The verify split boundary fence is audited only for the reduced out-projection, the default routing form, "
        "no dst ping-pong and the default static expert loop forms (moe_latent_load, moe_routed_store_form, "
        "moe_glue_dmas, moe_direct_latent, moe_router_rows_drop_selection, moe_latent_gather_mx)",
    )
    kernel_assert(
        not return_attn_res_bank
        or (
            banks_snapshot(first_layer + num_layers)
            and lm_head_weight is None
            and final_attn_res_weight is None
            and final_norm_weight is None
        ),
        "return_attn_res_bank is for a call that feeds a next call: the layer after the range must be a bank "
        "snapshot layer, and the call takes no final weights and no lm_head_weight",
    )
    kernel_assert(
        moe_out_proj_form == "reduced" or dp_size == DP_SIZE,
        f"moe_out_proj_form {moe_out_proj_form!r} needs DP{DP_SIZE}, got dp_size {dp_size}",
    )
    kernel_assert(
        lm_head_weight is not None or not (greedy or return_logits),
        "greedy and return_logits need lm_head_weight",
    )
    kernel_assert(
        dp_size != 1 or "x" not in tuple(moe_dst_pingpong),
        "moe_dst_pingpong 'x' needs the DP x gather, and dp_size 1 gathers no DP tokens: use ('gather',)",
    )
    kernel_assert(
        not mla_k1_rows
        or (
            mla_mode == "k1"
            and not mla_cp
            and dp_size == DP_SIZE
            and mla_loop_path
            and rows // queries in ROWS_SEQUENCES
            and (rows <= ROWS_PASS_ROWS or mla_block_table.shape[2] >= _K1_FIT_BLOCKS)
        ),
        "mla_k1_rows is the CP32 loop-path k1 verify of B in ROWS_SEQUENCES sequences for each replica (the form of its "
        f"decode twin, decode_model mla_rows_twin): set mla_mode k1, no mla_cp, DP{DP_SIZE}, the MLA loop path, B in "
        f"{ROWS_SEQUENCES}, and above {ROWS_PASS_ROWS} rows the fit form (at least {_K1_FIT_BLOCKS} local blocks)",
    )
    kernel_assert(
        not mla_k1_packed or (mla_mode == "k1" and not mla_cp and mla_loop_path and not mla_k1_rows and is_trn3()),
        "mla_k1_packed is the packed prior-key loop of the CP32 k1 verify on Trn3: set mla_mode k1, no mla_cp, the MLA "
        "loop path, and no mla_k1_rows",
    )
    kernel_assert(mla_k1_packed or not mla_k1_pack_options, "mla_k1_pack_options needs mla_k1_packed")
    kernel_assert(
        (mla_k1_rows or not mla_k1_rows_options) and set(mla_k1_rows_options or ()) <= set(ROWS_OPTIONS),
        f"mla_k1_rows_options needs mla_k1_rows and takes the keys {ROWS_OPTIONS}, got {mla_k1_rows_options}",
    )
    kernel_assert(
        not ((mla_k1_rows_options or {}).get("rows_pair_tail") and (mla_k1_rows_options or {}).get("rows_mx_keys")),
        "mla_k1_rows_options rows_pair_tail does not take rows_mx_keys (an unmeasured pair)",
    )
    kernel_assert(
        (mla_cp or not mla_cp_rows_options) and set(dict(mla_cp_rows_options or {})) <= set(CP_ROWS_OPTIONS),
        f"mla_cp_rows_options needs mla_cp and takes the keys {CP_ROWS_OPTIONS}, got {mla_cp_rows_options}",
    )
    kernel_assert(
        not (mla_k1_rows_options or {}).get("rows_pad_skip") and not (mla_k1_pack_options or {}).get("pad_skip"),
        "the pad skips need the runner's live-row mask, which this entry does not take",
    )
    # The SBUF routed all-reduce of the base body has one persistent source and destination; a peer's next chunk
    # could write the destination while this rank still reads the chunk before, so it needs one feed-forward chunk.
    kernel_assert(
        "routed" not in check_moe_sbuf_reduces(moe_sbuf_reduces) or dp_size == 1 or rows <= ffn_rows,
        f"moe_sbuf_reduces routed needs one feed-forward chunk: rows {rows}, ffn_rows {ffn_rows}",
    )
    kernel_assert(
        not moe_x_gather_hbm or dp_size == DP_SIZE,
        f"moe_x_gather_hbm needs the DP x gather of DP{DP_SIZE}, got dp_size {dp_size}",
    )
    kernel_assert(
        not moe_x_world_gather
        or (
            dp_size == DP_SIZE
            and mla_loop_path
            and "x" in tuple(moe_dst_pingpong)
            and X_GATHER_PARITY_MAX_TOKENS < rows <= ffn_rows
            and not split
            and not attn_rs_moe_input
            and not mla_cp
        ),
        f"moe_x_world_gather needs DP{DP_SIZE}, the MLA loop path, 'x' in moe_dst_pingpong and one feed-forward chunk of "
        f"more than {X_GATHER_PARITY_MAX_TOKENS} rows, and not the split verify, attn_rs_moe_input or mla_cp: got "
        f"dp_size {dp_size}, mla_loop_path {mla_loop_path}, moe_dst_pingpong {tuple(moe_dst_pingpong)}, rows {rows}, "
        f"ffn_rows {ffn_rows}, split {split}, attn_rs_moe_input {attn_rs_moe_input}, mla_cp {mla_cp}",
    )
    # The world x gather runs in the base MoE body of the TP128 projections (decode_block_body), which takes the block
    # MoE sources: the one-chunk loop path above gives them (no feed-forward chunk sources).
    kernel_assert(
        not moe_x_world_gather
        or (
            not moe_options_in_use(routed_kernel, router_mode, router_dtype, routed_reduce_dtype)
            and shared_sharding == "tp128"
            and latent_sharding == "tp128"
            and moe_routed_exchange == "all_reduce"
        ),
        "moe_x_world_gather needs the base MoE body of the TP128 projections (the static expert loop, the all_reduce "
        f"exchange): got routed_kernel {routed_kernel!r}, router_mode {router_mode!r}, router_dtype {router_dtype!r}, "
        f"routed_reduce_dtype {routed_reduce_dtype!r}, shared_sharding {shared_sharding!r}, latent_sharding "
        f"{latent_sharding!r}, moe_routed_exchange {moe_routed_exchange!r}",
    )
    # The latent gather keys run in the HBM latent world gather of the loop path (decode_block.collective_sources), in
    # the base MoE body of the TP128 projections, as in the decode model.
    gather_keys = moe_router_rows_drop_selection or moe_latent_gather_mx
    kernel_assert(
        not gather_keys
        or (
            dp_size == DP_SIZE
            and mla_loop_path
            and not split
            and not attn_rs_moe_input
            and not mla_cp
            and not moe_options_in_use(routed_kernel, router_mode, router_dtype, routed_reduce_dtype)
            and shared_sharding == "tp128"
            and latent_sharding == "tp128"
            and moe_routed_exchange == "all_reduce"
        ),
        f"moe_router_rows_drop_selection and moe_latent_gather_mx need DP{DP_SIZE}, the MLA loop path (the HBM latent "
        "gather) and the base MoE body of the TP128 projections (the static expert loop, the all_reduce exchange), and "
        f"not the split verify, attn_rs_moe_input or mla_cp: got dp_size {dp_size}, mla_loop_path {mla_loop_path}, "
        f"split {split}, attn_rs_moe_input {attn_rs_moe_input}, mla_cp {mla_cp}, routed_kernel {routed_kernel!r}, "
        f"router_mode {router_mode!r}, router_dtype {router_dtype!r}, routed_reduce_dtype {routed_reduce_dtype!r}, "
        f"shared_sharding {shared_sharding!r}, latent_sharding {latent_sharding!r}, moe_routed_exchange "
        f"{moe_routed_exchange!r}",
    )
    kernel_assert(
        not moe_latent_gather_mx or (moe_router_rows_drop_selection and moe_latent_load == "pe32"),
        "moe_latent_gather_mx needs moe_router_rows_drop_selection and moe_latent_load 'pe32': got "
        f"moe_router_rows_drop_selection {moe_router_rows_drop_selection}, moe_latent_load {moe_latent_load!r}",
    )
    # The latent gather keys make their own HBM latent gather tensors (tkg_moe.moe_collective_hbm), as in the decode.
    kernel_assert(
        "moe_gather" not in tuple(pair_shared_cc_dsts) or not moe_latent_gather_mx,
        "pair_shared_cc_dsts moe_gather: the UINT32 latent gather of moe_latent_gather_mx stays private",
    )
    kernel_assert(
        "moe_gather" not in tuple(pair_shared_cc_dsts) or not moe_router_rows_drop_selection,
        "pair_shared_cc_dsts moe_gather is not proven with moe_router_rows_drop_selection: its 21-word latent gather "
        "destination is not measured in shared HBM",
    )
    kernel_assert(
        moe_weight_buffers is None
        or (
            moe_weight_buffers in (1, 2, 3)
            and not moe_preamble_buffers
            and not moe_half_weight_buffers
            and not moe_k1_two_weight_buffers
            and not moe_options_in_use(routed_kernel, router_mode, router_dtype, routed_reduce_dtype)
        ),
        f"moe_weight_buffers takes 1 to 3 whole weight buffers of the static expert loop of the base MoE body, not with "
        f"moe_preamble_buffers, moe_half_weight_buffers, moe_k1_two_weight_buffers or a routed kernel option; got "
        f"{moe_weight_buffers!r}",
    )
    kernel_assert(
        not moe_x_dst_pair or (mla_loop_path and dp_size == DP_SIZE and not moe_x_gather_hbm),
        f"moe_x_dst_pair needs the loop path, the DP{DP_SIZE} x gather and the SBUF destination (not "
        f"moe_x_gather_hbm); got mla_loop_path {mla_loop_path}, dp_size {dp_size}",
    )
    all_pools = (
        kv_pool_0,
        kv_pool_1,
        kv_pool_2,
        kv_pool_3,
        kv_pool_4,
        kv_pool_5,
        kv_pool_6,
        kv_pool_7,
        kv_pool_8,
        kv_pool_9,
        kv_pool_10,
        kv_pool_11,
        kv_pool_12,
        kv_pool_13,
        kv_pool_14,
        kv_pool_15,
        kv_pool_16,
        kv_pool_17,
        kv_pool_18,
        kv_pool_19,
        kv_pool_20,
        kv_pool_21,
        kv_pool_22,
        kv_pool_23,
    )
    if pool_of_layer is None:
        pool_of_layer = tuple(range(num_layers))
    pool_of_layer = tuple(int(i) for i in pool_of_layer)
    kernel_assert(len(pool_of_layer) == num_layers, "Give the pool index of each layer of the range")
    n_pools = max(pool_of_layer) + 1
    kernel_assert(n_pools <= MAX_POOLS, f"The kernel takes at most {MAX_POOLS} page pools")
    pools = all_pools[:n_pools]
    kernel_assert(all(p is not None for p in pools), "Give each page pool that pool_of_layer uses")
    layer_pools = tuple(pools[i] for i in pool_of_layer)

    aux_layers = tuple(int(i) for i in aux_layers)
    aux_out = None
    if return_aux:
        aux_out = nl.ndarray((len(aux_layers), rows, H), dtype=stream.dtype, buffer=nl.shared_hbm, name="aux_out")
    layer_streams = None
    if debug:
        layer_streams = nl.ndarray(
            (num_layers, rows, H), dtype=stream.dtype, buffer=nl.shared_hbm, name="layer_streams"
        )

    kernel_assert(aux_tap in AUX_TAPS, f"aux_tap must be one of {AUX_TAPS}")

    # Persistent aux staging ring on the region (nki-dma-source-liveness); off the region the ring is None.
    aux_rows = {"ring": None, "waits": {}}

    def store_aux(layer, s_hT):
        if aux_out is not None and aux_tap == "prefix" and layer in aux_layers:
            store_hT_rows(aux_out[aux_layers.index(layer)], s_hT, ring=aux_rows["ring"], waits=aux_rows["waits"])

    def store_mix(layer, mixed):
        if aux_out is None or aux_tap != "mixture" or layer - 1 not in aux_layers:
            return
        # The mix can live in an MLA frame, so the store reads its own copy: a fresh tile the frame cannot
        # reuse, so the DMA keeps a stable source.
        index = aux_layers.index(layer - 1)
        # attn_res_tensor_sums and attn_res_one_round_norm give the FP32 mix: the copy rounds it to the stream dtype.
        staged = nl.ndarray(mixed.shape, dtype=stream.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=staged, src=mixed)
        store_hT_rows(aux_out[index], staged, ring=aux_rows["ring"], waits=aux_rows["waits"])

    kda_weights = (kda_w_in, kda_w_out, kda_w_small, kda_params)
    if kda_w_in_scale is not None:
        kda_weights += (kda_w_in_scale,)
    if kda_w_out_scale is not None:
        kda_weights += (kda_w_out_scale,)
    mla_weights = (mla_in_proj, mla_q_a_norm, mla_q_b_proj, mla_kv_a_norm, mla_kv_b_proj, mla_o_proj)
    if mla_in_proj_scale is not None:
        mla_weights += (mla_in_proj_scale,)
    if mla_q_b_proj_scale is not None:
        mla_weights += (mla_q_b_proj_scale,)
    if mla_kv_b_proj_scale is not None:
        mla_weights += (mla_kv_b_proj_scale,)
    if mla_o_proj_scale is not None:
        mla_weights += (mla_o_proj_scale,)
    moe_weights = (
        moe_in_proj,
        moe_out_proj,
        moe_vectors,
        expert_gate_up_weights,
        expert_gate_up_weights_scale,
        expert_down_weights,
        expert_down_weights_scale,
    )
    dense_weights = (dense_gate_up, dense_down)
    totals = block_layer_counts(first_layer, num_layers)
    # A TP128 routed kernel (sel_load / tp_dense) needs an explicit SBUF region, as the decode model.
    moe_options = moe_options_in_use(routed_kernel, router_mode, router_dtype, routed_reduce_dtype)
    kernel_assert(
        not (split and moe_options),
        "The split verify takes the default MoE options: the dead-end collective audit of the fence covers no routed kernel",
    )
    w_in_ring = kda_w_in_buffers or _w_in_buffers(kda_w_out.shape[-2] // HEAD_DIM, kda_state_page.shape[1])
    w_in_prefetch = trn3_kda_w_in_prefetch(kda_w_in_prefetch, w_in_ring, kda_w_in_prefetch_chunks)
    pair_shared = check_pair_shared_forms(
        pair_shared_cc_dsts,
        dp_moe=shared_sharding == "dp4xtp32",
        a2av_moe=None if moe_routed_exchange == "all_reduce" else moe_routed_exchange,
        moe_expert_path=pair_shared_expert_path(
            routed_kernel,
            moe_options_in_use(routed_kernel, router_mode, router_dtype, routed_reduce_dtype),
            moe_routed_store_form,
            active_expert_slots=moe_active_expert_slots,
        ),
        token_rs=bool(attn_rs_moe_input),
        # The world x gather runs through the same HBM x pair (collective_sources x_hbm).
        x_gather_hbm=bool(moe_x_gather_hbm or moe_x_world_gather),
    )
    shared_kw = {"pair_shared_cc_dsts": pair_shared} if pair_shared else {}
    moe_body = select_moe_body(shared_sharding, latent_sharding, tp_group, moe_routed_exchange)
    dp_moe = getattr(moe_body, "moe_sharding", "tp128") == "dp4xtp32"
    kernel_assert(
        not dp_moe or (dp_size == DP_SIZE and not mla_cp and not moe_options),
        "The DP4 x TP32 verify MoE body runs at DP4 with the CP32 MLA and the static expert loop",
    )
    kernel_assert(
        not (dp_moe and (moe_half_weight_buffers or moe_active_expert_slots)),
        "The DP4 x TP32 verify MoE body has no half-weight buffers and no active-expert slots",
    )
    region = None
    if mla_loop_path or moe_options:
        region = ExplicitSbuf()
        install_identities(region)
        # The ring takes 512 B of the region: the forms above AUX_RING_MAX_ROWS rows (B = 4 at Q = 4) have no room.
        if aux_out is not None and rows <= AUX_RING_MAX_ROWS:
            aux_rows["ring"] = tuple(region.heap((n_tiles, P_MAX), stream.dtype, f"aux_rows_{r}") for r in range(2))
    n_entries = n_in + sum(1 for i in range(first_layer, first_layer + num_layers) if banks_snapshot(i))
    bank_shape = (P_MAX, n_entries, rows, n_tiles)
    sbuf_entries = attn_res_split_entries(attn_res_sbuf_entries, rows, mla_mode, region is not None, n_entries, split)
    # The B <= 2 served form (rows <= 16) keeps the bank in SBUF, byte-identical to the B <= 2 form.
    # The k1 call above 8 rows needs the SBUF bytes of the bank: its bank goes to HBM, as above 16 rows.
    k1_hbm_bank = mla_mode == "k1" and rows > _BANK_HBM_ROWS
    static_sbuf_bank = region is None and rows <= 2 * _BANK_HBM_ROWS and not k1_hbm_bank
    if static_sbuf_bank:
        bank = (nl.ndarray(bank_shape, dtype=stream.dtype, buffer=nl.sbuf), 0)
    elif sbuf_entries:
        # attn_res_sbuf_entries: the first entries in the region heap, the others in HBM.
        sbuf_part = region.heap((P_MAX, sbuf_entries, rows, n_tiles), stream.dtype, "bank")
        hbm_part = None
        if n_entries > sbuf_entries:
            hbm_part = nl.ndarray(
                (P_MAX, n_entries - sbuf_entries, rows, n_tiles),
                dtype=stream.dtype,
                buffer=nl.private_hbm,
                name="verify_attn_res_bank",
            )
        bank = (SplitBank(sbuf_part, hbm_part), 0)
    elif region is None or rows > _BANK_HBM_ROWS:
        # The 93-layer verify bank is tens of KB per partition above 16 rows, so it stays in HBM.
        bank = (nl.ndarray(bank_shape, dtype=stream.dtype, buffer=nl.private_hbm, name="verify_attn_res_bank"), 0)
    else:
        bank = (region.heap(bank_shape, stream.dtype, "bank"), 0)
    # attn_res_cached_bank_sums: the bank keeps the rstd of each entry from its push, as with mix_cache, and the mixes
    # keep the bits of the plain mix (decode_block.decode_block_body). It does nothing with mix_cache or one row.
    bank_sums = attn_res_cached_bank_sums and not mix_cache and rows > 1
    # attn_res_tensor_sums keeps the rstd of each bank entry from its push too.
    if mix_cache or bank_sums or attn_res_tensor_sums:
        bank = (*bank, new_bank_rstd(n_entries, rows, region))
    if attn_res_bank_in is not None:
        # The entries of the call before go first; the pushes of this range follow them.
        for c in range(n_in):
            nisa.dma_copy(dst=bank[0][0:P_MAX, c], src=attn_res_bank_in[0:P_MAX, c], name=f"attn_res_bank_in_{c}")
        bank = (bank[0], n_in, *bank[2:])
    s = stream
    if attn_res_fence_in is not None:
        # The first block takes this SBUF stream: on the loop path it must be a stream tile of the region.
        if region is None:
            s = nl.ndarray((P_MAX, rows, n_tiles), dtype=stream.dtype, buffer=nl.sbuf)
        else:
            s = region_streams(region, rows, n_tiles, stream.dtype)[0]
        load_rows_hT(stream, s)
        fence_word = nl.ndarray(FENCE_SHAPE, dtype=stream.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=fence_word, src=attn_res_fence_in, name="attn_res_fence_in")
        bits = s.reshape((P_MAX, rows * n_tiles)).view(nl.uint16)
        nisa.tensor_scalar(
            dst=bits,
            data=bits,
            op0=nl.bitwise_or,
            operand0=fence_word.view(nl.uint16)[0:P_MAX, 0:1],
            engine=nisa.engine.vector,
        )
    gate = ()
    starts = [0, 0, 0, 0]
    # The AttnRes bank holds every row of the step and gains one entry per snapshot layer, so a long loop
    # path needs the bytes the extra expert-loop weight buffers would take (loop_weight_buffers): a 64-row step
    # keeps one buffer, a 32-row step two, smaller steps the default; with moe_k1_two_weight_buffers a k1 step
    # with its bank in HBM keeps two.
    # The buffer trim applies to the static expert loop only; a TP128 routed kernel (sel_load / tp_dense)
    # takes its default buffers (it rejects non-default weight_buffers / scale_buffers).
    weight_buffers = None
    if not moe_options:
        if moe_weight_buffers is not None:
            weight_buffers = moe_weight_buffers
        elif region is not None:
            weight_buffers = loop_weight_buffers(rows, moe_k1_two_weight_buffers and k1_hbm_bank)
        elif rows > 8:
            # Static path (the e2e): one expert-loop weight buffer in the verify only, so the 32-row MoE expert
            # loop does not spill. Decode keeps its default (two), so decode stays byte-identical.
            weight_buffers = 1
    # The block collective tiles (FFN all-reduce, lm_head argmax gather) and the MLA sites of the verify (two
    # per-token parity sets, or the "k1" set), all in one arena so tail_ranks reserves them to the kernel end.
    # mla=False: the verify MLA bodies do not read the block MLA tiles. The loop path self-allocates.
    k1 = mla_mode == "k1"
    # DP1 runs the attention bodies and the tail over the world group; DP4 keeps the TP32 groups.
    attn_group = world_group if dp_size == 1 else tp_group
    tail_tp = WORLD_TP_RANKS if dp_size == 1 else TAIL_TP_RANKS
    # DP1 gathers no DP tokens, so one feed-forward chunk covers all R <= 32 rows (R <= 128 fits SBUF).
    ff_rows = rows if dp_size == 1 else ffn_rows
    tail_ranks = tail_tp if lm_head_weight is not None and greedy else 0
    # The per-token body sub-batches each query's rows into QUERY_SET_TOKENS-row calls, so the parity sets stay
    # at set_rows (the B=4 frame) even at B=8. A larger set would make the per-token MLA frame spill the
    # collective destination arena at 93 layers, which a reload DMA then shares with a peer's gather output.
    verify_mla_rows = min(rows // queries, QUERY_SET_TOKENS) if (not k1 and region is None) else 0
    # Size the MoE gather tiles by the feed-forward chunk, so DP_SIZE * chunk fits 128 partitions.
    moe_tokens = min(rows, ffn_rows) if ffn_rows else 0
    # Above 16 rows the q input all-gather of the parity sets runs in HBM.
    q_hbm = region is None and rows > 16
    # Per-query MLA collective tensors above 16 rows (also their out and input gathers on the loop path).
    deparity_rows = min(rows // queries, QUERY_SET_TOKENS) if (not k1 and rows > 16) else 0
    deparity_out = deparity_q = region is not None
    # Above one feed-forward chunk (not DP1) the block drops the MoE tiles and ffn_sources sizes them.
    chunk_moe = region is None and rows > ffn_rows and dp_size != 1
    ffn_chunks = -(-rows // max(1, min(rows, ff_rows)))
    if mla_cp:
        verify_mla_rows = deparity_rows = 0
    # One set per MLA call: each query runs ceil(B / deparity_rows) sub-batches (B = 5..7 has a partial one).
    deparity_calls = queries * -(-(rows // queries) // deparity_rows) if deparity_rows else 0
    k1_static = k1 and region is None and not mla_cp
    k1_loop = k1 and region is not None and not mla_cp
    # Above 16 rows at a 256K table the k1 call needs the fit forms (``mla_block_attention`` ``fit_passes``).
    k1_fit = k1_loop and rows > 16 and mla_block_table.shape[2] >= _K1_FIT_BLOCKS
    kernel_assert(
        0 <= mla_state_rows <= MLA_PASS_ROWS and (mla_state_rows == 0 or k1_loop),
        f"mla_state_rows is 0 to {MLA_PASS_ROWS} rows of the state reduce-scatter of the k1 loop-path MLA call "
        f"(mla_mode 'k1', the MLA loop path, CP32); got {mla_state_rows}",
    )
    kernel_assert(
        kda_readout is None or kda_readout in KDA_READOUTS,
        f"kda_readout {kda_readout!r} must be None or one of {KDA_READOUTS}",
    )
    # The packed prior loop runs whole sequences in each pass: one pass of up to 16 rows, or the sequence-major passes of
    # the fit form (each pass holds 16 / Q whole sequences).
    kernel_assert(
        not mla_k1_packed or rows <= 16 or (k1_fit and 16 % queries == 0 and (rows // queries) % (16 // queries) == 0),
        f"mla_k1_packed needs passes of whole sequences: at most 16 rows, or the sequence-major passes above 16 rows "
        f"(a table of {_K1_FIT_BLOCKS} local blocks or more, Q dividing 16); got {rows} rows of Q = {queries} at "
        f"{mla_block_table.shape[2]} local blocks",
    )
    # A TP128 routed kernel takes the persistent sites of its collectives, as in the decode model.
    sites = moe_option_sites(routed_kernel, router_mode, router_dtype, routed_reduce_dtype, mla_loop_path)
    # The HBM x gather pair sits in the block sources, which the feed-forward chunk sources replace.
    kernel_assert(
        not moe_x_gather_hbm or (not chunk_moe and sites is None),
        "moe_x_gather_hbm needs the block MoE sources (no feed-forward chunk sources) and the base MoE body",
    )
    if moe_x_dst_pair is None:
        moe_x_dst_pair = (
            k1_fit
            and dp_size == DP_SIZE
            and sites is None
            and not dp_moe
            and not moe_x_gather_hbm
            and not attn_rs_moe_input
            and not moe_preamble_buffers
            and moe_weight_buffers is None
        )
    kernel_assert(
        not moe_x_dst_pair or (sites is None and not dp_moe and not attn_rs_moe_input),
        "moe_x_dst_pair needs the base MoE body (no routed kernel options, not the DP4 x TP32 body) and its DP x gather "
        "(not attn_rs_moe_input)",
    )
    # fp16_local in the base MoE body: the local router logits ride the packed rows of the DP x gather.
    x_logits = router_dtype == "fp16_local" and sites is None and dp_size == DP_SIZE and not dp_moe
    sources = collective_sources(
        rows,
        n_tiles,
        stream.dtype,
        region,
        tail_ranks,
        moe_options=sites,
        mla=False,
        moe=not chunk_moe,
        kda=False,
        verify_mla_rows=verify_mla_rows,
        pingpong=cc_source_pingpong,
        mla_cp=mla_cp,
        dp_size=dp_size,
        sbuf_routed_reduce="routed" in check_moe_sbuf_reduces(moe_sbuf_reduces),
        sbuf_routed_options="routed_tp128" in check_moe_sbuf_reduces(moe_sbuf_reduces),
        moe_tokens=moe_tokens,
        q_hbm=q_hbm,
        deparity_rows=deparity_rows,
        deparity_out=deparity_out,
        deparity_q=deparity_q,
        deparity_calls=deparity_calls,
        out_sets=1 if chunk_moe else ffn_chunks,
        verify_k1_heads=_MLA_LOCAL_HEADS if k1_static else 0,
        dp_moe=dp_moe,
        a2av_moe=getattr(moe_body, "a2av_expert_form", None),
        dst_pingpong=moe_dst_pingpong,
        moe_out_proj_form=moe_out_proj_form,
        token_world=attn_rs_moe_input,
        x_logits=x_logits,
        **({"x_hbm": True} if moe_x_gather_hbm or moe_x_world_gather else {}),
        **({"x_world": True} if moe_x_world_gather else {}),
        **shared_kw,
        **({"moe_gather_cols": SCORE_GATHER_COLS} if moe_router_rows_drop_selection else {}),
        **({"moe_gather_mx": True} if moe_latent_gather_mx else {}),
        **({"x_pair": True} if moe_x_dst_pair else {}),
    )
    kernel_assert(
        moe_out_proj_form not in COLUMN_FORMS
        or ffn_chunks == 1
        or len(sources.get("gather_hbm_sets", ())) >= ffn_chunks,
        f"moe_out_proj_form {moe_out_proj_form!r} with {ffn_chunks} feed-forward chunks needs a gather pair set for "
        "each chunk",
    )
    kernel_assert(
        moe_out_proj_form != "column_shared_ag"
        or ffn_chunks == 1
        or len(sources.get("shared_ag_sets", ())) >= ffn_chunks,
        f"moe_out_proj_form 'column_shared_ag' with {ffn_chunks} feed-forward chunks needs shared gather tiles for "
        "each chunk",
    )
    ffn_sources = None
    if chunk_moe:
        ffn_sources = collective_sources(
            ffn_rows,
            n_tiles,
            stream.dtype,
            region,
            moe_options=sites,
            mla=False,
            kda=False,
            name="_ffn",
            dp_size=dp_size,
            pingpong=cc_source_pingpong,
            ffn_x=False,
            out_sets=ffn_chunks,
            dp_moe=dp_moe,
            **shared_kw,
            x_logits=x_logits,
        )
    # Per-token: two parity sets, used round-robin so a token's all-gather source is not taken by the next
    # token after only the local completion. "k1": one set with its state reduce-scatter in HBM; on the loop
    # path the set is region heap tiles plus HBM pairs, with reserved output reduce-scatter sources, because a
    # frame tile of the call can take a collective source's bytes after only the local completion. The per-token
    # loop path self-allocates (mla_sets stays None).
    mla_sets = None
    if mla_cp:
        mla_sets = sources["mla_cp"]
    elif k1_static:
        mla_sets = sources["verify_k1"]
    elif k1_loop:
        # The loop frame has no room for SBUF query slots, and an output gather destination after the key
        # loops must be persistent (the handshake-skip rule), so both gathers always run in HBM here.
        k1_kw = {"q_hbm": True, "out_ag_call": True}
        mla_sets = {
            **mla_block_collective_sources(
                rows, _MLA_LOCAL_HEADS, SbufTiles(region, heap=True), with_rs=True, rs_call=k1_fit, **k1_kw
            ),
            **mla_block_collective_hbm(
                rows,
                _MLA_LOCAL_HEADS,
                name="_vk1_loop",
                state_call=k1_fit,
                **({"reduce_rows": mla_state_rows} if mla_state_rows else {}),
                **k1_kw,
                **({"q_fp8": q8} if (q8 := q_call_fp8(mla_k1_pack_options, mla_k1_rows, mla_k1_rows_options)) else {}),
            ),
        }
        pack_form = packed_prior_form(mla_k1_pack_options)
        # ``hoist_plans``: the item plans of the passes, built once for the MLA layers of the step.
        if mla_k1_packed and pack_form["hoist_plans"]:
            mla_sets["pass_plans"] = mla_pass_plans(
                position_rows[0],
                rank_info,
                mla_block_table[0],
                queries,
                mla_k1_pack_options,
                SbufTiles(region, heap=True),
                "mla_k1_plan",
            )
        # Clear the SBUF sites at the start, so the compiler places no tile on their bytes before the first layer.
        rs_tiles = [mla_sets["rs_call"]] if k1_fit else mla_sets["rs"]
        for tile in [mla_sets["qkv"], mla_sets["qkv_dst"], *rs_tiles, mla_sets["ag_call"]]:
            nisa.memset(dst=tile, value=0.0)
    elif region is None:
        mla_sets = sources["verify_mla"]
    # Per-query state HBM for the de-paritied static verify (rows > 16); None otherwise (shared parity state).
    mla_state = sources.get("verify_mla_state")
    # Two persistent HBM pairs alternate by KDA layer, as in the decode, so a store never hits a source a peer still reads.
    # A shared destination (``kda_ar``) is one persistent tensor, so the loop path takes the pairs too.
    kda_ar_pairs = None
    kda_shared = "kda_ar" in pair_shared
    if not mla_loop_path or kda_shared:
        kda_ar_pairs = tuple(
            tuple(
                nl.ndarray(
                    (P_MAX, rows, n_tiles),
                    dtype=stream.dtype,
                    buffer=pair_shared_buffer(kda_shared and kind == "dst"),
                    name=f"verify_kda_ar_{kind}{i}",
                )
                for kind in ("src", "dst")
            )
            for i in range(2)
        )
    # Persistent slot-state ring of the sequential states chain, shared across the KDA calls and the blocks
    # (nki-dma-source-liveness): a ring tile keeps its bytes past the store trigger, and the waits record carries the
    # cross-call edges, as kda_conv_rows_tiles / kda_conv_waits do for the conv-state stores. Off the region (no
    # persistent heap) the body keeps the within-call ring only.
    # kda_out_ar_sbuf: one persistent SBUF source of the KDA output all-reduce for the call (a region-heap tile, so no
    # other tile takes its bytes while the collective and its peers read it).
    kda_ar_sb = None
    if kda_out_ar_sbuf is None:
        # On in the DP1 verify with a region (the 1M-context batch-1 deployment), off in the split calls and the other forms.
        split_call = attn_res_bank_in is not None or return_attn_res_bank
        kda_out_ar_sbuf = dp_size == 1 and region is not None and not attn_rs_moe_input and not split_call
    if kda_out_ar_sbuf:
        kernel_assert(region is not None, "kda_out_ar_sbuf needs a region (the MLA loop path or the MoE options)")
        kernel_assert(
            not attn_rs_moe_input, "kda_out_ar_sbuf replaces the output all-reduce: not with attn_rs_moe_input"
        )
        # The source is held to the call end (hold_to_end): the last all-reduce reads it until its completion.
        kda_ar_sb = SbufTiles(region, heap=True).tile((P_MAX, rows, n_tiles), stream.dtype, "verify_kda_ar_sb_src")
    kda_slot_state_ring = None
    kda_slot_state_waits = None
    if kda_replay == "states" and kda_chain == "sequential" and region is not None:
        slot_tiles = kda_slot_state_tiles(
            rows // queries,
            kda_local_heads(kda_w_out[0], kda_w_small[0]),
            kda_state_dtype,
            SbufTiles(region, heap=True),
            depth=kda_slot_state_depth,
        )
        kda_slot_state_ring = tuple(slot_tiles[f"kda_slot_state_{r}"] for r in range(kda_slot_state_depth))
        kda_slot_state_waits = {r: [] for r in range(kda_slot_state_depth)}
    ffn_calls = itertools.count()
    for start, count in model_layer_groups(first_layer, num_layers):
        li = start - first_layer
        counts = block_layer_counts(start, count)
        kda_k, mla_k, moe_k, dense_k = starts
        n_kda, n_mla, n_moe, n_dense = counts
        bodies = verify_bodies(
            queries,
            _rows(kda_state_page, kda_k, n_kda),
            _rows(kda_record_page, kda_k, n_kda),
            kda_replay_len,
            kda_state_position,
            ff_rows,
            sbuf=AUTO_SBUF if region is None else SbufTiles(region),
            weight_buffers=weight_buffers,
            mla_mode=mla_mode,
            mla_sources=mla_sets,
            mla_state=mla_state,
            kda_ar=None if kda_ar_pairs is None else [kda_ar_pairs[(kda_k + i) % 2] for i in range(n_kda)],
            # The call's first KDA all-reduce stays in HBM (its group's first collective can skip the handshake).
            **(
                {"kda_ar_sbuf": [None if kda_k + i == 0 else kda_ar_sb for i in range(n_kda)]}
                if kda_ar_sb is not None
                else {}
            ),
            ffn_sources=ffn_sources,
            lora_norm_eps=lora_norm_eps,
            ffn_calls=ffn_calls,
            mla_cp=mla_cp,
            kda_replay=kda_replay,
            kda_state_slot_page=None if kda_state_slot_page is None else _rows(kda_state_slot_page, kda_k, n_kda),
            kda_write_state_slots=None if kda_write_state_slots is None else _rows(kda_write_state_slots, kda_k, n_kda),
            kda_prework_ahead=kda_prework_ahead,
            kda_conv_dtype=kda_conv_dtype,
            kda_chain=kda_chain,
            kda_token_tiles=kda_token_tiles,
            kda_slot_store_queues=kda_slot_store_queues,
            kda_slot_store_engine=kda_slot_store_engine,
            kda_slot_state_ring=kda_slot_state_ring,
            kda_slot_state_waits=kda_slot_state_waits,
            kda_slot_state_depth=kda_slot_state_depth,
            kda_w_in_buffers=kda_w_in_buffers,
            kda_readout=kda_readout,
            **({"kda_state_loads_first": True} if kda_state_loads_first else {}),
            **({"kda_slot_pages_first": True} if kda_slot_pages_first else {}),
            kda_conv_load_gather=kda_conv_load_gather,
            kda_batched_conv_stores=kda_batched_conv_stores,
            kda_null_page=kda_null_page,
            **({"kda_slot_pages_contiguous": True} if kda_slot_pages_contiguous else {}),
            **({"kda_batched_slot_conv_stores": True} if kda_batched_slot_conv_stores else {}),
            **({"kda_bf16_update_rows": True} if kda_bf16_update_rows else {}),
            half_weight_buffers=moe_half_weight_buffers,
            gate_up_half_major=moe_gate_up_half_major,
            mla_q_hbm=k1_loop,
            mla_out_ag=k1_loop,
            mla_k1_fit=k1_fit,
            mla_zeroed_pages=mla_zeroed_pages,
            moe_body=moe_body,
            mla_late_tail_weights=mla_late_tail_weights,
            mla_pass_q_prefetch=mla_pass_q_prefetch,
            mla_q_gather_buffers=mla_q_gather_buffers,
            mla_tail_weights_after_rs=mla_tail_weights_after_rs,
            mla_pass_rows_ahead=mla_pass_rows_ahead,
            mla_k1_rows=mla_k1_rows,
            mla_k1_rows_options=mla_k1_rows_options,
            mla_k1_packed=mla_k1_packed,
            mla_k1_pack_options=mla_k1_pack_options,
            mla_cp_own_key_select=mla_cp_own_key_select,
            mla_loop_first_static=mla_loop_first_static,
            mla_scores_ahead=mla_scores_ahead,
            mla_folded_reference=mla_folded_reference,
            mla_cp_loop_p2=mla_cp_loop_p2,
            mla_cp_rows_options=mla_cp_rows_options,
            mla_cp_loop_preload=mla_cp_loop_preload,
            **({"attn_rs_moe_input": True} if attn_rs_moe_input else {}),
            mla_critical_dma_priority=mla_critical_dma_priority,
            mla_cache_write_off_sp=mla_cache_write_off_sp,
            mla_gemv_bands=mla_gemv_bands,
            kda_state_dtype=kda_state_dtype,
        )
        if w_in_prefetch is not None:
            w_in_prefetch.set_next_block(
                start + count, first_layer + num_layers, kda_w_in, kda_w_in_scale, kda_k + n_kda, totals[0]
            )
        s, bank, gate = decode_block_body(
            s,
            bank,
            _rows(norm_weights, li, count),
            _rows(attn_res_weights, li, count),
            tuple(_group_stack(t, kda_k, n_kda, totals[0]) for t in kda_weights),
            tuple(_group_stack(t, mla_k, n_mla, totals[1]) for t in mla_weights),
            _moe_group(moe_weights, moe_k, n_moe, totals[2], routed_kernel),
            tuple(_group_stack(t, dense_k, n_dense, totals[3]) for t in dense_weights),
            layer_pools[li : li + count],
            _LayerIndex(),
            _rows(mla_block_table, mla_k, n_mla),
            _rows(mla_slot, mla_k, n_mla),
            position_rows,
            rank_info,
            attn_group,
            dp_group,
            world_group,
            start,
            eps,
            *bodies,
            None if layer_streams is None else _rows(layer_streams, li, count),
            None,
            dma_gates,
            gate,
            mla_fp8_pages=mla_fp8_pages,
            region=region,
            mla_loop_path=mla_loop_path,
            after_layer=store_aux,
            before_attention=store_mix,
            sources=sources,
            last_block=start + count == first_layer + num_layers,
            moe_active_expert_slots=moe_active_expert_slots,
            routed_kernel=routed_kernel,
            routed_reduce_dtype=routed_reduce_dtype,
            router_mode=router_mode,
            router_dtype=router_dtype,
            moe_router_weights=_group_stack(moe_router_weight, moe_k, n_moe, totals[2]),
            moe_in_proj_scales=_group_stack(moe_in_proj_scale, moe_k, n_moe, totals[2]),
            moe_expert_base=moe_k,
            affinity_scaling=affinity_scaling,
            affinity_hi_lo=affinity_hi_lo,
            dp_size=dp_size,
            mla_options=mla_options,
            mix_cache=mix_cache,
            moe_sel_load_options=moe_sel_load_options,
            early_ffn_gate=early_ffn_gate,
            verify_bank=rows > 16 or k1_hbm_bank,
            **({"ffn_weights_after_attn": True} if ffn_weights_after_attn else {}),
            routing_form=routing_form,
            **({"attn_rs_moe_input": True} if attn_rs_moe_input else {}),
            **({} if moe_out_proj_form == "reduced" else {"moe_out_proj_form": moe_out_proj_form}),
            **({} if moe_latent_load == "strided" else {"moe_latent_load": moe_latent_load}),
            **({} if moe_routed_store_form == "relayout" else {"moe_routed_store_form": moe_routed_store_form}),
            **({"moe_glue_dmas": tuple(moe_glue_dmas)} if moe_glue_dmas else {}),
            **({"moe_direct_latent": True} if moe_direct_latent else {}),
            **({"moe_router_rows_drop_selection": True} if moe_router_rows_drop_selection else {}),
            **({"moe_latent_gather_mx": True} if moe_latent_gather_mx else {}),
            **({"moe_static_alt_tiles": moe_static_alt_tiles} if moe_static_alt_tiles else {}),
            **({"attn_res_cached_bank_sums": True} if bank_sums else {}),
            **({"attn_res_bank_dma": attn_res_bank_dma} if attn_res_bank_dma else {}),
            **({"norm_one_mul": True} if norm_one_mul else {}),
            **({"attn_res_early_sums": True} if attn_res_early_sums else {}),
            **(
                {"attn_res_kda_mlp_early_entries": attn_res_mlp_early_entries, "attn_res_kda_mlp_early_hbm": True}
                if attn_res_mlp_early_entries
                else {}
            ),
            **({"moe_x_gather_hbm": True} if moe_x_gather_hbm else {}),
            **({"moe_x_world_gather": True} if moe_x_world_gather else {}),
            **({"kda_conv_keep": True} if kda_conv_keep else {}),
            **({"moe_pad_rows": moe_pad_rows} if moe_pad_rows is not None else {}),
            **({"attn_res_tensor_sums": True} if attn_res_tensor_sums else {}),
            **({"attn_res_one_round_norm": True} if attn_res_one_round_norm else {}),
            **({"norm_scalar_squares": norm_scalar_squares} if norm_scalar_squares else {}),
            **({"moe_preamble_buffers": moe_preamble_buffers} if moe_preamble_buffers else {}),
            **({"moe_hot_prefetch": dict(moe_hot_prefetch)} if moe_hot_prefetch is not None else {}),
            **({"moe_preamble_reuse": True} if moe_preamble_reuse else {}),
            **({"x_parity": True} if moe_x_dst_pair else {}),
            **({"kda_w_in_prefetch": w_in_prefetch} if w_in_prefetch is not None else {}),
        )
        starts = [k + n for k, n in zip(starts, counts, strict=True)]

    if final_attn_res_weight is not None or final_norm_weight is not None:
        ones = new_ones() if region is None else region.ones
        if final_attn_res_weight is not None:
            bank_hT, count = bank[:2]
            bank_view = bank_hT if count == n_entries else bank_hT[0:P_MAX, 0:count]
            final_w = load_vector_hT(final_attn_res_weight)
            mix = (
                {"bank_rstd": bank[2][0:P_MAX, 0:count], "w_row": WeightRow(final_w), "vector_exp": True}
                if mix_cache and count
                else {}
            )
            if bank_sums and count:
                mix = {"bank_rstd": bank[2][0:P_MAX, 0:count]}
            s = attn_res_hT(bank_view if count else None, s, final_w, eps, ones, **mix)
        if final_norm_weight is not None:
            # The final hidden takes the FFN input tile, so the collective tiles stay reserved through lm_head.
            final_x = layer_sources(sources, first_layer + num_layers)["ffn_x"]
            s = rmsnorm_hT(s, load_vector_hT(final_norm_weight), eps, ones, out=final_x)

    out = nl.ndarray((rows, H), dtype=stream.dtype, buffer=nl.shared_hbm, name="stream_out")
    bank_out = fence_sb = None
    if return_attn_res_bank:
        bank_out = nl.ndarray(bank_shape, dtype=stream.dtype, buffer=nl.shared_hbm, name="attn_res_bank_out")
        # The outputs in return order; the call has no tokens and no logits.
        fence_outputs = (out, *(t for t in (aux_out, layer_streams) if t is not None), bank_out, *pools)
        fence_sb = fence_tiles(fence_outputs, stream.dtype)
    store_hT_rows(out, s)
    if return_attn_res_bank:
        # No tile comes after the bank stores, so none takes the bytes of an SBUF bank source.
        for c in range(n_entries):
            nisa.dma_copy(dst=bank_out[0:P_MAX, c], src=bank[0][0:P_MAX, c], name=f"attn_res_bank_out_{c}")
    tail = ()
    if lm_head_weight is not None and (greedy or return_logits):
        logits = None
        if return_logits:
            logits = nl.ndarray(
                (rows, lm_head_weight.shape[1]), dtype=stream.dtype, buffer=nl.shared_hbm, name="logits"
            )
        with weight_gates(dma_gates):
            set_weight_gate(gate)
            tiles = (sources["argmax_src"], sources["argmax_dst"], sources["argmax_token"]) if tail_ranks else None
            token = lm_head_greedy_hT(
                s,
                lm_head_weight,
                attn_group,
                "tail",
                sample=greedy,
                logits_out=logits,
                tp_size=tail_tp,
                gather_tiles=tiles,
                chunk_logits=dp_size != 1 and rows > 16,
            )
        if greedy:
            tokens = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.shared_hbm, name="tokens")
            nisa.dma_copy(dst=tokens, src=token, name="tail_tokens_store")
            tail += (tokens,)
        if return_logits:
            tail += (logits,)
    if aux_out is not None:
        tail += (aux_out,)
    if layer_streams is not None:
        tail += (layer_streams,)
    if bank_out is not None:
        tail += (bank_out,)
    if kda_ar_sb is not None:
        # The persistent SBUF all-reduce source stays live to the end of the call (kda_out_ar_sbuf).
        hold_to_end([kda_ar_sb], list(tail), out)
    outputs = (out, *tail, *pools)
    if fence_sb is not None:
        # The last instructions of the call: the boundary fence reads every output it returns (``boundary_fence``).
        outputs = (out, *tail, boundary_fence(outputs, world_group, fence_sb), *pools)
    if region is not None:
        check_identities(region)
    return outputs
