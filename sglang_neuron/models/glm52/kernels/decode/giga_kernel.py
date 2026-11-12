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

"""Consecutive GLM-5.2 decoder layers in one ``@nki.jit`` trace, on 128 ranks: the attention layer
(``glm52_dsa_attention_decode_layer_kernel``) and the MoE or dense MLP layer chained with the two bf16 residual adds. Both
leading norms run inside those layer kernels, so the adds are the only arithmetic this module adds.

The two layers need no permutation between them: the attention layer's entry and exit token slots and the MoE
layer's ``[2r, 2r + 2)`` are the same rows on every rank.

Index sharing. Only the ``"full"`` layers of ``config.indexer_types`` run the indexer; a ``"shared"`` layer has no
indexer weights or key cache and reuses the preceding full layer's selection, which skips four sub-kernels and four
collectives. From layer 6 on the pattern is groups of four, one full and three shared (``index_share_period``).

Per-layer weights carry a leading layer axis and are indexed by the Python loop counter, so ``W[i]`` is that layer's
tensor in the shape its sub-kernel takes (NKI takes no list arguments). The counter is the same on every rank, which
matters because one NEFF, traced once, runs on all ranks. Indexer weights and keys have one slot per full layer, and
the MoE stacks may skip the dense layers. Layer-invariant inputs (RoPE tables, positions, slot mappings, block table,
replica groups) are passed once.
"""

import math
import os

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from .attention.indexer.indexer_block_rows import indexer_block_rows
from .attention.indexer.indexer_constants import TOPK_MAX_N, indexer_cache_is_blocked, indexer_cache_keys
from .attention.indexer.indexer_residency_mask import indexer_residency_mask
from .utils.cc_dst_ring import close_cc_dst_ring, open_cc_dst_ring
from .utils.cc_pin import close_pins, next_layer, open_pins, pin
from .utils.kernel_assert import kernel_assert
from .utils.row_layout import spread_rows_shape
from .utils.trace_names import unique_name
from . import attention_layer as _dsa_layer
from .attention_layer import (
    build_norm_reduce_ones,
    build_shared_prior_index,
    glm52_dsa_attention_decode_layer_kernel,
    reset_rope_table_cache,
)
from .lm_head_tail import glm52_lm_head_tail
from .moe_layer import glm52_moe_decode_layer_dp_tp_ep_kernel

_ERROR_PREFIX = "[GLM-5.2 Decode Giga Kernel]"

_ATTN_PIN_WINDOW_MIN_KEYS = 262144
"""Keys a rank holds from which the attention collective operands are pinned for a window (see ``open_pins``).
Above the 704k point (180224 keys a rank at cp 4), so the served contexts pin them to the kernel end."""

_ATTN_PIN_WINDOW = 2
"""Layers that the attention collective operands stay live from ``_ATTN_PIN_WINDOW_MIN_KEYS``."""

GK_MAX_IDX_KEYS = 262144
"""The largest ``idx_keys`` (indexer keys of one rank) this copy serves: 1,048,576 tokens at key_size 4.

The local select ranks such a shard in chunks (``indexer_topk_local_carry_index._pick_local_chunks``: 8 of 32768,
phase 2 over 8 x 2048), so no select is wider than ``IDX_TOPK_MAX_CALL_N``; the chunker itself reaches 524288. The
kernel does not read this value: the plugin's ``models/glm52/decode/kernel_limits.py`` reads it from this source (an int literal,
parsed, not imported) and refuses a geometry above it, for this giga kernel and the ``mtp/`` fork's rows chain."""

GK_OPTIONS = ("router_tp", "moe_sparse_slots", "idx_fp8")
"""The 1M HBM cuts this copy serves that no argument names, because the kernel selects them from an input's shape
or dtype: ``router_tp`` (``router_weights_all`` of ``[L, H, E / tp_size]``), ``moe_sparse_slots`` (MoE stacks of
``L - first_dense_layers`` slots, ``_moe_layer_skip``) and ``idx_fp8`` (a uint8 ``idx_k_cache_all``, e4m3 keys).
The kernel does not read this value: the plugin's ``models/glm52/decode/kernel_limits.py`` parses it from this source (a tuple
of str literals) and refuses a point that sets an option the copy does not declare. The cuts named by arguments
(``latent_replica_group``; ``dense_replica_group`` / ``dense_tp_size``; ``shared_replica_group`` /
``shared_tp_size``) are read from the signature instead."""

_COLLECTIVE_HBM = nl.private_hbm
"""Every HBM buffer here is ``private_hbm``. At LNC=1 the runtime rejects a collective whose SOURCES are
``shared_hbm``, and a collective still enforces a single buffer class across all of its sources and
destinations, so private on both ends is the only self-consistent choice. Only this kernel's declared
OUTPUTS stay ``shared_hbm``."""

ATTN_EP_SYNC_FIRST = os.environ.get("GLM52K_ATTN_EP_SYNC_FIRST", "0") == "1"
"""Issue a 32 B-per-rank 128-rank AllGather right before the FIRST MoE layer of each GK call, so no rank starts that
layer's EP collectives before every rank has reached it. In a mixed-length batch the DP halves reach the first MoE
layer of a call far apart (call 1: three indexer layers without a 128-rank sync), and the early half's
handshake-skipped writes into the late ranks' first-use ring destinations then corrupt them as soon as the
allocation around layers 1-3 / 26-29 moves (a dynamic loop there, GLM52K_CC_PIN_ATTN=1): NaN hidden states on the
early half. Same values: bit-identical; one tiny collective per call. 0 (default) = off."""

def _residual_add(
    h: nl.NkiTensor, delta: nl.NkiTensor, tag: str, h_sb: nl.NkiTensor = None
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """``residual + hidden_states`` on this rank's token shard, in the residual stream's dtype (bf16, as the reference
    adds it).

    The add runs on the partition-spread view of the rows (``spread_rows_shape``, ``[128, 96]`` at the served shape):
    the same elements, 96 per partition instead of 6144, which makes the Vector op much shorter. The SBUF
    copy of the sum is returned so the next consumer (the attention input norm, the next residual add) skips its load.

    Args:
        h: [T_shard, H] the residual, in HBM.
        delta: [T_shard, H] the block's contribution in HBM, or an SBUF tile of the spread shape.
        tag: debug-name suffix, unique per call site.
        h_sb: optional SBUF copy of ``h`` in the spread shape; skips the load.

    Returns:
        The sum as [T_shard, H] in HBM in ``h``'s dtype, and in SBUF in the spread shape (token-major when the rows do
        not spread).
    """
    T_shard, H = h.shape
    sb_shape = spread_rows_shape(T_shard, H) or (T_shard, H)
    if h_sb is None:
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
    return out, sum_sb


@nki.jit
def glm52_decode_giga_kernel(
    # ── input: this rank's 128-way token shard of the residual stream entering the stack ──
    x_shard: nl.NkiTensor,  # [T_shard, H]
    # ── attention, per layer: leading axis [L, ...] ──
    input_norm_w_all: nl.NkiTensor,  # [L, 1, H]
    wq_a_all: nl.NkiTensor,  # [L, H, q_lora_shard]
    wkv_a_all: nl.NkiTensor,  # [L, H, kv_row_shard]      columns ordered [k_pe | kv_latent]
    q_norm_w_all: nl.NkiTensor,  # [L, 1, q_lora_rank]
    kv_norm_w_all: nl.NkiTensor,  # [L, 1, C]
    wq_b_all: nl.NkiTensor,  # [L, q_lora_rank, q_up_shard]
    q_absorb_w_all: nl.NkiTensor,  # [L, Nh / token_group, nope, C]
    out_absorb_w_all: nl.NkiTensor,  # [L, C, nh_out, v_head] latent-major
    o_proj_w_all: nl.NkiTensor,  # [L, nh_out * v_head, H]
    # ── attention, layer-invariant ──
    cos: nl.NkiTensor,  # [T_local, rope // 2] fp32, TRUE positions
    sin: nl.NkiTensor,  # [T_local, rope // 2] fp32
    # ── DSA indexer, per FULL layer: leading axis [L_full, ...] ──
    idx_wq_b_all: nl.NkiTensor,  # [L_full, q_lora_rank, idx_q_shard]
    idx_wk_all: nl.NkiTensor,  # [L_full, H, idx_head_dim]
    idx_k_norm_w_all: nl.NkiTensor,  # [L_full, 1, idx_head_dim] fp32
    idx_k_norm_b_all: nl.NkiTensor,  # [L_full, 1, idx_head_dim] fp32
    idx_weights_proj_all: nl.NkiTensor,  # [L_full, H, idx_n_heads]
    # ── DSA indexer, layer-invariant ──
    idx_cos: nl.NkiTensor,  # [spec, idx_rope // 2] fp32
    idx_sin: nl.NkiTensor,  # [spec, idx_rope // 2] fp32
    # ── caches: per layer, MUTATED in place ──
    kv_cache_all: nl.NkiTensor,  # [L, num_blocks, 1, block_size, 640] uint8
    idx_k_cache_all: nl.NkiTensor,  # [L_full, n_blocks, 1, idx_block_size, idx_head_dim], or legacy [L_full, idx_keys, idx_head_dim]
    # ── per-step runtime tensors, all layer-invariant ──
    kv_new_slot_mapping: nl.NkiTensor,  # [num_seqs_local, spec] int32
    block_table: nl.NkiTensor,  # [num_seqs_local, local_blocks + 1] int32
    positions: nl.NkiTensor,  # [T_local] int32, CP-SHIFTED
    idx_new_slot_mapping: nl.NkiTensor,  # [spec] int32
    idx_positions: nl.NkiTensor,  # [spec] int32, KEY-SHIFTED
    key_offset: nl.NkiTensor,  # [1, 1] int32
    cp_pos_offset: nl.NkiTensor,  # [1, 1] int32
    # ── MoE, per layer: leading axis [L, ...], or [L - first_dense_layers, ...] with no dense-layer slot ──
    gamma_all: nl.NkiTensor,  # [L, 1, H] post_attention_layernorm weight
    router_weights_all: nl.NkiTensor,  # [L, H, E] bf16, or [L, H, E / tp_size] (router_tp)
    router_bias_all: nl.NkiTensor,  # [L, 1, E]
    routed_gate_up_weights_all: nl.NkiTensor,  # [L, E_L, 128, 2, H/512, I]
    routed_gate_up_weights_scale_all: nl.NkiTensor,  # [L, E_L, 2, 64, H/512, I/4]
    routed_down_weights_all: nl.NkiTensor,  # [L, E_L, 128, I/512, H]
    routed_down_weights_scale_all: nl.NkiTensor,  # [L, E_L, 64, H]
    shared_gate_up_weights_all: nl.NkiTensor,  # [L, 1, 128, 2, H/512, I_sh]
    shared_gate_up_weights_scale_all: nl.NkiTensor,  # [L, 1, 16, 2, H/512, I_sh]
    shared_down_weights_all: nl.NkiTensor,  # [L, 1, I_sh/4, 1, H]
    shared_down_weights_scale_all: nl.NkiTensor,  # [L, 1, I_sh/32, 1, H]
    # ── MoE, layer-invariant ──
    shared_affinities: nl.NkiTensor,  # [T_sh, 1] ones
    # ── dense MLP, per DENSE layer: leading axis [L_dense, ...] ──
    # Their own stacks, not the shared-expert ones, because the intermediate widths differ: a dense
    # layer is intermediate_size / tp_size (12288/8 = 1536) while a sparse layer's shared expert is
    # moe_intermediate_size * n_shared_experts / tp_size (2048/8 = 256). One leading axis cannot carry
    # both. A stack with no dense layers passes a single unread dummy entry, as the indexer stacks do.
    dense_gamma_all: nl.NkiTensor,  # [L_dense, 1, H]
    # bf16 and UNPACKED, because the checkpoint keeps layers 0..2's MLP unquantized. No micro-scales
    # and no affinity column: the dense MLP is a plain matmul chain, not an expert.
    dense_gate_weights_all: nl.NkiTensor,  # [L_dense, H, I_dense]
    dense_up_weights_all: nl.NkiTensor,  # [L_dense, H, I_dense]
    dense_down_weights_all: nl.NkiTensor,  # [L_dense, I_dense, H]
    rank_id: nl.NkiTensor,  # [1, 1] uint32, EP rank
    shared_rank_id: nl.NkiTensor,  # [1, 1] uint32, zero
    # ── replica groups, all layer-invariant ──
    token_replica_group: ReplicaGroup,
    cp_replica_group: ReplicaGroup,
    head_replica_group: ReplicaGroup,
    seq_replica_group: ReplicaGroup,
    key_replica_group: ReplicaGroup,
    ep_replica_group: ReplicaGroup,
    tp_replica_group: ReplicaGroup,
    # ── stack shape, rank-invariant ──
    num_layers: int,
    index_share_period: int,
    # ── attention degrees, rank-invariant ──
    cp_degree: int,
    head_shard: int,
    token_shard: int,
    key_size: int,
    # ── model scalars ──
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
    index_skip_topk_offset: int = 0,
    layer_offset: int = 0,
    first_dense_layers: int = 0,
    prev_gather_indices: nl.NkiTensor = None,  # [T_local, index_topk] int32, or None
    # ── block-sharded o_proj, only when out_absorb_w_all carries fewer than Nh / cp_degree heads ──
    block_replica_group: ReplicaGroup = None,  # the Nh / nh_out consecutive ranks sharing this rank's o_proj block
    qblock_replica_group: ReplicaGroup = None,  # the consecutive ranks sharing this rank's wq_b block (block-sharded query path)
    latent_replica_group: ReplicaGroup = None,  # the consecutive ranks sharing this rank's wq_a / wkv_a columns (latent-sharded down-projection)
    # ── MoE-side HBM cuts: the dense MLP and the shared expert over groups wider than TP ──
    dense_replica_group: ReplicaGroup = None,  # the consecutive ranks splitting the dense MLP's columns, or None for the TP group
    dense_tp_size: int = 0,  # ranks in one dense_replica_group group; 0 = tp_size
    shared_replica_group: ReplicaGroup = None,  # the consecutive ranks splitting the shared expert's columns, or None for the TP group
    shared_tp_size: int = 0,  # ranks in one shared_replica_group group; 0 = tp_size (shared_affinities has T_shard * this rows)
    # ── the POOLED indexer (idx_pool_degree > 0): this rank scores its key shard of EVERY sequence of its pool ──
    pool_replica_group: ReplicaGroup = None,  # the E ranks at this rank's key position, one per CP group, in slot order
    idx_pool_degree: int = 0,  # E, the CP groups pooled; 0 = the dealt-per-group indexer above
    idx_pool_tokens: int = 0,  # query tokens scored per sequence on the pooled path (1 without MTP, spec with it)
    idx_block_table: nl.NkiTensor = None,  # [1, idx_keys / idx_block_size + 1] int32 logical->physical block map for the INDEXER cache, or None for a position-ordered cache
    # ── the on-device sampling tail of the last part, rank-sharded vocab ──
    lm_head_weight_t: nl.NkiTensor = None,  # [H, V / ep_size] bf16, this rank's lm_head columns, TRANSPOSED
    final_norm_w: nl.NkiTensor = None,  # [1, H] fp32 model.norm weight
    sampling_params: nl.NkiTensor = None,  # [T_shard, 3|4] fp32 [top_k, top_p, temperature(, draw)], rows this rank OWNS
    lm_head_hops: tuple = None,  # all_to_all hop sizes multiplying to ep_size, e.g. (128,) or (8, 16)
    lm_head_k: int = 64,
    kv_cache_hi: nl.NkiTensor = None,  # [L_hi, num_blocks, 1, block_size, 640] uint8, the next layers of a SPLIT latent stack, or None
    kv_cache_hi2: nl.NkiTensor = None,  # [L_hi2, num_blocks, 1, block_size, 640] uint8, the last layers of a three-way split, or None
    idx_shard_degree: int = 1,  # degree the indexer's keys are DEALT over; 1 = its own contiguous slice
    idx_block_size: int = 0,  # the indexer cache's block size; also the deal unit under a shared table
    # ── layer window (the point's GK split, ``gk_split``): run stack layers [layer_start, layer_stop) only ──
    layer_start: int = 0,  # first stack layer of this call; a FULL layer when > 0 (no carried selection)
    layer_stop: int = 0,  # one past the last stack layer of this call; 0 = num_layers. The sampling tail runs
    # only in the call that ends the stack. Each call is its own NEFF function, with its own block of collective
    # control state, which must fit on chip (points.GK_SPLIT has the sizes) or every collective of the block
    # reads its steps from HBM (slower in each collective-core phase).
    # ── the CHUNK-DEALT pool (CDP, layout/decode/idx_pool.py deal="chunk"; R regions = top-k chunks a
    # rank), with idx_pool_degree ──
    idx_pool_items: nl.NkiTensor = None,  # [R, 8] int32, this rank's live foreign regions
    idx_pool_n_items: nl.NkiTensor = None,  # [1, 1] int32, the foreign loop's trip count
    idx_pool_region_base: nl.NkiTensor = None,  # [R] int32, the step cut of each region's owner (-1 = empty)
    idx_pool_new_key_rows: nl.NkiTensor = None,  # [E * idx_pool_tokens, 1] int32, this rank's row of each pool new key
    idx_pool_unrot: nl.NkiTensor = None,  # [R, 2] int32, recv offsets (values, positions) of phase-1 slot q
    kv_cache_hi3: nl.NkiTensor = None,  # [L_hi3, num_blocks, 1, block_size, 640] uint8, a four-way split's last layers
):
    """``num_layers`` consecutive GLM-5.2 decoder layers, collectives included (see the module docstring).

    Shapes: ``L`` layers in the stack, ``L_full`` of them running their own indexer, ``L_moe`` with an MoE slot
    (``L``, or ``L - first_dense_layers`` under ``moe_sparse_slots``), ``L_dense`` dense layers; ``T_shard``,
    ``T_local``, ``H``, ``C`` and ``Nh`` as in the attention layer; ``E_L`` local routed experts, ``I`` the routed and
    ``I_sh`` the shared-expert intermediate on this rank.

    Args:
        x_shard: [T_shard, H] this rank's slots of the residual stream entering the stack.
        input_norm_w_all, wq_a_all, wkv_a_all, q_norm_w_all, kv_norm_w_all, wq_b_all, q_absorb_w_all,
            out_absorb_w_all, o_proj_w_all: [L, ...] the attention layer's weights, one slot per layer, in the
            shards ``glm52_dsa_attention_decode_layer_kernel`` documents (a latent-sharded down-projection needs
            ``latent_replica_group``, a block-sharded exit ``block_replica_group``).
        cos, sin, idx_cos, idx_sin: the RoPE tables, shared by every layer.
        idx_wq_b_all, idx_wk_all, idx_k_norm_w_all, idx_k_norm_b_all, idx_weights_proj_all: [L_full, ...] the
            indexer weights of the full layers.
        kv_cache_all: [L_lo, num_blocks, 1, block_size, 640] uint8 latent caches, updated in place.
        kv_cache_hi, kv_cache_hi2, kv_cache_hi3: further stacks of the same caches for the later layers, or None. An
            indirect DMA's static offset is 32 bits, so every layer must start below 2**32 bytes of its stack; long
            contexts split the 78 layers over two to four stacks. Each is returned after every other output.
        idx_k_cache_all: [L_full, n_blocks, 1, idx_block_size, idx_head_dim] indexer key caches, updated in place.
        kv_new_slot_mapping, block_table, positions, cp_pos_offset: the attention inputs shared by every layer.
        idx_new_slot_mapping, idx_block_table, idx_positions, key_offset, idx_shard_degree, idx_block_size: the
            indexer inputs shared by every layer; ``idx_block_table`` makes the key cache paged, None keeps it
            position-ordered.
        gamma_all, router_weights_all, router_bias_all, routed_*_all, shared_*_all: [L_moe, ...] the MoE layer's
            norm, router and expert weights (MX-packed with their uint8 scales).
        shared_affinities: [T_shard * shared_tp_size, 1] all-ones affinity column of the shared expert.
        dense_gamma_all, dense_gate_weights_all, dense_up_weights_all, dense_down_weights_all: [L_dense, ...] the dense
            layers' bf16 MLP, column-sharded over ``dense_replica_group`` (down row-sharded).
        rank_id, shared_rank_id: [1, 1] uint32 EP rank selector and zero.
        token_replica_group, cp_replica_group, head_replica_group, seq_replica_group, key_replica_group: the
            attention layer's groups.
        ep_replica_group, tp_replica_group: the MoE expert-parallel group and the shared expert's TP groups.
        dense_replica_group, dense_tp_size, shared_replica_group, shared_tp_size: consecutive-rank groups that split
            the dense MLP and the shared expert; None and 0 fall back to ``tp_replica_group`` and ``tp_size``.
        num_layers: layers in this stack.
        index_share_period, index_skip_topk_offset: layer ``i`` runs its own indexer when
            ``max(i - offset + 1, 0) % period == 0`` (with offset 0, ``i % period == 0``); period 0 makes every layer
            shared and needs ``prev_gather_indices``.
        cp_degree, head_shard, token_shard, key_size, num_heads, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank,
            idx_n_heads, idx_rope_dim, index_topk, spec, softmax_scale, kv_scale: as in the attention layer.
        rms_norm_eps, latent_norm_eps, idx_norm_eps: 1e-5 for the two layer norms, 1e-6 for the latent and indexer
            norms.
        ep_size, tp_size, top_k, routed_scaling_factor: the MoE layer's parallelism and routing.
        prev_gather_indices: [T_local, index_topk] int32 in true positions, the selection that enters a stack starting
            on a shared layer, or None.
        idx_pool_items, idx_pool_n_items, idx_pool_region_base, idx_pool_new_key_rows, idx_pool_unrot: the 1M point's
            chunk-dealt indexer pool tables, or None (see the attention layer).

    Returns:
        ``(out, kv_cache_all, idx_k_cache_all, ...)``: the residual stream leaving the last layer, the updated caches,
        then the sampling tail's outputs and last the ``kv_cache_hi*`` stacks that were passed.
    """
    T_shard, H = x_shard.shape
    num_full = idx_wq_b_all.shape[0]
    # Leading layers with no slot in the MoE stacks: 0, or the dense layers under moe_sparse_slots.
    moe_skip = _moe_layer_skip(num_layers, first_dense_layers, gamma_all.shape[0])
    moe_axes = [
        stack.shape[0]
        for stack in (
            router_weights_all,
            router_bias_all,
            routed_gate_up_weights_all,
            routed_gate_up_weights_scale_all,
            routed_down_weights_all,
            routed_down_weights_scale_all,
            shared_gate_up_weights_all,
            shared_gate_up_weights_scale_all,
            shared_down_weights_all,
            shared_down_weights_scale_all,
        )
    ]
    kernel_assert(
        all(axis == gamma_all.shape[0] for axis in moe_axes),
        f"{_ERROR_PREFIX} every MoE stack must have gamma_all's leading axis {gamma_all.shape[0]}, got {moe_axes}: "
        f"a stack one slot short would hand a layer its neighbour's weights",
    )
    # The latent stacks in layer order; a single stack when kv_cache_hi is not passed (see kv_cache_hi).
    latent_stacks = _latent_stacks(kv_cache_all, kv_cache_hi, kv_cache_hi2, kv_cache_hi3)

    _validate_giga_stack(
        num_layers=num_layers,
        index_share_period=index_share_period,
        index_skip_topk_offset=index_skip_topk_offset,
        layer_offset=layer_offset,
        num_full=num_full,
        attn_layer_axis=input_norm_w_all.shape[0],
        moe_layer_axis=gamma_all.shape[0],
        kv_cache_layer_axis=sum(stack.shape[0] for stack in latent_stacks),
        idx_cache_layer_axis=idx_k_cache_all.shape[0],
        has_prev_indices=prev_gather_indices is not None,
    )
    _validate_latent_stacks(kv_cache_all, kv_cache_hi, kv_cache_hi2, kv_cache_hi3)
    layer_stop = layer_stop or num_layers
    kernel_assert(
        0 <= layer_start < layer_stop <= num_layers,
        f"{_ERROR_PREFIX} layer window [{layer_start}, {layer_stop}) must lie in [0, {num_layers})",
    )
    kernel_assert(
        layer_start == 0
        or (
            prev_gather_indices is None
            and _is_full_layer(layer_start, index_share_period, index_skip_topk_offset, layer_offset)
        ),
        f"{_ERROR_PREFIX} a layer window must start at a full layer with no carried selection, got layer {layer_start}",
    )

    T_local = T_shard * cp_degree * head_shard
    hidden = x_shard
    # The residual stream's SBUF copy in the partition-spread shape (see _residual_add); None until the
    # first residual add produces one, so layer 0's input norm loads x_shard itself.
    sb_shape = spread_rows_shape(T_shard, H) or (T_shard, H)
    hidden_sb = None
    norm_reduce_ones = build_norm_reduce_ones(T_shard, H)
    # The reference threads ``topk_indices`` layer to layer as a plain value; so does this.
    # A leading layer axis switches on the per-layer diagnostic form; see the arg docs.
    per_layer_indices = prev_gather_indices is not None and len(prev_gather_indices.shape) == 3
    carried_indices = None if per_layer_indices else prev_gather_indices
    # The prior-index tables (local cache rows, snake gather index, key-position mask operands) that
    # ``carried_indices`` resolves to on this rank. Built at the FIRST shared layer of a group and reused
    # by the rest: a shared layer's selection, block table and CP shift are all the full layer's, so its
    # tables are byte-identical and rebuilding them in each layer was pure duplication. Reset
    # whenever the carried selection changes.
    carried_tables = None
    # The indexer's key-residency mask depends only on idx_positions[0] and this rank's key range, so it
    # is one tensor per STEP, not per layer: built here once and handed to every full layer, instead of
    # 21 in-loop recomputations on the Vector engine. Skipped entirely when no layer runs an indexer.
    idx_mask_add = None
    idx_block_rows = None
    # The chunk-dealt pool (idx_pool_items) scores the chunk loop's chunks, so it takes the same per-step mask with one
    # cut per region (that of the region's owner, idx_pool_region_base), built here once rather than in each of the 21
    # full layers.
    cdp = idx_pool_items is not None
    kernel_assert(
        cdp == (idx_pool_degree > 0) and (not cdp or (idx_pool_region_base is not None and idx_block_table is None)),
        f"{_ERROR_PREFIX} a pooled indexer (idx_pool_degree) is the chunk-dealt pool: it needs idx_pool_items, "
        f"idx_pool_region_base and the position-ordered indexer cache (no idx_block_table)",
    )
    if any(
        _is_full_layer(i, index_share_period, index_skip_topk_offset, layer_offset) for i in range(num_layers)
    ):
        # The legacy flat cache selects the pre-block path: contiguous shard, no deal.
        kernel_assert(
            idx_shard_degree == 1 or indexer_cache_is_blocked(idx_k_cache_all, stacked=True),
            f"idx_shard_degree={idx_shard_degree} needs the block-form indexer cache; the legacy flat "
            f"[idx_keys, idx_head_dim] per layer cache keeps the contiguous shard only",
        )
        idx_mask_add = indexer_residency_mask(
            idx_positions,
            indexer_cache_keys(idx_k_cache_all, stacked=True),
            spec,
            key_base=0,
            shard_degree=idx_shard_degree,
            block_size=idx_block_size,
            rank_offset=key_offset if idx_shard_degree > 1 else None,
            **({"region_bases": idx_pool_region_base} if cdp else {}),
        )
        # The indexer's logical->physical row table is per STEP too: it depends on the block table
        # only, which is layer-invariant. None leaves every indexer layer reading its cache in
        # position order, which is the behaviour before paging.
        if idx_block_table is not None:
            idx_keys = indexer_cache_keys(idx_k_cache_all, stacked=True)
            idx_block_rows = indexer_block_rows(idx_block_table, idx_keys, idx_keys // (idx_block_table.shape[1] - 1))

    # Long context (idx_keys >= TOPK_MAX_N a rank, 512k at cp 4): pin every collective operand of the layers that
    # the rings below do not cover -- every source, the ReduceScatter destinations, the HBM rows_tp -- to the kernel
    # end (utils/cc_pin.py). The key count is the cache's, not idx_k_cache_all.shape[1] (n_blocks in the block
    # form). Shorter contexts keep the unpinned graph: with pinning off nothing below emits an instruction.
    idx_cache_keys = indexer_cache_keys(idx_k_cache_all, stacked=True)
    long_context = idx_cache_keys >= TOPK_MAX_N
    reset_rope_table_cache()  # GLM52K_ROPE_TABLES_ONCE: no table survives from another trace
    open_pins(
        default=long_context,
        attn_window=_ATTN_PIN_WINDOW if idx_cache_keys >= _ATTN_PIN_WINDOW_MIN_KEYS else 0,
    )
    # Collective destinations rotate through kernel-lifetime rings (see cc_dst_ring).
    open_cc_dst_ring()
    ep_synced = False  # GLM52K_ATTN_EP_SYNC_FIRST
    for i in range(layer_start, layer_stop):
        next_layer()  # the attention pin window: release the operands that leave it
        # Every value here is a trace-time Python int, identical on every rank, so each subscript
        # below is plain HBM integer-indexing resolved during tracing -- not a runtime tensor branch.
        is_full = _is_full_layer(i, index_share_period, index_skip_topk_offset, layer_offset)
        # Index into the indexer stacks: how many FULL layers precede this one. Under the uniform
        # period this is i // period, but the real pattern is not uniform (layers 0..2 are full and
        # 3..5 are not), so it has to be counted rather than divided.
        f = _num_full_layers(i, index_share_period, index_skip_topk_offset, layer_offset)

        layer_prev_indices = None if is_full else (prev_gather_indices[i] if per_layer_indices else carried_indices)
        if not is_full and not per_layer_indices and carried_tables is None and carried_indices is not None:
            carried_tables = build_shared_prior_index(
                gather_indices_true=carried_indices,
                cp_pos_offset=cp_pos_offset,
                kv_cache=_latent_layer(latent_stacks, i),
                block_table=block_table,
                positions=positions,
                T_local=T_local,
                num_seqs_local=T_local // spec,
                spec=spec,
                index_topk=index_topk,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                cp_degree=cp_degree,
                io_dtype=x_shard.dtype,
            )
        layer_tables = carried_tables if (not is_full and not per_layer_indices) else (None, None, None)
        # A full layer hands out the tables its own attention resolved, so the group's shared layers
        # import them and build_shared_prior_index above is only reached for an externally supplied
        # selection (prev_gather_indices) that no layer in this trace resolved.
        exported_tables = [] if (is_full and not per_layer_indices) else None

        _layer_out = glm52_dsa_attention_decode_layer_kernel(
            x_shard=hidden,
            input_norm_w=input_norm_w_all[i],
            wq_a=wq_a_all[i],
            wkv_a=wkv_a_all[i],
            q_norm_w=q_norm_w_all[i],
            kv_norm_w=kv_norm_w_all[i],
            wq_b=wq_b_all[i],
            q_absorb_w=q_absorb_w_all[i],
            out_absorb_w=out_absorb_w_all[i],
            o_proj_w=o_proj_w_all[i],
            cos=cos,
            sin=sin,
            # A shared layer's checkpoint carries no indexer tensors at all, so it is handed none.
            idx_wq_b=idx_wq_b_all[f] if is_full else None,
            idx_wk=idx_wk_all[f] if is_full else None,
            idx_k_norm_w=idx_k_norm_w_all[f] if is_full else None,
            idx_k_norm_b=idx_k_norm_b_all[f] if is_full else None,
            idx_weights_proj=idx_weights_proj_all[f] if is_full else None,
            idx_cos=idx_cos if is_full else None,
            idx_sin=idx_sin if is_full else None,
            kv_cache=_latent_layer(latent_stacks, i),
            kv_new_slot_mapping=kv_new_slot_mapping,
            block_table=block_table,
            positions=positions,
            idx_k_cache=idx_k_cache_all[f] if is_full else None,
            idx_new_slot_mapping=idx_new_slot_mapping if is_full else None,
            idx_positions=idx_positions if is_full else None,
            key_offset=key_offset if is_full else None,
            idx_shard_degree=idx_shard_degree,
            idx_block_size=idx_block_size,
            cp_pos_offset=cp_pos_offset,
            token_replica_group=token_replica_group,
            cp_replica_group=cp_replica_group,
            head_replica_group=head_replica_group,
            seq_replica_group=seq_replica_group,
            key_replica_group=key_replica_group,
            block_replica_group=block_replica_group,
            qblock_replica_group=qblock_replica_group,
            latent_replica_group=latent_replica_group,
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
            prev_gather_indices=layer_prev_indices,
            prior_snake_rows=layer_tables[0],
            prior_key_pos=layer_tables[1],
            prior_natural_rows=layer_tables[2],
            idx_mask_add=idx_mask_add if is_full else None,
            idx_block_rows=idx_block_rows if is_full else None,
            pool_replica_group=pool_replica_group,
            idx_pool_degree=idx_pool_degree,
            idx_pool_tokens=idx_pool_tokens,
            # The chunk-dealt pool's tables (the per-region mask is idx_mask_add above), passed only when set so the
            # unpooled trace is unchanged.
            **(
                {
                    "idx_pool_items": idx_pool_items,
                    "idx_pool_n_items": idx_pool_n_items,
                    "idx_pool_region_base": idx_pool_region_base,
                    "idx_pool_new_key_rows": idx_pool_new_key_rows,
                    "idx_pool_unrot": idx_pool_unrot,
                }
                if cdp and is_full
                else {}
            ),
            # Delivered to SBUF: the residual add below is the only consumer, so the layer's own
            # shared_hbm output and its store + reload are skipped.
            output=nl.ndarray(sb_shape, dtype=hidden.dtype, buffer=nl.sbuf),
            x_shard_sb=hidden_sb,
            input_norm_reduce_ones=norm_reduce_ones,
            prior_tables_out=exported_tables,
        )
        # A full layer returns [num_seqs_local, spec, index_topk] and a shared one the flat
        # [T_local, index_topk] it was given; the flat form is what the next layer consumes.
        attn_out, selection, _kv_cache, _idx_cache = _layer_out
        if is_full:
            # A new selection: the tables resolved for the previous group no longer apply. The full layer's
            # own tables (exported by its attention core) are what the group's shared layers import.
            carried_tables = exported_tables[0] if exported_tables else None
        carried_indices = selection.reshape((T_local, index_topk))

        # the MoE norm adds the residual and the attention output itself.
        attn_rs = _dsa_layer.take_attn_rs()
        moe_x_parts = (hidden, attn_rs) if attn_rs is not None else None
        hidden, hidden_sb = _residual_add(hidden, attn_out, f"attn_L{i}", h_sb=hidden_sb)

        # ``mlp_layer_types[i]`` is "dense" for the first ``first_k_dense_replace`` layers of the
        # model and "sparse" after. A dense layer has no router and no routed expert at all, so it
        # runs the same kernel with ``dense_only`` and its own MLP weights in the shared slots.
        is_dense = i < first_dense_layers
        # The MoE stacks' slot of this layer. A dense layer reads no MoE weight; the shared slots only give
        # the layer kernel a shape to validate, so entry 0 serves.
        m = 0 if is_dense else i - moe_skip
        if ATTN_EP_SYNC_FIRST and not is_dense and not ep_synced:
            ep_synced = True
            sync_sb = nl.ndarray((1, 8), dtype=nl.int32, buffer=nl.sbuf)
            nisa.memset(dst=sync_sb, value=0)
            sync_src = nl.ndarray((1, 8), dtype=nl.int32, buffer=nl.private_hbm, name=unique_name("ep_sync_src"))
            sync_dst = nl.ndarray((ep_size, 8), dtype=nl.int32, buffer=nl.private_hbm, name=unique_name("ep_sync_dst"))
            nisa.dma_copy(dst=sync_src, src=sync_sb)
            pin(sync_src), pin(sync_dst)
            ncc.all_gather(srcs=[sync_src], dsts=[sync_dst], replica_group=ep_replica_group, collective_dim=0)
        moe_out, _affinities = glm52_moe_decode_layer_dp_tp_ep_kernel(
            x_shard=hidden,
            gamma=dense_gamma_all[i] if is_dense else gamma_all[m],
            router_weights=None if is_dense else router_weights_all[m],
            router_bias=None if is_dense else router_bias_all[m],
            routed_gate_up_weights=None if is_dense else routed_gate_up_weights_all[m],
            routed_gate_up_weights_scale=None if is_dense else routed_gate_up_weights_scale_all[m],
            routed_down_weights=None if is_dense else routed_down_weights_all[m],
            routed_down_weights_scale=None if is_dense else routed_down_weights_scale_all[m],
            dense_gate_weights=dense_gate_weights_all[i] if is_dense else None,
            dense_up_weights=dense_up_weights_all[i] if is_dense else None,
            dense_down_weights=dense_down_weights_all[i] if is_dense else None,
            shared_gate_up_weights=shared_gate_up_weights_all[m],
            shared_gate_up_weights_scale=shared_gate_up_weights_scale_all[m],
            shared_down_weights=shared_down_weights_all[m],
            shared_down_weights_scale=shared_down_weights_scale_all[m],
            shared_affinities=shared_affinities,
            rank_id=rank_id,
            shared_rank_id=shared_rank_id,
            dense_only=is_dense,
            ep_replica_group=ep_replica_group,
            tp_replica_group=tp_replica_group,
            ep_size=ep_size,
            tp_size=tp_size,
            top_k=top_k,
            eps=rms_norm_eps,
            routed_scaling_factor=routed_scaling_factor,
            return_affinities=False,
            output=nl.ndarray(sb_shape, dtype=hidden.dtype, buffer=nl.sbuf),
            # a full (indexer) layer sees another SBUF state at its MoE block and keeps the default load order
            gu_prefetch=0 if is_full else None,
            mlp_replica_group=dense_replica_group if is_dense else shared_replica_group,
            mlp_tp_size=dense_tp_size if is_dense else shared_tp_size,
            x_parts=None if is_dense else moe_x_parts,
        )
        hidden, hidden_sb = _residual_add(hidden, moe_out, f"moe_L{i}", h_sb=hidden_sb)


    close_cc_dst_ring()

    # The last residual add already produced a shared_hbm tensor, but it was written by a plain DMA
    # into an INTERNAL buffer. Copying it into a dedicated output keeps the kernel's IO tensors
    # distinct from the collective buffers the sub-kernels allocate.
    out = nl.ndarray((T_shard, H), dtype=x_shard.dtype, buffer=nl.shared_hbm, name=unique_name("giga_out"))
    out_sb = nl.ndarray((T_shard, H), dtype=out.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=out_sb, src=hidden)
    close_pins(out_sb)  # no-op when pinning is off
    nisa.dma_copy(dst=out, src=out_sb)

    # ═══════ Sampling tail: final norm -> lm_head -> top-k -> sampler, on device ═══════
    # After the last layer: the vocab-sharded lm_head, the full top-k and the sampler, with the hidden-state
    # AllGather and the logits all_to_all inside the trace. Its three outputs (topk_values [T_shard, k] bf16,
    # topk_indices [T_shard, k] uint32, tokens [T_shard, 1] uint32) follow every other output; which batch rows a
    # rank's tokens belong to is lm_head_row_owner.
    lm_head_outs = ()
    if layer_stop == num_layers:
        kernel_assert(
            lm_head_weight_t is not None and final_norm_w is not None and sampling_params is not None,
            f"{_ERROR_PREFIX} the sampling tail needs lm_head_weight_t, final_norm_w and sampling_params",
        )
        hops = tuple(lm_head_hops) if lm_head_hops else (ep_size,)
        tokens, topk_values, topk_indices = glm52_lm_head_tail(
            hidden_shard=hidden,
            final_norm_w=final_norm_w,
            lm_head_weight_t=lm_head_weight_t,
            sampling_params=sampling_params,
            ep_replica_group=ep_replica_group,
            ranks=ep_size,
            lm_head_hops=hops,
            eps=rms_norm_eps,
            k=lm_head_k,
            # the HF RMSNorm's roundings, and HF's top-p rule (keep the token whose cumulative
            # probability crosses top_p)
            norm_rounding="bf16",
            top_p_rule="keep_crossing",
        )
        # Republished into this kernel's own IO tensors, as ``out`` is.
        tokens_out = nl.ndarray(tokens.shape, dtype=tokens.dtype, buffer=nl.shared_hbm, name=unique_name("giga_tokens"))
        topk_values_out = nl.ndarray(
            topk_values.shape, dtype=topk_values.dtype, buffer=nl.shared_hbm, name=unique_name("giga_topk_values")
        )
        topk_indices_out = nl.ndarray(
            topk_indices.shape, dtype=topk_indices.dtype, buffer=nl.shared_hbm, name=unique_name("giga_topk_indices")
        )
        nisa.dma_copy(dst=tokens_out, src=tokens)
        nisa.dma_copy(dst=topk_values_out, src=topk_values)
        nisa.dma_copy(dst=topk_indices_out, src=topk_indices)
        lm_head_outs = (topk_values_out, topk_indices_out, tokens_out)

    # The later stacks of a split latent cache are mutated too, so they are returned after every other
    # output (hi, then hi2, then hi3), so no existing position moves. All empty by default.
    hi = latent_stacks[1:]
    return (out, kv_cache_all, idx_k_cache_all, *lm_head_outs, *hi)


_DGE_STATIC_OFFSET_LIMIT = 2**32
"""Exclusive bound on an indirect DMA's static byte offset: the DGE base-address immediate is 32 bits."""


def _latent_stacks(kv_cache_all: nl.NkiTensor, kv_cache_hi: nl.NkiTensor, kv_cache_hi2: nl.NkiTensor,
                   kv_cache_hi3: nl.NkiTensor = None) -> tuple:
    """The latent stacks in layer order. A third stack requires a second, a fourth a third."""
    kernel_assert(
        kv_cache_hi2 is None or kv_cache_hi is not None,
        f"{_ERROR_PREFIX} kv_cache_hi2 is the third stack of a split latent cache and needs kv_cache_hi",
    )
    kernel_assert(
        kv_cache_hi3 is None or kv_cache_hi2 is not None,
        f"{_ERROR_PREFIX} kv_cache_hi3 is the fourth stack of a split latent cache and needs kv_cache_hi2",
    )
    return tuple(stack for stack in (kv_cache_all, kv_cache_hi, kv_cache_hi2, kv_cache_hi3) if stack is not None)


def _latent_layer(stacks: tuple, layer: int) -> nl.NkiTensor:
    """Layer ``layer``'s paged latent cache, from whichever stack holds it.

    With one stack this is ``kv_cache_all[layer]``, the same view as an unsplit cache.
    """
    for stack in stacks:
        if layer < stack.shape[0]:
            return stack[layer]
        layer -= stack.shape[0]
    kernel_assert(False, f"{_ERROR_PREFIX} latent layer past the last stack")


def _validate_latent_stacks(kv_cache_all: nl.NkiTensor, kv_cache_hi: nl.NkiTensor, kv_cache_hi2: nl.NkiTensor,
                            kv_cache_hi3: nl.NkiTensor = None) -> None:
    """Reject a latent stack whose deepest layer the DGE cannot address (see ``kv_cache_hi``)."""
    for name, stack in (("kv_cache_all", kv_cache_all), ("kv_cache_hi", kv_cache_hi), ("kv_cache_hi2", kv_cache_hi2),
                        ("kv_cache_hi3", kv_cache_hi3)):
        if stack is None:
            continue
        kernel_assert(
            stack.shape[1:] == kv_cache_all.shape[1:] and stack.dtype == nl.uint8,
            f"{_ERROR_PREFIX} {name} {stack.shape} {stack.dtype} must be uint8 packed rows in kv_cache_all's "
            f"per-layer shape {kv_cache_all.shape[1:]}",
        )
        layer_bytes = math.prod(stack.shape[1:])
        last_start = (stack.shape[0] - 1) * layer_bytes
        kernel_assert(
            last_start < _DGE_STATIC_OFFSET_LIMIT,
            f"{_ERROR_PREFIX} {name}[{stack.shape[0] - 1}] starts at byte {last_start}, past the DGE's 32-bit "
            f"static offset (neuronx-cc NCC_IEIL015 / NCC_IBTN127); split the latent stack further "
            f"(kv_cache_hi, kv_cache_hi2)",
        )


def _is_full_layer(layer: int, index_share_period: int, index_skip_topk_offset: int, layer_offset: int = 0) -> bool:
    """Whether stack layer ``layer`` runs its own indexer.

    With ``index_skip_topk_offset == 0`` it is ``layer % period == 0``. Otherwise it is the reference's rule,
    ``max(layer - index_skip_topk_offset + 1, 0) % index_topk_freq == 0``, which is not uniform at the head of the
    model: at ``freq = 4, offset = 3`` layers 0-2 are full, 3-5 shared, and the ``F s s s`` groups start at layer 6.

    Args:
        layer: index within the stack.
        index_share_period: the checkpoint's ``index_topk_freq``; 0 means no layer runs an indexer (and
            ``prev_gather_indices`` must be given).
        index_skip_topk_offset: the checkpoint's ``index_skip_topk_offset``.
        layer_offset: the model layer of the stack's layer 0; the rule is on model position.
    """
    if index_share_period <= 0:
        return False
    model_layer = layer + layer_offset
    if index_skip_topk_offset == 0:
        return model_layer % index_share_period == 0
    return max(model_layer - index_skip_topk_offset + 1, 0) % index_share_period == 0


def _num_full_layers(
    num_layers: int, index_share_period: int, index_skip_topk_offset: int, layer_offset: int = 0
) -> int:
    """Count the stack's full layers, i.e. the required leading axis of the indexer stacks."""
    return sum(
        1 for i in range(num_layers) if _is_full_layer(i, index_share_period, index_skip_topk_offset, layer_offset)
    )


def _moe_layer_skip(num_layers: int, first_dense_layers: int, moe_layer_axis: int) -> int:
    """Leading stack layers that have no slot in the MoE stacks: 0, or the dense layers when the stacks omit them.

    A dense layer reads no router, routed-expert or shared-expert weight, so a store can drop its slots
    (``moe_sparse_slots``): at 78 layers that is 3 of 78 MoE slots, 131 MiB a rank. Both forms are accepted,
    so a store built with a slot for every layer still runs, and its trace is unchanged.

    Args:
        num_layers (int): Layers in the stack.
        first_dense_layers (int): Leading layers that use the dense MLP.
        moe_layer_axis (int): Leading axis of the MoE stacks.

    Returns:
        int: Subtract this from a sparse layer's stack index to get its MoE slot.
    """
    dense = min(first_dense_layers, num_layers)
    compact = max(num_layers - dense, 1)
    kernel_assert(
        moe_layer_axis in (num_layers, compact),
        f"{_ERROR_PREFIX} the MoE stacks must have leading axis num_layers={num_layers}, or {compact} "
        f"(num_layers - first_dense_layers, at least 1) with no slot for a dense layer; got {moe_layer_axis}",
    )
    return 0 if moe_layer_axis == num_layers else dense


def _validate_giga_stack(
    num_layers: int,
    index_share_period: int,
    index_skip_topk_offset: int,
    layer_offset: int,
    num_full: int,
    attn_layer_axis: int,
    moe_layer_axis: int,
    kv_cache_layer_axis: int,
    idx_cache_layer_axis: int,
    has_prev_indices: bool,
) -> None:
    """Reject stacks whose layer axes do not line up with the index-share pattern.

    Every check here is on a shape or a rank-invariant Python int, so all of it resolves at trace time.
    The failure these prevent is the quiet one: a leading axis one entry short silently reuses the
    wrong layer's weights, which produces a running kernel and a degraded golden rather than an error.

    Args:
        num_layers (int): Layers in the stack.
        index_share_period (int): Index-share period; 0 means every layer is shared.
        num_full (int): Leading axis of the indexer weight stacks.
        attn_layer_axis (int): Leading axis of the attention weight stacks.
        moe_layer_axis (int): Leading axis of the MoE weight stacks (checked by ``_moe_layer_skip``).
        kv_cache_layer_axis (int): Layers over every latent stack (``kv_cache_all`` plus any ``kv_cache_hi*``).
        idx_cache_layer_axis (int): Leading axis of ``idx_k_cache_all``.
        has_prev_indices (bool): Whether ``prev_gather_indices`` was supplied.
    """
    kernel_assert(num_layers >= 1, f"{_ERROR_PREFIX} num_layers={num_layers} must be at least 1")
    kernel_assert(
        index_share_period >= 0,
        f"{_ERROR_PREFIX} index_share_period={index_share_period} must be non-negative",
    )
    kernel_assert(
        attn_layer_axis == num_layers and kv_cache_layer_axis == num_layers,
        f"{_ERROR_PREFIX} per-layer stacks must have leading axis num_layers={num_layers}, got "
        f"attention={attn_layer_axis}, kv_cache={kv_cache_layer_axis}",
    )

    expected_full = _num_full_layers(num_layers, index_share_period, index_skip_topk_offset, layer_offset)
    kernel_assert(
        num_full == max(expected_full, 1) and idx_cache_layer_axis == max(expected_full, 1),
        f"{_ERROR_PREFIX} num_layers={num_layers} at index_share_period={index_share_period} has "
        f"{expected_full} full layers, so the indexer stacks need leading axis {max(expected_full, 1)} "
        f"(one dummy entry when there are none), got weights={num_full}, cache={idx_cache_layer_axis}",
    )
    # The condition is whether the stack's FIRST layer is full, not whether the period is zero. Those
    # coincide only for a uniform pattern: a stack over reference layers 3..77 (layer_offset=3, freq=4,
    # offset=3) has 18 full layers yet OPENS with three shared ones, so it needs an incoming selection
    # exactly as a period-0 stack does.
    starts_shared = not _is_full_layer(0, index_share_period, index_skip_topk_offset, layer_offset)
    kernel_assert(
        (not starts_shared) or has_prev_indices,
        f"{_ERROR_PREFIX} layer 0 of this stack is a shared layer at index_share_period="
        f"{index_share_period}, index_skip_topk_offset={index_skip_topk_offset}, "
        f"layer_offset={layer_offset}, so prev_gather_indices is required -- the reference raises the "
        f"same error ('Shared DSA layers require top-k indices from a previous full indexer layer')",
    )
    kernel_assert(
        starts_shared or not has_prev_indices,
        f"{_ERROR_PREFIX} layer 0 runs its own indexer at index_share_period={index_share_period}, "
        f"index_skip_topk_offset={index_skip_topk_offset}, layer_offset={layer_offset}, so "
        f"prev_gather_indices must not be supplied",
    )


__all__ = ["glm52_decode_giga_kernel"]
