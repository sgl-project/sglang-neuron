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

"""Kimi-K3 decode model giga-kernel: up to all 93 decoder layers in one NEFF.

The entry splits the layer range into AttnRes blocks. A block starts at a bank snapshot layer
(``layer % 12 == 0``), so 93 layers give 7 blocks of 12 layers and one block of 9 layers. The entry
calls ``decode_block_body`` once for each block. The residual stream and the AttnRes bank stay in
SBUF between the blocks, and the weight DMA gates continue across the block boundaries.

With the final weights, the entry also runs the model tail on chip: the final AttnRes mix over the
whole bank and the final RMSNorm. The output is then the ``lm_head`` input. With the ``lm_head`` shard,
the entry also gives the bf16 logits of the shard or the greedy token of each sequence
(the decode tail). The embedding and a non-greedy sampler stay outside the kernel.
"""

import nki
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from .utils.kernel_assert import kernel_assert
from .utils.trace_names import restart_calls
from .common import (
    P_MAX,
    WeightRow,
    attn_res_hT,
    load_vector_hT,
    new_ones,
    rmsnorm_hT,
    row_sums_after_ffn_output,
    store_hT_rows,
)
from .decode_block import (
    _KDA_FRONT_ROWS_MAX_TOKENS,
    ATTN_RES_BLOCK,
    FULL_ATTENTION_PERIOD,
    MLA_STAGE_SPLITS,
    NUM_LAYERS,
    SEL_LOAD_EXPERT_SLOTS,
    banks_snapshot,
    block_layer_counts,
    check_moe_sbuf_reduces,
    check_pair_shared_forms,
    check_x_world_gather,
    collective_sources,
    decode_block_body,
    layer_sources,
    mla_stage_unit_tokens,
    mla_staged,
    moe_option_sites,
    new_bank_rstd,
    pair_shared_expert_path,
)
from .tkg_lm_head_greedy import lm_head_greedy_hT
from .tkg_weight_dma_gate import set_weight_gate, weight_gates
from .tkg_sbuf_alloc import ExplicitSbuf, SbufTiles, check_identities, install_identities
from .tkg_kda import (
    KDA_STATE_DTYPES,
    kda_conv_body,
    kda_decode_attention,
    kda_default_readout,
    kda_options_body,
)
from .tkg_mla_block import (
    ROWS_GROUP_ROWS,
    ROWS_LOCAL_HEADS,
    ROWS_TWIN_SEQUENCES,
    mla_block_collective_hbm,
    mla_block_collective_sources,
    mla_cp_decode_attention,
    mla_rows_decode_attention,
)
from .tkg_mla_paged import LORA_NORM_EPS, mla_decode_attention
from .tkg_moe import (
    DP_SIZE,
    SCORE_GATHER_COLS,
    X_GATHER_PARITY_MAX_TOKENS,
    dense_mlp_decode,
    moe_options_in_use,
    x_gather_parity,
)
from .tkg_moe_dp import select_moe_body

MAX_POOLS = 24
"""Page pools that the entry takes: the cache shares one pool between layers 4k .. 4k+3."""
TAIL_TP_RANKS = 32
WORLD_TP_RANKS = 128
"""Ranks of the DP1 tail argmax gather (the world group, lm_head sharded V/128)."""
"""Ranks of the argmax gather of the decode tail (the TP group)."""


def model_layer_groups(first_layer: int, num_layers: int) -> tuple:
    """Return the AttnRes blocks ``(first layer, layer count)`` of a layer range.

    A new block starts at each bank snapshot layer.

    Example:
        >>> model_layer_groups(0, 16)
        ((0, 12), (12, 4))
    """
    groups = []
    start = first_layer
    end = first_layer + num_layers
    while start < end:
        stop = min(end, (start // ATTN_RES_BLOCK + 1) * ATTN_RES_BLOCK)
        groups.append((start, stop - start))
        start = stop
    return tuple(groups)


def shared_pool_of_layer(first_layer: int, num_layers: int) -> tuple:
    """Return the shared pool index of each layer of a range: layers 4k .. 4k+3 share one pool.

    The index counts from the pool of ``first_layer``.

    Example:
        >>> shared_pool_of_layer(0, 6)
        (0, 0, 0, 0, 1, 1)
    """
    base = first_layer // FULL_ATTENTION_PERIOD
    return tuple(i // FULL_ATTENTION_PERIOD - base for i in range(first_layer, first_layer + num_layers))


def _group_stack(stack, start: int, count: int, total: int):
    """Return the entries ``start .. start + count - 1`` of a stack of ``total`` layers of one type.

    A stack with a leading axis of 1 holds one tensor for all layers. It passes whole, and so does a
    stack that the group does not use or uses completely. None (a weight slot that the routed kernel does
    not read) stays None.
    """
    if stack is None:
        return None
    kernel_assert(stack.shape[0] in (1, total), "Stack a weight once for each layer of its type, or once")
    if count == 0 or count == total or stack.shape[0] != total:
        return stack
    return stack[start : start + count]


def _moe_group(moe_weights, start: int, count: int, total: int, routed_kernel: str):
    """Return the MoE weight slots of a block: ``_group_stack`` views, and whole TP128 expert stacks for sel_load.

    A slice of a TP128 expert stack carries the layer base as a static DMA offset, which passes 4 GiB in the late
    layers (NCC_IEIL015). The sel_load kernel takes the whole stack and the layer in ``expert_offset`` instead.
    """
    return tuple(
        t if routed_kernel == "sel_load" and i in SEL_LOAD_EXPERT_SLOTS else _group_stack(t, start, count, total)
        for i, t in enumerate(moe_weights)
    )


def _kda_body(
    conv_dtype: str,
    w_in_buffers: int,
    state_loads_first: bool,
    batched_conv_stores: bool = False,
    readout: str = None,
    conv_load_gather: bool = False,
    conv_stores_early: bool = False,
    state_loads_early: bool = False,
    state_dtype: str = "bf16",
    bf16_update_rows: bool = False,
    bf16_parts: tuple = (),
    bf16_rows_flat: bool = False,
    null_page_rows: int = 0,
    null_page: int = 0,
    paired_pages: bool = False,
):
    """Return the KDA decode body with the options that differ from the defaults bound (the plain body without any)."""
    kernel_assert(state_dtype in KDA_STATE_DTYPES, f"kda_state_dtype {state_dtype!r} must be one of {KDA_STATE_DTYPES}")
    options = {
        name: value
        for name, value in (
            ("bf16_parts", tuple(bf16_parts)),
            ("w_in_buffers", w_in_buffers),
            ("state_loads_first", state_loads_first),
            ("batched_conv_stores", batched_conv_stores),
            ("readout", readout),
            ("conv_load_gather", conv_load_gather),
            ("conv_stores_early", conv_stores_early),
            ("state_loads_early", state_loads_early),
            ("state_dtype", None if state_dtype == "bf16" else state_dtype),
            ("bf16_update_rows", bf16_update_rows),
            ("bf16_rows_flat", bf16_rows_flat),
            ("null_page_rows", null_page_rows),
            ("null_page", null_page),
            ("paired_pages", paired_pages),
        )
        if value
    }
    if not options:
        return kda_decode_attention if conv_dtype == "bf16" else kda_conv_body(kda_decode_attention, conv_dtype)
    if conv_dtype != "bf16":
        options["conv_dtype"] = conv_dtype
    return kda_options_body(kda_decode_attention, **options)


KDA_UPDATE_ROWS_STATE_DTYPES = ("bf16", "fp32")
"""The KDA state forms where ``kda_bf16_update_rows`` None (the default) turns the bf16 update rows on. Their twin tests
keep the bits; ``fp32_hilo`` keeps the option opt-in."""

KDA_UPDATE_ROWS_AUTO_SEQUENCES = (32,)
"""The sequences per call where ``kda_bf16_update_rows`` None (the default) turns the bf16 update rows on: the measured
decode form, 32 sequences per replica with the batched conv-state stores and gathers. Other batches keep the option opt-in
until they are measured."""


def kda_update_rows_default(
    n_seq: int,
    readout: str | None,
    state_dtype: str,
    bf16_parts: tuple,
    mx_w_in: bool,
    batched_conv_stores: bool,
    conv_load_gather: bool,
) -> bool:
    """Return the ``kda_bf16_update_rows`` form of a decode call that leaves it at None (the default).

    The bf16 update rows go on only in the measured form: the ``decoupled_bf16`` readout (``kda_default_readout(n_seq)``
    above one sequence, or set) at ``KDA_UPDATE_ROWS_AUTO_SEQUENCES`` sequences with ``kda_batched_conv_stores`` and
    ``kda_conv_load_gather``, a state form of ``KDA_UPDATE_ROWS_STATE_DTYPES``, no ``bf16_parts`` and a BF16 ``kda_w_in``.
    Their twin tests keep the bits of the output and the pools. Every other call keeps its program.

    Args:
        n_seq (int): Sequences of the call.
        readout (str): ``kda_readout`` of the call (None for the default of ``n_seq``).
        state_dtype (str): ``kda_state_dtype`` of the call.
        bf16_parts (tuple): ``kda_bf16_parts`` of the call.
        mx_w_in (bool): True when the call has ``kda_w_in_scale`` (an MXFP8 ``kda_w_in``).
        batched_conv_stores (bool): ``kda_batched_conv_stores`` of the call.
        conv_load_gather (bool): ``kda_conv_load_gather`` of the call.

    Returns:
        bool: True to run the bf16 update rows.

    Example:
        >>> kda_update_rows_default(32, None, "bf16", (), False, True, True)
        True
        >>> kda_update_rows_default(16, None, "bf16", (), False, True, True)
        False
    """
    form = kda_default_readout(n_seq) if readout is None else readout
    measured = n_seq in KDA_UPDATE_ROWS_AUTO_SEQUENCES and batched_conv_stores and conv_load_gather
    covered = form == "decoupled_bf16" and state_dtype in KDA_UPDATE_ROWS_STATE_DTYPES
    return measured and covered and not bf16_parts and not mx_w_in


def check_kda_bf16_rows_flat(rows_flat: bool, update_rows: bool) -> None:
    """Stop the trace when ``kda_bf16_rows_flat`` is set without the bf16 update rows of the call.

    Args:
        rows_flat (bool): ``kda_bf16_rows_flat`` of the call.
        update_rows (bool): the resolved ``kda_bf16_update_rows`` of the call (after ``kda_update_rows_default``).

    Example:
        >>> check_kda_bf16_rows_flat(True, True)
    """
    kernel_assert(
        not rows_flat or update_rows,
        "kda_bf16_rows_flat needs the bf16 update rows: set kda_bf16_update_rows, or leave it None in its default form",
    )


def _rows(tensor, start: int, count: int):
    """Return the rows ``start .. start + count - 1`` of a per-layer tensor (the whole tensor for no rows)."""
    if count == 0 or (start == 0 and count == tensor.shape[0]):
        return tensor
    return tensor[start : start + count]


def _mx_keys_options(mla_options: dict | None, mx_keys: str | None) -> dict | None:
    """Return ``mla_options`` with ``pack_mx_keys`` set to ``mx_keys``, or unchanged for None (the default program)."""
    if mx_keys is None:
        return mla_options
    return {**(mla_options or {}), "pack_mx_keys": mx_keys}


def _dr_keys_options(mla_options: dict | None, dr_keys: bool) -> dict | None:
    """Return ``mla_options`` with ``pack_dr_keys`` set, or unchanged for False (the default program)."""
    if not dr_keys:
        return mla_options
    return {**(mla_options or {}), "pack_dr_keys": True}


def _rope_ahead_options(mla_options: dict | None, rope_ahead: bool) -> dict | None:
    """Return ``mla_options`` with ``pack_rope_ahead`` set, or unchanged for False (the default program)."""
    if not rope_ahead:
        return mla_options
    return {**(mla_options or {}), "pack_rope_ahead": True}


@nki.jit
def kimi_k3_decode_model(
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
    kda_page_ids: nl.NkiTensor,
    mla_block_table: nl.NkiTensor,
    mla_slot: nl.NkiTensor,
    position: nl.NkiTensor,
    rank_info: nl.NkiTensor,
    tp_group: ReplicaGroup,
    dp_group: ReplicaGroup,
    world_group: ReplicaGroup,
    first_layer: int,
    eps: float,
    debug: bool = False,
    dma_gates: bool | str = True,
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
    mla_fp8_pages: bool = False,
    routed_kernel: str = "static",
    routed_reduce_dtype: str = "bf16",
    router_mode: str = None,
    router_dtype: str = "bf16",
    moe_router_weight: nl.NkiTensor = None,
    affinity_scaling: str = "pre_quant",
    affinity_hi_lo: bool = True,
    moe_sel_load_options: tuple = (),
    mla_packed_loop: bool = False,
    mla_pack_slots: int | None = None,
    mla_pack_tail_slots: int | tuple | None = None,
    mla_pack_tail_low_batch: bool = False,
    mla_pack_page_runs: int = 0,
    mla_pack_null_copy_vector: bool = False,
    mla_pack_null_runs: bool = False,
    mla_batched_cache_write: bool = False,
    mla_zeroed_pages: bool = False,
    mla_loop_first_static: bool = False,
    mla_scores_ahead: bool = False,
    mla_folded_reference: bool = False,
    mla_cp_rows_options: dict | None = None,
    mla_pair_pages: bool = False,
    moe_sbuf_reduces: tuple = (),
    lora_norm_eps: float = LORA_NORM_EPS,
    moe_early_loads: str = "all",
    moe_latent_load: str = "strided",
    moe_half_weight_buffers: bool = False,
    moe_dynamic_loop: bool = False,
    routing_form: str = None,
    shared_sharding: str = "tp128",
    latent_sharding: str = "tp128",
    moe_routed_exchange: str = "all_reduce",
    cc_source_pingpong: bool = True,
    moe_dst_pingpong: tuple = (),
    moe_active_expert_slots: int = 0,
    mla_cp: int = 0,
    dp_size: int = DP_SIZE,
    mla_options: dict = None,
    reduce_rows: int = None,
    mix_cache: bool = False,
    early_ffn_gate: bool = None,
    moe_active_expert_counts: bool = False,
    kda_conv_dtype: str = "bf16",
    moe_gate_up_half_major: bool = False,
    moe_shared_scale_buffer: bool = False,
    moe_ring_reclaim: bool = False,
    moe_half_weight_buffer_count: int | None = None,
    kda_w_in_buffers: int = 0,
    kda_state_loads_first: bool = False,
    kda_state_loads_early: bool = False,
    kda_batched_conv_stores: bool = False,
    kda_null_page: int = 0,
    moe_in_proj_form: str = "gathered",
    moe_peer_in_proj: nl.NkiTensor = None,
    ffn_tail_order: bool = False,
    attn_res_cached_bank_sums: bool = False,
    moe_routed_store_form: str = "relayout",
    ffn_weights_after_attn: bool = False,
    moe_glue_dmas: tuple = (),
    moe_direct_latent: bool = False,
    moe_router_rows_drop_selection: bool = False,
    moe_latent_gather_mx: bool = False,
    attn_res_early_sums: bool = False,
    kda_readout: str = None,
    kda_conv_load_gather: bool = False,
    moe_out_proj_form: str = "reduced",
    mla_pack_group_plans: bool = False,
    moe_routing_tiles: bool = False,
    mla_pack_pairs: bool = False,
    mla_stage_cc_split: str = "off",
    mla_stage_post: bool = False,
    mla_pack_pair_plans: bool = False,
    mla_pack_setup_ahead: bool = False,
    mla_pack_state_store_queues: int = 1,
    mla_current_after_rs: bool = False,
    mla_stage_pre: bool = False,
    moe_latent_shard_form: str = "one_hot",
    attn_res_kda_mlp_early_entries: int = 0,
    attn_res_kda_mlp_early_in_front: bool = False,
    moe_x_world_gather: bool = False,
    mla_pack_sum_after_copy: bool = False,
    attn_res_scalar_squares: int = 0,
    norm_scalar_squares: int = 0,
    attn_res_scalar_squares_order: bool = False,
    attn_res_scalar_products: int = 0,
    attn_res_bank_dma: str = None,
    norm_one_mul: bool = False,
    kda_conv_stores_early: bool = False,
    mla_rows_twin: bool = False,
    mla_pack_mx_keys: str | None = None,
    mla_pack_rope_ahead: bool = False,
    mla_pack_dr_keys: bool = False,
    kda_state_dtype: str = "bf16",
    kda_bf16_update_rows: bool | None = None,
    kda_w_in_scale: nl.NkiTensor = None,
    mla_in_proj_scale: nl.NkiTensor = None,
    mla_q_b_proj_scale: nl.NkiTensor = None,
    mla_kv_b_proj_scale: nl.NkiTensor = None,
    mla_o_proj_scale: nl.NkiTensor = None,
    moe_in_proj_scale: nl.NkiTensor = None,
    kda_w_out_scale: nl.NkiTensor = None,
    kda_bf16_parts: tuple = (),
    attn_res_tensor_sums: bool = False,
    attn_res_one_round_norm: bool = False,
    attn_res_tensor_sums_idle: bool = False,
    moe_x_path_act: bool = False,
    pair_shared_cc_dsts: tuple = (),
    kda_conv_keep: bool = False,
    moe_static_alt_tiles: bool | str = False,
    moe_pad_rows: nl.NkiTensor = None,
    kda_bf16_rows_flat: bool = False,
    attn_res_sums_after_moe_output: bool = False,
    kda_null_page_rows: int = 0,
    moe_preamble_buffers: int = 0,
    moe_preamble_reuse: bool = False,
    mla_rs_hold: bool = True,
):
    """Run the Kimi-K3 decoder layers of a range for one DP replica on one rank.

    The kernel runs the layers ``first_layer .. first_layer + L - 1``, where ``L`` is the leading
    dimension of ``norm_weights``. The AttnRes bank starts empty, so the range starts at a bank
    snapshot layer: ``first_layer = 0`` runs the model from the embeddings. The kernel updates the page
    pools in place and returns them, so that they alias the inputs. The arguments are those of
    ``kimi_k3_decode_block``, with more layers and pools, and the optional tail arguments.

    Dimensions:
        H: hidden size, 7168. L: layers in the range, 1 to 93 (``first_layer + L <= 93``).
        B: sequences of the replica, 1 to 32.

    Args:
        stream: ``[B, H]`` bf16 stream at the first layer, one row for each sequence of the replica.
        norm_weights: ``[L, 2, H]`` bf16 input and post-attention norm weights, in hT order.
        attn_res_weights: ``[L, 2, H]`` fp32 AttnRes score weights (``norm * proj``) of the
            attention and MLP mixes, in hT order.
        kda_*: the KDA weights (``tkg_kda.KDA_WEIGHT_NAMES``), stacked ``[n_kda, ...]`` over the
            KDA layers of the range, or ``[1, ...]`` for one tensor that all KDA layers share.
            ``kda_w_in_scale`` (optional, the same stacking) holds the scales of an MXFP8 ``kda_w_in``
            (``kda_decode_attention``).
        mla_*: the MLA weights (``tkg_mla_paged.MLA_WEIGHT_NAMES``), stacked ``[n_mla, ...]`` or ``[1, ...]``.
            ``mla_in_proj_scale`` (optional, the same stacking) holds the scales of an MXFP8 ``mla_in_proj``
            (``mla_decode_attention``).
        moe_in_proj_scale: None (default, a BF16 ``moe_in_proj``), or the scales of an MXFP8 MoE in-projection, stacked
            as ``moe_in_proj`` (``tkg_moe.MX_IN_PROJ_SHAPES``): ``moe_in_proj`` then stacks the MXFP8 latent down and
            shared gate / up columns, run MXFP8 x MXFP8 (``tkg_moe.latent_moe_decode`` ``in_proj_scale``). It needs
            an FP16 router. The values change.
        kda_w_out_scale: None (default, a BF16 ``kda_w_out``), or the scales of an MXFP8 padded-lane ``kda_w_out`` stack
            (``tkg_kda.kda_w_out_form``: ``kda_w_out`` then stacks ``[n, 1, 128, 4 H]`` uint8 and the scales
            ``[n, 128, 1, H]``): the KDA output projection of every KDA layer runs MXFP8 x MXFP8. The values change.
        kda_bf16_parts: the ``tkg_kda.KDA_BF16_PARTS`` that every KDA layer runs in BF16 (an accuracy study; the
            values change). () (the default) keeps the program.
        moe_preamble_buffers: K in 1 to 3 extra whole weight buffers of the static expert loop for the first K
            projections of each MoE call (``decode_block.decode_block_body`` ``moe_preamble_buffers``). At B = 4 on the
            1M loop path (3 whole buffers) K = 1 does not fit SBUF (NCC_EGCA111). The values do not change; 0 (default)
            keeps the program.
        moe_preamble_reuse: with ``moe_preamble_buffers``, each extra buffer also takes one more projection of the next
            expert (``decode_block.decode_block_body`` ``moe_preamble_reuse``). The values do not change; False
            (default) keeps the program.
        moe_*, expert_*: the latent MoE weights of ``routed_kernel`` (``tkg_moe.moe_weight_names``),
            stacked ``[n_moe, ...]`` or ``[1, ...]`` (see ``decode_block.kimi_k3_decode_block`` for ``sel_load``).
        dense_*: the dense MLP weights (``tkg_moe.DENSE_WEIGHT_NAMES``), stacked ``[1, ...]``.
        kda_page_ids: ``[n_kda, B, 1]`` int32 state page of each KDA layer and sequence.
        mla_block_table: ``[n_mla, B, NB]`` int32 local block table of each MLA layer and sequence.
        mla_slot: ``[n_mla, B, 1]`` int32 slot of the current row of each MLA layer and sequence.
        position: ``[B, 1]`` int32 position of the current token of each sequence.
        rank_info: ``[1, 4]`` int32 (global rank, DP rank, TP rank, 0).
        tp_group, dp_group, world_group: the TP32, DP4 and world replica groups.
        first_layer: global index of the first layer.
        eps: RMSNorm epsilon.
        debug: also return the stream after each layer and the input and output of each body (tests).
        dma_gates: gate the weight DMAs of each body on the collectives of the previous body
            (``"per_queue"``: see ``decode_block.decode_block_body``).
        pool_of_layer: static index of the page pool of each layer of the range (see
            ``kimi_k3_decode_block``). ``shared_pool_of_layer`` gives the shared layout, 24 pools for
            93 layers. None gives each layer its own pool (24 layers or fewer).
        mla_loop_path: let the MLA layers take the loop path, for a block table of more than 16 local
            blocks (up to 256, 1M context). The entry then gives explicit SBUF addresses to the tiles
            that live through the loops (see ``tkg_sbuf_alloc``).
        mla_packed_loop: with ``mla_loop_path``, the MLA layers run one loop over the key groups of all
            sequences (``mla_decode_attention(packed_loop=True)``). The output is bitwise the same.
        mla_pack_slots: items of one packed iteration, a power of two up to 64 (None: 16 at 256K+ tables with
            B >= 4, else 8).
        mla_pack_tail_slots: None, or the items of one iteration of the packed tail loop at 16 sequences and up, or a
            pair such as (32, 8) for two tail loops (``mla_decode_attention(pack_tail_slots=...)``, ``decode_block``);
            bitwise the same output.
        mla_pack_tail_low_batch: also take ``mla_pack_tail_slots`` at 8 sequences and fewer (``decode_block``);
            bitwise the same output. False (default) keeps the programs of 8 sequences and fewer.
        mla_pack_page_runs: 0 (default), or 2 or 4: the packed loop loads the 4 pages of a key group with one DMA per
            run of that many pages (``decode_block``), for block tables whose groups are runs of pages (2: the
            128-row view of 256-row FP8 pages; 4: of 512-row pages, which 4 needs: the MLA body refuses it on other
            pools); bitwise the same output, and NaN for a sequence with a group that is not made of runs.
        mla_pack_null_copy_vector: with ``mla_packed_loop`` and ``mla_zeroed_pages``, the packed loop fixes the null
            table entries of a key group on VectorE only (``decode_block``). This fixes a trn3 hang of the 2-server
            verify at that cross-engine wait. The values do not change. False (default) keeps the program.
        mla_pack_null_runs: with ``mla_pack_page_runs``, a padded sequence (position 0, a zero block table row)
            counts as page runs, so its rows are finite, not NaN (``decode_block``). The rows of real sequences do
            not change. False (default) keeps the program.
        mla_batched_cache_write: each MLA layer (each staged group of 8 above 8 sequences) writes its current cache
            rows with one vector-indirect DMA, not one DMA per sequence, which the compiler runs one after another
            (``decode_block``). The rows that a rank does not own go to their own rows of the null page 0, not all to
            row 0. The outputs and the owned rows of the pools do not change; only the null page holds other bytes.
            Not with ``mla_pair_pages`` or ``mla_cp``. False (default) keeps the program.
        kv_pool_i: ``[P, 73728]`` bf16 page pool i (``[P, 147456]`` for 512-row pages). Pools ``0 .. k-1`` are
            used, where k is the number of pools.
        final_attn_res_weight: optional ``[H]`` fp32 score weight of the final AttnRes mix
            (``output_attn_res_norm * output_attn_res_proj``), in hT order. The kernel mixes the bank of
            the whole range and the stream after the last layer.
        final_norm_weight: optional ``[H]`` bf16 final RMSNorm weight (``model.norm``), in hT order.
            The norm runs after the final mix.
        lm_head_weight: optional ``[H, V_s]`` bf16 vocab shard of ``lm_head`` of this TP rank, ``[in, out]``
            (the transpose of the checkpoint rows ``t * V_s .. (t + 1) * V_s - 1``). The kernel
            multiplies the first output by it.
        greedy: with ``lm_head_weight``, return the greedy token of each sequence (the greedy
            sampler: the largest logit, and the smallest vocab id among equal logits).
        return_logits: with ``lm_head_weight``, return the bf16 logits ``[B, V_s]`` of the shard.
        mla_fp8_pages: the MLA layers keep float8_e4m3fn pages in their bf16 pools (see
            ``decode_block.kimi_k3_decode_block``).
        moe_sel_load_options: ``sel_load``: (name, value) pairs of extra ``moe_sel_load_tp_start`` options, for
            example ``long_stream``, ``chunk_tokens`` or ``sbuf_reserve`` (``decode_block.decode_block_body``). The
            values do not change; () (default) keeps the program.
        routed_kernel: routed-expert kernel of the MoE layers, ``static`` (default), ``sel_load`` or ``tp_dense``
            (see ``decode_block.kimi_k3_decode_block``).
        routed_reduce_dtype: add type of the routed all-reduce of the MoE layers, ``bf16`` (default) or ``fp32``.
        router_mode: router form of the MoE layers (see ``decode_block.kimi_k3_decode_block``).
        router_dtype, moe_router_weight: the FP16 router of the MoE layers (see ``decode_block.kimi_k3_decode_block``),
            ``moe_router_weight`` stacked ``[n_moe, 128, 56, 7]`` (``[n_moe, 128, 56, 28]`` with ``fp16_local``) over
            the MoE layers of the range, or ``[1, ...]``.
        affinity_scaling: ``sel_load``: ``pre_quant`` (default) or ``post_down`` (``tkg_moe.AFFINITY_SCALINGS``).
        affinity_hi_lo: ``sel_load``: True (default) or False, one quantized term (``tkg_moe.latent_moe_decode``).
        mla_zeroed_pages: the MLA loop path reads its cache pages with no clamp. Set it only when the caller
            zeroes each MLA page at allocation (see ``tkg_mla_paged.mla_decode_attention``).
        mla_loop_first_static: with ``mla_cp``, the CP128 MLA loop path runs its first iteration as static code
            (``tkg_mla_block.mla_block_attention`` ``loop_first_static``); the verify of the deployment takes the same key.
            The bits do not change. False (default) keeps the program.
        mla_scores_ahead: with ``mla_cp``, the CP128 MLA loop computes the weights of each pair of key groups before
            the sums of the pair before it (``tkg_mla_block.mla_block_attention`` ``scores_ahead``); the verify of the
            deployment takes the same key. The bits do not change. False (default) keeps the program.
        mla_folded_reference: with ``mla_cp``, the CP128 MLA loop adds the softmax reference in its rope matmul
            (``tkg_mla_block.mla_block_attention`` ``folded_reference``); the verify of the deployment must take the
            same key. False (default).
        mla_cp_rows_options: with ``mla_cp``, keys of the rows form for the CP128 MLA decode
            (``tkg_mla_block.CP_ROWS_OPTIONS``, ``rows_dr`` and its forms; see ``tkg_mla_block.mla_cp_decode_attention``). The
            verify of the deployment must take the same keys. None (default) keeps the program.
        mla_pair_pages: the FP8 MLA pages hold the token-pair layout of ``pair_pages`` (see
            ``tkg_mla_paged.mla_decode_attention``). CP32 MLA only.
        moe_sbuf_reduces: the MoE all-reduces that run in SBUF on the loop path: ``("routed",)`` for the routed
            all-reduce (``decode_block.collective_sources``). Default: all in HBM.
        lora_norm_eps: the epsilon of the q_a and kv_a RMSNorms of the MLA layers (``lora_norm_eps`` of
            ``tkg_mla_paged.mla_decode_attention``). The default is 1e-5; the checkpoint's KimiRMSNorm uses 1e-6.
        moe_early_loads: the MoE layers whose expert loop issues its first weight and scale loads before its latent
            work (``decode_block.decode_block_body``): ``all`` (default), ``kda`` (the layers of a KDA attention; in
            an MLA layer the early loads slow down the MLA output reduce-scatter) or ``none``.
        moe_latent_load: the load of the HBM latent gather in the base MoE body (``decode_block.decode_block_body``,
            ``tkg_moe.LATENT_LOADS``). The values do not change; ``strided`` (default) keeps the program.
        moe_half_weight_buffers: above 8 sequences, half-size weight buffers in the static expert loop in place of
            its 2 whole ones (``decode_block.decode_block_body``; the same SBUF). The values do not change; False
            (default) keeps the program.
        moe_dynamic_loop: the opt-in dynamic expert loop of the MoE layers (``tkg_moe.dynamic_expert_loop``). Only
            the active local experts run, at B <= 2 (T <= 8 tokens, ``tkg_moe.DYNAMIC_EXPERT_LOOP_MAX_TOKENS``)
            and off the MLA loop path (``mla_loop_path`` False, a context of 64K or less with CP32). Elsewhere the
            static loop runs, so the NEFF of B >= 3, or of a longer context, is the NEFF without it. The values do
            not change.
        routing_form: TP128 kernels: the form of the top-16 routing (``tkg_moe.ROUTING_FORMS``); None (default)
            takes the default of ``tkg_moe.latent_moe_decode`` (``index_first``).
        shared_sharding, latent_sharding: the sharding of the shared expert and of the latent projections of the MoE
            layers (see ``decode_block.kimi_k3_decode_block``).
        moe_routed_exchange: the routed exchange of the MoE layers (see ``decode_block.kimi_k3_decode_block``).
        cc_source_pingpong: two sets of the collective sources (see ``decode_block.kimi_k3_decode_block``).
        moe_dst_pingpong: the gather sites of the MoE layers whose persistent HBM destination alternates by layer
            parity (``decode_block.collective_sources`` ``dst_pingpong``): ``("gather",)``, the latent world gather, or
            ``("gather", "x")``, also the DP x gather. With one destination for all layers, the runtime keeps the
            start handshake of each gather after the first: it skips it only when the destination does not overlap
            the one of the previous gather of the group. A site whose gather does not run in HBM (off the MLA loop
            path) keeps one destination. Each named HBM destination adds one tensor of its size. The values do not
            change; () (default) keeps the program.
        moe_active_expert_slots: K, the opt-in active-expert slots of the MoE layers (``active_expert_slots`` of
            ``moe_tkg_decode``): only the first K active local experts run, in straight-line slots, and every
            expert in a one-trip loop when more are active. The values do not change. One dispatch point
            (``tkg_moe.routed_expert_path``) picks the path of a layer: the dynamic loop when
            ``moe_dynamic_loop`` applies, else the slots when K > 0, else the static loop. A layer never runs
            both. K > 0 needs the base MoE body (``tkg_moe.base_moe_body``: no MoE option such as an FP16 router,
            not the DP4 x TP32 body) and at most 4 sequences; else, or with K above 7, the call fails. 0 (default)
            changes nothing.
        mla_cp: 0, or 128 for the CP128 MLA layers (see ``decode_block.kimi_k3_decode_block``).
        reduce_rows: None, or the rows of the verify step of the deployment (``decode_block.decode_block_body``).
        mix_cache: keep the rstd of each AttnRes bank entry from its push (``decode_block.decode_block_body``). Always on
            with ``dp_size`` 1.
        early_ffn_gate: start the weights of each attention during the MoE tail (``decode_block.decode_block_body``).
            None turns it on with ``dp_size`` 1.
        ffn_tail_order: order the next layer's AttnRes bank sums and the next KDA body's small weight loads after the
            output collective of each feed-forward body (``decode_block.decode_block_body``). The values do not change.
            False (default) keeps the program.
        moe_active_expert_counts: debug, with the slots only: also return the number of active local experts of
            this rank in each MoE layer of the range (``tkg_moe.active_expert_counts``). A count above
            ``moe_active_expert_slots`` marks a layer whose overflow loop ran in this step. Asking for the counts
            when the layers do not run the slots (K = 0, or the dynamic loop) fails. False (default) changes
            nothing.
        kda_conv_dtype: form of the KDA short convolution, ``bf16`` (default) or ``fp32`` (``tkg_kda.CONV_DTYPES``).
        moe_in_proj_form, moe_peer_in_proj: the in-projection form of the MoE layers (``tkg_moe.IN_PROJ_FORMS``):
            ``gathered`` (default) keeps the program; ``dp_local`` takes ``moe_peer_in_proj`` stacked
            ``[n_moe, 128, 56, 140]`` BF16 over the MoE layers of the range and the ``dp_local`` ``moe_in_proj``
            ``[n_moe, 128, 56, 96]`` and ``moe_vectors`` ``[n_moe, 1, 63]`` (``latent_moe_dp_local_rank_weights``).
            ``fp32`` needs the ``kda_params`` of ``build_kda_rank_weights(..., conv_dtype="fp32")``.
        moe_gate_up_half_major: the EP128 static gate/up weights are stored half-major (``gate_up_half_major`` of
            ``moe_tkg_decode``): one DMA descriptor per partition for each half-projection load. The values do not
            change; False (default) reads the standard layout.
        moe_shared_scale_buffer: the EP128 static expert loop keeps its gate/up and down MX scales in one buffer of 4
            slots (``decode_block_body`` ``moe_shared_scale_buffer``): 21 KB less SBUF per partition across the loop.
            The values do not change; False (default) keeps the program.
        moe_ring_reclaim: with ``moe_half_weight_buffers``, one more half weight buffer of the static expert loop at
            an explicit address over feed-forward collective tiles that no layer touches from its latent world gather
            to the end of its expert loop (``decode_block_body`` ``moe_ring_reclaim``). The values do not change;
            False (default) keeps the program.
        moe_half_weight_buffer_count: with ``moe_half_weight_buffers``, the compiler-placed half weight buffers of the
            static expert loop (``decode_block_body``). None (default) keeps 4.
        kda_w_in_buffers: SBUF buffers of the ``kda_w_in`` ring of each KDA layer (``kda_decode_attention``
            ``w_in_buffers``). 0 (default) keeps the program.
        kda_state_loads_first: trace the KDA recurrent-state loads before the input projection, so the state tile
            gets its own bytes (``kda_decode_attention`` ``state_loads_first``). False (default) keeps the program.
        kda_state_loads_early: trace the KDA recurrent-state loads and their selects right after the first fills of the
            ``kda_w_in`` ring, with SP-queue schedule edges, so the loads land during the attention mix and the selects
            run in the idle VectorE time of the input projection (``kda_decode_attention`` ``state_loads_early``). Not
            with ``kda_state_loads_first``. The values do not change. False (default) keeps the program.
        kda_batched_conv_stores: store the KDA conv states of each group of sequences with one scatter DMA
            (``kda_decode_attention`` ``batched_conv_stores``). The write page of each sequence with a write must be
            its own, or ``kda_null_page_rows``. False (default) keeps the program.
        kda_null_page: 0 (default), or the KDA null page that the caller keeps for no request (a page other
            than 0 that no request holds, for example block 1). A sequence at it reads and writes like an off one
            (``kda_decode_attention`` ``null_page``): zero reads, zero conv-state scatter rows, and every off access
            goes to it, not to page 0. The rows of the other pages do not change.
        kda_conv_load_gather: load the KDA conv states of each group of sequences with one gather DMA on the Pool
            queue (``kda_decode_attention`` ``conv_load_gather``). The values do not change. False (default) keeps
            the program.
        attn_res_cached_bank_sums: keep the rstd of each AttnRes bank entry from its push, so the mixes, the final mix
            included, sum no squares of the bank (``decode_block.decode_block_body``). The values do not change. It
            does nothing with ``mix_cache`` or one row per replica. False (default) changes nothing.
        moe_routed_store_form: how the EP128 static expert loop fills the routed all-reduce source
            (``decode_block.decode_block_body``, ``tkg_moe.ROUTED_STORE_FORMS``): ``relayout`` (default) keeps the
            program; ``pe`` writes it from the expert loop through TensorE transposes, with no relayout DMA. The values
            do not change.
        ffn_weights_after_attn: the weight DMAs of each feed-forward body start after the residual add that reads the
            attention output, not during the flight of the attention output collective (``decode_block.decode_block_body``;
            with ``dma_gates`` on). The values do not change. False (default) keeps the program.
        moe_glue_dmas: TP128 MoE layers: names of ``tkg_moe.GLUE_DMA_FORMS``, the forms of the small DMAs between
            the MoE stages (``decode_block.decode_block_body``). The EP128 static MoE layers (the base body) take
            ``replica_rows`` and ``split_gather`` only. The values do not change; () (default) keeps the program.
        moe_direct_latent: the EP128 static expert loop reads the gathered latent tile in place, without its
            SBUF-to-SBUF copy DMA (``decode_block.decode_block_body``). The values do not change; False (default) keeps
            the program.
        moe_router_rows_drop_selection: the base MoE body (the EP128 ``static`` kernel, no MoE option) on the MLA loop
            path: the rows of the latent world gather leave out the 7 selection values (21 FP32 words in place of 28,
            25 % fewer bytes), and each rank adds the router bias of all 896 experts to the gathered scores
            (``tkg_moe.latent_moe_decode`` ``router_rows_drop_selection``). It needs ``moe_vectors``
            ``[n_moe, 1, 931]`` FP32: the 35 columns of the base layout, then the router correction bias of all 896
            experts in expert order (``tkg_moe.ALL_ROUTER_BIAS_COLS``, the same on every rank); the call fails
            without them, and with another MoE body. The values do not change; False (default) keeps the program.
        moe_latent_gather_mx: with ``moe_router_rows_drop_selection`` and ``moe_latent_load`` ``pe32``, on the static
            expert loop of the base MoE body: the latent world gather carries the MXFP8 expert input, quantized by the
            rank that computes each 32-column latent block, in one UINT32 [8, 2 B * 4] block per rank (64 bytes per
            token in place of 112; ``tkg_moe.latent_moe_decode`` ``latent_gather_mx``), and the receiving ranks
            load it straight into the expert input with no latent transpose or quantize. It needs ``moe_in_proj``
            ``[n_moe, 128, 56, 135]`` BF16 in the block layout (``tkg_moe.MX_IN_PROJ_COLS``: rank r < 112 holds
            latent-down columns [32 r, 32 r + 32) in columns 7..38, ranks 112..127 zeros there, the shared gate and up
            columns after them; ``moe_decode_torch.latent_moe_mx_rank_weights``). The values do not change; False
            (default) keeps the program.
        moe_x_world_gather: the EP128 MoE layers (base body) and the dense layer gather the DP replicas' hidden over the
            world group through HBM, each rank its 4 partition rows of its replica (``decode_block.decode_block_body``),
            on the MLA loop path with DP4. The destination gets the bytes of the DP gather. The values do not change;
            False (default) keeps the program. It needs ``"x"`` in ``moe_dst_pingpong`` and ``mla_stage_cc_split``
            "off": the other forms do not fit the collective tables (``decode_block.check_x_world_gather``). It
            fails closed at 8 sequences per replica and fewer, where the x gather has no HBM pair. Precondition (the
            kernel cannot check it): the world gather takes each partition of a replica from another TP rank, so the
            bits match the DP gather only when the 32 TP ranks of a replica hold the same hidden, that is the same
            replicated weights and all-reduces that give every rank the same bytes (reduce once, then copy; not a
            single-step all-reduce that each rank sums in its own order).
        moe_x_path_act: with ``moe_x_world_gather``, each layer's HBM x gather store, world gather slice and gather
            destination load run on the Activation hardware DGE queue (``decode_block.decode_block_body``), not behind
            the KDA recurrent-state stores on the SP queue. The values do not change; False (default) keeps the program.
        moe_routing_tiles: TP128 MoE layers with ``moe_in_proj_form`` dp_local: the [T, 16] top-16 routing outputs take
            explicit region addresses below the routed kernel's buffers (``decode_block.decode_block_body``), so the
            first weight gather does not wait for a write-after-read on them. The values do not change; False (default)
            keeps the program.
        attn_res_early_sums: with the kept bank rstd (``attn_res_cached_bank_sums``, or ``mix_cache``), the block mixes
            take their stream squares first and sum some bank rows early, where VectorE is idle
            (``decode_block.decode_block_body``). The values do not change. It does nothing with one row per replica.
            False (default) keeps the program.
        kda_readout: the form of the readout and the state matmuls of every KDA layer, one of
            ``tkg_kda.KDA_READOUTS`` (``kda_decode_attention`` ``readout``). None (default) keeps the program: the
            form of ``tkg_kda.kda_default_readout(B)``, ``reference`` at B = 1 and ``decoupled_bf16`` above.
            ``decoupled_bf16`` at B = 1 runs the operation order of the programs of B >= 2, so a sequence gets the
            same KDA outputs and new states in a B = 1 program as in a larger one (the decode buckets). It costs about
            some time in each KDA layer at B = 1. At B >= 2 it is the default form and keeps the program.
        moe_out_proj_form: the out-projection form of the MoE layers (``tkg_moe.OUT_PROJ_FORMS``). ``reduced``
            (default) keeps the program. ``column`` (the TP128 sel_load layers on the MLA loop path, with
            ``moe_sbuf_reduces`` ``routed_tp128``) takes the ``column`` ``moe_out_proj`` stack ``[n_moe, 76, 7168]``
            and ``moe_vectors`` with the 3,584 latent norm weights after the columns of the in-projection form
            (``latent_moe_column_rank_weights``); the output does not keep the bits of ``reduced``.
            ``reduced_wide`` keeps the weights and the output of ``reduced`` when the routed all-reduce adds in the
            same order at the larger message (a diagnostic). ``column_shared_ag`` takes the stacks of ``column`` and
            gives its output, with the routed all-reduce of ``reduced`` and the shared activation in its own world
            all-gather during the routed experts.
        mla_pack_group_plans: above 8 sequences, with ``mla_packed_loop`` on the loop path, build the items of the
            packed loop of each MLA group of 8 once for the MLA layers of a block (``decode_block.decode_block_body``),
            not in each layer. Bitwise the same output. False (default) keeps the program.
        mla_pack_pairs: at 32 sequences and more, with ``mla_packed_loop`` on the loop path, each pair of MLA groups
            of 8 runs one packed loop over its 16 sequences (``decode_block.decode_block_body``): half the loops, and
            fewer null slots at short contexts. Bitwise the same output. Fails closed below 32 sequences, with an odd
            group count or with ``mla_pack_group_plans``. False (default) keeps the program.
        mla_stage_cc_split: above 8 sequences, with ``mla_packed_loop`` on the loop path, issue the query all-gather
            ("ag") or the query all-gather and the state reduce-scatter ("ag_rs") of the staged MLA groups once for
            each unit of groups (a pair with ``mla_pack_pairs``, else a group), each as soon as its unit reaches it
            (``decode_block.decode_block_body``), so it flies under the work of the next unit. The all-gather moves the
            same bytes; the reduce-scatter adds the same values, in an order that the runtime may pick by message
            size. "off" (default) keeps the program.
        mla_stage_post: above 8 sequences, the staged MLA groups of each MLA layer run the work after the state
            reduce-scatter once for all groups: one value projection on 3 x B partitions and one o_proj on all B
            sequences (``decode_block.decode_block_body``). Bitwise the same output. False (default) keeps the
            program.
        mla_pack_pair_plans: with ``mla_pack_pairs``, build the items of the packed loop of each pair of MLA groups once
            for the MLA layers of a block (``decode_block.decode_block_body``), not in each layer. Bitwise the same
            output. False (default) keeps the program.
        mla_pack_setup_ahead: the staged MLA groups of the packed loop with a plan for each loop (``mla_pack_pairs`` with
            ``mla_pack_pair_plans``, or ``mla_pack_group_plans``) run the setup of every loop of a layer before its
            first loop (``decode_block.decode_block_body``). Bitwise the same output. False (default) keeps the program.
        mla_pack_state_store_queues: the staged MLA groups of the packed loop store the FP32 state of each loop before
            its reduce-scatter with this many DMAs on as many DMA queues, 2 or 3 (``decode_block.decode_block_body``).
            Bitwise the same output. 1 (default) keeps the program.
        mla_current_after_rs: the MLA layers of the packed loop with an FP32 HBM state add the current key once after the
            state reduce-scatter (``decode_block.decode_block_body``). The FP32 order of the state sums changes, so the
            output is not bitwise the same (GPU-parity class). False (default) keeps the program.
        mla_stage_pre: above 8 sequences, the staged MLA groups of each MLA layer run the work before the query gather
            once for all groups (``decode_block.decode_block_body``). Bitwise the same output. False (default) keeps the
            program.
        moe_latent_shard_form: the base MoE body at most 128 tokens: ``columns`` builds the transposed latent shard of
            the latent norm with 28 one-hot matmuls, a copy and a transpose in place of one matmul for each token
            (``decode_block.decode_block_body``). Bitwise the same output. "one_hot" (default) keeps the program.
        attn_res_kda_mlp_early_entries: with the plain AttnRes mix (no kept bank rstd) above 4 sequences, the MLP mix
            of each KDA layer sums its first this-many bank entries right after the attention norm, where VectorE waits
            for the KDA in-projection (``decode_block.decode_block_body``). The values do not change. Not with
            ``mix_cache``, ``attn_res_cached_bank_sums`` or ``attn_res_early_sums``, and above 4 sequences only with
            ``attn_res_kda_mlp_early_in_front``. 0 (default) keeps the program.
        attn_res_kda_mlp_early_in_front: with ``attn_res_kda_mlp_early_entries``, the KDA body traces those sums after
            its conv-state index ops and gathers, and the conv-state scatter keeps its rows and sources in persistent
            tiles (``decode_block.decode_block_body``), so neither the in-projection front nor the MLP mix waits behind
            the sums or the scatter. Above 16 sequences it needs ``kda_batched_conv_stores``: the block keeps the
            bytes of the scatter's views by reading them after the feed-forward norm. The values do not change. False
            (default) keeps the program.
        mla_pack_sum_after_copy: with ``mla_packed_loop`` and the interleaved slot pipeline, the VectorE weight sum add
            of slot j - 2 issues after the VectorE latent key copy of tile 0 of slot j + 2
            (``tkg_mla_paged.mla_decode_attention`` ``pack_sum_after_copy``). Bitwise the same output. False (default) keeps the
            program.
        attn_res_scalar_squares: with the plain AttnRes mix above 4 sequences, the sums of squares of the last this-many
            sequences of the stream of each mix, and of the bank entries of the attention mixes and the final mix, run on
            ScalarE, beside the VectorE sums of the others (``decode_block.decode_block_body``). Each row keeps its bits. Not with
            ``mix_cache``, ``attn_res_cached_bank_sums`` or ``attn_res_early_sums``. 0 (default) keeps the program; 20 is
            the value to use at 32 sequences.
        norm_scalar_squares: above 4 sequences, the same for the sums of squares of each RMSNorm after a mix (and the
            final norm). 0 (default) keeps the program; 9 is the value to use at 32 sequences.
        attn_res_scalar_squares_order: with ``attn_res_scalar_squares``, the conv-state transposes of each KDA body
            follow the ones-sum matmuls of its attention mix, and the bank push of a block's last layer follows the
            mix's last ScalarE square (``decode_block.decode_block_body``), so the scheduler keeps the order of the
            plain-mix program there. The values do not change. False (default) keeps the program.
        attn_res_scalar_products: with the plain AttnRes mix above 4 sequences, the products of the last this-many
            sequences of each bank entry after the first run on ScalarE in every mix with 2 or more bank entries, and in
            the final mix (``decode_block.decode_block_body``). VectorE keeps the other products and every add in
            candidate order, so the values do not change. Not with ``mix_cache``, ``attn_res_cached_bank_sums`` or
            ``attn_res_early_sums``. 0 (default) keeps the program.
        attn_res_bank_dma: None (default) keeps the program. ``act`` or ``sp`` (``common.BANK_DMAS``): the bank-entry
            loads of a DRAM AttnRes bank go on that hardware DGE queue at DMA priority 0, beside the expert stream
            (``decode_block.decode_block_body`` ``attn_res_bank_dma``). The values do not change.
        norm_one_mul: without ``mix_cache``, the weight multiply of each RMSNorm after an AttnRes mix is one
            instruction (``decode_block.decode_block_body`` ``norm_one_mul``). The values do not change. False
            (default) keeps the program.
        kda_conv_stores_early: with ``kda_batched_conv_stores``, the KDA conv-state scatters run in the KDA body, before
            the recurrent-state stores (``kda_decode_attention`` ``conv_stores_early``), so the MLP mix of the layer
            does not wait behind the recurrent-store chain. The output and the pools do not change. False (default)
            keeps the program.
        mla_rows_twin: test only, the identity twin of the CP32 rows-form verify (``verify_model`` ``mla_k1_rows``):
            the MLA layers run ``tkg_mla_block.mla_rows_decode_attention`` in place of ``mla_decode_attention``. It
            changes the per-row sums. ``tkg_mla_block.ROWS_TWIN_SEQUENCES`` sequences for each replica (calls of up to 8
            rows: 16 runs two), the CP32 MLA on the loop path, and no packed loop, page runs, pair pages, batched cache
            write or MLA options. False (default) keeps the program.
        mla_pack_mx_keys: None (default, keeps the program), "single" or "split": with ``mla_packed_loop`` and the
            interleaved slot pipeline, the latent part of each score of the packed loop is an MX matmul of FP8 queries
            and keys (``tkg_mla_paged.mla_decode_attention`` ``pack_mx_keys``). "single" rounds the query to e4m3 as the GPU decode
            does; "split" keeps about the BF16 query precision with two e4m3 parts. The output is not bitwise the same.
            The MLA body refuses every other form.
        mla_pack_rope_ahead: with ``mla_packed_loop`` and the carried interleaved slot pipeline (not carried in the
            main loop only) with transpose-mode rope keys, each packed slot traces the rope key transposes and copy of
            the next slot right after its rope matmul, between its P@V matmuls, and the copy before its exp
            (``tkg_mla_paged.mla_decode_attention`` ``pack_rope_ahead``). Bitwise the same output. Not with ``mla_pack_mx_keys``.
            The MLA body refuses every other form. False (default) keeps the program.
        mla_pack_dr_keys: with ``mla_packed_loop`` and the interleaved slot pipeline on transpose-mode keys, the latent
            part of each score of the packed loop takes 2 double_row matmuls of e4m3 queries and the FP8 keys of 2-byte
            key transposes instead of 4 BF16 x FP8 matmuls (``tkg_mla_paged.mla_decode_attention`` ``pack_dr_keys``); the query is
            rounded to e4m3 as the GPU decode does. The output is not bitwise the same. Not with ``mla_pack_mx_keys``.
            The MLA body refuses every other form. False (default) keeps the program.
        kda_state_dtype: the form of the KDA recurrent state in the pages (``tkg_kda.KDA_STATE_DTYPES``): ``bf16``
            (default, keeps the program), ``fp32`` (the GPU's fp32 state; ``decoupled_bf16`` predicts with one bf16
            copy of it as the stationary) or ``fp32_hilo`` (the fp32 state with bf16 high and low stationaries).
            An fp32 state takes 196608 bytes of each KDA page at 3 local heads, so the pools must be at least
            101760 bf16 elements wide: the 512-row form ``[P, 147456]`` (``kda_decode_attention`` ``state_dtype``).
            It changes the page layout, so the caller's KV-cache layout must follow it.
        kda_bf16_update_rows: with the ``decoupled_bf16`` KDA readout (the default above one sequence), the state update
            of every KDA layer takes its bf16 high and low rows from one bf16 TensorE transpose per 4 sequences, in
            place of an fp32 identity matmul per sequence, and adds them in one broadcast matmul per head in place of
            two (``kda_decode_attention`` ``bf16_update_rows``). The outputs and the pools do not change. No effect at
            the other readouts. None (default) turns it on only in the measured form (``kda_update_rows_default``:
            ``decoupled_bf16`` at 32 sequences with ``kda_batched_conv_stores`` and ``kda_conv_load_gather``, ``bf16``
            or ``fp32`` state, no ``kda_bf16_parts``, no ``kda_w_in_scale``) and keeps the program of every other call.
            True sets it at every form; False keeps the program of the calls without it.
        kda_bf16_rows_flat: with the bf16 update rows on (``kda_bf16_update_rows`` True, or its default form), each
            transpose of the rows reads its data, the stationary operand, with one free dimension
            (``kda_decode_attention`` ``bf16_rows_flat``): the same bytes and bits, in the access pattern form that the
            backend verifier takes. It needs the rows (the trace stops otherwise). False (default) keeps the program.
        attn_res_tensor_sums: above 4 sequences on trn3, the AttnRes mixes of the blocks take the GPU form: the
            weighted sums on TensorE, the bank rstd kept from the pushes, and one rounding of each mix and its norm
            (``decode_block.decode_block_body``). The final mix keeps its form. Not bitwise the same (GPU parity). Not
            with ``mix_cache`` (or one DP replica, which turns it on), ``attn_res_cached_bank_sums`` or
            ``attn_res_early_sums``. False (default) keeps the program.
        attn_res_one_round_norm: the GPU-parity norm of every AttnRes mix of the blocks, without TensorE sums (form
            F0): each mix keeps its sums and returns its FP32 accumulator, and its norm rounds once
            (``decode_block.decode_block_body``). The final mix keeps its form. Not bitwise the same (GPU parity). Not
            with ``mix_cache`` (or one DP replica, which turns it on): its norms take weight rows. False (default) keeps
            the program.
        attn_res_tensor_sums_idle: ``attn_res_tensor_sums`` only at the mixes where TensorE is idle in decode (form F1):
            the MLA-layer mixes and the KDA attention mixes take the TensorE weighted sums; the KDA MLP mixes keep their
            sums and early path; every mix has the GPU-parity norm (``decode_block.decode_block_body``). The final mix
            keeps its form. Not bitwise the same (GPU parity). Not with ``attn_res_tensor_sums``, and with the same
            checks as that key. False (default) keeps the program.
        pair_shared_cc_dsts: names of ``common.PAIR_SHARED_CC_DST_SITES``, the collective destinations that go to LNC1
            shared HBM: ``moe_out`` (the world output all-reduce of the MoE and dense layers), ``moe_routed`` (the world
            routed all-reduce of the MoE layers where it runs in HBM), ``moe_gather`` (the world latent all-gather of the
            MoE layers where it runs in HBM; with ``"gather"`` in ``moe_dst_pingpong``, only the first of its two
            destinations; its start rests on the distance-2 order of ``common.PAIR_SHARED_CC_DST_SITES``, which both cores
            of a stack running the same program and the program's collectives (one stream, a non-empty collective of the
            group that does not touch the destination before each gather) must give) and ``kda_ar`` (the KDA output
            all-reduce over the attention group). The two cores of an HBM stack then keep one copy of each named destination, which the
            collective writes once
            (``decode_block.collective_sources``). The sources and every other tensor stay private, and the values do not
            change. It needs the neuronx-cc option --lnc1-with-shared-mem and a runtime with pair-shared HBM (one process
            per core); without the option the compiler keeps the tensors private. trn3 only. It fails closed with the
            all-to-all-v routed exchange and with every MoE expert path but the base body's static loop with
            ``moe_routed_store_form`` "pe" (the others write per-core tensors in shared HBM), with the DP4 x TP32 MoE body
            and with ``mla_stage_cc_split`` other than "off" (``decode_block.check_pair_shared_forms``). A named site whose
            collective does not run in these tensors in a form (an SBUF latent gather or routed all-reduce, the output
            all-reduce of a column ``moe_out_proj_form``) shares nothing. ``moe_gather`` fails with
            ``moe_latent_gather_mx`` (its UINT32 latent gather stays private) and with ``moe_router_rows_drop_selection``
            (a shared 21-word latent gather destination is not measured). () (default) keeps the program.
        kda_conv_keep: keep the KDA conv-state store sources until their stores have read them, and chain the
            persistent conv-state tile rewrites to the stores of the KDA layers before, across the blocks
            (``decode_block.decode_block_body`` ``kda_conv_keep``). Placement and sync only: the values do not change.
            False (default) keeps the program.
        moe_static_alt_tiles: the EP128 static expert loop (base MoE body) takes its gate sigmoid, up tanh, down PSUM
            copy and activation eviction tiles from two buffers each, so that under SBUF pressure the compiler cannot
            stack them on one address (``decode_block.decode_block_body``): True all of them, "act" only the sigmoid /
            tanh tiles, "act_copy" those and the PSUM copies. The values do not change; False (default) keeps the
            program.
        moe_pad_rows: None (default) keeps the program. The HBM int32 ``[B, 1]`` padding-row input of the step (1 for a
            padding row, 0 for a real row; ``decode_block.decode_block_body`` ``moe_pad_rows``): the MoE layers route
            those rows to no expert. Needs a MoE body that takes the flags (``takes_pad_rows``: the DP4 x TP32 body;
            the TP128 bodies route every token of the world, so they do not take a replica's flags).
        attn_res_sums_after_moe_output: the AttnRes bank sums of the layer after each MoE layer (and of the final mix)
            follow the consumer of the routed partial of the base MoE body in the schedule (its routed store DMA, or
            its routed all-reduce from an SBUF source; ``decode_block.decode_block_body``
            ``row_sums_after_ffn_output``), so VectorE runs the PSUM copies of the column-group partial
            (``moe_routed_store_form`` ``pe``) before them. A schedule edge, not a semaphore wait: the values do not
            change. Not with ``ffn_tail_order``. False (default) keeps the program.
        kda_null_page_rows: 0 (default) keeps the program. Else the KDA page of every invalid row (a pad row or a row with no
            KDA block, the KDA null page of the server), which no valid row has. With ``kda_batched_conv_stores``, each
            sequence at this page writes its conv rows to its own rows of the page (``kda_decode_attention``
            ``null_page_rows``), so no conv-state scatter writes one pool row twice. The outputs and the other pages do not
            change; only the null page holds other bytes. It needs ``B * 3 * Hl`` conv rows of 384 elements in a page
            (all B at 147456-element pages). No effect without ``kda_batched_conv_stores``.
        mla_rs_hold: True (default) runs the state and output reduce-scatters of each MLA layer from persistent HBM
            tensors in two parity sets (``decode_block.collective_sources`` ``mla_rs_hold``), also on the MLA loop
            path, where the output all-gather then also runs in HBM (the form of the static path). An MLA layer then
            writes a reduce-scatter source only after the reduce-scatters of the MLA layer before it. This is the
            kernel workaround for the defect of the switch-mesh reduce-scatter completion in Neuron runtime
            2.35 (``decode_block.mla_rs_hold_sources``). The values do not change. False keeps the program without
            the hold (the SBUF reduce-scatters of the loop path), for a runtime with the fix.

    Returns:
        The stream ``[B, H]`` after the last layer, then (with ``moe_active_expert_counts``) the INT32 counts
        ``[n_moe, 1]`` of the MoE layers of the range in layer order, then (with ``debug``) the stream after each layer
        ``[L, B, H]`` and the per-layer body inputs and outputs ``[L, 4, B, H]``, then the k page pools.
        With the final weights, the first output is the stream after the final mix and the final norm
        (the ``lm_head`` input). The debug outputs do not change. With ``lm_head_weight``, the tokens
        ``[B, 1]`` int32 (``greedy``) and the logits ``[B, V_s]`` bf16 (``return_logits``) follow the first
        output, before the counts and the debug outputs.
    """
    restart_calls()
    B, H = stream.shape
    if kda_bf16_update_rows is None:
        kda_bf16_update_rows = kda_update_rows_default(
            B,
            kda_readout,
            kda_state_dtype,
            kda_bf16_parts,
            kda_w_in_scale is not None,
            kda_batched_conv_stores,
            kda_conv_load_gather,
        )
    check_kda_bf16_rows_flat(kda_bf16_rows_flat, kda_bf16_update_rows)
    mix_cache = mix_cache or dp_size == 1
    early_ffn_gate = dp_size == 1 if early_ffn_gate is None else early_ffn_gate
    n_tiles = H // P_MAX
    num_layers = norm_weights.shape[0]
    kernel_assert(first_layer + num_layers <= NUM_LAYERS, f"The model has {NUM_LAYERS} layers")
    kernel_assert(not (mla_pair_pages and mla_cp), "mla_pair_pages needs the CP32 MLA body")
    kernel_assert(not (mla_pack_page_runs and mla_cp), "mla_pack_page_runs needs the CP32 MLA body")
    kernel_assert(
        not mla_rows_twin
        or (
            B in ROWS_TWIN_SEQUENCES
            and not mla_cp
            and dp_size == DP_SIZE
            and mla_loop_path
            and not (mla_packed_loop or mla_pack_page_runs or mla_pair_pages or mla_batched_cache_write or mla_options)
        ),
        f"mla_rows_twin is the twin of the CP32 rows-form verify: it needs B in {ROWS_TWIN_SEQUENCES} (calls of up to 8 rows), "
        f"DP{DP_SIZE}, the CP32 MLA on the loop path, and no packed loop, page runs, pair pages, batched cache write or "
        "MLA options",
    )
    kernel_assert(
        mla_stage_cc_split in MLA_STAGE_SPLITS,
        f"mla_stage_cc_split must be one of {tuple(MLA_STAGE_SPLITS)}, got {mla_stage_cc_split!r}",
    )
    split_sites = MLA_STAGE_SPLITS[mla_stage_cc_split]
    pair_shared = check_pair_shared_forms(
        pair_shared_cc_dsts,
        dp_moe=shared_sharding == "dp4xtp32",
        a2av_moe=None if moe_routed_exchange == "all_reduce" else moe_routed_exchange,
        mla_stage_split=split_sites,
        moe_expert_path=pair_shared_expert_path(
            routed_kernel,
            moe_options_in_use(routed_kernel, router_mode, router_dtype, routed_reduce_dtype),
            moe_routed_store_form,
            moe_dynamic_loop,
            moe_active_expert_slots,
        ),
    )
    # The latent gather keys make their own HBM latent gather tensors (tkg_moe.moe_collective_hbm).
    kernel_assert(
        "moe_gather" not in pair_shared or not moe_latent_gather_mx,
        "pair_shared_cc_dsts moe_gather: the UINT32 latent gather of moe_latent_gather_mx stays private",
    )
    kernel_assert(
        "moe_gather" not in pair_shared or not moe_router_rows_drop_selection,
        "pair_shared_cc_dsts moe_gather is not proven with moe_router_rows_drop_selection: its 21-word latent gather "
        "destination is not measured in shared HBM",
    )
    if moe_x_world_gather:
        check_x_world_gather(mla_stage_cc_split, "x" in tuple(moe_dst_pingpong))
        # Up to X_GATHER_PARITY_MAX_TOKENS sequences the DP x gather alternates two SBUF destinations and has no
        # HBM pair to gather into.
        kernel_assert(
            mla_loop_path and dp_size == DP_SIZE and not x_gather_parity(B),
            f"moe_x_world_gather needs the HBM x gather pair: the MLA loop path, the DP{DP_SIZE} layout and more than "
            f"{X_GATHER_PARITY_MAX_TOKENS} sequences per replica, got {B} on DP{dp_size} (mla_loop_path {mla_loop_path})",
        )
    kernel_assert(not moe_x_path_act or moe_x_world_gather, "moe_x_path_act needs moe_x_world_gather")
    kernel_assert(
        not attn_res_kda_mlp_early_in_front or B <= _KDA_FRONT_ROWS_MAX_TOKENS or kda_batched_conv_stores,
        f"attn_res_kda_mlp_early_in_front above {_KDA_FRONT_ROWS_MAX_TOKENS} sequences needs kda_batched_conv_stores: "
        "the block keeps the bytes of the conv-state scatter by reading its views after the feed-forward norm",
    )
    kernel_assert(
        lm_head_weight is not None or not (greedy or return_logits),
        "greedy and return_logits need lm_head_weight",
    )
    kernel_assert(banks_snapshot(first_layer), "The AttnRes bank starts empty: start at a bank snapshot layer")
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
    kernel_assert(sorted(set(pool_of_layer)) == list(range(n_pools)), "pool_of_layer must use each pool 0 .. k-1")
    pools = all_pools[:n_pools]
    kernel_assert(all(p is not None for p in pools), "Give each page pool that pool_of_layer uses")
    layer_pools = tuple(pools[i] for i in pool_of_layer)

    layer_streams = None
    layer_debug = None
    if debug:
        layer_streams = nl.ndarray((num_layers, B, H), dtype=stream.dtype, buffer=nl.shared_hbm, name="layer_streams")
        layer_debug = nl.ndarray((num_layers, 4, B, H), dtype=stream.dtype, buffer=nl.shared_hbm, name="layer_debug")

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
    expert_counts = None
    if moe_active_expert_counts:
        expert_counts = nl.ndarray(
            (totals[2], 1), dtype=nl.int32, buffer=nl.shared_hbm, name="moe_active_expert_counts"
        )

    region = None
    if mla_loop_path or moe_options_in_use(routed_kernel, router_mode, router_dtype, routed_reduce_dtype):
        region = ExplicitSbuf()
        install_identities(region)
    # The bank holds one entry for each snapshot layer of the range, up to 8 for 93 layers.
    n_entries = sum(1 for i in range(first_layer, first_layer + num_layers) if banks_snapshot(i))
    bank_shape = (P_MAX, n_entries, B, n_tiles)
    if region is None:
        bank = (nl.ndarray(bank_shape, dtype=stream.dtype, buffer=nl.sbuf), 0)
    else:
        bank = (region.heap(bank_shape, stream.dtype, "bank"), 0)
    bank_sums = attn_res_cached_bank_sums and not mix_cache and B > 1
    if mix_cache or bank_sums or attn_res_tensor_sums or attn_res_tensor_sums_idle:
        bank = (*bank, new_bank_rstd(n_entries, B, region))
    s = stream
    gate = ()
    starts = [0, 0, 0, 0]
    # DP1 runs the attention bodies and the tail over the world group; DP4 keeps the TP32 groups.
    attn_group = world_group if dp_size == 1 else tp_group
    tail_tp = WORLD_TP_RANKS if dp_size == 1 else TAIL_TP_RANKS
    tail_ranks = tail_tp if lm_head_weight is not None and greedy else 0
    sites = moe_option_sites(routed_kernel, router_mode, router_dtype, routed_reduce_dtype, mla_loop_path)
    x_parity = sites is None and dp_size != 1 and x_gather_parity(B)
    moe_body = select_moe_body(shared_sharding, latent_sharding, tp_group, moe_routed_exchange)
    dp_moe = getattr(moe_body, "moe_sharding", "tp128") == "dp4xtp32"
    # fp16_local in the base MoE body: the local router logits ride the packed rows of the DP x gather.
    x_logits = router_dtype == "fp16_local" and sites is None and dp_size == DP_SIZE and not dp_moe
    sources = collective_sources(
        B,
        n_tiles,
        stream.dtype,
        region,
        tail_ranks,
        moe_options=sites,
        x_pair=x_parity,
        x_hbm=mla_loop_path and sites is None and dp_size != 1 and not x_parity and not dp_moe,
        mla_stage=mla_staged(B, mla_loop_path),
        dp_moe=dp_moe,
        a2av_moe=getattr(moe_body, "a2av_expert_form", None),
        sbuf_routed_reduce="routed" in check_moe_sbuf_reduces(moe_sbuf_reduces),
        sbuf_routed_options="routed_tp128" in check_moe_sbuf_reduces(moe_sbuf_reduces),
        pingpong=cc_source_pingpong,
        dst_pingpong=moe_dst_pingpong,
        x_world=moe_x_world_gather,
        x_logits=x_logits,
        mla=not (mla_cp or mla_rows_twin),
        mla_cp=mla_cp,
        dp_size=dp_size,
        reduce_rows=reduce_rows,
        moe_out_proj_form=moe_out_proj_form,
        **(
            {"mla_stage_split": split_sites, "mla_stage_unit_tokens": mla_stage_unit_tokens(mla_pack_pairs)}
            if split_sites
            else {}
        ),
        **({"pair_shared_cc_dsts": pair_shared} if pair_shared else {}),
        **({"moe_gather_cols": SCORE_GATHER_COLS} if moe_router_rows_drop_selection else {}),
        **({"moe_gather_mx": True} if moe_latent_gather_mx else {}),
        **({"ring_reclaim": True} if moe_ring_reclaim else {}),
        **({"mla_rs_hold": True} if mla_rs_hold else {}),
    )
    if mla_rows_twin:
        # The sources of the loop-path k1 verify: region heap tiles with reserved reduce-scatter sources, and the
        # query and output gathers in persistent HBM. Cleared at the start, so no tile takes their bytes first.
        # Above 8 sequences the twin runs groups of 8, each with its own set.
        rows_kw = {"q_hbm": True, "out_ag_call": True}
        group_rows = [min(ROWS_GROUP_ROWS, B - g0) for g0 in range(0, B, ROWS_GROUP_ROWS)]
        rows_sets = []
        for g, n in enumerate(group_rows):
            name = "_rows_twin" if len(group_rows) == 1 else f"_rows_twin{g}"
            rows_set = {
                **mla_block_collective_sources(
                    n, ROWS_LOCAL_HEADS, SbufTiles(region, heap=True), with_rs=True, **rows_kw
                ),
                **mla_block_collective_hbm(n, ROWS_LOCAL_HEADS, name=name, **rows_kw),
            }
            for tile in [rows_set["qkv"], rows_set["qkv_dst"], *rows_set["rs"], rows_set["ag_call"]]:
                nisa.memset(dst=tile, value=0.0)
            rows_sets.append(rows_set)
        sources = {**sources, "mla_rows": rows_sets[0] if len(rows_sets) == 1 else rows_sets}
    mla_body = mla_cp_decode_attention if mla_cp else mla_decode_attention
    if mla_rows_twin:
        mla_body = mla_rows_decode_attention
    if kda_conv_keep:
        # The conv-state stores of the last two KDA layers, across the blocks (decode_block_body).
        sources = {**sources, "kda_conv_waits": []}
    for start, count in model_layer_groups(first_layer, num_layers):
        li = start - first_layer
        counts = block_layer_counts(start, count)
        kda_k, mla_k, moe_k, dense_k = starts
        n_kda, n_mla, n_moe, n_dense = counts
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
            _rows(kda_page_ids, kda_k, n_kda),
            _rows(mla_block_table, mla_k, n_mla),
            _rows(mla_slot, mla_k, n_mla),
            position,
            rank_info,
            attn_group,
            dp_group,
            world_group,
            start,
            eps,
            _kda_body(
                kda_conv_dtype,
                kda_w_in_buffers,
                kda_state_loads_first,
                kda_batched_conv_stores,
                kda_readout,
                kda_conv_load_gather,
                kda_conv_stores_early,
                kda_state_loads_early,
                state_dtype=kda_state_dtype,
                bf16_update_rows=kda_bf16_update_rows,
                bf16_parts=kda_bf16_parts,
                bf16_rows_flat=kda_bf16_rows_flat,
                null_page_rows=kda_null_page_rows,
                null_page=kda_null_page,
            ),
            mla_body,
            moe_body,
            dense_mlp_decode,
            None if layer_streams is None else _rows(layer_streams, li, count),
            None if layer_debug is None else _rows(layer_debug, li, count),
            dma_gates,
            gate,
            mla_fp8_pages=mla_fp8_pages,
            region=region,
            mla_loop_path=mla_loop_path,
            mla_packed_loop=mla_packed_loop,
            mla_pack_slots=mla_pack_slots,
            mla_pack_tail_slots=mla_pack_tail_slots,
            mla_pack_tail_low_batch=mla_pack_tail_low_batch,
            mla_pack_page_runs=mla_pack_page_runs,
            mla_pack_null_copy_vector=mla_pack_null_copy_vector,
            mla_pack_null_runs=mla_pack_null_runs,
            **({"mla_batched_cache_write": True} if mla_batched_cache_write else {}),
            sources=sources,
            last_block=start + count == first_layer + num_layers,
            routed_kernel=routed_kernel,
            routed_reduce_dtype=routed_reduce_dtype,
            router_mode=router_mode,
            router_dtype=router_dtype,
            moe_router_weights=_group_stack(moe_router_weight, moe_k, n_moe, totals[2]),
            moe_in_proj_scales=_group_stack(moe_in_proj_scale, moe_k, n_moe, totals[2]),
            moe_expert_base=moe_k,
            affinity_scaling=affinity_scaling,
            affinity_hi_lo=affinity_hi_lo,
            mla_zeroed_pages=mla_zeroed_pages,
            mla_loop_first_static=mla_loop_first_static,
            mla_scores_ahead=mla_scores_ahead,
            mla_folded_reference=mla_folded_reference,
            mla_cp_rows_options=mla_cp_rows_options,
            mla_pair_pages=mla_pair_pages,
            x_parity=x_parity,
            kernel_first_layer=first_layer,
            lora_norm_eps=lora_norm_eps,
            moe_early_loads=moe_early_loads,
            moe_latent_load=moe_latent_load,
            moe_half_weight_buffers=moe_half_weight_buffers,
            moe_gate_up_half_major=moe_gate_up_half_major,
            **({"moe_shared_scale_buffer": True} if moe_shared_scale_buffer else {}),
            **({"moe_ring_reclaim": True} if moe_ring_reclaim else {}),
            **(
                {"moe_half_weight_buffer_count": moe_half_weight_buffer_count}
                if moe_half_weight_buffer_count is not None
                else {}
            ),
            moe_routed_store_form=moe_routed_store_form,
            moe_dynamic_loop=moe_dynamic_loop,
            mla_cp=mla_cp,
            dp_size=dp_size,
            mla_options=_dr_keys_options(
                _rope_ahead_options(
                    _mx_keys_options(
                        {**(mla_options or {}), "pack_sum_after_copy": True}
                        if mla_pack_sum_after_copy
                        else mla_options,
                        mla_pack_mx_keys,
                    ),
                    mla_pack_rope_ahead,
                ),
                mla_pack_dr_keys,
            ),
            reduce_rows=reduce_rows,
            **({"ffn_tail_order": True} if ffn_tail_order else {}),
            **({"ffn_weights_after_attn": True} if ffn_weights_after_attn else {}),
            **({"moe_glue_dmas": tuple(moe_glue_dmas)} if moe_glue_dmas else {}),
            **({"moe_direct_latent": True} if moe_direct_latent else {}),
            **({"moe_router_rows_drop_selection": True} if moe_router_rows_drop_selection else {}),
            **({"moe_latent_gather_mx": True} if moe_latent_gather_mx else {}),
            **({"moe_static_alt_tiles": moe_static_alt_tiles} if moe_static_alt_tiles else {}),
            **({"moe_routing_tiles": True} if moe_routing_tiles else {}),
            **({"moe_x_world_gather": True} if moe_x_world_gather else {}),
            **({"moe_x_path_act": True} if moe_x_path_act else {}),
            mix_cache=mix_cache,
            early_ffn_gate=early_ffn_gate,
            attn_res_cached_bank_sums=bank_sums,
            **({"attn_res_early_sums": True} if attn_res_early_sums else {}),
            routing_form=routing_form,
            moe_active_expert_slots=moe_active_expert_slots,
            moe_sel_load_options=moe_sel_load_options,
            moe_active_expert_counts=None
            if expert_counts is None or not n_moe
            else expert_counts[moe_k : moe_k + n_moe],
            **(
                {}
                if moe_in_proj_form == "gathered"
                else {
                    "moe_in_proj_form": moe_in_proj_form,
                    "moe_peer_in_projs": _group_stack(moe_peer_in_proj, moe_k, n_moe, totals[2]),
                }
            ),
            **({} if moe_out_proj_form == "reduced" else {"moe_out_proj_form": moe_out_proj_form}),
            **({"mla_pack_group_plans": True} if mla_pack_group_plans else {}),
            **({"mla_pack_pairs": True} if mla_pack_pairs else {}),
            **({"mla_stage_cc_split": mla_stage_cc_split} if split_sites else {}),
            **({"mla_stage_post": True} if mla_stage_post else {}),
            **({"mla_pack_pair_plans": True} if mla_pack_pair_plans else {}),
            **({"mla_pack_setup_ahead": True} if mla_pack_setup_ahead else {}),
            **(
                {"mla_pack_state_store_queues": mla_pack_state_store_queues} if mla_pack_state_store_queues != 1 else {}
            ),
            **({"mla_current_after_rs": True} if mla_current_after_rs else {}),
            **({"mla_stage_pre": True} if mla_stage_pre else {}),
            **({"moe_latent_shard_form": moe_latent_shard_form} if moe_latent_shard_form != "one_hot" else {}),
            **(
                {"attn_res_kda_mlp_early_entries": attn_res_kda_mlp_early_entries}
                if attn_res_kda_mlp_early_entries
                else {}
            ),
            **({"attn_res_kda_mlp_early_in_front": True} if attn_res_kda_mlp_early_in_front else {}),
            **({"attn_res_bank_dma": attn_res_bank_dma} if attn_res_bank_dma else {}),
            **({"norm_one_mul": True} if norm_one_mul else {}),
            **({"attn_res_scalar_squares": attn_res_scalar_squares} if attn_res_scalar_squares else {}),
            **({"norm_scalar_squares": norm_scalar_squares} if norm_scalar_squares else {}),
            **({"attn_res_scalar_squares_order": True} if attn_res_scalar_squares_order else {}),
            **({"attn_res_scalar_products": attn_res_scalar_products} if attn_res_scalar_products else {}),
            **({"attn_res_tensor_sums": True} if attn_res_tensor_sums else {}),
            **({"attn_res_one_round_norm": True} if attn_res_one_round_norm else {}),
            **({"attn_res_tensor_sums_idle": True} if attn_res_tensor_sums_idle else {}),
            **({"kda_conv_keep": True} if kda_conv_keep else {}),
            **({"moe_pad_rows": moe_pad_rows} if moe_pad_rows is not None else {}),
            **({"row_sums_after_ffn_output": True} if attn_res_sums_after_moe_output else {}),
            **({"moe_preamble_buffers": moe_preamble_buffers} if moe_preamble_buffers else {}),
            **({"moe_preamble_reuse": True} if moe_preamble_reuse else {}),
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
            # The ScalarE squares of the bank entries follow the last collective of the last block.
            if attn_res_scalar_squares and count and not mix:
                mix = {
                    "act_square_rows": attn_res_scalar_squares,
                    "act_bank_square_rows": attn_res_scalar_squares,
                    "act_square_after": tuple(gate or ()),
                }
            if attn_res_scalar_products and count > 1 and "bank_rstd" not in mix:
                mix = {**mix, "act_product_rows": attn_res_scalar_products}
            with row_sums_after_ffn_output(attn_res_sums_after_moe_output):
                s = attn_res_hT(bank_view if count else None, s, final_w, eps, ones, **mix)
        if final_norm_weight is not None:
            # The final hidden takes the FFN input tile, so the collective tiles stay reserved through lm_head.
            final_x = layer_sources(sources, first_layer + num_layers)["ffn_x"]
            act_norm = {"act_square_rows": norm_scalar_squares} if norm_scalar_squares else {}
            s = rmsnorm_hT(s, load_vector_hT(final_norm_weight), eps, ones, out=final_x, **act_norm)

    out = nl.ndarray((B, H), dtype=stream.dtype, buffer=nl.shared_hbm, name="stream_out")
    store_hT_rows(out, s)
    tail = ()
    if lm_head_weight is not None and (greedy or return_logits):
        logits = None
        if return_logits:
            logits = nl.ndarray((B, lm_head_weight.shape[1]), dtype=stream.dtype, buffer=nl.shared_hbm, name="logits")
        with weight_gates(dma_gates):
            # The lm_head weights wait for the output collectives of the last feed-forward body.
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
            )
        if greedy:
            tokens = nl.ndarray((B, 1), dtype=nl.int32, buffer=nl.shared_hbm, name="tokens")
            nisa.dma_copy(dst=tokens, src=token, name="tail_tokens_store")
            tail += (tokens,)
        if return_logits:
            tail += (logits,)
    if region is not None:
        check_identities(region)
    if expert_counts is not None:
        tail += (expert_counts,)
    if debug:
        return (out, *tail, layer_streams, layer_debug, *pools)
    return (out, *tail, *pools)
