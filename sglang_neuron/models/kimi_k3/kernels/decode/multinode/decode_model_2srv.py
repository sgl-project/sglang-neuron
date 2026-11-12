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

"""Kimi-K3 multi-server Trn3 decode model giga-kernel: the decoder layers of one step over several Trn3 servers.

The entry is the multi-server form of ``decode_model.kimi_k3_decode_model``. It runs the same block body
(``decode_block.decode_block_body``) with the same attention, KDA, MLA, AttnRes and tail code and options, with the
layout below:

- the world has ``128 * nnodes`` LNC1 ranks (256 for 2 servers);
- the attention runs in ``dp = world / 32`` replicas of TP32 / CP32, each inside one server (rank ``g``: TP rank
  ``g % 32``, DP rank ``g // 32``), as the DP4 replicas of one server;
- the MoE layers run ``tkg_moe_2srv`` (DP x TP32 projections, EP x TP experts in the static MXFP4 loop), and the
  dense layer 0 runs ``tkg_dense_mlp_2srv`` (the world MLP of ``tkg_ffn_world_cc``). Each runs its collectives once for all
  rows, and its matmuls or expert calls on chunks of at most 128 of the gathered tokens.

Every feed-forward collective that spans the servers runs in HBM; the attention and tail collectives stay inside the
TP32 group of a server.
"""

import math

import nki
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from ..utils.kernel_assert import kernel_assert
from ..utils.trace_names import restart_calls
from ..common import (
    P_MAX,
    WeightRow,
    attn_res_hT,
    load_vector_hT,
    new_ones,
    rmsnorm_hT,
    row_sums_after_ffn_output,
    store_hT_rows,
)
from ..decode_block import (
    _KDA_FRONT_ROWS_MAX_TOKENS,
    MLA_STAGE_SPLITS,
    NUM_LAYERS,
    _layer_view,
    banks_snapshot,
    block_layer_counts,
    check_pair_shared_forms,
    collective_sources,
    decode_block_body,
    is_dense_layer,
    is_mla_layer,
    layer_sources,
    mla_stage_unit_tokens,
    mla_staged,
    new_bank_rstd,
)
from ..decode_model import MAX_POOLS, TAIL_TP_RANKS, _group_stack, _kda_body, _rows, model_layer_groups
from ..tkg_lm_head_greedy import lm_head_greedy_hT
from ..tkg_weight_dma_gate import set_weight_gate, weight_gates
from ..tkg_sbuf_alloc import EXPLICIT_BYTES, ExplicitSbuf, SbufTiles, check_identities, install_identities
from ..tkg_kda import kda_front_ahead_body, kda_options_body, kda_w_in_head_bytes
from ..tkg_mla_block import (
    ROWS_LOCAL_HEADS,
    ROWS_OPTIONS,
    ROWS_SEQUENCES,
    ROWS_TWIN_OPTIONS,
    mla_block_collective_hbm,
    mla_block_collective_sources,
    mla_rows_decode_attention,
)
from ..tkg_mla_paged import LORA_NORM_EPS, mla_decode_attention
from . import tkg_moe_2srv as moe_2srv_module
from .tkg_dense_mlp_2srv import DENSE_SOURCES_KEY, dense_collective_hbm, dense_ffn_2srv
from .tkg_ffn_world_cc import world_layout
from .moe_routed_plan import (
    ALT_STACK_ARGUMENTS,
    call_csr_gather,
    call_plan,
    check_alternates,
    moe_layers_of,
    plan_moe,
    plan_selector_splits,
)
from .tkg_moe_2srv import (
    EXPERT_CALL_AUTO,
    MOE_FORMS,
    MOE_GATHERS,
    MOE_SOURCES_KEY,
    OUT_PROJ_FORMS,
    ROUTED_REDUCES,
    WIDE_HALF_WEIGHT_BUFFERS,
    check_column_private,
    moe_2srv_body,
    moe_2srv_collective_hbm,
    moe_2srv_compact_constants,
    moe_2srv_glue_selectors,
    moe_2srv_world_split,
    routed_layers_options,
)
from .tkg_moe_2srv import MAX_REPLICA_ROWS as MOE_MAX_REPLICA_ROWS

SERVED_ROWS_OPTIONS = {
    "folded_reference": True,
    "rows_v_stationary": True,
    "rows_prefetch": True,
    "rows_setup_ahead": True,
    "rows_q_fp8": True,
    "rows_mx_keys": "single",
}
"""The rows-form keys (``tkg_mla_block.ROWS_OPTIONS``) of the k3 form with FP8 pages: the folded reference, V as the PV
stationary, the head pages of each loop iteration loaded one iteration early, the sequence setup of every sequence
before the first loop, the e4m3 query gather (``rows_q_fp8``: the one query gather of the call moves the e4m3 bytes
that the latent scores read), and the MX latent QK (``rows_mx_keys`` "single": each latent score block is one MX
matmul of the FP8 keys, in place of the double-row scores)."""
ATTENTION_MAX_SEQUENCES = 32
"""Most sequences of a replica that the attention bodies take in this entry. It does not follow the one-server bound
of the decode block, so a one-server change cannot raise it: a larger value is a multi-server change of its own."""
MAX_REPLICA_ROWS = min(ATTENTION_MAX_SEQUENCES, MOE_MAX_REPLICA_ROWS)
"""Most sequences of a replica of the entry: the attention bound and the row bound of the MoE body."""
ROUTED_RS_AG_FORMS = ((256, 32), (256, 16))
"""The (world_size, B) forms that take ``moe_routed_reduce`` ``rs_ag``: the 2-server form with 32 rows of each replica,
where the reduce-scatter and the all-gather take fewer collective table entries than the routed all-reduce, and with 16
rows. At 16 rows the all-reduce runs a shorter algorithm, so the two collectives add table entries: a 16-row program
needs the collective-table check of its own program. Other forms fail."""
ROUTED_RS_AG_FORMS_4SRV = ((512, 32), (512, 16))
"""The 4-server (world 512) forms of ``moe_routed_reduce`` ``rs_ag``, beside ``ROUTED_RS_AG_FORMS``: the same rows of a
replica, the world reduce-scatter over 512 ranks. Opt-in (the routed all-reduce stays the default); the
collective-table entries of a 512 program have no measurement yet."""
TWO_SERVER_TAKES_B16_STACK_KEYS = True
"""The entry takes the AttnRes, KDA and MLA order options of the one-server decode (``attn_res_kda_mlp_early_entries``
.. ``kda_state_loads_early``), so a caller can check for them before it passes any."""
TWO_SERVER_TAKES_KDA_STATE_DTYPE = True
"""The entry takes ``kda_state_dtype`` (the fp32 KDA state pages of the one-server decode), so a caller can check
for it before it passes any value but ``bf16``."""
TWO_SERVER_TAKES_KDA_AHEAD_KEYS = True
"""The entry takes ``kda_w_in_head`` and ``kda_front_ahead``, so a caller can check for them before it passes either."""
TWO_SERVER_TAKES_KDA_BF16_UPDATE_ROWS = True
"""The entry takes ``kda_bf16_update_rows`` (the bf16 update rows of the one-server decode), so a caller can check for it
before it passes True."""
TWO_SERVER_TAKES_MOE_IN_PROJ_AT_KDA_REDUCE = True
"""The entry takes ``moe_in_proj_at_kda_reduce``, so a caller can check for it before it passes True."""
TWO_SERVER_TAKES_MOE_2SRV_SELECTORS_ONCE = True
"""The entry takes ``moe_2srv_selectors_once``, so a caller can check for it before it passes True."""
TWO_SERVER_TAKES_MOE_STATIC_ALT_TILES = True
"""The entry takes ``moe_static_alt_tiles`` (``moe_routed.STATIC_ALT_TILE_FORMS``), so a caller can check for it
before it passes a value."""


def _kda_w_in_heads(plan, kda_w_in: nl.NkiTensor, first_layer: int, num_layers: int, n_kda: int) -> dict:
    """Return the ``kda_w_in_heads`` of ``decode_block_body`` for the ``kda_w_in_head`` entries ``(layer, address,
    chunks)`` of the entry: ``{layer: (address, chunks, the layer's kda_w_in)}`` ({} for 0 or ())."""
    if not plan:
        return {}
    kernel_assert(
        isinstance(plan, (tuple, list)) and all(isinstance(e, (tuple, list)) and len(e) == 3 for e in plan),
        f"kda_w_in_head must be 0, (), or a tuple of (layer, address, chunks), got {plan!r}",
    )
    plan = tuple(tuple(int(v) for v in entry) for entry in plan)
    kda_layers = [i for i in range(first_layer, first_layer + num_layers) if not is_mla_layer(i)]
    heads = {}
    for layer, address, chunks in plan:
        kernel_assert(
            first_layer < layer < first_layer + num_layers
            and not is_mla_layer(layer)
            and not is_dense_layer(layer - 1)
            and layer not in heads,
            f"kda_w_in_head layer {layer}: a KDA layer of the range after a MoE layer of the range, once",
        )
        w_in = _layer_view(kda_w_in, kda_layers.index(layer), n_kda)
        end = address + chunks * kda_w_in_head_bytes(w_in)
        kernel_assert(
            0 <= address and end <= EXPLICIT_BYTES,
            f"kda_w_in_head layer {layer}: [{address}, {end}) must be inside the {EXPLICIT_BYTES} explicit SBUF bytes",
        )
        heads[layer] = (address, chunks, w_in)
    return heads


def first_cc_destinations(sources: dict) -> list:
    """Return the HBM destinations of the first collective of each group in a step: the KDA all-reduce of layer 0 (TP32),
    and the DP x gather and the world output all-reduce of the dense layer 0.

    These collectives can skip their start handshake, and nothing before them in the step bounds their window: a peer
    that runs ahead writes their destinations while this rank still runs the step before. So no other tensor of the
    kernel may take their bytes (``hold_to_end``).
    """
    dense = sources.get(DENSE_SOURCES_KEY, {})
    names = [dense.get("x_dst"), dense.get("out_dst"), sources.get("kda_ar_dst_hbm")]
    return [t for t in names if t is not None]


def hold_to_end(tensors: list, last: list, out: nl.NkiTensor) -> None:
    """Keep each HBM tensor (or SBUF tile) of ``tensors`` live to the end of the kernel, so that no other tensor takes
    its bytes.

    The kernel reads the first word of ``out`` and of each output of ``last`` (the outputs written last), then, after
    those reads, the first word of each tensor of ``tensors`` (of partition 0 for an SBUF tile). It ORs
    ``sum(bits & 0)`` of the words into the bits of the first two elements of ``out`` and stores them again: the values
    stay the same, bit for bit.
    """
    sinks = [*last, *tensors]
    kernel_assert(
        all(t.dtype in (nl.int32, nl.uint32, nl.float32, nl.bfloat16) for t in sinks) and out.dtype == nl.bfloat16,
        "hold_to_end reads 4-byte words of int32, uint32, float32 and bfloat16 tensors into a bfloat16 output",
    )
    words = nl.ndarray((1, len(sinks)), dtype=nl.int32, buffer=nl.sbuf)
    head = nl.ndarray((1, 2), dtype=out.dtype, buffer=nl.sbuf)
    zero_i = nl.ndarray((1, len(sinks)), dtype=nl.int32, buffer=nl.sbuf)
    total = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    zero16 = nl.ndarray((1, 1), dtype=nl.uint16, buffer=nl.sbuf)

    def handle(instr):
        return instr if hasattr(instr, "depends_on") else nisa.NkiInstruction(instr)

    def read_word(i: int, tensor: nl.NkiTensor, name: str):
        k = 2 if tensor.dtype == nl.bfloat16 else 1
        if tensor.buffer == nl.sbuf:
            src = tensor.reshape((tensor.shape[0], math.prod(tensor.shape[1:])))[0:1, 0:k]
        else:
            src = tensor.reshape((1, math.prod(tensor.shape)))[0:1, 0:k]
        return handle(nisa.dma_copy(dst=words.view(tensor.dtype)[0:1, i * k : i * k + k], src=src, name=name))

    reads = [read_word(i, t, f"hold_to_end_last_{i}") for i, t in enumerate(last)]
    reads.append(handle(nisa.dma_copy(dst=head, src=out[0:1, 0:2], name="hold_to_end_head")))
    for i, tensor in enumerate(tensors):
        read_word(len(last) + i, tensor, f"hold_to_end_read_{i}").depends_on(*reads, kind=nisa.dependency.schedule)
    nisa.tensor_scalar(dst=zero_i, data=words, op0=nl.bitwise_and, operand0=0, engine=nisa.engine.vector)
    nisa.tensor_reduce(dst=total, data=zero_i, op=nl.add, axis=(1,))
    nisa.tensor_copy(dst=zero16, src=total)
    bits = head.view(nl.uint16)
    nisa.tensor_scalar(dst=bits, data=bits, op0=nl.bitwise_or, operand0=zero16, engine=nisa.engine.vector)
    nisa.dma_copy(dst=out[0:1, 0:2], src=head, name="hold_to_end_store")


def router_block_options(router_dtype: str, moe_router_weight, moe_k: int, n_moe: int, total: int) -> dict:
    """Return the FP16 router options of one block of ``decode_block_body``; the BF16 router gives none."""
    kernel_assert(
        (router_dtype == "bf16") == (moe_router_weight is None),
        f"router_dtype {router_dtype!r} needs moe_router_weight, and the BF16 router takes none",
    )
    if router_dtype == "bf16":
        return {}
    return {"router_dtype": router_dtype, "moe_router_weights": _group_stack(moe_router_weight, moe_k, n_moe, total)}


def in_proj_block_options(moe_in_proj_scale, moe_k: int, n_moe: int, total: int) -> dict:
    """Return the MXFP8 MoE in-projection options of one block of ``decode_block_body``; None gives none."""
    if moe_in_proj_scale is None:
        return {}
    return {"moe_in_proj_scales": _group_stack(moe_in_proj_scale, moe_k, n_moe, total)}


def out_proj_block_options(moe_out_proj_scale, moe_k: int, n_moe: int, total: int) -> dict:
    """Return the MXFP8 MoE out-projection options of one block of ``decode_block_body``; None gives none."""
    if moe_out_proj_scale is None:
        return {}
    return {"moe_out_proj_scales": _group_stack(moe_out_proj_scale, moe_k, n_moe, total)}


@nki.jit
def kimi_k3_decode_model_2srv(
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
    mla_packed_loop: bool = False,
    mla_pack_slots: int | None = None,
    mla_pack_tail_slots: int | tuple | None = None,
    mla_pack_tail_low_batch: bool = False,
    mla_pack_page_runs: int = 0,
    mla_pack_null_copy_vector: bool = False,
    mla_pack_null_runs: bool = False,
    mla_batched_cache_write: bool = False,
    mla_zeroed_pages: bool = False,
    mla_pair_pages: bool = False,
    lora_norm_eps: float = LORA_NORM_EPS,
    cc_source_pingpong: bool = True,
    mla_options: dict = None,
    mix_cache: bool = False,
    early_ffn_gate: bool = None,
    kda_conv_dtype: str = "bf16",
    kda_w_in_buffers: int = 0,
    kda_state_loads_first: bool = False,
    kda_batched_conv_stores: bool = False,
    kda_readout: str = None,
    kda_conv_load_gather: bool = False,
    attn_res_cached_bank_sums: bool = False,
    attn_res_early_sums: bool = False,
    ffn_weights_after_attn: bool = False,
    mla_pack_group_plans: bool = False,
    mla_pack_pairs: bool = False,
    mla_stage_cc_split: str = "off",
    mla_stage_post: bool = False,
    mla_pack_pair_plans: bool = False,
    mla_pack_setup_ahead: bool = False,
    mla_pack_state_store_queues: int = 1,
    mla_current_after_rs: bool = False,
    mla_stage_pre: bool = False,
    mla_pack_shared_hbm: bool = True,
    moe_half_weight_buffers: bool = False,
    moe_weight_buffers: int = None,
    moe_scale_buffers: int = None,
    moe_transpose_lookahead: int = None,
    moe_cross_expert_order: bool = False,
    moe_cross_expert_transpose_blocks: bool = False,
    moe_token_tiles_transpose_blocks: bool = False,
    moe_transpose_block: bool = False,
    moe_routed_store_form: str = "pe",
    world_size: int = 256,
    routed_tp: int = None,
    moe_form: str = "moe_2srv",
    dp_size: int = None,
    max_replica_rows: int = None,
    moe_gather: str = "dp_tp",
    moe_wide_half_weight_buffers: int = None,
    moe_expert_call_tokens: int | str = EXPERT_CALL_AUTO,
    moe_router_rows_drop_selection: bool = False,
    moe_routed_reduce: str = "ar",
    moe_routed_stats_ag: bool = False,
    moe_out_proj_form: str = "reduced",
    moe_rs_ag_allow_dr_keys: bool = False,
    moe_quantize_first: bool = False,
    moe_weights_after_gather: bool = False,
    moe_head_row_chunks: bool = False,
    moe_sums_after_routed: bool = False,
    moe_routed_twin: bool = False,
    moe_rows_latent_fp8: bool = False,
    moe_rows_load_split: int = 1,
    moe_routed_own_rows: bool = False,
    moe_rows_dst_pingpong: bool = True,
    moe_latent_gather_mx: bool = False,
    attn_res_kda_mlp_early_entries: int = 0,
    attn_res_kda_mlp_early_in_front: bool = False,
    mla_pack_sum_after_copy: bool = False,
    attn_res_scalar_squares: int = 0,
    norm_scalar_squares: int = 0,
    attn_res_scalar_squares_order: bool = False,
    attn_res_scalar_products: int = 0,
    kda_conv_stores_early: bool = False,
    kda_state_loads_early: bool = False,
    kda_state_dtype: str = "bf16",
    kda_paired_pages: bool = False,
    attn_res_tensor_sums: bool = False,
    attn_res_one_round_norm: bool = False,
    attn_res_tensor_sums_idle: bool = False,
    pair_shared_cc_dsts: tuple = (),
    kda_conv_keep: bool = False,
    kda_w_in_head: tuple | int = 0,
    kda_front_ahead: bool = False,
    moe_pad_rows: nl.NkiTensor = None,
    kda_bf16_update_rows: bool = False,
    moe_in_proj_at_kda_reduce: bool = False,
    moe_2srv_selectors_once: bool = False,
    attn_res_sums_after_moe_output: bool = False,
    moe_static_alt_tiles: bool | str = False,
    kda_null_page_rows: int = 0,
    kda_null_page: int = 0,
    mla_rows_twin: bool = False,
    mla_rows_twin_options: dict = None,
    live_rows: nl.NkiTensor = None,
    mla_state_bf16_o: bool = False,
    moe_glue: dict = None,
    kda_w_in_scale: nl.NkiTensor = None,
    mla_in_proj_scale: nl.NkiTensor = None,
    mla_q_b_proj_scale: nl.NkiTensor = None,
    mla_kv_b_proj_scale: nl.NkiTensor = None,
    mla_o_proj_scale: nl.NkiTensor = None,
    router_dtype: str = "bf16",
    moe_router_weight: nl.NkiTensor = None,
    moe_in_proj_scale: nl.NkiTensor = None,
    kda_w_out_scale: nl.NkiTensor = None,
    moe_out_proj_scale: nl.NkiTensor = None,
    moe_mx_tile: int = None,
    routed_layers: tuple = None,
    moe_routed_dynamic: bool = False,
    routed_alt0_expert_gate_up_weights: nl.NkiTensor = None,
    routed_alt0_expert_gate_up_weights_scale: nl.NkiTensor = None,
    routed_alt0_expert_down_weights: nl.NkiTensor = None,
    routed_alt0_expert_down_weights_scale: nl.NkiTensor = None,
    routed_alt1_expert_gate_up_weights: nl.NkiTensor = None,
    routed_alt1_expert_gate_up_weights_scale: nl.NkiTensor = None,
    routed_alt1_expert_down_weights: nl.NkiTensor = None,
    routed_alt1_expert_down_weights_scale: nl.NkiTensor = None,
    mla_rs_hold: bool = True,
):
    """Run the Kimi-K3 decoder layers of a range for one DP replica on one rank of a multi-server Trn3 world.

    The positional arguments are those of ``decode_model.kimi_k3_decode_model``, with the MoE weights of
    ``tkg_moe_2srv``. The keyword arguments are its attention, KDA, MLA, AttnRes, tail and expert-loop options with
    the same names, plus the world below. Its TP128 MoE options (``routed_kernel``, ``moe_sbuf_reduces``, the
    routing and projection forms) and ``mla_cp`` do not apply: the call fails with an unknown argument. The kernel
    updates the page pools in place and returns them, so that they alias the inputs.

    Dimensions:
        H: hidden size, 7168. L: layers in the range, 1 to 93 (``first_layer + L <= 93``).
        W: ``world_size``, 128 ranks for each server. DP: ``W / 32`` replicas.
        B: sequences of the replica, 1 to ``max_replica_rows``. The feed-forward token gathers run in chunks of
            ``128 / DP`` rows.
        I: the dense intermediate shard, ``33792 / W`` (132 at 256 ranks).

    Args:
        stream, norm_weights, attn_res_weights, kda_*, mla_*: as in ``kimi_k3_decode_model`` (TP32 / CP32 layouts).
        moe_in_proj, moe_out_proj, moe_vectors: the DP projections of TP rank ``g % 32`` (``tkg_moe_dp`` layout),
            stacked ``[n_moe, 128, 56, 524]`` BF16, ``[n_moe, 304, 7168]`` BF16 and ``[n_moe, 1, 140]`` FP32
            (``[n_moe, 1, 1036]`` with ``moe_router_rows_drop_selection``).
        expert_gate_up_weights .. expert_down_weights_scale: the experts of group ``g % G``, intermediate part
            ``g // G`` (``tkg_moe_2srv.moe_2srv_expert_shapes``), stacked ``[n_moe, ...]``.
        dense_gate_up, dense_down: the dense MLP weights of layer 0 for ``W`` ranks (``tkg_ffn_world_cc.dense_ffn_world``),
            stacked ``[1, 128, 56, 2 I]`` and ``[1, I, 7168]`` BF16.
        kda_page_ids, mla_block_table, mla_slot, position: as in ``kimi_k3_decode_model``.
        rank_info: ``[1, 4]`` int32 (global rank, DP rank, TP rank, 0) with DP rank ``g // 32``.
        tp_group, dp_group, world_group: the TP32 groups (one for each replica), the DP groups (rank ``t`` of every
            replica) and the world (``tkg_ffn_world_cc.replica_groups_world``).
        first_layer, eps, debug, dma_gates, pool_of_layer, mla_loop_path, kv_pool_i: as in ``kimi_k3_decode_model``.
        final_attn_res_weight, final_norm_weight, lm_head_weight, greedy, return_logits: the tail, as in
            ``kimi_k3_decode_model``; ``lm_head_weight`` is the V/32 vocab shard of the TP rank.
        mla_fp8_pages .. mla_stage_pre: as in ``kimi_k3_decode_model``.
        mla_pack_shared_hbm: with ``mla_packed_loop`` on the loop path, the packed MLA loops of all layers share one
            pair of HBM rows and items (``mla_pack_shared_hbm`` of ``kimi_k3_decode_model``). The staged groups (above
            8 sequences) always share them. Bitwise the same output; True (default) gives the form of 8 sequences and
            fewer one pair in place of one for each MLA layer.
        mla_pack_null_copy_vector: with ``mla_packed_loop`` and ``mla_zeroed_pages``, the null-entry page fix of the
            packed loop on VectorE (``decode_block.decode_block_body``). False (default) keeps the program.
        mla_pack_null_runs: with ``mla_pack_page_runs``, a padded sequence counts as page runs, so its rows are
            finite (``decode_block.decode_block_body``). False (default) keeps the program.
        moe_half_weight_buffers: above 8 sequences, half-size weight buffers in the static expert loop
            (``decode_block.decode_block_body``). The values do not change; False (default) keeps the program.
        moe_weight_buffers, moe_scale_buffers, moe_transpose_lookahead: the whole weight buffers (1 to 5), the gate/up
            scale buffers (1 or 2) and the activation chunks transposed ahead (1 or 2) of the static expert loop
            (``decode_block.decode_block_body``). The values do not change; None (default) keeps the program.
        moe_cross_expert_order: the gate matmuls of the next expert run between the up and the down matmuls of each
            expert of the static expert loop (``decode_block.decode_block_body``; needs moe_weight_buffers 3 to 5). The
            values do not change; False (default) keeps the program.
        moe_cross_expert_transpose_blocks: the activation transposes of ``moe_cross_expert_order`` run in tile blocks
            between the gate chunks of the next expert, with no MX matmul inside a block
            (``decode_block.decode_block_body``; needs moe_cross_expert_order). The values do not change; False
            (default) keeps the program.
        moe_token_tiles_transpose_blocks: the token-tiled expert calls (more than 128 tokens) run one fixed TensorE
            order with every PE transpose in a block between MX matmul groups (``decode_block.decode_block_body``).
            The values do not change; False (default) keeps the program.
        moe_transpose_block: with ``moe_cross_expert_order``, the activation transposes of each expert run in one
            block with no MX matmul inside (``decode_block.decode_block_body``). The values do not change; False
            (default) keeps the program.
        moe_routed_store_form: how the expert loop fills the routed all-reduce source (``tkg_moe.ROUTED_STORE_FORMS``):
            ``pe`` (default) through TensorE transposes, ``relayout`` through an HBM output and a relayout DMA. The
            values do not change.
        world_size: W, 256 (2 servers, default), or 128 for a one-server test world of the same code.
        routed_tp: the intermediate parts of each expert (``tkg_moe_2srv.moe_2srv_split``); None gives the server
            count. 2 on 128 ranks runs the 2-server expert split (EP64 x TP2) on one server.
        moe_mx_tile: None (default) keeps the expert split of 512-column MX tiles (``tkg_moe_2srv.moe_2srv_split``);
            else the MX tile width of the expert split
            (``tkg_moe_2srv.moe_2srv_world_split``), for the sub-512 intermediate slices of a 4-server world.
        routed_layers: None (default) keeps ``routed_tp`` and ``moe_mx_tile``; else the per-layer plan
            ``(default_tag, ((layer, tag), ...))`` of ``moe_routed_layout`` (the default tag and at most
            ``moe_routed_layout.MAX_ALT_TAGS`` other tags), whose default tag sets both
            (``tkg_moe_2srv.routed_layers_options``).
        moe_routed_dynamic: False (default) keeps the program; True runs the routed experts of a split of sub-512 MX
            tiles in the dynamic active-expert form (``tkg_moe_2srv.moe_2srv_body`` ``routed_dynamic``).
        routed_alt0_expert_gate_up_weights .. routed_alt1_expert_down_weights_scale: None (default), or the expert
            stacks of the other tags of a ``routed_layers`` plan (``moe_routed_plan``): slot s holds the
            call's layers of the s-th other tag (in order of its first layer), one entry a layer, in the layout of its
            split. With such a plan the base expert stacks hold the call's layers of the default tag only. Each MoE
            layer runs the split of its tag, fixed at load; the collectives are those of every layer.
        moe_form: the MoE body, ``moe_2srv`` (``tkg_moe_2srv.MOE_FORMS``).
        dp_size: None, or ``W / 32`` (checked).
        max_replica_rows: the most sequences of a replica that the caller sends, at most ``MAX_REPLICA_ROWS`` (the
            attention bound ``ATTENTION_MAX_SEQUENCES`` and the bound of the MoE body). None (default) takes that bound.
        moe_gather: the router-row gathers of the MoE layers, ``tkg_moe_2srv.MOE_GATHERS``: ``dp_tp`` (default) or
            ``world``. The values do not change.
        moe_wide_half_weight_buffers: None (default, ``tkg_moe_2srv.WIDE_HALF_WEIGHT_BUFFERS``), or the
            half-projection weight buffers (2 to 6) of the whole-expert form (routed_tp 1) above
            ``tkg_moe_2srv.WIDE_ROWS`` sequences. The values do not change.
        moe_expert_call_tokens: the tokens of one expert call of the MoE layers, passed to the body: "auto" (default,
            ``tkg_moe_2srv.auto_expert_call_tokens``), or one of ``tkg_moe_2srv.EXPERT_CALL_FORMS`` (128, or the
            token-tiled calls of 256 or 512 tokens). The values do not change.
        moe_router_rows_drop_selection: the router rows of the MoE gathers leave out the selection values (84 FP32
            words in place of 112, 25 % fewer bytes in the DP and TP32 row gathers), and each rank adds the router
            bias to the gathered scores (``tkg_moe_2srv.latent_moe_decode_2srv`` ``drop_selection``). It needs
            ``moe_vectors`` ``[n_moe, 1, 1036]`` FP32: the 140 columns of the DP layout, then the router correction bias
            of all 896 experts in expert order (``tkg_moe_2srv.ROUTER_BIAS_COLS``, the same on every rank); the
            call fails without them. The values do not change; False (default) keeps the program.
        moe_rows_dst_pingpong: with ``cc_source_pingpong`` and ``moe_gather`` ``dp_tp``, the odd MoE layers' DP gather of
            the router rows writes a second persistent destination (``tkg_moe_2srv.moe_2srv_collective_hbm``
            ``rows_dst_pingpong``), so no gather writes the destination of the gather before it. True (default): the
            gathers skip their start handshakes. The values do not change; False keeps one destination.
        moe_routed_reduce: the reduction of the routed partials of the MoE layers (``tkg_moe_2srv.ROUTED_REDUCES``):
            ``ar`` (default) keeps the program; ``rs_ag`` runs a world reduce-scatter and a TP32 all-gather in place of
            the world all-reduce, with alternating destinations by layer parity. It rounds differently (other reduction
            owners), needs ``cc_source_pingpong`` and takes only the ``ROUTED_RS_AG_FORMS`` forms. ``rs_fold`` takes the
            same reduce-scatter and folds the latent norm into one FP32 output all-reduce
            (``tkg_moe_2srv.fold_out_reduce``), with the same rules; it also changes the rounding of the norm.
        moe_routed_stats_ag: with ``moe_routed_reduce`` ``rs_ag``, the TP32 all-gather carries the FP32 sums of squares
            of the norm instead of the full rows. False (default).
        moe_out_proj_form: the out projection of the MoE layers (``tkg_moe_2srv.OUT_PROJ_FORMS``): ``reduced``
            (default) keeps the program; ``column`` computes 224 hidden columns of the replica's rows on each TP rank
            from all 9,728 inputs and runs a TP32 all-gather of the columns (and one of the shared-expert activation)
            in place of the TP32 output all-reduce. It needs ``moe_out_proj`` in the column layout and ``moe_vectors``
            with the 3,584 latent norm weights after its columns (``[n_moe, 1, 3724]``, or ``[n_moe, 1, 4620]`` with
            ``moe_router_rows_drop_selection``). With
            ``moe_routed_reduce`` ``rs_ag`` the norm reads the replica's rows from the routed TP32 all-gather (not with
            ``moe_routed_stats_ag``). The sum order differs from ``reduced``: an accuracy-class option.
        moe_rs_ag_allow_dr_keys: lift the refusal of ``moe_routed_reduce`` ``rs_ag`` with the FP8 double-row MLA keys
            (``mla_options`` ``pack_dr_keys``, form B, with the e4m3 query): on Trn3 the pair missed a 1M needle at depth
            0.9, while each one alone passed. Pass True only after a passing 1M needle row of the pair. False (default)
            refuses the pair at trace time; it changes no program.
        moe_quantize_first: each expert call of the MoE layers quantizes its latent to the MXFP8 expert input before
            the routing affinities of its tokens (and before the bias add of ``moe_router_rows_drop_selection``), so the
            Vector engine's in-order stream does not hold the quantize, and with it the first MXFP8 matmul of the
            expert loop, behind the affinity instructions; the transposes read the gathered latent tile in place
            (``tkg_moe_2srv.latent_moe_decode_2srv`` ``quantize_first``). Needs expert calls of at most 128 tokens.
            The same instructions run in another order, so the values do not change; False (default) keeps the
            program.
        moe_weights_after_gather: the first loads of the expert weights and scales of each MoE layer (the stream start
            of the static expert loop) get a schedule dependency on the last router-row gather
            (``tkg_moe_2srv.latent_moe_decode_2srv`` ``weights_after_gather``). neuronx-cc then does not move them to right after the DP gather with a wait for
            its end, so they stream before the forward phase of the TP32 gather instead of during it. Schedule edges
            only, so the values do not change; False (default) keeps the program.
        moe_head_row_chunks: each expert call of the MoE layers loads its gathered router rows in 4 DMAs of 8 TP blocks
            and swizzles and quantizes each D512 tile of its latent as soon as the chunks that hold its columns are
            copied, so the first MXFP8 matmuls start before the last chunk lands
            (``tkg_moe_2srv.latent_moe_decode_2srv`` ``head_row_chunks``). Needs ``moe_quantize_first``, the
            ``dp_tp`` gather and expert calls of at most 128 tokens. The same instructions on the same bytes, so the
            values do not change; False (default) keeps the program.
        moe_sums_after_routed: the row sums of each layer (its AttnRes bank sums and norms) follow the routed
            reduction of the previous MoE layer in the schedule (``decode_block.decode_block_body``
            ``moe_sums_after_routed``), so the Vector engine copies the routed partials out of PSUM before it runs the
            next layer's bank sums. The values do not change; False (default) keeps the program.
        moe_routed_twin: test only, the identity twin of a verify that reduces the routed partials with ``rs_ag`` or
            ``rs_fold``: those reductions then take any B, not only the ``ROUTED_RS_AG_FORMS`` forms. Each element keeps
            the reduction owner of the verify (reduce-scatter block 32 r + t holds the rows of replica r and the latent
            columns of TP rank t, at every B), so the decode steps can give the bits of the verify rows. The program
            of such a B is not checked against the collective table. False (default) keeps the rule.
        moe_rows_latent_fp8: the MoE router rows carry each rank's 112 latent values as float8_e4m3fn with one FP32
            scale (amax / 448), 26 words fewer a row (``tkg_moe_2srv`` ``rows_latent`` ``fp8``). It rounds the expert
            input: an accuracy-class option. False (default) keeps the program.
        moe_rows_load_split: the parts of the router-rows load of each expert call, as in
            ``kimi_k3_verify_model_2srv`` (``latent_moe_decode_2srv`` ``rows_load_split``). 1 (default) keeps the
            program. The values do not change.
        moe_routed_own_rows: load only the replica's rows of the routed all-reduce, as in
            ``kimi_k3_verify_model_2srv`` (``latent_moe_decode_2srv`` ``routed_own_rows``). False (default) keeps the
            program. The values do not change.
        moe_latent_gather_mx: send the MXFP8 latent codes in the MoE router rows and run no latent quantize in the
            expert calls, as in ``kimi_k3_verify_model_2srv`` (``tkg_moe_2srv`` ``latent_gather_mx``). ``moe_in_proj``
            has the 540-column block layout. Needs
            ``moe_gather`` ``dp_tp``, whole replicas in each 128-token tile of an expert call and no
            ``moe_rows_latent_fp8``. Against the BF16
            rows the expert input is byte equal. False (default) keeps the program.
        attn_res_kda_mlp_early_entries, attn_res_kda_mlp_early_in_front, mla_pack_sum_after_copy,
        attn_res_scalar_squares, norm_scalar_squares, attn_res_scalar_squares_order, attn_res_scalar_products,
        kda_conv_stores_early, kda_state_loads_early: the AttnRes, KDA and MLA order options of
            ``kimi_k3_decode_model``, with the same meaning, checks and defaults. The attention bodies of a replica are
            those of the one-server decode, so each option acts on the B sequences of the replica as there. The values
            do not change; the defaults keep the program.
        kda_state_dtype: the form of the KDA recurrent state in the pages, as in ``kimi_k3_decode_model``: ``bf16``
            (default) keeps the program; ``fp32`` and ``fp32_hilo`` keep an fp32 state and need pools of at least
            101760 bf16 elements per page (512-row pages ``[P, 147456]``).
        kda_paired_pages: with an fp32 ``kda_state_dtype``, the KDA page ids name the even page of an aligned pair of
            the 256-row pools (``tkg_kda.kda_page_span``): each KDA page is the pair, with the bytes of the 512-row
            form at id / 2, and the MLA pages keep their rows. The pools need 2 x width >= the fp32 page end (101760 at
            3 local heads). The valid page ids are the EVEN ids in [1, P - 1) (``tkg_kda.kda_page_bound``), so
            id + 1 < P: an odd id or an id >= P - 1 is off. Bitwise vs the 512-row form of the KDA pages. False
            (default) keeps the program.
        attn_res_tensor_sums: the GPU form of the AttnRes mixes of ``kimi_k3_decode_model`` (TensorE weighted sums,
            the bank rstd kept from the pushes, one rounding of each mix and its norm), with the same checks. Not
            bitwise the same (GPU parity). False (default) keeps the program.
        attn_res_one_round_norm: the GPU-parity norm of every AttnRes mix without TensorE sums (form F0), as in
            ``kimi_k3_decode_model``, with the same checks. False (default) keeps the program.
        attn_res_tensor_sums_idle: the TensorE weighted sums only at the MLA-layer mixes and the KDA attention mixes,
            with the GPU-parity norm of every mix (form F1), as in ``kimi_k3_decode_model``, with the same checks. False
            (default) keeps the program.
        pair_shared_cc_dsts: the collective destinations in LNC1 shared HBM, as in ``kimi_k3_decode_model``: ``moe_out``
            the TP32 output all-reduce of the MoE layers and the world output all-reduce of the dense layer
            (``tkg_moe_2srv.moe_2srv_collective_hbm`` ``out_dst``, ``tkg_dense_mlp_2srv.dense_collective_hbm``),
            ``moe_routed`` the world routed all-reduce (``routed_dst``; ``moe_routed_reduce`` ``ar`` only) and
            ``kda_ar`` the KDA output all-reduce. ``moe_gather`` is refused: the gathers of the multi-server MoE body stay
            private. ``moe_out_proj_form`` ``column`` refuses ``moe_out`` and ``moe_routed``
            (``tkg_moe_2srv.COLUMN_PRIVATE_SITES``). The sources stay private and the values do not change. It needs the neuronx-cc option
            --lnc1-with-shared-mem and a runtime with pair-shared HBM. () (default) keeps the program.
        kda_conv_keep: keep the KDA conv-state store sources until their stores have read them, and chain the
            persistent conv-state tile rewrites to the stores of the KDA layers before, across the blocks
            (``decode_block.decode_block_body`` ``kda_conv_keep``). Placement and sync only: the values do not change.
            False (default) keeps the program.
        kda_w_in_head: 0 or () (default) keeps the program; or a tuple of ``(layer, address, chunks)``: the MoE layer
            ``layer - 1`` loads the first ``chunks`` chunks (``4 * 128`` hidden rows each) of the ``kda_w_in`` of KDA
            layer ``layer`` into SBUF at the explicit byte ``address`` (per partition, a multiple of 32; chunk ``c`` at
            ``address + c * 4 * N * 2``) on the ACT queue, in its output glue right after the load of the result of its
            routed reduction's gate collective (the routed all-reduce of ``moe_routed_reduce`` ``ar``, or the TP32
            all-gather of ``rs_ag``, with either ``moe_out_proj_form``), and the input projection of ``layer`` reads
            them in place of the first chunks of its ring (``decode_block.decode_block_body`` ``kda_w_in_heads``). ``layer`` is a KDA layer of the range after a MoE
            layer of the range, each at most once. The caller picks addresses that no compiler-placed tile uses while a
            head is live (an SBUF map of the program). The values do not change.
        kda_front_ahead: the MoE layer before each KDA layer of an AttnRes block traces that layer's KDA front (page
            counts, conv rows, conv gathers and the staged conv states, and the recurrent-state loads on the ACT queue)
            in the flight of its output collective (the TP32 output all-reduce, or the output all-gather of
            ``moe_out_proj_form`` ``column``; ``decode_block.decode_block_body`` ``kda_front_ahead``). Only with the
            ``bf16`` ``kda_state_dtype``; not with ``kda_state_loads_first`` or ``kda_conv_keep``. The values do not
            change; False (default) keeps the program.
        moe_pad_rows: None (default) keeps the program. The HBM int32 ``[B, 1]`` padding-row input of the step (1 for a
            padding row, 0 for a real row; ``decode_block.decode_block_body`` ``moe_pad_rows``): the MoE layers route
            those rows to no expert (0 router scores before the router-row gather), so a padding row adds no routed row
            on any rank. The other rows keep their bits.
        kda_bf16_update_rows: with the ``decoupled_bf16`` KDA readout (the default above one sequence), the state
            update of every KDA layer takes its bf16 high and low rows from one bf16 TensorE transpose per 4 sequences,
            in place of an fp32 identity matmul, a ScalarE copy and a VectorE subtract per sequence, and adds them in
            one broadcast matmul per head in place of two (``kda_decode_attention`` ``bf16_update_rows``, the form the
            one-server decode turns on at 32 sequences). The transposes read their data with one free dimension
            (``bf16_rows_flat``). The outputs and the pools do not change. No effect at the other readouts. False
            (default) keeps the program.
        moe_in_proj_at_kda_reduce: each MoE layer after a KDA layer loads its in-projection right after the KDA
            attention all-reduce, before the recurrent-state stores in the program, so the loads do not wait for the
            state stores (``decode_block.decode_block_body`` ``moe_in_proj_at_kda_reduce``). Not with
            ``ffn_weights_after_attn``. The values do not change; False (default) keeps the program.
        moe_2srv_selectors_once: the MoE layers' rank one-hots (a rank-row load and 8 VectorE / GpSimd ops per layer)
            are built once per kernel and reused (``decode_block.decode_block_body`` ``moe_2srv_selectors_once``). The
            values do not change; False (default) keeps the program.
        attn_res_sums_after_moe_output: the AttnRes bank sums of the layer after each MoE layer (and of the final mix)
            follow the store of the routed partial into the source of the routed collective in the schedule
            (``decode_block.decode_block_body`` ``row_sums_after_ffn_output``), so VectorE runs the PSUM copies of the
            column-group partial before them. A schedule edge, not a semaphore wait: the values do not change. False
            (default) keeps the program.
        moe_static_alt_tiles: the static expert loop of the 2-server MoE body takes its gate sigmoid / up tanh tiles
            ("act"), and its down PSUM copies ("act_copy"), and its activation evictions (True) from two buffers each
            (``tkg_moe_2srv.latent_moe_decode_2srv`` ``static_alt_tiles``), so that the compiler cannot stack them
            on one SBUF address (the serialized ScalarE / VectorE waits of a collapsed loop). The values do not
            change; False (default) keeps the program.
        kda_null_page_rows: 0 (default) keeps the program. Else the KDA page of every invalid row, which no valid row has:
            with ``kda_batched_conv_stores`` each sequence at this page writes its conv rows to its own rows of the
            page (``decode_model.kimi_k3_decode_model`` ``kda_null_page_rows``; the front traced ahead with
            ``kda_front_ahead`` too). The outputs and the other pages do not change. No effect without
            ``kda_batched_conv_stores``.
        kda_null_page: 0 (default) keeps the program. Else the KDA null page that the caller keeps for no request
            (``decode_model.kimi_k3_decode_model`` ``kda_null_page``; a page other than 0 that no request holds, for
            example block 1): a sequence at it reads and writes like an off one (``kda_decode_attention``
            ``null_page``): zero reads, zero conv-state scatter rows, and every off access goes to it, not to page 0
            (the front traced ahead with ``kda_front_ahead`` too). The rows of the other pages do not change. Not with
            ``kda_null_page_rows``.
        mla_rows_twin: test only, the identity twin of the CP32 rows-form verify (``verify_model_2srv``
            ``mla_k1_rows``): the MLA layers run ``tkg_mla_block.mla_rows_decode_attention`` in place of
            ``mla_decode_attention``, as ``kimi_k3_decode_model`` ``mla_rows_twin``. It changes the per-row sums. B in
            ``tkg_mla_block.ROWS_SEQUENCES`` sequences (one call), the MLA loop path, and no packed loop, page runs, pair
            pages, batched cache write or MLA options. Default off.
        mla_rows_twin_options: with ``mla_rows_twin``, the rows-form keys of the twin (``tkg_mla_block.ROWS_OPTIONS``):
            the values of the verify's ``mla_k1_rows_options`` for the keys in ``tkg_mla_block.ROWS_TWIN_OPTIONS``.
            None (default): with FP8 pages, those keys of ``SERVED_ROWS_OPTIONS`` (the twin of the served rows
            verify); else the plain rows form. {} keeps the plain rows form.
        live_rows: test only, with ``mla_rows_twin``: the runner's INT32 ``[B, 1]`` live-row mask of the twin's
            ``rows_pad_skip`` (as the verify's ``live_rows``), which needs it.
        mla_state_bf16_o: the MLA state reduce-scatter of each layer carries the numerators in BF16 and the sums in
            FP32 (``mla_decode_attention`` ``state_bf16_o``), the decode twin of the verify's ``mla_state_bf16_o``.
            It rounds the numerators: an accuracy-class option. Not with the staged MLA groups. False (default)
            keeps the program.
        moe_glue: None (default) keeps the program, or a dict of the glue options of the MoE body
            (``tkg_moe_2srv.GLUE_KEYS``: ``norm_tail``, ``rows_queue``, ``router_first``, ``out_proj_early``,
            ``selectors_once``, ``routed_compact``, ``store_queues``). The values do not change. ``mx_scores_align``
            also sizes the collective router-row tensors of the entry (65 words in place of 61).
        kda_w_in_scale: None (default, a BF16 ``kda_w_in``), or the scales of an MXFP8 ``kda_w_in`` stack, as in
            ``kimi_k3_decode_model`` (the same stacking as ``kda_w_in``): the KDA input projection of every KDA
            layer runs on the MXFP8 weights (``kda_decode_attention`` ``kda_w_in_scale``).
        mla_in_proj_scale: None (default, a BF16 ``mla_in_proj``), or the scales of an MXFP8 ``mla_in_proj`` stack, as
            in ``kimi_k3_decode_model`` (the same stacking as ``mla_in_proj``): the MLA q_a / kv_a and gate projections of every
            MLA layer run on the MXFP8 weights.
        router_dtype, moe_router_weight: ``bf16`` (default) keeps the program. ``fp16`` routes each MoE layer with the
            FP16 router of ``tkg_moe_2srv.latent_moe_decode_2srv``: ``moe_router_weight`` stacks its
            ``[128, 56, 28]`` FP16 rows over the MoE layers (``[n_moe, 128, 56, 28]``). The collectives do not change.
        moe_in_proj_scale: None (default, a BF16 ``moe_in_proj``), or the scales of an MXFP8 MoE in-projection,
            stacked as ``moe_in_proj`` (``[n_moe, 128, 2, 496]``; ``tkg_moe_2srv.MX_IN_PROJ_SHAPES``):
            ``moe_in_proj`` then stacks the MXFP8 in-projection (``[n_moe, 14, 128, 1984]`` uint8, no router columns),
            and the latent down and shared gate / up of every MoE layer run MXFP8 x MXFP8 (``latent_moe_decode_2srv``
            ``in_proj_scale``). It needs the FP16 router.
        kda_w_out_scale: None (default, a BF16 ``kda_w_out``), or the scales of an MXFP8 padded-lane ``kda_w_out`` stack
            (``tkg_kda.kda_w_out_form``: ``kda_w_out`` then stacks ``[n, 1, 128, 4 H]`` uint8 and the scales
            ``[n, 128, 1, H]``): the KDA output projection of every KDA layer runs MXFP8 x MXFP8. The values change.
        moe_out_proj_scale: None (default, a BF16 ``moe_out_proj``), or the scales of an MXFP8 padded-lane MoE
            out-projection, stacked as ``moe_out_proj`` (``[n_moe, 128, 1, H]``;
            ``tkg_moe_2srv.MX_OUT_PROJ_SHAPES``): ``moe_out_proj`` then stacks ``[n_moe, 1, 128, 4 H]`` uint8 (the
            304 rows in the lanes of 512, the rows past them zero), and the out-projection of every MoE layer runs
            MXFP8 x MXFP8 (``latent_moe_decode_2srv`` ``out_proj_scale``). The values change.
        mla_rs_hold: True (default) runs the state and output reduce-scatters of each MLA layer from persistent HBM
            tensors in two parity sets, also on the MLA loop path, as ``kimi_k3_decode_model`` ``mla_rs_hold`` (the
            kernel workaround for the defect of the switch-mesh reduce-scatter completion in Neuron runtime
            2.35). The values do not change. False keeps the program without the hold, for a runtime with the fix.

    Returns:
        As ``kimi_k3_decode_model``: the stream ``[B, H]`` (with the final weights the ``lm_head`` input), the
        tokens ``[B, 1]`` (``greedy``) and the logits ``[B, V/32]`` (``return_logits``), the debug outputs
        (``debug``), then the page pools.
    """
    restart_calls()
    B, H = stream.shape
    layout = world_layout(world_size)
    dp = layout["dp"]
    routed_tp, moe_mx_tile = routed_layers_options(world_size, routed_layers, routed_tp, moe_mx_tile)
    n_tiles = H // P_MAX
    num_layers = norm_weights.shape[0]
    kernel_assert(dp_size is None or dp_size == dp, f"dp_size {dp_size} is not world_size / 32 = {dp}")
    kernel_assert(moe_form in MOE_FORMS, f"moe_form must be one of {MOE_FORMS}, got {moe_form!r}")
    max_replica_rows = MAX_REPLICA_ROWS if max_replica_rows is None else max_replica_rows
    kernel_assert(
        max_replica_rows <= MAX_REPLICA_ROWS,
        f"max_replica_rows={max_replica_rows}: the entry takes at most MAX_REPLICA_ROWS = {MAX_REPLICA_ROWS} sequences "
        f"of a replica (the attention bound {ATTENTION_MAX_SEQUENCES}, the MoE body {MOE_MAX_REPLICA_ROWS})",
    )
    kernel_assert(
        1 <= B <= max_replica_rows,
        f"B={B} sequences of a replica: use 1 <= B <= {max_replica_rows} (max_replica_rows, at most MAX_REPLICA_ROWS = "
        f"{MAX_REPLICA_ROWS})",
    )
    kernel_assert(moe_gather in MOE_GATHERS, f"moe_gather must be one of {MOE_GATHERS}, got {moe_gather!r}")
    kernel_assert(
        moe_routed_reduce in ROUTED_REDUCES,
        f"moe_routed_reduce must be one of {ROUTED_REDUCES}, got {moe_routed_reduce!r}",
    )
    kernel_assert(
        not moe_routed_stats_ag or moe_routed_reduce == "rs_ag",
        f"moe_routed_stats_ag needs moe_routed_reduce 'rs_ag', got {moe_routed_reduce!r}",
    )
    kernel_assert(
        moe_out_proj_form in OUT_PROJ_FORMS,
        f"moe_out_proj_form must be one of {OUT_PROJ_FORMS}, got {moe_out_proj_form!r}",
    )
    kernel_assert(
        moe_out_proj_form == "reduced" or not moe_routed_stats_ag,
        f"moe_out_proj_form {moe_out_proj_form!r} norms the full routed rows, and moe_routed_stats_ag gathers only the "
        "sums of squares: use moe_routed_reduce 'ar', or 'rs_ag' without moe_routed_stats_ag",
    )
    if moe_routed_reduce != "ar":
        kernel_assert(
            (world_size, B) in ROUTED_RS_AG_FORMS + ROUTED_RS_AG_FORMS_4SRV or moe_routed_twin,
            f"moe_routed_reduce {moe_routed_reduce!r} takes the (world_size, B) forms "
            f"{ROUTED_RS_AG_FORMS + ROUTED_RS_AG_FORMS_4SRV} only, got "
            f"({world_size}, {B})",
        )
        kernel_assert(
            cc_source_pingpong,
            f"moe_routed_reduce {moe_routed_reduce!r} alternates its collective tensors by layer parity: it needs "
            "cc_source_pingpong",
        )
        kernel_assert(
            moe_rs_ag_allow_dr_keys or not (mla_options or {}).get("pack_dr_keys", False),
            f"moe_routed_reduce {moe_routed_reduce!r} with the FP8 double-row MLA keys (mla_options pack_dr_keys, form B) "
            "missed a 1M needle (depth 0.9) on Trn3: pass moe_rs_ag_allow_dr_keys=True only after a passing 1M needle row",
        )
    kernel_assert(first_layer + num_layers <= NUM_LAYERS, f"The model has {NUM_LAYERS} layers")
    kernel_assert(
        not attn_res_kda_mlp_early_in_front or B <= _KDA_FRONT_ROWS_MAX_TOKENS or kda_batched_conv_stores,
        f"attn_res_kda_mlp_early_in_front above {_KDA_FRONT_ROWS_MAX_TOKENS} sequences needs kda_batched_conv_stores: "
        "the block keeps the bytes of the conv-state scatter by reading its views after the feed-forward norm",
    )
    kernel_assert(
        mla_stage_cc_split in MLA_STAGE_SPLITS,
        f"mla_stage_cc_split must be one of {tuple(MLA_STAGE_SPLITS)}, got {mla_stage_cc_split!r}",
    )
    split_sites = MLA_STAGE_SPLITS[mla_stage_cc_split]
    pair_shared = check_pair_shared_forms(pair_shared_cc_dsts, mla_stage_split=split_sites)
    kernel_assert(
        "moe_gather" not in pair_shared,
        "pair_shared_cc_dsts moe_gather: the gathers of the multi-server MoE body stay private",
    )
    kernel_assert(
        "moe_routed" not in pair_shared or moe_routed_reduce == "ar",
        f"pair_shared_cc_dsts moe_routed needs moe_routed_reduce 'ar' (the world all-reduce), got {moe_routed_reduce!r}",
    )
    check_column_private(moe_out_proj_form, pair_shared)
    kernel_assert(
        lm_head_weight is not None or not (greedy or return_logits),
        "greedy and return_logits need lm_head_weight",
    )
    kernel_assert(banks_snapshot(first_layer), "The AttnRes bank starts empty: start at a bank snapshot layer")
    kernel_assert(
        not mla_rows_twin
        or (
            B in ROWS_SEQUENCES
            and mla_loop_path
            and not (mla_packed_loop or mla_pack_page_runs or mla_pair_pages or mla_batched_cache_write or mla_options)
        ),
        f"mla_rows_twin is the twin of the CP32 rows-form verify: it needs B in {ROWS_SEQUENCES} (one call), the MLA "
        "loop path, and no packed loop, page runs, pair pages, batched cache write or MLA options",
    )
    if mla_rows_twin and mla_rows_twin_options is None and mla_fp8_pages:
        # The twin of the served rows verify (verify_model_2srv.served_defaults): the keys that set the bits of a row.
        mla_rows_twin_options = {k: v for k, v in SERVED_ROWS_OPTIONS.items() if k in ROWS_TWIN_OPTIONS}
    kernel_assert(
        (mla_rows_twin or not mla_rows_twin_options) and set(mla_rows_twin_options or ()) <= set(ROWS_OPTIONS),
        f"mla_rows_twin_options needs mla_rows_twin and takes the keys {ROWS_OPTIONS}, got {mla_rows_twin_options}",
    )
    kernel_assert(
        not (mla_rows_twin_options or {}).get("rows_pad_skip") or live_rows is not None,
        "the twin's rows_pad_skip needs live_rows (the runner's live-row mask): a live row can sit at position 0",
    )
    kernel_assert(
        not (kda_front_ahead and kda_state_loads_first),
        "kda_front_ahead loads the recurrent states ahead: it does not take kda_state_loads_first",
    )
    kernel_assert(
        not kda_front_ahead or (kda_state_dtype == "bf16" and not kda_conv_keep),
        f"kda_front_ahead takes the bf16 KDA state without kda_conv_keep, got kda_state_dtype {kda_state_dtype!r} and "
        f"kda_conv_keep {kda_conv_keep}",
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
    kda_w_in_heads = _kda_w_in_heads(kda_w_in_head, kda_w_in, first_layer, num_layers, totals[0])
    kda_ahead_carry = {}
    # The CSR gathers of the call's dynamic packed MoE layers: "staged" (None) while they fit the DGE table.
    moe_dynamic_csr = None
    if moe_routed_dynamic:
        moe_dynamic_csr = call_csr_gather(world_size, routed_layers, first_layer, num_layers, routed_tp, moe_mx_tile)
        moe_dynamic_csr = None if moe_dynamic_csr == "staged" else moe_dynamic_csr

    def make_moe_body(body_tp: int, body_tile: int):
        return moe_2srv_body(
            tp_group,
            world_size,
            body_tp,
            moe_gather,
            WIDE_HALF_WEIGHT_BUFFERS if moe_wide_half_weight_buffers is None else moe_wide_half_weight_buffers,
            expert_call_tokens=moe_expert_call_tokens,
            routed_reduce=moe_routed_reduce,
            stats_gather=moe_routed_stats_ag,
            **({"drop_selection": True} if moe_router_rows_drop_selection else {}),
            **({"out_proj_form": moe_out_proj_form} if moe_out_proj_form != "reduced" else {}),
            **({"quantize_first": True} if moe_quantize_first else {}),
            **({"weights_after_gather": True} if moe_weights_after_gather else {}),
            **({"head_row_chunks": True} if moe_head_row_chunks else {}),
            **({"rows_latent": "fp8"} if moe_rows_latent_fp8 else {}),
            **({"rows_load_split": moe_rows_load_split} if moe_rows_load_split != 1 else {}),
            **({"routed_own_rows": True} if moe_routed_own_rows else {}),
            **({"latent_gather_mx": True} if moe_latent_gather_mx else {}),
            **({"glue": dict(moe_glue)} if moe_glue else {}),
            **({"mx_tile": body_tile} if body_tile is not None else {}),
            # The dynamic form runs the splits of sub-512 MX tiles (body_tile); a served split of a plan keeps its loop.
            **({"routed_dynamic": True} if moe_routed_dynamic and body_tile is not None else {}),
            # Its CSR form goes with it (latent_moe_decode_2srv refuses routed_dynamic_csr without routed_dynamic).
            **(
                {"routed_dynamic_csr": moe_dynamic_csr} if moe_dynamic_csr is not None and body_tile is not None else {}
            ),
        )

    moe_body = make_moe_body(routed_tp, moe_mx_tile)
    # A routed_layers plan whose layers of the call take other tags: each MoE layer runs the split and the stacks of its
    # tag (moe_routed_plan); None (no plan, or one tag) keeps the single split.
    routed_plan = call_plan(world_size, routed_layers, first_layer, num_layers)
    alternates = dict(
        zip(
            ALT_STACK_ARGUMENTS,
            (
                routed_alt0_expert_gate_up_weights,
                routed_alt0_expert_gate_up_weights_scale,
                routed_alt0_expert_down_weights,
                routed_alt0_expert_down_weights_scale,
                routed_alt1_expert_gate_up_weights,
                routed_alt1_expert_gate_up_weights_scale,
                routed_alt1_expert_down_weights,
                routed_alt1_expert_down_weights_scale,
            ),
            strict=True,
        )
    )
    check_alternates(alternates, routed_plan)
    kernel_assert(
        not moe_routed_dynamic or moe_mx_tile is not None or routed_plan is not None,
        "moe_routed_dynamic runs the splits of sub-512 MX tiles: it needs moe_mx_tile, or a routed_layers plan with "
        "such a tag",
    )
    if routed_plan is not None:
        moe_body, moe_weights = plan_moe(
            routed_plan,
            routed_layers[0],
            moe_body,
            make_moe_body,
            moe_weights,
            alternates,
            moe_layers_of(first_layer, num_layers),
        )
    dense_body = dense_ffn_2srv(world_size)
    kda_body = _kda_body(
        kda_conv_dtype,
        kda_w_in_buffers,
        kda_state_loads_first,
        kda_batched_conv_stores,
        kda_readout,
        kda_conv_load_gather,
        kda_conv_stores_early,
        kda_state_loads_early,
        state_dtype=kda_state_dtype,
        **({"bf16_update_rows": True} if kda_bf16_update_rows else {}),
        null_page_rows=kda_null_page_rows,
        null_page=kda_null_page,
        paired_pages=kda_paired_pages,
    )
    if kda_bf16_update_rows:
        kda_body = kda_options_body(kda_body, bf16_rows_flat=True)
    if kda_front_ahead:
        kda_body = kda_front_ahead_body(
            kda_body,
            batched_conv_stores=kda_batched_conv_stores,
            conv_load_gather=kda_conv_load_gather,
            null_page_rows=kda_null_page_rows,
            null_page=kda_null_page,
        )

    region = None
    if mla_loop_path:
        region = ExplicitSbuf()
        install_identities(region)
    if moe_glue and moe_glue.get("selectors_once"):
        # The selectors live for the whole kernel: on the loop path they take heap addresses before the first block.
        # One name for each tag's selectors: the served name for the default split (plan_selector_splits).
        default_tag = routed_layers[0] if routed_layers is not None else None
        for sel_name, split_tp, split_tile in plan_selector_splits(
            world_size, routed_tp, moe_mx_tile, routed_plan, default_tag
        ):
            moe_split = moe_2srv_world_split(world_size, split_tp, split_tile)
            moe_2srv_glue_selectors(
                rank_info,
                min(P_MAX, moe_split["dp"] * B),
                moe_split,
                sel_name,
                SbufTiles(region, heap=True) if region is not None else None,
            )
    # The token-major G1a rows feed the compaction of expert calls of more than 128 tokens, with the KDA expert ring
    # of the verify entry; the decode entry has neither (the kill switch drops the key with the compaction).
    kernel_assert(
        not ((moe_glue or {}).get("mx_rows_stream") and moe_2srv_module.ROUTED_COMPACT_ON),
        "mx_rows_stream is a verify form: its rows feed the compaction of expert calls of more than 128 tokens with "
        "the KDA expert ring, and the decode entry has neither; drop it from the decode moe_glue",
    )
    if moe_glue and moe_glue.get("routed_compact") and moe_2srv_module.ROUTED_COMPACT_ON:
        # routed_compact's slot-table constants: built once for the kernel, before its first block.
        moe_2srv_compact_constants(rank_info, sb=SbufTiles(region, heap=True) if region is not None else None)
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
    greedy_tail = lm_head_weight is not None and greedy
    # Off the loop path the collective tiles are views of two arenas, and the argmax tiles keep the destination arena
    # non-empty when no feed-forward tile is in it (moe=False below).
    tail_ranks = TAIL_TP_RANKS if greedy_tail or not mla_loop_path else 0
    # The attention and tail sites of the one-server DP4 form; the feed-forward bodies bring their own HBM tensors.
    sources = collective_sources(
        B,
        n_tiles,
        stream.dtype,
        region,
        tail_ranks,
        moe=False,
        mla_stage=mla_staged(B, mla_loop_path),
        pingpong=cc_source_pingpong,
        dp_size=dp,
        **({"mla": False} if mla_rows_twin else {}),
        # The BF16-o state needs the FP32 HBM state pair of each MLA group, which the loop path has not.
        **({"mla_state_hbm": True} if mla_state_bf16_o else {}),
        **(
            {"mla_stage_split": split_sites, "mla_stage_unit_tokens": mla_stage_unit_tokens(mla_pack_pairs)}
            if split_sites
            else {}
        ),
        **({"pair_shared_cc_dsts": pair_shared} if pair_shared else {}),
        **({"mla_rs_hold": True} if mla_rs_hold else {}),
    )
    ps = {"pair_shared": pair_shared} if pair_shared else {}
    moe_hbm, moe_second = moe_2srv_collective_hbm(
        B,
        world_size,
        routed_tp,
        moe_gather,
        routed_reduce=moe_routed_reduce,
        stats_gather=moe_routed_stats_ag,
        **({"drop_selection": True} if moe_router_rows_drop_selection else {}),
        **ps,
        **({"out_proj_form": moe_out_proj_form} if moe_out_proj_form != "reduced" else {}),
        **({"rows_latent": "fp8"} if moe_rows_latent_fp8 else {}),
        **({"latent_gather_mx": True} if moe_latent_gather_mx else {}),
        # The moe_glue key mx_scores_align (tkg_moe_2srv.GLUE_KEYS): the 65-word G1a router rows.
        **({"mx_scores_align": True} if (moe_glue or {}).get("mx_scores_align") else {}),
        # The moe_glue key mx_rows_stream: the 64-word token-major G1a router rows (the kill switch
        # ROUTED_COMPACT_ON False drops it with the compaction, as moe_2srv_body does).
        **(
            {"mx_rows_stream": True}
            if (moe_glue or {}).get("mx_rows_stream") and moe_2srv_module.ROUTED_COMPACT_ON
            else {}
        ),
        rows_dst_pingpong=moe_rows_dst_pingpong,
        **({"mx_tile": moe_mx_tile} if moe_mx_tile is not None else {}),
    )
    kernel_assert(MOE_SOURCES_KEY not in sources and DENSE_SOURCES_KEY not in sources, "feed-forward source keys clash")
    sources[MOE_SOURCES_KEY] = moe_hbm
    sources[DENSE_SOURCES_KEY] = dense_collective_hbm(B, world_size, **ps)
    if cc_source_pingpong:
        sources["pingpong"] = {**sources.get("pingpong", {}), MOE_SOURCES_KEY: moe_second}
    if kda_conv_keep:
        # The conv-state stores of the last two KDA layers, across the blocks (decode_block_body).
        sources = {**sources, "kda_conv_waits": []}
    if mla_pack_shared_hbm and mla_packed_loop and mla_loop_path:
        # The layers write the shared rows and items in turn, as the staged groups do (decode_block.collective_sources).
        sources.setdefault("mla_pack_hbm", {})
    if mla_rows_twin:
        # The sites of the loop-path k1 verify, as kimi_k3_decode_model: cleared first, so no tile takes their bytes.
        rows_kw = {"q_hbm": True, "out_ag_call": True}
        rows_set = {
            **mla_block_collective_sources(B, ROWS_LOCAL_HEADS, SbufTiles(region, heap=True), with_rs=True, **rows_kw),
            **mla_block_collective_hbm(B, ROWS_LOCAL_HEADS, name="_rows_twin", **rows_kw),
        }
        for tile in [rows_set["qkv"], rows_set["qkv_dst"], *rows_set["rs"], rows_set["ag_call"]]:
            nisa.memset(dst=tile, value=0.0)
        sources["mla_rows"] = rows_set
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
            tuple(_group_stack(t, moe_k, n_moe, totals[2]) for t in moe_weights),
            tuple(_group_stack(t, dense_k, n_dense, totals[3]) for t in dense_weights),
            layer_pools[li : li + count],
            _rows(kda_page_ids, kda_k, n_kda),
            _rows(mla_block_table, mla_k, n_mla),
            _rows(mla_slot, mla_k, n_mla),
            position,
            rank_info,
            tp_group,
            dp_group,
            world_group,
            start,
            eps,
            kda_body,
            mla_rows_decode_attention if mla_rows_twin else mla_decode_attention,
            moe_body,
            dense_body,
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
            mla_zeroed_pages=mla_zeroed_pages,
            mla_pair_pages=mla_pair_pages,
            kernel_first_layer=first_layer,
            lora_norm_eps=lora_norm_eps,
            moe_half_weight_buffers=moe_half_weight_buffers,
            **({"moe_weight_buffers": moe_weight_buffers} if moe_weight_buffers is not None else {}),
            **({"moe_scale_buffers": moe_scale_buffers} if moe_scale_buffers is not None else {}),
            **({"moe_transpose_lookahead": moe_transpose_lookahead} if moe_transpose_lookahead is not None else {}),
            **({"moe_cross_expert_order": True} if moe_cross_expert_order else {}),
            **({"moe_cross_expert_transpose_blocks": True} if moe_cross_expert_transpose_blocks else {}),
            **({"moe_token_tiles_transpose_blocks": True} if moe_token_tiles_transpose_blocks else {}),
            **({"moe_transpose_block": True} if moe_transpose_block else {}),
            moe_routed_store_form=moe_routed_store_form,
            dp_size=dp,
            mla_options=(
                {**(mla_options or {}), "pack_sum_after_copy": True} if mla_pack_sum_after_copy else mla_options
            )
            if not (mla_state_bf16_o or mla_rows_twin_options)
            else {
                **(mla_options or {}),
                **({"pack_sum_after_copy": True} if mla_pack_sum_after_copy else {}),
                **({"state_bf16_o": True} if mla_state_bf16_o else {}),
                **dict(mla_rows_twin_options or {}),
                **({"live_rows": live_rows} if live_rows is not None else {}),
            },
            **({"ffn_weights_after_attn": True} if ffn_weights_after_attn else {}),
            **({"moe_sums_after_routed": True} if moe_sums_after_routed else {}),
            mix_cache=mix_cache,
            early_ffn_gate=bool(early_ffn_gate),
            attn_res_cached_bank_sums=bank_sums,
            **({"attn_res_early_sums": True} if attn_res_early_sums else {}),
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
            **(
                {"attn_res_kda_mlp_early_entries": attn_res_kda_mlp_early_entries}
                if attn_res_kda_mlp_early_entries
                else {}
            ),
            **({"attn_res_kda_mlp_early_in_front": True} if attn_res_kda_mlp_early_in_front else {}),
            **({"attn_res_scalar_squares": attn_res_scalar_squares} if attn_res_scalar_squares else {}),
            **({"norm_scalar_squares": norm_scalar_squares} if norm_scalar_squares else {}),
            **({"attn_res_scalar_squares_order": True} if attn_res_scalar_squares_order else {}),
            **({"attn_res_scalar_products": attn_res_scalar_products} if attn_res_scalar_products else {}),
            **({"attn_res_tensor_sums": True} if attn_res_tensor_sums else {}),
            **({"attn_res_one_round_norm": True} if attn_res_one_round_norm else {}),
            **({"attn_res_tensor_sums_idle": True} if attn_res_tensor_sums_idle else {}),
            **({"kda_conv_keep": True} if kda_conv_keep else {}),
            **({"kda_w_in_heads": kda_w_in_heads, "kda_ahead_carry": kda_ahead_carry} if kda_w_in_heads else {}),
            **({"kda_front_ahead": True} if kda_front_ahead else {}),
            **({"moe_pad_rows": moe_pad_rows} if moe_pad_rows is not None else {}),
            **({"moe_in_proj_at_kda_reduce": True} if moe_in_proj_at_kda_reduce else {}),
            **({"moe_2srv_selectors_once": True} if moe_2srv_selectors_once else {}),
            **({"row_sums_after_ffn_output": True} if attn_res_sums_after_moe_output else {}),
            **({"moe_static_alt_tiles": moe_static_alt_tiles} if moe_static_alt_tiles else {}),
            **router_block_options(router_dtype, moe_router_weight, moe_k, n_moe, totals[2]),
            **in_proj_block_options(moe_in_proj_scale, moe_k, n_moe, totals[2]),
            **out_proj_block_options(moe_out_proj_scale, moe_k, n_moe, totals[2]),
        )
        starts = [k + n for k, n in zip(starts, counts, strict=True)]
    kernel_assert(not kda_ahead_carry, f"kda_w_in heads loaded but not read: layers {sorted(kda_ahead_carry)}")

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
            tiles = (sources["argmax_src"], sources["argmax_dst"], sources["argmax_token"]) if greedy_tail else None
            token = lm_head_greedy_hT(
                s,
                lm_head_weight,
                tp_group,
                "tail",
                sample=greedy,
                logits_out=logits,
                tp_size=TAIL_TP_RANKS,
                gather_tiles=tiles,
            )
        if greedy:
            tokens = nl.ndarray((B, 1), dtype=nl.int32, buffer=nl.shared_hbm, name="tokens")
            nisa.dma_copy(dst=tokens, src=token, name="tail_tokens_store")
            tail += (tokens,)
        if return_logits:
            tail += (logits,)
    # Keep: a first-of-step collective skips its handshake, so no later tensor may take its destination's bytes.
    hold_to_end(first_cc_destinations(sources), list(tail), out)
    if region is not None:
        check_identities(region)
    if debug:
        return (out, *tail, layer_streams, layer_debug, *pools)
    return (out, *tail, *pools)
