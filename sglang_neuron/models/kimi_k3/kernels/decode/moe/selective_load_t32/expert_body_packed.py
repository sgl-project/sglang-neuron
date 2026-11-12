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

"""Standalone routed-expert body of the kimi_t32 subpackage (T = 32 tokens, K = 16, H = 3584, I = 24, E = 896): the
routing comes from the caller (the top-K index and affinities, and for dedupe the selected scores), and the experts
read a bf16 latent. No router and no norm in the kernel."""

from typing import Optional, Sequence

import nki
import nki.language as nl

from ...utils.allocator import SbufManager
from ...utils.kernel_assert import kernel_assert
from ...utils.logging import get_logger
from .latent_mx_quant_packed import latent_scratch_tiles, quantize_latent_mx
from .selective_load_packed import cached_alloc, group_body, group_body_phases, named_modules, run_phases

_P_MAX = 128
_X4 = 4
_SCORE_SUM_EPS = 1e-20  # the affinities divide by the selected score sum + this


def expert_body_packed(
    latent: nl.NkiTensor,
    expert_gate_up_packed: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_down_weights_scale: nl.NkiTensor,
    expert_index: nl.NkiTensor,
    expert_affinities: nl.NkiTensor,
    selected_scores: Optional[nl.NkiTensor] = None,
    sbm: Optional[SbufManager] = None,
    sbuf_reserve: int = 0,
    output: Optional[nl.NkiTensor] = None,
    output_row: int = 0,
    expert_offset: int = 0,
    n_experts: Optional[int] = None,
    dedupe_experts: Optional[str] = None,
    dedupe_static_groups: int = 1,
    dedupe_block_groups: Optional[Sequence[int]] = None,
    dedupe_block_mode: str = "binary",
    dedupe_early_indices: bool = True,
    situ: Optional[tuple] = (4.0, 25.0),
    sigmoid_from_tanh: bool = False,
    affinity_dtype: str = "bf16",
    group0_queues: str = "sp_act",
    sp_group_period: int = 0,
    gate_up_slots: int = 4,
    moving_slots: int = 3,
    down_slots: int = 4,
    dedupe_prefetch: bool = False,
    gate_up_pair_gathers: bool = False,
    evict_split: str = "rows",
    name_suffix: Optional[str] = None,
    persistent_buffers: bool = False,
    quantize_first: bool = False,
    merged_down_gathers: bool = True,
    latent_scratch: str = "ring",
    dedupe_late_trips: bool = False,
    dedupe_fold_first: bool = False,
    setup_memsets: str = "pool",
    dedupe_onehot_parts: bool = False,
    valid_tokens: Optional[int] = None,
    skip_down_padding: bool = False,
    dedupe_order: str = "id",
    down_lag: int = 1,
    skip_empty_slots: bool = False,
    weights_arena: bool = False,
    dedupe_early_first: bool = False,
    scales_after_gate_up: bool = False,
    pair_head_in_ring: bool = False,
    act_slots: int = 2,
    fold_dma: bool = False,
    skip_scale_duplicates: bool = False,
    down_psum_dtype: str = "bf16",
    rank_info: Optional[nl.NkiTensor] = None,
    stationary_slots: int = 3,
    head_finish_first: bool = False,
    dedupe_token_rank: str = "tiles",
    dedupe_index_first: bool = False,
    dedupe_precomputed: Optional[tuple] = None,
    prefetch_groups: int = 0,
    prefetch_load: str = "gather",
    prefetch_indices: Optional[nl.NkiTensor] = None,
    fold_store_order: str = "columns",
    fold_store_cols: int = 512,
    dedupe_trips_after_stores: bool = False,
    two_phase_state: Optional[dict] = None,
) -> Optional[nl.NkiTensor]:
    """Routed experts of T tokens on the caller's routing: out[t] = sum over k of a[t, k] * SiTU(x W_g, x W_u) W_d
    of expert expert_index[t, k] (SiLU(gate) * up with situ=None), x = the MXFP8 latent of token t.

    Dimensions:
        T: tokens (32). K: experts per token (16). H: latent size. E: experts of one layer.

    Args:
        latent (nl.NkiTensor): bfloat16 [T, H] @HBM. Quantized to MXFP8 here without a norm, in groups of 32
            consecutive values of one token (quantize_latent_mx); the gate/up weights use h = 512*hb + 4*p + q.
        expert_gate_up_packed (nl.NkiTensor): uint8 [E, 128, 768] @HBM, or stacked [L*E, ...] / [L, E, ...].
        expert_down_weights (nl.NkiTensor): float4_e2m1fn_x4 [E, 8, H] @HBM (or stacked).
        expert_down_weights_scale (nl.NkiTensor): uint8 [E, H] @HBM (or stacked), or [E, 2, H] (two down scale
            groups, with rank_info and down_psum_dtype="fp32"; see group_body).
        expert_index (nl.NkiTensor): int32 [T, K] @HBM, the routed experts of each token.
        expert_affinities (nl.NkiTensor): float32 [T, K] @HBM, their affinities (the baseline reads them).
        selected_scores (Optional[nl.NkiTensor]): float32 [T, E] @HBM: the score of each selected expert, 0
            elsewhere. Needed with dedupe_experts="dynamic": the compaction reads it, and the affinities are the
            scores times 1 / (their sum over E + 1e-20) (computed here).
        sbm (Optional[SbufManager]): the caller's SBUF manager (the body opens its own stack scope). None: a
            manager of SBUF without the top sbuf_reserve bytes.
        sbuf_reserve (int): with sbm=None, the SBUF bytes per partition at the top that stay free. Default 0.
        output, output_row, expert_offset, n_experts, dedupe_experts, dedupe_static_groups, dedupe_block_groups,
            dedupe_block_mode, dedupe_early_indices, situ, sigmoid_from_tanh, affinity_dtype, group0_queues,
            sp_group_period, gate_up_slots, moving_slots, down_slots, dedupe_prefetch, gate_up_pair_gathers,
            evict_split, name_suffix, quantize_first, merged_down_gathers, dedupe_late_trips, dedupe_fold_first,
            setup_memsets, dedupe_onehot_parts, valid_tokens, skip_down_padding, dedupe_order, down_lag,
            skip_empty_slots, weights_arena, dedupe_early_first, scales_after_gate_up, pair_head_in_ring, act_slots,
            fold_dma, skip_scale_duplicates, down_psum_dtype, rank_info, stationary_slots, head_finish_first,
            dedupe_token_rank, dedupe_index_first, prefetch_groups, prefetch_load, prefetch_indices,
            fold_store_order, fold_store_cols, dedupe_trips_after_stores: as in group_body. With dedupe_index_first the selected-score sum and its reciprocal (the affinity normalization)
            follow the early gather indices (late_setup of group_body).
        dedupe_precomputed (Optional[tuple]): dedupe: (indices, affinities) @HBM, the compaction results of the
            caller (see group_body): the body does not load selected_scores and skips the compaction. With a third
            item pool_cols > 0, the first gather waits for the loaded expert_index and then a stand-in Pool op (see
            group_body). Default None.
        two_phase_state (Optional[dict]): internal (expert_body_start): latent is None and the call stops before
            the latent quantization; the dict gets the state of the started call (see expert_body_start).
        latent_scratch (str): where the temporaries of the latent quantization go (the bf16 latent and its swizzled
            copy, 8,960 bytes per partition at T = 32, H = 3584). "ring" (default): a stack scope at the addresses of
            the gate/up row ring of the body, so the first gate/up gather waits for the latent quantization (in a
            program with many calls, also for the PE work of the previous call). "heap": their own heap space (SBUF
            peak + 8,960 bytes), so the gathers do not wait for the latent quantization. The result does not change.
        persistent_buffers (bool): the calls on one SbufManager (sbm) share their SBUF buffers when the name, shape,
            dtype and address match (the cache lives on the manager), so the DMAs of every call have the destinations of
            the first call and can reuse its DMA descriptor table entries. In a program with many calls (for example
            one per layer), the issue cost of the DMAs then stays flat. Needs sbm. Default False. The result does not
            change. The defaults here are the
            latent-MoE semantics (SiTU beta 4, linear_beta 25, bf16 affinities).

    Returns:
        output (nl.NkiTensor): bfloat16 [T, H] @HBM (or the caller's output with rows output_row .. + T written).

    Notes:
        - PSUM: all 8 banks at fixed addresses (bank 0 also holds the latent transposes before the body).
        - With dedupe_experts="dynamic" the affinities come from selected_scores, so an affinity can differ from
          expert_affinities in the last bits (the sum over E runs in another order).
    """
    nisa, nl = named_modules(name_suffix)
    two_phase = two_phase_state != None
    if two_phase:
        # start: no latent yet (T from the routing, H from the down weights [.., 8, H])
        T, H = expert_index.shape[0], expert_down_weights.shape[-1]
    else:
        T, H = latent.shape
    K = expert_index.shape[1]
    n_h512 = H // 512
    dynamic = dedupe_experts == "dynamic"
    kernel_assert(not dynamic or selected_scores != None, "dedupe_experts='dynamic' needs selected_scores")
    if n_experts == None:
        if selected_scores != None:
            n_experts = selected_scores.shape[1]
        elif len(expert_gate_up_packed.shape) == 4:
            n_experts = expert_gate_up_packed.shape[1]
        else:
            n_experts = expert_gate_up_packed.shape[0]
    E = n_experts
    kernel_assert(
        selected_scores == None or tuple(selected_scores.shape) == (T, E), f"selected_scores must be [{T}, {E}]"
    )
    kernel_assert(not persistent_buffers or sbm != None, "persistent_buffers needs the caller's sbm")
    if sbm == None:
        sbm = SbufManager(
            0, nl.tile_size.total_available_sbuf_size - sbuf_reserve, get_logger("expert_body_packed_kimi_t32")
        )
    cache = None
    if persistent_buffers:
        if not hasattr(sbm, "kimi_t32_buffers"):
            sbm.kimi_t32_buffers = {}
        cache = sbm.kimi_t32_buffers
    alloc_heap = cached_alloc(sbm, name_suffix, cache, heap=True)
    heap_depth = len(sbm.heap)  # the heap tiles of this call return to the region at the end
    # The hand-off tiles on the heap: the MXFP8 latent, the routing on token partitions.
    hidden_qtz = alloc_heap((_P_MAX, n_h512 * T), nl.float8_e4m3fn_x4, name="body_hidden_qtz")
    hidden_scale = alloc_heap((_P_MAX, n_h512 * T), nl.uint8, name="body_hidden_scale")
    index_tk = alloc_heap((T, K), nl.uint32, name="body_expert_index")
    affinity_tk = alloc_heap((T, K), nl.float32, name="body_expert_affinity")
    routing_loads = [  # the routing loads (expert_body_start: a caller can order its work after them)
        nisa.dma_copy(
            dst=index_tk.view(nl.int32),
            src=expert_index,
            dge_mode=nisa.dge_mode.hwdge,
            engine=nisa.engine.sync,
            name="body_load_index",
        ),
        nisa.dma_copy(
            dst=affinity_tk,
            src=expert_affinities,
            dge_mode=nisa.dge_mode.hwdge,
            engine=nisa.engine.sync,
            name="body_load_affinity",
        ),
    ]
    scores, recip = None, None
    late_setup = None
    if dynamic:
        scores = alloc_heap((T, E), nl.float32, name="body_selected_scores")
        recip = alloc_heap((T, 1), nl.float32, name="body_score_sum_recip")
        if dedupe_precomputed == None:
            routing_loads.append(
                nisa.dma_copy(
                    dst=scores,
                    src=selected_scores,
                    dge_mode=nisa.dge_mode.hwdge,
                    engine=nisa.engine.sync,
                    name="body_load_scores",
                )
            )

        def score_sum_recip(after=None) -> None:
            """recip = 1 / (sum over E of the selected scores + 1e-20)."""
            first = nisa.tensor_reduce(dst=recip, op=nl.add, data=scores, axis=1, name="body_score_sum")
            if after is not None:
                first.depends_on(after, kind=nisa.dependency.schedule)
            nisa.tensor_scalar(dst=recip, data=recip, op0=nl.add, operand0=_SCORE_SUM_EPS, name="body_score_sum_eps")
            nisa.reciprocal(dst=recip, data=recip, name="body_score_sum_recip_op")

        if dedupe_index_first:
            late_setup = score_sum_recip
        elif dedupe_precomputed == None:
            score_sum_recip()
    # The MXFP8 latent (PSUM bank 0 for the transposes, before the body uses it). Its temporaries: a stack scope at the
    # addresses of the body's gate/up row ring, or (latent_scratch="heap") heap tiles that no body buffer shares.
    kernel_assert(latent_scratch in ("ring", "heap"), f"latent_scratch must be 'ring' or 'heap', got {latent_scratch}")
    on_ring = latent_scratch == "ring"
    latent_prefix = "body_" + (name_suffix + "_" if name_suffix else "")

    def quantize(latent: nl.NkiTensor, scratch: Optional[tuple] = None):
        """The MXFP8 latent into hidden_qtz / hidden_scale (scratch: the tiles of latent_scratch_tiles); returns the
        quantize_mx instruction."""
        latent_ps = nl.ndarray(
            (_P_MAX, n_h512 * _X4 * T), dtype=nl.bfloat16, buffer=nl.psum, address=(0, 0), name="body_latent_ps"
        )
        _, quantized = quantize_latent_mx(
            latent,
            cached_alloc(sbm, None, cache, heap=not on_ring),  # the prefix carries name_suffix
            latent_ps,
            hidden_qtz,
            hidden_scale,
            nisa.engine.scalar,
            latent_prefix,
            scratch=scratch,
        )
        return quantized

    latent_tiles = None
    if two_phase:
        # The latent scratch on the heap now (the heap layout of the one-phase body); its quantization in finish.
        kernel_assert(not on_ring, "expert_body_start needs latent_scratch='heap' (the ring holds the started loads)")
        latent_tiles = latent_scratch_tiles(
            cached_alloc(sbm, None, cache, heap=True), T, H, two_phase_state["latent_in_sbuf"], latent_prefix
        )
    else:
        if on_ring:
            sbm.open_scope()
        quantize(latent)
        if on_ring:
            sbm.close_scope()
    body_options = {}
    if situ != None:
        body_options.update(situ=situ, sigmoid_from_tanh=sigmoid_from_tanh)
    # The schedule levers of group_body, passed only when they are on.
    schedule_options = {}
    if dedupe_late_trips:
        schedule_options.update(dedupe_late_trips=True)
    if dedupe_fold_first:
        schedule_options.update(dedupe_fold_first=True)
    if setup_memsets != "pool":
        schedule_options.update(setup_memsets=setup_memsets)
    if dedupe_onehot_parts:
        schedule_options.update(dedupe_onehot_parts=True)
    if valid_tokens != None:
        schedule_options.update(valid_tokens=valid_tokens)
    if skip_down_padding:
        schedule_options.update(skip_down_padding=True)
    if dedupe_order != "id":
        schedule_options.update(dedupe_order=dedupe_order)
    if down_lag != 1:
        schedule_options.update(down_lag=down_lag)
    if skip_empty_slots:
        schedule_options.update(skip_empty_slots=True)
    if weights_arena:
        schedule_options.update(weights_arena=True)
    if dedupe_early_first:
        schedule_options.update(dedupe_early_first=True)
    if scales_after_gate_up:
        schedule_options.update(scales_after_gate_up=True)
    if pair_head_in_ring:
        schedule_options.update(pair_head_in_ring=True)
    if act_slots != 2:
        schedule_options.update(act_slots=act_slots)
    if fold_dma:
        schedule_options.update(fold_dma=True)
    if skip_scale_duplicates:
        schedule_options.update(skip_scale_duplicates=True)
    if down_psum_dtype != "bf16":
        schedule_options.update(down_psum_dtype=down_psum_dtype)
    if rank_info != None:
        schedule_options.update(rank_info=rank_info)
    if stationary_slots != 3:
        schedule_options.update(stationary_slots=stationary_slots)
    if head_finish_first:
        schedule_options.update(head_finish_first=True)
    if dedupe_token_rank != "tiles":
        schedule_options.update(dedupe_token_rank=dedupe_token_rank)
    if dedupe_index_first:
        schedule_options.update(dedupe_index_first=True, late_setup=late_setup)
    if prefetch_groups != 0:
        schedule_options.update(prefetch_groups=prefetch_groups)
    if prefetch_load != "gather":
        schedule_options.update(prefetch_load=prefetch_load)
    if prefetch_indices != None:
        schedule_options.update(prefetch_indices=prefetch_indices)
    if fold_store_order != "columns":
        schedule_options.update(fold_store_order=fold_store_order)
    if fold_store_cols != 512:
        schedule_options.update(fold_store_cols=fold_store_cols)
    if dedupe_trips_after_stores:
        schedule_options.update(dedupe_trips_after_stores=True)
    if dedupe_precomputed != None:
        pre = tuple(dedupe_precomputed[0:2])
        if len(dedupe_precomputed) > 2 and dedupe_precomputed[2] > 0:
            pre = pre + (index_tk[:, 0:1], dedupe_precomputed[2])
        schedule_options.update(dedupe_precomputed=pre)
    body = group_body_phases if two_phase else group_body
    output = body(
        sbm,
        hidden_qtz,
        hidden_scale,
        expert_gate_up_packed,
        expert_down_weights,
        expert_down_weights_scale,
        index_tk,
        affinity_tk,
        scores,
        affinity_tk,  # router_top: not read with router_scores
        recip,
        after_norm=None,  # the routing is ready at the start: the gather indices need not wait
        dedupe_experts=dedupe_experts,
        sp_group_period=sp_group_period,
        gate_up_slots=gate_up_slots,
        moving_slots=moving_slots,
        down_slots=down_slots,
        dedupe_static_groups=dedupe_static_groups,
        dedupe_block_groups=dedupe_block_groups,
        affinity_dtype=affinity_dtype,
        router_scores=scores,
        dedupe_early_indices=dedupe_early_indices,
        group0_queues=group0_queues,
        dedupe_block_mode=dedupe_block_mode,
        expert_offset=expert_offset,
        n_experts=E,
        output=output,
        output_row=output_row,
        dedupe_prefetch=dedupe_prefetch,
        gate_up_pair_gathers=gate_up_pair_gathers,
        evict_split=evict_split,
        name_suffix=name_suffix,
        buffer_cache=cache,
        quantize_first=quantize_first,
        merged_down_gathers=merged_down_gathers,
        **body_options,
        **schedule_options,
        **({"two_phase": True} if two_phase else {}),
        **({"two_phase_latent_first": True} if two_phase and two_phase_state.get("latent_first") else {}),
    )

    def release_heap() -> None:
        """The heap tiles return to the region (a later tile at their addresses waits for their last reads)."""
        while len(sbm.heap) > heap_depth:
            sbm.pop_heap()

    if not two_phase:
        release_heap()
        return output
    phases = output
    next(phases)  # to the end of the routing phase (the yield before the first read of hidden_qtz)

    def finish(latent: nl.NkiTensor) -> nl.NkiTensor:
        """The latent quantization, then the rest of the body; returns the output."""
        kernel_assert(
            tuple(latent.shape) == (T, H) and latent.dtype == nl.bfloat16,
            f"The latent must be bfloat16 [{T}, {H}], got {latent.dtype} {latent.shape}",
        )
        kernel_assert(
            (latent.buffer == nl.sbuf) == two_phase_state["latent_in_sbuf"],
            f"The latent is in {latent.buffer}, but start had latent_in_sbuf={two_phase_state['latent_in_sbuf']}",
        )
        quantized = quantize(latent, latent_tiles)
        try:
            phases.send(quantized)  # resume after the routing phase
            result = run_phases(phases)
        except StopIteration as stop:
            result = stop.value
        release_heap()
        return result

    two_phase_state.update(finish=finish, tokens=T, hidden=H, routing_loads=routing_loads)
    return None


class ExpertBodyStart:
    """The state of a started call, from expert_body_start to expert_body_finish."""

    def __init__(self, **state):
        self.__dict__.update(state)


def expert_body_start(
    expert_index: nl.NkiTensor,
    expert_affinities: nl.NkiTensor,
    selected_scores: nl.NkiTensor,
    expert_gate_up_packed: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_down_weights_scale: nl.NkiTensor,
    latent_in_sbuf: bool = False,
    latent_first: bool = False,
    **options,
) -> ExpertBodyStart:
    """Start a dedupe call of expert_body_packed before its latent exists: all the work that needs only the routing.

    A latent MoE layer knows its routing before its latent (the latent comes from a projection and an all-gather).
    The start issues the routing loads, the selected-score sum, the setup, the compaction (the distinct experts, their
    positions and affinities, the trip counts of the blocks), the gather indices, the Pool loads of the head
    pipeline's gate/up ring (gate_up_slots rows), its first down units and scale tiles, and the re-layout of group 0.
    expert_body_finish(started, latent) quantizes the latent to MXFP8 and runs the group loop, the blocks and the
    stores. The output is bit-identical to expert_body_packed on the same inputs and options.

    Caller rules between start and finish:
        - SBUF: the call keeps its stack scope of the region (sbm) open and its heap tiles (the MXFP8 latent, the
          routing, the selected scores and the latent scratch) live. A stack frame or heap tile that the caller adds
          to the region after start must be released before finish.
        - PSUM: banks 6 (head ids) and 7 (the compaction) are in use by start. finish writes bank 0 (the latent
          transposes) first and then all 8 banks. The caller may use banks 1-5 for tiles that are dead at finish.
        - Engines: the caller's instructions run after the start's instructions in each engine's program order. A
          wait on an engine (for example a DMA that waits for the latent) holds only the later work of that engine.

    Args:
        expert_index, expert_affinities, selected_scores, expert_gate_up_packed, expert_down_weights,
            expert_down_weights_scale: as in expert_body_packed (HBM).
        latent_in_sbuf (bool): the latent of finish is a bfloat16 [T, H] SBUF tile with token t on partition t
            (False, default: HBM, loaded by finish on the ACT queue).
        latent_first (bool): finish orders the latent swizzle and quantization ahead of the DVE re-layout of head
            group 1 (group_body_phases two_phase_latent_first). Default False. The outputs are the same.
        **options: the options of expert_body_packed (sbm, sbuf_reserve, output, output_row, expert_offset,
            name_suffix, the dedupe structure and the schedule levers). Needs dedupe_experts="dynamic",
            dedupe_static_groups > 1 and latent_scratch="heap".

    Returns:
        ExpertBodyStart: the state for expert_body_finish.
    """
    kernel_assert(options.get("dedupe_experts") == "dynamic", "expert_body_start needs dedupe_experts='dynamic'")
    state = {"latent_in_sbuf": latent_in_sbuf, "latent_first": latent_first}
    expert_body_packed(
        None,
        expert_gate_up_packed,
        expert_down_weights,
        expert_down_weights_scale,
        expert_index,
        expert_affinities,
        selected_scores,
        two_phase_state=state,
        **options,
    )
    return ExpertBodyStart(**state)


def expert_body_finish(started: ExpertBodyStart, latent: nl.NkiTensor) -> nl.NkiTensor:
    """Finish a call of expert_body_start: quantize the bfloat16 [T, H] latent (HBM, or SBUF with latent_in_sbuf at
    start) into the MXFP8 stationary, then run the rest of the body. Returns the output (bfloat16 [T, H] @HBM, or the
    caller's output)."""
    return started.finish(latent)


@nki.jit
def moe_tkg_selective_load_packed(
    latent: nl.NkiTensor,
    expert_gate_up_packed: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_down_weights_scale: nl.NkiTensor,
    expert_index: nl.NkiTensor,
    expert_affinities: nl.NkiTensor,
    selected_scores: Optional[nl.NkiTensor] = None,
    sbuf_reserve: int = 0,
    expert_offset: int = 0,
    n_experts: Optional[int] = None,
    dedupe_experts: Optional[str] = None,
    dedupe_static_groups: int = 1,
    dedupe_block_groups: Optional[Sequence[int]] = None,
    dedupe_block_mode: str = "binary",
    dedupe_early_indices: bool = True,
    beta: float = 4.0,
    linear_beta: float = 25.0,
    sigmoid_from_tanh: bool = False,
    affinity_dtype: str = "bf16",
    group0_queues: str = "sp_act",
    sp_group_period: int = 0,
    gate_up_slots: int = 4,
    moving_slots: int = 3,
    down_slots: int = 4,
    dedupe_prefetch: bool = False,
    gate_up_pair_gathers: bool = False,
    evict_split: str = "rows",
    quantize_first: bool = False,
    merged_down_gathers: bool = True,
    latent_scratch: str = "ring",
    dedupe_late_trips: bool = False,
    dedupe_fold_first: bool = False,
    setup_memsets: str = "pool",
    dedupe_onehot_parts: bool = False,
    valid_tokens: Optional[int] = None,
    skip_down_padding: bool = False,
    dedupe_order: str = "id",
    down_lag: int = 1,
    skip_empty_slots: bool = False,
    weights_arena: bool = False,
    dedupe_early_first: bool = False,
    scales_after_gate_up: bool = False,
    pair_head_in_ring: bool = False,
    act_slots: int = 2,
    fold_dma: bool = False,
    skip_scale_duplicates: bool = False,
    down_psum_dtype: str = "bf16",
    rank_info: Optional[nl.NkiTensor] = None,
    stationary_slots: int = 3,
    head_finish_first: bool = False,
    dedupe_token_rank: str = "tiles",
    dedupe_index_first: bool = False,
    prefetch_groups: int = 0,
    prefetch_load: str = "gather",
    prefetch_indices: Optional[nl.NkiTensor] = None,
    fold_store_order: str = "columns",
    fold_store_cols: int = 512,
    dedupe_trips_after_stores: bool = False,
) -> nl.NkiTensor:
    """Standalone T = 32 routed-expert kernel (latent-MoE semantics): expert_body_packed with HBM inputs and a new
    [T, H] output. See expert_body_packed for the arguments (beta, linear_beta: the SiTU constants).

    Returns:
        output (nl.NkiTensor): bfloat16 [T, H] @HBM.
    """
    kernel_assert(
        nl.program_ndim() == 0 or (nl.program_ndim() == 1 and nl.num_programs(0) == 1),
        "moe_tkg_selective_load_packed (kimi_t32) supports LNC1 only",
    )
    return expert_body_packed(
        latent,
        expert_gate_up_packed,
        expert_down_weights,
        expert_down_weights_scale,
        expert_index,
        expert_affinities,
        selected_scores=selected_scores,
        sbuf_reserve=sbuf_reserve,
        expert_offset=expert_offset,
        n_experts=n_experts,
        dedupe_experts=dedupe_experts,
        dedupe_static_groups=dedupe_static_groups,
        dedupe_block_groups=dedupe_block_groups,
        dedupe_block_mode=dedupe_block_mode,
        dedupe_early_indices=dedupe_early_indices,
        situ=(beta, linear_beta),
        sigmoid_from_tanh=sigmoid_from_tanh,
        affinity_dtype=affinity_dtype,
        group0_queues=group0_queues,
        sp_group_period=sp_group_period,
        gate_up_slots=gate_up_slots,
        moving_slots=moving_slots,
        down_slots=down_slots,
        dedupe_prefetch=dedupe_prefetch,
        gate_up_pair_gathers=gate_up_pair_gathers,
        evict_split=evict_split,
        quantize_first=quantize_first,
        merged_down_gathers=merged_down_gathers,
        latent_scratch=latent_scratch,
        dedupe_late_trips=dedupe_late_trips,
        dedupe_fold_first=dedupe_fold_first,
        setup_memsets=setup_memsets,
        dedupe_onehot_parts=dedupe_onehot_parts,
        valid_tokens=valid_tokens,
        skip_down_padding=skip_down_padding,
        dedupe_order=dedupe_order,
        down_lag=down_lag,
        skip_empty_slots=skip_empty_slots,
        weights_arena=weights_arena,
        dedupe_early_first=dedupe_early_first,
        scales_after_gate_up=scales_after_gate_up,
        pair_head_in_ring=pair_head_in_ring,
        act_slots=act_slots,
        fold_dma=fold_dma,
        skip_scale_duplicates=skip_scale_duplicates,
        down_psum_dtype=down_psum_dtype,
        rank_info=rank_info,
        stationary_slots=stationary_slots,
        head_finish_first=head_finish_first,
        dedupe_token_rank=dedupe_token_rank,
        dedupe_index_first=dedupe_index_first,
        prefetch_groups=prefetch_groups,
        prefetch_load=prefetch_load,
        prefetch_indices=prefetch_indices,
        fold_store_order=fold_store_order,
        fold_store_cols=fold_store_cols,
        dedupe_trips_after_stores=dedupe_trips_after_stores,
    )
