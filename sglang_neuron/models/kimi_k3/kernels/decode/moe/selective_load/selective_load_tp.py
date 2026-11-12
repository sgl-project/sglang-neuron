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

"""Two-phase routed-expert call of the kimi packed selective-load body for one tensor-parallel rank (8 tokens).

A latent MoE layer knows its routing before its latent: the router reads the hidden state, and the latent comes from a
projection and an all-gather. ``selective_load_tp_start`` issues all the work that needs only the routing (the
routing broadcast, the expert weight loads and the Pool weight stream, the affinity rows, the first re-layouts), and
``selective_load_tp_finish`` quantizes the latent and runs the compute. A caller can put the instructions that make
the latent between the two calls. ``selective_load_tp`` is the standalone kernel of both.

Semantics: SiTU experts (beta*tanh(g/beta)*sigmoid(g) * linear_beta*tanh(u/linear_beta) in fp32) on a bf16 latent
quantized to MXFP8 without a norm, [T, K] affinities rounded to bf16 once and applied before the MXFP8 quantization of
each pair (pre_quant), as hi + lo terms. These are the options of the latent-MoE semantics of
``moe_tkg_selective_load_packed``, so the result is that of the standalone kernel with
activation="situ", hidden_input="bf16", affinity_layout="topk" and affinity_hi_lo=True.

Dimensions: T = 8 tokens, K = 16 experts per token (N = T*K = 128 routed pairs, 8 groups of 16), H latent columns
(H % 512 == 0, 2048 <= H <= 8192), I intermediate columns of the rank slice (I % 4 == 0, I <= 32), E expert rows
(E % 128 == 0; a stack of L layers gives L*E rows).

Algorithm:
    Step 1 (start): Check the inputs, take the SBUF region and the call's name suffix, and allocate the hand-off tiles
        on the heap of the region: hidden_qtz / hidden_scale (the MXFP8 stationary that finish writes) and the
        broadcast routing tiles index / affinity.
    Step 2 (start): Broadcast the routing to every partition (broadcast_routing): index[a, n] = the weight row of the
        expert of pair n (clamped, + expert_offset), affinity[a, n] = its bf16 affinity.
    Step 3 (start): Run the first phases of the body (selective_load_packed_phases with the broadcast routing as SBUF
        inputs): the buffers (with the ring depths that leave sbuf_reserve bytes of the region free), the HWDGE expert
        loads, the gather indices and the Pool weight stream (after_index is called after the first gathers), the
        affinity rows, and with relayout_before_latent the re-layout of groups 0 and 1.
    Step 4 (finish): Quantize the latent into hidden_qtz / hidden_scale (quantize_latent_mx: 4 PE transposes per H512
        block into PSUM bank 1, one DVE swizzle, quantize_mx), then run the rest of the body (Steps 6-7: gate/up, SiTU,
        down projection, the output stores).

Layouts:
    - hidden_qtz float8_e4m3fn_x4 / hidden_scale uint8 [128, (H/512) * T] (heap): element [p, hb*T + t] packs the 4
      values of latent column 512*hb + 4*p + q (q = 0..3) of token t; the scales in the quadrant layout of quantize_mx.
      Each MX group holds 32 consecutive latent values of one token, so the gate/up weights use the hidden order
      h = 512*hb + 4*p + q (row p, H512 block hb, lane q of the packed row).
    - index int32 / affinity bf16 [128, N] (heap): pair n = t*K + k on every partition.
    - PSUM: the routing broadcast uses banks 2-5 before the body (bank 6 with overlap_calls); the latent transposes use
      bank 1 in finish; the body uses all 8 banks at fixed addresses (0-1 gate/up, 2-5 the output, 6 free unless
      last_gate_up_split or last_group_halves, 7 the transposed activation).
"""

from typing import Callable, Optional, Tuple

import nki
import nki.isa as nisa
import nki.language as nl

from ...utils.allocator import SbufManager
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import div_ceil
from ...utils.logging import get_logger
from .selective_load_packed import (
    _WEIGHT_SBUF_BUDGET,
    _named_modules,
    finish_phases,
    quantize_latent_mx,
    selective_load_packed_phases,
)

_PMAX = 128
_X4 = 4
_H512 = 512
_TOP_K = 16
TOKENS = 8  # tokens of one call (128 routed pairs at top-16)
SHORT_TOKENS = 4  # chunk_tokens=4: a call of 4 tokens (64 routed pairs, 4 groups: the short schedule of the body)
_MX_ROWS = 8  # down rows of an expert (I padded to 32 = 8 rows x 4 lanes)
_PSUM_BANK_BYTES = 2048
_BANK_FP32 = _PSUM_BANK_BYTES // 4  # FP32 columns of one PSUM bank: the routing broadcast moves this many pairs
_ROUTING_PSUM_BANK = 2  # banks 2-5: the routing broadcast (the body uses them only after start)
_FIRST_GATHER_PSUM_BANK = 4  # fast_first_gather: token 0's expert ids on every partition (bank 4 is free in start)
_FREE_PSUM_BANK = 6  # overlap_calls: the routing broadcast (the body does not use bank 6)
_LATENT_PSUM_BANK = 1  # finish: the latent transposes (the body's gate/up tile of odd groups comes later)
_BODY_ALIGN = 512  # alignment of the body's stack scope (its first buffers are DMA destinations)
# Standalone (sbuf=None): SBUF bytes per partition at the top that the region leaves to the compiler (the identity of
# the PE transposes and the tiles of a caller without a region).
COMPILER_SBUF_RESERVE = 16384
# The heap bytes of finish beyond the quantization scratch (heap alignment).
_FINISH_HEAP_SLACK = 64
_CALLS = [0]  # calls of selective_load_tp_start in the kernel being traced
# The Pool stream instructions of the earlier calls of the kernel: each call's Pool setup and stream follow them.
_STREAM_CHAIN = []
# overlap_calls: the tail instructions (DVE, ACT) of the earlier calls; each call's group-0 re-layout follows them.
_TAIL_CHAIN = []


def restart_calls():
    """Restart the call count of selective_load_tp_start (the default name suffixes) and the Pool stream chain. Call it
    at the start of a kernel so that its op names do not depend on the kernels traced before it."""
    _CALLS[0] = 0
    del _STREAM_CHAIN[:]
    del _TAIL_CHAIN[:]


class SelectiveLoadStart:
    """The state of a started call, from selective_load_tp_start to selective_load_tp_finish."""

    def __init__(self, **state):
        self.__dict__.update(state)


@nki.jit
def selective_load_tp(
    latent: nl.NkiTensor,
    expert_index: nl.NkiTensor,
    expert_affinities: nl.NkiTensor,
    expert_gate_up_packed: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_down_weights_scale: nl.NkiTensor,
    beta: float = 4.0,
    linear_beta: float = 25.0,
    affinity_hi_lo: bool = True,
    sigmoid_from_tanh: bool = False,
    hold_sources: bool = False,
    chunk_overlap: Optional[bool] = None,
    affinity_scaling: str = "pre_quant",
    chunk_tokens: int = TOKENS,
    down_psum_dtype: str = "bf16",
    rank_info: Optional[nl.NkiTensor] = None,
    sbuf_reserve: int = 0,
    group0_gather: bool = True,
    schedule: str = "standalone",
    relayout_before_latent: bool = True,
    merged_gathers: bool = True,
    weights_arena: bool = True,
    skip_down_padding: bool = False,
    last_gate_up_first: bool = False,
    last_relayout_runs: int = 1,
    last_swiglu_first: bool = False,
    last_gate_up_split: bool = False,
    merged_down_quads: Tuple[int, ...] = (),
    fast_first_gather: bool = False,
    store_parts: int = 1,
    relayout0_act: bool = False,
    last_group_halves: bool = False,
    act_relayout_blocks: Optional[int] = None,
    last_finish_early: bool = False,
) -> nl.NkiTensor:
    """One rank's routed-expert partial of 8 tokens: selective_load_tp_start, then selective_load_tp_finish.

    The arguments up to rank_info are those of a selective-load kernel call of 8 tokens (in that order); see
    selective_load_tp_start for their values.

    Args:
        latent (nl.NkiTensor): bfloat16 [8, H] @HBM, or @SBUF with token t on partition t.
        expert_index (nl.NkiTensor): int32 (or uint32) [8, 16] @HBM or @SBUF, the routed experts of each token.
        expert_affinities (nl.NkiTensor): float32 [8, 16], the affinity of each routed expert (pair n = t*K + k), in
            the buffer of expert_index.
        expert_gate_up_packed (nl.NkiTensor): uint8 [E, 128, 4*I*(H/512 + S)] @HBM, the packed gate/up rows with
            their MX scales (layout of moe_tkg_selective_load_packed), in the latent hidden order.
        expert_down_weights (nl.NkiTensor): float4_e2m1fn_x4 (or a uint16 carrier) [E, 8, H] @HBM. Row p, lane q holds
            intermediate column i = q*(I/4) + p; rows p >= I/4 are not read.
        expert_down_weights_scale (nl.NkiTensor): uint8 [E, H] @HBM, one MX scale per expert and latent column.
        beta, linear_beta, affinity_hi_lo, sigmoid_from_tanh, hold_sources, chunk_overlap, affinity_scaling,
            chunk_tokens, down_psum_dtype, rank_info, sbuf_reserve, group0_gather, schedule, relayout_before_latent,
            merged_gathers, weights_arena, skip_down_padding, last_gate_up_first, last_relayout_runs,
            last_swiglu_first, last_gate_up_split, merged_down_quads, fast_first_gather, store_parts, relayout0_act,
            last_group_halves, act_relayout_blocks, last_finish_early: as in selective_load_tp_start.

    Returns:
        output (nl.NkiTensor): bfloat16 [8, H] @HBM.
    """
    kernel_assert(
        nl.program_ndim() == 0 or (nl.program_ndim() == 1 and nl.num_programs(0) == 1),
        "selective_load_tp supports LNC1 only",
    )
    kernel_assert(nisa.get_nc_version() >= nisa.nc_version.gen4, "selective_load_tp requires Trn3 or newer")
    restart_calls()
    started = selective_load_tp_start(
        expert_index,
        expert_affinities,
        expert_gate_up_packed,
        expert_down_weights,
        expert_down_weights_scale,
        beta,
        linear_beta,
        affinity_hi_lo,
        sigmoid_from_tanh,
        hold_sources,
        chunk_overlap=chunk_overlap,
        affinity_scaling=affinity_scaling,
        chunk_tokens=chunk_tokens,
        latent_in_sbuf=latent.buffer == nl.sbuf,
        sbuf_reserve=sbuf_reserve,
        down_psum_dtype=down_psum_dtype,
        rank_info=rank_info,
        group0_gather=group0_gather,
        schedule=schedule,
        relayout_before_latent=relayout_before_latent,
        merged_gathers=merged_gathers,
        weights_arena=weights_arena,
        skip_down_padding=skip_down_padding,
        last_gate_up_first=last_gate_up_first,
        last_relayout_runs=last_relayout_runs,
        last_swiglu_first=last_swiglu_first,
        last_gate_up_split=last_gate_up_split,
        merged_down_quads=merged_down_quads,
        fast_first_gather=fast_first_gather,
        store_parts=store_parts,
        relayout0_act=relayout0_act,
        last_group_halves=last_group_halves,
        act_relayout_blocks=act_relayout_blocks,
        last_finish_early=last_finish_early,
    )
    return selective_load_tp_finish(started, latent)


def selective_load_tp_start(
    expert_index: nl.NkiTensor,
    expert_affinities: nl.NkiTensor,
    expert_gate_up_packed: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_down_weights_scale: nl.NkiTensor,
    beta: float = 4.0,
    linear_beta: float = 25.0,
    affinity_hi_lo: bool = True,
    sigmoid_from_tanh: bool = False,
    hold_sources: bool = False,
    sbuf=None,
    output: Optional[nl.NkiTensor] = None,
    output_row: int = 0,
    chunk_overlap: Optional[bool] = None,
    affinity_scaling: str = "pre_quant",
    expert_offset: int = 0,
    after_index: Optional[Callable] = None,
    chunk_tokens: int = TOKENS,
    latent_in_sbuf: bool = True,
    sbuf_reserve: int = 0,
    persistent_buffers: bool = False,
    sparse_schedule: bool = False,
    group0_gather: bool = True,
    setup_after_stream: bool = True,
    down_psum_dtype: str = "bf16",
    rank_info: Optional[nl.NkiTensor] = None,
    down_group_mask: Optional[nl.NkiTensor] = None,
    long_stream: bool = True,
    name_suffix: Optional[str] = None,
    schedule: str = "standalone",
    relayout_before_latent: bool = True,
    after_index_point: str = "gathers",
    hwdge_act_loads: Optional[int] = None,
    merged_gathers: bool = True,
    weights_arena: bool = True,
    overlap_calls: bool = False,
    reserve_hwdge_ids: bool = False,
    skip_down_padding: bool = False,
    last_gate_up_first: bool = False,
    last_relayout_runs: int = 1,
    last_swiglu_first: bool = False,
    last_gate_up_split: bool = False,
    merged_down_quads: Tuple[int, ...] = (),
    fast_first_gather: bool = False,
    store_parts: int = 1,
    relayout0_act: bool = False,
    last_group_halves: bool = False,
    act_relayout_blocks: Optional[int] = None,
    last_finish_early: bool = False,
    pe_keep_alive: int = 0,
    paired_downs: bool = False,
    long_stream_short: bool = False,
    private_out: bool = False,
) -> SelectiveLoadStart:
    """Start a call before the latent is ready: the routing broadcast, the expert weight loads and the Pool stream.

    Between start and finish the call keeps its stack scope of the region open: a stack frame that the caller opens
    after start must close before finish. The caller may use PSUM between start and finish only for tiles that are
    dead at finish, and not banks 2-5 (bank 6 with overlap_calls) until the routing broadcast is read (its DVE
    copies).

    Args:
        expert_index (nl.NkiTensor): int32 (or uint32) [8, 16], the routed experts of each token: @HBM, or @SBUF on
            the token partitions.
        expert_affinities (nl.NkiTensor): float32 [8, 16], the affinity of each routed expert (pair n = t*K + k). Same
            buffer as expert_index.
        expert_gate_up_packed (nl.NkiTensor): uint8 [E, 128, row bytes] @HBM (see selective_load_tp), or a stack of
            layers [L, E, 128, row bytes] (taken as [L*E, ...] rows, see expert_offset).
        expert_down_weights (nl.NkiTensor): float4_e2m1fn_x4 (or uint16) [E, 8, H] or [L, E, 8, H] @HBM (padded rows).
        expert_down_weights_scale (nl.NkiTensor): uint8 [E, H] or [L, E, H] @HBM (one scale set), or [E, 2, H] or
            [L, E, 2, H] (two scale groups: the scales of the two checkpoint blocks of the rank slice, so every down
            value is the checkpoint value; needs down_group_mask or rank_info).
        beta (float): SiTU gate soft-clipping factor.
        linear_beta (float): SiTU up soft-clipping factor.
        affinity_hi_lo (bool): the affinity-scaled activation as two MXFP8 terms, hi + lo (True, default), or one.
        sigmoid_from_tanh (bool): compute sigmoid(g) as (1 + tanh(g/2)) / 2 (the activation engine uses only the tanh
            table).
        hold_sources (bool): no effect (the compiler orders every later writer of a store source after the store).
        sbuf: the SBUF region of the call: an SbufManager with explicit addresses, or an object with an ``sbm``
            attribute. The call takes its buffers from a new stack scope and from the heap of the region and returns
            all of them at finish. None: a manager of SBUF without the top COMPILER_SBUF_RESERVE bytes (standalone).
        output (Optional[nl.NkiTensor]): bfloat16 HBM destination with H columns. None: a new [8, H] tensor.
        output_row (int): the row of output for token 0.
        chunk_overlap (Optional[bool]): no effect (one 8-token call).
        affinity_scaling (str): "pre_quant" only (the affinity scales the activation before its MXFP8 quantization).
        expert_offset (int): the weight row of expert 0: expert e reads row expert_offset + e (layer l of a stack of
            E-row layers: l*E). The offset goes into the dynamic index of every weight DMA.
        after_index (Optional[Callable]): ``after_index(after)``, called once after the call issues its first Pool
            weight gathers; ``after`` is the last instruction that builds the gather indices. The caller can put work
            there that must not delay the weight stream (for example the copy of the latent). None: no call.
        chunk_tokens (int): the tokens of the call. 8 (default): 8 groups, the long-stream schedule of the body. 4: 4
            groups (64 pairs) in the short schedule of the body (no long stream: no late_down, one down-scale tile, so
            no merged_scales, and the last group runs as the others); the routing is [4, 16] and the latent [4, H].
            The output of each token does not depend on the call size (each group holds one token).
        latent_in_sbuf (bool): the latent of finish is in SBUF (True, default) or in HBM (finish loads it on the SP
            queue into heap scratch of H*2 bytes per partition, which start leaves free).
        sbuf_reserve (int): SBUF bytes per partition of the region that the call leaves free while it runs, for the
            tiles that the compiler places (for example the buffers of a collective between start and finish). The
            body takes the weight-ring depths that fit (Step 1.2 of the body: fewer moving and row slots); the result
            does not change.
        persistent_buffers (bool): the calls on one region share their SBUF buffers (the body buffers, the hand-off
            and routing tiles) when the name, shape, dtype and address match, so the DMAs of every call have the
            destinations of the first call. The result does not change. The cache lives on the region (an attribute
            of the SbufManager). With output None the calls also share one HBM output, so read it before the next
            call.
        sparse_schedule (bool): no effect.
        group0_gather (bool): True (default): every gate/up group comes from a Pool gather (8 gate/up gathers, no
            dynamic HWDGE expert load; group0_on_pool of the body), as in the model call. False: the gate/up rows of
            group 0 come from 16 per-expert HWDGE loads (8 on SP, 8 on ACT) and those of group 3 from 16 more on SP,
            so Pool runs 2 gate/up gathers fewer (faster for one call with the latent at the start).
            Each call writes its weights into a new DGE table slot of the NEFF, and a dynamic HWDGE DMA whose
            destination slot index is 129 or more issues much more slowly (Section 6.10); the Pool gathers do not.
            Use False only where the slots of the calls stay below 129. The result does not change.
        setup_after_stream (bool): no effect (one call per start).
        down_psum_dtype (str): "bf16" (default): bf16 PSUM rounds the hi and the lo term, then the store rounds their
            sum. "fp32": the down projection adds in fp32 PSUM (2 output halves of 2048 and 1536 columns in banks 2-5),
            out_sb stays fp32 and the store DMA adds the fp32 hi and lo terms, so each output value rounds once (the
            rounding of the selective-load call of the model with "fp32").
        rank_info (Optional[nl.NkiTensor]): int32 [1, 4] @HBM, (global rank, DP rank, TP rank, 0). Read only with two
            down scale groups and no down_group_mask: start then builds the mask (build_down_group_mask) on the heap.
        down_group_mask (Optional[nl.NkiTensor]): with two down scale groups, a bfloat16 [128, 2] SBUF tile from
            build_down_group_mask that lives until finish (a caller with many calls builds it once). Column s is 1 on
            partition a if down row a % 8 uses scale group s. The down projection then runs once per scale group:
            the stationary of group s holds only its own rows (the row weights of the other rows are 0) with its own
            MXFP8 scales, and both matmuls add into the same PSUM rows. Costs: twice the down matmuls and the down
            stationary, and twice the down-scale bytes (the arena grows by 7 KB per partition at H = 3584, so the
            ring depths can be smaller). None: one scale set, or built from rank_info.
        long_stream (bool): no effect: 8 groups always run the long-stream schedule of the body.
        name_suffix (Optional[str]): appended to every op and tensor name of the call (names must be unique in a
            kernel). None: "_k<c>" for the c-th call of the kernel (restart_calls resets the count).
        schedule (str): the long-stream schedule of the body. "standalone" (default): the standalone set
            (early_gather, merged_scales, late_down=5, last_relayout="dve"; the HWDGE group is group 3). "fused": the
            schedule of the fused block (the HWDGE group is group 2, 2 of its loads on ACT, finish(g) after 2 gate/up
            units). The result does not change.
        relayout_before_latent (bool): True (default): start also issues the re-layout of groups 0 and 1, so the
            weight copies do not wait for the latent in program order. False: finish issues them after the latent
            quantization. The result does not change.
        after_index_point (str): where in program order start calls after_index. "gathers" (default): right after
            the first Pool gathers (the point of the served call). "relayout" (with relayout_before_latent): at the
            end of start, after the re-layout copies of groups 0 and 1, so a DVE copy of the latent that the callback
            issues does not hold those copies while it waits for the latent. ``after`` is the same instruction. The
            result does not change.
        hwdge_act_loads (Optional[int]): with group0_gather=False, the loads of group 3 that go on the ACT queue (after
            the group-0 re-layout). None (default): 0 (all 16 on SP; a call issues 24 HWDGE loads on SP and 8 on ACT).
            8: 16 / 16. The result does not change.
        merged_gathers (bool): True (default): one Pool gather loads the down weights of two groups
            (merged_down_gathers) and one loads both down-scale tiles (merged_scales): 13 gathers (11 with
            group0_gather=False), faster for one call than False (one down gather per group and one per
            scale tile: 18 and 16 gathers). The result does not change.
        weights_arena (bool): True (default): one SBUF tensor holds the gate/up row ring, the down weights and the down
            scales (weights_arena of the body), so the Pool gathers and HWDGE loads of a call write one tensor, and a
            call takes one DGE table slot of the NEFF, as the model call. False: separate tensors; with
            merged_gathers each merged gather then takes one more slot in every call, and in a kernel with many calls
            the dynamic HWDGE DMAs whose slots pass about 128 issue much more slowly (Section 6.10). The result does
            not change.
        overlap_calls (bool): False (default). True: consecutive calls of a kernel overlap more. The routing broadcast
            uses PSUM bank 6 (free in the body) instead of banks 2-3, so it does not wait for the output banks of
            the call before, and the group-0 re-layout copies follow the last MXFP8 quantization (DVE) and SwiGLU
            (ACT) of the call before (tail_chain of the body), so they do not hold that call's tail while they wait
            for this call's first gather. The next call's routing, gathers and re-layouts then run during this call's
            tail: less time per call in a chain of calls; one call alone does not change. Nothing stays live
            across calls, and a call still takes one DGE table slot. With fast_first_gather, token 0's ids go to byte
            512 of bank 6 (not bank 4, an output bank). last_gate_up_split and last_group_halves also use bank 6, so
            the next call's broadcast waits for this call's last activation. The result does not change.
        reserve_hwdge_ids (bool): False (default). True: a group0_gather=True call also takes the heap bytes of the
            hwdge_ids tile of a group0_gather=False call (64 B per partition, unused), so both kinds of call have one
            heap layout. Use it in a kernel that switches from group0_gather=False calls to True calls: else the routing
            scratch of the first True call reuses the hwdge_ids bytes of the call before, whose last HWDGE loads read
            them late, and the compiler can put that scratch's first memset ahead of the call before on DVE (a delay
            once at the switch). No effect with group0_gather=False. The result does not change.
        skip_down_padding (bool): False (default): the down-weight gathers load the padding rows 6-7 of each
            expert's 8-row block, and the merged down-scale gather loads a real scale row on the partitions that the
            matmuls do not read. True: these rows get out-of-bounds gather indices, so the gathers skip them (down
            weights 25% fewer bytes, down scales half; down_real_rows=False of the body). Their SBUF bytes keep old
            values: the matmuls meet them only with zero stationary rows (FP4 values are always finite) or not at
            all, so the result does not change. The gathers and their destinations do not change (DGE slots).
        last_gate_up_first (bool): False (default). True: the down weights of groups 4-5 load after the gate/up rows
            of group 7, and down(4) and down(5) run after gate/up(7) (default: inside gate/up(6)), so the last group's
            rows land earlier and its activation runs under these down projections. The result does not change.
        last_relayout_runs (int): 1 (default) or more: DVE re-lays group 7 in this many runs of H512 blocks, so
            gate/up(7) starts after the first run. The result does not change.
        last_swiglu_first (bool): False (default). True: the activation of group 6 runs before the re-layout of
            group 7 on ACT and DVE, and finish(6) before gate/up(7) on the PE. The result does not change.
        last_gate_up_split (bool): False (default). True: gate/up(7) runs all gate matmuls, then all up matmuls into
            PSUM bank 6, so ACT starts the gate terms of the activation during the up matmuls. The caller must leave
            PSUM bank 6 free until finish. The result does not change.
        merged_down_quads (Tuple[int, ...]): () (default): the down weights load in gathers of two groups. (0,) or
            (0, 4): one gather of 512 rows loads the down weights of groups q .. q + 3 (one fixed Pool gather cost
            less per quad, but down(q) waits for the four groups' rows). The gathers write the same tensor (DGE
            slots). The result does not change.
        fast_first_gather (bool): False (default). True (SBUF routing tiles): the gather indices of group 0 come
            straight from token 0's row of the routing tile (one DVE cast, one ones matmul of 16 columns, two DVE
            ops), ahead of the full routing broadcast on DVE, and DVE (not Pool) picks the experts of the other
            gather-index columns (dve_experts of the body). Pool then starts with the group-0 gather. The gather
            indices and the result do not change. PSUM bank 4 (bank 6 at byte 512 with overlap_calls) holds token 0's
            ids during start.
        store_parts (int): 1 (default) or more: each 512-column output chunk is stored by this many DMAs of
            consecutive token rows (each one adds the hi and lo rows), so the last chunk stores can use more DMA
            engines. The result does not change.
        relayout0_act (bool): False (default): DVE copies the whole group-0 re-layout. True: ACT copies its first 4
            H512 blocks of the gate and of the up weights (as for groups 1-6), so DVE copies less during the
            descriptor window of the group-2 gather (DVE and Pool share one SBUF port). The result does not change.
        last_group_halves (bool): False (default). True: group 7 runs in two halves of 8 pairs: two Pool gathers of 8
            experts (the first one before the down gather of groups 4-5), and a re-layout, gate/up matmuls (the second
            half's sums in PSUM bank 6) and activation per half, so less work follows the last gate/up rows. Not with
            last_gate_up_split. The caller must leave PSUM bank 6 free until finish. The result does not change.
        act_relayout_blocks (Optional[int]): None (default): 4. The H512 blocks of the gate and of the up weights that
            ACT copies in the re-layout of groups 1-6 (7: all, DVE copies none), so DVE accesses SBUF less while Pool
            writes gather descriptors (one shared SBUF port). The result does not change.
        last_finish_early (bool): with last_gate_up_first: the PE runs gate/up(7) whole, then finish(6), down(4),
            finish(7), down(5), down(6), down(7), so the MXFP8 chain of group 7 runs under down(5) and down(6) and
            down(7) does not wait for it. The result does not change.
        pe_keep_alive (int): 0 (default) or the count of wide bf16 keep-alive matmuls that the PE runs during the
            re-layout of group 0, right before gate/up(0) (pe_keep_alive of the body; PSUM bank 7, free until the
            first activation transposes). After the idle time from the call before, the PE runs throttled for a short
            time; the keep-alive takes that time instead of gate/up(0). The result does not change.
        paired_downs (bool): False (default). True: the PE runs down(g - 1) and down(g) together inside gate/up(g + 1)
            for odd g (the two groups of one merged down gather), so down(1) and down(3) run one gate/up earlier,
            where the PE waits for gate/up rows (paired_downs of the body; for two down scale groups, whose down
            projections take twice the PE time). Not with last_gate_up_first. The result does not change.
        long_stream_short (bool): False (default). True (chunk_tokens=4): the 4 groups run the long-stream schedule
            of the body with late_down=1 and last_relayout="dve" (the standalone set of 8 groups, without
            merged_scales: one down-scale tile), so the last-group options (last_gate_up_split, last_relayout_runs,
            last_swiglu_first) apply. The result does not change.
        private_out (bool): False (default). True: the new HBM output of output None is private HBM instead of
            shared HBM, for a caller kernel that keeps every per-core scratch tensor private (as one compiled with
            ``--lnc1-with-shared-mem``, where a ``shared_hbm`` scratch tensor is one buffer for the two cores of an HBM
            stack). Only the buffer changes, so the result is the same. No effect with an output.

    Returns:
        SelectiveLoadStart: the state for selective_load_tp_finish.
    """
    # Step 1: Check the inputs, take the region and the name suffix, allocate the hand-off tiles on the heap.
    if name_suffix == None:
        name_suffix = f"_k{_CALLS[0]}"
        _CALLS[0] = _CALLS[0] + 1
    _, nl = _named_modules(name_suffix)
    expert_gate_up_packed, expert_down_weights, expert_down_weights_scale = _expert_rows(
        expert_gate_up_packed, expert_down_weights, expert_down_weights_scale
    )
    if expert_index.dtype == nl.uint32:
        expert_index = expert_index.view(nl.int32)
    if expert_down_weights.dtype == nl.uint16:
        expert_down_weights = expert_down_weights.view(nl.float4_e2m1fn_x4)
    T = expert_index.shape[0]
    n_rows, _, row_bytes = expert_gate_up_packed.shape
    H = expert_down_weights.shape[2]
    n_h512 = H // _H512
    _validate(expert_index, expert_affinities, expert_gate_up_packed, expert_down_weights, expert_down_weights_scale)
    kernel_assert(
        chunk_tokens in (SHORT_TOKENS, TOKENS) and T == chunk_tokens,
        f"selective_load_tp_start runs calls of 4 or 8 tokens (chunk_tokens = the routed tokens), got "
        f"chunk_tokens={chunk_tokens} and {T} tokens",
    )
    short = chunk_tokens == SHORT_TOKENS  # 4 groups: the short schedule of the body (long_stream_short: the long one)
    kernel_assert(not long_stream_short or short, "long_stream_short needs chunk_tokens=4")
    short_set = short and not long_stream_short  # 4 groups in the short schedule: no late_down, no last_relayout
    kernel_assert(0 <= expert_offset < n_rows, f"expert_offset must be in [0, {n_rows}), got {expert_offset}")
    kernel_assert(n_rows * _PMAX <= 1 << 24, f"At most {(1 << 24) // _PMAX} expert rows, got {n_rows}")
    kernel_assert(affinity_scaling == "pre_quant", f"affinity_scaling must be 'pre_quant', got {affinity_scaling}")
    kernel_assert(
        down_psum_dtype in ("bf16", "fp32"), f"down_psum_dtype must be 'bf16' or 'fp32', got {down_psum_dtype}"
    )
    # Two down scale groups ([E, 2, H]): each down row on the scale of its own checkpoint block (down_group_mask).
    n_scale_sets = len(expert_down_weights_scale.shape) - 1
    kernel_assert(
        n_scale_sets == 2 or down_group_mask is None, "down_group_mask needs down scales [E, 2, H] (two scale groups)"
    )
    kernel_assert(
        n_scale_sets == 1 or down_group_mask is not None or rank_info is not None,
        "Down scales [E, 2, H] need rank_info or down_group_mask",
    )
    kernel_assert(
        down_group_mask is None
        or (tuple(down_group_mask.shape) == (_PMAX, 2) and down_group_mask.dtype == nl.bfloat16),
        f"down_group_mask must be bfloat16 [{_PMAX}, 2], got {down_group_mask}",
    )
    kernel_assert(schedule in ("standalone", "fused"), f"schedule must be 'standalone' or 'fused', got {schedule}")
    kernel_assert(
        after_index_point == "gathers" or (after_index_point == "relayout" and relayout_before_latent),
        f"after_index_point must be 'gathers' or 'relayout' (with relayout_before_latent), got {after_index_point}",
    )
    if sbuf is None:
        sbuf = SbufManager(
            0, nl.tile_size.total_available_sbuf_size - COMPILER_SBUF_RESERVE, get_logger("selective_load_tp")
        )
    sbm = getattr(sbuf, "sbm", sbuf)
    cache = None
    if persistent_buffers:
        if not hasattr(sbm, "selective_load_tp_buffers"):
            sbm.selective_load_tp_buffers = {}
        cache = sbm.selective_load_tp_buffers
    if output is None and cache is not None and ("slp_out", T, H) in cache:
        output = cache[("slp_out", T, H)]  # persistent_buffers: the HBM output of the first call
    elif output is None:
        buffer = nl.private_hbm if private_out else nl.shared_hbm
        output = nl.ndarray((T, H), dtype=nl.bfloat16, buffer=buffer, name="slp_out")
        if cache is not None:
            cache[("slp_out", T, H)] = output
    kernel_assert(
        output.shape[1] == H and 0 <= output_row and output_row + T <= output.shape[0],
        f"output needs {H} columns and rows {output_row} .. {output_row + T - 1}, got {output.shape}",
    )

    def alloc_heap(shape, dtype, name):
        tile = sbm.alloc_heap(shape, dtype, name=name + name_suffix)
        return tile if cache is None else cache.setdefault((name, tuple(shape), str(dtype), sbm.heap_curr_addr), tile)

    heap_depth = len(sbm.heap)
    if n_scale_sets == 2 and down_group_mask is None:
        # Two scale groups without a caller mask: build it from rank_info (heap, until finish).
        down_group_mask = alloc_heap((_PMAX, 2), nl.bfloat16, "slp_down_group_mask")
        build_down_group_mask(
            rank_info,
            _rank_slice(row_bytes, n_h512),
            down_group_mask,
            "slp_down_groups" + name_suffix,
            lambda shape, dtype, tile_name: sbm.alloc_heap(shape, dtype, name=tile_name),
        )
    hidden_qtz = alloc_heap((_PMAX, n_h512 * T), nl.float8_e4m3fn_x4, "slp_hidden_qtz")
    hidden_scale = alloc_heap((_PMAX, n_h512 * T), nl.uint8, "slp_hidden_scale")
    index = alloc_heap((_PMAX, T * _TOP_K), nl.int32, "slp_index")
    affinity = alloc_heap((_PMAX, T * _TOP_K), nl.bfloat16, "slp_affinity")
    # The weight rows of the HWDGE expert loads before the broadcast: [T, K] for SBUF routing, a row [1, N] for HBM.
    hwdge_ids = None
    if not group0_gather or reserve_hwdge_ids:
        in_sbuf = expert_index.buffer == nl.sbuf
        hwdge_ids = alloc_heap((T, _TOP_K) if in_sbuf else (1, T * _TOP_K), nl.int32, "slp_hwdge_ids")
        if group0_gather:
            hwdge_ids = None  # reserve_hwdge_ids: only the heap bytes

    # Step 2: Broadcast the routing (its scratch returns to the region; a later tile at its addresses waits for its
    # last reads). fast_first_gather: the group-0 gather indices first (their tile stays until finish).
    first_gather_index, first_gather_op = None, None
    if fast_first_gather:
        kernel_assert(
            expert_index.buffer == nl.sbuf and group0_gather,
            "fast_first_gather needs the routing tiles in SBUF and group0_gather=True",
        )
        first_gather_index = alloc_heap((_PMAX, _TOP_K), nl.int32, "slp_fg_index")
    routing_depth = len(sbm.heap)
    if fast_first_gather:
        first_gather_op = first_gather_indices(
            sbm,
            expert_index,
            first_gather_index,
            name_suffix,
            expert_offset,
            n_rows - 1 - expert_offset,
            after_stream=_STREAM_CHAIN[-1] if _STREAM_CHAIN else None,
            psum_address=(_FREE_PSUM_BANK * _PSUM_BANK_BYTES + _PSUM_BANK_BYTES // 4)
            if overlap_calls
            else _FIRST_GATHER_PSUM_BANK * _PSUM_BANK_BYTES,
        )
    broadcast_routing(
        sbm,
        expert_index,
        expert_affinities,
        index,
        affinity,
        name_suffix,
        expert_offset,
        n_rows - 1 - expert_offset,
        hwdge_ids=hwdge_ids,
        after_stream=_STREAM_CHAIN[-1] if _STREAM_CHAIN else None,
        after_dve=first_gather_op,
        psum_bank=_FREE_PSUM_BANK if overlap_calls else _ROUTING_PSUM_BANK,
    )
    while len(sbm.heap) > routing_depth:
        sbm.pop_heap()

    # Step 3: The body's first phases, with the ring depths that leave sbuf_reserve and the finish scratch free.
    sbm.open_scope(name="slp_call" + name_suffix)
    sbm.align_stack_curr_addr(_BODY_ALIGN)
    finish_scratch = T * n_h512 * _X4 * 2 + (0 if latent_in_sbuf else H * 2) + _FINISH_HEAP_SLACK
    standalone = schedule == "standalone"
    deferred = []  # after_index_point="relayout": the instruction for after_index, called at the end of start
    finish_affinities = after_index
    if after_index is not None and after_index_point == "relayout":
        finish_affinities = deferred.append
    phases = selective_load_packed_phases(
        sbm,
        hidden_qtz,
        hidden_scale,
        expert_gate_up_packed,
        expert_down_weights,
        expert_down_weights_scale,
        index,
        affinity,
        inputs_in_sbuf=True,
        affinity_hi_lo=affinity_hi_lo,
        hi_mask_tile=affinity_hi_lo,
        finish_affinities=finish_affinities,
        situ=(beta, linear_beta),
        sigmoid_from_tanh=sigmoid_from_tanh,
        output=output,
        output_row=output_row,
        merged_down_gathers=merged_gathers,
        down_real_rows=not skip_down_padding,
        early_gather=standalone,
        # 4-token calls (short schedule): one down-scale tile and no long stream
        merged_scales=standalone and merged_gathers and not short,
        late_down=(1 if short else 5) if standalone and not short_set else 0,
        last_relayout="dve" if standalone and not short_set else None,
        weight_sbuf_budget=_WEIGHT_SBUF_BUDGET,
        sbuf_reserve=sbuf_reserve + finish_scratch,
        name_suffix=name_suffix,
        schedule=schedule,
        group0_on_pool=group0_gather,
        hwdge_index=hwdge_ids,
        stream_chain=_STREAM_CHAIN,
        buffer_cache=cache,
        **({} if hwdge_act_loads is None else {"hwdge_act_loads": hwdge_act_loads}),
        **({"weights_arena": True} if weights_arena else {}),
        **({"tail_chain": _TAIL_CHAIN} if overlap_calls else {}),
        last_gate_up_first=last_gate_up_first,
        last_relayout_runs=last_relayout_runs,
        last_swiglu_first=last_swiglu_first,
        last_gate_up_split=last_gate_up_split,
        merged_down_quads=merged_down_quads,
        **({"first_gather_index": first_gather_index, "dve_experts": True} if fast_first_gather else {}),
        store_parts=store_parts,
        relayout0_act=relayout0_act,
        last_group_halves=last_group_halves,
        act_relayout_blocks=act_relayout_blocks,
        last_finish_early=last_finish_early,
        pe_keep_alive=pe_keep_alive,
        **({"paired_downs": True} if paired_downs else {}),
        **({"long_stream_short": True} if long_stream_short else {}),
        **({"down_psum_dtype": "fp32", "fp32_down_rounding": "once"} if down_psum_dtype == "fp32" else {}),
        **({"down_group_mask": down_group_mask} if n_scale_sets == 2 else {}),
    )
    next(phases)  # "routing"
    if relayout_before_latent:
        next(phases)  # "relayout"
    if deferred:
        after_index(deferred[0])
    return SelectiveLoadStart(
        sbm=sbm,
        name_suffix=name_suffix,
        tokens=T,
        hidden=H,
        latent_in_sbuf=latent_in_sbuf,
        hidden_qtz=hidden_qtz,
        hidden_scale=hidden_scale,
        phases=phases,
        heap_depth=heap_depth,
        output=output,
    )


def selective_load_tp_finish(started: SelectiveLoadStart, latent: nl.NkiTensor) -> nl.NkiTensor:
    """Finish a started call: quantize the latent, then run the compute.

    Args:
        started (SelectiveLoadStart): the state of selective_load_tp_start.
        latent (nl.NkiTensor): bfloat16 [8, H] @HBM, or @SBUF with token t on partition t (latent_in_sbuf at start).

    Returns:
        output (nl.NkiTensor): bfloat16 [8, H] @HBM, the output of start (rows output_row .. output_row + 7).
    """
    sbm, name_suffix = started.sbm, started.name_suffix
    nisa, nl = _named_modules(name_suffix)
    T, H = latent.shape
    kernel_assert(
        len(latent.shape) == 2 and latent.dtype == nl.bfloat16 and (T, H) == (started.tokens, started.hidden),
        f"Expected a bf16 latent [{started.tokens}, {started.hidden}], got {latent.dtype} {latent.shape}",
    )
    kernel_assert(
        (latent.buffer == nl.sbuf) == started.latent_in_sbuf,
        f"The latent is in {latent.buffer}, but start had latent_in_sbuf={started.latent_in_sbuf}",
    )
    # Step 4: Quantize the latent (heap scratch), then the rest of the body.
    depth = len(sbm.heap)

    def alloc_heap(shape, dtype, name=None, **kwargs):
        return sbm.alloc_heap(shape, dtype, name=None if name == None else name + name_suffix, **kwargs)

    latent_ps = nl.ndarray(
        (_PMAX, (H // _H512) * _X4 * T),
        dtype=nl.bfloat16,
        buffer=nl.psum,
        address=(0, _LATENT_PSUM_BANK * _PSUM_BANK_BYTES),
        name="slp_latent_ps",
    )
    quantize_latent_mx(
        latent,
        alloc_heap,
        latent_ps,
        started.hidden_qtz,
        started.hidden_scale,
        (nisa.engine.sync,),
        prefix="slp_",
        name_suffix=name_suffix,
    )
    while len(sbm.heap) > depth:
        sbm.pop_heap()
    output = finish_phases(started.phases)
    sbm.close_scope()
    while len(sbm.heap) > started.heap_depth:
        sbm.pop_heap()
    return output


def _expert_rows(
    expert_gate_up_packed: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_down_weights_scale: nl.NkiTensor,
):
    """The weights with a leading layer dimension [L, E, ...] merged into the expert rows [L*E, ...]."""
    if len(expert_gate_up_packed.shape) == 3:
        return expert_gate_up_packed, expert_down_weights, expert_down_weights_scale
    shapes = tuple(tuple(w.shape) for w in (expert_gate_up_packed, expert_down_weights, expert_down_weights_scale))
    kernel_assert(
        len(shapes[0]) == 4
        and len(shapes[1]) == 4
        and len(shapes[2]) in (3, 4)
        and shapes[1][:2] == shapes[0][:2]
        and shapes[2][:2] == shapes[0][:2],
        f"Expected weights [E, ...] or stacks [L, E, ...], got {shapes}",
    )
    rows = shapes[0][0] * shapes[0][1]
    return (
        expert_gate_up_packed.reshape((rows, *shapes[0][2:])),
        expert_down_weights.reshape((rows, *shapes[1][2:])),
        expert_down_weights_scale.reshape((rows, *shapes[2][2:])),
    )


def _validate(
    expert_index: nl.NkiTensor,
    expert_affinities: nl.NkiTensor,
    expert_gate_up_packed: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_down_weights_scale: nl.NkiTensor,
):
    """Check the shapes, the dtypes and the buffers of the routing and the weights (8-token calls, top-16)."""
    E = expert_gate_up_packed.shape[0]
    H = expert_down_weights.shape[2]
    kernel_assert(
        tuple(expert_index.shape) in ((SHORT_TOKENS, _TOP_K), (TOKENS, _TOP_K)) and expert_index.dtype == nl.int32,
        f"Expected int32 expert_index [{TOKENS} or {SHORT_TOKENS}, {_TOP_K}], got {expert_index.dtype} "
        f"{expert_index.shape}",
    )
    kernel_assert(
        tuple(expert_affinities.shape) == tuple(expert_index.shape) and expert_affinities.dtype == nl.float32,
        f"Expected float32 expert_affinities {tuple(expert_index.shape)}, got {expert_affinities.dtype} "
        f"{expert_affinities.shape}",
    )
    kernel_assert(
        expert_index.buffer == expert_affinities.buffer, "expert_index and expert_affinities must be in the same buffer"
    )
    kernel_assert(H % _H512 == 0, f"Expected H % 512 == 0, got H={H}")
    kernel_assert(
        len(expert_gate_up_packed.shape) == 3 and expert_gate_up_packed.dtype == nl.uint8,
        f"Expected uint8 expert_gate_up_packed [E, 128, row bytes], got {expert_gate_up_packed.shape}",
    )
    kernel_assert(
        tuple(expert_down_weights.shape) == (E, _MX_ROWS, H) and expert_down_weights.dtype == nl.float4_e2m1fn_x4,
        f"Expected MXFP4 expert_down_weights [{E}, 8, {H}] (padded rows), got {expert_down_weights.dtype} "
        f"{expert_down_weights.shape}",
    )
    kernel_assert(
        tuple(expert_down_weights_scale.shape) in ((E, H), (E, 2, H)) and expert_down_weights_scale.dtype == nl.uint8,
        f"Expected uint8 expert_down_weights_scale [{E}, {H}] or [{E}, 2, {H}] (two scale groups), got "
        f"{expert_down_weights_scale.shape}",
    )


def _rank_slice(row_bytes: int, n_h512: int) -> int:
    """The intermediate rows of one rank (I) from the packed gate/up row bytes 4*I*(H/512 + S)."""
    return row_bytes // (_X4 * (n_h512 + div_ceil(2 * div_ceil(n_h512, 4), 4)))


def build_down_group_mask(
    rank_info: nl.NkiTensor, intermediate: int, mask: nl.NkiTensor, name: str, alloc: Callable
) -> nisa.NkiInstruction:
    """Write the down scale group of each partition's down row into mask (two down scale groups).

    The rank slice holds values [I*r, I*r + I) of each expert (r = the global rank). Its first checkpoint block of 32
    values holds n = min(I, 32 - (I*r) % 32) of them. Down row p = a % 8 holds values 4p .. 4p + 3, so it uses scale
    group 0 if 4p < n, else group 1 (the block boundary falls on a row: I and 32 are multiples of 4). By r % 4 at
    I = 24, group 0 has 6, 2, 4 or 6 rows.

    Args:
        rank_info (nl.NkiTensor): int32 [1, 4] @HBM, (global rank, DP rank, TP rank, 0).
        intermediate (int): I, the intermediate rows of one rank.
        mask (nl.NkiTensor): bfloat16 [128, 2] SBUF destination: column s is 1 where the row uses scale group s,
            else 0.
        name (str): a unique instruction name prefix.
        alloc (Callable): ``alloc(shape, dtype, name)`` returns an SBUF tile for the scratch values.

    Returns:
        nisa.NkiInstruction: the last instruction that writes mask.
    """
    kernel_assert(
        intermediate % _X4 == 0 and intermediate <= 32, f"Expected I % 4 == 0 and I <= 32, got I={intermediate}"
    )
    rank = alloc((_PMAX, 1), nl.int32, name + "_rank")
    work = alloc((_PMAX, 2), nl.int32, name + "_work")  # [a, (n, 4 * (a % 8))]
    nisa.dma_copy(
        dst=rank,
        src=rank_info.ap(pattern=[[0, _PMAX], [1, 1]], offset=0),
        dge_mode=nisa.dge_mode.hwdge,
        engine=nisa.engine.sync,
        name=name + "_load_rank",
    )
    dve = nisa.vector_engine
    # n = min(I, 32 - (I*r) % 32). An arithmetic op and a bitwise op do not share one instruction (the arithmetic
    # runs in fp32).
    nisa.tensor_scalar(
        dst=work[:, 0:1], data=rank, op0=nl.multiply, operand0=intermediate, engine=dve, name=name + "_start"
    )
    nisa.tensor_scalar(
        dst=work[:, 0:1], data=work[:, 0:1], op0=nl.bitwise_and, operand0=31, engine=dve, name=name + "_offset"
    )
    nisa.tensor_scalar(
        dst=work[:, 0:1],
        data=work[:, 0:1],
        op0=nl.multiply,
        operand0=-1,
        op1=nl.add,
        operand1=32,
        engine=dve,
        name=name + "_first_block",
    )
    nisa.tensor_scalar(
        dst=work[:, 0:1], data=work[:, 0:1], op0=nl.minimum, operand0=intermediate, engine=dve, name=name + "_count"
    )
    # 4 * (a % 8): the first value of the partition's down row
    nisa.iota(dst=work[:, 1:2], pattern=[[0, 1]], offset=0, channel_multiplier=1, name=name + "_partition")
    nisa.tensor_scalar(
        dst=work[:, 1:2],
        data=work[:, 1:2],
        op0=nl.bitwise_and,
        operand0=_MX_ROWS - 1,
        op1=nl.left_shift,
        operand1=2,
        engine=dve,
        name=name + "_row_value",
    )
    nisa.tensor_tensor(
        dst=mask[:, 0:1], data1=work[:, 1:2], data2=work[:, 0:1], op=nl.less, engine=dve, name=name + "_group0"
    )
    return nisa.tensor_tensor(
        dst=mask[:, 1:2], data1=work[:, 1:2], data2=work[:, 0:1], op=nl.greater_equal, engine=dve, name=name + "_group1"
    )


def _cast_index(nisa, dst: nl.NkiTensor, src: nl.NkiTensor, max_index: Optional[int], name: str):
    """Cast the expert ids src into dst; with max_index, clamp them to 0 .. max_index in the same op."""
    if max_index is None:
        return nisa.tensor_copy(dst=dst, src=src, engine=nisa.vector_engine, name=name)
    return nisa.tensor_scalar(
        dst=dst,
        data=src,
        op0=nl.maximum,
        operand0=0.0,
        op1=nl.minimum,
        operand1=float(max_index),
        engine=nisa.vector_engine,
        name=name,
    )


def broadcast_routing(
    sbm: SbufManager,
    expert_index: nl.NkiTensor,
    expert_affinities: nl.NkiTensor,
    index: nl.NkiTensor,
    affinity: nl.NkiTensor,
    name_suffix: str = "",
    expert_offset: int = 0,
    max_index: Optional[int] = None,
    hwdge_ids: Optional[nl.NkiTensor] = None,
    after_stream: Optional[nisa.NkiInstruction] = None,
    after_dve: Optional[nisa.NkiInstruction] = None,
    psum_bank: int = _ROUTING_PSUM_BANK,
):
    """Write the weight row of the expert of each pair n = t*16 + k (int32) and its bf16 affinity on every partition.

    index[a, n] = clamp(expert_index[t, k], 0, max_index) + expert_offset, affinity[a, n] = bf16(expert_affinities[t,
    k]). The offset is added after the broadcast, in FP32 (exact below 2^24). The clamp keeps the weight loads of a
    token with a bad id (for example -1 for NaN router scores) on valid expert rows.

    A ones matmul sums one nonzero term for each pair (fp16 ids: exact below 2048; bf16 affinities), so the broadcast
    is exact. SBUF routing tiles [T, K] on the token partitions: the moving operand of each 512-pair batch holds token
    t's values only in the columns of token t and exact zeros elsewhere (a predicated copy over a zeroed tile, not a
    multiply, so a NaN or Inf value stays in the columns of its own token). HBM routing: the index and the affinities
    load as one row on partition 0 (SP queue). PSUM: banks 2 (index) and 3 (affinity) for each batch of 512 pairs.

    Args:
        sbm (SbufManager): the region; the scratch tiles go on its heap (the caller pops them).
        expert_index (nl.NkiTensor): int32 [T, K] @HBM, or @SBUF on the token partitions.
        expert_affinities (nl.NkiTensor): float32 [T, K], same buffer as expert_index.
        index (nl.NkiTensor): int32 [128, T*K] SBUF destination.
        affinity (nl.NkiTensor): bfloat16 [128, T*K] SBUF destination.
        name_suffix (str): appended to every op and tensor name.
        expert_offset (int): added to every id after the clamp.
        max_index (Optional[int]): the largest id kept; None: no clamp.
        hwdge_ids (Optional[nl.NkiTensor]): int32 destination of the same weight rows before the broadcast, for the
            HWDGE expert loads: [T, K] on the token partitions (SBUF routing) or [1, T*K] on partition 0 (HBM routing).
            DVE writes it first, so the loads do not wait for the broadcast.
        after_stream (Optional[nisa.NkiInstruction]): the Pool iotas are scheduled after it (the last Pool stream
            instruction of an earlier call), so they do not hold that stream.
        after_dve (Optional[nisa.NkiInstruction]): SBUF routing: the cast of the expert ids is scheduled after it
            (fast_first_gather: the group-0 gather indices go first on DVE).
        psum_bank (int): _ROUTING_PSUM_BANK (2, default): banks 2 + 2 * (batch % 2) (index) and the next bank
            (affinity). _FREE_PSUM_BANK (6, one batch of at most 256 pairs): the index at byte 0 of bank 6 and the
            affinity at byte 1024.
    """
    nisa, nl = _named_modules(name_suffix)

    def alloc_heap(shape, dtype, name):
        return sbm.alloc_heap(shape, dtype, name=name + name_suffix)

    T, K = expert_index.shape
    n_pairs = T * K
    in_sbuf = expert_index.buffer == nl.sbuf
    rows = T if in_sbuf else 1

    def weight_rows(src: nl.NkiTensor):
        """hwdge_ids = clamp(src, 0, max_index) + expert_offset (int32; FP32 arithmetic is exact below 2^24)."""
        _cast_index(nisa, hwdge_ids, src, max_index, "slp_hwdge_ids_op")
        if expert_offset:
            nisa.tensor_scalar(
                dst=hwdge_ids,
                data=hwdge_ids,
                op0=nl.add,
                operand0=float(expert_offset),
                engine=nisa.vector_engine,
                name="slp_hwdge_ids_offset",
            )

    if hwdge_ids is not None and in_sbuf:
        weight_rows(expert_index)
    ones_f16 = alloc_heap((rows, _PMAX), nl.float16, name="slp_ones_f16")
    ones_bf16 = alloc_heap((rows, _PMAX), nl.bfloat16, name="slp_ones_bf16")
    nisa.memset(dst=ones_f16, value=1.0, name="slp_memset_ones_f16")
    nisa.memset(dst=ones_bf16, value=1.0, name="slp_memset_ones_bf16")
    moving_cols = _BANK_FP32 if in_sbuf else n_pairs
    index_moving = alloc_heap((rows, moving_cols), nl.float16, name="slp_index_moving")
    affinity_moving = alloc_heap((rows, moving_cols), nl.bfloat16, name="slp_affinity_moving")
    if in_sbuf:
        offsets = alloc_heap((T, _BANK_FP32), nl.float32, name="slp_pair_offsets")
        token_mask = alloc_heap((T, _BANK_FP32), nl.uint16, name="slp_pair_tokens")
        # The values in the moving dtypes (the cast rounds the affinities to bf16 once).
        index_values = alloc_heap((T, K), nl.float16, name="slp_index_values")
        affinity_values = alloc_heap((T, K), nl.bfloat16, name="slp_affinity_values")
        cast = _cast_index(nisa, index_values, expert_index, max_index, "slp_cast_index_values")
        if after_dve is not None:
            cast.depends_on(after_dve, kind=nisa.dependency.schedule)
        nisa.tensor_copy(
            dst=affinity_values, src=expert_affinities, engine=nisa.vector_engine, name="slp_cast_affinity_values"
        )
    else:
        # Expert ids are below 2048, so fp16 is exact. The affinities round to bf16 once.
        index_row = alloc_heap((1, n_pairs), nl.int32, name="slp_index_row")
        affinity_row = alloc_heap((1, n_pairs), nl.float32, name="slp_affinity_row")
        for values, row, name in ((expert_index, index_row, "index"), (expert_affinities, affinity_row, "affinity")):
            nisa.dma_copy(
                dst=row[0:1, 0:n_pairs],
                src=values.reshape((1, n_pairs)),
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.sync,
                name=f"slp_load_{name}_row",
            )
        if hwdge_ids is not None:
            weight_rows(index_row)
        _cast_index(nisa, index_moving, index_row, max_index, "slp_cast_index_row")
        nisa.tensor_copy(dst=affinity_moving, src=affinity_row, engine=nisa.vector_engine, name="slp_cast_affinity_row")

    tokens_per_batch = _BANK_FP32 // K
    one_bank = psum_bank != _ROUTING_PSUM_BANK
    kernel_assert(
        psum_bank in (_ROUTING_PSUM_BANK, _FREE_PSUM_BANK) and (not one_bank or n_pairs <= _BANK_FP32 // 2),
        f"psum_bank must be {_ROUTING_PSUM_BANK} or {_FREE_PSUM_BANK} (at most {_BANK_FP32 // 2} pairs), got {psum_bank}",
    )
    for batch in range(div_ceil(n_pairs, _BANK_FP32)):
        first = batch * _BANK_FP32
        cols = min(_BANK_FP32, n_pairs - first)
        if in_sbuf:
            moving = slice(0, cols)
            # token_mask[t, (r, k)] = 1 if t == tokens_per_batch * batch + r
            iota = nisa.iota(
                dst=offsets[:, 0:cols],
                pattern=[[-1, cols // K], [0, K]],
                offset=-tokens_per_batch * batch,
                channel_multiplier=1,
                name=f"slp_iota_pair_tokens{batch}",
            )
            if after_stream is not None:
                iota.depends_on(after_stream, kind=nisa.dependency.schedule)
            nisa.tensor_scalar(
                dst=token_mask[:, 0:cols],
                data=offsets[:, 0:cols],
                op0=nl.equal,
                operand0=0.0,
                engine=nisa.vector_engine,
                name=f"slp_pair_tokens{batch}",
            )
            for values, moving_tile, name in (
                (index_values, index_moving, "index"),
                (affinity_values, affinity_moving, "affinity"),
            ):
                nisa.memset(
                    dst=moving_tile[:, moving],
                    value=0.0,
                    engine=nisa.vector_engine,
                    name=f"slp_{name}_moving_zero{batch}",
                )
                nisa.tensor_copy_predicated(
                    dst=moving_tile[:, moving],
                    src=values.ap(pattern=[[K, T], [0, cols // K], [1, K]], offset=0),  # [t, (r, k)]: value of (t, k)
                    predicate=token_mask[:, 0:cols],
                    name=f"slp_{name}_moving{batch}",
                )
        else:
            moving = slice(first, first + cols)
        for j, (ones, moving_tile, dst, name) in enumerate(
            ((ones_f16, index_moving, index, "index"), (ones_bf16, affinity_moving, affinity, "affinity"))
        ):
            bank = _ROUTING_PSUM_BANK + 2 * (batch % 2) + j
            # one_bank: the index at byte 0 and the affinity at byte 1024 of psum_bank
            broadcast = nl.ndarray(
                (_PMAX, cols if one_bank else _BANK_FP32),
                dtype=nl.float32,
                buffer=nl.psum,
                address=(0, psum_bank * _PSUM_BANK_BYTES + j * _PSUM_BANK_BYTES // 2)
                if one_bank
                else (0, bank * _PSUM_BANK_BYTES),
                name=f"slp_{name}_bcast_ps{batch}",
            )
            # broadcast[a, n] = sum over the rows of moving[:, n]: one nonzero term per pair
            nisa.nc_matmul(
                dst=broadcast[:, 0:cols], stationary=ones, moving=moving_tile[:, moving], name=f"slp_mm_{name}{batch}"
            )
            if name == "index" and expert_offset:
                nisa.tensor_scalar(
                    dst=dst[:, first : first + cols],
                    data=broadcast[:, 0:cols],
                    op0=nl.add,
                    operand0=float(expert_offset),
                    engine=nisa.vector_engine,
                    name=f"slp_copy_{name}{batch}",
                )
            else:
                nisa.tensor_copy(
                    dst=dst[:, first : first + cols],
                    src=broadcast[:, 0:cols],
                    engine=nisa.vector_engine,
                    name=f"slp_copy_{name}{batch}",
                )


def first_gather_indices(
    sbm: SbufManager,
    expert_index: nl.NkiTensor,
    first_gather_index: nl.NkiTensor,
    name_suffix: str,
    expert_offset: int,
    max_index: int,
    after_stream: Optional[nisa.NkiInstruction] = None,
    psum_address: int = _FIRST_GATHER_PSUM_BANK * _PSUM_BANK_BYTES,
) -> nisa.NkiInstruction:
    """fast_first_gather: the gather indices of group 0 (the 16 experts of token 0) from the SBUF routing tile.

    first_gather_index[a, b] = 128 * (clamp(expert_index[0, a % 16], 0, max_index) + expert_offset) + 8b + a / 16, the
    values of the body's early-gather index (gather row r = 128b + a lands on partition 8b + a / 16, slot a % 16). A
    ones matmul of 16 columns puts token 0's fp16 ids (exact below 2048) on every partition (PSUM bank 4), and a masked
    row sum keeps the id of pair a % 16. The constants need no routing, so the scheduler places them early.

    Args:
        sbm (SbufManager): the region; the scratch goes on its heap (the caller pops it).
        expert_index (nl.NkiTensor): int32 [T, K] SBUF routing tile (token t on partition t).
        first_gather_index (nl.NkiTensor): int32 [128, 16] SBUF destination.
        name_suffix (str): appended to every op and tensor name.
        expert_offset (int): the weight row of expert 0.
        max_index (int): the largest id kept (the clamp of broadcast_routing).
        after_stream (Optional[nisa.NkiInstruction]): the Pool iotas are scheduled after it.
        psum_address (int): PSUM byte address of token 0's ids (64 B per partition): bank 4 (default), or byte 512 of
            bank 6 with overlap_calls (between the index and the affinity of the routing broadcast).

    Returns:
        nisa.NkiInstruction: the last DVE instruction (the gather-index op).
    """
    nisa, nl = _named_modules(name_suffix)
    dve = nisa.vector_engine

    def alloc_heap(shape, dtype, name):
        return sbm.alloc_heap(shape, dtype, name=name + name_suffix)

    def iota(**kwargs):
        op = nisa.iota(**kwargs)
        if after_stream is not None:
            op.depends_on(after_stream, kind=nisa.dependency.schedule)
        return op

    ones = alloc_heap((1, _PMAX), nl.float16, "slp_fg_ones")
    ids = alloc_heap((1, _TOP_K), nl.float16, "slp_fg_ids")
    part = alloc_heap((_PMAX, 2), nl.int32, "slp_fg_part")  # [a, 0]: a % 16, [a, 1]: a / 16
    cols = alloc_heap((_PMAX, _TOP_K), nl.int32, "slp_fg_cols")  # [a, k]: k, then 8k (the lanes base)
    mask = alloc_heap((_PMAX, _TOP_K), nl.float32, "slp_fg_mask")  # [a, k]: 1 if k == a % 16
    lanes = alloc_heap((_PMAX, _TOP_K), nl.int32, "slp_fg_lanes")  # [a, b]: 8b + a / 16
    prod = alloc_heap((_PMAX, _TOP_K), nl.float32, "slp_fg_prod")
    expert = alloc_heap((_PMAX, 1), nl.float32, "slp_fg_expert")
    # Constants.
    nisa.memset(dst=ones, value=1.0, engine=dve, name="slp_fg_memset_ones")
    iota(dst=part, pattern=[[0, 2]], offset=0, channel_multiplier=1, name="slp_fg_iota_part")  # a
    nisa.tensor_scalar(
        dst=part[:, 0:1], data=part[:, 0:1], op0=nl.bitwise_and, operand0=_TOP_K - 1, engine=dve, name="slp_fg_amod"
    )
    nisa.tensor_scalar(
        dst=part[:, 1:2], data=part[:, 1:2], op0=nl.right_shift, operand0=4, engine=dve, name="slp_fg_adiv"
    )
    iota(dst=cols, pattern=[[1, _TOP_K]], offset=0, channel_multiplier=0, name="slp_fg_iota_cols")
    nisa.tensor_tensor(
        dst=mask,
        data1=cols,
        data2=part.ap(pattern=[[2, _PMAX], [0, _TOP_K]], offset=0),  # [a, k]: a % 16
        op=nl.equal,
        engine=dve,
        name="slp_fg_mask_op",
    )
    iota(dst=lanes, pattern=[[_MX_ROWS, _TOP_K]], offset=0, channel_multiplier=0, name="slp_fg_iota_lanes")  # 8b
    nisa.tensor_tensor(
        dst=lanes,
        data1=lanes,
        data2=part.ap(pattern=[[2, _PMAX], [0, _TOP_K]], offset=1),  # [a, b]: a / 16
        op=nl.add,
        engine=dve,
        name="slp_fg_lanes_op",
    )
    # Token 0's ids on every partition, then the id of pair a % 16 (+ the offset), then the gather indices.
    _cast_index(nisa, ids, expert_index[0:1, 0:_TOP_K], max_index, "slp_fg_cast")
    ids_ps = nl.ndarray(
        (_PMAX, _TOP_K),
        dtype=nl.float32,
        buffer=nl.psum,
        address=(0, psum_address),
        name="slp_fg_ids_ps",
    )
    nisa.nc_matmul(dst=ids_ps, stationary=ones, moving=ids, name="slp_fg_mm")  # [a, k]: id of pair k
    nisa.scalar_tensor_tensor(
        dst=prod,
        data=ids_ps,
        op0=nl.add,
        operand0=float(expert_offset),
        op1=nl.multiply,
        operand1=mask,
        reduce_res=expert,  # the row sum: the weight row of the expert of pair a % 16
        name="slp_fg_expert_op",
    )
    return nisa.NkiInstruction(  # scalar_tensor_tensor returns a raw instruction
        nisa.scalar_tensor_tensor(
            dst=first_gather_index,
            data=expert.ap(pattern=[[1, _PMAX], [0, _TOP_K]], offset=0),  # [a, b]: the weight row
            op0=nl.multiply,
            operand0=_PMAX,
            op1=nl.add,
            operand1=lanes,  # [a, b]: row 8b + a / 16
            name="slp_fg_gather_index",
        )
    )
