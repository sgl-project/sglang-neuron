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

"""Kimi-K3 TP128 selective-load routed-expert decode: one rank's slice of all experts, top-16 routing."""

import itertools
from typing import Callable

import nki
import nki.isa as nisa
import nki.language as nl

from ...utils.allocator import SbufManager
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import div_ceil
from ...utils.logging import get_logger
from ...utils.trace_names import next_call, restart_calls
from .selective_load_packed import (
    _WEIGHT_SBUF_BUDGET,
    _instruction,
    _Suffixed,
    finish_phases,
    selective_load_packed_body,
    selective_load_packed_phases,
)

_PMAX = 128
_X4 = 4
_H512 = 512
_TOP_K = 16
# Tokens of one call of the packed body: 64 routed pairs at top-16 (chunk_tokens = 8: 128 pairs).
CHUNK_TOKENS = 4
MAX_TOKENS = 256
# Most tokens of a call whose expert pick (one nc_n_gather from 16 pairs per token on each partition) runs in one
# pass: 512 data elements per partition. Above it the gather runs several passes and signals completion only after
# the last one, which a reader on another engine cannot rely on. Served callers stay at or below it.
PICK_GATHER_MAX_TOKENS = 32
_PSUM_BANK_BYTES = 2048
# FP32 columns of one PSUM bank: the routing broadcast moves this many pairs per matmul.
_BANK_FP32 = _PSUM_BANK_BYTES // 4
# PSUM banks of the prologue: latent transposes in banks 0-1, routing broadcasts in banks 2-5. The body reuses all of
# them after the prologue.
_LATENT_PSUM_BANK = 0
_ROUTING_PSUM_BANK = 2
_ISA, _LANGUAGE = nisa, nl
# Packed-body gather layout for one sub-group per group (H < 4608 or short rows), per chunk of N/16 groups: the
# gathered gate/up groups 1 .. N/16 - 1 x 16 pairs, the down weights of each group, one down-scale tile per 4 groups.
# A gather row index past _GATHER_SKIP is out of bounds.
_GROUP_PAIRS = 16
_SCALE_GROUPS = 4
_DOWN_ROWS = 8
_GATHER_SKIP = 1 << 20
# index_matmul: the cast clamps the expert ids to at most this (fp16 holds the integers up to 2048 exactly), and a
# Vector engine transpose moves at most this many token rows (32) at a time.
_FP16_EXACT_ID = 2047
_VECTOR_TRANSPOSE_ROWS = 32
# Default chunk_overlap from this many chunks on.
_OVERLAP_MIN_CHUNKS = 2
# Alignment of the first SBUF buffer of the body (its gate/up rows, the DMA destination of the weight stream).
_BODY_ALIGN = 512
# Standalone: SBUF bytes per partition at the top that the region leaves to the compiler (the identity matrix of the
# PE transposes and the tiles of a caller without a region).
COMPILER_SBUF_RESERVE = 16384


@nki.jit
def moe_sel_load_tp(
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
    chunk_overlap: bool = None,
    affinity_scaling: str = "pre_quant",
    chunk_tokens: int = CHUNK_TOKENS,
    down_psum_dtype: str = "bf16",
    rank_info: nl.NkiTensor = None,
) -> nl.NkiTensor:
    """Compute one TP128 rank's unnormalized routed-expert partial of a Kimi-K3 decode step.

    Each rank holds its 24-column intermediate slice of all 896 experts. The kernel loads only the routed experts of
    each token (selective load) and runs the packed selective-load body once per chunk of 4 tokens (64 routed
    pairs), or of 8 tokens (128 routed pairs, see chunk_tokens), and once for the last 1 or 2 tokens. The sum of
    the partials of the 128 ranks is the routed-expert output.

    Dimensions:
        T: tokens, 1 <= T <= 256. K: experts per token, 16. E: experts, 896. H: latent size, 3584.
        I: intermediate slice of one rank, 24. n = t*K + k: the routed pair of token t and its k-th expert.

    Args:
        latent (nl.NkiTensor): bfloat16 [T, H] @HBM, or @SBUF with token t on partition t (T <= 128).
        expert_index (nl.NkiTensor): int32 (or uint32) [T, K], the routed experts of each token. @HBM, or @SBUF on
            the token partitions (T <= 128).
        expert_affinities (nl.NkiTensor): float32 [T, K], the routing weights of the routed experts (the sigmoid
            scores renormalized over the K selected experts, times the routed scaling factor). Same buffer as
            expert_index.
        expert_gate_up_packed (nl.NkiTensor): uint8 [E, 128, 768] @HBM, the gate/up weights of this rank's slice
            with their MX scales packed into the rows.
        expert_down_weights (nl.NkiTensor): float4_e2m1fn_x4 (or a uint16 carrier) [E, 8, H] @HBM. Row p, lane q
            holds intermediate column i = q*(I/4) + p; rows p >= I/4 are not read. The compact form [E, I/4, H]
            holds only the rows that the kernel reads, with the same output.
        expert_down_weights_scale (nl.NkiTensor): uint8 [E, H] @HBM, one MX scale per expert and latent column. Or
            uint8 [E, 2, H]: the scales of the two checkpoint blocks
            of the rank slice, so every down value is the checkpoint value (needs rank_info).
        beta (float): SiTU gate soft-clipping factor.
        linear_beta (float): SiTU up soft-clipping factor.
        affinity_hi_lo (bool): carry the affinity-scaled SiTU output as two MXFP8 terms (hi + lo, about 8
            significant bits) instead of one (about 4).
        sigmoid_from_tanh (bool): compute sigmoid(g) as (1 + tanh(g/2)) / 2, so that the activation engine uses only
            the tanh table.
        hold_sources (bool): ignored. The compiler orders every later writer of an output-store source after the store
            completes, so the kernel needs no hold.
        chunk_overlap (bool): build the gather indices of every chunk in the start phase, alternate two down-scale
            buffers and run the Pool weight streams of the chunks back to back, so the loads of chunk c + 1 overlap
            the compute of chunk c. The result does not change. None (default): on for 2 or more chunks (T > 4);
            on trn3 it saves time from T = 8 on, and costs a little time at T = 4.
        affinity_scaling (str): "pre_quant" (default): the affinity scales the activation before its MXFP8
            quantization (hi + lo terms with affinity_hi_lo). "post_down": the activation is quantized unscaled, the
            affinity scales each pair's down output and a matmul adds the 16 pairs of a token (affinity_hi_lo does
            not apply).
        chunk_tokens (int): tokens of one body call. 4 (default): 64 routed pairs. 8: 128 routed pairs (8 groups),
            then 4-token calls for the rest. The result does not change. An 8-token call uses about 215 KB of SBUF
            for each partition.
        down_psum_dtype (str): the PSUM dtype of the down projection, "bf16" (default) or "fp32" (see
            ``moe_sel_load_tp_start``).
        rank_info (nl.NkiTensor): int32 [1, 4] @HBM, (global rank, DP rank, TP rank, 0). The global rank is the
            TP128 rank. Needed with two down scale groups (and read only then).

    Returns:
        output (nl.NkiTensor): bfloat16 [T, H] @HBM, this rank's partial.

    Notes:
        - Every chunk is one call of the packed body, and the latent quantization and the routing broadcast work on
          each token alone. A body call gives each token its own stationary column and its own group (K = 16), and
          the other groups add zeros to its output row (a one-token call adds the zero in its eviction). So the
          result of a token depends only on its latent, experts and affinities, and not on T, the chunk size, the
          chunk or the position in the chunk.
        - The last 1 or 2 tokens (T % 4) run in a short call of 1 or 2 groups, with no padding tokens. 3 last tokens,
          post_down and a last token without hi / lo pad to a 4-token call: a padding token has a zero latent, expert
          0 and zero affinities.
        - Requires LNC1 and Trn3. The body uses all 8 PSUM banks at fixed addresses and about 130 KB of SBUF for
          each partition (4-token calls).
    """
    restart_calls()
    kernel_assert(
        nl.program_ndim() == 0 or (nl.program_ndim() == 1 and nl.num_programs(0) == 1),
        "moe_sel_load_tp supports LNC1 only",
    )
    kernel_assert(nisa.get_nc_version() >= nisa.nc_version.gen4, "moe_sel_load_tp requires Trn3 or newer")
    return moe_sel_load_tp_body(
        latent,
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
        None,
        None,
        0,
        chunk_overlap,
        affinity_scaling,
        chunk_tokens=chunk_tokens,
        down_psum_dtype=down_psum_dtype,
        rank_info=rank_info,
    )


def moe_sel_load_tp_body(
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
    sbuf=None,
    output: nl.NkiTensor = None,
    output_row: int = 0,
    chunk_overlap: bool = None,
    affinity_scaling: str = "pre_quant",
    expert_offset: int = 0,
    chunk_tokens: int = CHUNK_TOKENS,
    down_psum_dtype: str = "bf16",
    rank_info: nl.NkiTensor = None,
    down_group_mask: nl.NkiTensor = None,
) -> nl.NkiTensor:
    """The body of ``moe_sel_load_tp`` for a caller that inlines it (not ``@nki.jit``): start, then finish.

    Args:
        latent, expert_index, expert_affinities, expert_gate_up_packed, expert_down_weights,
        expert_down_weights_scale, beta, linear_beta, affinity_hi_lo, sigmoid_from_tanh, hold_sources,
            chunk_overlap, affinity_scaling, chunk_tokens, down_psum_dtype, rank_info: as in ``moe_sel_load_tp``.
        sbuf, output, output_row, expert_offset, down_group_mask: as in ``moe_sel_load_tp_start``.

    Returns:
        output (nl.NkiTensor): bfloat16 [T, H] @HBM.
    """
    started = moe_sel_load_tp_start(
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
        sbuf,
        output,
        output_row,
        chunk_overlap,
        affinity_scaling,
        expert_offset,
        chunk_tokens=chunk_tokens,
        down_psum_dtype=down_psum_dtype,
        latent_in_sbuf=latent.buffer == nl.sbuf,
        rank_info=rank_info,
        down_group_mask=down_group_mask,
    )
    return moe_sel_load_tp_finish(started, latent)


class SelLoadStart:
    """The state of a started kernel call, from ``moe_sel_load_tp_start`` to ``moe_sel_load_tp_finish``."""

    def __init__(self, **state):
        self.__dict__.update(state)


def moe_sel_load_tp_start(
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
    output: nl.NkiTensor = None,
    output_row: int = 0,
    chunk_overlap: bool = None,
    affinity_scaling: str = "pre_quant",
    expert_offset: int = 0,
    after_index: Callable = None,
    chunk_tokens: int = CHUNK_TOKENS,
    latent_in_sbuf: bool = True,
    sbuf_reserve: int = 0,
    persistent_buffers: bool = False,
    sparse_schedule: bool = False,
    group0_gather: bool = True,
    setup_after_stream: bool = True,
    down_psum_dtype: str = "bf16",
    rank_info: nl.NkiTensor = None,
    down_group_mask: nl.NkiTensor = None,
    long_stream: bool = False,
    index_broadcast_first: bool = False,
    index_matmul: bool = False,
    stream_after=None,
    before_stream: Callable = None,
    index_onehot_resident: bool = False,
    relayout_order: bool = False,
    g0_relayout_engine: str = "dve",
    cross_call_order: bool = False,
    private_out: bool = False,
) -> SelLoadStart:
    """Start a call before the latent is ready: routing broadcast, chunk-0 weight loads and Pool gathers.

    The work needs only the routing, so a caller can issue it before the instructions that make the latent (for
    example the latent projection and its all-gather). Between start and finish the kernel keeps its stack scope of
    the region open: a region stack frame that the caller opens after start must close before finish. The caller
    may use PSUM between start and finish only for tiles that are dead at finish.

    Args:
        expert_index, expert_affinities, expert_gate_up_packed, expert_down_weights, expert_down_weights_scale,
        beta, linear_beta, affinity_hi_lo, sigmoid_from_tanh, hold_sources, chunk_overlap, affinity_scaling,
        chunk_tokens, rank_info: as in ``moe_sel_load_tp``. The weights may also be stacks of layers: gate/up
            [L, E, 128, 768], down [L, E, 8, H] (or the compact [L, E, I/4, H]) and scales [L, E, H] (or [L, E, 2, H]),
            taken as [L*E, ...] (see expert_offset).
        sbuf: the SBUF region of the kernel: an ``SbufManager`` with explicit addresses, or an object with an
            ``sbm`` attribute (``ExplicitSbuf``). The kernel takes its buffers from a new stack scope and from the heap
            of the region, and returns all of them at finish. None: a manager of SBUF without the top
            COMPILER_SBUF_RESERVE bytes (standalone).
        output: optional bfloat16 HBM destination with H columns. None: a new [T, H] tensor.
        output_row: the row of output for token 0.
        expert_offset: the weight row of expert 0: expert e reads row expert_offset + e (layer l of a stack:
            l*E). The offset goes into the dynamic index of every weight DMA, so no DMA has a static offset of
            4 GiB or more.
        after_index: optional callback ``after_index(after)``, called once after chunk 0 issues its first Pool
            weight gathers. ``after`` is the last instruction that builds their gather indices. The caller can put
            work there that must not delay the weight stream. None: no call (the default program).
        latent_in_sbuf: the latent of finish is in SBUF. With chunk_tokens = 8 the ring depths of the 8-token calls
            fit the free SBUF, and an HBM latent (False) needs room for its load next to the first call.
        sbuf_reserve: SBUF bytes per partition of the region that the 8-token calls leave free, for the tiles that
            the compiler places while a call runs (for example the buffers of a collective between start and finish).
        persistent_buffers: the calls on one region share their buffers (the body buffers, the stationary, the
            routing and gather tiles, and the HBM output when output is None) when the name, shape, dtype and
            address match. Every call then reuses the DMA descriptor table entries of the first one. The result does
            not change. The calls share one HBM output when output is None, so read it before the next call.
        sparse_schedule: the body calls chain only the first gate/up matmul of each run to the PE schedule (fewer
            schedule edges, a faster trace of a large kernel). The result does not change.
        group0_gather: each body call loads the gate/up rows of group 0 with one Pool gather instead of 16 HWDGE loads
            on SP and ACT (default). The start phase then builds the gather indices of every chunk (as with
            chunk_overlap), also for one chunk. The result does not change. In a large kernel the issue of a dynamic
            HWDGE DMA on ACT gets slow from a high descriptor table index (from about layer 25 of a 93-layer decode
            step), and the Pool gather does not. False: the HWDGE loads (faster in a small standalone kernel at
            T >= 16).
        down_psum_dtype: the PSUM dtype of the down projection of the body calls ("bf16" or "fp32", see
            selective_load_packed_phases). "fp32" with hi / lo rounds each output value once (the store DMA adds the
            fp32 hi and lo terms); "bf16" rounds the two terms and their sum.
        setup_after_stream: the Pool iotas of each body call after chunk 0 are scheduled after the gathers of the
            call before it (default). Otherwise the compiler can put them ahead of those gathers, where they wait
            for the late DVE setup of that call and hold the Pool weight stream (a delay in each call). The result
            does not change.
        down_group_mask: with two down scale groups, a bfloat16 [128, 2] SBUF tile from ``build_down_group_mask``
            that lives until finish. A caller that runs many calls builds it once. None: the start builds it from
            rank_info.
        long_stream: the schedule option of the body calls (``selective_load_packed_phases``); it acts on the
            8-token calls of chunk_tokens = 8. The result does not change. False (default) keeps the program.
        index_broadcast_first: with an SBUF routing, the Vector engine copies the broadcast expert index out of PSUM
            before it starts on the affinities (their cast and moving tiles), so the gather indices of the first Pool
            gathers do not wait for the affinity work. The result does not change. False (default) keeps the program.
        index_matmul: with an SBUF routing and group0_gather, TensorE one-hot matmuls on the transposed expert index
            give the gather indices (``_gather_indices_matmul``), in place of the broadcast index tile and the Pool
            expert pick. The gate/up indices of chunk 0 come first, and after_index gets their instruction. The
            affinity broadcast follows all gather indices, and index_broadcast_first has no effect. The index of every
            loaded weight row and the result do not change for expert ids up to 2047 (the cast clamps larger ids to
            2047). False (default) keeps the program.
        stream_after: an instruction that the caller issued before the start, for example a collective trigger. The
            Pool weight stream is scheduled after it (a schedule edge to the first gather of chunk 0; the later
            gathers follow that one), so every weight gather depends on it. It needs the precomputed gather indices
            (group0_gather or chunk_overlap). None (default) keeps the program.
        before_stream: optional callback ``before_stream(after)``, called once after the gather indices and before
            the body of chunk 0. ``after`` is the instruction that ``after_index`` gets. The callback may return an
            instruction: the Pool weight stream (and the setup iotas of chunk 0) is then scheduled after it, as after
            stream_after. A schedule edge adds no hardware wait, so the gathers still issue when their indices are
            ready, while the compiler orders their consumers on the other engines after that instruction. It needs
            the precomputed gather indices. None (default) keeps the program.
        index_onehot_resident: with index_matmul, the routing-independent one-hot stationary of the index matmuls
            (``_index_onehot``: 6 GpSimd iotas and a Vector mask) is built once, in the first call, on the heap of
            ``sbuf`` below the heaps of every call, and later calls read it. Each call otherwise rebuilds it, and its
            iotas hold the in-order GpSimd queue in front of the work that precedes the next MoE layer. The
            gather indices and the result do not change. False (default) keeps the program.
        relayout_order: the re-layout order of long_stream for the body calls of up to 4 groups
            (``selective_load_packed_phases``): the DVE copies of group g + 2 run after finish(g) and before
            SwiGLU(g + 1), so gate/up(g + 2) does not wait for copies that the compiler put behind SwiGLU(g + 1). The
            result does not change. False (default) keeps the program.
        g0_relayout_engine: "dve" (default) or "act": the engine of the group-0 weight re-layout copies of the body
            calls after the first one (the first call keeps DVE). ACT is not blocked by the Pool descriptor windows of
            the call's gathers. The copies move uint16 bits, so the result does not change.
        cross_call_order: with g0_relayout_engine="act": at each boundary between two body calls of the same size,
            the ACT evictions of the first call's last group (and the stores of their chunks) are emitted in the next
            call, right after its ACT group-0 re-layout copies, and scheduled after them. The next call's first gate/up
            matmul then does not wait for its group-0 copies behind the previous call's evictions on ACT. (A schedule
            edge cannot point back to a later instruction, so the evictions move instead.) Both calls keep the same
            instructions on the same data; the result does not change. No effect with one body call. False (default)
            keeps the program.
        private_out: the new HBM output of output None is private HBM instead of shared HBM (opt-in; for a caller
            kernel that keeps every per-core scratch tensor private, as one compiled with ``--lnc1-with-shared-mem``,
            where a ``shared_hbm`` scratch tensor is one buffer for the two cores of an HBM stack). Only the buffer
            changes, so the result is the same. No effect with an output. False (default) keeps the program.

    Returns:
        SelLoadStart: the state for ``moe_sel_load_tp_finish``.
    """
    expert_gate_up_packed, expert_down_weights, expert_down_weights_scale = _expert_rows(
        expert_gate_up_packed, expert_down_weights, expert_down_weights_scale
    )
    _validate_weights(
        expert_index, expert_affinities, expert_gate_up_packed, expert_down_weights, expert_down_weights_scale
    )
    n_rows = expert_gate_up_packed.shape[0]
    kernel_assert(0 <= expert_offset < n_rows, f"expert_offset must be in [0, {n_rows}), got {expert_offset}")
    # Two down scale groups ([E, 2, H]): each down row on the scale of its own checkpoint block.
    n_scale_sets = len(expert_down_weights_scale.shape) - 1
    kernel_assert(
        n_scale_sets == 1 or down_group_mask is not None or rank_info is not None,
        "Down scales [E, 2, H] need rank_info or down_group_mask",
    )
    # A skipped down row must stay out of bounds, and FP32 holds every gate/up gather row exactly.
    kernel_assert(
        n_rows * _DOWN_ROWS <= _GATHER_SKIP and n_rows * _PMAX <= 1 << 24,
        f"At most {_GATHER_SKIP // _DOWN_ROWS} expert rows, got {n_rows}",
    )
    kernel_assert(chunk_tokens in (CHUNK_TOKENS, 2 * CHUNK_TOKENS), f"chunk_tokens must be 4 or 8, got {chunk_tokens}")
    kernel_assert(
        g0_relayout_engine in ("dve", "act"), f"g0_relayout_engine must be 'dve' or 'act', got {g0_relayout_engine}"
    )
    kernel_assert(
        not cross_call_order or g0_relayout_engine == "act", "cross_call_order needs g0_relayout_engine='act'"
    )
    # Op names must be unique in a kernel: the second and later calls of a kernel add a suffix to every name.
    call = next_call("sel_load_tp")
    suffix = f"_c{call}" if call else ""
    T = expert_index.shape[0]
    H = expert_down_weights.shape[2]
    n_h512 = H // _H512
    row_bytes = expert_gate_up_packed.shape[2]
    one_sub_group = not (n_h512 >= 9 and row_bytes >= 2048)  # U = 1
    group0_gather = group0_gather and one_sub_group
    # The last 1 or 2 tokens run in a short call (with group0_gather: precomputed gather indices). 3 tokens,
    # post_down, one token without hi / lo (an odd down stationary) and group-0 HWDGE loads pad to a 4-token call.
    rest = T % CHUNK_TOKENS
    keep_short = affinity_scaling == "pre_quant" and group0_gather and (rest == 2 or (rest == 1 and affinity_hi_lo))
    padded = T if keep_short else div_ceil(T, CHUNK_TOKENS) * CHUNK_TOKENS
    plan = _chunk_plan(padded, chunk_tokens)
    # A 1-token call has a zero second stationary column (nc_matmul_mx needs an even stationary width).
    stationary_plan = [(first, tokens + tokens % 2) for first, tokens in plan]
    stationary_cols = sum(tokens for _, tokens in stationary_plan) * n_h512
    n_chunks = len(plan)
    if sbuf is None:
        sbuf = SbufManager(
            0, nl.tile_size.total_available_sbuf_size - COMPILER_SBUF_RESERVE, get_logger("moe_sel_load_tp")
        )
    sbm = getattr(sbuf, "sbm", sbuf)
    if expert_index.dtype == nl.uint32:
        expert_index = expert_index.view(nl.int32)
    if expert_down_weights.dtype == nl.uint16:
        expert_down_weights = expert_down_weights.view(nl.float4_e2m1fn_x4)
    cache = None
    if persistent_buffers:
        if not hasattr(sbm, "sel_load_buffers"):
            sbm.sel_load_buffers = {}
        cache = sbm.sel_load_buffers

    def alloc_heap(shape, dtype, name):
        tile = sbm.alloc_heap(shape, dtype, name=name + suffix)
        return tile if cache is None else cache.setdefault((name, shape, str(dtype), sbm.heap_curr_addr), tile)

    if output is None and cache is not None and ("output", T, H) in cache:
        output = cache[("output", T, H)]
    elif output is None:
        buffer = nl.private_hbm if private_out else nl.shared_hbm
        output = nl.ndarray((T, H), dtype=nl.bfloat16, buffer=buffer, name="moe_sel_load_tp_out" + suffix)
        if cache is not None:
            cache[("output", T, H)] = output

    # index_onehot_resident: the one-hot stationary lives on the heap below every call's heap (allocated before the
    # heap depth that finish restores), so the first call builds it and the others read it.
    onehot = None
    if index_onehot_resident:
        kernel_assert(index_matmul, "index_onehot_resident needs index_matmul")
        onehot = getattr(sbm, "sel_index_onehot", None)
        if onehot is None:
            onehot = sbm.alloc_heap(_INDEX_ONEHOT_SHAPE, nl.float16, name="sel_index_onehot_resident")
            _index_onehot(_Suffixed(_ISA, suffix), onehot)
            sbm.sel_index_onehot = onehot
    # The prologue tiles live on the heap, so the stack of the body starts at the scope base, as in the standalone
    # body, and the weight stream of chunk 0 does not wait for the prologue reads.
    heap_depth = len(sbm.heap)
    # The stationary of a chunk of C tokens from token f: columns n_h512*f .. n_h512*(f + C') - 1, [hb, t] (C' = C
    # rounded up to even).
    if n_scale_sets == 2 and down_group_mask is None:
        down_group_mask = alloc_heap((_PMAX, 2), nl.bfloat16, "sel_down_group_mask")
        build_down_group_mask(
            rank_info, _rank_slice(expert_gate_up_packed, H), down_group_mask, "sel_down_groups" + suffix, alloc_heap
        )
    hidden_qtz = alloc_heap((_PMAX, stationary_cols), nl.float8_e4m3fn_x4, "sel_hidden_qtz")
    hidden_scale = alloc_heap((_PMAX, stationary_cols), nl.uint8, "sel_hidden_scale")
    index_all = alloc_heap((_PMAX, padded * _TOP_K), nl.int32, "sel_index")
    affinity_all = alloc_heap((_PMAX, padded * _TOP_K), nl.bfloat16, "sel_affinity")
    # Gather indices of every chunk and two down-scale buffers: the Pool stream of chunk c + 1 waits only for the
    # buffers that chunk c still reads, not for DVE work in program order.
    if chunk_overlap is None:
        chunk_overlap = n_chunks >= _OVERLAP_MIN_CHUNKS
    # group0_gather reads the precomputed gather indices, also with one chunk.
    precomputed = (chunk_overlap or group0_gather) and one_sub_group
    gather_first = list(itertools.accumulate((_gather_cols(tokens, group0_gather) for _, tokens in plan), initial=0))
    scale_tiles = max(_scale_tiles(tokens) for _, tokens in plan)  # of the down-scale buffer of a chunk
    gather_all = down_s_all = None
    setups = [None] * n_chunks
    if precomputed:
        gather_all = alloc_heap((_PMAX, gather_first[-1]), nl.int32, "sel_gather_index")
        down_s_all = alloc_heap((_PMAX, min(2, n_chunks) * scale_tiles * n_scale_sets * H), nl.uint8, "sel_down_s")
        # The setup iotas of chunk c write set c % 2 (a call has tokens**2 token offsets at top-16), so they wait only
        # for the reads of the call two chunks before.
        n_sets, offset_cols = min(2, n_chunks), max(tokens * tokens for _, tokens in plan)
        offsets_all = alloc_heap((_PMAX, n_sets * _GROUP_PAIRS), nl.float32, "sel_setup_offsets")
        token_offsets_all = alloc_heap((_PMAX, n_sets * offset_cols), nl.float32, "sel_setup_token_offsets")
        setups = [
            (
                offsets_all[:, (chunk % 2) * _GROUP_PAIRS : (chunk % 2 + 1) * _GROUP_PAIRS],
                token_offsets_all[:, (chunk % 2) * offset_cols : (chunk % 2 + 1) * offset_cols],
            )
            for chunk in range(n_chunks)
        ]
    persistent_depth = len(sbm.heap)
    segments = div_ceil(2 * div_ceil(n_h512, 4), 4)
    indices_done = all_indices = None
    if index_matmul:
        kernel_assert(
            expert_index.buffer == nl.sbuf and group0_gather,
            "index_matmul needs an SBUF routing and group0_gather (with one gate/up sub-group)",
        )
        # The gather indices first, the affinity broadcast after them (index_all is not written).
        indices_done, all_indices = _gather_indices_matmul(
            sbm,
            expert_index,
            gather_all,
            _chunk_runs(plan, gather_first),
            padded,
            row_bytes // (4 * (n_h512 + segments)) // _X4,
            suffix,
            expert_down_weights.shape[1],
            expert_offset,
            n_rows - 1 - expert_offset,
            onehot,
        )
    _broadcast_routing(
        sbm,
        expert_index,
        expert_affinities,
        index_all,
        affinity_all,
        padded,
        suffix,
        expert_offset,
        n_rows - 1 - expert_offset,
        **({"affinity_after": all_indices} if index_matmul else {}),
        **({"index_first": True} if index_broadcast_first and not index_matmul else {}),
    )
    if precomputed and not index_matmul:
        indices_done = _gather_indices(
            sbm,
            index_all,
            gather_all,
            _chunk_runs(plan, gather_first),
            row_bytes // (4 * (n_h512 + segments)) // _X4,
            suffix,
            group0_gather,
            down_stride=expert_down_weights.shape[1],
        )
    # The routing scratch returns to the region. A later tile at its addresses waits for its last reads.
    while len(sbm.heap) > persistent_depth:
        sbm.pop_heap()

    # One self-contained body call for each chunk. Chunk 0 runs its first phase now.
    sbm.open_scope(name="sel_chunks" + suffix)
    sbm.align_stack_curr_addr(_BODY_ALIGN)
    budgets = [None] * n_chunks  # 4-token calls: the default ring depths
    if chunk_tokens > CHUNK_TOKENS:
        # The rings of an 8-token call fit the free SBUF; the latent scratch of finish sits next to chunk 0.
        free = sbm.heap_curr_addr - sbm.stack_curr_addr
        scratch = min(padded, _PMAX) * n_h512 * _X4 * 2 + (0 if latent_in_sbuf else H * 2)
        intermediate = row_bytes // (_X4 * (n_h512 + div_ceil(2 * div_ceil(n_h512, 4), 4)))
        for chunk, (_, tokens) in enumerate(plan):
            if tokens > CHUNK_TOKENS:
                other = _body_other_bytes(
                    tokens, H, intermediate, precomputed, affinity_scaling == "post_down", n_scale_sets
                )
                budgets[chunk] = min(_WEIGHT_SBUF_BUDGET, free - sbuf_reserve - other - (scratch if chunk == 0 else 0))
    kernel_assert(stream_after is None or precomputed, "stream_after needs the precomputed gather indices")
    # Pool runs the weight stream of the chunks one after the other (after stream_after, if given).
    stream_chain = [] if stream_after is None else [_instruction(stream_after)]
    chunks = [
        dict(
            sbm=sbm,
            hidden_qtz=hidden_qtz[:, first * n_h512 : (first + tokens + tokens % 2) * n_h512],
            hidden_scale=hidden_scale[:, first * n_h512 : (first + tokens + tokens % 2) * n_h512],
            routed_tokens=tokens,
            expert_gate_up_packed=expert_gate_up_packed,
            expert_down_weights=expert_down_weights,
            expert_down_weights_scale=expert_down_weights_scale,
            expert_index=index_all[:, first * _TOP_K : (first + tokens) * _TOP_K],
            expert_affinities=affinity_all[:, first * _TOP_K : (first + tokens) * _TOP_K],
            inputs_in_sbuf=True,
            affinity_hi_lo=affinity_hi_lo,
            situ=(beta, linear_beta),
            sigmoid_from_tanh=sigmoid_from_tanh,
            output=output,
            output_row=output_row + first,
            valid_tokens=min(tokens, T - first),
            gather_index_in=None
            if gather_all is None
            else gather_all[:, gather_first[chunk] : gather_first[chunk + 1]],
            down_s_in=None
            if down_s_all is None
            else down_s_all[
                :,
                (chunk % 2) * scale_tiles * n_scale_sets * H : ((chunk % 2) * scale_tiles + _scale_tiles(tokens))
                * n_scale_sets
                * H,
            ],
            stream_chain=stream_chain if precomputed else None,
            setup_in=setups[chunk],
            **({} if budgets[chunk] is None else {"weight_sbuf_budget": budgets[chunk]}),
            **({} if cache is None else {"buffer_cache": cache}),
            **({"sparse_schedule": True} if sparse_schedule else {}),
            **({"group0_gather": True} if group0_gather else {}),
            **({"setup_after_stream": True} if setup_after_stream else {}),
            **({"down_psum_dtype": down_psum_dtype} if down_psum_dtype != "bf16" else {}),
            **({"down_group_mask": down_group_mask} if n_scale_sets == 2 else {}),
            **({"long_stream": True} if long_stream else {}),
            **({"relayout_order": True} if relayout_order else {}),
            **({"g0_relayout_engine": g0_relayout_engine} if g0_relayout_engine != "dve" and chunk > 0 else {}),
            **({"hooks": {}} if cross_call_order and n_chunks > 1 else {}),
            affinity_scaling=affinity_scaling,
        )
        for chunk, (first, tokens) in enumerate(plan)
    ]
    if cross_call_order and n_chunks > 1:
        _wire_call_boundaries(chunks, plan)
    if after_index is not None:
        # With precomputed indices the body has no gather-index instruction of its own.
        chunks[0]["finish_affinities"] = lambda after: after_index(indices_done if after is None else after)
    if before_stream is not None:
        kernel_assert(precomputed, "before_stream needs the precomputed gather indices")
        pin = before_stream(indices_done)
        if pin is not None:
            stream_chain.append(_instruction(pin))
    first = selective_load_packed_phases(**chunks[0])
    next(first)
    return SelLoadStart(
        sbm=sbm,
        suffix=suffix,
        tokens=T,
        plan=stationary_plan,
        latent_in_sbuf=latent_in_sbuf,
        hidden_qtz=hidden_qtz,
        hidden_scale=hidden_scale,
        chunks=chunks,
        first=first,
        heap_depth=heap_depth,
        output=output,
    )


def moe_sel_load_tp_finish(
    started: SelLoadStart, latent: nl.NkiTensor, between_chunks: Callable = None
) -> nl.NkiTensor:
    """Finish a started call: quantize the latent, then run the compute of every chunk.

    Args:
        started (SelLoadStart): the state of ``moe_sel_load_tp_start``.
        latent (nl.NkiTensor): bfloat16 [T, H] @HBM, or @SBUF with token t on partition t (T <= 128).
        between_chunks: optional callback ``between_chunks(after)``, called once after the body of chunk 0 and before
            chunk 1 (after chunk 0 with one chunk). ``after`` is the last Pool weight gather of chunk 0. The callback
            may return an instruction, for example a collective trigger that it schedules after ``after``; the Pool
            stream of chunk 1 is then scheduled after it, so the instruction runs between the two streams on Pool and
            the later weight gathers depend on it. It needs the precomputed gather indices (group0_gather or
            chunk_overlap). The caller must not use PSUM in it. None (default) keeps the program.

    Returns:
        output (nl.NkiTensor): bfloat16 [T, H] @HBM.
    """
    sbm = started.sbm
    T, H = latent.shape
    kernel_assert(
        len(latent.shape) == 2 and latent.dtype == nl.bfloat16 and T == started.tokens,
        f"Expected bf16 latent [{started.tokens}, H], got {latent.dtype} {latent.shape}",
    )
    kernel_assert(
        H == started.chunks[0]["expert_down_weights"].shape[2], f"The latent has {H} columns; the weights have more"
    )
    kernel_assert(
        T <= (_PMAX if latent.buffer == nl.sbuf else MAX_TOKENS), f"Unsupported T={T} for a latent in {latent.buffer}"
    )
    kernel_assert(
        latent.buffer == nl.sbuf or started.latent_in_sbuf is False or started.plan[0][1] == CHUNK_TOKENS,
        "With chunk_tokens = 8, an HBM latent needs latent_in_sbuf=False at start",
    )
    persistent_depth = len(sbm.heap)
    _quantize_latent(sbm, latent, started.hidden_qtz, started.hidden_scale, started.plan, started.suffix)
    while len(sbm.heap) > persistent_depth:
        sbm.pop_heap()
    finish_phases(started.first)
    if between_chunks is not None:
        chain = started.chunks[0]["stream_chain"]
        kernel_assert(bool(chain), "between_chunks needs the precomputed gather indices")
        pin = between_chunks(chain[-1])
        if pin is not None:
            chain.append(_instruction(pin))
    for chunk in started.chunks[1:]:
        selective_load_packed_body(**chunk)
    sbm.close_scope()
    while len(sbm.heap) > started.heap_depth:
        sbm.pop_heap()
    return started.output


def _wire_call_boundaries(chunks: list, plan: list) -> None:
    """cross_call_order: at each boundary between two body calls of the same size, the first call defers the ACT
    evictions of its last group (and the stores of their chunks), and the next call emits them right after its group-0
    re-layout, scheduled after its last ACT re-layout instruction. Calls of the same size have the same SBUF buffers at
    the same addresses (checked at the boundary), so a deferred eviction writes only the previous call's output tile."""
    for i in range(len(chunks) - 1):
        if plan[i][1] != plan[i + 1][1]:
            continue
        before, after = chunks[i]["hooks"], chunks[i + 1]["hooks"]
        before["defer_act_evicts"] = True

        def flush(anchor, before=before, after=after):
            n = len(after["layout"])
            kernel_assert(
                before["layout"][:n] == after["layout"],
                "cross_call_order: two body calls of the same size must have the same SBUF buffers",
            )
            for emit in before["deferred"]:
                emit(anchor)
            del before["deferred"][:]

        after["after_g0"] = flush


def _chunk_plan(tokens: int, chunk_tokens: int) -> list[tuple[int, int]]:
    """The (first token, tokens) of each body call: chunks of chunk_tokens, 4-token chunks, then the last 1-2 tokens."""
    wide = tokens // chunk_tokens * chunk_tokens
    full = tokens // CHUNK_TOKENS * CHUNK_TOKENS
    plan = [(first, chunk_tokens) for first in range(0, wide, chunk_tokens)]
    plan += [(first, CHUNK_TOKENS) for first in range(wide, full, CHUNK_TOKENS)]
    return plan + ([(full, tokens - full)] if tokens > full else [])


def _scale_tiles(tokens: int) -> int:
    """Down-scale tiles of a body call of `tokens` tokens (one group per token at top-16, 4 groups per tile)."""
    return div_ceil(tokens * _TOP_K // _GROUP_PAIRS, _SCALE_GROUPS)


def _gather_cols(tokens: int, group0: bool = False) -> int:
    """Gather-index columns of a body call of `tokens` tokens (layout of the body's gather_index, U = 1). group0: with
    the gate/up columns of group 0 (group0_gather)."""
    groups = tokens * _TOP_K // _GROUP_PAIRS
    return (groups - 1 + group0) * _GROUP_PAIRS + groups + _scale_tiles(tokens)


def _body_other_bytes(
    tokens: int, H: int, intermediate: int, precomputed: bool, post_down: bool, scale_sets: int = 1
) -> int:
    """An upper bound of the stack bytes per partition of a body call outside its weight budget (top-16, SBUF routing,
    SiTU): out_sb, the down scales (without precomputed buffers), the SiTU and gated tiles, the stationary and the
    constants, and the post_down tiles."""
    other = 2 * H + 3 * 2 * _GROUP_PAIRS * intermediate * 4 + 2 * _X4 * _PMAX * 2 + 4096 * scale_sets
    if not precomputed:
        other += _scale_tiles(tokens) * scale_sets * H
    if post_down:
        other += 2 * H + 8192
    return other


def _chunk_runs(plan: list[tuple[int, int]], gather_first: list[int]) -> list[tuple[int, int, int, int]]:
    """(first token, chunks, tokens per chunk, first gather_all column) of each run of equal chunks of plan."""
    runs = []
    for chunk, (first, tokens) in enumerate(plan):
        if runs and runs[-1][2] == tokens:
            runs[-1][1] += 1
        else:
            runs.append([first, 1, tokens, gather_first[chunk]])
    return [tuple(run) for run in runs]


def _expert_rows(
    expert_gate_up_packed: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_down_weights_scale: nl.NkiTensor,
) -> tuple[nl.NkiTensor, nl.NkiTensor, nl.NkiTensor]:
    """Return the weights with a leading layer dimension [L, E, ...] merged into the expert rows [L*E, ...]."""
    shapes = tuple(tuple(w.shape) for w in (expert_gate_up_packed, expert_down_weights, expert_down_weights_scale))
    # Two down scale groups add an axis to the scales: [E, 2, H] or [L, E, 2, H].
    scale_dims = len(shapes[2]) - (len(shapes[2]) == len(shapes[0]))
    if len(shapes[0]) == 3 and len(shapes[1]) == 3 and scale_dims == 2:
        return expert_gate_up_packed, expert_down_weights, expert_down_weights_scale
    kernel_assert(
        len(shapes[0]) == 4
        and len(shapes[1]) == 4
        and scale_dims == 3
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


def _validate_weights(
    expert_index: nl.NkiTensor,
    expert_affinities: nl.NkiTensor,
    expert_gate_up_packed: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_down_weights_scale: nl.NkiTensor,
):
    """Check the shapes, the dtypes and the buffers of the routing and the weights."""
    T = expert_index.shape[0]
    E = expert_gate_up_packed.shape[0]
    H = expert_down_weights.shape[2]
    kernel_assert(1 <= T <= MAX_TOKENS, f"Expected 1 <= T <= {MAX_TOKENS}, got T={T}")
    kernel_assert(H % _H512 == 0, f"Expected H % 512 == 0, got H={H}")
    kernel_assert(
        tuple(expert_index.shape) == (T, _TOP_K) and expert_index.dtype in (nl.int32, nl.uint32),
        f"Expected int32 expert_index [T, {_TOP_K}], got {expert_index.dtype} {expert_index.shape}",
    )
    kernel_assert(
        tuple(expert_affinities.shape) == (T, _TOP_K) and expert_affinities.dtype == nl.float32,
        f"Expected float32 expert_affinities [{T}, {_TOP_K}], got {expert_affinities.dtype} {expert_affinities.shape}",
    )
    kernel_assert(
        expert_index.buffer == expert_affinities.buffer,
        "expert_index and expert_affinities must be in the same buffer",
    )
    kernel_assert(
        expert_index.buffer != nl.sbuf or T <= _PMAX, f"SBUF routing tiles hold at most {_PMAX} tokens, got T={T}"
    )
    kernel_assert(
        len(expert_gate_up_packed.shape) == 3 and expert_gate_up_packed.dtype == nl.uint8,
        f"Expected uint8 expert_gate_up_packed [E, 128, row bytes], got {expert_gate_up_packed.shape}",
    )
    # The down rows of each expert in HBM: 8 (padded) or I/4 (compact: only the rows that the kernel reads).
    n_h512 = H // _H512
    compact_rows = expert_gate_up_packed.shape[2] // (_X4 * (n_h512 + div_ceil(2 * div_ceil(n_h512, 4), 4))) // _X4
    kernel_assert(
        tuple(expert_down_weights.shape) in ((E, 8, H), (E, compact_rows, H))
        and expert_down_weights.dtype in (nl.float4_e2m1fn_x4, nl.uint16),
        f"Expected MXFP4 expert_down_weights [{E}, 8, H] or [{E}, {compact_rows}, H], got {expert_down_weights.dtype} "
        f"{expert_down_weights.shape}",
    )
    kernel_assert(
        tuple(expert_down_weights_scale.shape) in ((E, H), (E, 2, H)) and expert_down_weights_scale.dtype == nl.uint8,
        f"Expected uint8 expert_down_weights_scale [{E}, {H}] or [{E}, 2, {H}], got {expert_down_weights_scale.shape}",
    )


def _rank_slice(expert_gate_up_packed: nl.NkiTensor, H: int) -> int:
    """The intermediate rows of one rank (I) from the packed gate/up row bytes 4*I*(H/512 + S)."""
    n_h512 = H // _H512
    return expert_gate_up_packed.shape[-1] // (_X4 * (n_h512 + div_ceil(2 * div_ceil(n_h512, 4), 4)))


def build_down_group_mask(
    rank_info: nl.NkiTensor, intermediate: int, mask: nl.NkiTensor, name: str, alloc: Callable
) -> nisa.NkiInstruction:
    """Write the down scale group of each partition's down row into mask (two down scale groups).

    The rank slice holds values [I*r, I*r + I) of each expert (r = the global rank). Its first checkpoint block of 32
    values holds n = min(I, 32 - (I*r) % 32) of them. Down row p = a % 8 holds values 4p .. 4p + 3, so it uses scale
    group 0 if 4p < n, else group 1 (the block boundary falls on a row: I and 32 are multiples of 4).

    Args:
        rank_info: int32 [1, 4] @HBM, (global rank, DP rank, TP rank, 0).
        intermediate: I, the intermediate rows of one rank.
        mask: bfloat16 [128, 2] SBUF destination: column s is 1 where the row uses scale group s, else 0.
        name: a unique instruction name prefix.
        alloc: ``alloc(shape, dtype, name)`` returns an SBUF tile for the scratch values.

    Returns:
        nisa.NkiInstruction: the last instruction that writes mask.
    """
    kernel_assert(
        intermediate % _X4 == 0 and intermediate <= 32, f"Expected I % 4 == 0 and I <= 32, got I={intermediate}"
    )
    rank = alloc((_PMAX, 1), nl.int32, name + "_rank")
    work = alloc((_PMAX, 2), nl.int32, name + "_work")
    nisa.dma_copy(
        dst=rank,
        src=rank_info.ap(pattern=[[0, _PMAX], [1, 1]], offset=0),
        dge_mode=nisa.dge_mode.hwdge,
        engine=nisa.engine.sync,
        name=name + "_load_rank",
    )
    dve, pool = nisa.vector_engine, nisa.gpsimd_engine
    # n = min(I, 32 - (I*r) % 32). An arithmetic op and a bitwise op do not share one instruction. The int32
    # multiplies by integers are the exact integer address op, which only GpSimd has; the minimum runs in fp32.
    nisa.tensor_scalar(
        dst=work[:, 0:1], data=rank, op0=nl.multiply, operand0=intermediate, engine=pool, name=name + "_start"
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
        engine=pool,
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
        operand0=_DOWN_ROWS - 1,
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


def _quantize_latent(
    sbm: SbufManager,
    latent: nl.NkiTensor,
    hidden_qtz: nl.NkiTensor,
    hidden_scale: nl.NkiTensor,
    plan: list[tuple[int, int]],
    suffix: str,
):
    """Quantize the natural [T, H] latent to the MXFP8 stationary layout of the body, chunk-major.

    Element [p, n_h512*f + C*hb + r] of hidden_qtz packs the 4 values of H = 512*hb + 4*p + q (q = 0..3) of token
    f + r of the chunk of C tokens from token f (plan), and hidden_scale has their MX scales in the quadrant layout
    of quantize_mx. Each stride-4 column view moves to the partitions with a PE transpose. A quantization group (8
    partitions x 4 lanes) holds the values of one token.
    """
    nisa, nl = _Suffixed(_ISA, suffix), _Suffixed(_LANGUAGE, suffix, functions=("ndarray",))

    def alloc_heap(shape, dtype, name):
        return sbm.alloc_heap(shape, dtype, name=name + suffix)

    T, H = latent.shape
    n_h512 = H // _H512
    padded = sum(tokens for _, tokens in plan)
    in_sbuf = latent.buffer == nl.sbuf
    swizzled_cols = min(padded, _PMAX) * n_h512 * _X4
    swizzled = alloc_heap((_PMAX, swizzled_cols), nl.bfloat16, name="sel_latent_swizzled")  # [p, (c, hb, r, q)]
    if not in_sbuf:
        source = alloc_heap((min(T, _PMAX), H), nl.bfloat16, name="sel_latent")
    for tile in range(div_ceil(padded, _PMAX)):
        first_token = tile * _PMAX
        rows = min(_PMAX, T - first_token)
        # (first token in the tile, chunks, chunk tokens, real tokens per chunk): runs of equal full chunks, and
        # each chunk with padding tokens alone.
        runs = []
        for first, tokens in plan:
            if not first_token <= first < first_token + _PMAX:
                continue
            first -= first_token
            n_rows = min(tokens, rows - first)
            if runs and runs[-1][2] == tokens and runs[-1][3] == tokens == n_rows:
                runs[-1][1] += 1
            else:
                runs.append([first, 1, tokens, n_rows])
        tile_padded = runs[-1][0] + runs[-1][1] * runs[-1][2]
        if any(n_rows < tokens for _, _, tokens, n_rows in runs):
            nisa.memset(dst=swizzled, value=0.0, name=f"sel_latent_pad{tile}")
        if in_sbuf:
            tile_src = latent
        else:
            nisa.dma_copy(
                dst=source[0:rows, :],
                src=latent[first_token : first_token + rows, :],
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.sync,
                name=f"sel_load_latent{tile}",
            )
            tile_src = source
        # PSUM writes start on 4-byte boundaries, so the lane blocks of an odd token count get one more column.
        lane_cols = tile_padded + tile_padded % 2
        for block in range(n_h512):
            # [p, (q, t)]: H = 512*block + 4*p + q of token t of the tile
            transposed = nl.ndarray(
                (_PMAX, _X4 * lane_cols),
                dtype=nl.bfloat16,
                buffer=nl.psum,
                address=(0, (_LATENT_PSUM_BANK + block % 2) * _PSUM_BANK_BYTES),
                name=f"sel_latent_ps{tile}_{block}",
            )
            for q in range(_X4):
                nisa.nc_transpose(
                    dst=transposed[:, q * lane_cols : q * lane_cols + rows],
                    # [t, p]: column 512*block + 4*p + q of the token rows of the tile
                    data=tile_src.ap(pattern=[[H, rows], [_X4, _PMAX]], offset=block * _H512 + q),
                    name=f"sel_latent_transpose_t{tile}_b{block}_q{q}",
                )
            # [p, (c, r, q)] of this block for the real tokens of each run of chunks.
            for first, n_cols, tokens, n_rows in runs:
                nisa.tensor_copy(
                    dst=swizzled.ap(
                        pattern=[[swizzled_cols, _PMAX], [n_h512 * tokens * _X4, n_cols], [_X4, n_rows], [1, _X4]],
                        offset=(first * n_h512 + block * tokens) * _X4,
                    ),
                    src=transposed.ap(
                        pattern=[[_X4 * lane_cols, _PMAX], [tokens, n_cols], [1, n_rows], [lane_cols, _X4]],
                        offset=first,
                    ),
                    engine=nisa.vector_engine,
                    name=f"sel_latent_swizzle_t{tile}_b{block}_c{first // CHUNK_TOKENS}",
                )
        cols = slice(first_token * n_h512, (first_token + tile_padded) * n_h512)
        nisa.quantize_mx(
            dst=hidden_qtz[:, cols],
            src=swizzled[:, 0 : tile_padded * n_h512 * _X4],
            dst_scale=hidden_scale[:, cols],
            name=f"sel_latent_quantize{tile}",
        )


def _cast_index(nisa, dst: nl.NkiTensor, src: nl.NkiTensor, max_index: int, name: str):
    """Cast the expert ids ``src`` into ``dst``; with ``max_index``, clamp them to 0 .. max_index in the same op."""
    if max_index is None:
        nisa.tensor_copy(dst=dst, src=src, engine=nisa.vector_engine, name=name)
        return
    nisa.tensor_scalar(
        dst=dst,
        data=src,
        op0=nl.maximum,
        operand0=0.0,
        op1=nl.minimum,
        operand1=float(max_index),
        engine=nisa.vector_engine,
        name=name,
    )


def _broadcast_routing(
    sbm: SbufManager,
    expert_index: nl.NkiTensor,
    expert_affinities: nl.NkiTensor,
    index_all: nl.NkiTensor,
    affinity_all: nl.NkiTensor,
    padded: int,
    suffix: str,
    expert_offset: int = 0,
    max_index: int = None,
    index_first: bool = False,
    affinity_after: nisa.NkiInstruction = None,
):
    """Write the expert of each pair n = t*16 + k (int32) and its bf16 affinity on every partition.

    The index tile gets expert + expert_offset (the weight row of the expert), added after the broadcast in FP32.
    With ``max_index``, the cast clamps each expert id to 0 .. max_index. The top-16 routing gives id -1 to a token
    with NaN scores, and the clamp keeps the weight gathers of that token on valid expert rows.

    A ones matmul sums one nonzero term for each pair, so the broadcast is exact. SBUF tiles: the moving operand of
    each 512-pair batch holds token t's values only in the columns of token t, and exact zeros elsewhere (a select
    over the token partitions, not a multiply, so a NaN or Inf value stays in the columns of its own token).
    HBM: the index and the affinities go to one row on partition 0. The padding pairs get expert 0 and affinity 0.
    With ``index_first`` (SBUF only), the affinity work of each batch (the cast, then the moving tile) follows the
    index copy out of PSUM on the Vector engine (schedule edges). With ``affinity_after`` (an instruction, SBUF only),
    only the affinities are broadcast (index_all is not written), and the affinity work of each batch follows that
    instruction on the Vector engine.
    """
    nisa, nl = _Suffixed(_ISA, suffix), _Suffixed(_LANGUAGE, suffix, functions=("ndarray",))

    def alloc_heap(shape, dtype, name):
        return sbm.alloc_heap(shape, dtype, name=name + suffix)

    T = expert_index.shape[0]
    n_pairs = padded * _TOP_K
    in_sbuf = expert_index.buffer == nl.sbuf
    kernel_assert(in_sbuf or not index_first, "index_first needs the routing in SBUF")
    kernel_assert(in_sbuf or affinity_after is None, "affinity_after needs the routing in SBUF")
    with_index = affinity_after is None
    deferred = index_first or not with_index  # the affinity work of each batch follows an instruction
    rows = T if in_sbuf else 1
    ones_f16 = alloc_heap((rows, _PMAX), nl.float16, name="sel_ones_f16") if with_index else None
    ones_bf16 = alloc_heap((rows, _PMAX), nl.bfloat16, name="sel_ones_bf16")
    if with_index:
        nisa.memset(dst=ones_f16, value=1.0, name="sel_memset_ones_f16")
    nisa.memset(dst=ones_bf16, value=1.0, name="sel_memset_ones_bf16")
    moving_shape = (rows, n_pairs if not in_sbuf else _BANK_FP32)
    index_moving = alloc_heap(moving_shape, nl.float16, name="sel_index_moving") if with_index else None
    affinity_moving = alloc_heap(moving_shape, nl.bfloat16, name="sel_affinity_moving")
    if in_sbuf:
        offsets = alloc_heap((T, _BANK_FP32), nl.float32, name="sel_pair_offsets")
        token_mask = alloc_heap((T, _BANK_FP32), nl.uint16, name="sel_pair_tokens")
        # The values in the moving dtypes (the rounding of the cast is that of the former multiply by 1.0).
        index_values = alloc_heap((T, _TOP_K), nl.float16, name="sel_index_values") if with_index else None
        affinity_values = alloc_heap((T, _TOP_K), nl.bfloat16, name="sel_affinity_values")
        if with_index:
            _cast_index(nisa, index_values, expert_index, max_index, "sel_cast_index_values")

        def cast_affinities():
            return nisa.tensor_copy(
                dst=affinity_values, src=expert_affinities, engine=nisa.vector_engine, name="sel_cast_affinity_values"
            )

        if not deferred:
            cast_affinities()
    else:
        # Expert ids are below 2048, so fp16 is exact. The affinities round to bf16 once.
        index_row = alloc_heap((1, n_pairs), nl.int32, name="sel_index_row")
        affinity_row = alloc_heap((1, n_pairs), nl.float32, name="sel_affinity_row")
        if padded > T:
            nisa.memset(dst=index_row, value=0, name="sel_index_row_pad")
            nisa.memset(dst=affinity_row, value=0.0, name="sel_affinity_row_pad")
        for values, row, name in ((expert_index, index_row, "index"), (expert_affinities, affinity_row, "affinity")):
            nisa.dma_copy(
                dst=row[0:1, 0 : T * _TOP_K],
                src=values.reshape((1, T * _TOP_K)),
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.sync,
                name=f"sel_load_{name}_row",
            )
        _cast_index(nisa, index_moving, index_row, max_index, "sel_cast_index_row")
        nisa.tensor_copy(dst=affinity_moving, src=affinity_row, engine=nisa.vector_engine, name="sel_cast_affinity_row")

    tokens_per_batch = _BANK_FP32 // _TOP_K
    for batch in range(div_ceil(n_pairs, _BANK_FP32)):
        first = batch * _BANK_FP32
        cols = min(_BANK_FP32, n_pairs - first)
        if in_sbuf:
            moving_cols = slice(0, cols)
            # token_mask[t, (r, k)] = 1 if t == tokens_per_batch * batch + r
            nisa.iota(
                dst=offsets[:, 0:cols],
                pattern=[[-1, cols // _TOP_K], [0, _TOP_K]],
                offset=-tokens_per_batch * batch,
                channel_multiplier=1,
                name=f"sel_iota_pair_tokens{batch}",
            )
            nisa.tensor_scalar(
                dst=token_mask[:, 0:cols],
                data=offsets[:, 0:cols],
                op0=nl.equal,
                operand0=0.0,
                engine=nisa.vector_engine,
                name=f"sel_pair_tokens{batch}",
            )

            def moving_tile(values, moving, name):
                zero = nisa.memset(
                    dst=moving[:, moving_cols],
                    value=0.0,
                    engine=nisa.vector_engine,
                    name=f"sel_{name}_moving_zero{batch}",
                )
                nisa.tensor_copy_predicated(
                    dst=moving[:, moving_cols],
                    src=values.ap(pattern=[[_TOP_K, T], [0, cols // _TOP_K], [1, _TOP_K]], offset=0),  # [t, (r, k)]
                    predicate=token_mask[:, 0:cols],
                    name=f"sel_{name}_moving{batch}",
                )
                return zero

            if with_index:
                moving_tile(index_values, index_moving, "index")
            if not deferred:
                moving_tile(affinity_values, affinity_moving, "affinity")
        else:
            moving_cols = slice(first, first + cols)
        copy = None
        for j, (ones, moving, dst, name) in enumerate(
            ((ones_f16, index_moving, index_all, "index"), (ones_bf16, affinity_moving, affinity_all, "affinity"))
        ):
            if name == "index" and not with_index:
                continue
            if deferred and name == "affinity":
                start = cast_affinities() if batch == 0 else None
                zero = moving_tile(affinity_values, affinity_moving, name)
                _instruction(zero if start is None else start).depends_on(
                    _instruction(copy if with_index else affinity_after), kind=_ISA.dependency.schedule
                )
            bank = _ROUTING_PSUM_BANK + 2 * (batch % 2) + j
            broadcast = nl.ndarray(
                (_PMAX, _BANK_FP32),
                dtype=nl.float32,
                buffer=nl.psum,
                address=(0, bank * _PSUM_BANK_BYTES),
                name=f"sel_{name}_bcast_ps{batch}",
            )
            nisa.nc_matmul(
                dst=broadcast[:, 0:cols], stationary=ones, moving=moving[:, moving_cols], name=f"sel_mm_{name}{batch}"
            )
            if name == "index" and expert_offset:
                copy = nisa.tensor_scalar(
                    dst=dst[:, first : first + cols],
                    data=broadcast[:, 0:cols],
                    op0=nl.add,
                    operand0=float(expert_offset),
                    engine=nisa.vector_engine,
                    name=f"sel_copy_{name}{batch}",
                )
            else:
                copy = nisa.tensor_copy(
                    dst=dst[:, first : first + cols],
                    src=broadcast[:, 0:cols],
                    engine=nisa.vector_engine,
                    name=f"sel_copy_{name}{batch}",
                )


def _gather_indices(
    sbm: SbufManager,
    index_all: nl.NkiTensor,
    gather_all: nl.NkiTensor,
    runs: list[tuple[int, int, int, int]],
    down_rows: int,
    suffix: str,
    group0: bool = False,
    down_stride: int = _DOWN_ROWS,
) -> nisa.NkiInstruction:
    """Build the gather indices of the packed body for every chunk (layout of its gather_index, one sub-group).

    A chunk of C tokens has G = C groups (top-16), so G - 1 gathered gate/up groups and St = G/4 down-scale tiles.
    Column (j - 1)*16 + b (gate/up of group j = 1..G-1): 128 * expert(16j + a%16) + 8b + a/16. Column 16(G - 1) + g
    (down weights of group g): 8 * expert(16g + a/8) + a%8, + skip for a%8 >= down_rows (I/4). Column 17G - 16 + s
    (down-scale tile s): expert(64s + 16(a/4 % 4) + 4(a/32) + a%4), + skip for a/16 odd. Pair n of the chunk from
    token f is column 16f + n of index_all. runs: (first token, chunks, C, first gather_all column) of each run of
    equal chunks. group0: the gate/up columns start with group 0 (16 more columns, group0_gather). down_stride: the
    HBM rows of each expert's down block (8, or I/4 for the compact down weights) in place of the 8 of the down index.

    Returns:
        nisa.NkiInstruction: the last instruction that writes gather_all.
    """
    nisa, nl = _Suffixed(_ISA, suffix), _Suffixed(_LANGUAGE, suffix, functions=("ndarray",))

    def alloc_heap(shape, dtype, name):
        return sbm.alloc_heap(shape, dtype, name=name + suffix)

    dve = nisa.vector_engine
    total_cols = gather_all.shape[1]
    depth = len(sbm.heap)
    # fields[a, :] = a%16, a/16, a/8, a%8 (+ skip for a%8 >= down_rows), down-scale pair, skip for a/16 odd, a
    fields = alloc_heap((_PMAX, 7), nl.int32, name="sel_gather_fields")
    work = alloc_heap((_PMAX, 2), nl.int32, name="sel_gather_work")
    nisa.iota(dst=fields[:, 6:7], pattern=[[0, 1]], offset=0, channel_multiplier=1, name="sel_gather_partition")
    a = fields[:, 6:7]
    for col, op0, operand0 in ((0, nl.bitwise_and, 15), (1, nl.right_shift, 4), (2, nl.right_shift, 3)):
        nisa.tensor_scalar(
            dst=fields[:, col : col + 1], data=a, op0=op0, operand0=operand0, engine=dve, name=f"sel_gather_field{col}"
        )
    nisa.tensor_scalar(
        dst=work[:, 0:1], data=a, op0=nl.bitwise_and, operand0=7, engine=dve, name="sel_gather_row_in_block"
    )
    nisa.tensor_scalar(
        dst=work[:, 1:2],
        data=work[:, 0:1],
        op0=nl.greater_equal,
        operand0=down_rows,
        op1=nl.multiply,
        operand1=_GATHER_SKIP,
        engine=dve,
        name="sel_gather_down_row_skip",
    )
    nisa.tensor_tensor(
        dst=fields[:, 3:4], data1=work[:, 0:1], data2=work[:, 1:2], op=nl.add, engine=dve, name="sel_gather_down_row"
    )
    # down-scale pair ((a & 12) << 2) | ((a >> 3) & 12) | (a & 3)
    nisa.tensor_scalar(
        dst=work[:, 0:1],
        data=a,
        op0=nl.bitwise_and,
        operand0=12,
        op1=nl.left_shift,
        operand1=2,
        engine=dve,
        name="sel_gather_scale_pair_hi",
    )
    nisa.tensor_scalar(
        dst=work[:, 1:2],
        data=a,
        op0=nl.right_shift,
        operand0=3,
        op1=nl.bitwise_and,
        operand1=12,
        engine=dve,
        name="sel_gather_scale_pair_mid",
    )
    nisa.scalar_tensor_tensor(
        dst=fields[:, 4:5],
        data=a,
        op0=nl.bitwise_and,
        operand0=3,
        op1=nl.bitwise_or,
        operand1=work[:, 0:1],
        name="sel_gather_scale_pair_lo",
    )
    nisa.tensor_tensor(
        dst=fields[:, 4:5],
        data1=fields[:, 4:5],
        data2=work[:, 1:2],
        op=nl.bitwise_or,
        engine=dve,
        name="sel_gather_scale_pair",
    )
    nisa.tensor_scalar(
        dst=fields[:, 5:6],
        data=a,
        op0=nl.bitwise_and,
        operand0=16,
        op1=nl.left_shift,
        operand1=16,
        engine=dve,
        name="sel_gather_scale_skip",
    )
    lanes_cols = (max(tokens for _, _, tokens, _ in runs) * _TOP_K // _GROUP_PAIRS - 1 + group0) * _GROUP_PAIRS
    for run, (first_token, n_run, tokens, gather_col) in enumerate(runs):
        tag = f"_r{run}" if run else ""
        pairs = tokens * _TOP_K
        groups = pairs // _GROUP_PAIRS
        gate_up_groups = groups - 1 + group0  # gathered gate/up groups (group 0 too with group0)
        gate_up_cols = gate_up_groups * _GROUP_PAIRS
        scale_tiles = _scale_tiles(tokens)
        chunk_gcols = _gather_cols(tokens, group0)
        cols = n_run * chunk_gcols
        # select[a, (c, col)] = 16f + pairs*c + the pair of column col; lanes[a, b] = 8b + a/16 (gate/up source row)
        select = alloc_heap((_PMAX, cols), nl.int32, name="sel_gather_select" + tag)
        if run == 0:
            lanes = alloc_heap((_PMAX, lanes_cols), nl.int32, name="sel_gather_lanes")
        experts = alloc_heap((_PMAX, cols), nl.int32, name="sel_gather_experts" + tag)
        if run == 0:
            nisa.iota(
                dst=lanes.ap(
                    pattern=[[lanes_cols, _PMAX], [_GROUP_PAIRS, lanes_cols // _GROUP_PAIRS], [1, _GROUP_PAIRS]],
                    offset=0,
                ),
                pattern=[[0, lanes_cols // _GROUP_PAIRS], [_DOWN_ROWS, _GROUP_PAIRS]],
                offset=0,
                name="sel_gather_iota_lanes",
            )
            nisa.tensor_tensor(
                dst=lanes,
                data1=lanes,
                data2=fields.ap(pattern=[[7, _PMAX], [0, lanes_cols]], offset=1),
                op=nl.add,
                engine=dve,
                name="sel_gather_lanes_add",
            )
        pair_col = first_token * _TOP_K
        scale_step = _SCALE_GROUPS * _GROUP_PAIRS if scale_tiles > 1 else 0  # pairs per down-scale tile
        # (first column, free dims of the columns, iota pattern over (chunk, dims), iota offset, fields column)
        for first, dims, pattern, offset, field in (
            (
                0,
                [[_GROUP_PAIRS, gate_up_groups], [1, _GROUP_PAIRS]],
                [[pairs, n_run], [_GROUP_PAIRS, gate_up_groups], [0, _GROUP_PAIRS]],
                pair_col + (0 if group0 else _GROUP_PAIRS),
                0,
            ),
            (gate_up_cols, [[1, groups]], [[pairs, n_run], [_GROUP_PAIRS, groups]], pair_col, 2),
            (gate_up_cols + groups, [[1, scale_tiles]], [[pairs, n_run], [scale_step, scale_tiles]], pair_col, 4),
        ):
            view = select.ap(pattern=[[cols, _PMAX], [chunk_gcols, n_run], *dims], offset=first)
            nisa.iota(dst=view, pattern=pattern, offset=offset, name=f"sel_gather_iota_select{first}{tag}")
            nisa.tensor_tensor(
                dst=view,
                data1=view,
                data2=fields.ap(pattern=[[7, _PMAX], [0, n_run], *[[0, n] for _, n in dims]], offset=field),
                op=nl.add,
                engine=dve,
                name=f"sel_gather_select{first}{tag}",
            )
        # A kept-short last call picks down-scale pairs up to 63 past its first pair; clamp those picks (their rows skip).
        last_pick = pair_col + pairs * (n_run - 1) + scale_step * (scale_tiles - 1) + _SCALE_GROUPS * _GROUP_PAIRS - 1
        if last_pick >= index_all.shape[1]:
            nisa.tensor_scalar(
                dst=view,
                data=view,
                op0=nl.minimum,
                operand0=index_all.shape[1] - 1,
                engine=dve,
                name=f"sel_gather_select_clamp{tag}",
            )
        nisa.nc_n_gather(dst=experts, data=index_all, indices=select.view(nl.uint32), name="sel_gather_pick" + tag)

        def chunk_cols(tile, first, n_cols, tile_cols=cols, base=0):
            return tile.ap(pattern=[[tile_cols, _PMAX], [chunk_gcols, n_run], [1, n_cols]], offset=base + first)

        nisa.scalar_tensor_tensor(
            dst=chunk_cols(gather_all, 0, gate_up_cols, total_cols, gather_col),
            data=chunk_cols(experts, 0, gate_up_cols),
            op0=nl.multiply,
            operand0=_PMAX,
            op1=nl.add,
            operand1=lanes.ap(pattern=[[lanes_cols, _PMAX], [0, n_run], [1, gate_up_cols]], offset=0),
            name="sel_gather_index_gate_up" + tag,
        )
        nisa.scalar_tensor_tensor(
            dst=chunk_cols(gather_all, gate_up_cols, groups, total_cols, gather_col),
            data=chunk_cols(experts, gate_up_cols, groups),
            op0=nl.multiply,
            operand0=down_stride,
            op1=nl.add,
            operand1=fields.ap(pattern=[[7, _PMAX], [0, n_run], [0, groups]], offset=3),
            name="sel_gather_index_down_w" + tag,
        )
        scale_skip = fields.ap(pattern=[[7, _PMAX], [0, n_run], [0, scale_tiles]], offset=5)
        if groups < _SCALE_GROUPS:
            # A short call has fewer groups than a scale tile: the rows of the other groups (a/4 % 4) skip too.
            nisa.tensor_scalar(
                dst=work[:, 0:1],
                data=a,
                op0=nl.right_shift,
                operand0=2,
                op1=nl.bitwise_and,
                operand1=_SCALE_GROUPS - 1,
                engine=dve,
                name="sel_gather_scale_group" + tag,
            )
            nisa.tensor_scalar(
                dst=work[:, 1:2],
                data=work[:, 0:1],
                op0=nl.greater_equal,
                operand0=groups,
                op1=nl.multiply,
                operand1=_GATHER_SKIP,
                engine=dve,
                name="sel_gather_scale_group_skip" + tag,
            )
            nisa.tensor_tensor(
                dst=work[:, 1:2],
                data1=work[:, 1:2],
                data2=fields[:, 5:6],
                op=nl.add,
                engine=dve,
                name="sel_gather_scale_skip_short" + tag,
            )
            scale_skip = work.ap(pattern=[[2, _PMAX], [0, n_run], [0, scale_tiles]], offset=1)
        done = nisa.tensor_tensor(
            dst=chunk_cols(gather_all, gate_up_cols + groups, scale_tiles, total_cols, gather_col),
            data1=chunk_cols(experts, gate_up_cols + groups, scale_tiles),
            data2=scale_skip,
            op=nl.add,
            engine=dve,
            name="sel_gather_index_down_s" + tag,
        )
    while len(sbm.heap) > depth:
        sbm.pop_heap()
    return done


_INDEX_ONEHOT_SHAPE = (_TOP_K, (2 + _SCALE_GROUPS) * _PMAX)
"""The one-hot stationary of ``_gather_indices_matmul`` [16, 6 * 128] fp16 (``_index_onehot``)."""


def _index_onehot(nisa, onehot: nl.NkiTensor):
    """Write the routing-independent one-hot stationary of ``_gather_indices_matmul``: onehot[k, (kind, a)] = 1 if
    k == sigma(a), kinds: gate/up, down weights, the down scales of token r = 0..3 of a tile. The iotas write
    sigma(a) - k (exact in fp16): for the down scales a = 32q + 16h + 4j + l holds slot 4q + l of token j, and the step
    `far` keeps every j != r away from zero. ``nisa`` is the (suffixed) ISA of the caller."""
    far = _TOP_K * _SCALE_GROUPS
    patterns = [
        ([[0, _PMAX // _GROUP_PAIRS], [1, _GROUP_PAIRS]], 0),  # a % 16
        ([[1, _PMAX // _DOWN_ROWS], [0, _DOWN_ROWS]], 0),  # a / 8
    ]
    patterns += [([[4, 4], [0, 2], [far, 4], [1, 4]], -far * r) for r in range(_SCALE_GROUPS)]  # (q, h, j, l)
    for kind, (pattern, offset) in enumerate(patterns):
        nisa.iota(
            dst=onehot[:, kind * _PMAX : (kind + 1) * _PMAX],
            pattern=pattern,
            offset=offset,
            channel_multiplier=-1,
            name=f"sel_index_iota_onehot{kind}",
        )
    nisa.tensor_scalar(
        dst=onehot,
        data=onehot,
        op0=_LANGUAGE.equal,
        operand0=0.0,
        engine=nisa.vector_engine,
        name="sel_index_onehot_mask",
    )


def _gather_indices_matmul(
    sbm: SbufManager,
    expert_index: nl.NkiTensor,
    gather_all: nl.NkiTensor,
    runs: list[tuple[int, int, int, int]],
    padded: int,
    down_rows: int,
    suffix: str,
    down_stride: int,
    expert_offset: int,
    max_index: int,
    onehot: nl.NkiTensor = None,
) -> tuple[nisa.NkiInstruction, nisa.NkiInstruction]:
    """Build gather_all of ``_gather_indices`` (group0) from the SBUF expert index [T, 16] with one-hot matmuls.

    On partition a, each gather index column takes the expert of one slot sigma(a) of one token: gate/up a % 16 and
    down weights a / 8 of the token of the column, down scales 4(a / 32) + a % 4 of token 4s + (a / 4) % 4 of tile s
    (``_gather_indices``). The cast clamps the ids into fp16 [T, 16] (exact up to 2047), a Vector engine transpose
    moves them to idx_t [16, T] (slot k on partition k), and a TensorE matmul with the one-hot stationary
    [k == sigma(a)] [16, 128] gives experts[a, t] = id(t, sigma(a)) in PSUM: one term 1.0 * id and exact zeros. The
    down scales take 4 accumulating matmuls, one for each token r of a tile on the columns 4s + r, whose stationary
    keeps the partitions with (a / 4) % 4 == r. The index ops read PSUM and add the expert offset with their
    per-partition terms: gate/up 128 * id + (lane + 128 * offset), down weights stride * id + (row + stride * offset),
    down scales id + (offset + skips), the integers of ``_gather_indices``. In a short call (1 or 2 tokens) the
    down-scale rows of the missing tokens of the tile get expert 0 (idx_t is 0 past T) and the skip, so the gather
    always skips them; ``_gather_indices`` picks their ids from outside the index tile. The gate/up indices of chunk 0
    come first. Pool runs only the iotas of the routing-independent setup. The tiles stay on the heap for the caller
    to pop.

    Args:
        expert_index: int32 [T, 16] SBUF, the routed experts of each token on its partition.
        gather_all, runs, down_rows, suffix, down_stride: as in ``_gather_indices``.
        padded: the tokens of the calls (at most 128); the padding tokens get expert 0.
        expert_offset: the weight row of expert 0.
        max_index: the largest expert id of the clamp (at most 2047 is kept).
        onehot: the one-hot stationary of ``_index_onehot`` (``moe_sel_load_tp_start`` index_onehot_resident), or
            None to build it on the heap here.

    Returns:
        (first, last): the gate/up index op of chunk 0 and the last instruction that writes gather_all.
    """
    nisa, nl = _Suffixed(_ISA, suffix), _Suffixed(_LANGUAGE, suffix, functions=("ndarray",))

    def alloc_heap(shape, dtype, name):
        return sbm.alloc_heap(shape, dtype, name=name + suffix)

    dve = nisa.vector_engine
    T = expert_index.shape[0]
    total_cols = gather_all.shape[1]
    n_tiles = div_ceil(padded, _SCALE_GROUPS)  # down-scale tiles of 4 tokens
    token_cols = n_tiles * _SCALE_GROUPS
    # The gate/up, down-weight and down-scale regions of experts_ps start _PMAX columns apart in one PSUM bank.
    kernel_assert(
        token_cols <= _PMAX,
        f"index_matmul supports at most {_PMAX} tokens (its PSUM index regions are {_PMAX} columns apart), got "
        f"T={T} ({token_cols} token columns)",
    )
    n_terms = 5

    # The routing-independent setup: the one-hot stationary (or the resident one) and the terms.
    if onehot is None:
        onehot = alloc_heap(_INDEX_ONEHOT_SHAPE, nl.float16, name="sel_index_onehot")
        _index_onehot(nisa, onehot)
    # terms[a, :] = a, a / 16, the down row (+ skip) + stride * offset, the down-scale skip + offset, and that skip
    # with the token skip of a short call (fewer than 4 groups).
    terms = alloc_heap((_PMAX, n_terms), nl.int32, name="sel_index_terms")
    work = alloc_heap((_PMAX, 2), nl.int32, name="sel_index_work")
    nisa.iota(dst=terms[:, 0:1], pattern=[[0, 1]], offset=0, channel_multiplier=1, name="sel_index_partition")
    a = terms[:, 0:1]
    nisa.tensor_scalar(dst=terms[:, 1:2], data=a, op0=nl.right_shift, operand0=4, engine=dve, name="sel_index_lane")
    nisa.tensor_scalar(
        dst=work[:, 0:1], data=a, op0=nl.bitwise_and, operand0=_DOWN_ROWS - 1, engine=dve, name="sel_index_row"
    )
    nisa.tensor_scalar(
        dst=work[:, 1:2],
        data=work[:, 0:1],
        op0=nl.greater_equal,
        operand0=down_rows,
        op1=nl.multiply,
        operand1=_GATHER_SKIP,
        engine=dve,
        name="sel_index_row_skip",
    )
    nisa.tensor_tensor(
        dst=terms[:, 2:3], data1=work[:, 0:1], data2=work[:, 1:2], op=nl.add, engine=dve, name="sel_index_down_row"
    )
    nisa.tensor_scalar(
        dst=terms[:, 3:4],
        data=a,
        op0=nl.bitwise_and,
        operand0=16,
        op1=nl.left_shift,
        operand1=16,
        engine=dve,
        name="sel_index_scale_skip",
    )
    if expert_offset:
        for col, offset in ((2, down_stride * expert_offset), (3, expert_offset)):
            nisa.tensor_scalar(
                dst=terms[:, col : col + 1],
                data=terms[:, col : col + 1],
                op0=nl.add,
                operand0=offset,
                engine=dve,
                name=f"sel_index_offset{col}",
            )
    short_groups = [tokens for _, _, tokens, _ in runs if tokens < _SCALE_GROUPS]
    if short_groups:
        # A short call has fewer groups than a scale tile: the rows of the other groups (a/4 % 4) skip too.
        nisa.tensor_scalar(
            dst=work[:, 0:1],
            data=a,
            op0=nl.right_shift,
            operand0=2,
            op1=nl.bitwise_and,
            operand1=_SCALE_GROUPS - 1,
            engine=dve,
            name="sel_index_scale_group",
        )
        nisa.tensor_scalar(
            dst=work[:, 1:2],
            data=work[:, 0:1],
            op0=nl.greater_equal,
            operand0=short_groups[0],
            op1=nl.multiply,
            operand1=_GATHER_SKIP,
            engine=dve,
            name="sel_index_scale_group_skip",
        )
        nisa.tensor_tensor(
            dst=terms[:, 4:5],
            data1=work[:, 1:2],
            data2=terms[:, 3:4],
            op=nl.add,
            engine=dve,
            name="sel_index_scale_skip_short",
        )
    # lanes[a, (j, b)] = 8b + a/16 + 128 * offset: the gate/up source row of index column b, with the expert offset
    lanes_cols = max(tokens for _, _, tokens, _ in runs) * _TOP_K
    lanes = alloc_heap((_PMAX, lanes_cols), nl.int32, name="sel_index_lanes")
    nisa.iota(
        dst=lanes.ap(
            pattern=[[lanes_cols, _PMAX], [_GROUP_PAIRS, lanes_cols // _GROUP_PAIRS], [1, _GROUP_PAIRS]], offset=0
        ),
        pattern=[[0, lanes_cols // _GROUP_PAIRS], [_DOWN_ROWS, _GROUP_PAIRS]],
        offset=_PMAX * expert_offset,
        name="sel_index_iota_lanes",
    )
    nisa.tensor_tensor(
        dst=lanes,
        data1=lanes,
        data2=terms.ap(pattern=[[n_terms, _PMAX], [0, lanes_cols]], offset=1),
        op=nl.add,
        engine=dve,
        name="sel_index_lanes_add",
    )
    idx_t = alloc_heap((_TOP_K, token_cols), nl.float16, name="sel_index_t")
    if token_cols > T:
        nisa.memset(dst=idx_t[:, T:token_cols], value=0.0, engine=dve, name="sel_index_t_pad")
    # experts_ps[a, region + t]: gate/up at column 0, down weights at 128, down-scale tile s at 256 + s. The bank of
    # the index broadcast, which this form does not run.
    regions = (0, _PMAX, 2 * _PMAX)
    experts_ps = nl.ndarray(
        (_PMAX, _BANK_FP32),
        dtype=nl.float32,
        buffer=nl.psum,
        address=(0, _ROUTING_PSUM_BANK * _PSUM_BANK_BYTES),
        name="sel_index_experts_ps",
    )

    # The chain from the routing to the first gather index: cast, transpose, matmul, gate/up index op of chunk 0.
    index_values = alloc_heap((T, _TOP_K), nl.float16, name="sel_index_values")
    _cast_index(nisa, index_values, expert_index, min(max_index, _FP16_EXACT_ID), "sel_cast_index_values")
    for block in range(div_ceil(T, _VECTOR_TRANSPOSE_ROWS)):
        rows = slice(block * _VECTOR_TRANSPOSE_ROWS, min(T, (block + 1) * _VECTOR_TRANSPOSE_ROWS))
        nisa.nc_transpose(
            dst=idx_t[:, rows], data=index_values[rows, :], engine=dve, name=f"sel_index_transpose{block}"
        )
    for kind in range(2):
        nisa.nc_matmul(
            dst=experts_ps[:, regions[kind] : regions[kind] + token_cols],
            stationary=onehot[:, kind * _PMAX : (kind + 1) * _PMAX],
            moving=idx_t,
            accumulate=False,
            name=("sel_mm_index_gate_up", "sel_mm_index_down_w")[kind],
        )
    for r in range(_SCALE_GROUPS):
        nisa.nc_matmul(
            dst=experts_ps[:, regions[2] : regions[2] + n_tiles],
            stationary=onehot[:, (2 + r) * _PMAX : (3 + r) * _PMAX],
            # [k, s]: slot k of token 4s + r
            moving=idx_t.ap(pattern=[[token_cols, _TOP_K], [_SCALE_GROUPS, n_tiles]], offset=r),
            accumulate=r > 0,
            name=f"sel_mm_index_scale{r}",
        )

    def gate_up(first_token, chunk, tokens, gather_col, tag):
        """The gate/up columns (j, b) of one chunk of a run: 128 * expert(token j, slot a % 16) + lanes (one chunk per
        op: the destination of an op has at most 2 free dimensions)."""
        return nisa.scalar_tensor_tensor(
            dst=gather_all.ap(
                pattern=[[total_cols, _PMAX], [_GROUP_PAIRS, tokens], [1, _GROUP_PAIRS]],
                offset=gather_col + chunk * _gather_cols(tokens, True),
            ),
            data=experts_ps.ap(
                pattern=[[_BANK_FP32, _PMAX], [1, tokens], [0, _GROUP_PAIRS]],
                offset=regions[0] + first_token + chunk * tokens,
            ),
            op0=nl.multiply,
            operand0=_PMAX,
            op1=nl.add,
            operand1=lanes.ap(pattern=[[lanes_cols, _PMAX], [_GROUP_PAIRS, tokens], [1, _GROUP_PAIRS]], offset=0),
            name="sel_gather_index_gate_up" + tag + (f"_k{chunk}" if chunk else ""),
        )

    first_token, n_run, tokens, gather_col = runs[0]
    first = last = gate_up(first_token, 0, tokens, gather_col, "")
    # Then the down scales and down weights of each run (the Pool stream of a chunk reads them after its first two
    # gate/up gathers), and the gate/up columns of the later chunks.
    for run, (first_token, n_run, tokens, gather_col) in enumerate(runs):
        tag = f"_r{run}" if run else ""
        groups = tokens * _TOP_K // _GROUP_PAIRS
        scale_tiles = _scale_tiles(tokens)
        chunk_gcols = _gather_cols(tokens, True)

        def chunk_cols(first_col, n_cols):
            return gather_all.ap(
                pattern=[[total_cols, _PMAX], [chunk_gcols, n_run], [1, n_cols]], offset=gather_col + first_col
            )

        nisa.tensor_tensor(
            dst=chunk_cols(groups * _GROUP_PAIRS + groups, scale_tiles),
            # [a, c, s]: down-scale tile s of chunk c (tile first_token / 4 + scale_tiles * c + s of experts_ps)
            data1=experts_ps.ap(
                pattern=[[_BANK_FP32, _PMAX], [scale_tiles, n_run], [1, scale_tiles]],
                offset=regions[2] + first_token // _SCALE_GROUPS,
            ),
            data2=terms.ap(
                pattern=[[n_terms, _PMAX], [0, n_run], [0, scale_tiles]], offset=4 if groups < _SCALE_GROUPS else 3
            ),
            op=nl.add,
            engine=dve,
            name="sel_gather_index_down_s" + tag,
        )
        last = nisa.scalar_tensor_tensor(
            dst=chunk_cols(groups * _GROUP_PAIRS, groups),
            data=experts_ps.ap(
                pattern=[[_BANK_FP32, _PMAX], [tokens, n_run], [1, groups]], offset=regions[1] + first_token
            ),
            op0=nl.multiply,
            operand0=down_stride,
            op1=nl.add,
            operand1=terms.ap(pattern=[[n_terms, _PMAX], [0, n_run], [0, groups]], offset=2),
            name="sel_gather_index_down_w" + tag,
        )
        for chunk in range(0 if run else 1, n_run):
            last = gate_up(first_token, chunk, tokens, gather_col, tag)
    return first, last
