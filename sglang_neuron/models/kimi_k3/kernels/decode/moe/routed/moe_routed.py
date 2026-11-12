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

"""Kimi-K3 routed-expert decode with serving-layout MXFP4 weights."""

import math
from contextlib import contextmanager

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np

from ..projection_mx_constants import (
    GATE_FUSED_IDX,
    MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM,
    UP_FUSED_IDX,
    _q_width,
)
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import div_ceil
from ...utils.trace_hazards import REPEATED_TWO_BUFFER_EXPERT_CALLS, record_hazard

_PMAX = 128
_MX_SCALE_PARTITIONS = 16
# FP32 columns of one 2 KiB PSUM bank. The matmuls accumulate in FP32, and a
# matmul destination must stay within one bank, so this is the widest gate/up
# and down matmul output.
_PSUM_BANK_FP32 = 512
_DOWN_TILE_H = _PSUM_BANK_FP32
_GATE_UP_CHUNK_I = _PSUM_BANK_FP32
# SiTU output columns transposed, evicted and quantized together: two I/512 tiles.
_TRANSPOSE_CHUNK_I = 1024
# Down H/512 tiles accumulated at once: one FP32 PSUM bank each. The rest of
# PSUM holds the activation transposes and the next projection's chunk, and a
# fused caller's tiles. Every group runs the same matmuls, so a group costs
# nothing beyond its own PSUM evictions.
_DOWN_TILES_PER_GROUP = 4
# Output columns converted to BF16 and stored per DMA.
_OUTPUT_TILE_H = 1024
_PSUM_BANK_BYTES = 2048
# trn3 SBUF ECC block: a partial-block write is a read-modify-write of the whole block.
_SBUF_BLOCK_BYTES = 32
_SBUF_QUADRANT_PARTITIONS = 32
# MX scale rows per quadrant: 32 partitions hold 4 rows of 8-partition groups.
_SCALE_ROWS_PER_QUADRANT = 4
_SCALE_ROWS_ACROSS_QUADRANTS = False
"""The form of the static MX scale loads (``_load_scale_rows``) of the trace, set by ``scale_rows_across_quadrants``."""
SCALE_ROWS_QUEUES = ("static", "sync_hw", "act_hw")
"""The DMA queues of the static MX scale loads (``scale_rows_queue``): ``static`` (default) the static DMA queue of the
expert weights; ``sync_hw`` / ``act_hw`` hardware DGE from the Sync / Activation engine, a queue of their own."""
_SCALE_ROWS_QUEUE = "static"
"""The queue of the static MX scale loads (``_load_scale_rows``) of the trace, set by ``scale_rows_queue``."""
# Gate, up and down; the down projection uses the slot after the fused gate/up.
_PROJECTIONS_PER_EXPERT = 3
_DOWN_IDX = 2
# Whole weight buffers of the cross-expert order and of the static loop with half buffers or the half-major layout:
# up to 5 (the dynamic loop takes 3, the active-expert slots at most 3).
_STATIC_MAX_WEIGHT_BUFFERS = 5
# Scale start offsets the matmul accepts within a quadrant: 0, 4, 8 and 12.
_SCALE_SLOTS_PER_QUADRANT = 4
_GATE_UP_SCALE_BUFFERS = 2
# Whole weight buffers of the static loop: up to two experts' projections ahead (``static_expert_stream``).
_MAX_STATIC_WEIGHT_BUFFERS = 2 * _PROJECTIONS_PER_EXPERT
EXPERT_QUEUES = (1, 2)
"""The DMA queues of the static loop's weight loads (``expert_queues``)."""
LAST_EXPERT_ORDERS = ("default", "chunked", "up_first")
"""The TensorE orders of the last experts of the cross-expert order with ``transpose_block`` (``last_expert_order``):
``default`` keeps the program; ``chunked`` issues the last expert's activation transposes chunk by chunk; ``up_first``
also runs up(E_L - 1) and those transposes before down(E_L - 2) (``_last_two_up_first``)."""
# Activation chunks transposed before the down matmuls start.
_TRANSPOSE_LOOKAHEAD_CHUNKS = 2
# Gate/up chunks of the dynamic loop whose gate terms are live at once: the
# gate and up matmuls run in groups of this many chunks, so that at T = 4 the
# FP32 gate terms fit beside the expert buffers in the first 32 partitions.
_DYNAMIC_GATE_UP_GROUP_CHUNKS = 3
_TWO_WEIGHT_BUFFERS = 2
# Static-loop calls with two whole weight buffers in the open layer scope; None when no scope is open.
_LAYER_SCOPE = {"two_buffer_calls": None}
# The values of moe_tkg_decode's static_alt_tiles, and the roles each key-on value takes from two buffers
# (``_StaticLoopTiles``): True all of them, "act" the gate sigmoid / up tanh ring, "act_copy" that ring and the
# down PSUM copies. "ring_act" / "ring_act_eviction" (``STATIC_ALT_TILE_PSUM_CONSUME``, moe_tkg_decode only) take the
# gate sigmoid / up tanh ring (and the eviction ring) with ``psum_consume`` and ``resident_out`` kept: no PSUM copies,
# so no copy ring.
STATIC_ALT_TILE_FORMS = (False, True, "act", "act_copy")
_STATIC_ALT_TILE_ROLES = {
    True: ("gate_up", "expert_out", "eviction"),
    "act": ("gate_up",),
    "act_copy": ("gate_up", "expert_out"),
    "ring_act": ("gate_up",),
    "ring_act_eviction": ("gate_up", "eviction"),
    "ring4_act_eviction": ("gate_up", "eviction"),
    "ring6_act_eviction": ("gate_up", "eviction"),
}
# The buffers of the gate sigmoid / up tanh ring of a static_alt_tiles value (2 when absent).
_STATIC_ALT_TILE_GATE_UP_DEPTH = {"ring4_act_eviction": 4, "ring6_act_eviction": 6}
STATIC_ALT_TILE_PSUM_CONSUME = ("ring_act", "ring_act_eviction", "ring4_act_eviction", "ring6_act_eviction")
"""The ``static_alt_tiles`` values of ``moe_tkg_decode`` beyond ``STATIC_ALT_TILE_FORMS`` that keep ``psum_consume``
and ``resident_out`` (the other key-on values drop both): their rings hold no PSUM copy. "ring4_act_eviction" /
"ring6_act_eviction" take 4 / 6 buffers for the gate sigmoid / up tanh ring (``_STATIC_ALT_TILE_GATE_UP_DEPTH``)."""


@contextmanager
def expert_loop_layer():
    """Count the static-loop calls of ``moe_tkg_decode`` with two whole weight buffers in one layer, at trace time.

    Open the scope around all the ``moe_tkg_decode`` calls of one layer. Each static-loop call with two whole weight
    buffers (``weight_buffers=2`` without ``half_weight_buffers``) adds one to the count of the scope, and the largest
    count goes to the trace record ``REPEATED_TWO_BUFFER_EXPERT_CALLS`` (``utils.trace_hazards``). Each such call
    loads the same expert weights again. With the compiler's own SBUF allocation, the load merging of its
    ``dma_optimization_sb`` pass gave wrong routed partials from the third such call of a layer on trn3, so the code
    that compiles the trace reads the record and decides. A nested scope adds to the outer scope. Outside a scope,
    no call is counted. The scope adds no instruction.

    Example:
        >>> with expert_loop_layer():
        ...     partials = [moe_tkg_decode(*chunk_args) for chunk_args in chunks]
    """
    outer = _LAYER_SCOPE["two_buffer_calls"] is not None
    if not outer:
        _LAYER_SCOPE["two_buffer_calls"] = 0
    try:
        yield
    finally:
        if not outer:
            _LAYER_SCOPE["two_buffer_calls"] = None


def _count_static_call(weight_buffers: int, half_weight_buffers: int):
    """Count one static-loop call in the open layer scope (``expert_loop_layer``) and record the count of the layer."""
    calls = _LAYER_SCOPE["two_buffer_calls"]
    if calls is None or weight_buffers != _TWO_WEIGHT_BUFFERS or half_weight_buffers:
        return
    _LAYER_SCOPE["two_buffer_calls"] = calls + 1
    record_hazard(REPEATED_TWO_BUFFER_EXPERT_CALLS, calls + 1)


@nki.jit
def moe_tkg_decode(
    latent: nl.NkiTensor,
    local_affinities: nl.NkiTensor,
    expert_gate_up_weights: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_gate_up_weights_scale: nl.NkiTensor,
    expert_down_weights_scale: nl.NkiTensor,
    beta: float = 4.0,
    linear_beta: float = 25.0,
    weight_buffers: int = _PROJECTIONS_PER_EXPERT,
    scale_buffers: int = _GATE_UP_SCALE_BUFFERS,
    early_loads: bool = True,
    is_all_expert_dynamic: bool = False,
    n_static_experts: int = 0,
    active_expert_slots: int = 0,
    active_expert_count: nl.NkiTensor = None,
    half_weight_buffers: int = 0,
    gate_up_half_major: bool = False,
    output: nl.NkiTensor = None,
    column_groups_out: nl.NkiTensor = None,
    direct_latent: bool = False,
    transpose_lookahead: int = _TRANSPOSE_LOOKAHEAD_CHUNKS,
    cross_expert_order: bool = False,
    quantized_latent: tuple = None,
    shared_scale_buffer: bool = False,
    extra_weight_buffers: tuple = (),
    extra_buffers_after=None,
    static_alt_tiles: bool | str = False,
    cross_expert_transpose_blocks: bool = False,
    stream: "_StaticExpertStream" = None,
    expert_queues: int = 1,
    preamble_buffers: int = 0,
    preamble_reuse: bool = False,
    transpose_block: bool = False,
    psum_consume: bool = False,
    accum_memset: tuple = None,
    first_add_zero: bool = False,
    up_product_split: bool = False,
    resident_out: dict = None,
    last_expert_order: str = "default",
    affinity_sbuf_copy: bool = False,
) -> nl.NkiTensor:
    """Compute Kimi-K3's unnormalized local routed-expert decode contribution.

    This LNC1 Trn3 kernel consumes the model's natural latent layout and serving
    MXFP4 weights directly. The gate/up and down matmuls accumulate in FP32
    PSUM, as the GPU GEMMs do. The two-sided SiTU activation, affinity
    scaling, and cross-expert accumulation use FP32. The SiTU output is
    rounded to FP16, the widest ``quantize_mx`` input, before its MXFP8
    quantization.

    Three expert paths compute bit-identical results. The choice is made at
    trace time, like the static and dynamic all-expert algorithms of
    ``moe_tkg`` (the C128 and C512 calls of the gpt_oss kernels). A call
    takes one path: the dynamic loop with ``is_all_expert_dynamic``, else
    the slots with ``active_expert_slots`` > 0, else the static loop; asking
    for both the dynamic loop and the slots fails:

    - static (``is_all_expert_dynamic=False``, the default): every local
      expert streams through static DMAs that overlap with the matmuls of the
      previous expert. The cost does not depend on the routing (on
      trn3 for T = 4-128).
    - dynamic (``is_all_expert_dynamic=True``): only the active experts run,
      those with a nonzero affinity for at least one token. The sorted
      active ids are compacted on the device, and a dynamic loop runs one
      iteration per active expert, with the expert id as a runtime DMA
      offset. A zero-affinity expert would add exactly zero. The cost is
      a fixed part for the first active expert and a part for each
      further one, so this loop wins only when few experts are active.
      Its first ``n_static_experts`` slots run before the loop, unrolled
      and without loop barriers. A slot past the active count skips its
      DMAs, but its matmuls still run, on stale buffers, and add nothing.
    - slots (``active_expert_slots`` = K > 0): the first K active experts
      run in K straight-line slots, with no loop, in ascending order, with
      the expert id as a runtime DMA offset. If more than K are active, a
      one-trip loop runs every expert and overwrites the output. The cost is
      a fixed part, a part for each active slot and a smaller part for
      each idle one, whatever the routing. The slots compile beside the MLA
      loop path. They take ``weight_buffers`` but keep their own buffer
      order (projection p in buffer p % weight_buffers) and their own
      scale buffer, as does their overflow loop.

    Each dynamic-loop iteration ends with a drain and an all-engine barrier.
    When the kernel is inlined into a larger kernel, the barrier also waits
    for the caller's fused work in flight, so a fused caller may prefer the
    static loop. The dynamic loop also holds more SBUF (3 weight buffers and
    the scales of a whole expert): in the Kimi-K3 decode giga-kernel it
    compiles only beside the unrolled MLA table. With the MLA loop path,
    neuronx-cc stops with NCC_EGCA111, because spilling is disabled for
    kernels. The giga-kernel therefore selects it only at a small T off the
    MLA loop path (``tkg_moe.dynamic_expert_loop``).

    With top-16-of-896 routing over EP128, the busiest of the 128 ranks sees
    on average 2.8 active local experts at T=4, 3.7 at T=8, 5.0 at T=16, 6.3
    at T=32 and 7 from T=64 up.

    Args:
        latent: BF16 ``[T, D]`` latent activations, where ``1 <= T <= 128``.
        local_affinities: BF16 or FP32 ``[T, E_L]`` local affinities, in HBM
            or in SBUF (token t on partition t). Entries outside the selected
            local top-k must be zero.
        expert_gate_up_weights: Packed MXFP4
            ``[E_L, 128, 2, D/512, I]`` gate/up weights.
        expert_down_weights: Packed MXFP4
            ``[E_L, 128, I/512, D]`` down weights.
        expert_gate_up_weights_scale: UINT8 E8M0 scales
            ``[E_L, 16, 2, D/512, I]``.
        expert_down_weights_scale: UINT8 E8M0 scales
            ``[E_L, 16, I/512, D]``.
        beta: SiTU gate-branch soft-clipping factor.
        linear_beta: SiTU up-branch soft-clipping factor.
        weight_buffers: SBUF weight buffers of the static loop, 1 to 5
            (up to 6 whole buffers in the static loop without half buffers or
            the half-major layout: the next expert's projections then load
            ahead). Projection p (gate, up, down) of expert e uses buffer
            (3e + p) % weight_buffers, so with 2 each load waits for the
            matmuls of the projection two steps before it, with 1 the
            three loads run one after the other, and with 4 or 5 each load
            waits for the matmuls of the projection 4 or 5 steps before it.
            The result does not change, and each buffer holds D*I/512
            weights for each partition (43 KB at the Kimi-K3 sizes, 21 KB at
            I = 1536). The active-expert slots take 1 to 3 buffers in their
            own order (``active_expert_slots``), the dynamic loop 3. In a
            layer scope (``expert_loop_layer``), the static-loop calls with 2
            whole buffers are counted into a trace record.
        scale_buffers: SBUF buffers of the gate/up scales in the static loop,
            2 or 1. Each buffer holds the scales of 2 experts. With 1, the
            scale load of the next 2 experts waits for the up matmuls of the
            previous expert. The result does not change, and the loop holds
            21 KB less SBUF for each partition at the Kimi-K3 sizes.
        early_loads: True (default) allocates the expert loop's buffers and
            issues the first load of each one before the latent is quantized
            (see _StaticExpertStream). False allocates them after the latent's
            work tiles and issues each first load where the first expert uses
            it, so a first load can wait for the bytes of a work tile. In a
            fused caller the first loads then start after the latent work, not
            during a collective that the caller runs before the call. The
            result does not change. The static loop, and with
            ``active_expert_slots`` the order of the slots' buffers.
        is_all_expert_dynamic: If True, run only the active experts in a
            dynamic loop. If False (default), run every local expert with
            static DMAs. On trn3 the busiest rank's expected latency is lower
            with the dynamic loop at T=4 and T=8, and higher at T=16.
            The dynamic loop takes 3 weight buffers and 2 scale buffers.
        n_static_experts: Number of active-expert slots, from 0 to E_L, that
            run before the dynamic loop. Only valid with the dynamic loop.
            The default is 0: on trn3 a static slot costs about as much as a
            loop iteration when its expert is active, and less when it
            is not, so no slot count beat 0 at any T from 4 to 32.
        active_expert_slots: 0 (default) runs the static loop or, with
            ``is_all_expert_dynamic``, the dynamic loop. K in 1 to E_L runs
            only the active experts (any nonzero affinity) in K straight-line
            slots, in ascending order, with the expert id as a runtime DMA
            offset. When more than K are active, a one-trip loop runs every
            expert and overwrites the output. The result has the bits of the
            static loop either way. The slots cost about K experts whatever
            the routing, so pick K for the busiest EP rank at the call's T:
            under uniform top-16 routing it has at most 2 active experts in
            98% of the calls at T = 1, and at most 3, 4, 5 and 6 in 99.3% or
            more at T = 2, 4, 8 and 16. The loop's block boundary drains the
            in-flight DMAs of a fused caller. K > 0 needs T <= 16, and with
            ``is_all_expert_dynamic`` (and so with ``n_static_experts``) it
            fails.
        active_expert_count: debug, with ``active_expert_slots`` only: an
            INT32 ``[1, 1]`` HBM tensor that gets the number of active local
            experts of the call (0 to E_L). More than K means that the
            one-trip loop over every expert ran. None (default) adds no
            instruction; a tensor without the slots fails.
        half_weight_buffers: static loop only. 0 (default) streams whole
            projections through ``weight_buffers`` buffers. N in 2 to 6
            streams half projections through N buffers of half the size
            in place of them: the I/512 chunks of gate and up, and the I/512
            tiles of down, split into two halves. Half h of projection p of
            expert e uses buffer (6e + 2p + h) % N, so a load waits only for
            the matmuls of the half N loads before it, and a projection's
            matmuls start when its first half lands. 4 holds the SBUF of 2
            whole buffers (86 KB for each partition at the Kimi-K3 sizes).
        gate_up_half_major: static loop only. False (default) reads gate/up
            weights stored ``[E_L, 128, 2, D/512, I]``. True reads the same
            shape stored half-major: in each partition row of a projection,
            the I columns ``[0, I/2)`` of every D/512 row come first, then
            ``[I/2, I)`` (``[2, D/512, I/2]``). A half then is one contiguous
            run per partition, so a half load needs one DMA descriptor per
            partition in place of D/512. The SBUF tiles and the result do not
            change.
            The matmuls and their PSUM order do not change, so the result
            does not change.
        output: optional BF16 ``[T, D]`` HBM tensor that receives the result
            (the static loop and the dynamic loop). None (default) returns a
            new shared HBM tensor.
        column_groups_out: static loop only, without ``output``. None (default) returns the
            ``[T, D]`` output in HBM. A BF16 SBUF ``[128, T, D/128]`` tile
            gets the output in the column-group layout instead: element
            ``(g, t, c)`` is output column ``g * D/128 + c`` of token ``t``,
            the layout of the Kimi-K3 routed all-reduce source. The last
            expert writes one SBUF ``[T, D]`` tile, with no HBM store, and
            D/256 TensorE transposes of 32-bit words (two adjacent columns of
            one group) and copies on the Vector and Scalar engines move its
            bytes into the tile, which the call returns. The values do not
            change. Needs ``D`` a multiple of 256, and whole 32-byte SBUF
            blocks per row of the tile (T a multiple of 4 at D = 3584).
        stream: a stream of ``static_expert_stream`` with the same weights,
            buffers and order (``use_order`` equal to ``cross_expert_order``),
            whose first loads the caller issued earlier (static loop,
            ``early_loads``). The call takes its buffers in place of its own
            stream start. The result does not change.
        expert_queues: static loop only (``EXPERT_QUEUES``). 1 (default)
            loads every weight on the loop's one in-order static DMA queue.
            2 loads the weights of the odd experts with hardware DGE from the
            Sync engine, a second DMA queue, so the transfers of two
            consecutive experts can run at the same time. The scales stay on
            the first queue. The values do not change.
        preamble_buffers: static loop with whole weight buffers and
            ``early_loads`` only. 0 (default) keeps the ring. K in 1 to 3 gives
            the first K projections of the call (gate 0, up 0, down 0, in use
            order) K extra whole buffers that live only until the last matmul
            of their projection; the ring of ``weight_buffers`` starts at
            projection K. The stream start then issues the loads of K more
            projections. In the default ring the first load into a reused
            buffer waits for the matmuls of its previous projection, which
            wait for the latent, so with 2 buffers down 0 streams only after
            the gate 0 matmuls; with K = 1 it streams during the work before
            the first matmul. Each extra buffer adds 43 KB per partition at the
            Kimi-K3 sizes until its projection's last matmul. The matmuls and
            their order do not change, so the result does not change.
        preamble_reuse: with ``preamble_buffers`` K only. False (default)
            frees each extra buffer after its projection. True gives extra
            buffer k one more projection, K + ``weight_buffers`` + k in use
            order (gate 1 for K = 1 with 2 ring buffers), and the ring then
            starts again at projection 2K + ``weight_buffers``. So the load of
            that projection waits for the matmuls of the first projection,
            the first to end, in place of the next ring buffer's; the extra
            buffer then lives until the matmuls of its second projection end.
            The result does not change.
        direct_latent: static loop only. False (default) copies the latent
            into an SBUF tile of the kernel with one DMA (from HBM or SBUF).
            True reads an SBUF latent in place: the TensorE transposes that
            swizzle it for its MXFP8 quantization read the caller's tile, and
            the copy DMA is not issued. The transposes read the same bytes, so
            the result does not change. Needs ``latent`` a whole SBUF
            ``[T, D]`` tile (or a reshape of one: view offset 0, rows of D
            elements) with T a multiple of 4, so that no padding row is
            needed. The caller must not write the tile during the call; the
            last readers of the tile are those transposes.
        transpose_lookahead: static loop only. The 1024-column chunks of
            each expert's SiTU output that are transposed, evicted and
            quantized before its first down matmul: 2 (default) or 1. Each
            later chunk is transposed after the matmuls of the chunk before
            it. At I = 1536 (two chunks) the default puts the whole
            activation, including the SiTU of the last up chunk, before the
            first down matmul. With 1 that matmul waits only for chunk 0,
            whose SiTU products end before the up projection does. The
            transposes of each later chunk then follow the first down matmul
            of the chunk before it on TensorE (a schedule edge), so that they
            do not interleave with chunk 0's. The matmuls, their PSUM order
            and the result do not change.
        cross_expert_order: static loop only. False (default) runs each
            expert's gate, up and down matmuls in turn, so TensorE waits
            between an expert's up and down matmuls while the SiTU output is
            transposed, evicted and quantized. True runs the gate matmuls of
            the next expert in that wait: up(e), the transposes of activation
            chunk 0, gate(e + 1) chunk 0, the transposes of chunk 1, the other
            gate(e + 1) chunks, down(e), up(e + 1). Schedule edges keep the
            order on TensorE and ScalarE, and each H tile's PSUM copy names
            its engine. The weights then take the buffers in that use order,
            and the stream start loads gate(0), up(0) and gate(1). With 1
            gate/up scale buffer each gate and up projection loads its own
            scale slot (the slot that the group load fills with this option
            off) where it is used, so the gate scales of the next expert load
            while the up matmuls of this one read theirs. It needs 3 to 5
            whole weight buffers, the default ``transpose_lookahead`` and the
            standard gate/up layout. With 4 or 5 buffers the stream start
            also loads down(0) (and up(1)), and each later load waits for the
            matmuls of the use 4 (or 5) before it. Each expert's matmuls,
            their PSUM accumulation order and the order in which the experts
            add up do not change, so the result does not change.
        quantized_latent: static loop only. None (default) loads and
            quantizes ``latent``. A tuple (codes, scales) gives the MXFP8
            expert input already in the layout of ``_quantize_natural_latent``
            (codes ``[128, D/512, T]`` ``float8_e4m3fn_x4`` and E8M0 scales
            ``[128, D/512, T]`` UINT8 SBUF, T a multiple of 4), and ``latent``
            must be None: the call runs no latent load, transpose or
            quantize. The expert loop reads the same bytes, so the result is
            the one of the latent that quantizes to them.
        shared_scale_buffer: static loop only, with ``scale_buffers=1``.
            False (default) keeps the gate/up scales of 2 experts in one buffer
            and the down scales of 4 experts in another. True keeps the scales
            of all three projections in one buffer of 4 slots: projection p of
            expert e takes slot (3e + p) % 4, and each projection loads its own
            slot where it is used. A slot load then waits for the matmuls of
            the projection 4 steps before it. The loop holds 21 KB less SBUF
            for each partition at the Kimi-K3 sizes. The matmuls read the same
            scale bytes, so the result does not change.
        extra_weight_buffers: static loop with ``half_weight_buffers`` only.
            Caller SBUF tiles that join the ring of half buffers after the
            kernel's own ``half_weight_buffers``, each ``[128, I/1024 * D]`` in
            the weight dtype (one half projection), e.g. tiles at explicit
            addresses whose bytes the caller does not use during the loop.
            Half h of projection p of expert e then uses buffer
            (6e + 2p + h) % (half_weight_buffers + len(extra_weight_buffers)).
            The caller must not touch the bytes of these tiles from the first
            load into them to the end of the call. Default (): none.
        extra_buffers_after: with ``extra_weight_buffers``: an instruction
            (e.g. the caller's last access of the tiles' bytes before the
            call), or a tuple of instructions, that the first load into each
            extra buffer waits for (a runtime dependency). None (default)
            adds no dependency.
            The weights, the matmuls and the result do not change.
        static_alt_tiles: static loop only. False (default) lets the compiler
            place a new tile for each chunk's gate sigmoid, up tanh, down PSUM
            copy and activation eviction. Under SBUF pressure it can stack them
            on a few addresses, and each ScalarE write then waits for VectorE
            to read the previous chunk's tile. True takes each of them from the
            next of two buffers (``_StaticLoopTiles``; the sigmoid and the up
            tanh share theirs), so a write waits only for the reads of the use
            before the last, and runs the PSUM copies of the even H tiles on
            ScalarE and of the odd ones on VectorE (``_copy_engine``). The
            buffers hold 8192 + 32 T bytes for each partition during the loop
            (12,288 at T = 128). "act" takes only the gate sigmoid / up tanh
            tiles from their two buffers (4096 bytes), "act_copy" also the PSUM
            copies (8192 bytes); the other tiles stay compiler-placed, and the
            copy engines are named as with True (``STATIC_ALT_TILE_FORMS``).
            "ring_act" / "ring_act_eviction" take the gate sigmoid / up tanh
            ring (4096 bytes) and with "ring_act_eviction" the eviction ring
            (32 T bytes), and keep ``psum_consume`` and ``resident_out``
            (``STATIC_ALT_TILE_PSUM_CONSUME``); the copy engines stay unnamed.
            Only the tiles of the instructions and the copy engines change, so
            the result does not change.
        cross_expert_transpose_blocks: with ``cross_expert_order`` only.
            False (default) keeps its order. True issues each expert's
            activation transposes as one block per I/512 tile (its 4
            transposes, one SiTU tile) between the gate chunks of the next
            expert: up(e), tile 0, gate(e + 1) chunk 0, tile 1, gate chunk 1,
            ..., down(e). TensorE schedule edges keep every matmul out of a
            block (``_transpose_blocks_and_gates``): the compiler otherwise
            puts single transposes between MX matmuls, and each one costs a
            switch out of and back into the MX matmul mode. The
            instructions and the result do not change.
        transpose_block: cross-expert order only. False (default) keeps
            the program, where the compiler puts most activation transposes
            one by one between MX matmuls; each such gap between two MX
            matmuls costs a background-transpose drain. True issues all the
            activation transposes of an expert in one block with no MX
            matmul inside, after gate(e + 1) chunk 0, so that the SiTU of
            the last up chunk ends first: up(e), gate(e + 1) chunk 0, the
            transposes of every chunk, the other gate(e + 1) chunks,
            down(e). TensorE schedule edges keep the order; the instructions
            and the result do not change.
        psum_consume: the static loop without ``static_alt_tiles``, or with
            one of ``STATIC_ALT_TILE_PSUM_CONSUME`` (also with
            ``cross_expert_order``). True: the affinity multiply-add
            of each down H tile reads the FP32 PSUM tile, so the PSUM -> SBUF
            copy of that tile goes. The FP32 affinities sit in the columns
            after the accumulator's, in one SBUF tile, so the two SBUF
            operands of the multiply-add share their base partitions
            (``_OutputTiles``). The multiply-add reads the same FP32 values:
            the result does not change. False (default) keeps the copies.
            The other paths ignore it.
        accum_memset: None (default) keeps the program. ``(engine, after)``:
            the zero fill of the FP32 output accumulator runs on ``engine``
            and, when ``after`` is an instruction, follows it in the schedule
            (the caller keeps the fill off its critical chain). The same zero
            bits.
        first_add_zero: False (default) keeps the program (the zero fill of
            the FP32 output accumulator). True, on the static loop without
            ``accum_memset``: no fill, and expert 0's multiply-add of each H
            slice adds a scalar +0.0 in place of the filled accumulator (the
            same FP32 add, so the same bits). The dynamic loop ignores it.
        up_product_split: static loop only. False (default) keeps the
            program. True: each up chunk's SiTU product (the VectorE
            scalar_tensor_tensor into the FP16 activation) runs as one
            instruction for each 128 columns, so the first transposes of the
            chunk wait only for their own columns. The same elementwise
            operations: the result does not change.
        resident_out: None (default) keeps the program. A dict (static loop
            only): the call fills it with the SBUF buffers that hold the last
            expert's weights and scales when the loop ends
            (``_resident_last_expert``), or leaves it empty when the loop form
            keeps no whole last expert. A later call that reads the buffers
            (``active_experts_in_slots`` ``resident``) keeps them alive. The
            result does not change.
        last_expert_order: ``LAST_EXPERT_ORDERS``, with
            ``cross_expert_order`` and ``transpose_block`` only. "default"
            keeps the program. "chunked" issues the activation transposes of
            the last expert chunk by chunk: without the edges the compiler
            puts the transposes of the last chunk, which wait for the last
            SiTU product, between those of the first chunk, and the first
            chunk's eviction and quantize wait for them. "up_first" (at least
            2 local experts, no ``resident_out``) also issues up(E_L - 1) and
            those transposes before down(E_L - 2), so that down(E_L - 2) runs
            on TensorE while ScalarE and VectorE evict and quantize the last
            activation. The ring then loads up(E_L - 1) before down(E_L - 2)
            and swaps their buffers (``static_expert_stream``
            ``last_up_first``; a prefetched stream must take the same). The
            instructions do not change, and the affinity products still add
            expert E_L - 2 before E_L - 1 into each output column, so the
            result does not change.
        affinity_sbuf_copy: static loop only. False (default) keeps the
            program. True: when ``local_affinities`` is already an SBUF tile
            (the slot affinities of a compacted call), the loop copies it into
            its FP32 affinity tile with one VectorE ``tensor_copy`` in place of
            the SBUF -> SBUF DMA into a raw tile and the copy after it. The
            compiler issues that DMA from the in-order Sync engine, where it
            waits for the slot-affinity gather and holds the weight-stream
            triggers behind it. The same values are copied, and the padding
            rows are zeroed as with the DMA, so the result does not change. An HBM
            ``local_affinities`` keeps the DMA.

    Returns:
        BF16 ``[T, D]`` unnormalized local routed-expert contribution, or
        ``column_groups_out`` when it is given.
    """
    kernel_assert(
        nl.program_ndim() == 0 or (nl.program_ndim() == 1 and nl.num_programs(0) == 1),
        "moe_tkg_decode supports LNC1 only",
    )
    kernel_assert(nisa.get_nc_version() >= nisa.nc_version.gen4, "moe_tkg_decode requires Trn3 or newer")
    if quantized_latent is not None:
        latent = _check_quantized_latent(
            latent, quantized_latent, is_all_expert_dynamic, active_expert_slots, direct_latent
        )
    kernel_assert(len(latent.shape) == 2, f"Expected latent [T, D], got {latent.shape}")
    kernel_assert(len(local_affinities.shape) == 2, f"Expected local_affinities [T, E_L], got {local_affinities.shape}")
    kernel_assert(
        len(expert_gate_up_weights.shape) == 5,
        f"Expected expert_gate_up_weights rank 5, got {expert_gate_up_weights.shape}",
    )
    kernel_assert(
        len(expert_down_weights.shape) == 4,
        f"Expected expert_down_weights rank 4, got {expert_down_weights.shape}",
    )
    kernel_assert(latent.dtype == nl.bfloat16, f"Expected BF16 latent, got {latent.dtype}")
    kernel_assert(
        local_affinities.dtype == nl.bfloat16 or local_affinities.dtype == nl.float32,
        f"Expected BF16 or FP32 local_affinities, got {local_affinities.dtype}",
    )
    kernel_assert(
        expert_gate_up_weights.dtype == nl.float4_e2m1fn_x4,
        f"Expected MXFP4 gate/up weights, got {expert_gate_up_weights.dtype}",
    )
    kernel_assert(
        expert_down_weights.dtype == nl.float4_e2m1fn_x4,
        f"Expected MXFP4 down weights, got {expert_down_weights.dtype}",
    )
    kernel_assert(
        expert_gate_up_weights_scale.dtype == nl.uint8,
        f"Expected UINT8 gate/up MX scales, got {expert_gate_up_weights_scale.dtype}",
    )
    kernel_assert(
        expert_down_weights_scale.dtype == nl.uint8,
        f"Expected UINT8 down MX scales, got {expert_down_weights_scale.dtype}",
    )
    kernel_assert(beta > 0.0, f"Expected beta > 0, got {beta}")
    kernel_assert(linear_beta > 0.0, f"Expected linear_beta > 0, got {linear_beta}")

    T_physical, D = latent.shape
    E_L = expert_gate_up_weights.shape[0]
    I = expert_gate_up_weights.shape[-1]
    n_D512 = D // MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
    n_I512 = I // MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
    T = div_ceil(T_physical, _q_width) * _q_width

    kernel_assert(T_physical > 0 and T_physical <= _PMAX, f"Expected 1 <= T <= 128, got T={T_physical}")
    kernel_assert(
        D > 0 and D % MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM == 0,
        f"Expected D divisible by 512, got D={D}",
    )
    kernel_assert(
        I > 0 and I % MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM == 0,
        f"Expected I divisible by 512, got I={I}",
    )
    kernel_assert(E_L > 0, f"Expected at least one local expert, got E_L={E_L}")
    kernel_assert(
        local_affinities.shape == (T_physical, E_L),
        f"Expected local_affinities shape {(T_physical, E_L)}, got {local_affinities.shape}",
    )
    kernel_assert(
        expert_gate_up_weights.shape == (E_L, _PMAX, 2, n_D512, I),
        f"Unexpected gate/up weight shape {expert_gate_up_weights.shape}",
    )
    kernel_assert(
        expert_gate_up_weights_scale.shape == (E_L, _MX_SCALE_PARTITIONS, 2, n_D512, I),
        f"Unexpected gate/up scale shape {expert_gate_up_weights_scale.shape}",
    )
    kernel_assert(
        expert_down_weights.shape == (E_L, _PMAX, n_I512, D),
        f"Unexpected down weight shape {expert_down_weights.shape}",
    )
    kernel_assert(
        expert_down_weights_scale.shape == (E_L, _MX_SCALE_PARTITIONS, n_I512, D),
        f"Unexpected down scale shape {expert_down_weights_scale.shape}",
    )

    # More than one expert's projections ahead only in the static loop with whole buffers.
    static_whole = not (is_all_expert_dynamic or active_expert_slots or half_weight_buffers or gate_up_half_major)
    max_weight_buffers = _MAX_STATIC_WEIGHT_BUFFERS if static_whole else _STATIC_MAX_WEIGHT_BUFFERS
    kernel_assert(
        1 <= weight_buffers <= max_weight_buffers,
        f"Expected 1 <= weight_buffers <= {max_weight_buffers}, got {weight_buffers}",
    )
    kernel_assert(
        weight_buffers <= _PROJECTIONS_PER_EXPERT or not active_expert_slots,
        f"The active-expert slots take 1 to {_PROJECTIONS_PER_EXPERT} weight buffers, got {weight_buffers}",
    )
    if stream is not None:
        kernel_assert(
            not (is_all_expert_dynamic or active_expert_slots) and early_loads,
            "A prefetched stream feeds the static loop with early_loads only",
        )
        stream.check(
            expert_gate_up_weights.shape,
            expert_down_weights.shape,
            half_weight_buffers or weight_buffers,
            scale_buffers,
            gate_up_half_major,
            expert_queues,
            use_order=cross_expert_order,
            last_up_first=last_expert_order == "up_first",
        )
    kernel_assert(
        expert_queues in EXPERT_QUEUES and (expert_queues == 1 or not (is_all_expert_dynamic or active_expert_slots)),
        f"expert_queues must be one of {EXPERT_QUEUES} (2: static loop only), got {expert_queues}",
    )
    kernel_assert(
        0 <= preamble_buffers <= _PROJECTIONS_PER_EXPERT,
        f"Expected 0 <= preamble_buffers <= {_PROJECTIONS_PER_EXPERT}, got {preamble_buffers}",
    )
    kernel_assert(
        not preamble_buffers or (static_whole and early_loads and stream is None and not cross_expert_order),
        "preamble_buffers needs the static loop with whole weight buffers and early_loads, without a prefilled "
        "stream or cross_expert_order",
    )
    kernel_assert(not preamble_reuse or preamble_buffers, "preamble_reuse needs preamble_buffers")
    kernel_assert(
        1 <= scale_buffers <= _GATE_UP_SCALE_BUFFERS,
        f"Expected 1 <= scale_buffers <= {_GATE_UP_SCALE_BUFFERS}, got {scale_buffers}",
    )
    kernel_assert(
        0 <= n_static_experts <= E_L,
        f"Expected 0 <= n_static_experts <= E_L={E_L}, got {n_static_experts}",
    )
    kernel_assert(
        is_all_expert_dynamic or n_static_experts == 0,
        f"n_static_experts={n_static_experts} applies only with is_all_expert_dynamic=True",
    )
    kernel_assert(
        not is_all_expert_dynamic
        or (weight_buffers == _PROJECTIONS_PER_EXPERT and scale_buffers == _GATE_UP_SCALE_BUFFERS),
        f"The dynamic loop takes 3 weight and 2 scale buffers, got {weight_buffers} and {scale_buffers}",
    )
    kernel_assert(
        0 <= active_expert_slots <= E_L,
        f"Expected 0 <= active_expert_slots <= E_L={E_L}, got {active_expert_slots}",
    )
    kernel_assert(
        not (is_all_expert_dynamic and active_expert_slots),
        "A call runs one expert path: the dynamic loop or the active-expert slots, not both",
    )
    kernel_assert(
        not active_expert_slots or T_physical <= _SLOT_MAX_TOKENS,
        f"The active-expert slots are verified for T <= {_SLOT_MAX_TOKENS} tokens, got T={T_physical}",
    )
    kernel_assert(
        active_expert_count is None or active_expert_slots,
        "active_expert_count reports the active-expert slots: it needs active_expert_slots > 0",
    )
    kernel_assert(
        half_weight_buffers == 0 or 2 <= half_weight_buffers <= 2 * _PROJECTIONS_PER_EXPERT,
        f"Expected half_weight_buffers 0 or 2 to {2 * _PROJECTIONS_PER_EXPERT}, got {half_weight_buffers}",
    )
    kernel_assert(
        not half_weight_buffers or not (is_all_expert_dynamic or active_expert_slots),
        "half_weight_buffers applies to the static loop only",
    )
    kernel_assert(
        not gate_up_half_major or not (is_all_expert_dynamic or active_expert_slots or n_static_experts),
        "gate_up_half_major applies to the static loop only",
    )
    kernel_assert(
        not gate_up_half_major or expert_gate_up_weights.shape[-1] % (2 * _GATE_UP_CHUNK_I) == 0,
        f"gate_up_half_major needs I a multiple of {2 * _GATE_UP_CHUNK_I}, got {expert_gate_up_weights.shape}",
    )
    kernel_assert(
        1 <= transpose_lookahead <= _TRANSPOSE_LOOKAHEAD_CHUNKS,
        f"Expected transpose_lookahead 1 to {_TRANSPOSE_LOOKAHEAD_CHUNKS}, got {transpose_lookahead}",
    )
    kernel_assert(
        transpose_lookahead == _TRANSPOSE_LOOKAHEAD_CHUNKS or not (is_all_expert_dynamic or active_expert_slots),
        "transpose_lookahead applies to the static loop only",
    )
    if cross_expert_order:
        kernel_assert(
            not (is_all_expert_dynamic or active_expert_slots), "cross_expert_order applies to the static loop only"
        )
        kernel_assert(
            _PROJECTIONS_PER_EXPERT <= weight_buffers <= _STATIC_MAX_WEIGHT_BUFFERS and not half_weight_buffers,
            f"cross_expert_order needs {_PROJECTIONS_PER_EXPERT} to {_STATIC_MAX_WEIGHT_BUFFERS} whole weight buffers, "
            f"got weight_buffers={weight_buffers} half_weight_buffers={half_weight_buffers}",
        )
        kernel_assert(
            transpose_lookahead == _TRANSPOSE_LOOKAHEAD_CHUNKS,
            f"cross_expert_order orders the transposes itself: it takes the default transpose_lookahead, got "
            f"{transpose_lookahead}",
        )
        kernel_assert(not gate_up_half_major, "cross_expert_order reads the standard gate/up layout")
        # The second expert queue keeps the buffer order of the default loop. A prefetched stream takes the
        # order of the call (stream.check above).
        kernel_assert(expert_queues == 1, f"cross_expert_order takes one expert queue, got {expert_queues}")
    if extra_weight_buffers:
        kernel_assert(
            half_weight_buffers and not (is_all_expert_dynamic or active_expert_slots or cross_expert_order),
            "extra_weight_buffers joins the half buffers of the static loop: it needs half_weight_buffers",
        )
        kernel_assert(
            half_weight_buffers + len(extra_weight_buffers) <= 2 * _PROJECTIONS_PER_EXPERT,
            f"Expected at most {2 * _PROJECTIONS_PER_EXPERT} half buffers in all, got {half_weight_buffers} + "
            f"{len(extra_weight_buffers)}",
        )
    kernel_assert(extra_buffers_after is None or extra_weight_buffers, "extra_buffers_after needs extra_weight_buffers")
    if shared_scale_buffer:
        kernel_assert(
            not (is_all_expert_dynamic or active_expert_slots), "shared_scale_buffer applies to the static loop only"
        )
        kernel_assert(scale_buffers == 1, f"shared_scale_buffer needs scale_buffers=1, got {scale_buffers}")
        kernel_assert(not cross_expert_order, "shared_scale_buffer keeps the expert order: not with cross_expert_order")
    kernel_assert(
        static_alt_tiles in STATIC_ALT_TILE_FORMS + STATIC_ALT_TILE_PSUM_CONSUME,
        f"static_alt_tiles must be one of {STATIC_ALT_TILE_FORMS + STATIC_ALT_TILE_PSUM_CONSUME}, got "
        f"{static_alt_tiles!r}",
    )
    kernel_assert(
        not static_alt_tiles or not (is_all_expert_dynamic or active_expert_slots),
        "static_alt_tiles applies to the static loop only",
    )
    kernel_assert(
        not cross_expert_transpose_blocks or cross_expert_order,
        "cross_expert_transpose_blocks orders the activation transposes of the cross-expert order: it needs "
        "cross_expert_order",
    )
    kernel_assert(not transpose_block or cross_expert_order, "transpose_block needs cross_expert_order")
    kernel_assert(
        not (transpose_block and cross_expert_transpose_blocks),
        "transpose_block and cross_expert_transpose_blocks both order the transposes of the cross-expert order: "
        "pass one of them",
    )
    kernel_assert(
        last_expert_order in LAST_EXPERT_ORDERS,
        f"last_expert_order must be one of {LAST_EXPERT_ORDERS}, got {last_expert_order!r}",
    )
    kernel_assert(
        last_expert_order == "default" or (cross_expert_order and transpose_block),
        f"last_expert_order {last_expert_order!r} orders the last experts of the cross-expert order with "
        "transpose_block: it needs both",
    )
    kernel_assert(
        last_expert_order != "up_first" or (expert_gate_up_weights.shape[0] >= 2 and resident_out is None),
        f"last_expert_order 'up_first' needs at least 2 local experts and no resident_out, got "
        f"{expert_gate_up_weights.shape[0]} experts",
    )
    if column_groups_out is not None:
        kernel_assert(
            not (is_all_expert_dynamic or active_expert_slots), "column_groups_out applies to the static loop only"
        )
        kernel_assert(D % (2 * _PMAX) == 0, f"column_groups_out needs D a multiple of {2 * _PMAX}, got D={D}")
        block_tokens = _column_group_block_tokens(D)
        kernel_assert(
            T_physical % block_tokens == 0,
            f"column_groups_out needs T a multiple of {block_tokens} at D={D}, so that each row of the tile is whole "
            f"{_SBUF_BLOCK_BYTES}-byte SBUF blocks, got T={T_physical}",
        )
        kernel_assert(
            tuple(column_groups_out.shape) == (_PMAX, T_physical, D // _PMAX)
            and column_groups_out.dtype == nl.bfloat16
            and column_groups_out.buffer == nl.sbuf,
            f"Expected a BF16 SBUF column_groups_out {(_PMAX, T_physical, D // _PMAX)}, got "
            f"{column_groups_out.dtype} {column_groups_out.buffer} {column_groups_out.shape}",
        )
        kernel_assert(output is None, "column_groups_out is the output of the call: it takes no output tensor")
    if direct_latent:
        kernel_assert(
            not (is_all_expert_dynamic or active_expert_slots), "direct_latent applies to the static loop only"
        )
        # The partition step of the transposes' access pattern is D, so .ap() also rejects a tile whose rows are not
        # D elements wide (a column slice of a wider tile).
        kernel_assert(
            latent.buffer == nl.sbuf and latent.offset == 0 and T == T_physical,
            f"direct_latent reads the latent in place: it needs a whole SBUF [T, D] tile (view offset 0) with T a "
            f"multiple of {_q_width}, got {latent.buffer} offset {latent.offset} T={T_physical}",
        )
    if active_expert_count is not None:
        kernel_assert(
            tuple(active_expert_count.shape) == (1, 1) and active_expert_count.dtype == nl.int32,
            f"Expected an INT32 [1, 1] active_expert_count, got {active_expert_count.dtype} {active_expert_count.shape}",
        )
    kernel_assert(
        output is None or (tuple(output.shape) == (T_physical, D) and output.dtype == nl.bfloat16),
        f"Expected a BF16 output {(T_physical, D)}, got {None if output is None else output.shape}",
    )
    kernel_assert(
        not (active_expert_slots and output is not None),
        "The active-expert slots return their own output",
    )
    if active_expert_slots:
        return _active_experts_in_slots(
            latent=latent,
            local_affinities=local_affinities,
            expert_gate_up_weights=expert_gate_up_weights,
            expert_down_weights=expert_down_weights,
            expert_gate_up_weights_scale=expert_gate_up_weights_scale,
            expert_down_weights_scale=expert_down_weights_scale,
            beta=beta,
            linear_beta=linear_beta,
            T=T,
            T_physical=T_physical,
            weight_buffers=weight_buffers,
            early_loads=early_loads,
            n_slots=active_expert_slots,
            active_expert_count=active_expert_count,
        )
    if not is_all_expert_dynamic:
        _count_static_call(weight_buffers, half_weight_buffers)

    stream_args = dict(
        expert_gate_up_weights=expert_gate_up_weights,
        expert_down_weights=expert_down_weights,
        expert_gate_up_weights_scale=expert_gate_up_weights_scale,
        expert_down_weights_scale=expert_down_weights_scale,
        weight_buffers=weight_buffers,
        scale_buffers=scale_buffers,
        half_weight_buffers=half_weight_buffers,
        gate_up_half_major=gate_up_half_major,
        use_order=cross_expert_order,
        **({"last_up_first": True} if last_expert_order == "up_first" else {}),
        **({"shared_scale": True} if shared_scale_buffer else {}),
        **(
            {"extra_buffers": tuple(extra_weight_buffers), "extra_after": extra_buffers_after}
            if extra_weight_buffers
            else {}
        ),
        expert_queues=expert_queues,
        **({"preamble_buffers": preamble_buffers} if preamble_buffers else {}),
        **({"preamble_reuse": True} if preamble_reuse else {}),
    )
    if quantized_latent is not None:
        latent_sb = None
    elif direct_latent:
        latent_sb = latent
    else:
        latent_sb = _load_natural_latent(latent=latent, T=T, T_physical=T_physical, D=D)
    # The static loop loads each of its buffers for the first time before any
    # work tile of this kernel exists (see _StaticExpertStream).
    if stream is None and early_loads and not is_all_expert_dynamic:
        stream = _StaticExpertStream(**stream_args, first_loads=True)
    elif stream is not None:
        stream.finish_first_loads(
            expert_gate_up_weights, expert_down_weights, expert_gate_up_weights_scale, expert_down_weights_scale
        )
    if quantized_latent is not None:
        input_quant_sb, input_scale_sb = quantized_latent
    else:
        input_quant_sb, input_scale_sb = _quantize_natural_latent(latent_sb=latent_sb, T=T, D=D)
    # psum_consume: the static loop's FP32 affinities in the columns after the accumulator's (one SBUF tile).
    static_psum_consume = (
        psum_consume
        and not is_all_expert_dynamic
        and (not static_alt_tiles or static_alt_tiles in STATIC_ALT_TILE_PSUM_CONSUME)
    )
    affinity_columns = E_L if static_psum_consume else 0
    accum_aff_sb = nl.ndarray((T, D + affinity_columns), dtype=nl.float32, buffer=nl.sbuf)
    output_accum_sb = accum_aff_sb[:, 0:D] if affinity_columns else accum_aff_sb
    # first_add_zero: the static loop's first expert writes its products plus a scalar +0.0, the same FP32 add as onto the
    # zero fill, so the fill goes (not on the dynamic loop, nor with a caller's accum_memset).
    first_adds_zero = first_add_zero and accum_memset is None and not is_all_expert_dynamic
    if accum_memset is None:
        if not first_adds_zero:
            nisa.memset(dst=output_accum_sb, value=0.0)
    else:
        fill = nisa.memset(dst=output_accum_sb, value=0.0, engine=accum_memset[0])
        if accum_memset[1] is not None:
            _instruction(fill).depends_on(_instruction(accum_memset[1]), kind=nisa.dependency.schedule)

    # The static loop stores the output itself, from its last expert (with column_groups_out: no HBM output).
    if output is None and column_groups_out is None:
        output = nl.ndarray((T_physical, D), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    if is_all_expert_dynamic:
        _active_experts_dynamic(
            input_quant_sb=input_quant_sb,
            input_scale_sb=input_scale_sb,
            output_accum_sb=output_accum_sb,
            local_affinities=local_affinities,
            expert_gate_up_weights=expert_gate_up_weights,
            expert_down_weights=expert_down_weights,
            expert_gate_up_weights_scale=expert_gate_up_weights_scale,
            expert_down_weights_scale=expert_down_weights_scale,
            beta=beta,
            linear_beta=linear_beta,
            T=T,
            T_physical=T_physical,
            n_static_experts=n_static_experts,
        )
        _store_accumulated_output(output_accum_sb=output_accum_sb, output=output, T=T)
        return output
    if stream is None:
        stream = _StaticExpertStream(**stream_args, first_loads=False)
    if up_product_split:
        # Read by _ExpertSteps.up.
        stream.up_product_split = True
    _all_experts_static(
        input_quant_sb=input_quant_sb,
        input_scale_sb=input_scale_sb,
        output_accum_sb=output_accum_sb,
        first_adds_zero=first_adds_zero,
        local_affinities=local_affinities,
        expert_gate_up_weights=expert_gate_up_weights,
        expert_down_weights=expert_down_weights,
        expert_gate_up_weights_scale=expert_gate_up_weights_scale,
        expert_down_weights_scale=expert_down_weights_scale,
        beta=beta,
        linear_beta=linear_beta,
        T=T,
        T_physical=T_physical,
        stream=stream,
        output=output,
        column_groups_out=column_groups_out,
        transpose_lookahead=transpose_lookahead,
        cross_expert_order=cross_expert_order,
        static_alt_tiles=static_alt_tiles,
        cross_expert_transpose_blocks=cross_expert_transpose_blocks,
        transpose_block=transpose_block,
        affinity_f32_sb=accum_aff_sb[:, D : D + affinity_columns] if affinity_columns else None,
        **({"last_expert_order": last_expert_order} if last_expert_order != "default" else {}),
        **({"affinity_sbuf_copy": True} if affinity_sbuf_copy else {}),
    )
    if resident_out is not None and (not static_alt_tiles or static_alt_tiles in STATIC_ALT_TILE_PSUM_CONSUME):
        resident_out.update(_resident_last_expert(stream))
    return output if column_groups_out is None else column_groups_out


def _resident_last_expert(stream: "_StaticExpertStream") -> dict:
    """Return the buffers of the static loop that hold the last expert's weights and scales when the loop ends.

    In the cross-expert order (``use_order``) the last uses are gate(E_L - 1), down(E_L - 2), up(E_L - 1) and
    down(E_L - 1), so with at least 4 whole weight buffers the 3 projections of expert E_L - 1 stay in their buffers
    (with 3, down(E_L - 1) takes the buffer of gate(E_L - 1)).
    Its gate/up scales are the last group of the gate/up scale buffers (slots 0 and 1 when E_L is odd), its down
    scales slot (E_L - 1) % 4 of the down scale buffer. Empty for the other forms (half or half-major pieces,
    preamble or extra buffers, the shared scale buffer, an even E_L).
    """
    E_L = stream.shapes[0][0]
    gate_up_experts_per_buffer = _SCALE_SLOTS_PER_QUADRANT // 2
    group_idx, group_expert = divmod(E_L - 1, gate_up_experts_per_buffer)
    if (
        stream.use_index is None
        or len(stream.first_tiles) != 1
        or stream.gate_up_half_major
        or stream.preamble
        or stream.extra_first
        or stream.shared_scale
        or len(stream.buffers) < _PROJECTIONS_PER_EXPERT + 1
        or group_expert != 0
    ):
        return {}
    return {
        "expert": E_L - 1,
        "gate": stream._buffer(GATE_FUSED_IDX, E_L - 1),
        "up": stream._buffer(UP_FUSED_IDX, E_L - 1),
        "down": stream._buffer(_DOWN_IDX, E_L - 1),
        "gate_up_scale": stream.gate_up_scale_buffers[group_idx % len(stream.gate_up_scale_buffers)],
        "down_scale": stream.down_scale_buffer,
        "down_scale_slot": (E_L - 1) % _SCALE_SLOTS_PER_QUADRANT,
    }


def _cross_expert_uses(E_L: int, last_up_first: bool = False) -> dict:
    """Return the use index of each (projection, expert) in the cross-expert order (``_cross_expert_steps``):
    gate(0), up(0), then for each expert e gate(e + 1), down(e), up(e + 1). With ``last_up_first`` (``last_expert_order``
    "up_first", at least 2 experts) up(E_L - 1) comes before down(E_L - 2)."""
    order = [(GATE_FUSED_IDX, 0), (UP_FUSED_IDX, 0)]
    for expert_idx in range(E_L):
        if expert_idx + 1 < E_L:
            order.append((GATE_FUSED_IDX, expert_idx + 1))
        order.append((_DOWN_IDX, expert_idx))
        if expert_idx + 1 < E_L:
            order.append((UP_FUSED_IDX, expert_idx + 1))
    if last_up_first:
        kernel_assert(E_L >= 2, f"last_up_first needs at least 2 local experts, got {E_L}")
        down_idx = order.index((_DOWN_IDX, E_L - 2))
        up_idx = order.index((UP_FUSED_IDX, E_L - 1))
        order[down_idx], order[up_idx] = order[up_idx], order[down_idx]
    return {use: idx for idx, use in enumerate(order)}


def shared_scale_slot(expert_idx: int, projection_idx: int) -> int:
    """Return the slot of projection ``projection_idx`` of expert ``expert_idx`` in the shared scale buffer of the
    static loop (``shared_scale_buffer`` of ``moe_tkg_decode``): the projections take the 4 slots in turn."""
    return (_PROJECTIONS_PER_EXPERT * expert_idx + projection_idx) % _SCALE_SLOTS_PER_QUADRANT


class _StaticExpertStream:
    """The SBUF buffers of the expert loop, each with its first load issued (``first_loads``).

    The static weight and scale loads all run on one in-order DMA queue. From
    the second expert on, a load waits there for the matmuls that free its
    buffer, and every later load waits behind it. The allocator may give a
    buffer that is first written after a work tile of this kernel has died the
    bytes of that tile. The first load then waits for the tile's last reader,
    and the queue puts it behind the next expert's loads. In the fused decode
    layer, the first expert's down scales landed behind the second expert's
    gate and up weights, so the first down matmul waited for them. A
    first load issued before the latent is quantized can only wait for a tile
    that the caller freed before the call.

    Without ``first_loads`` the stream only allocates the buffers, and the
    loop issues each first load where the first expert uses it. With
    ``zero_scales`` the scale buffers are zeroed before the first loads.

    With ``half_weight_buffers`` a projection streams in two pieces (halves):
    gate and up split their I/512 chunks, down its I/512 tiles. A piece is
    ``(first I/512 tile, SBUF weights)``, and the loop takes tile ``i`` of a
    projection from the last piece whose first tile is at most ``i``.
    With ``gate_up_half_major`` gate and up always come in two pieces; without
    half buffers both are views of one whole-projection load.

    With ``shared_scale`` one scale buffer of 4 slots holds the scales of every
    projection: projection p of expert e takes slot (3e + p) % 4
    (``shared_scale_slot``), and ``load_scale_slot`` loads one slot. The stream
    start loads the slots of the first 4 projections (``first_scale_slots``).

    With ``expert_queues`` 2 the weights of the odd experts load on a second
    DMA queue (hardware DGE from the Sync engine), so the transfers of two
    consecutive experts can overlap; the scales stay on the first queue.

    ``first_load_priority`` (trn3 DMA QoS level, 0 is the highest) applies to
    the first loads only, so a stream started early yields the DMA rings to
    the work it overlaps; the loop's own loads keep the default priority.

    With ``preamble_buffers`` K, the first K whole projections in use order
    take K buffers of their own, and projection u >= K takes ring buffer
    (u - K) % weight_buffers; the stream start loads the first K + weight_buffers
    projections. With ``preamble_reuse`` projection K + weight_buffers + k takes
    buffer k of its own too, and projection u >= 2K + weight_buffers takes ring
    buffer (u - 2K) % weight_buffers.

    With ``prefetch_uses`` k (``use_order`` only, 0 to 3), the stream start loads the projections of the first k uses
    only (with the same scale groups), and the call issues the first loads of the other buffers at its start
    (``finish_first_loads``). With ``prefetch_scales`` False the call also loads the scale groups at its start. The
    later buffers are first written there, so only k weight buffers hold bytes before the call. The call-start loads
    take no DMA QoS level (as a call's own stream start), the stream-start loads take ``first_load_priority``. None
    (default) loads every buffer at the stream start.
    """

    def __init__(
        self,
        expert_gate_up_weights: nl.NkiTensor,
        expert_down_weights: nl.NkiTensor,
        expert_gate_up_weights_scale: nl.NkiTensor,
        expert_down_weights_scale: nl.NkiTensor,
        weight_buffers: int,
        scale_buffers: int,
        first_loads: bool,
        half_weight_buffers: int = 0,
        gate_up_half_major: bool = False,
        zero_scales: bool = False,
        use_order: bool = False,
        shared_scale: bool = False,
        extra_buffers: tuple = (),
        extra_after=None,
        expert_queues: int = 1,
        preamble_buffers: int = 0,
        preamble_reuse: bool = False,
        first_load_priority: int = None,
        prefetch_uses: int = None,
        prefetch_scales: bool = True,
        last_up_first: bool = False,
    ):
        E_L = expert_gate_up_weights.shape[0]
        self.last_up_first = last_up_first
        self.prefetch_uses = prefetch_uses
        self.prefetch_scales = prefetch_scales
        self.expert_queues = expert_queues
        self.use_order = use_order
        I = expert_gate_up_weights.shape[-1]
        D = expert_down_weights.shape[-1]
        n_D512 = D // MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
        n_I512 = I // MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
        # One weight buffer per projection, reused by every expert: a load then
        # waits only for the same projection of the previous expert. Every
        # projection has D*I/512 weights and scales per partition row, so the
        # scales share buffers of that size, several projections per buffer (see
        # _load_scale_rows). With fewer buffers, projection p of expert e uses
        # buffer (3e + p) % weight_buffers.
        projection_row_size = n_D512 * I
        # The first I/512 tile of each piece of a projection. A half of gate or
        # up (its I/512 chunks) and a half of down (its I/512 tiles) hold the
        # same number of weights per partition.
        self.first_tiles = (0, div_ceil(n_I512, 2)) if half_weight_buffers else (0,)
        piece_row_size = self.first_tiles[-1] * D if half_weight_buffers else projection_row_size
        self.gate_up_half_major = gate_up_half_major
        self.n_D512, self.n_I512, self.I = n_D512, n_I512, I
        self.shapes = (tuple(expert_gate_up_weights.shape), tuple(expert_down_weights.shape))
        # The half-major gate/up weights as [E_L, 128, 2, half, D/512, I/2]: a half is one run per partition.
        self.gate_up_halves = (
            expert_gate_up_weights.reshape((E_L, _PMAX, 2, 2, n_D512, I // 2)) if gate_up_half_major else None
        )
        self.buffers = []
        for _ in range(half_weight_buffers or weight_buffers):
            self.buffers.append(nl.ndarray((_PMAX, piece_row_size), dtype=expert_down_weights.dtype, buffer=nl.sbuf))
        # The caller's buffers join the ring after the own ones; the first load into each waits for extra_after.
        for buffer in extra_buffers:
            kernel_assert(
                tuple(buffer.shape) == (_PMAX, piece_row_size)
                and buffer.dtype == expert_down_weights.dtype
                and buffer.buffer == nl.sbuf,
                f"Expected an SBUF {(_PMAX, piece_row_size)} {expert_down_weights.dtype} extra weight buffer, got "
                f"{buffer.buffer} {buffer.shape} {buffer.dtype}",
            )
            self.buffers.append(buffer)
        self.extra_first = {len(self.buffers) - len(extra_buffers) + i for i in range(len(extra_buffers))}
        self.extra_after = extra_after
        self.shared_scale = shared_scale
        self.first_scale_slots = set()
        self.preamble_reuse = preamble_reuse
        self.preamble = []
        for _ in range(preamble_buffers):
            self.preamble.append(nl.ndarray((_PMAX, piece_row_size), dtype=expert_down_weights.dtype, buffer=nl.sbuf))
        # The projections (pieces) whose load the stream start issues: the first use of each buffer.
        self.n_first = min(
            len(self.buffers) + len(self.preamble), _PROJECTIONS_PER_EXPERT * E_L * len(self.first_tiles)
        )
        self.gate_up_scale_buffers = []
        if shared_scale:
            # One buffer of 4 slots for the scales of every projection (gate/up and down).
            self.down_scale_buffer = nl.ndarray((_PMAX, projection_row_size), dtype=nl.uint8, buffer=nl.sbuf)
            self.gate_up_scale_buffers.append(self.down_scale_buffer)
        else:
            for _ in range(scale_buffers):
                self.gate_up_scale_buffers.append(
                    nl.ndarray((_PMAX, projection_row_size), dtype=nl.uint8, buffer=nl.sbuf)
                )
            self.down_scale_buffer = nl.ndarray((_PMAX, projection_row_size), dtype=nl.uint8, buffer=nl.sbuf)
        if zero_scales:
            # A dynamic loop that loads scale slots needs every byte of the buffer written before the loop.
            for scale_buffer in self.gate_up_scale_buffers + [self.down_scale_buffer]:
                nisa.memset(dst=scale_buffer, value=0)
        # The pieces of the first expert and the gate/up scale groups whose
        # first load the stream start issued, and whether it loaded the first
        # down scale group.
        self.first_weights = {}
        gate_up_experts_per_buffer = _SCALE_SLOTS_PER_QUADRANT // 2
        self.first_gate_up_groups = min(scale_buffers, div_ceil(E_L, gate_up_experts_per_buffer)) if first_loads else 0
        self.first_down_scales = first_loads
        # With use_order, the projections take the buffers in the order the cross-expert loop uses them.
        self.use_index = _cross_expert_uses(E_L, last_up_first) if use_order else None
        if not first_loads:
            return
        priority = first_load_priority
        if use_order:
            self._first_loads_in_use_order(
                expert_gate_up_weights,
                expert_down_weights,
                expert_gate_up_weights_scale,
                expert_down_weights_scale,
                priority,
            )
            return
        if shared_scale:
            self._first_loads_shared_scale(
                expert_gate_up_weights, expert_down_weights, expert_gate_up_weights_scale, expert_down_weights_scale
            )
            return

        # The first loads, in the order the experts use them: the first
        # expert's pieces that start a buffer and its scales, then the second
        # gate/up scale group.
        self._load_gate_up_scale_group(expert_gate_up_weights_scale, buffer_idx=0, priority=priority)
        first_gate_up_weights = expert_gate_up_weights.select(dim=0, index=0)
        for projection_idx in (GATE_FUSED_IDX, UP_FUSED_IDX):
            self._load_first_pieces(
                projection_idx, first_gate_up_weights.select(dim=1, index=projection_idx), priority=priority
            )
        _load_scale_rows(
            dst=self.down_scale_buffer,
            scale=expert_down_weights_scale,
            first_expert=0,
            n_experts=min(_SCALE_SLOTS_PER_QUADRANT, E_L),
            projections_per_expert=1,
            priority=priority,
        )
        self._load_first_pieces(_DOWN_IDX, expert_down_weights.select(dim=0, index=0), priority=priority)
        # Whole buffers past three: the next expert's projections take their first use of a buffer here too. Its
        # scales are in the first scale groups, which the loads above issued.
        if not (half_weight_buffers or gate_up_half_major):
            for piece in range(_PROJECTIONS_PER_EXPERT, self.n_first):
                expert_idx, projection_idx = divmod(piece, _PROJECTIONS_PER_EXPERT)
                weights = expert_down_weights if projection_idx == _DOWN_IDX else expert_gate_up_weights
                weight = weights.select(dim=0, index=expert_idx)
                if projection_idx != _DOWN_IDX:
                    weight = weight.select(dim=1, index=projection_idx)
                self.first_weights[(projection_idx, expert_idx, 0)] = self._load_piece(
                    projection_idx, expert_idx, 0, weight, priority=priority
                )
        for buffer_idx in range(1, self.first_gate_up_groups):
            self._load_gate_up_scale_group(expert_gate_up_weights_scale, buffer_idx=buffer_idx, priority=priority)

    def _first_loads_in_use_order(
        self,
        expert_gate_up_weights: nl.NkiTensor,
        expert_down_weights: nl.NkiTensor,
        expert_gate_up_weights_scale: nl.NkiTensor,
        expert_down_weights_scale: nl.NkiTensor,
        priority: int = None,
    ):
        """The first loads of the cross-expert order: the first gate/up scale group, the projections of the first
        uses (one per buffer; gate(0), up(0), gate(1) with 3 buffers, then down(0) with 4 and up(1) with 5), the first
        down scale group, and the other gate/up scale groups that the stream start loads."""
        if self.prefetch_scales:
            self._load_gate_up_scale_group(expert_gate_up_weights_scale, buffer_idx=0, priority=priority)
        n_uses = len(self.buffers) if self.prefetch_uses is None else self.prefetch_uses
        self._load_uses(expert_gate_up_weights, expert_down_weights, 0, n_uses, priority)
        if self.prefetch_scales:
            self._load_later_scales(expert_gate_up_weights_scale, expert_down_weights_scale, priority)

    def _load_later_scales(self, expert_gate_up_weights_scale, expert_down_weights_scale, priority: int = None):
        """Load the first down scale group and the gate/up scale groups past the first that the stream start loads."""
        E_L = expert_gate_up_weights_scale.shape[0]
        _load_scale_rows(
            dst=self.down_scale_buffer,
            scale=expert_down_weights_scale,
            first_expert=0,
            n_experts=min(_SCALE_SLOTS_PER_QUADRANT, E_L),
            projections_per_expert=1,
            priority=priority,
        )
        for buffer_idx in range(1, self.first_gate_up_groups):
            self._load_gate_up_scale_group(expert_gate_up_weights_scale, buffer_idx=buffer_idx, priority=priority)

    def _first_loads_shared_scale(
        self,
        expert_gate_up_weights: nl.NkiTensor,
        expert_down_weights: nl.NkiTensor,
        expert_gate_up_weights_scale: nl.NkiTensor,
        expert_down_weights_scale: nl.NkiTensor,
    ):
        """The first loads with the shared scale buffer, in the order the experts use them: the scale slots of gate(0)
        and up(0), the first expert's gate and up pieces that start a buffer, the slot of down(0) and its pieces, then
        the slot of gate(1)."""
        E_L = expert_gate_up_weights.shape[0]
        first_gate_up_weights = expert_gate_up_weights.select(dim=0, index=0)
        for projection_idx in (GATE_FUSED_IDX, UP_FUSED_IDX):
            self.load_scale_slot(expert_gate_up_weights_scale, 0, projection_idx)
            self.first_scale_slots.add((0, projection_idx))
        for projection_idx in (GATE_FUSED_IDX, UP_FUSED_IDX):
            self._load_first_pieces(projection_idx, first_gate_up_weights.select(dim=1, index=projection_idx))
        self.load_scale_slot(expert_down_weights_scale, 0, _DOWN_IDX)
        self.first_scale_slots.add((0, _DOWN_IDX))
        self._load_first_pieces(_DOWN_IDX, expert_down_weights.select(dim=0, index=0))
        if E_L > 1:
            self.load_scale_slot(expert_gate_up_weights_scale, 1, GATE_FUSED_IDX)
            self.first_scale_slots.add((1, GATE_FUSED_IDX))

    def load_scale_slot(self, scale: nl.NkiTensor, expert_idx: int, projection_idx: int):
        """Load the 16 scale rows of one projection into its slot of the shared scale buffer (``shared_scale``).
        ``scale`` is the gate/up scale tensor for gate and up, the down scale tensor for down."""
        slot = shared_scale_slot(expert_idx, projection_idx)
        if projection_idx == _DOWN_IDX:
            _load_scale_rows(
                dst=self.down_scale_buffer,
                scale=scale,
                first_expert=expert_idx,
                n_experts=1,
                projections_per_expert=1,
                first_slot=slot,
            )
            return
        _load_scale_rows(
            dst=self.down_scale_buffer,
            scale=scale,
            first_expert=expert_idx,
            n_experts=1,
            projections_per_expert=2,
            projections=(projection_idx,),
            first_slot=slot - projection_idx,
        )

    def _load_uses(self, expert_gate_up_weights, expert_down_weights, first: int, end: int, priority: int = None):
        """Issue the first loads of uses ``first .. end - 1`` of the cross-expert order (one projection per buffer)."""
        uses = sorted(self.use_index, key=self.use_index.get)[first:end]
        for projection_idx, expert_idx in uses:
            if projection_idx == _DOWN_IDX:
                weight = expert_down_weights.select(dim=0, index=expert_idx)
            else:
                weight = expert_gate_up_weights.select(dim=0, index=expert_idx).select(dim=1, index=projection_idx)
            self.first_weights[(projection_idx, expert_idx, 0)] = self._load_piece(
                projection_idx, expert_idx, 0, weight, priority=priority
            )

    def finish_first_loads(
        self,
        expert_gate_up_weights: nl.NkiTensor,
        expert_down_weights: nl.NkiTensor,
        expert_gate_up_weights_scale: nl.NkiTensor,
        expert_down_weights_scale: nl.NkiTensor,
    ) -> None:
        """Issue, at the call's start, the first loads that a ``prefetch_uses`` or ``prefetch_scales`` stream left."""
        if not self.prefetch_scales:
            self._load_gate_up_scale_group(expert_gate_up_weights_scale, buffer_idx=0)
            self._load_later_scales(expert_gate_up_weights_scale, expert_down_weights_scale)
        if self.prefetch_uses is not None:
            self._load_uses(expert_gate_up_weights, expert_down_weights, self.prefetch_uses, len(self.buffers))

    def check(
        self,
        gate_up_shape,
        down_shape,
        weight_buffers: int,
        scale_buffers: int,
        gate_up_half_major: bool,
        expert_queues: int = 1,
        use_order: bool = False,
        last_up_first: bool = False,
    ):
        """Fail unless this stream fits a call with these weights, buffers, queues and order (a prefetched stream)."""
        kernel_assert(
            self.use_order == use_order,
            f"The prefetched stream loads in the {'cross-expert' if self.use_order else 'default'} order, the call "
            f"takes cross_expert_order {use_order}: give static_expert_stream use_order equal to cross_expert_order",
        )
        kernel_assert(
            self.last_up_first == last_up_first,
            f"The prefetched stream takes last_up_first {self.last_up_first}, the call {last_up_first}: give "
            "static_expert_stream last_up_first True exactly for a call with last_expert_order 'up_first'",
        )
        kernel_assert(
            self.expert_queues == expert_queues,
            f"The prefetched stream loads on {self.expert_queues} queues, the call takes {expert_queues}",
        )
        kernel_assert(
            self.shapes == (tuple(gate_up_shape), tuple(down_shape)),
            f"The prefetched stream holds the weights {self.shapes}, the call takes {gate_up_shape} and {down_shape}",
        )
        kernel_assert(
            (len(self.buffers), len(self.gate_up_scale_buffers), self.gate_up_half_major)
            == (weight_buffers, scale_buffers, gate_up_half_major),
            f"The prefetched stream has {len(self.buffers)} weight and {len(self.gate_up_scale_buffers)} scale buffers "
            f"(half-major {self.gate_up_half_major}), the call takes {weight_buffers} and {scale_buffers} "
            f"(half-major {gate_up_half_major})",
        )

    def _load_gate_up_scale_group(
        self, expert_gate_up_weights_scale: nl.NkiTensor, buffer_idx: int, priority: int = None
    ):
        """Load the gate/up scales of the experts of group ``buffer_idx`` into its buffer."""
        E_L = expert_gate_up_weights_scale.shape[0]
        gate_up_experts_per_buffer = _SCALE_SLOTS_PER_QUADRANT // 2
        first_expert = buffer_idx * gate_up_experts_per_buffer
        _load_scale_rows(
            dst=self.gate_up_scale_buffers[buffer_idx],
            scale=expert_gate_up_weights_scale,
            first_expert=first_expert,
            n_experts=min(gate_up_experts_per_buffer, E_L - first_expert),
            projections_per_expert=2,
            priority=priority,
        )

    def _load_first_pieces(self, projection_idx: int, weight: nl.NkiTensor, priority: int = None):
        """Load the first expert's pieces of one projection that start a buffer."""
        n_pieces = len(self.first_tiles)
        for piece_idx in range(n_pieces):
            if n_pieces * projection_idx + piece_idx < self.n_first:
                self.first_weights[(projection_idx, 0, piece_idx)] = self._load_piece(
                    projection_idx, 0, piece_idx, weight, priority=priority
                )

    def _n_pieces(self, projection_idx: int) -> int:
        """Return the pieces of one projection: half-major gate/up always has two."""
        if self.gate_up_half_major and projection_idx != _DOWN_IDX:
            return 2
        return len(self.first_tiles)

    def _buffer(self, projection_idx: int, expert_idx: int, piece_idx: int = 0) -> nl.NkiTensor:
        """Return the weight buffer of one piece of one projection: the buffers rotate over the pieces (with
        ``use_order``, over the whole projections in the order of the cross-expert loop)."""
        if self.use_index is not None:
            return self.buffers[self.use_index[(projection_idx, expert_idx)] % len(self.buffers)]
        n_pieces = len(self.first_tiles)
        piece = n_pieces * (_PROJECTIONS_PER_EXPERT * expert_idx + projection_idx) + piece_idx % n_pieces
        k, w = len(self.preamble), len(self.buffers)
        if piece < k:
            return self.preamble[piece]
        if self.preamble_reuse and k + w <= piece < 2 * k + w:
            return self.preamble[piece - k - w]
        if self.preamble_reuse and piece >= 2 * k + w:
            return self.buffers[(piece - 2 * k) % w]
        return self.buffers[(piece - k) % w]

    def _queue(self, expert_idx: int, priority: int = None) -> dict:
        """Return the DMA options of one expert's weight loads: the second queue for an odd expert of two queues, and
        the DMA priority when one is given."""
        level = {} if priority is None else {"priority": priority}
        if self.expert_queues == 2 and expert_idx % 2:
            return {"dge_mode": nisa.dge_mode.hwdge, "engine": nisa.engine.sync, **level}
        return {"dge_mode": nisa.dge_mode.none, **level}

    def _load_piece(
        self, projection_idx: int, expert_idx: int, piece_idx: int, weight: nl.NkiTensor, priority: int = None
    ) -> tuple:
        """Load one piece of a projection's packed MX weights; return ``(first I/512 tile, SBUF weights)``."""
        weight_buffer = self._buffer(projection_idx, expert_idx, piece_idx)
        if self.gate_up_half_major and projection_idx != _DOWN_IDX:
            return self._load_half_major_piece(
                projection_idx, expert_idx, piece_idx, weight, weight_buffer, priority=priority
            )
        if len(self.first_tiles) == 1:
            return 0, _load_mx_weight(weight=weight, weight_buffer=weight_buffer, **self._queue(expert_idx, priority))
        n_I512_max = self.first_tiles[1]
        first = self.first_tiles[piece_idx]
        if projection_idx == _DOWN_IDX:
            # [128, n_I512, D]: I/512 tiles [first, first + n) are contiguous in each partition row.
            n = (self.first_tiles[1] if piece_idx == 0 else weight.shape[1]) - first
            weight_sb = weight_buffer.reshape((_PMAX, n_I512_max, weight.shape[2]))[:, 0:n, :]
            src = weight[:, nl.ds(first, n), :]
        else:
            # [128, D/512, I]: the I columns of chunks [first, first + n).
            n = (self.first_tiles[1] if piece_idx == 0 else weight.shape[2] // _GATE_UP_CHUNK_I) - first
            cols_max = n_I512_max * _GATE_UP_CHUNK_I
            weight_sb = weight_buffer.reshape((_PMAX, weight.shape[1], cols_max))[:, :, 0 : n * _GATE_UP_CHUNK_I]
            src = weight[:, :, nl.ds(first * _GATE_UP_CHUNK_I, n * _GATE_UP_CHUNK_I)]
        self._after_extra(weight_buffer, nisa.dma_copy(dst=weight_sb, src=src, **self._queue(expert_idx, priority)))
        return first, weight_sb

    def _after_extra(self, weight_buffer: nl.NkiTensor, load):
        """Make the first load into an extra (caller) buffer wait for ``extra_after`` (a runtime dependency)."""
        idx = next(i for i, buffer in enumerate(self.buffers) if buffer is weight_buffer)
        if idx not in self.extra_first:
            return
        self.extra_first.discard(idx)
        if self.extra_after is not None:
            # A tuple: several instructions (the 2 halves of a split latent load), the load waits for each.
            afters = self.extra_after if isinstance(self.extra_after, tuple) else (self.extra_after,)
            _instruction(load).depends_on(*(_instruction(after) for after in afters), kind=nisa.dependency.runtime)

    def _load_half_major_piece(
        self,
        projection_idx: int,
        expert_idx: int,
        piece_idx: int,
        weight: nl.NkiTensor,
        weight_buffer: nl.NkiTensor,
        priority: int = None,
    ) -> tuple:
        """Load half ``piece_idx`` of a half-major gate/up projection; return ``(first I/512 tile, SBUF weights)``.

        With half buffers each half is one load into its own buffer. Without them, half 0 loads the whole
        projection (both halves, one run per partition) and half 1 is a view of the same buffer.
        """
        half_cols = self.I // 2
        first = piece_idx * (self.n_I512 // 2)
        if len(self.first_tiles) == 1:
            if piece_idx == 0:
                nisa.dma_copy(dst=weight_buffer.reshape(weight.shape), src=weight, **self._queue(expert_idx, priority))
            halves = weight_buffer.reshape((_PMAX, 2 * self.n_D512, half_cols))
            return first, halves[:, piece_idx * self.n_D512 : (piece_idx + 1) * self.n_D512, :]
        src = self.gate_up_halves.select(dim=0, index=expert_idx).select(dim=1, index=projection_idx)
        weight_sb = weight_buffer.reshape((_PMAX, self.n_D512, half_cols))
        self._after_extra(
            weight_buffer,
            nisa.dma_copy(dst=weight_sb, src=src.select(dim=1, index=piece_idx), **self._queue(expert_idx, priority)),
        )
        return first, weight_sb

    def weight(self, projection_idx: int, expert_idx: int, weight: nl.NkiTensor) -> list:
        """Return the pieces of one projection, loading those that the stream start did not."""
        pieces = []
        for piece_idx in range(self._n_pieces(projection_idx)):
            if (projection_idx, expert_idx, piece_idx) in self.first_weights:
                pieces.append(self.first_weights[(projection_idx, expert_idx, piece_idx)])
            else:
                pieces.append(self._load_piece(projection_idx, expert_idx, piece_idx, weight))
        return pieces


def static_expert_stream(
    expert_gate_up_weights: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_gate_up_weights_scale: nl.NkiTensor,
    expert_down_weights_scale: nl.NkiTensor,
    weight_buffers: int = _PROJECTIONS_PER_EXPERT,
    scale_buffers: int = _GATE_UP_SCALE_BUFFERS,
    expert_queues: int = 1,
    use_order: bool = False,
    priority: int = None,
    prefetch_uses: int = None,
    prefetch_scales: bool = True,
    last_up_first: bool = False,
) -> _StaticExpertStream:
    """Allocate the static loop's SBUF buffers and issue their first loads now, for a later ``moe_tkg_decode`` call.

    Pass the result as ``stream`` of one call with the same weights, ``weight_buffers`` (1 to
    ``_MAX_STATIC_WEIGHT_BUFFERS`` whole buffers) and ``scale_buffers``, and ``early_loads`` True: the call's
    experts then start on loaded buffers. A caller issues it before other work (for example the attention of the
    same layer), so the first loads stream during that work. The values do not change: the loads are those of the
    call's own stream start, issued earlier. ``expert_queues`` must be that of the call. ``use_order`` (for a call
    with ``cross_expert_order``) loads the buffers in the cross-expert use order, with its 3 whole weight buffers.
    ``priority`` gives these first loads a trn3 DMA QoS level (0 to 3, 0 is the highest): at 3 they take only the DMA
    time that the higher-level loads of the overlapped work leave. The bytes do not change, so the values do not
    change. None (default) gives no level and keeps the program. ``prefetch_uses`` k (with ``use_order``, 0 to 3)
    loads only the projections of the first k uses now and leaves the others to the call's start, so only k weight
    buffers hold bytes before the call; None (default) loads all three. ``prefetch_scales`` False (with ``use_order``)
    leaves the scale groups to the call's start too; True (default) loads them now. ``last_up_first`` (with
    ``use_order``) puts up(E_L - 1) before down(E_L - 2) in the use order, for a call with ``last_expert_order``
    "up_first"; False (default) keeps the program.
    """
    kernel_assert(
        1 <= weight_buffers <= _MAX_STATIC_WEIGHT_BUFFERS,
        f"Expected 1 <= weight_buffers <= {_MAX_STATIC_WEIGHT_BUFFERS}, got {weight_buffers}",
    )
    kernel_assert(
        1 <= scale_buffers <= _GATE_UP_SCALE_BUFFERS,
        f"Expected 1 <= scale_buffers <= {_GATE_UP_SCALE_BUFFERS}, got {scale_buffers}",
    )
    kernel_assert(expert_queues in EXPERT_QUEUES, f"expert_queues must be one of {EXPERT_QUEUES}, got {expert_queues}")
    kernel_assert(
        not use_order or (weight_buffers == _PROJECTIONS_PER_EXPERT and expert_queues == 1),
        f"A stream in the order of cross_expert_order takes {_PROJECTIONS_PER_EXPERT} whole weight buffers and one "
        f"expert queue, got weight_buffers={weight_buffers} expert_queues={expert_queues}",
    )
    kernel_assert(not last_up_first or use_order, "last_up_first takes a stream in the order of cross_expert_order")
    kernel_assert(priority in (None, 0, 1, 2, 3), f"priority must be None or a DMA QoS level 0 to 3, got {priority}")
    kernel_assert(
        prefetch_uses is None or (use_order and 0 <= prefetch_uses <= weight_buffers),
        f"prefetch_uses takes a stream in the order of cross_expert_order and 0 to {weight_buffers} uses, got "
        f"{prefetch_uses}",
    )
    kernel_assert(
        prefetch_scales or (use_order and prefetch_uses),
        "prefetch_scales False takes a stream in the order of cross_expert_order that loads at least one use now",
    )
    return _StaticExpertStream(
        expert_gate_up_weights=expert_gate_up_weights,
        expert_down_weights=expert_down_weights,
        expert_gate_up_weights_scale=expert_gate_up_weights_scale,
        expert_down_weights_scale=expert_down_weights_scale,
        weight_buffers=weight_buffers,
        scale_buffers=scale_buffers,
        first_loads=True,
        use_order=use_order,
        expert_queues=expert_queues,
        first_load_priority=priority,
        **({"prefetch_uses": prefetch_uses} if prefetch_uses is not None else {}),
        **({} if prefetch_scales else {"prefetch_scales": False}),
        **({"last_up_first": True} if last_up_first else {}),
    )


def _piece(pieces: list, tile_idx: int) -> tuple:
    """Return ``(first I/512 tile, SBUF weights)`` of the piece that holds I/512 tile ``tile_idx``."""
    found = pieces[0]
    for piece in pieces:
        if piece[0] <= tile_idx:
            found = piece
    return found


def _accumulate_expert(dst, expert_out_f32, affinity_f32_sb, expert_idx: int, accum, first_adds_zero: bool):
    """Write ``expert_out * affinity[expert] + accum`` of one H slice to ``dst`` (VectorE). With ``first_adds_zero``,
    expert 0 adds a scalar +0.0 in place of the zero-filled ``accum``: the same FP32 add, so the same bits."""
    if first_adds_zero and expert_idx == 0:
        nisa.tensor_scalar(
            dst=dst,
            data=expert_out_f32,
            op0=nl.multiply,
            operand0=affinity_f32_sb[:, 0:1],
            op1=nl.add,
            operand1=0.0,
            engine=nisa.engine.vector,
        )
        return
    nisa.scalar_tensor_tensor(
        dst=dst,
        data=expert_out_f32,
        op0=nl.multiply,
        operand0=affinity_f32_sb[:, expert_idx : expert_idx + 1],
        op1=nl.add,
        operand1=accum,
    )


def _all_experts_static(
    input_quant_sb: nl.NkiTensor,
    input_scale_sb: nl.NkiTensor,
    output_accum_sb: nl.NkiTensor,
    local_affinities: nl.NkiTensor,
    expert_gate_up_weights: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_gate_up_weights_scale: nl.NkiTensor,
    expert_down_weights_scale: nl.NkiTensor,
    beta: float,
    linear_beta: float,
    T: int,
    T_physical: int,
    stream: _StaticExpertStream,
    output: nl.NkiTensor,
    column_groups_out: nl.NkiTensor = None,
    transpose_lookahead: int = _TRANSPOSE_LOOKAHEAD_CHUNKS,
    cross_expert_order: bool = False,
    static_alt_tiles: bool | str = False,
    cross_expert_transpose_blocks: bool = False,
    transpose_block: bool = False,
    affinity_f32_sb: nl.NkiTensor = None,
    first_adds_zero: bool = False,
    last_expert_order: str = "default",
    affinity_sbuf_copy: bool = False,
):
    """Accumulate every local expert with static DMAs, whatever its affinities.

    The weights stream through one buffer per projection, so each load waits
    only for the same projection of the previous expert, and the DMAs of
    consecutive experts overlap with their matmuls. With fewer buffers,
    projection p of expert e uses buffer (3e + p) % weight_buffers. With
    ``half_weight_buffers`` each projection streams in two halves (the
    pieces of ``stream``), and each matmul reads the piece of its I/512 tile.
    ``stream`` holds the buffers and the first loads that its start issued;
    the loop issues the others. The last expert's epilogue writes the BF16
    ``output`` tiles and stores them. With ``column_groups_out`` (``output``
    None) it writes one SBUF ``[T, D]`` tile, whose bytes then move to
    ``column_groups_out`` (``_to_column_groups``). ``cross_expert_order``
    issues the experts in the cross-expert order (``_cross_expert_steps``),
    with ``cross_expert_transpose_blocks`` its transposes in tile blocks, or
    with ``transpose_block`` all of an expert's transposes in one block.
    ``static_alt_tiles`` takes the gate sigmoid, up tanh, PSUM copy and
    eviction tiles (or the roles of its value, ``_STATIC_ALT_TILE_ROLES``) from
    two buffers each (``_StaticLoopTiles``), and names the engine of each PSUM
    copy (``_copy_engine``). ``affinity_f32_sb`` (``moe_tkg_decode`` ``psum_consume``): the FP32 ``[T, E_L]``
    affinity columns of the accumulator's SBUF tile; each multiply-add then reads its down PSUM tile, with no copy.
    None keeps the copies and a separate affinity tile. ``last_expert_order``: ``LAST_EXPERT_ORDERS``
    (``_cross_expert_steps``). ``affinity_sbuf_copy`` (``moe_tkg_decode``): an SBUF ``local_affinities`` goes into the
    FP32 affinity tile with one VectorE copy, with no SBUF -> SBUF DMA.
    """
    D = expert_down_weights.shape[-1]
    E_L = expert_gate_up_weights.shape[0]
    output_tiles = _OutputTiles(output=output, T=T, sbuf_columns=D if output is None else 0)
    psum_consume = affinity_f32_sb is not None
    kernel_assert(
        not psum_consume or not static_alt_tiles or static_alt_tiles in STATIC_ALT_TILE_PSUM_CONSUME,
        f"psum_consume takes the static loop without static_alt_tiles, or with {STATIC_ALT_TILE_PSUM_CONSUME}",
    )

    if affinity_sbuf_copy and local_affinities.buffer == nl.sbuf:
        # The affinities are already in SBUF: one VectorE copy (the engine of the copy below), no Sync-engine DMA.
        # The padding rows take the +0.0 the copy of the zeroed raw rows gives.
        if affinity_f32_sb is None:
            affinity_f32_sb = nl.ndarray((T, E_L), dtype=nl.float32, buffer=nl.sbuf)
        if T > T_physical:
            nisa.memset(dst=affinity_f32_sb[T_physical:T, :], value=0.0, engine=nisa.engine.vector)
        nisa.tensor_copy(dst=affinity_f32_sb[:T_physical, :], src=local_affinities, engine=nisa.engine.vector)
    else:
        # One contiguous load for all experts: a per-expert column load is an
        # element-granular DMA, which goes to the software DGE and can stall the
        # in-order VectorE stream behind it for a long time.
        affinity_raw_sb = nl.ndarray((T, E_L), dtype=local_affinities.dtype, buffer=nl.sbuf)
        if T > T_physical:
            nisa.memset(dst=affinity_raw_sb, value=0.0)
        nisa.dma_copy(dst=affinity_raw_sb[:T_physical, :], src=local_affinities, dge_mode=nisa.dge_mode.none)
        if affinity_f32_sb is None:
            affinity_f32_sb = nl.ndarray((T, E_L), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=affinity_f32_sb, src=affinity_raw_sb)
    tiles = (
        _StaticLoopTiles(
            T,
            D,
            _STATIC_ALT_TILE_ROLES[static_alt_tiles],
            **(
                {"gate_up_depth": _STATIC_ALT_TILE_GATE_UP_DEPTH[static_alt_tiles]}
                if static_alt_tiles in _STATIC_ALT_TILE_GATE_UP_DEPTH
                else {}
            ),
        )
        if static_alt_tiles
        else None
    )
    if cross_expert_order:

        def consume_of(expert_idx):
            def consume(start, size, H_slice, expert_out_f32):
                sum_dst_sb = output_accum_sb[:, H_slice]
                if expert_idx == E_L - 1:
                    # The last expert writes BF16 output columns (see _OutputTiles).
                    sum_dst_sb = output_tiles.columns(start, size)
                _accumulate_expert(
                    sum_dst_sb,
                    expert_out_f32,
                    affinity_f32_sb,
                    expert_idx,
                    output_accum_sb[:, H_slice],
                    first_adds_zero,
                )
                if expert_idx == E_L - 1:
                    output_tiles.store_when_written(start, size)

            return consume

        steps = [
            _ExpertSteps(
                stream=stream,
                expert_idx=expert_idx,
                input_quant_sb=input_quant_sb,
                input_scale_sb=input_scale_sb,
                expert_gate_up_weights=expert_gate_up_weights,
                expert_down_weights=expert_down_weights,
                expert_gate_up_weights_scale=expert_gate_up_weights_scale,
                expert_down_weights_scale=expert_down_weights_scale,
                beta=beta,
                linear_beta=linear_beta,
                T=T,
                tiles=tiles,
                scale_slots=len(stream.gate_up_scale_buffers) == 1,
            )
            for expert_idx in range(E_L)
        ]
        _cross_expert_steps(
            steps,
            consume_of,
            transpose_blocks=cross_expert_transpose_blocks,
            transpose_block=transpose_block,
            psum_consume=psum_consume,
            **({"last_expert_order": last_expert_order} if last_expert_order != "default" else {}),
        )
        if column_groups_out is not None:
            _to_column_groups(output_tiles.tiles[0], T_physical, column_groups_out)
        return
    for expert_idx in nl.sequential_range(E_L):

        def consume(start, size, H_slice, expert_out_f32, expert_idx=expert_idx):
            sum_dst_sb = output_accum_sb[:, H_slice]
            if expert_idx == E_L - 1:
                # The last expert writes BF16 output columns (see _OutputTiles).
                sum_dst_sb = output_tiles.columns(start, size)
            _accumulate_expert(
                sum_dst_sb, expert_out_f32, affinity_f32_sb, expert_idx, output_accum_sb[:, H_slice], first_adds_zero
            )
            if expert_idx == E_L - 1:
                output_tiles.store_when_written(start, size)

        _expert_ffn(
            stream=stream,
            expert_idx=expert_idx,
            input_quant_sb=input_quant_sb,
            input_scale_sb=input_scale_sb,
            expert_gate_up_weights=expert_gate_up_weights,
            expert_down_weights=expert_down_weights,
            expert_gate_up_weights_scale=expert_gate_up_weights_scale,
            expert_down_weights_scale=expert_down_weights_scale,
            beta=beta,
            linear_beta=linear_beta,
            T=T,
            consume=consume,
            tiles=tiles,
            transpose_lookahead=transpose_lookahead,
            copy_engines=None if tiles is None else _copy_engine,
            psum_consume=psum_consume,
        )
    if column_groups_out is not None:
        _to_column_groups(output_tiles.tiles[0], T_physical, column_groups_out)


def _after(later, earlier):
    """Keep ``later`` after ``earlier`` on their engine: a schedule edge, no semaphore wait (both on one engine)."""
    _instruction(later).depends_on(_instruction(earlier), kind=nisa.dependency.schedule)


def _copy_engine(tile_h: int):
    """The engine of an H tile's PSUM copy in the cross-expert order: ScalarE for even tiles, VectorE for odd ones
    (4 and 3 of the 7 tiles at D = 3584; VectorE also runs every tile's affinity product)."""
    return nisa.engine.scalar if tile_h % 2 == 0 else nisa.engine.vector


def _cross_expert_steps(
    steps: list,
    consume_of,
    transpose_blocks: bool = False,
    transpose_block: bool = False,
    psum_consume: bool = False,
    last_expert_order: str = "default",
):
    """Issue the experts of the static loop in the cross-expert order.

    The gate matmuls of expert e + 1 run on TensorE between the up and the down matmuls of expert e, while ScalarE and
    VectorE evict, quantize and transpose expert e's activation: up(e), the transposes of activation chunk 0,
    gate(e + 1) chunk 0, the transposes of chunk 1, the other gate(e + 1) chunks, down(e), up(e + 1). Each expert
    issues the same instructions as in expert order, with the same PSUM accumulation order, and the affinity products
    still add the experts in order, so the result does not change. TensorE schedule edges keep the order: gate(e + 1)
    after up(e) and after the chunk transposes before it, each chunk's transposes after the gate chunk before them,
    every down(e) accumulation after gate(e + 1), and up(e + 1) after down(e). On ScalarE each gate(e + 1) chunk's
    sigmoid follows the eviction of the activation chunk before it. The SiTU products of up(e) are traced before any
    of gate(e + 1), so the in-order VectorE products keep their time order. With ``transpose_blocks`` the transposes
    and gate(e + 1) chunks of each expert alternate in tile blocks instead (``_transpose_blocks_and_gates``), and every
    down(e) accumulation also follows the last transpose.

    ``steps`` holds one ``_ExpertSteps`` per local expert; ``consume_of(e)`` returns expert e's ``consume``.
    ``transpose_block`` issues each expert's transposes in one block (``_blocked_steps``). ``last_expert_order``
    (``LAST_EXPERT_ORDERS``, with ``transpose_block``): "chunked" issues the last expert's transposes chunk by chunk;
    "up_first" issues the last two experts as ``_last_two_up_first``.
    """
    first = steps[0]
    first.gate_start()
    first_gates = [first.gate_chunk(chunk_idx) for chunk_idx in range(first.n_chunks)]
    up = first.up()
    down = None
    for expert_idx, expert in enumerate(steps):
        nxt = steps[expert_idx + 1] if expert_idx + 1 < len(steps) else None
        if transpose_block:
            if last_expert_order == "up_first" and expert_idx == len(steps) - 2:
                _last_two_up_first(expert, nxt, up, consume_of, expert_idx, psum_consume)
                break
            chunked = last_expert_order == "chunked" and nxt is None
            down_after = _blocked_steps(expert, nxt, up["last"], chunked=chunked)
            expert.down_start()
            down = expert.down(consume=consume_of(expert_idx), copy_engines=_copy_engine, psum_consume=psum_consume)
            for matmul in down["firsts"]:
                for earlier in down_after:
                    _after(matmul, earlier)
            if nxt is not None:
                up = nxt.up()
                for matmul in up["firsts"]:
                    _after(matmul, down["last"])
            continue
        if transpose_blocks:
            # The matmul groups that the edges do not yet order before up(e): gate(0), or the H tiles of down(e - 1)
            # (up(e) follows only its last matmul).
            before = [gate["last"] for gate in first_gates] if down is None else down["lasts"]
            gate_last, last_transpose = _transpose_blocks_and_gates(expert, nxt, up, before)
        else:
            n_gate = nxt.n_chunks if nxt is not None else 0
            gate_last = None
            for idx in range(max(expert.n_transpose_chunks, n_gate)):
                issued = None
                if idx < expert.n_transpose_chunks:
                    issued = expert.transpose(idx, after=gate_last)
                if idx < n_gate:
                    if idx == 0:
                        nxt.gate_start()
                    chunk = nxt.gate_chunk(idx)
                    if idx == 0:
                        _after(chunk["first"], up["last"])
                    if issued is not None:
                        # Chunk 0's transposes are ready when up(e) ends: the last one ahead of gate(e + 1) is enough.
                        for transpose in issued["transposes"][-1:] if idx == 0 else issued["transposes"]:
                            _after(chunk["first"], transpose)
                        _after(chunk["sigmoid"], issued["eviction"])
                    gate_last = chunk["last"]
        expert.down_start()
        down = expert.down(consume=consume_of(expert_idx), copy_engines=_copy_engine, psum_consume=psum_consume)
        if transpose_blocks:
            for matmul in down["firsts"]:
                _after(matmul, last_transpose)
        if nxt is None:
            continue
        for matmul in down["firsts"]:
            _after(matmul, gate_last)
        up = nxt.up()
        for matmul in up["firsts"]:
            _after(matmul, down["last"])


def _transpose_blocks_and_gates(expert: "_ExpertSteps", nxt, up: dict, before=()) -> tuple:
    """Issue the activation transposes of ``expert`` and the gate chunks of ``nxt`` (the next expert, None after the
    last one) in tile blocks; return the last gate(e + 1) matmul (None without ``nxt``) and the last transpose.
    ``before`` (the last matmuls of the groups that the edges do not yet order before up(e)) also precede the first
    block.

    The default edges order only gate chunk 0 after the last transpose of chunk 0, and gate chunk 1 after chunk 1's
    transposes, so the compiler puts the matmuls of gate chunk 2 between chunk 0's transposes and the first down(e)
    matmuls between chunk 1's: single transposes between MX matmuls, each of which costs a switch out of and back into
    the MX matmul mode (more TensorE time per transpose than inside a block).

    Here TensorE runs one block per I/512 tile (its 4 transposes, the output of one SiTU product) and then one gate
    chunk: tile 0, gate chunk 0, tile 1, gate chunk 1, ... (I = 1536: 3 tiles, 3 gate chunks). Schedule edges keep
    every matmul out of a block: the block's transposes follow the matmul group before it (every up(e) chunk, and
    ``before``, for tile 0, else the gate chunk before), the next gate chunk's first matmul follows the block's last
    transpose, and the gate chunks run in order after up(e). So VectorE has one gate chunk's matmuls to finish the
    next tile's SiTU product.

    An activation chunk is evicted and quantized after its last tile (``_ExpertSteps.transpose_tile``). On ScalarE the
    sigmoid of the gate chunk issued next follows that eviction, which keeps the rule of the cross-expert order (each
    gate chunk's sigmoid follows the eviction of the activation chunk before it on TensorE). The first gate chunk has no
    evicted chunk before it, so its sigmoid takes no edge.
    """
    n_gate = nxt.n_chunks if nxt is not None else 0
    group_last, gate_last, last_transpose, eviction = None, None, None, None
    for idx in range(max(expert.n_I512, n_gate)):
        block_last = None
        if idx < expert.n_I512:
            issued = expert.transpose_tile(idx, after=group_last)
            if group_last is None:
                for transpose in issued["transposes"]:
                    for matmul in (*up["lasts"], *before):
                        _after(transpose, matmul)
            block_last = last_transpose = issued["transposes"][-1]
            eviction = issued.get("eviction", eviction)
        if idx < n_gate:
            if idx == 0:
                nxt.gate_start()
            chunk = nxt.gate_chunk(idx)
            _after(chunk["first"], up["last"] if gate_last is None else gate_last)
            if block_last is not None:
                _after(chunk["first"], block_last)
            if eviction is not None:
                _after(chunk["sigmoid"], eviction)
                eviction = None
            gate_last = group_last = chunk["last"]
    return gate_last, last_transpose


def _blocked_steps(expert: "_ExpertSteps", nxt: "_ExpertSteps", up_last, chunked: bool = False) -> list:
    """Issue gate(e + 1) chunk 0, every activation transpose of ``expert`` in one block, then the other gate(e + 1)
    chunks (``nxt`` None for the last expert: the block only); return the TensorE instructions that the down matmuls
    of ``expert`` follow.

    Each step follows the one before it on TensorE, so no MX matmul runs inside the block: gate chunk 0 follows up(e),
    every transpose follows gate chunk 0, gate chunk 1 follows every transpose and each later gate chunk the one
    before it. On ScalarE gate chunk 1's sigmoid follows the evictions of the block. With ``chunked`` each chunk's
    transposes also follow the last transpose of the chunk before it, so the first chunk's eviction and quantize do
    not wait for the transposes of a later chunk (``last_expert_order``).
    """
    n_gate = nxt.n_chunks if nxt is not None else 0
    last = up_last
    if n_gate:
        nxt.gate_start()
        chunk = nxt.gate_chunk(0)
        _after(chunk["first"], last)
        last = chunk["last"]
    block, evictions = [], []
    for idx in range(expert.n_transpose_chunks):
        issued = expert.transpose(idx, after=block[-1] if chunked and block else last)
        block += issued["transposes"]
        evictions.append(issued["eviction"])
    for idx in range(1, n_gate):
        chunk = nxt.gate_chunk(idx)
        for earlier in block or [last]:
            _after(chunk["first"], earlier)
        for eviction in evictions:
            _after(chunk["sigmoid"], eviction)
        last, block, evictions = chunk["last"], [], []
    return block or [last]


def _last_two_up_first(
    expert: "_ExpertSteps", last: "_ExpertSteps", up: dict, consume_of, expert_idx: int, psum_consume
):
    """Issue experts E_L - 2 (``expert``, whose up(e) is ``up``) and E_L - 1 (``last``) in the order of
    ``last_expert_order`` "up_first": gate(E_L - 1) chunk 0, the transposes of E_L - 2, the other gate(E_L - 1) chunks
    (``_blocked_steps``), up(E_L - 1), the transposes of E_L - 1 chunk by chunk, down(E_L - 2), down(E_L - 1).

    Down(E_L - 2) then runs on TensorE while ScalarE and VectorE evict and quantize the last activation, which no
    TensorE work covers in the default order. TensorE schedule edges keep the order: up(E_L - 1) after the gate chunks,
    down(E_L - 2) after the transposes of both experts, down(E_L - 1) after down(E_L - 2). The steps issue the same
    instructions as in the default order, and the affinity products of down(E_L - 1) read the accumulator columns that
    those of down(E_L - 2) wrote, so the result does not change. The ring loads up(E_L - 1) before down(E_L - 2) (the
    trace order) into the buffers of the swapped use order (``_cross_expert_uses`` ``last_up_first``).
    """
    down_after = _blocked_steps(expert, last, up["last"])
    last_up = last.up()
    for matmul in last_up["firsts"]:
        for earlier in down_after:
            _after(matmul, earlier)
    last_after = _blocked_steps(last, None, last_up["last"], chunked=True)
    expert.down_start()
    down = expert.down(consume=consume_of(expert_idx), copy_engines=_copy_engine, psum_consume=psum_consume)
    for matmul in down["firsts"]:
        for earlier in (*down_after, *last_after):
            _after(matmul, earlier)
    last.down_start()
    last_down = last.down(consume=consume_of(expert_idx + 1), copy_engines=_copy_engine, psum_consume=psum_consume)
    for matmul in last_down["firsts"]:
        _after(matmul, down["last"])


def _column_group_block_tokens(D: int) -> int:
    """Tokens of the column-group layout (D/128 BF16 columns each) that fill whole 32-byte SBUF blocks: 4 at D = 3584."""
    token_bytes = 2 * (D // _PMAX)
    return _SBUF_BLOCK_BYTES // math.gcd(_SBUF_BLOCK_BYTES, token_bytes)


def _to_column_groups(output_sb: nl.NkiTensor, T_physical: int, column_groups_out: nl.NkiTensor):
    """Move the BF16 ``[T, D]`` SBUF output to the column-group layout ``[128, T_physical, D/128]``.

    Group g holds columns ``[g G, g G + G)`` (G = D/128, even), so 32-bit word j
    of group g packs columns ``g G + 2 j`` and ``g G + 2 j + 1`` of a token. One
    TensorE transpose per word index j moves the stride-``G/2`` word view
    ``[T, 128]`` to PSUM ``[128, T]`` (bit-accurate on trn3), and 32-bit copies
    write each token's ``G/2`` words contiguously. Every transpose slot is a
    power of two of at most 512 bytes, so no transpose crosses a PSUM bank. At
    64 tokens or more the Vector and Scalar engines copy about one half each,
    split on a whole 32-byte SBUF block (a multiple of 4 tokens at D = 3584):
    a trn3 SBUF write of part of a block is a read-modify-write of the whole
    block, so two unordered engines must not write the same block.
    """
    T, D = output_sb.shape
    words = D // (2 * _PMAX)
    slot = 1 << (T_physical - 1).bit_length()
    banks = div_ceil(words * slot * 4, _PSUM_BANK_BYTES)
    word_view = output_sb.view(nl.float32).reshape((T, _PMAX, words))
    psum = nl.ndarray((_PMAX, banks * _PSUM_BANK_BYTES // 4), dtype=nl.float32, buffer=nl.psum)
    psum_words = psum[:, 0 : words * slot].reshape((_PMAX, words, slot))
    for word in range(words):
        nisa.nc_transpose(
            dst=psum_words[:, word, 0:T_physical], data=word_view[0:T_physical, :, word], engine=nisa.engine.tensor
        )
    dst_words = column_groups_out.reshape((_PMAX, T_physical * 2 * words)).view(nl.float32)
    dst_words = dst_words.reshape((_PMAX, T_physical, words))
    block_tokens = _column_group_block_tokens(D)
    bounds = [0, T_physical]
    if T_physical >= 64:
        bounds.insert(1, T_physical // 2 // block_tokens * block_tokens)
    for half, (first, last) in enumerate(zip(bounds, bounds[1:])):
        nisa.tensor_copy(
            dst=dst_words[:, first:last, :],
            src=psum.ap(pattern=[[psum.shape[1], _PMAX], [1, last - first], [slot, words]], offset=first),
            engine=(nisa.engine.vector, nisa.engine.scalar)[half],
        )


def _expert_ffn(
    stream: "_StaticExpertStream",
    expert_idx: int,
    input_quant_sb: nl.NkiTensor,
    input_scale_sb: nl.NkiTensor,
    expert_gate_up_weights: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_gate_up_weights_scale: nl.NkiTensor,
    expert_down_weights_scale: nl.NkiTensor,
    beta: float,
    linear_beta: float,
    T: int,
    consume,
    loaded: dict = None,
    tiles: "_AlternatingTiles" = None,
    transpose_lookahead: int = _TRANSPOSE_LOOKAHEAD_CHUNKS,
    copy_engines=None,
    psum_consume: bool = False,
):
    """Run the gate/up, SiTU and down projections of one local expert on the T quantized tokens.

    The weights and scales come through ``stream`` in expert order (the buffers of ``_StaticExpertStream``). For each
    down H tile, ``consume(start, size, H_slice, expert_out_f32)`` takes the FP32 [T, size] output of columns
    ``[start, start + size)``: the caller scales and accumulates it. ``loaded`` (from ``_load_expert``) gives the
    weight pieces and the scale views of the expert, loaded before: the call then issues no load. ``tiles`` gives the
    SiTU and output tiles from two alternating buffers each. ``transpose_lookahead`` is the activation chunks
    transposed before the first down matmul (``moe_tkg_decode``); below the default, each later chunk's transposes
    follow the first down matmul of the chunk before it (a TensorE schedule edge). ``copy_engines`` names the engine
    of each H tile's PSUM copy (``_ExpertSteps.down``); None leaves the choice to the compiler. ``psum_consume``
    gives ``consume`` the FP32 PSUM tile itself, with no copy (``_ExpertSteps.down``).
    """
    steps = _ExpertSteps(
        stream=stream,
        expert_idx=expert_idx,
        input_quant_sb=input_quant_sb,
        input_scale_sb=input_scale_sb,
        expert_gate_up_weights=expert_gate_up_weights,
        expert_down_weights=expert_down_weights,
        expert_gate_up_weights_scale=expert_gate_up_weights_scale,
        expert_down_weights_scale=expert_down_weights_scale,
        beta=beta,
        linear_beta=linear_beta,
        T=T,
        loaded=loaded,
        tiles=tiles,
    )
    steps.gate_start()
    for chunk_idx in nl.sequential_range(steps.n_chunks):
        steps.gate_chunk(chunk_idx)
    steps.up()
    steps.down_start()
    for chunk_idx in range(min(transpose_lookahead, steps.n_transpose_chunks)):
        steps.transpose(chunk_idx)
    steps.down(
        consume=consume, transpose_lookahead=transpose_lookahead, copy_engines=copy_engines, psum_consume=psum_consume
    )


class _ExpertSteps:
    """The steps of one local expert of the static loop, issued in the order that the caller picks.

    ``_expert_ffn`` issues them in expert order: ``gate_start``, each ``gate_chunk``, ``up``, ``down_start``, the
    ``transpose`` of the lookahead chunks, then ``down``. The cross-expert order of ``_all_experts_static`` issues the
    gate chunks of the next expert between this expert's transposes and its ``down``. A step issues the same
    instructions in either order, and returns the TensorE and ScalarE instructions that the cross-expert order keeps
    in place with schedule edges. Constructing the steps issues no instruction.
    """

    def __init__(
        self,
        stream: "_StaticExpertStream",
        expert_idx: int,
        input_quant_sb: nl.NkiTensor,
        input_scale_sb: nl.NkiTensor,
        expert_gate_up_weights: nl.NkiTensor,
        expert_down_weights: nl.NkiTensor,
        expert_gate_up_weights_scale: nl.NkiTensor,
        expert_down_weights_scale: nl.NkiTensor,
        beta: float,
        linear_beta: float,
        T: int,
        loaded: dict = None,
        tiles: "_AlternatingTiles" = None,
        scale_slots: bool = False,
    ):
        self.stream, self.expert_idx = stream, expert_idx
        # With scale_slots, gate_start and up each load only their own projection's scale slot (see _cross_expert_steps).
        self.scale_slots = scale_slots
        # The shared scale buffer of the static loop (``_StaticExpertStream`` ``shared_scale``).
        self.shared_scale = getattr(stream, "shared_scale", False)
        self.input_quant_sb, self.input_scale_sb = input_quant_sb, input_scale_sb
        self.expert_gate_up_weights, self.expert_down_weights = expert_gate_up_weights, expert_down_weights
        self.expert_gate_up_weights_scale = expert_gate_up_weights_scale
        self.expert_down_weights_scale = expert_down_weights_scale
        self.beta, self.linear_beta, self.T = beta, linear_beta, T
        self.loaded, self.tiles = loaded, tiles
        self.D = expert_down_weights.shape[-1]
        self.E_L = expert_gate_up_weights.shape[0]
        self.I = expert_gate_up_weights.shape[-1]
        self.n_D512 = self.D // MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
        self.n_I512 = self.I // MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
        self.n_chunks = div_ceil(self.I, _GATE_UP_CHUNK_I)
        self.tiles_per_chunk = _TRANSPOSE_CHUNK_I // MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
        self.n_transpose_chunks = div_ceil(self.n_I512, self.tiles_per_chunk)
        self.gate_terms_f32 = []
        self.quantized_chunks = []

    def gate_start(self):
        """Issue the gate/up scale load of the expert's group (if the stream start did not) and its gate weights."""
        stream, expert_idx, loaded = self.stream, self.expert_idx, self.loaded
        gate_up_scale_buffers = stream.gate_up_scale_buffers
        scale_buffers = len(gate_up_scale_buffers)
        gate_up_experts_per_buffer = _SCALE_SLOTS_PER_QUADRANT // 2
        gate_up_group_idx, gate_up_group_expert = divmod(expert_idx, gate_up_experts_per_buffer)
        gate_up_scale_buffer = gate_up_scale_buffers[gate_up_group_idx % scale_buffers]
        # The stream start may have loaded the group of each buffer's first use.
        if self.shared_scale:
            if (expert_idx, GATE_FUSED_IDX) not in stream.first_scale_slots:
                stream.load_scale_slot(self.expert_gate_up_weights_scale, expert_idx, GATE_FUSED_IDX)
        elif self.scale_slots:
            self._load_scale_slot(gate_up_scale_buffer, GATE_FUSED_IDX)
        elif loaded is None and gate_up_group_expert == 0 and gate_up_group_idx >= stream.first_gate_up_groups:
            _load_scale_rows(
                dst=gate_up_scale_buffer,
                scale=self.expert_gate_up_weights_scale,
                first_expert=expert_idx,
                n_experts=min(gate_up_experts_per_buffer, self.E_L - expert_idx),
                projections_per_expert=2,
            )
        self.gate_up_scale_buffer = gate_up_scale_buffer
        self.gate_up_scale_sb = gate_up_scale_buffer.reshape((_PMAX, self.n_D512, self.I))
        self.gate_up_scale_offset = 2 * _SCALE_ROWS_PER_QUADRANT * gate_up_group_expert
        self.up_scale_offset = self.gate_up_scale_offset + _SCALE_ROWS_PER_QUADRANT
        if self.shared_scale:
            self.gate_up_scale_offset = _SCALE_ROWS_PER_QUADRANT * shared_scale_slot(expert_idx, GATE_FUSED_IDX)
            self.up_scale_offset = _SCALE_ROWS_PER_QUADRANT * shared_scale_slot(expert_idx, UP_FUSED_IDX)
        if loaded is not None:
            self.gate_up_scale_sb, self.gate_up_scale_offset = loaded["gate_up_scale"], 0
            self.up_scale_offset = _SCALE_ROWS_PER_QUADRANT

        self.gate_pieces = (
            loaded["gate"]
            if loaded is not None
            else stream.weight(
                GATE_FUSED_IDX,
                expert_idx,
                self.expert_gate_up_weights.select(dim=0, index=expert_idx).select(dim=1, index=GATE_FUSED_IDX),
            )
        )

    def _load_scale_slot(self, gate_up_scale_buffer: nl.NkiTensor, projection_idx: int):
        """Load one projection's gate/up scale rows into its slot of the buffer (slot 2 (e % 2) + projection, the
        slot of the group load), unless the stream start loaded the expert's group. Each such load waits only for the
        reader of the same slot two experts before."""
        gate_up_experts_per_buffer = _SCALE_SLOTS_PER_QUADRANT // 2
        gate_up_group_idx, gate_up_group_expert = divmod(self.expert_idx, gate_up_experts_per_buffer)
        if gate_up_group_idx < self.stream.first_gate_up_groups:
            return
        _load_scale_rows(
            dst=gate_up_scale_buffer,
            scale=self.expert_gate_up_weights_scale,
            first_expert=self.expert_idx,
            n_experts=1,
            projections_per_expert=2,
            projections=(projection_idx,),
            first_slot=2 * gate_up_group_expert,
        )

    def gate_chunk(self, chunk_idx: int) -> dict:
        """Issue one gate chunk (its matmuls and SiTU gate term); return its first and last matmul and its sigmoid."""
        T, I, tiles, beta = self.T, self.I, self.tiles, self.beta
        gate_weight_scale_sb = self.gate_up_scale_sb[self.gate_up_scale_offset :, :, :]
        chunk_size = min(_GATE_UP_CHUNK_I, I - chunk_idx * _GATE_UP_CHUNK_I)
        first_chunk, gate_weight_sb = _piece(self.gate_pieces, chunk_idx)
        matmuls = []
        gate_psum = _project_chunk(
            input_quant_sb=self.input_quant_sb,
            input_scale_sb=self.input_scale_sb,
            weight_sb=gate_weight_sb,
            weight_scale_sb=gate_weight_scale_sb,
            n_H512=self.n_D512,
            chunk_idx=chunk_idx,
            chunk_size=chunk_size,
            T=T,
            weight_first_chunk=first_chunk,
            matmuls=matmuls,
        )
        gate_sigmoid_f32 = _tile(tiles, "gate_sigmoid", T, chunk_size)
        sigmoid = nisa.activation(
            dst=gate_sigmoid_f32,
            op=nl.sigmoid,
            data=gate_psum,
            engine=nisa.engine.scalar,
        )
        gate_term_f32 = nl.ndarray((T, chunk_size), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=gate_term_f32,
            op=nl.tanh,
            data=gate_psum,
            scale=1.0 / beta,
            engine=nisa.engine.scalar,
        )
        nisa.scalar_tensor_tensor(
            dst=gate_term_f32,
            data=gate_term_f32,
            op0=nl.multiply,
            operand0=beta,
            op1=nl.multiply,
            operand1=gate_sigmoid_f32,
        )
        self.gate_terms_f32.append(gate_term_f32)
        return {"first": matmuls[0], "last": matmuls[-1], "sigmoid": sigmoid}

    def up(self) -> dict:
        """Issue the up weights and the up chunks (matmuls and SiTU products into the FP16 activation); return the
        first and the last matmul of each chunk, and the last matmul."""
        T, I, tiles, linear_beta = self.T, self.I, self.tiles, self.linear_beta
        loaded = self.loaded
        if self.shared_scale:
            if (self.expert_idx, UP_FUSED_IDX) not in self.stream.first_scale_slots:
                self.stream.load_scale_slot(self.expert_gate_up_weights_scale, self.expert_idx, UP_FUSED_IDX)
        elif self.scale_slots:
            self._load_scale_slot(self.gate_up_scale_buffer, UP_FUSED_IDX)
        up_pieces = (
            loaded["up"]
            if loaded is not None
            else self.stream.weight(
                UP_FUSED_IDX,
                self.expert_idx,
                self.expert_gate_up_weights.select(dim=0, index=self.expert_idx).select(dim=1, index=UP_FUSED_IDX),
            )
        )
        up_weight_scale_sb = self.gate_up_scale_sb[self.up_scale_offset :, :, :]
        self.activation_token_major = nl.ndarray((T, I), dtype=nl.float16, buffer=nl.sbuf)
        firsts, lasts, matmuls = [], [], []
        for chunk_idx in nl.sequential_range(self.n_chunks):
            chunk_size = min(_GATE_UP_CHUNK_I, I - chunk_idx * _GATE_UP_CHUNK_I)
            first_chunk, up_weight_sb = _piece(up_pieces, chunk_idx)
            up_psum = _project_chunk(
                input_quant_sb=self.input_quant_sb,
                input_scale_sb=self.input_scale_sb,
                weight_sb=up_weight_sb,
                weight_scale_sb=up_weight_scale_sb,
                n_H512=self.n_D512,
                chunk_idx=chunk_idx,
                chunk_size=chunk_size,
                T=T,
                weight_first_chunk=first_chunk,
                matmuls=matmuls,
            )
            firsts.append(matmuls[-self.n_D512])
            lasts.append(matmuls[-1])
            up_tanh_f32 = _tile(tiles, "up_tanh", T, chunk_size)
            nisa.activation(
                dst=up_tanh_f32,
                op=nl.tanh,
                data=up_psum,
                scale=1.0 / linear_beta,
                engine=nisa.engine.scalar,
            )
            if not getattr(self.stream, "up_product_split", False):
                nisa.scalar_tensor_tensor(
                    dst=self.activation_token_major[:, nl.ds(chunk_idx * _GATE_UP_CHUNK_I, chunk_size)],
                    data=up_tanh_f32,
                    op0=nl.multiply,
                    operand0=linear_beta,
                    op1=nl.multiply,
                    operand1=self.gate_terms_f32[chunk_idx],
                )
            else:
                # up_product_split (moe_tkg_decode): one product for each 128 columns, so a transpose waits for its own.
                for first_col in range(0, chunk_size, _PMAX):
                    width = min(_PMAX, chunk_size - first_col)
                    nisa.scalar_tensor_tensor(
                        dst=self.activation_token_major[:, nl.ds(chunk_idx * _GATE_UP_CHUNK_I + first_col, width)],
                        data=up_tanh_f32[:, first_col : first_col + width],
                        op0=nl.multiply,
                        operand0=linear_beta,
                        op1=nl.multiply,
                        operand1=self.gate_terms_f32[chunk_idx][:, first_col : first_col + width],
                    )
        return {"firsts": firsts, "lasts": lasts, "last": matmuls[-1]}

    def down_start(self):
        """Issue the down scale load of the expert's group (if due) and the expert's down weights."""
        stream, expert_idx, loaded, D = self.stream, self.expert_idx, self.loaded, self.D
        down_scale_buffer = stream.down_scale_buffer
        down_experts_per_buffer = _SCALE_SLOTS_PER_QUADRANT
        down_group_expert = expert_idx % down_experts_per_buffer
        if self.shared_scale:
            # The slot of the shared scale buffer (the view below starts there).
            down_group_expert = shared_scale_slot(expert_idx, _DOWN_IDX)
            if (expert_idx, _DOWN_IDX) not in stream.first_scale_slots:
                stream.load_scale_slot(self.expert_down_weights_scale, expert_idx, _DOWN_IDX)
        elif loaded is None and down_group_expert == 0 and (expert_idx > 0 or not stream.first_down_scales):
            # Waits for the previous down matmuls, as the down weight load does.
            _load_scale_rows(
                dst=down_scale_buffer,
                scale=self.expert_down_weights_scale,
                first_expert=expert_idx,
                n_experts=min(down_experts_per_buffer, self.E_L - expert_idx),
                projections_per_expert=1,
            )
        if loaded is not None:
            self.down_pieces = loaded["down"]() if callable(loaded["down"]) else loaded["down"]
        else:
            self.down_pieces = stream.weight(
                _DOWN_IDX, expert_idx, self.expert_down_weights.select(dim=0, index=expert_idx)
            )
        self.down_weight_scale_sb = down_scale_buffer.reshape((_PMAX, self.n_I512, D))[
            _SCALE_ROWS_PER_QUADRANT * down_group_expert :, :, :
        ]
        if loaded is not None:
            self.down_weight_scale_sb = loaded["down_scale"]

    def transpose(self, chunk_idx: int, after=None) -> dict:
        """Transpose, evict and quantize activation chunk ``chunk_idx`` (the next one); return its transposes and its
        eviction (``_transpose_and_quantize_chunk``)."""
        issued = {}
        # The static loop's alternating tiles can also hold the eviction tiles; the other tiles of ``tiles`` do not.
        evictions = self.tiles if isinstance(self.tiles, _StaticLoopTiles) and self.tiles.evictions else None
        self.quantized_chunks.append(
            _transpose_and_quantize_chunk(
                activation_token_major=self.activation_token_major,
                chunk_idx=chunk_idx,
                I=self.I,
                T=self.T,
                after=after,
                issued=issued,
                evictions=evictions,
            )
        )
        return issued

    def transpose_tile(self, tile_idx: int, after=None) -> dict:
        """Transpose I/512 tile ``tile_idx`` of the activation (the next one): its transposes follow ``after`` on
        TensorE. After the last tile of an activation chunk, evict and quantize the chunk. Return the tile's
        transposes and, after a chunk's last tile, the chunk's eviction (``_ActivationChunk``)."""
        chunk_idx, chunk_tile = divmod(tile_idx, self.tiles_per_chunk)
        if chunk_tile == 0:
            evictions = self.tiles if isinstance(self.tiles, _StaticLoopTiles) and self.tiles.evictions else None
            self.activation_chunk = _ActivationChunk(
                self.activation_token_major, chunk_idx, self.I, self.T, evictions=evictions
            )
        issued = {"transposes": self.activation_chunk.transpose_tile(chunk_tile, after=after)}
        if chunk_tile == self.activation_chunk.n_tiles - 1:
            quantized, issued["eviction"] = self.activation_chunk.finish()
            self.quantized_chunks.append(quantized)
        return issued

    def down(
        self, consume, transpose_lookahead: int = _TRANSPOSE_LOOKAHEAD_CHUNKS, copy_engines=None, psum_consume=False
    ) -> dict:
        """Issue the down matmuls, the PSUM copies and ``consume`` of each H tile; return the first and the last matmul
        of each H tile, and the last matmul.

        The chunks not yet transposed are transposed after the matmuls of the chunk ``transpose_lookahead`` before
        them. ``copy_engines`` (H tile -> engine), when given, names the engine of each H tile's PSUM copy; None leaves
        the choice to the compiler, as the default program does.
        """
        T, D, tiles = self.T, self.D, self.tiles
        down_pieces, down_weight_scale_sb = self.down_pieces, self.down_weight_scale_sb
        # The down matmuls run I/512 tile by tile into one FP32 PSUM accumulator
        # per H/512 tile, so each tile needs only its own chunk of the
        # transposed and quantized activation. Chunk c's transposes are issued
        # after the matmuls of chunk c - 2, which hides the SiTU and
        # quantization latency of the last chunks. The H tiles run in groups,
        # so only a group's accumulators occupy PSUM; a group repeats no work,
        # because every (I tile, H tile) pair runs exactly one matmul.
        tile_H = min(D, _DOWN_TILE_H)
        n_H_tiles = div_ceil(D, tile_H)
        tiles_per_chunk = self.tiles_per_chunk
        n_chunks = self.n_transpose_chunks
        quantized_chunks = self.quantized_chunks
        # Below the default lookahead, a later chunk's transposes follow the first down matmul of the chunk before it.
        pin_transposes = transpose_lookahead < _TRANSPOSE_LOOKAHEAD_CHUNKS
        firsts, lasts, last = {}, {}, None
        for group_first_tile in range(0, n_H_tiles, _DOWN_TILES_PER_GROUP):
            group_tiles = range(group_first_tile, min(group_first_tile + _DOWN_TILES_PER_GROUP, n_H_tiles))
            expert_out_psums = {}
            for tile_h in group_tiles:
                tile_H_actual = min(tile_H, D - tile_h * tile_H)
                expert_out_psums[tile_h] = nl.ndarray((T, tile_H_actual), dtype=nl.float32, buffer=nl.psum)
            for chunk_idx in range(n_chunks):
                next_chunk_idx = chunk_idx + transpose_lookahead
                activation_quant_sb, activation_scale_sb = quantized_chunks[chunk_idx]
                chunk_first_matmul = None
                for chunk_tile in range(activation_quant_sb.shape[1]):
                    tile_i = chunk_idx * tiles_per_chunk + chunk_tile
                    first_tile, down_weight_sb = _piece(down_pieces, tile_i)
                    for tile_h in group_tiles:
                        H_slice = nl.ds(tile_h * tile_H, expert_out_psums[tile_h].shape[1])
                        matmul = nisa.nc_matmul_mx(
                            dst=expert_out_psums[tile_h],
                            stationary=activation_quant_sb[:, chunk_tile, :],
                            moving=down_weight_sb[:, tile_i - first_tile, H_slice],
                            stationary_scale=activation_scale_sb[:, chunk_tile, :],
                            moving_scale=down_weight_scale_sb[:, tile_i, H_slice],
                        )
                        if chunk_first_matmul is None:
                            chunk_first_matmul = matmul
                        firsts.setdefault(tile_h, matmul)
                        lasts[tile_h] = matmul
                        last = matmul
                if next_chunk_idx < n_chunks and len(quantized_chunks) <= next_chunk_idx:
                    self.transpose(next_chunk_idx, after=chunk_first_matmul if pin_transposes else None)

            for tile_h in group_tiles:
                H_slice = nl.ds(tile_h * tile_H, expert_out_psums[tile_h].shape[1])
                tile_H_actual = expert_out_psums[tile_h].shape[1]
                if psum_consume:
                    # The consumer reads the FP32 PSUM tile itself (moe_tkg_decode psum_consume).
                    consume(tile_h * tile_H, tile_H_actual, H_slice, expert_out_psums[tile_h])
                    continue
                expert_out_f32 = _tile(tiles, "expert_out", T, tile_H_actual)
                engine = nisa.engine.unknown if copy_engines is None else copy_engines(tile_h)
                nisa.tensor_copy(dst=expert_out_f32, src=expert_out_psums[tile_h], engine=engine)
                consume(tile_h * tile_H, tile_H_actual, H_slice, expert_out_f32)
        return {
            "firsts": [firsts[tile_h] for tile_h in sorted(firsts)],
            "lasts": [lasts[tile_h] for tile_h in sorted(lasts)],
            "last": last,
        }


# The most 128-token tiles of one moe_tkg_decode_token_tiles call.
_TOKEN_TILES_MAX = 4


def moe_tkg_decode_token_tiles(
    latent,
    local_affinities,
    expert_gate_up_weights: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_gate_up_weights_scale: nl.NkiTensor,
    expert_down_weights_scale: nl.NkiTensor,
    beta: float = 4.0,
    linear_beta: float = 25.0,
    weight_buffers: int = _PROJECTIONS_PER_EXPERT,
    scale_buffers: int = _GATE_UP_SCALE_BUFFERS,
    early_loads: bool = True,
    stream: "_StaticExpertStream" = None,
    half_weight_buffers: int = 0,
    gate_up_half_major: bool = False,
    output: nl.NkiTensor = None,
    column_groups_out: nl.NkiTensor = None,
    transpose_blocks: bool = False,
    quantized_latent: tuple = None,
    static_alt_tiles: bool | str = False,
    psum_consume: bool = False,
    cross_expert_order: bool = False,
    transpose_block: bool = False,
    token_pairs: bool = False,
) -> nl.NkiTensor:
    """Run the static expert loop of ``moe_tkg_decode`` on T = 128 n tokens, n = 2 to 4 token tiles of 128.

    Not ``@nki.jit``: a kernel inlines it. Each weight tile loads once (the buffers of ``_StaticExpertStream``) and
    its matmuls run once for each token tile, so a call moves the expert bytes of one T <= 128 call. Each token tile
    runs the operations of a T = 128 call of the static loop: the per-token MXFP8 quantization, the same K order
    and the same expert order. So the result has the bits of n ``moe_tkg_decode`` calls of 128 consecutive tokens.

    Order of one expert: gate and up for each I/512 chunk, then each token tile (the latent tile is the stationary,
    the weights the moving operand, as at T = 128). Down for each token tile, then each group of
    ``_DOWN_TILES_PER_GROUP`` H/512 tiles, as at T = 128. The transposes of the SiTU output run
    ``_TRANSPOSE_LOOKAHEAD_CHUNKS`` chunks ahead over the token tiles, so a token tile does not wait for its first
    chunks.

    Args:
        latent: BF16 HBM ``[T, D]``, or a sequence of n whole SBUF ``[128, D]`` tiles (view offset 0), token tile
            j = tokens ``[128 j, 128 j + 128)``. The call does not write the tiles.
        local_affinities: BF16 or FP32 HBM ``[T, E_L]``, or a sequence of n SBUF ``[128, E_L]`` tiles. Entries outside
            the selected local top-k must be zero.
        expert_gate_up_weights .. expert_down_weights_scale: as ``moe_tkg_decode``.
        beta, linear_beta, weight_buffers, scale_buffers, early_loads, half_weight_buffers, gate_up_half_major: as
            the static loop of ``moe_tkg_decode``.
        stream: None (default) starts the call's own stream. Else a stream of ``static_expert_stream`` with the same
            weights, ``weight_buffers`` whole buffers, ``scale_buffers`` and the order of the call (``use_order`` equal
            to ``cross_expert_order``), whose first loads the caller issued earlier (``early_loads``). The call takes
            its buffers in place of its own stream start: the same buffer order and the same loads, issued earlier, so
            the result does not change.
        output: optional BF16 ``[T, D]`` HBM tensor that receives the result. None (default) returns a new shared
            HBM tensor, unless ``column_groups_out`` is given.
        column_groups_out: BF16 SBUF ``[128, T, D/128]``: the result in the column-group layout of
            ``moe_tkg_decode``; token tile j fills tokens ``[128 j, 128 j + 128)``. Takes no ``output``.
        transpose_blocks: False (default) keeps the order that the compiler picks. True gives TensorE one fixed order
            (``_TensorChain``), with every PE transpose in a block between MX matmul groups: the latent swizzle
            transposes of all token tiles before the first gate matmul, the activation transposes as one block per
            (token tile, 1024-column chunk) one up chunk after the up matmuls of the chunk (the last chunk's between
            the down matmul groups, ``_ChunkBlocks``), and the column-group transposes of the last expert after the
            first H group of the next token tile. The compiler otherwise puts single transposes between MX matmuls, and each
            one costs a switch out of and back into the MX matmul mode. The instructions and the result do not
            change.
        quantized_latent: None (default) loads and quantizes ``latent``. A tuple (codes, scales) gives the MXFP8
            expert input of all T tokens in the layout of ``_quantize_natural_latent`` (codes
            ``[128, D/512, T]`` ``float8_e4m3fn_x4`` and E8M0 scales ``[128, D/512, T]`` UINT8, SBUF), and
            ``latent`` must be None. Token tile j reads tokens ``[128 j, 128 j + 128)`` of both: the scale groups
            are 32 values along D of one token, so the slice is the input of the tile's own quantize, and the
            call runs no latent load, transpose or quantize.
        static_alt_tiles: ``static_alt_tiles`` of ``moe_tkg_decode`` (``STATIC_ALT_TILE_FORMS``) for the 128-token
            work tiles of every token tile: the gate sigmoid / up tanh tiles ("act"), the down PSUM copies
            ("act_copy") and the activation evictions (True) take the next of two buffers each
            (``_StaticLoopTiles``), so under SBUF pressure the compiler cannot stack them on one address. Not with
            ``transpose_blocks``. The instructions and the result do not change.
        psum_consume: as ``moe_tkg_decode``: True makes each token tile's affinity multiply-add read the FP32 down
            PSUM tile, with the tile's FP32 affinities in the columns after its accumulator's; no PSUM copy. The result
            does not change. False (default) keeps the copies; so does ``static_alt_tiles``.
        cross_expert_order: False (default) runs each expert's gate, up and down matmuls in turn. True is the
            cross-expert order of ``moe_tkg_decode`` (``_cross_expert_steps``) over all token tiles: up(e), the
            transposes of (token tile 0, chunk 0), gate(e + 1) chunk 0, the next transposes, gate(e + 1) chunk 1, and
            so on over the (token tile, 1024-column chunk) order, then down(e), up(e + 1). Each step runs every token
            tile (``_TokenTilesExpertSteps``). The weights take the 3 whole buffers in that use order, and with 1
            gate/up scale buffer each gate and up projection loads its own scale slot. Needs 3 whole weight buffers,
            the standard gate/up layout and no ``transpose_blocks``. Each expert's matmuls, their PSUM accumulation
            order and the order in which the experts add up do not change, so the result does not change.
        transpose_block: with ``cross_expert_order`` only. True issues all the activation transposes of an expert (every
            token tile) in one block after gate(e + 1) chunk 0 (``_blocked_steps``). The instructions and the result
            do not change.
        token_pairs: False (default) keeps the program. True (with ``column_groups_out``, n even) takes the token tiles
            in pairs: token q of token tile 2 i + j is token ``256 i + 2 q + j`` of ``column_groups_out``
            (``_to_column_groups_at`` ``pair_slot``). The inputs keep their tile order; only the output map changes.
            Not with ``cross_expert_order``.

    Returns:
        BF16 ``[T, D]`` unnormalized local routed-expert contribution, or ``column_groups_out`` when it is given.
    """
    kernel_assert(
        nl.program_ndim() == 0 or (nl.program_ndim() == 1 and nl.num_programs(0) == 1),
        "moe_tkg_decode_token_tiles supports LNC1 only",
    )
    kernel_assert(nisa.get_nc_version() >= nisa.nc_version.gen4, "moe_tkg_decode_token_tiles requires Trn3 or newer")
    E_L = expert_gate_up_weights.shape[0]
    I = expert_gate_up_weights.shape[-1]
    D = expert_down_weights.shape[-1]
    n_D512 = D // MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
    n_I512 = I // MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
    if quantized_latent is not None:
        kernel_assert(latent is None, "moe_tkg_decode_token_tiles takes latent None with quantized_latent")
        kernel_assert(len(quantized_latent) == 2, "quantized_latent is a (codes, scales) pair")
        codes, scales = quantized_latent
        kernel_assert(
            tuple(codes.shape[:2]) == (_PMAX, n_D512)
            and len(codes.shape) == 3
            and codes.shape[2] % _PMAX == 0
            and codes.dtype == nl.float8_e4m3fn_x4
            and codes.buffer == nl.sbuf
            and tuple(scales.shape) == tuple(codes.shape)
            and scales.dtype == nl.uint8
            and scales.buffer == nl.sbuf,
            f"quantized_latent must be SBUF codes [128, {n_D512}, 128 n] float8_e4m3fn_x4 and UINT8 scales of the "
            f"same shape, got {codes.dtype} {codes.shape}, {scales.dtype} {scales.shape}",
        )
        latent_tiles = []
        n_tiles = codes.shape[2] // _PMAX
    else:
        latent_tiles = _token_tiles_of(latent, D, "latent")
        n_tiles = len(latent_tiles)
    affinity_tiles = _token_tiles_of(local_affinities, E_L, "local_affinities")
    T = n_tiles * _PMAX
    kernel_assert(
        2 <= n_tiles <= _TOKEN_TILES_MAX and len(affinity_tiles) == n_tiles,
        f"Expected 2 to {_TOKEN_TILES_MAX} token tiles of {_PMAX} in latent and local_affinities, got "
        f"{n_tiles} and {len(affinity_tiles)}",
    )
    for tile in latent_tiles:
        kernel_assert(tile.dtype == nl.bfloat16, f"Expected a BF16 latent, got {tile.dtype}")
    for tile in affinity_tiles:
        kernel_assert(
            tile.dtype == nl.bfloat16 or tile.dtype == nl.float32,
            f"Expected BF16 or FP32 local_affinities, got {tile.dtype}",
        )
    kernel_assert(
        expert_gate_up_weights.dtype == nl.float4_e2m1fn_x4 and expert_down_weights.dtype == nl.float4_e2m1fn_x4,
        f"Expected MXFP4 weights, got {expert_gate_up_weights.dtype} and {expert_down_weights.dtype}",
    )
    kernel_assert(
        expert_gate_up_weights_scale.dtype == nl.uint8 and expert_down_weights_scale.dtype == nl.uint8,
        f"Expected UINT8 MX scales, got {expert_gate_up_weights_scale.dtype} and {expert_down_weights_scale.dtype}",
    )
    kernel_assert(
        D % MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM == 0 and I % MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM == 0,
        f"Expected D and I divisible by 512, got D={D} I={I}",
    )
    kernel_assert(
        expert_gate_up_weights.shape == (E_L, _PMAX, 2, n_D512, I)
        and expert_gate_up_weights_scale.shape == (E_L, _MX_SCALE_PARTITIONS, 2, n_D512, I)
        and expert_down_weights.shape == (E_L, _PMAX, n_I512, D)
        and expert_down_weights_scale.shape == (E_L, _MX_SCALE_PARTITIONS, n_I512, D),
        f"Unexpected expert shapes {expert_gate_up_weights.shape} {expert_gate_up_weights_scale.shape} "
        f"{expert_down_weights.shape} {expert_down_weights_scale.shape}",
    )
    kernel_assert(beta > 0.0 and linear_beta > 0.0, f"Expected beta, linear_beta > 0, got {beta}, {linear_beta}")
    kernel_assert(
        1 <= weight_buffers <= _PROJECTIONS_PER_EXPERT and 1 <= scale_buffers <= _GATE_UP_SCALE_BUFFERS,
        f"Expected 1 to {_PROJECTIONS_PER_EXPERT} weight and 1 to {_GATE_UP_SCALE_BUFFERS} scale buffers, got "
        f"{weight_buffers} and {scale_buffers}",
    )
    kernel_assert(
        half_weight_buffers == 0 or 2 <= half_weight_buffers <= 2 * _PROJECTIONS_PER_EXPERT,
        f"Expected half_weight_buffers 0 or 2 to {2 * _PROJECTIONS_PER_EXPERT}, got {half_weight_buffers}",
    )
    kernel_assert(
        not gate_up_half_major or I % (2 * _GATE_UP_CHUNK_I) == 0,
        f"gate_up_half_major needs I a multiple of {2 * _GATE_UP_CHUNK_I}, got I={I}",
    )
    kernel_assert(
        static_alt_tiles in STATIC_ALT_TILE_FORMS,
        f"static_alt_tiles must be one of {STATIC_ALT_TILE_FORMS}, got {static_alt_tiles!r}",
    )
    kernel_assert(
        not (static_alt_tiles and transpose_blocks),
        "static_alt_tiles of the token-tiled call takes no transpose_blocks",
    )
    if cross_expert_order:
        kernel_assert(
            weight_buffers == _PROJECTIONS_PER_EXPERT and not half_weight_buffers,
            f"cross_expert_order needs {_PROJECTIONS_PER_EXPERT} whole weight buffers, got weight_buffers="
            f"{weight_buffers} half_weight_buffers={half_weight_buffers}",
        )
        kernel_assert(not gate_up_half_major, "cross_expert_order reads the standard gate/up layout")
        kernel_assert(
            not transpose_blocks,
            "cross_expert_order and transpose_blocks both order the TensorE work of the call: pass one of them",
        )
        kernel_assert(
            not static_alt_tiles and not psum_consume,
            "cross_expert_order runs its own work tiles and PSUM copies: it takes no static_alt_tiles or psum_consume",
        )
    kernel_assert(not transpose_block or cross_expert_order, "transpose_block needs cross_expert_order")
    if column_groups_out is not None:
        kernel_assert(output is None, "column_groups_out is the output of the call: it takes no output tensor")
        kernel_assert(D % (2 * _PMAX) == 0, f"column_groups_out needs D a multiple of {2 * _PMAX}, got D={D}")
        kernel_assert(
            tuple(column_groups_out.shape) == (_PMAX, T, D // _PMAX)
            and column_groups_out.dtype == nl.bfloat16
            and column_groups_out.buffer == nl.sbuf,
            f"Expected a BF16 SBUF column_groups_out {(_PMAX, T, D // _PMAX)}, got "
            f"{column_groups_out.dtype} {column_groups_out.buffer} {column_groups_out.shape}",
        )
    kernel_assert(
        not token_pairs or (column_groups_out is not None and T % (2 * _PMAX) == 0),
        f"token_pairs maps pairs of token tiles into column_groups_out: it needs column_groups_out and T a multiple of "
        f"{2 * _PMAX}, got T={T}",
    )
    kernel_assert(
        not (token_pairs and cross_expert_order),
        "token_pairs maps the token tiles of the static loop, not the tiles of the cross-expert order",
    )
    kernel_assert(
        output is None or (tuple(output.shape) == (T, D) and output.dtype == nl.bfloat16),
        f"Expected a BF16 output {(T, D)}, got {None if output is None else output.shape}",
    )
    if stream is not None:
        kernel_assert(
            early_loads and not half_weight_buffers,
            "A prefetched stream feeds the token-tiled call with early_loads and whole weight buffers only",
        )
        stream.check(
            expert_gate_up_weights.shape,
            expert_down_weights.shape,
            weight_buffers,
            scale_buffers,
            gate_up_half_major,
            use_order=cross_expert_order,
        )

    stream_args = dict(
        expert_gate_up_weights=expert_gate_up_weights,
        expert_down_weights=expert_down_weights,
        expert_gate_up_weights_scale=expert_gate_up_weights_scale,
        expert_down_weights_scale=expert_down_weights_scale,
        weight_buffers=weight_buffers,
        scale_buffers=scale_buffers,
        half_weight_buffers=half_weight_buffers,
        gate_up_half_major=gate_up_half_major,
        **({"use_order": True} if cross_expert_order else {}),
    )
    # As the static loop: the first load of each buffer before any work tile of this call exists.
    if stream is None:
        stream = _StaticExpertStream(**stream_args, first_loads=True) if early_loads else None
    chain = _TensorChain() if transpose_blocks else None
    quantized = []
    if quantized_latent is not None:
        # Token tile j: the token slice [128 j, 128 j + 128) of the codes and the scales.
        for tile_idx in range(n_tiles):
            tokens = nl.ds(tile_idx * _PMAX, _PMAX)
            quantized.append((codes[:, :, tokens], scales[:, :, tokens]))
    for tile in latent_tiles:
        latent_sb = tile
        if tile.buffer != nl.sbuf:
            latent_sb = _load_natural_latent(latent=tile, T=_PMAX, T_physical=_PMAX, D=D)
        else:
            kernel_assert(
                tuple(tile.shape) == (_PMAX, D) and tile.offset == 0,
                f"An SBUF latent tile must be a whole [{_PMAX}, {D}] tile (view offset 0), got {tile.shape}",
            )
        swizzles = [] if chain is not None else None
        quantized.append(_quantize_natural_latent(latent_sb=latent_sb, T=_PMAX, D=D, issued=swizzles))
        if chain is not None:
            chain.extend(swizzles)
    if stream is None:
        stream = _StaticExpertStream(**stream_args, first_loads=False)

    # static_alt_tiles keeps the PSUM copies (their tiles are a role of its rings).
    psum_consume = psum_consume and not static_alt_tiles
    accumulators, affinities_f32, output_tiles = [], [], []
    if output is None and column_groups_out is None:
        output = nl.ndarray((T, D), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for tile_idx in range(n_tiles):
        # psum_consume: the FP32 affinities in the columns after the accumulator's, in one SBUF tile.
        affinity_columns = E_L if psum_consume else 0
        accum_aff_sb = nl.ndarray((_PMAX, D + affinity_columns), dtype=nl.float32, buffer=nl.sbuf)
        accumulator = accum_aff_sb[:, 0:D] if psum_consume else accum_aff_sb
        nisa.memset(dst=accumulator, value=0.0)
        accumulators.append(accumulator)
        # One contiguous load for all experts, as the static loop.
        affinity_raw_sb = nl.ndarray((_PMAX, E_L), dtype=affinity_tiles[tile_idx].dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=affinity_raw_sb, src=affinity_tiles[tile_idx], dge_mode=nisa.dge_mode.none)
        if psum_consume:
            affinity_f32_sb = accum_aff_sb[:, D : D + E_L]
        else:
            affinity_f32_sb = nl.ndarray((_PMAX, E_L), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=affinity_f32_sb, src=affinity_raw_sb)
        affinities_f32.append(affinity_f32_sb)
        tile_output = None if output is None else output[nl.ds(tile_idx * _PMAX, _PMAX), :]
        output_tiles.append(_OutputTiles(output=tile_output, T=_PMAX, sbuf_columns=D if output is None else 0))
    # As the static loop: the rings of the 128-token work tiles, used in turn by every token tile and expert.
    tiles = _StaticLoopTiles(_PMAX, D, _STATIC_ALT_TILE_ROLES[static_alt_tiles]) if static_alt_tiles else None

    if cross_expert_order:

        def consume_of(expert_idx):
            def consume(tile_idx, start, size, H_slice, expert_out_f32):
                sum_dst_sb = accumulators[tile_idx][:, H_slice]
                if expert_idx == E_L - 1:
                    sum_dst_sb = output_tiles[tile_idx].columns(start, size)
                nisa.scalar_tensor_tensor(
                    dst=sum_dst_sb,
                    data=expert_out_f32,
                    op0=nl.multiply,
                    operand0=affinities_f32[tile_idx][:, expert_idx : expert_idx + 1],
                    op1=nl.add,
                    operand1=accumulators[tile_idx][:, H_slice],
                )
                if expert_idx == E_L - 1:
                    output_tiles[tile_idx].store_when_written(start, size)

            return consume

        def tile_done_of(expert_idx):
            def tile_done(tile_idx):
                if expert_idx == E_L - 1 and column_groups_out is not None:
                    _to_column_groups_at(output_tiles[tile_idx].tiles[0], column_groups_out, tile_idx * _PMAX)

            return tile_done

        steps = [
            _TokenTilesExpertSteps(
                stream=stream,
                expert_idx=expert_idx,
                quantized=quantized,
                expert_gate_up_weights=expert_gate_up_weights,
                expert_down_weights=expert_down_weights,
                expert_gate_up_weights_scale=expert_gate_up_weights_scale,
                expert_down_weights_scale=expert_down_weights_scale,
                beta=beta,
                linear_beta=linear_beta,
                tile_done=tile_done_of(expert_idx),
                scale_slots=len(stream.gate_up_scale_buffers) == 1,
            )
            for expert_idx in range(E_L)
        ]
        _cross_expert_steps(steps, consume_of, transpose_block=transpose_block)
        return output if column_groups_out is None else column_groups_out

    for expert_idx in nl.sequential_range(E_L):

        def consume(tile_idx, start, size, H_slice, expert_out_f32, expert_idx=expert_idx):
            sum_dst_sb = accumulators[tile_idx][:, H_slice]
            if expert_idx == E_L - 1:
                sum_dst_sb = output_tiles[tile_idx].columns(start, size)
            nisa.scalar_tensor_tensor(
                dst=sum_dst_sb,
                data=expert_out_f32,
                op0=nl.multiply,
                operand0=affinities_f32[tile_idx][:, expert_idx : expert_idx + 1],
                op1=nl.add,
                operand1=accumulators[tile_idx][:, H_slice],
            )
            if expert_idx == E_L - 1:
                output_tiles[tile_idx].store_when_written(start, size)

        def tile_done(tile_idx, expert_idx=expert_idx):
            if expert_idx == E_L - 1 and column_groups_out is not None and token_pairs:
                _to_column_groups_at(
                    output_tiles[tile_idx].tiles[0],
                    column_groups_out,
                    tile_idx // 2 * 2 * _PMAX,
                    chain=chain,
                    pair_slot=tile_idx % 2,
                )
            elif expert_idx == E_L - 1 and column_groups_out is not None:
                _to_column_groups_at(output_tiles[tile_idx].tiles[0], column_groups_out, tile_idx * _PMAX, chain=chain)

        _token_tiles_expert_ffn(
            stream=stream,
            expert_idx=expert_idx,
            quantized=quantized,
            expert_gate_up_weights=expert_gate_up_weights,
            expert_down_weights=expert_down_weights,
            expert_gate_up_weights_scale=expert_gate_up_weights_scale,
            expert_down_weights_scale=expert_down_weights_scale,
            beta=beta,
            linear_beta=linear_beta,
            consume=consume,
            tile_done=tile_done,
            chain=chain,
            tiles=tiles,
            psum_consume=psum_consume,
        )
    return output if column_groups_out is None else column_groups_out


def _token_tiles_of(tensor, columns: int, name: str) -> list:
    """Return the 128-token tiles of an HBM ``[T, columns]`` tensor (row views) or of a sequence of SBUF tiles."""
    if isinstance(tensor, (list, tuple)):
        for tile in tensor:
            kernel_assert(
                tuple(tile.shape) == (_PMAX, columns) and tile.buffer == nl.sbuf,
                f"{name}: expected SBUF [{_PMAX}, {columns}] tiles, got {tile.buffer} {tile.shape}",
            )
        return list(tensor)
    kernel_assert(
        len(tensor.shape) == 2 and tensor.shape[1] == columns and tensor.shape[0] % _PMAX == 0,
        f"{name}: expected HBM [128 n, {columns}], got {tensor.shape}",
    )
    kernel_assert(tensor.buffer != nl.sbuf, f"{name}: an SBUF input comes as a sequence of [{_PMAX}, {columns}] tiles")
    return [tensor[nl.ds(tile_idx * _PMAX, _PMAX), :] for tile_idx in range(tensor.shape[0] // _PMAX)]


def _to_column_groups_at(
    output_sb: nl.NkiTensor,
    column_groups_out: nl.NkiTensor,
    token_offset: int,
    chain: "_TensorChain" = None,
    pair_slot: int = None,
):
    """``_to_column_groups`` of one BF16 ``[128, D]`` token tile into tokens ``[token_offset, token_offset + 128)`` of
    ``column_groups_out`` ``[128, T, D/128]``. A token tile is 128 x D/128 BF16 per partition, whole 32-byte SBUF
    blocks, so no two tiles share a block. ``chain`` (``_TensorChain``), when given, takes the transposes in order.

    ``pair_slot`` 0 or 1 (``moe_tkg_decode_token_tiles`` ``token_pairs``) writes token q of the tile to token
    ``token_offset + 2 q + pair_slot`` instead. The two tiles of a pair then share 32-byte blocks, but each half of
    the copies ends on a whole block (token 64 of the tile is token 128 of the pair), so only copies of one engine
    write the same block."""
    T, D = output_sb.shape
    kernel_assert(T == _PMAX, f"_to_column_groups_at takes one token tile of {_PMAX} tokens, got {T}")
    T_total = column_groups_out.shape[1]
    words = D // (2 * _PMAX)
    banks = div_ceil(words * T * 4, _PSUM_BANK_BYTES)
    word_view = output_sb.view(nl.float32).reshape((T, _PMAX, words))
    psum = nl.ndarray((_PMAX, banks * _PSUM_BANK_BYTES // 4), dtype=nl.float32, buffer=nl.psum)
    psum_words = psum[:, 0 : words * T].reshape((_PMAX, words, T))
    for word in range(words):
        transpose = nisa.nc_transpose(
            dst=psum_words[:, word, 0:T], data=word_view[0:T, :, word], engine=nisa.engine.tensor
        )
        if chain is not None:
            chain.add(transpose)
    dst_words = column_groups_out.reshape((_PMAX, T_total * 2 * words)).view(nl.float32)
    dst_words = dst_words.reshape((_PMAX, T_total, words))
    if pair_slot is not None:
        kernel_assert(
            pair_slot in (0, 1) and token_offset % (2 * T) == 0 and token_offset + 2 * T <= T_total,
            f"pair_slot {pair_slot} writes 2 x {T} tokens from token {token_offset} of {T_total}",
        )
        # [128, T_total / 2, 2, words]: token 2 i + j at (i, j).
        dst_words = dst_words.reshape((_PMAX, T_total // 2, 2, words))
    block_tokens = _column_group_block_tokens(D)
    bounds = [0, T // 2 // block_tokens * block_tokens, T]
    for half, (first, last) in enumerate(zip(bounds, bounds[1:])):
        if pair_slot is None:
            dst = dst_words[:, token_offset + first : token_offset + last, :]
        else:
            dst = dst_words[:, token_offset // 2 + first : token_offset // 2 + last, pair_slot, :]
        nisa.tensor_copy(
            dst=dst,
            src=psum.ap(pattern=[[psum.shape[1], _PMAX], [1, last - first], [T, words]], offset=first),
            engine=(nisa.engine.vector, nisa.engine.scalar)[half],
        )


class _TensorChain:
    """One fixed TensorE order (``transpose_blocks`` of ``moe_tkg_decode_token_tiles``): each instruction added follows
    the one added before it on TensorE (a schedule edge, no semaphore wait)."""

    def __init__(self):
        self.last = None

    def add(self, instruction):
        if self.last is not None:
            _after(instruction, self.last)
        self.last = instruction

    def extend(self, instructions):
        for instruction in instructions:
            self.add(instruction)


class _ChunkBlocks:
    """The activation transposes of one expert of the token-tiled loop in blocks of one (token tile, 1024-column
    chunk): the 8 transposes of the chunk's two SiTU tiles (4 for a last 512-column chunk), issued back to back on the
    TensorE order of ``chain`` (``_TensorChain``), then the chunk's eviction and quantization (``_ActivationChunk``, the
    instructions of ``_transpose_and_quantize_chunk``).

    The block of (token tile t, chunk c) is issued after the up matmuls of the up chunk after the chunk's last I/512
    tile (I/512 tile i is up chunk i), for token tile t: the SiTU products of the block then have the matmuls of the
    token tiles' next up chunk to finish on VectorE. The blocks of the last chunk follow: the first token tile's before
    the down matmuls (``before_down``, so that the chunk is quantized while the first down group runs), the others one
    between two down matmul groups (``issue_next``). A chunk that a down group needs is issued first
    (``quantized_chunk``)."""

    def __init__(self, activations: list, I: int, chain: "_TensorChain"):
        self.activations, self.I, self.chain = activations, I, chain
        n_I512 = I // MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
        tiles_per_chunk = _TRANSPOSE_CHUNK_I // MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
        n_chunks = div_ceil(n_I512, tiles_per_chunk)
        # The last I/512 tile of each chunk.
        self.last_tile = [min(n_I512, (c + 1) * tiles_per_chunk) - 1 for c in range(n_chunks)]
        self.pending = [(tile_idx, c) for c in range(n_chunks) for tile_idx in range(len(activations))]
        self.quantized = {}

    def _issue(self, block):
        self.pending.remove(block)
        tile_idx, chunk_idx = block
        chunk = _ActivationChunk(self.activations[tile_idx], chunk_idx, self.I, _PMAX)
        for chunk_tile in range(chunk.n_tiles):
            self.chain.extend(chunk.transpose_tile(chunk_tile))
        self.quantized[block], _ = chunk.finish()

    def after_up(self, up_chunk_idx: int, tile_idx: int):
        """After the up matmuls of up chunk ``up_chunk_idx`` of token tile ``tile_idx``: the block of the activation
        chunk that ended with the up chunk before."""
        for chunk_idx, last in enumerate(self.last_tile):
            if last == up_chunk_idx - 1 and (tile_idx, chunk_idx) in self.pending:
                self._issue((tile_idx, chunk_idx))

    def before_down(self):
        """Before the down matmuls: the pending blocks of the first token tile."""
        for block in [block for block in self.pending if block[0] == 0]:
            self._issue(block)

    def issue_next(self):
        """Between down matmul groups: the next pending block, if any."""
        if self.pending:
            self._issue(self.pending[0])

    def quantized_chunk(self, tile_idx: int, chunk_idx: int) -> tuple:
        """The MXFP8 tiles and scales of a chunk, after issuing the pending blocks up to it."""
        while (tile_idx, chunk_idx) not in self.quantized:
            self.issue_next()
        return self.quantized[tile_idx, chunk_idx]


def _token_tiles_expert_ffn(
    stream: "_StaticExpertStream",
    expert_idx: int,
    quantized: list,
    expert_gate_up_weights: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_gate_up_weights_scale: nl.NkiTensor,
    expert_down_weights_scale: nl.NkiTensor,
    beta: float,
    linear_beta: float,
    consume,
    tile_done,
    chain: "_TensorChain" = None,
    tiles: "_StaticLoopTiles" = None,
    psum_consume: bool = False,
):
    """Run the gate/up, SiTU and down projections of one local expert on the quantized token tiles ``quantized``.

    The operations of each token tile are those of ``_expert_ffn`` at T = 128 (without ``loaded``).
    ``consume(tile_idx, start, size, H_slice, expert_out_f32)`` takes the FP32 [128, size] output of columns
    ``[start, start + size)`` of a token tile, and ``tile_done(tile_idx)`` runs after the last H tile of a token tile.
    With ``chain`` (``transpose_blocks`` of ``moe_tkg_decode_token_tiles``) every matmul and transpose joins the
    TensorE order of the chain, the activation transposes run in chunk blocks (``_ChunkBlocks``), and
    ``tile_done(tile_idx)`` runs after the first H group of the next token tile (the last one's at the end).
    ``tiles`` (``static_alt_tiles`` of ``moe_tkg_decode_token_tiles``) gives the gate sigmoid / up tanh, PSUM copy and
    eviction tiles of its rings, as in the static loop; the other roles take new tiles.
    ``psum_consume`` gives ``consume`` the FP32 down PSUM tile itself, with no copy.
    """
    D = expert_down_weights.shape[-1]
    E_L = expert_gate_up_weights.shape[0]
    I = expert_gate_up_weights.shape[-1]
    n_D512 = D // MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
    n_I512 = I // MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
    T = _PMAX
    n_tiles = len(quantized)
    gate_up_scale_buffers = stream.gate_up_scale_buffers
    scale_buffers = len(gate_up_scale_buffers)
    down_scale_buffer = stream.down_scale_buffer
    gate_up_experts_per_buffer = _SCALE_SLOTS_PER_QUADRANT // 2
    down_experts_per_buffer = _SCALE_SLOTS_PER_QUADRANT
    gate_up_group_idx, gate_up_group_expert = divmod(expert_idx, gate_up_experts_per_buffer)
    gate_up_scale_buffer = gate_up_scale_buffers[gate_up_group_idx % scale_buffers]
    # The stream start may have loaded the group of each buffer's first use.
    if gate_up_group_expert == 0 and gate_up_group_idx >= stream.first_gate_up_groups:
        _load_scale_rows(
            dst=gate_up_scale_buffer,
            scale=expert_gate_up_weights_scale,
            first_expert=expert_idx,
            n_experts=min(gate_up_experts_per_buffer, E_L - expert_idx),
            projections_per_expert=2,
        )
    gate_up_scale_sb = gate_up_scale_buffer.reshape((_PMAX, n_D512, I))
    gate_up_scale_offset = 2 * _SCALE_ROWS_PER_QUADRANT * gate_up_group_expert

    gate_pieces = stream.weight(
        GATE_FUSED_IDX,
        expert_idx,
        expert_gate_up_weights.select(dim=0, index=expert_idx).select(dim=1, index=GATE_FUSED_IDX),
    )
    gate_weight_scale_sb = gate_up_scale_sb[gate_up_scale_offset:, :, :]
    n_chunks = div_ceil(I, _GATE_UP_CHUNK_I)
    gate_terms_f32 = [[] for _ in range(n_tiles)]
    for chunk_idx in nl.sequential_range(n_chunks):
        chunk_size = min(_GATE_UP_CHUNK_I, I - chunk_idx * _GATE_UP_CHUNK_I)
        first_chunk, gate_weight_sb = _piece(gate_pieces, chunk_idx)
        for tile_idx, (input_quant_sb, input_scale_sb) in enumerate(quantized):
            matmuls = [] if chain is not None else None
            gate_psum = _project_chunk(
                input_quant_sb=input_quant_sb,
                input_scale_sb=input_scale_sb,
                weight_sb=gate_weight_sb,
                weight_scale_sb=gate_weight_scale_sb,
                n_H512=n_D512,
                chunk_idx=chunk_idx,
                chunk_size=chunk_size,
                T=T,
                weight_first_chunk=first_chunk,
                matmuls=matmuls,
            )
            if chain is not None:
                chain.extend(matmuls)
            gate_sigmoid_f32 = _tile(tiles, "gate_sigmoid", T, chunk_size)
            nisa.activation(dst=gate_sigmoid_f32, op=nl.sigmoid, data=gate_psum)
            gate_term_f32 = nl.ndarray((T, chunk_size), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=gate_term_f32, op=nl.tanh, data=gate_psum, scale=1.0 / beta)
            nisa.scalar_tensor_tensor(
                dst=gate_term_f32,
                data=gate_term_f32,
                op0=nl.multiply,
                operand0=beta,
                op1=nl.multiply,
                operand1=gate_sigmoid_f32,
            )
            gate_terms_f32[tile_idx].append(gate_term_f32)

    up_pieces = stream.weight(
        UP_FUSED_IDX,
        expert_idx,
        expert_gate_up_weights.select(dim=0, index=expert_idx).select(dim=1, index=UP_FUSED_IDX),
    )
    up_weight_scale_sb = gate_up_scale_sb[gate_up_scale_offset + _SCALE_ROWS_PER_QUADRANT :, :, :]
    activations = [nl.ndarray((T, I), dtype=nl.float16, buffer=nl.sbuf) for _ in range(n_tiles)]
    blocks = _ChunkBlocks(activations, I, chain) if chain is not None else None
    for chunk_idx in nl.sequential_range(n_chunks):
        chunk_size = min(_GATE_UP_CHUNK_I, I - chunk_idx * _GATE_UP_CHUNK_I)
        first_chunk, up_weight_sb = _piece(up_pieces, chunk_idx)
        for tile_idx, (input_quant_sb, input_scale_sb) in enumerate(quantized):
            matmuls = [] if chain is not None else None
            up_psum = _project_chunk(
                input_quant_sb=input_quant_sb,
                input_scale_sb=input_scale_sb,
                weight_sb=up_weight_sb,
                weight_scale_sb=up_weight_scale_sb,
                n_H512=n_D512,
                chunk_idx=chunk_idx,
                chunk_size=chunk_size,
                T=T,
                weight_first_chunk=first_chunk,
                matmuls=matmuls,
            )
            if chain is not None:
                chain.extend(matmuls)
            up_tanh_f32 = _tile(tiles, "up_tanh", T, chunk_size)
            nisa.activation(dst=up_tanh_f32, op=nl.tanh, data=up_psum, scale=1.0 / linear_beta)
            nisa.scalar_tensor_tensor(
                dst=activations[tile_idx][:, nl.ds(chunk_idx * _GATE_UP_CHUNK_I, chunk_size)],
                data=up_tanh_f32,
                op0=nl.multiply,
                operand0=linear_beta,
                op1=nl.multiply,
                operand1=gate_terms_f32[tile_idx][chunk_idx],
            )
            if blocks is not None:
                blocks.after_up(chunk_idx, tile_idx)
    down_group_expert = expert_idx % down_experts_per_buffer
    if down_group_expert == 0 and (expert_idx > 0 or not stream.first_down_scales):
        # Waits for the previous down matmuls, as the down weight load does.
        _load_scale_rows(
            dst=down_scale_buffer,
            scale=expert_down_weights_scale,
            first_expert=expert_idx,
            n_experts=min(down_experts_per_buffer, E_L - expert_idx),
            projections_per_expert=1,
        )
    down_pieces = stream.weight(_DOWN_IDX, expert_idx, expert_down_weights.select(dim=0, index=expert_idx))
    down_weight_scale_sb = down_scale_buffer.reshape((_PMAX, n_I512, D))[
        _SCALE_ROWS_PER_QUADRANT * down_group_expert :, :, :
    ]

    # The down matmuls of _expert_ffn by token tile; the transposes run ahead over the (token tile, chunk) sequence.
    tile_H = min(D, _DOWN_TILE_H)
    n_H_tiles = div_ceil(D, tile_H)
    tiles_per_chunk = _TRANSPOSE_CHUNK_I // MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
    n_chunks = div_ceil(n_I512, tiles_per_chunk)
    order = [(tile_idx, chunk_idx) for tile_idx in range(n_tiles) for chunk_idx in range(n_chunks)]
    quantized_chunks = []
    evictions = tiles if tiles is not None and tiles.evictions else None

    def transpose_next():
        tile_idx, chunk_idx = order[len(quantized_chunks)]
        quantized_chunks.append(
            _transpose_and_quantize_chunk(
                activation_token_major=activations[tile_idx], chunk_idx=chunk_idx, I=I, T=T, evictions=evictions
            )
        )

    if blocks is None:
        for _ in range(min(_TRANSPOSE_LOOKAHEAD_CHUNKS, len(order))):
            transpose_next()
    else:
        blocks.before_down()
    for tile_idx in range(n_tiles):
        for group_first_tile in range(0, n_H_tiles, _DOWN_TILES_PER_GROUP):
            group_tiles = range(group_first_tile, min(group_first_tile + _DOWN_TILES_PER_GROUP, n_H_tiles))
            expert_out_psums = {}
            for tile_h in group_tiles:
                tile_H_actual = min(tile_H, D - tile_h * tile_H)
                expert_out_psums[tile_h] = nl.ndarray((T, tile_H_actual), dtype=nl.float32, buffer=nl.psum)
            for chunk_idx in range(n_chunks):
                step = tile_idx * n_chunks + chunk_idx
                if blocks is None:
                    activation_quant_sb, activation_scale_sb = quantized_chunks[step]
                else:
                    activation_quant_sb, activation_scale_sb = blocks.quantized_chunk(tile_idx, chunk_idx)
                for chunk_tile in range(activation_quant_sb.shape[1]):
                    tile_i = chunk_idx * tiles_per_chunk + chunk_tile
                    first_tile, down_weight_sb = _piece(down_pieces, tile_i)
                    for tile_h in group_tiles:
                        H_slice = nl.ds(tile_h * tile_H, expert_out_psums[tile_h].shape[1])
                        matmul = nisa.nc_matmul_mx(
                            dst=expert_out_psums[tile_h],
                            stationary=activation_quant_sb[:, chunk_tile, :],
                            moving=down_weight_sb[:, tile_i - first_tile, H_slice],
                            stationary_scale=activation_scale_sb[:, chunk_tile, :],
                            moving_scale=down_weight_scale_sb[:, tile_i, H_slice],
                        )
                        if chain is not None:
                            chain.add(matmul)
                if blocks is not None:
                    blocks.issue_next()
                elif len(quantized_chunks) < len(order) and len(quantized_chunks) <= step + _TRANSPOSE_LOOKAHEAD_CHUNKS:
                    transpose_next()

            for tile_h in group_tiles:
                H_slice = nl.ds(tile_h * tile_H, expert_out_psums[tile_h].shape[1])
                tile_H_actual = expert_out_psums[tile_h].shape[1]
                if psum_consume:
                    # The consumer reads the FP32 PSUM tile itself (moe_tkg_decode_token_tiles psum_consume).
                    consume(tile_idx, tile_h * tile_H, tile_H_actual, H_slice, expert_out_psums[tile_h])
                    continue
                expert_out_f32 = _tile(tiles, "expert_out", T, tile_H_actual)
                nisa.tensor_copy(dst=expert_out_f32, src=expert_out_psums[tile_h])
                consume(tile_idx, tile_h * tile_H, tile_H_actual, H_slice, expert_out_f32)
            if blocks is not None and group_first_tile == 0 and tile_idx > 0:
                # The previous token tile's outputs are written by now: its column-group transposes follow this group.
                tile_done(tile_idx - 1)
        if blocks is None:
            tile_done(tile_idx)
    if blocks is not None:
        tile_done(n_tiles - 1)


class _TokenTilesExpertSteps(_ExpertSteps):
    """The steps of one local expert of ``moe_tkg_decode_token_tiles`` for the cross-expert order
    (``_cross_expert_steps``).

    Each step issues the instructions of ``_token_tiles_expert_ffn`` for every token tile. ``gate_start`` and
    ``down_start`` are the loads of ``_ExpertSteps``, which are those of ``_token_tiles_expert_ffn`` (with
    ``scale_slots``, each gate and up projection loads its own slot of the group load). ``transpose(idx)`` runs step
    ``idx`` of the (token tile, 1024-column chunk) order of ``_token_tiles_expert_ffn``. ``down`` runs the token tiles
    in turn: ``consume(tile_idx, start, size, H_slice, expert_out_f32)`` takes the FP32 output of an H tile, and
    ``tile_done(tile_idx)`` runs after a token tile's last H group. Constructing the steps issues no instruction.
    """

    def __init__(
        self,
        stream: "_StaticExpertStream",
        expert_idx: int,
        quantized: list,
        expert_gate_up_weights: nl.NkiTensor,
        expert_down_weights: nl.NkiTensor,
        expert_gate_up_weights_scale: nl.NkiTensor,
        expert_down_weights_scale: nl.NkiTensor,
        beta: float,
        linear_beta: float,
        tile_done,
        scale_slots: bool = False,
    ):
        super().__init__(
            stream=stream,
            expert_idx=expert_idx,
            input_quant_sb=None,
            input_scale_sb=None,
            expert_gate_up_weights=expert_gate_up_weights,
            expert_down_weights=expert_down_weights,
            expert_gate_up_weights_scale=expert_gate_up_weights_scale,
            expert_down_weights_scale=expert_down_weights_scale,
            beta=beta,
            linear_beta=linear_beta,
            T=_PMAX,
            scale_slots=scale_slots,
        )
        self.quantized, self.tile_done = quantized, tile_done
        self.n_tiles = len(quantized)
        # The 1024-column chunks of one token tile; the transpose steps run over every (token tile, chunk).
        self.tile_chunks = self.n_transpose_chunks
        self.n_transpose_chunks = self.n_tiles * self.tile_chunks
        self.gate_terms_f32 = [[] for _ in range(self.n_tiles)]
        self.activations = []

    def gate_chunk(self, chunk_idx: int) -> dict:
        """Issue one gate chunk for every token tile; return its first and last matmul and its first sigmoid."""
        T, I, beta = self.T, self.I, self.beta
        gate_weight_scale_sb = self.gate_up_scale_sb[self.gate_up_scale_offset :, :, :]
        chunk_size = min(_GATE_UP_CHUNK_I, I - chunk_idx * _GATE_UP_CHUNK_I)
        first_chunk, gate_weight_sb = _piece(self.gate_pieces, chunk_idx)
        matmuls, sigmoids = [], []
        for tile_idx, (input_quant_sb, input_scale_sb) in enumerate(self.quantized):
            gate_psum = _project_chunk(
                input_quant_sb=input_quant_sb,
                input_scale_sb=input_scale_sb,
                weight_sb=gate_weight_sb,
                weight_scale_sb=gate_weight_scale_sb,
                n_H512=self.n_D512,
                chunk_idx=chunk_idx,
                chunk_size=chunk_size,
                T=T,
                weight_first_chunk=first_chunk,
                matmuls=matmuls,
            )
            gate_sigmoid_f32 = nl.ndarray((T, chunk_size), dtype=nl.float32, buffer=nl.sbuf)
            sigmoid = nisa.activation(dst=gate_sigmoid_f32, op=nl.sigmoid, data=gate_psum, engine=nisa.engine.scalar)
            if sigmoids:
                # ScalarE keeps the token tiles in order, so the first sigmoid's edge holds for the chunk.
                _after(sigmoid, sigmoids[-1])
            sigmoids.append(sigmoid)
            gate_term_f32 = nl.ndarray((T, chunk_size), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=gate_term_f32, op=nl.tanh, data=gate_psum, scale=1.0 / beta, engine=nisa.engine.scalar)
            nisa.scalar_tensor_tensor(
                dst=gate_term_f32,
                data=gate_term_f32,
                op0=nl.multiply,
                operand0=beta,
                op1=nl.multiply,
                operand1=gate_sigmoid_f32,
            )
            self.gate_terms_f32[tile_idx].append(gate_term_f32)
        return {"first": matmuls[0], "last": matmuls[-1], "sigmoid": sigmoids[0]}

    def up(self) -> dict:
        """Issue the up weights and the up chunks of every token tile (matmuls and SiTU products into each tile's FP16
        activation); return the first and the last matmul of each (chunk, token tile), and the last matmul."""
        T, I, linear_beta = self.T, self.I, self.linear_beta
        if self.scale_slots:
            self._load_scale_slot(self.gate_up_scale_buffer, UP_FUSED_IDX)
        up_pieces = self.stream.weight(
            UP_FUSED_IDX,
            self.expert_idx,
            self.expert_gate_up_weights.select(dim=0, index=self.expert_idx).select(dim=1, index=UP_FUSED_IDX),
        )
        up_weight_scale_sb = self.gate_up_scale_sb[self.up_scale_offset :, :, :]
        self.activations = [nl.ndarray((T, I), dtype=nl.float16, buffer=nl.sbuf) for _ in range(self.n_tiles)]
        firsts, lasts = [], []
        for chunk_idx in nl.sequential_range(self.n_chunks):
            chunk_size = min(_GATE_UP_CHUNK_I, I - chunk_idx * _GATE_UP_CHUNK_I)
            first_chunk, up_weight_sb = _piece(up_pieces, chunk_idx)
            for tile_idx, (input_quant_sb, input_scale_sb) in enumerate(self.quantized):
                matmuls = []
                up_psum = _project_chunk(
                    input_quant_sb=input_quant_sb,
                    input_scale_sb=input_scale_sb,
                    weight_sb=up_weight_sb,
                    weight_scale_sb=up_weight_scale_sb,
                    n_H512=self.n_D512,
                    chunk_idx=chunk_idx,
                    chunk_size=chunk_size,
                    T=T,
                    weight_first_chunk=first_chunk,
                    matmuls=matmuls,
                )
                firsts.append(matmuls[0])
                lasts.append(matmuls[-1])
                up_tanh_f32 = nl.ndarray((T, chunk_size), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=up_tanh_f32, op=nl.tanh, data=up_psum, scale=1.0 / linear_beta)
                nisa.scalar_tensor_tensor(
                    dst=self.activations[tile_idx][:, nl.ds(chunk_idx * _GATE_UP_CHUNK_I, chunk_size)],
                    data=up_tanh_f32,
                    op0=nl.multiply,
                    operand0=linear_beta,
                    op1=nl.multiply,
                    operand1=self.gate_terms_f32[tile_idx][chunk_idx],
                )
        return {"firsts": firsts, "lasts": lasts, "last": lasts[-1]}

    def transpose(self, chunk_idx: int, after=None) -> dict:
        """Transpose, evict and quantize step ``chunk_idx`` (the next one) of the (token tile, chunk) order; return its
        transposes and its eviction (``_transpose_and_quantize_chunk``)."""
        kernel_assert(
            chunk_idx == len(self.quantized_chunks),
            f"The transposes run in order: expected step {len(self.quantized_chunks)}, got {chunk_idx}",
        )
        tile_idx, tile_chunk = divmod(chunk_idx, self.tile_chunks)
        issued = {}
        self.quantized_chunks.append(
            _transpose_and_quantize_chunk(
                activation_token_major=self.activations[tile_idx],
                chunk_idx=tile_chunk,
                I=self.I,
                T=self.T,
                after=after,
                issued=issued,
            )
        )
        return issued

    def transpose_tile(self, tile_idx: int, after=None) -> dict:
        kernel_assert(False, "The token-tiled cross-expert order has no tile blocks (cross_expert_transpose_blocks)")

    def down(
        self, consume, transpose_lookahead: int = _TRANSPOSE_LOOKAHEAD_CHUNKS, copy_engines=None, psum_consume=False
    ) -> dict:
        """Issue the down matmuls, the PSUM copies and ``consume`` of each (token tile, H tile), the token tiles in turn
        as in ``_token_tiles_expert_ffn``; return the first and the last matmul of each (token tile, H tile), and the
        last matmul. The transpose steps not yet issued run ``transpose_lookahead`` steps ahead, as there.
        ``copy_engines`` (H tile -> engine), when given, names the engine of each H tile's PSUM copy."""
        kernel_assert(not psum_consume, "the token-tiled cross-expert order takes no psum_consume")
        T, D = self.T, self.D
        tile_H = min(D, _DOWN_TILE_H)
        n_H_tiles = div_ceil(D, tile_H)
        firsts, lasts, last = {}, {}, None
        for tile_idx in range(self.n_tiles):
            for group_first_tile in range(0, n_H_tiles, _DOWN_TILES_PER_GROUP):
                group_tiles = range(group_first_tile, min(group_first_tile + _DOWN_TILES_PER_GROUP, n_H_tiles))
                expert_out_psums = {}
                for tile_h in group_tiles:
                    tile_H_actual = min(tile_H, D - tile_h * tile_H)
                    expert_out_psums[tile_h] = nl.ndarray((T, tile_H_actual), dtype=nl.float32, buffer=nl.psum)
                for chunk_idx in range(self.tile_chunks):
                    step = tile_idx * self.tile_chunks + chunk_idx
                    while len(self.quantized_chunks) <= step:
                        self.transpose(len(self.quantized_chunks))
                    activation_quant_sb, activation_scale_sb = self.quantized_chunks[step]
                    for chunk_tile in range(activation_quant_sb.shape[1]):
                        tile_i = chunk_idx * self.tiles_per_chunk + chunk_tile
                        first_tile, down_weight_sb = _piece(self.down_pieces, tile_i)
                        for tile_h in group_tiles:
                            H_slice = nl.ds(tile_h * tile_H, expert_out_psums[tile_h].shape[1])
                            matmul = nisa.nc_matmul_mx(
                                dst=expert_out_psums[tile_h],
                                stationary=activation_quant_sb[:, chunk_tile, :],
                                moving=down_weight_sb[:, tile_i - first_tile, H_slice],
                                stationary_scale=activation_scale_sb[:, chunk_tile, :],
                                moving_scale=self.down_weight_scale_sb[:, tile_i, H_slice],
                            )
                            firsts.setdefault((tile_idx, tile_h), matmul)
                            lasts[(tile_idx, tile_h)] = matmul
                            last = matmul
                    if (
                        len(self.quantized_chunks) < self.n_transpose_chunks
                        and len(self.quantized_chunks) <= step + transpose_lookahead
                    ):
                        self.transpose(len(self.quantized_chunks))
                for tile_h in group_tiles:
                    H_slice = nl.ds(tile_h * tile_H, expert_out_psums[tile_h].shape[1])
                    tile_H_actual = expert_out_psums[tile_h].shape[1]
                    expert_out_f32 = nl.ndarray((T, tile_H_actual), dtype=nl.float32, buffer=nl.sbuf)
                    engine = nisa.engine.unknown if copy_engines is None else copy_engines(tile_h)
                    nisa.tensor_copy(dst=expert_out_f32, src=expert_out_psums[tile_h], engine=engine)
                    consume(tile_idx, tile_h * tile_H, tile_H_actual, H_slice, expert_out_f32)
            self.tile_done(tile_idx)
        return {"firsts": list(firsts.values()), "lasts": list(lasts.values()), "last": last}


# Token tiles of one moe_tkg_decode_per_expert call: 256 tokens, so a local expert has at most 2 blocks of 128 rows.
_PER_EXPERT_TILES = 2
# Late blocks of one local expert (pair p = 2 e + b of the late loop).
_LATE_BLOCKS = 2
# Default columns of one scatter / gather DMA of the per-expert combine: two down H tiles. The combine holds 4 FP32
# tiles of that width (2 gather-back, 2 staging) next to the expert loop; 512 takes half the SBUF and twice the DMAs.
PER_EXPERT_COMBINE_COLS = 2 * _DOWN_TILE_H
PER_EXPERT_SLOT_GATHERS = ("gpsimd", "scalar", "vector")
"""The engines of the per-expert slot gathers (``moe_tkg_decode_per_expert`` ``slot_gather``): ``gpsimd`` runs one
``nc_n_gather`` per D512 tile for the codes and one for the scales; ``scalar`` / ``vector`` run one tensor-indirection
``tensor_copy`` for the codes and one for the scales, on the Scalar or the Vector engine."""
# Partitions of one tensor-indirection group: offset i of a group comes from index[i % 16, i // 16].
_TI_GROUP = 16


def per_expert_scratch(E_L: int, D: int, name: str) -> dict:
    """Allocate the HBM scratch of ``moe_tkg_decode_per_expert``, for one call or for the calls of one kernel.

    ``rows``: FP32 ``[(E_L + 1) * 128, D]``, the expert outputs in slot order (block 0 of expert e at rows
    ``[128 e, 128 e + 128)``, the late block at rows ``[128 E_L, 128 E_L + 128)``). ``late_rows``: INT32
    ``[2 E_L, 128]``, the token of each slot of each late (expert, block) pair. ``readers``: the reads of the rows by
    the last call, which the stores of the next call that takes the scratch wait for.
    """
    return {
        "rows": nl.ndarray(((E_L + 1) * _PMAX, D), dtype=nl.float32, buffer=nl.private_hbm, name=f"{name}_rows"),
        "late_rows": nl.ndarray(
            (_LATE_BLOCKS * E_L, _PMAX), dtype=nl.int32, buffer=nl.private_hbm, name=f"{name}_late_rows"
        ),
        "readers": {},
    }


def moe_tkg_decode_per_expert(
    latent,
    local_affinities,
    expert_gate_up_weights: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_gate_up_weights_scale: nl.NkiTensor,
    expert_down_weights_scale: nl.NkiTensor,
    beta: float = 4.0,
    linear_beta: float = 25.0,
    weight_buffers: int = _PROJECTIONS_PER_EXPERT,
    scale_buffers: int = _GATE_UP_SCALE_BUFFERS,
    early_loads: bool = True,
    output: nl.NkiTensor = None,
    column_groups_out: nl.NkiTensor = None,
    scratch: dict = None,
    cross_expert_order: bool = False,
    transpose_block: bool = False,
    combine_cols: int = PER_EXPERT_COMBINE_COLS,
    slot_gather: str = "gpsimd",
    name: str = "per_expert",
) -> nl.NkiTensor:
    """Run each local expert on its own routed tokens of T = 256 tokens: the bits of ``moe_tkg_decode_token_tiles``.

    Not ``@nki.jit``: a kernel inlines it. The token-tiled call runs every expert on both 128-token tiles, so each
    expert's weights stream against 2 stationary blocks. Here each expert runs on one block of up to 128 slots that
    holds its routed tokens (block 0), in the static expert loop, so the 7 experts take 7 blocks. A token with an
    affinity for more than one local expert has one slot in each. No token is dropped: an expert with more than 128
    routed tokens adds a late block in one dynamic loop (``_late_blocks``), which takes no trip on a call without
    one. LNC1, Trn3.

    Slots: in each expert's list the tokens with another local expert come first, then the tokens of this expert
    alone, each in token order (``_per_expert_tables``), so the overflow of a hot expert is mostly its own tokens. A
    token whose position is 128 or more in any of its experts adds all its local terms in the late phase; every other
    token adds all of them in block 0.

    Bits: a slot gathers its token's MXFP8 codes and scales (GpSimd gathers of the call's codes), and a matmul output
    row depends only on its own stationary column, so each slot's expert output has the bits of that token's row of
    the token-tiled call. The FP32 slot outputs go to an HBM scratch (a scatter DMA) and come back in token order (a
    gather DMA); the sum then runs the instruction of the token-tiled call, ``acc = out * affinity + acc`` on the
    Vector engine, in expert order, the last expert writing the BF16 output. The affinity operand is 0 for a token
    that adds in the other phase (``out`` holds that token's own earlier value, finite for a finite latent), and the
    accumulator starts at +0 and is never -0, so a 0 term keeps every bit. Data moves between tokens only by copies,
    so a non-finite value stays in its own token's row.

    Args:
        latent, local_affinities, expert_gate_up_weights .. expert_down_weights_scale, beta, linear_beta,
            weight_buffers (2 or 3), scale_buffers, early_loads, output, column_groups_out: as
            ``moe_tkg_decode_token_tiles``, with 2 token tiles.
        scratch: the HBM scratch (``per_expert_scratch``); None (default) allocates one for the call.
        cross_expert_order, transpose_block: the expert order of ``moe_tkg_decode`` (3 whole weight buffers); the
            result does not change.
        combine_cols: the output columns of one scatter or gather DMA of the combine, a multiple of 512 (default
            1024, two down H tiles). Each token tile holds one FP32 ``[128, combine_cols]`` gather tile, and the
            scatters stage one or two (512 and 1024) such tiles; fewer columns take less SBUF and more DMAs (7 D /
            combine_cols scatters and twice as many gathers for each layer's block 0). The result does not change.
        slot_gather: the engine of the slot gathers (``PER_EXPERT_SLOT_GATHERS``): ``gpsimd`` (default) 2 D/512
            GpSimd gathers per slot tile; ``scalar`` or ``vector`` 2 tensor-indirection copies on that engine (a
            uint16 snake index of the slots' columns in every D512 tile). Copies either way: the result does not
            change.
        name: prefix of the DMA names, unique for each call.

    Returns:
        BF16 ``[256, D]`` unnormalized local routed-expert contribution, or ``column_groups_out`` when it is given.
    """
    kernel_assert(
        nl.program_ndim() == 0 or (nl.program_ndim() == 1 and nl.num_programs(0) == 1),
        "moe_tkg_decode_per_expert supports LNC1 only",
    )
    kernel_assert(nisa.get_nc_version() >= nisa.nc_version.gen4, "moe_tkg_decode_per_expert requires Trn3 or newer")
    E_L = expert_gate_up_weights.shape[0]
    I = expert_gate_up_weights.shape[-1]
    D = expert_down_weights.shape[-1]
    n_D512 = D // MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
    n_I512 = I // MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
    latent_tiles = _token_tiles_of(latent, D, "latent")
    affinity_tiles = _token_tiles_of(local_affinities, E_L, "local_affinities")
    n_tiles = len(latent_tiles)
    T = n_tiles * _PMAX
    # More than 256 tokens would need more than 2 blocks per expert: refused at trace time.
    kernel_assert(
        n_tiles == _PER_EXPERT_TILES and len(affinity_tiles) == n_tiles,
        f"moe_tkg_decode_per_expert takes {_PER_EXPERT_TILES} token tiles of {_PMAX}, got {n_tiles} and "
        f"{len(affinity_tiles)}",
    )
    for tile in latent_tiles:
        kernel_assert(tile.dtype == nl.bfloat16, f"Expected a BF16 latent, got {tile.dtype}")
    for tile in affinity_tiles:
        kernel_assert(
            tile.dtype == nl.bfloat16 or tile.dtype == nl.float32,
            f"Expected BF16 or FP32 local_affinities, got {tile.dtype}",
        )
    kernel_assert(
        expert_gate_up_weights.dtype == nl.float4_e2m1fn_x4 and expert_down_weights.dtype == nl.float4_e2m1fn_x4,
        f"Expected MXFP4 weights, got {expert_gate_up_weights.dtype} and {expert_down_weights.dtype}",
    )
    kernel_assert(
        expert_gate_up_weights_scale.dtype == nl.uint8 and expert_down_weights_scale.dtype == nl.uint8,
        f"Expected UINT8 MX scales, got {expert_gate_up_weights_scale.dtype} and {expert_down_weights_scale.dtype}",
    )
    kernel_assert(
        expert_gate_up_weights.shape == (E_L, _PMAX, 2, n_D512, I)
        and expert_gate_up_weights_scale.shape == (E_L, _MX_SCALE_PARTITIONS, 2, n_D512, I)
        and expert_down_weights.shape == (E_L, _PMAX, n_I512, D)
        and expert_down_weights_scale.shape == (E_L, _MX_SCALE_PARTITIONS, n_I512, D),
        f"Unexpected expert shapes {expert_gate_up_weights.shape} {expert_gate_up_weights_scale.shape} "
        f"{expert_down_weights.shape} {expert_down_weights_scale.shape}",
    )
    kernel_assert(2 <= E_L and E_L * _RAGGED_EXPERT_STRIDE <= _PMAX, f"Expected 2 to 8 local experts, got E_L={E_L}")
    kernel_assert(
        D % MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM == 0 and I % MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM == 0,
        f"Expected D and I divisible by 512, got D={D} I={I}",
    )
    kernel_assert(beta > 0.0 and linear_beta > 0.0, f"Expected beta, linear_beta > 0, got {beta}, {linear_beta}")
    # The late loop loads an expert into whole buffers (_load_expert).
    kernel_assert(
        weight_buffers in (_TWO_WEIGHT_BUFFERS, _PROJECTIONS_PER_EXPERT)
        and 1 <= scale_buffers <= _GATE_UP_SCALE_BUFFERS,
        f"Expected 2 or 3 whole weight buffers and 1 or 2 scale buffers, got {weight_buffers} and {scale_buffers}",
    )
    kernel_assert(
        not cross_expert_order or weight_buffers == _PROJECTIONS_PER_EXPERT,
        f"cross_expert_order takes {_PROJECTIONS_PER_EXPERT} whole weight buffers, got {weight_buffers}",
    )
    kernel_assert(not transpose_block or cross_expert_order, "transpose_block needs cross_expert_order")
    kernel_assert(
        combine_cols > 0 and combine_cols % _DOWN_TILE_H == 0,
        f"combine_cols must be a multiple of {_DOWN_TILE_H}, got {combine_cols}",
    )
    kernel_assert(
        slot_gather in PER_EXPERT_SLOT_GATHERS,
        f"slot_gather must be one of {PER_EXPERT_SLOT_GATHERS}, got {slot_gather!r}",
    )
    ti_engine = None if slot_gather == "gpsimd" else getattr(nisa.engine, slot_gather)
    if column_groups_out is not None:
        kernel_assert(output is None, "column_groups_out is the output of the call: it takes no output tensor")
        kernel_assert(
            tuple(column_groups_out.shape) == (_PMAX, T, D // _PMAX)
            and column_groups_out.dtype == nl.bfloat16
            and column_groups_out.buffer == nl.sbuf,
            f"Expected a BF16 SBUF column_groups_out {(_PMAX, T, D // _PMAX)}, got "
            f"{column_groups_out.dtype} {column_groups_out.buffer} {column_groups_out.shape}",
        )
    kernel_assert(
        output is None or (tuple(output.shape) == (T, D) and output.dtype == nl.bfloat16),
        f"Expected a BF16 output {(T, D)}, got {None if output is None else output.shape}",
    )
    if scratch is None:
        scratch = per_expert_scratch(E_L, D, name)
    rows_hbm, late_rows_hbm = scratch["rows"], scratch["late_rows"]
    kernel_assert(
        tuple(rows_hbm.shape) == ((E_L + 1) * _PMAX, D) and rows_hbm.dtype == nl.float32,
        f"Expected the FP32 [{(E_L + 1) * _PMAX}, {D}] scratch rows, got {rows_hbm.dtype} {rows_hbm.shape}",
    )
    experts = (expert_gate_up_weights, expert_down_weights, expert_gate_up_weights_scale, expert_down_weights_scale)
    stream_args = dict(
        expert_gate_up_weights=expert_gate_up_weights,
        expert_down_weights=expert_down_weights,
        expert_gate_up_weights_scale=expert_gate_up_weights_scale,
        expert_down_weights_scale=expert_down_weights_scale,
        weight_buffers=weight_buffers,
        scale_buffers=scale_buffers,
        zero_scales=True,
        use_order=cross_expert_order,
    )
    # As the static loop: the first load of each buffer before any work tile of this call exists.
    stream = _StaticExpertStream(**stream_args, first_loads=True) if early_loads else None

    # The MXFP8 codes and scales of both token tiles in one tensor (token t in column t), the column T zero: the
    # source of the slot gathers. Each tile runs the quantize of the token-tiled call (a quantize writes one contiguous
    # run per partition), and GpSimd copies its words into the tensor.
    if ti_engine is None:
        codes = nl.ndarray((_PMAX, n_D512, T + 1), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
        scales = nl.ndarray((_PMAX, n_D512, T + 1), dtype=nl.uint8, buffer=nl.sbuf)
    else:
        # A tensor-indirection source must be a whole 2D tile: the [D/512, T + 1] columns of a partition in a row.
        codes = nl.ndarray((_PMAX, n_D512 * (T + 1)), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf).reshape(
            (_PMAX, n_D512, T + 1)
        )
        scales = nl.ndarray((_PMAX, n_D512 * (T + 1)), dtype=nl.uint8, buffer=nl.sbuf).reshape((_PMAX, n_D512, T + 1))
    # The zero column: code 0 and scale byte 0 (a finite E8M0 scale; 0xFF would be NaN).
    nisa.memset(dst=codes.view(nl.uint32)[:, :, T : T + 1], value=0, engine=nisa.engine.gpsimd)
    nisa.memset(dst=scales[:, :, T : T + 1], value=0, engine=nisa.engine.gpsimd)
    for tile_idx, tile in enumerate(latent_tiles):
        latent_sb = tile
        if tile.buffer != nl.sbuf:
            latent_sb = _load_natural_latent(latent=tile, T=_PMAX, T_physical=_PMAX, D=D)
        else:
            kernel_assert(
                tuple(tile.shape) == (_PMAX, D) and tile.offset == 0,
                f"An SBUF latent tile must be a whole [{_PMAX}, {D}] tile (view offset 0), got {tile.shape}",
            )
        cols = nl.ds(tile_idx * _PMAX, _PMAX)
        tile_codes, tile_scales = _quantize_natural_latent(latent_sb=latent_sb, T=_PMAX, D=D)
        nisa.tensor_copy(
            dst=codes.view(nl.uint32)[:, :, cols], src=tile_codes.view(nl.uint32), engine=nisa.engine.gpsimd
        )
        nisa.tensor_copy(dst=scales[:, :, cols], src=tile_scales, engine=nisa.engine.gpsimd)
    if stream is None:
        stream = _StaticExpertStream(**stream_args, first_loads=False)

    # The affinities as the token-tiled call reads them: one contiguous load and an FP32 copy per tile.
    affinities = []
    for tile in affinity_tiles:
        affinity_raw_sb = nl.ndarray((_PMAX, E_L), dtype=tile.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=affinity_raw_sb, src=tile, dge_mode=nisa.dge_mode.none)
        affinity_f32_sb = nl.ndarray((_PMAX, E_L), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=affinity_f32_sb, src=affinity_raw_sb)
        affinities.append(affinity_f32_sb)
    consts = _per_expert_constants()
    if ti_engine is not None:
        consts.update(_snake_constants(n_D512, T))
    tables = _per_expert_tables(affinities, consts)

    staging = _GroupStaging(D, combine_cols)
    accumulators, backs, outs = [], [], []
    for _ in range(n_tiles):
        accumulator = nl.ndarray((_PMAX, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=accumulator, value=0.0)
        accumulators.append(accumulator)
        # The expert outputs of the tile's tokens, gathered back: row p is always token 128 k + p, and a token that
        # an expert skips keeps its own earlier value, so the tile is zeroed once.
        back = nl.ndarray((_PMAX, staging.cols), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=back, value=0.0)
        backs.append(back)
        outs.append(nl.ndarray((_PMAX, D), dtype=nl.bfloat16, buffer=nl.sbuf))
    if output is None and column_groups_out is None:
        output = nl.ndarray((T, D), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    # Block 0 of every expert in the static loop: two slot tiles in turn; the gather of expert e + 1 is issued before
    # the matmuls of expert e, and the combine of expert e after the up matmuls of expert e + 1.
    ring = [
        (
            nl.ndarray((_PMAX, n_D512, _PMAX), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf),
            nl.ndarray((_PMAX, n_D512, _PMAX), dtype=nl.uint8, buffer=nl.sbuf),
        )
        for _ in range(2)
    ]
    store_rows, stores = {}, {}  # stores: expert -> {first column: the group's scatter store}
    previous_readers = scratch["readers"]
    readers = {}

    def prepare(expert_idx):
        gather_idx, store_rows[expert_idx] = _slot_lists(tables, consts, expert_idx, snake=ti_engine is not None)
        _gather_slot_tile(codes, scales, gather_idx, *ring[expert_idx % 2], engine=ti_engine)

    def consume_of(expert_idx):
        return _group_store(
            rows_hbm,
            staging,
            lambda: store_rows[expert_idx],
            stores.setdefault(expert_idx, {}),
            f"{name}_e{expert_idx}",
            after=previous_readers.get(expert_idx, ()),
        )

    def combine(expert_idx):
        for tile_idx in range(n_tiles):
            readers.setdefault(expert_idx, []).extend(
                _combine_tile(
                    rows_hbm,
                    staging,
                    stores[expert_idx],
                    tables["back_rows"][tile_idx][:, expert_idx : expert_idx + 1],
                    tables["aff0"][tile_idx][:, expert_idx : expert_idx + 1],
                    backs[tile_idx],
                    accumulators[tile_idx],
                    outs[tile_idx] if expert_idx == E_L - 1 else None,
                    f"{name}_e{expert_idx}_t{tile_idx}",
                )
            )

    prepare(0)
    steps = [
        _PerExpertSteps(
            stream=stream,
            expert_idx=expert_idx,
            input_quant_sb=ring[expert_idx % 2][0],
            input_scale_sb=ring[expert_idx % 2][1],
            expert_gate_up_weights=expert_gate_up_weights,
            expert_down_weights=expert_down_weights,
            expert_gate_up_weights_scale=expert_gate_up_weights_scale,
            expert_down_weights_scale=expert_down_weights_scale,
            beta=beta,
            linear_beta=linear_beta,
            T=_PMAX,
            tiles=staging,
            scale_slots=cross_expert_order and len(stream.gate_up_scale_buffers) == 1,
            before_gate=(lambda e=expert_idx: prepare(e + 1)) if expert_idx + 1 < E_L else None,
            after_up=(lambda e=expert_idx: combine(e - 1)) if expert_idx > 0 else None,
        )
        for expert_idx in range(E_L)
    ]
    if cross_expert_order:
        _cross_expert_steps(steps, consume_of, transpose_block=transpose_block)
    else:
        for expert_idx in range(E_L):
            _run_expert_steps(steps[expert_idx], consume_of(expert_idx))
    combine(E_L - 1)

    _late_blocks(
        stream=stream,
        staging=staging,
        experts=experts,
        codes=codes,
        scales=scales,
        slot_tile=ring[0],
        tables=tables,
        consts=consts,
        affinities=affinities,
        rows_hbm=rows_hbm,
        late_rows_hbm=late_rows_hbm,
        accumulators=accumulators,
        backs=backs,
        outs=outs,
        beta=beta,
        linear_beta=linear_beta,
        name=name,
        ti_engine=ti_engine,
    )
    # The output after the late loop: block 0 wrote BF16 0 for the late tokens, the loop their values.
    for tile_idx in range(n_tiles):
        if column_groups_out is not None:
            _to_column_groups_at(outs[tile_idx], column_groups_out, tile_idx * _PMAX)
        else:
            nisa.dma_copy(
                dst=output[nl.ds(tile_idx * _PMAX, _PMAX), :], src=outs[tile_idx], dge_mode=nisa.dge_mode.none
            )
    scratch["readers"] = readers
    return output if column_groups_out is None else column_groups_out


def _run_expert_steps(steps: "_ExpertSteps", consume):
    """Issue one expert's steps in expert order (the order of ``_expert_ffn``)."""
    steps.gate_start()
    for chunk_idx in nl.sequential_range(steps.n_chunks):
        steps.gate_chunk(chunk_idx)
    steps.up()
    steps.down_start()
    for chunk_idx in range(min(_TRANSPOSE_LOOKAHEAD_CHUNKS, steps.n_transpose_chunks)):
        steps.transpose(chunk_idx)
    steps.down(consume=consume)


class _PerExpertSteps(_ExpertSteps):
    """The steps of one expert of ``moe_tkg_decode_per_expert``: ``gate_start`` first runs ``before_gate`` (the slot
    gather of the next expert) and ``up`` then runs ``after_up`` (the combine of the expert before), in either expert
    order. Constructing the steps issues no instruction."""

    def __init__(self, *args, before_gate=None, after_up=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.before_gate, self.after_up = before_gate, after_up

    def gate_start(self):
        if self.before_gate is not None:
            self.before_gate()
        super().gate_start()

    def up(self) -> dict:
        issued = super().up()
        if self.after_up is not None:
            self.after_up()
        return issued


class _GroupStaging:
    """FP32 SBUF tiles for the down PSUM copies of one expert, ``cols`` output columns (a combine group) each, so one
    scatter DMA stores a group (the ``tiles`` of ``_ExpertSteps``: role ``expert_out``; any other role gets a new
    tile). Groups of at most 1024 columns take two tiles in turn, so a group's copies do not wait for the scatter of
    the group before it. ``groups`` holds the ``(first column, columns)`` of each group."""

    def __init__(self, D: int, cols: int):
        self.tile_H = min(D, _DOWN_TILE_H)
        self.n_H_tiles = div_ceil(D, self.tile_H)
        self.cols = min(D, cols)
        self.groups = [(first, min(self.cols, D - first)) for first in range(0, D, self.cols)]
        n_buffers = 2 if self.cols <= 2 * self.tile_H else 1
        self.buffers = [nl.ndarray((_PMAX, self.cols), dtype=nl.float32, buffer=nl.sbuf) for _ in range(n_buffers)]
        self.uses = 0

    def tile(self, role: str, rows: int, cols: int) -> nl.NkiTensor:
        if role != "expert_out":
            return nl.ndarray((rows, cols), dtype=nl.float32, buffer=nl.sbuf)
        tile_h = self.uses % self.n_H_tiles
        self.uses += 1
        group, start = divmod(tile_h * self.tile_H, self.cols)
        return self.buffer(group)[0:rows, start : start + cols]

    def buffer(self, group: int) -> nl.NkiTensor:
        """The staging tile of combine group ``group`` of an expert."""
        return self.buffers[group % len(self.buffers)]

    def group_of(self, start: int, size: int):
        """The ``(first column, columns)`` of the group that columns ``[start, start + size)`` complete, else None."""
        for first, cols in self.groups:
            if start + size == first + cols:
                return first, cols
        return None


def _group_store(rows_hbm: nl.NkiTensor, staging: _GroupStaging, store_rows, stores: dict, name: str, after=()):
    """Return the ``consume`` of an expert's down H tiles: after each group's last tile, one scatter DMA stores the
    group's columns of the slots with a token into the scratch rows ``store_rows()`` (INT32 ``[128, 1]``, -1 skips a
    slot). ``stores`` gets each store by its first column; each store waits for ``after``."""
    D = rows_hbm.shape[1]

    def consume(start, size, H_slice, expert_out_f32):
        group = staging.group_of(start, size)
        if group is None:
            return
        first, cols = group
        store = _instruction(
            nisa.dma_copy(
                dst=rows_hbm.ap(
                    pattern=[[D, _PMAX], [1, cols]], offset=first, vector_offset=store_rows(), indirect_dim=0
                ),
                src=staging.buffer(first // staging.cols)[:, 0:cols],
                oob_mode=nisa.oob_mode.skip,
                dge_mode=nisa.dge_mode.swdge,
                name=f"{name}_store_c{first}",
            )
        )
        for reader in after:
            store.depends_on(reader, kind=nisa.dependency.runtime)
        stores[first] = store

    return consume


def _combine_tile(
    rows_hbm: nl.NkiTensor,
    staging: _GroupStaging,
    stores: dict,
    back_rows: nl.NkiTensor,
    affinity: nl.NkiTensor,
    back: nl.NkiTensor,
    accumulator: nl.NkiTensor,
    output_tile: nl.NkiTensor = None,
    name: str = None,
    late_output: tuple = None,
) -> list:
    """Add one expert's outputs to one token tile: per H group, a gather DMA of the tokens' scratch rows
    ``back_rows`` (INT32 ``[128, 1]``, -1 skips a token) into ``back`` (after the group's store, ``stores``), then per
    H tile the multiply-add of the token-tiled call, ``accumulator = back * affinity + accumulator``. With
    ``output_tile`` (the last expert) the multiply-add writes the BF16 output in place of the accumulator. With
    ``late_output`` = (BF16 output tile, UINT8 ``[128, 1]`` predicate) the multiply-add also writes BF16 columns,
    which a predicated copy moves into the output tile for the tokens of the predicate. Return the gathers."""
    D = rows_hbm.shape[1]
    gathers = []
    for first, cols in staging.groups:
        gather = _instruction(
            nisa.dma_copy(
                dst=back[:, 0:cols],
                src=rows_hbm.ap(pattern=[[D, _PMAX], [1, cols]], offset=first, vector_offset=back_rows, indirect_dim=0),
                oob_mode=nisa.oob_mode.skip,
                dge_mode=nisa.dge_mode.swdge,
                **({"name": f"{name}_back_c{first}"} if name else {}),
            )
        )
        gather.depends_on(stores[first], kind=nisa.dependency.runtime)
        gathers.append(gather)
        for start in range(first, first + cols, staging.tile_H):
            size = min(staging.tile_H, D - start)
            H_slice = nl.ds(start, size)
            back_cols = back[:, nl.ds(start - first, size)]
            if late_output is not None:
                output_bf16, predicate = late_output
                rounded = nl.ndarray((_PMAX, size), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.scalar_tensor_tensor(
                    dst=rounded,
                    data=back_cols,
                    op0=nl.multiply,
                    operand0=affinity,
                    op1=nl.add,
                    operand1=accumulator[:, H_slice],
                )
            nisa.scalar_tensor_tensor(
                dst=accumulator[:, H_slice] if output_tile is None else output_tile[:, H_slice],
                data=back_cols,
                op0=nl.multiply,
                operand0=affinity,
                op1=nl.add,
                operand1=accumulator[:, H_slice],
            )
            if late_output is not None:
                nisa.tensor_copy_predicated(
                    dst=output_bf16[:, H_slice],
                    src=rounded,
                    predicate=predicate.ap(pattern=[[1, _PMAX], [0, size]], offset=0),
                )
    return gathers


def _per_expert_constants() -> dict:
    """The data-independent tiles of the per-expert tables: the 0/1 stationary ``before`` (1 where p < j), the
    all-ones stationary ``ones``, the slot ids ``slot_ids`` (column j holds j) and ``slot_plus_one`` (partition j holds
    j + 1)."""
    fp32, bf16 = nl.float32, nl.bfloat16
    vector = nisa.engine.vector
    ramp = nl.ndarray((_PMAX, _PMAX), dtype=fp32, buffer=nl.sbuf)
    nisa.iota(dst=ramp, pattern=[[1, _PMAX]], offset=0, channel_multiplier=-1)
    before = nl.ndarray((_PMAX, _PMAX), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=before, data=ramp, op0=nl.greater, operand0=0.0, engine=vector)
    ones = nl.ndarray((_PMAX, _PMAX), dtype=bf16, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0, engine=vector)
    slot_ids = nl.ndarray((_PMAX, _PMAX), dtype=fp32, buffer=nl.sbuf)
    nisa.iota(dst=slot_ids, pattern=[[1, _PMAX]], offset=0, channel_multiplier=0)
    slot_plus_one = nl.ndarray((_PMAX, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.iota(dst=slot_plus_one, pattern=[[0, 1]], offset=1, channel_multiplier=1)
    return {"before": before, "ones": ones, "slot_ids": slot_ids, "slot_plus_one": slot_plus_one}


def _prefix_counts(flags16: list, consts: dict, total: bool = False) -> list:
    """For each token tile, the FP32 PSUM ``[128, n]`` count of the flagged tokens before each token (token order over
    the tiles) of the BF16 0/1 ``[128, n]`` flags; with ``total`` also every flagged token of the call. 0/1 matmuls:
    exact integers."""
    out = []
    n_tiles = len(flags16)
    for tile_idx in range(n_tiles):
        counts = nl.ndarray(flags16[tile_idx].shape, dtype=nl.float32, buffer=nl.psum)
        terms = [(consts["before"], flags16[tile_idx])] + [(consts["ones"], flags16[k]) for k in range(tile_idx)]
        for term_idx, (stationary, moving) in enumerate(terms):
            nisa.nc_matmul(dst=counts, stationary=stationary, moving=moving, accumulate=term_idx > 0)
        out.append(counts)
    return out


def _per_expert_tables(affinities: list, consts: dict) -> dict:
    """Return the slot tables of ``moe_tkg_decode_per_expert`` for the FP32 ``[128, E_L]`` affinities of its token
    tiles (token 128 k + p on partition p of tile k), each an SBUF tile per token tile, token on the partitions.

    A token takes expert e when its affinity for e is nonzero (the Vector engine's compare: NaN and Inf count; +0 and
    -0 do not). Expert e's list holds first the tokens with another local expert, then the tokens of e alone, each in
    token order: a token's position is the count of the tokens before it in the list, from 0/1 matmuls (exact). A
    token is late when its position in any of its experts is 128 or more.

    Returns (each a list over the tiles):
        ``hit``: FP32 0/1 ``[128, E_L]``. ``slots``: FP32 ``[128, E_L]``, the token's position in each expert's list,
        or -1. ``late``: FP32 0/1 ``[128, 1]``. ``early_slots``: FP32 ``[128, E_L]``, the
        token's block-0 slot in each expert, or -1 (no affinity, or a late token). ``aff0``: FP32 ``[128, E_L]``, the
        affinity where the token adds in block 0, else 0. ``back_rows``: INT32 ``[128, E_L]``, the token's scratch row
        ``128 e + slot`` in each expert, or -1. ``key1``: BF16 ``[128, 1]``, token + 1. ``key_rows``: BF16
        ``[128, 128]``, token + 1 on every column.
    """
    fp32, bf16, i32 = nl.float32, nl.bfloat16, nl.int32
    vector = nisa.engine.vector
    E_L = affinities[0].shape[1]
    hit, multi, classes16, shifted16 = [], [], [], []
    for affinity in affinities:
        hit_k = nl.ndarray((_PMAX, E_L), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=hit_k, data=affinity, op0=nl.not_equal, operand0=0.0, engine=vector)
        local = nl.ndarray((_PMAX, 1), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=local, op=nl.add, data=hit_k, axis=(1,))
        multi_k = nl.ndarray((_PMAX, 1), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=multi_k, data=local, op0=nl.greater, operand0=1.0, engine=vector)
        hit_multi = nl.ndarray((_PMAX, E_L), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=hit_multi, data=hit_k, op0=nl.multiply, operand0=multi_k, engine=vector)
        # [multi-expert | single-expert] flags, and [0 | multi-expert] for the multi count before the single ones.
        classes = nl.ndarray((_PMAX, 2 * E_L), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=classes[:, 0:E_L], src=hit_multi, engine=vector)
        nisa.tensor_tensor(dst=classes[:, E_L : 2 * E_L], data1=hit_k, data2=hit_multi, op=nl.subtract, engine=vector)
        shifted = nl.ndarray((_PMAX, 2 * E_L), dtype=bf16, buffer=nl.sbuf)
        nisa.memset(dst=shifted[:, 0:E_L], value=0.0, engine=vector)
        nisa.tensor_copy(dst=shifted[:, E_L : 2 * E_L], src=hit_multi, engine=vector)
        hit.append(hit_k)
        multi.append(multi_k)
        classes16.append(classes)
        shifted16.append(shifted)
    n_tiles = len(affinities)
    tables = {
        "hit": hit,
        "slots": [],
        "late": [],
        "early_slots": [],
        "aff0": [],
        "back_rows": [],
        "key1": [],
        "key_rows": [],
    }
    expert_rows = nl.ndarray((_PMAX, E_L), dtype=fp32, buffer=nl.sbuf)
    nisa.iota(dst=expert_rows, pattern=[[_PMAX, E_L]], offset=0, channel_multiplier=0)
    for tile_idx in range(n_tiles):
        # [multi rank | multi count + single rank]: the position in the list of each class.
        position_psum = nl.ndarray((_PMAX, 2 * E_L), dtype=fp32, buffer=nl.psum)
        terms = [(consts["before"], classes16[tile_idx])]
        terms += [(consts["ones"], classes16[k]) for k in range(tile_idx)]
        terms += [(consts["ones"], shifted16[k]) for k in range(n_tiles)]
        for term_idx, (stationary, moving) in enumerate(terms):
            nisa.nc_matmul(dst=position_psum, stationary=stationary, moving=moving, accumulate=term_idx > 0)
        positions = nl.ndarray((_PMAX, 2 * E_L), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=positions, src=position_psum, engine=vector)
        # position = multi ? multi rank : single position; slot = hit ? position : -1.
        diff = nl.ndarray((_PMAX, E_L), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=diff, data1=positions[:, 0:E_L], data2=positions[:, E_L : 2 * E_L], op=nl.subtract, engine=vector
        )
        position = nl.ndarray((_PMAX, E_L), dtype=fp32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(
            dst=position,
            data=diff,
            op0=nl.multiply,
            operand0=multi[tile_idx],
            op1=nl.add,
            operand1=positions[:, E_L : 2 * E_L],
        )
        slots = nl.ndarray((_PMAX, E_L), dtype=fp32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(
            dst=slots, data=position, op0=nl.add, operand0=1.0, op1=nl.multiply, operand1=hit[tile_idx]
        )
        nisa.tensor_scalar(dst=slots, data=slots, op0=nl.add, operand0=-1.0, engine=vector)
        last = nl.ndarray((_PMAX, 1), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=last, op=nl.maximum, data=slots, axis=(1,))
        late = nl.ndarray((_PMAX, 1), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=late, data=last, op0=nl.greater, operand0=_PMAX - 0.5, engine=vector)
        early = nl.ndarray((_PMAX, 1), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=early, data=late, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0, engine=vector
        )
        early_slots = nl.ndarray((_PMAX, E_L), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=early_slots, data=slots, op0=nl.add, operand0=1.0, op1=nl.multiply, operand1=early, engine=vector
        )
        nisa.tensor_scalar(dst=early_slots, data=early_slots, op0=nl.add, operand0=-1.0, engine=vector)
        # The affinity of the block-0 terms: a copy where the token adds in block 0, else 0.
        keep = nl.ndarray((_PMAX, E_L), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=keep, data=hit[tile_idx], op0=nl.multiply, operand0=early, engine=vector)
        aff0 = nl.ndarray((_PMAX, E_L), dtype=fp32, buffer=nl.sbuf)
        nisa.memset(dst=aff0, value=0.0, engine=vector)
        nisa.tensor_copy_predicated(dst=aff0, src=affinities[tile_idx], predicate=keep)
        # Scratch row 128 e + slot, or -1: (slot + 128 e + 1) * (slot >= 0) - 1.
        valid = nl.ndarray((_PMAX, E_L), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=valid, data=early_slots, op0=nl.greater_equal, operand0=0.0, engine=vector)
        rows_f = nl.ndarray((_PMAX, E_L), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=rows_f, data1=early_slots, data2=expert_rows, op=nl.add, engine=vector)
        nisa.scalar_tensor_tensor(dst=rows_f, data=rows_f, op0=nl.add, operand0=1.0, op1=nl.multiply, operand1=valid)
        back_rows = nl.ndarray((_PMAX, E_L), dtype=i32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=back_rows, data=rows_f, op0=nl.add, operand0=-1.0, engine=vector)
        # The token keys (token + 1 <= 256: exact in BF16).
        key = nl.ndarray((_PMAX, 1), dtype=fp32, buffer=nl.sbuf)
        nisa.iota(dst=key, pattern=[[0, 1]], offset=tile_idx * _PMAX + 1, channel_multiplier=1)
        key1 = nl.ndarray((_PMAX, 1), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=key1, src=key, engine=vector)
        key_rows = nl.ndarray((_PMAX, _PMAX), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=key_rows, data=consts["ones"], op0=nl.multiply, operand0=key, engine=vector)
        for field, value in (
            ("slots", slots),
            ("late", late),
            ("early_slots", early_slots),
            ("aff0", aff0),
            ("back_rows", back_rows),
            ("key1", key1),
            ("key_rows", key_rows),
        ):
            tables[field].append(value)
    return tables


def _slot_lists(tables: dict, consts: dict, expert_idx: int, snake: bool = False) -> tuple:
    """Return the block-0 lists of one expert: the UINT32 ``[128, 128]`` gather index (the token of each slot along
    the columns, on every partition; T for an empty slot, the zero column) and the INT32 ``[128, 1]`` scratch row of
    each slot (``128 e + slot``, or -1 for an empty slot). One-hot of the slots, then 0/1 matmuls with the token keys:
    exact integers. With ``snake`` the gather index is the UINT16 tensor-indirection index of ``_snake_index``."""
    fp32 = nl.float32
    vector = nisa.engine.vector
    early_slots = tables["early_slots"]
    n_tiles = len(early_slots)
    T = n_tiles * _PMAX
    onehots = []
    for tile_idx in range(n_tiles):
        onehot = nl.ndarray((_PMAX, _PMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=onehot,
            data=consts["slot_ids"],
            op0=nl.equal,
            operand0=early_slots[tile_idx][:, expert_idx : expert_idx + 1],
            engine=vector,
        )
        onehots.append(onehot)
    if not snake:
        # On every partition: token + 1 of each slot (0 when empty).
        columns_psum = nl.ndarray((_PMAX, _PMAX), dtype=fp32, buffer=nl.psum)
        for tile_idx in range(n_tiles):
            nisa.nc_matmul(
                dst=columns_psum,
                stationary=tables["key_rows"][tile_idx],
                moving=onehots[tile_idx],
                accumulate=tile_idx > 0,
            )
    # On the slots' partitions: token + 1 of the slot (0 when empty).
    slot_psum = nl.ndarray((_PMAX, 1), dtype=fp32, buffer=nl.psum)
    for tile_idx in range(n_tiles):
        nisa.nc_matmul(
            dst=slot_psum, stationary=onehots[tile_idx], moving=tables["key1"][tile_idx], accumulate=tile_idx > 0
        )
    if not snake:
        filled = nl.ndarray((_PMAX, _PMAX), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=filled, data=columns_psum, op0=nl.greater, operand0=0.0, engine=vector)
        # (key - (T + 1)) * filled + T: the token, or the zero column T.
        index_f = nl.ndarray((_PMAX, _PMAX), dtype=fp32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(
            dst=index_f, data=columns_psum, op0=nl.add, operand0=-(T + 1.0), op1=nl.multiply, operand1=filled
        )
        gather_idx = nl.ndarray((_PMAX, _PMAX), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=gather_idx, data=index_f, op0=nl.add, operand0=float(T), engine=vector)
    slot_filled = nl.ndarray((_PMAX, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=slot_filled, data=slot_psum, op0=nl.greater, operand0=0.0, engine=vector)
    if snake:
        # The token of the slot on its partition, or the zero column T: (key - (T + 1)) * filled + T.
        slot_token = nl.ndarray((_PMAX, 1), dtype=fp32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(
            dst=slot_token, data=slot_psum, op0=nl.add, operand0=-(T + 1.0), op1=nl.multiply, operand1=slot_filled
        )
        nisa.tensor_scalar(dst=slot_token, data=slot_token, op0=nl.add, operand0=float(T), engine=vector)
        gather_idx = _snake_index(slot_token, consts)
    # (slot + 1 + 128 e) * filled - 1.
    row_f = nl.ndarray((_PMAX, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=row_f,
        data=consts["slot_plus_one"],
        op0=nl.add,
        operand0=float(_PMAX * expert_idx),
        op1=nl.multiply,
        operand1=slot_filled,
        engine=vector,
    )
    store_rows = nl.ndarray((_PMAX, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=store_rows, data=row_f, op0=nl.add, operand0=-1.0, engine=vector)
    return gather_idx, store_rows


def _gather_slot_tile(
    codes: nl.NkiTensor, scales: nl.NkiTensor, gather_idx: nl.NkiTensor, slot_codes, slot_scales, engine=None
):
    """Gather the code words and the scales of the slots' tokens (``gather_idx``) from the call's ``[128, D/512,
    T + 1]`` codes and scales into a ``[128, D/512, 128]`` slot tile: GpSimd gathers along each partition, one per
    D512 tile (a gather reads at most 512 elements of a partition). With ``engine`` (Scalar or Vector) one
    tensor-indirection copy each for the codes and the scales, ``gather_idx`` the UINT16 snake index of
    ``_snake_index`` (the source a whole 2D tile)."""
    n_D512 = codes.shape[1]
    if engine is not None:
        columns = n_D512 * codes.shape[2]
        slots = n_D512 * slot_codes.shape[2]
        nisa.tensor_copy(
            dst=slot_codes.view(nl.uint32).reshape((_PMAX, slots)),
            src=codes.view(nl.uint32).reshape((_PMAX, columns)).indirect(gather_idx, slots),
            engine=engine,
        )
        nisa.tensor_copy(
            dst=slot_scales.reshape((_PMAX, slots)),
            src=scales.reshape((_PMAX, columns)).indirect(gather_idx, slots),
            engine=engine,
        )
        return
    for tile_h in range(n_D512):
        nisa.nc_n_gather(
            dst=slot_codes.view(nl.uint32)[:, tile_h, :], data=codes.view(nl.uint32)[:, tile_h, :], indices=gather_idx
        )
        nisa.nc_n_gather(dst=slot_scales[:, tile_h, :], data=scales[:, tile_h, :], indices=gather_idx)


def _snake_constants(n_D512: int, T: int) -> dict:
    """The data-independent tiles of ``_snake_index``: ``snake_lanes`` BF16 0/1 ``[128, 128]`` (1 where j % 16 ==
    p % 16, from a 0/1 matmul), ``snake_groups`` BF16 0/1 ``[128, 8]`` (1 where j // 16 == c) and ``snake_offsets``
    FP32 ``[128, D/512, 8]`` ((T + 1) d: the first column of D512 tile d in the call's codes)."""
    fp32, bf16 = nl.float32, nl.bfloat16
    vector = nisa.engine.vector
    groups = _PMAX // _TI_GROUP
    # lanes[r, x] = (x % 16 == r) for the 16 lanes r; snake_lanes = lanes^T lanes.
    lane_f = nl.ndarray((_TI_GROUP, _PMAX), dtype=fp32, buffer=nl.sbuf)
    nisa.iota(dst=lane_f, pattern=[[0, groups], [1, _TI_GROUP]], offset=0, channel_multiplier=-1)
    lanes = nl.ndarray((_TI_GROUP, _PMAX), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=lanes, data=lane_f, op0=nl.equal, operand0=0.0, engine=vector)
    lanes_psum = nl.ndarray((_PMAX, _PMAX), dtype=fp32, buffer=nl.psum)
    nisa.nc_matmul(dst=lanes_psum, stationary=lanes, moving=lanes)
    snake_lanes = nl.ndarray((_PMAX, _PMAX), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=snake_lanes, src=lanes_psum, engine=vector)
    # groups[j, c] = (0 <= j - 16 c < 16).
    group_f = nl.ndarray((_PMAX, groups), dtype=fp32, buffer=nl.sbuf)
    nisa.iota(dst=group_f, pattern=[[-_TI_GROUP, groups]], offset=0, channel_multiplier=1)
    low = nl.ndarray((_PMAX, groups), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=low, data=group_f, op0=nl.greater_equal, operand0=0.0, engine=vector)
    high = nl.ndarray((_PMAX, groups), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=high, data=group_f, op0=nl.less, operand0=_TI_GROUP - 0.5, engine=vector)
    snake_groups = nl.ndarray((_PMAX, groups), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=snake_groups, data1=low, data2=high, op=nl.multiply, engine=vector)
    snake_offsets = nl.ndarray((_PMAX, n_D512, groups), dtype=fp32, buffer=nl.sbuf)
    nisa.iota(dst=snake_offsets, pattern=[[T + 1, n_D512], [0, groups]], offset=0, channel_multiplier=0)
    return {"snake_lanes": snake_lanes, "snake_groups": snake_groups, "snake_offsets": snake_offsets}


def _snake_index(slot_token: nl.NkiTensor, consts: dict) -> nl.NkiTensor:
    """Return the UINT16 ``[128, 8 D/512]`` tensor-indirection index of a slot tile from the FP32 ``[128, 1]`` token
    of each slot on its partition (T for the zero column).

    Offset i = 128 d + j of a 16-partition group comes from index[i % 16, i // 16], and every partition takes the
    same offsets, so index[p, 8 d + c] = (T + 1) d + token[16 c + p % 16]. One 0/1 matmul moves the tokens there:
    moving[j, c] = token[j] (j // 16 == c), stationary[j, p] = (j % 16 == p % 16), so each output is one token
    (exact in BF16 up to 256)."""
    groups = _PMAX // _TI_GROUP
    n_D512 = consts["snake_offsets"].shape[1]
    moving = nl.ndarray((_PMAX, groups), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=moving, data=consts["snake_groups"], op0=nl.multiply, operand0=slot_token, engine=nisa.engine.vector
    )
    tokens_psum = nl.ndarray((_PMAX, groups), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=tokens_psum, stationary=consts["snake_lanes"], moving=moving)
    index = nl.ndarray((_PMAX, n_D512 * groups), dtype=nl.uint16, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=index.reshape((_PMAX, n_D512, groups)),
        data1=tokens_psum.ap(pattern=[[groups, _PMAX], [0, n_D512], [1, groups]], offset=0),
        data2=consts["snake_offsets"],
        op=nl.add,
        engine=nisa.engine.vector,
    )
    return index


def _late_blocks(
    stream: "_StaticExpertStream",
    staging: _GroupStaging,
    experts: tuple,
    codes: nl.NkiTensor,
    scales: nl.NkiTensor,
    slot_tile: tuple,
    tables: dict,
    consts: dict,
    affinities: list,
    rows_hbm: nl.NkiTensor,
    late_rows_hbm: nl.NkiTensor,
    accumulators: list,
    backs: list,
    outs: list,
    beta: float,
    linear_beta: float,
    name: str,
    ti_engine=None,
):
    """Run the late tokens of ``moe_tkg_decode_per_expert``: one dynamic loop over the active (expert, block) pairs.

    The late tokens of expert e (a token is late when its position in any of its experts is 128 or more) take the
    pairs 2 e and 2 e + 1, 128 slots each, in token order. A pair is active when it holds a token; the loop runs the
    active pairs in ascending order, so in expert order, and a call without a late token takes no trip. Each trip
    loads the pair's expert by a runtime id (``_load_expert``), gathers its slots' codes, runs the expert, stores the
    slot rows, and adds them to each token tile with the multiply-add of block 0; the affinity is the token's where it
    is in the pair, else 0. A late token adds no block-0 term (its block-0 affinity is 0), so each token sums in one
    phase. Each trip also writes the BF16 result of its tokens into the output tiles (a predicated copy), so a token's
    last trip writes its value. ``ti_engine``: the slot gathers as tensor-indirection copies on that engine
    (``_gather_slot_tile``).
    """
    fp32, bf16, i32 = nl.float32, nl.bfloat16, nl.int32
    vector = nisa.engine.vector
    E_L = affinities[0].shape[1]
    n_tiles = len(affinities)
    T = n_tiles * _PMAX
    n_pairs = _LATE_BLOCKS * E_L
    late_first = E_L * _PMAX
    late16, late_f = [], []
    for tile_idx in range(n_tiles):
        flags = nl.ndarray((_PMAX, E_L), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=flags, data=tables["hit"][tile_idx], op0=nl.multiply, operand0=tables["late"][tile_idx], engine=vector
        )
        flags16 = nl.ndarray((_PMAX, E_L), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=flags16, src=flags, engine=vector)
        late_f.append(flags)
        late16.append(flags16)
    count_psum = nl.ndarray((_PMAX, E_L), dtype=fp32, buffer=nl.psum)
    for tile_idx in range(n_tiles):
        nisa.nc_matmul(dst=count_psum, stationary=consts["ones"], moving=late16[tile_idx], accumulate=tile_idx > 0)
    counts = nl.ndarray((1, E_L), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=counts, src=count_psum[0:1, :], engine=vector)
    # Pair 2 e + b is active when expert e has more than 128 b late tokens.
    pair_flags = nl.ndarray((1, n_pairs), dtype=fp32, buffer=nl.sbuf)
    for block in range(_LATE_BLOCKS):
        nisa.tensor_scalar(
            dst=pair_flags.ap(pattern=[[n_pairs, 1], [_LATE_BLOCKS, E_L]], offset=block),
            data=counts,
            op0=nl.greater,
            operand0=float(_PMAX * block),
            engine=vector,
        )
    trips = nl.ndarray((1, n_pairs + 1), dtype=i32, buffer=nl.sbuf)
    nisa.nonzero_with_count(dst=trips, src=pair_flags, index_offset=0, padding_val=n_pairs)
    trip_count = nl.ndarray((1, 1), dtype=i32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=trip_count, src=trips[0:1, n_pairs : n_pairs + 1], engine=vector)
    # The expert of each trip: the count of m in [1, E_L) with pair >= 2 m.
    pairs_f = nl.ndarray((1, n_pairs), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pairs_f, src=trips[0:1, 0:n_pairs], engine=vector)
    thresholds = nl.ndarray((1, n_pairs, E_L - 1), dtype=fp32, buffer=nl.sbuf)
    nisa.iota(
        dst=thresholds, pattern=[[0, n_pairs], [_LATE_BLOCKS, E_L - 1]], offset=_LATE_BLOCKS, channel_multiplier=0
    )
    passed = nl.ndarray((1, n_pairs, E_L - 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=passed,
        data1=pairs_f.ap(pattern=[[n_pairs, 1], [1, n_pairs], [0, E_L - 1]], offset=0),
        data2=thresholds,
        op=nl.greater_equal,
        engine=vector,
    )
    trip_experts_f = nl.ndarray((1, n_pairs), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=trip_experts_f, op=nl.add, data=passed, axis=(2,))
    trip_experts = nl.ndarray((1, n_pairs), dtype=i32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=trip_experts, src=trip_experts_f, engine=vector)

    # The late lists: the late tokens of expert e on partition 16 e in token order (T past the count), in HBM.
    flags_by_expert = nl.ndarray((_PMAX, T), dtype=fp32, buffer=nl.sbuf)
    for tile_idx in range(n_tiles):
        spread = nl.ndarray((_PMAX, _PMAX), dtype=fp32, buffer=nl.sbuf)
        nisa.memset(dst=spread, value=0.0, engine=vector)
        nisa.tensor_copy(
            dst=spread.ap(pattern=[[_PMAX, _PMAX], [_RAGGED_EXPERT_STRIDE, E_L]], offset=0), src=late_f[tile_idx]
        )
        spread_psum = nl.ndarray((_PMAX, _PMAX), dtype=fp32, buffer=nl.psum)
        nisa.nc_transpose(dst=spread_psum, data=spread)
        nisa.tensor_copy(dst=flags_by_expert[:, nl.ds(tile_idx * _PMAX, _PMAX)], src=spread_psum, engine=vector)
    found = nl.ndarray((_PMAX, T + 1), dtype=i32, buffer=nl.sbuf)
    nisa.memset(dst=found, value=T)
    nisa.nonzero_with_count(dst=found, src=flags_by_expert, index_offset=0, padding_val=T)
    late_lists = late_rows_hbm.reshape((E_L, T))
    for expert_idx in range(E_L):
        row = expert_idx * _RAGGED_EXPERT_STRIDE
        nisa.dma_copy(
            dst=late_lists[expert_idx : expert_idx + 1, :],
            src=found[row : row + 1, 0:T],
            dge_mode=nisa.dge_mode.none,
            name=f"{name}_late_list_e{expert_idx}",
        )

    # On the token partitions, for each pair: the token's late scratch row (128 E_L + slot, or -1) and its affinity
    # (0 outside the pair).
    late_rows, late_affinity = [], []
    rank_psums = _prefix_counts(late16, consts)
    for tile_idx in range(n_tiles):
        ranks = nl.ndarray((_PMAX, E_L), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ranks, src=rank_psums[tile_idx], engine=vector)
        first_block = nl.ndarray((_PMAX, E_L), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=first_block, data=ranks, op0=nl.less, operand0=_PMAX - 0.5, engine=vector)
        in_pair = [nl.ndarray((_PMAX, E_L), dtype=fp32, buffer=nl.sbuf) for _ in range(_LATE_BLOCKS)]
        nisa.tensor_tensor(dst=in_pair[0], data1=late_f[tile_idx], data2=first_block, op=nl.multiply, engine=vector)
        nisa.tensor_tensor(dst=in_pair[1], data1=late_f[tile_idx], data2=in_pair[0], op=nl.subtract, engine=vector)
        rows_i = nl.ndarray((_PMAX, n_pairs), dtype=i32, buffer=nl.sbuf)
        affinity = nl.ndarray((_PMAX, n_pairs), dtype=fp32, buffer=nl.sbuf)
        nisa.memset(dst=affinity, value=0.0, engine=vector)
        for block in range(_LATE_BLOCKS):
            row_f = nl.ndarray((_PMAX, E_L), dtype=fp32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(
                dst=row_f,
                data=ranks,
                op0=nl.add,
                operand0=float(late_first - _PMAX * block + 1),
                op1=nl.multiply,
                operand1=in_pair[block],
            )
            nisa.tensor_scalar(
                dst=rows_i.ap(pattern=[[n_pairs, _PMAX], [_LATE_BLOCKS, E_L]], offset=block),
                data=row_f,
                op0=nl.add,
                operand0=-1.0,
                engine=vector,
            )
            predicate = nl.ndarray((_PMAX, E_L), dtype=nl.uint8, buffer=nl.sbuf)
            nisa.tensor_copy(dst=predicate, src=in_pair[block], engine=vector)
            nisa.tensor_copy_predicated(
                dst=affinity.ap(pattern=[[n_pairs, _PMAX], [_LATE_BLOCKS, E_L]], offset=block),
                src=affinities[tile_idx],
                predicate=predicate,
            )
        late_rows.append(rows_i)
        late_affinity.append(affinity)
    late_store_base = nl.ndarray((_PMAX, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.iota(dst=late_store_base, pattern=[[0, 1]], offset=late_first + 1, channel_multiplier=1)

    trip_reg = nisa.register_alloc()
    nisa.register_load(trip_reg, trip_count)
    slot_codes, slot_scales = slot_tile
    sync = {"dge_mode": nisa.dge_mode.hwdge, "engine": nisa.engine.sync}

    def trip(iteration):
        pair = nl.ndarray((1, 1), dtype=i32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=pair,
            src=trips.ap(pattern=[[n_pairs + 1, 1], [1, 1]], offset=0, scalar_offset=iteration, indirect_dim=1),
            **sync,
        )
        expert = nl.ndarray((1, 1), dtype=i32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=expert,
            src=trip_experts.ap(pattern=[[n_pairs, 1], [1, 1]], offset=0, scalar_offset=iteration, indirect_dim=1),
            **sync,
        )
        loaded = _load_expert(stream, expert, *experts)
        if ti_engine is None:
            tokens = nl.ndarray((_PMAX, _PMAX), dtype=i32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=tokens,
                src=late_rows_hbm.ap(pattern=[[0, _PMAX], [1, _PMAX]], offset=0, scalar_offset=pair, indirect_dim=0),
                **sync,
            )
        slot_tokens = nl.ndarray((_PMAX, 1), dtype=i32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=slot_tokens,
            src=late_rows_hbm.ap(pattern=[[1, _PMAX], [1, 1]], offset=0, scalar_offset=pair, indirect_dim=0),
            **sync,
        )
        if ti_engine is None:
            _gather_slot_tile(codes, scales, tokens.view(nl.uint32), slot_codes, slot_scales)
        else:
            slot_token_f = nl.ndarray((_PMAX, 1), dtype=fp32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=slot_token_f, src=slot_tokens, engine=vector)
            _gather_slot_tile(
                codes, scales, _snake_index(slot_token_f, consts), slot_codes, slot_scales, engine=ti_engine
            )
        filled = nl.ndarray((_PMAX, 1), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=filled, data=slot_tokens, op0=nl.less, operand0=T - 0.5, engine=vector)
        store_rows = nl.ndarray((_PMAX, 1), dtype=i32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=store_rows,
            data=late_store_base,
            op0=nl.multiply,
            operand0=filled,
            op1=nl.add,
            operand1=-1.0,
            engine=vector,
        )
        steps = _ExpertSteps(
            stream,
            0,
            slot_codes,
            slot_scales,
            *experts,
            beta=beta,
            linear_beta=linear_beta,
            T=_PMAX,
            loaded=loaded,
            tiles=staging,
        )
        stores = {}
        _run_expert_steps(steps, _group_store(rows_hbm, staging, lambda: store_rows, stores, f"{name}_late"))
        for tile_idx in range(n_tiles):
            back_rows = nl.ndarray((_PMAX, 1), dtype=i32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=back_rows,
                src=late_rows[tile_idx].ap(
                    pattern=[[n_pairs, _PMAX], [1, 1]], offset=0, scalar_offset=pair, indirect_dim=1
                ),
                **sync,
            )
            affinity = nl.ndarray((_PMAX, 1), dtype=fp32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=affinity,
                src=late_affinity[tile_idx].ap(
                    pattern=[[n_pairs, _PMAX], [1, 1]], offset=0, scalar_offset=pair, indirect_dim=1
                ),
                **sync,
            )
            predicate = nl.ndarray((_PMAX, 1), dtype=nl.uint8, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=predicate, data=back_rows, op0=nl.greater_equal, operand0=0.0, engine=vector)
            _combine_tile(
                rows_hbm,
                staging,
                stores,
                back_rows,
                affinity,
                backs[tile_idx],
                accumulators[tile_idx],
                late_output=(outs[tile_idx], predicate),
            )

    nl.fori_loop(0, trip_reg, trip)


_RAGGED_MAX_ROWS = 2 * _PMAX
# GpSimd partition of the row list of local expert e: nonzero_with_count works on partitions 0, 16, ..., 112.
_RAGGED_EXPERT_STRIDE = 16
_RAGGED_OUTPUT_TILE_H = 1024


def moe_tkg_decode_ragged(
    latent_rows: nl.NkiTensor,
    local_affinities: nl.NkiTensor,
    expert_gate_up_weights: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_gate_up_weights_scale: nl.NkiTensor,
    expert_down_weights_scale: nl.NkiTensor,
    accumulator: nl.NkiTensor,
    output: nl.NkiTensor,
    expert_rows: int,
    beta: float = 4.0,
    linear_beta: float = 25.0,
    weight_buffers: int = _PROJECTIONS_PER_EXPERT,
    scale_buffers: int = _GATE_UP_SCALE_BUFFERS,
    expert_counts: nl.NkiTensor = None,
) -> nl.NkiTensor:
    """Compute the local routed-expert contribution of C rows, each local expert on its own rows only.

    The ragged form of the static loop, for rows that a token-to-expert exchange packs (C may exceed 128). Expert e
    runs on the rows with a nonzero affinity for it: ``nonzero_with_count`` gives the rows, one indirect DMA gathers
    them into a tile of ``expert_rows`` rows, and the static-loop matmuls run on that tile. So the weights stream once
    and each matmul has at most 128 stationary columns, whatever C is. Each expert adds ``affinity * output`` to the
    FP32 ``accumulator`` rows with an indirect DMA add, in expert order, and the result rounds once to BF16. Not
    ``@nki.jit``: a caller kernel inlines it. LNC1, Trn3.

    Args:
        latent_rows: BF16 ``[C, D]`` HBM rows (or their UINT16 bits), C <= 256. Rows that no expert takes may hold
            any value.
        local_affinities: FP32 ``[C, E_L]`` HBM affinities; zero outside the local top-k (and for unused rows).
        expert_gate_up_weights .. expert_down_weights_scale: the MXFP4 experts, as ``moe_tkg_decode``.
        accumulator: FP32 ``[C, D]`` HBM scratch; the call zeroes it first.
        output: BF16 ``[C, D]`` HBM destination.
        expert_rows: rows of each expert tile, a multiple of 4 up to 128. An expert with more rows drops the rows
            past ``expert_rows`` (``expert_counts`` shows it).
        beta, linear_beta, weight_buffers, scale_buffers: as ``moe_tkg_decode``.
        expert_counts: None, or an INT32 ``[E_L, 1]`` HBM tensor that gets the rows of each expert.

    Returns:
        ``output``.
    """
    C, D = latent_rows.shape
    E_L = expert_gate_up_weights.shape[0]
    kernel_assert(1 <= C <= _RAGGED_MAX_ROWS, f"Expected 1 <= C <= {_RAGGED_MAX_ROWS} rows, got {C}")
    kernel_assert(
        expert_rows % _q_width == 0 and _q_width <= expert_rows <= _PMAX,
        f"expert_rows must be a multiple of {_q_width} up to {_PMAX}, got {expert_rows}",
    )
    kernel_assert(E_L * _RAGGED_EXPERT_STRIDE <= _PMAX, f"At most 8 local experts, got {E_L}")
    kernel_assert(tuple(local_affinities.shape) == (C, E_L), f"Expected affinities {(C, E_L)}")
    kernel_assert(tuple(accumulator.shape) == (C, D) and accumulator.dtype == nl.float32, "Expected FP32 [C, D]")
    kernel_assert(tuple(output.shape) == (C, D) and output.dtype == nl.bfloat16, "Expected a BF16 [C, D] output")
    stream = _StaticExpertStream(
        expert_gate_up_weights=expert_gate_up_weights,
        expert_down_weights=expert_down_weights,
        expert_gate_up_weights_scale=expert_gate_up_weights_scale,
        expert_down_weights_scale=expert_down_weights_scale,
        weight_buffers=weight_buffers,
        scale_buffers=scale_buffers,
        first_loads=True,
    )
    rows_idx = _ragged_row_lists(local_affinities, expert_rows, expert_counts)
    # Each indirect add waits for the previous write of its columns (the zeroing, then the add of the previous
    # expert), so the sum of a row runs in expert order whatever the DMA queues do.
    zeroing = _zero_rows_f32(accumulator)
    last_write = {}
    for expert_idx in nl.sequential_range(E_L):
        offsets = rows_idx[:, expert_idx * _RAGGED_EXPERT_STRIDE : expert_idx * _RAGGED_EXPERT_STRIDE + 1]
        # The rows may be BF16 or their UINT16 bits: the gather keeps the dtype, and the tile is read as BF16.
        latent_sb = nl.ndarray((expert_rows, D), dtype=latent_rows.dtype, buffer=nl.sbuf)
        nisa.memset(dst=latent_sb, value=0)
        nisa.dma_copy(
            dst=latent_sb,
            src=latent_rows.ap(pattern=[[D, expert_rows], [1, D]], offset=0, vector_offset=offsets, indirect_dim=0),
            oob_mode=nisa.oob_mode.skip,
            dge_mode=nisa.dge_mode.swdge,
        )
        if latent_sb.dtype != nl.bfloat16:
            latent_sb = latent_sb.view(nl.bfloat16)
        affinity_sb = nl.ndarray((expert_rows, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=affinity_sb, value=0.0)
        nisa.dma_copy(
            dst=affinity_sb,
            src=local_affinities.ap(
                pattern=[[E_L, expert_rows], [1, 1]], offset=expert_idx, vector_offset=offsets, indirect_dim=0
            ),
            oob_mode=nisa.oob_mode.skip,
            dge_mode=nisa.dge_mode.swdge,
        )
        input_quant_sb, input_scale_sb = _quantize_natural_latent(latent_sb=latent_sb, T=expert_rows, D=D)

        def consume(start, size, H_slice, expert_out_f32, offsets=offsets, affinity_sb=affinity_sb):
            scaled = nl.ndarray((expert_rows, size), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=scaled, data=expert_out_f32, op0=nl.multiply, operand0=affinity_sb)
            rows = accumulator.ap(
                pattern=[[D, expert_rows], [1, size]], offset=start, vector_offset=offsets, indirect_dim=0
            )
            add = _instruction(
                nisa.dma_compute(
                    dst=rows, srcs=[rows, scaled], scales=[1.0, 1.0], reduce_op=nl.add, oob_mode=nisa.oob_mode.skip
                )
            )
            for previous in last_write.get(start, zeroing):
                add.depends_on(previous, kind=nisa.dependency.runtime)
            last_write[start] = [add]

        _expert_ffn(
            stream=stream,
            expert_idx=expert_idx,
            input_quant_sb=input_quant_sb,
            input_scale_sb=input_scale_sb,
            expert_gate_up_weights=expert_gate_up_weights,
            expert_down_weights=expert_down_weights,
            expert_gate_up_weights_scale=expert_gate_up_weights_scale,
            expert_down_weights_scale=expert_down_weights_scale,
            beta=beta,
            linear_beta=linear_beta,
            T=expert_rows,
            consume=consume,
        )
    _round_rows_bf16(accumulator, output, [add for adds in last_write.values() for add in adds])
    return output


_BLOCK_ROWS = 32
_BLOCKS_MAX_ROWS = 8 * _PMAX


def moe_tkg_decode_blocks(
    latent_rows: nl.NkiTensor,
    local_affinities: nl.NkiTensor,
    expert_gate_up_weights: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_gate_up_weights_scale: nl.NkiTensor,
    expert_down_weights_scale: nl.NkiTensor,
    accumulator: nl.NkiTensor,
    output: nl.NkiTensor,
    row_table: nl.NkiTensor,
    block_rows: int = _BLOCK_ROWS,
    beta: float = 4.0,
    linear_beta: float = 25.0,
    scale_buffers: int = _GATE_UP_SCALE_BUFFERS,
    expert_counts: nl.NkiTensor = None,
    name: str = "blocks",
    weight_buffers: int = _PROJECTIONS_PER_EXPERT,
) -> nl.NkiTensor:
    """Compute the local routed-expert contribution of C rows, each local expert on its own rows in blocks.

    The no-drop form of ``moe_tkg_decode_ragged``. Expert e runs on its rows (``nonzero_with_count``) in blocks of
    ``block_rows`` rows. Block 0 of every expert runs in the static expert loop, so the weight stream overlaps the
    matmuls as in the ragged form. It takes the rows that are among the first ``block_rows`` rows of every local
    expert that they use. Each other row adds all its local terms in the later blocks: a dynamic loop of each expert
    over its late rows. So each row adds its terms in one phase, in expert order, and its sum does not depend on the
    other rows of the call; no row is dropped, and the cost tracks the rows. Each iteration loads the weights and
    scales of expert e and writes every tile that it reads, so no SBUF value carries from one iteration to the next.
    Each block adds ``affinity * output`` to the FP32 ``accumulator`` rows with an indirect DMA add, and the result
    rounds once to BF16. Not ``@nki.jit``: a caller kernel inlines it. LNC1, Trn3.

    Args:
        latent_rows .. expert_down_weights_scale, accumulator, output, beta, linear_beta, scale_buffers,
            expert_counts: as ``moe_tkg_decode_ragged``; C <= 1024, a multiple of ``block_rows``.
        row_table: INT32 ``[E_L * C, 1]`` HBM scratch, the row list of each expert for the dynamic loops.
        block_rows: rows of each block, a multiple of 4 up to 128.
        name: prefix of the accumulator DMA names (``<name>_acc_*``), unique for each call.
        weight_buffers: SBUF weight buffers of the expert loop, 2 or 3. With 2, a later block loads the down
            weights into the gate buffer after the gate and up matmuls.

    Returns:
        ``output``.
    """
    C, D = latent_rows.shape
    E_L = expert_gate_up_weights.shape[0]
    kernel_assert(
        block_rows % _q_width == 0 and _q_width <= block_rows <= _PMAX,
        f"block_rows must be a multiple of {_q_width} up to {_PMAX}, got {block_rows}",
    )
    kernel_assert(
        C % block_rows == 0 and C <= _BLOCKS_MAX_ROWS,
        f"Expected C <= {_BLOCKS_MAX_ROWS} rows, a multiple of {block_rows}, got {C}",
    )
    kernel_assert(E_L * _RAGGED_EXPERT_STRIDE <= _PMAX, f"At most 8 local experts, got {E_L}")
    kernel_assert(tuple(local_affinities.shape) == (C, E_L), f"Expected affinities {(C, E_L)}")
    kernel_assert(tuple(accumulator.shape) == (C, D) and accumulator.dtype == nl.float32, "Expected FP32 [C, D]")
    kernel_assert(tuple(output.shape) == (C, D) and output.dtype == nl.bfloat16, "Expected a BF16 [C, D] output")
    kernel_assert(tuple(row_table.shape) == (E_L * C, 1), f"Expected an INT32 [{E_L * C}, 1] row table")
    kernel_assert(weight_buffers in (2, 3), f"Expected 2 or 3 weight buffers, got {weight_buffers}")
    experts = (expert_gate_up_weights, expert_down_weights, expert_gate_up_weights_scale, expert_down_weights_scale)
    dynamic = C > block_rows
    stream = _StaticExpertStream(
        *experts,
        weight_buffers=weight_buffers,
        scale_buffers=scale_buffers,
        first_loads=True,
        zero_scales=dynamic,
    )
    tiles = _AlternatingTiles(
        block_rows,
        {
            "gate_sigmoid": _GATE_UP_CHUNK_I,
            "up_tanh": _GATE_UP_CHUNK_I,
            "expert_out": _DOWN_TILE_H,
            "scaled": _DOWN_TILE_H,
        },
    )
    flags = _ragged_flags(local_affinities)
    if dynamic:
        early_flags, late_flags = _phase_flags(local_affinities, flags, block_rows)
        found = _found_of(early_flags)
        found_late = _found_of(late_flags)
        if expert_counts is not None:
            _found_of(flags, expert_counts)
    else:
        found = _found_of(flags, expert_counts)
    rows_idx = _first_rows(found, block_rows)
    zeroing = _zero_rows_f32(accumulator, f"{name}_acc_zero")
    # Block 0: each indirect add waits for the previous write of its columns, so a row sums in expert order.
    last_write = {}
    for expert_idx in nl.sequential_range(E_L):
        offsets = rows_idx[:, expert_idx * _RAGGED_EXPERT_STRIDE : expert_idx * _RAGGED_EXPERT_STRIDE + 1]
        latent_sb = nl.ndarray((block_rows, D), dtype=latent_rows.dtype, buffer=nl.sbuf)
        nisa.memset(dst=latent_sb, value=0)
        affinity_sb = nl.ndarray((block_rows, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=affinity_sb, value=0.0)
        _gather_block(latent_rows, local_affinities, offsets, expert_idx, latent_sb, affinity_sb)
        input_quant_sb, input_scale_sb = _quantize_natural_latent(latent_sb=_as_bf16(latent_sb), T=block_rows, D=D)
        _expert_ffn(
            stream,
            expert_idx,
            input_quant_sb,
            input_scale_sb,
            *experts,
            beta=beta,
            linear_beta=linear_beta,
            T=block_rows,
            consume=_block_add(
                accumulator,
                offsets,
                affinity_sb,
                tiles,
                f"{name}_acc_b0_e{expert_idx}",
                (last_write, zeroing),
            ),
            tiles=tiles,
        )
    block0_adds = [add for adds in last_write.values() for add in adds]
    if dynamic:
        _later_blocks(
            stream,
            tiles,
            found_late,
            latent_rows,
            local_affinities,
            experts,
            accumulator,
            row_table,
            beta,
            linear_beta,
            block0_adds,
            name,
        )
    _round_rows_bf16(accumulator, output, block0_adds, f"{name}_acc_round")
    return output


def _later_blocks(
    stream: "_StaticExpertStream",
    tiles: "_AlternatingTiles",
    found: nl.NkiTensor,
    latent_rows: nl.NkiTensor,
    local_affinities: nl.NkiTensor,
    experts: tuple,
    accumulator: nl.NkiTensor,
    row_table: nl.NkiTensor,
    beta: float,
    linear_beta: float,
    block0_adds: list,
    name: str,
):
    """Run the late rows of each expert (``found``) in blocks, in a dynamic loop over first rows ``[0, count)``.

    No SBUF value carries across iterations: each iteration loads the weights and the scale slots of the expert, and
    zeroes the gather tiles before the gather (a row that the gather skips must not keep an older value). The scale
    rows outside the slots are zeroed before the loops and never written in them. Order of the accumulator adds: the row table and the trip counts wait for every block-0
    add (runtime edges), and every loop DMA reads them, so no later-block add starts before the block-0 adds end.
    Each loop iteration ends with a fence that waits for its DMAs, so the adds of one loop end before the next loop
    and before the rounding loads.
    """
    C, D = latent_rows.shape
    E_L = local_affinities.shape[1]
    block_rows = tiles.rows
    for expert_idx in range(E_L):
        row = expert_idx * _RAGGED_EXPERT_STRIDE
        store = _instruction(
            nisa.dma_copy(
                dst=row_table.ap(pattern=[[C, 1], [1, C]], offset=expert_idx * C),
                src=found[row : row + 1, 0:C],
                dge_mode=nisa.dge_mode.none,
            )
        )
        for add in block0_adds:
            store.depends_on(add, kind=nisa.dependency.runtime)
    # The expert ids, as the runtime offsets of the loads: a loop body takes no static-queue load of an input.
    expert_ids = nl.ndarray((1, E_L), dtype=nl.int32, buffer=nl.sbuf)
    for expert_idx in range(E_L):
        nisa.memset(dst=expert_ids[0:1, expert_idx : expert_idx + 1], value=expert_idx)
    offsets = nl.ndarray((block_rows, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=offsets, value=-1)
    latent_sb = nl.ndarray((block_rows, D), dtype=latent_rows.dtype, buffer=nl.sbuf)
    nisa.memset(dst=latent_sb, value=0)
    affinity_sb = nl.ndarray((block_rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=affinity_sb, value=0.0)
    # The counts on partition 0, for the register loads.
    counts = nl.ndarray((1, E_L), dtype=nl.int32, buffer=nl.sbuf)
    for expert_idx in range(E_L):
        row = expert_idx * _RAGGED_EXPERT_STRIDE
        copy = _instruction(
            nisa.dma_copy(dst=counts[0:1, expert_idx : expert_idx + 1], src=found[row : row + 1, C : C + 1])
        )
        for add in block0_adds:
            copy.depends_on(add, kind=nisa.dependency.runtime)
    for expert_idx in range(E_L):
        count = nisa.register_alloc()
        nisa.register_load(dst=count, src=counts[0:1, expert_idx : expert_idx + 1])

        def block(first, expert_idx=expert_idx):
            loaded = _load_expert(stream, expert_ids[0:1, expert_idx : expert_idx + 1], *experts)
            nisa.dma_copy(
                dst=offsets,
                src=row_table.ap(
                    pattern=[[1, block_rows], [1, 1]], offset=expert_idx * C, scalar_offset=first, indirect_dim=0
                ),
            )
            nisa.memset(dst=latent_sb, value=0)
            nisa.memset(dst=affinity_sb, value=0.0)
            _gather_block(latent_rows, local_affinities, offsets, expert_idx, latent_sb, affinity_sb)
            input_quant_sb, input_scale_sb = _quantize_natural_latent(latent_sb=_as_bf16(latent_sb), T=block_rows, D=D)
            _expert_ffn(
                stream,
                expert_idx,
                input_quant_sb,
                input_scale_sb,
                *experts,
                beta=beta,
                linear_beta=linear_beta,
                T=block_rows,
                consume=_block_add(accumulator, offsets, affinity_sb, tiles, f"{name}_acc_dyn_e{expert_idx}"),
                loaded=loaded,
                tiles=tiles,
            )

        nl.fori_loop(0, count, block, step=block_rows)


def _as_bf16(rows_sb: nl.NkiTensor) -> nl.NkiTensor:
    """Return BF16 rows (or their UINT16 bits) as BF16."""
    return rows_sb if rows_sb.dtype == nl.bfloat16 else rows_sb.view(nl.bfloat16)


def _gather_block(
    latent_rows: nl.NkiTensor,
    local_affinities: nl.NkiTensor,
    offsets: nl.NkiTensor,
    expert_idx: int,
    latent_sb: nl.NkiTensor,
    affinity_sb: nl.NkiTensor,
):
    """Gather the latent rows and the expert's affinities of the rows ``offsets`` (-1 skips a row)."""
    D = latent_rows.shape[1]
    E_L = local_affinities.shape[1]
    rows = offsets.shape[0]
    nisa.dma_copy(
        dst=latent_sb,
        src=latent_rows.ap(pattern=[[D, rows], [1, D]], offset=0, vector_offset=offsets, indirect_dim=0),
        oob_mode=nisa.oob_mode.skip,
        dge_mode=nisa.dge_mode.swdge,
    )
    nisa.dma_copy(
        dst=affinity_sb,
        src=local_affinities.ap(
            pattern=[[E_L, rows], [1, 1]], offset=expert_idx, vector_offset=offsets, indirect_dim=0
        ),
        oob_mode=nisa.oob_mode.skip,
        dge_mode=nisa.dge_mode.swdge,
    )


def _block_add(
    accumulator: nl.NkiTensor,
    offsets: nl.NkiTensor,
    affinity_sb: nl.NkiTensor,
    tiles: "_AlternatingTiles",
    name: str,
    chain: tuple = None,
):
    """Return the ``consume`` of ``_expert_ffn`` that adds ``affinity * output`` to the accumulator rows ``offsets``.

    The add of columns ``[start, start + size)`` is named ``<name>_c<start>``. With ``chain = (last_write,
    first_writes)``, each add waits for the previous write of its columns.
    """
    D = accumulator.shape[1]
    rows = offsets.shape[0]

    def consume(start, size, H_slice, expert_out_f32):
        scaled = tiles.tile("scaled", rows, size)
        nisa.tensor_scalar(dst=scaled, data=expert_out_f32, op0=nl.multiply, operand0=affinity_sb)
        dst = accumulator.ap(pattern=[[D, rows], [1, size]], offset=start, vector_offset=offsets, indirect_dim=0)
        add = _instruction(
            nisa.dma_compute(
                dst=dst,
                srcs=[dst, scaled],
                scales=[1.0, 1.0],
                reduce_op=nl.add,
                oob_mode=nisa.oob_mode.skip,
                name=f"{name}_c{start}",
            )
        )
        if chain is None:
            return
        last_write, first_writes = chain
        for previous in last_write.get(start, first_writes):
            add.depends_on(previous, kind=nisa.dependency.runtime)
        last_write[start] = [add]

    return consume


class _AlternatingTiles:
    """Two FP32 SBUF buffers for each tile role of the expert loop, used in turn.

    A new tile for each chunk can land on the bytes of the previous chunk's tile, and its first write then waits for
    every read of that tile (a write-after-read stall). With two buffers, a write waits only for the reads of the use
    before the last. ``aliases`` maps a role to the role whose buffers it takes in turn with it.
    """

    def __init__(self, rows: int, columns: dict, aliases: dict = None, depths: dict = None):
        self.rows = rows
        self.buffers = {}
        self.uses = {}
        self.aliases = aliases or {}
        for role, cols in columns.items():
            depth = (depths or {}).get(role, 2)
            self.buffers[role] = [nl.ndarray((rows, cols), dtype=nl.float32, buffer=nl.sbuf) for _ in range(depth)]
            self.uses[role] = 0

    def tile(self, role: str, rows: int, cols: int) -> nl.NkiTensor:
        """Return the FP32 ``[rows, cols]`` view of the next buffer of ``role``."""
        role = self.aliases.get(role, role)
        buffer = self.buffers[role][self.uses[role] % len(self.buffers[role])]
        self.uses[role] += 1
        kernel_assert(rows <= buffer.shape[0] and cols <= buffer.shape[1], f"{role}: [{rows}, {cols}] does not fit")
        return buffer[0:rows, 0:cols]


class _StaticLoopTiles(_AlternatingTiles):
    """The alternating tiles of the static expert loop (``static_alt_tiles`` of ``moe_tkg_decode``).

    Under SBUF pressure the compiler can stack each chunk's new tile on the bytes of the previous chunk's tile. Each
    ScalarE write then waits for VectorE to read the tile of the chunk before it, so the two engines do not overlap
    across chunks. Here each tile takes the next of two buffers, so a write waits only for the reads of the use before
    the last:

    - the gate sigmoid and the up tanh share two FP32 ``[T, 512]`` buffers. Each expert's gate sigmoids are all read
      (by the gate-term products) before its first up tanh, so the two roles never hold a buffer at the same time;
    - the FP32 ``[T, 512]`` PSUM copy of each down H tile, before the affinity product;
    - the FP16 ``[128, 2, T, 4]`` ScalarE eviction of each transposed activation chunk, before its quantization.

    The buffers live across the whole loop: 2 x 2048 + 2 x 2048 + 2 x 16 T bytes per partition, 12,288 at T = 128.
    The gate terms stay compiler-placed (all the chunks of an expert are live at once until its up products).
    ``roles`` (``_STATIC_ALT_TILE_ROLES``) names the rings to build: ``gate_up`` (the sigmoid / tanh ring),
    ``expert_out`` (the PSUM copies) and ``eviction``. A tile of a role without a ring is a new compiler-placed tile.
    ``gate_up_depth`` (default 2) is the number of buffers of the sigmoid / tanh ring.
    """

    def __init__(self, rows: int, D: int, roles: tuple = ("gate_up", "expert_out", "eviction"), gate_up_depth: int = 2):
        columns = {"gate_up": _GATE_UP_CHUNK_I, "expert_out": min(D, _DOWN_TILE_H)}
        super().__init__(
            rows,
            {role: columns[role] for role in roles if role in columns},
            aliases={"gate_sigmoid": "gate_up", "up_tanh": "gate_up"},
            depths={"gate_up": gate_up_depth},
        )
        tiles_per_chunk = _TRANSPOSE_CHUNK_I // MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
        self.evictions = None
        if "eviction" in roles:
            self.evictions = [
                nl.ndarray((_PMAX, tiles_per_chunk, rows, _q_width), dtype=nl.float16, buffer=nl.sbuf) for _ in range(2)
            ]
        self.eviction_uses = 0

    def tile(self, role: str, rows: int, cols: int) -> nl.NkiTensor:
        """Return the FP32 ``[rows, cols]`` view of the next buffer of ``role``, or a new tile if it has no ring."""
        if self.aliases.get(role, role) not in self.buffers:
            return nl.ndarray((rows, cols), dtype=nl.float32, buffer=nl.sbuf)
        return super().tile(role, rows, cols)

    def eviction(self, n_tiles: int) -> nl.NkiTensor:
        """Return the FP16 ``[128, n_tiles, T, 4]`` view of the next eviction buffer."""
        buffer = self.evictions[self.eviction_uses % 2]
        self.eviction_uses += 1
        kernel_assert(n_tiles <= buffer.shape[1], f"eviction: {n_tiles} tiles do not fit")
        return buffer if n_tiles == buffer.shape[1] else buffer[:, 0:n_tiles, :, :]


def _tile(tiles: "_AlternatingTiles", role: str, rows: int, cols: int) -> nl.NkiTensor:
    """Return a new FP32 ``[rows, cols]`` SBUF tile, or the next buffer of ``role`` in ``tiles``."""
    if tiles is None:
        return nl.ndarray((rows, cols), dtype=nl.float32, buffer=nl.sbuf)
    return tiles.tile(role, rows, cols)


def _load_expert(
    stream: "_StaticExpertStream",
    expert_id: nl.NkiTensor,
    expert_gate_up_weights: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_gate_up_weights_scale: nl.NkiTensor,
    expert_down_weights_scale: nl.NkiTensor,
) -> dict:
    """Load the weights and scales of expert ``expert_id`` (INT32 [1, 1] SBUF) into the stream buffers.

    The gate, up and down weights go to weight buffers 0, 1 and 2 (with two buffers, down goes to buffer 0, and
    ``loaded["down"]`` is a function that loads it), the gate/up scales to slots 0 and 1 of the first gate/up scale
    buffer, and the down scales to slot 0. The loads take the id as a runtime offset (DGE queues), so a
    dynamic loop body can issue them. Returns the ``loaded`` views of ``_expert_ffn``.
    """
    I = expert_gate_up_weights.shape[-1]
    D = expert_down_weights.shape[-1]
    n_D512 = D // MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
    n_I512 = I // MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
    row = n_D512 * I
    gate_up_scale = stream.gate_up_scale_buffers[0]
    for slot in (GATE_FUSED_IDX, UP_FUSED_IDX):
        _load_scale_rows_of(gate_up_scale, expert_gate_up_weights_scale, expert_id, slot=slot, projections_per_row=2)
    loaded = {"gate_up_scale": gate_up_scale.reshape((_PMAX, n_D512, I))}
    for projection_idx, key in ((GATE_FUSED_IDX, "gate"), (UP_FUSED_IDX, "up")):
        buffer = stream.buffers[projection_idx]
        _load_rows_of(buffer, expert_gate_up_weights, expert_id, row_stride=2 * row, offset=projection_idx * row)
        loaded[key] = [(0, buffer.reshape((_PMAX, n_D512, I)))]
    _load_scale_rows_of(stream.down_scale_buffer, expert_down_weights_scale, expert_id, slot=0, projections_per_row=1)
    loaded["down_scale"] = stream.down_scale_buffer.reshape((_PMAX, n_I512, D))
    down = stream.buffers[_DOWN_IDX % len(stream.buffers)]

    def load_down():
        _load_rows_of(down, expert_down_weights, expert_id, row_stride=row, offset=0)
        return [(0, down.reshape((_PMAX, n_I512, D)))]

    # With two buffers the down weights reuse the gate buffer, so they load after the gate matmuls in program order.
    loaded["down"] = load_down() if len(stream.buffers) > _DOWN_IDX else load_down
    return loaded


def _load_rows_of(dst: nl.NkiTensor, src: nl.NkiTensor, expert_id: nl.NkiTensor, row_stride: int, offset: int):
    """Load the ``dst.shape[1]`` elements of each partition row of expert ``expert_id``."""
    nisa.dma_copy(
        dst=dst,
        src=src.ap(
            pattern=[[row_stride, dst.shape[0]], [1, dst.shape[1]]],
            offset=offset,
            scalar_offset=expert_id,
            indirect_dim=0,
        ),
        dge_mode=nisa.dge_mode.hwdge,
        engine=nisa.engine.sync,
    )


def _load_scale_rows_of(
    dst: nl.NkiTensor, src: nl.NkiTensor, expert_id: nl.NkiTensor, slot: int, projections_per_row: int
):
    """Load the 16 MX scale rows of a projection of expert ``expert_id`` into partitions 32 q + 4 slot + r, r < 4.

    The layout of ``_load_scale_rows``, one DMA for each r (four partitions, one in each quadrant).
    """
    row_size = dst.shape[1]
    n_quadrants = _PMAX // _SBUF_QUADRANT_PARTITIONS
    first_partition = _SCALE_ROWS_PER_QUADRANT * slot
    for row_idx in range(_SCALE_ROWS_PER_QUADRANT):
        nisa.dma_copy(
            dst=dst[first_partition + row_idx : _PMAX : _SBUF_QUADRANT_PARTITIONS, :],
            src=src.ap(
                pattern=[[_SCALE_ROWS_PER_QUADRANT * projections_per_row * row_size, n_quadrants], [1, row_size]],
                offset=(projections_per_row * row_idx + slot) * row_size,
                scalar_offset=expert_id,
                indirect_dim=0,
            ),
            dge_mode=nisa.dge_mode.hwdge,
            engine=nisa.engine.sync,
        )


def _instruction(handle):
    """Return an ISA call result as an NkiInstruction, for ``depends_on``."""
    return handle if isinstance(handle, nisa.NkiInstruction) else nisa.NkiInstruction(handle)


def _ragged_row_lists(local_affinities: nl.NkiTensor, expert_rows: int, expert_counts) -> nl.NkiTensor:
    """Return the INT32 [expert_rows, 128] row lists: column 16 e holds the rows of expert e (-1 past its count)."""
    return _first_rows(_ragged_found(local_affinities, expert_counts), expert_rows)


def _ragged_found(local_affinities: nl.NkiTensor, expert_counts) -> nl.NkiTensor:
    """Return the INT32 [128, C + 1] row lists: partition 16 e holds the rows of expert e (-1 past its count), and
    column C its count."""
    return _found_of(_ragged_flags(local_affinities), expert_counts)


def _ragged_flags(local_affinities: nl.NkiTensor) -> nl.NkiTensor:
    """Return the FP32 [128, C] expert row flags: partition 16 e holds the affinity of each row for expert e, else 0."""
    C, E_L = local_affinities.shape
    flags = nl.ndarray((_PMAX, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=flags, value=0.0)
    for first in range(0, C, _PMAX):
        size = min(_PMAX, C - first)
        spread = _spread_affinities(local_affinities, first, size)
        flags_psum = nl.ndarray((_PMAX, _PMAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=flags_psum[:, 0:size], data=spread)
        nisa.tensor_copy(dst=flags[:, first : first + size], src=flags_psum[:, 0:size])
    return flags


def _spread_affinities(local_affinities: nl.NkiTensor, first: int, size: int) -> nl.NkiTensor:
    """Return the FP32 [size, 128] affinities of rows ``[first, first + size)``: column 16 e holds expert e."""
    E_L = local_affinities.shape[1]
    affinity_sb = nl.ndarray((size, E_L), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=affinity_sb, src=local_affinities[first : first + size, :], dge_mode=nisa.dge_mode.none)
    spread = nl.ndarray((size, _PMAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=spread, value=0.0)
    nisa.tensor_copy(dst=spread.ap(pattern=[[_PMAX, size], [_RAGGED_EXPERT_STRIDE, E_L]], offset=0), src=affinity_sb)
    return spread


def _found_of(flags: nl.NkiTensor, expert_counts=None) -> nl.NkiTensor:
    """Return the INT32 [128, C + 1] row lists of the nonzero ``flags`` of each partition, and column C the counts."""
    C = flags.shape[1]
    found = nl.ndarray((_PMAX, C + 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=found, value=-1)
    nisa.nonzero_with_count(dst=found, src=flags, index_offset=0, padding_val=-1)
    if expert_counts is not None:
        for expert_idx in range(expert_counts.shape[0]):
            row = expert_idx * _RAGGED_EXPERT_STRIDE
            nisa.dma_copy(dst=expert_counts[expert_idx : expert_idx + 1, :], src=found[row : row + 1, C : C + 1])
    return found


def _phase_flags(local_affinities: nl.NkiTensor, flags: nl.NkiTensor, block_rows: int) -> tuple:
    """Split the expert row flags into the rows that add in block 0 and the rows that add in the later blocks.

    A row adds in block 0 only if it is one of the first ``block_rows`` rows of each local expert that it uses; else
    all its local terms add in the later blocks. Each row then adds its terms in one phase, in expert order, so its
    FP32 sum does not depend on the other rows of the call.
    """
    C = flags.shape[1]
    hit = nl.ndarray((_PMAX, C), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=hit, data=flags, op0=nl.not_equal, operand0=0.0)
    hit_rows = []
    for first in range(0, C, _PMAX):
        size = min(_PMAX, C - first)
        rows_hit = nl.ndarray((size, _PMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        spread = _spread_affinities(local_affinities, first, size)
        nisa.tensor_scalar(dst=rows_hit, data=spread, op0=nl.not_equal, operand0=0.0)
        hit_rows.append(rows_hit)
    # The position of each row in each row list (0/1 products, FP32 sums: exact), and the rows past block 0.
    late = nl.ndarray((_PMAX, C), dtype=nl.bfloat16, buffer=nl.sbuf)
    for first in range(0, C, _PSUM_BANK_FP32):
        cols = min(_PSUM_BANK_FP32, C - first)
        rank_psum = nl.ndarray((_PMAX, cols), dtype=nl.float32, buffer=nl.psum)
        k_tiles = list(range(0, min(C, first + cols), _PMAX))
        for k_first in k_tiles:
            size = min(_PMAX, C - k_first)
            index = nl.ndarray((size, cols), dtype=nl.float32, buffer=nl.sbuf)
            nisa.iota(index, pattern=[[1, cols]], offset=first - k_first, channel_multiplier=-1)
            upper = nl.ndarray((size, cols), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=upper, data=index, op0=nl.greater_equal, operand0=0.0)
            nisa.nc_matmul(
                dst=rank_psum,
                stationary=hit_rows[k_first // _PMAX],
                moving=upper,
                accumulate=k_first > 0,
            )
        past = nl.ndarray((_PMAX, cols), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=past, data=rank_psum, op0=nl.greater, operand0=float(block_rows))
        nisa.tensor_tensor(
            dst=late[:, first : first + cols], data1=past, data2=hit[:, first : first + cols], op=nl.multiply
        )
    # A row is late if it is past block 0 of any expert: the sum over the partitions, on every partition.
    ones = nl.ndarray((_PMAX, _PMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    early_flags = nl.ndarray((_PMAX, C), dtype=nl.float32, buffer=nl.sbuf)
    late_flags = nl.ndarray((_PMAX, C), dtype=nl.float32, buffer=nl.sbuf)
    for first in range(0, C, _PSUM_BANK_FP32):
        cols = min(_PSUM_BANK_FP32, C - first)
        any_psum = nl.ndarray((_PMAX, cols), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=any_psum, stationary=ones, moving=late[:, first : first + cols])
        is_late = nl.ndarray((_PMAX, cols), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=is_late, data=any_psum, op0=nl.greater, operand0=0.0)
        part = slice(first, first + cols)
        nisa.tensor_tensor(dst=late_flags[:, part], data1=flags[:, part], data2=is_late, op=nl.multiply)
        nisa.tensor_tensor(dst=early_flags[:, part], data1=flags[:, part], data2=late_flags[:, part], op=nl.subtract)
    return early_flags, late_flags


def _first_rows(found: nl.NkiTensor, expert_rows: int) -> nl.NkiTensor:
    """Return the first ``expert_rows`` entries of each row list of ``found``, transposed to [expert_rows, 128]."""
    first_rows = nl.ndarray((_PMAX, expert_rows), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=first_rows, src=found[:, 0:expert_rows])
    lists_psum = nl.ndarray((expert_rows, _PMAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=lists_psum, data=first_rows)
    lists = nl.ndarray((expert_rows, _PMAX), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=lists, src=lists_psum)
    return lists


def _zero_rows_f32(rows_hbm: nl.NkiTensor, name: str = None) -> list:
    """Zero an FP32 [C, D] HBM tensor from one zero SBUF tile (D a multiple of 512); return the DMA instructions.

    ``name``: the DMAs are named ``<name>_<first row>``."""
    C, D = rows_hbm.shape
    zero_sb = nl.ndarray((_PMAX, _PSUM_BANK_FP32), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zero_sb, value=0.0)
    blocks = D // _PSUM_BANK_FP32
    writes = []
    for first in range(0, C, _PMAX):
        size = min(_PMAX, C - first)
        writes.append(
            _instruction(
                nisa.dma_copy(
                    dst=rows_hbm[first : first + size, :].reshape((size, blocks, _PSUM_BANK_FP32)),
                    src=zero_sb.ap(pattern=[[_PSUM_BANK_FP32, size], [0, blocks], [1, _PSUM_BANK_FP32]], offset=0),
                    dge_mode=nisa.dge_mode.none,
                    **({"name": f"{name}_{first}"} if name else {}),
                )
            )
        )
    return writes


def _round_rows_bf16(rows_f32: nl.NkiTensor, output: nl.NkiTensor, after: list = (), name: str = None):
    """Round an FP32 [C, D] HBM tensor to the BF16 ``output`` on VectorE, in tiles of 128 rows and 1024 columns.

    Each load waits for the instructions of ``after`` (the last writes of the tensor). ``name``: the loads are named
    ``<name>_<first row>_<first column>``.
    """
    C, D = rows_f32.shape
    for first in range(0, C, _PMAX):
        size = min(_PMAX, C - first)
        for col in range(0, D, _RAGGED_OUTPUT_TILE_H):
            cols = min(_RAGGED_OUTPUT_TILE_H, D - col)
            values = nl.ndarray((size, cols), dtype=nl.float32, buffer=nl.sbuf)
            load = _instruction(
                nisa.dma_copy(
                    dst=values,
                    src=rows_f32[first : first + size, col : col + cols],
                    dge_mode=nisa.dge_mode.none,
                    **({"name": f"{name}_{first}_{col}"} if name else {}),
                )
            )
            for write in after:
                load.depends_on(write, kind=nisa.dependency.runtime)
            rounded = nl.ndarray((size, cols), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=rounded, src=values)
            nisa.dma_copy(dst=output[first : first + size, col : col + cols], src=rounded, dge_mode=nisa.dge_mode.none)


class _OutputTiles:
    """The BF16 output of the static loop, in tiles that the last expert writes directly.

    The last expert's multiply-add rounds its FP32 result to BF16 on the way
    out, as the BF16 copy of the accumulator after the loop did. The chain
    after the last matmul is then the PSUM copy, the multiply-add and the
    output DMA, without a BF16 copy, and each output tile is stored as soon
    as its last columns are written. Without ``psum_consume`` the
    multiply-add reads the SBUF copy of the PSUM tile, as for the other
    experts: with the PSUM tile as its input and the affinity and the
    accumulator in separate SBUF tiles, neuronx-cc may place the two SBUF
    operands at different base partitions, which its verifier then rejects
    (NCC_IBIR297, with one local expert). With ``psum_consume`` the
    multiply-add reads the PSUM tile, and the affinities sit in the columns
    after the accumulator's, in one SBUF tile.

    With ``output`` None the output stays in SBUF: one tile of ``sbuf_columns``
    columns (``tiles[0]``), which nothing stores.
    """

    def __init__(self, output: nl.NkiTensor, T: int, dge_mode=nisa.dge_mode.none, sbuf_columns: int = 0):
        self.output = output
        self.T = T
        self.dge_mode = dge_mode
        self.columns_total = sbuf_columns if output is None else output.shape[1]
        self.tile_H = sbuf_columns if output is None else min(output.shape[1], _OUTPUT_TILE_H)
        self.tiles = {}
        self.written = {}

    def _tile(self, column: int) -> tuple:
        tile_idx = column // self.tile_H
        tile_start = tile_idx * self.tile_H
        if tile_idx not in self.tiles:
            tile_size = min(self.tile_H, self.columns_total - tile_start)
            self.tiles[tile_idx] = nl.ndarray((self.T, tile_size), dtype=nl.bfloat16, buffer=nl.sbuf)
            self.written[tile_idx] = 0
        return tile_idx, tile_start

    def columns(self, start: int, size: int) -> nl.NkiTensor:
        """Return the SBUF view of output columns ``[start, start + size)``, within one tile."""
        tile_idx, tile_start = self._tile(start)
        return self.tiles[tile_idx][:, nl.ds(start - tile_start, size)]

    def store_when_written(self, start: int, size: int):
        """Record columns ``[start, start + size)`` as written, and store their tile once it is complete."""
        tile_idx, tile_start = self._tile(start)
        self.written[tile_idx] += size
        tile_sb = self.tiles[tile_idx]
        if self.written[tile_idx] == tile_sb.shape[1] and self.output is not None:
            T_physical = self.output.shape[0]
            nisa.dma_copy(
                dst=self.output[:, nl.ds(tile_start, tile_sb.shape[1])],
                src=tile_sb[:T_physical, :],
                dge_mode=self.dge_mode,
                **({"engine": nisa.engine.sync} if self.dge_mode == nisa.dge_mode.hwdge else {}),
            )


def _active_experts_in_slots(
    latent: nl.NkiTensor,
    local_affinities: nl.NkiTensor,
    expert_gate_up_weights: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_gate_up_weights_scale: nl.NkiTensor,
    expert_down_weights_scale: nl.NkiTensor,
    beta: float,
    linear_beta: float,
    T: int,
    T_physical: int,
    weight_buffers: int,
    early_loads: bool,
    n_slots: int,
    active_expert_count: nl.NkiTensor = None,
) -> nl.NkiTensor:
    """Run the active experts in ``n_slots`` slots, with every expert as the fallback, and return the output.

    The one dispatch point: the slots run the first active experts, and if
    more experts are active than there are slots, a one-trip loop runs every
    expert in the slot form, which overwrites the output. With
    ``active_expert_count``, the count of active experts goes there too.
    """
    D = latent.shape[1]
    E_L = expert_gate_up_weights.shape[0]
    latent_sb = _load_natural_latent(latent=latent, T=T, T_physical=T_physical, D=D)
    # Each expert path stores the output itself, from its last expert.
    output = nl.ndarray((T_physical, D), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    expert_args = dict(
        local_affinities=local_affinities,
        expert_gate_up_weights=expert_gate_up_weights,
        expert_down_weights=expert_down_weights,
        expert_gate_up_weights_scale=expert_gate_up_weights_scale,
        expert_down_weights_scale=expert_down_weights_scale,
        beta=beta,
        linear_beta=linear_beta,
        T=T,
        T_physical=T_physical,
        weight_buffers=weight_buffers,
        output=output,
    )
    overflow_sb, input_quant_sb, input_scale_sb = _active_expert_slots(
        latent_sb=latent_sb,
        **expert_args,
        early_loads=early_loads,
        n_slots=n_slots,
        active_expert_count=active_expert_count,
    )
    if n_slots < E_L:
        overflow = nisa.register_alloc()
        nisa.register_load(overflow, overflow_sb)

        def _overflow_body(_iteration):
            _all_experts_in_slots(**expert_args, input_quant_sb=input_quant_sb, input_scale_sb=input_scale_sb)

        nl.fori_loop(0, overflow, _overflow_body)
    return output


def _active_expert_slots(
    latent_sb: nl.NkiTensor,
    local_affinities: nl.NkiTensor,
    expert_gate_up_weights: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_gate_up_weights_scale: nl.NkiTensor,
    expert_down_weights_scale: nl.NkiTensor,
    beta: float,
    linear_beta: float,
    T: int,
    T_physical: int,
    weight_buffers: int,
    output: nl.NkiTensor,
    early_loads: bool,
    n_slots: int,
    active_expert_count: nl.NkiTensor = None,
) -> tuple:
    """Accumulate the first ``n_slots`` active experts in ascending order, and store the BF16 ``output``.

    An expert is active if any token has a nonzero affinity for it. Slot k
    runs the k-th active expert, with the expert id as a runtime offset of
    its weight and scale DMAs. A slot past the active count skips its DMAs
    (an out-of-bounds id) and runs on the weights and scales of the slot
    before it, with zero affinities: for the finite expert output x, each
    multiply-add then adds 0 * x = +-0 to an accumulator that is never -0,
    which leaves it unchanged. If no expert is active, slot 0 loads the
    last expert, which then has zero affinities too. So the output has the
    bits of the static loop whenever at most ``n_slots`` experts are active.

    The slots' buffers are allocated before the latent's work tiles with
    ``early_loads``, and after them without it, as in the static loop. Their
    first loads need the expert ids, so they start after the routing either
    way. ``active_expert_count`` (INT32 ``[1, 1]`` HBM) gets the count of
    active experts.

    Returns:
        The INT32 ``[1, 1]`` SBUF flag, 1 if more than ``n_slots`` experts
        are active, and the quantized latent and its scales.
    """
    D = latent_sb.shape[1]
    slot_ids_sb, overflow_sb, count_sb = _slot_expert_ids(
        local_affinities=local_affinities, T_physical=T_physical, n_slots=n_slots
    )
    if active_expert_count is not None:
        nisa.dma_copy(dst=active_expert_count, src=count_sb)
    stream_args = dict(
        expert_gate_up_weights=expert_gate_up_weights,
        expert_down_weights=expert_down_weights,
        expert_gate_up_weights_scale=expert_gate_up_weights_scale,
        expert_down_weights_scale=expert_down_weights_scale,
        weight_buffers=weight_buffers,
        slot_ids_sb=slot_ids_sb,
    )
    stream = _SlotExpertStream(**stream_args) if early_loads else None
    affinities = _slot_affinities(
        local_affinities=local_affinities, slot_ids_sb=slot_ids_sb, T=T, T_physical=T_physical
    )
    input_quant_sb, input_scale_sb = _quantize_natural_latent(latent_sb=latent_sb, T=T, D=D)
    if stream is None:
        stream = _SlotExpertStream(**stream_args)
    _run_slots(
        stream=stream,
        affinities=affinities,
        input_quant_sb=input_quant_sb,
        input_scale_sb=input_scale_sb,
        beta=beta,
        linear_beta=linear_beta,
        T=T,
        output=output,
    )
    return overflow_sb, input_quant_sb, input_scale_sb


def _all_experts_in_slots(
    input_quant_sb: nl.NkiTensor,
    input_scale_sb: nl.NkiTensor,
    local_affinities: nl.NkiTensor,
    expert_gate_up_weights: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_gate_up_weights_scale: nl.NkiTensor,
    expert_down_weights_scale: nl.NkiTensor,
    beta: float,
    linear_beta: float,
    T: int,
    T_physical: int,
    weight_buffers: int,
    output: nl.NkiTensor,
):
    """Accumulate every local expert in the slot form, and store the BF16 ``output``.

    This is the exact fallback of the slots, inside a loop body, so it uses
    DGE DMAs only: a loop body has no static DMA queue (NCC_ISQI001). Every
    tile is new and fully written in the body: a tile that a loop body reads
    before it writes all of it carries a dependency across iterations, which
    neuronx-cc rejects for a tile defined in the body (NCC_IGCA108).
    """
    E_L = expert_gate_up_weights.shape[0]
    stream = _SlotExpertStream(
        expert_gate_up_weights=expert_gate_up_weights,
        expert_down_weights=expert_down_weights,
        expert_gate_up_weights_scale=expert_gate_up_weights_scale,
        expert_down_weights_scale=expert_down_weights_scale,
        weight_buffers=weight_buffers,
    )
    affinity_raw_sb = nl.ndarray((T, E_L), dtype=local_affinities.dtype, buffer=nl.sbuf)
    if T > T_physical:
        nisa.memset(dst=affinity_raw_sb, value=0.0)
    if local_affinities.buffer == nl.sbuf:
        nisa.tensor_copy(dst=affinity_raw_sb[:T_physical, :], src=local_affinities)
    else:
        nisa.dma_copy(
            dst=affinity_raw_sb[:T_physical, :],
            src=local_affinities,
            dge_mode=nisa.dge_mode.hwdge,
            engine=nisa.engine.sync,
        )
    affinity_f32_sb = nl.ndarray((T, E_L), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=affinity_f32_sb, src=affinity_raw_sb)
    _run_slots(
        stream=stream,
        affinities=[affinity_f32_sb[:, expert_idx : expert_idx + 1] for expert_idx in range(E_L)],
        input_quant_sb=input_quant_sb,
        input_scale_sb=input_scale_sb,
        beta=beta,
        linear_beta=linear_beta,
        T=T,
        output=output,
    )


def _run_slots(
    stream: "_SlotExpertStream",
    affinities: list,
    input_quant_sb: nl.NkiTensor,
    input_scale_sb: nl.NkiTensor,
    beta: float,
    linear_beta: float,
    T: int,
    output: nl.NkiTensor,
):
    """Accumulate the experts of the slots in slot order, each scaled by its ``[T, 1]`` FP32 affinity.

    Each slot runs the instructions of an expert of the static loop (see
    _all_experts_static), so a slot gives the bits of the static loop's
    expert. The last slot writes and stores the BF16 ``output`` tiles.
    """
    D = stream.expert_down_weights.shape[-1]
    I = stream.expert_gate_up_weights.shape[-1]
    n_D512 = D // MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
    n_I512 = I // MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
    n_slots = len(affinities)
    output_accum_sb = nl.ndarray((T, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=output_accum_sb, value=0.0)
    # In the loop body of the fallback, the stores also take the DGE.
    store_mode = nisa.dge_mode.hwdge if stream.slot_ids_sb is None else nisa.dge_mode.none
    output_tiles = _OutputTiles(output=output, T=T, dge_mode=store_mode)
    # The first slot's loads go first on their queues, for the projections that have a buffer of their own. With
    # fewer buffers, projection p shares the buffer of projection p - weight_buffers, so it loads at its use, after
    # the matmuls of that projection, as in the later slots.
    weight_buffers = len(stream.buffers)
    stream.load_scales(slot_idx=0)
    for projection_idx in range(min(_PROJECTIONS_PER_EXPERT, weight_buffers)):
        stream.load(projection_idx, slot_idx=0)

    def loads_at_use(projection_idx: int, slot_idx: int) -> bool:
        return slot_idx > 0 or projection_idx >= weight_buffers

    for slot_idx in range(n_slots):
        if loads_at_use(GATE_FUSED_IDX, slot_idx):
            stream.load(GATE_FUSED_IDX, slot_idx)
        gate_weight_sb, gate_weight_scale_sb = stream.views(GATE_FUSED_IDX, slot_idx)
        n_chunks = div_ceil(I, _GATE_UP_CHUNK_I)
        gate_terms_f32 = []
        for chunk_idx in range(n_chunks):
            chunk_size = min(_GATE_UP_CHUNK_I, I - chunk_idx * _GATE_UP_CHUNK_I)
            gate_psum = _project_chunk(
                input_quant_sb=input_quant_sb,
                input_scale_sb=input_scale_sb,
                weight_sb=gate_weight_sb,
                weight_scale_sb=gate_weight_scale_sb,
                n_H512=n_D512,
                chunk_idx=chunk_idx,
                chunk_size=chunk_size,
                T=T,
            )
            gate_sigmoid_f32 = nl.ndarray((T, chunk_size), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=gate_sigmoid_f32, op=nl.sigmoid, data=gate_psum)
            gate_term_f32 = nl.ndarray((T, chunk_size), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=gate_term_f32, op=nl.tanh, data=gate_psum, scale=1.0 / beta)
            nisa.scalar_tensor_tensor(
                dst=gate_term_f32,
                data=gate_term_f32,
                op0=nl.multiply,
                operand0=beta,
                op1=nl.multiply,
                operand1=gate_sigmoid_f32,
            )
            gate_terms_f32.append(gate_term_f32)

        if loads_at_use(UP_FUSED_IDX, slot_idx):
            stream.load(UP_FUSED_IDX, slot_idx)
        up_weight_sb, up_weight_scale_sb = stream.views(UP_FUSED_IDX, slot_idx)
        activation_token_major = nl.ndarray((T, I), dtype=nl.float16, buffer=nl.sbuf)
        for chunk_idx in range(n_chunks):
            chunk_size = min(_GATE_UP_CHUNK_I, I - chunk_idx * _GATE_UP_CHUNK_I)
            up_psum = _project_chunk(
                input_quant_sb=input_quant_sb,
                input_scale_sb=input_scale_sb,
                weight_sb=up_weight_sb,
                weight_scale_sb=up_weight_scale_sb,
                n_H512=n_D512,
                chunk_idx=chunk_idx,
                chunk_size=chunk_size,
                T=T,
            )
            up_tanh_f32 = nl.ndarray((T, chunk_size), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=up_tanh_f32, op=nl.tanh, data=up_psum, scale=1.0 / linear_beta)
            nisa.scalar_tensor_tensor(
                dst=activation_token_major[:, nl.ds(chunk_idx * _GATE_UP_CHUNK_I, chunk_size)],
                data=up_tanh_f32,
                op0=nl.multiply,
                operand0=linear_beta,
                op1=nl.multiply,
                operand1=gate_terms_f32[chunk_idx],
            )

        if loads_at_use(_DOWN_IDX, slot_idx):
            stream.load(_DOWN_IDX, slot_idx)
        if slot_idx + 1 < n_slots:
            stream.load_scales(slot_idx + 1)
        down_weight_sb, down_weight_scale_sb = stream.views(_DOWN_IDX, slot_idx)
        tile_H = min(D, _DOWN_TILE_H)
        n_H_tiles = div_ceil(D, tile_H)
        tiles_per_chunk = _TRANSPOSE_CHUNK_I // MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
        n_chunks = div_ceil(n_I512, tiles_per_chunk)
        quantized_chunks = []
        for chunk_idx in range(min(_TRANSPOSE_LOOKAHEAD_CHUNKS, n_chunks)):
            quantized_chunks.append(
                _transpose_and_quantize_chunk(
                    activation_token_major=activation_token_major, chunk_idx=chunk_idx, I=I, T=T
                )
            )
        last_slot = slot_idx == n_slots - 1
        for group_first_tile in range(0, n_H_tiles, _DOWN_TILES_PER_GROUP):
            group_tiles = range(group_first_tile, min(group_first_tile + _DOWN_TILES_PER_GROUP, n_H_tiles))
            expert_out_psums = {}
            for tile_h in group_tiles:
                tile_H_actual = min(tile_H, D - tile_h * tile_H)
                expert_out_psums[tile_h] = nl.ndarray((T, tile_H_actual), dtype=nl.float32, buffer=nl.psum)
            for chunk_idx in range(n_chunks):
                next_chunk_idx = chunk_idx + _TRANSPOSE_LOOKAHEAD_CHUNKS
                activation_quant_sb, activation_scale_sb = quantized_chunks[chunk_idx]
                for chunk_tile in range(activation_quant_sb.shape[1]):
                    tile_i = chunk_idx * tiles_per_chunk + chunk_tile
                    for tile_h in group_tiles:
                        H_slice = nl.ds(tile_h * tile_H, expert_out_psums[tile_h].shape[1])
                        nisa.nc_matmul_mx(
                            dst=expert_out_psums[tile_h],
                            stationary=activation_quant_sb[:, chunk_tile, :],
                            moving=down_weight_sb[:, tile_i, H_slice],
                            stationary_scale=activation_scale_sb[:, chunk_tile, :],
                            moving_scale=down_weight_scale_sb[:, tile_i, H_slice],
                        )
                if next_chunk_idx < n_chunks and len(quantized_chunks) <= next_chunk_idx:
                    quantized_chunks.append(
                        _transpose_and_quantize_chunk(
                            activation_token_major=activation_token_major, chunk_idx=next_chunk_idx, I=I, T=T
                        )
                    )

            for tile_h in group_tiles:
                H_slice = nl.ds(tile_h * tile_H, expert_out_psums[tile_h].shape[1])
                tile_H_actual = expert_out_psums[tile_h].shape[1]
                expert_out_f32 = nl.ndarray((T, tile_H_actual), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=expert_out_f32, src=expert_out_psums[tile_h])
                sum_dst_sb = output_accum_sb[:, H_slice]
                if last_slot:
                    sum_dst_sb = output_tiles.columns(tile_h * tile_H, tile_H_actual)
                nisa.scalar_tensor_tensor(
                    dst=sum_dst_sb,
                    data=expert_out_f32,
                    op0=nl.multiply,
                    operand0=affinities[slot_idx],
                    op1=nl.add,
                    operand1=output_accum_sb[:, H_slice],
                )
                if last_slot:
                    output_tiles.store_when_written(tile_h * tile_H, tile_H_actual)


def _slot_expert_ids(local_affinities: nl.NkiTensor, T_physical: int, n_slots: int) -> tuple:
    """Find the experts of the slots: the active experts in ascending order.

    Returns:
        INT32 ``[1, n_slots]`` expert ids: the first ``n_slots`` active
        experts, then E_L (out of bounds), except that slot 0 takes expert
        E_L - 1 when no expert is active. The INT32 ``[1, 1]`` flag that
        more than ``n_slots`` experts are active, and the INT32 ``[1, 1]``
        count of active experts (a view of the ``nonzero_with_count``
        result).
    """
    E_L = local_affinities.shape[1]
    if local_affinities.buffer == nl.sbuf:
        # An SBUF input holds token t on partition t, and a DMA cannot gather
        # its rows onto one partition. A matmul with a ones column counts each
        # expert's tokens instead.
        nonzero_sb = nl.ndarray((T_physical, E_L), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=nonzero_sb, data=local_affinities, op0=nl.not_equal, operand0=0.0)
        ones_sb = nl.ndarray((T_physical, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(dst=ones_sb, value=1.0)
        count_psum = nl.ndarray((1, E_L), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=count_psum, stationary=ones_sb, moving=nonzero_sb, is_stationary_onezero=True)
        active_sb = nl.ndarray((1, E_L), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=active_sb, src=count_psum)
    else:
        affinity_row_sb = nl.ndarray((1, T_physical, E_L), dtype=local_affinities.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=affinity_row_sb,
            src=local_affinities.ap(pattern=[[T_physical * E_L, 1], [E_L, T_physical], [1, E_L]], offset=0),
            dge_mode=nisa.dge_mode.none,
        )
        nonzero_sb = nl.ndarray((1, T_physical, E_L), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=nonzero_sb, data=affinity_row_sb, op0=nl.not_equal, operand0=0.0)
        active_sb = nl.ndarray((1, E_L), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(
            dst=active_sb,
            op=nl.maximum,
            data=nonzero_sb.ap(pattern=[[T_physical * E_L, 1], [1, E_L], [E_L, T_physical]], offset=0),
            axis=2,
        )
    # [ids of the active experts, E_L padding, count]
    ids_and_count_sb = nl.ndarray((1, E_L + 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.nonzero_with_count(dst=ids_and_count_sb, src=active_sb, index_offset=0, padding_val=E_L)
    slot_ids_sb = nl.ndarray((1, n_slots), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=slot_ids_sb, src=ids_and_count_sb[0:1, 0:n_slots])
    nisa.tensor_scalar(dst=slot_ids_sb[0:1, 0:1], data=slot_ids_sb[0:1, 0:1], op0=nl.minimum, operand0=E_L - 1)
    overflow_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=overflow_sb, data=ids_and_count_sb[0:1, E_L : E_L + 1], op0=nl.greater, operand0=n_slots)
    return slot_ids_sb, overflow_sb, ids_and_count_sb[0:1, E_L : E_L + 1]


def _slot_affinities(local_affinities: nl.NkiTensor, slot_ids_sb: nl.NkiTensor, T: int, T_physical: int) -> list:
    """Return the FP32 ``[T, 1]`` affinities of each slot's expert, zero past the active count.

    A table ``[T, E_L + 1]`` with a zero last column holds the affinities,
    so the out-of-bounds id E_L of an unused slot selects zeros.
    """
    E_L = local_affinities.shape[1]
    n_slots = slot_ids_sb.shape[1]
    table_sb = nl.ndarray((T, E_L + 1), dtype=local_affinities.dtype, buffer=nl.sbuf)
    nisa.memset(dst=table_sb, value=0.0)
    if local_affinities.buffer == nl.sbuf:
        nisa.tensor_copy(dst=table_sb[:T_physical, 0:E_L], src=local_affinities)
    else:
        nisa.dma_copy(dst=table_sb[:T_physical, 0:E_L], src=local_affinities, dge_mode=nisa.dge_mode.none)
    slot_affinities = []
    for slot_idx in range(n_slots):
        affinity_raw_sb = nl.ndarray((T, 1), dtype=local_affinities.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=affinity_raw_sb,
            src=table_sb.ap(
                pattern=[[E_L + 1, T], [1, 1]],
                offset=0,
                scalar_offset=slot_ids_sb[0:1, slot_idx : slot_idx + 1],
                indirect_dim=1,
            ),
            dge_mode=nisa.dge_mode.hwdge,
            engine=nisa.engine.sync,
        )
        affinity_f32_sb = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=affinity_f32_sb, src=affinity_raw_sb)
        slot_affinities.append(affinity_f32_sb)
    return slot_affinities


# Scale rows of the slot form: partitions 32 q + offset + r of each quadrant q
# hold rows 4 q + r of the gate, up and down scales of a slot.
_SLOT_SCALE_OFFSETS = (0, 4, 8)
# The queue of the slots' scale DMAs ("sync" or "scalar" hardware DGE, or
# "gpsimd" software DGE), their form ("quad": one DMA of 4 contiguous
# partitions per quadrant; "strided": one DMA per scale row of all 4
# quadrants), and their buffers (2 load each slot's scales one slot ahead).
_SLOT_SCALE_ENGINE = "sync"
_SLOT_SCALE_FORM = "quad"
_SLOT_SCALE_BUFFERS = 1
# The slots give the bits of the static loop for 1 to 16 tokens with 1, 2 or 3
# weight buffers (simulator); more tokens are not verified.
_SLOT_MAX_TOKENS = 16


class _SlotExpertStream:
    """The SBUF buffers of the slot form, and their loads.

    With ``slot_ids_sb``, slot k loads the expert ``slot_ids_sb[0, k]``, a
    runtime offset of hardware-DGE DMAs on the SP queue, and an out-of-bounds
    id skips the DMA. Without it, slot k is expert k, and the scale buffer is
    zeroed first, so that every byte the matmuls read is written in the same
    loop body.

    Each projection keeps its scale rows at a fixed offset of a scale buffer,
    so with one scale buffer its weights and scales are free at the same
    time: after its matmuls in the slot before.
    """

    def __init__(
        self,
        expert_gate_up_weights: nl.NkiTensor,
        expert_down_weights: nl.NkiTensor,
        expert_gate_up_weights_scale: nl.NkiTensor,
        expert_down_weights_scale: nl.NkiTensor,
        weight_buffers: int,
        slot_ids_sb: nl.NkiTensor = None,
    ):
        self.expert_gate_up_weights = expert_gate_up_weights
        self.expert_down_weights = expert_down_weights
        self.expert_gate_up_weights_scale = expert_gate_up_weights_scale
        self.expert_down_weights_scale = expert_down_weights_scale
        self.slot_ids_sb = slot_ids_sb
        D = expert_down_weights.shape[-1]
        I = expert_gate_up_weights.shape[-1]
        self.n_D512 = D // MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
        self.n_I512 = I // MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
        self.row_size = self.n_D512 * I
        self.buffers = []
        for _ in range(weight_buffers):
            self.buffers.append(nl.ndarray((_PMAX, self.row_size), dtype=expert_down_weights.dtype, buffer=nl.sbuf))
        static_ids = slot_ids_sb is None
        self.gather = not static_ids and _SLOT_SCALE_FORM == "gather"
        self.scales_ahead = not static_ids and (_SLOT_SCALE_BUFFERS > 1 or self.gather)
        self.scale_buffers = []
        for _ in range(2 if self.scales_ahead else 1):
            self.scale_buffers.append(nl.ndarray((_PMAX, self.row_size), dtype=nl.uint8, buffer=nl.sbuf))
        if static_ids:
            nisa.memset(dst=self.scale_buffers[0], value=0)
        elif self.scales_ahead:
            # An unused slot runs on the scales of its buffer: slot 1 may be
            # the first one to read the second buffer.
            nisa.memset(dst=self.scale_buffers[1], value=0, engine=nisa.engine.gpsimd)
        if self.gather:
            self.gather_rows = self._gather_rows()

    def _gather_rows(self) -> tuple:
        """Return the INT32 ``[128, n_slots]`` scale rows that each partition gathers, for gate/up and down.

        Partition 32 q + offset + r takes row 4 q + r of a projection of the
        slot's expert: a row of ``scale.reshape((E_L * 16 * k, row_size))``
        with k = 2 for gate/up (gate and up rows alternate) and 1 for down.
        Every other partition takes an out-of-bounds row, which the DMA skips.
        """
        E_L = self.expert_gate_up_weights.shape[0]
        n_slots = self.slot_ids_sb.shape[1]
        # Per partition: [row stride of the expert id, first row] for gate/up, then for down.
        table = np.zeros((_PMAX, 4), dtype=np.float32)
        table[:, 1] = E_L * 2 * _MX_SCALE_PARTITIONS
        table[:, 3] = E_L * _MX_SCALE_PARTITIONS
        for quadrant_idx in range(_PMAX // _SBUF_QUADRANT_PARTITIONS):
            for row_idx in range(_SCALE_ROWS_PER_QUADRANT):
                scale_row = quadrant_idx * _SCALE_ROWS_PER_QUADRANT + row_idx
                for projection_idx in range(_PROJECTIONS_PER_EXPERT):
                    partition = quadrant_idx * _SBUF_QUADRANT_PARTITIONS + _SLOT_SCALE_OFFSETS[projection_idx] + row_idx
                    if projection_idx == _DOWN_IDX:
                        table[partition, 2:4] = (_MX_SCALE_PARTITIONS, scale_row)
                    else:
                        table[partition, 0:2] = (2 * _MX_SCALE_PARTITIONS, 2 * scale_row + projection_idx)
        table_sb = nl.ndarray((_PMAX, 4), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=table_sb, src=nl.shared_constant(table), dge_mode=nisa.dge_mode.none)
        # The slot ids on every partition: a matmul with a ones row.
        ids_bf16 = nl.ndarray((1, n_slots), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ids_bf16, src=self.slot_ids_sb)
        ones_sb = nl.ndarray((1, _PMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(dst=ones_sb, value=1.0)
        ids_psum = nl.ndarray((_PMAX, n_slots), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=ids_psum, stationary=ones_sb, moving=ids_bf16)
        rows = []
        for column in (0, 2):
            rows_f32 = nl.ndarray((_PMAX, n_slots), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=rows_f32,
                data=ids_psum,
                op0=nl.multiply,
                operand0=table_sb[:, column : column + 1],
                op1=nl.add,
                operand1=table_sb[:, column + 1 : column + 2],
            )
            rows_i32 = nl.ndarray((_PMAX, n_slots), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=rows_i32, src=rows_f32)
            rows.append(rows_i32)
        return tuple(rows)

    def _buffer(self, projection_idx: int, slot_idx: int) -> nl.NkiTensor:
        # Projection p uses buffer p % weight_buffers, as in the static loop.
        return self.buffers[projection_idx % len(self.buffers)]

    def _scale_buffer(self, slot_idx: int) -> nl.NkiTensor:
        return self.scale_buffers[slot_idx % len(self.scale_buffers)]

    def _dma(self, dst: nl.NkiTensor, src: nl.NkiTensor, pattern: list, offset: int, slot_idx: int, engine: str):
        """Copy ``src[pattern]`` of the slot's expert to ``dst``."""
        if self.slot_ids_sb is None:
            # A loop body takes DGE DMAs only (no static DMA queue).
            expert_stride = 1
            for dim in src.shape[1:]:
                expert_stride *= dim
            nisa.dma_copy(
                dst=dst,
                src=src.ap(pattern=pattern, offset=offset + slot_idx * expert_stride),
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.sync,
            )
            return
        queue = {}
        if engine == "gpsimd":
            queue["dge_mode"] = nisa.dge_mode.swdge
        else:
            queue["dge_mode"] = nisa.dge_mode.hwdge
            queue["engine"] = nisa.engine.scalar if engine == "scalar" else nisa.engine.sync
        nisa.dma_copy(
            dst=dst,
            src=src.ap(
                pattern=pattern,
                offset=offset,
                scalar_offset=self.slot_ids_sb[0:1, slot_idx : slot_idx + 1],
                indirect_dim=0,
            ),
            oob_mode=nisa.oob_mode.skip,
            **queue,
        )

    def _projection(self, projection_idx: int) -> tuple:
        """Return (weights, partition stride, offset) and (scales, row stride, offset) of a projection."""
        row_size = self.row_size
        if projection_idx == _DOWN_IDX:
            return (self.expert_down_weights, row_size, 0), (self.expert_down_weights_scale, row_size, 0)
        offset = projection_idx * row_size
        return (
            (self.expert_gate_up_weights, 2 * row_size, offset),
            (self.expert_gate_up_weights_scale, 2 * row_size, offset),
        )

    def load(self, projection_idx: int, slot_idx: int):
        """Load the weights of one projection of the slot's expert, and its scale rows with one scale buffer."""
        (weight, weight_stride, weight_offset), _ = self._projection(projection_idx)
        self._dma(
            dst=self._buffer(projection_idx, slot_idx),
            src=weight,
            pattern=[[weight_stride, _PMAX], [1, self.row_size]],
            offset=weight_offset,
            slot_idx=slot_idx,
            engine="sync",
        )
        if not self.scales_ahead:
            self._load_scale_rows(projection_idx, slot_idx)

    def load_scales(self, slot_idx: int):
        """With two scale buffers, load the scale rows of every projection of the slot's expert."""
        if not self.scales_ahead:
            return
        if self.gather:
            # One software-DGE gather of 128 partitions for each scale tensor.
            for scale, rows in zip(
                (self.expert_gate_up_weights_scale, self.expert_down_weights_scale), self.gather_rows, strict=True
            ):
                n_rows = 1
                for dim in scale.shape:
                    n_rows *= dim
                n_rows //= self.row_size
                nisa.dma_copy(
                    dst=self._scale_buffer(slot_idx),
                    src=scale.reshape((n_rows, self.row_size)).ap(
                        pattern=[[self.row_size, _PMAX], [1, self.row_size]],
                        vector_offset=rows[:, slot_idx : slot_idx + 1],
                        indirect_dim=0,
                    ),
                    oob_mode=nisa.oob_mode.skip,
                    dge_mode=nisa.dge_mode.swdge,
                )
            return
        for projection_idx in range(_PROJECTIONS_PER_EXPERT):
            self._load_scale_rows(projection_idx, slot_idx)

    def _load_scale_rows(self, projection_idx: int, slot_idx: int):
        _, (scale, scale_stride, scale_offset) = self._projection(projection_idx)
        partition_offset = _SLOT_SCALE_OFFSETS[projection_idx]
        engine = "sync" if self.slot_ids_sb is None else _SLOT_SCALE_ENGINE
        n_quadrants = _PMAX // _SBUF_QUADRANT_PARTITIONS
        for row_idx in range(_SCALE_ROWS_PER_QUADRANT):
            if _SLOT_SCALE_FORM == "quad16" and self.slot_ids_sb is not None:
                # Quadrant row_idx, each partition row in 4 DMA elements: a
                # hardware-DGE DMA hands its elements to the 16 DMA engines in
                # turn, so 16 elements spread over all of them.
                n_pieces = _SCALE_ROWS_PER_QUADRANT
                piece = self.row_size // n_pieces
                buffer = self._scale_buffer(slot_idx)
                dst = buffer.ap(
                    pattern=[[self.row_size, _SCALE_ROWS_PER_QUADRANT], [piece, n_pieces], [1, piece]],
                    offset=(row_idx * _SBUF_QUADRANT_PARTITIONS + partition_offset) * self.row_size,
                )
                pattern = [[scale_stride, _SCALE_ROWS_PER_QUADRANT], [piece, n_pieces], [1, piece]]
                offset = row_idx * _SCALE_ROWS_PER_QUADRANT * scale_stride + scale_offset
            elif _SLOT_SCALE_FORM == "quad" or self.slot_ids_sb is None:
                # Quadrant row_idx: 4 contiguous partitions, rows 4 q .. 4 q + 3.
                dst = self._scale_buffer(slot_idx)[
                    nl.ds(row_idx * _SBUF_QUADRANT_PARTITIONS + partition_offset, _SCALE_ROWS_PER_QUADRANT), :
                ]
                pattern = [[scale_stride, _SCALE_ROWS_PER_QUADRANT], [1, self.row_size]]
                offset = row_idx * _SCALE_ROWS_PER_QUADRANT * scale_stride + scale_offset
            else:
                # Row row_idx of each quadrant: partitions 32 q + offset + row_idx.
                dst = self._scale_buffer(slot_idx)[partition_offset + row_idx : _PMAX : _SBUF_QUADRANT_PARTITIONS, :]
                pattern = [[_SCALE_ROWS_PER_QUADRANT * scale_stride, n_quadrants], [1, self.row_size]]
                offset = row_idx * scale_stride + scale_offset
            self._dma(dst=dst, src=scale, pattern=pattern, offset=offset, slot_idx=slot_idx, engine=engine)

    def views(self, projection_idx: int, slot_idx: int) -> tuple:
        """Return the SBUF weights ``[128, n_K512, N]`` and scales of one projection."""
        if projection_idx == _DOWN_IDX:
            shape = (_PMAX, self.n_I512, self.expert_down_weights.shape[-1])
        else:
            shape = (_PMAX, self.n_D512, self.expert_gate_up_weights.shape[-1])
        weight_sb = self._buffer(projection_idx, slot_idx).reshape(shape)
        scale_sb = self._scale_buffer(slot_idx).reshape(shape)[_SLOT_SCALE_OFFSETS[projection_idx] :, :, :]
        return weight_sb, scale_sb


def _active_experts_dynamic(
    input_quant_sb: nl.NkiTensor,
    input_scale_sb: nl.NkiTensor,
    output_accum_sb: nl.NkiTensor,
    local_affinities: nl.NkiTensor,
    expert_gate_up_weights: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_gate_up_weights_scale: nl.NkiTensor,
    expert_down_weights_scale: nl.NkiTensor,
    beta: float,
    linear_beta: float,
    T: int,
    T_physical: int,
    n_static_experts: int,
):
    """Accumulate the active experts only, one dynamic-loop iteration per expert.

    An expert is active if any token has a nonzero affinity for it. The
    sorted active ids are compacted on the device, the loop runs one
    iteration per active expert, and its DMAs take the expert id as a
    runtime offset. A zero-affinity expert would add exactly zero, so
    skipping it is exact.
    """
    E_L = expert_gate_up_weights.shape[0]
    # The active experts (any nonzero affinity) in ascending order, padded with
    # the out-of-bounds id E_L, whose loads are skipped.
    expert_ids_sb, active_count_sb = _active_expert_ids(
        local_affinities=local_affinities, T_physical=T_physical, E_L=E_L
    )
    expert = _ExpertPipeline(
        input_quant_sb=input_quant_sb,
        input_scale_sb=input_scale_sb,
        output_accum_sb=output_accum_sb,
        expert_ids_sb=expert_ids_sb,
        local_affinities=local_affinities,
        expert_gate_up_weights=expert_gate_up_weights,
        expert_down_weights=expert_down_weights,
        expert_gate_up_weights_scale=expert_gate_up_weights_scale,
        expert_down_weights_scale=expert_down_weights_scale,
        beta=beta,
        linear_beta=linear_beta,
        T=T,
        T_physical=T_physical,
    )
    expert.load_gate_up(id_slot=0)
    # The first n_static_experts slots run without loop barriers, so their
    # loads overlap; the loop runs the remaining active experts.
    if n_static_experts > 0:
        expert.load_affinity_table()
    for _ in range(n_static_experts):
        expert.run(None, static_slot=True)
    if n_static_experts > 0:
        loop_count_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=loop_count_sb,
            data=active_count_sb,
            op0=nl.subtract,
            operand0=n_static_experts,
            op1=nl.maximum,
            operand1=0,
        )
        active_count_sb = loop_count_sb
    loop_count = nisa.register_alloc()
    nisa.register_load(loop_count, active_count_sb)
    nl.fori_loop(0, loop_count, expert.run)


def _active_expert_ids(local_affinities: nl.NkiTensor, T_physical: int, E_L: int) -> tuple:
    """Find the experts with a nonzero affinity for any token.

    Returns:
        INT32 ``[1, E_L + 1]`` active expert ids in ascending order, padded
        with E_L, and the INT32 ``[1, 1]`` count of active experts.
    """
    if local_affinities.buffer == nl.sbuf:
        # An SBUF input holds token t on partition t, and a DMA cannot gather
        # its rows onto one partition. A matmul with a ones column counts each
        # expert's tokens instead.
        nonzero_sb = nl.ndarray((T_physical, E_L), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=nonzero_sb, data=local_affinities, op0=nl.not_equal, operand0=0.0)
        ones_sb = nl.ndarray((T_physical, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(dst=ones_sb, value=1.0)
        count_psum = nl.ndarray((1, E_L), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=count_psum, stationary=ones_sb, moving=nonzero_sb, is_stationary_onezero=True)
        active_sb = nl.ndarray((1, E_L), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=active_sb, src=count_psum)
    else:
        affinity_row_sb = nl.ndarray((1, T_physical, E_L), dtype=local_affinities.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=affinity_row_sb,
            src=local_affinities.ap(pattern=[[T_physical * E_L, 1], [E_L, T_physical], [1, E_L]], offset=0),
        )
        nonzero_sb = nl.ndarray((1, T_physical, E_L), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=nonzero_sb, data=affinity_row_sb, op0=nl.not_equal, operand0=0.0)
        active_sb = nl.ndarray((1, E_L), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(
            dst=active_sb,
            op=nl.maximum,
            data=nonzero_sb.ap(pattern=[[T_physical * E_L, 1], [1, E_L], [E_L, T_physical]], offset=0),
            axis=2,
        )
    # [ids of the active experts, E_L padding, count]
    ids_and_count_sb = nl.ndarray((1, E_L + 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.nonzero_with_count(dst=ids_and_count_sb, src=active_sb, index_offset=0, padding_val=E_L)
    active_count_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=active_count_sb, src=ids_and_count_sb[0:1, E_L : E_L + 1])
    expert_ids_sb = nl.ndarray((1, E_L + 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=expert_ids_sb, value=E_L)
    nisa.tensor_copy(dst=expert_ids_sb[0:1, 0:E_L], src=ids_and_count_sb[0:1, 0:E_L])
    return expert_ids_sb, active_count_sb


SKIP_IDS = []
"""Trace-time record of the skip id of each ``_ExpertPipeline`` (``_stack_skip_id``), for the tests. No instruction."""


def _stack_skip_id(weights: tuple) -> int:
    """Return the expert id whose runtime-offset DMAs skip for every one of ``weights`` (the expert weights and scales,
    [E_L, ...] each): the first id past the end of the tensor that holds them.

    A layer's experts are often a static view of a stack of layers (``decode_model._group_stack``). A runtime-offset
    DMA's bound is the whole stack, so the id E_L is the next layer's expert 0, inside the bound, and its DMA does not
    skip. The id ceil((stack elements - view offset) / elements of one expert) lands at the stack's end for every
    layer: E_L for a tensor of its own."""
    E_L = weights[0].shape[0]
    skip = E_L
    for weight in weights:
        stride = math.prod(weight.shape[1:])
        stack = math.prod(getattr(weight, "_storage_shape", None) or weight.shape)
        offset = getattr(weight, "offset", 0) or 0
        kernel_assert(
            isinstance(offset, int) and stack >= offset + E_L * stride,
            f"_stack_skip_id: a static view of {E_L} experts inside its tensor, got offset {offset}, {stack} elements",
        )
        skip = max(skip, -(-(stack - offset) // stride))
    for weight in weights:
        stride = math.prod(weight.shape[1:])
        kernel_assert(
            (getattr(weight, "offset", 0) or 0) + skip * stride
            >= math.prod(getattr(weight, "_storage_shape", None) or weight.shape),
            "_stack_skip_id: the skip id must lie past the end of every weight tensor",
        )
    return skip


class _ExpertPipeline:
    """The loop body: the projections of one active expert per iteration.

    ``expert_ids_sb[0]`` is the expert of the current iteration and
    ``expert_ids_sb[1]`` the next one. The DMAs use the ids as runtime
    offsets into the weights, and an out-of-bounds id skips the DMA. Every
    DMA completes within its iteration, because the loop ends each iteration
    with a barrier. An iteration therefore loads the current expert's down
    projection, and the next expert's gate and up projections as soon as
    the current matmuls have read their buffers.

    The weight and scale DMAs take their ids from ``load_ids_sb``: the same
    list with the padding id E_L moved to ``skip_id`` (``_stack_skip_id``),
    past the end of the weight tensors, so a padding load (the next expert
    after the last active one) skips; E_L itself is the next layer's expert 0
    when the experts are a view of a layer stack. The affinity loads keep
    the ids.

    With ``resident`` (``moe_tkg_decode`` ``resident_out``) the pipeline's
    buffers are the static loop's buffers of its last expert, and the first
    load id becomes ``skip_id`` too when the resident expert is first in the
    list (``skip_first_resident``): its DMAs skip, so that expert's
    projections read the bytes the static loop left in the buffers.
    """

    def __init__(
        self,
        input_quant_sb: nl.NkiTensor,
        input_scale_sb: nl.NkiTensor,
        output_accum_sb: nl.NkiTensor,
        expert_ids_sb: nl.NkiTensor,
        local_affinities: nl.NkiTensor,
        expert_gate_up_weights: nl.NkiTensor,
        expert_down_weights: nl.NkiTensor,
        expert_gate_up_weights_scale: nl.NkiTensor,
        expert_down_weights_scale: nl.NkiTensor,
        beta: float,
        linear_beta: float,
        T: int,
        T_physical: int,
        resident: dict = None,
    ):
        self.input_quant_sb = input_quant_sb
        self.input_scale_sb = input_scale_sb
        self.output_accum_sb = output_accum_sb
        self.expert_ids_sb = expert_ids_sb
        self.local_affinities = local_affinities
        self.expert_gate_up_weights = expert_gate_up_weights
        self.expert_down_weights = expert_down_weights
        self.expert_gate_up_weights_scale = expert_gate_up_weights_scale
        self.expert_down_weights_scale = expert_down_weights_scale
        self.beta = beta
        self.linear_beta = linear_beta
        self.T = T
        self.T_physical = T_physical
        self.E_L = expert_gate_up_weights.shape[0]
        self.skip_id = _stack_skip_id(
            (expert_gate_up_weights, expert_down_weights, expert_gate_up_weights_scale, expert_down_weights_scale)
        )
        SKIP_IDS.append(self.skip_id)
        self.load_ids_sb = self._load_ids(expert_ids_sb)
        self.D = expert_down_weights.shape[-1]
        self.I = expert_gate_up_weights.shape[-1]
        n_D512 = self.D // MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
        n_I512 = self.I // MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
        weight_dtype = expert_gate_up_weights.dtype
        self.down_scale_slot = 0
        if resident:
            self._take_resident(resident, n_D512, n_I512)
            return
        self.gate_weight_sb = nl.ndarray((_PMAX, n_D512, self.I), dtype=weight_dtype, buffer=nl.sbuf)
        self.up_weight_sb = nl.ndarray((_PMAX, n_D512, self.I), dtype=weight_dtype, buffer=nl.sbuf)
        self.down_weight_sb = nl.ndarray((_PMAX, n_I512, self.D), dtype=expert_down_weights.dtype, buffer=nl.sbuf)
        # The gate and the up scales share one buffer, in the slots of the
        # static loop: gate at partition offset 0 of each quadrant, up at 4
        # (see _load_scale_rows). Each of the 5 buffers spans all partitions,
        # and every tile of T <= 32 tokens lands in the first 32 partitions,
        # so a second scale buffer left the T = 4 tiles too little SBUF there.
        self.gate_up_scale_sb = nl.ndarray((_PMAX, n_D512, self.I), dtype=nl.uint8, buffer=nl.sbuf)
        self.down_scale_sb = nl.ndarray((_PMAX, n_I512, self.D), dtype=nl.uint8, buffer=nl.sbuf)
        self.down_scale_view = self.down_scale_sb
        self._affinity_tile()

    def _affinity_tile(self):
        self.expert_affinity_raw_sb = nl.ndarray((self.T, 1), dtype=self.local_affinities.dtype, buffer=nl.sbuf)
        if self.T > self.T_physical:
            nisa.memset(dst=self.expert_affinity_raw_sb, value=0.0)

    def _take_resident(self, resident: dict, n_D512: int, n_I512: int):
        """Use the static loop's buffers of its last expert (``_resident_last_expert``) as the pipeline's buffers."""
        kernel_assert(
            resident["expert"] == self.E_L - 1,
            f"resident buffers are the static loop's buffers of expert {self.E_L - 1}, got expert {resident['expert']}",
        )
        for key in ("gate", "up", "down"):
            buffer = resident[key]
            kernel_assert(
                buffer.dtype == self.expert_gate_up_weights.dtype and tuple(buffer.shape) == (_PMAX, n_D512 * self.I),
                f"resident {key} buffer: expected [{_PMAX}, {n_D512 * self.I}] {self.expert_gate_up_weights.dtype}, "
                f"got {tuple(buffer.shape)} {buffer.dtype}",
            )
        self.gate_weight_sb = resident["gate"].reshape((_PMAX, n_D512, self.I))
        self.up_weight_sb = resident["up"].reshape((_PMAX, n_D512, self.I))
        self.down_weight_sb = resident["down"].reshape((_PMAX, n_I512, self.D))
        self.gate_up_scale_sb = resident["gate_up_scale"].reshape((_PMAX, n_D512, self.I))
        self.down_scale_sb = resident["down_scale"].reshape((_PMAX, n_I512, self.D))
        self.down_scale_slot = resident["down_scale_slot"]
        self.down_scale_view = self.down_scale_sb[_SCALE_ROWS_PER_QUADRANT * self.down_scale_slot :, :, :]
        self._affinity_tile()

    def load_affinity_table(self, dge_mode=None, engine=None):
        """Load ``[T, E_L + 1]`` affinities with a zero last column, for the static slots. ``dge_mode`` / ``engine``
        (None, the default: the static queue): the queue of the load, e.g. the SP hardware DGE inside a
        branch body."""
        E_L = self.E_L
        self.affinity_table_sb = nl.ndarray((self.T, E_L + 1), dtype=self.local_affinities.dtype, buffer=nl.sbuf)
        nisa.memset(dst=self.affinity_table_sb, value=0.0)
        nisa.dma_copy(
            dst=self.affinity_table_sb[: self.T_physical, 0:E_L],
            src=self.local_affinities,
            dge_mode=nisa.dge_mode.none if dge_mode is None else dge_mode,
            **({} if engine is None else {"engine": engine}),
        )

    def _load_ids(self, expert_ids_sb: nl.NkiTensor) -> nl.NkiTensor:
        """Return the INT32 [1, E_L + 1] weight-load ids: ``expert_ids_sb`` with E_L moved to ``skip_id``."""
        vector = nisa.engine.vector
        n = expert_ids_sb.shape[1]
        padding_sb = nl.ndarray((1, n), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=padding_sb,
            data=expert_ids_sb,
            op0=nl.equal,
            operand0=self.E_L,
            op1=nl.multiply,
            operand1=self.skip_id - self.E_L,
            engine=vector,
        )
        load_ids_sb = nl.ndarray((1, n), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=load_ids_sb, data1=expert_ids_sb, data2=padding_sb, op=nl.add, engine=vector)
        return load_ids_sb

    def skip_first_resident(self, expert: int):
        """Move the first load id to ``skip_id`` when the first active expert is ``expert`` (the resident one): its
        weight and scale DMAs skip, past the end of the weight tensors."""
        vector = nisa.engine.vector
        first_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=first_sb,
            data=self.expert_ids_sb[0:1, 0:1],
            op0=nl.equal,
            operand0=expert,
            op1=nl.multiply,
            operand1=self.skip_id - expert,
            engine=vector,
        )
        moved_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=moved_sb, data1=self.load_ids_sb[0:1, 0:1], data2=first_sb, op=nl.add, engine=vector)
        nisa.tensor_copy(dst=self.load_ids_sb[0:1, 0:1], src=moved_sb, engine=vector)

    def _expert_id(self, id_slot: int) -> nl.NkiTensor:
        return self.expert_ids_sb[0:1, id_slot : id_slot + 1]

    def _load_id(self, id_slot: int) -> nl.NkiTensor:
        return self.load_ids_sb[0:1, id_slot : id_slot + 1]

    def _load_rows(self, dst: nl.NkiTensor, src: nl.NkiTensor, row_stride: int, offset: int, id_slot: int):
        """Load ``dst.shape[0]`` rows of ``dst.shape[1]`` elements of expert ``expert_ids[id_slot]``."""
        nisa.dma_copy(
            dst=dst,
            src=src.ap(
                pattern=[[row_stride, dst.shape[0]], [1, dst.shape[1]]],
                offset=offset,
                scalar_offset=self._load_id(id_slot),
                indirect_dim=0,
            ),
            oob_mode=nisa.oob_mode.skip,
            dge_mode=nisa.dge_mode.hwdge,
            engine=nisa.engine.sync,
        )

    def _load_scale_rows(
        self,
        dst: nl.NkiTensor,
        src: nl.NkiTensor,
        id_slot: int,
        slot: int = 0,
        projections_per_row: int = 1,
        dst_slot: int = None,
    ):
        """Load the 16 MX scale rows of a projection into partitions 32 q + 4 slot + r, r < 4, of each quadrant q.

        A scale row of ``src`` holds ``projections_per_row`` projections (the
        gate and the up scales in the gate/up layout), and ``slot`` picks one.
        DMA r moves rows r, 4 + r, 8 + r and 12 + r into partition
        4 * slot + r of the four quadrants. A runtime-offset DMA moves its
        partitions on as many DMA engines at once, and engines writing the
        same group of 8 partitions share its bandwidth: one DMA per quadrant
        (4 neighboring partitions) ran slower per byte than the
        weight loads. ``dst_slot`` (None: ``slot``) puts the rows in another
        slot of ``dst`` than the projection they come from.
        """
        row_size = dst.shape[1]
        n_quadrants = _PMAX // _SBUF_QUADRANT_PARTITIONS
        first_partition = _SCALE_ROWS_PER_QUADRANT * (slot if dst_slot is None else dst_slot)
        for row_idx in range(_SCALE_ROWS_PER_QUADRANT):
            nisa.dma_copy(
                dst=dst[first_partition + row_idx : _PMAX : _SBUF_QUADRANT_PARTITIONS, :],
                src=src.ap(
                    pattern=[
                        [_SCALE_ROWS_PER_QUADRANT * projections_per_row * row_size, n_quadrants],
                        [1, row_size],
                    ],
                    offset=(projections_per_row * row_idx + slot) * row_size,
                    scalar_offset=self._load_id(id_slot),
                    indirect_dim=0,
                ),
                oob_mode=nisa.oob_mode.skip,
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.sync,
            )

    def _load_gate_up_weight(self, weight_sb: nl.NkiTensor, projection_idx: int, id_slot: int):
        row = weight_sb.shape[1] * weight_sb.shape[2]
        self._load_rows(
            dst=weight_sb.reshape((_PMAX, row)),
            src=self.expert_gate_up_weights,
            row_stride=2 * row,
            offset=projection_idx * row,
            id_slot=id_slot,
        )

    def load_gate_up(self, id_slot: int, gate: bool = True, up: bool = True):
        """Load the gate and/or up weights, and with the up weights the gate/up scales, of ``expert_ids[id_slot]``."""
        if gate:
            self._load_gate_up_weight(self.gate_weight_sb, GATE_FUSED_IDX, id_slot)
        if up:
            self._load_gate_up_weight(self.up_weight_sb, UP_FUSED_IDX, id_slot)
            scale_row = self.gate_up_scale_sb.shape[1] * self.gate_up_scale_sb.shape[2]
            for slot in (GATE_FUSED_IDX, UP_FUSED_IDX):
                self._load_scale_rows(
                    dst=self.gate_up_scale_sb.reshape((_PMAX, scale_row)),
                    src=self.expert_gate_up_weights_scale,
                    id_slot=id_slot,
                    slot=slot,
                    projections_per_row=2,
                )

    def _load_down(self):
        row = self.down_weight_sb.shape[1] * self.down_weight_sb.shape[2]
        self._load_rows(
            dst=self.down_weight_sb.reshape((_PMAX, row)),
            src=self.expert_down_weights,
            row_stride=row,
            offset=0,
            id_slot=0,
        )
        self._load_scale_rows(
            dst=self.down_scale_sb.reshape((_PMAX, row)),
            src=self.expert_down_weights_scale,
            id_slot=0,
            **({"dst_slot": self.down_scale_slot} if self.down_scale_slot else {}),
        )

    def run(self, _iteration, static_slot: bool = False):
        """Run the expert ``expert_ids[0]``, then advance the list by one.

        A static slot runs before the loop, whatever the active count. Past
        the count its id is out of bounds: its DMAs are skipped, its affinity
        stays zero, and the accumulation skips every token whose affinity is
        zero, so the stale buffers it computes on add nothing.
        """
        T, D, I, E_L = self.T, self.D, self.I, self.E_L
        n_D512 = D // MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
        n_I512 = I // MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
        if static_slot:
            # Column E_L of the table is zero: the affinity of a slot past the count.
            nisa.dma_copy(
                dst=self.expert_affinity_raw_sb,
                src=self.affinity_table_sb.ap(
                    pattern=[[E_L + 1, T], [1, 1]], offset=0, scalar_offset=self._expert_id(0), indirect_dim=1
                ),
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.sync,
            )
        else:
            nisa.dma_copy(
                dst=self.expert_affinity_raw_sb[: self.T_physical, :],
                src=self.local_affinities.ap(
                    pattern=[[E_L, self.T_physical], [1, 1]], offset=0, scalar_offset=self._expert_id(0), indirect_dim=1
                ),
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.sync,
            )
        self._load_down()

        # The gate and up matmuls run in groups of chunks, gate then up, so only
        # a group's gate terms are live. The next expert's gate weights load
        # after the last gate matmuls, its up weights and scales after the last
        # up matmuls.
        n_chunks = div_ceil(I, _GATE_UP_CHUNK_I)
        activation_token_major = nl.ndarray((T, I), dtype=nl.float16, buffer=nl.sbuf)
        for group_start in range(0, n_chunks, _DYNAMIC_GATE_UP_GROUP_CHUNKS):
            group = range(group_start, min(group_start + _DYNAMIC_GATE_UP_GROUP_CHUNKS, n_chunks))
            gate_terms_f32 = {}
            for chunk_idx in group:
                chunk_size = min(_GATE_UP_CHUNK_I, I - chunk_idx * _GATE_UP_CHUNK_I)
                gate_psum = _project_chunk(
                    input_quant_sb=self.input_quant_sb,
                    input_scale_sb=self.input_scale_sb,
                    weight_sb=self.gate_weight_sb,
                    weight_scale_sb=self.gate_up_scale_sb[_SCALE_ROWS_PER_QUADRANT * GATE_FUSED_IDX :, :, :],
                    n_H512=n_D512,
                    chunk_idx=chunk_idx,
                    chunk_size=chunk_size,
                    T=T,
                )
                gate_sigmoid_f32 = nl.ndarray((T, chunk_size), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=gate_sigmoid_f32, op=nl.sigmoid, data=gate_psum)
                gate_term_f32 = nl.ndarray((T, chunk_size), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=gate_term_f32, op=nl.tanh, data=gate_psum, scale=1.0 / self.beta)
                nisa.scalar_tensor_tensor(
                    dst=gate_term_f32,
                    data=gate_term_f32,
                    op0=nl.multiply,
                    operand0=self.beta,
                    op1=nl.multiply,
                    operand1=gate_sigmoid_f32,
                )
                gate_terms_f32[chunk_idx] = gate_term_f32
            if group.stop == n_chunks:
                self.load_gate_up(id_slot=1, up=False)
            for chunk_idx in group:
                chunk_size = min(_GATE_UP_CHUNK_I, I - chunk_idx * _GATE_UP_CHUNK_I)
                up_psum = _project_chunk(
                    input_quant_sb=self.input_quant_sb,
                    input_scale_sb=self.input_scale_sb,
                    weight_sb=self.up_weight_sb,
                    weight_scale_sb=self.gate_up_scale_sb[_SCALE_ROWS_PER_QUADRANT * UP_FUSED_IDX :, :, :],
                    n_H512=n_D512,
                    chunk_idx=chunk_idx,
                    chunk_size=chunk_size,
                    T=T,
                )
                up_tanh_f32 = nl.ndarray((T, chunk_size), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=up_tanh_f32, op=nl.tanh, data=up_psum, scale=1.0 / self.linear_beta)
                nisa.scalar_tensor_tensor(
                    dst=activation_token_major[:, nl.ds(chunk_idx * _GATE_UP_CHUNK_I, chunk_size)],
                    data=up_tanh_f32,
                    op0=nl.multiply,
                    operand0=self.linear_beta,
                    op1=nl.multiply,
                    operand1=gate_terms_f32[chunk_idx],
                )
        self.load_gate_up(id_slot=1, gate=False)

        # The down matmuls run I/512 tile by tile into one FP32 PSUM accumulator
        # per H/512 tile, so each tile needs only its own chunk of the
        # transposed and quantized activation. Chunk c's transposes are issued
        # after the matmuls of chunk c - 2, which hides the SiTU and
        # quantization latency of the last chunks. The H tiles run in groups,
        # so only a group's accumulators occupy PSUM; a group repeats no work,
        # because every (I tile, H tile) pair runs exactly one matmul.
        tile_H = min(D, _DOWN_TILE_H)
        n_H_tiles = div_ceil(D, tile_H)
        tiles_per_chunk = _TRANSPOSE_CHUNK_I // MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
        n_chunks = div_ceil(n_I512, tiles_per_chunk)
        quantized_chunks = []
        for chunk_idx in range(min(_TRANSPOSE_LOOKAHEAD_CHUNKS, n_chunks)):
            quantized_chunks.append(
                _transpose_and_quantize_chunk(
                    activation_token_major=activation_token_major, chunk_idx=chunk_idx, I=I, T=T
                )
            )
        for group_first_tile in range(0, n_H_tiles, _DOWN_TILES_PER_GROUP):
            group_tiles = range(group_first_tile, min(group_first_tile + _DOWN_TILES_PER_GROUP, n_H_tiles))
            expert_out_psums = {}
            for tile_h in group_tiles:
                tile_H_actual = min(tile_H, D - tile_h * tile_H)
                expert_out_psums[tile_h] = nl.ndarray((T, tile_H_actual), dtype=nl.float32, buffer=nl.psum)
            for chunk_idx in range(n_chunks):
                next_chunk_idx = chunk_idx + _TRANSPOSE_LOOKAHEAD_CHUNKS
                activation_quant_sb, activation_scale_sb = quantized_chunks[chunk_idx]
                for chunk_tile in range(activation_quant_sb.shape[1]):
                    tile_i = chunk_idx * tiles_per_chunk + chunk_tile
                    for tile_h in group_tiles:
                        H_slice = nl.ds(tile_h * tile_H, expert_out_psums[tile_h].shape[1])
                        nisa.nc_matmul_mx(
                            dst=expert_out_psums[tile_h],
                            stationary=activation_quant_sb[:, chunk_tile, :],
                            moving=self.down_weight_sb[:, tile_i, H_slice],
                            stationary_scale=activation_scale_sb[:, chunk_tile, :],
                            moving_scale=self.down_scale_view[:, tile_i, H_slice],
                        )
                if next_chunk_idx < n_chunks and len(quantized_chunks) <= next_chunk_idx:
                    quantized_chunks.append(
                        _transpose_and_quantize_chunk(
                            activation_token_major=activation_token_major, chunk_idx=next_chunk_idx, I=I, T=T
                        )
                    )

            if group_first_tile == 0:
                expert_affinity_f32_sb = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=expert_affinity_f32_sb, src=self.expert_affinity_raw_sb)
                if static_slot:
                    routed_sb = nl.ndarray((T, 1), dtype=nl.uint32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=routed_sb, data=expert_affinity_f32_sb, op0=nl.not_equal, operand0=0.0)
            for tile_h in group_tiles:
                tile_H_actual = expert_out_psums[tile_h].shape[1]
                H_slice = nl.ds(tile_h * tile_H, tile_H_actual)
                expert_out_f32 = nl.ndarray((T, tile_H_actual), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=expert_out_f32, src=expert_out_psums[tile_h])
                if static_slot:
                    # out + 0 * x == out for a finite x, so skipping a zero
                    # affinity gives the bits of the multiply-add.
                    weighted_sum_f32 = nl.ndarray((T, tile_H_actual), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.scalar_tensor_tensor(
                        dst=weighted_sum_f32,
                        data=expert_out_f32,
                        op0=nl.multiply,
                        operand0=expert_affinity_f32_sb,
                        op1=nl.add,
                        operand1=self.output_accum_sb[:, H_slice],
                    )
                    nisa.tensor_copy_predicated(
                        dst=self.output_accum_sb[:, H_slice],
                        src=weighted_sum_f32,
                        predicate=routed_sb.ap(pattern=[[1, T], [0, tile_H_actual]], offset=0),
                    )
                else:
                    nisa.scalar_tensor_tensor(
                        dst=self.output_accum_sb[:, H_slice],
                        data=expert_out_f32,
                        op0=nl.multiply,
                        operand0=expert_affinity_f32_sb,
                        op1=nl.add,
                        operand1=self.output_accum_sb[:, H_slice],
                    )
        # Advance to the next active expert.
        expert_ids_next_sb = nl.ndarray((1, E_L), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=expert_ids_next_sb, src=self.expert_ids_sb[0:1, 1 : E_L + 1])
        nisa.tensor_copy(dst=self.expert_ids_sb[0:1, 0:E_L], src=expert_ids_next_sb)
        load_ids_next_sb = nl.ndarray((1, E_L), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=load_ids_next_sb, src=self.load_ids_sb[0:1, 1 : E_L + 1], engine=nisa.engine.vector)
        nisa.tensor_copy(dst=self.load_ids_sb[0:1, 0:E_L], src=load_ids_next_sb, engine=nisa.engine.vector)


def active_experts_in_slots(
    input_quant_sb: nl.NkiTensor,
    input_scale_sb: nl.NkiTensor,
    affinities_sb: nl.NkiTensor,
    affinities_hbm: nl.NkiTensor,
    expert_gate_up_weights: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_gate_up_weights_scale: nl.NkiTensor,
    expert_down_weights_scale: nl.NkiTensor,
    beta: float,
    linear_beta: float,
    output: nl.NkiTensor,
    resident: dict = None,
    static_first: bool = False,
):
    """Accumulate only the active local experts of a tile of 128 tokens (or slots), and store the BF16 ``output``.

    The dynamic loop of ``_active_experts_dynamic``, for a loop body (the later slot tiles of the 2-server MoE
    compaction): the active ids come from the SBUF ``affinities_sb`` [128, E_L] with a ones matmul (no DMA), the loop
    reads each expert's affinity column from ``affinities_hbm`` (the same values in HBM) with a runtime offset, and
    the output stores use DGE DMAs (a loop body has no static DMA queue). A zero-affinity expert would add exactly
    zero, so skipping it gives the bits of ``_all_experts_in_slots`` for finite values.

    ``resident`` (the ``resident_out`` of the call's static loop; None or empty keeps the program): the loop
    runs in the static loop's buffers of its last expert E_L - 1. When that expert is the first in the ascending
    list (so the only active one), its weight and scale DMAs take the skip id past the weight tensors and skip
    (``_ExpertPipeline.skip_first_resident``): it reads the bytes the static loop left in the buffers, the same values
    as a reload. Every other expert loads as without ``resident``.
    ``static_first`` (False, the default, keeps the program): the first active expert runs as a static slot before the
    loop (``_ExpertPipeline.run`` ``static_slot``, as ``_active_experts_dynamic`` does), and the loop the other
    ``count - 1``, so a tile of one active expert (the common later tile) runs no loop trip and no trip barrier. The
    same experts in the same ascending order; the static slot's accumulation skips the zero-affinity slots, which
    the loop's adds leave unchanged for finite values.
    """
    T, E_L = affinities_sb.shape
    D = expert_down_weights.shape[-1]
    output_accum_sb = nl.ndarray((T, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=output_accum_sb, value=0.0)
    expert_ids_sb, active_count_sb = _active_expert_ids(local_affinities=affinities_sb, T_physical=T, E_L=E_L)
    expert = _ExpertPipeline(
        input_quant_sb=input_quant_sb,
        input_scale_sb=input_scale_sb,
        output_accum_sb=output_accum_sb,
        expert_ids_sb=expert_ids_sb,
        local_affinities=affinities_hbm,
        expert_gate_up_weights=expert_gate_up_weights,
        expert_down_weights=expert_down_weights,
        expert_gate_up_weights_scale=expert_gate_up_weights_scale,
        expert_down_weights_scale=expert_down_weights_scale,
        beta=beta,
        linear_beta=linear_beta,
        T=T,
        T_physical=T,
        **({"resident": resident} if resident else {}),
    )
    if resident:
        expert.skip_first_resident(resident["expert"])
    expert.load_gate_up(id_slot=0)
    if static_first:
        # The first active expert as a static slot (its affinity column from a table with a zero last column), the
        # loop the other count - 1.
        expert.load_affinity_table(dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
        expert.run(None, static_slot=True)
        rest_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=rest_sb,
            data=active_count_sb,
            op0=nl.subtract,
            operand0=1,
            op1=nl.maximum,
            operand1=0,
            engine=nisa.engine.vector,
        )
        active_count_sb = rest_sb
    loop_count = nisa.register_alloc()
    nisa.register_load(loop_count, active_count_sb)
    nl.fori_loop(0, loop_count, expert.run)
    tile_H = min(D, _OUTPUT_TILE_H)
    for tile_h in range(div_ceil(D, tile_H)):
        H_slice = nl.ds(tile_h * tile_H, min(tile_H, D - tile_h * tile_H))
        output_bf16_sb = nl.ndarray((T, min(tile_H, D - tile_h * tile_H)), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=output_bf16_sb, src=output_accum_sb[:, H_slice])
        nisa.dma_copy(dst=output[:, H_slice], src=output_bf16_sb, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)


def _store_accumulated_output(output_accum_sb: nl.NkiTensor, output: nl.NkiTensor, T: int):
    """Round the FP32 accumulator of the dynamic loop to BF16 and store it, one output tile at a time."""
    T_physical, D = output.shape
    tile_H = min(D, _OUTPUT_TILE_H)
    n_H_tiles = div_ceil(D, tile_H)
    for tile_h in nl.sequential_range(n_H_tiles):
        tile_H_offset = tile_h * tile_H
        tile_H_actual = min(tile_H, D - tile_H_offset)
        H_slice = nl.ds(tile_H_offset, tile_H_actual)
        output_bf16_sb = nl.ndarray((T, tile_H_actual), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=output_bf16_sb, src=output_accum_sb[:, H_slice])
        nisa.dma_copy(
            dst=output[:, H_slice],
            src=output_bf16_sb[:T_physical, :],
            dge_mode=nisa.dge_mode.none,
        )


class _LatentShape:
    """The shape and dtype of the natural latent that a ``quantized_latent`` input stands for (``moe_tkg_decode``)."""

    def __init__(self, tokens: int, columns: int):
        self.shape = (tokens, columns)
        self.dtype = nl.bfloat16
        self.buffer = nl.sbuf
        self.offset = 0


def _check_quantized_latent(latent, quantized_latent: tuple, dynamic: bool, slots: int, direct_latent: bool):
    """Check a ``quantized_latent`` (codes, scales) input of ``moe_tkg_decode`` and return the [T, D] latent shape it
    stands for."""
    kernel_assert(latent is None, "moe_tkg_decode takes latent None with quantized_latent")
    kernel_assert(not (dynamic or slots), "quantized_latent applies to the static loop only")
    kernel_assert(not direct_latent, "quantized_latent has no latent tile: it takes no direct_latent")
    kernel_assert(len(quantized_latent) == 2, "quantized_latent is a (codes, scales) pair")
    codes, scales = quantized_latent
    kernel_assert(
        len(codes.shape) == 3
        and codes.shape[0] == _PMAX
        and codes.dtype == nl.float8_e4m3fn_x4
        and codes.buffer == nl.sbuf
        and tuple(scales.shape) == tuple(codes.shape)
        and scales.dtype == nl.uint8
        and scales.buffer == nl.sbuf
        and codes.shape[2] % _q_width == 0,
        f"quantized_latent must be SBUF codes [128, D/512, T] float8_e4m3fn_x4 and scales of the same shape UINT8 with "
        f"T a multiple of {_q_width}, got {codes.dtype} {codes.shape}, {scales.dtype} {scales.shape}",
    )
    return _LatentShape(codes.shape[2], codes.shape[1] * MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM)


def _load_natural_latent(
    latent: nl.NkiTensor,
    T: int,
    T_physical: int,
    D: int,
) -> nl.NkiTensor:
    """Load the natural ``[T, D]`` latent with one contiguous DMA, zero-padded to ``T`` rows."""
    latent_sb = nl.ndarray((T, D), dtype=nl.bfloat16, buffer=nl.sbuf)
    if T > T_physical:
        nisa.memset(dst=latent_sb, value=0.0)
    # Static descriptors: with the default mode, a latent of 32 or more rows
    # goes to the software DGE, which moved it at T=128 behind the
    # first expert's weights, and delayed the first matmul.
    nisa.dma_copy(dst=latent_sb[:T_physical, :], src=latent, dge_mode=nisa.dge_mode.none)
    return latent_sb


def _quantize_natural_latent(
    latent_sb: nl.NkiTensor,
    T: int,
    D: int,
    issued: list = None,
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """Swizzle the loaded natural ``[T,D]`` latent and quantize it to MXFP8.

    Each stride-4 column view is moved to partitions with a PE transpose.
    Element-granular swizzle DMAs cost much descriptor generation time
    per row and dominate large T. ``issued`` (a list), when given, gets the
    transposes in order.
    """
    n_D512 = D // MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
    input_swizzled_sb = nl.ndarray((_PMAX, n_D512, T, _q_width), dtype=nl.bfloat16, buffer=nl.sbuf)
    for tile_h in nl.affine_range(n_D512):
        swizzle_psum = nl.ndarray((_PMAX, _q_width, T), dtype=nl.bfloat16, buffer=nl.psum)
        for q_idx in nl.affine_range(_q_width):
            transpose = nisa.nc_transpose(
                dst=swizzle_psum[:, q_idx, :],
                data=latent_sb.ap(
                    pattern=[[D, T], [_q_width, _PMAX]],
                    offset=tile_h * MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM + q_idx,
                ),
            )
            if issued is not None:
                issued.append(transpose)
        nisa.tensor_copy(
            dst=input_swizzled_sb[:, tile_h, :, :],
            src=swizzle_psum.ap(pattern=[[_q_width * T, _PMAX], [1, T], [T, _q_width]]),
        )

    input_quant_sb = nl.ndarray((_PMAX, n_D512, T), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
    input_scale_sb = nl.ndarray((_PMAX, n_D512, T), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.quantize_mx(
        src=input_swizzled_sb,
        dst=input_quant_sb,
        dst_scale=input_scale_sb,
    )
    return input_quant_sb, input_scale_sb


def _load_mx_weight(
    weight: nl.NkiTensor, weight_buffer: nl.NkiTensor, dge_mode=nisa.dge_mode.none, engine=None, priority: int = None
) -> nl.NkiTensor:
    """Load one projection's packed MX ``[128, n_K512, N]`` weights into ``weight_buffer`` (a static DMA by default)."""
    weight_sb = weight_buffer.reshape(weight.shape)
    options = {
        **({"engine": engine} if engine is not None else {}),
        **({"priority": priority} if priority is not None else {}),
    }
    nisa.dma_copy(dst=weight_sb, src=weight, dge_mode=dge_mode, **options)
    return weight_sb


@contextmanager
def scale_rows_queue(queue: str):
    """Trace the static MX scale loads (``_load_scale_rows``) of the scope on the DMA queue ``queue``
    (``SCALE_ROWS_QUEUES``).

    ``static`` (the default form) puts them on the static DMA queue of the expert weights. A DMA of that queue starts
    after the earlier DMAs of the queue on each DMA engine, and the queue has one completion semaphore for each of its
    last 16 DMAs. So a 43 KB scale load waits behind 2.75 MB weight loads, and the matmuls that read the scales wait
    too. ``sync_hw`` and ``act_hw`` put the scale loads on hardware DGE from the Sync or the Activation engine, a queue
    of their own. The same bytes go to the same partitions, so the values do not change.

    Example:
        >>> with scale_rows_queue("sync_hw"):
        ...     pass
    """
    global _SCALE_ROWS_QUEUE
    kernel_assert(queue in SCALE_ROWS_QUEUES, f"scale_rows_queue must be one of {SCALE_ROWS_QUEUES}, got {queue!r}")
    previous = _SCALE_ROWS_QUEUE
    _SCALE_ROWS_QUEUE = queue
    try:
        yield
    finally:
        _SCALE_ROWS_QUEUE = previous


def _scale_rows_dma_options() -> dict:
    """Return the DMA queue options of a static MX scale load (``_SCALE_ROWS_QUEUE``)."""
    if _SCALE_ROWS_QUEUE == "sync_hw":
        return {"dge_mode": nisa.dge_mode.hwdge, "engine": nisa.engine.sync}
    if _SCALE_ROWS_QUEUE == "act_hw":
        return {"dge_mode": nisa.dge_mode.hwdge, "engine": nisa.engine.scalar}
    return {"dge_mode": nisa.dge_mode.none}


@contextmanager
def scale_rows_across_quadrants(on: bool):
    """Trace the static MX scale loads (``_load_scale_rows``) of the scope across the quadrants when ``on``.

    False (the default form) issues one DMA for each quadrant, expert and projection (4 neighboring partitions, one DMA
    engine). True issues one DMA for each row r of a quadrant, expert and projection, over partition r of the 4
    quadrants (4 DMA engines), as ``_load_scale_rows_of`` does. The same bytes go to the same partitions, so the values
    do not change.
    """
    global _SCALE_ROWS_ACROSS_QUADRANTS
    previous = _SCALE_ROWS_ACROSS_QUADRANTS
    _SCALE_ROWS_ACROSS_QUADRANTS = bool(on)
    try:
        yield
    finally:
        _SCALE_ROWS_ACROSS_QUADRANTS = previous


def _load_scale_rows(
    dst: nl.NkiTensor,
    scale: nl.NkiTensor,
    first_expert: int,
    n_experts: int,
    projections_per_expert: int,
    projections: tuple = None,
    first_slot: int = 0,
    priority: int = None,
):
    """Load the E8M0 scale rows of consecutive experts into one SBUF buffer.

    The matmul reads a projection's 16 scale rows from 4 partitions per SBUF
    quadrant, starting at partition offset 0, 4, 8 or 12. Projection slot
    ``k = projections_per_expert * expert + projection`` of the group goes to
    offset 4 * k. A static DMA picks its DMA engines by destination
    partition, so rows at offset 0 alone would put every scale packet on two
    of the 16 engines and make them the bottleneck of the weight stream; the
    four slot offsets spread them over eight engines.

    Args:
        dst: ``[128, D*I/512]`` UINT8 SBUF buffer.
        scale: UINT8 ``[E_L, 16, 2, D/512, I]`` gate/up scales when
            ``projections_per_expert == 2``, or ``[E_L, 16, I/512, D]`` down
            scales when it is 1.
        first_expert: First expert of the group.
        n_experts: Number of experts in the group.
        projections_per_expert: 2 for the fused gate/up scales, 1 for down.
        projections: the projections to load (default: all of each expert).
        first_slot: the slot of the first expert's first projection (default 0); slot
            ``first_slot + projections_per_expert * expert + projection`` holds a projection.
        priority: the DMA QoS level of the loads (trn3; 0 is the highest), or None (default) for no level.
    """
    row_size = dst.shape[1]
    scale_rows = scale.shape[1]
    expert_stride = scale_rows * projections_per_expert * row_size
    if _SCALE_ROWS_ACROSS_QUADRANTS:
        # Row r of the 4 quadrants in one DMA: rows r, 4 + r, 8 + r, 12 + r of the projection go to partition
        # 4 * slot + r of quadrants 0..3, the bytes and the partitions of the per-quadrant DMAs below.
        n_quadrants = _PMAX // _SBUF_QUADRANT_PARTITIONS
        for group_expert in range(n_experts):
            for projection_idx in range(projections_per_expert) if projections is None else projections:
                slot_idx = first_slot + group_expert * projections_per_expert + projection_idx
                first_partition = slot_idx * _SCALE_ROWS_PER_QUADRANT
                for row_idx in range(_SCALE_ROWS_PER_QUADRANT):
                    nisa.dma_copy(
                        dst=dst[first_partition + row_idx : _PMAX : _SBUF_QUADRANT_PARTITIONS, :],
                        src=scale.ap(
                            pattern=[
                                [_SCALE_ROWS_PER_QUADRANT * projections_per_expert * row_size, n_quadrants],
                                [1, row_size],
                            ],
                            offset=(first_expert + group_expert) * expert_stride
                            + (projections_per_expert * row_idx + projection_idx) * row_size,
                        ),
                        **_scale_rows_dma_options(),
                        **({"priority": priority} if priority is not None else {}),
                    )
        return
    for quadrant_idx in range(_PMAX // _SBUF_QUADRANT_PARTITIONS):
        for group_expert in range(n_experts):
            for projection_idx in range(projections_per_expert) if projections is None else projections:
                slot_idx = first_slot + group_expert * projections_per_expert + projection_idx
                nisa.dma_copy(
                    dst=dst[
                        nl.ds(
                            quadrant_idx * _SBUF_QUADRANT_PARTITIONS + slot_idx * _SCALE_ROWS_PER_QUADRANT,
                            _SCALE_ROWS_PER_QUADRANT,
                        ),
                        :,
                    ],
                    src=scale.ap(
                        pattern=[[projections_per_expert * row_size, _SCALE_ROWS_PER_QUADRANT], [1, row_size]],
                        offset=(first_expert + group_expert) * expert_stride
                        + quadrant_idx * _SCALE_ROWS_PER_QUADRANT * projections_per_expert * row_size
                        + projection_idx * row_size,
                    ),
                    **_scale_rows_dma_options(),
                    **({"priority": priority} if priority is not None else {}),
                )


def _project_chunk(
    input_quant_sb: nl.NkiTensor,
    input_scale_sb: nl.NkiTensor,
    weight_sb: nl.NkiTensor,
    weight_scale_sb: nl.NkiTensor,
    n_H512: int,
    chunk_idx: int,
    chunk_size: int,
    T: int,
    weight_first_chunk: int = 0,
    matmuls: list = None,
) -> nl.NkiTensor:
    """Run one chunk of an MX gate/up projection with the weights as the moving operand.

    The quantized latent is the stationary operand, so each matmul streams
    up to 512 weight columns into one FP32 PSUM bank. For MX matmuls a
    weight-stationary load of 128 columns takes about as long as streaming
    them as the moving operand, because the scales load through the same
    SBUF ports. The PE di/dt throttle does not depend on the operand form:
    it starts after TensorE is idle for a short time and lowers its rate
    for the work that follows, so keep TensorE busy.

    The caller runs the SiTU of each chunk right after its matmuls, so one
    chunk's PSUM bank is live at a time instead of the whole projection's.
    ``matmuls`` (a list), when given, gets the chunk's matmul instructions in
    order, for schedule edges.

    Returns:
        The FP32 PSUM chunk ``[T, <=512]``, accumulated over the H/512 tiles.
    """
    chunk_psum = nl.ndarray((T, chunk_size), dtype=nl.float32, buffer=nl.psum)
    I_slice = nl.ds(chunk_idx * _GATE_UP_CHUNK_I, chunk_size)
    weight_I_slice = nl.ds((chunk_idx - weight_first_chunk) * _GATE_UP_CHUNK_I, chunk_size)
    for tile_h in range(n_H512):
        matmul = nisa.nc_matmul_mx(
            dst=chunk_psum,
            stationary=input_quant_sb[:, tile_h, :],
            moving=weight_sb[:, tile_h, weight_I_slice],
            stationary_scale=input_scale_sb[:, tile_h, :],
            moving_scale=weight_scale_sb[:, tile_h, I_slice],
        )
        if matmuls is not None:
            matmuls.append(matmul)
    return chunk_psum


class _ActivationChunk:
    """One 1024-column activation chunk of ``_transpose_and_quantize_chunk``, issued tile by tile, so that other
    instructions can be traced between its tiles: ``transpose_tile`` issues one I/512 tile's transposes, ``finish``
    the ScalarE eviction and the MXFP8 quantization of the chunk. The instructions, their operands and the result are
    those of ``_transpose_and_quantize_chunk``; ``evictions``, when given, holds the eviction's FP16 tile (the next of
    its two buffers), as there."""

    def __init__(
        self,
        activation_token_major: nl.NkiTensor,
        chunk_idx: int,
        I: int,
        T: int,
        evictions: "_StaticLoopTiles" = None,
    ):
        tiles_per_chunk = _TRANSPOSE_CHUNK_I // MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
        self.activation_token_major, self.T = activation_token_major, T
        self.first_tile = chunk_idx * tiles_per_chunk
        self.n_tiles = min(tiles_per_chunk, I // MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM - self.first_tile)
        if evictions is None:
            self.activation_f16 = nl.ndarray((_PMAX, self.n_tiles, T, _q_width), dtype=nl.float16, buffer=nl.sbuf)
        else:
            self.activation_f16 = evictions.eviction(self.n_tiles)
        self.transpose_psum = nl.ndarray((_PMAX, self.n_tiles, _q_width, T), dtype=nl.float16, buffer=nl.psum)

    def transpose_tile(self, chunk_tile: int, after=None) -> list:
        """Issue the transposes of tile ``chunk_tile`` of the chunk; each follows ``after`` (a TensorE instruction) on
        TensorE when given. Return them."""
        tile_i = self.first_tile + chunk_tile
        transposes = []
        for q_idx in range(_q_width):
            transpose = nisa.nc_transpose(
                dst=self.transpose_psum[:, chunk_tile, q_idx, :],
                data=self.activation_token_major[
                    :, nl.ds(tile_i * MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM + q_idx * _PMAX, _PMAX)
                ],
                engine=nisa.engine.tensor,
            )
            if after is not None:
                _after(transpose, after)
            transposes.append(transpose)
        return transposes

    def finish(self) -> tuple:
        """Evict (ScalarE) and quantize the chunk after its last tile; return its MXFP8 tiles and scales, and the
        eviction."""
        n_tiles, T = self.n_tiles, self.T
        eviction = nisa.tensor_copy(
            dst=self.activation_f16,
            src=self.transpose_psum.ap(
                pattern=[[n_tiles * _q_width * T, _PMAX], [_q_width * T, n_tiles], [1, T], [T, _q_width]],
            ),
            engine=nisa.scalar_engine,
        )
        activation_quant_sb = nl.ndarray((_PMAX, n_tiles, T), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
        activation_scale_sb = nl.ndarray((_PMAX, n_tiles, T), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.quantize_mx(src=self.activation_f16, dst=activation_quant_sb, dst_scale=activation_scale_sb)
        return (activation_quant_sb, activation_scale_sb), eviction


def _transpose_and_quantize_chunk(
    activation_token_major: nl.NkiTensor,
    chunk_idx: int,
    I: int,
    T: int,
    after=None,
    issued: dict = None,
    evictions: "_StaticLoopTiles" = None,
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """Move one 1024-column chunk of the SiTU output to the down projection's MX layout and quantize it.

    Element ``[p, i, t, q]`` of the transposed chunk holds intermediate column
    ``chunk_idx * 1024 + 512 * i + 128 * q + p`` of token ``t``. With ``after``
    (a TensorE instruction), every transpose of the chunk follows it on TensorE
    (a schedule edge, no semaphore wait). ``issued`` (a dict), when given,
    gets the transposes (``"transposes"``) and the ScalarE eviction
    (``"eviction"``), for schedule edges. ``evictions``, when given, holds the
    eviction's FP16 tile (the next of its two buffers); else the tile is new.

    Returns:
        The MXFP8 activation tiles ``[128, <=2, T]`` of the chunk and their scales.
    """
    tiles_per_chunk = _TRANSPOSE_CHUNK_I // MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
    first_tile = chunk_idx * tiles_per_chunk
    n_tiles = min(tiles_per_chunk, I // MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM - first_tile)
    if evictions is None:
        activation_f16 = nl.ndarray((_PMAX, n_tiles, T, _q_width), dtype=nl.float16, buffer=nl.sbuf)
    else:
        activation_f16 = evictions.eviction(n_tiles)
    transpose_psum = nl.ndarray((_PMAX, n_tiles, _q_width, T), dtype=nl.float16, buffer=nl.psum)
    for chunk_tile in range(n_tiles):
        tile_i = first_tile + chunk_tile
        for q_idx in range(_q_width):
            transpose = nisa.nc_transpose(
                dst=transpose_psum[:, chunk_tile, q_idx, :],
                data=activation_token_major[
                    :, nl.ds(tile_i * MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM + q_idx * _PMAX, _PMAX)
                ],
                engine=nisa.engine.tensor,
            )
            if after is not None:
                _instruction(transpose).depends_on(_instruction(after), kind=nisa.dependency.schedule)
            if issued is not None:
                issued.setdefault("transposes", []).append(transpose)
    # ScalarE evicts the transposes: VectorE also quantizes and runs the SiTU products.
    eviction = nisa.tensor_copy(
        dst=activation_f16,
        src=transpose_psum.ap(
            pattern=[[n_tiles * _q_width * T, _PMAX], [_q_width * T, n_tiles], [1, T], [T, _q_width]],
        ),
        engine=nisa.scalar_engine,
    )
    if issued is not None:
        issued["eviction"] = eviction
    activation_quant_sb = nl.ndarray((_PMAX, n_tiles, T), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
    activation_scale_sb = nl.ndarray((_PMAX, n_tiles, T), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.quantize_mx(src=activation_f16, dst=activation_quant_sb, dst_scale=activation_scale_sb)
    return activation_quant_sb, activation_scale_sb
