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

"""Packed selective-load MoE expert MLP for Trn3 decode, tuned for the kimi shape (H = 3584, I = 24, E = 896, K = 16).

The docstring of moe_tkg_selective_load_packed explains the algorithm, the layouts and the design choices.

Kimi-K3 additions to the body (off by default): the SiTU activation
(``situ``), an output view with a row offset and a valid row count (``output``, ``output_row``, ``valid_tokens``),
two phases (``selective_load_packed_phases``), caller-built gather indices and down-scale buffer
(``gather_index_in``, ``down_s_in``), a Pool stream chain across calls (``stream_chain``), a weight budget for the
ring depths (``weight_sbuf_budget``), buffers shared with earlier calls (``buffer_cache``), fewer schedule edges
(``sparse_schedule``) and post_down for any T and K powers of two (``affinity_scaling``).
"""

import inspect
from typing import Callable, Generator, Optional, Sequence, Tuple

import nki
import nki.isa as nisa
import nki.language as nl

from ...utils.allocator import SbufManager
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import div_ceil
from ...utils.logging import get_logger
from ...utils.trace_names import next_call, restart_calls

_ISA, _LANGUAGE = nisa, nl


class _Suffixed:
    """A view of a module whose functions (or the listed ones) append ``suffix`` to their ``name`` argument."""

    def __init__(self, module, suffix: str, functions: tuple = ()):
        self._module, self._suffix, self._functions = module, suffix, functions

    def __getattr__(self, attr):
        value = getattr(self._module, attr)
        if attr in self._functions or (not self._functions and inspect.isfunction(value)):

            def call(*args, **kwargs):
                if kwargs.get("name") != None:
                    kwargs["name"] = kwargs["name"] + self._suffix
                return value(*args, **kwargs)

            return call
        return value


def _instruction(handle) -> nisa.NkiInstruction:
    """Return handle as an instruction that takes dependency edges (some ISA calls return a raw operation)."""
    return handle if hasattr(handle, "depends_on") else _ISA.NkiInstruction(handle)


_P_MAX = 128
_QUADRANT = 32
_X4 = 4
_MX_ROWS = 8
_BLOCKS = 4  # H512 blocks per gate/up scale chunk
_GROUP = 16  # routed pairs per group
_SCALE_GROUPS = 4  # groups per down-scale tile (an MX scale tile starts at partition 0, 4, 8 or 12 of a quadrant)
_SKIP = 1 << 20  # gather index offset that makes a row out of bounds, so the gather skips it
_PSUM_COLS = 512  # half of a bf16 gate/up PSUM bank: gate at column 0, up at column _PSUM_COLS
_DOWN_CHUNK = 512  # H columns per down-projection matmul (moving down weights)
_DOWN_AFTER_SIXTEENTHS = 13  # down(g) runs inside gate/up(g + 1) after 13/16 of its H512 blocks
_LAST_FINISH_CHUNKS_LEFT = 3  # down chunks of the second-to-last group left after the last group's finish
# long_stream (more than 4 groups, U = 1): one more gate/up group comes from per-expert HWDGE loads on SP (group 3 in
# the standalone kernel with pre_quant, else group 2), ACT copies the first H512 blocks of each re-layout of groups
# 1.., and with the inputs in SBUF finish(g) runs after this many gate/up units of group g + 1.
_LONG_HWDGE_GROUP = 2
_LONG_HWDGE_GROUP_STANDALONE = 3
_LONG_ACT_HWDGE_LOADS = 2  # inputs in SBUF: the last loads of the HWDGE group go on ACT, after the group-0 re-layout
_LONG_ACT_RELAYOUT_BLOCKS = 4
_LONG_FINISH_AFTER_UNITS = 2
_OUT_PSUM_COLS = 4096  # bf16 columns of out_ps (4 PSUM banks); larger H is split into halves on more partitions
_OUT_PSUM_COLS_FP32 = 2048  # fp32 columns of out_ps in banks 2-5 (down_psum_dtype="fp32")
_OUT_PSUM_COLS_FP32_WIDE = 3072  # fp32 columns of out_ps in banks 2-7 (the gated values move into the gate/up tiles)
_OUT_PSUM_COLS_TWO_BANKS = 2048  # bf16 columns of out_ps in banks 4-5 (fp32 gate/up tiles of two banks each)
# Per-partition SBUF bytes for the gate/up rows, the moving re-layout rings and the down weights.
_WEIGHT_SBUF_BUDGET = 208 * 1024
# Sub-groups of 8 pairs from this many H512 blocks on, if the packed rows are at least this long. An 8-expert Pool
# gather has a fixed descriptor cost, so shorter rows (H = 6144, I = 16: 896 B) would make Pool slower than the DMA.
_SUB_GROUP_MIN_H512 = 9
_SUB_GROUP_MIN_ROW_BYTES = 2048
_PSUM_BANK_BYTES = 2048
# int16 0xFFF0: an AND with the bf16 bits keeps the sign, the exponent and the top 3 mantissa bits, so the result
# (4 significant bits) is exactly representable in MXFP8 e4m3.
_E4M3_BF16_MASK = -16


@nki.jit
def moe_tkg_selective_load_packed(
    hidden_qtz: nl.NkiTensor,
    hidden_scale: nl.NkiTensor,
    expert_gate_up_packed: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_down_weights_scale: nl.NkiTensor,
    expert_index: nl.NkiTensor,
    expert_affinities: nl.NkiTensor,
    affinity_scaling: str = "pre_quant",
    down_psum_dtype: str = "bf16",
    gate_up_psum_dtype: str = "bf16",
) -> nl.NkiTensor:
    """Selective-expert MoE MLP (SiLU(gate) * up, no bias) that packs 16 routed experts into each matmul.

    Optimized for decode on Trn3 (LNC1): T*K = 64 or 128 routed pairs (16 to 48 for a short call), a small
    intermediate size and MXFP4 weights.
    This copy is tuned and tested for T = 8, K = 16, H = 3584, I = 24, E = 896 (supported shapes in Notes). The kernel
    loads only the routed experts and keeps the full [T, H] output in PSUM until it stores it.

    Dimensions:
        T: tokens. K: experts per token. E: experts. H: hidden size. I: intermediate size of one expert.
        N = T*K: routed pairs (64 or 128). n = t*K + k: the routed pair of token t and its k-th expert, 0 <= n < N.
        g = n // 16: the group of pair n (N/16 groups: 4 or 8). c = n % 16: the column of pair n in its group.
        hb: H512 block, 0 <= hb < H/512. q: x4 lane, 0..3. p: row in an 8-row MX block. a, m: SBUF partition.
        i = q*(I/4) + p: the intermediate index that lane q of row p holds.
        Hh: output halves. Hh = 1 for H <= 4096 with bf16 PSUM, else 2 (fp32 PSUM: 2 for H > 2048).
            out_cols = H / Hh.
        R, Rm: slots of the gate/up row ring and of the moving-weight ring (Step 1.2).
        S: 4*I-byte scale segments per packed row (1 for H <= 4096, else 2).
        U, M: sub-groups per group and pairs per sub-group. U = 2, M = 8 for H > 4096 with packed rows of at least
            2 KB (H = 8192, I = 32), else U = 1, M = 16. Sub-group j = 0 .. N/M - 1 holds pairs Mj .. Mj + M - 1.
        Index arithmetic uses integer division: a/8 is a // 8.

    Args:
        hidden_qtz (nl.NkiTensor): float8_e4m3fn_x4 [128, H/512, T] @HBM, the RMSNorm-MX output. Element [p, hb, t]
            packs the 4 values of H = q*H/4 + 128*hb + p, q = 0..3.
        hidden_scale (nl.NkiTensor): uint8 [128, H/512, T] @HBM, the MX scales of hidden_qtz in the quadrant layout:
            scale row r (partitions 8r..8r+7) is partition 32*(r/4) + r%4.
        expert_gate_up_packed (nl.NkiTensor): uint8 [E, 128, 4*I*(H/512 + S)] @HBM, the gate/up weights of each
            expert with their MX scales packed into the same rows (see Layouts).
        expert_down_weights (nl.NkiTensor): float4_e2m1fn_x4 [E, 8, H] @HBM, the 8 MX rows of each expert (I padded to
            32 = 8 rows x 4 lanes). Row p, lane q holds i = q*(I/4) + p; rows p >= I/4 are not read. The compact form
            [E, I/4, H] holds only the rows that the kernel reads. It gives the same SBUF bytes and the same output.
        expert_down_weights_scale (nl.NkiTensor): uint8 [E, H] @HBM, the MX scale row of each expert (one scale per
            8 rows x 4 lanes).
        expert_index (nl.NkiTensor): int32 [T, K] @HBM, the routed experts of each token.
        expert_affinities (nl.NkiTensor): float32 [T, E] @HBM, zero for experts that are not routed.
        affinity_scaling (str): where the POST_SCALE affinity is applied (see selective_load_packed_body):
            "pre_quant" (default): the affinity scales SiLU(gate) * up before its MXFP8 quantization, and the down
            projection adds the K experts of a token in its contraction.
            "post_down": SiLU(gate) * up is quantized unscaled, the down projection gives one output row per routed
            pair, the affinity scales each row when it leaves PSUM, and a matmul adds the K rows of a token (the
            reference's order of operations).
        down_psum_dtype (str): the PSUM dtype of the down-projection output, "bf16" (default) or "fp32" (see
            selective_load_packed_body).
        gate_up_psum_dtype (str): the PSUM dtype of the gate and up projections, "bf16" (default) or "fp32" (see
            selective_load_packed_body). At H = 3584 with I = 24, "fp32" asserts for the same reason.

    Returns:
        output (nl.NkiTensor): bfloat16 [T, H] @HBM.

    Notes:
        - POST_SCALE affinities. With pre_quant, the kernel multiplies SiLU(gate) * up by the affinity, rounds the
          product to fp16 and then quantizes it to MXFP8. This changes the MXFP8 rounding relative to a scale after
          the down projection. The fused MoE block carries the product as two MXFP8 terms, hi + lo (about 8
          significant bits), which keeps it close to the reference (selective_load_packed_body, affinity_hi_lo).
        - Requires LNC1, T*K == 16, 32, 48, 64 or 128, 16 % K == 0, H % 512 == 0, 2048 <= H <= 8192, I % 4 == 0,
          I <= 32 and E % 128 == 0. post_down needs T and K powers of two. The expert ids go through fp16 PE
          broadcasts, which are exact for E <= 2048.
        - selective_load_packed_body holds the code. The fused MoE block calls it with SBUF inputs.

    Algorithm:
        The N pairs form N/16 groups of 16 (16/K tokens per group; one token per group at K = 16). The group is the
        unit of work: the gate (or up) columns of 16 experts fill one moving tile, and 16 experts x 8 MX rows fill
        the 128 partitions of one down-projection stationary. The code comments in selective_load_packed_body use
        the step numbers below.

        Step 1: Plan the buffers.
            Step 1.1: Get the sizes from the tensor shapes and check the constraints.
            Step 1.2: Choose the ring depths. R = N/16 (one gate/up row slot per group) when it fits the weight
                budget, else the largest R >= 2 that fits. Rm = 3 when it fits with R = N/16, else 2. H = 8192 uses
                R = Rm = 2. The post_down buffers reduce the budget.
            Step 1.3: Allocate the SBUF buffers. The weights come first, so their DMA writes stay aligned.
            Step 1.4: Allocate the PSUM tiles at fixed banks.
            Step 1.5: post_down: zero the down-projection stationary and its scales once (Step 6.4 writes only the
                pair-slot columns). Else, for Hh = 2, zero qmx_in once (the first token block of each term stays
                zero).
        Step 2: Load the routing, the hidden and the group-0 weights.
            Step 2.1: Fused block with router tiles only: broadcast the index and the affinities to every partition
                with a block-diagonal moving tile and a ones stationary (index first).
            Step 2.2: Standalone only: load the index as one row on partition 0 (SP queue), then the hidden and its
                scales (ACT queue; with U = 2 the SP queue, ahead of the expert loads). With U = 2 ACT also zeroes
                column 0 of gated while it waits for the index: this loads its activation table before its expert
                loads.
            Step 2.3: Sub-group 0: read each expert id from the index row and load the expert's 128 packed rows
                (weights and scales) with one HWDGE DMA. The M loads alternate the SP and ACT queues.
            Step 2.4: Standalone only: load the affinities, and broadcast the index row to every partition with a
                ones matmul. DVE copies the broadcast index before its other early work.
        Step 3: Build the gather indices and the constants.
            Step 3.1: Get the bit fields of the partition index a.
            Step 3.2: Get the pair selectors (the pair that feeds partition a in each gather).
            Step 3.3: Get the source rows inside the expert (lanes). Rows that do not exist get +_SKIP.
            Step 3.4: Build the constants: the block-diagonal mask, the zero rows of gated, the expert ids, the
                ones stationary and the token rows (post_down: the pair-slot masks and the token-sum stationary).
            Step 3.5: Gather the expert of each selected pair from the index, then form gather_index =
                rows per expert * expert + source row.
        Step 4: Start the Pool weight stream. The batched SWDGE gathers run in PE consumption order: gate/up g1,
            down scales (groups 0-3), down g0, gate/up g2, down g1, ..., gate/up g(last), then the remaining down
            weights. The scale tile of groups 4-7 goes right before down g4. With U = 2 every gate/up
            gather is split into its 2 sub-groups, and the second sub-group of g0 goes first. A gather into a reused
            gate/up row slot waits for the re-layout of the slot's previous group (Step 6.1). All other gathers start
            here.
        Step 5: Build the per-pair affinities while the PE is idle.
            Step 5.1: Standalone only: transpose the affinities to [expert on partitions, t]. Keep column n where
                index[n] is the expert of the partition (one-hot), and sum over the experts with a ones matmul.
                The result is the affinity of pair n on every partition.
            Step 5.2: Keep the affinity of pair n only on its own 8 rows (affinity_mask). Sum it to one value per
                row and group (affinity_rows), then spread it over the tokens (row_weights). post_down: get the
                affinity of the pair of each output row (aff_rows) instead.
        Step 6: Define the per-group stages. Step 7 calls them.
            Step 6.1: relayout(g, runs): DVE copies the gate/up weights of the pair runs `runs` of group g from the
                packed rows to the moving layout. ACT copies their scale segments. After the last run, the stream
                gathers that reuse the row slot of g start.
            Step 6.2: gate_up_projection(g, units): one gate and one up matmul per (sub-group, H512 block hb) unit.
                Stationary: block hb of all T tokens [128, T]. Moving: the M*I gate (or up) columns of the
                sub-group for block hb. PSUM accumulates over the blocks.
            Step 6.3: swiglu(g): ACT computes SiLU(gate). DVE multiplies it by up and writes gated[t, (q, c, p)].
            Step 6.4: finish(g): 4 PE transposes (one per lane q) put the I values of each pair on its own 8
                partitions. DVE keeps the token column of each row, multiplies it by the affinity of the pair
                (row_weights) and quantizes it to MXFP8: the down-projection stationary [128 = (c, p), (j, t)]. With
                hi / lo (fused block) DVE first splits the product v into hi = v with 4 significant bits (exact in
                e4m3: an AND of the bf16 bits) and lo = v - hi (exact in bf16), and the stationary holds both terms.
                post_down: DVE puts the unscaled value of each row into the column of its pair slot (k, t) and
                quantizes the N pair-slot columns.
            Step 6.5: down_projection(g, chunks): the stationary against the down weights of the group
                [128 = (c, p), 512 h] for each 512-column chunk of H. The contraction adds the K experts of each
                token, and PSUM accumulation adds the groups. For the last group, DVE and ACT evict each chunk
                as soon as it is done (ACT takes every third chunk, never the last). Without hi / lo the SP queue
                stores each chunk; with hi / lo a DMA adds the hi and lo rows while it stores the chunk
                (dma_compute). post_down: half h of pair t*K + k lands on output row h*N + T*k + t. The eviction of
                the last group scales each row by its affinity, and one chunk later a bf16 matmul with the 0/1
                token_sum stationary adds the K rows of each token before the store.
        Step 7: Run the PE software pipeline. Each PE instruction is chained after the previous one (the 4
            transposes of finish(g) through the first one).
            Step 7.1: Re-lay groups 0 (in two parts) and 1, then run gate/up and SwiGLU of group 0.
            Step 7.2: For g = 0 .. N/16 - 2: re-lay group g + 2 (if it exists), run the first half of gate/up(g + 1),
                finish(g), the rest of gate/up(g + 1) and its SwiGLU. With U = 1, all of gate/up(last) runs before
                finish(second last), so the PE does not wait for SwiGLU(second last). down(0) follows gate/up(1).
                down(g) for 0 < g < N/16 - 2 runs inside gate/up(g + 1) after 13/16 of its units. down(N/16 - 2)
                waits for Step 7.3. With U = 2 the rows of a
                group may still be in flight when DVE / ACT reach them, so DVE re-lays the first sub-group of
                group g + 2 after finish(g), and the second sub-group of group g + 1 after SwiGLU(g) (ACT and DVE
                run in order).
            Step 7.3: Run down(second last) with finish(last) before its last 3 chunks, then down(last) with the
                output stores.
            Step 7.4: Chain the Pool stream gathers in program order.

        PE order for 4 groups; 8 groups repeat the middle stage (gate_up(g)[a:b] = (sub-group, H512 block) units
        a..b-1, h = half of the units; with U = 2 the units are all blocks of sub-group 0, then all blocks of
        sub-group 1):
            gate_up(0) | gate_up(1)[:h] finish(0) gate_up(1)[h:] down(0)
            | gate_up(2)[:h] finish(1) gate_up(2)[h:13/16] down(1) gate_up(2)[13/16:]
            | gate_up(3) finish(2) down(2)[first chunks] finish(3) down(2)[last 3] down(3)
            (U = 2: gate_up(3)[:h] finish(2) gate_up(3)[h:] down(2)[first chunks] ...)

    Layouts:
        HBM:
        - expert_gate_up_packed[e, p, :]: the float4_e2m1fn_x4 gate/up row [gate_or_up 2][hb H/512][I]
          (4*I*H/512 bytes), then S scale segments of 4*I bytes. Column chunk k = gate_or_up * n_chunks + hb/4
          covers 4 H512 blocks (n_chunks = ceil(H/2048)). Its scales are in segment k/4: partition
          32q + 4(k%4) + j holds scale row 4q + j, at columns [hb%4][I]. An MX scale tile must start at partition
          offset 0, 4, 8 or 12 of a quadrant, so a segment holds 4 chunks. The other 64 partitions are zero.
        SBUF:
        - gate_up_w uint16 [128, R * 16 * row_bytes/2]: ring of R group slots of packed rows, as loaded. Pair c of
          group g is at bytes ((g%R) * 16 + c) * row_bytes. One uint16 is one float4_e2m1fn_x4 weight. The uint8
          view is the DMA destination.
        - gate_up_moving uint16 [128, Rm * 16 * 2 * (H/512) * I]: ring of Rm slots of moving weights, group g at
          column (g%Rm) * 16 * 2 * (H/512) * I, laid out [gate_or_up][hb][c][I]. The gate (or up) columns of block
          hb for the 16 experts are one contiguous run of 16*I columns: one moving tile.
        - gate_up_moving_s uint16 [128, Rm * S * 4 * 16 * I/2]: ring of Rm slots of moving scales, group g at byte
          (g%Rm) * S * 4 * 16 * I, laid out [segment][hb%4][c][I] on the HBM partitions.
        - down_w float4_e2m1fn_x4 [128, (N/16) * H]: group g at columns g*H. Partition 8c + p holds row p of the expert
          of pair c.
        - down_s uint8 [128, St * H] (St = N/64 scale tiles of 4 groups): in tile s = g/4, partition 32q + 4(g%4) + j
          holds the scale row of pair c = 4q + j of group g, so the scale tile of group g starts at partition 4(g%4).
          An MX scale tile must start at partition offset 0, 4, 8 or 12 of a quadrant, so a tile holds 4 groups.
          With two scale groups (down_group_mask) each tile has 2H columns: scale group s at columns s*H of the tile.
        - hidden_raw / hidden_raw_s [128, (H/512) * T']: hidden_qtz / hidden_scale, column hb*T' + t (T' = T rounded
          up to even). Block hb of all tokens is the contiguous gate/up stationary [:, hb*T' : (hb + 1)*T'].
        - index_row int32 [1, N]: expert_index as one row on partition 0 (column n = pair n).
        - index int32 [128, N]: index[a, n] = the expert of pair n, on every partition a.
        - partition int32 [128, 7]: a, a%8, a%16, a/8, a/16, the down-scale pair, 4*(a/32) (Step 3.1).
        - select / experts int32 [128, G + N/16 + St] (G = N/M - 1 gathered gate/up sub-groups): the pair that
          feeds partition a / its expert. Column j - 1: gate/up of sub-group j = 1..G. Columns G .. G + N/16 - 1:
          down weights of group g. Columns G + N/16 + s: down-scale tile s.
        - lanes int32 [128, 18]: the source row inside the expert. Columns 0..M-1: gate/up row for index column b.
          Column 16: down-weight row. Column 17: down-scale row.
        - gather_index int32 [128, G*M + N/16 + St]: columns (j - 1)M + b: gate/up of sub-group j. Column G*M + g:
          down weights of group g. Column G*M + N/16 + s: down-scale tile s. Each index is rows per expert * expert +
          source row.
        - Gather rows: a batched SWDGE gather with an index tile [128, M] uses the index at partition a, column b
          for gathered row r = 128b + a, and writes row r to partition r/M, slot r%M. So the pair that feeds row r
          is a bit-field function of a:
          gate/up of sub-group j (M index columns, 128 rows per expert): row r lands on partition (128/M)b + a/M,
          slot a%M. The pair is Mj + a%M and the source row is (128/M)b + a/M (M = 16: 8b + a/16; M = 8:
          16b + a/8).
          down weights of group g (M = 1, 8 rows per expert): partition a = 8c + p holds row p of pair c = a/8. The
          pair is 16g + a/8 and the source row is a%8. Rows p >= I/4 get +_SKIP.
          down-scale tile s (M = 1, 1 row per expert): partition a = 32q + 4(g%4) + j holds the scale row of pair
          c = 4q + j of group g = 4s + (a/4 % 4) (a%32 < 16 only). The pair is 64s + 16(a/4 % 4) + 4(a/32) + a%4.
          Partitions with a/16 odd get +_SKIP.
        - masks uint16 [128, 16]: masks[m, c] = 1 if 0 <= m - 8c < 8 (block diagonal).
        - token_rows uint16 [128, (N/16) * T]: [m, (g, t)] = 1 if pair 16g + m/8 belongs to token t.
        - expert_id fp32 [128, E/128]: [a, h] = a + 128h. affinity_hot bf16 [128, (E/128) * N]: tile h, column n
          = the affinity of pair n if index[n] == a + 128h, else 0. ones bf16 [128, 128]: the stationary that sums
          over the partitions.
        - affinity bf16 [128, N]: the affinity of pair n on every partition. affinity_mask bf16 [128, N]: the
          same, only on rows m = 8(n%16) + p. affinity_rows bf16 [128, N/16]: [m, g] = the affinity of pair
          16g + m/8. row_weights bf16 [128, (N/16) * T] = affinity_rows where token_rows, else 0. Two scale groups:
          row_weights [128, 2 * (N/16) * T], [m, (s, g, t)], is 0 also where row m is not on scale group s.
          All exact in bf16 (0/1 values times the bf16 affinity, one nonzero term per row).
        - gate_act fp32 [T, 2 * 16 * I]: ring of 2 groups, SiLU(gate) of group g at columns (g%2) * 16 * I.
        - gated bf16 [T, 2 * 4 * 128]: ring of 2 groups, group g at columns 512 (g%2), laid out [q][c][p]: SiLU(gate)
          * up of i = q*(I/4) + p. Rows p >= I/4 stay zero.
        - qmx_in [128, (N/16) * Q * 4], bf16 with hi / lo or post_down, else fp16: per group the values to quantize, q
          innermost so quantize_mx packs the 4 lanes. pre_quant: Q = Hh * T per term, [m, (term, j, t, q)] (term 0 =
          hi, 1 = lo with hi / lo). Token block j = Hh - 1 of each term holds the values; block j = 0 of Hh = 2 stays
          zero. With Hh = 2 an odd T has blocks of T + 1 columns (the last column stays zero), so every stationary
          width is even. post_down: Q = N, [m, (k, t, q)]: the value of row m in the slot of its pair t*K + k, else 0.
          Two scale groups: per group the values of scale group 0, then of scale group 1 (Q columns each).
        - inter float8_e4m3fn_x4 / inter_s uint8 [128, (N/16) * Ns]: per group the down-projection stationary (Ns
          columns) and its MX scales. pre_quant: Hh * T columns per term, [m, (term, j, t)]. post_down: Ns = Hh * N,
          zero except the N pair-slot columns [m, (k, t)] at the end of the group. Two scale groups: 2 * Ns columns
          per group, the stationary of scale group 0, then of scale group 1.
        - out_sb bf16 [Ns, out_cols]: the evicted output, rows as in out_ps. post_down: out_tokens bf16
          [Hh * T, out_cols] holds the token sums, token t of output half h on partition h*T + t.
        - post_down constants: slot_mask bf16 [128, (N/16) * N]: [m, (g, k, t)] = 1 if row m of group g is pair
          t*K + k. aff_rows fp32 [128, 1]: the affinity of the pair of output row r. token_sum bf16 [128, Hh * T]:
          [r, h*T + t] = 1 if output row r is h*N + T*k + t for some k.
        PSUM (8 banks of 2 KB per partition):
        - gate_up_ps bf16 [T, 1024], banks 0 and 1: group g in bank g%2, [t, (c, I)]. Gate at column 0, up at 512.
          fp32 changes this plan (see gate_up_psum_dtype in selective_load_packed_body).
        - out_ps bf16 [Ns, out_cols], banks 2-5: the output, accumulated over the groups. pre_quant: output half h
          of token t is row h*T + t (the lo term of hi / lo: row Hh*T + h*T + t, which the store DMA adds to the hi
          row). Half h uses the stationary from token block Hh - 1 - h of the first term to the end of the group, so
          its values land on its own rows and the zero blocks add 0 to the other rows. post_down: half h of pair
          t*K + k is row h*N + T*k + t. The widest half goes first, so group 0 writes every row. fp32 changes this
          plan (see down_psum_dtype in selective_load_packed_body).
        - index_ps fp32 [128, N] at bank 6, affinity_ps bf16 [128, N] at bank 6 + 1 KB and affinity_t_ps fp32
          [128, (E/128) * T] at bank 6 + 1.5 KB. routing_ps fp32 [128, 2N] at bank 6 (router tiles only).
        - gated_ps bf16 [128, 2 * 4 * T'], bank 7 (T' = T rounded up to even): group g at columns 4T' (g%2),
          [(c, p), (q, t)] with lane q at column qT'.
        - sum_ps0 .. sum_ps3 bf16 [Hh * T, 512], banks 0-1 (post_down, after the last gate/up): the token sums of 4
          output chunks in flight.

    Why this design:
        - Weights move, tokens are stationary. A decode step has only a few tokens. The [128, T] stationary loads
          fast, and the matmul time depends on the moving width. One matmul moves the gate (or up) columns of 16
          experts (512 columns for I = 32). The PE computes every token for each expert and the kernel uses only
          row t_n of pair n. The extra rows add no PE time.
        - 16 pairs per group. The 16*I columns fill one moving tile, and 16 x 8 MX rows fill the 128 partitions of
          the down-projection stationary.
        - Scales packed into each expert row. One DMA per expert row brings the weights and their MX scales. The
          scales are already at the partition offsets of an MX scale tile, so there is no scale DMA and no scale
          move across partitions.
        - HWDGE for group 0, SWDGE gathers for the other groups. A HWDGE load needs only the expert id from the index
          row, so the first experts arrive while DVE builds the gather indices. After that, one Pool gather loads
          the 16 experts of a group, in PE consumption order.
        - Sub-groups for long rows (U = 2). At H = 8192 a group is 4.7 MB of rows, so the PE would wait for the
          whole group before its first matmul. Two sub-groups of 8 experts (8*I = 256 moving columns, the same cost
          per column) let the PE start on half of the group, and only the first 8 experts need HWDGE loads.
        - Affinity before MXFP8 (pre_quant). The down projection adds the K experts of a token in one matmul, so a
          scale per pair after the matmul is not possible. A scale on the stationary rows before the quantization
          removes the affinity eviction, the K reduction and the output transpose. The fused block carries hi + lo:
          the extra stationary columns cost no PE time, and the store DMA adds the two output rows.
        - fp16 expert ids. The PE broadcasts of the index use fp16, which holds integers exactly up to 2048 (bf16:
          up to 256), so E = 896 is exact.
        - bf16 PSUM. It halves the PSUM footprint. Both gate/up tiles, the full [T, 4096] output and the routing
          tiles fit the 8 banks at fixed addresses (an fp32 out_ps alone needs all 8 banks).
        - Chained PE schedule. The PE runs in order and stops at an instruction whose inputs are not ready. The
          chain fixes an order where finish(g) and down(g) run inside gate/up(g + 1), so DVE and ACT prepare the
          next inputs while the PE works.
    """
    restart_calls()
    kernel_assert(
        nl.program_ndim() == 0 or (nl.program_ndim() == 1 and nl.num_programs(0) == 1),
        "selective_load_packed supports LNC1 only",
    )
    kernel_assert(nisa.get_nc_version() >= nisa.nc_version.gen4, "selective_load_packed requires Trn3 or newer")
    sbm = SbufManager(0, nl.tile_size.total_available_sbuf_size, get_logger("selective_load_packed"))
    return selective_load_packed_body(
        sbm,
        hidden_qtz,
        hidden_scale,
        expert_gate_up_packed,
        expert_down_weights,
        expert_down_weights_scale,
        expert_index,
        expert_affinities,
        inputs_in_sbuf=False,
        affinity_scaling=affinity_scaling,
        down_psum_dtype=down_psum_dtype,
        gate_up_psum_dtype=gate_up_psum_dtype,
    )


def _scheduled_after(op: Callable[..., nisa.NkiInstruction], after: nisa.NkiInstruction) -> Callable:
    """Wrap op so that its instruction is scheduled after the instruction after."""

    def scheduled(*args, **kwargs) -> nisa.NkiInstruction:
        instruction = op(*args, **kwargs)
        instruction.depends_on(after, kind=nisa.dependency.schedule)
        return instruction

    return scheduled


def selective_load_packed_body(*args, **kwargs) -> nl.NkiTensor:
    """Run both phases of selective_load_packed_phases (same arguments). Returns the output."""
    phases = selective_load_packed_phases(*args, **kwargs)
    next(phases)
    return finish_phases(phases)


def finish_phases(phases) -> nl.NkiTensor:
    """Run the compute phase of a started selective_load_packed_phases generator. Returns the output."""
    try:
        next(phases)
    except StopIteration as stop:
        return stop.value
    raise RuntimeError("selective_load_packed_phases has more than two phases")


def selective_load_packed_phases(
    sbm: SbufManager,
    hidden_qtz: nl.NkiTensor,
    hidden_scale: nl.NkiTensor,
    expert_gate_up_packed: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_down_weights_scale: nl.NkiTensor,
    expert_index: nl.NkiTensor,
    expert_affinities: nl.NkiTensor,
    inputs_in_sbuf: bool,
    expert_index_tk: Optional[nl.NkiTensor] = None,
    affinity_hi_lo: Optional[bool] = None,
    finish_affinities: Optional[Callable] = None,
    after_norm: Optional[nisa.NkiInstruction] = None,
    situ: Optional[Tuple[float, float]] = None,
    sigmoid_from_tanh: bool = False,
    output: Optional[nl.NkiTensor] = None,
    output_row: int = 0,
    valid_tokens: Optional[int] = None,
    routed_tokens: Optional[int] = None,
    gather_index_in: Optional[nl.NkiTensor] = None,
    down_s_in: Optional[nl.NkiTensor] = None,
    setup_in: Optional[Tuple[nl.NkiTensor, nl.NkiTensor]] = None,
    stream_chain: Optional[list] = None,
    weight_sbuf_budget: Optional[int] = None,
    buffer_cache: Optional[dict] = None,
    sparse_schedule: bool = False,
    group0_gather: bool = False,
    setup_after_stream: bool = False,
    affinity_scaling: str = "pre_quant",
    down_psum_dtype: str = "bf16",
    gate_up_psum_dtype: str = "bf16",
    down_group_mask: Optional[nl.NkiTensor] = None,
    long_stream: bool = False,
    relayout_order: bool = False,
    g0_relayout_engine: str = "dve",
    hooks: Optional[dict] = None,
) -> Generator[None, None, nl.NkiTensor]:
    """Packed expert MLP of the standalone kernel (inputs in HBM) and the fused MoE block (inputs in SBUF).

    A generator of two phases. The first phase (Steps 1-5) needs only the routing: it allocates the buffers, issues
    the group-0 HWDGE loads and the Pool weight stream and builds the affinity rows. The second phase (Steps 6-7)
    reads hidden_qtz and hidden_scale, so a caller can write them between the phases. The generator returns the
    output (selective_load_packed_body runs both phases).

    The step numbers in the comments match the Algorithm section of moe_tkg_selective_load_packed.

    Args:
        sbm (SbufManager): SBUF manager. The body allocates its buffers in its own stack scope. With
            inputs_in_sbuf, the hand-off tiles of the caller must not overlap that scope (the fused block puts them
            on the heap), so the weight buffers start at the stack base as in the standalone kernel.
        hidden_qtz, hidden_scale (nl.NkiTensor): HBM [128, H/512, T] (standalone, loaded here) or SBUF
            [128, (H/512) * T] tiles (fused, written by the RMSNorm-MX subkernel). Both have the same element layout.
        expert_gate_up_packed, expert_down_weights, expert_down_weights_scale (nl.NkiTensor): HBM weights, as in
            moe_tkg_selective_load_packed. expert_down_weights_scale may also be uint8 [E, 2, H]: two scale groups
            per expert row (see down_group_mask).
        expert_index, expert_affinities (nl.NkiTensor): the routing, in one of three forms (pair n = t*K + k):
            - HBM int32 [T, K] and float32 [T, E], zero for experts that are not routed (standalone);
            - SBUF int32/uint32 [T, K] and float32 [T, K] on partitions 0..T-1 (router tiles; Step 2.1 broadcasts
              them to every partition, the index first);
            - SBUF int32 [128, T*K] and bfloat16 [128, T*K]: index[a, n] and affinity[a, n] already on every
              partition a (the body uses them as its index and affinity buffers).
        inputs_in_sbuf (bool): selects HBM or SBUF inputs. In SBUF mode, a 128-row expert_index selects the
            broadcast form.
        expert_index_tk (Optional[nl.NkiTensor]): router tile uint32 [T, K] on partitions 0..T-1 (pair n = t*K + k
            at [t, k]) with the same experts as expert_index. The group-0 HWDGE loads read their expert ids from it,
            so they do not wait for the broadcast of the index. With inputs_in_sbuf, the first Pool gather also gets
            its indices from it (needs K = 8 or 16).
        affinity_hi_lo (Optional[bool]): carry the affinity-scaled intermediate v as two MXFP8 terms, hi = v with 4
            significant bits (exact in e4m3) and lo = v - hi, and add the two output rows in the store DMA. None
            (default) selects hi + lo with inputs_in_sbuf (the fused block) and one term for the standalone kernel.
            One MXFP8 term of a * h rounds differently from the reference's a * MXFP8(h), by up to about 5.9%;
            hi + lo keeps about 8 significant bits. The extra token columns cost no PE time. It does not apply to
            post_down.
        finish_affinities (Optional[Callable]): finish_affinities(after) emits the deferred affinity work of the
            router. The body calls it after the first Pool gathers, with the last gather-index op as `after`, so the
            affinities never delay the expert weights.
        after_norm (Optional[nisa.NkiInstruction]): the constant setup of Steps 1.5-3 (its iotas and memsets) is
            scheduled after this instruction (fused: the end of the RMSNorm), so the setup does not delay the
            RMSNorm.
        situ (Optional[Tuple[float, float]]): (beta, linear_beta): use the SiTU activation
            beta*tanh(g/beta)*sigmoid(g) * linear_beta*tanh(u/linear_beta) in FP32 instead of SiLU(g) * u. None: SiLU.
        sigmoid_from_tanh (bool): with situ, compute sigmoid(g) as (1 + tanh(g/2)) / 2, so that ACT uses only the
            tanh table (no table switch for each group).
        output (Optional[nl.NkiTensor]): bfloat16 HBM destination with H columns. None: a new [T, H] tensor.
        output_row (int): first row of output for token 0.
        valid_tokens (Optional[int]): tokens 0 .. valid_tokens - 1 are stored (default T). The other tokens are
            padding.
        gather_index_in (Optional[nl.NkiTensor]): int32 [128, G*M + N/16 + St] gather indices (layout of gather_index),
            built by the caller. Steps 3.1-3.3 and 3.5 are skipped, so the Pool stream waits for no DVE work of this
            call, and Step 3.4 follows the first gather on Pool.
        down_s_in (Optional[nl.NkiTensor]): uint8 [128, St * H] buffer for the down scales (default: a buffer of this
            call).
        setup_in (Optional[Tuple[nl.NkiTensor, nl.NkiTensor]]): float32 [128, 16] and [128, >= N/16 * T] heap tiles
            for the Step 3.4 iotas (offsets, token_offsets) (default: tiles of this call). The caller gives every
            other call the same tiles, so the iota of a call waits only for the DVE setup reads of the call two
            calls before, not for the reads of another tile at the same bytes (a call of other size has another
            stack layout).
        stream_chain (Optional[list]): Pool stream instructions of earlier calls: the first gather of this call is
            scheduled after the last one, and the last gather of this call is appended (Pool runs in order).
        weight_sbuf_budget (Optional[int]): SBUF bytes per partition for the gate/up rows, the moving rings and the
            down weights (Step 1.2). A smaller budget gives fewer ring slots. None: _WEIGHT_SBUF_BUDGET.
        buffer_cache (Optional[dict]): the SBUF buffers of earlier calls. A buffer with the name, shape, dtype and
            address of an earlier buffer is that tensor, so its DMAs keep one descriptor table entry. None: new
            buffers.
        sparse_schedule (bool): chain only the first gate/up matmul of each run of units to the PE schedule. The
            compiler then orders the gate and up matmuls inside a run. The trace cost of a schedule edge grows with
            the size of the kernel, so fewer edges make a large kernel trace faster.
        group0_gather (bool): load the gate/up rows of group 0 with one Pool gather, as the other groups, instead of
            the 16 HWDGE loads of Step 2.3. Needs gather_index_in with M more gate/up columns, those of group 0
            first.
        setup_after_stream (bool): schedule the Pool iotas of this call after the last gather of stream_chain.
            Pool runs in order, and an iota that the compiler puts ahead of those gathers holds them: it waits for
            the DVE reads of the tile in the call before, which DVE runs late.
        affinity_scaling (str): where the POST_SCALE affinity a of pair n is applied.
            "pre_quant" (default): v = a * SiLU(gate) * up is quantized to MXFP8 (optionally as hi + lo terms, see
            affinity_hi_lo), and the down projection adds the K experts of a token in its contraction, so the output
            needs no reduction. The MXFP8 rounding of a * h differs from the reference's a * MXFP8(h).
            "post_down": h = SiLU(gate) * up is quantized unscaled (the same MXFP8 values as the reference, up to the
            kernel's bf16 rounding of h). The down stationary has one column per pair, so PSUM holds y[n, :] =
            MXFP8(h_n) @ W_n for every pair; the eviction multiplies each row by its affinity (per-partition scale)
            and a bf16 matmul with a 0/1 stationary adds the K rows of each token. Requires T and K powers of two and
            at most 128 / (T*K) output halves; affinity_hi_lo does not apply. Costs: T*K stationary columns per
            output half instead of T, and one more matmul and PSUM eviction per output chunk of the last group.
        down_psum_dtype (str): the dtype of out_ps, the down-projection output that PSUM accumulates over the
            groups.
            "bf16" (default): out_ps uses banks 2-5.
            "fp32": the matmul results stay in fp32 until the eviction. out_ps has twice the bytes: H > 2048 uses
            2 output halves of H / 2 columns, a multiple of 512 (pre_quant: of 256; at H = 3584 the last chunk of a
            half has 256 columns). out_cols <= 2048 keeps out_ps in banks 2-5; 2048 < out_cols <= 3072 uses banks
            2-7, and the gated values move into the gate/up tiles (16 * I + 4 * T <= 512). H <= 6144.
            Each group holds whole tokens (16 % K == 0), so the other groups add exact zeros to the output rows of a
            token. With one term, bf16 PSUM rounds each output value once, so pre_quant gives the same output in both
            dtypes. With hi / lo, bf16 rounds the hi term, the lo term and their sum; fp32 keeps both terms in fp32
            until the store DMA adds them, so each output value rounds once. post_down loses at most 1 bf16 ulp per
            value.
        gate_up_psum_dtype (str): the dtype of gate_up_ps, the gate and up projections that PSUM accumulates over the
            H512 blocks.
            "bf16" (default): each group parity has one bank of [T, 1024] bf16, gate at column 0 and up at 512.
            "fp32": the gate and up sums stay in fp32, so h = SiLU(gate) * up gets only the bf16 rounding of its
            DVE write and its MXFP8 values match the reference's MXFP8(h) more often. For I <= 16 each parity has one
            bank of [T, 512] fp32 (gate at 0, up at 256) and the bank plan is unchanged. For I > 16 each parity has
            two banks of [T, 1024] fp32 (gate in the first, up in the second; banks 0-1 and 2-3), and out_ps moves to
            banks 4-5 with 2 output halves of at most 2048 columns (H <= 4096, H / 2 a multiple of 512, bf16 down
            PSUM only; so I = 24 at H = 3584 asserts).
        down_group_mask (Optional[nl.NkiTensor]): bfloat16 [128, 2] SBUF, needed with down scales [E, 2, H] (and only
            then). Column s is 1 on partition a if the 4 values of down row a % 8 use scale group s, else 0. The two
            scale groups give each down row the scale of its own checkpoint block, so the weights are the checkpoint
            values. The down projection runs once for each group: the stationary of group s holds only its own rows
            (the other rows are 0), with its own MX scales, and both runs add into the same PSUM rows. pre_quant only.
            Costs: twice the down matmuls and the down stationary, and twice the down-scale bytes.
        long_stream (bool): the schedule of a call with more than 4 groups (8 tokens at K = 16) and U = 1. Without
            group0_gather, SP loads the 16 experts of one more group by HWDGE (group 2; group 3 in the standalone
            kernel with pre_quant), so Pool has one gate/up gather less; with the inputs in SBUF, ACT loads the last 2
            of them after its group-0 re-layout copies. ACT copies the first 4 H512 blocks of each re-layout of groups 1.. (Pool descriptor
            windows block DVE, not ACT). The re-layout of group g + 2 runs after finish(g) on DVE and after the
            activation of group g on ACT. With the inputs in SBUF, finish(g) runs after 2 gate/up units of group
            g + 1. down(g) runs after all gate/up units of group g + 1 (not in the standalone kernel with post_down).
            With the inputs in SBUF and pre_quant, the group-0 activation follows the ACT copies of group 1. The
            SBUF bytes, the PSUM accumulation order and the result do not change. No effect with at most 4 groups.
        relayout_order (bool): the re-layout order of long_stream for every call. The DVE copies of relayout(g + 2)
            run after finish(g)'s quantize, and SwiGLU(g + 1) runs after them (its DVE ops after the DVE copies, its ACT
            ops after the ACT copies). Otherwise the compiler may run them after SwiGLU(g + 1), and the PE waits for
            them before gate/up(g + 2). It changes only the order of the same instructions, so the result does not
            change. No effect with long_stream (it has this order).
        g0_relayout_engine (str): the engine of the group-0 weight copies (Step 6.1): "dve" (default) or "act". ACT is
            not blocked by Pool descriptor windows. The copies move uint16 bits, so the result does not change.
        hooks (Optional[dict]): the call boundary of a caller that runs calls back to back (moe_sel_load_tp
            cross_call_order). With hooks["defer_act_evicts"], the ACT evictions of the last group and the stores of
            their chunks are not emitted here: they go to hooks["deferred"] as functions f(after) that emit them
            scheduled after the instruction after. hooks["after_g0"] (or None) is called right after the group-0
            re-layout with its last ACT instruction, so a caller can emit the previous call's deferred evictions there.
            The body records its SBUF buffers in hooks["layout"] (name, shape, dtype, address), so the caller can check
            that two calls have the same buffers before it moves work from one call into the next.

    Returns:
        output (nl.NkiTensor): bfloat16 [T, H] @HBM (output when it is given).
    """
    kernel_assert(
        g0_relayout_engine in ("dve", "act"), f"g0_relayout_engine must be 'dve' or 'act', got {g0_relayout_engine}"
    )
    if hooks != None:
        hooks.update(deferred=[], layout=[])
    # Op names must be unique in a kernel: the second and later body calls of a kernel add a suffix to every name.
    call = next_call("sel_load_body")
    suffix = f"_c{call}" if call else ""
    nisa, nl = _Suffixed(_ISA, suffix), _Suffixed(_LANGUAGE, suffix, functions=("ndarray",))
    # Step 1: Plan the buffers.
    # Step 1.1: Get the sizes from the tensor shapes and check the constraints.
    E, _, row_bytes = expert_gate_up_packed.shape
    H = expert_down_weights.shape[2]
    n_h512 = H // 512
    routing_broadcast = inputs_in_sbuf and expert_index.shape[0] == _P_MAX
    if routing_broadcast:
        T = hidden_qtz.shape[1] // n_h512 if routed_tokens == None else routed_tokens
        K = expert_index.shape[1] // T
    else:
        T, K = expert_index.shape
    n_chunks = div_ceil(n_h512, _BLOCKS)
    n_scale_segments = div_ceil(2 * n_chunks, _X4)  # S: 4 column chunks per 4*I-byte scale segment
    I = row_bytes // (_X4 * (n_h512 + n_scale_segments))
    n_pairs = T * K
    # nc_matmul_mx needs an even stationary width (a start on an even column), so the hidden of an odd T has one more
    # token column in each H512 block (zero, or unused) and the gate/up row of that column is not read.
    stationary_tokens = T + T % 2
    n_groups = n_pairs // _GROUP
    n_scale_tiles = div_ceil(n_groups, _SCALE_GROUPS)  # down-scale tiles of 4 groups each
    down_rows = I // _X4  # valid rows of an expert's 8-row down MX block
    group_cols = _GROUP * I  # gate (or up) columns of a group in the moving layout: 16 experts x I
    n_sub_groups = 2 if n_h512 >= _SUB_GROUP_MIN_H512 and row_bytes >= _SUB_GROUP_MIN_ROW_BYTES else 1  # U
    sub_pairs = _GROUP // n_sub_groups  # M: pairs per sub-group, index columns of its gather
    sub_cols = sub_pairs * I  # moving columns of one gate (or up) matmul
    first_gathered = 0 if group0_gather else 1  # sub-group 0 comes from the HWDGE loads unless group0_gather
    n_gathered = n_groups * n_sub_groups - first_gathered  # G: the sub-groups from Pool gathers
    long = long_stream and n_groups > _SCALE_GROUPS and n_sub_groups == 1 and n_h512 > _LONG_ACT_RELAYOUT_BLOCKS
    # long_stream: the gate/up group that SP loads by HWDGE. Its gather_index columns stay; no Pool gather reads them.
    # With group0_gather all gate/up groups stay on Pool: in a large kernel SP issues other DMAs (output stores, the
    # weight loads of the next layer), and the expert loads behind them stall the call for a long time.
    hwdge_group = None
    if long and first_gathered:
        standalone_pre_quant = not inputs_in_sbuf and affinity_scaling != "post_down"
        hwdge_group = _LONG_HWDGE_GROUP_STANDALONE if standalone_pre_quant else _LONG_HWDGE_GROUP
    early_head = not inputs_in_sbuf and n_sub_groups > 1  # standalone U = 2: hidden on SP, ACT table before its loads
    gate_up_cols = 2 * n_h512 * I  # float4_e2m1fn_x4 elements of one expert's gate/up weight row
    kernel_assert(
        down_psum_dtype in ("bf16", "fp32"), f"down_psum_dtype must be 'bf16' or 'fp32', got {down_psum_dtype}"
    )
    kernel_assert(
        gate_up_psum_dtype in ("bf16", "fp32"),
        f"gate_up_psum_dtype must be 'bf16' or 'fp32', got {gate_up_psum_dtype}",
    )
    down_fp32 = down_psum_dtype == "fp32"
    gate_up_fp32 = gate_up_psum_dtype == "fp32"
    # PSUM banks per gate/up tile (one per group parity): fp32 gate and up sums of 16 * I > 256 columns need two.
    gate_up_banks = 2 if gate_up_fp32 and group_cols > _PSUM_COLS // 2 else 1
    gate_up_row = _PSUM_COLS * gate_up_banks if gate_up_fp32 else 2 * _PSUM_COLS  # elements per partition of a tile
    up_col = gate_up_row // 2  # the up sums start here (the gate sums at column 0)
    out_bank = 2 * gate_up_banks  # first PSUM bank of out_ps
    if gate_up_banks == 2:
        kernel_assert(not down_fp32, "gate_up_psum_dtype='fp32' with I > 16 needs down_psum_dtype='bf16'")
        kernel_assert(
            H <= 2 * _OUT_PSUM_COLS_TWO_BANKS, f"gate_up_psum_dtype='fp32' with I > 16 needs H <= 4096, got H={H}"
        )
        n_out_halves = div_ceil(H, _OUT_PSUM_COLS_TWO_BANKS)  # Hh: out_ps in banks 4-5
    elif down_fp32:
        kernel_assert(
            H <= 2 * _OUT_PSUM_COLS_FP32_WIDE,
            f"down_psum_dtype='fp32' needs H <= {2 * _OUT_PSUM_COLS_FP32_WIDE}, got H={H}",
        )
        n_out_halves = min(2, div_ceil(H, _OUT_PSUM_COLS_FP32))  # Hh
    else:
        n_out_halves = div_ceil(H, _OUT_PSUM_COLS)  # Hh
    out_cols = H // n_out_halves
    # fp32 pre_quant output halves may end on a 256-column chunk (H = 3584: 2 halves of 3 x 512 + 256 columns).
    tail_chunk = down_fp32 and affinity_scaling != "post_down" and gate_up_banks == 1 and out_cols % _DOWN_CHUNK != 0
    if down_fp32 or gate_up_banks == 2:
        kernel_assert(
            out_cols % _DOWN_CHUNK == 0 or (tail_chunk and out_cols % (_DOWN_CHUNK // 2) == 0),
            f"fp32 PSUM with {n_out_halves} output halves needs H / {n_out_halves} to be a multiple of 512 (pre_quant:"
            f" 256), got H={H}",
        )
    # fp32 out_ps of more than 2048 columns: banks 2-7, so the gated values of group g move into gate_up_ps(g % 2)
    wide_out_ps = down_fp32 and out_cols > _OUT_PSUM_COLS_FP32
    if gate_up_fp32:
        kernel_assert(not wide_out_ps, "gate_up_psum_dtype='fp32' with down_psum_dtype='fp32' needs H <= 4096")
    if wide_out_ps:
        kernel_assert(
            group_cols + _X4 * stationary_tokens <= _PSUM_COLS,
            f"down_psum_dtype='fp32' with out_cols > 2048 needs 16 * I + 4 * T <= {_PSUM_COLS}, got I={I}",
        )
    n_down_chunks = div_ceil(out_cols, _DOWN_CHUNK)  # per output half

    def chunk_width(chunk: int) -> int:
        """Columns of down chunk `chunk` of an output half (512, or less for the last chunk of a tail half)."""
        return min(_DOWN_CHUNK, out_cols - chunk * _DOWN_CHUNK)

    kernel_assert(
        affinity_scaling in ("pre_quant", "post_down"),
        f"affinity_scaling must be 'pre_quant' or 'post_down', got {affinity_scaling}",
    )
    post_down = affinity_scaling == "post_down"
    # pre_quant: stationary columns of one output half and term, one per token. nc_matmul_mx needs an even stationary
    # width, and output half 0 starts on block 1, so with 2 halves an odd T gets one more (zero) column per block.
    token_block = T + T % 2 if n_out_halves > 1 else T
    token_cols = n_out_halves * token_block  # token columns of one term (hi or lo) of a group's down stationary
    hi_lo = (inputs_in_sbuf if affinity_hi_lo == None else affinity_hi_lo) and not post_down
    kernel_assert(T % 2 == 0 or hi_lo or post_down, f"An odd T needs hi / lo or post_down, got T={T}")
    kernel_assert(
        not inputs_in_sbuf or hidden_qtz.shape[1] == n_h512 * stationary_tokens,
        f"Expected {n_h512 * stationary_tokens} hidden columns for T={T}, got {hidden_qtz.shape[1]}",
    )
    stationary_cols = (2 if hi_lo else 1) * token_cols  # the hi term, then the lo term (one term without hi / lo)
    # post_down: one stationary column per pair slot (k, t) and output half, so output half h of pair n = t*K + k is
    # partition h*T*K + T*k + t (one row per pair).
    stationary_block = token_block  # stationary columns per output half and term
    if post_down:
        kernel_assert(T & (T - 1) == 0 and K & (K - 1) == 0, f"post_down needs T and K powers of two, got T={T}, K={K}")
        kernel_assert(
            n_out_halves * n_pairs <= _P_MAX,
            f"post_down needs at most {_P_MAX // n_pairs} output halves, got {n_out_halves}",
        )
        stationary_block = n_pairs
        stationary_cols = n_out_halves * n_pairs
    n_expert_tiles = E // _P_MAX  # 128-expert tiles of the affinity one-hot
    kernel_assert(
        n_pairs in (16, 32, 48, 64, 128) and _GROUP % K == 0,
        f"Expected T*K == 16, 32, 48, 64 or 128 and K dividing 16, got T={T}, K={K}",
    )
    kernel_assert(
        H == 512 * n_h512 and _BLOCKS <= n_h512 <= 4 * _BLOCKS and (out_cols % _DOWN_CHUNK == 0 or tail_chunk),
        f"Expected H % 512 == 0 and 2048 <= H <= 8192, got H={H}",
    )
    kernel_assert(I % _X4 == 0 and I <= _QUADRANT, f"Expected I % 4 == 0 and I <= 32, got I={I}")
    kernel_assert(E % _P_MAX == 0, f"Expected E % 128 == 0, got E={E}")
    # HBM rows of each expert's down block: 8 (padded) or I/4 (compact, only the rows that the gather reads).
    down_rows_hbm = expert_down_weights.shape[1]
    kernel_assert(
        tuple(expert_down_weights.shape) in ((E, _MX_ROWS, H), (E, down_rows, H))
        and tuple(expert_down_weights_scale.shape) in ((E, H), (E, 2, H)),
        f"Expected down weights [E, 8, H] or [E, {down_rows}, H] and scales [E, H] or [E, 2, H], got "
        f"{expert_down_weights.shape} and {expert_down_weights_scale.shape}",
    )
    # Down scale groups of each expert row: 1, or 2 (each row on the scale of its own checkpoint block).
    n_scale_sets = 1 if len(expert_down_weights_scale.shape) == 2 else 2
    kernel_assert(
        (n_scale_sets == 2) == (down_group_mask != None),
        "Down scales [E, 2, H] need down_group_mask, and down_group_mask needs down scales [E, 2, H]",
    )
    kernel_assert(
        n_scale_sets == 1 or (not post_down and tuple(down_group_mask.shape) == (_P_MAX, 2)),
        f"Two down scale groups need pre_quant and a [{_P_MAX}, 2] down_group_mask",
    )
    group_stride = n_scale_sets * stationary_cols  # down stationary columns of a group (scale group s at s * Ns)

    # Step 1.2: Choose the ring depths: Rm moving slots (3 if they fit with all 4 row slots), then R row slots.
    moving_slot_bytes = _GROUP * gate_up_cols * 2 + n_scale_segments * _BLOCKS * group_cols
    # post_down buffers per partition: slot_val + slot_mask, qmx_in (bf16) and out_sb. They reduce the weight budget.
    post_down_bytes = n_groups * n_pairs * (4 + 2 + _X4 * 2) + out_cols * 2 if post_down else 0
    weight_budget = (_WEIGHT_SBUF_BUDGET if weight_sbuf_budget == None else weight_sbuf_budget) - post_down_bytes
    n_moving_slots = 2
    if n_groups * _GROUP * row_bytes + 3 * moving_slot_bytes + n_groups * 2 * H <= weight_budget:
        n_moving_slots = 3
    moving_bytes = n_moving_slots * moving_slot_bytes
    n_gate_up_slots = n_groups
    while (
        n_gate_up_slots > 2 and n_gate_up_slots * _GROUP * row_bytes + moving_bytes + n_groups * 2 * H > weight_budget
    ):
        n_gate_up_slots -= 1

    # Step 1.3: Allocate the SBUF buffers (layouts in the kernel docstring).
    sbm.open_scope()

    def alloc(*args, name=None, **kwargs):
        tile = sbm.alloc_stack(*args, name=None if name == None else name + suffix, **kwargs)
        if hooks != None:
            hooks["layout"].append((name, tuple(tile.shape), str(tile.dtype), sbm.stack_curr_addr))
        if buffer_cache == None:
            return tile
        return buffer_cache.setdefault((name, tuple(tile.shape), str(tile.dtype), sbm.stack_curr_addr), tile)

    gate_up_w = alloc((_P_MAX, n_gate_up_slots * _GROUP * row_bytes // 2), nl.uint16, name="gate_up_w")
    gate_up_moving = alloc((_P_MAX, n_moving_slots * _GROUP * gate_up_cols), nl.uint16, name="gate_up_moving")
    gate_up_moving_s = alloc(
        (_P_MAX, n_moving_slots * n_scale_segments * _BLOCKS * group_cols // 2), nl.uint16, name="gate_up_moving_s"
    )
    down_w = alloc((_P_MAX, n_groups * H), expert_down_weights.dtype, name="down_w")
    down_s_cols = n_scale_tiles * n_scale_sets * H
    down_s = alloc((_P_MAX, down_s_cols), nl.uint8, name="down_s") if down_s_in == None else down_s_in
    if inputs_in_sbuf:
        hidden_raw, hidden_raw_s = hidden_qtz, hidden_scale
    else:
        hidden_raw = alloc((_P_MAX, n_h512 * stationary_tokens), hidden_qtz.dtype, name="hidden_raw")
        hidden_raw_s = alloc((_P_MAX, n_h512 * stationary_tokens), nl.uint8, name="hidden_raw_s")
    if routing_broadcast:
        index = expert_index
    else:
        """post_down: the 512 B alignment keeps index away from the bytes of the hidden-scale DMA. A DVE write next to a
        DMA destination waits for that DMA, and the late post_down hidden loads would delay the index copy (and every
        Pool gather)."""
        index = alloc((_P_MAX, n_pairs), nl.int32, name="index", align=512 if post_down else None)
    if inputs_in_sbuf and not routing_broadcast:
        pair_offsets = alloc((T, n_pairs), nl.float32, name="pair_offsets")
        pair_tokens = alloc((T, n_pairs), nl.float32, name="pair_tokens")
        routing_moving = alloc((T, 2 * n_pairs), nl.float16, name="routing_moving")
        ones_tokens = alloc((T, _P_MAX), nl.float16, name="ones_tokens")
    elif not inputs_in_sbuf:
        index_row = alloc((1, n_pairs), nl.int32, name="index_row")
        index_row_f16 = alloc((1, n_pairs), nl.float16, name="index_row_f16")
        ones_row = alloc((1, _P_MAX), nl.float16, name="ones_row")
    partition = alloc((_P_MAX, 7), nl.int32, name="partition")
    select = alloc((_P_MAX, n_gathered + n_groups + n_scale_tiles), nl.int32, name="select")
    experts = alloc((_P_MAX, n_gathered + n_groups + n_scale_tiles), nl.int32, name="experts")
    lanes = alloc((_P_MAX, 18), nl.int32, name="lanes")
    gather_index = alloc((_P_MAX, n_gathered * sub_pairs + n_groups + n_scale_tiles), nl.int32, name="gather_index")
    tile_first = inputs_in_sbuf and expert_index_tk != None  # sub-group 1 indexed from the router tile
    if tile_first:
        tile_a = alloc((T, _P_MAX), nl.int32, name="tile_a")
        tile_t = alloc((T, _P_MAX), nl.int32, name="tile_t")
        tile_rows = alloc((T, _P_MAX), nl.float16, name="tile_rows")
        tile_k = alloc((_P_MAX, K), nl.int32, name="tile_k")
        tile_k_mask = alloc((_P_MAX, K), nl.float32, name="tile_k_mask")
        tile_index_f16 = alloc((T, K), nl.float16, name="tile_index_f16")
        tile_picked = alloc((_P_MAX, K), nl.float32, name="tile_picked")
        tile_expert = alloc((_P_MAX, 1), nl.float32, name="tile_expert")
    if not inputs_in_sbuf:
        affinity_te = alloc((T, E), nl.float32, name="affinity_te")
        expert_id = alloc((_P_MAX, n_expert_tiles), nl.float32, name="expert_id")
        affinity_hot = alloc((_P_MAX, n_expert_tiles * n_pairs), nl.bfloat16, name="affinity_hot")
    ones = alloc((_P_MAX, _P_MAX), nl.bfloat16, name="ones")
    affinity = expert_affinities if routing_broadcast else alloc((_P_MAX, n_pairs), nl.bfloat16, name="affinity")
    affinity_mask = alloc((_P_MAX, n_pairs), nl.bfloat16, name="affinity_mask")  # exact in bf16: bf16 affinity x 0/1
    if setup_in == None:
        token_offsets = alloc((_P_MAX, n_groups * T), nl.float32, name="token_offsets")
    token_rows = alloc((_P_MAX, n_groups * T), nl.uint16, name="token_rows")  # 0/1 select predicate
    if setup_in == None:
        offsets = alloc((_P_MAX, _GROUP), nl.float32, name="offsets")
    else:
        offsets, token_offsets = setup_in[0], setup_in[1][:, 0 : n_groups * T]
    masks = alloc((_P_MAX, _GROUP), nl.uint16, name="masks")  # 0/1 select predicate
    # exact in bf16: one bf16 affinity per row
    affinity_rows = alloc((_P_MAX, n_groups), nl.bfloat16, name="affinity_rows")
    # exact in bf16: 0/1 x bf16 affinity
    row_weights = alloc((_P_MAX, n_scale_sets * n_groups * T), nl.bfloat16, name="row_weights")
    if n_scale_sets == 2:
        scale_set_rows = alloc((_P_MAX, 2 * n_groups * T), nl.uint16, name="scale_set_rows")  # 0/1 select predicate
    gate_act = alloc((T, 2 * group_cols), nl.float32, name="gate_act")
    if situ != None:
        situ_sigmoid = alloc((T, 2 * group_cols), nl.float32, name="situ_sigmoid")
        situ_up = alloc((T, 2 * group_cols), nl.float32, name="situ_up")
    gated = alloc((T, 2 * _X4 * _P_MAX), nl.bfloat16, name="gated")
    qmx_in_dtype = nl.bfloat16 if hi_lo or post_down else nl.float16
    qmx_in_cols = n_pairs if post_down else stationary_cols  # post_down quantizes only the pair-slot block
    qmx_in_row = n_groups * n_scale_sets * qmx_in_cols * _X4
    qmx_in = alloc((_P_MAX, qmx_in_row), qmx_in_dtype, name="qmx_in")
    inter = alloc((_P_MAX, n_groups * group_stride), nl.float8_e4m3fn_x4, name="inter")
    inter_s = alloc((_P_MAX, n_groups * group_stride), nl.uint8, name="inter_s")
    # fp32 PSUM with hi / lo keeps the evicted terms in fp32: the store DMA adds them in fp32 and rounds once.
    out_sb_fp32 = down_fp32 and hi_lo and not post_down
    out_sb = alloc((stationary_cols, out_cols), nl.float32 if out_sb_fp32 else nl.bfloat16, name="out_sb")
    if post_down:
        # slot_mask[m, (g, k, t)] = 1 if row m of group g (pair 16g + m/8) is pair t*K + k; aff_rows[r] = the affinity
        # of the pair of output row r (r = h*T*K + T*k + t); token_sum[r, h*T + t] = 1 if output row r is
        # h*T*K + T*k + t (any k): the stationary that adds the K pair rows of each token; out_tokens
        # [Hh * T, out_cols] holds the sums (partition h*T + t).
        rows = alloc((_P_MAX, 4), nl.int32, name="pd_rows")  # r, r / 8, the pair of row r, the token column of row r
        slot_val = alloc((_P_MAX, n_groups * n_pairs), nl.int32, name="slot_val")
        slot_mask = alloc((_P_MAX, n_groups * n_pairs), nl.bfloat16, name="slot_mask")
        pair_ids = alloc((_P_MAX, n_pairs), nl.int32, name="pair_ids")
        row_select = alloc((_P_MAX, n_pairs), nl.uint16, name="row_select")
        row_affinity = alloc((_P_MAX, n_pairs), nl.bfloat16, name="row_affinity")
        aff_rows = alloc((_P_MAX, 1), nl.float32, name="aff_rows")
        sum_cols = alloc((_P_MAX, n_out_halves * T), nl.int32, name="sum_cols")
        token_sum = alloc((_P_MAX, n_out_halves * T), nl.bfloat16, name="token_sum")
        out_tokens = alloc((n_out_halves * T, out_cols), nl.bfloat16, name="out_tokens")
        # The finite pair rows of a token-sum chunk (a non-finite row would reach every token through the 0/1 matmul).
        sum_check = alloc((stationary_cols, _DOWN_CHUNK), nl.bfloat16, name="sum_check")
        sum_finite = alloc((stationary_cols, _DOWN_CHUNK), nl.uint16, name="sum_finite")
        sum_rows = alloc((stationary_cols, _DOWN_CHUNK), nl.bfloat16, name="sum_rows")

    # Step 1.4: Allocate the PSUM tiles at fixed banks (layouts in the kernel docstring).
    """Gate/up of group g is in bank g % 2, so gate/up(g + 1) never waits for ACT / DVE still reading group g. With a
    wide fp32 out_ps (banks 2-7) the tiles cover all partitions, and the gated values of group g use columns
    _PSUM_COLS - 4T .. _PSUM_COLS - 1 of its tile (between the gate and up columns): the compiler rejects two PSUM tiles
    in one bank whose matmul writes interleave, so gated_ps cannot get its own tile in banks 0-1."""
    gate_up_ps_by_parity = [
        nl.ndarray(
            (_P_MAX if wide_out_ps else stationary_tokens, gate_up_row),
            dtype=nl.float32 if gate_up_fp32 else nl.bfloat16,
            buffer=nl.psum,
            address=(0, parity * gate_up_banks * _PSUM_BANK_BYTES),
            name=f"gate_up_ps{parity}",
        )
        for parity in range(2)
    ]
    # A wide fp32 out_ps takes banks 2-7 (the bank-6 index, affinity and routing tiles are done before down(0)).
    out_ps = nl.ndarray(
        (stationary_cols, out_cols),
        dtype=nl.float32 if down_fp32 else nl.bfloat16,
        buffer=nl.psum,
        address=(0, out_bank * _PSUM_BANK_BYTES),
        name="out_ps",
    )
    index_ps = nl.ndarray(
        (_P_MAX, n_pairs), dtype=nl.float32, buffer=nl.psum, address=(0, 6 * _PSUM_BANK_BYTES), name="index_ps"
    )
    affinity_ps = nl.ndarray(
        (_P_MAX, n_pairs),
        dtype=nl.bfloat16,
        buffer=nl.psum,
        address=(0, 6 * _PSUM_BANK_BYTES + 1024),
        name="affinity_ps",
    )
    # Standalone only: a stack of layers as E would not fit the PSUM bank.
    affinity_t_ps = None
    if not inputs_in_sbuf:
        affinity_t_ps = nl.ndarray(
            (_P_MAX, n_expert_tiles * T),
            dtype=nl.float32,
            buffer=nl.psum,
            address=(0, 6 * _PSUM_BANK_BYTES + 1536),
            name="affinity_t_ps",
        )
    # Router tiles only: the broadcast index (columns 0..63) and affinities (64..127).
    routing_ps = nl.ndarray(
        (_P_MAX, 2 * n_pairs), dtype=nl.float32, buffer=nl.psum, address=(0, 6 * _PSUM_BANK_BYTES), name="routing_ps"
    )
    gated_ps = None
    if not wide_out_ps:
        gated_ps = nl.ndarray(
            (_P_MAX, 2 * _X4 * stationary_tokens),
            dtype=nl.bfloat16,
            buffer=nl.psum,
            address=(0, 7 * _PSUM_BANK_BYTES),
            name="gated_ps",
        )
    # post_down: the token sums of 4 output chunks in flight, [h*T + t, 512] each, in banks 0-1 (gate/up is done).
    sum_ps_ring = [
        nl.ndarray(
            (n_out_halves * T, _DOWN_CHUNK),
            dtype=nl.bfloat16,
            buffer=nl.psum,
            address=(0, (slot // 2) * _PSUM_BANK_BYTES + (slot % 2) * _PSUM_BANK_BYTES // 2),
            name=f"sum_ps{slot}",
        )
        for slot in range(4 if post_down else 0)
    ]
    if output == None:
        output = nl.ndarray((T, H), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="output")
    valid = T if valid_tokens == None else valid_tokens
    dve, act = nisa.vector_engine, nisa.scalar_engine

    def after_norm_op(op: Callable[..., nisa.NkiInstruction]) -> Callable[..., nisa.NkiInstruction]:
        """Wrap op so that its instruction is scheduled after after_norm (fused), and the setup does not delay the
        RMSNorm."""

        def scheduled(*args, **kwargs) -> nisa.NkiInstruction:
            instruction = op(*args, **kwargs)
            if after_norm != None:
                instruction.depends_on(after_norm, kind=nisa.dependency.schedule)
            return instruction

        return scheduled

    iota, memset = after_norm_op(nisa.iota), after_norm_op(nisa.memset)
    if setup_after_stream and stream_chain:
        iota = _scheduled_after(iota, stream_chain[-1])

    # Step 1.5: post_down: zero the down-projection stationary. Else zero qmx_in for Hh = 2 (its first token blocks).
    if post_down:
        # The stationary columns before the pair-slot block of each group stay zero (with valid MX scales).
        memset(dst=inter.view(nl.int32), value=0, name="memset_inter")
        memset(dst=inter_s, value=0, name="memset_inter_s")
    elif n_out_halves > 1:
        memset(dst=qmx_in, value=0.0, name="memset_qmx_in")

    # Step 2: Load the routing, the hidden and the group-0 weights.
    if routing_broadcast:
        index_copy = None  # the router already wrote index / affinity on every partition
    elif inputs_in_sbuf:
        # Step 2.1: Router tiles: broadcast index[t, k] and affinity[t, k] to every partition (index first).
        # pair_tokens[t, n] = 1 if pair n belongs to token t: 0 <= n - K*t < K.
        iota(dst=pair_offsets, pattern=[[1, n_pairs]], offset=0, channel_multiplier=-K, name="iota_pair_offsets")
        nisa.tensor_scalar(
            dst=pair_tokens, data=pair_offsets, op0=nl.greater_equal, operand0=0, engine=dve, name="pair_tokens_ge0"
        )
        nisa.scalar_tensor_tensor(
            dst=pair_tokens,
            data=pair_offsets,
            op0=nl.less,
            operand0=K,
            op1=nl.multiply,
            operand1=pair_tokens,
            name="build_pair_tokens",
        )
        memset(dst=ones_tokens, value=1.0, engine=dve, name="memset_ones_tokens")
        # Expert ids are below 2048, so fp16 is exact. The affinities round to bf16 as in the standalone path.
        routing_copies = []
        for j, (values, name, dst) in enumerate(
            ((expert_index, "index", index), (expert_affinities, "affinity", affinity))
        ):
            # routing_moving[t, (j, n)] = value j of pair n if pair n belongs to token t, else 0.
            nisa.tensor_tensor(
                dst=routing_moving[:, j * n_pairs : (j + 1) * n_pairs],
                data1=values.ap(pattern=[[K, T], [0, T], [1, K]], offset=0),  # [t, (t', k)]: value of (t, k)
                data2=pair_tokens,
                op=nl.multiply,
                engine=dve,
                name=f"routing_moving_{name}",
            )
            # routing_ps[a, (j, n)] = sum over t of routing_moving[t, (j, n)]: value j of pair n on every partition.
            nisa.nc_matmul(
                dst=routing_ps[:, j * n_pairs : (j + 1) * n_pairs],
                stationary=ones_tokens,
                moving=routing_moving[:, j * n_pairs : (j + 1) * n_pairs],
                name=f"mm_{name}_bcast",
            )
            routing_copies.append(
                nisa.tensor_copy(
                    dst=dst, src=routing_ps[:, j * n_pairs : (j + 1) * n_pairs], engine=dve, name=f"copy_{name}"
                )
            )
        index_copy = routing_copies[0]
    if inputs_in_sbuf:
        index_row = index[0:1, :]  # the group-0 HWDGE loads read their expert ids from partition 0
    else:
        # Step 2.2: Standalone: small HWDGE loads of the index as one 256 B row on partition 0, then the hidden.
        """With U = 2 the PE starts early, so the hidden goes on the SP queue ahead of the expert loads. On ACT, the
        compiler would queue it behind the ACT expert loads."""
        hidden_engine = nisa.engine.sync if early_head else nisa.engine.scalar
        nisa.dma_copy(
            dst=index_row,
            src=expert_index.reshape((1, n_pairs)),
            dge_mode=nisa.dge_mode.hwdge,
            engine=nisa.engine.sync,
            name="load_index",
        )
        nisa.dma_copy(
            dst=hidden_raw.reshape((_P_MAX, n_h512, stationary_tokens))[:, :, 0:T],
            src=hidden_qtz.reshape((_P_MAX, n_h512 * T)),
            dge_mode=nisa.dge_mode.hwdge,
            engine=hidden_engine,
            name="load_hidden",
        )
        hidden_s_load = nisa.dma_copy(
            dst=hidden_raw_s.reshape((_P_MAX, n_h512, stationary_tokens))[:, :, 0:T],
            src=hidden_scale.reshape((_P_MAX, n_h512 * T)),
            dge_mode=nisa.dge_mode.hwdge,
            engine=hidden_engine,
            name="load_hidden_s",
        )
    if early_head:
        # U = 2: ACT zeroes column 0 of gated (0 * offsets, Step 3.4) while it waits for the index row. This loads the
        # ACT table before its expert loads; otherwise the first ACT scale copy loads it on the critical path.
        iota(dst=offsets, pattern=[[-_MX_ROWS, _GROUP]], offset=0, channel_multiplier=1, name="iota_block_diag_offsets")
        zero_gated_col0 = nisa.activation(
            dst=gated[:, 0:1], op=nl.copy, data=offsets[0:T, 0:1], scale=0.0, name="zero_gated_col0"
        )

    # Step 2.3: Sub-group 0: one HWDGE DMA per expert (weights and scales), alternating SP and ACT (long_stream: then
    # hwdge_group on SP).
    kernel_assert(not group0_gather or gather_index_in != None, "group0_gather needs gather_index_in")

    def load_expert(pair: int, engine: nisa.engine) -> nisa.NkiInstruction:
        """One HWDGE DMA of the 128 packed rows of the expert of pair into its row slot (group pair / 16)."""
        if expert_index_tk != None:
            # expert id of pair n = t*K + k from the router tile [t, k]
            expert = nisa.scalar_load(
                expert_index_tk.view(nl.int32)[pair // K : pair // K + 1, pair % K : pair % K + 1]
            )
        else:
            expert = nisa.scalar_load(index_row[0:1, pair : pair + 1])
        slot_pair = ((pair // _GROUP) % n_gate_up_slots) * _GROUP + pair % _GROUP  # row slot of the group, pair c
        return nisa.dma_copy(
            dst=gate_up_w.view(nl.uint8)[:, slot_pair * row_bytes : (slot_pair + 1) * row_bytes],
            # [p, bytes]: the 128 packed rows of expert `expert`
            src=expert_gate_up_packed.ap(
                pattern=[[row_bytes, _P_MAX], [1, row_bytes]], offset=0, scalar_offset=expert, indirect_dim=0
            ),
            dge_mode=nisa.dge_mode.hwdge,
            engine=engine,
            name=f"load_gate_up_p{pair}",
        )

    hw_loads = [
        load_expert(pair, (nisa.engine.sync, nisa.engine.scalar)[pair % 2])
        for pair in range(sub_pairs if first_gathered else 0)
    ]
    if hwdge_group != None:
        # long_stream: SP loads the 16 experts of hwdge_group in order, so Pool has one gate/up gather less.
        kernel_assert(hwdge_group < n_gate_up_slots, "The long_stream HWDGE group needs its own gate/up row slot")
        previous = hw_loads[-2] if len(hw_loads) > 1 else None  # the last group-0 load on SP
        n_act_loads = _LONG_ACT_HWDGE_LOADS if inputs_in_sbuf else 0  # Step 7.1 issues them on ACT
        for pair in range(hwdge_group * _GROUP, (hwdge_group + 1) * _GROUP - n_act_loads):
            load = load_expert(pair, nisa.engine.sync)
            if previous != None:
                load.depends_on(previous, kind=nisa.dependency.schedule)
            previous = load
    if early_head:
        # The hidden loads and the ACT table load go first on their queues.
        hw_loads[0].depends_on(hidden_s_load, kind=nisa.dependency.schedule)
        hw_loads[1].depends_on(zero_gated_col0, kind=nisa.dependency.schedule)

    if not inputs_in_sbuf:
        # Step 2.4: Standalone: load the affinities, then broadcast the index row to every partition.
        nisa.dma_copy(
            dst=affinity_te,
            src=expert_affinities,
            dge_mode=nisa.dge_mode.hwdge,
            engine=nisa.engine.sync,
            name="load_affinity",
        )
        # index[a, n] = index_row[n] with a ones matmul (a stride-0 broadcast DMA of 256 B rows is slow). Expert ids
        # are below 2048, so fp16 is exact (bf16 is exact only below 256).
        memset(dst=ones_row, value=1.0, engine=dve, name="memset_ones_row")
        nisa.tensor_copy(dst=index_row_f16, src=index_row, engine=dve, name="cast_index_row")
        nisa.nc_matmul(dst=index_ps, stationary=ones_row, moving=index_row_f16, name="mm_index_bcast")
        index_copy = nisa.tensor_copy(dst=index, src=index_ps, engine=dve, name="copy_index")

    if gather_index_in == None:
        # Step 3: Build the gather indices and the constants (gather row mapping in the kernel docstring).
        # Step 3.1: Bit fields of the partition index a in partition[a, 0..6].
        iota(dst=partition[:, 0:1], pattern=[[0, 1]], offset=0, channel_multiplier=1, name="iota_partition")
        nisa.tensor_scalar(
            dst=partition[:, 1:2],  # a % 8
            data=partition[:, 0:1],
            op0=nl.bitwise_and,
            operand0=7,
            engine=dve,
            name="field_a_and7",
        )
        nisa.tensor_scalar(
            dst=partition[:, 2:3],  # a % 16
            data=partition[:, 0:1],
            op0=nl.bitwise_and,
            operand0=15,
            engine=dve,
            name="field_a_and15",
        )
        nisa.tensor_scalar(
            dst=partition[:, 3:4],  # a / 8
            data=partition[:, 0:1],
            op0=nl.right_shift,
            operand0=3,
            engine=dve,
            name="field_a_shr3",
        )
        nisa.tensor_scalar(
            dst=partition[:, 4:5],  # a / 16
            data=partition[:, 0:1],
            op0=nl.right_shift,
            operand0=4,
            engine=dve,
            name="field_a_shr4",
        )
        # Down-scale pair 16(a/4 % 4) + 4(a/32) + a%4 = ((a & 12) << 2) | ((a >> 3) & 12) | (a & 3) in column 5.
        nisa.tensor_scalar(
            dst=partition[:, 5:6],  # (a & 12) << 2
            data=partition[:, 0:1],
            op0=nl.bitwise_and,
            operand0=12,
            op1=nl.left_shift,
            operand1=2,
            engine=dve,
            name="field_a_and12_shl2",
        )
        nisa.tensor_scalar(
            dst=partition[:, 6:7],  # (a >> 3) & 12 = 4(a/32)
            data=partition[:, 0:1],
            op0=nl.right_shift,
            operand0=3,
            op1=nl.bitwise_and,
            operand1=12,
            engine=dve,
            name="field_a_shr3_and12",
        )
        nisa.scalar_tensor_tensor(
            dst=partition[:, 5:6],
            data=partition[:, 0:1],
            op0=nl.bitwise_and,
            operand0=3,
            op1=nl.bitwise_or,
            operand1=partition[:, 5:6],
            name="field_down_s_pair_lo",
        )
        nisa.tensor_tensor(
            dst=partition[:, 5:6],
            data1=partition[:, 5:6],
            data2=partition[:, 6:7],
            op=nl.bitwise_or,
            engine=dve,
            name="field_down_s_pair",
        )

        # Step 3.2: Pair selectors: select[a, j - 1] = Mj + a%M (gate/up of sub-group j = 1..G), select[a, G + g] =
        # 16g + a/8 (down weights), select[a, G + N/16 + s] = the pair of down-scale tile s.
        a_mod_m_col, a_div_m_col = (2, 4) if sub_pairs == _GROUP else (1, 3)  # columns of a % M and a / M in partition
        down_select = select[:, n_gathered : n_gathered + n_groups]
        down_scale_select = select[:, n_gathered + n_groups : n_gathered + n_groups + n_scale_tiles]
        iota(
            dst=select[:, 0:n_gathered],
            pattern=[[sub_pairs, n_gathered]],
            offset=sub_pairs,
            name="iota_select_gate_up_w",
        )
        nisa.tensor_tensor(
            dst=select[:, 0:n_gathered],
            data1=select[:, 0:n_gathered],
            data2=partition.ap(pattern=[[7, _P_MAX], [0, n_gathered]], offset=a_mod_m_col),  # [a, j - 1]: a % M
            op=nl.add,
            engine=dve,
            name="select_gate_up_w",
        )
        iota(dst=down_select, pattern=[[_GROUP, n_groups]], offset=0, name="iota_select_down_w")
        nisa.tensor_tensor(
            dst=down_select,
            data1=down_select,
            data2=partition.ap(pattern=[[7, _P_MAX], [0, n_groups]], offset=3),  # [a, g]: a / 8 for each group
            op=nl.add,
            engine=dve,
            name="select_down_w",
        )
        # [a, s]: 64s, the first pair of down-scale tile s (one tile: step 0, the instruction of the 4-group program)
        scale_step = _SCALE_GROUPS * _GROUP if n_scale_tiles > 1 else 0
        iota(dst=down_scale_select, pattern=[[scale_step, n_scale_tiles]], offset=0, name="iota_select_down_s")
        nisa.tensor_tensor(
            dst=down_scale_select,
            data1=down_scale_select,
            data2=partition.ap(pattern=[[7, _P_MAX], [0, n_scale_tiles]], offset=5),  # [a, s]: the down-scale pair
            op=nl.add,
            engine=dve,
            name="select_down_s",
        )

        # Step 3.3: Source rows inside each expert: lanes[a, b] = (128/M)b + a/M (gate/up), lanes[a, 16] = a%8, + SKIP
        # if a%8 >= I/4 (down weights), lanes[a, 17] = (a & 16) << 16 = SKIP if a/16 is odd (down scales).
        iota(
            dst=lanes[:, 0:sub_pairs], pattern=[[_P_MAX // sub_pairs, sub_pairs]], offset=0, name="iota_lane_gate_up_w"
        )
        nisa.tensor_tensor(
            dst=lanes[:, 0:sub_pairs],
            data1=lanes[:, 0:sub_pairs],
            data2=partition.ap(pattern=[[7, _P_MAX], [0, sub_pairs]], offset=a_div_m_col),  # [a, b]: a / M for each b
            op=nl.add,
            engine=dve,
            name="lane_gate_up_w",
        )
        nisa.tensor_scalar(
            dst=lanes[:, 16:17],
            data=partition[:, 1:2],  # a % 8
            op0=nl.greater_equal,
            operand0=down_rows,
            op1=nl.multiply,
            operand1=_SKIP,
            engine=dve,
            name="lane_down_w_skip",
        )
        nisa.tensor_tensor(
            dst=lanes[:, 16:17],
            data1=lanes[:, 16:17],
            data2=partition[:, 1:2],  # a % 8
            op=nl.add,
            engine=dve,
            name="lane_down_w",
        )
        nisa.tensor_scalar(
            dst=lanes[:, 17:18],
            data=partition[:, 0:1],
            op0=nl.bitwise_and,
            operand0=16,
            op1=nl.left_shift,
            operand1=16,
            engine=dve,
            name="lane_down_s",
        )

    def build_constants(iota: Callable[..., nisa.NkiInstruction]) -> None:
        """Step 3.4: the constants (iotas on Pool, the rest on DVE)."""
        # Block-diagonal mask masks[m, c] = 1 if 0 <= m - 8c < 8 (offsets come from Step 2.2 with the early head).
        if not early_head:
            iota(
                dst=offsets,
                pattern=[[-_MX_ROWS, _GROUP]],
                offset=0,
                channel_multiplier=1,
                name="iota_block_diag_offsets",
            )
        if not post_down:
            masks_ge0 = nisa.tensor_scalar(
                dst=masks, data=offsets, op0=nl.greater_equal, operand0=0, engine=dve, name="masks_ge0"
            )
            if index_copy != None:
                masks_ge0.depends_on(index_copy, kind=nisa.dependency.schedule)
            nisa.scalar_tensor_tensor(
                dst=masks,
                data=offsets,
                op0=nl.less,
                operand0=_MX_ROWS,
                op1=nl.multiply,
                operand1=masks,
                name="mask_block_diag",
            )
        # Rows p >= I/4 of the transposed SiLU(gate) * up stay zero (the early head zeroed column 0 in Step 2.2).
        zero_from_col = 1 if early_head else 0
        memset(dst=gated[:, zero_from_col:], value=0.0, engine=dve, name="zero_gated")
        # Affinity constants: expert_id[a, h] = a + 128h and the ones stationary.
        if not inputs_in_sbuf:
            iota(
                dst=expert_id, pattern=[[_P_MAX, n_expert_tiles]], offset=0, channel_multiplier=1, name="iota_expert_id"
            )
        memset(dst=ones, value=1.0, engine=nisa.engine.vector, name="memset_ones")
        if not post_down:
            # token_rows[m, (g, t)] = 1 if row m of group g (pair 16g + m/8) is of token t: 0 <= m - 8Kt + 128g < 8K.
            iota(
                dst=token_offsets,
                pattern=[[_P_MAX, n_groups], [-_MX_ROWS * K, T]],
                offset=0,
                channel_multiplier=1,
                name="iota_token_rows",
            )
            token_rows_ge0 = nisa.tensor_scalar(
                dst=token_rows, data=token_offsets, op0=nl.greater_equal, operand0=0, engine=dve, name="token_rows_ge0"
            )
            if index_copy != None:
                token_rows_ge0.depends_on(index_copy, kind=nisa.dependency.schedule)
            nisa.scalar_tensor_tensor(
                dst=token_rows,
                data=token_offsets,
                op0=nl.less,
                operand0=_MX_ROWS * K,
                op1=nl.multiply,
                operand1=token_rows,
                name="build_token_rows",
            )

        if post_down:
            log_t, log_k = T.bit_length() - 1, K.bit_length() - 1
            # pd_rows[r, :] = r, r / 8, the pair of output row r, scratch, the token column of row r, scratch.
            iota(dst=rows[:, 0:1], pattern=[[0, 1]], offset=0, channel_multiplier=1, name="iota_pd_rows")
            nisa.tensor_scalar(
                dst=rows[:, 1:2], data=rows[:, 0:1], op0=nl.right_shift, operand0=3, engine=dve, name="pd_row_div8"
            )
            # slot_mask[m, (g, k, t)] = 1 if row m of group g (pair 16g + m/8) is pair t*K + k: K*t + k - 16g == m/8.
            iota(dst=slot_val, pattern=[[-_GROUP, n_groups], [1, K], [K, T]], offset=0, name="iota_slot_pairs")
            nisa.tensor_tensor(
                dst=slot_mask,
                data1=slot_val,
                data2=rows.ap(pattern=[[4, _P_MAX], [0, n_groups * n_pairs]], offset=1),  # [m, (g, k, t)]: m / 8
                op=nl.equal,
                engine=dve,
                name="build_slot_mask",
            )
            # The pair of output row r = h*T*K + T*k + t: K*(r % T) + (r / T) % K.
            nisa.tensor_scalar(
                dst=rows[:, 2:3],
                data=rows[:, 0:1],
                op0=nl.bitwise_and,
                operand0=T - 1,
                op1=nl.left_shift,
                operand1=log_k,
                engine=dve,
                name="pd_row_token",
            )
            nisa.tensor_scalar(
                dst=rows[:, 3:4],
                data=rows[:, 0:1],
                op0=nl.right_shift,
                operand0=log_t,
                op1=nl.bitwise_and,
                operand1=K - 1,
                engine=dve,
                name="pd_row_slot",
            )
            nisa.tensor_tensor(
                dst=rows[:, 2:3],
                data1=rows[:, 2:3],
                data2=rows[:, 3:4],
                op=nl.bitwise_or,
                engine=dve,
                name="pd_row_pair",
            )
            iota(dst=pair_ids, pattern=[[1, n_pairs]], offset=0, name="iota_pair_ids")
            nisa.tensor_tensor(
                dst=row_select,
                data1=pair_ids,
                data2=rows.ap(pattern=[[4, _P_MAX], [0, n_pairs]], offset=2),  # [r, n]: the pair of row r
                op=nl.equal,
                engine=dve,
                name="build_row_select",
            )
            # token_sum[r, j] = 1 if j == T*(r / (T*K)) + r % T (the output half and the token of row r).
            nisa.tensor_scalar(
                dst=rows[:, 3:4],
                data=rows[:, 0:1],
                op0=nl.right_shift,
                operand0=n_pairs.bit_length() - 1,
                op1=nl.left_shift,
                operand1=log_t,
                engine=dve,
                name="pd_sum_half",
            )
            nisa.scalar_tensor_tensor(
                dst=rows[:, 3:4],
                data=rows[:, 0:1],
                op0=nl.bitwise_and,
                operand0=T - 1,
                op1=nl.bitwise_or,
                operand1=rows[:, 3:4],
                name="pd_sum_col",
            )
            iota(dst=sum_cols, pattern=[[1, n_out_halves * T]], offset=0, name="iota_sum_cols")
            nisa.tensor_tensor(
                dst=token_sum,
                data1=sum_cols,
                data2=rows.ap(pattern=[[4, _P_MAX], [0, n_out_halves * T]], offset=3),  # [r, j]: the column of row r
                op=nl.equal,
                engine=dve,
                name="build_token_sum",
            )

    # Step 3.4: Constants, built while Pool waits for the index load. With caller-built gather indices, after Step 4.
    if gather_index_in == None:
        build_constants(iota)

    # Step 3.5: experts[a, j] = index[a, select[a, j]], then gather_index = rows per expert * expert + source row.
    n_select = n_gathered + n_groups + n_scale_tiles
    gate_up_index_col, down_index_col = 0, n_gathered * sub_pairs
    down_scale_index_col = down_index_col + n_groups
    if gather_index_in != None:
        gather_index, gather_index_done = gather_index_in, None
    else:
        if tile_first:
            """Sub-group 1 (pairs M .. 2M - 1): partition a needs the expert of pair n = M + a%M, token n / K, slot a%K
            (M % K == 0). tile_rows[t, a] = 1 if t == n / K, so a matmul gives picked[a, k] = index_tk[n / K, k]; the
            slot mask keeps k = a%K, and a sum over k leaves the expert."""
            kernel_assert(
                K in (8, 16) and sub_pairs % K == 0, f"The tile-indexed first gather needs K = 8 or 16, got K={K}"
            )
            iota(dst=tile_a, pattern=[[1, _P_MAX]], offset=0, channel_multiplier=0, name="iota_tile_a")
            iota(dst=tile_t, pattern=[[0, _P_MAX]], offset=0, channel_multiplier=1, name="iota_tile_t")
            nisa.tensor_scalar(
                dst=tile_a,
                data=tile_a,
                op0=nl.bitwise_and,
                operand0=sub_pairs - 1,
                op1=nl.bitwise_or,  # M + a%M (M is a power of two)
                operand1=sub_pairs,
                engine=dve,
                name="tile_pair",
            )
            nisa.tensor_scalar(
                dst=tile_a, data=tile_a, op0=nl.right_shift, operand0=K.bit_length() - 1, engine=dve, name="tile_token"
            )  # n / K
            nisa.tensor_tensor(dst=tile_rows, data1=tile_a, data2=tile_t, op=nl.equal, engine=dve, name="tile_rows_op")
            iota(dst=tile_k, pattern=[[1, K]], offset=0, channel_multiplier=0, name="iota_tile_k")
            nisa.tensor_tensor(
                dst=tile_k_mask,
                data1=tile_k,
                data2=partition.ap(pattern=[[7, _P_MAX], [0, K]], offset=1 if K == _MX_ROWS else 2),  # [a, k]: a%K
                op=nl.equal,
                engine=dve,
                name="tile_k_mask_op",
            )
            nisa.tensor_copy(dst=tile_index_f16, src=expert_index_tk, engine=dve, name="tile_index_f16_op")
            tile_ps = nl.ndarray(
                (_P_MAX, K), dtype=nl.float32, buffer=nl.psum, address=(0, 6 * _PSUM_BANK_BYTES), name="tile_ps"
            )
            nisa.nc_matmul(dst=tile_ps, stationary=tile_rows, moving=tile_index_f16, name="mm_tile_index")
            nisa.tensor_tensor(
                dst=tile_picked, data1=tile_ps, data2=tile_k_mask, op=nl.multiply, engine=dve, name="tile_pick"
            )
            nisa.tensor_reduce(dst=tile_expert, op=nl.add, data=tile_picked, axis=1, name="tile_expert_op")
            nisa.scalar_tensor_tensor(
                dst=gather_index[:, gate_up_index_col : gate_up_index_col + sub_pairs],
                data=tile_expert.ap(pattern=[[1, _P_MAX], [0, sub_pairs]], offset=0),  # [a, b]: expert of pair M + a%M
                op0=nl.multiply,
                operand0=_P_MAX,
                op1=nl.add,
                operand1=lanes[:, 0:sub_pairs],  # [a, b]: row (128/M)b + a/M
                name="gather_index_gate_up_tile",
            )
        nisa.nc_n_gather(dst=experts, data=index, indices=select.view(nl.uint32), name="gather_experts")
        nisa.scalar_tensor_tensor(
            dst=gather_index[:, gate_up_index_col + tile_first * sub_pairs : down_index_col],
            # [a, j - 1, b]: expert of sub-group j
            data=experts.ap(
                pattern=[[n_select, _P_MAX], [1, n_gathered - tile_first], [0, sub_pairs]], offset=int(tile_first)
            ),
            op0=nl.multiply,
            operand0=_P_MAX,
            op1=nl.add,
            # [a, j - 1, b]: row (128/M)b + a/M
            operand1=lanes.ap(pattern=[[18, _P_MAX], [0, n_gathered - tile_first], [1, sub_pairs]], offset=0),
            name="gather_index_gate_up_w",
        )
        nisa.scalar_tensor_tensor(
            dst=gather_index[:, down_index_col : down_index_col + n_groups],
            # [a, g, 0]: expert of group g
            data=experts.ap(pattern=[[n_select, _P_MAX], [1, n_groups], [0, 1]], offset=n_gathered),
            op0=nl.multiply,
            operand0=down_rows_hbm,
            op1=nl.add,
            operand1=lanes.ap(pattern=[[18, _P_MAX], [0, n_groups], [1, 1]], offset=16),  # [a, g, 0]: row a%8 (+ SKIP)
            name="gather_index_down_w",
        )
        gather_index_done = nisa.tensor_tensor(
            dst=gather_index[:, down_scale_index_col : down_scale_index_col + n_scale_tiles],
            # experts of the scale pairs
            data1=experts[:, n_gathered + n_groups : n_gathered + n_groups + n_scale_tiles],
            data2=lanes.ap(pattern=[[18, _P_MAX], [0, n_scale_tiles]], offset=17),  # [a, s]: 0 (+ SKIP)
            op=nl.add,
            engine=nisa.vector_engine,
            name="gather_index_down_s",
        )

    # Step 4: Start the Pool weight stream, in PE consumption order.
    gate_up_rows_hbm = expert_gate_up_packed.reshape((E * _P_MAX, row_bytes))  # row 128e + p: expert e, row p
    down_w_rows_hbm = expert_down_weights.reshape((E * down_rows_hbm, H))  # row R e + p: expert e, row p (R = 8 or I/4)
    down_s_rows_hbm = expert_down_weights_scale.reshape((E, n_scale_sets * H))  # row e: expert e (scale groups)
    """Stream order: the gate/up rows of group g (one gather per sub-group j = gU + u > 0), then the down weights of
    group g - 1. A gather into a reused row slot comes after (in program order) the re-layout that reads the previous
    group of the slot, so its write is ordered after those reads."""
    stream_plan = []  # (name, destination, HBM rows, first gather_index column, index columns, earliest issue step)
    for g in range(n_groups + 1):
        for u in range(n_sub_groups if g < n_groups else 0):
            j = g * n_sub_groups + u
            if j < first_gathered or j == hwdge_group:
                continue  # sub-group 0 (and the long_stream hwdge_group) come from the HWDGE loads (Step 2.3)
            slot = ((g % n_gate_up_slots) * _GROUP + u * sub_pairs) * row_bytes  # bytes: rows of the sub-group
            stream_plan.append(
                (
                    f"gather_gate_up_g{g}" + (f"_u{u}" if n_sub_groups > 1 else ""),
                    gate_up_w.view(nl.uint8)[:, slot : slot + sub_pairs * row_bytes],
                    gate_up_rows_hbm,
                    gate_up_index_col + (j - first_gathered) * sub_pairs,
                    sub_pairs,
                    max(-1, g - n_gate_up_slots),  # the re-layout step that frees the slot
                )
            )
        if g == 0:
            continue
        if (g - 1) % _SCALE_GROUPS == 0:
            tile = (g - 1) // _SCALE_GROUPS  # down-scale tile of groups 4s .. 4s + 3, before their down weights
            stream_plan.append(
                (
                    "gather_down_s" + (f"{tile}" if tile > 0 else ""),
                    down_s[:, tile * n_scale_sets * H : (tile + 1) * n_scale_sets * H],
                    down_s_rows_hbm,
                    down_scale_index_col + tile,
                    1,
                    -1,
                )
            )
        stream_plan.append(
            (f"gather_down_w_g{g - 1}", down_w[:, (g - 1) * H : g * H], down_w_rows_hbm, down_index_col + g - 1, 1, -1)
        )
    # Issue step of each gather (-1: before the compute loop, s: after relayout(s)), not before the gather ahead of it.
    issue_steps, latest = [], -1
    for *_, earliest in stream_plan:
        latest = max(latest, earliest)
        issue_steps.append(latest)
    stream = [None] * len(stream_plan)

    def issue_stream(step: int) -> None:
        """Issue the stream gathers planned right after the re-layout of group `step` (-1: before the compute loop)."""
        for position, (name, dst, rows, first_col, n_cols, _) in enumerate(stream_plan):
            if issue_steps[position] == step:
                stream[position] = nisa.dma_copy(
                    dst=dst,
                    # [128 * n_cols rows, row]: row 128b + a is HBM row gather_index[a, first_col + b]
                    src=rows.ap(
                        pattern=[[rows.shape[1], _P_MAX * n_cols], [1, rows.shape[1]]],
                        offset=0,
                        vector_offset=gather_index[:, first_col : first_col + n_cols],
                        indirect_dim=0,
                    ),
                    dge_mode=nisa.dge_mode.swdge,
                    oob_mode=nisa.oob_mode.skip,
                    name=name,
                )

    issue_stream(-1)
    if finish_affinities != None:
        finish_affinities(gather_index_done)
    if gather_index_in != None:
        # Pool runs in order. An iota that the compiler puts ahead of the gathers of an earlier call waits for the late
        # DVE reads of its tile in that call and holds the stream, so the iotas follow the first gather of this call.
        build_constants(_scheduled_after(iota, stream[0]))

    # Step 5: Build the per-pair affinities while the PE is idle.
    if not inputs_in_sbuf:
        # Step 5.1: affinity[:, n] = expert_affinities[t_n, e_n] on every partition, for each tile of 128 experts:
        # transpose to [e on partitions, t], keep column n where index[n] == e, sum over e with a ones matmul.
        for half in range(n_expert_tiles):
            nisa.nc_transpose(
                dst=affinity_t_ps[:, half * T : (half + 1) * T],
                data=affinity_te[:, half * _P_MAX : (half + 1) * _P_MAX],
                name=f"transpose_affinity_{half}",
            )
        for half in range(n_expert_tiles):
            nisa.scalar_tensor_tensor(
                dst=affinity_hot[:, half * n_pairs : (half + 1) * n_pairs],
                data=index,
                op0=nl.equal,
                operand0=expert_id[:, half : half + 1],
                op1=nl.multiply,
                # [e, (t, k)]: affinity of token t for expert e of this tile, for each k
                operand1=affinity_t_ps.ap(pattern=[[n_expert_tiles * T, _P_MAX], [1, T], [0, K]], offset=half * T),
                name=f"affinity_hot_{half}",
            )
        for half in range(n_expert_tiles):
            nisa.nc_matmul(
                dst=affinity_ps,
                stationary=ones,
                moving=affinity_hot[:, half * n_pairs : (half + 1) * n_pairs],
                accumulate=half > 0,
                name=f"mm_affinity_{half}",
            )
        nisa.tensor_copy(dst=affinity, src=affinity_ps, engine=dve, name="copy_affinity")
    if not post_down:
        # Step 5.2: affinity_mask[m, n] = affinity of pair n on its own rows m = 8(n % 16) + p. Selects, not
        # multiplies by 0/1, so a NaN or Inf affinity stays in the rows and columns of its own pair and token.
        nisa.memset(dst=affinity_mask, value=0.0, engine=dve, name="zero_affinity_mask")
        nisa.tensor_copy_predicated(
            dst=affinity_mask,
            src=affinity,
            predicate=masks.ap(pattern=[[_GROUP, _P_MAX], [0, n_groups], [1, _GROUP]], offset=0),  # [m, (g, c)]
            name="build_affinity_mask",
        )
        # affinity_rows[m, g] = sum over c of affinity_mask[m, 16g + c] = affinity of pair 16g + m/8.
        nisa.tensor_reduce(
            dst=affinity_rows,
            op=nl.add,
            data=affinity_mask.ap(pattern=[[n_pairs, _P_MAX], [_GROUP, n_groups], [1, _GROUP]], offset=0),  # [m, g, c]
            axis=[2],
            name="build_affinity_rows",
        )
        # row_weights[m, (g, t)] = affinity_rows[m, g] if pair 16g + m/8 belongs to token t, else 0.
        nisa.memset(dst=row_weights, value=0.0, engine=dve, name="zero_row_weights")
        if n_scale_sets == 1:
            nisa.tensor_copy_predicated(
                dst=row_weights,
                src=affinity_rows.ap(pattern=[[n_groups, _P_MAX], [1, n_groups], [0, T]], offset=0),  # [m, (g, t)]
                predicate=token_rows,
                name="build_row_weights",
            )
        else:
            # Two scale groups: row_weights[m, (s, g, t)] keeps row m only in the stationary of its scale group s.
            nisa.tensor_tensor(
                dst=scale_set_rows,
                data1=token_rows.ap(pattern=[[n_groups * T, _P_MAX], [0, 2], [1, n_groups * T]], offset=0),
                data2=down_group_mask.ap(pattern=[[2, _P_MAX], [1, 2], [0, n_groups * T]], offset=0),
                op=nl.multiply,
                engine=dve,
                name="build_scale_set_rows",
            )
            nisa.tensor_copy_predicated(
                dst=row_weights,
                # [m, (s, g, t)]: [m, g]
                src=affinity_rows.ap(pattern=[[n_groups, _P_MAX], [0, 2], [1, n_groups], [0, T]], offset=0),
                predicate=scale_set_rows,
                name="build_row_weights",
            )
    else:
        # Step 5.2 (post_down): aff_rows[r] = the affinity of the pair of output row r, the eviction scale.
        nisa.memset(dst=row_affinity, value=0.0, engine=dve, name="zero_row_affinity")
        nisa.tensor_copy_predicated(dst=row_affinity, src=affinity, predicate=row_select, name="row_affinity_op")
        nisa.tensor_reduce(dst=aff_rows, op=nl.add, data=row_affinity, axis=1, name="build_aff_rows")

    # The hidden is read from here on: the caller may write it between the two phases.
    yield

    # Step 6: Define the per-group stages (Step 7 calls them).
    relayout_copies = {dve: [], act: []}

    def in_order(copy: nisa.NkiInstruction, engine: nisa.engine) -> None:
        """Schedule copy after the previous re-layout copy of engine (for ACT, after the last group-0 load).

        Without this order, the compiler runs the copies of the gathered groups before those of group 0 (HWDGE). ACT
        also issues half of the group-0 loads, so its copies follow them.
        """
        previous = (
            relayout_copies[engine][-1]
            if relayout_copies[engine]
            else (hw_loads[-1] if engine == act and hw_loads else None)
        )
        if previous != None:
            copy.depends_on(previous, kind=nisa.dependency.schedule)
        relayout_copies[engine].append(copy)

    def relayout(
        group: int,
        n_parts: int,
        runs: Optional[Sequence[int]] = None,
        after_dve: Optional[nisa.NkiInstruction] = None,
        after_act: Optional[nisa.NkiInstruction] = None,
    ) -> None:
        """Step 6.1: copy the pair runs `runs` (default: all n_parts runs) of the packed rows of group to its moving
        slot. After the last run, issue the gathers that reuse its row slot.

        DVE copies the weights (twice the ACT copy rate) and ACT copies the scale segments, run by run, so a run
        starts as soon as its experts land (group 0 arrives one expert at a time, a sub-group with its gather). With
        U = 2, group 0 copies each run in two halves of the H512 blocks, so its first matmuls start after a quarter of
        the copies. long_stream: for group > 0 ACT also copies the first 4 H512 blocks of the weights, because Pool
        descriptor windows block DVE and not ACT. The first DVE / ACT copy is scheduled after after_dve / after_act.
        """
        runs = range(n_parts) if runs == None else runs
        rows_base = (group % n_gate_up_slots) * _GROUP * row_bytes // 2  # row slot of the group, in 2-byte units
        moving_base = (group % n_moving_slots) * _GROUP * gate_up_cols  # moving slot of the group
        part_pairs = _GROUP // n_parts
        n_block_parts = 2 if group == 0 and n_sub_groups > 1 else 1
        part_blocks = n_h512 // n_block_parts
        for part in runs:
            for block_part in range(n_block_parts):
                for gate_or_up in range(2):
                    # (first H512 block, blocks, engine) of each copy
                    if long and group > 0:
                        block_runs = [
                            (0, _LONG_ACT_RELAYOUT_BLOCKS, act),
                            (_LONG_ACT_RELAYOUT_BLOCKS, n_h512 - _LONG_ACT_RELAYOUT_BLOCKS, dve),
                        ]
                    else:
                        # g0_relayout_engine="act": ACT copies all blocks of group 0 (it is not blocked by Pool windows)
                        weight_engine = act if group == 0 and g0_relayout_engine == "act" else dve
                        block_runs = [(block_part * part_blocks, part_blocks, weight_engine)]
                    for run_index, (first_block, run_blocks, engine) in enumerate(block_runs):
                        copy = nisa.tensor_copy(
                            # [p, hb, c, I]: moving slot, gate_or_up columns, blocks and pairs of this part
                            dst=gate_up_moving.ap(
                                pattern=[
                                    [n_moving_slots * _GROUP * gate_up_cols, _P_MAX],
                                    [group_cols, run_blocks],
                                    [I, part_pairs],
                                    [1, I],
                                ],
                                offset=moving_base
                                + (gate_or_up * n_h512 + first_block) * group_cols
                                + part * part_pairs * I,
                            ),
                            # [p, hb, c, I]: packed row [gate_or_up][hb][I] of each pair of this part
                            src=gate_up_w.ap(
                                pattern=[
                                    [n_gate_up_slots * _GROUP * row_bytes // 2, _P_MAX],
                                    [I, run_blocks],
                                    [row_bytes // 2, part_pairs],
                                    [1, I],
                                ],
                                offset=rows_base
                                + part * part_pairs * row_bytes // 2
                                + (gate_or_up * n_h512 + first_block) * I,
                            ),
                            engine=engine,
                            name=f"relayout_{('gate', 'up')[gate_or_up]}_g{group}_part{part}"
                            + (f"_h{block_part}" if n_block_parts > 1 else "")
                            + (f"_s{run_index}" if len(block_runs) > 1 else ""),
                        )
                        if engine == dve and after_dve != None:
                            copy.depends_on(after_dve, kind=nisa.dependency.schedule)
                            after_dve = None
                        if engine == act and after_act != None:
                            copy.depends_on(after_act, kind=nisa.dependency.schedule)
                            after_act = None
                        in_order(copy, engine)
            for segment in range(n_scale_segments):
                copy = nisa.tensor_copy(
                    # [p, hb % 4, c, I/2 words]: moving scale slot, segment `segment`, pairs of this part
                    dst=gate_up_moving_s.ap(
                        pattern=[
                            [n_moving_slots * n_scale_segments * _BLOCKS * group_cols // 2, _P_MAX],
                            [group_cols // 2, _BLOCKS],
                            [I // 2, part_pairs],
                            [1, I // 2],
                        ],
                        offset=((group % n_moving_slots) * n_scale_segments + segment) * _BLOCKS * group_cols // 2
                        + part * part_pairs * I // 2,
                    ),
                    # [p, hb % 4, c, I/2 words]: scale segment `segment` of each pair of this part, after its
                    # gate_up_cols weights
                    src=gate_up_w.ap(
                        pattern=[
                            [n_gate_up_slots * _GROUP * row_bytes // 2, _P_MAX],
                            [I // 2, _BLOCKS],
                            [row_bytes // 2, part_pairs],
                            [1, I // 2],
                        ],
                        offset=rows_base
                        + part * part_pairs * row_bytes // 2
                        + gate_up_cols
                        + segment * _BLOCKS * I // 2,
                    ),
                    engine=act,
                    name=f"relayout_scales_g{group}_s{segment}_part{part}",
                )
                if after_act != None:
                    copy.depends_on(after_act, kind=nisa.dependency.schedule)
                    after_act = None
                in_order(copy, act)
        if n_parts - 1 in runs:
            issue_stream(group)

    def gate_up_projection(
        group: int, units: Sequence[Tuple[int, int]], after: Optional[nisa.NkiInstruction]
    ) -> nisa.NkiInstruction:
        """Step 6.2: gate and up matmuls of group for the (sub-group, H512 block) `units`, chained after `after`.

        Returns the last matmul.
        """
        gate_up_ps = gate_up_ps_by_parity[group % 2]
        matmul = after
        for sub_group, block in units:
            sub_col = sub_group * sub_cols  # first (c, I) column of the sub-group
            for gate_or_up in range(2):
                # Moving columns [gate_or_up][block][c][I] of the sub-group in the moving slot of the group.
                weight_col = (
                    (group % n_moving_slots) * _GROUP * gate_up_cols
                    + (gate_or_up * n_h512 + block) * group_cols
                    + sub_col
                )
                # Scales of chunk k = gate_or_up * n_chunks + block / 4: segment k / 4, partition offset 4 (k % 4).
                scale_chunk = gate_or_up * n_chunks + block // _BLOCKS
                scale_col = (
                    ((group % n_moving_slots) * n_scale_segments + scale_chunk // _X4) * _BLOCKS * group_cols
                    + (block % _BLOCKS) * group_cols
                    + sub_col
                )
                psum_col = gate_or_up * up_col + sub_col
                previous = matmul
                matmul = nisa.nc_matmul_mx(
                    dst=gate_up_ps[:stationary_tokens, psum_col : psum_col + sub_cols],  # [t, (c, I)] of the sub-group
                    # [p, t]: H512 block `block` of all T tokens (the only reads of the hidden layout)
                    stationary=hidden_raw[:, block * stationary_tokens : (block + 1) * stationary_tokens],
                    moving=gate_up_moving.view(nl.float4_e2m1fn_x4)[:, weight_col : weight_col + sub_cols],
                    stationary_scale=hidden_raw_s[:, block * stationary_tokens : (block + 1) * stationary_tokens],
                    moving_scale=gate_up_moving_s.view(nl.uint8)[
                        _X4 * (scale_chunk % _X4) :, scale_col : scale_col + sub_cols
                    ],
                    accumulate=block > 0,
                    name=f"mm_{('gate', 'up')[gate_or_up]}_g{group}_b{block}"
                    + (f"_u{sub_group}" if n_sub_groups > 1 else ""),
                )
                if previous != None and (previous is after or not sparse_schedule):
                    matmul.depends_on(previous, kind=nisa.dependency.schedule)
        return matmul

    def swiglu(
        group: int,
        after_dve: Optional[nisa.NkiInstruction] = None,
        after_act: Optional[nisa.NkiInstruction] = None,
    ) -> Tuple[nisa.NkiInstruction, nisa.NkiInstruction]:
        """Step 6.3: gated[t, (q, c, p)] = SiLU(gate) * up of i = q*(I/4) + p, in bf16 (ACT SiLU, DVE multiply).

        The first DVE / ACT instruction is scheduled after after_dve / after_act. Returns the SiLU and the multiply.
        """
        if situ != None:
            return situ_activation(group, after_dve, after_act)
        gate_up_ps = gate_up_ps_by_parity[group % 2]
        gate_col = (group % 2) * group_cols  # gate_act slot of the group
        silu = nisa.activation(
            dst=gate_act[:, gate_col : gate_col + group_cols],
            op=nl.silu,
            data=gate_up_ps[:T, :group_cols],  # [t, (c, I)]: gate
            name=f"silu_g{group}",
        )
        if after_act != None:
            _instruction(silu).depends_on(after_act, kind=nisa.dependency.schedule)
        multiply = nisa.tensor_tensor(
            # [t, q, c, p]: gated slot of the group, rows p < I/4
            dst=gated.ap(
                pattern=[[2 * _X4 * _P_MAX, T], [_P_MAX, _X4], [_MX_ROWS, _GROUP], [1, down_rows]],
                offset=(group % 2) * _X4 * _P_MAX,
            ),
            # [t, q, c, p]: SiLU(gate) of pair c, i = q*(I/4) + p
            data1=gate_act.ap(
                pattern=[[2 * group_cols, T], [down_rows, _X4], [I, _GROUP], [1, down_rows]], offset=gate_col
            ),
            # [t, q, c, p]: up of pair c, i = q*(I/4) + p
            data2=gate_up_ps.ap(
                pattern=[[gate_up_row, T], [down_rows, _X4], [I, _GROUP], [1, down_rows]], offset=up_col
            ),
            op=nl.multiply,
            name=f"mul_up_g{group}",
        )
        if after_dve != None:
            _instruction(multiply).depends_on(after_dve, kind=nisa.dependency.schedule)
        return silu, multiply

    def situ_activation(
        group: int,
        after_dve: Optional[nisa.NkiInstruction] = None,
        after_act: Optional[nisa.NkiInstruction] = None,
    ) -> Tuple[nisa.NkiInstruction, nisa.NkiInstruction]:
        """Step 6.3 with SiTU: gated[t, (q, c, p)] = lb*tanh(u/lb) * beta*tanh(g/beta) * sigmoid(g), in FP32.

        ACT computes the tanh and sigmoid terms and DVE the products. The first DVE / ACT instruction is scheduled
        after after_dve / after_act. Returns the last ACT and the last DVE instruction.
        """
        beta, linear_beta = situ
        gate_up_ps = gate_up_ps_by_parity[group % 2]
        col = (group % 2) * group_cols  # slot of the group in gate_act, situ_sigmoid and situ_up
        gate_ps, up_ps = gate_up_ps[:T, :group_cols], gate_up_ps[:T, up_col : up_col + group_cols]
        gate_term, up_term = gate_act[:, col : col + group_cols], situ_up[:, col : col + group_cols]
        sigmoid_term = situ_sigmoid[:, col : col + group_cols]
        first_act = nisa.activation(
            dst=gate_term, op=nl.tanh, data=gate_ps, scale=1.0 / beta, name=f"tanh_gate_g{group}"
        )
        if after_act != None:
            _instruction(first_act).depends_on(after_act, kind=nisa.dependency.schedule)
        nisa.activation(dst=up_term, op=nl.tanh, data=up_ps, scale=1.0 / linear_beta, name=f"tanh_up_g{group}")
        # [t, q, c, p]: pair c, i = q*(I/4) + p, in a gate_act / situ_up slot, and in the gated slot of the group
        pairs = [[2 * group_cols, T], [down_rows, _X4], [I, _GROUP], [1, down_rows]]
        gated_pairs = gated.ap(
            pattern=[[2 * _X4 * _P_MAX, T], [_P_MAX, _X4], [_MX_ROWS, _GROUP], [1, down_rows]],
            offset=(group % 2) * _X4 * _P_MAX,
        )
        if sigmoid_from_tanh:
            # 2 sigmoid(g) = 1 + tanh(g/2): gate_term = (1 + tanh(g/2)) * tanh(g/beta), beta/2 goes to the up term.
            last_act = nisa.activation(
                dst=sigmoid_term, op=nl.tanh, data=gate_ps, scale=0.5, name=f"tanh_half_gate_g{group}"
            )
            first_dve = nisa.scalar_tensor_tensor(
                dst=gate_term,
                data=sigmoid_term,
                op0=nl.add,
                operand0=1.0,
                op1=nl.multiply,
                operand1=gate_term,
                name=f"situ_gate_g{group}",
            )
            if after_dve != None:
                _instruction(first_dve).depends_on(after_dve, kind=nisa.dependency.schedule)
            nisa.scalar_tensor_tensor(
                dst=up_term,
                data=up_term,
                op0=nl.multiply,
                operand0=linear_beta * beta / 2.0,
                op1=nl.multiply,
                operand1=gate_term,
                name=f"situ_g{group}",
            )
            multiply = nisa.tensor_copy(
                dst=gated_pairs, src=situ_up.ap(pattern=pairs, offset=col), engine=dve, name=f"mul_up_g{group}"
            )
            return last_act, multiply
        last_act = nisa.activation(dst=sigmoid_term, op=nl.sigmoid, data=gate_ps, name=f"sigmoid_gate_g{group}")
        # gate_term = tanh(g/beta) * beta*lb * sigmoid(g), then gated = tanh(u/lb) * gate_term
        first_dve = nisa.scalar_tensor_tensor(
            dst=gate_term,
            data=gate_term,
            op0=nl.multiply,
            operand0=beta * linear_beta,
            op1=nl.multiply,
            operand1=sigmoid_term,
            name=f"situ_gate_g{group}",
        )
        if after_dve != None:
            _instruction(first_dve).depends_on(after_dve, kind=nisa.dependency.schedule)
        multiply = nisa.tensor_tensor(
            dst=gated_pairs,
            data1=situ_up.ap(pattern=pairs, offset=col),
            data2=gate_act.ap(pattern=pairs, offset=col),
            op=nl.multiply,
            name=f"mul_up_g{group}",
        )
        return last_act, multiply

    def finish(group: int, after: nisa.NkiInstruction) -> Tuple[nisa.NkiInstruction, nisa.NkiInstruction]:
        """Step 6.4: the down-projection stationary of group (transposes chained after `after`).

        Returns the last transpose and the quantization.
        """
        if wide_out_ps:
            # gated columns of the tile
            gated_tile, ps_col = gate_up_ps_by_parity[group % 2], _PSUM_COLS - _X4 * stationary_tokens
        else:
            gated_tile, ps_col = gated_ps, (group % 2) * _X4 * stationary_tokens  # gated_ps slot of the group
        gated_row = gated_tile.shape[1]  # elements per partition of the tile
        transpose = after
        for q in range(_X4):
            transpose = nisa.nc_transpose(
                # [(c, p), t] of lane q (PSUM writes start on 4-byte boundaries: lanes stationary_tokens apart)
                dst=gated_tile[:, ps_col + q * stationary_tokens : ps_col + q * stationary_tokens + T],
                data=gated[:, ((group % 2) * _X4 + q) * _P_MAX : ((group % 2) * _X4 + q + 1) * _P_MAX],  # [t, (c, p)]
                name=f"transpose_gated_g{group}_q{q}",
            )
            # The compiler keeps the 4 transposes in program order, so only the first one needs the chain edge.
            if q == 0 and after != None:
                transpose.depends_on(after, kind=nisa.dependency.schedule)
        if post_down:
            # qmx_in[m, (k, t, q)] = act of row m, lane q, in the slot (k, t) of its pair t*K + k and zero in the other
            # slots: gated_ps[m, (q, t)] times slot_mask[m, (group, k, t)] (the mask also picks the row's own token).
            # The unscaled values are quantized into the pair-slot block of the group's stationary.
            slot_first = group * n_pairs * _X4
            nisa.tensor_tensor(
                dst=qmx_in[:, slot_first : slot_first + n_pairs * _X4],
                # [m, k, t, q]
                data1=gated_tile.ap(
                    pattern=[[gated_row, _P_MAX], [0, K], [1, T], [stationary_tokens, _X4]], offset=ps_col
                ),
                # [m, k, t, q]: slot_mask[m, (group, k, t)] for each q
                data2=slot_mask.ap(
                    pattern=[[n_groups * n_pairs, _P_MAX], [T, K], [1, T], [0, _X4]], offset=group * n_pairs
                ),
                op=nl.multiply,
                name=f"qmx_in_g{group}",
            )
            slot_col = (group + 1) * stationary_cols - n_pairs  # the pair-slot block of the group
            quantize = nisa.quantize_mx(
                dst=inter[:, slot_col : slot_col + n_pairs],
                src=qmx_in[:, slot_first : slot_first + n_pairs * _X4],
                dst_scale=inter_s[:, slot_col : slot_col + n_pairs],
                name=f"qmx_g{group}",
            )
            return transpose, quantize
        # Last token block of the group (of the last term) in qmx_in: [m, (t, q)] = v = row m of token t times its
        # row weight. Two scale groups: the block of each group s, with the row weights of s.
        group_first = group * group_stride  # first stationary column of the group
        lo_first = (group_first + stationary_cols - token_block) * _X4
        hi_first = (group_first + token_cols - token_block) * _X4
        set_step = stationary_cols * _X4  # qmx_in elements from one scale group to the next

        def blocks(tile, first):
            """The last-token-block values [m, (t, q)] of each scale group from element `first` of tile."""
            if n_scale_sets == 1:
                return tile[:, first : first + T * _X4]
            return tile.ap(pattern=[[qmx_in_row, _P_MAX], [set_step, 2], [_X4, T], [1, _X4]], offset=first)

        nisa.tensor_tensor(
            dst=blocks(qmx_in, lo_first),
            # [m, (t, q)] (two scale groups: [m, (s, t, q)])
            data1=gated_tile.ap(pattern=[[gated_row, _P_MAX], [1, T], [stationary_tokens, _X4]], offset=ps_col)
            if n_scale_sets == 1
            else gated_tile.ap(pattern=[[gated_row, _P_MAX], [0, 2], [1, T], [stationary_tokens, _X4]], offset=ps_col),
            # [m, (t, q)]: row_weights[m, (group, t)] for each q (two scale groups: row_weights[m, (s, group, t)])
            data2=row_weights.ap(pattern=[[n_groups * T, _P_MAX], [1, T], [0, _X4]], offset=group * T)
            if n_scale_sets == 1
            else row_weights.ap(
                pattern=[[2 * n_groups * T, _P_MAX], [n_groups * T, 2], [1, T], [0, _X4]], offset=group * T
            ),
            op=nl.multiply,
            name=f"qmx_in_g{group}",
        )
        if hi_lo:
            # hi = v with 4 significant bits (an AND of the bf16 bits, exact in e4m3) in the last token block of the
            # hi term, then lo = v - hi (exact in bf16) in place.
            nisa.tensor_scalar(
                dst=blocks(qmx_in.view(nl.int16), hi_first),
                data=blocks(qmx_in.view(nl.int16), lo_first),
                op0=nl.bitwise_and,
                operand0=_E4M3_BF16_MASK,
                engine=dve,
                name=f"qmx_hi_g{group}",
            )
            nisa.tensor_tensor(
                dst=blocks(qmx_in, lo_first),
                data1=blocks(qmx_in, lo_first),
                data2=blocks(qmx_in, hi_first),
                op=nl.subtract,
                engine=dve,
                name=f"qmx_lo_g{group}",
            )
        quantize = nisa.quantize_mx(
            dst=inter[:, group_first : group_first + group_stride],
            src=qmx_in[:, group_first * _X4 : (group_first + group_stride) * _X4],
            dst_scale=inter_s[:, group_first : group_first + group_stride],
            name=f"qmx_g{group}",
        )
        return transpose, quantize

    pending_sums = []  # post_down: evicted chunks of the last group whose token sum is not issued yet

    def sum_tokens(chunk: int, after: nisa.NkiInstruction) -> nisa.NkiInstruction:
        """post_down: add the K scaled pair rows of each token for one chunk on the PE (bf16 matmul with the 0/1
        token_sum stationary, chained after `after`), evict the [Hh * T, 512] sums and store them. Returns the
        matmul."""
        col = chunk * _DOWN_CHUNK
        sum_ps = sum_ps_ring[chunk % len(sum_ps_ring)]
        # x - x is 0 only for a finite x: the other values become exact zeros, so they stay out of every token sum.
        rows_in = out_sb[:, col : col + _DOWN_CHUNK]
        check = nisa.tensor_tensor(
            dst=sum_check, data1=rows_in, data2=rows_in, op=nl.subtract, engine=dve, name=f"check_rows_c{chunk}"
        )
        check.depends_on(after, kind=nisa.dependency.schedule)
        nisa.tensor_scalar(
            dst=sum_finite, data=sum_check, op0=nl.equal, operand0=0.0, engine=dve, name=f"finite_rows_c{chunk}"
        )
        nisa.memset(dst=sum_rows, value=0.0, engine=dve, name=f"zero_sum_rows_c{chunk}")
        nisa.tensor_copy_predicated(dst=sum_rows, src=rows_in, predicate=sum_finite, name=f"select_rows_c{chunk}")
        matmul = nisa.nc_matmul(
            dst=sum_ps,  # [h*T + t, 512]: sum over k of out_sb[h*T*K + T*k + t, chunk]
            stationary=token_sum[:stationary_cols],  # the Hh * T*K output rows
            moving=sum_rows,
            accumulate=False,
            name=f"mm_token_sum_c{chunk}",
        )
        matmul.depends_on(after, kind=nisa.dependency.schedule)
        nisa.tensor_copy(dst=out_tokens[:, col : col + _DOWN_CHUNK], src=sum_ps, engine=dve, name=f"evict_sum_c{chunk}")
        for half in range(n_out_halves):
            nisa.dma_copy(
                dst=output[
                    output_row : output_row + valid, half * out_cols + col : half * out_cols + col + _DOWN_CHUNK
                ],
                src=out_tokens[half * T : half * T + valid, col : col + _DOWN_CHUNK],  # [t, 512] of output half
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.sync,
                name=f"store_out_h{half}_c{chunk}",
            )
        return matmul

    def down_projection(group: int, chunks: Sequence[int], after: nisa.NkiInstruction) -> nisa.NkiInstruction:
        """Step 6.5: down projection of group for the H chunks `chunks`, chained after `after`.

        Output half h uses the stationary token blocks Hh - 1 - h .. Hh - 1 (post_down: pair-slot blocks; layout in
        the kernel docstring). For the last group, each chunk is evicted and stored while the next chunks compute: ACT
        (half the DVE rate) evicts every third chunk in eviction order except the last one, DVE the others. Without
        hi / lo, SP stores every chunk right after its eviction (the ACT queue is in order behind its evictions, which
        wait on the PE). With hi / lo, a dma_compute store adds the hi and lo rows. post_down: the eviction scales
        each pair row by its affinity, and sum_tokens adds the K rows of each token one chunk later. Returns the last
        matmul.
        """
        matmul = after
        for chunk in chunks:
            col = chunk * _DOWN_CHUNK  # first column of the chunk in each output half
            width = chunk_width(chunk)
            for half in reversed(range(n_out_halves)):
                h_col = half * out_cols + col  # first H column of the chunk
                # pre_quant: (half + 1) * T rows per term; post_down: (half + 1) * T*K pair rows
                out_rows = stationary_cols - (n_out_halves - 1 - half) * stationary_block
                # Two scale groups: one matmul per group s (its stationary and its down scales) into the same rows.
                for scale_set in range(n_scale_sets):
                    set_first = group * group_stride + scale_set * stationary_cols
                    # pre_quant: from token block Hh - 1 - half; post_down: from block Hh - 1 - half of T*K pair slots
                    first_block = set_first + (n_out_halves - 1 - half) * stationary_block
                    # scale rows of group g: tile g / 4 (scale group s at column s * H of the tile), from partition
                    # 4 (g % 4)
                    scale_col = ((group // _SCALE_GROUPS) * n_scale_sets + scale_set) * H + h_col
                    previous = matmul
                    matmul = nisa.nc_matmul_mx(
                        dst=out_ps[:out_rows, col : col + width],  # [h*T + t, 512], then the lo rows with hi / lo
                        # [(c, p), (term, j, t)]: from token block Hh - 1 - half (of the first term) to the end
                        stationary=inter[:, first_block : set_first + stationary_cols],
                        moving=down_w[:, group * H + h_col : group * H + h_col + width],  # [(c, p), 512]
                        stationary_scale=inter_s[:, first_block : set_first + stationary_cols],
                        moving_scale=down_s[_X4 * (group % _SCALE_GROUPS) :, scale_col : scale_col + width],
                        accumulate=group > 0 or half < n_out_halves - 1 or scale_set > 0,
                        name=f"mm_down_g{group}_h{half}_c{chunk}" + (f"_s{scale_set}" if n_scale_sets > 1 else ""),
                    )
                    if previous != None:
                        matmul.depends_on(previous, kind=nisa.dependency.schedule)
            if group == n_groups - 1:
                # The last chunk's eviction gates the final store, so it always goes to the faster DVE.
                position = chunk_order.index(chunk)
                on_dve = position % 3 != 0 or position == len(chunk_order) - 1
                if post_down:
                    # Each pair row times its affinity (per-partition scale) on the way out of PSUM.
                    if on_dve:
                        nisa.tensor_scalar(
                            dst=out_sb[:, col : col + _DOWN_CHUNK],
                            data=out_ps[:, col : col + _DOWN_CHUNK],
                            op0=nl.multiply,
                            operand0=aff_rows[:stationary_cols],  # the Hh * T*K output rows
                            engine=dve,
                            name=f"evict_c{chunk}",
                        )
                    else:
                        nisa.activation(
                            dst=out_sb[:, col : col + _DOWN_CHUNK],
                            op=nl.copy,
                            data=out_ps[:, col : col + _DOWN_CHUNK],
                            scale=aff_rows[:stationary_cols],
                            name=f"evict_c{chunk}",
                        )
                    # The token sum of the previous chunk follows this chunk's matmuls, so its eviction is done.
                    if pending_sums:
                        matmul = sum_tokens(pending_sums.pop(0), after=matmul)
                    pending_sums.append(chunk)
                    if position == len(chunk_order) - 1:
                        matmul = sum_tokens(pending_sums.pop(0), after=matmul)
                    continue
                if not on_dve and hooks != None and hooks.get("defer_act_evicts"):
                    # cross_call_order: this ACT eviction and its stores run in the next call, after its group-0
                    # re-layout (moe_sel_load_tp_finish).
                    hooks["deferred"].append(
                        lambda after, chunk=chunk, col=col, width=width: evict_store(
                            chunk, col, width, on_dve=False, after=after
                        )
                    )
                    continue
                evict_store(chunk, col, width, on_dve)
        return matmul

    def evict_store(
        chunk: int, col: int, width: int, on_dve: bool, after: Optional[nisa.NkiInstruction] = None
    ) -> None:
        """Step 6.5: evict chunk `chunk` of the last group from PSUM (DVE or ACT) and store it, the eviction scheduled
        after `after`."""
        if on_dve and n_groups == 1:
            # The other groups of a longer call add +0 to each output row; the same add keeps the bits.
            nisa.tensor_scalar(
                dst=out_sb[:, col : col + width],
                data=out_ps[:, col : col + width],
                op0=nl.add,
                operand0=0.0,
                engine=dve,
                name=f"evict_c{chunk}",
            )
        elif on_dve:
            nisa.tensor_copy(
                dst=out_sb[:, col : col + width],
                src=out_ps[:, col : col + width],
                engine=dve,
                name=f"evict_c{chunk}",
            )
        else:
            evict = nisa.activation(
                dst=out_sb[:, col : col + width],
                op=nl.copy,
                data=out_ps[:, col : col + width],
                engine=act,
                name=f"evict_c{chunk}",
            )
            if after != None:
                _instruction(evict).depends_on(after, kind=nisa.dependency.schedule)
        for half in range(n_out_halves):
            out_rows = slice(output_row, output_row + valid)
            if hi_lo:
                # The DMA adds the hi and lo rows of the output half while it stores them.
                nisa.dma_compute(
                    dst=output[out_rows, half * out_cols + col : half * out_cols + col + width],
                    srcs=[
                        out_sb[half * token_block : half * token_block + valid, col : col + width],  # hi
                        # lo [t, 512]
                        out_sb[
                            token_cols + half * token_block : token_cols + half * token_block + valid,
                            col : col + width,
                        ],
                    ],
                    reduce_op=nl.add,
                    name=f"store_out_h{half}_c{chunk}",
                )
            else:
                nisa.dma_copy(
                    dst=output[out_rows, half * out_cols + col : half * out_cols + col + width],
                    # [t, 512] of output half
                    src=out_sb[half * token_block : half * token_block + valid, col : col + width],
                    dge_mode=nisa.dge_mode.hwdge,
                    engine=nisa.engine.sync,
                    name=f"store_out_h{half}_c{chunk}",
                )

    # Step 7: Run the PE software pipeline (PE order in the kernel docstring).
    # Even chunks first, then odd: two chunks share a PSUM bank, so evicting chunk 2j never blocks the write of 2j + 1.
    chunk_order = list(range(0, n_down_chunks, 2)) + list(range(1, n_down_chunks, 2))
    # Gate/up units (sub-group, H512 block) in PE order: all blocks of sub-group 0, then of sub-group 1 (U = 2).
    units = [(sub_group, block) for sub_group in range(n_sub_groups) for block in range(n_h512)]
    # Units of gate/up(g + 1) before finish(g). long_stream with the inputs in SBUF finishes sooner, so the MXFP8 chain
    # of finish(g) runs before the next Pool descriptor window blocks DVE.
    finish_units = _LONG_FINISH_AFTER_UNITS if long and inputs_in_sbuf else len(units) // 2
    first_half = units[:finish_units]
    second_half = units[finish_units:]
    # Units of gate/up(g + 1) before down(g). long_stream runs them all first (not in the standalone kernel with
    # post_down): the activation of g + 1 then starts before down(g). The PSUM accumulation order does not change.
    all_first = long and (inputs_in_sbuf or not post_down)
    down_after = len(units) if all_first else len(units) * _DOWN_AFTER_SIXTEENTHS // 16
    # The second sub-group of group g + 1 is re-laid after SwiGLU(g): its rows may still be in flight, and finish(g)
    # needs SwiGLU(g) first (ACT and DVE run in order).
    later_runs = range(1, n_sub_groups)
    # The re-layout order of long_stream (relayout(g + 2) after finish(g), SwiGLU(g + 1) after it); relayout_order gives
    # it to every call with U = 1.
    ordered = long or (relayout_order and n_sub_groups == 1)
    # Step 7.1: Re-lay groups 0 (in two parts, while the second half of its rows is in flight) and 1, then gate/up(0).
    relayout(0, n_parts=2)
    if hooks != None and hooks.get("after_g0") != None:
        # cross_call_order: the previous call's deferred ACT evictions follow this call's group-0 re-layout on ACT.
        hooks["after_g0"](relayout_copies[act][-1] if relayout_copies[act] else None)
    if hwdge_group != None and inputs_in_sbuf:
        # long_stream: ACT loads the last experts of the HWDGE group after its group-0 copies (ACT order).
        for pair in range((hwdge_group + 1) * _GROUP - _LONG_ACT_HWDGE_LOADS, (hwdge_group + 1) * _GROUP):
            load = load_expert(pair, nisa.engine.scalar)
            if relayout_copies[act]:
                load.depends_on(relayout_copies[act][-1], kind=nisa.dependency.schedule)
            relayout_copies[act].append(load)
    if n_groups == 1:
        last = gate_up_projection(0, units, after=None)
        swiglu(0)
        last, _ = finish(0, after=last)
        down_projection(0, chunk_order, after=last)
        _chain_stream(stream, stream_chain)
        sbm.close_scope()
        return output
    relayout(1, n_parts=n_sub_groups, runs=[0])
    last = gate_up_projection(0, units, after=None)
    # long_stream with the inputs in SBUF and pre_quant: the group-0 activation follows the ACT copies of group 1, so
    # the compiler does not put it (it waits for gate/up(0)) ahead of them on ACT.
    act0_after = relayout_copies[act][-1] if long and inputs_in_sbuf and not post_down else None
    silu, multiply = swiglu(0, after_act=act0_after)
    if later_runs:
        relayout(1, n_parts=n_sub_groups, runs=later_runs, after_dve=multiply, after_act=silu)
    # Step 7.2: Gate/up of group g + 1 with finish(g) and down(g) inside it.
    for group in range(n_groups - 1):
        nxt = group + 1
        if nxt + 1 < n_groups and not later_runs and not ordered:
            relayout(nxt + 1, n_parts=1)
        # U = 1, last group: all of gate/up(last) before finish(g), so the PE does not wait for SwiGLU(g) and
        # SwiGLU(last) starts earlier. down(g) follows gate/up(last) in both orders.
        whole = nxt == n_groups - 1 and not later_runs
        last = gate_up_projection(nxt, units if whole else first_half, after=last)
        last, quantize = finish(group, after=last)
        relaid = None
        if nxt + 1 < n_groups and ordered:
            # long_stream: DVE re-lays group g + 2 after finish(g) and ACT after the activation of g, and the
            # activation of g + 1 follows the re-layout, so neither the MXFP8 chain nor the activation waits for late
            # rows. The HWDGE group arrives one expert at a time: two runs.
            relayout(nxt + 1, n_parts=2 if nxt + 1 == hwdge_group else 1, after_dve=quantize, after_act=silu)
            relaid = (relayout_copies[dve][-1], relayout_copies[act][-1])
        if nxt + 1 < n_groups and later_runs:
            # U = 2: DVE re-lays group g + 2 after finish(g), because down(g) waits for finish(g) while the rows of
            # g + 2 may still be in flight.
            relayout(nxt + 1, n_parts=n_sub_groups, runs=[0], after_dve=quantize)
        if group == 0 or nxt == n_groups - 1:
            last = gate_up_projection(nxt, [] if whole else second_half, after=last)
        else:
            last = gate_up_projection(nxt, units[finish_units:down_after], after=last)
            last = down_projection(group, chunk_order, after=last)
            last = gate_up_projection(nxt, units[down_after:], after=last)
        # whole: DVE quantizes finish(g) before SwiGLU(last), so down(g) does not wait behind it.
        if relaid != None:
            silu, multiply = swiglu(nxt, after_dve=relaid[0], after_act=relaid[1])
        else:
            silu, multiply = swiglu(nxt, after_dve=quantize if whole else None)
        if nxt + 1 < n_groups and later_runs:
            relayout(nxt + 1, n_parts=n_sub_groups, runs=later_runs, after_dve=multiply, after_act=silu)
        if group == 0 and n_groups > 2:
            # The group-0 down weights arrive after the gate/up rows of group 1 (two groups: Step 7.3 runs down(0)).
            last = down_projection(group, chunk_order, after=last)
    # Step 7.3: down(second last) with finish(last) before its last chunks, then down(last).
    # The SiLU * up of the last group runs while the PE does the first chunks; the last chunks cover its MXFP8 chain.
    second_last, final = n_groups - 2, n_groups - 1
    finish_after = max(2, n_down_chunks - _LAST_FINISH_CHUNKS_LEFT)
    last = down_projection(second_last, chunk_order[:finish_after], after=last)
    last, _ = finish(final, after=last)
    last = down_projection(second_last, chunk_order[finish_after:], after=last)
    down_projection(final, chunk_order, after=last)

    _chain_stream(stream, stream_chain)
    sbm.close_scope()
    return output


def _chain_stream(stream: list, stream_chain: Optional[list]) -> None:
    """Step 7.4: Pool runs the stream in order, so nothing delays the first gathers. The stream of this call follows
    the last gather of stream_chain, and its last gather goes to stream_chain."""
    for prev, nxt in zip(stream, stream[1:]):
        nxt.depends_on(prev, kind=nisa.dependency.schedule)
    if stream_chain != None:
        if stream_chain:
            stream[0].depends_on(stream_chain[-1], kind=nisa.dependency.schedule)
        stream_chain.append(stream[-1])
