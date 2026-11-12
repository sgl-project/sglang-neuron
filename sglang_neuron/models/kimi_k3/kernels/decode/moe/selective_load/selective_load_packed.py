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
"""

import inspect
from typing import Callable, Generator, Optional, Sequence, Tuple

import nki
import nki.isa as nisa
import nki.language as nl

from ...utils.allocator import SbufManager, align_to
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import div_ceil
from ...utils.logging import get_logger

_ISA, _LANGUAGE = nisa, nl


class _Suffixed:
    """A view of a module whose functions (or only the listed ones) append ``suffix`` to their ``name`` argument.

    Op and tensor names must be unique in a kernel, so a kernel that calls the body more than once gives each call its
    own name suffix (name_suffix of selective_load_packed_phases)."""

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


def _named_modules(name_suffix: Optional[str]):
    """(nisa, nl) for the body: the modules themselves without a suffix, else views that append name_suffix to every
    op and tensor name."""
    if not name_suffix:
        return _ISA, _LANGUAGE
    return _Suffixed(_ISA, name_suffix), _Suffixed(_LANGUAGE, name_suffix, functions=("ndarray",))


_P_MAX = 128
_QUADRANT = 32
_X4 = 4
_MX_ROWS = 8
_BLOCKS = 4  # H512 blocks per gate/up scale chunk
_GROUP = 16  # routed pairs per group
_SCALE_GROUPS = 4  # groups per down-scale tile (an MX scale tile starts at partition 0, 4, 8 or 12 of a quadrant)
_SKIP = 1 << 20  # gather index offset that makes a row out of bounds, so the gather skips it
# Expert id offset that puts every gate/up (128 rows) and down (8 rows) row of the expert out of bounds.
_SKIP_EXPERT = _SKIP // 128
_PSUM_COLS = 512  # half of a bf16 gate/up PSUM bank: gate at column 0, up at column _PSUM_COLS
_DOWN_CHUNK = 512  # H columns per down-projection matmul (moving down weights)
_DOWN_AFTER_SIXTEENTHS = 13  # down(g) runs inside gate/up(g + 1) after 13/16 of its H512 blocks
_LAST_FINISH_CHUNKS_LEFT = 3  # down chunks of the second-to-last group left after the last group's finish
# Long stream (8 groups, U = 1; Step 1.1): one more group comes from per-expert HWDGE loads (SP queue; group 3 in the
# standalone kernel with pre_quant, else group 2), ACT copies the first H512 blocks of each re-layout copy of groups
# 1.. (DVE the rest), and in the fused block finish(g) runs after this many gate/up units of group g + 1.
_LONG_HWDGE_GROUP = 2
_LONG_HWDGE_GROUP_STANDALONE = 3
_LONG_ACT_HWDGE_LOADS = 2  # fused block: the last loads of the HWDGE group go on ACT, after the group-0 re-layout
_LONG_ACT_RELAYOUT_BLOCKS = 4
_LONG_FINISH_AFTER_UNITS = 2
_LATE_DOWN = 5  # standalone long stream: late_down default (down(4) and down(5) run together after gate/up(6))
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
# dedupe_experts="dynamic": expert id of the padding slots (any id >= E puts the gather rows out of bounds; fp16
# holds id - _DYN_PAD exactly for 0 <= id < _DYN_PAD).
_DYN_PAD = 2048
# hidden_input="bf16" (quantize_latent_mx): the latent copies sit in this many partition blocks (fewer DMAs, more SBUF).
_LATENT_PARTS = 2
# sbuf_reserve: SBUF bytes per partition that the body allocates after Step 1.3 (the hi mask tile, alignment).
_LATE_SBUF_BYTES = 64


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
    down_after_units: Optional[int] = None,
    silu0_after_relayout: Optional[bool] = None,
    pe_keep_alive: int = 0,
    pe_keep_alive_cols: int = 512,
    pe_keep_alive_tail: int = 0,
    dedupe_experts: Optional[str] = None,
    dyn_head_groups: int = 6,
    dyn_verbatim_groups: int = 1,
    dyn_tail_loads: str = "sp",
    dyn_pad_rows: str = "expert0",
    dyn_tail_gate: Optional[int] = None,
    dyn_fast_g1: bool = False,
    merged_down_gathers: Optional[bool] = None,
    merged_down_pairs: Optional[Tuple[int, ...]] = None,
    down_real_rows: bool = True,
    dyn_direct_g1: bool = False,
    dyn_g1_dve_first: bool = False,
    dyn_late_relayout: bool = False,
    dyn_merged_stores: bool = False,
    hidden_queue: str = "sp",
    early_gather: Optional[bool] = None,
    merged_scales: Optional[bool] = None,
    late_down: Optional[int] = None,
    last_relayout: Optional[str] = "auto",
    activation: str = "silu",
    beta: float = 4.0,
    linear_beta: float = 25.0,
    sigmoid_from_tanh: bool = False,
    affinity_hi_lo: Optional[bool] = None,
    hidden_input: str = "mx",
    affinity_layout: str = "dense",
    fp32_down_rounding: str = "terms",
    latent_quant: str = "transpose",
    weight_sbuf_budget: Optional[int] = None,
    dyn_head_store: bool = False,
) -> nl.NkiTensor:
    """Selective-expert MoE MLP (SiLU(gate) * up, no bias) that packs 16 routed experts into each matmul.

    Optimized for decode on Trn3 (LNC1): T*K = 64 or 128 routed pairs, a small intermediate size and MXFP4 weights.
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
            32 = 8 rows x 4 lanes). Row p, lane q holds i = q*(I/4) + p; rows p >= I/4 are not read.
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
            selective_load_packed_body). At H = 3584, "fp32" uses two uneven output halves (2048 + 1536 columns);
            with hi / lo the groups of a token then add in fp32 and the output rounds once.
        gate_up_psum_dtype (str): the PSUM dtype of the gate and up projections, "bf16" (default) or "fp32" (see
            selective_load_packed_body). At H = 3584 with I = 24, "fp32" asserts: it needs 2 output halves, and H / 2
            is not a multiple of 512.
        down_after_units (Optional[int]): gate/up units of group g + 1 that run before down(g) (see
            selective_load_packed_body). None (default): all units with K = 16 (8 groups) and pre_quant, else
            13/16 of them (post_down keeps 13/16 in the standalone kernel).
        silu0_after_relayout, pe_keep_alive, pe_keep_alive_cols, pe_keep_alive_tail: schedule options of the body
            (see selective_load_packed_body). In the standalone kernel they are off by default.
        dedupe_experts (Optional[str]): repeated experts (see selective_load_packed_body). None (default): every
            pair loads and computes its expert. "static": a pair whose expert an earlier pair loads skips its weight
            loads (same schedule). "dynamic": the repeated experts are compacted away and the tail groups that no
            unique expert fills are skipped (fast with the barrier-free loop compiler).
        dyn_head_groups, dyn_verbatim_groups, dyn_tail_loads, dyn_pad_rows, dyn_tail_gate, dyn_fast_g1,
            dyn_direct_g1, dyn_g1_dve_first, dyn_late_relayout, dyn_merged_stores, dyn_head_store: options of
            dedupe_experts="dynamic" (see selective_load_packed_body).
        merged_down_gathers (Optional[bool]): one Pool gather loads the down weights of two groups (see
            selective_load_packed_body). None (default): on, except with dedupe_experts="static" (which needs one
            down gather per group). With down_real_rows (the default), the pair is faster (IRAM 64) at the
            tested shapes; without it the merged gathers are slower here (their out-of-bounds rows are slow in the
            DMA). The outputs are the same.
        merged_down_pairs (Optional[Tuple[int, ...]]): the first groups of the merged pairs (see
            selective_load_packed_body). Default None (all pairs).
        down_real_rows (bool): the down gathers load real rows instead of out-of-bounds rows (see
            selective_load_packed_body). Default True. The outputs are the same.
        hidden_queue (str): the HWDGE queue of the hidden loads (see selective_load_packed_body). Default "sp" (every
            mode); "act" is the previous queue.
        early_gather, merged_scales, late_down, last_relayout: schedule options of the standalone kernel (see
            selective_load_packed_body). Default None / "auto": early_gather=True, merged_scales=True, late_down=5
            and last_relayout="dve" in the long stream (8 groups, one sub-group) with pre_quant and no
            dedupe_experts, else off. Together with hidden_queue="sp" they are faster (IRAM 64 and the default
            runtime) at the tested shape; one at a time they gain little or lose. The outputs are the same.

        The options below select the semantics of a latent MoE with SiTU experts and sigmoid top-K routing (all off
        by default; together they give the inputs and the numerics of a selective-load routed-expert call on a bf16
        latent with [T, K] affinities):
        activation (str): "silu" (default): SiLU(gate) * up. "situ": the SiTU activation
            beta*tanh(g/beta)*sigmoid(g) * linear_beta*tanh(u/linear_beta), in fp32 (see selective_load_packed_body,
            situ).
        beta, linear_beta (float): the SiTU gate and up soft-clipping factors (default 4.0 and 25.0), with
            activation="situ".
        sigmoid_from_tanh (bool): with activation="situ", compute sigmoid(g) as (1 + tanh(g/2)) / 2, so that ACT
            uses only the tanh table. Default False.
        affinity_hi_lo (Optional[bool]): carry the affinity-scaled intermediate as two MXFP8 terms, hi + lo (see
            selective_load_packed_body). None (default): one term. True: hi + lo (about 8 significant
            bits), with the hi mask as a tile (hi_mask_tile; same output).
        hidden_input (str): "mx" (default): hidden_qtz / hidden_scale are the MXFP8 hidden as above. "bf16":
            hidden_qtz is the bfloat16 latent [T, H] @HBM in its natural layout and hidden_scale is unused (None). The
            kernel quantizes it to MXFP8 without a norm: lane q of row p of H512 block hb holds latent column
            512*hb + 4*p + q, so each MX group holds 32 consecutive latent values of one token. The gate/up weights
            must use the same order: row p, block hb, lane q of expert_gate_up_packed is hidden index 512*hb + 4*p + q.
        affinity_layout (str): "dense" (default): expert_affinities is float32 [T, E] as above. "topk":
            expert_affinities is float32 [T, K], the affinity of the k-th expert of each token in the order of
            expert_index (pair n = t*K + k). Each affinity rounds to bf16 once, as with "dense".
        fp32_down_rounding (str): with down_psum_dtype="fp32" and hi / lo, the rounding of the output (see
            selective_load_packed_body). "terms" (default): the hi and lo sums round to bf16 one at a time and the
            store adds them, the rounding of a single-group bf16 PSUM. "once": the store adds the fp32 sums and rounds
            once.
        latent_quant (str): with hidden_input="bf16", how the latent reaches the partitions (see quantize_latent_mx).
            "transpose" (default): 4 PE transposes per H512 block of the stride-4 column views. "matmul": 4 shifted
            copies of the latent and one identity matmul per H512 block (4x fewer PE instructions, 8 latent loads on
            SP and ACT ahead of the expert loads; slower here). The same output.
        weight_sbuf_budget (Optional[int]): SBUF bytes per partition for the weight rings (see
            selective_load_packed_body). None (default): 208 KB. A smaller budget leaves SBUF free for the caller (for
            example a 32 KB reserve); the output does not change.

    Returns:
        output (nl.NkiTensor): bfloat16 [T, H] @HBM.

    Notes:
        - POST_SCALE affinities. With pre_quant, the kernel multiplies SiLU(gate) * up by the affinity, rounds the
          product to fp16 and then quantizes it to MXFP8. This changes the MXFP8 rounding relative to a scale after
          the down projection. The fused MoE block carries the product as two MXFP8 terms, hi + lo (about 8
          significant bits), which keeps it close to the reference (selective_load_packed_body, affinity_hi_lo).
        - Requires LNC1, T*K == 64 or 128, 16 % K == 0, H % 512 == 0, 2048 <= H <= 8192, I % 4 == 0, I <= 32 and
          E % 128 == 0. post_down needs T == 8 and K = 8 or 16. The expert ids go through fp16 PE broadcasts,
          which are exact for E <= 2048.
        - selective_load_packed_body holds the code. The fused MoE block calls it with SBUF inputs.

    Algorithm:
        The N pairs form N/16 groups of 16 (16/K tokens per group; one token per group at K = 16). The group is the
        unit of work: the gate (or up) columns of 16 experts fill one moving tile, and 16 experts x 8 MX rows fill
        the 128 partitions of one down-projection stationary. The code comments in selective_load_packed_body use
        the step numbers below.

        Step 1: Plan the buffers.
            Step 1.1: Get the sizes from the tensor shapes and check the constraints. Long stream: N/16 = 8 groups
                with U = 1 (K = 16). Then the Pool gathers limit the block, and Steps 2.3, 6.1 and 7.2 use the
                long-stream tuning.
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
                (weights and scales) with one HWDGE DMA. The M loads alternate the SP and ACT queues. Long stream:
                the SP queue then loads the 16 experts of one more group the same way (group 3 in the standalone
                kernel with pre_quant, else group 2), so Pool has one gather less. In the fused block ACT loads the
                last 2 of them after its group-0 re-layout copies (Step 7.1).
            Step 2.4: Standalone only: load the affinities, and broadcast the index row to every partition with a
                ones matmul. DVE copies the broadcast index before its other early work.
            Step 2.5: hidden_input="bf16" only: load the bf16 latent as 4 copies shifted by one column each and
                quantize it to the MXFP8 stationary (one identity matmul per H512 block, one DVE swizzle copy,
                quantize_mx; see quantize_latent_mx).
        Step 3: Build the gather indices and the constants.
            Step 3.0: dedupe_experts="dynamic" only: compact the unique experts into slots (pair order, padding at
                the end) and count them (U). The gathers and the HWDGE loads read the compacted ids, and the tail
                groups get trip counts (ceil(U / 16) == G) for Step 7.5.
            Step 3.1: Get the bit fields of the partition index a.
            Step 3.2: Get the pair selectors (the pair that feeds partition a in each gather).
            Step 3.3: Get the source rows inside the expert (lanes). Rows that do not exist get +_SKIP.
            Step 3.4: Build the constants: the block-diagonal mask, the zero rows of gated, the expert ids, the
                ones stationary and the token rows (post_down: the pair-slot masks and the token-sum stationary).
            Step 3.5: Gather the expert of each selected pair from the index, then form gather_index =
                rows per expert * expert + source row.
            Step 3.6: dedupe_experts only: count the earlier pairs with the same expert for each pair. Two matmuls
                with 0/1 pair selectors move the count of the pair that feeds each gather row to that row, and the
                row gets + _SKIP per earlier pair (gather_index_dedupe), so the gathers of the later sub-groups and
                groups skip a repeated expert. The HWDGE group reads index_dedupe (+ _SKIP_EXPERT per earlier
                pair). The row slots of these groups get zero scale segments first.
        Step 4: Start the Pool weight stream. The batched SWDGE gathers run in PE consumption order: gate/up g1,
            down scales (groups 0-3), down g0, gate/up g2, down g1, ..., gate/up g(last), then the remaining down
            weights. The scale tile of groups 4-7 goes right before down g4. Long stream: no gate/up gather for
            the HWDGE group of Step 2.3. With U = 2 every gate/up
            gather is split into its 2 sub-groups, and the second sub-group of g0 goes first. A gather into a reused
            gate/up row slot waits for the re-layout of the slot's previous group (Step 6.1). All other gathers start
            here.
        Step 5: Build the per-pair affinities while the PE is idle.
            Step 5.1: Standalone only: transpose the affinities to [expert on partitions, t]. Keep column n where
                index[n] is the expert of the partition (one-hot), and sum over the experts with a ones matmul.
                The result is the affinity of pair n on every partition. affinity_layout="topk": a ones matmul
                broadcasts the [T, K] affinity row instead.
            Step 5.2: Keep the affinity of pair n only on its own 8 rows (affinity_mask). Sum it to one value per
                row and group (affinity_rows), then spread it over the tokens (row_weights). post_down: get the
                affinity of the pair of each output row (aff_rows) instead. dedupe_experts: the rows of the first
                pair of each expert get the affinity of every token that routes to the expert, and the rows of a
                repeated pair get zero.
        Step 6: Define the per-group stages. Step 7 calls them.
            Step 6.1: relayout(g, runs): DVE copies the gate/up weights of the pair runs `runs` of group g from the
                packed rows to the moving layout. ACT copies their scale segments. After the last run, the stream
                gathers that reuse the row slot of g start. Long stream: for g > 0, ACT copies the first 4 H512
                blocks of the gate and of the up weights, and DVE the other blocks.
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
                (dma_compute). post_down: half h of pair t*K + k lands on output row h*N + 8k + t. The eviction of
                the last group scales each row by its affinity, and one chunk later a bf16 matmul with the 0/1
                token_sum stationary adds the K rows of each token before the store.
        Step 7: Run the PE software pipeline. Each PE instruction is chained after the previous one.
            Step 7.1: Re-lay groups 0 (in two parts) and 1, then run gate/up and SwiGLU of group 0. Fused long
                stream: ACT loads the last 2 experts of the HWDGE group after its group-0 copies, and with
                pre_quant SiLU(0) runs after the ACT copies of group 1 (silu0_after_relayout). Option: PE
                keep-alive matmuls before gate/up(0) (pe_keep_alive).
            Step 7.2: For g = 0 .. N/16 - 2: re-lay group g + 2 (if it exists), run the first half of gate/up(g + 1),
                finish(g), the rest of gate/up(g + 1) and its SwiGLU. down(0) follows gate/up(1). down(g) for 0 < g <
                N/16 - 2 runs inside gate/up(g + 1) after 13/16 of its units (long stream: after all of them, except the
                standalone kernel with post_down). down(N/16 - 2) waits for Step 7.3. With U = 2 the rows of a group may
                still be in flight when DVE / ACT reach them, so DVE re-lays the first sub-group of group g + 2 after
                finish(g), and the second sub-group of group g + 1 after SwiGLU(g) (ACT and DVE run in order). Long
                stream: the re-layout of group g + 2 runs after finish(g) on DVE and after SiLU(g) on ACT, before
                SwiGLU(g + 1); in the fused block finish(g) follows the first 2 units of gate/up(g + 1).
            Step 7.3: Run down(second last) with finish(last) before its last 3 chunks, then down(last) with the
                output stores.
            Step 7.4: Chain the Pool stream gathers in program order.
            Step 7.5: dedupe_experts="dynamic" only: the tail blocks (nl.fori_loop, trip count 0 or 1) run the groups
                after dyn_head_groups for the actual number of unique-expert groups.
            Step 7.6: dedupe_experts="dynamic" only: store the output (out_sb) after the tails.

        PE order for 4 groups; 8 groups repeat the middle stage (gate_up(g)[a:b] = (sub-group, H512 block) units
        a..b-1, h = half of the units (fused long stream: 2); d = 13/16 of the units (long stream: all of them,
        except the standalone kernel with post_down); with U = 2 the units are all blocks of sub-group 0, then all
        blocks of sub-group 1):
            gate_up(0) | gate_up(1)[:h] finish(0) gate_up(1)[h:] down(0)
            | gate_up(2)[:h] finish(1) gate_up(2)[h:d] down(1) gate_up(2)[d:]
            | gate_up(3)[:h] finish(2) gate_up(3)[h:] down(2)[first chunks] finish(3) down(2)[last 3] down(3)

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
        - hidden_raw / hidden_raw_s [128, (H/512) * T]: hidden_qtz / hidden_scale, column hb*T + t. Block hb of all
          tokens is the contiguous gate/up stationary [:, hb*T : (hb + 1)*T].
        - index_row int32 [1, N]: expert_index as one row on partition 0 (column n = pair n).
        - index int32 [128, N]: index[a, n] = the expert of pair n, on every partition a.
        - partition int32 [128, 7]: a, a%8, a%16, a/8, a/16, the down-scale pair, 4*(a/32) (Step 3.1).
        - select / experts int32 [128, G + N/16 + St] (G gathered gate/up sub-groups: 1 .. N/M - 1, without the
          HWDGE group of the long stream): the pair that feeds partition a / its expert. Column i: gate/up of the i-th
          gathered sub-group. Columns G .. G + N/16 - 1: down weights of group g. Columns G + N/16 + s: down-scale
          tile s.
        - lanes int32 [128, 18]: the source row inside the expert. Columns 0..M-1: gate/up row for index column b.
          Column 16: down-weight row. Column 17: down-scale row.
        - gather_index int32 [128, G*M + N/16 + St]: columns iM + b: gate/up of the i-th gathered sub-group. Column
          G*M + g: down weights of group g. Column G*M + N/16 + s: down-scale tile s. Each index is rows per expert *
          expert + source row.
        - Gather rows: a batched SWDGE gather with an index tile [128, M] uses the index at partition a, column b
          for gathered row r = 128b + a, and writes row r to partition r/M, slot r%M. So the pair that feeds row r
          is a bit-field function of a:
          gate/up of sub-group j (M index columns, 128 rows per expert): row r lands on partition (128/M)b + a/M,
          slot a%M. The pair is Mj + a%M and the source row is (128/M)b + a/M (M = 16: 8b + a/16; M = 8:
          16b + a/8).
          down weights of group g (M = 1, 8 rows per expert): partition a = 8c + p holds row p of pair c = a/8. The
          pair is 16g + a/8 and the source row is a%8. Rows p >= I/4 get +_SKIP. With merged_down_gathers, column
          2q + b of the down-weight indices feeds the merged gather of groups 2q, 2q + 1 (see
          selective_load_packed_body).
          down-scale tile s (M = 1, 1 row per expert): partition a = 32q + 4(g%4) + j holds the scale row of pair
          c = 4q + j of group g = 4s + (a/4 % 4) (a%32 < 16 only). The pair is 64s + 16(a/4 % 4) + 4(a/32) + a%4.
          Partitions with a/16 odd get +_SKIP.
        - masks bf16 [128, 16]: masks[m, c] = 1 if 0 <= m - 8c < 8 (block diagonal).
        - token_rows bf16 [128, (N/16) * T]: [m, (g, t)] = 1 if pair 16g + m/8 belongs to token t.
        - expert_id fp32 [128, E/128]: [a, h] = a + 128h. affinity_hot bf16 [128, (E/128) * N]: tile h, column n
          = the affinity of pair n if index[n] == a + 128h, else 0. ones bf16 [128, 128]: the stationary that sums
          over the partitions.
        - affinity bf16 [128, N]: the affinity of pair n on every partition. affinity_mask bf16 [128, N]: the
          same, only on rows m = 8(n%16) + p. affinity_rows bf16 [128, N/16]: [m, g] = the affinity of pair
          16g + m/8. row_weights bf16 [128, (N/16) * T] = affinity_rows * token_rows.
          All exact in bf16 (0/1 values times the bf16 affinity, one nonzero term per row).
        - gate_act fp32 [T, 2 * 16 * I]: ring of 2 groups, SiLU(gate) of group g at columns (g%2) * 16 * I.
        - gated bf16 [T, 2 * 4 * 128]: ring of 2 groups, group g at columns 512 (g%2), laid out [q][c][p]: SiLU(gate)
          * up of i = q*(I/4) + p. Rows p >= I/4 stay zero.
        - qmx_in [128, (N/16) * Q * 4], bf16 with hi / lo or post_down, else fp16: per group the values to quantize, q
          innermost so quantize_mx packs the 4 lanes. pre_quant: Q = Hh * T per term, [m, (term, j, t, q)] (term 0 =
          hi, 1 = lo with hi / lo). Token block j = Hh - 1 of each term holds the values; block j = 0 of Hh = 2 stays
          zero. post_down: Q = N, [m, (k, t, q)]: the value of row m in the slot of its pair t*K + k, else 0.
        - inter float8_e4m3fn_x4 / inter_s uint8 [128, (N/16) * Ns]: per group the down-projection stationary (Ns
          columns) and its MX scales. pre_quant: Hh * T columns per term, [m, (term, j, t)]. post_down: Ns = Hh * N,
          zero except the N pair-slot columns [m, (k, t)] at the end of the group.
        - out_sb bf16 [Ns, out_cols]: the evicted output, rows as in out_ps. post_down: out_tokens bf16
          [Hh * T, out_cols] holds the token sums, token t of output half h on partition h*T + t.
        - post_down constants: slot_mask bf16 [128, (N/16) * N]: [m, (g, k, t)] = 1 if row m of group g is pair
          t*K + k. aff_rows fp32 [128, 1]: the affinity of the pair of output row r. token_sum bf16 [128, Hh * T]:
          [r, h*T + t] = 1 if output row r is h*N + 8k + t for some k.
        PSUM (8 banks of 2 KB per partition):
        - gate_up_ps bf16 [T, 1024], banks 0 and 1: group g in bank g%2, [t, (c, I)]. Gate at column 0, up at 512.
          fp32 changes this plan (see gate_up_psum_dtype in selective_load_packed_body).
        - out_ps bf16 [Ns, out_cols], banks 2-5: the output, accumulated over the groups. pre_quant: output half h
          of token t is row h*T + t (the lo term of hi / lo: row Hh*T + h*T + t, which the store DMA adds to the hi
          row). Half h uses the stationary from token block Hh - 1 - h of the first term to the end of the group, so
          its values land on its own rows and the zero blocks add 0 to the other rows. post_down: half h of pair
          t*K + k is row h*N + 8k + t. The widest half goes first, so group 0 writes every row. fp32 changes this
          plan (see down_psum_dtype in selective_load_packed_body).
        - index_ps fp32 [128, N] at bank 6, affinity_ps bf16 [128, N] at bank 6 + 1 KB and affinity_t_ps fp32
          [128, (E/128) * T] at bank 6 + 1.5 KB. routing_ps fp32 [128, 2N] at bank 6 (router tiles only).
          dedupe_experts: dedupe_ps fp32 [128, N] at bank 6 + 512 B, row_weights_ps fp32 [128, (N/16) * T] at
          bank 6 + 1280 B, column_counts_ps fp32 [128, G - 1 + N/16 - 1] at bank 7 + 1 KB.
        - gated_ps bf16 [128, 2 * 4 * T], bank 7: group g at columns 4T (g%2), [(c, p), (q, t)].
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
        - Long stream (8 groups). Each gate/up gather costs Pool time for its descriptors, and Pool is the
          bottleneck. SP issues its HWDGE expert loads quickly and is free after group 0, so it loads one more
          group: the first group with enough slack (group 3 in the standalone kernel with pre_quant, else group 2;
          group 1 would arrive too late). While Pool writes the descriptors of a gate/up gather, DVE SBUF access is
          blocked or slow; ACT is not. So ACT does more than half of each re-layout, and in the fused block finish(g)
          starts after 2 units of gate/up(g + 1): its MXFP8 chain then runs before the next descriptor window. With
          3 units, some executions lose time when that chain falls into a window. down(g) follows all units of
          gate/up(g + 1), so SwiGLU(g + 1) runs during down(g) and the PE does not wait for the SwiGLU and MXFP8
          chains (the groups in the middle then run at the PE rate).
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
    kernel_assert(
        nl.program_ndim() == 0 or (nl.program_ndim() == 1 and nl.num_programs(0) == 1),
        "selective_load_packed supports LNC1 only",
    )
    kernel_assert(nisa.get_nc_version() >= nisa.nc_version.gen4, "selective_load_packed requires Trn3 or newer")
    kernel_assert(activation in ("silu", "situ"), f"activation must be 'silu' or 'situ', got {activation}")
    sbm = SbufManager(0, nl.tile_size.total_available_sbuf_size, get_logger("selective_load_packed"))
    # The semantics options are passed only when they are on, so the default call is unchanged.
    semantics = {}
    if activation == "situ":
        semantics.update(situ=(beta, linear_beta), sigmoid_from_tanh=sigmoid_from_tanh)
    if affinity_hi_lo != None:
        semantics.update(affinity_hi_lo=affinity_hi_lo, hi_mask_tile=affinity_hi_lo)
    if hidden_input != "mx":
        semantics.update(hidden_input=hidden_input)
    if affinity_layout != "dense":
        semantics.update(affinity_layout=affinity_layout)
    if fp32_down_rounding != "terms":
        semantics.update(fp32_down_rounding=fp32_down_rounding)
    if latent_quant != "transpose":
        semantics.update(latent_quant=latent_quant)
    if weight_sbuf_budget != None:
        semantics.update(weight_sbuf_budget=weight_sbuf_budget)
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
        down_after_units=down_after_units,
        silu0_after_relayout=silu0_after_relayout,
        pe_keep_alive=pe_keep_alive,
        pe_keep_alive_cols=pe_keep_alive_cols,
        pe_keep_alive_tail=pe_keep_alive_tail,
        dedupe_experts=dedupe_experts,
        dyn_head_groups=dyn_head_groups,
        dyn_verbatim_groups=dyn_verbatim_groups,
        dyn_tail_loads=dyn_tail_loads,
        dyn_pad_rows=dyn_pad_rows,
        dyn_tail_gate=dyn_tail_gate,
        dyn_fast_g1=dyn_fast_g1,
        merged_down_gathers=dedupe_experts != "static" if merged_down_gathers is None else merged_down_gathers,
        merged_down_pairs=merged_down_pairs,
        down_real_rows=down_real_rows,
        dyn_direct_g1=dyn_direct_g1,
        dyn_g1_dve_first=dyn_g1_dve_first,
        dyn_late_relayout=dyn_late_relayout,
        dyn_merged_stores=dyn_merged_stores,
        dyn_head_store=dyn_head_store,
        hidden_queue=hidden_queue,
        early_gather=early_gather,
        merged_scales=merged_scales,
        late_down=late_down,
        last_relayout=last_relayout,
        **semantics,
    )


def selective_load_packed_body(*args, **kwargs) -> nl.NkiTensor:
    """Run every phase of selective_load_packed_phases (same arguments) and return the output."""
    return finish_phases(selective_load_packed_phases(*args, **kwargs))


def finish_phases(phases: Generator) -> nl.NkiTensor:
    """Run the remaining phases of a selective_load_packed_phases generator and return the output."""
    while True:
        try:
            next(phases)
        except StopIteration as stop:
            return stop.value


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
    affinity_scaling: str = "pre_quant",
    down_psum_dtype: str = "bf16",
    gate_up_psum_dtype: str = "bf16",
    down_after_units: Optional[int] = None,
    silu0_after_relayout: Optional[bool] = None,
    pe_keep_alive: int = 0,
    pe_keep_alive_cols: int = 512,
    pe_keep_alive_tail: int = 0,
    dedupe_experts: Optional[str] = None,
    dyn_head_groups: int = 6,
    dyn_verbatim_groups: int = 1,
    dyn_tail_loads: str = "sp",
    dyn_pad_rows: str = "expert0",
    dyn_tail_gate: Optional[int] = None,
    dyn_fast_g1: bool = False,
    merged_down_gathers: bool = False,
    merged_down_pairs: Optional[Tuple[int, ...]] = None,
    down_real_rows: bool = False,
    dyn_direct_g1: bool = False,
    dyn_g1_dve_first: bool = False,
    dyn_late_relayout: bool = False,
    dyn_merged_stores: bool = False,
    hi_mask_tile: bool = False,
    hidden_queue: str = "act",
    early_gather: bool = False,
    merged_scales: bool = False,
    late_down: int = 0,
    last_relayout: Optional[str] = None,
    situ: Optional[Tuple[float, float]] = None,
    sigmoid_from_tanh: bool = False,
    hidden_input: str = "mx",
    affinity_layout: str = "dense",
    output: Optional[nl.NkiTensor] = None,
    output_row: int = 0,
    valid_tokens: Optional[int] = None,
    fp32_down_rounding: str = "terms",
    latent_quant: str = "transpose",
    weight_sbuf_budget: Optional[int] = None,
    sbuf_reserve: Optional[int] = None,
    name_suffix: Optional[str] = None,
    schedule: Optional[str] = None,
    group0_on_pool: bool = False,
    hwdge_index: Optional[nl.NkiTensor] = None,
    stream_chain: Optional[list] = None,
    buffer_cache: Optional[dict] = None,
    dyn_head_store: bool = False,
    hwdge_act_loads: Optional[int] = None,
    weights_arena: bool = False,
    tail_chain: Optional[list] = None,
    last_gate_up_first: bool = False,
    last_relayout_runs: int = 1,
    last_swiglu_first: bool = False,
    last_gate_up_split: bool = False,
    merged_down_quads: Tuple[int, ...] = (),
    first_gather_index: Optional[nl.NkiTensor] = None,
    dve_experts: bool = False,
    store_parts: int = 1,
    relayout0_act: bool = False,
    last_group_halves: bool = False,
    act_relayout_blocks: Optional[int] = None,
    last_finish_early: bool = False,
    down_group_mask: Optional[nl.NkiTensor] = None,
    paired_downs: bool = False,
    long_stream_short: bool = False,
) -> Generator[str, None, nl.NkiTensor]:
    """Packed expert MLP of the standalone kernel (inputs in HBM) and the fused MoE block (inputs in SBUF).

    The step numbers in the comments match the Algorithm section of moe_tkg_selective_load_packed.

    A generator of three phases (selective_load_packed_body runs all of them). The first phase (Steps 1-5) needs only
    the routing: it allocates the buffers, issues the group-0 HWDGE loads and the Pool weight stream and builds the
    affinity rows, then yields "routing". The second phase (the start of Step 7.1: the re-layout of groups 0 and 1)
    yields "relayout". The last phase reads hidden_qtz / hidden_scale (the first gate/up matmul), so with
    inputs_in_sbuf a caller can write them between the phases, as selective_load_tp_finish does. The generator
    returns the output.

    Args:
        sbm (SbufManager): SBUF manager. The body allocates its buffers in its own stack scope. With
            inputs_in_sbuf, the hand-off tiles of the caller must not overlap that scope (the fused block puts them
            on the heap), so the weight buffers start at the stack base as in the standalone kernel.
        hidden_qtz, hidden_scale (nl.NkiTensor): HBM [128, H/512, T] (standalone, loaded here) or SBUF
            [128, (H/512) * T] tiles (fused, written by the RMSNorm-MX subkernel). Both have the same element layout.
        expert_gate_up_packed, expert_down_weights, expert_down_weights_scale (nl.NkiTensor): HBM weights, as in
            moe_tkg_selective_load_packed.
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
        affinity_scaling (str): where the POST_SCALE affinity a of pair n is applied.
            "pre_quant" (default): v = a * SiLU(gate) * up is quantized to MXFP8 (optionally as hi + lo terms, see
            affinity_hi_lo), and the down projection adds the K experts of a token in its contraction, so the output
            needs no reduction. The MXFP8 rounding of a * h differs from the reference's a * MXFP8(h).
            "post_down": h = SiLU(gate) * up is quantized unscaled (the same MXFP8 values as the reference, up to the
            kernel's bf16 rounding of h). The down stationary has one column per pair, so PSUM holds y[n, :] =
            MXFP8(h_n) @ W_n for every pair; the eviction multiplies each row by its affinity (per-partition scale)
            and a bf16 matmul with a 0/1 stationary adds the K rows of each token. Requires T == 8, K = 8 or 16, and
            at most 128 / (T*K) output halves; affinity_hi_lo does not apply. Costs: T*K stationary columns per
            output half instead of T, and one more matmul and PSUM eviction per output chunk of the last group.
        down_psum_dtype (str): the dtype of out_ps, the down-projection output that PSUM accumulates over the
            groups.
            "bf16" (default): out_ps uses banks 2-5.
            "fp32": the matmul results stay in fp32 until the eviction. out_ps has twice the bytes: H > 2048 uses
            2 output halves. If H / 2 is not a multiple of 512 (H = 3584), the halves are uneven: half 0 has
            ceil(H / 1024) chunks of 512 columns and the last half the rest (2048 + 1536 columns at H = 3584), and
            the chunks past the last half's end run for half 0 only. out_cols <= 2048 keeps out_ps in banks 2-5;
            2048 < out_cols <= 3072 uses banks 2-7, and the gated values move into the gate/up tiles
            (16 * I + 4 * T <= 512). H <= 6144. With hi / lo, out_sb is fp32 too (the dynamic tails add in fp32),
            and the store DMA adds the hi and lo rows in fp32 and rounds once to bf16.
            Without dedupe each group holds whole tokens (16 % K == 0), so the other groups add exact zeros to the
            output rows of a token, and bf16 PSUM rounds each output value once (one term) or once per term (hi /
            lo, then the store adds the two bf16 rows). With dedupe_experts the experts of a token sit in several
            groups, and bf16 PSUM rounds each group's partial sum; "fp32" adds them in fp32. post_down loses at most
            1 bf16 ulp per value in bf16. The gate/up rounding has a much larger effect (gate_up_psum_dtype).
        gate_up_psum_dtype (str): the dtype of gate_up_ps, the gate and up projections that PSUM accumulates over the
            H512 blocks.
            "bf16" (default): each group parity has one bank of [T, 1024] bf16, gate at column 0 and up at 512.
            "fp32": the gate and up sums stay in fp32, so h = SiLU(gate) * up gets only the bf16 rounding of its
            DVE write and its MXFP8 values match the reference's MXFP8(h) more often. For I <= 16 each parity has one
            bank of [T, 512] fp32 (gate at 0, up at 256) and the bank plan is unchanged. For I > 16 each parity has
            two banks of [T, 1024] fp32 (gate in the first, up in the second; banks 0-1 and 2-3), and out_ps moves to
            banks 4-5 with 2 output halves of at most 2048 columns (H <= 4096, H / 2 a multiple of 512, bf16 down
            PSUM only; so I = 24 at H = 3584 asserts).
        down_after_units (Optional[int]): the gate/up units (sub-group, H512 block) of group g + 1 that the PE runs
            before down(g), for 0 < g < N/16 - 2 (Step 7.2). None (default): all units in the long stream (fused
            block: pre_quant and post_down; standalone kernel: pre_quant), so SwiGLU(g + 1) starts before down(g)
            and the MXFP8 chain of finish(g) has the whole gate/up(g + 1) to finish; else 13/16 of the units. The
            order of the PSUM accumulation does not change, so the output is the same for every value.
        silu0_after_relayout (Optional[bool]): schedule SiLU(0) after the ACT copies of the group-1 re-layout
            (Step 7.1). None (default): True in the fused long stream with pre_quant, else False. In the fused long
            stream the compiler can place SiLU(0), which waits for gate/up(0), ahead of these copies on ACT (behind
            the ACT expert loads), so gate/up(1) starts later and runs throttled after the PE idle time. Same
            output.
        pe_keep_alive (int): wide bf16 keep-alive matmuls right before gate/up(0) (Step 7.1). Default 0. After an
            idle time the PE runs throttled for a short time; the PE is idle from the routing to the group-0
            weights. The chain starts after the first ACT scale copy of group 0 (an ACT op writes 0 into the ones
            tile that the matmuls read), accumulates into PSUM bank 7 (free until the gated transposes), and SiLU(0)
            adds 0 times its output as its bias, so the compiler keeps it. Same output. Not with
            dedupe_experts="dynamic" (asserts: its compaction tiles use bank 7).
        pe_keep_alive_cols (int): moving columns of each keep-alive matmul, 128 to 512. Default 512.
        pe_keep_alive_tail (int): wide keep-alive matmuls right before the last gate/up, after down(N/16 - 3) in PE
            order (Step 7.2). Default 0. The last gate/up waits for its weights and then runs throttled. The
            matmuls accumulate into PSUM bank 6 (the routing tiles are done), and the SiLU of the last group adds 0
            times their output as its bias. Same output.
        dedupe_experts (Optional[str]): skip the weight loads (static) or the work (dynamic) of repeated experts.
            Tokens often share experts (real decode routing: about 92 distinct experts in the 128 pairs of 8 tokens
            at K = 16). Needs pre_quant. None (default): every pair loads and computes its expert.
            "static": the same static schedule (the PE work does not change), but a pair whose expert an earlier pair
            (lower n) has does not load it. Its gate/up gather rows and down gather rows are out of bounds
            (no data, the Pool descriptors stay), and its HWDGE load in the HWDGE group has an out-of-bounds expert id
            (no packets). The first pair of each expert serves every token that routes to it: gate/up computes all T
            tokens, and its rows of the down stationary carry the affinity of each of these tokens
            (row_weights). The rows of a repeated pair get zero affinities. The skipped rows keep stale bytes:
            FP4 weights are always finite, and the scale segments of the row slots of the groups that can skip are
            zeroed before their loads (an MX scale byte 0xFF is NaN, and 0 * NaN = NaN). Gathered position 0 (the
            first gather) and the down weights of group 0 load every pair and do not wait for the dedupe work, and
            the down scales are never skipped. All dedupe work runs on DVE and PE (no extra Pool instruction).
            The output is bit-identical to None without repeated experts. With repeats, a token's output adds the
            parts of several groups in the bf16 PSUM, so values can differ by about 1 bf16 ulp. The Pool descriptors
            of the skipped rows remain, so the gather stream takes the same time: the saving is DMA bytes (the
            weights of the later groups arrive sooner), not Pool time.
            Layouts (Step 3.6): earlier_mask / pair_identity bf16 [128, N]: [a, n] = 1 if n < a / if n == a.
            earlier_count fp32 [128, 1]: the earlier pairs with the expert of pair a (a < N). index_dedupe int32
            [128, N]: index[a, n] + _SKIP_EXPERT * count of pair n. column_groups / column_counts bf16 [128, C]
            (C = G - 1 + N/16 - 1 dedupe columns: gathered positions 1 .., then the down weights of groups 1 ..):
            [n, i] = 1 / the count of pair n if pair n belongs to the group of column i. pair_select bf16
            [128, 256]: [n, a] = 1 if n % 16 == a % 16 (gate/up rows), then [n, 128 + a] = 1 if n % 16 == a / 8
            (down rows). gather_index_dedupe int32 [128, (G - 1) * M + N/16 - 1]: the gather_index columns of the
            dedupe gathers + _SKIP * the count of the pair of each row. slot_weights bf16 [128, T]: [j, t] = the
            affinity of token t for the expert of pair j if pair j is the first with it. dedupe_row_select bf16
            [128, (N/16) * 128]: [j, (g, m)] = 1 if j == 16g + m/8.
            "dynamic": skip the work of repeated experts. Needs T*K = 128 (K = 16). It is fast only with a compiler
            that does not put a scheduling barrier at each loop-iteration boundary; a compiler that does builds and
            runs it correctly, but slower. A token may have the same expert twice: a repeat after the verbatim
            tokens folds into the slot of the first pair (its affinity adds to the row weights), and a repeat inside
            the verbatim tokens keeps its own slot (the expert loads twice, as with None).
            - Compaction (Step 3.0): the pairs whose expert an earlier pair has are folded into the slot of the first
              occurrence. Unique slots stay in pair order (token 0 is always slots 0-15), U = the number of unique
              experts, and the slots u >= U are padding. Equality row x strictly-lower mask (first occurrence), a
              triangular-ones prefix-sum matmul (slots and U), a one-hot scatter matmul (cidx[a, u] = expert of
              slot u on every partition) and a slot-map matmul (pair -> slot). Exact in fp16 / bf16 with fp32 PSUM.
            - The gathers and the expert loads read the compacted ids. row_weights[m, (g, t)] = the affinity of
              token t for the expert of slot 16g + m/8 (a one-hot matmul), so one slot serves every token that chose
              its expert.
            - The static schedule runs groups 0 .. dyn_head_groups - 1. Then tail block G (an nl.fori_loop with trip
              count (ceil(U / 16) == G)) runs groups dyn_head_groups .. G - 1 with their own pipeline, for each G
              above the head. A PSUM tile cannot be live across a loop boundary, so the head evicts its output into
              out_sb, a tail adds its own output into out_sb (DVE), and Step 7.6 stores out_sb after the tails.
            At U = 128 the output is bit-identical to dedupe_experts=None. At U < 128 a token's output adds terms of
            several groups in bf16 PSUM, so about a quarter of the values differ by up to about 0.5% of max|out|.
        dyn_head_groups (int): groups of the static head (dynamic dedupe). Default 6.
        dyn_verbatim_groups (int): 1 (default): only token 0 keeps its pairs as slots. 2: tokens 0-1 do, so the
            first gate/up gather does not wait for the compaction, but the token-1 repeats of token 0 add slots.
        dyn_tail_loads (str): where the gate/up rows of the tail groups come from. "sp" (default): SP per-expert
            HWDGE loads after the HWDGE group (a padding slot skips the load, no packets); "sp6": only the first
            tail group on SP, the others Pool gathers; "sp_prio": "sp" with DMA priority 3 on the tail loads and 1 on
            the head's gathers; "pool": Pool gathers at the end of the stream.
        dyn_pad_rows (str): padding slots of the head's Pool gathers. "expert0" (default) loads expert 0 (real
            rows, affinity 0): the DMA engines finish out-of-bounds gather rows later than real rows.
            "oob" skips them (stale rows, the scale segments of the slots are zeroed first).
        dyn_tail_gate (Optional[int]): None (default), or the head group after whose re-layout the SP tail loads
            start (their expert ids are copied by DVE there). Slower in every measured case.
        dyn_fast_g1 (bool): False (default). True: the gather indices of group 1 come from a 48-pair compaction
            (Stage A, tokens 0-2), ahead of the full one; holes of group 1 shift the later slots. Slower in the fused
            block as measured.
        merged_down_gathers (bool): False (default): one Pool gather per group loads its down weights (128 rows).
            True: one gather of 256 rows (a [128, 2] index tile) loads the down weights of groups 2q and 2q + 1, so the
            Pool stream has one fixed gather cost less per two groups (less Pool time). Index [a, b] feeds row
            r = 128b + a, which lands on partition r / 2 = 64b + a / 2, slot a % 2 (group 2q + a % 2): the pair is
            32q + 8b + 16(a % 2) + a / 16 and the source row is (a / 2) % 8. The stream issues the gather where the
            gather of group 2q was. Same data, same output. Needs one sub-group and an even number of groups, and not
            dedupe_experts="static".
        merged_down_pairs (Optional[Tuple[int, ...]]): with merged_down_gathers, the first group g of each merged
            pair (g, g + 1). None (default): all pairs (0, 1), (2, 3), ... A tuple, for example (1, 6) for the
            standalone kernel: only these pairs, so no down gather moves ahead of a gate/up gather (the other groups
            keep their own gathers; the merged columns use their own source-row column, lanes[:, 18]).
        merged_down_quads (Tuple[int, ...]): with merged_down_gathers (all pairs), the first groups (multiples of 4)
            of 4-group down gathers: one gather of 512 rows (a [128, 4] index) loads the down weights of groups q ..
            q + 3, where the gather of the pair (q, q + 1) was. Index [a, b] feeds row r = 128b + a, which lands on
            partition 32b + a / 4, slot a % 4 (group q + a % 4): the pair is 16q + 16(a % 4) + 4b + a / 32 and the
            source row is (a / 4) % 8 (lanes[:, 18]). Default (): pairs only. Same data, same output.
        down_real_rows (bool): False (default): the down-weight gathers skip rows p >= I/4 of each expert and the
            down-scale gathers skip the unused partitions (a / 16 odd) with out-of-bounds indices. True: these rows
            load real rows (the expert's padding rows p >= I/4, and the scale row of a real pair), which the matmuls
            do not use (zero stationary rows, unread scale partitions). An out-of-bounds gather row is slow in the
            DMA engines, so the gathers complete sooner. Same output.
        dyn_direct_g1 (bool): False (default). True (dynamic dedupe, fused block): the gather indices of group 1 come
            straight from the prefix count of the compaction: select[n, a] = kept[n] * (incl[n] == 17 + a % 16), and
            a matmul with the fp16 expert ids gives the expert of slot 16 + a % 16 on partition a (0 for a hole:
            expert 0, as cidx0). The group-1 gather goes out sooner, before the rest of the compaction. Pool
            then gathers the other expert ids behind it, and DVE forms their gather indices after the group-0
            re-layout (a DVE copy of the ids is ordered there, so no DVE op waits for the Pool queue), with the router
            affinities and the row weights after them. Same output.
        dyn_g1_dve_first (bool): False (default). True (dynamic dedupe): DVE copies the first 4 H512 blocks of the
            group-1 re-layout and ACT the scale segments first, then the other blocks, so gate/up(1) starts
            sooner after its rows land. Same output.
        dyn_late_relayout (bool): False (default). True (dynamic dedupe): ACT re-lays the first tail group (all
            blocks) after the SiLU of the head's last group, so that re-layout does not delay the head's last
            SwiGLU (it is needed only when a tail block runs). Same output.
        dyn_head_store (bool): False (default). True (dynamic dedupe): the head stores the output chunk by chunk in
            its last evictions (as with dedupe_experts=None), and a tail that runs stores the whole output again after
            its own evictions (same queue, so the tail's store lands last). Step 7.6 is skipped. Without a tail (U <=
            16 * dyn_head_groups), the output DMAs overlap the head's last down projection, and with a compiler that
            fences each block, they also overlap the fence's DMA drain instead of following the fence. Same output.
        dyn_merged_stores (bool): False (default). True (dynamic dedupe, one output half, hi / lo): Step 7.6 stores
            the output in 4 contiguous DMAs instead of one per chunk: the last even chunk, chunks 0-2, 3-4, then the
            last chunk (in eviction order). Same output.
        hi_mask_tile (bool): False (default; the fused block passes True). True (hi / lo): Step 6.4 forms hi with a
            tensor_tensor AND against a mask tile instead of a tensor_scalar AND. The tensor_scalar runs in the DVE
            fast mode, which stalls while a Pool gather generates descriptors (the op then takes longer), and the
            stall delays down(g). Same output (the same AND).
        hidden_queue (str): the HWDGE queue of the standalone hidden loads (Step 2.2). "act" (default): ACT, where
            the compiler queues load_hidden behind the ACT expert loads and the ACT table load, so gate/up(0) waits
            for it. "sp": SP, ahead of the SP expert loads. Same output.
        early_gather (bool): False (default). True (standalone, no dedupe): the gather indices of the first gathered
            gate/up sub-group come from the PSUM index broadcast (a masked row sum over its 16 pairs) into their own
            tile, so the first gather follows the Pool expert gather at once and waits for none of the other
            gather-index writes. Same output.
        merged_scales (bool): False (default). True: one Pool gather of 256 rows ([128, 2] index) loads both
            down-scale tiles (needs N/16 = 8 groups and one sub-group, not dedupe_experts="static"). Index [a, b]
            feeds row r = 128b + a, which lands on partition m = 64b + a / 2 of tile a % 2, so it selects the
            down-scale pair of partition m in tile a % 2 (Step 3.2). The Pool stream has one fixed gather cost less
            (less Pool time), and the scales of groups 4-7 arrive with those of groups 0-3. Same output.
        late_down (int): 0 (default). c > 0 (standalone long stream, pre_quant, no dedupe): down(g - 1) runs inside
            gate/up(g + 1) after its units for 1 <= g < c, and at g = c both down(c - 1) and down(c) run there (Step
            7.2); the down gathers of group d move after the gate/up gather of group d + 2 in the Pool stream (Step
            4). So the gate/up rows of group 2 follow those of group 1 at once, and the PE starts the steady state
            sooner. The order of the PSUM accumulation changes, but each group adds exact zeros to the output rows of
            the other tokens (16 % K == 0), so the output is the same.
        last_relayout (Optional[str]): None (default) or "dve" (long stream): DVE copies all H512 blocks of the last
            group's re-layout (ACT its scale segments). It runs after the Pool stream, when DVE copies do not stall.
            Same output.
        early_gather, merged_scales, late_down = None and last_relayout = "auto" (the standalone entry defaults):
            the standalone set, early_gather=True, merged_scales=True, late_down=5 and last_relayout="dve" in the
            standalone long stream with pre_quant and no dedupe_experts, else off. The standalone entry also uses
            hidden_queue="sp".
        situ (Optional[Tuple[float, float]]): None (default): SiLU(gate) * up. (beta, linear_beta): the SiTU
            activation beta*tanh(g/beta)*sigmoid(g) * linear_beta*tanh(u/linear_beta) in fp32 (Step 6.3). ACT computes
            tanh(g/beta), tanh(u/linear_beta) and sigmoid(g) into fp32 tiles, DVE forms
            tanh(g/beta) * (beta*linear_beta) * sigmoid(g) and multiplies it by tanh(u/linear_beta) into gated (bf16).
        sigmoid_from_tanh (bool): with situ, compute sigmoid(g) as (1 + tanh(g/2)) / 2 (ACT: tanh(g/2); DVE:
            (1 + tanh(g/2)) * tanh(g/beta), then times linear_beta*beta/2 * tanh(u/linear_beta)), so ACT uses only the
            tanh table. Default False.
        hidden_input (str): "mx" (default): hidden_qtz / hidden_scale are the MXFP8 hidden. "bf16" (standalone):
            hidden_qtz is the bfloat16 latent [T, H] @HBM and hidden_scale is unused. Step 2.5 loads it as 4 copies
            shifted by q columns, one identity matmul per H512 block puts latent column 512*hb + 4*p + q on partition
            p (lane q), DVE swizzles the PSUM tile to [p, (hb, t, q)] and quantize_mx writes hidden_raw /
            hidden_raw_s (quantize_latent_mx). No norm.
            Each MX group then holds 32 consecutive latent values of one token, and the gate/up weights must use the
            same hidden order.
        affinity_layout (str): "dense" (default): HBM float32 [T, E] affinities (Step 5.1 picks the K routed values
            of each token with a one-hot). "topk" (standalone): HBM float32 [T, K] affinities in the order of
            expert_index. Step 2.4 loads them as one row on partition 0, rounds them to bf16 and a ones matmul puts
            the affinity of pair n on every partition (the affinity buffer of Step 5.2, as with "dense").
        output (Optional[nl.NkiTensor]): bfloat16 HBM destination with H columns and at least output_row + T rows.
            None (default): a new [T, H] tensor.
        output_row (int): the row of output for token 0. Default 0.
        valid_tokens (Optional[int]): tokens 0 .. valid_tokens - 1 are stored (the others are padding). None
            (default): all T tokens.
        fp32_down_rounding (str): the output rounding with down_psum_dtype="fp32" and hi / lo. PSUM adds the groups of
            a token in fp32 in both modes.
            "terms" (default): the hi and the lo sum round to bf16 one at a time (the eviction writes bf16 out_sb), and
            the store DMA adds the two bf16 rows and rounds again. Without repeated experts a token sits in one group,
            so this is the rounding of the bf16 PSUM; with dedupe_experts the output has the same rounding steps as
            without (the fp32 order of the group sums can flip a bf16 rounding in rare values). dynamic dedupe: out_sb
            stays fp32 for the tails, the head also evicts the bf16 terms into out_terms, and a tail writes
            bf16(out_ps + out_sb) there; Step 7.6 stores out_terms.
            "once": out_sb is fp32, and the store DMA adds the fp32 hi and lo sums and rounds once.
        latent_quant (str): hidden_input="bf16": "transpose" (default) or "matmul" (see quantize_latent_mx).
        weight_sbuf_budget (Optional[int]): SBUF bytes per partition for the gate/up row ring, the moving rings and
            the down weights (Step 1.2). A smaller budget gives fewer ring slots (first 2 moving slots instead of 3,
            then fewer row slots), so the call leaves more SBUF free. The output does not change. None (default):
            _WEIGHT_SBUF_BUDGET.
        sbuf_reserve (Optional[int]): None (default): the weight rings come first in the stack scope and use
            weight_sbuf_budget. An int (inputs_in_sbuf only): the other buffers come first, then the weight rings
            (512 B aligned) with the ring depths that leave at least sbuf_reserve bytes per partition free between the
            stack and the heap of sbm (the budget is the smaller of weight_sbuf_budget and that room). The output does
            not change.
        name_suffix (Optional[str]): appended to every op and tensor name of the call, so a kernel can run the body
            more than once (names must be unique in a kernel). None (default): the names as they are.
        schedule (Optional[str]): the schedule of the long stream. None (default): "fused" with inputs_in_sbuf, else
            "standalone". "standalone" with inputs_in_sbuf gives an SBUF-input call the standalone schedule: the
            HWDGE group is group 3, all of its loads on SP, finish(g) after half of the gate/up units of group g + 1,
            and the standalone set (early_gather, merged_scales, late_down, last_relayout) is allowed. The output
            does not change.
        group0_on_pool (bool): False (default): the gate/up rows of group 0 (and of the long-stream HWDGE group) come
            from per-expert HWDGE loads on SP and ACT (Step 2.3). True: every gate/up group comes from a Pool gather
            (no dynamic HWDGE expert load), as in a large kernel where the issue of a dynamic HWDGE DMA gets slow.
            Pool then runs two more gathers (U = 1, 8 groups: 8 gate/up gathers instead of 6). Not with
            dedupe_experts. The output does not change.
        hwdge_index (Optional[nl.NkiTensor]): int32 SBUF expert ids for the HWDGE expert loads (Step 2.3), with the
            same experts as expert_index: a row [1, N] on partition 0, or a tile [T, K] on the token partitions (pair
            n = t*K + k at [t, k]). The loads then do not wait for the routing broadcast. None (default): the loads
            read index (or expert_index_tk).
        stream_chain (Optional[list]): the Pool stream instructions of the earlier calls of a kernel that runs the
            body more than once. The Pool setup ops of this call (iotas, the Pool reduce) and its first gather are
            scheduled after the last instruction of the list, and the last gather of this call is appended. Pool runs
            in order, so a setup op of a later call that the compiler puts ahead of the gathers of an earlier call
            would hold them (it waits for the reads of the earlier call at its addresses). None (default): no chain.
        buffer_cache (Optional[dict]): the SBUF buffers of earlier calls. A buffer with the name, shape, dtype and
            address of an earlier buffer is that tensor, so the DMAs of the calls that write it have the same
            destination (their descriptor table entries can be shared). None (default): new buffers.
        hwdge_act_loads (Optional[int]): long stream: the last loads of the HWDGE group that go on the ACT queue,
            after its group-0 re-layout copies (Step 7.1); SP loads the others. None (default): 2 in the fused
            schedule, else 0. 8 at K = 16 splits the 32 HWDGE loads of a call 16 / 16 between SP and ACT. Same
            output.
        weights_arena (bool): False (default): the row ring, the down weights and the down scales are separate
            tensors. True: one tensor holds all three (the down weights and scales are views of its last columns), so
            every Pool gather and HWDGE load writes one tensor. The NEFF then gives each call one DGE table slot for
            these destinations instead of one per destination tensor of a multi-row gather. Same output.
        tail_chain (Optional[list]): the (DVE, ACT) tail instructions of the earlier calls of a kernel that runs the
            body more than once (pre_quant, no dedupe_experts). The first DVE and ACT re-layout copies of group 0 of
            this call are scheduled after the last entry, and this call appends its own: the MXFP8 quantization of its
            last group (DVE) and the last ACT op of its last SwiGLU. DVE and ACT run in order, so without the chain the
            compiler can put the group-0 copies of a call, which wait for its first gather, ahead of the last SwiGLU
            and MXFP8 chain of the call before it, and that call's tail then waits for the next call's gather. None
            (default): no chain. Same output.

        first_gather_index (Optional[nl.NkiTensor]): with early_gather, an int32 [128, 16] SBUF tile that the caller
            writes with the gather indices of the first gathered gate/up sub-group (the values of the body's own
            early-gather index: 128 * expert of pair 16 g1 + a % 16 + 8b + a / 16), so the body skips its early-gather
            ops. None (default): the body builds them. Same output.
        dve_experts (bool): False (default): one Pool gather (nc_n_gather) picks the expert of each gather-index
            column from the index (Step 3.5), before the first gate/up gather in the Pool stream. True: DVE picks them
            with masked row sums (each column reads a window of pairs that holds its pair: the 16 pairs of a gate/up
            group, the 32 or 64 pairs of a merged down gather, all 128 pairs for the down scales; one nonzero term,
            exact in fp32), so Pool starts with the first gate/up gather. Needs early_gather, one sub-group,
            consecutive gathered groups, merged_down_gathers with all pairs and merged_scales. Same output.

        relayout0_act (bool): False (default; long stream): DVE copies all H512 blocks of the group-0 re-layout. True:
            ACT copies the first 4 blocks of the gate and of the up weights of each part, as for the later groups:
            Pool descriptor windows slow down while DVE copies (one SBUF port), and the group-2 gather is in such a
            window. Same output.
        act_relayout_blocks (Optional[int]): long stream: the H512 blocks of the gate and of the up weights that ACT
            copies in the re-layout of groups 1 .. (DVE copies the others; the last group with last_relayout="dve"
            stays on DVE). None (default): 4. More blocks on ACT: fewer DVE SBUF accesses while Pool writes gather
            descriptors (DVE and Pool share one SBUF port; ACT copies at about half the DVE rate). Same output.
        store_parts (int): 1 (default) or more (hi / lo, one output half, no dynamic dedupe): each 512-column output
            chunk is stored by this many dma_compute DMAs of consecutive token rows, so the stores of the last
            chunks can use more DMA engines. Same output.

        The last-group options below change only the order of the work at the end of the long stream (8 groups,
        standalone schedule with late_down = N/16 - 3). The output does not change.
        last_gate_up_first (bool): False (default). True: the down gather of groups late_down - 1 and late_down goes
            after the last gate/up gather in the Pool stream, and their down projections run after the last gate/up
            on the PE (default: inside gate/up(late_down + 1)). The last group's gate/up rows then do not wait behind
            these down weights in the DMA queues, and the PE does the two down projections while ACT and DVE run the
            last activation. Each group adds exact zeros to the output rows of the other tokens, so the output is
            the same.
        last_finish_early (bool): with last_gate_up_first (not last_group_halves): the PE runs gate/up(last) without
            finish(second-to-last) inside it, then finish(second-to-last), down(late_down - 1), finish(last),
            down(late_down), down(second-to-last), down(last), so the MXFP8 chain of the last group runs under the
            down projections and down(last) does not wait for it. Same output.
        last_relayout_runs (int): 1 (default) or more, with last_relayout="dve": DVE re-lays the last group in this
            many runs of H512 blocks, gate then up for each run, so the last gate/up starts after the first run.
        last_swiglu_first (bool): False (default). True: the activation of the second-to-last group runs before the
            re-layout of the last group on ACT and DVE (default: after it, so it waits for the last gate/up rows),
            and finish(second-to-last) runs before the last gate/up on the PE (default: after 3 of its H512 blocks).
        last_gate_up_split (bool): False (default). True: the last group runs all its gate matmuls before all its up
            matmuls (finish(second-to-last) between them, or before both with last_swiglu_first), and the up sums go
            to PSUM bank 6 (free in the two-phase call). ACT then starts tanh(g / beta) and sigmoid(g) on the gate
            sums while the PE runs the up matmuls (the compiler orders all accesses of one PSUM bank). Needs situ. The
            sums do not change (each one adds the same blocks in the same order).
        last_group_halves (bool): False (default). True (broadcast routing, situ, all gate/up groups on Pool, not with
            last_gate_up_split): the last group runs in two halves of 8 pairs. Two Pool gathers of 8 experts ([128, 8]
            index: row r = 128b + a lands on partition 16b + a / 8, slot a % 8) load its rows, the first one before
            the down gather of groups late_down - 1 and late_down; each half has its own re-layout, gate/up matmuls
            (the second half's sums in PSUM bank 6) and activation, so the work after the last gate/up rows is half
            of a group's. Each output column adds the same blocks in the same order, so the output is the same.
        down_group_mask (Optional[nl.NkiTensor]): None (default): one down scale set (scales [E, H]). Else a bfloat16
            [128, 2] SBUF tile, with down scales [E, 2, H] (two scale groups: the scales of the two checkpoint
            blocks of the rank slice): column s is 1 on partition a if down row a % 8 uses scale group s, else 0.
            The down projection runs once for each scale group s: its stationary holds only the rows of group s (the
            row weights of the other rows are 0) with their own MXFP8 scales, the matmul reads the down scales of
            group s, and both matmuls add into the same PSUM rows (group s = 1 after s = 0). Layouts: row_weights
            [m, (s, g, t)], qmx_in / inter / inter_s per group [s][term][j][t], and each down-scale tile [s][H].
            pre_quant only, no dedupe_experts. Costs: twice the down matmuls, the down stationary and the down-scale
            bytes.
        long_stream_short (bool): False (default). True (4 groups, one sub-group): run the long-stream schedule
            (Steps 2.3, 6.1, 7.2) with 4 groups, as with 8: the standalone set gets late_down = 1 (down(0) and down(1)
            inside gate/up(2), down(2) and down(3) at the end), so the last-group options apply (late_down = N/16 - 3),
            and last_relayout="dve". merged_scales stays off (one down-scale tile). Default: 4 groups run the short
            schedule. Same output.
        paired_downs (bool): False (default). True (late_down odd, not last_gate_up_first): the PE runs down(g - 1)
            and down(g) together inside gate/up(g + 1) for odd g, the two groups of one merged down gather (default:
            down(g - 1) inside gate/up(g + 1) for every g < late_down). So the down projections of groups 1 and 3 run
            one gate/up earlier, where the PE waits for gate/up rows (with two down scale groups, whose down
            projections take twice the PE time). Each group adds exact zeros to the output rows of the other tokens,
            so the output does not change.

    Returns:
        output (nl.NkiTensor): bfloat16 [T, H] @HBM (output when it is given).
    """
    # name_suffix: every op and tensor name of this call ends with it (the module views of _named_modules).
    nisa, nl = _named_modules(name_suffix)
    # Step 1: Plan the buffers.
    # Step 1.1: Get the sizes from the tensor shapes and check the constraints.
    E, _, row_bytes = expert_gate_up_packed.shape
    H = expert_down_weights.shape[2]
    n_h512 = H // 512
    routing_broadcast = inputs_in_sbuf and expert_index.shape[0] == _P_MAX
    kernel_assert(
        schedule in (None, "fused", "standalone"), f"schedule must be None, 'fused' or 'standalone', got {schedule}"
    )
    # The long-stream schedule: the fused block's (inputs in SBUF) or the standalone kernel's (Step 1.1).
    fused_schedule = inputs_in_sbuf if schedule == None else schedule == "fused"
    if routing_broadcast:
        T = hidden_qtz.shape[1] // n_h512
        K = expert_index.shape[1] // T
    else:
        T, K = expert_index.shape
    n_chunks = div_ceil(n_h512, _BLOCKS)
    n_scale_segments = div_ceil(2 * n_chunks, _X4)  # S: 4 column chunks per 4*I-byte scale segment
    I = row_bytes // (_X4 * (n_h512 + n_scale_segments))
    n_pairs = T * K
    n_groups = n_pairs // _GROUP
    n_scale_tiles = div_ceil(n_groups, _SCALE_GROUPS)  # down-scale tiles of 4 groups each
    down_rows = I // _X4  # valid rows of an expert's 8-row down MX block
    group_cols = _GROUP * I  # gate (or up) columns of a group in the moving layout: 16 experts x I
    n_sub_groups = 2 if n_h512 >= _SUB_GROUP_MIN_H512 and row_bytes >= _SUB_GROUP_MIN_ROW_BYTES else 1  # U
    kernel_assert(
        dedupe_experts in (None, "static", "dynamic"),
        f"dedupe_experts must be None, 'static' or 'dynamic', got {dedupe_experts}",
    )
    dynamic = dedupe_experts == "dynamic"
    if dynamic:
        """Compacted unique experts (Step 3.0): the static schedule runs the first dyn_head_groups groups of
        unique-expert slots; the later groups run in one of the tail blocks (trip count 0 or 1, Step 7.5)."""
        kernel_assert(
            n_groups == 8 and n_sub_groups == 1 and affinity_scaling == "pre_quant",
            "dedupe_experts='dynamic' needs T*K = 128, one sub-group and affinity_scaling='pre_quant'",
        )
        kernel_assert(K == _GROUP and E <= _DYN_PAD, "dedupe_experts='dynamic' needs K = 16 and E <= 2048")
        kernel_assert(
            2 <= dyn_head_groups < n_groups, f"dyn_head_groups must be in 2 .. {n_groups - 1}, got {dyn_head_groups}"
        )
        kernel_assert(dyn_verbatim_groups in (1, 2), f"dyn_verbatim_groups must be 1 or 2, got {dyn_verbatim_groups}")
        kernel_assert(
            dyn_pad_rows in ("expert0", "oob"), f"dyn_pad_rows must be 'expert0' or 'oob', got {dyn_pad_rows}"
        )
        kernel_assert(
            dyn_tail_loads in ("pool", "sp", "sp6", "sp_prio"),
            f"dyn_tail_loads must be 'pool', 'sp', 'sp6' or 'sp_prio', got {dyn_tail_loads}",
        )
    sub_pairs = _GROUP // n_sub_groups  # M: pairs per sub-group, index columns of its gather
    sub_cols = sub_pairs * I  # moving columns of one gate (or up) matmul
    # Long stream: 8 groups of 16 pairs (K = 16), so the Pool gathers limit the block (tuning in Steps 2.3, 6.1, 7.2).
    long_stream = (
        (n_groups > _SCALE_GROUPS or (long_stream_short and n_groups == _SCALE_GROUPS))
        and n_sub_groups == 1
        and n_h512 > _LONG_ACT_RELAYOUT_BLOCKS
    )
    # The group loaded by HWDGE in addition to group 0. The standalone pre_quant PE starts sooner (no
    # router), so its group 2 would be needed before SP delivers it; with post_down group 2 is faster (measured).
    hwdge_group = None
    if long_stream and not group0_on_pool:
        standalone_pre_quant = not fused_schedule and affinity_scaling != "post_down"
        hwdge_group = _LONG_HWDGE_GROUP_STANDALONE if standalone_pre_quant else _LONG_HWDGE_GROUP
    # Sub-groups that come from Pool gathers (G of them): all but sub-group 0 and hwdge_group, which come from HWDGE
    # (group0_on_pool: all of them).
    gathered = [
        j
        for j in range(0 if group0_on_pool else 1, n_groups * n_sub_groups)
        if j != hwdge_group
        and not (
            dynamic
            and dyn_tail_loads != "pool"
            and j >= dyn_head_groups
            and not (dyn_tail_loads == "sp6" and j > dyn_head_groups)
        )
    ]  # dynamic dedupe with dyn_tail_loads="sp": the tail groups come from SP HWDGE loads (Step 3.0)
    n_gathered = len(gathered)  # G
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
    if down_fp32 and gate_up_banks == 1 and H % (n_out_halves * _DOWN_CHUNK) != 0:
        # fp32 down PSUM with H / Hh not a multiple of 512 (H = 3584): uneven halves of whole 512-column chunks. Half 0
        # takes ceil(H / (512 Hh)) chunks, the last half the rest (2048 + 1536 columns at H = 3584).
        out_cols = div_ceil(H, n_out_halves * _DOWN_CHUNK) * _DOWN_CHUNK
    if down_fp32 or gate_up_banks == 2:
        kernel_assert(
            out_cols % _DOWN_CHUNK == 0,
            f"fp32 PSUM with {n_out_halves} output halves needs H / {n_out_halves} to be a multiple of 512, got H={H}",
        )
    # fp32 out_ps of more than 2048 columns: banks 2-7, so the gated values of group g move into gate_up_ps(g % 2)
    wide_out_ps = down_fp32 and out_cols > _OUT_PSUM_COLS_FP32
    if gate_up_fp32:
        kernel_assert(not wide_out_ps, "gate_up_psum_dtype='fp32' with down_psum_dtype='fp32' needs H <= 4096")
    if wide_out_ps:
        kernel_assert(
            group_cols + _X4 * T <= _PSUM_COLS,
            f"down_psum_dtype='fp32' with out_cols > 2048 needs 16 * I + 4 * T <= {_PSUM_COLS}, got I={I}",
        )
    n_down_chunks = out_cols // _DOWN_CHUNK  # per output half (of the widest half, half 0)
    # 512-column chunks of each output half: all n_down_chunks, except the last half of uneven halves
    half_chunks = [min(n_down_chunks, (H - half * out_cols) // _DOWN_CHUNK) for half in range(n_out_halves)]
    kernel_assert(
        half_chunks[-1] > 0 and sum(half_chunks) * _DOWN_CHUNK == H,
        f"Output halves of {out_cols} columns do not cover H={H} in 512-column chunks",
    )
    kernel_assert(
        affinity_scaling in ("pre_quant", "post_down"),
        f"affinity_scaling must be 'pre_quant' or 'post_down', got {affinity_scaling}",
    )
    post_down = affinity_scaling == "post_down"
    static_dedupe = dedupe_experts == "static"
    if static_dedupe:
        kernel_assert(not post_down, "dedupe_experts='static' needs affinity_scaling='pre_quant'")
    if merged_down_gathers:
        kernel_assert(
            n_sub_groups == 1 and n_groups % 2 == 0 and not static_dedupe,
            "merged_down_gathers needs one sub-group, an even number of groups and no static dedupe",
        )
    # merged_down_pairs: the first groups of the merged down-weight pairs (None: all pairs 2q, 2q + 1)
    pair_starts = tuple(range(0, n_groups, 2)) if merged_down_pairs is None else tuple(merged_down_pairs)
    mixed_down = merged_down_gathers and merged_down_pairs is not None  # merged and single down gathers
    if mixed_down:
        kernel_assert(
            all(0 <= g < n_groups - 1 for g in pair_starts)
            and all(b - a >= 2 for a, b in zip(sorted(pair_starts), sorted(pair_starts)[1:])),
            f"merged_down_pairs must hold disjoint pairs inside 0 .. {n_groups - 1}, got {merged_down_pairs}",
        )
    quad_starts = tuple(merged_down_quads)
    kernel_assert(
        not quad_starts or (merged_down_gathers and not mixed_down and n_groups % 4 == 0 and dedupe_experts == None),
        "merged_down_quads needs merged_down_gathers with all pairs, a multiple of 4 groups and no dedupe_experts",
    )
    kernel_assert(
        all(g % 4 == 0 and 0 <= g < n_groups for g in quad_starts),
        f"merged_down_quads must hold multiples of 4 below {n_groups}, got {merged_down_quads}",
    )
    merged_with = {}  # group -> first group of its merged pair (or quad)
    if merged_down_gathers:
        merged_with = {g + b: g for g in pair_starts for b in range(2)}
        merged_with.update({g + b: g for g in quad_starts for b in range(4)})
    # The standalone schedule set (None / "auto", the standalone entry defaults): on in the standalone long stream with
    # pre_quant and no dedupe_experts, else off.
    standalone_set = not fused_schedule and long_stream and dedupe_experts == None and affinity_scaling == "pre_quant"
    early_gather = standalone_set if early_gather == None else early_gather
    merged_scales = (standalone_set and n_scale_tiles == 2) if merged_scales == None else merged_scales
    # standalone set: late_down = N/16 - 3 (5 for 8 groups, 1 for 4 groups with long_stream_short)
    late_down = (min(_LATE_DOWN, n_groups - 3) if standalone_set else 0) if late_down == None else late_down
    last_relayout = ("dve" if standalone_set else None) if last_relayout == "auto" else last_relayout
    kernel_assert(hidden_queue in ("act", "sp"), f"hidden_queue must be 'act' or 'sp', got {hidden_queue}")
    kernel_assert(
        not early_gather
        or (
            (not inputs_in_sbuf or routing_broadcast)
            and not fused_schedule
            and dedupe_experts == None
            and n_sub_groups == 1
        ),
        "early_gather needs the standalone schedule (HBM or broadcast routing), one sub-group and no dedupe_experts",
    )
    kernel_assert(
        not merged_scales or (n_scale_tiles == 2 and n_sub_groups == 1 and dedupe_experts != "static"),
        "merged_scales needs 2 down-scale tiles, one sub-group and no static dedupe",
    )
    kernel_assert(
        late_down == 0
        or (long_stream and not fused_schedule and dedupe_experts == None and affinity_scaling == "pre_quant"),
        "late_down needs the standalone long stream with pre_quant and no dedupe_experts",
    )
    last_options = (
        last_gate_up_first or last_relayout_runs != 1 or last_swiglu_first or last_gate_up_split or last_group_halves
    )
    kernel_assert(
        not last_group_halves
        or (routing_broadcast and situ != None and group0_on_pool and n_sub_groups == 1 and not last_gate_up_split),
        "last_group_halves needs the broadcast routing tiles, situ, group0_on_pool, one sub-group and no "
        "last_gate_up_split",
    )
    kernel_assert(
        not last_options or (long_stream and late_down == n_groups - 3 and dedupe_experts == None),
        f"The last-group options need the standalone long stream with late_down={n_groups - 3}, got "
        f"late_down={late_down}",
    )
    kernel_assert(
        last_relayout_runs == 1 or (last_relayout == "dve" and 1 <= last_relayout_runs <= n_h512),
        f"last_relayout_runs={last_relayout_runs} needs last_relayout='dve' and at most {n_h512} runs",
    )
    kernel_assert(not last_gate_up_split or situ != None, "last_gate_up_split needs the situ activation")
    kernel_assert(
        not last_finish_early or (last_gate_up_first and not last_group_halves and not last_swiglu_first),
        "last_finish_early needs last_gate_up_first, and not last_group_halves or last_swiglu_first",
    )
    kernel_assert(
        store_parts == 1 or (1 <= store_parts <= T and n_out_halves == 1 and dedupe_experts == None),
        f"store_parts={store_parts} needs 1 .. {T} parts, one output half and no dedupe_experts",
    )
    act_blocks = _LONG_ACT_RELAYOUT_BLOCKS if act_relayout_blocks == None else act_relayout_blocks
    kernel_assert(1 <= act_blocks <= n_h512, f"act_relayout_blocks must be in 1 .. {n_h512}, got {act_relayout_blocks}")
    kernel_assert(not group0_on_pool or dedupe_experts == None, "group0_on_pool needs dedupe_experts=None")
    kernel_assert(sbuf_reserve == None or inputs_in_sbuf, "sbuf_reserve needs inputs_in_sbuf")
    kernel_assert(last_relayout in (None, "dve"), f"last_relayout must be None or 'dve', got {last_relayout}")
    kernel_assert(
        dynamic or not (dyn_direct_g1 or dyn_g1_dve_first or dyn_late_relayout or dyn_merged_stores or dyn_head_store),
        "dyn_direct_g1, dyn_g1_dve_first, dyn_late_relayout, dyn_merged_stores and dyn_head_store need "
        "dedupe_experts='dynamic'",
    )
    token_cols = n_out_halves * T  # token columns of one term (hi or lo) of a group's down-projection stationary
    hi_lo = (inputs_in_sbuf if affinity_hi_lo == None else affinity_hi_lo) and not post_down
    stationary_cols = (2 if hi_lo else 1) * token_cols  # the hi term, then the lo term (one term without hi / lo)
    # post_down: one stationary column per pair slot (k, t) and output half, so output half h of pair n = t*K + k is
    # partition h*T*K + 8k + t (one row per pair; T = 8).
    stationary_block = T  # stationary columns per output half and term (pre_quant: one per token)
    if post_down:
        kernel_assert(T == _MX_ROWS and K in (8, 16), f"post_down needs T == 8 and K = 8 or 16, got T={T}, K={K}")
        kernel_assert(
            n_out_halves * n_pairs <= _P_MAX,
            f"post_down needs at most {_P_MAX // n_pairs} output halves, got {n_out_halves}",
        )
        stationary_block = n_pairs
        stationary_cols = n_out_halves * n_pairs
    n_expert_tiles = E // _P_MAX  # 128-expert tiles of the affinity one-hot
    kernel_assert(
        n_pairs in (64, 128) and _GROUP % K == 0, f"Expected T*K == 64 or 128 and K dividing 16, got T={T}, K={K}"
    )
    kernel_assert(
        H == 512 * n_h512 and _BLOCKS <= n_h512 <= 4 * _BLOCKS and out_cols % _DOWN_CHUNK == 0,
        f"Expected H % 512 == 0 and 2048 <= H <= 8192, got H={H}",
    )
    kernel_assert(I % _X4 == 0 and I <= _QUADRANT, f"Expected I % 4 == 0 and I <= 32, got I={I}")
    kernel_assert(E % _P_MAX == 0, f"Expected E % 128 == 0, got E={E}")
    kernel_assert(
        tuple(expert_down_weights.shape) == (E, _MX_ROWS, H)
        and tuple(expert_down_weights_scale.shape) in ((E, H), (E, 2, H)),
        f"Expected down weights [E, 8, H] and scales [E, H] or [E, 2, H], got {expert_down_weights.shape} and "
        f"{expert_down_weights_scale.shape}",
    )
    # Down scale groups of each expert row: 1, or 2 (down_group_mask: each row on the scale of its own checkpoint
    # block). The stationary of a group holds the scale groups one after the other (scale group s at s * Ns).
    n_scale_sets = 1 if len(expert_down_weights_scale.shape) == 2 else 2
    kernel_assert(
        (n_scale_sets == 2) == (down_group_mask != None),
        "Down scales [E, 2, H] need down_group_mask, and down_group_mask needs down scales [E, 2, H]",
    )
    kernel_assert(
        n_scale_sets == 1 or (not post_down and dedupe_experts == None and tuple(down_group_mask.shape) == (_P_MAX, 2)),
        f"Two down scale groups need pre_quant, no dedupe_experts and a [{_P_MAX}, 2] down_group_mask",
    )
    group_stride = n_scale_sets * stationary_cols  # down stationary columns of a group
    kernel_assert(
        not paired_downs or (late_down % 2 == 1 and not last_gate_up_first and not dynamic),
        f"paired_downs needs an odd late_down, no last_gate_up_first and no dynamic dedupe, got late_down={late_down}",
    )
    kernel_assert(hidden_input in ("mx", "bf16"), f"hidden_input must be 'mx' or 'bf16', got {hidden_input}")
    kernel_assert(
        affinity_layout in ("dense", "topk"), f"affinity_layout must be 'dense' or 'topk', got {affinity_layout}"
    )
    kernel_assert(
        fp32_down_rounding in ("terms", "once"),
        f"fp32_down_rounding must be 'terms' or 'once', got {fp32_down_rounding}",
    )
    kernel_assert(
        latent_quant in ("matmul", "transpose"), f"latent_quant must be 'matmul' or 'transpose', got {latent_quant}"
    )
    latent_input = hidden_input == "bf16"  # Step 2.2 quantizes the bf16 latent
    topk_affinities = affinity_layout == "topk"  # Step 2.4 broadcasts [T, K] affinities
    if latent_input:
        kernel_assert(
            not inputs_in_sbuf and tuple(hidden_qtz.shape) == (T, H) and hidden_qtz.dtype == nl.bfloat16,
            f"hidden_input='bf16' needs a bfloat16 HBM latent [{T}, {H}], got {hidden_qtz.dtype} {hidden_qtz.shape}",
        )
        kernel_assert(n_sub_groups == 1, "hidden_input='bf16' needs one sub-group (the early U = 2 head reads it)")
    if topk_affinities:
        kernel_assert(
            not inputs_in_sbuf and tuple(expert_affinities.shape) == (T, K),
            f"affinity_layout='topk' needs HBM float32 affinities [{T}, {K}], got {expert_affinities.shape}",
        )
    if situ != None:
        kernel_assert(
            len(situ) == 2 and situ[0] > 0 and situ[1] > 0, f"situ must be (beta, linear_beta) > 0, got {situ}"
        )

    # Step 1.2: Choose the ring depths: Rm moving slots (3 if they fit with all 4 row slots), then R row slots.
    moving_slot_bytes = _GROUP * gate_up_cols * 2 + n_scale_segments * _BLOCKS * group_cols
    # post_down buffers per partition: slot_val + slot_mask, qmx_in (bf16) and out_sb. They reduce the weight budget.
    post_down_bytes = n_groups * n_pairs * (4 + 2 + _X4 * 2) + out_cols * 2 if post_down else 0
    weight_budget = (_WEIGHT_SBUF_BUDGET if weight_sbuf_budget == None else weight_sbuf_budget) - post_down_bytes

    def ring_depths(budget: int) -> Tuple[int, int]:
        """(Rm, R) for a weight budget: 3 moving slots if they fit with all row slots, then the most row slots."""
        n_moving = 3 if n_groups * _GROUP * row_bytes + 3 * moving_slot_bytes + n_groups * 2 * H <= budget else 2
        n_rows = n_groups
        while n_rows > 2 and n_rows * _GROUP * row_bytes + n_moving * moving_slot_bytes + n_groups * 2 * H > budget:
            n_rows -= 1
        return n_moving, n_rows

    n_moving_slots, n_gate_up_slots = ring_depths(weight_budget)

    # Step 1.3: Allocate the SBUF buffers (layouts in the kernel docstring).
    # dedupe: the highest SBUF address of the caller's earlier stack scopes (the fused block's router temporaries).
    caller_stack_top = sbm.lower_bound + sbm.max_stack_usage
    sbm.open_scope()
    alloc = sbm.alloc_stack
    if name_suffix or buffer_cache != None:

        def alloc(shape, dtype, name=None, **kwargs) -> nl.NkiTensor:
            """sbm.alloc_stack with name_suffix on the tensor name; with buffer_cache, the tensor of an earlier call
            with the same name, shape, dtype and address."""
            tile = sbm.alloc_stack(shape, dtype, name=None if name == None else name + (name_suffix or ""), **kwargs)
            if buffer_cache == None:
                return tile
            return buffer_cache.setdefault((name, tuple(shape), str(dtype), sbm.get_stack_curr_addr()), tile)

    def alloc_weights(align: Optional[int] = None) -> Tuple[nl.NkiTensor, ...]:
        """The weight buffers: the gate/up row ring, the moving rings, the down weights and the down scales.

        weights_arena: one tensor holds the row ring, then the down weights, then the down scales of each partition
        (gate_up_w; down_w and down_s are views of its columns), so every Pool gather writes one tensor."""
        if weights_arena:
            rows_elems = n_gate_up_slots * _GROUP * row_bytes // 2  # uint16 elements of the row ring
            down_elems = n_groups * H  # 2-byte float4_e2m1fn_x4 elements of the down weights
            arena = alloc(
                (_P_MAX, rows_elems + down_elems + n_scale_tiles * n_scale_sets * H // 2),
                nl.uint16,
                name="gate_up_w",
                align=align,
            )
            moving = alloc((_P_MAX, n_moving_slots * _GROUP * gate_up_cols), nl.uint16, name="gate_up_moving")
            moving_s = alloc(
                (_P_MAX, n_moving_slots * n_scale_segments * _BLOCKS * group_cols // 2),
                nl.uint16,
                name="gate_up_moving_s",
            )
            weights = arena.view(expert_down_weights.dtype)[:, rows_elems : rows_elems + down_elems]
            scale_byte = 2 * (rows_elems + down_elems)  # first byte of the down scales in each partition
            scales = arena.view(nl.uint8)[:, scale_byte : scale_byte + n_scale_tiles * n_scale_sets * H]
            return arena, moving, moving_s, weights, scales
        rows = alloc((_P_MAX, n_gate_up_slots * _GROUP * row_bytes // 2), nl.uint16, name="gate_up_w", align=align)
        moving = alloc((_P_MAX, n_moving_slots * _GROUP * gate_up_cols), nl.uint16, name="gate_up_moving")
        moving_s = alloc(
            (_P_MAX, n_moving_slots * n_scale_segments * _BLOCKS * group_cols // 2), nl.uint16, name="gate_up_moving_s"
        )
        weights = alloc((_P_MAX, n_groups * H), expert_down_weights.dtype, name="down_w", align=align)
        scales = alloc((_P_MAX, n_scale_tiles * n_scale_sets * H), nl.uint8, name="down_s", align=align)
        return rows, moving, moving_s, weights, scales

    if sbuf_reserve == None:
        # The weights come first, so their DMA writes stay aligned.
        gate_up_w_base = sbm.get_stack_curr_addr()
        gate_up_w, gate_up_moving, gate_up_moving_s, down_w, down_s = alloc_weights()
    if inputs_in_sbuf:
        hidden_raw, hidden_raw_s = hidden_qtz, hidden_scale
    else:
        hidden_raw_dtype = nl.float8_e4m3fn_x4 if latent_input else hidden_qtz.dtype
        hidden_raw = alloc((_P_MAX, n_h512 * T), hidden_raw_dtype, name="hidden_raw")
        hidden_raw_s = alloc((_P_MAX, n_h512 * T), nl.uint8, name="hidden_raw_s")
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
    if merged_scales:
        scale_fields = alloc((_P_MAX, 5), nl.int32, name="scale_fields")  # merged_scales: the pair bit fields, sum
    if merged_down_gathers:
        # [a, 0]: 16(a % 2) + a / 16, the pair of index [a, b] minus 32q + 8b; [a, 1]: (a / 2) % 8, its source row
        # merged_down_quads: [a, 2]: 16(a % 4) + a / 32, the pair of quad index [a, b] minus 16q + 4b; [a, 3]:
        # (a / 4) % 8, its source row
        merged_down_fields = alloc((_P_MAX, 4 if quad_starts else 2), nl.int32, name="merged_down_fields")
    # Selector columns: the gathered gate/up sub-groups, the down weights, the down-scale tiles, and with dedupe the
    # pair a % N of partition a (Step 3.6).
    n_select = n_gathered + n_groups + n_scale_tiles + (1 if static_dedupe else 0)
    pair_col = n_select - 1
    select = alloc((_P_MAX, n_select), nl.int32, name="select")
    # dve_experts: fp32 (masked row sums; exact below 2^24), else int32 (the Pool gather)
    experts = alloc((_P_MAX, n_select), nl.float32 if dve_experts else nl.int32, name="experts")
    # column 18: the source row of the merged columns of merged_down_pairs, or of the quads of merged_down_quads
    lanes = alloc((_P_MAX, 19 if mixed_down or quad_starts else 18), nl.int32, name="lanes")
    gather_index = alloc((_P_MAX, n_gathered * sub_pairs + n_groups + n_scale_tiles), nl.int32, name="gather_index")
    # sub-group 1 from the router tile (dynamic dedupe: when group 1 is verbatim)
    tile_first = inputs_in_sbuf and expert_index_tk != None and (not dynamic or dyn_verbatim_groups > 1)
    # standalone dynamic dedupe with group 1 verbatim: its gather indices come from the index, not the compaction
    index_first = dynamic and dyn_verbatim_groups > 1 and not inputs_in_sbuf
    # dynamic dedupe, token 0 verbatim: group 1 from a 48-pair compaction (Stage A, dyn_fast_g1)
    fast_first = dynamic and dyn_fast_g1 and dyn_verbatim_groups == 1
    # dynamic dedupe, token 0 verbatim: group 1 from the prefix count (dyn_direct_g1, fused block)
    direct_g1 = dynamic and dyn_direct_g1
    if direct_g1:
        kernel_assert(
            inputs_in_sbuf and not fast_first and dyn_verbatim_groups == 1 and dyn_pad_rows == "expert0",
            "dyn_direct_g1 needs the fused block, dyn_fast_g1=False, dyn_verbatim_groups=1 and dyn_pad_rows='expert0'",
        )
    # gathered sub-group 1: no compaction wait (early_gather: its own index tile, Step 3.5)
    direct_first = tile_first or index_first or fast_first or direct_g1 or early_gather
    if dve_experts:
        kernel_assert(
            early_gather
            and n_sub_groups == 1
            and gathered == list(range(gathered[0], gathered[0] + n_gathered))
            and merged_down_gathers
            and not mixed_down
            and (merged_scales or n_scale_tiles == 1)
            and dedupe_experts == None,
            "dve_experts needs early_gather, one sub-group, consecutive gathered groups, merged_down_gathers with all "
            "pairs, merged_scales (or one down-scale tile) and no dedupe_experts",
        )
        # [a, (b, m)] masks of the merged down columns (32 pairs), the quads (64 pairs) and the scale columns (128
        # pairs; each first holds its iota), and the products of the masked row sums (one tile, DVE runs in order)
        dx_mask_dw = alloc((_P_MAX, 2 * 2 * _GROUP), nl.float32, name="dx_mask_dw")
        dx_mask_ds = alloc((_P_MAX, n_scale_tiles * n_pairs), nl.float32, name="dx_mask_ds")
        dx_mask_q = alloc((_P_MAX, 4 * 4 * _GROUP), nl.float32, name="dx_mask_q") if quad_starts else None
        dx_prod = alloc(
            (_P_MAX, max(n_gathered * _GROUP, n_groups * 2 * _GROUP, n_scale_tiles * n_pairs, 16 * _GROUP)),
            nl.float32,
            name="dx_prod",
        )
        dx_width = dx_prod.shape[1]  # the partition stride of its views
    kernel_assert(first_gather_index == None or early_gather, "first_gather_index needs early_gather")
    half_pairs = _GROUP // 2  # last_group_halves: pairs per half of the last group
    if last_group_halves:
        # lh_cols / lh_mask [a, k]: k, 1 if k == a % 8; lh_lanes [a, b]: 16b + a / 8 (the source row of index [a, b]);
        # lh_index [a, (h, b)]: the gather indices of half h
        lh_cols = alloc((_P_MAX, half_pairs), nl.int32, name="lh_cols")
        lh_mask = alloc((_P_MAX, half_pairs), nl.float32, name="lh_mask")
        lh_lanes = alloc((_P_MAX, half_pairs), nl.int32, name="lh_lanes")
        lh_prod = alloc((_P_MAX, half_pairs), nl.float32, name="lh_prod")
        lh_expert = alloc((_P_MAX, 2), nl.float32, name="lh_expert")
        lh_index = alloc((_P_MAX, 2 * half_pairs), nl.int32, name="lh_index")
    if early_gather:
        eg_cols = alloc((_P_MAX, _GROUP), nl.int32, name="eg_cols")
        eg_mask = alloc((_P_MAX, _GROUP), nl.float32, name="eg_mask")
        eg_prod = alloc((_P_MAX, _GROUP), nl.float32, name="eg_prod")
        eg_expert = alloc((_P_MAX, 1), nl.float32, name="eg_expert")
        eg_index = alloc((_P_MAX, _GROUP), nl.int32, name="eg_index")
    if tile_first:
        tile_a = alloc((T, _P_MAX), nl.int32, name="tile_a")
        tile_t = alloc((T, _P_MAX), nl.int32, name="tile_t")
        tile_rows = alloc((T, _P_MAX), nl.float16, name="tile_rows")
        tile_k = alloc((_P_MAX, K), nl.int32, name="tile_k")
        tile_k_mask = alloc((_P_MAX, K), nl.float32, name="tile_k_mask")
        tile_index_f16 = alloc((T, K), nl.float16, name="tile_index_f16")
        tile_picked = alloc((_P_MAX, K), nl.float32, name="tile_picked")
        tile_expert = alloc((_P_MAX, 1), nl.float32, name="tile_expert")
    if not inputs_in_sbuf and not topk_affinities:
        affinity_te = alloc((T, E), nl.float32, name="affinity_te")
        expert_id = alloc((_P_MAX, n_expert_tiles), nl.float32, name="expert_id")
        affinity_hot = alloc((_P_MAX, n_expert_tiles * n_pairs), nl.bfloat16, name="affinity_hot")
    if topk_affinities:
        # affinity_layout="topk" (Step 2.4): the [T, K] affinities as one row on partition 0, fp32 and bf16.
        affinity_row = alloc((1, n_pairs), nl.float32, name="affinity_row")
        affinity_row_bf16 = alloc((1, n_pairs), nl.bfloat16, name="affinity_row_bf16")
    ones = alloc((_P_MAX, _P_MAX), nl.bfloat16, name="ones")
    affinity = expert_affinities if routing_broadcast else alloc((_P_MAX, n_pairs), nl.bfloat16, name="affinity")
    affinity_mask = alloc((_P_MAX, n_pairs), nl.bfloat16, name="affinity_mask")  # exact in bf16: bf16 affinity x 0/1
    token_offsets = alloc((_P_MAX, n_groups * T), nl.float32, name="token_offsets")
    token_rows = alloc((_P_MAX, n_groups * T), nl.bfloat16, name="token_rows")  # exact in bf16: 0/1
    offsets = alloc((_P_MAX, _GROUP), nl.float32, name="offsets")
    masks = alloc((_P_MAX, _GROUP), nl.bfloat16, name="masks")  # exact in bf16: 0/1
    # exact in bf16: one bf16 affinity per row
    affinity_rows = alloc((_P_MAX, n_groups), nl.bfloat16, name="affinity_rows")
    # exact in bf16: 0/1 x bf16 affinity; two scale groups: [m, (s, g, t)], 0 also where row m is not on group s
    row_weights = alloc((_P_MAX, n_scale_sets * n_groups * T), nl.bfloat16, name="row_weights")
    scale_set_rows = None
    if n_scale_sets == 2:
        # [m, (s, g, t)] = token_rows[m, (g, t)] * down_group_mask[m, s] (0/1, exact in bf16)
        scale_set_rows = alloc((_P_MAX, 2 * n_groups * T), nl.bfloat16, name="scale_set_rows")
    if static_dedupe:
        # Step 3.6 (layouts in the docstring of selective_load_packed_body, dedupe_experts).
        pair_order = alloc((_P_MAX, n_pairs), nl.float32, name="dedupe_pair_order")
        earlier_mask = alloc((_P_MAX, n_pairs), nl.bfloat16, name="dedupe_earlier_mask")
        pair_identity = alloc((_P_MAX, n_pairs), nl.bfloat16, name="dedupe_pair_identity")
        same_earlier = alloc((_P_MAX, n_pairs), nl.bfloat16, name="dedupe_same_earlier")
        earlier_count = alloc((_P_MAX, 1), nl.float32, name="dedupe_earlier_count")
        count_diag = alloc((_P_MAX, n_pairs), nl.bfloat16, name="dedupe_count_diag")
        index_dedupe = alloc((_P_MAX, n_pairs), nl.int32, name="index_dedupe")
        # Dedupe columns: gathered positions 1 .. G - 1 (gate/up), then the down weights of groups 1 .. N/16 - 1.
        n_dedupe_gate_up, n_dedupe_down = n_gathered - 1, n_groups - 1
        n_dedupe_cols = n_dedupe_gate_up + n_dedupe_down
        partition_f = alloc((_P_MAX, 3), nl.float32, name="dedupe_partition_f")  # a, a % 16, a / 16
        column_ids = alloc((_P_MAX, _GROUP), nl.float32, name="dedupe_column_ids")  # [n, c] = c
        column_groups = alloc((_P_MAX, n_dedupe_cols), nl.bfloat16, name="dedupe_column_groups")
        column_counts = alloc((_P_MAX, n_dedupe_cols), nl.bfloat16, name="dedupe_column_counts")
        pair_select = alloc((_P_MAX, 2 * _P_MAX), nl.bfloat16, name="dedupe_pair_select")
        # Away from gather_index: a DMA reading gather_index must not wait for the writes of this tile.
        gather_index_dedupe = alloc(
            (_P_MAX, n_dedupe_gate_up * sub_pairs + n_dedupe_down), nl.int32, name="gather_index_dedupe", align=512
        )
        slot_expert = alloc((_P_MAX, 1), nl.float32, name="dedupe_slot_expert")
        same_affinity = alloc((_P_MAX, n_pairs), nl.bfloat16, name="dedupe_same_affinity")
        slot_weights = alloc((_P_MAX, T), nl.bfloat16, name="dedupe_slot_weights")
        dedupe_row_select = alloc((_P_MAX, n_pairs * _MX_ROWS), nl.bfloat16, name="dedupe_row_select")
    if dynamic:
        # Step 3.0 tiles (layouts in the docstring of selective_load_packed_body, dedupe_experts).
        dd_ident = alloc((_P_MAX, n_pairs), nl.float32, name="dd_ident")
        dd_upper = alloc((_P_MAX, n_pairs), nl.bfloat16, name="dd_upper")
        dd_iota_slot = alloc((_P_MAX, n_pairs), nl.float32, name="dd_iota_slot")
        dd_iota_row = alloc((_P_MAX, n_pairs), nl.int32, name="dd_iota_row")
        dd_iota_group = alloc((_P_MAX, n_groups * T), nl.int32, name="dd_iota_group")
        dd_token_mask = alloc((_P_MAX, n_groups * T), nl.float32, name="dd_token_mask")
        dd_ones = alloc((_P_MAX, _P_MAX), nl.float16, name="dd_ones")
        dd_prod = alloc((_P_MAX, n_pairs), nl.float32, name="dd_prod")
        dd_id = alloc((_P_MAX, 1), nl.float32, name="dd_id")
        dd_lower = alloc((_P_MAX, n_pairs), nl.float32, name="dd_lower")
        dd_earlier = alloc((_P_MAX, 1), nl.float32, name="dd_earlier")
        dd_first = alloc((_P_MAX, 1), nl.bfloat16, name="dd_first")
        dd_kept = alloc((_P_MAX, 1), nl.bfloat16, name="dd_kept")
        # [a, 0]: a < 16 * dyn_verbatim_groups, [a, 1]: 1 - that, [a, 2]: a times [a, 0], [a, 3]: a
        dd_verbatim = alloc((_P_MAX, 4), nl.float32, name="dd_verbatim")
        dd_map = alloc((_P_MAX, 1), nl.float32, name="dd_map")
        dd_index_first = alloc((_P_MAX, 1), nl.int32, name="dd_index_first") if index_first else None
        dd_same = alloc((_P_MAX, n_pairs), nl.bfloat16, name="dd_same")
        dd_slot = alloc((_P_MAX, 1), nl.float32, name="dd_slot")
        dd_first_slot = alloc((_P_MAX, 1), nl.bfloat16, name="dd_first_slot")
        dd_first_id = alloc((_P_MAX, 1), nl.float32, name="dd_first_id")
        dd_scatter = alloc((_P_MAX, n_pairs), nl.float16, name="dd_scatter")
        dd_count = alloc((_P_MAX, 1), nl.int32, name="dd_count")
        dd_slot_int = alloc((_P_MAX, 4), nl.int32, name="dd_slot_int")
        dd_slot_f = alloc((_P_MAX, 4), nl.float32, name="dd_slot_f")
        dd_affinity = alloc((_P_MAX, 1), nl.float32, name="dd_affinity")
        dd_slot_rows = alloc((_P_MAX, _P_MAX), nl.bfloat16, name="dd_slot_rows")
        dd_token_affinity = alloc((_P_MAX, n_groups * T), nl.float32, name="dd_token_affinity")
        dd_group_affinity = alloc((_P_MAX, n_groups * T), nl.bfloat16, name="dd_group_affinity")
        # cidx_both: [cidx | cidx0], the compacted index (padding PAD) and its copy with expert 0 for padding.
        cidx_both = alloc((_P_MAX, 2 * n_pairs), nl.int32, name="dd_cidx_both")
        cidx, cidx0 = cidx_both[:, 0:n_pairs], cidx_both[:, n_pairs : 2 * n_pairs]
        dd_flags = alloc((_P_MAX, n_groups), nl.int32, name="dd_flags")
        dd_trips = alloc((_P_MAX, n_groups), nl.int32, name="dd_trips")
        dd_tail_ids = alloc((1, n_pairs), nl.int32, name="dd_tail_ids")  # dyn_tail_gate: late copy of the tail ids
        if direct_g1:
            # [n, a]: 17 + a % 16 (the incl of slot 16 + a % 16), the one-hot select [n, a], the fp16 expert ids, and a
            # DVE copy of the expert ids of the gate/up and down-weight gathers
            dd_g1_pattern = alloc((_P_MAX, _P_MAX), nl.float32, name="dd_g1_pattern")
            dd_g1_select = alloc((_P_MAX, _P_MAX), nl.float16, name="dd_g1_select")
            dd_id16 = alloc((_P_MAX, 1), nl.float16, name="dd_id16")
            dd_experts_late = alloc((_P_MAX, n_gathered + n_groups), nl.int32, name="dd_experts_late")
        # Stage A (dyn_fast_g1): pairs 0 .. 3*16 - 1 on partitions; [n, a]: a % 16 + 17 (the incl of slot 16 + a%16)
        n_a = 3 * _GROUP
        da_slot_pattern = alloc((n_a, _P_MAX), nl.float32, name="da_slot_pattern")
        da_slot_int = alloc((n_a, _P_MAX), nl.int32, name="da_slot_int")
        da_id = alloc((n_a, 1), nl.float32, name="da_id")
        da_id16 = alloc((n_a, 1), nl.float16, name="da_id16")
        da_earlier = alloc((n_a, 1), nl.float32, name="da_earlier")
        da_first = alloc((n_a, 1), nl.bfloat16, name="da_first")
        da_prod = alloc((n_a, n_a), nl.float32, name="da_prod")
        da_select = alloc((n_a, _P_MAX), nl.float16, name="da_select")
        # shift of the later slots: h = max(0, 32 - incl[47]) on the pairs >= 48 (the holes of group 1)
        dd_pick47 = alloc((_P_MAX, _P_MAX), nl.bfloat16, name="dd_pick47")
        dd_ge48 = alloc((_P_MAX, 1), nl.float32, name="dd_ge48")
        dd_room = alloc((_P_MAX, 1), nl.bfloat16, name="dd_room")
        dd_shift = alloc((_P_MAX, 1), nl.float32, name="dd_shift")
        dd_incl = alloc((_P_MAX, 1), nl.float32, name="dd_incl")
    gate_act = alloc((T, 2 * group_cols), nl.float32, name="gate_act")
    if situ != None:
        # SiTU (Step 6.3): rings of 2 groups like gate_act, [t, (c, I)]: sigmoid(g) (or tanh(g/2)) and tanh(u/lb).
        situ_sigmoid = alloc((T, 2 * group_cols), nl.float32, name="situ_sigmoid")
        situ_up = alloc((T, 2 * group_cols), nl.float32, name="situ_up")
    keep_alive_src, keep_alive_sink, tail_sink = None, None, None  # PE keep-alive tiles (pe_keep_alive*)
    if pe_keep_alive > 0 or pe_keep_alive_tail > 0:
        kernel_assert(not wide_out_ps, "pe_keep_alive needs PSUM banks 6-7 (not with a wide fp32 out_ps)")
        kernel_assert(
            _P_MAX <= pe_keep_alive_cols <= 4 * _P_MAX,
            f"pe_keep_alive_cols must be in 128 .. 512, got {pe_keep_alive_cols}",
        )
        keep_alive_src = alloc((_P_MAX, pe_keep_alive_cols), nl.bfloat16, name="keep_alive_src")  # ones
    if pe_keep_alive > 0:
        # The keep-alive PSUM tile and the dynamic compaction tiles share bank 7 at fixed addresses, and no PSUM bank
        # is free there with dynamic dedupe, so the keep-alive output would overwrite compaction values.
        kernel_assert(
            not dynamic,
            "pe_keep_alive needs PSUM bank 7, which dedupe_experts='dynamic' uses for its compaction tiles",
        )
        keep_alive_sink = alloc((T, 1), nl.float32, name="keep_alive_sink")
    if pe_keep_alive_tail > 0:
        tail_sink = alloc((T, 1), nl.float32, name="keep_alive_tail_sink")
    gated = alloc((T, 2 * _X4 * _P_MAX), nl.bfloat16, name="gated")
    qmx_in_dtype = nl.bfloat16 if hi_lo or post_down else nl.float16
    qmx_in_cols = n_pairs if post_down else stationary_cols  # post_down quantizes only the pair-slot block
    qmx_in = alloc((_P_MAX, n_groups * n_scale_sets * qmx_in_cols * _X4), qmx_in_dtype, name="qmx_in")
    inter = alloc((_P_MAX, n_groups * group_stride), nl.float8_e4m3fn_x4, name="inter")
    inter_s = alloc((_P_MAX, n_groups * group_stride), nl.uint8, name="inter_s")
    # fp32 down PSUM with hi / lo (fp32_down_rounding): "once" keeps fp32 in out_sb and the store adds hi and lo in
    # fp32; "terms" evicts bf16 terms (dynamic dedupe: out_sb stays fp32 for the tails, out_terms holds the bf16 terms).
    fp32_once = down_fp32 and hi_lo and fp32_down_rounding == "once"
    terms_tile = down_fp32 and hi_lo and fp32_down_rounding == "terms" and dynamic
    out_sb_dtype = nl.float32 if fp32_once or terms_tile else nl.bfloat16
    out_sb = alloc((stationary_cols, out_cols), out_sb_dtype, name="out_sb")
    out_terms, latent_terms = out_sb, None
    if terms_tile and latent_input and latent_quant == "transpose":
        # The bf16 latent copy (Step 2.5) and out_terms (written at the end) share one tile: the latent is dead after
        # its quantization, and two tiles do not fit SBUF.
        latent_terms = alloc((max(stationary_cols, T), max(H, out_cols)), nl.bfloat16, name="latent_terms")
        out_terms = latent_terms[0:stationary_cols, 0:out_cols]
    elif terms_tile:
        out_terms = alloc((stationary_cols, out_cols), nl.bfloat16, name="out_terms")
    tail_sum, tail_final, store_zero = None, None, None
    if dyn_head_store:
        """dyn_head_store: a tail stores the whole output again inside its block, where a static DMA cannot write the
        output (and dma_compute is static). The PE adds the hi and lo rows of each output half (tail_sum[h][p, t] = 1
        for p = hT + t and token_cols + hT + t; exact in the fp32 PSUM), DVE rounds the sums to bf16 into tail_final
        (fp32, then bf16: the arithmetic of the hi + lo DMA of the head), and one HWDGE DMA with a register offset of
        0 (a dynamic DMA) stores each half."""
        pool = nisa.gpsimd_engine
        tail_sum = []
        for half in range(n_out_halves):
            ident = alloc((stationary_cols, T), nl.float32, name=f"tail_sum_v{half}")
            stat = alloc((stationary_cols, T), nl.bfloat16, name=f"tail_sum{half}")
            nisa.iota(dst=ident, pattern=[[-1, T]], offset=-half * T, channel_multiplier=1, name=f"iota_tail_sum{half}")
            nisa.tensor_scalar(
                dst=stat, data=ident, op0=nl.equal, operand0=0.0, engine=pool, name=f"tail_sum_hi{half}"
            )  # p - t - hT == 0: the hi row of token t
            if hi_lo:
                lo = alloc((stationary_cols, T), nl.bfloat16, name=f"tail_sum_lo{half}")
                nisa.tensor_scalar(
                    dst=lo,
                    data=ident,
                    op0=nl.equal,
                    operand0=float(token_cols),
                    engine=pool,
                    name=f"tail_sum_lo_op{half}",
                )  # p - t - hT == token_cols: the lo row of token t
                nisa.tensor_tensor(dst=stat, data1=stat, data2=lo, op=nl.add, engine=pool, name=f"tail_sum_op{half}")
            tail_sum.append(stat)
        tail_final = alloc((n_out_halves * T, out_cols), nl.bfloat16, name="tail_final")
        store_zero = alloc((1, 1), nl.int32, name="store_zero")
        nisa.memset(dst=store_zero, value=0, engine=pool, name="memset_store_zero")
    if post_down:
        # slot_mask[m, (g, k, t)] = 1 if row m of group g (pair 16g + m/8) is pair t*K + k; aff_rows[r] = the affinity
        # of the pair of output row r (r = h*T*K + 8k + t).
        slot_val = alloc((_P_MAX, n_groups * n_pairs), nl.int32, name="slot_val")
        slot_mask = alloc((_P_MAX, n_groups * n_pairs), nl.bfloat16, name="slot_mask")
        row_pair = alloc((_P_MAX, 3), nl.int32, name="row_pair")
        pair_ids = alloc((_P_MAX, n_pairs), nl.int32, name="pair_ids")
        row_select = alloc((_P_MAX, n_pairs), nl.float32, name="row_select")
        row_affinity = alloc((_P_MAX, n_pairs), nl.float32, name="row_affinity")
        aff_rows = alloc((_P_MAX, 1), nl.float32, name="aff_rows")
        # token_sum[r, 8h + t] = 1 if output row r = h*T*K + 8k + t (any k): the stationary that adds the K pair
        # rows of each token; out_tokens [Hh * T, out_cols] holds the summed output (partition h*T + t).
        sum_fields = alloc((_P_MAX, 2), nl.int32, name="sum_fields")
        sum_cols = alloc((_P_MAX, n_out_halves * T), nl.int32, name="sum_cols")
        token_sum = alloc((_P_MAX, n_out_halves * T), nl.bfloat16, name="token_sum")
        out_tokens = alloc((n_out_halves * T, out_cols), nl.bfloat16, name="out_tokens")
    if sbuf_reserve != None:
        """sbuf_reserve: the weight buffers come last (512 B aligned), with the deepest rings (of the order of Step 1.2:
        Rm = 3 with all row slots, then Rm = 2 with fewer row slots) that leave sbuf_reserve bytes free between the
        stack and the heap (and _LATE_SBUF_BYTES for the later allocations)."""
        weights_base = align_to(sbm.get_stack_curr_addr(), 512)
        weights_limit = sbm.heap_curr_addr - sbuf_reserve - _LATE_SBUF_BYTES

        def weights_end(n_moving: int, n_rows: int) -> int:
            """The stack address after alloc_weights(align=512) with these ring depths."""
            if weights_arena:
                # one tensor: the rows, the down weights and the scales, then the moving rings
                return (
                    weights_base
                    + n_rows * _GROUP * row_bytes
                    + n_groups * H * 2
                    + n_scale_tiles * n_scale_sets * H
                    + (n_moving * moving_slot_bytes)
                )
            end = weights_base + n_rows * _GROUP * row_bytes + n_moving * moving_slot_bytes  # rows, moving rings
            end = align_to(end, 512) + n_groups * H * 2  # down_w (2-byte float4_e2m1fn_x4 elements)
            return align_to(end, 512) + n_scale_tiles * n_scale_sets * H  # down_s

        n_moving_slots, n_gate_up_slots = ring_depths(weight_budget)
        while weights_end(n_moving_slots, n_gate_up_slots) > weights_limit and n_gate_up_slots > 2:
            if n_moving_slots == 3:
                n_moving_slots = 2  # 3 moving slots only with all row slots (Step 1.2)
            else:
                n_gate_up_slots -= 1
        gate_up_w_base = align_to(sbm.get_stack_curr_addr(), 512)
        gate_up_w, gate_up_moving, gate_up_moving_s, down_w, down_s = alloc_weights(align=512)
        kernel_assert(
            sbm.heap_curr_addr - sbm.get_stack_curr_addr() >= sbuf_reserve + _LATE_SBUF_BYTES,
            f"sbuf_reserve={sbuf_reserve} does not fit: {sbm.heap_curr_addr - sbm.get_stack_curr_addr()} B free with "
            f"{n_gate_up_slots} gate/up row slots and {n_moving_slots} moving slots",
        )

    # Step 1.4: Allocate the PSUM tiles at fixed banks (layouts in the kernel docstring).
    """Gate/up of group g is in bank g % 2, so gate/up(g + 1) never waits for ACT / DVE still reading group g. With a
    wide fp32 out_ps (banks 2-7) the tiles cover all partitions, and the gated values of group g use columns
    _PSUM_COLS - 4T .. _PSUM_COLS - 1 of its tile (between the gate and up columns): the compiler rejects two PSUM tiles
    in one bank whose matmul writes interleave, so gated_ps cannot get its own tile in banks 0-1."""
    gate_up_ps_by_parity = [
        nl.ndarray(
            (_P_MAX if wide_out_ps else T, gate_up_row),
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
    # Only the dense HBM affinities read it (sbuf_reserve: not declared; a stack of layers has many expert tiles).
    affinity_t_ps = None
    if sbuf_reserve == None:
        affinity_t_ps = nl.ndarray(
            (_P_MAX, n_expert_tiles * T),
            dtype=nl.float32,
            buffer=nl.psum,
            address=(0, 6 * _PSUM_BANK_BYTES + 1536),
            name="affinity_t_ps",
        )
    # dedupe_experts (Step 3.6): the counts of the dedupe columns, the counts of all pairs on every partition, and the
    # row weights.
    dedupe_ps, row_weights_ps, column_counts_ps = None, None, None
    if static_dedupe:
        column_counts_ps = nl.ndarray(
            (_P_MAX, n_dedupe_cols),
            dtype=nl.float32,
            buffer=nl.psum,
            address=(0, 7 * _PSUM_BANK_BYTES + 1024),
            name="column_counts_ps",
        )
        dedupe_ps = nl.ndarray(
            (_P_MAX, n_pairs),
            dtype=nl.float32,
            buffer=nl.psum,
            address=(0, 6 * _PSUM_BANK_BYTES + 512),
            name="dedupe_ps",
        )
        row_weights_ps = nl.ndarray(
            (_P_MAX, n_groups * T),
            dtype=nl.float32,
            buffer=nl.psum,
            address=(0, 6 * _PSUM_BANK_BYTES + 1280),
            name="row_weights_ps",
        )
    # Router tiles only: the broadcast index (columns 0..63) and affinities (64..127).
    routing_ps = nl.ndarray(
        (_P_MAX, 2 * n_pairs), dtype=nl.float32, buffer=nl.psum, address=(0, 6 * _PSUM_BANK_BYTES), name="routing_ps"
    )
    # PE keep-alive outputs: bank 7 before gate/up(0) (free until the gated transposes), bank 6 before the last
    # gate/up (the index, affinity and routing tiles are done).
    keep_alive_ps, tail_ps = None, None
    if pe_keep_alive > 0:
        keep_alive_ps = nl.ndarray(
            (_P_MAX, pe_keep_alive_cols),
            dtype=nl.float32,
            buffer=nl.psum,
            address=(0, 7 * _PSUM_BANK_BYTES),
            name="keep_alive_ps",
        )
    if pe_keep_alive_tail > 0:
        tail_ps = nl.ndarray(
            (_P_MAX, pe_keep_alive_cols),
            dtype=nl.float32,
            buffer=nl.psum,
            address=(0, 6 * _PSUM_BANK_BYTES),
            name="keep_alive_tail_ps",
        )
    lh_ps = None
    if last_group_halves:
        # The gate / up sums of the second half of the last group, in bank 6 (free in the two-phase call), at the
        # columns of the gate / up tile.
        lh_ps = nl.ndarray(
            (T, gate_up_row), dtype=nl.bfloat16, buffer=nl.psum, address=(0, 6 * _PSUM_BANK_BYTES), name="lh_ps"
        )
    last_up_ps = None
    if last_gate_up_split:
        # The up sums of the last group [t, (c, I)] in bank 6, which the two-phase call (broadcast routing) leaves free.
        kernel_assert(
            routing_broadcast and not gate_up_fp32 and pe_keep_alive_tail == 0,
            "last_gate_up_split needs the broadcast routing tiles, bf16 gate/up PSUM and no tail keep-alive",
        )
        last_up_ps = nl.ndarray(
            (T, _PSUM_COLS), dtype=nl.bfloat16, buffer=nl.psum, address=(0, 6 * _PSUM_BANK_BYTES), name="last_up_ps"
        )
    gated_ps = None
    if not wide_out_ps:
        gated_ps = nl.ndarray(
            (_P_MAX, 2 * _X4 * T), dtype=nl.bfloat16, buffer=nl.psum, address=(0, 7 * _PSUM_BANK_BYTES), name="gated_ps"
        )
    if output == None:
        output = nl.ndarray((T, H), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="output")
    else:
        kernel_assert(
            output.shape[1] == H and 0 <= output_row and output_row + T <= output.shape[0],
            f"output needs {H} columns and rows {output_row} .. {output_row + T - 1}, got {output.shape}",
        )
    valid = T if valid_tokens == None else valid_tokens  # the stored tokens
    kernel_assert(1 <= valid <= T, f"valid_tokens must be in 1 .. {T}, got {valid_tokens}")
    output_rows = slice(output_row, output_row + valid)  # the output rows of the stored tokens
    if latent_input:
        # hidden_input="bf16" (Step 2.2): the transposed latent [p, (hb, q, t)] in bank 1 (free until gate/up(1)).
        latent_ps = nl.ndarray(
            (_P_MAX, n_h512 * _X4 * T),
            dtype=nl.bfloat16,
            buffer=nl.psum,
            address=(0, _PSUM_BANK_BYTES),
            name="latent_ps",
        )
    if dynamic:
        # Step 3.0 PSUM (bank 7, before the gated transposes): the compacted index, the row weights, the prefix
        # count and the slot map.
        dd_cidx_ps = nl.ndarray(
            (_P_MAX, n_pairs), dtype=nl.float32, buffer=nl.psum, address=(0, 7 * _PSUM_BANK_BYTES), name="dd_cidx_ps"
        )
        dd_rw_ps = nl.ndarray(
            (_P_MAX, n_groups * T),
            dtype=nl.float32,
            buffer=nl.psum,
            address=(0, 7 * _PSUM_BANK_BYTES + 512),
            name="dd_rw_ps",
        )
        dd_incl_ps = nl.ndarray(
            (_P_MAX, 1), dtype=nl.float32, buffer=nl.psum, address=(0, 7 * _PSUM_BANK_BYTES + 1024), name="dd_incl_ps"
        )
        da_incl_ps = nl.ndarray(
            (3 * _GROUP, 1),
            dtype=nl.float32,
            buffer=nl.psum,
            address=(0, 7 * _PSUM_BANK_BYTES + 1280),
            name="da_incl_ps",
        )
        da_expert_ps = nl.ndarray(
            (_P_MAX, 1), dtype=nl.float32, buffer=nl.psum, address=(0, 7 * _PSUM_BANK_BYTES + 1792), name="da_expert_ps"
        )
        dd_shift_ps = nl.ndarray(
            (_P_MAX, 1), dtype=nl.float32, buffer=nl.psum, address=(0, 7 * _PSUM_BANK_BYTES + 1920), name="dd_shift_ps"
        )
        dd_map_ps = nl.ndarray(
            (_P_MAX, 1), dtype=nl.float32, buffer=nl.psum, address=(0, 7 * _PSUM_BANK_BYTES + 1536), name="dd_map_ps"
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
    dve, act = nisa.vector_engine, nisa.scalar_engine

    # stream_chain: the last Pool stream instruction of the earlier calls (the setup of this call follows it)
    after_stream = stream_chain[-1] if stream_chain else None

    def after_norm_op(op: Callable[..., nisa.NkiInstruction]) -> Callable[..., nisa.NkiInstruction]:
        """Wrap op so that its instruction is scheduled after after_norm (fused), and the setup does not delay the
        RMSNorm (and after after_stream, so it does not hold the Pool stream of an earlier call)."""

        def scheduled(*args, **kwargs) -> nisa.NkiInstruction:
            instruction = op(*args, **kwargs)
            if after_norm != None:
                instruction.depends_on(after_norm, kind=nisa.dependency.schedule)
            if after_stream != None:
                instruction.depends_on(after_stream, kind=nisa.dependency.schedule)
            return instruction

        return scheduled

    iota, memset = after_norm_op(nisa.iota), after_norm_op(nisa.memset)

    # Step 1.5: post_down: zero the down-projection stationary. Else zero qmx_in for Hh = 2 (its first token blocks).
    if post_down:
        # The stationary columns before the pair-slot block of each group stay zero (with valid MX scales).
        memset(dst=inter.view(nl.int32), value=0, name="memset_inter")
        memset(dst=inter_s, value=0, name="memset_inter_s")
    elif n_out_halves > 1:
        memset(dst=qmx_in, value=0.0, name="memset_qmx_in")

    # Step 2: Load the routing, the hidden and the group-0 weights.
    act_latent_load = None  # hidden_input="bf16": the last latent load on the ACT queue (Step 2.5)
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
        hidden_engine = nisa.engine.sync if early_head or hidden_queue == "sp" else nisa.engine.scalar
        nisa.dma_copy(
            dst=index_row,
            src=expert_index.reshape((1, n_pairs)),
            dge_mode=nisa.dge_mode.hwdge,
            engine=nisa.engine.sync,
            name="load_index",
        )
        if latent_input:
            # hidden_input="bf16" (Step 2.5): load the latent (4 shifted copies) and quantize it to the MXFP8 stationary.
            latent_loads, _ = quantize_latent_mx(
                hidden_qtz,
                alloc,
                latent_ps,
                hidden_raw,
                hidden_raw_s,
                (hidden_engine, nisa.engine.scalar),  # the SP and ACT queues in turn, ahead of their expert loads
                form=latent_quant,
                latent_tile=latent_terms,
                name_suffix=name_suffix,
            )
            # The SP expert loads follow the SP latent loads (hidden_queue), the ACT expert loads the ACT ones.
            hidden_s_load = latent_loads[-2] if len(latent_loads) > 1 else latent_loads[-1]
            act_latent_load = latent_loads[-1] if len(latent_loads) > 1 else None
        else:
            nisa.dma_copy(
                dst=hidden_raw,
                src=hidden_qtz.reshape((_P_MAX, n_h512 * T)),
                dge_mode=nisa.dge_mode.hwdge,
                engine=hidden_engine,
                name="load_hidden",
            )
            hidden_s_load = nisa.dma_copy(
                dst=hidden_raw_s,
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

    # Step 2.3: Sub-group 0: one HWDGE DMA per expert (weights and scales), alternating SP and ACT (long stream: then
    # hwdge_group on SP).
    def load_expert(
        pair: int,
        engine: nisa.engine,
        id_row: Optional[nl.NkiTensor] = None,
        oob_skip: bool = False,
        priority: Optional[int] = None,
    ) -> nisa.NkiInstruction:
        """One HWDGE DMA of the 128 packed rows of the expert of pair into its row slot (group pair / 16).

        id_row (dynamic dedupe): the expert of slot `pair` is id_row[0, pair] (the compacted index). Static dedupe:
        the loads of hwdge_group read the deduped expert id (Step 3.6), so a repeated expert is out of bounds and its
        DMA moves no data.
        """
        if static_dedupe and pair // _GROUP == hwdge_group:
            id_row, oob_skip = index_dedupe, True
        if id_row != None:
            expert = nisa.scalar_load(id_row[0:1, pair : pair + 1])
        elif hwdge_index != None and hwdge_index.shape[0] == 1:
            expert = nisa.scalar_load(hwdge_index[0:1, pair : pair + 1])  # the caller's row [1, N]
        elif hwdge_index != None:
            # the caller's tile [t, k] on the token partitions (pair n = t*K + k)
            expert = nisa.scalar_load(hwdge_index[pair // K : pair // K + 1, pair % K : pair % K + 1])
        elif expert_index_tk != None:
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
            oob_mode=nisa.oob_mode.skip if oob_skip else nisa.oob_mode.error,
            priority=priority,
            name=f"load_gate_up_p{pair}",
        )

    hw_loads = []  # group0_on_pool: group 0 comes from a Pool gather (Step 4)
    if not group0_on_pool:
        hw_loads = [load_expert(pair, (nisa.engine.sync, nisa.engine.scalar)[pair % 2]) for pair in range(sub_pairs)]
    n_act_loads = (_LONG_ACT_HWDGE_LOADS if fused_schedule else 0) if hwdge_act_loads == None else hwdge_act_loads
    kernel_assert(0 <= n_act_loads < _GROUP, f"hwdge_act_loads must be in 0 .. {_GROUP - 1}, got {n_act_loads}")
    sp_last = [hw_loads[-2] if hw_loads else None]  # the last HWDGE expert load on SP so far

    def load_hwdge_group(id_row: Optional[nl.NkiTensor] = None) -> None:
        """Long stream: the SP queue then loads the 16 experts of hwdge_group, so Pool has one gate/up gather less.
        SP issues the experts one at a time, so the group must not be needed soon after the index.
        Static dedupe: the loads follow Step 3.6 in program order (they read the deduped expert ids). Dynamic dedupe:
        the expert ids come from the compacted index (id_row), after Step 3.0."""
        kernel_assert(hwdge_group < n_gate_up_slots, "The HWDGE group needs its own gate/up row slot")
        previous = hw_loads[-2]  # the last group-0 load on SP
        for pair in range(hwdge_group * _GROUP, (hwdge_group + 1) * _GROUP - n_act_loads):
            load = load_expert(pair, nisa.engine.sync, id_row=id_row)
            load.depends_on(previous, kind=nisa.dependency.schedule)
            previous = load
        sp_last[0] = previous

    if hwdge_group != None and not static_dedupe and not dynamic:
        load_hwdge_group()
    if not inputs_in_sbuf and hidden_queue == "sp" and not early_head:
        # hidden_queue="sp": the hidden loads go first on the SP queue, ahead of its expert loads.
        for load in hw_loads[0::2]:
            load.depends_on(hidden_s_load, kind=nisa.dependency.schedule)
        if latent_input and act_latent_load != None:
            # hidden_input="bf16": the ACT latent loads go first on the ACT queue too.
            for load in hw_loads[1::2]:
                load.depends_on(act_latent_load, kind=nisa.dependency.schedule)
    if early_head:
        # The hidden loads and the ACT table load go first on their queues.
        hw_loads[0].depends_on(hidden_s_load, kind=nisa.dependency.schedule)
        hw_loads[1].depends_on(zero_gated_col0, kind=nisa.dependency.schedule)

    if not inputs_in_sbuf:
        # Step 2.4: Standalone: load the affinities, then broadcast the index row to every partition.
        nisa.dma_copy(
            dst=affinity_row if topk_affinities else affinity_te,  # topk: [T, K] as one row (pair n = t*K + k)
            src=expert_affinities.reshape((1, n_pairs)) if topk_affinities else expert_affinities,
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

    # Step 3: Build the gather indices and the constants (gather row mapping in the kernel docstring).
    if dynamic:
        # Step 3.0.1: Constants. dd_ident[a, n] = (n == a), dd_lower[a, j] = (j < a), dd_upper[k, n] = (k <= n).
        iota(dst=dd_prod, pattern=[[1, n_pairs]], offset=0, channel_multiplier=-1, name="dd_iota_diff")  # n - a
        nisa.tensor_scalar(dst=dd_ident, data=dd_prod, op0=nl.equal, operand0=0, engine=dve, name="dd_ident_op")
        nisa.tensor_scalar(dst=dd_lower, data=dd_prod, op0=nl.less, operand0=0, engine=dve, name="dd_lower_op")
        nisa.tensor_scalar(dst=dd_upper, data=dd_prod, op0=nl.greater_equal, operand0=0, engine=dve, name="dd_upper_op")
        iota(dst=dd_iota_slot, pattern=[[1, n_pairs]], offset=1, channel_multiplier=0, name="dd_iota_slot_op")  # u+1
        # row-weight constants: [n, m] m / 8, [n, (g, t)] g, [n, (g, t)] (t == n / 16)
        iota(dst=dd_iota_row, pattern=[[1, _P_MAX]], offset=0, channel_multiplier=0, name="dd_iota_row_op")
        nisa.tensor_scalar(
            dst=dd_iota_row, data=dd_iota_row, op0=nl.right_shift, operand0=3, engine=dve, name="dd_row_slot"
        )
        iota(
            dst=dd_iota_group, pattern=[[1, n_groups], [0, T]], offset=0, channel_multiplier=0, name="dd_iota_group_op"
        )
        iota(
            dst=dd_token_mask.view(nl.int32),
            pattern=[[0, n_groups], [1, T]],
            offset=0,
            channel_multiplier=0,
            name="dd_iota_token",
        )  # [n, (g, t)]: t (int; the row-weight math compares it with n / 16)
        iota(dst=dd_slot_int[:, 3:4], pattern=[[0, 1]], offset=0, channel_multiplier=1, name="dd_iota_pair_int")
        nisa.tensor_scalar(
            dst=dd_slot_int[:, 3:4],
            data=dd_slot_int[:, 3:4],
            op0=nl.right_shift,
            operand0=4,
            engine=dve,
            name="dd_token_of_pair",
        )  # n / 16 (K = 16)

        iota(dst=dd_verbatim[:, 3:4], pattern=[[0, 1]], offset=0, channel_multiplier=1, name="dd_iota_pair")  # a
        nisa.tensor_scalar(
            dst=dd_verbatim[:, 0:1],
            data=dd_verbatim[:, 3:4],
            op0=nl.less,
            operand0=_GROUP * dyn_verbatim_groups,
            engine=dve,
            name="dd_verbatim_op",
        )
        nisa.tensor_scalar(
            dst=dd_verbatim[:, 1:2],
            data=dd_verbatim[:, 0:1],
            op0=nl.multiply,
            operand0=-1.0,
            op1=nl.add,
            operand1=1.0,
            engine=dve,
            name="dd_compacted_op",
        )
        nisa.tensor_tensor(
            dst=dd_verbatim[:, 2:3],
            data1=dd_verbatim[:, 0:1],
            data2=dd_verbatim[:, 3:4],
            op=nl.multiply,
            engine=dve,
            name="dd_verbatim_slot_op",
        )
        memset(dst=dd_ones, value=1.0, engine=dve, name="dd_memset_ones")
        if direct_g1:
            iota(
                dst=dd_g1_pattern,
                pattern=[[0, _P_MAX // _GROUP], [1, _GROUP]],
                offset=_GROUP + 1,
                channel_multiplier=0,
                name="dd_iota_g1_pattern",
            )  # [n, a]: 17 + a % 16
        # Stage A / shift constants: [n, a] a % 16 + 17, pick47[k, n] = (k == 47), ge48[n] = (n >= 48)
        iota(dst=da_slot_int, pattern=[[1, _P_MAX]], offset=0, channel_multiplier=0, name="da_iota_slot")
        nisa.tensor_scalar(
            dst=da_slot_int, data=da_slot_int, op0=nl.bitwise_and, operand0=_GROUP - 1, engine=dve, name="da_slot_mod"
        )
        nisa.tensor_scalar(
            dst=da_slot_pattern,
            data=da_slot_int,
            op0=nl.add,
            operand0=_GROUP + 1,
            engine=dve,
            name="da_slot_pattern_op",
        )
        iota(dst=dd_pick47, pattern=[[0, _P_MAX]], offset=-(3 * _GROUP - 1), channel_multiplier=1, name="dd_iota_pick")
        nisa.tensor_scalar(dst=dd_pick47, data=dd_pick47, op0=nl.equal, operand0=0.0, engine=dve, name="dd_pick47_op")
        iota(dst=dd_ge48, pattern=[[0, 1]], offset=-(3 * _GROUP), channel_multiplier=1, name="dd_iota_ge48")
        nisa.tensor_scalar(dst=dd_ge48, data=dd_ge48, op0=nl.greater_equal, operand0=0.0, engine=dve, name="dd_ge48_op")
        # Step 3.0: Zero the scale segments of the row slots (Pool; padding rows keep stale bytes).
        """A skipped (padding) gather row keeps the stale bytes of its slot. FP4 weights are always finite, but an MX
        scale byte 0xFF is NaN and 0 * NaN = NaN, so Pool zeroes the scale segments of the gathered row slots first.
        The down scales of padding slots load expert 0 (cidx0)."""
        for slot in range(1, n_gate_up_slots):
            pool_gathered = (
                slot < dyn_head_groups
                or dyn_tail_loads == "pool"
                or (dyn_tail_loads == "sp6" and slot > dyn_head_groups)
            )
            if slot == hwdge_group or (pool_gathered and dyn_pad_rows == "expert0"):
                continue  # every row of the slot is loaded (HWDGE group and expert0 padding: cidx0)
            memset(
                # [p, c, w]: scale word w of the packed row of pair c in the slot
                dst=gate_up_w.ap(
                    pattern=[
                        [gate_up_w.shape[1], _P_MAX],
                        [row_bytes // 2, _GROUP],
                        [1, row_bytes // 2 - gate_up_cols],
                    ],
                    offset=slot * _GROUP * row_bytes // 2 + gate_up_cols,
                ),
                value=0,
                engine=nisa.gpsimd_engine,
                name=f"dd_zero_scales_slot{slot}",
            )

    def load_sp_tail(id_row: nl.NkiTensor) -> None:
        """The tail groups (dyn_head_groups ..) come from SP HWDGE loads after the HWDGE group: SP is free then, and
        Pool would spend the descriptor time of a gather even for out-of-bounds (padding) rows. A padding slot has
        expert PAD: the load skips it (no packets)."""
        previous = sp_last[0]  # in SP order after the HWDGE group (else the compiler interleaves the two chains)
        n_sp_tail = {"pool": 0, "sp": n_groups - dyn_head_groups, "sp_prio": n_groups - dyn_head_groups, "sp6": 1}
        for pair in range(dyn_head_groups * _GROUP, (dyn_head_groups + n_sp_tail[dyn_tail_loads]) * _GROUP):
            # sp_prio: the lowest DMA priority, so the head's data goes first
            load = load_expert(
                pair,
                nisa.engine.sync,
                id_row=id_row,
                oob_skip=True,
                priority=3 if dyn_tail_loads == "sp_prio" else None,
            )
            if previous != None:
                load.depends_on(previous, kind=nisa.dependency.schedule)
            previous = load

    def stage_a() -> None:
        """Stage A (dyn_fast_g1): group 1 = the first 16 first-occurrence experts of pairs 16 .. 47 (tokens 1-2),
        from a 48-pair compaction, so its gather starts early. Holes (fewer than 16 new experts) load expert 0.
        Writes the gather indices of gathered sub-group 1 (columns 0 .. 15)."""
        # id[n] (diagonal), earlier[n] = the pairs j < n (of the 48) with the same expert, first = (earlier == 0)
        nisa.scalar_tensor_tensor(
            dst=da_prod,
            data=index[0:n_a, 0:n_a],
            op0=nl.multiply,
            operand0=1.0,
            op1=nl.multiply,
            operand1=dd_ident[0:n_a, 0:n_a],
            reduce_res=da_id,
            name="da_diag_id",
        )
        nisa.scalar_tensor_tensor(
            dst=da_prod,
            data=index[0:n_a, 0:n_a],  # [n, j]: id[j]
            op0=nl.equal,
            operand0=da_id,
            op1=nl.multiply,
            operand1=dd_lower[0:n_a, 0:n_a],
            reduce_res=da_earlier,
            name="da_earlier_op",
        )
        nisa.tensor_scalar(dst=da_first, data=da_earlier, op0=nl.equal, operand0=0.0, engine=dve, name="da_first_op")
        # incl[n] = sum over k <= n of first[k]: the first occurrences 16 .. 31 after token 0 have incl 17 .. 32
        nisa.nc_matmul(dst=da_incl_ps, stationary=dd_upper[0:n_a, 0:n_a], moving=da_first, name="da_prefix_mm")
        nisa.tensor_copy(dst=da_id16, src=da_id, engine=dve, name="da_id16_op")
        # select[n, a] = first[n] * (incl[n] == a % 16 + 17): pair n holds slot 16 + a % 16
        nisa.scalar_tensor_tensor(
            dst=da_select,
            data=da_slot_pattern,
            op0=nl.equal,
            operand0=da_incl_ps,
            op1=nl.multiply,
            operand1=da_first.ap(pattern=[[1, n_a], [0, _P_MAX]], offset=0),  # [n, a]: first[n]
            name="da_select_op",
        )
        # expert of slot 16 + a % 16 on partition a (0 for a hole: expert 0, affinity 0)
        nisa.nc_matmul(dst=da_expert_ps, stationary=da_select, moving=da_id16, name="da_expert_mm")
        nisa.scalar_tensor_tensor(
            dst=gather_index[:, 0:sub_pairs],
            data=da_expert_ps.ap(pattern=[[1, _P_MAX], [0, sub_pairs]], offset=0),  # [a, b]: the expert
            op0=nl.multiply,
            operand0=_P_MAX,
            op1=nl.add,
            operand1=lanes[:, 0:sub_pairs],  # [a, b]: row (128/M)b + a/M
            name="da_gather_index",
        )

    direct_g1_ops = {}  # dyn_direct_g1: the group-1 expert matmul and gather, the PAD offset op of the compaction

    def compaction() -> None:
        """Step 3.0: Compact the unique experts (layouts in the docstring of selective_load_packed_body)."""
        # Step 3.0.2: id[n] = index[n, n] on partition n (the diagonal of the broadcast index).
        nisa.scalar_tensor_tensor(
            dst=dd_prod,
            data=index,
            op0=nl.multiply,
            operand0=1.0,
            op1=nl.multiply,
            operand1=dd_ident,
            reduce_res=dd_id,  # the row sum of index * identity
            name="dd_diag_id",
        )
        # first_id = id - PAD (the scatter moving below)
        nisa.tensor_scalar(
            dst=dd_first_id, data=dd_id, op0=nl.subtract, operand0=float(_DYN_PAD), engine=dve, name="dd_first_id_op"
        )
        # Step 3.0.3: earlier[n] = the number of pairs j < n with the expert of pair n (one DVE op: the equality row
        # times the strictly-lower mask, summed); first[n] = (earlier[n] == 0).
        nisa.scalar_tensor_tensor(
            dst=dd_prod,
            data=index,  # [n, j]: id[j]
            op0=nl.equal,
            operand0=dd_id,  # id[n]
            op1=nl.multiply,
            operand1=dd_lower,
            reduce_res=dd_earlier,
            name="dd_earlier_op",
        )
        # kept[n] = first[n], or 1 for the verbatim pairs (n < 16 * dyn_verbatim_groups: own slot n, also when the
        # token repeats an expert; the repeat then loads it twice, as with dedupe_experts=None). From earlier[n]
        # directly, so first[n] (for the slot map) runs after the prefix matmul and adds no time to the chain.
        nisa.tensor_scalar(
            dst=dd_kept,
            data=dd_earlier,
            op0=nl.equal,
            operand0=0.0,
            op1=nl.maximum,
            operand1=dd_verbatim[:, 0:1],
            engine=dve,
            name="dd_kept_op",
        )
        kept = dd_kept
        # Step 3.0.4: incl[n] = sum over k <= n of kept[k] (prefix count matmul); U' = incl[N - 1]. A kept pair n
        # has the unique slot incl[n] - 1; incl_kept = incl * kept is 0 for the other pairs.
        nisa.nc_matmul(dst=dd_incl_ps, stationary=dd_upper, moving=kept, name="dd_prefix_mm")
        nisa.tensor_scalar(dst=dd_first, data=dd_earlier, op0=nl.equal, operand0=0.0, engine=dve, name="dd_first_op")
        incl = dd_incl_ps
        if direct_g1:
            """dyn_direct_g1: select[n, a] = kept[n] * (incl[n] == 17 + a % 16): pair n holds unique slot 16 + a % 16. A
            matmul with the fp16 expert ids gives the expert of that slot on partition a (0 for a hole), and the
            group-1 gather goes out before the rest of the compaction."""
            nisa.tensor_copy(dst=dd_id16, src=dd_id, engine=dve, name="dd_id16_op")  # exact below 2048
            nisa.scalar_tensor_tensor(
                dst=dd_g1_select,
                data=dd_g1_pattern,  # [n, a]: 17 + a % 16
                op0=nl.equal,
                operand0=dd_incl_ps,  # incl[n]
                op1=nl.multiply,
                operand1=kept.ap(pattern=[[1, _P_MAX], [0, _P_MAX]], offset=0),  # [n, a]: kept[n]
                name="dd_g1_select_op",
            )
            direct_g1_ops["matmul"] = nisa.nc_matmul(
                dst=da_expert_ps, stationary=dd_g1_select, moving=dd_id16, name="dd_g1_expert_mm"
            )
            nisa.scalar_tensor_tensor(
                dst=gather_index[:, 0:sub_pairs],
                data=da_expert_ps.ap(pattern=[[1, _P_MAX], [0, sub_pairs]], offset=0),  # [a, b]: the expert
                op0=nl.multiply,
                operand0=_P_MAX,
                op1=nl.add,
                operand1=lanes[:, 0:sub_pairs],  # [a, b]: row (128/M)b + a/M
                name="dd_g1_gather_index",
            )
            direct_g1_ops["gather"] = nisa.dma_copy(
                dst=gate_up_w.view(nl.uint8)[:, _GROUP * row_bytes : 2 * _GROUP * row_bytes],
                # [128 * 16 rows, row]: row 128b + a is HBM row gather_index[a, b]
                src=expert_gate_up_packed.reshape((E * _P_MAX, row_bytes)).ap(
                    pattern=[[row_bytes, _P_MAX * sub_pairs], [1, row_bytes]],
                    offset=0,
                    vector_offset=gather_index[:, 0:sub_pairs],
                    indirect_dim=0,
                ),
                dge_mode=nisa.dge_mode.swdge,
                oob_mode=nisa.oob_mode.skip,
                name="gather_gate_up_g1",
            )
        if fast_first:
            """Stage A gave group 1 the first 16 first occurrences of pairs 16 .. 47. If there were fewer (holes),
            the pairs >= 48 move up by h = 32 - incl[47] slots, so group 1 keeps its holes."""
            nisa.tensor_scalar(
                dst=dd_room,
                data=dd_incl_ps,
                op0=nl.multiply,
                operand0=-1.0,
                op1=nl.add,
                operand1=2.0 * _GROUP,
                engine=dve,
                name="dd_room_op",
            )  # 32 - incl[n]
            nisa.nc_matmul(dst=dd_shift_ps, stationary=dd_pick47, moving=dd_room, name="dd_shift_mm")  # 32 - incl[47]
            nisa.tensor_scalar(
                dst=dd_shift,
                data=dd_shift_ps,
                op0=nl.maximum,
                operand0=0.0,
                op1=nl.multiply,
                operand1=dd_ge48,
                engine=dve,
                name="dd_shift_op",
            )
            nisa.tensor_tensor(dst=dd_incl, data1=dd_incl_ps, data2=dd_shift, op=nl.add, engine=dve, name="dd_incl_op")
            incl = dd_incl
        slot_op = nisa.tensor_tensor(dst=dd_slot, data1=incl, data2=kept, op=nl.multiply, engine=dve, name="dd_slot_op")
        if direct_g1:
            slot_op.depends_on(direct_g1_ops["matmul"], kind=nisa.dependency.schedule)  # DVE: the group-1 select first
        # Step 3.0.5: scatter[n, u] = (u + 1 == incl_kept[n]) * (id[n] - PAD); cidx[a, u] = sum over n of
        # scatter[n, u] + PAD = the expert of unique slot u on every partition (PAD for u >= U').
        nisa.scalar_tensor_tensor(
            dst=dd_scatter,
            data=dd_iota_slot,  # [n, u]: u + 1
            op0=nl.equal,
            operand0=dd_slot,
            op1=nl.multiply,
            operand1=dd_first_id.ap(pattern=[[1, _P_MAX], [0, n_pairs]], offset=0),  # [n, u]: id[n] - PAD
            name="dd_scatter_op",
        )
        nisa.nc_matmul(dst=dd_cidx_ps, stationary=dd_ones, moving=dd_scatter, name="dd_cidx_mm")
        direct_g1_ops["cidx"] = nisa.tensor_scalar(
            dst=cidx, data=dd_cidx_ps, op0=nl.add, operand0=float(_DYN_PAD), engine=dve, name="dd_cidx_op"
        )
        if dyn_pad_rows == "expert0":
            # cidx0: the padding slots load expert 0 (real rows, affinity 0)
            nisa.scalar_tensor_tensor(
                dst=cidx0, data=cidx, op0=nl.less, operand0=E, op1=nl.multiply, operand1=cidx, name="dd_cidx0_op"
            )

    def compaction_rest() -> None:
        """Step 3.0 (rest, after the gather indices that need only cidx): cidx0, the slot map, the group count and the
        tail flags, and the HWDGE loads that read the compacted ids."""
        # dd_same[j, n] = (id[n] == id[j]) * (j <= n): the stationary of the slot map (same expert, j not later).
        nisa.scalar_tensor_tensor(
            dst=dd_same,
            data=index,  # [j, n]: id[n]
            op0=nl.equal,
            operand0=dd_id,  # id[j]
            op1=nl.multiply,
            operand1=dd_upper,
            name="dd_same_op",
        )
        if dyn_pad_rows != "expert0":
            # cidx0: the padding slots of the down scales and the HWDGE group load expert 0 (finite; affinity 0).
            nisa.scalar_tensor_tensor(
                dst=cidx0, data=cidx, op0=nl.less, operand0=E, op1=nl.multiply, operand1=cidx, name="dd_cidx0_op"
            )
        # first_slot = first * (incl - 1): the unique slot of a first occurrence (the slot-map moving)
        nisa.scalar_tensor_tensor(
            dst=dd_first_slot,
            data=dd_incl if fast_first else dd_incl_ps,
            op0=nl.subtract,
            operand0=1.0,
            op1=nl.multiply,
            operand1=dd_first,
            name="dd_first_slot_op",
        )
        # Step 3.0.6: G = ceil(U / 16) groups (partition N - 1, where incl = U); tail flags (G == g) for the tails.
        count_op = nisa.tensor_copy(
            dst=dd_count, src=dd_incl if fast_first else dd_incl_ps, engine=dve, name="dd_count_op"
        )
        if direct_g1:
            # the group count after the compaction in DVE order (it is needed late, for the tail trip counts)
            count_op.depends_on(direct_g1_ops["cidx"], kind=nisa.dependency.schedule)
        nisa.tensor_scalar(
            dst=dd_count, data=dd_count, op0=nl.add, operand0=_GROUP - 1, engine=dve, name="dd_groups_round"
        )
        nisa.tensor_scalar(dst=dd_count, data=dd_count, op0=nl.right_shift, operand0=4, engine=dve, name="dd_groups_op")
        for tail_groups in range(dyn_head_groups + 1, n_groups + 1):
            nisa.tensor_scalar(
                dst=dd_flags[:, tail_groups - 1 : tail_groups],
                data=dd_count,
                op0=nl.equal,
                operand0=tail_groups,
                engine=dve,
                name=f"dd_flag_g{tail_groups}",
            )
        # Step 3.0.7: slot map[n] = the unique slot of the expert of pair n = sum over j <= n of same[j, n] *
        # first_slot[j] (one nonzero term: the first occurrence).
        nisa.nc_matmul(dst=dd_map_ps, stationary=dd_same, moving=dd_first_slot, name="dd_map_mm")
        if hwdge_group != None:
            load_hwdge_group(id_row=cidx0)
        if dyn_tail_gate == None:
            load_sp_tail(cidx)

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

    # Step 3.2: Pair selectors: select[a, i] = Mj + a%M (gate/up of the gathered sub-group j = gathered[i]),
    # select[a, G + g] = 16g + a/8 (down weights), select[a, G + N/16 + s] = the pair of down-scale tile s.
    a_mod_m_col, a_div_m_col = (2, 4) if sub_pairs == _GROUP else (1, 3)  # columns of a % M and a / M in partition
    down_select = select[:, n_gathered : n_gathered + n_groups]
    down_scale_select = select[:, n_gathered + n_groups : n_gathered + n_groups + n_scale_tiles]
    run_first = 0  # one iota per run of consecutive gathered sub-groups: column i holds M * gathered[i]
    for position in range(1, n_gathered + 1):
        if position == n_gathered or gathered[position] != gathered[position - 1] + 1:
            iota(
                dst=select[:, run_first:position],
                pattern=[[sub_pairs, position - run_first]],
                offset=sub_pairs * gathered[run_first],
                name="iota_select_gate_up_w" + (f"_{run_first}" if run_first > 0 else ""),
            )
            run_first = position
    nisa.tensor_tensor(
        dst=select[:, 0:n_gathered],
        data1=select[:, 0:n_gathered],
        data2=partition.ap(pattern=[[7, _P_MAX], [0, n_gathered]], offset=a_mod_m_col),  # [a, i]: a % M
        op=nl.add,
        engine=dve,
        name="select_gate_up_w",
    )
    if merged_down_gathers and not mixed_down:
        # merged_down_gathers: column 2q + b selects pair 32q + 8b + 16(a % 2) + a / 16 (layouts in the docstring)
        nisa.tensor_scalar(
            dst=merged_down_fields[:, 0:1],
            data=partition[:, 0:1],
            op0=nl.bitwise_and,
            operand0=1,
            op1=nl.left_shift,
            operand1=4,
            engine=dve,
            name="merged_down_field_lo",
        )  # 16(a % 2)
        nisa.tensor_tensor(
            dst=merged_down_fields[:, 0:1],
            data1=merged_down_fields[:, 0:1],
            data2=partition[:, 4:5],  # a / 16
            op=nl.add,
            engine=dve,
            name="merged_down_field",
        )
        iota(dst=down_select, pattern=[[2 * _GROUP, n_groups // 2], [_MX_ROWS, 2]], offset=0, name="iota_select_down_w")
        nisa.tensor_tensor(
            dst=down_select,
            data1=down_select,
            data2=merged_down_fields.ap(
                pattern=[[merged_down_fields.shape[1], _P_MAX], [0, n_groups]], offset=0
            ),  # [a, (q, b)]: field [a, 0]
            op=nl.add,
            engine=dve,
            name="select_down_w",
        )
        if quad_starts:
            # merged_down_quads: column q + b of a quad selects pair 16q + 4b + 16(a % 4) + a / 32
            nisa.tensor_scalar(
                dst=merged_down_fields[:, 2:3],
                data=partition[:, 0:1],
                op0=nl.bitwise_and,
                operand0=3,
                op1=nl.left_shift,
                operand1=4,
                engine=dve,
                name="quad_down_field_lo",
            )  # 16(a % 4)
            nisa.tensor_scalar(
                dst=merged_down_fields[:, 3:4],
                data=partition[:, 0:1],
                op0=nl.right_shift,
                operand0=5,
                engine=dve,
                name="quad_down_field_hi",
            )  # a / 32
            nisa.tensor_tensor(
                dst=merged_down_fields[:, 2:3],
                data1=merged_down_fields[:, 2:3],
                data2=merged_down_fields[:, 3:4],
                op=nl.add,
                engine=dve,
                name="quad_down_field",
            )
        for g in quad_starts:
            iota(dst=down_select[:, g : g + 4], pattern=[[4, 4]], offset=_GROUP * g, name=f"iota_select_down_w_q{g}")
            nisa.tensor_tensor(
                dst=down_select[:, g : g + 4],
                data1=down_select[:, g : g + 4],
                data2=merged_down_fields.ap(
                    pattern=[[merged_down_fields.shape[1], _P_MAX], [0, 4]], offset=2
                ),  # [a, b]: field [a, 2]
                op=nl.add,
                engine=dve,
                name=f"select_down_w_q{g}",
            )
    else:
        iota(dst=down_select, pattern=[[_GROUP, n_groups]], offset=0, name="iota_select_down_w")
        nisa.tensor_tensor(
            dst=down_select,
            data1=down_select,
            data2=partition.ap(pattern=[[7, _P_MAX], [0, n_groups]], offset=3),  # [a, g]: a / 8 for each group
            op=nl.add,
            engine=dve,
            name="select_down_w",
        )
    if mixed_down:
        # merged_down_pairs: columns g, g + 1 of a merged pair select pair 16g + 8b + 16(a % 2) + a / 16
        nisa.tensor_scalar(
            dst=merged_down_fields[:, 0:1],
            data=partition[:, 0:1],
            op0=nl.bitwise_and,
            operand0=1,
            op1=nl.left_shift,
            operand1=4,
            engine=dve,
            name="merged_down_field_lo",
        )  # 16(a % 2)
        nisa.tensor_tensor(
            dst=merged_down_fields[:, 0:1],
            data1=merged_down_fields[:, 0:1],
            data2=partition[:, 4:5],  # a / 16
            op=nl.add,
            engine=dve,
            name="merged_down_field",
        )
        for g in pair_starts:
            iota(
                dst=down_select[:, g : g + 2], pattern=[[_MX_ROWS, 2]], offset=_GROUP * g, name=f"iota_select_down_w{g}"
            )
            nisa.tensor_tensor(
                dst=down_select[:, g : g + 2],
                data1=down_select[:, g : g + 2],
                data2=merged_down_fields.ap(
                    pattern=[[merged_down_fields.shape[1], _P_MAX], [0, 2]], offset=0
                ),  # [a, b]: field [a, 0]
                op=nl.add,
                engine=dve,
                name=f"select_down_w{g}",
            )
    if merged_scales:
        """merged_scales: one gather of 256 rows ([128, 2] index) loads both down-scale tiles. Index [a, b] feeds row
        r = 128b + a, which lands on partition m = r / 2 = 64b + a / 2 of tile s = a % 2. Its pair is the down-scale pair
        of partition m in tile s: 64s + 16((m / 4) % 4) + 4(m / 32) + m % 4 = 64(a % 2) + 16((a / 8) % 4) + 4(a / 64)
        + (a / 2) % 4 + 8b (the bit fields do not overlap). Partitions with m % 32 >= 16 are not read."""
        for column, (op0, operand0, op1, operand1) in enumerate(
            (
                (nl.bitwise_and, 1, nl.left_shift, 6),  # 64(a % 2)
                (nl.left_shift, 1, nl.bitwise_and, 48),  # 16((a / 8) % 4)
                (nl.right_shift, 4, nl.bitwise_and, 4),  # 4(a / 64)
                (nl.right_shift, 1, nl.bitwise_and, 3),  # (a / 2) % 4
            )
        ):
            nisa.tensor_scalar(
                dst=scale_fields[:, column : column + 1],
                data=partition[:, 0:1],
                op0=op0,
                operand0=operand0,
                op1=op1,
                operand1=operand1,
                engine=dve,
                name=f"merged_scale_field{column}",
            )
        scale_pair = nisa.tensor_reduce(
            dst=scale_fields[:, 4:5], op=nl.add, data=scale_fields[:, 0:4], axis=1, name="merged_scale_pair"
        )
        if after_stream != None:
            # a Pool op: after the stream of the earlier calls (stream_chain)
            scale_pair = scale_pair if hasattr(scale_pair, "depends_on") else nisa.NkiInstruction(scale_pair)
            scale_pair.depends_on(after_stream, kind=nisa.dependency.schedule)
        iota(
            dst=down_scale_select, pattern=[[_MX_ROWS, n_scale_tiles]], offset=0, name="iota_select_down_s"
        )  # [a, b]: 8b
        nisa.tensor_tensor(
            dst=down_scale_select,
            data1=down_scale_select,
            data2=scale_fields.ap(pattern=[[5, _P_MAX], [0, n_scale_tiles]], offset=4),  # [a, b]: the pair field
            op=nl.add,
            engine=dve,
            name="select_down_s",
        )
    else:
        iota(
            dst=down_scale_select,
            pattern=[[_SCALE_GROUPS * _GROUP, n_scale_tiles]],
            offset=0,
            name="iota_select_down_s",
        )  # [a, s]: 64s, the first pair of down-scale tile s
        nisa.tensor_tensor(
            dst=down_scale_select,
            data1=down_scale_select,
            data2=partition.ap(pattern=[[7, _P_MAX], [0, n_scale_tiles]], offset=5),  # [a, s]: the down-scale pair
            op=nl.add,
            engine=dve,
            name="select_down_s",
        )
    if static_dedupe:
        nisa.tensor_scalar(
            dst=select[:, pair_col : pair_col + 1],  # a % N: the pair that partition a tests (Step 3.6)
            data=partition[:, 0:1],
            op0=nl.bitwise_and,
            operand0=n_pairs - 1,
            engine=dve,
            name="select_dedupe_pair",
        )

    # Step 3.3: Source rows inside each expert: lanes[a, b] = (128/M)b + a/M (gate/up), lanes[a, 16] = a%8, + SKIP
    # if a%8 >= I/4 (down weights), lanes[a, 17] = (a & 16) << 16 = SKIP if a/16 is odd (down scales).
    iota(dst=lanes[:, 0:sub_pairs], pattern=[[_P_MAX // sub_pairs, sub_pairs]], offset=0, name="iota_lane_gate_up_w")
    nisa.tensor_tensor(
        dst=lanes[:, 0:sub_pairs],
        data1=lanes[:, 0:sub_pairs],
        data2=partition.ap(pattern=[[7, _P_MAX], [0, sub_pairs]], offset=a_div_m_col),  # [a, b]: a / M for each b
        op=nl.add,
        engine=dve,
        name="lane_gate_up_w",
    )
    down_row = partition[:, 1:2]  # a % 8
    if merged_down_gathers and not mixed_down:
        nisa.tensor_scalar(
            dst=merged_down_fields[:, 1:2],
            data=partition[:, 0:1],
            op0=nl.right_shift,
            operand0=1,
            op1=nl.bitwise_and,
            operand1=_MX_ROWS - 1,
            engine=dve,
            name="merged_down_row",
        )  # (a / 2) % 8: the source row of index [a, b] of a merged down gather
        down_row = merged_down_fields[:, 1:2]
    nisa.tensor_scalar(
        dst=lanes[:, 16:17],
        data=down_row,
        op0=nl.greater_equal,
        operand0=down_rows,
        op1=nl.multiply,
        operand1=0 if down_real_rows else _SKIP,  # down_real_rows: rows p >= I/4 load their padding rows
        engine=dve,
        name="lane_down_w_skip",
    )
    nisa.tensor_tensor(
        dst=lanes[:, 16:17],
        data1=lanes[:, 16:17],
        data2=down_row,
        op=nl.add,
        engine=dve,
        name="lane_down_w",
    )
    if quad_starts:
        # merged_down_quads: (a / 4) % 8 (+ SKIP if it is a padding row) in lanes[:, 18], the source row of a quad index
        nisa.tensor_scalar(
            dst=merged_down_fields[:, 3:4],
            data=partition[:, 0:1],
            op0=nl.right_shift,
            operand0=2,
            op1=nl.bitwise_and,
            operand1=_MX_ROWS - 1,
            engine=dve,
            name="quad_down_row",
        )
        nisa.tensor_scalar(
            dst=lanes[:, 18:19],
            data=merged_down_fields[:, 3:4],
            op0=nl.greater_equal,
            operand0=down_rows,
            op1=nl.multiply,
            operand1=0 if down_real_rows else _SKIP,
            engine=dve,
            name="lane_down_w_quad_skip",
        )
        nisa.tensor_tensor(
            dst=lanes[:, 18:19],
            data1=lanes[:, 18:19],
            data2=merged_down_fields[:, 3:4],
            op=nl.add,
            engine=dve,
            name="lane_down_w_quad",
        )
    if mixed_down:
        # merged_down_pairs: (a / 2) % 8 (+ SKIP) in lanes[:, 18], the source row of the merged columns
        nisa.tensor_scalar(
            dst=merged_down_fields[:, 1:2],
            data=partition[:, 0:1],
            op0=nl.right_shift,
            operand0=1,
            op1=nl.bitwise_and,
            operand1=_MX_ROWS - 1,
            engine=dve,
            name="merged_down_row",
        )
        nisa.tensor_scalar(
            dst=lanes[:, 18:19],
            data=merged_down_fields[:, 1:2],
            op0=nl.greater_equal,
            operand0=down_rows,
            op1=nl.multiply,
            operand1=0 if down_real_rows else _SKIP,
            engine=dve,
            name="lane_down_w_merged_skip",
        )
        nisa.tensor_tensor(
            dst=lanes[:, 18:19],
            data1=lanes[:, 18:19],
            data2=merged_down_fields[:, 1:2],
            op=nl.add,
            engine=dve,
            name="lane_down_w_merged",
        )
    nisa.tensor_scalar(
        dst=lanes[:, 17:18],
        data=partition[:, 0:1],
        op0=nl.bitwise_and,
        # down_real_rows: the unused partitions load a real scale row; merged_scales: they are a & 32 (m % 32 >= 16)
        operand0=0 if down_real_rows else 32 if merged_scales else 16,
        op1=nl.left_shift,
        operand1=16,
        engine=dve,
        name="lane_down_s",
    )

    # Step 3.4: Constants, built while Pool waits for the index load.
    # Block-diagonal mask masks[m, c] = 1 if 0 <= m - 8c < 8 (offsets come from Step 2.2 with the early head).
    if not early_head:
        iota(dst=offsets, pattern=[[-_MX_ROWS, _GROUP]], offset=0, channel_multiplier=1, name="iota_block_diag_offsets")
    if not post_down and not static_dedupe:
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
    # Rows p >= I/4 of the transposed SiLU(gate) * up stay zero (with the early head ACT zeroed column 0 in Step 2.2).
    zero_from_col = 1 if early_head else 0
    memset(dst=gated[:, zero_from_col:], value=0.0, engine=dve, name="zero_gated")
    if hi_mask_tile and hi_lo:
        # hi_mask_tile: the hi mask of Step 6.4 as a tile, so that the AND is a tensor_tensor
        hi_mask = alloc((_P_MAX, 1), nl.int16, name="hi_mask")
        memset(dst=hi_mask, value=_E4M3_BF16_MASK, engine=dve, name="hi_mask_op")
    # Affinity constants: expert_id[a, h] = a + 128h and the ones stationary.
    if not inputs_in_sbuf and not topk_affinities:
        iota(dst=expert_id, pattern=[[_P_MAX, n_expert_tiles]], offset=0, channel_multiplier=1, name="iota_expert_id")
    memset(dst=ones, value=1.0, engine=nisa.engine.vector, name="memset_ones")
    if keep_alive_src != None:
        memset(dst=keep_alive_src, value=1.0, engine=nisa.engine.vector, name="memset_keep_alive_src")
    if static_dedupe:
        """Step 3.6 selectors. column_groups[n, i] = 1 if pair n belongs to the group of dedupe column i (gathered
        position 1 + i: select[n, 1 + i] == n; down group 1 + g: n / 16 == 1 + g). pair_select[n, a] = 1 if pair n
        feeds partition a: columns 0..127 for the gate/up gathers (n % 16 == a % 16), 128..255 for the down weights
        (n % 16 == a / 8); with the group mask this picks pair 16 g + a % 16 or 16 g + a / 8."""
        nisa.tensor_copy(
            dst=partition_f,
            src=partition.ap(pattern=[[7, _P_MAX], [2, 3]], offset=0),  # [a, 3]: a, a % 16, a / 16
            engine=dve,
            name="dedupe_partition_f_op",
        )
        nisa.tensor_scalar(
            dst=column_groups[:, 0:n_dedupe_gate_up],
            data=select[:, 1:n_gathered],  # the pair of gathered position 1 + i at partition n
            op0=nl.equal,
            operand0=partition_f[:, 0:1],
            engine=dve,
            name="dedupe_column_groups_gate_up",
        )
        iota(dst=column_ids, pattern=[[1, _GROUP]], offset=0, channel_multiplier=0, name="iota_dedupe_column_ids")
        nisa.tensor_scalar(
            dst=column_groups[:, n_dedupe_gate_up:],
            data=column_ids[:, 1:n_groups],  # [n, g]: 1 + g
            op0=nl.equal,
            operand0=partition_f[:, 2:3],  # n / 16
            engine=dve,
            name="dedupe_column_groups_down",
        )
        nisa.tensor_scalar(
            dst=pair_select[:, 0:_P_MAX],
            data=column_ids.ap(
                pattern=[[_GROUP, _P_MAX], [0, _P_MAX // _GROUP], [1, _GROUP]], offset=0
            ),  # [n, a]: a % 16
            op0=nl.equal,
            operand0=partition_f[:, 1:2],  # n % 16
            engine=dve,
            name="dedupe_pair_select_gate_up",
        )
        nisa.tensor_scalar(
            dst=pair_select[:, _P_MAX:],
            data=column_ids.ap(pattern=[[_GROUP, _P_MAX], [1, _GROUP], [0, _MX_ROWS]], offset=0),  # [n, a]: a / 8
            op0=nl.equal,
            operand0=partition_f[:, 1:2],  # n % 16
            engine=dve,
            name="dedupe_pair_select_down",
        )
        # earlier_mask[a, n] = 1 if pair n comes before pair a, pair_identity[a, n] = 1 if n == a (Step 3.6).
        iota(dst=pair_order, pattern=[[1, n_pairs]], offset=0, channel_multiplier=-1, name="iota_dedupe_pair_order")
        nisa.tensor_scalar(
            dst=earlier_mask, data=pair_order, op0=nl.less, operand0=0.0, engine=dve, name="dedupe_earlier_mask_op"
        )
        nisa.tensor_scalar(
            dst=pair_identity, data=pair_order, op0=nl.equal, operand0=0.0, engine=dve, name="dedupe_pair_identity_op"
        )
        # dedupe_row_select[j, (g, m)] = 1 if j == 16g + m/8: pair_identity with each column repeated over the 8 rows of a
        # pair.
        nisa.tensor_copy(
            dst=dedupe_row_select,
            src=pair_identity.ap(pattern=[[n_pairs, _P_MAX], [1, n_pairs], [0, _MX_ROWS]], offset=0),  # [j, n, p]
            engine=dve,
            name="dedupe_row_select_op",
        )
    elif not post_down and not dynamic:
        # token_rows[m, (g, t)] = 1 if row m of group g (pair 16g + m/8) belongs to token t: 0 <= m - 8Kt + 128g < 8K.
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
    elif post_down:
        # slot_mask[m, (g, k, t)] = 1 if row m of group g (pair 16g + m/8) is pair t*K + k: 8t + k - 16g == m/8.
        iota(dst=slot_val, pattern=[[-_GROUP, n_groups], [1, K], [K, T]], offset=0, name="iota_slot_pairs")
        nisa.tensor_tensor(
            dst=slot_mask,
            data1=slot_val,
            data2=partition.ap(pattern=[[7, _P_MAX], [0, n_groups * n_pairs]], offset=3),  # [m, (g, k, t)]: m / 8
            op=nl.equal,
            engine=dve,
            name="build_slot_mask",
        )
        # row_pair[r, 2] = the pair of output row r = h*T*K + 8k + t: t*K + k = K * (r % 8) + (r / 8) % K.
        nisa.tensor_scalar(
            dst=row_pair[:, 0:1],
            data=partition[:, 3:4],
            op0=nl.bitwise_and,
            operand0=K - 1,
            engine=dve,
            name="row_slot",
        )  # (r / 8) % K = k
        nisa.tensor_scalar(
            dst=row_pair[:, 1:2],
            data=partition[:, 1:2],
            op0=nl.left_shift,
            operand0=K.bit_length() - 1,
            engine=dve,
            name="row_token",
        )  # K * (r % 8) = t*K
        nisa.tensor_tensor(
            dst=row_pair[:, 2:3],
            data1=row_pair[:, 0:1],
            data2=row_pair[:, 1:2],
            op=nl.bitwise_or,
            engine=dve,
            name="row_pair_id",
        )
        iota(dst=pair_ids, pattern=[[1, n_pairs]], offset=0, name="iota_pair_ids")
        nisa.tensor_tensor(
            dst=row_select,
            data1=pair_ids,
            data2=row_pair.ap(pattern=[[3, _P_MAX], [0, n_pairs]], offset=2),  # [r, n]: pair of row r
            op=nl.equal,
            engine=dve,
            name="build_row_select",
        )
        # token_sum[r, j] = 1 if j == 8 * (r / (T*K)) + r % 8 (output half and token of row r).
        nisa.tensor_scalar(
            dst=sum_fields[:, 0:1],
            data=partition[:, 0:1],
            op0=nl.right_shift,
            operand0=n_pairs.bit_length() - 1,
            op1=nl.left_shift,
            operand1=3,
            engine=dve,
            name="sum_row_half",
        )  # 8 * (r / (T*K))
        nisa.tensor_tensor(
            dst=sum_fields[:, 1:2],
            data1=sum_fields[:, 0:1],
            data2=partition[:, 1:2],  # r % 8
            op=nl.bitwise_or,
            engine=dve,
            name="sum_row_col",
        )
        iota(dst=sum_cols, pattern=[[1, n_out_halves * T]], offset=0, name="iota_sum_cols")
        nisa.tensor_tensor(
            dst=token_sum,
            data1=sum_cols,
            data2=sum_fields.ap(pattern=[[2, _P_MAX], [0, n_out_halves * T]], offset=1),  # [r, j]: column of row r
            op=nl.equal,
            engine=dve,
            name="build_token_sum",
        )

    # Step 3.5: experts[a, j] = index[a, select[a, j]], then gather_index = rows per expert * expert + source row.
    gate_up_index_col, down_index_col = 0, n_gathered * sub_pairs
    down_scale_index_col = down_index_col + n_groups
    if early_gather:
        """early_gather: partition a needs the expert of pair n = 16 g1 + a % 16 (g1 = gathered[0]): index_ps[a, n],
        the PSUM broadcast of Step 2.4. A masked row sum over the 16 pairs of the group gives it without the index
        copy and the Pool expert gather, and its own index tile keeps the first gather free of the other gather-index
        writes."""
        first_group = gathered[0]
        iota(dst=eg_cols, pattern=[[1, _GROUP]], offset=0, channel_multiplier=0, name="eg_iota_cols")
        nisa.tensor_tensor(
            dst=eg_mask,
            data1=eg_cols,
            data2=partition.ap(pattern=[[7, _P_MAX], [0, _GROUP]], offset=2),  # [a, k]: a % 16
            op=nl.equal,
            engine=dve,
            name="eg_mask_op",
        )  # [a, k]: 1 if k == a % 16
        # broadcast routing: the caller's SBUF index instead of the PSUM broadcast
        index_source = index if routing_broadcast else index_ps
    if early_gather and first_gather_index != None:
        eg_index = first_gather_index  # the caller wrote the first gather indices
    elif early_gather:
        nisa.scalar_tensor_tensor(
            dst=eg_prod,
            data=index_source[:, first_group * _GROUP : (first_group + 1) * _GROUP],  # [a, k]: expert of pair 16 g1 + k
            op0=nl.multiply,
            operand0=1.0,
            op1=nl.multiply,
            operand1=eg_mask,
            reduce_res=eg_expert,  # the row sum: the expert of pair 16 g1 + a % 16
            name="eg_expert_op",
        )
        nisa.scalar_tensor_tensor(
            dst=eg_index,
            data=eg_expert.ap(pattern=[[1, _P_MAX], [0, sub_pairs]], offset=0),  # [a, b]: the expert
            op0=nl.multiply,
            operand0=_P_MAX,
            op1=nl.add,
            operand1=lanes[:, 0:sub_pairs],  # [a, b]: row 8b + a / 16
            name="eg_gather_index",
        )
    if tile_first:
        """The first gathered sub-group j = gathered[0] = 1 (pairs M .. 2M - 1): partition a needs the expert of pair
        n = Mj + a%M, token n / K, slot a%K (M % K == 0). tile_rows[t, a] = 1 if t == n / K, so a matmul gives
        picked[a, k] = index_tk[n / K, k]; the slot mask keeps k = a%K, and a sum over k leaves the expert."""
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
            op1=nl.bitwise_or,  # Mj + a%M for the first gathered sub-group j (M is a power of two)
            operand1=sub_pairs * gathered[0],
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
    """dedupe: the gathers of the later sub-groups (positions 1 .. G - 1) and the down weights of groups 1 .. N/16 - 1
    skip the experts that an earlier pair loads (Step 3.6), and the HWDGE group skips them too. Their row slots get
    zero scale segments before their loads: a skipped row keeps the stale bytes of its slot, FP4 weights are always
    finite, but an MX scale byte 0xFF is NaN and 0 * NaN = NaN. The HWDGE group's slot is zeroed before the index
    arrives (fused: Pool, standalone: DVE), and in the fused block Pool also zeroes every slot above the router
    temporaries (Pool is idle during the router). DVE zeroes the other slots after the dedupe counts (Step 3.6), before
    the gathers into them: a memset that waits for the router temporaries would run beside the router's last DVE ops
    and slow them (shared SBUF bus)."""
    zero_slots = []  # row slots that a dedupe group uses first
    if static_dedupe:
        dedupe_groups = {gathered[position] // n_sub_groups for position in range(1, n_gathered)}
        zero_slots = sorted({group for group in dedupe_groups if group < n_gate_up_slots})
    slot_bytes = _GROUP * row_bytes
    pool_zero_slots = [hwdge_group] if static_dedupe and hwdge_group != None else []
    if inputs_in_sbuf:
        pool_zero_slots += [slot for slot in zero_slots if gate_up_w_base + slot * slot_bytes >= caller_stack_top]
    dve_zero_slots = [slot for slot in zero_slots if slot not in pool_zero_slots]

    def zero_scale_segments(slot: int, engine: nisa.engine) -> nisa.NkiInstruction:
        """Memset of the scale segments of the 16 packed rows in row slot `slot`."""
        return nisa.memset(
            # [p, c, w]: scale word w of the packed row of pair c in the slot
            dst=gate_up_w.ap(
                pattern=[
                    [gate_up_w.shape[1], _P_MAX],
                    [row_bytes // 2, _GROUP],
                    [1, row_bytes // 2 - gate_up_cols],
                ],
                offset=slot * _GROUP * row_bytes // 2 + gate_up_cols,
            ),
            value=0,
            engine=engine,
            name=f"zero_scales_slot{slot}",
        )

    for slot in pool_zero_slots:
        # Standalone: DVE (faster than Pool, and idle until the index lands; Pool builds the gather indices first).
        zero_scale_segments(slot, nisa.gpsimd_engine if inputs_in_sbuf else dve)
    if index_first:
        # gathered sub-group 1 from the (verbatim) index: expert of pair M + a%M, then its gather index
        nisa.nc_n_gather(
            dst=dd_index_first, data=index, indices=select[:, 0:1].view(nl.uint32), name="dd_gather_first_expert"
        )
        nisa.scalar_tensor_tensor(
            dst=gather_index[:, gate_up_index_col : gate_up_index_col + sub_pairs],
            data=dd_index_first.ap(pattern=[[1, _P_MAX], [0, sub_pairs]], offset=0),  # [a, b]: expert of pair M + a%M
            op0=nl.multiply,
            operand0=_P_MAX,
            op1=nl.add,
            operand1=lanes[:, 0:sub_pairs],  # [a, b]: row (128/M)b + a/M
            name="dd_gather_index_first",
        )
    n_weight_cols = n_gathered + n_groups  # gate/up and down-weight selector columns (the scale columns follow)
    first_gather = None
    if fast_first:
        stage_a()
        # the first gate/up gather (sub-group 1) right away, ahead of the full compaction on Pool and DVE
        first_gather = nisa.dma_copy(
            dst=gate_up_w.view(nl.uint8)[:, _GROUP * row_bytes : 2 * _GROUP * row_bytes],
            # [128 * 16 rows, row]: row 128b + a is HBM row gather_index[a, b]
            src=expert_gate_up_packed.reshape((E * _P_MAX, row_bytes)).ap(
                pattern=[[row_bytes, _P_MAX * sub_pairs], [1, row_bytes]],
                offset=0,
                vector_offset=gather_index[:, 0:sub_pairs],
                indirect_dim=0,
            ),
            dge_mode=nisa.dge_mode.swdge,
            oob_mode=nisa.oob_mode.skip,
            name="gather_gate_up_g1",
        )
    if dynamic:
        compaction()
        if direct_g1:
            first_gather = direct_g1_ops["gather"]
        # the weight columns need only cidx; the scale columns read cidx0 (after compaction_rest)
        """Padding slots of the Pool gathers: dyn_pad_rows="expert0" loads expert 0 (real rows; the DMA engines are
        slower on out-of-bounds rows than on real ones), "oob" skips them (stale rows, zeroed scales)."""
        experts_gather = nisa.nc_n_gather(
            dst=experts[:, 0:n_weight_cols],
            data=cidx0 if dyn_pad_rows == "expert0" else cidx,
            indices=select[:, 0:n_weight_cols].view(nl.uint32),
            name="gather_experts",
        )
        if direct_g1:
            experts_gather.depends_on(first_gather, kind=nisa.dependency.schedule)  # Pool: the group-1 gather first
    elif dve_experts:
        """dve_experts: experts[a, j] = index[a, select[a, j]] as masked row sums on DVE (layouts in the docstring of
        dve_experts): the masks are constants (the setup builds them before the index arrives)."""
        experts_gather = None
        iota(dst=dx_mask_dw, pattern=[[-_MX_ROWS, 2], [1, 2 * _GROUP]], offset=0, name="dx_iota_dw_op")  # m - 8b
        nisa.tensor_tensor(
            dst=dx_mask_dw,
            data1=dx_mask_dw,
            data2=merged_down_fields.ap(pattern=[[merged_down_fields.shape[1], _P_MAX], [0, 4 * _GROUP]], offset=0),
            op=nl.equal,
            engine=dve,
            name="dx_mask_dw_op",
        )  # [a, (b, m)]: m == 16(a % 2) + 8b + a / 16
        if merged_scales:
            iota(dst=dx_mask_ds, pattern=[[-_MX_ROWS, n_scale_tiles], [1, n_pairs]], offset=0, name="dx_iota_ds_op")
            nisa.tensor_tensor(
                dst=dx_mask_ds,
                data1=dx_mask_ds,
                data2=scale_fields.ap(pattern=[[5, _P_MAX], [0, n_scale_tiles * n_pairs]], offset=4),
                op=nl.equal,
                engine=dve,
                name="dx_mask_ds_op",
            )  # [a, (b, n)]: n == the scale pair of partition a + 8b
        else:
            # one down-scale tile (4 groups): [a, n]: n == the down-scale pair of partition a (the select of Step 3.2)
            iota(dst=dx_mask_ds, pattern=[[1, n_scale_tiles * n_pairs]], offset=0, name="dx_iota_ds_op")
            nisa.tensor_tensor(
                dst=dx_mask_ds,
                data1=dx_mask_ds,
                data2=partition.ap(pattern=[[7, _P_MAX], [0, n_scale_tiles * n_pairs]], offset=5),
                op=nl.equal,
                engine=dve,
                name="dx_mask_ds_op",
            )
        if quad_starts:
            iota(dst=dx_mask_q, pattern=[[-4, 4], [1, 4 * _GROUP]], offset=0, name="dx_iota_q_op")  # m - 4b
            nisa.tensor_tensor(
                dst=dx_mask_q,
                data1=dx_mask_q,
                data2=merged_down_fields.ap(
                    pattern=[[merged_down_fields.shape[1], _P_MAX], [0, 16 * _GROUP]], offset=2
                ),
                op=nl.equal,
                engine=dve,
                name="dx_mask_q_op",
            )  # [a, (b, m)]: m == 16(a % 4) + 4b + a / 32
        first_gathered = gathered[0]
        nisa.tensor_tensor(
            dst=dx_prod[:, 0 : n_gathered * _GROUP],
            # [a, i, k]: the expert of pair 16 g_i + k
            data1=index.ap(
                pattern=[[n_pairs, _P_MAX], [_GROUP, n_gathered], [1, _GROUP]], offset=_GROUP * first_gathered
            ),
            data2=eg_mask.ap(
                pattern=[[_GROUP, _P_MAX], [0, n_gathered], [1, _GROUP]], offset=0
            ),  # [a, i, k]: k == a % 16
            op=nl.multiply,
            engine=dve,
            name="dx_prod_gate_up",
        )
        nisa.tensor_reduce(
            dst=experts[:, 0:n_gathered],
            op=nl.add,
            data=dx_prod.ap(pattern=[[dx_width, _P_MAX], [_GROUP, n_gathered], [1, _GROUP]], offset=0),
            axis=[2],
            name="dx_experts_gate_up",
        )
        nisa.tensor_tensor(
            dst=dx_prod[:, 0 : n_groups * 2 * _GROUP],
            # [a, q, b, m]: the expert of pair 32q + m
            data1=index.ap(pattern=[[n_pairs, _P_MAX], [2 * _GROUP, n_groups // 2], [0, 2], [1, 2 * _GROUP]], offset=0),
            data2=dx_mask_dw.ap(
                pattern=[[4 * _GROUP, _P_MAX], [0, n_groups // 2], [2 * _GROUP, 2], [1, 2 * _GROUP]], offset=0
            ),
            op=nl.multiply,
            engine=dve,
            name="dx_prod_down_w",
        )
        nisa.tensor_reduce(
            dst=experts[:, n_gathered : n_gathered + n_groups],  # column 2q + b
            op=nl.add,
            data=dx_prod.ap(pattern=[[dx_width, _P_MAX], [2 * _GROUP, n_groups], [1, 2 * _GROUP]], offset=0),
            axis=[2],
            name="dx_experts_down_w",
        )
        for g in quad_starts:
            nisa.tensor_tensor(
                dst=dx_prod[:, 0 : 16 * _GROUP],
                # [a, b, m]: the expert of pair 16g + m
                data1=index.ap(pattern=[[n_pairs, _P_MAX], [0, 4], [1, 4 * _GROUP]], offset=_GROUP * g),
                data2=dx_mask_q,
                op=nl.multiply,
                engine=dve,
                name=f"dx_prod_down_w_q{g}",
            )
            nisa.tensor_reduce(
                dst=experts[:, n_gathered + g : n_gathered + g + 4],  # column g + b of the quad
                op=nl.add,
                data=dx_prod.ap(pattern=[[dx_width, _P_MAX], [4 * _GROUP, 4], [1, 4 * _GROUP]], offset=0),
                axis=[2],
                name=f"dx_experts_down_w_q{g}",
            )
        nisa.tensor_tensor(
            dst=dx_prod[:, 0 : n_scale_tiles * n_pairs],
            # [a, b, n]: the expert of pair n
            data1=index.ap(pattern=[[n_pairs, _P_MAX], [0, n_scale_tiles], [1, n_pairs]], offset=0),
            data2=dx_mask_ds,
            op=nl.multiply,
            engine=dve,
            name="dx_prod_down_s",
        )
        nisa.tensor_reduce(
            dst=experts[:, n_gathered + n_groups : n_gathered + n_groups + n_scale_tiles],
            op=nl.add,
            data=dx_prod.ap(pattern=[[dx_width, _P_MAX], [n_pairs, n_scale_tiles], [1, n_pairs]], offset=0),
            axis=[2],
            name="dx_experts_down_s",
        )
    else:
        experts_gather = nisa.nc_n_gather(
            dst=experts, data=index, indices=select.view(nl.uint32), name="gather_experts"
        )

    def weight_gather_index(after_dve: Optional[nisa.NkiInstruction] = None) -> None:
        """Step 3.5: the gather indices of the gathered gate/up sub-groups and of the down weights. With after_dve
        (dyn_direct_g1), DVE first copies the expert ids after after_dve, and the index ops read the copy."""
        ids = experts
        if after_dve != None:
            ids_copy = nisa.tensor_copy(
                dst=dd_experts_late, src=experts[:, 0:n_weight_cols], engine=dve, name="dd_experts_late_op"
            )
            ids_copy.depends_on(after_dve, kind=nisa.dependency.schedule)
            ids = dd_experts_late
        nisa.scalar_tensor_tensor(
            dst=gather_index[:, gate_up_index_col + direct_first * sub_pairs : down_index_col],
            # [a, i, b]: expert of the i-th gathered sub-group
            data=ids.ap(
                pattern=[[ids.shape[1], _P_MAX], [1, n_gathered - direct_first], [0, sub_pairs]],
                offset=int(direct_first),
            ),
            op0=nl.multiply,
            operand0=_P_MAX,
            op1=nl.add,
            # [a, i, b]: row (128/M)b + a/M
            operand1=lanes.ap(
                pattern=[[lanes.shape[1], _P_MAX], [0, n_gathered - direct_first], [1, sub_pairs]], offset=0
            ),
            name="gather_index_gate_up_w",
        )
        nisa.scalar_tensor_tensor(
            dst=gather_index[:, down_index_col : down_index_col + n_groups],
            # [a, g, 0]: expert of group g
            data=ids.ap(pattern=[[ids.shape[1], _P_MAX], [1, n_groups], [0, 1]], offset=n_gathered),
            op0=nl.multiply,
            operand0=_MX_ROWS,
            op1=nl.add,
            # [a, g, 0]: row a%8 (+ SKIP)
            operand1=lanes.ap(pattern=[[lanes.shape[1], _P_MAX], [0, n_groups], [1, 1]], offset=16),
            name="gather_index_down_w",
        )
        for g in quad_starts:
            # merged_down_quads: the quad columns use the quad source row (lanes[:, 18])
            nisa.scalar_tensor_tensor(
                dst=gather_index[:, down_index_col + g : down_index_col + g + 4],
                # [a, b, 0]: expert of column g + b
                data=ids.ap(pattern=[[ids.shape[1], _P_MAX], [1, 4], [0, 1]], offset=n_gathered + g),
                op0=nl.multiply,
                operand0=_MX_ROWS,
                op1=nl.add,
                operand1=lanes.ap(pattern=[[lanes.shape[1], _P_MAX], [0, 4], [1, 1]], offset=18),
                name=f"gather_index_down_w_q{g}",
            )
        for g in pair_starts if mixed_down else []:
            # merged_down_pairs: the merged columns use the merged source row (lanes[:, 18])
            nisa.scalar_tensor_tensor(
                dst=gather_index[:, down_index_col + g : down_index_col + g + 2],
                # [a, b, 0]: expert of column g + b
                data=ids.ap(pattern=[[ids.shape[1], _P_MAX], [1, 2], [0, 1]], offset=n_gathered + g),
                op0=nl.multiply,
                operand0=_MX_ROWS,
                op1=nl.add,
                operand1=lanes.ap(pattern=[[lanes.shape[1], _P_MAX], [0, 2], [1, 1]], offset=18),
                name=f"gather_index_down_w{g}",
            )

    if not direct_g1:
        weight_gather_index()  # dyn_direct_g1: after the group-0 re-layout (Step 7.1)
    if last_group_halves:
        """last_group_halves: index [a, b] of the gather of half h feeds partition 16b + a / 8, slot a % 8: the expert of
        pair 16(N/16 - 1) + 8h + a % 8 (a masked row sum over the 8 pairs of the half) and source row 16b + a / 8."""
        iota(dst=lh_cols, pattern=[[1, half_pairs]], offset=0, channel_multiplier=0, name="lh_iota_cols")
        nisa.tensor_tensor(
            dst=lh_mask,
            data1=lh_cols,
            data2=partition.ap(pattern=[[7, _P_MAX], [0, half_pairs]], offset=1),  # [a, k]: a % 8
            op=nl.equal,
            engine=dve,
            name="lh_mask_op",
        )
        iota(dst=lh_lanes, pattern=[[16, half_pairs]], offset=0, channel_multiplier=0, name="lh_iota_lanes")  # 16b
        nisa.tensor_tensor(
            dst=lh_lanes,
            data1=lh_lanes,
            data2=partition.ap(pattern=[[7, _P_MAX], [0, half_pairs]], offset=3),  # [a, b]: a / 8
            op=nl.add,
            engine=dve,
            name="lh_lanes_op",
        )
        for half in range(2):
            first_pair = (n_groups - 1) * _GROUP + half * half_pairs
            nisa.scalar_tensor_tensor(
                dst=lh_prod,
                data=index[:, first_pair : first_pair + half_pairs],  # [a, k]: the expert of pair first_pair + k
                op0=nl.multiply,
                operand0=1.0,
                op1=nl.multiply,
                operand1=lh_mask,
                reduce_res=lh_expert[:, half : half + 1],
                name=f"lh_expert_h{half}",
            )
            nisa.scalar_tensor_tensor(
                dst=lh_index[:, half * half_pairs : (half + 1) * half_pairs],
                data=lh_expert.ap(pattern=[[2, _P_MAX], [0, half_pairs]], offset=half),  # [a, b]: the expert
                op0=nl.multiply,
                operand0=_P_MAX,
                op1=nl.add,
                operand1=lh_lanes,
                name=f"lh_gather_index_h{half}",
            )

    def scale_gather_index(after_dve: Optional[nisa.NkiInstruction] = None) -> nisa.NkiInstruction:
        """The down-scale gather indices (last; dynamic dedupe: after the slot map and the cidx0 expert ids). With
        dyn_direct_g1, Pool gathers the scale expert ids after the group-1 gather, and the DVE op follows after_dve."""
        if dynamic:
            if not direct_g1:
                compaction_rest()
            scale_experts = nisa.nc_n_gather(
                dst=experts[:, n_weight_cols:n_select],
                data=cidx0,
                indices=select[:, n_weight_cols:n_select].view(nl.uint32),
                name="dd_gather_scale_experts",
            )
            if direct_g1:
                scale_experts.depends_on(first_gather, kind=nisa.dependency.schedule)
        scale_index = nisa.tensor_tensor(
            dst=gather_index[:, down_scale_index_col : down_scale_index_col + n_scale_tiles],
            data1=experts[:, n_gathered + n_groups : n_gathered + n_groups + n_scale_tiles],  # experts of scale pairs
            data2=lanes.ap(pattern=[[lanes.shape[1], _P_MAX], [0, n_scale_tiles]], offset=17),  # [a, s]: 0 (+ SKIP)
            op=nl.add,
            engine=nisa.vector_engine,
            name="gather_index_down_s",
        )
        if after_dve != None:
            scale_index.depends_on(after_dve, kind=nisa.dependency.schedule)
        return scale_index

    # dynamic dedupe: Pool issues the first gate/up gather before the scale expert ids (Step 4)
    gather_index_done = None if dynamic else scale_gather_index()

    if static_dedupe:
        """Step 3.6: earlier_count[n] = the number of pairs before pair n (n % N, on partition n) with its expert. Each
        dedupe gather row gets + _SKIP per earlier pair of the pair that feeds it, so a repeated expert is out of
        bounds: column_counts[n, i] = earlier_count[n] if pair n belongs to the group of dedupe column i, and a matmul
        with pair_select moves the count of the pair that feeds partition a to [a, i] (one nonzero term: exact).
        gather_index_dedupe = gather_index + _SKIP * that count. The HWDGE group reads index_dedupe[0, n] = index[0, n]
        + _SKIP_EXPERT * count of pair n, from a ones matmul that puts the counts on every partition."""
        nisa.scalar_tensor_tensor(
            dst=same_earlier,  # [n, n']: pair n' has the expert of pair n and comes before it
            data=index,
            op0=nl.equal,
            operand0=experts[:, pair_col : pair_col + 1],  # the expert of pair n
            op1=nl.multiply,
            operand1=earlier_mask,
            name="dedupe_same_earlier_op",
        )
        nisa.tensor_reduce(dst=earlier_count, op=nl.add, data=same_earlier, axis=1, name="dedupe_count")
        dedupe_counts = nisa.tensor_scalar(
            dst=column_counts,
            data=column_groups,
            op0=nl.multiply,
            operand0=earlier_count[:, 0:1],
            engine=dve,
            name="dedupe_column_counts_op",
        )
        nisa.nc_matmul(
            dst=column_counts_ps[:, 0:n_dedupe_gate_up],  # [a, i]: count of pair 16 g_i + a % 16
            stationary=pair_select[:, 0:_P_MAX],
            moving=column_counts[:, 0:n_dedupe_gate_up],
            name="mm_dedupe_gate_up_counts",
        )
        nisa.nc_matmul(
            dst=column_counts_ps[:, n_dedupe_gate_up:],  # [a, g]: count of pair 16 (1 + g) + a / 8
            stationary=pair_select[:, _P_MAX:],
            moving=column_counts[:, n_dedupe_gate_up:],
            name="mm_dedupe_down_counts",
        )
        nisa.scalar_tensor_tensor(
            dst=gather_index_dedupe[:, 0 : n_dedupe_gate_up * sub_pairs],
            # [a, i, b]: count of the pair of gathered position 1 + i
            data=column_counts_ps.ap(
                pattern=[[n_dedupe_cols, _P_MAX], [1, n_dedupe_gate_up], [0, sub_pairs]], offset=0
            ),
            op0=nl.multiply,
            operand0=float(_SKIP),
            op1=nl.add,
            operand1=gather_index[:, gate_up_index_col + sub_pairs : down_index_col],
            name="gather_index_gate_up_w_dedupe",
        )
        nisa.scalar_tensor_tensor(
            dst=gather_index_dedupe[:, n_dedupe_gate_up * sub_pairs :],
            data=column_counts_ps[:, n_dedupe_gate_up:],  # [a, g]: count of the down pair of group 1 + g
            op0=nl.multiply,
            operand0=float(_SKIP),
            op1=nl.add,
            operand1=gather_index[:, down_index_col + 1 : down_index_col + n_groups],
            name="gather_index_down_w_dedupe",
        )
        for slot in dve_zero_slots:
            # After the dedupe chain in DVE order (a memset has no inputs, so the scheduler would hoist it).
            zero_scale_segments(slot, dve).depends_on(dedupe_counts, kind=nisa.dependency.schedule)
        if hwdge_group != None:
            nisa.tensor_scalar(
                dst=count_diag,  # [n, n']: the count of pair n at n' = n
                data=pair_identity,
                op0=nl.multiply,
                operand0=earlier_count[:, 0:1],
                engine=dve,
                name="dedupe_count_diag_op",
            )
            nisa.nc_matmul(dst=dedupe_ps, stationary=ones, moving=count_diag, name="mm_dedupe_bcast")
            nisa.scalar_tensor_tensor(
                dst=index_dedupe,
                data=dedupe_ps,  # [a, n]: the count of pair n
                op0=nl.multiply,
                operand0=float(_SKIP_EXPERT),
                op1=nl.add,
                operand1=index,
                name="dedupe_index",
            )
            load_hwdge_group()

    # Step 4: Start the Pool weight stream, in PE consumption order.
    gate_up_rows_hbm = expert_gate_up_packed.reshape((E * _P_MAX, row_bytes))  # row 128e + p: expert e, row p
    down_w_rows_hbm = expert_down_weights.reshape((E * _MX_ROWS, H))  # row 8e + p: expert e, row p
    # row e: expert e (two scale groups: both groups of expert e, [s][H])
    down_s_rows_hbm = expert_down_weights_scale.reshape((E, n_scale_sets * H))
    """Stream order: the gate/up rows of group g (one gather per sub-group j = gU + u > 0), then the down weights of
    group g - 1. A gather into a reused row slot comes after (in program order) the re-layout that reads the previous
    group of the slot, so its write is ordered after those reads."""
    # (name, destination, HBM rows, (index tile, first column), index columns, earliest issue step)
    stream_plan = []
    for g in range(n_groups + 1):
        for u in range(n_sub_groups if g < n_groups else 0):
            j = g * n_sub_groups + u
            if j not in gathered:
                continue  # sub-group 0 and hwdge_group come from the HWDGE loads (Step 2.3)
            slot = ((g % n_gate_up_slots) * _GROUP + u * sub_pairs) * row_bytes  # bytes: rows of the sub-group
            if last_group_halves and g == n_groups - 1:
                # last_group_halves: one gather of 8 experts per half (index tile lh_index)
                for half in range(2):
                    first = slot + half * half_pairs * row_bytes
                    stream_plan.append(
                        (
                            f"gather_gate_up_g{g}_h{half}",
                            gate_up_w.view(nl.uint8)[:, first : first + half_pairs * row_bytes],
                            gate_up_rows_hbm,
                            (lh_index, half * half_pairs),
                            half_pairs,
                            max(-1, g - n_gate_up_slots),
                        )
                    )
                continue
            stream_plan.append(
                (
                    f"gather_gate_up_g{g}" + (f"_u{u}" if n_sub_groups > 1 else ""),
                    gate_up_w.view(nl.uint8)[:, slot : slot + sub_pairs * row_bytes],
                    gate_up_rows_hbm,
                    # dedupe: gathered positions >= 1 read the deduped indices
                    (gather_index_dedupe, (gathered.index(j) - 1) * sub_pairs)
                    if static_dedupe and gathered.index(j) > 0
                    else (eg_index, 0)  # early_gather: the first gathered sub-group has its own index tile
                    if early_gather and gathered.index(j) == 0
                    else (gather_index, gate_up_index_col + gathered.index(j) * sub_pairs),
                    sub_pairs,
                    max(-1, g - n_gate_up_slots),  # the re-layout step that frees the slot
                )
            )
        if g == 0:
            continue
        if (g - 1) % _SCALE_GROUPS == 0 and not (merged_scales and g > 1):
            tile = (g - 1) // _SCALE_GROUPS  # down-scale tile of groups 4s .. 4s + 3, before their down weights
            n_tiles = n_scale_tiles if merged_scales else 1  # merged_scales: both tiles in the first gather
            stream_plan.append(
                (
                    "gather_down_s" + (f"{tile}" if tile > 0 else ""),
                    down_s[:, tile * n_scale_sets * H : (tile + n_tiles) * n_scale_sets * H],
                    down_s_rows_hbm,
                    (gather_index, down_scale_index_col + tile),
                    n_tiles,
                    -1,
                )
            )
        if merged_down_gathers and g - 1 in merged_with:
            if merged_with[g - 1] == g - 1:
                # groups g - 1 and g (merged_down_quads: g - 1 .. g + 2) in one gather (index columns g - 1 ..), where
                # the gather of group g - 1 was
                width = 4 if g - 1 in quad_starts else 2
                stream_plan.append(
                    (
                        "gather_down_w_g" + "".join(str(g - 1 + b) for b in range(width)),
                        down_w[:, (g - 1) * H : (g - 1 + width) * H],
                        down_w_rows_hbm,
                        (gather_index, down_index_col + g - 1),
                        width,
                        -1,
                    )
                )
            continue
        stream_plan.append(
            (
                f"gather_down_w_g{g - 1}",
                down_w[:, (g - 1) * H : g * H],
                down_w_rows_hbm,
                # dedupe: groups >= 1 read the deduped indices
                (gather_index_dedupe, n_dedupe_gate_up * sub_pairs + g - 2)
                if static_dedupe and g - 1 > 0
                else (gather_index, down_index_col + g - 1),
                1,
                -1,
            )
        )
    if late_down > 0:
        """late_down: the down gathers of group d (scales and weights) go after the gate/up gather of group d + 2
        (default: d + 1), in the PE order of Step 7.2."""

        def stream_key(name: str) -> Tuple[int, int]:
            if last_group_halves and name == f"gather_gate_up_g{n_groups - 1}_h0":
                return n_groups - 2, 0  # the first half of the last group: before the down gather of late_down - 1
            if name.startswith("gather_gate_up_g"):
                return int(name[len("gather_gate_up_g")]), 0
            if name.startswith("gather_down_s"):
                tile = name[len("gather_down_s") :]
                return _SCALE_GROUPS * (int(tile) if tile else 0) + 2, 1
            down_group = int(name[len("gather_down_w_g")])  # the first group of the gather
            if last_gate_up_first and down_group in (late_down - 1, late_down):
                return n_groups - 1, 1  # after the last gate/up gather, before the down weights of the last groups
            return down_group + 2, 1

        stream_plan = sorted(stream_plan, key=lambda entry: stream_key(entry[0]))
    if dynamic:
        # The gate/up gathers of the tail groups go after the down gather of the head's last group.
        tail_names = [f"gather_gate_up_g{g}" for g in range(dyn_head_groups, n_groups)]
        tail_entries = [entry for entry in stream_plan if entry[0] in tail_names]
        stream_plan = [entry for entry in stream_plan if entry[0] not in tail_names]
        last = dyn_head_groups - 1  # the down-weight gather of the head's last group
        last_pair = merged_with.get(last)
        last_down = f"gather_down_w_g{last_pair}{last_pair + 1}" if last_pair is not None else f"gather_down_w_g{last}"
        after = [entry[0] for entry in stream_plan].index(last_down) + 1
        stream_plan = stream_plan[:after] + tail_entries + stream_plan[after:]
    # Issue step of each gather (-1: before the compute loop, s: after relayout(s)), not before the gather ahead of it.
    issue_steps, latest = [], -1
    for *_, earliest in stream_plan:
        latest = max(latest, earliest)
        issue_steps.append(latest)
    stream = [None] * len(stream_plan)

    def issue_stream(step: int, positions: Optional[Sequence[int]] = None) -> None:
        """Issue the stream gathers planned right after the re-layout of group `step` (-1: before the compute loop),
        only those at `positions` if given."""
        for position, (name, dst, rows, (index_tile, first_col), n_cols, _) in enumerate(stream_plan):
            if issue_steps[position] == step and (positions == None or position in positions):
                stream[position] = nisa.dma_copy(
                    dst=dst,
                    # [128 * n_cols rows, row]: row 128b + a is HBM row gather_index[a, first_col + b]
                    src=rows.ap(
                        pattern=[[rows.shape[1], _P_MAX * n_cols], [1, rows.shape[1]]],
                        offset=0,
                        vector_offset=index_tile[:, first_col : first_col + n_cols],
                        indirect_dim=0,
                    ),
                    dge_mode=nisa.dge_mode.swdge,
                    oob_mode=nisa.oob_mode.skip,
                    priority=1 if dynamic and dyn_tail_loads == "sp_prio" else None,
                    name=name,
                )

    def issue_rest(after_dve: Optional[nisa.NkiInstruction] = None) -> nisa.NkiInstruction:
        """Dynamic dedupe: the scale gather indices, then the stream after the first gather. Returns the last
        gather-index op."""
        done = scale_gather_index(after_dve=after_dve)
        issue_stream(-1, positions=range(1, len(stream_plan)))
        return done

    if dynamic:
        if first_gather != None:
            stream[0] = first_gather  # issued in Step 3.5 (Stage A or dyn_direct_g1)
        else:
            issue_stream(-1, positions=[0])  # the first gate/up gather, then the scale expert ids, then the rest
        if direct_g1:
            compaction_rest()  # the rest of Step 4 follows the group-0 re-layout (Step 7.1)
        else:
            gather_index_done = issue_rest()
    else:
        issue_stream(-1)
        if early_gather and experts_gather != None:
            stream[0].depends_on(experts_gather, kind=nisa.dependency.schedule)  # Pool: the expert gather first
    if finish_affinities != None and not direct_g1:
        finish_affinities(gather_index_done)

    # Step 5: Build the per-pair affinities while the PE is idle.
    if topk_affinities:
        # Step 5.1 (affinity_layout="topk"): affinity[a, n] = bf16(affinity_row[n]) on every partition: a ones matmul
        # with one nonzero term per output, so the broadcast is exact.
        nisa.tensor_copy(dst=affinity_row_bf16, src=affinity_row, engine=dve, name="cast_affinity_row")
        nisa.nc_matmul(dst=affinity_ps, stationary=ones[0:1, :], moving=affinity_row_bf16, name="mm_affinity_bcast")
        nisa.tensor_copy(dst=affinity, src=affinity_ps, engine=dve, name="copy_affinity")
    elif not inputs_in_sbuf:
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

    def row_weight_math(after: Optional[nisa.NkiInstruction] = None) -> nisa.NkiInstruction:
        """Step 5.2 (dynamic, emitted in Step 7.1 so that it follows the re-layout of groups 0-1 on DVE):
        row_weights[m, (g, t)] = the affinity of token t for the expert of unique slot
        16g + m/8 (0 if token t did not choose it) = sum over pairs n of [map[n] % 16 == m/8] (stationary) times
        [map[n] / 16 == g][n / K == t] affinity[n] (moving). One nonzero term per output (a token has distinct
        experts), so the bf16 values are exact."""
        # map = the slot map for compacted pairs, n for the verbatim pairs (they keep their own slot)
        nisa.scalar_tensor_tensor(
            dst=dd_map,
            data=dd_map_ps,
            op0=nl.multiply,
            operand0=dd_verbatim[:, 1:2],
            op1=nl.add,
            operand1=dd_verbatim[:, 2:3],
            name="dd_map_op",
        )
        map_int = nisa.tensor_copy(dst=dd_slot_int[:, 0:1], src=dd_map, engine=dve, name="dd_map_int")
        if after != None:
            map_int.depends_on(after, kind=nisa.dependency.schedule)
        nisa.tensor_scalar(
            dst=dd_slot_int[:, 1:2],
            data=dd_slot_int[:, 0:1],
            op0=nl.bitwise_and,
            operand0=_GROUP - 1,
            engine=dve,
            name="dd_map_lo",
        )  # map % 16
        nisa.tensor_scalar(
            dst=dd_slot_int[:, 2:3],
            data=dd_slot_int[:, 0:1],
            op0=nl.right_shift,
            operand0=4,
            engine=dve,
            name="dd_map_hi",
        )  # map / 16
        # per-partition scalars must be fp32: (map, map % 16, map / 16, n / 16)
        nisa.tensor_copy(dst=dd_slot_f, src=dd_slot_int, engine=dve, name="dd_slot_f_op")
        # affinity of pair n on partition n (the diagonal of the broadcast affinity)
        nisa.tensor_tensor(
            dst=dd_prod, data1=affinity, data2=dd_ident, op=nl.multiply, engine=dve, name="dd_diag_affinity"
        )
        nisa.tensor_reduce(dst=dd_affinity, op=nl.add, data=dd_prod, axis=1, name="dd_affinity_op")
        nisa.tensor_scalar(
            dst=dd_slot_rows,
            data=dd_iota_row,
            op0=nl.equal,
            operand0=dd_slot_f[:, 1:2],
            engine=dve,
            name="dd_slot_rows_op",
        )
        nisa.tensor_scalar(
            dst=dd_token_mask,
            data=dd_token_mask.view(nl.int32),
            op0=nl.equal,
            operand0=dd_slot_f[:, 3:4],
            engine=dve,
            name="dd_token_mask_op",
        )  # t == n / 16 (K = 16)
        nisa.tensor_scalar(
            dst=dd_token_affinity,
            data=dd_token_mask,
            op0=nl.multiply,
            operand0=dd_affinity,
            engine=dve,
            name="dd_token_affinity_op",
        )
        nisa.scalar_tensor_tensor(
            dst=dd_group_affinity,
            data=dd_iota_group,
            op0=nl.equal,
            operand0=dd_slot_f[:, 2:3],
            op1=nl.multiply,
            operand1=dd_token_affinity,
            name="dd_group_affinity_op",
        )
        nisa.nc_matmul(dst=dd_rw_ps, stationary=dd_slot_rows, moving=dd_group_affinity, name="dd_rw_mm")
        return nisa.tensor_copy(dst=row_weights, src=dd_rw_ps, engine=dve, name="dd_row_weights")

    if static_dedupe:
        """Step 5.2 (dedupe): the first pair of each expert serves every token that routes to it (gate/up computes all
        tokens). On partition j (pair j), slot_weights[j, t] = the sum over k of the affinity of pair t*K + k if that
        pair has the deduped expert of pair j (index_dedupe[j, j] = slot_expert[j]). A repeated pair j has an out-of-
        range deduped id, so it matches no index entry and its weights are zero. Then one matmul per group moves the
        weights to the rows of the down stationary: row_weights[m, (g, t)] = slot_weights[16g + m/8, t], with the 0/1
        stationary row_select[j, (g, m)]. Exact in bf16 for top-K routing (one nonzero term)."""
        nisa.scalar_tensor_tensor(
            dst=slot_expert,  # expert of pair j + _SKIP_EXPERT * its earlier count
            data=earlier_count,
            op0=nl.multiply,
            operand0=float(_SKIP_EXPERT),
            op1=nl.add,
            operand1=experts[:, pair_col : pair_col + 1],
            name="dedupe_slot_expert_op",
        )
        nisa.scalar_tensor_tensor(
            dst=same_affinity,  # [j, n]: the affinity of pair n if it has the deduped expert of pair j
            data=index,
            op0=nl.equal,
            operand0=slot_expert,
            op1=nl.multiply,
            operand1=affinity,
            name="dedupe_same_affinity_op",
        )
        nisa.tensor_reduce(
            dst=slot_weights,
            op=nl.add,
            data=same_affinity.ap(pattern=[[n_pairs, _P_MAX], [K, T], [1, K]], offset=0),  # [j, t, k]: pair t*K + k
            axis=[2],
            name="dedupe_slot_weights_op",
        )
        for group in range(n_groups):
            nisa.nc_matmul(
                dst=row_weights_ps[:, group * T : (group + 1) * T],  # [m, t] of group g
                stationary=dedupe_row_select[:, group * _P_MAX : (group + 1) * _P_MAX],  # [j, m]: 1 if j == 16g + m/8
                moving=slot_weights,
                name=f"mm_dedupe_row_weights_g{group}",
            )
        nisa.tensor_copy(dst=row_weights, src=row_weights_ps, engine=dve, name="dedupe_row_weights")
    elif not dynamic and not post_down:
        # Step 5.2: affinity_mask[m, n] = affinity of pair n on its own rows m = 8(n % 16) + p.
        nisa.tensor_tensor(
            dst=affinity_mask,
            data1=affinity,
            data2=masks.ap(
                pattern=[[_GROUP, _P_MAX], [0, n_groups], [1, _GROUP]], offset=0
            ),  # [m, (g, c)]: masks[m, c]
            op=nl.multiply,
            engine=dve,
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
        if n_scale_sets == 1:
            nisa.tensor_tensor(
                dst=row_weights,
                data1=token_rows,
                data2=affinity_rows.ap(
                    pattern=[[n_groups, _P_MAX], [1, n_groups], [0, T]], offset=0
                ),  # [m, (g, t)]: [m, g]
                op=nl.multiply,
                engine=dve,
                name="build_row_weights",
            )
        else:
            # Two scale groups: row_weights[m, (s, g, t)] keeps row m only in the stationary of its scale group s
            # (token_rows times down_group_mask[m, s], then times the affinity; all values exact in bf16).
            group_cols_t = n_groups * T
            nisa.tensor_tensor(
                dst=scale_set_rows,
                data1=token_rows.ap(pattern=[[group_cols_t, _P_MAX], [0, 2], [1, group_cols_t]], offset=0),
                data2=down_group_mask.ap(pattern=[[2, _P_MAX], [1, 2], [0, group_cols_t]], offset=0),
                op=nl.multiply,
                engine=dve,
                name="build_scale_set_rows",
            )
            nisa.tensor_tensor(
                dst=row_weights,
                data1=scale_set_rows,
                # [m, (s, g, t)]: [m, g]
                data2=affinity_rows.ap(pattern=[[n_groups, _P_MAX], [0, 2], [1, n_groups], [0, T]], offset=0),
                op=nl.multiply,
                engine=dve,
                name="build_row_weights",
            )
    elif post_down:
        # Step 5.2 (post_down): aff_rows[r] = the affinity of the pair of output row r, the eviction scale.
        nisa.tensor_tensor(
            dst=row_affinity, data1=affinity, data2=row_select, op=nl.multiply, engine=dve, name="row_affinity_op"
        )
        nisa.tensor_reduce(dst=aff_rows, op=nl.add, data=row_affinity, axis=1, name="build_aff_rows")

    if sbuf_reserve != None:
        kernel_assert(
            sbm.heap_curr_addr - sbm.get_stack_curr_addr() >= sbuf_reserve,
            f"sbuf_reserve={sbuf_reserve}: {sbm.heap_curr_addr - sbm.get_stack_curr_addr()} B free after Step 5",
        )
    # The first phase ends: everything above needs only the routing.
    yield "routing"

    # Step 6: Define the per-group stages (Step 7 calls them).
    relayout_copies = {dve: [], act: []}
    in_loop = [False]  # ops inside the dynamic loop body get no name= (the body is emitted more than once)
    psum = {"gate_up": gate_up_ps_by_parity, "gated": gated_ps, "out": out_ps}  # the tails have their own tiles
    evict = {"add": False}  # the dynamic tails add their output to the head's out_sb

    def nm(name: str) -> Optional[str]:
        """name, or None inside the dynamic loop body."""
        return None if in_loop[0] else name

    def in_order(copy: nisa.NkiInstruction, engine: nisa.engine) -> None:
        """Schedule copy after the previous re-layout copy of engine (for ACT, after the last group-0 load).

        Without this order, the compiler runs the copies of the gathered groups before those of group 0 (HWDGE). ACT
        also issues half of the group-0 loads, so its copies follow them.
        """
        previous = None
        if relayout_copies[engine]:
            previous = relayout_copies[engine][-1]
        elif engine == act and hw_loads:
            previous = hw_loads[-1]
        if previous != None:
            copy.depends_on(previous, kind=nisa.dependency.schedule)
        relayout_copies[engine].append(copy)

    def relayout_scales(
        group: int, part: int, part_pairs: int, rows_base: int, after_act: Optional[nisa.NkiInstruction]
    ) -> None:
        """Step 6.1: ACT copies the scale segments of the pairs of `part` of group to its moving scale slot (the first
        copy after after_act)."""
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
                        [gate_up_w.shape[1], _P_MAX],
                        [I // 2, _BLOCKS],
                        [row_bytes // 2, part_pairs],
                        [1, I // 2],
                    ],
                    offset=rows_base + part * part_pairs * row_bytes // 2 + gate_up_cols + segment * _BLOCKS * I // 2,
                ),
                engine=act,
                name=nm(f"relayout_scales_g{group}_s{segment}_part{part}"),
            )
            if after_act != None:
                copy.depends_on(after_act, kind=nisa.dependency.schedule)
                after_act = None
            in_order(copy, act)

    def relayout(
        group: int,
        n_parts: int,
        runs: Optional[Sequence[int]] = None,
        after_dve: Optional[nisa.NkiInstruction] = None,
        after_act: Optional[nisa.NkiInstruction] = None,
        all_act: bool = False,
    ) -> None:
        """Step 6.1: copy the pair runs `runs` (default: all n_parts runs) of the packed rows of group to its moving
        slot. After the last run, issue the gathers that reuse its row slot.

        DVE copies the weights (twice the ACT copy rate) and ACT copies the scale segments, run by run, so a run
        starts as soon as its experts land (group 0 arrives one expert at a time, a sub-group with its gather). With
        U = 2, group 0 copies each run in two halves of the H512 blocks, so its first matmuls start after a quarter of
        the copies. Long stream: for group > 0 ACT also copies the first 4 H512 blocks of the weights, because Pool
        descriptor windows block DVE and not ACT. The first DVE / ACT copy is scheduled after after_dve / after_act.
        all_act (dyn_late_relayout): ACT copies all blocks. dyn_g1_dve_first: for group 1, ACT copies the scale
        segments first and DVE the first 4 blocks.
        """
        runs = range(n_parts) if runs == None else runs
        rows_base = (group % n_gate_up_slots) * _GROUP * row_bytes // 2  # row slot of the group, in 2-byte units
        moving_base = (group % n_moving_slots) * _GROUP * gate_up_cols  # moving slot of the group
        part_pairs = _GROUP // n_parts
        n_block_parts = 2 if group == 0 and n_sub_groups > 1 else 1
        part_blocks = n_h512 // n_block_parts
        dve_first = dynamic and dyn_g1_dve_first and long_stream and group == 1 and not all_act
        for part in runs:
            if dve_first:
                relayout_scales(group, part, part_pairs, rows_base, after_act)
                after_act = None
            for block_part in range(n_block_parts):
                last_dve = long_stream and group == n_groups - 1 and last_relayout == "dve"
                copies = []  # (gate_or_up, first H512 block, blocks, engine, name tag) of each copy, in DVE / ACT order
                if last_dve and last_relayout_runs > 1:
                    # last_relayout_runs: the gate and the up blocks of each run, run by run, so the first matmuls of
                    # the last group need only the first run.
                    edges = [n_h512 * run // last_relayout_runs for run in range(last_relayout_runs + 1)]
                    copies = [
                        (gate_or_up, edges[run], edges[run + 1] - edges[run], dve, f"_r{run}")
                        for run in range(last_relayout_runs)
                        for gate_or_up in range(2)
                    ]
                for gate_or_up in range(2) if not copies else []:
                    # (first H512 block, blocks, engine) of each copy
                    if all_act:
                        block_runs = [(0, n_h512, act)]
                    elif dve_first:
                        block_runs = [
                            (0, _LONG_ACT_RELAYOUT_BLOCKS, dve),
                            (_LONG_ACT_RELAYOUT_BLOCKS, n_h512 - _LONG_ACT_RELAYOUT_BLOCKS, act),
                        ]
                    elif last_dve:
                        # The last group is re-laid after the Pool stream: DVE copies do not stall then.
                        block_runs = [(0, n_h512, dve)]
                    elif long_stream and (group > 0 or relayout0_act):
                        # ACT copies the first act_blocks blocks (all of them: no DVE copy), DVE the others
                        block_runs = [(0, act_blocks, act)] + (
                            [(act_blocks, n_h512 - act_blocks, dve)] if act_blocks < n_h512 else []
                        )
                    else:
                        block_runs = [(block_part * part_blocks, part_blocks, dve)]
                    copies += [
                        (gate_or_up, first_block, run_blocks, engine, f"_s{run_index}" if len(block_runs) > 1 else "")
                        for run_index, (first_block, run_blocks, engine) in enumerate(block_runs)
                    ]
                for gate_or_up, first_block, run_blocks, engine, tag in copies:
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
                                [gate_up_w.shape[1], _P_MAX],
                                [I, run_blocks],
                                [row_bytes // 2, part_pairs],
                                [1, I],
                            ],
                            offset=rows_base
                            + part * part_pairs * row_bytes // 2
                            + (gate_or_up * n_h512 + first_block) * I,
                        ),
                        engine=engine,
                        name=nm(
                            f"relayout_{('gate', 'up')[gate_or_up]}_g{group}_part{part}"
                            + (f"_h{block_part}" if n_block_parts > 1 else "")
                            + tag
                        ),
                    )
                    if engine == dve and after_dve != None:
                        copy.depends_on(after_dve, kind=nisa.dependency.schedule)
                        after_dve = None
                    if engine == act and after_act != None:
                        copy.depends_on(after_act, kind=nisa.dependency.schedule)
                        after_act = None
                    in_order(copy, engine)
            if not dve_first:
                relayout_scales(group, part, part_pairs, rows_base, after_act)
                after_act = None
        if n_parts - 1 in runs:
            issue_stream(group)

    def gate_up_projection(
        group: int,
        units: Sequence[Tuple[int, int]],
        after: Optional[nisa.NkiInstruction],
        halves: Sequence[int] = (0, 1),
        pairs: Optional[Tuple[int, int]] = None,
    ) -> nisa.NkiInstruction:
        """Step 6.2: gate and up matmuls of group for the (sub-group, H512 block) `units`, chained after `after`.
        halves: the gate (0) and / or up (1) matmuls of each unit (last_gate_up_split: the up sums of the last group
        go to last_up_ps). pairs (last_group_halves): (first pair, pairs) of the group: only their columns (the sums
        of a second half go to lh_ps).

        Returns the last matmul.
        """
        gate_up_ps = psum["gate_up"][group % 2]
        cols = sub_cols  # moving columns of each matmul
        if pairs != None:
            cols = pairs[1] * I
            if pairs[0] > 0:
                gate_up_ps = lh_ps
        matmul = after
        for sub_group, block in units:
            sub_col = sub_group * sub_cols + (pairs[0] * I if pairs != None else 0)  # first (c, I) column
            for gate_or_up in halves:
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
                # [t, (c, I)] of the sub-group (last_gate_up_split: the up sums of the last group in last_up_ps)
                dst = gate_up_ps[:T, psum_col : psum_col + cols]
                if last_up_ps != None and gate_or_up == 1 and group == n_groups - 1:
                    dst = last_up_ps[:T, sub_col : sub_col + cols]
                previous = matmul
                matmul = nisa.nc_matmul_mx(
                    dst=dst,
                    # [p, t]: H512 block `block` of all T tokens (the only reads of the hidden layout)
                    stationary=hidden_raw[:, block * T : (block + 1) * T],
                    moving=gate_up_moving.view(nl.float4_e2m1fn_x4)[:, weight_col : weight_col + cols],
                    stationary_scale=hidden_raw_s[:, block * T : (block + 1) * T],
                    moving_scale=gate_up_moving_s.view(nl.uint8)[
                        _X4 * (scale_chunk % _X4) :, scale_col : scale_col + cols
                    ],
                    accumulate=block > 0,
                    name=nm(
                        f"mm_{('gate', 'up')[gate_or_up]}_g{group}_b{block}"
                        + (f"_u{sub_group}" if n_sub_groups > 1 else "")
                        + (f"_p{pairs[0]}" if pairs != None else "")
                    ),
                )
                if previous != None:
                    matmul.depends_on(previous, kind=nisa.dependency.schedule)
        return matmul

    def swiglu(group: int) -> Tuple[nisa.NkiInstruction, nisa.NkiInstruction]:
        """Step 6.3: gated[t, (q, c, p)] = SiLU(gate) * up of i = q*(I/4) + p, in bf16 (ACT SiLU, DVE multiply).

        Returns the SiLU and the multiply (with situ: the last ACT and the last DVE instruction).
        """
        if situ != None:
            return situ_activation(group)
        gate_up_ps = psum["gate_up"][group % 2]
        gate_col = (group % 2) * group_cols  # gate_act slot of the group
        silu = nisa.activation(
            dst=gate_act[:, gate_col : gate_col + group_cols],
            op=nl.silu,
            data=gate_up_ps[:T, :group_cols],  # [t, (c, I)]: gate
            # 0 (keeps the PE keep-alive matmuls): the front chain for group 0, the tail chain for the last group
            bias=keep_alive_sink if group == 0 else (tail_sink if group == n_groups - 1 else None),
            name=nm(f"silu_g{group}"),
        )
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
            name=nm(f"mul_up_g{group}"),
        )
        return silu, multiply

    # situ: the first ACT and the first DVE instruction of each group's activation (ordered as the SiLU and the multiply)
    situ_first_act, situ_first_dve = {}, {}

    def situ_activation(
        group: int, pairs: Optional[Tuple[int, int]] = None
    ) -> Tuple[nisa.NkiInstruction, nisa.NkiInstruction]:
        """Step 6.3 with situ: gated[t, (q, c, p)] = lb*tanh(u/lb) * beta*tanh(g/beta) * sigmoid(g) in fp32, written
        as bf16 (ACT: the tanh and sigmoid terms into fp32 tiles; DVE: the products).

        The three ACT instructions run in program order (the first one waits for gate/up(group)), so the callers'
        ordering of SiLU applies to the first one (situ_first_act) and to the last one (returned), and their ordering
        of the multiply to the first DVE instruction (situ_first_dve) and the last one (returned). Returns the last
        ACT and the last DVE instruction. pairs (last_group_halves): (first pair, pairs) of the group: only their
        columns (a second half from lh_ps).
        """
        beta, linear_beta = situ
        gate_up_ps = psum["gate_up"][group % 2]
        col = (group % 2) * group_cols  # slot of the group in gate_act, situ_sigmoid and situ_up
        first_pair, n_cols, n_pair, label = 0, group_cols, _GROUP, ""  # the pairs of the activation
        if pairs != None:
            first_pair, n_pair = pairs
            n_cols, label = n_pair * I, f"_p{first_pair}"
            gate_up_ps = lh_ps if first_pair > 0 else gate_up_ps
        col_p = col + first_pair * I  # first column of the pairs
        gate_ps = gate_up_ps[:T, first_pair * I : first_pair * I + n_cols]  # [t, (c, I)]
        up_ps = gate_up_ps[:T, up_col + first_pair * I : up_col + first_pair * I + n_cols]
        split = last_up_ps != None and group == n_groups - 1  # last_gate_up_split: up in last_up_ps, gate terms first
        if split:
            up_ps = last_up_ps[:T, :group_cols]
        gate_term, up_term = gate_act[:, col_p : col_p + n_cols], situ_up[:, col_p : col_p + n_cols]
        sigmoid_term = situ_sigmoid[:, col_p : col_p + n_cols]
        tanh_gate = nisa.activation(
            dst=gate_term,
            op=nl.tanh,
            data=gate_ps,
            scale=1.0 / beta,
            # 0 (keeps the PE keep-alive matmuls): the front chain for group 0, the tail chain for the last group
            bias=keep_alive_sink if group == 0 else (tail_sink if group == n_groups - 1 else None),
            name=nm(f"tanh_gate_g{group}{label}"),
        )
        if first_pair == 0:
            situ_first_act[group] = tanh_gate
        # [t, q, c, p]: pair c, i = q*(I/4) + p, in a gate_act / situ_up slot, and in the gated slot of the group
        pairs = [[2 * group_cols, T], [down_rows, _X4], [I, n_pair], [1, down_rows]]
        gated_pairs = gated.ap(
            pattern=[[2 * _X4 * _P_MAX, T], [_P_MAX, _X4], [_MX_ROWS, n_pair], [1, down_rows]],
            offset=(group % 2) * _X4 * _P_MAX + _MX_ROWS * first_pair,
        )
        if split:
            """The gate sums are done before the up matmuls: ACT computes both gate terms (tanh(g / beta) and
            sigmoid(g)), then tanh(u / lb), so DVE's gate product runs while ACT waits for the up sums."""
            kernel_assert(not sigmoid_from_tanh, "last_gate_up_split needs sigmoid_from_tanh=False")
            sigmoid = nisa.activation(
                dst=sigmoid_term, op=nl.sigmoid, data=gate_ps, name=nm(f"sigmoid_gate_g{group}{label}")
            )
            sigmoid.depends_on(tanh_gate, kind=nisa.dependency.schedule)
            situ_first_dve[group] = nisa.NkiInstruction(  # scalar_tensor_tensor returns a raw instruction
                nisa.scalar_tensor_tensor(
                    dst=gate_term,
                    data=gate_term,
                    op0=nl.multiply,
                    operand0=beta * linear_beta,
                    op1=nl.multiply,
                    operand1=sigmoid_term,
                    name=nm(f"situ_gate_g{group}{label}"),
                )
            )
            last_act = nisa.activation(
                dst=up_term, op=nl.tanh, data=up_ps, scale=1.0 / linear_beta, name=nm(f"tanh_up_g{group}{label}")
            )
            last_act.depends_on(sigmoid, kind=nisa.dependency.schedule)
            multiply = nisa.tensor_tensor(
                dst=gated_pairs,
                data1=situ_up.ap(pattern=pairs, offset=col_p),
                data2=gate_act.ap(pattern=pairs, offset=col_p),
                op=nl.multiply,
                name=nm(f"mul_up_g{group}{label}"),
            )
            return last_act, multiply
        tanh_up = nisa.activation(
            dst=up_term, op=nl.tanh, data=up_ps, scale=1.0 / linear_beta, name=nm(f"tanh_up_g{group}{label}")
        )
        tanh_up.depends_on(tanh_gate, kind=nisa.dependency.schedule)
        if sigmoid_from_tanh:
            # 2 sigmoid(g) = 1 + tanh(g/2): gate_term = (1 + tanh(g/2)) * tanh(g/beta); beta/2 goes to the up term.
            last_act = nisa.activation(
                dst=sigmoid_term, op=nl.tanh, data=gate_ps, scale=0.5, name=nm(f"tanh_half_gate_g{group}{label}")
            )
            last_act.depends_on(tanh_up, kind=nisa.dependency.schedule)
            situ_first_dve[group] = nisa.NkiInstruction(  # scalar_tensor_tensor returns a raw instruction
                nisa.scalar_tensor_tensor(
                    dst=gate_term,
                    data=sigmoid_term,
                    op0=nl.add,
                    operand0=1.0,
                    op1=nl.multiply,
                    operand1=gate_term,
                    name=nm(f"situ_gate_g{group}{label}"),
                )
            )
            nisa.scalar_tensor_tensor(
                dst=up_term,
                data=up_term,
                op0=nl.multiply,
                operand0=linear_beta * beta / 2.0,
                op1=nl.multiply,
                operand1=gate_term,
                name=nm(f"situ_g{group}{label}"),
            )
            multiply = nisa.tensor_copy(
                dst=gated_pairs,
                src=situ_up.ap(pattern=pairs, offset=col_p),
                engine=dve,
                name=nm(f"mul_up_g{group}{label}"),
            )
            return last_act, multiply
        last_act = nisa.activation(
            dst=sigmoid_term, op=nl.sigmoid, data=gate_ps, name=nm(f"sigmoid_gate_g{group}{label}")
        )
        last_act.depends_on(tanh_up, kind=nisa.dependency.schedule)
        # gate_term = tanh(g/beta) * (beta*lb) * sigmoid(g), then gated = tanh(u/lb) * gate_term
        first_dve = nisa.NkiInstruction(  # scalar_tensor_tensor returns a raw instruction
            nisa.scalar_tensor_tensor(
                dst=gate_term,
                data=gate_term,
                op0=nl.multiply,
                operand0=beta * linear_beta,
                op1=nl.multiply,
                operand1=sigmoid_term,
                name=nm(f"situ_gate_g{group}{label}"),
            )
        )
        if first_pair == 0:
            situ_first_dve[group] = first_dve
        multiply = nisa.tensor_tensor(
            dst=gated_pairs,
            data1=situ_up.ap(pattern=pairs, offset=col_p),
            data2=gate_act.ap(pattern=pairs, offset=col_p),
            op=nl.multiply,
            name=nm(f"mul_up_g{group}{label}"),
        )
        return last_act, multiply

    def finish(group: int, after: nisa.NkiInstruction) -> Tuple[nisa.NkiInstruction, nisa.NkiInstruction]:
        """Step 6.4: the down-projection stationary of group (transposes chained after `after`).

        Returns the last transpose and the quantization.
        """
        if wide_out_ps:
            gated_tile, ps_col = psum["gate_up"][group % 2], _PSUM_COLS - _X4 * T  # gated columns of the tile
        else:
            gated_tile, ps_col = psum["gated"], (group % 2) * _X4 * T  # gated_ps slot of the group
        gated_row = gated_tile.shape[1]  # elements per partition of the tile
        transpose = after
        for q in range(_X4):
            previous = transpose
            transpose = nisa.nc_transpose(
                dst=gated_tile[:, ps_col + q * T : ps_col + (q + 1) * T],  # [(c, p), t] of lane q
                data=gated[:, ((group % 2) * _X4 + q) * _P_MAX : ((group % 2) * _X4 + q + 1) * _P_MAX],  # [t, (c, p)]
                name=nm(f"transpose_gated_g{group}_q{q}"),
            )
            if previous != None:
                transpose.depends_on(previous, kind=nisa.dependency.schedule)
        if post_down:
            """qmx_in[m, (k, t, q)] = SiLU(gate) * up of row m, lane q, in the slot (k, t) of its pair t*K + k, and zero
            in the other slots: gated_ps[m, (q, t)] times slot_mask[m, (group, k, t)] (the mask also picks the token of
            the row). The unscaled values are quantized into the pair-slot block of the group's stationary."""
            slot_first = group * n_pairs * _X4
            nisa.tensor_tensor(
                dst=qmx_in[:, slot_first : slot_first + n_pairs * _X4],
                data1=gated_tile.ap(pattern=[[gated_row, _P_MAX], [0, K], [1, T], [T, _X4]], offset=ps_col),  # m,k,t,q
                # [m, k, t, q]: slot_mask[m, (group, k, t)] for each q
                data2=slot_mask.ap(
                    pattern=[[n_groups * n_pairs, _P_MAX], [T, K], [1, T], [0, _X4]], offset=group * n_pairs
                ),
                op=nl.multiply,
                name=nm(f"qmx_in_g{group}"),
            )
            slot_col = (group + 1) * stationary_cols - n_pairs  # the pair-slot block of the group
            quantize = nisa.quantize_mx(
                dst=inter[:, slot_col : slot_col + n_pairs],
                src=qmx_in[:, slot_first : slot_first + n_pairs * _X4],
                dst_scale=inter_s[:, slot_col : slot_col + n_pairs],
                name=nm(f"qmx_g{group}"),
            )
            return transpose, quantize
        # Last token block of the group (of the last term) in qmx_in: [m, (t, q)] = v = row m of token t times its
        # row weight. Two scale groups: the block of each scale group s, with the row weights of s.
        slot = group % n_groups  # stationary slot (the dynamic loop groups reuse the slots of the static head)
        group_first = slot * group_stride  # first stationary column of the group
        lo_first = (group_first + stationary_cols - T) * _X4
        set_step = stationary_cols * _X4  # qmx_in elements from one scale group to the next

        def blocks(tile: nl.NkiTensor, first: int) -> nl.NkiTensor:
            # [m, (t, q)] from element `first` of tile (two scale groups: [m, (s, t, q)], one block per group)
            if n_scale_sets == 1:
                return tile[:, first : first + T * _X4]
            return tile.ap(pattern=[[tile.shape[1], _P_MAX], [set_step, 2], [_X4, T], [1, _X4]], offset=first)

        nisa.tensor_tensor(
            dst=blocks(qmx_in, lo_first),
            # [m, (t, q)] (two scale groups: [m, (s, t, q)], the same activation for both groups)
            data1=gated_tile.ap(pattern=[[gated_row, _P_MAX], [1, T], [T, _X4]], offset=ps_col)
            if n_scale_sets == 1
            else gated_tile.ap(pattern=[[gated_row, _P_MAX], [0, 2], [1, T], [T, _X4]], offset=ps_col),
            # [m, (t, q)]: row_weights[m, (group, t)] for each q (two scale groups: row_weights[m, (s, group, t)])
            data2=row_weights.ap(pattern=[[row_weights.shape[1], _P_MAX], [1, T], [0, _X4]], offset=group * T)
            if n_scale_sets == 1
            else row_weights.ap(
                pattern=[[row_weights.shape[1], _P_MAX], [n_groups * T, 2], [1, T], [0, _X4]], offset=group * T
            ),
            op=nl.multiply,
            name=nm(f"qmx_in_g{group}"),
        )
        if hi_lo:
            # hi = v with 4 significant bits (an AND of the bf16 bits, exact in e4m3) in the last token block of the
            # hi term, then lo = v - hi (exact in bf16) in place.
            hi_first = (group_first + token_cols - T) * _X4
            if hi_mask_tile:
                # tensor_tensor: no DVE fast mode, so a Pool descriptor window does not stall it
                nisa.tensor_tensor(
                    dst=blocks(qmx_in.view(nl.int16), hi_first),
                    data1=blocks(qmx_in.view(nl.int16), lo_first),
                    # [m, (t, q)]: the mask (two scale groups: [m, (s, t, q)])
                    data2=hi_mask.ap(pattern=[[1, _P_MAX], [0, T * _X4]], offset=0)
                    if n_scale_sets == 1
                    else hi_mask.ap(pattern=[[1, _P_MAX], [0, 2], [0, T], [0, _X4]], offset=0),
                    op=nl.bitwise_and,
                    engine=dve,
                    name=nm(f"qmx_hi_g{group}"),
                )
            else:
                nisa.tensor_scalar(
                    dst=blocks(qmx_in.view(nl.int16), hi_first),
                    data=blocks(qmx_in.view(nl.int16), lo_first),
                    op0=nl.bitwise_and,
                    operand0=_E4M3_BF16_MASK,
                    engine=dve,
                    name=nm(f"qmx_hi_g{group}"),
                )
            nisa.tensor_tensor(
                dst=blocks(qmx_in, lo_first),
                data1=blocks(qmx_in, lo_first),
                data2=blocks(qmx_in, hi_first),
                op=nl.subtract,
                engine=dve,
                name=nm(f"qmx_lo_g{group}"),
            )
        quantize = nisa.quantize_mx(
            dst=inter[:, group_first : group_first + group_stride],
            src=qmx_in[:, group_first * _X4 : (group_first + group_stride) * _X4],
            dst_scale=inter_s[:, group_first : group_first + group_stride],
            name=nm(f"qmx_g{group}"),
        )
        return transpose, quantize

    pending_sums = []  # post_down: evicted chunks of the last group whose token sum is not issued yet

    def sum_tokens(chunk: int, after: nisa.NkiInstruction) -> nisa.NkiInstruction:
        """post_down: add the K scaled pair rows of each token for one chunk on the PE (bf16 matmul with the 0/1
        token_sum stationary, chained after `after`), evict the [Hh * T, 512] sums and store them. Returns the
        matmul."""
        col = chunk * _DOWN_CHUNK
        sum_ps = sum_ps_ring[chunk % len(sum_ps_ring)]
        matmul = nisa.nc_matmul(
            dst=sum_ps,  # [h*T + t, 512]: sum over k of out_sb[h*T*K + 8k + t, chunk]
            stationary=token_sum[:stationary_cols],  # the Hh * T*K output rows
            moving=out_sb[:, col : col + _DOWN_CHUNK],
            accumulate=False,
            name=nm(f"mm_token_sum_c{chunk}"),
        )
        matmul.depends_on(after, kind=nisa.dependency.schedule)
        nisa.tensor_copy(
            dst=out_tokens[:, col : col + _DOWN_CHUNK], src=sum_ps, engine=dve, name=nm(f"evict_sum_c{chunk}")
        )
        for half in range(n_out_halves):
            nisa.dma_copy(
                dst=output[output_rows, half * out_cols + col : half * out_cols + col + _DOWN_CHUNK],
                src=out_tokens[half * T : half * T + valid, col : col + _DOWN_CHUNK],  # [t, 512] of output half
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.sync,
                name=nm(f"store_out_h{half}_c{chunk}"),
            )
        return matmul

    def down_projection(
        group: int,
        chunks: Sequence[int],
        after: nisa.NkiInstruction,
        fresh: bool = False,
        final: Optional[bool] = None,
    ) -> nisa.NkiInstruction:
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
        slot = group % n_groups  # stationary and down-weight slot (dynamic loop groups reuse the head slots)
        scale_tile = (group // _SCALE_GROUPS) % n_scale_tiles
        out_ps = psum["out"]
        for chunk in chunks:
            col = chunk * _DOWN_CHUNK  # first column of the chunk in each output half
            # The halves with this chunk (uneven halves: the last half has fewer chunks); the first one opens the
            # accumulation and writes the rows it covers.
            halves = [half for half in reversed(range(n_out_halves)) if chunk < half_chunks[half]]
            opener = halves[0]
            evict_rows = slice(0, stationary_cols - (n_out_halves - 1 - opener) * stationary_block)  # rows written
            for half in halves:
                h_col = half * out_cols + col  # first H column of the chunk
                # pre_quant: (half + 1) * T rows per term; post_down: (half + 1) * T*K pair rows
                out_rows = stationary_cols - (n_out_halves - 1 - half) * stationary_block
                # Two scale groups: one matmul per group s (its stationary and its down scales) into the same rows.
                for scale_set in range(n_scale_sets):
                    set_first = slot * group_stride + scale_set * stationary_cols  # first column of the stationary
                    # pre_quant: from token block Hh - 1 - half; post_down: from block Hh - 1 - half of T*K pair slots
                    first_block = set_first + (n_out_halves - 1 - half) * stationary_block
                    # scale rows of group g: tile g / 4 (scale group s at column s * H of the tile), from partition
                    # 4 (g % 4)
                    scale_col = (scale_tile * n_scale_sets + scale_set) * H + h_col
                    previous = matmul
                    matmul = nisa.nc_matmul_mx(
                        dst=out_ps[:out_rows, col : col + _DOWN_CHUNK],  # [h*T + t, 512], then the lo rows (hi / lo)
                        # [(c, p), (term, j, t)]: from token block Hh - 1 - half (of the first term) to the end
                        stationary=inter[:, first_block : set_first + stationary_cols],
                        moving=down_w[:, slot * H + h_col : slot * H + h_col + _DOWN_CHUNK],  # [(c, p), 512]
                        stationary_scale=inter_s[:, first_block : set_first + stationary_cols],
                        moving_scale=down_s[_X4 * (group % _SCALE_GROUPS) :, scale_col : scale_col + _DOWN_CHUNK],
                        accumulate=(group > 0 and not fresh) or half < opener or scale_set > 0,
                        name=nm(f"mm_down_g{group}_h{half}_c{chunk}" + (f"_s{scale_set}" if n_scale_sets > 1 else "")),
                    )
                    if previous != None:
                        matmul.depends_on(previous, kind=nisa.dependency.schedule)
            if (group == n_groups - 1) if final == None else final:
                # The last chunk's eviction gates the final store, so it always goes to the faster DVE.
                position = chunk_order.index(chunk)
                # dynamic dedupe: DVE only, so ACT reaches the tails early (their ACT table prefetch hides)
                on_dve = position % 3 != 0 or position == len(chunk_order) - 1 or dynamic
                if post_down:
                    # Each pair row times its affinity (per-partition scale) on the way out of PSUM.
                    if on_dve:
                        nisa.tensor_scalar(
                            dst=out_sb[:, col : col + _DOWN_CHUNK],
                            data=out_ps[:, col : col + _DOWN_CHUNK],
                            op0=nl.multiply,
                            operand0=aff_rows[:stationary_cols],  # the Hh * T*K output rows
                            engine=dve,
                            name=nm(f"evict_c{chunk}"),
                        )
                    else:
                        nisa.activation(
                            dst=out_sb[:, col : col + _DOWN_CHUNK],
                            op=nl.copy,
                            data=out_ps[:, col : col + _DOWN_CHUNK],
                            scale=aff_rows[:stationary_cols],
                            name=nm(f"evict_c{chunk}"),
                        )
                    # The token sum of the previous chunk follows this chunk's matmuls, so its eviction is done.
                    if pending_sums:
                        matmul = sum_tokens(pending_sums.pop(0), after=matmul)
                    pending_sums.append(chunk)
                    if position == len(chunk_order) - 1:
                        matmul = sum_tokens(pending_sums.pop(0), after=matmul)
                    continue
                if evict["add"]:
                    # dynamic tail: out_sb (out_terms with the bf16 terms) = the head's partial + the tail's output
                    nisa.tensor_tensor(
                        dst=out_terms[evict_rows, col : col + _DOWN_CHUNK],
                        data1=out_ps[evict_rows, col : col + _DOWN_CHUNK],
                        data2=out_sb[evict_rows, col : col + _DOWN_CHUNK],
                        op=nl.add,
                        engine=dve,
                    )
                    if dyn_head_store:
                        store_chunk(chunk, col, out_terms)  # dyn_head_store: the tail stores the sum again
                    continue
                if terms_tile:
                    # The head's bf16 terms, the output if no tail runs.
                    nisa.tensor_copy(
                        dst=out_terms[evict_rows, col : col + _DOWN_CHUNK],
                        src=out_ps[evict_rows, col : col + _DOWN_CHUNK],
                        engine=dve,
                        name=nm(f"evict_terms_c{chunk}"),
                    )
                if on_dve:
                    nisa.tensor_copy(
                        dst=out_sb[evict_rows, col : col + _DOWN_CHUNK],
                        src=out_ps[evict_rows, col : col + _DOWN_CHUNK],
                        engine=dve,
                        name=nm(f"evict_c{chunk}"),
                    )
                else:
                    nisa.activation(
                        dst=out_sb[evict_rows, col : col + _DOWN_CHUNK],
                        op=nl.copy,
                        data=out_ps[evict_rows, col : col + _DOWN_CHUNK],
                        name=nm(f"evict_c{chunk}"),
                    )
                # dynamic dedupe: Step 7.6 stores after the tails (dyn_head_store: the head stores here, from the
                # bf16 terms, and a tail that runs stores again)
                if not dynamic or dyn_head_store:
                    store_chunk(chunk, col, out_terms if dynamic else out_sb)
        return matmul

    def tail_store() -> None:
        """dyn_head_store, in a tail after its last eviction: the whole output again (see tail_sum)."""
        sum_tiles = psum["sum"]  # [t, 512] fp32, alternating banks
        for half in range(n_out_halves):
            cols = min(half_chunks[half] * _DOWN_CHUNK, out_cols)
            for chunk in range(half_chunks[half]):
                c0 = chunk * _DOWN_CHUNK
                sum_ps = sum_tiles[chunk % 2]
                nisa.nc_matmul(
                    dst=sum_ps,  # [t, 512]: hi + lo of token t
                    stationary=tail_sum[half],
                    moving=out_terms[0:stationary_cols, c0 : c0 + _DOWN_CHUNK],
                    accumulate=False,
                )
                nisa.tensor_copy(
                    dst=tail_final[half * T : (half + 1) * T, c0 : c0 + _DOWN_CHUNK],
                    src=sum_ps,
                    engine=dve,
                )
            zero = nisa.scalar_load(store_zero[0:1, 0:1])
            nisa.dma_copy(
                dst=output.ap(
                    pattern=[[output.shape[1], valid], [1, cols]],
                    offset=output_row * output.shape[1] + half * out_cols,
                    scalar_offset=zero,
                    indirect_dim=0,
                ),
                src=tail_final[half * T : half * T + valid, 0:cols],
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.sync,
            )

    def store_chunk(chunk: int, col: int, src: nl.NkiTensor) -> None:
        """The output DMAs of one down chunk (both output halves) from the bf16 tile src (hi / lo: the DMA adds the
        hi and lo rows). In a dynamic loop (dyn_head_store, a tail), a static DMA cannot write the output, so the
        destination gets a register offset of 0 (a dynamic DMA)."""
        for half in range(n_out_halves):
            if chunk >= half_chunks[half]:
                continue  # uneven halves: the last half has no such chunk
            first_col = half * out_cols + col
            if in_loop[0]:
                continue  # a tail (dyn_head_store): tail_store() after its last eviction
            dst = output[output_rows, first_col : first_col + _DOWN_CHUNK]
            if hi_lo and store_parts > 1:
                # store_parts: one DMA per run of consecutive token rows (the DMA adds the hi and lo rows)
                edges = [valid * part // store_parts for part in range(store_parts + 1)]
                for part in range(store_parts):
                    rows = slice(edges[part], edges[part + 1])
                    if rows.start == rows.stop:
                        continue
                    nisa.dma_compute(
                        dst=output[
                            output_row + rows.start : output_row + rows.stop, first_col : first_col + _DOWN_CHUNK
                        ],
                        srcs=[
                            src[rows, col : col + _DOWN_CHUNK],  # hi [t, 512] of the part
                            src[token_cols + rows.start : token_cols + rows.stop, col : col + _DOWN_CHUNK],  # lo
                        ],
                        reduce_op=nl.add,
                        name=nm(f"store_out_h{half}_c{chunk}_p{part}"),
                    )
            elif hi_lo:
                # The DMA adds the hi and lo rows of the output half while it stores them.
                nisa.dma_compute(
                    dst=dst,
                    srcs=[
                        src[half * T : half * T + valid, col : col + _DOWN_CHUNK],  # hi [t, 512]
                        src[token_cols + half * T : token_cols + half * T + valid, col : col + _DOWN_CHUNK],  # lo
                    ],
                    reduce_op=nl.add,
                    name=nm(f"store_out_h{half}_c{chunk}"),
                )
            else:
                nisa.dma_copy(
                    dst=dst,
                    src=src[half * T : half * T + valid, col : col + _DOWN_CHUNK],  # [t, 512] of output half
                    dge_mode=nisa.dge_mode.hwdge,
                    engine=nisa.engine.sync,
                    name=nm(f"store_out_h{half}_c{chunk}"),
                )

    # Step 7: Run the PE software pipeline (PE order in the kernel docstring).
    # Even chunks first, then odd: two chunks share a PSUM bank, so evicting chunk 2j never blocks the write of 2j + 1.
    chunk_order = list(range(0, n_down_chunks, 2)) + list(range(1, n_down_chunks, 2))
    # Gate/up units (sub-group, H512 block) in PE order: all blocks of sub-group 0, then of sub-group 1 (U = 2).
    units = [(sub_group, block) for sub_group in range(n_sub_groups) for block in range(n_h512)]
    # Units of gate/up(g + 1) before finish(g): they cover SwiGLU(g). The fused long stream finishes sooner, so the
    # MXFP8 chain of finish(g) runs before the next Pool descriptor window blocks DVE (the standalone kernel is
    # faster with half of the units).
    finish_units = _LONG_FINISH_AFTER_UNITS if long_stream and fused_schedule else len(units) // 2
    first_half = units[:finish_units]
    second_half = units[finish_units:]
    # Units of gate/up(g + 1) before down(g). The long stream runs them all first (not measured for the standalone
    # kernel with post_down): SwiGLU(g + 1) then starts before down(g), and finish(g) has the whole gate/up(g + 1)
    # to cover its MXFP8 chain.
    if down_after_units == None:
        all_first = long_stream and (fused_schedule or not post_down)
        down_after = len(units) if all_first else len(units) * _DOWN_AFTER_SIXTEENTHS // 16
    else:
        down_after = down_after_units
    kernel_assert(
        finish_units <= down_after <= len(units),
        f"down_after_units must be in {finish_units} .. {len(units)}, got {down_after}",
    )
    # The second sub-group of group g + 1 is re-laid after SwiGLU(g): its rows may still be in flight, and finish(g)
    # needs SwiGLU(g) first (ACT and DVE run in order).
    later_runs = range(1, n_sub_groups)
    # Step 7.1: Re-lay groups 0 (in two parts, while the second half of its rows is in flight) and 1, then gate/up(0).
    if tail_chain:
        # tail_chain: the group-0 copies follow the tail of the call before on DVE and ACT
        relayout(0, n_parts=2, after_dve=tail_chain[-1][0], after_act=tail_chain[-1][1])
    else:
        relayout(0, n_parts=2)
    if hwdge_group != None and n_act_loads > 0:
        # Long stream (fused, or hwdge_act_loads): ACT loads the last experts of the HWDGE group after its group-0
        # copies (ACT order).
        for pair in range((hwdge_group + 1) * _GROUP - n_act_loads, (hwdge_group + 1) * _GROUP):
            load = load_expert(pair, nisa.engine.scalar, id_row=cidx0 if dynamic else None)
            load.depends_on(relayout_copies[act][-1], kind=nisa.dependency.schedule)
            relayout_copies[act].append(load)
    keep_alive = None
    if pe_keep_alive > 0:
        # PE keep-alive before gate/up(0): an ACT op after the first ACT scale copy of group 0 (same-engine order)
        # writes 0 into the ones tile, the matmuls read it (data dependency), and SiLU(0) adds 0 * their output.
        anchor = nisa.activation(
            dst=keep_alive_src[:, 0:1],
            op=nl.copy,
            data=ones[:, 0:1],  # any ready tile: the schedule dependency on ACT places the anchor
            scale=0.0,
            name="keep_alive_anchor",
        )
        anchor.depends_on(relayout_copies[act][0], kind=nisa.dependency.schedule)
        for index in range(pe_keep_alive):
            previous = keep_alive
            keep_alive = nisa.nc_matmul(
                dst=keep_alive_ps,
                stationary=keep_alive_src[:, 0:_P_MAX],
                moving=keep_alive_src,
                accumulate=index > 0,
                name=f"pe_keep_alive_{index}",
            )
            if previous != None:
                keep_alive.depends_on(previous, kind=nisa.dependency.schedule)
        sink = nisa.tensor_scalar(
            dst=keep_alive_sink,
            data=keep_alive_ps[0:T, 0:1],
            op0=nl.multiply,
            operand0=0.0,
            engine=dve,
            name="keep_alive_sink_op",
        )
        sink.depends_on(relayout_copies[dve][-1], kind=nisa.dependency.schedule)
    row_weights_done = None
    if direct_g1:
        """dyn_direct_g1: DVE forms the rest of the gather indices after the group-0 re-layout (Pool gathers their
        expert ids behind the group-1 gather), then the router affinities and the row weights, then its group-1
        copies."""
        anchor = relayout_copies[dve][-1]
        weight_gather_index(after_dve=anchor)
        gather_index_done = issue_rest(after_dve=anchor)
        if finish_affinities != None:
            finish_affinities(gather_index_done)
        row_weights_done = row_weight_math(after=relayout_copies[dve][-1])
    relayout(1, n_parts=n_sub_groups, runs=[0], after_dve=row_weights_done)
    if dynamic:
        if row_weights_done == None:
            row_weight_math()  # DVE work while the PE runs gate_up(0); finish(0) needs it
        """The tail trip counts become registers on every engine (TENSOR_LOAD), placed as soon as their tile is
        written. DVE writes the tile here, after the group-0/1 re-layout copies, so the loads do not sit in ACT's
        queue ahead of its group-0 expert loads."""
        trips_copy = nisa.tensor_copy(dst=dd_trips, src=dd_flags, engine=dve, name="dd_trips_op")
        trips_copy.depends_on(relayout_copies[dve][-1], kind=nisa.dependency.schedule)
    # The second phase ends: the first gate/up matmul reads the hidden.
    yield "relayout"
    last = gate_up_projection(0, units, after=keep_alive)
    silu, multiply = swiglu(0)
    if silu0_after_relayout == None:
        silu0_after_relayout = long_stream and fused_schedule and not post_down
    if silu0_after_relayout:
        # ACT copies its part of the group-1 re-layout before SiLU(0), which waits for gate/up(0).
        silu.depends_on(relayout_copies[act][-1], kind=nisa.dependency.schedule)
        if situ != None:
            situ_first_act[0].depends_on(relayout_copies[act][-1], kind=nisa.dependency.schedule)
    if later_runs:
        relayout(1, n_parts=n_sub_groups, runs=later_runs, after_dve=multiply, after_act=silu)
    if dynamic:
        # The trip counts of the tails (a register load on every engine), after the early work of Step 7.1.
        tail_trips = {
            tail_groups: nisa.scalar_load(dd_trips[_P_MAX - 1 : _P_MAX, tail_groups - 1 : tail_groups])
            for tail_groups in range(dyn_head_groups + 1, n_groups + 1)
        }
    # Step 7.2: Gate/up of group g + 1 with finish(g) and down(g) inside it.
    # Groups of the static schedule: all, or the head of the dynamic dedupe (the tails run the others, Step 7.5).
    n_run = dyn_head_groups if dynamic else n_groups
    for group in range(n_run - 1):
        nxt = group + 1
        if nxt + 1 < n_groups and not later_runs and not long_stream:
            relayout(nxt + 1, n_parts=1)
        if nxt == n_run - 1 and pe_keep_alive_tail > 0:
            # PE keep-alive before the last gate/up: the matmuls follow down(g) in PE order, and the SiLU of the last
            # group adds 0 * their output as its bias (the sink runs on DVE after SwiGLU(g)).
            for index in range(pe_keep_alive_tail):
                previous = last
                last = nisa.nc_matmul(
                    dst=tail_ps,
                    stationary=keep_alive_src[:, 0:_P_MAX],
                    moving=keep_alive_src,
                    accumulate=index > 0,
                    name=f"pe_keep_alive_tail_{index}",
                )
                last.depends_on(previous, kind=nisa.dependency.schedule)
            tail_sink_op = nisa.tensor_scalar(
                dst=tail_sink,
                data=tail_ps[0:T, 0:1],
                op0=nl.multiply,
                operand0=0.0,
                engine=dve,
                name="keep_alive_tail_sink_op",
            )
            tail_sink_op.depends_on(multiply, kind=nisa.dependency.schedule)
        last_iteration = last_options and nxt == n_run - 1  # the last-group options (gate/up of the last group)
        halves_done = None  # last_group_halves: the activation of the last group's second half
        finished_last = False  # last_finish_early: finish(last) is issued here (not in Step 7.3)
        if last_iteration and last_finish_early:
            """last_finish_early: gate/up(last), its activation, finish(second-to-last), down(late_down - 1),
            finish(last), down(late_down); Step 7.3 then runs down(second-to-last) and down(last)."""
            if last_gate_up_split:
                last = gate_up_projection(nxt, units, after=last, halves=(0,))
                last = gate_up_projection(nxt, units, after=last, halves=(1,))
            else:
                last = gate_up_projection(nxt, units, after=last)
            halves_done = swiglu(nxt)
            last, quantize = finish(group, after=last)
            last = down_projection(late_down - 1, chunk_order, after=last)
            last, final_quantize = finish(nxt, after=last)
            last = down_projection(late_down, chunk_order, after=last)
            finished_last = True
        elif last_iteration and last_group_halves:
            """last_group_halves: gate/up of the first half, finish(second-to-last), the first half's activation, the
            re-layout of the second half (after that activation on ACT and DVE), gate/up and activation of the second
            half."""
            if last_swiglu_first:
                last, quantize = finish(group, after=last)
            last = gate_up_projection(nxt, units, after=last, pairs=(0, half_pairs))
            if not last_swiglu_first:
                last, quantize = finish(group, after=last)
            act_h0, dve_h0 = situ_activation(nxt, pairs=(0, half_pairs))
            relayout(nxt, n_parts=2, runs=[1], after_dve=dve_h0, after_act=act_h0)
            last = gate_up_projection(nxt, units, after=last, pairs=(half_pairs, half_pairs))
            if last_gate_up_first:
                last = down_projection(late_down - 1, chunk_order, after=last)
                last = down_projection(late_down, chunk_order, after=last)
            halves_done = situ_activation(nxt, pairs=(half_pairs, half_pairs))
        elif not last_iteration:
            last = gate_up_projection(nxt, first_half, after=last)
            last, quantize = finish(group, after=last)
        elif last_swiglu_first:
            # finish(second-to-last) first: its activation ran before the last re-layout (iteration n - 3)
            last, quantize = finish(group, after=last)
            last = gate_up_projection(nxt, units, after=last, halves=(0,) if last_gate_up_split else (0, 1))
        else:
            split_units = units if last_gate_up_split else first_half  # last_gate_up_split: all gate matmuls first
            last = gate_up_projection(nxt, split_units, after=last, halves=(0,) if last_gate_up_split else (0, 1))
            last, quantize = finish(group, after=last)
        # dyn_late_relayout: the first tail group is re-laid after the head's last SiLU (below)
        late_relayout = dynamic and dyn_late_relayout and nxt + 1 == n_run
        # last_swiglu_first: the last group is re-laid after the activation of the second-to-last group (below)
        swiglu_first = last_swiglu_first and nxt + 1 == n_groups - 1
        if nxt + 1 < n_groups and long_stream and not late_relayout and not swiglu_first:
            # Long stream: DVE re-lays group g + 2 after finish(g) and ACT after SiLU(g), and SwiGLU(g + 1) follows
            # the re-layout (program order per engine), so neither the MXFP8 chain nor SiLU waits for late rows.
            if last_group_halves and nxt + 1 == n_groups - 1:
                # last_group_halves: the first half here, the second half after the first half's activation
                relayout(nxt + 1, n_parts=2, runs=[0], after_dve=quantize, after_act=silu)
            else:
                relayout(nxt + 1, n_parts=2 if nxt + 1 == hwdge_group else 1, after_dve=quantize, after_act=silu)
            if dynamic and dyn_tail_gate == nxt + 1 and dyn_tail_loads != "pool":
                """The SP tail loads read their expert ids from a copy that DVE writes after this re-layout (a data
                dependency), so their data follows the head's gathers in DMA order."""
                ids_copy = nisa.tensor_copy(
                    dst=dd_tail_ids[0:1, dyn_head_groups * _GROUP : n_pairs],
                    src=cidx[0:1, dyn_head_groups * _GROUP : n_pairs],
                    engine=dve,
                    name="dd_tail_ids_op",
                )
                ids_copy.depends_on(relayout_copies[dve][-1], kind=nisa.dependency.schedule)
                load_sp_tail(dd_tail_ids)
        if nxt + 1 < n_groups and later_runs:
            # U = 2: DVE re-lays group g + 2 after finish(g), because down(g) waits for finish(g) while the rows of
            # g + 2 may still be in flight.
            relayout(nxt + 1, n_parts=n_sub_groups, runs=[0], after_dve=quantize)
        if late_down > 0 and 1 <= group <= late_down:
            """late_down = c: down(g - 1) runs inside gate/up(g + 1), after its units, for g < c. At g = c both
            down(c - 1) and down(c) run there (unless g + 1 is the last group: Step 7.3 runs down(c)), then the
            default order."""
            last = gate_up_projection(nxt, units[finish_units:down_after], after=last)
            if paired_downs:
                # paired_downs: down(g - 1) and down(g) for odd g (one merged down gather), none for even g
                if group % 2 == 1:
                    last = down_projection(group - 1, chunk_order, after=last)
                    if group < late_down or nxt != n_run - 1:
                        last = down_projection(group, chunk_order, after=last)
            elif not (last_gate_up_first and group == late_down):  # last_gate_up_first: after the last gate/up
                last = down_projection(group - 1, chunk_order, after=last)
                if group == late_down and nxt != n_run - 1:
                    last = down_projection(group, chunk_order, after=last)
            last = gate_up_projection(nxt, units[down_after:], after=last)
        elif halves_done != None:
            pass  # last_group_halves: the PE work of the last group is issued above
        elif last_iteration:
            if last_gate_up_split:
                last = gate_up_projection(nxt, units, after=last, halves=(1,))  # the up matmuls (PSUM bank 6)
            elif not last_swiglu_first:
                last = gate_up_projection(nxt, second_half, after=last)
            if last_gate_up_first:
                # The down projections of groups late_down - 1 and late_down (their weights follow the last gate/up
                # rows in the Pool stream), while ACT and DVE run the last activation.
                last = down_projection(late_down - 1, chunk_order, after=last)
                last = down_projection(late_down, chunk_order, after=last)
        elif group == 0 or nxt == n_run - 1:
            last = gate_up_projection(nxt, second_half, after=last)
        else:
            last = gate_up_projection(nxt, units[finish_units:down_after], after=last)
            last = down_projection(group, chunk_order, after=last)
            last = gate_up_projection(nxt, units[down_after:], after=last)
        silu, multiply = swiglu(nxt) if halves_done == None else halves_done
        if swiglu_first:
            # last_swiglu_first: the last group's re-layout after this activation (ACT and DVE program order)
            relayout(
                nxt + 1,
                n_parts=2 if last_group_halves else 1,
                runs=[0] if last_group_halves else None,
                after_dve=multiply,
                after_act=silu,
            )
        elif nxt + 1 < n_groups and long_stream:
            # (act_relayout_blocks: an engine may have no re-layout copy, then there is nothing to follow)
            if relayout_copies[dve]:
                multiply.depends_on(relayout_copies[dve][-1], kind=nisa.dependency.schedule)
            if relayout_copies[act]:
                silu.depends_on(relayout_copies[act][-1], kind=nisa.dependency.schedule)
            if situ != None and relayout_copies[act]:
                situ_first_act[nxt].depends_on(relayout_copies[act][-1], kind=nisa.dependency.schedule)
            if situ != None and relayout_copies[dve]:
                situ_first_dve[nxt].depends_on(relayout_copies[dve][-1], kind=nisa.dependency.schedule)
        if nxt + 1 < n_groups and later_runs:
            relayout(nxt + 1, n_parts=n_sub_groups, runs=later_runs, after_dve=multiply, after_act=silu)
        if group == 0 and late_down == 0:
            # The group-0 down weights arrive after the gate/up rows of group 1.
            last = down_projection(group, chunk_order, after=last)
    if dynamic and dyn_late_relayout:
        relayout(n_run, n_parts=1, after_act=silu, all_act=True)  # ACT, after the head's last SiLU
    # Step 7.3: down(second last) with finish(last) before its last chunks, then down(last).
    # The SiLU * up of the last group runs while the PE does the first chunks; the last chunks cover its MXFP8 chain.
    second_last, final = n_run - 2, n_run - 1
    finish_after = max(2, n_down_chunks - _LAST_FINISH_CHUNKS_LEFT)
    if finished_last:
        last = down_projection(second_last, chunk_order, after=last)  # last_finish_early: finish(last) is done
    else:
        last = down_projection(second_last, chunk_order[:finish_after], after=last)
        last, final_quantize = finish(final, after=last)
        last = down_projection(second_last, chunk_order[finish_after:], after=last)
    last = down_projection(final, chunk_order, after=last, final=True)
    if tail_chain != None:
        # tail_chain: the next call's group-0 copies follow the last MXFP8 chain (DVE) and SwiGLU (ACT) of this call
        kernel_assert(not dynamic and not post_down, "tail_chain needs dedupe_experts=None and pre_quant")
        tail_chain.append((final_quantize, silu))

    if dynamic:
        """Step 7.5: the tails. Tail G (trip count 1 if G = ceil(U / 16), else 0) runs groups dyn_head_groups .. G - 1
        with their own pipeline. A PSUM tile cannot be live across a loop boundary, so the head evicts and stores
        its output (the final output for G <= dyn_head_groups), and a tail's store adds the head partial out_sb
        (at U = 128 each output row has one nonzero term, so the sum is exact). All weight loads are static."""
        head_last = last
        for tail_groups in range(dyn_head_groups + 1, n_groups + 1):
            tiles = dict(
                gate_up=[
                    nl.ndarray(
                        tile.shape,
                        dtype=tile.dtype,
                        buffer=nl.psum,
                        address=(0, parity * gate_up_banks * _PSUM_BANK_BYTES),
                        name=f"gate_up_ps{parity}_tail{tail_groups}",
                    )
                    for parity, tile in enumerate(gate_up_ps_by_parity)
                ],
                gated=nl.ndarray(
                    gated_ps.shape,
                    dtype=gated_ps.dtype,
                    buffer=nl.psum,
                    address=(0, 7 * _PSUM_BANK_BYTES),
                    name=f"gated_ps_tail{tail_groups}",
                ),
                out=nl.ndarray(
                    out_ps.shape,
                    dtype=out_ps.dtype,
                    buffer=nl.psum,
                    address=(0, out_bank * _PSUM_BANK_BYTES),
                    name=f"out_ps_tail{tail_groups}",
                ),
            )
            if dyn_head_store:
                # dyn_head_store: the hi + lo sums of the tail's store, double-buffered in banks 6 (the routing tiles
                # are done) and 7 (the tail's gated tile is done by then), so the PE and DVE overlap
                tiles["sum"] = [
                    nl.ndarray(
                        (T, _DOWN_CHUNK),
                        dtype=nl.float32,
                        buffer=nl.psum,
                        address=(0, bank * _PSUM_BANK_BYTES),
                        name=f"sum_ps{bank}_tail{tail_groups}",
                    )
                    for bank in (6, 7)
                ]

            def tail(_, tail_groups=tail_groups, tiles=tiles) -> None:
                """Groups dyn_head_groups .. tail_groups - 1, pipelined as in Step 7.2 / 7.3."""
                in_loop[0] = True
                psum.update(tiles)
                evict.update(add=True)
                first, final_group = dyn_head_groups, tail_groups - 1
                for group in range(first + 1, tail_groups):
                    relayout(group, n_parts=1)  # the head re-laid group `first` before the cut
                matmul = gate_up_projection(first, units, after=head_last)
                swiglu(first)
                for group in range(first, final_group):
                    nxt = group + 1
                    matmul = gate_up_projection(nxt, first_half, after=matmul)
                    matmul, _ = finish(group, after=matmul)
                    matmul = gate_up_projection(nxt, second_half, after=matmul)
                    swiglu(nxt)
                    if nxt < final_group:
                        matmul = down_projection(group, chunk_order, after=matmul, fresh=group == first)
                if final_group > first:
                    split = max(2, n_down_chunks - _LAST_FINISH_CHUNKS_LEFT)
                    fresh = final_group - 1 == first
                    matmul = down_projection(final_group - 1, chunk_order[:split], after=matmul, fresh=fresh)
                    matmul, _ = finish(final_group, after=matmul)
                    matmul = down_projection(final_group - 1, chunk_order[split:], after=matmul, fresh=fresh)
                    down_projection(final_group, chunk_order, after=matmul, final=True)
                else:
                    matmul, _ = finish(first, after=matmul)
                    down_projection(first, chunk_order, after=matmul, fresh=True, final=True)
                if dyn_head_store:
                    tail_store()
                in_loop[0] = False
                psum.update(gate_up=gate_up_ps_by_parity, gated=gated_ps, out=out_ps)
                evict.update(add=False)

            nl.fori_loop(0, tail_trips[tail_groups], tail)
        # Step 7.6: store the output out_sb (the head, plus the tail that ran); the DMA adds hi + lo.
        # (dyn_head_store: the head and the tail stored it)
        if dyn_merged_stores and not dyn_head_store:
            """In eviction order (even chunks, then odd), chunks 0 .. 2j + 2 are evicted once odd chunk 2j + 1 is, so
            a store covers the contiguous chunks that are complete then (0-2, 3-4, 5-6 at H = 3584). The last even
            chunk goes first and alone, so the last store (the last chunk) is small."""
            kernel_assert(n_out_halves == 1 and hi_lo, "dyn_merged_stores needs one output half and hi / lo")
            store_runs, stored = [], 0  # (first, last) chunk of each store, in issue order
            for odd in range(1, n_down_chunks, 2):
                upto = min(odd + 2, n_down_chunks)  # chunks stored .. odd + 1 are evicted
                store_runs.append((stored, upto - 1))
                stored = upto
            first, last = store_runs[-1]
            if last > first and last % 2 == 0:
                store_runs = [(last, last)] + store_runs[:-1] + [(first, last - 1)]
            for first, last in store_runs:
                cols = slice(first * _DOWN_CHUNK, (last + 1) * _DOWN_CHUNK)
                nisa.dma_compute(
                    dst=output[output_rows, cols],
                    srcs=[out_terms[0:valid, cols], out_terms[token_cols : token_cols + valid, cols]],  # hi, then lo
                    reduce_op=nl.add,
                    name=f"store_out_c{first}_{last}",
                )
        for chunk in chunk_order if not (dyn_merged_stores or dyn_head_store) else []:
            col = chunk * _DOWN_CHUNK
            for half in range(n_out_halves):
                if chunk >= half_chunks[half]:
                    continue  # uneven halves: the last half has no such chunk
                rows = [slice(half * T, half * T + valid)]  # hi rows [t, 512] of the half (then the lo rows)
                if hi_lo:
                    rows.append(slice(token_cols + half * T, token_cols + half * T + valid))
                nisa.dma_compute(
                    dst=output[output_rows, half * out_cols + col : half * out_cols + col + _DOWN_CHUNK],
                    srcs=[out_terms[r, col : col + _DOWN_CHUNK] for r in rows],
                    reduce_op=nl.add,
                    name=f"store_out_h{half}_c{chunk}",
                )

    # Step 7.4: Pool runs the stream in order, so nothing delays the first gathers.
    for prev, nxt in zip(stream, stream[1:]):
        nxt.depends_on(prev, kind=nisa.dependency.schedule)
    if stream_chain != None:
        # stream_chain: this call's stream follows the streams of the earlier calls
        if after_stream != None:
            stream[0].depends_on(after_stream, kind=nisa.dependency.schedule)
        stream_chain.append(stream[-1])

    sbm.close_scope()
    return output


def quantize_latent_mx(
    latent: nl.NkiTensor,
    alloc: Callable[..., nl.NkiTensor],
    latent_ps: nl.NkiTensor,
    hidden_qtz: nl.NkiTensor,
    hidden_scale: nl.NkiTensor,
    engines: Sequence[nisa.engine],
    prefix: str = "",
    form: str = "transpose",
    latent_tile: Optional[nl.NkiTensor] = None,
    name_suffix: Optional[str] = None,
) -> Tuple[nisa.NkiInstruction, nisa.NkiInstruction]:
    """Load a bf16 latent [T, H] (HBM) and quantize it to the MXFP8 gate/up stationary of the body. No norm.

    Layouts (q = x4 lane 0..3, p = partition, hb = H512 block):
        latent_rep  bf16 [128, C + 256] SBUF, C = B*512 columns of B = ceil((H/512) / P) H512 blocks per part,
                    P = min(2, 128/4T) parts: part c of the latent (columns c*C .. c*C + C - 1) on partitions c*4T .. c*4T + 4T - 1, copy
                    q on the T partitions c*4T + q*T .., shifted by q columns, so latent_rep[c*4T + q*T + t, j] =
                    latent[t, c*C + j + q]: 2 parts at T = 8 (B = 4), 4 HWDGE loads per part (one per copy) on the
                    `engines` queues in turn; 512 B of padding at the end of each row.
        identity    bf16 [128, 4T] SBUF: identity[c*4T + k, n] = 1 if k == n (the identity of each part's partitions).
        latent_ps   bf16 [128, (H/512) * 4 * T] PSUM: [p, (hb, q, t)] = latent[t, 512*hb + 4*p + q]. One matmul per
                    H512 block hb of part c: the stationary is the stride-4 view latent_rep[c*4T + (q, t), 512*(hb %
                    B) + 4*p] (4T rows x 128 columns) and the moving is the identity of the part, so out[p, (q, t)]
                    has one nonzero term (exact).
        swizzled    bf16 [128, (H/512) * T * 4] SBUF: [p, (hb, t, q)], q innermost (the lanes of one x4 element).
        hidden_qtz / hidden_scale [128, (H/512) * T]: quantize_mx of swizzled, element [p, hb*T + t] and the MX
                    scales in the quadrant layout. Each MX group (partitions 8r .. 8r + 7 x 4 lanes) holds the latent
                    columns 512*hb + 32*r .. + 31 of token t, so the gate/up weights must use the hidden order
                    h = 512*hb + 4*p + q.

    Args:
        latent (nl.NkiTensor): bfloat16 [T, H] @HBM, 4*T <= 128 a power of two. "transpose" also takes an SBUF latent
            with token t on partition t (no load: the transposes read it in place).
        alloc (Callable): SBUF allocator for the latent_rep, identity and swizzled tiles.
        latent_ps (nl.NkiTensor): bfloat16 [128, (H/512) * 4 * T] @PSUM scratch (448 B per partition at H = 3584,
            T = 8).
        hidden_qtz (nl.NkiTensor): float8_e4m3fn_x4 [128, (H/512) * T] @SBUF output.
        hidden_scale (nl.NkiTensor): uint8 [128, (H/512) * T] @SBUF output.
        engines (Sequence[nisa.engine]): the HWDGE queues of the latent loads, used in turn.
        prefix (str): prefix of the instruction and tensor names.
        form (str): "transpose" (default): one copy of the latent [T, H] (one load on engines[0]) and 4 PE
            transposes per H512 block, one per lane q, of the stride-4 column view latent[t, 512*hb + 4*p + q] (the
            same latent_ps values). "matmul": the layouts above (4x fewer PE instructions, 4 loads per part).
        latent_tile (Optional[nl.NkiTensor]): "transpose" only: a bfloat16 SBUF tile with at least T rows and H
            columns for the copy (a caller can share it with a tile that it writes after the quantization). None:
            a new tile.
        name_suffix (Optional[str]): appended to every op and tensor name (see selective_load_packed_phases).

    Returns:
        (the latent loads in issue order, the quantize_mx).
    """
    nisa, nl = _named_modules(name_suffix)
    T, H = latent.shape
    n_h512 = H // 512
    rows = _X4 * T  # (q, t) partitions of the shifted copies
    kernel_assert(
        rows <= _P_MAX and rows & (rows - 1) == 0, f"quantize_latent_mx needs 4*T <= 128, a power of two, got T={T}"
    )
    kernel_assert(form in ("matmul", "transpose"), f"latent_quant must be 'matmul' or 'transpose', got {form}")
    swizzled = alloc((_P_MAX, n_h512 * T * _X4), nl.bfloat16, name=f"{prefix}latent_swizzled")
    if form == "transpose":
        # One copy [t, H] (latent_tile if given), then 4 PE transposes per H512 block (one per lane q) into
        # latent_ps[p, (hb, q, t)]. The PE transposes read their data from partition 0 (as the identity).
        if latent.buffer == nl.sbuf:
            latent_sb, loads = latent, []  # the transposes read the SBUF latent in place
        else:
            latent_sb = alloc((T, H), nl.bfloat16, name=f"{prefix}latent_sb") if latent_tile == None else latent_tile
            loads = [
                nisa.dma_copy(
                    dst=latent_sb[0:T, 0:H],
                    src=latent,
                    dge_mode=nisa.dge_mode.hwdge,
                    engine=engines[0],
                    name=f"{prefix}load_latent",
                )
            ]
        for block in range(n_h512):
            for lane in range(_X4):
                psum_col = (block * _X4 + lane) * T
                nisa.nc_transpose(
                    dst=latent_ps[:, psum_col : psum_col + T],  # [p, t]: lane `lane` of block `block`
                    # [t, p]: latent column 512*block + 4*p + lane of each token
                    data=latent_sb.ap(pattern=[[latent_sb.shape[1], T], [_X4, _P_MAX]], offset=block * 512 + lane),
                    name=f"{prefix}transpose_latent_b{block}_q{lane}",
                )
    else:
        """4 shifted copies of each part of the latent (parts of blocks_per_part H512 blocks, one 4T-partition block per
        part), the identity of each 4T-partition block, and one identity matmul per H512 block into latent_ps[p, (hb,
        q, t)]."""
        n_parts = min(_LATENT_PARTS, _P_MAX // rows)  # partition blocks of 4T rows
        blocks_per_part = div_ceil(n_h512, n_parts)
        part_cols = blocks_per_part * 512  # latent columns of one part
        """The identity tiles come before latent_rep, and latent_rep has 512 B of padding at its end: a compute write
        next to a DMA destination waits for that DMA."""
        identity_offsets = alloc((_P_MAX, rows), nl.int32, name=f"{prefix}latent_identity_offsets")
        identity = alloc((_P_MAX, rows), nl.bfloat16, name=f"{prefix}latent_identity")
        rep_cols = part_cols + 256  # 512 B of padding
        latent_rep = alloc((n_parts * rows, rep_cols), nl.bfloat16, name=f"{prefix}latent_rep", align=512)
        loads = []
        for part in range(div_ceil(n_h512, blocks_per_part)):
            first_col = part * part_cols
            width = min(part_cols, H - first_col)
            for lane in range(_X4):
                row = part * rows + lane * T  # first partition of copy (part, lane)
                load = nisa.dma_copy(
                    dst=latent_rep[row : row + T, 0 : width - lane],  # [t, j]: copy (part, lane)
                    src=latent[:, first_col + lane : first_col + width],  # [t, first_col + j + lane]
                    dge_mode=nisa.dge_mode.hwdge,
                    engine=engines[len(loads) % len(engines)],
                    name=f"{prefix}load_latent_p{part}_q{lane}",
                )
                loads.append(load)
        # identity[part*4T + k, n] = 1 where n == k: iota gives n - a, which is 0 mod 4T only for n == k (4T a power
        # of two).
        nisa.iota(
            dst=identity_offsets,
            pattern=[[1, rows]],
            offset=0,
            channel_multiplier=-1,
            name=f"{prefix}iota_latent_identity",
        )
        nisa.tensor_scalar(
            dst=identity_offsets,
            data=identity_offsets,
            op0=nl.bitwise_and,
            operand0=rows - 1,
            engine=nisa.vector_engine,
            name=f"{prefix}latent_identity_mod",
        )
        nisa.tensor_scalar(
            dst=identity,
            data=identity_offsets,
            op0=nl.equal,
            operand0=0,
            engine=nisa.vector_engine,
            name=f"{prefix}latent_identity_op",
        )
        for block in range(n_h512):
            part, local = block // blocks_per_part, block % blocks_per_part
            nisa.nc_matmul(
                dst=latent_ps[:, block * rows : (block + 1) * rows],  # [p, (q, t)] of block `block`
                # [(q, t), p]: latent column 512*block + 4*p + q of token t (copy (part, q) is shifted by q columns)
                stationary=latent_rep.ap(
                    pattern=[[rep_cols, rows], [_X4, _P_MAX]], offset=part * rows * rep_cols + local * 512
                ),
                moving=identity[part * rows : (part + 1) * rows, :],  # the identity of the part's partitions
                name=f"{prefix}transpose_latent_b{block}",
            )
    nisa.tensor_copy(
        # [p, hb, t, q]: the values to quantize, q innermost
        dst=swizzled.ap(pattern=[[n_h512 * T * _X4, _P_MAX], [T * _X4, n_h512], [_X4, T], [1, _X4]], offset=0),
        # [p, hb, t, q]: latent_ps[p, (hb, q, t)]
        src=latent_ps.ap(pattern=[[n_h512 * _X4 * T, _P_MAX], [_X4 * T, n_h512], [1, T], [T, _X4]], offset=0),
        engine=nisa.vector_engine,
        name=f"{prefix}swizzle_latent",
    )
    quantize = nisa.quantize_mx(dst=hidden_qtz, src=swizzled, dst_scale=hidden_scale, name=f"{prefix}quantize_latent")
    return loads, quantize
